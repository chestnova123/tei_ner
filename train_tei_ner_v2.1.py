import argparse
from pathlib import Path

import numpy as np
from datasets import load_from_disk
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    EarlyStoppingCallback,
    AutoConfig,
)
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from seqeval.scheme import IOB2
import inspect
import torch
from torch.nn import CrossEntropyLoss
from collections import Counter
import math
import torch.nn.functional as F
import re
from transformers import set_seed
import csv
from datetime import datetime
import os

from torch import nn
from torchcrf import CRF
from transformers import AutoModel



# =====================================================================
# 1. Label definitions (must match build_tei_ner_corpus.py exactly)
# =====================================================================

TYPES = [
    "ARTEFACT",
    "CONCEPT",
    "PERSON",
    "PLACE",
    "ORG",
    "CULTURE",
    "BIBL",
]

LABELS = ["O"]
for t in TYPES:
    LABELS.extend([f"B-{t}", f"I-{t}"])

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

# =====================================================================
# 1a. Label entity counts for weighting
# =====================================================================
# Entity counts from split analysis

TRAIN_ENTITY_COUNTS = {
    "ARTEFACT": 14961,
    "CONCEPT":  15255,
    "PERSON":    2833,
    "PLACE":     2862,
    "BIBL":      2021,
    "CULTURE":   1195,
    "ORG":        680,
}



# =====================================================================
# 2. Metrics: seqeval for entity-level F1
# =====================================================================


def align_predictions(predictions, label_ids):
    """
    Supports:
      - predictions as logits [B, T, C]  (argmax)
      - predictions as decoded ids [B, T] (already final tags)
    """
    if predictions.ndim == 3:
        preds = np.argmax(predictions, axis=2)
    elif predictions.ndim == 2:
        preds = predictions
    else:
        raise ValueError(f"Unexpected predictions shape: {predictions.shape}")

    batch_size, seq_len = preds.shape
    out_label_list = []
    out_pred_list = []

    for i in range(batch_size):
        example_labels = []
        example_preds = []
        for j in range(seq_len):
            if label_ids[i, j] == -100:
                continue
            example_labels.append(id2label[label_ids[i, j]])
            example_preds.append(id2label[int(preds[i, j])])
        out_label_list.append(example_labels)
        out_pred_list.append(example_preds)

    return out_pred_list, out_label_list

def per_type_scores(labels_list, preds_list):
    """
    Returns dict with per-type precision/recall/f1 from seqeval classification_report.
    Uses IOB2 strict mode.
    """
    rep = classification_report(
        labels_list,
        preds_list,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )

    out = {}
    for t in TYPES:
        if t in rep:
            out[f"precision_{t}"] = rep[t]["precision"]
            out[f"recall_{t}"] = rep[t]["recall"]
            out[f"f1_{t}"] = rep[t]["f1-score"]
    return out

def append_metrics_to_csv(
    output_dir: str,
    split: str,
    metrics: dict,
    run_info: dict,
):
    """
    Append one row to output_dir/metrics.csv.
    Automatically creates the file with a header if it doesn't exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics.csv")

    # Flatten: keep metrics as columns + run_info as columns
    row = {}
    row.update(run_info)
    row["split"] = split

    # Normalize metric keys (Trainer returns keys like eval_loss, eval_f1, etc.)
    # We'll store them exactly as received.
    for k, v in metrics.items():
        # Convert numpy types to plain python for CSV writing
        if hasattr(v, "item"):
            v = v.item()
        row[k] = v

    # Stable column order:
    # - run info first
    # - then core metrics
    # - then per-type metrics (sorted)
    core = ["eval_loss", "precision", "recall", "f1"]
    per_type = sorted([k for k in row.keys() if k.startswith(("precision_", "recall_", "f1_"))])
    other_metrics = sorted([k for k in row.keys() if k not in set(run_info.keys()) | {"split"} | set(core) | set(per_type)])

    fieldnames = list(run_info.keys()) + ["split"] + core + other_metrics + per_type

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if hasattr(predictions, "cpu"):
        predictions = predictions.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()
    preds_list, labels_list = align_predictions(predictions, labels)

    rep = classification_report(
        labels_list,
        preds_list,
        mode="strict",
        scheme=IOB2,
        output_dict=True,
        zero_division=0,
    )

    out = {
        "precision": rep["weighted avg"]["precision"],
        "recall":    rep["weighted avg"]["recall"],
        "f1":        rep["weighted avg"]["f1-score"],
        "f1_macro":  rep["macro avg"]["f1-score"],   # primary model selection metric
    }
    out.update(per_type_scores(labels_list, preds_list))
    return out


def print_predicted_tag_counts(trainer, dataset, id2label, topk=30):
    
    pred = trainer.predict(dataset)
    preds = pred.predictions     # could be [N,T,C] (logits) OR [N,T] (decoded ids)
    gold  = pred.label_ids       # [N,T]

    preds = np.asarray(preds)
    gold  = np.asarray(gold)

    # Convert logits -> ids if needed
    if preds.ndim == 3:
        pred_ids = np.argmax(preds, axis=-1)
    elif preds.ndim == 2:
        pred_ids = preds
    else:
        raise ValueError(f"Unexpected predictions shape: {preds.shape}")

    if gold.ndim != 2:
        raise ValueError(f"Unexpected label_ids shape: {gold.shape}")

    pred_counts = Counter()
    gold_counts = Counter()
    bibl_pred = Counter()
    bibl_gold = Counter()

    n_tokens = 0
    N, T = pred_ids.shape
    for i in range(N):
        for j in range(T):
            if gold[i, j] == -100:
                continue
            n_tokens += 1
            pl = id2label[int(pred_ids[i, j])]
            gl = id2label[int(gold[i, j])]
            pred_counts[pl] += 1
            gold_counts[gl] += 1
            if pl.endswith("-BIBL"):
                bibl_pred[pl] += 1
            if gl.endswith("-BIBL"):
                bibl_gold[gl] += 1

    print(f"Counted {n_tokens} tokens (excluding -100).")

    print("\nTop GOLD tags:")
    for lab, c in gold_counts.most_common(topk):
        print(f"{lab:10s} {c}")

    print("\nTop PRED tags:")
    for lab, c in pred_counts.most_common(topk):
        print(f"{lab:10s} {c}")

    print("\nBIBL GOLD breakdown:", dict(bibl_gold), "TOTAL:", sum(bibl_gold.values()))
    print("BIBL PRED breakdown:", dict(bibl_pred), "TOTAL:", sum(bibl_pred.values()))


# =============================
# 2.1 Weighted labels
# ==============================

def make_entity_count_weights(
    label2id,
    entity_counts,
    alpha=0.3,
    o_weight=0.0,
    min_w=0.8,
    max_w=4.0,
):
    """
    Weight by inverse entity frequency rather than token frequency.
    This correctly upweights short-but-rare types like BIBL.
    """
    import numpy as np

    counts = np.array(list(entity_counts.values()), dtype=np.float64)
    median_c = float(np.median(counts))

    type_weight = {
        t: (median_c / c) ** alpha
        for t, c in entity_counts.items()
    }

    print("\n=== Entity-count-based type weights (before clamping) ===")
    for t, w in sorted(type_weight.items(), key=lambda x: -x[1]):
        print(f"  {t:12s}: {w:.4f}  (entities={entity_counts[t]})")

    weights = torch.ones(len(label2id), dtype=torch.float)
    for lab, lab_id in label2id.items():
        if lab == "O":
            continue
        t = tag_to_type(lab)
        if t and t in type_weight:
            weights[lab_id] = float(type_weight[t])

    weights = torch.clamp(weights, min=min_w, max=max_w)
    weights[label2id["O"]] = o_weight
    weights = weights / weights.mean()
    return weights

def tag_to_type(label: str):
    """
    Map a label like 'B-PERSON' or 'I-ORG' to its TYPE (e.g. 'PERSON', 'ORG').
    Returns None for 'O' or anything unexpected.
    IOB2 only: valid prefixes are B and I.
    """
    if label == "O":
        return None
    m = re.match(r"^[BI]-(.+)$", label)
    return m.group(1) if m else None


class WeightedTokenTrainer(Trainer):
    def __init__(
        self,
        class_weights=None,
        label_smoothing=0.1,
        llrd=False,
        layer_decay=0.9,
        head_lr_mult=4.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # torch tensor [num_labels]
        self.label_smoothing = float(label_smoothing)

        # LLRD controls
        self.llrd = bool(llrd)
        self.layer_decay = float(layer_decay)
        self.head_lr_mult = float(head_lr_mult)

        self._printed_weights_debug = False

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Use default behavior for non-CRF
        if not hasattr(model, "crf"):
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        model.eval()
        with torch.no_grad():
            labels = inputs.get("labels")
            outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
            )
            logits = outputs["logits"]  # [B, T, C]

            # Mask = labeled positions if labels exist, else attention mask
            attn = inputs.get("attention_mask")

            # Start from attention mask (real tokens/padding)
            mask = attn.bool() if attn is not None else torch.ones_like(inputs["input_ids"], dtype=torch.bool)

            # Turn off positions you intended to ignore (-100), if labels exist
            if labels is not None:
                ignore = (labels == -100)
                mask = mask & (~ignore)

            # torchcrf requirement: first timestep must be on
            mask[:, 0] = True

            # Safety: if a sequence somehow has no valid positions, keep timestep 0 on
            empty = mask.sum(dim=1) == 0
            if empty.any():
                mask[empty, 0] = True

            decoded = model.decode(logits, mask=mask)  # list of lists

            # Pad decoded sequences back to [B, T] with 0s (won't be read where label=-100)
            B, T = logits.shape[:2]
            pred_ids = torch.zeros((B, T), dtype=torch.long, device=logits.device)
            for i, seq in enumerate(decoded):
                L = min(len(seq), T)
                pred_ids[i, :L] = torch.tensor(seq[:L], device=logits.device)

            loss = None
            if labels is not None:
                # compute loss for reporting
                safe_inputs = dict(inputs)
                loss = model(**safe_inputs)["loss"].detach()

        # return (loss, predictions, labels) where predictions are tensors
        if prediction_loss_only:
            return (loss, None, None)

        pred_out = pred_ids.detach().cpu()
        lab_out = labels.detach().cpu() if labels is not None else None
        return (loss, pred_out, lab_out)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ---- CRF path ----
        if hasattr(model, "crf"):
            outputs = model(**inputs)
            loss = outputs["loss"]
            return (loss, outputs) if return_outputs else loss
        
        labels = inputs.get("labels")  # [B, T]
        outputs = model(**inputs)
        logits = outputs.get("logits")  # [B, T, C]

        # Flatten
        B, T, C = logits.shape
        logits = logits.view(-1, C)  # [B*T, C]
        labels = labels.view(-1)     # [B*T]

        # Mask out ignored labels (-100)
        mask = labels != -100
        logits = logits[mask]
        labels = labels[mask]

        # If everything is masked (edge case)
        if labels.numel() == 0:
            loss = logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        # Log-probs for KL / smoothed CE
        log_probs = F.log_softmax(logits, dim=-1)  # [N, C]

        # Class weights
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        eps = self.label_smoothing

        if eps > 0.0:
            nll = F.nll_loss(log_probs, labels, reduction="none")  # [N]
            smooth = -log_probs.mean(dim=-1)                      # [N]
            loss_per_token = (1.0 - eps) * nll + eps * smooth

            # Apply class weights to tokens by gold label
            if weight is not None:
                loss_per_token = loss_per_token * weight[labels]

            loss = loss_per_token.mean()
        else:
            loss = F.cross_entropy(
                logits,
                labels,
                weight=weight,
                reduction="mean",
            )

        if (self.class_weights is not None) and (not self._printed_weights_debug):
            print("DEBUG: class weights enabled")
            self._printed_weights_debug = True

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        # If Trainer already made one, reuse it
        if self.optimizer is not None:
            return self.optimizer

        if self.llrd:
            param_groups = build_llrd_param_groups(
                model=self.model,
                base_lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                layer_decay=self.layer_decay,
                head_lr_mult=self.head_lr_mult,
            )
            self.optimizer = torch.optim.AdamW(
                param_groups,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        else:
            self.optimizer = super().create_optimizer()

        return self.optimizer
        
def print_weights(class_weights, id2label, topk=999):
    rows = [(id2label[i], float(w)) for i, w in enumerate(class_weights)]
    rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
    print("\nClass weights (sorted high→low):")
    for lab, w in rows_sorted[:topk]:
        print(f"{lab:<12s} {w:8.4f}")

# layer learning rate decay

def build_llrd_param_groups(
    model,
    base_lr: float,
    weight_decay: float,
    layer_decay: float = 0.9,
    head_lr_mult: float = 4.0,
):
    """
    Build AdamW param groups with layer-wise LR decay (LLRD).
    - base_lr: LR for the *top* transformer layer (not the head)
    - layer_decay: multiply LR by this as you go down layers (0.8–0.95 typical)
    - head_lr_mult: classifier head LR = base_lr * head_lr_mult

    Works for BERT/RoBERTa-like models with .encoder.layer
    """
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")

    
    # Correct: base_model_prefix is a STRING like "bert"
    base_prefix = getattr(model, "base_model_prefix", None)
    if not isinstance(base_prefix, str) or not base_prefix:
        raise ValueError(f"Unexpected base_model_prefix: {base_prefix!r}")

    # Correct: model.bert (or model.roberta, etc.) is the base transformer
    base_model = getattr(model, base_prefix, None)
    if base_model is None:
        raise ValueError(f"Model has base_model_prefix={base_prefix!r} but no attribute model.{base_prefix}")

    # BERT encoder layers live here
    if not hasattr(base_model, "encoder") or not hasattr(base_model.encoder, "layer"):
        raise ValueError("Expected base_model.encoder.layer (BERT/RoBERTa style).")

    layers = base_model.encoder.layer
    n_layers = len(layers)

    # We'll create param groups:
    # - embeddings (treated as layer 0)
    # - each encoder layer i
    # - head/classifier
    param_groups = []

    def add_group(params, lr, wd):
        if params:
            param_groups.append({"params": params, "lr": lr, "weight_decay": wd})

    # 1) Embeddings group (lowest LR)
    emb_lr = base_lr * (layer_decay ** n_layers)
    emb_decay, emb_nodecay = [], []
    for n, p in base_model.embeddings.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            emb_nodecay.append(p)
        else:
            emb_decay.append(p)
    add_group(emb_decay, emb_lr, weight_decay)
    add_group(emb_nodecay, emb_lr, 0.0)

    # 2) Encoder layers: lower layers smaller LR, top layer ~ base_lr
    for layer_idx in range(n_layers):
        layer_lr = base_lr * (layer_decay ** (n_layers - 1 - layer_idx))
        layer_decay_params, layer_nodecay_params = [], []
        for n, p in layers[layer_idx].named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in n for nd in no_decay):
                layer_nodecay_params.append(p)
            else:
                layer_decay_params.append(p)
        add_group(layer_decay_params, layer_lr, weight_decay)
        add_group(layer_nodecay_params, layer_lr, 0.0)

    # 3) Head / classifier (highest LR)
    head_lr = base_lr * head_lr_mult

    # For token classification, common head module names include "classifier"
    # But to be safe, collect params not in base_model
    base_param_ids = {id(p) for p in base_model.parameters()}
    head_decay, head_nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in base_param_ids:
            continue
        if any(nd in n for nd in no_decay):
            head_nodecay.append(p)
        else:
            head_decay.append(p)

    add_group(head_decay, head_lr, weight_decay)
    add_group(head_nodecay, head_lr, 0.0)

    return param_groups

#CRF

class TokenClassifierWithCRF(nn.Module):
    """
    Transformer encoder + linear emissions + CRF.
    Uses the same label2id/id2label mapping as your script (IOB2 + O).
    """

    def __init__(self, model_name: str, config: AutoConfig):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.backbone = AutoModel.from_pretrained(model_name, config=config)
        self.base_model_prefix = "backbone"
        
        dropout_prob = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # batch_first=True => emissions shape [B, T, C]
        self.crf = CRF(self.num_labels, batch_first=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        seq = outputs.last_hidden_state  # [B, T, H]
        seq = self.dropout(seq)
        emissions = self.classifier(seq)  # [B, T, C]

        out = {"logits": emissions}

        if labels is not None:
            # labels: [B, T], with -100 for special/padded tokens
            # CRF needs valid label ids everywhere but a mask to ignore positions.
            mask = attention_mask.bool() if attention_mask is not None else (labels != -100)
            safe_labels = labels.clone()
            ignore = safe_labels == -100
            safe_labels[ignore] = 0
            mask = mask & (~ignore)
            mask[:, 0] = True
            
            empty = mask.sum(dim=1) == 0
            if empty.any():
                mask[empty, 0] = True

            # CRF returns log-likelihood; we minimize negative log-likelihood
            nll = -self.crf(emissions, safe_labels, mask=mask, reduction="mean")
            out["loss"] = nll

        return out

    @torch.no_grad()
    def decode(self, logits, mask):
        """
        logits: [B, T, C]
        mask:   [B, T] bool
        returns: List[List[int]] decoded label ids (variable lengths)
        """
        mask = mask.clone()
        mask[:, 0] = True
        empty = mask.sum(dim=1) == 0
        if empty.any():
            mask[empty, 0] = True
        return self.crf.decode(logits, mask=mask)

# =====================================================================
# 3. Main training function
# =====================================================================


def main(
    dataset_path: str,
    model_name: str,
    output_dir: str,
    num_train_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    resume_from_checkpoint: str | None = None,
    use_crf: bool=False,
):
    dataset_path = str(Path(dataset_path).expanduser().resolve())
    output_dir = str(Path(output_dir).expanduser().resolve())

    print(f"Using random seed: {seed}")
    set_seed(seed)
    
    print(f"Loading dataset from: {dataset_path}")
    ds_dict = load_from_disk(dataset_path)

    train_dataset = ds_dict["train"]
    eval_dataset = ds_dict["validation"]
    test_dataset = ds_dict.get("test", None)
    
    def add_length_column(ds, input_key="input_ids"):
        return ds.map(lambda x: {"length": len(x[input_key])})

    train_dataset = add_length_column(train_dataset)
    eval_dataset  = add_length_column(eval_dataset)
    if test_dataset is not None:
        test_dataset = add_length_column(test_dataset)


    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    if test_dataset is not None:
        print(f"Test examples: {len(test_dataset)}")

    # -----------------------------
    # Compute class weights
    # -----------------------------
    
    entity_counts = TRAIN_ENTITY_COUNTS
    
    class_weights = make_entity_count_weights(
        label2id=label2id,
        entity_counts = entity_counts,
        alpha=0.3,          
        o_weight=0,
        min_w=0.8,
        max_w=4.0,
    )

    print_weights(class_weights, id2label)

    print("O weight:", float(class_weights[label2id["O"]]))
    print("Mean entity weight:", float(class_weights[1:].mean()))
    print("Max weight:", float(class_weights.max()), "Min weight:", float(class_weights.min()))

    print(f"Loading tokenizer & model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    if use_crf:
        model = TokenClassifierWithCRF(model_name=model_name, config=config)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,

        eval_strategy="steps",
        eval_steps=400,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=6,           # must cover patience window to keep best checkpoint

        logging_strategy="steps",
        logging_steps=50,

        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,

        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,

        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",  # macro F1: all types weighted equally
        greater_is_better=True,

        bf16=True,
        disable_tqdm=False,
        report_to="none",

        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        group_by_length=True,
        length_column_name="length",
        max_grad_norm=1.0,
        seed=seed,
        data_seed=seed,
    )

    trainer = WeightedTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=0.1,
        llrd=True,
        layer_decay=0.9,
        head_lr_mult=2.0,             # head LR = base_lr * 2.0; conservative for DAPT model
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # ----------- Train -----------
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ----------- Evaluate -----------
    print("Evaluating on validation set.")
    val_metrics = trainer.evaluate()
    print(val_metrics)

    run_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_name": model_name,
        "dataset_path": dataset_path,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "num_train_epochs": num_train_epochs,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "eval_steps": training_args.eval_steps,
        "save_steps": training_args.save_steps,
        "best_metric": getattr(trainer.state, "best_metric", None),
        "best_model_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
        "global_step": trainer.state.global_step,
    }

    append_metrics_to_csv(output_dir, "validation", val_metrics, run_info)

    if test_dataset is not None:
        print("Evaluating on test set.")
        test_metrics = trainer.evaluate(test_dataset)
        print(test_metrics)
        append_metrics_to_csv(output_dir, "test", test_metrics, run_info)


    print_predicted_tag_counts(trainer, eval_dataset, id2label)
    # ----------- Save final model -----------
    print(f"Saving model and tokenizer to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
# =====================================================================
# 4. CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TEI NER model with IOB2 labels."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to HF dataset saved by build_tei_ner_corpus.py (hf_dataset directory).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-german-cased",
        help="Base transformer model name (Hugging Face hub).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (for top encoder layer; lower layers get less via LLRD).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    parser.add_argument("--use_crf", action="store_true", help="Use CRF decoding head")

    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
        use_crf=args.use_crf,
    )
