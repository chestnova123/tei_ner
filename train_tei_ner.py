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
from seqeval.scheme import BILOU
import inspect
import torch
from torch.nn import CrossEntropyLoss
from collections import Counter
import math
import torch.nn.functional as F
import re


# =====================================================================
# 1. Label definitions (must match build_tei_ner_corpus.py exactly)
# =====================================================================

TYPES = [
    "PERSON",
    "PEOPLE",
    "PLACE",
    "ORG",
    "BIBL",
    "ARTEFACT",
    "CONCEPT",
    "OTHER",
]

LABELS = ["O"]
for t in TYPES:
    LABELS.extend([f"B-{t}", f"I-{t}", f"L-{t}", f"U-{t}"])

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}


# =====================================================================
# 2. Metrics: seqeval for entity-level F1
# =====================================================================


def align_predictions(predictions, label_ids):
    """
    Convert model outputs + gold labels into label strings for seqeval.
    Ignores tokens with label_id == -100.
    """
    preds = np.argmax(predictions, axis=2)

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
            example_preds.append(id2label[preds[i, j]])
        out_label_list.append(example_labels)
        out_pred_list.append(example_preds)

    return out_pred_list, out_label_list

def per_type_scores(labels_list, preds_list):
    """
    Returns dict with per-type precision/recall/f1 from seqeval classification_report.
    Uses BILOU strict mode.
    """
    rep = classification_report(
        labels_list,
        preds_list,
        mode="strict",
        scheme=BILOU,
        output_dict=True,
        zero_division=0,
    )

    # rep keys are entity type names (e.g. "PERSON") plus averages
    out = {}
    for t in TYPES:
        if t in rep:
            out[f"recall_{t}"] = rep[t]["recall"]
            out[f"f1_{t}"] = rep[t]["f1-score"]
    return out

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds_list, labels_list = align_predictions(predictions, labels)

    # DEBUG: inspect what seqeval is seeing
    flat_gold = [t for seq in labels_list for t in seq]
    flat_pred = [t for seq in preds_list for t in seq]
    gold_counts = Counter(flat_gold)
    pred_counts = Counter(flat_pred)

    print("DEBUG gold top tags:", gold_counts.most_common(10))
    print("DEBUG pred top tags:", pred_counts.most_common(10))
    print("DEBUG #gold non-O:", sum(1 for t in flat_gold if t != "O"))
    print("DEBUG #pred non-O:", sum(1 for t in flat_pred if t != "O"))
    
    # Tell seqeval explicitly that we are using BILOU, in strict mode.
    precision = precision_score(labels_list, preds_list, mode="strict", scheme=BILOU)
    recall = recall_score(labels_list, preds_list, mode="strict", scheme=BILOU)
    f1 = f1_score(labels_list, preds_list, mode="strict", scheme=BILOU)

    out = {"precision": precision, "recall": recall, "f1": f1}
    out.update(per_type_scores(labels_list, preds_list))
    return out


# =============================
# 2.1 Weighted labels
# ==============================

def compute_label_counts(train_dataset):
    counts = Counter()
    for labels in train_dataset["labels"]:
        for lab in labels:
            if lab != -100:
                counts[int(lab)] += 1
    return counts
    
def make_class_weights(label2id, counts, scheme="log_inv", smoothing=1.0):
    """
    Returns torch.FloatTensor of shape [num_labels]
    - scheme:
        "inv"      : w = 1 / (count + smoothing)
        "sqrt_inv" : w = 1 / sqrt(count + smoothing)
        "log_inv"  : w = 1 / log(count + c)  (very gentle)
    """
    num_labels = len(label2id)
    weights = torch.ones(num_labels, dtype=torch.float)

    for lab_id in range(num_labels):
        c = counts.get(lab_id, 0)
        if c == 0:
            # if a label never appears, don't blow up the weight
            weights[lab_id] = 0.0
            continue

        if scheme == "inv":
            weights[lab_id] = 1.0 / (c + smoothing)
        elif scheme == "sqrt_inv":
            weights[lab_id] = 1.0 / math.sqrt(c + smoothing)
        elif scheme == "log_inv":
            # c must be >= 1 here
            weights[lab_id] = 1.0 / math.log(c + 10.0)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    # Normalize weights
    nonzero = weights[weights > 0]
    if len(nonzero) > 0:
        weights = weights / nonzero.mean()

    return weights


def tag_to_type(label: str):
    """
    Map a label like 'B-PERSON' or 'U-ORG' to its TYPE (e.g. 'PERSON', 'ORG').
    Returns None for 'O' or anything unexpected.
    """
    if label == "O":
        return None
    # Expect format X-TYPE where X in {B,I,L,U}
    m = re.match(r"^[BILU]-(.+)$", label)
    if not m:
        return None
    return m.group(1)


def compute_type_counts(train_dataset, id2label):
    """
    Count tokens per TYPE from train_dataset['labels'] ignoring -100.
    Returns Counter like {'PERSON': 1234, 'ORG': 567, ...}
    """
    counts = Counter()
    for labs in train_dataset["labels"]:
        for lab_id in labs:
            if lab_id == -100:
                continue
            lab = id2label[int(lab_id)]
            t = tag_to_type(lab)
            if t is not None:
                counts[t] += 1
    return counts


def make_type_level_class_weights(
    label2id,
    id2label,
    train_dataset,
    scheme="sqrt_inv",
    smoothing=1.0,
    o_weight=0.9,
    min_w=0.2,
    max_w=3.0,
):
    """
    Create per-label weights where all B/I/L/U tags of a TYPE share the same weight.

    scheme:
      - "inv":      w_type = 1/(count + smoothing)
      - "sqrt_inv": w_type = 1/sqrt(count + smoothing)
      - "log_inv":  w_type = 1/log(count + 10)  (gentle)

    o_weight:
      Multiplicative factor applied to O after normalization (keep near 1.0 to avoid entity spam)

    min_w/max_w:
      Clamp weights to avoid extremes.
    """
    type_counts = compute_type_counts(train_dataset, id2label)

    # Compute raw weights per TYPE
    type_weight = {}
    for t, c in type_counts.items():
        if c <= 0:
            continue
        if scheme == "inv":
            w = 1.0 / (c + smoothing)
        elif scheme == "sqrt_inv":
            w = 1.0 / math.sqrt(c + smoothing)
        elif scheme == "log_inv":
            w = 1.0 / math.log(c + 10.0)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
        type_weight[t] = w

    # Build per-label weights
    num_labels = len(label2id)
    weights = torch.ones(num_labels, dtype=torch.float)

    # Set weights for entity tags based on their TYPE
    for lab, lab_id in label2id.items():
        if lab == "O":
            continue
        t = tag_to_type(lab)
        if t is None:
            continue
        # If a type never appears, set to 0 to avoid nonsense gradients
        if t not in type_weight:
            weights[lab_id] = 0.0
        else:
            weights[lab_id] = float(type_weight[t])

    # Normalize weights so mean of nonzero weights is 1.0
    nonzero = weights[weights > 0]
    if len(nonzero) > 0:
        weights = weights / nonzero.mean()

    # Apply O weight after normalization (keeps it interpretable)
    if "O" in label2id:
        weights[label2id["O"]] *= float(o_weight)

    # Clamp to avoid extremes
    weights = torch.clamp(weights, min=min_w, max=max_w)

    return weights

class WeightedTokenTrainer(Trainer):
    def __init__(
        self,
        class_weights=None,
        label_smoothing=0.0,
        llrd=False,
        layer_decay=0.9,
        head_lr_mult=5.0,
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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
    head_lr_mult: float = 5.0,
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
    resume_from_checkpoint: str | None = None,
):
    dataset_path = str(Path(dataset_path).expanduser().resolve())
    output_dir = str(Path(output_dir).expanduser().resolve())

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
    counts = compute_label_counts(train_dataset)
    class_weights = make_type_level_class_weights(
        label2id=label2id,
        id2label=id2label,
        train_dataset=train_dataset,
        scheme="sqrt_inv",   # or "log_inv" for gentler
        smoothing=1.0,
        o_weight=0.9,        # IMPORTANT: keep closer to 1 to avoid entity spam
        min_w=0.2,
        max_w=3.0,
        )

    print_weights(class_weights, id2label)


    print(f"Loading tokenizer & model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    config = AutoConfig.from_pretrained(
        model_name,
        hidden_dropout_prob=0.05,
        attention_probs_dropout_prob=0.05,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,

        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,

        logging_strategy="steps",
        logging_steps=50,

        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,

        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        bf16=True,
        disable_tqdm=False,
        report_to="none",

        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        group_by_length=True,
        length_column_name="length",
    )

    trainer = WeightedTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=0.05,
        llrd=True,
        layer_decay=0.9,
        head_lr_mult=5.0,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ----------- Train -----------
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ----------- Evaluate -----------
    print("Evaluating on validation set...")
    metrics = trainer.evaluate()
    print(metrics)

    if test_dataset is not None:
        print("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_dataset)
        print(test_metrics)

    # ----------- Save final model -----------
    print(f"Saving model and tokenizer to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
# =====================================================================
# 4. CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TEI NER model with BILOU labels."
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
        default=5e-5,
        help="Learning rate.",
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
    )
