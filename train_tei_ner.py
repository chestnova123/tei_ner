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
    
def make_class_weights(label2id, counts, scheme="sqrt_inv", smoothing=1.0):
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

class WeightedTokenTrainer(Trainer):
    def __init__(self, class_weights=None, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # torch tensor
        self.label_smoothing = float(label_smoothing)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")  # [B, T]
        outputs = model(**inputs)
        logits = outputs.get("logits")  # [B, T, C]

        # Flatten
        B, T, C = logits.shape
        logits = logits.view(-1, C)          # [B*T, C]
        labels = labels.view(-1)             # [B*T]

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
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)  # [C]
        else:
            weight = None

        eps = self.label_smoothing

        if eps > 0.0:
            # Smoothed NLL:
            # loss = (1-eps)*NLL + eps*uniform_loss (excluding correct class is common;
            # uniform over all classes is also fine. We'll do uniform over all classes.)
            nll = F.nll_loss(log_probs, labels, reduction="none")  # [N]
            smooth = -log_probs.mean(dim=-1)                      # [N]  mean over classes

            loss_per_token = (1.0 - eps) * nll + eps * smooth     # [N]

            # Apply class weights to tokens by gold label (common practical choice)
            if weight is not None:
                loss_per_token = loss_per_token * weight[labels]

            loss = loss_per_token.mean()
        else:
            # Standard weighted CE
            loss = F.cross_entropy(
                logits,
                labels,
                weight=weight,
                reduction="mean",
            )

        return (loss, outputs) if return_outputs else loss
        
def print_weights(class_weights, id2label, topk=999):
    rows = [(id2label[i], float(w)) for i, w in enumerate(class_weights)]
    rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
    print("\nClass weights (sorted high→low):")
    for lab, w in rows_sorted[:topk]:
        print(f"{lab:<12s} {w:8.4f}")


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

    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    if test_dataset is not None:
        print(f"Test examples: {len(test_dataset)}")

    # -----------------------------
    # Compute class weights
    # -----------------------------
    counts = compute_label_counts(train_dataset)

    class_weights = make_class_weights(
        label2id,
        counts,
        scheme="sqrt_inv",
        smoothing=1.0,
    )

    # Downweight "O" a bit so the model doesn't learn to spam O.
    O_WEIGHT = 0.5   # try 0.7, 0.5, 0.3
    class_weights[label2id["O"]] = class_weights[label2id["O"]] * O_WEIGHT

    # Optional: clamp to avoid extreme weights
    class_weights = torch.clamp(class_weights, min=0.1, max=10.0)

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
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,

        logging_strategy="steps",
        logging_steps=1000,

        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
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
        lr_scheduler_type="linear",
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
