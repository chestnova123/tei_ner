import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
)
from seqeval.metrics import classification_report
from seqeval.scheme import BILOU


# ------------------------------------------------------
# 1. Labels (must match training)
# ------------------------------------------------------

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


# ------------------------------------------------------
# 2. Helper: align predictions to label strings
# ------------------------------------------------------


def align_predictions(predictions, label_ids):
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


# ------------------------------------------------------
# 3. Confusion matrix at entity-type level (token-based)
# ------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


def build_type_confusion_matrix(labels_list, preds_list, ignore_background=True):
    """
    Build a confusion matrix over entity TYPES (PERSON, PLACE, ...).

    - We strip the B/I/L/U prefix, so we compare at the type level.
    - If ignore_background=True:
        we skip tokens where gold='O' AND pred='O'
        (true negatives which massively dominate and ruin the color scale).
    - We still include cases where either gold or pred is 'O'
        (missed entities or false positives).
    """
    types = TYPES + ["O"]  # last entry is O
    idx = {t: i for i, t in enumerate(types)}
    mat = np.zeros((len(types), len(types)), dtype=int)

    def extract_type(tag):
        if tag == "O":
            return "O"
        parts = tag.split("-", 1)
        if len(parts) == 2:
            return parts[1]
        return "O"

    for gold_seq, pred_seq in zip(labels_list, preds_list):
        for g, p in zip(gold_seq, pred_seq):
            g_type = extract_type(g)
            p_type = extract_type(p)

            if g_type == "O" and p_type == "O" and ignore_background:
                # Skip pure background tokens
                continue

            if g_type not in idx:
                g_type = "O"
            if p_type not in idx:
                p_type = "O"

            mat[idx[g_type], idx[p_type]] += 1

    return types, mat


def plot_confusion_matrix(types, mat, out_path, drop_O=True):
    """
    Plot confusion matrix.

    If drop_O=True, we remove the 'O' row/column from the plot so you only see
    entity types (but the counts still included O-misses / O-false-positives).
    """
    if drop_O and "O" in types:
        o_index = types.index("O")
        # remove row and column for 'O'
        mat = np.delete(mat, o_index, axis=0)
        mat = np.delete(mat, o_index, axis=1)
        types = [t for t in types if t != "O"]

    plt.figure(figsize=(8, 6))
    plt.imshow(mat, aspect="auto")
    plt.xticks(ticks=range(len(types)), labels=types, rotation=45, ha="right")
    plt.yticks(ticks=range(len(types)), labels=types)
    plt.xlabel("Predicted type")
    plt.ylabel("Gold type")
    plt.title("Entity Type Confusion Matrix (token-level, no O/O background)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------------------------------
# 4. Per-type F1 bar chart
# ------------------------------------------------------


def parse_classification_report_text(report_text):
    """
    Parse seqeval's classification_report (text) into a dict:
    {label: {"precision": float, "recall": float, "f1": float, "support": int}, ...}
    """
    lines = report_text.strip().splitlines()
    metrics = {}
    for line in lines[2:]:  # skip header lines
        parts = line.split()
        if len(parts) < 5:
            continue
        label = parts[0]
        # skip 'micro avg', 'macro avg', etc.
        if "-" in label and label not in TYPES:
            # seqeval uses names like 'B-PERSON', but with scheme=BILOU, labels are types
            pass
        if label in ["micro", "macro", "weighted"]:
            continue
        try:
            precision = float(parts[1])
            recall = float(parts[2])
            f1 = float(parts[3])
            support = int(parts[4])
        except ValueError:
            continue
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return metrics


def plot_f1_bar_chart(type_metrics, out_path):
    labels = []
    f1s = []
    for t in TYPES:
        if t in type_metrics:
            labels.append(t)
            f1s.append(type_metrics[t]["f1"])

    plt.figure(figsize=(8, 4))
    plt.bar(labels, f1s)
    plt.xlabel("Entity type")
    plt.ylabel("F1 score")
    plt.title("Per-type F1 (test set)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------------------------------
# 5. Main evaluation routine
# ------------------------------------------------------


def main(model_dir, dataset_path, out_dir=None):
    model_dir = Path(model_dir).expanduser().resolve()
    dataset_path = Path(dataset_path).expanduser().resolve()
    if out_dir is None:
        out_dir = model_dir
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {dataset_path}")
    ds_dict = load_from_disk(str(dataset_path))
    test_dataset = ds_dict["test"]
    print(f"Test examples: {len(test_dataset)}")

    print(f"Loading model and tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir),
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Running prediction on test set...")
    predictions_output = trainer.predict(test_dataset)
    predictions = predictions_output.predictions
    label_ids = predictions_output.label_ids

    preds_list, labels_list = align_predictions(predictions, label_ids)

    print("\n=== Classification Report (test set, BILOU, strict mode) ===")
    report_text = classification_report(
        labels_list,
        preds_list,
        digits=3,
        mode="strict",
        scheme=BILOU,
    )
    print(report_text)

    # Save report to file
    (out_dir / "test_classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )

    # Confusion matrix
    types, mat = build_type_confusion_matrix(labels_list, preds_list)
    plot_confusion_matrix(types, mat, out_dir / "test_confusion_matrix.png")
    print(f"Saved confusion matrix to {out_dir / 'test_confusion_matrix.png'}")

    # Per-type F1 bar chart
    type_metrics = parse_classification_report_text(report_text)
    plot_f1_bar_chart(type_metrics, out_dir / "test_per_type_f1.png")
    print(f"Saved per-type F1 bar chart to {out_dir / 'test_per_type_f1.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate TEI NER model with BILOU scheme and generate plots."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory with the trained model (same as output_dir from training).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to HF dataset (hf_dataset directory).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save plots; defaults to model_dir if not set.",
    )

    args = parser.parse_args()
    main(args.model_dir, args.dataset_path, args.out_dir)
