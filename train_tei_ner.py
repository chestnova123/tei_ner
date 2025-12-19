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
)
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
from seqeval.scheme import BILOU
import inspect


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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds_list, labels_list = align_predictions(predictions, labels)

    # Tell seqeval explicitly that we are using BILOU, in strict mode.
    precision = precision_score(labels_list, preds_list, mode="strict", scheme=BILOU)
    recall = recall_score(labels_list, preds_list, mode="strict", scheme=BILOU)
    f1 = f1_score(labels_list, preds_list, mode="strict", scheme=BILOU)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =====================================================================
# 3. Main training function
# =====================================================================


def main(
    dataset_path: str,
    model_name: str,
    output_dir: str,
    num_train_epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
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

    print(f"Loading tokenizer & model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    print("Transformers version:", transformers.__version__)
    print("TrainingArguments class:", transformers.TrainingArguments)
    print("TrainingArguments module:", transformers.TrainingArguments.__module__)
    print("Signature:", inspect.signature(transformers.TrainingArguments.__init__))

    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        bf16=True,
        report_to=[],  # set to ["wandb"] etc. if you want logging
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ----------- Train -----------
    trainer.train()

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

    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
