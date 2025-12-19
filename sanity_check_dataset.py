import json
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer


# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

MODEL_NAME = "bert-base-german-cased"
DATASET_PATH = r"C:\Users\elena\Documents\GitHub\semper-tei_new\scripts\R_transformer_based_entity_prediction\extraction_test\test_out\hf_dataset"

# How many examples to inspect
N_EXAMPLES = 15


# --------------------------------------------------------
# Load tokenizer & dataset
# --------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset_dict = load_from_disk(DATASET_PATH)

# You can inspect train/validation/test separately:
ds = dataset_dict["train"]

print(f"Dataset loaded. Train examples: {len(ds)}")
print("-" * 80)


# --------------------------------------------------------
# Helper: decode token-level BILOU labels
# --------------------------------------------------------


def decode_labels(input_ids, labels_ids, id2label):
    """
    Converts token IDs and label IDs into readable tokens & BILOU labels.
    Skips label = -100 (special tokens).
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    decoded = []

    for tok, lab_id in zip(tokens, labels_ids):
        if lab_id == -100:
            continue  # skip special tokens
        label = id2label[lab_id]
        decoded.append((tok, label))

    return decoded


# Load the label maps from your corpus-building script
# They must exist exactly as produced during corpus construction:
LABELS = [
    "O",
    "B-PERSON",
    "I-PERSON",
    "L-PERSON",
    "U-PERSON",
    "B-PEOPLE",
    "I-PEOPLE",
    "L-PEOPLE",
    "U-PEOPLE",
    "B-PLACE",
    "I-PLACE",
    "L-PLACE",
    "U-PLACE",
    "B-ORG",
    "I-ORG",
    "L-ORG",
    "U-ORG",
    "B-BIBL",
    "I-BIBL",
    "L-BIBL",
    "U-BIBL",
    "B-ARTEFACT",
    "I-ARTEFACT",
    "L-ARTEFACT",
    "U-ARTEFACT",
    "B-CONCEPT",
    "I-CONCEPT",
    "L-CONCEPT",
    "U-CONCEPT",
    "B-OTHER",
    "I-OTHER",
    "L-OTHER",
    "U-OTHER",
]

id2label = {i: lab for i, lab in enumerate(LABELS)}


# --------------------------------------------------------
# Pretty-print examples
# --------------------------------------------------------


def print_example(example):
    print("=" * 80)
    print(f"Document {example['doc_id']} Paragraph {example['p_index']}")
    print("-" * 80)

    # Reconstructed text
    print("TEXT:")
    print(example["text"])
    print()

    # Entities
    print("ENTITIES:")
    if example["entities"]:
        for ent in example["entities"]:
            span = example["text"][ent["start"] : ent["end"]]
            print(f"  {ent} → '{span}'")
    else:
        print("  No entities found")
    print()

    # Token-level BILOU labels
    print("TOKENS WITH BILOU LABELS:")
    decoded = decode_labels(example["input_ids"], example["labels"], id2label)
    for tok, lab in decoded:
        print(f"{tok:20s}  {lab}")
    print()


# --------------------------------------------------------
# Show a few examples
# --------------------------------------------------------

for i in range(min(N_EXAMPLES, len(ds))):
    print_example(ds[i])
