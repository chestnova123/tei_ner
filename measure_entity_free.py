from datasets import load_from_disk
from collections import Counter
import numpy as np

ds = load_from_disk(r"C:\Users\elena\Documents\WORK\USI\STIL PROJECT\ML\corpus_all\hf_dataset")

# Define labels to match corpus builder exactly
TYPES = [
    "ARTEFACT", "CONCEPT", "PERSON", "PLACE", "ORG", "CULTURE", "BIBL",
]
LABELS   = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

# Extract splits
train_dataset = ds["train"]
eval_dataset  = ds["validation"]
test_dataset  = ds["test"]

# Token length stats
for split_name, split in [("train", train_dataset), ("validation", eval_dataset), ("test", test_dataset)]:
    total        = len(split)
    empty        = sum(1 for ex in split if len(ex["entities"]) == 0)
    real_lengths = [sum(ex["attention_mask"]) for ex in split]

    print(f"\n{split_name}:")
    print(f"  Total examples:          {total}")
    print(f"  Entity-free examples:    {empty}  ({100*empty/total:.1f}%)")
    print(f"  Mean real tokens:        {np.mean(real_lengths):.0f}")
    print(f"  Median real tokens:      {np.median(real_lengths):.0f}")
    print(f"  Examples < 50 tokens:    {sum(1 for l in real_lengths if l < 50)}")
    print(f"  Examples < 100 tokens:   {sum(1 for l in real_lengths if l < 100)}")

# Check what entity tokens would be lost at threshold=40
threshold = 37
filtered_out = [
    ex for ex in train_dataset
    if sum(ex["attention_mask"]) < threshold
]

lost_entities = Counter()
for ex in filtered_out:
    for label_id in ex["labels"]:
        if label_id not in (-100, label2id["O"]):
            lost_entities[id2label[label_id]] += 1

print(f"\nExamples removed at threshold={threshold}: {len(filtered_out)}")
print(f"Entity tokens lost:")
for label, count in sorted(lost_entities.items()):
    print(f"  {label:15s}: {count}")