from datasets import load_from_disk
from collections import Counter
from pathlib import Path

# 1) Point this to your saved dataset folder (the directory that contains train/validation/test)
DATASET_PATH = r"C:\Users\elena\Documents\WORK\USI\STIL PROJECT\ML\dapt\hf_dataset"  # <-- change this

# 2) Define labels exactly like in your training script
TYPES = ["PERSON","PEOPLE","PLACE","ORG","BIBL","ARTEFACT","CONCEPT","OTHER"]
LABELS = ["O"]
for t in TYPES:
    LABELS.extend([f"B-{t}", f"I-{t}", f"L-{t}", f"U-{t}"])

id2label = {i: l for i, l in enumerate(LABELS)}

# 3) Load dataset and count
ds = load_from_disk(DATASET_PATH)
train = ds["train"]

counts = Counter()
bibl_counts = Counter()

for seq in train["labels"]:
    for lab_id in seq:
        if lab_id == -100:
            continue
        lab = id2label[int(lab_id)]
        counts[lab] += 1
        if lab.endswith("-BIBL"):
            bibl_counts[lab] += 1

total_labeled_tokens = sum(counts.values())
total_bibl_tokens = sum(bibl_counts.values())

print("Train examples:", len(train))
print("Total labeled tokens (excluding -100):", total_labeled_tokens)
print("Total BIBL tokens:", total_bibl_tokens)
print("BIBL % of labeled tokens:", (total_bibl_tokens / total_labeled_tokens) if total_labeled_tokens else 0)

print("\nBIBL breakdown:")
for k, v in bibl_counts.most_common():
    print(f"{k:8s} {v}")