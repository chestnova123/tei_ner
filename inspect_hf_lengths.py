import argparse
import numpy as np
from pathlib import Path
from datasets import load_from_disk


def summarize(lengths, name):
    lengths = np.asarray(lengths, dtype=np.int64)
    if len(lengths) == 0:
        print(f"\n=== {name} ===\n(empty)")
        return

    print(f"\n=== {name} ===")
    print(f"count:  {len(lengths)}")
    print(f"mean:   {lengths.mean():.2f}")
    print(f"median: {np.median(lengths):.2f}")
    print(f"min:    {lengths.min()}")
    print(f"max:    {lengths.max()}")
    for p in [90, 95, 99]:
        print(f"p{p}:    {np.percentile(lengths, p):.0f}")


def compute_lengths(ds, input_key="input_ids", label_key="labels"):
    token_lens = []
    label_lens = []
    active_label_lens = []
    mismatches = 0

    # Iterating row-by-row is safest across dataset formats
    for ex in ds:
        inp = ex[input_key]
        lab = ex[label_key]

        tl = len(inp)
        ll = len(lab)

        token_lens.append(tl)
        label_lens.append(ll)
        active_label_lens.append(sum(1 for x in lab if x != -100))

        if tl != ll:
            mismatches += 1

    return token_lens, label_lens, active_label_lens, mismatches


def inspect_split(ds, split_name, input_key="input_ids", label_key="labels"):
    print(f"\n\n#############################")
    print(f"Split: {split_name} | examples: {len(ds)}")
    print(f"Columns: {ds.column_names}")
    print(f"#############################")

    token_lens, label_lens, active_label_lens, mismatches = compute_lengths(
        ds, input_key=input_key, label_key=label_key
    )

    summarize(token_lens, f"{split_name}: Tokenized sequence length (len({input_key}))")
    summarize(label_lens, f"{split_name}: Label sequence length (len({label_key}))")
    summarize(active_label_lens, f"{split_name}: Active label length (# {label_key} != -100)")

    print(f"\n{split_name}: mismatched {input_key} vs {label_key} lengths: {mismatches} / {len(ds)}")


def main(dataset_path, input_key="input_ids", label_key="labels"):
    dataset_path = str(Path(dataset_path).expanduser().resolve())
    print(f"Loading dataset from: {dataset_path}")

    ds_dict = load_from_disk(dataset_path)

    # ds_dict is usually a DatasetDict with keys like train/validation/test
    # But sometimes you may have saved a single Dataset.
    if hasattr(ds_dict, "keys"):
        splits = list(ds_dict.keys())
        print(f"Found splits: {splits}")

        for split in ["train", "validation", "test"]:
            if split in ds_dict:
                inspect_split(ds_dict[split], split, input_key=input_key, label_key=label_key)

        # Also inspect any non-standard splits
        for split in splits:
            if split not in {"train", "validation", "test"}:
                inspect_split(ds_dict[split], split, input_key=input_key, label_key=label_key)
    else:
        # Single Dataset
        inspect_split(ds_dict, "dataset", input_key=input_key, label_key=label_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect tokenized and label sequence lengths for a HF dataset saved with save_to_disk()."
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the HF dataset directory (load_from_disk).")
    parser.add_argument("--input_key", type=str, default="input_ids", help="Column name for tokenized IDs.")
    parser.add_argument("--label_key", type=str, default="labels", help="Column name for labels.")
    args = parser.parse_args()

    main(args.dataset_path, input_key=args.input_key, label_key=args.label_key)
