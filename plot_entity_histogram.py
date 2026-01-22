import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_entity_counts(jsonl_path: str) -> Counter:
    counts = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)

            # Your file uses: ex["entities"] = [{"start":..,"end":..,"type":..}, ...]
            for ent in ex.get("entities", []):
                t = ent.get("type") or ent.get("label")
                if t:
                    counts[t] += 1

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to train.jsonl")
    parser.add_argument("--outdir", default="plots", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    counts = load_entity_counts(str(input_path))
    counts.pop("OTHER", None)
    counts.pop("O", None)

    if not counts:
        raise RuntimeError("No entities found. Check JSONL format / field names.")

    # Convert to dataframe for sorting + printing
    df = (
        pd.DataFrame({"entity_type": list(counts.keys()), "count": list(counts.values())})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


    # Save counts CSV (handy for presentation / debugging)
    csv_path = outdir / "entity_counts.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.bar(df["entity_type"], df["count"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Entity counts in train set")
    plt.ylabel("Count")
    plt.tight_layout()

    img_path = outdir / "entity_histogram.png"
    plt.savefig(img_path, dpi=200)

    print("=== Entity counts ===")
    print(df.to_string(index=False))
    print()
    print(f"Saved CSV:  {csv_path}")
    print(f"Saved plot: {img_path}")


if __name__ == "__main__":
    main()
