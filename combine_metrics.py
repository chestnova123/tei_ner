import argparse
from pathlib import Path
import pandas as pd


def find_metrics_files(root: Path, pattern: str = "*metrics.csv") -> list[Path]:
    return sorted(root.rglob(pattern))


def combine_metrics(root: Path, out_csv: Path) -> pd.DataFrame:
    files = find_metrics_files(root)
    if not files:
        raise SystemExit(f"No metrics.csv files found under: {root}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] Skipping {f} (could not read CSV): {e}")
            continue

        # Provenance columns
        df.insert(0, "metrics_file", str(f))
        df.insert(1, "run_dir", str(f.parent))
        df.insert(2, "run_name", f.parent.name)

        # Helpful: normalize split name if missing
        if "split" not in df.columns:
            df["split"] = ""

        # Helpful: ensure common numeric columns are numeric (ignore errors)
        for col in df.columns:
            if col.startswith(("eval_", "precision", "recall", "f1")) or col in (
                "best_metric",
                "global_step",
                "learning_rate",
                "weight_decay",
                "batch_size",
                "num_train_epochs",
                "gradient_accumulation_steps",
                "eval_steps",
                "save_steps",
                "seed",
                "data_seed",
            ):
                df[col] = pd.to_numeric(df[col], errors="ignore")

        dfs.append(df)

    if not dfs:
        raise SystemExit("Found metrics.csv files, but none could be read successfully.")

    combined = pd.concat(dfs, ignore_index=True, sort=False)

    # Sort to make comparison easier (if columns exist)
    sort_cols = [c for c in ["model_name", "learning_rate", "seed", "split", "timestamp"] if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    return combined


def main():
    ap = argparse.ArgumentParser(description="Combine all metrics.csv files under a root directory into one CSV.")
    ap.add_argument("--root", required=True, type=Path, help="Root folder that contains run subfolders with metrics.csv")
    ap.add_argument("--out", required=True, type=Path, help="Output combined CSV path")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve()

    combined = combine_metrics(root, out)
    print(f"Combined {len(combined)} rows from {len(find_metrics_files(root))} files.")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()