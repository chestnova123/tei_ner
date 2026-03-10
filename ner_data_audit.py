import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification


# ----------------------------
# Helpers: BIO spans from tags
# ----------------------------

def id2label_from_dataset_or_model(ds, model) -> Dict[int, str]:
    # Try model config first (most reliable)
    if hasattr(model, "config") and getattr(model.config, "id2label", None):
        return {int(k): v for k, v in model.config.id2label.items()}
    # Fall back to dataset features if present
    try:
        feat = ds["train"].features["labels"]
        if hasattr(feat, "feature") and hasattr(feat.feature, "names"):
            names = feat.feature.names
            return {i: names[i] for i in range(len(names))}
    except Exception:
        pass
    raise RuntimeError("Could not infer id2label mapping. Ensure your model has config.id2label.")

def strip_prefix(tag: str) -> str:
    if tag == "O":
        return "O"
    if "-" in tag:
        return tag.split("-", 1)[1]
    return tag

def is_bio(tag: str) -> bool:
    return tag == "O" or tag.startswith("B-") or tag.startswith("I-")

@dataclass
class Span:
    split: str
    example_idx: int
    start: int
    end: int  # exclusive
    ent_type: str
    token_text: str
    token_len: int

def bio_spans_from_labels(tokens: List[str], tags: List[str]) -> List[Tuple[int, int, str]]:
    """
    Return list of (start, end_excl, type) spans from BIO tags at token level.
    Assumes tags length matches tokens length (already filtered for -100 positions).
    """
    spans = []
    i = 0
    while i < len(tags):
        t = tags[i]
        if t.startswith("B-"):
            ent = strip_prefix(t)
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{ent}":
                j += 1
            spans.append((i, j, ent))
            i = j
        else:
            i += 1
    return spans

def join_tokens_for_display(tok_list: List[str]) -> str:
    # Tokenizer-dependent; "▁" (sentencepiece) indicates word starts in XLM-R.
    # We'll make it readable by replacing ▁ with space and collapsing.
    s = "".join(tok_list)
    s = s.replace("▁", " ")
    s = " ".join(s.split())
    return s.strip()

def detect_marker(token_str: str) -> bool:
    # catches your literal markers like ⟦ADD⟧, ⟦DEL⟧, ⟦HI⟧, ⟦LAT⟧, ⟦KUR⟧
    return "⟦" in token_str or "⟧" in token_str


# ----------------------------
# Model predictions
# ----------------------------

def predict_tags(
    model,
    ds_split,
    id2label: Dict[int, str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8,
) -> List[List[str]]:
    """
    Returns predicted tags per example (filtered to positions where gold != -100 if labels exist).
    If CRF is present and torchcrf is installed, will decode. Otherwise argmax logits.
    """
    model.to(device)
    model.eval()

    use_crf = hasattr(model, "crf")
    crf_ok = False
    if use_crf:
        try:
            import torchcrf  # noqa: F401
            crf_ok = True
        except Exception:
            crf_ok = False

    preds_all: List[List[str]] = []

    for start in range(0, len(ds_split), batch_size):
        batch = ds_split[start:start + batch_size]
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
        attn = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
        labels = batch.get("labels", None)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device) if labels is not None else None

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits  # [B,T,C]

            if use_crf and crf_ok:
                # mask out ignored tokens if labels exist, else use attention mask
                mask = attn.bool()
                if labels_t is not None:
                    ignore = labels_t.eq(-100)
                    mask = mask & (~ignore)
                # torchcrf requirement
                mask[:, 0] = True
                decoded = model.crf.decode(logits, mask=mask)  # list[list[int]]
                pred_ids = torch.zeros((logits.size(0), logits.size(1)), dtype=torch.long, device=device)
                for i, seq in enumerate(decoded):
                    L = min(len(seq), logits.size(1))
                    pred_ids[i, :L] = torch.tensor(seq[:L], device=device)
            else:
                pred_ids = torch.argmax(logits, dim=-1)

        # convert to tag strings, filtering to gold != -100 if available
        for b in range(pred_ids.size(0)):
            if labels_t is not None:
                keep = labels_t[b].ne(-100).detach().cpu().numpy().astype(bool)
            else:
                keep = attn[b].ne(0).detach().cpu().numpy().astype(bool)

            ids = pred_ids[b].detach().cpu().numpy()[keep]
            preds_all.append([id2label[int(x)] for x in ids])

    return preds_all


# ----------------------------
# Audit computations
# ----------------------------

def label_counts(ds_split, id2label: Dict[int, str]) -> Counter:
    c = Counter()
    for seq in ds_split["labels"]:
        for x in seq:
            if x == -100:
                continue
            c[id2label[int(x)]] += 1
    return c

def entity_counts_and_lengths(ds_split, tokenizer, id2label: Dict[int, str], split_name: str) -> Tuple[Counter, Dict[str, List[int]], List[Span]]:
    ent_counts = Counter()
    lengths_by_type = defaultdict(list)
    all_spans: List[Span] = []

    for idx, ex in enumerate(ds_split):
        labels = ex["labels"]
        input_ids = ex["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # filter by labels != -100
        keep_idx = [i for i, lab in enumerate(labels) if lab != -100]
        toks_f = [tokens[i] for i in keep_idx]
        tags_f = [id2label[int(labels[i])] for i in keep_idx]

        spans = bio_spans_from_labels(toks_f, tags_f)
        for s, e, t in spans:
            ent_counts[t] += 1
            lengths_by_type[t].append(e - s)
            span_text = join_tokens_for_display(toks_f[s:e])
            all_spans.append(Span(split_name, idx, s, e, t, span_text, e - s))

    return ent_counts, lengths_by_type, all_spans

def split_drift_summary(train_counts: Counter, val_counts: Counter, test_counts: Counter) -> pd.DataFrame:
    def norm(cnt: Counter) -> Dict[str, float]:
        total = sum(cnt.values())
        return {k: (v / total if total else 0.0) for k, v in cnt.items()}

    all_keys = sorted(set(train_counts) | set(val_counts) | set(test_counts))
    tr = norm(train_counts); va = norm(val_counts); te = norm(test_counts)
    rows = []
    for k in all_keys:
        rows.append({
            "label": k,
            "train_frac": tr.get(k, 0.0),
            "val_frac": va.get(k, 0.0),
            "test_frac": te.get(k, 0.0),
            "val_minus_train": va.get(k, 0.0) - tr.get(k, 0.0),
            "test_minus_train": te.get(k, 0.0) - tr.get(k, 0.0),
        })
    df = pd.DataFrame(rows).sort_values(by="train_frac", ascending=False)
    return df

def type_confusion_matrix(
    gold_tags: List[List[str]],
    pred_tags: List[List[str]],
    labels_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Type-level confusion ignoring BIO prefix: B-ORG/I-ORG -> ORG. O stays O.
    """
    cm = Counter()
    gold_types = Counter()
    pred_types = Counter()

    for g_seq, p_seq in zip(gold_tags, pred_tags):
        # sequences are already filtered to comparable lengths
        for g, p in zip(g_seq, p_seq):
            gt = strip_prefix(g)
            pt = strip_prefix(p)
            cm[(gt, pt)] += 1
            gold_types[gt] += 1
            pred_types[pt] += 1

    types = sorted(set(gold_types) | set(pred_types))
    mat = pd.DataFrame(0, index=types, columns=types, dtype=int)
    for (gt, pt), v in cm.items():
        mat.loc[gt, pt] = v
    return mat

def marker_leakage(ds_split, tokenizer, id2label: Dict[int, str], split_name: str) -> pd.DataFrame:
    """
    For each label type, compute marker token rate.
    Marker token = token string contains ⟦ or ⟧.
    """
    type_stats = defaultdict(lambda: {"tokens": 0, "marker_tokens": 0})
    overall = {"tokens": 0, "marker_tokens": 0}

    for ex in ds_split:
        labels = ex["labels"]
        input_ids = ex["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        for tok, lab_id in zip(tokens, labels):
            if lab_id == -100:
                continue
            lab = id2label[int(lab_id)]
            t = strip_prefix(lab)
            is_m = detect_marker(tok)
            type_stats[t]["tokens"] += 1
            type_stats[t]["marker_tokens"] += int(is_m)
            overall["tokens"] += 1
            overall["marker_tokens"] += int(is_m)

    rows = []
    for t, d in type_stats.items():
        rate = d["marker_tokens"] / d["tokens"] if d["tokens"] else 0.0
        rows.append({
            "split": split_name,
            "type": t,
            "tokens": d["tokens"],
            "marker_tokens": d["marker_tokens"],
            "marker_rate": rate,
        })
    df = pd.DataFrame(rows).sort_values(by="marker_rate", ascending=False)
    return df

def sample_random_spans_per_type(spans: List[Span], per_type: int = 100, seed: int = 42) -> Dict[str, List[Span]]:
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for sp in spans:
        by_type[sp.ent_type].append(sp)

    sampled = {}
    for t, arr in by_type.items():
        if not arr:
            continue
        k = min(per_type, len(arr))
        sampled[t] = rng.sample(arr, k)
    return sampled

def alignment_spotcheck(ds_split, tokenizer, id2label: Dict[int, str], split_name: str, n: int = 20, seed: int = 42) -> List[Dict]:
    """
    Prints random GOLD spans with local token context to spot alignment issues.
    """
    rng = random.Random(seed)
    picks = []

    # collect spans first
    _, _, spans = entity_counts_and_lengths(ds_split, tokenizer, id2label, split_name)
    if not spans:
        return []

    chosen = rng.sample(spans, min(n, len(spans)))
    for sp in chosen:
        ex = ds_split[sp.example_idx]
        labels = ex["labels"]
        input_ids = ex["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # filtered indices (labels != -100)
        keep_idx = [i for i, lab in enumerate(labels) if lab != -100]
        toks_f = [tokens[i] for i in keep_idx]
        tags_f = [id2label[int(labels[i])] for i in keep_idx]

        # context window around span
        left = max(0, sp.start - 10)
        right = min(len(toks_f), sp.end + 10)
        ctx_tokens = toks_f[left:right]
        ctx_tags = tags_f[left:right]

        picks.append({
            "split": split_name,
            "example_idx": sp.example_idx,
            "type": sp.ent_type,
            "span_token_len": sp.token_len,
            "span_text": sp.token_text,
            "context_tokens": join_tokens_for_display(ctx_tokens),
            "context_tags": " ".join(ctx_tags),
        })
    return picks


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True, help="Path to HF DatasetDict saved with save_to_disk()")
    ap.add_argument("--model_path", default=None, help="Optional: path to a saved NER model for prediction/confusion matrix")
    ap.add_argument("--out_dir", default=None, help="Output folder for CSV/JSON/txt outputs")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for prediction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spans_per_type", type=int, default=100)
    ap.add_argument("--spotcheck_n", type=int, default=20)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ds = load_from_disk(args.dataset_path)
    if not all(s in ds for s in ("train", "validation", "test")):
        raise SystemExit(f"Expected splits train/validation/test. Found: {list(ds.keys())}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.dataset_path) / "_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    else:
        # fallback: try loading tokenizer from dataset info is not reliable; require model_path for token display
        raise SystemExit("Please provide --model_path to decode tokens and compute confusion matrix.")

    id2label = id2label_from_dataset_or_model(ds, model)

    # 1) Label counts per split
    lc_train = label_counts(ds["train"], id2label)
    lc_val = label_counts(ds["validation"], id2label)
    lc_test = label_counts(ds["test"], id2label)

    pd.DataFrame(lc_train.most_common(), columns=["label", "count"]).to_csv(out_dir / "label_counts_train.csv", index=False)
    pd.DataFrame(lc_val.most_common(), columns=["label", "count"]).to_csv(out_dir / "label_counts_val.csv", index=False)
    pd.DataFrame(lc_test.most_common(), columns=["label", "count"]).to_csv(out_dir / "label_counts_test.csv", index=False)

    # 2) Entity counts & span length stats per type
    ent_train, len_train, spans_train = entity_counts_and_lengths(ds["train"], tokenizer, id2label, "train")
    ent_val, len_val, spans_val = entity_counts_and_lengths(ds["validation"], tokenizer, id2label, "validation")
    ent_test, len_test, spans_test = entity_counts_and_lengths(ds["test"], tokenizer, id2label, "test")

    def length_stats(lengths_by_type: Dict[str, List[int]]) -> pd.DataFrame:
        rows = []
        for t, arr in lengths_by_type.items():
            a = np.array(arr, dtype=float)
            rows.append({
                "type": t,
                "n_spans": int(len(a)),
                "mean_len": float(a.mean()) if len(a) else 0.0,
                "median_len": float(np.median(a)) if len(a) else 0.0,
                "p90_len": float(np.percentile(a, 90)) if len(a) else 0.0,
                "p99_len": float(np.percentile(a, 99)) if len(a) else 0.0,
                "max_len": float(a.max()) if len(a) else 0.0,
            })
        return pd.DataFrame(rows).sort_values(by="n_spans", ascending=False)

    pd.DataFrame(ent_train.most_common(), columns=["type", "count"]).to_csv(out_dir / "entity_counts_train.csv", index=False)
    pd.DataFrame(ent_val.most_common(), columns=["type", "count"]).to_csv(out_dir / "entity_counts_val.csv", index=False)
    pd.DataFrame(ent_test.most_common(), columns=["type", "count"]).to_csv(out_dir / "entity_counts_test.csv", index=False)

    length_stats(len_train).to_csv(out_dir / "span_length_stats_train.csv", index=False)
    length_stats(len_val).to_csv(out_dir / "span_length_stats_val.csv", index=False)
    length_stats(len_test).to_csv(out_dir / "span_length_stats_test.csv", index=False)

    # 3) Split drift summary (label distribution)
    drift = split_drift_summary(lc_train, lc_val, lc_test)
    drift.to_csv(out_dir / "split_drift_label_fractions.csv", index=False)

    # 4) Marker-token leakage (by type)
    ml_train = marker_leakage(ds["train"], tokenizer, id2label, "train")
    ml_val = marker_leakage(ds["validation"], tokenizer, id2label, "validation")
    ml_test = marker_leakage(ds["test"], tokenizer, id2label, "test")
    pd.concat([ml_train, ml_val, ml_test], ignore_index=True).to_csv(out_dir / "marker_leakage_by_type.csv", index=False)

    # 5) Extract 100 random spans per type (train)
    sampled = sample_random_spans_per_type(spans_train, per_type=args.spans_per_type, seed=args.seed)
    with open(out_dir / "random_spans_train.txt", "w", encoding="utf-8") as f:
        for t in sorted(sampled.keys()):
            f.write(f"\n### {t} ({len(sampled[t])})\n")
            for sp in sampled[t]:
                f.write(f"[ex={sp.example_idx} len={sp.token_len}] {sp.token_text}\n")

    # 6) Alignment spot-check (20 random labeled spans)
    spot = alignment_spotcheck(ds["train"], tokenizer, id2label, "train", n=args.spotcheck_n, seed=args.seed)
    pd.DataFrame(spot).to_csv(out_dir / "alignment_spotcheck_train.csv", index=False)

    # 7) Type-level confusion matrix (val) using model predictions
    # Build gold tag sequences from labels for validation
    gold_val = []
    for ex in ds["validation"]:
        labels = ex["labels"]
        # filter -100
        tags = [id2label[int(x)] for x in labels if x != -100]
        gold_val.append(tags)

    pred_val = predict_tags(model, ds["validation"], id2label, batch_size=args.batch_size)

    # Ensure same lengths per example (should match if filtering is consistent)
    min_len = min(len(g) for g in gold_val)
    # if lengths mismatch, truncate each pair to min of the two
    gold_adj, pred_adj = [], []
    for g, p in zip(gold_val, pred_val):
        L = min(len(g), len(p))
        gold_adj.append(g[:L])
        pred_adj.append(p[:L])

    cm = type_confusion_matrix(gold_adj, pred_adj)
    cm.to_csv(out_dir / "type_confusion_matrix_val.csv", index=True)

    # Save a summary JSON
    summary = {
        "dataset_path": args.dataset_path,
        "model_path": args.model_path,
        "out_dir": str(out_dir),
        "label_counts_train_top10": lc_train.most_common(10),
        "entity_counts_train_top10": ent_train.most_common(10),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n✅ Audit written to: {out_dir}\n"
          f"- label_counts_*.csv\n"
          f"- entity_counts_*.csv\n"
          f"- span_length_stats_*.csv\n"
          f"- split_drift_label_fractions.csv\n"
          f"- marker_leakage_by_type.csv\n"
          f"- random_spans_train.txt\n"
          f"- alignment_spotcheck_train.csv\n"
          f"- type_confusion_matrix_val.csv\n")

if __name__ == "__main__":
    main()