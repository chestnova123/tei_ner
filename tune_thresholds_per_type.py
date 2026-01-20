#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification


# ----------------------------
# BILOU decoding (strict)
# ----------------------------
def bilou_to_spans(tags):
    """
    tags: list[str]
    Returns list of (start, end_exclusive, TYPE) spans in token indices.
    Strict BILOU decoding:
      - U-TYPE => single-token entity
      - B-TYPE ... I-TYPE ... L-TYPE => multi-token entity
      - malformed sequences are ignored (do not form spans)
    """
    spans = []
    i = 0
    n = len(tags)
    while i < n:
        t = tags[i]
        if t == "O":
            i += 1
            continue

        if t.startswith("U-"):
            spans.append((i, i + 1, t[2:]))
            i += 1
            continue

        if t.startswith("B-"):
            ent_type = t[2:]
            j = i + 1
            while j < n and tags[j] == f"I-{ent_type}":
                j += 1
            if j < n and tags[j] == f"L-{ent_type}":
                spans.append((i, j + 1, ent_type))
                i = j + 1
            else:
                # malformed: skip just the B token
                i += 1
            continue

        # I- or L- without valid start => skip
        i += 1

    return spans


def spans_from_label_ids(label_ids, id2label):
    tags = [id2label[int(i)] for i in label_ids]
    return bilou_to_spans(tags)


def spans_from_pred_ids_with_scores(pred_ids, probs, id2label):
    """
    pred_ids: [N] int
    probs: [N, C] float
    Returns list of (s, e, type, score)
    where score = mean prob of the predicted label at each token inside the span.
    """
    tags = [id2label[int(i)] for i in pred_ids]
    spans = bilou_to_spans(tags)

    out = []
    for (s, e, t) in spans:
        token_scores = []
        for k in range(s, e):
            lab_id = int(pred_ids[k])
            token_scores.append(float(probs[k, lab_id]))
        score = float(np.mean(token_scores)) if token_scores else 0.0
        out.append((s, e, t, score))
    return out


# ----------------------------
# Strict span+type PRF
# ----------------------------
def strict_prf(gold_spans, pred_spans):
    """
    gold_spans: list[(s,e,type)]
    pred_spans: list[(s,e,type)]
    strict match requires exact tuple equality.
    """
    gold_set = set(gold_spans)
    pred_set = set(pred_spans)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, tp, fp, fn


def prf_by_type(gold_all, pred_all, types):
    """
    gold_all, pred_all: list[(s,e,type)] across entire split
    returns dict {type: (p,r,f1,tp,fp,fn)}
    """
    out = {}
    for t in types:
        gold_t = [(s, e, tt) for (s, e, tt) in gold_all if tt == t]
        pred_t = [(s, e, tt) for (s, e, tt) in pred_all if tt == t]
        out[t] = strict_prf(gold_t, pred_t)
    return out


# ----------------------------
# Cache model outputs on split
# ----------------------------
def cache_scored_spans(ds, model, device):
    """
    Returns:
      gold_spans_list: list of list[(s,e,type)]
      pred_scored_list: list of list[(s,e,type,score)]
    Per example, tokens with label=-100 are removed consistently.
    """
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    gold_spans_list = []
    pred_scored_list = []

    model.eval()
    with torch.no_grad():
        for ex in ds:
            labels = np.asarray(ex["labels"], dtype=int)
            keep = labels != -100

            input_ids = torch.tensor(ex["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = torch.tensor(ex["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(0)  # [N,C]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # [N,C]
            pred_ids = probs.argmax(axis=-1)  # [N]

            # filter to non-ignored
            gold_ids = labels[keep]
            pred_ids_k = pred_ids[keep]
            probs_k = probs[keep]

            gold_spans = spans_from_label_ids(gold_ids.tolist(), id2label)
            pred_scored = spans_from_pred_ids_with_scores(pred_ids_k.tolist(), probs_k, id2label)

            gold_spans_list.append(gold_spans)
            pred_scored_list.append(pred_scored)

    return gold_spans_list, pred_scored_list


# ----------------------------
# Threshold evaluation
# ----------------------------
def eval_thresholds(gold_spans_list, pred_scored_list, thr_by_type, default_thr=0.0):
    gold_all = []
    pred_all = []
    for gold_spans, pred_scored in zip(gold_spans_list, pred_scored_list):
        kept = []
        for (s, e, t, sc) in pred_scored:
            thr = thr_by_type.get(t, default_thr)
            if sc >= thr:
                kept.append((s, e, t))
        gold_all.extend(gold_spans)
        pred_all.extend(kept)

    p, r, f1, tp, fp, fn = strict_prf(gold_all, pred_all)
    return (p, r, f1, tp, fp, fn), gold_all, pred_all


def tune_per_type_thresholds(
    gold_spans_list,
    pred_scored_list,
    types,
    grid,
    objective="f1",
    min_precision=0.0,
    base_threshold=0.0,
):
    """
    Simple coordinate ascent:
      - start with base_threshold for all types
      - for each type, pick the threshold that best improves objective
        while the global precision stays >= min_precision
      - iterate a few times

    objective:
      - "f1"  : maximize global F1
      - "recall": maximize global recall
    """
    thr_by_type = {t: base_threshold for t in types}

    def score(metrics):
        p, r, f1, *_ = metrics
        if p < min_precision:
            return -1e9
        if objective == "f1":
            return f1
        elif objective == "recall":
            return r
        else:
            raise ValueError("objective must be f1 or recall")

    best_metrics, _, _ = eval_thresholds(gold_spans_list, pred_scored_list, thr_by_type, base_threshold)
    best_score = score(best_metrics)

    # coordinate ascent
    for _iter in range(3):
        improved = False
        for t in types:
            best_t_thr = thr_by_type[t]
            best_t_score = best_score
            best_t_metrics = best_metrics

            for thr in grid:
                cand = dict(thr_by_type)
                cand[t] = float(thr)
                m, _, _ = eval_thresholds(gold_spans_list, pred_scored_list, cand, base_threshold)
                s = score(m)
                if s > best_t_score:
                    best_t_score = s
                    best_t_thr = float(thr)
                    best_t_metrics = m

            thr_by_type[t] = best_t_thr
            if best_t_score > best_score:
                best_score = best_t_score
                best_metrics = best_t_metrics
                improved = True

        if not improved:
            break

    return thr_by_type, best_metrics


def pretty_print(title, metrics, by_type=None, types=None):
    p, r, f1, tp, fp, fn = metrics
    print(f"\n=== {title} ===")
    print(f"P={p:.3f}  R={r:.3f}  F1={f1:.3f}   tp={tp} fp={fp} fn={fn}")
    if by_type and types:
        print("\nPer-type (strict):")
        for t in types:
            pp, rr, ff, ttp, tfp, tfn = by_type[t]
            print(f"{t:<10s}  P={pp:.3f} R={rr:.3f} F1={ff:.3f}  tp={ttp} fp={tfp} fn={tfn}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True, help="Path to HF dataset saved with load_from_disk.")
    ap.add_argument("--model_dir", required=True, help="Trained model directory.")
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--out_dir", required=True, help="Where to write thresholds_*.json")
    ap.add_argument("--grid_min", type=float, default=0.30)
    ap.add_argument("--grid_max", type=float, default=0.95)
    ap.add_argument("--grid_step", type=float, default=0.05)

    # preset constraints
    ap.add_argument("--auto_min_precision", type=float, default=0.65)
    ap.add_argument("--suggest_min_precision", type=float, default=0.30)

    # types list (must match your training)
    ap.add_argument(
        "--types",
        default="PERSON,PEOPLE,PLACE,ORG,BIBL,ARTEFACT,CONCEPT,OTHER",
        help="Comma-separated entity types."
    )

    args = ap.parse_args()

    dataset_path = str(Path(args.dataset_path).expanduser().resolve())
    model_dir = str(Path(args.model_dir).expanduser().resolve())
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    types = [t.strip() for t in args.types.split(",") if t.strip()]
    grid = [round(x, 4) for x in np.arange(args.grid_min, args.grid_max + 1e-9, args.grid_step)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("Loading dataset:", dataset_path)
    ds_dict = load_from_disk(dataset_path)
    ds = ds_dict[args.split]
    print(f"Split={args.split}  examples={len(ds)}")

    print("Loading model:", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)  # not strictly needed but useful sanity check
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)

    print("Caching scored spans on split (this can take a bit)...")
    gold_spans_list, pred_scored_list = cache_scored_spans(ds, model, device)
    print("Cached:", len(gold_spans_list))

    # ---- Baseline: no thresholding (thr=0 for all types)
    base_thr = 0.0
    thr_all_zero = {t: base_thr for t in types}
    base_metrics, gold_all, pred_all = eval_thresholds(gold_spans_list, pred_scored_list, thr_all_zero, base_thr)
    by_type = prf_by_type(gold_all, pred_all, types)
    pretty_print("Baseline (no filtering)", base_metrics, by_type, types)

    # ---- SUGGEST preset: maximize recall subject to minimum precision
    suggest_thr, suggest_metrics = tune_per_type_thresholds(
        gold_spans_list,
        pred_scored_list,
        types=types,
        grid=grid,
        objective="recall",
        min_precision=args.suggest_min_precision,
        base_threshold=0.0,
    )
    m_s, gold_s, pred_s = eval_thresholds(gold_spans_list, pred_scored_list, suggest_thr, 0.0)
    by_type_s = prf_by_type(gold_s, pred_s, types)
    pretty_print(f"SUGGEST tuned (maximize recall, min P={args.suggest_min_precision})", m_s, by_type_s, types)

    # ---- AUTO preset: maximize F1 subject to minimum precision
    auto_thr, auto_metrics = tune_per_type_thresholds(
        gold_spans_list,
        pred_scored_list,
        types=types,
        grid=grid,
        objective="f1",
        min_precision=args.auto_min_precision,
        base_threshold=0.0,
    )
    m_a, gold_a, pred_a = eval_thresholds(gold_spans_list, pred_scored_list, auto_thr, 0.0)
    by_type_a = prf_by_type(gold_a, pred_a, types)
    pretty_print(f"AUTO tuned (maximize F1, min P={args.auto_min_precision})", m_a, by_type_a, types)

    # ---- Save
    suggest_path = out_dir / "thresholds_suggest.json"
    auto_path = out_dir / "thresholds_auto.json"

    suggest_payload = {
        "mode": "suggest",
        "split": args.split,
        "min_precision": args.suggest_min_precision,
        "grid": {"min": args.grid_min, "max": args.grid_max, "step": args.grid_step},
        "thresholds_by_type": suggest_thr,
        "metrics": {"precision": m_s[0], "recall": m_s[1], "f1": m_s[2]},
    }
    auto_payload = {
        "mode": "auto",
        "split": args.split,
        "min_precision": args.auto_min_precision,
        "grid": {"min": args.grid_min, "max": args.grid_max, "step": args.grid_step},
        "thresholds_by_type": auto_thr,
        "metrics": {"precision": m_a[0], "recall": m_a[1], "f1": m_a[2]},
    }

    suggest_path.write_text(json.dumps(suggest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    auto_path.write_text(json.dumps(auto_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nWrote:")
    print(" ", suggest_path)
    print(" ", auto_path)


if __name__ == "__main__":
    main()
