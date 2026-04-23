"""
audit_entity_distribution.py

Scans all TEI XML files in a directory and reports entity type distribution
per file and per family (from the CSV). Helps decide how to create more
granular family assignments for balanced train/dev/test splits.

Usage:
    python audit_entity_distribution.py <root_dir> <csv_path> <out_dir>

Outputs:
    - audit_per_file.csv       : entity counts per XML file
    - audit_per_family.csv     : entity counts aggregated per family
    - audit_summary.txt        : console report saved to file
"""

import re
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

from lxml import etree
from lxml.etree import XMLSyntaxError

# ============================================================
# Config — must match your corpus builder settings
# ============================================================

ENTITIES_DTD_PATH = r"C:\Users\elena\Documents\WORK\USI\STIL PROJECT\ML\training_data\Zeichen.dtd"

RS_TYPE_TO_LABEL = {
    "artefact": "ARTEFACT",
    "artifact": "ARTEFACT",
    "term":     "CONCEPT",
    "concept":  "CONCEPT",
    "person":   "PERSON",
    "place":    "PLACE",
    "org":      "ORG",
    "people":   "CULTURE",
    "bibl":     "BIBL",
}

ALL_LABELS = ["ARTEFACT", "CONCEPT", "PERSON", "PLACE", "ORG", "CULTURE", "BIBL"]

# ============================================================
# XML utilities (minimal — no char map needed for counting)
# ============================================================

def build_entity_map_from_dtd(dtd_path):
    text = Path(dtd_path).read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(r'<!ENTITY\s+([^"\s]+)\s+"&#(\d+);"')
    return {name: chr(int(code)) for name, code in pat.findall(text)}


def expand_custom_entities(xml_text, entity_map):
    pat = re.compile(r"&([^\d#][^;\s]*);")
    def repl(m):
        return entity_map.get(m.group(1), m.group(0))
    return pat.sub(repl, xml_text)


def remove_xml_comments(xml_text):
    return re.sub(r"<!--.*?-->", "", xml_text, flags=re.DOTALL)


def sanitize_xml_text(xml_text, entity_map):
    xml_text = expand_custom_entities(xml_text, entity_map)
    def repl_unknown(m):
        name = m.group(1)
        return m.group(0) if name in entity_map else "&amp;" + name + ";"
    xml_text = re.sub(r"&([A-Za-z_][A-Za-z0-9._:-]*);", repl_unknown, xml_text)
    bare_amp = re.compile(r"&(?![A-Za-z_][A-Za-z0-9._:-]*;|#[0-9]+;|#x[0-9A-Fa-f]+;)")
    return bare_amp.sub("&amp;", xml_text)


def parse_xml_file(xml_path, entity_map):
    xml_text = Path(xml_path).read_text(encoding="utf-8", errors="ignore")
    xml_text = remove_xml_comments(xml_text)
    xml_text = sanitize_xml_text(xml_text, entity_map)
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, no_network=True)
    try:
        return etree.fromstring(xml_text.encode("utf-8"), parser=parser)
    except XMLSyntaxError as e:
        print(f"  [XML ERROR] {Path(xml_path).name}: {e}")
        return None


# ============================================================
# CSV loading
# ============================================================

def load_family_csv(csv_path):
    file_to_family = {}
    file_to_split  = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(1024)
        f.seek(0)
        delimiter = "\t" if "\t" in sample.split("\n")[0] else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            key = Path(row["filepath"].strip()).name.lower()
            file_to_family[key] = row["family_id"].strip()
            file_to_split[key]  = row["split"].strip().lower()
    return file_to_family, file_to_split


# ============================================================
# Entity counting
# ============================================================

def count_entities_in_file(xml_path, entity_map):
    """
    Count rs/@type occurrences in a single XML file, excluding teiHeader.
    Returns:
        counts        : Counter of label -> count
        unknown_types : Counter of unrecognised rs/@type values
        n_paragraphs  : number of <p> elements outside teiHeader
    """
    root = parse_xml_file(xml_path, entity_map)
    if root is None:
        return Counter(), Counter(), 0

    TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

    counts        = Counter()
    unknown_types = Counter()

    # Count entities only outside teiHeader
    rs_elems = root.xpath(
        "//*[local-name()='rs' "
        "and not(ancestor::*[local-name()='teiHeader'])]"
    )
    for rs in rs_elems:
        rs_type = (rs.get("type") or "").strip().lower()
        label = RS_TYPE_TO_LABEL.get(rs_type)
        if label:
            counts[label] += 1
        elif rs_type:
            unknown_types[rs_type] += 1

    p_elems = root.xpath(
        "//*[local-name()='p' "
        "and not(ancestor::*[local-name()='teiHeader'])]"
    )
    n_paragraphs = len(p_elems)

    return counts, unknown_types, n_paragraphs


# ============================================================
# Reporting
# ============================================================

def pct(part, total):
    return f"{100 * part / total:.1f}%" if total else "0.0%"


def format_table(rows, headers):
    """Format a list of dicts as a fixed-width text table."""
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row.get(h, ""))))

    sep   = "  ".join("-" * col_widths[h] for h in headers)
    head  = "  ".join(h.ljust(col_widths[h]) for h in headers)
    lines = [head, sep]
    for row in rows:
        lines.append("  ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers))
    return "\n".join(lines)


def run_audit(root_dir, csv_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entity_map = build_entity_map_from_dtd(ENTITIES_DTD_PATH)
    file_to_family, file_to_split = load_family_csv(csv_path)

    xml_files = sorted(Path(root_dir).rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files.")

    # Per-file results
    per_file    = []
    all_unknown = Counter()

    # Aggregations
    per_family  = defaultdict(lambda: Counter())
    per_split   = defaultdict(lambda: Counter())
    grand_total = Counter()

    family_splits = {}   # family_id -> split (for the summary)

    for xml_path in xml_files:
        fname       = xml_path.name
        fname_lower = fname.lower()
        family_id   = file_to_family.get(fname_lower, "UNLISTED")
        split       = file_to_split.get(fname_lower,  "UNLISTED")

        counts, unknown, n_para = count_entities_in_file(str(xml_path), entity_map)
        all_unknown += unknown

        total = sum(counts.values())
        row = {
            "file":       fname,
            "family_id":  family_id,
            "split":      split,
            "paragraphs": n_para,
            "total":      total,
        }
        for label in ALL_LABELS:
            row[label] = counts[label]
        per_file.append(row)

        per_family[family_id] += counts
        per_split[split]      += counts
        grand_total            += counts

        if family_id not in family_splits:
            family_splits[family_id] = split

    # ── Per-file CSV ──────────────────────────────────────────
    file_headers = ["file", "family_id", "split", "paragraphs", "total"] + ALL_LABELS
    with open(out_dir / "audit_per_file.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=file_headers)
        writer.writeheader()
        writer.writerows(per_file)
    print(f"Wrote audit_per_file.csv ({len(per_file)} files)")

    # ── Per-family CSV ────────────────────────────────────────
    family_rows = []
    for fam_id, counts in sorted(per_family.items()):
        total = sum(counts.values())
        row = {
            "family_id": fam_id,
            "split":     family_splits.get(fam_id, "?"),
            "total":     total,
        }
        for label in ALL_LABELS:
            row[label]           = counts[label]
            row[f"{label}_%"]    = pct(counts[label], total)
        family_rows.append(row)

    family_headers = (
        ["family_id", "split", "total"]
        + [col for label in ALL_LABELS for col in (label, f"{label}_%")]
    )
    with open(out_dir / "audit_per_family.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=family_headers)
        writer.writeheader()
        writer.writerows(family_rows)
    print(f"Wrote audit_per_family.csv ({len(family_rows)} families)")

    # ── Console / summary report ──────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("ENTITY DISTRIBUTION AUDIT")
    lines.append("=" * 70)

    # Grand total
    g_total = sum(grand_total.values())
    lines.append(f"\nTotal entities across all files: {g_total}")
    lines.append(f"Total XML files scanned:         {len(xml_files)}")
    lines.append(f"Files not in CSV (UNLISTED):     "
                 f"{sum(1 for r in per_file if r['family_id'] == 'UNLISTED')}")

    lines.append("\n--- Grand total by entity type ---")
    for label in ALL_LABELS:
        lines.append(f"  {label:10s}: {grand_total[label]:6d}  ({pct(grand_total[label], g_total):>6s})")

    # Per-split summary
    lines.append("\n--- Entity counts per split ---")
    split_headers = ["label"] + sorted(per_split.keys()) + ["total"]
    split_rows = []
    for label in ALL_LABELS:
        row = {"label": label}
        for s in sorted(per_split.keys()):
            row[s]      = per_split[s][label]
            row[f"{s}%"] = pct(per_split[s][label], sum(per_split[s].values()))
        row["total"] = grand_total[label]
        split_rows.append(row)

    # Simple formatted table for splits
    lines.append(f"\n  {'Label':<10}", )
    for s in sorted(per_split.keys()):
        lines[-1] += f"  {s:>8s}  {'%':>6s}"
    lines[-1] += f"  {'Total':>8s}"

    lines.append("  " + "-" * (10 + (8 + 8) * len(per_split) + 10))
    for row in split_rows:
        line = f"  {row['label']:<10}"
        for s in sorted(per_split.keys()):
            line += f"  {row[s]:>8d}  {pct(row[s], sum(per_split[s].values())):>6s}"
        line += f"  {row['total']:>8d}"
        lines.append(line)

    # Totals row
    line = f"  {'TOTAL':<10}"
    for s in sorted(per_split.keys()):
        s_total = sum(per_split[s].values())
        line += f"  {s_total:>8d}  {'100%':>6s}"
    line += f"  {g_total:>8d}"
    lines.append(line)

    # Per-family summary
    lines.append("\n--- Entity counts per family ---")
    fam_table_rows = []
    for fam_id, counts in sorted(per_family.items()):
        total = sum(counts.values())
        row   = {"family": fam_id, "split": family_splits.get(fam_id, "?"), "total": total}
        for label in ALL_LABELS:
            row[label] = f"{counts[label]} ({pct(counts[label], total)})"
        fam_table_rows.append(row)

    fam_headers = ["family", "split", "total"] + ALL_LABELS
    lines.append(format_table(fam_table_rows, fam_headers))

    # Unknown rs/@type values
    if all_unknown:
        lines.append("\n--- Unrecognised rs/@type values (not mapped to any label) ---")
        for t, n in all_unknown.most_common():
            lines.append(f"  '{t}': {n} occurrences")
    else:
        lines.append("\n--- No unrecognised rs/@type values found ---")

    # Files with highest entity density (top 20)
    lines.append("\n--- Top 20 files by total entity count ---")
    top_files = sorted(per_file, key=lambda r: r["total"], reverse=True)[:20]
    top_headers = ["file", "family_id", "split", "total"] + ALL_LABELS
    lines.append(format_table(top_files, top_headers))

    # Files with no entities (potential XML errors or empty pages)
    empty_files = [r for r in per_file if r["total"] == 0]
    lines.append(f"\n--- Files with 0 entities: {len(empty_files)} ---")
    if empty_files:
        for r in empty_files[:30]:  # cap at 30
            lines.append(f"  {r['file']}  (family={r['family_id']}, paras={r['paragraphs']})")
        if len(empty_files) > 30:
            lines.append(f"  ... and {len(empty_files) - 30} more (see audit_per_file.csv)")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    with open(out_dir / "audit_summary.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nWrote audit_summary.txt")

    # Also save machine-readable JSON
    json_report = {
        "grand_total":       dict(grand_total),
        "per_split":         {s: dict(c) for s, c in per_split.items()},
        "per_family":        {f: dict(c) for f, c in per_family.items()},
        "unknown_rs_types":  dict(all_unknown),
    }
    with open(out_dir / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    print(f"Wrote audit_report.json")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit entity distribution across TEI XML files."
    )
    parser.add_argument("root_dir", help="Root directory containing TEI XML files")
    parser.add_argument("csv_path", help="CSV file: filepath,family_id,split")
    parser.add_argument("out_dir",  help="Output directory for audit reports")

    args = parser.parse_args()
    run_audit(args.root_dir, args.csv_path, args.out_dir)
