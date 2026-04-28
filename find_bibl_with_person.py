"""
find_bibl_with_person.py

Scans all TEI XML files in a folder and extracts all <rs type="bibl"> elements
that contain a nested <rs type="person"> child element.

Outputs:
    - Console report with file, paragraph id, and text content
    - bibl_with_person.csv  with one row per found instance
    - bibl_with_person.txt  with full text context for manual review

Usage:
    python find_bibl_with_person.py <root_dir> <out_dir>
"""

import re
import csv
from pathlib import Path
from lxml import etree
from lxml.etree import XMLSyntaxError


# ============================================================
# XML parsing utilities (minimal, consistent with corpus builder)
# ============================================================

def remove_xml_comments(xml_text):
    cleaned = re.sub(r"<!--.*?-->", "", xml_text, flags=re.DOTALL)
    if "<!--" in cleaned:
        print("  WARNING: possible unclosed comment after removal")
    return cleaned


def sanitize_xml_text(xml_text):
    bare_amp = re.compile(r"&(?![A-Za-z_][A-Za-z0-9._:-]*;|#[0-9]+;|#x[0-9A-Fa-f]+;)")
    return bare_amp.sub("&amp;", xml_text)


def parse_xml_file(xml_path):
    try:
        xml_text = Path(xml_path).read_text(encoding="utf-8", errors="ignore")
        xml_text = remove_xml_comments(xml_text)
        xml_text = sanitize_xml_text(xml_text)
        parser   = etree.XMLParser(
            load_dtd=False, resolve_entities=False, no_network=True
        )
        return etree.fromstring(xml_text.encode("utf-8"), parser=parser)
    except XMLSyntaxError as e:
        print(f"  [XML ERROR] {Path(xml_path).name}: {e}")
        return None


# ============================================================
# Text extraction (strips all tags, preserves whitespace)
# ============================================================

def get_text_content(elem):
    """
    Extract all text content from an element, collapsing whitespace.
    """
    parts = []
    def recurse(node):
        if node.text:
            parts.append(node.text)
        for child in node:
            recurse(child)
            if child.tail:
                parts.append(child.tail)
    recurse(elem)
    return " ".join(" ".join(parts).split())  # collapse whitespace


def get_person_names(bibl_elem):
    """
    Extract text content of all nested <rs type="person"> elements.
    """
    person_elems = bibl_elem.xpath(
        ".//*[local-name()='rs' and @type='person']"
    )
    return [get_text_content(p) for p in person_elems]


def get_paragraph_id(elem):
    """
    Walk up the tree to find the enclosing <p> and return its xml:id.
    """
    XML_NS = "http://www.w3.org/XML/1998/namespace"
    node = elem
    while node is not None:
        tag = node.tag
        if isinstance(tag, str):
            local = tag.split("}", 1)[1] if "}" in tag else tag
            if local == "p":
                p_id = (
                    node.get(f"{{{XML_NS}}}id")
                    or node.get("xml:id")
                    or node.get("id")
                    or "unknown"
                )
                return p_id
        node = node.getparent()
    return "unknown"


# ============================================================
# Main scan
# ============================================================

def find_bibl_with_person(root_dir, out_dir):
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(Path(root_dir).rglob("*.xml"))
    print(f"Scanning {len(xml_files)} XML files...\n")

    results = []

    for xml_path in xml_files:
        root = parse_xml_file(str(xml_path))
        if root is None:
            continue

        # Find all <rs type="bibl"> outside teiHeader
        bibl_elems = root.xpath(
            "//*[local-name()='rs' "
            "and @type='bibl' "
            "and not(ancestor::*[local-name()='teiHeader']) "
            "and .//*[local-name()='rs' and @type='person']]"
        )

        for bibl in bibl_elems:
            bibl_text   = get_text_content(bibl)
            person_names = get_person_names(bibl)
            para_id      = get_paragraph_id(bibl)
            ref_attr     = bibl.get("ref", "")

            results.append({
                "file":         xml_path.name,
                "paragraph_id": para_id,
                "bibl_ref":     ref_attr,
                "bibl_text":    bibl_text,
                "person_names": " | ".join(person_names),
                "n_persons":    len(person_names),
            })

    # ── Console summary ────────────────────────────────────────
    print(f"Found {len(results)} <rs type='bibl'> elements containing "
          f"<rs type='person'> across {len(xml_files)} files.\n")

    # Group by file for summary
    from collections import Counter
    per_file = Counter(r["file"] for r in results)
    print("Files with most occurrences:")
    for fname, count in per_file.most_common(10):
        print(f"  {fname}: {count}")

    # ── CSV output ─────────────────────────────────────────────
    csv_path = out_dir / "bibl_with_person.csv"
    fieldnames = [
        "file", "paragraph_id", "bibl_ref",
        "bibl_text", "person_names", "n_persons"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {csv_path}  ({len(results)} rows)")

    # ── Plain text output for manual review ───────────────────
    txt_path = out_dir / "bibl_with_person.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"BIBL elements containing PERSON — {len(results)} instances\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"File:       {r['file']}\n")
            f.write(f"Paragraph:  {r['paragraph_id']}\n")
            f.write(f"Bibl ref:   {r['bibl_ref']}\n")
            f.write(f"Person(s):  {r['person_names']}\n")
            f.write(f"Full text:  {r['bibl_text']}\n")
            f.write("-" * 70 + "\n\n")
    print(f"Wrote {txt_path}")

    return results


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find <rs type='bibl'> elements containing nested "
                    "<rs type='person'> in TEI XML files."
    )
    parser.add_argument("root_dir", help="Root directory containing TEI XML files")
    parser.add_argument("out_dir",  help="Output directory for results")
    args = parser.parse_args()

    find_bibl_with_person(args.root_dir, args.out_dir)