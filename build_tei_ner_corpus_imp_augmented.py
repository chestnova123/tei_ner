import re
import json
import random
from pathlib import Path


from lxml import etree
from lxml.etree import XMLSyntaxError

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from collections import Counter

# version 3.0. augments data by extracting two versions of every paragraph, one with contents of the del tags and one without

# ============================================================
# 1. Global config
# ============================================================

SKIPPED_RS_TYPES = Counter()

MODEL_NAME = "xlm-roberta-base"  # or another HF model
MAX_LENGTH = 512  # truncation length
RANDOM_SEED = 42

MARK_DEL = True
MARK_ADD = True
MARK_HANDSHIFT = True
MARK_HI = True

HI_START, HI_END = " ⟦HI⟧ ", " ⟦/HI⟧ "
DEL_START, DEL_END = " ⟦DEL⟧ ", " ⟦/DEL⟧ "
ADD_START, ADD_END = " ⟦ADD⟧ ", " ⟦/ADD⟧ "
LAT_MARK, KUR_MARK = " ⟦LAT⟧ ", " ⟦KUR⟧ "

# Final entity types
TYPES = [
    "PERSON",
    "PEOPLE",
    "PLACE",
    "ORG",
    "BIBL",
    "ARTEFACT",
    "CONCEPT",
]

# BILOU label set
LABELS = ["O"]
for t in TYPES:
    LABELS.extend([f"B-{t}", f"I-{t}", f"L-{t}", f"U-{t}"])

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# expansion of xml entitites

# Path to Zeichen.dtd
ENTITIES_DTD_PATH = r"C:\Users\elena\Documents\GitHub\semper-tei_new\scripts\R_transformer_based_entity_prediction\extraction_test\test_in\Zeichen.dtd"


def build_entity_map_from_dtd(dtd_path):
    """
    Parse a DTD file like Zeichen.dtd and build a mapping
    from entity name -> Unicode character.
    Assumes lines like: <!ENTITY semikol "&#59;" >
    """
    text = Path(dtd_path).read_text(encoding="utf-8", errors="ignore")

    # capture: entity name, numeric code
    pat = re.compile(r'<!ENTITY\s+([^"\s]+)\s+"&#(\d+);"')
    entity_map = {}
    for name, code in pat.findall(text):
        entity_map[name] = chr(int(code))
    return entity_map


ENTITY_MAP = build_entity_map_from_dtd(ENTITIES_DTD_PATH)


def expand_custom_entities(xml_text, entity_map):
    """
    Replace &name; with the corresponding character if name is in entity_map.
    Leave numeric entities (&#...;) and unknown names untouched.

    Matches &NAME; where NAME does *not* start with a digit or '#'.
    """
    pat = re.compile(r"&([^\d#][^;\s]*);")

    def repl(m):
        name = m.group(1)
        return entity_map.get(name, m.group(0))

    return pat.sub(repl, xml_text)


# sanitize text
def escape_lt_in_any_quotes(xml_text):
    """
    Replace any '<' inside either single or double quoted strings with '&lt;'.
    Handles multi-line quoted strings.
    """

    def repl(m):
        quote = m.group(1)  # ' or "
        content = m.group(2)  # inside quotes, possibly multiline
        content_fixed = content.replace("<", "&lt;")
        return f"{quote}{content_fixed}{quote}"

    # Supports:
    #   " ... possibly multiline ... "
    #   ' ... possibly multiline ... '
    pattern = re.compile(r'(["\'])(.*?)\1', re.DOTALL)
    return pattern.sub(repl, xml_text)


def sanitize_xml_text(xml_text, entity_map):
    """
    1) Expand known custom entities using entity_map (e.g. &semikol; -> ';').
    2) Escape remaining &name; sequences that are NOT in entity_map.
    3) Escape any bare '&' that does not start a valid entity/char ref.
    4) Escape any unescaped '<' that appears inside double quotes
       (illegal in attribute values).
    """

    # 1) Expand known entities like &semikol;
    xml_text = expand_custom_entities(xml_text, entity_map)

    # 2) Escape &name; that are not known entities -> &amp;name;
    def repl_unknown_entity(m):
        name = m.group(1)
        if name in entity_map:
            # already expanded or intended; leave as is
            return m.group(0)
        else:
            # treat as literal text
            return "&amp;" + name + ";"

    # &Name; where Name looks like an XML Name
    xml_text = re.sub(r"&([A-Za-z_][A-Za-z0-9._:-]*);", repl_unknown_entity, xml_text)

    # 3) Escape any bare '&' that doesn't start a valid entity or char ref
    bare_amp_pattern = re.compile(
        r"&(?![A-Za-z_][A-Za-z0-9._:-]*;|#[0-9]+;|#x[0-9A-Fa-f]+;)"
    )
    xml_text = bare_amp_pattern.sub("&amp;", xml_text)

    # 4) escape '<' inside ANY quoted string, multiline-aware
    xml_text = escape_lt_in_any_quotes(xml_text)

    return xml_text


# ============================================================
# 2. Normalization of <p> and char map
# ============================================================


from lxml import etree


def normalize_paragraph(
    p_elem,
    *,
    mark_del=True,
    mark_add=True,
    mark_handshift=True,
    mark_hi=True,
    keep_del_content=True,
):
    text_chars, char_map = [], []

    def append_char(ch, node, kind, offset):
        text_chars.append(ch)
        char_map.append((node, kind, offset))

    def append_literal(s, node, kind):
        # literal markers: no real XML offsets => use negative offsets
        for k, ch in enumerate(s):
            append_char(ch, node, kind, -1 - k)

    def recurse(node):
        if node.text:
            for i, ch in enumerate(node.text):
                append_char(ch, node, "text", i)

        for child in list(node):
            if not isinstance(child, etree._Element):
                continue

            tag = child.tag
            if isinstance(tag, str) and "}" in tag:
                tag = tag.split("}", 1)[1]

            if tag == "lb":
                lb_type = child.get("type")
                if lb_type == "hyph":
                    if text_chars and text_chars[-1] == "-":
                        text_chars.pop()
                        char_map.pop()
                else:
                    append_char(" ", child, "lb", 0)

            elif tag == "metamark":
                pass

            elif tag == "del":
                # Version A: optionally mark, and optionally keep deleted content
                if mark_del:
                    append_literal(DEL_START, child, "marker")
                if keep_del_content:
                    recurse(child)  # keep deleted text
                # else: skip recursion => deletion content removed
                if mark_del:
                    append_literal(DEL_END, child, "marker")

            elif tag == "add":
                if mark_add:
                    append_literal(ADD_START, child, "marker")
                recurse(child)
                if mark_add:
                    append_literal(ADD_END, child, "marker")

            elif tag == "handShift":
                if mark_handshift:
                    scr = (child.get("script") or "").lower()
                    if "latein" in scr or "latin" in scr:
                        append_literal(LAT_MARK, child, "marker")
                    elif "kurrent" in scr:
                        append_literal(KUR_MARK, child, "marker")
                recurse(child)

            elif tag == "hi":
                if mark_hi:
                    append_literal(HI_START, child, "marker")
                recurse(child)
                if mark_hi:
                    append_literal(HI_END, child, "marker")

            else:
                recurse(child)

            if child.tail:
                for i, ch in enumerate(child.tail):
                    append_char(ch, child, "tail", i)

    recurse(p_elem)
    return "".join(text_chars), char_map


def build_inverse_char_map(char_map):
    """
    Build a mapping (node, kind, offset) -> char_index.
    """
    idx_map = {}
    for char_index, (node, kind, offset) in enumerate(char_map):
        idx_map[(node, kind, offset)] = char_index
    return idx_map


# ============================================================
# 3. Entity span extraction from <rs>
# ============================================================


def map_rs_type_to_label(rs_type):
    """
    Map TEI rs/@type to NER label types (without BILOU prefix).
    """
    if not rs_type:
        SKIPPED_RS_TYPES["<EMPTY>"] += 1
        return None
    
    mapping = {
        "person": "PERSON",
        "people": "PEOPLE",
        "place": "PLACE",
        "organisation": "ORG",
        "org": "ORG",
        "bibl": "BIBL",
        "artefact": "ARTEFACT",
        "artifact": "ARTEFACT",
        "concept": "CONCEPT",
        "term": "CONCEPT",
    }
    
    return mapping.get(rs_type)


def get_span_for_rs(rs_elem, idx_map):
    """
    Given an <rs> element and the inverse char map, return (start, end)
    character offsets of the rs content in the normalized paragraph text.
    """
    start = None
    end = None

    def mark_char(node, kind, i):
        nonlocal start, end
        key = (node, kind, i)
        if key in idx_map:
            idx = idx_map[key]
            if start is None or idx < start:
                start = idx
            if end is None or idx >= end:
                end = idx + 1  # end is exclusive

    def recurse(node):
        if node.text:
            for i, _ in enumerate(node.text):
                mark_char(node, "text", i)

        for child in node:
            recurse(child)
            if child.tail:
                for i, _ in enumerate(child.tail):
                    mark_char(child, "tail", i)

    recurse(rs_elem)
    return start, end


def extract_entities_from_paragraph(p_elem, text, char_map):
    """
    Given a TEI <p>, its normalized text and char_map, extract gold entities
    from <rs> tags as char spans.

    Returns
    -------
    entities : list of dict
        Each dict has {"start", "end", "label"}.
        label is the TYPE (e.g. "PERSON", "PLACE", "BIBL", ...).
    """
    idx_map = build_inverse_char_map(char_map)
    entities = []

    for rs in p_elem.xpath(".//tei:rs", namespaces=TEI_NS):
        rs_type = rs.get("type")
        label = map_rs_type_to_label(rs_type)
        if label is None:
            SKIPPED_RS_TYPES[rs_type] += 1
        start, end = get_span_for_rs(rs, idx_map)
        if label is None:
            continue
        if start is not None and end is not None and start < end:
            entities.append({"start": start, "end": end, "label": label})
    entities.sort(key=lambda e: e["start"])
    return entities


# ============================================================
# 4. Char-spans to BILOU token labels
# ============================================================


def spans_to_bilou_labels(text, entities, tokenizer, label2id, max_length=512):
    """
    Convert char-based entity spans into BILOU token labels.

    Parameters
    ----------
    text : str
        Normalized paragraph text.
    entities : list of dict
        Each dict: {"start": int, "end": int, "label": str}
        where label is the TYPE (e.g. "PERSON", "PLACE", ...), no BILOU prefix.
    tokenizer : HF tokenizer
    label2id : dict
        Mapping from label string to integer id (e.g. "B-PERSON" -> 1).
    max_length : int
        Max sequence length for tokenizer (with truncation).

    Returns
    -------
    encoding : dict
        Tokenized input plus "labels" field with label ids per token.
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = encoding["offset_mapping"]

    # Start with all tokens as "O"
    token_tags = ["O"] * len(offsets)

    # Assign BILOU tags for each entity
    def assign_entity(entity):
        start, end, ent_type = entity["start"], entity["end"], entity["label"]

        token_indices = []
        for i, (tok_start, tok_end) in enumerate(offsets):
            # (0,0) for special tokens like [CLS], [SEP]
            if tok_start == tok_end == 0:
                continue
            # check overlap
            if tok_end <= start or tok_start >= end:
                continue
            token_indices.append(i)

        if not token_indices:
            return  # entity truncated or misaligned

        token_indices.sort()
        n = len(token_indices)

        if n == 1:
            token_tags[token_indices[0]] = f"U-{ent_type}"
        else:
            token_tags[token_indices[0]] = f"B-{ent_type}"
            for ti in token_indices[1:-1]:
                token_tags[ti] = f"I-{ent_type}"
            token_tags[token_indices[-1]] = f"L-{ent_type}"

    for ent in entities:
        assign_entity(ent)

    # Convert tags to label IDs, ignoring special tokens in the loss
    labels_ids = []
    for (tok_start, tok_end), tag in zip(offsets, token_tags):
        if tok_start == tok_end == 0:
            labels_ids.append(-100)
        else:
            labels_ids.append(label2id[tag])

    encoding["labels"] = labels_ids
    # Optionally drop offset_mapping if you don’t need it later
    # del encoding["offset_mapping"]
    return encoding


# ============================================================
# 5. Walking the TEI corpus and collecting examples
# ============================================================


def find_xml_files(root_dir):
    """
    Recursively find all .xml files under root_dir.
    """
    root = Path(root_dir)
    return sorted(str(p) for p in root.rglob("*.xml"))


def process_document(xml_path, doc_id):
    examples = []

    # 1) Read raw XML file as text
    xml_text = Path(xml_path).read_text(encoding="utf-8", errors="ignore")

    # 2) Sanitize entities and ampersands
    xml_text = sanitize_xml_text(xml_text, ENTITY_MAP)

    # 3) Parse the resulting XML string with a parser that ignores DTD/network
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, no_network=True)
    try:
        root = etree.fromstring(xml_text.encode("utf-8"), parser=parser)
    except XMLSyntaxError as e:
        print(f"[XML ERROR] Skipping file: {xml_path}")
        print(f"  Reason: {e}")
        return []

    # 4) Find paragraphs (namespace-agnostic)
    p_elems = root.xpath("//*[local-name()='p' and not(ancestor::*[local-name()='teiHeader'])]")

    for p_idx, p_elem in enumerate(p_elems):

        variants = [
            # Version A: mark boundaries + keep deletions
            {
                "variant": "marked_keep_del",
                "norm_kwargs": dict(
                    mark_del=MARK_DEL,
                    mark_add=MARK_ADD,
                    mark_handshift=MARK_HANDSHIFT,
                    mark_hi=MARK_HI,
                    keep_del_content=True,
                ),
            },
            # Version B: plain text + drop deletions + no markers
            {
                "variant": "plain_drop_del",
                "norm_kwargs": dict(
                    mark_del=False,
                    mark_add=False,
                    mark_handshift=False,
                    mark_hi=False,
                    keep_del_content=False,
                ),
            },
        ]

        for v in variants:
            text, char_map = normalize_paragraph(p_elem, **v["norm_kwargs"])
            if not text.strip():
                continue

            entities = extract_entities_from_paragraph(p_elem, text, char_map)
            encoding = spans_to_bilou_labels(
                text,
                entities,
                tokenizer,
                label2id,
                max_length=MAX_LENGTH,
            )

            example = {
                "text": text,
                "entities": entities,
                "doc_id": doc_id,
                "p_index": p_idx,
                "variant": v["variant"],  # <--- add this so you can debug/filter
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": encoding["labels"],
            }
            examples.append(example)
    return examples


def load_corpus(root_dir):
    """
    Walk all XML files under root_dir and collect examples.

    Returns a list of example dicts.
    """
    xml_files = find_xml_files(root_dir)
    print(f"Found {len(xml_files)} XML files.")
    all_examples = []
    for doc_id, path in enumerate(xml_files):
        doc_examples = process_document(path, doc_id)
        all_examples.extend(doc_examples)
    return all_examples


# ============================================================
# 6. Train/dev/test split (by document)
# ============================================================


def split_by_document(examples, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15):
    """
    Split examples by doc_id to avoid leakage between train/dev/test.
    Ratios must sum to 1.0 (approximately).
    """
    random.seed(RANDOM_SEED)
    doc_ids = sorted({ex["doc_id"] for ex in examples})
    random.shuffle(doc_ids)

    n_docs = len(doc_ids)
    n_train = int(n_docs * train_ratio)
    n_dev = int(n_docs * dev_ratio)
    # rest go to test

    train_docs = set(doc_ids[:n_train])
    dev_docs = set(doc_ids[n_train : n_train + n_dev])
    test_docs = set(doc_ids[n_train + n_dev :])

    train_examples = [ex for ex in examples if ex["doc_id"] in train_docs]
    dev_examples = [ex for ex in examples if ex["doc_id"] in dev_docs]
    test_examples = [ex for ex in examples if ex["doc_id"] in test_docs]

    return train_examples, dev_examples, test_examples


# ============================================================
# 7. Convert to HF Dataset and write JSONL
# ============================================================


def examples_to_dataset(examples):
    """
    Convert a list of examples into a Hugging Face Dataset.
    """
    # datasets expects columns as lists
    data = {
        "text": [ex["text"] for ex in examples],
        "entities": [ex["entities"] for ex in examples],
        "doc_id": [ex["doc_id"] for ex in examples],
        "p_index": [ex["p_index"] for ex in examples],
        "variant": [ex.get("variant") for ex in examples],
        "input_ids": [ex["input_ids"] for ex in examples],
        "attention_mask": [ex["attention_mask"] for ex in examples],
        "labels": [ex["labels"] for ex in examples],
    }
    return Dataset.from_dict(data)


def write_jsonl(examples, path):
    """
    Write examples to a JSONL file.
    """
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def build_and_save_corpus(root_dir, out_dir):
    """
    Full pipeline:
      - load corpus from root_dir
      - split into train/dev/test
      - build DatasetDict
      - save HF dataset and JSONL files
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {root_dir}...")
    examples = load_corpus(root_dir)
    print(f"Total examples: {len(examples)}")

    train_ex, dev_ex, test_ex = split_by_document(examples)
    print(f"Train: {len(train_ex)}, Dev: {len(dev_ex)}, Test: {len(test_ex)}")

    # Save JSONL
    write_jsonl(train_ex, out_dir / "train.jsonl")
    write_jsonl(dev_ex, out_dir / "dev.jsonl")
    write_jsonl(test_ex, out_dir / "test.jsonl")
    print(f"Wrote JSONL files to {out_dir}")

    # Build DatasetDict
    train_ds = examples_to_dataset(train_ex)
    dev_ds = examples_to_dataset(dev_ex)
    test_ds = examples_to_dataset(test_ex)

    ds_dict = DatasetDict(
        {
            "train": train_ds,
            "validation": dev_ds,
            "test": test_ds,
        }
    )

    # Save HF dataset to disk
    ds_dict.save_to_disk(str(out_dir / "hf_dataset"))
    print(f"Saved HF Dataset to {out_dir / 'hf_dataset'}")


# ============================================================
# 8. CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build TEI NER corpus for BILOU NER training."
    )
    parser.add_argument("root_dir", help="Root directory containing TEI XML files")
    parser.add_argument("out_dir", help="Output directory for JSONL and HF dataset")

    args = parser.parse_args()

    build_and_save_corpus(args.root_dir, args.out_dir)
    print("Top skipped rs/@type values:")
    for k, v in SKIPPED_RS_TYPES.most_common(25):
        print(f"{k:20s} {v}")