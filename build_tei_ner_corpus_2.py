import re
import json
import random
from pathlib import Path


from lxml import etree
from lxml.etree import XMLSyntaxError

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

# version 2.0. Adapts the extraction process to keep structural information in the extracting text by marking edits.

# ============================================================
# 1. Global config
# ============================================================

LB_HYPH = "⟦LB_HYPH⟧"
LB_NORM = "⟦LB⟧"

# chars to treat as hyphen at line breaks
HYPH_SET = {"-", "\u2010", "\u2011", "\u2212", "\u00AD"}  # - ‐ - − soft hyphen
DASH_SET = {"\u2013"}  # – (optional)
JOIN_SET = HYPH_SET | DASH_SET

def _is_word_char(ch: str) -> bool:
    # matches letters/digits/underscore; good enough for joining
    return ch.isalnum() or ch == "_"

MODEL_NAME = r"C:\Users\elena\Documents\WORK\USI\STIL PROJECT\ML\dapt\model_rxlm_large_dapt"  # or another HF model
MAX_LENGTH = 512  # truncation length
RANDOM_SEED = 42
CHUNK_LEN = 480        # model input length per chunk (<=512)
CHUNK_STRIDE = 128

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
    "OTHER",
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

def postprocess_hyphenation(text_chars, char_map):
    """
    Operates on the character stream + char_map, removing hyphenation across LB_HYPH
    and converting LB_NORM to a space, while keeping char_map aligned.
    """

    # 1) remove soft hyphen chars everywhere (optional but usually correct)
    new_chars = []
    new_map = []
    for ch, m in zip(text_chars, char_map):
        if ch == "\u00AD":
            continue
        new_chars.append(ch)
        new_map.append(m)

    text_chars, char_map = new_chars, new_map

    # Helper: match sentinel at a position
    def match_at(i, needle):
        if i + len(needle) > len(text_chars):
            return False
        return "".join(text_chars[i:i+len(needle)]) == needle

    # 2) main scan
    out_chars = []
    out_map = []

    i = 0
    while i < len(text_chars):
        # Replace normal LB marker with a single space
        if match_at(i, LB_NORM):
            out_chars.append(" ")
            # Map this space to the first marker char's map entry (good enough)
            out_map.append(char_map[i])
            i += len(LB_NORM)
            continue

        if match_at(i, LB_HYPH):
            left_ok = len(out_chars) > 0 and _is_word_char(out_chars[-1])
            j = i + len(LB_HYPH)
            while j < len(text_chars) and text_chars[j] == " ":
                j += 1
            right_ok = j < len(text_chars) and _is_word_char(text_chars[j])
            if left_ok and right_ok:
                i = j
                continue
            i += len(LB_HYPH)
            continue
          
        # Hyphenation join:
        # If we see a JOIN_SET char followed (optionally with spaces) by LB_HYPH,
        # remove the join char + LB_HYPH and surrounding spaces to join words.
        #
        # We look for pattern:
        #   <wordchar> [spaces] <hyphen/dash> [spaces] LB_HYPH [spaces] <wordchar>
        #
        # Since we are scanning left-to-right, we'll detect when we're at a hyphen/dash
        # and see if LB_HYPH comes next.
        if text_chars[i] in JOIN_SET:
            # Look ahead skipping spaces
            j = i + 1
            while j < len(text_chars) and text_chars[j] == " ":
                j += 1

            if match_at(j, LB_HYPH):
                k = j + len(LB_HYPH)
                while k < len(text_chars) and text_chars[k] == " ":
                    k += 1

                # Check word chars on both sides
                left_ok = len(out_chars) > 0 and _is_word_char(out_chars[-1])
                right_ok = k < len(text_chars) and _is_word_char(text_chars[k])

                if left_ok and right_ok:
                    # Drop the hyphen/dash and the LB_HYPH marker completely
                    i = k
                    continue

        # Also handle case where hyphen is BEFORE spaces already in output
        # (e.g. "...- ⟦LB_HYPH⟧ ..."): the above handles it because hyphen is current char.

        # Drop standalone LB_HYPH if it survived without a join char (leave as nothing)
        if match_at(i, LB_HYPH):
            i += len(LB_HYPH)
            continue

        # Otherwise, copy char through
        out_chars.append(text_chars[i])
        out_map.append(char_map[i])
        i += 1

    # 3) collapse multiple spaces (keep char_map aligned)
    final_chars = []
    final_map = []
    prev_space = False
    for ch, m in zip(out_chars, out_map):
        if ch == " ":
            if prev_space:
                continue
            prev_space = True
        else:
            prev_space = False
        final_chars.append(ch)
        final_map.append(m)

    # strip leading/trailing space (again keep map aligned)
    while final_chars and final_chars[0] == " ":
        final_chars.pop(0); final_map.pop(0)
    while final_chars and final_chars[-1] == " ":
        final_chars.pop(); final_map.pop()

    return "".join(final_chars), final_map

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


def normalize_paragraph(p_elem):
    text_chars, char_map = [], []

    def append_char(ch, node, kind, offset):
        text_chars.append(ch)
        char_map.append((node, kind, offset))

    def append_literal(s, node, kind):
        # literal markers: they don’t correspond to an XML char offset,
        # but we still need them in the text stream.
        # Use offset=-1 so they can’t be confused with real node text offsets.
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
                lb_type = (child.get("type") or "").lower()
                if lb_type == "hyph":
                    append_literal(" " + LB_HYPH + " ", child, "marker")
                else:
                    append_literal(" " + LB_NORM + " ", child, "marker")

            elif tag == "metamark":
                # usually safest to skip
                pass

            elif tag == "del":
                if MARK_DEL:
                    append_literal(DEL_START, child, "marker")
                recurse(child)  # keep the deleted text
                if MARK_DEL:
                    append_literal(DEL_END, child, "marker")

            elif tag == "add":
                if MARK_ADD:
                    append_literal(ADD_START, child, "marker")
                recurse(child)
                if MARK_ADD:
                    append_literal(ADD_END, child, "marker")

            elif tag == "handShift":
                if MARK_HANDSHIFT:
                    scr = (child.get("script") or "").lower()
                    if "latein" in scr or "latin" in scr:
                        append_literal(LAT_MARK, child, "marker")
                    elif "kurrent" in scr:
                        append_literal(KUR_MARK, child, "marker")
                # handShift often has no useful text; still recurse in case it does
                recurse(child)

            elif tag == "hi":
                if MARK_HI:
                    append_literal(HI_START, child, "marker")
                recurse(child)
                if MARK_HI:
                    append_literal(HI_END, child, "marker")

            else:
                recurse(child)

            if child.tail:
                for i, ch in enumerate(child.tail):
                    append_char(ch, child, "tail", i)

    recurse(p_elem)
    text, cmap = postprocess_hyphenation(text_chars, char_map)
    return text, cmap

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
    # Unknown or unlisted types fall back to OTHER
    return mapping.get(rs_type, "OTHER")


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
        rs_type = (rs.get("type") or "").strip().lower()
        label = map_rs_type_to_label(rs_type)
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


def spans_to_bilou_chunked(text, entities, tokenizer, label2id, chunk_len=480, stride=128):
    """
    Tokenize WITHOUT truncation, then create overlapping windows (chunks).
    Returns a list of dicts, each containing input_ids/attention_mask/labels
    for one chunk.
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=True,
    )

    input_ids_all = enc["input_ids"]
    attn_all = enc["attention_mask"]
    offsets_all = enc["offset_mapping"]

    # We'll build windows over the *full* tokenized sequence.
    # Keep special tokens in place; offsets for special tokens are (0,0).
    n = len(input_ids_all)

    # Helper: build token-level BILOU tags for a given offset list
    def bilou_for_offsets(offsets):
        token_tags = ["O"] * len(offsets)

        for ent in entities:
            start, end, ent_type = ent["start"], ent["end"], ent["label"]

            token_indices = []
            for i, (ts, te) in enumerate(offsets):
                if ts == te == 0:
                    continue
                if te <= start or ts >= end:
                    continue
                token_indices.append(i)

            if not token_indices:
                continue

            token_indices.sort()
            if len(token_indices) == 1:
                token_tags[token_indices[0]] = f"U-{ent_type}"
            else:
                token_tags[token_indices[0]] = f"B-{ent_type}"
                for ti in token_indices[1:-1]:
                    token_tags[ti] = f"I-{ent_type}"
                token_tags[token_indices[-1]] = f"L-{ent_type}"

        labels = []
        for (ts, te), tag in zip(offsets, token_tags):
            if ts == te == 0:
                labels.append(-100)
            else:
                labels.append(label2id[tag])
        return labels

    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_len, n)

        chunk_input_ids = input_ids_all[start:end]
        chunk_attn = attn_all[start:end]
        chunk_offsets = offsets_all[start:end]

        # Compute labels for this chunk
        chunk_labels = bilou_for_offsets(chunk_offsets)

        # Pad to chunk_len so trainer batches cleanly (optional but convenient)
        pad_id = tokenizer.pad_token_id
        pad_len = chunk_len - len(chunk_input_ids)
        if pad_len > 0:
            chunk_input_ids = chunk_input_ids + [pad_id] * pad_len
            chunk_attn = chunk_attn + [0] * pad_len
            chunk_labels = chunk_labels + [-100] * pad_len

        chunks.append(
            {
                "input_ids": chunk_input_ids,
                "attention_mask": chunk_attn,
                "labels": chunk_labels,
                # keep for debugging if you want:
                # "offset_mapping": chunk_offsets,
                "chunk_start": start,
                "chunk_end": end,
            }
        )

        if end == n:
            break
        # overlap
        start = end - stride

    return chunks

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
        text, char_map = normalize_paragraph(p_elem)
        if not text.strip():
            continue

        entities = extract_entities_from_paragraph(p_elem, text, char_map)
        chunk_encodings = spans_to_bilou_chunked(
            text,
            entities,
            tokenizer,
            label2id,
            chunk_len=CHUNK_LEN,
            stride=CHUNK_STRIDE,
        )

        for c_idx, encoding in enumerate(chunk_encodings):
            example = {
                "text": text,
                "entities": entities,
                "doc_id": doc_id,
                "p_index": p_idx,
                "chunk_index": c_idx,
                "chunk_start": encoding["chunk_start"],
                "chunk_end": encoding["chunk_end"],
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
