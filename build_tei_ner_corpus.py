import re
import json
import random
from pathlib import Path
import math

from lxml import etree
from lxml.etree import XMLSyntaxError

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from collections import Counter, defaultdict

# version 2.0. Adapts the extraction process to keep structural information in the extracting text by marking edits.

# ============================================================
# 1. Global config
# ============================================================

QUOTA_VAL = {"BIBL": 150, "ORG": 60}
QUOTA_TEST = {"BIBL": 150, "ORG": 60}
GROUP_BLOCK_SIZE = 4
LB_HYPH = "⟦LB_HYPH⟧"
LB_NORM = "⟦LB⟧"

# chars to treat as hyphen at line breaks
HYPH_SET = {"-", "\u2010", "\u2011", "\u2212", "\u00AD"}  # - ‐ - − soft hyphen
DASH_SET = {"\u2013"}  # – (optional)
JOIN_SET = HYPH_SET | DASH_SET

def _is_word_char(ch: str) -> bool:
    # matches letters/digits/underscore; good enough for joining
    return ch.isalnum() or ch == "_"

MODEL_NAME = r"C:\Users\elena\Documents\WORK\USI\STIL PROJECT\ML\dapt_model_bio"  # or another HF model
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

# BIO label set
LABELS = ["O"]
for t in TYPES:
    LABELS.extend([f"B-{t}", f"I-{t}"])

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# expansion of xml entitites

# Path to Zeichen.dtd
ENTITIES_DTD_PATH = r"C:\Users\elena\Documents\GitHub\semper-tei_new\scripts\R_transformer_based_entity_prediction\extraction_test\test_in\Zeichen.dtd"

# Max allowed entity span length in TOKENS (per chunk). Spans longer than this are dropped to O.
MAX_SPAN_TOKENS = {
    "BIBL": 40,
    "PLACE": 10,
    "ARTEFACT": 10,
}

# Logging for capped/dropped spans
CAP_LOG_SPANS = Counter()   # how many spans dropped per type
CAP_LOG_TOKENS = Counter()  # how many tokens dropped per type

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


def spans_to_bio_chunked(text, entities, tokenizer, label2id, chunk_len=480, stride=128):
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

    # Helper: build token-level BIO tags for a given offset list
    def bio_for_offsets(offsets):
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

            ent_type = ent_type.upper()
            cap = MAX_SPAN_TOKENS.get(ent_type)

            # If span too long, drop ENTIRE span to O (i.e., do not assign any labels)
            if cap is not None and len(token_indices) > cap:
                CAP_LOG_SPANS[ent_type] += 1
                CAP_LOG_TOKENS[ent_type] += len(token_indices)
                continue

            # Otherwise normal BIO tagging
            token_tags[token_indices[0]] = f"B-{ent_type}"
            for ti in token_indices[1:]:
                token_tags[ti] = f"I-{ent_type}"

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
        chunk_labels = bio_for_offsets(chunk_offsets)

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
        chunk_encodings = spans_to_bio_chunked(
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
                "group_id": f"{doc_id}::{p_idx}::{c_idx // GROUP_BLOCK_SIZE}",
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
# 6. Train/dev/test split (by document, stratified)
# ============================================================
def count_entity_spans_in_labels(label_ids):
    """
    Count entity spans by type in BIO labels (label IDs).
    Returns Counter like {"BIBL": 3, "ORG": 1, ...}
    """
    counts = Counter()
    prev_type = None
    prev_is_entity = False

    for lid in label_ids:
        if lid == -100:
            prev_type = None
            prev_is_entity = False
            continue

        lab = id2label[int(lid)]
        if lab == "O":
            prev_type = None
            prev_is_entity = False
            continue

        # BIO: B-X starts a new span; I-X continues if same type, else treat as new span
        prefix, typ = lab.split("-", 1)
        if prefix == "B":
            counts[typ] += 1
            prev_type = typ
            prev_is_entity = True
        else:  # "I"
            if not prev_is_entity or prev_type != typ:
                counts[typ] += 1  # broken I- sequence → new span
            prev_type = typ
            prev_is_entity = True

    return counts

def split_by_group_stratified(
    examples,
    train_ratio=0.7,
    dev_ratio=0.15,
    test_ratio=0.15,
    group_field="group_id",
):
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6
    random.seed(RANDOM_SEED)

    # Aggregate label counts per group
    group_counts = defaultdict(Counter)
    group_tokens = Counter()
    group_labels_flat = defaultdict(list)

    for ex in examples:
        g = ex[group_field]
        labs = ex["labels"]

        # for BIO span counting
        group_labels_flat[g].extend(labs)

        # for token label distribution (exclude -100)
        for lab in labs:
            if lab == -100:
                continue
            group_counts[g][lab] += 1
            group_tokens[g] += 1

    groups = list(group_counts.keys())
    if len(groups) < 3:
        random.shuffle(groups)
        train_g = set(groups)
        return [ex for ex in examples if ex[group_field] in train_g], [], []

    # BIO span counts per group (safe default)
    group_span_counts = {}
    for g in groups:
        group_span_counts[g] = count_entity_spans_in_labels(group_labels_flat.get(g, []))
    
    def dist(cnt, total):
        if total <= 0:
            return {}
        return {k: v / total for k, v in cnt.items()}
    global_counts = Counter()
    global_total=0
   
    target = dist(global_counts, global_total)

    budgets = {
        "train": int(global_total * train_ratio),
        "validation": int(global_total * dev_ratio),
        "test": int(global_total * test_ratio),
    }

    splits = {
        "train": {"groups": set(), "counts": Counter(), "tokens": 0},
        "validation": {"groups": set(), "counts": Counter(), "tokens": 0},
        "test": {"groups": set(), "counts": Counter(), "tokens": 0},
    }
    
    def l1(cnt, total):
        if total <= 0:
            return 1e9
        p = dist(cnt, total)
        keys = set(target) | set(p)
        return sum(abs(p.get(k, 0.0) - target.get(k, 0.0)) for k in keys)
    
    
    # Prefer groups rich in rare types for val/test quotas
    def score_group_for_type(g, typ):
        return group_span_counts.get(g, Counter()).get(typ, 0)

    # Start with empty split assignments
    train_groups, val_groups, test_groups = set(), set(), set()

    val_have = Counter()    
    test_have = Counter()

    # Sort groups by BIBL+ORG richness
    rare_sorted = sorted(
        groups,
        key=lambda g: (score_group_for_type(g, "BIBL") + score_group_for_type(g, "ORG")),
        reverse=True
    )

    # Fill validation quotas
    for g in rare_sorted:
        if g in val_groups or g in test_groups or g in train_groups:
            continue
        need_any = any(val_have[t] < QUOTA_VAL[t] for t in QUOTA_VAL)
        if not need_any:
            break
        val_groups.add(g)
        for t in QUOTA_VAL:
            val_have[t] += group_span_counts[g].get(t, 0)

    # Fill test quotas (avoid overlap with val)
    for g in rare_sorted:
        if g in val_groups or g in test_groups or g in train_groups:
            continue
        need_any = any(test_have[t] < QUOTA_TEST[t] for t in QUOTA_TEST)
        if not need_any:
            break
        test_groups.add(g)
        for t in QUOTA_TEST:
            test_have[t] += group_span_counts[g].get(t, 0)
    
    for g in val_groups:
        splits["validation"]["groups"].add(g)
        splits["validation"]["counts"].update(group_counts[g])
        splits["validation"]["tokens"] += group_tokens[g]

    for g in test_groups:
        splits["test"]["groups"].add(g)
        splits["test"]["counts"].update(group_counts[g])
        splits["test"]["tokens"] += group_tokens[g]

    # Everything else goes into the greedy optimizer (including train)
    preassigned = val_groups | test_groups
    remaining_groups = [g for g in groups if g not in preassigned]

    # Sort groups by size (big first) for stability
    remaining_sorted = sorted(remaining_groups, key=lambda g: group_tokens[g], reverse=True)

    for g in remaining_sorted:
        g_cnt = group_counts[g]
        g_tok = group_tokens[g]

        candidates = []
        for sname in ["train", "validation", "test"]:
            # HARD budget gate: if split is already beyond budget, penalize strongly
            projected_tokens = splits[sname]["tokens"] + g_tok
            over = max(0, projected_tokens - budgets[sname])
            over_frac = over / max(1, budgets[sname])

            projected_counts = splits[sname]["counts"] + g_cnt
            score = l1(projected_counts, projected_tokens) + 5.0 * over_frac
            candidates.append((score, sname))

        candidates.sort()
        best = candidates[0][1]

        splits[best]["groups"].add(g)
        splits[best]["counts"].update(g_cnt)
        splits[best]["tokens"] += g_tok

    # Materialize
    train_g = splits["train"]["groups"]
    val_g = splits["validation"]["groups"]
    test_g = splits["test"]["groups"]

    train = [ex for ex in examples if ex[group_field] in train_g]
    val = [ex for ex in examples if ex[group_field] in val_g]
    test = [ex for ex in examples if ex[group_field] in test_g]

    print("Quota achieved:")
    print("  val :", {t: val_have[t] for t in QUOTA_VAL})
    print("  test:", {t: test_have[t] for t in QUOTA_TEST})
    print("Preassigned groups:", len(val_groups), len(test_groups))

    print("\n=== Group-level stratified split summary ===")
    for name in ["train", "validation", "test"]:
        c = splits[name]["counts"]; tot = splits[name]["tokens"]
        print(f"{name:10s} groups={len(splits[name]['groups']):5d} tokens={tot:9d} L1={l1(c, tot):.4f}")

    return train, val, test


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
        "group_id": [ex["group_id"] for ex in examples], 
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

    train_ex, dev_ex, test_ex = split_by_group_stratified(examples, group_field="group_id")
    print("Split sizes:", len(train_ex), len(dev_ex), len(test_ex))
    if len(dev_ex) == 0 or len(test_ex) == 0:
        raise RuntimeError("Split produced empty dev/test — check doc_id diversity and split logic.")

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
    
    # --- Report capping ---
    print("\n=== Span capping report (dropped to O) ===")
    total_spans = sum(CAP_LOG_SPANS.values())
    total_tokens = sum(CAP_LOG_TOKENS.values())
    print(f"Total spans dropped: {total_spans}")
    print(f"Total tokens dropped: {total_tokens}")

    for t in sorted(CAP_LOG_SPANS.keys()):
        print(f"{t:10s} spans={CAP_LOG_SPANS[t]:6d}  tokens={CAP_LOG_TOKENS[t]:8d}")

    # Save as JSON for reproducibility
    cap_report = {
        "max_span_tokens": MAX_SPAN_TOKENS,
        "dropped_spans": dict(CAP_LOG_SPANS),
        "dropped_tokens": dict(CAP_LOG_TOKENS),
        "total_dropped_spans": total_spans,
        "total_dropped_tokens": total_tokens,
    }
    with open(out_dir / "span_capping_report.json", "w", encoding="utf-8") as f:
        json.dump(cap_report, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_dir / 'span_capping_report.json'}")



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
