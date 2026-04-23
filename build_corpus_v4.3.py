import re
import json
import random
from pathlib import Path
from collections import defaultdict

from lxml import etree
from lxml.etree import XMLSyntaxError

from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from collections import Counter

# version 4.3.0
# Changes from 4.0.0:
#   - Extended to all seven entity types:
#     ARTEFACT, CONCEPT, PERSON, PLACE, ORG, CULTURE, BIBL
#   - Updated MAX_SPAN_TOKENS for new types
#   - Updated map_rs_type_to_label for new TEI rs/@type values:
#     org, people, person, bibl, place
# Changes from 3.0.1:
#   - Family-aware splitting via CSV (filepath, family_id, split)
#   - Cross-file paragraph chain reconstruction using prev/next attributes
#   - <join> target merging: fragments joined into one logical paragraph
#   - floatingText paragraphs included if they have prev/next attributes
#   - Short/non-content paragraphs filtered by MIN_CONTENT_CHARS
#   - Hyphenation processing deferred to after join merging
#   - doc_id is now per reconstructed chain: family_id::chain_root_p_id

# ============================================================
# 1. Global config
# ============================================================

GROUP_BLOCK_SIZE = 4
LB_HYPH = "⟦LB_HYPH⟧"
LB_NORM = "⟦LB⟧"

HYPH_SET = {"-", "\u2010", "\u2011", "\u2212", "\u00AD"}
DASH_SET = {"\u2013"}
JOIN_SET = HYPH_SET | DASH_SET

# Minimum characters in normalized text for a paragraph to be kept
MIN_CONTENT_CHARS = 50

def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"

MODEL_NAME = r"C:\Users\elena\Documents\WORK\USI\STIL PROJECT\ML\model_dapt"
MAX_LENGTH = 512
RANDOM_SEED = 42
CHUNK_LEN = 480
CHUNK_STRIDE = 128

MARK_DEL = True
MARK_ADD = True
MARK_HANDSHIFT = True
MARK_HI = True

HI_START,  HI_END  = " ⟦HI⟧ ",   " ⟦/HI⟧ "
DEL_START, DEL_END = " ⟦DEL⟧ ",  " ⟦/DEL⟧ "
ADD_START, ADD_END = " ⟦ADD⟧ ",  " ⟦/ADD⟧ "
LAT_MARK, KUR_MARK, GR_MARK = " ⟦LAT⟧ ", " ⟦KUR⟧ ", " ⟦GR⟧ "

TYPES = [
    "ARTEFACT",  # rs type="artefact" / "artifact"
    "CONCEPT",   # rs type="term" / "concept"
    "PERSON",    # rs type="person"
    "PLACE",     # rs type="place"
    "ORG",       # rs type="org"
    "CULTURE",   # rs type="people"
    "BIBL",      # rs type="bibl"
]

LABELS = ["O"]
for t in TYPES:
    LABELS.extend([f"B-{t}", f"I-{t}"])

label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

ENTITIES_DTD_PATH = r"C:\Users\elena\Documents\WORK\USI\STIL PROJECT\ML\training_data\Zeichen.dtd"

MAX_SPAN_TOKENS = {
    "ARTEFACT": 40,
    "CONCEPT":  40,
    "PERSON":   20,  # names are short; 20 is generous
    "PLACE":    20,
    "ORG":      40,  # organisation names can be long
    "CULTURE":  20,
    "BIBL":     60,  # bibliographic references can be quite long
}

CAP_LOG_SPANS  = Counter()
CAP_LOG_TOKENS = Counter()


# ============================================================
# 2. XML utilities (unchanged from v3)
# ============================================================

def postprocess_hyphenation(text_chars, char_map):
    """
    Post-process the merged character stream to resolve hyphenation markers.
    Called AFTER join merging so cross-join hyphens are handled correctly.
    """
    new_chars, new_map = [], []
    for ch, m in zip(text_chars, char_map):
        if ch == "\u00AD":
            continue
        new_chars.append(ch)
        new_map.append(m)
    text_chars, char_map = new_chars, new_map

    def match_at(i, needle):
        if i + len(needle) > len(text_chars):
            return False
        return "".join(text_chars[i:i+len(needle)]) == needle

    out_chars, out_map = [], []
    i = 0
    while i < len(text_chars):
        if match_at(i, LB_NORM):
            out_chars.append(" ")
            out_map.append(char_map[i])
            i += len(LB_NORM)
            continue

        if match_at(i, LB_HYPH):
            left_ok  = len(out_chars) > 0 and _is_word_char(out_chars[-1])
            j = i + len(LB_HYPH)
            while j < len(text_chars) and text_chars[j] == " ":
                j += 1
            right_ok = j < len(text_chars) and _is_word_char(text_chars[j])
            if left_ok and right_ok:
                i = j
                continue
            i += len(LB_HYPH)
            continue

        if text_chars[i] in JOIN_SET:
            j = i + 1
            while j < len(text_chars) and text_chars[j] == " ":
                j += 1
            if match_at(j, LB_HYPH):
                k = j + len(LB_HYPH)
                while k < len(text_chars) and text_chars[k] == " ":
                    k += 1
                left_ok  = len(out_chars) > 0 and _is_word_char(out_chars[-1])
                right_ok = k < len(text_chars) and _is_word_char(text_chars[k])
                if left_ok and right_ok:
                    i = k
                    continue

        if match_at(i, LB_HYPH):
            i += len(LB_HYPH)
            continue

        out_chars.append(text_chars[i])
        out_map.append(char_map[i])
        i += 1

    final_chars, final_map = [], []
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

    while final_chars and final_chars[0]  == " ": final_chars.pop(0);  final_map.pop(0)
    while final_chars and final_chars[-1] == " ": final_chars.pop();   final_map.pop()

    return "".join(final_chars), final_map


def build_entity_map_from_dtd(dtd_path):
    text = Path(dtd_path).read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(r'<!ENTITY\s+([^"\s]+)\s+"&#(\d+);"')
    return {name: chr(int(code)) for name, code in pat.findall(text)}

ENTITY_MAP = build_entity_map_from_dtd(ENTITIES_DTD_PATH)


def expand_custom_entities(xml_text, entity_map):
    pat = re.compile(r"&([^\d#][^;\s]*);")
    def repl(m):
        return entity_map.get(m.group(1), m.group(0))
    return pat.sub(repl, xml_text)


def remove_xml_comments(xml_text):
    return re.sub(r"<!--.*?-->", "", xml_text, flags=re.DOTALL)


def sanitize_xml_text(xml_text, entity_map):
    xml_text = expand_custom_entities(xml_text, entity_map)

    def repl_unknown_entity(m):
        name = m.group(1)
        if name in entity_map:
            return m.group(0)
        return "&amp;" + name + ";"

    xml_text = re.sub(r"&([A-Za-z_][A-Za-z0-9._:-]*);", repl_unknown_entity, xml_text)
    bare_amp = re.compile(r"&(?![A-Za-z_][A-Za-z0-9._:-]*;|#[0-9]+;|#x[0-9A-Fa-f]+;)")
    return bare_amp.sub("&amp;", xml_text)


def parse_xml_file(xml_path):
    """
    Parse one XML file, returning the lxml root element, or None on error.
    """
    xml_text = Path(xml_path).read_text(encoding="utf-8", errors="ignore")
    xml_text = remove_xml_comments(xml_text)
    xml_text = sanitize_xml_text(xml_text, ENTITY_MAP)
    parser = etree.XMLParser(load_dtd=False, resolve_entities=False, no_network=True)
    try:
        return etree.fromstring(xml_text.encode("utf-8"), parser=parser)
    except XMLSyntaxError as e:
        print(f"[XML ERROR] Skipping {xml_path}: {e}")
        return None


# ============================================================
# 3. Paragraph normalization (raw — no hyphenation yet)
# ============================================================

def normalize_paragraph_raw(p_elem):
    """
    Build (text_chars, char_map) from a <p> element WITHOUT calling
    postprocess_hyphenation. Hyphenation is deferred so that it can be
    applied to the fully merged character stream after join stitching.
    """
    text_chars, char_map = [], []

    def append_char(ch, node, kind, offset):
        text_chars.append(ch)
        char_map.append((node, kind, offset))

    def append_literal(s, node, kind):
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
                pass

            elif tag == "del":
                if MARK_DEL:
                    append_literal(DEL_START, child, "marker")
                recurse(child)
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
                    elif "griechisch" in scr:
                        append_literal(GR_MARK, child, "marker")
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
    return text_chars, char_map


# ============================================================
# 4. Entity extraction (unchanged logic, works on final text+cmap)
# ============================================================

def build_inverse_char_map(char_map):
    idx_map = {}
    for char_index, (node, kind, offset) in enumerate(char_map):
        idx_map[(node, kind, offset)] = char_index
    return idx_map


def map_rs_type_to_label(rs_type):
    mapping = {
        # Artefact
        "artefact": "ARTEFACT",
        "artifact": "ARTEFACT",
        # Concept / term
        "term":     "CONCEPT",
        "concept":  "CONCEPT",
        # Person
        "person":   "PERSON",
        # Place
        "place":    "PLACE",
        # Organisation
        "org":      "ORG",
        # Cultural group
        "people":   "CULTURE",
        # Bibliographic reference
        "bibl":     "BIBL",
    }
    return mapping.get(rs_type)


def get_span_for_rs(rs_elem, idx_map):
    start, end = None, None

    def mark_char(node, kind, i):
        nonlocal start, end
        key = (node, kind, i)
        if key in idx_map:
            idx = idx_map[key]
            if start is None or idx < start:
                start = idx
            if end is None or idx >= end:
                end = idx + 1

    def recurse(node):
        if node.text:
            for i in range(len(node.text)):
                mark_char(node, "text", i)
        for child in node:
            recurse(child)
            if child.tail:
                for i in range(len(child.tail)):
                    mark_char(child, "tail", i)

    recurse(rs_elem)
    return start, end


def extract_entities_from_elems(p_elems, char_map):
    """
    Extract entities from a list of <p> elements (the fragments forming
    one logical paragraph) using the provided unified char_map.
    Entities that straddle a join boundary are dropped silently.
    """
    idx_map = build_inverse_char_map(char_map)
    entities = []

    for p_elem in p_elems:
        for rs in p_elem.xpath(".//tei:rs", namespaces=TEI_NS):
            rs_type = (rs.get("type") or "").strip().lower()
            label = map_rs_type_to_label(rs_type)
            if label is None:
                continue
            start, end = get_span_for_rs(rs, idx_map)
            if start is not None and end is not None and start < end:
                entities.append({"start": start, "end": end, "label": label})

    entities.sort(key=lambda e: e["start"])
    return entities


# ============================================================
# 5. Cross-file paragraph chain reconstruction
# ============================================================

def _normalise_ref(ref: str, current_filename: str) -> tuple[str, str]:
    """
    Normalise a prev/next/join ref to (filename_lower, p_id).
    Handles:
      - "#p.foo"            -> (current_filename_lower, "p.foo")
      - "other.xml#p.foo"   -> ("other.xml_lower", "p.foo")
    """
    ref = ref.strip()
    if ref.startswith("#"):
        return current_filename.lower(), ref[1:]
    if "#" in ref:
        fname, pid = ref.split("#", 1)
        return fname.lower(), pid
    # bare id, no '#'
    return current_filename.lower(), ref


def load_all_paragraphs(xml_files: list[str]) -> dict:
    """
    Parse all XML files and build a global paragraph registry:

      registry[(filename_lower, p_id)] = {
          "elem":     lxml element,
          "prev":     (fname_lower, p_id) | None,
          "next":     (fname_lower, p_id) | None,
          "filename": original filename string,
          "file_path": full path string,
      }

    Also returns join_pairs: list of [(fname, pid), (fname, pid)] that
    should be merged into one logical paragraph.
    """
    registry   = {}
    join_pairs = []

    for xml_path in xml_files:
        fname = Path(xml_path).name
        fname_lower = fname.lower()
        root = parse_xml_file(xml_path)
        if root is None:
            continue

        # Collect all <p> elements outside teiHeader
        # Include those inside <floatingText>
        p_elems = root.xpath(
            "//*[local-name()='p' and not(ancestor::*[local-name()='teiHeader'])]"
        )

        for p in p_elems:
            p_id = p.get("{http://www.tei-c.org/ns/1.0}id") or p.get("xml:id") or p.get("id")
            if not p_id:
                continue

            prev_ref = p.get("prev")
            next_ref = p.get("next")

            key = (fname_lower, p_id)
            registry[key] = {
                "elem":      p,
                "prev":      _normalise_ref(prev_ref, fname_lower) if prev_ref else None,
                "next":      _normalise_ref(next_ref, fname_lower) if next_ref else None,
                "filename":  fname,
                "file_path": xml_path,
            }

        # Collect <join> elements -> pairs to merge
        join_elems = root.xpath(
            "//*[local-name()='join']"
        )
        for j in join_elems:
            target = j.get("target", "").strip()
            if not target:
                continue
            refs = target.split()
            if len(refs) == 2:
                a = _normalise_ref(refs[0], fname_lower)
                b = _normalise_ref(refs[1], fname_lower)
                join_pairs.append((a, b))

    return registry, join_pairs


def build_join_set(join_pairs: list) -> set:
    """
    Return a set of frozensets, each containing two paragraph keys that
    should be merged. We use frozenset so lookup is order-independent.
    """
    return {frozenset([a, b]) for a, b in join_pairs}


def build_chains(registry: dict, join_set: set) -> list[list]:
    """
    Walk prev/next links to build ordered chains of paragraph keys.
    Each chain is a list of (fname_lower, p_id) keys in reading order.

    Rules:
    - Start each chain at a paragraph with no prev (chain head).
    - Within each chain, wherever two consecutive paragraphs form a
      join pair, they are flagged for text merging later.
    - Paragraphs with neither prev nor next are stand-alone and form
      single-element chains (will be filtered by content length later).
    """
    visited = set()
    chains  = []

    # Sort for determinism
    all_keys = sorted(registry.keys())

    for key in all_keys:
        if key in visited:
            continue

        entry = registry[key]

        # Only start a chain at a head (no prev pointer)
        if entry["prev"] is not None:
            continue

        chain = []
        current = key
        while current is not None:
            if current in visited:
                break
            if current not in registry:
                break
            visited.add(current)
            chain.append(current)
            current = registry[current]["next"]

        if chain:
            chains.append(chain)

    # Any paragraph not yet visited (e.g. orphaned, mid-chain only)
    # gets its own single-element chain
    for key in all_keys:
        if key not in visited:
            chains.append([key])

    return chains


# ============================================================
# 6. Building logical paragraphs from chains + joins
# ============================================================

def build_logical_paragraphs(chain: list, registry: dict, join_set: set) -> list[dict]:
    """
    Given an ordered chain of paragraph keys, group consecutive keys that
    belong to the same <join> into merged logical paragraphs.

    Returns a list of logical paragraph dicts:
      {
        "keys":      [(fname, pid), ...],   # 1 key = simple; 2 = joined
        "is_joined": bool,
      }
    """
    if not chain:
        return []

    logical = []
    i = 0
    while i < len(chain):
        key_a = chain[i]
        # Check if this paragraph and the next form a join pair
        if i + 1 < len(chain):
            key_b = chain[i + 1]
            if frozenset([key_a, key_b]) in join_set:
                logical.append({"keys": [key_a, key_b], "is_joined": True})
                i += 2
                continue
        logical.append({"keys": [key_a], "is_joined": False})
        i += 1

    return logical


def materialise_logical_paragraph(lp: dict, registry: dict) -> tuple:
    """
    Given a logical paragraph (one or two fragment keys), produce the
    final (text, char_map, p_elems) by:
      1. Calling normalize_paragraph_raw on each fragment
      2. Concatenating with a space separator (unless the join is
         hyphenated, which postprocess_hyphenation will detect and fix)
      3. Running postprocess_hyphenation on the merged stream

    Returns (text, char_map, p_elems) or None if content too short.
    """
    all_chars, all_map, p_elems = [], [], []

    for idx, key in enumerate(lp["keys"]):
        if key not in registry:
            continue
        p_elem = registry[key]["elem"]
        p_elems.append(p_elem)
        chars, cmap = normalize_paragraph_raw(p_elem)

        if idx > 0:
            # Insert a space between fragments; hyphenation processing
            # will remove it if the join is actually a hyphenated word
            sentinel_node = p_elems[0]  # dummy node reference
            all_chars.append(" ")
            all_map.append((sentinel_node, "join_sep", -999))

        all_chars.extend(chars)
        all_map.extend(cmap)

    if not all_chars:
        return None

    # Defer hyphenation processing to now, after merging
    text, char_map = postprocess_hyphenation(all_chars, all_map)

    # Filter out short/non-content paragraphs
    # Strip markers to measure real content length
    marker_pattern = re.compile(r"⟦[^⟧]+⟧")
    clean = marker_pattern.sub("", text).strip()
    if len(clean) < MIN_CONTENT_CHARS:
        return None

    return text, char_map, p_elems


# ============================================================
# 7. BIO chunking (unchanged logic from v3)
# ============================================================

def spans_to_bio_chunked(text, entities, tokenizer, label2id, chunk_len=480, stride=128):
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=True,
    )

    input_ids_all = enc["input_ids"]
    attn_all      = enc["attention_mask"]
    offsets_all   = enc["offset_mapping"]
    n = len(input_ids_all)

    def bio_for_offsets(offsets):
        token_tags = ["O"] * len(offsets)
        for ent in entities:
            start, end, ent_type = ent["start"], ent["end"], ent["label"]
            token_indices = [
                i for i, (ts, te) in enumerate(offsets)
                if not (ts == te == 0) and ts < end and te > start
            ]
            if not token_indices:
                continue
            ent_type = ent_type.upper()
            cap = MAX_SPAN_TOKENS.get(ent_type)
            if cap is not None and len(token_indices) > cap:
                CAP_LOG_SPANS[ent_type]  += 1
                CAP_LOG_TOKENS[ent_type] += len(token_indices)
                continue
            token_tags[token_indices[0]] = f"B-{ent_type}"
            for ti in token_indices[1:]:
                token_tags[ti] = f"I-{ent_type}"

        return [
            -100 if ts == te == 0 else label2id[tag]
            for (ts, te), tag in zip(offsets, token_tags)
        ]

    chunks = []
    start  = 0
    pad_id = tokenizer.pad_token_id

    while start < n:
        end = min(start + chunk_len, n)

        chunk_input_ids = input_ids_all[start:end]
        chunk_attn      = attn_all[start:end]
        chunk_offsets   = offsets_all[start:end]
        chunk_labels    = bio_for_offsets(chunk_offsets)

        pad_len = chunk_len - len(chunk_input_ids)
        if pad_len > 0:
            chunk_input_ids = chunk_input_ids + [pad_id]  * pad_len
            chunk_attn      = chunk_attn      + [0]        * pad_len
            chunk_labels    = chunk_labels    + [-100]     * pad_len

        chunks.append({
            "input_ids":      chunk_input_ids,
            "attention_mask": chunk_attn,
            "labels":         chunk_labels,
            "chunk_start":    start,
            "chunk_end":      end,
        })

        if end == n:
            break
        start = end - stride

    return chunks


# ============================================================
# 8. CSV loading and family/split mapping
# ============================================================

def load_family_csv(csv_path: str) -> dict:
    """
    Load the family CSV and return two dicts:
      file_to_family: {filepath_lower -> family_id}
      file_to_split:  {filepath_lower -> "train" | "dev" | "test"}

    CSV format (with header):
      filepath,family_id,split
      /path/to/file.xml,vol1_sec1,train
      ...

    - filepath matching is case-insensitive on the filename stem.
    - Files not in the CSV are assigned family_id="mixed", split="train".
    """
    import csv

    file_to_family = {}
    file_to_split  = {}

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        # Auto-detect tab vs comma delimiter from the header line
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
# 9. Main corpus building pipeline
# ============================================================

def build_examples_from_chains(
    chains:        list,
    registry:      dict,
    join_set:      set,
    file_to_family: dict,
    file_to_split:  dict,
) -> list[dict]:
    """
    For each chain -> logical paragraphs -> materialise -> BIO chunk -> examples.
    """
    all_examples = []

    for chain in chains:
        if not chain:
            continue

        # Determine family and split from the first file in the chain
        # (floatingText from a different file uses the host file's assignment)
        first_fname = registry[chain[0]]["filename"].lower() if chain[0] in registry else ""
        family_id   = file_to_family.get(first_fname, "mixed")
        split       = file_to_split.get(first_fname,  "train")

        logical_paragraphs = build_logical_paragraphs(chain, registry, join_set)

        for lp_idx, lp in enumerate(logical_paragraphs):
            result = materialise_logical_paragraph(lp, registry)
            if result is None:
                continue  # too short / empty

            text, char_map, p_elems = result
            entities = extract_entities_from_elems(p_elems, char_map)

            # doc_id encodes family + chain root paragraph id
            chain_root_pid = chain[0][1]  # p_id of chain head
            doc_id = f"{family_id}::{chain_root_pid}"

            chunk_encodings = spans_to_bio_chunked(
                text, entities, tokenizer, label2id,
                chunk_len=CHUNK_LEN, stride=CHUNK_STRIDE,
            )

            for c_idx, encoding in enumerate(chunk_encodings):
                example = {
                    "text":           text,
                    "entities":       entities,
                    "family_id":      family_id,
                    "split":          split,
                    "doc_id":         doc_id,
                    "lp_index":       lp_idx,
                    "chunk_index":    c_idx,
                    "group_id":       f"{doc_id}::{lp_idx}::{c_idx // GROUP_BLOCK_SIZE}",
                    "is_joined":      lp["is_joined"],
                    "chunk_start":    encoding["chunk_start"],
                    "chunk_end":      encoding["chunk_end"],
                    "input_ids":      encoding["input_ids"],
                    "attention_mask": encoding["attention_mask"],
                    "labels":         encoding["labels"],
                }
                all_examples.append(example)

    return all_examples


def load_corpus(root_dir: str, csv_path: str) -> list[dict]:
    """
    Full pipeline:
      1. Load family CSV
      2. Find all XML files
      3. Parse all files and build global paragraph registry
      4. Build join set from <join> elements
      5. Build chains from prev/next links
      6. Build examples
    """
    file_to_family, file_to_split = load_family_csv(csv_path)

    xml_files = sorted(str(p) for p in Path(root_dir).rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files.")

    print("Parsing XML files and building paragraph registry...")
    registry, join_pairs = load_all_paragraphs(xml_files)
    print(f"  Registered {len(registry)} paragraphs, {len(join_pairs)} join pairs.")

    join_set = build_join_set(join_pairs)
    chains   = build_chains(registry, join_set)
    print(f"  Built {len(chains)} paragraph chains.")

    print("Building training examples...")
    examples = build_examples_from_chains(
        chains, registry, join_set, file_to_family, file_to_split
    )
    print(f"  Total examples (chunks): {len(examples)}")
    return examples


# ============================================================
# 10. Family-aware split (using CSV split column directly)
# ============================================================

def family_aware_split(examples: list[dict]) -> tuple[list, list, list]:
    """
    Partition examples into train/dev/test using the 'split' field
    already set from the CSV. No randomisation needed — the split is
    determined entirely by your manual family assignments.
    """
    train = [ex for ex in examples if ex["split"] == "train"]
    dev   = [ex for ex in examples if ex["split"] == "dev"]
    test  = [ex for ex in examples if ex["split"] == "test"]

    print("\n=== Family-aware split summary ===")
    families = defaultdict(set)
    for ex in examples:
        families[ex["split"]].add(ex["family_id"])

    for split_name in ("train", "dev", "test"):
        fams = sorted(families[split_name])
        n    = len([ex for ex in examples if ex["split"] == split_name])
        print(f"  {split_name:5s}: {n:6d} examples  families={fams}")

    # Sanity check: no family appears in both dev/test and train
    train_families = families["train"]
    for split_name in ("dev", "test"):
        overlap = train_families & families[split_name]
        if overlap:
            print(f"  WARNING: families in both train and {split_name}: {overlap}")

    return train, dev, test


def entity_distribution_report(
    split_examples: dict[str, list[dict]],
    out_dir: Path,
):
    """
    Count entity spans per label per split (from the 'entities' field,
    which holds char-level spans before chunking, so each entity is
    counted exactly once regardless of how many chunks it appears in).

    Also counts unique logical paragraphs (doc_id::lp_index) to avoid
    double-counting entities that appear in overlapping chunks.

    Prints a formatted table and saves a JSON report.
    """
    # entity_counts[split][label] = count of unique entity spans
    entity_counts: dict[str, Counter] = {}

    for split_name, examples in split_examples.items():
        counts = Counter()
        # Track (doc_id, lp_index) already counted to avoid double-counting
        # across overlapping chunks of the same logical paragraph
        seen_lp: set = set()

        for ex in examples:
            lp_key = (ex["doc_id"], ex["lp_index"])
            if lp_key in seen_lp:
                continue
            seen_lp.add(lp_key)
            for ent in ex["entities"]:
                counts[ent["label"]] += 1

        entity_counts[split_name] = counts

    # All label types that appear anywhere
    all_labels = sorted({
        label
        for counts in entity_counts.values()
        for label in counts
    })

    # --- Console report ---
    col_w   = 10   # split column width
    label_w = 10   # label column width
    splits  = ["train", "dev", "test"]

    print("\n=== Entity distribution per split ===")
    header = f"  {'Label':{label_w}s}" + "".join(
        f"  {s:>{col_w}s}" for s in splits
    ) + f"  {'Total':>{col_w}s}"
    print(header)
    print("  " + "-" * (label_w + (col_w + 2) * (len(splits) + 1)))

    totals_per_split: Counter = Counter()
    report_rows = {}

    for label in all_labels:
        row_counts = {s: entity_counts.get(s, Counter())[label] for s in splits}
        total = sum(row_counts.values())
        report_rows[label] = {**row_counts, "total": total}
        for s in splits:
            totals_per_split[s] += row_counts[s]

        row_str = f"  {label:{label_w}s}" + "".join(
            f"  {row_counts[s]:>{col_w}d}" for s in splits
        ) + f"  {total:>{col_w}d}"
        print(row_str)

    # Totals row
    grand_total = sum(totals_per_split.values())
    print("  " + "-" * (label_w + (col_w + 2) * (len(splits) + 1)))
    totals_str = f"  {'TOTAL':{label_w}s}" + "".join(
        f"  {totals_per_split[s]:>{col_w}d}" for s in splits
    ) + f"  {grand_total:>{col_w}d}"
    print(totals_str)

    # Per-split percentage of total entities
    print()
    for s in splits:
        pct = 100 * totals_per_split[s] / grand_total if grand_total else 0
        print(f"  {s:5s}: {totals_per_split[s]:6d} entities  ({pct:.1f}% of total)")

    # --- JSON report ---
    json_report = {
        "entity_counts_per_split": {
            label: report_rows[label] for label in all_labels
        },
        "totals_per_split": dict(totals_per_split),
        "grand_total": grand_total,
    }
    report_path = out_dir / "entity_distribution_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    print(f"\n  Wrote entity_distribution_report.json")




def examples_to_dataset(examples: list[dict]) -> Dataset:
    data = {
        "text":           [ex["text"]           for ex in examples],
        "entities":       [ex["entities"]       for ex in examples],
        "family_id":      [ex["family_id"]      for ex in examples],
        "doc_id":         [ex["doc_id"]         for ex in examples],
        "lp_index":       [ex["lp_index"]       for ex in examples],
        "group_id":       [ex["group_id"]       for ex in examples],
        "input_ids":      [ex["input_ids"]      for ex in examples],
        "attention_mask": [ex["attention_mask"] for ex in examples],
        "labels":         [ex["labels"]         for ex in examples],
    }
    return Dataset.from_dict(data)


def write_jsonl(examples: list[dict], path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def build_and_save_corpus(root_dir: str, csv_path: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_corpus(root_dir, csv_path)
    if not examples:
        raise RuntimeError("No examples produced — check XML files and CSV.")

    train_ex, dev_ex, test_ex = family_aware_split(examples)

    if len(dev_ex) == 0 or len(test_ex) == 0:
        raise RuntimeError(
            "Split produced empty dev or test set. "
            "Check that your CSV assigns families to all three splits."
        )

    write_jsonl(train_ex, out_dir / "train.jsonl")
    write_jsonl(dev_ex,   out_dir / "dev.jsonl")
    write_jsonl(test_ex,  out_dir / "test.jsonl")
    print(f"\nWrote JSONL to {out_dir}")

    ds_dict = DatasetDict({
        "train":      examples_to_dataset(train_ex),
        "validation": examples_to_dataset(dev_ex),
        "test":       examples_to_dataset(test_ex),
    })
    ds_dict.save_to_disk(str(out_dir / "hf_dataset"))
    print(f"Saved HF Dataset to {out_dir / 'hf_dataset'}")

    # Span capping report
    print("\n=== Span capping report ===")
    total_spans  = sum(CAP_LOG_SPANS.values())
    total_tokens = sum(CAP_LOG_TOKENS.values())
    print(f"  Total spans dropped : {total_spans}")
    print(f"  Total tokens dropped: {total_tokens}")
    for t in sorted(CAP_LOG_SPANS):
        print(f"  {t:10s} spans={CAP_LOG_SPANS[t]:6d}  tokens={CAP_LOG_TOKENS[t]:8d}")

    cap_report = {
        "max_span_tokens":    MAX_SPAN_TOKENS,
        "dropped_spans":      dict(CAP_LOG_SPANS),
        "dropped_tokens":     dict(CAP_LOG_TOKENS),
        "total_dropped_spans":  total_spans,
        "total_dropped_tokens": total_tokens,
    }
    with open(out_dir / "span_capping_report.json", "w", encoding="utf-8") as f:
        json.dump(cap_report, f, ensure_ascii=False, indent=2)
    print(f"Wrote span_capping_report.json")

    # Entity distribution report
    entity_distribution_report(
        {"train": train_ex, "dev": dev_ex, "test": test_ex},
        out_dir,
    )


# ============================================================
# 12. CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build TEI NER corpus with family-aware splitting (v4)."
    )
    parser.add_argument("root_dir", help="Root directory containing TEI XML files")
    parser.add_argument("csv_path", help="CSV file: filepath,family_id,split")
    parser.add_argument("out_dir",  help="Output directory for JSONL and HF dataset")

    args = parser.parse_args()
    build_and_save_corpus(args.root_dir, args.csv_path, args.out_dir)
