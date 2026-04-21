import argparse
import re
from pathlib import Path
from lxml import etree

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# Toggle these depending on whether you want the model to see editorial markers during DAPT
MARK_DEL = True
MARK_ADD = True
MARK_HANDSHIFT = True
MARK_HI = True

HI_START, HI_END = " ⟦HI⟧ ", " ⟦/HI⟧ "
DEL_START, DEL_END = " ⟦DEL⟧ ", " ⟦/DEL⟧ "
ADD_START, ADD_END = " ⟦ADD⟧ ", " ⟦/ADD⟧ "
LAT_MARK, KUR_MARK, GR_MARK = " ⟦LAT⟧ ", " ⟦KUR⟧ ", " ⟦GR⟧ "

LB_HYPH = "⟦LB_HYPH⟧"
LB_NORM = "⟦LB⟧"

# hyphen-like chars we want to remove at hyphenation breaks
HYPH_CHARS = r"\-\u2010\u2011\u2212\u00AD"  # - ‐ - − soft hyphen
DASH_CHARS = r"\u2013"  # – (en dash) sometimes used as hyphen in transcriptions

def postprocess_linebreak_hyphens(text: str) -> str:
    """
    1) Collapse whitespace
    2) Join words split by <lb type="hyph"/> by removing hyphen/dash near the break marker.
    3) Turn normal LB markers into a space.
    """
    # Normalize whitespace early
    text = re.sub(r"\s+", " ", text)

    # Remove soft hyphens anywhere (often noise in historical text)
    text = text.replace("\u00AD", "")

    # Join: letter + optional spaces + hyphen/dash + optional spaces + LB_HYPH + optional spaces + letter
    # Example: "Wort- ⟦LB_HYPH⟧ fort" -> "Wortfort"
    # Also handles: "Wort –⟦LB_HYPH⟧fort" or "Wort-⟦LB_HYPH⟧ fort"
    text = re.sub(
        rf"(?<=\w)\s*([{HYPH_CHARS}{DASH_CHARS}])\s*{re.escape(LB_HYPH)}\s*(?=\w)",
        "",
        text,
    )

    # In case the transcriber encoded the break without an explicit hyphen char,
    # you might want to *just remove the marker* and join:
    # text = re.sub(rf"(?<=\w)\s*{re.escape(LB_HYPH)}\s*(?=\w)", "", text)

    # Replace any remaining LB markers with a space
    text = text.replace(LB_NORM, " ").replace(LB_HYPH, "")

    # Final whitespace cleanup
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def remove_xml_comments(xml_text: str) -> str:
    """
    Remove anything inside XML comments: <!-- ... -->
    Works across multiple lines.
    """
    return re.sub(r"<!--.*?-->", "", xml_text, flags=re.DOTALL)

# Path to your TEI entity DTD
ENTITIES_DTD_PATH = r"C:\Users\elena\Documents\WORK\USI\STIL PROJECT\ML\training_data\Zeichen.dtd"


def build_entity_map_from_dtd(dtd_path: str) -> dict:
    """
    Parse a DTD like Zeichen.dtd and map entity name -> Unicode character.
    Assumes lines like <!ENTITY semikol "&#59;" >
    """
    text = Path(dtd_path).read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(r'<!ENTITY\s+([^"\s]+)\s+"&#(\d+);"')
    entity_map = {}
    for name, code in pat.findall(text):
        entity_map[name] = chr(int(code))
    return entity_map


ENTITY_MAP = build_entity_map_from_dtd(ENTITIES_DTD_PATH)


def expand_custom_entities(xml_text: str, entity_map: dict) -> str:
    """
    Replace known custom entities like &semikol; with their character.
    Leave unknown entities untouched for the next sanitation step.
    """
    pat = re.compile(r"&([^\d#][^;\s]*);")

    def repl(m):
        name = m.group(1)
        return entity_map.get(name, m.group(0))

    return pat.sub(repl, xml_text)


def sanitize_xml_text(xml_text: str, entity_map: dict) -> str:
    """
    1) Expand known custom entities.
    2) Escape unknown named entities as literal text.
    3) Escape bare '&' that do not begin a valid entity/reference.
    """
    xml_text = expand_custom_entities(xml_text, entity_map)

    def repl_unknown_entity(m):
        name = m.group(1)
        if name in entity_map:
            return m.group(0)
        return "&amp;" + name + ";"

    xml_text = re.sub(r"&([A-Za-z_][A-Za-z0-9._:-]*);", repl_unknown_entity, xml_text)

    bare_amp_pattern = re.compile(
        r"&(?![A-Za-z_][A-Za-z0-9._:-]*;|#[0-9]+;|#x[0-9A-Fa-f]+;)"
    )
    xml_text = bare_amp_pattern.sub("&amp;", xml_text)

    return xml_text

def paragraph_to_text(p_elem: etree._Element) -> str:
    """Flatten a TEI <p> into text, similar to your NER corpus normalizer."""
    out = []

    def append(s: str):
        if s:
            out.append(s)

    def recurse(node: etree._Element):
        # node text
        if node.text:
            append(node.text)

        for child in node:
            # Skip comments / processing instructions etc.
            if not isinstance(child.tag, str):
                if child.tail:
                    append(child.tail)
                continue
            
            tag = etree.QName(child).localname

            if tag == "lb":
                lb_type = (child.get("type") or "").lower()
                if lb_type == "hyph":
                    append(f" {LB_HYPH} ")
                else:
                    append(" ")

            #elif tag == "choice":
                # Prefer <corr>, else <sic>, else recurse
                #corr = child.find("tei:corr", namespaces=TEI_NS)
                #sic = child.find("tei:sic", namespaces=TEI_NS)
                #if corr is not None:
                    #append("".join(corr.itertext()))
                #elif sic is not None:
                    #append("".join(sic.itertext()))
                #else:
                    #recurse(child)

            elif tag == "del":
                if MARK_DEL:
                    append(DEL_START)
                recurse(child)  # keep deleted text for domain style
                if MARK_DEL:
                    append(DEL_END)

            elif tag == "add":
                if MARK_ADD:
                    append(ADD_START)
                recurse(child)
                if MARK_ADD:
                    append(ADD_END)

            elif tag == "handShift":
                if MARK_HANDSHIFT:
                    scr = (child.get("script") or "").lower()
                    if "latein" in scr or "latin" in scr:
                        append(LAT_MARK)
                    elif "kurrent" in scr:
                        append(KUR_MARK)
                    elif "griechisch" in scr:
                        append(GR_MARK)    
                recurse(child)

            elif tag == "hi":
                if MARK_HI:
                    append(HI_START)
                recurse(child)
                if MARK_HI:
                    append(HI_END)

            else:
                # default: drop tags but keep content (includes <rs>, <unclear>, etc.)
                recurse(child)

            # child tail
            if child.tail:
                append(child.tail)

    recurse(p_elem)
    raw = "".join(out)
    return postprocess_linebreak_hyphens(raw)


def extract_paragraphs_from_file(xml_path: Path) -> list[str]:
    xml_text = xml_path.read_text(encoding="utf-8", errors="ignore")

    # remove commented-out XML so it never enters the corpus
    xml_text = remove_xml_comments(xml_text)
    
    # sanitize entities / bare ampersands before parsing
    xml_text = sanitize_xml_text(xml_text, ENTITY_MAP)

    # parse without loading DTDs; do NOT silently recover broken markup
    parser = etree.XMLParser(
        load_dtd=False,
        resolve_entities=False,
        no_network=True,
        recover=False,
        remove_comments=True,
    )
    root = etree.fromstring(xml_text.encode("utf-8"), parser=parser)

    # paragraphs in body only (exclude teiHeader)
    p_elems = root.xpath("//tei:text//tei:body//tei:p", namespaces=TEI_NS)

    lines = []
    for p in p_elems:
        t = paragraph_to_text(p)
        if t:
            if "<lb" in t or "</p>" in t or "<rs" in t or "<handShift" in t:
                print(f"[LEAKED TAGS] {xml_path}")
                print(t[:1000])
            lines.append(t)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, type=Path, help="Folder with TEI XML files (recursive)")
    ap.add_argument("--out_txt", required=True, type=Path, help="Output text file (one paragraph per line)")
    ap.add_argument("--min_chars", type=int, default=5, help="Drop lines shorter than this")
    args = ap.parse_args()

    xml_files = sorted(args.in_dir.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files under {args.in_dir}")

    args.out_txt.parent.mkdir(parents=True, exist_ok=True)

    n_lines = 0
    with args.out_txt.open("w", encoding="utf-8") as out:
        for f in xml_files:
            try:
                lines = extract_paragraphs_from_file(f)
            except Exception as e:
                print(f"[WARN] Skipping {f}: {e}")
                continue

            for line in lines:
                if len(line) < args.min_chars:
                    continue
                out.write(line + "\n")
                n_lines += 1

    print(f"Wrote {n_lines} lines to {args.out_txt}")


if __name__ == "__main__":
    main()