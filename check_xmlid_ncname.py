import os
import re
import sys
from pathlib import Path

# Simple NCName regex according to XML Namespaces:
# - must NOT contain ':'
# - first char: letter or '_'
# - rest: letters, digits, '.', '-', '_'
NCNAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9._-]*$")

# Regex to capture xml:id values: xml:id="...".
XMLID_RE = re.compile(r'\bxml:id\s*=\s*"([^"]+)"')


def is_ncname(value: str) -> bool:
    """Return True if value is a valid NCName, False otherwise."""
    # NCName must not contain colon
    if ":" in value:
        return False
    return bool(NCNAME_RE.match(value))


def check_file(path: Path):
    """
    Check a single XML file for xml:id values that are not NCNames.

    Returns a list of dicts, each with:
      - line_no
      - value
    """
    results = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for lineno, line in enumerate(f, start=1):
                for m in XMLID_RE.finditer(line):
                    val = m.group(1)
                    if not is_ncname(val):
                        results.append(
                            {
                                "line_no": lineno,
                                "value": val,
                            }
                        )
    except Exception as e:
        print(f"ERROR reading {path}: {e}", file=sys.stderr)
    return results


def walk_and_check(root_dir: str):
    """
    Walk root_dir recursively, check all *.xml files,
    and print any invalid xml:id values.
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        print(f"Root directory does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning for XML files under: {root}")
    total_files = 0
    total_invalid = 0

    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(".xml"):
                continue
            total_files += 1
            path = Path(dirpath) / fname
            invalids = check_file(path)
            if invalids:
                print(f"\nFile: {path}")
                for item in invalids:
                    total_invalid += 1
                    print(
                        f'  Line {item["line_no"]}: xml:id="{item["value"]}"  <-- NOT NCName'
                    )

    print("\n--- Summary ---")
    print(f"XML files scanned: {total_files}")
    print(f"Invalid xml:id values: {total_invalid}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_xmlid_ncname.py PATH_TO_XML_ROOT", file=sys.stderr)
        sys.exit(1)

    root_dir = sys.argv[1]
    walk_and_check(root_dir)
