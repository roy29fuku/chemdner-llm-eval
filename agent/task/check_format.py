"""Check the format of a predictions file (it does not score).

Usage: python3 check_format.py /workspace/output/predictions.tsv
Each line must be `PMID<TAB>T:start:end` or `PMID<TAB>A:start:end`, where PMID is a test article and the offsets
lie within that section (0 <= start < end <= length).
"""
import sys

ABSTRACTS = "/workspace/data/test.abstracts.txt"


def main(path: str) -> int:
    texts = {}
    with open(ABSTRACTS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pmid, title, abstract = line.rstrip("\n").split("\t")
                texts[pmid] = {"T": title, "A": abstract}
    errors, n = [], 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            cols = line.rstrip("\n").split("\t")
            try:
                pmid = cols[0]
                section, start, end = cols[1].split(":")
                start, end = int(start), int(end)
            except (IndexError, ValueError):
                errors.append(f"line {i}: expected PMID<TAB>T|A:start:end, got {line.rstrip()!r}")
                continue
            if pmid not in texts:
                errors.append(f"line {i}: {pmid} is not a test article")
            elif section not in ("T", "A"):
                errors.append(f"line {i}: section must be T or A")
            elif not (0 <= start < end <= len(texts[pmid][section])):
                errors.append(f"line {i}: offsets {start}:{end} are outside the {section} section")
    for e in errors[:20]:
        print(e)
    print(f"{n} predictions, {len({l.split(chr(9))[0] for l in open(path, encoding='utf-8') if l.strip()})} articles, "
          f"{len(errors)} format errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
