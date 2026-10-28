#!/usr/bin/env python3
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUTPUT = ROOT / 'output'

BANNED = {
    'water','Water','aqueous','Aqueous','proton','Proton','lead','Lead','gold','Gold'
}


def load_abstracts(path: Path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            pmid, title, abstract = line.rstrip('\n').split('\t')
            yield pmid, title, abstract


def load_annotations(path: Path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            pmid, sec, start, end, text, cls = line.rstrip('\n').split('\t')
            yield pmid, sec, int(start), int(end), text, cls


def build_phrase_list(annotations_path: Path, min_count: int = 1, min_len: int = 3):
    cnt = Counter(text for *_rest, text, _cls in load_annotations(annotations_path))
    phrases = [p for p, c in cnt.items() if c >= min_count and len(p) >= min_len and p not in BANNED]
    phrases.sort(key=len, reverse=True)
    return phrases


def _find_phrase_spans(text: str, phrase: str):
    spans = []
    start = 0
    L = len(phrase)
    while True:
        i = text.find(phrase, start)
        if i == -1:
            break
        j = i + L
        # Avoid matching substrings inside alphanumeric tokens.
        if (i == 0 or not text[i - 1].isalnum()) and (j == len(text) or not text[j].isalnum()):
            spans.append((i, j))
        start = i + 1
    return spans


def predict_spans(text: str, phrases):
    candidates = set()
    for p in phrases:
        for s, e in _find_phrase_spans(text, p):
            if text[s:e] not in BANNED:
                candidates.add((s, e))

    # Keep non-overlapping spans, preferring longer spans at same start.
    ordered = sorted(candidates, key=lambda x: (x[0], -(x[1] - x[0]), x[1]))
    out = []
    last_end = -1
    for s, e in ordered:
        if s >= last_end:
            out.append((s, e))
            last_end = e
    return out


def main():
    phrases = build_phrase_list(DATA / 'training.annotations.txt', min_count=1, min_len=3)

    out_lines = []
    for pmid, title, abstract in load_abstracts(DATA / 'test.abstracts.txt'):
        for sec, text in (('T', title), ('A', abstract)):
            for s, e in predict_spans(text, phrases):
                out_lines.append(f"{pmid}\t{sec}:{s}:{e}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT / 'predictions.tsv'
    pred_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding='utf-8')
    print(f"Wrote {len(out_lines)} predictions to {pred_path}")


if __name__ == '__main__':
    main()
