#!/usr/bin/env python3
"""Simple chemical mention tagger.

Rules:
- Uses mention surface forms learned from training+dev annotations (allowed).
- Adds regex-based matches for CAS numbers, InChI strings, ions, and formula-like tokens.
- Adds limited one-letter element matches based on observed training usage.

Outputs exact character offsets (0-based, end exclusive) for title and abstract.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


WORDCHAR_RE = re.compile(r"[A-Za-z0-9]")

CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
INCHI_RE = re.compile(r"\bInChI=\S+")
ION_RE = re.compile(r"\b[A-Z][a-z]?\(\d?[+\-]\)\b")
# Broad formula-ish: C6H12O6, CuSO4(aq), (13)C, NADP+/H (we will filter).
FORMULA_RE = re.compile(r"\b\(?\d{1,3}\)?(?:[A-Z][a-z]?\d{0,3}){2,}(?:\([^\s\)]{1,5}\))?(?:[+\-]\d*)?\b")

# One-letter element symbols that were tagged in training.
ONE_LETTER_ELEMENTS = {
    "B",
    "C",
    "D",
    "F",
    "H",
    "I",
    "K",
    "N",
    "O",
    "P",
    "S",
    "V",
    "W",
}


@dataclass(frozen=True)
class Mention:
    start: int
    end: int


def load_annotations(paths: Sequence[Path]) -> Counter:
    counter: Counter = Counter()
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 6:
                    continue
                text = parts[4]
                counter[text] += 1
    return counter


def build_surface_list(counter: Counter) -> List[str]:
    surfaces: List[str] = []
    for text, cnt in counter.items():
        if len(text) < 2:
            continue
        if cnt < 2 and len(text) <= 3:
            continue
        surfaces.append(text)

    surfaces.sort(key=lambda x: (-len(x), x))
    return surfaces


def _find_all(text: str, sub: str) -> Iterable[Tuple[int, int]]:
    start = 0
    while True:
        idx = text.find(sub, start)
        if idx == -1:
            return
        yield idx, idx + len(sub)
        start = idx + 1


def _is_wordchar(ch: str) -> bool:
    return bool(ch) and bool(WORDCHAR_RE.match(ch))


def _boundary_ok(text: str, s: int, e: int) -> bool:
    if s > 0 and _is_wordchar(text[s - 1]) and _is_wordchar(text[s]):
        return False
    if e < len(text) and _is_wordchar(text[e - 1]) and _is_wordchar(text[e]):
        return False
    return True


def _trim_span(text: str, s: int, e: int) -> Tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while s < e and text[e - 1].isspace():
        e -= 1
    while s < e and text[s] in ",;:":
        s += 1
    while s < e and text[e - 1] in ",;:.":
        e -= 1
    return s, e


def predict_section(text: str, surfaces: Sequence[str]) -> List[Mention]:
    mentions: List[Mention] = []
    occupied = [False] * (len(text) + 1)

    def add_span(s: int, e: int) -> None:
        s2, e2 = _trim_span(text, s, e)
        if s2 >= e2:
            return
        if any(occupied[s2:e2]):
            return
        for i in range(s2, e2):
            occupied[i] = True
        mentions.append(Mention(s2, e2))

    # Regex passes (high precision-ish)
    for m in CAS_RE.finditer(text):
        add_span(m.start(), m.end())
    for m in INCHI_RE.finditer(text):
        add_span(m.start(), m.end())
    for m in ION_RE.finditer(text):
        add_span(m.start(), m.end())
    for m in FORMULA_RE.finditer(text):
        tok = m.group(0)
        # Avoid grabbing common capitalized words; require either digit or length>=4.
        if any(ch.isdigit() for ch in tok) or len(tok) >= 4:
            add_span(m.start(), m.end())

    # Dictionary (greedy by length)
    for surf in surfaces:
        for s, e in _find_all(text, surf):
            if not _boundary_ok(text, s, e):
                continue
            add_span(s, e)

    # One-letter element symbols (only when standalone)
    for i, ch in enumerate(text):
        if ch not in ONE_LETTER_ELEMENTS:
            continue
        if i > 0 and text[i - 1].isalnum():
            continue
        if i + 1 < len(text) and text[i + 1].isalnum():
            continue
        add_span(i, i + 1)

    mentions.sort(key=lambda m: (m.start, m.end))
    return mentions


def load_abstracts(path: Path) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            pmid, title, abstract = line.rstrip("\n").split("\t")
            rows.append((pmid, title, abstract))
    return rows


def write_predictions(
    abstracts_path: Path,
    train_ann_paths: Sequence[Path],
    out_path: Path,
) -> None:
    counter = load_annotations(train_ann_paths)
    surfaces = build_surface_list(counter)

    rows = load_abstracts(abstracts_path)
    out_lines: List[str] = []
    for pmid, title, abstract in rows:
        for sec, txt in (("T", title), ("A", abstract)):
            for m in predict_section(txt, surfaces):
                out_lines.append(f"{pmid}\t{sec}:{m.start}:{m.end}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)


if __name__ == "__main__":
    # Default run: generate predictions for the provided test set.
    base = Path(__file__).resolve().parent
    write_predictions(
        abstracts_path=base / "data" / "test.abstracts.txt",
        train_ann_paths=[
            base / "data" / "training.annotations.txt",
            base / "data" / "development.annotations.txt",
        ],
        out_path=base / "output" / "predictions.tsv",
    )
