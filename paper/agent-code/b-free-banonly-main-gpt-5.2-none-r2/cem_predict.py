#!/usr/bin/env python3
"""Dictionary-based chemical entity mention finder.

Rules:
- No external NER models.
- Learns mention surface forms from training annotations only.
- Predicts by exact (case-insensitive) surface form matching with
  conservative boundary checks.

Writes predictions for test abstracts to output/predictions.tsv.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


def load_abstracts(path: Path) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            pmid, title, abstract = line.rstrip("\n").split("\t")
            rows[pmid] = {"T": title, "A": abstract}
    return rows


def iter_training_mentions(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue
            text = parts[4]
            yield text


STOP_MENTIONS = {
    # Explicitly excluded in guidelines.
    "lead",
    "gold",
    "water",
    "aqueous",
    "organic",
    "inorganic",
    "proton",
}


def build_dictionary(
    training_annotations: Path,
    min_single_freq: int = 4,
) -> List[str]:
    freq: Counter[str] = Counter()
    mentions: Set[str] = set()
    for m in iter_training_mentions(training_annotations):
        ml = m.lower()
        mentions.add(ml)
        freq[ml] += 1

    dictionary: List[str] = []
    for m in mentions:
        if m in STOP_MENTIONS:
            continue
        if len(m) <= 1:
            continue
        if " " in m:
            dictionary.append(m)
        else:
            if freq[m] >= min_single_freq:
                dictionary.append(m)

    # Longest-first matching within each first-character bucket.
    dictionary.sort(key=len, reverse=True)
    return dictionary


def build_patterns(dictionary: Sequence[str]) -> Dict[str, re.Pattern[str]]:
    by_first: Dict[str, List[str]] = defaultdict(list)
    for m in dictionary:
        by_first[m[0]].append(m)

    patterns: Dict[str, re.Pattern[str]] = {}
    for ch, ms in by_first.items():
        ms = sorted(ms, key=len, reverse=True)
        alt = "|".join(re.escape(x) for x in ms)
        # Custom boundaries: avoid matching inside alphanumeric runs.
        patterns[ch] = re.compile(r"(?i)(?<![A-Za-z0-9])(?:" + alt + r")(?![A-Za-z0-9])")
    return patterns


def predict_spans(text: str, patterns: Dict[str, re.Pattern[str]]) -> Set[Tuple[int, int]]:
    spans: Set[Tuple[int, int]] = set()
    lower = text.lower()
    i = 0
    n = len(text)
    while i < n:
        ch = lower[i]
        pat = patterns.get(ch)
        if pat is not None:
            m = pat.match(lower, i)
            if m is not None:
                spans.add((m.start(), m.end()))
                i = m.end()
                continue
        i += 1
    return spans


def main() -> None:
    root = Path("/workspace")
    train_ann = root / "data" / "training.annotations.txt"
    test_abs = root / "data" / "test.abstracts.txt"
    out_path = root / "output" / "predictions.tsv"

    dictionary = build_dictionary(train_ann, min_single_freq=4)
    patterns = build_patterns(dictionary)

    test_rows = load_abstracts(test_abs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for pmid, secs in test_rows.items():
            for sec in ("T", "A"):
                text = secs[sec]
                spans = sorted(predict_spans(text, patterns))
                for s, e in spans:
                    out.write(f"{pmid}\t{sec}:{s}:{e}\n")


if __name__ == "__main__":
    main()
