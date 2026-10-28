#!/usr/bin/env python3
"""Rule/gazetteer-based chemical entity mention tagger.

This is a lightweight baseline built only from the provided training data.
It intentionally avoids any external chemical NER tools or pretrained models.

Usage:
  python3 cem_tagger.py \
    --train-abstracts data/training.abstracts.txt \
    --train-annotations data/training.annotations.txt \
    --test-abstracts data/test.abstracts.txt \
    --output output/predictions.tsv
"""

from __future__ import annotations

import argparse
import collections
import re
from typing import DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple


Span = Tuple[int, int]


def read_abstracts(path: str) -> Dict[str, Tuple[str, str]]:
    articles: Dict[str, Tuple[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            pmid, title, abstract = line.rstrip("\n").split("\t")
            articles[pmid] = (title, abstract)
    return articles


def iter_annotation_texts(path: str) -> Iterable[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            # PMID<TAB>section<TAB>start<TAB>end<TAB>text<TAB>class
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 6:
                continue
            yield parts[4]


def build_gazetteer(annotation_path: str, min_freq: int) -> Set[str]:
    freq: collections.Counter[str] = collections.Counter()
    for text in iter_annotation_texts(annotation_path):
        freq[text.lower()] += 1
    return {k for k, v in freq.items() if v >= min_freq}


def make_predictor(gazetteer: Set[str]):
    # A few high-precision regexes for chemistry-ish strings.
    formula_re = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
    ion_re = re.compile(r"\b[A-Z][a-z]?(?:\([IVX]+\))?[0-9]*[+-]\b")
    cas_re = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
    chebi_re = re.compile(r"\bCHEBI:\d+\b", re.I)
    pubchem_re = re.compile(r"\bCID:?\s*\d+\b", re.I)
    chembl_re = re.compile(r"\bChEMBL\d+\b", re.I)

    # Token pattern: letters/digits with common chemistry punctuation allowed.
    # Includes greek letters (often used in names like beta-lactam).
    tok_re = re.compile(r"[A-Za-z0-9α-ωΑ-Ω][A-Za-z0-9α-ωΑ-Ω+\-/'\(\)\[\]\.,]*")

    # Guideline-driven exclusions (very small list; prefer recall).
    bad_exact = {
        "water",
        "aqueous",
        "organic",
        "inorganic",
        "proton",
        "lead",
        "gold",
    }

    def is_bad(surface: str) -> bool:
        return surface.lower() in bad_exact

    def predict_spans(text: str) -> List[Span]:
        spans: List[Span] = []

        for r in (cas_re, chebi_re, pubchem_re, chembl_re, ion_re, formula_re):
            spans.extend((m.start(), m.end()) for m in r.finditer(text))

        toks = [(m.start(), m.end()) for m in tok_re.finditer(text)]
        nmax = 6
        i = 0
        while i < len(toks):
            found: Span | None = None
            for n in range(nmax, 0, -1):
                if i + n > len(toks):
                    continue
                s = toks[i][0]
                e = toks[i + n - 1][1]
                phrase = text[s:e]
                if phrase.lower() in gazetteer and not is_bad(phrase):
                    found = (s, e)
                    break

            if found is not None:
                spans.append(found)
                j = i
                while j < len(toks) and toks[j][0] < found[1]:
                    j += 1
                i = j
            else:
                i += 1

        return sorted(set(spans))

    return predict_spans


def write_predictions(
    articles: Dict[str, Tuple[str, str]],
    predict_spans,
    out_path: str,
) -> None:
    with open(out_path, "w", encoding="utf-8") as out:
        for pmid, (title, abstract) in articles.items():
            for sec, text in (("T", title), ("A", abstract)):
                for s, e in predict_spans(text):
                    out.write(f"{pmid}\t{sec}:{s}:{e}\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-abstracts", required=False)
    ap.add_argument("--train-annotations", required=True)
    ap.add_argument("--test-abstracts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-freq", type=int, default=4)
    args = ap.parse_args(argv)

    gazetteer = build_gazetteer(args.train_annotations, min_freq=args.min_freq)
    predict_spans = make_predictor(gazetteer)

    test_articles = read_abstracts(args.test_abstracts)
    write_predictions(test_articles, predict_spans, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

