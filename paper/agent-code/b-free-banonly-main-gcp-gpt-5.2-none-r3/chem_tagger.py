#!/usr/bin/env python3
"""Rule-based chemical mention tagger.

Constraints: no external NER/LLM; uses training data only.

Approach:
- Build a gazetteer of mention strings from training annotations.
- Find exact (case-sensitive) occurrences in text.
- Add regex-based candidates for common chemical-like strings (formulae, ions, element symbols in context, common suffixes).
- Postprocess: drop blacklisted role words; trim trailing punctuation; dedupe overlaps preferring longer spans.

This is intentionally simple and fast.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class Span:
    start: int
    end: int


ROLE_WORDS = {
    # guidelines: role not structure
    "analogue",
    "analogues",
    "inhibitor",
    "inhibitors",
    "agonist",
    "agonists",
    "antagonist",
    "antagonists",
    "antioxidant",
    "antioxidants",
    "drug",
    "drugs",
    "compound",
    "compounds",
    "hormone",
    "hormones",
    "antibiotic",
    "antibiotics",
    "toxin",
    "toxins",
    "metabolite",
    "metabolites",
    "solvent",
    "solvents",
    "reagent",
    "reagents",
    "buffer",
    "buffers",
    "catalyst",
    "catalysts",
    "electrophile",
    "electrophiles",
    "cation",
    "cations",
    "anion",
    "anions",
    "acid",
    "base",
    "salt",
    "metal",
    # non-specific structural concepts
    "atom",
    "atoms",
    "ion",
    "ions",
    "molecule",
    "molecules",
    "polymer",
    "polymers",
    "isomer",
    "isomers",
    "enantiomer",
    "enantiomers",
    "monomer",
    "monomers",
    "dimer",
    "dimers",
    "lipid",
    "lipids",
    "protein",
    "proteins",
    "functional group",
}

# by convention: do not tag these when standalone
NEVER_TAG_STANDALONE = {"water", "aqueous", "proton", "lead", "gold"}

# allow iron explicitly (guidelines)

# Chemical-like suffixes (very rough)
SUFFIX_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9]*)(?:"
    r"ol|one|ene|yne|ane|ase|ide|ate|ite|ium|amine|amide|ether|ester|acid|salt|metal|oxide|hydroxide|chloride|bromide|fluoride|iodide|nitrate|sulfate|phosphate|carbonate"
    r")\b"
)

# Element symbols list (subset helpful); include single-letter ones too but filter later
ELEMENTS = {
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "K",
    "V",
    "Y",
    "I",
    "W",
    "U",
    "He",
    "Li",
    "Be",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "Cl",
    "Ar",
    "Ca",
    "Sc",
    "Ti",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Zr",
    "Mo",
    "Ag",
    "Cd",
    "Sn",
    "Sb",
    "Te",
    "Xe",
    "Cs",
    "Ba",
    "Pt",
    "Au",
    "Hg",
    "Pb",
    "Bi",
}

# Formulas / ions like Fe2+, Cu(II), NO, NAD+, C6H12O6, (14)C
# Keep this conservative; overly broad all-caps matching hurts precision.
FORMULA_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"\([0-9]{1,3}\)\s*[A-Z][a-z]?"  # (14)C, (2)H
    r"|(?:[A-Z][a-z]?(?:[0-9]{0,3}))+[0-9]{0,3}(?:[+-]{1,2}|\([IVX]{1,4}\))"  # require charge/oxidation
    r"|[A-Z]{2,5}[+-]"  # require charge for all-caps tokens (e.g., NAD+)
    r")(?!(?:[A-Za-z0-9]))"
)

# CAS number
CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")

# simple InChI token
INCHI_RE = re.compile(r"\bInChI=[^\s]+")

# Don't include trailing punctuation/brackets etc.
TRIM_RE = re.compile(r"[\s\]\[\)\(\.,;:]+$")


def read_abstracts(path: Path) -> List[Tuple[str, str, str]]:
    rows = []
    with path.open(encoding="utf8") as f:
        for line in f:
            pmid, title, abstract = line.rstrip("\n").split("\t")
            rows.append((pmid, title, abstract))
    return rows


def read_annotations(path: Path) -> List[Tuple[str, str, int, int, str, str]]:
    out = []
    with path.open(encoding="utf8") as f:
        for line in f:
            pmid, sec, start, end, text, cls = line.rstrip("\n").split("\t")
            out.append((pmid, sec, int(start), int(end), text, cls))
    return out


def build_gazetteer(train_ann: Sequence[Tuple[str, str, int, int, str, str]]) -> Set[str]:
    # Keep as-is (case sensitive) plus simple variants (strip trademark, possessive already excluded by gold)
    g = set()
    for _pmid, _sec, _s, _e, text, _cls in train_ann:
        if not text:
            continue
        g.add(text)
    # remove leading/trailing spaces just in case
    g = {t.strip() for t in g if t.strip()}
    return g


def is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch in {"_"}


def find_all(text: str, needle: str) -> Iterable[Span]:
    # naive search; faster than regex compilation for many needles would be Aho-Corasick, but keep simple.
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            return
        yield Span(idx, idx + len(needle))
        start = idx + 1


def token_boundary_ok(text: str, s: int, e: int) -> bool:
    # Ensure we don't match inside a larger word (guidelines: don't break a word)
    if s > 0 and is_word_char(text[s - 1]) and is_word_char(text[s]):
        return False
    if e < len(text) and is_word_char(text[e - 1]) and is_word_char(text[e]):
        return False
    return True


def normalize_for_filter(span_text: str) -> str:
    return span_text.strip().strip("'\"").lower()


def filter_span(text: str, s: int, e: int) -> Tuple[int, int] | None:
    # Trim trailing punctuation/spaces but keep internal brackets as per guideline.
    sub = text[s:e]
    # Never include leading/trailing spaces
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    # trim trailing punctuation (.,;:) and closing parens/brackets if not balanced? keep simple
    while e > s and text[e - 1] in {".", ",", ";", ":"}:
        e -= 1
    if e <= s:
        return None
    sub = text[s:e]
    low = normalize_for_filter(sub)
    if low in {"dna", "rna", "mrna", "cdna"}:
        return None
    if low in ROLE_WORDS:
        return None
    if low in NEVER_TAG_STANDALONE:
        return None
    # exclude pure numbers
    if re.fullmatch(r"\d+", sub):
        return None
    # exclude 1-letter acronyms unless element symbol.
    # Also exclude 2-letter tokens unless element symbol or common small-gas tokens.
    if len(sub) == 1 and sub.isalpha() and sub not in ELEMENTS:
        return None
    if len(sub) == 2 and sub.isalpha() and sub not in ELEMENTS and sub not in {"NO", "CO", "OH"}:
        return None
    return (s, e)


def merge_spans(spans: List[Span]) -> List[Span]:
    # prefer longer spans; remove overlaps by greedy on length then start
    spans_sorted = sorted(spans, key=lambda sp: (-(sp.end - sp.start), sp.start, sp.end))
    kept: List[Span] = []
    occupied: List[Tuple[int, int]] = []
    for sp in spans_sorted:
        overlap = False
        for ks in kept:
            if not (sp.end <= ks.start or sp.start >= ks.end):
                overlap = True
                break
        if not overlap:
            kept.append(sp)
    return sorted(kept, key=lambda sp: (sp.start, sp.end))


def predict_spans(text: str, gazetteer: Sequence[str]) -> List[Span]:
    spans: List[Span] = []

    # Gazetteer exact matches
    for term in gazetteer:
        # skip very long terms to keep runtime reasonable? keep all but require >=2 chars
        if len(term) < 2:
            continue
        for sp in find_all(text, term):
            if token_boundary_ok(text, sp.start, sp.end):
                filtered = filter_span(text, sp.start, sp.end)
                if filtered:
                    spans.append(Span(*filtered))

    # Regex candidates
    for regex in (CAS_RE, INCHI_RE, FORMULA_RE, SUFFIX_RE):
        for m in regex.finditer(text):
            s, e = m.start(), m.end()
            if not token_boundary_ok(text, s, e):
                continue
            filtered = filter_span(text, s, e)
            if filtered:
                spans.append(Span(*filtered))

    return merge_spans(spans)


def score_f1(
    pred: Dict[Tuple[str, str], Set[Tuple[int, int]]],
    gold: Dict[Tuple[str, str], Set[Tuple[int, int]]],
) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    keys = set(pred) | set(gold)
    for k in keys:
        p = pred.get(k, set())
        g = gold.get(k, set())
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def gold_map(ann: Sequence[Tuple[str, str, int, int, str, str]]) -> Dict[Tuple[str, str], Set[Tuple[int, int]]]:
    g: Dict[Tuple[str, str], Set[Tuple[int, int]]] = {}
    for pmid, sec, s, e, _t, _cls in ann:
        g.setdefault((pmid, sec), set()).add((s, e))
    return g


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ann", default="data/training.annotations.txt")
    ap.add_argument("--dev-abs", default="data/development.abstracts.txt")
    ap.add_argument("--dev-ann", default="data/development.annotations.txt")
    ap.add_argument("--test-abs", default="data/test.abstracts.txt")
    ap.add_argument("--out", default="output/predictions.tsv")
    ap.add_argument("--no-regex", action="store_true")
    ap.add_argument("--eval-dev", action="store_true")
    args = ap.parse_args()

    train_ann = read_annotations(Path(args.train_ann))
    gaz = sorted(build_gazetteer(train_ann), key=len, reverse=True)

    if args.eval_dev:
        dev_abs = read_abstracts(Path(args.dev_abs))
        dev_gold = gold_map(read_annotations(Path(args.dev_ann)))
        pred: Dict[Tuple[str, str], Set[Tuple[int, int]]] = {}
        for pmid, title, abstract in dev_abs:
            for sec, txt in (('T', title), ('A', abstract)):
                spans = []
                # use gaz always
                spans.extend(predict_spans(txt, gaz))
                pred[(pmid, sec)] = {(sp.start, sp.end) for sp in spans}
        p, r, f1 = score_f1(pred, dev_gold)
        print(f"DEV precision={p:.4f} recall={r:.4f} f1={f1:.4f} (gazetteer+regex)")

    # predict test
    test_abs = read_abstracts(Path(args.test_abs))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf8') as out:
        for pmid, title, abstract in test_abs:
            for sec, txt in (('T', title), ('A', abstract)):
                spans = predict_spans(txt, gaz)
                for sp in spans:
                    out.write(f"{pmid}\t{sec}:{sp.start}:{sp.end}\n")


if __name__ == "__main__":
    main()
