#!/usr/bin/env python3
"""Train a CRF tagger on the provided train+dev data and predict mentions on test.

Outputs:
  /workspace/output/predictions.tsv

Notes:
- Uses only the provided dataset annotations for supervision.
- Does not call any external APIs.
"""

from __future__ import annotations

import os
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pycrfsuite


DATA_DIR = Path("/workspace/data")
OUT_DIR = Path("/workspace/output")

TRAIN_ABSTRACTS = DATA_DIR / "training.abstracts.txt"
TRAIN_ANN = DATA_DIR / "training.annotations.txt"
DEV_ABSTRACTS = DATA_DIR / "development.abstracts.txt"
DEV_ANN = DATA_DIR / "development.annotations.txt"
TEST_ABSTRACTS = DATA_DIR / "test.abstracts.txt"

MODEL_PATH = OUT_DIR / "chemner.crfsuite"
PRED_PATH = OUT_DIR / "predictions.tsv"


# Tokenization is intentionally simple but Unicode-aware.
# - "\w+" captures letters/digits across languages (and underscores).
# - Remaining non-space characters are emitted as single-character tokens.
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


ELEMENTS = {
    # 1–118, plus common one-letter element symbols used in text.
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
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
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
}


# By-convention exclusions from the guideline.
HARD_EXCLUDE = {
    "water",
    "aqueous",
    "proton",
    "organic",
    "inorganic",
    "lead",
    "gold",
}


CAS_RE = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
CHEBI_RE = re.compile(r"\b(?:CHEBI|ChEBI):\d+\b")
INCHI_RE = re.compile(r"\bInChI=[^\s]+")


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Mention:
    start: int
    end: int


def read_abstracts(path: Path) -> Dict[str, Dict[str, str]]:
    """Return pmid -> {"T": title, "A": abstract}."""
    out: Dict[str, Dict[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            pmid, title, abstract = line.split("\t")
            out[pmid] = {"T": title, "A": abstract}
    return out


def read_annotations(path: Path) -> Dict[Tuple[str, str], List[Mention]]:
    """Return (pmid, section) -> list of mentions."""
    out: Dict[Tuple[str, str], List[Mention]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            pmid, section, start, end, _text, _cls = line.split("\t")
            out[(pmid, section)].append(Mention(int(start), int(end)))
    for k in out:
        out[k].sort(key=lambda m: (m.start, m.end))
    return out


def tokenize(text: str) -> List[Token]:
    return [Token(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def _shape(token: str) -> str:
    mapped = []
    for ch in token:
        if ch.isupper():
            mapped.append("X")
        elif ch.islower():
            mapped.append("x")
        elif ch.isdigit():
            mapped.append("d")
        else:
            mapped.append(ch)
    # compress repeats
    out = []
    for ch in mapped:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def _contains_any(s: str, chars: str) -> bool:
    return any(c in s for c in chars)


def build_lexicons(
    docs: Dict[str, Dict[str, str]],
    anns: Dict[Tuple[str, str], List[Mention]],
) -> Tuple[Counter[str], Dict[str, float]]:
    """Build token frequency and a per-token chemical probability estimate."""
    total = Counter()
    chem = Counter()
    for pmid, sec2txt in docs.items():
        for sec, text in sec2txt.items():
            toks = tokenize(text)
            for t in toks:
                total[t.text.lower()] += 1
            for m in anns.get((pmid, sec), []):
                mention_text = text[m.start : m.end]
                for t in tokenize(mention_text):
                    chem[t.text.lower()] += 1

    prob: Dict[str, float] = {}
    for w, c in chem.items():
        prob[w] = c / max(1, total.get(w, 0))
    return total, prob


def label_tokens(tokens: List[Token], mentions: Sequence[Mention]) -> List[str]:
    labels = ["O"] * len(tokens)
    if not mentions:
        return labels

    # Mentions are non-overlapping in this dataset.
    mi = 0
    for ti, tok in enumerate(tokens):
        while mi < len(mentions) and mentions[mi].end <= tok.start:
            mi += 1
        if mi >= len(mentions):
            break
        m = mentions[mi]
        if tok.start >= m.start and tok.end <= m.end:
            labels[ti] = "B" if tok.start == m.start else "I"
    # Repair any stray I's.
    for i in range(len(labels)):
        if labels[i] == "I" and (i == 0 or labels[i - 1] == "O"):
            labels[i] = "B"
    return labels


def token_features(
    tokens: List[Token],
    i: int,
    section: str,
    freq: Counter[str],
    chem_prob: Dict[str, float],
) -> Dict[str, str]:
    tok = tokens[i].text
    low = tok.lower()
    sh = _shape(tok)

    feats: Dict[str, str] = {
        "bias": "1",
        "sec": section,
        "shape": sh,
        "len": str(min(len(tok), 10)),
    }

    # Token identity (pruned).
    if freq.get(low, 0) >= 2:
        feats["w"] = low
    else:
        feats["w"] = "<RARE>"

    # Character property features.
    feats["is_punct"] = "1" if (len(tok) == 1 and not tok.isalnum()) else "0"
    feats["is_digit"] = "1" if tok.isdigit() else "0"
    feats["has_digit"] = "1" if any(c.isdigit() for c in tok) else "0"
    feats["has_alpha"] = "1" if any(c.isalpha() for c in tok) else "0"
    feats["has_upper"] = "1" if any(c.isupper() for c in tok) else "0"
    feats["has_lower"] = "1" if any(c.islower() for c in tok) else "0"
    feats["has_dash"] = "1" if "-" in tok else "0"
    feats["has_slash"] = "1" if "/" in tok else "0"
    feats["has_plus"] = "1" if "+" in tok else "0"
    feats["has_comma"] = "1" if "," in tok else "0"
    feats["has_paren"] = "1" if _contains_any(tok, "()[]{}") else "0"
    feats["in_excl"] = "1" if low in HARD_EXCLUDE else "0"

    # Prefix/suffix hints.
    for n in (1, 2, 3, 4):
        if len(low) >= n:
            feats[f"pre{n}"] = low[:n]
            feats[f"suf{n}"] = low[-n:]

    # Chemical-leaning token patterns.
    feats["mix_alnum"] = "1" if (any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok)) else "0"
    feats["looks_cas"] = "1" if re.fullmatch(r"\d{2,7}-\d{2}-\d", tok) else "0"
    feats["looks_id"] = "1" if re.fullmatch(r"(CHEBI:|ChEBI:|CID)\d+", tok, flags=re.IGNORECASE) else "0"

    # Learned lexicon features from training.
    p = chem_prob.get(low)
    if p is None:
        feats["chem_p"] = "0"
    elif p >= 0.8:
        feats["chem_p"] = ">=0.8"
    elif p >= 0.5:
        feats["chem_p"] = ">=0.5"
    elif p >= 0.2:
        feats["chem_p"] = ">=0.2"
    else:
        feats["chem_p"] = "<0.2"

    # Special-casing element symbols.
    feats["is_element"] = "1" if tok in ELEMENTS else "0"

    # Local context.
    if i > 0:
        p_tok = tokens[i - 1].text
        p_low = p_tok.lower()
        feats["-1:w"] = p_low if freq.get(p_low, 0) >= 2 else "<RARE>"
        feats["-1:shape"] = _shape(p_tok)
        feats["w-1+w0"] = f"{feats['-1:w']}|{feats['w']}"
    else:
        feats["BOS"] = "1"
    if i + 1 < len(tokens):
        n_tok = tokens[i + 1].text
        n_low = n_tok.lower()
        feats["+1:w"] = n_low if freq.get(n_low, 0) >= 2 else "<RARE>"
        feats["+1:shape"] = _shape(n_tok)
        feats["w0+w+1"] = f"{feats['w']}|{feats['+1:w']}"
    else:
        feats["EOS"] = "1"

    if i > 1:
        p2_tok = tokens[i - 2].text
        p2_low = p2_tok.lower()
        feats["-2:w"] = p2_low if freq.get(p2_low, 0) >= 2 else "<RARE>"
        feats["-2:shape"] = _shape(p2_tok)
    if i + 2 < len(tokens):
        n2_tok = tokens[i + 2].text
        n2_low = n2_tok.lower()
        feats["+2:w"] = n2_low if freq.get(n2_low, 0) >= 2 else "<RARE>"
        feats["+2:shape"] = _shape(n2_tok)

    return feats


def featurize(
    text: str,
    section: str,
    freq: Counter[str],
    chem_prob: Dict[str, float],
) -> Tuple[List[Token], List[Dict[str, str]]]:
    tokens = tokenize(text)
    feats = [token_features(tokens, i, section, freq, chem_prob) for i in range(len(tokens))]
    return tokens, feats


def spans_from_bio(tokens: List[Token], labels: Sequence[str]) -> List[Mention]:
    out: List[Mention] = []
    i = 0
    while i < len(tokens):
        lab = labels[i]
        if lab not in ("B", "I"):
            i += 1
            continue
        start = tokens[i].start
        end = tokens[i].end
        j = i + 1
        while j < len(tokens) and labels[j] == "I":
            end = tokens[j].end
            j += 1
        out.append(Mention(start, end))
        i = j
    return out


def normalize_predicted_spans(text: str, spans: List[Mention]) -> List[Mention]:
    """Heuristic post-processing to better match the annotation guidelines."""

    def strip_span(s: int, e: int) -> Tuple[int, int]:
        # Remove leading/trailing whitespace (model can include them via punctuation tokens).
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1

        # Remove leading "non-" (guideline says to exclude "non-").
        if text[s : min(e, s + 4)].lower() == "non-":
            s += 4

        # Trim possessive and trailing hyphen.
        if e - s >= 2 and text[e - 2 : e].lower() == "'s":
            e -= 2
        if e > s and text[e - 1] == "-":
            e -= 1

        # Trim trademark symbols if they slip in.
        while e > s and text[e - 1] in "®™©":
            e -= 1
        return s, e

    merged: List[Mention] = []
    for m in spans:
        s, e = strip_span(m.start, m.end)
        if e <= s:
            continue
        lowered = text[s:e].strip().lower()
        if lowered in HARD_EXCLUDE:
            continue

        # Drop 1–2 letter all-caps acronyms unless they are element symbols.
        piece = text[s:e]
        if len(piece) <= 2 and piece.isalpha() and piece.upper() == piece:
            if piece not in ELEMENTS:
                continue
        merged.append(Mention(s, e))

    # Deduplicate and sort.
    merged = sorted(set(merged), key=lambda x: (x.start, x.end))
    return merged


def train_crf(
    train_docs: Dict[str, Dict[str, str]],
    train_anns: Dict[Tuple[str, str], List[Mention]],
    freq: Counter[str],
    chem_prob: Dict[str, float],
) -> pycrfsuite.Tagger:
    trainer = pycrfsuite.Trainer(verbose=False)

    # Add training instances.
    for pmid, sec2txt in train_docs.items():
        for sec, text in sec2txt.items():
            tokens, feats = featurize(text, sec, freq, chem_prob)
            labels = label_tokens(tokens, train_anns.get((pmid, sec), []))
            trainer.append(feats, labels)

    trainer.set_params(
        {
            "c1": 0.15,
            "c2": 0.05,
            "max_iterations": 200,
            "feature.possible_transitions": True,
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.train(str(MODEL_PATH))

    tagger = pycrfsuite.Tagger()
    tagger.open(str(MODEL_PATH))
    return tagger


def predict_mentions(
    tagger: pycrfsuite.Tagger,
    docs: Dict[str, Dict[str, str]],
    freq: Counter[str],
    chem_prob: Dict[str, float],
) -> List[Tuple[str, str, int, int]]:
    preds: List[Tuple[str, str, int, int]] = []
    for pmid, sec2txt in docs.items():
        for sec, text in sec2txt.items():
            tokens, feats = featurize(text, sec, freq, chem_prob)
            if not tokens:
                continue
            labels = tagger.tag(feats)
            spans = spans_from_bio(tokens, labels)
            spans = normalize_predicted_spans(text, spans)

            # Add a few very high-precision identifier-like patterns.
            for rx in (CAS_RE, CHEBI_RE, INCHI_RE):
                for m in rx.finditer(text):
                    spans.append(Mention(m.start(), m.end()))
            spans = normalize_predicted_spans(text, spans)

            for m in spans:
                # Final sanity check: offsets must be inside section.
                if 0 <= m.start < m.end <= len(text):
                    preds.append((pmid, sec, m.start, m.end))
    preds.sort(key=lambda x: (int(x[0]), x[1], x[2], x[3]))
    return preds


def write_predictions(preds: Sequence[Tuple[str, str, int, int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pmid, sec, start, end in preds:
            f.write(f"{pmid}\t{sec}:{start}:{end}\n")


def main() -> int:
    # Train on train + dev, because the goal is best test performance.
    train_docs = read_abstracts(TRAIN_ABSTRACTS)
    dev_docs = read_abstracts(DEV_ABSTRACTS)
    train_anns = read_annotations(TRAIN_ANN)
    dev_anns = read_annotations(DEV_ANN)

    all_docs = dict(train_docs)
    all_docs.update(dev_docs)
    all_anns = dict(train_anns)
    all_anns.update(dev_anns)

    freq, chem_prob = build_lexicons(all_docs, all_anns)
    tagger = train_crf(all_docs, all_anns, freq, chem_prob)

    test_docs = read_abstracts(TEST_ABSTRACTS)
    preds = predict_mentions(tagger, test_docs, freq, chem_prob)
    write_predictions(preds, PRED_PATH)
    print(f"Wrote {len(preds)} predictions to {PRED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

