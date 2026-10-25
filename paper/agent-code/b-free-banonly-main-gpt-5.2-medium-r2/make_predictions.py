#!/usr/bin/env python3
"""Train a span tagger on the provided data and predict on the test set.

This script is intentionally self-contained (no LLM calls, no external NER
packages/models). It trains a CRF sequence labeler using only the supplied
training/development annotations.

Output:
  /workspace/output/predictions.tsv
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import sklearn_crfsuite


# ---------------------------- Tokenization ----------------------------

BASE_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
AA3_DIG_RE = re.compile(r"^([A-Z][a-z]{2})(\d+)$")
AA3_DIG_AA3_RE = re.compile(r"^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")


@dataclass(frozen=True)
class Tok:
    text: str
    start: int
    end: int


def tokenize(text: str) -> List[Tok]:
    """Tokenize into small units to better align with gold boundaries."""
    toks: List[Tok] = []
    for m in BASE_TOKEN_RE.finditer(text):
        tok = m.group(0)
        start = m.start()

        m2 = AA3_DIG_AA3_RE.match(tok)
        if m2:
            p1, p2, p3 = m2.groups()
            a = start
            toks.append(Tok(p1, a, a + len(p1)))
            a += len(p1)
            toks.append(Tok(p2, a, a + len(p2)))
            a += len(p2)
            toks.append(Tok(p3, a, a + len(p3)))
            continue

        m1 = AA3_DIG_RE.match(tok)
        if m1:
            p1, p2 = m1.groups()
            a = start
            toks.append(Tok(p1, a, a + len(p1)))
            a += len(p1)
            toks.append(Tok(p2, a, a + len(p2)))
            continue

        toks.append(Tok(tok, m.start(), m.end()))
    return toks


# ---------------------------- Data loading ----------------------------


def load_abstracts(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pmid, title, abstract = line.rstrip("\n").split("\t")
            out[pmid] = {"T": title, "A": abstract}
    return out


def load_spans(path: str) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    out: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pmid, section, start, end, _txt, _cls = line.rstrip("\n").split("\t")
            out[(pmid, section)].append((int(start), int(end)))
    for k in list(out.keys()):
        out[k].sort()
    return out


# ---------------------------- Features ----------------------------


CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
CHEBI_RE = re.compile(r"^CHEBI:\d+$", re.IGNORECASE)
PUBCHEM_RE = re.compile(r"^(CID|SID):?\d+$", re.IGNORECASE)


def shape(token: str) -> str:
    out = []
    for ch in token:
        if ch.isupper():
            out.append("A")
        elif ch.islower():
            out.append("a")
        elif ch.isdigit():
            out.append("0")
        else:
            out.append(ch)
    # Compress repeats a bit to reduce sparsity
    s = "".join(out)
    s = re.sub(r"A{3,}", "AAA", s)
    s = re.sub(r"a{3,}", "aaa", s)
    s = re.sub(r"0{3,}", "000", s)
    return s


STOP_EXACT = {
    "water",
    "aqueous",
    "proton",
    "lead",
    "gold",
    "organic",
    "inorganic",
    "h2o",
}

ROLE_WORDS = {
    "analogue",
    "analog",
    "inhibitor",
    "agonist",
    "antagonist",
    "antibiotic",
    "hormone",
    "drug",
    "compound",
    "metabolite",
    "solvent",
    "reagent",
    "buffer",
    "catalyst",
    "toxin",
    "electrophile",
    "cation",
    "anion",
}


ELEMENTS = {
    # 1-2 letter element symbols (common ones; fine to include all)
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


def build_gazetteer(
    texts: Dict[str, Dict[str, str]],
    spans: Dict[Tuple[str, str], List[Tuple[int, int]]],
) -> Tuple[Counter[str], Counter[str]]:
    """Return (token_counter, full_mention_counter) from annotated spans."""
    token_counter: Counter[str] = Counter()
    mention_counter: Counter[str] = Counter()
    for (pmid, section), sps in spans.items():
        section_text = texts[pmid][section]
        for start, end in sps:
            mention = section_text[start:end]
            mention_counter[mention.lower()] += 1
            for t in tokenize(mention):
                if t.text.strip():
                    token_counter[t.text.lower()] += 1
    return token_counter, mention_counter


def token2features(tokens: Sequence[Tok], i: int, gaz_tokens: Counter[str]) -> Dict[str, object]:
    tok = tokens[i].text
    low = tok.lower()
    feats: Dict[str, object] = {
        "bias": 1.0,
        "tok": tok,
        "low": low,
        "shape": shape(tok),
        "len": len(tok),
        "is_upper": tok.isupper(),
        "is_title": tok.istitle(),
        "is_digit": tok.isdigit(),
        "has_digit": any(c.isdigit() for c in tok),
        "has_alpha": any(c.isalpha() for c in tok),
        "has_hyphen": "-" in tok,
        "has_slash": "/" in tok,
        "has_plus": "+" in tok,
        "has_hash": "#" in tok,
        "has_dot": "." in tok,
        "is_punct": bool(re.fullmatch(r"[^\w\s]", tok)),
        "gaz_tok": gaz_tokens[low] > 0,
        "gaz_tok_freq": min(gaz_tokens[low], 5),
        "is_stop": low in STOP_EXACT,
        "is_role": low in ROLE_WORDS,
        "is_cas": bool(CAS_RE.match(tok)),
        "is_chebi": bool(CHEBI_RE.match(tok)),
        "is_pubchem": bool(PUBCHEM_RE.match(tok)),
    }
    for n in (1, 2, 3, 4):
        if len(low) >= n:
            feats[f"pref{n}"] = low[:n]
            feats[f"suf{n}"] = low[-n:]
    # context window
    if i > 0:
        feats["-1:low"] = tokens[i - 1].text.lower()
        feats["-1:shape"] = shape(tokens[i - 1].text)
        feats["-1:is_punct"] = bool(re.fullmatch(r"[^\w\s]", tokens[i - 1].text))
    else:
        feats["BOS"] = True
    if i < len(tokens) - 1:
        feats["+1:low"] = tokens[i + 1].text.lower()
        feats["+1:shape"] = shape(tokens[i + 1].text)
        feats["+1:is_punct"] = bool(re.fullmatch(r"[^\w\s]", tokens[i + 1].text))
    else:
        feats["EOS"] = True
    return feats


def spans_to_labels(tokens: Sequence[Tok], spans: Sequence[Tuple[int, int]]) -> List[str]:
    labels = ["O"] * len(tokens)
    if not spans:
        return labels
    si = 0
    cur_start, cur_end = spans[si]
    for i, t in enumerate(tokens):
        # advance span pointer
        while si < len(spans) and t.start >= cur_end:
            si += 1
            if si < len(spans):
                cur_start, cur_end = spans[si]
        if si >= len(spans):
            break
        if t.start >= cur_start and t.end <= cur_end:
            labels[i] = "B" if t.start == cur_start else "I"
    return labels


def labels_to_spans(tokens: Sequence[Tok], labels: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = end = None
    for t, lab in zip(tokens, labels):
        if lab == "B" or (lab == "I" and start is None):
            if start is not None:
                spans.append((start, end))
            start, end = t.start, t.end
        elif lab == "I" and start is not None:
            end = t.end
        else:
            if start is not None:
                spans.append((start, end))
                start = end = None
    if start is not None:
        spans.append((start, end))
    return spans


# ---------------------------- Post-processing ----------------------------


TRAILING_CHARS = {"®", "™"}


def trim_span(text: str, start: int, end: int) -> Tuple[int, int]:
    # Trim trademark symbols
    while end > start and text[end - 1] in TRAILING_CHARS:
        end -= 1
    # Trim possessive 's / ’s
    if end - start >= 2 and text[end - 2 : end] in {"'s", "’s"}:
        end -= 2
    return start, end


def filter_span(text: str, start: int, end: int) -> bool:
    """Return True if span should be kept."""
    start, end = trim_span(text, start, end)
    if end <= start:
        return False
    s = text[start:end]
    low = s.lower()
    if low in STOP_EXACT:
        return False
    if low in ROLE_WORDS:
        return False
    if s.isdigit():
        return False
    # 1-2 chars: only keep if it's an element symbol or has charge/formula markers
    if len(s) <= 2:
        if s in ELEMENTS:
            return True
        if re.fullmatch(r"[A-Za-z][0-9]", s):
            return True
        return False
    return True


# ---------------------------- Gazetteer back-off ----------------------------


def iter_phrase_matches(text: str, phrase: str) -> Iterator[Tuple[int, int]]:
    """Case-insensitive substring matches with simple boundary checks.

    This is a deliberately cheap matcher; it assumes the phrase's whitespace is
    already normalized (as it is in this dataset).
    """
    if not phrase:
        return

    low_text = text.lower()
    start = 0
    while True:
        i = low_text.find(phrase, start)
        if i < 0:
            return
        a, b = i, i + len(phrase)
        before = text[a - 1] if a > 0 else " "
        after = text[b] if b < len(text) else " "
        if (before.isalnum() or before == "_") or (after.isalnum() or after == "_"):
            start = i + 1
            continue
        yield (a, b)
        start = i + 1


def add_gazetteer_spans(
    text: str,
    existing: Iterable[Tuple[int, int]],
    mention_counter: Counter[str],
) -> List[Tuple[int, int]]:
    existing_set = set(existing)
    out = set(existing_set)
    # Only use very frequent mentions, and cap the total scanned phrases.
    # (Scanning thousands of phrases per document is too slow.)
    phrases: List[Tuple[str, int]] = [
        (phrase, cnt) for phrase, cnt in mention_counter.items() if cnt >= 12 and len(phrase) >= 4
    ]
    phrases.sort(key=lambda x: (-x[1], -len(x[0])))
    for phrase, _cnt in phrases[:800]:
        for a, b in iter_phrase_matches(text, phrase):
            a, b = trim_span(text, a, b)
            if filter_span(text, a, b):
                out.add((a, b))
    return sorted(out)


# ---------------------------- Training / Prediction ----------------------------


def build_dataset(
    texts: Dict[str, Dict[str, str]],
    spans: Dict[Tuple[str, str], List[Tuple[int, int]]],
    pmids: Sequence[str],
    section: str,
    gaz_tokens: Counter[str],
) -> Tuple[List[List[Dict[str, object]]], List[List[str]], List[Tuple[str, str, List[Tok]]]]:
    X: List[List[Dict[str, object]]] = []
    y: List[List[str]] = []
    meta: List[Tuple[str, str, List[Tok]]] = []
    for pmid in pmids:
        txt = texts[pmid][section]
        toks = tokenize(txt)
        gold = spans.get((pmid, section), [])
        labels = spans_to_labels(toks, gold)
        feats = [token2features(toks, i, gaz_tokens) for i in range(len(toks))]
        X.append(feats)
        y.append(labels)
        meta.append((pmid, section, toks))
    return X, y, meta


def evaluate(
    crf: sklearn_crfsuite.CRF,
    X: Sequence[Sequence[Dict[str, object]]],
    meta: Sequence[Tuple[str, str, List[Tok]]],
    texts: Dict[str, Dict[str, str]],
    gold_spans: Dict[Tuple[str, str], List[Tuple[int, int]]],
) -> Tuple[float, float, float]:
    pred = crf.predict(X)
    gold_set = set()
    for (pmid, section), sps in gold_spans.items():
        for a, b in sps:
            gold_set.add((pmid, section, a, b))
    pred_set = set()
    for (pmid, section, toks), labs in zip(meta, pred):
        sec_txt = texts[pmid][section]
        for a, b in labels_to_spans(toks, labs):
            a, b = trim_span(sec_txt, a, b)
            if filter_span(sec_txt, a, b):
                pred_set.add((pmid, section, a, b))
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def train_and_predict(
    train_abs: str,
    train_ann: str,
    dev_abs: str,
    dev_ann: str,
    test_abs: str,
    out_path: str,
    quick_eval: bool,
) -> None:
    train_texts = load_abstracts(train_abs)
    dev_texts = load_abstracts(dev_abs)
    test_texts = load_abstracts(test_abs)

    train_spans = load_spans(train_ann)
    dev_spans = load_spans(dev_ann)

    all_train_texts = {**train_texts, **dev_texts}
    all_train_spans = dict(train_spans)
    for k, v in dev_spans.items():
        all_train_spans[k] = v

    gaz_tokens, _mention_counter = build_gazetteer(all_train_texts, all_train_spans)

    # Optional: quick eval to catch obvious mistakes
    if quick_eval:
        tr_pmids = sorted(train_texts.keys())
        dev_pmids = sorted(dev_texts.keys())

        X_tr_T, y_tr_T, _ = build_dataset(train_texts, train_spans, tr_pmids, "T", gaz_tokens)
        X_tr_A, y_tr_A, _ = build_dataset(train_texts, train_spans, tr_pmids, "A", gaz_tokens)
        X_tr = X_tr_T + X_tr_A
        y_tr = y_tr_T + y_tr_A

        X_dev_T, y_dev_T, meta_dev_T = build_dataset(dev_texts, dev_spans, dev_pmids, "T", gaz_tokens)
        X_dev_A, y_dev_A, meta_dev_A = build_dataset(dev_texts, dev_spans, dev_pmids, "A", gaz_tokens)
        X_dev = X_dev_T + X_dev_A
        meta_dev = meta_dev_T + meta_dev_A

        crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=120,
            all_possible_transitions=True,
        )
        crf.fit(X_tr, y_tr)
        p, r, f1 = evaluate(crf, X_dev, meta_dev, dev_texts, dev_spans)
        print(f"DEV span P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # Final model on train+dev
    all_pmids = sorted(all_train_texts.keys())
    X_all_T, y_all_T, _ = build_dataset(all_train_texts, all_train_spans, all_pmids, "T", gaz_tokens)
    X_all_A, y_all_A, _ = build_dataset(all_train_texts, all_train_spans, all_pmids, "A", gaz_tokens)
    X_all = X_all_T + X_all_A
    y_all = y_all_T + y_all_A

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.08,
        c2=0.12,
        max_iterations=160,
        all_possible_transitions=True,
    )
    crf.fit(X_all, y_all)

    # Predict test
    test_pmids = sorted(test_texts.keys())
    preds: List[Tuple[str, str, int, int]] = []
    for section in ("T", "A"):
        X_test, _y_dummy, meta = build_dataset(test_texts, {}, test_pmids, section, gaz_tokens)
        y_hat = crf.predict(X_test)
        for (pmid, sec, toks), labs in zip(meta, y_hat):
            sec_txt = test_texts[pmid][sec]
            spans = []
            for a, b in labels_to_spans(toks, labs):
                a, b = trim_span(sec_txt, a, b)
                if filter_span(sec_txt, a, b):
                    spans.append((a, b))
            for a, b in spans:
                preds.append((pmid, sec, a, b))

    preds.sort(key=lambda x: (int(x[0]), x[1], x[2], x[3]))
    with open(out_path, "w", encoding="utf-8") as f:
        for pmid, sec, a, b in preds:
            f.write(f"{pmid}\t{sec}:{a}:{b}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick-eval", action="store_true", help="print a quick dev-set score")
    ap.add_argument(
        "--out",
        default="/workspace/output/predictions.tsv",
        help="output TSV path",
    )
    args = ap.parse_args()

    train_and_predict(
        train_abs="/workspace/data/training.abstracts.txt",
        train_ann="/workspace/data/training.annotations.txt",
        dev_abs="/workspace/data/development.abstracts.txt",
        dev_ann="/workspace/data/development.annotations.txt",
        test_abs="/workspace/data/test.abstracts.txt",
        out_path=args.out,
        quick_eval=args.quick_eval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

