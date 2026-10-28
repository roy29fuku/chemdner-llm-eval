#!/usr/bin/env python3
"""Chemical entity mention recognition for PubMed titles/abstracts.

This trains a CRF sequence tagger from the provided training/dev annotations and
writes test predictions in the required format.

Design goals:
- No external chemical NER tools or LLMs.
- Exact character offsets in the original text.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


try:
    import sklearn_crfsuite  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency sklearn-crfsuite. Install with: pip install sklearn-crfsuite"
    ) from exc


Token = Tuple[str, int, int]  # (text, start, end)


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclasses.dataclass(frozen=True)
class SectionExample:
    pmid: str
    section: str  # 'T' or 'A'
    text: str
    tokens: List[Token]
    labels: Optional[List[str]] = None


def read_abstracts(path: str) -> Dict[str, Tuple[str, str]]:
    articles: Dict[str, Tuple[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no}: expected 3 tab-separated fields")
            pmid, title, abstract = parts
            articles[pmid] = (title, abstract)
    return articles


def read_annotations(path: str) -> DefaultDict[Tuple[str, str], List[Tuple[int, int]]]:
    ann: DefaultDict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) != 6:
                raise ValueError(f"{path}:{line_no}: expected 6 tab-separated fields")
            pmid, section, start_s, end_s, _text, _cls = parts
            if section not in ("T", "A"):
                raise ValueError(f"{path}:{line_no}: unknown section {section!r}")
            start = int(start_s)
            end = int(end_s)
            if end <= start:
                continue
            ann[(pmid, section)].append((start, end))

    for key in list(ann.keys()):
        ann[key].sort()
    return ann


def tokenize(text: str) -> List[Token]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def word_shape(w: str) -> str:
    out = []
    last = ""
    for ch in w:
        if ch.isupper():
            c = "A"
        elif ch.islower():
            c = "a"
        elif ch.isdigit():
            c = "0"
        else:
            c = "x"
        if c != last:
            out.append(c)
            last = c
    return "".join(out)[:8]


def token_features(tokens: Sequence[Token], i: int) -> Dict[str, object]:
    w, _s, _e = tokens[i]
    wl = w.lower()

    def has(re_pat: str) -> bool:
        return re.search(re_pat, w) is not None

    feats: Dict[str, object] = {
        "bias": 1.0,
        "w": w,
        "wl": wl,
        "len": len(w),
        "shape": word_shape(w),
        "is_upper": w.isupper(),
        "is_title": w.istitle(),
        "is_digit": w.isdigit(),
        "has_digit": any(ch.isdigit() for ch in w),
        "has_alpha": any(ch.isalpha() for ch in w),
        "has_hyphen": "-" in w,
        "has_plus": "+" in w,
        "has_minus": "-" in w,
        "has_slash": "/" in w,
        "has_paren": any(ch in "()[]{}" for ch in w),
        "pref1": w[:1],
        "pref2": w[:2],
        "pref3": w[:3],
        "suf1": w[-1:],
        "suf2": w[-2:],
        "suf3": w[-3:],
        # Rough formula-like signals.
        "re_formula": bool(re.fullmatch(r"(?:[A-Z][a-z]?\d*){2,}", w)),
        "re_charge": bool(re.search(r"(?:\+|\-){1,3}$", w)),
        "re_cas": bool(re.fullmatch(r"\d{2,7}-\d{2}-\d", w)),
        # Chemistry-ish suffixes.
        "suf_ol": wl.endswith("ol"),
        "suf_ate": wl.endswith("ate"),
        "suf_ine": wl.endswith("ine"),
        "suf_ide": wl.endswith("ide"),
        "suf_ic": wl.endswith("ic"),
        "suf_one": wl.endswith("one"),
        "suf_ium": wl.endswith("ium"),
        "suf_ane": wl.endswith("ane"),
        "suf_ene": wl.endswith("ene"),
        "suf_yne": wl.endswith("yne"),
        "suf_ase": wl.endswith("ase"),
        # Bracketed stereochem and related.
        "re_stereo": bool(re.fullmatch(r"\([RSZErsze]\)", w)),
        "re_oxid": bool(re.fullmatch(r"\([IVX]+\)", w)),
    }

    if i > 0:
        pw = tokens[i - 1][0]
        feats.update(
            {
                "-1:w": pw,
                "-1:wl": pw.lower(),
                "-1:shape": word_shape(pw),
                "-1:is_punct": (not any(ch.isalnum() for ch in pw)),
            }
        )
    else:
        feats["BOS"] = True

    if i < len(tokens) - 1:
        nw = tokens[i + 1][0]
        feats.update(
            {
                "+1:w": nw,
                "+1:wl": nw.lower(),
                "+1:shape": word_shape(nw),
                "+1:is_punct": (not any(ch.isalnum() for ch in nw)),
            }
        )
    else:
        feats["EOS"] = True

    return feats


def seq_features(tokens: Sequence[Token]) -> List[Dict[str, object]]:
    return [token_features(tokens, i) for i in range(len(tokens))]


def labels_from_mentions(tokens: Sequence[Token], mentions: Sequence[Tuple[int, int]]) -> List[str]:
    labels = ["O"] * len(tokens)
    if not mentions or not tokens:
        return labels

    m_i = 0
    for t_i, (_w, ts, te) in enumerate(tokens):
        while m_i < len(mentions) and mentions[m_i][1] <= ts:
            m_i += 1
        if m_i >= len(mentions):
            break
        ms, me = mentions[m_i]
        if ts >= ms and te <= me:
            # Determine if this token is the first token fully inside this mention.
            # We label B if no previous token in the same mention was labeled.
            prev_in_same = False
            if t_i > 0:
                _pw, ps, pe = tokens[t_i - 1]
                if ps >= ms and pe <= me and labels[t_i - 1] != "O":
                    prev_in_same = True
            labels[t_i] = "I" if prev_in_same else "B"
    return labels


def token_spans_from_mentions(tokens: Sequence[Token], mentions: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Map character-level mention spans to (token_start, token_end_exclusive).

    Tokens are included if they are fully contained in the mention span.
    """
    out: List[Tuple[int, int]] = []
    if not mentions or not tokens:
        return out
    m_i = 0
    t_i = 0
    while m_i < len(mentions) and t_i < len(tokens):
        ms, me = mentions[m_i]
        # Advance to first token that could overlap.
        while t_i < len(tokens) and tokens[t_i][2] <= ms:
            t_i += 1
        if t_i >= len(tokens):
            break
        if tokens[t_i][1] < ms:
            # Token starts before mention; ignore it and move on.
            t_i += 1
            continue
        start_t = t_i
        end_t = start_t
        while end_t < len(tokens) and tokens[end_t][1] >= ms and tokens[end_t][2] <= me:
            end_t += 1
        if end_t > start_t:
            out.append((start_t, end_t))
        m_i += 1
    return out


def build_lexicon(
    examples: Sequence[SectionExample],
    annotations: DefaultDict[Tuple[str, str], List[Tuple[int, int]]],
    min_count: int = 3,
    max_len: int = 12,
) -> Dict[Tuple[str, ...], int]:
    """Build a token-sequence lexicon from gold mentions."""
    counts: DefaultDict[Tuple[str, ...], int] = defaultdict(int)
    for ex in examples:
        spans = token_spans_from_mentions(ex.tokens, annotations.get((ex.pmid, ex.section), []))
        for ts, te in spans:
            toks = tuple(t[0] for t in ex.tokens[ts:te])
            if not toks or len(toks) > max_len:
                continue
            mention = "".join(toks)
            if mention.isdigit():
                continue
            if len(mention) <= 2 and mention.isalpha():
                continue
            counts[toks] += 1

    return {k: v for k, v in counts.items() if v >= min_count}


def lexicon_match_spans(tokens: Sequence[Token], lexicon: Dict[Tuple[str, ...], int], max_len: int = 12) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    if not tokens or not lexicon:
        return spans
    n = len(tokens)
    # Brute-force n-gram matching with a tight max_len; fast enough here.
    for i in range(n):
        seq: List[str] = []
        for j in range(i, min(n, i + max_len)):
            seq.append(tokens[j][0])
            key = tuple(seq)
            if key in lexicon:
                spans.append((tokens[i][1], tokens[j][2]))
    return dedupe_spans(spans)


def spans_from_bio(tokens: Sequence[Token], labels: Sequence[str], text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    end: Optional[int] = None

    def flush() -> None:
        nonlocal start, end
        if start is None or end is None or end <= start:
            start = None
            end = None
            return
        s, e = trim_span(text, start, end)
        if e > s:
            spans.append((s, e))
        start = None
        end = None

    for (w, ts, te), lab in zip(tokens, labels):
        if lab == "O":
            flush()
            continue

        if lab == "B" or start is None:
            flush()
            start, end = ts, te
        else:
            # Continuation.
            end = te

        # Conservative sentence boundary enforcement: don't let a mention include
        # a strong sentence-final punctuation token.
        if w in (".", "?", "!"):
            flush()

    flush()
    return dedupe_spans(spans)


TRIM_LEFT = set(["(", "[", "{", ",", ";", ":"])
TRIM_RIGHT = set([").", ")", "]", "}", ",", ";", ":", "."])


def trim_span(text: str, s: int, e: int) -> Tuple[int, int]:
    # Strip whitespace (shouldn't exist from token boundaries, but safe).
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1

    # Strip common flanking punctuation that shouldn't be part of mentions.
    while s < e and text[s] in TRIM_LEFT:
        s += 1
    while e > s and text[e - 1] in TRIM_RIGHT:
        e -= 1
    # Trademark and possessive.
    if e > s and text[e - 1] in ("®", "™"):
        e -= 1
    if e > s + 2 and text[e - 2 : e] == "'s":
        e -= 2

    return s, e


def dedupe_spans(spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for s, e in spans:
        if (s, e) in seen:
            continue
        seen.add((s, e))
        out.append((s, e))
    out.sort()
    return out


BLACKLIST_SINGLE = {
    "water",
    "aqueous",
    "proton",
    "lead",
    "gold",
}


def filter_spans(text: str, spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for s, e in spans:
        ss, ee = trim_span(text, s, e)
        if ee <= ss:
            continue
        mention = text[ss:ee]
        if not any(ch.isalpha() or ch.isdigit() for ch in mention):
            continue
        if mention.lower() in BLACKLIST_SINGLE:
            continue
        if len(mention) <= 2 and mention.isalpha():
            # By guideline, 1-2 letter acronyms generally not tagged.
            continue
        if mention.isdigit():
            continue
        out.append((ss, ee))
    return dedupe_spans(out)


def build_examples(
    abstracts_path: str, annotations_path: Optional[str] = None
) -> List[SectionExample]:
    articles = read_abstracts(abstracts_path)
    ann = read_annotations(annotations_path) if annotations_path else defaultdict(list)

    examples: List[SectionExample] = []
    for pmid, (title, abstract) in articles.items():
        for section, text in (("T", title), ("A", abstract)):
            toks = tokenize(text)
            labels = None
            if annotations_path:
                labels = labels_from_mentions(toks, ann.get((pmid, section), []))
            examples.append(SectionExample(pmid=pmid, section=section, text=text, tokens=toks, labels=labels))
    return examples


def train_crf(examples: Sequence[SectionExample]) -> sklearn_crfsuite.CRF:
    X = [seq_features(ex.tokens) for ex in examples]
    y = [ex.labels for ex in examples]
    if any(lbls is None for lbls in y):
        raise ValueError("Training examples must have labels")
    y = [lbls for lbls in y if lbls is not None]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.08,
        c2=0.02,
        max_iterations=80,
        all_possible_transitions=True,
    )
    crf.fit(X, y)
    return crf


def predict_spans(
    crf: sklearn_crfsuite.CRF,
    examples: Sequence[SectionExample],
    lexicon: Optional[Dict[Tuple[str, ...], int]] = None,
) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    preds: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    X = [seq_features(ex.tokens) for ex in examples]
    y_pred = crf.predict(X)
    for ex, labs in zip(examples, y_pred):
        spans = spans_from_bio(ex.tokens, labs, ex.text)
        if lexicon:
            spans.extend(lexicon_match_spans(ex.tokens, lexicon))
        spans = filter_spans(ex.text, spans)
        preds[(ex.pmid, ex.section)] = spans
    return preds


def score(
    gold: DefaultDict[Tuple[str, str], List[Tuple[int, int]]],
    pred: Dict[Tuple[str, str], List[Tuple[int, int]]],
) -> Tuple[float, float, float]:
    gset = set()
    pset = set()
    for k, spans in gold.items():
        for s, e in spans:
            gset.add((k[0], k[1], s, e))
    for k, spans in pred.items():
        for s, e in spans:
            pset.add((k[0], k[1], s, e))
    tp = len(gset & pset)
    fp = len(pset - gset)
    fn = len(gset - pset)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def write_predictions(
    out_path: str, preds: Dict[Tuple[str, str], List[Tuple[int, int]]]
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for (pmid, section), spans in sorted(preds.items(), key=lambda x: (int(x[0][0]), x[0][1])):
            for s, e in spans:
                f.write(f"{pmid}\t{section}:{s}:{e}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-abstracts", default="data/training.abstracts.txt")
    p.add_argument("--train-annotations", default="data/training.annotations.txt")
    p.add_argument("--dev-abstracts", default="data/development.abstracts.txt")
    p.add_argument("--dev-annotations", default="data/development.annotations.txt")
    p.add_argument("--test-abstracts", default="data/test.abstracts.txt")
    p.add_argument("--out", default="output/predictions.tsv")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--use-lexicon", action="store_true")
    args = p.parse_args(argv)

    train_ex = build_examples(args.train_abstracts, args.train_annotations)
    dev_ex = build_examples(args.dev_abstracts, args.dev_annotations)

    full_train = list(train_ex) + list(dev_ex)
    if not args.no_eval:
        crf = train_crf(train_ex)
        gold_dev = read_annotations(args.dev_annotations)
        lex_train = None
        if args.use_lexicon:
            gold_train = read_annotations(args.train_annotations)
            lex_train = build_lexicon(train_ex, gold_train)
        pred_dev = predict_spans(crf, dev_ex, lexicon=lex_train)
        prec, rec, f1 = score(gold_dev, pred_dev)
        print(f"DEV precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")

    # Train on train+dev for final predictions.
    lex_full = None
    if args.use_lexicon:
        gold_full = read_annotations(args.train_annotations)
        gold_dev = read_annotations(args.dev_annotations)
        gold_full.update(gold_dev)
        lex_full = build_lexicon(full_train, gold_full)
    crf_full = train_crf(full_train)
    test_ex = build_examples(args.test_abstracts, None)
    pred_test = predict_spans(crf_full, test_ex, lexicon=lex_full)
    write_predictions(args.out, pred_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

