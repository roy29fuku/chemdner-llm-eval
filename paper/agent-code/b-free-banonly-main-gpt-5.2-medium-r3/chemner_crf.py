#!/usr/bin/env python3

import argparse
import dataclasses
import os
import re
import sys
from collections import defaultdict

import pycrfsuite
from tqdm import tqdm


@dataclasses.dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> list[Token]:
    return [Token(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def load_abstracts(path: str) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no}: expected 3 TSV fields, got {len(parts)}")
            pmid, title, abstract = parts
            out[pmid] = (title, abstract)
    return out


def load_annotations(path: str) -> dict[tuple[str, str], list[tuple[int, int]]]:
    out: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            parts = line.split("\t")
            if len(parts) != 6:
                raise ValueError(f"{path}:{line_no}: expected 6 TSV fields, got {len(parts)}")
            pmid, section, start, end, _text, _cls = parts
            if section not in {"T", "A"}:
                raise ValueError(f"{path}:{line_no}: bad section {section!r}")
            out[(pmid, section)].append((int(start), int(end)))

    # Keep deterministic order.
    for k, spans in out.items():
        out[k] = sorted(spans)
    return dict(out)


def spans_to_bio(tokens: list[Token], spans: list[tuple[int, int]]) -> list[str]:
    labels = ["O"] * len(tokens)
    for start, end in spans:
        inside = [
            i
            for i, t in enumerate(tokens)
            if not (t.end <= start or t.start >= end)  # any overlap
        ]
        if not inside:
            continue
        for j, idx in enumerate(inside):
            labels[idx] = "B" if j == 0 else "I"
    return labels


def bio_to_spans(tokens: list[Token], labels: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end: int | None = None

    for t, lab in zip(tokens, labels):
        if lab == "O":
            if cur_start is not None:
                spans.append((cur_start, cur_end if cur_end is not None else cur_start))
                cur_start, cur_end = None, None
            continue

        if lab == "B" or cur_start is None:
            if cur_start is not None:
                spans.append((cur_start, cur_end if cur_end is not None else cur_start))
            cur_start = t.start
            cur_end = t.end
        else:  # I
            # Expand the current span to include any whitespace between tokens.
            cur_end = t.end

    if cur_start is not None:
        spans.append((cur_start, cur_end if cur_end is not None else cur_start))
    return spans


def word_shape(token: str) -> str:
    out = []
    for ch in token:
        if ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        elif ch.isdigit():
            out.append("0")
        else:
            out.append(ch)
    # Compress runs.
    compressed = []
    for ch in out:
        if compressed and compressed[-1] == ch:
            continue
        compressed.append(ch)
    return "".join(compressed)


def token_features(
    tokens: list[Token],
    i: int,
    section: str,
    chem_token_vocab: set[str],
) -> dict[str, object]:
    tok = tokens[i].text
    low = tok.lower()

    feats: dict[str, object] = {
        "bias": 1.0,
        f"section={section}": True,
        f"low={low}": True,
        f"shape={word_shape(tok)}": True,
        f"len={len(tok)}": True,
        "isupper": tok.isupper(),
        "islower": tok.islower(),
        "istitle": tok.istitle(),
        "isdigit": tok.isdigit(),
        "hasdigit": any(ch.isdigit() for ch in tok),
        "hasalpha": any(ch.isalpha() for ch in tok),
        "has-hyphen": "-" in tok,
        "has-plus": "+" in tok,
        "has-slash": "/" in tok,
        "has-comma": "," in tok,
        "has-dot": "." in tok,
        "has-paren": ("(" in tok) or (")" in tok),
        "has-bracket": ("[" in tok) or ("]" in tok),
        "has-eq": "=" in tok,
        "chem_vocab": low in chem_token_vocab,
    }

    for k in (2, 3, 4):
        if len(tok) >= k:
            feats[f"pref{k}={low[:k]}"] = True
            feats[f"suf{k}={low[-k:]}"] = True

    if i > 0:
        prev = tokens[i - 1].text
        prev_low = prev.lower()
        feats.update(
            {
                f"-1:low={prev_low}": True,
                f"-1:shape={word_shape(prev)}": True,
                "-1:isupper": prev.isupper(),
                "-1:istitle": prev.istitle(),
            }
        )
    else:
        feats["BOS"] = True

    if i + 1 < len(tokens):
        nxt = tokens[i + 1].text
        nxt_low = nxt.lower()
        feats.update(
            {
                f"+1:low={nxt_low}": True,
                f"+1:shape={word_shape(nxt)}": True,
                "+1:isupper": nxt.isupper(),
                "+1:istitle": nxt.istitle(),
            }
        )
    else:
        feats["EOS"] = True

    return feats


def featurize(
    text: str,
    section: str,
    spans: list[tuple[int, int]] | None,
    chem_token_vocab: set[str],
):
    tokens = tokenize(text)
    xseq = [token_features(tokens, i, section, chem_token_vocab) for i in range(len(tokens))]
    if spans is None:
        return tokens, xseq, None
    yseq = spans_to_bio(tokens, spans)
    return tokens, xseq, yseq


def build_chem_token_vocab(
    abstracts: dict[str, tuple[str, str]],
    annotations: dict[tuple[str, str], list[tuple[int, int]]],
) -> set[str]:
    vocab: set[str] = set()
    for pmid, (title, abstract) in abstracts.items():
        for section, text in ("T", title), ("A", abstract):
            spans = annotations.get((pmid, section), [])
            if not spans:
                continue
            toks = tokenize(text)
            for t in toks:
                for s, e in spans:
                    if not (t.end <= s or t.start >= e):
                        vocab.add(t.text.lower())
                        break
    return vocab


def score_f1(
    gold: dict[tuple[str, str], list[tuple[int, int]]],
    pred: dict[tuple[str, str], list[tuple[int, int]]],
) -> tuple[float, float, float]:
    tp = fp = fn = 0
    keys = set(gold) | set(pred)
    for k in keys:
        g = set(gold.get(k, []))
        p = set(pred.get(k, []))
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def train_crf(
    x_train: list[list[dict[str, object]]],
    y_train: list[list[str]],
    c1: float,
    c2: float,
    max_iterations: int,
) -> pycrfsuite.Tagger:
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(x_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params(
        {
            "c1": c1,
            "c2": c2,
            "max_iterations": max_iterations,
            "feature.possible_transitions": True,
        }
    )

    model_path = "_tmp_crf_model.bin"
    trainer.train(model_path)
    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)
    os.remove(model_path)
    return tagger


def predict_sections(
    tagger: pycrfsuite.Tagger,
    abstracts: dict[str, tuple[str, str]],
    chem_token_vocab: set[str],
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    pred: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for pmid, (title, abstract) in tqdm(abstracts.items(), desc="predict", unit="doc"):
        for section, text in ("T", title), ("A", abstract):
            tokens, xseq, _ = featurize(text, section, None, chem_token_vocab)
            tagger.set(xseq)
            yhat = tagger.tag()
            spans = bio_to_spans(tokens, yhat)
            if spans:
                pred[(pmid, section)] = spans
    return pred


def build_train_matrices(
    abstracts: dict[str, tuple[str, str]],
    annotations: dict[tuple[str, str], list[tuple[int, int]]],
    chem_token_vocab: set[str],
):
    x: list[list[dict[str, object]]] = []
    y: list[list[str]] = []
    for pmid, (title, abstract) in abstracts.items():
        for section, text in ("T", title), ("A", abstract):
            spans = annotations.get((pmid, section), [])
            tokens, xseq, yseq = featurize(text, section, spans, chem_token_vocab)
            if not tokens:
                continue
            x.append(xseq)
            y.append(yseq or [])
    return x, y


def write_predictions(
    out_path: str,
    preds: dict[tuple[str, str], list[tuple[int, int]]],
    test_abstracts: dict[str, tuple[str, str]],
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pmid in test_abstracts.keys():
            for section in ("T", "A"):
                spans = sorted(preds.get((pmid, section), []))
                for s, e in spans:
                    f.write(f"{pmid}\t{section}:{s}:{e}\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-abstracts", default="data/training.abstracts.txt")
    ap.add_argument("--train-annotations", default="data/training.annotations.txt")
    ap.add_argument("--dev-abstracts", default="data/development.abstracts.txt")
    ap.add_argument("--dev-annotations", default="data/development.annotations.txt")
    ap.add_argument("--test-abstracts", default="data/test.abstracts.txt")
    ap.add_argument("--out", default="output/predictions.tsv")
    ap.add_argument("--max-iterations", type=int, default=80)
    args = ap.parse_args(argv)

    train_abs = load_abstracts(args.train_abstracts)
    train_ann = load_annotations(args.train_annotations)
    dev_abs = load_abstracts(args.dev_abstracts)
    dev_ann = load_annotations(args.dev_annotations)
    test_abs = load_abstracts(args.test_abstracts)

    # Token-level vocabulary from labeled mentions (cheap gazetteer feature).
    chem_vocab = build_chem_token_vocab(train_abs, train_ann)
    chem_vocab |= build_chem_token_vocab(dev_abs, dev_ann)

    x_train, y_train = build_train_matrices(train_abs, train_ann, chem_vocab)
    x_dev, y_dev = build_train_matrices(dev_abs, dev_ann, chem_vocab)
    _ = y_dev  # unused, but keeps the dev featurization consistent.

    # Small grid search on dev.
    candidates = [(0.1, 0.01), (0.2, 0.02)]
    best = None
    best_tagger = None
    for c1, c2 in candidates:
        tagger = train_crf(x_train, y_train, c1=c1, c2=c2, max_iterations=args.max_iterations)
        dev_pred = {}
        for pmid, (title, abstract) in dev_abs.items():
            for section, text in ("T", title), ("A", abstract):
                tokens, xseq, _ = featurize(text, section, None, chem_vocab)
                tagger.set(xseq)
                yhat = tagger.tag()
                dev_pred[(pmid, section)] = bio_to_spans(tokens, yhat)
        prec, rec, f1 = score_f1(dev_ann, dev_pred)
        print(
            f"dev c1={c1} c2={c2} -> P={prec:.4f} R={rec:.4f} F1={f1:.4f}",
            flush=True,
        )
        if best is None or f1 > best[2]:
            best = (c1, c2, f1)
            best_tagger = tagger

    assert best is not None and best_tagger is not None
    best_c1, best_c2, _best_f1 = best
    print(f"best params: c1={best_c1} c2={best_c2}", flush=True)

    # Train final model on train+dev.
    merged_abs = {**train_abs, **dev_abs}
    merged_ann = dict(train_ann)
    for k, v in dev_ann.items():
        merged_ann[k] = v
    x_all, y_all = build_train_matrices(merged_abs, merged_ann, chem_vocab)
    final_tagger = train_crf(
        x_all, y_all, c1=best_c1, c2=best_c2, max_iterations=args.max_iterations
    )
    test_pred = predict_sections(final_tagger, test_abs, chem_vocab)
    write_predictions(args.out, test_pred, test_abs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

