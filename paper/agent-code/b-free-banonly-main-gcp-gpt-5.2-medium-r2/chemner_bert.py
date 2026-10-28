import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)


LABEL_O = 0
LABEL_B = 1
LABEL_I = 2
ID2LABEL = {0: "O", 1: "B-CHEM", 2: "I-CHEM"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def read_abstracts(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            pmid, title, abstract = line.split("\t")
            out[pmid] = {"T": title, "A": abstract}
    return out


def read_annotations(path: str) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    out: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            pmid, section, start, end, _text, _klass = line.split("\t")
            key = (pmid, section)
            out.setdefault(key, []).append((int(start), int(end)))
    for key in list(out.keys()):
        out[key] = sorted(out[key])
    return out


def _spans_for(ann: Dict[Tuple[str, str], List[Tuple[int, int]]], pmid: str, section: str) -> List[Tuple[int, int]]:
    return ann.get((pmid, section), [])


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _dedupe_and_resolve_overlaps(spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    uniq = sorted(set(spans), key=lambda x: (x[0], -(x[1] - x[0]), x[1]))
    out: List[Tuple[int, int]] = []
    for s in uniq:
        if not out:
            out.append(s)
            continue
        if _overlaps(out[-1], s):
            continue
        out.append(s)
    return out


@dataclass
class EncodedWindow:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    pmid: str
    section: str
    offset_mapping: List[Tuple[int, int]]


def encode_with_overflow(
    tokenizer,
    pmid: str,
    section: str,
    text: str,
    spans: Sequence[Tuple[int, int]],
    *,
    max_length: int,
    stride: int,
) -> List[EncodedWindow]:
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
    )

    windows: List[EncodedWindow] = []
    for i in range(len(enc["input_ids"])):
        offsets: List[Tuple[int, int]] = [tuple(x) for x in enc["offset_mapping"][i]]
        labels: List[int] = []
        for (s, e), tok_id in zip(offsets, enc["input_ids"][i]):
            if s == 0 and e == 0:
                labels.append(-100)
                continue
            lbl = LABEL_O
            for ms, me in spans:
                if s >= me or e <= ms:
                    continue
                if s == ms:
                    lbl = LABEL_B
                else:
                    lbl = LABEL_I
                break
            labels.append(lbl)

        windows.append(
            EncodedWindow(
                input_ids=enc["input_ids"][i],
                attention_mask=enc["attention_mask"][i],
                labels=labels,
                pmid=pmid,
                section=section,
                offset_mapping=offsets,
            )
        )
    return windows


def encode_many_with_overflow(
    tokenizer,
    pmids: Sequence[str],
    sections: Sequence[str],
    texts: Sequence[str],
    spans_list: Sequence[Sequence[Tuple[int, int]]],
    *,
    max_length: int,
    stride: int,
) -> List[EncodedWindow]:
    enc = tokenizer(
        list(texts),
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
    )
    mapping = enc["overflow_to_sample_mapping"]

    windows: List[EncodedWindow] = []
    for i in range(len(enc["input_ids"])):
        sample_idx = mapping[i]
        pmid = pmids[sample_idx]
        section = sections[sample_idx]
        spans = spans_list[sample_idx]
        offsets: List[Tuple[int, int]] = [tuple(x) for x in enc["offset_mapping"][i]]
        labels: List[int] = []
        for (s, e) in offsets:
            if s == 0 and e == 0:
                labels.append(-100)
                continue
            lbl = LABEL_O
            for ms, me in spans:
                if s >= me or e <= ms:
                    continue
                lbl = LABEL_B if s == ms else LABEL_I
                break
            labels.append(lbl)

        windows.append(
            EncodedWindow(
                input_ids=enc["input_ids"][i],
                attention_mask=enc["attention_mask"][i],
                labels=labels,
                pmid=pmid,
                section=section,
                offset_mapping=offsets,
            )
        )
    return windows


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, windows: Sequence[EncodedWindow]):
        self.windows = list(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        return {
            "input_ids": w.input_ids,
            "attention_mask": w.attention_mask,
            "labels": w.labels,
        }


@dataclass
class PredWindow:
    input_ids: List[int]
    attention_mask: List[int]
    pmid: str
    section: str
    offset_mapping: List[Tuple[int, int]]
    text_len: int


class PredWindowDataset(torch.utils.data.Dataset):
    def __init__(self, windows: Sequence[PredWindow]):
        self.windows = list(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        return {
            "input_ids": w.input_ids,
            "attention_mask": w.attention_mask,
            "_idx": idx,
        }


def train_model(
    model,
    tokenizer,
    train_windows: Sequence[EncodedWindow],
    *,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    collator = DataCollatorForTokenClassification(tokenizer, padding=True)
    dataset = WindowDataset(train_windows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=max(1, total_steps // 20), num_training_steps=total_steps
    )

    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)


@torch.no_grad()
def predict_token_labels(model, tokenizer, text: str, *, max_length: int, stride: int) -> List[Tuple[int, int]]:
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
    )

    spans: List[Tuple[int, int]] = []
    for i in range(len(enc["input_ids"])):
        input_ids = torch.tensor([enc["input_ids"][i]], device=model.device)
        attention_mask = torch.tensor([enc["attention_mask"][i]], device=model.device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
        pred = logits.argmax(dim=-1).tolist()
        offsets = [tuple(x) for x in enc["offset_mapping"][i]]

        cur_start = None
        cur_end = None
        for (s, e), lab in zip(offsets, pred):
            if s == 0 and e == 0:
                continue
            if lab == LABEL_O:
                if cur_start is not None:
                    spans.append((cur_start, cur_end))
                    cur_start, cur_end = None, None
                continue
            if lab == LABEL_B or cur_start is None:
                if cur_start is not None:
                    spans.append((cur_start, cur_end))
                cur_start, cur_end = s, e
            else:
                cur_end = max(cur_end, e)
        if cur_start is not None:
            spans.append((cur_start, cur_end))

    spans = [(s, e) for s, e in spans if 0 <= s < e <= len(text)]
    return _dedupe_and_resolve_overlaps(spans)


def _decode_window(offsets: Sequence[Tuple[int, int]], labels: Sequence[int], text_len: int) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cur_start = None
    cur_end = None
    for (s, e), lab in zip(offsets, labels):
        if s == 0 and e == 0:
            continue
        if lab == LABEL_O:
            if cur_start is not None:
                spans.append((cur_start, cur_end))
                cur_start, cur_end = None, None
            continue
        if lab == LABEL_B or cur_start is None:
            if cur_start is not None:
                spans.append((cur_start, cur_end))
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    if cur_start is not None:
        spans.append((cur_start, cur_end))
    return [(s, e) for s, e in spans if 0 <= s < e <= text_len]


@torch.no_grad()
def predict_many(
    model,
    tokenizer,
    pmids: Sequence[str],
    sections: Sequence[str],
    texts: Sequence[str],
    *,
    max_length: int,
    stride: int,
    batch_size: int,
) -> Dict[Tuple[str, str], List[Tuple[int, int]]]:
    out: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    model.eval()
    collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    text_chunk_size = 256
    for start in range(0, len(texts), text_chunk_size):
        sub_pmids = pmids[start : start + text_chunk_size]
        sub_secs = sections[start : start + text_chunk_size]
        sub_texts = texts[start : start + text_chunk_size]

        enc = tokenizer(
            list(sub_texts),
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=max_length,
            stride=stride,
        )
        mapping = enc["overflow_to_sample_mapping"]

        windows: List[PredWindow] = []
        for i in range(len(enc["input_ids"])):
            sample_idx = mapping[i]
            windows.append(
                PredWindow(
                    input_ids=enc["input_ids"][i],
                    attention_mask=enc["attention_mask"][i],
                    pmid=sub_pmids[sample_idx],
                    section=sub_secs[sample_idx],
                    offset_mapping=[tuple(x) for x in enc["offset_mapping"][i]],
                    text_len=len(sub_texts[sample_idx]),
                )
            )

        dataset = PredWindowDataset(windows)

        def pred_collate(features):
            idxs = torch.tensor([f.pop("_idx") for f in features], dtype=torch.long)
            batch = collator(features)
            batch["_idx"] = idxs
            return batch

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pred_collate)

        for batch in loader:
            idxs = batch.pop("_idx")
            batch = {k: v.to(model.device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            for row, win_idx in enumerate(idxs.tolist()):
                win = windows[win_idx]
                labels = preds[row][: len(win.offset_mapping)]
                spans = _decode_window(win.offset_mapping, labels, win.text_len)
                out.setdefault((win.pmid, win.section), []).extend(spans)

    return {k: _dedupe_and_resolve_overlaps(v) for k, v in out.items()}


def score_f1(gold: Dict[Tuple[str, str], List[Tuple[int, int]]], pred: Dict[Tuple[str, str], List[Tuple[int, int]]]):
    gold_set = set((pmid, sec, s, e) for (pmid, sec), spans in gold.items() for s, e in spans)
    pred_set = set((pmid, sec, s, e) for (pmid, sec), spans in pred.items() for s, e in spans)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-cased")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--do-eval", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument(
        "--output",
        default="/workspace/output/predictions.tsv",
        help="where to write predictions",
    )
    args = ap.parse_args()

    train_abs = read_abstracts("/workspace/data/training.abstracts.txt")
    dev_abs = read_abstracts("/workspace/data/development.abstracts.txt")
    train_ann = read_annotations("/workspace/data/training.annotations.txt")
    dev_ann = read_annotations("/workspace/data/development.annotations.txt")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )
    model.to(torch.device("cpu"))

    # Speed/regularization on CPU: train classifier + last transformer block.
    for name, p in model.named_parameters():
        if name.startswith("classifier") or ".transformer.layer.5." in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    def make_windows(abs_map, ann_map) -> List[EncodedWindow]:
        pmids: List[str] = []
        secs: List[str] = []
        texts: List[str] = []
        spans_list: List[List[Tuple[int, int]]] = []
        for pmid, sections in abs_map.items():
            for sec in ("T", "A"):
                pmids.append(pmid)
                secs.append(sec)
                texts.append(sections[sec])
                spans_list.append(_spans_for(ann_map, pmid, sec))

        return encode_many_with_overflow(
            tokenizer,
            pmids,
            secs,
            texts,
            spans_list,
            max_length=args.max_length,
            stride=args.stride,
        )

    train_windows = make_windows(train_abs, train_ann)
    if args.do_eval:
        dev_windows = make_windows(dev_abs, dev_ann)
        _ = dev_windows  # silence linter

    if not args.do_eval:
        train_windows = train_windows + make_windows(dev_abs, dev_ann)

    train_model(
        model,
        tokenizer,
        train_windows,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    if args.do_eval:
        dev_pmids: List[str] = []
        dev_secs: List[str] = []
        dev_texts: List[str] = []
        for pmid, sections in dev_abs.items():
            for sec in ("T", "A"):
                dev_pmids.append(pmid)
                dev_secs.append(sec)
                dev_texts.append(sections[sec])
        pred_dev = predict_many(
            model,
            tokenizer,
            dev_pmids,
            dev_secs,
            dev_texts,
            max_length=args.max_length,
            stride=args.stride,
            batch_size=args.batch_size,
        )
        print(json.dumps(score_f1(dev_ann, pred_dev), indent=2, sort_keys=True))

    if args.do_eval and args.skip_test:
        return 0

    test_abs = read_abstracts("/workspace/data/test.abstracts.txt")

    test_pmids: List[str] = []
    test_secs: List[str] = []
    test_texts: List[str] = []
    for pmid, sections in test_abs.items():
        for sec in ("T", "A"):
            test_pmids.append(pmid)
            test_secs.append(sec)
            test_texts.append(sections[sec])

    pred_test = predict_many(
        model,
        tokenizer,
        test_pmids,
        test_secs,
        test_texts,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size,
    )

    lines: List[str] = []
    for (pmid, sec), spans in pred_test.items():
        for s, e in spans:
            lines.append(f"{pmid}\t{sec}:{s}:{e}\n")

    def sort_key(line: str):
        pmid, rest = line.split("\t")
        sec, s, e = rest.split(":")
        return (int(pmid), sec, int(s), int(e))

    lines.sort(key=sort_key)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

