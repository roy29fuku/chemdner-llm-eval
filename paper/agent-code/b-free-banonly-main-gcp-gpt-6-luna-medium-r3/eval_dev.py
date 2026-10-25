import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
ROOT = "/workspace/data"
MAX_LEN, WINDOW = 384, 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)


def main():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/workspace/output/tokenizer", use_fast=True)
    sections, gold = {}, defaultdict(set)
    for line in open(f"{ROOT}/development.abstracts.txt", encoding="utf8"):
        pmid, title, abstract = line.rstrip("\n").split("\t")
        sections[(pmid, "T")] = title
        sections[(pmid, "A")] = abstract
    for line in open(f"{ROOT}/development.annotations.txt", encoding="utf8"):
        pmid, sec, start, end, text, cl = line.rstrip("\n").split("\t")
        gold[(pmid, sec)].add((int(start), int(end)))

    encodings, logits, counts, rows = {}, {}, {}, []
    for key, text in sections.items():
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        encodings[key] = offsets
        logits[key] = np.zeros((len(ids), 3), dtype=np.float32)
        counts[key] = np.zeros(len(ids), dtype=np.float32)
        for st in range(0, len(ids), WINDOW):
            en = min(st + MAX_LEN - 2, len(ids))
            if en == len(ids) and st > 0 and len(ids) - st < 80:
                continue
            rows.append((key, [tokenizer.cls_token_id] + ids[st:en] + [tokenizer.sep_token_id], list(range(st, en))))

    def collate(batch):
        size = max(len(row[1]) for row in batch)
        input_ids = torch.full((len(batch), size), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), size), dtype=torch.long)
        for i, row in enumerate(batch):
            input_ids[i, :len(row[1])] = torch.tensor(row[1])
            mask[i, :len(row[1])] = 1
        return input_ids, mask, batch

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=3, id2label={0: "O", 1: "B-CHEM", 2: "I-CHEM"},
        label2id={"O": 0, "B-CHEM": 1, "I-CHEM": 2})
    model.load_state_dict(torch.load("/workspace/output/chemical_ner.pt", map_location="cpu", weights_only=True))
    model.to(DEVICE).eval()
    loader = DataLoader(rows, batch_size=16, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for input_ids, mask, batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                out = model(input_ids=input_ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits
            out = out.float().cpu().numpy()
            for i, (key, ids, indices) in enumerate(batch):
                logits[key][indices] += out[i, 1:len(ids)-1]
                counts[key][indices] += 1

    gold_set = {(key, a, b) for key, spans in gold.items() for a, b in spans}
    results = []
    for bias in [-5, -4, -3, -2.5, -2, -1.75, -1.5, -1.25, -1, -.5, 0]:
        pred_set = set()
        for key, offsets in encodings.items():
            scores = logits[key] / np.maximum(counts[key][:, None], 1)
            labels = np.argmax(scores + np.array([0, bias, bias], dtype=np.float32), axis=1)
            current = None
            for label, (start, end) in zip(labels, offsets):
                if label == 1:
                    if current:
                        pred_set.add((key, *current))
                    current = (start, end)
                elif label == 2:
                    current = (current[0], end) if current else (start, end)
                elif current:
                    pred_set.add((key, *current))
                    current = None
            if current:
                pred_set.add((key, *current))
        tp = len(pred_set & gold_set)
        precision = tp / max(1, len(pred_set))
        recall = tp / len(gold_set)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        results.append((f1, bias, precision, recall, len(pred_set)))
    print("best thresholds:", sorted(results, reverse=True))


if __name__ == "__main__":
    main()
