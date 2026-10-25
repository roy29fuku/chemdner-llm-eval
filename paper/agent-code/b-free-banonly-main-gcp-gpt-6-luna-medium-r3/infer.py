import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
ROOT = "/workspace/data"
OUT = "/workspace/output/predictions.tsv"
MAX_LEN = 384
WINDOW = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)


class Sections(Dataset):
    def __init__(self, sections):
        self.rows = []
        self.sections = sections
        for key, text in sections.items():
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ids, offsets = enc["input_ids"], enc["offset_mapping"]
            if not ids:
                self.rows.append((key, [], [(0, 0), (0, 0)], []))
                continue
            for st in range(0, len(ids), WINDOW):
                en = min(st + MAX_LEN - 2, len(ids))
                if en == len(ids) and st > 0 and len(ids) - st < 80:
                    continue
                window_ids = ids[st:en]
                self.rows.append((key, [tokenizer.cls_token_id] + window_ids + [tokenizer.sep_token_id],
                                  [(0, 0)] + offsets[st:en] + [(0, 0)], list(range(st, en))))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch):
    n = max(len(x[1]) for x in batch)
    ids = torch.full((len(batch), n), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, row in enumerate(batch):
        ids[i, :len(row[1])] = torch.tensor(row[1])
        mask[i, :len(row[1])] = 1
    return ids, mask, batch


def main():
    sections = {}
    order = []
    with open(f"{ROOT}/test.abstracts.txt", encoding="utf-8") as f:
        for line in f:
            pmid, title, abstract = line.rstrip("\n").split("\t")
            order.append(pmid)
            sections[(pmid, "T")] = title
            sections[(pmid, "A")] = abstract

    meta = json.load(open("/workspace/output/model_meta.json", encoding="utf-8"))
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=3, id2label={0: "O", 1: "B-CHEM", 2: "I-CHEM"},
        label2id={"O": 0, "B-CHEM": 1, "I-CHEM": 2})
    model.load_state_dict(torch.load("/workspace/output/chemical_ner.pt", map_location="cpu", weights_only=True))
    model.to(DEVICE).eval()
    ds = Sections(sections)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
    sums = defaultdict(lambda: torch.zeros(3))
    counts = defaultdict(int)
    with torch.no_grad():
        for ids, mask, batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                logits = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE)).logits
            logits = logits.float().cpu()
            for i, row in enumerate(batch):
                key, token_ids, offsets, global_indices = row
                for j, gi in enumerate(global_indices):
                    sums[(key, gi)] += logits[i, j + 1]
                    counts[(key, gi)] += 1

    predicted = defaultdict(set)
    bias = float(meta["bias"])
    for key, text in sections.items():
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        tokens = []
        for gi, (start, end) in enumerate(enc["offset_mapping"]):
            scores = sums[(key, gi)] / max(counts[(key, gi)], 1)
            scores[1:] += bias
            tokens.append((int(scores.argmax()), start, end))
        current = None
        for label, start, end in tokens:
            if label == 1:
                if current:
                    predicted[key].add(current)
                current = (start, end)
            elif label == 2:
                if current:
                    current = (current[0], end)
                else:
                    current = (start, end)
            else:
                if current:
                    predicted[key].add(current)
                    current = None
        if current:
            predicted[key].add(current)

    with open(OUT, "w", encoding="utf-8") as f:
        for pmid in order:
            for sec in ("T", "A"):
                for start, end in sorted(predicted[(pmid, sec)]):
                    f.write(f"{pmid}\t{sec}:{start}:{end}\n")
    print("wrote", sum(map(len, predicted.values())), "mentions; validation bias", bias)


tokenizer = AutoTokenizer.from_pretrained("/workspace/output/tokenizer", use_fast=True)
if __name__ == "__main__":
    main()
