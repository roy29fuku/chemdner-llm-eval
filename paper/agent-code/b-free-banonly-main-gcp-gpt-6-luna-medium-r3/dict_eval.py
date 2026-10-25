from collections import Counter, defaultdict
import re

ROOT = "/workspace/data"


def read(split):
    texts = {}
    for line in open(f"{ROOT}/{split}.abstracts.txt", encoding="utf8"):
        pmid, title, abstract = line.rstrip("\n").split("\t")
        texts[(pmid, "T")] = title
        texts[(pmid, "A")] = abstract
    gold = defaultdict(set)
    if split != "test":
        for line in open(f"{ROOT}/{split}.annotations.txt", encoding="utf8"):
            pmid, sec, start, end, text, cl = line.rstrip("\n").split("\t")
            gold[(pmid, sec)].add((int(start), int(end)))
    return texts, gold


train_texts, _ = read("training")
dev_texts, dev_gold = read("development")
counts = Counter()
for line in open(f"{ROOT}/training.annotations.txt", encoding="utf8"):
    pmid, sec, start, end, text, cl = line.rstrip("\n").split("\t")
    counts[text.lower()] += 1
terms = sorted(counts, key=len, reverse=True)
pat = re.compile(r"(?=(?<!\w)(" + "|".join(re.escape(t) for t in terms) + r")(?!\w))")
occurrences = defaultdict(list)
for key, text in dev_texts.items():
    low = text.lower()
    occurrences[key] = [(m.start(1), m.end(1), counts[m.group(1)]) for m in pat.finditer(low)]

for minimum in [1, 2, 3, 5, 10]:
    pred = defaultdict(set)
    for key, hits in occurrences.items():
        candidates = [(a, b) for a, b, freq in hits if freq >= minimum]
        candidates.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
        chosen = []
        for a, b in candidates:
            if not any(a < x[1] and b > x[0] for x in chosen):
                chosen.append((a, b))
        pred[key].update(chosen)
    ps = {(key, a, b) for key, vals in pred.items() for a, b in vals}
    gs = {(key, a, b) for key, vals in dev_gold.items() for a, b in vals}
    tp = len(ps & gs)
    precision = tp / max(1, len(ps))
    recall = tp / len(gs)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(minimum, "terms", sum(n >= minimum for n in counts.values()), "P/R/F1",
          *(round(x, 4) for x in (precision, recall, f1)), "mentions", len(ps))
