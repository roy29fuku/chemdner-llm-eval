"""採点器（cem.py）を、BioCreative VII NLM-Chem トラックの公式評価スクリプトと突き合わせる。

公式の strict の span 評価は、CEM と同じく位置の完全一致で採点する。evaluation 分割の正解に、決まった乱数で
誤り（欠落・境界のずれ・余分）を混ぜた予測を作り、両方の採点器で TP / FP / FN が一致するかを確かめる。

公式スクリプトは NCBI の配布物（BC7T2-evaluation_v3.zip）を data/ に取得して使う（repo には置かない）。

    uv run python scripts/crosscheck_bc7_official.py

公式スクリプトの照合のキーは (文書 ID, 種類, 位置, 文字列) で、節（title / abstract）を含まない。CHEMDNER は節ごとに
位置を 0 から数えるので、title と abstract の同じ位置に同じ文字列の言及があると、公式スクリプトでは 1 件に重なる。
その件数も数えて表示する。
"""

import io
import random
import re
import subprocess
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from chemdner_bench.cem import score
from chemdner_bench.corpus import SECTIONS, corpus_path, data_dir, gold_spans, load

EVAL_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-evaluation_v3.zip"
SPLIT = "evaluation"


def official_script() -> Path:
    dest = data_dir() / "bc7t2-evaluation-v3"
    if not (dest / "evaluate.py").exists():
        with urllib.request.urlopen(EVAL_URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
    return dest / "evaluate.py"


def perturb(gold, docs, seed=0):
    """正解から決まった乱数で予測を作る: 10% を落とし、5% の終了位置を 1 文字ずらし、文書の 5% に余分な言及を足す。"""
    rng = random.Random(seed)
    pred = {}
    for d in docs:
        spans = set()
        for s, b, e in sorted(gold[d.pmid]):
            u = rng.random()
            if u < 0.10:
                continue
            if u < 0.15 and e < len(d.text(s)):
                e += 1
            spans.add((s, b, e))
        if rng.random() < 0.05 and len(d.abstract) > 10:
            b = rng.randrange(0, len(d.abstract) - 5)
            spans.add(("A", b, b + 5))
        pred[d.pmid] = spans
    return pred


def to_bioc(reference: ET.Element, spans, docs) -> ET.Element:
    """正解の BioC から Chemical の注釈を外し、予測の言及を入れる（本文と他の注釈はそのまま）。"""
    by_pmid = {d.pmid: d for d in docs}
    n = 0
    for doc in reference.findall("document"):
        pmid = doc.findtext("id")
        for p in doc.findall("passage"):
            section = SECTIONS[{i.get("key"): i.text for i in p.findall("infon")}.get("type")]
            for a in p.findall("annotation"):
                if {i.get("key"): i.text for i in a.findall("infon")}.get("type") == "Chemical":
                    p.remove(a)
            for s, b, e in sorted(spans.get(pmid, ())):
                if s != section:
                    continue
                n += 1
                a = ET.SubElement(p, "annotation", id=f"P{n}")
                ET.SubElement(a, "infon", key="type").text = "Chemical"
                ET.SubElement(a, "location", offset=str(b), length=str(e - b))
                ET.SubElement(a, "text").text = by_pmid[pmid].text(s)[b:e]
    return reference


def score_without_section(gold, pred, docs):
    """公式スクリプトと同じキー（位置と文字列。節を含まない）で数え直す。公式と厳密に一致するはずの値。"""
    by_pmid = {d.pmid: d for d in docs}

    def keyed(spans):
        return {pmid: {(b, e, by_pmid[pmid].text(s)[b:e]) for s, b, e in ss} for pmid, ss in spans.items()}

    g, p = keyed(gold), keyed(pred)
    tp = sum(len(g[k] & p.get(k, set())) for k in g)
    fp = sum(len(p[k] - g.get(k, set())) for k in p)
    fn = sum(len(g[k] - p.get(k, set())) for k in g)
    return tp, fp, fn


def collisions(spans, docs) -> int:
    """title と abstract の同じ位置に同じ文字列がある言及の数（公式スクリプトでは 1 件に重なる）。"""
    by_pmid = {d.pmid: d for d in docs}
    n = 0
    for pmid, ss in spans.items():
        seen = set()
        for s, b, e in ss:
            key = (b, e, by_pmid[pmid].text(s)[b:e])
            n += key in seen
            seen.add(key)
    return n


def main() -> int:
    docs = load(SPLIT)
    gold = gold_spans(docs)
    pred = perturb(gold, docs)
    ours = score(gold, pred, [d.pmid for d in docs])

    with tarfile.open(corpus_path()) as tar:
        raw = tar.extractfile(f"BC7T2-CHEMDNER-corpus-{SPLIT}.BioC.xml").read()
    raw = raw.replace(b'encoding="utf8"', b'encoding="utf-8"', 1)
    work = data_dir() / "crosscheck"
    work.mkdir(parents=True, exist_ok=True)
    (work / "reference.xml").write_bytes(raw)
    ET.ElementTree(to_bioc(ET.fromstring(raw), pred, docs)).write(work / "prediction.xml", encoding="utf-8", xml_declaration=True)

    script = official_script()
    proc = subprocess.run(
        [sys.executable, str(script), "-r", str(work / "reference.xml"), "-p", str(work / "prediction.xml"),
         "-t", "span", "-m", "strict", "-a", "Chemical", "-l", "info"],
        capture_output=True, text=True, cwd=script.parent,
    )
    log = proc.stdout + proc.stderr
    m = re.search(r"TP = (\d+), FP = (\d+), FN = (\d+)", log)
    if not m:
        print(log[-3000:])
        return 1
    theirs = tuple(map(int, m.groups()))
    same_key = score_without_section(gold, pred, docs)
    print(f"cem.py（節を区別する・CEM の定義）       TP {ours.tp}  FP {ours.fp}  FN {ours.fn}  F1 {ours.f1:.4f}")
    print(f"同じ予測を公式と同じキーで数え直した値   TP {same_key[0]}  FP {same_key[1]}  FN {same_key[2]}")
    print(f"BC7 公式スクリプト                       TP {theirs[0]}  FP {theirs[1]}  FN {theirs[2]}")
    print(f"title と abstract で重なる言及（公式スクリプトでは 1 件に数える）: 正解 {collisions(gold, docs)} 件・予測 {collisions(pred, docs)} 件")
    ok = same_key == theirs
    print("一致: 公式スクリプトと同じキーで数えると TP / FP / FN が完全に一致する" if ok else "不一致: 原因を調べる")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
