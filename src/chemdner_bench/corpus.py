"""CHEMDNER コーパスの取得と読み込み。

コーパスの本体は repo に置かない（明示されたライセンスが無く、本文は PubMed のアブストラクトのため）。読み込める版は 2 つ。

- official: 2013 年の BioCreative IV の公式配布物（正本）。件数が公式の表と完全に一致する。元の配布元は閉鎖されて
  いて公開では手に入らないので、手元のコピー（data/official-2013/ か環境変数 CHEMDNER_OFFICIAL_DIR）から読む
- ncbi: NCBI が BioCreative VII の NLM-Chem トラックで公開している BioC 版。公開で手に入るので、第三者が再現する
  ときの代わりになる。本文は公式とバイト単位で同一だが、各分割で 1 文書ずつ言及がまとめて欠けている
  （training: PMID 23645249 の 16 件、development: 23646356 の 3 件、evaluation: 23645677 の 5 件）

どちらも SHA-256 で原本と同一かを確かめる。evaluation には長さ 0 の正解が 4 件（いずれも MULTIPLE）あり、
公式の時点で長さ 0 なので、そのまま読む（公式の採点ではどのシステムにも取れない FN になる）。

    uv run python -m chemdner_bench.corpus fetch --official <official-2013 のディレクトリ>   # 公式配布物を data/ に写す
    uv run python -m chemdner_bench.corpus fetch --ncbi                                      # NCBI 版を取得する
    uv run python -m chemdner_bench.corpus stats [--source ncbi]
"""

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

SPLITS = ("training", "development", "evaluation")  # evaluation が公式の test（3,000 件）
SOURCES = ("official", "ncbi")
SECTIONS = {"title": "T", "abstract": "A"}  # 公式の CEM と同じく、文字位置は節ごとに 0 から数える

# 2013 年の公式配布物（BioCreative IV CHEMDNER）。パスは official-2013/ からの相対
OFFICIAL_SHA256 = {
    "chemdner_corpus/training.abstracts.txt": "a1d3b12eadc92a2c6311c2021b3f824f007f08686720611c65c26b306647dc6f",
    "chemdner_corpus/training.annotations.txt": "9df6472b75d9914bd68a51dadacd240eb61500b5a437f60a6315037318a7f54d",
    "chemdner_corpus/development.abstracts.txt": "2cd33eb283b846f3ef128e40dea24806bf1e2595d660a18ee8cfcad4bd9abf87",
    "chemdner_corpus/development.annotations.txt": "6ceb3c76fb4a30cbcaf8fbf1eb40477bb14cb8dec5f73ba8c7d3b35fd29c5b25",
    "chemdner_corpus/evaluation.abstracts.txt": "14674c6c01704f53bd5ee3b74427c90578f5d7fecbf08d9a2780c0731b2eb916",
    "chemdner_corpus/evaluation.annotations.txt": "fb8b95dfceda919c934908cbc722dc6a946a812db51bd4a640872ffa1e689335",
    "chemdner_corpus/cem_ann_test_13-09-13.txt": "b254a373c6d9d3568d158a0b2a07777f5b40f1ce805a5132fe0fae0fd8d547a6",
    "CHEMDNER_SAMPLE_JUNE25/chemdner_sample_cem_gold_standard.txt": "0a8d9dae8025e97ef255e6b5fcb4ec68ca6dce8e7f8c63b28fb53f05423299a3",
    "CHEMDNER_SAMPLE_JUNE25/chemdner_cem_prediction_1.txt": "8708353bf84c7f4efce35af71915283820a0ab448c01dbaf6fb3a35ffe1e3c59",
    "CHEMDNER_SAMPLE_JUNE25/chemdner_cem_prediction_1_EVAL.txt": "6e8d157561d6d32f0b1ca516f35d36cabb562b62b33acfb4ff5c79ae5f3fb75a",
}

NCBI_URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-CHEMDNER-corpus_v2.BioC.xml.gz"
NCBI_SHA256 = "2662ff1a984ce5202af557cce71548ce7a910003b77a41229c9d621b0b4bd086"
NCBI_FILENAME = "BC7T2-CHEMDNER-corpus_v2.BioC.xml.gz"  # 拡張子は .xml.gz だが中身は非圧縮の tar


@dataclass(frozen=True)
class Mention:
    section: str  # "T"（title）か "A"（abstract）
    start: int
    end: int
    text: str
    cls: str = ""  # 公式の 8 分類（TRIVIAL など）。採点には使わない


@dataclass
class Document:
    pmid: str
    title: str
    abstract: str
    mentions: list[Mention] = field(default_factory=list)

    def text(self, section: str) -> str:
        return self.title if section == "T" else self.abstract


def data_dir() -> Path:
    return Path(os.environ.get("CHEMDNER_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


def official_dir() -> Path:
    return Path(os.environ.get("CHEMDNER_OFFICIAL_DIR", data_dir() / "official-2013"))


def ncbi_path() -> Path:
    return data_dir() / NCBI_FILENAME


def available(source: str) -> bool:
    if source == "official":
        return all((official_dir() / rel).exists() for rel in OFFICIAL_SHA256)
    return ncbi_path().exists()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_official(source_dir: Path) -> Path:
    """公式配布物を data/official-2013/ に写し、全ファイルの SHA-256 を確かめる。"""
    dest = data_dir() / "official-2013"
    for rel, digest in OFFICIAL_SHA256.items():
        src = Path(source_dir) / rel
        if sha256(src) != digest:
            raise ValueError(f"{src} の SHA-256 が原本と一致しない")
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest / rel)
    return dest


def fetch_ncbi(source: Path | None = None) -> Path:
    """NCBI 版を data/ に置き（取得済みのファイルを渡せばネットワークに出ない）、SHA-256 を確かめる。"""
    dest = ncbi_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        shutil.copyfile(source, dest)
    elif not dest.exists():
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(NCBI_URL) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(dest)
    digest = sha256(dest)
    if digest != NCBI_SHA256:
        raise ValueError(f"SHA-256 が原本と一致しない: {digest}（期待値 {NCBI_SHA256}）")
    return dest


def _check(doc: Document, m: Mention) -> None:
    if doc.text(m.section)[m.start:m.end] != m.text:
        raise ValueError(f"{doc.pmid}: 位置 {m.section}:{m.start}:{m.end} の文字列が注釈と一致しない")


def parse_official(abstracts: str, annotations: str) -> list[Document]:
    """公式の形式を読む。abstracts は「PMID<TAB>タイトル<TAB>本文」、annotations は「PMID<TAB>節<TAB>開始<TAB>終了<TAB>文字列<TAB>クラス」。"""
    docs: dict[str, Document] = {}
    for line in abstracts.splitlines():
        if line.strip():
            pmid, title, abstract = line.split("\t")
            docs[pmid] = Document(pmid, title, abstract)
    for line in annotations.splitlines():
        if line.strip():
            pmid, section, start, end, text, cls = line.split("\t")
            m = Mention(section, int(start), int(end), text, cls)
            _check(docs[pmid], m)
            docs[pmid].mentions.append(m)
    return list(docs.values())


def parse_bioc(raw: bytes) -> list[Document]:
    """NCBI の BioC XML を読む。CHEMDNER の注釈（type=Chemical）だけを言及として取り出す。

    NCBI が足した MeSH_Indexing_Chemical（BC7 の索引付けタスク用）は CHEMDNER の注釈ではないので捨てる。
    宣言が encoding="utf8" で標準の XML パーサが読めないため、"utf-8" に置き換えてから読む。
    """
    root = ET.fromstring(raw.replace(b'encoding="utf8"', b'encoding="utf-8"', 1))
    docs = []
    for d in root.findall("document"):
        doc = Document(d.findtext("id"), "", "")
        for p in d.findall("passage"):
            section = SECTIONS[{i.get("key"): i.text for i in p.findall("infon")}.get("type")]
            if section == "T":
                doc.title = p.findtext("text") or ""
            else:
                doc.abstract = p.findtext("text") or ""
            base = int(p.findtext("offset"))
            for a in p.findall("annotation"):
                infons = {i.get("key"): i.text for i in a.findall("infon")}
                if infons.get("type") != "Chemical":
                    continue
                for loc in a.findall("location"):
                    start = int(loc.get("offset")) - base
                    doc.mentions.append(Mention(section, start, start + int(loc.get("length")),
                                                a.findtext("text") or "", infons.get("class") or ""))
        for m in doc.mentions:
            _check(doc, m)
        docs.append(doc)
    return docs


def load(split: str, source: str = "official") -> list[Document]:
    if split not in SPLITS:
        raise ValueError(f"split は {SPLITS} のどれか: {split}")
    if source not in SOURCES:
        raise ValueError(f"source は {SOURCES} のどれか: {source}")
    if not available(source):
        hint = "fetch --official <official-2013 のディレクトリ>" if source == "official" else "fetch --ncbi"
        raise FileNotFoundError(f"{source} のコーパスが無い。先に `python -m chemdner_bench.corpus {hint}` を実行する")
    if source == "official":
        base = official_dir() / "chemdner_corpus"
        return parse_official((base / f"{split}.abstracts.txt").read_text(encoding="utf-8"),
                              (base / f"{split}.annotations.txt").read_text(encoding="utf-8"))
    with tarfile.open(ncbi_path()) as tar:
        return parse_bioc(tar.extractfile(f"BC7T2-CHEMDNER-corpus-{split}.BioC.xml").read())


def gold_spans(docs: list[Document]) -> dict[str, set[tuple[str, int, int]]]:
    """採点器（cem.py）が読む形の正解。言及の無い文書も空の集合として含める。"""
    return {d.pmid: {(m.section, m.start, m.end) for m in d.mentions} for d in docs}


def main(argv: list[str] | None = None) -> int:
    from chemdner_bench.cem import write_tsv

    parser = argparse.ArgumentParser(prog="python -m chemdner_bench.corpus")
    sub = parser.add_subparsers(dest="command", required=True)
    p_fetch = sub.add_parser("fetch", help="コーパスを data/ に置いて SHA-256 を確かめる")
    g = p_fetch.add_mutually_exclusive_group(required=True)
    g.add_argument("--official", type=Path, metavar="DIR", help="公式配布物（official-2013/）のディレクトリから写す")
    g.add_argument("--ncbi", nargs="?", const=True, metavar="FILE", help="NCBI から取得する（取得済みのファイルを渡せばそれを使う）")
    p_stats = sub.add_parser("stats", help="分割ごとのアブストラクト数・言及数を表示する")
    p_stats.add_argument("--source", choices=SOURCES, default="official")
    p_out = sub.add_parser("gold", help="正解を採点器の形式（PMID<TAB>節:開始:終了）で、文書の一覧を 1 行 1 件で書き出す")
    p_out.add_argument("--split", choices=SPLITS, required=True)
    p_out.add_argument("--source", choices=SOURCES, default="official")
    p_out.add_argument("--out", type=Path, required=True)
    p_out.add_argument("--pmids", type=Path, help="全文書の PMID の一覧の書き出し先（副の採点 --scope all で使う）")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        if args.official:
            print(f"ok: {fetch_official(args.official)}")
        else:
            print(f"ok: {fetch_ncbi(None if args.ncbi is True else Path(args.ncbi))}")
    elif args.command == "gold":
        docs = load(args.split, args.source)
        write_tsv(args.out, gold_spans(docs))
        if args.pmids:
            args.pmids.write_text("".join(f"{d.pmid}\n" for d in docs), encoding="utf-8")
        print(f"wrote {args.out} ({len(docs)} abstracts, {sum(len(d.mentions) for d in docs)} mentions)")
    else:
        for split in SPLITS:
            docs = load(split, args.source)
            n_with = sum(1 for d in docs if d.mentions)
            print(f"{split:<12} abstracts {len(docs):>5,}  with mentions {n_with:>5,}  "
                  f"mentions {sum(len(d.mentions) for d in docs):>6,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
