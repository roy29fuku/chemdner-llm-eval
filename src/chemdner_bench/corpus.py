"""CHEMDNER コーパスの取得と読み込み。

コーパスの本体は repo に置かない（明示されたライセンスが無く、本文は PubMed のアブストラクトのため）。
NCBI が BioCreative VII の NLM-Chem トラックで配布している BioC 版を取得し、SHA-256 で原本と同一かを確かめてから使う。
この配布ファイルは BioCreative IV（2013）の CHEMDNER を BioC に変換したもので、本文は公式とバイト単位で一致し、
化学物質の言及は公式より 24 件少ない（training −16・development −3・evaluation −5）。

    uv run python -m chemdner_bench.corpus fetch            # NCBI から取得して data/ に置く
    uv run python -m chemdner_bench.corpus fetch --source <手元のファイル>   # 取得済みのファイルを使う
    uv run python -m chemdner_bench.corpus stats            # 分割ごとの件数を表示する
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

URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-CHEMDNER-corpus_v2.BioC.xml.gz"
SHA256 = "2662ff1a984ce5202af557cce71548ce7a910003b77a41229c9d621b0b4bd086"
FILENAME = "BC7T2-CHEMDNER-corpus_v2.BioC.xml.gz"  # 拡張子は .xml.gz だが中身は非圧縮の tar
SPLITS = ("training", "development", "evaluation")  # evaluation が公式の test（3,000 件）
SECTIONS = {"title": "T", "abstract": "A"}  # 公式の CEM と同じく、文字位置は節ごとに 0 から数える


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


def corpus_path() -> Path:
    return data_dir() / FILENAME


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(source: Path | None = None) -> Path:
    """配布ファイルを data/ に置き、SHA-256 が原本と一致することを確かめる。"""
    dest = corpus_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        shutil.copyfile(source, dest)
    elif not dest.exists():
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(URL) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(dest)
    digest = sha256(dest)
    if digest != SHA256:
        raise ValueError(f"SHA-256 が原本と一致しない: {digest}（期待値 {SHA256}）")
    return dest


def parse_bioc(raw: bytes) -> list[Document]:
    """BioC XML を読む。CHEMDNER の注釈（type=Chemical）だけを言及として取り出す。

    NCBI が足した MeSH_Indexing_Chemical（BC7 の索引付けタスク用）は CHEMDNER の注釈ではないので捨てる。
    宣言が encoding="utf8" で標準の XML パーサが読めないため、"utf-8" に置き換えてから読む。
    """
    root = ET.fromstring(raw.replace(b'encoding="utf8"', b'encoding="utf-8"', 1))
    docs = []
    for d in root.findall("document"):
        texts = {}
        mentions = []
        for p in d.findall("passage"):
            ptype = {i.get("key"): i.text for i in p.findall("infon")}.get("type")
            section = SECTIONS[ptype]
            text = p.findtext("text") or ""
            texts[section] = text
            base = int(p.findtext("offset"))
            for a in p.findall("annotation"):
                infons = {i.get("key"): i.text for i in a.findall("infon")}
                if infons.get("type") != "Chemical":
                    continue
                for loc in a.findall("location"):
                    start = int(loc.get("offset")) - base
                    end = start + int(loc.get("length"))
                    mention = Mention(section, start, end, a.findtext("text") or "", infons.get("class") or "")
                    if text[start:end] != mention.text:
                        raise ValueError(f"{d.findtext('id')}: 位置 {section}:{start}:{end} の文字列が注釈と一致しない")
                    mentions.append(mention)
        docs.append(Document(d.findtext("id"), texts.get("T", ""), texts.get("A", ""), mentions))
    return docs


def load(split: str, path: Path | None = None) -> list[Document]:
    if split not in SPLITS:
        raise ValueError(f"split は {SPLITS} のどれか: {split}")
    path = path or corpus_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} が無い。先に `python -m chemdner_bench.corpus fetch` を実行する")
    with tarfile.open(path) as tar:
        raw = tar.extractfile(f"BC7T2-CHEMDNER-corpus-{split}.BioC.xml").read()
    return parse_bioc(raw)


def gold_spans(docs: list[Document]) -> dict[str, set[tuple[str, int, int]]]:
    """採点器（cem.py）が読む形の正解。言及の無い文書も空の集合として含める。"""
    return {d.pmid: {(m.section, m.start, m.end) for m in d.mentions} for d in docs}


def main(argv: list[str] | None = None) -> int:
    from chemdner_bench.cem import write_tsv

    parser = argparse.ArgumentParser(prog="python -m chemdner_bench.corpus")
    sub = parser.add_subparsers(dest="command", required=True)
    p_fetch = sub.add_parser("fetch", help="NCBI から取得して SHA-256 を確かめる")
    p_fetch.add_argument("--source", type=Path, help="取得済みのファイルを使う（ネットワークに出ない）")
    sub.add_parser("stats", help="分割ごとのアブストラクト数・言及数を表示する")
    p_gold = sub.add_parser("gold", help="正解を採点器の形式（PMID<TAB>節:開始:終了）で書き出す")
    p_gold.add_argument("--split", choices=SPLITS, required=True)
    p_gold.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        print(f"ok: {fetch(args.source)}")
    elif args.command == "gold":
        docs = load(args.split)
        write_tsv(args.out, gold_spans(docs))
        print(f"wrote {args.out} ({len(docs)} abstracts, {sum(len(d.mentions) for d in docs)} mentions)")
    else:
        for split in SPLITS:
            docs = load(split)
            print(f"{split:<12} abstracts {len(docs):>5,}  mentions {sum(len(d.mentions) for d in docs):>6,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
