"""Dictionary-based chemical mention tagger.

This solution intentionally avoids any external ML/NER models. It builds a
gazetteer from the provided training+development annotations and uses a fast
multi-pattern matcher (Aho–Corasick) to find exact string matches in the title
and abstract.

Because some very short tokens (e.g., "As", "In", "C") are highly ambiguous in
running text, we estimate their precision on the labeled train+dev set and only
keep those short patterns that are sufficiently reliable.

Usage:
  python3 /workspace/predict_mentions.py \
    --out /workspace/output/predictions.tsv
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable, Iterator


TRAIN_ABSTRACTS = Path("/workspace/data/training.abstracts.txt")
TRAIN_ANNOTATIONS = Path("/workspace/data/training.annotations.txt")
DEV_ABSTRACTS = Path("/workspace/data/development.abstracts.txt")
DEV_ANNOTATIONS = Path("/workspace/data/development.annotations.txt")
TEST_ABSTRACTS = Path("/workspace/data/test.abstracts.txt")


# The written-word forms are extremely ambiguous; keep element symbols like Au/Pb.
BLACKLIST_EXACT_LOWER = {"lead", "gold", "water", "aqueous", "proton"}


_RE_CAS = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
_RE_INCHI = re.compile(r"\bInChI=1S?/[A-Za-z0-9][A-Za-z0-9/\-\+\(\),\.]*")


def _boundary_ok(text: str, start: int, end: int) -> bool:
    """Reject matches that occur inside a larger alphanumeric token."""

    if not (0 <= start < end <= len(text)):
        return False
    if text[start].isalnum() and start > 0 and text[start - 1].isalnum():
        return False
    if text[end - 1].isalnum() and end < len(text) and text[end].isalnum():
        return False
    return True


def _not_blacklisted(text: str, start: int, end: int) -> bool:
    return text[start:end].lower() not in BLACKLIST_EXACT_LOWER


class ACAutomaton:
    """A small Aho–Corasick implementation for many exact string matches."""

    def __init__(self) -> None:
        self._next: list[dict[str, int]] = [{}]
        self._fail: list[int] = [0]
        self._out_lens: list[list[int]] = [[]]

    def add(self, pattern: str) -> None:
        state = 0
        for ch in pattern:
            nxt = self._next[state].get(ch)
            if nxt is None:
                nxt = len(self._next)
                self._next[state][ch] = nxt
                self._next.append({})
                self._fail.append(0)
                self._out_lens.append([])
            state = nxt
        self._out_lens[state].append(len(pattern))

    def build(self) -> None:
        q: deque[int] = deque()
        for nxt in self._next[0].values():
            q.append(nxt)
            self._fail[nxt] = 0

        while q:
            r = q.popleft()
            for ch, u in self._next[r].items():
                q.append(u)
                v = self._fail[r]
                while v and ch not in self._next[v]:
                    v = self._fail[v]
                self._fail[u] = self._next[v].get(ch, 0)
                self._out_lens[u].extend(self._out_lens[self._fail[u]])

    def find(self, text: str) -> Iterator[tuple[int, int]]:
        state = 0
        for i, ch in enumerate(text):
            while state and ch not in self._next[state]:
                state = self._fail[state]
            state = self._next[state].get(ch, 0)
            if not self._out_lens[state]:
                continue
            end = i + 1
            for plen in self._out_lens[state]:
                yield end - plen, end


def load_abstracts(path: Path) -> Iterator[tuple[str, str, str]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pmid, title, abstract = line.rstrip("\n").split("\t")
            yield pmid, title, abstract


def load_patterns(*annotation_paths: Path) -> set[str]:
    patterns: set[str] = set()
    for path in annotation_paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                cols = line.rstrip("\n").split("\t")
                patterns.add(cols[4])
    return patterns


def estimate_short_token_precision(
    texts: Iterable[str],
    annotation_paths: Iterable[Path],
) -> dict[str, float]:
    """Estimate how often 1-2 letter alphabetic tokens are actually chemicals."""

    token_re = re.compile(r"\b[A-Za-z]{1,2}\b")
    occurrences: Counter[str] = Counter()
    for txt in texts:
        for m in token_re.finditer(txt):
            occurrences[m.group(0)] += 1

    annotated: Counter[str] = Counter()
    for path in annotation_paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                mention = line.rstrip("\n").split("\t")[4]
                if re.fullmatch(r"[A-Za-z]{1,2}", mention):
                    annotated[mention] += 1

    precision: dict[str, float] = {}
    for tok, occ in occurrences.items():
        if occ:
            precision[tok] = annotated[tok] / occ
    return precision


@dataclass(frozen=True)
class Mention:
    pmid: str
    section: str  # "T" or "A"
    start: int
    end: int

    def to_tsv(self) -> str:
        return f"{self.pmid}\t{self.section}:{self.start}:{self.end}"


def build_matcher(short_token_precision_threshold: float) -> ACAutomaton:
    patterns = load_patterns(TRAIN_ANNOTATIONS, DEV_ANNOTATIONS)

    # Gather all train+dev texts for short-token precision estimation.
    all_texts: list[str] = []
    for _, title, abstract in load_abstracts(TRAIN_ABSTRACTS):
        all_texts.append(title)
        all_texts.append(abstract)
    for _, title, abstract in load_abstracts(DEV_ABSTRACTS):
        all_texts.append(title)
        all_texts.append(abstract)

    short_prec = estimate_short_token_precision(
        texts=all_texts,
        annotation_paths=[TRAIN_ANNOTATIONS, DEV_ANNOTATIONS],
    )

    keep: set[str] = set()
    for pat in patterns:
        if re.fullmatch(r"[A-Za-z]{1,2}", pat):
            if short_prec.get(pat, 0.0) >= short_token_precision_threshold:
                keep.add(pat)
        else:
            keep.add(pat)

    ac = ACAutomaton()
    for pat in keep:
        ac.add(pat)
    ac.build()
    return ac


def predict_mentions(ac: ACAutomaton, pmid: str, title: str, abstract: str) -> list[Mention]:
    out: list[Mention] = []

    for section, text in (("T", title), ("A", abstract)):
        spans: set[tuple[int, int]] = set()
        for start, end in ac.find(text):
            if _boundary_ok(text, start, end) and _not_blacklisted(text, start, end):
                spans.add((start, end))

        # High-precision identifier patterns that may be absent from the gazetteer.
        for m in _RE_CAS.finditer(text):
            spans.add((m.start(), m.end()))
        for m in _RE_INCHI.finditer(text):
            spans.add((m.start(), m.end()))

        for start, end in sorted(spans):
            out.append(Mention(pmid=pmid, section=section, start=start, end=end))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/workspace/output/predictions.tsv"),
        help="Output TSV path",
    )
    ap.add_argument(
        "--short-token-threshold",
        type=float,
        default=0.40,
        help="Min precision (estimated on train+dev) to keep 1-2 letter alpha patterns",
    )
    args = ap.parse_args()

    ac = build_matcher(short_token_precision_threshold=args.short_token_threshold)

    predictions: list[str] = []
    for pmid, title, abstract in load_abstracts(TEST_ABSTRACTS):
        for m in predict_mentions(ac, pmid, title, abstract):
            predictions.append(m.to_tsv())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

