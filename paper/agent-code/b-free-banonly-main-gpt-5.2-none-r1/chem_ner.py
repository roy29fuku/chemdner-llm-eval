#!/usr/bin/env python3
"""Rule-based chemical entity mention recognizer.

Constraints-friendly baseline:
- No external ML deps.
- Learns a lightweight gazetteer from training annotations.
- Adds a few regex detectors for formulas / CAS numbers / charged ions.

Outputs spans as (section, start, end).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple


Span = Tuple[int, int]


def iter_abstracts(path: str) -> Iterator[Tuple[str, str, str]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            pmid, title, abstract = line.rstrip("\n").split("\t")
            yield pmid, title, abstract


def iter_annotations(path: str) -> Iterator[Tuple[str, str, int, int, str, str]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            pmid, section, start, end, text, klass = line.rstrip("\n").split("\t")
            yield pmid, section, int(start), int(end), text, klass


_WORD_CHAR_RE = re.compile(r"[a-z0-9]")


def _is_word_char(ch: str) -> bool:
    return bool(_WORD_CHAR_RE.match(ch))


class _TrieNode:
    __slots__ = ("children", "end")

    def __init__(self) -> None:
        self.children: Dict[str, _TrieNode] = {}
        self.end = False


def _build_trie(keys: Sequence[str]) -> _TrieNode:
    root = _TrieNode()
    for key in keys:
        node = root
        for ch in key:
            node = node.children.setdefault(ch, _TrieNode())
        node.end = True
    return root


def _trie_find(text: str, trie: _TrieNode) -> List[Span]:
    """Find longest trie matches with strict word boundaries."""

    low = text.lower()
    spans: List[Span] = []
    n = len(low)
    for i in range(n):
        # Start boundary
        if i > 0 and _is_word_char(low[i - 1]) and _is_word_char(low[i]):
            continue

        node = trie
        j = i
        last_end = -1
        while j < n:
            ch = low[j]
            nxt = node.children.get(ch)
            if nxt is None:
                break
            node = nxt
            j += 1
            if node.end:
                # End boundary
                if j < n and _is_word_char(low[j - 1]) and _is_word_char(low[j]):
                    continue
                last_end = j
        if last_end != -1:
            spans.append((i, last_end))
    return spans


def _clean_span(text: str, span: Span) -> Span | None:
    start, end = span

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1

    # gold excludes trailing punctuation
    while end > start and text[end - 1] in ".,;:":
        end -= 1
    # and leading/trailing parens unless clearly part of a name (hard);
    # keeping it simple and consistent with guidelines.
    while start < end and text[start] in "([{":
        start += 1
    while end > start and text[end - 1] in ")]}":
        end -= 1

    if end - start < 2:
        return None
    return (start, end)


def _resolve_overlaps(spans: Iterable[Span]) -> List[Span]:
    uniq = sorted(set(spans), key=lambda s: (-(s[1] - s[0]), s[0], s[1]))
    chosen: List[Span] = []
    for start, end in uniq:
        if any(not (end <= cs or start >= ce) for cs, ce in chosen):
            continue
        chosen.append((start, end))
    chosen.sort()
    return chosen


@dataclass(frozen=True)
class RecognizerConfig:
    min_count: int = 3
    stoplist: frozenset[str] = frozenset(
        {
            # Very ambiguous short strings that explode false positives.
            "no",
            "co",
            "n",
            "o2",
            "h2",
            "c",
            "ca",
            "na",
            "k",
            "in",
            "as",
        }
    )


class ChemicalMentionRecognizer:
    RE_CAS = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
    RE_FORMULA = re.compile(r"\b(?:[A-Z][a-z]?\d{0,3}){2,}\b")
    RE_CHARGE = re.compile(r"\b[A-Z][a-z]?\(\d\+\)" r"|\b[A-Z][a-z]?\d*[+-]\b")

    def __init__(self, trie: _TrieNode, config: RecognizerConfig) -> None:
        self._trie = trie
        self._config = config

    @classmethod
    def from_training(
        cls, training_annotations_path: str, config: RecognizerConfig | None = None
    ) -> "ChemicalMentionRecognizer":
        config = config or RecognizerConfig()
        counts: Counter[str] = Counter()
        for _pmid, _sec, _s, _e, text, _klass in iter_annotations(training_annotations_path):
            key = text.lower()
            if len(key) >= 3:
                counts[key] += 1
        keys = [k for k, c in counts.items() if c >= config.min_count and k not in config.stoplist]
        keys.sort(key=len, reverse=True)
        trie = _build_trie(keys)
        return cls(trie, config)

    def predict_spans(self, text: str) -> List[Span]:
        spans: List[Span] = []

        spans.extend(_trie_find(text, self._trie))
        spans.extend((m.start(), m.end()) for m in self.RE_CAS.finditer(text))
        spans.extend((m.start(), m.end()) for m in self.RE_CHARGE.finditer(text))

        for m in self.RE_FORMULA.finditer(text):
            val = m.group(0)
            if any(ch.isdigit() for ch in val):
                spans.append((m.start(), m.end()))

        cleaned: List[Span] = []
        for sp in spans:
            c = _clean_span(text, sp)
            if c is not None:
                cleaned.append(c)

        return _resolve_overlaps(cleaned)


def predict_file(
    recognizer: ChemicalMentionRecognizer, abstracts_path: str
) -> Iterator[Tuple[str, str, int, int]]:
    for pmid, title, abstract in iter_abstracts(abstracts_path):
        for section, text in ("T", title), ("A", abstract):
            for start, end in recognizer.predict_spans(text):
                yield (pmid, section, start, end)

