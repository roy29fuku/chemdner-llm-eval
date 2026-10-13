"""インラインタグ（`<chem>…</chem>`）と文字位置の相互変換。

条件 a-A では、LLM に原文の化学物質名を `<chem>` と `</chem>` で囲んだ全文を返させる。ここでは
1. タグ付きの文を、タグを外した文と言及の位置に分解する（parse）
2. タグを外した文が原文と違うとき、決定的な整列で原文の位置に戻す（align）
3. 正解の位置から、例として見せるタグ付きの文を作る（tag）
を行う。形式の違反（入れ子・閉じていないタグなど）は黙って直さず、件数を数えて返す。
"""

import difflib
import re
from dataclasses import dataclass, field

OPEN, CLOSE = "<chem>", "</chem>"
_TAG = re.compile(r"</?chem>")


@dataclass
class Parsed:
    plain: str  # タグを外した文
    spans: list[tuple[int, int]]  # plain 上の言及の位置
    violations: list[str] = field(default_factory=list)


@dataclass
class Aligned:
    spans: list[tuple[int, int]]  # 原文上の言及の位置
    text_matches: bool  # タグを外した文が原文と完全に一致したか
    dropped: int  # 原文に戻せず落とした言及の数


def parse(tagged: str) -> Parsed:
    plain: list[str] = []
    spans: list[tuple[int, int]] = []
    violations: list[str] = []
    pos = 0  # plain の長さ
    start: int | None = None
    last = 0
    for m in _TAG.finditer(tagged):
        piece = tagged[last:m.start()]
        plain.append(piece)
        pos += len(piece)
        last = m.end()
        if m.group() == OPEN:
            if start is not None:
                violations.append(f"nested-open@{pos}")  # 入れ子は認めない。外側のタグを生かす
                continue
            start = pos
        else:
            if start is None:
                violations.append(f"unmatched-close@{pos}")
                continue
            if pos > start:
                spans.append((start, pos))
            else:
                violations.append(f"empty@{pos}")
            start = None
    plain.append(tagged[last:])
    if start is not None:
        violations.append(f"unclosed@{start}")
    return Parsed("".join(plain), spans, violations)


def align(parsed: Parsed, original: str) -> Aligned:
    """タグを外した文の言及の位置を、原文の位置に戻す。

    文が一致すればそのまま使う。違う場合は文字単位の差分（difflib、autojunk なし）で一致した区間だけを対応させ、
    言及の両端が対応し、かつ原文から切り出した文字列が言及の文字列と同じときだけ採る。それ以外は落として数える。
    """
    if parsed.plain == original:
        return Aligned(list(parsed.spans), True, 0)
    mapping: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(None, parsed.plain, original, autojunk=False)
    for tag, i1, i2, j1, _ in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
    spans: list[tuple[int, int]] = []
    dropped = 0
    for s, e in parsed.spans:
        if s in mapping and (e - 1) in mapping:
            os_, oe = mapping[s], mapping[e - 1] + 1
            if original[os_:oe] == parsed.plain[s:e]:
                spans.append((os_, oe))
                continue
        dropped += 1
    return Aligned(spans, False, dropped)


def tag(text: str, spans: list[tuple[int, int]]) -> str:
    """原文に言及の位置のタグを入れる（例を作るため）。重なりのある言及は表せないので、渡す前に除く。"""
    ordered = sorted(spans)
    for (s1, e1), (s2, _) in zip(ordered, ordered[1:]):
        if s2 < e1:
            raise ValueError(f"重なりのある言及はタグで表せない: {(s1, e1)} と {(s2, _)}")
    out: list[str] = []
    last = 0
    for s, e in ordered:
        out.append(text[last:s])
        out.append(OPEN + text[s:e] + CLOSE)
        last = e
    out.append(text[last:])
    return "".join(out)
