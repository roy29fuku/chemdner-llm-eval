"""実験 15（a'・b(128)・例の数の曲線）を test で走らせる前に、例・文書・回の一覧を凍結する。

原稿 repo の docs/experiment-protocol.md「追加の対照（実験 15）」の手順 1。書き出すもの:
- frozen/official2478.txt: test の正解のある 2,478 件の PMID（公式の採点範囲。a' と例なしの再実行はこの文書で走らせる）
- frozen/curve1000.txt: 2,478 件から種 1 で選んだ 1,000 件（例の数の曲線の文書）
- frozen/b128-pmids.txt: b(128) に渡す正解付きの 128 件（a'(乱数, 128) の種 1 の例と同じ）
- frozen/aprime.json: 例の候補の hash と内訳、種ごとの例の PMID（種 1 は曲線の最大の 1,024 件まで）、走らせる回の一覧、
  文書の一覧の hash、プロンプトを組み立てるコードの hash

test の正解は、文書の一覧（正解のある文書）を作るためにだけ読む。予測には使わない。
書き出した後に commit し、GitHub に push して tag を打ってから、test で走らせる（direct.check_aprime が確かめる）。

    uv run python scripts/freeze_aprime.py
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemdner_bench import cem, direct  # noqa: E402
from chemdner_bench.corpus import gold_spans, load  # noqa: E402
from chemdner_bench.shots import random_examples, standard_pool, subset_pmids  # noqa: E402

FROZEN = ROOT / "frozen"
SEEDS = (1, 2, 3)
CURVE_K = (8, 32, 256, 512, 1024)
MAIN = [("gpt-6-luna", "medium"), ("gpt-6.1-sol", "low"), ("gpt-6.1-sol", "medium")]
CODE = ["src/chemdner_bench/shots.py", "src/chemdner_bench/direct.py", "src/chemdner_bench/llm.py",
        "src/chemdner_bench/inline_tags.py", "prompts/direct/instructions.md", "task/guidelines.md"]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_list(name: str, pmids: list[str]) -> str:
    text = "\n".join(pmids) + "\n"
    (FROZEN / name).write_text(text, encoding="utf-8")
    return sha(text)


def main() -> int:
    official = cem.official_pmids(gold_spans(load("evaluation")))
    curve = subset_pmids(official, 1000, seed=1)
    pool, pool_info = standard_pool(load("training"), load("development"))
    examples = {seed: [d.pmid for d in random_examples(pool, max(CURVE_K) if seed == 1 else 128, seed)] for seed in SEEDS}
    hashes = {"official2478.txt": write_list("official2478.txt", official), "curve1000.txt": write_list("curve1000.txt", curve),
              "b128-pmids.txt": write_list("b128-pmids.txt", examples[1][:128])}

    off, cur = str(FROZEN / "official2478.txt"), str(FROZEN / "curve1000.txt")
    runs = []
    for model, effort in MAIN:
        runs.append(direct.Config("evaluation", model, effort, rep=10, pmids_file=off).name())  # 例なしの再実行
        for seed in SEEDS:
            runs.append(direct.Config("evaluation", model, effort, shots=128, select="random", shot_seed=seed, pmids_file=off).name())
    runs.append(direct.Config("evaluation", "gpt-6-luna", "medium", shots=128, select="similar", pmids_file=off).name())
    runs.append(direct.Config("evaluation", "gpt-6.1-sol", "medium", shots=32, select="similar", pmids_file=off).name())
    for k in CURVE_K:
        runs.append(direct.Config("evaluation", "gpt-6-luna", "medium", shots=k, select="random", shot_seed=1, pmids_file=cur).name())

    data = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "paper-bppb-2026-hands-and-yardstick docs/experiment-protocol.md 追加の対照（実験 15）",
        "pool": {**pool_info, "sha256": sha("\n".join(d.pmid for d in pool))},
        "examples": {str(s): v for s, v in examples.items()},
        "lists": hashes,
        "runs": runs,
        "code_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in CODE},
    }
    (FROZEN / "aprime.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"凍結した: 回 {len(runs)}・候補 {pool_info['pool']}・正解のある文書 {len(official)}・曲線の文書 {len(curve)}")
    print("commit して push し、tag を打ってから test で走らせる")
    return 0


if __name__ == "__main__":
    sys.exit(main())
