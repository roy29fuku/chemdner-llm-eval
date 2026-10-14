"""条件 b を並列に走らせる（1 つの波）。

回ごとに別のエージェント用サービスアカウント（slot）を割り当てて run_b.py を同時に起動する。すべて終わったら、
波の前に無かった資源がプロジェクトに残っていないかを確かめて消し（監査ログで作成者を追えなかったもの・遅れて
記録されたものの受け皿）、どの slot にも鍵が残っていないことを確かめる。

    uv run python agent/run_wave.py --mode main --job gpt-5.2:none:0 --job gpt-5.2:medium:0 --job gpt-6-luna:none:0

job は「モデル:effort:rep」。波の中の回の数は slot の数（run_b.SLOTS）まで。
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("pilot", "main"), required=True)
    p.add_argument("--job", action="append", required=True, help="モデル:effort:rep")
    p.add_argument("--hours", type=float, default=4)
    p.add_argument("--max-cost", type=float, default=20)
    p.add_argument("--project", default="chemdner-biobert-training")
    p.add_argument("--account", default="ryota.yamada@science-aid.com")
    args = p.parse_args()
    if len(args.job) > run_b.SLOTS:
        raise SystemExit(f"1 つの波は {run_b.SLOTS} 回まで")

    started = run_b.now()
    waves = run_b.ROOT / "runs" / "agent" / "waves"
    waves.mkdir(parents=True, exist_ok=True)
    stamp = started.replace(":", "").replace("+0000", "Z")
    before = run_b.inventory(args.project, args.account)
    procs = []
    for slot, job in enumerate(args.job):
        model, effort, rep = job.split(":")
        log = open(waves / f"{stamp}-slot{slot}.log", "w", encoding="utf-8")
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "run_b.py"), "--mode", args.mode, "--model", model,
               "--effort", effort, "--rep", rep, "--slot", str(slot), "--hours", str(args.hours),
               "--max-cost", str(args.max_cost), "--project", args.project, "--account", args.account]
        procs.append((job, slot, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"[{run_b.now()}] slot {slot}: {job} を起動した", flush=True)
        time.sleep(20)  # 鍵の作成・コンテナの起動が重ならないように少しずらす
    results = []
    for job, slot, proc, log in procs:
        code = proc.wait()
        log.close()
        results.append({"job": job, "slot": slot, "exit": code})
        print(f"[{run_b.now()}] slot {slot}: {job} が終わった（exit {code}）", flush=True)

    leftover = run_b.sweep(args.project, args.account, before)
    keys = {run_b.agent_sa(args.project, s): run_b.delete_user_keys(args.project, args.account, run_b.agent_sa(args.project, s))
            for s in range(run_b.SLOTS)}
    summary = {"started_at": started, "finished_at": run_b.now(), "mode": args.mode, "jobs": results,
               "remaining_after_sweep": leftover, "keys_deleted_after_runs": keys}
    (waves / f"{stamp}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
