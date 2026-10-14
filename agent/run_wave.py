"""条件 b を並列に走らせる（1 つの波）。

回ごとに別の GCP プロジェクト（slot。run_b.SLOT_PROJECTS）を割り当てて run_b.py を同時に起動する。すべて終わったら、
各プロジェクトに波の前に無かった資源が残っていないかを確かめて消し（監査ログで作成者を追えなかったもの・遅れて
記録されたものの受け皿）、どのプロジェクトのエージェントにも鍵が残っていないことを確かめる。

    uv run python agent/run_wave.py --mode main --task free --job gpt-5.2:none:0 --job gpt-5.2:medium:0

job は「モデル:effort:rep」。波の中の回の数は slot の数（run_b.SLOTS）まで。--first-slot で使う slot をずらせる
（別の波が走っているとき。その波の回の資源は、動いている回の印で掃除から外れる）。
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
    p.add_argument("--task", choices=tuple(run_b.RULE3), default="free")
    p.add_argument("--job", action="append", required=True, help="モデル:effort:rep")
    p.add_argument("--first-slot", type=int, default=0)
    p.add_argument("--plan-only", action="store_true", help="計画だけを立てさせる（run_b.py の --plan-only）")
    p.add_argument("--hours", type=float, default=4)
    p.add_argument("--stop-after", type=float, default=None, help="Codex を止めるまでの時間（run_b.py と同じ）")
    p.add_argument("--max-cost", type=float, default=20)
    p.add_argument("--account", default="ryota.yamada@science-aid.com")
    args = p.parse_args()
    slots = list(range(args.first_slot, args.first_slot + len(args.job)))
    if slots[-1] >= run_b.SLOTS:
        raise SystemExit(f"slot は 0〜{run_b.SLOTS - 1}")
    busy = [s for s in slots if run_b.slot_busy(s)]
    if busy:
        raise SystemExit(f"slot {busy} は使用中")

    started = run_b.now()
    waves = run_b.ROOT / "runs" / "agent" / "waves"
    waves.mkdir(parents=True, exist_ok=True)
    stamp = started.replace(":", "").replace("+0000", "Z")
    before = {s: run_b.inventory(run_b.slot_project(s), args.account) for s in slots}
    procs = []
    for slot, job in zip(slots, args.job):
        model, effort, rep = job.split(":")
        log = open(waves / f"{stamp}-slot{slot}.log", "w", encoding="utf-8")
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "run_b.py"), "--mode", args.mode, "--task", args.task,
               "--model", model,
               "--effort", effort, "--rep", rep, "--slot", str(slot), "--hours", str(args.hours),
               "--max-cost", str(args.max_cost), "--account", args.account] + (["--plan-only"] if args.plan_only else []) \
            + (["--stop-after", str(args.stop_after)] if args.stop_after else [])
        procs.append((job, slot, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"[{run_b.now()}] slot {slot}: {job} を起動した", flush=True)
        time.sleep(20)  # 鍵の作成・コンテナの起動が重ならないように少しずらす
    results = []
    for job, slot, proc, log in procs:
        code = proc.wait()
        log.close()
        results.append({"job": job, "slot": slot, "exit": code})
        print(f"[{run_b.now()}] slot {slot}: {job} が終わった（exit {code}）", flush=True)

    leftover, keys = {}, {}
    for s in slots:  # 他の波が使っている slot には触れない
        project = run_b.slot_project(s)
        leftover[project] = run_b.sweep(project, args.account, before[s])
        keys[project] = run_b.delete_user_keys(project, args.account, run_b.agent_sa(project))
    summary = {"started_at": started, "finished_at": run_b.now(), "mode": args.mode, "task": args.task,
               "plan_only": args.plan_only, "jobs": results,
               "remaining_after_sweep": leftover, "keys_deleted_after_runs": keys}
    (waves / f"{stamp}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
