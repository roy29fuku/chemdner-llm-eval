"""run_b.py が後片付けの後、summary.json を書く直前に止まった回の summary.json を、記録から組み立て直す。

2026-10-20 の b(128)（実験 15）で、--labeled の値（Path）を JSON に書けずに run_b.py が止まった（後で直した）。
Codex の実行・後片付け（エージェントの資源の削除・鍵の削除）・通信と DNS の記録は、その前に済んでいる。失われたのは
メモリの上の値だけなので、run_b.py と同じ計算を記録から行い、分からない値は null にして、組み立て直したことを
recovered に書く。

- 起動の引数: 波の記録（runs/agent/waves/<stamp>.json と <stamp>-slot<N>.log）と、run_wave.py に渡した引数
- 作った資源: run_b.owned と同じく監査ログから（since は波の開始。slot はこの回だけが使っていた）
- 後片付けの後の残り: 波の最後の掃除の結果（remaining_after_sweep）
- Codex の終了コード: 分からない（null）。codex.jsonl の最後の出来事を残す

    uv run python agent/recover_summary.py runs/agent/<run> --wave runs/agent/waves/<stamp>.json --slot 1 --rep 2 \\
        --tag gcp-b128 --labeled frozen/b128-pmids.txt
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_b  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--wave", type=Path, required=True, help="この回を起動した波の記録（runs/agent/waves/<stamp>.json）")
    p.add_argument("--slot", type=int, required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--effort", default="medium")
    p.add_argument("--tag", default="")
    p.add_argument("--labeled", default=None)
    p.add_argument("--hours", type=float, default=4)
    p.add_argument("--max-cost", type=float, default=20)
    p.add_argument("--account", default="ryota.yamada@science-aid.com", help="監査ログを読む gcloud のアカウント")
    p.add_argument("--reason", required=True, help="組み立て直す理由（止まったときのエラー）")
    args = p.parse_args()

    run = args.run
    if (run / "summary.json").exists():
        raise SystemExit(f"{run}/summary.json がすでにある")
    wave = json.loads(args.wave.read_text())
    slot_log = args.wave.with_name(f"{args.wave.stem}-slot{args.slot}.log")
    project = run_b.slot_project(args.slot)
    sa = run_b.agent_sa(project)
    since = wave["started_at"]

    events = [json.loads(line) for line in (run / "logs" / "codex.jsonl").read_text().splitlines() if line.strip()]
    usage = [json.loads(line) for line in (run / "logs" / "openai_usage.jsonl").read_text().splitlines()] \
        if (run / "logs" / "openai_usage.jsonl").exists() else []
    flows = [json.loads(line) for line in (run / "logs" / "flows.jsonl").read_text().splitlines()] \
        if (run / "logs" / "flows.jsonl").exists() else []
    pred_path = run / "workspace" / "output" / "predictions.tsv"
    created = {k: sorted(v) for k, v in run_b.owned(project, args.account, sa, since).items()}

    meta = {
        "mode": wave["mode"], "task": wave["task"], "plan_only": wave["plan_only"], "tag": args.tag,
        "labeled": args.labeled, "hours": args.hours, "stop_after": None, "max_cost": args.max_cost,
        "max_gpus": 1, "max_vcpus": 24, "max_project_gpus": 1, "model": args.model, "effort": args.effort,
        "rep": args.rep, "slot": args.slot, "account": "", "agent_cpus": "6", "agent_memory": "24g",
        "project": project, "agent_sa": sa, "started_at": None, "prompt": run_b.PROMPT, "image": run_b.IMAGE,
        "proxy_image": run_b.PROXY_IMAGE, "task_template_sha256": hashlib.sha256(run_b.TASK.read_bytes()).hexdigest(),
        "task_sha256": hashlib.sha256((run / "workspace" / "task.md").read_bytes()).hexdigest(),
        "codex_exit": None, "finished_at": None,
        "remaining_after_cleanup": wave["remaining_after_sweep"][project],
        "created": created,
        "openai_cost_usd": usage[-1]["total_usd"] if usage else 0.0,
        "blocked_requests": sorted({(f["host"], f["blocked"]) for f in flows if f.get("blocked")}),
        "predictions": pred_path.exists(),
        "recovered": {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": args.reason,
            "wave": str(args.wave), "slot_log_first_line": slot_log.read_text().splitlines()[0] if slot_log.exists() else None,
            "wave_job": next((j for j in wave["jobs"] if j["slot"] == args.slot), None),
            "created_since": since,
            "codex_last_event": events[-1].get("type") if events else None,
            "remaining_after_cleanup_from": "波の最後の掃除（remaining_after_sweep）",
            "unknown": ["started_at", "finished_at", "codex_exit"],
        },
    }
    (run / "summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=list) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("openai_cost_usd", "created", "remaining_after_cleanup", "predictions")},
                     indent=2, ensure_ascii=False, default=list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
