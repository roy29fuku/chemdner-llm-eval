"""条件 b の基盤の確認（Codex に課題は解かせない）。

proxy とエージェントのコンテナを起動し、次を確かめてから片付ける:
- 読むだけの通信はどこにでも届く（PyPI・Hugging Face のモデル・一般のサイト）
- 書き込みを伴う通信は許可リストの宛先にだけ届く。他の LLM の API・Hugging Face のデータセット・CHEMDNER の名前を
  含む URL・BioBERT などが配るデータセットは 403 になる
- proxy を通らない直接の通信はできない
- gcloud がこの slot のプロジェクトのエージェント用サービスアカウントで使え、他の slot のプロジェクトは見えない
- 作った資源（バケット・小さな VM）を監査ログで追って消せる
- Codex が web 検索を使え、Codex が実行するコマンドからは OpenAI の API キーが見えない

    uv run python agent/smoke_test.py --slot 3
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b  # noqa: E402

CODE = "curl -s -o /dev/null -w '%{http_code}'"
CHECKS = [
    ("pypi に届く", f"{CODE} https://pypi.org/simple/numpy/", "200"),
    ("HF のモデルに届く", f"{CODE} -L https://huggingface.co/api/models/google-bert/bert-base-uncased", "200"),
    ("一般のサイトを読める", f"{CODE} https://www.example.com/", "200"),
    ("一般のサイトへの書き込みは遮断", f"{CODE} -X POST -d x=1 https://www.example.com/", "403"),
    ("他の LLM の API は遮断", f"{CODE} https://api.anthropic.com/v1/models", "403"),
    ("HF のデータセットは遮断", f"{CODE} https://huggingface.co/datasets/bigbio/chemdner", "403"),
    ("CHEMDNER の名前を含む URL は遮断", f"{CODE} https://github.com/lingluodlut/Att-ChemdNER", "403"),
    ("BioBERT の配布データセットは遮断",
     f"{CODE} http://nlp.dmis.korea.ac.kr/projects/biobert-2020-checkpoints/datasets.tar.gz", "403"),
    ("直接の通信はできない", f"{CODE} --noproxy '*' --max-time 10 https://pypi.org/ || true", "000"),
    ("gcloud が使える", "gcloud compute zones list --filter=region:us-central1 --format='value(name)' | head -1", "us-central1-"),
]


def main() -> int:
    from dotenv import load_dotenv

    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=run_b.SLOTS - 1)
    args = ap.parse_args()
    load_dotenv(run_b.ROOT / ".env")
    account = "ryota.yamada@science-aid.com"
    project = run_b.slot_project(args.slot)
    other_project = run_b.slot_project((args.slot + 1) % run_b.SLOTS)
    sa = run_b.agent_sa(project)
    run = run_b.ROOT / "runs" / "agent" / f"smoke-{int(time.time())}"
    (run / "workspace").mkdir(parents=True)
    name = f"b-smoke-{args.slot}"
    started = run_b.now()
    key_id = None
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"{'OK ' if passed else 'NG '} {label}{('：' + detail) if detail else ''}", flush=True)

    def agent(cmd: str, timeout: int = 600) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", "exec", f"{name}-agent", "bash", "-lc", cmd], capture_output=True, text=True,
                              timeout=timeout)

    try:
        key_id = run_b.create_key(run, project, account, sa)
        run_b.start_containers(run, name, project, max_cost=1.0, model="gpt-6-luna", effort="medium")
        for label, cmd, expect in CHECKS:
            out = agent(cmd)
            got = (out.stdout or "").strip()
            passed = got.startswith(expect) if expect.endswith("-") else got == expect
            check(label, passed, f"{got!r}（期待 {expect!r}）" + ("" if passed else (out.stderr or "")[-200:]))

        # 資源を作り、監査ログで作成者を追って消せるか
        bucket = f"{project}-smoke-{args.slot}-{int(time.time())}"
        vm = f"smoke-{args.slot}-{int(time.time())}"
        agent(f"gcloud storage buckets create gs://{bucket} --location={run_b.REGION} --quiet")
        agent(f"gcloud compute instances create {vm} --zone=us-central1-a --machine-type=e2-micro --quiet "
              "--image-family=debian-12 --image-project=debian-cloud --no-service-account --no-scopes")
        for _ in range(10):
            got = run_b.owned(project, account, sa, started)
            if bucket in got["buckets"] and vm in got["instances"]:
                break
            time.sleep(15)
        check("監査ログで作成者を追える", bucket in got["buckets"] and vm in got["instances"],
              json.dumps({k: sorted(v) for k, v in got.items() if v}))
        out = agent(f"gcloud compute instances list --project={other_project} --format='value(name)' 2>&1")
        check("他の slot のプロジェクトの VM は見えない", "permission" in out.stdout.lower(),
              " ".join(out.stdout.split())[:200])
        remaining = run_b.cleanup_owned(project, account, sa, started)
        check("作った資源を消せる", not any(remaining.values()), json.dumps(remaining))

        # Codex: web 検索と、API キーが見えないこと
        out = subprocess.run(["docker", "exec", f"{name}-agent", "codex", "exec", "--json", "--skip-git-repo-check",
                              "--cd", "/workspace",
                              "First, use web search to find the latest released version of the Python package "
                              "'requests'. Then run the shell command `env | grep -c OPENAI || true`. "
                              "Reply with the version and the number."],
                             capture_output=True, text=True, timeout=900)
        events = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
        items = [e.get("item") or {} for e in events if e.get("type") == "item.completed"]
        searched = [i for i in items if "search" in (i.get("type") or "")]
        check("Codex が web 検索を使える", bool(searched), json.dumps(searched[:2], ensure_ascii=False)[:300])
        cmds = [i for i in items if i.get("type") == "command_execution" and "grep -c OPENAI" in i.get("command", "")]
        check("コマンドから API キーが見えない", bool(cmds) and cmds[0].get("aggregated_output", "").strip().startswith("0"),
              (cmds[0].get("aggregated_output", "") if cmds else "コマンドが実行されなかった").strip()[:80])
        usage = run / "logs" / "openai_usage.jsonl"
        print("API の使用量の記録:", usage.read_text().splitlines()[-1] if usage.exists() else "無し")
    finally:
        subprocess.run(["docker", "rm", "-f", f"{name}-agent"], capture_output=True)
        if key_id:
            run_b.gcloud(["iam", "service-accounts", "keys", "delete", key_id, f"--iam-account={sa}"],
                         project, account, check=False)
        shutil.rmtree(run / "creds", ignore_errors=True)
        leftover = run_b.cleanup_owned(project, account, sa, started)
        subprocess.run(["docker", "rm", "-f", f"{name}-proxy"], capture_output=True)
        subprocess.run(["docker", "network", "rm", f"{name}-net"], capture_output=True)
    check("後片付けの後の残りが無い", not any(leftover.values()), json.dumps(leftover))
    check("鍵が残っていない", not run_b.delete_user_keys(project, account, sa))
    flows = run / "logs" / "flows.jsonl"
    if flows.exists():
        blocked = [json.loads(line) for line in flows.read_text().splitlines() if json.loads(line).get("blocked")]
        print("遮断の記録:", sorted({(b["host"], b["blocked"]) for b in blocked}))
    print("記録:", run)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
