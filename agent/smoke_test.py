"""条件 b の基盤の確認（Codex に課題は解かせない）。

proxy とエージェントのコンテナを起動し、次を確かめてから片付ける:
- 許可した宛先（PyPI・Hugging Face のモデル・Google の API）に届く
- 遮断する宛先（Hugging Face のデータセット・CHEMDNER の名前を含む URL・許可リストに無い宛先）は 403 になる
- proxy を通らない直接の通信はできない
- gcloud がサービスアカウントで使える
- Codex が gpt-6-luna で応答し、Codex が実行するコマンドからは OpenAI の API キーが見えない

    uv run python agent/smoke_test.py
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b  # noqa: E402

CHECKS = [
    ("pypi に届く", "curl -s -o /dev/null -w '%{http_code}' https://pypi.org/simple/numpy/", "200"),
    ("HF のモデルに届く", "curl -sL -o /dev/null -w '%{http_code}' https://huggingface.co/api/models/google-bert/bert-base-uncased", "200"),
    ("HF のデータセットは遮断", "curl -s -o /dev/null -w '%{http_code}' https://huggingface.co/datasets/bigbio/chemdner", "403"),
    ("CHEMDNER の名前を含む URL は遮断", "curl -s -o /dev/null -w '%{http_code}' https://github.com/lingluodlut/Att-ChemdNER", "403"),
    ("許可リストに無い宛先は遮断", "curl -s -o /dev/null -w '%{http_code}' https://www.example.com/", "403"),
    ("直接の通信はできない", "curl -s -o /dev/null -w '%{http_code}' --noproxy '*' --max-time 10 https://pypi.org/ || true", "000"),
    ("gcloud が使える", "gcloud compute zones list --filter=region:us-central1 --format='value(name)' | head -1", "us-central1-"),
]


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(run_b.ROOT / ".env")
    project, account = "chemdner-biobert-training", "ryota.yamada@science-aid.com"
    run = run_b.ROOT / "runs" / "agent" / "smoke"
    shutil.rmtree(run, ignore_errors=True)
    (run / "workspace").mkdir(parents=True)
    name = "b-smoke"
    key_id = None
    ok = True
    try:
        key_id = run_b.create_key(run, project, account)
        run_b.start_containers(run, name, project, max_cost=1.0, model="gpt-6-luna", effort="medium")
        for label, cmd, expect in CHECKS:
            out = subprocess.run(["docker", "exec", f"{name}-agent", "bash", "-lc", cmd], capture_output=True, text=True)
            got = (out.stdout or "").strip()
            passed = got.startswith(expect) if expect.endswith("-") else got == expect
            ok &= passed
            print(f"{'OK ' if passed else 'NG '} {label}: {got!r}（期待 {expect!r}）{(out.stderr or '')[-200:] if not passed else ''}")
        out = subprocess.run(["docker", "exec", f"{name}-agent", "codex", "exec", "--skip-git-repo-check", "--cd", "/workspace",
                              "Run the shell command `env | grep -c OPENAI || true` and report only its output number."],
                             capture_output=True, text=True, timeout=600)
        print("Codex の応答:", (out.stdout or "")[-400:].strip(), (out.stderr or "")[-400:].strip())
        usage = run / "logs" / "openai_usage.jsonl"
        print("API の使用量の記録:", usage.read_text().splitlines()[-1] if usage.exists() else "無し")
    finally:
        if key_id:
            run_b.gcloud(["iam", "service-accounts", "keys", "delete", key_id,
                          f"--iam-account=agent-b@{project}.iam.gserviceaccount.com"], project, account, check=False)
        shutil.rmtree(run / "creds", ignore_errors=True)
        for c in (f"{name}-agent", f"{name}-proxy"):
            subprocess.run(["docker", "rm", "-f", c], capture_output=True)
        subprocess.run(["docker", "network", "rm", f"{name}-net"], capture_output=True)
    flows = run / "logs" / "flows.jsonl"
    if flows.exists():
        blocked = [json.loads(l) for l in flows.read_text().splitlines() if json.loads(l).get("blocked")]
        print("遮断の記録:", [(b["host"], b["blocked"]) for b in blocked])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
