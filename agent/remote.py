"""条件 b を司令塔の VM（GCP の管理用プロジェクト sa-agent-control）で走らせるための、Mac 側の操作。

司令塔の VM では run_b.py・run_wave.py をそのまま動かす（Docker も VM の上）。Mac は送る・起動する・回収するだけにする。
gcloud の操作は VM のサービスアカウント（orchestrator）で行う（GCLOUD_ACCOUNT を空にする）。

    uv run python agent/remote.py prepare ctl-1          # コードとコーパス（test の正解を除く）を送り、環境を作る
    uv run python agent/remote.py sync ctl-1             # コードだけを送り直す
    uv run python agent/remote.py smoke ctl-1 --slot 0   # 基盤の確認
    uv run python agent/remote.py start ctl-1 -- --mode pilot --task free-banonly --tag gcp --first-slot 0 --job gpt-6-luna:medium:0
    uv run python agent/remote.py status ctl-1
    uv run python agent/remote.py fetch ctl-1            # 記録を Mac の runs/ へ回収する（データの写し・秘密は除く）

OpenAI の API キーは、Mac の .env から読んで標準入力で VM の処理に渡す。VM のディスクには書かない。
"""

import argparse
import shlex
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemdner_bench.corpus import TEST_GOLD  # noqa: E402

CONTROL = "sa-agent-control"
ZONE = "us-central1-a"
ACCOUNT = "ryota.yamada@science-aid.com"
REMOTE = "chemdner-llm-eval"  # VM の上の置き場（ホームの下）
SHIP = ["agent", "src", "task", "prompts", "frozen", "pyproject.toml", "uv.lock", "uv.toml", "README.md", "data/official-2013"]
PYTHON = "3.13"  # Mac と同じ版（司令塔の uv が新しい版を選んで依存を手元で組み立てないように）
EXCLUDE_PARTS = {"__pycache__", ".venv", "runs", "vendor"}  # vendor（イメージの部品）は司令塔で直接取る


def gcloud_ssh(vm: str, command: str, stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["gcloud", "compute", "ssh", vm, f"--project={CONTROL}", f"--zone={ZONE}", f"--account={ACCOUNT}",
                           "--quiet", "--command", command], input=stdin, text=True, capture_output=True, check=check)


def gcloud_scp(src: str, dst: str) -> None:
    subprocess.run(["gcloud", "compute", "scp", f"--project={CONTROL}", f"--zone={ZONE}", f"--account={ACCOUNT}", "--quiet",
                    src, dst], check=True, capture_output=True, text=True)


def prepare(vm: str) -> None:
    gold = [ROOT / "data" / "official-2013" / g for g in TEST_GOLD]
    if any(g.exists() for g in gold):
        raise SystemExit("test の正解が data/ にある。司令塔に送らないよう、先に取り除く")
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle.tgz"
        with tarfile.open(bundle, "w:gz") as tar:
            for rel in SHIP:
                tar.add(ROOT / rel, arcname=rel, filter=lambda ti: None if EXCLUDE_PARTS & set(Path(ti.name).parts) else ti)
        gcloud_scp(str(bundle), f"{vm}:bundle.tgz")
    script = f"""set -e
mkdir -p {REMOTE} waves && tar xzf bundle.tgz -C {REMOTE} && rm bundle.tgz
sudo usermod -aG docker "$USER"
command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
cd {REMOTE} && ~/.local/bin/uv sync --quiet --python {PYTHON}
bash agent/image/fetch_vendor.sh >/dev/null
sudo docker build -q -t chemdner-agent:0.1 agent/image >/dev/null
sudo docker pull -q mitmproxy/mitmproxy:latest >/dev/null
echo prepared
"""
    out = gcloud_ssh(vm, script)
    print(out.stdout.strip() or out.stderr[-500:])


def sync(vm: str) -> None:
    """コードだけを送り直す（コーパスとイメージは prepare で用意済みのものを使う）。"""
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "code.tgz"
        with tarfile.open(bundle, "w:gz") as tar:
            for rel in ("agent", "src", "task", "prompts", "frozen"):
                tar.add(ROOT / rel, arcname=rel, filter=lambda ti: None if EXCLUDE_PARTS & set(Path(ti.name).parts) else ti)
        gcloud_scp(str(bundle), f"{vm}:code.tgz")
    print(gcloud_ssh(vm, f"tar xzf code.tgz -C {REMOTE} && rm code.tgz && echo synced").stdout.strip())


def _key() -> str:
    from dotenv import dotenv_values
    key = dotenv_values(ROOT / ".env").get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Mac の .env に OPENAI_API_KEY が無い")
    return key


def _remote_env() -> str:
    return f'read -r K; export OPENAI_API_KEY="$K"; export GCLOUD_ACCOUNT=""; export UV_PYTHON={PYTHON}; cd {REMOTE}'


def start(vm: str, wave_args: list[str]) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cmd = (f"{_remote_env()} && setsid nohup ~/.local/bin/uv run python agent/run_wave.py {shlex.join(wave_args)} "
           f"> ~/waves/{stamp}.log 2>&1 < /dev/null & echo started ~/waves/{stamp}.log")
    print(gcloud_ssh(vm, cmd, stdin=_key() + "\n").stdout.strip())


def smoke(vm: str, slot: int) -> None:
    cmd = f"{_remote_env()} && ~/.local/bin/uv run python agent/smoke_test.py --slot {slot} 2>&1 | grep -E '^(OK|NG)'"
    print(gcloud_ssh(vm, cmd, stdin=_key() + "\n", check=False).stdout)


def status(vm: str) -> None:
    cmd = (f"cd {REMOTE}/runs/agent 2>/dev/null && for d in b-* plan-*; do [ -d $d ] || continue; "
           "if [ -f $d/summary.json ]; then echo \"$d done\"; else echo \"$d running\"; fi; done; "
           "tail -n 2 ~/waves/*.log 2>/dev/null | tail -n 12")
    print(gcloud_ssh(vm, cmd, check=False).stdout)


def fetch(vm: str) -> None:
    excludes = " ".join(f"--exclude={e}" for e in ("*/workspace/data", "*/mitmproxy", "*/creds", "*/certs", "agent/active"))
    out = gcloud_ssh(vm, f"tar czf ~/fetch.tgz -C {REMOTE}/runs {excludes} agent && echo ok", check=False)
    if "ok" not in out.stdout:
        raise SystemExit(f"VM で記録をまとめられない: {out.stderr[-300:]}")
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "fetch.tgz"
        gcloud_scp(f"{vm}:fetch.tgz", str(local))
        with tarfile.open(local) as tar:
            tar.extractall(ROOT / "runs", filter="data")
    print(f"回収した → {ROOT / 'runs' / 'agent'}")


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("prepare", "sync", "status", "fetch"):
        sub.add_parser(c).add_argument("vm")
    s = sub.add_parser("smoke")
    s.add_argument("vm")
    s.add_argument("--slot", type=int, default=0)
    st = sub.add_parser("start")
    st.add_argument("vm")
    st.add_argument("wave_args", nargs=argparse.REMAINDER, help="-- の後に run_wave.py の引数")
    a = p.parse_args()
    if a.cmd == "prepare":
        prepare(a.vm)
    elif a.cmd == "sync":
        sync(a.vm)
    elif a.cmd == "smoke":
        smoke(a.vm, a.slot)
    elif a.cmd == "start":
        start(a.vm, [x for x in a.wave_args if x != "--"])
    elif a.cmd == "status":
        status(a.vm)
    else:
        fetch(a.vm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
