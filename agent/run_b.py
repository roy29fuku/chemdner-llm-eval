"""条件 b（答えを得る手段から探索させる）を 1 回走らせる。

エージェント（Codex CLI）に、課題・データ・GCP のプロジェクトを渡して「コーディングとクラウドを自由に使って解いて」と
頼む。環境の構築（VM・GPU・Terraform など）も含めて手段はエージェントに任せ、こちらは安全策だけを受け持つ。

1. 準備: 課題文・課題の定義・データ・形式チェッカーを作業場所に置く。エージェント用のサービスアカウントの鍵を作る
2. proxy（mitmproxy）とエージェントのコンテナを起動する。エージェントは proxy を通してしか外に出られない
3. 番人: 1 分ごとにプロジェクトの VM を見て、上限（GPU 全体 1 枚・L4 だけ・vCPU 24）を超えた VM を消す
4. 時間の上限で Codex を止める。記録（Codex のログ・proxy の全通信・API の使用量・作られた資源・DNS の問い合わせ）を回収する
5. 後片付け: エージェントが作った資源をすべて消し、残っていないことを確かめる。鍵を消す
6. 予測を採点する

    uv run python agent/run_b.py --mode pilot --hours 4 --max-cost 10

mode が pilot のときは、development の先頭 100 件（PMID の順。条件 a の試行と同じ文書）を正解を隠して test として渡し、
残りの development と training をラベル付きで渡す。main のときは test（evaluation）を渡す。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemdner_bench import cem  # noqa: E402
from chemdner_bench.corpus import Document, gold_spans, load  # noqa: E402

AGENT = ROOT / "agent"
IMAGE = "chemdner-agent:0.1"
PROXY_IMAGE = "mitmproxy/mitmproxy:latest"
REGION = "us-central1"
PROMPT = "Read /workspace/task.md and solve the task."


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sh(args: list[str], check: bool = True, capture: bool = False, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=check, text=True, capture_output=capture, **kw)


def gcloud(args: list[str], project: str, account: str, check: bool = True) -> subprocess.CompletedProcess:
    return sh(["gcloud", *args, f"--project={project}", f"--account={account}", "--quiet"], check=check, capture=True)


# ---------------------------------------------------------------- 準備
def write_abstracts(docs: list[Document], path: Path) -> None:
    path.write_text("".join(f"{d.pmid}\t{d.title}\t{d.abstract}\n" for d in docs), encoding="utf-8")


def write_annotations(docs: list[Document], path: Path) -> None:
    lines = [f"{d.pmid}\t{m.section}\t{m.start}\t{m.end}\t{m.text}\t{m.cls}\n" for d in docs for m in d.mentions]
    path.write_text("".join(lines), encoding="utf-8")


def prepare(run: Path, mode: str, project: str, hours: float, max_gpus: int, max_vcpus: int) -> list[Document]:
    """作業場所を作り、test として渡す文書（正解つき・採点用）を返す。正解は作業場所に置かない。"""
    ws = run / "workspace"
    (ws / "data").mkdir(parents=True)
    (ws / "output").mkdir()
    training = load("training")
    if mode == "pilot":
        dev = sorted(load("development"), key=lambda d: d.pmid)
        test, development = dev[:100], dev[100:]
    else:
        development = load("development")
        test = load("evaluation")  # 正解は採点にだけ使う（test の正解が data/ に無ければここで止まる）
    write_abstracts(training, ws / "data" / "training.abstracts.txt")
    write_annotations(training, ws / "data" / "training.annotations.txt")
    write_abstracts(development, ws / "data" / "development.abstracts.txt")
    write_annotations(development, ws / "data" / "development.annotations.txt")
    write_abstracts(test, ws / "data" / "test.abstracts.txt")
    guidelines = (ROOT / "task" / "guidelines.md").read_text(encoding="utf-8")
    if guidelines.startswith("<!--"):
        guidelines = guidelines[guidelines.index("-->") + 3:].lstrip()  # 条件 a と同じく出典の注記は渡さない
    (ws / "guidelines.md").write_text(guidelines, encoding="utf-8")
    task = (AGENT / "task" / "task.md").read_text(encoding="utf-8")
    for key, value in {"project": project, "hours": f"{hours:g}", "max_gpus": max_gpus, "max_vcpus": max_vcpus}.items():
        task = task.replace("{" + key + "}", str(value))
    (ws / "task.md").write_text(task, encoding="utf-8")
    shutil.copy(AGENT / "task" / "check_format.py", ws / "check_format.py")
    return test


def create_key(run: Path, project: str, account: str) -> str:
    creds = run / "creds"
    creds.mkdir()
    gcloud(["iam", "service-accounts", "keys", "create", str(creds / "key.json"),
            f"--iam-account=agent-b@{project}.iam.gserviceaccount.com"], project, account)
    return json.loads((creds / "key.json").read_text())["private_key_id"]  # 後で鍵を消すための ID


# ---------------------------------------------------------------- コンテナ
def start_containers(run: Path, name: str, project: str, max_cost: float, model: str, effort: str) -> None:
    logs, conf, certs, codex = run / "logs", run / "mitmproxy", run / "certs", run / "codex-home"
    for d in (logs, conf, certs, codex):
        d.mkdir(exist_ok=True)
    net = f"{name}-net"
    sh(["docker", "network", "create", "--internal", net], capture=True)
    sh(["docker", "run", "-d", "--name", f"{name}-proxy", "--network", net, "--network-alias", "proxy",
        "-v", f"{logs}:/logs", "-v", f"{AGENT / 'proxy'}:/addon:ro", "-v", f"{conf}:/home/mitmproxy/.mitmproxy",
        "-e", f"MAX_COST_USD={max_cost}", "-e", f"PROJECT_ID={project}", PROXY_IMAGE,
        "mitmdump", "-s", "/addon/addon.py", "--listen-port", "8080", "--set", r"ignore_hosts=^.*:22$",
        "--set", "block_global=false"], capture=True)
    sh(["docker", "network", "connect", "bridge", f"{name}-proxy"], capture=True)
    ca = conf / "mitmproxy-ca-cert.pem"
    for _ in range(60):
        if ca.exists():
            break
        time.sleep(1)
    else:
        raise RuntimeError("mitmproxy の CA 証明書ができない")
    system = sh(["docker", "run", "--rm", IMAGE, "cat", "/etc/ssl/certs/ca-certificates.crt"], capture=True).stdout
    (certs / "ca-bundle.crt").write_text(system + "\n" + ca.read_text(), encoding="utf-8")
    shutil.copy(ca, certs / "mitmproxy-ca.pem")
    # Codex の設定。KEY・TOKEN・SECRET を含む環境変数はコマンドに渡さない（OpenAI の API キーをコードから使えなくする）
    (codex / "config.toml").write_text(f'''model = "{model}"
model_reasoning_effort = "{effort}"
approval_policy = "never"
sandbox_mode = "danger-full-access"

[shell_environment_policy]
inherit = "all"
ignore_default_excludes = false
exclude = ["OPENAI_*", "CODEX_API_KEY"]
''', encoding="utf-8")
    env = {
        "HTTP_PROXY": "http://proxy:8080", "HTTPS_PROXY": "http://proxy:8080",
        "http_proxy": "http://proxy:8080", "https_proxy": "http://proxy:8080", "NO_PROXY": "localhost,127.0.0.1",
        "SSL_CERT_FILE": "/certs/ca-bundle.crt", "REQUESTS_CA_BUNDLE": "/certs/ca-bundle.crt",
        "CURL_CA_BUNDLE": "/certs/ca-bundle.crt", "PIP_CERT": "/certs/ca-bundle.crt",
        "NODE_EXTRA_CA_CERTS": "/certs/mitmproxy-ca.pem", "CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE": "/certs/ca-bundle.crt",
        "GOOGLE_APPLICATION_CREDENTIALS": "/creds/key.json", "CLOUDSDK_CORE_PROJECT": project,
        "CLOUDSDK_COMPUTE_REGION": REGION, "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        "CODEX_API_KEY": os.environ["OPENAI_API_KEY"],  # codex exec はこちらを読む版がある
    }
    args = ["docker", "run", "-d", "--name", f"{name}-agent", "--network", net, "-v", f"{run / 'workspace'}:/workspace",
            "-v", f"{certs}:/certs:ro", "-v", f"{run / 'creds'}:/creds:ro", "-v", f"{codex}:/home/agent/.codex"]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    sh(args + [IMAGE, "sleep", "infinity"], capture=True)
    sh(["docker", "exec", f"{name}-agent", "bash", "-lc",
        "gcloud auth activate-service-account --key-file=/creds/key.json --quiet && gcloud config set project "
        f"{project} --quiet && gcloud config set compute/region {REGION} --quiet"], capture=True)


def run_codex(name: str, run: Path, hours: float) -> int:
    cmd = ["docker", "exec", f"{name}-agent", "timeout", "--signal=INT", f"{int(hours * 3600)}",
           "codex", "exec", "--skip-git-repo-check", "--json", "--cd", "/workspace", PROMPT]
    with open(run / "logs" / "codex.jsonl", "w", encoding="utf-8") as out, open(run / "logs" / "codex.stderr", "w") as err:
        return subprocess.run(cmd, stdout=out, stderr=err).returncode


# ---------------------------------------------------------------- 番人と後片付け
def list_instances(project: str, account: str) -> list[dict]:
    out = gcloud(["compute", "instances", "list", "--format=json"], project, account, check=False)
    return json.loads(out.stdout or "[]")


def machine_vcpus(project: str, account: str, zone: str, machine_type: str, cache: dict) -> int:
    key = (zone, machine_type)
    if key not in cache:
        out = gcloud(["compute", "machine-types", "describe", machine_type, f"--zone={zone}", "--format=json"],
                     project, account, check=False)
        cache[key] = json.loads(out.stdout or "{}").get("guestCpus", 0)
    return cache[key]


def watchdog(project: str, account: str, stop: threading.Event, log: Path, max_gpus: int, max_vcpus: int) -> None:
    """1 分ごとに VM を見て、上限を超えた VM を消す。見えた VM はすべて記録する。"""
    cache: dict = {}
    while not stop.wait(60):
        gpus = vcpus = 0
        for inst in sorted(list_instances(project, account), key=lambda i: i.get("creationTimestamp", "")):
            zone = inst["zone"].split("/")[-1]
            mtype = inst["machineType"].split("/")[-1]
            accels = [(a["acceleratorType"].split("/")[-1], a["acceleratorCount"]) for a in inst.get("guestAccelerators", [])]
            n_gpu = sum(c for _, c in accels)
            n_cpu = machine_vcpus(project, account, zone, mtype, cache) if inst.get("status") == "RUNNING" else 0
            reason = None
            if any(t != "nvidia-l4" for t, _ in accels) or (mtype.startswith(("a2-", "a3-", "a4-")) and not accels):
                reason = "L4 以外の GPU"
            elif inst.get("status") == "RUNNING" and gpus + n_gpu > max_gpus:
                reason = f"GPU の上限 {max_gpus} 枚を超える"
            elif inst.get("status") == "RUNNING" and vcpus + n_cpu > max_vcpus:
                reason = f"vCPU の上限 {max_vcpus} を超える"
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": now(), "name": inst["name"], "zone": zone, "machine_type": mtype,
                                    "accelerators": accels, "status": inst.get("status"), "deleted_for": reason}) + "\n")
            if reason:
                gcloud(["compute", "instances", "delete", inst["name"], f"--zone={zone}"], project, account, check=False)
            elif inst.get("status") == "RUNNING":
                gpus += n_gpu
                vcpus += n_cpu


def inventory(project: str, account: str) -> dict:
    def j(args):
        out = gcloud(args + ["--format=json"], project, account, check=False)
        return json.loads(out.stdout or "[]")
    return {
        "instances": [i["name"] for i in j(["compute", "instances", "list"])],
        "disks": [d["name"] for d in j(["compute", "disks", "list"])],
        "images": [i["name"] for i in j(["compute", "images", "list", "--no-standard-images"])],
        "snapshots": [s["name"] for s in j(["compute", "snapshots", "list"])],
        "addresses": [a["name"] for a in j(["compute", "addresses", "list"])],
        "buckets": [b["name"] for b in j(["storage", "buckets", "list"])],
    }


def cleanup(project: str, account: str, before: dict) -> dict:
    """エージェントが作った資源（実行の前に無かったもの）をすべて消し、残りを返す。"""
    now_ = inventory(project, account)
    new = {k: [x for x in v if x not in before.get(k, [])] for k, v in now_.items()}
    for inst in list_instances(project, account):
        if inst["name"] in new["instances"]:
            gcloud(["compute", "instances", "delete", inst["name"], f"--zone={inst['zone'].split('/')[-1]}",
                    "--delete-disks=all"], project, account, check=False)
    disks = json.loads(gcloud(["compute", "disks", "list", "--format=json"], project, account, check=False).stdout or "[]")
    for d in disks:
        if d["name"] in new["disks"]:
            gcloud(["compute", "disks", "delete", d["name"], f"--zone={d['zone'].split('/')[-1]}"], project, account, check=False)
    for name in new["images"]:
        gcloud(["compute", "images", "delete", name], project, account, check=False)
    for name in new["snapshots"]:
        gcloud(["compute", "snapshots", "delete", name], project, account, check=False)
    addrs = json.loads(gcloud(["compute", "addresses", "list", "--format=json"], project, account, check=False).stdout or "[]")
    for a in addrs:
        if a["name"] in new["addresses"]:
            loc = f"--region={a['region'].split('/')[-1]}" if "region" in a else "--global"
            gcloud(["compute", "addresses", "delete", a["name"], loc], project, account, check=False)
    for name in new["buckets"]:
        gcloud(["storage", "rm", "--recursive", f"gs://{name}"], project, account, check=False)
    after = inventory(project, account)
    return {k: [x for x in v if x not in before.get(k, [])] for k, v in after.items()}


def dns_log(project: str, account: str, since: str, path: Path) -> None:
    out = gcloud(["logging", "read", f'resource.type="dns_query" AND timestamp>="{since}"', "--format=json",
                  "--limit=20000"], project, account, check=False)
    rows = [{"t": e.get("timestamp"), "query": e.get("jsonPayload", {}).get("queryName"),
             "vm": e.get("jsonPayload", {}).get("vmInstanceName")} for e in json.loads(out.stdout or "[]")]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ---------------------------------------------------------------- 本体
def main() -> int:
    from dotenv import load_dotenv

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("pilot", "main"), required=True)
    p.add_argument("--hours", type=float, default=4)
    p.add_argument("--max-cost", type=float, default=10, help="Codex の API の費用の上限（米ドル）")
    p.add_argument("--max-gpus", type=int, default=1)
    p.add_argument("--max-vcpus", type=int, default=24)
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--effort", default="medium")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--project", default="chemdner-biobert-training")
    p.add_argument("--account", default="ryota.yamada@science-aid.com")
    args = p.parse_args()
    load_dotenv(ROOT / ".env")

    name = f"b-{args.mode}-{args.model}-{args.effort}-r{args.rep}"
    run = ROOT / "runs" / "agent" / name
    if run.exists():
        raise SystemExit(f"{run} がすでにある（消すか --rep を変える）")
    run.mkdir(parents=True)
    started = now()
    meta = {**vars(args), "started_at": started, "prompt": PROMPT, "image": IMAGE, "proxy_image": PROXY_IMAGE}
    before = inventory(args.project, args.account)
    meta["resources_before"] = before
    test = prepare(run, args.mode, args.project, args.hours, args.max_gpus, args.max_vcpus)
    key_id = None
    stop = threading.Event()
    dog = threading.Thread(target=watchdog, args=(args.project, args.account, stop, run / "logs" / "instances.jsonl",
                                                    args.max_gpus, args.max_vcpus), daemon=True)
    try:
        key_id = create_key(run, args.project, args.account)
        start_containers(run, name, args.project, args.max_cost, args.model, args.effort)
        dog.start()
        print(f"[{now()}] Codex を起動した（上限 {args.hours} 時間・${args.max_cost}）。記録: {run}", flush=True)
        meta["codex_exit"] = run_codex(name, run, args.hours)
    finally:
        stop.set()
        meta["finished_at"] = now()
        meta["resources_at_end"] = inventory(args.project, args.account)
        dns_log(args.project, args.account, started, run / "logs" / "dns.jsonl")
        meta["remaining_after_cleanup"] = cleanup(args.project, args.account, before)
        if key_id:
            gcloud(["iam", "service-accounts", "keys", "delete", key_id,
                    f"--iam-account=agent-b@{args.project}.iam.gserviceaccount.com"], args.project, args.account, check=False)
        shutil.rmtree(run / "creds", ignore_errors=True)
        for c in (f"{name}-agent", f"{name}-proxy"):
            sh(["docker", "rm", "-f", c], check=False, capture=True)
        sh(["docker", "network", "rm", f"{name}-net"], check=False, capture=True)

    usage = [json.loads(line) for line in (run / "logs" / "openai_usage.jsonl").read_text().splitlines()] \
        if (run / "logs" / "openai_usage.jsonl").exists() else []
    flows = [json.loads(line) for line in (run / "logs" / "flows.jsonl").read_text().splitlines()] \
        if (run / "logs" / "flows.jsonl").exists() else []
    meta["openai_cost_usd"] = usage[-1]["total_usd"] if usage else 0.0
    meta["blocked_requests"] = sorted({(f["host"], f["blocked"]) for f in flows if f.get("blocked")})
    pred_path = run / "workspace" / "output" / "predictions.tsv"
    if pred_path.exists():
        gold = gold_spans(test)
        pred = cem.read_tsv(pred_path)
        result = {}
        for scope, pmids, drop in (("official", cem.official_pmids(gold), True), ("all", [d.pmid for d in test], False)):
            s = cem.score(gold, pred, pmids, drop_unknown=True)
            f1, lo, hi = cem.bootstrap(cem.per_document(gold, pred, pmids))
            result[scope] = {"documents": len(pmids), "tp": s.tp, "fp": s.fp, "fn": s.fn, "precision": s.precision,
                             "recall": s.recall, "f1": f1, "f1_ci95": [lo, hi]}
        meta["score"] = result
    (run / "summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=list) + "\n", encoding="utf-8")
    print(json.dumps({k: meta.get(k) for k in ("codex_exit", "openai_cost_usd", "remaining_after_cleanup", "score")},
                     indent=2, ensure_ascii=False, default=list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
