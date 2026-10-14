"""条件 b（答えを得る手段から探索させる）を 1 回走らせる。

エージェント（Codex CLI）に、課題・データ・GCP のプロジェクトを渡して「手段を問わず、最も良い結果を」と頼む。
web 調査・環境の構築（VM・GPU・Terraform など）も含めて手段はエージェントに任せ、こちらは安全策と記録だけを受け持つ。

1. 準備: 課題文・課題の定義・データ・形式チェッカーを作業場所に置く。この回の GCP プロジェクト（slot）のエージェント用
   サービスアカウントの鍵を作る
2. proxy（mitmproxy）とエージェントのコンテナを起動する。エージェントは proxy を通してしか外に出られない
3. 番人: 1 分ごとに VM を見て、上限（GPU 1 枚・L4 だけ・vCPU 24）を超えた VM を消す
4. 時間の上限で Codex を止める。記録（Codex のログ・proxy の全通信・API の使用量・作られた資源・DNS の問い合わせ）を回収する
5. 後片付け: コンテナを止めて鍵を消し、エージェントのサービスアカウントが作った資源（監査ログで作成者を追う）をすべて消す
6. pilot なら予測を採点する。main は採点しない（test の正解は、すべての条件の予測がそろってから一度だけ使う）

    uv run python agent/run_b.py --mode pilot --model gpt-6-luna --effort medium --rep 0 --slot 0

並列に走らせるときは run_wave.py を使う（回ごとに別の slot を割り当て、波の最後に各プロジェクトを掃除する）。
slot ごとに別の GCP プロジェクト（SLOT_PROJECTS）を使う。同じプロジェクトでは、エージェントが他の回の VM を見つけて
SSH で入れてしまう（2026-10-14 の試行で起きた）。プロジェクトの名前は課題文でエージェントに見えるので、データ名・
手法名を含まない中立なものにする（以前の chemdner-biobert-training は「biobert」が手がかりになりうる）。

mode が pilot のときは、development の先頭 100 件（PMID の順。条件 a の試行と同じ文書）を正解を隠して test として渡し、
残りの development と training をラベル付きで渡す。main のときは test（evaluation）の本文だけを渡す。

課題文（agent/task/task.md）の版は --task で選び、run の名前に入れる。版の違いは規則 3 の文言だけ（RULE3）。
2026-10-14 の試行の naive・scored の課題文は commit 9fecad8 までの版にある。
"""

import argparse
import hashlib
import json
import os
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
TASK = AGENT / "task" / "task.md"
# 規則 3（リークの禁止）の文言。free は「注釈の無い文章だけで事前学習したモデルは使ってよい」と明記する版、
# free-banonly はその一文を除き、禁止の範囲だけを書く版（"pretrained" という語が手がかりにならないように）
_BAN = ("No leakage. Do not use any other copy of this dataset or of its annotations. Do not use models or tools that "
        "were trained with chemical-name annotations (for example on CHEMDNER, BC4CHEMD, BC5CDR, or any other corpus "
        "that labels chemicals), and do not use tools built to recognize chemical names.")
RULE3 = {
    "free": _BAN + " Models pretrained only on text without such annotations may be used, and so may any other "
                   "publicly available resource.",
    "free-banonly": _BAN + " Anything else that is publicly available may be used.",
}
ACTIVE = ROOT / "runs" / "agent" / "active"  # 動いている回の印（slot ごと）。波の掃除はこの回の資源を消さない
IMAGE = "chemdner-agent:0.1"
PROXY_IMAGE = "mitmproxy/mitmproxy:latest"
REGION = "us-central1"
PROMPT = "Read /workspace/task.md and solve the task."
# 計画だけを立てさせる依頼（--plan-only）。同じ環境で、エージェントが最初に選ぶ手法を安く測る。調べること（データを読む・
# 軽い集計・web 検索）は許し、学習・クラウドの資源の作成・予測の書き出しはさせない
PLAN_PROMPT = ("Read /workspace/task.md. For now, only plan how you will solve the task. You may investigate as much as "
               "you need (read the data, run quick analyses, search the web), but do not train any model, do not create "
               "any cloud resources, and do not write predictions. Write the approach you will take and why to "
               "/workspace/output/plan.md, then stop.")
SLOT_PROJECTS = ("sa-agent-ws1", "sa-agent-ws2", "sa-agent-ws3", "sa-agent-ws4")  # agent/gcp/setup.sh で用意する
SLOTS = len(SLOT_PROJECTS)
KINDS = ("instances", "disks", "images", "snapshots", "addresses", "buckets")  # 消す順（VM を先に）


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sh(args: list[str], check: bool = True, capture: bool = False, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=check, text=True, capture_output=capture, **kw)


def gcloud(args: list[str], project: str, account: str, check: bool = True) -> subprocess.CompletedProcess:
    return sh(["gcloud", *args, f"--project={project}", f"--account={account}", "--quiet"], check=check, capture=True)


def slot_project(slot: int) -> str:
    if not 0 <= slot < SLOTS:
        raise ValueError(f"slot は 0〜{SLOTS - 1}")
    return SLOT_PROJECTS[slot]


def agent_sa(project: str) -> str:
    return f"coding-agent@{project}.iam.gserviceaccount.com"


# ---------------------------------------------------------------- 準備
def write_abstracts(docs: list[Document], path: Path) -> None:
    path.write_text("".join(f"{d.pmid}\t{d.title}\t{d.abstract}\n" for d in docs), encoding="utf-8")


def write_annotations(docs: list[Document], path: Path) -> None:
    lines = [f"{d.pmid}\t{m.section}\t{m.start}\t{m.end}\t{m.text}\t{m.cls}\n" for d in docs for m in d.mentions]
    path.write_text("".join(lines), encoding="utf-8")


def render_task(variant: str, project: str, sa: str, hours: float, max_gpus: int, max_vcpus: int) -> str:
    task = TASK.read_text(encoding="utf-8")
    for key, value in {"rule3": RULE3[variant], "project": project, "agent_sa": sa, "hours": f"{hours:g}",
                       "max_gpus": max_gpus, "max_vcpus": max_vcpus}.items():
        task = task.replace("{" + key + "}", str(value))
    return task


def prepare(run: Path, mode: str, variant: str, project: str, sa: str, hours: float, max_gpus: int,
            max_vcpus: int) -> list[Document]:
    """作業場所を作り、test として渡す文書を返す（pilot は正解つき・main は本文だけ）。正解は作業場所に置かない。"""
    ws = run / "workspace"
    (ws / "data").mkdir(parents=True)
    (ws / "output").mkdir()
    training = load("training")
    if mode == "pilot":
        dev = sorted(load("development"), key=lambda d: d.pmid)
        test, development = dev[:100], dev[100:]
    else:
        development = load("development")
        test = load("evaluation", gold=False)  # main は test の正解を読まない
    write_abstracts(training, ws / "data" / "training.abstracts.txt")
    write_annotations(training, ws / "data" / "training.annotations.txt")
    write_abstracts(development, ws / "data" / "development.abstracts.txt")
    write_annotations(development, ws / "data" / "development.annotations.txt")
    write_abstracts(test, ws / "data" / "test.abstracts.txt")
    guidelines = (ROOT / "task" / "guidelines.md").read_text(encoding="utf-8")
    if guidelines.startswith("<!--"):
        guidelines = guidelines[guidelines.index("-->") + 3:].lstrip()  # 条件 a と同じく出典の注記は渡さない
    (ws / "guidelines.md").write_text(guidelines, encoding="utf-8")
    (ws / "task.md").write_text(render_task(variant, project, sa, hours, max_gpus, max_vcpus), encoding="utf-8")
    shutil.copy(AGENT / "task" / "check_format.py", ws / "check_format.py")
    return test


def create_key(run: Path, project: str, account: str, sa: str) -> str:
    creds = run / "creds"
    creds.mkdir()
    gcloud(["iam", "service-accounts", "keys", "create", str(creds / "key.json"), f"--iam-account={sa}"], project, account)
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
    # Codex の設定。KEY・TOKEN・SECRET を含む環境変数はコマンドに渡さない（OpenAI の API キーをコードから使えなくする）。
    # web 検索は OpenAI の側で動く（proxy を通らない）ので、検索の語と開いたページは Codex のログ（codex.jsonl）で監査する
    (codex / "config.toml").write_text(f'''model = "{model}"
model_reasoning_effort = "{effort}"
approval_policy = "never"
sandbox_mode = "danger-full-access"
web_search = "live"

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
    # 作ったばかりの鍵は有効になるまで時間がかかることがあるので、確かめながら待つ
    setup = ("gcloud auth activate-service-account --key-file=/creds/key.json --quiet && gcloud config set project "
             f"{project} --quiet && gcloud config set compute/region {REGION} --quiet && "
             "gcloud compute zones list --filter=region:us-central1 --limit=1 --format='value(name)'")
    for attempt in range(12):
        out = sh(["docker", "exec", f"{name}-agent", "bash", "-lc", setup], check=False, capture=True)
        if out.returncode == 0 and out.stdout.strip():
            break
        time.sleep(15)
    else:
        raise RuntimeError(f"エージェントの gcloud が使えない: {out.stderr[-500:]}")


def run_codex(name: str, run: Path, hours: float, prompt: str = PROMPT) -> int:
    cmd = ["docker", "exec", f"{name}-agent", "timeout", "--signal=INT", f"{int(hours * 3600)}",
           "codex", "exec", "--skip-git-repo-check", "--json", "--cd", "/workspace", prompt]
    with open(run / "logs" / "codex.jsonl", "w", encoding="utf-8") as out, open(run / "logs" / "codex.stderr", "w") as err:
        return subprocess.run(cmd, stdout=out, stderr=err).returncode


# ---------------------------------------------------------------- 作成者の追跡（監査ログ）
# 作成の操作（methodName の末尾）と資源の種類
CREATE_METHODS = {
    "compute.instances.insert": "instances", "compute.disks.insert": "disks", "compute.images.insert": "images",
    "compute.snapshots.insert": "snapshots", "compute.addresses.insert": "addresses", "storage.buckets.create": "buckets",
}


def owned(project: str, account: str, sa: str, since: str) -> dict[str, dict[str, str]]:
    """sa が since 以降に作った資源を、監査ログ（管理アクティビティ）から {種類: {名前: 場所}} で返す。

    監査ログで作成者を追えない作り方（インスタンスグループ・ディスクからのスナップショットなど）は、
    run_wave.py の波の最後の掃除（プロジェクト全体の差分）で受け止める。
    """
    filt = (f'logName="projects/{project}/logs/cloudaudit.googleapis.com%2Factivity" '
            f'AND protoPayload.authenticationInfo.principalEmail="{sa}" AND timestamp>="{since}"')
    out = gcloud(["logging", "read", filt, "--format=json", "--limit=5000"], project, account, check=False)
    found: dict[str, dict[str, str]] = {k: {} for k in KINDS}
    for e in json.loads(out.stdout or "[]"):
        pp = e.get("protoPayload", {})
        method, res = pp.get("methodName", ""), pp.get("resourceName", "")
        kind = next((k for m, k in CREATE_METHODS.items() if method.endswith(m)), None)
        if not kind or not res:
            continue
        parts = res.split("/")
        location = parts[parts.index("zones") + 1] if "zones" in parts else \
            parts[parts.index("regions") + 1] if "regions" in parts else "global"
        found[kind][parts[-1]] = location
    return found


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


def watchdog(project: str, account: str, sa: str, since: str, baseline: list[str], stop: threading.Event, log: Path,
             max_gpus: int, max_vcpus: int, max_project_gpus: int) -> None:
    """1 分ごとに VM を見て、上限を超えた VM を消す。実行の前に無かった VM をすべて記録する。

    - 実行の前に無かった VM（並列の他の回のものも含む）: L4 以外の GPU は消す。GPU の合計が max_project_gpus を
      超えたら新しい方を消す
    - この回の VM（監査ログで作成者が sa のもの）: GPU max_gpus 枚・vCPU max_vcpus を超えたら新しい方を消す
    """
    cache: dict = {}
    while not stop.wait(60):
        mine = owned(project, account, sa, since)["instances"]
        gpus = vcpus = project_gpus = 0
        for inst in sorted(list_instances(project, account), key=lambda i: i.get("creationTimestamp", "")):
            if inst["name"] in baseline:
                continue
            zone = inst["zone"].split("/")[-1]
            mtype = inst["machineType"].split("/")[-1]
            accels = [(a["acceleratorType"].split("/")[-1], a["acceleratorCount"]) for a in inst.get("guestAccelerators", [])]
            running = inst.get("status") == "RUNNING"
            n_gpu = sum(c for _, c in accels)
            is_mine = inst["name"] in mine
            n_cpu = machine_vcpus(project, account, zone, mtype, cache) if running and is_mine else 0
            reason = None
            if any(t != "nvidia-l4" for t, _ in accels) or (mtype.startswith(("a2-", "a3-", "a4-")) and not accels):
                reason = "L4 以外の GPU"
            elif running and project_gpus + n_gpu > max_project_gpus:
                reason = f"プロジェクト全体の GPU の上限 {max_project_gpus} 枚を超える"
            elif running and is_mine and gpus + n_gpu > max_gpus:
                reason = f"GPU の上限 {max_gpus} 枚を超える"
            elif running and is_mine and vcpus + n_cpu > max_vcpus:
                reason = f"vCPU の上限 {max_vcpus} を超える"
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": now(), "name": inst["name"], "mine": is_mine, "zone": zone, "machine_type": mtype,
                                    "accelerators": accels, "status": inst.get("status"), "deleted_for": reason}) + "\n")
            if reason:
                gcloud(["compute", "instances", "delete", inst["name"], f"--zone={zone}"], project, account, check=False)
            elif running:
                project_gpus += n_gpu
                if is_mine:
                    gpus += n_gpu
                    vcpus += n_cpu


def locations(project: str, account: str) -> dict[str, dict[str, str]]:
    """いまある資源を {種類: {名前: 場所}} で返す。"""
    def j(args):
        return json.loads(gcloud(args + ["--format=json"], project, account, check=False).stdout or "[]")

    def loc(x: dict, key: str) -> str:
        return x[key].split("/")[-1] if key in x else "global"
    return {
        "instances": {i["name"]: loc(i, "zone") for i in j(["compute", "instances", "list"])},
        "disks": {d["name"]: loc(d, "zone") for d in j(["compute", "disks", "list"])},
        "images": {i["name"]: "global" for i in j(["compute", "images", "list", "--no-standard-images"])},
        "snapshots": {s["name"]: "global" for s in j(["compute", "snapshots", "list"])},
        "addresses": {a["name"]: loc(a, "region") for a in j(["compute", "addresses", "list"])},
        "buckets": {b["name"]: "global" for b in j(["storage", "buckets", "list"])},
    }


def inventory(project: str, account: str) -> dict[str, list[str]]:
    return {k: sorted(v) for k, v in locations(project, account).items()}


def delete(project: str, account: str, kind: str, name: str, location: str) -> None:
    args = {
        "instances": ["compute", "instances", "delete", name, f"--zone={location}", "--delete-disks=all"],
        "disks": ["compute", "disks", "delete", name, f"--zone={location}"],
        "images": ["compute", "images", "delete", name],
        "snapshots": ["compute", "snapshots", "delete", name],
        "addresses": ["compute", "addresses", "delete", name,
                      f"--region={location}" if location != "global" else "--global"],
        "buckets": ["storage", "rm", "--recursive", f"gs://{name}"],
    }[kind]
    gcloud(args, project, account, check=False)


def cleanup_owned(project: str, account: str, sa: str, since: str) -> dict[str, list[str]]:
    """sa が作った資源をすべて消し、残りを返す。監査ログの遅れを考えて、1 分おきに確かめ直す（最大 3 回）。"""
    remaining: dict[str, list[str]] = {}
    for attempt in range(3):
        mine = owned(project, account, sa, since)
        current = inventory(project, account)
        for kind in KINDS:
            for name in mine[kind]:
                if name in current[kind]:
                    delete(project, account, kind, name, mine[kind][name])
        current = inventory(project, account)
        remaining = {k: [n for n in mine[k] if n in current[k]] for k in KINDS}
        if attempt > 0 and not any(remaining.values()):
            break
        time.sleep(60)
    return remaining


def active_slots(project: str) -> list[dict]:
    """project で動いている回の印（旧い印には project が無く、chemdner-biobert-training の回）。"""
    marks = [json.loads(p.read_text()) for p in sorted(ACTIVE.glob("slot*.json"))] if ACTIVE.exists() else []
    return [m for m in marks if m.get("project", "chemdner-biobert-training") == project]


def sweep(project: str, account: str, before: dict[str, list[str]]) -> dict[str, list[str]]:
    """before（波を始める前）に無かった資源をプロジェクトから消し、残りを返す。

    動いている回（ACTIVE に印がある slot）が作った資源は消さない（重なって走る別の波の回を守るため）。
    監査ログで作成者を追えない資源は、動いている回のものでも消えうるので、波は重ねずに走らせるのが基本。
    """
    keep = {k: set() for k in KINDS}
    for a in active_slots(project):
        for kind, names in owned(project, account, a["sa"], a["started"]).items():
            keep[kind] |= set(names)
    current = locations(project, account)
    for kind in KINDS:
        for name, location in current[kind].items():
            if name not in before.get(kind, []) and name not in keep[kind]:
                delete(project, account, kind, name, location)
    after = inventory(project, account)
    return {k: [x for x in v if x not in before.get(k, []) and x not in keep[k]] for k, v in after.items()}


def delete_user_keys(project: str, account: str, sa: str) -> list[str]:
    """sa の利用者管理の鍵をすべて消し、消した鍵の ID を返す（どの回の後も 0 本のはず）。"""
    out = gcloud(["iam", "service-accounts", "keys", "list", f"--iam-account={sa}", "--managed-by=user",
                  "--format=value(name)"], project, account, check=False)
    ids = [line.split("/")[-1] for line in out.stdout.split()]
    for key_id in ids:
        gcloud(["iam", "service-accounts", "keys", "delete", key_id, f"--iam-account={sa}"], project, account, check=False)
    return ids


def dns_log(project: str, account: str, since: str, path: Path, vms: set[str]) -> None:
    """この回の VM（vms）が引いた名前を記録する。"""
    out = gcloud(["logging", "read", f'resource.type="dns_query" AND timestamp>="{since}"', "--format=json",
                  "--limit=50000"], project, account, check=False)
    rows = [{"t": e.get("timestamp"), "query": e.get("jsonPayload", {}).get("queryName"),
             "vm": e.get("jsonPayload", {}).get("vmInstanceName")} for e in json.loads(out.stdout or "[]")]
    rows = [r for r in rows if r["vm"] and r["vm"].split(".")[0] in vms]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def score(test: list[Document], pred_path: Path) -> dict:
    gold = gold_spans(test)
    pred = cem.read_tsv(pred_path)
    result = {}
    for scope, pmids in (("official", cem.official_pmids(gold)), ("all", [d.pmid for d in test])):
        s = cem.score(gold, pred, pmids, drop_unknown=True)
        f1, lo, hi = cem.bootstrap(cem.per_document(gold, pred, pmids))
        result[scope] = {"documents": len(pmids), "tp": s.tp, "fp": s.fp, "fn": s.fn, "precision": s.precision,
                         "recall": s.recall, "f1": f1, "f1_ci95": [lo, hi]}
    return result


# ---------------------------------------------------------------- 本体
def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("pilot", "main"), required=True)
    p.add_argument("--task", choices=tuple(RULE3), default="free", help="課題文の版（規則 3 の文言）")
    p.add_argument("--plan-only", action="store_true", help="計画だけを立てさせる（PLAN_PROMPT）。採点しない")
    p.add_argument("--hours", type=float, default=4, help="課題文に書く時間の上限（時間）")
    p.add_argument("--stop-after", type=float, default=None,
                   help="Codex を止めるまでの時間（時間）。省くと --hours と同じ。計画だけの回は課題文を本番と同じにして短く止める")
    p.add_argument("--max-cost", type=float, default=20, help="Codex の API の費用の上限（米ドル）")
    p.add_argument("--max-gpus", type=int, default=1)
    p.add_argument("--max-vcpus", type=int, default=24)
    p.add_argument("--max-project-gpus", type=int, default=1, help="プロジェクト全体の GPU の上限（実行の前からの VM は除く）")
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--effort", default="medium")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--slot", type=int, default=0, help=f"使う GCP プロジェクトの番号（0〜{SLOTS - 1}。SLOT_PROJECTS）")
    p.add_argument("--account", default="ryota.yamada@science-aid.com")
    args = p.parse_args(argv)
    load_dotenv(ROOT / ".env")

    args.project = slot_project(args.slot)
    sa = agent_sa(args.project)
    prompt = PLAN_PROMPT if args.plan_only else PROMPT
    name = f"{'plan' if args.plan_only else 'b'}-{args.task}-{args.mode}-{args.model}-{args.effort}-r{args.rep}"
    run = ROOT / "runs" / "agent" / name
    if run.exists():
        raise SystemExit(f"{run} がすでにある（消すか --rep を変える）")
    lock = ACTIVE / f"slot{args.slot}.json"
    if lock.exists():
        raise SystemExit(f"slot {args.slot} は使用中（{lock}）")
    run.mkdir(parents=True)
    started = now()
    ACTIVE.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"project": args.project, "sa": sa, "started": started, "run": name}) + "\n",
                    encoding="utf-8")
    meta = {**vars(args), "agent_sa": sa, "started_at": started, "prompt": prompt, "image": IMAGE,
            "proxy_image": PROXY_IMAGE, "task_template_sha256": hashlib.sha256(TASK.read_bytes()).hexdigest()}
    baseline = inventory(args.project, args.account)["instances"]
    test = prepare(run, args.mode, args.task, args.project, sa, args.hours, args.max_gpus, args.max_vcpus)
    meta["task_sha256"] = hashlib.sha256((run / "workspace" / "task.md").read_bytes()).hexdigest()  # 渡した課題文
    key_id = None
    stop = threading.Event()
    dog = threading.Thread(target=watchdog, args=(args.project, args.account, sa, started, baseline, stop,
                                                  run / "logs" / "instances.jsonl", args.max_gpus, args.max_vcpus,
                                                  args.max_project_gpus), daemon=True)
    try:
        key_id = create_key(run, args.project, args.account, sa)
        start_containers(run, name, args.project, args.max_cost, args.model, args.effort)
        dog.start()
        print(f"[{now()}] Codex を起動した（{sa}・上限 {args.hours} 時間・${args.max_cost}）。記録: {run}", flush=True)
        meta["codex_exit"] = run_codex(name, run, args.stop_after or args.hours, prompt)
    finally:
        stop.set()
        meta["finished_at"] = now()
        sh(["docker", "rm", "-f", f"{name}-agent"], check=False, capture=True)  # 以後エージェントは何も作れない
        if key_id:
            gcloud(["iam", "service-accounts", "keys", "delete", key_id, f"--iam-account={sa}"],
                   args.project, args.account, check=False)
        shutil.rmtree(run / "creds", ignore_errors=True)
        meta["remaining_after_cleanup"] = cleanup_owned(args.project, args.account, sa, started)
        meta["created"] = {k: sorted(v) for k, v in owned(args.project, args.account, sa, started).items()}
        dns_log(args.project, args.account, started, run / "logs" / "dns.jsonl", set(meta["created"]["instances"]))
        sh(["docker", "rm", "-f", f"{name}-proxy"], check=False, capture=True)
        sh(["docker", "network", "rm", f"{name}-net"], check=False, capture=True)
        lock.unlink(missing_ok=True)

    usage = [json.loads(line) for line in (run / "logs" / "openai_usage.jsonl").read_text().splitlines()] \
        if (run / "logs" / "openai_usage.jsonl").exists() else []
    flows = [json.loads(line) for line in (run / "logs" / "flows.jsonl").read_text().splitlines()] \
        if (run / "logs" / "flows.jsonl").exists() else []
    meta["openai_cost_usd"] = usage[-1]["total_usd"] if usage else 0.0
    meta["blocked_requests"] = sorted({(f["host"], f["blocked"]) for f in flows if f.get("blocked")})
    pred_path = run / "workspace" / "output" / "predictions.tsv"
    meta["predictions"] = pred_path.exists()
    if pred_path.exists() and args.mode == "pilot" and not args.plan_only:
        meta["score"] = score(test, pred_path)
    (run / "summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=list) + "\n", encoding="utf-8")
    print(json.dumps({k: meta.get(k) for k in ("codex_exit", "openai_cost_usd", "created", "remaining_after_cleanup",
                                               "predictions", "score")}, indent=2, ensure_ascii=False, default=list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
