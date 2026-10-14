"""実行記録を、非公開の保管 repo（roy29fuku/chemdner-llm-eval-runs）へ写す。

後から検証できるように、a・b の各回の記録を写し、全ファイルの SHA-256 の一覧（SHA256SUMS）を作る。
記録は PubMed の本文を含むので、保管 repo は非公開にする。公開の repo には SHA256SUMS だけを置ける。

写すもの:
- runs/direct/<回>/: 設定・呼び出しの記録・モデルの生の出力（private/raw.jsonl）・予測・要約
- runs/agent/<回>/: 要約・ログ一式（Codex・proxy の全通信・API の使用量・VM・DNS）・監査の材料・エージェントの
  作業場所（コード・予測・計画）・その回の proxy の設定・Codex の会話の記録（codex-home/sessions）と設定
- runs/agent/waves/: 波ごとの要約とログ

写さないもの:
- proxy の CA（mitmproxy/。秘密鍵を含む）・証明書の束（certs/）・鍵の置き場（creds/）
- Codex の内部の DB やキャッシュ（codex-home の sessions と config.toml 以外）
- 作業場所のデータの写し（workspace/data/。コーパスと分割の規則から作り直せる）
- 50 MB を超えるファイル（SHA256SUMS の後ろに「写していない」として一覧を残す）

写す前に、秘密鍵や API キーらしい文字列が無いかを確かめ、見つかったら止める。

    uv run python agent/archive_runs.py ../chemdner-llm-eval-runs
"""

import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
MAX_BYTES = 50 * 1024 * 1024
# 「sk-」は前が英数字・_・- でないものだけを数える（暗号化された推論の base64url の文字列や、basilisk- のような
# モデル名の中にたまたま現れるものを除く）
SECRET = re.compile(rb"BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|\"private_key\"\s*:|(?<![A-Za-z0-9_-])sk-(proj-)?[A-Za-z0-9_-]{20,}")
SKIP_DIRS = {"mitmproxy", "certs", "creds"}
SKIP_TOP = {"active"}  # 動いている回の印（runs/agent/active/）は記録ではない


def wanted(rel: Path) -> bool:
    parts = rel.parts
    if parts[0] == "agent" and len(parts) > 1 and parts[1] in SKIP_TOP:
        return False
    if parts[0] == "agent" and len(parts) > 2:
        top = parts[2]
        if top in SKIP_DIRS:
            return False
        if top == "workspace" and len(parts) > 3 and parts[3] == "data":
            return False
        if top == "codex-home":
            return len(parts) > 3 and (parts[3] == "sessions" or parts[3] == "config.toml")
        if top.startswith(".watch"):
            return False
    if rel.name.startswith(".watch"):
        return False
    if "__pycache__" in parts:
        return False
    return True


def main(dest: Path) -> int:
    if not (dest / ".git").exists():
        raise SystemExit(f"{dest} は git の作業場所ではない（保管 repo を clone しておく）")
    files, skipped, secrets = [], [], []
    for p in sorted(RUNS.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(RUNS)
        if not wanted(rel):
            continue
        if p.stat().st_size > MAX_BYTES:
            skipped.append(rel)
            continue
        if SECRET.search(p.read_bytes()):
            secrets.append(rel)
        files.append(rel)
    if secrets:
        print("秘密らしい文字列があるので止めた:", *secrets, sep="\n  ")
        return 1
    sums = []
    for rel in files:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RUNS / rel, out)
        sums.append(f"{hashlib.sha256(out.read_bytes()).hexdigest()}  {rel.as_posix()}")
    body = "\n".join(sums) + "\n"
    if skipped:
        body += "".join(f"# 写していない（50 MB 超）: {s.as_posix()}  {hashlib.sha256((RUNS / s).read_bytes()).hexdigest()}\n"
                        for s in skipped)
    (dest / "SHA256SUMS").write_text(body, encoding="utf-8")
    print(f"写した {len(files)} ファイル・写していない {len(skipped)} ファイル → {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
