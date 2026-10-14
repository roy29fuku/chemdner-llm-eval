#!/bin/bash
# 条件 b の司令塔（エージェントのコンテナと proxy を動かす VM）を用意する（人が自分のアカウントで実行する）。
#
# 司令塔は、エージェントのプロジェクト（sa-agent-ws1〜ws4）とは別の管理用プロジェクト（sa-agent-control）に置く。
# エージェントの権限（coding-agent）では管理用プロジェクトは見えず、触れない。
#
# - orchestrator: 司令塔の VM に付けるサービスアカウント。各回のプロジェクトで、エージェントの鍵の作成と削除・
#   VM の監視と後片付け・監査ログと DNS のログの読み取りをする（run_b.py が使う）
# - 司令塔の VM（ctl-1〜ctl-4）: Arm（t2a-standard-8・8 vCPU・32 GB・Debian 12）。GPU は付けない（エージェントの
#   「手元」で、GPU は自分のプロジェクトに立てて使う）。Docker・uv を入れる
#
# 使い方: ACCOUNT=ryota.yamada@science-aid.com N=1 bash agent/gcp/setup_control.sh
set -eu

ACCOUNT="${ACCOUNT:?ACCOUNT を指定する}"
N="${N:-1}"  # 作る司令塔の数（ctl-1〜ctl-N）
CONTROL=sa-agent-control
ZONE=us-central1-a
SLOTS="sa-agent-ws1 sa-agent-ws2 sa-agent-ws3 sa-agent-ws4"
G="gcloud --account=$ACCOUNT --quiet"
ORCH="orchestrator@$CONTROL.iam.gserviceaccount.com"

retry() {
  for i in 1 2 3 4 5 6; do
    "$@" >/dev/null 2>&1 && return 0
    sleep 10
  done
  "$@" >/dev/null
}

$G iam service-accounts describe "$ORCH" --project=$CONTROL >/dev/null 2>&1 \
  || $G iam service-accounts create orchestrator --project=$CONTROL --display-name="b: 司令塔（各回のプロジェクトの鍵・監視・後片付け）"
for p in $SLOTS; do
  for role in roles/editor roles/iam.serviceAccountKeyAdmin roles/logging.viewer; do
    retry $G projects add-iam-policy-binding "$p" --member="serviceAccount:$ORCH" --role="$role" --condition=None
  done
done

STARTUP='#!/bin/bash
if ! command -v docker >/dev/null; then
  apt-get update -y && apt-get install -y docker.io rsync git
  systemctl enable --now docker
fi
'
for i in $(seq 1 "$N"); do
  name="ctl-$i"
  $G compute instances describe "$name" --project=$CONTROL --zone=$ZONE >/dev/null 2>&1 && continue
  $G compute instances create "$name" --project=$CONTROL --zone=$ZONE --machine-type=t2a-standard-8 \
    --image-family=debian-12-arm64 --image-project=debian-cloud --boot-disk-size=100GB \
    --service-account="$ORCH" --scopes=cloud-platform --metadata=startup-script="$STARTUP"
done
echo "ok: $ORCH / ctl-1〜ctl-${N}（${ZONE}）"
