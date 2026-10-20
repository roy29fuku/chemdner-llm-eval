#!/bin/bash
# 条件 b のエージェントが使う GCP の権限を用意する（プロジェクトごとに 1 回。人が自分のアカウントで実行する）。
#
# 並列に走らせる回は、slot ごとに別のプロジェクト（run_b.SLOT_PROJECTS）を使う。同じプロジェクトでは、
# エージェントが他の回の VM を見つけて入れてしまう（2026-10-14 の試行で起きた）。プロジェクトの名前は、
# 課題文でエージェントに見えるので、データ名・手法名を含まない中立なものにする。
#
# - coding-agent: エージェント本人。プロジェクトの中で VM・ディスク・バケットを作って消せるだけ。
#            ネットワーク・ファイアウォール・IAM・組織ポリシー・割り当ては変えられない
# - agent-vm: エージェントが作る VM に付けるサービスアカウント（バケットの読み書きとログの書き込みだけ）。
#            エージェントは既定の Compute Engine のサービスアカウントは使えず、これだけを VM に付けられる
# - DNS のログ: エージェントの VM が引いた名前を事後に確かめるため、default ネットワークの DNS 問い合わせを記録する
#
# 使い方: PROJECT=sa-agent-ws1 ACCOUNT=ryota.yamada@science-aid.com bash agent/gcp/setup.sh
set -eu

PROJECT="${PROJECT:?PROJECT を指定する}"
ACCOUNT="${ACCOUNT:?ACCOUNT を指定する}"
G="gcloud --project=$PROJECT --account=$ACCOUNT --quiet"
VM="agent-vm@$PROJECT.iam.gserviceaccount.com"
AGENTS="${AGENTS:-coding-agent}"


# 作ったばかりのサービスアカウントは、権限を付けられるまで時間がかかることがあるので、再試行する
retry() {
  for i in 1 2 3 4 5 6; do
    "$@" >/dev/null 2>&1 && return 0
    sleep 10
  done
  "$@" >/dev/null
}

$G services enable compute.googleapis.com iam.googleapis.com storage.googleapis.com dns.googleapis.com iap.googleapis.com

$G iam service-accounts describe "$VM" >/dev/null 2>&1 \
  || $G iam service-accounts create agent-vm --display-name="chemdner b: VMs created by the agent"
for role in roles/storage.objectAdmin roles/logging.logWriter; do
  retry $G projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$VM" --role="$role" --condition=None
done

for name in $AGENTS; do
  AGENT="$name@$PROJECT.iam.gserviceaccount.com"
  $G iam service-accounts describe "$AGENT" >/dev/null 2>&1 \
    || $G iam service-accounts create "$name" --display-name="chemdner b: coding agent ($name)"
  for role in roles/compute.instanceAdmin.v1 roles/storage.admin roles/serviceusage.serviceUsageConsumer roles/iap.tunnelResourceAccessor; do
    retry $G projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$AGENT" --role="$role" --condition=None
  done
  # エージェントが VM に付けてよいのは agent-vm だけ
  retry $G iam service-accounts add-iam-policy-binding "$VM" --member="serviceAccount:$AGENT" --role=roles/iam.serviceAccountUser
done

$G dns policies describe agent-dns-logging >/dev/null 2>&1 \
  || $G dns policies create agent-dns-logging --networks=default --enable-logging \
       --description="chemdner b: エージェントの VM の DNS 問い合わせを記録する"

echo "ok: $AGENTS / $VM / DNS のログ"
