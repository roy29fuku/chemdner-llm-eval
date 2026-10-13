#!/bin/bash
# 条件 b のエージェントが使う GCP の権限を用意する（1 回だけ。人が自分のアカウントで実行する）。
#
# - agent-b: エージェント本人。プロジェクトの中で VM・ディスク・バケットを作って消せるだけ。
#            ネットワーク・ファイアウォール・IAM・組織ポリシー・割り当ては変えられない
# - agent-vm: エージェントが作る VM に付けるサービスアカウント（バケットの読み書きとログの書き込みだけ）。
#            agent-b は既定の Compute Engine のサービスアカウントは使えず、これだけを VM に付けられる
# - DNS のログ: エージェントの VM が引いた名前を事後に確かめるため、default ネットワークの DNS 問い合わせを記録する
#
# 使い方: PROJECT=chemdner-biobert-training ACCOUNT=ryota.yamada@science-aid.com bash agent/gcp/setup.sh
set -eu

PROJECT="${PROJECT:?PROJECT を指定する}"
ACCOUNT="${ACCOUNT:?ACCOUNT を指定する}"
G="gcloud --project=$PROJECT --account=$ACCOUNT --quiet"
AGENT="agent-b@$PROJECT.iam.gserviceaccount.com"
VM="agent-vm@$PROJECT.iam.gserviceaccount.com"

$G services enable compute.googleapis.com iam.googleapis.com storage.googleapis.com dns.googleapis.com iap.googleapis.com

$G iam service-accounts describe "$AGENT" >/dev/null 2>&1 \
  || $G iam service-accounts create agent-b --display-name="chemdner b: coding agent"
$G iam service-accounts describe "$VM" >/dev/null 2>&1 \
  || $G iam service-accounts create agent-vm --display-name="chemdner b: VMs created by the agent"

for role in roles/compute.instanceAdmin.v1 roles/storage.admin roles/serviceusage.serviceUsageConsumer roles/iap.tunnelResourceAccessor; do
  $G projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$AGENT" --role="$role" --condition=None >/dev/null
done
for role in roles/storage.objectAdmin roles/logging.logWriter; do
  $G projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$VM" --role="$role" --condition=None >/dev/null
done
# agent-b が VM に付けてよいのは agent-vm だけ
$G iam service-accounts add-iam-policy-binding "$VM" --member="serviceAccount:$AGENT" --role=roles/iam.serviceAccountUser >/dev/null

$G dns policies describe agent-dns-logging >/dev/null 2>&1 \
  || $G dns policies create agent-dns-logging --networks=default --enable-logging \
       --description="chemdner b: エージェントの VM の DNS 問い合わせを記録する"

echo "ok: $AGENT / $VM / DNS のログ"
