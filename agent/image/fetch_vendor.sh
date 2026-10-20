#!/bin/bash
# エージェントのイメージに入れる単体のバイナリをホストで取得する（arm64。colima の Linux 用）
set -eu
cd "$(dirname "$0")"
mkdir -p vendor
for u in \
  "https://github.com/openai/codex/releases/download/rust-v0.162.1/codex-aarch64-unknown-linux-musl.tar.gz" \
  "https://github.com/openai/codex/releases/download/rust-v0.162.1/codex-code-mode-host-aarch64-unknown-linux-musl.tar.gz" \
  "https://releases.hashicorp.com/terraform/1.9.8/terraform_1.9.8_linux_arm64.zip"; do
  f="vendor/$(basename "$u")"
  for i in 1 2 3 4 5 6; do curl -fsSL --retry 5 -C - -o "$f" "$u" && break; echo "retry $i"; done
done
