# 条件 b のエージェントのコンテナ。Codex CLI・gcloud・Terraform・Python を入れる。
# 外への通信は proxy（mitmproxy）を通るしかないネットワークにつなぐ（agent/run_b.py）。
FROM python:3.12-slim-bookworm

# vendor/ の codex の版（記録のため）
ARG CODEX_VERSION=0.162.1
ARG TERRAFORM_VERSION=1.9.8

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg git unzip jq openssh-client netcat-openbsd procps less \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL --retry 5 --http1.1 https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Terraform と Codex CLI の単体のバイナリは、ホストで vendor/ に取得してから取り込む（agent/image/fetch_vendor.sh）。
# コンテナの中からの大きなダウンロードが途中で切れることがあったため
COPY vendor/terraform_${TERRAFORM_VERSION}_linux_arm64.zip vendor/codex-aarch64-unknown-linux-musl.tar.gz \
     vendor/codex-code-mode-host-aarch64-unknown-linux-musl.tar.gz /tmp/
RUN unzip /tmp/terraform_${TERRAFORM_VERSION}_linux_arm64.zip -d /usr/local/bin \
    && tar -xzf /tmp/codex-aarch64-unknown-linux-musl.tar.gz -C /tmp \
    && tar -xzf /tmp/codex-code-mode-host-aarch64-unknown-linux-musl.tar.gz -C /tmp \
    && install -m 0755 /tmp/codex-aarch64-unknown-linux-musl /usr/local/bin/codex \
    && install -m 0755 /tmp/codex-code-mode-host-aarch64-unknown-linux-musl /usr/local/bin/codex-code-mode-host \
    && rm -f /tmp/*.zip /tmp/*.tar.gz /tmp/codex-*

RUN useradd -m -s /bin/bash agent
USER agent
WORKDIR /workspace
