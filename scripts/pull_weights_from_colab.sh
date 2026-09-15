#!/usr/bin/env bash
# Pull model weights (.gguf and merged HF models) from Colab to local PC
# Usage: ./scripts/pull_weights_from_colab.sh <ssh_host> [ssh_port]

set -e

SSH_HOST="${1}"
SSH_PORT="${2:-22}"

if [ -z "$SSH_HOST" ]; then
    echo "Usage: $0 <ssh_host> [ssh_port]"
    echo "Example: $0 root@some-cloudflare-tunnel.trycloudflare.com 22"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_MODELS_DIR="$PROJECT_DIR/models"

mkdir -p "$LOCAL_MODELS_DIR"

echo "=== Pulling model weights from Colab: $SSH_HOST (Port: $SSH_PORT) ==="
rsync -avz -P -e "ssh -p $SSH_PORT" \
    "$SSH_HOST:/content/qwen2.5-bash-finetune/models/" "$LOCAL_MODELS_DIR/"

echo "✅ Weights pulled successfully to $LOCAL_MODELS_DIR"
ls -lh "$LOCAL_MODELS_DIR"
