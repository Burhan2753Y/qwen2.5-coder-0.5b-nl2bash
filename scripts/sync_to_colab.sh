#!/usr/bin/env bash
# Sync local code to Colab SSH instance
# Usage: ./scripts/sync_to_colab.sh <ssh_host> [ssh_port]

set -e

SSH_HOST="${1}"
SSH_PORT="${2:-22}"

if [ -z "$SSH_HOST" ]; then
    echo "Usage: $0 <ssh_host> [ssh_port]"
    echo "Example: $0 root@some-cloudflare-tunnel.trycloudflare.com 22"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Syncing project to Colab: $SSH_HOST (Port: $SSH_PORT) ==="
rsync -avz -P -e "ssh -p $SSH_PORT" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'models/' \
    --exclude 'outputs/' \
    "$PROJECT_DIR/" "$SSH_HOST:/content/qwen2.5-bash-finetune/"

echo "✅ Code sync complete!"
