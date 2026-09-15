#!/usr/bin/env bash
# Real-time monitoring script for remote Colab training
# Shows streaming training loss and GPU stats (nvidia-smi)
# Usage: ./scripts/monitor_training.sh <ssh_host> [ssh_port]

set -e

SSH_HOST="${1}"
SSH_PORT="${2:-22}"

if [ -z "$SSH_HOST" ]; then
    echo "Usage: $0 <ssh_host> [ssh_port]"
    echo "Example: $0 root@some-cloudflare-tunnel.trycloudflare.com 22"
    exit 1
fi

echo "=========================================================="
echo "📊 Monitoring Remote Colab: $SSH_HOST (Port: $SSH_PORT)"
echo "=========================================================="
echo "1. Checking GPU specifications & current VRAM usage..."
ssh -p "$SSH_PORT" "$SSH_HOST" "nvidia-smi"

echo ""
echo "2. Streaming real-time training progress from Colab logs..."
echo "(Press Ctrl+C to stop monitoring)"
ssh -p "$SSH_PORT" "$SSH_HOST" "tail -n 30 -f /content/qwen2.5-bash-finetune/outputs/logs/*.json 2>/dev/null || tail -n 30 -f /content/qwen2.5-bash-finetune/train.log 2>/dev/null || nvidia-smi -l 2"
