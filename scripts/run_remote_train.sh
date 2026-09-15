#!/usr/bin/env bash
# End-to-end automation: Syncs code to Colab, installs dependencies, runs training with monitoring, and pulls weights back.
# Usage: ./scripts/run_remote_train.sh <ssh_host> [ssh_port] [wandb_api_key]

set -e

SSH_HOST="${1}"
SSH_PORT="${2:-22}"
WANDB_KEY="${3:-}"

if [ -z "$SSH_HOST" ]; then
    echo "Usage: $0 <ssh_host> [ssh_port] [optional_wandb_api_key]"
    echo "Example: $0 root@some-cloudflare-tunnel.trycloudflare.com 22"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================="
echo " Step 1: Syncing local project to Colab"
echo "=========================================="
"$PROJECT_DIR/scripts/sync_to_colab.sh" "$SSH_HOST" "$SSH_PORT"

echo "=========================================="
echo " Step 2: Running fine-tuning on Colab GPU"
echo " Model: Qwen2.5-Coder-0.5B-Instruct"
echo "=========================================="

REPORT_TO="tensorboard"
WANDB_ENV=""
if [ -n "$WANDB_KEY" ]; then
    REPORT_TO="wandb"
    WANDB_ENV="export WANDB_API_KEY=\"$WANDB_KEY\";"
fi

ssh -p "$SSH_PORT" "$SSH_HOST" bash -c "'
    cd /content/qwen2.5-bash-finetune
    pip install -q --no-deps \"xformers<0.0.29\" \"trl<0.9.0\" peft accelerate bitsandbytes
    pip install -q unsloth tensorboard wandb
    $WANDB_ENV
    python src/train.py \
        --model_name \"unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit\" \
        --max_steps 300 \
        --batch_size 4 \
        --grad_accum_steps 4 \
        --logging_steps 10 \
        --report_to \"$REPORT_TO\" \
        2>&1 | tee train.log
'"

echo "=========================================="
echo " Step 3: Pulling weights back to local PC"
echo "=========================================="
"$PROJECT_DIR/scripts/pull_weights_from_colab.sh" "$SSH_HOST" "$SSH_PORT"

echo "=========================================="
echo "🎉 Done! Qwen2.5-Coder model weights are saved locally in ./models/"
echo "=========================================="
