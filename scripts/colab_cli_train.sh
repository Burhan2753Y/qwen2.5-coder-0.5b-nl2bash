#!/usr/bin/env bash
# Official Google Colab CLI pipeline for Qwen2.5-Coder NL2Bash fine-tuning
# Provisions a T4 GPU VM, installs dependencies via uv on VM, executes training, downloads weights locally, and cleans up.

set -e

SESSION_NAME="${1:-qwen-trainer}"
GPU_TYPE="${2:-T4}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check if session already exists
EXISTING_SESSION=$(colab sessions 2>/dev/null | grep "$SESSION_NAME" || true)

if [ -z "$EXISTING_SESSION" ]; then
    echo "=========================================================="
    echo "🚀 Step 1: Provisioning Colab VM with $GPU_TYPE GPU ($SESSION_NAME)"
    echo "=========================================================="
    colab new -s "$SESSION_NAME" --gpu "$GPU_TYPE"
else
    echo "=========================================================="
    echo "ℹ️ Using active Colab session: $SESSION_NAME"
    echo "=========================================================="
fi

echo "=========================================================="
echo "📦 Step 2: Ensuring ML dependencies on Colab VM (via fast uv)"
echo "=========================================================="
colab install -s "$SESSION_NAME" unsloth "trl<0.9.0" peft accelerate bitsandbytes datasets transformers tensorboard

echo "=========================================================="
echo "🔥 Step 3: Executing training script on remote Colab GPU"
echo "=========================================================="
colab exec -s "$SESSION_NAME" --timeout 3600 -f "$PROJECT_DIR/src/train.py"

echo "=========================================================="
echo "📥 Step 4: Downloading trained model weights to local PC"
echo "=========================================================="
mkdir -p "$PROJECT_DIR/models/qwen2.5-coder-0.5b-bash-gguf"
colab download -s "$SESSION_NAME" \
    content/models/qwen2.5-coder-0.5b-bash-gguf/qwen2.5-coder-0.5b-bash-gguf.gguf \
    "$PROJECT_DIR/models/qwen2.5-coder-0.5b-bash-gguf/qwen2.5-coder-0.5b-bash-gguf.gguf" || true

echo "=========================================================="
echo "🛑 Step 5: Terminating Colab VM session"
echo "=========================================================="
colab stop -s "$SESSION_NAME"

echo "=========================================================="
echo "🦙 Step 6: Registering model in local Ollama"
echo "=========================================================="
"$PROJECT_DIR/scripts/register_ollama.sh" bash-coder-assistant

echo "🎉 Pipeline finished! Model is ready to use locally."
