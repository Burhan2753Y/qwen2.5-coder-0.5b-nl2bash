#!/usr/bin/env bash
# Registers the GGUF model with local Ollama and runs a test query.
# Usage: ./scripts/register_ollama.sh [model_tag]

set -e

MODEL_TAG="${1:-bash-coder-assistant}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGUF_FILE=$(find "$PROJECT_DIR/models" -name "*.gguf" | head -n 1)

if [ -z "$GGUF_FILE" ]; then
    echo "❌ Error: No .gguf file found in $PROJECT_DIR/models/"
    echo "Make sure training/export completed and weights were pulled locally."
    exit 1
fi

echo "Found GGUF model at: $GGUF_FILE"

# Create a temporary Modelfile pointing directly to the found GGUF file
TEMP_MODELFILE="$PROJECT_DIR/Modelfile.local"
cat <<EOF > "$TEMP_MODELFILE"
FROM $GGUF_FILE

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

SYSTEM """You are a Linux and Ubuntu terminal assistant. Return only the exact, executable Bash command matching the user request with no markdown formatting or conversational filler."""

PARAMETER temperature 0.0
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER top_k 1
PARAMETER top_p 1.0
EOF

echo "Registering model '$MODEL_TAG' in Ollama..."
ollama create "$MODEL_TAG" -f "$TEMP_MODELFILE"
rm -f "$TEMP_MODELFILE"

echo "✅ Model '$MODEL_TAG' registered successfully!"
echo "--- Testing query: 'Show real-time streaming disk read and write speeds' ---"
ollama run "$MODEL_TAG" "Show real-time streaming disk read and write speeds"
