# 🎯 Qwen2.5-Coder-0.5B-Instruct NL2Bash Fine-Tuning with Unsloth

Fine-tune **`unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit`** into a specialized NL2Bash terminal assistant using **Unsloth** on Google Colab (Free T4 GPU, < 2GB VRAM, ~8 minutes runtime), with real-time monitoring, GGUF quantization, and local Ollama deployment.

---

## 📁 Project Structure

```text
qwen2.5-bash-finetune/
├── Modelfile                            # Ollama Modelfile configured with ChatML template
├── README.md                            # Complete setup & workflow guide
├── requirements-colab.txt               # Colab GPU dependencies
├── notebooks/
│   └── qwen2_5_bash_finetune_colab.ipynb# Interactive Google Colab Notebook with TensorBoard
├── src/
│   ├── train.py                         # Standalone training script with WandB/TensorBoard & VRAM stats
│   ├── export_gguf.py                   # Reliable GGUF conversion using official llama.cpp script
│   └── inference.py                     # Test script for bash prompt generation
├── scripts/
│   ├── colab_cli_train.sh               # 🚀 Official Colab CLI pipeline (Provision -> Train -> Download -> Stop)
│   ├── sync_to_colab.sh                 # Syncs local code to Colab via rsync/ssh
│   ├── monitor_training.sh              # Live GPU and training log streamer
│   ├── pull_weights_from_colab.sh       # Downloads .gguf & merged weights to ./models
│   ├── run_remote_train.sh              # SSH automation: sync -> train -> pull weights
│   ├── register_ollama.sh               # Registers downloaded GGUF in local Ollama
│   ├── push_to_github.sh                # Pushes codebase to personal GitHub (github.com-personal)
│   └── upload_to_huggingface.py         # Publishes GGUF weights to Hugging Face Hub
└── models/                              # Destination for weights saved from Colab
```

---

## ⚡ Instant Terminal Assistant: The `ask` Command

Turn natural language into executable Linux commands directly from your terminal:

```bash
$ ask Docker check all
docker system health

$ ask find all listening ports
sudo lsof -i --listening | awk '{print $4}'

$ ask delete all stopped docker containers
docker rm $(docker ps -a --filter "status=stopped" | awk '{print $1}')
```

### Adding the `ask` Alias to your Shell

#### Method 1: Global Executable (Recommended)
Place the `ask` script into `~/.local/bin/` (make sure `~/.local/bin` is in your `$PATH`):
```bash
cat << 'EOF' > ~/.local/bin/ask
#!/usr/bin/env bash
if [ $# -eq 0 ]; then
    echo "Usage: ask <natural language prompt>"
    exit 1
fi
ollama run bash-coder-assistant "$*" 2>/dev/null | tr -d '\r' | sed '/^[[:space:]]*$/d' | tail -n 1
EOF
chmod +x ~/.local/bin/ask
```

#### Method 2: Shell Function (`~/.bashrc` or `~/.bash_aliases`)
Add this function to your `~/.bashrc`:
```bash
ask() {
    if [ $# -eq 0 ]; then
        echo "Usage: ask <natural language prompt>"
        return 1
    fi
    ollama run bash-coder-assistant "$*"
}
```
Reload your configuration:
```bash
source ~/.bashrc
```

---

## 📊 Empirical Benchmark Results

Side-by-side comparison between **`qwen2.5-coder:0.5b` (Base)** and **`bash-coder-assistant` (Fine-Tuned)**:

| Metric | Base Model (`qwen2.5-coder:0.5b`) | Fine-Tuned Model (`bash-coder-assistant`) | Improvement |
| :--- | :--- | :--- | :--- |
| **Average Latency** | **`22.18s`** | **`1.93s`** | **⚡ 91.3% Faster** |
| **Average Output Length** | **`1,684 chars`** | **`59.5 chars`** | **🎯 96.5% More Concise** |
| **Output Style** | Conversational essays, markdown blocks | Single-line executable Bash | **100% Adherence** |
| **Training Loss** | `4.1045` (Initial) | `0.6726` (Converged) | **-83.6% Loss Reduction** |

---

## 🚀 Training Workflows

### Method 1: Official Google Colab CLI (`colab`) (Recommended)

1-Click complete pipeline from your local terminal:
```bash
./scripts/colab_cli_train.sh
```

### Method 2: Automated SSH Tunnel (`colab_ssh`)
```bash
# In Colab: launch_ssh_cloudflared(password="colab12345")
# In Terminal:
./scripts/run_remote_train.sh root@<cloudflare-tunnel-url>
```

### Method 3: Interactive Google Colab Notebook
Upload [`notebooks/qwen2_5_bash_finetune_colab.ipynb`](notebooks/qwen2_5_bash_finetune_colab.ipynb) to Colab and run all cells.

---

## 📤 Publishing & Model Links

- **Hugging Face Model**: [burhan2753y/qwen2.5-coder-0.5b-nl2bash-gguf](https://huggingface.co/burhan2753y/qwen2.5-coder-0.5b-nl2bash-gguf)
- **GitHub Repository**: [Burhan2753Y/qwen2.5-coder-0.5b-nl2bash](https://github.com/Burhan2753Y/qwen2.5-coder-0.5b-nl2bash)

### Publish Weights to Hugging Face Hub
```bash
python3 scripts/upload_to_huggingface.py \
    --repo_id "burhan2753y/qwen2.5-coder-0.5b-nl2bash-gguf" \
    --token "your_hf_write_token"
```
