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

## 📊 Dual-Track Benchmark Results

We evaluated the base model (`qwen2.5-coder:0.5b`) against our fine-tuned model (`bash-coder-assistant`) across both **Academic Standards** and **Real-World DevOps/SysAdmin Workloads**:

### 🎓 Track 1: Academic NL2Bash Benchmark (EMNLP 2018 / Microsoft CodeXGLUE Standard)
Evaluates exact match, token F1, BLEU-4, and abstract syntax tree (AST) parsing on standard NL2Bash evaluation splits:

| Metric | Base Model (`qwen2.5-coder:0.5b`) | Fine-Tuned Model (`bash-coder-assistant`) | Improvement |
| :--- | :--- | :--- | :--- |
| **Exact Match (EM %)** | `0.0%` | **`44.0%`** | **+44.0%** |
| **Token F1 Score** | `12.93` | **`81.60`** | **+68.7 pts** |
| **BLEU-4 Score** | `4.60` | **`65.01`** | **+60.4 pts** |
| **Bash Syntax Validity** | `88.0%` | **`100.0%`** | **+12.0%** (0 syntax errors) |
| **Zero-Chatter Clean Format** | `4.0%` | **`100.0%`** | **+96.0%** |
| **Mean Inference Latency** | `19.03s` | **`1.58s`** | **⚡ 91.7% Faster** |

### 🛠️ Track 2: Real-World DevOps, Cloud & SysAdmin Benchmark
Evaluates complex real-world administration tasks across Docker, Git, Networking, Sockets, Systemd, and Process management:

| Metric | Base Model (`qwen2.5-coder:0.5b`) | Fine-Tuned Model (`bash-coder-assistant`) | Improvement |
| :--- | :--- | :--- | :--- |
| **Exact Match (EM %)** | `0.0%` | **`15.0%`** | **+15.0%** |
| **Token F1 Score** | `6.76` | **`62.42`** | **+55.7 pts** |
| **Bash Syntax Validity** | `90.0%` | **`100.0%`** | **+10.0%** (100% executable) |
| **Mean Inference Latency** | `20.98s` | **`1.78s`** | **⚡ 91.5% Faster** |

> **Reproduce Benchmark Locally**:
> ```bash
> python3 scripts/run_academic_benchmark.py
> ```

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
