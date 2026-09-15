"""
Publish Fine-Tuned Qwen2.5-Coder NL2Bash GGUF model to Hugging Face Hub.
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct
tags:
- unsloth
- text-generation-inference
- llama-cpp
- gguf
- ollama
- nl2bash
- bash
- terminal-assistant
datasets:
- emirkaanozdemr/bash_command_data_6K
language:
- en
pipeline_tag: text-generation
---

# 🚀 Qwen2.5-Coder-0.5B-Instruct - NL2Bash (GGUF Quantized)

A fine-tuned version of **`Qwen/Qwen2.5-Coder-0.5B-Instruct`** specialized for converting natural language instructions directly into executable Linux/Ubuntu Bash commands with **zero conversational filler**.

## 📊 Dual-Track Benchmark Results

### 🎓 Track 1: Academic NL2Bash Benchmark (EMNLP 2018 / Microsoft CodeXGLUE)
| Metric | Base Model (`qwen2.5-coder:0.5b`) | Fine-Tuned Model (`bash-coder-assistant`) | Improvement |
| :--- | :--- | :--- | :--- |
| **Exact Match (EM %)** | `0.0%` | **`44.0%`** | **+44.0%** |
| **Token F1 Score** | `12.93` | **`81.60`** | **+68.7 pts** |
| **BLEU-4 Score** | `4.60` | **`65.01`** | **+60.4 pts** |
| **Bash Syntax Validity** | `88.0%` | **`100.0%`** | **+12.0%** |
| **Zero-Chatter Clean Format** | `4.0%` | **`100.0%`** | **+96.0%** |
| **Mean Latency (CPU)** | `19.03s` | **`1.58s`** | **⚡ 91.7% Faster** |

### 🛠️ Track 2: Real-World DevOps, Cloud & SysAdmin Benchmark
| Metric | Base Model (`qwen2.5-coder:0.5b`) | Fine-Tuned Model (`bash-coder-assistant`) | Improvement |
| :--- | :--- | :--- | :--- |
| **Exact Match (EM %)** | `0.0%` | **`15.0%`** | **+15.0%** |
| **Token F1 Score** | `6.76` | **`62.42`** | **+55.7 pts** |
| **Bash Syntax Validity** | `90.0%` | **`100.0%`** | **+10.0%** |
| **Mean Latency (CPU)** | `20.98s` | **`1.78s`** | **⚡ 91.5% Faster** |

## 🦙 Quickstart with Ollama

1. Download the `qwen2.5-coder-0.5b-bash-gguf.gguf` file and `Modelfile`.
2. Create the model in Ollama:
```bash
ollama create bash-coder-assistant -f Modelfile
```
3. Run a query:
```bash
ollama run bash-coder-assistant "Find all processes running on port 8080 and terminate them immediately"
# Output: sudo kill -9 $(lsof -i :8080 | awk '{print $2}')
```

## 🛠️ Training Details
- **Architecture**: Qwen2.5-Coder (0.5B parameters)
- **Framework**: [Unsloth](https://github.com/unslothai/unsloth) + TRL SFTTrainer
- **Dataset**: `emirkaanozdemr/bash_command_data_6K`
- **LoRA Rank**: `r=16`, `lora_alpha=16`
- **Training Compute**: Google Colab Tesla T4 GPU (< 2GB VRAM, ~8 min runtime)
- **Final Loss**: `0.6726` (converged from initial `4.1045`)
"""

def upload_to_hf(repo_id: str, token: str = None, private: bool = False):
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    
    print(f"Creating / verifying Hugging Face repository: {repo_id}...")
    create_repo(repo_id=repo_id, token=token or os.environ.get("HF_TOKEN"), private=private, exist_ok=True)

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    gguf_path = os.path.join(project_dir, "models/qwen2.5-coder-0.5b-bash-gguf/qwen2.5-coder-0.5b-bash-gguf.gguf")
    modelfile_path = os.path.join(project_dir, "Modelfile")

    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF file not found at: {gguf_path}")

    print("Uploading README.md (Model Card)...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD_TEMPLATE.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card documentation",
    )

    print("Uploading Modelfile for Ollama...")
    if os.path.exists(modelfile_path):
        api.upload_file(
            path_or_fileobj=modelfile_path,
            path_in_repo="Modelfile",
            repo_id=repo_id,
            commit_message="Add Ollama Modelfile",
        )

    print(f"Uploading GGUF weights ({os.path.getsize(gguf_path)/(1024*1024):.2f} MB)...")
    api.upload_file(
        path_or_fileobj=gguf_path,
        path_in_repo="qwen2.5-coder-0.5b-bash-gguf.gguf",
        repo_id=repo_id,
        commit_message="Upload quantized GGUF weights",
    )

    print(f"\n🎉 Successfully published model to: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload fine-tuned model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo id (e.g. username/qwen2.5-coder-0.5b-nl2bash-gguf)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Make Hugging Face repo private")
    args, _ = parser.parse_known_args()

    upload_to_hf(args.repo_id, args.token, args.private)
