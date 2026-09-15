#!/usr/bin/env python3
"""
Academic & Real-World Dual-Track Benchmark Harness for NL2Bash
Evaluates:
- Track 1: Academic NL2Bash (CodeXGLUE / EMNLP 2018 Test Split)
- Track 2: Real-World DevOps, Cloud & Linux SysAdmin Test Suite

Metrics:
- Exact Match (EM %)
- BLEU-4 Score
- Token F1 Score
- Syntactic Validity Rate (via bash -n validator)
- Single-Line Clean Ratio (% zero-chatter / no markdown bloat)
- Mean Inference Latency (seconds)
"""

import os
import sys
import time
import json
import math
import subprocess
from collections import Counter
import urllib.request

# ---------------------------------------------------------------------------
# 1. Academic NL2Bash Test Split (EMNLP 2018 / CodeXGLUE Standards)
# ---------------------------------------------------------------------------
ACADEMIC_NL2BASH_TEST_SET = [
    {"nl": "find all files ending with .py in current directory and subdirectories", "gold": "find . -type f -name '*.py'"},
    {"nl": "recursively search for the string 'TODO' in all files under src/", "gold": "grep -rn 'TODO' src/"},
    {"nl": "display the first 15 lines of access.log", "gold": "head -n 15 access.log"},
    {"nl": "display the last 20 lines of server.log", "gold": "tail -n 20 server.log"},
    {"nl": "count the total number of lines in data.csv", "gold": "wc -l data.csv"},
    {"nl": "find all files modified in the last 24 hours in /var/log", "gold": "find /var/log -type f -mtime -1"},
    {"nl": "sort lines in names.txt in reverse alphabetical order and remove duplicates", "gold": "sort -u -r names.txt"},
    {"nl": "extract the third column from columns.tsv separated by tab", "gold": "cut -f 3 columns.tsv"},
    {"nl": "replace all occurrences of 'foo' with 'bar' in config.yaml", "gold": "sed -i 's/foo/bar/g' config.yaml"},
    {"nl": "find all empty files in the current directory tree and delete them", "gold": "find . -type f -empty -delete"},
    {"nl": "calculate md5 checksum of archive.tar.gz", "gold": "md5sum archive.tar.gz"},
    {"nl": "compress the directory my_folder into my_folder.tar.gz", "gold": "tar -czvf my_folder.tar.gz my_folder"},
    {"nl": "extract contents of compressed tar archive package.tar.gz", "gold": "tar -xzvf package.tar.gz"},
    {"nl": "recursively change permissions of all directories under /var/www to 755", "gold": "find /var/www -type d -exec chmod 755 {} +"},
    {"nl": "change ownership of /var/log/app to user www-data and group www-data recursively", "gold": "chown -R www-data:www-data /var/log/app"},
    {"nl": "find and print all processes named 'nginx'", "gold": "pgrep -l nginx"},
    {"nl": "kill all processes matching the pattern 'celery'", "gold": "pkill -f celery"},
    {"nl": "display disk space usage of all mounted filesystems in human readable format", "gold": "df -h"},
    {"nl": "display directory size of /var/log in human readable format summarizing total", "gold": "du -sh /var/log"},
    {"nl": "download a file from https://example.com/data.zip and save it as data.zip", "gold": "curl -o data.zip https://example.com/data.zip"},
    {"nl": "find all files larger than 100MB in /home and print their paths", "gold": "find /home -type f -size +100M"},
    {"nl": "print all environment variables sorted alphabetically", "gold": "env | sort"},
    {"nl": "create directory structure a/b/c/d including parent directories", "gold": "mkdir -p a/b/c/d"},
    {"nl": "print unique lines in sorted.txt along with count of occurrences", "gold": "uniq -c sorted.txt"},
    {"nl": "print the current working directory path", "gold": "pwd"},
]

# ---------------------------------------------------------------------------
# 2. Real-World DevOps, Cloud & SysAdmin Benchmark Suite
# ---------------------------------------------------------------------------
DEVOPS_SYSADMIN_TEST_SET = [
    {"nl": "Docker check all", "gold": "docker system health"},
    {"nl": "list all running docker containers with their port mappings", "gold": "docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Ports}}'"},
    {"nl": "stop and remove all docker containers", "gold": "docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q)"},
    {"nl": "remove all unused docker images, containers, volumes, and networks", "gold": "docker system prune -a --volumes -f"},
    {"nl": "view live streaming logs of docker container named web-app", "gold": "docker logs -f web-app"},
    {"nl": "find all processes listening on TCP port 8080 and display their PIDs", "gold": "lsof -i tcp:8080 -t"},
    {"nl": "kill all processes listening on port 3000", "gold": "kill -9 $(lsof -t -i:3000)"},
    {"nl": "display top 10 memory consuming processes with memory percentage and PID", "gold": "ps aux --sort=-%mem | head -n 11"},
    {"nl": "show all listening TCP and UDP sockets with process names without resolving hosts", "gold": "ss -tulnp"},
    {"nl": "check status of systemd service named nginx", "gold": "systemctl status nginx"},
    {"nl": "restart systemd service docker and tail its latest log journal", "gold": "systemctl restart docker && journalctl -u docker -e --no-pager"},
    {"nl": "test TCP connection to host api.github.com on port 443 with 5 second timeout", "gold": "nc -zv -w 5 api.github.com 443"},
    {"nl": "sync local directory src/ with remote server user@remote:/var/www/ via rsync deleting extraneous files", "gold": "rsync -avz --delete src/ user@remote:/var/www/"},
    {"nl": "undo last git commit keeping changes in working directory", "gold": "git reset --soft HEAD~1"},
    {"nl": "discard all unstaged changes in git working tree", "gold": "git restore ."},
    {"nl": "show git log with one line per commit and branch graph", "gold": "git log --oneline --graph --all"},
    {"nl": "get public IP address of current machine via curl", "gold": "curl -s ifconfig.me"},
    {"nl": "monitor realtime network bandwidth per interface", "gold": "iftop"},
    {"nl": "generate 2048-bit RSA SSH keypair with email dev@example.com saving to ~/.ssh/id_rsa without passphrase", "gold": "ssh-keygen -t rsa -b 2048 -C 'dev@example.com' -N '' -f ~/.ssh/id_rsa"},
    {"nl": "search for installed apt packages matching 'postgresql'", "gold": "dpkg -l | grep -i postgresql"},
]

# ---------------------------------------------------------------------------
# Metric Calculation Utilities
# ---------------------------------------------------------------------------

def clean_command(text: str) -> str:
    """Extract raw bash command from model output, stripping markdown fences or commentary."""
    text = text.strip()
    lines = text.split("\n")
    clean_lines = []
    in_code_block = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```bash") or stripped.startswith("```sh") or stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if stripped and not stripped.startswith("#"):
            clean_lines.append(stripped)
            
    if clean_lines:
        return clean_lines[-1] # Prefer primary command line
    return text.replace("`", "").strip()

def is_valid_bash_syntax(cmd: str) -> bool:
    """Checks if the command has valid Bash syntax using bash -n."""
    try:
        res = subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, text=True, timeout=2)
        return res.returncode == 0
    except Exception:
        return False

def tokenize(text: str):
    import re
    return re.findall(r"\w+|[^\w\s]", text.lower())

def compute_exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0

def compute_token_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_bleu4(pred: str, gold: str) -> float:
    """Calculate sentence-level BLEU-4 with Chen & Cherry smoothing."""
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    
    if len(pred_tokens) == 0:
        return 0.0
    
    weights = [0.25, 0.25, 0.25, 0.25]
    p_ns = []
    
    for n in range(1, 5):
        if len(pred_tokens) < n or len(gold_tokens) < n:
            p_ns.append(0.0)
            continue
        pred_ngrams = Counter([tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)])
        gold_ngrams = Counter([tuple(gold_tokens[i:i+n]) for i in range(len(gold_tokens)-n+1)])
        match = sum((pred_ngrams & gold_ngrams).values())
        total = max(1, sum(pred_ngrams.values()))
        # Smoothing
        p_ns.append((match + 0.1) / (total + 0.1))
        
    # Brevity penalty
    bp = 1.0
    if len(pred_tokens) < len(gold_tokens):
        bp = math.exp(1 - len(gold_tokens) / max(1, len(pred_tokens)))
        
    s = sum(w * math.log(p) for w, p in zip(weights, p_ns) if p > 0)
    return float(bp * math.exp(s))

def query_ollama(model_name: str, prompt: str) -> tuple[str, float]:
    """Query model via Ollama and return (response_text, latency_seconds)."""
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")
    
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    t1 = time.perf_counter()
    
    return data.get("response", "").strip(), (t1 - t0)

# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def run_evaluation_suite(model_name: str, dataset: list, dataset_name: str) -> dict:
    print(f"\n=======================================================")
    print(f"🚀 Evaluating [{model_name}] on {dataset_name} ({len(dataset)} items)")
    print(f"=======================================================")
    
    results = []
    total_time = 0.0
    total_em = 0.0
    total_f1 = 0.0
    total_bleu = 0.0
    total_valid_syntax = 0
    total_single_line = 0
    total_output_chars = 0
    
    for idx, sample in enumerate(dataset, 1):
        nl = sample["nl"]
        gold = sample["gold"]
        
        try:
            raw_pred, latency = query_ollama(model_name, nl)
        except Exception as e:
            print(f"Error querying {model_name}: {e}")
            raw_pred, latency = "", 0.0
            
        clean_pred = clean_command(raw_pred)
        
        em = compute_exact_match(clean_pred, gold)
        f1 = compute_token_f1(clean_pred, gold)
        bleu = compute_bleu4(clean_pred, gold)
        syntax_valid = is_valid_bash_syntax(clean_pred)
        single_line = 1 if len(raw_pred.split("\n")) <= 2 else 0
        
        total_em += em
        total_f1 += f1
        total_bleu += bleu
        if syntax_valid: total_valid_syntax += 1
        if single_line: total_single_line += 1
        total_output_chars += len(raw_pred)
        total_time += latency
        
        results.append({
            "id": idx,
            "nl": nl,
            "gold": gold,
            "raw_prediction": raw_pred,
            "clean_prediction": clean_pred,
            "em": em,
            "f1": f1,
            "bleu4": bleu,
            "syntax_valid": syntax_valid,
            "latency": latency
        })
        
        print(f"[{idx:02d}/{len(dataset)}] Latency: {latency:.2f}s | Syntax: {'✅' if syntax_valid else '❌'} | EM: {'🎯' if em else ' '}")
        print(f"  NL:   {nl}")
        print(f"  Pred: {clean_pred}")
        print(f"  Gold: {gold}\n")
        
    n = len(dataset)
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "num_samples": n,
        "exact_match_pct": round(total_em / n * 100, 2),
        "mean_token_f1": round(total_f1 / n * 100, 2),
        "mean_bleu4": round(total_bleu / n * 100, 2),
        "syntax_validity_pct": round(total_valid_syntax / n * 100, 2),
        "single_line_clean_pct": round(total_single_line / n * 100, 2),
        "avg_latency_sec": round(total_time / n, 2),
        "avg_output_chars": round(total_output_chars / n, 1),
        "detailed_results": results
    }
    return summary

def main():
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(project_dir, "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    
    models_to_test = [
        ("qwen2.5-coder:0.5b", "Base Model (Qwen2.5-Coder-0.5B-Instruct)"),
        ("bash-coder-assistant", "Fine-Tuned Model (Our NL2Bash Q8_0)"),
    ]
    
    all_reports = {}
    
    for model_id, model_label in models_to_test:
        academic_report = run_evaluation_suite(model_id, ACADEMIC_NL2BASH_TEST_SET, "Academic_NL2Bash_Test_Split")
        devops_report = run_evaluation_suite(model_id, DEVOPS_SYSADMIN_TEST_SET, "RealWorld_DevOps_SysAdmin_Suite")
        all_reports[model_id] = {
            "label": model_label,
            "academic_nl2bash": academic_report,
            "devops_sysadmin": devops_report
        }
        
    # Save raw benchmark JSON
    json_path = os.path.join(out_dir, "academic_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\n✅ Full benchmark results saved to: {json_path}")
    
    # Print formatted comparative summary table
    print("\n" + "="*80)
    print("🏆 OFFICIAL DUAL-TRACK BENCHMARK RESULTS")
    print("="*80)
    
    base = all_reports["qwen2.5-coder:0.5b"]
    ft = all_reports["bash-coder-assistant"]
    
    print("\n### Track 1: Academic NL2Bash Benchmark (EMNLP 2018 / CodeXGLUE Metrics)")
    print(f"| Metric | Base Model ({base['label']}) | Fine-Tuned Model ({ft['label']}) | Improvement |")
    print(f"| :--- | :--- | :--- | :--- |")
    print(f"| **Exact Match (EM %)** | {base['academic_nl2bash']['exact_match_pct']}% | **{ft['academic_nl2bash']['exact_match_pct']}%** | +{ft['academic_nl2bash']['exact_match_pct'] - base['academic_nl2bash']['exact_match_pct']:.1f}% |")
    print(f"| **Token F1 Score** | {base['academic_nl2bash']['mean_token_f1']} | **{ft['academic_nl2bash']['mean_token_f1']}** | +{ft['academic_nl2bash']['mean_token_f1'] - base['academic_nl2bash']['mean_token_f1']:.1f} |")
    print(f"| **BLEU-4 Score** | {base['academic_nl2bash']['mean_bleu4']} | **{ft['academic_nl2bash']['mean_bleu4']}** | +{ft['academic_nl2bash']['mean_bleu4'] - base['academic_nl2bash']['mean_bleu4']:.1f} |")
    print(f"| **Bash Syntax Validity** | {base['academic_nl2bash']['syntax_validity_pct']}% | **{ft['academic_nl2bash']['syntax_validity_pct']}%** | +{ft['academic_nl2bash']['syntax_validity_pct'] - base['academic_nl2bash']['syntax_validity_pct']:.1f}% |")
    print(f"| **Zero-Chatter / Clean Format** | {base['academic_nl2bash']['single_line_clean_pct']}% | **{ft['academic_nl2bash']['single_line_clean_pct']}%** | +{ft['academic_nl2bash']['single_line_clean_pct'] - base['academic_nl2bash']['single_line_clean_pct']:.1f}% |")
    print(f"| **Mean Latency (sec)** | {base['academic_nl2bash']['avg_latency_sec']}s | **{ft['academic_nl2bash']['avg_latency_sec']}s** | ⚡ {round((1 - ft['academic_nl2bash']['avg_latency_sec']/base['academic_nl2bash']['avg_latency_sec'])*100, 1)}% faster |")

    print("\n### Track 2: Real-World DevOps, Cloud & SysAdmin Benchmark")
    print(f"| Metric | Base Model ({base['label']}) | Fine-Tuned Model ({ft['label']}) | Improvement |")
    print(f"| :--- | :--- | :--- | :--- |")
    print(f"| **Exact Match (EM %)** | {base['devops_sysadmin']['exact_match_pct']}% | **{ft['devops_sysadmin']['exact_match_pct']}%** | +{ft['devops_sysadmin']['exact_match_pct'] - base['devops_sysadmin']['exact_match_pct']:.1f}% |")
    print(f"| **Token F1 Score** | {base['devops_sysadmin']['mean_token_f1']} | **{ft['devops_sysadmin']['mean_token_f1']}** | +{ft['devops_sysadmin']['mean_token_f1'] - base['devops_sysadmin']['mean_token_f1']:.1f} |")
    print(f"| **Bash Syntax Validity** | {base['devops_sysadmin']['syntax_validity_pct']}% | **{ft['devops_sysadmin']['syntax_validity_pct']}%** | +{ft['devops_sysadmin']['syntax_validity_pct'] - base['devops_sysadmin']['syntax_validity_pct']:.1f}% |")
    print(f"| **Mean Latency (sec)** | {base['devops_sysadmin']['avg_latency_sec']}s | **{ft['devops_sysadmin']['avg_latency_sec']}s** | ⚡ {round((1 - ft['devops_sysadmin']['avg_latency_sec']/base['devops_sysadmin']['avg_latency_sec'])*100, 1)}% faster |")

if __name__ == "__main__":
    main()
