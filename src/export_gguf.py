"""
Reliable GGUF export for Hugging Face models using llama.cpp's convert_hf_to_gguf.py.
Works seamlessly without CMake / Makefile build failures.
"""

import os
import sys
import subprocess
import argparse

def export_to_gguf(
    model_dir: str = "models/qwen2.5-coder-0.5b-bash-merged",
    output_gguf: str = "models/qwen2.5-coder-0.5b-bash-gguf/qwen2.5-coder-0.5b-bash-gguf.gguf",
    outtype: str = "q8_0",  # q8_0 or f16
):
    print("=" * 60)
    print(f"📦 Exporting '{model_dir}' to GGUF: '{output_gguf}' ({outtype})")
    print("=" * 60)

    llama_cpp_dir = "/tmp/llama.cpp"
    if not os.path.exists(llama_cpp_dir):
        print("Cloning llama.cpp repository...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir], check=True)
        print("Installing llama.cpp requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{llama_cpp_dir}/requirements.txt"], check=True)

    os.makedirs(os.path.dirname(output_gguf) or ".", exist_ok=True)

    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    cmd = [
        sys.executable,
        convert_script,
        model_dir,
        "--outfile", output_gguf,
        "--outtype", outtype,
    ]

    print(f"Running conversion: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    file_size_mb = os.path.getsize(output_gguf) / (1024 * 1024)
    print(f"✅ GGUF export successful! Size: {file_size_mb:.2f} MB")
    print(f"Saved at: {output_gguf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export merged HF model to GGUF format")
    parser.add_argument("--model_dir", type=str, default="models/qwen2.5-coder-0.5b-bash-merged")
    parser.add_argument("--output_gguf", type=str, default="models/qwen2.5-coder-0.5b-bash-gguf/qwen2.5-coder-0.5b-bash-gguf.gguf")
    parser.add_argument("--outtype", type=str, default="q8_0")
    args, _ = parser.parse_known_args()

    export_to_gguf(args.model_dir, args.output_gguf, args.outtype)
