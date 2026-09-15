"""
Fine-tuning Qwen2.5-Coder-0.5B-Instruct on NL2Bash using Unsloth.
Optimized for Google Colab Tesla T4 GPU (< 2GB VRAM usage, < 15 min runtime).
Includes real-time GPU memory tracking, TensorBoard, and Weights & Biases (WandB) monitoring.
"""

import os
import sys
import subprocess
import argparse

# Auto-install dependencies if running in Colab / cloud environment where unsloth is missing
try:
    import unsloth
except ImportError:
    print("Installing Unsloth and fine-tuning dependencies...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "xformers<0.0.29", "trl<0.9.0", "peft", "accelerate", "bitsandbytes"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "unsloth", "tensorboard", "datasets", "transformers"], check=True)
    print("Dependencies installed successfully!", flush=True)

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def print_gpu_stats(label=""):
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"[{label}] GPU: {gpu_stats.name} | Total VRAM: {max_memory} GB | Reserved VRAM: {start_gpu_memory} GB")

def train(
    model_name: str = "unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit",
    dataset_name: str = "emirkaanozdemr/bash_command_data_6K",
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 16,
    max_steps: int = 300,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    learning_rate: float = 2e-4,
    logging_steps: int = 10,
    report_to: str = "tensorboard",
    wandb_project: str = "qwen2.5-coder-bash",
    output_dir: str = "outputs",
    merged_dir: str = "models/qwen2.5-coder-0.5b-bash-merged",
    gguf_dir: str = "models/qwen2.5-coder-0.5b-bash-gguf",
    quantization_method: str = "q4_k_m",
):
    print("=" * 60)
    print(f"Step 1: Loading model '{model_name}' (4-bit)")
    print("=" * 60)
    print_gpu_stats("Initial State")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detects fp16 (T4) or bf16 (Ampere+)
        load_in_4bit=True,
    )
    print_gpu_stats("After Loading 4-bit Base Model")

    print("=" * 60)
    print(f"Step 2: Attaching LoRA Adapters (r={lora_r}, alpha={lora_alpha})")
    print("=" * 60)

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,  # Unsloth optimized
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print_gpu_stats("After LoRA Attached")

    print("=" * 60)
    print(f"Step 3: Preparing Dataset '{dataset_name}' with ChatML template")
    print("=" * 60)

    dataset = load_dataset(dataset_name, split="train")

    def format_prompts(batch):
        # Support various column names (prompt/completion, instruction/command, etc.)
        prompts = batch.get("prompt") or batch.get("instruction") or batch.get("input") or batch.get("question")
        completions = batch.get("completion") or batch.get("command") or batch.get("output") or batch.get("response")
        
        formatted_texts = []
        for instruction, command in zip(prompts, completions):
            messages = [
                {
                    "role": "system",
                    "content": "You are a Linux and Ubuntu terminal assistant. Return only the exact, executable Bash command matching the user request with no markdown formatting or conversational filler."
                },
                {"role": "user", "content": str(instruction).strip()},
                {"role": "assistant", "content": str(command).strip()}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            formatted_texts.append(text)
        return {"text": formatted_texts}

    dataset = dataset.map(format_prompts, batched=True)
    print(f"Dataset mapped. Total examples: {len(dataset)}")

    print("=" * 60)
    print(f"Step 4: Training with SFTTrainer (Monitor: {report_to})")
    print(f"Max steps: {max_steps} | Batch: {batch_size} (accum: {grad_accum_steps}) | LR: {learning_rate}")
    print("=" * 60)

    if "wandb" in report_to:
        os.environ["WANDB_PROJECT"] = wandb_project

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Keep short commands isolated
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            warmup_steps=10,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            logging_first_step=True,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to=report_to,
            logging_dir=os.path.join(output_dir, "logs"),
        ),
    )

    trainer_stats = trainer.train()

    print("\n" + "=" * 60)
    print("🎉 Training Finished! Performance & VRAM Summary:")
    print("=" * 60)
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_pct = round(used_memory / max_memory * 100, 2)
        print(f"Peak VRAM used: {used_memory} GB / {max_memory} GB ({used_pct}%)")
    print(f"Runtime: {trainer_stats.metrics.get('train_runtime', 0):.2f} seconds")
    print(f"Train loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")

    print("=" * 60)
    print("Step 5: Testing Inference")
    print("=" * 60)

    FastLanguageModel.for_inference(model)

    test_prompt = [
        {
            "role": "system",
            "content": "You are a Linux and Ubuntu terminal assistant. Return only the exact, executable Bash command matching the user request with no markdown formatting or conversational filler."
        },
        {"role": "user", "content": "Find all processes running on port 8080 and terminate them immediately."}
    ]

    inputs = tokenizer.apply_chat_template(
        test_prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True)
    generated = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"[Prompt]: Find all processes running on port 8080 and terminate them immediately.")
    print(f"[Generated Command]: {generated.strip()}\n")

    print("=" * 60)
    print("Step 6: Exporting Merged 16-bit Model & GGUF Format")
    print("=" * 60)

    os.makedirs(os.path.dirname(merged_dir) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(gguf_dir) or ".", exist_ok=True)

    print(f"Saving merged 16-bit model to: {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    # Reliable GGUF conversion using official llama.cpp convert script
    print(f"Converting merged model to GGUF format...")
    llama_cpp_dir = "/tmp/llama.cpp"
    if not os.path.exists(llama_cpp_dir):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", llama_cpp_dir], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{llama_cpp_dir}/requirements.txt"], check=True)
    
    out_gguf_path = os.path.join(gguf_dir, "qwen2.5-coder-0.5b-bash-gguf.gguf") if not gguf_dir.endswith(".gguf") else gguf_dir
    os.makedirs(os.path.dirname(out_gguf_path) or ".", exist_ok=True)
    
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    subprocess.run([sys.executable, convert_script, merged_dir, "--outfile", out_gguf_path, "--outtype", "q8_0"], check=True)

    print("\n✅ All tasks completed successfully!")
    print(f"- GGUF file located at: {out_gguf_path}")
    print(f"- Merged weights located at: {merged_dir}")

    # Auto-copy to Google Drive if drive is mounted
    if os.path.exists("/content/drive/MyDrive"):
        import shutil
        drive_path = "/content/drive/MyDrive/qwen2.5-coder-0.5b-bash-gguf.gguf"
        print(f"\n📂 Auto-copying to Google Drive: {drive_path}")
        shutil.copy(out_gguf_path, drive_path)
        print("✅ Successfully copied model to your Google Drive!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-Coder-0.5B on NL2Bash using Unsloth")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit")
    parser.add_argument("--dataset_name", type=str, default="emirkaanozdemr/bash_command_data_6K")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none"])
    parser.add_argument("--wandb_project", type=str, default="qwen2.5-coder-bash")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--merged_dir", type=str, default="models/qwen2.5-coder-0.5b-bash-merged")
    parser.add_argument("--gguf_dir", type=str, default="models/qwen2.5-coder-0.5b-bash-gguf")
    parser.add_argument("--quantization", type=str, default="q4_k_m")
    args, _ = parser.parse_known_args()

    train(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        wandb_project=args.wandb_project,
        output_dir=args.output_dir,
        merged_dir=args.merged_dir,
        gguf_dir=args.gguf_dir,
        quantization_method=args.quantization,
    )
