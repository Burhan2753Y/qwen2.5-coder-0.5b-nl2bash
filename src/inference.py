"""
Interactive or one-shot inference testing for fine-tuned Qwen2.5-0.5B Bash assistant.
"""

import sys
import argparse
import torch
from unsloth import FastLanguageModel

SYSTEM_PROMPT = (
    "You are a Linux and Ubuntu terminal assistant. "
    "Return only the exact, executable Bash command matching the user request "
    "with no markdown formatting or conversational filler."
)

def run_inference(model_path: str, instruction: str):
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model.generate(input_ids=inputs, max_new_tokens=64, use_cache=True)
    generated = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return generated.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NL2Bash fine-tuned Qwen model")
    parser.add_argument("--model_path", type=str, default="models/qwen2.5-coder-0.5b-bash-merged")
    parser.add_argument("--query", type=str, default="Find all processes running on port 8080 and terminate them immediately.")
    args, _ = parser.parse_known_args()

    cmd = run_inference(args.model_path, args.query)
    print("\n--- Output ---")
    print(f"Instruction: {args.query}")
    print(f"Bash Command: {cmd}")
