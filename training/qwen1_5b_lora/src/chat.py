#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="base_models/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default="outputs/qwen2.5-1.5b-lora/final_adapter")
    p.add_argument("--question", required=True)
    p.add_argument("--system", default="你是一名严谨的地理学问答助手。")
    args = p.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    model_path = Path(args.model)
    adapter_path = Path(args.adapter)
    if not model_path.is_absolute() and (project_dir / model_path).exists():
        args.model = str(project_dir / model_path)
    if not adapter_path.is_absolute() and (project_dir / adapter_path).exists():
        args.adapter = str(project_dir / adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    )
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": args.system}, {"role": "user", "content": args.question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=384, do_sample=False)
    print(tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))


if __name__ == "__main__":
    main()
