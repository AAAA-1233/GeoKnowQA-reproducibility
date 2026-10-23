#!/usr/bin/env python3
"""Evaluate choice and true/false NPEE items by candidate-answer likelihood."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--disable-thinking", action="store_true",
                   help="Render a Qwen3 non-thinking chat prefix.")
    return p.parse_args()


def candidates(row: dict) -> list[str]:
    if row["question_type"] == "choice":
        # Determine choices present in the source item, preserving A--D order.
        return [x for x in "ABCD" if f"\n{x}." in "\n" + row["question"]]
    if row["question_type"] == "tf":
        return ["True", "False"]
    raise ValueError(row["question_type"])


def prompt_for(row: dict, tokenizer, disable_thinking: bool) -> str:
    if row["question_type"] == "choice":
        instruction = (
            "Answer this multiple-choice geology question. Return only the single "
            "correct option letter, with no explanation.\n\n" + row["question"]
        )
    else:
        instruction = (
            "Decide whether this geology statement is true or false. Return only "
            "True or False, with no explanation.\n\nStatement: " + row["question"]
        )
    template_args = {"enable_thinking": False} if disable_thinking else {}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False,
        add_generation_prompt=True, **template_args
    )


def main() -> None:
    args = parse_args()
    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines()]
    rows = [x for x in rows if x["question_type"] in {"choice", "tf"}]
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True, device_map={"": 0},
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    expanded = []
    for idx, row in enumerate(rows):
        prompt = prompt_for(row, tokenizer, args.disable_thinking)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for candidate in candidates(row):
            full_ids = tokenizer(prompt + candidate, add_special_tokens=False)["input_ids"]
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError(f"Candidate tokenization changed prompt prefix for {row['id']}")
            answer_ids = full_ids[len(prompt_ids):]
            if not answer_ids:
                raise ValueError(f"Empty candidate tokens: {candidate}")
            expanded.append((idx, candidate, full_ids, len(prompt_ids), answer_ids))

    scores: list[dict[str, float]] = [dict() for _ in rows]
    with torch.inference_mode():
        for start in range(0, len(expanded), args.batch_size):
            batch = expanded[start : start + args.batch_size]
            ids = [x[2] for x in batch]
            encoded = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
            encoded.pop("token_type_ids", None)
            encoded = encoded.to(model.device)
            logits = model(**encoded).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            for i, (row_idx, candidate, _full_ids, prompt_len, answer_ids) in enumerate(batch):
                positions = torch.arange(prompt_len - 1, prompt_len - 1 + len(answer_ids), device=logits.device)
                target = torch.tensor(answer_ids, device=logits.device)
                score = log_probs[i, positions, :].gather(1, target[:, None]).sum().item()
                scores[row_idx][candidate] = score
            print(f"scored {min(start + len(batch), len(expanded))}/{len(expanded)} candidates", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    by_type: dict[str, list[int]] = {"choice": [0, 0], "tf": [0, 0]}
    with output.open("w", encoding="utf-8") as f:
        for row, row_scores in zip(rows, scores):
            pred = max(row_scores, key=row_scores.get)
            hit = pred == row["answer"]
            correct += hit
            by_type[row["question_type"]][0] += int(hit)
            by_type[row["question_type"]][1] += 1
            f.write(json.dumps({
                "id": row["id"], "question_type": row["question_type"],
                "answer": row["answer"], "prediction": pred, "correct": hit,
                "scores": row_scores,
            }, ensure_ascii=False) + "\n")
    summary = {
        "model": args.model, "adapter": args.adapter, "items": len(rows), "correct": correct,
        "accuracy": correct / len(rows),
        "by_type": {k: {"correct": v[0], "items": v[1], "accuracy": v[0] / v[1]} for k, v in by_type.items()},
        "scoring": "sum of conditional log-probabilities of exact permitted answers",
        "disable_thinking": args.disable_thinking,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
