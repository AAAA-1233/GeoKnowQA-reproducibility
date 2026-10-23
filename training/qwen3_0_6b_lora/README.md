# Qwen3-0.6B LoRA

This folder archives the exact training and evaluation code used for the Qwen3-0.6B experiment. The final adapter is located at `../../models/adapters/qwen3-0.6b-lora/`.

The configuration records three epochs, an effective batch size of 64 (batch size 4 with 16 accumulation steps), maximum length 512, BF16 LoRA over attention and MLP projections, and learning rate `5e-5`. Qwen3 training disables thinking and strips `<think>...</think>` traces from target answers. Generation uses the non-thinking template, 512 generated tokens, repetition penalty 1.15, and 4-gram blocking.

Set the base-model and data paths in `configs/qwen3_0_6b_lora.json` before re-running training. `run_qwen3_0.6b_npee.sh` evaluates choice and true/false NPEE items before and after adaptation.
