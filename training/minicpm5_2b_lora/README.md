# MiniCPM5-2B LoRA

This folder archives the exact training and evaluation code used for the MiniCPM5-2B experiment. The final adapter is located at `../../models/adapters/minicpm5-2b-lora/`.

The configuration records three epochs, an effective batch size of 64, maximum length 512, BF16 LoRA (`q_proj`, `v_proj`), and learning rate `1e-5`. `src/evaluate_minicpm_generation.py` uses the non-thinking template, 512 generated tokens, repetition penalty 1.15, and 4-gram blocking.

Set the base-model and data paths in `configs/minicpm5_2b_lora.json` before re-running training. The archived scripts retain the original experiment paths as a provenance record.
