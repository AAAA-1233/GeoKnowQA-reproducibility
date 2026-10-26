<p align="right">
  <a href="./README-EN.md">English</a> |
  <a href="./README.md">简体中文</a>
</p>

# GeoKnowQA Reproducibility

This repository provides the GeoKnowQA dataset and the associated code for model fine-tuning, evaluation, and semantic similarity analysis, supporting the reproducibility of the experimental results reported in the paper.

The repository covers six models: Qwen2.5-1.5B, Qwen2.5-7B, InternLM2.5-1.8B, Randeng-T5-784M-QA-Chinese, Qwen3-0.6B, and MiniCPM5-2B.

## Models

- `models/`: Contains model adapters and scripts for downloading the base models.
- `models/adapters/`: Stores the fine-tuned LoRA/QLoRA adapters and corresponding tokenizer files. The availability of specific models depends on the files included in this directory.
- `models/download_scripts/`: Contains scripts for downloading the base models. Downloaded models are saved to the `base_models/` directory within the corresponding training directories.

## Training

- `training/`: Contains the code for data splitting, fine-tuning, text generation, and automated evaluation of the six models.

The main files in each model's training directory include:

- `src/prepare_data.py`: Creates fixed training, validation, and test splits with a ratio of 90%/5%/5% using a random seed of 42, and saves the corresponding split records.
- `src/train_lora.py`: Loads the corresponding base model and performs LoRA or QLoRA fine-tuning, depending on the model configuration.
- `src/evaluate_generation.py`: Generates model predictions and computes BLEU-1 to BLEU-4, ROUGE-1/2/L, and Distinct-1/2 scores.
- `src/compare_metrics.py`: Summarizes the automatic evaluation metrics before and after fine-tuning and calculates their differences.
- `src/common.py`: Provides shared utilities for data loading, random seed initialization, prompt formatting, and atomic file writing.
- `configs/*.json`: Specifies model configurations, dataset paths, output directories, and training hyperparameters.
- `run_pipeline.sh`: Executes the complete pipeline, including data splitting, base model evaluation, fine-tuning, post-fine-tuning evaluation, and performance comparison.

## Datasets

- `datasets/GeoQA-train/`: Contains the original GeoKnowQA-Train instruction fine-tuning dataset in JSON format.
- `datasets/GeoQABench/`: Contains the GeoKnowQA-Bench evaluation dataset, organized by geographical subdiscipline.
- `datasets/NPEEBench/`: Contains the NPEE subjective and objective question datasets, data split records, and data splitting scripts.

## Evaluation

- `evaluation/npee/evaluate_npee.py`: Evaluates model performance on the NPEE dataset before and after fine-tuning.
- `evaluation/npee/normalize_and_rescore.py`: Normalizes model outputs for NPEE objective questions and recalculates accuracy for different question types.
- `evaluation/npee/summarize_before_after.py`: Summarizes NPEE evaluation results before and after fine-tuning.
- `evaluation/semantic_similarity.py`: Performs semantic similarity analysis using sentence embeddings and cosine-based nearest-neighbor retrieval, and outputs question pairs exceeding a specified similarity threshold.
- `evaluation/llm_as_judge/prompt.txt`: Contains the prompt used for LLM-as-a-Judge evaluation.
- `evaluation/llm_as_judge/llm_as_judge_gpt55.py`: Reads prediction files in JSONL format, invokes the LLM for evaluation, validates the output score format, and supports resuming interrupted evaluations.

## Dependencies

- `requirements.txt`: Lists the required Python dependencies.
