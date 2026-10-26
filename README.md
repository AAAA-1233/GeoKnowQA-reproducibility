# GeoKnowQA Reproducibility

本仓库提供 GeoKnowQA 数据集相关的模型微调、生成评测、语义重复性分析及 LLM-as-a-Judge 评测代码，用于支持论文实验结果的复现。

共涉及六个模型：Qwen2.5-1.5B、Qwen2.5-7B、InternLM2.5-1.8B、Randeng-T5-784M-QA-Chinese、Qwen3-0.6B 和 MiniCPM5-2B。

## Models

- `models/`：模型适配器与基座模型下载脚本的存放目录。
- `models/adapters/`：存放已公开模型的最终 LoRA/QLoRA 适配器及匹配的 tokenizer 文件；具体可用模型以目录中的实际文件为准。
- `models/download_scripts/`：基座模型下载脚本；下载后的模型写入对应训练目录的 `base_models/`。

## Training

- `training/`：六个模型的数据划分、参数高效微调、文本生成及自动评测代码。
- `training/qwen1_5b_lora/`：Qwen2.5-1.5B 的固定划分、LoRA 微调、生成评测和微调前后指标比较代码。
- `training/qwen7b_lora/`：Qwen2.5-7B 的固定划分、QLoRA 微调、生成评测和微调前后指标比较代码。
- `training/internlm2_5_1_8b_lora/`：InternLM2.5-1.8B 的固定划分、LoRA 微调、生成评测和微调前后指标比较代码。
- `training/randeng_t5_784m_qa_chinese_lora/`：Randeng-T5-784M-QA-Chinese 的固定划分、三轮 LoRA 微调、生成评测、NPEE 评测和微调前后指标比较代码。
- `training/qwen3_0_6b_lora/`：Qwen3-0.6B 的数据划分、参数高效微调、生成评测和微调前后指标比较代码。
- `training/minicpm5_2b_lora/`：MiniCPM5-2B 的数据划分、参数高效微调、生成评测和微调前后指标比较代码。

各模型训练目录中的主要文件按实际实现包括：

- `src/prepare_data.py`：在随机种子 42 下建立 90%/5%/5% 的固定训练、验证和测试划分，并写入划分清单。
- `src/train_lora.py`：加载相应基座模型并执行 LoRA 或 QLoRA 微调，具体方案由模型配置决定。
- `src/evaluate_generation.py`：生成预测，并计算 BLEU-1～4、ROUGE-1/2/L、Distinct-1/2、Token-F1 和 EM。
- `src/compare_metrics.py`：汇总微调前后自动评价指标及其差值。
- `src/common.py`：封装训练和评测共用的数据读取、随机种子、提示词及原子写入功能。
- `configs/*.json`：配置模型、数据、输出位置及训练超参数。
- `run_pipeline.sh`：依次执行数据划分、基座评测、微调、微调后评测及指标对比。

## Datasets

- `datasets/GeoQA-train/`：GeoKnowQA 指令微调原始 JSON 数据集（保留仓库中的历史目录名称）。
- `datasets/GeoQABench/`：GeoKnowQA-Bench 分领域测试题数据。
- `datasets/NPEEBench/`：NPEE 主观题和客观题测试集、划分清单及数据划分脚本。

## Evaluation

- `evaluation/npee/evaluate_npee.py`：统一计算 NPEE 主观题的 BLEU/ROUGE 等文本指标、选择题与判断题准确率，以及填空题文本指标。
- `evaluation/npee/normalize_and_rescore.py`：对 NPEE 客观题输出进行归一化并重新计算各题型准确率。
- `evaluation/npee/summarize_before_after.py`：汇总 NPEE 微调前后结果。
- `evaluation/semantic_similarity.py`：基于句向量与余弦近邻检索检查语义相似度，并输出达到设定阈值的题对。
- `evaluation/llm_as_judge/prompt.txt`：GPT-5.5 地理问答 LLM-as-a-Judge 的固定评分提示词。
- `evaluation/llm_as_judge/llm_as_judge_gpt55.py`：读取预测 JSONL 文件、调用 GPT-5.5、校验评分输出格式，并支持断点续评。

## Dependencies

- `requirements.txt`：训练、自动评测、NPEE、语义相似度及 LLM-as-a-Judge 所需的 Python 依赖。
