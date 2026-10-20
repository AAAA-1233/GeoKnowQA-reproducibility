# GeoBenchmark-NPEE 测试集

来源：<https://huggingface.co/datasets/daven3/geobench>

原始文件：`raw/geobenchmark_npee.json`

本目录保留完整原始文件，并将全部 1,408 道测试题按题型拆为：

- 客观题 466 道：选择题 `choice` 182、填空题 `completion` 150、判断题 `tf` 134。
- 主观题 942 道：名词解释 `noun` 454、简答题 `qa` 153、论述题 `discussion` 335。

## 输出文件

- `test/objective.json`：客观题 JSON 数组。
- `test/objective.jsonl`：客观题逐行 JSON，适合批量推理。
- `test/subjective.json`：主观题 JSON 数组。
- `test/subjective.jsonl`：主观题逐行 JSON，适合批量推理。
- `manifest.json`：来源、分类规则、题量和 SHA256 校验值。
- `split_test.py`：可重复执行的拆分脚本。

每条扁平记录包含 `id`、`category`、`question_type`、`question_type_zh`、
`source_index`、`question` 和 `answer`。拆分只改变组织形式，不修改题目或答案文本。

数据仓库标注许可证为 Apache-2.0。具体使用条件以来源仓库为准。
