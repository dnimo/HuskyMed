# HuskyMed

English | [日本語](README.ja.md) | 中文

**HuskyMed** 是基于大语言模型（LLM）的长临床记录自动摘要系统，整合了多种动态上下文扩展策略，旨在缓解超长电子病历（EMR）场景下的信息丢失问题。本项目源于作者在京都大学医学部附属病院的硕士研究工作，由 **黒田知宏** 教授指导完成。

## 研究背景

临床记录常包含数万字的非结构化文本，传统摘要方法在处理超长上下文时面临信息丢失与计算资源瓶颈。本项目提出并实现了：

- **NBCE（Neighbor-Based Context Extension）**：基于信息熵的动态采样策略，通过邻域密度与信息增益筛选关键句段
- **PCW（Piecewise Context Window）**：分段窗口上下文管理，保证长文档推理时的连贯性与计算效率

相关研究成果已发表或投稿至以下会议/期刊：

1. **ZHANG Guoqing**, Keita Fukuyama, Kazumasa Kishimoto, Kana Eguchi, Tomohiro Kuroda (2024).  
   *Long Electronic Medical Record Summarization with Dynamic Minimize Entropy Decoding for LLM.*  
   日本医療情報学会看護学術大会.

2. **Guoqing ZHANG**, Keita Fukuyama, Kazumasa Kishimoto, Tomohiro Kuroda (2024).  
   *Mitigating Context Loss in Clinical Notes Auto-Summarization through Information Theory-Based Dynamic Decoding.*  
   第44回医療情報学連合大会.

3. Zhang, G., Fukuyama, K., Kishimoto, K., & Kuroda, T. (2024).  
   *Optimizing Automatic Summarization of Long Clinical Records Using Dynamic Context Extension: Testing and Evaluation of the NBCE Method.*  
   arXiv preprint: [arXiv:2411.08586](https://arxiv.org/abs/2411.08586)

## 核心特性

- **多种上下文扩展策略**：支持 NBCE 与 PCW 两种生成模式，可通过统一 API 切换
- **参数高效微调**：基于 LoRA 的指令微调流程，支持 GPT-NeoX 与 LLaMA 系列模型
- **完整评测流水线**：集成 ROUGE/BLEU/自定义指标，支持批量实验与性能矩阵聚合
- **工程化架构**：模块化设计，配置与代码分离，易于扩展与复现

## 项目结构

```text
config/                   # 配置管理（路径、超参、模板）
  loader.py              # 统一配置加载器

dataset_builder/         # 数据预处理与指令构建
  datasets/
    build_dataset.py     # 指令数据集生成（tokenization + TF-IDF 对齐）
    data_pipeline.py     # EMR 原始数据 → 标准化格式流水线

model/                   # 模型加载与推理
  base_loader.py         # 基础模型 + LoRA 统一加载
  wrappers/              # 上下文扩展策略实现
    generation.py        # NBCE / PCW 核心逻辑
    generation_manager.py # 策略管理器
    attention.py         # Flash Attention 替换（训练优化）
    constants.py         # 提示词模板与常量
    utils.py             # 采样率 sweep、长文本过滤等工具

evaluation/              # 评测指标与聚合
  rouge.py               # ROUGE 指标（支持 MeCab 分词）
  bleu.py                # BLEU 指标（SacreBLEU）
  aggregator.py          # 批量评分与分组聚合引擎

scripts/                 # 命令行入口
  train.py               # LoRA 指令微调（支持 GPT-NeoX/LLaMA）
  generate.py            # 统一生成脚本（NBCE/PCW + 采样率 sweep）
  evaluate.py            # 性能矩阵构建

tests/                   # 单元测试
  test_*.py              # pytest 测试用例

requirements.txt         # 依赖清单
config.json              # 运行时配置（路径、超参、提示词模板）
```

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt
```

**核心依赖**：

- `torch >= 2.1.0`（GPU 推理需 CUDA）
- `transformers >= 4.42.0`
- `peft >= 0.11.1`（LoRA 微调）
- `rouge-score >= 0.1.2`、`sacrebleu >= 2.4.0`（评测）
- `mecab-python3 >= 1.0.9`（日语分词，可选）

### 2. 配置文件

编辑 `config.json` 设置模型路径与数据目录：

```jsonc
{
  "data_paths": {
    "input": "./data/input/",
    "output": "./data/output/"
  },
  "model_paths": {
    "pretrained_model": "./models/pretrained/",
    "peft_model": "./models/peft/"
  },
  "prompt_templates": {
    "summarization": "あなたは{department}の医師です。以下の臨床記録に基づいて、{smr_type}を作成してください。"
  }
}
```

### 3. 数据预处理

`dataset_builder/datasets/data_pipeline.py` 提供可配置的流水线，将原始 EMR JSON 转为标准化指令数据集：

```bash
python -m dataset_builder.datasets.data_pipeline \
  --data-dir ./raw_emr_json/ \
  --output-dir ./data/processed/instruction/ \
  --chunk-size 500
```

或在代码中调用：

```python
from dataset_builder.datasets.data_pipeline import build_instruction_records, write_instruction_chunks

records = build_instruction_records(["./raw_emr_json/sample.json"])
write_instruction_chunks(records, "./data/processed/instruction/")
```

### 4. LoRA 微调

支持 GPT-NeoX 与 LLaMA 系列模型，可指定多个训练文件：

```bash
python scripts/train.py \
  --data-path ./data/train.json \
  --base-model ./models/pretrained/ \
  --output-dir ./models/peft/run1 \
  --lora-type gptneox \
  --epochs 3 \
  --learning-rate 1e-4 \
  --batch-size 64 \
  --micro-batch-size 1
```

**LLaMA 训练优化**（需 Flash Attention 支持）：

```bash
python scripts/train.py \
  --data-path ./data/train.json \
  --base-model /path/to/llama \
  --lora-type llama \
  --llama-attn-replace \
  --llama-attn-full
```

### 5. 生成（NBCE / PCW）

使用 `scripts/generate.py` 进行推理：

```bash
# NBCE 模式（动态采样）
python scripts/generate.py \
  --method nbce \
  --contexts "片段A,片段B,片段C" \
  --task "内科医師として以下記録を要約してください" \
  --max-new-tokens 256

# PCW 模式（分段窗口）
python scripts/generate.py \
  --method pcw \
  --contexts file:./data/contexts.txt \
  --task "要約してください" \
  --window-size 512 \
  --right-indentation
```

**批量采样率实验**（Sampling Rate Sweep）：

```bash
python scripts/generate.py \
  --method nbce \
  --contexts json:./data/contexts.json \
  --task "要約" \
  --sampling-rates 0.15:0.65:0.05 \
  --window-size 256 \
  --output-prefix result \
  --output ./out/sweep_agg.json
```

### 6. 评测

使用 `scripts/evaluate.py` 构建性能矩阵：

```bash
python scripts/evaluate.py \
  --file ./out/gen.json \
  --metrics rouge1,rougeL,bleu \
  --group-by method,sampling_rate,window_size \
  --out-json ./out/matrix.json \
  --out-csv ./out/matrix.csv
```

输入 JSON 格式示例：

```jsonc
[
  {
    "method": "nbce",
    "sampling_rate": 0.2,
    "window_size": 256,
    "ref": "参照文本",
    "output": "生成文本"
  }
]
```

## 使用示例

### 实验工作流

```bash
# 1. 准备数据
python -m dataset_builder.datasets.data_pipeline \
  --data-dir ./raw_data/ \
  --output-dir ./processed/

# 2. 微调模型
python scripts/train.py \
  --data-path ./processed/*.json \
  --base-model ./models/gptneox-base \
  --output-dir ./models/peft/medical_v1 \
  --epochs 3

# 3. 采样率扫描实验
python scripts/generate.py \
  --method nbce \
  --contexts json:./test_contexts.json \
  --task "医師として要約してください" \
  --sampling-rates 0.1:0.7:0.05 \
  --output ./results/sweep.json

# 4. 评测
python scripts/evaluate.py \
  --file ./results/sweep.json \
  --metrics rouge1,rougeL \
  --group-by sampling_rate \
  --out-csv ./results/performance.csv
```

## 技术细节

### NBCE 策略

NBCE 通过计算每个句子的信息熵与邻域密度，动态选择高信息量片段作为上下文：

1. **熵计算**：基于 token 概率分布计算每句信息熵
2. **邻域密度**：考虑前后句的语义相似度
3. **自适应采样**：根据目标采样率与熵阈值筛选关键句

### PCW 策略

PCW 将长文档分割为多个固定大小窗口，逐窗口生成并合并输出：

- 支持左对齐（默认）与右对齐（`--right-indentation`）
- 自动管理 `past_key_values` 缓存以提升效率
- 可配置窗口大小（`--window-size`）

### 配置系统

`config/loader.py` 提供统一配置接口：

```python
from config import get_config

cfg = get_config()
model_path = cfg.get("model_paths.pretrained_model")
template = cfg.get("prompt_templates.summarization")
```

支持嵌套键访问与环境变量覆盖。

## 测试

运行单元测试（需先安装 `pytest`）：

```bash
pip install pytest
pytest tests/ -v
```

## 贡献指南

欢迎提交 Issue 或 Pull Request 改进本项目：

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/your-feature`）
3. 提交变更（`git commit -m 'Add your feature'`）
4. 推送到分支（`git push origin feature/your-feature`）
5. 开启 Pull Request

## 许可证

本项目仅供学术研究使用，数据来源于京都大学医学部附属病院，未经授权不得用于商业用途。

## 联系方式

- **作者**：ZHANG Guoqing（张国庆）
- **导师**：黒田知宏 教授（京都大学）
- **邮箱**：[请通过 GitHub Issues 联系]

## 致谢

感谢京都大学医学部附属病院提供数据支持，以及福山圭太、岸本一正、江口加奈等合作者的贡献。
