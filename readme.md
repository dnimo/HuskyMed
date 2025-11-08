# HuskyMed

English | [日本語](README.ja.md) | [中文](readme.zh.md)

**HuskyMed** is an automatic summarization system for long clinical records built on Large Language Models (LLMs). It integrates multiple dynamic context extension strategies to mitigate information loss in ultra‑long Electronic Medical Record (EMR) scenarios. The project originates from the author's master's research at Kyoto University Hospital supervised by **Professor Tomohiro Kuroda**.

## Research Background

Clinical records often contain tens of thousands of tokens of unstructured text. Conventional summarization approaches struggle with (1) information loss under truncated context windows and (2) computational bottlenecks for full‑length encoding. HuskyMed implements two complementary strategies:

- **NBCE (Neighbor-Based Context Extension)**: Dynamic sampling guided by sentence information entropy and neighborhood density to retain high‑information segments.
- **PCW (Piecewise Context Window)**: Segmented window management enabling coherent long‑form generation with bounded memory usage.

Related publications and submissions:

1. **ZHANG Guoqing**, Keita Fukuyama, Kazumasa Kishimoto, Kana Eguchi, Tomohiro Kuroda (2024).  
   *Long Electronic Medical Record Summarization with Dynamic Minimize Entropy Decoding for LLM.* (Japan Association for Medical Informatics Nursing Conference)
2. **Guoqing ZHANG**, Keita Fukuyama, Kazumasa Kishimoto, Tomohiro Kuroda (2024).  
   *Mitigating Context Loss in Clinical Notes Auto-Summarization through Information Theory-Based Dynamic Decoding.* (The 44th Joint Conference on Medical Informatics)
3. Zhang, G., Fukuyama, K., Kishimoto, K., & Kuroda, T. (2024).  
   *Optimizing Automatic Summarization of Long Clinical Records Using Dynamic Context Extension: Testing and Evaluation of the NBCE Method.*  
   arXiv preprint: [arXiv:2411.08586](https://arxiv.org/abs/2411.08586)

## Core Features

- Multiple context extension strategies (NBCE / PCW) with a unified API
- Parameter‑efficient instruction fine‑tuning (LoRA) for GPT‑NeoX & LLaMA families
- Complete evaluation pipeline (ROUGE / BLEU / custom metrics) with aggregation
- Modular, configuration‑driven architecture for reproducibility & extension

## Project Structure

```text
config/                   # Configuration management (paths, hyperparams, templates)
  loader.py               # Unified config loader

dataset_builder/
  datasets/
    build_dataset.py      # Instruction dataset build (tokenization + TF-IDF alignment)
    data_pipeline.py      # Raw EMR JSON -> standardized instruction records

model/
  base_loader.py          # Base + LoRA unified loader
  wrappers/               # Context extension implementations
    generation.py         # NBCE / PCW core logic
    generation_manager.py # Strategy manager facade
    attention.py          # Flash Attention replacement hooks
    constants.py          # Prompt templates & constants
    utils.py              # Sampling sweep utilities / long text filtering

evaluation/
  rouge.py                # ROUGE metrics (MeCab support optional)
  bleu.py                 # SacreBLEU metrics
  aggregator.py           # Batch scoring & grouped aggregation

scripts/
  train.py                # LoRA fine-tuning entry
  generate.py             # Unified generation (NBCE / PCW + sweeps)
  evaluate.py             # Performance matrix construction

tests/                    # Pytest unit tests
requirements.txt          # Dependency list
config.json               # Runtime config (paths, hyperparams, prompts)
```

## Quick Start

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Core dependencies:

- `torch >= 2.1.0` (CUDA recommended for GPU inference)
- `transformers >= 4.42.0`
- `peft >= 0.11.1` (LoRA fine-tuning)
- `rouge-score >= 0.1.2`, `sacrebleu >= 2.4.0` (evaluation)
- `mecab-python3 >= 1.0.9` (optional Japanese tokenization)

### 2. Configuration File

Edit `config.json` to set paths and templates:

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
    "summarization": "You are a physician in {department}. Please create {smr_type} based on the following clinical records."
  }
}
```

### 3. Data Preprocessing

Transform raw EMR JSON into standardized instruction records:

```bash
python -m dataset_builder.datasets.data_pipeline \
  --data-dir ./raw_emr_json/ \
  --output-dir ./data/processed/instruction/ \
  --chunk-size 500
```

Programmatic usage:

```python
from dataset_builder.datasets.data_pipeline import build_instruction_records, write_instruction_chunks

records = build_instruction_records(["./raw_emr_json/sample.json"])
write_instruction_chunks(records, "./data/processed/instruction/")
```

### 4. LoRA Fine-tuning

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

Flash Attention optimization for LLaMA:

```bash
python scripts/train.py \
  --data-path ./data/train.json \
  --base-model /path/to/llama \
  --lora-type llama \
  --llama-attn-replace \
  --llama-attn-full
```

### 5. Generation (NBCE / PCW)

```bash
# NBCE mode (dynamic sampling)
python scripts/generate.py \
  --method nbce \
  --contexts "Segment A,Segment B,Segment C" \
  --task "Please summarize the following medical records" \
  --max-new-tokens 256

# PCW mode (piecewise window)
python scripts/generate.py \
  --method pcw \
  --contexts file:./data/contexts.txt \
  --task "Summarize" \
  --window-size 512 \
  --right-indentation

# Sampling rate sweep
python scripts/generate.py \
  --method nbce \
  --contexts json:./data/contexts.json \
  --task "Summarize" \
  --sampling-rates 0.15:0.65:0.05 \
  --window-size 256 \
  --output-prefix result \
  --output ./out/sweep_agg.json
```

### 6. Evaluation

```bash
python scripts/evaluate.py \
  --file ./out/gen.json \
  --metrics rouge1,rougeL,bleu \
  --group-by method,sampling_rate,window_size \
  --out-json ./out/matrix.json \
  --out-csv ./out/matrix.csv
```

Input JSON example:

```jsonc
[
  {
    "method": "nbce",
    "sampling_rate": 0.2,
    "window_size": 256,
    "ref": "Reference text",
    "output": "Generated text"
  }
]
```

### 7. Example Workflow

```bash
# 1. Prepare data
python -m dataset_builder.datasets.data_pipeline \
  --data-dir ./raw_data/ \
  --output-dir ./processed/

# 2. Fine-tune model
python scripts/train.py \
  --data-path ./processed/*.json \
  --base-model ./models/gptneox-base \
  --output-dir ./models/peft/medical_v1 \
  --epochs 3

# 3. Sampling rate sweep
python scripts/generate.py \
  --method nbce \
  --contexts json:./test_contexts.json \
  --task "Summarize as a physician" \
  --sampling-rates 0.1:0.7:0.05 \
  --output ./results/sweep.json

# 4. Evaluate
python scripts/evaluate.py \
  --file ./results/sweep.json \
  --metrics rouge1,rougeL \
  --group-by sampling_rate \
  --out-csv ./results/performance.csv
```

## Technical Details

### NBCE Strategy

1. Entropy calculation per sentence (token probability distribution)
2. Neighborhood density via semantic similarity of adjacent sentences
3. Adaptive sampling selecting key sentences under target sampling rate & entropy threshold

### PCW Strategy

Splits long documents into fixed-size windows and generates per window while

- Supporting left alignment (default) & right alignment (`--right-indentation`)
- Managing `past_key_values` cache for efficiency
- Allowing configurable `--window-size`


### Configuration System

```python
from config import get_config

cfg = get_config()
model_path = cfg.get("model_paths.pretrained_model")
template = cfg.get("prompt_templates.summarization")
```

Nested key access & environment overrides supported.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Contributing

Contributions welcome:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add your feature'`
4. Push branch: `git push origin feature/your-feature`
5. Open a Pull Request

## License

Academic research use only. Data sourced from Kyoto University Hospital. Commercial use requires explicit authorization.

## Contact

- Author: ZHANG Guoqing
- Supervisor: Professor Tomohiro Kuroda (Kyoto University)
- Email: (please open a GitHub Issue)

## Acknowledgments

Thanks to Kyoto University Hospital for data support and collaborators Keita Fukuyama, Kazumasa Kishimoto, Kana Eguchi, and others for their contributions.
