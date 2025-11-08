# HuskyMed

[English](README.md) | 日本語 | [中文](readme.zh.md)

**HuskyMed** は、大規模言語モデル（LLM）に基づく長文臨床記録自動要約システムであり、超長文電子カルテ（EMR）における情報損失を軽減するための複数の動的コンテキスト拡張戦略を統合しています。本プロジェクトは、京都大学医学部附属病院における著者の修士研究に基づき、**黒田知宏** 教授の指導のもとで完成されました。

## 研究背景

臨床記録は数万文字の非構造化テキストを含むことが多く、従来の要約手法は超長コンテキストの処理において情報損失と計算リソースのボトルネックに直面しています。本プロジェクトでは以下を提案・実装しました：

- **NBCE（Neighbor-Based Context Extension）**：情報エントロピーに基づく動的サンプリング戦略で、近隣密度と情報利得によって重要な文章セグメントを選択
- **PCW（Piecewise Context Window）**：分割ウィンドウコンテキスト管理により、長文書推論時の一貫性と計算効率を保証

関連研究成果は以下の学会・論文誌に発表または投稿されています：

1. **ZHANG Guoqing**, Keita Fukuyama, Kazumasa Kishimoto, Kana Eguchi, Tomohiro Kuroda (2024).  
   *Long Electronic Medical Record Summarization with Dynamic Minimize Entropy Decoding for LLM.*  
   日本医療情報学会看護学術大会.

2. **Guoqing ZHANG**, Keita Fukuyama, Kazumasa Kishimoto, Tomohiro Kuroda (2024).  
   *Mitigating Context Loss in Clinical Notes Auto-Summarization through Information Theory-Based Dynamic Decoding.*  
   第44回医療情報学連合大会.

3. Zhang, G., Fukuyama, K., Kishimoto, K., & Kuroda, T. (2024).  
   *Optimizing Automatic Summarization of Long Clinical Records Using Dynamic Context Extension: Testing and Evaluation of the NBCE Method.*  
   arXiv preprint: [arXiv:2411.08586](https://arxiv.org/abs/2411.08586)

## 主な特徴

- **複数のコンテキスト拡張戦略**：NBCEとPCWの2つの生成モードをサポートし、統一APIで切り替え可能
- **パラメータ効率的なファインチューニング**：LoRAベースの命令ファインチューニングプロセスで、GPT-NeoXとLLaMAシリーズのモデルをサポート
- **完全な評価パイプライン**：ROUGE/BLEU/カスタムメトリクスを統合し、バッチ実験とパフォーマンスマトリクス集約をサポート
- **エンジニアリングアーキテクチャ**：モジュール設計、設定とコードの分離、拡張と再現が容易

## プロジェクト構造

```text
config/                   # 設定管理（パス、ハイパーパラメータ、テンプレート）
  loader.py              # 統一設定ローダー

dataset_builder/         # データ前処理と命令構築
  datasets/
    build_dataset.py     # 命令データセット生成（トークン化 + TF-IDF アライメント）
    data_pipeline.py     # EMR 生データ → 標準化フォーマットパイプライン

model/                   # モデル読み込みと推論
  base_loader.py         # 基本モデル + LoRA 統一読み込み
  wrappers/              # コンテキスト拡張戦略実装
    generation.py        # NBCE / PCW コアロジック
    generation_manager.py # 戦略マネージャー
    attention.py         # Flash Attention 置換（訓練最適化）
    constants.py         # プロンプトテンプレートと定数
    utils.py             # サンプリングレート sweep、長文フィルタリングツール

evaluation/              # 評価メトリクスと集約
  rouge.py               # ROUGE メトリクス（MeCab トークン化対応）
  bleu.py                # BLEU メトリクス（SacreBLEU）
  aggregator.py          # バッチスコアリングとグループ集約エンジン

scripts/                 # コマンドラインエントリーポイント
  train.py               # LoRA 命令ファインチューニング（GPT-NeoX/LLaMA 対応）
  generate.py            # 統一生成スクリプト（NBCE/PCW + サンプリングレート sweep）
  evaluate.py            # パフォーマンスマトリクス構築

tests/                   # ユニットテスト
  test_*.py              # pytest テストケース

requirements.txt         # 依存関係リスト
config.json              # ランタイム設定（パス、ハイパーパラメータ、プロンプトテンプレート）
```

## クイックスタート

### 1. 環境設定

```bash
# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

**コア依存関係**：

- `torch >= 2.1.0`（GPU推論にはCUDAが必要）
- `transformers >= 4.42.0`
- `peft >= 0.11.1`（LoRAファインチューニング）
- `rouge-score >= 0.1.2`、`sacrebleu >= 2.4.0`（評価）
- `mecab-python3 >= 1.0.9`（日本語トークン化、オプション）

### 2. 設定ファイル

`config.json` を編集してモデルパスとデータディレクトリを設定：

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

### 3. データ前処理

`dataset_builder/datasets/data_pipeline.py` は設定可能なパイプラインを提供し、生EMR JSONを標準化された命令データセットに変換：

```bash
python -m dataset_builder.datasets.data_pipeline \
  --data-dir ./raw_emr_json/ \
  --output-dir ./data/processed/instruction/ \
  --chunk-size 500
```

またはコード内で呼び出し：

```python
from dataset_builder.datasets.data_pipeline import build_instruction_records, write_instruction_chunks

records = build_instruction_records(["./raw_emr_json/sample.json"])
write_instruction_chunks(records, "./data/processed/instruction/")
```

### 4. LoRAファインチューニング

GPT-NeoXとLLaMAシリーズのモデルをサポートし、複数のトレーニングファイルを指定可能：

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

**LLaMA訓練最適化**（Flash Attentionサポートが必要）：

```bash
python scripts/train.py \
  --data-path ./data/train.json \
  --base-model /path/to/llama \
  --lora-type llama \
  --llama-attn-replace \
  --llama-attn-full
```

### 5. 生成（NBCE / PCW）

`scripts/generate.py` を使用して推論を実行：

```bash
# NBCEモード（動的サンプリング）
python scripts/generate.py \
  --method nbce \
  --contexts "セグメントA,セグメントB,セグメントC" \
  --task "内科医師として以下記録を要約してください" \
  --max-new-tokens 256

# PCWモード（分割ウィンドウ）
python scripts/generate.py \
  --method pcw \
  --contexts file:./data/contexts.txt \
  --task "要約してください" \
  --window-size 512 \
  --right-indentation
```

**バッチサンプリングレート実験**（Sampling Rate Sweep）：

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

### 6. 評価

`scripts/evaluate.py` を使用してパフォーマンスマトリクスを構築：

```bash
python scripts/evaluate.py \
  --file ./out/gen.json \
  --metrics rouge1,rougeL,bleu \
  --group-by method,sampling_rate,window_size \
  --out-json ./out/matrix.json \
  --out-csv ./out/matrix.csv
```

入力JSON形式の例：

```jsonc
[
  {
    "method": "nbce",
    "sampling_rate": 0.2,
    "window_size": 256,
    "ref": "参照テキスト",
    "output": "生成テキスト"
  }
]
```

## 使用例

### 実験ワークフロー

```bash
# 1. データ準備
python -m dataset_builder.datasets.data_pipeline \
  --data-dir ./raw_data/ \
  --output-dir ./processed/

# 2. モデルのファインチューニング
python scripts/train.py \
  --data-path ./processed/*.json \
  --base-model ./models/gptneox-base \
  --output-dir ./models/peft/medical_v1 \
  --epochs 3

# 3. サンプリングレートスキャン実験
python scripts/generate.py \
  --method nbce \
  --contexts json:./test_contexts.json \
  --task "医師として要約してください" \
  --sampling-rates 0.1:0.7:0.05 \
  --output ./results/sweep.json

# 4. 評価
python scripts/evaluate.py \
  --file ./results/sweep.json \
  --metrics rouge1,rougeL \
  --group-by sampling_rate \
  --out-csv ./results/performance.csv
```

## 技術詳細

### NBCE戦略

NBCEは各文の情報エントロピーと近隣密度を計算し、高情報量のセグメントをコンテキストとして動的に選択：

1. **エントロピー計算**：トークン確率分布に基づいて各文の情報エントロピーを計算
2. **近隣密度**：前後の文の意味的類似度を考慮
3. **適応的サンプリング**：目標サンプリングレートとエントロピー閾値に基づいて重要文を選択

### PCW戦略

PCWは長文書を複数の固定サイズウィンドウに分割し、ウィンドウごとに生成して出力を結合：

- 左揃え（デフォルト）と右揃え（`--right-indentation`）をサポート
- `past_key_values` キャッシュを自動管理して効率を向上
- ウィンドウサイズを設定可能（`--window-size`）

### 設定システム

`config/loader.py` は統一された設定インターフェースを提供：

```python
from config import get_config

cfg = get_config()
model_path = cfg.get("model_paths.pretrained_model")
template = cfg.get("prompt_templates.summarization")
```

ネストされたキーアクセスと環境変数オーバーライドをサポート。

## テスト

ユニットテストを実行（`pytest` の事前インストールが必要）：

```bash
pip install pytest
pytest tests/ -v
```

## 貢献ガイドライン

IssueやPull Requestの提出を歓迎します：

1. このリポジトリをフォーク
2. フィーチャーブランチを作成（`git checkout -b feature/your-feature`）
3. 変更をコミット（`git commit -m 'Add your feature'`）
4. ブランチにプッシュ（`git push origin feature/your-feature`）
5. Pull Requestを開く

## ライセンス

本プロジェクトは学術研究目的のみで使用可能です。データは京都大学医学部附属病院から提供されており、許可なく商用利用することはできません。

## 連絡先

- **著者**：ZHANG Guoqing（張国慶）
- **指導教員**：黒田知宏 教授（京都大学）
- **メール**：[GitHub Issues経由でご連絡ください]

## 謝辞

京都大学医学部附属病院のデータサポート、および福山圭太、岸本一正、江口加奈などの共同研究者の貢献に感謝いたします。
