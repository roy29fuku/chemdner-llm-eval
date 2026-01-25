# CHEMDNER Evaluation with LLMs

BioCreative IV CHEMDNERタスクを現代のLLMで評価するプロジェクト。

## タスク概要

CHEMDNER (Chemical Document Named Entity Recognition) は、PubMedアブストラクトから化学物質名を抽出するNERタスク（2013年開催）。

- **データセット**: 10,000 PubMed abstracts, 84,355 chemical entity mentions
- **評価対象モデル**: GPT-4.1, GPT-5.2

### ベンチマークスコア

| 年 | 手法 | F1 | 備考 |
|----|------|-----|------|
| 2013 | tmChem (CRF + 辞書 + ルール) | 87.39% | BioCreative IV 1位 |
| 2019 | [BioBERT](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506) | 89.7% | Transformer初期 |
| 2020 | [BioBERT-MRC](https://ar5iv.labs.arxiv.org/html/2009.01560) | 92.92% | NERをMRC問題に変換 |
| 2022 | [ChemDataExtractor 2.1](https://pubs.acs.org/doi/full/10.1021/acs.jcim.1c01199) | 89.7% | 有機・無機統合モデル |
| 2024 | [MatSciBERT-MRC](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00874-5) | **94.30%** | 現在のSOTA |

**理論上限**: 約91%（人間のアノテーター間一致率）

## データセット

### 公式データ分割

CHEMDNERコーパスは以下の3つに分割されている：

| Split | 用途 | 本プロジェクトでの使用 |
|-------|------|----------------------|
| training | モデル学習用 | Few-shot例の取得元 |
| development | ハイパーパラメータ調整用 | 未使用 |
| evaluation | 最終評価用 | **推論・評価対象** |

### データ形式

BIO形式（B=Begin, I=Inside, O=Outside）でアノテーションされたトークン列。
本プロジェクトでは10文ごとに1ドキュメントとして結合して処理。

### データソース

公式サイトからのダウンロードが困難なため、GitHub上のミラー（[lingluodlut/Att-ChemdNER](https://github.com/lingluodlut/Att-ChemdNER)）からBIO形式データを取得。

## メソドロジー

### プロンプト設計

Few-shot promptingを採用。CHEMDNERアノテーションガイドラインに基づいた詳細なシステムプロンプトを使用。

**重要**: Few-shot例は**trainingセットから取得**（評価対象のevaluationセットからのデータリークを防止）

#### 含めるべきエンティティ
- 化学物質名（trivial names）: aspirin, caffeine
- IUPAC名（systematic names）: 2-acetoxybenzoic acid
- 化学略語: ATP, ADP, NAD+
- 分子式: H2O, C6H12O6, NaCl
- 医薬品名: saquinavir, metformin

#### 除外すべきエンティティ
- タンパク質・酵素: CYP3A4, kinase
- 遺伝子: BRCA1, eNOS
- 一般的な薬物クラス: "drugs", "medications"
- トランスポーター: OATP, MCT1

### Few-shot例

以下の4例をtrainingセットから抽出：

1. **haloperidol例**: 薬物名を抽出、タンパク質（Dpp6, Kv4）は除外
2. **chromium/malondialdehyde例**: 元素と化合物を抽出
3. **PAH例**: 化学物質ファミリーと略語を抽出、酵素（catalase, peroxidase）は除外
4. **RAP例**: 薬物略語を抽出、キナーゼ等は除外

### 並列処理

OpenAI APIへのリクエストを並列化（デフォルト: 10並列、Tier 5では50並列推奨）。
asyncio + semaphoreによるレート制限を実装。

### コスト見積もり

推論実行前にtiktokenを使用してトークン数を推定し、モデルごとのコストを表示。
ユーザー確認（y/n）後に実行。

## セットアップ

```bash
# 環境変数の設定
cp .env.example .env
# .envにOPENAI_API_KEYを設定

# Dockerで実行
docker compose up --build
```

## ディレクトリ構造

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── data_loader.py    # データセット読み込み
│   ├── inference.py      # LLM推論
│   ├── evaluate.py       # 評価スクリプト
│   └── biobert/          # BioBERTモジュール
│       ├── config.py     # ハイパーパラメータ
│       ├── data_prep.py  # データ変換
│       ├── train.py      # 学習スクリプト
│       └── inference.py  # 推論スクリプト
├── infrastructure/       # Terraform設定（GCP）
├── scripts/              # 実行スクリプト
├── outputs/              # 結果出力
├── experiments/          # 実験レポート
└── README.md
```

## 実験管理

### ディレクトリ構造

```
outputs/
├── exp001_fewshot4_evaluation/   # 実験ごとにディレクトリを分離
│   ├── predictions_gpt-4_1.json
│   └── predictions_gpt-5_2.json
└── exp002_fewshot8_evaluation/   # 次の実験...

experiments/
├── exp001.md                     # 実験レポート
└── exp002.md
```

### 命名規則

- **出力ディレクトリ**: `outputs/exp{番号}_{説明}/`
- **実験レポート**: `experiments/exp{番号}.md`

### 実験レポートに含める内容

1. 実験設定（モデル、パラメータ、Few-shot例数等）
2. 結果（Precision, Recall, F1）
3. 考察
4. 再現用コマンド
5. コスト

## 実験結果

### サマリー（2026-01-25）

#### LLMアプローチ（プロンプトエンジニアリング）

| # | 手法 | Precision | Recall | F1 | SOTAとの差 |
|---|------|-----------|--------|-----|-----------|
| Exp001 | Few-shot (4例) | 0.7307 | 0.7060 | 0.7182 | -22.5pt |
| Exp002 | Responses API | 0.7256 | 0.7062 | 0.7158 | -22.7pt |
| Exp003 | Chain-of-Thought | 0.7703 | 0.6401 | 0.6992 | -24.4pt |
| Exp004 | Few-shot (8例) | 0.7064 | 0.7205 | 0.7134 | -22.9pt |
| Exp005 | Self-verify | 0.8077 | 0.6917 | 0.7452 | -19.8pt |
| Exp006a | MRC | 0.6091 | 0.7841 | 0.6856 | -25.7pt |
| Exp006b | MRC + verify | 0.7559 | 0.7633 | 0.7596 | -18.3pt |
| Exp007 | GLiNER-BioMed | 0.3628 | 0.5267 | 0.4297 | -50.0pt |

**LLM最高**: GPT-5.2 + MRC + Self-verification = **F1 75.96%**

#### Fine-tuningアプローチ

| # | 手法 | Precision | Recall | F1 | SOTAとの差 |
|---|------|-----------|--------|-----|-----------|
| **Exp008** | **BioBERT fine-tuning** | **0.9085** | **0.9135** | **0.9110** | **-3.2pt** |

**プロジェクト最高**: BioBERT v1.1 fine-tuning = **F1 91.10%**

- ベースモデル: `dmis-lab/biobert-v1.1`
- インフラ: GCP A100 GPU（Terraform管理）
- 学習時間: 約25分
- コスト: 約$2

詳細: [experiments/summary.md](experiments/summary.md)

## 実行方法

```bash
# 5件でテスト
docker compose run --rm chemdner python src/inference.py --limit 5

# 全件実行（実験ごとに出力先を指定）
docker compose run --rm chemdner python src/inference.py \
  --parallel 50 \
  --models gpt-4.1 gpt-5.2 \
  --output-dir outputs/exp001_fewshot4_evaluation \
  -y

# 評価
docker compose run --rm chemdner python src/evaluate.py \
  --input-dir outputs/exp001_fewshot4_evaluation
```

## 参考文献

- [CHEMDNER Corpus Paper](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S2)
- [Hugging Face Dataset](https://huggingface.co/datasets/bigbio/chemdner)


