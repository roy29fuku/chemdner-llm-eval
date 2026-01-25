# CHEMDNER Evaluation with LLMs

BioCreative IV CHEMDNERタスクを現代のLLMで評価するプロジェクト。

## タスク概要

CHEMDNER (Chemical Document Named Entity Recognition) は、PubMedアブストラクトから化学物質名を抽出するNERタスク（2013年開催）。

- **データセット**: 10,000 PubMed abstracts, 84,355 chemical entity mentions
- **評価対象モデル**: GPT-4.1, GPT-5.2
- **2013年最高スコア**: F1 = 87.39

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
│   └── evaluate.py       # 評価スクリプト
├── outputs/              # 結果出力
└── README.md
```

## 実行方法

```bash
# 5件でテスト
docker compose run --rm chemdner python src/inference.py --limit 5

# 全件実行
docker compose run --rm chemdner python src/inference.py

# 評価
docker compose run --rm chemdner python src/evaluate.py
```

## 参考文献

- [CHEMDNER Corpus Paper](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-7-S1-S2)
- [Hugging Face Dataset](https://huggingface.co/datasets/bigbio/chemdner)


