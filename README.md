# CHEMDNER Evaluation with LLMs

BioCreative IV CHEMDNERタスクを現代のLLMで評価するプロジェクト。

## タスク概要

CHEMDNER (Chemical Document Named Entity Recognition) は、PubMedアブストラクトから化学物質名を抽出するNERタスク。

- **データセット**: 10,000 PubMed abstracts, 84,355 chemical entity mentions
- **評価対象モデル**: GPT-4.1, GPT-5.2

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
