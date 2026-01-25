# CHEMDNER NER実験サマリー

## 背景と目的

### 背景
- **CHEMDNER**: BioCreative IV (2013) のベンチマーク。PubMed論文から化学物質名を抽出するタスク
- **2013年SOTA**: F1=87.39%（tmChem: CRF + 辞書 + ルール）
- **現在のSOTA**: F1=94.30%（MatSciBERT-MRC, 2024年）

### 仮説
「GPT-5.2などの最新LLMなら、fine-tuningなしでも古典的なCRF手法を超えられるのではないか？」

### 目的
1. LLMのゼロショット/few-shot能力でどこまで迫れるか検証
2. 各種プロンプト技法の効果を比較
3. fine-tuningとの差分を明確化

## 実験タイムライン

| 日付 | 実験 | 所要時間 |
|------|------|----------|
| 2026-01-25 午前 | Exp001-007 (LLMアプローチ) | 約4時間 |
| 2026-01-25 午後 | Exp008 (BioBERT fine-tuning) | 約1時間 |

## 実験結果一覧

| # | 手法 | Precision | Recall | F1 | 変化 | コスト | 時間 |
|---|------|-----------|--------|-----|------|--------|------|
| Exp001 | GPT-5.2 Few-shot (4例) | 73.07% | 70.60% | 71.82% | baseline | $12.42 | 3分 |
| Exp002 | Responses API | 72.56% | 70.62% | 71.58% | -0.24pt | $12.42 | 3分 |
| Exp003 | Chain-of-Thought | 77.03% | 64.01% | 69.92% | -1.90pt | $15 | 4分 |
| Exp004 | Few-shot 8例 | 70.64% | 72.05% | 71.34% | -0.48pt | $14.57 | 4分 |
| Exp005 | Self-verify | 80.77% | 69.17% | 74.52% | +2.70pt | $17 | 3分 |
| Exp006a | MRC | 60.91% | 78.41% | 68.56% | -3.26pt | $12.42 | 3分 |
| Exp006b | MRC + verify | 75.59% | 76.33% | **75.96%** | +4.14pt | $17 | 3分 |
| Exp007 | GLiNER-BioMed | 36.28% | 52.67% | 42.97% | -28.9pt | $0 | 1分 |
| **Exp008** | **BioBERT fine-tuning** | **90.85%** | **91.35%** | **91.10%** | **+15.14pt** | **~$2** | **25分** |

## 精度改善の経緯

```
Zero-shot (初期)
├── GPT-4.1: F1=46.33%
└── GPT-5.2: F1=56.89%
    │
    │ [Few-shot追加: +14.9pt]
    ↓
Few-shot 4例 (Exp001): F1=71.82%
    │
    │ [Self-verification: +2.7pt]
    ↓
Self-verify (Exp005): F1=74.52%
    │
    │ [MRC形式 + verify: +1.4pt]
    ↓
MRC + verify (Exp006b): F1=75.96%  ← LLM最高値
    │
    │ [BioBERT fine-tuning: +15.1pt]
    ↓
BioBERT (Exp008): F1=91.10%  ← 最終結果
```

## SOTAとの比較

| 手法 | F1 | 備考 |
|------|-----|------|
| **MatSciBERT-MRC (2024)** | **94.30%** | 現在のSOTA |
| **BioBERT fine-tuning (本実験)** | **91.10%** | **SOTAまで3.2pt** |
| BioBERT-MRC (2020) | 92.92% | - |
| 2013年1位 (tmChem) | 87.39% | CRF + 辞書 + ルール |
| GPT-5.2 MRC + verify | 75.96% | LLM最高値 |
| GPT-5.2 Few-shot | 71.82% | LLMベースライン |
| GPT-5.2 Zero-shot | 56.89% | 初期値 |

## コスト分析

### LLMアプローチ（Exp001-007）

| 項目 | コスト |
|------|--------|
| OpenAI API | 約$70 |
| 実行時間 | 約20分 |
| 最高F1 | 75.96% |

### Fine-tuningアプローチ（Exp008）

| 項目 | コスト |
|------|--------|
| GCP A100 VM | ~$2 |
| 学習時間 | 25分 |
| 最高F1 | **91.10%** |

### コストパフォーマンス比較

| アプローチ | コスト | F1 | コスト効率 |
|------------|--------|-----|------------|
| LLM (GPT-5.2) | $70+ | 75.96% | 低 |
| **Fine-tuning (BioBERT)** | **$2** | **91.10%** | **非常に高** |

## 主要な発見

### 1. LLMアプローチの限界

| 施策 | 効果 | 理由 |
|------|------|------|
| Few-shot (4例) | +14.9pt | 境界ケース（タンパク質 vs 化学物質）を明示 |
| Self-verification | +2.7pt | 抽出と検証を分離、FP削減に特化 |
| MRC + verify | +4.1pt | タスク形式の変換 + 検証の組み合わせ |

**壁**: 試した範囲では**F1=76%前後で頭打ち**

### 2. Fine-tuningの圧倒的優位性

- **精度**: 91.10% vs 75.96%（+15pt）
- **コスト**: $2 vs $70+（1/35以下）
- **時間**: 25分 vs 20分（同程度）

### 3. 技術的知見

1. **NERは認識タスク、推論タスクではない**
   - CoTのような「考えさせる」手法は逆効果
   - パターンマッチングが本質

2. **ドメイン特化の重要性**
   - BioBERTの事前学習（PubMed）が効果的
   - 汎用LLMは化学物質の境界判定が苦手

3. **データ量 > プロンプト工夫**
   - Few-shot数例より、full datasetでのfine-tuningが圧倒的

## 最大の学び

> **「OpenAI APIだけでもっと解けると思ったが、ドメイン特化タスクではfine-tuningがまだまだ有効」**

### なぜLLMは限界があったか

1. **化学物質名の特殊性**
   - 「docosahexaenoic acid」は化学物質
   - 「insulin」はタンパク質（化学物質ではない）
   - この境界はドメイン知識が必要

2. **Few-shotの限界**
   - 数例では多様なパターンをカバーできない
   - CHEMDNER training setには8,000+の化学物質名が含まれる

3. **推論 vs パターン認識**
   - LLMは「考える」ことで判断
   - NERは「見たことがあるか」で判断する方が正確

### Fine-tuningが有効な理由

1. **大量のパターン学習**
   - 8,000+の化学物質名を直接学習
   - 境界ケースも含めて網羅

2. **ドメイン特化**
   - BioBERTは既にバイオ医学テキストを理解
   - 追加学習で化学物質認識に特化

3. **低コスト・高効率**
   - A100 GPUで25分、$2以下
   - 一度学習すれば推論は高速・低コスト

## 結論

| 指標 | LLM (GPT-5.2) | Fine-tuning (BioBERT) | 勝者 |
|------|---------------|----------------------|------|
| F1スコア | 75.96% | **91.10%** | Fine-tuning |
| コスト | $70+ | **~$2** | Fine-tuning |
| 実装難易度 | 低 | 中 | LLM |
| 再現性 | 中（API依存） | **高** | Fine-tuning |

**結論**: ドメイン特化NERタスクでは、LLMのプロンプトエンジニアリングより、
**小規模言語モデルのfine-tuningの方が精度・コストの両面で優れている**。

## ファイル構造

```
experiments/
├── summary.md          # このファイル
├── exp001.md          # Few-shot 4例
├── exp002.md          # Responses API
├── exp003.md          # Chain-of-Thought
├── exp004.md          # Few-shot 8例
├── exp005.md          # Self-verification
├── exp006.md          # MRC形式
├── exp007.md          # GLiNER
└── exp008.md          # BioBERT fine-tuning

outputs/
├── exp001_fewshot4_evaluation/
├── exp005_verify/
├── exp006_mrc/
├── exp006b_mrc_verify/
├── exp007_gliner/
└── exp008_biobert/
    ├── predictions_biobert.json
    ├── training_metrics.json
    └── model/  (GCP上、削除済み)

infrastructure/
├── main.tf
├── variables.tf
└── ...

src/biobert/
├── train.py
├── inference.py
├── data_prep.py
└── config.py
```

## 今後の可能性

1. **SOTAへの追い込み**
   - PubMedBERTやMatSciBERTの試行
   - CRFレイヤーの追加
   - ハイパーパラメータチューニング

2. **LLM + Fine-tunedモデルのハイブリッド**
   - BioBERTで抽出 → GPTで検証
   - 高精度かつ説明可能な出力

3. **他ドメインへの適用**
   - 同じワークフロー（Terraform + BioBERT）を他のNERタスクに

---

*実験期間: 2026-01-25*
*総コスト: 約$72（LLM $70 + GCP $2）*
*最終F1: 91.10%（2013年SOTA超え、現行SOTAまで3.2pt）*
