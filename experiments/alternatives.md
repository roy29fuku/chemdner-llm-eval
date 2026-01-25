# CHEMDNER精度向上：LLM以外の選択肢

## 現状の限界

| 手法 | F1 | SOTAとの差 |
|------|-----|-----------|
| GPT-5.2 + MRC + verify | 75.96% | -18.3pt |
| **SOTA (MatSciBERT-MRC)** | **94.30%** | - |

**結論**: プロンプトエンジニアリングだけではSOTAに届かない

## 選択肢の比較

### 1. エンコーダモデル（BioBERT/PubMedBERT）★★★推奨

| 項目 | 内容 |
|------|------|
| **期待F1** | 89-93%（ファインチューニング時） |
| **コスト** | 初期: GPUコスト（A100で数時間） / 推論: ほぼ無料 |
| **実装難易度** | 中（Hugging Faceで実装可能） |
| **メリット** | 高精度、低推論コスト、オフライン実行可能 |
| **デメリット** | GPU必要、初期学習コスト |

**参考資料**:
- [BioBERT公式](https://github.com/dmis-lab/biobert) - PubMed事前学習済み
- [Fine-tuning BioBERT for Custom NER](https://medium.com/@maneyogesh065/fine-tuning-biobert-for-custom-named-entity-recognition-a-complete-guide-a05b124edda0)
- [Biomedical-NER-Fine-Tuned-BERT](https://github.com/nirmal2i43a5/Biomedical-NER-Fine-Tuned-BERT)

### 2. GLiNER-BioMed ★☆☆ 非推奨

| 項目 | 内容 |
|------|------|
| **期待F1** | ~~79-83%~~ → **実測 43%** |
| **コスト** | 推論: ほぼ無料 / 追加学習不要 |
| **実装難易度** | 低（Hugging Faceからダウンロード） |
| **メリット** | ゼロショット対応、軽量、即座に使える |
| **デメリット** | **CHEMDNERには不適合**。タンパク質・遺伝子を化学物質と誤認識 |

**⚠️ 実験結果 (Exp007)**: F1=0.43。タンパク質（cyp3a4, enos）やサイトカイン（interleukin）を大量に誤抽出。CHEMDNERの「化学構造が決定可能なもの」という定義に対応できない。

**参考資料**:
- [GLiNER-BioMed論文](https://arxiv.org/html/2504.00676v2)
- [Hugging Face: gliner-biomed-base](https://huggingface.co/Ihor/gliner-biomed-base-v1.0)
- [BioASQ 2025での実績](https://arxiv.org/html/2510.08588) - 辞書後処理で0.79→0.83

### 3. scispaCy ★★☆

| 項目 | 内容 |
|------|------|
| **期待F1** | 76-80%（BC4CHEMDで学習済み） |
| **コスト** | 推論: 無料 |
| **実装難易度** | 低（pip install） |
| **メリット** | 簡単、高速、spaCyエコシステム |
| **デメリット** | Transformerより精度低い |

**参考資料**:
- [scispaCy公式](https://allenai.github.io/scispacy/)
- [GitHub: allenai/scispacy](https://github.com/allenai/scispacy)

### 4. GPT-4.1-mini ファインチューニング ★★☆

| 項目 | 内容 |
|------|------|
| **期待F1** | 80-85%（推定） |
| **コスト** | 学習: 約$50-100 / 推論: $0.30-1.20/1M tokens |
| **実装難易度** | 低（OpenAI API） |
| **メリット** | 簡単、プロンプトより高精度 |
| **デメリット** | 推論コストが高い、API依存 |

**参考資料**:
- [OpenAI Fine-tuning Pricing](https://platform.openai.com/docs/pricing)
- GPT-4.1-mini: $3.00/1M tokens (学習)

### 5. ハイブリッド（GLiNER + LLM）★★☆

| 項目 | 内容 |
|------|------|
| **期待F1** | 85%+（推定） |
| **コスト** | 中（GLiNER推論 + LLM検証） |
| **実装難易度** | 高 |
| **メリット** | 各モデルの強みを活かせる |
| **デメリット** | 複雑、調整が必要 |

## 推奨アプローチ

### Phase 1: 即座に試せるもの（1日以内）

1. **GLiNER-BioMed**
   - Hugging Faceからダウンロードして評価
   - 辞書後処理を追加
   - 期待: F1 79-83%

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("Ihor/gliner-biomed-base-v1.0")
```

2. **scispaCy**
   - BC4CHEMD学習済みモデルで評価
   - 期待: F1 76-80%

```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

### Phase 2: 本格的な精度向上（1週間）

3. **BioBERT/PubMedBERT ファインチューニング**
   - CHEMDNER training setで学習
   - 期待: F1 89-93%
   - 必要: GPU（Google Colab Pro可）

### Phase 3: 最高精度を目指す

4. **MRC形式 + PubMedBERT**
   - SOTAと同じアプローチ
   - 期待: F1 92%+

## コスト比較

| 手法 | 初期コスト | 推論コスト (2,646件) | 精度 |
|------|-----------|---------------------|------|
| GPT-5.2 MRC+verify | $0 | $17 | **76%** |
| GPT-4.1-mini FT | $50-100 | $3-5 | 80-85% |
| ~~GLiNER-BioMed~~ | $0 | ほぼ$0 | ~~79-83%~~ **43%** |
| scispaCy | $0 | ほぼ$0 | 76-80% |
| BioBERT FT | GPU数時間 | ほぼ$0 | 89-93% |

**結論**: 精度を追求するならBioBERT、現状最良はGPT-5.2 MRC+verify (F1=76%)

## 次のアクション

1. ~~**GLiNER-BioMedを評価**~~ ✅ 完了 - **不適合** (F1=43%)
2. **scispaCyを評価**（今すぐ可能）
3. **BioBERTファインチューニング環境構築**（Google Colab）- **推奨**

## 参考文献

- [BioBERT: a pre-trained biomedical language representation model](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)
- [GLiNER-BioMed: A Suite of Efficient Models for Open Biomedical NER](https://arxiv.org/abs/2504.00676)
- [scispaCy: Fast and Robust Models for Biomedical NLP](https://allenai.github.io/scispacy/)
- [BioBERT-MRC (CHEMDNER SOTA)](https://ar5iv.labs.arxiv.org/html/2009.01560)

---

*作成日: 2026-01-25*
