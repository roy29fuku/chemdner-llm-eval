# CHEMDNER精度向上計画

## 現状分析

### 現在の最高性能
- **手法**: Few-shot (4例) + Self-verification
- **F1**: 0.7452
- **SOTAとの差**: -19.78pt（現SOTA: 94.30%）

### CHEMDNER SOTA推移

| 年 | 手法 | F1 | 備考 |
|----|------|-----|------|
| 2013 | tmChem (CRF + 辞書) | 87.39% | BioCreative IV 1位 |
| 2019 | BioBERT | 89.7% | Transformer初期 |
| 2020 | **BioBERT-MRC** | 92.92% | NERをMRC問題に変換 |
| 2024 | **MatSciBERT-MRC** | **94.30%** | 現在のSOTA |

**注目**: MRC (Machine Reading Comprehension) 手法がSOTAを達成

### ボトルネック分析
| 指標 | 現在値 | 課題 |
|------|--------|------|
| Precision | 0.8077 | 比較的良好 |
| Recall | 0.6917 | **改善余地あり** |
| FN | 4,574件 | 見逃しが多い |

## 調査結果

### 1. 効果が期待できる手法

#### A. MRC (Machine Reading Comprehension) 変換 ★最有望
- **効果**: [BioBERT-MRC](https://ar5iv.labs.arxiv.org/html/2009.01560)でF1=92.92%、[MatSciBERT-MRC](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00874-5)でF1=94.30%
- **原理**: NERをシーケンスラベリングではなく、質問応答問題として定式化
  - 質問: 「このテキストに含まれる化学物質名は何ですか？」
  - 回答: エンティティのスパン（開始・終了位置）
- **LLMへの適用**:
  - 生成ベースのLLMはMRC的なプロンプトと相性が良い
  - 現在のプロンプトも実質的にMRC形式（質問→回答）
- **改善方向**: エンティティの開始・終了位置を明示的に出力させる

#### B. ファインチューニング
- **効果**: [研究によると](https://pmc.ncbi.nlm.nih.gov/articles/PMC11339492/)、ファインチューニングでF1が5%〜30%向上する事例あり
- **課題**: GPT-5.2はファインチューニング非対応、GPT-4.1-miniは対応
- **コスト**: 学習データ準備 + API費用

#### B. 指示ベース学習（Instruction Tuning）
- **効果**: [BioNER-LLaMA](https://academic.oup.com/bioinformatics/article/40/4/btae163/7633405)はNERを生成タスクに変換し、5-30%改善
- **方法**: シーケンスラベリング → テキスト生成への変換

#### C. Prompt Ensemble
- **効果**: [DIPPER研究](https://www.comp.nus.edu.sg/~greglau/assets/pdf/dipper_neurips_mint.pdf)によると、多様なプロンプトの並列実行で精度向上
- **方法**: 複数のプロンプトバリエーションで推論し、多数決
- **注意**: [研究](https://arxiv.org/abs/2412.00166)では、個別性能に大きな差がある場合は効果薄

#### D. 辞書ベース後処理
- **効果**: [GLiNER-BioMed](https://arxiv.org/html/2510.08588)では辞書後処理でF1が0.79→0.83に改善（+4pt）
- **方法**: 化学物質辞書との照合、誤分類の修正

#### E. ハイブリッドアプローチ
- **効果**: [研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC12335919/)によると、長いエンティティにはLLMが優位、短いものはエンコーダモデルが効率的
- **方法**: GLiNER + LLMのアンサンブル

### 2. 効果が限定的な手法

| 手法 | 理由 |
|------|------|
| CoT単独 | Exp003で逆効果（-1.9pt）、NERは認識タスク |
| Few-shot例増加 | Exp004で効果なし（-0.48pt）、量より質 |
| Responses API | Exp002で効果なし（-0.24pt）、単一ターンには不向き |

## 改善計画（優先度順）

### Phase 1: プロンプト最適化（低コスト・高速）

#### Exp006: MRC形式プロンプト ★推奨
- **内容**: NERをMRC問題として再定式化したプロンプト
- **方法**:
  - 質問形式で化学物質タイプごとに抽出
  - 開始・終了位置を明示的に出力
- **期待効果**: +2-5pt（SOTA手法の原理を応用）
- **コスト**: 〜$15

#### Exp007: Self-verify強化
- **内容**: 検証プロンプトにFew-shot例を追加
- **期待効果**: +1-2pt
- **コスト**: 〜$20

#### Exp008: Prompt Ensemble
- **内容**: 3-5種類のプロンプトバリエーションで多数決
- **期待効果**: +1-3pt
- **コスト**: 〜$50（3バリエーション × $17）

### Phase 2: 辞書・ルール統合（中コスト）

#### Exp009: 辞書ベース後処理
- **内容**: PubChem/ChEBI辞書との照合、誤分類修正
- **期待効果**: +2-4pt（GLiNER-BioMed実績）
- **コスト**: 辞書構築の工数

#### Exp010: タイプ別処理
- **内容**: 化学物質タイプ（TRIVIAL/SYSTEMATIC/ABBREVIATION）別に最適化
- **期待効果**: +1-2pt

### Phase 3: ファインチューニング（高コスト・高効果）

#### Exp011: GPT-4.1-mini ファインチューニング
- **内容**: CHEMDNER training setでファインチューニング
- **期待効果**: +5-15pt（文献実績）
- **コスト**: 〜$50-100（学習費用）

### Phase 4: ハイブリッド（最高性能を目指す）

#### Exp012: GLiNER + LLM Ensemble
- **内容**: エンコーダモデル（GLiNER-BioMed）とLLMの組み合わせ
- **期待効果**: SOTA接近（85%+）
- **コスト**: モデル構築の工数

## 実行スケジュール案

| Phase | 実験 | 予想F1 | コスト | 優先度 |
|-------|------|--------|--------|--------|
| 1 | Exp006: MRC形式 | 0.77+ | $15 | ★★★ |
| 1 | Exp007: Self-verify強化 | 0.76+ | $20 | ★★☆ |
| 1 | Exp008: Prompt Ensemble | 0.77+ | $50 | ★★☆ |
| 2 | Exp009: 辞書後処理 | 0.79+ | 工数 | ★★☆ |
| 3 | Exp011: Fine-tuning | 0.82+ | $100 | ★★★ |
| 4 | Exp012: Hybrid | 0.85+ | 工数 | ★☆☆ |

## 次のアクション

1. **最優先**: Exp006（MRC形式プロンプト）- SOTA手法の原理を応用
2. **即座に実行可能**: Exp007（Self-verify強化）
3. **準備が必要**: Exp011（Fine-tuning用データ準備）
4. **調査が必要**: Exp012（GLiNERモデルの評価）

## 参考文献

### SOTA手法（MRC）
- [BioBERT-MRC (F1=92.92%)](https://ar5iv.labs.arxiv.org/html/2009.01560) - NERをMRC問題に変換
- [MatSciBERT-MRC (F1=94.30%)](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00874-5) - 現在のSOTA

### プロンプトエンジニアリング
- [Improving LLMs for Clinical NER via Prompt Engineering](https://pmc.ncbi.nlm.nih.gov/articles/PMC11339492/)
- [DIPPER: Diversity in Prompts](https://www.comp.nus.edu.sg/~greglau/assets/pdf/dipper_neurips_mint.pdf)
- [Majority Voting Strategies for LLMs](https://arxiv.org/abs/2412.00166)

### ファインチューニング・ハイブリッド
- [BioNER-LLaMA: Instruction Tuning for Biomedical NER](https://academic.oup.com/bioinformatics/article/40/4/btae163/7633405)
- [GLiNER-BioMed with Dictionary Post-processing](https://arxiv.org/html/2510.08588)
- [LLMs vs Encoders for Biomedical NER](https://pmc.ncbi.nlm.nih.gov/articles/PMC12335919/)

### ベースライン
- [BioBERT](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)
- [ChemDataExtractor 2.1](https://pubs.acs.org/doi/full/10.1021/acs.jcim.1c01199)

---

*作成日: 2026-01-25*
