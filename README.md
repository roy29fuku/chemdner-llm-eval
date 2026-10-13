# CHEMDNER evaluation

BioCreative IV（2013）の CHEMDNER タスク（PubMed のアブストラクトから化学物質名を抽出する）で、LLM を使ったいくつかの解き方を、**1 本の採点器（CEM）でそろえて**評価するための実験基盤。

> [!NOTE]
> いまの v2 は整備の途中。2026-01 の探索的な実験（v1）は [legacy/v1-2026-01/](legacy/v1-2026-01/) にあり、そこで報告した 2 つの値は F1 の定義などがそろっておらず比較に使えない（理由は同ディレクトリの README）。

## データ

CHEMDNER のコーパスは、NCBI が BioCreative VII の NLM-Chem トラックで配布している BioC 版を使う。BioCreative IV のコーパスを BioC に変換したもので、元の配布元（biocreative.org）からは 2026-10 時点で取得できない。

**コーパスはこの repo に含めない。** 明示されたライセンスが見つからず、本文は PubMed のアブストラクトのため。取得のスクリプトは配布ファイルの SHA-256 を確かめる。

| 分割 | アブストラクト | 使える言及 | 公式（Krallinger et al. 2015） |
|---|---|---|---|
| training | 3,500 | 29,462 | 29,478 |
| development | 3,500 | 29,523 | 29,526 |
| evaluation（公式の test） | 3,000 | 25,342 | 25,351 |

- 本文は公式とバイト単位で一致する
- 言及は公式より少ない。evaluation では、注釈が 5 件無く、位置が失われた長さ 0 の注釈が 4 件ある（読み込みで除く）
- NCBI が足した MeSH の索引語の注釈は CHEMDNER のものではないので使わない

## 採点器（CEM）

BioCreative IV の CEM（chemical entity mention）と同じ定義で採点する。

- 言及は (PMID, 節, 開始, 終了) の組。節は title（T）か abstract（A）で、位置は節ごとに 0 から数える
- 4 つすべてが正解と一致したものだけを正解とする。予測の重複は 1 つに数える
- 全文書を合算した micro precision / recall / F1。区間はアブストラクトを単位にした bootstrap で出す
- ファイルの形式は公式と同じ `PMID<TAB>節:開始:終了`（3 列目以降は無視する）

採点器は、NCBI が配布している BioCreative VII の公式評価スクリプト（strict の span 評価）と突き合わせ、同じキーで数えると TP / FP / FN が完全に一致することを確かめている（`scripts/crosscheck_bc7_official.py`）。公式スクリプトは節を区別しないため、title と abstract の同じ位置・同じ文字列の言及を 1 件に数える点だけが違う。

## 使い方

[uv](https://docs.astral.sh/uv/) を使う。

```sh
uv sync
uv run python -m chemdner_bench.corpus fetch          # コーパスを NCBI から data/ に取得（SHA-256 を確認）
uv run python -m chemdner_bench.corpus stats          # 分割ごとの件数
uv run python -m chemdner_bench.corpus gold --split evaluation --out gold.tsv
uv run python -m chemdner_bench.cem --gold gold.tsv --pred pred.tsv
uv run pytest                                         # コーパスが data/ にあれば、件数と満点の確認も走る
uv run python scripts/crosscheck_bc7_official.py      # 公式評価スクリプトとの突き合わせ
```

## 構成

```
.
├── src/chemdner_bench/
│   ├── corpus.py      コーパスの取得と読み込み
│   └── cem.py         採点器
├── tests/
├── scripts/           公式評価スクリプトとの突き合わせなど
├── legacy/v1-2026-01/ 2026-01 の探索的な実験（当時のまま）
└── data/              コーパスなどの取得物（git 管理しない）
```

## 引用

コーパスを使うときは次を引用する。

- Krallinger, M., et al. The CHEMDNER corpus of chemicals and drugs and its annotation principles. J. Cheminform. 7, S2 (2015). https://doi.org/10.1186/1758-2946-7-S1-S2
- Krallinger, M., et al. CHEMDNER: the drugs and chemical names extraction challenge. J. Cheminform. 7, S1 (2015). https://doi.org/10.1186/1758-2946-7-S1-S1
