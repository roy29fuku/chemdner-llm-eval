# CHEMDNER evaluation

BioCreative IV（2013）の CHEMDNER タスク（PubMed のアブストラクトから化学物質名を抽出する）で、LLM を使ったいくつかの解き方を、**1 本の採点器（CEM）でそろえて**評価するための実験基盤。

> [!NOTE]
> いまの v2 は整備の途中。2026-01 の探索的な実験（v1）は [legacy/v1-2026-01/](legacy/v1-2026-01/) にあり、そこで報告した 2 つの値は F1 の定義などがそろっておらず比較に使えない（理由は同ディレクトリの README）。

## データ

**コーパスはこの repo に含めない。** 明示されたライセンスが見つからず、本文は PubMed のアブストラクトのため。読み込める版は 2 つで、どちらも SHA-256 で原本と同一かを確かめる。

| 版 | 内容 | 入手 |
|---|---|---|
| official（正本） | 2013 年の BioCreative IV の公式配布物。件数が公式の表と完全に一致する | 元の配布元（biocreative.org）は 2026-10 時点で使えない。手元のコピーから `fetch --official` で読む |
| ncbi（公開の代替） | NCBI が BioCreative VII の NLM-Chem トラックで公開している BioC 版（BioCreative IV のコーパスを変換したもの） | `fetch --ncbi` で NCBI から取得する |

| 分割 | アブストラクト | 言及のある文書（official / ncbi） | 言及（official / ncbi） |
|---|---|---|---|
| training | 3,500 | 2,916 / 2,915 | 29,478 / 29,462 |
| development | 3,500 | 2,907 / 2,906 | 29,526 / 29,523 |
| evaluation（公式の test） | 3,000 | 2,478 / 2,477 | 25,351 / 25,346 |

- official の件数は公式の表（Krallinger et al. 2015, Table 1）と一致する
- ncbi は本文が公式とバイト単位で同一だが、各分割で 1 文書ずつ、その文書の言及がすべて欠けている（training: PMID 23645249、development: 23646356、evaluation: 23645677）。NCBI が足した MeSH の索引語の注釈は CHEMDNER のものではないので使わない
- evaluation には長さ 0 の正解が 4 件ある（いずれも MULTIPLE）。公式の時点で長さ 0 なので、そのまま読む（公式の採点ではどのシステムにも取れない FN になる）

## 採点器（CEM）

BioCreative IV の CEM（chemical entity mention）と同じ定義で採点する。

- 言及は (PMID, 節, 開始, 終了) の組。節は title（T）か abstract（A）で、位置は節ごとに 0 から数える
- 4 つすべてが正解と一致したものだけを正解とする。予測の重複は 1 つに数える
- micro precision / recall / F1。区間は採点対象の文書を単位にした bootstrap で出す
- ファイルの形式は公式と同じ `PMID<TAB>節:開始:終了`（3 列目以降は無視する）
- 採点範囲は 2 つ
  - **official（主）**: 2013 年の公式の評価（bc-evaluate）と同じく、正解のある文書だけを採点し、それ以外の文書の予測は捨てる（test では 2,478 件）
  - **all（副）**: 全文書（test では 3,000 件）を採点し、正解の無い文書への予測も FP に数える

**公式と同じ採点であることの確認**: 2013 年の公式の評価の見本（CHEMDNER_SAMPLE_JUNE25。bc-evaluate の採点結果つき）を、official の範囲で再現する（29 文書・TP 226・FP 302・FN 334・micro P 0.42803 / R 0.40357 / F 0.41544。`tests/test_cem.py`）。補助として、BioCreative VII の公式評価スクリプト（strict の span 評価）とも突き合わせ、同じキーで数えると TP / FP / FN が完全に一致することを確かめている（`scripts/crosscheck_bc7_official.py`。このスクリプトは節を区別せず、長さ 0 の注釈を捨てる点が BC IV と違う）。

## 使い方

[uv](https://docs.astral.sh/uv/) を使う。

```sh
uv sync
uv run python -m chemdner_bench.corpus fetch --official <official-2013 のディレクトリ>   # 公式配布物（手元のコピー）
uv run python -m chemdner_bench.corpus fetch --ncbi                                      # NCBI 版を取得
uv run python -m chemdner_bench.corpus stats [--source ncbi]
uv run python -m chemdner_bench.corpus gold --split evaluation --out gold.tsv --pmids pmids.txt
uv run python -m chemdner_bench.cem --gold gold.tsv --pred pred.tsv                           # 主（official）
uv run python -m chemdner_bench.cem --gold gold.tsv --pred pred.tsv --scope all --pmids pmids.txt  # 副（all）
uv run pytest                                         # 手元にコーパスがあれば、件数・公式の見本の再現も走る
uv run python scripts/crosscheck_bc7_official.py      # BC7 の公式評価スクリプトとの突き合わせ（ncbi が要る）
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
