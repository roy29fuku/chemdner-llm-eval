# Analysis plan: version history

Every committed version of the analysis plan (`docs/experiment-protocol.md` in the manuscript repository), in commit order. The manuscript repository itself is private; these files are verbatim copies of the file as it was at each commit on its `main` branch, except for the redactions listed below. Each file is named `<committer date in JST>_<short commit SHA>.md`. The plan is written in Japanese.

Timeline notes (from the plan's own decision log and the run records):

- Main comparison and the GCP re-run of the agent (Experiment 14): the re-run and the addition of GPT-6.1 sol were recorded at `87e76c8` (2026-10-14 15:27 JST). The test labels were first copied for scoring at 2026-10-15 03:56 JST (2026-10-14T18:56:22Z).
- Example-shown control and the 128-example agent (Experiment 15, section 「追加の対照（実験 15）」): first added at `010e5e7` (2026-10-20 17:08 JST), i.e. after the main comparison had been scored. The Experiment 15 runs were frozen at 17:09 JST and scored from 19:45 JST the same day. `81209e7` and `a3bda4e` were committed after that scoring; they add to the decision log a clarification of the category rule (decided before scoring and implemented in code commit `e9efb7d`), the reconstruction of the b(128) run summaries, and the scoring record.

Redactions (text not related to the analysis; each is marked 「[伏せた: …]」 in place, 23 in total):

- Line 3 of every version (20): an internal task-tracking number and a link to the authors' plan for revising the manuscript.
- One row of the decision log in versions 18–20 (3): a decision about the manuscript submission procedure, not about the experiment.

Nothing else was changed. References to the authors' internal records (`dev_logs`, `review/`, and private repositories) are left as written; those records are not public.

20 versions.

| # | Version time (JST) | Commit | File | Commit subject |
|---|---|---|---|---|
| 1 | 2026-10-13 21:09 | `80d85f5` | [2026-10-13T2109+0900_80d85f5.md](2026-10-13T2109+0900_80d85f5.md) | docs: 改訂の計画と再実験のプロトコルを書く |
| 2 | 2026-10-13 23:14 | `d7685cb` | [2026-10-13T2314+0900_d7685cb.md](2026-10-13T2314+0900_d7685cb.md) | docs(review): 横断監査の結果を記録し、事実確認・計画・プロトコルに反映する |
| 3 | 2026-10-13 23:30 | `4a994f7` | [2026-10-13T2330+0900_4a994f7.md](2026-10-13T2330+0900_4a994f7.md) | docs: 監査で出た判断事項の決定をプロトコルと計画に反映する |
| 4 | 2026-10-14 00:36 | `4c40eec` | [2026-10-14T0036+0900_4c40eec.md](2026-10-14T0036+0900_4c40eec.md) | docs: まず gpt-6-luna だけで実験する決定と、test の隔離の限界を記録する |
| 5 | 2026-10-14 00:43 | `9959445` | [2026-10-14T0043+0900_9959445.md](2026-10-14T0043+0900_9959445.md) | docs: 実験を最もナイーブな使い方どうしの最小の比較にする |
| 6 | 2026-10-14 00:46 | `f2e5cdf` | [2026-10-14T0046+0900_f2e5cdf.md](2026-10-14T0046+0900_f2e5cdf.md) | docs: a の試行の結果と、a-B を足さない判定を記録する |
| 7 | 2026-10-14 01:16 | `21ced19` | [2026-10-14T0116+0900_21ced19.md](2026-10-14T0116+0900_21ced19.md) | docs: b は計算環境の構築からエージェントに任せ、こちらは安全策だけを用意する |
| 8 | 2026-10-14 09:21 | `6cb3a27` | [2026-10-14T0921+0900_6cb3a27.md](2026-10-14T0921+0900_6cb3a27.md) | docs(protocol): b を「手段を問わず最良を」に再設計し、条件・判定・監査を事前登録する |
| 9 | 2026-10-14 09:30 | `ffcf379` | [2026-10-14T0930+0900_ffcf379.md](2026-10-14T0930+0900_ffcf379.md) | docs(protocol): 規則 3 の 2 版の試行・VM の通信・a の反復の決定を記録する |
| 10 | 2026-10-14 10:14 | `f07b61b` | [2026-10-14T1014+0900_f07b61b.md](2026-10-14T1014+0900_f07b61b.md) | docs(protocol): プロジェクトでの隔離と、計画だけを立てさせる比較の判定を事前登録する |
| 11 | 2026-10-14 12:19 | `9c508a1` | [2026-10-14T1219+0900_9c508a1.md](2026-10-14T1219+0900_9c508a1.md) | docs(protocol): 本番の b の課題文を free に決め、API の拒否の扱いを事前登録する |
| 12 | 2026-10-14 12:20 | `ca2bb9f` | [2026-10-14T1220+0900_ca2bb9f.md](2026-10-14T1220+0900_ca2bb9f.md) | fix(protocol): 本番の b の課題文を free-banonly に直す（読み違えの訂正） |
| 13 | 2026-10-14 12:42 | `32b80d1` | [2026-10-14T1242+0900_32b80d1.md](2026-10-14T1242+0900_32b80d1.md) | docs(protocol): 本番の b の 1 回目を基盤の不具合で中止し、走らせ直すことを記録する |
| 14 | 2026-10-14 15:27 | `87e76c8` | [2026-10-14T1527+0900_87e76c8.md](2026-10-14T1527+0900_87e76c8.md) | docs(protocol): 追加実験（GCP の司令塔での b の走らせ直しと gpt-6.1-sol）を事前登録する |
| 15 | 2026-10-14 19:23 | `d5ad1d2` | [2026-10-14T1923+0900_d5ad1d2.md](2026-10-14T1923+0900_d5ad1d2.md) | docs(protocol): 追加実験の sol の条件・反復の追加・禁止のモデルを止められた回の扱いを記録する |
| 16 | 2026-10-14 20:55 | `e96887e` | [2026-10-14T2055+0900_e96887e.md](2026-10-14T2055+0900_e96887e.md) | docs(protocol): 予測なしの回の扱いと、sol の反復を早めに始めたことを記録する |
| 17 | 2026-10-14 23:09 | `00672cf` | [2026-10-14T2309+0900_00672cf.md](2026-10-14T2309+0900_00672cf.md) | docs(protocol): API の混雑で止まった sol medium の r2 を走らせ直すことを記録する |
| 18 | 2026-10-20 17:08 | `010e5e7` | [2026-10-20T1708+0900_010e5e7.md](2026-10-20T1708+0900_010e5e7.md) | docs(protocol): 例を見せて答えさせる対照 a' と、128 件だけのエージェント b(128) を追記する（実験 15） |
| 19 | 2026-10-20 19:51 | `81209e7` | [2026-10-20T1951+0900_81209e7.md](2026-10-20T1951+0900_81209e7.md) | docs(protocol): 実験 15 の採点の前に決めた区分の読み方と、b(128) の記録の組み立て直しと、採点を決定の記録に足す |
| 20 | 2026-10-20 20:02 | `a3bda4e` | [2026-10-20T2002+0900_a3bda4e.md](2026-10-20T2002+0900_a3bda4e.md) | docs(protocol): 決定の記録の dev log の参照を、まとめ直した再実験 7 に直す |

Full commit SHAs:

- `80d85f5` = `80d85f539cd6fb05b2584e13c78c4e910571be3d`
- `d7685cb` = `d7685cb3fa191b80d5ea48612294a34ccacb8cc8`
- `4a994f7` = `4a994f7760e645780557b8a4732d6bc7c258d14e`
- `4c40eec` = `4c40eeccace4a47ea0ee817119cd5a4fd3774f6f`
- `9959445` = `9959445b84d480a5e64e6240c94fd7862e1cf116`
- `f2e5cdf` = `f2e5cdfc271b3559704ee2c3bc4abb34c8ec33fb`
- `21ced19` = `21ced196df3bdc58dfe224c06811ed35ea2dd5b5`
- `6cb3a27` = `6cb3a278e7f8776b546b59f2e93bbf369b9eec1b`
- `ffcf379` = `ffcf379d6fe3909761d790ff4bafbf8229602b19`
- `f07b61b` = `f07b61b31175c53d9df43bc65b8cbc36f42b67aa`
- `9c508a1` = `9c508a1efbaafdd7949cfd580af48ec71bddcb29`
- `ca2bb9f` = `ca2bb9f09beddd6c031f651c0f43e8d5bd5a135d`
- `32b80d1` = `32b80d13744b73f06756592bc951bef683fadd0e`
- `87e76c8` = `87e76c8839e3b3c20b9b08061858089095f212bf`
- `d5ad1d2` = `d5ad1d2b30a6975fdca875256a1bfc0150d5e6de`
- `e96887e` = `e96887e7c5b73136a03a3078f9c0d8a9f1f5c55d`
- `00672cf` = `00672cfad1ce786d2cc513fc78bbca4abfdb2d4f`
- `010e5e7` = `010e5e7e22994da1d0e8feee3d7523064a4f1854`
- `81209e7` = `81209e7795ee42706e53edb03ad414e2bbefb5d7`
- `a3bda4e` = `a3bda4e20e113bed08f7c953be53970ece9a6144`
