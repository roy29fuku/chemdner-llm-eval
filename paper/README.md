# Materials for the paper

Materials for the revised Note "When does a reasoning brain need hands? Let a yardstick decide — chemical named-entity recognition with an AI scientist" (Biophysics and Physicobiology, BSJ-2026038), referred to in its Data and code availability statement and Supplementary Note 7.

| Path | Contents |
|---|---|
| [`analysis-plan/`](analysis-plan/) | Every version of the analysis plan (comparisons, decision rules, and the decision log), with commit times. Text not related to the analysis is redacted and marked |
| [`predictions/`](predictions/) | Test-set predictions of every run scored in the paper (document ID, section and character positions only; no corpus text). With a copy of the CHEMDNER corpus, they can be rescored with the scripts of this repository |
| [`agent-code/`](agent-code/) | Code written by the coding agent in each agent run (corpus-derived files and model files excluded; see its README) |
| [`run-records-sha256.txt`](run-records-sha256.txt) | SHA-256 of every file of the full run records. The records contain corpus text and are kept private |

The CHEMDNER corpus itself is not redistributed here because its license is not stated (see Supplementary Note 1 of the paper for where to obtain it).
