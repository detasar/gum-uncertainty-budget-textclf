### Reference metrics per instrument class (250 cal / 250 test, alpha = 0.10; 95% CI = joint bootstrap of cal and test, 1000 draws)

**Not a leaderboard.** The classes differ in output type and in the information they use. Read across rows only *within* a class.

**deterministic schema classifier (zero-shot)**: output = single-pass option probabilities (softmax of head logits), no sampling

| model | accuracy [95% CI] | coverage [95% CI] | mean set size [95% CI] | shifted test (0.3): acc / cov / size |
|---|---|---|---|---|
| clef-flash-8bit (zero-shot) | 0.868 [0.824, 0.908] | 0.860 [0.788, 0.940] | 0.984 [0.868, 1.100] | 0.880 / 0.880 / 1.000 |

**generative chat LLM (zero-shot)**: output = first-token logprobs over letters A-D (renormalised); the sampled answer is a separate output

| model | accuracy [95% CI] | coverage [95% CI] | mean set size [95% CI] | shifted test (0.3): acc / cov / size |
|---|---|---|---|---|
| qwen3.5 (zero-shot) | 0.892 [0.852, 0.928] | 0.892 [0.840, 0.932] | 1.000 [0.924, 1.048] | 0.880 / 0.884 / 1.008 |
| qwen38-q5-fast (zero-shot) | 0.892 [0.852, 0.928] | 0.896 [0.804, 0.944] | 1.016 [0.920, 1.192] | not run |

**supervised probe**: output = single-pass class probabilities from an MLP trained on 2,000 labelled items

| model | accuracy [95% CI] | coverage [95% CI] | mean set size [95% CI] | shifted test (0.3): acc / cov / size |
|---|---|---|---|---|
| minilm+MLP (supervised) | 0.832 [0.784, 0.876] | 0.880 [0.796, 0.948] | 1.132 [0.948, 1.476] | 0.808 / 0.876 / 1.152 |
| bgesmall+MLP (supervised) | 0.868 [0.828, 0.908] | 0.888 [0.832, 0.940] | 1.060 [0.968, 1.152] | 0.856 / 0.876 / 1.076 |
| embeddinggemma+MLP (supervised) | 0.852 [0.804, 0.892] | 0.888 [0.800, 0.936] | 1.084 [0.940, 1.216] | 0.832 / 0.892 / 1.148 |


### Budget A_supervised_minilm: reference = minilm+MLP (set size 1.132, coverage 0.880)

| component | evaluation type | u(set size) | share of variance | u(coverage) | share of variance | notes |
|---|---|---|---|---|---|---|
| calibration-set sampling (n=250) | A (bootstrap) | 0.1457 | 71% | 0.0340 | 52% | Bootstrap of the calibration set. |
| test-set sampling (n=250) | A (bootstrap) | 0.0240 | 2% | 0.0212 | 20% | Finite evaluation sample. |
| random seed (MLP init/order) | A (repeat) | 0.0581 | 11% | 0.0154 | 11% | 15 refits, same data. |
| training-data sampling (bootstrap, n=2000) | A (bootstrap) | 0.0621 | 13% | 0.0186 | 16% | 15 refits on bootstrap resamples. |
| model choice within class (3 embedders) | B (discrete uniform over alternatives) | 0.0299 | 3% | 0.0038 | 1% | minilm / bge-small / embeddinggemma, same head. |
| distribution shift (word drop + typos, level 0.3) | B (rectangular, half-width = abs(shift effect)) | 0.0115 | 0% | 0.0023 | 0% | Uncorrected systematic effect; assumes the actual shift lies anywhere between none and the tested level. |
| **combined (root-sum-square)** | | **0.1734** | | **0.0470** | | independence + additivity assumed |

### Budget B_zeroshot_clef: reference = clef-flash-8bit, w0_official|q0_topic (set size 0.984, coverage 0.860)

| component | evaluation type | u(set size) | share of variance | u(coverage) | share of variance | notes |
|---|---|---|---|---|---|---|
| run-to-run repeatability (identical runs) | A (repeat, >=4 identical runs) | 0.0000 | 0% | 0.0000 | 0% | 4 runs x 200 items (half cal / half test); max |dp| = 0; items with changed conformal set = 0.0% |
| calibration-set sampling (n=250) | A (bootstrap) | 0.0656 | 64% | 0.0339 | 60% | Bootstrap of the calibration set. |
| test-set sampling (n=250) | A (bootstrap) | 0.0078 | 1% | 0.0223 | 26% | Finite evaluation sample. |
| label/option wording (4 paraphrase sets) | B (discrete uniform over alternatives) | 0.0467 | 32% | 0.0077 | 3% | Same question, 4 verbalizer sets; each recalibrated. |
| question phrasing (3 instructions) | B (discrete uniform over alternatives) | 0.0118 | 2% | 0.0086 | 4% | Same options, 3 instructions; each recalibrated. |
| model choice within class (deterministic schema classifiers) | B | 0.0000 | 0% (excluded) | 0.0000 | 0% (excluded) | NOT EVALUATED: only one model of this instrument type was run (GLiNER2 install failed). Not zero, unknown. |
| distribution shift (word drop + typos, level 0.3) | B (rectangular, half-width = abs(shift effect)) | 0.0092 | 1% | 0.0115 | 7% | Uncorrected systematic effect; assumes the actual shift lies anywhere between none and the tested level. |
| **combined (root-sum-square)** | | **0.0822** | | **0.0438** | | independence + additivity assumed |

### Budget C_zeroshot_qwen3.5_9b: reference = qwen3.5, w0_official, temperature 0 (set size 1.000, coverage 0.892)

| component | evaluation type | u(set size) | share of variance | u(coverage) | share of variance | notes |
|---|---|---|---|---|---|---|
| run-to-run repeatability, temperature 0 (operating mode) | A (repeat, >=4 identical runs) | 0.0000 | 0% | 0.0000 | 0% | 4 runs x 150 items (half cal / half test); max |dp| = 0; items with changed conformal set = 0.0% |
| run-to-run repeatability IF run at default temperature (not in RSS) | A (repeat, >=4 identical runs) | 0.0000 | 0% (excluded) | 0.0000 | 0% (excluded) | 4 runs x 150 items (half cal / half test); max |dp| = 0; items with changed conformal set = 0.0% |
| calibration-set sampling (n=250) | A (bootstrap) | 0.0278 | 39% | 0.0159 | 30% | Bootstrap of the calibration set. |
| test-set sampling (n=250) | A (bootstrap) | 0.0059 | 2% | 0.0206 | 51% | Finite evaluation sample. |
| model choice within class (generative LLMs: qwen3.5, qwen38-q5-fast) | B (discrete uniform over alternatives) | 0.0080 | 3% | 0.0020 | 0% | Same prompt and wording w0; each recalibrated. |
| label/option wording (4 paraphrase sets) | B (discrete uniform over alternatives) | 0.0329 | 55% | 0.0114 | 16% | Same prompt template, different verbalizers; each recalibrated. |
| distribution shift (word drop + typos, level 0.3) | B (rectangular, half-width = abs(shift effect)) | 0.0046 | 1% | 0.0046 | 3% | Uncorrected systematic effect; assumes the actual shift lies anywhere between none and the tested level. |
| **combined (root-sum-square)** | | **0.0444** | | **0.0288** | | independence + additivity assumed |

**Monte Carlo check (Clef, GUM-S1 style, 2000 joint draws over prompt config x cal bootstrap x test bootstrap):** SD(set size) = 0.0727 vs RSS of the same four components = 0.0817; SD(coverage) = 0.0378 vs RSS = 0.0423.

### Calibration size vs wording (Clef, 4 wordings, pool = 500 calibration items, 300 paired draws per n)

| n_cal | SD from calibration sampling | wording, systematic part (SD of per-wording means) [95% CI] | wording, within-draw SD |
|---|---|---|---|
| 25 | 0.9115 | 0.0779 [0.056, 0.107] | 0.3572 |
| 50 | 0.2317 | 0.0283 [0.022, 0.035] | 0.0766 |
| 100 | 0.0956 | 0.0225 [0.019, 0.027] | 0.0434 |
| 200 | 0.0625 | 0.0197 [0.017, 0.022] | 0.0313 |
| 400 | 0.0431 | 0.0197 [0.018, 0.021] | 0.0256 |
| 500 | 0.0392 | 0.0191 [0.018, 0.021] | 0.0236 |

### Entropy decomposition (TU = AU + EU) depends on what is called 'the model ensemble' (test set, nats)

| setup | mean TU | mean AU | mean EU | EU share | EU share under shift |
|---|---|---|---|---|---|
| minilm+MLP, ensemble = training bootstraps | 0.225 | 0.164 | 0.061 | 27% | 27% |
| bgesmall+MLP, ensemble = training bootstraps | 0.168 | 0.124 | 0.043 | 26% | 25% |
| embeddinggemma+MLP, ensemble = training bootstraps | 0.161 | 0.125 | 0.036 | 22% | 22% |
| clef, single prompt (deterministic => no ensemble) | 0.406 | 0.406 | 0.000 | 0% | - |
| clef, ensemble = 6 prompt configs | 0.380 | 0.360 | 0.019 | 5% | - |

| pair | Spearman rho of per-item EU | overlap of top-10% EU items |
|---|---|---|
| minilm+MLP, ensemble = training bootstraps  vs  bgesmall+MLP, ensemble = training bootstraps | 0.70 | 36% |
| minilm+MLP, ensemble = training bootstraps  vs  embeddinggemma+MLP, ensemble = training bootstraps | 0.62 | 36% |
| minilm+MLP, ensemble = training bootstraps  vs  clef, ensemble = 6 prompt configs | 0.42 | 16% |
| bgesmall+MLP, ensemble = training bootstraps  vs  embeddinggemma+MLP, ensemble = training bootstraps | 0.65 | 48% |
| bgesmall+MLP, ensemble = training bootstraps  vs  clef, ensemble = 6 prompt configs | 0.53 | 16% |
| embeddinggemma+MLP, ensemble = training bootstraps  vs  clef, ensemble = 6 prompt configs | 0.50 | 44% |

### Test-retest repeatability (identical requests, >= 4 runs)

| setup | runs x items | max abs dp (any pair) | mean per-item SD(p) | ICC(3,1) / ICC(2,1) on p(true) | Bland-Altman run1-run0: mean [LoA] | mean pairwise TV / JS (bits) | label source | Cohen kappa mean [range] | Fleiss kappa | items same label all runs | SD set size / coverage across runs | test items whose set changes | saturation: max p>0.99 / entries outside [0.01,0.99] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clef-flash-8bit / w0_official / deterministic | 4 x 200 | 0 | 0 | 1.000 / 1.000 | 0 [0, 0] | 0 / 0 | argmax | 1.000 [1.000, 1.000] | 1.000 | 100.0% | 0 / 0 | 0.0% | 0% / 4% |
| clef-flash-8bit / w1_news / deterministic | 4 x 20 | 0 | 0 | 1.000 / 1.000 | 0 [0, 0] | 0 / 0 | argmax | 1.000 [1.000, 1.000] | 1.000 | 100.0% | 0 / 0 | 0.0% | 0% / 2% |
| clef-flash-8bit / w2_topic / deterministic | 4 x 20 | 0 | 0 | 1.000 / 1.000 | 0 [0, 0] | 0 / 0 | argmax | 1.000 [1.000, 1.000] | 1.000 | 100.0% | 0 / 0 | 0.0% | 0% / 15% |
| clef-flash-8bit / w3_short / deterministic | 4 x 20 | 0 | 0 | 1.000 / 1.000 | 0 [0, 0] | 0 / 0 | argmax | 1.000 [1.000, 1.000] | 1.000 | 100.0% | 0 / 0 | 0.0% | 0% / 14% |
| qwen3.5 / w0_official / default | 4 x 150 | 0 | 0 | 1.000 / 1.000 | 0 [0, 0] | 0 / 0 | chosen answer | 0.996 [0.991, 1.000] | 0.996 | 99.3% | 0 / 0 | 0.0% | 76% / 86% |
| qwen3.5 / w0_official / t0 | 4 x 150 | 0 | 0 | 1.000 / 1.000 | 0 [0, 0] | 0 / 0 | chosen answer | 1.000 [1.000, 1.000] | 1.000 | 100.0% | 0 / 0 | 0.0% | 76% / 86% |
| qwen3.5 / w2_topic / t0 | 4 x 50 | 0 | 0 | 1.000 / 1.000 | 0 [0, 0] | 0 / 0 | chosen answer | 1.000 [1.000, 1.000] | 1.000 | 100.0% | 0 / 0 | 0.0% | 80% / 87% |

### Configuration shift: calibrate on one label wording, test on another (alpha = 0.10, target coverage 0.90)

| model | coverage, same wording: mean [min, max] | coverage, cross wording: mean [min, max] | wording ensemble (mean prob): coverage / set size |
|---|---|---|---|
| clef-flash-8bit | 0.855 [0.844, 0.864] | 0.863 [0.800, 0.920] | 0.844 / 0.940 |
| qwen3.5 | 0.891 [0.876, 0.908] | 0.890 [0.832, 0.920] | 0.896 / 1.004 |

### Option name vs definition (Clef-flash, 250 test items, definitions = w2_topic texts)

| condition | selects the correct definition | selects the option whose NAME matches the true class |
|---|---|---|
| neutral_ids_A-D | 0.896 | nan |
| matched | 0.884 | 0.884 |
| swapped | 0.808 | 0.140 |

decision_flip_rate_neutral_to_matched: 3.6%

decision_flip_rate_matched_to_swapped: 15.2%

Cross-session check (main run in a separate process vs repeatability run 0, identical requests): clef-flash-8bit w0|q0 (probabilities): max abs dp = 0.0 over 200 items; qwen3.5 w0 t0 (A-D probabilities): max abs dp = 0.0 over 150 items
