# Label wording and calibration size in conformal zero-shot text classification: a preregistered re-test of a GUM-style uncertainty budget (v2, interim report)

*Davut Emre Tasar. Interim report, 2026-10-17, 16:23 Istanbul time (UTC+3). Branch `v2-comprehensive` of https://github.com/detasar/gum-uncertainty-budget-textclf. Preregistration: commit `fab3c8a` (2026-10-17 14:18:58 +0300).*

> **Status: interim.** Scoring jobs are still running (see [Pending runs](#pending-runs)). This report uses only the results that existed when it was written: 16 zero-shot cells (15 bi-encoder, 1 small LLM) and 15 supervised reference cells. The NLI cross-encoder family (T1) has **no** finished cell yet. Every number below is copied from `results/` (mainly `results/report_tables.md`, `results/summary.json`, `results/v1_reanalysis.json`, `results/template_factor.json`, `results/repeatability.json` and `results/repeat_rep0_vs_main.json`).

> **Comparison rule.** Models are compared **only within their own instrument type**. Generative LLMs scored by label likelihood (T3), deterministic zero-shot classifiers (T1 NLI cross-encoders, T2 bi-encoders) and supervised probes (T4) differ in their output and in the information they use. They are never ranked against each other here, in any table or figure. Across types we only compare the *shape* of the budget (which component dominates), never its level.

## Abstract

v1 of this study (an X Article and the public repo above) built GUM-style uncertainty budgets for the mean set size of split-conformal zero-shot text classifiers on AG News. It reported that label wording explained about 32% of the variance and calibration sampling about 64%. It also reported that the wording component stalled near 0.02 as the calibration set grew, which it read as a candidate "definitional" floor. v2 re-tests those claims under a preregistered design. Wording sets are drawn at random from a validity-checked paraphrase population (W = 30 per dataset). They are crossed with 200 random calibration/test splits, and the components are estimated by two-way random-effects method of moments with two-way bootstrap CIs. The design spans five datasets (AG News, BBC News 2024-10 to 2025-06, SST-2, DBpedia-14, MASSIVE-18), three conformal methods and three α levels.

In all 16 zero-shot cells finished so far, the wording component of mean set size is clearly non-zero, and at n_cal = 250 it is usually **larger** than split sampling. For bi-encoders the median σ_wording/σ_split ratio is 2.94 and the median wording share is 0.75 (range 0.31 to 0.88). For the one LLM cell (Qwen2.5-0.5B, AG News) the share is 0.92 [0.87, 0.94]. Wording does not shrink with n: the log-log slope over n ≥ 250 lies between −0.04 and 0.03 in every cell, while split sampling falls with slopes from −0.35 to −0.87. The plateau does not come from having few paraphrases (W = 4 subsets average 0.87 to 0.96 of the W = 30 estimate).

Wording leaves coverage untouched when calibration and test use the same wording (wording share of coverage variance 0.00 everywhere). When the wording changes between calibration and deployment, coverage drops: mean coverage over wording sets falls to as low as 0.732 against a 0.90 target. The size of a label change in meaning predicts how much the predictions move (bi-encoders: Spearman ρ = 0.56 [0.51, 0.62]).

Re-analysing the v1 outputs, the published 32% share has a 95% CI of [0.00, 0.84], and the share itself depends on n (0.03 at n = 50 to 0.25 at n = 500 for Clef). The "floor exists" claim is confirmed. The specific share numbers are not a stable property of the instrument. The "definitional" reading is weakened: the floor test separates systematic from random effects, not definitional content from instrument non-invariance.

## 1. Research questions, preregistration and deviations

### 1.1 Preregistration
`plan.md` (research questions, hypotheses, refutation criteria, analysis plan) and `gaps.md` (red-team analysis of v1) were committed as `fab3c8a` at 2026-10-17 14:18:58 +0300. The current `plan.md` is byte-identical to the committed version (checked with `git show fab3c8a:v2/plan.md | diff`). The earliest v2 raw classifier output (`raw/agnews__bienc__all-MiniLM-L6-v2.npz`) is timestamped 14:19:55 Istanbul time, after the commit. `plan.md` discloses what was done before the commit: data pools, paraphrase generation and validity judgement, throughput pilots on ≤ 100 AG News items with official labels, and a bf16 numerics check.

### 1.2 Research questions (from `plan.md`)
- **RQ1 (magnitude).** At n_cal = 250, α = 0.10, LAC: is the wording component of mean set size comparable to split sampling? H1a: σ_wording CI excludes 0 in ≥ 80% of zero-shot cells. H1b: σ_w/σ_s ∈ [1/3, 3] in ≥ 50% of cells. Refuted if the ratio is < 1/3 or > 3 in > 50% of cells.
- **RQ2 (floor).** H2a: split slope in [−0.65, −0.35] (n ≥ 100). H2b: wording slope > −0.15 (n ≥ 250) and σ_w(n_max) CI excludes 0 in ≥ 2/3 of cells. H2c: the W = 4 estimate is within ×[0.8, 1.25] of W = 30.
- **RQ3 (what wording is made of).** H3a: case-only changes flip predictions (≥ 1 model per type). H3b: dose-response ρ > 0. H3c: σ_w(VALID) < σ_w(ALL). The "definitional" reading is refuted if surface-only flip rates are ≥ 50% of valid-paraphrase flip rates in most cells.
- **RQ4 (contamination proxy, exploratory).** H4a: Min-K% Prob higher for AG News than BBC-2025. H4b: lower accuracy and larger σ_w on BBC-2025.
- **RQ5 (repeatability vs reproducibility on CPU).** H5a: identical fp32 runs give max |Δp| = 0. H5b: batch size changes in fp32 give |Δp| < 1e-4; bf16 changes conformal sets.

### 1.3 Deviations from the preregistration (all known ones)
1. **T1 NLI cross-encoders: no finished cell.** `cross-encoder/nli-deberta-v3-small` on AG News (1,200 items, all strings) has been scoring since 15:14 under CPU contention. The other nine T1 cells (both NLI models × the remaining datasets, and ModernBERT on all five) have not started. A separate BBC subset run (`run_nli_bbc_subset.sh`) was killed by an earlier worker at 16:02 as infeasible under contention (`logs/nli_bbc_subset.log`). All T1 statements in the plan (RQ1–RQ3, H3a for NLI) are therefore **untested**.
2. **T3 LLMs: one cell instead of ≥ 2 models × 5 datasets.** Only Qwen2.5-0.5B-Instruct on AG News is finished. Qwen2.5-0.5B on BBC-2025 is running. SmolLM2-360M and the optional Qwen2.5-1.5B have not started. No within-type model comparison is possible for T3, and all T3 numbers are a single cell.
3. **Cell-share thresholds were evaluated on 16 cells, 15 of them bi-encoders.** The "share of cells" criteria in RQ1–RQ3 therefore mostly describe T2. Holm correction was applied across the 16 available cells.
4. **RQ1 H1b sits exactly on a boundary.** 50% of cells have a ratio in [1/3, 3] (supported at the ≥ 50% threshold) and the other 50% have a ratio > 3. The refutation rule ("> 3 in > 50% of cells") is not triggered, but only because the share is exactly 0.50. We report this as *supported at the margin*: the data say wording is at least comparable to, and often larger than, sampling.
5. **RQ4 incomplete.** Min-K% Prob (H4a) has not run (no `results/contamination/` yet). H4b is evaluated for the three bi-encoders only, using point estimates. No CI on the AG−BBC accuracy difference was computed.
6. **RQ5 partial.** Only 2 of 4 identical fresh-process runs exist for NLI and bi-encoder, and 1 of 4 for the LLM (`llm_rep1` is running). None of the intermediate-precision conditions (batch size 1, 1 thread, bf16) has run.
7. **Post-queue steps were started early.** An earlier worker launched the repeatability and Min-K% steps before the scoring queues finished (`run_early_after.sh` at about 15:26, then `run_repeat_resume.sh` at 16:02, which skips finished outputs). The commands and arguments are the same as in `run_repeat.sh`/`run_after.sh`, but the runs share the CPU with the scoring jobs. This matters for timing, not for fp32 numerics, which is what RQ5 tests.
8. **Exploratory addition at report time.** `src/repeat_vs_main.py` compares repeat run 0 with the main-grid run of the same model on the same items and shared label strings. It is not preregistered and is labelled exploratory (§3.7).
9. **Crossing points n\* below the grid.** Where n\* (σ_split = σ_wording) is below 50, it is an extrapolation with slope −0.5 outside the measured grid, as the plan allows.
10. **v1 instruments not re-run** (Clef-flash, qwen3.5:9b and the 27B model need the author's Mac). The v1 numbers are only re-analysed from committed outputs (§3.8), so v2 tests the v1 *claims* with different, CPU-feasible instruments.

## 2. Data and methods

**Measurand (stated explicitly).** Mean prediction-set size (primary). Secondary: coverage, accuracy, ECE, singleton rate and worst-class coverage. These are computed for split-conformal prediction with method C at level α, calibrated on n exchangeable labelled items and evaluated on held-out items, *when the class concepts are verbalised by any meaning-preserving wording*. Under this definition the spread over valid wordings is a candidate for definitional uncertainty: VIM 2.27, called "intrinsic uncertainty" in GUM D.3.4. If the wording is instead part of the procedure, the same spread is a realization/procedure effect (GUM 3.3.2 b, i).

**Datasets.**

| dataset | content | classes | pool size |
|---|---|---|---|
| AG News | topics, 2004–05 news | 4 | 2,600 |
| BBC News 2024-10…2025-06 | same 4 AG classes; published after the release of every model used | 4 | 2,238 |
| SST-2 | sentiment | 2 | 2,600 |
| DBpedia-14 | encyclopedic categories | 14 | 1,400 |
| MASSIVE en-US | voice-assistant request scenarios | 18 | 2,600 |

BBC article texts are not redistributed; only links, labels and SHA-256 hashes are.

**Instruments (within-type only).**
- T2 bi-encoders: all-MiniLM-L6-v2, bge-small-en-v1.5, e5-base-v2. Probability = softmax(cos("text", "This text is about {label}.")/τ) with τ = 0.05.
- T3: Qwen2.5-0.5B-Instruct, scored by PMI label log-likelihood over the first 1,200 pool items.
- T4: logistic regression on the T2 embeddings, refit on 20 training bootstraps (BBC: trained on AG News).
- T1 NLI cross-encoders are planned but not yet finished.
- Everything ran in fp32 on CPU with torch 2.14.1+cpu and transformers 5.20.1.

**Wording population.** Each class has candidate wordings from generators G0–G7:
- G0: official name
- G1: case change only
- G2: WordNet synonym
- G3: WordNet hypernym
- G4: χ² keywords from the training split
- G5: a fixed short definition
- G6 and G7: two Qwen2.5-3B paraphrases
- the four v1 hand sets, for AG News and BBC

A candidate is VALID if the judges (all-mpnet-base-v2 and roberta-large-mnli, neither used as a classifier) find it discriminable and the researcher audit does not reject it. STRICT additionally requires judge cosine ≥ 0.5; ALL is every candidate. The main analysis uses W = 30 random VALID sets per dataset (seed 2026), the same sets for every model. Controls: identity, surface-only (case), nonsense pseudo-words, shuffled names.

**Design A (primary).** For each n, S = 200 random splits (test = 300, calibration = n of the rest) are crossed with the W sets. Two-way crossed random-effects method of moments gives σ²_w, σ²_s and σ²_ws+e, with negative estimates truncated at 0. 95% CIs come from a two-way bootstrap (wordings × splits, 1,000 draws). **Design B (v1 replication):** a fixed 300-item test set, with calibration bootstrapped from the rest.

**Conformal methods.** LAC, APS (randomised) and RAPS (λ = 0.01, k_reg = 1), at α ∈ {0.05, 0.10, 0.20}.

**Code.** `src/analyze.py`, `src/aggregate.py`, `src/report_tables.py`, `src/figures.py`, `src/template_factor.py`, `src/repeat_analysis.py`, `src/v1_reanalysis.py`.

## 3. Results

### 3.1 RQ1: magnitude at n_cal = 250 (Table 1 in `results/report_tables.md`)
H1a is **supported**: the σ_wording CI excludes 0 in 16/16 cells, and all Holm-corrected F tests are significant. H1b is **supported at the margin** (deviation 4): 50% of cells have a ratio in [1/3, 3], 50% have a ratio > 3, and none has a ratio < 1/3.

| type | dataset | model | σ_wording [95% CI] | σ_split | ratio [CI] | wording share [CI] |
|---|---|---|---|---|---|---|
| T2 | AG News | MiniLM-L6 | 0.341 [0.237, 0.426] | 0.105 | 3.25 [2.10, 4.19] | 0.80 [0.65, 0.86] |
| T2 | AG News | bge-small | 0.255 [0.175, 0.312] | 0.103 | 2.47 [1.67, 2.95] | 0.70 [0.55, 0.77] |
| T2 | AG News | e5-base | 0.399 [0.236, 0.528] | 0.084 | 4.75 [2.63, 6.29] | 0.88 [0.73, 0.92] |
| T2 | BBC 2024-25 | MiniLM-L6 | 0.199 [0.156, 0.237] | 0.100 | 1.99 [1.47, 2.42] | 0.58 [0.45, 0.67] |
| T2 | BBC 2024-25 | bge-small | 0.169 [0.131, 0.198] | 0.094 | 1.80 [1.32, 2.14] | 0.50 [0.37, 0.58] |
| T2 | BBC 2024-25 | e5-base | 0.212 [0.157, 0.256] | 0.087 | 2.45 [1.72, 2.99] | 0.65 [0.50, 0.74] |
| T2 | SST-2 | MiniLM-L6 | 0.109 [0.086, 0.126] | 0.022 | 4.85 [3.47, 5.43] | 0.80 [0.72, 0.84] |
| T2 | SST-2 | bge-small | 0.119 [0.097, 0.132] | 0.040 | 2.94 [2.32, 3.37] | 0.80 [0.73, 0.83] |
| T2 | SST-2 | e5-base | 0.135 [0.106, 0.153] | 0.038 | 3.57 [2.62, 4.22] | 0.84 [0.76, 0.88] |
| T2 | DBpedia-14 | MiniLM-L6 | 0.870 [0.643, 1.060] | 0.272 | 3.19 [2.16, 4.08] | 0.76 [0.63, 0.83] |
| T2 | DBpedia-14 | bge-small | 0.906 [0.595, 1.196] | 0.294 | 3.08 [1.97, 3.97] | 0.75 [0.58, 0.83] |
| T2 | DBpedia-14 | e5-base | 0.750 [0.583, 0.889] | 0.166 | 4.53 [3.23, 5.49] | 0.85 [0.77, 0.88] |
| T2 | MASSIVE-18 | MiniLM-L6 | 0.687 [0.548, 0.798] | 0.577 | 1.19 [0.93, 1.41] | 0.47 [0.36, 0.55] |
| T2 | MASSIVE-18 | bge-small | 0.796 [0.650, 0.910] | 0.573 | 1.39 [1.10, 1.65] | 0.56 [0.45, 0.63] |
| T2 | MASSIVE-18 | e5-base | 0.580 [0.434, 0.718] | 0.734 | 0.79 [0.59, 0.98] | 0.31 [0.20, 0.40] |
| T3 | AG News | Qwen2.5-0.5B | 0.453 [0.344, 0.537] | 0.071 | 6.38 [4.43, 7.51] | 0.92 [0.87, 0.94] |

Within T2, the median ratio is 2.94 and the median wording share is 0.75. MASSIVE-18 (18 classes, short requests) is the only dataset where sampling is of the same order as or larger than wording (share 0.31 to 0.56).

![Variance shares at n = 250](figures/fig01_variance_shares_n250.png)

*Figure 1. Variance shares of mean set size at n_cal = 250 (LAC, α = 0.10). Compare within a panel only.*

**Coverage.** The wording share of coverage variance is 0.00 in every cell (upper CI ≤ 0.02; Table 1b). This holds when calibration and test use the same wording, where conformal coverage is guaranteed. Mean coverage was within 0.007 of 1 − α across all cells × methods × α × n (Table 6).

**Methods and α (Table 6, Fig. 5).** For T2 the median wording share is 0.71–0.75 under LAC, 0.53–0.67 under APS and 0.55–0.68 under RAPS across α ∈ {0.05, 0.10, 0.20}. The dominance of wording is therefore not specific to LAC, though it is somewhat smaller with the adaptive scores. The T3 cell gives 0.83–0.92 across methods and α.

**Probability scale (Table 9, Fig. 16).** Changing τ from 0.05 to 0.02 or 0.1 moves the T2 wording share by at most 0.13 (MASSIVE e5: 0.31 becomes 0.44 at τ = 0.02); in most cells it moves by ≤ 0.03. For the T3 cell, raw log-likelihood instead of PMI gives 0.97 instead of 0.92.

![Method x alpha grid](figures/fig05_method_alpha_grid.png)

*Figure 5. Wording share across conformal method and α at n = 250.*

### 3.2 RQ2: the floor (Table 2, 2b, 2c)
- **H2b supported (16/16 cells).** The log-log slope of σ_wording over n ≥ 250 lies between −0.04 and 0.03. The CI at n_max excludes 0 in every cell (for example, AG MiniLM σ_w(n_max = 2,300) = 0.332 [0.227, 0.409]).
- **H2a not supported.** Split sampling falls in every cell, but only 50% of slopes fall in the preregistered window [−0.65, −0.35]. The range is −0.35 (SST-2 MiniLM) to −0.87 (DBpedia e5), and the T3 cell is −0.64 [−0.74, −0.53]. The steeper slopes appear near n_max (Fig. 3). One plausible reading, not tested here, is that the calibration set then exhausts the finite pool. Either way, sampling noise falls *at least* as fast as 1/√n.
- **H2c supported (16/16).** The mean of σ_w over 200 random W = 4 subsets is 0.87–0.96 of the W = 30 estimate, and the spread narrows as W grows (AG MiniLM: W = 4 [0.091, 0.591], W = 8 [0.167, 0.464], W = 16 [0.226, 0.411]). The v1-style estimator (population SD of 4 means) averages lower (AG MiniLM 0.280 vs 0.341), as expected for ddof = 0.
- **Crossing point.** n\* (σ_split = σ_wording) is between 3 and 31 for AG News, BBC, SST-2 and DBpedia (below the grid; extrapolated), and 78 to 302 for MASSIVE. v1 projected a crossing around n ≈ 2,000 for Clef. For these instruments, wording already dominates at any practical calibration size.
- **Design B (v1 replication, Table 2c)** gives the same picture. For AG MiniLM, σ_split falls from 0.253 (n = 50) to 0.034 (n = 2,300) while σ_wording stays at 0.345 → 0.340.

![Floor](figures/fig03_floor_normalised.png)

*Figure 3. σ_wording (orange) and σ_split (blue), each normalised to its value at n = 250.*

![W subsampling](figures/fig09_W_subsampling.png)

*Figure 9. How the σ_wording estimate and its spread depend on the number of paraphrase sets W.*

### 3.3 RQ3: what the wording component is made of (Table 3, 3b, 4b)
- **Surface-only control (H3a).** For all T2 bi-encoders, case-only changes flip 0.000 of predictions (Clopper-Pearson upper bound ≤ 0.003). The T2 encoders may well normalise case, so for T2 this control cannot tell non-invariance apart from definitional spread (pre-declared exploratory for T2). For the T3 cell, case-only changes flip 0.151 of predictions on average (max 0.210 [0.187, 0.234]). That is 0.35 of the valid-paraphrase flip rate (0.429), below the preregistered 0.50 refutation threshold. H3a is supported for T3 and untested for T1. Even so, about a third of the LLM's wording sensitivity is reproduced by changes that carry no meaning at all.
- **Dose-response (H3b).** T2: supported, ρ(semantic distance, total-variation shift) = 0.56 [0.51, 0.62] over 936 variant × model pairs, and 0.48–0.63 per dataset. T3: not supported, ρ = 0.28 [−0.06, 0.58] over 41 pairs. The mean TV shift is identical for valid and invalid variants (0.192 and 0.192), so this LLM's prediction shifts do not track meaning.
- **Validity filtering (H3c).** Supported: σ_w(VALID) < σ_w(ALL) in 0.88 of cells (exceptions: SST-2 MiniLM 0.109 vs 0.078; Qwen AG 0.453 vs 0.415). Filtering removes part, not all, of the spread. For DBpedia MiniLM, σ_w is 0.549 for STRICT, 0.870 for VALID and 1.706 for ALL.
- **Definitions vs names (Table 4b, exploratory).** The homogeneous G5 sets (every class given its short definition) give smaller mean sets than the official names in all 15 T2 cells. Examples: AG MiniLM 2.04 → 1.46, DBpedia MiniLM 3.50 → 1.37, MASSIVE e5 10.27 → 4.79. This is consistent with an underspecification component (Pecher et al.) in the *level*. Whether definitions also shrink the *spread* was not measured, because no population of alternative definitions was scored. For the T3 cell, definitions keep the set size (2.26 → 2.23) but lower accuracy (0.654 → 0.454).

![Dose-response](figures/fig04_dose_response.png)

*Figure 4. Prediction shift vs semantic distance of a single class label from the official name.*

![Surface vs valid](figures/fig15_surface_vs_valid.png)

*Figure 15. Flip rates from case-only changes vs valid paraphrases.*

### 3.4 Configuration shift: calibrate with the official names, deploy with another valid wording (Table 5)
The coverage guarantee holds only when calibration and deployment use the same wording. Mean coverage over the 30 valid sets ranges from 0.732 (SST-2 MiniLM; 0.77 of sets below 0.85) to 0.908 (MASSIVE MiniLM). For the T3 AG cell it is 0.770, with 0.70 of sets below 0.85 and a minimum of 0.514. The AG News bi-encoders give 0.877–0.900, with minima 0.787–0.818.

![Configuration shift](figures/fig13_config_shift_coverage.png)

*Figure 13. Coverage when calibrating on official names and deploying with each valid wording set (target 0.90).*

### 3.5 Controls and the supervised reference (Table 4, Table 8)
- **Identity control.** Flip rate 0.000 in all cells, so the pipeline is sound.
- **Nonsense labels.** Accuracy near chance, e.g. AG MiniLM 0.196 against chance 0.25, DBpedia 0.064–0.077 against 0.071.
- **Shuffled names.** Accuracy below chance in every cell, e.g. AG 0.074–0.093, SST-2 0.140–0.315 against 0.50. The instruments use label semantics.
- **T4 supervised probes (wording-free by construction).** σ_train-bootstrap is 0.003–0.050 and σ_split 0.024–0.100 at n = 250, so the budget is dominated by split sampling (Fig. 14).
- **AG → BBC real shift.** Probe accuracy is 0.850–0.863 on BBC 2024-25, against 0.875–0.898 on AG News (same probes, within T4).

![Controls](figures/fig10_controls_accuracy.png)

*Figure 10. Accuracy with official, valid, nonsense and shuffled labels (dotted: chance).*

### 3.6 RQ4: AG News vs BBC 2024-25 (Table 7; partial)
For the three bi-encoders, mean valid-set accuracy is lower on BBC in 3/3 models:

| model | AG News | BBC 2024-25 |
|---|---|---|
| MiniLM | 0.689 | 0.663 |
| bge-small | 0.700 | 0.692 |
| e5-base | 0.756 | 0.739 |

The differences are small, and no CI was computed. With official names, bge-small is *higher* on BBC (0.738 vs 0.708).

σ_wording is **smaller** on BBC in 3/3 models (0.199 vs 0.341, 0.169 vs 0.255, 0.212 vs 0.399), the opposite of H4b. Saturation (max p > 0.99) is ≤ 0.003 on both. For T2, the idea that contamination is what keeps AG wording effects small is not supported: if anything, AG shows *more* wording spread. H4a (Min-K%) and the LLM side are pending. The T3 AG cell has 0.273 of items with max p > 0.99.

### 3.7 RQ5: repeatability (partial; Table 10, Fig. 7)
- **Identical fresh-process runs (fp32, same settings).** 2 runs each so far for nli-deberta-v3-small and bge-small on 200 AG News items × 5 wordings: max |Δp| = 0.0, ICC = 1.000, Fleiss κ = 1.000, and 0.000 of conformal sets changed. H5a holds so far for T1 and T2. The LLM is pending.
- **Exploratory: repeat run 0 vs the main-grid run** (`results/repeat_rep0_vs_main.json`). Same model, dtype and nominal settings, but a different process with different batch composition: the main runs scored more strings and items. Here the results are **not** bit-identical. For bge-small, max |Δp| = 2.0e-6 (raw cosine 3.6e-7). For Qwen2.5-0.5B, max |Δp| = 1.8e-5 (raw PMI 8.9e-5). Neither changes any argmax. This anticipates the reproducibility/repeatability distinction (VIM 2.25 vs 2.21) that H5b is designed to test.

![Repeatability](figures/fig07_repeatability.png)

*Figure 7. Repeatability of identical fresh-process runs (partial: 2 runs).*

### 3.8 Re-analysis of the published v1 numbers (Table 12, Fig. 12)
- **Clef.** Published u_wording 0.047 (population SD, W = 4); with ddof = 1, 0.054 with χ² 95% CI [0.031, 0.201]. The published-style wording share (wording vs calibration only) is 0.34 with bootstrap CI [0.00, 0.84]. Under the v2 random-effects method the share is 0.23 [0.01, 0.37] and the ratio 0.69 [0.11, 1.07]. The share also depends on n: 0.03 (n = 50), 0.06 (100), 0.12 (200), 0.15 (250), 0.20 (400), 0.25 (500), with σ_w steady at 0.021–0.039 while σ_s falls 0.192 → 0.034.
- **qwen3.5.** Published 0.033; ddof = 1 0.038 [0.022, 0.142]; published-style share 0.56 [0.00, 0.95]; v2-method share 0.36 [0.01, 0.58]; ratio 1.18 [0.13, 1.59].
- **Coverage.** Under the v2 method the wording share of coverage is 0.01 [0.00, 0.03] for Clef and 0.08 [0.01, 0.12] for qwen3.5.

![v1 reanalysis](figures/fig12_v1_reanalysis.png)

*Figure 12. Published v1 numbers re-analysed from committed v1 outputs.*

### 3.9 Exploratory: the template factor (Table 11, T2 only)
Three hypothesis templates ("This text is about {}.", "{}", "Category: {}.") were crossed with W valid sets and 200 splits at n = 250:
- σ_wording exceeds σ_template in 12 of 15 bi-encoder cells.
- The exceptions are BBC bge (0.133 vs 0.193), BBC e5 (0.198 vs 0.244) and MASSIVE e5 (0.657 vs 1.349).
- The template × wording interaction is often of the same order as the template main effect (e.g. AG e5: 0.263 vs 0.173).

The template is a second, separate source of the same kind, and the wording estimate depends on which template it is measured under (σ_w within template, BBC e5: 0.212 / 0.571 / 0.201).

## 4. Verdicts on the v1 claims

| # | v1 claim (X Article / repo) | v2 evidence | Verdict |
|---|---|---|---|
| 1 | "Label wording explained about 32% of the variance, calibration sampling about 64%" (Clef; qwen3.5: 55%/39%) | Re-analysis: the published-style share is 0.34 with CI [0.00, 0.84] (qwen3.5: 0.56 [0.00, 0.95]). The share depends on n (0.03 → 0.25 over n = 50…500), and the v2 method gives 0.23 [0.01, 0.37]. On v2 instruments the share is 0.31–0.88 (T2) and 0.92 (T3). | **Refuted as a stable quantity.** The specific 32%/64% split is not a property of the instrument: its CI spans almost the whole range and it changes with n. The qualitative point that wording is a major component is confirmed (see 2). |
| 2 | "For zero-shot instruments, label wording is as large as calibration sampling at 250 calibration items" (ratio 0.71 Clef, 1.18 qwen3.5) | H1a 16/16; H1b at the margin (50% in [1/3, 3], 50% > 3); median T2 ratio 2.94; T3 6.38 | **Confirmed**, and stronger than claimed: on v2 instruments wording is usually *larger* than split sampling. One dataset (MASSIVE) is the exception, with sampling of the same order. |
| 3 | "Calibration noise fell 0.23 → 0.04 from n = 50 to 500", roughly as 1/√n | Falls in every cell; slopes −0.35 to −0.87 (H2a window met in only 50%); Design B AG MiniLM 0.253 → 0.034 | **Confirmed** that it falls; the √n rate is **weakened**: the decline is often steeper, especially near the pool limit. |
| 4 | "The wording part stalled near 0.02" (a floor) | Wording slope −0.04…0.03 with CI > 0 at n_max in 16/16; robust to W (0.87–0.96) | The **existence of a floor is confirmed** under a design that also varies the test set. The value 0.02 is instrument-specific (v2: 0.11–0.91), and the published CI [0.018, 0.021] was too narrow, since it held wordings fixed (χ² CI for W = 4: [0.031, 0.201]). |
| 5 | Crossing "somewhere around n = 2,000" | n\* = 3–31 for 4 datasets, 78–302 for MASSIVE | **Refuted for v2 instruments.** Wording overtakes sampling far earlier. The 2,000 figure was specific to Clef's small wording SD. |
| 6 | The floor "is the pattern a definitional floor would produce"; wording as a candidate for VIM 2.27 definitional uncertainty (v1 said "suggestive") | Floor confirmed, but any systematic effect gives a floor. T3: case-only changes reproduce 0.35 of the valid-paraphrase flip rate. T2: dose-response ρ = 0.56, validity filtering shrinks σ_w; T3 dose-response not significant. Templates add a comparable component. Definitions shrink set-size levels in 15/15 T2 cells. | **Weakened.** The preregistered refutation threshold (≥ 0.5) was not reached, so not refuted. But the floor test only diagnoses *systematic vs random*. In the LLM a substantial part is instrument non-invariance, and part of the T2 spread tracks how far a paraphrase drifts in meaning. Wording spread is better described as "systematic, partly definitional, partly instrument non-invariance". |
| 7 | "Wording mostly changes how big the sets are, less whether they contain the right answer" (coverage share 3%) | Coverage wording share 0.00 in all 16 cells (matched wording); re-analysis 0.01 for Clef | **Confirmed**, but only when calibration and deployment use the same wording (see 8). |
| 8 | Calibrate on one wording, deploy on another: "mean coverage barely moved", the range widened (Clef 0.855 → 0.863) | Mean coverage over valid sets 0.732–0.908; up to 0.77 of sets below 0.85; T3 AG 0.770 (min 0.514) | **Refuted** for "mean coverage barely moves" (it drops well below target in several cells). The wider range is confirmed. |
| 9 | "Repeatability can be exactly zero for a deterministic classifier … on my stack" | CPU, fp32: 2 identical fresh-process runs give max \|Δp\| = 0.0 for NLI and bi-encoder (LLM pending). A different batch composition gives 2e-6 to 2e-5, with no argmax change. | **Confirmed (partially tested)** for repeatability conditions. Reproducibility under changed conditions is not bit-exact. The full RQ5 is pending. |
| 10 | AG News contamination "could explain the saturation" (raised as a gap) | T2: saturation ≈ 0 on both AG and BBC; σ_w *smaller* on BBC; accuracy only slightly lower on BBC. T3 and Min-K% pending | **Weakened for T2** (no sign that newer texts make wording effects larger). For LLMs it is **open**. |
| 11 | "Determinism didn't remove the uncertainty. It moved it into choices I make as the person posing the question." | Deterministic zero-shot instruments: 0 repeatability spread, wording share 0.31–0.92. Supervised probes (no wording): split sampling dominates. Template adds a further choice component. | **Confirmed.** |

Not re-tested in v2: the aleatoric/epistemic split (H4 of v1), the Clef name/definition swap, the RSS-vs-Monte-Carlo check and the synthetic typo shift. v1's "9B vs 27B made no difference" is not tested either; Qwen2.5-1.5B is pending.

## 5. Limitations
1. **Interim and unbalanced.** 15 of 16 zero-shot cells are bi-encoders. There is no NLI result, and the LLM evidence is one cell (one model, one dataset). Claims about "zero-shot instruments" in general are mostly claims about small bi-encoders.
2. **Small models only** (≤ 0.5B LLM, ≤ 110M-class encoders). Frontier LLMs may show smaller wording effects and more saturation.
3. **Paraphrase population is constructed** (automatic generators, one LLM paraphraser, fixed definitions), judged by models and one researcher, with no human raters. σ_wording is defined relative to this population.
4. **Case-only control is uninformative for T2**, since flips are 0, so for T2 we cannot separate non-invariance from definitional spread with this control.
5. **Finite pools.** Large-n behaviour is affected by pool exhaustion, which is one plausible reason for the steep split slopes. BBC and DBpedia pools are smaller (n_max 1,938 and 1,100).
6. **RQ4 is a proxy comparison.** AG and BBC differ in source and style as well as date, there are no accuracy-difference CIs yet, and Min-K% has not run.
7. **CPU stack only.** Repeatability was tested on one box and does not transfer to other hardware or libraries. The v1 Mac instruments were not re-run.
8. **English only.**
9. **The template factor is exploratory and T2 only.** The main design holds the template fixed, so σ_wording is conditional on the template "This text is about {}." for T2.

## 6. A short philosophical discussion
v1 asked whether label wording is part of the measurand. The v2 data make the question sharper, but they cannot settle it, because the answer is a decision about the measurand and not an empirical finding. If the measurand is "set size given the class *concepts*", every valid wording realises the same definition, and the spread is the price of the definition's limited detail (VIM 2.27). If the measurand is "set size given *this* label string", the spread is an instrument non-invariance under a change of procedure (GUM 3.3.2 b, i). The floor test cannot tell these apart: anything systematic refuses to average out. What the data do show is that the spread is *graded*. It grows with the semantic distance of a paraphrase (bi-encoders, ρ = 0.56). Validity filtering removes part of it. Meaningless case changes reproduce a third of it in the small LLM. Swapping a name for a definition moves the level a lot. The wording component is therefore not one thing: it is a mixture of definitional looseness and instrument non-invariance, in proportions that depend on the instrument.

That does not make the budget useless. It makes it relative to a choice the person posing the question must state. Determinism does not remove uncertainty; it moves it from the run into the specification. A metrologically honest report of a zero-shot classifier's conformal set size should state the measurand (concept or string), the wording population and the template, and report the wording component next to sampling. Otherwise the conformal guarantee holds for a measurand nobody specified. The configuration-shift result (coverage 0.732–0.908 when wording changes after calibration) shows what is at stake: the guarantee belongs to the wording it was calibrated on.

## 7. Pending runs
These jobs were still running at 16:23 Istanbul time; none was stopped. After they finish, rerun `analyze_pending.sh`, `src/aggregate.py`, `src/report_tables.py`, `src/repeat_analysis.py` and `src/figures.py`.

| job | current step | adds |
|---|---|---|
| `run_queue_a.sh` | `score_nli.py cross-encoder/nli-deberta-v3-small agnews --items 1200 --strings all` (since 15:14) | First T1 NLI cell, then deberta on BBC, SST-2, DBpedia, MASSIVE, and ModernBERT-base-zeroshot-v2.0 on all five. Gives RQ1–RQ3 for T1 (incl. H3a and dose-response for NLI) and the T1 side of RQ4. |
| `run_queue_b.sh` | `score_llm.py Qwen/Qwen2.5-0.5B-Instruct bbc2025 --items 1200` (since 15:40) | Second T3 dataset (RQ4 for the LLM: accuracy, σ_w, saturation on post-cutoff news), then Qwen on SST-2, MASSIVE, DBpedia, and SmolLM2-360M on all five. Gives the first within-T3 model comparison. |
| `run_repeat_resume.sh` | `llm_rep1` (Qwen, 100 items) running; nli/bienc rep0–1 and llm rep0 done | Repeat runs 2–3 for all three instruments; intermediate-precision conditions (fp32 bs 1, 1 thread, bf16 bs 64, bf16 bs 1) for each; then Min-K% Prob for Qwen2.5-0.5B and SmolLM2 (500 items). Completes RQ5 (H5a, H5b) and RQ4 H4a. |
| `run_after.sh` | waiting for `QUEUE_B_DONE` | Min-K% (again; it does not skip existing outputs), then after `QUEUE_A_DONE` `run_repeat.sh` (re-runs the repeat set; it does not skip existing files), then Qwen2.5-1.5B on AG/BBC (600 items, valid strings) and its Min-K%. Gives a second, larger T3 model on the two news datasets. |

## 8. Reproducibility
Raw per-item scores (`raw/*.npz`), per-cell analyses (`results/cells/`), label candidates and validity judgements (`data/labels_*.json`), wording sets (`data/wordsets_*.json`) and every script are in `v2/`. BBC-2025 texts are replaced by hashes (`data/bbc2025_public.json`); rebuild them with `src/prep_data.py bbc`. Progress log: `PROGRESS.md`.

## References
- JCGM 200:2012, International Vocabulary of Metrology (VIM 3), 2.27 definitional uncertainty: https://jcgm.bipm.org/vim/en/2.27.html
- JCGM 100:2008, Evaluation of measurement data: Guide to the expression of uncertainty in measurement (GUM): https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf
- v1 article: https://x.com/mehanixs/status/2111363921655255405 ; repo: https://github.com/detasar/gum-uncertainty-budget-textclf
- Petri & Mari (2026), Uncertainty Evaluation: Bridging Machine Learning and Measurement Science. https://arxiv.org/abs/2610.18066
- Sadinle, Lei & Wasserman (2019), Least Ambiguous Set-Valued Classifiers with Bounded Error Levels. https://arxiv.org/abs/1609.00451
- Romano, Sesia & Candès (2020), Classification with Valid and Adaptive Coverage. https://arxiv.org/abs/2006.02544
- Angelopoulos, Bates, Malik et al. (2021), Uncertainty Sets for Image Classifiers using Conformal Prediction. https://arxiv.org/abs/2009.14193
- Angelopoulos & Bates (2021), A Gentle Introduction to Conformal Prediction. https://arxiv.org/abs/2107.07511
- Zhu, Yu & Li (2026), Conformalized Large Language Models under Configuration Shift. https://arxiv.org/abs/2608.01460
- Xin et al. (2026), Conformal Prediction with Paraphrase-Aware Scoring for LLM Uncertainty Quantification. https://arxiv.org/abs/2610.04239
- Kumar et al. (2023), Conformal Prediction with LLMs for Multi-Choice QA. https://arxiv.org/abs/2305.18404
- Quach et al. (2023), Conformal Language Modeling. https://arxiv.org/abs/2306.10193
- Su et al. (2024), API Is Enough: Conformal Prediction for LLMs Without Logit-Access. https://arxiv.org/abs/2403.01216
- Ye et al. (2024), Benchmarking LLMs via Uncertainty Quantification. https://arxiv.org/abs/2401.12794
- Stutz et al. (2023), Conformal prediction under ambiguous ground truth. https://arxiv.org/abs/2307.09302
- Sclar, Choi & Tsvetkov (2023), Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design. https://arxiv.org/abs/2310.11324
- Mizrahi et al. (2023), State of What Art? A Call for Multi-Prompt LLM Evaluation. https://arxiv.org/abs/2401.00595
- Zhuo et al. (2024), ProSA. https://arxiv.org/abs/2410.12405
- Chatterjee et al. (2024), POSIX: A Prompt Sensitivity Index. https://arxiv.org/abs/2410.02185
- Polo et al. (2024), Efficient multi-prompt evaluation of LLMs. https://arxiv.org/abs/2405.17202
- Weber, Bruni & Hupkes (2023), Mind the instructions. https://arxiv.org/abs/2310.13486
- Gonen et al. (2022), Demystifying Prompts in Language Models via Perplexity Estimation. https://arxiv.org/abs/2212.04037
- Pecher, Spiegel & Belanec (2026), Revisiting Prompt Sensitivity in LLMs for Text Classification: The Role of Prompt Underspecification. https://arxiv.org/abs/2602.04297
- Sun, Xu & Shi (2026), Type-Safe Is Not Error-Free. https://arxiv.org/abs/2609.26758
- Reiter, Kern & Miasnikov (2026), Reliable but Design-Sensitive: Instrument Uncertainty in LLM Annotation. https://arxiv.org/abs/2609.35824
- Holtzman et al. (2021), Surface Form Competition. https://arxiv.org/abs/2104.08315
- Zhao et al. (2021), Calibrate Before Use. https://arxiv.org/abs/2102.09690
- Zheng et al. (2023), Large Language Models Are Not Robust Multiple Choice Selectors. https://arxiv.org/abs/2309.03882
- Yin, Hay & Roth (2019), Benchmarking Zero-shot Text Classification. https://arxiv.org/abs/1909.00161
- Laurer et al. (2023), Building Efficient Universal Classifiers with Natural Language Inference. https://arxiv.org/abs/2312.17543
- Bouthillier et al. (2021), Accounting for Variance in Machine Learning Benchmarks. https://arxiv.org/abs/2103.03098
- Hassanzadeh & Subramanian (2026), Same Query, Different Answer. https://arxiv.org/abs/2610.13823
- Qin, Du & Zhang (2026), Vosti: Deterministic LLM Inference. https://arxiv.org/abs/2609.38981
- Shi et al. (2023), Detecting Pretraining Data from LLMs (Min-K% Prob). https://arxiv.org/abs/2310.16789
- Sainz et al. (2023), NLP Evaluation in trouble. https://arxiv.org/abs/2310.18018
- Li et al. (2023), LatestEval. https://arxiv.org/abs/2312.12343
- Sharma, Duncan & Caron (2026), TypedBench. https://arxiv.org/abs/2610.11392
- Plank (2022), The "Problem" of Human Label Variation. https://arxiv.org/abs/2211.02570
- Zhang, Zhao & LeCun (2015), Character-level Convolutional Networks for Text Classification (AG News, DBpedia). https://arxiv.org/abs/1509.01626
- Socher et al. (2013), Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank (SST-2). https://aclanthology.org/D13-1170/
- FitzGerald et al. (2022), MASSIVE. https://arxiv.org/abs/2204.08582
- BBC News 2024–25 source: https://huggingface.co/datasets/RealTimeData/bbc_news_alltime
