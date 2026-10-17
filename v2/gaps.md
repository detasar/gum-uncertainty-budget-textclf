# v2 red-team gap analysis of the published v1 experiment

*Written 2026-10-17 (Istanbul), before any v2 classifier was run. Target: the published X Article ("My classifier never changed its mind. The uncertainty was still there."), `experiment.md`, `results.json`, `repeatability.json` and the public repo `detasar/gum-uncertainty-budget-textclf` (main @ ef9cf46).*

Severity scale: **Critical** = a reviewer could reject or a critic could "blow up" a headline claim with it; **High** = materially weakens a claim; **Medium** = needs a caveat or an extra analysis; **Low** = cosmetic or for completeness.
For each gap: what the objection is, why it bites, and what v2 does (or how the limitation is stated if v2 cannot fix it).

## A. Claims that can be attacked directly

### A1. "Label wording explained about 32% of the variance, calibration sampling about 64%" — Critical
- **Objection.** This is not a variance decomposition. The shares are u_i² / Σu_j² of a root-sum-square budget, where u(wording) is the population SD (ddof = 0) of **four** hand-picked set sizes, each computed on the **same** 250 calibration and 250 test items, and u(cal) is a bootstrap SD for one wording. "Explained variance" suggests an ANOVA/random-effects model that was never fitted. There is no CI on the shares, and no interaction term. The wording SD also contains part of the calibration noise (the v1 Monte Carlo already showed the RSS is ~11% conservative), so the components are not separable as stated.
- **Why it bites.** With W = 4 alternatives, a 95% CI for an SD spans roughly ×0.57 to ×3.7 of the point estimate (χ² with 3 df), so "32%" is compatible with almost anything between "negligible" and "dominant". The share also changes with n_cal, because u(cal) shrinks with n while the wording part does not: a share is a property of a chosen n, not of the instrument.
- **v2.** Crossed random-effects design: wording sets drawn at random from a documented, validated paraphrase population (W = 30 per dataset × model), crossed with S random calibration/test splits. Variance components σ²_wording, σ²_split, σ²_wording×split estimated by method of moments (two-way crossed ANOVA, EMS), with 95% CIs from a two-way (wordings × splits) bootstrap. Shares reported **as a function of n**, never as a single number. The v1 data are re-analysed the same way to put CIs on the published 32%/64%.

### A2. "The wording part stalled near 0.02" (the "floor") — Critical
- **Objection 1 (by construction).** In v1 the calibration sets were bootstrap draws from one fixed 500-item pool and the test set was fixed. As n grows, each wording's mean set size converges to its value on *that pool and that test set*; the SD of four fixed numbers is then a constant. A plateau was guaranteed by the design; it does not by itself show anything "structural".
- **Objection 2 (fixed effects, not random).** The bootstrap CI [0.018, 0.021] at n = 500 only resamples calibration draws. It treats the 4 wordings and the 250 test items as fixed. It is not a CI for "wording uncertainty" in any population sense.
- **Objection 3 (systematic ≠ definitional).** *Any* systematic effect (a fixed bias between two instrument settings) fails to shrink with n. Not shrinking separates random from systematic; it does not separate "definitional" from "instrument non-invariance".
- **Objection 4 (number of paraphrases / models).** With 4 wordings and 1 model the floor value is a single draw.
- **v2.** (i) Repeated random splits of a larger pool (up to 2,600 items) so both calibration and test sampling vary; (ii) floor curves σ_wording(n) for n = 50…2,000 with bootstrap CIs over wordings *and* splits; (iii) W-subsampling (W = 4, 8, 16, 30) to show how the estimate and its CI depend on the number of paraphrases; (iv) several models per instrument type and 5 datasets; (v) a surface-only control (case changes that cannot alter any definition) to separate instrument non-invariance from definitional spread.

### A3. "Calibration noise fell 0.23 → 0.04 from n = 50 to 500" — Medium
- **Objection.** Bootstrap from a fixed pool, one test set, one wording. The n = 50 value is inflated by the coarse conformal quantile at small n (the ⌈(n+1)(1−α)⌉/n order statistic jumps) and by empty sets; the published log-log slope (−0.74) is steeper than −0.5 partly for that reason. Fine as a description, weak as a scaling law.
- **v2.** Same quantity under repeated random splits, 5 datasets, 3 instrument types, many models, n = 50…2,000, with bands; the slope is fitted only for n ≥ 100 and reported with CIs.

### A4. "A deterministic classifier was identical across 4 runs" — Medium
- **Objection.** True and well measured, but (a) it is a property of one stack (MLX on Apple Silicon), not of "the model"; (b) the four runs were in one session/process for most items; (c) κ and ICC are degenerate when nothing varies (acknowledged in v1); (d) repeatability (VIM 2.21) was conflated in places with reproducibility (VIM 2.25): different batch sizes, thread counts, dtypes or hardware were not tested.
- **v2.** On the CPU box: 4 identical runs in fresh processes for one model of each type, **plus** intermediate-precision conditions (batch size 1 vs 64, 1 vs 8 threads, bf16 vs fp32) on the same items; report max |Δp|, ICC(2,1)/(3,1), Fleiss κ, share of conformal sets that change. The Mac result is not re-tested (Davut's Mac needs per-command approval and he is asleep); stated as a limitation.

### A5. "'Definitional uncertainty' is the GUM/VIM metrology term" — High
- **What the documents say (checked 2026-10-17).** VIM 2.27 (JCGM 200:2012): "component of measurement uncertainty resulting from the finite amount of detail in the definition of a measurand"; Note 1: practical minimum; Note 2: "Any change in the descriptive detail leads to another definitional uncertainty"; Note 3: in the GUM (ISO/IEC Guide 98-3:2008, D.3.4) the concept is termed **"intrinsic uncertainty"**. So "definitional uncertainty" is the **VIM** term; the GUM's word is "intrinsic". Saying "GUM/VIM term" is slightly off and should say "VIM term (GUM: intrinsic uncertainty, D.3.4)".
- **The deeper problem.** GUM 3.3.2 lists separately (a) *incomplete definition of the measurand*, (b) *imperfect realization of the definition of the measurand* and (i) *approximations and assumptions in the measurement method and procedure*. Label wording is an input to the instrument. Whether its effect is (a) or (b)/(i) depends on how the measurand is defined:
  - If the measurand is "set size of model M on task T, where T's classes are given only by their concepts", then several wordings are all compatible with the definition and the spread across *meaning-equivalent* wordings is definitional in the VIM sense, **but only if the paraphrases really are equivalent**.
  - If the wording is part of the measurement procedure, the spread is a procedure/realization effect (GUM 3.3.2 (b)/(i)), i.e. the instrument's non-invariance, and calling it "definitional" is a category error.
  - v1 had no paraphrase-validity check, and at least one of its wordings ("international politics and conflicts" for World) arguably narrows the concept rather than restating it.
- **v2.** (i) The report states the measurand explicitly; (ii) paraphrase validity is checked with independent judges; (iii) the surface-only control estimates the pure realization/non-invariance part; (iv) the report reframes the claim: the floor test (does not shrink with n) diagnoses *systematic vs random*, not *definitional vs instrumental*.

### A6. "One dataset, AG News, conformal prediction sets" — Critical (external validity)
- **Objection.** One easy, short, 4-class, balanced, English topic dataset from 2004–2005 news. AG News is almost certainly in the pretraining data of the LLMs and of NLI/embedding models (TypedBench, arXiv:2610.11392, raises exactly this concern for decision models; Sainz et al. 2023, arXiv:2310.18018, argue contamination must be measured per benchmark). Contamination can inflate zero-shot accuracy, saturate probabilities and shrink wording effects.
- **v2.** Five datasets: AG News (topic, old), **BBC News Oct 2024–Jun 2025** mapped to the same 4 AG classes (topic, published after the release of every model used; same label space so the wording sets transfer exactly), SST-2 (binary sentiment), DBpedia-14 (14-class encyclopedic category), MASSIVE en-US scenarios (18-class voice-assistant requests; no banking/finance intents). Contamination proxy: Min-K% Prob (Shi et al. 2023, arXiv:2310.16789) of the texts under the LLMs, AG News vs BBC-2025 (same genre, different dates).

## B. Design gaps

### B1. Few, hand-picked label paraphrases; no validity check; no semantic-distance control — Critical
- v1 used 4 sets chosen by the author. A Type B "uniform over alternatives" depends entirely on which alternatives are included; four is not a sample from any stated population.
- **v2.** A documented generator per class: official name; surface variant (case only); WordNet synonym and hypernym (automatic); top-3 χ² keywords from the training split (automatic, data-derived); a short definition fixed in code before scoring; two alternatives from Qwen2.5-3B-Instruct with a fixed prompt (automatic). Validity judged by models **not used as classifiers** (all-mpnet-base-v2 cosine, roberta-large-mnli bidirectional entailment) plus a discriminability criterion and a researcher audit written down before any scoring. Wording sets for the random-effects analysis are random draws (one valid variant per class). Semantic distance (1 − judge cosine to the official name) is the dose in a dose-response analysis. Results are reported for valid-only, strict, and all-candidates populations.

### B2. Variance-decomposition method — Critical
- No model was stated (it was a GUM RSS, not ANOVA or REML), no interactions, no CIs, components not independent.
- **v2.** Two-way crossed random-effects model Y_ws = μ + a_w + b_s + (ab)_ws per dataset × model × n × α × method, MoM estimates (negative estimates truncated at 0 and reported), two-way bootstrap CIs; GUM-S1-style Monte Carlo of the full pipeline as the reference for the combined uncertainty; the RSS is compared with it.

### B3. Metric choice — High
- Set size and coverage were the only measurands; for saturated LLMs set size ≈ 1 and coverage ≈ accuracy (degenerate budget). Accuracy and ECE were context only. Coverage is guaranteed on average by conformal prediction, so its "budget" is largely the binomial sampling spread around 1 − α.
- **v2.** Primary: mean set size (efficiency, the quantity wording actually moves). Secondary: coverage, accuracy, ECE (15 bins, of the base probabilities), singleton rate, worst-class coverage (class-conditional gap). Wording effects reported for each.

### B4. Alpha level — Medium
- Only α = 0.10.
- **v2.** α ∈ {0.05, 0.10, 0.20} for everything; α = 0.01 only where n_cal ≥ 500 (the quantile needs ≥ 99 calibration points to be non-trivial).

### B5. Conformal method and exchangeability — High
- Only split conformal with the LAC score (Sadinle et al. 2019). LAC produces empty sets and the smallest average sets; APS (Romano et al. 2020) and RAPS (Angelopoulos et al. 2021) behave differently, and the wording effect could be method-specific. Exchangeability between calibration and test was assumed but not argued; with wording *changes between calibration and deployment* it is violated by design (configuration shift, Zhu et al. arXiv:2608.01460).
- **v2.** LAC, APS (randomized, as in Romano et al.), RAPS (λ = 0.01, k_reg = 1 fixed a priori; no tuning on test). Calibration and test are random partitions of one i.i.d. pool, so exchangeability holds within a wording; the cross-wording (calibrate on w_i, test on w_j) coverage is reported separately as an explicit exchangeability violation. BBC-2025 is a fresh sample but internally exchangeable.

### B6. Model families must not be ranked against each other — High
- v1 already warned "not a leaderboard", but the published budgets sit side by side and invite cross-type comparison.
- **v2.** Four instrument types, each with ≥ 2 models: (T1) zero-shot NLI cross-encoders, (T2) zero-shot bi-encoders (embedding similarity), (T3) small instruction LLMs scored by label log-likelihood (PMI), (T4) supervised embedding + logistic regression (the wording-free reference). Every figure and table compares **within** a type; cross-type statements are restricted to the *shape* of the budget (which component dominates), never levels. This is stated in every caption.

### B7. Seeds, bootstrap CIs, multiple comparisons — Medium
- v1: 15 MLP seeds, 15 training bootstraps, 1,000-draw percentile bootstraps, no correction for many comparisons, hypotheses stated during analysis.
- **v2.** Preregistered RQs with explicit refutation criteria committed to git before the runs; ≥ 200 random splits per n; two-way bootstrap (wordings × splits) with 1,000 draws for variance components; Holm correction across the confirmatory tests of each RQ; everything else labelled exploratory. Supervised probe: logistic regression is convex, so seed variance is zero by construction; training-sample bootstrap (20 refits) instead.

### B8. Interactions and correlated components — Medium
- Wording × model, wording × calibration split and wording × dataset interactions were not estimated.
- **v2.** σ²_wording×split is estimated; wording × model is shown via the model × paraphrase heatmaps and via the correlation of per-wording effects across models of the same type.

### B9. Negative and positive controls — High
- v1 had none.
- **v2.** (a) Identity control: the same wording repeated (must give σ_wording = 0 exactly for deterministic scorers; a pipeline check). (b) Surface-only control: case changes only. (c) Nonsense labels (seeded pseudo-words): what wording variance looks like when labels carry no meaning; accuracy should be near chance. (d) Shuffled labels (derangements of the official names): accuracy should fall below chance, showing the instruments use label semantics. (e) Supervised probe: wording does not enter at all (σ_wording ≡ 0), the structural negative control for "uncertainty that more data removes".

### B10. Distribution shift was synthetic only — Medium
- Typos and word drops gave a null result.
- **v2.** A real temporal/source shift: the AG-News-trained supervised probe evaluated on BBC-2025 (same label space, 20 years later), with calibration on AG and on BBC.

### B11. Contamination — High
- See A6. A proxy only: neither the LLM training data nor the NLI/embedding training data are public.
- **v2.** Min-K% Prob (k = 20%) of each text under each LLM; AG vs BBC-2025. Stated as a proxy: genre and style also differ between the two sources.

### B12. Question phrasing / template — Medium
- 3 instructions in v1 for one model.
- **v2.** Template is held fixed in the main design (stated). A small template factor (3 hypothesis templates) is run for the bi-encoders on all datasets (cheap) and reported as exploratory.

### B13. Probability scale and temperature — Medium
- LAC sets depend on the probability scale, not only on the ranking. For bi-encoders the softmax temperature is a free choice.
- **v2.** Fixed τ = 0.05 for bi-encoders set a priori; sensitivity τ ∈ {0.02, 0.1} reported. NLI: softmax over entailment logits (the Hugging Face zero-shot convention). LLM: softmax over PMI scores (raw log-likelihood as a sensitivity analysis).

## C. Literature we did not engage with (all links checked 2026-10-17)
- Prompt/format sensitivity: Sclar et al., ICLR 2024 (FormatSpread), arXiv:2310.11324; Mizrahi et al., TACL 2024, "State of What Art? A Call for Multi-Prompt LLM Evaluation", arXiv:2401.00595; Zhuo et al. 2024, ProSA, arXiv:2410.12405; Chatterjee et al. 2024, POSIX, arXiv:2410.02185; Polo et al. 2024, PromptEval, arXiv:2405.17202; Weber, Bruni, Hupkes 2023, "Mind the instructions", arXiv:2310.13486 (factor analysis of prompt-design choices and their interactions — closest methodological precedent for a variance decomposition).
- Verbalizer/label-name effects: Holtzman et al. 2021, surface form competition, arXiv:2104.08315; Zhao et al. 2021, "Calibrate Before Use", arXiv:2102.09690; Zheng et al. 2024, MCQ selection bias, arXiv:2309.03882.
- Zero-shot NLI classification: Yin, Hay, Roth 2019, arXiv:1909.00161; Laurer et al. 2023, arXiv:2312.17543.
- Conformal prediction: Sadinle, Lei, Wasserman 2019 (LAC), arXiv:1609.00451; Romano, Sesia, Candès 2020 (APS), arXiv:2006.02544; Angelopoulos et al. 2021 (RAPS), arXiv:2009.14193; Angelopoulos & Bates tutorial, arXiv:2107.07511. Conformal + LLMs: Kumar et al. 2023, arXiv:2305.18404; Quach et al. 2024, Conformal Language Modeling, arXiv:2306.10193; Ye et al. 2024, arXiv:2401.12794; Su et al. 2024, arXiv:2403.01216. Ambiguous ground truth: Stutz et al. 2023, arXiv:2307.09302.
- Label variation as a property of the task: Plank 2022, arXiv:2211.02570.
- Benchmark variance accounting: Bouthillier et al. 2021, arXiv:2103.03098.
- Contamination: Shi et al. 2023 (Min-K% Prob), arXiv:2310.16789; Sainz et al. 2023, arXiv:2310.18018; TypedBench, arXiv:2610.11392.

## D. Reproducibility gaps — Medium
- v1 model runs need a Mac (MLX, Ollama) and an 8-bit Clef build; the analysis is reproducible from committed outputs, the measurements are not on commodity hardware. Ollama digests are recorded, but GGUF quantisation and llama.cpp versions change numerics.
- **v2.** Everything runs on an 8-core CPU box with pinned versions (torch, transformers, sentence-transformers), Hugging Face model IDs, seeds and label sets committed; raw per-item scores committed (npz); a single `make`-style runner. Third-party texts are not redistributed where licensing is unclear (BBC: links and hashes are committed; the text is re-downloadable from the public Hugging Face dataset).

## E. What v2 cannot fix (stated as limitations)
1. v1 instruments (Clef-flash, qwen3.5:9b, Qwen3.8-27B) are not re-run: they need the Mac. v2 tests the *claims* with different, CPU-feasible instruments; it does not re-measure the published numbers on the same instruments (except the v1 re-analysis from committed outputs).
2. Models are small (≤ 1.7B LLMs, ≤ 400M encoders). Effects for frontier LLMs may differ, likely smaller wording effects but more saturation.
3. English only. A second language (MASSIVE tr-TR) is listed as next step.
4. The paraphrase population is still a construction: automatic generators + an LLM + a fixed definition per class. It is documented and validated, not "the" population of all wordings.
5. Contamination is measured by a proxy, not by access to training data.
6. Human paraphrase-validity ratings were not collected (no annotators); judges are models + one researcher audit.
