# v2 preregistration: research questions, hypotheses, refutation criteria, analysis plan

*Written 2026-10-17 between 13:45 and 14:18 Istanbul (UTC+3) and committed to git on branch `v2-comprehensive` BEFORE any v2 classifier output for the confirmatory analyses was produced. The commit hash and timestamp are the preregistration record.*

**What was done before this commit (disclosed):** dataset pools fixed (`src/prep_data.py`, seed 0); label paraphrase candidates generated and validity-judged (`src/labels.py`; judges are not classifiers); throughput pilots on ≤ 100 AG News items with the official labels only (to choose dtype and batch size); a numeric check showing that bf16 inference with padded batches changes scores (see RQ5). No variance component, floor, dose-response, coverage or accuracy result had been computed for v2 at commit time.

## Measurand (stated explicitly, because the "definitional" question depends on it)
For a dataset D with K classes defined by their **concepts** (the official class names plus the dataset documentation), an instrument M (a scoring model plus a fixed prompt/hypothesis template), a conformal method C and a miscoverage level α:

> Y = mean prediction-set size (primary) — and coverage, accuracy, ECE, singleton rate, worst-class coverage (secondary) — of split-conformal prediction with method C at level α, calibrated on n exchangeable labelled items from D and evaluated on held-out items from D, **when the class concepts are verbalised by any meaning-preserving wording**.

Under this definition, label wording is *left open by the measurand*, so the spread over valid wordings is a candidate for definitional uncertainty (VIM 2.27; GUM D.3.4 "intrinsic uncertainty"). If instead the wording is fixed as part of the measurement procedure, the same spread is a procedure/realization effect (GUM 3.3.2 b, i). v2 estimates the spread; the surface-only control (RQ3) estimates how much of it is pure instrument non-invariance (no definitional content at all).

## Instruments (compare within type only; never across types)
- **T1 zero-shot NLI cross-encoders**: `cross-encoder/nli-deberta-v3-small`, `MoritzLaurer/ModernBERT-base-zeroshot-v2.0`. Hypothesis "This text is about {label}."; class probability = softmax over the entailment logits of the K labels.
- **T2 zero-shot bi-encoders**: `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `intfloat/e5-base-v2`. Score = cosine(text, "This text is about {label}."); probability = softmax(cos/τ), τ = 0.05 fixed a priori (τ = 0.02, 0.1 as sensitivity).
- **T3 small instruction LLMs (label log-likelihood)**: `Qwen/Qwen2.5-0.5B-Instruct`, `HuggingFaceTB/SmolLM2-360M-Instruct` on all datasets; `Qwen/Qwen2.5-1.5B-Instruct` on AG News + BBC-2025 subsets if time allows. Score = PMI = log p(label+eos | prompt(text)) − log p(label+eos | prompt("N/A")); probability = softmax over the K labels' PMI. Raw log-likelihood as sensitivity.
- **T4 supervised probes (wording-free reference)**: logistic regression (C = 1, lbfgs) on the T2 embeddings, trained on the dataset's training split (BBC-2025: trained on AG News, a real temporal/source shift). 20 training-bootstrap refits.
- All inference in **fp32** on CPU (batch-invariant to ~1e-5 in pilots); torch 2.14.1+cpu, transformers 5.20.1.

## Datasets
AG News (4 topics, 2004–05 news); BBC News 2024-10…2025-06 mapped to the same 4 classes (after the release of every model used); SST-2 (2, sentiment); DBpedia-14 (14 encyclopedic categories); MASSIVE en-US scenarios (18; voice-assistant requests; no banking/finance intents). Pools: AG/SST-2/MASSIVE 2,600, BBC 2,238, DBpedia 1,400 items. T1/T3 score the first 1,200 pool items (DBpedia 1,000) because of CPU cost; T2/T4 score the full pool.

## Wording populations
From `data/labels_<ds>.json`: per class, candidates from generators G0–G7 (+ v1 hand sets for AG/BBC). **VALID** = discriminable under the judge AND not rejected by the researcher audit; **STRICT** = VALID and judge-cosine ≥ 0.5; **ALL** = every candidate. A *wording set* assigns one variant to each class.
- Random-effects population: W = 30 sets per dataset, each class drawing uniformly from its VALID variants (seed 2026; SST-2 has only 30 possible valid sets, so all are used). Same 30 sets for every model of a dataset.
- Also: official set; homogeneous generator sets (every class uses generator g, where available — used for dose-response); STRICT-population and ALL-population draws (30 each, sensitivity); controls: identity (official repeated), surface-only (G1 case changes; for K classes, sets that change the case of a random subset of classes), nonsense N1/N2, shuffled names (derangements).
- T1/T3 on SST-2/DBpedia/MASSIVE score only VALID variants + N1 (cost); ALL-population and dose-response analyses for T1/T3 therefore use AG News and BBC-2025 only. T2 scores everything everywhere.

## Research questions, hypotheses and refutation criteria
Cells = dataset × model within a type. "Confirmatory" tests are Holm-corrected within each RQ; everything else is exploratory.

**RQ1 — Magnitude.** At n_cal = 250, α = 0.10, LAC, is the wording component of mean set size comparable to split sampling (calibration + test, test size 300)?
- H1a: σ_wording > 0 with a 95% two-way-bootstrap CI excluding 0 in ≥ 80% of zero-shot cells.
- H1b: σ_wording / σ_split ∈ [1/3, 3] in ≥ 50% of zero-shot cells.
- **Refuted if** σ_wording's CI includes 0 in > 50% of cells, or the ratio is < 1/3 in > 50% of cells (wording negligible), or > 3 in > 50% (sampling negligible — then the v1 "64% sampling" framing is wrong in the other direction).
- Also reported: share of variance σ²_w/(σ²_w+σ²_s+σ²_ws) with CI, per cell, per n; T4 has σ_wording ≡ 0 by construction.

**RQ2 — Floor.** Does the wording component plateau with more calibration data while sampling falls, and is the floor an artefact of the number of paraphrases?
- H2a: σ_split(n) falls with log-log slope in [−0.65, −0.35] over n ≥ 100.
- H2b: σ_wording(n) has log-log slope > −0.15 over n ≥ 250 (no material decline) and σ_wording(n_max) CI excludes 0, in ≥ 2/3 of zero-shot cells.
- H2c: The point estimate of σ_wording at n_max does not depend on W: mean over 200 random W-subsets of size 4 is within ×[0.8, 1.25] of the W = 30 estimate (after the ddof = 1 correction); the CI width shrinks roughly as 1/√W.
- **Refuted if** σ_wording declines with slope ≤ −0.35 in a majority of cells (the "floor" was calibration noise), or if its n_max CI includes 0 in a majority of cells.
- Also: crossing n* where σ_split(n*) = σ_wording (extrapolated with slope −0.5 if beyond the grid), and the v1-style design (fixed test set, calibration resampled from the pool) for direct comparison with the published 0.23 → 0.04 and ~0.02 floor.

**RQ3 — What is the wording component made of?** (definitional content vs instrument non-invariance)
- H3a: Surface-only (case) changes alter predictions: mean argmax-flip rate vs official > 0 (CI excludes 0) for at least one model per type (exploratory for T2, whose encoders may be uncased).
- H3b (dose-response): across all candidate variants, the per-variant prediction shift (argmax-flip rate when only that class's label is replaced; also mean total-variation distance) increases with semantic distance (1 − judge cosine): Spearman ρ > 0 with CI excluding 0, pooled within each type.
- H3c: σ_wording(VALID) < σ_wording(ALL) in ≥ 2/3 of cells where both are available (validity filtering removes part of the spread).
- **Refuted (for the "definitional" reading)** if surface-only flip rates are ≥ 50% of VALID-paraphrase flip rates in a majority of cells — the component would then be mostly instrument non-invariance. **Refuted (dose-response)** if ρ ≤ 0 or CI includes 0 in all types.

**RQ4 — Contamination / novelty proxy (AG News vs BBC-2025, same 4 classes, same wordings).** Exploratory, because source and style differ too.
- H4a: Min-K% Prob (k = 20%) is higher (more "familiar") for AG News than BBC-2025 texts under each LLM (difference CI excludes 0).
- H4b: zero-shot accuracy is lower and σ_wording (set size, n = 250) is larger on BBC-2025 than AG News for most zero-shot models.
- **Refuted** (H4b) if differences go the other way or CIs include 0 for most models; then contamination is not what keeps AG wording effects small (or large).

**RQ5 — Repeatability vs reproducibility on CPU.**
- H5a: identical runs (4 fresh processes, same configuration, fp32) give max |Δp| = 0 for one model of each type.
- H5b: changing batch size (1 vs 64) in fp32 gives max |Δp| < 1e-4 and changes no conformal set; switching to bf16 (or bf16 with a different batch size) gives non-zero Δp that changes conformal sets for a measurable share of items.
- **Refuted** (H5a) by any non-zero difference between identical fp32 runs.

## Conformal methods, metrics, α
LAC (score 1 − p_y; Sadinle et al. 2019), APS (randomized, Romano et al. 2020; fixed RNG seed per split), RAPS (Angelopoulos et al. 2021; λ = 0.01, k_reg = 1 fixed a priori). Quantile level ⌈(n+1)(1−α)⌉/n, "higher" interpolation. α ∈ {0.05, 0.10, 0.20}; 0.01 only for n ≥ 500. Full method × α grid at n = 250 and n_max; floor curves for LAC and APS at α = 0.10.

## Statistical analysis
- Design A (primary): for each n, S = 200 random splits (test = 300 items, calibration = n of the remaining), crossed with the W wording sets. Two-way crossed random effects without replication; method-of-moments estimates σ²_w = (MS_w − MS_e)/S, σ²_s = (MS_s − MS_e)/W, σ²_ws+e = MS_e (negative estimates set to 0, count reported). 95% CIs: two-way bootstrap (resample wordings and splits independently), 1,000 draws.
- Design B (v1 replication): fixed test set (first 300 non-calibration items), calibration resampled with replacement from the remaining pool.
- Slopes: OLS on log σ vs log n, CI by bootstrap over wordings/splits.
- Dose-response: Spearman ρ with bootstrap CI over variants; also a mixed-effects view (variant nested in class nested in dataset) reported descriptively.
- Repeatability: max |Δp|, ICC(2,1), ICC(3,1) on p(true), Fleiss κ on argmax labels, share of conformal sets changed.
- Accuracy/coverage CIs: percentile bootstrap over items.
- No cross-type ranking. Holm correction across cells within each confirmatory RQ.

## v1 re-analysis (from committed v1 outputs, no new model runs)
Re-compute the published Clef budget with (i) wording treated as a random factor with W = 4 (χ²-based CI on σ_wording), (ii) a two-way bootstrap CI on the 32%/64% shares, (iii) the share as a function of n using the 500-item calibration pool.

## Deviations
Any deviation from this plan will be listed in `report.md` under "Deviations from preregistration".
