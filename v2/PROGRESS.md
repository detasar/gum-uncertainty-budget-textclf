# PROGRESS (report-writing session, 2026-10-17 from 16:16 Istanbul, UTC+3; times approximate)
- [16:16] Started. Background jobs alive: queue_a/b, run_after, score_nli agnews 1200, score_llm Qwen0.5B bbc2025, run_repeat_resume (nli_rep1). Not touching them.
- [16:18] All raw scores already analysed (no missing cells). Re-ran src/report_tables.py to add Table 11 (template factor, finished 16:12). Repeatability: only rep0 of each instrument exists -> RQ5 not analysable yet. Contamination (minkprob) not run yet.
- [16:19] Added exploratory src/repeat_vs_main.py -> results/repeat_rep0_vs_main.json (rep0 vs main-grid run: bge max|dp| 2.0e-6, Qwen 1.8e-5, 0 argmax flips). Verified plan.md identical to fab3c8a; first raw output 14:19:55 > commit 14:18:58. Next: write report.md.
- [16:20] nli_rep1, bienc_rep1 finished -> ran src/repeat_analysis.py (partial: 2 identical runs each, NLI+bi-enc: max|dp|=0). Re-ran report_tables (Table 10). Running figures.py.
- [16:21] figures.py re-run (23 s; fig07_repeatability added). Writing report.md now.
- [16:22] report.md written. Next: summary_tr.md, sync, push.
- [16:22] summary_tr.md written. Next: sync + commit + push.
- [16:23] Running sync_repo.sh, committing to v2-comprehensive.
- [16:24] Push failed: git credential helper points to /usr/bin/gh (missing); gh is at /usr/local/bin/gh. Pushed using that helper for this command only.

## Still pending (nothing was killed)
- run_queue_a.sh: NLI deberta-v3-small agnews (running), then remaining 9 T1 cells.
- run_queue_b.sh: Qwen2.5-0.5B bbc2025 (running), then Qwen sst2/massive/dbpedia, SmolLM2-360M x5.
- run_repeat_resume.sh: llm_rep1 running; reps 2-3; intermediate-precision conditions; Min-K% (Qwen0.5B, SmolLM2).
- run_after.sh: waits for queues; Min-K%, run_repeat.sh (does NOT skip existing outputs), Qwen2.5-1.5B AG/BBC, Min-K% 1.5B.
- After they finish: ./analyze_pending.sh; python src/aggregate.py; src/repeat_analysis.py; src/report_tables.py; src/figures.py; update report.md sections 3, 4, 7 and summary_tr.md; sync_repo.sh; commit + push v2-comprehensive.
- [16:24] NLI deberta agnews finished (queue A moved to bbc2025); llm_rep1 finished. Running analyze_pending.sh + aggregate + repeat_analysis + tables + figures.
- [16:26] NLI agnews cell analysed (24 s); aggregate/repeat_analysis/report_tables/figures re-run. H1b now REFUTED (ratio>3 in 9/17=0.53). Updating report.md.
- [16:26] report.md and summary_tr.md updated with the NLI cell; re-syncing and pushing.
