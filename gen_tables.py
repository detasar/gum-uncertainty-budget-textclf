"""Render results.json + repeatability.json into markdown tables (tables.md)."""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f"{H}/results.json")); P = json.load(open(f"{H}/repeatability.json")) if os.path.exists(f"{H}/repeatability.json") else {}
f2 = lambda x: "n/a" if x is None else f"{x:.3f}"
ci = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"
L = ["### Reference metrics per instrument class (250 cal / 250 test, alpha = 0.10; 95% CI = joint bootstrap of cal and test, 1000 draws)", "",
     "**Not a leaderboard.** The classes differ in output type and in the information they use. Read across rows only *within* a class.", ""]
for cls in sorted({v.get("instrument_class", "?") for v in R["models"].values()}):
    ot = next(v.get("output_type", "") for v in R["models"].values() if v.get("instrument_class") == cls)
    L += [f"**{cls}**: output = {ot}", "", "| model | accuracy [95% CI] | coverage [95% CI] | mean set size [95% CI] | shifted test (0.3): acc / cov / size |", "|---|---|---|---|---|"]
    for k, v in R["models"].items():
        if v.get("instrument_class") != cls: continue
        s = v.get("shift30"); ss = f"{s['accuracy']:.3f} / {s['coverage']:.3f} / {s['set_size']:.3f}" if s else "not run"
        L.append(f"| {k} | {v['accuracy']:.3f} {ci(v['ci95_accuracy'])} | {v['coverage']:.3f} {ci(v['ci95_coverage'])} | {v['set_size']:.3f} {ci(v['ci95_set_size'])} | {ss} |")
    L.append("")
for b, v in R["budgets"].items():
    rm = v["reference_metrics"]
    L += ["", f"### Budget {b}: reference = {v['reference']} (set size {rm['set_size']:.3f}, coverage {rm['coverage']:.3f})", "",
          "| component | evaluation type | u(set size) | share of variance | u(coverage) | share of variance | notes |", "|---|---|---|---|---|---|---|"]
    for c in v["components"]:
        inr = c.get("in_rss", True)
        L.append(f"| {c['component']} | {c['type']} | {c['u_set_size']:.4f} | {100*c['share_var_set_size']:.0f}%{'' if inr else ' (excluded)'} | {c['u_coverage']:.4f} | {100*c['share_var_coverage']:.0f}%{'' if inr else ' (excluded)'} | {c.get('note','')} |")
    L.append(f"| **combined (root-sum-square)** | | **{v['u_c_set_size']:.4f}** | | **{v['u_c_coverage']:.4f}** | | independence + additivity assumed |")
if "mc_check_clef" in R:
    m = R["mc_check_clef"]
    L += ["", f"**Monte Carlo check (Clef, GUM-S1 style, {m['n_draws']} joint draws over prompt config x cal bootstrap x test bootstrap):** SD(set size) = {m['sd_set_size']:.4f} vs RSS of the same four components = {m['rss_same_components_set_size']:.4f}; SD(coverage) = {m['sd_coverage']:.4f} vs RSS = {m['rss_same_components_coverage']:.4f}."]
if "scaling_clef_wording" in R:
    L += ["", f"### Calibration size vs wording (Clef, 4 wordings, pool = {R.get('scaling_pool_size')} calibration items, 300 paired draws per n)", "",
          "| n_cal | SD from calibration sampling | wording, systematic part (SD of per-wording means) [95% CI] | wording, within-draw SD |", "|---|---|---|---|"]
    for s in R["scaling_clef_wording"]:
        L.append(f"| {s['n_cal']} | {s['sd_cal_sampling']:.4f} | {s['sd_wording_systematic']:.4f} {ci(s['ci95_sd_wording_systematic'])} | {s['sd_wording_within_draw']:.4f} |")
ae = R["aleatoric_epistemic"]
L += ["", "### Entropy decomposition (TU = AU + EU) depends on what is called 'the model ensemble' (test set, nats)", "",
      "| setup | mean TU | mean AU | mean EU | EU share | EU share under shift |", "|---|---|---|---|---|---|"]
for k, v in ae["decomposition"].items():
    L.append(f"| {k} | {v['mean_TU']:.3f} | {v['mean_AU']:.3f} | {v['mean_EU']:.3f} | {100*v['EU_share']:.0f}% | {('%.0f%%' % (100*v['shift30_EU_share'])) if 'shift30_EU_share' in v else '-'} |")
L += ["", "| pair | Spearman rho of per-item EU | overlap of top-10% EU items |", "|---|---|---|"]
for k, v in ae["cross_model_agreement_on_EU"].items():
    L.append(f"| {k} | {v['spearman_EU']:.2f} | {100*v['overlap_top10pct_EU']:.0f}% |")
if P:
    L += ["", "### Test-retest repeatability (identical requests, >= 4 runs)", "",
          "| setup | runs x items | max abs dp (any pair) | mean per-item SD(p) | ICC(3,1) / ICC(2,1) on p(true) | Bland-Altman run1-run0: mean [LoA] | mean pairwise TV / JS (bits) | label source | Cohen kappa mean [range] | Fleiss kappa | items same label all runs | SD set size / coverage across runs | test items whose set changes | saturation: max p>0.99 / entries outside [0.01,0.99] |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in P.items():
        if k.startswith('_'): continue
        ba = v["bland_altman_run1_vs_run0_p_true"]; kr = v["cohen_kappa_pairwise_range"]
        L.append(f"| {k.replace(' | ', ' / ')} | {v['runs']} x {v['items']} | {v['max_abs_dp_any_pair']:.3g} | {v['mean_item_sd_p']:.3g} | {f2(v['icc31_p_true'])} / {f2(v['icc21_p_true'])} | {ba['mean_diff']:.2g} [{ba['loa_low']:.2g}, {ba['loa_high']:.2g}] | {v['mean_pairwise_total_variation']:.3g} / {v['mean_pairwise_js_bits']:.3g} | {v['label_source']} | {f2(v['cohen_kappa_pairwise_mean'])} {('[%.3f, %.3f]' % tuple(kr)) if kr else ''} | {f2(v['fleiss_kappa'])} | {100*v['pct_items_identical_label_all_runs']:.1f}% | {v['sd_set_size_across_runs']:.3g} / {v['sd_coverage_across_runs']:.3g} | {100*v['pct_test_items_whose_conformal_set_changes']:.1f}% | {100*v['saturation']['pct_max_p_gt_0.99']:.0f}% / {100*v['saturation']['pct_entries_outside_0.01_0.99']:.0f}% |")
if R.get("cross_wording"):
    L += ["", "### Configuration shift: calibrate on one label wording, test on another (alpha = 0.10, target coverage 0.90)", "",
          "| model | coverage, same wording: mean [min, max] | coverage, cross wording: mean [min, max] | wording ensemble (mean prob): coverage / set size |", "|---|---|---|---|"]
    for k, v in R["cross_wording"].items():
        e = v["wording_ensemble (mean prob over wordings)"]
        L.append(f"| {k} | {v['coverage_same_wording_mean']:.3f} [{v['coverage_same_wording_range'][0]:.3f}, {v['coverage_same_wording_range'][1]:.3f}] | {v['coverage_cross_wording_mean']:.3f} [{v['coverage_cross_wording_range'][0]:.3f}, {v['coverage_cross_wording_range'][1]:.3f}] | {e['coverage']:.3f} / {e['set_size']:.3f} |")
if R.get("name_vs_definition_clef"):
    n = R["name_vs_definition_clef"]
    L += ["", "### Option name vs definition (Clef-flash, 250 test items, definitions = w2_topic texts)", "", "| condition | selects the correct definition | selects the option whose NAME matches the true class |", "|---|---|---|"]
    for c in ["neutral_ids_A-D", "matched", "swapped"]:
        if c in n: L.append(f"| {c} | {n[c]['follows_definition (accuracy vs true label)']:.3f} | {n[c].get('follows_name', float('nan')):.3f} |")
    for k in ["decision_flip_rate_neutral_to_matched", "decision_flip_rate_matched_to_swapped"]:
        if k in n: L.append(f"\n{k}: {100*n[k]:.1f}%")
if P.get("_cross_session_main_vs_rep_run0"):
    L += ["", "Cross-session check (main run in a separate process vs repeatability run 0, identical requests): " + "; ".join(f"{k}: max abs dp = {v['max_abs_dp']} over {v['n_items']} items" for k, v in P["_cross_session_main_vs_rep_run0"].items() if v)]
open(f"{H}/tables.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
