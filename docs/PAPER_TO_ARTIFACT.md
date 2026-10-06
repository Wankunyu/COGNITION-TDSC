# Manuscript-to-artifact map

Paths below are relative to the release root. Internal A/B/D names remain in records to preserve provenance; the manuscript uses descriptive study names.

| Manuscript result | Evidence | Reproduction |
| --- | --- | --- |
| Exp1/Exp2 direct and optimized prompting | `conference/results/exp1/`, `exp2/`; `conference/figures/` | `conference/notebooks/plot.ipynb`; `cognition/visualize_results.py` |
| Exp3 finite retries and cost/latency | `conference/results/exp3/`; `conference/figures/` | `conference/notebooks/test_statistic.ipynb`; `cognition/exp2_to_exp3_predict.py` |
| Exp4 few-shot results | `conference/results/exp4/`, `conference/few_shot_examples.yaml` | `conference/cognition/run_single_experiment.py` |
| Exp5 external/static/adaptive evidence | `conference/results/exp5/` | `scripts/reproduce_conference.py` |
| CaptchaWorld and external solver table (`tab:ab`) | `tdsc/reports/solver_defense_v1/analysis/ab_table.csv` | `scripts/reproduce.py` |
| Paired solver contrasts | `analysis/ab_paired_comparisons.csv`, `ab_paired_items.csv` | Same |
| Submission outcomes (`tab:terminal`) | `analysis/interactive_terminal.csv`, `interactive_last_state.csv` | Same |
| Resource use (`tab:resources`) | `analysis/resources.csv`; attempts and call ledgers | Same |
| Defense conditions (`tab:defense-conditions`) | `tdsc/manifests/defense_v1_eval.json`; `tdsc/data/defense_v1/annotations.json` | Fixed manifest and independent source audit |
| Paired defense (`tab:defense`) | `analysis/defense_table.csv`, `defense_effects.csv`, `defense_pairs.csv` | `scripts/reproduce.py` |
| Eye/count diagnostics (`tab:defense-diagnostics`) | `analysis/eye_diagnostics.csv`; `analysis/summary.json` | Same |
| Tolerance and scorer diagnostics | `analysis/d1_tolerance.csv`, `d1_legacy_compatibility.csv` | Same; primary radius remains 15 px |
| Source defect and selection | `tdsc/manifests/answer_20260913_conference.json`, `answer_20260913_external.json`, `external_selection.json` | Fixed manifests and `tdsc/reports/external_eligibility.md` |
| Ambiguity decisions | `tdsc/reports/answer_20260913/blind_review/`, `solver_defense_v1/d_static/blind_review/` | Recorded reading overlays, no model re-query |

In rows abbreviated as `analysis/`, the full prefix is `tdsc/reports/solver_defense_v1/analysis/`. Machine-readable results, rather than rounded manuscript percentages, are the reference for reproduction checks.
