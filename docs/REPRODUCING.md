# Reproduction details

## Saved-output analysis

Use the root README commands first. They need no provider credentials and make no network/model calls after dependencies are installed. `scripts/reproduce.py --skip-plots` skips only plot rendering; it still repeats scoring and the fixed 10,000-repetition bootstrap. Outputs are written separately from the archived reference evidence.

The offline dependency pins describe the validated Python 3.10/3.11 analysis environment. The historical Apple Silicon inference environment is recorded separately in `tdsc/configs/captchamind_runtime_requirements.txt`; do not treat its platform-specific MLX packages as dependencies of offline re-analysis.

## Conference analysis

`scripts/reproduce_conference.py` rebuilds the expanded static/adaptive evidence table from the saved final summary inputs. The fixed data selections and source download manifest are in the full release under `conference/expanded_captcha_data/phase04_2/`. Archived Exp1–Exp4 CSVs and plots are under `conference/results/` and `conference/figures/`.

For the original figure and retry-statistics workflows, open `conference/notebooks/plot.ipynb` and `conference/notebooks/test_statistic.ipynb` with `conference/` as the working directory. Notebook execution outputs have been cleared in this release. The associated modules are `cognition/visualize_results.py` and `cognition/exp2_to_exp3_predict.py`. Their inputs remain the conference CSVs; the new TDSC studies use the separate frozen manifests.

## Optional new model evaluation

Saved-output reproduction does not require new inference. To run a new experiment, obtain the original BF16 checkpoint at the documented revision and use the recorded model/runtime settings. A changed hosted model, backend, checkpoint, or interface is a new evaluation and may produce different results.

The cleaned TDSC runner selects one formal module and protocol. It prints its planned items and maximum calls by default and writes new outputs outside the archived evidence:

```bash
.venv/bin/python tdsc/scripts/run_protocol.py --module A --solver C-I --output generated/runs/a-ci
```

Add `--execute` only when ready to make model requests. For CaptchaMind, supply a locally served OpenAI-compatible endpoint and its model identifier, for example `--base-url http://127.0.0.1:8100/v1 --model models/CaptchaMind-7B`. The runner uses `CAPTCHAMIND_API_KEY` if the local server requires a key. GEM-V reads `GEMINI_API_KEY`. Local credential files are optional; never commit them.

The original runtime preserves image markers in each originating message. The exact MLX-VLM 0.7.0 source patch is archived in `tdsc/reports/solver_defense_v1/runtime/history.patch`; source and rendering checks are in that directory. Apply it only to a dedicated runtime after checking that its original context matches. Serve sequentially with original BF16 weights, no visual-feature cache, and a 128,000-token context/KV limit. The original runtime versions, processor configuration, and single/multi-round template checks are included so deployment differences can be inspected. Host-specific service launchers and SSH controls are not part of this artifact.

The runner's A/B static protocol uses the archived answer-prompt builder; D uses the paired defense prompt; CM-I uses source-native B and adapted A/D environments. `tdsc/manifests/` also contains the fixed development splits for separately checking a new deployment. Do not include development results in formal estimates or select a protocol based on formal outcomes.

## Data and record conventions

- Coordinates are pixels in the specified full image; image dimensions and reference order are in each manifest.
- `id` uniquely identifies an item; `group_id` identifies the material unit for paired/cluster analysis; `partition` distinguishes formal and development material.
- `answer`/`final_answer` are parsed complete answers. Raw response text is retained separately. A null answer is not an invented zero or empty point set.
- `correct` is the run-level outcome; A/B derived records also distinguish `original_scorer_correct`, `source_valid`, and `final_correct`.
- `elapsed_seconds` is recorded per-episode wall time. Token counts are recorded usage; unknown totals remain null. Recorded price metadata are historical, not a current provider quotation.
- `submitted`, `done`, `status`, and `stop_reason` distinguish scoring failures, submission failures, budgets, and infrastructure states.

The release preserves response/action trajectories and deadline evidence, but omits redundant base64 image requests and rendered observation copies. Original inputs and the simulator code are included in the full archive. The absent deadline responses are explicitly missing; they cannot be regenerated from the recorded evidence.
