# Frozen protocol and scoring

GEM-V uses the recorded `gemini-3.8-flash` endpoint, temperature 0 and MEDIUM thinking. CM-S and CM-I use the same original BF16 `ATH-MaaS/CaptchaMind-7B` checkpoint, revision `b46c7a18f7647b0f4dbc4a2a10821e48239ca93a`. The study used MLX-VLM 0.7.0 on Apple Silicon. This deployment differs from the upstream paper's deployment; the experiment is not an exact numerical reproduction of that paper.

Static protocols provide the original images once and request one complete answer with a 16,384-token output cap. No model action is executed and no corrective model request is used to interpret an answer. Label-blind reading records preserve values and order; readers see task instructions and response text, not images, labels, or scores.

CM-I uses the enabled native observation/action prompt from `AlibabaResearch/captcha-mind`, revision `04e46b7673c361594e3f725720e49ab3bf7fb126`. B uses source-native environments; A and D use explicit adapted controls outside the original image. Limits are 30 calls, 8,192 output tokens per call, 65,536 output tokens per episode, a 600-second request timeout, and a 1,800-second episode limit. Full originating image history is retained within a 128,000-token model context; no truncated history is silently substituted. Final submission determines episode success.

The 30 D source images were audited before formal generation. Six fixed images are development material and 24 are formal material. D0 selects target-containing cells; D1 selects every visible target eye; D2 counts those eyes; D3 applies a fixed Pillow Gaussian blur of radius 2.0 to D0 puzzle pixels. D1 uses maximum-cardinality one-to-one matching at 15 px and requires no missing or extra points. The 30/45 px and legacy nearest-greedy views are diagnostics, not alternative primary results.

All fixed tasks remain in their specified denominators. A has one pre-identified source annotation defect: the primary source-valid comparison uses 140 items; the full diagnostic view retains all 141. B Dice_Count has a single material group, so a between-material bootstrap interval is not reported for that task or aggregates containing it. Confidence intervals otherwise use the recorded type-stratified material-group paired percentile bootstrap, 10,000 repetitions, seed 20260913; D conditions are paired by base image.

## Post-generation resolutions

Seven CM-S D1 outputs remained ambiguous after label-blind reading. The subsequent author-directed policy classifies each as `ambiguous_answer_failure`. The original text, blind decisions, and null coordinate answer are unchanged. These failures enter 24-item success denominators; coordinate diagnostics retain the 17 readable-answer denominator. The policy is recorded in `tdsc/reports/solver_defense_v1/execution_fixes/ambiguity_as_failure_v1/` and applied from `d_static/answer_readings.json`.

Ten A episodes reached the fixed episode deadline while a request was outstanding. An explicit execution overlay classifies the failure only when dispatch, timeout, and elapsed evidence establish deadline exhaustion without submission. Requests were not resent; unavailable late responses remain absent. Full token use remains unknown and observed-prefix tokens are separate. The per-episode `execution_resolution.json` and transport records preserve the evidence; `tdsc/scripts/episode_deadline_recovery.py` contains the policy. Other infrastructure failures are not silently converted to solver failures.

The complete recorded method is in `tdsc/reports/solver_defense_v1/METHOD.md`; raw and derived answers are stored separately. Development records do not contribute to formal success rates. The study provides no new human usability measurements and does not evaluate production CAPTCHA services.
