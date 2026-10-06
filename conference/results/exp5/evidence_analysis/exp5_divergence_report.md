# Phase 04.2 Corrected Evidence Analysis Divergence Report

## Cutoff Semantics
The 40% reporting heuristic is used only to compare original and corrected evidence directions; it is not a universal CAPTCHA security boundary.

## Agreement and Divergence
Rows analyzed: 30.
Divergent rows: 4.
Claim effects: supports_structural_hardness=26, weakens_structural_hardness=4

- openai/gpt-5_medium / Patch_Select (adaptive): Corrected Phase 04.2 evidence crosses the 40% reporting heuristic in the opposite direction from the original Exp2 rate. claim_effect=weakens_structural_hardness.
- openai/gpt-5_medium / Pick_Area (adaptive): Corrected Phase 04.2 evidence crosses the 40% reporting heuristic in the opposite direction from the original Exp2 rate. claim_effect=weakens_structural_hardness.
- openai/gpt-5_medium / Relation_Match (adaptive): Corrected Phase 04.2 evidence crosses the 40% reporting heuristic in the opposite direction from the original Exp2 rate. claim_effect=weakens_structural_hardness.
- openai/gpt-5_medium / Rotation_Match (adaptive): Corrected Phase 04.2 evidence crosses the 40% reporting heuristic in the opposite direction from the original Exp2 rate. claim_effect=weakens_structural_hardness.

## Corrected Provenance Boundary
Corrected direct static evidence is limited to Symbol_Count, Relation_Match, and Hole_Counting.
No staged OpenCaptchaWorld hard-type increment rows are retained in the final Phase 04.2 evidence artifacts; no hard-type dataset-increase percentages are claimed in Phase 04.2.
Staged OCW increment task types inspected: none.

## Adaptive Scope
Since most recognition-oriented tasks are already solved reliably under the non-adaptive setting, adaptive feedback would mainly introduce ceiling effects and provide limited additional insight. We therefore focus on the hard and boundary-hard task types, where adaptive memory has the greatest potential to change the security conclusion. This is the Phase 04.2 ceiling-effect rationale.
Adaptive rows report observed Success@3 over five memory-isolated rounds alongside the Exp2-derived Bernoulli baseline for k=3. Success@5 is retained only when available as auxiliary provenance.
