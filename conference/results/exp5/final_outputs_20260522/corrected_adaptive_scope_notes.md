# Phase 04.2 Adaptive Hard-Scope Notes

Since most recognition-oriented tasks are already solved reliably under the non-adaptive setting, adaptive feedback would mainly introduce ceiling effects and provide limited additional insight. We therefore focus on the hard and boundary-hard task types, where adaptive memory has the greatest potential to change the security conclusion. This is the Phase 04.2 ceiling-effect rationale.

Adaptive rows are five memory-isolated rounds; the final table reports Adaptive-Success@3, with Success@5 retained only when available as auxiliary provenance.

| task_type | provider_model | adaptive_success_at_3 | adaptive_success_at_5 | round_count | memory_isolation | adaptive_hard_scope_evidence | contextual_sota_only |
|---|---|---:|---:|---:|---|---|---|
| Click_Order | openai/gpt-5_medium | 0.2 | 0.2 | 5 | five_memory_isolated_rounds | True | False |
| Dice_Count | openai/gpt-5_medium | 0.2 | n/a | 5 | five_memory_isolated_rounds | True | False |
| Hole_Counting | openai/gpt-5_medium | 0.0 | 0.0 | 5 | five_memory_isolated_rounds | True | False |
| Patch_Select | openai/gpt-5_medium | 0.4 | 0.4 | 5 | five_memory_isolated_rounds | True | False |
| Pick_Area | openai/gpt-5_medium | 0.4 | 0.6 | 5 | five_memory_isolated_rounds | True | False |
| Place_Dot | openai/gpt-5_medium | 0.2 | 0.4 | 5 | five_memory_isolated_rounds | True | False |
| Relation_Match | openai/gpt-5_medium | 0.4 | 0.4 | 5 | five_memory_isolated_rounds | True | False |
| Rotation_Match | openai/gpt-5_medium | 0.4 | n/a | 5 | five_memory_isolated_rounds | True | False |
| Symbol_Count | openai/gpt-5_medium | 0.0 | 0.0 | 5 | five_memory_isolated_rounds | True | False |
