# External task eligibility

Reviewed 2026-09-11 against official CaptchaMind code, published task descriptions, and public initial images/labels. The inclusion rule is semantic equivalence, complete visible input, auditable scoring, and source provenance. Engineering readiness does not establish representativeness.

| Original task | CaptchaBench decision | Reason and required handling |
|---|---|---|
| Dice_Count | Eligible for static source comparison | Six panels contain the visible dice; numeric and pip faces are summed. Exact integer answer. Training exposure remains unknown. |
| Click_Order | Eligible for static source comparison | Full reference sequence is in the initial image; later clicks only add markers. Preserve ordered rectangle hits, rather than original-data circular tolerances. Training exposure remains unknown. |
| Patch_Select | Eligible for static source comparison | Initial 4x4 grid is complete; subsequent clicks toggle selection rather than replace images. Preserve zero-based row-major exact-set scoring. The official loader selects sample_1801–2000, but training exclusion is unverified. |
| Rotation_Match | Eligible after angle conversion | Both sources use clockwise-positive orientation. Supply only `image_1.png`, including its reference, and score additional clockwise rotation as `(target_angle - initial_angle) mod 360`, with the original 5-degree tolerance. Native candidate index and angle labels are cross-checked for every selected sample. |
| Place_Dot | Excluded from this source | The coordinates and slide-puzzle tasks are not path-endpoint tracing. Exclusion concerns task semantics, not model capability. |
| Pick_Area | Excluded from this source | No area-comparison task was found. Generic region clicking is not equivalent. |

All six original tasks remain in A. B conclusions apply only to the included task/source combinations.

## Frozen selection and material overlap

There are 50 selected samples in each of the four eligible types, for 200 total. Dice, Click and Patch selections use seed 20260911; Rotation uses seed 20260912 after the angle adapter was checked on excluded development samples 1, 2 and 37. Existing selections were preserved. Patch draws from 1801–2000; Dice and Click draw from 1–2000 excluding development IDs 1, 500 and 1500. Failed downloads are not replaced with easier samples.

The rotation direction check used original `cat.png`, `cat_90.png`, `direction_1.png` and external development samples. The original 90-degree image turns clockwise toward the right-hand reference. In the external samples, the 45-degree fox leans clockwise; the 180-degree giraffe and 90-degree target require 270 additional clockwise degrees. No transformed test images were generated. This validates a source-label adapter, not the deferred presentation-stability experiment.

Dice uses 156 distinct source panels in 300 panel positions. All 50 composite questions belong to one connected shared-panel group. Click uses 10 observable background groups. Rotation uses 40 base objects. The manifest records those groupings; no independent-item Wilson interval is reported for these correlated material sets. Patch source-photo IDs are unavailable, so absence of pixel duplicates does not establish independent source provenance.

The initial 150-question Dice/Click/Patch audit found zero exact complete-image duplicates within B or against A. Perceptual screening found no cross-source candidate at distance at most 8; one within-Dice candidate was visually checked and was not a repeated question. See `data_quality.json` for counts and `data_quality.md` for the audit method.

## Split evidence

The paper reports 2,000 training and 200 evaluation samples per type. Public repository metadata does not map all sample IDs to this split. The Patch loader's 1801–2000 range is evidence of upstream evaluation selection, not proof those images were excluded from training. Dice and Click loaders give no public test-ID mapping.

The earlier Hugging Face `siblings` cache is incomplete and cannot be used to infer a task is absent. Public directory requests and direct sample URLs established that Patch samples above 1409 and Rotation samples exist. Rotation labels are named `coordinates.json`, although the repository loader expects `rotation_match.json`.

## Sources

- [Official code, inspected revision](https://github.com/AlibabaResearch/captcha-mind/tree/04e46b7673c361594e3f725720e49ab3bf7fb126)
- [Paper and appendices](https://arxiv.org/html/2605.19538v1)
- [Patch loader test selection](https://github.com/AlibabaResearch/captcha-mind/blob/04e46b7673c361594e3f725720e49ab3bf7fb126/captcha/env/patch_select/data/__init__.py)
- [Dataset](https://huggingface.co/datasets/ATH-MaaS/Captcha), fixed revision `960a521e842bd164b08344151b0a4253d23cfc2f`
