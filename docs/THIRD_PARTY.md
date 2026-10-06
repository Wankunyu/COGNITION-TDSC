# Sources and third-party attribution

| Component | Fixed source | Terms / handling |
| --- | --- | --- |
| Conference artifact | [Zenodo 10.5281/zenodo.20406852](https://doi.org/10.5281/zenodo.20406852) | Original archive remains unchanged; existing author terms are carried forward |
| CaptchaMind agent and simulator | [AlibabaResearch/captcha-mind](https://github.com/AlibabaResearch/captcha-mind/tree/04e46b7673c361594e3f725720e49ab3bf7fb126), revision `04e46b7673c361594e3f725720e49ab3bf7fb126` | MIT; original license and source provenance included under `tdsc/baseline/captchamind_native/` |
| External CAPTCHA data | [ATH-MaaS/Captcha](https://huggingface.co/datasets/ATH-MaaS/Captcha/tree/960a521e842bd164b08344151b0a4253d23cfc2f), revision `960a521e842bd164b08344151b0a4253d23cfc2f` | Pinned dataset card declares CC BY-NC 4.0; source images/annotations are retained in the full release |
| Specialist checkpoint | [ATH-MaaS/CaptchaMind-7B](https://huggingface.co/ATH-MaaS/CaptchaMind-7B/tree/b46c7a18f7647b0f4dbc4a2a10821e48239ca93a), revision `b46c7a18f7647b0f4dbc4a2a10821e48239ca93a` | Model weights are not included; obtain the BF16 checkpoint from the upstream repository |
| Conference external categories | `conference/expanded_captcha_data/phase04_2/source_download_manifest.json` and selected/evidence manifests | Preserve each source's recorded attribution and material-level provenance |
| Runtime | MLX-VLM 0.7.0 and the recorded package versions | See `tdsc/configs/captchamind_runtime_requirements.txt` and runtime evidence |

External benchmark training exposure is unknown. `paper_only` split evidence must not be interpreted as proof of a held-out training set. The same source item can participate in multiple recorded protocol evaluations; group IDs control the paired analyses.

Local storage paths and research-host identifiers are normalized in the public copy. Scientific IDs, original answer text, labels, timestamps, counts, scores, model-call budgets, and upstream revisions are retained. File-level transformations are listed in `artifact_manifest.json`.

The upstream `image_utils.py` main-only demonstration, which used the upstream author's absolute local paths, is omitted. Its callable definitions and the native agent/environment behavior are unchanged. The upstream transport example contains literal `your-key`/`your-real-url` placeholders; the experiment replaces its completion function with the recorded provider adapter.
