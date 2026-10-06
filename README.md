# COGNITION: TDSC submission artifact

Experiments for **Evaluating and Strengthening Visual CAPTCHAs against Static and Interactive Multimodal Solvers**, by Junyu Wang, Yuanbo Zhou, Changjia Zhu, Lingyao Li, Xu He, Mingkui Wei, and Junjie Xiong.

This repository integrates the conference experiments and the TDSC solver/interaction and paired-defense studies. The default reproduction path reads saved model outputs and makes **no model requests**. The TDSC manuscript is a submission, not an accepted publication.

## Download and reproduce

Download `COGNITION-TDSC-full.zip` from the [submission release](https://github.com/Wankunyu/COGNITION-TDSC/releases/tag/tdsc-submission-v1.0) and extract it. The complete archive includes the code, recorded results, and image data. The Git repository contains code and textual evidence; large image-data directories are supplied in the release archive.

Use Python 3.10 or 3.11 for the pinned offline analysis environment:

```bash
cd COGNITION-TDSC
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-offline.txt
.venv/bin/python scripts/reproduce.py
.venv/bin/python scripts/reproduce_conference.py
```

The first command re-scores all **682 A/B static answers**, applies the archived label-blind readings, analyzes **629 additional formal episodes** containing **3,233 model calls**, and regenerates the TDSC tables and diagnostic plots. It checks the scientific contents of `summary.json` and all archived CSV tables against the frozen reference results. New files go to `generated/tdsc/`. The second command rebuilds the conference Exp5 evidence table from saved summaries, writing to `generated/conference/`.

Offline tests, from the artifact root:

```bash
(cd tdsc && ../.venv/bin/python -m pytest tests -q)
(cd conference && ../.venv/bin/python -m pytest tests -q)
```

Analysis can run from a Git clone without the large image archive. Tests that render puzzle environments and any new model evaluation require the full archive. See [REPRODUCING.md](docs/REPRODUCING.md) for the conference figure workflow and optional new inference.

## Experiments and expected results

| Study | Fixed evaluation material | Main saved results |
| --- | --- | --- |
| Conference Exp1–Exp4 | Cleaned CaptchaWorld benchmark; direct, optimized, retry, and few-shot protocols | `conference/results/exp1/` through `exp4/` |
| Conference Exp5 | Expanded static and adaptive session-memory evidence | `conference/results/exp5/` |
| CaptchaWorld solver comparison (A) | 141 executed items across six historical hard types; 140 source-valid items | GEM-V 67/140, CM-S 15/140, CM-I 9/140 |
| External solver comparison (B) | 200 fixed items across four eligible CaptchaMind benchmark types | GEM-V 115/200, CM-S 88/200, CM-I 101/200 |
| Paired defense study (D) | Same 24 formal base images × four conditions × three protocols | See table below |

| Condition | GEM-V | CM-S | CM-I |
| --- | ---: | ---: | ---: |
| D0: original grid selection | 24/24 | 13/24 | 16/24 |
| D1: all visible eye centers, 15 px matching | 9/24 | 0/24 | 2/24 |
| D2: visible-eye count | 20/24 | 19/24 | 18/24 |
| D3: grid selection after fixed 2 px blur | 24/24 | 16/24 | 13/24 |

The stored identifiers `G-V`, `C-S`, and `C-I` correspond to manuscript labels GEM-V, CM-S, and CM-I. Seven ambiguous CM-S D1 answers count as failures under the documented post-generation policy; their raw text and null coordinate readings are retained. One pre-identified A source annotation defect is kept in the full 141-item diagnostic view and excluded from the 140-item source-valid view. Development records are separate and do not enter formal estimates. See [PROTOCOL.md](docs/PROTOCOL.md).

## Contents

| Path | Contents |
| --- | --- |
| `conference/` | Necessary conference evaluation/analysis modules, cleaned data in the release, saved results, and figure notebooks |
| `tdsc/src/tdsc_eval/` | Static-answer evaluation, interaction environments, answer extraction, and scoring |
| `tdsc/manifests/` | Frozen formal/development item selections and provenance |
| `tdsc/results/answer_20260913/` | Original static-answer records |
| `tdsc/reports/answer_20260913/` | Label-blind readings and independently re-scorable derived answers |
| `tdsc/reports/solver_defense_v1/handoff/completed_remote/` | Complete formal attempt records, saved responses and execution evidence |
| `tdsc/reports/solver_defense_v1/analysis/` | Reference tables, plots and machine-readable scientific summary |
| `tdsc/reports/solver_defense_v1/development/` | Separate development and numeric-interface recheck records |
| `tdsc/baseline/captchamind_native/` | Pinned upstream simulator/action code with its MIT license |
| `docs/` | Reproduction instructions, result mapping, protocol, provenance, and release scope |
| `artifact_manifest.json` | File inventory, relative source locations, and packaging transformations |

The release contains response text and actions needed to audit scoring. Repeated base64 request payloads, redundant rendered observations, host administration logs, credentials, and model weights are excluded. The one-response timeouts retain their missing-response and unknown-token status; no missing outputs are reconstructed. New inference regenerates observations from the supplied puzzle assets and simulator code.

## Provenance and terms

The immutable [conference archive](https://doi.org/10.5281/zenodo.20406852) remains the record of the USENIX Security 2026 artifact. This TDSC package adds journal-specific evidence and reorganizes the necessary code without replacing that archive. CaptchaMind code, checkpoint, and external data are pinned by their upstream revisions. See [THIRD_PARTY.md](docs/THIRD_PARTY.md) and [LICENSE.md](LICENSE.md) for attribution and the existing reuse terms.

No production CAPTCHA service is contacted by the reproduction commands. Optional inference evaluates the included offline puzzle data and may incur model-provider charges. Provider credentials must be supplied locally and are not included in the release.
