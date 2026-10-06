# Conference experiments

This directory contains the conference code and evidence needed by the integrated TDSC artifact. The permanent original archive is <https://doi.org/10.5281/zenodo.20406852>.

Exp1–Exp4 evaluate direct, optimized, finite-retry, and few-shot prompting. Exp5 contains the final expanded static/adaptive evidence. Recorded results are under `results/`; image data and few-shot assets are included in the full release archive. See the root README and `../docs/PAPER_TO_ARTIFACT.md` for the result map and offline commands.

The retained code is the dependency closure of the evaluation, adaptive, statistical, and plotting entry points. Obsolete collection/staging scripts, local workflow files, scratch notebooks, and unrelated analysis utilities are omitted. Comment/whitespace cleanup preserves Python syntax-tree semantics. The existing `pyproject.toml` and `uv.lock` record the conference dependency contract.

Run `python -m cognition.run_single_experiment --help` from this directory for the original Exp1–Exp4 CLI. New inference is separate from the default saved-output reproduction and requires locally configured provider credentials. `secrets.example.yaml` contains placeholders only.
