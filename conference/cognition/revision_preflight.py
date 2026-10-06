import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .revision_artifacts import revision_run_dir, sha256_file, sha256_text
from .run_eval import DATASET_DIR_ALIASES, SUPPORTED_TYPES, TASK_ALIASES


PREFLIGHT_SCHEMA_VERSION = "cognition.revision.preflight.v1"


class PreflightTaskSummary(BaseModel):
    task_type: str
    canonical_task_type: str
    dataset_dir: str
    ground_truth_path: str
    item_count: int
    selected_count: int
    warnings: list[str] = Field(default_factory=list)


class PreflightCostPreview(BaseModel):
    expected_request_count: int
    approximate_cost_usd: float | None = None
    pricing_source: str | None = None
    unavailable_reason: str | None = None


class PreflightReport(BaseModel):
    schema_version: str = PREFLIGHT_SCHEMA_VERSION
    run_id: str
    provider: str
    model: str
    prompt_mode: str
    selected_task_types: list[str]
    expected_request_count: int
    prompt_config: dict[str, Any]
    cost_preview: PreflightCostPreview
    output_dir: str
    manifest_path: str
    attempts_path: str
    tasks: list[PreflightTaskSummary]
    warnings: list[str] = Field(default_factory=list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate revision run inputs before paid calls.")
    parser.add_argument("--dataset-root", default="./captcha_data")
    parser.add_argument("--types", nargs="+", required=True)
    parser.add_argument("--prompts-file", default="./prompts_optimized.yaml")
    parser.add_argument("--few-shot-config", default=None)
    parser.add_argument("--prompt-prefix", default=None)
    parser.add_argument("--prompt-suffix", default=None)
    parser.add_argument("--pricing-file", default=None)
    parser.add_argument("--output-root", default="./results/local_runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-mode", choices=["auto", "gt", "opt"], default="auto")
    parser.add_argument("--max-per-type", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    return parser


def _load_mapping(path: str | Path | None, label: str) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"{label} does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping: {config_path}")
    return payload


def _load_ground_truth(path: Path) -> dict[str, Any] | list[Any]:
    if not path.exists():
        raise FileNotFoundError(f"ground_truth.json does not exist: {path}")
    raw = path.read_text(encoding="utf-8")
    if raw.strip().startswith("version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(f"{path} is a Git LFS pointer; materialize dataset files first")
    payload = json.loads(raw)
    if not isinstance(payload, (dict, list)) or not payload:
        raise ValueError(f"ground_truth.json must be a non-empty list or mapping: {path}")
    return payload


def _canonical_task_type(task_type: str) -> str:
    canonical = TASK_ALIASES.get(task_type, task_type)
    if canonical not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported task type: {task_type}")
    return canonical


def _dataset_dir_name(canonical_task_type: str) -> str:
    return DATASET_DIR_ALIASES.get(canonical_task_type, canonical_task_type)


def _selected_count(item_count: int, max_per_type: int | None) -> int:
    if max_per_type is not None and max_per_type > 0:
        return min(item_count, max_per_type)
    return item_count


def _build_prompt_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prompts_file": args.prompts_file,
        "prompts_file_sha256": sha256_file(args.prompts_file),
        "few_shot_config": args.few_shot_config,
        "few_shot_config_sha256": sha256_file(args.few_shot_config),
        "prompt_prefix_sha256": sha256_text(args.prompt_prefix),
        "prompt_suffix_sha256": sha256_text(args.prompt_suffix),
    }


def _cost_preview(args: argparse.Namespace, expected_request_count: int) -> PreflightCostPreview:
    if not args.pricing_file:
        return PreflightCostPreview(
            expected_request_count=expected_request_count,
            unavailable_reason="pricing metadata not provided",
        )

    try:
        pricing_data = _load_mapping(args.pricing_file, "pricing metadata")
    except Exception as exc:
        return PreflightCostPreview(
            expected_request_count=expected_request_count,
            pricing_source=args.pricing_file,
            unavailable_reason=f"pricing metadata unreadable: {exc}",
        )

    model_pricing = (
        pricing_data.get("pricing", {})
        .get(args.provider, {})
        .get(args.model)
    )
    if not isinstance(model_pricing, dict) or "per_request_usd" not in model_pricing:
        return PreflightCostPreview(
            expected_request_count=expected_request_count,
            pricing_source=args.pricing_file,
            unavailable_reason="pricing metadata lacks per_request_usd for provider/model",
        )

    return PreflightCostPreview(
        expected_request_count=expected_request_count,
        approximate_cost_usd=expected_request_count * float(model_pricing["per_request_usd"]),
        pricing_source=args.pricing_file,
    )


def build_report(args: argparse.Namespace) -> PreflightReport:
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be >= 1")

    _load_mapping(args.prompts_file, "prompts file")
    if args.few_shot_config:
        _load_mapping(args.few_shot_config, "few-shot config")

    run_dir = revision_run_dir(args.output_root, args.run_id)
    if run_dir.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"Revision output directory exists; use --overwrite or --resume: {run_dir}"
        )

    tasks: list[PreflightTaskSummary] = []
    selected_task_types: list[str] = []
    for requested_type in args.types:
        canonical = _canonical_task_type(requested_type)
        dataset_dir = Path(args.dataset_root) / _dataset_dir_name(canonical)
        ground_truth_path = dataset_dir / "ground_truth.json"
        ground_truth = _load_ground_truth(ground_truth_path)
        item_count = len(ground_truth)
        selected = _selected_count(item_count, args.max_per_type)
        selected_task_types.append(canonical)
        tasks.append(
            PreflightTaskSummary(
                task_type=requested_type,
                canonical_task_type=canonical,
                dataset_dir=str(dataset_dir),
                ground_truth_path=str(ground_truth_path),
                item_count=item_count,
                selected_count=selected,
            )
        )

    expected_request_count = sum(task.selected_count for task in tasks) * args.max_attempts
    return PreflightReport(
        run_id=args.run_id,
        provider=args.provider,
        model=args.model,
        prompt_mode=args.prompt_mode,
        selected_task_types=selected_task_types,
        expected_request_count=expected_request_count,
        prompt_config=_build_prompt_config(args),
        cost_preview=_cost_preview(args, expected_request_count),
        output_dir=str(run_dir),
        manifest_path=str(run_dir / "run_manifest.json"),
        attempts_path=str(run_dir / "attempts.jsonl"),
        tasks=tasks,
    )


def _write_report(report: PreflightReport, overwrite: bool, resume: bool) -> Path:
    run_dir = Path(report.output_dir)
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=resume)
    report_path = run_dir / "preflight_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(mode="json"), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_report(args)
        if args.write_report:
            _write_report(report, overwrite=args.overwrite, resume=args.resume)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
