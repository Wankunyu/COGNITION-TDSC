import argparse
import csv
import hashlib
import inspect
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image
from pydantic import ValidationError

from . import adaptive_attacker
from . import adaptive_preflight
from . import revision_preflight
from . import run_eval
from .adaptive_artifacts import FEEDBACK_MODE, MEMORY_MODE, SAMPLING_MODE, STOPPING_RULE
from .phase042_artifacts import (
    PHASE042_ADAPTIVE_SUMMARY_SCHEMA_VERSION,
    PHASE042_EVIDENCE_ANALYSIS_SCHEMA_VERSION,
    PHASE042_GPT_IMAGE_SOURCE_KIND,
    PHASE042_FINAL_EVIDENCE_SCHEMA_VERSION,
    PHASE042_PREFLIGHT_MATRIX_SCHEMA_VERSION,
    PHASE042_REAL_EXTERNAL_SOURCE_KINDS,
    PHASE042_SELECTED_SOURCE_KINDS,
    PHASE042_STATIC_SUMMARY_SCHEMA_VERSION,
    Phase042AdaptiveSummaryRow,
    Phase042EvidenceAnalysisRow,
    Phase042FinalEvidenceRow,
    Phase042PreflightMatrixRow,
    Phase042SelectedManifestRow,
    Phase042StaticSummaryRow,
    Phase042ValidationReportRow,
    write_csv,
    write_json,
    write_phase042_selected_manifest,
    write_phase042_validation_report,
)
from .revision_artifacts import AttemptRecord, revision_run_dir
from .statistical_confidence import wilson_interval


PHASE042_SIDECAR_ROOT = Path("expanded_captcha_data/phase04_2")
PHASE042_CANDIDATES_ROOT = PHASE042_SIDECAR_ROOT / "candidates"
PHASE042_VALIDATION_REPORT_CSV = PHASE042_SIDECAR_ROOT / "phase042_validation_report.csv"
PHASE042_VALIDATION_REPORT_JSON = PHASE042_SIDECAR_ROOT / "phase042_validation_report.json"
PHASE042_SELECTED_MANIFEST_CSV = PHASE042_SIDECAR_ROOT / "phase042_selected_manifest.csv"
PHASE042_SELECTED_MANIFEST_JSON = PHASE042_SIDECAR_ROOT / "phase042_selected_manifest.json"
PHASE042_NOVELTY_HASH_REPORT_JSON = PHASE042_SIDECAR_ROOT / "novelty_hash_report.json"
PHASE042_EVALUATOR_SLICE = PHASE042_SIDECAR_ROOT / "evaluator_slice"
PHASE042_ADAPTIVE_EVALUATOR_SLICE = PHASE042_SIDECAR_ROOT / "adaptive_evaluator_slice"
PHASE042_STATIC_SUPPLEMENTAL_RUN_ID = "phase04_2_static_supplemental"
PHASE042_ADAPTIVE_SUPPLEMENTAL_RUN_ID = "phase04_2_adaptive_supplemental"
PHASE042_EVIDENCE_ANALYSIS_RUN_ID = "exp5_evidence_analysis"
PHASE042_EVIDENCE_ANALYSIS_DIR_NAME = "evidence_analysis"
PHASE042_FINAL_OUTPUTS_RUN_ID = "exp5_final_outputs_20260522"
PHASE042_FINAL_OUTPUTS_DIR_NAME = "final_outputs_20260522"
PHASE042_PUBLIC_PRICING_FILE = Path("pricing.phase04_2.yaml")
PHASE042_TARGET_TASK_TYPES = (
    "Dice_Count",
    "Click_Order",
    "Patch_Select",
    "Geometry_Click",
    "Symbol_Count",
    "Relation_Match",
    "Hole_Counting",
)
PHASE042_TARGET_NEW_TASK_TYPES = {"Symbol_Count", "Relation_Match", "Hole_Counting"}
PHASE042_ORIGINAL_HARD_TASK_TYPES = (
    "Dice_Count",
    "Place_Dot",
    "Pick_Area",
    "Click_Order",
    "Patch_Select",
    "Rotation_Match",
)
PHASE042_ADAPTIVE_TASK_TYPES = (
    *PHASE042_ORIGINAL_HARD_TASK_TYPES,
    "Symbol_Count",
    "Relation_Match",
    "Hole_Counting",
)
PHASE042_MIN_SAMPLES_PER_CATEGORY = 10
PHASE042_SOURCE_PRIORITY = (
    "peer_reviewed_paper_dataset",
    "open_source_dataset",
    "gpt_image_open_captchaworld_style",
)

PHASE042_INVALID_REFERENCE_MARKERS = (
    "phase04_1",
    "synthetic_fixture",
    "Copied from existing local captcha_data",
    "Scripted local prototype",
)
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
PHASE042_TASK_FAMILIES = {
    "Dice_Count": "Counting",
    "Click_Order": "Ordering",
    "Patch_Select": "Spatial Selection",
    "Geometry_Click": "Spatial Precision",
    "Symbol_Count": "Counting",
    "Relation_Match": "Relational Matching",
    "Hole_Counting": "Counting",
}
PHASE042_REPORTED_PROVIDER_MODELS = [
    "openai/gpt-5",
    "openai/gpt-5.1_medium",
    "openai/gpt-5.1_none",
    "anthropic/claude-sonnet-4-5",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
    "openrouter/qwen_qwen3-vl-235b-a22b-instruct",
]
PHASE042_LEGACY_FIREWORKS_QWEN_PROVIDER_MODEL = (
    "fireworks/accounts_fireworks_models_qwen3-vl-235b-a22b-instruct"
)
PHASE042_OPENROUTER_QWEN_PROVIDER_MODEL = (
    "openrouter/qwen_qwen3-vl-235b-a22b-instruct"
)
PHASE042_OPENROUTER_QWEN_MODEL_LABEL = "qwen_qwen3-vl-235b-a22b-instruct"
PHASE042_OPENROUTER_QWEN_RUNTIME_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
PHASE042_STATIC_TASK_TYPES = ("Symbol_Count", "Relation_Match", "Hole_Counting")
PHASE042_ADAPTIVE_ATTEMPT_BUDGET_K = 5
PHASE042_ADAPTIVE_INTERMEDIATE_BUDGET_K = 3
PHASE042_ADAPTIVE_ROUND_COUNT = 5
PHASE042_ADAPTIVE_SCOPE_RATIONALE = (
    "Since most recognition-oriented tasks are already solved reliably under the "
    "non-adaptive setting, adaptive feedback would mainly introduce ceiling effects "
    "and provide limited additional insight. We therefore focus on the hard and "
    "boundary-hard task types, where adaptive memory has the greatest potential to "
    "change the security conclusion. This is the Phase 04.2 ceiling-effect rationale."
)
PHASE042_ADAPTIVE_EXTERNAL_REFERENCE_RATES = {
    "Hole_Counting": 0.0,
    "Relation_Match": 0.0,
    "Symbol_Count": 0.03,
}
PHASE042_FINAL_STATIC_SUMMARY_PATHS = (
    Path("results/exp5/static_final_evidence_20260522")
    / "expanded_static_summary.json",
)
PHASE042_FINAL_ADAPTIVE_SUMMARY_PATH = (
    Path("results/exp5/adaptive_final_evidence_20260522")
    / "expanded_adaptive_summary.json"
)
PHASE042_FINAL_EVIDENCE_ANALYSIS_PATH = (
    Path("results/exp5")
    / PHASE042_EVIDENCE_ANALYSIS_DIR_NAME
    / "exp5_evidence_analysis.json"
)
PHASE042_ANALYSIS_CUTOFF = 0.40
PHASE042_MEMORY_ISOLATION_LABEL = "five_memory_isolated_rounds"
PHASE042_FINAL_INVALID_MARKERS = (
    "phase04_1",
    "synthetic_fixture",
    "Copied from existing local captcha_data",
    "Scripted local prototype",
    "phase04_1-symbol-count",
    "phase04_1-relation-match",
)
PHASE042_FINAL_SCAN_SUFFIXES = {".csv", ".json", ".md", ".tex", ".latex", ".txt"}


def _stable_text(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def assert_no_phase041_reference(value: object, *, context: str) -> None:
    text = _stable_text(value).lower()
    for marker in PHASE042_INVALID_REFERENCE_MARKERS:
        if marker.lower() in text:
            raise ValueError(
                f"{context} must not reference invalid Phase 04.1/source marker: "
                f"{marker}"
            )


def normalize_phase042_candidate_path(raw_path: str, *, context: str) -> str:
    assert_no_phase041_reference(raw_path, context=context)
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError(f"{context} must be relative to the project root")
    if ".." in path.parts:
        raise ValueError(f"{context} must not contain path traversal")
    try:
        path.relative_to(PHASE042_CANDIDATES_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{context} must resolve under {PHASE042_CANDIDATES_ROOT.as_posix()}"
        ) from exc
    return path.as_posix()


def _priority_rank(source_kind: str) -> int:
    try:
        return PHASE042_SOURCE_PRIORITY.index(source_kind)
    except ValueError as exc:
        raise ValueError(
            f"source_kind must be one of {list(PHASE042_SOURCE_PRIORITY)}"
        ) from exc


def _evidence_role(source_kind: str) -> str:
    if source_kind == "gpt_image_open_captchaworld_style":
        return "fallback_direct_evidence_with_provenance_caveat"
    return "preferred_direct_evidence"


def _default_license_status(source_kind: str) -> str:
    if source_kind == "gpt_image_open_captchaworld_style":
        return "generation_metadata_required"
    return "license_review_required_before_staging"


def build_source_triage_rows(
    source_candidates: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(source_candidates, start=1):
        assert_no_phase041_reference(candidate, context=f"source candidate {index}")
        task_type = str(candidate.get("task_type", ""))
        if task_type not in PHASE042_TARGET_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {list(PHASE042_TARGET_TASK_TYPES)}"
            )

        source_kind = str(candidate.get("source_kind", ""))
        source_priority_rank = _priority_rank(source_kind)
        candidate_path = candidate.get("candidate_path")
        row: dict[str, object] = {
            "task_type": task_type,
            "source_kind": source_kind,
            "source_priority_rank": source_priority_rank,
            "source_citation": str(candidate.get("source_citation", "")),
            "source_license": str(candidate.get("source_license", "")),
            "source_provenance_notes": str(
                candidate.get("source_provenance_notes", "")
            ),
            "sample_count_target": PHASE042_MIN_SAMPLES_PER_CATEGORY,
            "fallback_reason": str(candidate.get("fallback_reason", "")),
            "evidence_role": _evidence_role(source_kind),
            "license_status": str(
                candidate.get("license_status", _default_license_status(source_kind))
            ),
            "style_consistency_notes": str(
                candidate.get("style_consistency_notes", "")
            ),
        }
        if candidate_path is not None:
            row["candidate_path"] = normalize_phase042_candidate_path(
                str(candidate_path),
                context=f"source candidate {index} candidate_path",
            )
        rows.append(row)

    task_order = {task_type: index for index, task_type in enumerate(PHASE042_TARGET_TASK_TYPES)}
    return sorted(
        rows,
        key=lambda row: (
            task_order[str(row["task_type"])],
            int(row["source_priority_rank"]),
            str(row["source_citation"]),
            str(row.get("candidate_path", "")),
        ),
    )


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _iter_string_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for child in value.values():
            strings.extend(_iter_string_values(child))
        return strings
    if isinstance(value, list):
        strings = []
        for child in value:
            strings.extend(_iter_string_values(child))
        return strings
    return []


def _referenced_files(ground_truth: dict[str, object]) -> list[str]:
    referenced: set[str] = set()
    for puzzle_id, entry in ground_truth.items():
        if isinstance(puzzle_id, str) and Path(puzzle_id).suffix.lower() in IMAGE_SUFFIXES:
            referenced.add(puzzle_id)
        for value in _iter_string_values(entry):
            if Path(value).suffix.lower() in IMAGE_SUFFIXES:
                referenced.add(value)
    return sorted(referenced)


def _iter_image_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256_image(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_captcha_data_hash_index(captcha_root: Path) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for path in _iter_image_files(Path(captcha_root)):
        image_hash = sha256_image(path)
        index.setdefault(image_hash, []).append(path.as_posix())
    return dict(sorted(index.items()))


def average_hash(path: Path, hash_size: int = 8) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((hash_size, hash_size))
        pixels = list(gray.getdata())
    threshold = sum(pixels) / len(pixels)
    bits = 0
    for index, pixel in enumerate(pixels):
        if pixel >= threshold:
            bits |= 1 << index
    return bits


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _build_perceptual_index(captcha_root: Path) -> list[tuple[int, str]]:
    perceptual_index: list[tuple[int, str]] = []
    for path in _iter_image_files(Path(captcha_root)):
        try:
            perceptual_index.append((average_hash(path), path.as_posix()))
        except OSError:
            continue
    return perceptual_index


def _load_candidate_rows(candidate_manifest_path: Path) -> list[dict[str, Any]]:
    payload = _read_json(candidate_manifest_path)
    if isinstance(payload, dict):
        rows = payload.get("candidate_rows", payload.get("rows"))
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("candidate manifest must be a JSON object or array")
    if not isinstance(rows, list):
        raise ValueError("candidate manifest must contain candidate_rows")
    typed_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"candidate row {index} must be a JSON object")
        typed_rows.append(dict(row))
    return typed_rows


def _resolve_candidate_image_path(raw_path: str, project_root: Path) -> Path:
    assert_no_phase041_reference(raw_path, context="candidate image path")
    path = Path(raw_path)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        if ".." in path.parts:
            raise ValueError("candidate image paths must not contain path traversal")
        resolved = (project_root / path).resolve()

    candidate_root = (project_root / PHASE042_CANDIDATES_ROOT).resolve()
    captcha_root = (project_root / "captcha_data").resolve()
    if resolved == captcha_root or _is_relative_to(resolved, captcha_root):
        raise ValueError("candidate image paths must not resolve under captcha_data")
    if not _is_relative_to(resolved, candidate_root):
        raise ValueError(
            f"candidate image paths must resolve under {PHASE042_CANDIDATES_ROOT}"
        )
    return resolved


def _candidate_image_paths(row: dict[str, Any]) -> list[str]:
    raw_paths = row.get("candidate_image_paths")
    if raw_paths is None and row.get("candidate_path") is not None:
        raw_paths = [row["candidate_path"]]
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("candidate_image_paths must contain at least one image path")
    paths: list[str] = []
    for value in raw_paths:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("candidate_image_paths entries must be non-empty strings")
        paths.append(value)
    return paths


def _require_candidate_text(row: dict[str, Any], field_name: str, context: str) -> None:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required when {context}")
    if value.strip().lower() == "pending_generation":
        raise ValueError(f"{field_name} must be final metadata, not pending_generation")


def _source_path_for_images(image_paths: list[str]) -> str:
    if not image_paths:
        return PHASE042_CANDIDATES_ROOT.as_posix()
    return Path(image_paths[0]).parent.as_posix()


def _evidence_origin(task_type: str) -> str:
    if task_type in PHASE042_TARGET_NEW_TASK_TYPES:
        return "new_category"
    return "supplemented_category"


def _slice_type(task_type: str) -> str:
    if task_type in PHASE042_TARGET_NEW_TASK_TYPES:
        return "new_category"
    return "supplement_existing"


def _selected_row_values(
    row: dict[str, Any],
    *,
    image_paths: list[str],
    exact_match_paths: list[str],
    novelty_hashes: list[str],
    novelty_hash_report_path: Path,
    perceptual_warning_count: int,
    review_warnings: list[str],
) -> dict[str, Any]:
    task_type = str(row.get("task_type", ""))
    source_kind = str(row.get("source_kind", ""))
    if source_kind == PHASE042_GPT_IMAGE_SOURCE_KIND:
        default_provenance_class = "gpt_image_generated_fallback"
    else:
        default_provenance_class = "preferred_real_external"
    return {
        "selected_id": str(row.get("selected_id") or row.get("candidate_id")),
        "candidate_id": str(row.get("candidate_id")),
        "source_path": str(row.get("source_path") or _source_path_for_images(image_paths)),
        "candidate_image_paths": image_paths,
        "source_kind": source_kind,
        "source_provenance_class": str(
            row.get("source_provenance_class") or default_provenance_class
        ),
        "source_citation": str(row.get("source_citation", "")),
        "source_license": str(row.get("source_license", "")),
        "source_provenance_notes": str(row.get("source_provenance_notes", "")),
        "evidence_origin": str(row.get("evidence_origin") or _evidence_origin(task_type)),
        "slice_type": str(row.get("slice_type") or _slice_type(task_type)),
        "task_type": task_type,
        "task_family": str(row.get("task_family") or PHASE042_TASK_FAMILIES.get(task_type, "")),
        "sample_count": len(image_paths),
        "label_format": str(row.get("label_format", "task_specific")),
        "metadata_alignment_notes": str(row.get("metadata_alignment_notes", "")),
        "answer_format_normalization": str(
            row.get("answer_format_normalization", "task-specific normalization")
        ),
        "compatibility_status": str(
            row.get("compatibility_status", "ready_for_static_pipeline")
        ),
        "evaluation_status": str(row.get("evaluation_status", "selected_for_static")),
        "limitation_notes": str(row.get("limitation_notes", "")),
        "adaptive_eligible": bool(row.get("adaptive_eligible", True)),
        "static_compatibility_notes": str(row.get("static_compatibility_notes", "")),
        "novelty_sha256": novelty_hashes,
        "novelty_hash_report_path": novelty_hash_report_path.as_posix(),
        "exact_captcha_data_match": bool(exact_match_paths),
        "perceptual_warning_count": perceptual_warning_count,
        "review_warnings": review_warnings,
        "gpt_image_generation_prompt": str(row.get("gpt_image_generation_prompt", "")),
        "gpt_image_model": str(row.get("gpt_image_model", "")),
        "gpt_image_generation_date": str(row.get("gpt_image_generation_date", "")),
        "open_captchaworld_style_rationale": str(
            row.get("open_captchaworld_style_rationale", "")
        ),
    }


def _report_row_values(
    row: dict[str, Any],
    *,
    image_paths: list[str],
    validation_status: str,
    selected_manifest_eligible: bool,
    rejection_reason: str,
    exact_match_paths: list[str],
    novelty_hashes: list[str],
    novelty_hash_report_path: Path,
    perceptual_warning_count: int,
    review_warnings: list[str],
) -> dict[str, Any]:
    return {
        "candidate_id": str(row.get("candidate_id", "")),
        "task_type": str(row.get("task_type", "")),
        "source_kind": str(row.get("source_kind", "")),
        "source_path": str(row.get("source_path") or _source_path_for_images(image_paths)),
        "candidate_image_paths": image_paths,
        "validation_status": validation_status,
        "selected_manifest_eligible": selected_manifest_eligible,
        "rejection_reason": rejection_reason,
        "exact_captcha_data_match": bool(exact_match_paths),
        "exact_captcha_data_match_paths": exact_match_paths,
        "novelty_sha256": novelty_hashes,
        "novelty_hash_report_path": novelty_hash_report_path.as_posix(),
        "perceptual_warning_count": perceptual_warning_count,
        "review_warnings": review_warnings,
    }


def _perceptual_warnings_for_paths(
    resolved_paths: list[Path],
    perceptual_index: list[tuple[int, str]],
    threshold: int,
) -> list[str]:
    warnings: list[str] = []
    if not perceptual_index:
        return warnings
    for path in resolved_paths:
        try:
            candidate_hash = average_hash(path)
        except OSError:
            continue
        for baseline_hash, baseline_path in perceptual_index:
            distance = _hamming_distance(candidate_hash, baseline_hash)
            if distance <= threshold:
                warnings.append(
                    f"perceptual near match warning: {path.as_posix()} is "
                    f"distance {distance} from {baseline_path}"
                )
                break
    return warnings


def _validate_source_specific_fields(row: dict[str, Any]) -> None:
    source_kind = str(row.get("source_kind", ""))
    if source_kind not in PHASE042_SELECTED_SOURCE_KINDS:
        raise ValueError(
            f"source_kind must be one of {sorted(PHASE042_SELECTED_SOURCE_KINDS)}"
        )
    if source_kind in PHASE042_REAL_EXTERNAL_SOURCE_KINDS:
        _require_candidate_text(row, "source_citation", "candidate is a real external row")
        _require_candidate_text(row, "source_license", "candidate is a real external row")
        _require_candidate_text(
            row,
            "source_provenance_notes",
            "candidate is a real external row",
        )
    if source_kind == PHASE042_GPT_IMAGE_SOURCE_KIND:
        _require_candidate_text(
            row,
            "source_provenance_notes",
            "candidate is a GPT Image row",
        )
        _require_candidate_text(
            row,
            "gpt_image_generation_prompt",
            "candidate is a GPT Image row",
        )
        _require_candidate_text(
            row,
            "gpt_image_model",
            "candidate is a GPT Image row",
        )
        _require_candidate_text(
            row,
            "gpt_image_generation_date",
            "candidate is a GPT Image row",
        )
        _require_candidate_text(
            row,
            "open_captchaworld_style_rationale",
            "candidate is a GPT Image row",
        )


def _validate_one_candidate(
    row: dict[str, Any],
    *,
    captcha_hash_index: dict[str, list[str]],
    perceptual_index: list[tuple[int, str]],
    novelty_hash_report_path: Path,
    project_root: Path,
    perceptual_warning_threshold: int,
) -> tuple[Phase042ValidationReportRow, Phase042SelectedManifestRow | None, list[str]]:
    image_paths: list[str] = []
    novelty_hashes: list[str] = []
    exact_match_paths: list[str] = []
    review_warnings: list[str] = []
    try:
        assert_no_phase041_reference(row, context="candidate row")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id is required")
        task_type = row.get("task_type")
        if task_type not in PHASE042_TARGET_TASK_TYPES:
            raise ValueError(f"task_type must be one of {list(PHASE042_TARGET_TASK_TYPES)}")
        _validate_source_specific_fields(row)
        image_paths = _candidate_image_paths(row)
        resolved_paths = [
            _resolve_candidate_image_path(raw_path, project_root)
            for raw_path in image_paths
        ]
        missing_paths = [path.as_posix() for path in resolved_paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(
                "missing candidate image paths: " + ", ".join(missing_paths)
            )
        for path in resolved_paths:
            image_hash = sha256_image(path)
            novelty_hashes.append(image_hash)
            exact_match_paths.extend(captcha_hash_index.get(image_hash, []))
        perceptual_warnings = _perceptual_warnings_for_paths(
            resolved_paths,
            perceptual_index,
            perceptual_warning_threshold,
        )
        review_warnings = list(perceptual_warnings)
        if exact_match_paths:
            review_warnings.insert(
                0,
                "exact SHA-256 match warning: candidate image hash already exists "
                "under current captcha_data: "
                + ", ".join(sorted(set(exact_match_paths))),
            )
        selected_row = Phase042SelectedManifestRow.model_validate(
            _selected_row_values(
                row,
                image_paths=image_paths,
                exact_match_paths=sorted(set(exact_match_paths)),
                novelty_hashes=novelty_hashes,
                novelty_hash_report_path=novelty_hash_report_path,
                perceptual_warning_count=len(perceptual_warnings),
                review_warnings=review_warnings,
            )
        )
        report_row = Phase042ValidationReportRow.model_validate(
            _report_row_values(
                row,
                image_paths=image_paths,
                validation_status="accepted",
                selected_manifest_eligible=True,
                rejection_reason="",
                exact_match_paths=sorted(set(exact_match_paths)),
                novelty_hashes=novelty_hashes,
                novelty_hash_report_path=novelty_hash_report_path,
                perceptual_warning_count=len(perceptual_warnings),
                review_warnings=review_warnings,
            )
        )
        return report_row, selected_row, review_warnings
    except (FileNotFoundError, OSError, ValueError, ValidationError) as exc:
        rejection_reason = str(exc)
        if isinstance(exc, ValidationError):
            rejection_reason = "; ".join(error["msg"] for error in exc.errors())
        if not image_paths:
            try:
                image_paths = _candidate_image_paths(row)
            except ValueError:
                image_paths = []
        report_row = Phase042ValidationReportRow.model_validate(
            _report_row_values(
                row,
                image_paths=image_paths,
                validation_status="rejected",
                selected_manifest_eligible=False,
                rejection_reason=rejection_reason,
                exact_match_paths=sorted(set(exact_match_paths)),
                novelty_hashes=novelty_hashes,
                novelty_hash_report_path=novelty_hash_report_path,
                perceptual_warning_count=len(review_warnings),
                review_warnings=review_warnings,
            )
        )
        return report_row, None, review_warnings


def validate_phase042_candidates(
    *,
    candidate_manifest_path: Path = PHASE042_SIDECAR_ROOT / "candidate_manifest.json",
    captcha_root: Path = Path("captcha_data"),
    validation_report_csv: Path = PHASE042_VALIDATION_REPORT_CSV,
    validation_report_json: Path = PHASE042_VALIDATION_REPORT_JSON,
    selected_manifest_csv: Path = PHASE042_SELECTED_MANIFEST_CSV,
    selected_manifest_json: Path = PHASE042_SELECTED_MANIFEST_JSON,
    novelty_hash_report_path: Path = PHASE042_NOVELTY_HASH_REPORT_JSON,
    perceptual_warning_threshold: int = 4,
) -> dict[str, Any]:
    project_root = Path.cwd().resolve()
    candidate_rows = _load_candidate_rows(Path(candidate_manifest_path))
    captcha_hash_index = build_captcha_data_hash_index(Path(captcha_root))
    has_existing_candidate_files = False
    for row in candidate_rows:
        try:
            raw_paths = _candidate_image_paths(row)
            resolved_paths = [
                _resolve_candidate_image_path(raw_path, project_root)
                for raw_path in raw_paths
            ]
        except ValueError:
            continue
        if any(path.is_file() for path in resolved_paths):
            has_existing_candidate_files = True
            break
    perceptual_index = (
        _build_perceptual_index(Path(captcha_root)) if has_existing_candidate_files else []
    )

    report_rows: list[Phase042ValidationReportRow] = []
    selected_rows: list[Phase042SelectedManifestRow] = []
    all_review_warnings: list[str] = []
    for row in candidate_rows:
        report_row, selected_row, review_warnings = _validate_one_candidate(
            row,
            captcha_hash_index=captcha_hash_index,
            perceptual_index=perceptual_index,
            novelty_hash_report_path=Path(novelty_hash_report_path),
            project_root=project_root,
            perceptual_warning_threshold=perceptual_warning_threshold,
        )
        report_rows.append(report_row)
        if selected_row is not None:
            selected_rows.append(selected_row)
        all_review_warnings.extend(review_warnings)

    write_phase042_validation_report(
        Path(validation_report_csv),
        Path(validation_report_json),
        report_rows,
    )
    write_phase042_selected_manifest(
        Path(selected_manifest_csv),
        Path(selected_manifest_json),
        selected_rows,
    )

    exact_match_rejection_count = sum(
        1
        for row in report_rows
        if row.exact_captcha_data_match and row.validation_status == "rejected"
    )
    exact_match_selected_count = sum(
        1
        for row in report_rows
        if row.exact_captcha_data_match and row.validation_status == "accepted"
    )
    candidate_hash_count = sum(len(row.novelty_sha256) for row in report_rows)
    review_warning_count = sum(len(row.review_warnings) for row in report_rows)
    perceptual_warning_count = sum(row.perceptual_warning_count for row in report_rows)
    novelty_payload = {
        "schema_version": "cognition.revision.phase042.novelty_hash_report.v1",
        "captcha_data_hash_index_size": len(captcha_hash_index),
        "captcha_data_image_count": sum(len(paths) for paths in captcha_hash_index.values()),
        "candidate_count": len(candidate_rows),
        "candidate_hash_count": candidate_hash_count,
        "exact_match_rejection_count": exact_match_rejection_count,
        "exact_match_selected_count": exact_match_selected_count,
        "review_warning_count": review_warning_count,
        "perceptual_near_match_warning_count": perceptual_warning_count,
        "perceptual_near_match_warnings": [
            warning
            for warning in all_review_warnings
            if warning.startswith("perceptual near match warning:")
        ],
        "review_warnings": all_review_warnings,
        "selected_count": len(selected_rows),
        "rejected_count": len(report_rows) - len(selected_rows),
        "validation_report_json": Path(validation_report_json).as_posix(),
        "selected_manifest_json": Path(selected_manifest_json).as_posix(),
    }
    _write_json(Path(novelty_hash_report_path), novelty_payload)
    return {
        "report_count": len(report_rows),
        "selected_count": len(selected_rows),
        "rejected_count": len(report_rows) - len(selected_rows),
        "validation_report_csv": Path(validation_report_csv),
        "validation_report_json": Path(validation_report_json),
        "selected_manifest_csv": Path(selected_manifest_csv),
        "selected_manifest_json": Path(selected_manifest_json),
        "novelty_hash_report": Path(novelty_hash_report_path),
    }


def _assert_selected_manifest_path(path: Path) -> None:
    normalized = path.as_posix().lower()
    if "validation_report" in normalized:
        raise ValueError("materialization must read a selected manifest, not a validation report")
    assert_no_phase041_reference(normalized, context="selected manifest path")
    if path.name not in {"phase042_selected_manifest.json", "selected_manifest.json"}:
        raise ValueError(
            "selected manifest path must be phase042_selected_manifest.json "
            "or selected_manifest.json"
        )


def load_phase042_selected_manifest(
    selected_manifest_path: str | Path = PHASE042_SELECTED_MANIFEST_JSON,
) -> list[Phase042SelectedManifestRow]:
    path = Path(selected_manifest_path)
    _assert_selected_manifest_path(path)
    payload = _read_json(path)
    assert_no_phase041_reference(payload, context="selected manifest payload")
    raw_rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(raw_rows, list):
        raise ValueError("selected manifest must contain a rows array")
    return [Phase042SelectedManifestRow.model_validate(row) for row in raw_rows]


def normalize_phase042_selected_manifest_scope(
    rows: Iterable[Phase042SelectedManifestRow | dict[str, Any]],
) -> tuple[list[Phase042SelectedManifestRow], int]:
    validated = [
        row
        if isinstance(row, Phase042SelectedManifestRow)
        else Phase042SelectedManifestRow.model_validate(row)
        for row in rows
    ]
    corrected: list[Phase042SelectedManifestRow] = []
    excluded_updated_ocw_increment_count = 0
    for row in validated:
        if row.task_type in PHASE042_TARGET_NEW_TASK_TYPES:
            corrected.append(row)
            continue
        excluded_updated_ocw_increment_count += 1

    validate_phase042_selected_manifest(corrected)
    return sorted(corrected, key=lambda row: row.task_type), excluded_updated_ocw_increment_count


def validate_phase042_selected_manifest(
    rows: Iterable[Phase042SelectedManifestRow | dict[str, Any]],
) -> list[Phase042SelectedManifestRow]:
    validated = [
        row
        if isinstance(row, Phase042SelectedManifestRow)
        else Phase042SelectedManifestRow.model_validate(row)
        for row in rows
    ]
    by_task: dict[str, list[Phase042SelectedManifestRow]] = defaultdict(list)
    for row in validated:
        assert_no_phase041_reference(row.model_dump(mode="json"), context="selected manifest row")
        if row.task_type not in PHASE042_TARGET_NEW_TASK_TYPES:
            raise ValueError(
                "corrected Phase 04.2 direct static selection is limited to "
                f"{sorted(PHASE042_TARGET_NEW_TASK_TYPES)}; got {row.task_type}"
            )
        if row.sample_count < PHASE042_MIN_SAMPLES_PER_CATEGORY:
            raise ValueError(
                f"{row.task_type} sample_count must be >= "
                f"{PHASE042_MIN_SAMPLES_PER_CATEGORY}"
            )
        by_task[row.task_type].append(row)

    task_types = set(by_task)
    if task_types != PHASE042_TARGET_NEW_TASK_TYPES:
        raise ValueError(
            "corrected Phase 04.2 selected manifest must contain exactly "
            f"{sorted(PHASE042_TARGET_NEW_TASK_TYPES)}"
        )
    duplicates = {task_type: rows for task_type, rows in by_task.items() if len(rows) != 1}
    if duplicates:
        raise ValueError(
            "corrected Phase 04.2 selected manifest requires one row per task type: "
            f"{sorted(duplicates)}"
        )
    return sorted(validated, key=lambda row: row.task_type)


def _safe_prepare_phase042_output_root(
    sidecar_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
) -> Path:
    assert_no_phase041_reference(output_root.as_posix(), context="materialization output root")
    sidecar_resolved = sidecar_root.resolve()
    output_resolved = output_root.resolve()
    captcha_root = Path("captcha_data").resolve()
    if output_resolved == captcha_root or _is_relative_to(output_resolved, captcha_root):
        raise ValueError("materialization output_root must not be inside captcha_data")
    if output_resolved == sidecar_resolved:
        raise ValueError("materialization output_root must not be the sidecar root")
    if not _is_relative_to(output_resolved, sidecar_resolved):
        raise ValueError("materialization output_root must stay under Phase 04.2 sidecar root")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"output_root already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _resolve_phase042_source_dir(raw_path: str, sidecar_root: Path) -> Path:
    assert_no_phase041_reference(raw_path, context="selected manifest source_path")
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError("selected manifest source_path must be relative")
    if ".." in path.parts:
        raise ValueError("selected manifest source_path must not contain path traversal")
    resolved = path.resolve()
    candidates_root = (sidecar_root / "candidates").resolve()
    if not _is_relative_to(resolved, candidates_root):
        raise ValueError("selected manifest source_path must resolve under Phase 04.2 candidates")
    return resolved


def _copy_referenced_file(source_dir: Path, output_dir: Path, relative_path: str) -> None:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"referenced dataset file must be relative: {relative_path}")
    if ".." in relative.parts:
        raise ValueError(
            f"referenced dataset file must not contain path traversal: {relative_path}"
        )
    source_root = source_dir.resolve()
    source_path = (source_dir / relative).resolve()
    if not _is_relative_to(source_path, source_root):
        raise ValueError(f"referenced dataset file escapes source path: {relative_path}")
    if not source_path.is_file():
        raise FileNotFoundError(str(source_path))
    output_path = output_dir / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)


def _copy_task_ground_truth(
    *,
    source_dir: Path,
    task_output_root: Path,
) -> tuple[int, str]:
    source_gt_path = source_dir / "ground_truth.json"
    if not source_gt_path.is_file():
        raise FileNotFoundError(str(source_gt_path))
    ground_truth = _read_json(source_gt_path)
    if not isinstance(ground_truth, dict):
        raise ValueError(f"{source_gt_path} must contain a JSON object")
    copied_file_count = 0
    for relative_path in _referenced_files(ground_truth):
        _copy_referenced_file(source_dir, task_output_root, relative_path)
        copied_file_count += 1
    ground_truth_path = task_output_root / "ground_truth.json"
    task_output_root.mkdir(parents=True, exist_ok=True)
    with ground_truth_path.open("w", encoding="utf-8") as handle:
        json.dump(ground_truth, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return copied_file_count, ground_truth_path.as_posix()


def _write_task_materialization_metadata(
    task_output_root: Path,
    row: Phase042SelectedManifestRow,
) -> Path:
    metadata_path = task_output_root / "phase042_materialization_metadata.json"
    _write_json(
        metadata_path,
        {
            "schema_version": "cognition.revision.phase042.materialization_metadata.v1",
            "selected_id": row.selected_id,
            "candidate_id": row.candidate_id,
            "task_type": row.task_type,
            "source_kind": row.source_kind,
            "source_provenance_class": row.source_provenance_class,
            "source_citation": row.source_citation,
            "source_license": row.source_license,
            "source_provenance_notes": row.source_provenance_notes,
            "sample_count": row.sample_count,
            "selected_manifest_path": PHASE042_SELECTED_MANIFEST_JSON.as_posix(),
        },
    )
    return metadata_path


def materialize_phase042_evaluator_slice(
    rows: Iterable[Phase042SelectedManifestRow | dict[str, Any]],
    *,
    sidecar_root: str | Path = PHASE042_SIDECAR_ROOT,
    output_root: str | Path = PHASE042_EVALUATOR_SLICE,
    overwrite: bool = False,
) -> dict[str, Any]:
    sidecar = Path(sidecar_root)
    output = _safe_prepare_phase042_output_root(sidecar, Path(output_root), overwrite=overwrite)
    corrected_rows = validate_phase042_selected_manifest(rows)

    copied_file_count = 0
    ground_truth_files: list[str] = []
    metadata_files: list[str] = []
    sample_count_by_task_type: dict[str, int] = {}
    for row in corrected_rows:
        source_dir = _resolve_phase042_source_dir(row.source_path, sidecar)
        task_output_root = output / row.task_type
        copied, ground_truth_path = _copy_task_ground_truth(
            source_dir=source_dir,
            task_output_root=task_output_root,
        )
        copied_file_count += copied
        ground_truth_files.append(ground_truth_path)
        metadata_files.append(
            _write_task_materialization_metadata(task_output_root, row).as_posix()
        )
        sample_count_by_task_type[row.task_type] = row.sample_count

    return {
        "static_row_count": len(corrected_rows),
        "static_task_type_count": len(corrected_rows),
        "static_sample_count_by_task_type": dict(sorted(sample_count_by_task_type.items())),
        "copied_file_count": copied_file_count,
        "static_output_root": output.as_posix(),
        "static_ground_truth_files": sorted(ground_truth_files),
        "static_metadata_files": sorted(metadata_files),
    }


def _copy_original_hard_task(
    *,
    original_dataset_root: Path,
    task_type: str,
    adaptive_output_root: Path,
) -> tuple[int, str, int]:
    source_dir = (original_dataset_root / task_type).resolve()
    root_resolved = original_dataset_root.resolve()
    if not _is_relative_to(source_dir, root_resolved):
        raise ValueError(f"original hard task path escapes original dataset root: {task_type}")
    task_output_root = adaptive_output_root / task_type
    copied_file_count, ground_truth_path = _copy_task_ground_truth(
        source_dir=source_dir,
        task_output_root=task_output_root,
    )
    ground_truth = _read_json(task_output_root / "ground_truth.json")
    sample_count = len(ground_truth) if isinstance(ground_truth, dict) else 0
    return copied_file_count, ground_truth_path, sample_count


def materialize_phase042_adaptive_evaluator_slice(
    rows: Iterable[Phase042SelectedManifestRow | dict[str, Any]],
    *,
    sidecar_root: str | Path = PHASE042_SIDECAR_ROOT,
    output_root: str | Path = PHASE042_ADAPTIVE_EVALUATOR_SLICE,
    original_dataset_root: str | Path = "captcha_data",
    overwrite: bool = False,
) -> dict[str, Any]:
    sidecar = Path(sidecar_root)
    output = _safe_prepare_phase042_output_root(sidecar, Path(output_root), overwrite=overwrite)
    corrected_rows = validate_phase042_selected_manifest(rows)
    rows_by_task = {row.task_type: row for row in corrected_rows}

    copied_file_count = 0
    ground_truth_files: list[str] = []
    sample_count_by_task_type: dict[str, int] = {}
    original_root = Path(original_dataset_root)
    for task_type in PHASE042_ORIGINAL_HARD_TASK_TYPES:
        copied, ground_truth_path, sample_count = _copy_original_hard_task(
            original_dataset_root=original_root,
            task_type=task_type,
            adaptive_output_root=output,
        )
        copied_file_count += copied
        ground_truth_files.append(ground_truth_path)
        sample_count_by_task_type[task_type] = sample_count

    for task_type in sorted(PHASE042_TARGET_NEW_TASK_TYPES):
        row = rows_by_task[task_type]
        source_dir = _resolve_phase042_source_dir(row.source_path, sidecar)
        task_output_root = output / row.task_type
        copied, ground_truth_path = _copy_task_ground_truth(
            source_dir=source_dir,
            task_output_root=task_output_root,
        )
        copied_file_count += copied
        ground_truth_files.append(ground_truth_path)
        _write_task_materialization_metadata(task_output_root, row)
        sample_count_by_task_type[task_type] = row.sample_count

    task_types = tuple(sorted(path.name for path in output.iterdir() if path.is_dir()))
    expected = tuple(sorted(PHASE042_ADAPTIVE_TASK_TYPES))
    if task_types != expected:
        raise ValueError(f"adaptive evaluator slice must contain exactly {expected}")

    return {
        "adaptive_task_type_count": len(PHASE042_ADAPTIVE_TASK_TYPES),
        "adaptive_sample_count_by_task_type": dict(sorted(sample_count_by_task_type.items())),
        "adaptive_copied_file_count": copied_file_count,
        "adaptive_output_root": output.as_posix(),
        "adaptive_ground_truth_files": sorted(ground_truth_files),
    }


def materialize_phase042_selected(
    *,
    selected_manifest_path: str | Path = PHASE042_SELECTED_MANIFEST_JSON,
    selected_manifest_csv: str | Path = PHASE042_SELECTED_MANIFEST_CSV,
    source_download_manifest_path: str | Path = (
        PHASE042_SIDECAR_ROOT / "source_download_manifest.json"
    ),
    sidecar_root: str | Path = PHASE042_SIDECAR_ROOT,
    static_output_root: str | Path = PHASE042_EVALUATOR_SLICE,
    adaptive_output_root: str | Path = PHASE042_ADAPTIVE_EVALUATOR_SLICE,
    original_dataset_root: str | Path = "captcha_data",
    overwrite: bool = False,
) -> dict[str, Any]:
    selected_manifest_path = Path(selected_manifest_path)
    selected_manifest_hash = sha256_file(selected_manifest_path)
    rows = load_phase042_selected_manifest(selected_manifest_path)
    corrected_rows, excluded_count = normalize_phase042_selected_manifest_scope(rows)
    write_phase042_selected_manifest(
        Path(selected_manifest_csv),
        selected_manifest_path,
        corrected_rows,
    )
    corrected_hash = sha256_file(selected_manifest_path)
    static_summary = materialize_phase042_evaluator_slice(
        corrected_rows,
        sidecar_root=sidecar_root,
        output_root=static_output_root,
        overwrite=overwrite,
    )
    adaptive_summary = materialize_phase042_adaptive_evaluator_slice(
        corrected_rows,
        sidecar_root=sidecar_root,
        output_root=adaptive_output_root,
        original_dataset_root=original_dataset_root,
        overwrite=overwrite,
    )
    source_download_manifest_path = Path(source_download_manifest_path)
    summary = {
        "selected_manifest_sha256": selected_manifest_hash,
        "corrected_selected_manifest_sha256": corrected_hash,
        "source_download_manifest_sha256": (
            sha256_file(source_download_manifest_path)
            if source_download_manifest_path.is_file()
            else ""
        ),
        "selected_manifest_json": selected_manifest_path.as_posix(),
        "selected_manifest_csv": Path(selected_manifest_csv).as_posix(),
        "source_download_manifest": source_download_manifest_path.as_posix(),
        "excluded_updated_ocw_increment_count": excluded_count,
        **static_summary,
        **adaptive_summary,
    }
    summary["copied_file_count"] = int(static_summary["copied_file_count"]) + int(
        adaptive_summary["adaptive_copied_file_count"]
    )
    return summary


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-")


def _exp2_root(results_dir: Path) -> Path:
    return results_dir / "exp2" if (results_dir / "exp2").is_dir() else results_dir


def _discover_exp2_provider_models(results_dir: Path) -> set[str]:
    root = _exp2_root(results_dir)
    provider_models: set[str] = set()
    if not root.is_dir():
        return provider_models
    for provider_dir in root.iterdir():
        if not provider_dir.is_dir():
            continue
        for model_dir in provider_dir.iterdir():
            if model_dir.is_dir():
                provider_models.add(f"{provider_dir.name}/{model_dir.name}")
    if PHASE042_LEGACY_FIREWORKS_QWEN_PROVIDER_MODEL in provider_models:
        provider_models.add(PHASE042_OPENROUTER_QWEN_PROVIDER_MODEL)
    return provider_models


def _assert_no_phase041_path_values(values: Iterable[object], *, context: str) -> None:
    for value in values:
        assert_no_phase041_reference(value, context=context)


def build_phase042_reported_run_matrix(
    *,
    results_dir: str | Path = "results",
    materialized_dataset_root: str | Path = PHASE042_EVALUATOR_SLICE,
    output_root: str | Path = "results/exp5",
    run_id: str = PHASE042_STATIC_SUPPLEMENTAL_RUN_ID,
) -> list[dict[str, Any]]:
    _assert_no_phase041_path_values(
        [results_dir, materialized_dataset_root, output_root, run_id],
        context="Phase 04.2 static run matrix",
    )
    discovered = _discover_exp2_provider_models(Path(results_dir))
    missing = [
        provider_model
        for provider_model in PHASE042_REPORTED_PROVIDER_MODELS
        if provider_model not in discovered
    ]
    if missing:
        raise ValueError(f"results/exp2 is missing reported model rows: {missing}")

    rows: list[dict[str, Any]] = []
    for provider_model in PHASE042_REPORTED_PROVIDER_MODELS:
        provider, model = provider_model.split("/", 1)
        row_run_id = f"{run_id}-{_slug(provider)}-{_slug(model)}"
        rows.append(
            {
                "run_id": row_run_id,
                "provider": provider,
                "model": model,
                "provider_model": provider_model,
                "task_types": list(PHASE042_STATIC_TASK_TYPES),
                "materialized_dataset_root": str(materialized_dataset_root),
                "output_dir": str(revision_run_dir(output_root, row_run_id)),
                "overwrite": False,
                "resume": True,
            }
        )
    return rows


def _validate_materialized_dataset_root(dataset_root: Path, task_types: Iterable[str]) -> None:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"materialized dataset root does not exist: {dataset_root}")
    for task_type in task_types:
        ground_truth_path = dataset_root / task_type / "ground_truth.json"
        if not ground_truth_path.is_file():
            raise FileNotFoundError(
                f"missing ground_truth.json for {task_type}: {ground_truth_path}"
            )


def _preflight_report_payload(report: object) -> dict[str, Any]:
    if hasattr(report, "model_dump"):
        return report.model_dump(mode="json")  # type: ignore[no-any-return]
    if isinstance(report, dict):
        return dict(report)
    raise TypeError("preflight report must be a pydantic model or dict")


def _load_phase042_mapping(path: str | Path, label: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping: {config_path}")
    return payload


def _phase042_model_key_candidates(provider: str, model: str) -> list[str]:
    candidates = [model]
    if provider == "openai" and model.startswith("gpt-5.1_"):
        candidates.append("gpt-5.1")
    if provider == "openai" and model.startswith("gpt-5_"):
        candidates.append("gpt-5")
    if provider == "fireworks":
        candidates.append(model.replace("_", "/"))
    if provider == "openrouter" and model == PHASE042_OPENROUTER_QWEN_MODEL_LABEL:
        candidates.append(PHASE042_OPENROUTER_QWEN_RUNTIME_MODEL)
    candidates.extend(candidate.lower() for candidate in list(candidates))
    return list(dict.fromkeys(candidates))


def _phase042_token_summary_candidates(
    provider: str,
    model: str,
) -> list[tuple[str, str]]:
    candidates = [(provider, model)]
    if provider == "openrouter" and model == PHASE042_OPENROUTER_QWEN_MODEL_LABEL:
        candidates.append(
            (
                "fireworks",
                "accounts_fireworks_models_qwen3-vl-235b-a22b-instruct",
            )
        )
    return candidates


def _phase042_model_pricing(
    pricing_data: dict[str, Any],
    provider: str,
    model: str,
) -> tuple[str, dict[str, Any]] | None:
    provider_pricing = (pricing_data.get("pricing") or {}).get(provider, {})
    if not isinstance(provider_pricing, dict):
        return None
    for model_key in _phase042_model_key_candidates(provider, model):
        model_pricing = provider_pricing.get(model_key)
        if isinstance(model_pricing, dict):
            return model_key, model_pricing
    return None


def _phase042_exp2_token_summary_path(
    results_dir: str | Path,
    provider: str,
    model: str,
) -> Path | None:
    exp2_root = _exp2_root(Path(results_dir))
    for candidate_provider, candidate_model in _phase042_token_summary_candidates(
        provider,
        model,
    ):
        model_root = exp2_root / candidate_provider / candidate_model
        if not model_root.is_dir():
            continue
        summaries = sorted(model_root.glob("*token_summary.json"))
        if summaries:
            return summaries[0]
    return None


def _phase042_overall_token_averages(token_summary_path: Path) -> dict[str, float | int]:
    payload = _load_phase042_mapping(token_summary_path, "token summary")
    overall = payload.get("overall") or {}
    if not isinstance(overall, dict):
        raise ValueError(f"token summary overall must be a mapping: {token_summary_path}")
    question_count = int(overall.get("total_questions") or overall.get("count") or 0)
    tokens_in = int(overall.get("total_tokens_in") or overall.get("tokens_in") or 0)
    tokens_out = int(overall.get("total_tokens_out") or overall.get("tokens_out") or 0)
    if question_count <= 0:
        raise ValueError(f"token summary has no historical question count: {token_summary_path}")
    return {
        "historical_question_count": question_count,
        "tokens_in_per_request": tokens_in / question_count,
        "tokens_out_per_request": tokens_out / question_count,
    }


def _phase042_token_pricing_cost_preview(
    base_preview: dict[str, Any],
    *,
    pricing_file: str | Path | None,
    results_dir: str | Path,
    provider: str,
    model: str,
    expected_request_count: int,
) -> dict[str, Any]:
    preview = dict(base_preview)
    if preview.get("approximate_cost_usd") is not None or not pricing_file:
        return preview

    try:
        pricing_data = _load_phase042_mapping(pricing_file, "pricing metadata")
        pricing_match = _phase042_model_pricing(pricing_data, provider, model)
    except Exception as exc:
        preview["pricing_source"] = str(pricing_file)
        preview["unavailable_reason"] = f"pricing metadata unreadable: {exc}"
        return preview
    if not pricing_match:
        preview["pricing_source"] = str(pricing_file)
        preview["unavailable_reason"] = "pricing metadata not provided for provider/model"
        return preview

    pricing_model, model_pricing = pricing_match
    in_per_1k = model_pricing.get("in_per_1k")
    out_per_1k = model_pricing.get("out_per_1k")
    if in_per_1k is None or out_per_1k is None:
        return preview

    token_summary_path = _phase042_exp2_token_summary_path(results_dir, provider, model)
    if not token_summary_path:
        preview["pricing_source"] = str(pricing_file)
        preview["unavailable_reason"] = (
            "historical Exp2 token summary not found for provider/model"
        )
        return preview

    try:
        token_averages = _phase042_overall_token_averages(token_summary_path)
        tokens_in_per_request = float(token_averages["tokens_in_per_request"])
        tokens_out_per_request = float(token_averages["tokens_out_per_request"])
        expected_tokens_in = tokens_in_per_request * expected_request_count
        expected_tokens_out = tokens_out_per_request * expected_request_count
        approximate_cost = (
            expected_tokens_in * float(in_per_1k) / 1000
            + expected_tokens_out * float(out_per_1k) / 1000
        )
    except Exception as exc:
        preview["pricing_source"] = str(pricing_file)
        preview["unavailable_reason"] = f"historical token estimate unavailable: {exc}"
        return preview

    preview.update(
        {
            "expected_request_count": expected_request_count,
            "approximate_cost_usd": round(approximate_cost, 6),
            "pricing_source": str(pricing_file),
            "pricing_model": pricing_model,
            "pricing_basis": "in_per_1k/out_per_1k using Exp2 overall average tokens per request",
            "token_estimate_source": str(token_summary_path),
            "historical_question_count": int(token_averages["historical_question_count"]),
            "estimated_tokens_in_per_request": round(tokens_in_per_request, 3),
            "estimated_tokens_out_per_request": round(tokens_out_per_request, 3),
            "estimated_total_tokens_in": round(expected_tokens_in),
            "estimated_total_tokens_out": round(expected_tokens_out),
            "in_per_1k": float(in_per_1k),
            "out_per_1k": float(out_per_1k),
        }
    )
    preview.pop("unavailable_reason", None)
    return preview


def _csv_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_phase042_preflight_matrix(
    output_csv: Path,
    output_json: Path,
    rows: Iterable[Phase042PreflightMatrixRow | dict[str, Any]],
) -> tuple[Path, Path]:
    validated = [
        row if isinstance(row, Phase042PreflightMatrixRow) else Phase042PreflightMatrixRow(**row)
        for row in rows
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(Phase042PreflightMatrixRow.model_fields)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in validated:
            writer.writerow(
                {key: _csv_value(value) for key, value in row.model_dump(mode="json").items()}
            )
    _write_json(
        output_json,
        {
            "schema_version": PHASE042_PREFLIGHT_MATRIX_SCHEMA_VERSION,
            "rows": [row.model_dump(mode="json") for row in validated],
        },
    )
    return output_csv, output_json


def _load_phase042_preflight_matrix(path: Path) -> list[Phase042PreflightMatrixRow]:
    if not path.is_file():
        raise FileNotFoundError(f"preflight matrix does not exist: {path}")
    payload = _read_json(path)
    assert_no_phase041_reference(payload, context="Phase 04.2 preflight matrix")
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("preflight matrix must contain a rows array")
    return [Phase042PreflightMatrixRow.model_validate(row) for row in rows]


def build_phase042_static_preflight_matrix(
    *,
    selected_manifest_path: str | Path = PHASE042_SELECTED_MANIFEST_JSON,
    dataset_root: str | Path = PHASE042_EVALUATOR_SLICE,
    results_dir: str | Path = "results",
    output_root: str | Path = "results/exp5",
    run_id: str = PHASE042_STATIC_SUPPLEMENTAL_RUN_ID,
    prompts_file: str | Path = "prompts_optimized.yaml",
    prompt_mode: str = "opt",
    max_attempts: int = 1,
    resume: bool = True,
    overwrite: bool = False,
    write_reports: bool = False,
    pricing_file: str | Path | None = None,
) -> list[Phase042PreflightMatrixRow]:
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    if max_attempts != 1:
        raise ValueError("Phase 04.2 static supplemental runs require max_attempts=1")
    _assert_no_phase041_path_values(
        [selected_manifest_path, dataset_root, results_dir, output_root, run_id],
        context="Phase 04.2 static preflight",
    )

    manifest_path = Path(selected_manifest_path)
    manifest_rows = validate_phase042_selected_manifest(
        load_phase042_selected_manifest(manifest_path)
    )
    task_types = list(PHASE042_STATIC_TASK_TYPES)
    if [row.task_type for row in manifest_rows] != sorted(PHASE042_STATIC_TASK_TYPES):
        raise ValueError("selected manifest must be normalized to the three static categories")
    dataset_root = Path(dataset_root)
    _validate_materialized_dataset_root(dataset_root, task_types)

    run_matrix_rows = build_phase042_reported_run_matrix(
        results_dir=results_dir,
        materialized_dataset_root=dataset_root,
        output_root=output_root,
        run_id=run_id,
    )
    static_run_dir = revision_run_dir(output_root, run_id)
    reports_dir = static_run_dir / "preflight_reports"
    manifest_hash = sha256_file(manifest_path)

    preflight_rows: list[Phase042PreflightMatrixRow] = []
    for matrix_row in run_matrix_rows:
        effective_overwrite = overwrite or bool(matrix_row["overwrite"])
        effective_resume = False if effective_overwrite else (resume or bool(matrix_row["resume"]))
        preflight_args = argparse.Namespace(
            dataset_root=str(dataset_root),
            types=task_types,
            prompts_file=str(prompts_file),
            few_shot_config=None,
            prompt_prefix=None,
            prompt_suffix=None,
            pricing_file=str(pricing_file) if pricing_file else None,
            output_root=str(output_root),
            run_id=matrix_row["run_id"],
            provider=matrix_row["provider"],
            model=matrix_row["model"],
            prompt_mode=prompt_mode,
            max_per_type=None,
            max_attempts=max_attempts,
            overwrite=effective_overwrite,
            resume=effective_resume,
            write_report=False,
        )
        report = revision_preflight.build_report(preflight_args)
        report_payload = _preflight_report_payload(report)
        cost_preview = _phase042_token_pricing_cost_preview(
            dict(report_payload.get("cost_preview") or {}),
            pricing_file=pricing_file,
            results_dir=results_dir,
            provider=str(matrix_row["provider"]),
            model=str(matrix_row["model"]),
            expected_request_count=int(report_payload.get("expected_request_count") or 0),
        )
        report_payload["cost_preview"] = cost_preview
        report_path = reports_dir / f"{_slug(str(matrix_row['provider_model']))}.json"
        if write_reports:
            _write_json(report_path, report_payload)
        preflight_rows.append(
            Phase042PreflightMatrixRow(
                run_id=str(matrix_row["run_id"]),
                provider=str(matrix_row["provider"]),
                model=str(matrix_row["model"]),
                provider_model=str(matrix_row["provider_model"]),
                run_scope="static",
                selected_manifest_path=str(manifest_path),
                selected_manifest_sha256=manifest_hash,
                materialized_dataset_root=str(dataset_root),
                task_types=task_types,
                prompt_config=dict(report_payload.get("prompt_config") or {}),
                expected_request_count=int(report_payload.get("expected_request_count") or 0),
                cost_preview=cost_preview,
                output_dir=str(report_payload.get("output_dir") or matrix_row["output_dir"]),
                preflight_report_path=str(report_path),
                overwrite=effective_overwrite,
                resume=effective_resume,
            )
        )

    _write_phase042_preflight_matrix(
        static_run_dir / "expanded_static_preflight_matrix.csv",
        static_run_dir / "expanded_static_preflight_matrix.json",
        preflight_rows,
    )
    return preflight_rows


def _runtime_model_kwargs(provider: str, model_label: str) -> dict[str, object]:
    if provider == "openai" and model_label.startswith("gpt-5_"):
        effort = model_label.rsplit("_", 1)[1]
        if effort == "none":
            return {"model": "gpt-5", "thinking": False, "thinking_options": None}
        return {
            "model": "gpt-5",
            "thinking": True,
            "thinking_options": {"effort": effort},
        }
    if provider == "openai" and model_label.startswith("gpt-5.1_"):
        effort = model_label.rsplit("_", 1)[1]
        if effort == "none":
            return {"model": "gpt-5.1", "thinking": False, "thinking_options": None}
        return {
            "model": "gpt-5.1",
            "thinking": True,
            "thinking_options": {"effort": effort},
        }
    if provider == "fireworks" and model_label.startswith("accounts_fireworks_models_"):
        model_name = model_label.removeprefix("accounts_fireworks_models_")
        return {
            "model": f"accounts/fireworks/models/{model_name}",
            "thinking": False,
            "thinking_options": None,
        }
    if provider == "openrouter" and model_label == PHASE042_OPENROUTER_QWEN_MODEL_LABEL:
        return {
            "model": PHASE042_OPENROUTER_QWEN_RUNTIME_MODEL,
            "thinking": False,
            "thinking_options": None,
        }
    return {"model": model_label, "thinking": False, "thinking_options": None}


def _run_eval_with_supported_kwargs(kwargs: dict[str, object]) -> dict[str, Any]:
    signature = inspect.signature(run_eval.run_eval)
    supports_arbitrary_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if supports_arbitrary_kwargs:
        return run_eval.run_eval(**kwargs)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return run_eval.run_eval(**supported)


def _load_attempts(path: Path) -> list[AttemptRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"attempt log does not exist: {path}")
    attempts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                attempts.append(AttemptRecord.model_validate_json(line))
    return attempts


def _failure_bucket(attempt: AttemptRecord) -> str | None:
    if attempt.correct:
        return None
    category = (attempt.error_category or "").lower()
    if not category:
        return "scientific_wrong_count"
    if (
        category == "parse_error"
        and not attempt.parsed_answer
        and not attempt.tokens_in
        and not attempt.tokens_out
        and not attempt.cost_usd
    ):
        return "infrastructure_failure_count"
    if any(token in category for token in ("infra", "provider", "network", "timeout", "rate")):
        return "infrastructure_failure_count"
    return "protocol_failure_count"


def _write_phase042_static_summary(
    output_csv: Path,
    output_json: Path,
    rows: Iterable[Phase042StaticSummaryRow | dict[str, Any]],
) -> tuple[Path, Path]:
    validated = [
        row if isinstance(row, Phase042StaticSummaryRow) else Phase042StaticSummaryRow(**row)
        for row in rows
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(Phase042StaticSummaryRow.model_fields)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in validated:
            writer.writerow(
                {key: _csv_value(value) for key, value in row.model_dump(mode="json").items()}
            )
    _write_json(
        output_json,
        {
            "schema_version": PHASE042_STATIC_SUMMARY_SCHEMA_VERSION,
            "rows": [row.model_dump(mode="json") for row in validated],
        },
    )
    return output_csv, output_json


def _build_phase042_static_summary_rows(
    matrix_rows: list[Phase042PreflightMatrixRow],
    manifest_rows: list[Phase042SelectedManifestRow],
) -> list[Phase042StaticSummaryRow]:
    manifest_by_task = {row.task_type: row for row in manifest_rows}
    summary_rows: list[Phase042StaticSummaryRow] = []
    for matrix_row in matrix_rows:
        run_dir = Path(matrix_row.output_dir)
        attempts = _load_attempts(run_dir / "attempts.jsonl")
        counts_by_task: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "attempt_count": 0,
                "success_count": 0,
                "scientific_wrong_count": 0,
                "protocol_failure_count": 0,
                "infrastructure_failure_count": 0,
            }
        )
        for attempt in attempts:
            counts = counts_by_task[attempt.task_type]
            counts["attempt_count"] += 1
            counts["success_count"] += int(attempt.correct)
            bucket = _failure_bucket(attempt)
            if bucket:
                counts[bucket] += 1

        for task_type, counts in sorted(counts_by_task.items()):
            manifest_row = manifest_by_task.get(task_type)
            summary_rows.append(
                Phase042StaticSummaryRow(
                    run_id=matrix_row.run_id,
                    provider=matrix_row.provider,
                    model=matrix_row.model,
                    provider_model=matrix_row.provider_model,
                    task_type=task_type,
                    task_family=manifest_row.task_family if manifest_row else "unknown",
                    evidence_origin=(
                        manifest_row.evidence_origin if manifest_row else "new_category"
                    ),
                    sample_count=manifest_row.sample_count if manifest_row else 0,
                    attempt_count=counts["attempt_count"],
                    success_count=counts["success_count"],
                    scientific_wrong_count=counts["scientific_wrong_count"],
                    protocol_failure_count=counts["protocol_failure_count"],
                    infrastructure_failure_count=counts["infrastructure_failure_count"],
                    pass_rate=(
                        counts["success_count"] / counts["attempt_count"]
                        if counts["attempt_count"]
                        else None
                    ),
                    run_manifest_path=str(run_dir / "run_manifest.json"),
                    attempt_log_path=str(run_dir / "attempts.jsonl"),
                    summary_source_path=str(run_dir / "summary.json"),
                    selected_manifest_path=matrix_row.selected_manifest_path,
                    claim_use="main_body_direct_evidence",
                )
            )
    return summary_rows


def collect_phase042_static_runs(
    *,
    preflight_matrix_path: str | Path,
    output_root: str | Path = "results/exp5",
    selected_manifest_path: str | Path | None = None,
    resume: bool = True,
    overwrite: bool = False,
    secrets_file: str | Path = "secrets.yaml",
    stream: bool = True,
    timeout_sec: float = 600.0,
    seed: int = 1234,
) -> dict[str, object]:
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    _assert_no_phase041_path_values(
        [preflight_matrix_path, output_root, selected_manifest_path or "", secrets_file],
        context="Phase 04.2 static collection",
    )
    matrix_rows = _load_phase042_preflight_matrix(Path(preflight_matrix_path))
    if not matrix_rows:
        raise ValueError("preflight matrix must contain at least one row")
    if any(row.run_scope != "static" for row in matrix_rows):
        raise ValueError("collect-static requires static preflight matrix rows")

    manifest_path = Path(selected_manifest_path or matrix_rows[0].selected_manifest_path)
    manifest_rows = validate_phase042_selected_manifest(
        load_phase042_selected_manifest(manifest_path)
    )

    run_results: list[dict[str, Any]] = []
    for matrix_row in matrix_rows:
        runtime_model = _runtime_model_kwargs(matrix_row.provider, matrix_row.model)
        prompts_file = matrix_row.prompt_config.get("prompts_file") or "prompts_optimized.yaml"
        run_kwargs: dict[str, object] = {
            "dataset_root": matrix_row.materialized_dataset_root,
            "types": list(matrix_row.task_types),
            "provider": matrix_row.provider,
            "model": runtime_model["model"],
            "max_per_type": None,
            "out_csv": str(Path(matrix_row.output_dir) / "results.csv"),
            "secrets_file": str(secrets_file),
            "stream": stream,
            "timeout_sec": timeout_sec,
            "seed": seed,
            "prompts_file": str(prompts_file),
            "prompt_mode": "opt",
            "revision_run_id": matrix_row.run_id,
            "revision_output_root": str(output_root),
            "write_attempts": True,
            "overwrite_revision_output": overwrite,
            "resume_revision_output": resume,
            "max_attempts": 1,
            "thinking": runtime_model["thinking"],
            "thinking_options": runtime_model["thinking_options"],
        }
        run_results.append(_run_eval_with_supported_kwargs(run_kwargs))

    summary_rows = _build_phase042_static_summary_rows(matrix_rows, manifest_rows)
    summary_dir = Path(preflight_matrix_path).parent
    summary_csv = summary_dir / "expanded_static_summary.csv"
    summary_json = summary_dir / "expanded_static_summary.json"
    _write_phase042_static_summary(summary_csv, summary_json, summary_rows)
    return {
        "run_count": len(matrix_rows),
        "run_result_count": len(run_results),
        "static_summary_row_count": len(summary_rows),
        "static_summary_csv": str(summary_csv),
        "static_summary_json": str(summary_json),
    }


def _adaptive_preflight_report_payload(report: object) -> dict[str, Any]:
    if hasattr(report, "model_dump"):
        return report.model_dump(mode="json")  # type: ignore[no-any-return]
    if isinstance(report, dict):
        return dict(report)
    raise TypeError("adaptive preflight report must be a pydantic model or dict")


def _round_run_id(run_id: str, round_index: int, provider_model: str) -> str:
    return f"{run_id}-round{round_index:02d}-{_slug(provider_model)}"


def _round_seed(seed: int, round_index: int, provider_index: int) -> int:
    return seed + (round_index - 1) * 1000 + provider_index


def build_phase042_adaptive_preflight_matrix(
    *,
    selected_manifest_path: str | Path = PHASE042_SELECTED_MANIFEST_JSON,
    dataset_root: str | Path = PHASE042_ADAPTIVE_EVALUATOR_SLICE,
    results_dir: str | Path = "results",
    output_root: str | Path = "results/exp5",
    run_id: str = PHASE042_ADAPTIVE_SUPPLEMENTAL_RUN_ID,
    prompts_file: str | Path = "prompts_optimized.yaml",
    prompt_mode: str = "opt",
    attempt_budget_k: int = PHASE042_ADAPTIVE_ATTEMPT_BUDGET_K,
    intermediate_budget_k: int = PHASE042_ADAPTIVE_INTERMEDIATE_BUDGET_K,
    round_count: int = PHASE042_ADAPTIVE_ROUND_COUNT,
    seed: int = 1234,
    resume: bool = True,
    overwrite: bool = False,
    write_reports: bool = False,
    pricing_file: str | Path | None = None,
) -> list[Phase042PreflightMatrixRow]:
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    if attempt_budget_k != PHASE042_ADAPTIVE_ATTEMPT_BUDGET_K:
        raise ValueError("Phase 04.2 adaptive supplemental runs require attempt_budget_k=5")
    if intermediate_budget_k != PHASE042_ADAPTIVE_INTERMEDIATE_BUDGET_K:
        raise ValueError("Phase 04.2 adaptive summaries require intermediate_budget_k=3")
    if round_count != PHASE042_ADAPTIVE_ROUND_COUNT:
        raise ValueError("Phase 04.2 adaptive supplemental runs require round_count=5")
    _assert_no_phase041_path_values(
        [selected_manifest_path, dataset_root, results_dir, output_root, run_id],
        context="Phase 04.2 adaptive preflight",
    )

    manifest_path = Path(selected_manifest_path)
    validate_phase042_selected_manifest(load_phase042_selected_manifest(manifest_path))
    task_types = list(PHASE042_ADAPTIVE_TASK_TYPES)
    dataset_root = Path(dataset_root)
    _validate_materialized_dataset_root(dataset_root, task_types)

    provider_rows = build_phase042_reported_run_matrix(
        results_dir=results_dir,
        materialized_dataset_root=dataset_root,
        output_root=output_root,
        run_id=run_id,
    )
    adaptive_run_dir = revision_run_dir(output_root, run_id)
    reports_dir = adaptive_run_dir / "adaptive_preflight_reports"
    manifest_hash = sha256_file(manifest_path)

    preflight_rows: list[Phase042PreflightMatrixRow] = []
    for provider_index, matrix_row in enumerate(provider_rows):
        provider_model = str(matrix_row["provider_model"])
        for round_index in range(1, round_count + 1):
            row_run_id = _round_run_id(run_id, round_index, provider_model)
            row_seed = _round_seed(seed, round_index, provider_index)
            effective_overwrite = overwrite or bool(matrix_row["overwrite"])
            effective_resume = False if effective_overwrite else (
                resume or bool(matrix_row["resume"])
            )
            preflight_args = argparse.Namespace(
                dataset_root=str(dataset_root),
                types=task_types,
                prompts_file=str(prompts_file),
                few_shot_config=None,
                prompt_prefix=None,
                prompt_suffix=None,
                pricing_file=str(pricing_file) if pricing_file else None,
                output_root=str(output_root),
                run_id=row_run_id,
                provider=matrix_row["provider"],
                model=matrix_row["model"],
                prompt_mode=prompt_mode,
                max_per_type=None,
                attempt_budget_k=attempt_budget_k,
                sampling_mode=SAMPLING_MODE,
                feedback_mode=FEEDBACK_MODE,
                memory_mode=MEMORY_MODE,
                stopping_rule=STOPPING_RULE,
                overwrite=effective_overwrite,
                resume=effective_resume,
                write_report=False,
            )
            report = adaptive_preflight.build_report(preflight_args)
            report_payload = _adaptive_preflight_report_payload(report)
            cost_preview = _phase042_token_pricing_cost_preview(
                dict(report_payload.get("cost_preview") or {}),
                pricing_file=pricing_file,
                results_dir=results_dir,
                provider=str(matrix_row["provider"]),
                model=str(matrix_row["model"]),
                expected_request_count=int(
                    report_payload.get("expected_request_count_max")
                    or report_payload.get("expected_request_count")
                    or 0
                ),
            )
            report_payload["cost_preview"] = cost_preview
            report_path = reports_dir / f"{_slug(provider_model)}-round{round_index:02d}.json"
            if write_reports:
                _write_json(report_path, report_payload)
            preflight_rows.append(
                Phase042PreflightMatrixRow(
                    run_id=row_run_id,
                    provider=str(matrix_row["provider"]),
                    model=str(matrix_row["model"]),
                    provider_model=provider_model,
                    run_scope="adaptive",
                    selected_manifest_path=str(manifest_path),
                    selected_manifest_sha256=manifest_hash,
                    materialized_dataset_root=str(dataset_root),
                    task_types=task_types,
                    prompt_config=dict(report_payload.get("prompt_config") or {}),
                    expected_request_count=int(
                        report_payload.get("expected_request_count_max") or 0
                    ),
                    cost_preview=cost_preview,
                    output_dir=str(
                        report_payload.get("output_dir")
                        or revision_run_dir(output_root, row_run_id)
                    ),
                    preflight_report_path=str(report_path),
                    overwrite=effective_overwrite,
                    resume=effective_resume,
                    attempt_budget_k=attempt_budget_k,
                    sampling_mode=str(report_payload.get("sampling_mode") or SAMPLING_MODE),
                    feedback_mode=str(report_payload.get("feedback_mode") or FEEDBACK_MODE),
                    memory_mode=str(report_payload.get("memory_mode") or MEMORY_MODE),
                    stopping_rule=str(report_payload.get("stopping_rule") or STOPPING_RULE),
                    solve_request_count=int(report_payload.get("solve_request_count") or 0),
                    reflection_request_count_max=int(
                        report_payload.get("reflection_request_count_max") or 0
                    ),
                    expected_request_count_max=int(
                        report_payload.get("expected_request_count_max") or 0
                    ),
                    round_id=f"round{round_index:02d}",
                    round_index=round_index,
                    round_count=round_count,
                    seed=row_seed,
                    intermediate_budget_k=intermediate_budget_k,
                    adaptive_scope_rationale=PHASE042_ADAPTIVE_SCOPE_RATIONALE,
                )
            )

    _write_phase042_preflight_matrix(
        adaptive_run_dir / "expanded_adaptive_preflight_matrix.csv",
        adaptive_run_dir / "expanded_adaptive_preflight_matrix.json",
        preflight_rows,
    )
    return preflight_rows


def _load_adaptive_summary_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"adaptive summary does not exist: {path}")
    payload = _read_json(path)
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"adaptive summary must contain rows: {path}")
    return [row for row in rows if isinstance(row, dict)]


def _load_adaptive_attempt_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"adaptive attempts do not exist: {path}")
    attempts: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    attempts.append(payload)
    return attempts


def derive_phase042_adaptive_success_at_k(
    attempts: Iterable[dict[str, Any]],
    *,
    task_type: str,
    k: int,
) -> dict[str, object]:
    task_attempts = [
        attempt
        for attempt in attempts
        if attempt.get("task_type") == task_type and int(attempt.get("attempt_index") or 0) <= k
    ]
    success_attempts = [
        int(attempt.get("attempt_index") or 0)
        for attempt in task_attempts
        if bool(attempt.get("correct"))
    ]
    attempts_to_success = min(success_attempts) if success_attempts else None
    return {
        "success": attempts_to_success is not None,
        "attempts_to_success": attempts_to_success,
    }


def _phase042_task_sample_count(
    dataset_root: Path,
    manifest_by_task: dict[str, Phase042SelectedManifestRow],
    task_type: str,
) -> int:
    manifest_row = manifest_by_task.get(task_type)
    if manifest_row:
        return manifest_row.sample_count
    ground_truth_path = dataset_root / task_type / "ground_truth.json"
    if ground_truth_path.is_file():
        ground_truth = _read_json(ground_truth_path)
        if isinstance(ground_truth, dict | list):
            return len(ground_truth)
    return 0


def _build_phase042_adaptive_summary_rows(
    matrix_rows: list[Phase042PreflightMatrixRow],
    manifest_rows: list[Phase042SelectedManifestRow],
) -> list[Phase042AdaptiveSummaryRow]:
    manifest_by_task = {row.task_type: row for row in manifest_rows}
    expanded_rows: list[Phase042AdaptiveSummaryRow] = []
    for matrix_row in matrix_rows:
        run_dir = Path(matrix_row.output_dir)
        attempts = _load_adaptive_attempt_rows(run_dir / "adaptive_attempts.jsonl")
        for row in _load_adaptive_summary_rows(run_dir / "adaptive_summary.json"):
            task_type = str(row.get("task_type") or "")
            manifest_row = manifest_by_task.get(task_type)
            success_at_3 = derive_phase042_adaptive_success_at_k(
                attempts,
                task_type=task_type,
                k=matrix_row.intermediate_budget_k or PHASE042_ADAPTIVE_INTERMEDIATE_BUDGET_K,
            )
            success_at_5 = derive_phase042_adaptive_success_at_k(
                attempts,
                task_type=task_type,
                k=matrix_row.attempt_budget_k or PHASE042_ADAPTIVE_ATTEMPT_BUDGET_K,
            )
            expanded_rows.append(
                Phase042AdaptiveSummaryRow(
                    run_id=matrix_row.run_id,
                    provider=matrix_row.provider,
                    model=matrix_row.model,
                    provider_model=matrix_row.provider_model,
                    task_type=task_type,
                    task_family=(
                        manifest_row.task_family
                        if manifest_row
                        else PHASE042_TASK_FAMILIES.get(task_type, "unknown")
                    ),
                    evidence_origin=(
                        manifest_row.evidence_origin
                        if manifest_row
                        else "supplemented_category"
                    ),
                    sample_count=_phase042_task_sample_count(
                        Path(matrix_row.materialized_dataset_root),
                        manifest_by_task,
                        task_type,
                    ),
                    session_count=1,
                    round_id=matrix_row.round_id,
                    round_index=matrix_row.round_index,
                    round_count=matrix_row.round_count,
                    attempt_budget_k=int(row.get("attempt_budget_k") or 0),
                    intermediate_budget_k=matrix_row.intermediate_budget_k,
                    success_count=int(row.get("n_success") or 0),
                    success_at_3=bool(success_at_3["success"]),
                    success_at_5=bool(success_at_5["success"]),
                    attempts_to_success_at_3=success_at_3["attempts_to_success"],
                    attempts_to_success_at_5=success_at_5["attempts_to_success"],
                    scientific_wrong_count=int(row.get("scientific_wrong_count") or 0),
                    protocol_failure_count=int(row.get("protocol_failure_count") or 0),
                    infrastructure_failure_count=int(
                        row.get("infrastructure_failure_count") or 0
                    ),
                    adaptive_success_rate=float(row.get("success_rate") or 0.0),
                    feedback_mode=str(row.get("feedback_mode") or FEEDBACK_MODE),
                    memory_mode=str(row.get("memory_mode") or MEMORY_MODE),
                    stopping_rule=STOPPING_RULE,
                    run_manifest_path=str(run_dir / "run_manifest.json"),
                    adaptive_attempt_log_path=str(run_dir / "adaptive_attempts.jsonl"),
                    adaptive_summary_source_path=str(run_dir / "adaptive_summary.json"),
                    selected_manifest_path=matrix_row.selected_manifest_path,
                    claim_use="main_body_caveated",
                )
            )
    return expanded_rows


def _write_phase042_adaptive_summary(
    output_csv: Path,
    output_json: Path,
    rows: Iterable[Phase042AdaptiveSummaryRow | dict[str, Any]],
) -> tuple[Path, Path]:
    validated = [
        row if isinstance(row, Phase042AdaptiveSummaryRow) else Phase042AdaptiveSummaryRow(**row)
        for row in rows
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(Phase042AdaptiveSummaryRow.model_fields)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in validated:
            writer.writerow(
                {key: _csv_value(value) for key, value in row.model_dump(mode="json").items()}
            )
    _write_json(
        output_json,
        {
            "schema_version": PHASE042_ADAPTIVE_SUMMARY_SCHEMA_VERSION,
            "rows": [row.model_dump(mode="json") for row in validated],
        },
    )
    return output_csv, output_json


def collect_phase042_adaptive_runs(
    *,
    preflight_matrix_path: str | Path,
    output_root: str | Path = "results/local_runs",
    selected_manifest_path: str | Path | None = None,
    confirmed_adaptive_cost: bool = False,
    resume: bool = True,
    overwrite: bool = False,
    secrets_file: str | Path = "secrets.yaml",
    stream: bool = False,
    timeout_sec: float = 600.0,
) -> dict[str, object]:
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    _assert_no_phase041_path_values(
        [preflight_matrix_path, output_root, selected_manifest_path or "", secrets_file],
        context="Phase 04.2 adaptive collection",
    )
    matrix_rows = _load_phase042_preflight_matrix(Path(preflight_matrix_path))
    if not matrix_rows:
        raise ValueError("preflight matrix must contain at least one row")
    if any(row.run_scope != "adaptive" for row in matrix_rows):
        raise ValueError("collect-adaptive requires adaptive preflight matrix rows")
    if not confirmed_adaptive_cost:
        raise ValueError("collect-adaptive requires --confirmed-adaptive-cost")
    for row in matrix_rows:
        if row.attempt_budget_k != PHASE042_ADAPTIVE_ATTEMPT_BUDGET_K:
            raise ValueError(f"adaptive row must have attempt_budget_k=5: {row.run_id}")
        if row.intermediate_budget_k != PHASE042_ADAPTIVE_INTERMEDIATE_BUDGET_K:
            raise ValueError(f"adaptive row must have intermediate_budget_k=3: {row.run_id}")
        if row.round_count != PHASE042_ADAPTIVE_ROUND_COUNT:
            raise ValueError(f"adaptive row must have round_count=5: {row.run_id}")

    manifest_path = Path(selected_manifest_path or matrix_rows[0].selected_manifest_path)
    manifest_rows = validate_phase042_selected_manifest(
        load_phase042_selected_manifest(manifest_path)
    )

    run_results: list[dict[str, Any]] = []
    for matrix_row in matrix_rows:
        runtime_model = _runtime_model_kwargs(matrix_row.provider, matrix_row.model)
        prompts_file = matrix_row.prompt_config.get("prompts_file") or "prompts_optimized.yaml"
        run_results.append(
            adaptive_attacker.run_adaptive_experiment(
                dataset_root=matrix_row.materialized_dataset_root,
                types=list(matrix_row.task_types),
                provider=matrix_row.provider,
                model=str(runtime_model["model"]),
                run_id=matrix_row.run_id,
                output_root=str(output_root),
                attempt_budget_k=PHASE042_ADAPTIVE_ATTEMPT_BUDGET_K,
                max_per_type=None,
                prompts_file=str(prompts_file),
                prompt_mode="opt",
                secrets_file=str(secrets_file),
                timeout_sec=timeout_sec,
                seed=int(matrix_row.seed or 1234),
                stream=stream,
                overwrite=overwrite,
                resume=resume,
                thinking=bool(runtime_model["thinking"]),
                thinking_options=runtime_model["thinking_options"],
            )
        )

    summary_rows = _build_phase042_adaptive_summary_rows(matrix_rows, manifest_rows)
    summary_dir = Path(preflight_matrix_path).parent
    summary_csv = summary_dir / "expanded_adaptive_summary.csv"
    summary_json = summary_dir / "expanded_adaptive_summary.json"
    _write_phase042_adaptive_summary(summary_csv, summary_json, summary_rows)
    return {
        "run_count": len(matrix_rows),
        "run_result_count": len(run_results),
        "adaptive_summary_row_count": len(summary_rows),
        "adaptive_summary_csv": str(summary_csv),
        "adaptive_summary_json": str(summary_json),
    }


def _payload_rows(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    payload = _read_json(path)
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{label} must contain a rows array: {path}")
    assert_no_phase041_reference(rows, context=label)
    return [dict(row) for row in rows if isinstance(row, dict)]


def _load_phase042_static_summary_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_like in paths:
        path = Path(path_like)
        assert_no_phase041_reference(path.as_posix(), context="static summary path")
        for row in _payload_rows(path, label="Phase 04.2 static summary"):
            enriched = dict(row)
            enriched["_source_artifact_path"] = path.as_posix()
            rows.append(enriched)
    return rows


def _load_phase042_adaptive_summary_rows_for_analysis(path: str | Path) -> list[dict[str, Any]]:
    summary_path = Path(path)
    assert_no_phase041_reference(summary_path.as_posix(), context="adaptive summary path")
    return [
        {**row, "_source_artifact_path": summary_path.as_posix()}
        for row in _payload_rows(summary_path, label="Phase 04.2 adaptive summary")
    ]


def _load_phase042_original_exp2_rates(results_dir: str | Path) -> dict[tuple[str, str], float]:
    root = Path(results_dir)
    exp2_root = root / "exp2" if (root / "exp2").is_dir() else root
    rates: dict[tuple[str, str], float] = {}
    for results_path in sorted(exp2_root.glob("*/*/results.csv")):
        provider = results_path.parent.parent.name
        model = results_path.parent.name
        provider_model = f"{provider}/{model}"
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                task_type = row.get("type") or row.get("task_type")
                raw_rate = row.get("pass_at_1") or row.get("pass_rate")
                if not task_type or raw_rate in (None, ""):
                    continue
                rates[(provider_model, task_type)] = float(raw_rate)

    for (provider_model, task_type), rate in list(rates.items()):
        if provider_model == PHASE042_LEGACY_FIREWORKS_QWEN_PROVIDER_MODEL:
            rates[(PHASE042_OPENROUTER_QWEN_PROVIDER_MODEL, task_type)] = rate
        if provider_model == "openai/gpt-5":
            rates[("openai/gpt-5_medium", task_type)] = rate
    return rates


def _rate_side(rate: float | None) -> str | None:
    if rate is None:
        return None
    return "high" if rate >= PHASE042_ANALYSIS_CUTOFF else "low"


def _phase042_agreement_status(
    original_rate: float | None,
    corrected_rate: float | None,
) -> tuple[str, bool, str]:
    original_side = _rate_side(original_rate)
    corrected_side = _rate_side(corrected_rate)
    if original_side is None or corrected_side is None:
        return "inconclusive", False, "Original or corrected rate is unavailable."
    if original_side != corrected_side:
        return (
            "diverges_from_original",
            True,
            "Corrected Phase 04.2 evidence crosses the 40% reporting heuristic "
            "in the opposite direction from the original Exp2 rate.",
        )
    return "supports_original", False, ""


def _bernoulli_success_at_k(original_rate: float | None, k: int) -> float | None:
    if original_rate is None:
        return None
    return 1.0 - (1.0 - float(original_rate)) ** k


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _int_value(value: object) -> int:
    return int(value or 0)


def _is_infrastructure_only(row: dict[str, Any]) -> bool:
    attempts = _int_value(row.get("attempt_count"))
    return (
        attempts > 0
        and _int_value(row.get("success_count")) == 0
        and _int_value(row.get("scientific_wrong_count")) == 0
        and _int_value(row.get("protocol_failure_count")) == 0
        and _int_value(row.get("infrastructure_failure_count")) >= attempts
    )


def _merge_infra_remediation_row(
    existing: dict[str, Any],
    remediation: dict[str, Any],
) -> dict[str, Any]:
    existing_infra = _int_value(existing.get("infrastructure_failure_count"))
    if existing_infra <= 0:
        return remediation

    existing_attempts = _int_value(existing.get("attempt_count"))
    remediation_attempts = _int_value(remediation.get("attempt_count"))
    merged = dict(existing)
    merged["attempt_count"] = max(0, existing_attempts - existing_infra) + remediation_attempts
    merged["success_count"] = _int_value(existing.get("success_count")) + _int_value(
        remediation.get("success_count")
    )
    merged["scientific_wrong_count"] = _int_value(
        existing.get("scientific_wrong_count")
    ) + _int_value(remediation.get("scientific_wrong_count"))
    merged["protocol_failure_count"] = _int_value(
        existing.get("protocol_failure_count")
    ) + _int_value(remediation.get("protocol_failure_count"))
    merged["infrastructure_failure_count"] = max(0, existing_infra - remediation_attempts)
    merged["pass_rate"] = (
        merged["success_count"] / merged["attempt_count"]
        if merged["attempt_count"]
        else None
    )
    merged["_source_artifact_path"] = ";".join(
        value
        for value in [
            str(existing.get("_source_artifact_path") or ""),
            str(remediation.get("_source_artifact_path") or ""),
        ]
        if value
    )
    return merged


def _merge_phase042_static_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        provider_model = str(row.get("provider_model") or "")
        if provider_model == PHASE042_LEGACY_FIREWORKS_QWEN_PROVIDER_MODEL:
            continue
        task_type = str(row.get("task_type") or "")
        if task_type not in PHASE042_STATIC_TASK_TYPES:
            continue
        key = (provider_model, task_type)
        if key in merged:
            merged[key] = _merge_infra_remediation_row(merged[key], row)
        else:
            merged[key] = dict(row)
    return [merged[key] for key in sorted(merged)]


def _confidence_fields(
    *,
    success_count: int,
    denominator: int,
    denominator_label: str,
    meaningful: bool = True,
) -> dict[str, object]:
    if denominator <= 0 or not meaningful:
        return {
            "ci_low": None,
            "ci_high": None,
            "ci_method": "",
            "ci_note": f"Confidence interval unavailable: {denominator_label} is not meaningful.",
        }
    ci_low, ci_high = wilson_interval(success_count, denominator)
    return {
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_method": "wilson_95",
        "ci_note": f"Wilson interval over {denominator_label}.",
    }


def _manifest_source_fields(
    manifest_by_task: dict[str, Phase042SelectedManifestRow],
    task_type: str,
) -> dict[str, object]:
    manifest_row = manifest_by_task.get(task_type)
    if manifest_row is None:
        return {
            "source_kind": "original_captchaworld_hard_scope",
            "source_provenance_class": "original_hard_task_context",
            "source_citation": "",
            "source_license": "",
            "source_provenance_notes": (
                "Original hard-task adaptive scope; not corrected direct expanded "
                "static sidecar evidence."
            ),
            "provenance_caveat": (
                "Original hard-task adaptive context, reported separately from "
                "corrected three-category expanded static evidence."
            ),
            "real_external_evidence": False,
        }
    is_gpt = manifest_row.source_kind == PHASE042_GPT_IMAGE_SOURCE_KIND
    return {
        "source_kind": manifest_row.source_kind,
        "source_provenance_class": manifest_row.source_provenance_class,
        "source_citation": manifest_row.source_citation,
        "source_license": manifest_row.source_license,
        "source_provenance_notes": manifest_row.source_provenance_notes,
        "provenance_caveat": (
            "GPT Image fallback evidence is generated Open CaptchaWorld-style "
            "material and must not be described as real external dataset evidence."
            if is_gpt
            else ""
        ),
        "real_external_evidence": manifest_row.source_kind in PHASE042_REAL_EXTERNAL_SOURCE_KINDS,
    }


def _claim_effect(
    *,
    observed_rate: float | None,
    scientific_wrong_count: int,
    infrastructure_only: bool,
    source_kind: str,
) -> str:
    if observed_rate is None or infrastructure_only:
        return "neutral_or_inconclusive"
    if source_kind == PHASE042_GPT_IMAGE_SOURCE_KIND:
        return "neutral_or_inconclusive"
    if observed_rate >= PHASE042_ANALYSIS_CUTOFF:
        return "weakens_structural_hardness"
    if scientific_wrong_count > 0:
        return "supports_structural_hardness"
    return "neutral_or_inconclusive"


def _claim_boundary_note(
    *,
    evidence_mode: str,
    claim_effect: str,
    source_kind: str,
    infrastructure_failure_count: int,
) -> str:
    base = (
        "Corrected Phase 04.2 evidence uses the 40% reporting heuristic only "
        "for claim-boundary reporting, not as a universal CAPTCHA security boundary."
    )
    if source_kind == PHASE042_GPT_IMAGE_SOURCE_KIND:
        return (
            f"{base} GPT Image fallback rows remain provenance-caveated and are "
            "not real external dataset evidence."
        )
    if infrastructure_failure_count:
        return (
            f"{base} Infrastructure failures are visible and are not counted as "
            "structural-hardness evidence."
        )
    if evidence_mode == "adaptive":
        return (
            f"{base} Adaptive rows are five memory-isolated hard-scope rounds, "
            "not precise probability estimates."
        )
    if claim_effect == "weakens_structural_hardness":
        return f"{base} This row weakens or constrains the structural-hardness claim."
    if claim_effect == "supports_structural_hardness":
        return f"{base} This row supports the structural-hardness claim within scope."
    return f"{base} This row is neutral or inconclusive for structural-hardness claims."


def derive_phase042_adaptive_round_rates(
    adaptive_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in adaptive_rows:
        task_type = str(row.get("task_type") or "")
        if task_type not in PHASE042_ADAPTIVE_TASK_TYPES:
            raise ValueError(f"adaptive evidence task is out of Phase 04.2 scope: {task_type}")
        grouped[(str(row.get("provider_model") or ""), task_type)].append(dict(row))

    derived: list[dict[str, Any]] = []
    for (provider_model, task_type), rows in sorted(grouped.items()):
        round_ids = {
            str(row.get("round_id") or row.get("round_index") or index)
            for index, row in enumerate(rows, start=1)
        }
        round_count = len(round_ids)
        success_at_3_count = sum(1 for row in rows if bool(row.get("success_at_3")))
        success_at_5_values = [row.get("success_at_5") for row in rows]
        success_at_5_count = (
            sum(1 for value in success_at_5_values if bool(value))
            if any(value is not None for value in success_at_5_values)
            else None
        )
        first = rows[0]
        derived.append(
            {
                "provider": str(first.get("provider") or ""),
                "model": str(first.get("model") or ""),
                "provider_model": provider_model,
                "task_type": task_type,
                "task_family": str(
                    first.get("task_family") or PHASE042_TASK_FAMILIES.get(task_type, "unknown")
                ),
                "evidence_origin": str(first.get("evidence_origin") or "supplemented_category"),
                "sample_count": _int_value(first.get("sample_count")),
                "attempt_count": sum(
                    min(
                        _int_value(row.get("attempt_budget_k")),
                        PHASE042_ADAPTIVE_INTERMEDIATE_BUDGET_K,
                    )
                    for row in rows
                ),
                "success_count": success_at_3_count,
                "adaptive_success_at_3": (
                    success_at_3_count / round_count if round_count else None
                ),
                "adaptive_success_at_5": (
                    success_at_5_count / round_count
                    if round_count and success_at_5_count is not None
                    else None
                ),
                "adaptive_round_count": round_count,
                "memory_isolation": (
                    PHASE042_MEMORY_ISOLATION_LABEL
                    if round_count == PHASE042_ADAPTIVE_ROUND_COUNT
                    else f"{round_count}_memory_isolated_rounds"
                ),
                "scientific_wrong_count": sum(
                    _int_value(row.get("scientific_wrong_count")) for row in rows
                ),
                "protocol_failure_count": sum(
                    _int_value(row.get("protocol_failure_count")) for row in rows
                ),
                "infrastructure_failure_count": sum(
                    _int_value(row.get("infrastructure_failure_count")) for row in rows
                ),
                "selected_manifest_path": str(first.get("selected_manifest_path") or ""),
                "source_artifact_path": ";".join(
                    sorted(
                        {
                            str(row.get("_source_artifact_path") or "")
                            for row in rows
                            if row.get("_source_artifact_path")
                        }
                    )
                ),
            }
        )
    return derived


def _phase042_static_analysis_row(
    row: dict[str, Any],
    *,
    manifest_by_task: dict[str, Phase042SelectedManifestRow],
    original_rates: dict[tuple[str, str], float],
    run_id: str,
) -> Phase042EvidenceAnalysisRow:
    provider_model = str(row.get("provider_model") or "")
    task_type = str(row.get("task_type") or "")
    provider = str(row.get("provider") or provider_model.split("/", 1)[0])
    model = str(row.get("model") or provider_model.split("/", 1)[-1])
    original_rate = original_rates.get((provider_model, task_type))
    corrected_rate = _float_or_none(row.get("pass_rate"))
    agreement_status, diverges, divergence_reason = _phase042_agreement_status(
        original_rate,
        corrected_rate,
    )
    attempts = _int_value(row.get("attempt_count"))
    successes = _int_value(row.get("success_count"))
    scientific_wrong_count = _int_value(row.get("scientific_wrong_count"))
    infrastructure_only = _is_infrastructure_only(row)
    source_fields = _manifest_source_fields(manifest_by_task, task_type)
    claim_effect = _claim_effect(
        observed_rate=corrected_rate,
        scientific_wrong_count=scientific_wrong_count,
        infrastructure_only=infrastructure_only,
        source_kind=str(source_fields["source_kind"]),
    )
    ci = _confidence_fields(
        success_count=successes,
        denominator=attempts,
        denominator_label="corrected static attempts",
        meaningful=not infrastructure_only,
    )
    return Phase042EvidenceAnalysisRow(
        analysis_id=f"{run_id}-{_slug(provider_model)}-{_slug(task_type)}-static",
        run_id=run_id,
        evidence_mode="static",
        provider=provider,
        model=model,
        provider_model=provider_model,
        task_type=task_type,
        task_family=str(row.get("task_family") or PHASE042_TASK_FAMILIES.get(task_type, "")),
        evidence_origin=str(row.get("evidence_origin") or "new_category"),
        sample_count=_int_value(row.get("sample_count")),
        attempt_count=attempts,
        success_count=successes,
        original_rate=original_rate,
        corrected_static_rate=corrected_rate,
        corrected_adaptive_rate=None,
        adaptive_success_at_3=None,
        adaptive_success_at_5=None,
        bernoulli_success_at_3=_bernoulli_success_at_k(original_rate, 3),
        bernoulli_success_at_5=_bernoulli_success_at_k(original_rate, 5),
        adaptive_round_count=0,
        memory_isolation="",
        agreement_status=agreement_status,
        diverges_from_original=diverges,
        divergence_reason=divergence_reason,
        scientific_wrong_count=scientific_wrong_count,
        protocol_failure_count=_int_value(row.get("protocol_failure_count")),
        infrastructure_failure_count=_int_value(row.get("infrastructure_failure_count")),
        selected_manifest_path=str(row.get("selected_manifest_path") or ""),
        claim_boundary_note=_claim_boundary_note(
            evidence_mode="static",
            claim_effect=claim_effect,
            source_kind=str(source_fields["source_kind"]),
            infrastructure_failure_count=_int_value(row.get("infrastructure_failure_count")),
        ),
        claim_effect=claim_effect,
        adaptive_scope_rationale="",
        source_artifact_path=str(row.get("_source_artifact_path") or ""),
        direct_evidence=True,
        **source_fields,
        **ci,
    )


def _phase042_adaptive_analysis_row(
    row: dict[str, Any],
    *,
    manifest_by_task: dict[str, Phase042SelectedManifestRow],
    original_rates: dict[tuple[str, str], float],
    run_id: str,
) -> Phase042EvidenceAnalysisRow:
    provider_model = str(row.get("provider_model") or "")
    task_type = str(row.get("task_type") or "")
    original_rate = original_rates.get((provider_model, task_type))
    adaptive_success_at_3 = _float_or_none(row.get("adaptive_success_at_3"))
    adaptive_success_at_5 = _float_or_none(row.get("adaptive_success_at_5"))
    agreement_status, diverges, divergence_reason = _phase042_agreement_status(
        original_rate,
        adaptive_success_at_3,
    )
    source_fields = _manifest_source_fields(manifest_by_task, task_type)
    scientific_wrong_count = _int_value(row.get("scientific_wrong_count"))
    claim_effect = _claim_effect(
        observed_rate=adaptive_success_at_3,
        scientific_wrong_count=scientific_wrong_count,
        infrastructure_only=False,
        source_kind=str(source_fields["source_kind"]),
    )
    round_count = _int_value(row.get("adaptive_round_count"))
    success_count = _int_value(row.get("success_count"))
    ci = _confidence_fields(
        success_count=success_count,
        denominator=round_count,
        denominator_label="five memory-isolated adaptive rounds",
        meaningful=round_count > 0,
    )
    ci["ci_note"] = (
        f"{ci['ci_note']} Treat as an observed round-level interval, not a "
        "precise single-session probability estimate."
    )
    return Phase042EvidenceAnalysisRow(
        analysis_id=f"{run_id}-{_slug(provider_model)}-{_slug(task_type)}-adaptive",
        run_id=run_id,
        evidence_mode="adaptive",
        provider=str(row.get("provider") or ""),
        model=str(row.get("model") or ""),
        provider_model=provider_model,
        task_type=task_type,
        task_family=str(row.get("task_family") or PHASE042_TASK_FAMILIES.get(task_type, "")),
        evidence_origin=str(row.get("evidence_origin") or "supplemented_category"),
        sample_count=_int_value(row.get("sample_count")),
        attempt_count=_int_value(row.get("attempt_count")),
        success_count=success_count,
        original_rate=original_rate,
        corrected_static_rate=None,
        corrected_adaptive_rate=adaptive_success_at_3,
        adaptive_success_at_3=adaptive_success_at_3,
        adaptive_success_at_5=adaptive_success_at_5,
        bernoulli_success_at_3=_bernoulli_success_at_k(original_rate, 3),
        bernoulli_success_at_5=_bernoulli_success_at_k(original_rate, 5),
        adaptive_round_count=round_count,
        memory_isolation=str(row.get("memory_isolation") or ""),
        agreement_status=agreement_status,
        diverges_from_original=diverges,
        divergence_reason=divergence_reason,
        scientific_wrong_count=scientific_wrong_count,
        protocol_failure_count=_int_value(row.get("protocol_failure_count")),
        infrastructure_failure_count=_int_value(row.get("infrastructure_failure_count")),
        selected_manifest_path=str(row.get("selected_manifest_path") or ""),
        claim_boundary_note=_claim_boundary_note(
            evidence_mode="adaptive",
            claim_effect=claim_effect,
            source_kind=str(source_fields["source_kind"]),
            infrastructure_failure_count=_int_value(row.get("infrastructure_failure_count")),
        ),
        claim_effect=claim_effect,
        adaptive_scope_rationale=PHASE042_ADAPTIVE_SCOPE_RATIONALE,
        source_artifact_path=str(row.get("source_artifact_path") or ""),
        direct_evidence=True,
        **source_fields,
        **ci,
    )


def build_phase042_evidence_analysis_rows(
    *,
    selected_manifest_path: str | Path = PHASE042_SELECTED_MANIFEST_JSON,
    source_download_manifest_path: str | Path = (
        PHASE042_SIDECAR_ROOT / "source_download_manifest.json"
    ),
    static_summary_paths: Iterable[str | Path] = PHASE042_FINAL_STATIC_SUMMARY_PATHS,
    adaptive_summary_path: str | Path = PHASE042_FINAL_ADAPTIVE_SUMMARY_PATH,
    results_dir: str | Path = "results",
    run_id: str = PHASE042_EVIDENCE_ANALYSIS_RUN_ID,
) -> list[Phase042EvidenceAnalysisRow]:
    assert_no_phase041_reference(run_id, context="evidence analysis run_id")
    selected_manifest = Path(selected_manifest_path)
    source_download_manifest = Path(source_download_manifest_path)
    assert_no_phase041_reference(
        selected_manifest.as_posix(),
        context="evidence analysis selected manifest path",
    )
    assert_no_phase041_reference(
        source_download_manifest.as_posix(),
        context="evidence analysis source download manifest path",
    )
    manifest_rows = validate_phase042_selected_manifest(
        load_phase042_selected_manifest(selected_manifest)
    )
    manifest_by_task = {row.task_type: row for row in manifest_rows}
    static_rows = _merge_phase042_static_rows(
        _load_phase042_static_summary_rows(static_summary_paths)
    )
    adaptive_rows = derive_phase042_adaptive_round_rates(
        _load_phase042_adaptive_summary_rows_for_analysis(adaptive_summary_path)
    )
    original_rates = _load_phase042_original_exp2_rates(results_dir)
    for task_type, reference_rate in PHASE042_ADAPTIVE_EXTERNAL_REFERENCE_RATES.items():
        original_rates[("openai/gpt-5_medium", task_type)] = reference_rate

    analysis_rows = [
        _phase042_static_analysis_row(
            row,
            manifest_by_task=manifest_by_task,
            original_rates=original_rates,
            run_id=run_id,
        )
        for row in static_rows
    ]
    analysis_rows.extend(
        _phase042_adaptive_analysis_row(
            row,
            manifest_by_task=manifest_by_task,
            original_rates=original_rates,
            run_id=run_id,
        )
        for row in adaptive_rows
    )
    assert_no_phase041_reference(
        [row.model_dump(mode="json") for row in analysis_rows],
        context="evidence analysis rows",
    )
    return analysis_rows


def render_phase042_divergence_report(
    rows: Iterable[Phase042EvidenceAnalysisRow | dict[str, Any]],
    *,
    source_download_manifest_path: str | Path = (
        PHASE042_SIDECAR_ROOT / "source_download_manifest.json"
    ),
) -> str:
    validated = [
        row if isinstance(row, Phase042EvidenceAnalysisRow) else Phase042EvidenceAnalysisRow(**row)
        for row in rows
    ]
    source_payload = _read_json(Path(source_download_manifest_path))
    source_rows = source_payload.get("rows") if isinstance(source_payload, dict) else []
    staged_ocw_tasks = sorted(
        {
            str(row.get("task_type"))
            for row in source_rows
            if isinstance(row, dict)
            and row.get("dataset_increase_percent_vs_local_legacy") is not None
        }
    )
    divergence_rows = [row for row in validated if row.diverges_from_original]
    effect_counts = defaultdict(int)
    for row in validated:
        effect_counts[row.claim_effect] += 1
    lines = [
        "# Phase 04.2 Corrected Evidence Analysis Divergence Report",
        "",
        "## Cutoff Semantics",
        "The 40% reporting heuristic is used only to compare original and corrected "
        "evidence directions; it is not a universal CAPTCHA security boundary.",
        "",
        "## Agreement and Divergence",
        f"Rows analyzed: {len(validated)}.",
        f"Divergent rows: {len(divergence_rows)}.",
        "Claim effects: "
        + ", ".join(f"{key}={value}" for key, value in sorted(effect_counts.items())),
        "",
    ]
    if divergence_rows:
        for row in divergence_rows:
            lines.append(
                f"- {row.provider_model} / {row.task_type} ({row.evidence_mode}): "
                f"{row.divergence_reason} claim_effect={row.claim_effect}."
            )
    else:
        lines.append("- No corrected rows diverged from original Exp2 direction.")
    lines.extend(
        [
            "",
            "## Corrected Provenance Boundary",
            "Corrected direct static evidence is limited to Symbol_Count, "
            "Relation_Match, and Hole_Counting.",
            (
                "No staged OpenCaptchaWorld hard-type increment rows are retained "
                "in the final Phase 04.2 evidence artifacts; no hard-type "
                "dataset-increase percentages are claimed in Phase 04.2."
                if not staged_ocw_tasks
                else (
                    "The updated local OpenCaptchaWorld hard-type incremental rows "
                    "were staged but excluded from corrected direct expanded evidence; "
                    "therefore no Dice_Count/Click_Order/Patch_Select/Geometry_Click "
                    "dataset-increase percentages are claimed in Phase 04.2."
                )
            ),
            f"Staged OCW increment task types inspected: {', '.join(staged_ocw_tasks) or 'none'}.",
            "",
            "## Adaptive Scope",
            PHASE042_ADAPTIVE_SCOPE_RATIONALE,
            "Adaptive rows report observed Success@3 over five memory-isolated "
            "rounds alongside the Exp2-derived Bernoulli baseline for k=3. "
            "Success@5 is retained only when available as auxiliary provenance.",
            "",
        ]
    )
    report = "\n".join(lines)
    assert_no_phase041_reference(report, context="evidence analysis divergence report")
    return report


def write_phase042_evidence_analysis(
    *,
    selected_manifest_path: str | Path = PHASE042_SELECTED_MANIFEST_JSON,
    source_download_manifest_path: str | Path = (
        PHASE042_SIDECAR_ROOT / "source_download_manifest.json"
    ),
    static_summary_paths: Iterable[str | Path] = PHASE042_FINAL_STATIC_SUMMARY_PATHS,
    adaptive_summary_path: str | Path = PHASE042_FINAL_ADAPTIVE_SUMMARY_PATH,
    results_dir: str | Path = "results",
    output_root: str | Path = "results/exp5",
    run_id: str = PHASE042_EVIDENCE_ANALYSIS_RUN_ID,
) -> dict[str, object]:
    rows = build_phase042_evidence_analysis_rows(
        selected_manifest_path=selected_manifest_path,
        source_download_manifest_path=source_download_manifest_path,
        static_summary_paths=static_summary_paths,
        adaptive_summary_path=adaptive_summary_path,
        results_dir=results_dir,
        run_id=run_id,
    )
    run_dir = Path(output_root) / PHASE042_EVIDENCE_ANALYSIS_DIR_NAME
    csv_path = run_dir / "exp5_evidence_analysis.csv"
    json_path = run_dir / "exp5_evidence_analysis.json"
    report_path = run_dir / "exp5_divergence_report.md"
    write_csv(csv_path, Phase042EvidenceAnalysisRow.model_fields, rows)
    write_json(json_path, PHASE042_EVIDENCE_ANALYSIS_SCHEMA_VERSION, rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_phase042_divergence_report(
            rows,
            source_download_manifest_path=source_download_manifest_path,
        ),
        encoding="utf-8",
    )
    return {
        "row_count": len(rows),
        "output_csv": str(csv_path),
        "output_json": str(json_path),
        "divergence_report": str(report_path),
        "claim_effect_counts": {
            effect: sum(1 for row in rows if row.claim_effect == effect)
            for effect in sorted({row.claim_effect for row in rows})
        },
    }


def _load_phase042_evidence_analysis_rows_for_final_outputs(
    path: str | Path,
) -> list[Phase042EvidenceAnalysisRow]:
    evidence_path = Path(path)
    assert_no_phase041_reference(
        evidence_path.as_posix(),
        context="final-output evidence analysis path",
    )
    rows = _payload_rows(evidence_path, label="Phase 04.2 evidence analysis")
    return [Phase042EvidenceAnalysisRow(**row) for row in rows]


def build_phase042_final_evidence_rows(
    evidence_rows: Iterable[Phase042EvidenceAnalysisRow | dict[str, Any]],
    *,
    run_id: str = PHASE042_FINAL_OUTPUTS_RUN_ID,
    source_artifact_path: str = "",
) -> list[Phase042FinalEvidenceRow]:
    validated = [
        row if isinstance(row, Phase042EvidenceAnalysisRow) else Phase042EvidenceAnalysisRow(**row)
        for row in evidence_rows
    ]
    final_rows: list[Phase042FinalEvidenceRow] = []
    for row in validated:
        direct_evidence = (
            row.evidence_mode == "static" and row.task_type in PHASE042_STATIC_TASK_TYPES
        )
        adaptive_hard_scope = row.evidence_mode == "adaptive"
        claim_use = (
            "main_body_direct_evidence"
            if direct_evidence
            and row.real_external_evidence
            and row.claim_effect != "neutral_or_inconclusive"
            else "main_body_caveated"
        )
        final_rows.append(
            Phase042FinalEvidenceRow(
                run_id=run_id,
                evidence_row_id=row.analysis_id,
                evidence_mode=row.evidence_mode,
                provider=row.provider,
                model=row.model,
                provider_model=row.provider_model,
                task_type=row.task_type,
                task_family=row.task_family,
                evidence_origin=row.evidence_origin,
                source_kind=row.source_kind,
                source_provenance_class=row.source_provenance_class,
                source_provenance_notes=row.source_provenance_notes,
                provenance_caveat=row.provenance_caveat,
                real_external_evidence=row.real_external_evidence,
                sample_count=row.sample_count,
                attempt_count=row.attempt_count,
                success_count=row.success_count,
                original_rate=row.original_rate,
                corrected_static_rate=row.corrected_static_rate,
                corrected_adaptive_rate=row.corrected_adaptive_rate,
                adaptive_success_at_3=row.adaptive_success_at_3,
                adaptive_success_at_5=row.adaptive_success_at_5,
                bernoulli_success_at_3=row.bernoulli_success_at_3,
                bernoulli_success_at_5=row.bernoulli_success_at_5,
                round_count=row.adaptive_round_count,
                attempt_budget_k=(
                    PHASE042_ADAPTIVE_ATTEMPT_BUDGET_K if adaptive_hard_scope else 0
                ),
                intermediate_budget_k=(
                    PHASE042_ADAPTIVE_INTERMEDIATE_BUDGET_K if adaptive_hard_scope else 0
                ),
                memory_isolation=row.memory_isolation,
                ci_low=row.ci_low,
                ci_high=row.ci_high,
                ci_method=row.ci_method,
                ci_note=row.ci_note,
                agreement_status=row.agreement_status,
                diverges_from_original=row.diverges_from_original,
                divergence_reason=row.divergence_reason,
                scientific_wrong_count=row.scientific_wrong_count,
                protocol_failure_count=row.protocol_failure_count,
                infrastructure_failure_count=row.infrastructure_failure_count,
                claim_boundary_note=row.claim_boundary_note,
                claim_effect=row.claim_effect,
                direct_evidence=direct_evidence,
                adaptive_hard_scope_evidence=adaptive_hard_scope,
                contextual_sota_only=False,
                claim_use=claim_use,
                source_artifact_path=source_artifact_path or row.source_artifact_path,
                selected_manifest_path=row.selected_manifest_path,
                adaptive_scope_rationale=row.adaptive_scope_rationale,
            )
        )
    assert_no_phase041_reference(
        [row.model_dump(mode="json") for row in final_rows],
        context="final evidence rows",
    )
    return final_rows


def _gpt_fallback_statement(manifest_rows: list[Phase042SelectedManifestRow]) -> str:
    gpt_rows = [row for row in manifest_rows if row.source_kind == PHASE042_GPT_IMAGE_SOURCE_KIND]
    if not gpt_rows:
        return "no GPT Image fallback instances were used"
    return (
        f"{len(gpt_rows)} GPT Image fallback task categories were used and remain "
        "provenance-caveated as not real external dataset evidence"
    )


def render_phase042_claim_boundary_notes(
    final_rows: Iterable[Phase042FinalEvidenceRow | dict[str, Any]],
    *,
    manifest_rows: Iterable[Phase042SelectedManifestRow | dict[str, Any]],
) -> str:
    rows = [
        row if isinstance(row, Phase042FinalEvidenceRow) else Phase042FinalEvidenceRow(**row)
        for row in final_rows
    ]
    selected = [
        row if isinstance(row, Phase042SelectedManifestRow) else Phase042SelectedManifestRow(**row)
        for row in manifest_rows
    ]
    direct_count = sum(1 for row in rows if row.direct_evidence)
    adaptive_count = sum(1 for row in rows if row.adaptive_hard_scope_evidence)
    contextual_count = sum(1 for row in rows if row.contextual_sota_only)
    real_external_count = sum(
        1 for row in selected if row.source_kind in PHASE042_REAL_EXTERNAL_SOURCE_KINDS
    )
    gpt_statement = _gpt_fallback_statement(selected)
    lines = [
        "# Phase 04.2 Corrected Claim Boundary Notes",
        "",
        "## Corrected expanded-dataset evidence",
        (
            f"Direct evidence rows: {direct_count}. These rows are corrected "
            "expanded-dataset evidence for the three new categories only."
        ),
        f"real external samples: {real_external_count} selected task categories.",
        f"GPT Image fallback status: {gpt_statement}.",
        "",
        "## Evidence flags",
        f"adaptive_hard_scope_evidence rows: {adaptive_count}.",
        f"contextual_sota_only rows: {contextual_count}.",
        "",
        "## Scope limits",
        "These outputs do not provide a population-level deployment estimate.",
        (
            "No hard-type dataset-increase percentage claims are made in the "
            "final Phase 04.2 evidence outputs."
        ),
        "",
        "## Claim boundary rows",
    ]
    for row in rows:
        lines.append(
            f"- {row.provider_model} / {row.task_type} / {row.evidence_mode}: "
            f"{row.claim_effect}; claim_boundary_note={row.claim_boundary_note}"
        )
    text = "\n".join(lines) + "\n"
    assert_no_phase041_reference(text, context="final claim boundary notes")
    return text


def render_phase042_source_provenance_notes(
    manifest_rows: Iterable[Phase042SelectedManifestRow | dict[str, Any]],
    *,
    source_download_manifest_path: str | Path,
) -> str:
    selected = [
        row if isinstance(row, Phase042SelectedManifestRow) else Phase042SelectedManifestRow(**row)
        for row in manifest_rows
    ]
    source_payload = _read_json(Path(source_download_manifest_path))
    source_rows = source_payload.get("rows") if isinstance(source_payload, dict) else []
    staged_ocw = sorted(
        {
            str(row.get("task_type"))
            for row in source_rows
            if isinstance(row, dict)
            and row.get("dataset_increase_percent_vs_local_legacy") is not None
        }
    )
    lines = [
        "# Phase 04.2 Source Provenance Notes",
        "",
        "## Three new categories",
        "| task_type | source_kind | source_provenance_class | evidence class |",
        "|---|---|---|---|",
    ]
    for row in selected:
        evidence_class = (
            "real external samples"
            if row.source_kind in PHASE042_REAL_EXTERNAL_SOURCE_KINDS
            else "GPT Image fallback, not real external dataset evidence"
        )
        lines.append(
            f"| {row.task_type} | {row.source_kind} | "
            f"{row.source_provenance_class} | {evidence_class} |"
        )
    lines.extend(
        [
            "",
            "## Model-provider remediation",
            (
                "Qwen-3-VL-235B-Instruct static evidence is represented with the "
                "OpenRouter provider_model because the legacy Fireworks run was "
                "infrastructure-only; infrastructure-only Fireworks rows are not "
                "included in the final evidence set."
            ),
            "",
            "## Excluded staged hard-type increments",
            (
                "No staged OpenCaptchaWorld hard-type increment rows are retained "
                "in the final Phase 04.2 evidence artifacts."
                if not staged_ocw
                else (
                    "Updated local OpenCaptchaWorld hard-type incremental rows were "
                    "staged but excluded from corrected direct expanded evidence."
                )
            ),
            f"Excluded staged task types: {', '.join(staged_ocw) or 'none'}.",
            "No hard-type dataset-increase percentage claims are generated here.",
            "",
        ]
    )
    text = "\n".join(lines)
    assert_no_phase041_reference(text, context="final source provenance notes")
    return text


def render_phase042_adaptive_scope_notes(
    final_rows: Iterable[Phase042FinalEvidenceRow | dict[str, Any]],
) -> str:
    rows = [
        row if isinstance(row, Phase042FinalEvidenceRow) else Phase042FinalEvidenceRow(**row)
        for row in final_rows
    ]
    adaptive_rows = [row for row in rows if row.adaptive_hard_scope_evidence]
    lines = [
        "# Phase 04.2 Adaptive Hard-Scope Notes",
        "",
        PHASE042_ADAPTIVE_SCOPE_RATIONALE,
        "",
        (
            "Adaptive rows are five memory-isolated rounds; the final table reports "
            "Adaptive-Success@3, with Success@5 retained only when available as "
            "auxiliary provenance."
        ),
        "",
        (
            "| task_type | provider_model | adaptive_success_at_3 | "
            "adaptive_success_at_5 | round_count | memory_isolation | "
            "adaptive_hard_scope_evidence | contextual_sota_only |"
        ),
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in adaptive_rows:
        success_at_3 = (
            "n/a" if row.adaptive_success_at_3 is None else row.adaptive_success_at_3
        )
        success_at_5 = (
            "n/a" if row.adaptive_success_at_5 is None else row.adaptive_success_at_5
        )
        lines.append(
            f"| {row.task_type} | {row.provider_model} | {success_at_3} | "
            f"{success_at_5} | {row.round_count} | {row.memory_isolation} | "
            f"{row.adaptive_hard_scope_evidence} | {row.contextual_sota_only} |"
        )
    text = "\n".join(lines) + "\n"
    assert_no_phase041_reference(text, context="final adaptive scope notes")
    return text


def _render_phase042_divergence_notes(
    final_rows: Iterable[Phase042FinalEvidenceRow],
) -> str:
    rows = list(final_rows)
    divergent = [row for row in rows if row.diverges_from_original]
    lines = [
        "# Phase 04.2 Corrected Divergence Notes",
        "",
        "Divergence uses the 40% reporting heuristic only as a reporting comparison.",
        "",
    ]
    if not divergent:
        lines.append("No corrected final rows diverged from original Exp2 direction.")
    for row in divergent:
        lines.append(
            f"- {row.provider_model} / {row.task_type}: {row.divergence_reason}"
        )
    text = "\n".join(lines) + "\n"
    assert_no_phase041_reference(text, context="final divergence notes")
    return text


def _render_phase042_latex_rows(final_rows: Iterable[Phase042FinalEvidenceRow]) -> str:
    lines = [
        "% Exp5 corrected expanded-dataset evidence rows",
        (
            "% task_type & provider_model & direct_evidence & "
            "adaptive_hard_scope_evidence & claim_effect \\\\"
        ),
    ]
    for row in final_rows:
        lines.append(
            f"{row.task_type} & {row.provider_model} & {row.direct_evidence} & "
            f"{row.adaptive_hard_scope_evidence} & {row.claim_effect} \\\\"
        )
    text = "\n".join(lines) + "\n"
    assert_no_phase041_reference(text, context="final table rows")
    return text


def _render_phase042_shepherd_response_snippet(
    final_rows: Iterable[Phase042FinalEvidenceRow],
    *,
    manifest_rows: Iterable[Phase042SelectedManifestRow],
) -> str:
    rows = list(final_rows)
    selected = list(manifest_rows)
    static_tasks = sorted({row.task_type for row in rows if row.direct_evidence})
    adaptive_tasks = sorted({row.task_type for row in rows if row.adaptive_hard_scope_evidence})
    lines = [
        (
            "Exp5 corrected expanded-dataset evidence now uses provenance- "
            "and novelty-compliant rows only."
        ),
        f"Direct corrected categories: {', '.join(static_tasks)}.",
        f"Adaptive hard-scope categories: {', '.join(adaptive_tasks)}.",
        f"GPT Image fallback status: {_gpt_fallback_statement(selected)}.",
        (
            "The selected direct rows use real external samples where source_kind is "
            "peer_reviewed_paper_dataset or open_source_dataset."
        ),
        (
            "No staged OpenCaptchaWorld hard-type increment rows are retained in "
            "the final Phase 04.2 evidence artifacts."
        ),
    ]
    text = "\n".join(lines) + "\n"
    assert_no_phase041_reference(text, context="shepherd response snippet")
    return text


def assert_no_invalid_phase041_rows(paths: str | Path | Iterable[str | Path]) -> None:
    if isinstance(paths, str | Path):
        candidates = [Path(paths)]
    else:
        candidates = [Path(path) for path in paths]

    files: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            files.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file() and path.suffix in PHASE042_FINAL_SCAN_SUFFIXES
            )
        elif candidate.is_file() and candidate.suffix in PHASE042_FINAL_SCAN_SUFFIXES:
            files.append(candidate)

    violations: list[str] = []
    for path in sorted(files):
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for marker in PHASE042_FINAL_INVALID_MARKERS:
            if marker.lower() in lowered:
                violations.append(path.as_posix())
                break
    if violations:
        raise ValueError(
            "invalid marker found in generated Phase 04.2 final outputs: "
            + ", ".join(sorted(set(violations)))
        )


def write_phase042_final_outputs(
    *,
    selected_manifest_path: str | Path = PHASE042_SELECTED_MANIFEST_JSON,
    source_download_manifest_path: str | Path = (
        PHASE042_SIDECAR_ROOT / "source_download_manifest.json"
    ),
    evidence_analysis_path: str | Path = PHASE042_FINAL_EVIDENCE_ANALYSIS_PATH,
    static_summary_paths: Iterable[str | Path] = PHASE042_FINAL_STATIC_SUMMARY_PATHS,
    adaptive_summary_path: str | Path = PHASE042_FINAL_ADAPTIVE_SUMMARY_PATH,
    output_root: str | Path = "results/exp5",
    run_id: str = PHASE042_FINAL_OUTPUTS_RUN_ID,
) -> dict[str, object]:
    assert_no_phase041_reference(run_id, context="final outputs run_id")
    manifest_rows = validate_phase042_selected_manifest(
        load_phase042_selected_manifest(selected_manifest_path)
    )
    # Read summaries as an input gate; final rows are derived from evidence analysis.
    _load_phase042_static_summary_rows(static_summary_paths)
    _load_phase042_adaptive_summary_rows_for_analysis(adaptive_summary_path)
    evidence_rows = _load_phase042_evidence_analysis_rows_for_final_outputs(
        evidence_analysis_path
    )
    final_rows = build_phase042_final_evidence_rows(
        evidence_rows,
        run_id=run_id,
        source_artifact_path=Path(evidence_analysis_path).as_posix(),
    )
    run_dir = Path(output_root) / PHASE042_FINAL_OUTPUTS_DIR_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    rows_csv = run_dir / "corrected_expanded_table_rows.csv"
    rows_json = run_dir / "corrected_expanded_table_rows.json"
    write_csv(rows_csv, Phase042FinalEvidenceRow.model_fields, final_rows)
    write_json(rows_json, PHASE042_FINAL_EVIDENCE_SCHEMA_VERSION, final_rows)

    output_texts = {
        run_dir / "corrected_claim_boundary_notes.md": render_phase042_claim_boundary_notes(
            final_rows,
            manifest_rows=manifest_rows,
        ),
        run_dir / "corrected_divergence_notes.md": _render_phase042_divergence_notes(
            final_rows
        ),
        run_dir / "corrected_source_provenance_notes.md": render_phase042_source_provenance_notes(
            manifest_rows,
            source_download_manifest_path=source_download_manifest_path,
        ),
        run_dir / "corrected_adaptive_scope_notes.md": render_phase042_adaptive_scope_notes(
            final_rows
        ),
        run_dir / "corrected_expanded_table_rows.tex": _render_phase042_latex_rows(
            final_rows
        ),
        (
            run_dir / "corrected_shepherd_response_snippet.txt"
        ): _render_phase042_shepherd_response_snippet(
            final_rows,
            manifest_rows=manifest_rows,
        ),
    }
    for path, text in output_texts.items():
        path.write_text(text, encoding="utf-8")

    assert_no_invalid_phase041_rows(run_dir)
    scan_path = run_dir / "invalid_marker_scan.json"
    _write_json(
        scan_path,
        {
            "schema_version": "cognition.revision.phase042.final_invalid_marker_scan.v1",
            "status": "passed",
            "scanned_file_count": len(
                [
                    path
                    for path in run_dir.rglob("*")
                    if path.is_file() and path.suffix in PHASE042_FINAL_SCAN_SUFFIXES
                ]
            ),
        },
    )
    assert_no_invalid_phase041_rows(run_dir)
    return {
        "final_row_count": len(final_rows),
        "output_csv": str(rows_csv),
        "output_json": str(rows_json),
        "claim_boundary_notes": str(run_dir / "corrected_claim_boundary_notes.md"),
        "source_provenance_notes": str(run_dir / "corrected_source_provenance_notes.md"),
        "adaptive_scope_notes": str(run_dir / "corrected_adaptive_scope_notes.md"),
        "invalid_marker_scan": str(scan_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 04.2 corrected-provenance dataset utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser(
        "validate-sources",
        help="Validate Phase 04.2 candidate sources before selected-manifest use.",
    )
    validate_parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=PHASE042_SIDECAR_ROOT / "candidate_manifest.json",
    )
    validate_parser.add_argument(
        "--captcha-root",
        type=Path,
        default=Path("captcha_data"),
    )
    validate_parser.add_argument(
        "--validation-report-csv",
        type=Path,
        default=PHASE042_VALIDATION_REPORT_CSV,
    )
    validate_parser.add_argument(
        "--validation-report-json",
        type=Path,
        default=PHASE042_VALIDATION_REPORT_JSON,
    )
    validate_parser.add_argument(
        "--selected-manifest-csv",
        type=Path,
        default=PHASE042_SELECTED_MANIFEST_CSV,
    )
    validate_parser.add_argument(
        "--selected-manifest-json",
        type=Path,
        default=PHASE042_SELECTED_MANIFEST_JSON,
    )
    validate_parser.add_argument(
        "--novelty-hash-report",
        type=Path,
        default=PHASE042_NOVELTY_HASH_REPORT_JSON,
    )
    validate_parser.add_argument(
        "--perceptual-warning-threshold",
        type=int,
        default=4,
    )
    materialize_parser = subparsers.add_parser(
        "materialize-selected",
        help="Materialize corrected Phase 04.2 static and adaptive evaluator slices.",
    )
    materialize_parser.add_argument(
        "--selected-manifest",
        type=Path,
        default=PHASE042_SELECTED_MANIFEST_JSON,
    )
    materialize_parser.add_argument(
        "--selected-manifest-csv",
        type=Path,
        default=PHASE042_SELECTED_MANIFEST_CSV,
    )
    materialize_parser.add_argument(
        "--source-download-manifest",
        type=Path,
        default=PHASE042_SIDECAR_ROOT / "source_download_manifest.json",
    )
    materialize_parser.add_argument(
        "--sidecar-root",
        type=Path,
        default=PHASE042_SIDECAR_ROOT,
    )
    materialize_parser.add_argument(
        "--static-output-root",
        type=Path,
        default=PHASE042_EVALUATOR_SLICE,
    )
    materialize_parser.add_argument(
        "--adaptive-output-root",
        type=Path,
        default=PHASE042_ADAPTIVE_EVALUATOR_SLICE,
    )
    materialize_parser.add_argument(
        "--original-dataset-root",
        type=Path,
        default=Path("captcha_data"),
    )
    materialize_parser.add_argument("--overwrite", action="store_true")
    static_preflight_parser = subparsers.add_parser(
        "static-preflight-matrix",
        help="Build Phase 04.2 static supplemental preflight matrix.",
    )
    static_preflight_parser.add_argument(
        "--selected-manifest",
        type=Path,
        default=PHASE042_SELECTED_MANIFEST_JSON,
    )
    static_preflight_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PHASE042_EVALUATOR_SLICE,
    )
    static_preflight_parser.add_argument("--results-dir", type=Path, default=Path("results"))
    static_preflight_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/exp5"),
    )
    static_preflight_parser.add_argument("--run-id", default=PHASE042_STATIC_SUPPLEMENTAL_RUN_ID)
    static_preflight_parser.add_argument(
        "--prompts-file",
        type=Path,
        default=Path("prompts_optimized.yaml"),
    )
    static_preflight_parser.add_argument("--prompt-mode", default="opt")
    static_preflight_parser.add_argument("--pricing-file", type=Path, default=None)
    static_preflight_parser.add_argument("--max-attempts", type=int, default=1)
    static_preflight_parser.add_argument("--overwrite", action="store_true")
    static_preflight_parser.add_argument("--resume", action="store_true", default=True)
    static_preflight_parser.add_argument("--write-reports", action="store_true")
    collect_static_parser = subparsers.add_parser(
        "collect-static",
        help="Run Phase 04.2 static supplemental collection from a preflight matrix.",
    )
    collect_static_parser.add_argument(
        "--preflight-matrix",
        type=Path,
        default=(
            Path("results/local_runs")
            / PHASE042_STATIC_SUPPLEMENTAL_RUN_ID
            / "expanded_static_preflight_matrix.json"
        ),
    )
    collect_static_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/local_runs"),
    )
    collect_static_parser.add_argument("--selected-manifest", type=Path, default=None)
    collect_static_parser.add_argument("--secrets-file", type=Path, default=Path("secrets.yaml"))
    collect_static_parser.add_argument("--resume", action="store_true", default=True)
    collect_static_parser.add_argument("--overwrite", action="store_true")
    collect_static_parser.add_argument("--no-stream", action="store_true")
    collect_static_parser.add_argument("--timeout-sec", type=float, default=600.0)
    collect_static_parser.add_argument("--seed", type=int, default=1234)
    adaptive_preflight_parser = subparsers.add_parser(
        "adaptive-preflight-matrix",
        help="Build Phase 04.2 adaptive supplemental preflight matrix.",
    )
    adaptive_preflight_parser.add_argument(
        "--selected-manifest",
        type=Path,
        default=PHASE042_SELECTED_MANIFEST_JSON,
    )
    adaptive_preflight_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PHASE042_ADAPTIVE_EVALUATOR_SLICE,
    )
    adaptive_preflight_parser.add_argument("--results-dir", type=Path, default=Path("results"))
    adaptive_preflight_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/exp5"),
    )
    adaptive_preflight_parser.add_argument(
        "--run-id",
        default=PHASE042_ADAPTIVE_SUPPLEMENTAL_RUN_ID,
    )
    adaptive_preflight_parser.add_argument(
        "--prompts-file",
        type=Path,
        default=Path("prompts_optimized.yaml"),
    )
    adaptive_preflight_parser.add_argument("--prompt-mode", default="opt")
    adaptive_preflight_parser.add_argument("--pricing-file", type=Path, default=None)
    adaptive_preflight_parser.add_argument(
        "--attempt-budget-k",
        type=int,
        default=PHASE042_ADAPTIVE_ATTEMPT_BUDGET_K,
    )
    adaptive_preflight_parser.add_argument(
        "--intermediate-budget-k",
        type=int,
        default=PHASE042_ADAPTIVE_INTERMEDIATE_BUDGET_K,
    )
    adaptive_preflight_parser.add_argument(
        "--round-count",
        type=int,
        default=PHASE042_ADAPTIVE_ROUND_COUNT,
    )
    adaptive_preflight_parser.add_argument("--seed", type=int, default=1234)
    adaptive_preflight_parser.add_argument("--overwrite", action="store_true")
    adaptive_preflight_parser.add_argument("--resume", action="store_true", default=True)
    adaptive_preflight_parser.add_argument("--write-reports", action="store_true")
    collect_adaptive_parser = subparsers.add_parser(
        "collect-adaptive",
        help="Run Phase 04.2 adaptive supplemental collection after confirmed cost review.",
    )
    collect_adaptive_parser.add_argument(
        "--preflight-matrix",
        type=Path,
        default=(
            Path("results/local_runs")
            / PHASE042_ADAPTIVE_SUPPLEMENTAL_RUN_ID
            / "expanded_adaptive_preflight_matrix.json"
        ),
    )
    collect_adaptive_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/exp5"),
    )
    collect_adaptive_parser.add_argument("--selected-manifest", type=Path, default=None)
    collect_adaptive_parser.add_argument("--secrets-file", type=Path, default=Path("secrets.yaml"))
    collect_adaptive_parser.add_argument("--confirmed-adaptive-cost", action="store_true")
    collect_adaptive_parser.add_argument("--resume", action="store_true", default=True)
    collect_adaptive_parser.add_argument("--overwrite", action="store_true")
    collect_adaptive_parser.add_argument("--no-stream", action="store_true")
    collect_adaptive_parser.add_argument("--timeout-sec", type=float, default=600.0)
    evidence_parser = subparsers.add_parser(
        "evidence-analysis",
        help="Generate corrected Phase 04.2 evidence analysis and divergence reports.",
    )
    evidence_parser.add_argument(
        "--selected-manifest",
        type=Path,
        default=PHASE042_SELECTED_MANIFEST_JSON,
    )
    evidence_parser.add_argument(
        "--source-download-manifest",
        type=Path,
        default=PHASE042_SIDECAR_ROOT / "source_download_manifest.json",
    )
    evidence_parser.add_argument(
        "--static-summary",
        type=Path,
        action="append",
        default=None,
        help=(
            "Static summary path. May be repeated. If omitted, the final evidence "
            "Phase 04.2 static summary is used."
        ),
    )
    evidence_parser.add_argument(
        "--adaptive-summary",
        type=Path,
        default=PHASE042_FINAL_ADAPTIVE_SUMMARY_PATH,
        help="Corrected GPT-5 medium adaptive summary path.",
    )
    evidence_parser.add_argument("--results-dir", type=Path, default=Path("results"))
    evidence_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/exp5"),
    )
    evidence_parser.add_argument("--run-id", default=PHASE042_EVIDENCE_ANALYSIS_RUN_ID)
    final_parser = subparsers.add_parser(
        "final-outputs",
        help="Generate corrected Phase 04.2 table-ready rows and notes.",
    )
    final_parser.add_argument(
        "--selected-manifest",
        type=Path,
        default=PHASE042_SELECTED_MANIFEST_JSON,
    )
    final_parser.add_argument(
        "--source-download-manifest",
        type=Path,
        default=PHASE042_SIDECAR_ROOT / "source_download_manifest.json",
    )
    final_parser.add_argument(
        "--evidence-analysis",
        type=Path,
        default=PHASE042_FINAL_EVIDENCE_ANALYSIS_PATH,
    )
    final_parser.add_argument(
        "--static-summary",
        type=Path,
        action="append",
        default=None,
        help=(
            "Static summary path. May be repeated. If omitted, the final evidence "
            "Phase 04.2 static summary is used as an input gate."
        ),
    )
    final_parser.add_argument(
        "--adaptive-summary",
        type=Path,
        default=PHASE042_FINAL_ADAPTIVE_SUMMARY_PATH,
    )
    final_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/exp5"),
    )
    final_parser.add_argument("--run-id", default=PHASE042_FINAL_OUTPUTS_RUN_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-sources":
        result = validate_phase042_candidates(
            candidate_manifest_path=args.candidate_manifest,
            captcha_root=args.captcha_root,
            validation_report_csv=args.validation_report_csv,
            validation_report_json=args.validation_report_json,
            selected_manifest_csv=args.selected_manifest_csv,
            selected_manifest_json=args.selected_manifest_json,
            novelty_hash_report_path=args.novelty_hash_report,
            perceptual_warning_threshold=args.perceptual_warning_threshold,
        )
        print(
            "Phase 04.2 validation complete: "
            f"{result['selected_count']} selected, {result['rejected_count']} rejected"
        )
        return 0
    if args.command == "materialize-selected":
        result = materialize_phase042_selected(
            selected_manifest_path=args.selected_manifest,
            selected_manifest_csv=args.selected_manifest_csv,
            source_download_manifest_path=args.source_download_manifest,
            sidecar_root=args.sidecar_root,
            static_output_root=args.static_output_root,
            adaptive_output_root=args.adaptive_output_root,
            original_dataset_root=args.original_dataset_root,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "static-preflight-matrix":
        rows = build_phase042_static_preflight_matrix(
            selected_manifest_path=args.selected_manifest,
            dataset_root=args.dataset_root,
            results_dir=args.results_dir,
            output_root=args.output_root,
            run_id=args.run_id,
            prompts_file=args.prompts_file,
            prompt_mode=args.prompt_mode,
            max_attempts=args.max_attempts,
            resume=args.resume,
            overwrite=args.overwrite,
            write_reports=args.write_reports,
            pricing_file=args.pricing_file,
        )
        run_dir = revision_run_dir(args.output_root, args.run_id)
        print(
            json.dumps(
                {
                    "row_count": len(rows),
                    "output_csv": str(run_dir / "expanded_static_preflight_matrix.csv"),
                    "output_json": str(run_dir / "expanded_static_preflight_matrix.json"),
                    "preflight_reports_dir": str(run_dir / "preflight_reports"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "collect-static":
        result = collect_phase042_static_runs(
            preflight_matrix_path=args.preflight_matrix,
            output_root=args.output_root,
            selected_manifest_path=args.selected_manifest,
            resume=args.resume,
            overwrite=args.overwrite,
            secrets_file=args.secrets_file,
            stream=not args.no_stream,
            timeout_sec=args.timeout_sec,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "adaptive-preflight-matrix":
        rows = build_phase042_adaptive_preflight_matrix(
            selected_manifest_path=args.selected_manifest,
            dataset_root=args.dataset_root,
            results_dir=args.results_dir,
            output_root=args.output_root,
            run_id=args.run_id,
            prompts_file=args.prompts_file,
            prompt_mode=args.prompt_mode,
            attempt_budget_k=args.attempt_budget_k,
            intermediate_budget_k=args.intermediate_budget_k,
            round_count=args.round_count,
            seed=args.seed,
            resume=args.resume,
            overwrite=args.overwrite,
            write_reports=args.write_reports,
            pricing_file=args.pricing_file,
        )
        run_dir = revision_run_dir(args.output_root, args.run_id)
        print(
            json.dumps(
                {
                    "row_count": len(rows),
                    "output_csv": str(run_dir / "expanded_adaptive_preflight_matrix.csv"),
                    "output_json": str(run_dir / "expanded_adaptive_preflight_matrix.json"),
                    "preflight_reports_dir": str(run_dir / "adaptive_preflight_reports"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "collect-adaptive":
        result = collect_phase042_adaptive_runs(
            preflight_matrix_path=args.preflight_matrix,
            output_root=args.output_root,
            selected_manifest_path=args.selected_manifest,
            confirmed_adaptive_cost=args.confirmed_adaptive_cost,
            resume=args.resume,
            overwrite=args.overwrite,
            secrets_file=args.secrets_file,
            stream=not args.no_stream,
            timeout_sec=args.timeout_sec,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "evidence-analysis":
        result = write_phase042_evidence_analysis(
            selected_manifest_path=args.selected_manifest,
            source_download_manifest_path=args.source_download_manifest,
            static_summary_paths=args.static_summary or PHASE042_FINAL_STATIC_SUMMARY_PATHS,
            adaptive_summary_path=args.adaptive_summary,
            results_dir=args.results_dir,
            output_root=args.output_root,
            run_id=args.run_id,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "final-outputs":
        result = write_phase042_final_outputs(
            selected_manifest_path=args.selected_manifest,
            source_download_manifest_path=args.source_download_manifest,
            evidence_analysis_path=args.evidence_analysis,
            static_summary_paths=args.static_summary or PHASE042_FINAL_STATIC_SUMMARY_PATHS,
            adaptive_summary_path=args.adaptive_summary,
            output_root=args.output_root,
            run_id=args.run_id,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
