import csv
import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator


DATASET_SCOPE_SCHEMA_VERSION = "cognition.revision.dataset_scope_audit.v1"
EXTENDED_DATASET_MANIFEST_SCHEMA_VERSION = "cognition.revision.extended_dataset_manifest.v1"
EXTENDED_VALIDATION_COMPARISON_SCHEMA_VERSION = (
    "cognition.revision.extended_validation_comparison.v1"
)
PASS_RATE_CONFIDENCE_SCHEMA_VERSION = "cognition.revision.pass_rate_confidence.v1"
THRESHOLD_SENSITIVITY_SCHEMA_VERSION = "cognition.revision.threshold_sensitivity.v1"
RETRY_CALIBRATION_SCHEMA_VERSION = "cognition.revision.retry_calibration.v1"
FAILURE_TAXONOMY_SCHEMA_VERSION = "cognition.revision.failure_taxonomy.v1"

ALLOWED_SCOPE_STATUSES = {"included", "excluded", "incompatible", "underpowered"}
ALLOWED_SUPPORT_STATUSES = {
    "supported",
    "unsupported",
    "dataset_supported_not_evaluated",
    "removed_not_used",
}
ALLOWED_PIPELINE_COMPATIBILITY = {
    "compatible_static_answer",
    "incompatible_temporal_hold",
    "incompatible_slider_drag",
    "unknown",
}
ALLOWED_KEY_STATUSES = {"present", "missing", "not_applicable"}
ALLOWED_EVIDENCE_ORIGINS = {
    "original_captchaworld",
    "supplemented_category",
    "new_category",
}
ALLOWED_SLICE_TYPES = {"original", "supplement_existing", "new_category"}
ALLOWED_COMPATIBILITY_STATUSES = {
    "ready_for_static_pipeline",
    "needs_normalization",
    "incompatible_static_pipeline",
}
ALLOWED_EVALUATION_STATUSES = {
    "selected_for_validation",
    "evaluated",
    "excluded_with_reason",
}
ALLOWED_AGREEMENT_STATUSES = {
    "supports_original",
    "diverges_from_original",
    "inconclusive",
}
ALLOWED_CLAIM_USES = {
    "scientific_claim_eligible",
    "infrastructure_caveated",
    "protocol_caveated",
    "aggregate_only_caveated",
}


def _validate_allowed(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return value


def _row_to_json_dict(row: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(row, BaseModel):
        return row.model_dump(mode="json")
    return row


class DatasetScopeAuditRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = DATASET_SCOPE_SCHEMA_VERSION
    run_id: str
    task_type: str
    task_family: str
    dataset_dir: str
    scope_status: str
    support_status: str
    pipeline_compatibility: str
    dataset_sample_count: int = Field(ge=0)
    evaluated_n_exp1: int = Field(ge=0)
    evaluated_n_exp2: int = Field(ge=0)
    evaluated_n_exp3: int = Field(ge=0)
    evaluated_n_exp4: int = Field(ge=0)
    underpowered_threshold: int = Field(ge=0)
    underpowered: bool
    prompt_key_status: str
    few_shot_key_status: str
    answer_format_notes: str
    normalization_notes: str
    removal_decision: str
    reason: str

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(value, {DATASET_SCOPE_SCHEMA_VERSION}, "schema_version")

    @field_validator("scope_status")
    @classmethod
    def validate_scope_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_SCOPE_STATUSES, "scope_status")

    @field_validator("support_status")
    @classmethod
    def validate_support_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_SUPPORT_STATUSES, "support_status")

    @field_validator("pipeline_compatibility")
    @classmethod
    def validate_pipeline_compatibility(cls, value: str) -> str:
        return _validate_allowed(
            value, ALLOWED_PIPELINE_COMPATIBILITY, "pipeline_compatibility"
        )

    @field_validator("prompt_key_status", "few_shot_key_status")
    @classmethod
    def validate_key_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_KEY_STATUSES, "key_status")


class ExtendedDatasetManifestRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = EXTENDED_DATASET_MANIFEST_SCHEMA_VERSION
    run_id: str
    source_id: str
    evidence_origin: str
    slice_type: str
    task_type: str
    task_family: str
    source_path: str
    sample_count: int = Field(ge=0)
    label_format: str
    metadata_alignment_notes: str
    answer_format_normalization: str
    task_family_grouping_notes: str
    normalization_decisions: str
    compatibility_status: str
    evaluation_status: str
    adaptive_eligible: bool
    adaptive_slice_priority: int = Field(ge=0)
    validation_question: str
    rerun_policy: str
    limitation_note: str

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(
            value, {EXTENDED_DATASET_MANIFEST_SCHEMA_VERSION}, "schema_version"
        )

    @field_validator("evidence_origin")
    @classmethod
    def validate_evidence_origin(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_EVIDENCE_ORIGINS, "evidence_origin")

    @field_validator("slice_type")
    @classmethod
    def validate_slice_type(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_SLICE_TYPES, "slice_type")

    @field_validator("compatibility_status")
    @classmethod
    def validate_compatibility_status(cls, value: str) -> str:
        return _validate_allowed(
            value, ALLOWED_COMPATIBILITY_STATUSES, "compatibility_status"
        )

    @field_validator("evaluation_status")
    @classmethod
    def validate_evaluation_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_EVALUATION_STATUSES, "evaluation_status")


class ExtendedValidationComparisonRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = EXTENDED_VALIDATION_COMPARISON_SCHEMA_VERSION
    run_id: str
    source_id: str
    evidence_origin: str
    slice_type: str
    task_type: str
    task_family: str
    original_conclusion_label: str | None = None
    original_rate: float | None = Field(default=None, ge=0, le=1)
    validation_slice_rate: float | None = Field(default=None, ge=0, le=1)
    validation_sample_count: int = Field(ge=0)
    agreement_status: str
    divergence_reason: str
    comparison_caveat: str
    outcome_source_path: str

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(
            value, {EXTENDED_VALIDATION_COMPARISON_SCHEMA_VERSION}, "schema_version"
        )

    @field_validator("evidence_origin")
    @classmethod
    def validate_evidence_origin(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_EVIDENCE_ORIGINS, "evidence_origin")

    @field_validator("slice_type")
    @classmethod
    def validate_slice_type(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_SLICE_TYPES, "slice_type")

    @field_validator("agreement_status")
    @classmethod
    def validate_agreement_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_AGREEMENT_STATUSES, "agreement_status")


class PassRateConfidenceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PASS_RATE_CONFIDENCE_SCHEMA_VERSION
    run_id: str
    aggregation_level: str
    provider: str
    model: str
    provider_model: str
    experiment: str
    task_type: str
    task_family: str
    n_attempts: int = Field(ge=0)
    n_success: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)
    ci_method: str
    ci_confidence: float = Field(gt=0, le=1)
    ci_low: float | None = Field(default=None, ge=0, le=1)
    ci_high: float | None = Field(default=None, ge=0, le=1)
    underpowered_threshold: int = Field(ge=0)
    underpowered: bool
    source_path: str

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(value, {PASS_RATE_CONFIDENCE_SCHEMA_VERSION}, "schema_version")


class ThresholdSensitivityRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = THRESHOLD_SENSITIVITY_SCHEMA_VERSION
    run_id: str
    provider: str
    model: str
    provider_model: str
    task_type: str
    task_family: str
    primary_experiment: str
    primary_rate: float = Field(ge=0, le=1)
    max_observed_rate: float = Field(ge=0, le=1)
    label: str
    margin_to_cutoff: float
    cutoff: float = Field(ge=0, le=1)
    review_band_low: float = Field(ge=0, le=1)
    review_band_high: float = Field(ge=0, le=1)
    in_30_50_review_band: bool
    ci_crosses_cutoff: bool
    trend_sensitive: bool
    trend_delta: float
    trend_sources: str
    cutoff_note: str

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(
            value, {THRESHOLD_SENSITIVITY_SCHEMA_VERSION}, "schema_version"
        )


class RetryCalibrationRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = RETRY_CALIBRATION_SCHEMA_VERSION
    run_id: str
    provider: str
    model: str
    provider_model: str
    task_type: str
    task_family: str
    exp2_pass_at_1: float | None = Field(default=None, ge=0, le=1)
    attempt_budget_k: int = Field(ge=1)
    bernoulli_success_at_k: float | None = Field(default=None, ge=0, le=1)
    observed_fixed_retry_success: float | None = Field(default=None, ge=0, le=1)
    observed_adaptive_compatible_success: float | None = Field(default=None, ge=0, le=1)
    signed_error_fixed_retry: float | None = None
    absolute_error_fixed_retry: float | None = Field(default=None, ge=0)
    signed_error_adaptive: float | None = None
    absolute_error_adaptive: float | None = Field(default=None, ge=0)
    sample_count: int = Field(ge=0)
    scientific_wrong_count: int = Field(ge=0)
    protocol_failure_count: int = Field(ge=0)
    infrastructure_failure_count: int = Field(ge=0)
    raw_observed_rate: float | None = Field(default=None, ge=0, le=1)
    scientific_rate: float | None = Field(default=None, ge=0, le=1)
    comparison_contract: str

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(value, {RETRY_CALIBRATION_SCHEMA_VERSION}, "schema_version")


class FailureTaxonomyRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = FAILURE_TAXONOMY_SCHEMA_VERSION
    run_id: str
    aggregation_level: str
    provider: str
    model: str
    provider_model: str
    task_type: str
    task_family: str
    success_count: int = Field(ge=0)
    scientific_wrong_count: int = Field(ge=0)
    protocol_failure_count: int = Field(ge=0)
    infrastructure_failure_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    raw_observed_rate: float | None = Field(default=None, ge=0, le=1)
    scientific_rate: float | None = Field(default=None, ge=0, le=1)
    failure_taxonomy_source: str
    claim_use: str
    hardness_caveat: str | None = None

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(value, {FAILURE_TAXONOMY_SCHEMA_VERSION}, "schema_version")

    @field_validator("claim_use")
    @classmethod
    def validate_claim_use(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_CLAIM_USES, "claim_use")


def write_csv(
    path: Path,
    field_map: dict[str, Any],
    rows: Iterable[BaseModel | dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(field_map)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_json_dict(row))


def write_json(
    path: Path,
    schema_version: str,
    rows: Iterable[BaseModel | dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": schema_version,
                "rows": [_row_to_json_dict(row) for row in rows],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
