import csv
import json
from pathlib import Path
from typing import Any, Iterable, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PHASE042_SELECTED_MANIFEST_SCHEMA_VERSION = (
    "cognition.revision.phase042.selected_manifest.v1"
)
PHASE042_VALIDATION_REPORT_SCHEMA_VERSION = (
    "cognition.revision.phase042.validation_report.v1"
)
PHASE042_PREFLIGHT_MATRIX_SCHEMA_VERSION = (
    "cognition.revision.phase042.preflight_matrix.v1"
)
PHASE042_STATIC_SUMMARY_SCHEMA_VERSION = (
    "cognition.revision.phase042.static_summary.v1"
)
PHASE042_ADAPTIVE_SUMMARY_SCHEMA_VERSION = (
    "cognition.revision.phase042.adaptive_summary.v1"
)
PHASE042_EVIDENCE_ANALYSIS_SCHEMA_VERSION = (
    "cognition.revision.phase042.evidence_analysis.v1"
)
PHASE042_FINAL_EVIDENCE_SCHEMA_VERSION = (
    "cognition.revision.phase042.final_evidence.v1"
)

PHASE042_SELECTED_SOURCE_KINDS = {
    "peer_reviewed_paper_dataset",
    "open_source_dataset",
    "gpt_image_open_captchaworld_style",
}
PHASE042_REAL_EXTERNAL_SOURCE_KINDS = {
    "peer_reviewed_paper_dataset",
    "open_source_dataset",
}
PHASE042_GPT_IMAGE_SOURCE_KIND = "gpt_image_open_captchaworld_style"

ALLOWED_EVIDENCE_ORIGINS = {
    "supplemented_category",
    "new_category",
}
ALLOWED_SLICE_TYPES = {
    "supplement_existing",
    "new_category",
}
ALLOWED_COMPATIBILITY_STATUSES = {
    "ready_for_static_pipeline",
    "needs_normalization",
    "incompatible_static_pipeline",
}
ALLOWED_EVALUATION_STATUSES = {
    "selected_for_static",
    "selected_for_adaptive",
    "evaluated_static",
    "evaluated_adaptive",
    "excluded_with_reason",
}
ALLOWED_VALIDATION_STATUSES = {
    "accepted",
    "rejected",
}
ALLOWED_RUN_SCOPES = {
    "static",
    "adaptive",
}
ALLOWED_AGREEMENT_STATUSES = {
    "supports_original",
    "diverges_from_original",
    "inconclusive",
}
ALLOWED_CLAIM_EFFECTS = {
    "supports_structural_hardness",
    "weakens_structural_hardness",
    "neutral_or_inconclusive",
}
ALLOWED_CLAIM_USES = {
    "main_body_direct_evidence",
    "main_body_caveated",
    "appendix_context",
    "excluded_from_claim",
}
PHASE042_INVALID_REFERENCE_MARKERS = (
    "phase04_1",
    "synthetic_fixture",
    "copied from existing local captcha_data",
    "scripted local prototype",
)

RowModel = TypeVar("RowModel", bound=BaseModel)


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
        if marker in text:
            raise ValueError(
                f"{context} must not reference invalid Phase 04.1/source marker"
            )


def _validate_allowed(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return value


def _require_text(value: str, field_name: str, context: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} is required when {context}")


def _require_items(values: list[str], field_name: str, context: str) -> None:
    if not values:
        raise ValueError(f"{field_name} is required when {context}")
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} cannot contain blank items")


def _validate_sha256_values(values: list[str], field_name: str) -> list[str]:
    _require_items(values, field_name, "recording Phase 04.2 novelty")
    for value in values:
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"{field_name} entries must be lowercase SHA-256 hex")
    return values


def _row_to_json_dict(row: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(row, BaseModel):
        return row.model_dump(mode="json")
    return row


def _row_to_csv_dict(row: BaseModel | dict[str, Any]) -> dict[str, Any]:
    payload = _row_to_json_dict(row)
    return {
        key: json.dumps(value, ensure_ascii=False)
        if isinstance(value, (list, dict))
        else value
        for key, value in payload.items()
    }


def _validated_rows(
    model: type[RowModel],
    rows: Iterable[RowModel | dict[str, Any]],
) -> list[RowModel]:
    return [row if isinstance(row, model) else model.model_validate(row) for row in rows]


def _guard_phase042_downstream_row(row: BaseModel, context: str) -> None:
    assert_no_phase041_reference(row.model_dump(mode="json"), context=context)


class Phase042SelectedManifestRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PHASE042_SELECTED_MANIFEST_SCHEMA_VERSION
    selected_id: str
    candidate_id: str
    source_path: str
    candidate_image_paths: list[str]
    source_kind: str
    source_provenance_class: str
    source_citation: str = ""
    source_license: str = ""
    source_provenance_notes: str
    evidence_origin: str
    slice_type: str
    task_type: str
    task_family: str
    sample_count: int = Field(ge=1)
    label_format: str
    metadata_alignment_notes: str
    answer_format_normalization: str
    compatibility_status: str
    evaluation_status: str
    limitation_notes: str
    adaptive_eligible: bool
    static_compatibility_notes: str
    novelty_sha256: list[str]
    novelty_hash_report_path: str
    exact_captcha_data_match: bool
    perceptual_warning_count: int = Field(ge=0)
    review_warnings: list[str] = Field(default_factory=list)
    gpt_image_generation_prompt: str = ""
    gpt_image_model: str = ""
    gpt_image_generation_date: str = ""
    open_captchaworld_style_rationale: str = ""

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(
            value,
            {PHASE042_SELECTED_MANIFEST_SCHEMA_VERSION},
            "schema_version",
        )

    @field_validator("source_kind")
    @classmethod
    def validate_source_kind(cls, value: str) -> str:
        return _validate_allowed(value, PHASE042_SELECTED_SOURCE_KINDS, "source_kind")

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
            value,
            ALLOWED_COMPATIBILITY_STATUSES,
            "compatibility_status",
        )

    @field_validator("evaluation_status")
    @classmethod
    def validate_evaluation_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_EVALUATION_STATUSES, "evaluation_status")

    @field_validator("candidate_image_paths")
    @classmethod
    def validate_candidate_image_paths(cls, values: list[str]) -> list[str]:
        _require_items(
            values,
            "candidate_image_paths",
            "building a Phase 04.2 selected manifest row",
        )
        return values

    @field_validator("novelty_sha256")
    @classmethod
    def validate_novelty_sha256(cls, values: list[str]) -> list[str]:
        return _validate_sha256_values(values, "novelty_sha256")

    @model_validator(mode="after")
    def validate_selected_manifest_contract(self) -> "Phase042SelectedManifestRow":
        _guard_phase042_downstream_row(self, "selected manifest row")
        _require_text(
            self.source_provenance_class,
            "source_provenance_class",
            "row is selected for Phase 04.2 evaluation",
        )
        _require_text(
            self.source_provenance_notes,
            "source_provenance_notes",
            "row is selected for Phase 04.2 evaluation",
        )
        _require_text(
            self.novelty_hash_report_path,
            "novelty_hash_report_path",
            "row is selected for Phase 04.2 evaluation",
        )
        if self.exact_captcha_data_match and not any(
            "exact SHA-256 match" in warning for warning in self.review_warnings
        ):
            raise ValueError(
                "exact_captcha_data_match selected rows require an exact SHA-256 "
                "review warning"
            )
        if self.source_kind in PHASE042_REAL_EXTERNAL_SOURCE_KINDS:
            _require_text(
                self.source_citation,
                "source_citation",
                "selected real external row is used",
            )
            _require_text(
                self.source_license,
                "source_license",
                "selected real external row is used",
            )
        elif self.source_kind == PHASE042_GPT_IMAGE_SOURCE_KIND:
            _require_text(
                self.gpt_image_generation_prompt,
                "gpt_image_generation_prompt",
                "selected GPT Image row is used",
            )
            _require_text(
                self.gpt_image_model,
                "gpt_image_model",
                "selected GPT Image row is used",
            )
            _require_text(
                self.gpt_image_generation_date,
                "gpt_image_generation_date",
                "selected GPT Image row is used",
            )
            _require_text(
                self.open_captchaworld_style_rationale,
                "open_captchaworld_style_rationale",
                "selected GPT Image row is used",
            )
        return self


class Phase042ValidationReportRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PHASE042_VALIDATION_REPORT_SCHEMA_VERSION
    candidate_id: str
    task_type: str
    source_kind: str
    source_path: str
    candidate_image_paths: list[str]
    validation_status: str
    selected_manifest_eligible: bool
    rejection_reason: str = ""
    exact_captcha_data_match: bool
    exact_captcha_data_match_paths: list[str] = Field(default_factory=list)
    novelty_sha256: list[str] = Field(default_factory=list)
    novelty_hash_report_path: str
    perceptual_warning_count: int = Field(ge=0)
    review_warnings: list[str] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(
            value,
            {PHASE042_VALIDATION_REPORT_SCHEMA_VERSION},
            "schema_version",
        )

    @field_validator("validation_status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_VALIDATION_STATUSES, "validation_status")

    @model_validator(mode="after")
    def validate_validation_report_contract(self) -> "Phase042ValidationReportRow":
        if self.validation_status == "rejected" or not self.selected_manifest_eligible:
            _require_text(
                self.rejection_reason,
                "rejection_reason",
                "validation row is rejected or ineligible",
            )
        if self.validation_status == "accepted" and not self.selected_manifest_eligible:
            raise ValueError("accepted validation rows must be selected_manifest_eligible")
        return self


class Phase042PreflightMatrixRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PHASE042_PREFLIGHT_MATRIX_SCHEMA_VERSION
    run_id: str
    provider: str
    model: str
    provider_model: str
    run_scope: str
    selected_manifest_path: str
    selected_manifest_sha256: str
    materialized_dataset_root: str
    task_types: list[str]
    prompt_config: dict[str, Any]
    expected_request_count: int = Field(ge=0)
    cost_preview: dict[str, Any]
    output_dir: str
    preflight_report_path: str
    overwrite: bool
    resume: bool
    attempt_budget_k: int | None = Field(default=None, ge=1)
    sampling_mode: str | None = None
    feedback_mode: str | None = None
    memory_mode: str | None = None
    stopping_rule: str | None = None
    solve_request_count: int | None = Field(default=None, ge=0)
    reflection_request_count_max: int | None = Field(default=None, ge=0)
    expected_request_count_max: int | None = Field(default=None, ge=0)
    round_id: str | None = None
    round_index: int | None = Field(default=None, ge=1)
    round_count: int | None = Field(default=None, ge=1)
    seed: int | None = None
    intermediate_budget_k: int | None = Field(default=None, ge=1)
    adaptive_scope_rationale: str | None = None

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(
            value,
            {PHASE042_PREFLIGHT_MATRIX_SCHEMA_VERSION},
            "schema_version",
        )

    @field_validator("run_scope")
    @classmethod
    def validate_run_scope(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_RUN_SCOPES, "run_scope")

    @field_validator("task_types")
    @classmethod
    def validate_task_types(cls, values: list[str]) -> list[str]:
        _require_items(values, "task_types", "building a Phase 04.2 preflight row")
        return values

    @model_validator(mode="after")
    def validate_phase042_boundaries(self) -> "Phase042PreflightMatrixRow":
        _guard_phase042_downstream_row(self, "preflight matrix row")
        return self


class Phase042StaticSummaryRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PHASE042_STATIC_SUMMARY_SCHEMA_VERSION
    run_id: str
    provider: str
    model: str
    provider_model: str
    task_type: str
    task_family: str
    evidence_origin: str
    sample_count: int = Field(ge=0)
    attempt_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    scientific_wrong_count: int = Field(ge=0)
    protocol_failure_count: int = Field(ge=0)
    infrastructure_failure_count: int = Field(ge=0)
    pass_rate: float | None = Field(default=None, ge=0, le=1)
    run_manifest_path: str
    attempt_log_path: str
    summary_source_path: str
    selected_manifest_path: str
    claim_use: str

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(
            value,
            {PHASE042_STATIC_SUMMARY_SCHEMA_VERSION},
            "schema_version",
        )

    @field_validator("evidence_origin")
    @classmethod
    def validate_evidence_origin(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_EVIDENCE_ORIGINS, "evidence_origin")

    @field_validator("claim_use")
    @classmethod
    def validate_claim_use(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_CLAIM_USES, "claim_use")

    @model_validator(mode="after")
    def validate_phase042_boundaries(self) -> "Phase042StaticSummaryRow":
        _guard_phase042_downstream_row(self, "static summary row")
        return self


class Phase042AdaptiveSummaryRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PHASE042_ADAPTIVE_SUMMARY_SCHEMA_VERSION
    run_id: str
    provider: str
    model: str
    provider_model: str
    task_type: str
    task_family: str
    evidence_origin: str
    sample_count: int = Field(ge=0)
    session_count: int = Field(ge=0)
    round_id: str | None = None
    round_index: int | None = Field(default=None, ge=1)
    round_count: int | None = Field(default=None, ge=1)
    attempt_budget_k: int = Field(ge=1)
    intermediate_budget_k: int | None = Field(default=None, ge=1)
    success_count: int = Field(ge=0)
    success_at_3: bool | None = None
    success_at_5: bool | None = None
    attempts_to_success_at_3: int | None = Field(default=None, ge=1)
    attempts_to_success_at_5: int | None = Field(default=None, ge=1)
    scientific_wrong_count: int = Field(ge=0)
    protocol_failure_count: int = Field(ge=0)
    infrastructure_failure_count: int = Field(ge=0)
    adaptive_success_rate: float | None = Field(default=None, ge=0, le=1)
    feedback_mode: str
    memory_mode: str
    stopping_rule: str
    run_manifest_path: str
    adaptive_attempt_log_path: str
    adaptive_summary_source_path: str
    selected_manifest_path: str
    claim_use: str

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(
            value,
            {PHASE042_ADAPTIVE_SUMMARY_SCHEMA_VERSION},
            "schema_version",
        )

    @field_validator("evidence_origin")
    @classmethod
    def validate_evidence_origin(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_EVIDENCE_ORIGINS, "evidence_origin")

    @field_validator("claim_use")
    @classmethod
    def validate_claim_use(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_CLAIM_USES, "claim_use")

    @model_validator(mode="after")
    def validate_phase042_boundaries(self) -> "Phase042AdaptiveSummaryRow":
        _guard_phase042_downstream_row(self, "adaptive summary row")
        return self


class Phase042EvidenceAnalysisRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PHASE042_EVIDENCE_ANALYSIS_SCHEMA_VERSION
    analysis_id: str
    run_id: str = ""
    evidence_mode: str = ""
    provider: str = ""
    model: str = ""
    task_type: str
    task_family: str
    provider_model: str
    evidence_origin: str = ""
    sample_count: int = Field(ge=0)
    attempt_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    source_kind: str = ""
    source_provenance_class: str = ""
    source_citation: str = ""
    source_license: str = ""
    source_provenance_notes: str = ""
    provenance_caveat: str = ""
    direct_evidence: bool = True
    real_external_evidence: bool = False
    original_rate: float | None = Field(default=None, ge=0, le=1)
    corrected_static_rate: float | None = Field(default=None, ge=0, le=1)
    corrected_adaptive_rate: float | None = Field(default=None, ge=0, le=1)
    adaptive_success_at_3: float | None = Field(default=None, ge=0, le=1)
    adaptive_success_at_5: float | None = Field(default=None, ge=0, le=1)
    bernoulli_success_at_3: float | None = Field(default=None, ge=0, le=1)
    bernoulli_success_at_5: float | None = Field(default=None, ge=0, le=1)
    adaptive_round_count: int = Field(default=0, ge=0)
    memory_isolation: str = ""
    ci_low: float | None = Field(default=None, ge=0, le=1)
    ci_high: float | None = Field(default=None, ge=0, le=1)
    ci_method: str = ""
    ci_note: str = ""
    agreement_status: str
    diverges_from_original: bool = False
    divergence_reason: str
    scientific_wrong_count: int = Field(ge=0)
    protocol_failure_count: int = Field(ge=0)
    infrastructure_failure_count: int = Field(ge=0)
    selected_manifest_path: str
    claim_boundary_note: str
    claim_effect: str = "neutral_or_inconclusive"
    adaptive_scope_rationale: str = ""
    source_artifact_path: str = ""

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(
            value,
            {PHASE042_EVIDENCE_ANALYSIS_SCHEMA_VERSION},
            "schema_version",
        )

    @field_validator("agreement_status")
    @classmethod
    def validate_agreement_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_AGREEMENT_STATUSES, "agreement_status")

    @field_validator("claim_effect")
    @classmethod
    def validate_claim_effect(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_CLAIM_EFFECTS, "claim_effect")

    @model_validator(mode="after")
    def validate_analysis_contract(self) -> "Phase042EvidenceAnalysisRow":
        _guard_phase042_downstream_row(self, "evidence analysis row")
        if self.agreement_status == "diverges_from_original":
            _require_text(
                self.divergence_reason,
                "divergence_reason",
                "agreement_status is diverges_from_original",
            )
        if self.diverges_from_original != (
            self.agreement_status == "diverges_from_original"
        ):
            raise ValueError(
                "diverges_from_original must match agreement_status"
            )
        return self


class Phase042FinalEvidenceRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PHASE042_FINAL_EVIDENCE_SCHEMA_VERSION
    run_id: str
    evidence_row_id: str
    evidence_mode: str = ""
    provider: str
    model: str
    provider_model: str
    task_type: str
    task_family: str
    evidence_origin: str
    source_kind: str = ""
    source_provenance_class: str = ""
    source_provenance_notes: str = ""
    provenance_caveat: str = ""
    real_external_evidence: bool = False
    sample_count: int = Field(ge=0)
    attempt_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    original_rate: float | None = Field(default=None, ge=0, le=1)
    corrected_static_rate: float | None = Field(default=None, ge=0, le=1)
    corrected_adaptive_rate: float | None = Field(default=None, ge=0, le=1)
    adaptive_success_at_3: float | None = Field(default=None, ge=0, le=1)
    adaptive_success_at_5: float | None = Field(default=None, ge=0, le=1)
    bernoulli_success_at_3: float | None = Field(default=None, ge=0, le=1)
    bernoulli_success_at_5: float | None = Field(default=None, ge=0, le=1)
    round_count: int = Field(default=0, ge=0)
    attempt_budget_k: int = Field(default=0, ge=0)
    intermediate_budget_k: int = Field(default=0, ge=0)
    memory_isolation: str = ""
    ci_low: float | None = Field(default=None, ge=0, le=1)
    ci_high: float | None = Field(default=None, ge=0, le=1)
    ci_method: str = ""
    ci_note: str = ""
    agreement_status: str
    diverges_from_original: bool = False
    divergence_reason: str
    scientific_wrong_count: int = Field(default=0, ge=0)
    protocol_failure_count: int = Field(default=0, ge=0)
    infrastructure_failure_count: int = Field(default=0, ge=0)
    claim_boundary_note: str
    claim_effect: str = "neutral_or_inconclusive"
    direct_evidence: bool
    adaptive_hard_scope_evidence: bool = False
    contextual_sota_only: bool
    claim_use: str
    source_artifact_path: str
    selected_manifest_path: str
    adaptive_scope_rationale: str = ""

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(
            value,
            {PHASE042_FINAL_EVIDENCE_SCHEMA_VERSION},
            "schema_version",
        )

    @field_validator("evidence_origin")
    @classmethod
    def validate_evidence_origin(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_EVIDENCE_ORIGINS, "evidence_origin")

    @field_validator("agreement_status")
    @classmethod
    def validate_agreement_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_AGREEMENT_STATUSES, "agreement_status")

    @field_validator("claim_use")
    @classmethod
    def validate_claim_use(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_CLAIM_USES, "claim_use")

    @field_validator("claim_effect")
    @classmethod
    def validate_claim_effect(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_CLAIM_EFFECTS, "claim_effect")

    @model_validator(mode="after")
    def validate_final_contract(self) -> "Phase042FinalEvidenceRow":
        _guard_phase042_downstream_row(self, "final evidence row")
        if self.agreement_status == "diverges_from_original":
            _require_text(
                self.divergence_reason,
                "divergence_reason",
                "agreement_status is diverges_from_original",
            )
        if self.diverges_from_original != (
            self.agreement_status == "diverges_from_original"
        ):
            raise ValueError("diverges_from_original must match agreement_status")
        if self.contextual_sota_only:
            if self.direct_evidence:
                raise ValueError(
                    "contextual_sota_only rows cannot set direct_evidence to true"
                )
            _require_text(
                self.claim_boundary_note,
                "claim_boundary_note",
                "contextual_sota_only is true",
            )
        return self


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
            writer.writerow(_row_to_csv_dict(row))


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


def write_phase042_validation_report(
    output_csv: Path,
    output_json: Path,
    rows: Iterable[Phase042ValidationReportRow | dict[str, Any]],
) -> tuple[Path, Path]:
    validated_rows = _validated_rows(Phase042ValidationReportRow, rows)
    write_csv(output_csv, Phase042ValidationReportRow.model_fields, validated_rows)
    write_json(output_json, PHASE042_VALIDATION_REPORT_SCHEMA_VERSION, validated_rows)
    return output_csv, output_json


def write_phase042_selected_manifest(
    output_csv: Path,
    output_json: Path,
    rows: Iterable[Phase042SelectedManifestRow | dict[str, Any]],
) -> tuple[Path, Path]:
    validated_rows = _validated_rows(Phase042SelectedManifestRow, rows)
    write_csv(output_csv, Phase042SelectedManifestRow.model_fields, validated_rows)
    write_json(output_json, PHASE042_SELECTED_MANIFEST_SCHEMA_VERSION, validated_rows)
    return output_csv, output_json
