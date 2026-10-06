import csv
import json
from pathlib import Path
from typing import Any, Iterable, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


BASELINE_COVERAGE_SCHEMA_VERSION = "cognition.revision.baseline_coverage.v1"
EXTERNAL_IMPORT_VALIDATION_SCHEMA_VERSION = (
    "cognition.revision.external_import_validation.v1"
)
BASELINE_COMPARISON_SCHEMA_VERSION = "cognition.revision.baseline_comparison.v1"
PAPER_BASELINE_TABLE_SCHEMA_VERSION = "cognition.revision.paper_baseline_table.v1"

ALLOWED_PRIMARY_STATUSES = {
    "direct-run",
    "adapter-run",
    "literature-only",
    "approximate",
    "incompatible",
    "unavailable",
}
ALLOWED_CAVEAT_TAGS = {
    "metric-mismatch",
    "dataset-mismatch",
    "threat-model-mismatch",
    "artifact-unavailable",
    "license-unclear",
}
ALLOWED_SYSTEM_CLASSES = {
    "off_the_shelf_mllm_api",
    "specialized_solver",
    "benchmark_dataset",
    "hybrid_or_unknown",
}
ALLOWED_VALIDATION_STATUSES = {"pass", "warning", "fail", "not_applicable"}

_RUNNABLE_STATUSES = {"direct-run", "adapter-run"}
_AUDIT_REQUIRED_STATUSES = {"unavailable", "incompatible"}

RowModel = TypeVar("RowModel", bound=BaseModel)


def _validate_allowed(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return value


def _validate_caveat_tags(values: list[str]) -> list[str]:
    for value in values:
        _validate_allowed(value, ALLOWED_CAVEAT_TAGS, "caveat_tags")
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


def _require_text(value: str, field_name: str, status: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} is required when primary_status is {status}")


def _require_items(values: list[str], field_name: str, status: str) -> None:
    if not values:
        raise ValueError(f"{field_name} is required when primary_status is {status}")
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} cannot contain blank items")


def _validated_rows(
    model: type[RowModel],
    rows: Iterable[RowModel | dict[str, Any]],
) -> list[RowModel]:
    return [row if isinstance(row, model) else model.model_validate(row) for row in rows]


class BaselineCoverageRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = BASELINE_COVERAGE_SCHEMA_VERSION
    run_id: str
    system_name: str
    system_class: str
    evidence_source_type: str
    source_url: str
    source_year: int = Field(ge=0)
    solver_architecture: str
    threat_model: str
    dataset_scale: str
    captcha_families: list[str]
    external_task_label: str
    mapped_local_task_type: str
    mapped_local_family: str
    mapping_confidence: str
    new_or_supplemental_category_reason: str
    reported_metric_name: str
    reported_metric_value: float | None = None
    reported_metric_unit: str
    artifact_availability: str
    license: str
    data_use_constraints: str
    latency_coverage: str
    cost_coverage: str
    failure_mode_analysis: str
    defense_methodology_relevance: str
    primary_status: str
    caveat_tags: list[str]
    status_reason: str
    checked_sources: list[str]
    missing_items: list[str]
    last_checked_date: str

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(value, {BASELINE_COVERAGE_SCHEMA_VERSION}, "schema_version")

    @field_validator("primary_status")
    @classmethod
    def validate_primary_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_PRIMARY_STATUSES, "primary_status")

    @field_validator("system_class")
    @classmethod
    def validate_system_class(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_SYSTEM_CLASSES, "system_class")

    @field_validator("caveat_tags")
    @classmethod
    def validate_caveat_tags(cls, values: list[str]) -> list[str]:
        return _validate_caveat_tags(values)

    @model_validator(mode="after")
    def validate_status_contracts(self) -> "BaselineCoverageRow":
        if self.primary_status in _AUDIT_REQUIRED_STATUSES:
            _require_text(self.status_reason, "status_reason", self.primary_status)
            _require_items(self.checked_sources, "checked_sources", self.primary_status)
            _require_items(self.missing_items, "missing_items", self.primary_status)
            _require_text(self.last_checked_date, "last_checked_date", self.primary_status)
        if self.primary_status in _RUNNABLE_STATUSES:
            _require_text(self.license, "license", self.primary_status)
            _require_text(
                self.data_use_constraints,
                "data_use_constraints",
                self.primary_status,
            )
        return self


class ExternalImportValidationRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = EXTERNAL_IMPORT_VALIDATION_SCHEMA_VERSION
    run_id: str
    system_name: str
    source_key: str
    external_task_label: str
    mapped_local_task_type: str
    required_fields_status: str
    metric_definition_status: str
    task_label_status: str
    sample_count_status: str
    artifact_license_status: str
    data_use_status: str
    comparability_status: str
    validation_status: str
    sample_count: int = Field(ge=0)
    reported_metric_name: str
    reported_metric_value: float | None = None
    reported_metric_unit: str
    normalized_success_rate: float | None = Field(default=None, ge=0, le=1)
    diagnostic_notes: str
    user_confirmed_replacement: bool = False

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(
            value,
            {EXTERNAL_IMPORT_VALIDATION_SCHEMA_VERSION},
            "schema_version",
        )

    @field_validator(
        "required_fields_status",
        "metric_definition_status",
        "task_label_status",
        "sample_count_status",
        "artifact_license_status",
        "data_use_status",
        "comparability_status",
        "validation_status",
    )
    @classmethod
    def validate_validation_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_VALIDATION_STATUSES, "validation_status")


class BaselineComparisonRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = BASELINE_COMPARISON_SCHEMA_VERSION
    run_id: str
    system_name: str
    source_key: str
    system_class: str
    evidence_source_type: str
    primary_status: str
    caveat_tags: list[str]
    reported_metric_name: str
    reported_metric_value: float | None = None
    reported_metric_unit: str
    normalized_success_rate: float | None = Field(default=None, ge=0, le=1)
    metric_definition_status: str
    sample_count_status: str
    comparability_status: str
    directly_comparable: bool
    comparability_caveat: str
    comparability_note: str
    comparison_basis: str
    source_url: str

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(value, {BASELINE_COMPARISON_SCHEMA_VERSION}, "schema_version")

    @field_validator("primary_status")
    @classmethod
    def validate_primary_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_PRIMARY_STATUSES, "primary_status")

    @field_validator("system_class")
    @classmethod
    def validate_system_class(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_SYSTEM_CLASSES, "system_class")

    @field_validator("caveat_tags")
    @classmethod
    def validate_caveat_tags(cls, values: list[str]) -> list[str]:
        return _validate_caveat_tags(values)

    @field_validator(
        "metric_definition_status",
        "sample_count_status",
        "comparability_status",
    )
    @classmethod
    def validate_validation_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_VALIDATION_STATUSES, "validation_status")

    @model_validator(mode="after")
    def validate_comparability_contract(self) -> "BaselineComparisonRow":
        if (
            not self.directly_comparable
            and not self.comparability_caveat.strip()
            and not self.comparability_note.strip()
        ):
            raise ValueError(
                "comparability_caveat or comparability_note is required when "
                "directly_comparable is false"
            )
        return self


class PaperBaselineRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PAPER_BASELINE_TABLE_SCHEMA_VERSION
    run_id: str
    system_name: str
    system_class: str
    primary_status: str
    reported_metric_display: str
    reported_metric_name: str
    reported_metric_value: float | None = None
    reported_metric_unit: str
    normalized_success_rate: float | None = Field(default=None, ge=0, le=1)
    directly_comparable: bool
    comparability_caveat: str
    comparability_note: str
    caveat_tags: list[str]
    source_note: str
    paper_table_note: str

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        return _validate_allowed(
            value,
            {PAPER_BASELINE_TABLE_SCHEMA_VERSION},
            "schema_version",
        )

    @field_validator("primary_status")
    @classmethod
    def validate_primary_status(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_PRIMARY_STATUSES, "primary_status")

    @field_validator("system_class")
    @classmethod
    def validate_system_class(cls, value: str) -> str:
        return _validate_allowed(value, ALLOWED_SYSTEM_CLASSES, "system_class")

    @field_validator("caveat_tags")
    @classmethod
    def validate_caveat_tags(cls, values: list[str]) -> list[str]:
        return _validate_caveat_tags(values)

    @model_validator(mode="after")
    def validate_visible_comparability_note(self) -> "PaperBaselineRow":
        if (
            not self.directly_comparable
            and not self.comparability_caveat.strip()
            and not self.comparability_note.strip()
        ):
            raise ValueError(
                "comparability_caveat or comparability_note is required when "
                "directly_comparable is false"
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


def write_baseline_coverage(
    output_csv: Path,
    output_json: Path,
    rows: Iterable[BaselineCoverageRow | dict[str, Any]],
) -> tuple[Path, Path]:
    validated_rows = _validated_rows(BaselineCoverageRow, rows)
    write_csv(output_csv, BaselineCoverageRow.model_fields, validated_rows)
    write_json(output_json, BASELINE_COVERAGE_SCHEMA_VERSION, validated_rows)
    return output_csv, output_json


def write_external_import_validation(
    output_csv: Path,
    output_json: Path,
    rows: Iterable[ExternalImportValidationRow | dict[str, Any]],
) -> tuple[Path, Path]:
    validated_rows = _validated_rows(ExternalImportValidationRow, rows)
    write_csv(output_csv, ExternalImportValidationRow.model_fields, validated_rows)
    write_json(output_json, EXTERNAL_IMPORT_VALIDATION_SCHEMA_VERSION, validated_rows)
    return output_csv, output_json


def write_baseline_comparison(
    output_csv: Path,
    output_json: Path,
    rows: Iterable[BaselineComparisonRow | dict[str, Any]],
) -> tuple[Path, Path]:
    validated_rows = _validated_rows(BaselineComparisonRow, rows)
    write_csv(output_csv, BaselineComparisonRow.model_fields, validated_rows)
    write_json(output_json, BASELINE_COMPARISON_SCHEMA_VERSION, validated_rows)
    return output_csv, output_json


def write_paper_baseline_table(
    output_csv: Path,
    output_json: Path,
    rows: Iterable[PaperBaselineRow | dict[str, Any]],
) -> tuple[Path, Path]:
    validated_rows = _validated_rows(PaperBaselineRow, rows)
    write_csv(output_csv, PaperBaselineRow.model_fields, validated_rows)
    write_json(output_json, PAPER_BASELINE_TABLE_SCHEMA_VERSION, validated_rows)
    return output_csv, output_json
