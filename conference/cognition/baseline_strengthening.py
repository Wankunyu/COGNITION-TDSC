import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .phase4_artifacts import (
    BaselineComparisonRow,
    BaselineCoverageRow,
    ExternalImportValidationRow,
    PaperBaselineRow,
    write_baseline_comparison,
    write_baseline_coverage,
    write_external_import_validation,
    write_paper_baseline_table,
)
from .revision_artifacts import revision_run_dir


NAMED_BASELINE_SYSTEMS = {"Halligan", "Oedipus"}
MAX_SECONDARY_SYSTEMS = 2
DEFAULT_SOURCE_METADATA = Path("baseline_sources/phase4_baseline_sources.json")
DIRECTLY_COMPARABLE_STATUSES = {"direct-run", "adapter-run"}
BLOCKING_CAVEAT_TAGS = {
    "metric-mismatch",
    "dataset-mismatch",
    "threat-model-mismatch",
    "artifact-unavailable",
    "license-unclear",
}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_table(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    if path.suffix.lower() == ".json":
        payload = _read_json(path)
        if isinstance(payload, dict):
            rows = payload.get("rows", [])
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError(f"Expected rows array in {path}")
        return [dict(row) for row in rows if isinstance(row, dict)]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(str(value)))


def _as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(str(value))


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            payload = json.loads(raw)
            if isinstance(payload, list):
                return [str(item) for item in payload if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in raw.replace("|", ",").split(",") if part.strip()]


def _source_key(system_name: str, external_task_label: str) -> str:
    return f"{system_name}::{external_task_label}"


def _coverage_model_payload(raw_row: dict[str, object], run_id: str) -> dict[str, object]:
    payload = dict(raw_row)
    payload.pop("selection_reason", None)
    payload["run_id"] = run_id
    for field_name in ("captcha_families", "caveat_tags", "checked_sources", "missing_items"):
        payload[field_name] = _as_list(payload.get(field_name))
    payload["source_year"] = _as_int(payload.get("source_year"))
    payload["reported_metric_value"] = _as_float(payload.get("reported_metric_value"))
    return payload


def _secondary_systems(raw_rows: list[dict[str, object]]) -> set[str]:
    return {
        str(row.get("system_name") or "")
        for row in raw_rows
        if str(row.get("system_name") or "") not in NAMED_BASELINE_SYSTEMS
    }


def _validate_secondary_systems(raw_rows: list[dict[str, object]]) -> None:
    secondary_systems = _secondary_systems(raw_rows)
    if len(secondary_systems) > MAX_SECONDARY_SYSTEMS:
        raise ValueError("coverage metadata may include at most two additional systems")
    for row in raw_rows:
        system_name = str(row.get("system_name") or "")
        if system_name in NAMED_BASELINE_SYSTEMS:
            continue
        selection_reason = str(row.get("selection_reason") or "")
        if not any(named in selection_reason for named in NAMED_BASELINE_SYSTEMS):
            raise ValueError(
                f"{system_name} selection_reason must tie the row to Halligan or Oedipus"
            )


def validate_coverage_rows(
    rows: list[BaselineCoverageRow | dict[str, object]],
    run_id: str,
) -> list[BaselineCoverageRow]:
    raw_rows = [
        row.model_dump(mode="json") if isinstance(row, BaselineCoverageRow) else dict(row)
        for row in rows
    ]
    system_names = {str(row.get("system_name") or "") for row in raw_rows}
    if not NAMED_BASELINE_SYSTEMS <= system_names:
        raise ValueError("coverage metadata must include Halligan and Oedipus")
    _validate_secondary_systems(raw_rows)
    return [
        BaselineCoverageRow.model_validate(_coverage_model_payload(row, run_id))
        for row in raw_rows
    ]


def load_baseline_coverage_sources(path: Path, run_id: str) -> list[BaselineCoverageRow]:
    return validate_coverage_rows(_read_table(path), run_id=run_id)


def _load_baseline_coverage_artifact(path: Path, run_id: str) -> list[BaselineCoverageRow]:
    rows = _read_table(path)
    system_names = {str(row.get("system_name") or "") for row in rows}
    if not NAMED_BASELINE_SYSTEMS <= system_names:
        raise ValueError("coverage artifact must include Halligan and Oedipus")
    if len(_secondary_systems(rows)) > MAX_SECONDARY_SYSTEMS:
        raise ValueError("coverage artifact may include at most two additional systems")
    return [
        BaselineCoverageRow.model_validate(_coverage_model_payload(row, run_id))
        for row in rows
    ]


def load_external_import_rows(path: Path) -> list[dict[str, object]]:
    return _read_table(path)


def load_import_validation_rows(path: Path) -> list[ExternalImportValidationRow]:
    return [
        ExternalImportValidationRow.model_validate(row)
        for row in _read_table(path)
    ]


def _validation_status(condition: bool) -> str:
    return "pass" if condition else "fail"


def _coverage_index(
    rows: list[BaselineCoverageRow],
) -> dict[str, BaselineCoverageRow]:
    return {
        _source_key(row.system_name, row.external_task_label): row
        for row in rows
    }


def _validation_row_from_import(
    raw_row: dict[str, object],
    coverage_by_key: dict[str, BaselineCoverageRow],
    run_id: str,
) -> ExternalImportValidationRow:
    system_name = str(raw_row.get("system_name") or "")
    external_task_label = str(raw_row.get("external_task_label") or "")
    source_key = str(raw_row.get("source_key") or "") or _source_key(
        system_name, external_task_label
    )
    coverage_row = coverage_by_key.get(source_key)
    mapped_local_task_type = str(raw_row.get("mapped_local_task_type") or "")
    sample_count = _as_int(raw_row.get("sample_count"))
    reported_metric_name = str(raw_row.get("reported_metric_name") or "")
    reported_metric_value = _as_float(raw_row.get("reported_metric_value"))
    reported_metric_unit = str(raw_row.get("reported_metric_unit") or "")
    metric_definition = str(raw_row.get("metric_definition") or "")
    artifact_license = str(raw_row.get("artifact_license") or raw_row.get("license") or "")
    data_use_constraints = str(raw_row.get("data_use_constraints") or "")
    assumptions = str(raw_row.get("comparability_assumptions") or "")

    required_fields_ok = all(
        [
            system_name,
            source_key,
            external_task_label,
            mapped_local_task_type,
            reported_metric_name,
            reported_metric_unit,
        ]
    )
    metric_ok = bool(metric_definition and reported_metric_name and reported_metric_unit)
    task_label_ok = (
        coverage_row is not None
        and coverage_row.external_task_label == external_task_label
        and coverage_row.mapped_local_task_type == mapped_local_task_type
    )
    sample_count_ok = sample_count > 0
    artifact_license_ok = bool(artifact_license.strip())
    data_use_ok = bool(data_use_constraints.strip())
    comparability_ok = all(
        [
            assumptions.strip(),
            metric_ok,
            task_label_ok,
            sample_count_ok,
            artifact_license_ok,
            data_use_ok,
        ]
    )
    validation_ok = all(
        [
            required_fields_ok,
            metric_ok,
            task_label_ok,
            sample_count_ok,
            artifact_license_ok,
            data_use_ok,
            comparability_ok,
        ]
    )
    normalized_success_rate = (
        reported_metric_value
        if validation_ok
        and reported_metric_name == "success_rate"
        and reported_metric_unit == "rate"
        else None
    )
    notes = []
    if coverage_row is None:
        notes.append("no matching coverage row")
    if not validation_ok:
        notes.append("one or more import validation checks failed")
    if not notes:
        notes.append("validated offline import row")

    return ExternalImportValidationRow(
        run_id=run_id,
        system_name=system_name,
        source_key=source_key,
        external_task_label=external_task_label,
        mapped_local_task_type=mapped_local_task_type,
        required_fields_status=_validation_status(required_fields_ok),
        metric_definition_status=_validation_status(metric_ok),
        task_label_status=_validation_status(task_label_ok),
        sample_count_status=_validation_status(sample_count_ok),
        artifact_license_status=_validation_status(artifact_license_ok),
        data_use_status=_validation_status(data_use_ok),
        comparability_status=_validation_status(comparability_ok),
        validation_status=_validation_status(validation_ok),
        sample_count=sample_count,
        reported_metric_name=reported_metric_name,
        reported_metric_value=reported_metric_value,
        reported_metric_unit=reported_metric_unit,
        normalized_success_rate=normalized_success_rate,
        diagnostic_notes="; ".join(notes),
        user_confirmed_replacement=_as_bool(raw_row.get("user_confirmed_replacement")),
    )


def build_external_import_validation_rows(
    coverage_rows: list[BaselineCoverageRow],
    import_rows: list[dict[str, object]],
    run_id: str,
) -> list[ExternalImportValidationRow]:
    coverage_by_key = _coverage_index(coverage_rows)
    rows = [
        _validation_row_from_import(raw_row, coverage_by_key, run_id)
        for raw_row in import_rows
    ]
    named_pass = any(
        row.system_name in NAMED_BASELINE_SYSTEMS and row.validation_status == "pass"
        for row in rows
    )
    secondary_rows = [row for row in rows if row.system_name not in NAMED_BASELINE_SYSTEMS]
    if secondary_rows and not named_pass:
        unconfirmed = [row for row in secondary_rows if not row.user_confirmed_replacement]
        if unconfirmed:
            raise ValueError(
                "secondary smoke replacement requires user confirmation when no "
                "Halligan/Oedipus import row validates"
            )
    return rows


def _all_pass(row: ExternalImportValidationRow | None) -> bool:
    if row is None:
        return False
    return all(
        getattr(row, field_name) == "pass"
        for field_name in (
            "required_fields_status",
            "metric_definition_status",
            "task_label_status",
            "sample_count_status",
            "artifact_license_status",
            "data_use_status",
            "comparability_status",
            "validation_status",
        )
    )


def _unique_import_index(
    rows: list[ExternalImportValidationRow],
) -> dict[str, ExternalImportValidationRow]:
    indexed: dict[str, ExternalImportValidationRow] = {}
    for row in rows:
        if row.source_key in indexed:
            raise ValueError(f"duplicate import validation source_key: {row.source_key}")
        indexed[row.source_key] = row
    return indexed


def _comparison_caveat(
    coverage_row: BaselineCoverageRow,
    import_row: ExternalImportValidationRow | None,
    directly_comparable: bool,
) -> str:
    if directly_comparable:
        return ""
    reasons: list[str] = []
    if coverage_row.primary_status == "literature-only":
        reasons.append("literature-only evidence is not a direct head-to-head result")
    elif coverage_row.primary_status == "approximate":
        reasons.append("approximate evidence preserves reported metrics without direct parity")
    elif coverage_row.primary_status in {"unavailable", "incompatible"}:
        reasons.append(f"{coverage_row.primary_status} evidence requires caveated reporting")
    if coverage_row.caveat_tags:
        reasons.append(f"caveats: {', '.join(coverage_row.caveat_tags)}")
    if import_row is None:
        reasons.append("no validated import diagnostic row")
    elif not _all_pass(import_row):
        reasons.append("one or more import validation checks did not pass")
    return "; ".join(reasons)


def _comparison_basis(
    coverage_row: BaselineCoverageRow,
    import_row: ExternalImportValidationRow | None,
    directly_comparable: bool,
) -> str:
    if directly_comparable:
        return "validated import row with matching metric, sample, task, and use terms"
    if import_row is not None:
        return "validated import diagnostic present but not directly comparable"
    if coverage_row.primary_status in {"literature-only", "approximate"}:
        return "literature-only approximate context"
    return f"{coverage_row.primary_status} coverage evidence"


def build_baseline_comparison_rows(
    coverage_rows: list[BaselineCoverageRow],
    import_validation_rows: list[ExternalImportValidationRow],
    run_id: str,
) -> list[BaselineComparisonRow]:
    import_by_key = _unique_import_index(import_validation_rows)
    rows: list[BaselineComparisonRow] = []
    for coverage_row in coverage_rows:
        key = _source_key(coverage_row.system_name, coverage_row.external_task_label)
        import_row = import_by_key.get(key)
        blocking_caveats = set(coverage_row.caveat_tags) & BLOCKING_CAVEAT_TAGS
        directly_comparable = (
            coverage_row.primary_status in DIRECTLY_COMPARABLE_STATUSES
            and not blocking_caveats
            and _all_pass(import_row)
        )
        normalized_success_rate = (
            import_row.normalized_success_rate
            if directly_comparable and import_row is not None
            else None
        )
        metric_definition_status = (
            import_row.metric_definition_status if import_row else "not_applicable"
        )
        sample_count_status = import_row.sample_count_status if import_row else "not_applicable"
        comparability_status = import_row.comparability_status if import_row else "not_applicable"
        rows.append(
            BaselineComparisonRow(
                run_id=run_id,
                system_name=coverage_row.system_name,
                source_key=key,
                system_class=coverage_row.system_class,
                evidence_source_type=coverage_row.evidence_source_type,
                primary_status=coverage_row.primary_status,
                caveat_tags=coverage_row.caveat_tags,
                reported_metric_name=coverage_row.reported_metric_name,
                reported_metric_value=coverage_row.reported_metric_value,
                reported_metric_unit=coverage_row.reported_metric_unit,
                normalized_success_rate=normalized_success_rate,
                metric_definition_status=metric_definition_status,
                sample_count_status=sample_count_status,
                comparability_status=comparability_status,
                directly_comparable=directly_comparable,
                comparability_caveat=_comparison_caveat(
                    coverage_row,
                    import_row,
                    directly_comparable,
                ),
                comparability_note=(
                    ""
                    if directly_comparable
                    else "Approximate/contextual evidence only; do not read as direct parity."
                ),
                comparison_basis=_comparison_basis(
                    coverage_row,
                    import_row,
                    directly_comparable,
                ),
                source_url=coverage_row.source_url,
            )
        )
    return rows


def _reported_metric_display(row: BaselineComparisonRow) -> str:
    if row.reported_metric_value is None:
        return f"{row.reported_metric_name}: not reported"
    return (
        f"{row.reported_metric_name}={row.reported_metric_value} "
        f"{row.reported_metric_unit}"
    )


def build_paper_baseline_rows(
    comparison_rows: list[BaselineComparisonRow],
    run_id: str,
) -> list[PaperBaselineRow]:
    rows: list[PaperBaselineRow] = []
    for row in comparison_rows:
        rows.append(
            PaperBaselineRow(
                run_id=run_id,
                system_name=row.system_name,
                system_class=row.system_class,
                primary_status=row.primary_status,
                reported_metric_display=_reported_metric_display(row),
                reported_metric_name=row.reported_metric_name,
                reported_metric_value=row.reported_metric_value,
                reported_metric_unit=row.reported_metric_unit,
                normalized_success_rate=row.normalized_success_rate,
                directly_comparable=row.directly_comparable,
                comparability_caveat=row.comparability_caveat,
                comparability_note=(
                    ""
                    if row.directly_comparable
                    else row.comparability_note or row.comparability_caveat
                ),
                caveat_tags=row.caveat_tags,
                source_note=row.source_url,
                paper_table_note=row.comparison_basis,
            )
        )
    return rows


def _counts_by(rows: list[PaperBaselineRow], field_name: str) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(row, field_name)) for row in rows).items()))


def _format_counts(title: str, counts: dict[str, int]) -> str:
    if not counts:
        return f"{title}: none"
    return "\n".join(f"- {key}: {value}" for key, value in counts.items())


def render_baseline_notes(
    final_rows: list[PaperBaselineRow],
    import_validation_rows: list[ExternalImportValidationRow] | None = None,
) -> str:
    status_counts = _counts_by(final_rows, "primary_status")
    non_comparable_rows = [row for row in final_rows if not row.directly_comparable]
    unavailable_or_incompatible = [
        row
        for row in final_rows
        if row.primary_status in {"unavailable", "incompatible"}
    ]
    approximate_rows = [
        row
        for row in non_comparable_rows
        if row.primary_status in {"literature-only", "approximate"}
        or row.normalized_success_rate is None
    ]
    import_counts = (
        dict(
            sorted(
                Counter(row.validation_status for row in import_validation_rows or []).items()
            )
        )
        if import_validation_rows
        else {}
    )
    lines = [
        "# Phase 4 Baseline Comparison Notes",
        "",
        "## Status Counts",
        _format_counts("Primary-status row counts", status_counts),
        _format_counts("Import-validation row counts", import_counts),
        "",
        "## Unavailable And Incompatible Evidence",
        *(
            [
                f"- {row.system_name}: {row.primary_status}; {row.comparability_note}"
                for row in unavailable_or_incompatible
            ]
            or ["None."]
        ),
        "",
        "## Non-Comparable Rows",
        f"Non-comparable rows: {len(non_comparable_rows)}",
        *[
            f"- {row.system_name} ({row.primary_status}): {row.comparability_note}"
            for row in non_comparable_rows
        ],
        "",
        "## Approximate Comparison Basis",
        (
            "Approximate comparisons preserve reported metrics and mark "
            "directly_comparable=false unless validation proves metric, sample, task, "
            "license/data-use, and threat-model parity."
        ),
        *[
            f"- {row.system_name}: {row.reported_metric_display}; {row.paper_table_note}"
            for row in approximate_rows
        ],
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_baseline_notes(
    final_rows: list[PaperBaselineRow],
    output_md: Path,
    import_validation_rows: list[ExternalImportValidationRow] | None = None,
) -> Path:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        render_baseline_notes(final_rows, import_validation_rows),
        encoding="utf-8",
    )
    return output_md


def _load_paper_baseline_rows(path: Path) -> list[PaperBaselineRow]:
    return [PaperBaselineRow.model_validate(row) for row in _read_table(path)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Phase 4 baseline coverage and import-validation artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    coverage = subparsers.add_parser(
        "coverage",
        help="Validate baseline source metadata and write a coverage matrix.",
    )
    coverage.add_argument("--source-metadata", default=str(DEFAULT_SOURCE_METADATA))
    coverage.add_argument("--output-root", default="./results/local_runs")
    coverage.add_argument("--run-id", required=True)
    coverage.add_argument("--output-csv", default=None)
    coverage.add_argument("--output-json", default=None)

    validate_import = subparsers.add_parser(
        "validate-import",
        help="Validate external/imported baseline smoke rows.",
    )
    validate_import.add_argument("--coverage-json", required=True)
    validate_import.add_argument("--import-rows", required=True)
    validate_import.add_argument("--output-root", default="./results/local_runs")
    validate_import.add_argument("--run-id", required=True)
    validate_import.add_argument("--output-csv", default=None)
    validate_import.add_argument("--output-json", default=None)

    build_table = subparsers.add_parser(
        "build-table",
        help="Build baseline comparison and paper-table artifacts.",
    )
    build_table.add_argument("--coverage-json", required=True)
    build_table.add_argument("--import-validation-json", required=True)
    build_table.add_argument("--output-root", default="./results/local_runs")
    build_table.add_argument("--run-id", required=True)
    build_table.add_argument("--comparison-csv", default=None)
    build_table.add_argument("--comparison-json", default=None)
    build_table.add_argument("--paper-table-csv", default=None)
    build_table.add_argument("--paper-table-json", default=None)

    notes = subparsers.add_parser(
        "notes",
        help="Render concise reported notes for baseline table artifacts.",
    )
    notes.add_argument("--paper-table-json", required=True)
    notes.add_argument("--import-validation-json", default=None)
    notes.add_argument("--output-root", default="./results/local_runs")
    notes.add_argument("--run-id", required=True)
    notes.add_argument("--output-md", default=None)
    return parser


def _path_or_default(raw_path: str | None, default_path: Path) -> Path:
    return Path(raw_path) if raw_path else default_path


def _run_coverage(args: argparse.Namespace) -> dict[str, object]:
    run_dir = revision_run_dir(args.output_root, args.run_id)
    output_csv = _path_or_default(args.output_csv, run_dir / "coverage_matrix.csv")
    output_json = _path_or_default(args.output_json, run_dir / "coverage_matrix.json")
    rows = load_baseline_coverage_sources(Path(args.source_metadata), run_id=args.run_id)
    write_baseline_coverage(output_csv, output_json, rows)
    return {
        "row_count": len(rows),
        "output_csv": str(output_csv),
        "output_json": str(output_json),
    }


def _run_validate_import(args: argparse.Namespace) -> dict[str, object]:
    run_dir = revision_run_dir(args.output_root, args.run_id)
    output_csv = _path_or_default(args.output_csv, run_dir / "import_diagnostics.csv")
    output_json = _path_or_default(args.output_json, run_dir / "import_diagnostics.json")
    coverage_rows = _load_baseline_coverage_artifact(
        Path(args.coverage_json),
        run_id=args.run_id,
    )
    import_rows = load_external_import_rows(Path(args.import_rows))
    rows = build_external_import_validation_rows(
        coverage_rows,
        import_rows,
        run_id=args.run_id,
    )
    write_external_import_validation(output_csv, output_json, rows)
    return {
        "row_count": len(rows),
        "output_csv": str(output_csv),
        "output_json": str(output_json),
    }


def _run_build_table(args: argparse.Namespace) -> dict[str, object]:
    run_dir = revision_run_dir(args.output_root, args.run_id)
    comparison_csv = _path_or_default(
        args.comparison_csv,
        run_dir / "baseline_comparison.csv",
    )
    comparison_json = _path_or_default(
        args.comparison_json,
        run_dir / "baseline_comparison.json",
    )
    paper_table_csv = _path_or_default(
        args.paper_table_csv,
        run_dir / "paper_baseline_table.csv",
    )
    paper_table_json = _path_or_default(
        args.paper_table_json,
        run_dir / "paper_baseline_table.json",
    )
    coverage_rows = _load_baseline_coverage_artifact(
        Path(args.coverage_json),
        run_id=args.run_id,
    )
    import_rows = load_import_validation_rows(Path(args.import_validation_json))
    comparison_rows = build_baseline_comparison_rows(
        coverage_rows,
        import_rows,
        run_id=args.run_id,
    )
    final_rows = build_paper_baseline_rows(comparison_rows, run_id=args.run_id)
    write_baseline_comparison(comparison_csv, comparison_json, comparison_rows)
    write_paper_baseline_table(paper_table_csv, paper_table_json, final_rows)
    return {
        "comparison_row_count": len(comparison_rows),
        "final_row_count": len(final_rows),
        "comparison_csv": str(comparison_csv),
        "comparison_json": str(comparison_json),
        "paper_table_csv": str(paper_table_csv),
        "paper_table_json": str(paper_table_json),
    }


def _run_notes(args: argparse.Namespace) -> dict[str, object]:
    run_dir = revision_run_dir(args.output_root, args.run_id)
    output_md = _path_or_default(args.output_md, run_dir / "baseline_notes.md")
    final_rows = _load_paper_baseline_rows(Path(args.paper_table_json))
    import_rows = (
        load_import_validation_rows(Path(args.import_validation_json))
        if args.import_validation_json
        else []
    )
    write_baseline_notes(final_rows, output_md, import_rows)
    return {
        "final_row_count": len(final_rows),
        "output_md": str(output_md),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "coverage":
            summary = _run_coverage(args)
        elif args.command == "validate-import":
            summary = _run_validate_import(args)
        elif args.command == "build-table":
            summary = _run_build_table(args)
        elif args.command == "notes":
            summary = _run_notes(args)
        else:
            parser.error(f"unknown command: {args.command}")
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
