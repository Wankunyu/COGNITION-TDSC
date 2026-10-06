from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pandas as pd

from .phase3_artifacts import (
    PASS_RATE_CONFIDENCE_SCHEMA_VERSION,
    THRESHOLD_SENSITIVITY_SCHEMA_VERSION,
    PassRateConfidenceRow,
    ThresholdSensitivityRow,
    write_csv,
    write_json,
)
from .revision_artifacts import revision_run_dir
from .visualize_results import CAPTCHAVisualizer


EXPERIMENTS = ("exp1", "exp2", "exp3", "exp4")
CUTOFF_NOTE = (
    "40% working CAPTCHA threshold; operational reporting heuristic, not a universal "
    "CAPTCHA security boundary. 30%-50% is a revision-time caution band, not a new "
    "security tier."
)
PASS_RATE_COLUMNS = [
    "experiment",
    "provider",
    "model",
    "provider_model",
    "task_type",
    "task_family",
    "n_attempts",
    "n_success",
    "pass_rate",
    "source_path",
]
TREND_COLUMNS = [
    "provider",
    "model",
    "provider_model",
    "task_type",
    "task_family",
    "pass_rate",
    "source",
]
TREND_SOURCE_ORDER = (
    "exp1",
    "exp2",
    "exp3",
    "exp4",
    "adaptive_summary",
    "adaptive_comparison",
    "extended_validation_slice",
)


def wilson_interval(
    n_success: int,
    n_attempts: int,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    if n_attempts == 0:
        return None, None
    if n_attempts < 0:
        raise ValueError("n_attempts must be non-negative")
    if n_success < 0 or n_success > n_attempts:
        raise ValueError("n_success must be between 0 and n_attempts")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    n = float(n_attempts)
    p_hat = n_success / n
    z_squared = z**2
    denominator = 1.0 + z_squared / n
    center = (p_hat + z_squared / (2.0 * n)) / denominator
    half_width = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) + z_squared / (4.0 * n)) / n)
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def load_experiment_pass_rates(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = Path(results_dir)
    for experiment in EXPERIMENTS:
        for csv_file in sorted((root / experiment).glob("**/results.csv")):
            df = pd.read_csv(csv_file)
            rows.extend(_normalize_result_file(df, csv_file, root, experiment))
    if not rows:
        return pd.DataFrame(columns=PASS_RATE_COLUMNS)
    return pd.DataFrame(rows, columns=PASS_RATE_COLUMNS).where(pd.notna(pd.DataFrame(rows)), None)


def build_pass_rate_confidence_rows(
    df: pd.DataFrame,
    run_id: str,
    underpowered_n: int = 20,
    confidence: float = 0.95,
) -> list[PassRateConfidenceRow]:
    if underpowered_n < 0:
        raise ValueError("underpowered_n must be non-negative")
    if df.empty:
        return []

    rows: list[PassRateConfidenceRow] = []
    task_groups = _aggregate_pass_rates(
        df,
        [
            "provider",
            "model",
            "provider_model",
            "experiment",
            "task_type",
            "task_family",
        ],
    )
    for record in task_groups:
        rows.append(
            _confidence_row(
                record,
                run_id=run_id,
                aggregation_level="task_type",
                underpowered_n=underpowered_n,
                confidence=confidence,
            )
        )

    family_groups = _aggregate_pass_rates(
        df,
        ["provider", "model", "provider_model", "experiment", "task_family"],
    )
    for record in family_groups:
        record["task_type"] = "__family__"
        rows.append(
            _confidence_row(
                record,
                run_id=run_id,
                aggregation_level="task_family",
                underpowered_n=underpowered_n,
                confidence=confidence,
            )
        )
    return rows


def write_pass_rate_outputs(
    rows: list[PassRateConfidenceRow],
    output_csv: Path,
    output_json: Path,
) -> tuple[Path, Path]:
    validated_rows = [
        row if isinstance(row, PassRateConfidenceRow) else PassRateConfidenceRow.model_validate(row)
        for row in rows
    ]
    csv_path = Path(output_csv)
    json_path = Path(output_json)
    write_csv(csv_path, PassRateConfidenceRow.model_fields, validated_rows)
    write_json(json_path, PASS_RATE_CONFIDENCE_SCHEMA_VERSION, validated_rows)
    return csv_path, json_path


def classify_threshold_label(
    rate: float | None,
    *,
    cutoff: float = 0.40,
    review_band_low: float = 0.30,
    review_band_high: float = 0.50,
    ci_crosses_cutoff: bool = False,
    trend_sensitive: bool = False,
) -> str | None:
    if rate is None:
        return None
    if review_band_low <= rate <= review_band_high:
        return "borderline/near-broken"
    if ci_crosses_cutoff:
        return "borderline/near-broken"
    if trend_sensitive:
        return "borderline/near-broken"
    if rate < cutoff:
        return "hard"
    return "broken"


def load_adaptive_trend_rates(
    adaptive_summary: Path | None,
    adaptive_comparison: Path | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if adaptive_summary is not None:
        summary_df = _read_table(Path(adaptive_summary))
        if not summary_df.empty and "success_rate" in summary_df.columns:
            rows.extend(
                _trend_records_from_rate_column(
                    summary_df,
                    rate_column="success_rate",
                    source="adaptive_summary",
                )
            )
    if adaptive_comparison is not None:
        comparison_df = _read_table(Path(adaptive_comparison))
        if not comparison_df.empty:
            rows.extend(_adaptive_comparison_trend_records(comparison_df))
    if not rows:
        return pd.DataFrame(columns=TREND_COLUMNS)
    return pd.DataFrame(rows, columns=TREND_COLUMNS).where(pd.notna(pd.DataFrame(rows)), None)


def load_extended_validation_trend_rates(
    extended_validation_comparison: Path | None,
) -> pd.DataFrame:
    if extended_validation_comparison is None:
        return pd.DataFrame(columns=TREND_COLUMNS)
    df = _read_table(Path(extended_validation_comparison))
    if df.empty or "validation_slice_rate" not in df.columns:
        return pd.DataFrame(columns=TREND_COLUMNS)
    rows = _trend_records_from_rate_column(
        df,
        rate_column="validation_slice_rate",
        source="extended_validation_slice",
    )
    if not rows:
        return pd.DataFrame(columns=TREND_COLUMNS)
    return pd.DataFrame(rows, columns=TREND_COLUMNS).where(pd.notna(pd.DataFrame(rows)), None)


def build_threshold_sensitivity_rows(
    confidence_rows: list[PassRateConfidenceRow],
    run_id: str,
    adaptive_trend_rates: pd.DataFrame | None = None,
    extended_validation_trend_rates: pd.DataFrame | None = None,
    cutoff: float = 0.40,
    review_band_low: float = 0.30,
    review_band_high: float = 0.50,
    trend_delta: float = 0.10,
) -> list[ThresholdSensitivityRow]:
    task_rows = [
        row
        for row in confidence_rows
        if row.aggregation_level == "task_type" and row.task_type != "__family__"
    ]
    if not task_rows:
        return []

    trend_df = _combined_trend_rates(
        adaptive_trend_rates,
        extended_validation_trend_rates,
    )
    rows: list[ThresholdSensitivityRow] = []
    for key, group_rows in _group_confidence_rows(task_rows).items():
        primary = _primary_confidence_row(group_rows)
        observed = _observed_rates_for_group(group_rows, trend_df)
        primary_rate = primary.pass_rate
        max_observed_rate = max([primary_rate, *[rate for _, rate in observed]])
        threshold = primary_rate + trend_delta
        trend_sources = _trend_sources(observed, threshold)
        ci_crosses_cutoff = (
            primary.ci_low is not None
            and primary.ci_high is not None
            and primary.ci_low <= cutoff <= primary.ci_high
        )
        trend_sensitive = (
            primary_rate < cutoff and max_observed_rate - primary_rate >= trend_delta
        )
        label = classify_threshold_label(
            primary_rate,
            cutoff=cutoff,
            review_band_low=review_band_low,
            review_band_high=review_band_high,
            ci_crosses_cutoff=ci_crosses_cutoff,
            trend_sensitive=trend_sensitive,
        )
        rows.append(
            ThresholdSensitivityRow(
                run_id=run_id,
                provider=key[0],
                model=key[1],
                provider_model=key[2],
                task_type=key[3],
                task_family=primary.task_family,
                primary_experiment=primary.experiment,
                primary_rate=primary_rate,
                max_observed_rate=max_observed_rate,
                label=label or "",
                margin_to_cutoff=primary_rate - cutoff,
                cutoff=cutoff,
                review_band_low=review_band_low,
                review_band_high=review_band_high,
                in_30_50_review_band=review_band_low <= primary_rate <= review_band_high,
                ci_crosses_cutoff=ci_crosses_cutoff,
                trend_sensitive=trend_sensitive,
                trend_delta=trend_delta,
                trend_sources=";".join(trend_sources),
                cutoff_note=CUTOFF_NOTE,
            )
        )
    return rows


def write_threshold_outputs(
    rows: list[ThresholdSensitivityRow],
    output_csv: Path,
    output_json: Path,
) -> tuple[Path, Path]:
    validated_rows = [
        row
        if isinstance(row, ThresholdSensitivityRow)
        else ThresholdSensitivityRow.model_validate(row)
        for row in rows
    ]
    csv_path = Path(output_csv)
    json_path = Path(output_json)
    write_csv(csv_path, ThresholdSensitivityRow.model_fields, validated_rows)
    write_json(json_path, THRESHOLD_SENSITIVITY_SCHEMA_VERSION, validated_rows)
    return csv_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Phase 3 pass-rate confidence and threshold-sensitivity artifacts."
    )
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output-root", default="./results/local_runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--underpowered-n", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--pass-rate-csv",
        default=None,
        help="Default: results/local_runs/<run_id>/pass_rate_confidence.csv",
    )
    parser.add_argument(
        "--pass-rate-json",
        default=None,
        help="Default: results/local_runs/<run_id>/pass_rate_confidence.json",
    )
    parser.add_argument(
        "--threshold-csv",
        default=None,
        help="Default: results/local_runs/<run_id>/threshold_sensitivity.csv",
    )
    parser.add_argument(
        "--threshold-json",
        default=None,
        help="Default: results/local_runs/<run_id>/threshold_sensitivity.json",
    )
    parser.add_argument("--adaptive-summary", default=None)
    parser.add_argument("--adaptive-comparison", default=None)
    parser.add_argument("--extended-validation-comparison", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_dir = revision_run_dir(args.output_root, args.run_id)
        pass_rate_csv = (
            Path(args.pass_rate_csv)
            if args.pass_rate_csv is not None
            else run_dir / "pass_rate_confidence.csv"
        )
        pass_rate_json = (
            Path(args.pass_rate_json)
            if args.pass_rate_json is not None
            else run_dir / "pass_rate_confidence.json"
        )
        threshold_csv = (
            Path(args.threshold_csv)
            if args.threshold_csv is not None
            else run_dir / "threshold_sensitivity.csv"
        )
        threshold_json = (
            Path(args.threshold_json)
            if args.threshold_json is not None
            else run_dir / "threshold_sensitivity.json"
        )
        df = load_experiment_pass_rates(Path(args.results_dir))
        rows = build_pass_rate_confidence_rows(
            df,
            run_id=args.run_id,
            underpowered_n=args.underpowered_n,
            confidence=args.confidence,
        )
        output_csv, output_json = write_pass_rate_outputs(
            rows,
            pass_rate_csv,
            pass_rate_json,
        )
        adaptive_trend_rates = load_adaptive_trend_rates(
            Path(args.adaptive_summary) if args.adaptive_summary else None,
            Path(args.adaptive_comparison) if args.adaptive_comparison else None,
        )
        extended_trend_rates = load_extended_validation_trend_rates(
            Path(args.extended_validation_comparison)
            if args.extended_validation_comparison
            else None
        )
        threshold_rows = build_threshold_sensitivity_rows(
            rows,
            run_id=args.run_id,
            adaptive_trend_rates=adaptive_trend_rates,
            extended_validation_trend_rates=extended_trend_rates,
        )
        threshold_output_csv, threshold_output_json = write_threshold_outputs(
            threshold_rows,
            threshold_csv,
            threshold_json,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "pass_rate_row_count": len(rows),
                "pass_rate_csv": str(output_csv),
                "pass_rate_json": str(output_json),
                "threshold_row_count": len(threshold_rows),
                "threshold_csv": str(threshold_output_csv),
                "threshold_json": str(threshold_output_json),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _normalize_result_file(
    df: pd.DataFrame,
    csv_file: Path,
    results_dir: Path,
    experiment: str,
) -> list[dict[str, Any]]:
    provider, model = _provider_model_from_path(csv_file, results_dir, experiment)
    if experiment == "exp3":
        return _normalize_exp3(df, csv_file, experiment, provider, model)
    return _normalize_aggregate(df, csv_file, experiment, provider, model)


def _normalize_aggregate(
    df: pd.DataFrame,
    csv_file: Path,
    experiment: str,
    provider_from_path: str,
    model_from_path: str,
) -> list[dict[str, Any]]:
    required = {"n", "pass_at_1"}
    task_column = "task_type" if "task_type" in df.columns else "type"
    if task_column not in df.columns or not required.issubset(df.columns):
        raise ValueError(f"Unsupported {experiment} result schema: {csv_file}")

    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        provider = str(record.get("provider") or provider_from_path)
        model = str(record.get("model") or model_from_path)
        task_type = str(record[task_column])
        n_attempts = _to_int(record["n"])
        pass_rate = _to_float(record["pass_at_1"])
        n_success = int(round(pass_rate * n_attempts))
        rows.append(
            _pass_rate_record(
                experiment=experiment,
                provider=provider,
                model=model,
                task_type=task_type,
                n_attempts=n_attempts,
                n_success=n_success,
                pass_rate=pass_rate,
                source_path=csv_file,
            )
        )
    return rows


def _normalize_exp3(
    df: pd.DataFrame,
    csv_file: Path,
    experiment: str,
    provider_from_path: str,
    model_from_path: str,
) -> list[dict[str, Any]]:
    required = {"kind", "pass1"}
    task_column = "task_type" if "task_type" in df.columns else "type"
    if task_column not in df.columns or not required.issubset(df.columns):
        raise ValueError(f"Unsupported exp3 result schema: {csv_file}")

    summary = df[df["kind"] == "summary"].copy()
    if summary.empty:
        return []
    if "provider" not in summary.columns:
        summary["provider"] = provider_from_path
    if "model" not in summary.columns:
        summary["model"] = model_from_path

    grouped = (
        summary.groupby(["provider", "model", task_column], as_index=False)
        .agg(n_attempts=("pass1", "count"), n_success=("pass1", "sum"))
        .to_dict(orient="records")
    )
    rows: list[dict[str, Any]] = []
    for record in grouped:
        n_attempts = _to_int(record["n_attempts"])
        n_success = _to_int(record["n_success"])
        pass_rate = n_success / n_attempts if n_attempts else 0.0
        rows.append(
            _pass_rate_record(
                experiment=experiment,
                provider=str(record["provider"]),
                model=str(record["model"]),
                task_type=str(record[task_column]),
                n_attempts=n_attempts,
                n_success=n_success,
                pass_rate=pass_rate,
                source_path=csv_file,
            )
        )
    return rows


def _pass_rate_record(
    *,
    experiment: str,
    provider: str,
    model: str,
    task_type: str,
    n_attempts: int,
    n_success: int,
    pass_rate: float,
    source_path: Path,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "provider": provider,
        "model": model,
        "provider_model": f"{provider}/{model}",
        "task_type": task_type,
        "task_family": CAPTCHAVisualizer.TASK_FAMILY.get(task_type, "Unmapped"),
        "n_attempts": n_attempts,
        "n_success": n_success,
        "pass_rate": pass_rate,
        "source_path": str(source_path),
    }


def _aggregate_pass_rates(df: pd.DataFrame, keys: list[str]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for key_values, group in df.groupby(keys, dropna=False, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        record = dict(zip(keys, key_values))
        n_attempts = int(group["n_attempts"].astype(int).sum())
        n_success = int(group["n_success"].astype(int).sum())
        record["n_attempts"] = n_attempts
        record["n_success"] = n_success
        record["pass_rate"] = n_success / n_attempts if n_attempts else 0.0
        record["source_path"] = ";".join(sorted({str(path) for path in group["source_path"]}))
        grouped.append(record)
    return grouped


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("rows", payload if isinstance(payload, list) else [])
        return pd.DataFrame(rows).where(lambda frame: pd.notna(frame), None)
    return pd.read_csv(path).where(lambda frame: pd.notna(frame), None)


def _trend_records_from_rate_column(
    df: pd.DataFrame,
    *,
    rate_column: str,
    source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if rate_column not in df.columns:
        return rows
    task_column = "task_type" if "task_type" in df.columns else "type"
    if task_column not in df.columns:
        return rows
    for record in df.to_dict(orient="records"):
        provider, model, provider_model = _provider_model_from_record(record)
        task_type = str(record[task_column])
        rows.append(
            {
                "provider": provider,
                "model": model,
                "provider_model": provider_model,
                "task_type": task_type,
                "task_family": str(
                    record.get("task_family")
                    or CAPTCHAVisualizer.TASK_FAMILY.get(task_type, "Unmapped")
                ),
                "pass_rate": _to_float_or_none(record.get(rate_column)),
                "source": source,
            }
        )
    return rows


def _adaptive_comparison_trend_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    task_column = "task_type" if "task_type" in df.columns else "type"
    if task_column not in df.columns:
        return []
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        provider, model, provider_model = _provider_model_from_record(record)
        task_type = str(record[task_column])
        rows.append(
            {
                "provider": provider,
                "model": model,
                "provider_model": provider_model,
                "task_type": task_type,
                "task_family": str(
                    record.get("task_family")
                    or CAPTCHAVisualizer.TASK_FAMILY.get(task_type, "Unmapped")
                ),
                "pass_rate": _adaptive_comparison_rate(record),
                "source": "adaptive_comparison",
            }
        )
    return rows


def _adaptive_comparison_rate(record: dict[str, Any]) -> float | None:
    direct_rate = _to_float_or_none(record.get("adaptive_success_at_k"))
    if direct_rate is not None:
        return direct_rate
    label = _to_str_or_none(record.get("adaptive_label"))
    if label is None:
        return None
    normalized = label.strip().casefold()
    if normalized == "broken":
        return 1.0
    if normalized == "hard":
        return 0.0
    return None


def _combined_trend_rates(
    adaptive_trend_rates: pd.DataFrame | None,
    extended_validation_trend_rates: pd.DataFrame | None,
) -> pd.DataFrame:
    frames = [
        frame
        for frame in (adaptive_trend_rates, extended_validation_trend_rates)
        if frame is not None and not frame.empty
    ]
    if not frames:
        return pd.DataFrame(columns=TREND_COLUMNS)
    return pd.concat(frames, ignore_index=True).where(lambda frame: pd.notna(frame), None)


def _group_confidence_rows(
    rows: list[PassRateConfidenceRow],
) -> dict[tuple[str, str, str, str], list[PassRateConfidenceRow]]:
    groups: dict[tuple[str, str, str, str], list[PassRateConfidenceRow]] = {}
    for row in rows:
        key = (row.provider, row.model, row.provider_model, row.task_type)
        groups.setdefault(key, []).append(row)
    return groups


def _primary_confidence_row(rows: list[PassRateConfidenceRow]) -> PassRateConfidenceRow:
    exp2_rows = [row for row in rows if row.experiment == "exp2"]
    if exp2_rows:
        return sorted(exp2_rows, key=lambda row: row.source_path)[0]
    return sorted(rows, key=lambda row: EXPERIMENTS.index(row.experiment))[0]


def _observed_rates_for_group(
    confidence_rows: list[PassRateConfidenceRow],
    trend_df: pd.DataFrame,
) -> list[tuple[str, float]]:
    observed = [
        (row.experiment, row.pass_rate)
        for row in confidence_rows
        if row.pass_rate is not None
    ]
    if trend_df.empty:
        return observed
    first = confidence_rows[0]
    matching = trend_df[
        (trend_df["task_type"] == first.task_type)
        & (
            (trend_df["provider"] == first.provider)
            | trend_df["provider"].map(_is_null)
        )
        & (
            (trend_df["model"] == first.model)
            | trend_df["model"].map(_is_null)
        )
    ]
    for record in matching.to_dict(orient="records"):
        rate = _to_float_or_none(record.get("pass_rate"))
        source = _to_str_or_none(record.get("source"))
        if rate is not None and source is not None:
            observed.append((source, rate))
    return observed


def _trend_sources(
    observed: list[tuple[str, float]],
    threshold: float,
) -> list[str]:
    sources = {
        source
        for source, rate in observed
        if source in TREND_SOURCE_ORDER and rate >= threshold
    }
    return [source for source in TREND_SOURCE_ORDER if source in sources]


def _confidence_row(
    record: dict[str, Any],
    *,
    run_id: str,
    aggregation_level: str,
    underpowered_n: int,
    confidence: float,
) -> PassRateConfidenceRow:
    n_attempts = int(record["n_attempts"])
    n_success = int(record["n_success"])
    ci_low, ci_high = wilson_interval(n_success, n_attempts, confidence)
    return PassRateConfidenceRow(
        run_id=run_id,
        aggregation_level=aggregation_level,
        provider=str(record["provider"]),
        model=str(record["model"]),
        provider_model=str(record["provider_model"]),
        experiment=str(record["experiment"]),
        task_type=str(record["task_type"]),
        task_family=str(record["task_family"]),
        n_attempts=n_attempts,
        n_success=n_success,
        pass_rate=float(record["pass_rate"]),
        ci_method="wilson",
        ci_confidence=confidence,
        ci_low=ci_low,
        ci_high=ci_high,
        underpowered_threshold=underpowered_n,
        underpowered=n_attempts < underpowered_n,
        source_path=str(record["source_path"]),
    )


def _provider_model_from_path(
    csv_file: Path,
    results_dir: Path,
    experiment: str,
) -> tuple[str, str]:
    try:
        parts = csv_file.relative_to(results_dir).parts
    except ValueError:
        parts = csv_file.parts
    if len(parts) >= 3 and parts[0] == experiment:
        return parts[1], parts[2]
    return "", ""


def _provider_model_from_record(record: dict[str, Any]) -> tuple[str, str, str]:
    provider = _to_str_or_none(record.get("provider")) or ""
    model = _to_str_or_none(record.get("model")) or ""
    provider_model = _to_str_or_none(record.get("provider_model"))
    if provider_model is None:
        provider_model = f"{provider}/{model}" if provider or model else ""
    if (not provider or not model) and "/" in provider_model:
        provider, model = provider_model.split("/", 1)
    return provider, model, provider_model


def _is_null(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return value == ""


def _to_float(value: object) -> float:
    return float(value)


def _to_float_or_none(value: object) -> float | None:
    if _is_null(value):
        return None
    return float(value)


def _to_int(value: object) -> int:
    return int(round(float(value)))


def _to_str_or_none(value: object) -> str | None:
    if _is_null(value):
        return None
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
