import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .revision_artifacts import revision_run_dir


ADAPTIVE_POLICY_STATE_SCHEMA_VERSION = "cognition.revision.adaptive_policy_state.v1"
ADAPTIVE_ATTEMPT_SCHEMA_VERSION = "cognition.revision.adaptive_attempt.v1"
ADAPTIVE_SUMMARY_SCHEMA_VERSION = "cognition.revision.adaptive_summary.v1"
ADAPTIVE_COMPARISON_SCHEMA_VERSION = "cognition.revision.adaptive_comparison.v1"
SAMPLING_MODE = "without-replacement"
FEEDBACK_MODE = "binary-pass-fail"
MEMORY_MODE = "explicit-policy-notes"
STOPPING_RULE = "first-success-or-budget"
ALLOWED_FAILURE_CLASSES = {
    "scientific_wrong",
    "protocol_failure",
    "infrastructure_failure",
    "none",
}
ALLOWED_STOPPING_REASONS = {
    "continue",
    "first_success",
    "budget_exhausted",
    "pool_exhausted",
    "infrastructure_failure",
}

_BANNED_POLICY_TOKENS = (
    "ground_truth",
    "gt",
    "correct_answer",
    "correct coordinate",
    "correct coordinates",
    "label",
    "transcript",
    "raw_prompt",
    "raw_response",
)
_BANNED_POLICY_PATTERNS = (
    re.compile(r"\b(answer|value|index|indices|point)\b", re.IGNORECASE),
    re.compile(r"\b[xy]\s*[:=]\s*-?\d+(\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\)"),
    re.compile(r"\b\w[\w.-]*\.(png|jpg|jpeg|json)\b", re.IGNORECASE),
    re.compile(
        r"\b(count|counted|selected|clicked|chose|picked)\b\s*"
        r"(was|is|=|:)?\s*-?\d+(\.\d+)?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\b"),
)
_CI_NOT_APPLICABLE_REASON = "single adaptive session; repeated-run CI deferred to Phase 3"


def _contains_banned_token(text: str) -> str | None:
    lowered = text.casefold()
    for token in _BANNED_POLICY_TOKENS:
        if token == "gt":
            words = lowered.replace("_", " ").replace("-", " ").split()
            if token in words:
                return token
            continue
        if token in lowered:
            return token
    return None


def _validate_policy_note_texts(values: list[str]) -> list[str]:
    for value in values:
        banned = _contains_banned_token(value)
        if banned is not None:
            raise ValueError(f"adaptive policy state contains banned token: {banned}")
        if any(pattern.search(value) for pattern in _BANNED_POLICY_PATTERNS):
            raise ValueError("adaptive policy state contains instance-specific answer detail")
    return values


def _validate_mode(value: str, expected: str, field_name: str) -> str:
    if value != expected:
        raise ValueError(f"{field_name} must be {expected!r}")
    return value


class AdaptivePolicyState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = ADAPTIVE_POLICY_STATE_SCHEMA_VERSION
    task_type: str
    failed_attempt_count: int = Field(ge=0)
    tried_strategy_summaries: list[str] = Field(default_factory=list)
    next_prompt_rules: list[str] = Field(default_factory=list)
    updated_at: datetime

    @field_validator("tried_strategy_summaries", "next_prompt_rules")
    @classmethod
    def reject_leaky_policy_notes(cls, values: list[str]) -> list[str]:
        return _validate_policy_note_texts(values)


class AdaptiveAttemptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = ADAPTIVE_ATTEMPT_SCHEMA_VERSION
    run_id: str
    attempt_id: str
    provider: str
    model: str
    prompt_mode: str
    task_type: str
    puzzle_id: str
    attempt_index: int = Field(ge=1)
    attempt_budget_k: int = Field(ge=1)
    sampling_mode: str = SAMPLING_MODE
    feedback_mode: str = FEEDBACK_MODE
    memory_mode: str = MEMORY_MODE
    prior_failures: list[dict[str, Any]] = Field(default_factory=list)
    policy_state_before: AdaptivePolicyState
    policy_state_after: AdaptivePolicyState
    prompt_adaptation_metadata: dict[str, Any] = Field(default_factory=dict)
    parsed_answer: Any = None
    correct: bool
    failure_class: str
    latency_ms: float | None = Field(default=None, ge=0)
    tokens_in: int | None = Field(default=None, ge=0)
    tokens_out: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    cumulative_latency_ms: float = Field(ge=0)
    cumulative_cost_usd: float = Field(ge=0)
    stopping_reason: str
    timestamp: datetime

    @field_validator("sampling_mode")
    @classmethod
    def validate_sampling_mode(cls, value: str) -> str:
        return _validate_mode(value, SAMPLING_MODE, "sampling_mode")

    @field_validator("feedback_mode")
    @classmethod
    def validate_feedback_mode(cls, value: str) -> str:
        return _validate_mode(value, FEEDBACK_MODE, "feedback_mode")

    @field_validator("memory_mode")
    @classmethod
    def validate_memory_mode(cls, value: str) -> str:
        return _validate_mode(value, MEMORY_MODE, "memory_mode")

    @field_validator("failure_class")
    @classmethod
    def validate_failure_class(cls, value: str) -> str:
        if value not in ALLOWED_FAILURE_CLASSES:
            raise ValueError(f"failure_class must be one of {sorted(ALLOWED_FAILURE_CLASSES)}")
        return value

    @field_validator("stopping_reason")
    @classmethod
    def validate_stopping_reason(cls, value: str) -> str:
        if value not in ALLOWED_STOPPING_REASONS:
            raise ValueError(f"stopping_reason must be one of {sorted(ALLOWED_STOPPING_REASONS)}")
        return value


class AdaptiveSummaryRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = ADAPTIVE_SUMMARY_SCHEMA_VERSION
    run_id: str
    provider: str
    model: str
    task_type: str
    attempt_budget_k: int = Field(ge=1)
    sampling_mode: str = SAMPLING_MODE
    feedback_mode: str = FEEDBACK_MODE
    memory_mode: str = MEMORY_MODE
    solve_request_count: int = Field(ge=0)
    reflection_request_count: int = Field(ge=0)
    n_attempts: int = Field(ge=0)
    n_success: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    expected_attempts: float | None = Field(default=None, ge=0)
    attempts_to_success: int | None = Field(default=None, ge=1)
    cumulative_latency_ms: float = Field(ge=0)
    cumulative_cost_usd: float = Field(ge=0)
    scientific_wrong_count: int = Field(ge=0)
    protocol_failure_count: int = Field(ge=0)
    infrastructure_failure_count: int = Field(ge=0)
    stopping_reason: str
    confidence_interval_low: float | None = Field(default=None, ge=0, le=1)
    confidence_interval_high: float | None = Field(default=None, ge=0, le=1)
    confidence_interval_not_applicable_reason: str | None = None

    @field_validator("sampling_mode")
    @classmethod
    def validate_sampling_mode(cls, value: str) -> str:
        return _validate_mode(value, SAMPLING_MODE, "sampling_mode")

    @field_validator("feedback_mode")
    @classmethod
    def validate_feedback_mode(cls, value: str) -> str:
        return _validate_mode(value, FEEDBACK_MODE, "feedback_mode")

    @field_validator("memory_mode")
    @classmethod
    def validate_memory_mode(cls, value: str) -> str:
        return _validate_mode(value, MEMORY_MODE, "memory_mode")

    @field_validator("stopping_reason")
    @classmethod
    def validate_stopping_reason(cls, value: str) -> str:
        if value not in ALLOWED_STOPPING_REASONS:
            raise ValueError(f"stopping_reason must be one of {sorted(ALLOWED_STOPPING_REASONS)}")
        return value


class AdaptiveComparisonRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = ADAPTIVE_COMPARISON_SCHEMA_VERSION
    run_id: str
    provider: str
    model: str
    provider_model: str
    task_type: str
    attempt_budget_k: int = Field(ge=1)
    sampling_mode: str = SAMPLING_MODE
    feedback_mode: str = FEEDBACK_MODE
    memory_mode: str = MEMORY_MODE
    exp2_n: int = Field(ge=0)
    exp2_pass_at_1: float | None = Field(default=None, ge=0, le=1)
    bernoulli_success_at_k: float | None = Field(default=None, ge=0, le=1)
    bernoulli_expected_attempts: float | None = Field(default=None, ge=0)
    fixed_retry_observed_success: bool | None = None
    fixed_retry_attempts_to_success: int | None = Field(default=None, ge=1)
    fixed_retry_cumulative_latency_ms: float | None = Field(default=None, ge=0)
    adaptive_observed_success: bool
    adaptive_attempts_to_success: int | None = Field(default=None, ge=1)
    adaptive_cumulative_latency_ms: float = Field(ge=0)
    adaptive_solve_request_count: int = Field(ge=0)
    adaptive_reflection_request_count: int = Field(ge=0)
    adaptive_cumulative_cost_usd: float = Field(ge=0)
    scientific_wrong_count: int = Field(ge=0)
    protocol_failure_count: int = Field(ge=0)
    infrastructure_failure_count: int = Field(ge=0)
    baseline_label: str
    adaptive_label: str
    classification_change: str
    cutoff_note: str
    structural_bottleneck_tags: list[str] = Field(default_factory=list)
    persistent_failure_note: str | None = None
    confidence_interval_low: float | None = Field(default=None, ge=0, le=1)
    confidence_interval_high: float | None = Field(default=None, ge=0, le=1)
    confidence_interval_not_applicable_reason: str | None = None

    @field_validator("sampling_mode")
    @classmethod
    def validate_sampling_mode(cls, value: str) -> str:
        return _validate_mode(value, SAMPLING_MODE, "sampling_mode")

    @field_validator("feedback_mode")
    @classmethod
    def validate_feedback_mode(cls, value: str) -> str:
        return _validate_mode(value, FEEDBACK_MODE, "feedback_mode")

    @field_validator("memory_mode")
    @classmethod
    def validate_memory_mode(cls, value: str) -> str:
        return _validate_mode(value, MEMORY_MODE, "memory_mode")


class AdaptiveArtifactWriter:
    def __init__(
        self,
        output_root: str | Path,
        run_id: str,
        *,
        overwrite: bool = False,
        resume: bool = False,
    ) -> None:
        if overwrite and resume:
            raise ValueError("overwrite and resume are mutually exclusive")

        self._run_id = run_id
        self._run_dir = revision_run_dir(output_root, run_id)
        if self._run_dir.exists() and not overwrite and not resume:
            raise FileExistsError(f"Adaptive run directory already exists: {self._run_dir}")
        if self._run_dir.exists() and overwrite:
            shutil.rmtree(self._run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=resume)

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def adaptive_attempts_path(self) -> Path:
        return self.run_dir / "adaptive_attempts.jsonl"

    @property
    def adaptive_summary_csv_path(self) -> Path:
        return self.run_dir / "adaptive_summary.csv"

    @property
    def adaptive_summary_json_path(self) -> Path:
        return self.run_dir / "adaptive_summary.json"

    @property
    def adaptive_comparison_csv_path(self) -> Path:
        return self.run_dir / "adaptive_comparison.csv"

    @property
    def adaptive_comparison_json_path(self) -> Path:
        return self.run_dir / "adaptive_comparison.json"

    def output_paths(self) -> dict[str, str]:
        return {
            "adaptive_attempts": str(self.adaptive_attempts_path),
            "adaptive_summary_csv": str(self.adaptive_summary_csv_path),
            "adaptive_summary_json": str(self.adaptive_summary_json_path),
            "adaptive_comparison_csv": str(self.adaptive_comparison_csv_path),
            "adaptive_comparison_json": str(self.adaptive_comparison_json_path),
        }

    def append_attempt(self, attempt: AdaptiveAttemptRecord) -> Path:
        if attempt.run_id != self._run_id:
            raise ValueError("attempt run_id must match writer run_id")
        existing_attempt_ids = {existing.attempt_id for existing in self.iter_attempts()}
        if attempt.attempt_id in existing_attempt_ids:
            raise ValueError(
                f"Attempt already exists in adaptive_attempts.jsonl: {attempt.attempt_id}"
            )
        with self.adaptive_attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(attempt.model_dump_json())
            handle.write("\n")
        return self.adaptive_attempts_path

    def iter_attempts(self) -> Iterator[AdaptiveAttemptRecord]:
        if not self.adaptive_attempts_path.exists():
            return
        with self.adaptive_attempts_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield AdaptiveAttemptRecord.model_validate_json(line)

    def write_summary_from_attempts(self) -> tuple[Path, Path]:
        groups: dict[tuple[str, str, str, str, int], list[AdaptiveAttemptRecord]] = {}
        for attempt in self.iter_attempts():
            key = (
                attempt.run_id,
                attempt.provider,
                attempt.model,
                attempt.task_type,
                attempt.attempt_budget_k,
            )
            groups.setdefault(key, []).append(attempt)

        rows: list[AdaptiveSummaryRow] = []
        for (run_id, provider, model, task_type, attempt_budget_k), attempts in sorted(
            groups.items()
        ):
            n_attempts = len(attempts)
            n_success = sum(int(attempt.correct) for attempt in attempts)
            attempts_to_success = _first_success_attempt_index(attempts)
            rows.append(
                AdaptiveSummaryRow(
                    run_id=run_id,
                    provider=provider,
                    model=model,
                    task_type=task_type,
                    attempt_budget_k=attempt_budget_k,
                    solve_request_count=n_attempts,
                    reflection_request_count=sum(
                        _reflection_request_count(attempt) for attempt in attempts
                    ),
                    n_attempts=n_attempts,
                    n_success=n_success,
                    success_rate=n_success / n_attempts if n_attempts else 0.0,
                    expected_attempts=(
                        float(attempts_to_success) if attempts_to_success is not None else None
                    ),
                    attempts_to_success=attempts_to_success,
                    cumulative_latency_ms=sum(attempt.latency_ms or 0.0 for attempt in attempts),
                    cumulative_cost_usd=sum(attempt.cost_usd or 0.0 for attempt in attempts),
                    scientific_wrong_count=_failure_count(attempts, "scientific_wrong"),
                    protocol_failure_count=_failure_count(attempts, "protocol_failure"),
                    infrastructure_failure_count=_failure_count(
                        attempts, "infrastructure_failure"
                    ),
                    stopping_reason=_summary_stopping_reason(attempts),
                    confidence_interval_low=None,
                    confidence_interval_high=None,
                    confidence_interval_not_applicable_reason=_CI_NOT_APPLICABLE_REASON,
                )
            )

        _write_csv(self.adaptive_summary_csv_path, AdaptiveSummaryRow.model_fields, rows)
        _write_json(
            self.adaptive_summary_json_path,
            ADAPTIVE_SUMMARY_SCHEMA_VERSION,
            [row.model_dump(mode="json") for row in rows],
        )
        return self.adaptive_summary_csv_path, self.adaptive_summary_json_path

    def write_comparison_rows(
        self, rows: Iterable[AdaptiveComparisonRow]
    ) -> tuple[Path, Path]:
        materialized_rows = list(rows)
        _write_csv(
            self.adaptive_comparison_csv_path,
            AdaptiveComparisonRow.model_fields,
            materialized_rows,
        )
        _write_json(
            self.adaptive_comparison_json_path,
            ADAPTIVE_COMPARISON_SCHEMA_VERSION,
            [row.model_dump(mode="json") for row in materialized_rows],
        )
        return self.adaptive_comparison_csv_path, self.adaptive_comparison_json_path


def _reflection_request_count(attempt: AdaptiveAttemptRecord) -> int:
    metadata = attempt.prompt_adaptation_metadata
    count = metadata.get("reflection_request_count")
    if isinstance(count, bool):
        return int(count)
    if isinstance(count, int | float):
        return int(count)
    if metadata.get("reflection_requested") or metadata.get("reflection_used"):
        return 1
    return 0


def _first_success_attempt_index(attempts: list[AdaptiveAttemptRecord]) -> int | None:
    for attempt in attempts:
        if attempt.correct:
            return attempt.attempt_index
    return None


def _failure_count(attempts: list[AdaptiveAttemptRecord], failure_class: str) -> int:
    return sum(int(attempt.failure_class == failure_class) for attempt in attempts)


def _summary_stopping_reason(attempts: list[AdaptiveAttemptRecord]) -> str:
    for attempt in reversed(attempts):
        if attempt.stopping_reason != "continue":
            return attempt.stopping_reason
    if attempts:
        return attempts[-1].stopping_reason
    return "continue"


def _write_csv(
    path: Path,
    field_map: dict[str, Any],
    rows: Iterable[BaseModel],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(field_map)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def _write_json(path: Path, schema_version: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": schema_version,
                "rows": rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
