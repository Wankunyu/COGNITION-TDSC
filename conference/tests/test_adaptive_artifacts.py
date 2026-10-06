import csv
import json
from datetime import datetime, timezone

import pytest

from cognition.adaptive_artifacts import (
    ADAPTIVE_ATTEMPT_SCHEMA_VERSION,
    ADAPTIVE_COMPARISON_SCHEMA_VERSION,
    ADAPTIVE_POLICY_STATE_SCHEMA_VERSION,
    ADAPTIVE_SUMMARY_SCHEMA_VERSION,
    FEEDBACK_MODE,
    MEMORY_MODE,
    SAMPLING_MODE,
    STOPPING_RULE,
    AdaptiveArtifactWriter,
    AdaptiveAttemptRecord,
    AdaptiveComparisonRow,
    AdaptivePolicyState,
    AdaptiveSummaryRow,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _policy_state(
    task_type: str = "Dice_Count",
    failed_attempt_count: int = 0,
) -> AdaptivePolicyState:
    return AdaptivePolicyState(
        task_type=task_type,
        failed_attempt_count=failed_attempt_count,
        tried_strategy_summaries=["Count visible objects before answering."],
        next_prompt_rules=["Use the declared JSON schema."],
        updated_at=_now(),
    )


def _attempt(
    run_id: str,
    attempt_id: str,
    task_type: str,
    attempt_index: int,
    correct: bool,
    failure_class: str,
    *,
    latency_ms: float,
    cost_usd: float,
    stopping_reason: str,
    reflection_request_count: int = 0,
) -> AdaptiveAttemptRecord:
    before = _policy_state(task_type=task_type, failed_attempt_count=attempt_index - 1)
    after = _policy_state(
        task_type=task_type,
        failed_attempt_count=attempt_index if not correct else attempt_index - 1,
    )
    return AdaptiveAttemptRecord(
        run_id=run_id,
        attempt_id=attempt_id,
        provider="openai",
        model="gpt-5",
        prompt_mode="optimized",
        task_type=task_type,
        puzzle_id=f"{task_type}-{attempt_index}.png",
        attempt_index=attempt_index,
        attempt_budget_k=3,
        prior_failures=[],
        policy_state_before=before,
        policy_state_after=after,
        prompt_adaptation_metadata={
            "stopping_rule": STOPPING_RULE,
            "reflection_request_count": reflection_request_count,
        },
        parsed_answer={"answer_type": "number", "value": attempt_index},
        correct=correct,
        failure_class=failure_class,
        latency_ms=latency_ms,
        tokens_in=10,
        tokens_out=5,
        cost_usd=cost_usd,
        cumulative_latency_ms=latency_ms,
        cumulative_cost_usd=cost_usd,
        stopping_reason=stopping_reason,
        timestamp=_now(),
    )


def test_adaptive_models_expose_schema_versions_and_modes() -> None:
    before = _policy_state()
    after = _policy_state(failed_attempt_count=1)

    attempt = AdaptiveAttemptRecord(
        run_id="run-1",
        attempt_id="run-1:Dice_Count:1:dice1",
        provider="openai",
        model="gpt-5",
        prompt_mode="optimized",
        task_type="Dice_Count",
        puzzle_id="dice1.png",
        attempt_index=1,
        attempt_budget_k=3,
        prior_failures=[],
        policy_state_before=before,
        policy_state_after=after,
        prompt_adaptation_metadata={"stopping_rule": STOPPING_RULE},
        parsed_answer={"answer_type": "number", "value": 3},
        correct=False,
        failure_class="scientific_wrong",
        latency_ms=120.0,
        tokens_in=10,
        tokens_out=5,
        cost_usd=0.01,
        cumulative_latency_ms=120.0,
        cumulative_cost_usd=0.01,
        stopping_reason="continue",
        timestamp=_now(),
    )
    summary = AdaptiveSummaryRow(
        run_id="run-1",
        provider="openai",
        model="gpt-5",
        task_type="Dice_Count",
        attempt_budget_k=3,
        solve_request_count=1,
        reflection_request_count=1,
        n_attempts=1,
        n_success=0,
        success_rate=0.0,
        expected_attempts=None,
        attempts_to_success=None,
        cumulative_latency_ms=120.0,
        cumulative_cost_usd=0.01,
        scientific_wrong_count=1,
        protocol_failure_count=0,
        infrastructure_failure_count=0,
        stopping_reason="continue",
        confidence_interval_low=None,
        confidence_interval_high=None,
        confidence_interval_not_applicable_reason=(
            "single adaptive session; repeated-run CI deferred to Phase 3"
        ),
    )
    comparison = AdaptiveComparisonRow(
        run_id="run-1",
        provider="openai",
        model="gpt-5",
        provider_model="openai/gpt-5",
        task_type="Dice_Count",
        attempt_budget_k=3,
        exp2_n=10,
        exp2_pass_at_1=0.1,
        bernoulli_success_at_k=0.271,
        bernoulli_expected_attempts=2.71,
        fixed_retry_observed_success=False,
        fixed_retry_attempts_to_success=None,
        fixed_retry_cumulative_latency_ms=None,
        adaptive_observed_success=False,
        adaptive_attempts_to_success=None,
        adaptive_cumulative_latency_ms=120.0,
        adaptive_solve_request_count=1,
        adaptive_reflection_request_count=1,
        adaptive_cumulative_cost_usd=0.01,
        scientific_wrong_count=1,
        protocol_failure_count=0,
        infrastructure_failure_count=0,
        baseline_label="hard",
        adaptive_label="hard",
        classification_change="unchanged",
        cutoff_note="40 percent cutoff is operational, not a universal security boundary.",
        structural_bottleneck_tags=["counting"],
        persistent_failure_note="Still failed after binary-feedback adaptation.",
        confidence_interval_low=None,
        confidence_interval_high=None,
        confidence_interval_not_applicable_reason=(
            "single adaptive session; repeated-run CI deferred to Phase 3"
        ),
    )

    assert before.schema_version == ADAPTIVE_POLICY_STATE_SCHEMA_VERSION
    assert attempt.schema_version == ADAPTIVE_ATTEMPT_SCHEMA_VERSION
    assert summary.schema_version == ADAPTIVE_SUMMARY_SCHEMA_VERSION
    assert comparison.schema_version == ADAPTIVE_COMPARISON_SCHEMA_VERSION
    assert attempt.sampling_mode == SAMPLING_MODE == "without-replacement"
    assert summary.feedback_mode == FEEDBACK_MODE == "binary-pass-fail"
    assert comparison.memory_mode == MEMORY_MODE == "explicit-policy-notes"
    assert attempt.prompt_adaptation_metadata["stopping_rule"] == "first-success-or-budget"


def test_policy_state_rejects_banned_text() -> None:
    with pytest.raises(ValueError):
        AdaptivePolicyState(
            task_type="Dice_Count",
            failed_attempt_count=1,
            tried_strategy_summaries=["ground_truth says 3"],
            next_prompt_rules=[],
            updated_at=_now(),
        )
    with pytest.raises(ValueError, match="instance-specific answer detail"):
        AdaptivePolicyState(
            task_type="Dice_Count",
            failed_attempt_count=1,
            tried_strategy_summaries=["I answered value 0"],
            next_prompt_rules=[],
            updated_at=_now(),
        )
    with pytest.raises(ValueError, match="instance-specific answer detail"):
        AdaptivePolicyState(
            task_type="Geometry_Click",
            failed_attempt_count=1,
            tried_strategy_summaries=[],
            next_prompt_rules=["Avoid the point x=10 y=20"],
            updated_at=_now(),
        )
    for text in (
        "count was 3",
        "count: 3",
        "count=3",
        "selected: 2",
        "picked=4",
        "clicked 120, 88",
        "dice17.png failed",
    ):
        with pytest.raises(ValueError, match="instance-specific answer detail"):
            AdaptivePolicyState(
                task_type="Dice_Count",
                failed_attempt_count=1,
                tried_strategy_summaries=[text],
                next_prompt_rules=[],
                updated_at=_now(),
            )


def test_adaptive_writer_appends_jsonl_in_order_and_rejects_duplicate_or_unsafe_runs(
    tmp_path,
) -> None:
    writer = AdaptiveArtifactWriter(tmp_path / "results" / "revision", "run-1")
    attempts = [
        _attempt(
            "run-1",
            "one",
            "Dice_Count",
            1,
            False,
            "scientific_wrong",
            latency_ms=100.0,
            cost_usd=0.01,
            stopping_reason="continue",
            reflection_request_count=1,
        ),
        _attempt(
            "run-1",
            "two",
            "Dice_Count",
            2,
            True,
            "none",
            latency_ms=200.0,
            cost_usd=0.02,
            stopping_reason="first_success",
        ),
        _attempt(
            "run-1",
            "three",
            "Patch_Select",
            1,
            False,
            "protocol_failure",
            latency_ms=50.0,
            cost_usd=0.005,
            stopping_reason="budget_exhausted",
        ),
    ]

    for attempt in attempts:
        writer.append_attempt(attempt)

    lines = writer.adaptive_attempts_path.read_text(encoding="utf-8").splitlines()
    assert writer.adaptive_attempts_path.name == "adaptive_attempts.jsonl"
    assert [json.loads(line)["attempt_id"] for line in lines] == ["one", "two", "three"]
    assert [attempt.attempt_id for attempt in writer.iter_attempts()] == ["one", "two", "three"]

    with pytest.raises(ValueError, match="Attempt already exists"):
        writer.append_attempt(attempts[0])

    with pytest.raises(ValueError):
        AdaptiveArtifactWriter(tmp_path, "../escape")


def test_adaptive_summary_is_derived_from_persisted_attempts(tmp_path) -> None:
    writer = AdaptiveArtifactWriter(tmp_path / "results" / "revision", "run-1")
    writer.append_attempt(
        _attempt(
            "run-1",
            "one",
            "Dice_Count",
            1,
            False,
            "scientific_wrong",
            latency_ms=100.0,
            cost_usd=0.01,
            stopping_reason="continue",
            reflection_request_count=1,
        )
    )
    writer.append_attempt(
        _attempt(
            "run-1",
            "two",
            "Dice_Count",
            2,
            True,
            "none",
            latency_ms=200.0,
            cost_usd=0.02,
            stopping_reason="first_success",
        )
    )
    writer.append_attempt(
        _attempt(
            "run-1",
            "three",
            "Patch_Select",
            1,
            False,
            "protocol_failure",
            latency_ms=50.0,
            cost_usd=0.005,
            stopping_reason="budget_exhausted",
        )
    )

    csv_path, json_path = writer.write_summary_from_attempts()

    assert csv_path.name == "adaptive_summary.csv"
    assert json_path.name == "adaptive_summary.json"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["schema_version"] == ADAPTIVE_SUMMARY_SCHEMA_VERSION
    assert {row["task_type"] for row in rows} == {"Dice_Count", "Patch_Select"}
    dice_row = next(row for row in rows if row["task_type"] == "Dice_Count")
    assert dice_row["solve_request_count"] == "2"
    assert dice_row["reflection_request_count"] == "1"
    assert dice_row["n_attempts"] == "2"
    assert dice_row["n_success"] == "1"
    assert float(dice_row["success_rate"]) == 0.5
    assert dice_row["attempts_to_success"] == "2"
    assert float(dice_row["cumulative_latency_ms"]) == 300.0
    assert float(dice_row["cumulative_cost_usd"]) == 0.03
    assert dice_row["scientific_wrong_count"] == "1"
    assert dice_row["protocol_failure_count"] == "0"
    assert dice_row["infrastructure_failure_count"] == "0"
    assert dice_row["stopping_reason"] == "first_success"
    assert (
        dice_row["confidence_interval_not_applicable_reason"]
        == "single adaptive session; repeated-run CI deferred to Phase 3"
    )

    patch_row = next(row for row in rows if row["task_type"] == "Patch_Select")
    assert patch_row["protocol_failure_count"] == "1"
    assert patch_row["stopping_reason"] == "budget_exhausted"


def test_adaptive_writer_serializes_comparison_rows(tmp_path) -> None:
    writer = AdaptiveArtifactWriter(tmp_path / "results" / "revision", "run-1")
    row = AdaptiveComparisonRow(
        run_id="run-1",
        provider="openai",
        model="gpt-5",
        provider_model="openai/gpt-5",
        task_type="Dice_Count",
        attempt_budget_k=3,
        exp2_n=10,
        exp2_pass_at_1=0.1,
        bernoulli_success_at_k=0.271,
        bernoulli_expected_attempts=2.71,
        fixed_retry_observed_success=False,
        fixed_retry_attempts_to_success=None,
        fixed_retry_cumulative_latency_ms=None,
        adaptive_observed_success=True,
        adaptive_attempts_to_success=2,
        adaptive_cumulative_latency_ms=300.0,
        adaptive_solve_request_count=2,
        adaptive_reflection_request_count=1,
        adaptive_cumulative_cost_usd=0.03,
        scientific_wrong_count=1,
        protocol_failure_count=0,
        infrastructure_failure_count=0,
        baseline_label="hard",
        adaptive_label="borderline",
        classification_change="improved",
        cutoff_note="40 percent cutoff is operational, not a universal security boundary.",
        structural_bottleneck_tags=["counting"],
        persistent_failure_note=None,
        confidence_interval_low=None,
        confidence_interval_high=None,
        confidence_interval_not_applicable_reason=(
            "single adaptive session; repeated-run CI deferred to Phase 3"
        ),
    )

    csv_path, json_path = writer.write_comparison_rows([row])

    assert csv_path.name == "adaptive_comparison.csv"
    assert json_path.name == "adaptive_comparison.json"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert rows[0]["adaptive_reflection_request_count"] == "1"
    assert payload["schema_version"] == ADAPTIVE_COMPARISON_SCHEMA_VERSION
    assert payload["rows"][0]["structural_bottleneck_tags"] == ["counting"]

    with pytest.raises(ValueError):
        AdaptivePolicyState(
            task_type="Dice_Count",
            failed_attempt_count=1,
            tried_strategy_summaries=[],
            next_prompt_rules=["raw_response transcript"],
            updated_at=_now(),
        )
