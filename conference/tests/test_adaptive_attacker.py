import json
import inspect
from pathlib import Path

import pytest

from cognition import run_eval
from cognition.adaptive_artifacts import AdaptivePolicyState
from cognition.revision_artifacts import utc_now

from cognition import adaptive_attacker


def _policy_state() -> AdaptivePolicyState:
    return AdaptivePolicyState(
        task_type="Dice_Count",
        failed_attempt_count=1,
        tried_strategy_summaries=["Checked the most visible foreground objects first."],
        next_prompt_rules=["Double-check occluded objects before producing JSON."],
        updated_at=utc_now(),
    )


def test_helper_signatures_do_not_accept_task_ground_truth() -> None:
    reflection_signature = inspect.signature(adaptive_attacker.build_reflection_prompt)
    assert list(reflection_signature.parameters) == [
        "current_prompt",
        "raw_answer",
        "parsed_answer",
        "passed",
    ]
    assert "TaskItem" not in str(reflection_signature)
    assert "gt" not in str(reflection_signature)

    adapted_signature = inspect.signature(adaptive_attacker.build_adapted_prompt)
    assert list(adapted_signature.parameters) == ["base_prompt", "policy_state"]


def test_group_tasks_by_type_preserves_task_instances() -> None:
    tasks = [
        run_eval.TaskItem("Dice_Count", "dice1.png", "prompt 1", [], {"sum": 3}),
        run_eval.TaskItem("Patch_Select", "patch1.png", "prompt 2", [], {"correct_patches": [1]}),
        run_eval.TaskItem("Dice_Count", "dice2.png", "prompt 3", [], {"sum": 4}),
    ]

    grouped = adaptive_attacker.group_tasks_by_type(tasks)

    assert list(grouped) == ["Dice_Count", "Patch_Select"]
    assert [task.puzzle_id for task in grouped["Dice_Count"]] == ["dice1.png", "dice2.png"]
    assert grouped["Patch_Select"][0] is tasks[1]


def test_adapted_prompt_and_reflection_prompt_do_not_leak_sentinel_ground_truth() -> None:
    sentinel = "SECRET_GT_COORDINATE_999"
    task = run_eval.TaskItem(
        "Geometry_Click",
        "geometry1.png",
        "Click the marked target.",
        [],
        {"bbox": sentinel},
    )
    policy_state = _policy_state()

    adapted_prompt, metadata = adaptive_attacker.build_adapted_prompt(
        task.prompt,
        policy_state,
    )
    reflection_prompt = adaptive_attacker.build_reflection_prompt(
        current_prompt=adapted_prompt,
        raw_answer='{"answer_type":"single_point","point":{"x":1,"y":2}}',
        parsed_answer={"answer_type": "single_point", "point": {"x": 1, "y": 2}},
    )
    next_state = adaptive_attacker.parse_policy_state(
        task.type,
        policy_state,
        {
            "tried_strategy_summary": "The previous strategy over-focused on one visible mark.",
            "next_prompt_rule": "Inspect all visible target-like marks before choosing.",
        },
    )

    serialized = "\n".join(
        [
            adapted_prompt,
            str(metadata),
            reflection_prompt,
            next_state.model_dump_json(),
        ]
    )
    assert "Feedback: FAIL" in reflection_prompt
    assert sentinel not in serialized


def test_reflection_schema_only_requires_strategy_and_next_rule() -> None:
    schema = adaptive_attacker.reflection_json_schema()

    assert schema["type"] == "object"
    assert schema["required"] == ["tried_strategy_summary", "next_prompt_rule"]
    assert set(schema["properties"]) == {"tried_strategy_summary", "next_prompt_rule"}
    assert schema["properties"]["tried_strategy_summary"]["maxLength"] == 300
    assert schema["properties"]["next_prompt_rule"]["maxLength"] == 500


def test_gemini_schema_sanitizer_removes_unsupported_additional_properties() -> None:
    schema = adaptive_attacker.reflection_json_schema()

    sanitized = run_eval.GeminiProvider.sanitize_response_schema(schema)

    assert "additionalProperties" in schema
    assert "additionalProperties" not in sanitized
    assert sanitized["required"] == ["tried_strategy_summary", "next_prompt_rule"]
    assert sanitized["properties"]["tried_strategy_summary"]["maxLength"] == 300


def test_parse_policy_state_updates_memory_and_rejects_leaky_notes() -> None:
    previous = _policy_state()

    updated = adaptive_attacker.parse_policy_state(
        "Dice_Count",
        previous,
        {
            "tried_strategy_summary": "Counted only clear top faces.",
            "next_prompt_rule": "Separate top faces from side faces before answering.",
        },
    )

    assert updated.failed_attempt_count == 2
    assert updated.tried_strategy_summaries[-1] == "Counted only clear top faces."
    assert updated.next_prompt_rules[-1] == "Separate top faces from side faces before answering."

    with pytest.raises(ValueError):
        adaptive_attacker.parse_policy_state(
            "Dice_Count",
            previous,
            {
                "tried_strategy_summary": "ground_truth says 3",
                "next_prompt_rule": "Use the answer directly.",
            },
        )
    with pytest.raises(ValueError, match="instance-specific answer detail"):
        adaptive_attacker.parse_policy_state(
            "Dice_Count",
            previous,
            {
                "tried_strategy_summary": "I answered value 0.",
                "next_prompt_rule": "Avoid index 2.",
            },
        )
    with pytest.raises(ValueError, match="instance-specific answer detail"):
        adaptive_attacker.parse_policy_state(
            "Geometry_Click",
            previous,
            {
                "tried_strategy_summary": "The point was x=10 y=20.",
                "next_prompt_rule": "Move away from (10, 20).",
            },
        )
    with pytest.raises(ValueError, match="instance-specific answer detail"):
        adaptive_attacker.parse_policy_state(
            "Dice_Count",
            previous,
            {
                "tried_strategy_summary": "count: 3.",
                "next_prompt_rule": "selected: 2.",
            },
        )


def test_classify_failure_values() -> None:
    assert adaptive_attacker.classify_failure("{}", {}, True, schema_valid=True) == "none"
    assert (
        adaptive_attacker.classify_failure(
            "{}",
            {},
            False,
            schema_valid=True,
            provider_exception=RuntimeError("boom"),
        )
        == "infrastructure_failure"
    )
    assert adaptive_attacker.classify_failure(
        "__ERROR__: timeout", None, False, schema_valid=False
    ) == (
        "infrastructure_failure"
    )
    assert adaptive_attacker.classify_failure(
        "not json", None, False, schema_valid=False
    ) == "protocol_failure"
    assert adaptive_attacker.classify_failure("[]", [], False, schema_valid=False) == (
        "protocol_failure"
    )
    assert adaptive_attacker.classify_failure(
        "{}", {"answer_type": "number"}, False, schema_valid=False
    ) == "protocol_failure"
    assert adaptive_attacker.classify_failure(
        "{}",
        {"answer_type": "number", "value": 0},
        False,
        schema_valid=True,
    ) == (
        "scientific_wrong"
    )


class FakeAdaptiveProvider:
    def __init__(
        self,
        solve_responses: list[tuple[str, object, dict]],
        reflection_note: dict[str, str] | None = None,
        reflection_exception: Exception | None = None,
    ) -> None:
        self.solve_responses = list(solve_responses)
        self.reflection_note = reflection_note or {
            "tried_strategy_summary": "The previous strategy was too shallow.",
            "next_prompt_rule": "Inspect the full visual field before answering.",
        }
        self.reflection_exception = reflection_exception
        self.calls: list[dict] = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["images"] == []:
            if self.reflection_exception is not None:
                raise self.reflection_exception
            return (
                json.dumps(self.reflection_note),
                dict(self.reflection_note),
                {"e2e_ms": 5.0, "tokens_in": 3, "tokens_out": 2, "cost_usd": 0.002},
            )
        if not self.solve_responses:
            raise AssertionError("Unexpected solve call")
        return self.solve_responses.pop(0)


def _number_response(value: int, *, latency_ms: float = 10.0) -> tuple[str, object, dict]:
    parsed = {"answer_type": "number", "value": value}
    return (
        json.dumps(parsed),
        parsed,
        {"e2e_ms": latency_ms, "tokens_in": 11, "tokens_out": 7, "cost_usd": 0.01},
    )


def _protocol_response() -> tuple[str, object, dict]:
    return ("not-json", None, {"e2e_ms": 8.0, "tokens_in": 4, "tokens_out": 1})


def _infrastructure_response() -> tuple[str, object, dict]:
    return ("__ERROR__: timeout", None, {"e2e_ms": 9.0, "tokens_in": 0, "tokens_out": 0})


def _tasks(count: int, *, sentinel: str | None = None) -> list[run_eval.TaskItem]:
    return [
        run_eval.TaskItem(
            "Dice_Count",
            f"dice{i}.png",
            "Count the dice.",
            [f"dice{i}.png"],
            {"sum": 3, "sentinel": sentinel} if sentinel else {"sum": 3},
        )
        for i in range(count)
    ]


def _patch_adaptive_run(
    monkeypatch,
    tmp_path: Path,
    tasks: list[run_eval.TaskItem],
    fake_provider: FakeAdaptiveProvider,
    events: list[str] | None = None,
):
    run_dir = tmp_path / "results" / "revision" / "adaptive-run"
    monkeypatch.setattr(run_eval, "build_tasks", lambda *args, **kwargs: tasks)

    def fake_make_provider(*args, **kwargs):
        assert (run_dir / "run_manifest.json").exists()
        if events is not None:
            events.append("provider")
        return fake_provider

    monkeypatch.setattr(run_eval, "make_provider", fake_make_provider)
    return run_dir


def _run_adaptive(
    tmp_path: Path,
    *,
    attempt_budget_k: int,
    resume: bool = False,
) -> dict:
    return adaptive_attacker.run_adaptive_experiment(
        dataset_root=str(tmp_path / "captcha_data"),
        types=["Dice_Count"],
        provider="openai",
        model="gpt-5",
        run_id="adaptive-run",
        output_root=str(tmp_path / "results" / "revision"),
        attempt_budget_k=attempt_budget_k,
        max_per_type=None,
        prompts_file=None,
        secrets_file="",
        stream=False,
        overwrite=not resume,
        resume=resume,
    )


def _attempt_rows(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "adaptive_attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_manifest_is_written_before_provider_and_summary_is_derived(
    tmp_path,
    monkeypatch,
) -> None:
    events: list[str] = []
    fake_provider = FakeAdaptiveProvider([_number_response(3)])
    run_dir = _patch_adaptive_run(monkeypatch, tmp_path, _tasks(1), fake_provider, events)

    result = _run_adaptive(tmp_path, attempt_budget_k=3)

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    rows = _attempt_rows(run_dir)
    assert events == ["provider"]
    assert manifest["retry_policy"]["mode"] == "session_memory_adaptive"
    assert manifest["retry_policy"]["sampling_mode"] == "without-replacement"
    assert manifest["retry_policy"]["feedback_mode"] == "binary-pass-fail"
    assert manifest["retry_policy"]["memory_mode"] == "explicit-policy-notes"
    assert manifest["retry_policy"]["stopping_rule"] == "first-success-or-budget"
    assert rows[0]["stopping_reason"] == "first_success"
    assert rows[0]["failure_class"] == "none"
    assert (run_dir / "adaptive_summary.csv").exists()
    assert (run_dir / "adaptive_summary.json").exists()
    assert result["by_type"]["Dice_Count"]["stopping_reason"] == "first_success"


def test_without_replacement_and_budget_exhaustion_have_no_duplicate_puzzle_ids(
    tmp_path,
    monkeypatch,
) -> None:
    fake_provider = FakeAdaptiveProvider(
        [_number_response(0), _number_response(0), _number_response(0)]
    )
    run_dir = _patch_adaptive_run(monkeypatch, tmp_path, _tasks(3), fake_provider)

    _run_adaptive(tmp_path, attempt_budget_k=3)

    rows = _attempt_rows(run_dir)
    puzzle_ids = [row["puzzle_id"] for row in rows]
    assert len(rows) == 3
    assert len(puzzle_ids) == len(set(puzzle_ids))
    assert rows[-1]["stopping_reason"] == "budget_exhausted"
    assert sum(call["images"] == [] for call in fake_provider.calls) == 2


def test_first_success_stops_task_type(tmp_path, monkeypatch) -> None:
    fake_provider = FakeAdaptiveProvider([_number_response(0), _number_response(3)])
    run_dir = _patch_adaptive_run(monkeypatch, tmp_path, _tasks(3), fake_provider)

    _run_adaptive(tmp_path, attempt_budget_k=3)

    rows = _attempt_rows(run_dir)
    assert len(rows) == 2
    assert rows[-1]["stopping_reason"] == "first_success"
    assert rows[-1]["correct"] is True


def test_pool_exhaustion_stops_when_pool_smaller_than_budget(tmp_path, monkeypatch) -> None:
    fake_provider = FakeAdaptiveProvider([_number_response(0)])
    run_dir = _patch_adaptive_run(monkeypatch, tmp_path, _tasks(1), fake_provider)

    _run_adaptive(tmp_path, attempt_budget_k=3)

    rows = _attempt_rows(run_dir)
    assert len(rows) == 1
    assert rows[0]["stopping_reason"] == "pool_exhausted"
    assert sum(call["images"] == [] for call in fake_provider.calls) == 0


def test_infrastructure_and_protocol_failures_do_not_trigger_reflection(
    tmp_path,
    monkeypatch,
) -> None:
    fake_provider = FakeAdaptiveProvider([_infrastructure_response(), _protocol_response()])
    run_dir = _patch_adaptive_run(monkeypatch, tmp_path, _tasks(2), fake_provider)

    _run_adaptive(tmp_path, attempt_budget_k=2)

    rows = _attempt_rows(run_dir)
    assert [row["failure_class"] for row in rows] == [
        "infrastructure_failure",
        "protocol_failure",
    ]
    assert sum(call["images"] == [] for call in fake_provider.calls) == 0


def test_schema_invalid_parsed_answers_are_protocol_failures(
    tmp_path,
    monkeypatch,
) -> None:
    malformed = (
        '{"answer_type":"number"}',
        {"answer_type": "number"},
        {"e2e_ms": 8.0, "tokens_in": 4, "tokens_out": 1},
    )
    fake_provider = FakeAdaptiveProvider([malformed, malformed])
    run_dir = _patch_adaptive_run(monkeypatch, tmp_path, _tasks(2), fake_provider)

    _run_adaptive(tmp_path, attempt_budget_k=2)

    rows = _attempt_rows(run_dir)
    assert [row["failure_class"] for row in rows] == [
        "protocol_failure",
        "protocol_failure",
    ]
    assert sum(call["images"] == [] for call in fake_provider.calls) == 0


def test_schema_validator_rejects_nested_arrays_and_scorer_coercion(
    tmp_path,
    monkeypatch,
) -> None:
    assert not adaptive_attacker._schema_valid(
        {"answer_type": "multi_select", "indices": ["1"]},
        run_eval.build_json_schema("Patch_Select"),
    )
    assert not adaptive_attacker._schema_valid(
        {"answer_type": "click_order", "points": [{"x": 1}]},
        run_eval.build_json_schema("Click_Order"),
    )
    fake_provider = FakeAdaptiveProvider(
        [
            (
                '{"answer_type":"number","value":"3"}',
                {"answer_type": "number", "value": "3"},
                {"e2e_ms": 8.0, "tokens_in": 4, "tokens_out": 1},
            )
        ]
    )
    run_dir = _patch_adaptive_run(monkeypatch, tmp_path, _tasks(1), fake_provider)

    _run_adaptive(tmp_path, attempt_budget_k=1)

    rows = _attempt_rows(run_dir)
    assert rows[0]["correct"] is False
    assert rows[0]["failure_class"] == "protocol_failure"


def test_reflection_failure_is_recorded_without_losing_solve_attempt(
    tmp_path,
    monkeypatch,
) -> None:
    fake_provider = FakeAdaptiveProvider(
        [_number_response(0), _number_response(0)],
        reflection_exception=RuntimeError("reflection unavailable"),
    )
    run_dir = _patch_adaptive_run(monkeypatch, tmp_path, _tasks(2), fake_provider)

    _run_adaptive(tmp_path, attempt_budget_k=2)

    rows = _attempt_rows(run_dir)
    assert len(rows) == 2
    assert rows[0]["failure_class"] == "scientific_wrong"
    assert rows[0]["prompt_adaptation_metadata"]["reflection_request_count"] == 1
    assert rows[0]["prompt_adaptation_metadata"]["reflection_failed"] is True
    assert rows[0]["prompt_adaptation_metadata"]["reflection_error_type"] == "RuntimeError"


def test_rejected_reflection_note_is_recorded_without_aborting_attempt(
    tmp_path,
    monkeypatch,
) -> None:
    fake_provider = FakeAdaptiveProvider(
        [_number_response(0), _number_response(0)],
        reflection_note={
            "tried_strategy_summary": "I answered value 0.",
            "next_prompt_rule": "Avoid index 2.",
        },
    )
    run_dir = _patch_adaptive_run(monkeypatch, tmp_path, _tasks(2), fake_provider)

    _run_adaptive(tmp_path, attempt_budget_k=2)

    rows = _attempt_rows(run_dir)
    assert len(rows) == 2
    assert rows[0]["prompt_adaptation_metadata"]["reflection_note_rejected"] is True
    assert rows[0]["policy_state_after"]["failed_attempt_count"] == 1
    assert rows[0]["policy_state_after"]["tried_strategy_summaries"] == []
    assert rows[0]["policy_state_after"]["next_prompt_rules"] == []


def test_scientific_wrong_reflects_only_when_another_solve_attempt_remains(
    tmp_path,
    monkeypatch,
) -> None:
    fake_provider = FakeAdaptiveProvider([_number_response(0), _number_response(0)])
    run_dir = _patch_adaptive_run(monkeypatch, tmp_path, _tasks(2), fake_provider)

    _run_adaptive(tmp_path, attempt_budget_k=2)

    rows = _attempt_rows(run_dir)
    reflection_calls = [call for call in fake_provider.calls if call["images"] == []]
    assert len(reflection_calls) == 1
    assert "Feedback: FAIL" in reflection_calls[0]["prompt"]
    assert rows[0]["prompt_adaptation_metadata"]["reflection_request_count"] == 1
    assert rows[1]["prompt_adaptation_metadata"]["reflection_request_count"] == 0


def test_run_loop_does_not_persist_sentinel_from_task_state(tmp_path, monkeypatch) -> None:
    sentinel = "SECRET_GT_COORDINATE_999"
    fake_provider = FakeAdaptiveProvider([_number_response(0), _number_response(0)])
    run_dir = _patch_adaptive_run(
        monkeypatch,
        tmp_path,
        _tasks(2, sentinel=sentinel),
        fake_provider,
    )

    _run_adaptive(tmp_path, attempt_budget_k=2)

    serialized_calls = "\n".join(str(call) for call in fake_provider.calls)
    serialized_records = (run_dir / "adaptive_attempts.jsonl").read_text(encoding="utf-8")
    assert sentinel not in serialized_calls
    assert sentinel not in serialized_records


def test_resume_skips_completed_adaptive_attempts_before_provider_construction(
    tmp_path,
    monkeypatch,
) -> None:
    fake_provider = FakeAdaptiveProvider([_number_response(3)])
    run_dir = _patch_adaptive_run(monkeypatch, tmp_path, _tasks(1), fake_provider)
    _run_adaptive(tmp_path, attempt_budget_k=2)
    original_manifest = (run_dir / "run_manifest.json").read_text(encoding="utf-8")

    monkeypatch.setattr(run_eval, "build_tasks", lambda *args, **kwargs: _tasks(1))

    def fail_provider(*args, **kwargs):
        raise AssertionError("resume should skip completed adaptive attempts")

    monkeypatch.setattr(run_eval, "make_provider", fail_provider)
    _run_adaptive(tmp_path, attempt_budget_k=2, resume=True)

    rows = _attempt_rows(run_dir)
    assert len(rows) == 1
    assert rows[0]["stopping_reason"] == "first_success"
    assert (run_dir / "run_manifest.json").read_text(encoding="utf-8") == original_manifest


def test_resume_rejects_changed_adaptive_configuration(
    tmp_path,
    monkeypatch,
) -> None:
    fake_provider = FakeAdaptiveProvider([_number_response(0)])
    _patch_adaptive_run(monkeypatch, tmp_path, _tasks(1), fake_provider)
    _run_adaptive(tmp_path, attempt_budget_k=2)

    monkeypatch.setattr(run_eval, "build_tasks", lambda *args, **kwargs: _tasks(1))

    with pytest.raises(ValueError, match="resume configuration"):
        _run_adaptive(tmp_path, attempt_budget_k=3, resume=True)


def test_resume_rejects_changed_selected_task_identity(
    tmp_path,
    monkeypatch,
) -> None:
    fake_provider = FakeAdaptiveProvider([_number_response(0)])
    _patch_adaptive_run(monkeypatch, tmp_path, _tasks(1), fake_provider)
    _run_adaptive(tmp_path, attempt_budget_k=2)
    changed_tasks = [
        run_eval.TaskItem(
            "Dice_Count",
            "dice99.png",
            "Count the dice.",
            ["dice99.png"],
            {"sum": 3},
        )
    ]
    monkeypatch.setattr(run_eval, "build_tasks", lambda *args, **kwargs: changed_tasks)

    with pytest.raises(ValueError, match="manifest"):
        _run_adaptive(tmp_path, attempt_budget_k=2, resume=True)


def test_max_per_type_selection_is_controlled_by_adaptive_seed(
    tmp_path,
    monkeypatch,
) -> None:
    tasks = _tasks(5)
    observed_max_per_type: list[int | None] = []

    def fake_build_tasks(*args, **kwargs):
        observed_max_per_type.append(args[2] if len(args) > 2 else kwargs.get("max_per_type"))
        if len(observed_max_per_type) % 2:
            return list(tasks)
        return list(reversed(tasks))

    monkeypatch.setattr(run_eval, "build_tasks", fake_build_tasks)

    def run_seeded(run_id: str) -> list[str]:
        fake_provider = FakeAdaptiveProvider([_number_response(0), _number_response(0)])
        monkeypatch.setattr(run_eval, "make_provider", lambda *args, **kwargs: fake_provider)
        adaptive_attacker.run_adaptive_experiment(
            dataset_root=str(tmp_path / "captcha_data"),
            types=["Dice_Count"],
            provider="openai",
            model="gpt-5",
            run_id=run_id,
            output_root=str(tmp_path / "results" / "revision"),
            attempt_budget_k=2,
            max_per_type=2,
            prompts_file=None,
            secrets_file="",
            seed=99,
            overwrite=True,
        )
        return [
            row["puzzle_id"]
            for row in _attempt_rows(tmp_path / "results" / "revision" / run_id)
        ]

    assert run_seeded("seed-a") == run_seeded("seed-b")
    assert observed_max_per_type == [None, None]
