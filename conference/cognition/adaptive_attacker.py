import argparse
import json
import platform
import random
from pathlib import Path
from typing import Any

from .adaptive_artifacts import (
    FEEDBACK_MODE,
    MEMORY_MODE,
    SAMPLING_MODE,
    STOPPING_RULE,
    AdaptiveArtifactWriter,
    AdaptiveAttemptRecord,
    AdaptivePolicyState,
)
from .revision_artifacts import (
    PromptConfig,
    RunManifest,
    collect_code_revision,
    collect_dependency_versions,
    sha256_file,
    sha256_text,
    utc_now,
)
from .revision_secrets import load_local_config

from . import run_eval


_TERMINAL_STOPPING_REASONS = {"first_success", "budget_exhausted", "pool_exhausted"}


def group_tasks_by_type(
    tasks: list[run_eval.TaskItem],
) -> dict[str, list[run_eval.TaskItem]]:
    grouped: dict[str, list[run_eval.TaskItem]] = {}
    for task in tasks:
        grouped.setdefault(task.type, []).append(task)
    return grouped


def build_adapted_prompt(
    base_prompt: str,
    policy_state: AdaptivePolicyState,
) -> tuple[str, dict[str, Any]]:
    lines = [base_prompt.rstrip()]
    if policy_state.failed_attempt_count or policy_state.tried_strategy_summaries:
        lines.extend(
            [
                "",
                "Adaptive policy notes:",
                f"- Failed attempts for this task type: {policy_state.failed_attempt_count}",
            ]
        )
        for summary in policy_state.tried_strategy_summaries:
            lines.append(f"- Tried strategy: {summary}")
        for rule in policy_state.next_prompt_rules:
            lines.append(f"- Next prompt rule: {rule}")
    metadata = {
        "sampling_mode": SAMPLING_MODE,
        "feedback_mode": FEEDBACK_MODE,
        "memory_mode": MEMORY_MODE,
        "stopping_rule": STOPPING_RULE,
        "failed_attempt_count": policy_state.failed_attempt_count,
        "tried_strategy_summary_count": len(policy_state.tried_strategy_summaries),
        "next_prompt_rule_count": len(policy_state.next_prompt_rules),
    }
    return "\n".join(lines), metadata


def build_reflection_prompt(
    current_prompt: str,
    raw_answer: str,
    parsed_answer: Any,
    passed: bool = False,
) -> str:
    parsed_text = json.dumps(parsed_answer, ensure_ascii=False, sort_keys=True)
    return "\n".join(
        [
            "Review your last CAPTCHA-solving attempt and propose a reusable policy note.",
            "Use only the facts shown here. Return JSON matching the requested schema.",
            "Do not store attempted answers, values, counts, indices, coordinates, puzzle IDs,",
            "ground-truth labels, corrective hints, or raw prompt/response transcripts.",
            "",
            "Current prompt:",
            current_prompt,
            "",
            "Your raw answer:",
            raw_answer,
            "",
            "Your parsed answer:",
            parsed_text,
            "",
            "Feedback: FAIL",
        ]
    )


def reflection_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tried_strategy_summary": {
                "type": "string",
                "maxLength": 300,
            },
            "next_prompt_rule": {
                "type": "string",
                "maxLength": 500,
            },
        },
        "required": ["tried_strategy_summary", "next_prompt_rule"],
        "additionalProperties": False,
    }


def parse_policy_state(
    task_type: str,
    previous: AdaptivePolicyState,
    parsed_note: dict[str, Any] | None,
) -> AdaptivePolicyState:
    note = parsed_note or {}
    tried_strategy_summary = str(note.get("tried_strategy_summary") or "").strip()
    next_prompt_rule = str(note.get("next_prompt_rule") or "").strip()
    tried_strategy_summaries = list(previous.tried_strategy_summaries)
    next_prompt_rules = list(previous.next_prompt_rules)
    if tried_strategy_summary:
        tried_strategy_summaries.append(tried_strategy_summary)
    if next_prompt_rule:
        next_prompt_rules.append(next_prompt_rule)
    return AdaptivePolicyState(
        task_type=task_type,
        failed_attempt_count=previous.failed_attempt_count + 1,
        tried_strategy_summaries=tried_strategy_summaries,
        next_prompt_rules=next_prompt_rules,
        updated_at=utc_now(),
    )


def classify_failure(
    raw: str,
    parsed: Any,
    correct: bool,
    *,
    schema_valid: bool,
    provider_exception: Exception | None = None,
) -> str:
    if correct:
        return "none"
    if provider_exception is not None:
        return "infrastructure_failure"
    if isinstance(raw, str) and raw.startswith("__ERROR__"):
        return "infrastructure_failure"
    if parsed is None or not isinstance(parsed, dict) or not schema_valid:
        return "protocol_failure"
    return "scientific_wrong"


def _schema_valid(value: Any, schema: dict[str, Any]) -> bool:
    expected_type = schema.get("type")
    if expected_type == "object" and not isinstance(value, dict):
        return False
    if expected_type == "array" and not isinstance(value, list):
        return False
    if expected_type == "string" and not isinstance(value, str):
        return False
    if expected_type == "integer" and (
        not isinstance(value, int) or isinstance(value, bool)
    ):
        return False
    if expected_type == "number" and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        return False

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        return False

    if isinstance(value, dict):
        for field in schema.get("required", []):
            if field not in value:
                return False
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra_fields = set(value) - set(properties)
            if extra_fields:
                return False
        for field, field_schema in properties.items():
            if field in value and not _schema_valid(value[field], field_schema):
                return False

    if isinstance(value, list):
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            return False
        if isinstance(max_items, int) and len(value) > max_items:
            return False
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            return all(_schema_valid(item, item_schema) for item in value)
    return True


def _dataset_summary(
    dataset_root: str,
    tasks: list[run_eval.TaskItem],
) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    selected_puzzle_ids_by_type: dict[str, list[str]] = {}
    prompt_sha256_by_type: dict[str, dict[str, str | None]] = {}
    gt_sha256_by_type: dict[str, dict[str, str | None]] = {}
    image_sha256_by_type: dict[str, dict[str, list[str | None]]] = {}
    for task in tasks:
        by_type[task.type] = by_type.get(task.type, 0) + 1
        selected_puzzle_ids_by_type.setdefault(task.type, []).append(task.puzzle_id)
        prompt_sha256_by_type.setdefault(task.type, {})[task.puzzle_id] = sha256_text(
            task.prompt
        )
        gt_sha256_by_type.setdefault(task.type, {})[task.puzzle_id] = sha256_text(
            json.dumps(task.gt, ensure_ascii=False, sort_keys=True)
        )
        image_sha256_by_type.setdefault(task.type, {})[task.puzzle_id] = [
            sha256_file(image_path) for image_path in task.images
        ]
    return {
        "dataset_root": dataset_root,
        "selected_task_count": len(tasks),
        "selected_task_types": sorted(by_type),
        "by_type": dict(sorted(by_type.items())),
        "selected_puzzle_ids_by_type": {
            task_type: selected_puzzle_ids_by_type[task_type]
            for task_type in sorted(selected_puzzle_ids_by_type)
        },
        "selected_prompt_sha256_by_type": {
            task_type: dict(sorted(values.items()))
            for task_type, values in sorted(prompt_sha256_by_type.items())
        },
        "selected_ground_truth_sha256_by_type": {
            task_type: dict(sorted(values.items()))
            for task_type, values in sorted(gt_sha256_by_type.items())
        },
        "selected_image_sha256_by_type": {
            task_type: dict(sorted(values.items()))
            for task_type, values in sorted(image_sha256_by_type.items())
        },
    }


def _dependency_versions() -> dict[str, str]:
    return collect_dependency_versions(
        [
            "openai",
            "anthropic",
            "google-genai",
            "Pillow",
            "PyYAML",
            "tqdm",
            "numpy",
            "pandas",
            "pydantic",
        ]
    )


def _manifest_output_paths(writer: AdaptiveArtifactWriter) -> dict[str, str]:
    return {
        "run_manifest": str(writer.run_dir / "run_manifest.json"),
        **writer.output_paths(),
    }


def _write_manifest(writer: AdaptiveArtifactWriter, manifest: RunManifest) -> Path:
    manifest_path = writer.run_dir / "run_manifest.json"
    payload = manifest.model_dump(mode="json")
    payload["output_paths"] = payload.get("output_paths") or _manifest_output_paths(writer)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest_path


def _attempt_id(run_id: str, task_type: str, attempt_index: int, puzzle_id: str) -> str:
    return f"{run_id}:{task_type}:{attempt_index}:{puzzle_id}"


def _float_meta(meta: dict[str, Any], key: str) -> float:
    try:
        value = meta.get(key)
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float_meta(meta: dict[str, Any], key: str) -> float | None:
    try:
        value = meta.get(key)
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_meta(meta: dict[str, Any], key: str) -> int | None:
    try:
        value = meta.get(key)
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _new_policy_state(task_type: str) -> AdaptivePolicyState:
    return AdaptivePolicyState(
        task_type=task_type,
        failed_attempt_count=0,
        tried_strategy_summaries=[],
        next_prompt_rules=[],
        updated_at=utc_now(),
    )


def _prior_failures_from_attempts(
    attempts: list[AdaptiveAttemptRecord],
) -> list[dict[str, Any]]:
    return [
        {
            "attempt_index": attempt.attempt_index,
            "puzzle_id": attempt.puzzle_id,
            "failure_class": attempt.failure_class,
        }
        for attempt in attempts
        if not attempt.correct
    ]


def _existing_terminal_by_type(
    attempts: list[AdaptiveAttemptRecord],
) -> dict[str, AdaptiveAttemptRecord]:
    terminal: dict[str, AdaptiveAttemptRecord] = {}
    for attempt in sorted(attempts, key=lambda item: item.attempt_index):
        if attempt.stopping_reason in _TERMINAL_STOPPING_REASONS:
            terminal[attempt.task_type] = attempt
    return terminal


def _validate_resume_attempts(
    attempts: list[AdaptiveAttemptRecord],
    *,
    provider: str,
    model: str,
    prompt_mode: str,
    attempt_budget_k: int,
) -> None:
    for attempt in attempts:
        if (
            attempt.provider != provider
            or attempt.model != model
            or attempt.prompt_mode != prompt_mode
            or attempt.attempt_budget_k != attempt_budget_k
        ):
            raise ValueError("existing adaptive attempts do not match resume configuration")


def _validate_resume_manifest(manifest_path: Path, current_manifest: RunManifest) -> None:
    if not manifest_path.exists():
        raise ValueError("cannot resume adaptive run without existing run_manifest.json")
    with manifest_path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    expected_retry_policy = current_manifest.retry_policy
    if (
        existing.get("provider") != current_manifest.provider
        or existing.get("model") != current_manifest.model
        or existing.get("seed") != current_manifest.seed
        or existing.get("retry_policy") != expected_retry_policy
        or existing.get("prompt_config") != current_manifest.prompt_config.model_dump(mode="json")
        or existing.get("dataset_summary") != current_manifest.dataset_summary
        or existing.get("code_revision") != current_manifest.code_revision
        or existing.get("python_version") != current_manifest.python_version
        or existing.get("dependency_versions") != current_manifest.dependency_versions
    ):
        raise ValueError("existing adaptive manifest does not match resume configuration")


def _summary_from_attempts(
    attempts: list[AdaptiveAttemptRecord],
) -> dict[str, dict[str, Any]]:
    by_type: dict[str, list[AdaptiveAttemptRecord]] = {}
    for attempt in attempts:
        by_type.setdefault(attempt.task_type, []).append(attempt)
    summary: dict[str, dict[str, Any]] = {}
    for task_type, task_attempts in by_type.items():
        ordered = sorted(task_attempts, key=lambda item: item.attempt_index)
        last = ordered[-1]
        summary[task_type] = {
            "attempts": len(ordered),
            "success": any(attempt.correct for attempt in ordered),
            "stopping_reason": last.stopping_reason,
            "reflection_request_count": sum(
                int(attempt.prompt_adaptation_metadata.get("reflection_request_count") or 0)
                for attempt in ordered
            ),
        }
    return summary


def _cost_control(
    task_pools: dict[str, list[run_eval.TaskItem]],
    attempt_budget_k: int,
) -> dict[str, Any]:
    solve_request_count_max = sum(
        min(len(pool), attempt_budget_k) for pool in task_pools.values()
    )
    reflection_request_count_max = sum(
        max(0, min(len(pool), attempt_budget_k) - 1) for pool in task_pools.values()
    )
    return {
        "solve_request_count_max": solve_request_count_max,
        "reflection_request_count_max": reflection_request_count_max,
        "expected_request_count_max": solve_request_count_max
        + reflection_request_count_max,
        "unavailable_reason": "pre-call adaptive request pricing not available",
    }


def run_adaptive_experiment(
    *,
    dataset_root: str,
    types: list[str],
    provider: str,
    model: str,
    run_id: str,
    output_root: str = "./results/local_runs",
    attempt_budget_k: int = 6,
    max_per_type: int | None = None,
    prompts_file: str | None = "./prompts_optimized.yaml",
    prompt_mode: str = "opt",
    prompt_prefix: str = "",
    prompt_suffix: str = "",
    secrets_file: str = "./secrets.yaml",
    timeout_sec: float = 600.0,
    seed: int = 1234,
    stream: bool = False,
    overwrite: bool = False,
    resume: bool = False,
    thinking: bool = False,
    thinking_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if attempt_budget_k < 1:
        raise ValueError("attempt_budget_k must be >= 1")

    prompts_cfg = run_eval._load_prompts_yaml(prompts_file) if prompts_file else {}
    tasks = run_eval.build_tasks(
        dataset_root,
        types,
        None,
        prompts_cfg=prompts_cfg,
        prompt_prefix=prompt_prefix,
        prompt_suffix=prompt_suffix,
        prompt_mode=prompt_mode,
    )
    task_pools = group_tasks_by_type(tasks)
    rng = random.Random(seed)
    for task_type, pool in list(task_pools.items()):
        pool = sorted(pool, key=lambda task: task.puzzle_id)
        rng.shuffle(pool)
        if max_per_type is not None and max_per_type > 0:
            task_pools[task_type] = pool[:max_per_type]
        else:
            task_pools[task_type] = pool
    selected_tasks = [task for pool in task_pools.values() for task in pool]

    writer = AdaptiveArtifactWriter(
        output_root,
        run_id,
        overwrite=overwrite,
        resume=resume,
    )
    existing_attempts = list(writer.iter_attempts())
    if resume and existing_attempts:
        _validate_resume_attempts(
            existing_attempts,
            provider=provider,
            model=model,
            prompt_mode=prompt_mode,
            attempt_budget_k=attempt_budget_k,
        )
    existing_attempt_ids = {attempt.attempt_id for attempt in existing_attempts}
    terminal_by_type = _existing_terminal_by_type(existing_attempts)

    manifest = RunManifest(
        run_id=run_id,
        created_at=utc_now(),
        code_revision=collect_code_revision(),
        python_version=platform.python_version(),
        dependency_versions=_dependency_versions(),
        dataset_summary=_dataset_summary(dataset_root, selected_tasks),
        prompt_config=PromptConfig(
            prompt_mode=prompt_mode,
            prompts_file=prompts_file,
            prompts_file_sha256=sha256_file(prompts_file),
            few_shot_config=None,
            few_shot_config_sha256=None,
            prompt_prefix_sha256=sha256_text(prompt_prefix),
            prompt_suffix_sha256=sha256_text(prompt_suffix),
        ),
        provider=provider,
        model=model,
        seed=seed,
        retry_policy={
            "mode": "session_memory_adaptive",
            "attempt_budget_k": attempt_budget_k,
            "sampling_mode": SAMPLING_MODE,
            "feedback_mode": FEEDBACK_MODE,
            "memory_mode": MEMORY_MODE,
            "stopping_rule": STOPPING_RULE,
        },
        cost_control=_cost_control(task_pools, attempt_budget_k),
        output_paths=_manifest_output_paths(writer),
    )
    if resume and existing_attempts:
        manifest_path = writer.run_dir / "run_manifest.json"
        _validate_resume_manifest(manifest_path, manifest)
    else:
        manifest_path = _write_manifest(writer, manifest)

    remaining_task_types = [
        task_type for task_type in task_pools if task_type not in terminal_by_type
    ]
    if not remaining_task_types:
        summary_csv, summary_json = writer.write_summary_from_attempts()
        all_attempts = list(writer.iter_attempts())
        return {
            "run_id": run_id,
            "output_dir": str(writer.run_dir),
            "manifest_path": str(manifest_path),
            "adaptive_attempts_path": str(writer.adaptive_attempts_path),
            "adaptive_summary_csv": str(summary_csv),
            "adaptive_summary_json": str(summary_json),
            "by_type": _summary_from_attempts(all_attempts),
        }

    secrets = load_local_config(secrets_file)
    model_provider = run_eval.make_provider(
        provider,
        model,
        secrets,
        timeout_sec,
        thinking_enabled=thinking,
        thinking_options=thinking_options,
    )

    attempts_by_type: dict[str, list[AdaptiveAttemptRecord]] = {}
    for attempt in existing_attempts:
        attempts_by_type.setdefault(attempt.task_type, []).append(attempt)

    for task_type in remaining_task_types:
        previous_attempts = sorted(
            attempts_by_type.get(task_type, []),
            key=lambda item: item.attempt_index,
        )
        used_puzzle_ids = {attempt.puzzle_id for attempt in previous_attempts}
        pool = [
            task for task in task_pools[task_type] if task.puzzle_id not in used_puzzle_ids
        ]
        attempt_index = (
            max((attempt.attempt_index for attempt in previous_attempts), default=0) + 1
        )
        policy_state = (
            previous_attempts[-1].policy_state_after
            if previous_attempts
            else _new_policy_state(task_type)
        )
        prior_failures = _prior_failures_from_attempts(previous_attempts)
        cumulative_latency_ms = sum(
            attempt.latency_ms or 0.0 for attempt in previous_attempts
        )
        cumulative_cost_usd = sum(attempt.cost_usd or 0.0 for attempt in previous_attempts)

        while attempt_index <= attempt_budget_k:
            if not pool:
                break
            task = pool.pop(0)
            policy_state_before = policy_state
            adapted_prompt, prompt_metadata = build_adapted_prompt(
                task.prompt,
                policy_state_before,
            )
            provider_exception: Exception | None = None
            try:
                schema = run_eval.build_json_schema(task.type)
                raw, parsed, meta = model_provider.infer(
                    prompt=adapted_prompt,
                    images=task.images,
                    json_schema=schema,
                    stream=stream,
                )
            except Exception as exc:
                provider_exception = exc
                raw = f"__ERROR__: {type(exc).__name__}: {exc}"
                parsed = None
                meta = {"e2e_ms": 0.0, "tokens_in": 0, "tokens_out": 0}
                schema = run_eval.build_json_schema(task.type)

            schema_valid = _schema_valid(parsed, schema)
            correct = (
                False
                if provider_exception is not None or not schema_valid
                else run_eval.evaluate_pass1(task, parsed)
            )
            failure_class = classify_failure(
                raw,
                parsed,
                correct,
                schema_valid=schema_valid,
                provider_exception=provider_exception,
            )
            solve_latency_ms = _float_meta(meta, "e2e_ms")
            solve_cost_usd = _optional_float_meta(meta, "cost_usd")
            solve_tokens_in = _int_meta(meta, "tokens_in")
            solve_tokens_out = _int_meta(meta, "tokens_out")
            latency_ms = solve_latency_ms
            cost_usd = solve_cost_usd
            tokens_in = solve_tokens_in
            tokens_out = solve_tokens_out

            another_solve_remains = attempt_index < attempt_budget_k and bool(pool)
            reflection_request_count = 0
            reflection_meta: dict[str, Any] = {}
            parsed_note: dict[str, Any] | None = None
            if failure_class == "scientific_wrong" and another_solve_remains:
                reflection_prompt = build_reflection_prompt(
                    adapted_prompt,
                    raw,
                    parsed,
                    passed=False,
                )
                reflection_request_count = 1
                try:
                    reflection_raw, reflection_parsed, reflection_call_meta = model_provider.infer(
                        prompt=reflection_prompt,
                        images=[],
                        json_schema=reflection_json_schema(),
                        stream=stream,
                    )
                    if isinstance(reflection_parsed, dict):
                        parsed_note = reflection_parsed
                    reflection_latency_ms = _float_meta(reflection_call_meta, "e2e_ms")
                    reflection_cost_usd = _optional_float_meta(
                        reflection_call_meta,
                        "cost_usd",
                    )
                    latency_ms += reflection_latency_ms
                    if cost_usd is None:
                        cost_usd = reflection_cost_usd
                    elif reflection_cost_usd is not None:
                        cost_usd += reflection_cost_usd
                    reflection_tokens_in = _int_meta(reflection_call_meta, "tokens_in")
                    reflection_tokens_out = _int_meta(reflection_call_meta, "tokens_out")
                    tokens_in = (tokens_in or 0) + (reflection_tokens_in or 0)
                    tokens_out = (tokens_out or 0) + (reflection_tokens_out or 0)
                    reflection_meta = {
                        "reflection_requested": True,
                        "reflection_raw_present": bool(reflection_raw),
                        "reflection_latency_ms": reflection_latency_ms,
                        "reflection_tokens_in": reflection_tokens_in,
                        "reflection_tokens_out": reflection_tokens_out,
                        "reflection_cost_usd": reflection_cost_usd,
                    }
                except Exception as exc:
                    reflection_meta = {
                        "reflection_requested": True,
                        "reflection_failed": True,
                        "reflection_error_type": type(exc).__name__,
                    }

            if correct:
                policy_state_after = policy_state_before
            else:
                try:
                    policy_state_after = parse_policy_state(
                        task.type,
                        policy_state_before,
                        parsed_note,
                    )
                except ValueError:
                    reflection_meta["reflection_note_rejected"] = True
                    policy_state_after = parse_policy_state(
                        task.type,
                        policy_state_before,
                        None,
                    )

            cumulative_latency_ms += latency_ms
            cumulative_cost_usd += cost_usd or 0.0
            if correct:
                stopping_reason = "first_success"
            elif attempt_index >= attempt_budget_k:
                stopping_reason = "budget_exhausted"
            elif not pool:
                stopping_reason = "pool_exhausted"
            else:
                stopping_reason = "continue"

            adaptation_metadata = {
                **prompt_metadata,
                "solve_request_index": attempt_index,
                "reflection_request_count": reflection_request_count,
                "reflection_requested": bool(reflection_request_count),
                "solve_latency_ms": solve_latency_ms,
                "solve_cost_usd": solve_cost_usd,
                **reflection_meta,
            }
            adaptive_attempt = AdaptiveAttemptRecord(
                run_id=run_id,
                attempt_id=_attempt_id(run_id, task.type, attempt_index, task.puzzle_id),
                provider=provider,
                model=model,
                prompt_mode=prompt_mode,
                task_type=task.type,
                puzzle_id=task.puzzle_id,
                attempt_index=attempt_index,
                attempt_budget_k=attempt_budget_k,
                prior_failures=list(prior_failures),
                policy_state_before=policy_state_before,
                policy_state_after=policy_state_after,
                prompt_adaptation_metadata=adaptation_metadata,
                parsed_answer=parsed,
                correct=correct,
                failure_class=failure_class,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
                cumulative_latency_ms=cumulative_latency_ms,
                cumulative_cost_usd=cumulative_cost_usd,
                stopping_reason=stopping_reason,
                timestamp=utc_now(),
            )
            if adaptive_attempt.attempt_id not in existing_attempt_ids:
                writer.append_attempt(adaptive_attempt)
                existing_attempt_ids.add(adaptive_attempt.attempt_id)
            policy_state = policy_state_after
            if not correct:
                prior_failures.append(
                    {
                        "attempt_index": attempt_index,
                        "puzzle_id": task.puzzle_id,
                        "failure_class": failure_class,
                    }
                )
            if stopping_reason != "continue":
                break
            attempt_index += 1

    summary_csv, summary_json = writer.write_summary_from_attempts()
    all_attempts = list(writer.iter_attempts())
    return {
        "run_id": run_id,
        "output_dir": str(writer.run_dir),
        "manifest_path": str(manifest_path),
        "adaptive_attempts_path": str(writer.adaptive_attempts_path),
        "adaptive_summary_csv": str(summary_csv),
        "adaptive_summary_json": str(summary_json),
        "by_type": _summary_from_attempts(all_attempts),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an offline dataset-based adaptive CAPTCHA attacker experiment."
    )
    parser.add_argument("--dataset-root", default="./captcha_data")
    parser.add_argument("--types", nargs="+", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="./results/local_runs")
    parser.add_argument("--attempt-budget-k", type=int, default=6)
    parser.add_argument("--max-per-type", type=int, default=None)
    parser.add_argument("--prompts-file", default="./prompts_optimized.yaml")
    parser.add_argument("--prompt-mode", default="opt")
    parser.add_argument("--prompt-prefix", default="")
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--secrets-file", default="./secrets.yaml")
    parser.add_argument("--timeout-sec", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_adaptive_experiment(
        dataset_root=args.dataset_root,
        types=args.types,
        provider=args.provider,
        model=args.model,
        run_id=args.run_id,
        output_root=args.output_root,
        attempt_budget_k=args.attempt_budget_k,
        max_per_type=args.max_per_type,
        prompts_file=args.prompts_file,
        prompt_mode=args.prompt_mode,
        prompt_prefix=args.prompt_prefix,
        prompt_suffix=args.prompt_suffix,
        secrets_file=args.secrets_file,
        timeout_sec=args.timeout_sec,
        seed=args.seed,
        stream=args.stream,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
