import argparse
import json
from pathlib import Path

import pytest
import yaml

from cognition.adaptive_preflight import (
    ADAPTIVE_PREFLIGHT_SCHEMA_VERSION,
    AdaptivePreflightCostPreview,
    AdaptivePreflightReport,
    AdaptivePreflightTaskSummary,
    build_parser,
    main,
)
from cognition.revision_artifacts import sha256_file, sha256_text


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "captcha_data"
    _write_json(
        root / "Dice_Count" / "ground_truth.json",
        {
            "dice1.png": {"sum": 3},
            "dice2.png": {"sum": 4},
            "dice3.png": {"sum": 5},
        },
    )
    _write_json(
        root / "Connect_icon" / "ground_truth.json",
        {
            "puzzle1.json": {"choice": 0},
            "puzzle2.json": {"choice": 1},
            "puzzle3.json": {"choice": 2},
        },
    )
    return root


def _prompts_file(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.yaml"
    _write_yaml(path, {"types": {"Dice_Count": "count dice", "Connect_Icon": "connect"}})
    return path


def _few_shot_file(tmp_path: Path) -> Path:
    path = tmp_path / "few-shot.yaml"
    _write_yaml(path, {"Dice_Count": {"examples": []}, "Connect_icon": {"examples": []}})
    return path


def _pricing_file(tmp_path: Path, payload: object | None = None) -> Path:
    path = tmp_path / "pricing.yaml"
    _write_yaml(
        path,
        payload
        if payload is not None
        else {"pricing": {"openai": {"gpt-5": {"per_request_usd": 0.25}}}},
    )
    return path


def _base_args(tmp_path: Path, run_id: str = "adaptive-run") -> list[str]:
    return [
        "--dataset-root",
        str(_dataset_root(tmp_path)),
        "--types",
        "Dice_Count",
        "--prompts-file",
        str(_prompts_file(tmp_path)),
        "--output-root",
        str(tmp_path / "results" / "revision"),
        "--run-id",
        run_id,
        "--provider",
        "openai",
        "--model",
        "gpt-5",
        "--max-per-type",
        "1",
        "--attempt-budget-k",
        "2",
    ]


def test_schema_constant_and_models_expose_adaptive_semantics() -> None:
    assert ADAPTIVE_PREFLIGHT_SCHEMA_VERSION == "cognition.revision.adaptive_preflight.v1"

    task = AdaptivePreflightTaskSummary(
        task_type="Dice_Count",
        canonical_task_type="Dice_Count",
        dataset_dir="captcha_data/Dice_Count",
        ground_truth_path="captcha_data/Dice_Count/ground_truth.json",
        item_count=3,
        selected_count=1,
        solve_request_count=1,
        reflection_request_count_max=0,
    )
    cost_preview = AdaptivePreflightCostPreview(
        solve_request_count=1,
        reflection_request_count_max=0,
        expected_request_count_max=1,
        unavailable_reason="pricing metadata not provided",
    )
    report = AdaptivePreflightReport(
        run_id="adaptive-run",
        provider="openai",
        model="gpt-5",
        prompt_mode="opt",
        selected_task_types=["Dice_Count"],
        attempt_budget_k=2,
        sampling_mode="without-replacement",
        feedback_mode="binary-pass-fail",
        memory_mode="explicit-policy-notes",
        stopping_rule="first-success-or-budget",
        solve_request_count=1,
        reflection_request_count_max=0,
        expected_request_count_max=1,
        prompt_config={},
        cost_preview=cost_preview,
        output_dir="results/local_runs/adaptive-run",
        output_paths={"adaptive_preflight_report": "adaptive_preflight_report.json"},
        tasks=[task],
    )

    payload = report.model_dump(mode="json")
    assert payload["schema_version"] == ADAPTIVE_PREFLIGHT_SCHEMA_VERSION
    assert payload["sampling_mode"] == "without-replacement"
    assert payload["feedback_mode"] == "binary-pass-fail"
    assert payload["memory_mode"] == "explicit-policy-notes"
    assert payload["stopping_rule"] == "first-success-or-budget"


def test_parser_declares_required_adaptive_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--types",
            "Dice_Count",
            "--run-id",
            "adaptive-run",
            "--provider",
            "openai",
            "--model",
            "gpt-5",
        ]
    )

    assert isinstance(args, argparse.Namespace)
    assert args.attempt_budget_k == 6
    assert args.sampling_mode == "without-replacement"
    assert args.feedback_mode == "binary-pass-fail"
    assert args.memory_mode == "explicit-policy-notes"
    assert args.stopping_rule == "first-success-or-budget"
    assert args.prompt_mode == "opt"


def test_cli_prints_report_and_writes_only_when_requested(tmp_path, capsys) -> None:
    exit_code = main(_base_args(tmp_path))
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["schema_version"] == ADAPTIVE_PREFLIGHT_SCHEMA_VERSION
    assert report["selected_task_types"] == ["Dice_Count"]
    assert report["attempt_budget_k"] == 2
    assert report["sampling_mode"] == "without-replacement"
    assert report["feedback_mode"] == "binary-pass-fail"
    assert report["memory_mode"] == "explicit-policy-notes"
    assert report["stopping_rule"] == "first-success-or-budget"
    assert not (tmp_path / "results" / "revision" / "adaptive-run").exists()

    main([*_base_args(tmp_path, run_id="written-run"), "--write-report"])
    assert (
        tmp_path
        / "results"
        / "revision"
        / "written-run"
        / "adaptive_preflight_report.json"
    ).exists()


def test_preflight_does_not_construct_provider(tmp_path, monkeypatch, capsys) -> None:
    from cognition import run_eval

    def fail(*args, **kwargs):
        raise AssertionError("adaptive preflight must not construct providers")

    monkeypatch.setattr(run_eval, "make_provider", fail)

    main(_base_args(tmp_path, run_id="provider-free"))
    report = json.loads(capsys.readouterr().out)

    assert report["provider"] == "openai"
    assert report["model"] == "gpt-5"


def test_counts_hashes_output_paths_alias_and_pricing_preview(tmp_path, capsys) -> None:
    prompts = _prompts_file(tmp_path)
    few_shot = _few_shot_file(tmp_path)
    pricing = _pricing_file(tmp_path)
    args = _base_args(tmp_path, run_id="counted-run")
    args[args.index(str(prompts))] = str(prompts)
    args[args.index("1")] = "3"
    args[args.index("2")] = "3"
    args.extend(
        [
            "--few-shot-config",
            str(few_shot),
            "--prompt-prefix",
            "sk-local-secret-prefix",
            "--prompt-suffix",
            "local-secret-suffix",
            "--pricing-file",
            str(pricing),
        ]
    )

    main(args)
    report = json.loads(capsys.readouterr().out)

    assert report["attempt_budget_k"] == 3
    assert report["solve_request_count"] == 3
    assert report["reflection_request_count_max"] == 2
    assert report["expected_request_count_max"] == 5
    assert report["tasks"][0]["selected_count"] == 3
    assert report["tasks"][0]["solve_request_count"] == 3
    assert report["tasks"][0]["reflection_request_count_max"] == 2
    assert report["prompt_config"]["prompts_file_sha256"] == sha256_file(prompts)
    assert report["prompt_config"]["few_shot_config_sha256"] == sha256_file(few_shot)
    assert report["prompt_config"]["prompt_prefix_sha256"] == sha256_text(
        "sk-local-secret-prefix"
    )
    assert report["prompt_config"]["prompt_suffix_sha256"] == sha256_text(
        "local-secret-suffix"
    )
    assert report["cost_preview"]["approximate_cost_usd"] == 1.25
    assert report["cost_preview"]["pricing_source"] == str(pricing)
    assert report["cost_preview"]["unavailable_reason"] is None

    output_paths = report["output_paths"]
    assert output_paths["run_manifest"].endswith("run_manifest.json")
    assert output_paths["adaptive_preflight_report"].endswith(
        "adaptive_preflight_report.json"
    )
    assert output_paths["adaptive_attempts"].endswith("adaptive_attempts.jsonl")
    assert output_paths["adaptive_summary_csv"].endswith("adaptive_summary.csv")
    assert output_paths["adaptive_summary_json"].endswith("adaptive_summary.json")
    assert output_paths["adaptive_comparison_csv"].endswith("adaptive_comparison.csv")
    assert output_paths["adaptive_comparison_json"].endswith("adaptive_comparison.json")

    serialized = json.dumps(report)
    assert "sk-local-secret-prefix" not in serialized
    assert "local-secret-suffix" not in serialized


def test_connect_icon_alias_uses_canonical_task_and_dataset_directory(tmp_path, capsys) -> None:
    args = _base_args(tmp_path, run_id="connect-run")
    args[args.index("Dice_Count")] = "Connect_icon"

    main(args)
    report = json.loads(capsys.readouterr().out)

    assert report["selected_task_types"] == ["Connect_Icon"]
    assert report["tasks"][0]["task_type"] == "Connect_icon"
    assert report["tasks"][0]["canonical_task_type"] == "Connect_Icon"
    assert report["tasks"][0]["dataset_dir"].endswith("Connect_icon")


def test_existing_output_directory_fails_without_overwrite_or_resume(tmp_path) -> None:
    output_dir = tmp_path / "results" / "revision" / "existing-run"
    output_dir.mkdir(parents=True)

    with pytest.raises(SystemExit):
        main(_base_args(tmp_path, run_id="existing-run"))


def test_preflight_rejects_unsafe_run_id(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(_base_args(tmp_path, run_id="../escape"))


def test_pricing_file_without_model_price_reports_unavailable_reason(tmp_path, capsys) -> None:
    pricing = _pricing_file(tmp_path, {"pricing": {"openai": {"other-model": {}}}})

    main([*_base_args(tmp_path, run_id="missing-price"), "--pricing-file", str(pricing)])
    report = json.loads(capsys.readouterr().out)

    assert report["cost_preview"]["approximate_cost_usd"] is None
    assert (
        report["cost_preview"]["unavailable_reason"]
        == "pricing metadata lacks per_request_usd for provider/model"
    )
