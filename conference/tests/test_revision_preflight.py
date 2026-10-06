import json
from pathlib import Path

import pytest
import yaml

from cognition.revision_artifacts import sha256_file
from cognition.revision_preflight import main


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "captcha_data"
    _write_json(root / "Dice_Count" / "ground_truth.json", {"dice1.png": {"answer": 3}})
    _write_json(root / "Connect_icon" / "ground_truth.json", {"puzzle1.json": {"choice": 0}})
    return root


def _prompts_file(tmp_path: Path) -> Path:
    path = tmp_path / "prompts.yaml"
    _write_yaml(path, {"types": {"Dice_Count": "count dice", "Connect_Icon": "connect icon"}})
    return path


def _few_shot_file(tmp_path: Path) -> Path:
    path = tmp_path / "few.yaml"
    _write_yaml(path, {"Dice_Count": {"examples": []}})
    return path


def _base_args(tmp_path: Path, run_id: str = "run-1") -> list[str]:
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
        "--max-attempts",
        "1",
    ]


def test_preflight_reports_expected_request_count_and_writes_report(tmp_path, capsys) -> None:
    exit_code = main([*_base_args(tmp_path), "--write-report"])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["expected_request_count"] == 1
    assert report["tasks"][0]["canonical_task_type"] == "Dice_Count"
    assert report["cost_preview"]["expected_request_count"] == 1
    assert (tmp_path / "results" / "revision" / "run-1" / "preflight_report.json").exists()


def test_connect_icon_alias_uses_canonical_task_and_dataset_dir(tmp_path, capsys) -> None:
    args = _base_args(tmp_path, run_id="connect-run")
    type_index = args.index("Dice_Count")
    args[type_index] = "Connect_icon"

    main(args)
    report = json.loads(capsys.readouterr().out)

    assert report["selected_task_types"] == ["Connect_Icon"]
    assert report["tasks"][0]["task_type"] == "Connect_icon"
    assert report["tasks"][0]["canonical_task_type"] == "Connect_Icon"
    assert report["tasks"][0]["dataset_dir"].endswith("Connect_icon")


def test_existing_output_directory_fails_without_overwrite_or_resume(tmp_path) -> None:
    output_dir = tmp_path / "results" / "revision" / "run-1"
    output_dir.mkdir(parents=True)

    with pytest.raises(SystemExit):
        main(_base_args(tmp_path))


def test_preflight_rejects_unsafe_run_id(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(_base_args(tmp_path, run_id="../escape"))


def test_preflight_never_calls_provider_factory(tmp_path, monkeypatch, capsys) -> None:
    from cognition import run_eval

    def fail(*args, **kwargs):
        raise AssertionError("preflight must not construct providers")

    monkeypatch.setattr(run_eval, "make_provider", fail)

    main(_base_args(tmp_path, run_id="offline-only"))
    report = json.loads(capsys.readouterr().out)

    assert report["provider"] == "openai"
    assert report["model"] == "gpt-5"


def test_prompt_and_few_shot_hashes_are_reported(tmp_path, capsys) -> None:
    prompts = _prompts_file(tmp_path)
    few_shot = _few_shot_file(tmp_path)
    args = _base_args(tmp_path, run_id="hash-run")
    args[args.index(str(prompts))] = str(prompts)
    args.extend(["--few-shot-config", str(few_shot), "--prompt-prefix", "prefix"])

    main(args)
    report = json.loads(capsys.readouterr().out)
    prompt_config = report["prompt_config"]

    assert prompt_config["prompts_file_sha256"] == sha256_file(prompts)
    assert prompt_config["few_shot_config_sha256"] == sha256_file(few_shot)
    assert prompt_config["prompt_prefix_sha256"] is not None
    assert prompt_config["prompt_suffix_sha256"] is None


def test_cost_preview_without_pricing_has_unavailable_reason(tmp_path, capsys) -> None:
    main(_base_args(tmp_path, run_id="cost-run"))
    report = json.loads(capsys.readouterr().out)

    assert report["cost_preview"]["approximate_cost_usd"] is None
    assert report["cost_preview"]["unavailable_reason"] == "pricing metadata not provided"
