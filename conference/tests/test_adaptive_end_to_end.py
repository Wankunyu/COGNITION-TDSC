import csv
import json
from pathlib import Path

import yaml

from cognition import adaptive_attacker
from cognition import adaptive_compare
from cognition import adaptive_preflight
from cognition import run_eval
from cognition.adaptive_artifacts import AdaptiveArtifactWriter


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class OfflineAdaptiveProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.solve_responses = [
            (
                json.dumps({"answer_type": "number", "value": 0}),
                {"answer_type": "number", "value": 0},
                {"e2e_ms": 10.0, "tokens_in": 11, "tokens_out": 7, "cost_usd": 0.01},
            ),
            (
                json.dumps({"answer_type": "number", "value": 3}),
                {"answer_type": "number", "value": 3},
                {"e2e_ms": 12.0, "tokens_in": 12, "tokens_out": 7, "cost_usd": 0.01},
            ),
        ]

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["images"] == []:
            parsed = {
                "tried_strategy_summary": "count top faces only",
                "next_prompt_rule": "be stricter about top faces",
            }
            return (
                json.dumps(parsed),
                parsed,
                {"e2e_ms": 5.0, "tokens_in": 4, "tokens_out": 3, "cost_usd": 0.002},
            )
        return self.solve_responses.pop(0)


def test_adaptive_workflow_end_to_end_offline(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "captcha_data"
    output_root = tmp_path / "results" / "revision"
    prompts_file = tmp_path / "prompts.yaml"
    run_id = "e2e-run"
    sentinel = "SECRET_GT_DO_NOT_LEAK"
    _write_json(
        dataset_root / "Dice_Count" / "ground_truth.json",
        {
            "dice1.png": {"sum": 3},
            "dice2.png": {"sum": 3},
        },
    )
    _write_yaml(prompts_file, {"types": {"Dice_Count": "Count dice."}})

    adaptive_preflight.main(
        [
            "--dataset-root",
            str(dataset_root),
            "--types",
            "Dice_Count",
            "--provider",
            "openai",
            "--model",
            "gpt-5",
            "--run-id",
            run_id,
            "--max-per-type",
            "2",
            "--attempt-budget-k",
            "2",
            "--prompts-file",
            str(prompts_file),
            "--output-root",
            str(output_root),
            "--write-report",
        ]
    )

    preflight_report = json.loads(
        (output_root / run_id / "adaptive_preflight_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert preflight_report["solve_request_count"] == 2
    assert preflight_report["reflection_request_count_max"] == 1
    assert preflight_report["expected_request_count_max"] == 3
    assert preflight_report["sampling_mode"] == "without-replacement"
    assert preflight_report["feedback_mode"] == "binary-pass-fail"
    assert preflight_report["memory_mode"] == "explicit-policy-notes"
    assert preflight_report["stopping_rule"] == "first-success-or-budget"

    tasks = [
        run_eval.TaskItem(
            "Dice_Count",
            "dice1.png",
            "Count dice.",
            [str(dataset_root / "Dice_Count" / "dice1.png")],
            {"sum": 3, "sentinel": sentinel},
        ),
        run_eval.TaskItem(
            "Dice_Count",
            "dice2.png",
            "Count dice.",
            [str(dataset_root / "Dice_Count" / "dice2.png")],
            {"sum": 3, "sentinel": sentinel},
        ),
    ]
    fake_provider = OfflineAdaptiveProvider()
    monkeypatch.setattr(run_eval, "build_tasks", lambda *args, **kwargs: tasks)
    monkeypatch.setattr(
        run_eval,
        "make_provider",
        lambda *args, **kwargs: fake_provider,
    )

    result = adaptive_attacker.run_adaptive_experiment(
        dataset_root=str(dataset_root),
        types=["Dice_Count"],
        provider="openai",
        model="gpt-5",
        run_id=run_id,
        output_root=str(output_root),
        attempt_budget_k=2,
        max_per_type=2,
        prompts_file=str(prompts_file),
        secrets_file="",
        overwrite=True,
    )
    writer = AdaptiveArtifactWriter(output_root, run_id, resume=True)
    assert Path(result["adaptive_attempts_path"]) == writer.adaptive_attempts_path
    assert writer.adaptive_attempts_path.exists()
    assert writer.adaptive_summary_csv_path.exists()
    assert writer.adaptive_summary_json_path.exists()
    assert sentinel not in writer.adaptive_attempts_path.read_text(encoding="utf-8")
    assert sentinel not in writer.adaptive_summary_csv_path.read_text(encoding="utf-8")
    assert sentinel not in writer.adaptive_summary_json_path.read_text(encoding="utf-8")

    serialized_calls = "\n".join(str(call) for call in fake_provider.calls)
    assert "Feedback: FAIL" in serialized_calls
    assert "count top faces only" in writer.adaptive_attempts_path.read_text(
        encoding="utf-8"
    )
    assert sentinel not in serialized_calls

    results_dir = tmp_path / "results"
    _write_csv(
        results_dir / "exp2" / "openai" / "gpt-5" / "results.csv",
        [{"type": "Dice_Count", "n": 2, "pass_at_1": 0.5}],
    )
    _write_csv(
        results_dir / "exp3" / "openai" / "gpt-5" / "results.csv",
        [
            {
                "kind": "summary",
                "type": "Dice_Count",
                "pass1": 1,
                "attempt_idx": 2,
                "cumulative_ms": 30.0,
            }
        ],
    )
    rows = adaptive_compare.build_comparison_rows(
        results_dir=str(results_dir),
        adaptive_summary_path=writer.adaptive_summary_csv_path,
        run_id=run_id,
        provider="openai",
        model="gpt-5",
        attempt_budget_k=2,
    )
    output_csv, output_json = adaptive_compare.write_comparison(
        rows,
        writer.adaptive_comparison_csv_path,
        writer.adaptive_comparison_json_path,
    )

    comparison_csv = output_csv.read_text(encoding="utf-8")
    comparison_json = json.loads(output_json.read_text(encoding="utf-8"))
    assert "exp2_pass_at_1" in comparison_csv
    assert "bernoulli_success_at_k" in comparison_csv
    assert "adaptive_observed_success" in comparison_csv
    assert "classification_change" in comparison_csv
    assert "cutoff_note" in comparison_csv
    assert "structural_bottleneck_tags" in comparison_csv
    assert comparison_json["rows"][0]["exp2_pass_at_1"] == 0.5
    assert comparison_json["rows"][0]["adaptive_observed_success"] is True
    assert sentinel not in comparison_csv
    assert sentinel not in json.dumps(comparison_json)
