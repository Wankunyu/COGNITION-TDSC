import csv
import json
from pathlib import Path

import yaml

from cognition import run_eval


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "pricing": {
                    "openai": {
                        "gpt-5": {
                            "in_per_1k": 0.001,
                            "out_per_1k": 0.002,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_path_finder_classification_scoring() -> None:
    task = run_eval.TaskItem(
        type="Path_Finder",
        puzzle_id="path1.json",
        prompt="choose the path",
        images=[],
        gt={"correct_index": 2},
    )

    assert run_eval.evaluate_pass1(task, {"answer_type": "classify", "index": 2})
    assert not run_eval.evaluate_pass1(task, {"answer_type": "classify", "index": 1})
    assert not run_eval.evaluate_pass1(task, {"answer_type": "multi_select", "indices": [2]})


def test_symbol_count_scoring() -> None:
    task = run_eval.TaskItem(
        type="Symbol_Count",
        puzzle_id="sample1.png",
        prompt="count symbols",
        images=[],
        gt={"count": 7},
    )

    assert run_eval.evaluate_pass1(task, {"answer_type": "number", "value": 7})
    assert not run_eval.evaluate_pass1(task, {"answer_type": "number", "value": 6})
    assert not run_eval.evaluate_pass1(task, {"answer_type": "classify", "index": 7})


def test_relation_match_classification_scoring() -> None:
    task = run_eval.TaskItem(
        type="Relation_Match",
        puzzle_id="sample2.json",
        prompt="choose relation",
        images=[],
        gt={"correct_index": 1},
    )

    assert run_eval.evaluate_pass1(task, {"answer_type": "classify", "index": 1})
    assert not run_eval.evaluate_pass1(task, {"answer_type": "classify", "index": 0})
    assert not run_eval.evaluate_pass1(task, {"answer_type": "number", "value": 1})


def test_hole_counting_multiselect_scoring() -> None:
    task = run_eval.TaskItem(
        type="Hole_Counting",
        puzzle_id="holes.png",
        prompt="select cells with holes",
        images=[],
        gt={"correct_patches": [1, 3]},
    )

    assert run_eval.evaluate_pass1(task, {"answer_type": "multi_select", "indices": [3, 1]})
    assert not run_eval.evaluate_pass1(task, {"answer_type": "multi_select", "indices": [1]})
    assert not run_eval.evaluate_pass1(task, {"answer_type": "number", "value": 2})


def test_describe_failure_multiselect_no_name_error() -> None:
    task = run_eval.TaskItem(
        type="Image_Recognition",
        puzzle_id="image1.png",
        prompt="select items",
        images=[],
        gt={"indices_gt": [1, 2]},
    )

    description = run_eval._describe_failure(
        task,
        {"answer_type": "multi_select", "indices": [1]},
        raw='{"answer_type":"multi_select","indices":[1]}',
    )

    assert "Missing" in description


def test_summary_csv_writes_all_rows(tmp_path, monkeypatch) -> None:
    tasks = [
        run_eval.TaskItem("Dice_Count", "dice1.png", "prompt", [], {"sum": 3}),
        run_eval.TaskItem("Patch_Select", "patch1.png", "prompt", [], {"correct_patches": [1]}),
    ]
    responses = iter(
        [
            ({"answer_type": "number", "value": 3}, {"e2e_ms": 10.0, "ttft_ms": 2.0}),
            ({"answer_type": "multi_select", "indices": [1]}, {"e2e_ms": 20.0, "ttft_ms": 4.0}),
        ]
    )

    class MultiFakeProvider:
        def infer(self, **kwargs):
            parsed, meta = next(responses)
            return json.dumps(parsed), parsed, meta

    monkeypatch.setattr(run_eval, "build_tasks", lambda *args, **kwargs: tasks)
    monkeypatch.setattr(run_eval, "make_provider", lambda *args, **kwargs: MultiFakeProvider())
    secrets_file = tmp_path / "local-config.yaml"
    _write_config(secrets_file)
    summary_csv = tmp_path / "summary.csv"

    run_eval.run_eval(
        dataset_root=str(tmp_path / "captcha_data"),
        types=["Dice_Count", "Patch_Select"],
        out_csv=str(tmp_path / "legacy.csv"),
        secrets_file=str(secrets_file),
        stream=False,
        summary_csv=str(summary_csv),
    )

    with summary_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["type"] for row in rows] == ["Dice_Count", "Patch_Select"]
