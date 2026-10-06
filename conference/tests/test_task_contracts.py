import json
from pathlib import Path

import yaml

from cognition.revision_preflight import DATASET_DIR_ALIASES, TASK_ALIASES
from cognition.run_eval import SUPPORTED_TYPES, build_json_schema, build_tasks
from cognition.visualize_results import CAPTCHAVisualizer


IGNORED_NOT_USED_KEYS = {"Hold_Button(Not Used)", "Slide_Puzzle(Not Used)"}
PHASE041_SIDECAR_ONLY_TYPES = {"Symbol_Count", "Relation_Match", "Hole_Counting"}


def _canonical_task_type(task_type: str) -> str:
    return TASK_ALIASES.get(task_type, task_type)


def test_supported_types_have_dataset_directories() -> None:
    missing = []
    for task_type in sorted(SUPPORTED_TYPES):
        if task_type in PHASE041_SIDECAR_ONLY_TYPES:
            continue
        dataset_dir = Path("captcha_data") / DATASET_DIR_ALIASES.get(task_type, task_type)
        if not dataset_dir.is_dir():
            missing.append((task_type, str(dataset_dir)))

    assert not missing


def test_connect_icon_alias() -> None:
    assert TASK_ALIASES["Connect_icon"] == "Connect_Icon"
    assert DATASET_DIR_ALIASES["Connect_Icon"] == "Connect_icon"
    assert (Path("captcha_data") / DATASET_DIR_ALIASES["Connect_Icon"]).is_dir()


def test_build_tasks_accepts_connect_icon_alias_and_canonical_name() -> None:
    alias_tasks = build_tasks("./captcha_data", ["Connect_icon"], max_per_type=1)
    canonical_tasks = build_tasks("./captcha_data", ["Connect_Icon"], max_per_type=1)

    assert alias_tasks
    assert canonical_tasks
    assert alias_tasks[0].type == "Connect_Icon"
    assert canonical_tasks[0].type == "Connect_Icon"


def test_prompt_keys_are_known() -> None:
    with open("prompts_optimized.yaml", "r", encoding="utf-8") as handle:
        prompts = yaml.safe_load(handle) or {}

    unknown = []
    for task_type in (prompts.get("types") or {}):
        canonical = _canonical_task_type(task_type)
        if canonical not in SUPPORTED_TYPES:
            unknown.append(task_type)

    assert not unknown


def test_few_shot_keys_are_known() -> None:
    with open("few_shot_examples.yaml", "r", encoding="utf-8") as handle:
        few_shot = yaml.safe_load(handle) or {}

    unknown = []
    for task_type in few_shot:
        if task_type in IGNORED_NOT_USED_KEYS:
            continue
        canonical = _canonical_task_type(task_type)
        if canonical not in SUPPORTED_TYPES:
            unknown.append(task_type)

    assert not unknown


def test_phase041_new_task_types_are_registered() -> None:
    assert {"Symbol_Count", "Relation_Match"} <= SUPPORTED_TYPES
    assert "Hole_Counting" in SUPPORTED_TYPES


def test_phase041_prompt_keys_are_present() -> None:
    with open("prompts_optimized.yaml", "r", encoding="utf-8") as handle:
        prompts = yaml.safe_load(handle) or {}

    prompt_types = prompts.get("types") or {}
    assert "Symbol_Count" in prompt_types
    assert "Relation_Match" in prompt_types
    assert "Hole_Counting" in prompt_types


def test_phase041_visualizer_task_families_are_mapped() -> None:
    assert CAPTCHAVisualizer.TASK_FAMILY["Symbol_Count"] == "Counting/Generalization"
    assert CAPTCHAVisualizer.TASK_FAMILY["Relation_Match"] == "Semantic Matching"
    assert CAPTCHAVisualizer.TASK_FAMILY["Hole_Counting"] == "Grid Selection"


def test_phase041_json_schema_answer_shapes() -> None:
    symbol_schema = build_json_schema("Symbol_Count")
    assert symbol_schema["properties"]["answer_type"]["enum"] == ["number"]
    assert symbol_schema["properties"]["value"]["type"] == "integer"
    assert symbol_schema["required"] == ["answer_type", "value"]

    relation_schema = build_json_schema("Relation_Match")
    assert relation_schema["properties"]["answer_type"]["enum"] == ["classify"]
    assert relation_schema["properties"]["index"]["type"] == "integer"
    assert relation_schema["required"] == ["answer_type", "index"]

    hole_schema = build_json_schema("Hole_Counting")
    assert hole_schema["properties"]["answer_type"]["enum"] == ["multi_select"]
    assert hole_schema["properties"]["indices"]["type"] == "array"
    assert hole_schema["required"] == ["answer_type", "indices"]


def test_phase041_build_tasks_loads_sidecar_static_categories(tmp_path: Path) -> None:
    symbol_dir = tmp_path / "Symbol_Count"
    symbol_dir.mkdir()
    (symbol_dir / "sample1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (symbol_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "sample1.png": {
                    "count": 7,
                    "prompt": "Count the target symbols.",
                }
            }
        ),
        encoding="utf-8",
    )

    relation_dir = tmp_path / "Relation_Match"
    relation_dir.mkdir()
    for image_name in ("ref.png", "opt0.png", "opt1.png"):
        (relation_dir / image_name).write_bytes(b"\x89PNG\r\n\x1a\n")
    (relation_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "sample2.json": {
                    "reference_image": "ref.png",
                    "option_images": ["opt0.png", "opt1.png"],
                    "correct_index": 1,
                    "prompt": "Choose the matching relation.",
                }
            }
        ),
        encoding="utf-8",
    )

    hole_dir = tmp_path / "Hole_Counting"
    hole_dir.mkdir()
    (hole_dir / "holes.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (hole_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "holes.png": {
                    "grid_size": [4, 4],
                    "target_holes": 2,
                    "correct_patches": [1, 3],
                    "prompt": "Select cells with two holes.",
                }
            }
        ),
        encoding="utf-8",
    )

    tasks = build_tasks(
        str(tmp_path),
        ["Symbol_Count", "Relation_Match", "Hole_Counting"],
        prompts_cfg={
            "types": {
                "Symbol_Count": "Return JSON only.",
                "Relation_Match": "Return JSON only.",
                "Hole_Counting": "Return JSON only.",
            }
        },
    )

    assert [task.type for task in tasks] == [
        "Symbol_Count",
        "Relation_Match",
        "Hole_Counting",
    ]
    assert tasks[0].images == [str(symbol_dir / "sample1.png")]
    assert tasks[0].gt == {"count": 7}
    assert tasks[1].images == [
        str(relation_dir / "ref.png"),
        str(relation_dir / "opt0.png"),
        str(relation_dir / "opt1.png"),
    ]
    assert tasks[1].gt == {"correct_index": 1, "num_options": 2}
    assert tasks[2].images == [str(hole_dir / "holes.png")]
    assert tasks[2].gt == {"grid_size": [4, 4], "correct_patches": [1, 3]}
