"""Frozen conference loading, independent manifests, and source-native scoring."""
import json
import math
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from ._conference.run_eval import (
    TaskItem,
    _load_prompts_yaml,
    build_json_schema,
    build_tasks,
    evaluate_pass1,
)

ROOT = Path(__file__).resolve().parents[2]
TYPES = ("Dice_Count", "Click_Order", "Place_Dot", "Pick_Area", "Patch_Select", "Rotation_Match")


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def prepare_conference():
    random.seed(20260911)
    tasks = build_tasks(
        str(ROOT / "data/conference"), list(TYPES), max_per_type=None,
        prompts_cfg=_load_prompts_yaml(str(ROOT / "configs/conference_exp2_prompts.yaml")),
        prompt_mode="auto",
    )
    rows = []
    for task in sorted(tasks, key=lambda t: (TYPES.index(t.type), t.puzzle_id)):
        image_paths = [str(Path(p).relative_to(ROOT)) for p in task.images]
        group = f"conference:{task.type}:{task.puzzle_id}"
        if task.type == "Rotation_Match":
            group = f"conference:Rotation_Match:{Path(task.images[-1]).name}"
        rows.append({
            "id": f"conference/{task.type}/{task.puzzle_id}",
            "source": "conference", "type": task.type, "puzzle_id": task.puzzle_id,
            "images": image_paths,
            "image_sizes": [list(Image.open(p).size) for p in task.images],
            "prompt": task.prompt, "gt": task.gt,
            "schema": build_json_schema(task.type, include_reasoning=False),
            "scorer": "conference_v1", "group_id": group,
            "split_evidence": "historical_benchmark", "partition": "evaluation",
        })
    write_json(ROOT / "manifests/conference.json", rows)
    counts = dict(Counter(r["type"] for r in rows))
    write_json(ROOT / "reports/conference_inventory.json", {
        "records": len(rows), "by_type": counts,
        "material_groups": len({r["group_id"] for r in rows}),
    })
    return rows


def prompt_for(row):
    sizes = "; ".join(f"image {i + 1}: {w}x{h} pixels" for i, (w, h) in enumerate(row["image_sizes"]))
    schema = json.dumps(row["schema"], separators=(",", ":"))
    return (
        row["prompt"] + "\n\nInput dimensions: " + sizes
        + "\nReturn one complete final JSON answer matching this schema: " + schema
        + "\nDo not return tool calls or explanations."
    )


def parse_answer(text, row):
    """Extract one full JSON answer; never repair it with another model."""
    cleaned = text.strip()
    # CaptchaMind may wrap a final answer in its learned tool-call delimiters.
    # Only unwrap one complete envelope; the normal answer schema still applies.
    if (cleaned.startswith("<tool_call>") and cleaned.endswith("</tool_call>")
            and cleaned.count("<tool_call>") == cleaned.count("</tool_call>") == 1):
        cleaned = cleaned[len("<tool_call>"):-len("</tool_call>")].strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(cleaned, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError):
        return None, "invalid_json"
    if not isinstance(parsed, dict):
        return None, "not_object"
    schema = row["schema"]
    if any(k not in parsed for k in schema.get("required", [])):
        return None, "missing_field"
    expected = schema.get("properties", {}).get("answer_type", {}).get("enum", [])
    if expected and parsed.get("answer_type") not in expected:
        return None, "wrong_answer_type"

    def number(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

    def point(p):
        return isinstance(p, dict) and number(p.get("x")) and number(p.get("y"))

    kind = row["type"]
    if kind == "Dice_Count" and (not number(parsed.get("value")) or int(parsed["value"]) != parsed["value"]):
        return None, "non_integer_count"
    if kind in ("Place_Dot", "Pick_Area") and not point(parsed.get("point")):
        return None, "invalid_point"
    if kind == "Click_Order" and (
        not isinstance(parsed.get("points"), list) or not all(point(p) for p in parsed["points"])
    ):
        return None, "invalid_points"
    if kind == "Patch_Select":
        indices = parsed.get("indices")
        rows, cols = row["gt"]["grid_size"]
        if not isinstance(indices, list) or not all(
            isinstance(i, int) and not isinstance(i, bool) and 0 <= i < rows * cols for i in indices
        ):
            return None, "invalid_indices"
    if kind == "Rotation_Match" and not number(parsed.get("angle")):
        return None, "invalid_angle"
    return parsed, None


def score(row, answer):
    if answer is None:
        return False
    if row["scorer"] == "ordered_boxes_v1":
        points, boxes = answer.get("points", []), row["gt"]["ordered_boxes"]
        return len(points) == len(boxes) and all(
            b[0] <= p["x"] <= b[2] and b[1] <= p["y"] <= b[3]
            for p, b in zip(points, boxes)
        )
    task = TaskItem(row["type"], row["puzzle_id"], row["prompt"], row["images"], row["gt"])
    return bool(evaluate_pass1(task, answer))


def failure_label(row, answer):
    if row["type"] == "Dice_Count":
        return "incorrect_sum"
    if row["type"] == "Click_Order":
        gold = row["gt"].get("ordered_boxes", row["gt"].get("points_gt", []))
        if len(answer["points"]) != len(gold):
            return "wrong_point_count"
        return "ordered_localization_error"
    if row["type"] in ("Place_Dot", "Pick_Area"):
        return "outside_accepted_region"
    if row["type"] == "Patch_Select":
        return "incorrect_patch_set"
    return "incorrect_angle"
