"""Audited, paired defense tasks and order-independent visible-eye scoring."""
import argparse
import copy
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageFilter

from .tasks import ROOT, build_json_schema, load_json, write_json

DEV_IDS = (1, 7, 8, 23, 27, 30)
CONDITIONS = ("D0", "D1", "D2", "D3")
SCORER = "defense_v1_maximum_matching"
BLUR_RADIUS = 2.0


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _points(value):
    return isinstance(value, list) and all(
        isinstance(p, dict) and _number(p.get("x")) and _number(p.get("y")) for p in value
    )


def point_match_details(predicted, gold, radius=15, *, legacy=False):
    """Match valid standard point lists; legacy reproduces GT-order nearest greedy.

    The main matcher finds maximum cardinality in the within-radius bipartite
    graph. Distances include the boundary. No coordinates or ordering are repaired.
    """
    if not _number(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    if not _points(predicted) or not _points(gold):
        raise ValueError("points must be lists of finite x/y dictionaries")
    distances = [[math.hypot(g["x"] - p["x"], g["y"] - p["y"])
                  for p in predicted] for g in gold]
    if legacy:
        used = set()
        for row in distances:
            candidates = [i for i, distance in enumerate(row) if distance <= radius and i not in used]
            if candidates:
                # min retains the first prediction on equal-distance ties, as in the source.
                used.add(min(candidates, key=lambda i: row[i]))
        matched = len(used)
    else:
        owners = {}

        def augment(gold_index, visited):
            for pred_index, distance in enumerate(distances[gold_index]):
                if distance > radius or pred_index in visited:
                    continue
                visited.add(pred_index)
                if pred_index not in owners or augment(owners[pred_index], visited):
                    owners[pred_index] = gold_index
                    return True
            return False

        matched = sum(augment(i, set()) for i in range(len(gold)))
    return {"matched": matched, "missing_gt": len(gold) - matched,
            "extra_pred": len(predicted) - matched, "predicted_count": len(predicted),
            "gold_count": len(gold), "radius": radius,
            "matcher": "legacy_nearest_greedy" if legacy else "maximum_bipartite"}


def score_defense(row, answer, radius=15, legacy=False):
    """Score standard indices/points/value answers against the independent D labels."""
    if not isinstance(answer, dict):
        return False
    condition, gold = row["condition"], row["gt"]
    if condition in ("D0", "D3"):
        indices = answer.get("indices")
        layout = row.get("layout", {"rows": 2, "cols": 3})
        if not isinstance(indices, list) or not all(
            isinstance(i, int) and not isinstance(i, bool)
            and 0 <= i < layout["rows"] * layout["cols"] for i in indices
        ):
            return False
        return set(indices) == set(gold["indices"])
    if condition == "D2":
        value = answer.get("value")
        return _number(value) and value == int(value) and value == gold["value"]
    if condition == "D1":
        points = answer.get("points")
        if not _points(points):
            return False
        details = point_match_details(points, gold["points"], radius, legacy=legacy)
        return details["missing_gt"] == details["extra_pred"] == 0
    raise ValueError(f"Unknown defense condition: {condition}")


def _row(label, condition):
    base_id, target = label["base_id"], label["target"]
    blurred = condition == "D3"
    if condition in ("D0", "D3"):
        kind = "Select_Animal"
        prompt = f"Select all grid cells containing the {target}."
        gold = {"indices": label["indices"]}
    elif condition == "D1":
        kind = "Select_Animal_Optimized"
        prompt = (f"Identify all actually visible eyes of the {target}. "
                  "Include one point per visible eye; do not include hidden eyes. Order does not matter.")
        gold = {"points": label["points"]}
    else:
        kind = "Visible_Eye_Count"
        prompt = f"Count all actually visible eyes of the {target}. Do not count hidden eyes."
        gold = {"value": len(label["points"])}
    schema = build_json_schema("Dice_Count" if condition == "D2" else kind, include_reasoning=False)
    folder = "blur_radius2" if blurred else "original"
    return {
        "id": f"defense_v1/{condition}/{base_id}", "source": "conference_defense",
        "module": "D", "condition": condition, "base_id": base_id, "target": target,
        "type": kind, "puzzle_id": base_id + ".png",
        "images": [f"data/defense_v1/{folder}/{base_id}.png"], "image_sizes": [[1024, 1024]],
        "prompt": prompt, "gt": copy.deepcopy(gold), "schema": schema, "scorer": SCORER,
        "group_id": f"defense_v1/{base_id}", "partition": label["partition"],
        "split_evidence": "fixed_image_ids_before_new_model_calls",
        "training_exposure": "unknown; previously used conference material",
        "layout": {"rows": 2, "cols": 3, "main_image_index": 0, "grid_box": [0, 0, 1024, 1024]},
        "source_path": f"../captcha_data/Select_Animal/{base_id}.png",
        "label_provenance": "data/defense_v1/annotations.json",
        "image_transform": {"name": "GaussianBlur", "radius": BLUR_RADIUS, "unit": "px"}
        if blurred else {"name": "identity"},
    }


def prepare_defense(root=ROOT, source_root=None):
    """Materialize deterministic task copies from the completed visual label audit."""
    root = Path(root)
    source_root = Path(source_root) if source_root else root.parent / "captcha_data"
    labels = load_json(root / "data/defense_v1/annotations.json")
    if [r["base_id"] for r in labels] != [f"image{i}" for i in range(1, 31)]:
        raise ValueError("The frozen audit must contain exactly image1 through image30")
    base_dir = root / "data/defense_v1/original"
    blur_dir = root / "data/defense_v1/blur_radius2"
    base_dir.mkdir(parents=True, exist_ok=True)
    blur_dir.mkdir(parents=True, exist_ok=True)
    source_checks = []
    for label in labels:
        name = label["base_id"] + ".png"
        source = source_root / "Select_Animal" / name
        old_optimized = source_root / "Select_Animal_Optimized" / name
        bytes_equal = source.read_bytes() == old_optimized.read_bytes()
        with Image.open(source) as image, Image.open(old_optimized) as prior:
            if image.size != (1024, 1024):
                raise ValueError(f"Unexpected source dimensions: {name}: {image.size}")
            if image.size != prior.size or image.convert("RGBA").tobytes() != prior.convert("RGBA").tobytes():
                raise ValueError(f"Original and prior optimized image pixels differ: {name}")
            image.filter(ImageFilter.GaussianBlur(BLUR_RADIUS)).save(blur_dir / name)
        shutil.copyfile(source, base_dir / name)
        source_checks.append({"base_id": label["base_id"], "original_optimized_bytes_equal": bytes_equal,
                              "original_optimized_pixels_equal": True,
                              "journal_original_bytes_equal": source.read_bytes() == (base_dir / name).read_bytes(),
                              "size": [1024, 1024]})
    rows = [_row(label, condition) for label in labels if label["include"] for condition in CONDITIONS]
    for partition, suffix in (("development", "dev"), ("evaluation", "eval")):
        write_json(root / f"manifests/defense_v1_{suffix}.json", [r for r in rows if r["partition"] == partition])
    audit = {
        "version": "defense_v1", "audit_date": "2026-09-13",
        "method": "Actual full-image visual inspection of all 30 source images before new model requests; existing eye points retained where they identify a visible eye.",
        "source_grid_labels": "../captcha_data/Select_Animal/ground_truth.json",
        "source_eye_labels": "../captcha_data/Select_Animal_Optimized/ground_truth.json",
        "development_ids": [f"image{i}" for i in DEV_IDS],
        "candidate_base_images": 30, "included_base_images": len({r["base_id"] for r in rows}),
        "evaluation_base_images": len({r["base_id"] for r in rows if r["partition"] == "evaluation"}),
        "evaluation_rows": sum(r["partition"] == "evaluation" for r in rows),
        "excluded": [r for r in labels if not r["include"]],
        "changes": [r for r in labels if r["changes"]], "image_checks": source_checks,
        "visual_audit": labels,
        "blur": {"library": "Pillow.ImageFilter.GaussianBlur", "radius": BLUR_RADIUS, "unit": "px", "image_only": True},
        "matching_correction": "Maximum matching removes nearest-greedy order-dependent false negatives. The old nearest-greedy full-coverage success criterion is already monotone in radius; no nonmonotonicity claim is made.",
    }
    write_json(root / "reports/solver_defense_v1/defense_source_audit.json", audit)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    prepared = prepare_defense(args.root)
    print(json.dumps({"rows": len(prepared), "base_images": len({r["base_id"] for r in prepared})}))
