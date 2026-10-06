#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightweight experiment helper module with shared utilities for analysis,
error collection, and ground-truth loading. Compatible with cognition.run_eval.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class ErrorCase:
    """Container for a single failed prediction."""
    task_type: str
    puzzle_id: str
    prompt: str
    gt: Dict[str, Any]
    raw: str
    parsed: Optional[Dict[str, Any]]
    pass1: bool
    e2e_ms: float
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    error_description: Optional[str] = None
    reasoning: Optional[str] = None

class SimpleErrorCollector:
    """Collect failed cases and basic token statistics."""

    def __init__(self, experiment_name: str = "exp"):
        self.experiment_name = experiment_name
        self.errors: List[ErrorCase] = []
        self.stats = defaultdict(lambda: {"total": 0, "correct": 0, "tokens_in": 0, "tokens_out": 0})

    def record(self, task_type: str, puzzle_id: str, prompt: str, gt: Dict,
               raw: str, parsed: Optional[Dict], pass1: bool, meta: Dict,
               error_description: Optional[str] = None, reasoning: Optional[str] = None):
        """Record a single prediction outcome."""
        self.stats[task_type]["total"] += 1
        if pass1:
            self.stats[task_type]["correct"] += 1

        tokens_in = meta.get("tokens_in")
        tokens_out = meta.get("tokens_out")
        if tokens_in:
            self.stats[task_type]["tokens_in"] += tokens_in
        if tokens_out:
            self.stats[task_type]["tokens_out"] += tokens_out

        if not pass1:
            err_desc = (error_description or "").strip()
            if len(err_desc) > 300:
                err_desc = err_desc[:300]

            self.errors.append(ErrorCase(
                task_type=task_type,
                puzzle_id=puzzle_id,
                prompt=prompt[:200],
                gt=gt,
                raw=(raw or "")[:500],
                parsed=parsed,
                pass1=pass1,
                e2e_ms=meta.get("e2e_ms", 0),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                error_description=err_desc or None,
                reasoning=(reasoning or None)
            ))

    def save_summary(self, output_dir: str):
        """Persist errors and token statistics to disk."""
        os.makedirs(output_dir, exist_ok=True)

        errors_file = os.path.join(output_dir, "errors.csv")
        with open(errors_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "type",
                "puzzle_id",
                "raw_response",
                "parsed",
                "ground_truth",
                "error_description",
                "reasoning",
                "tokens_in",
                "tokens_out"
            ])
            for err in self.errors:
                parsed_without_reasoning = err.parsed.copy() if err.parsed else None
                if parsed_without_reasoning and "reasoning" in parsed_without_reasoning:
                    parsed_without_reasoning.pop("reasoning")

                writer.writerow([
                    err.task_type,
                    err.puzzle_id,
                    err.raw,
                    json.dumps(parsed_without_reasoning) if parsed_without_reasoning else "",
                    json.dumps(err.gt),
                    (err.error_description or ""),
                    err.reasoning or "",
                    err.tokens_in if err.tokens_in is not None else 0,
                    err.tokens_out if err.tokens_out is not None else 0
                ])

        stats_file = os.path.join(output_dir, "stats.json")
        stats_data = {
            "experiment": self.experiment_name,
            "by_task_type": {}
        }

        total_correct = 0
        total_cases = 0
        total_tokens_in = 0
        total_tokens_out = 0

        for task_type, data in self.stats.items():
            stats_data["by_task_type"][task_type] = {
                "total": data["total"],
                "correct": data["correct"],
                "pass1": data["correct"] / data["total"] if data["total"] > 0 else 0,
                "tokens_in": data["tokens_in"],
                "tokens_out": data["tokens_out"]
            }
            total_correct += data["correct"]
            total_cases += data["total"]
            total_tokens_in += data["tokens_in"]
            total_tokens_out += data["tokens_out"]

        stats_data["overall"] = {
            "total_cases": total_cases,
            "total_correct": total_correct,
            "overall_pass1": total_correct / total_cases if total_cases > 0 else 0,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_tokens": total_tokens_in + total_tokens_out
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)

        token_summary_file = os.path.join(output_dir, "token_summary.json")
        token_summary = {
            "experiment": self.experiment_name,
            "overall": {
                "total_questions": total_cases,
                "total_tokens_in": total_tokens_in,
                "total_tokens_out": total_tokens_out,
                "total_tokens": total_tokens_in + total_tokens_out
            },
            "by_task_type": {
                task_type: {
                    "count": data["total"],
                    "tokens_in": data["tokens_in"],
                    "tokens_out": data["tokens_out"],
                    "total_tokens": data["tokens_in"] + data["tokens_out"]
                }
                for task_type, data in sorted(self.stats.items())
            }
        }
        with open(token_summary_file, 'w', encoding='utf-8') as f:
            json.dump(token_summary, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved error analysis to: {output_dir}")
        print(f"   - Error cases: {errors_file}")
        print(f"   - Stats: {stats_file}")
        print(f"   - Token summary: {token_summary_file}")

def compare_experiments(exp1_dir: str, exp2_dir: str, exp3_dir: Optional[str] = None):
    """Print side-by-side comparison of three experiment outputs."""

    def load_stats(dir_path: str) -> Dict:
        stats_file = os.path.join(dir_path, "stats.json")
        if not os.path.exists(stats_file):
            return None
        with open(stats_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    exp1_stats = load_stats(exp1_dir)
    exp2_stats = load_stats(exp2_dir)
    exp3_stats = load_stats(exp3_dir) if exp3_dir else None

    if not exp1_stats or not exp2_stats:
        print("⚠️ Unable to load experiment stats")
        return

    print("\n" + "="*80)
    print("Experiment comparison")
    print("="*80)

    print("\n[Overall]")
    print(f"{'Metric':<20} {'Exp1 (GT)':<15} {'Exp2 (opt)':<15}", end="")
    if exp3_stats:
        print(f" {'Exp3 (iter)':<15}")
    else:
        print()

    print("-" * 80)

    exp1_overall = exp1_stats["overall"]
    exp2_overall = exp2_stats["overall"]

    print(f"{'Pass@1':<20} {exp1_overall['overall_pass1']:<15.2%} {exp2_overall['overall_pass1']:<15.2%}", end="")
    if exp3_stats:
        print(f" {exp3_stats['overall']['overall_pass1']:<15.2%}")
    else:
        print()

    print(f"{'Total tokens':<20} {exp1_overall['total_tokens']:<15,} {exp2_overall['total_tokens']:<15,}", end="")
    if exp3_stats:
        print(f" {exp3_stats['overall']['total_tokens']:<15,}")
    else:
        print()

    print("\n[Pass@1 by task type]")
    print(f"{'Task type':<20} {'Exp1':<12} {'Exp2':<12} {'Delta':<10}")
    print("-" * 80)

    for task_type in sorted(exp1_stats["by_task_type"].keys()):
        exp1_pass1 = exp1_stats["by_task_type"][task_type]["pass1"]
        exp2_pass1 = exp2_stats["by_task_type"].get(task_type, {}).get("pass1", 0)
        improvement = exp2_pass1 - exp1_pass1

        print(f"{task_type:<20} {exp1_pass1:<12.2%} {exp2_pass1:<12.2%} {improvement:+.2%}")

    print("="*80)

def load_tasks_from_ground_truth(dataset_root: str, task_types: List[str], max_per_type: int = 15) -> List[Dict]:
    """
    Load tasks from ground_truth.json entries into a unified task list.

    Returns a list of dicts with keys: type, puzzle_id, images, prompt, gt.
    """
    tasks = []

    for task_type in task_types:
        type_dir = Path(dataset_root) / task_type
        gt_file = type_dir / "ground_truth.json"

        if not gt_file.exists():
            print(f"⚠️ Skipping {task_type}: ground_truth.json not found")
            continue

        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️ {task_type}/ground_truth.json is malformed: {e}")
            continue

        count = 0
        for img_filename, data in all_data.items():
            if count >= max_per_type:
                break

            images = []
            img_path = str(type_dir / img_filename)
            if os.path.exists(img_path):
                images.append(img_path)

            if "order_image" in data:
                order_img = str(type_dir / data["order_image"])
                if os.path.exists(order_img):
                    images.append(order_img)

            gt = _normalize_ground_truth(task_type, data)

            tasks.append({
                "type": task_type,
                "puzzle_id": img_filename.replace(".png", "").replace(".jpg", ""),
                "images": images,
                "prompt": data.get("prompt", ""),
                "gt": gt
            })
            count += 1

    return tasks

def _normalize_ground_truth(task_type: str, data: Dict) -> Dict:
    """Normalize heterogeneous ground-truth formats into a common shape."""
    if task_type == "Dice_Count":
        return {"sum": data.get("sum")}

    elif task_type == "Click_Order":
        return {
            "points_gt": [{"x": p[0], "y": p[1]} for p in data.get("answer", [])],
            "tolerance": data.get("tolerance", 40.0)
        }

    elif task_type == "Patch_Select":
        return {"indices_gt": data.get("correct_patches", data.get("answer", []))}

    elif task_type in ("Place_Dot", "Geometry_Click"):
        target = data.get("target_position", data.get("answer"))
        if isinstance(target, list) and len(target) == 2:
            return {
                "target_position": {"x": target[0], "y": target[1]},
                "tolerance": data.get("tolerance", 15.0)
            }
        elif isinstance(target, dict):
            return {
                "target_position": {"x": target.get("x", 0), "y": target.get("y", 0)},
                "tolerance": data.get("tolerance", 15.0)
            }
        else:
            return {"target_position": {"x": 0, "y": 0}, "tolerance": 15.0}

    elif task_type == "Select_Animal_Optimized":
        points = data.get("target_position", data.get("targets_position", data.get("answer", [])))
        if isinstance(points, dict):
            points = [points]
        norm_points = []
        for p in points or []:
            if isinstance(p, dict) and "x" in p and "y" in p:
                norm_points.append({"x": p["x"], "y": p["y"]})
            elif isinstance(p, list) and len(p) == 2:
                norm_points.append({"x": p[0], "y": p[1]})
        return {
            "targets_positions": norm_points,
            "tolerance": data.get("tolerance", 15.0)
        }

    else:
        return data.get("answer", {})

__all__ = [
    'ErrorCase',
    'SimpleErrorCollector',
    'compare_experiments',
    'load_tasks_from_ground_truth',
]

if __name__ == "__main__":
    print("Helper module loaded.")
    print("Available functions:")
    print("  - SimpleErrorCollector: basic error collector")
    print("  - compare_experiments: compare experiment outputs")
    print("  - load_tasks_from_ground_truth: load tasks from ground_truth.json")
