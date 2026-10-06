"""Recompute the frozen TDSC results locally, without model or network calls."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TDSC = ROOT / "tdsc"
sys.path.insert(0, str(TDSC / "src"))


def read_json(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_static_answers(output):
    from tdsc_eval.tasks import score

    report = TDSC / "reports/answer_20260913"
    reference = read_json(report / "scored_answers.json")
    reviews = read_json(report / "answer_reading.json")
    rebuilt = {}
    for group in ("a_gemini", "a_captchamind", "b_gemini", "b_captchamind"):
        module = "conference" if group.startswith("a_") else "external"
        manifest = read_json(TDSC / f"manifests/answer_20260913_{module}.json")
        tasks = {row["id"]: row for row in manifest}
        raw = records(TDSC / "results/answer_20260913" / group / "attempts.jsonl")
        saved = {row["id"]: row for row in reference[group]}
        if len(raw) != len(tasks) or {r["id"] for r in raw} != set(tasks):
            raise ValueError(f"Incomplete or duplicate raw answers: {group}")
        rebuilt[group] = []
        for original in raw:
            row = copy.deepcopy(original)
            task = tasks[row["id"]]
            review = reviews.get(group, {}).get(row["id"])
            if review:
                row["answer"] = review["answer"]
                row["answer_reading"] = review
            infra = row["status"] == "infrastructure_failure"
            row["pending_answer_reading"] = bool(row.get("extraction_error") and not review and not infra)
            row["original_scorer_correct"] = None if infra else score(task, row.get("answer"))
            row["source_valid"] = not task.get("source_annotation_issue")
            row["final_correct"] = row["original_scorer_correct"] if row["source_valid"] else None
            for field in ("answer", "pending_answer_reading", "original_scorer_correct", "source_valid", "final_correct"):
                if row.get(field) != saved[row["id"]].get(field):
                    raise ValueError(f"Static re-scoring differs: {group}/{row['id']}/{field}")
            rebuilt[group].append(row)
    destination = output / "scored_answers.json"
    destination.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=2) + "\n")
    return destination, sum(map(len, rebuilt.values()))


def compare_summary(actual, expected):
    ignored = {"generated_at", "inputs", "plots"}
    actual = {k: v for k, v in actual.items() if k not in ignored}
    expected = {k: v for k, v in expected.items() if k not in ignored}
    if actual != expected:
        changed = [key for key in sorted(set(actual) | set(expected)) if actual.get(key) != expected.get(key)]
        raise ValueError("Recomputed summary differs in: " + ", ".join(changed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "generated/tdsc")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == TDSC or TDSC in output.parents or output == ROOT / "conference" or ROOT / "conference" in output.parents:
        parser.error("Use a separate output directory so archived evidence stays unchanged.")
    output.mkdir(parents=True, exist_ok=True)
    static_path, static_count = verify_static_answers(output)
    report = TDSC / "reports/solver_defense_v1"
    env = dict(os.environ, PYTHONPATH=str(TDSC / "src"), MPLBACKEND="Agg", MPLCONFIGDIR=str(output / ".matplotlib"))
    command = [sys.executable, str(TDSC / "scripts/summarize_solver_defense.py"),
               "--static-scores", str(static_path),
               "--runs", str(report / "handoff/completed_remote"),
               "--answer-readings", str(report / "d_static/answer_readings.json"),
               "--output", str(output), "--repetitions", "10000", "--seed", "20260913"]
    if args.skip_plots:
        command.append("--skip-plots")
    subprocess.run(command, cwd=TDSC, env=env, check=True)
    actual = read_json(output / "summary.json")
    compare_summary(actual, read_json(report / "analysis/summary.json"))
    csvs = sorted((report / "analysis").glob("*.csv"))
    for path in csvs:
        with (output / path.name).open(newline="") as actual_file, path.open(newline="") as expected_file:
            equal = list(csv.DictReader(actual_file)) == list(csv.DictReader(expected_file))
        if not equal:
            raise ValueError(f"CSV differs: {path.name}")
    formal = [row for group in ("a_native", "b_native", "d_gemini", "d_static", "d_native")
              for row in records(report / "handoff/completed_remote" / group / "attempts.jsonl")]
    validation = {"complete": actual["complete"], "static_answers_rescored": static_count,
                  "new_formal_episodes": len(formal), "new_formal_model_calls": sum(r["model_calls"] for r in formal),
                  "scientific_summary_matches": True, "matching_csv_files": len(csvs),
                  "model_requests_made": 0, "bootstrap_repetitions": 10000, "bootstrap_seed": 20260913}
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
