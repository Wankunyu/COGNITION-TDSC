"""Offline, fixed-manifest analysis of the A/B/D solver-defense experiment.

No model requests are made. Old A/B answers are read from the frozen derived
records; new D answers are rescored only against the audited D manifest. A
missing or unresolved observation is never silently counted as a solver failure.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOLVERS = ("G-V", "C-S", "C-I")
CONDITIONS = ("D0", "D1", "D2", "D3")
RADII = (15, 30, 45)
UNKNOWN_STATUSES = {
    "infrastructure_failure", "environment_failure", "environment_error",
    "pending", "pending_request", "unknown_request", "incomplete",
    "transport_failure", "needs_answer_review",
    "native_environment_failure", "adapted_environment_failure", "recovery_failure",
    "running", "run_call_budget_paused", "run_budget_paused",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def indexed(rows, label):
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ValueError(f"Duplicate item in {label}: {row['id']}")
        result[row["id"]] = row
    return result


def validate_record_identity(records, manifest, solver, label):
    tasks = indexed(manifest, label + " manifest")
    for record in records:
        if record["id"] not in tasks:
            raise ValueError(f"Unexpected {label} ID: {record['id']}")
        task = tasks[record["id"]]
        if record.get("solver", solver) != solver:
            raise ValueError(f"Wrong solver in {label}: {record['id']}")
        for field in ("type", "source", "condition", "base_id", "group_id", "gt"):
            if field in record and field in task and record[field] != task[field]:
                raise ValueError(f"Record/manifest {field} mismatch in {label}: {record['id']}")


def load_manifest(path, label, optional=False):
    if not Path(path).exists() and optional:
        return []
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{label} manifest must be a list")
    indexed(rows, label)
    if any(r.get("partition") not in (None, "evaluation", "eval", "test") for r in rows):
        raise ValueError(f"{label} contains development items")
    return rows


def outcome(record, *, old=False, full=False):
    """Return bool only for a resolved observation; preserve frozen blind readings."""
    if record is None:
        return None
    if record.get("pending_answer_reading"):
        return None
    if old:
        # Old raw status may still say needs_answer_review after a frozen reading.
        value = record.get("original_scorer_correct") if full else record.get("final_correct")
    else:
        if record.get("status") in UNKNOWN_STATUSES:
            return None
        value = record.get("correct")
        if full and value is None:
            value = record.get("original_scorer_correct")
    return value if isinstance(value, bool) else None


def is_source_valid(row):
    return not row.get("source_annotation_issue") and row.get("source_valid", True)


def measure(values):
    known = [v for v in values if isinstance(v, bool)]
    complete = len(known) == len(values) and bool(values)
    successes = sum(known)
    return {
        "planned": len(values), "resolved": len(known), "unresolved": len(values) - len(known),
        "correct_observed": successes, "complete": complete,
        "success_rate": successes / len(values) if complete else None,
        "success_rate_lower_bound": successes / len(values) if values else None,
        "success_rate_upper_bound": (successes + len(values) - len(known)) / len(values) if values else None,
    }


def interval(samples):
    if samples is None:
        return None
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def grouped_bootstrap(rows, value_vectors, *, reps=10000, seed=20260913):
    """Resample material groups within task type, sharing draws across conditions.

    Point estimates remain record-weighted within type; an entire material group
    is selected together, retaining all its records and all solver outcomes.
    """
    if not rows:
        return {}
    if any(len(v) != len(rows) for v in value_vectors.values()):
        raise ValueError("Bootstrap outcome vectors must align with all manifest rows")
    usable = {k: v for k, v in value_vectors.items() if all(isinstance(x, (bool, int, float)) and math.isfinite(x) for x in v)}
    if not usable:
        return {}
    keys = list(usable)
    rng = np.random.default_rng(seed)
    strata = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        strata[row["type"]][row.get("group_id", row["id"])].append(i)
    totals = np.zeros((reps, len(keys)))
    denominators = np.zeros(reps)
    macro = np.zeros_like(totals)
    output = {}
    for kind in sorted(strata):
        groups = list(strata[kind].values())
        sizes = np.array([len(g) for g in groups])
        sums = np.array([[sum(usable[key][i] for i in group) for key in keys] for group in groups], dtype=float)
        draws = rng.integers(0, len(groups), size=(reps, len(groups)))
        denominator = sizes[draws].sum(axis=1)
        numerator = sums[draws].sum(axis=1)
        rates = numerator / denominator[:, None]
        totals += numerator
        denominators += denominator
        macro += rates / len(strata)
        output[kind] = {key: rates[:, i] for i, key in enumerate(keys)}
    output["micro"] = {key: totals[:, i] / denominators for i, key in enumerate(keys)}
    output["macro"] = {key: macro[:, i] for i, key in enumerate(keys)}
    return output


def material_group_counts(rows):
    groups = defaultdict(set)
    for row in rows:
        groups[row["type"]].add(row.get("group_id", row["id"]))
    return {kind: len(values) for kind, values in sorted(groups.items())}


def ab_reported_interval(samples, group_counts, kind):
    """Withhold intervals that cannot represent between-material variation.

    This is a reporting restriction: the paired draws and the estimand stay
    unchanged, including every other task type's prespecified bootstrap.
    """
    selected = group_counts if kind in ("micro", "macro") else {kind: group_counts[kind]}
    insufficient = [task_type for task_type, count in selected.items() if count < 2]
    ci = None if insufficient else interval(samples)
    reason = "insufficient_material_groups" if insufficient else "unresolved_outcomes" if samples is None else None
    degenerate = ci is not None and ci[0] == ci[1]
    return ci, {
        "ci95_reason": reason,
        "ci95_insufficient_group_types": insufficient,
        "material_groups_by_type": selected,
        "ci95_degenerate": degenerate,
        "ci95_note": "Degenerate empirical bootstrap interval is not evidence of zero population uncertainty or equivalence." if degenerate else None,
    }


def paired_cells(left, right):
    cells = {"both_correct": 0, "left_only": 0, "right_only": 0, "both_incorrect": 0, "unresolved": 0}
    for a, b in zip(left, right):
        if not isinstance(a, bool) or not isinstance(b, bool):
            cells["unresolved"] += 1
        else:
            cells[{(True, True): "both_correct", (True, False): "left_only", (False, True): "right_only", (False, False): "both_incorrect"}[a, b]] += 1
    return cells


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def points(answer):
    if not isinstance(answer, dict) or not isinstance(answer.get("points"), list):
        return None
    parsed = []
    for point in answer["points"]:
        if isinstance(point, dict):
            pair = point.get("x"), point.get("y")
        elif isinstance(point, (tuple, list)) and len(point) == 2:
            pair = point
        else:
            return None
        if not all(number(v) for v in pair):
            return None
        parsed.append(tuple(float(v) for v in pair))
    return parsed


def matching_size(predicted, truth, radius):
    """Cardinality only, for interpretable missed/extra point diagnostics."""
    edges = [[j for j, t in enumerate(truth) if math.dist(p, t) <= radius] for p in predicted]
    assigned = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    return sum(augment(i, set()) for i in range(len(predicted)))


def eye_diagnostics(answer, gt, radius):
    predicted, truth = points(answer), points(gt)
    if truth is None:
        raise ValueError("Audited eye truth has no finite coordinate set")
    if predicted is None:
        return {"coordinates_readable": False, "ground_truth_count": len(truth), "predicted_count": None,
                "matched": None, "missed": None, "extra": None, "coverage": None,
                "pred_to_gt_nearest_mean_px": None, "gt_to_pred_nearest_mean_px": None}
    matched = matching_size(predicted, truth, radius)
    pred_nearest = [min(math.dist(p, t) for t in truth) for p in predicted] if truth else []
    gt_nearest = [min(math.dist(t, p) for p in predicted) for t in truth] if predicted else []
    return {
        "coordinates_readable": True, "ground_truth_count": len(truth), "predicted_count": len(predicted),
        "point_count_correct": len(predicted) == len(truth), "matched": matched,
        "missed": len(truth) - matched, "extra": len(predicted) - matched,
        "coverage": matched / len(truth) if truth else 1.0,
        "pred_to_gt_nearest_mean_px": mean(pred_nearest) if pred_nearest else None,
        "gt_to_pred_nearest_mean_px": mean(gt_nearest) if gt_nearest else None,
        "pred_to_gt_nearest_distances_px": pred_nearest, "gt_to_pred_nearest_distances_px": gt_nearest,
    }


def accepted_answer(record, solver):
    # A budget/protocol stop can leave an interpretable intermediate point set.
    # Its coordinates are diagnostic, but they cannot turn into a scored submit.
    if solver != "C-I":
        return True
    if "submitted" in record:
        return record["submitted"] is True
    return record.get("done") == 1 and record.get("status") not in UNKNOWN_STATUSES


def defense_outcome(row, record, solver, scorer, radius=15, legacy=False):
    resolved = outcome(record)
    if resolved is None:
        return None
    if not accepted_answer(record, solver):
        return False
    return bool(scorer(row, record.get("final_answer", record.get("answer")), radius=radius, legacy=legacy))


def get_resource(record, field):
    value = record.get(field)
    if value is None and isinstance(record.get("resources"), dict):
        value = record["resources"].get(field)
    return value if number(value) else None


def resource_summary(records, planned, label):
    result = {**label, "planned": planned, "saved": len(records), "status_counts": dict(Counter(r.get("status", "unspecified") for r in records)),
              "returned_models": sorted({r["returned_model"] for r in records if r.get("returned_model")})}
    result["usage_incomplete_ids"] = [r["id"] for r in records if r.get("usage_complete") is False]
    result["usage_incomplete_n"] = len(result["usage_incomplete_ids"])
    for field in ("model_calls", "input_tokens", "output_tokens", "thinking_tokens", "elapsed_seconds", "cost_usd"):
        values = [v for r in records if (v := get_resource(r, field)) is not None]
        result[field + "_known_n"] = len(values)
        result[field + "_sum"] = sum(values) if values else None
        result[field + "_mean"] = mean(values) if values else None
        result[field + "_median"] = median(values) if values else None
        result[field + "_p90"] = float(np.quantile(values, 0.9)) if values else None
    # elapsed_seconds is summed episode/request time, not parallel job makespan.
    result["elapsed_seconds_semantics"] = "sum and distribution of recorded per-episode wall time; not job makespan"
    return result


TERMINAL_CATEGORIES = (
    "submitted", "protocol_failure", "action_budget", "context_budget", "time_budget",
    "token_budget", "other_explicit_failure", "other_resolved_success", "unknown",
)


def terminal_category(record):
    """Classify the recorded terminal outcome, never an intermediate trajectory."""
    result = outcome(record)
    if result is None:
        return "unknown"
    if accepted_answer(record, "C-I"):
        return "submitted"
    reason = record.get("stop_reason", record.get("status"))
    budget_categories = {
        "protocol_failure": "protocol_failure",
        "action_budget_exhausted": "action_budget",
        "context_budget_exhausted": "context_budget",
        "episode_time_budget_exhausted": "time_budget",
        "output_budget_exhausted": "token_budget",
    }
    return budget_categories.get(reason, "other_explicit_failure" if result is False else "other_resolved_success")


def readable_final_state(row, answer, module):
    """Require a complete structural value without repairing it or checking GT."""
    if not isinstance(answer, dict):
        return False
    kind = row["type"]
    condition = row.get("condition") if module == "D" else None
    if kind == "Dice_Count" or condition == "D2":
        return number(answer.get("value"))
    if kind == "Rotation_Match":
        return number(answer.get("angle"))
    if kind in ("Place_Dot", "Pick_Area"):
        point = answer.get("point")
        return isinstance(point, dict) and number(point.get("x")) and number(point.get("y"))
    if kind == "Click_Order" or condition == "D1":
        value = answer.get("points")
        return isinstance(value, list) and all(isinstance(p, dict) and number(p.get("x")) and number(p.get("y")) for p in value)
    if kind == "Patch_Select" or condition in ("D0", "D3"):
        value = answer.get("indices")
        return isinstance(value, list) and all(isinstance(i, int) and not isinstance(i, bool) for i in value)
    return False


def analyze_interactive_states(module, manifest, records):
    """Separate episode outcome from last visible A/D answer-state diagnostics.

    B stays entirely under native reward semantics. No steps are read and no
    intermediate state is searched, selected, or substituted for submission.
    """
    by_id = indexed(records, module + "/C-I terminal states")
    validate_record_identity(records, manifest, "C-I", module + "/C-I terminal states")
    scorer = None
    if module == "A":
        from tdsc_eval.tasks import score
        scorer = score
    elif module == "D":
        from tdsc_eval.defense_tasks import score_defense
        scorer = score_defense
    items = []
    for row in manifest:
        record = by_id.get(row["id"])
        resolved = outcome(record) is not None
        answer = record.get("final_answer") if record else None
        readable = readable_final_state(row, answer, module) if scorer else None
        state_correct = bool(scorer(row, answer)) if resolved and readable else None
        submitted = accepted_answer(record, "C-I") if record else None
        items.append({
            "module": module, "type": row["type"], "condition": row.get("condition", "native_actions"),
            "solver": "C-I", "id": row["id"], "source_valid": is_source_valid(row),
            "source_annotation_issue": row.get("source_annotation_issue"),
            "status": record.get("status") if record else "missing", "terminal_category": terminal_category(record),
            "resolved": resolved, "episode_correct": outcome(record), "submitted": submitted,
            "final_state_readable": readable, "final_state_scored": state_correct is not None,
            "final_state_correct": state_correct,
            "unsubmitted_final_state_correct": state_correct is True and submitted is False,
            "state_diagnostic_scope": "last_saved_final_answer_only" if scorer else "not_applicable_source_native_B",
        })
    groups = defaultdict(list)
    for item in items:
        groups[(item["type"], item["condition"])].append(item)
        groups[("all", "all")].append(item)
    table = []
    for (kind, condition), group in sorted(groups.items()):
        counts = Counter(r["terminal_category"] for r in group)
        valid_hits = [r["id"] for r in group if r["unsubmitted_final_state_correct"] and r["source_valid"]]
        issue_hits = [r["id"] for r in group if r["unsubmitted_final_state_correct"] and not r["source_valid"]]
        table.append({
            "module": module, "type": kind, "condition": condition, "solver": "C-I", "planned": len(group),
            **{category: counts[category] for category in TERMINAL_CATEGORIES},
            "submitted_correct": sum(r["terminal_category"] == "submitted" and r["episode_correct"] is True for r in group),
            "submitted_incorrect": sum(r["terminal_category"] == "submitted" and r["episode_correct"] is False for r in group),
            "resolved_unsubmitted": sum(r["resolved"] and r["submitted"] is False for r in group),
            "final_state_scored": sum(r["final_state_scored"] for r in group),
            "unsubmitted_final_state_correct_source_valid": len(valid_hits),
            "unsubmitted_final_state_correct_source_valid_ids": valid_hits,
            "unsubmitted_final_state_correct_source_issue": len(issue_hits),
            "unsubmitted_final_state_correct_source_issue_ids": issue_hits,
            "source_issue_ids": [r["id"] for r in group if not r["source_valid"]],
        })
    return {"table": table, "items": items,
            "unsubmitted_final_state_correct": [r for r in items if r["unsubmitted_final_state_correct"]]}


def analyze_ab(module, manifest, records, *, reps, seed):
    tables, comparisons, items, resources = [], [], [], []
    by_solver = {}
    for solver, raw in records.items():
        by_solver[solver] = indexed(raw, f"{module}/{solver}")
        validate_record_identity(raw, manifest, solver, f"{module}/{solver}")
        resources.append(resource_summary(raw, len(manifest), {"module": module, "solver": solver, "condition": "all"}))
    scopes = ("source_valid", "full") if module == "A" else ("source_valid",)
    for scope in scopes:
        rows = [r for r in manifest if scope == "full" or is_source_valid(r)]
        vectors = {solver: [outcome(by_solver[solver].get(r["id"]), old=solver != "C-I", full=scope == "full") for r in rows] for solver in SOLVERS}
        bootstrap = grouped_bootstrap(rows, vectors, reps=reps, seed=seed)
        group_counts = material_group_counts(rows)
        kinds = sorted({r["type"] for r in rows})
        for solver in SOLVERS:
            typed = []
            for kind in kinds + ["micro", "macro"]:
                selected = [i for i, r in enumerate(rows) if kind in ("micro", "macro") or r["type"] == kind]
                values = [vectors[solver][i] for i in selected]
                metrics = measure(values)
                if kind == "macro":
                    metrics["success_rate"] = mean(t["success_rate"] for t in typed) if all(t["complete"] for t in typed) else None
                    metrics["success_rate_lower_bound"] = mean(t["success_rate_lower_bound"] for t in typed)
                    metrics["success_rate_upper_bound"] = mean(t["success_rate_upper_bound"] for t in typed)
                elif kind != "micro":
                    typed.append(metrics)
                ci, ci_metadata = ab_reported_interval(bootstrap.get(kind, {}).get(solver), group_counts, kind)
                tables.append({"module": module, "scope": scope, "type": kind, "solver": solver, **metrics,
                               "material_groups": len({(rows[i]["type"], rows[i].get("group_id", rows[i]["id"])) for i in selected}),
                               "ci95": ci, **ci_metadata})
        for left, right in (("C-I", "C-S"), ("G-V", "C-S"), ("G-V", "C-I")):
            for kind in kinds + ["micro", "macro"]:
                selected = [i for i, r in enumerate(rows) if kind in ("micro", "macro") or r["type"] == kind]
                lv, rv = ([vectors[s][i] for i in selected] for s in (left, right))
                cells = paired_cells(lv, rv)
                diff = None
                if not cells["unresolved"] and selected:
                    diff = mean(int(a) - int(b) for a, b in zip(lv, rv))
                    if kind == "macro":
                        diff = mean(mean(int(vectors[left][i]) - int(vectors[right][i]) for i, r in enumerate(rows) if r["type"] == t) for t in kinds)
                draws = bootstrap.get(kind, {})
                samples = draws[left] - draws[right] if left in draws and right in draws else None
                ci, ci_metadata = ab_reported_interval(samples, group_counts, kind)
                comparisons.append({"module": module, "scope": scope, "type": kind, "left": left, "right": right,
                                    "planned": len(selected), "difference_pp": 100 * diff if diff is not None else None,
                                    "ci95_pp": [100 * x for x in ci] if ci else None, **ci_metadata, **cells})
        if scope == "source_valid":
            for i, row in enumerate(rows):
                items.append({"module": module, "id": row["id"], "type": row["type"], "group_id": row.get("group_id"),
                              **{solver: vectors[solver][i] for solver in SOLVERS},
                              **{solver + "_status": by_solver[solver].get(row["id"], {}).get("status", "missing") for solver in SOLVERS}})
    excluded = [{"module": module, "id": r["id"], "source_annotation_issue": r.get("source_annotation_issue")} for r in manifest if not is_source_valid(r)]
    return {"table": tables, "comparisons": comparisons, "items": items, "resources": resources, "source_exclusions": excluded}


def analyze_defense(manifest, records, scorer, *, reps, seed):
    if not manifest:
        return {"complete": False, "reason": "D evaluation manifest unavailable", "table": [], "effects": [], "pairs": [], "diagnostics": [], "resources": [], "tolerance": [], "compatibility": []}
    tasks = defaultdict(dict)
    for row in manifest:
        base, condition = row["base_id"], row["condition"]
        if condition not in CONDITIONS or condition in tasks[base]:
            raise ValueError(f"Duplicate/unknown D condition: {base}/{condition}")
        tasks[base][condition] = row
    if any(set(v) != set(CONDITIONS) for v in tasks.values()):
        raise ValueError("Every audited D base image must contain all four fixed conditions")
    bases = sorted(tasks)
    bootstrap_rows = [{"id": b, "type": "D", "group_id": b} for b in bases]
    vectors, table, diagnostics, resources, compatibility = {}, [], [], [], []
    rows_by_solver = {}
    for solver in SOLVERS:
        raw = records[solver]
        by_id = indexed(raw, "D/" + solver)
        validate_record_identity(raw, manifest, solver, "D/" + solver)
        rows_by_solver[solver] = by_id
        for condition in CONDITIONS:
            selected = [by_id[r["id"]] for r in manifest if r["condition"] == condition and r["id"] in by_id]
            resources.append(resource_summary(selected, len(bases), {"module": "D", "solver": solver, "condition": condition}))
            vectors[f"{solver}/{condition}"] = [defense_outcome(tasks[b][condition], by_id.get(tasks[b][condition]["id"]), solver, scorer) for b in bases]
        for radius in RADII:
            vectors[f"{solver}/r{radius}"] = [defense_outcome(tasks[b]["D1"], by_id.get(tasks[b]["D1"]["id"]), solver, scorer, radius=radius) for b in bases]
        vectors[f"{solver}/legacy15"] = [defense_outcome(tasks[b]["D1"], by_id.get(tasks[b]["D1"]["id"]), solver, scorer, legacy=True) for b in bases]
        for i, base in enumerate(bases):
            task = tasks[base]["D1"]
            rec = by_id.get(task["id"])
            answer = rec.get("final_answer", rec.get("answer")) if rec else None
            for radius in RADII:
                diagnostics.append({"solver": solver, "base_id": base, "id": task["id"], "radius_px": radius,
                                    "status": rec.get("status") if rec else "missing",
                                    "resolved": outcome(rec) is not None, "submitted": accepted_answer(rec, solver) if rec else None,
                                    "correct": vectors[f"{solver}/r{radius}"][i], **eye_diagnostics(answer, task["gt"], radius)})
            compatibility.append({"solver": solver, "base_id": base, "id": task["id"],
                                  "maximum_matching_15": vectors[f"{solver}/D1"][i], "legacy_greedy_15": vectors[f"{solver}/legacy15"][i]})
    bootstrap = grouped_bootstrap(bootstrap_rows, vectors, reps=reps, seed=seed).get("micro", {})
    effects, pair_rows, tolerance = [], [], []
    for solver in SOLVERS:
        for condition in CONDITIONS:
            key = f"{solver}/{condition}"
            table.append({"module": "D", "solver": solver, "condition": condition, **measure(vectors[key]), "ci95": interval(bootstrap.get(key))})
        for radius in RADII:
            key = f"{solver}/r{radius}"
            tolerance.append({"solver": solver, "radius_px": radius, **measure(vectors[key]), "ci95": interval(bootstrap.get(key))})
        for label, left, right in (("defense_drop_D0_minus_D1", "D0", "D1"), ("blur_D3_minus_D0", "D3", "D0"), ("count_D2_minus_D1", "D2", "D1")):
            a, b = vectors[f"{solver}/{left}"], vectors[f"{solver}/{right}"]
            cells = paired_cells(a, b)
            diff = mean(int(x) - int(y) for x, y in zip(a, b)) if not cells["unresolved"] else None
            draws_a, draws_b = bootstrap.get(f"{solver}/{left}"), bootstrap.get(f"{solver}/{right}")
            ci = interval(draws_a - draws_b) if draws_a is not None and draws_b is not None else None
            effects.append({"solver": solver, "effect": label, "left": left, "right": right, "planned": len(bases),
                            "difference_pp": diff * 100 if diff is not None else None, "ci95_pp": [x * 100 for x in ci] if ci else None,
                            "left_successes_observed": sum(x is True for x in a), "right_successes_observed": sum(x is True for x in b), **cells})
        for i, base in enumerate(bases):
            pair_rows.append({"solver": solver, "base_id": base, **{c: vectors[f"{solver}/{c}"][i] for c in CONDITIONS},
                              "count_correct_eyes_wrong": vectors[f"{solver}/D2"][i] is True and vectors[f"{solver}/D1"][i] is False,
                              "all_resolved": all(vectors[f"{solver}/{c}"][i] is not None for c in CONDITIONS),
                              **{c + "_id": tasks[base][c]["id"] for c in CONDITIONS}})
    gamma_keys = ("C-S/D0", "C-S/D1", "C-I/D0", "C-I/D1")
    gamma_complete = all(all(v is not None for v in vectors[k]) for k in gamma_keys)
    gamma = mean(int(a) - int(b) - int(c) + int(d) for a, b, c, d in zip(*(vectors[k] for k in gamma_keys))) if gamma_complete else None
    gamma_ci = interval(bootstrap[gamma_keys[0]] - bootstrap[gamma_keys[1]] - bootstrap[gamma_keys[2]] + bootstrap[gamma_keys[3]]) if gamma_complete else None
    effects.append({"solver": "C-S versus C-I", "effect": "gamma_delta_CS_minus_delta_CI", "planned": len(bases),
                    "difference_pp": gamma * 100 if gamma is not None else None, "ci95_pp": [x * 100 for x in gamma_ci] if gamma_ci else None,
                    "complete": gamma_complete})
    diag_summary = []
    for solver in SOLVERS:
        selected = [r for r in diagnostics if r["solver"] == solver and r["radius_px"] == 15 and r["resolved"] and r["coordinates_readable"]]
        eligible = [r for r in selected if r["predicted_count"] and r["ground_truth_count"]]
        diag_summary.append({"solver": solver, "planned": len(bases), "resolved_readable_coordinate_sets": len(selected),
                             "point_count_correct": sum(r["point_count_correct"] for r in selected),
                             "missed_points_sum": sum(r["missed"] for r in selected), "extra_points_sum": sum(r["extra"] for r in selected),
                             "matched_points_sum": sum(r["matched"] for r in selected),
                             "distance_sample_n": len(eligible),
                             "pred_to_gt_nearest_mean_px": mean(r["pred_to_gt_nearest_mean_px"] for r in eligible) if eligible else None,
                             "gt_to_pred_nearest_mean_px": mean(r["gt_to_pred_nearest_mean_px"] for r in eligible) if eligible else None,
                             "legacy_disagreement_n": sum(r["maximum_matching_15"] != r["legacy_greedy_15"] for r in compatibility if r["solver"] == solver and r["maximum_matching_15"] is not None and r["legacy_greedy_15"] is not None)})
    return {"complete": all(t["complete"] for t in table), "base_images": len(bases), "table": table, "effects": effects,
            "pairs": pair_rows, "diagnostics": diagnostics, "diagnostic_summary": diag_summary, "resources": resources,
            "tolerance": tolerance, "compatibility": compatibility}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def pct(value):
    return "未完成" if value is None else f"{100 * value:.1f}%"


def ci_text(value, percent=True, reason=None):
    if value is None:
        return "不可估计：素材组不足" if reason == "insufficient_material_groups" else "未完成"
    factor = 100 if percent else 1
    return f"[{factor * value[0]:.1f}, {factor * value[1]:.1f}]"


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def write_report(path, report):
    lines = ["# 专用求解器与结构防御实验结果", "", f"生成时间：{report['generated_at']}。整体状态：**{'完成' if report['complete'] else '尚未完成'}**。", "",
             "A/B 复用冻结的两组静态答案；交互结果来自同一题目清单。未完成请求、环境问题和待盲读答案以未解析记录呈现；只有固定分母全部解析后才给出该组点估计。A/B 区间还要求涉及的每个题型至少有两个素材组。缺失不记为模型错误。", ""]
    ambiguity_failures = sum(r["status_counts"].get("ambiguous_answer_failure", 0) for r in report["resources"])
    if ambiguity_failures:
        lines.extend([f"按作者在完成盲读后明确指定的规则，{ambiguity_failures} 条已确认表达有歧义的完整响应计为失败，状态单列为 `ambiguous_answer_failure`。原回答、盲读证据和空的标准答案均保留；没有补造坐标。这些条目已定分，但不进入可读坐标的距离与点数诊断。该口径变更发生在正式生成之后，不改变未完成请求或基础设施故障的未知状态。", ""])
    for module in ("A", "B"):
        lines.extend([f"## {module}：三条件逐类结果", ""])
        table = [r for r in report["ab_table"] if r["module"] == module and r["scope"] == "source_valid"]
        lines.extend([markdown_table(["题型/汇总", "条件", "正确/固定分母", "已解析", "成功率", "95% 区间 (%)"],
                                    [[r["type"], r["solver"], f"{r['correct_observed']}/{r['planned']}", r["resolved"], pct(r["success_rate"]), ci_text(r["ci95"], reason=r["ci95_reason"])] for r in table]), ""])
        insufficient = {r["type"]: r["material_groups"] for r in table
                        if r["type"] not in ("micro", "macro") and r["ci95_reason"] == "insufficient_material_groups"}
        if insufficient:
            detail = "、".join(f"{kind}（{count} 个素材组）" for kind, count in sorted(insufficient.items()))
            lines.extend([f"{module} 的 {detail} 缺乏组间变异信息，保留成功率点估计和配对计数，但不报告该题型的区间。包含这些题型的 micro/macro 成功率及配对差值区间也置空，机器可读原因为 `insufficient_material_groups`，并列明受影响题型。其他题型继续使用同一预设素材组配对 bootstrap，不改用题级抽样。", ""])
    a_full = [r for r in report["ab_table"] if r["module"] == "A" and r["scope"] == "full" and r["type"] == "micro"]
    lines.extend(["A 主表按有效标注 140 条计算；原始 141 条视图及每类宏平均完整保留在 `ab_table.csv`。原始口径：" + "；".join(f"{r['solver']} {r['correct_observed']}/{r['planned']}（{pct(r['success_rate'])}）" for r in a_full) + "。", "",
                  "## D：同图防御与机制对照", ""])
    lines.extend([markdown_table(["条件", "求解方式", "正确/固定分母", "已解析", "成功率", "95% 区间 (%)"],
                                [[r["condition"], r["solver"], f"{r['correct_observed']}/{r['planned']}", r["resolved"], pct(r["success_rate"]), ci_text(r["ci95"])] for r in report["defense"]["table"]]), "",
                  markdown_table(["求解方式", "配对效应", "差值 (百分点)", "95% 区间"],
                                 [[r["solver"], r["effect"], "未完成" if r["difference_pp"] is None else f"{r['difference_pp']:.1f}", ci_text(r["ci95_pp"], False)] for r in report["defense"]["effects"]]), "",
                  markdown_table(["求解方式", "D0/D1 均成功", "仅 D0 成功", "仅 D1 成功", "均失败", "配对未解析"],
                                 [[r["solver"], r["both_correct"], r["left_only"], r["right_only"], r["both_incorrect"], r["unresolved"]]
                                  for r in report["defense"]["effects"] if r["effect"] == "defense_drop_D0_minus_D1"]), "",
                  "D0−D1 为防御降幅；Γ=(D0−D1)C-S−(D0−D1)C-I 为协议条件下的降幅差。Γ 为正表示交互条件下降幅较小。四格配对表包含原版失败而眼点成功的反例，D0 成功数用于判断基线地板效应。`defense_pairs.csv` 保存所有基础图，不能把四个条件视为更多独立素材。", "",
                  "D1 的主结果采用 15 px 最大二分匹配；30/45 px 与旧贪心 15 px 均来自同一次答案的离线重评。未提交的中间点集只进入明确标注的坐标诊断。`eye_diagnostics.csv` 给出点数、匹配数、遗漏与额外点；最近距离只在预测/标注均非空且坐标可解释时汇总，分母单独报告。D2 正确且 D1 失败是成分诊断，D3−D0 只描述固定 blur radius=2.0 px 控制。", "",
                  "## C-I 终止状态与最后答案状态", "",
                  markdown_table(["模块", "题型", "条件", "固定 N", "提交", "协议失败", "动作预算", "上下文预算", "时间预算", "输出预算", "其他明确失败", "未知"],
                                 [[r["module"], r["type"], r["condition"], r["planned"], r["submitted"], r["protocol_failure"], r["action_budget"], r["context_budget"], r["time_budget"], r["token_budget"], r["other_explicit_failure"], r["unknown"]]
                                  for r in report["interactive_terminal"]["table"]]), "",
                  "终止分类按固定清单统计；未知包括缺失记录、基础设施/环境异常和未完成请求，不记为明确失败。提交正确与提交错误的子计数、原始 status 和题目 ID 保存在 `interactive_terminal.csv` 与 `interactive_last_state.csv`。若出现已解析且未提交却被原记录记为成功的异常，另列 `other_resolved_success`，不混入失败。A 此表保留全部 141 条，来源标注有问题的题另有标记，不改变有效 140 条主成绩。", "",
                  markdown_table(["模块", "最后状态可评分", "未提交但最后状态命中：有效标注", "同类命中：来源标注有问题"],
                                 [[r["module"], r["final_state_scored"], r["unsubmitted_final_state_correct_source_valid"], r["unsubmitted_final_state_correct_source_issue"]]
                                  for r in report["interactive_terminal"]["table"] if r["type"] == "all"]), "",
                  "最后状态诊断仅对已解析且 `final_answer` 结构可读的 A/D 记录运行原评分器，A 使用 `tasks.score`，D 使用 `score_defense` 的 15 px 主规则。只看最后保存状态，不搜索轨迹或选择最佳步骤；未知记录没有状态正确性。`unsubmitted_final_state_matches.csv` 单列命中计数对应的 ID 和来源问题标记。未提交命中仍是 episode 失败，主正确率不变；B 不进行该项重评，尤其不把旋转候选选择送入静态角度评分器。", "",
                  "## 资源与完成检查", "",
                  markdown_table(["模块", "条件", "求解方式", "保存/计划", "调用", "已知输入 tokens", "已知输出 tokens", "累计题目时间 (s)", "已知费用 (USD)"],
                                 [[r["module"], r["condition"], r["solver"], f"{r['saved']}/{r['planned']}", r["model_calls_sum"], r["input_tokens_sum"], r["output_tokens_sum"], None if r["elapsed_seconds_sum"] is None else round(r["elapsed_seconds_sum"], 1), r["cost_usd_sum"]] for r in report["resources"]]), "",
                  "资源表的时间是逐题墙钟之和，不是并行作业总历时。输入/输出 token 汇总只使用完整已知的逐题用量，覆盖数见 resources.csv 的对应 known_n 字段。若单题时限已耗尽但最后响应未保存，该题仍可确定未在预算内完成；完整 token 用量保留未知，已收到部分的用量另存于记录的 observed_input_tokens/observed_output_tokens，相关 ID 见 usage_incomplete_ids。CaptchaMind 未知成本保留为空；Gemini thinking tokens 另列于机器可读资源表，不与可能已含 reasoning 的 CaptchaMind completion tokens 重复相加。", "",
                  f"统计采用 {report['bootstrap']['repetitions']:,} 次配对 bootstrap，seed={report['bootstrap']['seed']}。A/B 在题型内按素材组抽样，宏平均等权题型，微平均按记录数加权；D 按基础图绑定各求解器、条件和容差抽样。区间为 percentile 95% 区间。", "",
                  "即使至少有两个素材组，观察到的组级结果也可能没有变异，使经验 bootstrap 区间退化为单点。这不是总体零不确定性或条件等价的证据；保留原定方法与点区间，并在 A/B 机器可读结果中标记 `ci95_degenerate`。", "",
                  "这些比较对应完整求解协议，包含提示、表示、动作引起的新观察和计算额度，不能单独归因于反馈或训练。B 训练暴露未知；静态/交互的来源差异分层解释。实验未测量人类可用性、外部防御优越性或多次尝试能力。", ""])
    if report["issues"]:
        lines.extend(["未完成或需处理项：", ""] + ["- " + issue for issue in report["issues"]] + [""])
    Path(path).write_text("\n".join(lines))


def make_plots(directory, defense):
    if not defense["complete"]:
        return []
    os.environ.setdefault("MPLCONFIGDIR", str(directory / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    files = []
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for solver in SOLVERS:
        rows = [r for r in defense["tolerance"] if r["solver"] == solver]
        ax.plot([r["radius_px"] for r in rows], [100 * r["success_rate"] for r in rows], marker="o", label=solver)
        ax.fill_between([r["radius_px"] for r in rows], [100 * r["ci95"][0] for r in rows], [100 * r["ci95"][1] for r in rows], alpha=0.12)
    ax.set(xlabel="Matching radius (px)", ylabel="D1 success rate (%)", xticks=RADII, ylim=(-2, 102))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        name = "d1_tolerance." + ext
        fig.savefig(directory / name, dpi=300)
        files.append(name)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.2), sharey=True)
    for ax, solver in zip(axes, SOLVERS):
        pairs = [r for r in defense["pairs"] if r["solver"] == solver]
        cells = paired_cells([r["D0"] for r in pairs], [r["D1"] for r in pairs])
        for x in (0, 1):
            for y in (0, 1):
                label = {(1, 1): "both_correct", (1, 0): "left_only", (0, 1): "right_only", (0, 0): "both_incorrect"}[x, y]
                count = cells[label]
                if count:
                    color = {"both_correct": "#2ca02c", "left_only": "#ff7f0e", "right_only": "#9467bd", "both_incorrect": "#7f7f7f"}[label]
                    ax.plot([0, 1], [x, y], color=color, alpha=.55, linewidth=1 + 6 * count / len(pairs))
                    ax.text(.5, (x + y) / 2 + (0.08 if x > y else -.08), f"n={count}", ha="center", fontsize=9)
        ax.set(title=solver, xticks=[0, 1], xticklabels=["D0", "D1"], yticks=[0, 1], yticklabels=["Failed", "Solved"], ylim=(-.18, 1.18), xlim=(-.15, 1.15))
    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        name = "d0_d1_paired_transitions." + ext
        fig.savefig(directory / name, dpi=300)
        files.append(name)
    plt.close(fig)
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-scores", type=Path, default=ROOT / "reports/answer_20260913/scored_answers.json")
    parser.add_argument("--a-manifest", type=Path, default=ROOT / "manifests/answer_20260913_conference.json")
    parser.add_argument("--b-manifest", type=Path, default=ROOT / "manifests/answer_20260913_external.json")
    parser.add_argument("--d-manifest", type=Path, default=ROOT / "manifests/defense_v1_eval.json")
    parser.add_argument("--runs", type=Path, default=ROOT / "reports/solver_defense_v1")
    for group in ("a-native", "b-native", "d-gemini", "d-static", "d-native"):
        parser.add_argument("--" + group, type=Path, help="Run directory or attempts.jsonl; overrides --runs/<group>")
    parser.add_argument("--answer-readings", type=Path, help="Optional frozen D format-only readings, keyed by solver then item ID")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/solver_defense_v1/analysis")
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    old = read_json(args.static_scores)
    issues, sources, new_records = [], {}, {}
    readings = read_json(args.answer_readings) if args.answer_readings else {}
    for group in ("a_native", "b_native", "d_gemini", "d_static", "d_native"):
        path = getattr(args, group) or args.runs / group
        source = path if path.suffix == ".jsonl" else path / "attempts.jsonl"
        new_records[group] = read_records(source)
        sources[group] = str(source)
        pending = source.parent / "pending.json"
        if pending.exists() and read_json(pending):
            issues.append(f"{group} 有 pending 请求/episode，见 {pending}")
        if not source.exists():
            issues.append(f"{group} 尚无 attempts.jsonl")
    for group, solver in (("d_gemini", "G-V"), ("d_static", "C-S")):
        for record in new_records[group]:
            review = readings.get(solver, {}).get(record["id"])
            if review:
                ambiguous_failure = review.get("resolution") == "ambiguous_answer_failure"
                if ambiguous_failure and (record.get("status") != "needs_answer_review" or review.get("answer") is not None):
                    raise ValueError(f"Ambiguity failure requires a reviewed unparsed answer: {record['id']}")
                record["answer"] = review["answer"]
                record["final_answer"] = review["answer"]
                record["pending_answer_reading"] = False
                record["status"] = "ambiguous_answer_failure" if ambiguous_failure else "format_reading_resolved"
                # Only a resolved marker; the new D scorer determines correctness.
                record["correct"] = False
                record["answer_reading"] = review
    a_manifest = load_manifest(args.a_manifest, "A")
    b_manifest = load_manifest(args.b_manifest, "B")
    d_manifest = load_manifest(args.d_manifest, "D", optional=True)
    a = analyze_ab("A", a_manifest, {"G-V": old["a_gemini"], "C-S": old["a_captchamind"], "C-I": new_records["a_native"]}, reps=args.repetitions, seed=args.seed)
    b = analyze_ab("B", b_manifest, {"G-V": old["b_gemini"], "C-S": old["b_captchamind"], "C-I": new_records["b_native"]}, reps=args.repetitions, seed=args.seed)
    if d_manifest:
        from tdsc_eval.defense_tasks import score_defense
        d = analyze_defense(d_manifest, {"G-V": new_records["d_gemini"], "C-S": new_records["d_static"], "C-I": new_records["d_native"]}, score_defense, reps=args.repetitions, seed=args.seed)
    else:
        d = analyze_defense([], {}, None, reps=args.repetitions, seed=args.seed)
        issues.append("D 正式 manifest 尚未冻结，防御总体尚不可汇总")
    terminal_parts = [analyze_interactive_states(module, manifest, new_records[group])
                      for module, manifest, group in (("A", a_manifest, "a_native"),
                                                      ("B", b_manifest, "b_native"),
                                                      ("D", d_manifest, "d_native"))]
    terminal = {key: [row for part in terminal_parts for row in part[key]]
                for key in ("table", "items", "unsubmitted_final_state_correct")}
    incomplete = [r for r in a["table"] + b["table"] + d["table"] if not r["complete"]]
    for row in incomplete:
        if row.get("type") in ("macro", "micro"):
            continue
        identity = "/".join(str(v) for v in (row['module'], row.get('scope', row.get('condition')), row.get('type'), row['solver']) if v)
        issues.append(f"{identity} 未解析 {row['unresolved']}/{row['planned']} 条")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "complete": not incomplete and d["complete"] and not issues,
              "bootstrap": {"repetitions": args.repetitions, "seed": args.seed, "method": "type-stratified material-group paired percentile bootstrap"},
              "inputs": {"static_scores": str(args.static_scores), "a_manifest": str(args.a_manifest), "b_manifest": str(args.b_manifest), "d_manifest": str(args.d_manifest), "answer_readings": str(args.answer_readings) if args.answer_readings else None, **sources},
              "ab_table": a["table"] + b["table"], "ab_comparisons": a["comparisons"] + b["comparisons"], "ab_items": a["items"] + b["items"],
              "source_exclusions": a["source_exclusions"] + b["source_exclusions"], "defense": d,
              "resources": a["resources"] + b["resources"] + d["resources"], "interactive_terminal": terminal, "issues": issues}
    args.output.mkdir(parents=True, exist_ok=True)
    exports = {"ab_table": report["ab_table"], "ab_paired_comparisons": report["ab_comparisons"], "ab_paired_items": report["ab_items"],
               "defense_table": d["table"], "defense_effects": d["effects"], "defense_pairs": d["pairs"], "eye_diagnostics": d["diagnostics"],
               "d1_tolerance": d["tolerance"], "d1_legacy_compatibility": d["compatibility"], "resources": report["resources"],
               "interactive_terminal": terminal["table"], "interactive_last_state": terminal["items"],
               "unsubmitted_final_state_matches": terminal["unsubmitted_final_state_correct"]}
    for name, rows in exports.items():
        write_csv(args.output / (name + ".csv"), rows)
    report["plots"] = make_plots(args.output, d) if not args.skip_plots else []
    write_json(args.output / "summary.json", report)
    write_json(args.output / "case_index.json", {"ab_failures_or_solver_disagreements": [r for r in report["ab_items"] if not all(r[s] is True for s in SOLVERS)],
               "d_defense_transitions": [r for r in d["pairs"] if r["D0"] != r["D1"]], "d_count_correct_eyes_wrong": [r for r in d["pairs"] if r["count_correct_eyes_wrong"]],
               "d_unresolved": [r for r in d["pairs"] if not r["all_resolved"]],
               "unsubmitted_final_state_matches": terminal["unsubmitted_final_state_correct"]})
    write_report(args.output / "REPORT.md", report)
    print(json.dumps({"complete": report["complete"], "output": str(args.output), "unresolved_groups": len(incomplete), "issues": issues}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
