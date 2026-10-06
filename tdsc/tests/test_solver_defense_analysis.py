"""Scientific invariants of the fixed-manifest offline analysis."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tdsc_eval.defense_tasks import score_defense

spec = importlib.util.spec_from_file_location("solver_defense_analysis", Path(__file__).resolve().parents[1] / "scripts/summarize_solver_defense.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_frozen_blind_reading_outranks_old_raw_status():
    record = {"status": "needs_answer_review", "correct": False, "final_correct": True,
              "original_scorer_correct": True, "pending_answer_reading": False}
    assert analysis.outcome(record, old=True) is True
    assert analysis.outcome(record) is None
    record["pending_answer_reading"] = True
    assert analysis.outcome(record, old=True) is None


def test_missing_infrastructure_not_failure_denominator():
    rows = [None, {"status": "infrastructure_failure", "correct": False},
            {"status": "output_budget_exhausted", "correct": False}, {"status": "correct", "correct": True}]
    result = analysis.measure([analysis.outcome(r) for r in rows])
    assert result == {"planned": 4, "resolved": 2, "unresolved": 2, "correct_observed": 1, "complete": False,
                      "success_rate": None, "success_rate_lower_bound": .25, "success_rate_upper_bound": .75}


def test_deadline_failure_is_resolved_but_missing_usage_is_not_invented():
    record = {"id": "timeout", "status": "episode_time_budget_exhausted", "correct": False,
              "submitted": False, "usage_complete": False, "input_tokens": None,
              "output_tokens": None, "observed_input_tokens": 900, "observed_output_tokens": 20,
              "model_calls": 23}
    assert analysis.outcome(record) is False
    assert analysis.terminal_category(record) == "time_budget"
    summary = analysis.resource_summary([record], 1, {"module": "A"})
    assert summary["input_tokens_known_n"] == 0 and summary["input_tokens_sum"] is None
    assert summary["model_calls_sum"] == 23 and summary["usage_incomplete_ids"] == ["timeout"]


def test_group_bootstrap_binds_material_and_solver_conditions():
    # Two perfectly opposed material groups; duplicating records within BOTH
    # groups must not reduce cluster uncertainty or unpair the solver outcomes.
    rows = [{"id": "a", "type": "rotation", "group_id": "a"}, {"id": "b", "type": "rotation", "group_id": "b"}]
    first = analysis.grouped_bootstrap(rows, {"x": [0, 1], "y": [1, 0]}, reps=1000, seed=7)
    duplicated = [dict(row, id=row["id"] + str(j)) for row in rows for j in range(8)]
    second = analysis.grouped_bootstrap(duplicated, {"x": [0] * 8 + [1] * 8, "y": [1] * 8 + [0] * 8}, reps=1000, seed=7)
    assert np.array_equal(first["micro"]["x"], second["micro"]["x"])
    assert np.array_equal(first["micro"]["x"] + first["micro"]["y"], np.ones(1000))
    assert analysis.interval(first["micro"]["x"]) == [0, 1]


def test_macro_equal_type_micro_equal_record_weight():
    rows = [{"id": str(i), "type": "many", "group_id": "one"} for i in range(9)] + [{"id": "last", "type": "few", "group_id": "two"}]
    result = analysis.grouped_bootstrap(rows, {"x": [1] * 9 + [0]}, reps=20)
    assert result["macro"]["x"].tolist() == [.5] * 20
    assert result["micro"]["x"].tolist() == [.9] * 20


def d_fixture():
    manifest, records = [], {s: [] for s in analysis.SOLVERS}
    for base in ("image1", "image2"):
        for condition in analysis.CONDITIONS:
            gt = {"points": [{"x": 100, "y": 100}]} if condition == "D1" else {"value": 1} if condition == "D2" else {"indices": [2]}
            row = {"id": f"defense_v1/{condition}/{base}", "base_id": base, "type": "eyes", "condition": condition, "gt": gt}
            manifest.append(row)
            for solver in analysis.SOLVERS:
                records[solver].append({"id": row["id"], "correct": True, "answer": copy.deepcopy(gt), "status": "correct", "submitted": True, "done": 1})
    return manifest, records


def test_unsubmitted_intermediate_set_never_becomes_radius_success():
    row = {"condition": "D1", "gt": {"points": [{"x": 100, "y": 100}]}}
    record = {"correct": False, "submitted": False, "status": "episode_budget_exhausted", "final_answer": row["gt"]}
    for radius in (15, 30, 45):
        assert analysis.defense_outcome(row, record, "C-I", score_defense, radius) is False
    assert analysis.eye_diagnostics(record["final_answer"], row["gt"], 15)["matched"] == 1
    record["submitted"] = True
    record["status"] = "correct"
    record["correct"] = True
    assert analysis.defense_outcome(row, record, "C-I", score_defense) is True


def test_d_full_pairing_and_gamma_not_complete_case_deletion():
    manifest, records = d_fixture()
    # C-S drops by 50 pp; C-I drops by 0 pp. All quantities use the two base images.
    records["C-S"][1]["answer"] = {"points": []}
    result = analysis.analyze_defense(manifest, records, score_defense, reps=50, seed=3)
    gamma = next(r for r in result["effects"] if r["effect"].startswith("gamma"))
    assert gamma["difference_pp"] == 50
    assert result["complete"]
    records["C-I"].pop()  # D3 missing keeps D0/D1 effect valid, but full study incomplete.
    result = analysis.analyze_defense(manifest, records, score_defense, reps=50, seed=3)
    assert not result["complete"]
    assert next(r for r in result["table"] if r["solver"] == "C-I" and r["condition"] == "D3")["success_rate"] is None
    records["C-I"].pop(1)  # D1 missing means gamma cannot use a one-image subset.
    result = analysis.analyze_defense(manifest, records, score_defense, reps=50, seed=3)
    assert next(r for r in result["effects"] if r["effect"].startswith("gamma"))["difference_pp"] is None


def test_d_manifest_requires_whole_paired_condition_set():
    manifest, records = d_fixture()
    with pytest.raises(ValueError, match="all four"):
        analysis.analyze_defense(manifest[:-1], records, score_defense, reps=10, seed=1)


def test_reviewed_ambiguity_is_failure_without_inventing_coordinate_diagnostics():
    manifest, records = d_fixture()
    eye_records = [r for r in records["C-S"] if "/D1/" in r["id"]]
    for record in eye_records:
        record.update(answer=None, correct=False, status="ambiguous_answer_failure")
    eye_records[1]["status"] = "needs_answer_review"
    result = analysis.analyze_defense(manifest, records, score_defense, reps=20, seed=3)
    row = next(r for r in result["table"] if r["solver"] == "C-S" and r["condition"] == "D1")
    assert (row["resolved"], row["unresolved"], row["correct_observed"]) == (1, 1, 0)
    assert row["success_rate"] is None
    eye_records[1]["status"] = "ambiguous_answer_failure"
    result = analysis.analyze_defense(manifest, records, score_defense, reps=20, seed=3)
    assert result["complete"]
    for row in result["tolerance"]:
        if row["solver"] == "C-S":
            assert row["success_rate"] == 0 and row["unresolved"] == 0
    diagnostic = next(r for r in result["diagnostic_summary"] if r["solver"] == "C-S")
    assert diagnostic["resolved_readable_coordinate_sets"] == 0
    assert diagnostic["distance_sample_n"] == 0


def test_run_directory_cannot_silently_relabel_another_solver_or_labels():
    row = {"id": "same", "gt": {"value": 1}}
    with pytest.raises(ValueError, match="Wrong solver"):
        analysis.validate_record_identity([{"id": "same", "solver": "C-S"}], [row], "G-V", "D/G-V")
    with pytest.raises(ValueError, match="gt mismatch"):
        analysis.validate_record_identity([{"id": "same", "gt": {"value": 2}}], [row], "G-V", "D/G-V")


def test_point_diagnostics_use_maximum_matching_and_explicit_distance_denominator():
    gt = {"points": [{"x": 100, "y": 100}, {"x": 120, "y": 100}]}
    pred = {"points": [{"x": 110, "y": 100}, {"x": 80, "y": 100}]}
    details = analysis.eye_diagnostics(pred, gt, 30)
    assert details["matched"] == 2
    assert details["missed"] == details["extra"] == 0
    assert score_defense({"condition": "D1", "gt": gt}, pred, radius=30, legacy=True) is False
    assert analysis.eye_diagnostics(None, gt, 15)["matched"] is None
    empty = analysis.eye_diagnostics({"points": []}, gt, 15)
    assert empty["missed"] == 2
    assert empty["gt_to_pred_nearest_mean_px"] is None
    assert analysis.points({"points": [{"x": float("nan"), "y": 0}]}) is None


def test_source_invalid_item_preserved_only_in_full_a_view():
    rows = [{"id": "good", "type": "Click_Order"}, {"id": "bad", "type": "Click_Order", "source_annotation_issue": "known label defect"}]
    old = [{"id": r["id"], "final_correct": None if r["id"] == "bad" else True, "original_scorer_correct": True} for r in rows]
    native = [{"id": r["id"], "correct": True, "status": "correct"} for r in rows]
    result = analysis.analyze_ab("A", rows, {"G-V": old, "C-S": old, "C-I": native}, reps=10, seed=1)
    valid = [r for r in result["table"] if r["scope"] == "source_valid"]
    full = [r for r in result["table"] if r["scope"] == "full"]
    assert all(r["planned"] == 1 for r in valid)
    assert all(r["planned"] == 2 for r in full)
    assert result["source_exclusions"][0]["id"] == "bad"
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.indexed(old + old, "old")


@pytest.mark.parametrize("module", ["A", "D"])
def test_last_state_match_is_diagnostic_only_and_does_not_search_steps(module):
    task = {"id": "last", "type": "Dice_Count", "puzzle_id": "last", "prompt": "Count", "images": [],
            "scorer": "conference_v1", "gt": {"sum": 9}}
    if module == "D":
        task.update(type="Visible_Eye_Count", condition="D2", gt={"value": 9})
    answer = {"answer_type": "number", "value": 9}
    record = {"id": task["id"], "status": "action_budget_exhausted", "correct": False,
              "submitted": False, "final_answer": answer}
    wrong_last = dict(record, id="earlier_only", final_answer={"answer_type": "number", "value": 8},
                      steps=[{"final_answer": answer}])
    tasks, records = [task, dict(task, id="earlier_only")], [record, wrong_last]
    if module == "A":
        tasks.append(dict(task, id="source_issue", source_annotation_issue="known source label problem"))
        records.append(dict(record, id="source_issue"))
    before = copy.deepcopy(records)
    result = analysis.analyze_interactive_states(module, tasks, records)
    total = next(r for r in result["table"] if r["type"] == "all")
    assert total["unsubmitted_final_state_correct_source_valid_ids"] == ["last"]
    assert total["unsubmitted_final_state_correct_source_issue_ids"] == (["source_issue"] if module == "A" else [])
    assert next(r for r in result["items"] if r["id"] == "earlier_only")["final_state_correct"] is False
    assert records == before
    assert all(analysis.outcome(r) is False for r in records)
    if module == "D":
        assert analysis.defense_outcome(task, record, "C-I", score_defense) is False


def test_terminal_unknown_is_not_explicit_failure_or_state_correctness():
    reasons = ["protocol_failure", "action_budget_exhausted", "context_budget_exhausted",
               "episode_time_budget_exhausted", "output_budget_exhausted", "incorrect", "pending_request", "missing"]
    tasks = [{"id": reason, "type": "Visible_Eye_Count", "condition": "D2", "gt": {"value": 1}} for reason in reasons]
    records = [{"id": reason, "status": reason, "correct": False, "submitted": False, "final_answer": {"value": 1}}
               for reason in reasons if reason != "missing"]
    result = analysis.analyze_interactive_states("D", tasks, records)
    total = next(r for r in result["table"] if r["type"] == "all")
    assert total["unknown"] == 2
    assert total["other_explicit_failure"] == 1
    assert sum(total[k] for k in analysis.TERMINAL_CATEGORIES) == len(tasks)
    assert all(total[k] == 1 for k in ("protocol_failure", "action_budget", "context_budget", "time_budget", "token_budget"))
    unknown = [r for r in result["items"] if r["terminal_category"] == "unknown"]
    assert all(r["final_state_correct"] is None and not r["final_state_scored"] for r in unknown)
    assert total["unsubmitted_final_state_correct_source_valid"] == 6


def test_native_rotation_never_uses_static_final_state_scorer(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Source-native B must never enter a static scorer")

    monkeypatch.setattr("tdsc_eval.tasks.score", forbidden)
    monkeypatch.setattr("tdsc_eval.defense_tasks.score_defense", forbidden)
    row = {"id": "native_rotation", "type": "Rotation_Match"}
    record = {"id": row["id"], "status": "correct", "correct": True, "submitted": True,
              "final_answer": {"answer_type": "candidate_selection", "image_index_zero_based": 2}}
    result = analysis.analyze_interactive_states("B", [row], [record])
    assert result["items"][0]["episode_correct"] is True
    assert result["items"][0]["final_state_correct"] is None
    assert result["items"][0]["state_diagnostic_scope"] == "not_applicable_source_native_B"


def test_single_material_type_suppresses_its_and_aggregate_intervals_only():
    rows = [{"id": "s0", "type": "single", "group_id": "same"},
            {"id": "s1", "type": "single", "group_id": "same"},
            {"id": "m0", "type": "multi", "group_id": "first"},
            {"id": "m1", "type": "multi", "group_id": "second"}]
    values = {"G-V": [True, True, True, False], "C-S": [True, False, False, True], "C-I": [False, False, True, True]}
    records = {solver: [{"id": row["id"], "final_correct": correct, "original_scorer_correct": correct,
                         "correct": correct, "status": "correct" if correct else "incorrect"}
                        for row, correct in zip(rows, vector)] for solver, vector in values.items()}
    result = analysis.analyze_ab("B", rows, records, reps=100, seed=7)
    raw_draws = analysis.grouped_bootstrap(rows, values, reps=100, seed=7)
    for table_row in result["table"]:
        if table_row["type"] in ("single", "micro", "macro"):
            assert table_row["ci95"] is None
            assert table_row["ci95_reason"] == "insufficient_material_groups"
            assert table_row["ci95_insufficient_group_types"] == ["single"]
            assert table_row["success_rate"] is not None
        else:
            assert table_row["ci95"] == analysis.interval(raw_draws["multi"][table_row["solver"]])
            assert table_row["ci95_reason"] is None
    for pair in result["comparisons"]:
        assert pair["difference_pp"] is not None
        assert pair["unresolved"] == 0
        if pair["type"] in ("single", "micro", "macro"):
            assert pair["ci95_pp"] is None
            assert pair["ci95_reason"] == "insufficient_material_groups"
        else:
            expected = analysis.interval(raw_draws["multi"][pair["left"]] - raw_draws["multi"][pair["right"]])
            assert pair["ci95_pp"] == [100 * x for x in expected]
    single_pair = next(r for r in result["comparisons"] if r["type"] == "single" and r["left"] == "G-V" and r["right"] == "C-S")
    assert single_pair["difference_pp"] == 50
    assert single_pair["both_correct"] == single_pair["left_only"] == 1


def test_multi_material_zero_variation_interval_is_retained_with_note():
    ci, metadata = analysis.ab_reported_interval(np.ones(100), {"flat": 2}, "flat")
    assert ci == [1, 1]
    assert metadata["ci95_reason"] is None
    assert metadata["ci95_degenerate"] is True
    assert "not evidence of zero population uncertainty or equivalence" in metadata["ci95_note"]
    assert analysis.ci_text(None, reason="insufficient_material_groups") != "未完成"
