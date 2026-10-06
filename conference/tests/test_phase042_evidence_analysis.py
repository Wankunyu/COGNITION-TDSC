import json
from pathlib import Path

from cognition.expanded_dataset_phase042 import (
    PHASE042_ADAPTIVE_TASK_TYPES,
    PHASE042_OPENROUTER_QWEN_PROVIDER_MODEL,
    PHASE042_STATIC_TASK_TYPES,
    build_phase042_evidence_analysis_rows,
    render_phase042_divergence_report,
)


LEGACY_QWEN_PROVIDER_MODEL = (
    "fireworks/accounts_fireworks_models_qwen3-vl-235b-a22b-instruct"
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _selected_row(task_type: str, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "selected_id": f"phase042-{task_type.lower()}",
        "candidate_id": f"candidate-{task_type.lower()}",
        "source_path": f"expanded_captcha_data/phase04_2/candidates/{task_type}",
        "candidate_image_paths": [
            f"expanded_captcha_data/phase04_2/candidates/{task_type}/{index}.png"
            for index in range(10)
        ],
        "source_kind": "open_source_dataset",
        "source_provenance_class": "preferred_real_external",
        "source_citation": "Example external CAPTCHA dataset",
        "source_license": "CC-BY-4.0",
        "source_provenance_notes": "Real external CAPTCHA samples for Phase 04.2.",
        "evidence_origin": "new_category",
        "slice_type": "new_category",
        "task_type": task_type,
        "task_family": "Relational Matching"
        if task_type == "Relation_Match"
        else "Counting",
        "sample_count": 10,
        "label_format": "ground_truth.json",
        "metadata_alignment_notes": "source ids mapped to selected rows",
        "answer_format_normalization": "answers normalized for evaluator use",
        "compatibility_status": "ready_for_static_pipeline",
        "evaluation_status": "selected_for_static",
        "limitation_notes": "offline corrected Phase 04.2 sidecar",
        "adaptive_eligible": True,
        "static_compatibility_notes": "offline images with ground truth",
        "novelty_sha256": ["b" * 64 for _ in range(10)],
        "novelty_hash_report_path": (
            "expanded_captcha_data/phase04_2/novelty_hash_report.json"
        ),
        "exact_captcha_data_match": False,
        "perceptual_warning_count": 0,
        "review_warnings": [],
    }
    values.update(overrides)
    return values


def _static_row(
    provider_model: str,
    task_type: str,
    pass_rate: float,
    *,
    attempt_count: int = 10,
    success_count: int | None = None,
    scientific_wrong_count: int | None = None,
    protocol_failure_count: int = 0,
    infrastructure_failure_count: int = 0,
) -> dict[str, object]:
    provider, model = provider_model.split("/", 1)
    successes = int(round(pass_rate * attempt_count)) if success_count is None else success_count
    wrong = (
        max(0, attempt_count - successes - protocol_failure_count - infrastructure_failure_count)
        if scientific_wrong_count is None
        else scientific_wrong_count
    )
    return {
        "run_id": f"phase04_2_static_supplemental-{provider}-{model}",
        "provider": provider,
        "model": model,
        "provider_model": provider_model,
        "task_type": task_type,
        "task_family": "Relational Matching"
        if task_type == "Relation_Match"
        else "Counting",
        "evidence_origin": "new_category",
        "sample_count": 10,
        "attempt_count": attempt_count,
        "success_count": successes,
        "scientific_wrong_count": wrong,
        "protocol_failure_count": protocol_failure_count,
        "infrastructure_failure_count": infrastructure_failure_count,
        "pass_rate": pass_rate,
        "run_manifest_path": "results/local_runs/phase04_2_static/run_manifest.json",
        "attempt_log_path": "results/local_runs/phase04_2_static/attempts.jsonl",
        "summary_source_path": "results/local_runs/phase04_2_static/summary.json",
        "selected_manifest_path": (
            "expanded_captcha_data/phase04_2/phase042_selected_manifest.json"
        ),
        "claim_use": "main_body_direct_evidence",
    }


def _adaptive_round_row(
    task_type: str,
    round_index: int,
    success_at_3: bool,
    success_at_5: bool,
) -> dict[str, object]:
    return {
        "run_id": f"phase04_2_adaptive_gpt5_medium-round{round_index:02d}",
        "provider": "openai",
        "model": "gpt-5_medium",
        "provider_model": "openai/gpt-5_medium",
        "task_type": task_type,
        "task_family": "Counting",
        "evidence_origin": "supplemented_category"
        if task_type not in PHASE042_STATIC_TASK_TYPES
        else "new_category",
        "sample_count": 10,
        "session_count": 1,
        "round_id": f"round{round_index:02d}",
        "round_index": round_index,
        "round_count": 5,
        "attempt_budget_k": 5,
        "intermediate_budget_k": 3,
        "success_count": int(success_at_5),
        "success_at_3": success_at_3,
        "success_at_5": success_at_5,
        "attempts_to_success_at_3": 1 if success_at_3 else None,
        "attempts_to_success_at_5": 4 if success_at_5 and not success_at_3 else 1,
        "scientific_wrong_count": 0 if success_at_5 else 5,
        "protocol_failure_count": 0,
        "infrastructure_failure_count": 0,
        "adaptive_success_rate": 1.0 if success_at_5 else 0.0,
        "feedback_mode": "binary-pass-fail",
        "memory_mode": "explicit-policy-notes",
        "stopping_rule": "first-success-or-budget",
        "run_manifest_path": "results/local_runs/phase04_2_adaptive/run_manifest.json",
        "adaptive_attempt_log_path": (
            "results/local_runs/phase04_2_adaptive/adaptive_attempts.jsonl"
        ),
        "adaptive_summary_source_path": (
            "results/local_runs/phase04_2_adaptive/adaptive_summary.json"
        ),
        "selected_manifest_path": (
            "expanded_captcha_data/phase04_2/phase042_selected_manifest.json"
        ),
        "claim_use": "main_body_caveated",
    }


def _write_common_inputs(tmp_path: Path, *, gpt_symbol: bool = False) -> dict[str, Path]:
    selected_rows = [
        _selected_row("Hole_Counting"),
        _selected_row("Relation_Match"),
        _selected_row(
            "Symbol_Count",
            **(
                {
                    "source_kind": "gpt_image_open_captchaworld_style",
                    "source_provenance_class": "gpt_image_generated_fallback",
                    "source_citation": "",
                    "source_license": "",
                    "source_provenance_notes": "GPT Image fallback with recorded prompt.",
                    "gpt_image_generation_prompt": "Generate a counting CAPTCHA.",
                    "gpt_image_model": "gpt-image-1",
                    "gpt_image_generation_date": "2026-05-21",
                    "open_captchaworld_style_rationale": "Open CaptchaWorld-style layout.",
                }
                if gpt_symbol
                else {}
            ),
        ),
    ]
    selected_manifest = _write_json(
        tmp_path / "expanded_captcha_data/phase04_2/phase042_selected_manifest.json",
        {
            "schema_version": "cognition.revision.phase042.selected_manifest.v1",
            "rows": selected_rows,
        },
    )
    source_download_manifest = _write_json(
        tmp_path / "expanded_captcha_data/phase04_2/source_download_manifest.json",
        {
            "schema_version": "cognition.revision.phase042.source_download_manifest.v1",
            "rows": [
                {
                    "candidate_id": "phase042-ocw-dice-count-latest-additions",
                    "task_type": "Dice_Count",
                    "source_path": (
                        "expanded_captcha_data/phase04_2/candidates/"
                        "OpenCaptchaWorld_Dice_Count_latest_additions"
                    ),
                    "dataset_increase_percent_vs_local_legacy": 81.82,
                },
                {"candidate_id": "phase042-nextgen-hole-counting", "task_type": "Hole_Counting"},
            ],
        },
    )
    base_static = _write_json(
        tmp_path / "results/local_runs/phase04_2_static_supplemental/expanded_static_summary.json",
        {
            "schema_version": "cognition.revision.phase042.static_summary.v1",
            "rows": [
                _static_row("openai/gpt-5", "Relation_Match", 0.1),
                _static_row("openai/gpt-5", "Symbol_Count", 0.6),
                _static_row(
                    "gemini/gemini-2.5-pro",
                    "Hole_Counting",
                    0.0,
                    attempt_count=2,
                    success_count=0,
                    scientific_wrong_count=0,
                    infrastructure_failure_count=2,
                ),
                _static_row(
                    LEGACY_QWEN_PROVIDER_MODEL,
                    "Symbol_Count",
                    0.0,
                    infrastructure_failure_count=10,
                ),
            ],
        },
    )
    openrouter_static = _write_json(
        tmp_path
        / (
            "results/local_runs/"
            "phase04_2_static_openrouter_qwen_infra_remediation_20260522/"
            "expanded_static_summary.json"
        ),
        {
            "schema_version": "cognition.revision.phase042.static_summary.v1",
            "rows": [
                _static_row(PHASE042_OPENROUTER_QWEN_PROVIDER_MODEL, "Symbol_Count", 0.0),
            ],
        },
    )
    openai_static = _write_json(
        tmp_path
        / (
            "results/local_runs/"
            "phase04_2_static_openai_infra_remediation_20260522/"
            "expanded_static_summary.json"
        ),
        {
            "schema_version": "cognition.revision.phase042.static_summary.v1",
            "rows": [
                _static_row("openai/gpt-5", "Hole_Counting", 0.0, attempt_count=1),
            ],
        },
    )
    adaptive_rows = []
    for task_type in PHASE042_ADAPTIVE_TASK_TYPES:
        for round_index in range(1, 6):
            adaptive_rows.append(
                _adaptive_round_row(
                    task_type,
                    round_index,
                    success_at_3=task_type == "Dice_Count" and round_index <= 1,
                    success_at_5=task_type == "Dice_Count" and round_index <= 4,
                )
            )
    adaptive_summary = _write_json(
        tmp_path
        / (
            "results/local_runs/phase04_2_adaptive_gpt5_medium_20260522/"
            "expanded_adaptive_summary.json"
        ),
        {
            "schema_version": "cognition.revision.phase042.adaptive_summary.v1",
            "rows": adaptive_rows,
        },
    )
    results_dir = tmp_path / "results"
    exp2_path = results_dir / "exp2/openai/gpt-5/results.csv"
    exp2_path.parent.mkdir(parents=True, exist_ok=True)
    exp2_path.write_text(
        "\n".join(
            [
                "provider,model,type,n,pass_at_1",
                "openai,gpt-5,Relation_Match,10,0.2",
                "openai,gpt-5,Symbol_Count,10,0.2",
                "openai,gpt-5,Hole_Counting,10,0.6",
                "openai,gpt-5,Dice_Count,10,0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    qwen_exp2 = (
        results_dir
        / "exp2/fireworks/accounts_fireworks_models_qwen3-vl-235b-a22b-instruct"
        / "results.csv"
    )
    qwen_exp2.parent.mkdir(parents=True, exist_ok=True)
    qwen_exp2.write_text(
        "provider,model,type,n,pass_at_1\nfireworks,qwen,Symbol_Count,10,0.2\n",
        encoding="utf-8",
    )
    return {
        "selected_manifest": selected_manifest,
        "source_download_manifest": source_download_manifest,
        "base_static": base_static,
        "openrouter_static": openrouter_static,
        "openai_static": openai_static,
        "adaptive_summary": adaptive_summary,
        "results_dir": results_dir,
    }


def test_analysis_reports_agreement_and_divergence(tmp_path: Path) -> None:
    paths = _write_common_inputs(tmp_path)

    rows = build_phase042_evidence_analysis_rows(
        selected_manifest_path=paths["selected_manifest"],
        source_download_manifest_path=paths["source_download_manifest"],
        static_summary_paths=[
            paths["base_static"],
            paths["openrouter_static"],
            paths["openai_static"],
        ],
        adaptive_summary_path=paths["adaptive_summary"],
        results_dir=paths["results_dir"],
        run_id="exp5_evidence_analysis",
    )
    records = [row.model_dump(mode="json") for row in rows]
    required = {
        "provider",
        "model",
        "provider_model",
        "task_type",
        "task_family",
        "sample_count",
        "source_kind",
        "source_provenance_class",
        "direct_evidence",
        "original_rate",
        "corrected_static_rate",
        "adaptive_success_at_3",
        "adaptive_success_at_5",
        "bernoulli_success_at_3",
        "bernoulli_success_at_5",
        "adaptive_round_count",
        "memory_isolation",
        "ci_low",
        "ci_high",
        "ci_method",
        "ci_note",
        "scientific_wrong_count",
        "protocol_failure_count",
        "infrastructure_failure_count",
        "agreement_status",
        "diverges_from_original",
        "divergence_reason",
        "claim_effect",
        "adaptive_scope_rationale",
    }

    assert required <= set(records[0])
    static_direct_tasks = {
        row["task_type"]
        for row in records
        if row["corrected_static_rate"] is not None and row["direct_evidence"]
    }
    assert static_direct_tasks <= set(PHASE042_STATIC_TASK_TYPES)
    adaptive_tasks = {
        row["task_type"] for row in records if row["adaptive_success_at_5"] is not None
    }
    assert adaptive_tasks == set(PHASE042_ADAPTIVE_TASK_TYPES)
    assert LEGACY_QWEN_PROVIDER_MODEL not in {row["provider_model"] for row in records}
    assert PHASE042_OPENROUTER_QWEN_PROVIDER_MODEL in {
        row["provider_model"] for row in records
    }
    assert {
        "supports_structural_hardness",
        "weakens_structural_hardness",
        "neutral_or_inconclusive",
    } <= {row["claim_effect"] for row in records}
    infra_row = next(row for row in records if row["provider_model"] == "gemini/gemini-2.5-pro")
    assert infra_row["infrastructure_failure_count"] == 2
    assert infra_row["claim_effect"] != "supports_structural_hardness"
    dice_row = next(
        row
        for row in records
        if row["provider_model"] == "openai/gpt-5_medium"
        and row["task_type"] == "Dice_Count"
    )
    assert dice_row["adaptive_success_at_3"] == 0.2
    assert dice_row["adaptive_success_at_5"] == 0.8
    assert dice_row["bernoulli_success_at_3"] is not None
    assert dice_row["bernoulli_success_at_5"] is not None
    assert dice_row["adaptive_round_count"] == 5
    assert dice_row["memory_isolation"] == "five_memory_isolated_rounds"
    assert "ceiling effects" in dice_row["adaptive_scope_rationale"]

    report = render_phase042_divergence_report(
        rows,
        source_download_manifest_path=paths["source_download_manifest"],
    )
    assert "40% reporting heuristic" in report
    assert (
        "updated local OpenCaptchaWorld hard-type incremental rows were staged but excluded"
        in report
    )
    assert (
        "no Dice_Count/Click_Order/Patch_Select/Geometry_Click dataset-increase percentages"
        in report
    )


def test_analysis_preserves_gpt_image_caveats(tmp_path: Path) -> None:
    paths = _write_common_inputs(tmp_path, gpt_symbol=True)

    rows = build_phase042_evidence_analysis_rows(
        selected_manifest_path=paths["selected_manifest"],
        source_download_manifest_path=paths["source_download_manifest"],
        static_summary_paths=[paths["base_static"], paths["openrouter_static"]],
        adaptive_summary_path=paths["adaptive_summary"],
        results_dir=paths["results_dir"],
        run_id="exp5_evidence_analysis",
    )
    symbol_rows = [
        row.model_dump(mode="json")
        for row in rows
        if row.task_type == "Symbol_Count" and row.corrected_static_rate is not None
    ]

    assert symbol_rows
    assert {row["source_kind"] for row in symbol_rows} == {
        "gpt_image_open_captchaworld_style"
    }
    assert all(row["real_external_evidence"] is False for row in symbol_rows)
    assert all("GPT Image" in row["provenance_caveat"] for row in symbol_rows)
    assert all(
        row["source_provenance_class"] == "gpt_image_generated_fallback"
        for row in symbol_rows
    )
