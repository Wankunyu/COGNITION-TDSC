import json
from pathlib import Path

import pytest

from cognition.expanded_dataset_phase042 import (
    assert_no_invalid_phase041_rows,
    write_phase042_final_outputs,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _selected_row(task_type: str, *, source_kind: str = "open_source_dataset") -> dict[str, object]:
    is_gpt = source_kind == "gpt_image_open_captchaworld_style"
    return {
        "selected_id": f"phase042-{task_type.lower()}",
        "candidate_id": f"candidate-{task_type.lower()}",
        "source_path": f"expanded_captcha_data/phase04_2/candidates/{task_type}",
        "candidate_image_paths": [
            f"expanded_captcha_data/phase04_2/candidates/{task_type}/{index}.png"
            for index in range(10)
        ],
        "source_kind": source_kind,
        "source_provenance_class": (
            "gpt_image_generated_fallback" if is_gpt else "preferred_real_external"
        ),
        "source_citation": "" if is_gpt else "Example real external CAPTCHA dataset",
        "source_license": "" if is_gpt else "CC-BY-4.0",
        "source_provenance_notes": (
            "GPT Image fallback with recorded prompt."
            if is_gpt
            else "Real external CAPTCHA samples for Phase 04.2."
        ),
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
        "gpt_image_generation_prompt": "Generate a counting CAPTCHA." if is_gpt else "",
        "gpt_image_model": "gpt-image-1" if is_gpt else "",
        "gpt_image_generation_date": "2026-05-21" if is_gpt else "",
        "open_captchaworld_style_rationale": (
            "Open CaptchaWorld-style layout." if is_gpt else ""
        ),
    }


def _analysis_row(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "analysis_id": "exp5_evidence_analysis-openai-gpt-5-Symbol_Count-static",
        "run_id": "exp5_evidence_analysis",
        "evidence_mode": "static",
        "provider": "openai",
        "model": "gpt-5",
        "provider_model": "openai/gpt-5",
        "task_type": "Symbol_Count",
        "task_family": "Counting",
        "evidence_origin": "new_category",
        "sample_count": 10,
        "attempt_count": 10,
        "success_count": 0,
        "source_kind": "open_source_dataset",
        "source_provenance_class": "preferred_real_external",
        "source_citation": "Example real external CAPTCHA dataset",
        "source_license": "CC-BY-4.0",
        "source_provenance_notes": "Real external CAPTCHA samples.",
        "provenance_caveat": "",
        "direct_evidence": True,
        "real_external_evidence": True,
        "original_rate": 0.2,
        "corrected_static_rate": 0.0,
        "corrected_adaptive_rate": None,
        "adaptive_success_at_3": None,
        "adaptive_success_at_5": None,
        "bernoulli_success_at_3": 0.488,
        "bernoulli_success_at_5": 0.67232,
        "adaptive_round_count": 0,
        "memory_isolation": "",
        "ci_low": 0.0,
        "ci_high": 0.2775,
        "ci_method": "wilson_95",
        "ci_note": "Wilson interval over corrected static attempts.",
        "agreement_status": "supports_original",
        "diverges_from_original": False,
        "divergence_reason": "",
        "scientific_wrong_count": 10,
        "protocol_failure_count": 0,
        "infrastructure_failure_count": 0,
        "selected_manifest_path": (
            "expanded_captcha_data/phase04_2/phase042_selected_manifest.json"
        ),
        "claim_boundary_note": "claim_boundary_note: corrected expanded-dataset evidence only.",
        "claim_effect": "supports_structural_hardness",
        "adaptive_scope_rationale": "",
        "source_artifact_path": (
            "results/exp5/evidence_analysis/"
            "exp5_evidence_analysis.json"
        ),
    }
    values.update(overrides)
    return values


def _write_common_inputs(tmp_path: Path, *, gpt_symbol: bool = False) -> dict[str, Path]:
    selected_manifest = _write_json(
        tmp_path / "expanded_captcha_data/phase04_2/phase042_selected_manifest.json",
        {
            "schema_version": "cognition.revision.phase042.selected_manifest.v1",
            "rows": [
                _selected_row("Symbol_Count", source_kind="gpt_image_open_captchaworld_style")
                if gpt_symbol
                else _selected_row("Symbol_Count"),
                _selected_row("Relation_Match"),
                _selected_row("Hole_Counting"),
            ],
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
                    "dataset_increase_percent_vs_local_legacy": 81.82,
                },
                {
                    "candidate_id": "phase042-nextgen-symbol-count",
                    "task_type": "Symbol_Count",
                    "source_kind": "open_source_dataset",
                },
            ],
        },
    )
    source_kind = "gpt_image_open_captchaworld_style" if gpt_symbol else "open_source_dataset"
    evidence_rows = [
        _analysis_row(
            source_kind=source_kind,
            source_provenance_class=(
                "gpt_image_generated_fallback"
                if gpt_symbol
                else "preferred_real_external"
            ),
            real_external_evidence=not gpt_symbol,
            provenance_caveat=(
                "GPT Image fallback evidence is not real external dataset evidence."
                if gpt_symbol
                else ""
            ),
        ),
        _analysis_row(
            analysis_id="exp5_evidence_analysis-openai-gpt-5_medium-Dice_Count-adaptive",
            evidence_mode="adaptive",
            provider_model="openai/gpt-5_medium",
            model="gpt-5_medium",
            task_type="Dice_Count",
            evidence_origin="supplemented_category",
            source_kind="original_captchaworld_hard_scope",
            source_provenance_class="original_hard_task_context",
            source_citation="",
            source_license="",
            source_provenance_notes="Original hard-task adaptive context.",
            provenance_caveat="Original hard-task adaptive context.",
            direct_evidence=True,
            real_external_evidence=False,
            corrected_static_rate=None,
            corrected_adaptive_rate=0.2,
            adaptive_success_at_3=0.2,
            adaptive_success_at_5=0.8,
            adaptive_round_count=5,
            memory_isolation="five_memory_isolated_rounds",
            attempt_count=15,
            success_count=1,
            claim_effect="supports_structural_hardness",
            adaptive_scope_rationale=(
                "Since most recognition-oriented tasks are already solved reliably "
                "under the non-adaptive setting, adaptive feedback would mainly "
                "introduce ceiling effects."
            ),
            claim_boundary_note="Adaptive hard-scope evidence, not a population-level estimate.",
        ),
    ]
    evidence_analysis = _write_json(
        tmp_path / "results/exp5/evidence_analysis/exp5_evidence_analysis.json",
        {
            "schema_version": "cognition.revision.phase042.evidence_analysis.v1",
            "rows": evidence_rows,
        },
    )
    static_summary = _write_json(
        tmp_path / "results/local_runs/phase04_2_static/expanded_static_summary.json",
        {"schema_version": "cognition.revision.phase042.static_summary.v1", "rows": []},
    )
    adaptive_summary = _write_json(
        tmp_path / "results/local_runs/phase04_2_adaptive/expanded_adaptive_summary.json",
        {"schema_version": "cognition.revision.phase042.adaptive_summary.v1", "rows": []},
    )
    return {
        "selected_manifest": selected_manifest,
        "source_download_manifest": source_download_manifest,
        "evidence_analysis": evidence_analysis,
        "static_summary": static_summary,
        "adaptive_summary": adaptive_summary,
        "output_root": tmp_path / "results/exp5",
    }


def _write_outputs(tmp_path: Path, *, gpt_symbol: bool = False) -> Path:
    paths = _write_common_inputs(tmp_path, gpt_symbol=gpt_symbol)
    write_phase042_final_outputs(
        selected_manifest_path=paths["selected_manifest"],
        source_download_manifest_path=paths["source_download_manifest"],
        evidence_analysis_path=paths["evidence_analysis"],
        static_summary_paths=[paths["static_summary"]],
        adaptive_summary_path=paths["adaptive_summary"],
        output_root=paths["output_root"],
        run_id="exp5_final_outputs_20260522",
    )
    return paths["output_root"] / "final_outputs_20260522"


def test_final_outputs_reject_phase041_rows(tmp_path: Path) -> None:
    paths = _write_common_inputs(tmp_path)
    bad_payload = json.loads(paths["evidence_analysis"].read_text(encoding="utf-8"))
    bad_payload["rows"][0]["analysis_id"] = "phase04_1-symbol-count"
    _write_json(paths["evidence_analysis"], bad_payload)

    with pytest.raises(ValueError, match="invalid"):
        write_phase042_final_outputs(
            selected_manifest_path=paths["selected_manifest"],
            source_download_manifest_path=paths["source_download_manifest"],
            evidence_analysis_path=paths["evidence_analysis"],
            static_summary_paths=[paths["static_summary"]],
            adaptive_summary_path=paths["adaptive_summary"],
            output_root=paths["output_root"],
            run_id="exp5_final_outputs_20260522",
        )


def test_final_outputs_scan_latex_and_shepherd_response_text(tmp_path: Path) -> None:
    output_dir = _write_outputs(tmp_path)

    assert_no_invalid_phase041_rows(output_dir)
    assert (output_dir / "corrected_expanded_table_rows.tex").is_file()
    assert (output_dir / "corrected_shepherd_response_snippet.txt").is_file()

    bad_latex = output_dir / "bad.latex"
    bad_latex.write_text("synthetic_fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid marker"):
        assert_no_invalid_phase041_rows(output_dir)


def test_final_outputs_distinguish_gpt_image_from_real_external(tmp_path: Path) -> None:
    output_dir = _write_outputs(tmp_path, gpt_symbol=True)
    rows = json.loads(
        (output_dir / "corrected_expanded_table_rows.json").read_text(encoding="utf-8")
    )["rows"]
    symbol = next(row for row in rows if row["task_type"] == "Symbol_Count")
    notes = (output_dir / "corrected_claim_boundary_notes.md").read_text(encoding="utf-8")

    assert symbol["source_kind"] == "gpt_image_open_captchaworld_style"
    assert symbol["real_external_evidence"] is False
    assert "GPT Image fallback" in symbol["provenance_caveat"]
    assert "not real external dataset evidence" in notes
    assert "real external samples" in notes


def test_final_outputs_report_adaptive_success_at_3_and_5(tmp_path: Path) -> None:
    output_dir = _write_outputs(tmp_path)
    rows = json.loads(
        (output_dir / "corrected_expanded_table_rows.json").read_text(encoding="utf-8")
    )["rows"]
    adaptive = next(row for row in rows if row["adaptive_hard_scope_evidence"])
    adaptive_notes = (output_dir / "corrected_adaptive_scope_notes.md").read_text(
        encoding="utf-8"
    )

    assert adaptive["adaptive_success_at_3"] == 0.2
    assert adaptive["adaptive_success_at_5"] == 0.8
    assert adaptive["round_count"] == 5
    assert adaptive["attempt_budget_k"] == 5
    assert adaptive["intermediate_budget_k"] == 3
    assert "memory-isolated" in adaptive_notes
    assert "ceiling effects" in adaptive_notes
    assert "adaptive_hard_scope_evidence" in adaptive_notes
    assert "contextual_sota_only" in (
        output_dir / "corrected_expanded_table_rows.csv"
    ).read_text(encoding="utf-8")


def test_final_outputs_exclude_updated_ocw_increment_claims(tmp_path: Path) -> None:
    output_dir = _write_outputs(tmp_path)
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.iterdir()
        if path.suffix in {".csv", ".json", ".md", ".tex", ".txt"}
    )

    assert (
        "No staged OpenCaptchaWorld hard-type increment rows are retained"
        in combined
    )
    assert "no GPT Image fallback instances were used" in combined
    assert "dataset_increase_percent_vs_local_legacy" not in combined
    assert "81.82" not in combined
    assert "phase04_1-relation-match" not in combined
