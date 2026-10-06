"""One-generation answer evaluation with format-tolerant, label-blind extraction."""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .providers import Provider, ProviderError
from .runner import read_records, utc_now
from .tasks import ROOT, load_json, score, write_json


def answer_prompt(row):
    # Preserve task semantics, reference order, coordinate systems and counting rules.
    text = re.sub(r"Respond with JSON only[^\n]*", "", row["prompt"], flags=re.I)
    text = re.sub(r"Return JSON\s*\{[^\n]*", "", text, flags=re.I)
    text = re.sub(r"JSON:\s*\{[^\n]*", "", text, flags=re.I)
    text = text.replace("Use a single number (no units or extra text).",
                        "The final answer is a single angle in degrees.").strip()
    if row["type"] == "Rotation_Match" and row["source"] == "conference":
        text = text.replace(
            "Task: Output the rotation angle (in degrees, range [0, 360)) that best aligns the object with the reference direction.",
            "Task: Image 1 is the reference direction; image 2 is the current object. "
            "Output the additional CLOCKWISE rotation (in degrees, range [0, 360)) "
            "needed to make the object's facing direction match the reference.",
        )
        text = text.replace(
            "- If symmetric variants exist (e.g., 180° symmetry), prefer the smallest angle that is considered correct for this puzzle.\n", "",
        )
    hints = {
        "Dice_Count": "the integer total",
        "Click_Order": "the complete ordered list of (x, y) pixel coordinates in the MAIN puzzle image",
        "Place_Dot": "one (x, y) pixel coordinate",
        "Pick_Area": "one (x, y) pixel coordinate",
        "Patch_Select": "the complete list of zero-based row-major cell indices",
        "Rotation_Match": "one rotation angle in degrees, as specified by the task",
    }
    sizes = "; ".join(f"image {i + 1}: {w} x {h} pixels"
                      for i, (w, h) in enumerate(row["image_sizes"]))
    return (
        "Solve this visual puzzle carefully. You may reason before answering. "
        "Give the complete solution in this one response; no actions will be executed "
        "and no additional observations will be provided.\n\n"
        + text + "\n\nInput dimensions: " + sizes
        + "\nFor clarity, end with <answer>YOUR FINAL ANSWER</answer>. "
        + "Your final answer should contain " + hints[row["type"]] + "."
    )


def run_answers(rows, provider, output, config, *, workers=1, max_cost_usd=20):
    from .answer_extraction import extract_answer

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    definition = {"provider": provider.name, "model": provider.model, "config": config,
                  "manifest": rows, "prompts": {r["id"]: answer_prompt(r) for r in rows}}
    definition_path = output / "run.json"
    if definition_path.exists() and load_json(definition_path)["spec"] != definition:
        raise ValueError("Run definition changed; use a new output directory")
    if not definition_path.exists():
        write_json(definition_path, {"created_at": utc_now(), "spec": definition})
    path = output / "attempts.jsonl"
    old = read_records(path)
    if any(r["status"] == "infrastructure_failure" for r in old):
        raise ValueError("Reconcile prior infrastructure failures before resuming")
    pending_path = output / "pending.json"
    if pending_path.exists() and load_json(pending_path):
        raise ValueError("Reconcile interrupted pending requests before resuming")
    completed = {r["id"] for r in old}
    todo = [r for r in rows if r["id"] not in completed]
    spent = sum(r.get("cost_usd") or 0 for r in old)

    def attempt(row):
        record = {"id": row["id"], "type": row["type"], "source": row["source"],
                  "provider": provider.name, "model": provider.model, "model_calls": 1,
                  "started_at": utc_now(), "images": row["images"], "prompt": answer_prompt(row),
                  "gt": row["gt"], "scorer": row["scorer"], "group_id": row["group_id"],
                  "source_annotation_issue": row.get("source_annotation_issue"),
                  "partition": row["partition"], "split_evidence": row["split_evidence"]}
        for field in ("source_revision", "source_path", "training_exposure", "angle_mapping"):
            if field in row:
                record[field] = row[field]
        try:
            response = provider.solve(row)
            answer, error, method = extract_answer(response["text"], row)
            correct = score(row, answer)
            status = "correct" if correct else ("needs_answer_review" if error else "incorrect")
            output_truncated = response["finish_reason"] in ("MAX_TOKENS", "length")
            if output_truncated and answer is None:
                status = "output_budget_exhausted"
            source_score = correct
            if row.get("source_annotation_issue"):
                correct, status = None, "source_annotation_issue"
            record.update(response)
            record.update(answer=answer, correct=correct, status=status,
                          original_scorer_correct=source_score, output_truncated=output_truncated,
                          extraction_error=error, extraction_method=method)
        except ProviderError as exc:
            record.update(correct=None, status="infrastructure_failure", error=str(exc),
                          http_status=exc.status, cost_usd=None)
        record["finished_at"] = utc_now()
        return record

    with path.open("a") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(todo), workers):
            if provider.name == "gemini" and spent >= max_cost_usd:
                print(f"Cost limit reached: ${spent:.4f}", flush=True)
                break
            batch = todo[start:start + workers]
            pending = {r["id"] for r in batch}
            write_json(pending_path, sorted(pending))
            failed = False
            for future in as_completed([pool.submit(attempt, row) for row in batch]):
                record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                pending.remove(record["id"])
                write_json(pending_path, sorted(pending))
                spent += record.get("cost_usd") or 0
                print(json.dumps({"id": record["id"], "status": record["status"],
                                  "extraction": record.get("extraction_method"),
                                  "cost_usd_total": round(spent, 6) if provider.name == "gemini" else None}), flush=True)
                failed |= record["status"] == "infrastructure_failure"
            if failed:
                break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=["captchamind", "gemini"], required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url")
    p.add_argument("--secrets", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-cost-usd", type=float, default=20)
    args = p.parse_args()
    config = load_json(ROOT / "configs/answer_20260913.json")
    provider = Provider(args.provider, config, args.secrets, args.base_url,
                        prompt_builder=answer_prompt)
    provider.check()
    run_answers(load_json(args.manifest), provider, args.output, config,
                workers=args.workers, max_cost_usd=args.max_cost_usd)


if __name__ == "__main__":
    main()
