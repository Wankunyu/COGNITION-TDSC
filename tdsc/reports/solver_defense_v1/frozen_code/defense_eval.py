"""One static generation per D puzzle, with durable request/response accounting.

Reuses the frozen complete-answer provider and label-blind reader. The transport
is intercepted only to save its exact payload before dispatch and its response
before extraction. Runs are sequential, so the scoped transport replacement has
no concurrent callers. A saved response can be re-read without querying again;
a dispatched request with no response is never automatically repeated.
"""
import argparse
import ast
import json
import math
import re
import shutil
import time
from pathlib import Path
from unittest.mock import patch

from . import providers
from .answer_extraction import extract_answer, _valid_answer_tags
from .defense_tasks import score_defense
from .runner import read_records, utc_now
from .tasks import ROOT, load_json, write_json


PROXY_TYPES = {
    "Select_Animal": "Patch_Select",
    "Select_Animal_Optimized": "Click_Order",
    "Visible_Eye_Count": "Dice_Count",
}
HINTS = {
    "Select_Animal": 'A list of zero-based row-major grid indices, such as [0, 4].',
    "Select_Animal_Optimized": 'All eye-center coordinates in pixels as [(x1, y1), (x2, y2), ...]. Order does not matter. Include only actually visible eyes of the requested target.',
    "Visible_Eye_Count": 'One integer giving the total number of actually visible eyes of the requested target.',
}


def defense_prompt(row):
    sizes = "; ".join(f"image {i + 1}: {w}x{h} pixels" for i, (w, h) in enumerate(row["image_sizes"]))
    return (
        "Solve this visual puzzle. You may reason before answering, but provide one complete final answer. "
        "No actions are executed and no additional observations are available.\n\n"
        + row["prompt"] + "\n\nInput dimensions: " + sizes
        + "\nCoordinates use the top-left origin: x increases rightward and y downward."
        + "\nFinal answer: " + HINTS[row["type"]]
        + "\nPut your final answer between <answer> and </answer> tags."
    )


def extract_defense_answer(text, row):
    # The frozen reader inspects only task type, never labels or images.
    result = extract_answer(text, {"type": PROXY_TYPES[row["type"]]})
    if row["type"] != "Select_Animal_Optimized" or result[1] != "unparsed_answer":
        return result
    if not _valid_answer_tags(text):
        return result
    tagged = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if len(tagged) != 1:
        return result
    literal = tagged[0].strip()
    fenced = re.fullmatch(r"```(?:json|python)?\s*(.*?)\s*```", literal, flags=re.S)
    if fenced:
        literal = fenced[1]
    try:
        value = ast.literal_eval(literal)
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return result
    if isinstance(value, dict):
        if "points" not in value or set(value) - {"points", "answer_type"}:
            return result
        value = value["points"]
    numeric = lambda n: isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n)
    if not isinstance(value, (list, tuple)) or not value:
        return result
    if all(numeric(item) for item in value):
        if len(value) % 2:
            return result
        points = [{"x": value[i], "y": value[i + 1]} for i in range(0, len(value), 2)]
        method = "flat_xy_pairs"
    else:
        def flatten(group):
            if not isinstance(group, (list, tuple)) or not group:
                return None
            if len(group) == 2 and all(numeric(item) for item in group):
                return [{"x": group[0], "y": group[1]}]
            children = [flatten(child) for child in group]
            if any(child is None for child in children):
                return None
            return [point for child in children for point in child]
        points = flatten(value)
        if points is None:
            return result
        method = "grouped_point_pairs"
    # D asks for points, not boxes. Keep every coordinate and duplicate; never
    # average pairs into box centers or select a group using the image/labels.
    return {"answer_type": "click_order", "points": points}, None, "explicit_final/" + method


def _episode_path(output, row):
    parts = Path(row["id"]).parts
    if Path(row["id"]).is_absolute() or ".." in parts:
        raise ValueError("Unsafe puzzle identifier")
    return output / "requests" / Path(*parts)


def run_defense_static(rows, provider, output, config, *, limit=None,
                       max_total_calls=None, max_cost_usd=10):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    solver = "G-V" if provider.name == "gemini" else "C-S"
    spec = {"provider": provider.name, "model": provider.model, "solver": solver,
            "config": config, "task_ids": [r["id"] for r in rows],
            "prompts": [defense_prompt(r) for r in rows]}
    run_path, manifest_path = output / "run.json", output / "manifest.json"
    if run_path.exists() and load_json(run_path)["spec"] != spec:
        raise ValueError("Run definition changed; use a new output directory")
    if manifest_path.exists() and load_json(manifest_path) != rows:
        raise ValueError("Run manifest changed; use a new output directory")
    if not run_path.exists():
        frozen = output / "frozen_code"
        frozen.mkdir(exist_ok=True)
        for name in ("defense_eval.py", "defense_tasks.py", "answer_extraction.py", "providers.py"):
            shutil.copy2(Path(__file__).with_name(name), frozen / name)
        write_json(run_path, {"created_at": utc_now(), "spec": spec,
                              "code_snapshot": str(frozen.relative_to(output))})
        write_json(manifest_path, rows)
    records_path = output / "attempts.jsonl"
    records = read_records(records_path)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate completed puzzle records")
    done = {r["id"] for r in records}
    spent = sum(r.get("cost_usd") or 0 for r in records)
    selected = rows if limit is None else rows[:limit]
    transport = providers.request_json
    for row in selected:
        if row["id"] in done:
            continue
        episode = _episode_path(output, row)
        request_path, response_path = episode / "request.json", episode / "response.json"
        if request_path.exists() and not response_path.exists():
            write_json(output / "progress.json", {"status": "pending_request", "id": row["id"],
                       "completed": len(records), "planned": len(rows), "cost_usd": spent})
            return records
        calls = sum(1 for _ in (output / "requests").rglob("request.json"))
        if not response_path.exists() and (
            (max_total_calls is not None and calls >= max_total_calls)
            or (provider.name == "gemini" and spent >= max_cost_usd)
        ):
            write_json(output / "progress.json", {"status": "run_budget_paused", "completed": len(records),
                       "planned": len(rows), "model_calls": calls, "cost_usd": spent})
            return records
        episode.mkdir(parents=True, exist_ok=True)

        def capture(url, payload, headers, timeout):
            if response_path.exists():
                if load_json(request_path) != payload:
                    raise ValueError("Saved request differs from frozen request construction")
                return load_json(response_path)
            write_json(request_path, payload)
            write_json(episode / "dispatch.json", {"started_at": utc_now(), "timeout_seconds": timeout})
            started = time.monotonic()
            raw = transport(url, payload, headers, timeout)
            write_json(response_path, raw)
            write_json(episode / "timing.json", {"elapsed_seconds": time.monotonic() - started,
                                                "finished_at": utc_now()})
            return raw

        try:
            with patch.object(providers, "request_json", capture):
                response = provider.solve(row)
        except providers.ProviderError as exc:
            write_json(episode / "error.json", {"error": str(exc), "http_status": exc.status,
                                                "at": utc_now()})
            write_json(output / "progress.json", {"status": "infrastructure_failure", "id": row["id"],
                       "completed": len(records), "planned": len(rows), "cost_usd": spent})
            return records
        timing_path = episode / "timing.json"
        response["elapsed_seconds"] = load_json(timing_path)["elapsed_seconds"] if timing_path.exists() else None
        answer, error, method = extract_defense_answer(response["text"], row)
        correct = bool(score_defense(row, answer))
        record = {key: row[key] for key in (
            "id", "type", "source", "puzzle_id", "group_id", "partition", "images", "gt", "scorer",
            "condition", "base_id",
        ) if key in row}
        status = "correct" if correct else ("needs_answer_review" if error else "incorrect")
        if error and response["finish_reason"] in ("length", "MAX_TOKENS"):
            status = "output_budget_exhausted"
        record.update(response)
        record.update({"module": "D", "solver": solver, "provider": provider.name, "model": provider.model,
                       "prompt": defense_prompt(row), "answer": answer, "final_answer": answer,
                       "correct": correct, "status": status, "error": error, "extraction_method": method,
                       "model_calls": 1, "finished_at": utc_now(),
                       "truncated": response["finish_reason"] in ("length", "MAX_TOKENS")})
        with records_path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
        records.append(record)
        done.add(row["id"])
        spent += record.get("cost_usd") or 0
        write_json(output / "progress.json", {"status": "complete" if len(records) == len(rows) else "running",
                   "completed": len(records), "planned": len(rows), "model_calls": len(records),
                   "cost_usd": spent, "last_id": row["id"]})
        print(json.dumps({"completed": len(records), "planned": len(rows), "id": row["id"],
                          "status": status, "cost_usd": spent}), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/solver_defense_v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", required=True, choices=("gemini", "captchamind"))
    parser.add_argument("--secrets")
    parser.add_argument("--base-url")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-total-calls", type=int)
    parser.add_argument("--max-cost-usd", type=float, default=10)
    args = parser.parse_args()
    config = load_json(args.config)
    provider = providers.Provider(args.provider, config, args.secrets, args.base_url, defense_prompt)
    run_defense_static(load_json(args.manifest), provider, args.output, config,
                       limit=args.limit, max_total_calls=args.max_total_calls, max_cost_usd=args.max_cost_usd)


if __name__ == "__main__":
    main()
