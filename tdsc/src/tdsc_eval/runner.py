import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .providers import ProviderError
from .tasks import parse_answer, prompt_for, score, failure_label, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_records(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run(rows, provider, output, config, *, workers=1, limit=None, max_cost_usd=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = {"provider": provider.name, "model": provider.model, "config": config,
            "task_ids": [r["id"] for r in rows]}
    spec_path = output / "run.json"
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != rows:
        raise ValueError("Run manifest changed; use a new output directory")
    if spec_path.exists() and json.loads(spec_path.read_text())["spec"] != spec:
        raise ValueError("Run definition changed; use a new output directory")
    if not spec_path.exists():
        write_json(spec_path, {"created_at": utc_now(), "spec": spec})
        write_json(manifest_path, rows)
    records_path = output / "attempts.jsonl"
    old = read_records(records_path)
    done = {r["id"] for r in old}
    pending_path = output / "pending.json"
    if pending_path.exists() and json.loads(pending_path.read_text()):
        raise ValueError("An interrupted batch remains in pending.json; reconcile it before resuming")
    todo = [r for r in rows if r["id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    spent = sum(r.get("cost_usd") or 0 for r in old)

    def attempt(row):
        record = {
            "id": row["id"], "type": row["type"], "source": row["source"],
            "puzzle_id": row["puzzle_id"], "group_id": row["group_id"],
            "split_evidence": row["split_evidence"], "provider": provider.name,
            "model": provider.model, "started_at": utc_now(), "images": row["images"],
            "prompt": prompt_for(row), "gt": row["gt"], "scorer": row["scorer"],
            "model_calls": 1,
        }
        for name in ("source_revision", "source_path", "training_exposure", "source_assets", "angle_mapping"):
            if name in row:
                record[name] = row[name]
        try:
            response = provider.solve(row)
            answer, error = parse_answer(response["text"], row)
            correct = score(row, answer)
            status = "correct" if correct else ("protocol_failure" if error else "incorrect")
            if response["finish_reason"] in ("MAX_TOKENS", "length"):
                status, correct = "output_budget_exhausted", False
            record.update(response)
            record.update({"answer": answer, "correct": correct, "status": status,
                           "error": error, "failure_label": None if correct else (
                               error or failure_label(row, answer) if answer else error
                           )})
        except ProviderError as exc:
            record.update({"status": "infrastructure_failure", "correct": None,
                           "error": str(exc), "http_status": exc.status, "cost_usd": None})
        return record

    with records_path.open("a") as handle, ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(todo), workers):
            if provider.name == "gemini" and spent >= max_cost_usd:
                print(f"Cost limit reached: ${spent:.4f}", flush=True)
                break
            batch = todo[start:start + workers]
            write_json(pending_path, [r["id"] for r in batch])
            failed = False
            futures = {executor.submit(attempt, row): row for row in batch}
            pending = {r["id"] for r in batch}
            for future in as_completed(futures):
                record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                spent += record.get("cost_usd") or 0
                pending.remove(record["id"])
                write_json(pending_path, sorted(pending))
                print(json.dumps({"id": record["id"], "status": record["status"],
                                  "cost_usd_total": round(spent, 6) if provider.name == "gemini" else None}, ensure_ascii=False), flush=True)
                failed = failed or record["status"] == "infrastructure_failure"
            if failed:
                print("Stopped after provider failure; no automatic answer retries.", flush=True)
                break
    return read_records(records_path)
