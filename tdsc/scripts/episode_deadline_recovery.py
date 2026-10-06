"""Resolve a proven episode-deadline timeout without resending its request.

This execution policy leaves the frozen runner and its raw episode files intact.
An unavailable response remains unavailable. Only the separate derived outcome
states the already-observed failure to finish before the frozen episode deadline.
"""
import contextlib
import copy
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

from tdsc_eval import interactive_eval as runner

POLICY_VERSION = "episode_deadline_reconciliation_v1"
RESOLUTION = "execution_resolution.json"


def read(path):
    return json.loads(Path(path).read_text())


def deadline_evidence(directory):
    """Return evidence only for a timeout actually caused by the episode cap.

    A stale network failure does not become a time-budget result on a later run.
    All comparisons use the original dispatch/error/result timestamps.
    """
    directory = Path(directory)
    path = directory / "result.json"
    if not path.is_file():
        return None
    result = read(path)
    if (result.get("status") != "pending_request" or result.get("correct") is not None
            or result.get("submitted") or result.get("done")):
        return None
    number = result.get("pending_step")
    if type(number) is not int or number < 1:
        return None
    request = directory / f"request_{number:02d}.json"
    dispatch = directory / f"dispatch_{number:02d}.json"
    error = directory / f"transport_error_{number:02d}.json"
    response = directory / f"response_{number:02d}.json"
    if not all(p.is_file() for p in (request, dispatch, error)) or response.exists():
        return None
    requests = list(directory.glob("request_*.json"))
    if (len(requests) != number or result.get("model_calls") != number
            or any(not (directory / f"request_{i:02d}.json").is_file() for i in range(1, number + 1))
            or any(not (directory / f"response_{i:02d}.json").is_file() for i in range(1, number))):
        return None
    dispatch_data, error_data = read(dispatch), read(error)
    if "TimeoutError" not in str(error_data.get("error", "")) or error_data.get("http_status") is not None:
        return None
    try:
        timeout = float(dispatch_data["timeout_seconds"])
        remaining = float(dispatch_data["remaining_episode_seconds"])
        cap = float(result["budgets"]["episode_timeout_seconds"])
        request_cap = float(result["budgets"]["request_timeout_seconds"])
        elapsed = float(result["elapsed_seconds"])
        deadline = float(result["started_unix"]) + cap
        dispatched = datetime.fromisoformat(dispatch_data["started_at"]).timestamp()
        failed = datetime.fromisoformat(error_data.get("received_at") or error_data["at"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return None
    if (not all(math.isfinite(x) for x in (timeout, remaining, cap, request_cap, elapsed, deadline, dispatched, failed))
            or not 0 < remaining <= request_cap or abs(timeout - remaining) > 0.001
            or abs(dispatched + remaining - deadline) > 2
            or elapsed < cap - 0.1 or failed < deadline - 0.1):
        return None
    return {"policy": POLICY_VERSION, "episode_id": result["id"],
            "pending_step": number, "request_file": request.name,
            "dispatch": dispatch_data, "transport_error": error_data,
            "original_status": result["status"], "episode_deadline_unix": deadline,
            "response_available": False, "request_resent": False,
            "reason": "No action from this request was received or applied before the frozen episode deadline"}


def derived_result(original, evidence):
    result = copy.deepcopy(original)
    result.update(status="episode_time_budget_exhausted", stop_reason="episode_time_budget_exhausted",
                  correct=False, execution_resolution=evidence, usage_complete=False,
                  observed_input_tokens=original.get("input_tokens"),
                  observed_output_tokens=original.get("output_tokens"),
                  input_tokens=None, output_tokens=None)
    result.pop("pending_step", None)
    return result


def load_resolution(directory):
    directory = Path(directory)
    path = directory / RESOLUTION
    if not path.exists():
        return None
    saved = read(path)
    evidence = deadline_evidence(directory)
    if (evidence is None or saved.get("evidence") != evidence
            or saved.get("result") != derived_result(read(directory / "result.json"), evidence)):
        raise RuntimeError(f"Deadline resolution no longer matches original evidence: {directory}")
    return saved


def wait_for_idle(log_path, timeout=630, interval=1):
    """Drain the one owned inference request before starting another episode.

    The dedicated runtime logs each request start/completion with in_flight.
    A completion after the client deadline is used only as an idle signal.
    """
    limit = time.monotonic() + timeout
    while True:
        with Path(log_path).open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 262144))
            lines = handle.read().decode(errors="replace").splitlines()
        event = next((line for line in reversed(lines)
                      if re.search(r"Request (?:started|completed):.*in_flight=\d+", line)), None)
        if event and "Request completed:" in event and re.search(r"in_flight=0(?:\s|$)", event):
            return {"confirmed_at": runner.utc_now(), "event": event}
        if time.monotonic() >= limit:
            raise RuntimeError("Timed-out episode is closed, but the model runtime has not confirmed idle; dispatch stopped")
        time.sleep(interval)


def resolve(directory, runtime_log):
    directory = Path(directory)
    saved = load_resolution(directory)
    if saved:
        return saved["result"]
    evidence = deadline_evidence(directory)
    if evidence is None:
        return None
    idle = wait_for_idle(runtime_log)
    result = derived_result(read(directory / "result.json"), evidence)
    runner.atomic_json(directory / RESOLUTION, {"created_at": runner.utc_now(), "evidence": evidence,
                                              "runtime_idle": idle, "result": result})
    return result


def reconcile_existing(report, names, runtime_log):
    resolved = []
    for name in names:
        for result_path in sorted((Path(report) / name / "episodes").rglob("result.json")):
            record = resolve(result_path.parent, runtime_log)
            if record is not None:
                resolved.append({"run": name, "id": record["id"], "status": record["status"],
                                 "model_calls": record["model_calls"]})
    return resolved


@contextlib.contextmanager
def installed(runtime_log):
    original = runner.run_episode

    def run_episode(row, provider, output, **kwargs):
        saved = load_resolution(output)
        if saved:
            record = saved["result"]
            if (record["id"] != row["id"] or record["model"] != provider.model
                    or record["budgets"] != kwargs.get("budgets", runner.budgets_for(provider.config))):
                raise ValueError("Resolved episode definition changed")
            return record
        result = original(row, provider, output, **kwargs)
        return resolve(output, runtime_log) or result

    runner.run_episode = run_episode
    try:
        yield
    finally:
        runner.run_episode = original
