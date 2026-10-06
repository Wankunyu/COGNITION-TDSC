"""Sequential, resumable CaptchaMind episodes with the pinned native prompt.

An already dispatched request is never retried. A saved response can be resumed
by replaying its actions in the local deterministic environment. Ground truth,
environment logs, and rewards are never added to model-visible messages.
"""
import argparse
import contextlib
import copy
import json
import math
import os
import re
import shutil
import signal
import time
from pathlib import Path

from . import calibration
from .providers import Provider, ProviderError, request_json
from .runner import read_records, utc_now
from .tasks import ROOT, load_json

DEFAULT_BUDGETS = {
    "max_calls_per_episode": 30,
    "max_output_tokens_per_call": 8192,
    "max_output_tokens_per_episode": 65536,
    "request_timeout_seconds": 600,
    "episode_timeout_seconds": 1800,
}
PAUSED = {"pending_request", "run_call_budget_paused"}
STOP_RUN = PAUSED | {"infrastructure_failure", "native_environment_failure",
                     "adapted_environment_failure", "recovery_failure"}
PARSER_VERSION = "lossless_single_action_v1"


def atomic_json(path, value):
    """Durably save a request/response before advancing its checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def budgets_for(config):
    settings = config.get("interactive", {})
    result = {key: settings.get(key, default) for key, default in DEFAULT_BUDGETS.items()}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Invalid interactive budget: {key}")
        if not math.isfinite(value) or (not key.endswith("seconds") and int(value) != value):
            raise ValueError(f"Invalid interactive budget: {key}")
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


DECODER = json.JSONDecoder(parse_constant=_reject_constant, object_pairs_hook=_unique_object)


def parse_action(text):
    """Read one complete action, preserving numeric values and first-action order.

    Accept the source's JSON/XML wrappers and Markdown fences. Never complete a
    truncated object or extract an integer prefix from a decimal. JSON after an
    opening tool tag can be complete even if generation omitted its closing tag.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("No action text")
    body = text.split("</think>", 1)[-1] if "</think>" in text else text
    tagged = re.search(r"<(tool_call|action)>(.*)", body, re.S)
    if tagged:
        body = tagged.group(2).split(f"</{tagged.group(1)}>", 1)[0].strip()
    name_tag = re.search(r"<(?:function-name|name)>([^<]+)</(?:function-name|name)>", body)
    args_tag = re.search(r"<(?:args-json-object|arguments)>(.*?)</(?:args-json-object|arguments)>", body, re.S)
    if name_tag and args_tag:
        args_text = args_tag.group(1).strip()
        if args_text.startswith("{"):
            args, end = DECODER.raw_decode(args_text)
            if args_text[end:].strip():
                raise ValueError("Ambiguous arguments wrapper")
        else:
            # Source XML position fallback, extended only to full number values.
            fields = re.findall(r"<(position|from|to|number)>(.*?)</\1>", args_text, re.S)
            if not fields or len({key for key, _ in fields}) != len(fields):
                raise ValueError("Incomplete or duplicate XML arguments")
            args = {}
            for key, literal in fields:
                literal = literal.strip()
                args[key], end = DECODER.raw_decode(literal)
                if literal[end:].strip():
                    raise ValueError("Incomplete XML argument value")
        parsed = {"name": name_tag.group(1).strip(), "arguments": args}
    else:
        body = re.sub(r"^\s*```(?:json)?\s*", "", body)
        first = body.find("{")
        if first < 0:
            raise ValueError("No complete action object")
        # Decode only the first object. A malformed first action is never skipped.
        parsed, _ = DECODER.raw_decode(body[first:])
    if not isinstance(parsed, dict) or not isinstance(parsed.get("name"), str):
        raise ValueError("Action requires a name")
    if not parsed["name"] or not isinstance(parsed.get("arguments"), dict):
        raise ValueError("Action requires an arguments object")
    # JSON 1e999 is accepted as infinity by the stdlib decoder; reject recursively.
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Action contains a non-finite number")
        if isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)
    finite(parsed)
    return {"name": parsed["name"], "kwargs": parsed["arguments"]}


def make_environment(row, output):
    if row.get("source") == "captchabench":
        return calibration.make_env(row, output)
    from .interactive_envs import make_adapted_env
    return make_adapted_env(row, output)


def _environment_failure(row):
    return "native_environment_failure" if row.get("source") == "captchabench" else "adapted_environment_failure"


def _is_invalid_adapted_action(exc):
    # The adapted environment uses a specific exception for invalid model input.
    return any(cls.__name__ == "InvalidActionError" for cls in type(exc).__mro__)


def validate_native_structure(action, row):
    """Check only arguments that the source will dereference, never an allowlist.

    Source execution still controls unknown actions, coordinate ranges, numeric
    conversion, and done. A renderer limitation is an environment failure.
    """
    if row.get("source") != "captchabench":
        return
    if action.name == "click":
        position = action.kwargs.get("position")
        if (not isinstance(position, list) or len(position) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in position)):
            raise ValueError("click requires two finite coordinates")
    if action.name == "enter_number" and row["type"] == "Dice_Count":
        try:
            int(action.kwargs["number"])
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError("Native dice enter_number requires an int-convertible value") from None


def _public_attribute(env, name):
    value = getattr(env, name, None)
    return value() if callable(value) else value


def final_answer(env, row, submitted):
    value = _public_attribute(env, "final_answer")
    if value is not None or row.get("source") != "captchabench":
        return value
    state = env.current_state
    kind = row["type"]
    if kind == "Dice_Count":
        return {"answer_type": "number", "value": state["input_number"]}
    if kind == "Patch_Select":
        return {"answer_type": "multi_select", "indices": [
            index for index, selected in enumerate(state["sub_image_states"]) if selected]}
    if kind == "Click_Order":
        actions = env.actions[:-1] if submitted else env.actions
        return {"answer_type": "click_order", "points": [
            {"x": action.kwargs["position"][0], "y": action.kwargs["position"][1]}
            for action in actions if action.name == "click"]}
    if kind == "Rotation_Match":
        index = state["current_image_index"]
        return {"answer_type": "candidate_selection", "image_index_zero_based": index,
                "image_filename": Path(state["image_list"][index]).name}
    return None


def save_observation(observation, output, step):
    """Copy image bytes before native environments can overwrite their tmp files."""
    saved = []
    for index, filename in enumerate(observation or []):
        source = Path(filename)
        destination = output / "observations" / f"step_{step:02d}_{index:02d}{source.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise ValueError(f"Observation replay differs at step {step}, image {index}")
        else:
            shutil.copyfile(source, destination)
        saved.append(str(destination))
    return saved


def observation_content(paths, native):
    if not paths:
        return ""
    content = native.gen_image_message(paths)
    return calibration.transport_messages([{"role": "user", "content": content}])[0]["content"]


@contextlib.contextmanager
def request_deadline(seconds):
    """Bound the entire synchronous request, including repeated socket reads."""
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(signum, frame):
        raise TimeoutError("Interactive request wall-clock deadline reached")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


def _context_error(error):
    return bool(re.search(r"context[_ -]?(?:length|window)|maximum (?:context|sequence)|"
                          r"(?:input|prompt).{0,40}(?:too long|exceeds)|too many tokens|"
                          r"Request needs \d+ context tokens.*MAX_KV_SIZE", str(error), re.I | re.S))


def _response_step(raw, number, elapsed):
    choice = (raw.get("choices") or [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "") or ""
    usage = raw.get("usage", {})
    for field in ("prompt_tokens", "completion_tokens"):
        value = usage.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"Response missing valid {field}; budget accounting unavailable")
    if not isinstance(content, str):
        raise ValueError("Response message.content is not text")
    return {"step": number, "text": content, "usage": usage,
            "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
            "finish_reason": choice.get("finish_reason"),
            "truncated": choice.get("finish_reason") in ("length", "MAX_TOKENS"),
            "returned_model": raw.get("model"), "elapsed_seconds": elapsed,
            "raw_response": raw, "response_file": f"response_{number:02d}.json"}


def run_episode(row, provider, output, *, budgets=None, max_new_calls=None):
    """Run or resume one episode; max_new_calls limits only new dispatches."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    budgets = budgets or budgets_for(provider.config)
    checkpoint = output / "checkpoint.json"
    result_path = output / "result.json"
    if result_path.exists():
        saved = load_json(result_path)
        if saved["status"] not in PAUSED:
            return saved
    prior = load_json(checkpoint) if checkpoint.exists() else None
    if prior and (prior["id"] != row["id"] or prior["budgets"] != budgets
                  or prior["model"] != provider.model):
        raise ValueError("Episode definition changed")
    started_unix = prior["started_unix"] if prior else time.time()
    record = {"id": row["id"], "type": row["type"], "source": row.get("source"),
              "module": row.get("module", "B" if row.get("source") == "captchabench" else "A"),
              "condition": row.get("condition", "native_actions"), "solver": "C-I",
              "puzzle_id": row.get("puzzle_id"), "group_id": row.get("group_id"),
              "partition": row.get("partition"), "model": provider.model,
              "started_at": prior["started_at"] if prior else utc_now(),
              "started_unix": started_unix, "budgets": budgets,
              "protocol": "source_native" if row.get("source") == "captchabench" else "adapted_native_agent",
              "action_parser": PARSER_VERSION, "steps": [], "model_calls": 0,
              "input_tokens": 0, "output_tokens": 0, "reward": 0.0, "done": False,
              "correct": False, "status": "running", "submitted": False,
              "images": row.get("images", []),
              "cost_usd": None}
    for key in ("source_revision", "source_path", "training_exposure", "source_assets", "scorer", "split_evidence", "source_issue", "source_issues", "base_id", "gt"):
        if key in row:
            record[key] = row[key]
    messages = []
    native = calibration.native_modules()
    new_calls = 0

    def save():
        record["elapsed_seconds"] = time.time() - started_unix
        atomic_json(output / "messages.json", messages)
        atomic_json(checkpoint, record)

    def finish(status, correct=False, **details):
        record.update(status=status, stop_reason=status, correct=correct, finished_at=utc_now(), **details)
        record["model_calls"] = len(list(output.glob("request_*.json")))
        save()
        atomic_json(result_path, record)
        return record

    with (output / "environment.log").open("a") as log, contextlib.redirect_stdout(log):
        try:
            env, instruction = make_environment(row, output)
            agent = native.ReactAgent(provider.model, instruction, temperature=0)
            response = env.reset(task_index=0)
            if not response.observation:
                raise ValueError("Initial observation is empty")
            initial_paths = save_observation(response.observation, output, 0)
            messages = [{"role": "user", "content": [{"type": "text", "text": agent.prompt}]
                         + observation_content(initial_paths, native)}]
            record["initial_observation"] = initial_paths
            metadata = _public_attribute(env, "public_metadata")
            if metadata is not None:
                record["environment_metadata"] = metadata
        except Exception as exc:
            return finish(_environment_failure(row), None, error=f"{type(exc).__name__}: {exc}")

        for number in range(1, int(budgets["max_calls_per_episode"]) + 1):
            request_path = output / f"request_{number:02d}.json"
            response_path = output / f"response_{number:02d}.json"
            if request_path.exists():
                record["model_calls"] += 1
                request = load_json(request_path)
                expected_tokens = min(budgets["max_output_tokens_per_call"],
                                      budgets["max_output_tokens_per_episode"] - record["output_tokens"])
                expected_request = {"model": provider.model, "messages": messages,
                                    "temperature": 0, "max_tokens": expected_tokens, "stream": False}
                if request != expected_request:
                    return finish("recovery_failure", None,
                                  error=f"Saved request differs from reconstructed history at step {number}")
                if not response_path.exists():
                    return finish("pending_request", None, pending_step=number,
                                  error="Dispatched request has no saved response; automatic resend is prohibited")
                saved_response = load_json(response_path)
            else:
                remaining_seconds = budgets["episode_timeout_seconds"] - (time.time() - started_unix)
                if remaining_seconds <= 0:
                    return finish("episode_time_budget_exhausted")
                remaining_tokens = budgets["max_output_tokens_per_episode"] - record["output_tokens"]
                if remaining_tokens <= 0:
                    return finish("output_budget_exhausted")
                if max_new_calls is not None and new_calls >= max_new_calls:
                    return finish("run_call_budget_paused", None)
                max_tokens = int(min(budgets["max_output_tokens_per_call"], remaining_tokens))
                timeout = min(budgets["request_timeout_seconds"], remaining_seconds)
                payload = {"model": provider.model, "messages": copy.deepcopy(messages),
                           "temperature": 0, "max_tokens": max_tokens, "stream": False}
                # Files and the pending checkpoint are persisted before dispatch.
                atomic_json(request_path, payload)
                atomic_json(output / f"dispatch_{number:02d}.json",
                            {"started_at": utc_now(), "timeout_seconds": timeout,
                             "remaining_episode_seconds": remaining_seconds})
                record["model_calls"] += 1
                new_calls += 1
                record.update(status="pending_request", pending_step=number)
                save()
                headers = {"Content-Type": "application/json"}
                if provider.key:
                    headers["Authorization"] = "Bearer " + provider.key
                started = time.monotonic()
                try:
                    with request_deadline(timeout):
                        raw = request_json(provider.base_url + "/chat/completions", payload, headers, timeout)
                    saved_response = {"raw_response": raw, "elapsed_seconds": time.monotonic() - started,
                                      "received_at": utc_now()}
                    atomic_json(response_path, saved_response)
                except ProviderError as exc:
                    details = {"error": str(exc), "http_status": exc.status,
                               "elapsed_seconds": time.monotonic() - started, "received_at": utc_now()}
                    if exc.status is not None:
                        atomic_json(response_path, details)
                        saved_response = details
                    else:
                        atomic_json(output / f"transport_error_{number:02d}.json", details)
                        return finish("pending_request", None, pending_step=number, error=str(exc))
                except (TimeoutError, OSError, ValueError) as exc:
                    # A timeout or unreadable body may follow completed generation.
                    atomic_json(output / f"transport_error_{number:02d}.json",
                                {"error": f"{type(exc).__name__}: {exc}", "at": utc_now()})
                    return finish("pending_request", None, pending_step=number,
                                  error=f"{type(exc).__name__}: {exc}")
            record.pop("pending_step", None)
            record["status"] = "running"
            if "error" in saved_response:
                record["steps"].append({"step": number, **saved_response})
                status = "context_budget_exhausted" if _context_error(saved_response["error"]) else "infrastructure_failure"
                return finish(status, False if status == "context_budget_exhausted" else None,
                              error=saved_response["error"], http_status=saved_response.get("http_status"))
            try:
                step = _response_step(saved_response["raw_response"], number, saved_response.get("elapsed_seconds", 0))
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                return finish("infrastructure_failure", None, error=f"Malformed provider response: {exc}")
            record["steps"].append(step)
            record["input_tokens"] += step["input_tokens"]
            record["output_tokens"] += step["output_tokens"]
            try:
                action_dict = parse_action(step["text"])
                step["action"] = action_dict
                action = native.Action(**action_dict)
                validate_native_structure(action, row)
            except (ValueError, TypeError) as exc:
                messages.append({"role": "assistant", "content": step["text"]})
                return finish("output_budget_exhausted" if step["truncated"] else "protocol_failure",
                              error=f"{type(exc).__name__}: {exc}")
            messages.append({"role": "assistant", "content": step["text"]})
            try:
                response = env.step(action)
                paths = save_observation(response.observation, output, number)
                step.update(done=bool(response.done), reward=response.reward, observation=paths)
                messages.append({"role": "user", "content": observation_content(paths, native)})
                record.update(done=bool(response.done), reward=response.reward)
                submitted = bool(response.done) and (row.get("source") != "captchabench" or (
                    action.name == "click" and env.is_click_submit(action.kwargs["position"])))
                record["submitted"] = submitted
                step["submitted"] = submitted
                answer = final_answer(env, row, submitted)
                if answer is not None:
                    record.update(final_answer=answer, answer=answer)
            except Exception as exc:
                status = "protocol_failure" if _is_invalid_adapted_action(exc) else _environment_failure(row)
                return finish(status, False if status == "protocol_failure" else None,
                              error=f"{type(exc).__name__}: {exc}")
            # Truncated generations with complete actions are executed normally.
            save()
            if response.done:
                return finish("correct" if response.reward == 1 else "incorrect", response.reward == 1)
        return finish("action_budget_exhausted")


def episode_directory(output, row):
    # Preserve exact identity (including D conditions) without unsafe path parts.
    pieces = row["id"].split("/")
    if not pieces or any(piece in ("", ".", "..") for piece in pieces):
        raise ValueError("Invalid episode id")
    return Path(output) / "episodes" / Path(*pieces)


def freeze_code(output):
    """Keep actual implementation copies with each run, without content hashes."""
    names = ["interactive_eval.py", "interactive_envs.py", "calibration.py", "providers.py",
             "tasks.py", "answer_eval.py", "answer_extraction.py", "defense_tasks.py"]
    directory = Path(output) / "frozen_code"
    directory.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in names:
        source = Path(__file__).parent / name
        if not source.exists():
            continue
        destination = directory / name
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise ValueError(f"Implementation changed since this run was frozen: {name}")
        else:
            shutil.copyfile(source, destination)
        copied.append(name)
    return {"directory": str(directory.resolve()), "files": copied,
            "native_source": str(calibration.SNAPSHOT.resolve())}


def run_interactive(rows, provider, output, *, config=None, limit=None, max_total_calls=None):
    """Run the fixed manifest prefix, preserving completed and partial episodes."""
    config = config or provider.config
    budgets = budgets_for(config)
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a nonnegative fixed-prefix length")
    if max_total_calls is None:
        max_total_calls = config.get("interactive", {}).get("max_total_calls", len(rows) * budgets["max_calls_per_episode"])
    if type(max_total_calls) is not int or max_total_calls < 0:
        raise ValueError("max_total_calls must be a nonnegative integer")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate manifest ids")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    code_snapshot = freeze_code(output)
    spec = {"solver": "C-I", "provider": "captchamind", "model": provider.model,
            "config": config, "budgets": budgets, "task_ids": ids,
            "native_source": load_json(calibration.SNAPSHOT / "origin.json"),
            "code_snapshot": code_snapshot,
            "research_code_archive": config.get("code_snapshot", "reports/solver_defense_v1/frozen_code"),
            "action_parser": PARSER_VERSION, "history": "complete assistant message.content and original observations; independent reasoning fields are saved only",
            "image_transport": "PNG MIME normalization only; image bytes and originating turns unchanged",
            "execution": "sequential; native env decides done; local deterministic replay on recovery",
            "stopping": "submission, episode budgets, parse failure, or infrastructure/environment failure",
            "unknown_requests": "saved before dispatch, never automatically resent"}
    for name, value in (("manifest.json", rows), ("run.json", spec)):
        path = output / name
        if path.exists() and load_json(path) != value:
            raise ValueError(f"Run definition changed: {name}")
        if not path.exists():
            atomic_json(path, value)
    records_path = output / "attempts.jsonl"
    old = read_records(records_path)
    if len({row["id"] for row in old}) != len(old):
        raise ValueError("Duplicate completed episode records")
    done = {row["id"] for row in old}
    if any(record["status"] in STOP_RUN for record in old):
        raise ValueError("A recorded environment/provider failure requires reconciliation")
    selected = rows if limit is None else rows[:limit]
    calls = len(list((output / "episodes").rglob("request_*.json")))
    if calls > max_total_calls:
        raise ValueError("Recorded calls already exceed the requested total ceiling")
    atomic_json(output / "invocation.json", {"at": utc_now(), "limit": limit,
                "max_total_calls": max_total_calls, "previous_calls": calls})
    with records_path.open("a") as handle:
        for row in selected:
            if row["id"] in done:
                continue
            episode = episode_directory(output, row)
            existing = len(list(episode.glob("request_*.json")))
            if calls >= max_total_calls and not existing:
                break
            atomic_json(output / "pending.json", [row["id"]])
            result = run_episode(row, provider, episode, budgets=budgets,
                                 max_new_calls=max_total_calls - calls)
            calls += result["model_calls"] - existing
            atomic_json(output / "call_ledger.json", {"model_calls": calls, "ceiling": max_total_calls,
                                                       "updated_at": utc_now()})
            if result["status"] not in PAUSED:
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                done.add(row["id"])
                atomic_json(output / "pending.json", [])
            print(json.dumps({"id": row["id"], "status": result["status"],
                              "model_calls": result["model_calls"], "total_calls": calls}), flush=True)
            if result["status"] in STOP_RUN:
                break
    atomic_json(output / "coverage.json", {"total": len(rows), "prefix_limit": limit,
                "completed": len(done), "remaining_ids": [item for item in ids if item not in done],
                "model_calls": calls})
    return read_records(records_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--secrets", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-total-calls", type=int)
    args = parser.parse_args()
    config = load_json(args.config)
    provider = Provider("captchamind", config, secrets_path=args.secrets, base_url=args.base_url)
    run_interactive(load_json(args.manifest), provider, args.output, config=config,
                    limit=args.limit, max_total_calls=args.max_total_calls)


if __name__ == "__main__":
    main()
