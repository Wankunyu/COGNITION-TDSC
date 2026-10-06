"""Bounded development calibration using the pinned public CaptchaMind agent/envs.

The vendored source stays unchanged. Only filesystem roots and the completion
transport are bound here. Episodes are sequential because upstream loaders use
module-global paths. Ground truth and environment stdout never enter messages.
"""
import argparse
import contextlib
import copy
import importlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

from .providers import Provider, ProviderError, request_json
from .runner import read_records, run, utc_now
from .tasks import ROOT, load_json, write_json

SNAPSHOT = ROOT / "baseline/captchamind_native"
SELECTION = ROOT / "manifests/calibration_20260912_selection.json"
MANIFEST = ROOT / "manifests/calibration_20260912.json"
OUTPUT = ROOT / "results/calibration_20260912"
TASKS = {
    "Dice_Count": ("dice_count", "DiceCountEnv", "dice_counting.json",
                   "Sum up the numbers on all the dice"),
    "Click_Order": ("click_order", "ClickOrderEnv", "annotation.json",
                    "Click the icons in the correct order as shown in the sequence at the bottom left of the image."),
    "Patch_Select": ("patch_select", "PatchSelectEnv", "task.json", None),
    "Rotation_Match": ("rotation_match", "RotationMatchEnv", "rotation_match.json",
                       "Use the arrows to find the image where the object faces the same direction as the reference."),
}


def native_modules():
    if str(SNAPSHOT) not in sys.path:
        sys.path.insert(0, str(SNAPSHOT))
    agent = importlib.import_module("captcha.agents.react_agent")
    if not Path(agent.__file__).resolve().is_relative_to(SNAPSHOT.resolve()):
        raise RuntimeError("Another captcha package is already loaded")
    return agent


def make_env(row, episode_dir):
    """Copy only original inputs/label into an isolated upstream-compatible layout."""
    native_modules()
    name, cls, label_name, instruction = TASKS[row["type"]]
    source = ROOT / "data/external/captchabench" / row["source_path"]
    stage = Path(episode_dir) / "staging"
    sample = stage / row["puzzle_id"]
    sample.mkdir(parents=True, exist_ok=True)
    original_label = "coordinates.json" if name == "rotation_match" else label_name
    shutil.copyfile(source / original_label, sample / label_name)
    label = load_json(sample / label_name)
    images = label["image_list"] if name == "rotation_match" else [label["image"]]
    for filename in images:
        shutil.copyfile(source / filename, sample / filename)
    data = importlib.import_module(f"captcha.env.{name}.data")
    data.DATASET_ROOT = str(stage)
    assets = SNAPSHOT / "captcha/env" / name / "data"
    if name == "click_order":
        data.NUMBER_IMAGES_PATH = str(assets)
    if name == "patch_select":
        data.TICK_IMAGE_PATH = str(assets / "tick.png")
        instruction = f"select all squares of {label['target_object_type']}"
    module = importlib.import_module(f"captcha.env.{name}.env")
    env = getattr(module, cls)(task_index=0)
    return env, instruction


def validate_action(action, task_type):
    allowed = {"click", "enter_number"} if task_type == "Dice_Count" else {"click"}
    if action.name not in allowed:
        raise ValueError(f"Action {action.name!r} is not supported by this environment")
    if action.name == "click":
        position = action.kwargs.get("position")
        if (not isinstance(position, list) or len(position) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) for x in position)):
            raise ValueError("click requires two finite pixel coordinates")
    elif type(action.kwargs.get("number")) is not int:
        raise ValueError("enter_number requires an integer")


def transport_messages(messages):
    """Supply PNG MIME omitted by upstream, without changing image bytes/history."""
    transported = copy.deepcopy(messages)
    for message in transported:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:image;base64,"):
                        part["image_url"]["url"] = url.replace(
                            "data:image;base64,", "data:image/png;base64,", 1)
    return transported


def run_episode(row, provider, output, max_steps=16):
    agent_module = native_modules()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = {"id": row["id"], "type": row["type"], "condition": "native_actions",
              "started_at": utc_now(), "model": provider.model, "model_calls": 0,
              "status": "action_budget_exhausted", "correct": False,
              "steps": [], "input_tokens": 0, "output_tokens": 0}
    messages = []
    previous_completion = agent_module.completion

    def completion(model, messages, temperature):
        step_number = len(record["steps"]) + 1
        payload = {"model": model, "messages": transport_messages(messages), "temperature": temperature,
                   "max_tokens": provider.config["max_output_tokens"], "stream": False}
        # Exact requests preserve base64 observations before upstream overwrites tmp images.
        write_json(output / f"request_{step_number:02d}.json", payload)
        headers = {"Content-Type": "application/json"}
        if provider.key:
            headers["Authorization"] = "Bearer " + provider.key
        step = {"step": step_number, "started_at": utc_now()}
        record["steps"].append(step)
        record["model_calls"] += 1
        started = time.monotonic()
        try:
            raw = request_json(provider.base_url + "/chat/completions", payload,
                               headers, provider.timeout)
        finally:
            step["elapsed_seconds"] = time.monotonic() - started
        step["raw_response"] = raw
        choice = (raw.get("choices") or [{}])[0]
        text = choice.get("message", {}).get("content", "") or ""
        usage = raw.get("usage", {})
        step.update(text=text, usage=usage, finish_reason=choice.get("finish_reason"))
        record["input_tokens"] += usage.get("prompt_tokens", 0)
        record["output_tokens"] += usage.get("completion_tokens", 0)
        write_json(output / f"response_{step_number:02d}.json", step)
        return text, usage

    try:
        with (output / "environment.log").open("w") as audit_log, contextlib.redirect_stdout(audit_log):
            env, instruction = make_env(row, output)
            agent = agent_module.ReactAgent(provider.model, instruction, temperature=0)
            response = env.reset(task_index=0)
            messages = [{"role": "user", "content": [{"type": "text", "text": agent.prompt}]
                         + agent_module.gen_image_message(response.observation)}]
            agent_module.completion = completion
            for step_index in range(max_steps):
                try:
                    text, action, _ = agent.generate_next_step(messages, step_index + 1)
                    step = record["steps"][-1]
                    step["action"] = action.model_dump()
                    validate_action(action, row["type"])
                except ProviderError:
                    raise
                except (ValueError, AssertionError, KeyError, TypeError, AttributeError) as exc:
                    last = record["steps"][-1] if record["steps"] else {}
                    status = ("output_budget_exhausted" if last.get("finish_reason")
                              in ("length", "MAX_TOKENS") else "protocol_failure")
                    record.update(status=status, error=f"{type(exc).__name__}: {exc}")
                    break
                if step["finish_reason"] in ("length", "MAX_TOKENS"):
                    record["status"] = "output_budget_exhausted"
                    break
                try:
                    response = env.step(action)
                except Exception as exc:
                    record.update(status="native_environment_failure", correct=None,
                                  error=f"{type(exc).__name__}: {exc}")
                    break
                # Only the action-dependent observation is returned to the model.
                obs = agent_module.gen_image_message(response.observation) if response.observation else ""
                messages.extend([{"role": "assistant", "content": text},
                                 {"role": "user", "content": obs}])
                step.update(done=response.done, reward=response.reward)
                if response.done:
                    record.update(status="correct" if response.reward == 1 else "incorrect",
                                  correct=response.reward == 1)
                    break
    except ProviderError as exc:
        record.update(status="infrastructure_failure", correct=None,
                      error=str(exc), http_status=exc.status)
    finally:
        agent_module.completion = previous_completion
        record["finished_at"] = utc_now()
        write_json(output / "messages.json", messages)
        write_json(output / "result.json", record)
    return record


def run_native(rows, provider, output, selection):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    recovery_path = output.parent / "transport_recovery.json"
    recovery = load_json(recovery_path) if recovery_path.exists() else {}
    previous_requests = recovery.get("preceding_failed_http_requests", 0)
    spec = {"condition": "native_actions", "selection": selection,
            "manifest": rows, "model": provider.model,
            "max_output_tokens": provider.config["max_output_tokens"],
            "source": load_json(SNAPSHOT / "origin.json"),
            "system_prompt": None, "history": "full upstream observation history",
            "image_mime": "upstream data:image;base64 normalized to data:image/png;base64; bytes unchanged",
            "runtime_message_adapter": {
                "change": "MLX OpenAI endpoint preserves user content lists containing images",
                "purpose": "Keep image markers on their originating conversation turns",
                "validation_report": "reports/calibration_20260912_runtime_check.json",
                "temporary_runtime_patch": True,
            },
            "transport_recovery": recovery,
            "stopping": "submission, invalid action, output/action budget, or failure"}
    spec_path = output / "run.json"
    if spec_path.exists() and load_json(spec_path) != spec:
        raise ValueError("Native run definition changed")
    write_json(spec_path, spec)
    records_path = output / "attempts.jsonl"
    records = read_records(records_path)
    if any(r["status"] == "infrastructure_failure" for r in records):
        raise ValueError("Reconcile the recorded infrastructure failure before resuming")
    done = {r["id"] for r in records}
    spent = previous_requests + sum(r["model_calls"] for r in records)
    with records_path.open("a") as handle:
        for row in rows:
            if row["id"] in done:
                continue
            remaining = selection["max_calls"]["native_actions"] - spent
            if remaining <= 0:
                break
            episode = output / row["type"] / row["puzzle_id"]
            # Do not silently repeat an interrupted model call/episode.
            if episode.exists():
                raise ValueError(f"Reconcile existing episode before resuming: {episode}")
            record = run_episode(row, provider, episode,
                                 min(selection["max_actions_per_native_episode"], remaining))
            spent += record["model_calls"]
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(json.dumps({k: record[k] for k in ("id", "status", "model_calls")}), flush=True)
            if record["status"] in ("infrastructure_failure", "native_environment_failure"):
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=["direct_json", "native_actions"])
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    rows, selection = load_json(MANIFEST), load_json(SELECTION)
    formal_ids = {r["id"] for r in load_json(ROOT / "manifests/external.json")}
    if len(rows) != 8 or any(r["id"] in formal_ids for r in rows):
        raise ValueError("Calibration selection size or formal-test separation failed")
    if any(r["partition"] != "development_calibration" for r in rows):
        raise ValueError("Only development calibration is allowed here")
    config = load_json(ROOT / "configs/captchamind.json")
    provider = Provider("captchamind", config, base_url=args.base_url)
    provider.check()
    if args.condition == "direct_json":
        run(rows, provider, OUTPUT / args.condition, config, workers=1)
    else:
        run_native(rows, provider, OUTPUT / args.condition, selection)


if __name__ == "__main__":
    main()
