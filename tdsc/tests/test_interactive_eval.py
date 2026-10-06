"""Offline contracts for formal native episodes; no inference endpoint is used."""
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from tdsc_eval import calibration, interactive_eval as runner
from tdsc_eval.interactive_envs import InvalidActionError
from tdsc_eval.providers import ProviderError


def provider(**budgets):
    config = {"interactive": {**runner.DEFAULT_BUDGETS, **budgets}}
    return SimpleNamespace(model="offline-test", base_url="http://127.0.0.1:1/v1",
                           key="", config=config)


def row(identifier="test/number/1"):
    return {"id": identifier, "source": "conference", "module": "A", "type": "Dice_Count",
            "puzzle_id": identifier.rsplit("/", 1)[-1], "images": [], "gt": {"secret": 77293}}


def response(name="click", args=None, *, tokens=5, finish="stop", reasoning=None):
    message = {"content": '<think>Visible reasoning.</think><tool_call>' + json.dumps({
        "name": name, "arguments": args if args is not None else {"position": [99, 99]}
    }) + "</tool_call>"}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {"choices": [{"message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": tokens}, "model": "offline-test"}


class FakeEnv:
    def __init__(self, output):
        self.output = Path(output)
        self.value = None
        self.actions = []

    def _observation(self):
        image = Image.new("RGB", (120, 120), "white")
        ImageDraw.Draw(image).text((4, 4), str(self.value), fill="black")
        path = self.output / "overwritten.png"
        image.save(path)
        return [str(path)]

    def reset(self, task_index=0):
        self.value = None
        self.actions = []
        return SimpleNamespace(observation=self._observation())

    def step(self, action):
        self.actions.append(action)
        if action.name == "enter_number":
            self.value = action.kwargs["number"]
        done = action.name == "click" and action.kwargs["position"] == [99, 99]
        return SimpleNamespace(observation=self._observation(), done=done,
                               reward=float(done and self.value == 7))

    def final_answer(self):
        return {"value": self.value}

    def public_metadata(self):
        return {"coordinate_mapping": {"offset": [0, 0], "scale": [1, 1]}}


@pytest.fixture
def fake_env(monkeypatch):
    made = []
    def make(task, output):
        env = FakeEnv(output)
        made.append(env)
        return env, "Read the pictured integer, enter it, and click Submit."
    monkeypatch.setattr(runner, "make_environment", make)
    return made


@pytest.mark.parametrize("text,expected", [
    ('<tool_call>{"name":"enter_number","arguments":{"number":12.5}}</tool_call>', 12.5),
    ('<think>stuff</think><tool_call>{"name":"enter_number","arguments":{"number":-1.25e2}}', -125.0),
    ('```json\n{"name":"enter_number","arguments":{"number":0}}\n```', 0),
    ('<action><name>enter_number</name><arguments><number>-12.5</number></arguments></action>', -12.5),
    ('<function-name>enter_number</function-name><args-json-object>{"number":7}</args-json-object>', 7),
])
def test_lossless_complete_numeric_actions(text, expected):
    assert runner.parse_action(text) == {"name": "enter_number", "kwargs": {"number": expected}}


@pytest.mark.parametrize("literal", [
    '{"name":"enter_number","arguments":{"number":12.5',
    '{"name":"enter_number","arguments":{"number":12.5oops}}',
    '{"name":"enter_number","arguments":{"number":NaN}}',
    '{"name":"enter_number","arguments":{"number":1e999}}',
    '{"name":"enter_number","arguments":{"number":2,"number":3}}',
    '<action><name>enter_number</name><arguments><number>12.5oops</number></arguments></action>',
])
def test_no_number_prefix_or_incomplete_repair(literal):
    with pytest.raises(ValueError):
        runner.parse_action(literal)


def test_only_first_complete_action_is_executed():
    text = '{"name":"click","arguments":{"position":[1.25,-2]}}' + \
           '{"name":"click","arguments":{"position":[9,9]}}'
    assert runner.parse_action(text)["kwargs"] == {"position": [1.25, -2]}


def test_full_history_content_only_and_remaining_budget(monkeypatch, tmp_path, fake_env):
    calls = []
    answers = iter([response("enter_number", {"number": 7}, tokens=8192, finish="length",
                             reasoning="PRIVATE_REASONING_FIELD"), response(tokens=2)])
    def request(url, payload, headers, timeout):
        assert list(tmp_path.glob("request_*.json"))
        assert runner.load_json(tmp_path / "checkpoint.json")["status"] == "pending_request"
        calls.append(copy.deepcopy(payload))
        return next(answers)
    monkeypatch.setattr(runner, "request_json", request)
    result = runner.run_episode(row(), provider(max_output_tokens_per_episode=8200), tmp_path)
    assert result["correct"] and result["submitted"]
    assert result["model_calls"] == 2 and result["output_tokens"] == 8194
    assert [call["max_tokens"] for call in calls] == [8192, 8]
    assert len(calls[1]["messages"]) == 3
    assert "PRIVATE_REASONING_FIELD" not in json.dumps(calls)
    assert "77293" not in json.dumps(calls)
    assert "reward" not in json.dumps(calls)
    assert "PRIVATE_REASONING_FIELD" in json.dumps(result["steps"][0]["raw_response"])
    assert calls[1]["messages"][0] == calls[0]["messages"][0]
    assert calls[1]["messages"][-1]["content"] != calls[0]["messages"][0]["content"][1:]
    assert result["steps"][0]["truncated"] is True


def test_resume_saved_actions_without_regenerating(monkeypatch, tmp_path, fake_env):
    calls = []
    answers = iter([response("enter_number", {"number": 7}), response()])
    def request(*args):
        calls.append(copy.deepcopy(args[1]))
        return next(answers)
    monkeypatch.setattr(runner, "request_json", request)
    first = runner.run_episode(row(), provider(), tmp_path, max_new_calls=1)
    assert first["status"] == "run_call_budget_paused"
    assert not first["submitted"]
    second = runner.run_episode(row(), provider(), tmp_path, max_new_calls=1)
    assert second["status"] == "correct" and len(calls) == 2
    assert second["model_calls"] == 2 and len(second["steps"]) == 2
    assert fake_env[-1].value == 7
    assert calls[1]["messages"][1]["content"] == first["steps"][0]["text"]
    assert runner.run_episode(row(), provider(), tmp_path) == second
    assert len(calls) == 2


def test_response_saved_before_interruption_is_replayed(monkeypatch, tmp_path, fake_env):
    original = FakeEnv.step
    once = [True]
    def interrupt(self, action):
        if once.pop() if once else False:
            raise KeyboardInterrupt()
        return original(self, action)
    monkeypatch.setattr(FakeEnv, "step", interrupt)
    calls = []
    answers = iter([response("enter_number", {"number": 7}), response()])
    def request(*args):
        calls.append(1)
        return next(answers)
    monkeypatch.setattr(runner, "request_json", request)
    with pytest.raises(KeyboardInterrupt):
        runner.run_episode(row(), provider(), tmp_path)
    assert (tmp_path / "response_01.json").is_file()
    result = runner.run_episode(row(), provider(), tmp_path)
    assert result["correct"] and len(calls) == 2


def test_unknown_request_is_pending_and_not_retried(monkeypatch, tmp_path, fake_env):
    calls = []
    def request(*args):
        calls.append(1)
        raise ProviderError("Transport failure: TimeoutError")
    monkeypatch.setattr(runner, "request_json", request)
    for _ in range(2):
        result = runner.run_episode(row(), provider(), tmp_path)
        assert result["status"] == "pending_request" and result["correct"] is None
        assert result["model_calls"] == 1
    assert len(calls) == 1


@pytest.mark.parametrize("message,http,status,correct", [
    ("HTTP 400: maximum context length exceeded", 400, "context_budget_exhausted", False),
    ("HTTP 400: Request needs 130000 context tokens (121808 prompt + 8192 max generation), but MAX_KV_SIZE is 128000.", 400, "context_budget_exhausted", False),
    ("HTTP 503: unavailable", 503, "infrastructure_failure", None),
])
def test_known_http_errors_recorded_separately(monkeypatch, tmp_path, fake_env, message, http, status, correct):
    def request(*args):
        raise ProviderError(message, http)
    monkeypatch.setattr(runner, "request_json", request)
    result = runner.run_episode(row(), provider(), tmp_path)
    assert result["status"] == status and result["correct"] is correct
    assert (tmp_path / "response_01.json").exists()


def test_actual_request_wall_clock_timeout(monkeypatch, tmp_path, fake_env):
    def request(*args):
        time.sleep(1)
    monkeypatch.setattr(runner, "request_json", request)
    started = time.monotonic()
    result = runner.run_episode(row(), provider(request_timeout_seconds=0.025), tmp_path)
    assert time.monotonic() - started < 0.5
    assert result["status"] == "pending_request"


def test_native_patch_nonclick_is_not_rejected(monkeypatch, tmp_path):
    task = next(item for item in calibration.load_json(calibration.MANIFEST) if item["type"] == "Patch_Select")
    monkeypatch.setattr(runner, "request_json", lambda *args: response("enter_number", {"number": 7}))
    result = runner.run_episode(task, provider(max_calls_per_episode=2), tmp_path)
    assert result["status"] == "action_budget_exhausted"
    assert result["model_calls"] == 2 and not result["done"]


def test_missing_native_click_position_is_protocol_failure(monkeypatch, tmp_path):
    task = next(item for item in calibration.load_json(calibration.MANIFEST) if item["type"] == "Patch_Select")
    monkeypatch.setattr(runner, "request_json", lambda *args: response("click", {}))
    result = runner.run_episode(task, provider(), tmp_path)
    assert result["status"] == "protocol_failure" and result["correct"] is False


def test_native_rendering_range_failure_is_preserved(monkeypatch, tmp_path):
    task = next(item for item in calibration.load_json(calibration.MANIFEST) if item["type"] == "Dice_Count")
    monkeypatch.setattr(runner, "request_json", lambda *args: response("enter_number", {"number": 100}))
    result = runner.run_episode(task, provider(), tmp_path)
    assert result["status"] == "native_environment_failure" and result["correct"] is None
    assert result["steps"][0]["action"]["kwargs"]["number"] == 100


def test_invalid_adapted_action_is_protocol_failure(monkeypatch, tmp_path, fake_env):
    def invalid(self, action):
        raise InvalidActionError("Malformed click")
    monkeypatch.setattr(FakeEnv, "step", invalid)
    monkeypatch.setattr(runner, "request_json", lambda *args: response())
    assert runner.run_episode(row(), provider(), tmp_path)["status"] == "protocol_failure"


def test_environment_fault_is_not_model_failure(monkeypatch, tmp_path, fake_env):
    def broken(self, action):
        raise ValueError("Renderer defect")
    monkeypatch.setattr(FakeEnv, "step", broken)
    monkeypatch.setattr(runner, "request_json", lambda *args: response())
    result = runner.run_episode(row(), provider(), tmp_path)
    assert result["status"] == "adapted_environment_failure" and result["correct"] is None


def test_fixed_prefix_and_resume_do_not_duplicate_attempts(monkeypatch, tmp_path, fake_env):
    calls = []
    def request(*args):
        calls.append(1)
        return response()
    monkeypatch.setattr(runner, "request_json", request)
    rows = [row(f"test/number/{i}") for i in range(3)]
    first = runner.run_interactive(rows, provider(), tmp_path, limit=1, max_total_calls=3)
    assert len(first) == 1
    same = runner.run_interactive(rows, provider(), tmp_path, limit=1, max_total_calls=3)
    assert len(same) == 1 and len(calls) == 1
    last = runner.run_interactive(rows, provider(), tmp_path, limit=3, max_total_calls=3)
    assert len(last) == 3 and len(calls) == 3
    assert runner.load_json(tmp_path / "pending.json") == []


def test_total_call_ceiling_pauses_inside_episode(monkeypatch, tmp_path, fake_env):
    answers = iter([response("enter_number", {"number": 7}), response()])
    monkeypatch.setattr(runner, "request_json", lambda *args: next(answers))
    first = runner.run_interactive([row()], provider(), tmp_path, max_total_calls=1)
    assert first == [] and runner.load_json(tmp_path / "pending.json") == [row()["id"]]
    last = runner.run_interactive([row()], provider(), tmp_path, max_total_calls=2)
    assert len(last) == 1 and last[0]["model_calls"] == 2 and last[0]["correct"]


def test_recovery_refuses_changed_model_visible_request(monkeypatch, tmp_path, fake_env):
    monkeypatch.setattr(runner, "request_json", lambda *args: response("enter_number", {"number": 7}))
    runner.run_episode(row(), provider(), tmp_path, max_new_calls=1)
    request = runner.load_json(tmp_path / "request_01.json")
    request["messages"][0]["content"][0]["text"] += "changed"
    runner.atomic_json(tmp_path / "request_01.json", request)
    result = runner.run_episode(row(), provider(), tmp_path)
    assert result["status"] == "recovery_failure" and result["model_calls"] == 1


def test_incomplete_truncated_action_is_failed_without_execution(monkeypatch, tmp_path, fake_env):
    raw = response(finish="length")
    raw["choices"][0]["message"]["content"] = '<tool_call>{"name":"enter_number","arguments":{"number":12.5'
    monkeypatch.setattr(runner, "request_json", lambda *args: raw)
    result = runner.run_episode(row(), provider(), tmp_path)
    assert result["status"] == "output_budget_exhausted" and fake_env[0].actions == []


def test_usage_missing_stops_dispatch_without_guessing_tokens(monkeypatch, tmp_path, fake_env):
    raw = response()
    raw["usage"] = {}
    monkeypatch.setattr(runner, "request_json", lambda *args: raw)
    result = runner.run_episode(row(), provider(), tmp_path)
    assert result["status"] == "infrastructure_failure" and result["model_calls"] == 1


def test_episode_time_boundary_does_not_dispatch(monkeypatch, tmp_path, fake_env):
    def request(*args):
        raise AssertionError("Should not dispatch")
    monkeypatch.setattr(runner, "request_json", request)
    result = runner.run_episode(row(), provider(episode_timeout_seconds=0.000001), tmp_path)
    assert result["status"] == "episode_time_budget_exhausted" and result["model_calls"] == 0
