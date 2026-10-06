import json

import pytest

from tdsc_eval import defense_eval as de
from tdsc_eval import providers


def test_label_blind_d_reading():
    for kind, text, key, value in (
        ("Select_Animal", "<answer>[0, 4]</answer>", "indices", [0, 4]),
        ("Select_Animal_Optimized", "<answer>[(12.5, 40), (55, 66)]</answer>", "points",
         [{"x": 12.5, "y": 40}, {"x": 55, "y": 66}]),
        ("Visible_Eye_Count", "<answer>4</answer>", "value", 4),
    ):
        first = de.extract_defense_answer(text, {"type": kind, "gt": "SECRET_A"})
        second = de.extract_defense_answer(text, {"type": kind, "gt": "SECRET_B"})
        assert first == second
        assert first[0][key] == value and first[1] is None


def test_d_point_group_and_flat_notation_preserves_every_value():
    row = {"type": "Select_Animal_Optimized", "gt": "not inspected"}
    for text in ("<answer>[1.5,2,3,4,3,4]</answer>",
                 "<answer>[[(1.5,2),(3,4)],[(3,4)]]</answer>"):
        answer, error, method = de.extract_defense_answer(text, row)
        assert error is None
        assert answer["points"] == [{"x": 1.5, "y": 2}, {"x": 3, "y": 4}, {"x": 3, "y": 4}]
    assert de.extract_defense_answer("<answer>[1,2,3,4]</answer>", {"type": "Select_Animal"})[0]["indices"] == [1,2,3,4]


@pytest.mark.parametrize("text", [
    "<answer>[1,2,3]</answer>",
    "<answer>[[1,2,3,4],[5,6,7,8]]</answer>",
    "<answer>[1,2,[3,4]]</answer>",
    '<answer>{"bboxes":[[1,2,3,4]]}</answer>',
    '<answer>{"points":[1,2,3,4],"units":"normalized"}</answer>',
    "<answer>[1,2,3,4]",
])
def test_d_point_normalization_does_not_invent_or_select_coordinates(text):
    assert de.extract_defense_answer(text, {"type": "Select_Animal_Optimized"})[0] is None


@pytest.fixture
def static_setup(tmp_path, monkeypatch):
    (tmp_path / "input.png").write_bytes(b"development image bytes")
    monkeypatch.setattr(providers, "ROOT", tmp_path)
    row = {"id": "defense_v1/D2/image1", "module": "D", "condition": "D2", "base_id": "image1",
           "type": "Visible_Eye_Count", "source": "defense_v1", "puzzle_id": "image1",
           "group_id": "image1", "partition": "development", "images": ["input.png"],
           "image_sizes": [[1024, 1024]], "prompt": "Count visible cat eyes.",
           "gt": {"value": 4}, "scorer": "defense_v1"}
    config = {"max_output_tokens": 16384, "request_timeout_seconds": 600,
              "captchamind": {"model": "test-model", "temperature": 0}}
    provider = providers.Provider("captchamind", config, base_url="http://unused/v1",
                                  prompt_builder=de.defense_prompt)
    raw = {"choices": [{"message": {"content": "<answer>4</answer>"}, "finish_reason": "length"}],
           "usage": {"prompt_tokens": 10, "completion_tokens": 12}}
    return row, config, provider, raw, tmp_path / "run"


def test_request_saved_before_dispatch_and_length_answer_kept(static_setup, monkeypatch):
    row, config, provider, raw, output = static_setup
    def transport(url, payload, headers, timeout):
        saved = output / "requests" / row["id"] / "request.json"
        assert json.loads(saved.read_text()) == payload
        assert "gt" not in json.dumps(payload) and "SECRET" not in json.dumps(payload)
        return raw
    monkeypatch.setattr(providers, "request_json", transport)
    records = de.run_defense_static([row], provider, output, config)
    assert records[0]["correct"] is True
    assert records[0]["truncated"] is True
    assert (output / "requests" / row["id"] / "response.json").exists()


def test_response_recovery_avoids_second_generation(static_setup, monkeypatch):
    row, config, provider, raw, output = static_setup
    calls = []
    monkeypatch.setattr(providers, "request_json", lambda *a: calls.append(a) or raw)
    de.run_defense_static([row], provider, output, config)
    (output / "attempts.jsonl").unlink()  # interrupted after response, before final record
    records = de.run_defense_static([row], provider, output, config)
    assert len(calls) == 1 and len(records) == 1
    assert records[0]["elapsed_seconds"] >= 0


def test_pending_request_stops_without_retry(static_setup, monkeypatch):
    row, config, provider, raw, output = static_setup
    calls = []
    def failed(*args):
        calls.append(1)
        raise providers.ProviderError("Transport failure: TimeoutError")
    monkeypatch.setattr(providers, "request_json", failed)
    assert de.run_defense_static([row], provider, output, config) == []
    assert de.run_defense_static([row], provider, output, config) == []
    assert len(calls) == 1
    assert json.loads((output / "progress.json").read_text())["status"] == "pending_request"


def test_completed_run_does_not_repeat_and_definition_is_frozen(static_setup, monkeypatch):
    row, config, provider, raw, output = static_setup
    calls = []
    monkeypatch.setattr(providers, "request_json", lambda *a: calls.append(a) or raw)
    de.run_defense_static([row], provider, output, config)
    de.run_defense_static([row], provider, output, config)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="definition changed"):
        de.run_defense_static([{**row, "prompt": "changed"}], provider, output, config)
