"""Verify adapted interfaces with new drawings and known action trajectories."""
import copy
from collections import Counter

import pytest
from PIL import Image, ImageChops

from tdsc_eval import interactive_envs
from tdsc_eval.calibration import native_modules
from tdsc_eval.interactive_envs import InvalidActionError, make_adapted_env
from tdsc_eval.tasks import ROOT, TYPES, load_json, score


ROWS = load_json(ROOT / "manifests/interactive_dev_v1.json")


def action(name, **kwargs):
    return native_modules().Action(name=name, kwargs=kwargs)


def click(position):
    return action("click", position=position)


def center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def submit(env):
    return env.step(click(center(env.public_metadata()["controls"]["submit"])))


def row_for(kind):
    return copy.deepcopy(next(row for row in ROWS if row["type"] == kind))


def known_actions(row):
    kind, gt = row["type"], row["gt"]
    if kind == "Dice_Count":
        return [action("enter_number", number=gt["sum"])]
    if kind == "Rotation_Match":
        return [action("enter_number", number=gt["correct_angle"])]
    if kind == "Click_Order":
        return [click([point["x"], point["y"]]) for point in gt["points_gt"]]
    if kind == "Place_Dot":
        return [click([gt["targets_position"]["x"], gt["targets_position"]["y"]])]
    if kind == "Pick_Area":
        return [click(center([*gt["area_box"][0], *gt["area_box"][1]]))]
    layout = row["layout"]
    x1, y1, x2, y2 = layout["grid_box"]
    return [click([x1 + (index % layout["cols"] + .5) * (x2-x1) / layout["cols"],
                   y1 + (index // layout["cols"] + .5) * (y2-y1) / layout["rows"]])
            for index in gt["correct_patches"]]


def test_development_material_is_fixed_independent_population():
    assert len(ROWS) == 12
    assert Counter(row["type"] for row in ROWS) == {kind: 2 for kind in TYPES}
    assert len({row["id"] for row in ROWS}) == 12
    formal = load_json(ROOT / "manifests/answer_20260913_conference.json")
    formal_paths = {path for row in formal for path in row["images"]}
    for row in ROWS:
        assert row["source"] == "interactive_dev_v1"
        assert row["partition"] == "development"
        assert row["scorer"] == "conference_v1"
        assert not formal_paths.intersection(row["images"])
        assert all(path.startswith("data/interactive_dev_v1/") for path in row["images"])


@pytest.mark.parametrize("row", ROWS, ids=lambda row: row["id"])
def test_known_actions_submit_using_original_scorer(row, tmp_path):
    env, instruction = make_adapted_env(row, tmp_path)
    metadata = env.public_metadata()
    assert "Submit" in instruction
    assert "gt" not in metadata and "reward" not in metadata
    response = env.reset(task_index=0)
    for index, path in enumerate(response.observation):
        image = Image.open(path).convert("RGB")
        original = Image.open(ROOT / row["images"][index]).convert("RGB")
        if index == metadata["main_image_index"]:
            assert image.crop((0, 0, *original.size)).tobytes() == original.tobytes()
            assert image.height > original.height
        else:
            assert image.size == original.size
            assert image.tobytes() == original.tobytes()
    actions = known_actions(row)
    assert len(actions) + 1 <= 30
    for next_action in actions:
        response = env.step(next_action)
        assert response.done is False
        assert response.reward == 0
    final = submit(env)
    assert final.done is True
    assert final.reward == 1
    assert score(row, env.final_answer()) is True
    for index, path in enumerate(final.observation):
        if index != metadata["main_image_index"]:
            assert Image.open(path).tobytes() == Image.open(ROOT / row["images"][index]).convert("RGB").tobytes()
    with pytest.raises(InvalidActionError, match="after submission"):
        env.step(click([1, 1]))


@pytest.mark.parametrize("kind", ["Dice_Count", "Rotation_Match", "Place_Dot", "Pick_Area", "Click_Order"])
def test_empty_submission_ends_without_answer_correction(kind, tmp_path):
    env, _ = make_adapted_env(row_for(kind), tmp_path)
    env.reset()
    response = submit(env)
    assert response.done is True and response.reward == 0


def test_grid_toggle_boundaries_and_maximum_budget(tmp_path):
    row = next(row for row in ROWS if row["type"] == "Patch_Select" and row["layout"]["rows"] == 5)
    env, _ = make_adapted_env(row, tmp_path)
    env.reset()
    env.step(click([0, 0]))
    assert env.final_answer()["indices"] == [0]
    env.step(click([0, 0]))
    assert env.final_answer()["indices"] == []
    env.step(click([400, 0]))  # Half-open image bounds, not an extra column.
    assert env.final_answer()["indices"] == []
    env.reset()
    for next_action in known_actions(row):
        env.step(next_action)
    assert env.final_answer()["indices"] == list(range(25))
    assert submit(env).reward == 1
    assert env.step_index == 26


def test_single_point_replacement_ordered_append_and_unsupported_noop(tmp_path):
    point, _ = make_adapted_env(row_for("Place_Dot"), tmp_path / "point")
    point.reset()
    point.step(click([50.5, 60.25]))
    point.step(click([90, 80]))
    assert point.final_answer()["point"] == {"x": 90, "y": 80}
    ordered, _ = make_adapted_env(row_for("Click_Order"), tmp_path / "ordered")
    ordered.reset()
    ordered.step(click([50.5, 60.25]))
    ordered.step(click([90, 80]))
    expected = [{"x": 50.5, "y": 60.25}, {"x": 90, "y": 80}]
    assert ordered.final_answer()["points"] == expected
    before = ordered.step_index
    response = ordered.step(action("drag", **{"from": [50, 60], "to": [90, 80]}))
    assert ordered.step_index == before + 1
    assert response.reward == 0 and response.done is False
    assert ordered.final_answer()["points"] == expected


def test_rotation_preserves_fraction_and_does_not_rotate_original(tmp_path):
    env, instruction = make_adapted_env(row_for("Rotation_Match"), tmp_path)
    assert "CLOCKWISE" in instruction or "clockwise" in instruction
    initial = env.reset()
    original = Image.open(initial.observation[1]).crop((0, 0, env.width, env.height)).tobytes()
    response = env.step(action("enter_number", number=12.5))
    assert env.final_answer()["angle"] == 12.5
    assert Image.open(response.observation[1]).crop((0, 0, env.width, env.height)).tobytes() == original
    assert Image.open(response.observation[1]).tobytes() != Image.open(initial.observation[1]).tobytes()
    env.step(action("enter_number", number=270.125))
    assert env.final_answer()["angle"] == 270.125


@pytest.mark.parametrize("kind,value", [
    ("Dice_Count", 9), ("Rotation_Match", 270.125), ("Visible_Eye_Count", 13),
    ("Dice_Count", 123456789012345678901234567890),
    ("Rotation_Match", 1.234567890123456e100),
])
@pytest.mark.parametrize("width", [300, 400, 1024, 3024])
def test_numeric_input_box_is_visible_without_moving_puzzle_or_submit(kind, value, width, tmp_path):
    row = row_for("Dice_Count" if kind == "Visible_Eye_Count" else kind)
    row["type"] = kind
    if kind == "Visible_Eye_Count":
        row.update(module="D", condition="D2", prompt="Count all visible fox eyes.")
    main_index = row["layout"]["main_image_index"]
    source = tmp_path / "numeric_source.png"
    original = Image.new("RGB", (width, 160), "#dce5ef")
    original.save(source)
    row["images"][main_index] = str(source)
    row["image_sizes"][main_index] = [width, 160]
    env, _ = make_adapted_env(row, tmp_path / "episode")
    metadata = env.public_metadata()
    assert metadata["environment"] == "cognition_adapted_v1.1_numeric_input_box"
    box = metadata["input_box"]
    submit_box = metadata["controls"]["submit"]
    canvas_width, canvas_height = metadata["observation_main_size"]
    margin = max(12, canvas_width // 50)
    button_width = min(max(100, canvas_width // 6), canvas_width // 3)
    # The existing submit contract remains identical across all supported sizes.
    assert submit_box == [canvas_width-margin-button_width, 160+max(120, width//8)//2,
                          canvas_width-margin, canvas_height-margin]
    assert 0 <= box[0] < box[2] < submit_box[0] < submit_box[2] <= canvas_width
    assert 160 <= box[1] == submit_box[1] < box[3] == submit_box[3] <= canvas_height
    empty = env.reset()
    empty_image = Image.open(empty.observation[main_index])
    inset = max(2, width // 600) + 1
    interior = [box[0]+inset, box[1]+inset, box[2]-inset, box[3]-inset]
    assert empty_image.getpixel((box[0], (box[1]+box[3])//2)) == (86, 101, 120)
    assert empty_image.crop(interior).getextrema() == ((255, 255), (255, 255), (255, 255))
    assert empty_image.crop((0, 0, width, 160)).tobytes() == original.tobytes()
    filled = env.step(action("enter_number", number=value))
    filled_image = Image.open(filled.observation[main_index])
    key = "angle" if kind == "Rotation_Match" else "value"
    assert env.final_answer()[key] == value
    assert filled_image.crop(interior).tobytes() != empty_image.crop(interior).tobytes()
    changed = ImageChops.difference(empty_image, filled_image).getbbox()
    assert box[0] < changed[0] < changed[2] < box[2]
    assert box[1] < changed[1] < changed[3] < box[3]
    assert filled_image.crop((0, 0, width, 160)).tobytes() == original.tobytes()
    answer = env.final_answer()
    clicked = env.step(click(center(box)))
    assert env.final_answer() == answer
    assert Image.open(clicked.observation[main_index]).tobytes() == filled_image.tobytes()


@pytest.mark.parametrize("number", [True, "12.5", float("inf"), float("nan")])
def test_invalid_numeric_actions_are_protocol_errors(number, tmp_path):
    env, _ = make_adapted_env(row_for("Rotation_Match"), tmp_path)
    env.reset()
    with pytest.raises(InvalidActionError):
        env.step(action("enter_number", number=number))
    assert env.final_answer()["angle"] is None


def test_count_rejects_fraction_without_truncation(tmp_path):
    env, _ = make_adapted_env(row_for("Dice_Count"), tmp_path)
    env.reset()
    with pytest.raises(InvalidActionError):
        env.step(action("enter_number", number=12.5))
    assert env.final_answer()["value"] is None
    env.step(action("enter_number", number=12.0))
    assert type(env.final_answer()["value"]) is int


def test_label_is_not_consulted_until_submit(monkeypatch, tmp_path):
    original = row_for("Patch_Select")
    different = copy.deepcopy(original)
    different["gt"] = {"arbitrary_private_answer": "NEVER_IN_OBSERVATION"}
    calls = []

    def scorer(row, answer):
        calls.append((row, answer))
        return "arbitrary_private_answer" not in row["gt"]

    monkeypatch.setattr(interactive_envs, "score", scorer)
    environments = []
    for index, row in enumerate([original, different]):
        env, instruction = make_adapted_env(row, tmp_path / str(index))
        env.reset()
        response = env.step(click([60, 60]))
        assert calls == []
        environments.append((env, instruction, Image.open(response.observation[0]).tobytes()))
    left, right = environments
    assert left[1:] == right[1:]
    assert left[0].public_metadata() == right[0].public_metadata()
    assert submit(left[0]).reward == 1
    assert submit(right[0]).reward == 0
    assert len(calls) == 2


def test_historical_grid_layout_comes_from_public_prompt(tmp_path):
    formal = load_json(ROOT / "manifests/answer_20260913_conference.json")
    row = copy.deepcopy(next(row for row in formal if row["type"] == "Patch_Select"))
    row["gt"] = {}  # Construction and observations cannot depend on answer-side layout.
    env, _ = make_adapted_env(row, tmp_path)
    assert env.public_metadata()["grid"] == {
        "rows": 5, "cols": 5, "grid_box": [0, 0, 3024, 3024],
        "source": "public_prompt_grid_dimensions"}


def defense_row(tmp_path, kind, gt):
    path = tmp_path / "source.png"
    Image.new("RGB", (1024, 1024), "#ddeeff").save(path)
    condition = {"Select_Animal": "D0", "Select_Animal_Optimized": "D1", "Visible_Eye_Count": "D2"}[kind]
    return {"id": "defense/test", "module": "D", "condition": condition, "source": "defense_v1",
            "type": kind, "puzzle_id": "test", "images": [str(path)],
            "image_sizes": [[1024, 1024]], "prompt": "Locate the visible eyes of the fox.",
            "layout": {"main_image_index": 0, "rows": 2, "cols": 3, "grid_box": [0, 0, 1024, 1024]},
            "gt": gt}


def test_defense_eyes_undo_and_hollow_markers(tmp_path):
    row = defense_row(tmp_path, "Select_Animal_Optimized", {"points": [{"x": 40, "y": 50}]})
    env, instruction = make_adapted_env(row, tmp_path / "episode")
    assert "Undo" in instruction
    initial = env.reset()
    response = env.step(click([40, 50]))
    image = Image.open(response.observation[0])
    assert image.getpixel((40, 50)) == Image.open(initial.observation[0]).getpixel((40, 50))
    assert image.getpixel((49, 50)) != Image.open(initial.observation[0]).getpixel((49, 50))
    env.step(click([100.5, 200.25]))
    assert len(env.final_answer()["points"]) == 2
    env.step(click(center(env.public_metadata()["controls"]["undo"])))
    assert env.final_answer() == {"points": [{"x": 40, "y": 50}]}
    assert all(box[1] >= 1024 for box in env.public_metadata()["controls"].values())


@pytest.mark.parametrize("kind,kwargs,answer", [
    ("Select_Animal", {"position": [100, 600]}, {"indices": [3]}),
    ("Visible_Eye_Count", {"number": 3}, {"value": 3}),
])
def test_defense_standard_answer_shapes(kind, kwargs, answer, tmp_path):
    row = defense_row(tmp_path, kind, {})
    env, _ = make_adapted_env(row, tmp_path / "episode")
    env.reset()
    env.step(action("click" if "position" in kwargs else "enter_number", **kwargs))
    assert env.final_answer() == answer


@pytest.mark.parametrize("kind,gt,good,bad", [
    ("Select_Animal", {"indices": [3]}, ("click", {"position": [100, 600]}),
     ("click", {"position": [100, 100]})),
    ("Select_Animal_Optimized", {"points": [{"x": 40, "y": 50}]},
     ("click", {"position": [40, 50]}), ("click", {"position": [100, 100]})),
    ("Visible_Eye_Count", {"value": 3}, ("enter_number", {"number": 3}),
     ("enter_number", {"number": 4})),
])
def test_defense_submit_uses_independent_scorer_and_ends_on_failure(kind, gt, good, bad, tmp_path):
    row = defense_row(tmp_path, kind, gt)
    for label, next_action, expected in (("good", good, 1), ("bad", bad, 0)):
        env, _ = make_adapted_env(row, tmp_path / label)
        env.reset()
        env.step(action(next_action[0], **next_action[1]))
        response = submit(env)
        assert response.done is True
        assert response.reward == expected
