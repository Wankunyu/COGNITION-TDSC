import pytest

from tdsc_eval.answer_extraction import extract_answer


class NoGroundTruth(dict):
    def __getitem__(self, key):
        assert key == "type", "Extraction must not consult answer metadata"
        return super().__getitem__(key)


def read(text, kind):
    return extract_answer(text, NoGroundTruth(type=kind))


@pytest.mark.parametrize("text,kind,expected", [
    ('<answer>25</answer>', "Dice_Count", {"answer_type": "number", "value": 25}),
    ('Final answer: <answer>25</answer>', "Dice_Count", {"answer_type": "number", "value": 25}),
    ('Final answer: 270 degrees', "Rotation_Match", {"answer_type": "rotation", "angle": 270}),
    ('<think>I counted 4 and 19; maybe 23.</think>Answer: 25', "Dice_Count",
     {"answer_type": "number", "value": 25}),
    ('```json\n{"point":{"x":-3.5,"y":1001}}\n```', "Place_Dot",
     {"answer_type": "single_point", "point": {"x": -3.5, "y": 1001}}),
    ('(605, 300)', "Pick_Area", {"answer_type": "single_point", "point": {"x": 605, "y": 300}}),
    ('Answer: [(5, 9), (2, 4)]', "Click_Order",
     {"answer_type": "click_order", "points": [{"x": 5, "y": 9}, {"x": 2, "y": 4}]}),
    ('<answer>[12, 8, 12, -1, 30]</answer>', "Patch_Select",
     {"answer_type": "multi_select", "indices": [12, 8, 12, -1, 30]}),
    ('<answer>[]</answer>', "Patch_Select", {"answer_type": "multi_select", "indices": []}),
    ('Answer: x=605, y=300', "Place_Dot",
     {"answer_type": "single_point", "point": {"x": 605, "y": 300}}),
])
def test_unambiguous_answers_preserve_values_without_ground_truth(text, kind, expected):
    answer, error, method = read(text, kind)
    assert error is None and answer == expected
    assert method


def test_observed_omitted_y_keys_preserve_coordinate_count_order_and_values():
    text = ('<tool_call>{"answer_type":"click_order","points":'
            '[{"x":414,232},{"x":527,405},{"x":123,153},{"x":394,485}]}</tool_call>')
    answer, error, method = read(text, "Click_Order")
    assert error is None
    assert answer["points"] == [{"x": 414, "y": 232}, {"x": 527, "y": 405},
                                {"x": 123, "y": 153}, {"x": 394, "y": 485}]
    assert "omitted_y_key" in method


@pytest.mark.parametrize("text", [
    '<tool_call>{"name":"enter_number","arguments":{"number":24}}</tool_call>',
    'enter_number(number=24)', 'enter_number(24)',
])
def test_enter_number_is_read_without_execution(text):
    assert read(text, "Dice_Count")[:2] == ({"answer_type": "number", "value": 24}, None)


def test_native_clicks_are_read_as_full_ordered_sequence():
    text = ('<tool_call>{"name":"click","arguments":{"position":[605,300]}}</tool_call>'
            '<tool_call>{"name":"click","arguments":{"position":[9,-4]}}</tool_call>')
    assert read(text, "Click_Order")[:2] == (
        {"answer_type": "click_order", "points": [{"x": 605, "y": 300}, {"x": 9, "y": -4}]}, None)
    assert read(text, "Place_Dot")[1] == "ambiguous_answer"


def test_function_notation_click_is_literal_only():
    assert read('click(position=[605,300])', "Pick_Area")[:2] == (
        {"answer_type": "single_point", "point": {"x": 605, "y": 300}}, None)
    assert read('click(position=[100+5,300])', "Pick_Area")[1] == "unparsed_answer"


@pytest.mark.parametrize("text", [
    '<answer>25</answer><answer>24</answer>',
    'Answer: 25\nFinal answer: 24',
    '<answer>25</answer>Final answer: 24',
    'enter_number(25)\nenter_number(24)',
    '{"value":25}\n{"value":24}',
])
def test_conflicting_explicit_final_answers_are_not_chosen(text):
    assert read(text, "Dice_Count")[1] == "ambiguous_answer"


def test_repeated_identical_final_answer_is_not_ambiguous():
    assert read('<answer>25</answer>Final answer: 25', "Dice_Count")[:2] == (
        {"answer_type": "number", "value": 25}, None)


@pytest.mark.parametrize("text,kind", [
    ('There are 4 groups with 6 dice. I think 24 is likely.', "Dice_Count"),
    ('The original is 45 degrees; the target may be 270 degrees.', "Rotation_Match"),
    ('I saw bicycles in 3 cells and motorcycles in 2.', "Patch_Select"),
    ('<think>The answer is 25</think>', "Dice_Count"),
    ('<think>25', "Dice_Count"),
    ('25.5', "Dice_Count"),
    ('[true, 2]', "Patch_Select"),
    ('{"x":605,300,200}', "Place_Dot"),
])
def test_explanation_numbers_and_ambiguous_repairs_are_not_guessed(text, kind):
    assert read(text, kind)[1] == "unparsed_answer"


def test_patch_click_is_not_mistaken_for_complete_index_set():
    answer, error, _ = read('click(position=[605,300])', "Patch_Select")
    assert answer is None and error == "needs_review"


def test_explicit_final_ignores_prior_explanation_numbers():
    assert read('I first guessed 15, then checked 6 panels.\nFinal answer: 25', "Dice_Count")[:2] == (
        {"answer_type": "number", "value": 25}, None)


@pytest.mark.parametrize("text", [
    '<answer>25</answer><answer>24',
    '<answer>25</answer><answer',
    '<answer>25</answer></answer>',
    '<answer><answer>25</answer></answer>',
    '</answer><answer>25</answer>',
])
def test_residual_unfinished_or_nested_final_tags_require_review(text):
    answer, error, method = read(text, "Dice_Count")
    assert answer is None and error == "needs_review"
    assert method == "malformed_answer_tags"


@pytest.mark.parametrize("text,kind", [
    ('{"angle":1.5707963267948966,"unit":"radians"}', "Rotation_Match"),
    ('{"angle":0.25,"units":"turns"}', "Rotation_Match"),
    ('{"point":{"x":0.5,"y":0.3},"unit":"normalized"}', "Place_Dot"),
    ('{"point":{"x":0.5,"y":0.3,"unit":"normalized"}}', "Pick_Area"),
    ('{"points":[{"x":0.5,"y":0.3,"units":"normalized"}]}', "Click_Order"),
    ('{"point":{"x":0.5,"y":0.3},"normalized":true}', "Place_Dot"),
    ('{"point":{"x":0.5,"y":0.3},"coordinate_system":"normalized"}', "Place_Dot"),
    ('<tool_call>{"name":"click","arguments":{"position":[0.5,0.3],'
     '"unit":"normalized"}}</tool_call>', "Pick_Area"),
    ('click(position=[0.5,0.3],unit="normalized")', "Pick_Area"),
])
def test_non_task_units_are_not_silently_dropped_or_converted(text, kind):
    answer, error, method = read(text, kind)
    assert answer is None and error == "needs_review"
    assert "explicit_unit" in method


@pytest.mark.parametrize("unit", ["degree", "degrees", "°"])
def test_explicit_degree_units_preserve_angle(unit):
    text = '{"angle":1.5707963267948966,"unit":"' + unit + '"}'
    assert read(text, "Rotation_Match")[:2] == (
        {"answer_type": "rotation", "angle": 1.5707963267948966}, None)


@pytest.mark.parametrize("unit", ["pixel", "pixels", "px"])
def test_explicit_pixel_units_preserve_coordinates(unit):
    text = '{"point":{"x":0.5,"y":0.3,"unit":"' + unit + '"}}'
    assert read(text, "Place_Dot")[:2] == (
        {"answer_type": "single_point", "point": {"x": 0.5, "y": 0.3}}, None)
