"""Read a single generated answer without images, labels, or answer-dependent repair.

``extract_answer(text, row)`` returns ``(answer, error, method)``. Only ``row['type']``
is inspected. The result uses the existing scorer's answer keys. A failure means
the answer needs a separate reading check, not that the model solved incorrectly.
"""
import ast
import json
import math
import re

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
FINAL_MARKER = re.compile(r"(?im)(?<![\w])(?:final\s+answer|answer)\s*\*{0,2}\s*:\s*\*{0,2}")
OMITTED_Y = re.compile(
    r"\{\s*(?P<quote>['\"])x(?P=quote)\s*:\s*(?P<x>" + NUMBER
    + r")\s*,\s*(?P<y>" + NUMBER + r")\s*\}"
)
ANSWER_TYPES = {
    "Dice_Count": "number", "Place_Dot": "single_point", "Pick_Area": "single_point",
    "Click_Order": "click_order", "Patch_Select": "multi_select", "Rotation_Match": "rotation",
}


class _UnitReview(ValueError):
    """An explicit unit needs a separate reading, not an implicit conversion."""


def _check_units(value, kind):
    if kind == "Rotation_Match":
        allowed = {"degree", "degrees", "°"}
    elif kind in ("Place_Dot", "Pick_Area", "Click_Order"):
        allowed = {"pixel", "pixels", "px"}
    else:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).lower()
            if normalized_key in {
                "unit", "units", "coordinate_unit", "coordinate_units", "coordinate_system",
                "angle_unit", "angle_units",
            } and (not isinstance(item, str) or item.strip().lower() not in allowed):
                raise _UnitReview
            if normalized_key == "normalized" and item is not False:
                raise _UnitReview
            _check_units(item, kind)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_units(item, kind)


def _valid_answer_tags(text):
    """Reject unfinished, unmatched, or nested tags before selecting a final answer."""
    pattern = r"</?answer\b[^>]*>"
    opened = False
    for match in re.finditer(pattern, text, flags=re.I):
        token = match[0].lower()
        if token == "<answer>" and not opened:
            opened = True
        elif token == "</answer>" and opened:
            opened = False
        else:
            return False
    residual = re.sub(pattern, "", text, flags=re.I)
    return not opened and re.search(r"</?answer\b", residual, flags=re.I) is None


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _point(value):
    _check_units(value, "Place_Dot")
    if isinstance(value, dict) and _number(value.get("x")) and _number(value.get("y")):
        return {"x": value["x"], "y": value["y"]}
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(map(_number, value)):
        return {"x": value[0], "y": value[1]}
    return None


def _literal(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError, TypeError, RecursionError):
            return None


def _json_sequence(text):
    """Read concatenated complete JSON objects, without searching inside prose."""
    decoder, values, remaining = json.JSONDecoder(), [], text
    try:
        while remaining.strip():
            remaining = remaining.lstrip()
            value, end = decoder.raw_decode(remaining)
            if not isinstance(value, dict):
                return None
            values.append(value)
            remaining = remaining[end:]
        return values if len(values) > 1 else None
    except ValueError:
        return None


def _normalize(value, kind):
    """Normalize representation only; do not clip, sort, renumber, or score."""
    _check_units(value, kind)
    if isinstance(value, dict):
        fields = {
            "Dice_Count": "value", "Place_Dot": "point", "Pick_Area": "point",
            "Click_Order": "points", "Patch_Select": "indices", "Rotation_Match": "angle",
        }
        if fields[kind] in value:
            value = value[fields[kind]]
    result = {"answer_type": ANSWER_TYPES[kind]}
    if kind == "Dice_Count" and _number(value) and int(value) == value:
        return {**result, "value": value}
    if kind == "Rotation_Match" and _number(value):
        return {**result, "angle": value}
    if kind in ("Place_Dot", "Pick_Area") and (point := _point(value)) is not None:
        return {**result, "point": point}
    if kind == "Click_Order" and isinstance(value, (list, tuple)):
        points = [_point(item) for item in value]
        if all(point is not None for point in points):
            return {**result, "points": points}
    if kind == "Patch_Select" and isinstance(value, (list, tuple)) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        # Bounds are part of scoring; ground truth and grid metadata are not consulted.
        return {**result, "indices": list(value)}
    return None


def _tool(value):
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    arguments = value.get("arguments", value.get("kwargs"))
    if name in ("click", "enter_number") and isinstance(arguments, dict):
        return name, arguments
    return None


def _function_calls(text):
    """Read literal arguments from function notation; never execute the call."""
    try:
        statements = ast.parse(text, mode="exec").body
        calls = []
        for statement in statements:
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                return None
            call = statement.value
            if not isinstance(call.func, ast.Name) or call.func.id not in ("click", "enter_number"):
                return None
            if len(call.args) > 1 or any(keyword.arg is None for keyword in call.keywords):
                return None
            arguments = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
            if call.args:
                key = "position" if call.func.id == "click" else "number"
                if key in arguments:
                    return None
                arguments[key] = ast.literal_eval(call.args[0])
            calls.append((call.func.id, arguments))
        return calls or None
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return None


def _from_tools(calls, kind):
    for _, arguments in calls:
        _check_units(arguments, kind)
    if kind == "Patch_Select":
        return None, "needs_review", "native_click_not_patch_indices"
    if kind == "Dice_Count":
        values = [arguments.get("number") for name, arguments in calls if name == "enter_number"]
        answers = [_normalize(value, kind) for value in values]
        if not answers or any(answer is None for answer in answers):
            return None, "unparsed_answer", "native_arguments"
        return _agree(answers, "native_enter_number")
    if kind in ("Place_Dot", "Pick_Area", "Click_Order"):
        if any(name != "click" for name, _ in calls):
            return None, "unparsed_answer", "native_arguments"
        points = [_point(arguments.get("position", arguments)) for _, arguments in calls]
        if any(point is None for point in points):
            return None, "unparsed_answer", "native_arguments"
        if kind == "Click_Order":
            return _normalize(points, kind), None, "native_click_sequence"
        return _agree([_normalize(point, kind) for point in points], "native_click")
    return None, "unparsed_answer", "native_arguments"


def _agree(answers, method):
    if any(answer != answers[0] for answer in answers[1:]):
        return None, "ambiguous_answer", method
    return answers[0], None, method


def _read_segment(segment, kind):
    cleaned = segment.strip().strip("*").strip()
    # Fences are presentation, including separately fenced tool calls.
    cleaned = re.sub(r"(?im)^\s*```(?:json|python|text)?\s*\n?", "", cleaned).strip()
    cleaned = re.sub(r"(?m)^\s*```\s*$", "", cleaned).strip()
    repaired = OMITTED_Y.sub(lambda match: '{"x":' + match["x"] + ',"y":' + match["y"] + '}', cleaned)
    repair_method = "+omitted_y_key" if repaired != cleaned else ""
    cleaned = repaired

    envelopes = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", cleaned, flags=re.S))
    if envelopes:
        outside = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.S).strip()
        if outside:
            return None, "unparsed_answer", "mixed_tool_text"
        values = [_literal(match[1]) for match in envelopes]
        _check_units(values, kind)
        calls = [_tool(value) for value in values]
        if all(call is not None for call in calls):
            answer, error, method = _from_tools(calls, kind)
            return answer, error, method + repair_method
        answers = [_normalize(value, kind) for value in values]
        if all(answer is not None for answer in answers):
            answer, error, method = _agree(answers, "tool_envelope")
            return answer, error, method + repair_method
        return None, "unparsed_answer", "tool_envelope"

    value = _literal(cleaned)
    _check_units(value, kind)
    calls = [_tool(item) for item in value] if isinstance(value, list) else [_tool(value)]
    if calls and all(call is not None for call in calls):
        answer, error, method = _from_tools(calls, kind)
        return answer, error, method + repair_method
    answer = _normalize(value, kind)
    if answer is not None:
        return answer, None, "literal" + repair_method
    values = _json_sequence(cleaned)
    if values:
        _check_units(values, kind)
        calls = [_tool(item) for item in values]
        if all(call is not None for call in calls):
            return _from_tools(calls, kind)
        answers = [_normalize(item, kind) for item in values]
        if all(answer is not None for answer in answers):
            return _agree(answers, "json_sequence" + repair_method)
    calls = _function_calls(cleaned)
    if calls:
        return _from_tools(calls, kind)

    # Only a whole, explicit numeric answer is eligible, not numbers mined from prose.
    suffix = r"(?:\s*(?:degrees?|°))?" if kind == "Rotation_Match" else ""
    if kind in ("Dice_Count", "Rotation_Match") and re.fullmatch(NUMBER + suffix, cleaned, flags=re.I):
        numeric = re.match(NUMBER, cleaned)[0]
        answer = _normalize(_literal(numeric), kind)
        if answer is not None:
            return answer, None, "plain_number"
    if kind in ("Place_Dot", "Pick_Area"):
        point = re.fullmatch(r"x\s*[:=]\s*(" + NUMBER + r")\s*[,;]?\s*y\s*[:=]\s*(" + NUMBER + r")", cleaned, flags=re.I)
        if point:
            return _normalize([_literal(point[1]), _literal(point[2])], kind), None, "named_coordinates"
    return None, "unparsed_answer", "unparsed"


def extract_answer(text, row):
    """Return ``(answer, error, method)`` without inspecting images or ground truth."""
    kind = row["type"]
    if kind not in ANSWER_TYPES:
        return None, "unsupported_type", "none"
    if not isinstance(text, str):
        return None, "unparsed_answer", "none"
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.S | re.I)
    cleaned = re.sub(r"<think\b[^>]*>.*$", "", cleaned, flags=re.S | re.I).strip()
    if not _valid_answer_tags(cleaned):
        return None, "needs_review", "malformed_answer_tags"
    tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", cleaned, flags=re.S | re.I)
    # A marker immediately wrapping an answer tag is the same answer, not an
    # additional empty answer. Other explicit final answers still must agree.
    outside = re.sub(
        FINAL_MARKER.pattern + r"\s*<answer>.*?</answer>", "", cleaned, flags=re.S | re.I
    )
    outside = re.sub(r"<answer>.*?</answer>", "", outside, flags=re.S | re.I)
    markers = list(FINAL_MARKER.finditer(outside))
    marked = [outside[match.end():markers[i + 1].start() if i + 1 < len(markers) else len(outside)]
              for i, match in enumerate(markers)]
    segments = tagged + marked
    selection = "explicit_final" if segments else "whole_response"
    if not segments:
        segments = [cleaned]
    try:
        results = [_read_segment(segment, kind) for segment in segments]
    except _UnitReview:
        return None, "needs_review", selection + "/explicit_unit"
    readable = [answer for answer, error, _ in results if error is None]
    if len(readable) > 1 and any(answer != readable[0] for answer in readable[1:]):
        return None, "ambiguous_answer", selection
    for _, error, method in results:
        if error is not None:
            return None, error, selection + "/" + method
    answer, error, method = _agree(readable, selection + "/" + "+".join(sorted({r[2] for r in results})))
    return answer, error, method
