import copy
import itertools

from PIL import Image, ImageFilter

from tdsc_eval.defense_tasks import DEV_IDS, point_match_details, score_defense
from tdsc_eval.tasks import ROOT, load_json


def points(*pairs):
    return [{"x": x, "y": y} for x, y in pairs]


def test_maximum_matching_repairs_greedy_false_negative_and_is_order_independent():
    gold = points((100, 100), (120, 100))
    predicted = points((110, 100), (80, 100))
    row = {"condition": "D1", "gt": {"points": gold}}
    assert not score_defense(row, {"points": predicted}, radius=30, legacy=True)
    for gt_order in itertools.permutations(gold):
        for pred_order in itertools.permutations(predicted):
            reordered = {"condition": "D1", "gt": {"points": list(gt_order)}}
            assert score_defense(reordered, {"points": list(pred_order)}, radius=30)


def test_radius_monotonicity_and_inclusive_boundary():
    row = {"condition": "D1", "gt": {"points": points((100, 100), (120, 100))}}
    answer = {"points": points((110, 100), (80, 100))}
    assert [score_defense(row, answer, radius=r) for r in (15, 30, 45)] == [False, True, True]
    boundary = {"condition": "D1", "gt": {"points": points((100, 100))}}
    assert score_defense(boundary, {"points": points((109, 112))}, radius=15)
    assert not score_defense(boundary, {"points": points((109, 112.001))}, radius=15)


def test_point_scoring_requires_all_eyes_and_no_extras():
    gold = points((100, 100), (200, 100))
    row = {"condition": "D1", "gt": {"points": gold}}
    assert score_defense(row, {"points": gold})
    assert not score_defense(row, {"points": gold[:1]})
    assert not score_defense(row, {"points": gold + [gold[0]]})
    assert not score_defense(row, {"points": points((100, 100), (float("nan"), 100))})
    assert point_match_details(gold + [gold[0]], gold)["extra_pred"] == 1


def test_legacy_is_the_frozen_source_nearest_greedy():
    from tdsc_eval._conference.run_eval import _match_points
    gold = points((100, 100), (120, 100), (140, 100))
    for predicted in (gold, gold[:1], points((110, 100), (80, 100), (150, 100))):
        for radius in (15, 30, 45):
            result = point_match_details(predicted, gold, radius, legacy=True)
            assert (result["matched"], result["missing_gt"], result["extra_pred"]) == _match_points(predicted, gold, radius)


def test_grid_and_count_do_not_accept_coercion_or_extra_values():
    grid = {"condition": "D0", "gt": {"indices": [4]}, "layout": {"rows": 2, "cols": 3}}
    assert score_defense(grid, {"indices": [4, 4]})
    for value in ([4, 1], ["4"], [4.0], [True], [6], []):
        assert not score_defense(grid, {"indices": value})
    count = {"condition": "D2", "gt": {"value": 2}}
    assert score_defense(count, {"value": 2})
    for value in ("2", 2.1, True, float("inf")):
        assert not score_defense(count, {"value": value})


def test_manifest_pairing_labels_and_fixed_split():
    dev = load_json(ROOT / "manifests/defense_v1_dev.json")
    evaluation = load_json(ROOT / "manifests/defense_v1_eval.json")
    assert len(dev) == 24 and len(evaluation) == 96
    assert {r["base_id"] for r in dev} == {f"image{i}" for i in DEV_IDS}
    assert not {r["base_id"] for r in dev} & {r["base_id"] for r in evaluation}
    for base_id in {r["base_id"] for r in dev + evaluation}:
        rows = {r["condition"]: r for r in dev + evaluation if r["base_id"] == base_id}
        assert set(rows) == {"D0", "D1", "D2", "D3"}
        assert len({r["target"] for r in rows.values()}) == 1
        assert len({r["group_id"] for r in rows.values()}) == 1
        assert rows["D0"]["gt"] == rows["D3"]["gt"]
        assert len(rows["D1"]["gt"]["points"]) == rows["D2"]["gt"]["value"]
        for row in rows.values():
            assert score_defense(row, copy.deepcopy(row["gt"]))
            assert row["layout"] == {"rows": 2, "cols": 3, "main_image_index": 0, "grid_box": [0, 0, 1024, 1024]}
    d1 = {r["base_id"]: r for r in dev if r["condition"] == "D1"}
    assert d1["image8"]["gt"]["points"] == points((465, 615))
    assert d1["image23"]["gt"]["points"] == points((190, 128), (237, 128))
    assert d1["image27"]["target"] == "tiger"
    assert d1["image27"]["gt"]["points"] == points((413, 583), (444, 584))


def test_copied_images_and_fixed_blur_match_sources():
    for i in range(1, 31):
        name = f"image{i}.png"
        source = ROOT.parent / "conference/captcha_data/Select_Animal" / name
        original = ROOT / "data/defense_v1/original" / name
        blurred = ROOT / "data/defense_v1/blur_radius2" / name
        assert source.read_bytes() == original.read_bytes()
        with Image.open(original) as image, Image.open(blurred) as control:
            assert image.size == control.size == (1024, 1024)
            assert image.filter(ImageFilter.GaussianBlur(2.0)).tobytes() == control.tobytes()
