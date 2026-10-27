import numpy as np
import pytest

from measured_space.geometry import Camera, OrientedBox, make_pose
from measured_space.qa_schema import QAItem, QuestionSpec, Referent, is_correct, read_jsonl, write_jsonl
from measured_space.scene import FrameView, SceneObject
from measured_space.visibility import object_in_view

CAM = Camera(500.0, 500.0, 319.5, 239.5, 640, 480)


def view_at_origin() -> FrameView:
    return FrameView("synthetic", "s0", "cam0", "0", CAM, np.eye(4), "none")


def test_unoccluded_box_in_front_is_fully_visible():
    obj = SceneObject("a", "box", OrientedBox([0, 0, 3.0], [0.5, 0.5, 0.5], np.eye(3)))
    depth = np.full((480, 640), 10.0)                        # background far behind the box
    seen, diag = object_in_view(view_at_origin(), obj, depth)
    assert seen.visible_fraction == pytest.approx(1.0)
    assert seen.truncated_fraction == 0.0
    assert diag["no_depth_fraction"] == 0.0
    x0, y0, x1, y1 = seen.bbox2d
    assert x0 < 319.5 < x1 and y0 < 239.5 < y1


def test_box_behind_an_occluder_is_not_visible():
    obj = SceneObject("a", "box", OrientedBox([0, 0, 3.0], [0.5, 0.5, 0.5], np.eye(3)))
    depth = np.full((480, 640), 1.0)                         # a wall at 1 m hides everything at 3 m
    seen, _ = object_in_view(view_at_origin(), obj, depth)
    assert seen.visible_fraction == 0.0 and seen.bbox2d is None


def test_box_half_outside_the_image_is_truncated():
    # Centre on the left image border: about half of the facing surface projects outside.
    z = 3.0
    x_border = (-0.5 - CAM.cx) / CAM.fx * z
    obj = SceneObject("a", "box", OrientedBox([x_border, 0, z], [0.6, 0.6, 0.01], np.eye(3)))
    seen, _ = object_in_view(view_at_origin(), obj, None)
    assert 0.35 < seen.truncated_fraction < 0.65
    assert seen.visible_fraction == pytest.approx(1 - seen.truncated_fraction)


def test_box_behind_camera_is_not_visible():
    obj = SceneObject("a", "box", OrientedBox([0, 0, -3.0], [0.5, 0.5, 0.5], np.eye(3)))
    seen, _ = object_in_view(view_at_origin(), obj, None)
    assert seen.visible_fraction == 0.0


def test_visibility_uses_the_world_pose():
    # Camera moved 5 m back along world -z: a box at the world origin is 5 m in front of it.
    view = FrameView("synthetic", "s0", "cam0", "0", CAM, make_pose(np.eye(3), [0, 0, -5.0]), "none")
    obj = SceneObject("a", "box", OrientedBox([0, 0, 0], [0.5, 0.5, 0.5], np.eye(3)))
    seen, _ = object_in_view(view, obj, np.full((240, 320), 20.0))   # lower-resolution depth map
    assert seen.visible_fraction == pytest.approx(1.0)


def _spec(answer_type: str, tolerance=None, choices=None) -> QuestionSpec:
    return QuestionSpec("t_v1", "t", "synthetic", "s0", "cam0", "0", "img.png", (), "q?", answer_type,
                        tolerance=tolerance or {}, choices=choices)


def test_is_correct_numeric_uses_either_bound():
    spec = _spec("number", {"abs": 0.1, "rel": 0.1})
    assert is_correct(spec, 2.15, 2.0)                  # within 10% (0.2) though not within 0.1 m
    assert is_correct(spec, 0.35, 0.3)                  # within 0.1 m though not within 10%
    assert not is_correct(spec, 2.25, 2.0)
    assert not is_correct(_spec("number", {"abs": None, "rel": 0.1}), 0.35, 0.3)
    assert not is_correct(spec, None, 2.0) and not is_correct(spec, "n/a", 2.0)


def test_is_correct_count_and_choice():
    assert is_correct(_spec("count"), 3, 3) and is_correct(_spec("count"), "3", 3)
    assert not is_correct(_spec("count"), 3.5, 3) and not is_correct(_spec("count"), 2, 3)
    choice = _spec("choice", choices=("left", "right"))
    assert is_correct(choice, " Left ", "left") and not is_correct(choice, "right", "left")


def test_qa_item_jsonl_round_trip(tmp_path):
    spec = QuestionSpec("dist_center_v1", "distance_center", "synthetic", "s0", "cam0", "0", "img.png",
                        (Referent("a", "chair", "the chair"), Referent("b", "table", "the table")),
                        "How far is the chair from the table (centre to centre), in metres?", "number",
                        unit="m", tolerance={"abs": 0.1, "rel": 0.1})
    item = QAItem(spec, 1.25, "oracle", ({"object_id": "a", "center": np.array([0.0, 1.0, 2.0])},),
                  {"margin": np.float64(0.3)})
    assert write_jsonl([item], tmp_path / "qa.jsonl") == 1
    (back,) = list(read_jsonl(tmp_path / "qa.jsonl"))
    assert back.id == item.id and back.spec == spec
    assert back.answer == 1.25 and back.objects[0]["center"] == [0.0, 1.0, 2.0]
    assert back.extra["margin"] == 0.3
