"""Tests for matching detections to oracle objects: prism IoU, the gated 3D assignment and the 2D association.

Boxes are axis-aligned or yawed about world +z with hand-computable overlaps; 2D boxes are pixel rectangles with
hand-computable IoU.
"""

from __future__ import annotations

import numpy as np
import pytest

from measured_space.geometry import OrientedBox, axis_angle_to_matrix
from measured_space.matching import (
    Match2DConfig,
    MatchConfig,
    box_iou_2d,
    clip_convex,
    convex_hull_2d,
    match_detections,
    match_detections_2d,
    polygon_area,
    prism_iou,
)
from measured_space.scene import SceneObject


def box(center: tuple[float, float, float], size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        yaw_deg: float = 0.0) -> OrientedBox:
    """A box yawed about +z.

    Parameters
    ----------
    center : tuple of float
        Centre (m).
    size : tuple of float
        Full side lengths (m).
    yaw_deg : float
        Yaw about +z, degrees.

    Returns
    -------
    OrientedBox
        The box.
    """
    return OrientedBox(center, size, axis_angle_to_matrix([0.0, 0.0, np.radians(yaw_deg)]))


def obj(object_id: str, label: str, center: tuple[float, float, float],
        size: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> SceneObject:
    """An oracle object with an axis-aligned box.

    Parameters
    ----------
    object_id : str
        Its id.
    label : str
        Its label.
    center : tuple of float
        Box centre (m).
    size : tuple of float
        Full side lengths (m).

    Returns
    -------
    SceneObject
        The object.
    """
    return SceneObject(object_id, label, box(center, size))


def test_polygon_helpers() -> None:
    """Hull drops duplicates and interior points; clipping two unit squares offset by half gives half a square."""
    hull = convex_hull_2d([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0, 0]])
    assert len(hull) == 4 and polygon_area(hull) == pytest.approx(1.0)
    a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert polygon_area(clip_convex(a, a + [0.5, 0.0])) == pytest.approx(0.5)
    assert polygon_area(clip_convex(a, a + [2.0, 0.0])) == pytest.approx(0.0)


@pytest.mark.parametrize("other, expected", [
    (box((0.0, 0.0, 0.0)), 1.0),                                  # identical
    (box((0.5, 0.0, 0.0)), 1.0 / 3.0),                            # half overlap along x
    (box((0.0, 0.0, 0.5)), 1.0 / 3.0),                            # half overlap along up
    (box((0.0, 0.0, 0.0), yaw_deg=90.0), 1.0),                    # a cube yawed by 90 degrees is itself
    (box((0.0, 0.0, 0.0), yaw_deg=45.0), np.sqrt(2.0) / 2.0),     # octagon 2(sqrt2-1) over 2 - octagon
    (box((3.0, 0.0, 0.0)), 0.0),                                  # disjoint
])
def test_prism_iou(other: OrientedBox, expected: float) -> None:
    """Prism IoU of a unit cube with shifted or yawed copies matches the closed-form value.

    Parameters
    ----------
    other : OrientedBox
        The second box.
    expected : float
        Its IoU with the unit cube at the origin.
    """
    assert prism_iou(box((0.0, 0.0, 0.0)), other) == pytest.approx(expected, abs=1e-9)


def test_hungarian_maximises_matches_where_greedy_would_not() -> None:
    """Greedy closest-first would pair d0 with A and leave d1 unmatched; the assignment matches both."""
    objects = [obj("A", "chair", (0.0, 0.0, 0.0)), obj("B", "chair", (1.0, 0.0, 0.0))]
    detections = [box((0.1, 0.0, 0.0)), box((-0.3, 0.0, 0.0))]        # d0 is near both, d1 only near A
    config = MatchConfig(min_gate=1.0, gate_diagonal_fraction=0.0)    # d1-B (1.3 m) is outside the gate
    result = match_detections(detections, [("chair",), ("chair",)], objects, config)
    assert {(m.detection, m.object_id) for m in result.matches} == {(0, "B"), (1, "A")}
    assert result.missed == () and result.false_positives == ()
    by_det = result.by_detection()
    assert by_det[0].distance == pytest.approx(0.9) and by_det[1].distance == pytest.approx(0.3)
    assert by_det[1].iou == pytest.approx(0.7 / 1.3) and by_det[0].gate == pytest.approx(1.0)


def test_hungarian_minimises_total_distance() -> None:
    """With both pairings feasible, the one with the smaller summed distance wins (1.2 m over 1.4 m)."""
    objects = [obj("A", "chair", (0.0, 0.0, 0.0), (2.0, 2.0, 2.0)), obj("B", "chair", (1.0, 0.0, 0.0), (2.0, 2.0, 2.0))]
    detections = [box((0.1, 0.0, 0.0)), box((-0.3, 0.0, 0.0))]
    result = match_detections(detections, [("chair",), ("chair",)], objects)    # gates 0.5 * sqrt(12) m
    assert {(m.detection, m.object_id) for m in result.matches} == {(0, "B"), (1, "A")}


def test_labels_gate_and_unboxed() -> None:
    """Incompatible labels never match, the gate grows with the box, and box-less detections stay aside."""
    objects = [obj("t", "table", (0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),            # diagonal 0.87: gate 0.5 m
               obj("big", "Transporter", (5.0, 0.0, 0.0), (2.0, 1.0, 1.0))]    # diagonal 2.45: gate 1.22 m
    detections = [box((0.1, 0.0, 0.0)),        # a "chair" on the table: wrong class
                  box((0.0, 0.6, 0.0)),        # a table 0.6 m off: outside the 0.5 m gate
                  box((6.0, 0.0, 0.0)),        # a "mobile robot" 1.0 m off: inside the 1.22 m gate
                  None]                        # lifting failed
    labels = [("chair",), ("table",), ("NovaCarter", "Transporter"), ("table",)]
    result = match_detections(detections, labels, objects)
    assert [(m.detection, m.object_id) for m in result.matches] == [(2, "big")]
    assert result.matches[0].gate == pytest.approx(0.5 * np.sqrt(6.0))
    assert result.missed == ("t",)
    assert result.false_positives == (0, 1) and result.unboxed == (3,)


def test_empty_inputs() -> None:
    """No objects makes every boxed detection a false positive; no detections makes every object a miss."""
    result = match_detections([box((0.0, 0.0, 0.0))], [("chair",)], [])
    assert result.matches == () and result.false_positives == (0,)
    result = match_detections([], [], [obj("A", "chair", (0.0, 0.0, 0.0))])
    assert result.missed == ("A",)
    with pytest.raises(ValueError):
        match_detections([box((0.0, 0.0, 0.0))], [], [])


# --------------------------------------------------------------------------------------------------------
# 2D association
# --------------------------------------------------------------------------------------------------------


def test_box_iou_2d() -> None:
    """Pairwise 2D IoU: identical, half-shifted (1/3), disjoint, contained (1/4) and degenerate boxes."""
    a = [(0.0, 0.0, 10.0, 10.0)]
    b = [(0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0), (20.0, 0.0, 30.0, 10.0), (0.0, 0.0, 5.0, 5.0),
         (3.0, 3.0, 3.0, 8.0)]
    assert box_iou_2d(a, b) == pytest.approx(np.array([[1.0, 1.0 / 3.0, 0.0, 0.25, 0.0]]))
    assert box_iou_2d(np.zeros((0, 4)), b).shape == (0, 5)


def test_2d_pairs_a_class_confusion_and_records_it() -> None:
    """A "pallet jack" detection on a forklift is paired (label_correct False); the 3D association misses it."""
    forklift = obj("F", "Forklift", (10.0, 0.0, 1.0), (2.5, 1.2, 2.0))
    detection = box((10.2, 0.1, 1.0), (2.3, 1.1, 1.9))
    labels = [("PalletTruck",)]
    result = match_detections_2d([(100, 100, 200, 180)], [detection], labels, [forklift], [(105, 100, 200, 185)])
    (m,) = result.matches
    assert (m.detection, m.object_id, m.label_correct, m.gate) == (0, "F", False, None)
    assert m.iou2d == pytest.approx(95 * 80 / (100 * 80 + 95 * 85 - 95 * 80))
    assert m.distance == pytest.approx(np.hypot(0.2, 0.1)) and m.iou > 0.5
    assert result.missed == () and result.false_positives == ()
    assert match_detections([detection], labels, [forklift]).missed == ("F",)       # censored by the label rule


def test_2d_pairs_a_far_3d_error_and_measures_it() -> None:
    """A detection whose 3D box is 12 m too far (monocular depth at range) is paired; its distance is recorded."""
    person = obj("P", "Person", (40.0, 0.0, 0.9), (0.5, 0.5, 1.8))
    detection = box((52.0, 0.0, 0.9), (0.6, 0.5, 2.1))                 # same pixels, wrong range
    result = match_detections_2d([(500, 300, 520, 360)], [detection], [("Person",)], [person],
                                 [(501, 301, 520, 360)])
    (m,) = result.matches
    assert m.label_correct and m.distance == pytest.approx(12.0) and m.iou == 0.0 and m.iou2d > 0.9
    assert match_detections([detection], [("Person",)], [person]).matches == ()   # outside the 0.5 m gate


def test_2d_ambiguous_overlaps() -> None:
    """Summed IoU decides contested pairs, runner-up overlaps flag ambiguity, and weak or box-less pairs stay out.

    Detection 0 overlaps A by 0.43 and B by 0.25; detection 1 overlaps A by 0.8 only. With ``min_iou`` 0.2,
    maximising the summed IoU pairs 0-B and 1-A (1.05) rather than 0-A alone (0.43), and both pairs are
    ambiguous (runner-up 0.43). Detection 2 covers object C exactly and detection 3 slightly less (0.9); the
    second is a false positive. Detection 4 overlaps nothing, and detection 5 has no 3D box.
    """
    objects = [obj("A", "chair", (0.0, 0.0, 0.0)), obj("B", "table", (1.0, 0.0, 0.0)),
               obj("C", "sofa", (5.0, 0.0, 0.0))]
    object_boxes = [(0, 0, 10, 10), (10, 0, 20, 10), (100, 0, 110, 10)]
    boxes2d = [(4, 0, 14, 10),             # IoU 60/140 with A, 40/160 with B
               (0, 0, 8, 10),              # IoU 0.8 with A
               (100, 0, 110, 10),          # IoU 1 with C
               (101, 0, 110, 10),          # IoU 0.9 with C
               (50, 0, 60, 10),            # nothing
               (0, 0, 10, 10)]             # perfect on A, but no 3D box
    boxes = [box((0.5, 0.0, 0.0)), box((0.0, 0.0, 0.0)), box((5.0, 0.0, 0.0)), box((5.1, 0.0, 0.0)),
             box((9.0, 0.0, 0.0)), None]
    labels = [("table",), ("chair",), ("sofa",), ("sofa",), ("chair",), ("chair",)]
    overlap = box_iou_2d(boxes2d[:2], object_boxes[:2])
    assert overlap == pytest.approx(np.array([[0.6 / 1.4, 0.4 / 1.6], [0.8, 0.0]]))
    result = match_detections_2d(boxes2d, boxes, labels, objects, object_boxes, Match2DConfig(min_iou=0.2))
    by_det = result.by_detection()
    assert {(m.detection, m.object_id) for m in result.matches} == {(0, "B"), (1, "A"), (2, "C")}
    assert by_det[0].label_correct and by_det[1].label_correct
    assert by_det[0].ambiguous and by_det[0].runner_up_iou2d == pytest.approx(0.6 / 1.4)
    assert by_det[1].ambiguous and by_det[1].runner_up_iou2d == pytest.approx(0.6 / 1.4)
    assert by_det[2].ambiguous and by_det[2].runner_up_iou2d == pytest.approx(0.9)
    assert result.false_positives == (3, 4) and result.unboxed == (5,) and result.missed == ()
    # A stricter threshold drops the weak 0-B pair instead of forcing it.
    strict = match_detections_2d(boxes2d, boxes, labels, objects, object_boxes, Match2DConfig(min_iou=0.5))
    assert {(m.detection, m.object_id) for m in strict.matches} == {(1, "A"), (2, "C")}
    assert strict.missed == ("B",) and not strict.by_detection()[1].ambiguous
    with pytest.raises(ValueError):
        match_detections_2d(boxes2d, boxes, labels, objects, object_boxes[:2])
    with pytest.raises(ValueError):
        Match2DConfig(min_iou=0.0)
