"""Tests for matching detections to oracle objects: prism IoU and the gated Hungarian assignment.

Boxes are axis-aligned or yawed about world +z with hand-computable overlaps.
"""

from __future__ import annotations

import numpy as np
import pytest

from measured_space.geometry import OrientedBox, axis_angle_to_matrix
from measured_space.matching import MatchConfig, clip_convex, convex_hull_2d, match_detections, polygon_area, prism_iou
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
