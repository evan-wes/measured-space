"""Tests for the question generator: answers, referring expressions, filters and determinism.

Scenes are a few boxes with hand-computable geometry. Unless a test says otherwise, the world is z-up and
the camera sits at the origin looking along world +y, so a world point ``(x, y, z)`` has camera-frame
coordinates ``(x, -z, y)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from measured_space.geometry import Camera, OrientedBox, axis_angle_to_matrix, make_pose, transform
from measured_space.qa import (
    Drop,
    QAConfig,
    answer,
    generate_specs,
    make_items,
    phrase_for,
    plural,
    referring_expressions,
    visible_ids,
)
from measured_space.qa_schema import QuestionSpec, is_correct, read_jsonl, write_jsonl
from measured_space.scene import FrameView, ObjectInView, SceneObject

CAM = Camera(500.0, 500.0, 319.5, 239.5, 640, 480)
LOOK_ALONG_Y = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])   # columns: camera x, y, z in the world
LOOK_ALONG_X = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])
# Every filter off, so each candidate is generated and its answer can be checked.
NO_FILTERS = QAConfig(min_distance=0.0, relative_depth_min_abs=0.0, relative_depth_min_rel=0.0,
                      left_right_min_dx=0.0, left_right_require_image_agreement=False, closest_min_abs=0.0,
                      closest_min_rel=0.0, max_per_type=None)


def make_view(rotation: np.ndarray = LOOK_ALONG_Y, position: tuple[float, float, float] = (0.0, 0.0, 0.0)
              ) -> FrameView:
    """A synthetic z-up view.

    Parameters
    ----------
    rotation : ndarray, shape (3, 3)
        Camera-to-world rotation (camera axes as columns).
    position : tuple of float
        Camera centre in the world (m).

    Returns
    -------
    FrameView
        The view.
    """
    return FrameView("synthetic", "s0", "cam0", "0", CAM, make_pose(rotation, np.array(position)), "img.png",
                     up=np.array([0.0, 0.0, 1.0]))


def box_object(object_id: str, label: str, center: tuple[float, float, float],
               size: tuple[float, float, float], rotation: np.ndarray | None = None) -> SceneObject:
    """An object with an oriented box.

    Parameters
    ----------
    object_id : str
        Its id.
    label : str
        Its dataset label.
    center : tuple of float
        Box centre, world frame (m).
    size : tuple of float
        Full side lengths (m).
    rotation : ndarray, shape (3, 3), optional
        Box axes as columns; identity if omitted.

    Returns
    -------
    SceneObject
        The object.
    """
    rotation = np.eye(3) if rotation is None else rotation
    return SceneObject(object_id, label, OrientedBox(center, size, rotation))


def observe(view: FrameView, obj: SceneObject, visible: float = 1.0, truncated: float = 0.0,
            u: float | None = None) -> ObjectInView:
    """An object as a view sees it, with its 2D box from its projected corners.

    Parameters
    ----------
    view : FrameView
        The view.
    obj : SceneObject
        The object; must be in front of the camera.
    visible, truncated : float
        Its visible and truncated fractions.
    u : float, optional
        If given, the 2D box is replaced by a 20 px wide one centred at this column.

    Returns
    -------
    ObjectInView
        The observation.
    """
    uv, _ = view.camera.project(transform(view.T_cw, obj.box.corners()))
    x0, y0, x1, y1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
    if u is not None:
        x0, x1 = u - 10.0, u + 10.0
    return ObjectInView(obj, visible, truncated, (float(x0), float(y0), float(x1), float(y1)))


def find(specs: list[QuestionSpec], qtype: str, *ids: str) -> QuestionSpec | None:
    """The spec of a type about exactly these referents, in order.

    Parameters
    ----------
    specs : list of QuestionSpec
        Generated specs.
    qtype : str
        Question type.
    *ids : str
        Referent ids.

    Returns
    -------
    QuestionSpec or None
        The match, or None if there is none.
    """
    matches = [s for s in specs if s.qtype == qtype and tuple(r.object_id for r in s.referents) == ids]
    assert len(matches) <= 1
    return matches[0] if matches else None


def room() -> tuple[FrameView, list[ObjectInView], dict[str, SceneObject]]:
    """Two chairs and a table in front of the camera.

    Chair ``c1`` spans x in [-1.75, -1.25], chair ``c2`` x in [-0.25, 0.25] and y in [5.25, 5.75], table
    ``t`` x in [0.4, 1.6] and y in [3.6, 4.4]. All rest on z = 0 at their centres.

    Returns
    -------
    view : FrameView
        Camera at the origin looking along +y.
    seen : list of ObjectInView
        All three, fully visible.
    objects : dict of str to SceneObject
        The objects by id.
    """
    view = make_view()
    objs = [box_object("c1", "chair", (-1.5, 4.0, 0.0), (0.5, 0.5, 1.0)),
            box_object("c2", "chair", (0.0, 5.5, 0.0), (0.5, 0.5, 1.0)),
            box_object("t", "table", (1.0, 4.0, 0.0), (1.2, 0.8, 0.75))]
    return view, [observe(view, o) for o in objs], {o.object_id: o for o in objs}


def test_numeric_answers_match_hand_computed_values() -> None:
    """Centre and surface distance, height, longest dimension and camera distance, with their tolerances."""
    view, seen, objects = room()
    specs = generate_specs(view, seen, NO_FILTERS)
    expected = {
        ("distance_center", "c1", "t"): 2.5,
        ("distance_surface", "c1", "t"): 1.65,                     # x gap from -1.25 to 0.4
        ("distance_surface", "c2", "t"): np.hypot(0.15, 0.85),     # corner gap: x 0.25..0.4, y 4.4..5.25
        ("size_height", "t"): 0.75,
        ("size_longest", "t"): 1.2,
        ("ego_distance", "t"): np.hypot(0.4, 3.6),                 # nearest point (0.4, 3.6, 0)
    }
    for (qtype, *ids), value in expected.items():
        spec = find(specs, qtype, *ids)
        got, record, _ = answer(spec, objects, view)
        assert got == pytest.approx(value, abs=1e-6), qtype
        assert spec.unit == "m" and [r["object_id"] for r in record] == ids
    assert find(specs, "ego_distance", "t").question.startswith("What is the distance from the camera to "
                                                               "the nearest point of the table?")
    assert find(specs, "distance_center", "c1", "t").tolerance == {"abs": 0.10, "rel": 0.10}
    assert find(specs, "size_height", "t").tolerance == {"abs": 0.05, "rel": 0.10}


def test_sizes_of_tilted_boxes_do_not_assume_yaw_only() -> None:
    """Height is the extent along up, and the longest dimension uses the box's most horizontal axes."""
    view = make_view()
    pitch = np.radians(20.0)
    objs = [box_object("yawed", "cabinet", (-2.0, 6.0, 0.0), (1.5, 0.5, 0.9),
                       axis_angle_to_matrix([0, 0, 0.7])),
            box_object("on_side", "shelf", (0.0, 6.0, 0.0), (0.5, 0.3, 2.0),
                       axis_angle_to_matrix([np.pi / 2, 0, 0])),           # local z lies along world -y
            box_object("plank", "table", (2.0, 6.0, 0.0), (2.0, 0.5, 0.5),
                       axis_angle_to_matrix([0, pitch, 0]))]
    specs = generate_specs(view, [observe(view, o) for o in objs], NO_FILTERS)
    by_id = {o.object_id: o for o in objs}
    expected = {"yawed": (0.9, 1.5), "on_side": (0.3, 2.0),
                "plank": (2 * np.sin(pitch) + 0.5 * np.cos(pitch), 2 * np.cos(pitch) + 0.5 * np.sin(pitch))}
    for object_id, (height, longest) in expected.items():
        assert answer(find(specs, "size_height", object_id), by_id, view)[0] == pytest.approx(height)
        assert answer(find(specs, "size_longest", object_id), by_id, view)[0] == pytest.approx(longest)


def test_categorical_answers_and_margins() -> None:
    """Relative depth, left/right, closest instance and count, with their margins."""
    view, seen, objects = room()
    specs = generate_specs(view, seen, NO_FILTERS)

    depth = find(specs, "relative_depth", "c1", "t")
    assert depth.choices == ("the leftmost chair", "the table")
    value, _, extra = answer(depth, objects, view)
    assert value == "the table"
    assert extra["margin"] == pytest.approx(np.hypot(1.25, 3.75) - np.hypot(0.4, 3.6))

    value, _, extra = answer(find(specs, "left_right", "c1", "t"), objects, view)
    assert value == "left" and extra["margin"] == pytest.approx(2.5)

    closest = find(specs, "closest_instance", "t", "c1", "c2")
    assert closest.choices == ("the leftmost chair", "the rightmost chair")
    value, _, extra = answer(closest, objects, view)
    assert value == "the rightmost chair"
    assert extra["margin"] == pytest.approx(1.65 - np.hypot(0.15, 0.85))

    counts = {s.params["phrase"]: s for s in specs if s.qtype == "count"}
    assert counts["chair"].question == ("How many chairs are visible in this image? "
                                        "Answer with a whole number.")
    ids = visible_ids(seen)
    assert answer(counts["chair"], objects, view, in_view=ids)[0] == 2
    assert answer(counts["table"], objects, view, in_view=ids)[0] == 1


def test_left_right_uses_camera_frame_x_for_a_rotated_camera() -> None:
    """With the camera looking along world +x, left/right follows camera x, not world x."""
    view = make_view(LOOK_ALONG_X, position=(0.0, 0.0, 1.2))
    # Camera x is world -y: the chair (y = +1) is on the left although its world x is the larger.
    chair = box_object("a", "chair", (5.0, 1.0, 0.0), (0.4, 0.4, 0.8))
    table = box_object("b", "table", (3.0, -1.0, 0.0), (0.8, 0.8, 0.7))
    specs = generate_specs(view, [observe(view, chair), observe(view, table)], QAConfig())
    value, _, extra = answer(find(specs, "left_right", "a", "b"), {"a": chair, "b": table}, view)
    assert value == "left" and extra["margin"] == pytest.approx(2.0)


def test_ordinal_expressions_and_phrase_uniqueness() -> None:
    """Shared phrases are named left to right, and instances too close together are not named."""
    view = make_view()
    chairs = [box_object(f"c{i}", "chair", (x, 4.0, 0.0), (0.4, 0.4, 0.8)) for i, x in enumerate((-1, 0, 1))]
    seen = [observe(view, c, u=u) for c, u in zip(chairs, (500.0, 100.0, 300.0))]    # out of order on purpose
    assert referring_expressions(view, seen) == {
        "c1": "the leftmost chair", "c2": "the second chair from the left", "c0": "the rightmost chair"}
    # A fourth chair 20 px from c2 (minimum gap 5% of 640 = 32 px): neither of the two can be named.
    crowded = seen + [observe(view, box_object("c3", "chair", (0.2, 4.0, 0.0), (0.4, 0.4, 0.8)), u=320.0)]
    drops: list[Drop] = []
    assert referring_expressions(view, crowded, drops=drops) == {"c1": "the leftmost chair",
                                                                 "c0": "the rightmost chair"}
    assert {d.object_ids[0] for d in drops if d.reason == "ordinal_too_close"} == {"c2", "c3"}

    # Two labels with one phrase are as ambiguous as two instances of one label.
    robots = [box_object("r1", "NovaCarter", (-1.0, 4.0, 0.0), (0.6, 0.6, 0.4)),
              box_object("r2", "Transporter", (1.0, 4.0, 0.0), (0.6, 0.6, 0.4)),
              box_object("p", "Person", (0.0, 6.0, 0.0), (0.5, 0.4, 1.7))]
    assert referring_expressions(view, [observe(view, o) for o in robots]) == {
        "r1": "the leftmost mobile robot", "r2": "the rightmost mobile robot", "p": "the person"}
    # A partly visible second chair also makes "the chair" ambiguous, though it cannot be a referent itself.
    pair = [observe(view, chairs[0]), observe(view, chairs[2], visible=0.1)]
    assert referring_expressions(view, pair)["c0"] == "the leftmost chair"
    assert phrase_for("tv_monitor") == phrase_for("TV") == "TV" and phrase_for("CoffeeMug") == "coffee mug"
    assert plural("TV") == "TVs" and plural("person") == "people"
    assert plural("pallet truck") == "pallet trucks"


@pytest.mark.parametrize("gap, kept", [(-0.2, False), (0.0, False), (0.15, False), (0.25, True)])
def test_surface_distance_filter(gap: float, kept: bool) -> None:
    """Overlapping, touching and near-touching boxes are not asked about; a clear gap is.

    Parameters
    ----------
    gap : float
        Gap between the boxes along x (m); negative means overlap.
    kept : bool
        Whether the surface-distance question should survive the 0.2 m minimum.
    """
    view = make_view()
    a = box_object("a", "chair", (-1.0, 4.0, 0.0), (1.0, 1.0, 1.0))
    b = box_object("b", "table", (gap, 4.0, 0.0), (1.0, 1.0, 1.0))
    drops: list[Drop] = []
    specs = generate_specs(view, [observe(view, a), observe(view, b)], QAConfig(max_per_type=None),
                           diagnostics=drops)
    assert (find(specs, "distance_surface", "a", "b") is not None) == kept
    assert any(d.qtype == "distance_surface" and d.reason == "distance_below_min" for d in drops) != kept


@pytest.mark.parametrize("nearer, delta, kept", [(2.0, 0.25, False), (2.0, 0.35, True),    # 0.3 m bound
                                                 (5.0, 0.45, False), (5.0, 0.55, True)])   # 10% bound
def test_relative_depth_margin_filter(nearer: float, delta: float, kept: bool) -> None:
    """Relative depth needs a difference of at least max(0.3 m, 10% of the nearer distance).

    Parameters
    ----------
    nearer : float
        Camera distance of the nearer object (m).
    delta : float
        How much farther the other object is (m).
    kept : bool
        Whether the question should survive.
    """
    view = make_view()

    def depth_for(distance: float) -> float:
        """Centre y that puts a 0.4 m cube at x = +-1 at this camera distance.

        Parameters
        ----------
        distance : float
            Wanted distance to the cube's nearest point (m).

        Returns
        -------
        float
            The centre's y (m): the nearest point is (0.8, y - 0.2, 0).
        """
        return 0.2 + np.sqrt(distance ** 2 - 0.8 ** 2)

    a = box_object("a", "chair", (-1.0, depth_for(nearer), 0.0), (0.4, 0.4, 0.4))
    b = box_object("b", "table", (1.0, depth_for(nearer + delta), 0.0), (0.4, 0.4, 0.4))
    specs = generate_specs(view, [observe(view, a), observe(view, b)], QAConfig(max_per_type=None))
    spec = find(specs, "relative_depth", "a", "b")
    assert (spec is not None) == kept
    if kept:
        value, _, extra = answer(spec, {"a": a, "b": b}, view)
        assert value == "the chair" and extra["margin"] == pytest.approx(delta)


def test_left_right_filters_small_offsets_and_image_disagreement() -> None:
    """Left/right is dropped below 0.3 m of camera x, and when image order and 3D order disagree."""
    view = make_view()
    for dx, kept in ((0.25, False), (0.35, True)):
        a = box_object("a", "chair", (-dx / 2, 4.0, 0.0), (0.1, 0.1, 0.1))
        b = box_object("b", "table", (dx / 2, 4.0, 0.0), (0.1, 0.1, 0.1))
        specs = generate_specs(view, [observe(view, a), observe(view, b)], QAConfig())
        assert (find(specs, "left_right", "a", "b") is not None) == kept
    # a is 0.5 m further left in 3D, but far away: its image column (269.5) is right of b's (152.8).
    a = box_object("a", "chair", (-1.0, 10.0, 0.0), (0.1, 0.1, 0.1))
    b = box_object("b", "table", (-0.5, 1.5, 0.0), (0.1, 0.1, 0.1))
    seen = [observe(view, a), observe(view, b)]
    drops: list[Drop] = []
    assert find(generate_specs(view, seen, QAConfig(), diagnostics=drops), "left_right", "a", "b") is None
    assert any(d.reason == "image_order_disagrees" for d in drops)
    lenient = QAConfig(left_right_require_image_agreement=False)
    assert answer(find(generate_specs(view, seen, lenient), "left_right", "a", "b"),
                  {"a": a, "b": b}, view)[0] == "left"


def test_count_drops_borderline_instances_and_counts_from_in_view() -> None:
    """Counts need every instance clearly in or out; the answer counts the source's objects in view."""
    view, seen, objects = room()
    extra_chair = box_object("c3", "chair", (2.5, 6.0, 0.0), (0.5, 0.5, 1.0))
    objects = {**objects, "c3": extra_chair}

    def chair_count(visible: float, truncated: float) -> tuple[QuestionSpec | None, list[Drop]]:
        """Generate with a third chair at the given visibility.

        Parameters
        ----------
        visible, truncated : float
            The third chair's visible and truncated fractions.

        Returns
        -------
        spec : QuestionSpec or None
            The chair-count spec, if generated.
        drops : list of Drop
            The diagnostics.
        """
        drops: list[Drop] = []
        specs = generate_specs(view, seen + [observe(view, extra_chair, visible, truncated)], QAConfig(),
                               diagnostics=drops)
        counts = [s for s in specs if s.qtype == "count" and s.params["phrase"] == "chair"]
        return (counts[0] if counts else None), drops

    for visible, truncated in ((0.1, 0.0), (0.35, 0.65)):            # barely visible; mostly out of frame
        spec, drops = chair_count(visible, truncated)
        assert spec is None
        assert [d.object_ids for d in drops if d.reason == "borderline_instance"] == [("c3",)]
    spec, _ = chair_count(0.0, 1.0)                                  # not in the image at all
    oracle = make_items([spec], objects, view, "oracle", in_view=visible_ids(seen))[0]
    assert oracle.answer == 2 and [o["object_id"] for o in oracle.objects] == ["c1", "c2"]
    auto = make_items([spec], objects, view, "auto", in_view={"c1", "c3", "t"})[0]
    assert auto.answer == 2 and is_correct(spec, auto.answer, oracle.answer)
    assert not is_correct(spec, make_items([spec], objects, view, "auto", in_view={"c1"})[0].answer, 2)
    with pytest.raises(ValueError):
        answer(spec, objects, view)
    assert chair_count(1.0, 0.0)[0] is not None
    drops = []
    capped = generate_specs(view, seen, QAConfig(count_max=1), diagnostics=drops)
    assert [s.params["phrase"] for s in capped if s.qtype == "count"] == ["table"]
    assert any(d.reason == "count_above_max" and d.detail["phrase"] == "chair" for d in drops)


def test_perturbed_boxes_change_answers_and_missing_referents_are_skipped(tmp_path) -> None:
    """The same specs answered from moved boxes give the moved answers; missing boxes give no item.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory, for a JSONL round trip.
    """
    view, seen, objects = room()
    specs = [s for s in generate_specs(view, seen, NO_FILTERS) if s.qtype != "count"]
    oracle = {i.id: i for i in make_items(specs, objects, view, "oracle")}

    nudged = {**objects, "t": box_object("t", "table", (1.05, 4.0, 0.0), (1.2, 0.8, 0.75))}
    for item in make_items(specs, nudged, view, "auto"):
        assert is_correct(item.spec, item.answer, oracle[item.id].answer), item.spec.qtype

    moved = {**objects, "c1": box_object("c1", "chair", (3.0, 4.0, 0.0), (0.5, 0.5, 1.0))}
    auto = {i.id: i for i in make_items(specs, moved, view, "auto")}
    lr, dist = find(specs, "left_right", "c1", "t"), find(specs, "distance_center", "c1", "t")
    assert auto[lr.id].answer == "right" and not is_correct(lr, auto[lr.id].answer, oracle[lr.id].answer)
    assert auto[dist.id].answer == pytest.approx(2.0)
    assert not is_correct(dist, auto[dist.id].answer, oracle[dist.id].answer)
    assert auto[dist.id].objects[0]["center"] == [3.0, 4.0, 0.0]

    without_table = make_items(specs, {k: v for k, v in objects.items() if k != "t"}, view, "auto")
    expected_ids = {s.id for s in specs if "t" not in [r.object_id for r in s.referents]}
    assert {i.id for i in without_table} == expected_ids
    assert write_jsonl(oracle.values(), tmp_path / "qa.jsonl") == len(oracle)
    assert [i.answer for i in read_jsonl(tmp_path / "qa.jsonl")] == [i.answer for i in oracle.values()]


def test_generation_is_deterministic_order_independent_and_capped() -> None:
    """The same seed gives the same specs whatever the input order; the cap holds; seeds vary the choice."""
    view = make_view()
    labels = ("chair", "table", "sofa", "bed", "stool", "sink")
    positions = ((-2.0, 5.0), (-1.0, 3.5), (0.0, 6.0), (1.2, 4.0), (2.4, 7.0), (3.0, 9.0))
    seen = [observe(view, box_object(f"o{i}", label, (x, y, 0.0), (0.5, 0.5, 0.8)))
            for i, (label, (x, y)) in enumerate(zip(labels, positions))]
    config = QAConfig(max_per_type=2)

    def ids(specs: list[QuestionSpec]) -> list[str]:
        """Spec ids in order.

        Parameters
        ----------
        specs : list of QuestionSpec
            Specs.

        Returns
        -------
        list of str
            Their ids.
        """
        return [s.id for s in specs]

    first = generate_specs(view, seen, config, rng_seed=3)
    assert ids(first) == ids(generate_specs(view, seen, config, rng_seed=3))
    assert ids(first) == ids(generate_specs(view, seen[::-1], config, rng_seed=3))
    assert len(set(ids(first))) == len(first)
    assert all(sum(s.qtype == q for s in first) <= 2 for q in {s.qtype for s in first})
    selections = {tuple(s.id for s in generate_specs(view, seen, config, rng_seed=k)
                        if s.qtype == "distance_center") for k in range(10)}
    assert len(selections) > 1
