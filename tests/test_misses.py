"""Tests for the miss audit: recall and miss causes, effects on answers, flags without ground truth, and the script.

The scene is a table and three chairs in a row in front of a wall, seen by cameras looking along world +y (world
z up). Depth and instance masks are ray-cast exactly, so :func:`measured_space.pipeline.process_view` runs without
any model, and each test controls which objects the labeller detects (and for which depth source a lift fails).
The chair nearest the table (``chair-c``) is the answer to "which chair is closest to the table", so missing it
makes that question unanswered for the pipeline and silently wrong under open-world answering. The arithmetic of
the tables is checked on hand-made rows.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from measured_space import misses as M
from measured_space.box_sources import CutrBox, ViewPredictions
from measured_space.geometry import Array, Camera, OrientedBox, make_pose, transform
from measured_space.matching import box_iou_2d
from measured_space.perception import Instance
from measured_space.pipeline import ViewResult, process_view, to_json_line, visible_box_edges
from measured_space.scene import FrameView, SceneObject

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_miss_audit.py"
CAM = Camera(500.0, 500.0, 319.5, 239.5, 640, 480)
LOOK_ALONG_Y = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])     # columns: camera x, y, z in the world
WALL_Y = 8.0
TABLE = SceneObject("table-1", "table", OrientedBox([1.6, 4.0, 0.0], [1.2, 0.8, 0.8], np.eye(3)))
CHAIR_A = SceneObject("chair-a", "chair", OrientedBox([-1.8, 4.0, 0.0], [0.5, 0.5, 0.9], np.eye(3)))
CHAIR_B = SceneObject("chair-b", "chair", OrientedBox([-0.6, 4.0, 0.0], [0.5, 0.5, 0.9], np.eye(3)))
CHAIR_C = SceneObject("chair-c", "chair", OrientedBox([0.5, 4.0, 0.0], [0.5, 0.5, 0.9], np.eye(3)))
OBJECTS = [TABLE, CHAIR_A, CHAIR_B, CHAIR_C]
WALL_PATCH = (slice(20, 60), slice(300, 340))                     # a patch of bare wall (a false "chair")


def make_view(x: float = 0.0, *, camera_id: str = "wide", frame: str = "0", scene: str = "synthetic") -> FrameView:
    """A view looking along +y from ``(x, 0, 0)``.

    Parameters
    ----------
    x : float, default 0.0
        Camera position along world x, metres.
    camera_id : str, default "wide"
        Camera identifier.
    frame : str, default "0"
        Frame identifier.
    scene : str, default "synthetic"
        Scene identifier.

    Returns
    -------
    FrameView
        An ARKitScenes-like view (so the phrase mapping applies).
    """
    return FrameView("arkitscenes", scene, camera_id, frame, CAM, make_pose(LOOK_ALONG_Y, [x, 0.0, 0.0]), "img.png",
                     up=np.array([0.0, 0.0, 1.0]))


def render(view: FrameView, objects: list[SceneObject]) -> tuple[Array, NDArray[np.int64]]:
    """Ray-cast planar depth and an object-index map for axis-aligned boxes in front of a wall at ``y = WALL_Y``.

    Parameters
    ----------
    view : FrameView
        The view (looking towards +y).
    objects : list of SceneObject
        Axis-aligned boxes.

    Returns
    -------
    depth : numpy.ndarray, shape (H, W)
        Planar depth, metres.
    index : numpy.ndarray, shape (H, W), int
        Index of the nearest box per pixel, -1 for the wall.
    """
    cam = view.camera
    v, u = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
    rays = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], axis=-1) @ view.T_wc[:3, :3].T
    origin = view.camera_center
    depth = (WALL_Y - origin[1]) / rays[..., 1]            # camera z = 1 per ray: the ray parameter is planar depth
    index = np.full(depth.shape, -1)
    for k, o in enumerate(objects):
        lo, hi = o.box.center - o.box.size / 2, o.box.center + o.box.size / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - origin) / rays, (hi - origin) / rays
        near = np.nanmax(np.minimum(t1, t2), axis=-1)
        far = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (near <= far) & (near > 0) & (near < depth)
        depth[hit], index[hit] = near[hit], k
    return depth, index


def instance(phrase: str, mask: NDArray[np.bool_], score: float) -> Instance:
    """A detection with its mask's bounding box.

    Parameters
    ----------
    phrase : str
        Prompt phrase.
    mask : numpy.ndarray, shape (H, W), bool
        Mask.
    score : float
        Detection score.

    Returns
    -------
    Instance
        The detection.
    """
    rows, cols = np.nonzero(mask)
    return Instance(phrase, mask, (float(cols.min()), float(rows.min()), float(cols.max() + 1),
                                   float(rows.max() + 1)), score)


def labelled_view(detected: list[str], *, view: FrameView | None = None, wall_chair: bool = False,
                  lidar_fails: tuple[str, ...] = ()) -> ViewResult:
    """Run the pipeline on the synthetic scene with chosen detections.

    Parameters
    ----------
    detected : list of str
        Ids of the objects the labeller detects (each with its exact mask).
    view : FrameView, optional
        The view; defaults to :func:`make_view`.
    wall_chair : bool, default False
        Add a spurious "chair" detection on a patch of bare wall.
    lidar_fails : tuple of str, default ()
        Objects whose pixels have no valid ``lidar`` depth, so their detection's ``lidar`` lift fails.

    Returns
    -------
    ViewResult
        Sources ``laser`` (the exact depth, also the visibility reference) and ``lidar``.
    """
    view = view or make_view()
    depth, index = render(view, OBJECTS)
    instances = [instance(o.label, index == k, 0.9 - 0.1 * k) for k, o in enumerate(OBJECTS) if o.object_id in detected]
    if wall_chair:
        patch = np.zeros(index.shape, dtype=bool)
        patch[WALL_PATCH] = True
        assert not (index[patch] >= 0).any()
        instances.append(instance("chair", patch, 0.55))
    lidar = depth.copy()
    for k, o in enumerate(OBJECTS):
        if o.object_id in lidar_fails:
            lidar[index == k] = 0.0
    return process_view(view, OBJECTS, instances, {"laser": depth, "lidar": lidar}, visibility="laser")


def visible_edges(result: ViewResult, object_id: str) -> tuple[float, float, float, float]:
    """An oracle object's visible 2D box with pixel edges at integer coordinates.

    Parameters
    ----------
    result : ViewResult
        The view.
    object_id : str
        The object.

    Returns
    -------
    tuple of float
        ``(xmin, ymin, xmax, ymax)``.
    """
    seen = next(o for o in result.oracle if o.obj.object_id == object_id)
    return visible_box_edges(seen.bbox2d, result.visibility)                                # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------------------
# Recall and miss causes
# --------------------------------------------------------------------------------------------------------


def test_object_rows_record_pairs_and_why_objects_were_missed() -> None:
    """chair-b has no detection; chair-c's lidar lift fails, so it is paired under laser and unboxed under lidar."""
    result = labelled_view(["table-1", "chair-a", "chair-c"], lidar_fails=("chair-c",))
    rows = {(r["source"], r["object_id"]): r for s in ("laser", "lidar") for r in M.object_rows(result, s, "v0")}
    assert len(rows) == 8 and all(r["clearly_visible"] and r["range_bin"] == "0_10m" for r in rows.values())
    assert {k for k, r in rows.items() if r["paired"]} == {("laser", "table-1"), ("laser", "chair-a"),
                                                           ("laser", "chair-c"), ("lidar", "table-1"),
                                                           ("lidar", "chair-a")}
    assert rows[("laser", "chair-b")]["cause"] == rows[("lidar", "chair-b")]["cause"] == "no_detection"
    assert rows[("lidar", "chair-c")]["cause"] == "unboxed" and rows[("lidar", "chair-c")]["best_iou2d_unboxed"] > 0.9
    assert rows[("laser", "chair-c")]["iou2d"] > 0.9 and rows[("laser", "chair-c")]["label_correct"] is True
    assert M.object_rows(result, "moge", "v0") == []                                # a source the view lacks

    objects = pd.DataFrame(list(rows.values()))
    recall = M.recall_table(objects, n_boot=200)
    overall = recall[(recall["by"] == "all") & (recall["population"] == "present")].set_index("source")
    assert overall.loc["laser", "n_paired"] == 3 and overall.loc["lidar", "n_paired"] == 2
    assert overall.loc["laser", "recall"] == pytest.approx(0.75) and overall.loc["lidar", "recall"] == 0.5
    # One scene: every bootstrap replicate is the scene itself, so the interval collapses onto the estimate.
    assert overall.loc["laser", "boot_lo"] == overall.loc["laser", "boot_hi"] == pytest.approx(0.75)
    by_class = recall[(recall["by"] == "class") & (recall["source"] == "lidar") & (recall["population"] == "present")]
    assert dict(zip(by_class["group"], by_class["n_paired"])) == {"chair": 1, "table": 1}
    causes = M.miss_cause_table(objects).set_index(["source", "population"])
    lidar = causes.loc[("lidar", "present")]
    assert lidar["n_no_detection"] == 1 and lidar["n_unboxed"] == 1
    assert causes.loc[("laser", "present"), "share_no_detection"] == 1.0


def test_a_detection_spanning_two_objects_misses_one_by_assignment() -> None:
    """One detection covers two touching chairs; the assignment gives it to one, the other is an assignment miss."""
    left = SceneObject("chair-l", "chair", OrientedBox([-0.28, 4.0, 0.0], [0.5, 0.5, 0.9], np.eye(3)))
    right = SceneObject("chair-r", "chair", OrientedBox([0.28, 4.0, 0.0], [0.5, 0.5, 0.9], np.eye(3)))
    view = make_view()
    depth, index = render(view, [left, right])
    result = process_view(view, [left, right], [instance("chair", index >= 0, 0.9)], {"laser": depth},
                          visibility="laser")
    edges = [visible_edges(result, i) for i in ("chair-l", "chair-r")]
    assert (box_iou_2d([result.detections[0].instance.box2d], edges) >= 0.3).all()      # both pairings feasible
    rows = M.object_rows(result, "laser", "v0")
    assert sorted(r["cause"] for r in rows) == ["assignment", "paired"]


def test_recall_table_wilson_and_bootstrap_arithmetic() -> None:
    """7 of 10 paired gives the textbook Wilson interval; two opposite scenes give a wide bootstrap interval."""
    rows = [{"labeller": "x", "source": "s", "dataset": "d", "scene": "a" if i < 5 else "b", "view_key": f"v{i}",
             "paired": i < 5 or i in (5, 6), "clearly_visible": i % 2 == 0, "range_bin": "0_10m",
             "visible_bin": "0.6-1", "truncation_bin": "0", "label": "chair"} for i in range(10)]
    recall = M.recall_table(pd.DataFrame(rows), n_boot=2000, seed=1)
    row = recall[(recall["by"] == "all") & (recall["population"] == "present")].iloc[0]
    assert (row["n"], row["n_paired"], row["n_missed"]) == (10, 7, 3) and row["recall"] == pytest.approx(0.7)
    assert row["wilson_lo"] == pytest.approx(0.3968, abs=1e-4) and row["wilson_hi"] == pytest.approx(0.8922, abs=1e-4)
    assert row["n_scenes"] == 2 and row["boot_lo"] <= 0.4 + 1e-9 and row["boot_hi"] == pytest.approx(1.0)
    scenes = recall[(recall["by"] == "scene") & (recall["population"] == "present")].set_index("group")
    assert scenes.loc["a", "recall"] == 1.0 and scenes.loc["b", "recall"] == pytest.approx(0.4)
    assert np.isnan(scenes.loc["a", "boot_lo"])                                        # bootstrap: overall rows only
    clear = recall[(recall["by"] == "all") & (recall["population"] == "clearly_visible")].iloc[0]
    assert (clear["n"], clear["n_paired"]) == (5, 4)                                   # i = 0, 2, 4, 6 paired; 8 not


def test_bins() -> None:
    """Range bins match comparison.csv's names; fraction and truncation bins are half-open as documented."""
    assert [M.range_bin(d) for d in (0.0, 9.99, 10.0, 44.9, 45.0, 300.0)] == [
        "0_10m", "0_10m", "10_20m", "20_45m", "45m_plus", "45m_plus"]
    assert [M.fraction_bin(f) for f in (0.0, 0.1, 0.29, 0.6, 1.0)] == ["0-0.1", "0.1-0.3", "0.1-0.3", "0.6-1", "0.6-1"]
    assert [M.truncation_bin(f) for f in (0.0, 0.01, 0.3, 0.31, 1.0)] == ["0", "0-0.3", "0-0.3", "0.3-0.6", "0.6-1"]


# --------------------------------------------------------------------------------------------------------
# Effects on answers
# --------------------------------------------------------------------------------------------------------


def _questions(result: ViewResult, source: str = "laser") -> dict[str, dict[str, object]]:
    """Question rows keyed by type (and phrase for counts).

    Parameters
    ----------
    result : ViewResult
        The view.
    source : str, default "laser"
        The source.

    Returns
    -------
    dict of str to dict
        ``"count:<phrase>"``, ``"closest_instance"`` and other types (first row of each).
    """
    out: dict[str, dict[str, object]] = {}
    for row in M.question_rows(result, source, "v0"):
        spec = next(s for s in result.specs if s.id == row["question_id"])
        key = f"count:{spec.params['phrase']}" if row["qtype"] == "count" else str(row["qtype"])
        out.setdefault(key, row)
    return out


def test_a_missed_answer_object_is_unanswered_strictly_and_silently_wrong_open_world() -> None:
    """Missing chair-c: the chair count is low by one (misses only) and the closest chair is answered wrongly."""
    full = labelled_view(["table-1", "chair-a", "chair-b", "chair-c"])
    assert {s.qtype for s in full.specs} >= {"count", "closest_instance"}
    rows = M.question_rows(full, "laser", "v0")
    assert all(r["answered"] and not r["silent_miss"] for r in rows)

    result = labelled_view(["table-1", "chair-a", "chair-b"])
    q = _questions(result)
    count = q["count:chair"]
    assert (count["oracle_count"], count["auto_count"], count["count_missed"]) == (3, 2, 1)
    assert count["count_outcome"] == "under" and count["count_miss_role"] == "under_miss_only"
    assert count["correct"] is False and count["silent_miss"] is True and count["count_error_without_misses"] == 0
    closest = q["closest_instance"]
    assert closest["answered"] is False and closest["missing_ids"] == "chair-c"
    assert closest["closest_answer_id"] == "chair-c" and closest["closest_answer_missed"] is True
    assert closest["open_status"] == "open" and closest["open_answer_id"] == "chair-b"
    assert closest["open_correct"] is False and closest["silent_miss"] is True
    unanswered = [r for r in M.question_rows(result, "laser", "v0") if not r["answered"]]
    assert unanswered and all("chair-c" in str(r["missing_ids"]).split(";") for r in unanswered)

    effects = M.effects_table(pd.DataFrame(M.question_rows(result, "laser", "v0"))).set_index("qtype")
    assert effects.loc["count", "n_under_miss_only"] == 1 and effects.loc["count", "n_silent_miss"] == 1
    closest_row = effects.loc["closest_instance"]
    assert closest_row["n_answered"] == 0 and closest_row["n_open_answered"] == 1
    assert effects.loc["closest_instance", "silent_miss_share_of_wrong"] == 1.0
    assert effects.loc["all", "n_unanswered_miss"] == len(unanswered)


def test_open_world_answer_is_right_when_the_missed_chair_is_not_the_answer() -> None:
    """Missing chair-a: the open-world closest chair is still chair-c, so no silent error."""
    closest = _questions(labelled_view(["table-1", "chair-b", "chair-c"]))["closest_instance"]
    assert closest["open_status"] == "open" and closest["open_correct"] is True and closest["silent_miss"] is False
    lonely = _questions(labelled_view(["table-1", "chair-c"]))["closest_instance"]
    assert lonely["open_status"] == "unanswerable" and lonely["open_correct"] is None     # one chair found: no "which"


def test_an_extra_detection_masks_a_miss_in_the_count() -> None:
    """A wall "chair" offsets the missed chair-c: the count is exact although an instance was missed."""
    count = _questions(labelled_view(["table-1", "chair-a", "chair-b"], wall_chair=True))["count:chair"]
    assert (count["oracle_count"], count["auto_count"]) == (3, 3) and count["count_extra_in"] == 1
    assert count["count_miss_role"] == "masked" and count["correct"] is True and count["silent_miss"] is False


def test_effects_table_arithmetic() -> None:
    """Coverage, wrong rate and the count and closest-instance breakdowns from hand-made rows."""
    base = {"labeller": "x", "source": "s", "dataset": "d"}
    rows = [
        {**base, "qtype": "count", "answered": True, "correct": False, "silent_miss": True, "count_outcome": "under",
         "count_miss_role": "under_miss_only", "count_missed": 2},
        {**base, "qtype": "count", "answered": True, "correct": False, "silent_miss": False, "count_outcome": "over",
         "count_miss_role": "none", "count_missed": 0},
        {**base, "qtype": "count", "answered": True, "correct": True, "silent_miss": False, "count_outcome": "exact",
         "count_miss_role": "masked", "count_missed": 1},
        {**base, "qtype": "closest_instance", "answered": False, "correct": None, "silent_miss": True,
         "open_status": "open", "open_correct": False, "closest_answer_missed": True},
        {**base, "qtype": "closest_instance", "answered": True, "correct": False, "silent_miss": False,
         "open_status": "strict", "open_correct": False, "closest_answer_missed": False},
        {**base, "qtype": "closest_instance", "answered": False, "correct": None, "silent_miss": False,
         "open_status": "unanswerable", "open_correct": None, "closest_answer_missed": False},
        {**base, "qtype": "size_height", "answered": True, "correct": True, "silent_miss": False},
    ]
    t = M.effects_table(pd.DataFrame(rows)).set_index("qtype")
    assert t.loc["count", ["n_exact", "n_under", "n_over", "n_with_missed_instance", "n_masked_miss"]].tolist() == \
        [1, 1, 1, 2, 1]
    assert t.loc["count", "n_wrong"] == 2 and t.loc["count", "silent_miss_share_of_wrong"] == 0.5
    assert t.loc["count", "missed_instances"] == 3
    assert t.loc["closest_instance", ["n_questions", "n_answered", "n_open_answered", "n_open_only",
                                      "n_open_wrong", "n_silent_miss"]].tolist() == [3, 1, 2, 1, 2, 1]
    assert t.loc["closest_instance", "silent_miss_share_of_wrong"] == 0.5
    assert t.loc["all", "n_questions"] == 7 and t.loc["all", "n_answered"] == 5
    assert t.loc["all", "coverage"] == pytest.approx(5 / 7) and t.loc["all", "wrong_rate"] == pytest.approx(3 / 5)
    assert t.loc["all", "n_silent_miss"] == 2 and pd.isna(t.loc["all", "silent_miss_share_of_wrong"])


# --------------------------------------------------------------------------------------------------------
# Flags: evaluation arithmetic
# --------------------------------------------------------------------------------------------------------


def test_evaluate_flags_categories() -> None:
    """A flag on a miss hits; one on a paired object is found_object; one on nothing is unannotated."""
    objects = [(0, 0, 10, 10), (20, 0, 30, 10)]
    flags = [(0, 0, 10, 10), (1, 1, 11, 11), (20, 0, 30, 10), (50, 50, 60, 60)]
    e = M.evaluate_flags(flags, objects, ["miss", "found"], [True, False], 0.3)
    assert e.category == ["missed_object", "missed_object", "found_object", "unannotated"]
    assert e.hit_ids == [("miss",), ("miss",), (), ()] and e.n_flags == [2, 1]
    assert e.best_iou[3] == 0.0 and e.best_missed_iou[2] == 0.0
    empty = M.evaluate_flags([], objects, ["miss", "found"], [True, False], 0.3)
    assert empty.category == [] and empty.n_flags == [0, 0]


def test_flag_table_arithmetic() -> None:
    """Precision counts flags; recall counts distinct misses hit, also among clearly visible and flaggable misses."""
    key = {"dataset": "d", "scene": "s", "camera_id": "c", "frame": "0", "kind": "cross_frame", "labeller": "x",
           "source": "s", "witness": "other_frames"}
    flags = pd.DataFrame([
        {**key, "view_key": "v1", "category": "missed_object", "hit_ids": "m1", "support": 1, "score": 0.9,
         "annotated_class": True, "max_label_iou": 0.0, "best_gt_iou": 0.8},
        {**key, "view_key": "v1", "category": "missed_object", "hit_ids": "m1", "support": 2, "score": 0.8,
         "annotated_class": True, "max_label_iou": 0.2, "best_gt_iou": 0.7},
        {**key, "view_key": "v2", "category": "found_object", "hit_ids": "", "support": 1, "score": 0.7,
         "annotated_class": True, "max_label_iou": 0.05, "best_gt_iou": 0.5},
        {**key, "view_key": "v2", "category": "unannotated", "hit_ids": "", "support": 3, "score": 0.6,
         "annotated_class": False, "max_label_iou": 0.0, "best_gt_iou": 0.15},
    ], columns=list(M.FLAG_ROW_COLUMNS))
    targets = pd.DataFrame([
        {**key, "view_key": "v1", "object_id": "m1", "missed": True, "clearly_visible": True, "flaggable": True},
        {**key, "view_key": "v1", "object_id": "m2", "missed": True, "clearly_visible": False, "flaggable": True},
        {**key, "view_key": "v2", "object_id": "m3", "missed": True, "clearly_visible": False, "flaggable": False},
        {**key, "view_key": "v2", "object_id": "p1", "missed": False, "clearly_visible": True, "flaggable": None},
    ], columns=list(M.TARGET_ROW_COLUMNS))
    views = pd.DataFrame([{**key, "view_key": v, "n_flags": 2, "n_candidates": 3} for v in ("v1", "v2", "v3")],
                         columns=list(M.VIEW_ROW_COLUMNS))
    t = M.flag_table(flags, targets, views).set_index(["by", "group"])
    row = t.loc[("all", "all")]
    assert (row["n_views"], row["n_flags"], row["n_hit"]) == (3, 4, 2)
    assert (row["n_found_object"], row["n_unannotated"]) == (1, 1)
    assert row["precision"] == 0.5 and row["n_unannotated_near"] == 1 and row["flags_per_view"] == pytest.approx(4 / 3)
    assert (row["n_present"], row["n_missed"], row["n_missed_flagged"]) == (4, 3, 1)
    assert row["flag_recall"] == pytest.approx(1 / 3) and row["labeller_recall"] == pytest.approx(0.25)
    assert row["flag_recall_clear"] == 1.0 and row["n_missed_flaggable"] == 2 and row["flag_recall_flaggable"] == 0.5
    assert t.loc[("min_support", "2"), "n_flags"] == 2 and t.loc[("min_support", "2"), "precision"] == 0.5
    assert t.loc[("min_support", "3"), "n_hit"] == 0 and t.loc[("min_support", "3"), "flag_recall"] == 0.0
    assert t.loc[("max_label_iou", "0.1"), "n_flags"] == 3 and t.loc[("annotated_class", "True"), "n_flags"] == 3
    assert ("min_score", "0.25") not in t.index                                      # scores only for cross_labeller


# --------------------------------------------------------------------------------------------------------
# Flags: cross-labeller
# --------------------------------------------------------------------------------------------------------


def cutr_box(view: FrameView, box: OrientedBox, box2d: tuple[float, float, float, float], score: float) -> CutrBox:
    """A CuTR box as a predictions line would give it.

    Parameters
    ----------
    view : FrameView
        The view.
    box : OrientedBox
        World box.
    box2d : tuple of float
        Its 2D box.
    score : float
        Its score.

    Returns
    -------
    CutrBox
        With ``center_cam`` consistent with the view's pose.
    """
    return CutrBox(box, score, box2d, transform(view.T_cw, box.center[None])[0])


def test_cross_labeller_flags_both_directions() -> None:
    """CuTR sees chair-c and a wall object that v0 lacks; v0 sees two chairs that CuTR lacks."""
    result = labelled_view(["table-1", "chair-a", "chair-b"])
    view = result.view
    wall = OrientedBox([0.0, WALL_Y - 0.05, 2.5], [0.4, 0.1, 0.4], np.eye(3))
    boxes = (cutr_box(view, CHAIR_C.box, visible_edges(result, "chair-c"), 0.8),
             cutr_box(view, TABLE.box, visible_edges(result, "table-1"), 0.9),
             cutr_box(view, wall, (300.0, 20.0, 340.0, 60.0), 0.3))
    predictions = ViewPredictions(result.key, "cutr", None, 0.25, (640, 480), None, boxes)
    out = M.cross_labeller_flags(result, "laser", "v0", predictions, "cutr")
    flags = pd.DataFrame(out.flags)
    on_v0 = flags[flags["labeller"] == "v0"]
    assert sorted(on_v0["category"]) == ["missed_object", "unannotated"]
    assert on_v0.loc[on_v0["category"] == "missed_object", "hit_ids"].tolist() == ["chair-c"]
    on_cutr = flags[flags["labeller"] == "cutr"]
    assert (on_cutr["source"] == M.CUTR_BOXES_SOURCE).all() and (on_cutr["witness"] == "v0/laser").all()
    assert sorted(on_cutr["hit_ids"]) == ["chair-a", "chair-b"] and (on_cutr["phrase"] == "chair").all()
    metrics = M.flag_table(flags, pd.DataFrame(out.targets), pd.DataFrame(out.views)).set_index(
        ["labeller", "by", "group"])
    v0 = metrics.loc[("v0", "all", "all")]
    assert (v0["n_flags"], v0["n_hit"], v0["n_unannotated"], v0["n_missed"], v0["n_missed_flagged"]) == (2, 1, 1, 1, 1)
    cutr = metrics.loc[("cutr", "all", "all")]
    assert (cutr["precision"], cutr["flag_recall"], cutr["labeller_recall"]) == (1.0, 1.0, 0.5)
    assert metrics.loc[("v0", "min_score", "0.5"), "n_flags"] == 1                     # the wall box scores 0.3

    moved = ViewPredictions("arkitscenes/other/wide/0", "cutr", None, 0.25, (640, 480), None, boxes)
    with pytest.raises(ValueError, match="given for view"):
        M.cross_labeller_flags(result, "laser", "v0", moved, "cutr")


# --------------------------------------------------------------------------------------------------------
# Flags: projection, cross-frame and cross-camera
# --------------------------------------------------------------------------------------------------------


def test_project_box_reproduces_the_visible_box_and_respects_occlusion() -> None:
    """A ground-truth box projects onto its own visible box; an occluding depth map or the camera's back hides it."""
    result = labelled_view([])
    view = result.view
    depth, _ = render(view, OBJECTS)
    config = M.FlagConfig()
    box2d, info = M.project_box(view, CHAIR_C.box, depth, config)
    assert box2d is not None and info["unoccluded_fraction"] > 0.99
    assert box_iou_2d([box2d], [visible_edges(result, "chair-c")])[0, 0] == pytest.approx(1.0)
    wall_in_front = np.full_like(depth, 2.0)                                          # something at 2 m everywhere
    assert M.project_box(view, CHAIR_C.box, wall_in_front, config)[0] is None
    assert M.project_box(view, CHAIR_C.box, None, config)[0] is not None              # geometry only
    behind = OrientedBox([0.5, -4.0, 0.0], [0.5, 0.5, 0.9], np.eye(3))
    off_side = OrientedBox([40.0, 4.0, 0.0], [0.5, 0.5, 0.9], np.eye(3))
    assert not M.could_appear(view, behind) and not M.could_appear(view, off_side) and M.could_appear(view, CHAIR_C.box)


def test_cluster_boxes_groups_overlapping_boxes() -> None:
    """Two overlapping boxes form one cluster seeded by the higher score; a distant box forms its own."""
    a = OrientedBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], np.eye(3))
    b = OrientedBox([0.1, 0.0, 0.0], [1.0, 1.0, 1.0], np.eye(3))
    c = OrientedBox([5.0, 0.0, 0.0], [1.0, 1.0, 1.0], np.eye(3))
    assert M.cluster_boxes([a, b, c], [0.5, 0.9, 0.7], 0.25).tolist() == [0, 0, 1]
    assert M.cluster_boxes([a, b], [0.5, 0.9], 0.99).tolist() == [1, 0]
    assert M.cluster_boxes([], [], 0.25).tolist() == []


def test_cross_frame_flag_lands_on_the_object_missed_in_one_view() -> None:
    """chair-c is detected in one view and missed in another; its box from the first is flagged in the second."""
    first = labelled_view(["table-1", "chair-a", "chair-b", "chair-c"], view=make_view(0.0, frame="1"))
    second = labelled_view(["table-1", "chair-a", "chair-b"], view=make_view(0.3, frame="2"))
    depths = {r.key: render(r.view, OBJECTS)[0] for r in (first, second)}
    out = M.cross_frame_flags([first, second], "laser", "v0",
                              depth_of=lambda v: depths[f"arkitscenes/synthetic/wide/{v.frame}"])
    assert len(out.views) == 2 and len(out.flags) == 1
    flag = out.flags[0]
    assert flag["view_key"] == second.key and flag["origin"] == first.key and flag["support"] == 1
    assert flag["category"] == "missed_object" and flag["hit_ids"] == "chair-c" and flag["phrase"] == "chair"
    iou = box_iou_2d([(flag["x0"], flag["y0"], flag["x1"], flag["y1"])], [visible_edges(second, "chair-c")])[0, 0]
    assert iou > 0.6                                    # the lifted box covers the visible part: close, not exact
    missed = [t for t in out.targets if t["missed"]]
    assert [(t["object_id"], t["n_flags"], t["flaggable"]) for t in missed] == [("chair-c", 1, True)]

    occluded = M.cross_frame_flags([first, second], "laser", "v0",
                                   depth_of=lambda v: np.full((48, 64), 1.0))      # low-res depth, all occluding
    assert occluded.flags == [] and [t["flaggable"] for t in occluded.targets if t["missed"]] == [False]
    with pytest.raises(ValueError, match="one video"):
        M.cross_frame_flags([first, labelled_view([], view=make_view(scene="other"))], "laser", "v0")


def test_cross_camera_flags_at_the_same_frame() -> None:
    """Camera 1 misses chair-b, camera 2 sees it at the same frame; frames without a second camera are not covered."""
    cam1 = labelled_view(["table-1", "chair-a", "chair-c"], view=make_view(0.0, camera_id="cam1", frame="5"))
    cam2 = labelled_view(["table-1", "chair-a", "chair-b", "chair-c"],
                         view=make_view(-0.4, camera_id="cam2", frame="5"))
    alone = labelled_view(["table-1"], view=make_view(0.0, camera_id="cam1", frame="6"))
    out = M.cross_camera_flags([cam1, cam2, alone], "laser", "v0")
    assert {v["view_key"] for v in out.views} == {cam1.key, cam2.key}
    assert [(f["view_key"], f["category"], f["hit_ids"]) for f in out.flags] == [(cam1.key, "missed_object", "chair-b")]
    assert out.flags[0]["witness"] == "other_cameras" and out.flags[0]["kind"] == "cross_camera"


def test_flag_config_validation() -> None:
    """Thresholds outside their ranges are rejected."""
    with pytest.raises(ValueError):
        M.FlagConfig(min_iou=0.0)
    with pytest.raises(ValueError):
        M.FlagConfig(min_support=0)
    assert M.FlagConfig().to_json()["occlusion_margin"] == 0.3


# --------------------------------------------------------------------------------------------------------
# Reading shards and the script
# --------------------------------------------------------------------------------------------------------


def write_shard(root: Path, result: ViewResult) -> Path:
    """Store a view as ``scripts/run_labels.py`` does.

    Parameters
    ----------
    root : pathlib.Path
        Label folder.
    result : ViewResult
        The view.

    Returns
    -------
    pathlib.Path
        The shard file.
    """
    v = result.view
    path = root / v.dataset / "shards" / f"{v.scene.replace('/', '__')}__{v.camera_id}__{v.frame}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": json.loads(to_json_line(result.records()))}))
    return path


def test_load_view_reproduces_the_stored_pairs_and_rejects_tampering(tmp_path: Path) -> None:
    """A stored view re-associates to its own pairs; a stored pair that the association cannot give is an error.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Scratch folder.
    """
    result = labelled_view(["table-1", "chair-a", "chair-c"], lidar_fails=("chair-c",))
    path = write_shard(tmp_path, result)
    loaded = M.load_view(path)
    assert M.result_pairs(loaded) == M.result_pairs(result)
    assert M.stored_pairs(json.loads(path.read_text())["records"]) == M.result_pairs(result)
    payload = json.loads(path.read_text())
    record = next(r for r in payload["records"]["oracle_objects"] if r["object_id"] == "chair-c")
    record["match"]["lidar"] = dict(record["match"]["laser"])                          # a pair the lift cannot give
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="re-association"):
        M.load_view(path)
    assert M.load_view(path, check=False) is not None
    scenes = list(M.iter_scenes(tmp_path, "arkitscenes", check=False))
    assert [(s, len(r)) for s, r in scenes] == [("synthetic", 1)]


def _script() -> ModuleType:
    """Import ``scripts/run_miss_audit.py`` as a module (it is not part of the package).

    Returns
    -------
    types.ModuleType
        The module; importing it has no other side effects.
    """
    if "run_miss_audit" in sys.modules:
        return sys.modules["run_miss_audit"]
    spec = importlib.util.spec_from_file_location("run_miss_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_miss_audit"] = module
    spec.loader.exec_module(module)
    return module


def test_script_writes_tables_and_refuses_non_empty_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    """End to end on two stored views with CuTR predictions; outputs land in the data root's misses folder only.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Scratch folder, also the data root.
    monkeypatch : pytest.MonkeyPatch
        Points the data root at ``tmp_path``.
    capsys : pytest.CaptureFixture
        Captures the script's output.
    """
    monkeypatch.setenv("MEASURED_SPACE_DATA", str(tmp_path))
    labels = tmp_path / "labels"
    first = labelled_view(["table-1", "chair-a", "chair-b", "chair-c"], view=make_view(0.0, frame="1"))
    second = labelled_view(["table-1", "chair-a", "chair-b"], view=make_view(0.3, frame="2"))
    for r in (first, second):
        write_shard(labels, r)
    pred = tmp_path / "predictions" / "arkitscenes" / "synthetic.jsonl"
    pred.parent.mkdir(parents=True)
    line = {"view_key": second.key, "dataset": "arkitscenes", "scene": "synthetic", "camera_id": "wide", "frame": "2",
            "model": "cutr", "image_size": [640, 480],
            "boxes": [{"center": CHAIR_C.box.center.tolist(), "size": CHAIR_C.box.size.tolist(),
                       "rotation": CHAIR_C.box.rotation.tolist(), "score": 0.7,
                       "box2d": list(visible_edges(second, "chair-c")),
                       "center_cam": transform(second.view.T_cw, CHAIR_C.box.center[None])[0].tolist()}]}
    pred.write_text(json.dumps(line) + "\n")
    results = tmp_path / "agg"
    comparison = tmp_path / "comparison.csv"
    pd.DataFrame([{"dataset": "arkitscenes", "depth_source": "laser", "association": "2d", "n_present": 8,
                   "n_pairs": 7, "recall": 0.875}]).to_csv(comparison, index=False)
    script = _script()
    argv = ["--name", "synthetic", "--labels", f"v0={labels}", "arkitscenes:laser", "--predictions",
            f"cutr={tmp_path / 'predictions'}", "--flags", "cross_labeller", "--results", str(results),
            "--n-boot", "50", "--compare-with", str(comparison)]
    assert script.main(argv) == 0
    out = tmp_path / "outputs" / "misses" / "synthetic"
    assert {p.name for p in out.iterdir()} == {"objects.csv", "questions.csv", "flag_rows.csv", "flag_targets.csv",
                                               "flag_views.csv", "manifest.json"}
    assert {p.name for p in results.iterdir()} == {"recall.csv", "miss_causes.csv", "answer_effects.csv",
                                                   "flag_metrics.csv", "comparison_check.csv", "summary.json"}
    recall = pd.read_csv(results / "recall.csv")
    overall = recall[(recall["by"] == "all") & (recall["population"] == "present")].iloc[0]
    assert (overall["labeller"], overall["source"], overall["n"], overall["n_paired"]) == ("v0", "laser", 8, 7)
    flags = pd.read_csv(out / "flag_rows.csv")
    assert flags[flags["labeller"] == "v0"]["hit_ids"].tolist() == ["chair-c"]
    summary = json.loads((results / "summary.json").read_text())
    assert summary["counts"]["views_without_predictions"] == {"cutr": 1}
    assert [c["agrees"] for c in summary["comparison_check"]] == [True]
    assert "unannotated_flag" in summary["definitions"]

    with pytest.raises(SystemExit):                                                  # both folders now hold files
        script.parse_args(argv)
    assert "not empty" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        script.parse_args([a for a in argv if a not in ("--results", str(results))])  # --results is required
