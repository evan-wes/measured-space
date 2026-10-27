"""Tests for the label pipeline on a synthetic view: oracle visibility, lifting, matching and paired answers.

The scene is a chair and a table in front of a wall, seen by a camera at the origin looking along world +y
(world z up). Depth and instance masks are ray-cast exactly, so the pipeline runs without any model. An
opt-in test runs the real models on one ARKitScenes view (``MEASURED_SPACE_GPU_TESTS=1``).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from numpy.typing import NDArray

from measured_space.geometry import Array, Camera, OrientedBox, make_pose
from measured_space.perception import Instance
from measured_space.pipeline import (
    AUTO_KEY_PREFIX,
    GT_BOX2D,
    PipelineConfig,
    gt_box_visibility,
    process_view,
    projected_extent,
    to_json_line,
)
from measured_space.scene import FrameView, SceneObject

CAM = Camera(500.0, 500.0, 319.5, 239.5, 640, 480)
LOOK_ALONG_Y = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])   # columns: camera x, y, z in the world
WALL_Y = 8.0
CHAIR = SceneObject("chair-1", "chair", OrientedBox([-0.8, 3.0, 0.0], [0.5, 0.5, 0.9], np.eye(3)))
TABLE = SceneObject("table-1", "table", OrientedBox([0.8, 4.0, 0.0], [1.2, 0.8, 0.8], np.eye(3)))
HIDDEN = SceneObject("sofa-1", "sofa", OrientedBox([0.0, -3.0, 0.0], [2.0, 1.0, 0.8], np.eye(3)))  # behind


def make_view() -> FrameView:
    """The synthetic view (an ARKitScenes-like dataset name so the phrase mapping applies).

    Returns
    -------
    FrameView
        Camera at the origin looking along +y.
    """
    return FrameView("arkitscenes", "synthetic", "wide", "0", CAM, make_pose(LOOK_ALONG_Y, np.zeros(3)),
                     "img.png", up=np.array([0.0, 0.0, 1.0]))


def render(view: FrameView, objects: list[SceneObject]) -> tuple[Array, NDArray[np.int64]]:
    """Ray-cast planar depth and an object-index map for axis-aligned boxes in front of a wall.

    Parameters
    ----------
    view : FrameView
        The view.
    objects : list of SceneObject
        Axis-aligned boxes.

    Returns
    -------
    depth : numpy.ndarray, shape (H, W)
        Planar depth, metres (the wall where no box is hit).
    index : numpy.ndarray, shape (H, W), int
        Index of the nearest box per pixel, -1 for the wall.
    """
    cam = view.camera
    v, u = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
    rays = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], axis=-1) @ view.T_wc[:3, :3].T
    depth = WALL_Y / rays[..., 1]                       # planar depth equals the ray parameter (camera z = 1)
    index = np.full(depth.shape, -1)
    origin = view.camera_center
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


@pytest.fixture(scope="module")
def scene() -> tuple[FrameView, Array, list[Instance]]:
    """The view, its depth, and detections: chair, table, and a spurious second chair on the wall.

    Returns
    -------
    tuple
        ``(view, depth, instances)``.
    """
    view = make_view()
    depth, index = render(view, [CHAIR, TABLE])
    wall_patch = np.zeros_like(index, dtype=bool)
    wall_patch[20:60, 300:340] = True
    assert not (index[wall_patch] >= 0).any()
    return view, depth, [instance("chair", index == 0, 0.9), instance("table", index == 1, 0.8),
                         instance("chair", wall_patch, 0.6)]


def test_process_view_pairs_oracle_and_auto(scene: tuple[FrameView, Array, list[Instance]]) -> None:
    """Both objects are matched, the wall detection is a false positive, and answers pair by spec id.

    Parameters
    ----------
    scene : tuple
        The synthetic view, depth and detections.
    """
    view, depth, instances = scene
    result = process_view(view, [CHAIR, TABLE, HIDDEN], instances, {"laser": depth, "moge": depth * 1.1},
                          visibility="laser")
    seen = {o.obj.object_id: o for o in result.oracle}
    assert seen["chair-1"].visible_fraction > 0.9 and seen["table-1"].visible_fraction > 0.9
    assert seen["sofa-1"].visible_fraction == 0.0 and seen["sofa-1"].bbox2d is None

    match = result.matches["laser"]
    assert {(m.detection, m.object_id) for m in match.matches} == {(0, "chair-1"), (1, "table-1")}
    assert match.false_positives == (2,) and match.missed == ()
    assert all(m.distance < m.gate for m in match.matches)

    oracle = {i.id: i for i in result.items["oracle"]}
    assert set(oracle) == {s.id for s in result.specs} and len(oracle) > 0
    for source in ("auto_laser", "auto_moge"):
        auto = {i.id: i for i in result.items[source]}
        assert set(auto) <= set(oracle) and result.unanswered[source] == []
        assert all(i.source == source for i in auto.values())
    auto = {i.id: i for i in result.items["auto_laser"]}
    for item_id, item in auto.items():
        if item.spec.qtype == "count":
            continue
        for ref in item.extra["referents"]:
            assert ref["object_id"] in ("chair-1", "table-1") and ref["score"] in (0.9, 0.8)
            assert len(ref["features"]) == 17 and ref["match_distance"] < 0.5
        if item.spec.qtype == "size_height":
            # Exact depth, but erosion (3 px of a ~100 px tall mask) and percentile trimming shrink the height.
            assert -0.1 < item.answer - oracle[item_id].answer < 0.0

    counts = {i.spec.params["phrase"]: i for i in result.items["auto_laser"] if i.spec.qtype == "count"}
    oracle_counts = {i.spec.params["phrase"]: i.answer for i in oracle.values() if i.spec.qtype == "count"}
    assert oracle_counts == {"chair": 1, "table": 1}
    assert counts["chair"].answer == 2 and counts["table"].answer == 1         # the wall "chair" is counted
    counted = {c["object_id"] for c in counts["chair"].extra["counted"]}
    assert counted == {"chair-1", None} and counts["chair"].extra["n_detections_2d"] == 2
    assert any(o["object_id"] == f"{AUTO_KEY_PREFIX}2" for o in counts["chair"].objects)

    records = result.records()
    assert len(records["oracle_objects"]) == 3 and len(records["detections"]) == 3 * 2
    assert {r["status"] for r in records["detections"]} == {"matched", "false_positive"}
    json.loads(to_json_line(records))


def test_missed_referent_leaves_spec_unanswered(scene: tuple[FrameView, Array, list[Instance]]) -> None:
    """Without the table detection, every question naming the table has no automatic answer.

    Parameters
    ----------
    scene : tuple
        The synthetic view, depth and detections.
    """
    view, depth, instances = scene
    result = process_view(view, [CHAIR, TABLE], instances[:1], {"laser": depth}, visibility="laser")
    assert result.matches["laser"].missed == ("table-1",)
    unanswered = result.unanswered["auto_laser"]
    table_specs = [s for s in result.specs if "table-1" in [r.object_id for r in s.referents]]
    assert table_specs and {u["id"] for u in unanswered} == {s.id for s in table_specs}
    assert all(u["missing"] == ["table-1"] for u in unanswered)
    answered = {i.id for i in result.items["auto_laser"]}
    assert answered.isdisjoint(u["id"] for u in unanswered)


def test_lift_stride_keeps_the_box_and_full_resolution_mask_features(
        scene: tuple[FrameView, Array, list[Instance]]) -> None:
    """Lifting every second pixel gives nearly the same box, with mask features from the full mask.

    Parameters
    ----------
    scene : tuple
        The synthetic view, depth and detections.
    """
    view, depth, instances = scene
    full = process_view(view, [CHAIR, TABLE], instances[:2], {"laser": depth}, visibility="laser")
    strided = process_view(view, [CHAIR, TABLE], instances[:2], {"laser": depth}, visibility="laser",
                           config=PipelineConfig(lift_stride=2))
    for a, b in zip(full.lifts["laser"], strided.lifts["laser"]):
        assert np.linalg.norm(a.box.center - b.box.center) < 0.02
        assert np.allclose(a.box.size[2], b.box.size[2], atol=0.02)
        assert a.features["mask_area_px"] == b.features["mask_area_px"]
        assert a.features["n_points"] > 3 * b.features["n_points"] and (a.stride, b.stride) == (1, 2)
    exempt = process_view(view, [CHAIR, TABLE], instances[:2], {"laser": depth}, visibility="laser",
                          config=PipelineConfig(lift_stride=2, lift_stride_min_pixels=10**9))
    assert [lift.stride for lift in exempt.lifts["laser"]] == [1, 1]          # small masks: full resolution


def test_gt_box_proxy_visibility() -> None:
    """The 2D-box proxy: full projected extent gives 1, half of it about 0.5, no ground-truth box gives 0."""
    view = make_view()
    x0, y0, x1, y1 = projected_extent(view, CHAIR.box)
    full = (int(np.ceil(x0)), int(np.ceil(y0)), int(np.floor(x1)), int(np.floor(y1)))
    seen, info = gt_box_visibility(view, CHAIR, full)
    assert seen.visible_fraction == pytest.approx(1.0, abs=0.03) and seen.bbox2d == tuple(map(float, full))
    half = (full[0], full[1], full[0] + (full[2] - full[0]) // 2, full[3])
    assert gt_box_visibility(view, CHAIR, half)[0].visible_fraction == pytest.approx(0.5, abs=0.03)
    hidden, info = gt_box_visibility(view, CHAIR, None)
    assert hidden.visible_fraction == 0.0 and hidden.bbox2d is None and not info["gt_visible"]

    depth, index = render(view, [CHAIR, TABLE])
    result = process_view(view, [CHAIR, TABLE], [instance("chair", index == 0, 0.9)], {"moge": depth},
                          visibility=GT_BOX2D, gt_boxes={"chair-1": full})
    assert {o.obj.object_id for o in result.oracle if o.bbox2d is not None} == {"chair-1"}
    assert [m.object_id for m in result.matches["moge"].matches] == ["chair-1"]


@pytest.mark.skipif(os.environ.get("MEASURED_SPACE_GPU_TESTS") != "1",
                    reason="set MEASURED_SPACE_GPU_TESTS=1 to run model tests")
def test_run_view_on_a_real_arkitscenes_view() -> None:
    """SAM 3, MoGe and every ARKitScenes depth source run end to end on one real view."""
    from measured_space.datasets import arkitscenes
    from measured_space.pipeline import DEPTH_SOURCES, run_view

    video = arkitscenes.VIDEO_IDS[0]
    view = arkitscenes.list_frames(video)[10]
    result = run_view(view, arkitscenes.load_objects(video), sources=DEPTH_SOURCES["arkitscenes"],
                      visibility="laser", config=PipelineConfig(lift_stride=2))
    assert result.detections and set(result.lifts) == {"laser", "lidar", "moge"}
    assert result.items["oracle"] and set(result.items) == {"oracle", "auto_laser", "auto_lidar", "auto_moge"}
