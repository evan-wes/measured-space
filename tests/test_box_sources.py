"""Tests for the automatic sources from external boxes and their script, on a synthetic view and predictions.

The view is the one of ``tests/test_pipeline.py``: a chair and a table in front of a wall, seen by a camera at
the origin looking along world +y, with ray-cast depth and masks. The label pipeline runs on it (no models) and
its records stand in for a stored shard. CuTR's boxes are written by hand:

- box A is a slightly wrong chair;
- box B is a slightly wrong table, whose 2D box runs off the image;
- box C is a box elsewhere, or, in the CLIP test, a box on the chair that CLIP calls "wall".

The ``external_named`` tests label the same boxes (A "chair", B "table", C "lamp", not a dataset class) and add
a low-score box D. Everything runs on the CPU; the CLIP model is a fake that reads colours.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from measured_space.audit import load_labels, referent_features
from measured_space.box_sources import (
    CLIP_SOURCE,
    EXTERNAL_NAMED,
    NO_CUTR_BOX,
    NO_LABEL,
    NOT_ASSIGNED,
    SAM3_SOURCE,
    UNKNOWN_LABEL,
    UNNAMED_PREFIX,
    ViewPredictions,
    attach_cutr,
    box_mask_stats,
    check_predictions,
    check_source_name,
    clip_box2d,
    cover_by_nearer,
    cutr_clip_records,
    cutr_sam3_records,
    external_named_records,
    external_names,
    overlap_summary,
    parse_view_predictions,
    predictions_file,
    read_box_predictions,
    run_box_source,
)
from measured_space.clip_naming import PROMPT_TEMPLATE, UNNAMED_OTHER, BoxName, vocabulary
from measured_space.geometry import Array, Camera, OrientedBox, make_pose
from measured_space.matching import Match2DConfig
from measured_space.perception import Instance
from measured_space.pipeline import box_json, process_view, rematch_view, to_json_line
from measured_space.qa_schema import QAItem
from measured_space.scene import FrameView, SceneObject

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CAM = Camera(500.0, 500.0, 319.5, 239.5, 640, 480)
LOOK_ALONG_Y = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])   # columns: camera x, y, z in the world
WALL_Y = 8.0
CHAIR = SceneObject("chair-1", "chair", OrientedBox([-0.8, 3.0, 0.0], [0.5, 0.5, 0.9], np.eye(3)))
TABLE = SceneObject("table-1", "table", OrientedBox([0.8, 4.0, 0.0], [1.2, 0.8, 0.8], np.eye(3)))
HIDDEN = SceneObject("sofa-1", "sofa", OrientedBox([0.0, -3.0, 0.0], [2.0, 1.0, 0.8], np.eye(3)))  # behind


# --------------------------------------------------------------------------------------------------------
# Synthetic shards and predictions
# --------------------------------------------------------------------------------------------------------


def make_view(frame: str = "0") -> FrameView:
    """The synthetic view (an ARKitScenes-like dataset name so the phrase mapping applies).

    Parameters
    ----------
    frame : str, default "0"
        Frame identifier (several frames share the geometry).

    Returns
    -------
    FrameView
        Camera at the origin looking along +y.
    """
    return FrameView("arkitscenes", "synthetic", "wide", frame, CAM, make_pose(LOOK_ALONG_Y, np.zeros(3)),
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
    depth = WALL_Y / rays[..., 1]
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


def stored_records(frame: str = "0") -> dict[str, Any]:
    """A view's records as a shard stores them: SAM 3 chair, table and a spurious "chair" on the wall.

    Parameters
    ----------
    frame : str, default "0"
        Frame identifier.

    Returns
    -------
    dict
        ``ViewResult.records()`` after a JSON round trip, with depth sources ``laser`` and ``moge``.
    """
    view = make_view(frame)
    depth, index = render(view, [CHAIR, TABLE])
    wall = np.zeros_like(index, dtype=bool)
    wall[20:60, 300:340] = True
    instances = [instance("chair", index == 0, 0.9), instance("table", index == 1, 0.8),
                 instance("chair", wall, 0.6)]
    result = process_view(view, [CHAIR, TABLE, HIDDEN], instances, {"laser": depth, "moge": depth * 1.1},
                          visibility="laser")
    return json.loads(to_json_line(result.records()))


@pytest.fixture(scope="module")
def records() -> dict[str, Any]:
    """The stored records of frame "0".

    Returns
    -------
    dict
        See :func:`stored_records`.
    """
    return stored_records("0")


def cutr_box(view: FrameView, box: OrientedBox, score: float, box2d: Sequence[float]) -> dict[str, Any]:
    """One CuTR box in the predictions format, ``center_cam`` from the view's pose.

    Parameters
    ----------
    view : FrameView
        The view.
    box : OrientedBox
        World box.
    score : float
        CuTR score.
    box2d : sequence of float
        CuTR's 2D box.

    Returns
    -------
    dict
        ``center``, ``size``, ``rotation``, ``score``, ``box2d``, ``center_cam``.
    """
    center_cam = view.T_cw[:3, :3] @ box.center + view.T_cw[:3, 3]
    return {**box_json(box), "score": score, "box2d": [float(v) for v in box2d], "center_cam": center_cam.tolist()}


BOX_A = OrientedBox([-0.75, 3.1, 0.02], [0.55, 0.5, 0.85], np.eye(3))
BOX_B = OrientedBox([0.85, 4.1, 0.0], [1.3, 0.9, 0.75], np.eye(3))
BOX_C = OrientedBox([-2.0, 6.0, 1.0], [0.3, 0.3, 0.3], np.eye(3))


def prediction_line(records: dict[str, Any], *, c_on_chair: bool = False, c_score: float = 0.5) -> dict[str, Any]:
    """CuTR's output for a stored view: boxes A (chair), B (table) and C (elsewhere, or on the chair).

    Parameters
    ----------
    records : dict
        The stored view records (for the SAM 3 boxes and the view key).
    c_on_chair : bool, default False
        Put box C's 2D box exactly on the chair's.
    c_score : float, default 0.5
        Box C's score.

    Returns
    -------
    dict
        One line of a predictions file.
    """
    v = records["view"]
    view = make_view(v["frame"])
    det = {r["detection"]: r["box2d"] for r in records["detections"] if r["source"] == "laser"}
    chair, table = det[0], det[1]
    a2d = [chair[0] + 3, chair[1] + 2, chair[2] + 3, chair[3] - 2]
    b2d = [table[0] - 4, table[1], 700.0, table[3] + 3]               # runs off the 640 px wide image
    c2d = list(chair) if c_on_chair else [0.0, 0.0, 30.0, 30.0]
    return {"view_key": v["view_key"], "dataset": v["dataset"], "scene": v["scene"], "camera_id": v["camera_id"],
            "frame": v["frame"], "model": "cutr_rgbd", "checkpoint_sha256": "0" * 64, "score_threshold": 0.25,
            "image_size": [640, 480], "seconds": 0.12,
            "boxes": [cutr_box(view, BOX_A, 0.8, a2d), cutr_box(view, BOX_B, 0.7, b2d),
                      cutr_box(view, BOX_C, c_score, c2d)]}


# --------------------------------------------------------------------------------------------------------
# Reading and checks
# --------------------------------------------------------------------------------------------------------


def test_read_box_predictions_parses_lines_and_rejects_duplicates(records: dict[str, Any], tmp_path: Path) -> None:
    """Lines parse into world boxes and provenance; a repeated view or inconsistent identity is an error.

    Parameters
    ----------
    records : dict
        The stored view records.
    tmp_path : pathlib.Path
        Temporary folder.
    """
    line = prediction_line(records)
    path = tmp_path / "arkitscenes" / "synthetic.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(line) + "\n\n")
    views = read_box_predictions(path)
    view = views[records["view"]["view_key"]]
    assert view.model == "cutr_rgbd" and view.image_size == (640, 480) and view.seconds == 0.12
    assert len(view.boxes) == 3 and view.boxes[0].score == 0.8
    np.testing.assert_array_equal(view.boxes[1].box.center, BOX_B.center)
    assert view.meta()["n_cutr_boxes"] == 3
    assert predictions_file(tmp_path, "arkitscenes", "synthetic") == path
    assert predictions_file(tmp_path, "arkitscenes", "other") is None
    nested = tmp_path / "smartspaces" / "MTMC__train__W1.jsonl"
    nested.parent.mkdir()
    nested.write_text("")
    assert predictions_file(tmp_path, "smartspaces", "MTMC/train/W1") == nested
    path.write_text(json.dumps(line) + "\n" + json.dumps(line) + "\n")
    with pytest.raises(ValueError, match="twice"):
        read_box_predictions(path)
    with pytest.raises(ValueError, match="disagrees"):
        parse_view_predictions({**line, "frame": "99"})


def test_check_predictions_rejects_another_frame_size_or_pose(records: dict[str, Any]) -> None:
    """The view key, the image size and every ``center_cam`` must agree with the stored view.

    Parameters
    ----------
    records : dict
        The stored view records.
    """
    line = prediction_line(records)
    assert check_predictions(records["view"], parse_view_predictions(line)) < 1e-9
    with pytest.raises(ValueError, match="image"):
        check_predictions(records["view"], parse_view_predictions({**line, "image_size": [480, 640]}))
    shifted = copy.deepcopy(line)
    shifted["boxes"][1]["center_cam"][2] += 0.05
    with pytest.raises(ValueError, match="center_cam"):
        check_predictions(records["view"], parse_view_predictions(shifted))
    other = stored_records("1")
    with pytest.raises(ValueError, match="given for view"):
        check_predictions(other["view"], parse_view_predictions(line))


# --------------------------------------------------------------------------------------------------------
# Attachment
# --------------------------------------------------------------------------------------------------------


def test_attach_cutr_maximises_summed_iou_not_greedy() -> None:
    """Greedy would give d0 its best box c0 and leave d1 a poor one; the assignment maximises the sum."""
    dets = [(0.0, 0.0, 10.0, 10.0), (2.0, 0.0, 12.0, 10.0)]
    cutr = [(0.5, 0.0, 10.5, 10.0), (-3.0, 0.0, 9.0, 10.0)]
    # IoU: d0-c0 9.5/10.5 = 0.905, d0-c1 9/13 = 0.692, d1-c0 8.5/11.5 = 0.739, d1-c1 7/15 = 0.467.
    assigned = attach_cutr(dets, cutr)
    assert set(assigned) == {0, 1}
    assert assigned[0][0] == 1 and assigned[0][1] == pytest.approx(9 / 13)
    assert assigned[1][0] == 0 and assigned[1][1] == pytest.approx(8.5 / 11.5)


def test_attach_cutr_threshold_and_unassigned_detections() -> None:
    """IoU 0.316 attaches at the default 0.3, IoU 0.282 does not; extra detections stay unassigned."""
    det = [(0.0, 0.0, 10.0, 10.0)]
    assert attach_cutr(det, [(5.2, 0.0, 15.2, 10.0)])[0][1] == pytest.approx(4.8 / 15.2)
    assert attach_cutr(det, [(5.6, 0.0, 15.6, 10.0)]) == {}
    assert attach_cutr(det, [(5.2, 0.0, 15.2, 10.0)], Match2DConfig(min_iou=0.5)) == {}
    three = [(0.0, 0.0, 10.0, 10.0), (100.0, 0.0, 110.0, 10.0), (200.0, 0.0, 210.0, 10.0)]
    assigned = attach_cutr(three, [(1.0, 0.0, 11.0, 10.0), (201.0, 0.0, 211.0, 10.0)])
    assert {i: j for i, (j, _) in assigned.items()} == {0: 0, 2: 1}
    assert attach_cutr([], [(0.0, 0.0, 1.0, 1.0)]) == {} and attach_cutr(det, []) == {}


def test_box_mask_stats_and_clipping() -> None:
    """Border touch within ``boundary_px`` of an edge, counted per side; the box area; clipping to the image."""
    stats = box_mask_stats((3.0, 100.0, 200.0, 300.0), 640, 480, 5.0)
    assert stats["border_touch"] == 1.0 and stats["n_border_sides"] == 1.0
    assert stats["mask_area_px"] == 197.0 * 200.0 and stats["box_mask"] == 1.0
    assert "boundary_fraction" not in stats
    assert box_mask_stats((10.0, 10.0, 630.0, 470.0), 640, 480, 5.0)["border_touch"] == 0.0
    assert box_mask_stats((10.0, 10.0, 636.0, 476.0), 640, 480, 5.0)["n_border_sides"] == 2.0
    assert clip_box2d((-5.0, 10.0, 700.0, 500.0), 640, 480) == (0.0, 10.0, 640.0, 480.0)


# --------------------------------------------------------------------------------------------------------
# cutr_sam3
# --------------------------------------------------------------------------------------------------------


def _strip_match(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Oracle object records without their per-source ``match`` field.

    Parameters
    ----------
    rows : sequence of dict
        ``oracle_objects`` records.

    Returns
    -------
    list of dict
        The records with ``match`` removed.
    """
    return [{k: v for k, v in r.items() if k != "match"} for r in rows]


def test_cutr_sam3_attaches_cutr_geometry_and_keeps_oracle_records(records: dict[str, Any]) -> None:
    """SAM 3's detections keep their identity, get CuTR boxes and features; the oracle records are untouched.

    Parameters
    ----------
    records : dict
        The stored view records.
    """
    before = copy.deepcopy(records)
    predictions = parse_view_predictions(prediction_line(records))
    out, info = cutr_sam3_records(records, predictions)
    assert records == before                                                          # input not modified
    assert out["oracle_objects"] == records["oracle_objects"]
    assert out["qa"] == {"oracle": records["qa"]["oracle"]}
    assert out["view"]["depth_sources"] == [SAM3_SOURCE] and out["view"]["n_detections"] == 3
    assert set(out["view"]["timings"]) == {"oracle_visibility", "cutr"} and out["view"]["timings"]["cutr"] == 0.12
    assert info["n_assigned"] == 2 and info["attach_min_iou"] == 0.3 and info["n_cutr_boxes"] == 3

    sam3 = {r["detection"]: r for r in records["detections"] if r["source"] == "laser"}
    dets = {r["detection"]: r for r in out["detections"]}
    for i in range(3):
        for key in ("phrase", "labels", "score", "box2d", "mask"):
            assert dets[i][key] == sam3[i][key]
        assert dets[i]["source"] == SAM3_SOURCE
    assert dets[0]["box"] == box_json(BOX_A) and dets[1]["box"] == box_json(BOX_B)
    features = dets[0]["features"]
    assert features["depth_median"] == pytest.approx(3.1) and features["det_score"] == 0.8
    assert features["boundary_fraction"] == sam3[0]["mask"]["boundary_fraction"]
    assert features["border_touch"] == sam3[0]["mask"]["border_touch"]
    assert 0.3 <= features["pair_iou2d"] < 1.0 and features["cutr_box_index"] == 0.0
    assert not {"depth_mad_rel", "footprint_hull_fill", "n_points"} & set(features)
    assert dets[2]["box"] is None and dets[2]["features"] == {}
    assert dets[2]["lift_error"] == NOT_ASSIGNED.format(min_iou=0.3)


def test_cutr_sam3_items_come_from_rematch_with_the_new_source(records: dict[str, Any]) -> None:
    """Pairing, questions and answers are the existing procedure; items are ``auto_cutr_sam3``.

    Parameters
    ----------
    records : dict
        The stored view records.
    """
    result, info = run_box_source(records, parse_view_predictions(prediction_line(records)), SAM3_SOURCE)
    assert set(result.items) == {"oracle", "auto_cutr_sam3"} and result.sources == (SAM3_SOURCE,)
    assert result.config.association == "2d"
    assert json.loads(to_json_line([i.to_json() for i in result.items["oracle"]])) == records["qa"]["oracle"]
    auto = result.items["auto_cutr_sam3"]
    assert auto and all(i.source == "auto_cutr_sam3" for i in auto)
    assert {(m.detection, m.object_id) for m in result.matches[SAM3_SOURCE].matches} == {(0, "chair-1"),
                                                                                          (1, "table-1")}
    for item in auto:
        for obj in item.objects:
            if obj["object_id"] == "chair-1":
                assert obj["center"] == BOX_A.center.tolist() and obj["size"] == BOX_A.size.tolist()
        if item.spec.qtype != "count":
            feats = referent_features(item.extra)
            assert np.isfinite(feats["log_range"]) and np.isfinite(feats["det_score"])
            assert np.isnan(feats["hull_fill"]) and np.isnan(feats["log_n_points"])
    counts = {i.spec.params["phrase"]: i for i in auto if i.spec.qtype == "count"}
    assert counts["chair"].answer == 1                          # the wall "chair" got no CuTR box: not counted
    assert counts["chair"].extra["n_detections_2d"] == 2 and counts["chair"].extra["n_lift_failed"] == 1

    out = result.records()
    assert _strip_match(out["oracle_objects"]) == _strip_match(records["oracle_objects"])
    assert all(r["match"] is None or set(r["match"]) == {SAM3_SOURCE} for r in out["oracle_objects"])
    assert {r["status"] for r in out["detections"]} == {"matched", "unboxed"}
    assert out["view"]["depth_sources"] == [SAM3_SOURCE]
    again = rematch_view(json.loads(to_json_line(out)), association="2d")            # as a stored shard
    assert again.items["auto_cutr_sam3"] == auto


def test_cutr_sam3_without_cutr_boxes_answers_nothing_about_objects(records: dict[str, Any]) -> None:
    """With no CuTR box every detection fails, so no referent has an automatic box.

    Parameters
    ----------
    records : dict
        The stored view records.
    """
    line = {**prediction_line(records), "boxes": []}
    result, info = run_box_source(records, parse_view_predictions(line), SAM3_SOURCE)
    assert info["n_assigned"] == 0
    assert all(lift.error == NO_CUTR_BOX for lift in result.lifts[SAM3_SOURCE])
    assert all(i.spec.qtype == "count" for i in result.items["auto_cutr_sam3"])
    assert len(result.unanswered["auto_cutr_sam3"]) == sum(i.spec.qtype != "count" for i in result.items["oracle"])


# --------------------------------------------------------------------------------------------------------
# cutr_clip
# --------------------------------------------------------------------------------------------------------


def _name(phrase: str | None, best: str, prob: float) -> BoxName:
    """A CLIP verdict for tests.

    Parameters
    ----------
    phrase : str or None
        The name, or ``None``.
    best : str
        The best phrase.
    prob : float
        Its probability.

    Returns
    -------
    BoxName
        The verdict (an "other" verdict when ``phrase`` is ``None``).
    """
    return BoxName(phrase, best, prob, None if phrase else UNNAMED_OTHER, (prob, 1.0 - prob))


def test_cutr_clip_uses_cutr_boxes_and_unnamed_boxes_take_no_part(records: dict[str, Any]) -> None:
    """Box C covers the chair exactly but CLIP calls it "wall": it is kept unboxed and matches nothing.

    Parameters
    ----------
    records : dict
        The stored view records.
    """
    predictions = parse_view_predictions(prediction_line(records, c_on_chair=True, c_score=0.95))
    names = [_name("chair", "chair", 0.9), _name("table", "table", 0.8), _name(None, "wall", 0.7)]
    before = copy.deepcopy(records)
    out, info = cutr_clip_records(records, predictions, names, naming_seconds=0.5)
    assert records == before
    assert out["oracle_objects"] == records["oracle_objects"] and out["qa"] == {"oracle": records["qa"]["oracle"]}
    assert out["view"]["depth_sources"] == [CLIP_SOURCE] and out["view"]["n_detections"] == 3
    assert out["view"]["timings"] == {"oracle_visibility": records["view"]["timings"]["oracle_visibility"],
                                      "cutr": 0.12, "clip_naming": 0.5}
    assert info["cutr_box_index"] == [2, 0, 1] and info["n_named"] == 2 and info["n_unnamed"] == {"other": 1}

    dets = out["detections"]                                   # highest score first: C, A, B
    assert [d["score"] for d in dets] == [0.95, 0.8, 0.7]
    assert dets[0]["phrase"] == "wall" and dets[0]["labels"] == [] and dets[0]["box"] is None
    assert dets[0]["lift_error"] == f"{UNNAMED_PREFIX}{UNNAMED_OTHER}" and dets[0]["features"]["name_prob"] == 0.7
    assert dets[1]["phrase"] == "chair" and dets[1]["labels"] == ["chair"] and dets[1]["box"] == box_json(BOX_A)
    assert dets[2]["box2d"][2] == 640.0                                            # clipped to the image
    for d in dets:
        assert d["mask"]["box_mask"] == 1.0 and "boundary_fraction" not in d["mask"]
    f = dets[1]["features"]
    assert f["depth_median"] == pytest.approx(3.1) and f["det_score"] == 0.8 and f["name_prob"] == 0.9
    assert f["border_touch"] == dets[1]["mask"]["border_touch"] and f["cutr_box_index"] == 0.0
    assert dets[2]["features"]["border_touch"] == 1.0                              # B reaches the right edge

    result = rematch_view(out, association="2d")
    assert set(result.items) == {"oracle", "auto_cutr_clip"}
    assert all(i.source == "auto_cutr_clip" for i in result.items["auto_cutr_clip"])
    assert {(m.detection, m.object_id) for m in result.matches[CLIP_SOURCE].matches} == {(1, "chair-1"),
                                                                                          (2, "table-1")}
    assert result.matches[CLIP_SOURCE].unboxed == (0,)
    counts = {i.spec.params["phrase"]: i for i in result.items["auto_cutr_clip"] if i.spec.qtype == "count"}
    assert counts["chair"].answer == 1 and counts["chair"].extra["n_detections_2d"] == 1
    assert json.loads(to_json_line([i.to_json() for i in result.items["oracle"]])) == records["qa"]["oracle"]


def test_cutr_clip_input_errors(records: dict[str, Any]) -> None:
    """One name per box is required, names must be dataset phrases, and the variant must be known.

    Parameters
    ----------
    records : dict
        The stored view records.
    """
    predictions = parse_view_predictions(prediction_line(records))
    with pytest.raises(ValueError, match="names"):
        cutr_clip_records(records, predictions, [_name("chair", "chair", 0.9)])
    with pytest.raises(ValueError, match="not a arkitscenes phrase"):
        cutr_clip_records(records, predictions, [_name("forklift", "forklift", 0.9)] * 3)
    with pytest.raises(ValueError, match="needs CLIP names"):
        run_box_source(records, predictions, CLIP_SOURCE)
    with pytest.raises(ValueError, match="variant"):
        run_box_source(records, predictions, "cutr")


# --------------------------------------------------------------------------------------------------------
# external_named
# --------------------------------------------------------------------------------------------------------

BOX_D = OrientedBox([2.0, 6.0, 1.0], [0.3, 0.3, 0.3], np.eye(3))


def named_line(records: dict[str, Any]) -> dict[str, Any]:
    """A named labeller's output for a stored view: A "chair", B "table", C "lamp", and D "chair" at score 0.1.

    Parameters
    ----------
    records : dict
        The stored view records.

    Returns
    -------
    dict
        One line of a predictions file, with a ``label`` per box.
    """
    line = prediction_line(records)
    for box, label in zip(line["boxes"], ("chair", "table", "lamp")):
        box["label"] = label
    line["boxes"].append({**cutr_box(make_view(records["view"]["frame"]), BOX_D, 0.1, [600.0, 0.0, 630.0, 30.0]),
                          "label": "chair"})
    return line


def test_external_named_names_boxes_by_their_labels(records: dict[str, Any]) -> None:
    """Boxes become detections named by their labels; unknown labels are unnamed; low scores are left out.

    Parameters
    ----------
    records : dict
        The stored view records.
    """
    predictions = parse_view_predictions(named_line(records))
    assert [b.label for b in predictions.boxes] == ["chair", "table", "lamp", "chair"]
    assert parse_view_predictions(prediction_line(records)).boxes[0].label is None           # CuTR: no labels
    before = copy.deepcopy(records)
    out, info = external_named_records(records, predictions, "ext", min_score=0.2)
    assert records == before
    assert out["oracle_objects"] == records["oracle_objects"] and out["qa"] == {"oracle": records["qa"]["oracle"]}
    assert out["view"]["depth_sources"] == ["ext"] and out["view"]["n_detections"] == 3
    assert out["view"]["timings"] == {"oracle_visibility": records["view"]["timings"]["oracle_visibility"],
                                      "ext": 0.12}
    assert info["variant"] == EXTERNAL_NAMED and info["source"] == "ext" and info["min_score"] == 0.2
    assert info["n_below_min_score"] == 1 and info["n_named"] == 2
    unknown = UNKNOWN_LABEL.format(label="lamp", dataset="arkitscenes")
    assert info["n_unnamed"] == {unknown: 1}
    assert info["external_box_index"] == [0, 1, 2] and info["external_label"] == ["chair", "table", "lamp"]

    dets = out["detections"]                                             # highest score first: A, B, C
    assert [(d["phrase"], d["labels"], d["score"]) for d in dets] == [("chair", ["chair"], 0.8),
                                                                      ("table", ["table"], 0.7), ("lamp", [], 0.5)]
    assert dets[0]["box"] == box_json(BOX_A) and dets[2]["box"] is None
    assert dets[2]["lift_error"] == f"{UNNAMED_PREFIX}{unknown}"
    assert dets[1]["box2d"][2] == 640.0 and dets[1]["features"]["border_touch"] == 1.0
    assert all(d["mask"]["box_mask"] == 1.0 and d["source"] == "ext" for d in dets)
    f = dets[0]["features"]
    assert set(f) == {"depth_median", "border_touch", "n_border_sides", "det_score", "external_box_index",
                      "cover_by_nearer"}
    assert f["depth_median"] == pytest.approx(3.1) and f["det_score"] == 0.8 and f["external_box_index"] == 0.0

    every, info_all = external_named_records(records, predictions, "ext")
    assert every["view"]["n_detections"] == 4 and info_all["n_below_min_score"] == 0
    assert every["detections"][3]["phrase"] == "chair" and every["detections"][3]["score"] == 0.1


def test_external_named_items_come_from_rematch(records: dict[str, Any]) -> None:
    """The transformed records re-associate and answer as ``auto_<name>``; a stored shard re-matches the same.

    Parameters
    ----------
    records : dict
        The stored view records.
    """
    result, info = run_box_source(records, parse_view_predictions(named_line(records)), EXTERNAL_NAMED,
                                  source_name="sparse4d", min_score=0.2)
    assert set(result.items) == {"oracle", "auto_sparse4d"} and result.sources == ("sparse4d",)
    assert result.config.association == "2d"
    assert {(m.detection, m.object_id) for m in result.matches["sparse4d"].matches} == {(0, "chair-1"),
                                                                                        (1, "table-1")}
    assert all(m.label_correct for m in result.matches["sparse4d"].matches)
    assert result.matches["sparse4d"].unboxed == (2,)
    assert json.loads(to_json_line([i.to_json() for i in result.items["oracle"]])) == records["qa"]["oracle"]
    auto = result.items["auto_sparse4d"]
    assert auto and all(i.source == "auto_sparse4d" for i in auto)
    counts = {i.spec.params["phrase"]: i for i in auto if i.spec.qtype == "count"}
    assert counts["chair"].answer == 1 and counts["chair"].extra["n_detections_2d"] == 1     # D is below 0.2
    for item in auto:
        if item.spec.qtype != "count":
            feats = referent_features(item.extra)
            assert np.isfinite(feats["log_range"]) and np.isfinite(feats["det_score"])
    again = rematch_view(json.loads(to_json_line(result.records())), association="2d")      # as a stored shard
    assert again.items["auto_sparse4d"] == auto


def test_external_names_put_the_box_label_first_and_name_by_phrase() -> None:
    """A label names its class's phrase; the phrase's other classes follow, so pairing is phrase-level."""
    assert external_names("smartspaces", "Transporter") == ("mobile robot", ("Transporter", "NovaCarter"))
    assert external_names("smartspaces", "NovaCarter") == ("mobile robot", ("NovaCarter", "Transporter"))
    assert external_names("smartspaces", "PalletTruck") == ("pallet jack", ("PalletTruck",))
    assert external_names("arkitscenes", "tv_monitor") == ("tv", ("tv_monitor",))
    assert external_names("smartspaces", "Robot") is None and external_names("smartspaces", None) is None
    assert external_names("unknown_dataset", "Person") is None
    assert NO_LABEL == "no label"


def test_external_named_argument_checks(records: dict[str, Any]) -> None:
    """Source names are checked; the variant needs one; other variants take neither it nor a minimum score.

    Parameters
    ----------
    records : dict
        The stored view records.
    """
    assert check_source_name("sparse4d") == "sparse4d" and check_source_name("tao_sparse4d_v2") == "tao_sparse4d_v2"
    for bad in ("Sparse4D", "4d", "sparse-4d", "", "x" * 41):
        with pytest.raises(ValueError, match="lower case"):
            check_source_name(bad)
    for taken in ("moge", "lidar", "laser", "gt_depth", "oracle", SAM3_SOURCE, CLIP_SOURCE, EXTERNAL_NAMED):
        with pytest.raises(ValueError, match="taken"):
            check_source_name(taken)
    predictions = parse_view_predictions(named_line(records))
    with pytest.raises(ValueError, match="needs a source name"):
        run_box_source(records, predictions, EXTERNAL_NAMED)
    with pytest.raises(ValueError, match="only"):
        run_box_source(records, predictions, SAM3_SOURCE, min_score=0.2)
    with pytest.raises(ValueError, match="finite"):
        external_named_records(records, predictions, "ext", min_score=float("nan"))
    with pytest.raises(ValueError, match="taken"):
        external_named_records(records, predictions, "moge")


def test_cover_by_nearer_and_overlap_summary() -> None:
    """A far box half hidden by a near one is half covered; the near one is not covered at all."""
    boxes = [(0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0), (100.0, 100.0, 110.0, 110.0)]
    np.testing.assert_allclose(cover_by_nearer(boxes, [2.0, 5.0, 1.0]), [0.0, 0.5, 0.0])
    np.testing.assert_allclose(cover_by_nearer(boxes, [5.0, 2.0, 1.0]), [0.5, 0.0, 0.0])
    np.testing.assert_allclose(cover_by_nearer(boxes, [3.0, 3.0, 3.0]), [0.0, 0.0, 0.0])    # equal depth: no
    assert cover_by_nearer([], []).shape == (0,)
    with pytest.raises(ValueError):
        cover_by_nearer(boxes, [1.0])
    summary = overlap_summary(boxes, [2.0, 5.0, 1.0])
    assert summary == {"n_boxes": 3, "n_pairs_overlap": 1, "n_pairs_iou": 1, "n_boxes_overlap": 2, "n_covered": 1}
    assert overlap_summary(boxes, [2.0, 5.0, 1.0], min_iou=0.5, min_cover=0.6)["n_pairs_iou"] == 0
    touching = [(0.0, 0.0, 10.0, 10.0), (10.0, 0.0, 20.0, 10.0)]                             # a shared edge
    assert overlap_summary(touching, [1.0, 2.0])["n_pairs_overlap"] == 0
    assert overlap_summary(boxes[:1], [1.0])["n_boxes"] == 1


# --------------------------------------------------------------------------------------------------------
# The script
# --------------------------------------------------------------------------------------------------------


def _script(name: str) -> ModuleType:
    """Import a script of ``scripts/`` as a module, registered in ``sys.modules`` (sibling imports need it).

    Parameters
    ----------
    name : str
        Script name without ``.py``.

    Returns
    -------
    types.ModuleType
        The module; importing it has no other side effects.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _script_module() -> ModuleType:
    """``scripts/run_box_source.py``, with ``scripts/run_labels.py`` imported first for its sibling import.

    Returns
    -------
    types.ModuleType
        The script's module.
    """
    _script("run_labels")
    return _script("run_box_source")


@pytest.fixture
def folders(tmp_path: Path) -> dict[str, Path]:
    """A source label folder with two assembled synthetic views, and CuTR predictions for both.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.

    Returns
    -------
    dict of str to pathlib.Path
        ``labels``, ``predictions`` and ``out`` (not created).
    """
    run_labels = _script("run_labels")
    labels, preds = tmp_path / "labels", tmp_path / "preds"
    lines = []
    for frame in ("0", "1"):
        recs = stored_records(frame)
        run_labels.write_shard(labels / "arkitscenes" / "shards" / f"synthetic__wide__{frame}.json",
                               {"records": recs, "wall_seconds": 1.5, "started": "t0", "finished": "t1"})
        lines.append(prediction_line(recs))
    run_labels.assemble(labels, ["arkitscenes"], None)
    (preds / "arkitscenes").mkdir(parents=True)
    (preds / "arkitscenes" / "synthetic.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return {"labels": labels, "predictions": preds, "out": tmp_path / "out"}


def _argv(folders: dict[str, Path], variant: str, *extra: str) -> list[str]:
    """Command-line arguments for the script.

    Parameters
    ----------
    folders : dict of str to pathlib.Path
        From the ``folders`` fixture.
    variant : str
        The source to build.
    *extra : str
        More arguments.

    Returns
    -------
    list of str
        The arguments.
    """
    return ["--variant", variant, "--labels-from", str(folders["labels"]), "--predictions",
            str(folders["predictions"]), "--out-dir", str(folders["out"]), "--datasets", "arkitscenes", *extra]


def test_script_cutr_sam3_writes_run_labels_layout_and_logs_inputs(folders: dict[str, Path]) -> None:
    """Shards, assembled files and run log; the oracle file is byte-identical; the audit reads the result.

    Parameters
    ----------
    folders : dict of str to pathlib.Path
        Source labels, predictions and the output folder.
    """
    script = _script_module()
    assert script.main(_argv(folders, SAM3_SOURCE)) == 0
    out = folders["out"] / "arkitscenes"
    for name in ("views.jsonl", "oracle_objects.jsonl", "detections.jsonl", "qa_oracle.jsonl",
                 "qa_auto_cutr_sam3.jsonl", "summary.json"):
        assert (out / name).exists(), name
    assert not list(out.glob("qa_auto_laser*"))
    source_oracle = folders["labels"] / "arkitscenes" / "qa_oracle.jsonl"
    assert (out / "qa_oracle.jsonl").read_bytes() == source_oracle.read_bytes()
    items = [QAItem.from_json(json.loads(x)) for x in (out / "qa_auto_cutr_sam3.jsonl").read_text().splitlines()]
    assert items and {i.source for i in items} == {"auto_cutr_sam3"}
    views = [json.loads(x) for x in (out / "views.jsonl").read_text().splitlines()]
    assert len(views) == 2 and all(v["depth_sources"] == [SAM3_SOURCE] for v in views)

    shard = json.loads((out / "shards" / "synthetic__wide__0.json").read_text())
    assert shard["box_source"]["n_assigned"] == 2
    assert shard["box_source"]["predictions_file"] == "arkitscenes/synthetic.jsonl"
    assert shard["rematch"]["association"] == "2d" and shard["wall_seconds"] == 1.5
    runs = [json.loads(x) for x in (folders["out"] / "runs.jsonl").read_text().splitlines()]
    assert len(runs) == 1
    run = runs[0]
    pred_sha = hashlib.sha256((folders["predictions"] / "arkitscenes" / "synthetic.jsonl").read_bytes()).hexdigest()
    assert run["prediction_files"] == {"arkitscenes": {"arkitscenes/synthetic.jsonl": pred_sha}}
    assert run["source_shards"]["arkitscenes"]["n"] == 2 and run["clip"] is None
    assert run["oracle_identical"]["arkitscenes"]["identical"] is True and run["counts"]["processed"] == 2
    inputs = json.loads((folders["out"] / "inputs.json").read_text())
    assert script.shard_digest(inputs["shards"]["arkitscenes"]) == run["source_shards"]["arkitscenes"]["digest"]

    pairs = load_labels(out)
    assert set(pairs["depth_source"]) == {SAM3_SOURCE} and pairs["matched"].any()


def _colour_logits(images: Sequence[Image.Image], texts: Sequence[str]) -> NDArray[np.float64]:
    """Fake CLIP: red crops are chairs, green tables, blue walls (by the centre pixel).

    Parameters
    ----------
    images : sequence of PIL.Image.Image
        Crops.
    texts : sequence of str
        Prompts.

    Returns
    -------
    numpy.ndarray, shape (len(images), len(texts))
        Logits.
    """
    phrases = [t.removeprefix(PROMPT_TEMPLATE.format("")) for t in texts]
    out = np.zeros((len(images), len(texts)))
    for i, image in enumerate(images):
        a = np.asarray(image)
        out[i, phrases.index(("chair", "table", "wall")[int(np.argmax(a[a.shape[0] // 2, a.shape[1] // 2]))])] = 9.0
    return out


def _painted_image(view: FrameView) -> NDArray[np.uint8]:
    """An image with box A's region red, B's green and C's blue (the fake CLIP reads these colours).

    Parameters
    ----------
    view : FrameView
        The view.

    Returns
    -------
    numpy.ndarray, shape (480, 640, 3), uint8
        The image.
    """
    line = prediction_line(stored_records(view.frame))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    for channel, box in enumerate(line["boxes"]):
        x0, y0, x1, y1 = clip_box2d(box["box2d"], 640, 480)
        image[int(y0):int(y1), int(x0):int(x1), channel] = 255
    return image


def test_script_cutr_clip_with_a_fake_model(folders: dict[str, Path]) -> None:
    """The CLIP variant names boxes through the injected model and logs the vocabulary.

    Parameters
    ----------
    folders : dict of str to pathlib.Path
        Source labels, predictions and the output folder.
    """
    script = _script_module()
    assert script.main(_argv(folders, CLIP_SOURCE), logits_fn=_colour_logits, load_image=_painted_image) == 0
    out = folders["out"] / "arkitscenes"
    items = [json.loads(x) for x in (out / "qa_auto_cutr_clip.jsonl").read_text().splitlines()]
    assert items and {i["source"] for i in items} == {"auto_cutr_clip"}
    dets = [json.loads(x) for x in (out / "detections.jsonl").read_text().splitlines()]
    assert [d["phrase"] for d in dets[:3]] == ["chair", "table", "wall"]
    assert dets[2]["box"] is None and dets[2]["lift_error"] == f"{UNNAMED_PREFIX}{UNNAMED_OTHER}"
    run = json.loads((folders["out"] / "runs.jsonl").read_text())
    assert run["clip"]["model_id"] == "injected" and run["clip"]["naming"]["min_prob"] is None
    assert run["clip"]["vocabulary"]["arkitscenes"] == [list(p) for p in vocabulary("arkitscenes")]
    assert run["oracle_identical"]["arkitscenes"]["identical"] is True
    shard = json.loads((out / "shards" / "synthetic__wide__1.json").read_text())
    assert shard["box_source"]["clip_best_phrase"] == ["chair", "table", "wall"]
    assert "clip_naming" in shard["records"]["view"]["timings"]


def test_script_refuses_a_non_empty_or_overlapping_out_dir(folders: dict[str, Path]) -> None:
    """Nothing is written into an existing non-empty folder or into an input folder.

    Parameters
    ----------
    folders : dict of str to pathlib.Path
        Source labels, predictions and the output folder.
    """
    script = _script_module()
    folders["out"].mkdir()
    (folders["out"] / "keep.txt").write_text("mine")
    assert script.main(_argv(folders, SAM3_SOURCE)) == 1
    assert [p.name for p in folders["out"].iterdir()] == ["keep.txt"]
    assert (folders["out"] / "keep.txt").read_text() == "mine"
    inside = folders["labels"] / "new"
    assert script.main(_argv({**folders, "out": inside}, SAM3_SOURCE)) == 1 and not inside.exists()
    with pytest.raises(FileExistsError):
        script.refuse_existing(folders["out"])
    empty = folders["labels"].parent / "empty"
    empty.mkdir()
    script.refuse_existing(empty, folders["labels"])                                 # an empty folder is fine


def test_script_missing_predictions_stop_the_run_unless_skipped(folders: dict[str, Path]) -> None:
    """A view without a prediction line stops the run before anything is written, unless skipped.

    Parameters
    ----------
    folders : dict of str to pathlib.Path
        Source labels, predictions and the output folder.
    """
    script = _script_module()
    path = folders["predictions"] / "arkitscenes" / "synthetic.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    assert script.main(_argv(folders, SAM3_SOURCE)) == 1 and not folders["out"].exists()
    assert script.main(_argv(folders, SAM3_SOURCE, "--skip-missing")) == 0
    run = json.loads((folders["out"] / "runs.jsonl").read_text())
    assert run["counts"]["skipped_no_predictions"] == 1 and run["counts"]["processed"] == 1
    assert run["oracle_identical"]["arkitscenes"]["compared"] is False


def test_script_external_named_builds_a_named_source(folders: dict[str, Path]) -> None:
    """``--variant external_named --source-name ext --min-score 0.2`` writes ``qa_auto_ext`` and logs both.

    Parameters
    ----------
    folders : dict of str to pathlib.Path
        Source labels, predictions and the output folder.
    """
    script = _script_module()
    path = folders["predictions"] / "arkitscenes" / "synthetic.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(named_line(stored_records(x["frame"]))) + "\n" for x in lines))
    argv = _argv(folders, EXTERNAL_NAMED, "--source-name", "ext", "--min-score", "0.2")
    assert script.main(argv) == 0
    out = folders["out"] / "arkitscenes"
    items = [json.loads(x) for x in (out / "qa_auto_ext.jsonl").read_text().splitlines()]
    assert items and {i["source"] for i in items} == {"auto_ext"}
    run = json.loads((folders["out"] / "runs.jsonl").read_text())
    assert run["variant"] == EXTERNAL_NAMED and run["source_name"] == "ext" and run["min_score"] == 0.2
    assert run["oracle_identical"]["arkitscenes"]["identical"] is True and run["counts"]["processed"] == 2
    assert run["counts"]["detections"] == 6 and run["counts"]["named"] == 4 and run["counts"]["below_min_score"] == 2
    assert run["counts"]["overlap_n_boxes"] == 6
    shard = json.loads((out / "shards" / "synthetic__wide__0.json").read_text())
    assert shard["box_source"]["external_label"] == ["chair", "table", "lamp"]
    assert load_labels(out)["depth_source"].unique().tolist() == ["ext"]


@pytest.mark.parametrize("extra", [(EXTERNAL_NAMED, "--min-score", "0.2"), (EXTERNAL_NAMED, "--source-name", "ext"),
                                   (EXTERNAL_NAMED, "--source-name", "moge", "--min-score", "0.2"),
                                   (SAM3_SOURCE, "--min-score", "0.2"), (SAM3_SOURCE, "--source-name", "ext")])
def test_script_refuses_inconsistent_external_options(folders: dict[str, Path], extra: tuple[str, ...]) -> None:
    """The named variant needs a valid name and a minimum score; the CuTR variants take neither.

    Parameters
    ----------
    folders : dict of str to pathlib.Path
        Source labels, predictions and the output folder.
    extra : tuple of str
        The variant and its options.
    """
    script = _script_module()
    with pytest.raises(SystemExit):
        script.main(_argv(folders, extra[0], *extra[1:]))
    assert not folders["out"].exists()


def test_predictions_of_another_view_are_refused(records: dict[str, Any]) -> None:
    """A prediction line is never applied to another view.

    Parameters
    ----------
    records : dict
        The stored view records.
    """
    other: ViewPredictions = parse_view_predictions(prediction_line(stored_records("1")))
    with pytest.raises(ValueError):
        run_box_source(records, other, SAM3_SOURCE)
