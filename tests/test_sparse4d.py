"""Tests for the Sparse4D converter (:mod:`measured_space.sparse4d`), on a synthetic warehouse view.

The view is a level camera 1.5 m above the floor looking along world +y, as a fixed warehouse camera might be. A
person and a forklift stand in front of a wall; the label pipeline runs on the rendered depth (no models) and its
records stand in for stored shards at frames 0, 6 and 12. The Sparse4D run is written by hand: at frame 0 it holds
both ground-truth boxes exactly, a person hidden behind the forklift, a box of an unknown class, a low-score box,
and three boxes the view must drop (behind the camera, outside the image, mostly outside). The run processed
frames 0 and 6, so frame 6 gets an empty line and frame 12 none.

Everything runs on the CPU.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from measured_space.audit import load_labels
from measured_space.box_sources import check_predictions, parse_view_predictions, read_box_predictions
from measured_space.datasets.smartspaces import box_from_annotation, parse_ground_truth
from measured_space.geometry import Array, Camera, OrientedBox, make_pose
from measured_space.perception import SMARTSPACES_PHRASES, Instance
from measured_space.pipeline import process_view, projected_extent, to_json_line, view_from_record
from measured_space.qa import PHRASES
from measured_space.scene import FrameView, SceneObject
from measured_space.sparse4d import (
    BEHIND_CAMERA,
    BELOW_MIN_SCORE,
    CLASS_LABELS,
    MOSTLY_OUTSIDE,
    NEAR_PLANE,
    OUTSIDE_IMAGE,
    TOO_SMALL,
    KeepRule,
    amortised_seconds,
    box_from_wlh_yaw,
    convert_view,
    dataset_label,
    main,
    parse_row,
    processed_frames,
    project_box,
    read_scene,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CAM = Camera(500.0, 500.0, 319.5, 239.5, 640, 480)
LOOK_ALONG_Y = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])     # columns: camera x, y, z in the world
CAMERA_HEIGHT = 1.5
WALL_Y = 12.0
SCENE = "MTMC_Tracking_2026/train/Warehouse_900"
NAME = "Warehouse_900"


def gt_record(object_type: str, object_id: int, center: list[float], wlh: list[float], yaw: float
              ) -> dict[str, Any]:
    """One object in the ground truth's own format (``ground_truth.json``).

    Parameters
    ----------
    object_type : str
        The ``object type``.
    object_id : int
        The ``object id``.
    center : list of float
        ``3d location``, metres.
    wlh : list of float
        ``3d bounding box scale`` (width, length, height), metres.
    yaw : float
        Yaw, radians (pitch and roll 0).

    Returns
    -------
    dict
        The record.
    """
    return {"object type": object_type, "object id": object_id, "3d location": center,
            "3d bounding box scale": wlh, "3d bounding box rotation": [0.0, 0.0, yaw],
            "2d bounding box visible": {}}


def row(frame: int, class_id: int, class_name: str, score: float, center: list[float], wlh: list[float],
        yaw: float, track_id: int = 0) -> dict[str, Any]:
    """One Sparse4D row, in the format its runner writes.

    Parameters
    ----------
    frame : int
        Frame index.
    class_id : int
        Class id.
    class_name : str
        Class name.
    score : float
        Score.
    center : list of float
        Box centre, metres.
    wlh : list of float
        ``size_wlh``, metres.
    yaw : float
        Yaw, radians.
    track_id : int, default 0
        Instance-bank id.

    Returns
    -------
    dict
        The row.
    """
    return {"scene": NAME, "frame": frame, "class_id": class_id, "class_name": class_name, "model_class_index": 0,
            "score": score, "cls_score": min(1.0, score + 0.03), "track_id": track_id, "center": center,
            "size_wlh": wlh, "yaw": yaw, "velocity": [0.0, 0.0, 0.0]}


PERSON = SceneObject("Person_0", "Person", box_from_annotation([-0.8, 4.0, 0.9], [0.5, 0.4, 1.8], [0.0, 0.0, 0.3]))
FORKLIFT = SceneObject("Forklift_1", "Forklift",
                       box_from_annotation([1.0, 6.0, 1.0], [1.2, 2.4, 2.0], [0.0, 0.0, -0.5]))


# --------------------------------------------------------------------------------------------------------
# Boxes and classes
# --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("yaw", [0.0, 0.3, -1.1692, 1.5707963, 3.14159, -3.14159])
def test_a_ground_truth_box_through_the_converter_is_the_oracle_box(yaw: float) -> None:
    """The ground truth's (location, scale, yaw) read as a Sparse4D row gives the oracle's box, bit for bit.

    Parameters
    ----------
    yaw : float
        Yaw of the box, radians.
    """
    center, wlh = [5.6657, -9.1762, 0.9787], [0.8226, 1.8853, 1.9651]
    objects, _ = parse_ground_truth({"0": [gt_record("Forklift", 2, center, wlh, yaw)]})
    oracle = objects[0][0].box
    converted = parse_row(row(0, 1, "Forklift", 0.9, center, wlh, yaw), 1).box
    for field in ("center", "size", "rotation"):
        np.testing.assert_array_equal(getattr(converted, field), getattr(oracle, field))
    np.testing.assert_array_equal(converted.corners(), oracle.corners())
    np.testing.assert_array_equal(box_from_wlh_yaw(center, wlh, yaw).rotation, oracle.rotation)


def test_box_axes_follow_width_length_height_and_yaw() -> None:
    """Width lies along the box's x axis, turned by yaw counter-clockwise about +z; height is vertical."""
    box = box_from_wlh_yaw([0.0, 0.0, 1.0], [1.0, 3.0, 2.0], np.pi / 2)
    np.testing.assert_allclose(box.rotation[:, 0], [0.0, 1.0, 0.0], atol=1e-12)   # width axis: world +y
    np.testing.assert_allclose(box.rotation[:, 1], [-1.0, 0.0, 0.0], atol=1e-12)  # length axis: world -x
    np.testing.assert_allclose(box.extent_along([0.0, 0.0, 1.0]), 2.0)
    np.testing.assert_allclose(box.extent_along([1.0, 0.0, 0.0]), 3.0)


def test_class_ids_map_to_the_dataset_object_types() -> None:
    """Every class id names a dataset object type the oracle and the questions know; names must agree."""
    assert set(CLASS_LABELS.values()) == set(SMARTSPACES_PHRASES)
    assert set(CLASS_LABELS.values()) <= set(PHRASES)
    for class_id, label in CLASS_LABELS.items():
        assert dataset_label(class_id, label) == label
    assert dataset_label(6, "PalletTruck") == "PalletTruck" and dataset_label(3, "Transporter") == "Transporter"
    assert dataset_label(42, "Robot") is None
    with pytest.raises(ValueError, match="class id 1"):
        dataset_label(1, "PalletTruck")


def test_parse_row_rejects_missing_or_malformed_fields() -> None:
    """A row needs the documented keys, three-vectors and finite numbers; optional fields may be absent."""
    good = row(0, 0, "Person", 0.9, [0.0, 4.0, 0.9], [0.5, 0.4, 1.8], 0.1, track_id=7)
    box = parse_row(good, 3)
    assert (box.row, box.frame, box.label, box.track_id, box.cls_score) == (3, 0, "Person", 7, pytest.approx(0.93))
    lean = {k: v for k, v in good.items() if k not in ("cls_score", "track_id", "velocity")}
    assert parse_row(lean, 1).track_id is None and parse_row(lean, 1).cls_score is None
    with pytest.raises(ValueError, match="missing yaw"):
        parse_row({k: v for k, v in good.items() if k != "yaw"}, 1)
    with pytest.raises(ValueError, match="non-finite"):
        parse_row({**good, "center": [0.0, float("nan"), 1.0]}, 1)
    with pytest.raises(ValueError, match="non-finite"):
        parse_row({**good, "size_wlh": [0.5, 0.4]}, 1)


def test_read_scene_groups_by_frame_and_checks_the_scene(tmp_path: Path) -> None:
    """Rows group by frame in file order; two scenes in one file are refused; an empty file has no boxes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    """
    path = tmp_path / f"{NAME}.jsonl"
    rows = [row(6, 0, "Person", 0.5, [0.0, 4.0, 0.9], [0.5, 0.4, 1.8], 0.0),
            row(0, 1, "Forklift", 0.9, [1.0, 6.0, 1.0], [1.2, 2.4, 2.0], 0.0),
            row(6, 1, "Forklift", 0.8, [1.0, 6.0, 1.0], [1.2, 2.4, 2.0], 0.0)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n")
    scene = read_scene(path)
    assert scene.scene == NAME and scene.n_rows == 3 and list(scene.by_frame) == [0, 6]
    assert [b.row for b in scene.by_frame[6]] == [1, 3] and scene.frame_rows(6) == 2 and scene.frame_rows(12) == 0
    only = read_scene(path, frames=[0])                       # boxes for frame 0 only; every frame still counted
    assert list(only.by_frame) == [0] and only.frame_rows(6) == 2 and only.n_rows == 3
    assert only.rows_per_frame == {0: 1, 6: 2}
    path.write_text(json.dumps(rows[0]) + "\n" + json.dumps({**rows[1], "scene": "Warehouse_901"}) + "\n")
    with pytest.raises(ValueError, match="2 scenes"):
        read_scene(path)
    path.write_text("")
    assert read_scene(path).scene is None and read_scene(path).n_rows == 0


def test_processed_frames_are_rebuilt_from_the_meta_checksum() -> None:
    """A uniform stride reproduces the runner's ``ids_sha256``; anything else is not trusted."""
    # The digest of the stage-1 run on Warehouse_022 (frames 0, 6, ..., 894), from its meta.json.
    meta = {"frames": {"count": 150, "first": 0, "last": 894,
                       "ids_sha256": "0323b3204950a379010a77b0f7c8f6013467dc47381cd8643c5bd2f8261009bd"}}
    assert processed_frames(meta) == frozenset(range(0, 895, 6))
    one = {"frames": {"count": 1, "first": 150, "last": 150, "ids_sha256": hashlib.sha256(b"150").hexdigest()}}
    assert processed_frames(one) == frozenset({150})
    assert processed_frames({"frames": {**meta["frames"], "ids_sha256": "0" * 64}}) is None
    assert processed_frames({"frames": {**meta["frames"], "count": 149}}) is None
    assert processed_frames({}) is None


def test_amortised_seconds_divides_frame_time_by_cameras() -> None:
    """Model time per frame over the camera count; ``None`` without both."""
    assert amortised_seconds({"timing": {"model_ms_per_frame": 400.0}, "num_cameras": 4}) == pytest.approx(0.1)
    assert amortised_seconds({"timing": {}, "num_cameras": 4}) is None
    assert amortised_seconds({"timing": {"model_ms_per_frame": 400.0}, "num_cameras": 0}) is None


# --------------------------------------------------------------------------------------------------------
# Projection and the keep rule
# --------------------------------------------------------------------------------------------------------


def make_view(frame: str = "0") -> FrameView:
    """The synthetic warehouse view.

    Parameters
    ----------
    frame : str, default "0"
        Frame identifier (the camera is fixed: every frame shares the geometry).

    Returns
    -------
    FrameView
        A level camera 1.5 m above the floor at the origin, looking along world +y.
    """
    return FrameView("smartspaces", SCENE, "Camera_0000", frame, CAM,
                     make_pose(LOOK_ALONG_Y, np.array([0.0, 0.0, CAMERA_HEIGHT])),
                     f"{SCENE}/videos/Camera_0000.mp4::{frame}", up=np.array([0.0, 0.0, 1.0]))


def square(x: float, y: float, side: float = 1.0) -> OrientedBox:
    """A flat square facing the camera, centred at camera height (its projection is exact by hand).

    Parameters
    ----------
    x, y : float
        Centre's world x and y (y is the depth), metres.
    side : float, default 1.0
        Side length, metres.

    Returns
    -------
    OrientedBox
        A box of size ``(side, 0, side)``.
    """
    return box_from_wlh_yaw([x, y, CAMERA_HEIGHT], [side, 0.0, side], 0.0)


def test_projection_uses_pixel_edges_and_clips_to_the_image() -> None:
    """A 1 m square 5 m away spans 100 px: edges at 270..370 and 190..290 (centres 269.5..369.5 plus 0.5)."""
    projection, reason = project_box(make_view(), square(0.0, 5.0))
    assert reason is None and projection is not None
    assert projection.box2d == pytest.approx((270.0, 190.0, 370.0, 290.0))
    assert projection.box2d_unclipped == projection.box2d and projection.in_image_share == pytest.approx(1.0)
    np.testing.assert_allclose(projection.center_cam, [0.0, 0.0, 5.0], atol=1e-12)
    extent = projected_extent(make_view(), square(0.0, 5.0))
    assert extent is not None and projection.box2d == pytest.approx(tuple(v + 0.5 for v in extent))


def test_keep_rule_on_a_box_straddling_the_image_edge() -> None:
    """A square at u = 610..710 has 30% inside the 640 px image: kept at 25%, dropped at 35%; 10% is dropped."""
    view = make_view()
    projection, reason = project_box(view, square(3.4, 5.0))
    assert reason is None and projection is not None
    assert projection.box2d_unclipped[0] == pytest.approx(610.0) and projection.box2d[2] == 640.0
    assert projection.in_image_share == pytest.approx(0.3)
    assert project_box(view, square(3.4, 5.0), KeepRule(min_in_image_share=0.35)) == (None, MOSTLY_OUTSIDE)
    assert project_box(view, square(3.6, 5.0)) == (None, MOSTLY_OUTSIDE)


def test_dropped_boxes_and_their_reasons() -> None:
    """Behind the camera, through the camera plane, outside the image, and too small."""
    view = make_view()
    assert project_box(view, square(0.0, -2.0)) == (None, BEHIND_CAMERA)
    through = box_from_wlh_yaw([0.0, 0.3, CAMERA_HEIGHT], [0.5, 1.0, 0.5], 0.0)       # centre in front, corners not
    assert project_box(view, through) == (None, NEAR_PLANE)
    assert project_box(view, square(50.0, 5.0)) == (None, OUTSIDE_IMAGE)
    assert project_box(view, square(0.0, 5.0, side=0.02)) == (None, TOO_SMALL)            # 2 x 2 px
    assert project_box(view, square(0.0, 5.0, side=0.02), KeepRule(min_side_px=1.0))[1] is None
    with pytest.raises(ValueError):
        KeepRule(min_in_image_share=0.0)


def view_record(frame: str = "0") -> dict[str, Any]:
    """A shard's ``"view"`` record with what the converter reads.

    Parameters
    ----------
    frame : str, default "0"
        Frame identifier.

    Returns
    -------
    dict
        Identity, ``image_ref``, ``camera``, ``T_wc`` and ``up``.
    """
    view = make_view(frame)
    return {"view_key": f"smartspaces/{SCENE}/Camera_0000/{frame}", "dataset": "smartspaces", "scene": SCENE,
            "camera_id": "Camera_0000", "frame": frame, "image_ref": view.image_ref,
            "camera": {"fx": CAM.fx, "fy": CAM.fy, "cx": CAM.cx, "cy": CAM.cy, "width": CAM.width,
                       "height": CAM.height},
            "T_wc": view.T_wc.tolist(), "up": view.up.tolist()}


def test_convert_view_writes_the_box_source_format() -> None:
    """The line parses, agrees with the view's pose, orders boxes by score and counts every drop."""
    rows = [row(0, 0, "Person", 0.5, [0.0, 5.0, 1.5], [1.0, 0.0, 1.0], 0.0, track_id=3),
            row(0, 1, "Forklift", 0.9, [0.1, 9.0, 1.5], [1.0, 0.0, 1.0], 0.0, track_id=4),   # behind the person
            row(0, 0, "Person", 0.04, [-2.0, 5.0, 1.5], [1.0, 0.0, 1.0], 0.0),
            row(0, 0, "Person", 0.7, [0.0, -2.0, 1.5], [1.0, 0.0, 1.0], 0.0)]
    boxes = [parse_row(r, i + 1) for i, r in enumerate(rows)]
    line, counts = convert_view(view_record(), boxes, min_score=0.05,
                                provenance={"checkpoint_sha256": "a" * 64, "seconds": 0.1})
    assert counts["n_in"] == 4 and counts["n_kept"] == 2
    assert counts[BELOW_MIN_SCORE] == 1 and counts[BEHIND_CAMERA] == 1
    assert line["n_dropped"][BELOW_MIN_SCORE] == 1 and line["n_dropped"][OUTSIDE_IMAGE] == 0
    assert [b["score"] for b in line["boxes"]] == [0.9, 0.5]
    assert [b["label"] for b in line["boxes"]] == ["Forklift", "Person"] and line["boxes"][1]["track_id"] == 3
    assert line["boxes"][0]["cover_by_nearer"] == pytest.approx(1.0)       # the far box is hidden by the near one
    assert line["boxes"][1]["cover_by_nearer"] == 0.0
    assert line["overlap"]["n_pairs_overlap"] == 1 and line["overlap"]["n_covered"] == 1
    assert line["score_threshold"] == 0.05 and line["checkpoint_sha256"] == "a" * 64
    predictions = parse_view_predictions(json.loads(to_json_line(line)))
    assert check_predictions(view_record(), predictions) < 1e-9
    assert [b.label for b in predictions.boxes] == ["Forklift", "Person"]
    view = view_from_record(view_record())
    assert predictions.image_size == (view.camera.width, view.camera.height)


# --------------------------------------------------------------------------------------------------------
# End to end: shards -> converter -> box source -> audit readers
# --------------------------------------------------------------------------------------------------------


def render(view: FrameView, objects: list[SceneObject]) -> tuple[Array, NDArray[np.int64]]:
    """Ray-cast planar depth and an object-index map for oriented boxes in front of a wall.

    Parameters
    ----------
    view : FrameView
        The view.
    objects : list of SceneObject
        Boxes.

    Returns
    -------
    depth : numpy.ndarray, shape (H, W)
        Planar depth, metres (the wall where no box is hit).
    index : numpy.ndarray, shape (H, W), int
        Index of the nearest box per pixel, -1 for the wall.
    """
    cam = view.camera
    v, u = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
    rays_cam = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, np.ones_like(u)], axis=-1)
    rays = rays_cam @ view.T_wc[:3, :3].T                           # world directions; camera z component is 1
    depth = (WALL_Y - view.camera_center[1]) / rays[..., 1]
    index = np.full(depth.shape, -1)
    for k, o in enumerate(objects):
        origin = (view.camera_center - o.box.center) @ o.box.rotation
        local = rays @ o.box.rotation
        half = o.box.size / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - origin) / local, (half - origin) / local
        near = np.nanmax(np.minimum(t1, t2), axis=-1)
        far = np.nanmin(np.maximum(t1, t2), axis=-1)
        hit = (near <= far) & (near > 0) & (near < depth)
        depth[hit], index[hit] = near[hit], k
    return depth, index


def stored_records(frame: str) -> dict[str, Any]:
    """A view's records as a shard stores them: SAM 3 finds the person and the forklift; MoGe depth.

    Parameters
    ----------
    frame : str
        Frame identifier.

    Returns
    -------
    dict
        ``ViewResult.records()`` after a JSON round trip, depth source ``moge``, visibility from the depth.
    """
    view = make_view(frame)
    depth, index = render(view, [PERSON, FORKLIFT])
    instances = []
    for k, (phrase, score) in enumerate((("person", 0.9), ("forklift", 0.8))):
        rows, cols = np.nonzero(index == k)
        instances.append(Instance(phrase, index == k, (float(cols.min()), float(rows.min()), float(cols.max() + 1),
                                                       float(rows.max() + 1)), score))
    result = process_view(view, [PERSON, FORKLIFT], instances, {"moge": depth}, visibility="moge")
    return json.loads(to_json_line(result.records()))


def sparse4d_rows() -> list[dict[str, Any]]:
    """The hand-written Sparse4D run: every box at frame 0, none at frame 6.

    Returns
    -------
    list of dict
        Rows in the runner's format.
    """
    yaw_p, yaw_f = 0.3, -0.5
    return [row(0, 0, "Person", 0.9, [-0.8, 4.0, 0.9], [0.5, 0.4, 1.8], yaw_p, 11),          # = ground truth
            row(0, 1, "Forklift", 0.8, [1.0, 6.0, 1.0], [1.2, 2.4, 2.0], yaw_f, 12),       # = ground truth
            row(0, 0, "Person", 0.5, [1.0, 9.5, 0.9], [0.5, 0.4, 1.8], 0.0, 13),           # hidden by the forklift
            row(0, 9, "Robot", 0.6, [-0.2, 8.0, 0.5], [0.5, 0.5, 1.0], 0.0, 14),           # unknown class
            row(0, 0, "Person", 0.1, [-5.0, 9.0, 0.9], [0.5, 0.4, 1.8], 0.0, 15),          # low score
            row(0, 0, "Person", 0.7, [0.0, -3.0, 0.9], [0.5, 0.4, 1.8], 0.0, 16),          # behind the camera
            row(0, 1, "Forklift", 0.7, [20.0, 5.0, 1.0], [1.2, 2.4, 2.0], 0.0, 17),        # outside the image
            row(0, 1, "Forklift", 0.7, [3.9, 5.0, 0.5], [1.0, 1.0, 1.0], 0.0, 18)]         # mostly outside


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


@pytest.fixture
def folders(tmp_path: Path) -> dict[str, Path]:
    """A label folder with the view at frames 0, 6 and 12, and a Sparse4D run that processed frames 0 and 6.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.

    Returns
    -------
    dict of str to pathlib.Path
        ``labels``, ``sparse4d`` (the run folder), ``out`` (not created) and ``tmp``.
    """
    run_labels = _script("run_labels")
    labels, run = tmp_path / "labels", tmp_path / "sparse4d" / "rn50_v2.2"
    for frame in ("0", "6", "12"):
        name = f"{SCENE.replace('/', '__')}__Camera_0000__{frame}.json"
        run_labels.write_shard(labels / "smartspaces" / "shards" / name,
                               {"records": stored_records(frame), "wall_seconds": 1.0, "started": "t0",
                                "finished": "t1"})
    run_labels.assemble(labels, ["smartspaces"], None)
    run.mkdir(parents=True)
    rows = sparse4d_rows()
    (run / f"{NAME}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {"tag": "rn50_v2.2", "scene": NAME, "split": "train", "checkpoint_sha256": "c" * 64,
            "frames": {"count": 2, "first": 0, "last": 6, "ids_sha256": hashlib.sha256(b"0,6").hexdigest()},
            "num_cameras": 4, "jsonl_thr": 0.05, "num_output_topk": 300, "rows_jsonl": len(rows),
            "timing": {"model_ms_per_frame": 200.0}}
    (run / f"{NAME}.meta.json").write_text(json.dumps(meta))
    return {"labels": labels, "sparse4d": run, "out": tmp_path / "predictions", "tmp": tmp_path}


def _convert_argv(folders: dict[str, Path], *extra: str) -> list[str]:
    """Arguments of the converter's command line.

    Parameters
    ----------
    folders : dict of str to pathlib.Path
        From the ``folders`` fixture.
    *extra : str
        More arguments.

    Returns
    -------
    list of str
        The arguments.
    """
    return ["convert", "--labels", str(folders["labels"]), "--sparse4d", str(folders["sparse4d"]), "--out",
            str(folders["out"]), *extra]


def test_converter_writes_lines_sidecar_and_run_record(folders: dict[str, Path]) -> None:
    """Processed frames get lines (frame 6 an empty one), frame 12 none; the sidecar records the provenance.

    Parameters
    ----------
    folders : dict of str to pathlib.Path
        Labels, the Sparse4D run and the output folder.
    """
    assert main(_convert_argv(folders)) == 0
    stem = SCENE.replace("/", "__")
    path = folders["out"] / "smartspaces" / f"{stem}.jsonl"
    views = read_box_predictions(path)
    key0, key6 = (f"smartspaces/{SCENE}/Camera_0000/{f}" for f in ("0", "6"))
    assert list(views) == [key0, key6] and views[key6].boxes == ()
    line = json.loads(path.read_text().splitlines()[0])
    assert [b["score"] for b in line["boxes"]] == [0.9, 0.8, 0.6, 0.5, 0.1]
    assert [b["label"] for b in line["boxes"]] == ["Person", "Forklift", None, "Person", "Person"]
    assert line["boxes"][2]["class_name"] == "Robot" and line["boxes"][0]["track_id"] == 11
    assert line["n_dropped"] == {BELOW_MIN_SCORE: 0, BEHIND_CAMERA: 1, NEAR_PLANE: 0, OUTSIDE_IMAGE: 1,
                                 MOSTLY_OUTSIDE: 1, TOO_SMALL: 0}
    assert line["boxes"][3]["cover_by_nearer"] == pytest.approx(1.0)              # the hidden person
    assert line["model"] == "tao_sparse4d" and line["checkpoint_tag"] == "rn50_v2.2"
    assert line["score_threshold"] == 0.05 and line["seconds"] == pytest.approx(0.05)
    assert line["frame_n_boxes"] == 8 and line["frame_at_topk"] is False
    records = {r["view"]["view_key"]: r["view"] for r in (stored_records(f) for f in ("0", "6"))}
    for key, predictions in views.items():
        assert check_predictions(records[key], predictions) < 1e-9
    for field in ("center", "size", "rotation"):                              # the ground truth's box, exactly
        np.testing.assert_array_equal(getattr(views[key0].boxes[0].box, field), getattr(PERSON.box, field))

    sidecar = json.loads((folders["out"] / "smartspaces" / f"{stem}.meta.json").read_text())
    assert sidecar["checkpoint_tag"] == "rn50_v2.2" and sidecar["min_score"] == 0.05
    assert sidecar["keep_rule"] == KeepRule().to_json() and sidecar["frames_check"] == "checksum"
    assert sidecar["views_frame_not_run"] == [f"smartspaces/{SCENE}/Camera_0000/12"]
    assert sidecar["source_folder"] == str(folders["sparse4d"].resolve()) and sidecar["n_lines"] == 2
    assert sidecar["counts"]["n_kept"] == 5 and sidecar["label_counts"] == {"Forklift": 1, "None": 1, "Person": 3}
    assert sidecar["sparse4d_meta"]["tag"] == "rn50_v2.2"
    assert sidecar["source_sha256"] == hashlib.sha256((folders["sparse4d"] / f"{NAME}.jsonl").read_bytes()).hexdigest()
    record = json.loads((folders["out"] / "convert.json").read_text())
    assert record["n_lines"] == 2 and record["n_views_frame_not_run"] == 1 and record["scenes_missing"] == []


def test_converter_refusals(folders: dict[str, Path]) -> None:
    """A non-empty or overlapping output, a missing meta, an incomplete file and a wrong split are refused.

    Parameters
    ----------
    folders : dict of str to pathlib.Path
        Labels, the Sparse4D run and the output folder.
    """
    folders["out"].mkdir()
    (folders["out"] / "keep.txt").write_text("mine")
    assert main(_convert_argv(folders)) == 1
    assert [p.name for p in folders["out"].iterdir()] == ["keep.txt"]
    inside = {**folders, "out": folders["labels"] / "predictions"}
    assert main(_convert_argv(inside)) == 1 and not inside["out"].exists()
    fresh = {**folders, "out": folders["tmp"] / "fresh"}

    meta_path = folders["sparse4d"] / f"{NAME}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "rows_jsonl": 99}))
    assert main(_convert_argv(fresh)) == 1 and not fresh["out"].exists()
    meta_path.write_text(json.dumps({**meta, "split": "val"}))
    assert main(_convert_argv(fresh)) == 1 and not fresh["out"].exists()
    meta_path.write_text(json.dumps(meta))
    unfinished = folders["tmp"] / "unfinished"                             # a run still writing has no meta yet
    unfinished.mkdir()
    (unfinished / f"{NAME}.jsonl").write_text((folders["sparse4d"] / f"{NAME}.jsonl").read_text())
    assert main(_convert_argv({**fresh, "sparse4d": unfinished})) == 1 and not fresh["out"].exists()
    assert main(_convert_argv({**fresh, "sparse4d": unfinished}, "--skip-missing")) == 1   # no scene left
    assert not fresh["out"].exists()
    good_rows = (folders["sparse4d"] / f"{NAME}.jsonl").read_text()
    bad = copy.deepcopy(sparse4d_rows())
    bad[1]["class_name"] = "PalletTruck"
    (folders["sparse4d"] / f"{NAME}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bad))
    assert main(_convert_argv(fresh)) == 1 and not fresh["out"].exists()
    (folders["sparse4d"] / f"{NAME}.jsonl").write_text(good_rows)

    # A second scene whose meta names another checkpoint: one run folder holds one checkpoint.
    shards = folders["labels"] / "smartspaces" / "shards"
    shard = json.loads((shards / f"{SCENE.replace('/', '__')}__Camera_0000__0.json").read_text())
    other = SCENE.replace(NAME, "Warehouse_901")
    shard["records"]["view"].update(scene=other, view_key=f"smartspaces/{other}/Camera_0000/0")
    (shards / f"{other.replace('/', '__')}__Camera_0000__0.json").write_text(json.dumps(shard))
    rows = [{**r, "scene": "Warehouse_901"} for r in sparse4d_rows()]
    (folders["sparse4d"] / "Warehouse_901.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (folders["sparse4d"] / "Warehouse_901.meta.json").write_text(json.dumps(
        {**meta, "scene": "Warehouse_901", "checkpoint_sha256": "d" * 64}))
    assert main(_convert_argv(fresh)) == 1 and not fresh["out"].exists()
    assert main(_convert_argv(fresh, "--scenes", NAME)) == 0                    # one scene: one checkpoint


def test_converted_boxes_build_a_source_that_the_audit_readers_take(folders: dict[str, Path]) -> None:
    """Convert, build ``external_named`` at 0.2, and read the predictions as recall control and the miss audit do.

    Parameters
    ----------
    folders : dict of str to pathlib.Path
        Labels, the Sparse4D run and the output folder.
    """
    assert main(_convert_argv(folders)) == 0
    _script("run_labels")
    box_source = _script("run_box_source")
    labels_out = folders["tmp"] / "labels_sparse4d"
    argv = ["--variant", "external_named", "--source-name", "sparse4d", "--min-score", "0.2", "--labels-from",
            str(folders["labels"]), "--predictions", str(folders["out"]), "--out-dir", str(labels_out),
            "--datasets", "smartspaces"]
    assert box_source.main(argv) == 1 and not labels_out.exists()             # frame 12 has no line
    assert box_source.main([*argv, "--skip-missing"]) == 0
    out = labels_out / "smartspaces"
    assert (out / "qa_auto_sparse4d.jsonl").exists() and not list(out.glob("qa_auto_moge*"))
    run = json.loads((labels_out / "runs.jsonl").read_text())
    assert run["source_name"] == "sparse4d" and run["min_score"] == 0.2 and run["n_shard_oracle_mismatch"] == 0
    assert run["counts"]["processed"] == 2 and run["counts"]["skipped_no_predictions"] == 1
    shard = json.loads((out / "shards" / f"{SCENE.replace('/', '__')}__Camera_0000__0.json").read_text())
    info = shard["box_source"]
    assert info["n_below_min_score"] == 1 and info["n_detections"] == 4 and info["n_named"] == 3
    assert info["n_unnamed"] == {"no label": 1} and info["external_label"] == ["Person", "Forklift", None, "Person"]
    assert info["overlap"]["n_covered"] == 1
    objects = {r["object_id"]: r["match"]["sparse4d"] for r in shard["records"]["oracle_objects"] if r["present"]}
    assert set(objects) == {"Person_0", "Forklift_1"}
    for match in objects.values():                                       # the boxes are the ground truth's
        assert match["label_correct"] is True and match["iou"] == pytest.approx(1.0) and match["distance"] < 1e-9
    dets = shard["records"]["detections"]
    assert [d["status"] for d in dets] == ["matched", "matched", "unboxed", "false_positive"]
    assert dets[3]["features"]["cover_by_nearer"] == pytest.approx(1.0)  # the hidden person: an extra detection
    pairs = load_labels(out)
    assert set(pairs["depth_source"]) == {"sparse4d"} and pairs["matched"].any()

    recall = _script("run_recall_control")
    records, lines, provenance = recall.load_dataset(folders["labels"], folders["out"], "smartspaces",
                                                     skip_missing=True)
    assert len(records) == 2 and provenance["n_views_skipped"] == 1
    assert [len(recall.view_boxes(r["view"], lines[r["view"]["view_key"]])) for r in records] == [5, 0]
    assert recall.file_floor(list(lines.values()))[0] == 0.05
    misses = _script("run_miss_audit")
    cache = misses.PredictionCache({"sparse4d": folders["out"]})
    view = cache.view("sparse4d", "smartspaces", SCENE, f"smartspaces/{SCENE}/Camera_0000/0")
    assert view is not None and len(view.boxes) == 5
