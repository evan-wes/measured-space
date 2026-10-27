"""Tests for the label pipeline on a synthetic view: oracle visibility, lifting, association and paired answers.

The scene is a chair and a table in front of a wall, seen by a camera at the origin looking along world +y
(world z up). Depth and instance masks are ray-cast exactly, so the pipeline runs without any model; a depth
map scaled by 3 stands in for a badly wrong monocular estimate. Re-association from stored records is checked
against the full pipeline, including the pairing-ambiguity evidence it adds to old records on a view where one
detection covers two adjacent objects; on that view the audit's and the fusion evaluation's ``--exclude-ambiguous``
drop exactly the ambiguous items and detections. An opt-in test runs the real models on one ARKitScenes view
(``MEASURED_SPACE_GPU_TESTS=1``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from measured_space.audit import exclude_ambiguous, pair_table
from measured_space.fusion_eval import load_detections
from measured_space.geometry import Array, Camera, OrientedBox, make_pose
from measured_space.matching import Match2DConfig
from measured_space.perception import Instance
from measured_space.pipeline import (
    AUTO_KEY_PREFIX,
    EVIDENCE_KEYS_AMBIGUITY,
    GT_BOX2D,
    PipelineConfig,
    gt_box_visibility,
    json_close,
    process_view,
    projected_extent,
    rematch_view,
    to_json_line,
    visible_box_edges,
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
    assert all(m.gate is None and m.iou2d > 0.95 and m.label_correct and not m.ambiguous and m.distance < 0.5
               for m in match.matches)                                   # 2D association: no gate
    gated = process_view(view, [CHAIR, TABLE, HIDDEN], instances, {"laser": depth}, visibility="laser",
                         config=PipelineConfig(association="3d")).matches["laser"]
    assert {(m.detection, m.object_id) for m in gated.matches} == {(0, "chair-1"), (1, "table-1")}
    assert all(m.iou2d is None and m.distance < m.gate for m in gated.matches)

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
            assert ref["label_correct"] is True and ref["match_iou2d"] > 0.95
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
    assert all(r["match"]["label_correct"] and r["match"]["gate"] is None
               for r in records["detections"] if r["match"] is not None)
    json.loads(to_json_line(records))


def test_far_errors_and_class_confusions_are_paired_not_censored(
        scene: tuple[FrameView, Array, list[Instance]]) -> None:
    """A "stool" on the chair and boxes 3x too far are paired in 2D and answered; the 3D gate drops them.

    Parameters
    ----------
    scene : tuple
        The synthetic view, depth and detections.
    """
    view, depth, instances = scene
    stool = Instance("stool", instances[0].mask, instances[0].box2d, 0.9)          # the chair, misnamed
    detections = [stool, instances[1]]
    depths = {"laser": depth, "moge": depth * 3.0}
    result = process_view(view, [CHAIR, TABLE], detections, depths, visibility="laser")
    for source in depths:
        by_object = result.matches[source].by_object()
        assert set(by_object) == {"chair-1", "table-1"}
        assert not by_object["chair-1"].label_correct and by_object["table-1"].label_correct
        assert result.unanswered[f"auto_{source}"] == []                   # every question has an auto answer
    assert min(m.distance for m in result.matches["moge"].matches) > 5.0   # measured, not gated
    auto = {i.id: i for i in result.items["auto_moge"]}
    chair_items = [i for i in auto.values() if "chair-1" in [r.object_id for r in i.spec.referents]]
    assert chair_items
    for item in chair_items:
        ref = next(r for r in item.extra["referents"] if r["object_id"] == "chair-1")
        assert ref["label_correct"] is False and ref["phrase"] == "stool" and ref["match_distance"] > 5.0
        assert next(o["label"] for o in item.objects if o["object_id"] == "chair-1") == "stool"
    counts = {i.spec.params["phrase"]: i for i in result.items["auto_laser"] if i.spec.qtype == "count"}
    assert counts["chair"].answer == 0 and counts["chair"].extra["n_detections_2d"] == 0   # counts by own class
    records = result.records()
    chair_record = next(r for r in records["oracle_objects"] if r["object_id"] == "chair-1")
    assert chair_record["match"]["moge"]["label_correct"] is False

    gated = process_view(view, [CHAIR, TABLE], detections, depths, visibility="laser",
                         config=PipelineConfig(association="3d"))
    assert gated.matches["laser"].missed == ("chair-1",)                    # the confusion becomes a miss
    assert gated.matches["moge"].matches == ()                              # the far boxes fall outside the gate
    assert len(gated.unanswered["auto_moge"]) == len(gated.specs) - 2       # only the two counts remain


def test_rematch_view_reproduces_the_pipeline(scene: tuple[FrameView, Array, list[Instance]]) -> None:
    """Re-associating stored records gives the pipeline's own answers, keeps the oracle verbatim and checks it.

    Parameters
    ----------
    scene : tuple
        The synthetic view, depth and detections.
    """
    view, depth, instances = scene
    depths = {"laser": depth, "moge": depth * 3.0}
    run = {a: process_view(view, [CHAIR, TABLE, HIDDEN], instances, depths, visibility="laser",
                           config=PipelineConfig(association=a)) for a in ("2d", "3d")}
    stored = json.loads(to_json_line(run["3d"].records()))                 # as written to a shard
    for association in ("3d", "2d"):
        expected = json.loads(to_json_line(run[association].records()))
        got = json.loads(to_json_line(rematch_view(stored, association=association).records()))
        assert [to_json_line(r) for r in got["qa"]["oracle"]] == [to_json_line(r) for r in stored["qa"]["oracle"]]
        assert json_close(got["qa"], expected["qa"]) and json_close(got["detections"], expected["detections"])
        assert got["view"]["config"]["association"] == association
    assert rematch_view(stored, association="2d", match_2d=Match2DConfig(0.999)).matches["laser"].matches == ()

    tampered = json.loads(to_json_line(run["3d"].records()))
    numeric = next(r for r in tampered["qa"]["oracle"] if r["answer_type"] == "number")
    numeric["answer"] += 1e-6
    with pytest.raises(ValueError, match="oracle answer"):
        rematch_view(tampered, association="2d")


@pytest.fixture(scope="module")
def overlap() -> tuple[FrameView, list[SceneObject], Array, list[Instance]]:
    """A chair and a table side by side, one detection covering both, and a sofa well apart.

    Returns
    -------
    tuple
        ``(view, objects, depth, instances)``: the "chair" detection's mask is the chair's and the table's pixels
        together, so its 2D box overlaps both objects; the "table" and "sofa" detections are exact.
    """
    view = make_view()
    objects = [SceneObject("chair-1", "chair", OrientedBox([-0.3, 3.0, 0.0], [0.5, 0.5, 0.9], np.eye(3))),
               SceneObject("table-1", "table", OrientedBox([0.3, 3.0, 0.0], [0.5, 0.5, 0.8], np.eye(3))),
               SceneObject("sofa-1", "sofa", OrientedBox([-1.6, 4.0, 0.0], [0.8, 0.8, 0.8], np.eye(3)))]
    depth, index = render(view, objects)
    return view, objects, depth, [instance("chair", (index == 0) | (index == 1), 0.9),
                                  instance("table", index == 1, 0.8), instance("sofa", index == 2, 0.7)]


def _evidence_of(item: dict[str, Any]) -> list[dict[str, Any]]:
    """An automatic item record's evidence: its referents, or for a count its counted objects.

    Parameters
    ----------
    item : dict
        A :meth:`QAItem.to_json` record.

    Returns
    -------
    list of dict
        The evidence records.
    """
    return item["extra"].get("referents", []) + item["extra"].get("counted", [])


def test_rematch_adds_the_pairing_ambiguity_to_old_records(
        overlap: tuple[FrameView, list[SceneObject], Array, list[Instance]]) -> None:
    """Re-matching records written without the ambiguity evidence adds it: both pairs that share the covering
    detection are ambiguous, with its overlap as runner-up; the sofa's pair is not; the 3D association leaves
    both fields unknown.

    Parameters
    ----------
    overlap : tuple
        The overlapping view.
    """
    view, objects, depth, instances = overlap
    old = json.loads(to_json_line(process_view(view, objects, instances, {"laser": depth, "moge": depth * 1.2},
                                               visibility="laser", config=PipelineConfig(association="3d")).records()))
    for items in old["qa"].values():                                    # as written before the fields existed
        for item in items:
            for ev in _evidence_of(item):
                for key in EVIDENCE_KEYS_AMBIGUITY:
                    ev.pop(key)
    result = rematch_view(old, association="2d")
    by_object = result.matches["laser"].by_object()
    runner_up = by_object["chair-1"].runner_up_iou2d
    assert by_object["chair-1"].ambiguous and by_object["table-1"].ambiguous and not by_object["sofa-1"].ambiguous
    assert by_object["table-1"].runner_up_iou2d == runner_up and 0.3 <= runner_up < by_object["chair-1"].iou2d
    records = json.loads(to_json_line(result.records()))
    full = json.loads(to_json_line(process_view(view, objects, instances, {"laser": depth, "moge": depth * 1.2},
                                                visibility="laser").records()))
    assert json_close(records["qa"], full["qa"])                        # as the 2D pipeline writes them
    for source in ("auto_laser", "auto_moge"):
        assert records["qa"][source]
        for item in records["qa"][source]:
            for ev in _evidence_of(item):
                shared = ev["object_id"] in ("chair-1", "table-1")
                assert ev["match_ambiguous"] is shared
                assert ev["match_runner_up_iou2d"] == pytest.approx(runner_up if shared else 0.0)
    gated = json.loads(to_json_line(rematch_view(old, association="3d").records()))
    evidence = [ev for item in gated["qa"]["auto_laser"] for ev in _evidence_of(item)]
    assert evidence and all(ev["match_ambiguous"] is None and ev["match_runner_up_iou2d"] is None
                            for ev in evidence)


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
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_exclude_ambiguous_drops_exactly_the_ambiguous_items_and_detections(
        overlap: tuple[FrameView, list[SceneObject], Array, list[Instance]], tmp_path: Path) -> None:
    """On the overlapping view, the audit drops exactly the items naming the chair or the table (counts too), and
    the fusion evaluation drops exactly their two detections; the sofa's survive.

    Parameters
    ----------
    overlap : tuple
        The overlapping view.
    tmp_path : pathlib.Path
        Scratch folder from pytest.
    """
    view, objects, depth, instances = overlap
    result = process_view(view, objects, instances, {"laser": depth}, visibility="laser")
    pairs = pair_table(result.items["oracle"], result.items["auto_laser"], "laser")
    kept, counts = exclude_ambiguous(pairs)
    involved = {i.id for i in result.items["auto_laser"]
                if {"chair-1", "table-1"} & {ev["object_id"] for ev in _evidence_of(i.to_json())}}
    assert involved and set(pairs["id"]) - set(kept["id"]) == involved
    assert counts[view.dataset]["laser"]["n_dropped"] == len(involved)
    assert counts[view.dataset]["laser"]["n_unknown_kept"] == 0

    records = result.records()
    (tmp_path / "views.jsonl").write_text(to_json_line(records["view"]) + "\n")
    (tmp_path / "detections.jsonl").write_text("".join(to_json_line(r) + "\n" for r in records["detections"]))
    fusion = _script("run_fusion_eval")
    detections, _ = load_detections(tmp_path)
    ambiguous, unknown = fusion.pairing_ambiguity(tmp_path)
    kept_detections, record = fusion.drop_ambiguous(detections, ambiguous, unknown)
    assert {d.object_id for d in detections} == {"chair-1", "table-1", "sofa-1"}
    assert [d.object_id for d in kept_detections] == ["sofa-1"] and not unknown
    assert record["counts"] == {"laser": {"n_matched": 3, "n_dropped": 2, "n_unknown_kept": 0}}


def test_config_json_round_trip_and_legacy_association() -> None:
    """Settings survive JSON; a stored config from before the 2D association reads as the 3D association."""
    config = PipelineConfig(lift_stride=3, association="2d", match_2d=Match2DConfig(0.4))
    assert PipelineConfig.from_json(json.loads(json.dumps(config.to_json()))) == config
    legacy = config.to_json()
    del legacy["association"], legacy["match_2d"]
    assert PipelineConfig.from_json(legacy).association == "3d"
    with pytest.raises(ValueError):
        PipelineConfig(association="nearest")
    assert visible_box_edges((10.0, 20.0, 30.0, 40.0), GT_BOX2D) == (10.0, 20.0, 31.0, 41.0)
    assert visible_box_edges((10.0, 20.0, 30.0, 40.0), "laser") == (10.5, 20.5, 30.5, 40.5)


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
