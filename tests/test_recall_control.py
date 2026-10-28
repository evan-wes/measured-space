"""Tests for certified recall by conformal risk control (:mod:`measured_space.recall_control`) and its script.

Everything is synthetic and runs on the CPU in a few seconds:

- the CRC threshold on a four-scene example computed by hand;
- the loss's monotonicity in the threshold, and coverage against one-to-one pairing;
- the leave-one-scene-out bookkeeping against direct calibration on the other scenes;
- the guarantee itself on simulated scenes (mean held-out FNR at most alpha, and near it);
- the cost metrics on hand-built views;
- the record path (a stored view's present objects, 2D and projected boxes) and the script end to end, on a
  small label folder with CuTR-format predictions written here.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from measured_space.geometry import Camera, OrientedBox, invert_pose, make_pose
from measured_space.matching import box_iou_2d, match_detections_2d
from measured_space.pipeline import PipelineConfig, box_json, projected_extent, view_from_record
from measured_space.recall_control import (
    RecallConfig,
    ScoredBox,
    ViewEvidence,
    cap_floor,
    cap_free_floor,
    coverage_scores,
    crc_index,
    crc_threshold,
    leave_one_scene_out,
    loss_matrix,
    one_to_one_found,
    pooled_cost,
    range_bin,
    run_recall_control,
    scene_table_at,
    share_truncated,
    simulate_guarantee,
    synthetic_coverages,
    truncation_threshold,
    view_evidence,
    visible_bin,
)
from measured_space.scene import SceneObject

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
CAM = Camera(500.0, 500.0, 319.5, 239.5, 640, 480)
LOOK_ALONG_Y = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])   # columns: camera x, y, z in the world
HAND = {"A": [0.9, 0.5], "B": [0.8, 0.4], "C": [0.7, 0.6], "D": [0.95, 0.2]}


def _view(scene: str, scores: Sequence[float], iou: Any, clear: Sequence[bool] | None = None) -> ViewEvidence:
    """A hand-built view.

    Parameters
    ----------
    scene : str
        Scene name.
    scores : sequence of float
        Box scores (k).
    iou : array_like, shape (k, m)
        Box-object IoU (pass ``np.zeros((0, m))`` for a view without boxes).
    clear : sequence of bool, optional
        Clearly-visible flags (default all True).

    Returns
    -------
    ViewEvidence
        The view, with ``m`` objects at 5 m and visible fraction 0.5.
    """
    s = np.asarray(scores, dtype=np.float64)
    overlap = np.asarray(iou, dtype=np.float64)
    m = overlap.shape[1]
    return ViewEvidence("d", scene, f"d/{scene}/c/v", s, overlap,
                        tuple(f"o{i}" for i in range(m)), ("chair",) * m, np.full(m, 5.0), np.full(m, 0.5),
                        np.asarray(clear if clear is not None else [True] * m, dtype=bool))


# --------------------------------------------------------------------------------------------------------
# CRC formula, loss, bookkeeping, guarantee
# --------------------------------------------------------------------------------------------------------


def test_crc_threshold_matches_a_hand_computed_example() -> None:
    """Four scenes with two objects each; floor 0.1. Mean calibration FNR at the candidates
    0.1, 0.2, 0.4, 0.5, 0.6 is 0, 0, 1/8, 1/4, 3/8. CRC needs 4/5 R + 1/5 <= alpha:

    - alpha = 0.40: R <= 1/4, met with equality at 0.5 (the bound is inclusive), so t_hat = 0.5;
    - alpha = 0.39: R <= 0.2375, so t_hat = 0.4;
    - alpha = 0.15: 1/5 > alpha, nothing qualifies, so t_hat is the floor and infeasible."""
    coverages = list(HAND.values())
    result = crc_threshold(coverages, 0.40, floor=0.1)
    assert (result.threshold, result.feasible, result.n_cal) == (0.5, True, 4)
    assert result.cal_risk == pytest.approx(0.25) and result.adjusted_risk == pytest.approx(0.40)
    assert crc_threshold(coverages, 0.39, floor=0.1).threshold == 0.4
    infeasible = crc_threshold(coverages, 0.15, floor=0.1)
    assert (infeasible.threshold, infeasible.feasible, infeasible.floor_risk) == (0.1, False, 0.0)
    # The index rule on summed losses: leading run of sum + B <= alpha (n + 1).
    assert crc_index([0.0, 0.5, 1.0, 1.5], 4, 0.4) == 2
    assert crc_index([1.5, 0.0], 4, 0.4) == -1
    # A scene with an object no box covers raises the floor risk.
    assert crc_threshold([[0.9, -np.inf], [0.8, 0.7]], 0.5, floor=0.05).floor_risk == pytest.approx(0.25)


def test_loss_is_non_decreasing_and_left_continuous_in_the_threshold() -> None:
    """Coverage FNR on random views: raising the threshold never lowers it, a box scoring exactly the threshold
    is kept, and adding boxes never raises it."""
    rng = np.random.default_rng(3)
    grid = np.linspace(0.0, 1.0, 201)
    for _ in range(50):
        k, m = rng.integers(0, 8), rng.integers(1, 6)
        scores, iou = rng.random(k), rng.random((k, m)) * (rng.random((k, m)) < 0.5)
        cov = coverage_scores(scores, iou, 0.3)
        losses = loss_matrix([cov], grid)[0]
        assert np.all(np.diff(losses) >= 0)
        for s in scores:
            assert loss_matrix([cov], [s])[0, 0] == pytest.approx(np.mean(cov < s))
        more = coverage_scores(np.concatenate([scores, rng.random(3)]),
                               np.vstack([iou, rng.random((3, m))]), 0.3)
        assert np.all(loss_matrix([more], grid)[0] <= losses + 1e-12)


def test_coverage_counts_any_box_while_one_to_one_pairs_each_box_once() -> None:
    """One box over two objects covers both (IoU >= 0.3) but pairs with only one; IoU 0.29 covers nothing."""
    scores, iou = np.array([0.8, 0.6]), np.array([[0.5, 0.4], [0.29, 0.0]])
    assert coverage_scores(scores, iou, 0.3).tolist() == [0.8, 0.8]
    assert one_to_one_found(scores, iou, 0.5, 0.3).sum() == 1
    assert one_to_one_found(scores, iou, 0.9, 0.3).sum() == 0
    assert coverage_scores(np.zeros(0), np.zeros((0, 2)), 0.3).tolist() == [-np.inf, -np.inf]


def test_one_to_one_pairing_matches_the_pipeline_matcher() -> None:
    """The one-to-one pairing equals match_detections_2d's on random boxes."""
    rng = np.random.default_rng(5)
    for _ in range(30):
        k, m = rng.integers(1, 7), rng.integers(1, 6)
        det = rng.uniform(0, 100, (k, 2)), rng.uniform(10, 60, (k, 2))
        obj = rng.uniform(0, 100, (m, 2)), rng.uniform(10, 60, (m, 2))
        det_boxes = np.hstack([det[0], det[0] + det[1]])
        obj_boxes = np.hstack([obj[0], obj[0] + obj[1]])
        unit = OrientedBox([0, 0, 0], [1, 1, 1], np.eye(3))
        objects = [SceneObject(f"o{j}", "chair", unit) for j in range(m)]
        expected = match_detections_2d(list(det_boxes), [unit] * k, [("chair",)] * k, objects, list(obj_boxes))
        found = one_to_one_found(np.ones(k), box_iou_2d(det_boxes, obj_boxes), 0.5, 0.3)
        assert set(np.flatnonzero(found)) == {int(mt.object_id[1:]) for mt in expected.matches}


def test_leave_one_scene_out_calibrates_each_fold_on_the_other_scenes() -> None:
    """Each held-out scene's threshold is CRC on the remaining scenes, and its FNR is its own loss there."""
    rng = np.random.default_rng(7)
    coverages = {f"s{i}": np.where(rng.random(n) < 0.05, -np.inf, rng.random(n))
                 for i, n in enumerate(rng.integers(1, 30, 12))}
    for alpha in (0.1, 0.2, 0.35):
        loso = leave_one_scene_out(coverages, alpha, floor=0.0)
        assert loso["scene"].tolist() == list(coverages) and (loso["n_cal"] == 11).all()
        for _, row in loso.iterrows():
            others = [c for s, c in coverages.items() if s != row["scene"]]
            direct = crc_threshold(others, alpha, floor=0.0)
            own = coverages[row["scene"]]
            assert (row["threshold"], row["feasible"]) == (direct.threshold, direct.feasible)
            assert row["cal_risk"] == pytest.approx(direct.cal_risk)
            assert row["heldout_fnr"] == pytest.approx(np.mean(own < row["threshold"]))
            assert row["floor_fnr"] == pytest.approx(np.mean(own < 0.0))
            assert row["n_objects"] == len(own)
    with pytest.raises(ValueError):
        leave_one_scene_out({"only": [0.5]}, 0.1, floor=0.0)


@pytest.mark.parametrize("n_cal, alpha", [(19, 0.10), (39, 0.20), (59, 0.05)])
def test_crc_guarantee_holds_on_fresh_simulated_scenes(n_cal: int, alpha: float) -> None:
    """Calibrated on heterogeneous scenes (every object coverable at the floor, as Theorem 1 assumes) and
    measured on fresh scenes, the mean FNR over draws is at most alpha and above Theorem 2's lower bound, while
    single scenes do exceed alpha.

    Parameters
    ----------
    n_cal : int
        Calibration scenes per draw.
    alpha : float
        Target.
    """
    out = simulate_guarantee(n_cal, alpha, 300, np.random.default_rng(11), n_test=40)
    assert out["share_feasible"] == 1.0
    assert out["mean_fnr"] <= alpha + 3 * out["se"]
    assert out["mean_fnr"] >= out["lower_bound"] - 3 * out["se"]
    assert out["share_scenes_above_alpha"] > 0.05                     # marginal, not per scene


def test_an_uncorrected_threshold_breaks_the_guarantee() -> None:
    """The same simulation with the plain empirical rule (mean calibration FNR <= alpha, no n/(n+1) and B/(n+1)
    terms) exceeds alpha on fresh scenes, so the guarantee test has power."""
    rng = np.random.default_rng(23)
    alpha, n_cal, fresh = 0.1, 10, []
    for _ in range(400):
        cal = list(synthetic_coverages(n_cal, rng).values())
        grid = np.unique(np.concatenate([[0.05], *cal]))
        risk = loss_matrix(cal, grid).mean(axis=0)
        t = grid[np.flatnonzero(risk <= alpha).max()]
        fresh.append(loss_matrix(list(synthetic_coverages(40, rng).values()), [t])[:, 0].mean())
    assert np.mean(fresh) > alpha + 0.01


def test_leave_one_scene_out_mean_is_at_most_alpha_by_construction() -> None:
    """Whenever the mean floor over all scenes is at most alpha, the mean held-out FNR is too, for any data (the
    module docstring, step 5); it can exceed alpha otherwise."""
    rng = np.random.default_rng(29)
    exceeded_with_high_floor = False
    for _ in range(400):
        coverages = synthetic_coverages(int(rng.integers(2, 30)), rng, mean_objects=float(rng.uniform(1, 30)),
                                        never_found=float(rng.choice([0.0, 0.05, 0.3])))
        alpha = float(rng.uniform(0.02, 0.6))
        loso = leave_one_scene_out(coverages, alpha, floor=0.05)
        if loso["floor_fnr"].mean() <= alpha:
            assert loso["heldout_fnr"].mean() <= alpha + 1e-12
        else:
            exceeded_with_high_floor |= bool(loso["heldout_fnr"].mean() > alpha)
    assert exceeded_with_high_floor


def test_crc_is_infeasible_with_too_few_scenes_and_falls_back_to_the_floor() -> None:
    """With 5 calibration scenes B / (n + 1) = 1/6 > 0.1, so CRC keeps every box and is flagged."""
    out = simulate_guarantee(5, 0.1, 20, np.random.default_rng(2), floor=0.05)
    assert out["share_feasible"] == 0.0 and out["mean_threshold"] == pytest.approx(0.05)
    assert out["mean_fnr"] == 0.0                                     # every object covered at the floor here


# --------------------------------------------------------------------------------------------------------
# Cost, strata and the whole analysis
# --------------------------------------------------------------------------------------------------------


def test_cost_metrics_count_kept_boxes_per_view_and_those_unmatched_to_annotation() -> None:
    """Scene "a": view 1 has boxes 0.9 (on object 0), 0.5 (on nothing), 0.2 (on object 1); view 2 has no boxes and
    one object. At t = 0.4: 2 kept boxes over 2 views, 1 of them unmatched; 2 of 3 objects missed."""
    views = [_view("a", [0.9, 0.5, 0.2], [[0.8, 0.0], [0.1, 0.0], [0.0, 0.6]]), _view("a", [], np.zeros((0, 1)))]
    at_04 = scene_table_at(views, {"a": 0.4}, "present", 0.3)
    at_01 = scene_table_at(views, {"a": 0.1}, "present", 0.3)
    table = at_04.iloc[0]
    assert (table["n_views"], table["n_objects"], table["n_kept"], table["n_unmatched"]) == (2, 3, 2, 1)
    assert table["boxes_per_view"] == 1.0 and table["unmatched_share"] == 0.5
    assert table["fnr"] == pytest.approx(2 / 3) and table["fnr_one_to_one"] == pytest.approx(2 / 3)
    low = at_01.iloc[0]
    assert (low["boxes_per_view"], low["unmatched_share"], low["fnr"]) == (1.5, pytest.approx(1 / 3),
                                                                          pytest.approx(1 / 3))
    clear = scene_table_at([_view("a", [0.9], [[0.8, 0.0]], [True, False])], {"a": 0.5}, "clearly_visible", 0.3)
    assert clear.iloc[0]["n_objects"] == 1 and clear.iloc[0]["fnr"] == 0.0
    pooled = pooled_cost(pd.concat([at_04, at_01]))
    assert pooled["boxes_per_view"] == pytest.approx(5 / 4) and pooled["unmatched_share"] == pytest.approx(2 / 5)


def test_bins_follow_the_stated_edges() -> None:
    """Range bins as compare_association; visible-fraction bins with 1 in the last."""
    assert [range_bin(d) for d in (0.0, 9.99, 10.0, 44.9, 45.0, 300.0)] == [
        "0_10m", "0_10m", "10_20m", "20_45m", "45m_plus", "45m_plus"]
    assert [visible_bin(f) for f in (0.0, 0.05, 0.1, 0.3, 0.59, 0.6, 1.0)] == [
        "0-0.1", "0-0.1", "0.1-0.3", "0.3-0.6", "0.3-0.6", "0.6-1", "0.6-1"]


def test_run_recall_control_reports_consistent_tables() -> None:
    """On many synthetic scenes the analysis is feasible, its held-out FNRs agree with the per-view recount, the
    conditional table covers every object, and the published threshold's cost is reported."""
    rng = np.random.default_rng(13)
    views = []
    for s in range(25):
        for _ in range(3):
            m, k = rng.integers(1, 5), rng.integers(0, 6)
            iou = np.where(rng.random((k, m)) < 0.4, rng.uniform(0.3, 0.9, (k, m)), 0.0)
            views.append(_view(f"s{s:02d}", rng.uniform(0.05, 1.0, k), iou))
    result = run_recall_control(views, RecallConfig(alphas=(0.3,), mondrian=("range",), n_boot=200), floor=0.05)
    row = result.tables["by_alpha"].iloc[0]
    assert row["n_scenes"] == 25 and row["n_cal"] == 24
    assert row["boxes_per_view_published"] > 0 and 0 <= row["unmatched_share_published"] <= 1
    assert row["mean_heldout_fnr_lo"] <= row["mean_heldout_fnr"] <= row["mean_heldout_fnr_hi"]
    assert result.summary["checks"]["alpha_0.3_max_fnr_gap"] == 0.0
    conditional = result.tables["conditional"]
    assert conditional[conditional["stratum"] == "range"]["n_objects"].sum() == len(result.tables["objects"])
    assert set(result.tables["mondrian"]["bin"]) == {"0_10m"}
    with pytest.raises(ValueError):
        run_recall_control(views, RecallConfig(), floor=0.5)            # boxes below the floor


def test_box_cap_floor_and_truncated_share_follow_their_definitions() -> None:
    """A view with as many boxes as the cap is truncated up to its lowest score; the cap-free floor lies just
    above the largest such score, and the observation floor is kept when no view reaches it."""
    assert cap_floor([0.9, 0.3, 0.6], 3) == 0.3
    assert cap_floor([0.9, 0.3], 3) == -math.inf and cap_floor([0.9, 0.3], None) == -math.inf
    views = [replace(_view("a", [0.9], [[0.5]]), cap_floor=0.2), replace(_view("a", [0.9], [[0.5]]), cap_floor=0.1),
             _view("b", [0.9], [[0.5]])]
    assert truncation_threshold(views) == 0.2
    assert share_truncated(views, -math.inf) == pytest.approx(2 / 3)
    assert share_truncated(views, 0.1) == pytest.approx(2 / 3) and share_truncated(views, 0.15) == pytest.approx(1 / 3)
    assert share_truncated(views, cap_free_floor(0.05, 0.2)) == 0.0
    assert 0.2 < cap_free_floor(0.05, 0.2) < 0.2 + 1e-12
    assert cap_free_floor(0.05, -math.inf) == 0.05 and cap_free_floor(0.05, 0.04) == 0.05


def test_crc_floor_keeps_thresholds_out_of_the_truncated_region_or_flags_them() -> None:
    """With the cap-free floor no fold or deployable threshold lies at or below t_cap; with the observation floor
    the folds that do are flagged, and the curve marks its truncated part."""
    rng = np.random.default_rng(31)
    views = []
    for s in range(30):
        for _ in range(3):
            m, k = rng.integers(1, 5), rng.integers(1, 6)
            iou = np.where(rng.random((k, m)) < 0.5, rng.uniform(0.3, 0.9, (k, m)), 0.0)
            scores = rng.uniform(0.05, 1.0, k)
            capped = rng.random() < 0.3
            views.append(replace(_view(f"s{s:02d}", scores, iou), cap_floor=float(scores.min()) if capped else -np.inf))
    t_cap = truncation_threshold(views)
    config = RecallConfig(alphas=(0.1, 0.3), n_boot=100)
    strict = run_recall_control(views, config, floor=0.05, crc_floor=cap_free_floor(0.05, t_cap))
    assert (strict.tables["loso"]["threshold"] > t_cap).all()
    assert not strict.tables["loso"]["in_truncated_region"].any()
    assert (strict.tables["loso"]["share_views_truncated"] == 0).all()
    assert not strict.tables["by_alpha"]["deploy_in_truncated_region"].any()
    assert strict.summary["truncation"]["t_cap"] == t_cap and strict.summary["truncation"]["crc_floor_above_t_cap"]
    marked = run_recall_control(views, config, floor=0.05)
    loso = marked.tables["loso"]
    assert (loso["in_truncated_region"] == (loso["threshold"] <= t_cap)).all() and loso["in_truncated_region"].any()
    curve = marked.tables["curve"]
    assert (curve["fully_observed"] == (curve["threshold"] > t_cap)).all()
    assert curve["share_views_truncated"].iloc[0] == pytest.approx(share_truncated(views, 0.05))
    with pytest.raises(ValueError):
        run_recall_control(views, config, floor=0.05, crc_floor=0.01)


# --------------------------------------------------------------------------------------------------------
# Records and the script
# --------------------------------------------------------------------------------------------------------


def _records(scene: str, frame: str, objects: Sequence[tuple[str, float, float, float | None]]) -> dict[str, Any]:
    """Shard records of a view looking along world +y from the origin.

    Parameters
    ----------
    scene, frame : str
        Identity.
    objects : sequence of tuple
        ``(object_id, x, visible_fraction, None or 0)``: a 1 m cube at ``(x, 4, 0)``; the last entry ``None``
        makes the object not present (no visible box), else its visible box is its projected extent.

    Returns
    -------
    dict
        ``{"view": ..., "oracle_objects": [...]}`` as ``scripts/run_labels.py`` stores them.
    """
    T_wc = make_pose(LOOK_ALONG_Y, [0.0, 0.0, 0.0])
    view_record = {"view_key": f"arkitscenes/{scene}/wide/{frame}", "dataset": "arkitscenes", "scene": scene,
                   "camera_id": "wide", "frame": frame, "image_ref": "none",
                   "camera": {"fx": CAM.fx, "fy": CAM.fy, "cx": CAM.cx, "cy": CAM.cy, "width": CAM.width,
                              "height": CAM.height},
                   "T_wc": T_wc.tolist(), "up": [0.0, 0.0, 1.0], "visibility": "laser",
                   "config": PipelineConfig().to_json()}
    view = view_from_record(view_record)
    oracle = []
    for object_id, x, fraction, flag in objects:
        box = OrientedBox([x, 4.0, 0.0], [1.0, 1.0, 1.0], np.eye(3))
        extent = projected_extent(view, box)
        bbox = None if flag is None or extent is None else [max(extent[0], 0.0), max(extent[1], 0.0),
                                                            min(extent[2], 639.0), min(extent[3], 479.0)]
        fraction = fraction if bbox is not None else 0.0
        oracle.append({"view_key": view_record["view_key"], "object_id": object_id, "label": "chair",
                       "box": box_json(box), "visible_fraction": fraction, "truncated_fraction": 0.0,
                       "bbox2d": bbox, "present": bbox is not None and fraction > 0.0,
                       "clearly_visible": bbox is not None and fraction >= 0.3, "visibility": "laser",
                       "diagnostics": {}, "match": None})
    return {"view": view_record, "oracle_objects": oracle}


def test_view_evidence_rebuilds_present_objects_and_pairs_boxes() -> None:
    """Absent objects are left out; a box on the object's visible box covers it; the projected-box option uses
    the world box; a tampered presence flag is refused."""
    records = _records("s", "1", [("a", -1.0, 0.8, 0), ("b", 1.0, 0.2, 0), ("gone", 0.0, 0.5, None)])
    view = view_from_record(records["view"])
    a_box = OrientedBox([-1.0, 4.0, 0.0], [1.0, 1.0, 1.0], np.eye(3))
    x0, y0, x1, y1 = (v + 0.5 for v in projected_extent(view, a_box))                    # type: ignore[union-attr]
    boxes = [ScoredBox(0.7, (x0, y0, x1, y1), a_box), ScoredBox(0.4, None, None), ScoredBox(0.3, (0, 0, 0, 0))]
    ev = view_evidence(records, boxes)
    assert ev.object_ids == ("a", "b") and ev.n_unusable == 2 and ev.scores.tolist() == [0.7]
    assert ev.iou[0, 0] > 0.9 and ev.iou[0, 1] == 0.0
    assert ev.clearly_visible.tolist() == [True, False]
    assert ev.range_m[0] == pytest.approx(math.hypot(1.0, 4.0))
    projected = view_evidence(records, [ScoredBox(0.7, None, a_box)], project_3d=True)
    assert projected.n_unusable == 0 and projected.iou[0, 0] > 0.9
    behind = OrientedBox([0.0, -3.0, 0.0], [1.0, 1.0, 1.0], np.eye(3))
    assert view_evidence(records, [ScoredBox(0.7, None, behind)], project_3d=True).n_unusable == 1
    records["oracle_objects"][2]["present"] = True
    with pytest.raises(ValueError):
        view_evidence(records, boxes)


def _script() -> ModuleType:
    """Import ``scripts/run_recall_control.py`` as a module.

    Returns
    -------
    types.ModuleType
        The script's module; importing it has no other side effects.
    """
    spec = importlib.util.spec_from_file_location("run_recall_control", SCRIPTS / "run_recall_control.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_recall_control"] = module
    spec.loader.exec_module(module)
    return module


def _write_inputs(root: Path, n_scenes: int, rng: np.random.Generator) -> tuple[Path, Path]:
    """A label folder of ``n_scenes`` two-view scenes and CuTR-format predictions for it.

    Parameters
    ----------
    root : pathlib.Path
        Parent folder.
    n_scenes : int
        Scenes.
    rng : numpy.random.Generator
        Random source for scores.

    Returns
    -------
    labels, predictions : pathlib.Path
        The two folders.
    """
    labels, predictions = root / "labels", root / "predictions"
    (labels / "arkitscenes" / "shards").mkdir(parents=True)
    (predictions / "arkitscenes").mkdir(parents=True)
    for s in range(n_scenes):
        scene = f"{4000 + s}"
        lines = []
        for frame in ("1", "2"):
            records = _records(scene, frame, [("a", -1.0, 0.8, 0), ("b", 1.0, 0.5, 0)])
            (labels / "arkitscenes" / "shards" / f"{scene}__wide__{frame}.json").write_text(
                json.dumps({"records": records}))
            view = view_from_record(records["view"])
            T_cw = invert_pose(view.T_wc)
            boxes = []
            for o in records["oracle_objects"]:
                center = np.asarray(o["box"]["center"])
                boxes.append({**o["box"], "score": float(rng.uniform(0.05, 1.0)), "box2d": o["bbox2d"],
                              "center_cam": (T_cw[:3, :3] @ center + T_cw[:3, 3]).tolist()})
            lines.append({"view_key": records["view"]["view_key"], "model": "fake", "score_threshold": 0.05,
                          "image_size": [CAM.width, CAM.height], "boxes": boxes})
        (predictions / "arkitscenes" / f"{scene}.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    return labels, predictions


def test_script_writes_per_item_and_aggregate_outputs_and_refuses_non_empty_folders(tmp_path: Path) -> None:
    """The script reads shards and CuTR lines, writes every table, the summary and the figure, and refuses to
    write into a folder that holds anything or lies inside an input folder.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary folder.
    """
    script = _script()
    labels, predictions = _write_inputs(tmp_path, 12, np.random.default_rng(17))
    results, out_root = tmp_path / "results", tmp_path / "per_item"
    argv = ["--labels", str(labels), "--predictions", str(predictions), "--name", "run1", "--results",
            str(results), "--out-root", str(out_root), "--alphas", "0.1", "0.3", "--n-boot", "100",
            "--mondrian", "visible_fraction"]
    assert script.main(argv) == 0
    for name in ("objects", "boxes", "views"):
        assert (out_root / "run1" / f"{name}.csv").is_file()
    for name in ("by_alpha", "loso", "conditional", "curve", "curve_by_scene", "mondrian"):
        assert (results / f"{name}.csv").is_file()
    assert (results / "fnr_vs_threshold_arkitscenes.png").is_file()
    summary = json.loads((results / "summary.json").read_text())
    assert summary["inputs"]["arkitscenes"]["floor"] == 0.05
    assert len(summary["inputs"]["arkitscenes"]["prediction_files"]) == 12
    by_alpha = pd.read_csv(results / "by_alpha.csv")
    assert by_alpha["n_scenes"].tolist() == [12, 12]
    assert pd.read_csv(out_root / "run1" / "objects.csv").shape[0] == 12 * 2 * 2
    assert script.main(argv) == 1                                     # both folders now hold files
    inside = ["--results", str(labels / "aggregates"), "--name", "run2"]
    assert script.main(argv[:4] + inside + argv[8:]) == 1             # results inside an input folder
    assert not (labels / "aggregates").exists() and not (out_root / "run2").exists()


def test_script_reads_general_lines_and_refuses_missing_predictions(tmp_path: Path) -> None:
    """A line with only scores and 2D boxes is read (with --floor, as it states no threshold); a view without a
    line stops the run unless --skip-missing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary folder.
    """
    script = _script()
    labels, predictions = _write_inputs(tmp_path, 6, np.random.default_rng(19))
    for path in (predictions / "arkitscenes").glob("*.jsonl"):
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        general = [{"view_key": r["view_key"], "boxes": [{"score": b["score"], "box2d": b["box2d"]}
                                                          for b in r["boxes"]]} for r in lines]
        path.write_text("\n".join(json.dumps(r) for r in general[:1]) + "\n")      # second view missing
    base = ["--labels", str(labels), "--predictions", str(predictions), "--out-root", str(tmp_path / "items"),
            "--alphas", "0.3", "--n-boot", "50"]
    assert script.main(base + ["--name", "a", "--results", str(tmp_path / "r1"), "--floor", "0.05"]) == 1
    assert script.main(base + ["--name", "b", "--results", str(tmp_path / "r2"), "--skip-missing"]) == 1  # no floor
    assert script.main(base + ["--name", "c", "--results", str(tmp_path / "r3"), "--floor", "0.05",
                               "--skip-missing"]) == 0
    summary = json.loads((tmp_path / "r3" / "summary.json").read_text())
    assert summary["inputs"]["arkitscenes"]["n_views_skipped"] == 6


def test_script_applies_the_box_cap(tmp_path: Path) -> None:
    """Every view writes two boxes, so with --max-boxes 2 every view is capped: by default CRC chooses only above
    t_cap; with --truncation mark the folds in the truncated region are flagged.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary folder.
    """
    script = _script()
    labels, predictions = _write_inputs(tmp_path, 12, np.random.default_rng(37))
    base = ["--labels", str(labels), "--predictions", str(predictions), "--out-root", str(tmp_path / "items"),
            "--alphas", "0.3", "--n-boot", "50", "--max-boxes", "2"]
    assert script.main(base + ["--name", "floor", "--results", str(tmp_path / "floor")]) == 0
    summary = json.loads((tmp_path / "floor" / "summary.json").read_text())
    truncation = summary["datasets"]["arkitscenes"]["truncation"]
    assert truncation["n_views_capped"] == 24 and truncation["share_views_capped"] == 1.0
    loso = pd.read_csv(tmp_path / "floor" / "loso.csv")
    assert (loso["threshold"] > truncation["t_cap"]).all() and not loso["in_truncated_region"].any()
    assert summary["inputs"]["arkitscenes"]["crc_floor"] > truncation["t_cap"]
    assert script.main(base + ["--name", "mark", "--results", str(tmp_path / "mark"), "--truncation", "mark"]) == 0
    marked = pd.read_csv(tmp_path / "mark" / "loso.csv")
    assert (marked["in_truncated_region"] == (marked["threshold"] <= truncation["t_cap"])).all()
    views = pd.read_csv(tmp_path / "items" / "mark" / "views.csv")
    assert np.isfinite(views["cap_floor"]).all()
