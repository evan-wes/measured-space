"""Tests for ``scripts/eval_sparse4d.py``: the GT rows, the HOTA data preparation and similarities, HOTA on tiny
cases with hand-checked answers, NVIDIA's evaluator on a perfect prediction, and the 12-scene aggregation.

Hand-checked HOTA (TrackEval's 19 thresholds alpha = 0.05 ... 0.95):

* one object, two frames, the prediction shifted by half its width (footprint IoU 1/3): matched at the 6
  thresholds up to 0.30, so DetA = AssA = HOTA = 6/19 and LocA = (6 x 1/3) / 19 = 2/19;
* one object, two frames, a perfect box whose id changes between the frames: DetA = 1, AssA = 1/2, HOTA = sqrt(1/2);
* a perfect box of the wrong class: an FN for one class and an FP for the other, so the pooled DetA is 0.

``gt``, the data preparation and the weighted means need only numpy. The rest skips cleanly where its dependency
is missing: shapely (BEV IoU), pytorch3d (3D IoU) and NVIDIA's ``spatialai_data_utils`` (TrackEval HOTA and the
official evaluator; put it on ``PYTHONPATH`` or set ``SPATIALAI_DATA_UTILS`` to its folder). Everything is written
under ``tmp_path``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_sparse4d.py"
if os.environ.get("SPATIALAI_DATA_UTILS") and os.environ["SPATIALAI_DATA_UTILS"] not in sys.path:
    sys.path.insert(0, os.environ["SPATIALAI_DATA_UTILS"])


def _script() -> ModuleType:
    """Import ``scripts/eval_sparse4d.py`` as a module (it is not part of the package).

    Returns
    -------
    types.ModuleType
        The module; importing it loads numpy only.
    """
    if "eval_sparse4d" in sys.modules:
        return sys.modules["eval_sparse4d"]
    spec = importlib.util.spec_from_file_location("eval_sparse4d", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_sparse4d"] = module
    spec.loader.exec_module(module)
    return module


def _has(module: str) -> bool:
    """Whether a module imports.

    Parameters
    ----------
    module : str
        Dotted module name.

    Returns
    -------
    bool
        True if ``import module`` succeeds.
    """
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


HAS_HOTA = _has("spatialai_data_utils.eval.tracking.hota.metrics.hota")
HAS_SHAPELY = _has("shapely")
HAS_PYTORCH3D = _has("pytorch3d.ops")
needs_hota = pytest.mark.skipif(not HAS_HOTA, reason="spatialai_data_utils (NVIDIA evaluator) not importable")
needs_shapely = pytest.mark.skipif(not HAS_SHAPELY, reason="shapely not installed")
needs_pytorch3d = pytest.mark.skipif(not HAS_PYTORCH3D, reason="pytorch3d not installed")


def box(x: float, y: float, w: float = 1.0, l: float = 1.0, yaw: float = 0.0) -> np.ndarray:
    """A ``[x, y, z, w, l, h, yaw]`` box with z = 0.5 and h = 1.

    Parameters
    ----------
    x, y : float
        Centre, metres.
    w, l : float, optional
        Footprint sides, metres.
    yaw : float, optional
        Radians.

    Returns
    -------
    numpy.ndarray, shape (7,)
        The box.
    """
    return np.array([x, y, 0.5, w, l, 1.0, yaw])


def write_rows(path: Path, rows: list[tuple[int, int, int, int, float, float]]) -> Path:
    """Write 11-column rows ``scene class id frame x y`` (z 0.5, size 1 x 1 x 1, yaw 0).

    Parameters
    ----------
    path : pathlib.Path
        Output file.
    rows : list of tuple
        ``(scene_id, class_id, object_id, frame, x, y)``.

    Returns
    -------
    pathlib.Path
        ``path``.
    """
    path.write_text("".join(f"{s} {c} {i} {f} {x:.2f} {y:.2f} 0.50 1.00 1.00 1.00 0.00\n"
                            for s, c, i, f, x, y in rows))
    return path


# ---------------------------------------------------------------------------------------------------- gt


GROUND_TRUTH = {
    "0": [{"object type": "Person", "object id": 7, "3d location": [1.234, -5.678, 0.9],
           "3d bounding box scale": [0.5, 0.6, 1.8], "3d bounding box rotation": [0.0, 0.0, 1.5708]}],
    "6": [{"object type": "Forklift", "object id": 3, "3d location": [10.0, 20.0, 1.1],
           "3d bounding box scale": [1.2, 2.4, 2.2], "3d bounding box rotation": [0.1, 0.2, -3.14159]},
          {"object type": "Robot", "object id": 4, "3d location": [0, 0, 0],
           "3d bounding box scale": [1, 1, 1], "3d bounding box rotation": [0, 0, 0]}],
    "3": [{"object type": "Person", "object id": 8, "3d location": [0, 0, 0],
           "3d bounding box scale": [1, 1, 1], "3d bounding box rotation": [0, 0, 0]}],
}


def test_gt_rows_format_and_window(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Frames 0 and 6 of ``range(0, 12, 6)``: challenge class ids, 2 decimals, yaw from the third rotation angle;
    frame 3 is outside the window; an unknown type is counted, not written; an existing file is refused.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    capsys : pytest.CaptureFixture
        Captures the printed counts.
    """
    ev = _script()
    lines, counts = ev.gt_rows(GROUND_TRUTH, 22, range(0, 12, 6))
    assert lines == ["22 0 7 0 1.23 -5.68 0.90 0.50 0.60 1.80 1.57\n",
                     "22 1 3 6 10.00 20.00 1.10 1.20 2.40 2.20 -3.14\n"]
    assert counts == {"Person": 1, "Forklift": 1, "UNKNOWN:Robot": 1}
    scene = tmp_path / "Warehouse_022"
    scene.mkdir()
    (scene / "ground_truth.json").write_text(json.dumps(GROUND_TRUTH))
    out = tmp_path / "gt" / "Warehouse_022_f0-12s6.txt"
    assert ev.main(["gt", "--scene-dir", str(scene), "--scene-id", "22", "--end", "12", "--step", "6",
                    "--out", str(out)]) == 0
    assert out.read_text() == "".join(lines)
    assert "2 frames, 2 rows" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        ev.write_gt(scene, 22, [0], out)


def test_load_tracks_groups_by_scene_and_frame(tmp_path: Path) -> None:
    """Rows are grouped by the scene column (as written) and the frame, in file order; blank lines are skipped.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    """
    ev = _script()
    path = tmp_path / "rows.txt"
    path.write_text("1 0 5 0 1 2 3 4 5 6 0.5\n\n2 3 9 6 0 0 0 1 1 1 0\n1 1 4 0 0 0 0 1 1 1 0\n")
    tracks = ev.load_tracks(path)
    assert sorted(tracks) == ["1", "2"]
    assert [(c, i) for c, i, _ in tracks["1"][0]] == [(0, 5), (1, 4)]
    np.testing.assert_array_equal(tracks["1"][0][0][2], [1, 2, 3, 4, 5, 6, 0.5])
    assert list(tracks["2"]) == [6]


def test_seq_data_filters_the_class_and_relabels_ids() -> None:
    """Only class 0 is kept; GT ids 40 and 7 become 1 and 0, tracker id 100 becomes 0; empty timesteps get empty
    arrays and an empty similarity matrix.
    """
    ev = _script()
    gt = {0: [(0, 40, box(0, 0)), (0, 7, box(5, 0)), (1, 3, box(9, 9))], 2: [(0, 7, box(5, 0))]}
    pr = {0: [(0, 100, box(0, 0)), (2, 5, box(0, 0))]}
    data = ev.seq_data(gt, pr, 0, 3, lambda g, t: np.full((len(g), len(t)), 0.5))
    assert [a.tolist() for a in data["gt_ids"]] == [[1, 0], [], [0]]
    assert [a.tolist() for a in data["tracker_ids"]] == [[0], [], []]
    assert data["similarity_scores"][0].shape == (2, 1) and data["similarity_scores"][1].shape == (0, 0)
    assert data["similarity_scores"][2].shape == (1, 0)
    assert (data["num_gt_dets"], data["num_tracker_dets"], data["num_gt_ids"], data["num_tracker_ids"]) == (3, 1, 2, 1)
    assert data["num_timesteps"] == 3


def test_weighted_mean_uses_gt_rows() -> None:
    """Scenes with 100 and 300 GT rows and HOTA 40 and 80 average to 70; a missing metric makes the mean NaN."""
    ev = _script()
    per_scene = {"a": {"n_gt": 100, "official": {"HOTA": 40.0, "DetA": 50.0, "AssA": 30.0, "LocA": 80.0}},
                 "b": {"n_gt": 300, "official": {"HOTA": 80.0, "DetA": 70.0, "AssA": 90.0}}}
    mean = ev.weighted_mean(per_scene, ["a", "b"], "official")
    assert mean["HOTA"] == pytest.approx(70.0) and mean["DetA"] == pytest.approx(65.0)
    assert mean["AssA"] == pytest.approx(75.0) and math.isnan(mean["LocA"])


def test_count_lines_ignores_blank_lines_and_missing_files(tmp_path: Path) -> None:
    """Two non-blank lines count 2; a missing file counts 0.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    """
    ev = _script()
    path = tmp_path / "x.txt"
    path.write_text("a\n\n  \nb\n")
    assert ev.count_lines(path) == 2 and ev.count_lines(tmp_path / "missing.txt") == 0


# ------------------------------------------------------------------------------------------- similarities


@needs_shapely
def test_bev_iou_hand_checked() -> None:
    """Unit squares half a side apart overlap 1/2 of 3/2: IoU 1/3; a square turned 90 degrees is itself; squares
    far apart and squares that only touch have IoU 0.
    """
    ev = _script()
    s = ev.sim_bev([box(0, 0)], [box(0.5, 0), box(0, 0, yaw=math.pi / 2), box(10, 0), box(1, 0)])
    np.testing.assert_allclose(s, [[1 / 3, 1.0, 0.0, 0.0]], atol=1e-12)
    s2 = ev.sim_bev([box(0, 0, w=2, l=1)], [box(0, 0, w=1, l=2, yaw=math.pi / 2)])
    np.testing.assert_allclose(s2, [[1.0]], atol=1e-12)


@needs_shapely
def test_bev_prefilter_is_exact() -> None:
    """The circumradius pre-filter gives bit-identical IoUs to the exhaustive shapely loop on random boxes."""
    ev = _script()
    rng = np.random.default_rng(0)

    def boxes(n: int) -> list[np.ndarray]:
        """Random boxes in a 6 m square.

        Parameters
        ----------
        n : int
            Boxes.

        Returns
        -------
        list of numpy.ndarray
            ``[x, y, z, w, l, h, yaw]`` boxes.
        """
        return [box(*rng.uniform(0, 6, 2), *rng.uniform(0.3, 2.5, 2), yaw=rng.uniform(-math.pi, math.pi))
                for _ in range(n)]

    g, t = boxes(25), boxes(30)
    fast, slow = ev.sim_bev(g, t), ev.sim_bev(g, t, prefilter=False)
    assert np.array_equal(fast, slow) and (fast > 0).sum() > 10 and (fast == 0).sum() > 10


@needs_pytorch3d
def test_3d_iou_hand_checked() -> None:
    """Unit cubes half a side apart: IoU 1/3; a cube turned 90 degrees about z is itself."""
    ev = _script()
    s = ev.sim_3d([box(0, 0)], [box(0.5, 0), box(0, 0, yaw=math.pi / 2)])
    np.testing.assert_allclose(s, [[1 / 3, 1.0]], atol=1e-5)


# ------------------------------------------------------------------------------------------------- HOTA


def run_hota(tmp_path: Path, gt: list, pred: list, *extra: str) -> dict[str, Any]:
    """Write GT and predictions, run ``hota --sim bev``, and return the scene ``"1"`` block of the JSON.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    gt : list of tuple
        GT rows (:func:`write_rows`).
    pred : list of tuple
        Predicted rows.
    *extra : str
        Further command-line options.

    Returns
    -------
    dict
        The scene's report.
    """
    ev = _script()
    out = tmp_path / "hota" / "hota_bev.json"
    ev.main(["hota", "--sim", "bev", "--gt", str(write_rows(tmp_path / "gt.txt", gt)),
             "--pred", str(write_rows(tmp_path / "pred.txt", pred)), "--out", str(out), *extra])
    result = json.loads(out.read_text())
    assert result["sim"] == "bev" and list(result["scenes"]) == ["1"]
    return result["scenes"]["1"]


@needs_hota
@needs_shapely
def test_hota_partial_overlap(tmp_path: Path) -> None:
    """IoU 1/3 in both frames: matched at 6 of 19 thresholds; HOTA = DetA = AssA = 600/19 %, LocA = 200/19 %.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    """
    scene = run_hota(tmp_path, [(1, 0, 5, 0, 0.0, 0.0), (1, 0, 5, 1, 0.0, 0.0)],
                     [(1, 0, 9, 0, 0.5, 0.0), (1, 0, 9, 1, 0.5, 0.0)])
    person = scene["per_class"]["Person"]
    assert list(scene["per_class"]) == ["Person"]
    for field in ("HOTA", "DetA", "AssA"):
        assert person[field] == pytest.approx(600 / 19)
        assert scene["det_pooled"][field] == pytest.approx(600 / 19)
    assert person["LocA"] == pytest.approx(200 / 19)
    assert (person["nGT"], person["nPred"]) == (2, 2)
    raw = scene["det_pooled_raw"]
    assert sorted(raw) == sorted(_script().RAW_KEYS) and all(len(v) == 19 for v in raw.values())
    assert raw["HOTA_TP"] == [2.0] * 6 + [0.0] * 13
    assert sorted(scene["per_class_raw_arrays"]) == sorted(_script().CLASSES.values())


@needs_hota
@needs_shapely
def test_hota_identity_switch(tmp_path: Path) -> None:
    """A perfect box whose id changes after the first frame: DetA 100 %, AssA 50 %, HOTA 100 sqrt(1/2) %.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    """
    scene = run_hota(tmp_path, [(1, 0, 5, 0, 2.0, 3.0), (1, 0, 5, 1, 2.0, 3.0)],
                     [(1, 0, 7, 0, 2.0, 3.0), (1, 0, 8, 1, 2.0, 3.0)], "--no-raw-arrays")
    person = scene["per_class"]["Person"]
    assert person["DetA"] == pytest.approx(100.0) and person["AssA"] == pytest.approx(50.0)
    assert person["HOTA"] == pytest.approx(100 * math.sqrt(0.5)) and person["LocA"] == pytest.approx(100.0)
    assert sorted(scene) == ["det_pooled", "per_class"]


@needs_hota
@needs_shapely
def test_hota_wrong_class_is_a_miss_and_an_extra(tmp_path: Path) -> None:
    """A perfect box labelled Forklift for a Person: Person all FN, Forklift all FP, pooled DetA 0; the output file
    is never overwritten.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    """
    scene = run_hota(tmp_path, [(1, 0, 5, 0, 0.0, 0.0)], [(1, 1, 5, 0, 0.0, 0.0)])
    assert scene["per_class"]["Person"]["nGT"] == 1 and scene["per_class"]["Person"]["nPred"] == 0
    assert scene["per_class"]["Forklift"]["nGT"] == 0 and scene["per_class"]["Forklift"]["nPred"] == 1
    assert scene["det_pooled"]["DetA"] == 0.0
    with pytest.raises(FileExistsError):
        _script().main(["hota", "--sim", "bev", "--gt", str(tmp_path / "gt.txt"), "--pred",
                        str(tmp_path / "pred.txt"), "--out", str(tmp_path / "hota" / "hota_bev.json")])


# ------------------------------------------------------------------------------------------------ score


@needs_hota
def test_empty_result_is_all_misses() -> None:
    """A class with 5 GT rows and no predictions: TP 0, FN 5, DetA and HOTA 0 at every threshold."""
    res = _script().empty_result(5, 0)
    assert res["HOTA_FN"].tolist() == [5.0] * 19 and res["HOTA_TP"].tolist() == [0.0] * 19
    assert res["DetA"].tolist() == [0.0] * 19 and res["HOTA"].tolist() == [0.0] * 19


@needs_hota
@needs_pytorch3d
def test_score_perfect_prediction(tmp_path: Path) -> None:
    """Predictions equal to the GT score 100 under every aggregation, the official run and the re-run agree, and
    an existing output folder is refused.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    """
    ev = _script()
    rows = [(1, c, i, f, x, 0.0) for f in range(3) for c, i, x in ((0, 1, 0.0), (0, 2, 3.0), (1, 3, 8.0))]
    gt, pred = write_rows(tmp_path / "gt.txt", rows), write_rows(tmp_path / "pred.txt", rows)
    scene_ids = tmp_path / "scene_ids.json"
    scene_ids.write_text(json.dumps({"1": "Warehouse_001"}))
    out = tmp_path / "eval" / "Warehouse_001"
    ev.main(["score", "--gt", str(gt), "--pred", str(pred), "--scene-ids", str(scene_ids), "--out-dir", str(out)])
    combined = json.loads((out / "combined_scores.json").read_text())
    assert list(combined) == ["official", "present_missing0", "det_pooled", "per_class_raw",
                              "crosscheck_max_abs_diff", "det_pooled_raw", "per_class_raw_arrays", "inputs"]
    for block in (combined["official"]["final"], combined["present_missing0"]["Warehouse_001"],
                  combined["det_pooled"]["Warehouse_001"]):
        assert block["HOTA"] == pytest.approx(100.0) and block["DetA"] == pytest.approx(100.0)
    assert combined["crosscheck_max_abs_diff"] == pytest.approx(0.0, abs=1e-9)
    assert {c: v["n_gt"] for c, v in combined["per_class_raw"]["Warehouse_001"].items()} == {"Person": 6,
                                                                                             "Forklift": 3}
    assert (out / "aicity_mtmc_hota_summary.json").exists()
    with pytest.raises(FileExistsError):
        ev.main(["score", "--gt", str(gt), "--pred", str(pred), "--scene-ids", str(scene_ids), "--out-dir", str(out)])


# -------------------------------------------------------------------------------------------- aggregate


def raw(tp: float, fn: float, fp: float, ass: float) -> dict[str, list[float]]:
    """Per-alpha raw arrays, constant over the 19 thresholds.

    Parameters
    ----------
    tp, fn, fp : float
        Detection counts.
    ass : float
        AssA (= AssRe = AssPr).

    Returns
    -------
    dict of str to list of float
        :data:`RAW_KEYS` arrays; LocA 0.8.
    """
    return {"HOTA_TP": [tp] * 19, "HOTA_FN": [fn] * 19, "HOTA_FP": [fp] * 19, "AssA": [ass] * 19,
            "AssRe": [ass] * 19, "AssPr": [ass] * 19, "LocA": [0.8] * 19}


def write_scene(root: Path, tag: str, scene: str, n_gt: int, hota: float, raws: dict[str, Any]) -> None:
    """Write one scene's ``combined_scores.json``, ``hota_bev.json`` and runner meta (only the keys read).

    Parameters
    ----------
    root : pathlib.Path
        Output root (``eval/`` and ``<tag>/`` below it).
    tag : str
        Checkpoint tag.
    scene : str
        Scene number, e.g. ``"000"``.
    n_gt : int
        GT rows of its single class (Person).
    hota : float
        Official and missing=0 HOTA of the scene, %.
    raws : dict
        Raw per-alpha arrays of Person (3D and BEV alike).
    """
    name = f"Warehouse_{scene}"
    block = {"HOTA": hota, "DetA": hota, "AssA": hota, "LocA": 80.0}
    combined = {"official": {"per_scene": {name: block}},
                "present_missing0": {name: dict(block, classes=["Person"])},
                "det_pooled": {name: {"HOTA": hota, "DetA": hota}},
                "per_class_raw": {name: {"Person": {"n_gt": n_gt, "n_pred": 3, "HOTA": hota}}},
                "det_pooled_raw": {name: raws}, "per_class_raw_arrays": {name: {"Person": raws}}}
    (root / "eval" / tag / name).mkdir(parents=True)
    (root / "eval" / tag / name / "combined_scores.json").write_text(json.dumps(combined))
    (root / "eval" / tag / f"{name}.bev").mkdir(parents=True)
    (root / "eval" / tag / f"{name}.bev" / "hota_bev.json").write_text(json.dumps(
        {"scenes": {str(int(scene)): {"det_pooled": {"HOTA": hota, "DetA": hota}, "det_pooled_raw": raws}}}))
    (root / tag).mkdir(exist_ok=True)
    (root / tag / f"{name}.meta.json").write_text(json.dumps(
        {"num_cameras": 4, "frames": {"count": 10}, "timing": {"model_s_excl_first_frame": 1.0, "wall_s_total": 2.0}}))


@needs_hota
def test_aggregate_two_scenes(tmp_path: Path) -> None:
    """Scenes 000 (4 GT rows, HOTA 60; TP 3 FN 1 FP 0, AssA 1) and 020 (2 rows, HOTA 30; TP 1 FN 1 FP 2,
    AssA 0.5): official = (4 x 60 + 2 x 30) / 6 = 50; pooled TP 4, FN 2, FP 2, so DetA = 4/8, AssA = 3.5/4,
    HOTA = sqrt(0.4375); groups without a scored scene are left out; the summary is never overwritten.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    """
    ev = _script()
    write_scene(tmp_path, "t", "000", 4, 60.0, raw(3, 1, 0, 1.0))
    write_scene(tmp_path, "t", "020", 2, 30.0, raw(1, 1, 2, 0.5))
    summary_path = tmp_path / "summary.json"
    ev.main(["aggregate", "--out-root", str(tmp_path), "--tags", "t", "--summary", str(summary_path)])
    summary = json.loads(summary_path.read_text())["t"]
    assert list(summary["per_scene"]) == ["000", "020"]
    assert set(summary["groups"]) == {"all", "shared_layout(000-006,020)", "val(020)", "train(11 scenes)"}
    group = summary["groups"]["all"]
    assert group["n_gt"] == 6 and group["official"]["HOTA"] == pytest.approx(50.0)
    for key in ("pooled3d", "pooledBEV"):
        assert group[key]["DetA"] == pytest.approx(50.0) and group[key]["AssA"] == pytest.approx(87.5)
        assert group[key]["HOTA"] == pytest.approx(100 * math.sqrt(0.4375))
        assert group[key]["DetRe"] == pytest.approx(400 / 6) and group[key]["DetPr"] == pytest.approx(400 / 6)
    assert group["per_class_pooled3d"]["Person"]["n_gt"] == 6
    assert group["per_class_pooled3d"]["Person"]["DetA"] == pytest.approx(50.0)
    assert summary["groups"]["val(020)"]["pooled3d"]["DetA"] == pytest.approx(25.0)
    with pytest.raises(FileExistsError):
        ev.main(["aggregate", "--out-root", str(tmp_path), "--tags", "t", "--summary", str(summary_path)])
