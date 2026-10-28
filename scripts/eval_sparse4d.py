"""Score Sparse4D's 11-column tracking output on Smart Spaces scenes: ground truth, HOTA, NVIDIA's evaluator, summary.

Runs in the ``mga-sparse4d`` conda env (``envs/sparse4d/README.md``) with NVIDIA's ``spatialai_data_utils``
on the path (``PYTHONPATH=<video-search-and-summarization>/libs/analytics/spatialai-data-utils``, repository
https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization, Apache-2.0; its bundled TrackEval code is
MIT); it is imported, not copied. ``gt`` needs only the standard library; ``hota --sim bev`` needs shapely;
``hota --sim 3d`` and ``score`` need pytorch3d. Nothing here imports ``measured_space``. Four subcommands:

``gt``
    Convert a scene's ``ground_truth.json`` into the AI City MTMC 11-column text format for a frame window
    (:func:`write_gt`). Rows: ``<scene_id> <class_id> <object_id> <frame_id> <x> <y> <z> <w> <l> <h> <yaw>`` with
    the official 2026 class ids (0 Person, 1 Forklift, 2 NovaCarter, 3 Transporter, 4 FourierGR1T2,
    5 AgilityDigit, 6 PalletTruck), world coordinates of the dataset (no re-centring), floats to 2 decimals,
    exactly as EVA/GaugeAlign's ``build_pkl.write_gt_track1`` writes them. Unknown object types are counted and
    reported, never silently dropped.

``hota``
    HOTA with BEV-IoU (``--sim bev``) or 3D-IoU (``--sim 3d``) similarity, mirroring EVA/GaugeAlign's local
    metric (:func:`hota_report`). EVA's ``scripts/evaluate_hota.py`` uses ZIOVISION's AI City 2025 fork of
    TrackEval (TrackEval is MIT-licensed), whose ``AICityChallengeSingleFile._calculate_similarities`` is the
    **BEV footprint IoU** (shapely polygons of [-w/2..w/2] x [-l/2..l/2] rotated by yaw; z and height ignored),
    with per-class filtering and contiguous id relabelling, and EVA reports the ``cls_comb_det_av`` combination.
    That similarity and that data preparation are re-implemented here (:func:`sim_bev`, :func:`seq_data`); the
    HOTA metric itself is TrackEval's, imported from ``spatialai_data_utils``. The sequence length is the largest
    GT frame + 1 (EVA's override). ``--sim 3d`` uses pytorch3d ``box3d_overlap`` on the NVIDIA evaluator's corner
    convention; it is a harness check whose per-class values should equal the NVIDIA evaluator's.

``score``
    NVIDIA's ``evaluate_aicity_mtmc`` (``spatialai_data_utils.eval.tracking.aicity_mtmc_eval``), unchanged, plus
    two extra aggregations (:func:`score`):

    1. ``official``: ``run_aicity_mtmc_evaluation`` (the function the ``tools/evaluation/evaluate_aicity_mtmc.py``
       CLI wraps): per-(scene, class) HOTA with 3D-IoU matching, per-scene unweighted mean over classes that have
       BOTH GT and predictions, GT-count-weighted across scenes. A GT class with no predictions is skipped (not
       scored 0) and prediction-only classes cost nothing.
    2. ``present_missing0``: per-scene mean over the classes present in the GT, a class with no predictions
       scored 0 (EVA's ``evaluate_hota_present.py`` description of the official 2026 aggregation).
    3. ``det_pooled``: TrackEval ``combine_classes_det_averaged`` over all classes (GT-only classes add their FNs,
       prediction-only classes add their FPs) = the ``cls_comb_det_av`` row EVA's ``gauge_score.py`` /
       ``factorial_score.py`` parse.

    (2) and (3) re-run the SAME bundled TrackEval (``MTMCChallenge3DBBox``, pytorch3d ``box3d_overlap``) per
    class on the split files the official run wrote, take the raw HOTA arrays, and combine them with the
    library's own ``HOTA`` methods. Per-class numbers are cross-checked against the official run
    (``crosscheck_max_abs_diff``).

``aggregate``
    The 12-scene summary per checkpoint tag (:func:`aggregate`), per scene and per scene group (all; layout shared
    with other 2026 scenes 000-006 and 020 vs other layouts 010-016; val 020 vs the 11 train scenes):
    ``official`` (GT-row-weighted mean of per-scene class means; identical to running the evaluator on the
    concatenated scenes), ``missing0`` (the same weighting of ``present_missing0``), ``pooled3d`` / ``pooledBEV``
    (``combine_classes_det_averaged`` over all (scene, class) raw per-alpha arrays of the group), and per-class
    pooled 3D DetA / HOTA. It reads ``<eval-root>/<tag>/<scene>/combined_scores.json`` (``score``),
    ``<eval-root>/<tag>/<scene>.bev/hota_bev.json`` (``hota --sim bev``) and ``<pred-root>/<tag>/<scene>.meta.json``
    (``scripts/run_sparse4d.py``).

**Two output formats.** The first (single-scene) runs wrote ``hota`` and ``score`` JSON without the raw per-alpha
arrays; the 12-scene runs added them (``det_pooled_raw``, ``per_class_raw_arrays``), and ``aggregate`` needs
them. Both write the arrays by default; ``--no-raw-arrays`` writes the first format. The first ``hota`` version
also computed the BEV IoU of every pair with shapely; the 12-scene version skips pairs whose centres are farther
apart than the sum of their footprint circumradii (their IoU is exactly 0). The result is identical; the
exhaustive loop is kept behind ``--no-bev-prefilter``.

Usage (12-scene layout; ``<out>`` = the runner's ``--out-dir``)::

    python -I scripts/eval_sparse4d.py gt --scene-dir <data>/MTMC_Tracking_2026/train/Warehouse_000 \\
        --scene-id 0 --start 0 --end 9000 --step 6 --out <out>/gt/Warehouse_000_f0-9000s6.txt
    PYTHONPATH=<spatialai-data-utils> python scripts/eval_sparse4d.py hota --sim bev \\
        --gt <out>/gt/Warehouse_000_f0-9000s6.txt --pred <out>/<tag>/Warehouse_000.txt \\
        --out <out>/eval/<tag>/Warehouse_000.bev/hota_bev.json
    PYTHONPATH=<spatialai-data-utils> python scripts/eval_sparse4d.py score \\
        --gt <out>/gt/Warehouse_000_f0-9000s6.txt --pred <out>/<tag>/Warehouse_000.txt \\
        --scene-ids scene_ids.json --out-dir <out>/eval/<tag>/Warehouse_000
    PYTHONPATH=<spatialai-data-utils> python scripts/eval_sparse4d.py aggregate --out-root <out> \\
        --tags rn50_trainable_v2.2 rn101_trainable_v1.1 --summary <out>/eval/summary_stage2.json

``scene_ids.json`` maps the integer scene id (as a string) to the scene name, e.g. ``{"0": "Warehouse_000",
"20": "Warehouse_020"}``. Every subcommand refuses to overwrite its output (``score`` refuses an existing
``--out-dir``) and deletes nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# ------------------------------------------------------------------------------------------------ constants

TYPE_TO_CLASS_ID: dict[str, int] = {
    "Person": 0,
    "Forklift": 1,
    "NovaCarter": 2,
    "Transporter": 3,
    "FourierGR1T2": 4,
    "AgilityDigit": 5,
    "PalletTruck": 6,
}
"""dict of str to int: the dataset's object type -> the 2026 challenge's class id."""

CLASSES: dict[int, str] = {v: k for k, v in TYPE_TO_CLASS_ID.items()}
"""dict of int to str: class id -> object type, in class-id order (the order ``hota`` scores them)."""

UNIT = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
"""numpy.ndarray, shape (8, 3): unit-cube corners in pytorch3d ``box3d_overlap``'s order."""

N_ALPHA = 19
"""int: HOTA's localisation thresholds (alpha = 0.05, 0.10, ..., 0.95)."""

FIELDS = ["HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr"]
"""list of str: metrics reported per class by ``score``."""

RAW_KEYS = ("HOTA_TP", "HOTA_FN", "HOTA_FP", "AssA", "AssRe", "AssPr", "LocA")
"""tuple of str: the per-alpha arrays stored for detection pooling (``--no-raw-arrays`` omits them)."""

SCENES = ["000", "002", "004", "006", "010", "011", "012", "013", "014", "015", "016", "020"]
"""list of str: the 12 scenes of the summary (``Warehouse_<s>``)."""

GROUPS: dict[str, list[str]] = {
    "all": SCENES,
    "shared_layout(000-006,020)": ["000", "002", "004", "006", "020"],
    "other_layouts(010-016)": ["010", "011", "012", "013", "014", "015", "016"],
    "val(020)": ["020"],
    "train(11 scenes)": [s for s in SCENES if s != "020"],
}
"""dict of str to list of str: scene groups of the summary."""

SimFn = Callable[[list, list], np.ndarray]


def hota_class() -> type:
    """Import TrackEval's ``HOTA`` metric as bundled with NVIDIA's ``spatialai_data_utils``.

    Returns
    -------
    type
        ``spatialai_data_utils.eval.tracking.hota.metrics.hota.HOTA``.

    Raises
    ------
    ImportError
        If ``spatialai_data_utils`` (or its ``eval`` dependencies) is not importable; put
        ``<video-search-and-summarization>/libs/analytics/spatialai-data-utils`` on ``PYTHONPATH``.
    """
    try:
        from spatialai_data_utils.eval.tracking.hota.metrics.hota import HOTA
    except ImportError as exc:
        raise ImportError("spatialai_data_utils is not importable; set PYTHONPATH=<video-search-and-summarization>"
                          "/libs/analytics/spatialai-data-utils (see envs/sparse4d/README.md)") from exc
    return HOTA


# ----------------------------------------------------------------------------------------------- gt


def gt_rows(gt: Mapping[str, list[dict[str, Any]]], scene_id: int,
            frames: Sequence[int]) -> tuple[list[str], Counter]:
    """The 11-column GT rows of a frame window, and per-type row counts.

    Parameters
    ----------
    gt : mapping of str to list of dict
        Parsed ``ground_truth.json``: frame id (string) -> objects with ``object type``, ``object id``,
        ``3d location``, ``3d bounding box scale`` and ``3d bounding box rotation``.
    scene_id : int
        Integer written in the first column.
    frames : sequence of int
        0-based frame ids to include, in output order.

    Returns
    -------
    tuple[list[str], Counter]
        The rows (each ending in a newline) and the row count per object type (unknown types are counted
        under ``"UNKNOWN:<type>"`` and not written).
    """
    counts: Counter = Counter()
    lines = []
    for f in frames:
        for o in gt.get(str(f), []):
            t = o.get("object type")
            cid = TYPE_TO_CLASS_ID.get(t)
            if cid is None:
                counts[f"UNKNOWN:{t}"] += 1
                continue
            x, y, z = o["3d location"]
            w, l, h = o["3d bounding box scale"]
            yaw = o["3d bounding box rotation"][2]
            lines.append(f"{scene_id} {cid} {int(o['object id'])} {f} {x:.2f} {y:.2f} {z:.2f} "
                         f"{w:.2f} {l:.2f} {h:.2f} {yaw:.2f}\n")
            counts[t] += 1
    return lines, counts


def write_gt(scene_dir: Path, scene_id: int, frames: list[int], out: Path) -> Counter:
    """Write the GT rows for ``frames`` and return per-type row counts.

    Parameters
    ----------
    scene_dir : Path
        Scene directory containing ``ground_truth.json``.
    scene_id : int
        Integer written in the first column.
    frames : list[int]
        0-based frame ids to include.
    out : Path
        Destination text file; must not exist.

    Returns
    -------
    Counter
        Row count per object type (unknown types are counted under ``"UNKNOWN:<type>"`` and not written).

    Raises
    ------
    FileExistsError
        If ``out`` exists.
    """
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    gt = json.loads((scene_dir / "ground_truth.json").read_text())
    lines, counts = gt_rows(gt, scene_id, frames)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as fo:
        fo.writelines(lines)
    return counts


def run_gt(args: argparse.Namespace) -> int:
    """The ``gt`` subcommand: write the GT file and print row counts per type.

    Parameters
    ----------
    args : argparse.Namespace
        ``scene_dir``, ``scene_id``, ``start``, ``end``, ``step``, ``out``.

    Returns
    -------
    int
        Exit status 0.
    """
    frames = list(range(args.start, args.end, args.step))
    counts = write_gt(args.scene_dir, args.scene_id, frames, args.out)
    n_rows = sum(v for k, v in counts.items() if not k.startswith("UNKNOWN"))
    print(f"{args.scene_dir.name}: {len(frames)} frames, {n_rows} rows -> {args.out}")
    print("  per type:", dict(sorted(counts.items())))
    return 0


# --------------------------------------------------------------------------------------------- hota


def load_tracks(path: Path) -> dict[str, dict[int, list[tuple[int, int, np.ndarray]]]]:
    """Load an 11-column AI City file into ``{scene: {frame: [(class, id, det7)]}}``.

    Parameters
    ----------
    path : Path
        11-column text file (GT or predictions).

    Returns
    -------
    dict
        Scene id (the first column, as written) -> frame -> ``(class_id, object_id, [x, y, z, w, l, h, yaw])``
        rows in file order.
    """
    out: dict = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            out[p[0]][int(p[3])].append((int(p[1]), int(p[2]), np.array([float(v) for v in p[4:11]])))
    return out


def bev_poly(d: np.ndarray) -> Any:
    """ZIOVISION-fork BEV polygon of a ``[x, y, z, w, l, h, yaw]`` box.

    Parameters
    ----------
    d : np.ndarray, shape (7,)
        Box: centre, size ``(w, l, h)`` along the box's x, y, z, and yaw about +z (radians).

    Returns
    -------
    shapely.geometry.Polygon
        The footprint ``[-w/2, w/2] x [-l/2, l/2]`` rotated by yaw and moved to ``(x, y)``.
    """
    from shapely.geometry import Polygon

    x, y, _, w, l, _, yaw = d
    c, s = np.cos(yaw), np.sin(yaw)
    r = np.array([[c, -s], [s, c]])
    corners = np.array([[-w / 2, -l / 2], [-w / 2, l / 2], [w / 2, l / 2], [w / 2, -l / 2]])
    return Polygon(corners @ r.T + np.array([x, y]))


def sim_bev(g: list, t: list, prefilter: bool = True) -> np.ndarray:
    """BEV IoU matrix exactly as the ZIOVISION fork computes it.

    Parameters
    ----------
    g : list of np.ndarray
        GT boxes ``[x, y, z, w, l, h, yaw]``.
    t : list of np.ndarray
        Predicted boxes, same layout.
    prefilter : bool, optional
        If True (default, the 12-scene version), pairs whose centres are farther apart than the sum of their
        footprint circumradii cannot overlap; their IoU is exactly 0 and is not passed to shapely (a pure
        speed-up, identical result). If False, every pair goes through shapely (the first version).

    Returns
    -------
    np.ndarray, shape (len(g), len(t))
        Footprint intersection over union; 0 where the union has no area.
    """
    s = np.zeros((len(g), len(t)))
    if not prefilter:
        gp, tp = [bev_poly(d) for d in g], [bev_poly(d) for d in t]
        for i, a in enumerate(gp):
            for j, b in enumerate(tp):
                u = a.union(b).area
                if u > 0:
                    s[i, j] = a.intersection(b).area / u
        return s
    ga, ta = np.array(g), np.array(t)
    rg = 0.5 * np.hypot(ga[:, 3], ga[:, 4])
    rt = 0.5 * np.hypot(ta[:, 3], ta[:, 4])
    dist = np.linalg.norm(ga[:, None, :2] - ta[None, :, :2], axis=-1)
    cand = dist <= (rg[:, None] + rt[None, :]) + 1e-9
    if not cand.any():
        return s
    gpd = {i: bev_poly(g[i]) for i in np.unique(np.nonzero(cand)[0])}
    tpd = {j: bev_poly(t[j]) for j in np.unique(np.nonzero(cand)[1])}
    for i, j in zip(*np.nonzero(cand)):
        a, b = gpd[i], tpd[j]
        u = a.union(b).area
        if u > 0:
            s[i, j] = a.intersection(b).area / u
    return s


def sim_3d(g: list, t: list) -> np.ndarray:
    """3D IoU matrix with pytorch3d ``box3d_overlap`` on the NVIDIA evaluator's corner convention (yaw only).

    Parameters
    ----------
    g : list of np.ndarray
        GT boxes ``[x, y, z, w, l, h, yaw]`` (``z`` is the box centre).
    t : list of np.ndarray
        Predicted boxes, same layout.

    Returns
    -------
    np.ndarray, shape (len(g), len(t))
        Volume intersection over union.
    """
    import torch
    from pytorch3d.ops import box3d_overlap

    def corners(d: np.ndarray) -> np.ndarray:
        """The 8 corners of one box.

        Parameters
        ----------
        d : np.ndarray, shape (7,)
            Box ``[x, y, z, w, l, h, yaw]``.

        Returns
        -------
        np.ndarray, shape (8, 3)
            Corners in :data:`UNIT`'s order.
        """
        local = UNIT * d[3:6] - d[3:6] / 2.0
        c, s = np.cos(d[6]), np.sin(d[6])
        return local @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + d[:3]

    a = torch.tensor(np.stack([corners(d) for d in g]), dtype=torch.float32)
    b = torch.tensor(np.stack([corners(d) for d in t]), dtype=torch.float32)
    return box3d_overlap(a, b)[1].numpy().astype(float)


def seq_data(gt_s: Mapping[int, list], pr_s: Mapping[int, list], cls: int, n_t: int, sim: SimFn) -> dict[str, Any]:
    """Build the TrackEval per-class sequence data dict (fork semantics: class filter + contiguous id relabel).

    Parameters
    ----------
    gt_s : mapping of int to list
        One scene's GT rows per frame (:func:`load_tracks`).
    pr_s : mapping of int to list
        One scene's predicted rows per frame.
    cls : int
        Class id kept.
    n_t : int
        Number of timesteps (frame ids ``0 .. n_t - 1``; frames without rows are empty timesteps).
    sim : callable
        Similarity ``sim(gt_boxes, pred_boxes) -> (n_gt, n_pred)`` array (:func:`sim_bev` or :func:`sim_3d`).

    Returns
    -------
    dict[str, Any]
        TrackEval's ``data`` for ``HOTA.eval_sequence``: per-timestep ``gt_ids``, ``tracker_ids`` (relabelled to
        ``0 .. n - 1`` in sorted order of the original ids) and ``similarity_scores``, and the counts.
    """
    data: dict[str, Any] = {k: [None] * n_t for k in ("gt_ids", "tracker_ids", "similarity_scores")}
    ug, ut, ng, nt = set(), set(), 0, 0
    for t in range(n_t):
        g = [(i, d) for c, i, d in gt_s.get(t, []) if c == cls]
        p = [(i, d) for c, i, d in pr_s.get(t, []) if c == cls]
        data["gt_ids"][t] = np.array([i for i, _ in g], dtype=int)
        data["tracker_ids"][t] = np.array([i for i, _ in p], dtype=int)
        data["similarity_scores"][t] = (sim([d for _, d in g], [d for _, d in p]) if g and p
                                        else np.zeros((len(g), len(p))))
        ug.update(data["gt_ids"][t].tolist())
        ut.update(data["tracker_ids"][t].tolist())
        ng += len(g)
        nt += len(p)
    gmap = {o: n for n, o in enumerate(sorted(ug))}
    tmap = {o: n for n, o in enumerate(sorted(ut))}
    for t in range(n_t):
        data["gt_ids"][t] = np.array([gmap[i] for i in data["gt_ids"][t]], dtype=int)
        data["tracker_ids"][t] = np.array([tmap[i] for i in data["tracker_ids"][t]], dtype=int)
    data.update(num_gt_dets=ng, num_tracker_dets=nt, num_gt_ids=len(ug), num_tracker_ids=len(ut), num_timesteps=n_t)
    return data


def hota_report(gt: Mapping[str, Mapping[int, list]], pr: Mapping[str, Mapping[int, list]], sim: SimFn,
                raw_arrays: bool = True, label: str = "") -> dict[str, Any]:
    """Score every scene of the GT per class and detection-pooled; print a table per scene.

    Parameters
    ----------
    gt : mapping
        GT rows per scene and frame (:func:`load_tracks`).
    pr : mapping
        Predicted rows per scene and frame; a scene without predictions scores all FN.
    sim : callable
        Similarity function (:func:`sim_bev` or :func:`sim_3d`).
    raw_arrays : bool, optional
        Also store the per-alpha arrays ``det_pooled_raw`` and ``per_class_raw_arrays`` (default, the 12-scene
        format); False gives the first version's format.
    label : str, optional
        Similarity name for the printed header (``bev`` or ``3d``).

    Returns
    -------
    dict[str, Any]
        Per scene: ``per_class`` (HOTA, DetA, AssA, LocA in %, nGT, nPred; only classes with any GT or
        prediction), ``det_pooled`` (HOTA, DetA, AssA in %), and, with ``raw_arrays``, the raw arrays.
    """
    metric = hota_class()()
    report: dict[str, Any] = {}
    for scene in sorted(gt):
        n_t = max(gt[scene]) + 1  # EVA's override: sequence length = max GT frame + 1
        per_class = {}
        for cls, name in CLASSES.items():
            res = metric.eval_sequence(seq_data(gt[scene], pr.get(scene, {}), cls, n_t, sim))
            per_class[name] = res
        pooled_res = metric.combine_classes_det_averaged(per_class)
        rows = {n: {f: float(np.mean(r[f])) * 100 for f in ("HOTA", "DetA", "AssA", "LocA")} |
                {"nGT": int(r["HOTA_TP"][0] + r["HOTA_FN"][0]), "nPred": int(r["HOTA_TP"][0] + r["HOTA_FP"][0])}
                for n, r in per_class.items() if (r["HOTA_TP"][0] + r["HOTA_FN"][0] + r["HOTA_FP"][0]) > 0}
        report[scene] = {"per_class": rows,
                         "det_pooled": {f: float(np.mean(pooled_res[f])) * 100 for f in ("HOTA", "DetA", "AssA")}}
        if raw_arrays:
            report[scene]["det_pooled_raw"] = {k: np.asarray(pooled_res[k], dtype=float).tolist() for k in RAW_KEYS}
            report[scene]["per_class_raw_arrays"] = {
                n: {k: np.asarray(r[k], dtype=float).tolist() for k in RAW_KEYS} for n, r in per_class.items()}
        print(f"== scene {scene} ({label} IoU)")
        for n, v in rows.items():
            print(f"   {n:13s} nGT {v['nGT']:5d} nPred {v['nPred']:5d}  HOTA {v['HOTA']:6.2f} DetA {v['DetA']:6.2f} "
                  f"AssA {v['AssA']:6.2f} LocA {v['LocA']:6.2f}")
        d = report[scene]["det_pooled"]
        print(f"   det-pooled: HOTA {d['HOTA']:.2f} DetA {d['DetA']:.2f} AssA {d['AssA']:.2f}")
    return report


def run_hota(args: argparse.Namespace) -> int:
    """The ``hota`` subcommand: score per class and detection-pooled for every scene in the GT file.

    Parameters
    ----------
    args : argparse.Namespace
        ``gt``, ``pred``, ``sim``, ``out``, ``no_raw_arrays``, ``no_bev_prefilter``.

    Returns
    -------
    int
        Exit status 0.

    Raises
    ------
    FileExistsError
        If ``--out`` exists.
    """
    if args.out is not None and args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if args.sim == "bev":
        prefilter = not args.no_bev_prefilter

        def sim(g: list, t: list) -> np.ndarray:
            """:func:`sim_bev` with the chosen pre-filter.

            Parameters
            ----------
            g, t : list of np.ndarray
                GT and predicted boxes.

            Returns
            -------
            np.ndarray
                BEV IoU matrix.
            """
            return sim_bev(g, t, prefilter=prefilter)
    else:
        sim = sim_3d
    report = hota_report(load_tracks(args.gt), load_tracks(args.pred), sim, raw_arrays=not args.no_raw_arrays,
                         label=args.sim)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "x") as f:
            json.dump({"sim": args.sim, "gt": str(args.gt), "pred": str(args.pred), "scenes": report}, f, indent=1)
    return 0


# -------------------------------------------------------------------------------------------- score


def raw_hota(gt_file: Path, pred_file: Path, work_dir: Path, seq_name: str, seq_length: int) -> dict[str, Any]:
    """Run the bundled TrackEval 3D-IoU HOTA on one (scene, class) pair and return the raw per-alpha result dict.

    Parameters
    ----------
    gt_file : Path
        MOT-format GT split file written by the official run.
    pred_file : Path
        MOT-format prediction split file written by the official run.
    work_dir : Path
        Scratch directory for TrackEval's folder layout (must not exist yet).
    seq_name : str
        Scene name.
    seq_length : int
        Sequence length passed to TrackEval (same as the official run).

    Returns
    -------
    dict[str, Any]
        TrackEval HOTA result with per-alpha arrays (``HOTA_TP``, ``HOTA_FN``, ``HOTA_FP``, ``AssA``, ``LocA``, ...).
    """
    from spatialai_data_utils.eval.tracking.aicity_mtmc_eval import _TRACKER_NAME
    from spatialai_data_utils.eval.tracking.hota.trackeval_utils import (
        prepare_evaluation_folder,
        run_evaluation,
        setup_evaluation_configs,
    )

    dataset_config, eval_config = setup_evaluation_configs(str(work_dir), "bbox", 1)
    eval_config.update(PRINT_RESULTS=False, PRINT_CONFIG=False, TIME_PROGRESS=False, DISPLAY_LESS_PROGRESS=True,
                       OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False, PLOT_CURVES=False)
    out_pred, out_gt = prepare_evaluation_folder(dataset_config, seq_name, fps=30.0, seq_length=seq_length)
    shutil.copyfile(gt_file, out_gt)
    shutil.copyfile(pred_file, out_pred)
    output_res, _ = run_evaluation(out_gt, out_pred, dataset_config, eval_config, "bbox")
    return output_res["MTMCChallenge3DBBox"][_TRACKER_NAME][seq_name]["class"]["HOTA"]


def empty_result(n_gt: int, n_pred: int) -> dict[str, Any]:
    """Raw HOTA dict for a class with no matches possible (GT-only: all FN; prediction-only: all FP).

    Parameters
    ----------
    n_gt : int
        Number of GT object-frame rows.
    n_pred : int
        Number of predicted object-frame rows.

    Returns
    -------
    dict[str, Any]
        Result with the integer arrays filled and every float array zero, finalised with HOTA's own formulas.
    """
    res = {f: np.zeros(N_ALPHA) for f in ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA"]}
    res["HOTA_TP"] = np.zeros(N_ALPHA)
    res["HOTA_FN"] = np.full(N_ALPHA, float(n_gt))
    res["HOTA_FP"] = np.full(N_ALPHA, float(n_pred))
    return hota_class()._compute_final_fields(res)


def count_lines(path: Path) -> int:
    """Return the number of non-empty lines in ``path`` (0 if absent).

    Parameters
    ----------
    path : Path
        Text file.

    Returns
    -------
    int
        Lines with any non-whitespace character.
    """
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def score(gt: Path, pred: Path, scene_map: Mapping[str, str], out_dir: Path, num_frames_to_eval: int = 9000,
          raw_arrays: bool = True) -> dict[str, Any]:
    """Run NVIDIA's official evaluation and the two extra aggregations; write their JSON files.

    Parameters
    ----------
    gt : Path
        11-column GT file.
    pred : Path
        11-column prediction file.
    scene_map : mapping of str to str
        Integer scene id (string) -> scene name.
    out_dir : Path
        Evaluation folder; must not exist. Receives NVIDIA's ``aicity_mtmc_hota_summary.json`` and ``split/``
        folders, and ``combined_scores.json``.
    num_frames_to_eval : int, optional
        Frames evaluated (NVIDIA's ``--num_frames_to_eval``; default 9000).
    raw_arrays : bool, optional
        Also store ``det_pooled_raw`` and ``per_class_raw_arrays`` (default, the 12-scene format).

    Returns
    -------
    dict[str, Any]
        The ``combined_scores.json`` content.

    Raises
    ------
    FileExistsError
        If ``out_dir`` exists.
    """
    from spatialai_data_utils.eval.tracking.aicity_mtmc_eval import (
        run_aicity_mtmc_evaluation,
        save_aicity_mtmc_results,
    )

    if out_dir.exists():
        raise FileExistsError(f"refusing to reuse {out_dir}")
    official = run_aicity_mtmc_evaluation(
        ground_truth_file=str(gt), prediction_file=str(pred), scene_id_to_name=dict(scene_map),
        output_dir=str(out_dir), num_frames_to_eval=num_frames_to_eval, quiet=True)
    save_aicity_mtmc_results(official, str(out_dir))

    hota = hota_class()()
    split_root = out_dir / "split"
    combined: dict[str, Any] = {"official": {"per_scene": {}, "per_scene_per_class": {}}, "present_missing0": {},
                                "det_pooled": {}, "per_class_raw": {}, "crosscheck_max_abs_diff": 0.0}
    for scene_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        scene = scene_dir.name
        per_class = {}
        raws = {}
        for class_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
            gt_f, pred_f = class_dir / "gt.txt", class_dir / "pred.txt"
            n_gt, n_pred = count_lines(gt_f), count_lines(pred_f)
            if n_gt and n_pred:
                raw = raw_hota(gt_f, pred_f, class_dir / "trackeval_raw", scene, num_frames_to_eval)
            else:
                raw = empty_result(n_gt, n_pred)
            raws[class_dir.name] = raw
            per_class[class_dir.name] = {"n_gt": n_gt, "n_pred": n_pred,
                                         **{f: float(np.mean(raw[f])) * 100 for f in FIELDS}}
            off = (official["per_scene_per_class"].get(scene, {}) or {}).get(class_dir.name)
            if off is not None:
                for f in ("HOTA", "DetA", "AssA", "LocA"):
                    combined["crosscheck_max_abs_diff"] = max(combined["crosscheck_max_abs_diff"],
                                                              abs(off[f] * 100 - per_class[class_dir.name][f]))
        combined["per_class_raw"][scene] = per_class
        present = [c for c, v in per_class.items() if v["n_gt"] > 0]
        combined["present_missing0"][scene] = {
            f: float(np.mean([per_class[c][f] for c in present])) for f in ("HOTA", "DetA", "AssA", "LocA")}
        combined["present_missing0"][scene]["classes"] = present
        pooled_res = hota.combine_classes_det_averaged(raws)
        combined["det_pooled"][scene] = {f: float(np.mean(pooled_res[f])) * 100 for f in FIELDS}
        combined["det_pooled"][scene]["TP_alpha0.5"] = float(pooled_res["HOTA_TP"][9])
        combined["det_pooled"][scene]["FN_alpha0.5"] = float(pooled_res["HOTA_FN"][9])
        combined["det_pooled"][scene]["FP_alpha0.5"] = float(pooled_res["HOTA_FP"][9])
        if raw_arrays:
            combined.setdefault("det_pooled_raw", {})[scene] = {
                k: np.asarray(pooled_res[k], dtype=float).tolist() for k in RAW_KEYS}
            combined.setdefault("per_class_raw_arrays", {})[scene] = {
                c: {k: np.asarray(r[k], dtype=float).tolist() for k in RAW_KEYS} for c, r in raws.items()}
        if scene in official["per_scene"]:
            combined["official"]["per_scene"][scene] = {f: official["per_scene"][scene][f] * 100
                                                        for f in ("HOTA", "DetA", "AssA", "LocA")}
        combined["official"]["per_scene_per_class"][scene] = {
            c: (None if m is None else {f: m[f] * 100 for f in ("HOTA", "DetA", "AssA", "LocA")})
            for c, m in (official["per_scene_per_class"].get(scene) or {}).items()}
    combined["official"]["final"] = {f: official["final"][f] * 100 for f in ("HOTA", "DetA", "AssA", "LocA")}
    combined["inputs"] = {"gt": str(gt), "pred": str(pred), "num_frames_to_eval": num_frames_to_eval}
    with open(out_dir / "combined_scores.json", "x") as f:
        json.dump(combined, f, indent=1)
    return combined


def print_scores(combined: Mapping[str, Any]) -> None:
    """Print the per-class and per-scene tables of a ``combined_scores.json``.

    Parameters
    ----------
    combined : mapping
        :func:`score`'s result.
    """
    for scene in combined["per_class_raw"]:
        print(f"== {scene}")
        for c, v in combined["per_class_raw"][scene].items():
            print(f"   {c:13s} nGT {v['n_gt']:6d} nPred {v['n_pred']:6d}  HOTA {v['HOTA']:6.2f} DetA {v['DetA']:6.2f} "
                  f"AssA {v['AssA']:6.2f} LocA {v['LocA']:6.2f} DetRe {v['DetRe']:6.2f} DetPr {v['DetPr']:6.2f}")
        o = combined["official"]["per_scene"].get(scene)
        if o:
            print(f"   official (classes with GT and preds): HOTA {o['HOTA']:.2f} DetA {o['DetA']:.2f} "
                  f"AssA {o['AssA']:.2f} LocA {o['LocA']:.2f}")
        p = combined["present_missing0"][scene]
        print(f"   present-class mean, missing=0:        HOTA {p['HOTA']:.2f} DetA {p['DetA']:.2f} "
              f"AssA {p['AssA']:.2f} LocA {p['LocA']:.2f}")
        d = combined["det_pooled"][scene]
        print(f"   det-pooled (cls_comb_det_av):         HOTA {d['HOTA']:.2f} DetA {d['DetA']:.2f} "
              f"AssA {d['AssA']:.2f} LocA {d['LocA']:.2f} DetRe {d['DetRe']:.2f} DetPr {d['DetPr']:.2f}")
    print(f"official final: {combined['official']['final']}")
    print(f"crosscheck max |official - raw| per-class = {combined['crosscheck_max_abs_diff']:.6f}")


def run_score(args: argparse.Namespace) -> int:
    """The ``score`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        ``gt``, ``pred``, ``scene_ids``, ``out_dir``, ``num_frames_to_eval``, ``no_raw_arrays``.

    Returns
    -------
    int
        Exit status 0.

    Raises
    ------
    FileExistsError
        If ``--out-dir`` exists.
    """
    if args.out_dir.exists():
        raise FileExistsError(f"refusing to reuse {args.out_dir}")
    logging.basicConfig(level=logging.WARNING)
    scene_map = json.loads(args.scene_ids.read_text())
    combined = score(args.gt, args.pred, scene_map, args.out_dir, args.num_frames_to_eval,
                     raw_arrays=not args.no_raw_arrays)
    print_scores(combined)
    return 0


# ---------------------------------------------------------------------------------------- aggregate


def pooled(raws: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Detection-pool raw per-alpha HOTA dicts with TrackEval's own combine; return % metrics.

    Parameters
    ----------
    raws : sequence of mapping
        Raw per-alpha arrays (at least :data:`RAW_KEYS`) of (scene, class) pairs.

    Returns
    -------
    dict[str, float]
        HOTA, DetA, AssA, DetRe, DetPr in % (empty if ``raws`` is empty).
    """
    if not raws:
        return {}
    res = hota_class()().combine_classes_det_averaged({i: {k: np.asarray(r[k], dtype=float) for k in RAW_KEYS}
                                                       for i, r in enumerate(raws)})
    return {f: float(np.mean(res[f])) * 100 for f in ("HOTA", "DetA", "AssA", "DetRe", "DetPr")}


def weighted_mean(per_scene: Mapping[str, Mapping[str, Any]], members: Sequence[str], key: str) -> dict[str, float]:
    """GT-row-weighted mean over scenes of a per-scene metric block (the NVIDIA evaluator's cross-scene rule).

    Parameters
    ----------
    per_scene : mapping
        Scene -> block with ``n_gt`` and ``key`` (a mapping holding HOTA, DetA, AssA, LocA; a missing metric is
        NaN, so the mean is NaN).
    members : sequence of str
        Scenes of the group.
    key : str
        ``official`` or ``missing0``.

    Returns
    -------
    dict[str, float]
        HOTA, DetA, AssA, LocA weighted by each scene's ``n_gt``.
    """
    w = np.array([per_scene[s]["n_gt"] for s in members], float)
    return {f: float(np.sum(w * [per_scene[s][key].get(f, np.nan) for s in members]) / w.sum())
            for f in ("HOTA", "DetA", "AssA", "LocA")}


def aggregate(eval_root: Path, pred_root: Path, tags: Sequence[str]) -> dict[str, Any]:
    """Build the per-scene and per-group summary for each checkpoint tag; print the tables.

    Parameters
    ----------
    eval_root : Path
        Folder holding ``<tag>/<scene>/combined_scores.json`` and ``<tag>/<scene>.bev/hota_bev.json``.
    pred_root : Path
        Folder holding the runner's ``<tag>/<scene>.meta.json``.
    tags : sequence of str
        Checkpoint tags.

    Returns
    -------
    dict[str, Any]
        Tag -> ``per_scene`` (scenes with both score files) and ``groups``.
    """
    summary: dict[str, Any] = {}
    for tag in tags:
        per_scene, d3, dbev = {}, {}, {}
        for s in SCENES:
            name = f"Warehouse_{s}"
            p3 = eval_root / tag / name / "combined_scores.json"
            pb = eval_root / tag / f"{name}.bev" / "hota_bev.json"
            if not p3.exists() or not pb.exists():
                print(f"[{tag}] missing scores for {name}")
                continue
            c3 = json.loads(p3.read_text())
            cb = json.loads(pb.read_text())
            sb = next(iter(cb["scenes"].values()))
            off = c3["official"]["per_scene"].get(name, {})
            ngt = sum(v["n_gt"] for v in c3["per_class_raw"][name].values())
            npred = sum(v["n_pred"] for v in c3["per_class_raw"][name].values())
            meta = json.loads((pred_root / tag / f"{name}.meta.json").read_text())
            per_scene[s] = {
                "n_gt": ngt, "n_pred": npred, "cams": meta["num_cameras"], "frames": meta["frames"]["count"],
                "official": off, "missing0": c3["present_missing0"][name],
                "pooled3d": c3["det_pooled"][name], "pooledBEV": sb["det_pooled"],
                "model_s": meta["timing"]["model_s_excl_first_frame"], "wall_s": meta["timing"]["wall_s_total"],
                "per_class3d": c3["per_class_raw"][name],
            }
            d3[s] = c3["det_pooled_raw"][name]
            dbev[s] = sb["det_pooled_raw"]
            per_scene[s]["_pc_raw3d"] = c3["per_class_raw_arrays"][name]
        groups = {}
        for g, members in GROUPS.items():
            m = [s for s in members if s in per_scene]
            if not m:
                continue
            agg: dict[str, Any] = {"scenes": m, "n_gt": int(sum(per_scene[s]["n_gt"] for s in m))}
            for key in ("official", "missing0"):
                agg[key] = weighted_mean(per_scene, m, key)
            agg["pooled3d"] = pooled([d3[s] for s in m])
            agg["pooledBEV"] = pooled([dbev[s] for s in m])
            classes = sorted({c for s in m for c in per_scene[s]["_pc_raw3d"]})
            agg["per_class_pooled3d"] = {
                c: pooled([per_scene[s]["_pc_raw3d"][c] for s in m if c in per_scene[s]["_pc_raw3d"]]) |
                {"n_gt": int(sum(per_scene[s]["per_class3d"].get(c, {}).get("n_gt", 0) for s in m)),
                 "n_pred": int(sum(per_scene[s]["per_class3d"].get(c, {}).get("n_pred", 0) for s in m))}
                for c in classes}
            groups[g] = agg
        for s in per_scene:
            per_scene[s].pop("_pc_raw3d")
        summary[tag] = {"per_scene": per_scene, "groups": groups}
        print_summary(tag, per_scene, groups)
    return summary


def print_summary(tag: str, per_scene: Mapping[str, Any], groups: Mapping[str, Any]) -> None:
    """Print one tag's per-scene, per-group and per-class tables (markdown).

    Parameters
    ----------
    tag : str
        Checkpoint tag.
    per_scene : mapping
        :func:`aggregate`'s ``per_scene`` block.
    groups : mapping
        :func:`aggregate`'s ``groups`` block.
    """
    print(f"\n### {tag}: per scene (DetA / HOTA)")
    print("| scene | cams | GT rows | pred rows (>=0.2) | NVIDIA official | missing=0 | pooled 3D | pooled BEV |")
    print("|---|---|---|---|---|---|---|---|")
    for s, v in per_scene.items():
        o = v["official"]
        print(f"| W{s} | {v['cams']} | {v['n_gt']} | {v['n_pred']} | {o.get('DetA', float('nan')):.2f} / "
              f"{o.get('HOTA', float('nan')):.2f} | {v['missing0']['DetA']:.2f} / {v['missing0']['HOTA']:.2f} | "
              f"{v['pooled3d']['DetA']:.2f} / {v['pooled3d']['HOTA']:.2f} | "
              f"{v['pooledBEV']['DetA']:.2f} / {v['pooledBEV']['HOTA']:.2f} |")
    print(f"\n### {tag}: groups (DetA / HOTA)")
    print("| group | GT rows | NVIDIA official | missing=0 | pooled 3D | pooled BEV |")
    print("|---|---|---|---|---|---|")
    for g, a in groups.items():
        print(f"| {g} | {a['n_gt']} | {a['official']['DetA']:.2f} / {a['official']['HOTA']:.2f} | "
              f"{a['missing0']['DetA']:.2f} / {a['missing0']['HOTA']:.2f} | "
              f"{a['pooled3d']['DetA']:.2f} / {a['pooled3d']['HOTA']:.2f} | "
              f"{a['pooledBEV']['DetA']:.2f} / {a['pooledBEV']['HOTA']:.2f} |")
    if "all" in groups:
        print(f"\n### {tag}: per-class pooled 3D DetA / HOTA (all scenes)")
        for c, v in groups["all"]["per_class_pooled3d"].items():
            print(f"  {c:13s} nGT {v['n_gt']:7d} nPred {v['n_pred']:7d}  DetA {v.get('DetA', 0):6.2f}  "
                  f"HOTA {v.get('HOTA', 0):6.2f}  DetRe {v.get('DetRe', 0):6.2f}  DetPr {v.get('DetPr', 0):6.2f}")


def run_aggregate(args: argparse.Namespace) -> int:
    """The ``aggregate`` subcommand: build the summary and write it as JSON.

    Parameters
    ----------
    args : argparse.Namespace
        ``out_root``, ``eval_root``, ``pred_root``, ``tags``, ``summary``.

    Returns
    -------
    int
        Exit status 0.

    Raises
    ------
    FileExistsError
        If ``--summary`` exists.
    """
    if args.summary.exists():
        raise FileExistsError(args.summary)
    eval_root = args.eval_root if args.eval_root is not None else args.out_root / "eval"
    pred_root = args.pred_root if args.pred_root is not None else args.out_root
    summary = aggregate(eval_root, pred_root, args.tags)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary, "x") as f:
        json.dump(summary, f, indent=1)
    return 0


# --------------------------------------------------------------------------------------------- main


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments without the program name; ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The subcommand (``command``) and its options.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gt", help="write the 11-column GT of a frame window")
    p.add_argument("--scene-dir", type=Path, required=True, help="<data>/<split>/<scene> with ground_truth.json")
    p.add_argument("--scene-id", type=int, required=True, help="integer written in the first column")
    p.add_argument("--start", type=int, default=0, help="first frame (0)")
    p.add_argument("--end", type=int, required=True, help="exclusive")
    p.add_argument("--step", type=int, default=1, help="frame stride (1)")
    p.add_argument("--out", type=Path, required=True, help="output text file (must not exist)")

    p = sub.add_parser("hota", help="HOTA with BEV-IoU or 3D-IoU similarity (EVA/GaugeAlign style)")
    p.add_argument("--gt", type=Path, required=True, help="11-column GT")
    p.add_argument("--pred", type=Path, required=True, help="11-column predictions")
    p.add_argument("--sim", choices=("bev", "3d"), required=True, help="footprint IoU or 3D IoU")
    p.add_argument("--out", type=Path, default=None, help="JSON result (must not exist); printed only if omitted")
    p.add_argument("--no-raw-arrays", action="store_true",
                   help="first-version format: without det_pooled_raw / per_class_raw_arrays")
    p.add_argument("--no-bev-prefilter", action="store_true",
                   help="first-version BEV IoU loop over every pair (same result, slower)")

    p = sub.add_parser("score", help="NVIDIA evaluate_aicity_mtmc (3D IoU) plus present-class and pooled scores")
    p.add_argument("--gt", type=Path, required=True, help="11-column GT")
    p.add_argument("--pred", type=Path, required=True, help="11-column predictions")
    p.add_argument("--scene-ids", type=Path, required=True, help='JSON {"<scene id>": "<scene name>", ...}')
    p.add_argument("--out-dir", type=Path, required=True, help="evaluation folder (must not exist)")
    p.add_argument("--num-frames-to-eval", type=int, default=9000, help="frames evaluated (9000)")
    p.add_argument("--no-raw-arrays", action="store_true",
                   help="first-version format: without det_pooled_raw / per_class_raw_arrays")

    p = sub.add_parser("aggregate", help="12-scene summary per checkpoint tag")
    p.add_argument("--out-root", type=Path, required=True,
                   help="the runner's --out-dir (eval/ and <tag>/ below it unless overridden)")
    p.add_argument("--tags", nargs="+", required=True, help="checkpoint tags")
    p.add_argument("--summary", type=Path, required=True, help="output JSON (must not exist)")
    p.add_argument("--eval-root", type=Path, default=None, help="score folders (default <out-root>/eval)")
    p.add_argument("--pred-root", type=Path, default=None, help="runner outputs with meta (default <out-root>)")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one subcommand.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments without the program name; ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status 0 (errors raise).
    """
    args = parse_args(argv)
    return {"gt": run_gt, "hota": run_hota, "score": run_score, "aggregate": run_aggregate}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
