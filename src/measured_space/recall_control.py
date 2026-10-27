"""Certified recall for an automatic 3D labeller: conformal risk control of the false-negative rate.

The gate of :mod:`measured_space.audit` certifies ``P(wrong | accepted)``. That guarantee is conditioned on a
label existing, so it cannot see an object the labeller never found (``docs/LEARNINGS.md``, L16, aim A7). This
module is the missing-label counterpart. It chooses the labeller's score threshold so that the expected share of
present ground-truth objects it misses in a new scene stays at or below a target ``alpha``.

**Method.** Conformal risk control (CRC): Angelopoulos, Bates, Fisch, Lei and Schuster, "Conformal Risk Control",
ICLR 2024, arXiv:2208.02814. It is used as published, and false-negative-rate control is the paper's own example
(Section 3.1, Eq. 18). Nothing here is new. The paper's statement (Section 1.1 and Eq. 4; Theorem 1 in Section
2.1; Theorem 2 in Section 2.2): let ``L_1 .. L_{n+1}`` be exchangeable random functions of ``lambda``,
non-increasing, right-continuous, with ``sup L_i <= B`` and ``L_i(lambda_max) <= alpha``. Let ``R_n`` be the mean
of the ``n`` calibration losses, and::

    lambda_hat = inf {lambda : n / (n + 1) * R_n(lambda) + B / (n + 1) <= alpha}    (lambda_max if empty)

Then ``E[L_{n+1}(lambda_hat)] <= alpha`` (Theorem 1). If the losses are also i.i.d., non-negative and have no
jump at any fixed ``lambda`` with positive probability, then ``E[L_{n+1}(lambda_hat)] >= alpha - 2B / (n + 1)``
(Theorem 2), so with many scenes the bound is tight.

1. **Calibration unit: one scene.** An ARKitScenes video, or a Smart Spaces scene with all its cameras. Frames
   of one video share objects, layout and capture, so they are not exchangeable and are never units.
2. **Loss** (:func:`coverage_scores`, :func:`loss_matrix`). For a scene and a score threshold ``t``, ``L(t)``
   is the scene's false-negative rate: the share of its population object-views (one per object per view in
   which it is present) that **no** box with ``score >= t`` covers. A box covers an object when the 2D IoU
   (:func:`~measured_space.matching.box_iou_2d`) of the box and the object's visible 2D box
   (:func:`~measured_space.pipeline.visible_box_edges`) is at least ``min_iou`` (0.3, the pipeline's 2D
   association threshold). There is no one-to-one constraint. Each object therefore has a *coverage score*, the
   highest score among the boxes that cover it (``-inf`` if none), and it is found at ``t`` exactly when its
   coverage score is ``>= t``. Lowering ``t`` keeps a superset of the boxes, so ``L`` is non-decreasing and
   left-continuous in ``t`` and lies in [0, 1] (``B = 1``). In the paper's orientation, ``lambda = -t``, it is
   non-increasing and right-continuous, as Theorem 1 requires.

   The population is ``present`` (the pipeline's definition, :func:`~measured_space.qa.visible_ids`: a visible
   2D box and a visible fraction above the presence threshold, from the reference depth) or
   ``clearly_visible`` (:func:`~measured_space.pipeline.clearly_visible`).
3. **Threshold** (:func:`crc_threshold`). ``t_hat = max {t >= t_min : sum_i L_i(t) + B <= alpha (n + 1)}``,
   the paper's condition multiplied by ``n + 1``. ``R_n`` is a step function that jumps just above each
   calibration coverage score, so the maximum is ``t_min`` or one of those scores, and only those are tested.
   ``t_min`` is the lowest score the predictions keep (their file's ``score_threshold``), the paper's
   ``lambda_max``: below it the boxes were never written. If no candidate qualifies, ``t_hat = t_min`` and the
   result is flagged infeasible. Two limits follow from the formula:

   - feasibility needs ``B / (n + 1) <= alpha``, i.e. at least ``1 / alpha - 1`` calibration scenes (19 for
     ``alpha = 0.05``, 9 for 0.10, 4 for 0.20), and more the more the labeller misses;
   - Theorem 1 assumes ``L_i(t_min) <= alpha``: even keeping every box it wrote, the labeller misses at most
     ``alpha`` of every scene. An object no box covers at ``t_min`` is beyond any threshold. Each scene's floor
     ``L_i(t_min)`` is reported, and where floors exceed ``alpha`` the guarantee is not claimed.
4. **What is guaranteed.** ``E[L_test(t_hat)] <= alpha`` for a new scene exchangeable with the calibration
   scenes. The expectation runs over calibration sets and test scenes. It is a per-scene rate, so every scene
   weighs the same whatever its number of objects. It is marginal: single scenes can exceed ``alpha``, and so can
   strata such as far or occluded objects.
5. **Evaluation** (:func:`leave_one_scene_out`). Each scene is held out in turn, ``t_hat`` is calibrated on the
   other ``N - 1``, and the held-out scene's FNR at it is recorded. **The mean of these held-out FNRs is at most
   ``alpha`` by construction** whenever the mean floor over all ``N`` scenes is. This is the argument of
   Theorem 1's proof applied to the ``N`` scenes themselves: every fold's qualifying thresholds also qualify for
   the ``N``-scene risk, so each held-out loss is at most its loss at the ``N``-scene threshold, whose mean is at
   most ``alpha``. The mean therefore checks the implementation, not the guarantee, and says nothing about whether
   new scenes are exchangeable with these. (:func:`simulate_guarantee` tests the guarantee itself, on fresh
   scenes.) What leave-one-scene-out adds is the spread: the per-scene distribution, the share of scenes above
   ``alpha``, how much ``t_hat`` moves between folds, and the cost. The interval of the mean is a percentile
   bootstrap over scenes, descriptive only, since the folds share calibration scenes. The deployable threshold is
   CRC on all ``N`` scenes.
6. **Cost** (:func:`scene_table_at`). Boxes per view at ``t_hat`` against at the published threshold
   (:data:`PUBLISHED_THRESHOLD`, 0.25), and the share of kept boxes whose IoU with every present annotated object
   is below ``min_iou``. They are called *unmatched to annotation*, not false: ARKitScenes annotates 17
   furniture classes only, so a box on a lamp is unmatched but real.
7. **One-to-one FNR** (:func:`one_to_one_found`), for comparison with the pipeline's recall: Hungarian
   assignment of the kept boxes to every present object with the rule of
   :func:`~measured_space.matching.match_detections_2d` (summed IoU over pairs with ``IoU >= min_iou``,
   class-agnostic). It is never below the coverage FNR and is not monotone in ``t``, so it is reported, not
   controlled.
8. **Conditional breakdown** (:func:`conditional_table`). Held-out miss rate at each scene's ``t_hat`` by the
   camera range of the ground-truth box centre (0-10, 10-20, 20-45, 45+ m, as ``scripts/compare_association.py``),
   by visible fraction and by class. It shows where the marginal guarantee hides misses.
9. **Per-stratum (Mondrian) CRC** (:func:`mondrian_loso`, optional). A separate CRC per stratum, with each
   scene's loss restricted to its objects in the stratum and scenes without such objects left out. The guarantee
   then holds per stratum, for scenes that have objects in it. The strata are ground-truth attributes (range of
   the ground-truth box, visible fraction, class). Applying per-stratum thresholds to new boxes needs a stratum
   computed without ground truth (e.g. the box's own predicted range); that step is not evaluated here.
10. **Boxes** (:class:`ScoredBox`). Any labeller enters as a mapping from view key to scored boxes, each with a
    2D box, a world 3D box, or both. CuTR lines are converted by :func:`boxes_from_cutr`, other JSON lines by
    :func:`boxes_from_record`. A box's 2D box is its own, clipped to the image as
    :func:`~measured_space.box_sources.clip_box2d` does. With ``project_3d`` it is the 2D extent of its world
    box's projected corners (:func:`~measured_space.pipeline.projected_extent`), moved to pixel-edge coordinates
    and clipped. A projected box is larger than the object's visible silhouette, so its IoU with the visible box
    is lower. Boxes with no usable 2D box (none given, a corner behind the camera, or no area inside the image)
    take no part and are counted per view.
11. **Box cap** (:func:`cap_floor`, :func:`cap_free_floor`). A labeller that writes at most ``K`` boxes per view
    (CuTR: its top 100) truncates a view that hits the cap. Below that view's lowest written score, its *cap
    floor*, a lower threshold would have added boxes that were never written. The largest cap floor over all views
    is ``t_cap``. Above ``t_cap`` the observed loss is the uncapped labeller's. At or below it, the observed loss is
    exact for the capped labeller as run (top ``K``, then the threshold) and an upper bound for an uncapped one,
    because lowering the threshold can only add boxes. CRC on the capped losses is therefore valid for the capped
    labeller, and conservative for an uncapped one, by Theorem 1 applied to the capped losses. The analysis takes a
    separate CRC floor (``crc_floor`` of :func:`run_recall_control`): just above ``t_cap`` keeps every choice in
    the uncapped region; the observation floor allows choices inside it, which are then flagged. Every threshold
    reported carries the share of views truncated at it.

Caveats stated with every result: scenes are assumed exchangeable (one capture protocol, one site population);
the guarantee is marginal over scenes; object-views of one object in one video are counted once per view.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

from measured_space.box_sources import ViewPredictions, clip_box2d
from measured_space.cubify import SCORE_THRESHOLD
from measured_space.geometry import Array, OrientedBox
from measured_space.matching import Match2DConfig, box_iou_2d
from measured_space.pipeline import (
    PipelineConfig,
    clearly_visible,
    oracle_from_records,
    projected_extent,
    view_from_record,
    visible_box_edges,
)
from measured_space.qa import visible_ids
from measured_space.scene import FrameView, ObjectInView
from measured_space.scoring import cluster_bootstrap

Box2D = tuple[float, float, float, float]
"""tuple of float: ``(xmin, ymin, xmax, ymax)`` in pixels, pixel edges at integer coordinates."""

BOUND = 1.0
"""float: the loss bound ``B``; a false-negative rate lies in [0, 1]."""

ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.20)
"""tuple of float: target expected miss rates reported by default."""

POPULATIONS: tuple[str, ...] = ("present", "clearly_visible")
"""tuple of str: which ground-truth objects count (see the module docstring, step 2)."""

PUBLISHED_THRESHOLD: float = SCORE_THRESHOLD
"""float: CuTR's published score threshold (0.25), the cost baseline."""

STRATA: tuple[str, ...] = ("range", "visible_fraction", "label")
"""tuple of str: object attributes of the conditional breakdown and of the Mondrian variant."""

RANGE_BINS_M: tuple[float, ...] = (0.0, 10.0, 20.0, 45.0, math.inf)
"""tuple of float: edges of the camera-range bins, metres (those of ``scripts/compare_association.py``)."""

VISIBLE_BINS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.6, 1.0)
"""tuple of float: edges of the visible-fraction bins (0.3 is the clearly-visible threshold)."""

STRATUM_COLUMNS: dict[str, str] = {"range": "range_bin", "visible_fraction": "visible_bin", "label": "label"}
"""dict of str to str: the object-table column that holds each stratum's bin."""

_TIE_TOL = 1e-12   # slack on the CRC condition, so an exact tie with the bound is not lost to float rounding


@dataclass(frozen=True)
class RecallConfig:
    """Every setting of recall control.

    Attributes
    ----------
    alphas : tuple of float
        Target expected miss rates, each in (0, 1) (default :data:`ALPHAS`).
    population : str
        ``"present"`` or ``"clearly_visible"`` (default ``"present"``).
    min_iou : float
        A box covers an object at 2D IoU at least this (default 0.3, ``Match2DConfig().min_iou``).
    bound : float
        The loss bound ``B`` (default 1).
    published_threshold : float
        The cost baseline's score threshold (default :data:`PUBLISHED_THRESHOLD`).
    curve_points : int
        Evenly spaced thresholds of the FNR curve between the floor and 1, besides the published one (default 96).
    mondrian : tuple of str
        Strata (from :data:`STRATA`) to run per-stratum CRC on (default none).
    n_boot : int
        Bootstrap replicates over scenes for the intervals (default 2000).
    level : float
        Two-sided level of the bootstrap intervals (default 0.95).
    seed : int
        Seed of the bootstrap (default 0).
    """

    alphas: tuple[float, ...] = ALPHAS
    population: str = "present"
    min_iou: float = Match2DConfig().min_iou
    bound: float = BOUND
    published_threshold: float = PUBLISHED_THRESHOLD
    curve_points: int = 96
    mondrian: tuple[str, ...] = ()
    n_boot: int = 2000
    level: float = 0.95
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises
        ------
        ValueError
            If an ``alpha`` is outside (0, 1), the population or a Mondrian stratum is unknown, ``min_iou`` is
            outside (0, 1], or ``bound`` is not positive.
        """
        if not self.alphas or any(not 0.0 < a < 1.0 for a in self.alphas):
            raise ValueError(f"every alpha must be in (0, 1), got {self.alphas}")
        if self.population not in POPULATIONS:
            raise ValueError(f"population must be one of {POPULATIONS}, got {self.population!r}")
        if any(s not in STRATA for s in self.mondrian):
            raise ValueError(f"Mondrian strata must be among {STRATA}, got {self.mondrian}")
        if not 0.0 < self.min_iou <= 1.0:
            raise ValueError(f"min_iou must be in (0, 1], got {self.min_iou}")
        if not self.bound > 0.0:
            raise ValueError(f"bound must be positive, got {self.bound}")


# ============================================================================================================
# 1. Boxes and views
# ============================================================================================================


@dataclass(frozen=True)
class ScoredBox:
    """One automatic box in one view, from any labeller.

    Parameters
    ----------
    score : float
        The labeller's confidence; boxes are kept at ``score >= t``.
    box2d : tuple of float or None, default None
        ``(xmin, ymin, xmax, ymax)``, full-resolution pixels, pixel edges at integer coordinates.
    box3d : OrientedBox or None, default None
        World box, metres (needed only to project it, see :func:`box_edges`).
    """

    score: float
    box2d: Box2D | None = None
    box3d: OrientedBox | None = None


@dataclass(frozen=True)
class ViewEvidence:
    """One view's present ground-truth objects and usable automatic boxes, reduced to what recall control needs.

    Parameters
    ----------
    dataset : str
        Dataset name.
    scene : str
        Scene name, the calibration unit within the dataset.
    view_key : str
        ``"<dataset>/<scene>/<camera_id>/<frame>"``.
    scores : numpy.ndarray, shape (k,)
        Score of each usable box.
    iou : numpy.ndarray, shape (k, m)
        2D IoU of each usable box with each present object's visible box.
    object_ids : tuple of str
        The ``m`` present objects, in pipeline order.
    labels : tuple of str
        Each present object's class.
    range_m : numpy.ndarray, shape (m,)
        Distance from the camera centre to each object's ground-truth box centre, metres.
    visible_fraction : numpy.ndarray, shape (m,)
        Each object's visible fraction.
    clearly_visible : numpy.ndarray of bool, shape (m,)
        Whether each object passes the question generator's referent visibility thresholds.
    n_unusable : int, default 0
        Boxes without a usable 2D box (none given, a corner behind the camera, or no area in the image).
    cap_floor : float, default -inf
        The view's lowest written score if the labeller's box cap truncated it (:func:`cap_floor`), else ``-inf``.
    """

    dataset: str
    scene: str
    view_key: str
    scores: Array
    iou: Array
    object_ids: tuple[str, ...]
    labels: tuple[str, ...]
    range_m: Array
    visible_fraction: Array
    clearly_visible: NDArray[np.bool_]
    n_unusable: int = 0
    cap_floor: float = -math.inf

    def population(self, population: str) -> NDArray[np.bool_]:
        """Which present objects belong to a population.

        Parameters
        ----------
        population : str
            ``"present"`` (all) or ``"clearly_visible"``.

        Returns
        -------
        numpy.ndarray of bool, shape (m,)
            The mask.
        """
        if population == "present":
            return np.ones(len(self.object_ids), dtype=bool)
        return np.asarray(self.clearly_visible, dtype=bool)


def boxes_from_cutr(predictions: ViewPredictions) -> list[ScoredBox]:
    """Scored boxes of one view's CuTR predictions.

    Parameters
    ----------
    predictions : ViewPredictions
        One parsed line of a CuTR predictions file (:func:`measured_space.box_sources.parse_view_predictions`).

    Returns
    -------
    list of ScoredBox
        One per box, with CuTR's 2D box and world box, in file order.
    """
    return [ScoredBox(b.score, b.box2d, b.box) for b in predictions.boxes]


def boxes_from_record(record: Mapping[str, Any]) -> list[ScoredBox]:
    """Scored boxes of one prediction line in the general format.

    Each entry of ``record["boxes"]`` needs ``score`` and may hold ``box2d`` (4 numbers) and a world box
    (``center``, ``size``, ``rotation``, keys as :func:`measured_space.pipeline.box_json`).

    Parameters
    ----------
    record : mapping
        A parsed JSON line with ``boxes``.

    Returns
    -------
    list of ScoredBox
        One per box, in file order.

    Raises
    ------
    ValueError
        If a ``box2d`` does not have 4 values or a score is not finite.
    """
    boxes = []
    for i, b in enumerate(record["boxes"]):
        score = float(b["score"])
        if not math.isfinite(score):
            raise ValueError(f"box {i} has a non-finite score")
        box2d: Box2D | None = None
        if b.get("box2d") is not None:
            values = tuple(float(v) for v in b["box2d"])
            if len(values) != 4:
                raise ValueError(f"box {i} has a box2d with {len(values)} values")
            box2d = values                                                    # type: ignore[assignment]
        box3d = None
        if all(b.get(k) is not None for k in ("center", "size", "rotation")):
            box3d = OrientedBox(b["center"], b["size"], b["rotation"])
        boxes.append(ScoredBox(score, box2d, box3d))
    return boxes


def box_edges(box: ScoredBox, view: FrameView, *, project_3d: bool = False) -> Box2D | None:
    """A box's 2D box in the view, clipped to the image, or ``None`` if it has no usable one.

    Parameters
    ----------
    box : ScoredBox
        The box.
    view : FrameView
        The view (camera, pose and image size).
    project_3d : bool, default False
        Use the 2D extent of the world box's projected corners instead of the box's own 2D box. The extent of
        :func:`~measured_space.pipeline.projected_extent` has pixel centres at integers, so it is moved by half
        a pixel to pixel-edge coordinates.

    Returns
    -------
    tuple of float or None
        ``(xmin, ymin, xmax, ymax)`` with pixel edges at integers, inside ``[0, W] x [0, H]``; ``None`` if the
        needed box is missing, a corner lies behind the camera, or nothing of it is inside the image.
    """
    if project_3d:
        if box.box3d is None:
            return None
        extent = projected_extent(view, box.box3d)
        if extent is None:
            return None
        edges: Sequence[float] = tuple(v + 0.5 for v in extent)
    else:
        if box.box2d is None:
            return None
        edges = box.box2d
    x0, y0, x1, y1 = clip_box2d(edges, view.camera.width, view.camera.height)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def view_evidence(records: Mapping[str, Any], boxes: Sequence[ScoredBox], *, project_3d: bool = False,
                  check_flags: bool = True, cap_floor: float = -math.inf) -> ViewEvidence:
    """Rebuild a stored view's present objects and pair them with automatic boxes.

    The view and its oracle objects are rebuilt with the pipeline's readers
    (:func:`~measured_space.pipeline.view_from_record`, :func:`~measured_space.pipeline.oracle_from_records`).
    Presence and clear visibility are recomputed with the view's stored question settings.

    Parameters
    ----------
    records : mapping
        A view's records as stored in a shard of ``scripts/run_labels.py`` (``view`` and ``oracle_objects``;
        other keys are ignored).
    boxes : sequence of ScoredBox
        The labeller's boxes in this view.
    project_3d : bool, default False
        Use projected world boxes (see :func:`box_edges`).
    check_flags : bool, default True
        Require the recomputed presence and clear-visibility flags to equal the stored ones.
    cap_floor : float, default -inf
        The view's cap floor (:func:`cap_floor` of every box written for it, before any filtering).

    Returns
    -------
    ViewEvidence
        The view's present objects, its usable boxes' scores and their 2D IoU with the objects.

    Raises
    ------
    ValueError
        If ``check_flags`` and a recomputed flag differs from the stored one.
    """
    view_record = records["view"]
    view = view_from_record(view_record)
    qa = PipelineConfig.from_json(view_record["config"]).qa
    seen, _ = oracle_from_records(records["oracle_objects"])
    present_ids = visible_ids(seen, qa)
    if check_flags:
        stored_present = {r["object_id"] for r in records["oracle_objects"] if r["present"]}
        stored_clear = {r["object_id"] for r in records["oracle_objects"] if r["clearly_visible"]}
        clear_ids = {o.obj.object_id for o in seen if clearly_visible(o, qa)}
        if stored_present != present_ids or stored_clear != clear_ids:
            raise ValueError(f"{view_record['view_key']}: recomputed presence or clear visibility differs from the "
                             "stored flags")
    # A present object always has a visible 2D box (qa.visible_ids), so the second test only narrows the type.
    present: list[ObjectInView] = [o for o in seen if o.obj.object_id in present_ids and o.bbox2d is not None]
    object_boxes = np.array([visible_box_edges(o.bbox2d, view_record["visibility"]) for o in present
                             if o.bbox2d is not None], dtype=np.float64).reshape(-1, 4)
    usable = [(b.score, e) for b in boxes if (e := box_edges(b, view, project_3d=project_3d)) is not None]
    scores = np.array([s for s, _ in usable], dtype=np.float64)
    box_array = np.array([e for _, e in usable], dtype=np.float64).reshape(-1, 4)
    iou = box_iou_2d(box_array, object_boxes) if len(usable) and len(present) else np.zeros((len(usable),
                                                                                              len(present)))
    centre = view.camera_center
    return ViewEvidence(
        str(view_record["dataset"]), str(view_record["scene"]), str(view_record["view_key"]), scores, iou,
        tuple(o.obj.object_id for o in present), tuple(o.obj.label for o in present),
        np.array([float(np.linalg.norm(o.obj.box.center - centre)) for o in present], dtype=np.float64),
        np.array([o.visible_fraction for o in present], dtype=np.float64),
        np.array([clearly_visible(o, qa) for o in present], dtype=bool), len(boxes) - len(usable),
        float(cap_floor))


# ============================================================================================================
# 2. The loss and conformal risk control
# ============================================================================================================


def coverage_scores(scores: ArrayLike, iou: ArrayLike, min_iou: float) -> Array:
    """Each object's coverage score: the highest score of a box that covers it.

    Parameters
    ----------
    scores : array_like, shape (k,)
        Box scores.
    iou : array_like, shape (k, m)
        2D IoU of each box with each object.
    min_iou : float
        A box covers an object at IoU at least this.

    Returns
    -------
    numpy.ndarray, shape (m,)
        The coverage scores; ``-inf`` for an object no box covers. An object is found at threshold ``t``
        exactly when its coverage score is at least ``t``.

    Raises
    ------
    ValueError
        If ``iou`` is not of shape ``(k, m)``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    iou = np.asarray(iou, dtype=np.float64)
    if iou.ndim != 2 or iou.shape[0] != len(scores):
        raise ValueError(f"iou must have shape ({len(scores)}, m), got {iou.shape}")
    if not len(scores) or not iou.shape[1]:
        return np.full(iou.shape[1], -np.inf)
    return np.where(iou >= min_iou, scores[:, None], -np.inf).max(axis=0)


def candidate_thresholds(coverages: Iterable[ArrayLike], floor: float) -> Array:
    """The thresholds at which the calibration risk can be largest while still meeting a bound.

    Parameters
    ----------
    coverages : iterable of array_like
        Coverage scores of each scene's objects.
    floor : float
        The lowest threshold at which boxes are observed (``t_min``).

    Returns
    -------
    numpy.ndarray
        ``floor`` and every distinct coverage score at or above it, ascending.
    """
    values = [np.asarray(c, dtype=np.float64) for c in coverages]
    above = [c[np.isfinite(c) & (c >= floor)] for c in values]
    return np.unique(np.concatenate([np.array([floor], dtype=np.float64), *above]))


def loss_matrix(coverages: Sequence[ArrayLike], thresholds: ArrayLike) -> Array:
    """Every scene's false-negative rate at every threshold.

    Parameters
    ----------
    coverages : sequence of array_like
        Coverage scores of each scene's objects; every scene needs at least one object.
    thresholds : array_like, shape (K,)
        Score thresholds.

    Returns
    -------
    numpy.ndarray, shape (N, K)
        ``L[i, k]``: the share of scene ``i``'s objects with coverage score below ``thresholds[k]``.

    Raises
    ------
    ValueError
        If a scene has no objects.
    """
    t = np.asarray(thresholds, dtype=np.float64)
    out = np.empty((len(coverages), len(t)))
    for i, c in enumerate(coverages):
        c = np.sort(np.asarray(c, dtype=np.float64))
        if not len(c):
            raise ValueError(f"scene {i} has no objects; its false-negative rate is undefined")
        out[i] = np.searchsorted(c, t, side="left") / len(c)
    return out


def crc_index(loss_sum: ArrayLike, n: int, alpha: float, bound: float = BOUND) -> int:
    """Index of the CRC threshold among ascending candidates, or -1 if none qualifies.

    The paper's condition ``n / (n + 1) * R_n(t) + B / (n + 1) <= alpha`` is checked as
    ``sum_i L_i(t) + B <= alpha (n + 1)``, which is the same inequality multiplied by ``n + 1``.

    Parameters
    ----------
    loss_sum : array_like, shape (K,)
        Summed calibration loss at each candidate threshold, thresholds ascending (so non-decreasing).
    n : int
        Number of calibration scenes.
    alpha : float
        Target expected loss.
    bound : float, default BOUND
        The loss bound ``B``.

    Returns
    -------
    int
        The largest index of the leading run of candidates that meet the condition (the paper's infimum over
        ``lambda = -t``); -1 if the first candidate already fails.
    """
    ok = np.asarray(loss_sum, dtype=np.float64) + bound <= alpha * (n + 1) + _TIE_TOL * (n + 1)
    failing = np.flatnonzero(~ok)
    return int(failing[0]) - 1 if failing.size else len(ok) - 1


@dataclass(frozen=True)
class CRCResult:
    """A conformal-risk-control threshold and how it was reached.

    Parameters
    ----------
    threshold : float
        ``t_hat``: keep boxes with ``score >= threshold``. The floor when infeasible.
    feasible : bool
        Whether some threshold met the condition. If not, no guarantee is claimed.
    n_cal : int
        Number of calibration scenes.
    alpha : float
        Target expected loss.
    cal_risk : float
        Mean calibration loss at ``threshold``.
    adjusted_risk : float
        ``n / (n + 1) * cal_risk + B / (n + 1)``, at most ``alpha`` when feasible.
    floor : float
        The lowest threshold tested.
    floor_risk : float
        Mean calibration loss at the floor.
    """

    threshold: float
    feasible: bool
    n_cal: int
    alpha: float
    cal_risk: float
    adjusted_risk: float
    floor: float
    floor_risk: float


def crc_threshold(coverages: Sequence[ArrayLike], alpha: float, *, floor: float, bound: float = BOUND) -> CRCResult:
    """The largest score threshold whose CRC-adjusted calibration miss rate is at most ``alpha``.

    Parameters
    ----------
    coverages : sequence of array_like
        Coverage scores of each calibration scene's objects (at least one object per scene).
    alpha : float
        Target expected false-negative rate.
    floor : float
        ``t_min``, the lowest threshold at which boxes are observed.
    bound : float, default BOUND
        The loss bound ``B``.

    Returns
    -------
    CRCResult
        ``t_hat`` and its calibration risk.

    Raises
    ------
    ValueError
        If there are no calibration scenes.
    """
    n = len(coverages)
    if n < 1:
        raise ValueError("CRC needs at least one calibration scene")
    thresholds = candidate_thresholds(coverages, floor)
    loss_sum = loss_matrix(coverages, thresholds).sum(axis=0)
    index = crc_index(loss_sum, n, alpha, bound)
    k = max(index, 0)
    risk = float(loss_sum[k] / n)
    return CRCResult(float(thresholds[k]), index >= 0, n, alpha, risk, n / (n + 1) * risk + bound / (n + 1),
                     float(floor), float(loss_sum[0] / n))


def leave_one_scene_out(coverages: Mapping[str, ArrayLike], alpha: float, *, floor: float,
                        bound: float = BOUND) -> pd.DataFrame:
    """Hold out each scene in turn, calibrate ``t_hat`` on the others, and record the held-out FNR.

    The loss of every scene is computed once at every candidate threshold of all scenes. Adding the held-out
    scene's coverage scores to the candidates does not change any fold's ``t_hat``: the calibration risk is
    constant between consecutive calibration scores, so the largest qualifying candidate is still a calibration
    score (or the floor).

    Parameters
    ----------
    coverages : mapping of str to array_like
        Scene -> coverage scores of its objects (at least one each); at least two scenes.
    alpha : float
        Target expected false-negative rate.
    floor : float
        ``t_min``, the lowest threshold at which boxes are observed.
    bound : float, default BOUND
        The loss bound ``B``.

    Returns
    -------
    pandas.DataFrame
        One row per scene, in the mapping's order: ``scene``, ``n_objects``, ``n_cal``, ``threshold``
        (``t_hat`` of its fold), ``feasible``, ``cal_risk`` (mean calibration FNR at ``t_hat``), ``heldout_fnr``
        and ``floor_fnr`` (the held-out scene's FNR at the floor).

    Raises
    ------
    ValueError
        If there are fewer than two scenes.
    """
    scenes = list(coverages)
    if len(scenes) < 2:
        raise ValueError("leave-one-scene-out needs at least two scenes")
    values = [np.asarray(coverages[s], dtype=np.float64) for s in scenes]
    thresholds = candidate_thresholds(values, floor)
    losses = loss_matrix(values, thresholds)
    total = losses.sum(axis=0)
    n = len(scenes) - 1
    rows = []
    for j, scene in enumerate(scenes):
        loss_sum = total - losses[j]
        index = crc_index(loss_sum, n, alpha, bound)
        k = max(index, 0)
        rows.append({"scene": scene, "n_objects": len(values[j]), "n_cal": n, "threshold": float(thresholds[k]),
                     "feasible": index >= 0, "cal_risk": float(loss_sum[k] / n),
                     "heldout_fnr": float(losses[j, k]), "floor_fnr": float(losses[j, 0])})
    return pd.DataFrame(rows)


def cap_floor(scores: ArrayLike, max_boxes: int | None) -> float:
    """A view's cap floor: its lowest written score if the labeller's box cap truncated it.

    Parameters
    ----------
    scores : array_like, shape (k,)
        Scores of every box written for the view, before any filtering.
    max_boxes : int or None
        The labeller's cap on boxes per view (CuTR: 100); ``None`` or 0 for no cap.

    Returns
    -------
    float
        ``min(scores)`` if ``k >= max_boxes``, else ``-inf``. Below it the view may lack boxes that a lower
        threshold would have kept.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not max_boxes or len(scores) < max_boxes:
        return -math.inf
    return float(scores.min())


def truncation_threshold(views: Sequence[ViewEvidence]) -> float:
    """``t_cap``: the largest cap floor over the views.

    Parameters
    ----------
    views : sequence of ViewEvidence
        The views.

    Returns
    -------
    float
        At thresholds above it no view is truncated; ``-inf`` if none was capped.
    """
    return max((v.cap_floor for v in views), default=-math.inf)


def share_truncated(views: Sequence[ViewEvidence], threshold: float) -> float:
    """Share of views whose loss at a threshold the box cap leaves unobserved (``threshold <= cap floor``).

    Parameters
    ----------
    views : sequence of ViewEvidence
        The views.
    threshold : float
        Score threshold; ``-inf`` gives the share of capped views.

    Returns
    -------
    float
        The share; NaN for no views.
    """
    floors = np.array([v.cap_floor for v in views], dtype=np.float64)
    if not len(floors):
        return math.nan
    return float(np.mean(np.isfinite(floors) & (floors >= threshold)))


def cap_free_floor(floor: float, t_cap: float) -> float:
    """The lowest threshold at which no view is truncated by the box cap.

    Parameters
    ----------
    floor : float
        The observation floor.
    t_cap : float
        :func:`truncation_threshold` (``-inf`` without capped views).

    Returns
    -------
    float
        ``floor`` if ``t_cap < floor``, else the next float above ``t_cap``. At it, a box kept in a capped view
        scores above that view's cap floor, so no unwritten box could have been kept.
    """
    return float(floor) if t_cap < floor else float(np.nextafter(t_cap, math.inf))


# ============================================================================================================
# 3. Tables: objects, boxes, one-to-one recall, cost
# ============================================================================================================


def range_bin(distance: float) -> str:
    """The name of the camera-range bin a distance falls in.

    Parameters
    ----------
    distance : float
        Camera-to-object distance, metres.

    Returns
    -------
    str
        ``"0_10m"``, ``"10_20m"``, ``"20_45m"`` or ``"45m_plus"`` for the default bins.
    """
    for lo, hi in zip(RANGE_BINS_M[:-1], RANGE_BINS_M[1:]):
        if lo <= distance < hi:
            return f"{lo:g}_{hi:g}m" if math.isfinite(hi) else f"{lo:g}m_plus"
    return f"{RANGE_BINS_M[-2]:g}m_plus"


def visible_bin(fraction: float) -> str:
    """The name of the visible-fraction bin a fraction falls in (the last bin includes 1).

    Parameters
    ----------
    fraction : float
        Visible fraction, in [0, 1].

    Returns
    -------
    str
        ``"0-0.1"``, ``"0.1-0.3"``, ``"0.3-0.6"`` or ``"0.6-1"`` for the default bins.
    """
    edges = VISIBLE_BINS
    index = min(max(int(np.searchsorted(edges, fraction, side="right")) - 1, 0), len(edges) - 2)
    return f"{edges[index]:g}-{edges[index + 1]:g}"


def object_table(views: Sequence[ViewEvidence], min_iou: float) -> pd.DataFrame:
    """One row per present object per view, with its coverage score and strata.

    Parameters
    ----------
    views : sequence of ViewEvidence
        The views.
    min_iou : float
        Coverage IoU threshold.

    Returns
    -------
    pandas.DataFrame
        ``dataset, scene, view_key, object_id, label, range_m, visible_fraction, clearly_visible, coverage,
        range_bin, visible_bin``.
    """
    parts = []
    for v in views:
        if not v.object_ids:
            continue
        parts.append(pd.DataFrame({
            "dataset": v.dataset, "scene": v.scene, "view_key": v.view_key, "object_id": list(v.object_ids),
            "label": list(v.labels), "range_m": v.range_m, "visible_fraction": v.visible_fraction,
            "clearly_visible": v.clearly_visible, "coverage": coverage_scores(v.scores, v.iou, min_iou)}))
    columns = ["dataset", "scene", "view_key", "object_id", "label", "range_m", "visible_fraction",
               "clearly_visible", "coverage"]
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    table["range_bin"] = [range_bin(d) for d in table["range_m"]]
    table["visible_bin"] = [visible_bin(f) for f in table["visible_fraction"]]
    return table


def box_table(views: Sequence[ViewEvidence], min_iou: float) -> pd.DataFrame:
    """One row per usable box, with whether it overlaps any present annotated object.

    Parameters
    ----------
    views : sequence of ViewEvidence
        The views.
    min_iou : float
        A box is matched to annotation at IoU at least this with some present object.

    Returns
    -------
    pandas.DataFrame
        ``dataset, scene, view_key, score, max_iou, matched_annotation``.
    """
    parts = []
    for v in views:
        if not len(v.scores):
            continue
        best = v.iou.max(axis=1) if v.iou.shape[1] else np.zeros(len(v.scores))
        parts.append(pd.DataFrame({"dataset": v.dataset, "scene": v.scene, "view_key": v.view_key,
                                   "score": v.scores, "max_iou": best, "matched_annotation": best >= min_iou}))
    columns = ["dataset", "scene", "view_key", "score", "max_iou", "matched_annotation"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def view_table(views: Sequence[ViewEvidence]) -> pd.DataFrame:
    """One row per view: its box and object counts.

    Parameters
    ----------
    views : sequence of ViewEvidence
        The views.

    Returns
    -------
    pandas.DataFrame
        ``dataset, scene, view_key, n_boxes`` (usable), ``n_unusable, n_present, n_clearly_visible, cap_floor``
        (``-inf`` for a view the box cap did not truncate).
    """
    return pd.DataFrame([{"dataset": v.dataset, "scene": v.scene, "view_key": v.view_key, "n_boxes": len(v.scores),
                          "n_unusable": v.n_unusable, "n_present": len(v.object_ids),
                          "n_clearly_visible": int(np.sum(v.clearly_visible)), "cap_floor": v.cap_floor}
                         for v in views],
                        columns=["dataset", "scene", "view_key", "n_boxes", "n_unusable", "n_present",
                                 "n_clearly_visible", "cap_floor"])


def scene_coverages(objects: pd.DataFrame, population: str) -> dict[str, Array]:
    """Each scene's coverage scores over its population objects.

    Parameters
    ----------
    objects : pandas.DataFrame
        Output of :func:`object_table`.
    population : str
        ``"present"`` or ``"clearly_visible"``.

    Returns
    -------
    dict of str to numpy.ndarray
        Scene -> coverage scores, scenes in sorted order; scenes without population objects are left out.
    """
    rows = objects if population == "present" else objects[objects["clearly_visible"].astype(bool)]
    return {str(scene): group["coverage"].to_numpy(np.float64)
            for scene, group in rows.groupby("scene", sort=True)}


def one_to_one_found(scores: ArrayLike, iou: ArrayLike, threshold: float, min_iou: float) -> NDArray[np.bool_]:
    """Which objects a one-to-one assignment of the kept boxes pairs.

    The rule of :func:`~measured_space.matching.match_detections_2d`: Hungarian assignment maximising the summed
    IoU over pairs with ``IoU >= min_iou`` (cost ``1 - IoU`` for such a pair and 1 otherwise), class-agnostic.

    Parameters
    ----------
    scores : array_like, shape (k,)
        Box scores.
    iou : array_like, shape (k, m)
        2D IoU of each box with each object.
    threshold : float
        Boxes with ``score >= threshold`` are kept.
    min_iou : float
        Smallest IoU of a pair.

    Returns
    -------
    numpy.ndarray of bool, shape (m,)
        True for objects paired with a kept box.
    """
    scores = np.asarray(scores, dtype=np.float64)
    iou = np.asarray(iou, dtype=np.float64)
    found = np.zeros(iou.shape[1], dtype=bool)
    overlap = iou[scores >= threshold]
    if not overlap.size:
        return found
    feasible = overlap >= min_iou
    rows, cols = linear_sum_assignment(np.where(feasible, 1.0 - overlap, 1.0))
    found[cols[feasible[rows, cols]]] = True
    return found


def scene_table_at(views: Sequence[ViewEvidence], thresholds: Mapping[str, float], population: str,
                   min_iou: float) -> pd.DataFrame:
    """Miss rates and cost of each scene at its own threshold.

    Parameters
    ----------
    views : sequence of ViewEvidence
        The views; those of scenes missing from ``thresholds`` are skipped.
    thresholds : mapping of str to float
        Scene -> score threshold.
    population : str
        ``"present"`` or ``"clearly_visible"``.
    min_iou : float
        Coverage and pairing IoU threshold.

    Returns
    -------
    pandas.DataFrame
        One row per scene: ``scene, threshold, n_views, n_objects`` (population), ``fnr`` (coverage),
        ``fnr_one_to_one``, ``n_kept`` (boxes with score at or above the threshold), ``boxes_per_view``,
        ``n_unmatched`` (kept boxes matched to no present annotated object) and ``unmatched_share``.
    """
    acc: dict[str, dict[str, float]] = {}
    for v in views:
        t = thresholds.get(v.scene)
        if t is None:
            continue
        a = acc.setdefault(v.scene, {"threshold": t, "n_views": 0, "n_objects": 0, "n_missed": 0,
                                     "n_missed_one_to_one": 0, "n_kept": 0, "n_unmatched": 0})
        mask = v.population(population)
        kept = v.scores >= t
        covered = coverage_scores(v.scores, v.iou, min_iou) >= t
        paired = one_to_one_found(v.scores, v.iou, t, min_iou)
        matched = (v.iou[kept].max(axis=1) >= min_iou) if v.iou.shape[1] else np.zeros(int(kept.sum()), bool)
        a["n_views"] += 1
        a["n_objects"] += int(mask.sum())
        a["n_missed"] += int((mask & ~covered).sum())
        a["n_missed_one_to_one"] += int((mask & ~paired).sum())
        a["n_kept"] += int(kept.sum())
        a["n_unmatched"] += int((~matched).sum())
    rows = []
    for scene in sorted(acc):
        a = acc[scene]
        n_obj, n_kept = a["n_objects"], a["n_kept"]
        rows.append({"scene": scene, "threshold": a["threshold"], "n_views": int(a["n_views"]),
                     "n_objects": int(n_obj), "fnr": a["n_missed"] / n_obj if n_obj else math.nan,
                     "fnr_one_to_one": a["n_missed_one_to_one"] / n_obj if n_obj else math.nan,
                     "n_kept": int(n_kept), "boxes_per_view": n_kept / a["n_views"],
                     "n_unmatched": int(a["n_unmatched"]),
                     "unmatched_share": a["n_unmatched"] / n_kept if n_kept else math.nan})
    return pd.DataFrame(rows, columns=["scene", "threshold", "n_views", "n_objects", "fnr", "fnr_one_to_one",
                                       "n_kept", "boxes_per_view", "n_unmatched", "unmatched_share"])


def pooled_cost(scenes: pd.DataFrame) -> dict[str, float]:
    """Cost and miss rates pooled over scenes.

    Parameters
    ----------
    scenes : pandas.DataFrame
        Output of :func:`scene_table_at`.

    Returns
    -------
    dict of str to float
        ``mean_fnr`` and ``mean_fnr_one_to_one`` (per-scene means, every scene weighted equally),
        ``pooled_fnr`` (over objects), ``boxes_per_view`` (over views) and ``unmatched_share`` (over kept
        boxes).
    """
    n_obj, n_views, n_kept = (float(scenes[c].sum()) for c in ("n_objects", "n_views", "n_kept"))
    missed = float((scenes["fnr"] * scenes["n_objects"]).sum())
    return {"mean_fnr": float(scenes["fnr"].mean()), "mean_fnr_one_to_one": float(scenes["fnr_one_to_one"].mean()),
            "pooled_fnr": missed / n_obj if n_obj else math.nan,
            "boxes_per_view": n_kept / n_views if n_views else math.nan,
            "unmatched_share": float(scenes["n_unmatched"].sum()) / n_kept if n_kept else math.nan}


# ============================================================================================================
# 4. Intervals, conditional breakdown, Mondrian variant, curve
# ============================================================================================================


def bootstrap_interval(values: ArrayLike, clusters: ArrayLike, n_boot: int, seed: int | Sequence[int],
                       level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of a pooled mean, resampling whole clusters (scenes).

    Parameters
    ----------
    values : array_like, shape (n,)
        One value per item.
    clusters : array_like, shape (n,)
        The cluster of each item; one value per scene for a mean over scenes.
    n_boot : int
        Replicates.
    seed : int or sequence of int
        Seed of the replicates.
    level : float, default 0.95
        Two-sided level.

    Returns
    -------
    lower, upper : float
        The interval; NaN for no items.
    """
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return math.nan, math.nan
    boot = cluster_bootstrap(values, np.asarray(clusters), n_boot, np.random.default_rng(seed))
    lower, upper = np.quantile(boot, [(1 - level) / 2, (1 + level) / 2])
    return float(lower), float(upper)


def conditional_table(objects: pd.DataFrame, thresholds: Mapping[str, float], config: RecallConfig,
                      salt: int = 0) -> pd.DataFrame:
    """Held-out miss rate at each scene's threshold, by range, visible fraction and class.

    Parameters
    ----------
    objects : pandas.DataFrame
        Population rows of :func:`object_table`.
    thresholds : mapping of str to float
        Scene -> its held-out ``t_hat``; scenes missing from it are skipped.
    config : RecallConfig
        Published threshold and bootstrap settings.
    salt : int, default 0
        Mixed into the bootstrap seed.

    Returns
    -------
    pandas.DataFrame
        One row per stratum and bin: ``stratum, bin, n_objects, n_scenes, n_missed, miss_rate`` (pooled over
        objects), ``miss_rate_lo``, ``miss_rate_hi`` (bootstrap over scenes), ``miss_rate_published`` (at the
        published threshold) and ``miss_rate_floor`` (keeping every box).
    """
    rows = objects[objects["scene"].isin(list(thresholds))]
    t = rows["scene"].map(thresholds).to_numpy(np.float64)
    coverage = rows["coverage"].to_numpy(np.float64)
    missed = pd.Series(coverage < t, index=rows.index)
    published = pd.Series(coverage < config.published_threshold, index=rows.index)
    floor = pd.Series(~np.isfinite(coverage), index=rows.index)
    out = []
    for s, stratum in enumerate(STRATA):
        column = STRATUM_COLUMNS[stratum]
        for b, (value, group) in enumerate(rows.groupby(column, sort=True)):
            idx = group.index
            lo, hi = bootstrap_interval(missed[idx].to_numpy(float), group["scene"].to_numpy(), config.n_boot,
                                        [config.seed, salt, s, b], config.level)
            out.append({"stratum": stratum, "bin": value, "n_objects": len(idx), "n_scenes": group["scene"].nunique(),
                        "n_missed": int(missed[idx].sum()), "miss_rate": float(missed[idx].mean()),
                        "miss_rate_lo": lo, "miss_rate_hi": hi,
                        "miss_rate_published": float(published[idx].mean()),
                        "miss_rate_floor": float(floor[idx].mean())})
    return pd.DataFrame(out)


def mondrian_loso(objects: pd.DataFrame, stratum: str, alpha: float, *, floor: float,
                  bound: float = BOUND) -> pd.DataFrame:
    """Per-stratum CRC: a separate leave-one-scene-out calibration within each bin of a stratum.

    Parameters
    ----------
    objects : pandas.DataFrame
        Population rows of :func:`object_table`.
    stratum : str
        One of :data:`STRATA`.
    alpha : float
        Target expected false-negative rate within each bin.
    floor : float
        ``t_min``.
    bound : float, default BOUND
        The loss bound ``B``.

    Returns
    -------
    pandas.DataFrame
        :func:`leave_one_scene_out` rows for every bin with at least two scenes that have objects in it, plus
        ``stratum`` and ``bin``.
    """
    parts = []
    for value, group in objects.groupby(STRATUM_COLUMNS[stratum], sort=True):
        coverages = {str(s): g["coverage"].to_numpy(np.float64) for s, g in group.groupby("scene", sort=True)}
        if len(coverages) < 2:
            continue
        parts.append(leave_one_scene_out(coverages, alpha, floor=floor, bound=bound).assign(stratum=stratum,
                                                                                             bin=value))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def curve_thresholds(floor: float, config: RecallConfig, extra: Sequence[float] = ()) -> Array:
    """Thresholds of the FNR curve.

    Parameters
    ----------
    floor : float
        The observation floor.
    config : RecallConfig
        Number of points and the published threshold.
    extra : sequence of float, default ()
        Further thresholds to include (e.g. the CRC floor), kept if in ``[floor, 1]``.

    Returns
    -------
    numpy.ndarray
        ``floor``, ``curve_points`` evenly spaced values up to 1 (rounded to 6 decimals), and the published
        threshold and ``extra`` where they lie in ``[floor, 1]``; ascending and distinct.
    """
    grid = np.round(np.linspace(floor, 1.0, config.curve_points), 6)
    points = [t for t in (floor, config.published_threshold, *extra) if math.isfinite(t) and floor <= t <= 1.0]
    return np.unique(np.concatenate([grid[grid >= floor], np.asarray(points, dtype=np.float64)]))


def curve_tables(views: Sequence[ViewEvidence], thresholds: ArrayLike, scenes: Sequence[str],
                 config: RecallConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """False-negative rates and cost over a grid of thresholds, applied to every scene alike.

    Parameters
    ----------
    views : sequence of ViewEvidence
        The views.
    thresholds : array_like
        The grid.
    scenes : sequence of str
        Scenes to include (those with population objects).
    config : RecallConfig
        Population and IoU threshold.

    Returns
    -------
    curve : pandas.DataFrame
        One row per threshold: ``threshold``, :func:`pooled_cost`'s columns, ``share_views_truncated``
        (:func:`share_truncated` over the included scenes' views) and ``fully_observed`` (above ``t_cap``).
        ``mean_fnr`` is the calibration risk ``R_N`` of all scenes.
    by_scene : pandas.DataFrame
        :func:`scene_table_at` rows for every threshold.
    """
    included = [v for v in views if v.scene in set(scenes)]
    t_cap = truncation_threshold(included)
    parts, rows = [], []
    for t in np.asarray(thresholds, dtype=np.float64):
        table = scene_table_at(views, {s: float(t) for s in scenes}, config.population, config.min_iou)
        parts.append(table)
        rows.append({"threshold": float(t), **pooled_cost(table),
                     "share_views_truncated": share_truncated(included, float(t)), "fully_observed": bool(t > t_cap)})
    return pd.DataFrame(rows), pd.concat(parts, ignore_index=True)


# ============================================================================================================
# 5. The whole analysis
# ============================================================================================================


@dataclass
class RecallResult:
    """Every output of :func:`run_recall_control` for one dataset.

    Parameters
    ----------
    tables : dict of str to pandas.DataFrame
        ``objects``, ``boxes``, ``views`` (per item), ``loso`` (per alpha and scene), ``by_alpha``,
        ``conditional``, ``curve``, ``curve_by_scene`` and, if requested, ``mondrian`` (aggregates).
    summary : dict
        Headline numbers, settings and checks, JSON-ready.
    """

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


def _quantiles(values: pd.Series) -> dict[str, float]:
    """Minimum, quartiles and maximum of a series.

    Parameters
    ----------
    values : pandas.Series
        Values.

    Returns
    -------
    dict of str to float
        ``min, q25, median, q75, max``.
    """
    q = np.quantile(values.to_numpy(np.float64), [0.0, 0.25, 0.5, 0.75, 1.0])
    return dict(zip(("min", "q25", "median", "q75", "max"), (float(v) for v in q)))


def by_alpha_row(loso: pd.DataFrame, deploy: CRCResult, published: Mapping[str, float], config: RecallConfig,
                 salt: int, truncation: Mapping[str, Any]) -> dict[str, Any]:
    """The headline row of one alpha.

    Parameters
    ----------
    loso : pandas.DataFrame
        This alpha's leave-one-scene-out rows, joined with :func:`scene_table_at` at each scene's ``t_hat``.
    deploy : CRCResult
        CRC on all scenes (the threshold to deploy).
    published : mapping of str to float
        :func:`pooled_cost` at the published threshold.
    config : RecallConfig
        Bootstrap settings.
    salt : int
        Mixed into the bootstrap seed.
    truncation : mapping
        ``deploy_in_truncated_region`` and ``deploy_share_views_truncated`` of the deployable threshold.

    Returns
    -------
    dict
        Mean held-out FNR with its bootstrap interval, the per-scene spread, the share of scenes above alpha,
        feasibility, the floor, the ``t_hat`` distribution and how often it falls in the box cap's truncated
        region, the deployable threshold and the cost at ``t_hat`` against the published threshold.
    """
    alpha = float(deploy.alpha)
    fnr = loso["heldout_fnr"]
    lo, hi = bootstrap_interval(fnr.to_numpy(), np.arange(len(fnr)), config.n_boot, [config.seed, salt],
                                config.level)
    n_views, n_kept = float(loso["n_views"].sum()), float(loso["n_kept"].sum())
    t_hat = {f"threshold_{k}": v for k, v in _quantiles(loso["threshold"]).items()}
    spread = {f"heldout_fnr_{k}": v for k, v in _quantiles(fnr).items()}
    return {
        "alpha": alpha, "n_scenes": len(loso), "n_cal": int(loso["n_cal"].iloc[0]),
        "min_cal_scenes_for_feasibility": math.ceil(config.bound / alpha - 1 - 1e-9),
        "mean_heldout_fnr": float(fnr.mean()), "mean_heldout_fnr_lo": lo, "mean_heldout_fnr_hi": hi, **spread,
        "share_scenes_above_alpha": float((fnr > alpha).mean()),
        "pooled_heldout_fnr": float((fnr * loso["n_objects"]).sum() / loso["n_objects"].sum()),
        "mean_heldout_fnr_one_to_one": float(loso["fnr_one_to_one"].mean()),
        "share_feasible": float(loso["feasible"].mean()),
        "mean_floor_fnr": float(loso["floor_fnr"].mean()),
        "loso_mean_bounded_by_construction": bool(loso["floor_fnr"].mean() <= alpha),
        "share_scenes_floor_above_alpha": float((loso["floor_fnr"] > alpha).mean()), **t_hat,
        "share_folds_in_truncated_region": float(loso["in_truncated_region"].mean()),
        "share_views_truncated_at_threshold": float((loso["share_views_truncated"] * loso["n_views"]).sum()
                                                    / loso["n_views"].sum()),
        "deploy_threshold": deploy.threshold, "deploy_feasible": deploy.feasible, "deploy_cal_risk": deploy.cal_risk,
        "deploy_adjusted_risk": deploy.adjusted_risk, **truncation,
        "boxes_per_view": n_kept / n_views if n_views else math.nan,
        "unmatched_share": float(loso["n_unmatched"].sum()) / n_kept if n_kept else math.nan,
        "boxes_per_view_published": published.get("boxes_per_view", math.nan),
        "unmatched_share_published": published.get("unmatched_share", math.nan),
        "mean_fnr_published": published.get("mean_fnr", math.nan),
        "mean_fnr_one_to_one_published": published.get("mean_fnr_one_to_one", math.nan)}


def run_recall_control(views: Sequence[ViewEvidence], config: RecallConfig | None = None, *,
                       floor: float, crc_floor: float | None = None) -> RecallResult:
    """Leave-one-scene-out CRC of the false-negative rate, with its cost, breakdown and curve, for one dataset.

    Parameters
    ----------
    views : sequence of ViewEvidence
        Every view of one dataset.
    config : RecallConfig, optional
        Settings; defaults to ``RecallConfig()``.
    floor : float
        The observation floor: the lowest score the predictions keep. Curves and costs start here.
    crc_floor : float, optional
        ``t_min`` of CRC, the lowest threshold it may choose (default ``floor``). :func:`cap_free_floor` keeps
        every choice above the box cap's truncated region (module docstring, step 11).

    Returns
    -------
    RecallResult
        Tables and summary (see :class:`RecallResult`).

    Raises
    ------
    ValueError
        If the views span several datasets, fewer than two scenes have population objects, a box scores below
        the floor, or ``crc_floor`` is below ``floor``.
    """
    config = config or RecallConfig()
    datasets = sorted({v.dataset for v in views})
    if len(datasets) != 1:
        raise ValueError(f"run one dataset at a time (scenes of different datasets are not exchangeable): {datasets}")
    low = [v.view_key for v in views if len(v.scores) and float(v.scores.min()) < floor]
    if low:
        raise ValueError(f"{len(low)} views have boxes scored below the floor {floor}, e.g. {low[0]}")
    crc_floor = floor if crc_floor is None else float(crc_floor)
    if crc_floor < floor:
        raise ValueError(f"crc_floor {crc_floor} is below the observation floor {floor}")
    objects = object_table(views, config.min_iou)
    population = objects if config.population == "present" else objects[objects["clearly_visible"].astype(bool)]
    coverages = scene_coverages(objects, config.population)
    scenes = list(coverages)
    if len(scenes) < 2:
        raise ValueError(f"{len(scenes)} scenes have {config.population} objects; leave-one-scene-out needs two")
    calibrated = [v for v in views if v.scene in coverages]
    t_cap = truncation_threshold(calibrated)
    views_of = {s: [v for v in calibrated if v.scene == s] for s in scenes}
    published_ok = config.published_threshold >= floor
    at_published = (scene_table_at(views, {s: config.published_threshold for s in scenes}, config.population,
                                   config.min_iou) if published_ok else pd.DataFrame())
    published = pooled_cost(at_published) if published_ok else {}

    loso_parts, by_alpha, conditional, mondrian, checks = [], [], [], [], {}
    for a, alpha in enumerate(config.alphas):
        loso = leave_one_scene_out(coverages, alpha, floor=crc_floor, bound=config.bound)
        loso["in_truncated_region"] = loso["threshold"] <= t_cap
        loso["share_views_truncated"] = [share_truncated(views_of[s], t) for s, t in zip(loso["scene"],
                                                                                         loso["threshold"])]
        thresholds = dict(zip(loso["scene"], loso["threshold"]))
        at_hat = scene_table_at(views, thresholds, config.population, config.min_iou)
        loso = loso.merge(at_hat.drop(columns=["threshold", "n_objects"]), on="scene", validate="one_to_one")
        gap = float(np.max(np.abs(loso["fnr"] - loso["heldout_fnr"])))
        checks[f"alpha_{alpha:g}_max_fnr_gap"] = gap      # loss matrix vs per-view recount; must be ~0
        if gap > 1e-9:
            raise RuntimeError(f"held-out FNR from the loss matrix and from the views differ by {gap}")
        loso = loso.drop(columns=["fnr"])
        if published_ok:
            loso = loso.merge(at_published[["scene", "fnr", "fnr_one_to_one", "boxes_per_view", "unmatched_share"]]
                              .rename(columns=lambda c: c if c == "scene" else f"{c}_published"), on="scene")
        loso.insert(0, "alpha", alpha)
        loso_parts.append(loso)
        deploy = crc_threshold([coverages[s] for s in scenes], alpha, floor=crc_floor, bound=config.bound)
        by_alpha.append(by_alpha_row(loso, deploy, published, config, salt=a, truncation={
            "deploy_in_truncated_region": bool(deploy.threshold <= t_cap),
            "deploy_share_views_truncated": share_truncated(calibrated, deploy.threshold)}))
        conditional.append(conditional_table(population, thresholds, config, salt=a).assign(alpha=alpha))
        for stratum in config.mondrian:
            rows = mondrian_loso(population, stratum, alpha, floor=crc_floor, bound=config.bound)
            if len(rows):
                mondrian.append(rows.assign(alpha=alpha))

    grid = curve_thresholds(floor, config, extra=(crc_floor,))
    curve, curve_by_scene = curve_tables(views, grid, scenes, config)
    tables = {"objects": objects, "boxes": box_table(views, config.min_iou), "views": view_table(views),
              "loso": pd.concat(loso_parts, ignore_index=True), "by_alpha": pd.DataFrame(by_alpha),
              "conditional": pd.concat(conditional, ignore_index=True), "curve": curve,
              "curve_by_scene": curve_by_scene}
    if config.mondrian:
        tables["mondrian"] = pd.concat(mondrian, ignore_index=True) if mondrian else pd.DataFrame()
    for name in ("conditional", "mondrian"):
        if name in tables and len(tables[name]):
            cols = tables[name].columns.tolist()
            tables[name] = tables[name][["alpha"] + [c for c in cols if c != "alpha"]]
    summary = {
        "dataset": datasets[0], "population": config.population, "floor": floor, "crc_floor": crc_floor,
        "published_threshold": config.published_threshold if published_ok else None,
        "truncation": {
            "n_views_capped": int(sum(math.isfinite(v.cap_floor) for v in calibrated)),
            "share_views_capped": share_truncated(calibrated, -math.inf),
            "t_cap": t_cap if math.isfinite(t_cap) else None,
            "cap_floor_quantiles": (_quantiles(pd.Series([v.cap_floor for v in calibrated
                                                          if math.isfinite(v.cap_floor)]))
                                    if any(math.isfinite(v.cap_floor) for v in calibrated) else None),
            "crc_floor_above_t_cap": bool(crc_floor > t_cap),
            "published_share_views_truncated": (share_truncated(calibrated, config.published_threshold)
                                                if published_ok else None)},
        "n_scenes": len(scenes), "n_views": int(sum(v.scene in coverages for v in views)),
        "n_objects": int(len(population)), "n_scenes_without_population": len({v.scene for v in views}) - len(scenes),
        "n_unusable_boxes": int(sum(v.n_unusable for v in views)),
        "by_alpha": by_alpha, "checks": checks}
    return RecallResult(tables, summary)


# ============================================================================================================
# 6. Simulation helpers (tests and the guarantee check)
# ============================================================================================================


def synthetic_coverages(n_scenes: int, rng: np.random.Generator, *, floor: float = 0.05, mean_objects: float = 30.0,
                        never_found: float = 0.0, spread: float = 0.8) -> dict[str, Array]:
    """Coverage scores of synthetic scenes with scene-to-scene differences in difficulty.

    Each scene has ``1 + Poisson(mean_objects - 1)`` objects. A share ``never_found`` of them has no covering
    box (coverage ``-inf``). The others score ``floor + (1 - floor) * Beta(2, 2 exp(z))``, with one
    ``z ~ N(0, spread^2)`` per scene, so harder scenes have lower scores.

    Parameters
    ----------
    n_scenes : int
        Number of scenes.
    rng : numpy.random.Generator
        Random source.
    floor : float, default 0.05
        Lowest observed score.
    mean_objects : float, default 30.0
        Mean objects per scene (at least 1).
    never_found : float, default 0.0
        Probability that an object has no covering box at all.
    spread : float, default 0.8
        Standard deviation of the scene difficulty on the log scale.

    Returns
    -------
    dict of str to numpy.ndarray
        ``"s000"``, ``"s001"``, ... -> coverage scores.
    """
    out = {}
    for i in range(n_scenes):
        m = 1 + int(rng.poisson(max(mean_objects - 1.0, 0.0)))
        c = floor + (1.0 - floor) * rng.beta(2.0, 2.0 * math.exp(rng.normal(0.0, spread)), size=m)
        c[rng.random(m) < never_found] = -np.inf
        out[f"s{i:03d}"] = c
    return out


SceneSampler = Callable[[int, np.random.Generator], list[Array]]
"""Callable: ``(k, rng) -> k`` independent scenes' coverage scores, for :func:`simulate_guarantee`."""


def simulate_guarantee(n_cal: int, alpha: float, n_draws: int, rng: np.random.Generator, *, floor: float = 0.05,
                       n_test: int = 50, sampler: SceneSampler | None = None, **scene_kwargs: Any) -> dict[str, float]:
    """FNR on fresh scenes at the CRC threshold of independent calibration scenes, over repeated draws.

    Each draw calibrates ``t_hat`` on ``n_cal`` scenes and measures the FNR of ``n_test`` new scenes from the same
    source at it. Theorem 1 says the mean over draws is at most ``alpha`` when every scene's floor is; Theorem 2
    says it is at least ``alpha - 2B / (n_cal + 1)``. Unlike the leave-one-scene-out mean (see the module
    docstring, step 5), this can fail.

    Parameters
    ----------
    n_cal : int
        Calibration scenes per draw.
    alpha : float
        Target expected false-negative rate.
    n_draws : int
        Independent draws.
    rng : numpy.random.Generator
        Random source.
    floor : float, default 0.05
        ``t_min``.
    n_test : int, default 50
        New scenes per draw.
    sampler : callable, optional
        Draws independent scenes; defaults to :func:`synthetic_coverages` with ``floor`` and ``scene_kwargs``.
    **scene_kwargs : Any
        Passed to :func:`synthetic_coverages` (without a ``sampler``).

    Returns
    -------
    dict of str to float
        ``mean_fnr`` (over draws and new scenes) with its standard error ``se`` (over draws), ``lower_bound``,
        ``share_feasible``, ``share_scenes_above_alpha`` (new scenes) and ``mean_threshold``.
    """
    def synthetic(k: int, r: np.random.Generator) -> list[Array]:
        """Synthetic scenes.

        Parameters
        ----------
        k : int
            Number of scenes.
        r : numpy.random.Generator
            Random source.

        Returns
        -------
        list of numpy.ndarray
            Their coverage scores.
        """
        return list(synthetic_coverages(k, r, floor=floor, **scene_kwargs).values())

    draw = sampler or synthetic
    means, feasible, above, thresholds = [], [], [], []
    for _ in range(n_draws):
        result = crc_threshold(draw(n_cal, rng), alpha, floor=floor)
        fnr = loss_matrix(draw(n_test, rng), [result.threshold])[:, 0]
        means.append(fnr.mean())
        feasible.append(result.feasible)
        above.append((fnr > alpha).mean())
        thresholds.append(result.threshold)
    m = np.asarray(means)
    return {"n_cal": n_cal, "n_test": n_test, "alpha": alpha, "n_draws": n_draws, "mean_fnr": float(m.mean()),
            "se": float(m.std(ddof=1) / math.sqrt(len(m))) if len(m) > 1 else math.nan,
            "lower_bound": alpha - 2.0 * BOUND / (n_cal + 1), "share_feasible": float(np.mean(feasible)),
            "share_scenes_above_alpha": float(np.mean(above)), "mean_threshold": float(np.mean(thresholds))}


def fake_view_boxes(view: FrameView, present: Sequence[ObjectInView], visibility: str, rng: np.random.Generator, *,
                    floor: float = 0.05, miss_rate: float = 0.1, visibility_drop: float = 0.5, jitter: float = 0.08,
                    false_per_view: float = 3.0) -> list[ScoredBox]:
    """Fake labeller output for one view, derived from its ground truth (for checks only, never for results).

    Each present object is dropped with probability ``miss_rate + d (1 - miss_rate)(1 - v)^2``, with ``d`` the
    ``visibility_drop`` and ``v`` its visible fraction.
    A kept object gets its visible 2D box with every edge moved by ``N(0, (jitter * side)^2)``, a world box at
    the ground-truth box moved by ``N(0, 0.05^2)`` m per axis, and a score ``Beta(1 + 4 v, 1.5)`` with ``v`` its
    visible fraction. ``Poisson(false_per_view)`` further boxes land at random in the image with scores
    ``Beta(1, 4)`` and no world box. Boxes scoring below ``floor`` are dropped, as a prediction file would.

    Parameters
    ----------
    view : FrameView
        The view (image size).
    present : sequence of ObjectInView
        Its present ground-truth objects.
    visibility : str
        How the visible boxes were measured (see :func:`~measured_space.pipeline.visible_box_edges`).
    rng : numpy.random.Generator
        Random source.
    floor : float, default 0.05
        Lowest kept score.
    miss_rate : float, default 0.1
        Drop probability of a fully visible object.
    visibility_drop : float, default 0.5
        Extra drop probability of an invisible object (scaled as above).
    jitter : float, default 0.08
        Edge noise as a share of the box side.
    false_per_view : float, default 3.0
        Mean number of boxes on no object.

    Returns
    -------
    list of ScoredBox
        The boxes, highest score first.
    """
    width, height = view.camera.width, view.camera.height
    boxes = []
    for o in present:
        if o.bbox2d is None:
            continue
        v = float(o.visible_fraction)
        if rng.random() < miss_rate + visibility_drop * (1.0 - miss_rate) * (1.0 - v) ** 2:
            continue
        x0, y0, x1, y1 = visible_box_edges(o.bbox2d, visibility)
        w, h = x1 - x0, y1 - y0
        noise = rng.normal(0.0, jitter, 4) * np.array([w, h, w, h])
        box2d = clip_box2d((x0 + noise[0], y0 + noise[1], x1 + noise[2], y1 + noise[3]), width, height)
        box3d = OrientedBox(o.obj.box.center + rng.normal(0.0, 0.05, 3), o.obj.box.size, o.obj.box.rotation)
        boxes.append(ScoredBox(float(rng.beta(1.0 + 4.0 * v, 1.5)), box2d, box3d))
    for _ in range(int(rng.poisson(false_per_view))):
        w, h = rng.uniform(0.05, 0.4) * width, rng.uniform(0.05, 0.4) * height
        x0, y0 = rng.uniform(0, width - w), rng.uniform(0, height - h)
        boxes.append(ScoredBox(float(rng.beta(1.0, 4.0)), (x0, y0, x0 + w, y0 + h), None))
    return sorted((b for b in boxes if b.score >= floor), key=lambda b: -b.score)
