"""Missed objects (false negatives) of automatic 3D labellers: recall, effects on answers, flags without ground truth.

A per-label error bar is conditioned on a label existing, so it cannot see an object the labeller never found
(``docs/LEARNINGS.md``: aim A7, learning L16). This module measures those misses per labeller and source, measures
what they do to the answers, and measures how well disagreement flags them when no ground truth is available.

**Population and pairing.** For one labeller and source (a v0 depth source such as ``lidar``, or ``cutr_sam3``,
``cutr_clip``), the population of a view is its **present** ground-truth objects: those with a visible 2D box and
a visible fraction above ``QAConfig.presence_threshold`` (:func:`measured_space.qa.visible_ids`), with the
**clearly visible** subset of :func:`measured_space.pipeline.clearly_visible`. An object is **paired** when the
pipeline's association pairs a detection with it. Views are re-associated from their stored records by
:func:`measured_space.pipeline.rematch_view` with the ``"2d"`` association
(:func:`~measured_space.matching.match_detections_2d`: Hungarian on the 2D IoU of the detection's box and the
object's visible box, at IoU >= 0.3, class-agnostic), so the pairs are the pipeline's own. Where the shard stores
a 2D association, the re-association must reproduce its pairs exactly (:func:`load_view`). A **miss** is a present
object left unpaired. Each miss records why (:data:`MISS_CAUSES`):

- ``no_detection``: no detection overlaps the object's visible box at the association threshold;
- ``unboxed``: only detections without a box from this source overlap it (a failed lift, no CuTR box attached to a
  SAM 3 detection, an unnamed CuTR box). Such detections are not labels and take no part in the pairing;
- ``assignment``: a boxed detection overlaps it, but the assignment gave that detection to another object.

**Recall** (:func:`recall_table`) is paired / present, overall and by camera range (camera centre to box centre,
in the bins of ``scripts/compare_association.py``), visible fraction, truncation, class and scene. Every rate has a
Wilson 95% interval, which treats objects as independent. The overall rates also get a scene-cluster bootstrap 95%
interval (:func:`measured_space.scoring.cluster_bootstrap`), which resamples whole scenes (an ARKitScenes scene is
one video). With few scenes the bootstrap interval is itself rough.

**Effects on answers** (:func:`question_rows`, :func:`effects_table`). The questions are the oracle's, asked about
the view's ground truth, so a miss can damage an answer in two ways:

- **visible:** a question whose referent was missed gets no automatic answer
  (:func:`measured_space.pipeline.answer_auto`), so coverage drops;
- **silent:** an answer that depends on which objects exist goes wrong, with nothing to show it. In this paired
  design two question types do:

  - ``count``: the automatic count is ``oracle - missed - confused_out + extra_in``. ``missed`` counts the
    present objects of the class left unpaired, ``confused_out`` those paired with a detection counted under
    another phrase, and ``extra_in`` the counted boxes that are not objects of the class (unpaired detections and
    detections paired with an object of another class). An under-count with ``missed > 0`` is a silent error
    caused (at least in part) by misses;
  - ``closest_instance``: the pipeline answers only when every candidate instance is paired, so here a miss only
    removes the answer. A labeller that writes its own questions offers only the instances it found. The
    **open-world** answer reproduces that: with the reference object and at least two candidates paired, it
    answers among the paired candidates (:func:`measured_space.qa.answer` on the reduced spec). A wrong open-world
    answer whose true answer object was missed is a silent error caused by a miss.

  The other types depend only on their own referents.

**Flags without ground truth** (:func:`cross_labeller_flags`, :func:`cross_frame_flags`,
:func:`cross_camera_flags`). A flag is a 2D box in a view where some evidence says an object is, and where the
flagged labeller has no labelled box at 2D IoU >= ``FlagConfig.min_iou``. Against ground truth:

- a flag **hits** when it overlaps (IoU >= ``min_iou``) the visible box of a present object the flagged labeller
  missed. **Flag precision** is hits / flags;
- **flag recall** is the share of the labeller's misses, in the views the method covers, that some flag hits;
- a flag that overlaps a present object the labeller did pair is ``found_object``;
- a flag that overlaps no present ground-truth object is ``unannotated``, **not** false. ARKitScenes annotates
  only 17 furniture classes and CuTR is class-agnostic, so many such flags point at real objects that have no
  ground-truth box (or at objects too occluded to be present).

The three kinds of evidence are:

- **cross-labeller** (:func:`cross_labeller_flags`): a box of another labeller (CuTR) with no box of the flagged
  labeller at IoU >= ``min_iou``, in both directions. Disagreement between independent models as a signal of
  their errors is the query-by-committee idea (Seung, Opper and Sompolinsky, COLT 1992);
- **cross-frame**, for static scenes (:func:`cross_frame_flags`): the labeller's own 3D boxes from the *other*
  views of the same video, de-duplicated by 3D overlap and projected into the view. A box is flagged "seen
  elsewhere, missing here" when it lies in front of the camera and mostly inside the image, the view's
  independent depth (iPad LiDAR for ARKitScenes; never the laser depth, which is the reference) shows it
  unoccluded, and no labelled box in the view overlaps its projection. This is the established
  temporal-consistency check, used to find missed detections and missing labels by model assertions (Kang,
  Raghavan, Bailis and Zaharia, "Model Assertions for Monitoring and Improving ML Models", MLSys 2020) and Fixy
  (Kang, Arechiga, Pillai, Bailis and Zaharia, "Finding Label and Model Errors in Perception Data With Learned
  Observation Assertions", SIGMOD 2022). It is not new here; this module measures how well it catches one
  labeller's misses;
- **cross-camera**, for fixed synchronised cameras (:func:`cross_camera_flags`): the same projection test, with
  the boxes of the *other* cameras at the same frame index. Without a depth map independent of the reference for
  the receiving camera, the occlusion test is skipped and only geometry decides.

2D boxes are ``(xmin, ymin, xmax, ymax)`` in pixels with pixel edges at integer coordinates (as
:attr:`measured_space.perception.Instance.box2d`). Distances are in metres. Conventions are those of
:mod:`measured_space.geometry`.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from measured_space.audit import wilson_interval
from measured_space.box_sources import ViewPredictions, check_predictions, clip_box2d
from measured_space.geometry import Array, OrientedBox
from measured_space.matching import Match2DConfig, box_iou_2d, match_detections_2d, prism_iou
from measured_space.pipeline import (
    PipelineConfig,
    ViewResult,
    auto_objects,
    clearly_visible,
    rematch_view,
    visible_box_edges,
)
from measured_space.qa import answer as answer_spec
from measured_space.qa import phrase_for, visible_ids
from measured_space.qa_schema import QAItem, QuestionSpec, is_correct
from measured_space.scene import FrameView, ObjectInView, SceneObject
from measured_space.scoring import cluster_bootstrap
from measured_space.visibility import object_in_view

Box2D = tuple[float, float, float, float]
"""tuple of float: ``(xmin, ymin, xmax, ymax)`` in pixels, pixel edges at integer coordinates."""

DepthFn = Callable[[FrameView], NDArray[np.floating]]
"""Callable: ``depth(view) -> (h, w)`` planar depth in metres (0 or NaN invalid), any resolution."""

RANGE_BINS_M: tuple[float, ...] = (0.0, 10.0, 20.0, 45.0, math.inf)
"""tuple of float: edges of the camera-range bins (m), those of ``scripts/compare_association.py``."""

VISIBLE_FRACTION_BINS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.6, 1.0)
"""tuple of float: edges of the visible-fraction bins, ``[lo, hi)`` with the last bin closed."""

TRUNCATION_BINS: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0)
"""tuple of float: edges of the truncation bins, ``(lo, hi]``; a fraction of exactly 0 has its own bin ``"0"``."""

MISS_CAUSES: tuple[str, ...] = ("no_detection", "unboxed", "assignment")
"""tuple of str: why a present object was left unpaired (see the module docstring)."""

POPULATIONS: tuple[str, ...] = ("present", "clearly_visible")
"""tuple of str: the object populations recall is reported for."""

RECALL_BY: dict[str, str] = {"range": "range_bin", "visible_fraction": "visible_bin",
                             "truncation": "truncation_bin", "class": "label", "scene": "scene"}
"""dict of str to str: breakdown name -> column of :func:`object_rows` that defines its groups."""

STATIC_DATASETS: tuple[str, ...] = ("arkitscenes",)
"""tuple of str: datasets whose scenes are static, so boxes from other frames of a video may be projected."""

MULTI_CAMERA_DATASETS: tuple[str, ...] = ("smartspaces",)
"""tuple of str: datasets with fixed, synchronised cameras that share frame indices and a world frame."""

FLAG_KINDS: tuple[str, ...] = ("cross_labeller", "cross_frame", "cross_camera")
"""tuple of str: the kinds of miss flag."""

FLAG_CATEGORIES: tuple[str, ...] = ("missed_object", "found_object", "unannotated")
"""tuple of str: what a flag points at, by ground truth (see the module docstring)."""

CUTR_BOXES_SOURCE = "boxes"
"""str: the source name of a CuTR predictions set when it is the flagged labeller (its raw, class-agnostic boxes)."""

_PROJECTION_CONVENTION = "projection"
"""str: a non-:data:`~measured_space.pipeline.GT_BOX2D` name for :func:`visible_box_edges`: boxes from
:func:`~measured_space.visibility.object_in_view` put pixel centres at integer coordinates."""

_MIN_Z = 0.05
"""float: corners nearer than this to the camera plane count as behind it (m), as in
:func:`~measured_space.visibility.object_in_view`."""


# --------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------


def range_bin(distance: float, edges: Sequence[float] = RANGE_BINS_M) -> str:
    """The name of the camera-range bin a distance falls in.

    Parameters
    ----------
    distance : float
        Camera centre to box centre, metres.
    edges : sequence of float, default RANGE_BINS_M
        Increasing bin edges; the last may be infinite.

    Returns
    -------
    str
        ``"<lo>_<hi>m"``, or ``"<lo>m_plus"`` for the open last bin (``"0_10m"``, ``"10_20m"``, ``"20_45m"``,
        ``"45m_plus"`` for the default edges, the names of ``comparison.csv``).
    """
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= distance < hi:
            return f"{lo:g}_{hi:g}m" if math.isfinite(hi) else f"{lo:g}m_plus"
    lo, hi = edges[-2], edges[-1]
    return f"{lo:g}_{hi:g}m" if math.isfinite(hi) else f"{lo:g}m_plus"


def fraction_bin(value: float, edges: Sequence[float] = VISIBLE_FRACTION_BINS) -> str:
    """The name of the bin a fraction falls in, bins ``[lo, hi)`` with the last one closed.

    Parameters
    ----------
    value : float
        A fraction in [0, 1].
    edges : sequence of float, default VISIBLE_FRACTION_BINS
        Increasing bin edges.

    Returns
    -------
    str
        ``"<lo>-<hi>"``, e.g. ``"0.3-0.6"``.
    """
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= value < hi:
            return f"{lo:g}-{hi:g}"
    return f"{edges[-2]:g}-{edges[-1]:g}"


def truncation_bin(value: float, edges: Sequence[float] = TRUNCATION_BINS) -> str:
    """The name of the bin a truncated fraction falls in, bins ``(lo, hi]`` plus ``"0"`` for none.

    Parameters
    ----------
    value : float
        Share of the camera-facing surface outside the image or behind the camera, in [0, 1].
    edges : sequence of float, default TRUNCATION_BINS
        Increasing bin edges starting at 0.

    Returns
    -------
    str
        ``"0"`` for an untruncated object, else ``"<lo>-<hi>"``, e.g. ``"0-0.3"``.
    """
    if value <= 0.0:
        return "0"
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo < value <= hi:
            return f"{lo:g}-{hi:g}"
    return f"{edges[-2]:g}-{edges[-1]:g}"


def rate(k: int, n: int, level: float = 0.95) -> tuple[float | None, float | None, float | None]:
    """A proportion with its Wilson interval.

    Parameters
    ----------
    k : int
        Successes.
    n : int
        Trials.
    level : float, default 0.95
        Two-sided level of the interval.

    Returns
    -------
    tuple of float or None
        ``(k / n, lower, upper)``; all ``None`` when ``n == 0``.
    """
    if n <= 0:
        return None, None, None
    lower, upper = wilson_interval(k, n, level)
    return k / n, float(lower), float(upper)


def scene_bootstrap(values: ArrayLike, clusters: ArrayLike, *, n_boot: int, seed: int, salt: str,
                    level: float = 0.95) -> tuple[float | None, float | None]:
    """Scene-cluster bootstrap interval of a mean (whole scenes resampled with replacement).

    Parameters
    ----------
    values : array_like, shape (n,)
        One value per item, e.g. 1 for a paired object and 0 for a miss.
    clusters : array_like, shape (n,)
        The scene of each item.
    n_boot : int
        Bootstrap replicates; 0 skips the interval.
    seed : int
        Base seed.
    salt : str
        Names the group; mixed into the seed so that every group gets its own stream.
    level : float, default 0.95
        Two-sided level.

    Returns
    -------
    tuple of float or None
        Percentile interval ``(lower, upper)``; ``(None, None)`` for no items or ``n_boot == 0``.
    """
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or n_boot <= 0:
        return None, None
    rng = np.random.default_rng([seed, zlib.crc32(salt.encode())])
    boot = cluster_bootstrap(values, np.asarray(clusters), n_boot, rng)
    lower, upper = np.quantile(boot, [(1 - level) / 2, (1 + level) / 2])
    return float(lower), float(upper)


def _flag_is(values: pd.Series, value: bool) -> pd.Series:
    """Which entries of a column of optional booleans equal a value (``None`` and NaN match neither).

    Parameters
    ----------
    values : pandas.Series
        Booleans, possibly with ``None`` or NaN.
    value : bool
        The value to test for.

    Returns
    -------
    pandas.Series of bool
        True where the entry is a boolean equal to ``value``.
    """
    return values.map(lambda x: isinstance(x, (bool, np.bool_)) and bool(x) == value).astype(bool)


def identity(result: ViewResult) -> dict[str, str]:
    """A view's identity fields, as every row about it carries them.

    Parameters
    ----------
    result : ViewResult
        The view.

    Returns
    -------
    dict of str to str
        ``dataset``, ``scene``, ``camera_id``, ``frame`` and ``view_key``.
    """
    v = result.view
    return {"dataset": v.dataset, "scene": v.scene, "camera_id": v.camera_id, "frame": v.frame,
            "view_key": result.key}


def present_objects(result: ViewResult) -> list[ObjectInView]:
    """The view's present ground-truth objects (the population), in oracle order.

    Parameters
    ----------
    result : ViewResult
        The view.

    Returns
    -------
    list of ObjectInView
        Objects with a visible 2D box and a visible fraction above the presence threshold.
    """
    ids = visible_ids(result.oracle, result.config.qa)
    return [o for o in result.oracle if o.obj.object_id in ids]


def labelled_boxes2d(result: ViewResult, source: str) -> tuple[list[int], Array]:
    """The labeller's labels in a view: detections that have a box from the source, with their 2D boxes.

    Parameters
    ----------
    result : ViewResult
        The view.
    source : str
        The labeller's source.

    Returns
    -------
    indices : list of int
        Detection indices with a box.
    boxes2d : numpy.ndarray, shape (n, 4)
        Their 2D boxes.
    """
    indices = [d.index for d, lift in zip(result.detections, result.lifts[source]) if lift.box is not None]
    boxes = np.array([result.detections[i].instance.box2d for i in indices], dtype=np.float64).reshape(-1, 4)
    return indices, boxes


def _frame_order(frame: str) -> tuple[float, str]:
    """Sort key of a frame identifier: numerically when it is a number, else by text.

    Parameters
    ----------
    frame : str
        Frame identifier (a timestamp or a frame index).

    Returns
    -------
    tuple of (float, str)
        ``(number, frame)``, with ``inf`` for a non-numeric frame.
    """
    try:
        return float(frame), frame
    except ValueError:
        return math.inf, frame


# --------------------------------------------------------------------------------------------------------
# Reading stored views
# --------------------------------------------------------------------------------------------------------


def stored_pairs(records: Mapping[str, Any]) -> set[tuple[str, int, str]]:
    """The pairs a shard stores, from its oracle objects' match records.

    Parameters
    ----------
    records : mapping
        A view's records as stored in a shard (``oracle_objects`` with per-source ``match`` records).

    Returns
    -------
    set of tuple
        ``(source, detection index, object id)`` for every stored pair.
    """
    return {(source, int(m["detection"]), r["object_id"]) for r in records["oracle_objects"] if r.get("match")
            for source, m in r["match"].items() if m is not None}


def result_pairs(result: ViewResult) -> set[tuple[str, int, str]]:
    """The pairs of a (re-associated) view.

    Parameters
    ----------
    result : ViewResult
        The view.

    Returns
    -------
    set of tuple
        ``(source, detection index, object id)`` for every pair.
    """
    return {(source, m.detection, m.object_id) for source, match in result.matches.items() for m in match.matches}


def load_view(path: str | Path, *, check: bool = True) -> ViewResult:
    """Read one shard and re-associate it with the pipeline's 2D association (no models).

    Parameters
    ----------
    path : str or pathlib.Path
        A shard ``<labels>/<dataset>/shards/<view>.json`` of ``scripts/run_labels.py`` or
        ``scripts/run_box_source.py``.
    check : bool, default True
        If the shard stores a 2D association, require the re-association to reproduce its pairs exactly.

    Returns
    -------
    ViewResult
        The view, re-associated by :func:`~measured_space.pipeline.rematch_view` with ``association="2d"`` and the
        stored 2D threshold.

    Raises
    ------
    ValueError
        If the re-associated pairs differ from the stored 2D pairs, or as
        :func:`~measured_space.pipeline.rematch_view`.
    """
    records = json.loads(Path(path).read_text())["records"]
    result = rematch_view(records, association="2d")
    if check and PipelineConfig.from_json(records["view"]["config"]).association == "2d":
        stored, got = stored_pairs(records), result_pairs(result)
        if stored != got:
            raise ValueError(f"{path}: re-association gives {len(got)} pairs, the shard stores {len(stored)} "
                             f"({len(got ^ stored)} differ)")
    return result


def shard_scene(path: str | Path) -> str:
    """The scene part of a shard's file name, ``<scene with / as __>__<camera>__<frame>.json``.

    Parameters
    ----------
    path : str or pathlib.Path
        A shard file.

    Returns
    -------
    str
        The scene with ``/`` written as ``__``.
    """
    return Path(path).stem.rsplit("__", 2)[0]


def scene_shards(folder: str | Path, dataset: str, scenes: Collection[str] | None = None) -> dict[str, list[Path]]:
    """A dataset's shard files grouped by scene.

    Parameters
    ----------
    folder : str or pathlib.Path
        A label folder (holding ``<dataset>/shards/``).
    dataset : str
        Dataset name.
    scenes : collection of str, optional
        Keep only these scenes (``/`` or ``__`` spelling); all by default.

    Returns
    -------
    dict of str to list of pathlib.Path
        Scene (``__`` spelling) -> its shards, sorted by name; scenes sorted.

    Raises
    ------
    FileNotFoundError
        If the folder has no shards for the dataset.
    """
    shard_dir = Path(folder) / dataset / "shards"
    paths = sorted(shard_dir.glob("*.json")) if shard_dir.is_dir() else []
    if not paths:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    wanted = None if scenes is None else {s.replace("/", "__") for s in scenes}
    groups: dict[str, list[Path]] = {}
    for p in paths:
        key = shard_scene(p)
        if wanted is None or key in wanted:
            groups.setdefault(key, []).append(p)
    return dict(sorted(groups.items()))


def load_scene(paths: Sequence[str | Path], *, check: bool = True) -> list[ViewResult]:
    """Read and re-associate every view of one scene.

    Parameters
    ----------
    paths : sequence of str or pathlib.Path
        The scene's shards.
    check : bool, default True
        Passed to :func:`load_view`.

    Returns
    -------
    list of ViewResult
        The views, sorted by camera and then frame (numerically where possible).

    Raises
    ------
    ValueError
        If the shards belong to more than one scene, or as :func:`load_view`.
    """
    results = [load_view(p, check=check) for p in paths]
    if len({(r.view.dataset, r.view.scene) for r in results}) > 1:
        raise ValueError(f"shards of several scenes given as one: {sorted({r.view.scene for r in results})}")
    return sorted(results, key=lambda r: (r.view.camera_id, _frame_order(r.view.frame)))


def iter_scenes(folder: str | Path, dataset: str, scenes: Collection[str] | None = None, *, check: bool = True
                ) -> Iterator[tuple[str, list[ViewResult]]]:
    """Yield a dataset's views scene by scene, re-associated (one scene in memory at a time).

    Parameters
    ----------
    folder : str or pathlib.Path
        A label folder.
    dataset : str
        Dataset name.
    scenes : collection of str, optional
        Keep only these scenes.
    check : bool, default True
        Passed to :func:`load_view`.

    Yields
    ------
    tuple of (str, list of ViewResult)
        The scene (as the views name it) and its views.
    """
    for paths in scene_shards(folder, dataset, scenes).values():
        results = load_scene(paths, check=check)
        yield results[0].view.scene, results


# --------------------------------------------------------------------------------------------------------
# Recall
# --------------------------------------------------------------------------------------------------------


def object_rows(result: ViewResult, source: str, labeller: str) -> list[dict[str, Any]]:
    """One row per present ground-truth object: whether the labeller paired it, and why not.

    Parameters
    ----------
    result : ViewResult
        The view, re-associated.
    source : str
        The labeller's source; a view without it gives no rows.
    labeller : str
        The labeller's name, recorded on each row.

    Returns
    -------
    list of dict
        Identity, ``labeller``, ``source``, ``object_id``, ``label``, ``phrase``, ``visible_fraction``,
        ``truncated_fraction``, ``clearly_visible``, ``range_m`` (camera centre to box centre), ``range_bin``,
        ``visible_bin``, ``truncation_bin``, ``paired``, ``cause`` (``"paired"`` or one of :data:`MISS_CAUSES`),
        ``detection``, ``iou2d``, ``label_correct`` and ``ambiguous`` of the pair (``None`` for a miss),
        ``best_iou2d_boxed`` and ``best_iou2d_unboxed`` (largest 2D IoU of the object's visible box with a
        detection that has, or lacks, a box from the source; 0 if there is none).
    """
    if source not in result.sources:
        return []
    present = present_objects(result)
    if not present:
        return []
    min_iou = result.config.match_2d.min_iou
    by_object = result.matches[source].by_object()
    boxed = np.array([lift.box is not None for lift in result.lifts[source]], dtype=bool)
    det_boxes = np.array([d.instance.box2d for d in result.detections], dtype=np.float64).reshape(-1, 4)
    edges = np.array([visible_box_edges(o.bbox2d, result.visibility) for o in present])  # type: ignore[arg-type]
    iou = box_iou_2d(det_boxes, edges)                                                      # (detections, present)
    camera = result.view.camera_center
    base = {**identity(result), "labeller": labeller, "source": source}
    rows = []
    for j, seen in enumerate(present):
        oid = seen.obj.object_id
        m = by_object.get(oid)
        best_boxed = float(iou[boxed, j].max()) if boxed.any() else 0.0
        best_unboxed = float(iou[~boxed, j].max()) if (~boxed).any() else 0.0
        if m is not None:
            cause = "paired"
        elif best_boxed >= min_iou:
            cause = "assignment"
        elif best_unboxed >= min_iou:
            cause = "unboxed"
        else:
            cause = "no_detection"
        distance = float(np.linalg.norm(seen.obj.box.center - camera))
        rows.append({**base, "object_id": oid, "label": seen.obj.label, "phrase": phrase_for(seen.obj.label),
                     "visible_fraction": seen.visible_fraction, "truncated_fraction": seen.truncated_fraction,
                     "clearly_visible": clearly_visible(seen, result.config.qa), "range_m": distance,
                     "range_bin": range_bin(distance), "visible_bin": fraction_bin(seen.visible_fraction),
                     "truncation_bin": truncation_bin(seen.truncated_fraction), "paired": m is not None,
                     "cause": cause, "detection": None if m is None else m.detection,
                     "iou2d": None if m is None else m.iou2d,
                     "label_correct": None if m is None else m.label_correct,
                     "ambiguous": None if m is None else m.ambiguous,
                     "best_iou2d_boxed": best_boxed, "best_iou2d_unboxed": best_unboxed})
    return rows


RECALL_COLUMNS: tuple[str, ...] = (
    "labeller", "source", "dataset", "population", "by", "group", "n", "n_paired", "n_missed", "recall",
    "wilson_lo", "wilson_hi", "n_scenes", "n_views", "boot_lo", "boot_hi")
"""tuple of str: columns of :func:`recall_table`."""


def recall_table(objects: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Recall per labeller, source and dataset: overall and by range, visibility, truncation, class and scene.

    Parameters
    ----------
    objects : pandas.DataFrame
        Rows of :func:`object_rows`.
    n_boot : int, default 2000
        Scene-cluster bootstrap replicates for the overall rows (0: none).
    seed : int, default 0
        Bootstrap seed.
    level : float, default 0.95
        Two-sided level of the intervals.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`RECALL_COLUMNS`. ``population`` is ``"present"`` or ``"clearly_visible"``; ``by`` is
        ``"all"`` or a key of :data:`RECALL_BY`, and ``group`` the bin, class or scene. ``wilson_lo, wilson_hi``
        is the Wilson interval; ``boot_lo, boot_hi`` the scene-cluster bootstrap interval, for ``by == "all"``
        only.
    """
    rows: list[dict[str, Any]] = []
    if objects.empty:
        return pd.DataFrame(rows, columns=list(RECALL_COLUMNS))
    for (labeller, source, dataset), frame in objects.groupby(["labeller", "source", "dataset"], sort=True):
        for population in POPULATIONS:
            sub = frame if population == "present" else frame[frame["clearly_visible"].astype(bool)]
            if sub.empty:
                continue
            key = {"labeller": labeller, "source": source, "dataset": dataset, "population": population}
            groups: list[tuple[str, str, pd.DataFrame]] = [("all", "all", sub)]
            for by, column in RECALL_BY.items():
                groups += [(by, str(group), g) for group, g in sub.groupby(column, sort=True)]
            for by, group, g in groups:
                n, k = len(g), int(g["paired"].astype(bool).sum())
                value, lo, hi = rate(k, n, level)
                boot = (scene_bootstrap(g["paired"].astype(float), g["dataset"] + "/" + g["scene"], n_boot=n_boot,
                                        seed=seed, salt=f"{labeller}|{source}|{dataset}|{population}", level=level)
                        if by == "all" else (None, None))
                rows.append({**key, "by": by, "group": group, "n": n, "n_paired": k, "n_missed": n - k,
                             "recall": value, "wilson_lo": lo, "wilson_hi": hi, "n_scenes": int(g["scene"].nunique()),
                             "n_views": int(g["view_key"].nunique()), "boot_lo": boot[0], "boot_hi": boot[1]})
    return pd.DataFrame(rows, columns=list(RECALL_COLUMNS))


MISS_CAUSE_COLUMNS: tuple[str, ...] = (
    "labeller", "source", "dataset", "population", "n_present", "n_missed",
    *(f"n_{c}" for c in MISS_CAUSES), *(f"share_{c}" for c in MISS_CAUSES))
"""tuple of str: columns of :func:`miss_cause_table`."""


def miss_cause_table(objects: pd.DataFrame) -> pd.DataFrame:
    """Why objects were missed, per labeller, source, dataset and population.

    Parameters
    ----------
    objects : pandas.DataFrame
        Rows of :func:`object_rows`.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`MISS_CAUSE_COLUMNS`: counts of each cause in :data:`MISS_CAUSES` and their shares of the
        misses.
    """
    rows: list[dict[str, Any]] = []
    if objects.empty:
        return pd.DataFrame(rows, columns=list(MISS_CAUSE_COLUMNS))
    for (labeller, source, dataset), frame in objects.groupby(["labeller", "source", "dataset"], sort=True):
        for population in POPULATIONS:
            sub = frame if population == "present" else frame[frame["clearly_visible"].astype(bool)]
            missed = sub[~sub["paired"].astype(bool)]
            row: dict[str, Any] = {"labeller": labeller, "source": source, "dataset": dataset,
                                   "population": population, "n_present": len(sub), "n_missed": len(missed)}
            for cause in MISS_CAUSES:
                n = int((missed["cause"] == cause).sum())
                row[f"n_{cause}"] = n
                row[f"share_{cause}"] = n / len(missed) if len(missed) else None
            rows.append(row)
    return pd.DataFrame(rows, columns=list(MISS_CAUSE_COLUMNS))


# --------------------------------------------------------------------------------------------------------
# Effects of misses on answers
# --------------------------------------------------------------------------------------------------------


def count_effect(spec: QuestionSpec, result: ViewResult, source: str,
                 objects_by_id: Mapping[str, SceneObject]) -> dict[str, Any]:
    """Split a count question's automatic error into misses, class confusions and extra boxes.

    Parameters
    ----------
    spec : QuestionSpec
        A ``count`` question about the view.
    result : ViewResult
        The view, re-associated.
    source : str
        The labeller's source.
    objects_by_id : mapping of str to SceneObject
        The source's automatic boxes as answering keys them (:func:`measured_space.pipeline.auto_objects`).

    Returns
    -------
    dict
        ``oracle_count`` (present objects of the phrase), ``auto_count`` (automatic boxes counted under it),
        ``count_missed`` (present objects of the phrase left unpaired), ``count_confused_out`` (paired, but counted
        under another phrase), ``count_extra_in`` (counted boxes that are not objects of the phrase),
        ``count_outcome`` (``"exact"``, ``"under"`` or ``"over"``), ``count_miss_role`` (``"none"`` without a
        missed instance; else ``"masked"`` when the count is still exact, ``"under_miss_only"`` when misses alone
        explain the under-count, ``"under_miss_and_other"``, or ``"over_despite_miss"``) and
        ``count_error_without_misses`` (``extra_in - confused_out``: the error that would remain if every missed
        instance were found and named right).
    """
    phrase = spec.params["phrase"]
    present_ids = visible_ids(result.oracle, result.config.qa)
    truth = {o.obj.object_id for o in result.oracle
             if o.obj.object_id in present_ids and phrase_for(o.obj.label) == phrase}
    paired = set(result.matches[source].by_object())
    counted = {k for k, o in objects_by_id.items() if phrase_for(o.label) == phrase}
    missed = truth - paired
    confused_out = (truth & paired) - counted
    extra_in = counted - truth
    oracle_count, auto_count = len(truth), len(counted)
    outcome = "exact" if auto_count == oracle_count else ("under" if auto_count < oracle_count else "over")
    if not missed:
        role = "none"
    elif outcome == "exact":
        role = "masked"
    elif outcome == "under":
        role = "under_miss_only" if not confused_out and not extra_in else "under_miss_and_other"
    else:
        role = "over_despite_miss"
    return {"oracle_count": oracle_count, "auto_count": auto_count, "count_missed": len(missed),
            "count_confused_out": len(confused_out), "count_extra_in": len(extra_in), "count_outcome": outcome,
            "count_miss_role": role, "count_error_without_misses": len(extra_in) - len(confused_out)}


def closest_effect(spec: QuestionSpec, oracle_answer: Any, auto_item: QAItem | None, result: ViewResult,
                   source: str, objects_by_id: Mapping[str, SceneObject], min_candidates: int = 2
                   ) -> dict[str, Any]:
    """A closest-instance question's strict and open-world automatic answers, and whether a miss broke them.

    Parameters
    ----------
    spec : QuestionSpec
        A ``closest_instance`` question: reference first, then the candidates.
    oracle_answer : Any
        The oracle's answer (a candidate's expression).
    auto_item : QAItem or None
        The pipeline's automatic answer, ``None`` when a referent was missed.
    result : ViewResult
        The view, re-associated.
    source : str
        The labeller's source.
    objects_by_id : mapping of str to SceneObject
        The source's automatic boxes as answering keys them.
    min_candidates : int, default 2
        Paired candidates the open-world answer needs (a labeller asks "which" only among several instances).

    Returns
    -------
    dict
        ``closest_answer_id`` (the oracle's answer object), ``closest_answer_missed`` (whether the labeller missed
        it), ``open_status`` (``"strict"``: every referent paired, the pipeline's answer; ``"open"``: the reference
        and at least ``min_candidates`` candidates paired but not all; ``"unanswerable"``), ``open_answer_id``,
        ``open_correct`` and ``silent_miss`` (open-world answer wrong and its true answer object missed).
    """
    reference, candidates = spec.referents[0], spec.referents[1:]
    by_expression = {c.expression: c.object_id for c in candidates}
    answer_id = by_expression[str(oracle_answer)]
    paired = set(result.matches[source].by_object())
    answer_missed = answer_id not in paired
    found = [c for c in candidates if c.object_id in objects_by_id]
    open_answer: str | None = None
    if auto_item is not None:
        status, open_answer = "strict", by_expression[str(auto_item.answer)]
    elif reference.object_id in objects_by_id and len(found) >= min_candidates:
        reduced = replace(spec, referents=(reference, *found), choices=tuple(c.expression for c in found))
        value = answer_spec(reduced, objects_by_id, result.view)[0]
        status, open_answer = "open", by_expression[str(value)]
    else:
        status = "unanswerable"
    correct = None if open_answer is None else open_answer == answer_id
    return {"closest_answer_id": answer_id, "closest_answer_missed": answer_missed, "open_status": status,
            "open_answer_id": open_answer, "open_correct": correct,
            "silent_miss": bool(correct is False and answer_missed)}


def question_rows(result: ViewResult, source: str, labeller: str, *, min_candidates: int = 2
                  ) -> list[dict[str, Any]]:
    """One row per oracle question: the automatic side's status and how misses affected it.

    Parameters
    ----------
    result : ViewResult
        The view, re-associated.
    source : str
        The labeller's source; a view without it gives no rows.
    labeller : str
        The labeller's name.
    min_candidates : int, default 2
        Passed to :func:`closest_effect`.

    Returns
    -------
    list of dict
        Identity, ``labeller``, ``source``, ``question_id``, ``qtype``, ``n_referents``, ``answered`` (the pipeline
        gave an automatic answer), ``n_missing`` and ``missing_ids`` (missed referents, ``;``-joined),
        ``oracle_answer``, ``auto_answer``, ``correct`` (:func:`measured_space.qa_schema.is_correct`, ``None`` if
        unanswered) and ``silent_miss``; count questions add the fields of :func:`count_effect` (with
        ``silent_miss`` True for a wrong count with a missed instance), closest-instance questions those of
        :func:`closest_effect`.

    Raises
    ------
    ValueError
        If a question is unanswered without a missed referent, or a recomputed count differs from the answers.
    """
    if source not in result.sources:
        return []
    oracle = {i.id: i for i in result.items["oracle"]}
    auto = {i.id: i for i in result.items[f"auto_{source}"]}
    match = result.matches[source]
    paired = set(match.by_object())
    objects_by_id, _ = auto_objects(result.detections, result.lifts[source], match,
                                    {o.obj.object_id: o.obj.label for o in result.oracle})
    base = {**identity(result), "labeller": labeller, "source": source}
    rows = []
    for spec in result.specs:
        item, reference = auto.get(spec.id), oracle[spec.id]
        missing = [r.object_id for r in spec.referents if r.object_id not in paired]
        if item is None and not missing:
            raise ValueError(f"{result.key}: question {spec.id} is unanswered without a missed referent")
        correct = None if item is None else is_correct(spec, item.answer, reference.answer)
        row: dict[str, Any] = {**base, "question_id": spec.id, "qtype": spec.qtype,
                               "n_referents": len(spec.referents), "answered": item is not None,
                               "n_missing": len(missing), "missing_ids": ";".join(missing),
                               "oracle_answer": reference.answer, "auto_answer": None if item is None else item.answer,
                               "correct": correct, "silent_miss": False}
        if spec.qtype == "count":
            effect = count_effect(spec, result, source, objects_by_id)
            if item is None or (effect["oracle_count"], effect["auto_count"]) != (reference.answer, item.answer):
                raise ValueError(f"{result.key}: count {spec.id} recomputed as {effect['oracle_count']} / "
                                 f"{effect['auto_count']}, answered {reference.answer} / "
                                 f"{None if item is None else item.answer}")
            row.update(effect)
            row["silent_miss"] = bool(correct is False and effect["count_missed"] > 0)
        elif spec.qtype == "closest_instance":
            row.update(closest_effect(spec, reference.answer, item, result, source, objects_by_id, min_candidates))
        rows.append(row)
    return rows


EFFECT_COLUMNS: tuple[str, ...] = (
    "labeller", "source", "dataset", "qtype", "n_questions", "n_answered", "coverage", "coverage_lo",
    "coverage_hi", "n_unanswered_miss", "n_wrong", "wrong_rate", "wrong_lo", "wrong_hi",
    "n_exact", "n_under", "n_over", "n_with_missed_instance", "n_under_miss_only", "n_under_miss_and_other",
    "n_under_other", "n_masked_miss", "n_over_despite_miss", "missed_instances",
    "n_open_answered", "n_open_only", "n_open_wrong", "n_answer_object_missed",
    "n_silent_miss", "silent_miss_share_of_wrong", "silent_lo", "silent_hi")
"""tuple of str: columns of :func:`effects_table`."""


def effects_table(questions: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """How misses affect answers, per labeller, source, dataset and question type (and all types).

    Parameters
    ----------
    questions : pandas.DataFrame
        Rows of :func:`question_rows`.
    level : float, default 0.95
        Level of the Wilson intervals.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`EFFECT_COLUMNS`:

        - every type: ``coverage`` (answered / questions; every unanswered question has a missed referent, so
          ``n_unanswered_miss`` is the visible damage) and ``wrong_rate`` (wrong / answered, pipeline answers);
        - ``count``: exact / under / over counts and the role of misses (:func:`count_effect`); ``n_silent_miss``
          counts wrong counts with a missed instance, and ``silent_miss_share_of_wrong`` is their share of the
          wrong counts;
        - ``closest_instance``: ``n_open_answered`` (strict plus open-world answers), ``n_open_only``,
          ``n_open_wrong``; ``n_silent_miss`` counts wrong open-world answers whose answer object was missed, and
          ``silent_miss_share_of_wrong`` is their share of ``n_open_wrong``;
        - the ``qtype == "all"`` row sums the counts; its silent share is left empty (the two definitions differ).
    """
    rows: list[dict[str, Any]] = []
    if questions.empty:
        return pd.DataFrame(rows, columns=list(EFFECT_COLUMNS))
    for (labeller, source, dataset), frame in questions.groupby(["labeller", "source", "dataset"], sort=True):
        groups = [(str(q), g) for q, g in frame.groupby("qtype", sort=True)] + [("all", frame)]
        for qtype, g in groups:
            answered = g["answered"].astype(bool)
            n, k = len(g), int(answered.sum())
            wrong = int(_flag_is(g.loc[answered, "correct"], False).sum())
            coverage, c_lo, c_hi = rate(k, n, level)
            wrong_rate, w_lo, w_hi = rate(wrong, k, level)
            row: dict[str, Any] = {"labeller": labeller, "source": source, "dataset": dataset, "qtype": qtype,
                                   "n_questions": n, "n_answered": k, "coverage": coverage, "coverage_lo": c_lo,
                                   "coverage_hi": c_hi, "n_unanswered_miss": n - k, "n_wrong": wrong,
                                   "wrong_rate": wrong_rate, "wrong_lo": w_lo, "wrong_hi": w_hi,
                                   "n_silent_miss": int(g["silent_miss"].astype(bool).sum())}
            if qtype == "count":
                outcome, role = g["count_outcome"], g["count_miss_role"]
                row.update({"n_exact": int((outcome == "exact").sum()), "n_under": int((outcome == "under").sum()),
                            "n_over": int((outcome == "over").sum()),
                            "n_with_missed_instance": int((g["count_missed"] > 0).sum()),
                            "n_under_miss_only": int((role == "under_miss_only").sum()),
                            "n_under_miss_and_other": int((role == "under_miss_and_other").sum()),
                            "n_under_other": int(((outcome == "under") & (role == "none")).sum()),
                            "n_masked_miss": int((role == "masked").sum()),
                            "n_over_despite_miss": int((role == "over_despite_miss").sum()),
                            "missed_instances": int(g["count_missed"].sum())})
                share = rate(row["n_silent_miss"], wrong, level)
                row.update({"silent_miss_share_of_wrong": share[0], "silent_lo": share[1], "silent_hi": share[2]})
            elif qtype == "closest_instance":
                status = g["open_status"]
                open_answered = status.isin(["strict", "open"])
                open_wrong = int((open_answered & _flag_is(g["open_correct"], False)).sum())
                row.update({"n_open_answered": int(open_answered.sum()), "n_open_only": int((status == "open").sum()),
                            "n_open_wrong": open_wrong,
                            "n_answer_object_missed": int(g["closest_answer_missed"].astype(bool).sum())})
                share = rate(row["n_silent_miss"], open_wrong, level)
                row.update({"silent_miss_share_of_wrong": share[0], "silent_lo": share[1], "silent_hi": share[2]})
            rows.append(row)
    return pd.DataFrame(rows, columns=list(EFFECT_COLUMNS))


# --------------------------------------------------------------------------------------------------------
# Flags: shared parts
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FlagConfig:
    """Settings of the miss flags.

    Parameters
    ----------
    min_iou : float, default 0.3
        2D IoU at which a labelled box covers a flag (so there is none), and at which a flag hits a ground-truth
        object's visible box; the pipeline's association threshold.
    dedupe_iou : float, default 0.25
        Gravity-prism 3D IoU (:func:`~measured_space.matching.prism_iou`) at which two of the labeller's boxes
        from different views or cameras are taken for one object.
    flag_nms_iou : float, default 0.5
        2D IoU at which two projected flags in one view are one; the one with more support, then the higher
        score, is kept.
    max_truncated_fraction : float, default 0.5
        A projected box is flagged only if at most this share of its camera-facing surface falls outside the image
        or behind the camera ("mostly inside the image").
    min_unoccluded_fraction : float, default 0.5
        ... and at least this share of its in-image camera-facing surface is unoccluded according to the view's
        depth (ignored without depth).
    occlusion_margin : float, default 0.3
        Metres a surface sample may lie behind the depth map and still count as unoccluded, on top of
        :func:`~measured_space.visibility.depth_tolerance`. It absorbs the labeller's own box error: a box lifted
        in another view can sit a little behind the surface it was lifted from.
    min_support : int, default 1
        A projected box needs boxes from at least this many other views (or cameras).
    spacing : float, default 0.05
        Surface-sample spacing of the projection test, metres.
    score_thresholds : tuple of float, default (0.05, 0.25, 0.5)
        Witness-score thresholds at which cross-labeller flag precision and recall are also reported.
    support_thresholds : tuple of int, default (1, 2, 3)
        Support thresholds at which projected-flag precision and recall are also reported.
    label_iou_thresholds : tuple of float, default (0.1,)
        Flags whose largest 2D IoU with the flagged labeller's labels is below each of these are also reported
        (flags clear of every label, a filter that needs no ground truth).
    near_iou : float, default 0.1
        An ``unannotated`` flag whose best overlap with a present object reaches this is counted as *near* an
        annotated object: more likely a mislocalised box of that object than an unannotated one.
    """

    min_iou: float = 0.3
    dedupe_iou: float = 0.25
    flag_nms_iou: float = 0.5
    max_truncated_fraction: float = 0.5
    min_unoccluded_fraction: float = 0.5
    occlusion_margin: float = 0.3
    min_support: int = 1
    spacing: float = 0.05
    score_thresholds: tuple[float, ...] = (0.05, 0.25, 0.5)
    support_thresholds: tuple[int, ...] = (1, 2, 3)
    label_iou_thresholds: tuple[float, ...] = (0.1,)
    near_iou: float = 0.1

    def __post_init__(self) -> None:
        """Validate the thresholds.

        Raises
        ------
        ValueError
            If an IoU or fraction is outside (0, 1], the margin or spacing is not positive, or ``min_support``
            is below 1.
        """
        for name in ("min_iou", "dedupe_iou", "flag_nms_iou", "max_truncated_fraction", "near_iou"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {getattr(self, name)}")
        if not 0.0 <= self.min_unoccluded_fraction <= 1.0:
            raise ValueError(f"min_unoccluded_fraction must be in [0, 1], got {self.min_unoccluded_fraction}")
        if self.occlusion_margin < 0 or self.spacing <= 0 or self.min_support < 1:
            raise ValueError("occlusion_margin must be >= 0, spacing > 0 and min_support >= 1")

    def to_json(self) -> dict[str, Any]:
        """The settings as a JSON-serialisable dict.

        Returns
        -------
        dict
            Every field.
        """
        return asdict(self)


@dataclass
class FlagResult:
    """Flags and their ground-truth evaluation for some views.

    Parameters
    ----------
    flags : list of dict
        One row per flag: identity, ``kind``, ``labeller`` and ``source`` (whose misses the flag targets),
        ``witness`` (the evidence), ``x0, y0, x1, y1`` (the flag's 2D box), ``score``, ``support``, ``phrase``,
        ``annotated_class`` (whether the phrase maps to a dataset class; ``None`` without a phrase), ``origin``
        (the view the evidence came from), ``max_label_iou`` (largest 2D IoU with the flagged labeller's labels in
        the view; below ``min_iou`` by construction), ``category`` (one of :data:`FLAG_CATEGORIES`), ``hit_ids``
        (missed objects it overlaps, ``;``-joined), ``best_gt_iou`` (largest IoU with a present object's visible
        box) and ``best_missed_iou`` (with a missed one).
    targets : list of dict
        One row per present ground-truth object of a covered view: identity, ``kind``, ``labeller``,
        ``source``, ``witness``, ``object_id``, ``label``, ``clearly_visible``, ``range_bin``, ``missed`` (by the
        flagged labeller), ``n_flags`` (flags overlapping it at ``min_iou``; for a miss, those that hit it) and
        ``flaggable`` (projected flags only, for a miss: whether its ground-truth box itself passes the projection
        test, i.e. whether the method could catch it at all; ``None`` otherwise).
    views : list of dict
        One row per covered view: identity, ``kind``, ``labeller``, ``source``, ``witness``, ``n_flags``,
        ``n_candidates`` (evidence boxes considered).
    """

    flags: list[dict[str, Any]] = field(default_factory=list)
    targets: list[dict[str, Any]] = field(default_factory=list)
    views: list[dict[str, Any]] = field(default_factory=list)

    def extend(self, other: FlagResult) -> None:
        """Append another result's rows.

        Parameters
        ----------
        other : FlagResult
            Rows to append.
        """
        self.flags += other.flags
        self.targets += other.targets
        self.views += other.views


@dataclass(frozen=True)
class FlagEvaluation:
    """What a view's flags point at, by ground truth.

    Parameters
    ----------
    category : list of str
        Per flag, one of :data:`FLAG_CATEGORIES`.
    hit_ids : list of tuple of str
        Per flag, the missed objects it overlaps at the threshold.
    best_iou : list of float
        Per flag, its largest 2D IoU with a present object's visible box (0 if there is none).
    best_missed_iou : list of float
        Per flag, its largest 2D IoU with a missed object's visible box (0 if there is none).
    n_flags : list of int
        Per present object, how many flags overlap it at the threshold.
    """

    category: list[str]
    hit_ids: list[tuple[str, ...]]
    best_iou: list[float]
    best_missed_iou: list[float]
    n_flags: list[int]


def evaluate_flags(flag_boxes: ArrayLike, object_boxes: ArrayLike, object_ids: Sequence[str],
                   missed: Sequence[bool], min_iou: float) -> FlagEvaluation:
    """Compare a view's flags with its present ground-truth objects.

    Parameters
    ----------
    flag_boxes : array_like, shape (F, 4)
        The flags' 2D boxes.
    object_boxes : array_like, shape (P, 4)
        The present objects' visible 2D boxes, same convention.
    object_ids : sequence of str
        The present objects' ids.
    missed : sequence of bool
        Whether the flagged labeller missed each present object.
    min_iou : float
        Overlap at which a flag points at an object.

    Returns
    -------
    FlagEvaluation
        A flag is ``missed_object`` if it overlaps a missed object, else ``found_object`` if it overlaps a paired
        one, else ``unannotated``.
    """
    flags = np.asarray(flag_boxes, dtype=np.float64).reshape(-1, 4)
    objects = np.asarray(object_boxes, dtype=np.float64).reshape(-1, 4)
    miss = np.asarray(missed, dtype=bool).reshape(-1)
    iou = box_iou_2d(flags, objects)                                                # (F, P)
    overlap = iou >= min_iou
    hits = overlap & miss[None, :]
    category = ["missed_object" if hits[f].any() else ("found_object" if overlap[f].any() else "unannotated")
                for f in range(len(flags))]
    hit_ids = [tuple(object_ids[p] for p in np.flatnonzero(hits[f])) for f in range(len(flags))]
    best = [float(iou[f].max()) if objects.size else 0.0 for f in range(len(flags))]
    best_missed = [float(iou[f, miss].max()) if miss.any() else 0.0 for f in range(len(flags))]
    return FlagEvaluation(category, hit_ids, best, best_missed, [int(n) for n in overlap.sum(axis=0)])


def _evaluated(result: ViewResult, kind: str, labeller: str, source: str, witness: str,
               flags: Sequence[dict[str, Any]], missed_ids: Collection[str], min_iou: float,
               n_candidates: int, flaggable: Mapping[str, bool] | None = None) -> FlagResult:
    """Evaluate one view's flags against its present objects and build the result rows.

    Parameters
    ----------
    result : ViewResult
        The view.
    kind : str
        One of :data:`FLAG_KINDS`.
    labeller : str
        The flagged labeller.
    source : str
        The flagged labeller's source.
    witness : str
        The evidence's name.
    flags : sequence of dict
        One dict per flag with ``box`` (the 2D box), ``score``, ``support``, ``phrase``, ``labels``, ``origin`` and
        ``max_label_iou``.
    missed_ids : collection of str
        The present objects the flagged labeller missed.
    min_iou : float
        Overlap at which a flag points at an object.
    n_candidates : int
        Evidence boxes considered in the view.
    flaggable : mapping of str to bool, optional
        Missed object id -> whether the method could flag it at all (projected flags only).

    Returns
    -------
    FlagResult
        The view's flag, target and view rows.
    """
    present = present_objects(result)
    edges = [visible_box_edges(o.bbox2d, result.visibility) for o in present]               # type: ignore[arg-type]
    ids = [o.obj.object_id for o in present]
    missed = [i in missed_ids for i in ids]
    evaluation = evaluate_flags([f["box"] for f in flags], edges, ids, missed, min_iou)
    base = {**identity(result), "kind": kind, "labeller": labeller, "source": source, "witness": witness}
    flag_rows = []
    for f, category, hit, best, best_missed in zip(flags, evaluation.category, evaluation.hit_ids,
                                                   evaluation.best_iou, evaluation.best_missed_iou):
        x0, y0, x1, y1 = (float(v) for v in f["box"])
        labels = f.get("labels")
        flag_rows.append({**base, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "score": f.get("score"),
                          "support": f.get("support"), "phrase": f.get("phrase"),
                          "annotated_class": None if labels is None else bool(labels), "origin": f.get("origin"),
                          "max_label_iou": f.get("max_label_iou"), "category": category, "hit_ids": ";".join(hit),
                          "best_gt_iou": best, "best_missed_iou": best_missed})
    camera = result.view.camera_center
    target_rows = [{**base, "object_id": o.obj.object_id, "label": o.obj.label,
                    "clearly_visible": clearly_visible(o, result.config.qa),
                    "range_bin": range_bin(float(np.linalg.norm(o.obj.box.center - camera))),
                    "missed": m, "n_flags": n,
                    "flaggable": None if flaggable is None or not m else flaggable.get(o.obj.object_id)}
                   for o, m, n in zip(present, missed, evaluation.n_flags)]
    return FlagResult(flag_rows, target_rows, [{**base, "n_flags": len(flags), "n_candidates": n_candidates}])


# --------------------------------------------------------------------------------------------------------
# Cross-labeller flags
# --------------------------------------------------------------------------------------------------------


def cross_labeller_flags(result: ViewResult, source: str, labeller: str, predictions: ViewPredictions,
                         witness: str, config: FlagConfig | None = None) -> FlagResult:
    """Flags from another labeller's boxes in the same view, in both directions.

    The flagged labeller ("A") is ``labeller``/``source``: its labels are the detections with a box from the source.
    The witness ("B") is a CuTR predictions set: its boxes are the predicted boxes, 2D boxes clipped to the image.

    - **B sees, A does not:** every B box with no A label at 2D IoU >= ``config.min_iou`` is a flag on A
      (``labeller``, ``source``; ``witness`` = the predictions' name). It hits when it overlaps an object A missed.
    - **A sees, B does not:** every A label with no B box at that IoU is a flag on B (``labeller`` = the
      predictions' name, ``source`` = :data:`CUTR_BOXES_SOURCE`; ``witness`` = ``"<labeller>/<source>"``). B's
      misses come from pairing its boxes with the present objects by
      :func:`~measured_space.matching.match_detections_2d` (class-agnostic, same threshold): CuTR's raw boxes,
      not the ``cutr_sam3`` / ``cutr_clip`` sources.

    Parameters
    ----------
    result : ViewResult
        The view, re-associated.
    source : str
        A's source.
    labeller : str
        A's name.
    predictions : ViewPredictions
        B's boxes for this view.
    witness : str
        B's name (e.g. the predictions folder's name).
    config : FlagConfig, optional
        Settings; defaults to ``FlagConfig()``.

    Returns
    -------
    FlagResult
        Rows for both directions (empty if the view lacks the source).

    Raises
    ------
    ValueError
        If the predictions belong to another view or disagree with its pose
        (:func:`measured_space.box_sources.check_predictions`).
    """
    config = config or FlagConfig()
    if source not in result.sources:
        return FlagResult()
    cam = result.view.camera
    check_predictions({"view_key": result.key, "camera": {"width": cam.width, "height": cam.height},
                       "T_wc": result.view.T_wc.tolist()}, predictions)
    b_boxes = np.array([clip_box2d(b.box2d, cam.width, cam.height) for b in predictions.boxes],
                       dtype=np.float64).reshape(-1, 4)
    a_index, a_boxes = labelled_boxes2d(result, source)
    iou = box_iou_2d(a_boxes, b_boxes)                                                     # (A, B)
    out = FlagResult()

    best_a = iou.max(axis=0) if len(a_boxes) else np.zeros(len(b_boxes))
    best_b = iou.max(axis=1) if len(b_boxes) else np.zeros(len(a_boxes))
    b_flags = [{"box": b_boxes[j], "score": predictions.boxes[j].score, "support": None, "phrase": None,
                "labels": None, "origin": result.key, "max_label_iou": float(best_a[j])}
               for j in range(len(b_boxes)) if best_a[j] < config.min_iou]
    out.extend(_evaluated(result, "cross_labeller", labeller, source, witness, b_flags,
                          set(result.matches[source].missed), config.min_iou, len(b_boxes)))

    present = present_objects(result)
    edges = [visible_box_edges(o.bbox2d, result.visibility) for o in present]               # type: ignore[arg-type]
    b_match = match_detections_2d(list(b_boxes), [b.box for b in predictions.boxes], [()] * len(b_boxes),
                                  [o.obj for o in present], edges, Match2DConfig(config.min_iou), up=result.view.up)
    a_flags = []
    for row, i in enumerate(a_index):
        if best_b[row] >= config.min_iou:
            continue
        det = result.detections[i]
        a_flags.append({"box": a_boxes[row], "score": det.instance.score, "support": None,
                        "phrase": det.instance.phrase, "labels": det.labels, "origin": result.key,
                        "max_label_iou": float(best_b[row])})
    out.extend(_evaluated(result, "cross_labeller", witness, CUTR_BOXES_SOURCE, f"{labeller}/{source}", a_flags,
                          set(b_match.missed), config.min_iou, len(a_index)))
    return out


# --------------------------------------------------------------------------------------------------------
# Projection flags: cross-frame and cross-camera
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A labeller box from another view (or camera), proposed for projection into a view.

    Parameters
    ----------
    box : OrientedBox
        World box, metres.
    score : float
        Its detection's score.
    support : int
        How many other views (or cameras) have a box of the same object (its 3D cluster).
    phrase : str
        Its detection's phrase.
    labels : tuple of str
        The dataset labels the phrase may stand for (empty for a phrase outside the dataset's classes).
    origin : str
        The view key of the view it came from.
    """

    box: OrientedBox
    score: float
    support: int
    phrase: str
    labels: tuple[str, ...]
    origin: str


@dataclass(frozen=True)
class _SourceBox:
    """One labelled box of one view, for clustering across views.

    Parameters
    ----------
    group : str
        What a candidate must come from outside of: the view key (cross-frame) or the camera (cross-camera).
    view_key : str
        The view it came from.
    box : OrientedBox
        World box.
    score : float
        Detection score.
    phrase : str
        Detection phrase.
    labels : tuple of str
        Dataset labels of the phrase.
    """

    group: str
    view_key: str
    box: OrientedBox
    score: float
    phrase: str
    labels: tuple[str, ...]


def _source_boxes(results: Sequence[ViewResult], source: str, group: Callable[[ViewResult], str]
                  ) -> list[_SourceBox]:
    """Every labelled box of some views.

    Parameters
    ----------
    results : sequence of ViewResult
        The views.
    source : str
        The labeller's source; views without it contribute nothing.
    group : callable
        ``group(result) -> str``, the box's group (view key or camera).

    Returns
    -------
    list of _SourceBox
        One per detection with a box from the source.
    """
    out = []
    for r in results:
        if source not in r.sources:
            continue
        for det, lift in zip(r.detections, r.lifts[source]):
            if lift.box is not None:
                out.append(_SourceBox(group(r), r.key, lift.box, float(det.instance.score), det.instance.phrase,
                                      det.labels))
    return out


def cluster_boxes(boxes: Sequence[OrientedBox], scores: Sequence[float], iou_threshold: float,
                  up: ArrayLike = (0.0, 0.0, 1.0)) -> NDArray[np.int64]:
    """Group boxes of the same object by greedy 3D overlap (highest score first).

    Each box, in order of decreasing score, joins the existing cluster whose first (seed) box it overlaps most,
    if that gravity-prism IoU (:func:`~measured_space.matching.prism_iou`) reaches ``iou_threshold``; otherwise
    it seeds a new cluster.

    Parameters
    ----------
    boxes : sequence of OrientedBox
        World boxes.
    scores : sequence of float
        One score per box.
    iou_threshold : float
        Prism IoU at which a box joins a cluster.
    up : array_like, shape (3,), default (0, 0, 1)
        World up direction.

    Returns
    -------
    numpy.ndarray of int, shape (n,)
        Cluster index of each box (clusters numbered in seed order).
    """
    n = len(boxes)
    cluster = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return cluster
    up_vec = np.asarray(up, dtype=np.float64)
    # With up along world z, a gravity prism lies inside its corners' axis-aligned bounds, so disjoint bounds
    # mean IoU 0 and the exact test can be skipped.
    prefilter = bool(np.allclose(up_vec / np.linalg.norm(up_vec), (0.0, 0.0, 1.0)))
    corners = np.stack([b.corners() for b in boxes])
    lo, hi = corners.min(axis=1), corners.max(axis=1)
    seeds: list[int] = []
    for i in np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable"):
        best, best_iou = -1, 0.0
        for c, s in enumerate(seeds):
            if prefilter and (np.any(lo[i] >= hi[s]) or np.any(lo[s] >= hi[i])):
                continue
            value = prism_iou(boxes[i], boxes[s], up)
            if value > best_iou:
                best, best_iou = c, value
        if best >= 0 and best_iou >= iou_threshold:
            cluster[i] = best
        else:
            cluster[i] = len(seeds)
            seeds.append(int(i))
    return cluster


def _candidates(boxes: Sequence[_SourceBox], cluster: NDArray[np.int64], exclude: str) -> list[Candidate]:
    """One candidate per cluster with a box from outside ``exclude``: its best-scoring such box.

    Parameters
    ----------
    boxes : sequence of _SourceBox
        The boxes.
    cluster : numpy.ndarray of int
        Cluster index per box (:func:`cluster_boxes`).
    exclude : str
        The receiving view's group; its own boxes neither propose nor support a candidate.

    Returns
    -------
    list of Candidate
        In cluster order; ``support`` counts the distinct other groups in the cluster.
    """
    members: dict[int, list[int]] = {}
    for i, (b, c) in enumerate(zip(boxes, cluster)):
        if b.group != exclude:
            members.setdefault(int(c), []).append(i)
    out = []
    for c in sorted(members):
        rep = max(members[c], key=lambda i: (boxes[i].score, -i))
        b = boxes[rep]
        out.append(Candidate(b.box, b.score, len({boxes[i].group for i in members[c]}), b.phrase, b.labels,
                             b.view_key))
    return out


def could_appear(view: FrameView, box: OrientedBox) -> bool:
    """A cheap test that a box may project into the image (before the sampled test).

    Parameters
    ----------
    view : FrameView
        The view.
    box : OrientedBox
        World box.

    Returns
    -------
    bool
        False if every corner is behind the camera, or every corner is in front and their 2D extent misses the
        image; True otherwise.
    """
    uv, z = view.camera.project(box.transformed(view.T_cw).corners())
    if np.all(z <= _MIN_Z):
        return False
    if np.any(z <= _MIN_Z):
        return True
    cam = view.camera
    return bool(uv[:, 0].max() >= -0.5 and uv[:, 0].min() < cam.width - 0.5
                and uv[:, 1].max() >= -0.5 and uv[:, 1].min() < cam.height - 0.5)


def project_box(view: FrameView, box: OrientedBox, depth: NDArray[np.floating] | None, config: FlagConfig
                ) -> tuple[Box2D | None, dict[str, float]]:
    """Where a world box would appear in a view, and whether the view would see it.

    The test is :func:`measured_space.visibility.object_in_view` on the box, with the depth map pushed back by
    ``config.occlusion_margin`` where valid, so a sample counts as unoccluded unless the depth map is nearer than
    it by more than the tolerance plus the margin. Samples without valid depth count as occluded.

    Parameters
    ----------
    view : FrameView
        The receiving view.
    box : OrientedBox
        World box.
    depth : numpy.ndarray, shape (h, w), or None
        The view's planar depth in metres (any resolution; 0 or NaN invalid); ``None`` tests geometry only.
    config : FlagConfig
        Supplies the truncation, occlusion and sampling settings.

    Returns
    -------
    box2d : tuple of float or None
        The 2D box of the visible samples (pixel edges at integer coordinates) if the box passes, else ``None``.
    info : dict of str to float
        ``visible_fraction``, ``truncated_fraction`` and ``unoccluded_fraction`` (visible share of the in-image
        samples; 1 without depth).
    """
    pushed = None
    if depth is not None:
        d = np.asarray(depth, dtype=np.float64)
        valid = np.isfinite(d) & (d > 0)
        pushed = np.where(valid, d + config.occlusion_margin, 0.0)
    seen, _ = object_in_view(view, SceneObject("candidate", "", box), pushed, spacing=config.spacing)
    inside = 1.0 - seen.truncated_fraction
    unoccluded = seen.visible_fraction / inside if inside > 0 else 0.0
    info = {"visible_fraction": seen.visible_fraction, "truncated_fraction": seen.truncated_fraction,
            "unoccluded_fraction": unoccluded}
    passes = (seen.bbox2d is not None and seen.truncated_fraction <= config.max_truncated_fraction
              and unoccluded >= config.min_unoccluded_fraction)
    if not passes:
        return None, info
    return visible_box_edges(seen.bbox2d, _PROJECTION_CONVENTION), info                   # type: ignore[arg-type]


def projection_flags(result: ViewResult, source: str, labeller: str, kind: str, witness: str,
                     candidates: Sequence[Candidate], depth: Callable[[], NDArray[np.floating] | None],
                     config: FlagConfig) -> FlagResult:
    """Flag the candidates that the view should show but the labeller has no box for, and evaluate them.

    A candidate with enough support that passes :func:`project_box` becomes a flag unless a labelled box of the
    view overlaps its projection at ``config.min_iou``. Flags overlapping at ``config.flag_nms_iou`` are merged,
    keeping the one with more support, then the higher score. For the evaluation only, each missed object's
    ground-truth box is put through the same projection test (``flaggable``).

    Parameters
    ----------
    result : ViewResult
        The receiving view, re-associated.
    source : str
        The labeller's source.
    labeller : str
        The labeller's name.
    kind : str
        ``"cross_frame"`` or ``"cross_camera"``.
    witness : str
        The evidence's name.
    candidates : sequence of Candidate
        Boxes from other views or cameras.
    depth : callable
        ``depth() -> map or None``, called at most once: when some candidate may appear in the view, or to test
        whether the view's misses were flaggable.
    config : FlagConfig
        Settings.

    Returns
    -------
    FlagResult
        The view's rows; flags carry ``support``, ``score``, ``phrase``, ``labels`` and ``origin`` of their candidate.
    """
    _, labelled = labelled_boxes2d(result, source)
    view = result.view
    cache: list[NDArray[np.floating] | None] = []

    def depth_map() -> NDArray[np.floating] | None:
        """The view's depth, loaded on first use.

        Returns
        -------
        numpy.ndarray or None
            ``depth()``.
        """
        if not cache:
            cache.append(depth())
        return cache[0]

    proposals: list[tuple[Box2D, Candidate, float]] = []
    for c in candidates:
        if c.support < config.min_support or not could_appear(view, c.box):
            continue
        box2d, _ = project_box(view, c.box, depth_map(), config)
        if box2d is None:
            continue
        label_iou = float(box_iou_2d(np.asarray([box2d]), labelled).max()) if labelled.size else 0.0
        if label_iou >= config.min_iou:
            continue
        proposals.append((box2d, c, label_iou))
    proposals.sort(key=lambda p: (-p[1].support, -p[1].score))
    kept: list[tuple[Box2D, Candidate, float]] = []
    for box2d, c, label_iou in proposals:
        if kept and (box_iou_2d(np.asarray([box2d]), np.asarray([k[0] for k in kept])) >= config.flag_nms_iou).any():
            continue
        kept.append((box2d, c, label_iou))
    flags = [{"box": b, "score": c.score, "support": c.support, "phrase": c.phrase, "labels": c.labels,
              "origin": c.origin, "max_label_iou": label_iou} for b, c, label_iou in kept]
    missed = set(result.matches[source].missed)
    flaggable = {o.obj.object_id: project_box(view, o.obj.box, depth_map(), config)[0] is not None
                 for o in present_objects(result) if o.obj.object_id in missed}
    return _evaluated(result, kind, labeller, source, witness, flags, missed, config.min_iou, len(candidates),
                      flaggable)


def lidar_depth(view: FrameView) -> NDArray[np.float32]:
    """An ARKitScenes view's iPad LiDAR depth: the independent depth of the cross-frame test.

    Parameters
    ----------
    view : FrameView
        An ARKitScenes view.

    Returns
    -------
    numpy.ndarray, shape (h, w), float32
        Planar depth in metres at the sensor's resolution (0 invalid), from
        :func:`measured_space.datasets.arkitscenes.load_lidar_depth`.

    Raises
    ------
    ValueError
        For a view of another dataset.
    """
    if view.dataset != "arkitscenes":
        raise ValueError(f"LiDAR depth exists only for ARKitScenes views, not {view.dataset!r}")
    from measured_space.datasets import arkitscenes

    return arkitscenes.load_lidar_depth(view)


def cross_frame_flags(results: Sequence[ViewResult], source: str, labeller: str, config: FlagConfig | None = None,
                      *, depth_of: DepthFn | None = None) -> FlagResult:
    """"Seen elsewhere, missing here": project the labeller's boxes from the other views of a static video.

    All the labeller's boxes in the video are clustered by 3D overlap (:func:`cluster_boxes`). For each view, every
    cluster with a box from another view proposes its best-scoring such box, with ``support`` = the number of other
    views in the cluster, and :func:`projection_flags` decides, with the view's own depth for the occlusion test.

    Parameters
    ----------
    results : sequence of ViewResult
        Every view of one video of a static scene (:data:`STATIC_DATASETS`), re-associated.
    source : str
        The labeller's source (its boxes are projected, its labels checked).
    labeller : str
        The labeller's name.
    config : FlagConfig, optional
        Settings; defaults to ``FlagConfig()``.
    depth_of : callable, optional
        ``depth_of(view) -> depth``; defaults to :func:`lidar_depth`. It must be independent of the reference (for
        ARKitScenes: never the laser depth).

    Returns
    -------
    FlagResult
        Rows for every view that has the source (``kind="cross_frame"``, ``witness="other_frames"``).

    Raises
    ------
    ValueError
        If the views span several scenes or a dataset that is not static.
    """
    config = config or FlagConfig()
    if not results:
        return FlagResult()
    if len({(r.view.dataset, r.view.scene) for r in results}) != 1:
        raise ValueError("cross-frame flags take the views of one video")
    if results[0].view.dataset not in STATIC_DATASETS:
        raise ValueError(f"{results[0].view.dataset!r} scenes are not static; use cross-camera flags")
    depth_of = depth_of or lidar_depth
    boxes = _source_boxes(results, source, lambda r: r.key)
    cluster = cluster_boxes([b.box for b in boxes], [b.score for b in boxes], config.dedupe_iou,
                            results[0].view.up)
    out = FlagResult()
    for r in results:
        if source not in r.sources:
            continue
        candidates = _candidates(boxes, cluster, r.key)
        out.extend(projection_flags(r, source, labeller, "cross_frame", "other_frames", candidates,
                                    lambda view=r.view: depth_of(view), config))      # type: ignore[misc]
    return out


def cross_camera_flags(results: Sequence[ViewResult], source: str, labeller: str, config: FlagConfig | None = None,
                       *, depth_of: DepthFn | None = None) -> FlagResult:
    """"Seen by another camera, missing here": project the boxes of the other cameras at the same instant.

    For fixed, synchronised cameras that share frame indices and a world frame (Smart Spaces). Views are grouped
    by frame index; within an instant, the boxes of all cameras are clustered by 3D overlap and each view receives
    one candidate per cluster with a box from another camera (``support`` = the number of other cameras), and
    :func:`projection_flags` decides. A view whose instant has no other camera is not covered.

    The stored data allow this: the Smart Spaces label folders hold both cameras of every scene at the same frame
    indices. They hold no depth for the receiving camera that is independent of the reference (the simulator depth
    is the reference; MoGe depth maps are not stored), so by default the occlusion test is skipped and only geometry
    decides (in front of the camera, mostly inside the image). A multi-camera labeller with its own depth can pass
    ``depth_of``.

    Parameters
    ----------
    results : sequence of ViewResult
        Views of one scene, any cameras and frames, re-associated.
    source : str
        The labeller's source.
    labeller : str
        The labeller's name.
    config : FlagConfig, optional
        Settings; defaults to ``FlagConfig()``.
    depth_of : callable, optional
        ``depth_of(view) -> depth`` independent of the reference; ``None`` (default) tests geometry only.

    Returns
    -------
    FlagResult
        Rows for every covered view (``kind="cross_camera"``, ``witness="other_cameras"``).

    Raises
    ------
    ValueError
        If the views span several scenes.
    """
    config = config or FlagConfig()
    if not results:
        return FlagResult()
    if len({(r.view.dataset, r.view.scene) for r in results}) != 1:
        raise ValueError("cross-camera flags take the views of one scene")
    by_frame: dict[str, list[ViewResult]] = {}
    for r in results:
        by_frame.setdefault(r.view.frame, []).append(r)
    out = FlagResult()
    for frame in sorted(by_frame, key=_frame_order):
        instant = by_frame[frame]
        if len({r.view.camera_id for r in instant}) < 2:
            continue
        boxes = _source_boxes(instant, source, lambda r: r.view.camera_id)
        cluster = cluster_boxes([b.box for b in boxes], [b.score for b in boxes], config.dedupe_iou,
                                instant[0].view.up)
        for r in instant:
            if source not in r.sources:
                continue
            candidates = _candidates(boxes, cluster, r.view.camera_id)
            depth = (lambda: None) if depth_of is None else (lambda view=r.view: depth_of(view))  # type: ignore[misc]
            out.extend(projection_flags(r, source, labeller, "cross_camera", "other_cameras", candidates, depth,
                                        config))
    return out


# --------------------------------------------------------------------------------------------------------
# Flag metrics
# --------------------------------------------------------------------------------------------------------


_ROW_KEYS: tuple[str, ...] = ("dataset", "scene", "camera_id", "frame", "view_key", "kind", "labeller", "source",
                               "witness")

FLAG_ROW_COLUMNS: tuple[str, ...] = (*_ROW_KEYS, "x0", "y0", "x1", "y1", "score", "support", "phrase",
                                     "annotated_class", "origin", "max_label_iou", "category", "hit_ids",
                                     "best_gt_iou", "best_missed_iou")
"""tuple of str: columns of :attr:`FlagResult.flags` rows."""

TARGET_ROW_COLUMNS: tuple[str, ...] = (*_ROW_KEYS, "object_id", "label", "clearly_visible", "range_bin", "missed",
                                       "n_flags", "flaggable")
"""tuple of str: columns of :attr:`FlagResult.targets` rows."""

VIEW_ROW_COLUMNS: tuple[str, ...] = (*_ROW_KEYS, "n_flags", "n_candidates")
"""tuple of str: columns of :attr:`FlagResult.views` rows."""

FLAG_COLUMNS: tuple[str, ...] = (
    "kind", "labeller", "source", "witness", "dataset", "by", "group", "n_views", "n_flags", "flags_per_view",
    "n_hit", "precision", "precision_lo", "precision_hi", "n_found_object", "n_unannotated", "unannotated_share",
    "n_unannotated_near", "n_present", "n_missed", "labeller_recall", "n_missed_flagged", "flag_recall",
    "recall_lo", "recall_hi", "n_missed_clear", "n_missed_clear_flagged", "flag_recall_clear",
    "n_missed_flaggable", "n_missed_flaggable_flagged", "flag_recall_flaggable")
"""tuple of str: columns of :func:`flag_table`."""


def flag_table(flags: pd.DataFrame, targets: pd.DataFrame, views: pd.DataFrame, config: FlagConfig | None = None,
               level: float = 0.95) -> pd.DataFrame:
    """Flag precision and recall per kind, flagged labeller and source, witness and dataset.

    Parameters
    ----------
    flags : pandas.DataFrame
        :attr:`FlagResult.flags` rows.
    targets : pandas.DataFrame
        :attr:`FlagResult.targets` rows.
    views : pandas.DataFrame
        :attr:`FlagResult.views` rows (the covered views).
    config : FlagConfig, optional
        Supplies the score and support thresholds; defaults to ``FlagConfig()``.
    level : float, default 0.95
        Level of the Wilson intervals.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`FLAG_COLUMNS`. ``by`` is ``"all"``; ``"min_score"`` (flags whose witness score is at least
        ``group``) for cross-labeller flags with scores; ``"min_support"`` for projected flags;
        ``"max_label_iou"`` (flags whose largest IoU with the flagged labeller's labels is below ``group``); and
        ``"annotated_class"`` (``group`` ``"True"``: flags whose phrase maps to a dataset class). ``precision`` =
        flags that hit a miss / flags; ``n_unannotated_near`` counts the unannotated flags that overlap a present
        object at ``config.near_iou`` or more (likely mislocalised rather than unannotated); ``flag_recall`` =
        misses some flag of the subset hits / misses of the covered views, also for clearly visible misses and,
        for projected flags, for the *flaggable* misses (those whose ground-truth box passes the projection test);
        ``labeller_recall`` is the flagged labeller's recall in the covered views.
    """
    config = config or FlagConfig()
    keys = ["kind", "labeller", "source", "witness", "dataset"]
    flags = flags.reindex(columns=list(FLAG_ROW_COLUMNS)) if flags.empty else flags
    targets = targets.reindex(columns=list(TARGET_ROW_COLUMNS)) if targets.empty else targets
    rows: list[dict[str, Any]] = []
    if views.empty:
        return pd.DataFrame(rows, columns=list(FLAG_COLUMNS))
    flag_groups = dict(iter(flags.groupby(keys, sort=False)))
    target_groups = dict(iter(targets.groupby(keys, sort=False)))
    for key, v in views.groupby(keys, sort=True):
        select = dict(zip(keys, key))
        f = flag_groups.get(key, flags.iloc[:0])
        t = target_groups.get(key, targets.iloc[:0])
        missed = t[t["missed"].astype(bool)]
        clear = missed[missed["clearly_visible"].astype(bool)]
        flaggable = missed[_flag_is(missed["flaggable"], True)]
        miss_keys = set(zip(missed["view_key"], missed["object_id"]))
        clear_keys = set(zip(clear["view_key"], clear["object_id"]))
        flaggable_keys = set(zip(flaggable["view_key"], flaggable["object_id"]))
        has_flaggable = bool(missed["flaggable"].notna().any())
        subsets: list[tuple[str, str, pd.DataFrame]] = [("all", "all", f)]
        if key[0] == "cross_labeller" and f["score"].notna().any():
            subsets += [("min_score", f"{s:g}", f[f["score"].astype(float) >= s]) for s in config.score_thresholds]
        if f["support"].notna().any():
            subsets += [("min_support", str(s), f[f["support"].astype(float) >= s]) for s in config.support_thresholds]
        subsets += [("max_label_iou", f"{u:g}", f[f["max_label_iou"].astype(float) < u])
                    for u in config.label_iou_thresholds]
        if f["annotated_class"].notna().any():
            subsets.append(("annotated_class", "True", f[_flag_is(f["annotated_class"], True)]))
        for by, group, g in subsets:
            hit_keys = {(view_key, i) for view_key, ids in zip(g["view_key"], g["hit_ids"])
                        for i in (ids.split(";") if isinstance(ids, str) else []) if i}
            n_flags, n_present = len(g), len(t)
            n_hit = int((g["category"] == "missed_object").sum())
            unannotated = g["category"] == "unannotated"
            n_unannotated = int(unannotated.sum())
            precision = rate(n_hit, n_flags, level)
            recall = rate(len(hit_keys & miss_keys), len(miss_keys), level)
            rows.append({**select, "by": by, "group": group, "n_views": len(v), "n_flags": n_flags,
                         "flags_per_view": n_flags / len(v), "n_hit": n_hit, "precision": precision[0],
                         "precision_lo": precision[1], "precision_hi": precision[2],
                         "n_found_object": int((g["category"] == "found_object").sum()),
                         "n_unannotated": n_unannotated,
                         "unannotated_share": n_unannotated / n_flags if n_flags else None,
                         "n_unannotated_near": int((unannotated & (g["best_gt_iou"].astype(float)
                                                                   >= config.near_iou)).sum()),
                         "n_present": n_present, "n_missed": len(miss_keys),
                         "labeller_recall": 1 - len(miss_keys) / n_present if n_present else None,
                         "n_missed_flagged": len(hit_keys & miss_keys), "flag_recall": recall[0],
                         "recall_lo": recall[1], "recall_hi": recall[2], "n_missed_clear": len(clear_keys),
                         "n_missed_clear_flagged": len(hit_keys & clear_keys),
                         "flag_recall_clear": rate(len(hit_keys & clear_keys), len(clear_keys), level)[0],
                         "n_missed_flaggable": len(flaggable_keys) if has_flaggable else None,
                         "n_missed_flaggable_flagged": len(hit_keys & flaggable_keys) if has_flaggable else None,
                         "flag_recall_flaggable": (rate(len(hit_keys & flaggable_keys), len(flaggable_keys), level)[0]
                                                   if has_flaggable else None)})
    return pd.DataFrame(rows, columns=list(FLAG_COLUMNS))
