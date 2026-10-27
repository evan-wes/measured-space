"""A static HTML viewer of ground-truth objects, automatic labels and their pairing, and a spot-check page.

Pairing a frame's automatic detections with its ground-truth objects is a step we compute, not ground truth
(``docs/LEARNINGS.md``, L7): Hungarian assignment on the 2D IoU between a detection's box and the object's
*visible* 2D box, class-agnostic, at IoU >= 0.3. This module shows that step so a person can check it:

- **Per-view overlays.** For each present ground-truth object: its projected 3D box (solid) and visible 2D box
  (dashed), labelled with its class and a short id. For each detection of the *pairing source*: its 2D box,
  phrase and score. For every chosen automatic source: its lifted 3D boxes, projected, in the source's colour.
  A line joins each paired detection to its object's label: green if the phrase maps to the object's class,
  orange if not, red and dashed if the pair is *ambiguous* (another object or detection also cleared the IoU
  threshold, :attr:`~measured_space.matching.Match.ambiguous`). A legend strip sits under the image.
- **``index.html``.** Per scene, the views as thumbnails linking to the full overlays, and per view a table of
  the present objects with, per source, the paired detection, its 2D IoU, the ambiguity flag, the 3D centre
  distance and the label check. Two filters keep only views with an ambiguous pair or with a label mismatch.
- **``spotcheck.html``.** A seeded random sample of pairs of one source, half ambiguous and half clear,
  stratified across scenes (:func:`sample_pairs`). Each card shows a crop around the pair, the full frame,
  the ground-truth label and the detection's phrase, and asks :data:`QUESTION` with three verdicts. Answers are
  kept in the browser's ``localStorage`` and exported as JSON; :func:`summarize_answers` turns an export into
  wrong-pair rates with 95% Wilson intervals.

**Inputs.** Label folders written by ``scripts/run_labels.py`` or ``scripts/run_box_source.py``: one shard per
view, ``<labels>/<dataset>/shards/<view>.json``, holding ``{"records": ViewResult.records(), ...}``. Several
folders built on the same views can be combined (:func:`load_view_records`): the first folder chooses the
views, the others are matched to them by shard name and ``view_key``, and each automatic source is read from
the first folder that has it. The folders must agree about the view and its ground-truth objects.

**Pairing is recomputed, never read.** For every view and source, the detections, lifts and oracle objects are
rebuilt from the stored records (:func:`~measured_space.pipeline.view_from_record`,
:func:`~measured_space.pipeline.oracle_from_records`, :func:`~measured_space.pipeline.detections_from_records`)
and associated again with :func:`~measured_space.pipeline.match_view` under the ``"2d"`` association, with the
overlap threshold stored with that source's records. The stored ``match`` fields are not read, so the viewer
shows what the current association code computes, also for folders written under the 3D association. As in
the pipeline, only the present objects (:func:`~measured_space.qa.visible_ids`) take part, and a detection
without a box from a source takes no part in that source's pairing.

**Outputs** (:func:`build_viewer`) hold images derived from the datasets, so they are written only to the
folder the caller names, which must be new or empty; ``scripts/make_viewer.py`` puts it at
``<data_root>/outputs/viewer/<name>/``, outside the repository. The HTML is self-contained: inline CSS and
JavaScript, no external resources.

Pixel conventions are those of :mod:`measured_space.geometry` (pixel centres at integer coordinates for
projections) and :attr:`measured_space.perception.Instance.box2d` (pixel edges at integer coordinates for 2D
boxes). Distances are in metres.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from measured_space.datasets import load_rgb
from measured_space.geometry import OrientedBox
from measured_space.matching import MatchResult
from measured_space.pipeline import (
    Detection,
    Lift,
    PipelineConfig,
    detections_from_records,
    match_view,
    oracle_from_records,
    view_from_record,
    visible_box_edges,
)
from measured_space.qa import visible_ids
from measured_space.scene import FrameView, ObjectInView

Point = tuple[float, float]
"""tuple of float: an image point ``(x, y)`` in pixels."""

Segment = tuple[Point, Point]
"""tuple of Point: a line segment between two image points."""

Rect = tuple[float, float, float, float]
"""tuple of float: an image box ``(xmin, ymin, xmax, ymax)`` in pixels."""

Colour = tuple[int, int, int]
"""tuple of int: an RGB colour, 0-255 per channel."""

ImageLoader = Callable[[FrameView], NDArray[np.uint8]]
"""Callable: ``load_image(view) -> (H, W, 3) uint8 RGB`` in the view's orientation."""

DEPLOYMENT_SOURCES: dict[str, str] = {"arkitscenes": "lidar", "smartspaces": "moge"}
"""dict of str to str: the depth source a deployed labeller would have, per dataset (default source)."""

QUESTION = "Is this detection the same physical object as this ground-truth object?"
"""str: the question each spot-check card asks."""

GUIDANCE: tuple[str, ...] = (
    "Right pair: the white detection box and the cyan ground-truth box (dashed: its visible part; solid: its "
    "3D box) are on the same physical object, even if a box is loose or the words differ.",
    "Wrong pair: the detection is mainly on another object, on a group the ground-truth object is only part "
    "of, on a small part of it, or on background.",
    "Can't tell: the image does not let you decide (too small, dark, blurred or occluded). Judge identity "
    "only; label wording and box tightness do not matter here.",
)
"""tuple of str: the three lines of guidance shown under :data:`QUESTION`."""

VERDICTS: tuple[str, ...] = ("right", "wrong", "cant_tell")
"""tuple of str: spot-check verdicts (right pair, wrong pair, can't tell)."""

GT_COLOUR: Colour = (0, 229, 255)
"""Colour: ground-truth boxes (cyan)."""

DET_COLOUR: Colour = (255, 255, 255)
"""Colour: detection boxes of the pairing source (white)."""

UNBOXED_COLOUR: Colour = (170, 170, 170)
"""Colour: detections without a box from the pairing source (grey, dashed)."""

PAIR_OK: Colour = (46, 204, 64)
"""Colour: a pair whose phrase maps to the object's class (green)."""

PAIR_MISLABEL: Colour = (255, 133, 27)
"""Colour: a pair whose phrase does not map to the object's class (orange)."""

PAIR_AMBIGUOUS: Colour = (255, 65, 54)
"""Colour: an ambiguous pair (red, dashed)."""

SOURCE_COLOURS: dict[str, Colour] = {
    "laser": (80, 140, 255), "lidar": (255, 220, 0), "moge": (190, 120, 255), "gt_depth": (255, 160, 210),
    "cutr_sam3": (240, 30, 200), "cutr_clip": (150, 150, 255)}
"""dict of str to Colour: the colour of each automatic source's projected 3D boxes."""

_FALLBACK_COLOURS: tuple[Colour, ...] = ((255, 255, 140), (120, 200, 255), (255, 190, 120), (200, 255, 255))

INK: Colour = (20, 20, 20)
"""Colour: label backgrounds and outlines."""

VIEW_LONG_SIDE = 1024
"""int: long side of a full overlay picture, pixels."""

THUMB_LONG_SIDE = 256
"""int: long side of an index thumbnail, pixels."""

CROP_LONG_SIDE = 480
"""int: long side of a spot-check crop, pixels."""

FRAME_LONG_SIDE = 800
"""int: long side of a spot-check full-frame picture, pixels."""

NEAR_Z = 0.05
"""float: wireframe edges are clipped to this distance in front of the camera, metres."""

Z95 = 1.959963984540054
"""float: the standard normal quantile of a two-sided 95% interval."""


# --------------------------------------------------------------------------------------------------------
# Loading and pairing
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceView:
    """One automatic source of one view: its detections, their lifts and the recomputed pairing.

    Parameters
    ----------
    source : str
        The source's name (a depth source such as ``"lidar"``, or a box source such as ``"cutr_sam3"``).
    detections : list of Detection
        The detections of the folder the source was read from (shared by that folder's sources).
    lifts : list of Lift
        One lift per detection from this source; a lift without a box takes no part in pairing.
    result : MatchResult
        The 2D association of the boxed detections with the present objects.
    min_iou : float
        The overlap threshold used, from the source's stored settings.
    folder : str
        The label folder the source was read from.
    """

    source: str
    detections: list[Detection]
    lifts: list[Lift]
    result: MatchResult
    min_iou: float
    folder: str


@dataclass
class ViewData:
    """One view with its ground-truth objects and, per automatic source, the recomputed pairing.

    Parameters
    ----------
    key : str
        ``"<dataset>/<scene>/<camera_id>/<frame>"``.
    view : FrameView
        The view.
    visibility : str
        How the oracle's visible 2D boxes were measured (a depth source or ``"gt_box2d"``).
    oracle : list of ObjectInView
        Every ground-truth object of the view, in stored order.
    present_ids : set of str
        Ids of the objects present in the view (the ones that take part in pairing).
    sources : dict of str to SourceView
        Automatic source -> its detections and pairing; a source missing from the view is absent.
    """

    key: str
    view: FrameView
    visibility: str
    oracle: list[ObjectInView]
    present_ids: set[str]
    sources: dict[str, SourceView]

    @property
    def present(self) -> list[ObjectInView]:
        """list of ObjectInView: the present objects, in stored order."""
        return [o for o in self.oracle if o.obj.object_id in self.present_ids]

    @property
    def scene_key(self) -> tuple[str, str]:
        """tuple of str: ``(dataset, scene)``, the stratum of the spot-check sample."""
        return self.view.dataset, self.view.scene


@dataclass(frozen=True)
class Pair:
    """One detection paired with one ground-truth object by one source's 2D association.

    Parameters
    ----------
    pair_id : str
        ``"<source>:<view_key>:<object_id>:<detection>"``, unique across a viewer.
    source : str
        The automatic source.
    view_key : str
        The view's identity.
    dataset, scene : str
        The view's dataset and scene.
    object_id : str
        The ground-truth object's id.
    gt_label : str
        The ground-truth object's class.
    detection : int
        The detection's index in its folder's detection list.
    phrase : str
        The detection's prompt phrase.
    score : float
        The detection's score.
    iou2d : float
        2D IoU of the detection's box with the object's visible box.
    runner_up_iou2d : float
        Largest 2D IoU of either member with another candidate (0 if none).
    ambiguous : bool
        Whether the runner-up also cleared the threshold.
    label_correct : bool
        Whether the phrase maps to the object's class.
    distance : float
        3D centre distance between the lifted box and the object's box, metres.
    iou3d : float
        3D IoU of the two boxes' gravity-aligned prisms.
    gt_box2d : tuple of float
        The object's visible 2D box, pixel edges at integer coordinates.
    det_box2d : tuple of float
        The detection's 2D box, pixel edges at integer coordinates.
    """

    pair_id: str
    source: str
    view_key: str
    dataset: str
    scene: str
    object_id: str
    gt_label: str
    detection: int
    phrase: str
    score: float
    iou2d: float
    runner_up_iou2d: float
    ambiguous: bool
    label_correct: bool
    distance: float
    iou3d: float
    gt_box2d: Rect
    det_box2d: Rect

    def to_json(self) -> dict[str, Any]:
        """The pair as a JSON-serialisable dict.

        Returns
        -------
        dict
            Every field; boxes as lists.
        """
        record = asdict(self)
        record["gt_box2d"], record["det_box2d"] = list(self.gt_box2d), list(self.det_box2d)
        return record


def _frame_sort_key(frame: str) -> tuple[float, str]:
    """Sort key for frame identifiers that are numbers written as text.

    Parameters
    ----------
    frame : str
        Frame identifier.

    Returns
    -------
    tuple
        ``(numeric value, text)``; non-numeric frames sort first, by text.
    """
    try:
        return float(frame), frame
    except ValueError:
        return float("-inf"), frame


def _wanted(scene: str, filters: Sequence[str] | None) -> bool:
    """Whether a scene passes a ``--scenes`` filter (substrings, as in ``scripts/run_labels.py``).

    Parameters
    ----------
    scene : str
        Scene name.
    filters : sequence of str or None
        Substrings; ``None`` or empty accepts every scene.

    Returns
    -------
    bool
        True if no filter is set or the scene contains one of the substrings.
    """
    return not filters or any(f in scene for f in filters)


def load_view_records(folders: Sequence[str | Path], datasets: Sequence[str] | None = None,
                      scenes: Sequence[str] | None = None) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """Read the stored records of the selected views from one or more label folders.

    Parameters
    ----------
    folders : sequence of str or pathlib.Path
        Label folders (``<labels>/<dataset>/shards/*.json``). The first chooses the views; a later folder
        contributes the shard of the same name, if it has one.
    datasets : sequence of str, optional
        Datasets to read; defaults to every dataset folder of the first label folder that has shards.
    scenes : sequence of str, optional
        Keep scenes containing one of these substrings; ``None`` or empty keeps all.

    Returns
    -------
    dict of str to list of (str, dict)
        View key -> ``(folder, records)`` for every folder that has the view, first folder first. Views are
        ordered by dataset, scene, camera and frame.

    Raises
    ------
    ValueError
        If no folder is given, or a later folder's shard of the same name holds another view.
    """
    if not folders:
        raise ValueError("at least one label folder is needed")
    roots = [Path(f) for f in folders]
    if datasets is None:
        datasets = sorted(p.parent.name for p in roots[0].glob("*/shards") if p.is_dir())
    entries: list[tuple[tuple[Any, ...], str, list[tuple[str, dict[str, Any]]]]] = []
    for dataset in datasets:
        for path in sorted((roots[0] / dataset / "shards").glob("*.json")):
            records = json.loads(path.read_text(encoding="utf-8"))["records"]
            view = records["view"]
            if not _wanted(view["scene"], scenes):
                continue
            found = [(str(roots[0]), records)]
            for root in roots[1:]:
                other = root / dataset / "shards" / path.name
                if not other.exists():
                    continue
                other_records = json.loads(other.read_text(encoding="utf-8"))["records"]
                if other_records["view"]["view_key"] != view["view_key"]:
                    raise ValueError(f"{other} holds {other_records['view']['view_key']}, "
                                     f"not {view['view_key']}")
                found.append((str(root), other_records))
            order = (view["dataset"], view["scene"], view["camera_id"], _frame_sort_key(view["frame"]))
            entries.append((order, view["view_key"], found))
    entries.sort(key=lambda e: e[0])
    return {key: found for _, key, found in entries}


def available_sources(entries: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    """The automatic sources a view's folders hold, first folder first.

    Parameters
    ----------
    entries : sequence of (str, mapping)
        ``(folder, records)`` of one view, from :func:`load_view_records`.

    Returns
    -------
    list of str
        Source names without repeats, in folder and storage order.
    """
    seen: list[str] = []
    for _, records in entries:
        seen.extend(s for s in records["view"]["depth_sources"] if s not in seen)
    return seen


def _same_view(base: Mapping[str, Any], other: Mapping[str, Any], folder: str) -> None:
    """Check that two folders' records describe the same view and ground-truth objects.

    Parameters
    ----------
    base : mapping
        The first folder's records of the view.
    other : mapping
        Another folder's records of the view.
    folder : str
        The other folder, for the message.

    Raises
    ------
    ValueError
        If the camera, pose, visibility method or the oracle objects (ids, labels, visible boxes, fractions)
        differ.
    """
    fields = ("camera", "T_wc", "visibility")
    if any(base["view"][f] != other["view"][f] for f in fields):
        raise ValueError(f"{folder}: view {base['view']['view_key']} differs from the first folder's "
                         f"({', '.join(fields)})")

    def objects(records: Mapping[str, Any]) -> list[tuple[Any, ...]]:
        """The oracle fields that pairing reads.

        Parameters
        ----------
        records : mapping
            A view's records.

        Returns
        -------
        list of tuple
            ``(object_id, label, bbox2d, visible_fraction, truncated_fraction)`` per object.
        """
        return [(r["object_id"], r["label"], r["bbox2d"], r["visible_fraction"], r["truncated_fraction"])
                for r in records["oracle_objects"]]

    if objects(base) != objects(other):
        raise ValueError(f"{folder}: the ground-truth objects of {base['view']['view_key']} differ from the "
                         f"first folder's; the folders were not built on the same views")


def view_data(entries: Sequence[tuple[str, Mapping[str, Any]]], sources: Iterable[str]) -> ViewData:
    """Rebuild one view from its stored records and redo the 2D association for each requested source.

    Parameters
    ----------
    entries : sequence of (str, mapping)
        ``(folder, records)`` of the view, first folder first (:func:`load_view_records`).
    sources : iterable of str
        Automatic sources to pair. Each is read from the first folder that has it; a source no folder has
        is left out of the result.

    Returns
    -------
    ViewData
        The view, its objects and, per available source, the detections, lifts and recomputed pairing.

    Raises
    ------
    ValueError
        If the folders disagree about the view (:func:`_same_view`), or as
        :func:`~measured_space.pipeline.detections_from_records` and
        :func:`~measured_space.pipeline.match_view`.
    """
    _, base = entries[0]
    view = view_from_record(base["view"])
    visibility = base["view"]["visibility"]
    oracle, _ = oracle_from_records(base["oracle_objects"])
    config = PipelineConfig.from_json(base["view"]["config"])
    present_ids = visible_ids(oracle, config.qa)
    present = [o for o in oracle if o.obj.object_id in present_ids]
    rebuilt: dict[int, tuple[list[Detection], dict[str, list[Lift]]]] = {}
    out: dict[str, SourceView] = {}
    for source in sources:
        found = next(((i, f, r) for i, (f, r) in enumerate(entries) if source in r["view"]["depth_sources"]),
                     None)
        if found is None:
            continue
        i, folder, records = found
        if i > 0:
            _same_view(base, records, folder)
        if i not in rebuilt:
            rebuilt[i] = detections_from_records(records["detections"], records["view"]["depth_sources"],
                                                 int(records["view"]["n_detections"]))
        detections, lifts = rebuilt[i]
        stored = PipelineConfig.from_json(records["view"]["config"])
        source_config = replace(stored, association="2d")
        result = match_view(detections, lifts[source], present, visibility, source_config, view.up)
        out[source] = SourceView(source, detections, lifts[source], result, source_config.match_2d.min_iou,
                                 folder)
    return ViewData(base["view"]["view_key"], view, visibility, oracle, present_ids, out)


def view_pairs(data: ViewData, source: str) -> list[Pair]:
    """The pairs one source's association made in a view.

    Parameters
    ----------
    data : ViewData
        The view.
    source : str
        One of ``data.sources``.

    Returns
    -------
    list of Pair
        One per match, by detection index.
    """
    sv = data.sources[source]
    objects = {o.obj.object_id: o for o in data.oracle}
    pairs = []
    for m in sv.result.matches:
        seen = objects[m.object_id]
        det = sv.detections[m.detection]
        bbox = seen.bbox2d
        assert bbox is not None                                    # present objects always have a visible box
        x0, y0, x1, y1 = (float(v) for v in det.instance.box2d)
        pairs.append(Pair(
            pair_id=f"{source}:{data.key}:{m.object_id}:{m.detection}", source=source, view_key=data.key,
            dataset=data.view.dataset, scene=data.view.scene, object_id=m.object_id, gt_label=seen.obj.label,
            detection=m.detection, phrase=det.instance.phrase, score=float(det.instance.score),
            iou2d=float(m.iou2d if m.iou2d is not None else float("nan")),
            runner_up_iou2d=float(m.runner_up_iou2d if m.runner_up_iou2d is not None else float("nan")),
            ambiguous=bool(m.ambiguous), label_correct=bool(m.label_correct), distance=float(m.distance),
            iou3d=float(m.iou), gt_box2d=visible_box_edges(bbox, data.visibility),
            det_box2d=(x0, y0, x1, y1)))
    return pairs


# --------------------------------------------------------------------------------------------------------
# Spot-check sample and answers
# --------------------------------------------------------------------------------------------------------


def _round_robin(groups: Mapping[tuple[str, str], Sequence[Pair]], k: int, rng: random.Random) -> list[Pair]:
    """Take up to ``k`` pairs spread evenly over groups: one per group per round, in a shuffled group order.

    Parameters
    ----------
    groups : mapping of (str, str) to sequence of Pair
        Scene -> its pairs (in a fixed order).
    k : int
        Pairs wanted.
    rng : random.Random
        Source of the group order and of the order within each group.

    Returns
    -------
    list of Pair
        ``min(k, total)`` pairs; every group contributes ``floor`` or ``ceil`` of an equal share, or all its
        pairs if it has fewer.
    """
    keys = sorted(groups)
    rng.shuffle(keys)
    queues = {}
    for key in keys:
        items = list(groups[key])
        rng.shuffle(items)
        queues[key] = items
    taken: list[Pair] = []
    while len(taken) < k and any(queues.values()):
        for key in keys:
            if len(taken) == k:
                break
            if queues[key]:
                taken.append(queues[key].pop())
    return taken


def sample_pairs(pairs: Sequence[Pair], size: int, seed: int) -> list[Pair]:
    """A seeded spot-check sample: half ambiguous and half clear pairs, each half spread evenly over scenes.

    ``size // 2`` ambiguous and ``size - size // 2`` clear pairs are wanted. A half with fewer pairs available
    contributes all of them, and the sample is smaller (the other half is not topped up, so each half's
    wrong-pair rate keeps its intended precision). Within a half, scenes (``(dataset, scene)``) take turns in a
    shuffled order, one pair each, so every scene contributes an equal share or all it has. The result is in a
    shuffled order, so a reviewer cannot tell the halves apart by position. The input order does not matter:
    pairs are sorted by id before any random step.

    Parameters
    ----------
    pairs : sequence of Pair
        Candidate pairs, all of one source.
    size : int
        Pairs wanted, at least 0.
    seed : int
        Seed of every random choice.

    Returns
    -------
    list of Pair
        The sample, in presentation order.

    Raises
    ------
    ValueError
        If ``size`` is negative or pair ids repeat.
    """
    if size < 0:
        raise ValueError(f"size must be at least 0, got {size}")
    ordered = sorted(pairs, key=lambda p: p.pair_id)
    if len({p.pair_id for p in ordered}) != len(ordered):
        raise ValueError("pair ids must be unique")
    rng = random.Random(seed)
    chosen: list[Pair] = []
    for ambiguous, wanted in ((True, size // 2), (False, size - size // 2)):
        groups: dict[tuple[str, str], list[Pair]] = defaultdict(list)
        for p in ordered:
            if p.ambiguous == ambiguous:
                groups[(p.dataset, p.scene)].append(p)
        chosen.extend(_round_robin(groups, wanted, rng))
    rng.shuffle(chosen)
    return chosen


def sample_id(sample: Sequence[Pair], seed: int, source: str) -> str:
    """A short, stable identifier of a spot-check sample (it keys the answers in the browser's storage).

    Parameters
    ----------
    sample : sequence of Pair
        The sample, in presentation order.
    seed : int
        Its seed.
    source : str
        Its source description.

    Returns
    -------
    str
        The first 12 hex digits of the SHA-256 of the seed, source and pair ids.
    """
    payload = json.dumps({"seed": seed, "source": source, "pairs": [p.pair_id for p in sample]})
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval of a binomial proportion.

    Parameters
    ----------
    k : int
        Successes (here: wrong pairs), ``0 <= k <= n``.
    n : int
        Trials (judged pairs).
    z : float, default :data:`Z95`
        Standard normal quantile (1.96 for 95%).

    Returns
    -------
    tuple of float or None
        ``(low, high)`` in [0, 1], or ``None`` when ``n == 0``.

    Raises
    ------
    ValueError
        If ``k`` is outside ``[0, n]`` or ``n`` is negative.
    """
    if n < 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return None
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(answers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Wrong-pair rate of a set of answers, over the pairs judged right or wrong.

    Parameters
    ----------
    answers : sequence of mapping
        Exported answers (``verdict`` one of :data:`VERDICTS` or ``None``).

    Returns
    -------
    dict
        ``n_items``, ``n_right``, ``n_wrong``, ``n_cant_tell``, ``n_unanswered``, ``n_judged`` (right +
        wrong), ``wrong_rate`` (``None`` without judged pairs) and ``ci95`` (Wilson, ``None`` likewise).
    """
    counts = Counter(a.get("verdict") for a in answers)
    n_judged = counts["right"] + counts["wrong"]
    ci = wilson_interval(counts["wrong"], n_judged)
    return {"n_items": len(answers), "n_right": counts["right"], "n_wrong": counts["wrong"],
            "n_cant_tell": counts["cant_tell"], "n_unanswered": counts[None], "n_judged": n_judged,
            "wrong_rate": counts["wrong"] / n_judged if n_judged else None,
            "ci95": None if ci is None else [ci[0], ci[1]]}


def summarize_answers(export: Mapping[str, Any]) -> dict[str, Any]:
    """Wrong-pair rates of an exported spot-check, overall and split by ambiguous and clear pairs.

    Rates are over the pairs judged right or wrong ("can't tell" and unanswered pairs are counted, not used),
    with 95% Wilson intervals. The overall rate is the *sample's*: the sample takes half its pairs from the
    ambiguous ones, which are a minority of all pairs, so it over-represents them. When the export carries the
    population of each half (``population``, written by ``spotcheck.html``), ``weighted_wrong_rate`` reweights
    the two halves' rates by those counts: a point estimate for all pairs of the source, without an interval.

    Parameters
    ----------
    export : mapping
        The JSON ``spotcheck.html`` exports: ``sample_id``, ``seed``, ``source``, optional ``population``
        (``{"ambiguous": int, "clear": int}``) and ``answers`` (``pair_id``, ``view_key``, ``object_id``,
        ``detection``, ``ambiguous``, ``verdict``, ``note``).

    Returns
    -------
    dict
        ``sample_id``, ``seed``, ``source``, ``overall``, ``ambiguous`` and ``clear`` (each as :func:`_rate`),
        ``weighted_wrong_rate`` (float or ``None``) and ``population``.

    Raises
    ------
    ValueError
        If an answer has an unknown verdict or a pair id repeats.
    """
    answers = list(export["answers"])
    bad = sorted({str(a.get("verdict")) for a in answers if a.get("verdict") not in (*VERDICTS, None)})
    if bad:
        raise ValueError(f"unknown verdicts {bad}; expected one of {VERDICTS} or null")
    if len({a["pair_id"] for a in answers}) != len(answers):
        raise ValueError("pair ids repeat in the export")
    halves = {"ambiguous": _rate([a for a in answers if a["ambiguous"]]),
              "clear": _rate([a for a in answers if not a["ambiguous"]])}
    population = export.get("population")
    weighted = None
    if population:
        total = sum(int(population.get(h, 0)) for h in halves)
        if total and all(halves[h]["wrong_rate"] is not None for h in halves if int(population.get(h, 0))):
            weighted = sum(int(population.get(h, 0)) / total * (halves[h]["wrong_rate"] or 0.0) for h in halves)
    return {"sample_id": export.get("sample_id"), "seed": export.get("seed"), "source": export.get("source"),
            "overall": _rate(answers), **halves, "weighted_wrong_rate": weighted, "population": population}


# --------------------------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------------------------


def box_wireframe(view: FrameView, box: OrientedBox, near: float = NEAR_Z) -> list[Segment]:
    """The 12 edges of a world box projected into a view, clipped to the near plane.

    Parameters
    ----------
    view : FrameView
        The view.
    box : OrientedBox
        World-frame box.
    near : float, default :data:`NEAR_Z`
        Edges are cut where they come nearer than this to the camera plane, metres.

    Returns
    -------
    list of Segment
        Projected edges in pixels (centres at integer coordinates), not clipped to the image; edges wholly
        nearer than ``near`` are left out.
    """
    cam_box = box.transformed(view.T_cw)
    corners = cam_box.corners()
    signs = cam_box.to_local(corners) > 0                      # corner -> which side of each box axis
    segments: list[Segment] = []
    for a in range(8):
        for b in range(a + 1, 8):
            if int(np.sum(signs[a] != signs[b])) != 1:
                continue
            p, q = corners[a], corners[b]
            if p[2] < near and q[2] < near:
                continue
            if p[2] < near or q[2] < near:
                t = (near - p[2]) / (q[2] - p[2])
                cut = p + t * (q - p)
                p, q = (cut, q) if p[2] < near else (p, cut)
            uv, _ = view.camera.project(np.stack([p, q]))
            segments.append(((float(uv[0, 0]), float(uv[0, 1])), (float(uv[1, 0]), float(uv[1, 1]))))
    return segments


def clip_segment(segment: Segment, rect: Rect) -> Segment | None:
    """Clip a segment to a rectangle (Liang-Barsky).

    Parameters
    ----------
    segment : Segment
        The segment.
    rect : Rect
        ``(xmin, ymin, xmax, ymax)``.

    Returns
    -------
    Segment or None
        The part inside the rectangle, or ``None`` if none is.
    """
    (x0, y0), (x1, y1) = segment
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - rect[0]), (dx, rect[2] - x0), (-dy, y0 - rect[1]), (dy, rect[3] - y0)):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


def dashed(segment: Segment, dash: float = 8.0, gap: float = 6.0) -> list[Segment]:
    """Split a segment into dashes.

    Parameters
    ----------
    segment : Segment
        The segment.
    dash : float, default 8.0
        Dash length, pixels.
    gap : float, default 6.0
        Gap length, pixels.

    Returns
    -------
    list of Segment
        The dashes, from the first point on.
    """
    (x0, y0), (x1, y1) = segment
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return [segment]
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    out, s = [], 0.0
    while s < length:
        e = min(s + dash, length)
        out.append(((x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)))
        s += dash + gap
    return out


def short_ids(ids: Iterable[str], min_length: int = 4) -> dict[str, str]:
    """The shortest prefixes (at least ``min_length`` characters) that keep a set of ids distinct.

    Parameters
    ----------
    ids : iterable of str
        Object ids.
    min_length : int, default 4
        Shortest prefix.

    Returns
    -------
    dict of str to str
        Id -> its prefix.
    """
    unique = sorted(set(ids))
    length = min_length
    while length < max((len(i) for i in unique), default=0) and \
            len({i[:length] for i in unique}) != len(unique):
        length += 1
    return {i: i[:length] for i in unique}


def source_colour(source: str, order: Sequence[str] = ()) -> Colour:
    """The colour of a source's projected 3D boxes.

    Parameters
    ----------
    source : str
        Source name.
    order : sequence of str, optional
        The sources on show, to pick distinct fallback colours for unknown sources.

    Returns
    -------
    Colour
        :data:`SOURCE_COLOURS` entry, or a fallback colour.
    """
    if source in SOURCE_COLOURS:
        return SOURCE_COLOURS[source]
    unknown = [s for s in order if s not in SOURCE_COLOURS]
    index = unknown.index(source) if source in unknown else len(unknown)
    return _FALLBACK_COLOURS[index % len(_FALLBACK_COLOURS)]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Pillow's built-in font at a size.

    Parameters
    ----------
    size : int
        Font size, pixels.

    Returns
    -------
    PIL.ImageFont.FreeTypeFont or PIL.ImageFont.ImageFont
        The scalable built-in font (the fixed bitmap font if FreeType is unavailable).
    """
    return ImageFont.load_default(size=size)


class _Canvas:
    """Draw full-resolution image geometry onto a scaled (and possibly cropped) picture.

    Parameters
    ----------
    picture : PIL.Image.Image
        The picture to draw on (RGB).
    scale : float
        Picture pixels per full-resolution pixel.
    origin : tuple of float, default (0, 0)
        Full-resolution pixel edge at the picture's top-left corner (the crop offset).
    font_size : int, default 14
        Label font size, picture pixels.
    """

    def __init__(self, picture: Image.Image, scale: float, origin: Point = (0.0, 0.0), font_size: int = 14
                 ) -> None:
        """Wrap a picture.

        Parameters
        ----------
        picture : PIL.Image.Image
            The picture to draw on (RGB).
        scale : float
            Picture pixels per full-resolution pixel.
        origin : tuple of float, default (0, 0)
            Full-resolution pixel edge at the picture's top-left corner.
        font_size : int, default 14
            Label font size, picture pixels.
        """
        self.picture = picture
        self.draw = ImageDraw.Draw(picture)
        self.scale = scale
        self.origin = origin
        self.font = _font(font_size)
        w, h = picture.size
        self.bounds: Rect = (-50.0, -50.0, w + 50.0, h + 50.0)

    def centre(self, point: Point) -> Point:
        """Map a full-resolution point with pixel centres at integers to picture coordinates.

        Parameters
        ----------
        point : Point
            ``(u, v)``, full-resolution pixels, centres at integers.

        Returns
        -------
        Point
            Picture coordinates (centres at integers).
        """
        return ((point[0] - self.origin[0] + 0.5) * self.scale - 0.5,
                (point[1] - self.origin[1] + 0.5) * self.scale - 0.5)

    def edge(self, point: Point) -> Point:
        """Map a full-resolution point with pixel edges at integers to picture coordinates.

        Parameters
        ----------
        point : Point
            ``(x, y)``, full-resolution pixels, edges at integers.

        Returns
        -------
        Point
            Picture coordinates (centres at integers).
        """
        return ((point[0] - self.origin[0]) * self.scale - 0.5, (point[1] - self.origin[1]) * self.scale - 0.5)

    def line(self, segment: Segment, colour: Colour, width: int = 2, dash: bool = False,
             outline: bool = False) -> None:
        """Draw a segment given in picture coordinates.

        Parameters
        ----------
        segment : Segment
            Picture coordinates.
        colour : Colour
            Line colour.
        width : int, default 2
            Line width, picture pixels.
        dash : bool, default False
            Draw dashed.
        outline : bool, default False
            Draw a dark line 2 px wider underneath, for contrast on bright backgrounds.
        """
        clipped = clip_segment(segment, self.bounds)
        if clipped is None:
            return
        pieces = dashed(clipped) if dash else [clipped]
        for piece in pieces:
            if outline:
                self.draw.line(piece, fill=INK, width=width + 2)
            self.draw.line(piece, fill=colour, width=width)

    def wireframe(self, segments: Iterable[Segment], colour: Colour, width: int = 2) -> None:
        """Draw projected box edges (full-resolution, centres at integers).

        Parameters
        ----------
        segments : iterable of Segment
            From :func:`box_wireframe`.
        colour : Colour
            Line colour.
        width : int, default 2
            Line width, picture pixels.
        """
        for a, b in segments:
            self.line((self.centre(a), self.centre(b)), colour, width)

    def rect(self, box: Rect, colour: Colour, width: int = 2, dash: bool = False, outline: bool = True
             ) -> Rect:
        """Draw a full-resolution 2D box (edges at integers).

        Parameters
        ----------
        box : Rect
            ``(xmin, ymin, xmax, ymax)``, full-resolution pixels, edges at integers.
        colour : Colour
            Line colour.
        width : int, default 2
            Line width, picture pixels.
        dash : bool, default False
            Draw dashed.
        outline : bool, default True
            Draw a dark outline underneath.

        Returns
        -------
        Rect
            The box in picture coordinates.
        """
        x0, y0 = self.edge((box[0], box[1]))
        x1, y1 = self.edge((box[2], box[3]))
        for seg in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            self.line(seg, colour, width, dash, outline)
        return x0, y0, x1, y1

    def plate(self, anchor: Point, text: str, below: bool = False) -> Rect:
        """Where a text plate goes, kept inside the picture (nothing is drawn).

        Parameters
        ----------
        anchor : Point
            Top-left corner of the plate (bottom-left with ``below``), picture coordinates.
        text : str
            The text.
        below : bool, default False
            Put the plate's bottom, not its top, at the anchor.

        Returns
        -------
        Rect
            The plate, picture coordinates.
        """
        left, top, right, bottom = self.draw.textbbox((0, 0), text, font=self.font)
        w, h = right - left + 6, bottom - top + 4
        x, y = anchor[0], anchor[1] - h if below else anchor[1]
        pw, ph = self.picture.size
        x = min(max(x, 0.0), max(pw - w, 0.0))
        y = min(max(y, 0.0), max(ph - h, 0.0))
        return x, y, x + w, y + h

    def label(self, plate: Rect, text: str, colour: Colour) -> None:
        """Draw text on a dark plate placed by :meth:`plate`.

        Parameters
        ----------
        plate : Rect
            The plate, picture coordinates.
        text : str
            The text (the same as given to :meth:`plate`).
        colour : Colour
            Text colour.
        """
        left, top, _, _ = self.draw.textbbox((0, 0), text, font=self.font)
        self.draw.rectangle(plate, fill=INK)
        self.draw.text((plate[0] + 3 - left, plate[1] + 2 - top), text, fill=colour, font=self.font)


def _fit(width: int, height: int, long_side: int, max_upscale: float = 1.0) -> tuple[float, tuple[int, int]]:
    """Scale factor and size that fit an image's long side to ``long_side``.

    Parameters
    ----------
    width, height : int
        Image size, pixels.
    long_side : int
        Target long side, pixels.
    max_upscale : float, default 1.0
        Largest allowed enlargement.

    Returns
    -------
    scale : float
        Picture pixels per image pixel.
    size : tuple of int
        Picture ``(width, height)``.
    """
    scale = min(long_side / max(width, height), max_upscale)
    return scale, (max(1, round(width * scale)), max(1, round(height * scale)))


def _pairing_source(data: ViewData, sources: Sequence[str]) -> str | None:
    """The source whose detections and pairs an overlay shows: the first of ``sources`` the view has.

    Parameters
    ----------
    data : ViewData
        The view.
    sources : sequence of str
        Sources on show, in priority order.

    Returns
    -------
    str or None
        The source, or ``None`` if the view has none of them.
    """
    return next((s for s in sources if s in data.sources), None)


def draw_view(image: NDArray[np.uint8], data: ViewData, sources: Sequence[str],
              long_side: int = VIEW_LONG_SIDE) -> Image.Image:
    """Draw a view's ground truth, automatic boxes and pairing on its image.

    Drawn, bottom to top: each shown source's lifted 3D boxes in its colour (thin); each present object's
    projected 3D box (cyan) and visible 2D box (cyan, dashed); the pairing source's detection boxes (white;
    grey and dashed without a box from that source); the pair lines (detection centre to object label); the
    labels. The pairing source is the first of ``sources`` the view has.

    Parameters
    ----------
    image : numpy.ndarray, shape (H, W, 3), uint8
        The view's RGB image, matching ``data.view.camera``.
    data : ViewData
        The view.
    sources : sequence of str
        Automatic sources to show, the pairing source first.
    long_side : int, default :data:`VIEW_LONG_SIDE`
        Long side of the picture, pixels (never enlarged).

    Returns
    -------
    PIL.Image.Image
        The picture.

    Raises
    ------
    ValueError
        If the image's size differs from the view's camera.
    """
    cam = data.view.camera
    if image.shape[:2] != (cam.height, cam.width):
        raise ValueError(f"{data.key}: image is {image.shape[1]}x{image.shape[0]}, camera {cam.width}x{cam.height}")
    scale, size = _fit(cam.width, cam.height, long_side)
    picture = Image.fromarray(image).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    canvas = _Canvas(picture, scale)
    shown = [s for s in sources if s in data.sources]
    for source in reversed(shown):
        for lift in data.sources[source].lifts:
            if lift.box is not None:
                canvas.wireframe(box_wireframe(data.view, lift.box), source_colour(source, sources), 1)
    present = data.present
    sid = short_ids(o.obj.object_id for o in present)
    gt_rects: dict[str, Rect] = {}
    for seen in present:
        canvas.wireframe(box_wireframe(data.view, seen.obj.box), GT_COLOUR, 2)
        assert seen.bbox2d is not None
        gt_rects[seen.obj.object_id] = canvas.rect(visible_box_edges(seen.bbox2d, data.visibility), GT_COLOUR,
                                                   2, dash=True)
    pairing = _pairing_source(data, sources)
    det_rects: dict[int, Rect] = {}
    if pairing is not None:
        sv = data.sources[pairing]
        for det, lift in zip(sv.detections, sv.lifts):
            boxed = lift.box is not None
            det_rects[det.index] = canvas.rect(det.instance.box2d, DET_COLOUR if boxed else UNBOXED_COLOUR,
                                               2 if boxed else 1, dash=not boxed)
    gt_labels: dict[str, tuple[Rect, str]] = {}
    for seen in present:
        x0, y0, _, _ = gt_rects[seen.obj.object_id]
        text = f"{seen.obj.label} {sid[seen.obj.object_id]}"
        gt_labels[seen.obj.object_id] = canvas.plate((x0, y0), text), text
    det_labels: list[tuple[Rect, str, Colour]] = []
    if pairing is not None:
        sv = data.sources[pairing]
        for m in sv.result.matches:              # lines first, so the labels drawn next stay readable
            x0, y0, x1, y1 = det_rects[m.detection]
            centre = ((x0 + x1) / 2, (y0 + y1) / 2)
            px0, py0, px1, py1 = gt_labels[m.object_id][0]
            if m.ambiguous:
                colour, dash = PAIR_AMBIGUOUS, True
            else:
                colour, dash = (PAIR_OK if m.label_correct else PAIR_MISLABEL), False
            canvas.line((centre, ((px0 + px1) / 2, (py0 + py1) / 2)), colour, 3 if dash else 2, dash=dash,
                        outline=True)
            canvas.draw.ellipse((centre[0] - 4, centre[1] - 4, centre[0] + 4, centre[1] + 4), fill=colour,
                                outline=INK)
        for det, lift in zip(sv.detections, sv.lifts):
            x0, _, _, y1 = det_rects[det.index]
            text = f"#{det.index} {det.instance.phrase} {det.instance.score:.2f}"
            text = text if lift.box is not None else f"{text} (no box)"
            det_labels.append((canvas.plate((x0, y1), text, below=True), text,
                               DET_COLOUR if lift.box is not None else UNBOXED_COLOUR))
    for plate, text in gt_labels.values():
        canvas.label(plate, text, GT_COLOUR)
    for plate, text, colour in det_labels:
        canvas.label(plate, text, colour)
    return picture


def legend_items(sources: Sequence[str], pairing: str | None) -> list[tuple[str, Colour, str]]:
    """The legend entries of an overlay.

    Parameters
    ----------
    sources : sequence of str
        Sources on show.
    pairing : str or None
        The pairing source.

    Returns
    -------
    list of (str, Colour, str)
        ``(text, colour, glyph)`` with glyph ``"line"``, ``"dash"`` or ``"box"``.
    """
    items = [("GT 3D box", GT_COLOUR, "line"), ("GT visible box", GT_COLOUR, "dash")]
    if pairing is not None:
        items += [(f"detection (#index phrase score), {pairing}", DET_COLOUR, "box"),
                  (f"detection without a {pairing} box", UNBOXED_COLOUR, "dash")]
    items += [(f"{s} 3D box", source_colour(s, sources), "line") for s in sources]
    if pairing is not None:
        items += [("pair, label matches", PAIR_OK, "line"), ("pair, label differs", PAIR_MISLABEL, "line"),
                  ("ambiguous pair", PAIR_AMBIGUOUS, "dash")]
    return items


def frame_picture(picture: Image.Image, title: str, items: Sequence[tuple[str, Colour, str]],
                  font_size: int = 14) -> Image.Image:
    """Add a title strip above a picture and a legend strip below it.

    Parameters
    ----------
    picture : PIL.Image.Image
        The drawn view.
    title : str
        One line of text for the top strip.
    items : sequence of (str, Colour, str)
        Legend entries (:func:`legend_items`).
    font_size : int, default 14
        Text size, pixels.

    Returns
    -------
    PIL.Image.Image
        The framed picture, as wide as ``picture``.
    """
    font = _font(font_size)
    w = picture.width
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    line_h = font_size + 8
    # Flow the legend entries into rows that fit the width.
    rows: list[list[tuple[str, Colour, str, float]]] = [[]]
    x = 8.0
    for text, colour, glyph in items:
        tw = probe.textlength(text, font=font)
        need = 28 + tw + 16
        if rows[-1] and x + need > w:
            rows.append([])
            x = 8.0
        rows[-1].append((text, colour, glyph, x))
        x += need
    top_h, bottom_h = line_h + 4, line_h * len(rows) + 8
    out = Image.new("RGB", (w, top_h + picture.height + bottom_h), INK)
    out.paste(picture, (0, top_h))
    draw = ImageDraw.Draw(out)
    draw.text((8, 4), title, fill=(235, 235, 235), font=font)
    for r, row in enumerate(rows):
        y = top_h + picture.height + 4 + r * line_h + line_h / 2
        for text, colour, glyph, x0 in row:
            if glyph == "box":
                draw.rectangle((x0, y - 6, x0 + 22, y + 6), outline=colour, width=2)
            elif glyph == "dash":
                for a, b in dashed(((x0, y), (x0 + 22, y)), dash=6.0, gap=4.0):
                    draw.line((a, b), fill=colour, width=3)
            else:
                draw.line(((x0, y), (x0 + 22, y)), fill=colour, width=3)
            draw.text((x0 + 28, y - font_size / 2 - 2), text, fill=(235, 235, 235), font=font)
    return out


def crop_window(boxes: Sequence[Rect], width: int, height: int, margin: float = 0.2, min_margin: float = 24.0
                ) -> tuple[int, int, int, int]:
    """A crop around the union of boxes, with a margin, inside the image.

    Parameters
    ----------
    boxes : sequence of Rect
        Boxes to include, pixel edges at integers.
    width, height : int
        Image size, pixels.
    margin : float, default 0.2
        Margin on each side as a fraction of the union's longer side.
    min_margin : float, default 24.0
        Smallest margin, pixels.

    Returns
    -------
    tuple of int
        ``(xmin, ymin, xmax, ymax)`` pixel edges, ``0 <= xmin < xmax <= width`` and likewise for y.
    """
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    pad = max(min_margin, margin * max(x1 - x0, y1 - y0))
    cx0 = int(min(max(math.floor(x0 - pad), 0), width - 1))
    cy0 = int(min(max(math.floor(y0 - pad), 0), height - 1))
    cx1 = int(max(min(math.ceil(x1 + pad), width), cx0 + 1))
    cy1 = int(max(min(math.ceil(y1 + pad), height), cy0 + 1))
    return cx0, cy0, cx1, cy1


def draw_pair(image: NDArray[np.uint8], data: ViewData, pair: Pair, crop_long_side: int = CROP_LONG_SIDE,
              frame_long_side: int = FRAME_LONG_SIDE) -> tuple[Image.Image, Image.Image]:
    """The two pictures of a spot-check card: a crop around the pair and the full frame.

    Both show the ground-truth object's projected 3D box (cyan, thin), its visible 2D box (cyan, dashed) and the
    detection's 2D box (white). The full frame also outlines the crop (grey). Nothing else is drawn, so the
    card names one object and one detection.

    Parameters
    ----------
    image : numpy.ndarray, shape (H, W, 3), uint8
        The view's RGB image.
    data : ViewData
        The view.
    pair : Pair
        A pair of the view.
    crop_long_side : int, default :data:`CROP_LONG_SIDE`
        Long side of the crop picture (small crops are enlarged up to 3 times).
    frame_long_side : int, default :data:`FRAME_LONG_SIDE`
        Long side of the frame picture.

    Returns
    -------
    crop : PIL.Image.Image
        The crop picture.
    frame : PIL.Image.Image
        The full-frame picture.
    """
    h, w = image.shape[:2]
    obj = next(o.obj for o in data.oracle if o.obj.object_id == pair.object_id)
    wire = box_wireframe(data.view, obj.box)
    window = crop_window([pair.gt_box2d, pair.det_box2d], w, h)
    x0, y0, x1, y1 = window
    scale, size = _fit(x1 - x0, y1 - y0, crop_long_side, max_upscale=3.0)
    crop = Image.fromarray(image[y0:y1, x0:x1]).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    fscale, fsize = _fit(w, h, frame_long_side)
    frame = Image.fromarray(image).convert("RGB").resize(fsize, Image.Resampling.LANCZOS)
    for picture, s, origin in ((crop, scale, (float(x0), float(y0))), (frame, fscale, (0.0, 0.0))):
        canvas = _Canvas(picture, s, origin)
        canvas.wireframe(wire, GT_COLOUR, 1)
        canvas.rect(pair.gt_box2d, GT_COLOUR, 2, dash=True)
        canvas.rect(pair.det_box2d, DET_COLOUR, 2)
        if picture is frame:
            canvas.rect((float(x0), float(y0), float(x1), float(y1)), UNBOXED_COLOUR, 1, outline=False)
    return crop, frame


# --------------------------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------------------------


_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track",
                   "wbr"})


class _TagChecker(HTMLParser):
    """Collect unbalanced tags of an HTML document (every non-void element must be closed explicitly).

    Attributes
    ----------
    problems : list of str
        One message per problem.
    """

    def __init__(self) -> None:
        """Start with an empty stack."""
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, tuple[int, int]]] = []
        self.problems: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Push a non-void element.

        Parameters
        ----------
        tag : str
            Tag name.
        attrs : list of (str, str or None)
            Attributes (unused).
        """
        if tag not in _VOID:
            self.stack.append((tag, self.getpos()))

    def handle_endtag(self, tag: str) -> None:
        """Pop the matching element, recording a mismatch.

        Parameters
        ----------
        tag : str
            Tag name.
        """
        line = self.getpos()[0]
        if tag in _VOID:
            self.problems.append(f"line {line}: end tag for void element </{tag}>")
        elif not self.stack:
            self.problems.append(f"line {line}: </{tag}> with nothing open")
        elif self.stack[-1][0] != tag:
            self.problems.append(f"line {line}: </{tag}> closes <{self.stack[-1][0]}> "
                                 f"(opened on line {self.stack[-1][1][0]})")
            if any(t == tag for t, _ in self.stack):
                while self.stack and self.stack[-1][0] != tag:
                    self.stack.pop()
                self.stack.pop()
        else:
            self.stack.pop()


def html_problems(text: str) -> list[str]:
    """Check that an HTML document parses with every element closed in order.

    Parameters
    ----------
    text : str
        The document.

    Returns
    -------
    list of str
        Problems found by :class:`html.parser.HTMLParser`; empty if the document is well formed, and if it
        starts with a doctype.
    """
    checker = _TagChecker()
    checker.feed(text)
    checker.close()
    problems = list(checker.problems)
    problems += [f"<{tag}> opened on line {pos[0]} is never closed" for tag, pos in checker.stack]
    if not text.lstrip().lower().startswith("<!doctype html>"):
        problems.append("no <!doctype html>")
    return problems


def _esc(value: Any) -> str:
    """HTML-escape a value's text.

    Parameters
    ----------
    value : Any
        Value to show.

    Returns
    -------
    str
        Escaped text (quotes included).
    """
    return html.escape(str(value), quote=True)


def _json_script(element_id: str, payload: Any) -> str:
    """Embed JSON in a ``<script type="application/json">`` element, safe from ``</script>`` in strings.

    Parameters
    ----------
    element_id : str
        The element's id.
    payload : Any
        JSON-serialisable data.

    Returns
    -------
    str
        The element.
    """
    text = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e")
    return f'<script id="{_esc(element_id)}" type="application/json">{text}</script>'


_CSS = """
:root{--ink:#1d1f23;--muted:#5d6470;--line:#d9dde3;--bg:#f6f7f9;--card:#fff;--amb:#d9362b;--mis:#c86200;
--ok:#1e8e3e;--gt:#007c8c}
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--bg)}
header{padding:16px 20px;background:var(--card);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:2}
h1{font-size:20px;margin:0 0 6px}
h2{font-size:16px;margin:24px 0 8px}
main{padding:8px 20px 40px;max-width:1500px}
.meta{color:var(--muted);margin:2px 0}
.filters label{margin-right:18px;white-space:nowrap}
code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}
.view{display:flex;gap:14px;align-items:flex-start;background:var(--card);border:1px solid var(--line);
border-radius:6px;padding:10px;margin:8px 0}
.view img{display:block;width:180px;height:auto;border-radius:3px;background:#222}
.view .info{flex:1;min-width:0;overflow-x:auto}
.key{font-weight:600}
.counts{color:var(--muted);margin:2px 0 6px}
table{border-collapse:collapse;font-size:12.5px}
th,td{border:1px solid var(--line);padding:2px 6px;text-align:left;white-space:nowrap}
th{background:#eef0f3;font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.amb{color:var(--amb);font-weight:700}
.bad{color:var(--mis);font-weight:700}
.good{color:var(--ok)}
.none{color:var(--muted)}
.tag{display:inline-block;border-radius:3px;padding:0 5px;font-size:12px;margin-left:6px}
.tag.amb{background:#fde4e2}
.tag.bad{background:#ffeedd}
details{margin-top:6px}
"""

_INDEX_JS = """
(function(){
  var amb=document.getElementById('f-amb'), mis=document.getElementById('f-mis');
  var count=document.getElementById('shown-count');
  function apply(){
    var total=0;
    document.querySelectorAll('.view').forEach(function(v){
      var show=(!amb.checked||v.dataset.amb==='1')&&(!mis.checked||v.dataset.mis==='1');
      v.hidden=!show; if(show){total++;}
    });
    document.querySelectorAll('.scene').forEach(function(s){
      var n=s.querySelectorAll('.view:not([hidden])').length;
      s.hidden=(n===0); s.querySelector('.shown').textContent=n;
    });
    count.textContent=total;
  }
  amb.addEventListener('change',apply); mis.addEventListener('change',apply); apply();
})();
"""


def _fmt(value: float | None, digits: int = 2) -> str:
    """Format a number for a table cell.

    Parameters
    ----------
    value : float or None
        The number.
    digits : int, default 2
        Decimals.

    Returns
    -------
    str
        The number, or an en dash for ``None`` or NaN.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:.{digits}f}"


def view_table(data: ViewData, sources: Sequence[str]) -> str:
    """The HTML table of one view: present objects and, per source, the paired detection and its checks.

    Parameters
    ----------
    data : ViewData
        The view.
    sources : sequence of str
        Sources on show.

    Returns
    -------
    str
        A ``<table>`` (one row per present object, then a row of unpaired detections), or a short note if no
        object is present.
    """
    present = data.present
    shown = [s for s in sources if s in data.sources]
    if not present and not shown:
        return '<p class="none">No ground-truth object present and no source.</p>'
    sid = short_ids(o.obj.object_id for o in present)
    head1 = '<tr><th colspan="3">ground truth</th>' + "".join(
        f'<th colspan="5">{_esc(s)}</th>' for s in shown) + "</tr>"
    head2 = "<tr><th>id</th><th>label</th><th>vis</th>" + "".join(
        "<th>detection</th><th>IoU2d</th><th>amb</th><th>dist m</th><th>label</th>" for _ in shown) + "</tr>"
    by_object = {s: data.sources[s].result.by_object() for s in shown}
    rows = []
    for seen in present:
        oid = seen.obj.object_id
        cells = [f"<td><code>{_esc(sid[oid])}</code></td><td>{_esc(seen.obj.label)}</td>"
                 f'<td class="num">{seen.visible_fraction:.2f}</td>']
        for s in shown:
            m = by_object[s].get(oid)
            if m is None:
                cells.append('<td class="none">missed</td>' + '<td class="none">–</td>' * 4)
                continue
            det = data.sources[s].detections[m.detection]
            runner = _fmt(m.runner_up_iou2d)
            amb = (f'<td class="amb" title="runner-up IoU2d {runner}">yes</td>' if m.ambiguous
                   else f'<td class="none" title="runner-up IoU2d {runner}">no</td>')
            label = '<td class="good">ok</td>' if m.label_correct else '<td class="bad">differs</td>'
            cells.append(f"<td>#{m.detection} {_esc(det.instance.phrase)} ({det.instance.score:.2f})</td>"
                         f'<td class="num">{_fmt(m.iou2d)}</td>{amb}<td class="num">{_fmt(m.distance)}</td>'
                         f"{label}")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    extra = []
    for s in shown:
        sv = data.sources[s]
        paired = {m.detection for m in sv.result.matches}
        free = [f"#{d.index} {_esc(d.instance.phrase)}" + ("" if sv.lifts[d.index].box is not None else " (no box)")
                for d in sv.detections if d.index not in paired]
        extra.append(f'<td colspan="5">{", ".join(free) if free else "none"}</td>')
    if shown:
        rows.append('<tr><td colspan="3" class="none">unpaired detections</td>' + "".join(extra) + "</tr>")
    return f"<table><thead>{head1}{head2}</thead><tbody>{''.join(rows)}</tbody></table>"


def view_flags(data: ViewData, sources: Sequence[str]) -> tuple[bool, bool]:
    """Whether a view has an ambiguous pair, and whether it has a label mismatch, in any shown source.

    Parameters
    ----------
    data : ViewData
        The view.
    sources : sequence of str
        Sources on show.

    Returns
    -------
    ambiguous : bool
        Some shown source has an ambiguous pair.
    mismatch : bool
        Some shown source has a pair whose phrase does not map to the object's class.
    """
    matches = [m for s in sources if s in data.sources for m in data.sources[s].result.matches]
    return any(m.ambiguous for m in matches), any(not m.label_correct for m in matches)


def index_html(name: str, meta_lines: Sequence[str], scenes: Sequence[tuple[str, Sequence[str]]]) -> str:
    """The index page.

    Parameters
    ----------
    name : str
        The viewer's name.
    meta_lines : sequence of str
        Lines of run information, already HTML (escaped by the caller).
    scenes : sequence of (str, sequence of str)
        Per scene: its title and the HTML of its view cards.

    Returns
    -------
    str
        The document.
    """
    n_views = sum(len(cards) for _, cards in scenes)
    sections = []
    for title, cards in scenes:
        sections.append(f'<section class="scene"><h2>{_esc(title)} <small class="none">(<span class="shown">'
                        f"{len(cards)}</span> of {len(cards)} views)</small></h2>{''.join(cards)}</section>")
    meta = "".join(f'<p class="meta">{line}</p>' for line in meta_lines)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pairing viewer: {_esc(name)}</title>
<style>{_CSS}</style>
</head>
<body>
<header>
<h1>Pairing viewer: {_esc(name)}</h1>
{meta}
<p class="filters"><label><input type="checkbox" id="f-amb"> only views with an ambiguous pair</label>
<label><input type="checkbox" id="f-mis"> only views with a label mismatch</label>
<span class="none">showing <span id="shown-count">{n_views}</span> of {n_views} views</span>
· <a href="spotcheck.html">spot-check page</a></p>
</header>
<main>
<details><summary>How to read the pictures and tables</summary>
<p>Cyan: ground-truth object present in the view (solid: its 3D box projected; dashed: its visible 2D box),
labelled with its class and a short id. White: a detection of the pairing source (the first source listed),
with its index, phrase and score; grey dashed: a detection without a box from that source, which takes no part
in its pairing. Coloured thin boxes: each source's lifted 3D boxes. A line joins a paired detection to its
object's label: green when the phrase maps to the object's class, orange when it does not, red and dashed when
the pair is ambiguous (another object or detection also cleared the 2D IoU threshold).</p>
<p>Tables: one row per present object; per source the paired detection, the 2D IoU with the object's visible
box, whether the pair is ambiguous (hover for the runner-up IoU), the 3D centre distance in metres between the
source's box and the ground-truth box, and the label check. The filters use every source shown.</p>
</details>
{''.join(sections)}
</main>
<script>{_INDEX_JS}</script>
</body>
</html>
"""


def view_card(data: ViewData, sources: Sequence[str], image_href: str, thumb_href: str) -> str:
    """The index card of one view: thumbnail, counts and table.

    Parameters
    ----------
    data : ViewData
        The view.
    sources : sequence of str
        Sources on show.
    image_href : str
        Relative link to the full overlay.
    thumb_href : str
        Relative link to the thumbnail.

    Returns
    -------
    str
        A ``<div class="view">`` element with ``data-amb`` and ``data-mis`` flags.
    """
    amb, mis = view_flags(data, sources)
    counts = [f"{len(data.present)} present GT objects"]
    for s in sources:
        if s not in data.sources:
            counts.append(f"{_esc(s)}: not in the label folders")
            continue
        sv = data.sources[s]
        n_boxed = sum(lift.box is not None for lift in sv.lifts)
        n_amb = sum(m.ambiguous for m in sv.result.matches)
        n_mis = sum(not m.label_correct for m in sv.result.matches)
        counts.append(f"{_esc(s)}: {len(sv.detections)} detections ({n_boxed} boxed), {len(sv.result.matches)} "
                      f"pairs, {n_amb} ambiguous, {n_mis} label mismatches (IoU ≥ {sv.min_iou:g})")
    tags = ('<span class="tag amb">ambiguous</span>' if amb else "") + \
           ('<span class="tag bad">label mismatch</span>' if mis else "")
    v = data.view
    sep = " · "
    return (f'<div class="view" id="{_esc(data.key)}" data-amb="{int(amb)}" data-mis="{int(mis)}">'
            f'<a href="{_esc(image_href)}"><img src="{_esc(thumb_href)}" loading="lazy" '
            f'alt="{_esc(data.key)}"></a><div class="info"><div class="key">{_esc(v.camera_id)} / frame '
            f'{_esc(v.frame)}{tags}</div><div class="counts">{sep.join(counts)}</div>'
            f"{view_table(data, sources)}</div></div>")


_SPOT_CSS = _CSS + """
.question{font-size:17px;font-weight:600;margin:4px 0}
.guide{margin:2px 0;color:var(--muted);max-width:1100px}
.bar{display:flex;gap:16px;align-items:center;margin-top:8px;flex-wrap:wrap}
.progress{font-weight:600}
.warn{color:var(--amb)}
.card{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:12px;margin:12px 0;
display:flex;gap:16px;flex-wrap:wrap}
.card.done{border-color:#9bc8a7}
.card .pics{display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap}
.card .crop{max-width:480px;height:auto;border-radius:3px}
.card .frame{width:240px;height:auto;border-radius:3px}
.card .side{flex:1;min-width:260px}
.card .what{margin:4px 0}
.gt{color:var(--gt);font-weight:600}
.det{font-weight:600}
.verdicts button{font:inherit;padding:6px 12px;margin:6px 6px 6px 0;border:1px solid var(--line);
border-radius:4px;background:#fff;cursor:pointer}
.verdicts button.sel[data-verdict=right]{background:#d7f0dc;border-color:var(--ok)}
.verdicts button.sel[data-verdict=wrong]{background:#fde4e2;border-color:var(--amb)}
.verdicts button.sel[data-verdict=cant_tell]{background:#eceef1;border-color:var(--muted)}
.note{width:100%;max-width:520px;font:inherit;padding:4px 6px}
.small{font-size:12px;color:var(--muted)}
button.primary{font:inherit;padding:6px 14px;border-radius:4px;border:1px solid #2b5fb4;background:#2b6fd6;
color:#fff;cursor:pointer}
"""

_SPOT_JS = """
(function(){
  var META=JSON.parse(document.getElementById('spot-meta').textContent);
  var KEY='spotcheck:'+META.sample_id;
  var state={};
  var warn=document.getElementById('storage-warn');
  function warnStorage(){warn.textContent='Answers cannot be saved in this browser (storage unavailable): '+
    'export before closing the page.';}
  try{var raw=window.localStorage.getItem(KEY); if(raw){state=JSON.parse(raw)||{};}}catch(e){state={};warnStorage();}
  function save(){try{window.localStorage.setItem(KEY,JSON.stringify(state));}catch(e){warnStorage();}}
  var progress=document.getElementById('progress');
  function render(){
    var done=0;
    document.querySelectorAll('.card').forEach(function(card){
      var a=state[card.dataset.pair]||{};
      card.querySelectorAll('button[data-verdict]').forEach(function(b){
        b.classList.toggle('sel',a.verdict===b.dataset.verdict);
      });
      var note=card.querySelector('.note'); if(document.activeElement!==note){note.value=a.note||'';}
      card.classList.toggle('done',!!a.verdict); if(a.verdict){done++;}
    });
    progress.textContent=done+' / '+META.items.length+' answered';
  }
  document.querySelectorAll('.card').forEach(function(card){
    var id=card.dataset.pair;
    card.querySelectorAll('button[data-verdict]').forEach(function(b){
      b.addEventListener('click',function(){
        var a=state[id]||{};
        a.verdict=(a.verdict===b.dataset.verdict)?null:b.dataset.verdict;
        state[id]=a; save(); render();
      });
    });
    card.querySelector('.note').addEventListener('input',function(ev){
      var a=state[id]||{}; a.note=ev.target.value; state[id]=a; save();
    });
  });
  document.getElementById('export').addEventListener('click',function(){
    var answers=META.items.map(function(it){
      var a=state[it.pair_id]||{};
      return {pair_id:it.pair_id,view_key:it.view_key,object_id:it.object_id,detection:it.detection,
              ambiguous:it.ambiguous,verdict:a.verdict||null,note:a.note||''};
    });
    var out={sample_id:META.sample_id,seed:META.seed,source:META.source,population:META.population,
             answers:answers};
    var blob=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});
    var link=document.createElement('a');
    link.href=URL.createObjectURL(blob); link.download='spotcheck_'+META.sample_id+'.json';
    document.body.appendChild(link); link.click();
    setTimeout(function(){URL.revokeObjectURL(link.href); link.remove();},1000);
  });
  render();
})();
"""


def spot_card(number: int, pair: Pair, crop_href: str, frame_href: str, gt_short: str) -> str:
    """One spot-check card. The ambiguity flag and the label check are not shown.

    Parameters
    ----------
    number : int
        Position in the sample, from 1.
    pair : Pair
        The pair.
    crop_href : str
        Relative link to the crop picture.
    frame_href : str
        Relative link to the full-frame picture.
    gt_short : str
        Short id of the ground-truth object.

    Returns
    -------
    str
        A ``<div class="card">`` element.
    """
    verdicts = (("right", "Right pair"), ("wrong", "Wrong pair"), ("cant_tell", "Can't tell"))
    buttons = "".join(f'<button type="button" data-verdict="{v}">{t}</button>' for v, t in verdicts)
    return (f'<div class="card" data-pair="{_esc(pair.pair_id)}"><div class="pics">'
            f'<img class="crop" src="{_esc(crop_href)}" alt="crop {number}">'
            f'<a href="{_esc(frame_href)}"><img class="frame" src="{_esc(frame_href)}" loading="lazy" '
            f'alt="frame {number}"></a></div><div class="side"><div class="small">#{number} · '
            f"<code>{_esc(pair.view_key)}</code></div>"
            f'<p class="what">Ground truth (cyan): <span class="gt">{_esc(pair.gt_label)}</span> '
            f"<code>{_esc(gt_short)}</code></p>"
            f'<p class="what">Detection (white): <span class="det">“{_esc(pair.phrase)}”</span> '
            f'<span class="small">score {pair.score:.2f}</span></p>'
            f'<div class="verdicts">{buttons}</div>'
            f'<input class="note" type="text" placeholder="optional note"></div></div>')


def spotcheck_html(name: str, meta: Mapping[str, Any], cards: Sequence[str], meta_lines: Sequence[str]) -> str:
    """The spot-check page.

    Parameters
    ----------
    name : str
        The viewer's name.
    meta : mapping
        Embedded for the page's script: ``sample_id``, ``seed``, ``source``, ``population`` and ``items``
        (``pair_id``, ``view_key``, ``object_id``, ``detection``, ``ambiguous`` per card, in card order).
    cards : sequence of str
        The cards' HTML (:func:`spot_card`).
    meta_lines : sequence of str
        Lines of run information, already HTML.

    Returns
    -------
    str
        The document.
    """
    guide = "".join(f'<p class="guide">{_esc(line)}</p>' for line in GUIDANCE)
    info = "".join(f'<p class="meta">{line}</p>' for line in meta_lines)
    body = "".join(cards) if cards else '<p class="none">No pairs to check.</p>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pairing spot-check: {_esc(name)}</title>
<style>{_SPOT_CSS}</style>
</head>
<body>
<header>
<p class="question">{_esc(QUESTION)}</p>
{guide}
<div class="bar"><span class="progress" id="progress">0 / {len(cards)} answered</span>
<button type="button" class="primary" id="export">Export JSON</button>
<span class="small">Answers are saved in this browser as you go (sample <code>{_esc(meta['sample_id'])}</code>).
</span><span class="warn" id="storage-warn"></span></div>
</header>
<main>
{info}
{body}
</main>
{_json_script('spot-meta', meta)}
<script>{_SPOT_JS}</script>
</body>
</html>
"""


# --------------------------------------------------------------------------------------------------------
# Building a viewer
# --------------------------------------------------------------------------------------------------------


def safe_name(key: str) -> str:
    """A file name for a view key.

    Parameters
    ----------
    key : str
        A view key (``"<dataset>/<scene>/<camera_id>/<frame>"``).

    Returns
    -------
    str
        ``/`` replaced by ``__`` and any character outside ``[A-Za-z0-9._-]`` by ``_``.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", key.replace("/", "__"))


def evenly_spaced(n: int, k: int | None) -> list[int]:
    """Indices of at most ``k`` items spread evenly over ``n``, first and last included.

    Parameters
    ----------
    n : int
        Number of items.
    k : int or None
        Items wanted; ``None`` keeps all.

    Returns
    -------
    list of int
        Ascending indices without repeats.
    """
    if k is None or k >= n:
        return list(range(n))
    if k <= 0:
        return []
    if k == 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def check_out_dir(out_dir: Path) -> None:
    """Refuse an output folder that exists and is not empty.

    Parameters
    ----------
    out_dir : pathlib.Path
        The output folder.

    Raises
    ------
    FileExistsError
        If it exists and holds anything, or is a file.
    """
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"{out_dir} exists and is not empty; choose another name")


def _sources_for(dataset: str, requested: Sequence[str] | None, available: Sequence[str]) -> list[str]:
    """The sources to show for one dataset.

    Parameters
    ----------
    dataset : str
        Dataset name.
    requested : sequence of str or None
        Sources asked for; ``None`` means the dataset's deployment source (:data:`DEPLOYMENT_SOURCES`).
    available : sequence of str
        Sources the dataset's views hold.

    Returns
    -------
    list of str
        The requested sources the dataset has, in the requested order.

    Raises
    ------
    ValueError
        If none of them is available.
    """
    wanted = list(requested) if requested else [DEPLOYMENT_SOURCES.get(dataset, "")]
    shown = [s for s in wanted if s in available]
    if not shown:
        raise ValueError(f"{dataset}: none of the sources {wanted} is in the label folders "
                         f"(available: {list(available)})")
    return shown


def build_viewer(folders: Sequence[str | Path], out_dir: str | Path, *, name: str | None = None,
                 datasets: Sequence[str] | None = None, scenes: Sequence[str] | None = None,
                 sources: Sequence[str] | None = None, spot_source: str | None = None, sample_size: int = 100,
                 seed: int = 0, max_views: int | None = None, load_image: ImageLoader | None = None,
                 log: Callable[[str], None] = print) -> dict[str, Any]:
    """Build the viewer: overlays, ``index.html``, the spot-check page and ``manifest.json``.

    Every input is read and every pairing and the sample are computed before anything is written. Images that
    fail to load are replaced by a grey picture naming the error, and listed in the manifest.

    Parameters
    ----------
    folders : sequence of str or pathlib.Path
        Label folders (:func:`load_view_records`); the first chooses the views.
    out_dir : str or pathlib.Path
        Output folder; must not exist or be empty.
    name : str, optional
        Title of the pages; defaults to the folder's name.
    datasets : sequence of str, optional
        Datasets to show; defaults to all of the first folder's.
    scenes : sequence of str, optional
        Scene substrings to keep; all by default.
    sources : sequence of str, optional
        Automatic sources to show, the pairing source of the overlays first; per dataset, the ones it has are
        used. Defaults to each dataset's :data:`DEPLOYMENT_SOURCES` entry.
    spot_source : str, optional
        The spot-check's source; defaults to each dataset's :data:`DEPLOYMENT_SOURCES` entry.
    sample_size : int, default 100
        Spot-check pairs wanted (:func:`sample_pairs`).
    seed : int, default 0
        Seed of the sample.
    max_views : int, optional
        At most this many views per scene get an overlay and a card in ``index.html``, evenly spaced in frame
        order; the spot-check samples from every selected view regardless.
    load_image : callable, optional
        ``load_image(view) -> RGB``; defaults to :func:`measured_space.datasets.load_rgb`.
    log : callable, default print
        Progress messages.

    Returns
    -------
    dict
        The manifest written to ``manifest.json``.

    Raises
    ------
    FileExistsError
        If ``out_dir`` exists and is not empty.
    ValueError
        If no view is selected, a source is unavailable, or the folders disagree.
    """
    out = Path(out_dir)
    check_out_dir(out)
    loader = load_image or load_rgb
    name = name or out.name
    records = load_view_records(folders, datasets, scenes)
    if not records:
        raise ValueError(f"no views selected from {[str(f) for f in folders]} (datasets {datasets}, scenes {scenes})")

    # Sources per dataset, then every view rebuilt and paired (no writes yet).
    available: dict[str, list[str]] = defaultdict(list)
    for entries in records.values():
        dataset = entries[0][1]["view"]["dataset"]
        available[dataset].extend(s for s in available_sources(entries) if s not in available[dataset])
    shown = {d: _sources_for(d, sources, available[d]) for d in available}
    spot = {d: _sources_for(d, [spot_source] if spot_source else None, available[d])[0] for d in available}
    views: list[ViewData] = []
    for i, entries in enumerate(records.values()):
        dataset = entries[0][1]["view"]["dataset"]
        views.append(view_data(entries, dict.fromkeys([*shown[dataset], spot[dataset]])))
        if (i + 1) % 500 == 0:
            log(f"paired {i + 1} / {len(records)} views")
    candidates = [p for v in views if spot[v.view.dataset] in v.sources for p in view_pairs(v, spot[v.view.dataset])]
    spot_label = (next(iter(spot.values())) if len(set(spot.values())) == 1
                  else ",".join(f"{d}={s}" for d, s in sorted(spot.items())))
    sample = sample_pairs(candidates, sample_size, seed)
    sid = sample_id(sample, seed, spot_label)
    population = {"ambiguous": sum(p.ambiguous for p in candidates),
                  "clear": sum(not p.ambiguous for p in candidates)}

    by_scene: dict[tuple[str, str], list[ViewData]] = defaultdict(list)
    for v in views:
        by_scene[v.scene_key].append(v)
    drawn = {group[i].key for group in by_scene.values() for i in evenly_spaced(len(group), max_views)}

    check_out_dir(out)                                          # again: nothing may have appeared meanwhile
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("images/views", "images/thumbs", "images/spotcheck"):
        (out / sub).mkdir(parents=True)

    errors: dict[str, str] = {}
    cache: dict[str, NDArray[np.uint8]] = {}

    def image_of(data: ViewData) -> NDArray[np.uint8] | None:
        """Load a view's image once, recording failures.

        Parameters
        ----------
        data : ViewData
            The view.

        Returns
        -------
        numpy.ndarray or None
            The image, or ``None`` if loading failed.
        """
        if data.key in errors:
            return None
        if data.key not in cache:
            try:
                cache.clear()                                   # keep one decoded image in memory
                cache[data.key] = np.asarray(loader(data.view), dtype=np.uint8)
            except Exception as error:              # noqa: BLE001 - one unreadable image must not stop the build
                errors[data.key] = f"{type(error).__name__}: {error}"
                log(f"image failed for {data.key}: {errors[data.key]}")
                return None
        return cache[data.key]

    def placeholder(text: str, size: tuple[int, int]) -> Image.Image:
        """A grey picture naming why the image is missing.

        Parameters
        ----------
        text : str
            The error.
        size : tuple of int
            ``(width, height)``.

        Returns
        -------
        PIL.Image.Image
            The picture.
        """
        picture = Image.new("RGB", size, (90, 90, 90))
        ImageDraw.Draw(picture).text((10, 10), f"image unavailable:\n{text[:300]}", fill=(255, 255, 255),
                                     font=_font(14))
        return picture

    # Overlays and the index.
    manifest_views = []
    scene_sections: list[tuple[str, list[str]]] = []
    n_done = 0
    for (dataset, scene), group in by_scene.items():
        cards = []
        for data in group:
            if data.key not in drawn:
                continue
            stem = safe_name(data.key)
            image_rel, thumb_rel = f"images/views/{stem}.jpg", f"images/thumbs/{stem}.jpg"
            src = shown[dataset]
            pairing = _pairing_source(data, src)
            image = image_of(data)
            if image is None:
                cam = data.view.camera
                picture = placeholder(errors[data.key], _fit(cam.width, cam.height, VIEW_LONG_SIDE)[1])
            else:
                picture = draw_view(image, data, src)
            pair_note = (f"pairs: {pairing} (2D IoU >= {data.sources[pairing].min_iou:g})" if pairing
                         else "no pairing source")
            frame_picture(picture, f"{data.key} · {pair_note}", legend_items(
                [s for s in src if s in data.sources], pairing)).save(out / image_rel, quality=85)
            thumb = picture.copy()
            thumb.thumbnail((THUMB_LONG_SIDE, THUMB_LONG_SIDE), Image.Resampling.LANCZOS)
            thumb.save(out / thumb_rel, quality=80)
            cards.append(view_card(data, src, image_rel, thumb_rel))
            amb, mis = view_flags(data, src)
            manifest_views.append({"view_key": data.key, "image": image_rel, "thumb": thumb_rel,
                                   "pairing_source": pairing, "sources": [s for s in src if s in data.sources],
                                   "n_present": len(data.present), "ambiguous": amb, "label_mismatch": mis,
                                   "pairs": {s: len(data.sources[s].result.matches) for s in src
                                             if s in data.sources},
                                   "image_error": errors.get(data.key)})
            n_done += 1
            if n_done % 50 == 0:
                log(f"drew {n_done} / {len(drawn)} views")
        if cards:
            scene_sections.append((f"{dataset} / {scene}", cards))

    # Spot-check pictures and page.
    by_key = {v.key: v for v in views}
    spot_cards, items = [], []
    for number, pair in enumerate(sample, start=1):
        data = by_key[pair.view_key]
        crop_rel, frame_rel = f"images/spotcheck/{number:04d}_crop.jpg", f"images/spotcheck/{number:04d}_frame.jpg"
        image = image_of(data)
        if image is None:
            placeholder(errors[data.key], (CROP_LONG_SIDE, CROP_LONG_SIDE // 2)).save(out / crop_rel, quality=85)
            placeholder(errors[data.key], (FRAME_LONG_SIDE // 2, FRAME_LONG_SIDE // 2)).save(out / frame_rel,
                                                                                             quality=85)
        else:
            crop, frame = draw_pair(image, data, pair)
            crop.save(out / crop_rel, quality=88)
            frame.save(out / frame_rel, quality=85)
        gt_short = short_ids(o.obj.object_id for o in data.present).get(pair.object_id, pair.object_id)
        spot_cards.append(spot_card(number, pair, crop_rel, frame_rel, gt_short))
        items.append({"pair_id": pair.pair_id, "view_key": pair.view_key, "object_id": pair.object_id,
                      "detection": pair.detection, "ambiguous": pair.ambiguous})
    n_amb = sum(p.ambiguous for p in sample)
    spot_meta = {"sample_id": sid, "seed": seed, "source": spot_label, "population": population, "items": items}

    folder_text = ", ".join(f"<code>{_esc(f)}</code>" for f in folders)
    shown_text = "; ".join(f"{_esc(d)}: {_esc(', '.join(s))}" for d, s in sorted(shown.items()))
    pair_counts: dict[str, dict[str, int]] = {}
    for source in dict.fromkeys(x for d in shown.values() for x in d):
        matches = [m for v in views if source in v.sources for m in v.sources[source].result.matches]
        pair_counts[source] = {"pairs": len(matches), "ambiguous": sum(m.ambiguous for m in matches),
                               "label_mismatch": sum(not m.label_correct for m in matches)}
    pair_text = "; ".join(f"{_esc(s)}: {c['pairs']} pairs, {c['ambiguous']} ambiguous, {c['label_mismatch']} "
                          f"label mismatches" for s, c in pair_counts.items())
    index_meta = [f"Label folders: {folder_text}",
                  f"Sources shown (pairing source first): {shown_text}. Pairing recomputed from the stored records "
                  f"with the 2D association (class-agnostic Hungarian on 2D IoU), not read from them.",
                  f"{len(views)} views selected, {len(drawn)} shown"
                  + (f" (at most {max_views} per scene)" if max_views is not None else "") + f". {pair_text}."]
    spot_meta_lines = [f"Source: <code>{_esc(spot_label)}</code> · seed {seed} · {len(sample)} pairs "
                       f"sampled from {len(candidates)} ({population['ambiguous']} ambiguous, "
                       f"{population['clear']} clear) in {len(views)} views. Cards are in random order."]
    (out / "index.html").write_text(index_html(name, index_meta, scene_sections), encoding="utf-8")
    (out / "spotcheck.html").write_text(spotcheck_html(name, spot_meta, spot_cards, spot_meta_lines),
                                        encoding="utf-8")

    manifest = {
        "name": name, "created": datetime.now(timezone.utc).isoformat(),
        "label_folders": [str(Path(f).resolve()) for f in folders], "datasets": sorted(shown),
        "scenes_filter": list(scenes) if scenes else None, "sources_shown": shown, "association": "2d",
        "min_iou": {s: sorted({v.sources[s].min_iou for v in views if s in v.sources}) for s in pair_counts},
        "n_views_selected": len(views), "n_views_drawn": len(drawn), "max_views_per_scene": max_views,
        "pairs_per_source": pair_counts,
        "spotcheck": {"sample_id": sid, "seed": seed, "source": spot_label, "source_per_dataset": spot,
                      "sample_size_requested": sample_size, "n_sampled": len(sample), "n_sampled_ambiguous": n_amb,
                      "n_sampled_clear": len(sample) - n_amb, "population": population,
                      "pairs": [p.to_json() for p in sample]},
        "image_errors": errors, "views": manifest_views,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log(f"wrote {out}: {len(drawn)} overlays, {len(sample)} spot-check pairs ({n_amb} ambiguous), "
        f"{len(errors)} image errors")
    return manifest
