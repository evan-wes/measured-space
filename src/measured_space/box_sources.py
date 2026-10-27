"""Automatic label sources from externally produced 3D boxes: CuTR (Cubify Anything) and named external boxes.

CuTR predicts class-agnostic, metric 3D boxes from one RGB-D image. A separate step runs it and writes one
JSONL per scene, ``<predictions>/<dataset>/<scene>.jsonl``, one line per view::

    {"view_key", "dataset", "scene", "camera_id", "frame", "model", "checkpoint_sha256", "score_threshold",
     "image_size": [W, H], "seconds",
     "boxes": [{"center": [3], "size": [3], "rotation": [3x3], "score", "box2d": [x0, y0, x1, y1],
                "center_cam": [3], "label" (optional)}]}

``center``, ``size`` and ``rotation`` are a world :class:`~measured_space.geometry.OrientedBox` (keys as in
:func:`measured_space.pipeline.box_json`). ``box2d`` is CuTR's own 2D box in full-resolution upright image pixels,
with pixel edges at integer coordinates (as :attr:`measured_space.perception.Instance.box2d`). ``center_cam`` is
the box centre in the view's camera frame. ``label`` is optional: a dataset label (the ground truth's object type)
from a labeller that names its boxes; CuTR writes none. Other keys of a line or a box are ignored.
:func:`read_box_predictions` reads this format. Other labellers are converted to it, e.g. the multi-camera
detector Sparse4D by :mod:`measured_space.sparse4d`.

Three automatic sources are built from it. Each is a CPU transform of a view's stored records (one shard of
``scripts/run_labels.py``), followed by :func:`measured_space.pipeline.rematch_view` with the ``"2d"``
association. Pairing with ground truth, the questions, the answers and the oracle check are therefore exactly
the existing procedure.

``cutr_sam3`` (:func:`cutr_sam3_records`): **CuTR geometry on SAM 3's detections.** SAM 3's detections are
associated with the view's CuTR boxes by Hungarian assignment on 2D IoU (:func:`attach_cutr`), class-agnostic,
at IoU >= 0.3 (the default of :class:`~measured_space.matching.Match2DConfig`). The cost is the one
:func:`~measured_space.matching.match_detections_2d` uses: ``1 - IoU`` for a feasible pair, 1 otherwise, so the
summed feasible IoU is maximised. An assigned detection's :class:`~measured_space.pipeline.Lift` has the CuTR
world box. An unassigned one gets a failed lift (no box, with a reason). The detection list, phrases, labels and
``qa_label`` stay SAM 3's. Like a detection whose lift failed for a depth source, an unassigned detection is not
an automatic label. So an automatic count counts the class's SAM 3 detections that received a CuTR box, and
``n_detections_2d`` still reports all of the class's SAM 3 detections.

``cutr_clip`` (:func:`cutr_clip_records`): **pure CuTR, named by CLIP.** The detections are the CuTR boxes
themselves, highest score first, named zero-shot by :mod:`measured_space.clip_naming`. A box whose most
probable phrase is an "other" phrase (wall, floor, ...) gets **no name**. Its record is kept, with the best
phrase, no labels and ``lift_error = "unnamed: <reason>"``, but its lift carries no box. Detections without a
box take no part in the association, so an unnamed box answers no question and enters no count.

``external_named`` (:func:`external_named_records`): **external boxes named by their own labeller.** As
``cutr_clip``, the detections are the prediction line's boxes, highest score first, but each takes its name from
its ``label`` field instead of CLIP. The source has a configurable name (:func:`check_source_name`), e.g.
``sparse4d``, so its items are ``auto_sparse4d``. Boxes scoring below an operating threshold (``min_score``)
are left out entirely: they are not detections. A label that is one of the dataset's classes
(:data:`measured_space.perception.CLASS_PHRASES`) gives the detection that class's prompt phrase, the phrase
the SAM 3 sources store. Its ``labels`` are every class that phrase stands for, the box's own label first, so
the counts name the box by its own class (``qa_label``). ``label_correct`` then means what it means for the
other sources: the phrase names the paired object (a "Transporter" box on a NovaCarter is correct, as a
"mobile robot" detection is). The label itself stays in the prediction line and in the source's provenance
(``external_label``). A box without a label, or with a label outside the dataset's classes, is unnamed, as in
``cutr_clip``: kept with no box and ``lift_error = "unnamed: <reason>"``.

**Lift features** (keys as in :func:`~measured_space.perception.lift_instance` where the meaning is the same):

- ``depth_median``: the camera-frame z of the CuTR box centre (``center_cam[2]``), metres. It is a **proxy**:
  the SAM 3 lifts store the median depth of the mask's pixels, and a box centre lies behind the visible
  surface by up to half the box's depth;
- ``det_score``: the CuTR score;
- ``cutr_box_index``: the box's position in the view's prediction line, to trace it back;
- ``cutr_sam3`` also stores the five mask-only features of SAM 3's mask (``boundary_fraction``,
  ``border_touch``, ``mask_area_px``, ``n_components``, ``n_border_sides``; :data:`MASK_FEATURE_KEYS`) and
  ``pair_iou2d``, the assignment's 2D IoU;
- ``cutr_clip`` also stores ``border_touch`` and ``n_border_sides``: whether the clipped 2D box comes within the
  pipeline's ``boundary_px`` of an image edge, and of how many edges (:func:`box_mask_stats`). It also stores
  ``name_prob``, CLIP's probability of the name (or, for an unnamed box, of its best phrase);
- ``external_named`` stores ``depth_median``, ``det_score``, ``border_touch`` and ``n_border_sides`` as
  ``cutr_clip`` does, ``external_box_index`` (the box's position in its prediction line) instead of
  ``cutr_box_index``, and ``cover_by_nearer`` (:func:`cover_by_nearer`): the largest share of the box's 2D box
  covered by one other kept box whose centre is nearer the camera. The source has no depth map, so it cannot
  test occlusion; ``cover_by_nearer`` says how much an untested occlusion could concern the box. The audit's
  error model does not read it.

The depth-based features (``depth_mad_rel``, ``footprint_hull_fill``, ``n_points`` and the others) are
**absent**. CuTR returns a box, not a point cloud, so they do not exist for these sources, and the audit's
error model reads them as missing (standardised to 0).

**Masks.** A CuTR detection has no mask, and none is rasterised. A view rebuilt from records carries an empty
stand-in mask for every detection anyway (:func:`~measured_space.pipeline.detections_from_records`), and only
its stored mask *statistics* are used. For ``cutr_clip`` and ``external_named`` these statistics come from the
clipped 2D box (:func:`box_mask_stats`) and are marked ``box_mask = 1``. ``mask_area_px`` is then the box area, and
``boundary_fraction`` is left out: for a rectangle it depends only on the rectangle's size, which would make it
a disguised size feature.

**2D boxes.** CuTR's 2D box is clipped to the image before any IoU, because the oracle's visible boxes and SAM
3's boxes lie inside the image.

**Checks** (:func:`check_predictions`). A prediction line must have the view's key and image size. Every box's
``center_cam`` must equal its world centre moved into the view's camera frame with the stored pose, within
:data:`CENTER_CAM_TOLERANCE`. A mismatch means the line describes another frame or pose, so the source refuses
it rather than pair wrong geometry.

**Records.** The transformed records keep the oracle records (``oracle_objects`` and the oracle items) verbatim
and **only the new source**:

- ``cutr_clip`` and ``external_named`` cannot keep the SAM 3 sources: their detection lists differ, and all
  sources of a view must share one detection list;
- ``cutr_sam3`` follows the same rule for symmetry, and so that the depth sources' outputs live only in the
  folder that produced them, with no copies that could drift.

The oracle answers are byte-identical to the source folder's, so items pair by id across folders. The view's
stored timings keep only the stages this source used (image loading, SAM 3 for ``cutr_sam3``, oracle
visibility) and gain the CuTR time (``cutr``) and, for ``cutr_clip``, the CLIP time (``clip_naming``). An
``external_named`` source reads no image here: it keeps the oracle visibility time and gains the prediction
line's ``seconds`` under the source's name, if the line states it.
"""

from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from measured_space.clip_naming import BoxName
from measured_space.geometry import Array, OrientedBox, invert_pose
from measured_space.matching import Match2DConfig, box_iou_2d
from measured_space.perception import CLASS_PHRASES, labels_for_phrase
from measured_space.pipeline import (
    DEPTH_SOURCES,
    MASK_FEATURE_KEYS,
    Lift,
    PipelineConfig,
    ViewResult,
    box_json,
    detections_from_records,
    rematch_view,
)

Box2D = tuple[float, float, float, float]
"""tuple of float: ``(xmin, ymin, xmax, ymax)`` in pixels, pixel edges at integer coordinates."""

SAM3_SOURCE = "cutr_sam3"
"""str: CuTR geometry on SAM 3's detections; items get ``source="auto_cutr_sam3"``."""

CLIP_SOURCE = "cutr_clip"
"""str: CuTR's own boxes named by CLIP; items get ``source="auto_cutr_clip"``."""

EXTERNAL_NAMED = "external_named"
"""str: external boxes named by their ``label`` field; the source's name is chosen (:func:`check_source_name`)."""

VARIANTS: tuple[str, ...] = (SAM3_SOURCE, CLIP_SOURCE, EXTERNAL_NAMED)
"""tuple of str: the automatic sources built from external boxes (the two CuTR sources and the named one)."""

KEPT_TIMINGS: dict[str, tuple[str, ...]] = {SAM3_SOURCE: ("load_image", "segment", "oracle_visibility"),
                                            CLIP_SOURCE: ("load_image", "oracle_visibility"),
                                            EXTERNAL_NAMED: ("oracle_visibility",)}
"""dict of str to tuple of str: stored stage timings each variant depends on (the others are dropped)."""

SOURCE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
"""re.Pattern: an ``external_named`` source name: lower case, digits and underscores, starting with a letter."""

RESERVED_SOURCE_NAMES: frozenset[str] = frozenset(
    {"oracle", SAM3_SOURCE, CLIP_SOURCE, EXTERNAL_NAMED} | {s for v in DEPTH_SOURCES.values() for s in v})
"""frozenset of str: names an ``external_named`` source may not take (the oracle, the other sources, the depths)."""

COVER_SHARE = 0.5
"""float: a box at least this much covered by one nearer box counts as covered (:func:`overlap_summary`)."""

CENTER_CAM_TOLERANCE = 1e-3
"""float: largest allowed difference between ``center_cam`` and the world centre in the view's camera, metres."""

NO_CUTR_BOX = "no CuTR box in the view"
"""str: lift error of a SAM 3 detection in a view where CuTR found nothing."""

NOT_ASSIGNED = "no CuTR box assigned at 2D IoU >= {min_iou}"
"""str: lift error of a SAM 3 detection left unassigned (formatted with the threshold)."""

UNNAMED_PREFIX = "unnamed: "
"""str: prefix of the lift error of a box that got no name (followed by the reason)."""

NO_LABEL = "no label"
"""str: why an ``external_named`` box without a ``label`` is unnamed."""

UNKNOWN_LABEL = "label {label!r} is not a {dataset} class"
"""str: why an ``external_named`` box with a label outside the dataset's classes is unnamed (formatted)."""

UNLABELLED_PHRASE = "unlabelled"
"""str: the phrase stored for an ``external_named`` box without a label."""


@dataclass(frozen=True)
class CutrBox:
    """One external box in one view (CuTR's, or another labeller's in the same format).

    Parameters
    ----------
    box : OrientedBox
        World box, metres.
    score : float
        The labeller's confidence.
    box2d : tuple of float
        The 2D box ``(xmin, ymin, xmax, ymax)``, full-resolution upright pixels, edges at integer coordinates,
        as written (CuTR's are not clipped).
    center_cam : numpy.ndarray, shape (3,)
        Box centre in the view's camera frame, metres.
    label : str or None, default None
        The labeller's dataset label for the box, if it names its boxes (CuTR does not).
    """

    box: OrientedBox
    score: float
    box2d: Box2D
    center_cam: Array
    label: str | None = None


@dataclass(frozen=True)
class ViewPredictions:
    """CuTR's output for one view.

    Parameters
    ----------
    view_key : str
        ``"<dataset>/<scene>/<camera_id>/<frame>"`` (:func:`measured_space.pipeline.view_key`).
    model : str
        Model name, e.g. ``"cutr_rgbd"``.
    checkpoint_sha256 : str or None
        Checksum of the weights.
    score_threshold : float or None
        Score below which boxes were dropped before writing.
    image_size : tuple of int
        ``(W, H)`` of the image the 2D boxes refer to.
    seconds : float or None
        CuTR's time for this view.
    boxes : tuple of CutrBox
        The boxes, in file order.
    """

    view_key: str
    model: str
    checkpoint_sha256: str | None
    score_threshold: float | None
    image_size: tuple[int, int]
    seconds: float | None
    boxes: tuple[CutrBox, ...]

    def meta(self) -> dict[str, Any]:
        """The prediction's provenance, for the records.

        Returns
        -------
        dict
            ``model``, ``checkpoint_sha256``, ``score_threshold``, ``image_size``, ``seconds`` and ``n_cutr_boxes``.
        """
        return {"model": self.model, "checkpoint_sha256": self.checkpoint_sha256,
                "score_threshold": self.score_threshold, "image_size": list(self.image_size),
                "seconds": self.seconds, "n_cutr_boxes": len(self.boxes)}


# --------------------------------------------------------------------------------------------------------
# Reading predictions
# --------------------------------------------------------------------------------------------------------


def _optional_float(value: Any) -> float | None:
    """A finite float, or ``None``.

    Parameters
    ----------
    value : Any
        A JSON value.

    Returns
    -------
    float or None
        ``float(value)`` if it is a finite number, else ``None``.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def parse_view_predictions(record: Mapping[str, Any]) -> ViewPredictions:
    """Parse one line of a predictions file.

    Parameters
    ----------
    record : mapping
        The parsed JSON line (see the module docstring).

    Returns
    -------
    ViewPredictions
        The view's boxes and provenance.

    Raises
    ------
    ValueError
        If the identity fields disagree with ``view_key`` or a box is malformed.
    KeyError
        If a required field is missing.
    """
    key = str(record["view_key"])
    parts = [record.get(k) for k in ("dataset", "scene", "camera_id", "frame")]
    if all(p is not None for p in parts) and "/".join(str(p) for p in parts) != key:
        raise ValueError(f"prediction identity {parts} disagrees with view_key {key!r}")
    boxes = []
    for i, b in enumerate(record["boxes"]):
        box2d = tuple(float(v) for v in b["box2d"])
        center_cam = np.asarray(b["center_cam"], dtype=np.float64)
        if len(box2d) != 4 or center_cam.shape != (3,):
            raise ValueError(f"{key}: box {i} has a malformed box2d or center_cam")
        label = b.get("label")
        boxes.append(CutrBox(OrientedBox(b["center"], b["size"], b["rotation"]), float(b["score"]),
                             box2d, center_cam,                                       # type: ignore[arg-type]
                             None if label is None else str(label)))
    width, height = (int(v) for v in record["image_size"])
    return ViewPredictions(key, str(record.get("model", "")), record.get("checkpoint_sha256"),
                           _optional_float(record.get("score_threshold")), (width, height),
                           _optional_float(record.get("seconds")), tuple(boxes))


def read_box_predictions(path: str | Path) -> dict[str, ViewPredictions]:
    """Read one scene's predictions file.

    Parameters
    ----------
    path : str or pathlib.Path
        A ``<scene>.jsonl`` file.

    Returns
    -------
    dict of str to ViewPredictions
        View key -> predictions, in file order.

    Raises
    ------
    ValueError
        If a view appears twice, or as :func:`parse_view_predictions`.
    """
    out: dict[str, ViewPredictions] = {}
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            view = parse_view_predictions(json.loads(line))
            if view.view_key in out:
                raise ValueError(f"{path}: view {view.view_key} appears twice")
            out[view.view_key] = view
    return out


def predictions_file(root: str | Path, dataset: str, scene: str) -> Path | None:
    """Where a scene's predictions are, if anywhere.

    Parameters
    ----------
    root : str or pathlib.Path
        The predictions folder (holding one folder per dataset).
    dataset : str
        Dataset name.
    scene : str
        Scene name; a ``/`` in it may be kept (nested folders) or written as ``__``.

    Returns
    -------
    pathlib.Path or None
        ``<root>/<dataset>/<scene>.jsonl`` or ``<root>/<dataset>/<scene with / as __>.jsonl``, whichever exists;
        ``None`` if neither does.

    Raises
    ------
    ValueError
        If both spellings exist as different files.
    """
    folder = Path(root) / dataset
    candidates = list(dict.fromkeys([folder / f"{scene}.jsonl", folder / f"{scene.replace('/', '__')}.jsonl"]))
    found = [p for p in candidates if p.is_file()]
    if len(found) > 1:
        raise ValueError(f"predictions for {dataset}/{scene} exist twice: {found}")
    return found[0] if found else None


# --------------------------------------------------------------------------------------------------------
# Checks and 2D helpers
# --------------------------------------------------------------------------------------------------------


def check_predictions(view_record: Mapping[str, Any], predictions: ViewPredictions,
                      tolerance: float = CENTER_CAM_TOLERANCE) -> float:
    """Check that a prediction line belongs to a stored view and agrees with its pose.

    Parameters
    ----------
    view_record : mapping
        The shard's ``"view"`` record (``view_key``, ``camera``, ``T_wc``).
    predictions : ViewPredictions
        The view's predictions.
    tolerance : float, default CENTER_CAM_TOLERANCE
        Largest allowed ``|T_cw @ center - center_cam|`` per coordinate, metres.

    Returns
    -------
    float
        The largest such difference over the boxes (0 for no boxes), metres.

    Raises
    ------
    ValueError
        If the view key or the image size differs, or a box's ``center_cam`` disagrees with its world centre.
    """
    key = view_record["view_key"]
    if predictions.view_key != key:
        raise ValueError(f"predictions for {predictions.view_key} given for view {key}")
    camera = view_record["camera"]
    if predictions.image_size != (int(camera["width"]), int(camera["height"])):
        raise ValueError(f"{key}: predictions are for a {predictions.image_size} image, the view's camera is "
                         f"{camera['width']}x{camera['height']}")
    if not predictions.boxes:
        return 0.0
    T_cw = invert_pose(np.asarray(view_record["T_wc"], dtype=np.float64))
    centers = np.stack([b.box.center for b in predictions.boxes])
    expected = centers @ T_cw[:3, :3].T + T_cw[:3, 3]
    diff = float(np.max(np.abs(expected - np.stack([b.center_cam for b in predictions.boxes]))))
    if not diff <= tolerance:
        raise ValueError(f"{key}: center_cam differs from the world centre in the view's camera by {diff:.4g} m "
                         f"(tolerance {tolerance} m); the predictions use another frame or pose")
    return diff


def clip_box2d(box2d: Sequence[float], width: int, height: int) -> Box2D:
    """Clip a 2D box to the image.

    Parameters
    ----------
    box2d : sequence of float, shape (4,)
        ``(xmin, ymin, xmax, ymax)``, pixels, edges at integer coordinates.
    width, height : int
        Image size, pixels.

    Returns
    -------
    tuple of float
        The box clipped to ``[0, width] x [0, height]``.
    """
    x0, y0, x1, y1 = (float(v) for v in box2d)
    return (float(np.clip(x0, 0, width)), float(np.clip(y0, 0, height)), float(np.clip(x1, 0, width)),
            float(np.clip(y1, 0, height)))


def attach_cutr(det_boxes2d: Sequence[Sequence[float]], cutr_boxes2d: Sequence[Sequence[float]],
                config: Match2DConfig | None = None) -> dict[int, tuple[int, float]]:
    """Assign CuTR boxes to detections by Hungarian assignment on 2D IoU, class-agnostic.

    The assignment maximises the summed IoU over pairs with ``IoU >= config.min_iou``: a feasible pair costs
    ``1 - IoU`` and any other pair 1, the cost of leaving both unassigned (as in
    :func:`measured_space.matching.match_detections_2d`).

    Parameters
    ----------
    det_boxes2d : sequence of sequence of float, each shape (4,)
        The detections' 2D boxes, pixels, edges at integer coordinates.
    cutr_boxes2d : sequence of sequence of float, each shape (4,)
        The CuTR boxes' 2D boxes in the same convention.
    config : Match2DConfig, optional
        The IoU threshold; defaults to ``Match2DConfig()`` (0.3).

    Returns
    -------
    dict of int to tuple of (int, float)
        Detection index -> ``(CuTR box index, IoU)`` for every assigned detection.
    """
    config = config or Match2DConfig()
    if len(det_boxes2d) == 0 or len(cutr_boxes2d) == 0:
        return {}
    iou = box_iou_2d(np.asarray(det_boxes2d, dtype=np.float64).reshape(-1, 4),
                     np.asarray(cutr_boxes2d, dtype=np.float64).reshape(-1, 4))
    feasible = iou >= config.min_iou
    rows, cols = linear_sum_assignment(np.where(feasible, 1.0 - iou, 1.0))
    return {int(r): (int(c), float(iou[r, c])) for r, c in zip(rows, cols) if feasible[r, c]}


def box_mask_stats(box2d: Sequence[float], width: int, height: int, boundary_px: float) -> dict[str, float]:
    """Mask statistics of a detection that has a 2D box and no mask.

    Parameters
    ----------
    box2d : sequence of float, shape (4,)
        ``(xmin, ymin, xmax, ymax)`` inside the image, pixels, edges at integer coordinates.
    width, height : int
        Image size, pixels.
    boundary_px : float
        A box side within this many pixels of an image edge touches it.

    Returns
    -------
    dict of str to float
        ``mask_area_px`` (the box's area, px^2), ``n_components`` (1), ``border_touch`` (1 if a side is within
        ``boundary_px`` of an image edge, else 0), ``n_border_sides`` (how many sides are) and ``box_mask`` (1:
        these statistics come from a box, not a mask). ``boundary_fraction`` is left out (see the module
        docstring).
    """
    x0, y0, x1, y1 = (float(v) for v in box2d)
    sides = [y0 <= boundary_px, y1 >= height - boundary_px, x0 <= boundary_px, x1 >= width - boundary_px]
    return {"mask_area_px": max(0.0, x1 - x0) * max(0.0, y1 - y0), "n_components": 1.0,
            "border_touch": float(any(sides)), "n_border_sides": float(sum(sides)), "box_mask": 1.0}


def cover_by_nearer(boxes2d: Sequence[Sequence[float]], depths: Sequence[float]) -> Array:
    """For each 2D box, the largest share of its area that one box with a nearer centre covers.

    A box drawn from a 3D box without a depth map is kept even where a nearer object hides it. This measures
    how much that could matter: 0 means no nearer box overlaps it, 1 that one nearer box covers it whole.

    Parameters
    ----------
    boxes2d : sequence of sequence of float, each shape (4,)
        ``(xmin, ymin, xmax, ymax)``, pixels, edges at integer coordinates.
    depths : sequence of float
        Each box's centre depth in the camera (``center_cam`` z), metres; same length.

    Returns
    -------
    numpy.ndarray, shape (N,)
        Per box, ``max_j area(box ∩ box_j) / area(box)`` over boxes ``j`` with a strictly smaller depth; 0 for a
        box with no area or no nearer overlapping box.

    Raises
    ------
    ValueError
        If the two sequences differ in length.
    """
    b = np.asarray(boxes2d, dtype=np.float64).reshape(-1, 4)
    d = np.asarray(depths, dtype=np.float64).reshape(-1)
    if len(b) != len(d):
        raise ValueError(f"{len(b)} boxes but {len(d)} depths")
    if len(b) == 0:
        return np.zeros(0)
    width = np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0])
    height = np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1])
    intersection = np.clip(width, 0.0, None) * np.clip(height, 0.0, None)
    area = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(area[:, None] > 0.0, intersection / area[:, None], 0.0)
    nearer = d[None, :] < d[:, None]                                     # [i, j]: box j is nearer than box i
    return np.clip(np.where(nearer, share, 0.0), 0.0, 1.0).max(axis=1)


def overlap_summary(boxes2d: Sequence[Sequence[float]], depths: Sequence[float], *,
                    min_iou: float = Match2DConfig.min_iou, min_cover: float = COVER_SHARE) -> dict[str, int]:
    """How often a view's 2D boxes overlap: the cases an untested occlusion can affect.

    Parameters
    ----------
    boxes2d : sequence of sequence of float, each shape (4,)
        ``(xmin, ymin, xmax, ymax)``, pixels, edges at integer coordinates.
    depths : sequence of float
        Each box's centre depth in the camera, metres.
    min_iou : float, default Match2DConfig.min_iou
        Pairs at this 2D IoU or more could both pair with the same object under the 2D association.
    min_cover : float, default COVER_SHARE
        A box at least this much covered by one nearer box (:func:`cover_by_nearer`) counts as covered.

    Returns
    -------
    dict of str to int
        ``n_boxes``; ``n_pairs_overlap`` (pairs with any overlap); ``n_pairs_iou`` (pairs at IoU >= ``min_iou``);
        ``n_boxes_overlap`` (boxes overlapping at least one other); ``n_covered`` (boxes covered by a nearer one).
    """
    b = np.asarray(boxes2d, dtype=np.float64).reshape(-1, 4)
    n = len(b)
    if n < 2:
        return {"n_boxes": n, "n_pairs_overlap": 0, "n_pairs_iou": 0, "n_boxes_overlap": 0, "n_covered": 0}
    iou = box_iou_2d(b, b)
    width = np.minimum(b[:, None, 2], b[None, :, 2]) - np.maximum(b[:, None, 0], b[None, :, 0])
    height = np.minimum(b[:, None, 3], b[None, :, 3]) - np.maximum(b[:, None, 1], b[None, :, 1])
    overlap = (width > 0.0) & (height > 0.0)
    np.fill_diagonal(overlap, False)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    return {"n_boxes": n, "n_pairs_overlap": int((overlap & upper).sum()),
            "n_pairs_iou": int(((iou >= min_iou) & upper).sum()), "n_boxes_overlap": int(overlap.any(axis=1).sum()),
            "n_covered": int((cover_by_nearer(b, depths) >= min_cover).sum())}


def check_source_name(name: str) -> str:
    """Check the name of an ``external_named`` source.

    Parameters
    ----------
    name : str
        The proposed name; items get ``source="auto_<name>"``.

    Returns
    -------
    str
        The name, unchanged.

    Raises
    ------
    ValueError
        If it does not match :data:`SOURCE_NAME_PATTERN` or is one of :data:`RESERVED_SOURCE_NAMES`.
    """
    if not SOURCE_NAME_PATTERN.match(name):
        raise ValueError(f"source name {name!r} must be lower case letters, digits and underscores, starting with "
                         "a letter (at most 40 characters)")
    if name in RESERVED_SOURCE_NAMES:
        raise ValueError(f"source name {name!r} is taken (reserved: {sorted(RESERVED_SOURCE_NAMES)})")
    return name


def external_names(dataset: str, label: str | None) -> tuple[str, tuple[str, ...]] | None:
    """The phrase and candidate labels of a box named by its own labeller.

    Parameters
    ----------
    dataset : str
        Dataset name (a key of :data:`measured_space.perception.CLASS_PHRASES` to have any classes).
    label : str or None
        The box's dataset label.

    Returns
    -------
    tuple or None
        ``(phrase, labels)``: the class's prompt phrase and every label that phrase stands for, ``label`` first;
        ``None`` if ``label`` is ``None`` or not one of the dataset's classes.
    """
    phrases = CLASS_PHRASES.get(dataset, {})
    if label is None or label not in phrases:
        return None
    phrase = phrases[label]
    return phrase, (label, *(other for other in labels_for_phrase(dataset, phrase) if other != label))


# --------------------------------------------------------------------------------------------------------
# Transforming a view's records
# --------------------------------------------------------------------------------------------------------


def _identity(view_record: Mapping[str, Any]) -> dict[str, Any]:
    """A view's identity fields, as every record of it carries them.

    Parameters
    ----------
    view_record : mapping
        The shard's ``"view"`` record.

    Returns
    -------
    dict
        ``view_key``, ``dataset``, ``scene``, ``camera_id``, ``frame``.
    """
    return {k: view_record[k] for k in ("view_key", "dataset", "scene", "camera_id", "frame")}


def _detection_record(identity: Mapping[str, Any], index: int, source: str, phrase: str, labels: Sequence[str],
                      score: float, box2d: Sequence[float], mask: Mapping[str, float], lift: Lift
                      ) -> dict[str, Any]:
    """One stored detection record for one source, in the form :func:`detections_from_records` reads.

    Parameters
    ----------
    identity : mapping
        The view's identity fields.
    index : int
        Detection index (highest score first).
    source : str
        The automatic source.
    phrase : str
        The detection's phrase.
    labels : sequence of str
        Dataset labels the phrase stands for (empty for none).
    score : float
        Detection score.
    box2d : sequence of float, shape (4,)
        2D box, pixels, edges at integer coordinates.
    mask : mapping of str to float
        Mask statistics.
    lift : Lift
        The source's box, features and failure reason.

    Returns
    -------
    dict
        Identity, ``detection``, ``source``, ``phrase``, ``labels``, ``score``, ``box2d``, ``mask``,
        ``lift_stride``, ``lift_error``, ``box`` (``None`` without one) and ``features``.
    """
    return {**identity, "detection": index, "source": source, "phrase": phrase, "labels": list(labels),
            "score": float(score), "box2d": [float(v) for v in box2d], "mask": dict(mask),
            "lift_stride": lift.stride, "lift_error": lift.error,
            "box": None if lift.box is None else box_json(lift.box), "features": dict(lift.features)}


def _source_records(records: Mapping[str, Any], source: str, detection_records: list[dict[str, Any]],
                    extra_timings: Mapping[str, float | None], kept: Sequence[str] | None = None) -> dict[str, Any]:
    """A view's records with one automatic source: the oracle records verbatim, the new detections.

    Parameters
    ----------
    records : mapping
        The stored view records (``view``, ``oracle_objects``, ``detections``, ``qa``).
    source : str
        The new automatic source.
    detection_records : list of dict
        Its detection records, one per detection.
    extra_timings : mapping of str to float or None
        Stage timings to add (``None`` values are skipped), seconds.
    kept : sequence of str, optional
        Stored stage timings to keep; defaults to ``KEPT_TIMINGS[source]``.

    Returns
    -------
    dict
        ``view`` (a copy naming ``source`` as the only automatic source, with its detection count and the
        kept timings plus ``extra_timings``), ``oracle_objects`` (deep copy, unchanged), ``detections`` and
        ``qa`` (the oracle items only, deep copy, unchanged).
    """
    view = copy.deepcopy(dict(records["view"]))
    stored = view.get("timings") or {}
    timings = {k: float(stored[k]) for k in (KEPT_TIMINGS[source] if kept is None else kept) if k in stored}
    timings.update({k: float(v) for k, v in extra_timings.items() if v is not None})
    oracle_items = copy.deepcopy(list(records["qa"].get("oracle", [])))
    view.update(depth_sources=[source], n_detections=len(detection_records), timings=timings,
                seconds=float(sum(timings.values())), n_items={"oracle": len(oracle_items)},
                auto_unanswered={f"auto_{source}": []})
    return {"view": view, "oracle_objects": copy.deepcopy(list(records["oracle_objects"])),
            "detections": detection_records, "qa": {"oracle": oracle_items}}


def cutr_sam3_records(records: Mapping[str, Any], predictions: ViewPredictions, *,
                      config: Match2DConfig | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """A view's records with the ``cutr_sam3`` source: CuTR boxes attached to SAM 3's detections.

    Parameters
    ----------
    records : mapping
        The stored view records (any automatic sources; they must agree on the detections).
    predictions : ViewPredictions
        CuTR's output for the view.
    config : Match2DConfig, optional
        Threshold of the attachment (:func:`attach_cutr`); defaults to ``Match2DConfig()`` (0.3).

    Returns
    -------
    records : dict
        The records with :data:`SAM3_SOURCE` as the only automatic source (see :func:`_source_records`); input for
        :func:`measured_space.pipeline.rematch_view`.
    info : dict
        Provenance: ``variant``, the predictions' :meth:`~ViewPredictions.meta`, ``attach_min_iou``,
        ``n_detections``, ``n_assigned`` and ``center_cam_max_abs_diff_m``.

    Raises
    ------
    ValueError
        As :func:`check_predictions` and :func:`measured_space.pipeline.detections_from_records`.
    """
    config = config or Match2DConfig()
    view_record = records["view"]
    diff = check_predictions(view_record, predictions)
    detections, _ = detections_from_records(records["detections"], tuple(view_record["depth_sources"]),
                                            int(view_record["n_detections"]))
    width, height = predictions.image_size
    cutr_2d = [clip_box2d(b.box2d, width, height) for b in predictions.boxes]
    assigned = attach_cutr([d.instance.box2d for d in detections], cutr_2d, config)
    identity = _identity(view_record)
    failure = NOT_ASSIGNED.format(min_iou=config.min_iou) if predictions.boxes else NO_CUTR_BOX
    out = []
    for det in detections:
        hit = assigned.get(det.index)
        if hit is None:
            lift = Lift(None, {}, failure, 1)
        else:
            j, iou = hit
            b = predictions.boxes[j]
            features = {k: float(det.mask_stats[k]) for k in MASK_FEATURE_KEYS if k in det.mask_stats}
            features.update(depth_median=float(b.center_cam[2]), det_score=b.score, pair_iou2d=iou,
                            cutr_box_index=float(j))
            lift = Lift(b.box, features, None, 1)
        out.append(_detection_record(identity, det.index, SAM3_SOURCE, det.instance.phrase, det.labels,
                                     det.instance.score, det.instance.box2d, det.mask_stats, lift))
    info = {"variant": SAM3_SOURCE, **predictions.meta(), "attach_min_iou": config.min_iou,
            "n_detections": len(detections), "n_assigned": len(assigned), "center_cam_max_abs_diff_m": diff}
    return _source_records(records, SAM3_SOURCE, out, {"cutr": predictions.seconds}), info


def cutr_clip_records(records: Mapping[str, Any], predictions: ViewPredictions, names: Sequence[BoxName], *,
                      naming_seconds: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """A view's records with the ``cutr_clip`` source: CuTR's boxes as the detections, named by CLIP.

    Parameters
    ----------
    records : mapping
        The stored view records.
    predictions : ViewPredictions
        CuTR's output for the view.
    names : sequence of BoxName
        CLIP's verdict per CuTR box, in prediction order (:func:`measured_space.clip_naming.name_boxes` on the
        boxes' 2D boxes).
    naming_seconds : float, optional
        Time CLIP took for the view, recorded as the ``clip_naming`` stage.

    Returns
    -------
    records : dict
        The records with :data:`CLIP_SOURCE` as the only automatic source, one detection per CuTR box (highest
        score first); input for :func:`measured_space.pipeline.rematch_view`.
    info : dict
        Provenance: ``variant``, the predictions' :meth:`~ViewPredictions.meta`, ``n_detections``, ``n_named``,
        ``n_unnamed`` (by reason), ``center_cam_max_abs_diff_m``, and per detection ``cutr_box_index``,
        ``clip_best_phrase`` and ``clip_probs`` (rounded to 6 decimals).

    Raises
    ------
    ValueError
        If ``names`` does not have one entry per box, a name is not one of the dataset's phrases, or as
        :func:`check_predictions`.
    """
    view_record = records["view"]
    if len(names) != len(predictions.boxes):
        raise ValueError(f"{view_record['view_key']}: {len(names)} names for {len(predictions.boxes)} boxes")
    diff = check_predictions(view_record, predictions)
    boundary_px = PipelineConfig.from_json(view_record["config"]).boundary_px
    width, height = predictions.image_size
    dataset = view_record["dataset"]
    identity = _identity(view_record)
    order = sorted(range(len(predictions.boxes)), key=lambda j: (-predictions.boxes[j].score, j))
    out = []
    for index, j in enumerate(order):
        b, name = predictions.boxes[j], names[j]
        box2d = clip_box2d(b.box2d, width, height)
        mask = box_mask_stats(box2d, width, height, boundary_px)
        features = {"depth_median": float(b.center_cam[2]), "border_touch": mask["border_touch"],
                    "n_border_sides": mask["n_border_sides"], "det_score": b.score, "name_prob": name.prob,
                    "cutr_box_index": float(j)}
        if name.phrase is None:
            phrase, labels = name.best_phrase, ()
            lift = Lift(None, features, f"{UNNAMED_PREFIX}{name.unnamed_reason}", 1)
        else:
            phrase, labels = name.phrase, labels_for_phrase(dataset, name.phrase)
            if not labels:
                raise ValueError(f"{view_record['view_key']}: name {name.phrase!r} is not a {dataset} phrase")
            lift = Lift(b.box, features, None, 1)
        out.append(_detection_record(identity, index, CLIP_SOURCE, phrase, labels, b.score, box2d, mask, lift))
    unnamed = Counter(names[j].unnamed_reason for j in order if names[j].phrase is None)
    info = {"variant": CLIP_SOURCE, **predictions.meta(), "n_detections": len(order),
            "n_named": len(order) - sum(unnamed.values()), "n_unnamed": dict(sorted(unnamed.items())),
            "center_cam_max_abs_diff_m": diff, "cutr_box_index": order,
            "clip_best_phrase": [names[j].best_phrase for j in order],
            "clip_probs": [[round(p, 6) for p in names[j].probs] for j in order]}
    timings = {"cutr": predictions.seconds, "clip_naming": naming_seconds}
    return _source_records(records, CLIP_SOURCE, out, timings), info


def external_named_records(records: Mapping[str, Any], predictions: ViewPredictions, source: str, *,
                           min_score: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """A view's records with an ``external_named`` source: the line's boxes as detections, named by their labels.

    Parameters
    ----------
    records : mapping
        The stored view records.
    predictions : ViewPredictions
        The labeller's output for the view; each box's ``label`` names it.
    source : str
        The source's name (:func:`check_source_name`); items get ``source="auto_<source>"``.
    min_score : float, optional
        Operating threshold: boxes scoring below it are left out (not detections). ``None`` keeps every box.

    Returns
    -------
    records : dict
        The records with ``source`` as the only automatic source, one detection per kept box (highest score
        first); input for :func:`measured_space.pipeline.rematch_view`.
    info : dict
        Provenance: ``variant``, ``source``, the predictions' :meth:`~ViewPredictions.meta`, ``min_score``,
        ``n_below_min_score``, ``n_detections``, ``n_named``, ``n_unnamed`` (by reason),
        ``center_cam_max_abs_diff_m``, ``overlap`` (:func:`overlap_summary` of the kept boxes), and per detection
        ``external_box_index`` and ``external_label``.

    Raises
    ------
    ValueError
        For a bad source name or a non-finite ``min_score``, or as :func:`check_predictions`.
    """
    check_source_name(source)
    if min_score is not None and not math.isfinite(min_score):
        raise ValueError(f"min_score must be finite, got {min_score}")
    view_record = records["view"]
    diff = check_predictions(view_record, predictions)
    boundary_px = PipelineConfig.from_json(view_record["config"]).boundary_px
    width, height = predictions.image_size
    dataset = view_record["dataset"]
    identity = _identity(view_record)
    boxes = predictions.boxes
    kept_idx = [j for j, b in enumerate(boxes) if min_score is None or b.score >= min_score]
    order = sorted(kept_idx, key=lambda j: (-boxes[j].score, j))
    clipped = [clip_box2d(boxes[j].box2d, width, height) for j in order]
    cover = cover_by_nearer(clipped, [float(boxes[j].center_cam[2]) for j in order])
    out = []
    unnamed: Counter = Counter()
    for index, (j, box2d) in enumerate(zip(order, clipped)):
        b = boxes[j]
        mask = box_mask_stats(box2d, width, height, boundary_px)
        features = {"depth_median": float(b.center_cam[2]), "border_touch": mask["border_touch"],
                    "n_border_sides": mask["n_border_sides"], "det_score": b.score,
                    "external_box_index": float(j), "cover_by_nearer": float(cover[index])}
        named = external_names(dataset, b.label)
        if named is None:
            reason = NO_LABEL if b.label is None else UNKNOWN_LABEL.format(label=b.label, dataset=dataset)
            unnamed[reason] += 1
            phrase, labels = (UNLABELLED_PHRASE if b.label is None else b.label), ()
            lift = Lift(None, features, f"{UNNAMED_PREFIX}{reason}", 1)
        else:
            phrase, labels = named
            lift = Lift(b.box, features, None, 1)
        out.append(_detection_record(identity, index, source, phrase, labels, b.score, box2d, mask, lift))
    info = {"variant": EXTERNAL_NAMED, "source": source, **predictions.meta(), "min_score": min_score,
            "n_below_min_score": len(boxes) - len(kept_idx), "n_detections": len(order),
            "n_named": len(order) - sum(unnamed.values()), "n_unnamed": dict(sorted(unnamed.items())),
            "center_cam_max_abs_diff_m": diff,
            "overlap": overlap_summary(clipped, [float(boxes[j].center_cam[2]) for j in order]),
            "external_box_index": order, "external_label": [boxes[j].label for j in order]}
    transformed = _source_records(records, source, out, {source: predictions.seconds},
                                  kept=KEPT_TIMINGS[EXTERNAL_NAMED])
    return transformed, info


def run_box_source(records: Mapping[str, Any], predictions: ViewPredictions, variant: str, *,
                   names: Sequence[BoxName] | None = None, config: Match2DConfig | None = None,
                   naming_seconds: float | None = None, source_name: str | None = None,
                   min_score: float | None = None) -> tuple[ViewResult, dict[str, Any]]:
    """Build one view's source from external boxes and redo its association and answers (``"2d"``).

    Parameters
    ----------
    records : mapping
        The stored view records (one shard's ``"records"``).
    predictions : ViewPredictions
        The external boxes for the view (CuTR's, or a named labeller's).
    variant : str
        One of :data:`VARIANTS`.
    names : sequence of BoxName, optional
        Required for :data:`CLIP_SOURCE`: CLIP's verdict per CuTR box, in prediction order.
    config : Match2DConfig, optional
        Threshold of the :data:`SAM3_SOURCE` attachment; defaults to ``Match2DConfig()``.
    naming_seconds : float, optional
        CLIP's time for the view (:data:`CLIP_SOURCE` only).
    source_name : str, optional
        Required for :data:`EXTERNAL_NAMED`: the source's name (:func:`check_source_name`).
    min_score : float, optional
        :data:`EXTERNAL_NAMED` only: the operating threshold (:func:`external_named_records`).

    Returns
    -------
    result : ViewResult
        The view with the source's matches and items (``source="auto_<variant>"``, or ``"auto_<source_name>"``
        for :data:`EXTERNAL_NAMED`); the oracle items are the stored ones, verbatim.
    info : dict
        Provenance from :func:`cutr_sam3_records`, :func:`cutr_clip_records` or :func:`external_named_records`.

    Raises
    ------
    ValueError
        For an unknown variant, :data:`CLIP_SOURCE` without ``names``, :data:`EXTERNAL_NAMED` without
        ``source_name``, ``source_name`` or ``min_score`` with another variant, or as the transforms and
        :func:`~measured_space.pipeline.rematch_view`.
    """
    if variant != EXTERNAL_NAMED and (source_name is not None or min_score is not None):
        raise ValueError(f"source_name and min_score are for the {EXTERNAL_NAMED} variant only")
    if variant == SAM3_SOURCE:
        transformed, info = cutr_sam3_records(records, predictions, config=config)
    elif variant == CLIP_SOURCE:
        if names is None:
            raise ValueError(f"the {CLIP_SOURCE} source needs CLIP names for the boxes")
        transformed, info = cutr_clip_records(records, predictions, names, naming_seconds=naming_seconds)
    elif variant == EXTERNAL_NAMED:
        if source_name is None:
            raise ValueError(f"the {EXTERNAL_NAMED} variant needs a source name")
        transformed, info = external_named_records(records, predictions, source_name, min_score=min_score)
    else:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    return rematch_view(transformed, association="2d"), info
