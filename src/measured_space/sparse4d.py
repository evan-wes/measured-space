"""NVIDIA TAO Sparse4D as an automatic 3D labeller for fixed cameras: its scene boxes as per-view predictions.

Sparse4D is a published multi-camera 3D detector. A frozen NGC checkpoint is run **unmodified**, outside this
package, over every camera of a Smart Spaces scene at once. Per scene it writes, into one folder per checkpoint
tag:

- ``<sparse4d>/<scene>.jsonl``: one line per box, ``{"scene", "frame", "class_id", "class_name", "score",
  "cls_score", "track_id", "center", "size_wlh", "yaw", "velocity"}``, for every box with ``score`` at least the
  run's ``jsonl_thr``. ``scene`` is the scene's folder name (``Warehouse_017``), ``frame`` the frame index,
  ``class_id`` the 2026 challenge's class id, ``score`` the decoder's ``sigmoid(class) * sigmoid(centerness)``,
  ``track_id`` the detector's instance-bank id;
- ``<sparse4d>/<scene>.meta.json``, written after the JSONL is closed: checkpoint tag and SHA-256, split, the
  processed frames (count, first, last and ``ids_sha256``, the SHA-256 of the comma-joined frame indices),
  ``jsonl_thr``, ``num_output_topk``, ``rows_jsonl``, timings.

This module turns those scene boxes into the per-view prediction format of :mod:`measured_space.box_sources`, so
the boxes go through the same pairing, audit, miss audit, recall control and VLM scoring as the handheld
labellers' boxes.

**Boxes** (:func:`box_from_wlh_yaw`). The boxes are in the dataset's world frame (metres, z up), with ``size_wlh``
the full side lengths along the box's local x, y, z (width, length, height: the field order of the ground
truth's ``3d bounding box scale``) and ``yaw`` the rotation about +z in radians. Each is built with
:func:`measured_space.datasets.smartspaces.box_from_annotation` with pitch and roll 0, the function that builds the
oracle's boxes, so a ground-truth box passed through the converter is the oracle's :class:`OrientedBox`
exactly.

**Classes** (:func:`dataset_label`). ``class_id`` is mapped to the dataset's object type by
:data:`CLASS_LABELS` (the challenge's ids), and ``class_name`` must agree with it. The object types are the ground
truth's ``object type`` strings, the keys of :data:`measured_space.perception.SMARTSPACES_PHRASES`. An id outside
the table gets no label; the box source then leaves the box unnamed.

**Per view** (:func:`convert_view`). The views are the label folder's shards: one per scene, camera and frame.
For each view, the scene's boxes at that frame with ``score >= min_score`` are:

1. moved into the camera with the view's stored pose (:func:`measured_space.pipeline.view_from_record`). A box
   whose centre is at or behind the camera plane is dropped (``behind_camera``);
2. projected: the 2D extent of the box's 8 projected corners (:func:`measured_space.pipeline.projected_extent`,
   pixel centres at integers), moved half a pixel to the pixel-edge convention of
   :attr:`measured_space.perception.Instance.box2d`, and clipped to the image. A box with a corner within
   :attr:`KeepRule.min_z` of the camera plane is dropped (``near_camera_plane``): its extent is unbounded, so
   the next rule would drop it anyway;
3. kept if the clipped box has area (else ``outside_image``), holds at least
   :attr:`KeepRule.min_in_image_share` (25%) of the unclipped box's area (else ``mostly_outside_image``), and
   is at least :attr:`KeepRule.min_side_px` (4 px) on both sides (else ``too_small``). The share rule drops
   boxes that only graze the image edge. Both values are documented defaults, not tuned on any result.

A kept box is written with its world box, ``score``, the clipped ``box2d``, ``center_cam``, ``label``,
``class_name``, ``class_id``, ``track_id``, ``cls_score``, ``in_image_share``, the unclipped ``box2d_unclipped``,
and ``cover_by_nearer`` (:func:`measured_space.box_sources.cover_by_nearer` among the line's kept boxes).
Boxes are written highest score first.

**Known limitation: no occlusion test.** The detector fuses all cameras, and there is no depth map of the
receiving camera that is independent of the reference, so a box is kept in a view even where a nearer object
hides it. Such a box is an automatic detection the oracle does not have, because the oracle's presence needs the
ground truth to list the object as visible in that camera. Effects:

- **counts** include hidden objects of the class, so automatic counts can exceed the oracle's;
- **pairing**: the hidden object's projected box overlaps the occluder's visible box. The 2D association can
  pair it with the occluder when its IoU is higher, which shows as an ambiguous pairing, a wrong 3D box or a
  class confusion. An occluder that loses its pairing then looks missed;
- **2D boxes are projections** of whole 3D boxes, larger than the visible silhouettes the oracle's boxes are, so
  2D IoUs with partly hidden objects are low. Some true pairs fall below the association threshold and count
  as a miss plus an extra box.

How often projected boxes overlap is counted per line (``overlap``: pairs with any overlap, pairs at 2D IoU
>= 0.3, boxes at least half covered by one nearer box; :func:`measured_space.box_sources.overlap_summary`),
summed per scene in the sidecar, and recomputed at the operating threshold by the box source. These counts see
only occluders that are themselves detected boxes. Racks, walls and other static structure hide objects too, and
no count here measures that.

**Frames** (:func:`processed_frames`). The run's frame list is rebuilt from the meta as a uniform stride and
checked against ``ids_sha256``. A labelled frame the run did not process gets no line, and the box source must
then be run with ``--skip-missing``. A processed frame without boxes gets a line without boxes. If the list
cannot be rebuilt, a frame counts as processed only if it has a box.

**Thresholds and the top-k.** Each line's ``score_threshold`` is ``max(min_score, jsonl_thr)``, the lowest score
the line can hold: the observation floor of :mod:`measured_space.recall_control`. Sparse4D's decoder keeps its
top ``num_output_topk`` (300) boxes per *frame*, over all cameras, before the JSONL threshold. A frame with that
many rows is truncated, so lines carry ``frame_n_boxes`` and ``frame_at_topk``, and the sidecar counts such
frames. The recall-control cap (``--max-boxes``) is per view and does not describe this: use ``--max-boxes 0``
when no labelled frame is at the top-k.

**Output.** ``<out>/smartspaces/<scene with / as __>.jsonl``, one line per view, in the format
:func:`measured_space.box_sources.read_box_predictions` reads (also ``scripts/run_box_source.py --variant
external_named``, ``scripts/run_recall_control.py``, ``scripts/run_miss_audit.py --predictions``). Next to it,
``<scene with / as __>.meta.json`` records the source folder and file (SHA-256), Sparse4D's meta verbatim, the
checkpoint tag, the minimum score, the keep rule, the class map, the frame check and the drop and overlap
counts. ``<out>/convert.json`` records the run. ``seconds`` per line is the run's model time per frame divided
by its camera count: an amortised share, not a per-view measurement.

Command line (project env, CPU)::

    python -m measured_space.sparse4d convert --labels <label folder> --sparse4d <sparse4d>/<tag> \\
        --out <predictions folder> [--min-score 0.05] [--scenes Warehouse_000 ...] [--skip-missing]

Distances are in metres; conventions are those of :mod:`measured_space.geometry`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from measured_space.box_sources import clip_box2d, cover_by_nearer, overlap_summary
from measured_space.datasets.smartspaces import DATASET, box_from_annotation
from measured_space.geometry import Array, OrientedBox
from measured_space.pipeline import box_json, projected_extent, to_json_line, view_from_record
from measured_space.scene import FrameView

MODEL = "tao_sparse4d"
"""str: the model name written into every prediction line."""

CLASS_LABELS: dict[int, str] = {0: "Person", 1: "Forklift", 2: "NovaCarter", 3: "Transporter",
                                4: "FourierGR1T2", 5: "AgilityDigit", 6: "PalletTruck"}
"""dict of int to str: the 2026 challenge's class id -> the dataset's object type (the ground truth's strings)."""

DEFAULT_MIN_SCORE = 0.05
"""float: default lowest score converted, the runs' ``jsonl_thr`` (recall control sweeps thresholds above it)."""

BELOW_MIN_SCORE = "below_min_score"
BEHIND_CAMERA = "behind_camera"
NEAR_PLANE = "near_camera_plane"
OUTSIDE_IMAGE = "outside_image"
MOSTLY_OUTSIDE = "mostly_outside_image"
TOO_SMALL = "too_small"
DROP_REASONS: tuple[str, ...] = (BELOW_MIN_SCORE, BEHIND_CAMERA, NEAR_PLANE, OUTSIDE_IMAGE, MOSTLY_OUTSIDE, TOO_SMALL)
"""tuple of str: why a scene box at a view's frame is not written for the view (see the module docstring)."""

RUN_RECORD = "convert.json"
"""str: file name of the run record in the output folder."""

_ROW_KEYS = ("scene", "frame", "class_id", "class_name", "score", "center", "size_wlh", "yaw")


# ------------------------------------------------------------------------------------------------ boxes


@dataclass(frozen=True)
class KeepRule:
    """Which projected boxes a view keeps.

    Parameters
    ----------
    min_in_image_share : float, default 0.25
        Smallest share of the unclipped projected box's area that must lie inside the image, in (0, 1].
    min_side_px : float, default 4.0
        Smallest width and height of the clipped box, pixels.
    min_z : float, default 0.05
        A corner nearer than this to the camera plane makes the projection undefined, metres
        (:func:`measured_space.pipeline.projected_extent`).
    """

    min_in_image_share: float = 0.25
    min_side_px: float = 4.0
    min_z: float = 0.05

    def __post_init__(self) -> None:
        """Reject settings outside their ranges.

        Raises
        ------
        ValueError
            If ``min_in_image_share`` is not in (0, 1], ``min_side_px`` is negative or ``min_z`` not positive.
        """
        if not 0.0 < self.min_in_image_share <= 1.0:
            raise ValueError(f"min_in_image_share must be in (0, 1], got {self.min_in_image_share}")
        if not self.min_side_px >= 0.0:
            raise ValueError(f"min_side_px must be >= 0, got {self.min_side_px}")
        if not self.min_z > 0.0:
            raise ValueError(f"min_z must be > 0, got {self.min_z}")

    def to_json(self) -> dict[str, float]:
        """The rule as a JSON-friendly dict.

        Returns
        -------
        dict of str to float
            The three settings.
        """
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class Sparse4DBox:
    """One Sparse4D box of one frame, as read from a scene's JSONL.

    Parameters
    ----------
    row : int
        Line number in the source file, from 1 (to trace the box back).
    frame : int
        Frame index.
    class_id : int
        The challenge's class id.
    class_name : str
        Sparse4D's class name.
    label : str or None
        The dataset's object type (:func:`dataset_label`); ``None`` for an unknown class id.
    score : float
        Detection score.
    cls_score : float or None
        The class score before the centerness weighting.
    track_id : int or None
        The detector's instance-bank id.
    box : OrientedBox
        World box, metres (:func:`box_from_wlh_yaw`).
    """

    row: int
    frame: int
    class_id: int
    class_name: str
    label: str | None
    score: float
    cls_score: float | None
    track_id: int | None
    box: OrientedBox


def box_from_wlh_yaw(center: ArrayLike, size_wlh: ArrayLike, yaw: float) -> OrientedBox:
    """A Sparse4D box as the oracle builds a ground-truth box.

    Parameters
    ----------
    center : array_like, shape (3,)
        Box centre in the dataset's world frame, metres.
    size_wlh : array_like, shape (3,)
        Full side lengths along the box's local x, y, z (width, length, height), metres.
    yaw : float
        Rotation about world +z, radians.

    Returns
    -------
    OrientedBox
        :func:`measured_space.datasets.smartspaces.box_from_annotation` of ``(center, size_wlh, (0, 0, yaw))``.
    """
    return box_from_annotation(np.asarray(center, dtype=np.float64), np.asarray(size_wlh, dtype=np.float64),
                               (0.0, 0.0, float(yaw)))


def dataset_label(class_id: int, class_name: str) -> str | None:
    """The dataset's object type of a Sparse4D class.

    Parameters
    ----------
    class_id : int
        The challenge's class id.
    class_name : str
        Sparse4D's class name, which must agree with the id.

    Returns
    -------
    str or None
        ``CLASS_LABELS[class_id]``, or ``None`` for an id outside :data:`CLASS_LABELS`.

    Raises
    ------
    ValueError
        If the id is known and the name is not its object type.
    """
    label = CLASS_LABELS.get(int(class_id))
    if label is not None and class_name != label:
        raise ValueError(f"class id {class_id} is {label!r}, but the row names it {class_name!r}")
    return label


def _optional(value: Any, kind: type) -> Any:
    """A JSON value converted to ``kind``, or ``None``.

    Parameters
    ----------
    value : Any
        The value.
    kind : type
        ``int`` or ``float``.

    Returns
    -------
    int, float or None
        ``kind(value)``, or ``None`` if the value is missing.
    """
    return None if value is None else kind(value)


def parse_row(record: Mapping[str, Any], row: int) -> Sparse4DBox:
    """Parse one line of a Sparse4D scene file.

    Parameters
    ----------
    record : mapping
        The parsed JSON line.
    row : int
        Its line number, from 1.

    Returns
    -------
    Sparse4DBox
        The box.

    Raises
    ------
    ValueError
        If a required key is missing, a value is malformed or not finite, or the class name disagrees with
        the id.
    """
    missing = [k for k in _ROW_KEYS if k not in record]
    if missing:
        raise ValueError(f"row {row}: missing {', '.join(missing)}")
    center = np.asarray(record["center"], dtype=np.float64)
    size = np.asarray(record["size_wlh"], dtype=np.float64)
    numbers = [float(record["score"]), float(record["yaw"])]
    if center.shape != (3,) or size.shape != (3,) or not (np.all(np.isfinite(center)) and np.all(np.isfinite(size))
                                                          and all(math.isfinite(v) for v in numbers)):
        raise ValueError(f"row {row}: malformed or non-finite center, size_wlh, score or yaw")
    class_name = str(record["class_name"])
    return Sparse4DBox(row, int(record["frame"]), int(record["class_id"]), class_name,
                       dataset_label(int(record["class_id"]), class_name), numbers[0],
                       _optional(record.get("cls_score"), float), _optional(record.get("track_id"), int),
                       box_from_wlh_yaw(center, size, numbers[1]))


@dataclass(frozen=True)
class SceneBoxes:
    """Every box of one Sparse4D scene file, by frame.

    Parameters
    ----------
    scene : str or None
        The scene's folder name as the rows state it (e.g. ``"Warehouse_017"``); ``None`` for a file without rows.
    by_frame : dict of int to tuple of Sparse4DBox
        Frame index -> its boxes, in file order, for the frames that were read (all, or the requested ones).
    rows_per_frame : dict of int to int
        Frame index -> rows in the file, for every frame.
    n_rows : int
        Rows in the file.
    """

    scene: str | None
    by_frame: dict[int, tuple[Sparse4DBox, ...]]
    rows_per_frame: dict[int, int]
    n_rows: int

    def frame_rows(self, frame: int) -> int:
        """How many boxes the file holds for a frame.

        Parameters
        ----------
        frame : int
            Frame index.

        Returns
        -------
        int
            The count (0 for a frame without boxes).
        """
        return self.rows_per_frame.get(int(frame), 0)


def read_scene(path: str | Path, frames: Collection[int] | None = None) -> SceneBoxes:
    """Read one Sparse4D scene file.

    Parameters
    ----------
    path : str or pathlib.Path
        ``<sparse4d>/<scene>.jsonl``.
    frames : collection of int, optional
        Build boxes for these frames only (rows of other frames are counted and their scene checked, which keeps
        a long run's memory small); ``None`` builds every box.

    Returns
    -------
    SceneBoxes
        Its boxes by frame (none for an empty file, which a run without detections writes).

    Raises
    ------
    ValueError
        If a row lacks a key, a row of a read frame does not parse (:func:`parse_row`), or the rows name more
        than one scene.
    """
    wanted = None if frames is None else {int(f) for f in frames}
    scenes: set[str] = set()
    by_frame: dict[int, list[Sparse4DBox]] = {}
    counts: Counter = Counter()
    with Path(path).open() as f:
        for row, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [k for k in _ROW_KEYS if k not in record]
            if missing:
                raise ValueError(f"{path}: row {row}: missing {', '.join(missing)}")
            frame = int(record["frame"])
            scenes.add(str(record["scene"]))
            counts[frame] += 1
            if wanted is None or frame in wanted:
                by_frame.setdefault(frame, []).append(parse_row(record, row))
    if len(scenes) > 1:
        raise ValueError(f"{path}: rows name {len(scenes)} scenes ({sorted(scenes)[:3]}), expected one")
    return SceneBoxes(scenes.pop() if scenes else None, {k: tuple(v) for k, v in sorted(by_frame.items())},
                      dict(sorted(counts.items())), int(sum(counts.values())))


def processed_frames(meta: Mapping[str, Any]) -> frozenset[int] | None:
    """The frames a Sparse4D run processed, rebuilt from its meta and checked against its checksum.

    Parameters
    ----------
    meta : mapping
        The run's ``<scene>.meta.json``, with ``frames``: ``count``, ``first``, ``last`` and ``ids_sha256``
        (SHA-256 of the frame indices joined by commas).

    Returns
    -------
    frozenset of int or None
        The frames, if a uniform stride from ``first`` to ``last`` with ``count`` frames reproduces
        ``ids_sha256``; ``None`` if the meta lacks these fields or the stride does not reproduce the checksum.
    """
    info = meta.get("frames") or {}
    try:
        count, first, last, digest = int(info["count"]), int(info["first"]), int(info["last"]), str(info["ids_sha256"])
    except (KeyError, TypeError, ValueError):
        return None
    if count < 1:
        return None
    if count == 1:
        ids = [first]
    else:
        stride, rest = divmod(last - first, count - 1)
        if rest or stride <= 0:
            return None
        ids = list(range(first, last + 1, stride))
    if hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest() != digest:
        return None
    return frozenset(ids)


def amortised_seconds(meta: Mapping[str, Any]) -> float | None:
    """The run's model time per frame divided by its camera count: a view's amortised share.

    Parameters
    ----------
    meta : mapping
        The run's meta (``timing.model_ms_per_frame``, ``num_cameras``).

    Returns
    -------
    float or None
        Seconds per view, or ``None`` if the meta lacks a positive, finite value of either.
    """
    try:
        ms = float((meta.get("timing") or {})["model_ms_per_frame"])
        cameras = int(meta["num_cameras"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(ms) and ms > 0 and cameras > 0):
        return None
    return ms / 1000.0 / cameras


# ------------------------------------------------------------------------------------------------- views


@dataclass(frozen=True)
class Projection:
    """A world box as one view sees it.

    Parameters
    ----------
    center_cam : numpy.ndarray, shape (3,)
        Box centre in the camera frame, metres.
    box2d : tuple of float
        Projected 2D box clipped to the image, ``(xmin, ymin, xmax, ymax)``, pixel edges at integers.
    box2d_unclipped : tuple of float
        The same before clipping.
    in_image_share : float
        Clipped area over unclipped area.
    """

    center_cam: Array
    box2d: tuple[float, float, float, float]
    box2d_unclipped: tuple[float, float, float, float]
    in_image_share: float


def project_box(view: FrameView, box: OrientedBox, rule: KeepRule | None = None
                ) -> tuple[Projection | None, str | None]:
    """Project a world box into a view and apply the keep rule.

    Parameters
    ----------
    view : FrameView
        The view.
    box : OrientedBox
        World box.
    rule : KeepRule, optional
        Defaults to ``KeepRule()``.

    Returns
    -------
    projection : Projection or None
        The projection if the box is kept, else ``None``.
    reason : str or None
        ``None`` if kept, else one of :data:`DROP_REASONS` (not ``below_min_score``).
    """
    rule = rule or KeepRule()
    T_cw = view.T_cw
    center_cam = T_cw[:3, :3] @ box.center + T_cw[:3, 3]
    if not center_cam[2] > 0.0:
        return None, BEHIND_CAMERA
    extent = projected_extent(view, box, min_z=rule.min_z)
    if extent is None:
        return None, NEAR_PLANE
    x0, y0, x1, y1 = (v + 0.5 for v in extent)
    unclipped = (x0, y0, x1, y1)
    clipped = clip_box2d(unclipped, view.camera.width, view.camera.height)
    width, height = clipped[2] - clipped[0], clipped[3] - clipped[1]
    full = (x1 - x0) * (y1 - y0)
    if not (width > 0.0 and height > 0.0 and full > 0.0):
        return None, OUTSIDE_IMAGE
    share = width * height / full
    if share < rule.min_in_image_share:
        return None, MOSTLY_OUTSIDE
    if width < rule.min_side_px or height < rule.min_side_px:
        return None, TOO_SMALL
    return Projection(center_cam, clipped, unclipped, float(share)), None


def convert_view(view_record: Mapping[str, Any], boxes: Sequence[Sparse4DBox], *, min_score: float,
                 rule: KeepRule | None = None, provenance: Mapping[str, Any] | None = None
                 ) -> tuple[dict[str, Any], Counter]:
    """One view's prediction line from the scene's boxes at the view's frame.

    Parameters
    ----------
    view_record : mapping
        The shard's ``"view"`` record (identity, ``camera``, ``T_wc``, ``up``, ``image_ref``).
    boxes : sequence of Sparse4DBox
        The scene's boxes at the view's frame.
    min_score : float
        Boxes scoring below it are dropped (``below_min_score``).
    rule : KeepRule, optional
        Defaults to ``KeepRule()``.
    provenance : mapping, optional
        Extra line fields (``model``, ``checkpoint_sha256``, ``score_threshold``, ``seconds``, ...); they
        override the defaults.

    Returns
    -------
    line : dict
        The prediction line (module docstring): identity, ``model``, ``checkpoint_sha256``,
        ``score_threshold``, ``image_size``, ``seconds``, the provenance fields, ``n_dropped`` (by reason),
        ``overlap`` (:func:`measured_space.box_sources.overlap_summary`) and ``boxes``, highest score first.
    counts : collections.Counter
        ``n_in`` (boxes at the frame), ``n_kept`` and one count per drop reason.
    """
    rule = rule or KeepRule()
    view = view_from_record(view_record)
    counts: Counter = Counter({"n_in": len(boxes)})
    kept: list[tuple[Sparse4DBox, Projection]] = []
    for b in boxes:
        if b.score < min_score:
            counts[BELOW_MIN_SCORE] += 1
            continue
        projection, reason = project_box(view, b.box, rule)
        if projection is None:
            counts[str(reason)] += 1
            continue
        kept.append((b, projection))
    kept.sort(key=lambda bp: (-bp[0].score, bp[0].row))
    counts["n_kept"] = len(kept)
    boxes2d = [p.box2d for _, p in kept]
    depths = [float(p.center_cam[2]) for _, p in kept]
    cover = cover_by_nearer(boxes2d, depths)
    out_boxes = [{**box_json(b.box), "score": b.score, "box2d": list(p.box2d), "center_cam": p.center_cam.tolist(),
                  "label": b.label, "class_name": b.class_name, "class_id": b.class_id, "track_id": b.track_id,
                  "cls_score": b.cls_score, "in_image_share": p.in_image_share,
                  "box2d_unclipped": list(p.box2d_unclipped), "cover_by_nearer": float(cover[k]), "row": b.row}
                 for k, (b, p) in enumerate(kept)]
    line: dict[str, Any] = {k: view_record[k] for k in ("view_key", "dataset", "scene", "camera_id", "frame")}
    line.update(model=MODEL, checkpoint_sha256=None, score_threshold=float(min_score),
                image_size=[int(view.camera.width), int(view.camera.height)], seconds=None)
    line.update(dict(provenance or {}))
    line.update(n_dropped={r: int(counts[r]) for r in DROP_REASONS}, overlap=overlap_summary(boxes2d, depths),
                boxes=out_boxes)
    return line, counts


# ------------------------------------------------------------------------------------------------- folders


def refuse_output(out: Path, *inputs: Path) -> None:
    """Refuse an output folder that holds anything or overlaps an input folder.

    Parameters
    ----------
    out : pathlib.Path
        The output folder.
    *inputs : pathlib.Path
        Input folders, which must neither contain nor lie inside ``out``.

    Raises
    ------
    FileExistsError
        If ``out`` exists and is not an empty folder.
    ValueError
        If ``out`` is, contains or lies inside an input folder.
    """
    out = Path(out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"{out} exists and is not an empty folder; choose a new --out")
    for folder in (Path(f).resolve() for f in inputs):
        if out == folder or folder in out.parents or out in folder.parents:
            raise ValueError(f"--out {out} overlaps the input folder {folder}")


def sha256(path: Path) -> str:
    """SHA-256 of a file.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_name_and_split(scene: str) -> tuple[str, str | None]:
    """A label scene's folder name and split.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder, e.g. ``"MTMC_Tracking_2026/train/Warehouse_017"``.

    Returns
    -------
    tuple of (str, str or None)
        ``("Warehouse_017", "train")``; the split is ``None`` for a path with one part.
    """
    parts = scene.split("/")
    return parts[-1], (parts[-2] if len(parts) > 1 else None)


def read_view_records(labels: Path, scenes: Sequence[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """The ``"view"`` records of a label folder's Smart Spaces shards, by scene.

    Parameters
    ----------
    labels : pathlib.Path
        Label folder of ``scripts/run_labels.py`` (``<labels>/smartspaces/shards/*.json``; read only).
    scenes : sequence of str, optional
        Keep only scenes whose path contains one of these strings.

    Returns
    -------
    dict of str to list of dict
        Scene path -> its views' records, in shard-name order.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((Path(labels) / DATASET / "shards").glob("*.json")):
        view = json.loads(path.read_text())["records"]["view"]
        if not scenes or any(s in view["scene"] for s in scenes):
            out.setdefault(str(view["scene"]), []).append(view)
    return out


def convert_scene(views: Sequence[Mapping[str, Any]], scene_boxes: SceneBoxes, meta: Mapping[str, Any], *,
                  min_score: float, rule: KeepRule) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prediction lines for one scene's labelled views.

    Parameters
    ----------
    views : sequence of mapping
        The scene's ``"view"`` records.
    scene_boxes : SceneBoxes
        The scene's Sparse4D boxes.
    meta : mapping
        The run's meta (checkpoint, ``jsonl_thr``, ``num_output_topk``, frames, timing).
    min_score : float
        Lowest score converted.
    rule : KeepRule
        The keep rule.

    Returns
    -------
    lines : list of dict
        One line per view whose frame the run processed (:func:`convert_view`).
    summary : dict
        ``n_views``, ``n_lines``, ``views_frame_not_run`` (view keys), ``frames_check`` (``"checksum"`` or
        ``"rows_only"``), ``score_threshold``, ``frames_at_topk`` (labelled frames with ``num_output_topk``
        rows), ``counts`` (summed :func:`convert_view` counts), ``overlap`` (summed per-line overlap counts) and
        ``label_counts`` (kept boxes per label).
    """
    processed = processed_frames(meta)
    threshold = max(float(min_score), float(meta.get("jsonl_thr") or 0.0))
    topk = meta.get("num_output_topk")
    provenance = {"model": MODEL, "checkpoint_sha256": meta.get("checkpoint_sha256"),
                  "checkpoint_tag": meta.get("tag"), "score_threshold": threshold,
                  "seconds": amortised_seconds(meta)}
    lines: list[dict[str, Any]] = []
    not_run: list[str] = []
    at_topk: set[int] = set()
    counts: Counter = Counter()
    overlap: Counter = Counter()
    labels: Counter = Counter()
    for view in views:
        frame = int(view["frame"])
        ran = frame in processed if processed is not None else scene_boxes.frame_rows(frame) > 0
        if not ran:
            not_run.append(str(view["view_key"]))
            continue
        n_frame = scene_boxes.frame_rows(frame)
        frame_at_topk = topk is not None and n_frame >= int(topk)
        if frame_at_topk:
            at_topk.add(frame)
        line, view_counts = convert_view(view, scene_boxes.by_frame.get(frame, ()), min_score=threshold,
                                         rule=rule, provenance={**provenance, "sparse4d_frame": frame,
                                                                "frame_n_boxes": n_frame,
                                                                "frame_at_topk": bool(frame_at_topk)})
        lines.append(line)
        counts.update(view_counts)
        overlap.update(line["overlap"])
        labels.update(str(b["label"]) for b in line["boxes"])
    summary = {"n_views": len(views), "n_lines": len(lines), "views_frame_not_run": not_run,
               "frames_check": "checksum" if processed is not None else "rows_only", "score_threshold": threshold,
               "frames_at_topk": sorted(at_topk), "counts": {k: int(v) for k, v in sorted(counts.items())},
               "overlap": {k: int(v) for k, v in sorted(overlap.items())},
               "label_counts": dict(sorted(labels.items()))}
    return lines, summary


def convert(labels: Path, sparse4d: Path, out: Path, *, min_score: float = DEFAULT_MIN_SCORE,
            rule: KeepRule | None = None, scenes: Sequence[str] | None = None, skip_missing: bool = False,
            argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Convert a Sparse4D run folder into per-view predictions for a label folder's views.

    Every input is read and checked before anything is written: every row's keys and scene, and the labelled
    frames' rows in full (:func:`read_scene`). Only the labelled frames' boxes are kept in memory.

    Parameters
    ----------
    labels : pathlib.Path
        Label folder of ``scripts/run_labels.py`` (read only).
    sparse4d : pathlib.Path
        A Sparse4D run folder (one checkpoint tag): ``<scene>.jsonl`` and ``<scene>.meta.json`` (read only).
    out : pathlib.Path
        New or empty output folder, outside both inputs.
    min_score : float, default DEFAULT_MIN_SCORE
        Lowest score converted (raised to the run's ``jsonl_thr`` if that is higher).
    rule : KeepRule, optional
        Defaults to ``KeepRule()``.
    scenes : sequence of str, optional
        Only scenes whose path contains one of these strings.
    skip_missing : bool, default False
        Skip scenes without a Sparse4D file and meta instead of refusing.
    argv : sequence of str, optional
        The command line, for the run record.

    Returns
    -------
    dict
        The run record written to ``<out>/convert.json``.

    Raises
    ------
    FileExistsError, ValueError
        As :func:`refuse_output`.
    FileNotFoundError
        If no view is selected, or a scene lacks its Sparse4D file or meta and ``skip_missing`` is off.
    ValueError
        If ``min_score`` is not finite, a file's scene or split differs from the label scene's, its row count
        differs from its meta's ``rows_jsonl``, the scenes' metas name different checkpoints, or as
        :func:`read_scene`.
    """
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    rule = rule or KeepRule()
    labels, sparse4d, out = Path(labels).resolve(), Path(sparse4d).resolve(), Path(out).resolve()
    if not math.isfinite(min_score):
        raise ValueError(f"min_score must be finite, got {min_score}")
    refuse_output(out, labels, sparse4d)
    views = read_view_records(labels, scenes)
    if not views:
        raise FileNotFoundError(f"no {DATASET} views under {labels} (scenes {scenes})")

    # Read and check every input first.
    jobs: list[tuple[str, list[dict[str, Any]], Path, Path, dict[str, Any], SceneBoxes]] = []
    missing: list[str] = []
    for scene, scene_views in views.items():
        name, split = scene_name_and_split(scene)
        jsonl, meta_path = sparse4d / f"{name}.jsonl", sparse4d / f"{name}.meta.json"
        if not (jsonl.is_file() and meta_path.is_file()):
            if not skip_missing:
                raise FileNotFoundError(f"{scene}: no {jsonl.name} with its {meta_path.name} in {sparse4d} (a run "
                                        "writes the meta last; pass --skip-missing to skip the scene)")
            missing.append(scene)
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("scene") not in (None, name) or (split is not None and meta.get("split") not in (None, split)):
            raise ValueError(f"{meta_path} is for {meta.get('split')}/{meta.get('scene')}, not {scene}")
        boxes = read_scene(jsonl, frames={int(v["frame"]) for v in scene_views})
        if boxes.scene not in (None, name):
            raise ValueError(f"{jsonl} holds scene {boxes.scene!r}, not {name!r}")
        if meta.get("rows_jsonl") is not None and int(meta["rows_jsonl"]) != boxes.n_rows:
            raise ValueError(f"{jsonl} has {boxes.n_rows} rows, its meta says {meta['rows_jsonl']}: incomplete?")
        jobs.append((scene, scene_views, jsonl, meta_path, meta, boxes))
    if not jobs:
        raise FileNotFoundError(f"none of the {len(views)} selected scenes has Sparse4D output in {sparse4d}")
    checkpoints = {(job[4].get("tag"), job[4].get("checkpoint_sha256")) for job in jobs}
    if len(checkpoints) > 1:
        raise ValueError(f"{sparse4d} mixes checkpoints {sorted(map(str, checkpoints))}; convert one run at a time")

    folder = out / DATASET
    folder.mkdir(parents=True, exist_ok=True)
    record_scenes: dict[str, Any] = {}
    for scene, scene_views, jsonl, meta_path, meta, boxes in jobs:
        lines, summary = convert_scene(scene_views, boxes, meta, min_score=min_score, rule=rule)
        stem = scene.replace("/", "__")
        (folder / f"{stem}.jsonl").write_text("".join(to_json_line(line) + "\n" for line in lines))
        sidecar = {"converter": "measured_space.sparse4d", "scene": scene, "source_folder": str(sparse4d),
                   "source_file": jsonl.name, "source_sha256": sha256(jsonl), "meta_file": meta_path.name,
                   "meta_sha256": sha256(meta_path), "checkpoint_tag": meta.get("tag"),
                   "checkpoint_sha256": meta.get("checkpoint_sha256"), "min_score": float(min_score),
                   "keep_rule": rule.to_json(), "class_labels": {str(k): v for k, v in CLASS_LABELS.items()},
                   "n_rows": boxes.n_rows, **summary, "sparse4d_meta": meta}
        (folder / f"{stem}.meta.json").write_text(json.dumps(sidecar, indent=1))
        record_scenes[scene] = {k: sidecar[k] for k in ("source_file", "source_sha256", "meta_sha256",
                                                         "checkpoint_tag", "checkpoint_sha256", "n_rows", "n_views",
                                                         "n_lines", "frames_check", "score_threshold",
                                                         "frames_at_topk", "counts", "overlap", "label_counts")}
        record_scenes[scene]["n_views_frame_not_run"] = len(summary["views_frame_not_run"])
    record = {"argv": None if argv is None else list(argv), "started": started, "labels": str(labels),
              "sparse4d": str(sparse4d), "out": str(out), "min_score": float(min_score),
              "keep_rule": rule.to_json(), "scenes_filter": None if scenes is None else list(scenes),
              "scenes_missing": missing, "scenes": record_scenes,
              "n_lines": sum(s["n_lines"] for s in record_scenes.values()),
              "n_views_frame_not_run": sum(s["n_views_frame_not_run"] for s in record_scenes.values()),
              "seconds": round(time.perf_counter() - t0, 2), "finished": datetime.now(timezone.utc).isoformat()}
    (out / RUN_RECORD).write_text(json.dumps(record, indent=1))
    return record


# ------------------------------------------------------------------------------------------------------ CLI


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: ``python -m measured_space.sparse4d convert ...`` (project env, CPU).

    ``convert --labels <label folder> --sparse4d <run folder> --out <predictions folder> [--min-score 0.05]
    [--scenes ...] [--min-in-image-share 0.25] [--min-side-px 4] [--skip-missing]``: :func:`convert`.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments without the program name; ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        0 on success; 1 if the output folder is refused, an input is missing or inconsistent.
    """
    parser = argparse.ArgumentParser(prog="python -m measured_space.sparse4d",
                                     description="Sparse4D scene boxes -> per-view predictions of a label folder.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("convert", help="convert a Sparse4D run folder for a label folder's views")
    p.add_argument("--labels", type=Path, required=True, help="label folder of run_labels.py (read only)")
    p.add_argument("--sparse4d", type=Path, required=True,
                   help="Sparse4D run folder of one checkpoint tag: <scene>.jsonl and <scene>.meta.json (read only)")
    p.add_argument("--out", type=Path, required=True, help="new, empty predictions folder")
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="lowest score converted")
    p.add_argument("--scenes", nargs="*", default=None, help="only scenes whose path contains one of these")
    p.add_argument("--min-in-image-share", type=float, default=KeepRule.min_in_image_share,
                   help="keep a box if at least this share of its projected box lies inside the image")
    p.add_argument("--min-side-px", type=float, default=KeepRule.min_side_px,
                   help="keep a box if its clipped 2D box is at least this wide and tall (pixels)")
    p.add_argument("--skip-missing", action="store_true", help="skip scenes without Sparse4D output")
    args = parser.parse_args(argv)
    try:
        rule = KeepRule(min_in_image_share=args.min_in_image_share, min_side_px=args.min_side_px)
        record = convert(args.labels, args.sparse4d, args.out, min_score=args.min_score, rule=rule,
                         scenes=args.scenes, skip_missing=args.skip_missing,
                         argv=list(sys.argv[1:] if argv is None else argv))
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"refused: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps({k: record[k] for k in ("out", "n_lines", "n_views_frame_not_run", "scenes_missing", "seconds")}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
