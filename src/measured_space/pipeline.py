"""Paired oracle and automatic spatial labels for one view.

For each view the same questions are answered twice: from ground-truth boxes ("oracle") and from boxes that a
perception pipeline lifts from the image ("auto"), once per depth source. The steps are:

1. **Oracle visibility.** Each ground-truth object becomes an :class:`~measured_space.scene.ObjectInView`:

   - with a reference depth map (ARKitScenes: the Faro laser depth; Smart Spaces ``Warehouse_018``: the
     simulator depth), by :func:`~measured_space.visibility.object_in_view` against that depth;
   - for Smart Spaces scenes without depth, by the **ground-truth 2D box proxy** (:func:`gt_box_visibility`):
     geometry alone (:func:`~measured_space.visibility.object_in_view` with no depth) gives the truncated
     fraction; only objects the ground truth lists as visible in that camera are kept as present; and the
     visible fraction is the area of the ground truth's visible 2D box over the area of the 2D bounding box of
     the object's projected 3D box corners, clipped to [0, 1]. The proxy reads occlusion and truncation from
     the simulator's visible silhouette, but a box's projection is larger than the object's silhouette, so an
     unoccluded object scores below 1 (by how much depends on its shape and the viewing angle).

2. **Automatic boxes.** SAM 3 segments the dataset's prompt phrases once per image. For each depth source,
   :func:`~measured_space.perception.lift_instance` lifts every mask to a gravity-aligned world box with 17
   image-evidence features, and :func:`~measured_space.matching.match_detections` matches the boxes to the
   present oracle objects. Large masks may be lifted from a pixel subsample: a mask with at least
   ``PipelineConfig.lift_stride_min_pixels`` pixels is lifted from every ``s``-th pixel in each direction
   (``s = PipelineConfig.lift_stride``) with exactly rescaled intrinsics and an erosion radius of
   ``max(1, round(erode_px / s))`` pixels; smaller masks are lifted at full resolution, so the 50-point
   minimum never rejects a small object because of subsampling. The stride used is recorded per lift
   (:attr:`Lift.stride`). The five mask-only features (``mask_area_px``, ``n_components``,
   ``boundary_fraction``, ``border_touch``, ``n_border_sides``) are always computed on the full-resolution
   mask; ``n_points`` counts the points actually lifted (multiply by ``stride**2`` for a full-resolution
   equivalent); the other features are ratios or metric.

3. **Questions** come from the oracle view (:func:`~measured_space.qa.generate_specs`, fixed seed).

4. **Answers.** Every spec is answered from the oracle boxes (``source="oracle"``) and, per depth source, from
   the automatic boxes (``source="auto_<depth>"``). Automatic boxes are keyed by the id of the oracle object
   they matched, so a spec whose referent was not matched has no automatic answer; it is recorded as
   unanswered. Unmatched detections are keyed ``"auto:<index>"`` and take part only in counts. An automatic
   count is the number of that class's detections that have a box from that depth source (a detection whose
   lift failed is not an automatic label); the 2D detection count is stored next to it. Each automatic item's
   ``extra`` holds, per referent (per counted object for counts), the detection index, phrase, score, match
   distance and IoU, and the lift features; these are the inputs of the uncertainty model.

Depth sources: ARKitScenes ``laser`` (isolates perception error), ``lidar`` (iPad LiDAR resampled
nearest-neighbour to the RGB size) and ``moge``; Smart Spaces ``Warehouse_018`` ``gt_depth`` and ``moge``; the
Smart Spaces scenes without depth ``moge`` only. Images are upright for both datasets, so no rotation is
applied before SAM 3 or MoGe.

All distances are in metres; conventions are those of :mod:`measured_space.geometry`.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from measured_space.geometry import Camera, OrientedBox
from measured_space.matching import MatchConfig, MatchResult, match_detections
from measured_space.perception import (
    Instance,
    LiftError,
    labels_for_phrase,
    lift_instance,
    mask_features,
    prompt_phrases,
    resample_depth,
)
from measured_space.qa import QAConfig, generate_specs, make_items, phrase_for, visible_ids
from measured_space.qa_schema import QAItem, QuestionSpec
from measured_space.scene import FrameView, ObjectInView, SceneObject
from measured_space.visibility import object_in_view

GT_BOX2D = "gt_box2d"
"""str: the visibility method that uses the ground truth's visible 2D boxes (no depth)."""

DEPTH_SOURCES: dict[str, tuple[str, ...]] = {
    "arkitscenes": ("laser", "lidar", "moge"),
    "smartspaces_depth": ("gt_depth", "moge"),
    "smartspaces_no_depth": ("moge",),
}
"""dict of str to tuple of str: depth sources per scene kind."""

MASK_FEATURE_KEYS: tuple[str, ...] = ("mask_area_px", "n_components", "boundary_fraction", "border_touch",
                                      "n_border_sides")
"""tuple of str: lift features that depend on the mask only (always taken at full resolution)."""

AUTO_KEY_PREFIX = "auto:"
"""str: prefix of the key of an unmatched detection's automatic box (``"auto:<detection index>"``)."""

Box2D = tuple[int, int, int, int]
SegmentFn = Callable[..., list[Instance]]
DepthFn = Callable[[NDArray[np.uint8], Camera], NDArray[np.floating]]


@dataclass(frozen=True)
class PipelineConfig:
    """Every setting of the label pipeline.

    Parameters
    ----------
    qa : QAConfig, default QAConfig()
        Question-generation thresholds.
    qa_seed : int, default 0
        Seed of the per-type question cap.
    match : MatchConfig, default MatchConfig()
        Distance gate for matching.
    score_threshold : float, default 0.5
        Minimum SAM 3 detection score.
    dedupe_iou : float or None, default 0.7
        Mask-IoU threshold for cross-phrase duplicate suppression; ``None`` keeps every detection.
    lift_stride : int, default 1
        Pixel stride for lifting large masks (1 = every mask pixel); see the module docstring.
    lift_stride_min_pixels : int, default 0
        Masks with at least this many pixels are lifted with ``lift_stride``; smaller ones at full resolution.
    erode_px : int, default 3
        Mask erosion radius for lifting, full-resolution pixels.
    boundary_px : float, default 5.0
        Edge band width for the ``boundary_fraction`` feature, full-resolution pixels.
    visibility_spacing : float, default 0.05
        Surface-sample spacing of the oracle visibility test, metres.
    """

    qa: QAConfig = field(default_factory=QAConfig)
    qa_seed: int = 0
    match: MatchConfig = field(default_factory=MatchConfig)
    score_threshold: float = 0.5
    dedupe_iou: float | None = 0.7
    lift_stride: int = 1
    lift_stride_min_pixels: int = 0
    erode_px: int = 3
    boundary_px: float = 5.0
    visibility_spacing: float = 0.05

    def __post_init__(self) -> None:
        """Validate the stride.

        Raises
        ------
        ValueError
            If ``lift_stride`` is below 1.
        """
        if self.lift_stride < 1:
            raise ValueError("lift_stride must be at least 1")

    def to_json(self) -> dict[str, Any]:
        """The settings as a JSON-serialisable dict.

        Returns
        -------
        dict
            Every field, nested configs included.
        """
        return asdict(self)


@dataclass(frozen=True)
class Detection:
    """One SAM 3 detection with what matching and answering need about it.

    Parameters
    ----------
    index : int
        Position in the view's detection list (highest score first).
    instance : Instance
        The detection: phrase, full-resolution mask, 2D box and score.
    labels : tuple of str
        Dataset labels its phrase may stand for.
    qa_label : str or None
        A label whose question phrase (:func:`measured_space.qa.phrase_for`) names this detection in count
        questions; the first of ``labels``, or ``None`` if the phrase maps to no label.
    mask_stats : dict of str to float
        :func:`measured_space.perception.mask_features` of the full-resolution mask.
    """

    index: int
    instance: Instance
    labels: tuple[str, ...]
    qa_label: str | None
    mask_stats: dict[str, float]


@dataclass(frozen=True)
class Lift:
    """One detection lifted with one depth source.

    Parameters
    ----------
    box : OrientedBox or None
        World box, or ``None`` if lifting failed.
    features : dict of str to float
        The lift features (empty on failure).
    error : str or None, default None
        Why lifting failed, else ``None``.
    stride : int, default 1
        Pixel stride the mask was lifted with.
    """

    box: OrientedBox | None
    features: dict[str, float]
    error: str | None = None
    stride: int = 1


@dataclass
class ViewResult:
    """Everything the pipeline produced for one view.

    Parameters
    ----------
    view : FrameView
        The view.
    visibility : str
        How oracle visibility was measured: a depth source name or :data:`GT_BOX2D`.
    sources : tuple of str
        Depth sources used for automatic boxes.
    oracle : list of ObjectInView
        Every ground-truth object of the view's scene and time, as the view sees it.
    oracle_diagnostics : dict of str to dict
        Object id -> visibility diagnostics.
    detections : list of Detection
        SAM 3 detections.
    lifts : dict of str to list of Lift
        Depth source -> one lift per detection.
    matches : dict of str to MatchResult
        Depth source -> matching of the lifted detections to the present oracle objects.
    specs : list of QuestionSpec
        Questions asked about the view.
    items : dict of str to list of QAItem
        ``"oracle"`` and ``"auto_<depth>"`` -> answered items.
    unanswered : dict of str to list of dict
        ``"auto_<depth>"`` -> specs without an automatic answer (``id``, ``qtype``, ``missing`` referent ids).
    drops : dict of str to int
        ``"<qtype>/<reason>"`` -> how many candidates question generation left out.
    timings : dict of str to float
        Stage -> seconds.
    config : PipelineConfig
        The settings used.
    """

    view: FrameView
    visibility: str
    sources: tuple[str, ...]
    oracle: list[ObjectInView]
    oracle_diagnostics: dict[str, dict[str, Any]]
    detections: list[Detection]
    lifts: dict[str, list[Lift]]
    matches: dict[str, MatchResult]
    specs: list[QuestionSpec]
    items: dict[str, list[QAItem]]
    unanswered: dict[str, list[dict[str, Any]]]
    drops: dict[str, int]
    timings: dict[str, float]
    config: PipelineConfig

    @property
    def key(self) -> str:
        """str: the view's identity, ``"<dataset>/<scene>/<camera_id>/<frame>"``."""
        return view_key(self.view)

    def records(self) -> dict[str, Any]:
        """All output records of the view, JSON-serialisable.

        Returns
        -------
        dict
            ``"view"``: one dict; ``"oracle_objects"``: one dict per ground-truth object;
            ``"detections"``: one dict per detection and depth source; ``"qa"``: source -> list of
            :meth:`QAItem.to_json` records.
        """
        identity = {"view_key": self.key, "dataset": self.view.dataset, "scene": self.view.scene,
                    "camera_id": self.view.camera_id, "frame": self.view.frame}
        return {"view": self._view_record(identity), "oracle_objects": self._oracle_records(identity),
                "detections": self._detection_records(identity),
                "qa": {source: [item.to_json() for item in items] for source, items in self.items.items()}}

    def _view_record(self, identity: dict[str, str]) -> dict[str, Any]:
        """The per-view record.

        Parameters
        ----------
        identity : dict of str to str
            The view's identity fields.

        Returns
        -------
        dict
            Identity, image, camera, ``T_wc``, up, methods, counts, timings, unanswered specs and drops.
        """
        cam = self.view.camera
        present = visible_ids(self.oracle, self.config.qa)
        return {
            **identity, "image_ref": self.view.image_ref,
            "camera": {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy, "width": cam.width,
                       "height": cam.height},
            "T_wc": self.view.T_wc.tolist(), "up": self.view.up.tolist(), "visibility": self.visibility,
            "depth_sources": list(self.sources), "n_oracle_objects": len(self.oracle), "n_present": len(present),
            "n_detections": len(self.detections), "n_specs": len(self.specs),
            "n_items": {source: len(items) for source, items in self.items.items()},
            "auto_unanswered": self.unanswered, "qa_drops": self.drops,
            "timings": self.timings, "seconds": float(sum(self.timings.values())),
            "config": self.config.to_json(),
        }

    def _oracle_records(self, identity: dict[str, str]) -> list[dict[str, Any]]:
        """One record per ground-truth object.

        Parameters
        ----------
        identity : dict of str to str
            The view's identity fields.

        Returns
        -------
        list of dict
            Box, visibility, truncation, 2D box, presence and per-source match of each object.
        """
        present = visible_ids(self.oracle, self.config.qa)
        by_object = {source: result.by_object() for source, result in self.matches.items()}
        records = []
        for seen in self.oracle:
            oid = seen.obj.object_id
            match: dict[str, Any] = {}
            for source in self.sources:
                m = by_object[source].get(oid)
                match[source] = None if m is None else {"detection": m.detection, "distance": m.distance,
                                                        "iou": m.iou}
            records.append({
                **identity, "object_id": oid, "label": seen.obj.label, "box": box_json(seen.obj.box),
                "visible_fraction": seen.visible_fraction, "truncated_fraction": seen.truncated_fraction,
                "bbox2d": None if seen.bbox2d is None else list(seen.bbox2d), "present": oid in present,
                "clearly_visible": clearly_visible(seen, self.config.qa), "visibility": self.visibility,
                "diagnostics": self.oracle_diagnostics.get(oid, {}),
                "match": match if oid in present else None,
            })
        return records

    def _detection_records(self, identity: dict[str, str]) -> list[dict[str, Any]]:
        """One record per detection and depth source.

        Parameters
        ----------
        identity : dict of str to str
            The view's identity fields.

        Returns
        -------
        list of dict
            Phrase, labels, score, 2D box, mask statistics, world box, features and match of each lift.
        """
        oracle_labels = {o.obj.object_id: o.obj.label for o in self.oracle}
        records = []
        for source in self.sources:
            matches = self.matches[source].by_detection()
            for det, lift in zip(self.detections, self.lifts[source]):
                m = matches.get(det.index)
                if lift.box is None:
                    status = "unboxed"
                else:
                    status = "matched" if m is not None else "false_positive"
                records.append({
                    **identity, "detection": det.index, "source": source, "phrase": det.instance.phrase,
                    "labels": list(det.labels), "score": det.instance.score, "box2d": list(det.instance.box2d),
                    "mask": det.mask_stats, "lift_stride": lift.stride,
                    "lift_error": lift.error, "box": None if lift.box is None else box_json(lift.box),
                    "features": lift.features, "status": status,
                    "match": None if m is None else {"object_id": m.object_id, "label": oracle_labels[m.object_id],
                                                     "distance": m.distance, "iou": m.iou, "gate": m.gate},
                })
        return records


# --------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------


def view_key(view: FrameView) -> str:
    """A view's identity as one string.

    Parameters
    ----------
    view : FrameView
        The view.

    Returns
    -------
    str
        ``"<dataset>/<scene>/<camera_id>/<frame>"``.
    """
    return f"{view.dataset}/{view.scene}/{view.camera_id}/{view.frame}"


def box_json(box: OrientedBox) -> dict[str, list[Any]]:
    """A box as JSON-friendly lists.

    Parameters
    ----------
    box : OrientedBox
        The box.

    Returns
    -------
    dict
        ``center`` (3,), ``size`` (3,) full side lengths in metres, ``rotation`` (3, 3) with box axes as columns.
    """
    return {"center": box.center.tolist(), "size": box.size.tolist(), "rotation": box.rotation.tolist()}


def clearly_visible(seen: ObjectInView, config: QAConfig) -> bool:
    """Whether an object passes the question generator's referent visibility thresholds.

    Parameters
    ----------
    seen : ObjectInView
        The object as a view sees it.
    config : QAConfig
        Supplies ``min_visible_fraction`` and ``max_truncated_fraction``.

    Returns
    -------
    bool
        True if it has a 2D box, enough visible surface and little enough truncation.
    """
    return (seen.bbox2d is not None and seen.visible_fraction >= config.min_visible_fraction
            and seen.truncated_fraction <= config.max_truncated_fraction)


def scene_kind(view: FrameView, has_depth: bool) -> str:
    """The key of :data:`DEPTH_SOURCES` for a view.

    Parameters
    ----------
    view : FrameView
        The view.
    has_depth : bool
        Whether a Smart Spaces scene has simulator depth (ignored for ARKitScenes).

    Returns
    -------
    str
        ``"arkitscenes"``, ``"smartspaces_depth"`` or ``"smartspaces_no_depth"``.

    Raises
    ------
    ValueError
        For another dataset.
    """
    if view.dataset == "arkitscenes":
        return "arkitscenes"
    if view.dataset == "smartspaces":
        return "smartspaces_depth" if has_depth else "smartspaces_no_depth"
    raise ValueError(f"no depth sources defined for dataset {view.dataset!r}")


# --------------------------------------------------------------------------------------------------------
# Oracle visibility
# --------------------------------------------------------------------------------------------------------


def projected_extent(view: FrameView, box: OrientedBox, min_z: float = 0.05
                     ) -> tuple[float, float, float, float] | None:
    """2D bounding box of a 3D box's projected corners, not clipped to the image.

    Parameters
    ----------
    view : FrameView
        The view.
    box : OrientedBox
        World-frame box.
    min_z : float, default 0.05
        Corners nearer than this to the camera plane make the extent undefined, metres.

    Returns
    -------
    tuple of float or None
        ``(xmin, ymin, xmax, ymax)`` in pixels (centres at integers), or ``None`` if a corner is at or behind
        ``min_z``.
    """
    uv, z = view.camera.project(box.transformed(view.T_cw).corners())
    if np.any(z <= min_z):
        return None
    return float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())


def gt_box_visibility(view: FrameView, obj: SceneObject, gt_box: Box2D | None, *, spacing: float = 0.05
                      ) -> tuple[ObjectInView, dict[str, Any]]:
    """Visibility from the ground truth's visible 2D box, for views without depth (the proxy).

    Parameters
    ----------
    view : FrameView
        The view.
    obj : SceneObject
        The ground-truth object.
    gt_box : tuple of int or None
        The ground truth's visible box for this camera, ``(xmin, ymin, xmax, ymax)`` inclusive pixel indices,
        or ``None`` if the ground truth does not list the object as visible in this camera.
    spacing : float, default 0.05
        Surface-sample spacing of the geometry-only test, metres.

    Returns
    -------
    seen : ObjectInView
        Not visible (fraction 0, no 2D box) without ``gt_box``. Otherwise visible fraction = ground-truth box
        area / area of :func:`projected_extent`, clipped to [0, 1] (the geometry-only fraction if the extent is
        undefined); the truncated fraction from geometry; ``bbox2d`` the ground-truth box.
    diagnostics : dict
        ``gt_visible``, ``gt_box2d``, ``gt_area_px``, ``projected_area_px``, ``geometry_visible_fraction``,
        ``proxy_fallback`` (True if the geometry-only fraction was used), ``n_samples``.
    """
    geometry, diag = object_in_view(view, obj, None, spacing=spacing)
    info: dict[str, Any] = {"gt_visible": gt_box is not None, "gt_box2d": None if gt_box is None else list(gt_box),
                            "gt_area_px": None, "projected_area_px": None,
                            "geometry_visible_fraction": geometry.visible_fraction, "proxy_fallback": False,
                            "n_samples": diag["n_samples"]}
    if gt_box is None:
        return ObjectInView(obj, 0.0, geometry.truncated_fraction, None), info
    x0, y0, x1, y1 = gt_box
    gt_area = float((x1 - x0 + 1) * (y1 - y0 + 1))
    extent = projected_extent(view, obj.box)
    info["gt_area_px"] = gt_area
    if extent is None or (extent[2] - extent[0]) * (extent[3] - extent[1]) <= 0:
        fraction = geometry.visible_fraction
        info["proxy_fallback"] = True
    else:
        projected_area = (extent[2] - extent[0]) * (extent[3] - extent[1])
        info["projected_area_px"] = projected_area
        fraction = float(np.clip(gt_area / projected_area, 0.0, 1.0))
    return ObjectInView(obj, fraction, geometry.truncated_fraction,
                        (float(x0), float(y0), float(x1), float(y1))), info


def oracle_visibility(view: FrameView, objects: Sequence[SceneObject], *, depth: NDArray[np.floating] | None = None,
                      gt_boxes: Mapping[str, Box2D] | None = None, spacing: float = 0.05
                      ) -> tuple[list[ObjectInView], dict[str, dict[str, Any]]]:
    """Every ground-truth object as the view sees it, from a reference depth map or the 2D-box proxy.

    Parameters
    ----------
    view : FrameView
        The view.
    objects : sequence of SceneObject
        Ground-truth objects of the scene at the view's time.
    depth : numpy.ndarray, shape (H, W), optional
        Reference planar depth, metres (0 or NaN invalid); exactly one of ``depth`` and ``gt_boxes`` is given.
    gt_boxes : mapping of str to tuple of int, optional
        Object id -> this camera's ground-truth visible box (objects not listed are not visible).
    spacing : float, default 0.05
        Surface-sample spacing, metres.

    Returns
    -------
    seen : list of ObjectInView
        One per object, in input order.
    diagnostics : dict of str to dict
        Object id -> diagnostics (``n_samples`` and ``no_depth_fraction`` with depth; see
        :func:`gt_box_visibility` otherwise).

    Raises
    ------
    ValueError
        Unless exactly one of ``depth`` and ``gt_boxes`` is given.
    """
    if (depth is None) == (gt_boxes is None):
        raise ValueError("give exactly one of depth and gt_boxes")
    seen, diagnostics = [], {}
    for obj in objects:
        if depth is not None:
            result, diag = object_in_view(view, obj, depth, spacing=spacing)
            info: dict[str, Any] = {k: float(v) for k, v in diag.items()}
        else:
            result, info = gt_box_visibility(view, obj, gt_boxes.get(obj.object_id), spacing=spacing)  # type: ignore[union-attr]
        seen.append(result)
        diagnostics[obj.object_id] = info
    return seen, diagnostics


# --------------------------------------------------------------------------------------------------------
# Detections and lifting
# --------------------------------------------------------------------------------------------------------


def make_detections(view: FrameView, instances: Sequence[Instance], boundary_px: float = 5.0) -> list[Detection]:
    """Wrap a view's SAM 3 instances with their candidate labels and full-resolution mask statistics.

    Parameters
    ----------
    view : FrameView
        The view (its dataset selects the phrase-to-label mapping).
    instances : sequence of Instance
        Detections, in the order to index them.
    boundary_px : float, default 5.0
        Edge band width for ``boundary_fraction``, pixels.

    Returns
    -------
    list of Detection
        One per instance.
    """
    detections = []
    for i, inst in enumerate(instances):
        labels = labels_for_phrase(view.dataset, inst.phrase)
        detections.append(Detection(i, inst, labels, labels[0] if labels else None,
                                    mask_features(inst.mask, boundary_px)))
    return detections


def strided_camera(camera: Camera, stride: int, shape: tuple[int, int]) -> Camera:
    """Intrinsics of the image ``image[::stride, ::stride]``.

    Pixel ``(u, v)`` of the strided image is pixel ``(stride * u, stride * v)`` of the original, so the focal
    lengths and principal point divide by ``stride`` exactly.

    Parameters
    ----------
    camera : Camera
        Intrinsics of the full-resolution image.
    stride : int
        Pixel stride.
    shape : tuple of int
        ``(H, W)`` of the strided image.

    Returns
    -------
    Camera
        The strided image's intrinsics.
    """
    return Camera(camera.fx / stride, camera.fy / stride, camera.cx / stride, camera.cy / stride,
                  int(shape[1]), int(shape[0]))


def lift_detection(detection: Detection, depth: NDArray[np.floating], view: FrameView,
                   config: PipelineConfig) -> Lift:
    """Lift one detection's mask to a world box with one depth map.

    Parameters
    ----------
    detection : Detection
        The detection; its mask has the image's resolution.
    depth : numpy.ndarray, shape (H, W)
        Planar depth at the image's resolution, metres (0 or NaN invalid).
    view : FrameView
        The view (camera, ``T_wc`` and up).
    config : PipelineConfig
        Supplies ``lift_stride``, ``lift_stride_min_pixels``, ``erode_px`` and ``boundary_px``.

    Returns
    -------
    Lift
        The box and features, or the failure reason, with the stride used.

    Raises
    ------
    ValueError
        If the depth map's shape differs from the mask's.
    """
    mask = detection.instance.mask
    if depth.shape != mask.shape:
        raise ValueError(f"depth {depth.shape} and mask {mask.shape} differ; resample the depth first")
    large = detection.mask_stats["mask_area_px"] >= config.lift_stride_min_pixels
    s = config.lift_stride if large else 1
    if s > 1:
        mask_s, depth_s = mask[::s, ::s], depth[::s, ::s]
        camera = strided_camera(view.camera.scaled(mask.shape[1], mask.shape[0]), s, mask_s.shape)
        erode = max(1, int(round(config.erode_px / s)))
        boundary = config.boundary_px / s
    else:
        mask_s, depth_s = mask, depth
        camera = view.camera.scaled(mask.shape[1], mask.shape[0])
        erode, boundary = config.erode_px, config.boundary_px
    if not mask_s.any():
        return Lift(None, {}, "mask empty after subsampling", s)
    try:
        box, features = lift_instance(mask_s, depth_s, camera, view.T_wc, view.up, erode_px=erode,
                                      boundary_px=boundary)
    except LiftError as error:
        return Lift(None, {}, str(error), s)
    features.update({k: detection.mask_stats[k] for k in MASK_FEATURE_KEYS})
    return Lift(box, features, None, s)


# --------------------------------------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------------------------------------


def auto_objects(detections: Sequence[Detection], lifts: Sequence[Lift], match: MatchResult,
                 oracle_labels: Mapping[str, str]) -> tuple[dict[str, SceneObject], dict[str, int]]:
    """The automatic boxes of one depth source, keyed for answering.

    Parameters
    ----------
    detections : sequence of Detection
        The view's detections.
    lifts : sequence of Lift
        One lift per detection, from this depth source.
    match : MatchResult
        Matching of these lifts to the oracle objects.
    oracle_labels : mapping of str to str
        Oracle object id -> label.

    Returns
    -------
    objects_by_id : dict of str to SceneObject
        A matched detection under its oracle object's id and label; an unmatched one under
        ``"auto:<index>"`` with its :attr:`Detection.qa_label`. Detections without a box are left out.
    detection_of : dict of str to int
        Key -> detection index.
    """
    matched = match.by_detection()
    objects_by_id: dict[str, SceneObject] = {}
    detection_of: dict[str, int] = {}
    for det, lift in zip(detections, lifts):
        if lift.box is None:
            continue
        m = matched.get(det.index)
        if m is not None:
            key, label = m.object_id, oracle_labels[m.object_id]
        else:
            key, label = f"{AUTO_KEY_PREFIX}{det.index}", det.qa_label or det.instance.phrase
        objects_by_id[key] = SceneObject(key, label, lift.box, source="auto")
        detection_of[key] = det.index
    return objects_by_id, detection_of


def _evidence(key: str, detection_of: Mapping[str, int], detections: Sequence[Detection],
              lifts: Sequence[Lift], matched: Mapping[int, Any]) -> dict[str, Any]:
    """What the uncertainty model needs about one automatic box.

    Parameters
    ----------
    key : str
        The box's key in the automatic objects.
    detection_of : mapping of str to int
        Key -> detection index.
    detections : sequence of Detection
        The view's detections.
    lifts : sequence of Lift
        This source's lifts.
    matched : mapping of int to Match
        Detection index -> match.

    Returns
    -------
    dict
        ``object_id`` (the oracle id or ``None``), ``detection``, ``phrase``, ``score``, ``match_distance``
        and ``match_iou`` (``None`` if unmatched), and ``features``.
    """
    i = detection_of[key]
    m = matched.get(i)
    return {"object_id": None if m is None else m.object_id, "detection": i,
            "phrase": detections[i].instance.phrase, "score": detections[i].instance.score,
            "match_distance": None if m is None else m.distance, "match_iou": None if m is None else m.iou,
            "features": lifts[i].features}


def answer_auto(specs: Sequence[QuestionSpec], view: FrameView, source: str, detections: Sequence[Detection],
                lifts: Sequence[Lift], match: MatchResult, oracle_labels: Mapping[str, str]
                ) -> tuple[list[QAItem], list[dict[str, Any]]]:
    """Answer specs from one depth source's automatic boxes.

    Parameters
    ----------
    specs : sequence of QuestionSpec
        The view's questions.
    view : FrameView
        The view.
    source : str
        Depth source name; items get ``source="auto_<source>"``.
    detections : sequence of Detection
        The view's detections.
    lifts : sequence of Lift
        One lift per detection, from this source.
    match : MatchResult
        Matching of these lifts to the oracle objects.
    oracle_labels : mapping of str to str
        Oracle object id -> label.

    Returns
    -------
    items : list of QAItem
        One per answerable spec, in spec order, with the evidence in ``extra``.
    unanswered : list of dict
        Per spec without an answer: ``id``, ``qtype`` and the ``missing`` referent ids.
    """
    objects_by_id, detection_of = auto_objects(detections, lifts, match, oracle_labels)
    matched = match.by_detection()
    raw = make_items(specs, objects_by_id, view, f"auto_{source}", in_view=set(objects_by_id))
    answered = {item.id for item in raw}
    items = []
    for item in raw:
        extra = dict(item.extra)
        if item.spec.qtype == "count":
            phrase = item.spec.params["phrase"]
            of_phrase = [d for d in detections if d.qa_label is not None and phrase_for(d.qa_label) == phrase]
            extra["counted"] = [_evidence(o["object_id"], detection_of, detections, lifts, matched)
                                for o in item.objects]
            extra["n_detections_2d"] = len(of_phrase)
            extra["n_lift_failed"] = sum(lifts[d.index].box is None for d in of_phrase)
        else:
            extra["referents"] = [_evidence(r.object_id, detection_of, detections, lifts, matched)
                                  for r in item.spec.referents]
        items.append(replace(item, extra=extra))
    unanswered = [{"id": spec.id, "qtype": spec.qtype,
                   "missing": [r.object_id for r in spec.referents if r.object_id not in objects_by_id]}
                  for spec in specs if spec.id not in answered]
    return items, unanswered


def answer_oracle(specs: Sequence[QuestionSpec], view: FrameView, oracle: Sequence[ObjectInView],
                  config: QAConfig) -> list[QAItem]:
    """Answer specs from the ground-truth boxes, with the referents' visibility in ``extra``.

    Parameters
    ----------
    specs : sequence of QuestionSpec
        The view's questions.
    view : FrameView
        The view.
    oracle : sequence of ObjectInView
        Ground-truth objects as the view sees them.
    config : QAConfig
        Supplies the presence threshold for counts.

    Returns
    -------
    list of QAItem
        One per spec, ``source="oracle"``.
    """
    by_id = {o.obj.object_id: o for o in oracle}
    raw = make_items(specs, {k: o.obj for k, o in by_id.items()}, view, "oracle",
                     in_view=visible_ids(oracle, config))
    items = []
    for item in raw:
        ids = [r.object_id for r in item.spec.referents] or [o["object_id"] for o in item.objects]
        extra = dict(item.extra)
        extra["visibility"] = {i: {"visible_fraction": by_id[i].visible_fraction,
                                   "truncated_fraction": by_id[i].truncated_fraction} for i in ids}
        items.append(replace(item, extra=extra))
    return items


# --------------------------------------------------------------------------------------------------------
# One view
# --------------------------------------------------------------------------------------------------------


def process_view(view: FrameView, objects: Sequence[SceneObject], instances: Sequence[Instance],
                 depths: Mapping[str, NDArray[np.floating]], *, visibility: str,
                 gt_boxes: Mapping[str, Box2D] | None = None, config: PipelineConfig | None = None,
                 timings: dict[str, float] | None = None) -> ViewResult:
    """Run every model-free step for one view: oracle visibility, lifting, matching, questions and answers.

    Parameters
    ----------
    view : FrameView
        The view.
    objects : sequence of SceneObject
        Ground-truth objects of the scene at the view's time.
    instances : sequence of Instance
        SAM 3 detections at the image's resolution.
    depths : mapping of str to numpy.ndarray, shape (H, W)
        Depth source -> planar depth at the image's resolution, metres; every source gets automatic boxes.
    visibility : str
        A key of ``depths`` to measure oracle visibility against, or :data:`GT_BOX2D`.
    gt_boxes : mapping of str to tuple of int, optional
        Required with :data:`GT_BOX2D`: object id -> this camera's ground-truth visible box.
    config : PipelineConfig, optional
        Settings; defaults to ``PipelineConfig()``.
    timings : dict of str to float, optional
        Stage timings measured before this call (loading, models); stages here are added to a copy.

    Returns
    -------
    ViewResult
        Everything produced for the view.

    Raises
    ------
    ValueError
        If ``visibility`` names no depth map, or :data:`GT_BOX2D` is used without ``gt_boxes``.
    """
    config = config or PipelineConfig()
    timings = dict(timings or {})
    t = time.perf_counter()
    if visibility == GT_BOX2D:
        if gt_boxes is None:
            raise ValueError("the gt_box2d visibility method needs gt_boxes")
        oracle, oracle_diag = oracle_visibility(view, objects, gt_boxes=gt_boxes, spacing=config.visibility_spacing)
    elif visibility in depths:
        oracle, oracle_diag = oracle_visibility(view, objects, depth=depths[visibility],
                                                spacing=config.visibility_spacing)
    else:
        raise ValueError(f"visibility {visibility!r} is neither {GT_BOX2D!r} nor a depth source {list(depths)}")
    timings["oracle_visibility"] = time.perf_counter() - t

    detections = make_detections(view, instances, config.boundary_px)
    present_ids = visible_ids(oracle, config.qa)
    present = [o.obj for o in oracle if o.obj.object_id in present_ids]
    lifts: dict[str, list[Lift]] = {}
    matches: dict[str, MatchResult] = {}
    for source, depth in depths.items():
        t = time.perf_counter()
        lifts[source] = [lift_detection(det, depth, view, config) for det in detections]
        timings[f"lift_{source}"] = time.perf_counter() - t
        t = time.perf_counter()
        matches[source] = match_detections([lift.box for lift in lifts[source]], [d.labels for d in detections],
                                           present, config.match, up=view.up)
        timings[f"match_{source}"] = time.perf_counter() - t

    t = time.perf_counter()
    drop_list: list[Any] = []
    specs = generate_specs(view, oracle, config.qa, config.qa_seed, diagnostics=drop_list)
    drops = dict(Counter(f"{d.qtype}/{d.reason}" for d in drop_list))
    items = {"oracle": answer_oracle(specs, view, oracle, config.qa)}
    oracle_labels = {o.obj.object_id: o.obj.label for o in oracle}
    unanswered: dict[str, list[dict[str, Any]]] = {}
    for source in depths:
        items[f"auto_{source}"], unanswered[f"auto_{source}"] = answer_auto(
            specs, view, source, detections, lifts[source], matches[source], oracle_labels)
    timings["qa"] = time.perf_counter() - t
    return ViewResult(view, visibility, tuple(depths), oracle, oracle_diag, detections, lifts, matches, specs,
                      items, unanswered, drops, timings, config)


# --------------------------------------------------------------------------------------------------------
# Loading inputs and running models
# --------------------------------------------------------------------------------------------------------


def load_depth(view: FrameView, source: str, image: NDArray[np.uint8], moge: DepthFn | None = None
               ) -> NDArray[np.floating]:
    """One depth source's map for a view, at the image's resolution.

    Parameters
    ----------
    view : FrameView
        The view.
    source : str
        ``"laser"`` or ``"lidar"`` (ARKitScenes), ``"gt_depth"`` (Smart Spaces simulator) or ``"moge"``.
    image : numpy.ndarray, shape (H, W, 3), uint8
        The view's upright RGB image (input of MoGe; sets the output size).
    moge : callable, optional
        ``moge(image, camera) -> depth``; defaults to :func:`measured_space.perception.estimate_depth_moge`.

    Returns
    -------
    numpy.ndarray, shape (H, W)
        Planar depth in metres; 0 or NaN where invalid.

    Raises
    ------
    ValueError
        For an unknown source or one the view's dataset lacks.
    """
    shape = image.shape[:2]
    if source in ("laser", "lidar") and view.dataset == "arkitscenes":
        from measured_space.datasets import arkitscenes

        if source == "laser":
            depth = arkitscenes.load_laser_depth(view)
        else:
            depth = arkitscenes.load_lidar_depth(view)
        return depth if depth.shape == shape else resample_depth(depth, shape)
    if source == "gt_depth" and view.dataset == "smartspaces":
        from measured_space.datasets import smartspaces

        depth = smartspaces.load_depth(view)
        return depth if depth.shape == shape else resample_depth(depth, shape)
    if source == "moge":
        if moge is None:
            from measured_space.perception import estimate_depth_moge

            moge = estimate_depth_moge
        return moge(image, view.camera)
    raise ValueError(f"depth source {source!r} is not available for dataset {view.dataset!r}")


def run_view(view: FrameView, objects: Sequence[SceneObject], *, sources: Sequence[str], visibility: str,
             gt_boxes: Mapping[str, Box2D] | None = None, config: PipelineConfig | None = None,
             segment: SegmentFn | None = None, moge: DepthFn | None = None,
             load_image: Callable[[FrameView], NDArray[np.uint8]] | None = None) -> ViewResult:
    """Load a view's image and depths, run SAM 3 (and MoGe), then :func:`process_view`.

    Parameters
    ----------
    view : FrameView
        The view (upright image).
    objects : sequence of SceneObject
        Ground-truth objects of the scene at the view's time.
    sources : sequence of str
        Depth sources for automatic boxes (see :data:`DEPTH_SOURCES`).
    visibility : str
        One of ``sources`` to measure oracle visibility against, or :data:`GT_BOX2D`.
    gt_boxes : mapping of str to tuple of int, optional
        Required with :data:`GT_BOX2D`: object id -> this camera's ground-truth visible box.
    config : PipelineConfig, optional
        Settings; defaults to ``PipelineConfig()``.
    segment : callable, optional
        ``segment(image, phrases, score_threshold, rot90_k=..., dedupe_iou=...)``; defaults to
        :func:`measured_space.perception.segment_instances`.
    moge : callable, optional
        ``moge(image, camera) -> depth``; defaults to :func:`measured_space.perception.estimate_depth_moge`.
    load_image : callable, optional
        ``load_image(view) -> RGB``; defaults to :func:`measured_space.datasets.load_rgb`.

    Returns
    -------
    ViewResult
        Everything produced for the view, with ``timings`` for loading, segmentation and each depth source.

    Raises
    ------
    ValueError
        If the image does not match the view's camera, or as :func:`process_view` and :func:`load_depth`.
    """
    config = config or PipelineConfig()
    timings: dict[str, float] = {}
    t = time.perf_counter()
    if load_image is None:
        from measured_space.datasets import load_rgb as load_image
    image = load_image(view)
    if image.shape[:2] != (view.camera.height, view.camera.width):
        raise ValueError(f"image {image.shape[:2]} does not match camera {view.camera.height}x{view.camera.width}")
    timings["load_image"] = time.perf_counter() - t

    t = time.perf_counter()
    if segment is None:
        from measured_space.perception import segment_instances as segment
    instances = segment(image, prompt_phrases(view.dataset), config.score_threshold, rot90_k=0,
                        dedupe_iou=config.dedupe_iou)
    timings["segment"] = time.perf_counter() - t

    depths = {}
    for source in sources:
        t = time.perf_counter()
        depths[source] = load_depth(view, source, image, moge)
        timings[f"depth_{source}"] = time.perf_counter() - t
    return process_view(view, objects, instances, depths, visibility=visibility, gt_boxes=gt_boxes,
                        config=config, timings=timings)


def to_json_line(record: Mapping[str, Any]) -> str:
    """One record as a JSON line (numpy values converted; non-finite floats become ``null``).

    Parameters
    ----------
    record : mapping
        A record from :meth:`ViewResult.records`.

    Returns
    -------
    str
        Compact JSON without a trailing newline.
    """
    return json.dumps(_finite(record), separators=(",", ":"))


def _finite(value: Any) -> Any:
    """Recursively convert numpy values and replace NaN or infinite floats with ``None``.

    Parameters
    ----------
    value : Any
        A JSON-like value.

    Returns
    -------
    Any
        The same structure with plain Python numbers and no non-finite floats.
    """
    if isinstance(value, Mapping):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
