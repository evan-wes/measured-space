"""Match automatic detections to ground-truth objects in one view.

A detection is matched to at most one ground-truth ("oracle") object and vice versa. Candidate pairs must be
**label-compatible**: a detection's prompt phrase can stand for several dataset labels (a "mobile robot" may be
a NovaCarter or a Transporter), and it may only match an object with one of them. Among compatible pairs the
assignment minimises the summed 3D centre distance (the Hungarian method, :func:`scipy.optimize.
linear_sum_assignment`), subject to a **distance gate** per oracle object of

    gate = max(min_gate, gate_diagonal_fraction * |oracle box diagonal|)

(default ``max(0.5 m, 0.5 * diagonal)``), so a large object tolerates a larger centre offset than a small one.
Infeasible pairs (incompatible or beyond the gate) get a cost larger than any feasible assignment's total, so
the solver first maximises the number of feasible matches and then minimises their summed distance; infeasible
pairs it is forced to assign are then discarded.

Each match also reports the 3D intersection-over-union of the two boxes, computed for **gravity-aligned
prisms** (:func:`prism_iou`): each box is replaced by the vertical prism over the convex hull of its corners
projected onto the horizontal plane, spanning the box's extent along ``up``. For boxes that rotate only about
``up`` (all perception boxes, and all ground-truth boxes in the datasets used here except three ARKitScenes
boxes tilted by 2-9.5 degrees) the prism is the box itself and the IoU is exact; for a tilted box it is the IoU
of its gravity-aligned bounding prism, which overstates the box's volume.

Distances are in metres. Conventions are those of :mod:`measured_space.geometry`.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment

from measured_space.geometry import Array, OrientedBox
from measured_space.scene import SceneObject

_WORLD_UP = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class MatchConfig:
    """Distance gate for matching detections to oracle objects.

    Parameters
    ----------
    min_gate : float, default 0.5
        Smallest allowed centre distance gate, metres.
    gate_diagonal_fraction : float, default 0.5
        The gate grows to this fraction of the oracle box's space diagonal when that is larger.
    """

    min_gate: float = 0.5
    gate_diagonal_fraction: float = 0.5

    def gate(self, box: OrientedBox) -> float:
        """The centre-distance gate for one oracle box.

        Parameters
        ----------
        box : OrientedBox
            The oracle object's box.

        Returns
        -------
        float
            ``max(min_gate, gate_diagonal_fraction * norm(box.size))``, metres.
        """
        return float(max(self.min_gate, self.gate_diagonal_fraction * np.linalg.norm(box.size)))


@dataclass(frozen=True)
class Match:
    """One detection matched to one oracle object.

    Parameters
    ----------
    detection : int
        Index of the detection in the input sequence.
    object_id : str
        The oracle object's id.
    distance : float
        Distance between the two box centres, metres.
    iou : float
        3D IoU of the two boxes' gravity-aligned prisms (see :func:`prism_iou`), in [0, 1].
    gate : float
        The distance gate that applied to this oracle object, metres.
    """

    detection: int
    object_id: str
    distance: float
    iou: float
    gate: float


@dataclass(frozen=True)
class MatchResult:
    """The outcome of matching one view's detections to its oracle objects.

    Parameters
    ----------
    matches : tuple of Match
        Matched pairs, sorted by detection index.
    missed : tuple of str
        Oracle object ids left unmatched (misses), in input order.
    false_positives : tuple of int
        Indices of detections with a box that were left unmatched, ascending.
    unboxed : tuple of int, default ()
        Indices of detections without a box (e.g. lifting failed); they take no part in matching.
    """

    matches: tuple[Match, ...]
    missed: tuple[str, ...]
    false_positives: tuple[int, ...]
    unboxed: tuple[int, ...] = ()

    def by_detection(self) -> dict[int, Match]:
        """Index the matches by detection.

        Returns
        -------
        dict of int to Match
            Detection index -> its match.
        """
        return {m.detection: m for m in self.matches}

    def by_object(self) -> dict[str, Match]:
        """Index the matches by oracle object.

        Returns
        -------
        dict of str to Match
            Oracle object id -> its match.
        """
        return {m.object_id: m for m in self.matches}


# --------------------------------------------------------------------------------------------------------
# Gravity-aligned prism IoU
# --------------------------------------------------------------------------------------------------------


def _horizontal_basis(up: Array) -> tuple[Array, Array]:
    """Two unit vectors spanning the plane perpendicular to ``up``.

    Parameters
    ----------
    up : numpy.ndarray, shape (3,)
        Unit up direction.

    Returns
    -------
    e1, e2 : numpy.ndarray, shape (3,)
        Orthonormal horizontal directions with ``(e1, e2, up)`` right-handed.
    """
    seed = np.eye(3)[int(np.argmin(np.abs(up)))]
    e1 = seed - (seed @ up) * up
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def _cross2(o: Array, a: Array, b: Array) -> float:
    """z-component of the cross product ``(a - o) x (b - o)`` of 2D points.

    Parameters
    ----------
    o, a, b : numpy.ndarray, shape (2,)
        Points in the plane.

    Returns
    -------
    float
        Positive when ``o -> a -> b`` turns counter-clockwise.
    """
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def convex_hull_2d(points: ArrayLike) -> Array:
    """Convex hull of 2D points by Andrew's monotone chain, counter-clockwise, without repeated vertices.

    Parameters
    ----------
    points : array_like, shape (N, 2)
        Points in the plane (duplicates allowed).

    Returns
    -------
    numpy.ndarray, shape (M, 2)
        Hull vertices counter-clockwise; collinear points are dropped, so a degenerate input gives fewer than 3.
    """
    pts = np.unique(np.asarray(points, dtype=np.float64).reshape(-1, 2), axis=0)    # sorted by x, then y
    if len(pts) < 3:
        return pts

    def half(sequence: Array) -> list[Array]:
        """One monotone half of the hull.

        Parameters
        ----------
        sequence : numpy.ndarray, shape (N, 2)
            Points sorted along x (or reversed).

        Returns
        -------
        list of numpy.ndarray
            The chain's vertices, its last one shared with the other half.
        """
        chain: list[Array] = []
        for p in sequence:
            while len(chain) >= 2 and _cross2(chain[-2], chain[-1], p) <= 1e-15:
                chain.pop()
            chain.append(p)
        return chain

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(polygon: ArrayLike) -> float:
    """Area of a simple polygon (shoelace formula).

    Parameters
    ----------
    polygon : array_like, shape (M, 2)
        Vertices in order (either orientation).

    Returns
    -------
    float
        Unsigned area; 0 for fewer than 3 vertices.
    """
    p = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def clip_convex(subject: ArrayLike, clip: ArrayLike) -> Array:
    """Intersection of two convex polygons (Sutherland-Hodgman clipping).

    Parameters
    ----------
    subject : array_like, shape (M, 2)
        Convex polygon, counter-clockwise.
    clip : array_like, shape (K, 2)
        Convex polygon, counter-clockwise.

    Returns
    -------
    numpy.ndarray, shape (L, 2)
        The intersection's vertices counter-clockwise; fewer than 3 rows if the polygons do not overlap in area.
    """
    output = [np.asarray(p, dtype=np.float64) for p in np.asarray(subject, dtype=np.float64).reshape(-1, 2)]
    edges = np.asarray(clip, dtype=np.float64).reshape(-1, 2)
    if len(edges) < 3:
        return np.zeros((0, 2))
    for a, b in zip(edges, np.roll(edges, -1, axis=0)):
        if not output:
            break
        polygon, output = output, []
        for p, q in zip(polygon, polygon[1:] + polygon[:1]):
            p_in, q_in = _cross2(a, b, p) >= 0.0, _cross2(a, b, q) >= 0.0
            if p_in:
                output.append(p)
            if p_in != q_in:
                d_p, d_q = _cross2(a, b, p), _cross2(a, b, q)
                output.append(p + (q - p) * (d_p / (d_p - d_q)))
    return np.array(output).reshape(-1, 2)


def gravity_prism(box: OrientedBox, up: ArrayLike = _WORLD_UP) -> tuple[Array, float, float]:
    """The vertical prism a box is approximated by: its corners' horizontal hull and vertical span.

    Parameters
    ----------
    box : OrientedBox
        The box.
    up : array_like, shape (3,), default (0, 0, 1)
        Up direction in the box's frame; need not be unit length.

    Returns
    -------
    footprint : numpy.ndarray, shape (M, 2)
        Convex hull of the corners in a horizontal basis (metres), counter-clockwise.
    bottom, top : float
        The box's lowest and highest extent along ``up``, metres.
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    e1, e2 = _horizontal_basis(up)
    corners = box.corners()
    heights = corners @ up
    return convex_hull_2d(np.stack([corners @ e1, corners @ e2], axis=1)), float(heights.min()), \
        float(heights.max())


def prism_iou(a: OrientedBox, b: OrientedBox, up: ArrayLike = _WORLD_UP) -> float:
    """3D intersection-over-union of two boxes' gravity-aligned prisms (see :func:`gravity_prism`).

    Exact for boxes that rotate only about ``up``; for tilted boxes it is the IoU of their bounding prisms.

    Parameters
    ----------
    a, b : OrientedBox
        Boxes in the same frame.
    up : array_like, shape (3,), default (0, 0, 1)
        Up direction; need not be unit length.

    Returns
    -------
    float
        IoU in [0, 1]; 0 when either prism has no volume.
    """
    foot_a, lo_a, hi_a = gravity_prism(a, up)
    foot_b, lo_b, hi_b = gravity_prism(b, up)
    volume_a = polygon_area(foot_a) * (hi_a - lo_a)
    volume_b = polygon_area(foot_b) * (hi_b - lo_b)
    overlap_h = min(hi_a, hi_b) - max(lo_a, lo_b)
    if volume_a <= 0 or volume_b <= 0 or overlap_h <= 0:
        return 0.0
    intersection = polygon_area(clip_convex(foot_a, foot_b)) * overlap_h
    union = volume_a + volume_b - intersection
    return float(np.clip(intersection / union, 0.0, 1.0)) if union > 0 else 0.0


# --------------------------------------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------------------------------------


def match_detections(boxes: Sequence[OrientedBox | None], labels: Sequence[Collection[str]],
                     objects: Sequence[SceneObject], config: MatchConfig | None = None, *,
                     up: ArrayLike = _WORLD_UP) -> MatchResult:
    """Match one view's detections to its oracle objects (Hungarian assignment on centre distance).

    Parameters
    ----------
    boxes : sequence of OrientedBox or None
        Each detection's 3D box in the oracle boxes' frame; ``None`` for a detection without a box.
    labels : sequence of collection of str
        For each detection, the dataset labels it may correspond to (e.g. from
        :func:`measured_space.perception.labels_for_phrase`); same length as ``boxes``.
    objects : sequence of SceneObject
        Oracle objects that may be matched (normally those present in the view); ids must be unique.
    config : MatchConfig, optional
        Gate settings; defaults to ``MatchConfig()``.
    up : array_like, shape (3,), default (0, 0, 1)
        Up direction for the IoU prisms.

    Returns
    -------
    MatchResult
        Matches with their centre distance and IoU, unmatched oracle ids, unmatched detections, and
        detections without a box.

    Raises
    ------
    ValueError
        If ``boxes`` and ``labels`` differ in length or object ids repeat.
    """
    config = config or MatchConfig()
    if len(boxes) != len(labels):
        raise ValueError(f"{len(boxes)} boxes but {len(labels)} label sets")
    if len({o.object_id for o in objects}) != len(objects):
        raise ValueError("oracle object ids must be unique")
    boxed = [i for i, box in enumerate(boxes) if box is not None]
    unboxed = tuple(i for i, box in enumerate(boxes) if box is None)
    if not boxed or not objects:
        return MatchResult((), tuple(o.object_id for o in objects), tuple(boxed), unboxed)

    centers_d = np.stack([boxes[i].center for i in boxed])                    # type: ignore[union-attr]
    centers_o = np.stack([o.box.center for o in objects])
    distance = np.linalg.norm(centers_d[:, None, :] - centers_o[None, :, :], axis=2)
    gates = np.array([config.gate(o.box) for o in objects])
    compatible = np.array([[o.label in labels[i] for o in objects] for i in boxed], dtype=bool)
    feasible = compatible & (distance <= gates[None, :])
    # Any single infeasible pair costs more than every feasible assignment together.
    big = 1.0 + float(distance[feasible].sum()) if feasible.any() else 1.0
    cost = np.where(feasible, distance, big)
    rows, cols = linear_sum_assignment(cost)

    matches = []
    for r, c in zip(rows, cols):
        if feasible[r, c]:
            i = boxed[r]
            matches.append(Match(i, objects[c].object_id, float(distance[r, c]),
                                 prism_iou(boxes[i], objects[c].box, up), float(gates[c])))  # type: ignore[arg-type]
    matches.sort(key=lambda m: m.detection)
    matched_det = {m.detection for m in matches}
    matched_obj = {m.object_id for m in matches}
    return MatchResult(tuple(matches), tuple(o.object_id for o in objects if o.object_id not in matched_obj),
                       tuple(i for i in boxed if i not in matched_det), unboxed)
