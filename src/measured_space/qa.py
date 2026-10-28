"""Spatial questions about one image: ask once, answer from any set of boxes.

Asking and answering are kept separate so that oracle and automatic labels are paired by construction.
:func:`generate_specs` decides *what* to ask about *which* objects, from one view's boxes and visibility
(normally ground truth), and keeps only questions whose answer is unambiguous. Referents must be clearly
visible and nameable. Distances must not be near zero. Categorical answers must clear a margin from their
decision threshold. :func:`answer` then computes the answer to a spec from whatever boxes it is given,
oracle or perception, with the same code. :func:`make_items` wraps those answers as
:class:`~measured_space.qa_schema.QAItem` records.

Objects are named by a *phrase*, not by their dataset label, because the phrase is what a reader of the
question sees. Two labels that read the same ("NovaCarter" and "Transporter" are both "mobile robot") make
each other ambiguous. When several objects in the image share a phrase, they are told apart by their
left-to-right order in the image, and only where neighbours are far enough apart for that order to be
obvious.

Conventions are those of :mod:`measured_space.geometry`. All distances are in metres.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Collection, Mapping, Sequence

import numpy as np

from measured_space.geometry import Array, OrientedBox, center_distance, surface_distance, transform
from measured_space.qa_schema import QAItem, QuestionSpec, Referent
from measured_space.scene import FrameView, ObjectInView, SceneObject

QTYPES: tuple[str, ...] = (
    "distance_center", "distance_surface", "size_height", "size_longest", "ego_distance", "relative_depth",
    "left_right", "closest_instance", "count")

# qtype -> (template id, fixed wording). Changing a wording means a new version suffix, so that items asked
# with different wordings are never pooled by accident.
TEMPLATES: dict[str, tuple[str, str]] = {
    "distance_center": (
        "distance_center_v1",
        "What is the straight-line distance between the centre of {a} and the centre of {b}? "
        "Answer in metres."),
    "distance_surface": (
        "distance_surface_v1",
        "What is the shortest distance between {a} and {b}, measured between their closest surface points? "
        "Answer in metres."),
    "size_height": (
        "size_height_v1",
        "How tall is {a}, measured vertically from its lowest point to its highest point? Answer in metres."),
    "size_longest": (
        "size_longest_v1",
        "What is the longest horizontal dimension (length or width, not height) of {a}? Answer in metres."),
    "ego_distance": (
        "ego_distance_v1",
        "What is the distance from the camera to the nearest point of {a}? Answer in metres."),
    "relative_depth": (
        "relative_depth_v1",
        "Which is closer to the camera, {a} or {b}? Measure from the camera to the nearest point of each "
        'object. Answer "{a}" or "{b}".'),
    "left_right": (
        "left_right_v1",
        "From the camera's viewpoint in this image, is {a} to the left or to the right of {b}? "
        'Answer "left" or "right".'),
    "closest_instance": (
        "closest_instance_v1",
        "Which {phrase} is closest to {a}, measured between their closest surface points? "
        "Answer with one of: {choices}."),
    "count": (
        "count_v1",
        "How many {plural} are visible in this image? Answer with a whole number."),
}

# Dataset label -> the phrase a question uses. Labels not listed fall back to plain words split from the
# label (see :func:`phrase_for`). Distinct labels that share a phrase are deliberately indistinguishable.
PHRASES: dict[str, str] = {
    # Smart Spaces agent classes.
    "Person": "person",
    "Forklift": "forklift",
    "PalletTruck": "pallet truck",
    "NovaCarter": "mobile robot",
    "Transporter": "mobile robot",
    "AgilityDigit": "humanoid robot",
    "FourierGR1T2": "humanoid robot",
    # ARKitScenes 3DOD furniture classes.
    "bathtub": "bathtub",
    "bed": "bed",
    "cabinet": "cabinet",
    "chair": "chair",
    "dishwasher": "dishwasher",
    "fireplace": "fireplace",
    "oven": "oven",
    "refrigerator": "refrigerator",
    "shelf": "shelf",
    "sink": "sink",
    "sofa": "sofa",
    "stool": "stool",
    "stove": "stove",
    "table": "table",
    "toilet": "toilet",
    "tv_monitor": "TV",
    "washer": "washing machine",      # "washer" alone also names a dishwasher or a small ring
}

_PHRASE_VALUES = frozenset(PHRASES.values())
_IRREGULAR_PLURALS = {"person": "people", "shelf": "shelves"}
_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")
_NUMERIC_TOLERANCE_KIND = {"distance_center": "distance", "distance_surface": "distance",
                           "ego_distance": "distance", "size_height": "size", "size_longest": "size"}


@dataclass(frozen=True)
class QAConfig:
    """Every threshold and tolerance the question generator uses.

    Attributes
    ----------
    min_visible_fraction : float
        A referent needs at least this share of its camera-facing surface visible (default 0.3).
    max_truncated_fraction : float
        A referent may have at most this share of its camera-facing surface outside the image
        (default 0.3).
    presence_threshold : float
        An object with ``visible_fraction`` above this is *in* the image even if it cannot be a referent.
        It makes its phrase non-unique, takes part in left-to-right ordering, and makes a count ambiguous
        unless it is clearly visible. Default 0.0 (any visible sample). Must be below both
        ``min_visible_fraction`` and ``count_min_visible_fraction``.
    ordinal_min_gap_fraction : float
        Neighbouring instances of a shared phrase are named by position ("the leftmost chair") only if their
        visible-bbox centres are at least this fraction of the image width apart (default 0.05).
    min_distance : float
        Centre, surface and camera distances below this (m) are not asked: near-zero distances are
        ambiguous, and touching or overlapping boxes have surface distance 0 (default 0.2).
    relative_depth_min_abs, relative_depth_min_rel : float
        Relative depth is asked only if the two camera distances differ by at least
        ``max(abs, rel * nearer distance)`` (defaults 0.3 m and 0.10).
    left_right_min_dx : float
        Left/right is asked only if the centres' camera-frame x differ by at least this (m) (default 0.3).
    left_right_require_image_agreement : bool
        If True (default), left/right is also dropped when the order of the projected centres in the image
        disagrees with the camera-frame x order. A near object can lie further left in the image than a far
        object that is further left in 3D, and a reader could take either meaning.
    closest_min_abs, closest_min_rel : float
        Closest-instance is asked only if the second-closest instance is farther than the closest by at
        least ``max(abs, rel * closest distance)`` (defaults 0.3 m and 0.10).
    count_min_visible_fraction : float
        An instance is counted if at least this share of it is visible (default 0.3).
    count_max_truncated_fraction : float
        A counted instance with more than this share outside the image is heavily truncated, which makes
        the count ambiguous (default 0.5).
    count_max : int
        Counts above this are not asked (default 10). They are dropped rather than clipped, because a
        clipped reference answer would be wrong.
    max_per_type : int or None
        At most this many questions of each type per view, chosen with a seeded RNG (default 3; None keeps
        all).
    distance_tol_abs, distance_tol_rel : float
        Tolerance stored in centre-distance, surface-distance and camera-distance specs (0.10 m, 0.10).
    size_tol_abs, size_tol_rel : float
        Tolerance stored in height and longest-dimension specs (0.05 m, 0.10).
    """

    min_visible_fraction: float = 0.3
    max_truncated_fraction: float = 0.3
    presence_threshold: float = 0.0
    ordinal_min_gap_fraction: float = 0.05
    min_distance: float = 0.2
    relative_depth_min_abs: float = 0.3
    relative_depth_min_rel: float = 0.10
    left_right_min_dx: float = 0.3
    left_right_require_image_agreement: bool = True
    closest_min_abs: float = 0.3
    closest_min_rel: float = 0.10
    count_min_visible_fraction: float = 0.3
    count_max_truncated_fraction: float = 0.5
    count_max: int = 10
    max_per_type: int | None = 3
    distance_tol_abs: float = 0.10
    distance_tol_rel: float = 0.10
    size_tol_abs: float = 0.05
    size_tol_rel: float = 0.10

    def __post_init__(self) -> None:
        """Reject a presence threshold that would hide clearly visible objects.

        Raises
        ------
        ValueError
            If ``presence_threshold`` is not below both visibility thresholds.
        """
        if not self.presence_threshold < min(self.min_visible_fraction, self.count_min_visible_fraction):
            raise ValueError("presence_threshold must be below min_visible_fraction and "
                             "count_min_visible_fraction")


@dataclass(frozen=True)
class Drop:
    """A candidate question, or a would-be referent, that generation left out, and why.

    Attributes
    ----------
    qtype : str
        The question type, or ``"referent"`` for an object that cannot be referred to at all.
    reason : str
        A short machine-readable reason, e.g. ``"margin_below_min"`` or ``"borderline_instance"``.
    object_ids : tuple of str
        The objects involved.
    detail : dict
        Values behind the decision (the margin and the threshold, the phrase, ...).
    """

    qtype: str
    reason: str
    object_ids: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)


class MissingReferentError(KeyError):
    """A spec refers to an object the given boxes do not include (e.g. perception missed it)."""


# --------------------------------------------------------------------------------------------------------
# Phrases and referring expressions
# --------------------------------------------------------------------------------------------------------

def phrase_for(label: str) -> str:
    """The noun phrase a question uses for a dataset label.

    Parameters
    ----------
    label : str
        The dataset's class name.

    Returns
    -------
    str
        The phrase from :data:`PHRASES`, or, for an unlisted label, its words split at underscores, hyphens
        and lower-to-upper case changes, in lower case (``"CoffeeMug"`` becomes ``"coffee mug"``). A phrase is
        its own phrase, so boxes labelled with phrases (e.g. by a text-prompted detector) match too.
    """
    if label in PHRASES:
        return PHRASES[label]
    if label in _PHRASE_VALUES:
        return label
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label.replace("_", " ").replace("-", " "))
    return " ".join(words.lower().split())


def plural(phrase: str) -> str:
    """The plural of a noun phrase, inflecting its last word.

    Parameters
    ----------
    phrase : str
        A singular noun phrase, e.g. ``"pallet truck"``.

    Returns
    -------
    str
        The plural, e.g. ``"pallet trucks"``; ``"person"`` becomes ``"people"``.
    """
    head, _, last = phrase.rpartition(" ")
    if last in _IRREGULAR_PLURALS:
        last = _IRREGULAR_PLURALS[last]
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        last += "es"
    elif len(last) > 1 and last.endswith("y") and last[-2] not in "aeiou":
        last = last[:-1] + "ies"
    else:
        last += "s"
    return f"{head} {last}" if head else last


def _ordinal(k: int) -> str:
    """An English ordinal for a 1-based position.

    Parameters
    ----------
    k : int
        The position, starting at 1.

    Returns
    -------
    str
        A word up to "tenth", else digits with a suffix ("11th", "22nd").
    """
    if k <= len(_ORDINALS):
        return _ORDINALS[k - 1]
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def _position_expression(index: int, n: int, phrase: str) -> str:
    """The expression naming one of ``n`` same-phrase instances by its left-to-right position.

    Parameters
    ----------
    index : int
        The instance's 0-based position from the left.
    n : int
        How many instances share the phrase (at least 2).
    phrase : str
        The shared phrase.

    Returns
    -------
    str
        ``"the leftmost <phrase>"``, ``"the rightmost <phrase>"`` or ``"the <ordinal> <phrase> from the
        left"``.
    """
    if index == 0:
        return f"the leftmost {phrase}"
    if index == n - 1:
        return f"the rightmost {phrase}"
    return f"the {_ordinal(index + 1)} {phrase} from the left"


def _bbox_center_u(seen: ObjectInView) -> float:
    """Horizontal pixel coordinate of the centre of an object's visible 2D box.

    Parameters
    ----------
    seen : ObjectInView
        An object with a non-None ``bbox2d``.

    Returns
    -------
    float
        ``(xmin + xmax) / 2`` in pixels.
    """
    xmin, _, xmax, _ = seen.bbox2d
    return 0.5 * (xmin + xmax)


def _is_present(seen: ObjectInView, config: QAConfig) -> bool:
    """Whether an object is in the image at all (see ``QAConfig.presence_threshold``).

    Parameters
    ----------
    seen : ObjectInView
        The object as the view sees it.
    config : QAConfig
        Thresholds.

    Returns
    -------
    bool
        True if it has a visible 2D box and its visible fraction exceeds the presence threshold.
    """
    return seen.bbox2d is not None and seen.visible_fraction > config.presence_threshold


def _is_clearly_visible(seen: ObjectInView, config: QAConfig) -> bool:
    """Whether an object is visible enough, and inside the image enough, to be asked about.

    Parameters
    ----------
    seen : ObjectInView
        The object as the view sees it.
    config : QAConfig
        Thresholds.

    Returns
    -------
    bool
        True if it passes the referent visibility and truncation thresholds and has a 2D box.
    """
    return (seen.bbox2d is not None and seen.visible_fraction >= config.min_visible_fraction
            and seen.truncated_fraction <= config.max_truncated_fraction)


def visible_ids(seen: Sequence[ObjectInView], config: QAConfig | None = None) -> set[str]:
    """Ids of the objects that are in the image, for answering count questions.

    Parameters
    ----------
    seen : sequence of ObjectInView
        Objects as one view sees them (oracle or perception boxes).
    config : QAConfig, optional
        Thresholds; defaults to ``QAConfig()``.

    Returns
    -------
    set of str
        Ids of objects whose visible fraction exceeds ``config.presence_threshold``.
    """
    config = config or QAConfig()
    return {o.obj.object_id for o in seen if _is_present(o, config)}


def _present_by_phrase(seen: Sequence[ObjectInView], config: QAConfig) -> dict[str, list[ObjectInView]]:
    """Group the objects in the image by phrase, each group ordered left to right.

    Parameters
    ----------
    seen : sequence of ObjectInView
        Objects as one view sees them.
    config : QAConfig
        Thresholds.

    Returns
    -------
    dict of str to list of ObjectInView
        Phrase -> present instances, sorted by visible-bbox centre (ties by object id), phrases sorted.
    """
    groups: dict[str, list[ObjectInView]] = {}
    for o in seen:
        if _is_present(o, config):
            groups.setdefault(phrase_for(o.obj.label), []).append(o)
    return {phrase: sorted(groups[phrase], key=lambda o: (_bbox_center_u(o), o.obj.object_id))
            for phrase in sorted(groups)}


def referring_expressions(view: FrameView, seen: Sequence[ObjectInView], config: QAConfig | None = None,
                          drops: list[Drop] | None = None) -> dict[str, str]:
    """Unambiguous expressions for the objects in an image, by object id.

    Uniqueness is decided by phrase among every object in the image (see ``QAConfig.presence_threshold``),
    not only among clearly visible ones, since a reader sees a partly hidden second chair too. A phrase with
    one instance gives ``"the <phrase>"``. Several instances are named by their left-to-right order of
    visible-bbox centres, and an instance is nameable only if both neighbours' centres are at least
    ``ordinal_min_gap_fraction`` of the image width away.

    Parameters
    ----------
    view : FrameView
        The image (its width sets the ordinal gap in pixels).
    seen : sequence of ObjectInView
        Objects as the view sees them.
    config : QAConfig, optional
        Thresholds; defaults to ``QAConfig()``.
    drops : list of Drop, optional
        If given, one :class:`Drop` is appended for each present object left without an expression.

    Returns
    -------
    dict of str to str
        Object id -> expression, for the nameable objects only.
    """
    config = config or QAConfig()
    min_gap = config.ordinal_min_gap_fraction * view.camera.width
    expressions: dict[str, str] = {}
    for phrase, group in _present_by_phrase(seen, config).items():
        if len(group) == 1:
            expressions[group[0].obj.object_id] = f"the {phrase}"
            continue
        separated = np.diff([_bbox_center_u(o) for o in group]) >= min_gap
        left_ok = np.concatenate([[True], separated])
        right_ok = np.concatenate([separated, [True]])
        for i, o in enumerate(group):
            if left_ok[i] and right_ok[i]:
                expressions[o.obj.object_id] = _position_expression(i, len(group), phrase)
            elif drops is not None:
                drops.append(Drop("referent", "ordinal_too_close", (o.obj.object_id,),
                                  {"phrase": phrase, "min_gap_px": min_gap}))
    return expressions


# --------------------------------------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------------------------------------

def _camera_distance(box: OrientedBox, camera_center: Array) -> float:
    """Distance from the camera centre to the nearest point of a solid box.

    Parameters
    ----------
    box : OrientedBox
        The object's box, world frame.
    camera_center : ndarray, shape (3,)
        Camera centre, world frame (m).

    Returns
    -------
    float
        The distance (m); 0 if the camera is inside the box.
    """
    return float(np.linalg.norm(box.closest_point(camera_center[None])[0] - camera_center))


def longest_horizontal_dimension(box: OrientedBox, up: Array) -> float:
    """The longer of a box's horizontal length and width, for any box orientation.

    The box's two most horizontal axes are projected onto the horizontal plane, and the answer is the larger
    of the box's extents along those two directions. For a gravity-aligned box this is the larger of its two
    horizontal side lengths. A tilted box's extent along a direction includes the share of its other sides
    that projects onto it, so this is the span a tape measure laid horizontally along the box would show.

    Parameters
    ----------
    box : OrientedBox
        The box.
    up : ndarray, shape (3,)
        The up direction in the box's frame; need not be unit length.

    Returns
    -------
    float
        The longest horizontal dimension (m).
    """
    up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    horizontal = box.rotation - np.outer(up, up @ box.rotation)       # columns: axes minus their up part
    most_horizontal = np.argsort(np.linalg.norm(horizontal, axis=0), kind="stable")[-2:]
    return max(box.extent_along(horizontal[:, i]) for i in most_horizontal)


def _answer_distance_center(spec: QuestionSpec, objs: Sequence[SceneObject],
                            view: FrameView) -> tuple[float, dict[str, Any]]:
    """Centre-to-centre distance between the two referents.

    Parameters
    ----------
    spec : QuestionSpec
        The question.
    objs : sequence of SceneObject
        The referents' objects, in spec order.
    view : FrameView
        The image.

    Returns
    -------
    value : float
        Distance (m).
    extra : dict
        Empty.
    """
    return center_distance(objs[0].box, objs[1].box), {}


def _answer_distance_surface(spec: QuestionSpec, objs: Sequence[SceneObject],
                             view: FrameView) -> tuple[float, dict[str, Any]]:
    """Closest distance between the two referents' solid boxes.

    Parameters
    ----------
    spec : QuestionSpec
        The question.
    objs : sequence of SceneObject
        The referents' objects, in spec order.
    view : FrameView
        The image.

    Returns
    -------
    value : float
        Distance (m); 0 if the boxes touch or overlap.
    extra : dict
        Empty.
    """
    return surface_distance(objs[0].box, objs[1].box)[0], {}


def _answer_size_height(spec: QuestionSpec, objs: Sequence[SceneObject],
                        view: FrameView) -> tuple[float, dict[str, Any]]:
    """Height of the referent: its box's extent along the view's up direction.

    Parameters
    ----------
    spec : QuestionSpec
        The question.
    objs : sequence of SceneObject
        The referent's object.
    view : FrameView
        The image (its ``up`` is used).

    Returns
    -------
    value : float
        Height (m).
    extra : dict
        Empty.
    """
    return objs[0].box.extent_along(view.up), {}


def _answer_size_longest(spec: QuestionSpec, objs: Sequence[SceneObject],
                         view: FrameView) -> tuple[float, dict[str, Any]]:
    """Longest horizontal dimension of the referent (see :func:`longest_horizontal_dimension`).

    Parameters
    ----------
    spec : QuestionSpec
        The question.
    objs : sequence of SceneObject
        The referent's object.
    view : FrameView
        The image (its ``up`` is used).

    Returns
    -------
    value : float
        Length (m).
    extra : dict
        Empty.
    """
    return longest_horizontal_dimension(objs[0].box, view.up), {}


def _answer_ego_distance(spec: QuestionSpec, objs: Sequence[SceneObject],
                         view: FrameView) -> tuple[float, dict[str, Any]]:
    """Distance from the camera centre to the referent's nearest point.

    Parameters
    ----------
    spec : QuestionSpec
        The question.
    objs : sequence of SceneObject
        The referent's object.
    view : FrameView
        The image (its camera centre is used).

    Returns
    -------
    value : float
        Distance (m).
    extra : dict
        Empty.
    """
    return _camera_distance(objs[0].box, view.camera_center), {}


def _answer_relative_depth(spec: QuestionSpec, objs: Sequence[SceneObject],
                           view: FrameView) -> tuple[str, dict[str, Any]]:
    """Which referent is closer to the camera, by distance to its nearest point.

    Parameters
    ----------
    spec : QuestionSpec
        The question.
    objs : sequence of SceneObject
        The two referents' objects, in spec order.
    view : FrameView
        The image (its camera centre is used).

    Returns
    -------
    value : str
        The closer referent's expression.
    extra : dict
        ``margin``: absolute difference of the two distances (m); ``distances``: object id -> distance (m).
    """
    d = [_camera_distance(o.box, view.camera_center) for o in objs]
    closer = int(np.argmin(d))
    return spec.referents[closer].expression, {
        "margin": abs(d[0] - d[1]), "distances": {o.object_id: di for o, di in zip(objs, d)}}


def _answer_left_right(spec: QuestionSpec, objs: Sequence[SceneObject],
                       view: FrameView) -> tuple[str, dict[str, Any]]:
    """Whether the first referent is left or right of the second, by camera-frame x of the box centres.

    Parameters
    ----------
    spec : QuestionSpec
        The question.
    objs : sequence of SceneObject
        The two referents' objects, in spec order.
    view : FrameView
        The image (its world-to-camera pose is used).

    Returns
    -------
    value : str
        ``"left"`` or ``"right"``.
    extra : dict
        ``margin``: absolute difference of the centres' camera-frame x (m).
    """
    x = transform(view.T_cw, np.stack([o.box.center for o in objs]))[:, 0]
    return ("left" if x[0] < x[1] else "right"), {"margin": float(abs(x[0] - x[1]))}


def _answer_closest_instance(spec: QuestionSpec, objs: Sequence[SceneObject],
                             view: FrameView) -> tuple[str, dict[str, Any]]:
    """Which candidate instance is closest to the reference object, by surface distance.

    Parameters
    ----------
    spec : QuestionSpec
        The question; its first referent is the reference object, the rest are the candidates.
    objs : sequence of SceneObject
        The referents' objects, in spec order.
    view : FrameView
        The image.

    Returns
    -------
    value : str
        The closest candidate's expression.
    extra : dict
        ``margin``: second-closest minus closest distance (m); ``distances``: candidate id -> distance (m).
    """
    reference, candidates = objs[0], objs[1:]
    d = np.array([surface_distance(reference.box, c.box)[0] for c in candidates])
    order = np.argsort(d, kind="stable")
    return spec.referents[1 + int(order[0])].expression, {
        "margin": float(d[order[1]] - d[order[0]]),
        "distances": {c.object_id: float(di) for c, di in zip(candidates, d)}}


_Answerer = Callable[[QuestionSpec, Sequence[SceneObject], FrameView], tuple[Any, dict[str, Any]]]
_ANSWERERS: dict[str, _Answerer] = {
    "distance_center": _answer_distance_center,
    "distance_surface": _answer_distance_surface,
    "size_height": _answer_size_height,
    "size_longest": _answer_size_longest,
    "ego_distance": _answer_ego_distance,
    "relative_depth": _answer_relative_depth,
    "left_right": _answer_left_right,
    "closest_instance": _answer_closest_instance,
}


def _object_record(obj: SceneObject) -> dict[str, Any]:
    """The box an answer used, in JSON-friendly form.

    Parameters
    ----------
    obj : SceneObject
        The object.

    Returns
    -------
    dict
        ``object_id``, ``label``, and the world-frame ``center`` (3,), ``size`` (3,) full side lengths (m)
        and ``rotation`` (3, 3) with the box axes as columns, as nested lists.
    """
    return {"object_id": obj.object_id, "label": obj.label, "center": obj.box.center.tolist(),
            "size": obj.box.size.tolist(), "rotation": obj.box.rotation.tolist()}


def answer(spec: QuestionSpec, objects_by_id: Mapping[str, SceneObject], view: FrameView, *,
           in_view: Collection[str] | None = None) -> tuple[float | int | str, tuple[dict[str, Any], ...],
                                                            dict[str, Any]]:
    """Answer a spec from the given boxes.

    The same code answers from oracle boxes and from perception boxes; only ``objects_by_id`` differs.
    Referents are looked up by object id.

    Parameters
    ----------
    spec : QuestionSpec
        The question, normally from :func:`generate_specs`.
    objects_by_id : mapping of str to SceneObject
        Boxes from one source, by object id; must include every referent.
    view : FrameView
        The image the spec is about.
    in_view : collection of str, optional
        Required for ``count`` specs, ignored otherwise: ids of the objects this source places in the image
        (e.g. :func:`visible_ids`). The count is the number of those, among ``objects_by_id``, whose label
        maps to the counted phrase.

    Returns
    -------
    value : float, int or str
        Metres for numeric questions, an integer for counts, one of ``spec.choices`` for choices.
    objects_record : tuple of dict
        For each referent (for a count, each counted object), the box used (see :func:`_object_record`).
    extra : dict
        ``margin`` (m) for categorical answers, the distance from the nearest decision threshold, plus the
        per-object ``distances`` behind relative-depth and closest-instance answers. Empty otherwise.

    Raises
    ------
    MissingReferentError
        If a referent's id is not in ``objects_by_id``.
    ValueError
        If the spec belongs to another image, or a count spec is answered without ``in_view``.
    """
    if (spec.dataset, spec.scene, spec.camera_id, spec.frame) != (view.dataset, view.scene, view.camera_id,
                                                                  view.frame):
        raise ValueError(f"Spec {spec.id} is about another image than {view.scene}/{view.camera_id}/"
                         f"{view.frame}")
    if spec.qtype == "count":
        if in_view is None:
            raise ValueError("A count spec needs in_view: the ids this source places in the image")
        phrase = spec.params["phrase"]
        counted = [objects_by_id[i] for i in sorted(set(in_view) & set(objects_by_id))
                   if phrase_for(objects_by_id[i].label) == phrase]
        return len(counted), tuple(_object_record(o) for o in counted), {}
    missing = [r.object_id for r in spec.referents if r.object_id not in objects_by_id]
    if missing:
        raise MissingReferentError(f"Spec {spec.id} refers to objects with no box: {missing}")
    objs = [objects_by_id[r.object_id] for r in spec.referents]
    value, extra = _ANSWERERS[spec.qtype](spec, objs, view)
    return value, tuple(_object_record(o) for o in objs), extra


def make_items(specs: Sequence[QuestionSpec], objects_by_id: Mapping[str, SceneObject], view: FrameView,
               source: str, *, in_view: Collection[str] | None = None) -> list[QAItem]:
    """Answer specs about one image from one source's boxes.

    A spec whose referent is missing from ``objects_by_id`` (for perception: an object it did not find) gets
    no item; compare item ids with spec ids to find those.

    Parameters
    ----------
    specs : sequence of QuestionSpec
        Questions about ``view``.
    objects_by_id : mapping of str to SceneObject
        Boxes from one source, by object id.
    view : FrameView
        The image.
    source : str
        ``"oracle"`` or ``"auto"``, recorded on each item.
    in_view : collection of str, optional
        Ids of the objects this source places in the image; required if ``specs`` contains a count.

    Returns
    -------
    list of QAItem
        One item per answerable spec, in spec order.

    Raises
    ------
    ValueError
        As :func:`answer`.
    """
    items = []
    for spec in specs:
        try:
            value, record, extra = answer(spec, objects_by_id, view, in_view=in_view)
        except MissingReferentError:
            continue
        items.append(QAItem(spec, value, source, record, extra))
    return items


# --------------------------------------------------------------------------------------------------------
# Asking
# --------------------------------------------------------------------------------------------------------

def _make_spec(view: FrameView, qtype: str, referents: tuple[Referent, ...], config: QAConfig,
               params: dict[str, Any] | None = None) -> QuestionSpec:
    """Fill a template for one image.

    Parameters
    ----------
    view : FrameView
        The image.
    qtype : str
        One of :data:`QTYPES`.
    referents : tuple of Referent
        The objects asked about, in template order (for closest-instance: reference first, then candidates
        left to right; empty for a count).
    config : QAConfig
        Supplies the numeric tolerances.
    params : dict, optional
        Template values: ``{"phrase": ...}`` for count and closest-instance.

    Returns
    -------
    QuestionSpec
        The spec, with choices, unit and tolerance set for its answer type.
    """
    template_id, wording = TEMPLATES[qtype]
    params = params or {}
    exprs = [r.expression for r in referents]
    choices: tuple[str, ...] | None = None
    if qtype == "relative_depth":
        choices = tuple(exprs)
    elif qtype == "left_right":
        choices = ("left", "right")
    elif qtype == "closest_instance":
        choices = tuple(exprs[1:])
    fields = {"a": exprs[0] if exprs else "", "b": exprs[1] if len(exprs) > 1 else "",
              "phrase": params.get("phrase", ""), "plural": plural(params.get("phrase", "")),
              "choices": ", ".join(f'"{c}"' for c in choices or ())}
    if qtype in _NUMERIC_TOLERANCE_KIND:
        kind = _NUMERIC_TOLERANCE_KIND[qtype]
        tolerance = {"abs": getattr(config, f"{kind}_tol_abs"), "rel": getattr(config, f"{kind}_tol_rel")}
        answer_type, unit = "number", "m"
    else:
        tolerance, unit = {}, None
        answer_type = "count" if qtype == "count" else "choice"
    return QuestionSpec(template_id, qtype, view.dataset, view.scene, view.camera_id, view.frame,
                        view.image_ref, referents, wording.format(**fields), answer_type, unit=unit,
                        choices=choices, tolerance=tolerance, params=params)


def _referent(seen: ObjectInView, expressions: Mapping[str, str]) -> Referent:
    """The :class:`Referent` for an object that has an expression.

    Parameters
    ----------
    seen : ObjectInView
        The object.
    expressions : mapping of str to str
        Object id -> referring expression.

    Returns
    -------
    Referent
        Its id, dataset label and expression.
    """
    return Referent(seen.obj.object_id, seen.obj.label, expressions[seen.obj.object_id])


def _ids(*seen: ObjectInView) -> tuple[str, ...]:
    """Object ids of some objects in a view.

    Parameters
    ----------
    *seen : ObjectInView
        The objects.

    Returns
    -------
    tuple of str
        Their ids, in order.
    """
    return tuple(o.obj.object_id for o in seen)


def _single_object_candidates(view: FrameView, referents: Sequence[ObjectInView],
                              expressions: Mapping[str, str], objects: Mapping[str, SceneObject],
                              config: QAConfig, drops: list[Drop]) -> dict[str, list[QuestionSpec]]:
    """Height, longest-dimension and camera-distance questions for each referent.

    Parameters
    ----------
    view : FrameView
        The image.
    referents : sequence of ObjectInView
        Objects that may be asked about, sorted by id.
    expressions : mapping of str to str
        Object id -> referring expression.
    objects : mapping of str to SceneObject
        The boxes used to apply the filters, by id.
    config : QAConfig
        Thresholds.
    drops : list of Drop
        Receives one entry per rejected candidate.

    Returns
    -------
    dict of str to list of QuestionSpec
        Candidates by question type.
    """
    out: dict[str, list[QuestionSpec]] = {"size_height": [], "size_longest": [], "ego_distance": []}
    for o in referents:
        for qtype in out:
            spec = _make_spec(view, qtype, (_referent(o, expressions),), config)
            if qtype == "ego_distance":
                value = answer(spec, objects, view)[0]
                if value < config.min_distance:
                    drops.append(Drop(qtype, "distance_below_min", _ids(o),
                                      {"value": value, "min": config.min_distance}))
                    continue
            out[qtype].append(spec)
    return out


def _pair_candidates(view: FrameView, referents: Sequence[ObjectInView], expressions: Mapping[str, str],
                     objects: Mapping[str, SceneObject], config: QAConfig,
                     drops: list[Drop]) -> dict[str, list[QuestionSpec]]:
    """Distance, relative-depth and left/right questions for each pair of referents.

    Parameters
    ----------
    view : FrameView
        The image.
    referents : sequence of ObjectInView
        Objects that may be asked about, sorted by id; each pair is asked in that order.
    expressions : mapping of str to str
        Object id -> referring expression.
    objects : mapping of str to SceneObject
        The boxes used to apply the filters, by id.
    config : QAConfig
        Thresholds.
    drops : list of Drop
        Receives one entry per rejected candidate.

    Returns
    -------
    dict of str to list of QuestionSpec
        Candidates by question type.
    """
    out: dict[str, list[QuestionSpec]] = {"distance_center": [], "distance_surface": [], "relative_depth": [],
                                          "left_right": []}
    for a, b in combinations(referents, 2):
        refs = (_referent(a, expressions), _referent(b, expressions))
        for qtype in ("distance_center", "distance_surface"):
            spec = _make_spec(view, qtype, refs, config)
            value = answer(spec, objects, view)[0]
            if value < config.min_distance:
                drops.append(Drop(qtype, "distance_below_min", _ids(a, b),
                                  {"value": value, "min": config.min_distance}))
            else:
                out[qtype].append(spec)

        spec = _make_spec(view, "relative_depth", refs, config)
        extra = answer(spec, objects, view)[2]
        need = max(config.relative_depth_min_abs,
                   config.relative_depth_min_rel * min(extra["distances"].values()))
        if extra["margin"] < need:
            drops.append(Drop("relative_depth", "margin_below_min", _ids(a, b),
                              {"margin": extra["margin"], "min": need}))
        else:
            out["relative_depth"].append(spec)

        if phrase_for(a.obj.label) == phrase_for(b.obj.label):
            # Both are named by image position, so the expressions would give the answer away.
            drops.append(Drop("left_right", "same_phrase", _ids(a, b)))
            continue
        spec = _make_spec(view, "left_right", refs, config)
        value, _, extra = answer(spec, objects, view)
        if extra["margin"] < config.left_right_min_dx:
            drops.append(Drop("left_right", "margin_below_min", _ids(a, b),
                              {"margin": extra["margin"], "min": config.left_right_min_dx}))
            continue
        if config.left_right_require_image_agreement:
            centers = transform(view.T_cw, np.stack([objects[r.object_id].box.center for r in refs]))
            u = view.camera.project(centers)[0][:, 0]
            image_value = "left" if u[0] < u[1] else "right"
            if not np.all(np.isfinite(u)) or image_value != value:
                drops.append(Drop("left_right", "image_order_disagrees", _ids(a, b), {"u": u.tolist()}))
                continue
        out["left_right"].append(spec)
    return out


def _closest_instance_candidates(view: FrameView, referents: Sequence[ObjectInView],
                                 present: Mapping[str, list[ObjectInView]], expressions: Mapping[str, str],
                                 objects: Mapping[str, SceneObject], config: QAConfig,
                                 drops: list[Drop]) -> list[QuestionSpec]:
    """Closest-instance questions: for each phrase with several instances, which is closest to a referent.

    Every instance of the phrase that is in the image must be nameable and clearly visible, or the true
    answer could be one the question cannot offer.

    Parameters
    ----------
    view : FrameView
        The image.
    referents : sequence of ObjectInView
        Objects that may be asked about, sorted by id.
    present : mapping of str to list of ObjectInView
        Phrase -> instances in the image, left to right (from :func:`_present_by_phrase`).
    expressions : mapping of str to str
        Object id -> referring expression.
    objects : mapping of str to SceneObject
        The boxes used to apply the filters, by id.
    config : QAConfig
        Thresholds.
    drops : list of Drop
        Receives one entry per rejected candidate.

    Returns
    -------
    list of QuestionSpec
        The candidates.
    """
    referent_ids = set(_ids(*referents))
    out = []
    for phrase, instances in present.items():
        if len(instances) < 2:
            continue
        if not referent_ids.issuperset(_ids(*instances)):
            drops.append(Drop("closest_instance", "instance_not_referable", _ids(*instances),
                              {"phrase": phrase}))
            continue
        candidates = tuple(_referent(o, expressions) for o in instances)
        for r in referents:
            if phrase_for(r.obj.label) == phrase:
                continue
            spec = _make_spec(view, "closest_instance", (_referent(r, expressions), *candidates), config,
                              params={"phrase": phrase})
            extra = answer(spec, objects, view)[2]
            need = max(config.closest_min_abs, config.closest_min_rel * min(extra["distances"].values()))
            if extra["margin"] < need:
                drops.append(Drop("closest_instance", "margin_below_min", _ids(r, *instances),
                                  {"margin": extra["margin"], "min": need}))
            else:
                out.append(spec)
    return out


def _count_candidates(view: FrameView, present: Mapping[str, list[ObjectInView]], config: QAConfig,
                      drops: list[Drop]) -> list[QuestionSpec]:
    """Count questions, one per phrase in the image, unless an instance is borderline or the count too high.

    Parameters
    ----------
    view : FrameView
        The image.
    present : mapping of str to list of ObjectInView
        Phrase -> instances in the image (from :func:`_present_by_phrase`).
    config : QAConfig
        Thresholds.
    drops : list of Drop
        Receives one entry per rejected candidate.

    Returns
    -------
    list of QuestionSpec
        The candidates.
    """
    out = []
    for phrase, instances in present.items():
        borderline = [o for o in instances if o.visible_fraction < config.count_min_visible_fraction
                      or o.truncated_fraction > config.count_max_truncated_fraction]
        if borderline:
            drops.append(Drop("count", "borderline_instance", _ids(*borderline), {
                "phrase": phrase, "visible_fraction": [o.visible_fraction for o in borderline],
                "truncated_fraction": [o.truncated_fraction for o in borderline]}))
        elif len(instances) > config.count_max:
            drops.append(Drop("count", "count_above_max", _ids(*instances),
                              {"phrase": phrase, "count": len(instances), "max": config.count_max}))
        else:
            out.append(_make_spec(view, "count", (), config, params={"phrase": phrase}))
    return out


def _cap(view: FrameView, qtype: str, candidates: list[QuestionSpec], config: QAConfig, rng_seed: int,
         drops: list[Drop]) -> list[QuestionSpec]:
    """Keep at most ``config.max_per_type`` candidates of one type, chosen reproducibly.

    The RNG is seeded from ``rng_seed``, the image's identity and the question type, so a view's selection
    does not depend on which other views or types were generated before it.

    Parameters
    ----------
    view : FrameView
        The image.
    qtype : str
        The question type.
    candidates : list of QuestionSpec
        All candidates of that type, in a deterministic order.
    config : QAConfig
        Supplies ``max_per_type``.
    rng_seed : int
        Non-negative seed.
    drops : list of Drop
        Receives one entry per candidate left out.

    Returns
    -------
    list of QuestionSpec
        The kept candidates, in their original order.
    """
    if config.max_per_type is None or len(candidates) <= config.max_per_type:
        return candidates
    key = f"{view.dataset}|{view.scene}|{view.camera_id}|{view.frame}|{qtype}"
    rng = np.random.default_rng([rng_seed, zlib.crc32(key.encode())])
    keep = set(rng.choice(len(candidates), size=config.max_per_type, replace=False).tolist())
    for i, spec in enumerate(candidates):
        if i not in keep:
            drops.append(Drop(qtype, "capped", tuple(r.object_id for r in spec.referents), dict(spec.params)))
    return [spec for i, spec in enumerate(candidates) if i in keep]


def generate_specs(view: FrameView, seen: Sequence[ObjectInView], config: QAConfig | None = None,
                   rng_seed: int = 0, *, diagnostics: list[Drop] | None = None) -> list[QuestionSpec]:
    """The questions to ask about one image.

    Referents are objects that are clearly visible (``min_visible_fraction``, ``max_truncated_fraction``)
    and have an unambiguous expression (:func:`referring_expressions`). Candidates whose answer, computed
    from the boxes in ``seen``, is too close to zero or to a decision threshold are dropped (see
    :class:`QAConfig`). At most ``max_per_type`` questions of each type are kept.

    Parameters
    ----------
    view : FrameView
        The image.
    seen : sequence of ObjectInView
        Objects as the view sees them, normally with ground-truth boxes. Objects not listed are taken to be
        out of the image. The result does not depend on the order.
    config : QAConfig, optional
        Thresholds; defaults to ``QAConfig()``.
    rng_seed : int
        Non-negative seed for the per-type cap. The same inputs and seed give the same specs.
    diagnostics : list of Drop, optional
        If given, receives one :class:`Drop` per object or candidate left out, with the reason.

    Returns
    -------
    list of QuestionSpec
        Specs in :data:`QTYPES` order, and within a type in a deterministic order.
    """
    config = config or QAConfig()
    drops: list[Drop] = []
    seen = sorted(seen, key=lambda o: o.obj.object_id)
    expressions = referring_expressions(view, seen, config, drops)
    referents = []
    for o in seen:
        if not _is_present(o, config):
            continue
        if not _is_clearly_visible(o, config):
            drops.append(Drop("referent", "not_clearly_visible", _ids(o), {
                "visible_fraction": o.visible_fraction, "truncated_fraction": o.truncated_fraction}))
        elif o.obj.object_id in expressions:
            referents.append(o)
    objects = {o.obj.object_id: o.obj for o in seen}
    present = _present_by_phrase(seen, config)

    candidates: dict[str, list[QuestionSpec]] = {
        **_single_object_candidates(view, referents, expressions, objects, config, drops),
        **_pair_candidates(view, referents, expressions, objects, config, drops),
        "closest_instance": _closest_instance_candidates(view, referents, present, expressions, objects,
                                                         config, drops),
        "count": _count_candidates(view, present, config, drops),
    }
    specs = [spec for qtype in QTYPES
             for spec in _cap(view, qtype, candidates[qtype], config, rng_seed, drops)]
    if diagnostics is not None:
        diagnostics.extend(drops)
    return specs
