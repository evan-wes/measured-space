"""Shared data types: objects in a scene, one view of a scene, and an object as seen in a view.

Every dataset loader produces these, the question generator and the perception pipeline consume them, and
the oracle and the automatic pipeline differ only in where the boxes come from. Geometry conventions are
those of :mod:`measured_space.geometry` (OpenCV camera axes, ``T_wc`` camera-to-world, boxes with axes as
columns).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from measured_space.geometry import Array, Camera, OrientedBox, invert_pose


@dataclass(frozen=True)
class SceneObject:
    """An object with a 3D box in the scene's world frame.

    ``object_id`` is unique within a scene (and stable over time for objects that move). ``label`` is the
    dataset's class name. ``source`` says where the box came from: ``"oracle"`` (ground truth) or ``"auto"``
    (perception).
    """

    object_id: str
    label: str
    box: OrientedBox
    source: str = "oracle"


@dataclass
class FrameView:
    """One image of a scene, with its camera and pose.

    ``frame`` identifies the image within the scene (a timestamp or frame index, as text). ``image_ref``
    locates the RGB image relative to the data root (a file path, or ``"<zip>::<member>"`` for a file read
    straight out of an archive). ``up`` is the world's up direction (gravity), used for heights and
    "above/below".
    """

    dataset: str
    scene: str
    camera_id: str
    frame: str
    camera: Camera
    T_wc: Array
    image_ref: str
    up: Array = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    @property
    def T_cw(self) -> Array:
        return invert_pose(self.T_wc)

    @property
    def camera_center(self) -> Array:
        return self.T_wc[:3, 3].copy()


@dataclass(frozen=True)
class ObjectInView:
    """An object as one view sees it.

    ``visible_fraction``: share of the object's camera-facing surface samples that are inside the image and
    not occluded (by depth). ``truncated_fraction``: share of those samples that fall outside the image.
    ``bbox2d``: ``(xmin, ymin, xmax, ymax)`` pixel box of the projected visible samples, or ``None``.
    """

    obj: SceneObject
    visible_fraction: float
    truncated_fraction: float
    bbox2d: tuple[float, float, float, float] | None
