"""Shared data types: objects in a scene, one view of a scene, and an object as one view sees it.

Every dataset loader produces these, and both the question generator and the perception pipeline consume
them. The oracle and the automatic pipeline differ only in where the boxes come from. Geometry conventions
are those of :mod:`measured_space.geometry`: OpenCV camera axes, ``T_wc`` camera-to-world, and boxes with
their axes as rotation columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from measured_space.geometry import Array, Camera, OrientedBox, invert_pose


def _default_up() -> Array:
    """Return the default world up direction, +z.

    Returns
    -------
    numpy.ndarray, shape (3,)
        ``[0, 0, 1]``.
    """
    return np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class SceneObject:
    """An object with a 3D box in the scene's world frame.

    Parameters
    ----------
    object_id : str
        Identifier, unique within a scene and stable over time for objects that move.
    label : str
        The dataset's class name.
    box : OrientedBox
        The object's box in the world frame, metres.
    source : str, default "oracle"
        Where the box came from: ``"oracle"`` (ground truth) or ``"auto"`` (perception).
    """

    object_id: str
    label: str
    box: OrientedBox
    source: str = "oracle"


@dataclass
class FrameView:
    """One image of a scene, with its camera and pose.

    Parameters
    ----------
    dataset : str
        Dataset name, e.g. ``"arkitscenes"``.
    scene : str
        Scene identifier within the dataset.
    camera_id : str
        Camera identifier within the scene.
    frame : str
        Image identifier within the scene (a timestamp or frame index, as text).
    camera : Camera
        Intrinsics at the image's resolution.
    T_wc : numpy.ndarray, shape (4, 4)
        Camera-to-world pose.
    image_ref : str
        The RGB image's location relative to the dataset folder: a file path, or ``"<archive>::<member>"``
        for a file read straight out of an archive.
    up : numpy.ndarray, shape (3,), default +z
        World up direction (against gravity), used for heights and "above/below".
    """

    dataset: str
    scene: str
    camera_id: str
    frame: str
    camera: Camera
    T_wc: Array
    image_ref: str
    up: Array = field(default_factory=_default_up)

    @property
    def T_cw(self) -> Array:
        """numpy.ndarray, shape (4, 4): the world-to-camera pose."""
        return invert_pose(self.T_wc)

    @property
    def camera_center(self) -> Array:
        """numpy.ndarray, shape (3,): the camera's optical centre in the world frame, metres."""
        return self.T_wc[:3, 3].copy()


@dataclass(frozen=True)
class ObjectInView:
    """An object as one view sees it.

    Parameters
    ----------
    obj : SceneObject
        The object.
    visible_fraction : float
        Share of the object's camera-facing surface samples that lie inside the image, in front of the
        camera, and are not occluded according to depth.
    truncated_fraction : float
        Share of the camera-facing surface samples that fall outside the image or behind the camera.
    bbox2d : tuple of float or None
        ``(xmin, ymin, xmax, ymax)`` pixel box of the visible samples, or ``None`` if none are visible.
    """

    obj: SceneObject
    visible_fraction: float
    truncated_fraction: float
    bbox2d: tuple[float, float, float, float] | None
