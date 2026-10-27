"""How much of an object a view sees: sample its camera-facing surface, project, and test against depth.

A surface sample counts as **visible** when three things hold:
- it projects inside the image;
- it lies in front of the camera;
- the depth map at its pixel is not nearer than the sample by more than a tolerance (nothing occludes it).

Samples with no valid depth (0 or NaN) count as not visible. This is conservative, and the share of such
samples is reported so that a caller can tell occlusion from missing depth.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from measured_space.geometry import Array
from measured_space.scene import FrameView, ObjectInView, SceneObject


def depth_tolerance(z: ArrayLike, absolute: float = 0.05, relative: float = 0.03) -> Array:
    """Allowed disagreement between a surface sample's depth and the depth map.

    Parameters
    ----------
    z : array_like, shape (N,)
        Sample depths, metres.
    absolute : float, default 0.05
        Absolute bound, metres.
    relative : float, default 0.03
        Relative bound, as a fraction of ``z``.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The larger of the two bounds at each depth, metres.
    """
    return np.maximum(absolute, relative * np.asarray(z, dtype=np.float64))


def object_in_view(view: FrameView, obj: SceneObject, depth: Array | None, *, spacing: float = 0.05,
                   min_z: float = 0.05) -> tuple[ObjectInView, dict[str, float]]:
    """Measure how much of an object a view sees.

    Parameters
    ----------
    view : FrameView
        The image, its camera and its pose.
    obj : SceneObject
        The object, with its box in the world frame.
    depth : numpy.ndarray, shape (H, W), or None
        Planar depth map in metres (0 or NaN = invalid). It may have a different resolution from the
        camera; pixel coordinates are rescaled to it. With ``None``, only geometry (image bounds, being in
        front of the camera) decides visibility.
    spacing : float, default 0.05
        Surface sampling grid spacing, metres.
    min_z : float, default 0.05
        Samples nearer than this to the camera plane count as not visible, metres.

    Returns
    -------
    seen : ObjectInView
        Visible and truncated fractions, and the 2D box of the visible samples.
    diagnostics : dict of str to float
        ``"n_samples"``: number of camera-facing samples. ``"no_depth_fraction"``: share of them that project
        into the image but have no valid depth.
    """
    box_cam = obj.box.transformed(view.T_cw)
    points, normals = box_cam.sample_surface(spacing)
    facing = np.einsum("ij,ij->i", normals, -points) > 0          # camera sits at the origin
    points = points[facing]
    n = len(points)
    if n == 0:
        return ObjectInView(obj, 0.0, 1.0, None), {"n_samples": 0, "no_depth_fraction": 0.0}
    uv, z = view.camera.project(points)
    inside = view.camera.in_image(uv) & (z > min_z)
    visible = inside.copy()
    no_depth = np.zeros(n, dtype=bool)
    if depth is not None and inside.any():
        h, w = depth.shape[:2]
        sx, sy = w / view.camera.width, h / view.camera.height
        cols = np.clip(np.floor((uv[inside, 0] + 0.5) * sx).astype(int), 0, w - 1)
        rows = np.clip(np.floor((uv[inside, 1] + 0.5) * sy).astype(int), 0, h - 1)
        d = depth[rows, cols].astype(np.float64)
        valid = np.isfinite(d) & (d > 0)
        unoccluded = valid & (d >= z[inside] - depth_tolerance(z[inside]))
        visible[inside] = unoccluded
        no_depth[inside] = ~valid
    bbox = None
    if visible.any():
        u, v = uv[visible, 0], uv[visible, 1]
        bbox = (float(u.min()), float(v.min()), float(u.max()), float(v.max()))
    result = ObjectInView(obj, float(visible.mean()), float((~inside).mean()), bbox)
    return result, {"n_samples": n, "no_depth_fraction": float(no_depth.mean())}
