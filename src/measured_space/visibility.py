"""How much of an object a view sees: sample its camera-facing surface, project, and test against depth.

A surface sample counts as **visible** when it projects inside the image, lies in front of the camera, and
the depth map at its pixel is not nearer than the sample by more than a tolerance (nothing occludes it).
Samples with no valid depth (0 or NaN) count as not visible: this is conservative, and the fraction of such
samples is reported so a loader can tell occlusion from missing depth.
"""

from __future__ import annotations

import numpy as np

from measured_space.geometry import Array
from measured_space.scene import FrameView, ObjectInView, SceneObject


def depth_tolerance(z: Array, absolute: float = 0.05, relative: float = 0.03) -> Array:
    """Allowed depth disagreement (m) at depth ``z``: the larger of an absolute and a relative bound."""
    return np.maximum(absolute, relative * np.asarray(z))


def object_in_view(view: FrameView, obj: SceneObject, depth: Array | None, *, spacing: float = 0.05,
                   min_z: float = 0.05) -> tuple[ObjectInView, dict[str, float]]:
    """Visibility of ``obj`` in ``view`` against a planar depth map (metres), or geometry alone if ``None``.

    ``depth`` may have a different resolution from the camera; pixel coordinates are rescaled to it.
    Returns the :class:`ObjectInView` and diagnostics: the number of camera-facing samples, and the fraction
    that project into the image but have no valid depth.
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
