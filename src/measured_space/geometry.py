"""Rigid transforms, pinhole projection and oriented 3D boxes.

Conventions used throughout the package:

- Points are rows of an ``(N, 3)`` array.
- A pose ``T_ab`` is a 4x4 matrix taking coordinates in frame ``b`` to frame ``a``: ``x_a = T_ab x_b``.
  ``T_wc`` (camera to world) therefore places camera-frame points in the world.
- The camera frame is OpenCV's: x right, y down, z forward.
- An :class:`OrientedBox` holds its centre, its **full** side lengths, and a rotation whose **columns**
  are the box's local x, y, z axes expressed in the parent frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

Array = NDArray[np.float64]


def make_pose(rotation: Array, translation: Array) -> Array:
    """A 4x4 pose from a 3x3 rotation and a translation."""
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def invert_pose(pose: Array) -> Array:
    """The inverse of a rigid 4x4 pose, without a general matrix inverse."""
    rotation, translation = pose[:3, :3], pose[:3, 3]
    return make_pose(rotation.T, -rotation.T @ translation)


def transform(pose: Array, points: Array) -> Array:
    """Apply a 4x4 rigid pose to ``(N, 3)`` points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ pose[:3, :3].T + pose[:3, 3]


def axis_angle_to_matrix(axis_angle: Array) -> Array:
    """Rotation matrix for an axis-angle vector (radians), as in OpenCV's Rodrigues."""
    return Rotation.from_rotvec(np.asarray(axis_angle, dtype=np.float64)).as_matrix()


def interpolate_poses(times: Array, poses: Array, query: Array) -> tuple[Array, NDArray[np.bool_]]:
    """Poses at ``query`` times: rotation by slerp, translation linearly, between the two nearest samples.

    ``times`` must be increasing and ``poses`` has shape ``(N, 4, 4)``. Query times outside
    ``[times[0], times[-1]]`` are not extrapolated: their pose is NaN and their entry in the returned
    mask is False.
    """
    times = np.asarray(times, dtype=np.float64)
    query = np.atleast_1d(np.asarray(query, dtype=np.float64))
    valid = (query >= times[0]) & (query <= times[-1])
    out = np.full((len(query), 4, 4), np.nan)
    if valid.any():
        slerp = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))
        rotations = slerp(query[valid]).as_matrix()
        translations = np.stack([np.interp(query[valid], times, poses[:, i, 3]) for i in range(3)], axis=1)
        out[valid] = np.eye(4)
        out[valid, :3, :3] = rotations
        out[valid, :3, 3] = translations
    return out, valid


@dataclass(frozen=True)
class Camera:
    """A pinhole camera without distortion. Pixel centres are at integer coordinates."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> Array:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def project(self, points_cam: Array) -> tuple[Array, Array]:
        """Pixel coordinates ``(N, 2)`` and depth ``z`` ``(N,)`` of camera-frame points.

        Points at or behind the camera (``z <= 0``) get NaN pixel coordinates.
        """
        points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = points_cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(z > 0, self.fx * points_cam[:, 0] / z + self.cx, np.nan)
            v = np.where(z > 0, self.fy * points_cam[:, 1] / z + self.cy, np.nan)
        return np.stack([u, v], axis=1), z

    def backproject(self, u: Array, v: Array, z: Array) -> Array:
        """Camera-frame points for pixels ``(u, v)`` at planar depth ``z``."""
        u, v, z = (np.asarray(a, dtype=np.float64).ravel() for a in (u, v, z))
        return np.stack([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z], axis=1)

    def in_image(self, uv: Array) -> NDArray[np.bool_]:
        """Whether pixel coordinates fall inside the image (NaN counts as outside)."""
        u, v = uv[:, 0], uv[:, 1]
        with np.errstate(invalid="ignore"):
            return (u >= -0.5) & (u < self.width - 0.5) & (v >= -0.5) & (v < self.height - 0.5)

    def scaled(self, width: int, height: int) -> "Camera":
        """The same camera resampled to another resolution."""
        sx, sy = width / self.width, height / self.height
        return Camera(self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5,
                      width, height)


# The 8 corners of the unit cube [-1/2, 1/2]^3, and its 6 faces as (axis, sign).
_UNIT_CORNERS = np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)])
_FACES = [(axis, sign) for axis in range(3) for sign in (-1.0, 1.0)]


@dataclass(frozen=True)
class OrientedBox:
    """A 3D box: ``center`` (3,), full side lengths ``size`` (3,), ``rotation`` (3, 3) with axes as columns."""

    center: Array
    size: Array
    rotation: Array

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=np.float64).reshape(3))
        object.__setattr__(self, "size", np.asarray(self.size, dtype=np.float64).reshape(3))
        object.__setattr__(self, "rotation", np.asarray(self.rotation, dtype=np.float64).reshape(3, 3))

    def corners(self) -> Array:
        """The 8 corners, ``(8, 3)``."""
        return self.from_local(_UNIT_CORNERS * self.size)

    def to_local(self, points: Array) -> Array:
        """Points in the box's own frame (origin at the centre, axes along its sides)."""
        return (np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.center) @ self.rotation

    def from_local(self, points_local: Array) -> Array:
        return np.asarray(points_local, dtype=np.float64).reshape(-1, 3) @ self.rotation.T + self.center

    def contains(self, points: Array, margin: float = 0.0) -> NDArray[np.bool_]:
        """Whether points lie inside the box grown by ``margin`` on every side."""
        local = self.to_local(points)
        return np.all(np.abs(local) <= self.size / 2 + margin, axis=1)

    def closest_point(self, points: Array) -> Array:
        """The point of the (solid) box nearest to each input point."""
        half = self.size / 2
        return self.from_local(np.clip(self.to_local(points), -half, half))

    def distance_to_surface(self, points: Array) -> Array:
        """Unsigned distance from each point to the box's boundary surface."""
        local = self.to_local(points)
        half = self.size / 2
        outside = np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=1)
        inside = np.min(half - np.abs(local), axis=1)
        return np.where(np.all(np.abs(local) <= half, axis=1), inside, outside)

    def transformed(self, pose: Array) -> "OrientedBox":
        """The box expressed in another frame, via a rigid pose."""
        return OrientedBox(transform(pose, self.center[None])[0], self.size, pose[:3, :3] @ self.rotation)

    def extent_along(self, direction: Array) -> float:
        """Length of the box's projection onto a unit direction (its height, for the up vector)."""
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
        return float(np.sum(np.abs(self.rotation.T @ direction) * self.size))

    def sample_surface(self, spacing: float) -> tuple[Array, Array]:
        """Points on the six faces on a grid of about ``spacing`` metres, with each point's outward normal."""
        points, normals = [], []
        for axis, sign in _FACES:
            a, b = [i for i in range(3) if i != axis]
            na = max(2, int(np.ceil(self.size[a] / spacing)) + 1)
            nb = max(2, int(np.ceil(self.size[b] / spacing)) + 1)
            ga, gb = np.meshgrid(np.linspace(-0.5, 0.5, na), np.linspace(-0.5, 0.5, nb), indexing="ij")
            local = np.zeros((ga.size, 3))
            local[:, a] = ga.ravel() * self.size[a]
            local[:, b] = gb.ravel() * self.size[b]
            local[:, axis] = sign * self.size[axis] / 2
            points.append(self.from_local(local))
            normals.append(np.repeat((sign * self.rotation[:, axis])[None], len(local), axis=0))
        return np.concatenate(points), np.concatenate(normals)


def center_distance(a: OrientedBox, b: OrientedBox) -> float:
    """Euclidean distance between two boxes' centres."""
    return float(np.linalg.norm(a.center - b.center))


def surface_distance(a: OrientedBox, b: OrientedBox, iterations: int = 500, tol: float = 1e-9) -> tuple[float, Array, Array]:
    """Closest distance between two solid boxes, with a closest pair of points.

    Alternating projections between two closed convex sets converge to a closest pair
    (Cheney & Goldstein, 1959); overlapping boxes converge to a shared point, i.e. distance 0.
    """
    x = b.center
    for _ in range(iterations):
        pa = a.closest_point(x[None])[0]
        pb = b.closest_point(pa[None])[0]
        if np.linalg.norm(pb - x) < tol:
            x = pb
            break
        x = pb
    pa = a.closest_point(x[None])[0]
    return float(np.linalg.norm(pa - x)), pa, x
