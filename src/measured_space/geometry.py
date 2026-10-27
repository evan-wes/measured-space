"""Rigid transforms, pinhole projection and oriented 3D boxes.

Every other module builds on these primitives, so their conventions are fixed here once:

- Points are rows of an ``(N, 3)`` array, in metres.
- A pose ``T_ab`` is a 4x4 homogeneous matrix taking coordinates in frame ``b`` to frame ``a``,
  ``x_a = T_ab @ x_b``. ``T_wc`` (camera to world) therefore places camera-frame points in the world.
- The camera frame is OpenCV's: x right, y down, z forward. Pixel centres sit at integer coordinates.
- An :class:`OrientedBox` holds its centre, its **full** side lengths, and a rotation whose **columns** are
  the box's local x, y, z axes expressed in the parent frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation, Slerp

Array = NDArray[np.float64]
"""Alias for a float64 numpy array."""


def make_pose(rotation: ArrayLike, translation: ArrayLike) -> Array:
    """Build a 4x4 rigid pose.

    Parameters
    ----------
    rotation : array_like, shape (3, 3)
        Rotation matrix.
    translation : array_like, shape (3,)
        Translation in metres.

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        Homogeneous pose ``[[R, t], [0, 1]]``.
    """
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def invert_pose(pose: Array) -> Array:
    """Invert a rigid pose without a general matrix inverse.

    Parameters
    ----------
    pose : numpy.ndarray, shape (4, 4)
        Rigid pose ``T_ab``.

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        The inverse pose ``T_ba``.
    """
    rotation, translation = pose[:3, :3], pose[:3, 3]
    return make_pose(rotation.T, -rotation.T @ translation)


def transform(pose: Array, points: ArrayLike) -> Array:
    """Apply a rigid pose to points.

    Parameters
    ----------
    pose : numpy.ndarray, shape (4, 4)
        Rigid pose ``T_ab``.
    points : array_like, shape (N, 3) or (3,)
        Points in frame ``b``, metres.

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        The points in frame ``a``.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ pose[:3, :3].T + pose[:3, 3]


def axis_angle_to_matrix(axis_angle: ArrayLike) -> Array:
    """Convert an axis-angle (rotation) vector to a rotation matrix, as OpenCV's Rodrigues does.

    Parameters
    ----------
    axis_angle : array_like, shape (3,)
        Rotation axis scaled by the rotation angle in radians.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Rotation matrix.
    """
    return Rotation.from_rotvec(np.asarray(axis_angle, dtype=np.float64)).as_matrix()


def interpolate_poses(times: ArrayLike, poses: Array, query: ArrayLike) -> tuple[Array, NDArray[np.bool_]]:
    """Interpolate poses in time: rotation by spherical linear interpolation, translation linearly.

    Query times outside ``[times[0], times[-1]]`` are not extrapolated.

    Parameters
    ----------
    times : array_like, shape (N,)
        Strictly increasing sample times, seconds.
    poses : numpy.ndarray, shape (N, 4, 4)
        Rigid poses at ``times``.
    query : array_like, shape (M,) or scalar
        Times at which poses are wanted, seconds.

    Returns
    -------
    poses_at_query : numpy.ndarray, shape (M, 4, 4)
        Interpolated poses; all-NaN for query times outside the sampled span.
    valid : numpy.ndarray, shape (M,), bool
        True where the query time lies inside the sampled span.
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
    """A pinhole camera without lens distortion.

    Parameters
    ----------
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal point in pixels (pixel centres at integer coordinates).
    width, height : int
        Image size in pixels.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> Array:
        """numpy.ndarray, shape (3, 3): the intrinsic matrix."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def project(self, points_cam: ArrayLike) -> tuple[Array, Array]:
        """Project camera-frame points to pixels.

        Parameters
        ----------
        points_cam : array_like, shape (N, 3)
            Points in the camera frame, metres.

        Returns
        -------
        uv : numpy.ndarray, shape (N, 2)
            Pixel coordinates ``(u, v)``; NaN for points at or behind the camera (``z <= 0``).
        z : numpy.ndarray, shape (N,)
            Planar depth of each point, metres.
        """
        points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = points_cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(z > 0, self.fx * points_cam[:, 0] / z + self.cx, np.nan)
            v = np.where(z > 0, self.fy * points_cam[:, 1] / z + self.cy, np.nan)
        return np.stack([u, v], axis=1), z

    def backproject(self, u: ArrayLike, v: ArrayLike, z: ArrayLike) -> Array:
        """Lift pixels with planar depth to camera-frame points.

        Parameters
        ----------
        u, v : array_like, shape (N,)
            Pixel coordinates.
        z : array_like, shape (N,)
            Planar depth, metres.

        Returns
        -------
        numpy.ndarray, shape (N, 3)
            Points in the camera frame, metres.
        """
        u, v, z = (np.asarray(a, dtype=np.float64).ravel() for a in (u, v, z))
        return np.stack([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z], axis=1)

    def in_image(self, uv: Array) -> NDArray[np.bool_]:
        """Test whether pixel coordinates fall inside the image.

        Parameters
        ----------
        uv : numpy.ndarray, shape (N, 2)
            Pixel coordinates; NaN counts as outside.

        Returns
        -------
        numpy.ndarray, shape (N,), bool
            True for pixels within ``[-0.5, width - 0.5) x [-0.5, height - 0.5)``.
        """
        u, v = uv[:, 0], uv[:, 1]
        with np.errstate(invalid="ignore"):
            return (u >= -0.5) & (u < self.width - 0.5) & (v >= -0.5) & (v < self.height - 0.5)

    def scaled(self, width: int, height: int) -> Camera:
        """Return the same camera resampled to another image size.

        Parameters
        ----------
        width, height : int
            New image size in pixels.

        Returns
        -------
        Camera
            Camera whose pixels cover the same rays at the new resolution.
        """
        sx, sy = width / self.width, height / self.height
        return Camera(self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5,
                      width, height)


# The 8 corners of the unit cube [-1/2, 1/2]^3, and its 6 faces as (axis, sign).
_UNIT_CORNERS = np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)])
_FACES = [(axis, sign) for axis in range(3) for sign in (-1.0, 1.0)]


@dataclass(frozen=True)
class OrientedBox:
    """A solid 3D box with arbitrary orientation.

    Parameters
    ----------
    center : array_like, shape (3,)
        Box centre in the parent frame, metres.
    size : array_like, shape (3,)
        Full side lengths along the box's own x, y, z axes, metres.
    rotation : array_like, shape (3, 3)
        Rotation whose columns are the box axes expressed in the parent frame.
    """

    center: Array
    size: Array
    rotation: Array

    def __post_init__(self) -> None:
        """Coerce the fields to float64 arrays of the documented shapes."""
        object.__setattr__(self, "center", np.asarray(self.center, dtype=np.float64).reshape(3))
        object.__setattr__(self, "size", np.asarray(self.size, dtype=np.float64).reshape(3))
        object.__setattr__(self, "rotation", np.asarray(self.rotation, dtype=np.float64).reshape(3, 3))

    def corners(self) -> Array:
        """Return the box's corners.

        Returns
        -------
        numpy.ndarray, shape (8, 3)
            Corner coordinates in the parent frame.
        """
        return self.from_local(_UNIT_CORNERS * self.size)

    def to_local(self, points: ArrayLike) -> Array:
        """Express points in the box's own frame (origin at the centre, axes along its sides).

        Parameters
        ----------
        points : array_like, shape (N, 3)
            Points in the parent frame.

        Returns
        -------
        numpy.ndarray, shape (N, 3)
            The points in box coordinates.
        """
        return (np.asarray(points, dtype=np.float64).reshape(-1, 3) - self.center) @ self.rotation

    def from_local(self, points_local: ArrayLike) -> Array:
        """Express box-frame points in the parent frame.

        Parameters
        ----------
        points_local : array_like, shape (N, 3)
            Points in box coordinates.

        Returns
        -------
        numpy.ndarray, shape (N, 3)
            The points in the parent frame.
        """
        return np.asarray(points_local, dtype=np.float64).reshape(-1, 3) @ self.rotation.T + self.center

    def contains(self, points: ArrayLike, margin: float = 0.0) -> NDArray[np.bool_]:
        """Test whether points lie inside the box grown by ``margin`` on every side.

        Parameters
        ----------
        points : array_like, shape (N, 3)
            Points in the parent frame.
        margin : float, default 0.0
            Growth of each half-extent, metres (negative shrinks the box).

        Returns
        -------
        numpy.ndarray, shape (N,), bool
            True for points inside the grown box.
        """
        local = self.to_local(points)
        return np.all(np.abs(local) <= self.size / 2 + margin, axis=1)

    def closest_point(self, points: ArrayLike) -> Array:
        """Find the point of the solid box nearest to each input point.

        Parameters
        ----------
        points : array_like, shape (N, 3)
            Query points in the parent frame.

        Returns
        -------
        numpy.ndarray, shape (N, 3)
            Nearest points of the box (the query point itself if it is inside).
        """
        half = self.size / 2
        return self.from_local(np.clip(self.to_local(points), -half, half))

    def distance_to_surface(self, points: ArrayLike) -> Array:
        """Compute the unsigned distance from points to the box's boundary surface.

        Parameters
        ----------
        points : array_like, shape (N, 3)
            Query points in the parent frame.

        Returns
        -------
        numpy.ndarray, shape (N,)
            Distance to the nearest face, metres, for points inside or outside the box.
        """
        local = self.to_local(points)
        half = self.size / 2
        outside = np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=1)
        inside = np.min(half - np.abs(local), axis=1)
        return np.where(np.all(np.abs(local) <= half, axis=1), inside, outside)

    def transformed(self, pose: Array) -> OrientedBox:
        """Express the box in another frame.

        Parameters
        ----------
        pose : numpy.ndarray, shape (4, 4)
            Rigid pose ``T_ab`` from the box's current frame ``b`` to frame ``a``.

        Returns
        -------
        OrientedBox
            The same box in frame ``a``.
        """
        return OrientedBox(transform(pose, self.center[None])[0], self.size, pose[:3, :3] @ self.rotation)

    def extent_along(self, direction: ArrayLike) -> float:
        """Length of the box's projection onto a direction (its height, for the up vector).

        Parameters
        ----------
        direction : array_like, shape (3,)
            Direction in the parent frame; it need not be normalised.

        Returns
        -------
        float
            Extent of the box along the direction, metres.
        """
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / np.linalg.norm(direction)
        return float(np.sum(np.abs(self.rotation.T @ direction) * self.size))

    def sample_surface(self, spacing: float) -> tuple[Array, Array]:
        """Sample points on the six faces on a regular grid.

        Parameters
        ----------
        spacing : float
            Target grid spacing, metres; each face gets at least a 2x2 grid including its edges.

        Returns
        -------
        points : numpy.ndarray, shape (N, 3)
            Surface points in the parent frame.
        normals : numpy.ndarray, shape (N, 3)
            Outward unit normal of each point's face.
        """
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
    """Distance between two boxes' centres.

    Parameters
    ----------
    a, b : OrientedBox
        Boxes in the same frame.

    Returns
    -------
    float
        Euclidean distance between the centres, metres.
    """
    return float(np.linalg.norm(a.center - b.center))


def surface_distance(a: OrientedBox, b: OrientedBox, iterations: int = 500,
                     tol: float = 1e-9) -> tuple[float, Array, Array]:
    """Closest distance between two solid boxes, with a pair of points that attains it.

    Alternating projections between two closed convex sets converge to a closest pair
    (Cheney & Goldstein, 1959). Overlapping boxes converge to a shared point, giving distance 0.

    Parameters
    ----------
    a, b : OrientedBox
        Boxes in the same frame.
    iterations : int, default 500
        Maximum number of projection rounds.
    tol : float, default 1e-9
        Stop when an iterate moves less than this, metres.

    Returns
    -------
    distance : float
        Closest distance between the solids, metres (0 if they intersect).
    point_a : numpy.ndarray, shape (3,)
        Closest point on ``a``.
    point_b : numpy.ndarray, shape (3,)
        Closest point on ``b``.
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
