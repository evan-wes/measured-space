"""ARKitScenes: 3D object boxes, ARKit poses and laser-depth frames, as :mod:`measured_space.scene` types.

ARKitScenes is the project's real-sensor dataset. For each validation video with laser depth (the results use
106), the dataset's "upsampling" archive holds 2-69 filtered frames (median 17), with the RGB image (``wide``), a
depth map rendered from a Faro laser scan (``highres_depth``), the iPad LiDAR depth (``lowres_depth``) and its
confidence. The 3D boxes were annotated on the iPad's reconstruction (the ARKit mesh; ARKitScenes paper, Sec.
3.2); the laser depth is independent of them and is used for visibility and to check the conventions below.
Everything is read straight out of the downloaded archives; nothing is extracted.

Conventions, each checked against the laser depth (``scripts/check_arkitscenes_oracle.py``):

- Trajectory rows ``t rx ry rz tx ty tz`` are world-to-camera poses (axis-angle in radians, translation in
  metres). Their inverse is ``T_wc``, and the camera axes are already OpenCV's (x right, y down, z forward).
- ``obbAligned.normalizedAxes`` reshaped row-major to 3x3 has the box axes as **rows**, so
  :attr:`OrientedBox.rotation` (axes as columns) is its transpose. ``axesLengths`` are full side lengths.
- The world frame is gravity-aligned with z up.
- ``wide`` frame timestamps share the trajectory's clock. Poses (10 Hz) are interpolated to each frame, and
  frames outside the trajectory's time span, or between samples too far apart, are dropped.
- ``sky_direction`` (the dataset's metadata) names the side of the image that the sky, i.e. world up, is on.

Views describe the **upright** image a person or a VLM would see, not the sensor's sideways one: the stored
images are turned by a multiple of 90 degrees (chosen from the poses, so that world up points up the image),
and each view's camera and ``T_wc`` are those of the turned image. Image-frame notions such as left/right then
follow from camera x as usual. ``list_frames(..., upright=False)`` gives the sensor orientation instead.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from measured_space.config import dataset_dir
from measured_space.geometry import Array, Camera, OrientedBox, axis_angle_to_matrix, interpolate_poses, invert_pose
from measured_space.scene import FrameView, SceneObject

DATASET = "arkitscenes"
CAMERA_ID = "wide"
SENSOR_CAMERA_ID = "wide_sensor"          # views in the sensor's own (unrotated) orientation

VIDEO_IDS: tuple[str, ...] = ("42444946", "42444966", "42445021", "42445028", "42445429", "42446100")
"""The validation videos that have laser depth (``is_in_upsampling`` in the dataset's metadata)."""

CLASSES: tuple[str, ...] = ("cabinet", "refrigerator", "shelf", "stove", "bed", "sink", "washer", "toilet",
                            "bathtub", "oven", "dishwasher", "fireplace", "stool", "chair", "table", "tv_monitor",
                            "sofa")

WORLD_UP = np.array([0.0, 0.0, 1.0])

SKY_DIRECTION: dict[str, str] = {
    "42444946": "Left", "42444966": "Down", "42445021": "Left",
    "42445028": "Left", "42445429": "Down", "42446100": "Left",
}
"""The dataset's ``sky_direction`` per video: the side of the sensor image on which world up lies. The loader
measures the orientation from the poses instead (:func:`upright_quarter_turns`); this is the cross-check."""

# Counter-clockwise quarter turns (``numpy.rot90``'s ``k``) that bring the sky to the top of the image.
_UPRIGHT_QUARTER_TURNS = {"Up": 0, "Right": 1, "Down": 2, "Left": 3}

_DEPTH_UNITS_PER_METRE = 1000.0          # depth PNGs store millimetres
_KINDS = ("wide", "highres_depth", "lowres_depth", "confidence")


# --------------------------------------------------------------------------------------------------- files


def _root(root: Path | None) -> Path:
    """The ARKitScenes dataset folder.

    Parameters
    ----------
    root : pathlib.Path or None
        An explicit dataset folder, or ``None`` for ``config.dataset_dir("arkitscenes")``.

    Returns
    -------
    pathlib.Path
        The folder holding ``annotations/``, ``traj/``, ``upsampling/`` and ``wide_intrinsics/``.

    Raises
    ------
    RuntimeError
        If ``root`` is ``None`` and the data root is not configured or not mounted.
    """
    return Path(root) if root is not None else dataset_dir(DATASET)


@lru_cache(maxsize=16)
def _open_zip(path: str, mtime_ns: int, size: int) -> zipfile.ZipFile:
    """An open archive, cached; ``mtime_ns`` and ``size`` are part of the key so a changed file reopens.

    Parameters
    ----------
    path : str
        Archive path.
    mtime_ns : int
        Modification time of the file, in nanoseconds (cache key only).
    size : int
        File size in bytes (cache key only).

    Returns
    -------
    zipfile.ZipFile
        The archive, opened read-only.
    """
    return zipfile.ZipFile(path)


def _zip(path: Path) -> zipfile.ZipFile:
    """The cached, open archive at ``path``.

    Parameters
    ----------
    path : pathlib.Path
        Archive path.

    Returns
    -------
    zipfile.ZipFile
        The archive, opened read-only.

    Raises
    ------
    FileNotFoundError
        If the archive does not exist.
    """
    stat = path.stat()
    return _open_zip(str(path), stat.st_mtime_ns, stat.st_size)


def _upsampling_zip(video_id: str, root: Path | None) -> tuple[str, zipfile.ZipFile]:
    """The upsampling archive of a video, with its path relative to the dataset folder.

    Parameters
    ----------
    video_id : str
        ARKitScenes video id.
    root : pathlib.Path or None
        Dataset folder, or ``None`` for the configured one.

    Returns
    -------
    rel : str
        ``"upsampling/<video_id>.zip"``.
    archive : zipfile.ZipFile
        The open archive.
    """
    rel = f"upsampling/{video_id}.zip"
    return rel, _zip(_root(root) / rel)


# ---------------------------------------------------------------------------------------------- annotations


def box_from_obb(obb: Mapping[str, Any]) -> OrientedBox:
    """An :class:`OrientedBox` from an ARKitScenes ``obbAligned`` record.

    Parameters
    ----------
    obb : mapping
        ``{"centroid": [3], "axesLengths": [3], "normalizedAxes": [9]}``; metres, world frame. The 9 axis
        values, reshaped row-major to 3x3, hold one box axis per row.

    Returns
    -------
    OrientedBox
        The box, with the axes as the columns of its rotation.
    """
    axes_as_rows = np.asarray(obb["normalizedAxes"], dtype=np.float64).reshape(3, 3)
    return OrientedBox(obb["centroid"], obb["axesLengths"], axes_as_rows.T)


def load_objects(video_id: str, *, root: Path | None = None) -> list[SceneObject]:
    """The annotated objects of a video, with ground-truth boxes in the world frame.

    Parameters
    ----------
    video_id : str
        ARKitScenes video id.
    root : pathlib.Path or None, optional
        Dataset folder, or ``None`` for the configured one.

    Returns
    -------
    list of SceneObject
        One per annotation, in file order. ``object_id`` is the annotation's ``uid`` (its ``objectId`` is
        shared by every box in a file), ``label`` the class as given, ``source`` ``"oracle"``.

    Raises
    ------
    ValueError
        If two annotations share a ``uid``.
    """
    path = _root(root) / "annotations" / f"{video_id}_3dod_annotation.json"
    records = json.loads(path.read_text())["data"]
    objects = [SceneObject(str(r["uid"]), str(r["label"]), box_from_obb(r["segments"]["obbAligned"]))
               for r in records]
    if len({o.object_id for o in objects}) != len(objects):
        raise ValueError(f"{path}: duplicate uid")
    return objects


# ----------------------------------------------------------------------------------------------- trajectory


def parse_trajectory(text: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Times and camera-to-world poses from the text of a ``lowres_wide.traj`` file.

    Parameters
    ----------
    text : str
        Whitespace-separated rows ``t rx ry rz tx ty tz``: time in seconds, then a world-to-camera rotation
        (axis-angle, radians) and translation (metres).

    Returns
    -------
    times : numpy.ndarray, shape (N,), float64
        Seconds, increasing.
    T_wc : numpy.ndarray, shape (N, 4, 4), float64
        Camera-to-world poses (OpenCV camera axes), metres.

    Raises
    ------
    ValueError
        If a row does not have 7 values or the times are not strictly increasing.
    """
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if any(len(r) != 7 for r in rows):
        raise ValueError("trajectory rows must have 7 values: t rx ry rz tx ty tz")
    values = np.array(rows, dtype=np.float64).reshape(-1, 7)
    times = values[:, 0]
    if np.any(np.diff(times) <= 0):
        raise ValueError("trajectory times must be strictly increasing")
    T_cw = np.repeat(np.eye(4)[None], len(values), axis=0)
    T_cw[:, :3, :3] = axis_angle_to_matrix(values[:, 1:4])
    T_cw[:, :3, 3] = values[:, 4:7]
    return times, np.stack([invert_pose(pose) for pose in T_cw])


def load_trajectory(video_id: str, *, root: Path | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The ARKit trajectory of a video.

    Parameters
    ----------
    video_id : str
        ARKitScenes video id.
    root : pathlib.Path or None, optional
        Dataset folder, or ``None`` for the configured one.

    Returns
    -------
    times : numpy.ndarray, shape (N,), float64
        Seconds on the device clock (the clock of the ``wide`` frame timestamps).
    T_wc : numpy.ndarray, shape (N, 4, 4), float64
        Camera-to-world poses (OpenCV camera axes), metres.
    """
    return parse_trajectory((_root(root) / "traj" / f"{video_id}_lowres_wide.traj").read_text())


def _bracket_gap(times: NDArray[np.float64], query: NDArray[np.float64]) -> NDArray[np.float64]:
    """Time between the two trajectory samples that bracket each query time.

    Parameters
    ----------
    times : numpy.ndarray, shape (N,), float64
        Increasing sample times, seconds.
    query : numpy.ndarray, shape (M,), float64
        Query times, seconds.

    Returns
    -------
    numpy.ndarray, shape (M,), float64
        Seconds; 0 where a query equals a sample time, and 0 outside the span (such queries are rejected
        separately, by :func:`~measured_space.geometry.interpolate_poses`).
    """
    lo = np.clip(np.searchsorted(times, query, side="right") - 1, 0, len(times) - 1)
    hi = np.clip(np.searchsorted(times, query, side="left"), 0, len(times) - 1)
    return times[hi] - times[lo]


# -------------------------------------------------------------------------------------------- orientation


def _sky_side(x: float, y: float) -> str:
    """The image side an image-plane direction points to.

    Parameters
    ----------
    x, y : float
        Direction components along the camera's x (right) and y (down) axes.

    Returns
    -------
    str
        ``"Up"``, ``"Down"``, ``"Left"`` or ``"Right"``, by the larger component.
    """
    if abs(x) > abs(y):
        return "Left" if x < 0 else "Right"
    return "Up" if y < 0 else "Down"


def sky_direction_from_pose(T_wc: Array, up: Array = WORLD_UP) -> str:
    """The image side that world up points to in one camera, in the vocabulary of the dataset's ``sky_direction``.

    Parameters
    ----------
    T_wc : numpy.ndarray, shape (4, 4), float64
        Camera-to-world pose, OpenCV camera axes.
    up : numpy.ndarray, shape (3,), float64, optional
        World up direction.

    Returns
    -------
    str
        ``"Up"``, ``"Down"``, ``"Left"`` or ``"Right"``: the larger image-plane component of world up. It is
        ill-defined when the camera looks straight up or down.
    """
    x, y = (T_wc[:3, :3].T @ np.asarray(up, dtype=np.float64))[:2]
    return _sky_side(float(x), float(y))


def up_angle_in_image(T_wc: Array, up: Array = WORLD_UP) -> float:
    """Angle between world up, projected onto a camera's image plane, and image up (-v).

    Parameters
    ----------
    T_wc : numpy.ndarray, shape (4, 4), float64
        Camera-to-world pose, OpenCV camera axes.
    up : numpy.ndarray, shape (3,), float64, optional
        World up direction.

    Returns
    -------
    float
        Degrees in [0, 180]; 0 for an upright image. NaN if the camera looks exactly along ``up``. The
        projection uses the camera's x/y axes, i.e. assumes square pixels (fx = fy).
    """
    x, y = (T_wc[:3, :3].T @ np.asarray(up, dtype=np.float64))[:2]
    norm = float(np.hypot(x, y))
    return float(np.degrees(np.arccos(np.clip(-y / norm, -1.0, 1.0)))) if norm > 0 else float("nan")


def quarter_turns_for_sky(sky_direction: str) -> int:
    """How to rotate an image with the sky on a given side so that the sky is at the top.

    Parameters
    ----------
    sky_direction : str
        ``"Up"``, ``"Down"``, ``"Left"`` or ``"Right"``.

    Returns
    -------
    int
        ``k`` for ``numpy.rot90(image, k)`` (counter-clockwise quarter turns): 3 (one clockwise turn) for
        ``"Left"``, 2 for ``"Down"``, 1 for ``"Right"``, 0 for ``"Up"``.

    Raises
    ------
    KeyError
        For any other value.
    """
    return _UPRIGHT_QUARTER_TURNS[sky_direction]


@lru_cache(maxsize=32)
def _trajectory_quarter_turns(path: str, mtime_ns: int) -> int:
    """Upright quarter turns for a trajectory file, cached; ``mtime_ns`` is part of the key.

    Parameters
    ----------
    path : str
        ``lowres_wide.traj`` path.
    mtime_ns : int
        Its modification time in nanoseconds (cache key only).

    Returns
    -------
    int
        ``k`` for ``numpy.rot90``, from the summed image-plane components of world up over all poses.
    """
    _, T_wc = parse_trajectory(Path(path).read_text())
    up_cam = np.einsum("nji,j->ni", T_wc[:, :3, :3], WORLD_UP)          # R_wc^T up, per pose
    x, y = up_cam[:, :2].sum(axis=0)
    return quarter_turns_for_sky(_sky_side(float(x), float(y)))


def upright_quarter_turns(video_id: str, *, root: Path | None = None) -> int:
    """How to rotate a video's sensor images upright, measured from its poses.

    World up is projected into every camera of the trajectory; the image-plane components are summed (so
    cameras looking nearly straight up or down count little) and the side they point to is the video's sky
    side. On the six laser videos this agrees with the dataset's :data:`SKY_DIRECTION` (checked by the tests and
    ``scripts/check_arkitscenes_oracle.py``).

    Parameters
    ----------
    video_id : str
        ARKitScenes video id.
    root : pathlib.Path or None, optional
        Dataset folder, or ``None`` for the configured one.

    Returns
    -------
    int
        ``k`` for ``numpy.rot90(image, k)`` (counter-clockwise quarter turns).
    """
    path = _root(root) / "traj" / f"{video_id}_lowres_wide.traj"
    return _trajectory_quarter_turns(str(path), path.stat().st_mtime_ns)


# Raw camera coordinates -> coordinates of the camera of an image turned one quarter counter-clockwise.
_QUARTER_TURN = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def rotate_camera(camera: Camera, k: int) -> tuple[Camera, Array]:
    """The camera of an image turned by ``numpy.rot90(image, k)``, and the matching change of camera axes.

    Turning the image is a pixel permutation: one counter-clockwise quarter turn sends pixel ``(u, v)`` to
    ``(v, W - 1 - u)`` (pixel centres at integer coordinates), which is the same pinhole camera with its x and
    y axes turned about the optical axis.

    Parameters
    ----------
    camera : Camera
        Camera of the unrotated image.
    k : int
        Counter-clockwise quarter turns (any integer; taken modulo 4).

    Returns
    -------
    camera : Camera
        Camera of the turned image (width and height swap for odd ``k``).
    R : numpy.ndarray, shape (3, 3), float64
        Rotation taking unrotated camera coordinates to turned camera coordinates; the turned view's pose is
        ``T_wc @ [[R.T, 0], [0, 1]]``.
    """
    R = np.eye(3)
    for _ in range(k % 4):
        camera = Camera(camera.fy, camera.fx, camera.cy, camera.width - 1 - camera.cx, camera.height, camera.width)
        R = _QUARTER_TURN @ R
    return camera, R


def _turn_pose(T_wc: Array, R: Array) -> Array:
    """A camera-to-world pose for camera axes changed by ``R`` (new = ``R`` @ old).

    Parameters
    ----------
    T_wc : numpy.ndarray, shape (4, 4), float64
        Pose of the original camera axes.
    R : numpy.ndarray, shape (3, 3), float64
        Rotation from original to new camera coordinates.

    Returns
    -------
    numpy.ndarray, shape (4, 4), float64
        Pose of the new camera axes (same centre).
    """
    turned = T_wc.copy()
    turned[:3, :3] = T_wc[:3, :3] @ R.T
    return turned


# ------------------------------------------------------------------------------------------------- frames


def _read_pincam(text: str) -> Camera:
    """A :class:`Camera` from the text of a ``.pincam`` file.

    Parameters
    ----------
    text : str
        One line ``width height fx fy cx cy`` (pixels).

    Returns
    -------
    Camera
        The pinhole camera.

    Raises
    ------
    ValueError
        If the line does not have 6 values.
    """
    values = text.split()
    if len(values) != 6:
        raise ValueError(f"pincam must have 6 values, got {text!r}")
    width, height, fx, fy, cx, cy = map(float, values)
    return Camera(fx, fy, cx, cy, int(width), int(height))


def _stamp_key(stamp: str) -> int:
    """A timestamp's value in whole milliseconds, independent of how its file name was formatted.

    Parameters
    ----------
    stamp : str
        Seconds as written in a file name, e.g. ``"7031.8"`` or ``"7031.800"``.

    Returns
    -------
    int
        Milliseconds; ``"7031.8"`` and ``"7031.800"`` give the same key.
    """
    return round(float(stamp) * 1000.0)


def _pincam_members(archive: zipfile.ZipFile, video_id: str) -> dict[int, str]:
    """The ``.pincam`` members of a ``wide_intrinsics`` archive, keyed by timestamp value.

    Frame and intrinsics names do not always format a timestamp the same way: a frame may drop trailing
    zeros (``48458647_7031.8.png``) where its intrinsics keep three decimals (``48458647_7031.800.pincam``).
    Matching on the value rather than the text pairs them.

    Parameters
    ----------
    archive : zipfile.ZipFile
        The video's ``wide_intrinsics`` archive.
    video_id : str
        ARKitScenes video id.

    Returns
    -------
    dict of int to str
        :func:`_stamp_key` of each timestamp to its member name.
    """
    prefix = f"wide_intrinsics/{video_id}_"
    return {_stamp_key(n[len(prefix):-len(".pincam")]): n
            for n in archive.namelist() if n.startswith(prefix) and n.endswith(".pincam")}


def wide_timestamps(video_id: str, *, root: Path | None = None) -> list[str]:
    """Timestamps of every ``wide`` frame in a video's upsampling archive, as written in the file names.

    Parameters
    ----------
    video_id : str
        ARKitScenes video id.
    root : pathlib.Path or None, optional
        Dataset folder, or ``None`` for the configured one.

    Returns
    -------
    list of str
        Timestamps (seconds, e.g. ``"223612.562"``), sorted by time; including frames without a pose.
    """
    _, archive = _upsampling_zip(video_id, root)
    prefix = f"{video_id}/wide/{video_id}_"
    stamps = [n[len(prefix):-len(".png")] for n in archive.namelist() if n.startswith(prefix) and n.endswith(".png")]
    return sorted(stamps, key=float)


def list_frames(video_id: str, *, root: Path | None = None, max_gap: float = 0.25,
                upright: bool = True) -> list[FrameView]:
    """The ``wide`` frames of a video that have a valid interpolated pose.

    Parameters
    ----------
    video_id : str
        ARKitScenes video id.
    root : pathlib.Path or None, optional
        Dataset folder, or ``None`` for the configured one.
    max_gap : float, optional
        Seconds. A frame is dropped when the two trajectory samples around it are further apart than this,
        as well as when it lies outside the trajectory's time span (poses are never extrapolated).
    upright : bool, optional
        If True (default), each view describes the image turned upright (:func:`upright_quarter_turns`):
        its camera and ``T_wc`` are those of the turned image, ``camera_id`` is ``"wide"``, and the image
        loaders return turned arrays. If False, views describe the sensor's own orientation and carry
        ``camera_id`` ``"wide_sensor"``, which the loaders honour.

    Returns
    -------
    list of FrameView
        Sorted by time. ``frame`` is the timestamp text, ``camera`` the frame's own intrinsics,
        ``image_ref`` ``"upsampling/<id>.zip::<member>"`` relative to the dataset folder (the stored PNG,
        in sensor orientation; read it with :func:`load_rgb`), ``up`` world +z.

    Raises
    ------
    KeyError
        If a ``wide`` frame has no ``.pincam`` intrinsics.
    """
    rel, _ = _upsampling_zip(video_id, root)
    intrinsics = _zip(_root(root) / "wide_intrinsics" / f"{video_id}_wide_intrinsics.zip")
    pincams = _pincam_members(intrinsics, video_id)
    stamps = wide_timestamps(video_id, root=root)
    times, T_wc = load_trajectory(video_id, root=root)
    query = np.array([float(s) for s in stamps])
    poses, valid = interpolate_poses(times, T_wc, query)
    valid &= _bracket_gap(times, query) <= max_gap
    turns = upright_quarter_turns(video_id, root=root) if upright else 0
    camera_id = CAMERA_ID if upright else SENSOR_CAMERA_ID
    views = []
    for stamp, pose, ok in zip(stamps, poses, valid):
        if not ok:
            continue
        pincam = pincams.get(_stamp_key(stamp))
        if pincam is None:
            raise KeyError(f"no .pincam intrinsics for wide frame {stamp} of video {video_id}")
        camera, R = rotate_camera(_read_pincam(intrinsics.read(pincam).decode()), turns)
        member = f"{video_id}/wide/{video_id}_{stamp}.png"
        views.append(FrameView(DATASET, video_id, camera_id, stamp, camera, _turn_pose(pose, R),
                               f"{rel}::{member}", WORLD_UP.copy()))
    return views


def _read_kind(view: FrameView, kind: str, root: Path | None) -> NDArray:
    """Decode the ``kind`` image that shares ``view``'s timestamp, in the view's orientation.

    Parameters
    ----------
    view : FrameView
        A frame from :func:`list_frames` (upright, or sensor orientation for ``camera_id`` ``"wide_sensor"``).
    kind : str
        One of ``"wide"``, ``"highres_depth"``, ``"lowres_depth"``, ``"confidence"``.
    root : pathlib.Path or None
        Dataset folder, or ``None`` for the configured one.

    Returns
    -------
    numpy.ndarray
        The decoded PNG (``uint8`` RGB, ``uint16`` depth or ``uint8`` confidence), turned like the view.

    Raises
    ------
    ValueError
        If ``view`` is not an ARKitScenes view, ``kind`` is unknown, or the turned image does not have the
        aspect of the view's camera.
    """
    if view.dataset != DATASET or "::" not in view.image_ref or kind not in _KINDS \
            or view.camera_id not in (CAMERA_ID, SENSOR_CAMERA_ID):
        raise ValueError(f"not an ARKitScenes wide view or unknown kind: {view.image_ref!r}, {kind!r}")
    rel, member = view.image_ref.split("::", 1)
    with Image.open(io.BytesIO(_zip(_root(root) / rel).read(member.replace("/wide/", f"/{kind}/")))) as image:
        array = np.asarray(image.convert("RGB") if kind == "wide" else image)
    turns = upright_quarter_turns(view.scene, root=root) if view.camera_id == CAMERA_ID else 0
    array = np.ascontiguousarray(np.rot90(array, turns))
    h, w = array.shape[:2]
    if h * view.camera.width != w * view.camera.height:
        raise ValueError(f"{kind} image {w}x{h} does not match the view's {view.camera.width}x{view.camera.height}")
    return array


def load_rgb(view: FrameView, *, root: Path | None = None) -> NDArray[np.uint8]:
    """The RGB image of a frame, in the view's orientation (upright for views from :func:`list_frames`).

    Parameters
    ----------
    view : FrameView
        A frame from :func:`list_frames`.
    root : pathlib.Path or None, optional
        Dataset folder, or ``None`` for the configured one.

    Returns
    -------
    numpy.ndarray, shape (H, W, 3), uint8
        ``(view.camera.height, view.camera.width, 3)``: 1920 x 1440 upright for the "Left" and "Right"
        videos, 1440 x 1920 otherwise.
    """
    return _read_kind(view, "wide", root)


def load_laser_depth(view: FrameView, *, root: Path | None = None) -> NDArray[np.float32]:
    """The Faro laser depth rendered into a frame (the independent reference for the boxes).

    Parameters
    ----------
    view : FrameView
        A frame from :func:`list_frames`.
    root : pathlib.Path or None, optional
        Dataset folder, or ``None`` for the configured one.

    Returns
    -------
    numpy.ndarray, shape (H, W), float32
        Planar depth (z) in metres at the RGB resolution and orientation; 0 where there is no laser return.
    """
    return (_read_kind(view, "highres_depth", root) / _DEPTH_UNITS_PER_METRE).astype(np.float32)


def load_lidar_depth(view: FrameView, *, root: Path | None = None) -> NDArray[np.float32]:
    """The iPad LiDAR depth of a frame.

    Parameters
    ----------
    view : FrameView
        A frame from :func:`list_frames`.
    root : pathlib.Path or None, optional
        Dataset folder, or ``None`` for the configured one.

    Returns
    -------
    numpy.ndarray, shape (h, w), float32
        Planar depth (z) in metres over the RGB camera's field of view and in its orientation, at 256 x 192
        sensor resolution (192 x 256 for "Left"/"Right" videos when upright); use
        ``view.camera.scaled(w, h)``. 0 where invalid.
    """
    return (_read_kind(view, "lowres_depth", root) / _DEPTH_UNITS_PER_METRE).astype(np.float32)


def load_confidence(view: FrameView, *, root: Path | None = None) -> NDArray[np.uint8]:
    """ARKit's confidence in each LiDAR depth pixel.

    Parameters
    ----------
    view : FrameView
        A frame from :func:`list_frames`.
    root : pathlib.Path or None, optional
        Dataset folder, or ``None`` for the configured one.

    Returns
    -------
    numpy.ndarray, shape (h, w), uint8
        0 (low), 1 (medium) or 2 (high), aligned with :func:`load_lidar_depth`.
    """
    return _read_kind(view, "confidence", root)


# ------------------------------------------------------------------------------------------ laser reference


@dataclass(frozen=True)
class LaserBoxFit:
    """How well one box explains the laser points around it in one frame.

    Attributes
    ----------
    n_region : int
        Valid laser pixels whose camera ray passes through the box grown by the margin.
    n_inside : int
        Of those, pixels whose laser point lies inside the grown box.
    surface_distance : numpy.ndarray, shape (n_inside,), float64
        Unsigned distance (metres) of each inside point to the surface of the (ungrown) box.
    face_residual : numpy.ndarray, shape (n_inside,), float64
        Signed distance (metres) of each inside point to the plane of its nearest face, positive outside.
    face_normal : numpy.ndarray, shape (n_inside, 3), float64
        Outward unit normal (world frame) of that face.
    """

    n_region: int
    n_inside: int
    surface_distance: NDArray[np.float64]
    face_residual: NDArray[np.float64]
    face_normal: NDArray[np.float64]

    @property
    def explained(self) -> float:
        """Share of the box's ray region whose laser points lie inside it (NaN for an empty region).

        Returns
        -------
        float
            ``n_inside / n_region``.
        """
        return self.n_inside / self.n_region if self.n_region else float("nan")


def depth_rays(view: FrameView, depth: NDArray[np.floating], stride: int = 1) -> tuple[Array, Array]:
    """World points and camera rays of the valid pixels of a planar depth map.

    Parameters
    ----------
    view : FrameView
        The frame; its camera is rescaled to the depth map's resolution.
    depth : numpy.ndarray, shape (H, W), float
        Planar depth in metres; 0 or NaN means invalid.
    stride : int, optional
        Use every ``stride``-th pixel in each direction.

    Returns
    -------
    points : numpy.ndarray, shape (N, 3), float64
        Back-projected points, metres, world frame.
    rays : numpy.ndarray, shape (N, 3), float64
        World-frame ray directions scaled to unit planar depth, so ``points = camera_center + z * rays``.
    """
    h, w = depth.shape[:2]
    camera = view.camera.scaled(w, h)
    rows, cols = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride].astype(np.float64).ravel()
    ok = np.isfinite(z) & (z > 0)
    u, v, z = cols.ravel()[ok], rows.ravel()[ok], z[ok]
    rays_cam = np.stack([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, np.ones_like(z)], axis=1)
    rays = rays_cam @ view.T_wc[:3, :3].T
    return view.camera_center + z[:, None] * rays, rays


def _rays_hit_box(origin: Array, rays: Array, box: OrientedBox) -> NDArray[np.bool_]:
    """Whether rays from one origin pass through a box in front of the origin (slab test).

    Parameters
    ----------
    origin : numpy.ndarray, shape (3,), float64
        Ray origin, world frame.
    rays : numpy.ndarray, shape (N, 3), float64
        Ray directions, world frame (any scale).
    box : OrientedBox
        The box, world frame.

    Returns
    -------
    numpy.ndarray, shape (N,), bool
        True where the ray meets the box at a positive ray parameter.
    """
    o = (origin - box.center) @ box.rotation
    d = rays @ box.rotation
    half = box.size / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - o) / d, (half - o) / d
    # A ray parallel to a slab gives NaN or +-inf there; NaN (origin on the slab boundary) is ignored.
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    return (t_near <= t_far) & (t_far > 0)


def laser_box_fit(origin: Array, points: Array, rays: Array, box: OrientedBox, margin: float = 0.05) -> LaserBoxFit:
    """Compare a box with the laser points seen through it.

    Parameters
    ----------
    origin : numpy.ndarray, shape (3,), float64
        Camera centre, metres, world frame.
    points : numpy.ndarray, shape (N, 3), float64
        Laser points, metres, world frame (from :func:`depth_rays`).
    rays : numpy.ndarray, shape (N, 3), float64
        Their camera rays, world frame (from :func:`depth_rays`).
    box : OrientedBox
        The box, world frame.
    margin : float, optional
        Metres the box is grown by on every side for the region and inside tests.

    Returns
    -------
    LaserBoxFit
        Region and inside counts, and per inside point its distance to the box surface and its signed
        residual to the nearest face plane.
    """
    grown = OrientedBox(box.center, box.size + 2 * margin, box.rotation)
    region = _rays_hit_box(origin, rays, grown)
    candidates = points[region]
    inside = candidates[grown.contains(candidates)]
    local = box.to_local(inside)
    excess = np.abs(local) - box.size / 2                     # signed distance to each face pair's plane
    axis = np.argmax(excess, axis=1)                          # nearest face inside; most-violated face outside
    rows = np.arange(len(inside))
    normals = box.rotation[:, axis].T * np.sign(local[rows, axis])[:, None]
    return LaserBoxFit(int(region.sum()), len(inside), box.distance_to_surface(inside), excess[rows, axis], normals)
