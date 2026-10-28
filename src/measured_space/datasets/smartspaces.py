"""Physical AI Smart Spaces (MTMC tracking, 2026 split): fixed calibrated warehouse cameras with exact 3D GT.

A scene is addressed by its path below the dataset folder, e.g. ``"MTMC_Tracking_2026/train/Warehouse_018"``.
Synthetic scenes have ``calibration.json``, ``ground_truth.json``, ``videos/Camera_XXXX.mp4`` and
``depth_maps/Camera_XXXX.h5``. The two real test scenes (``Warehouse_026``, ``Warehouse_027``) have only
calibration and video, so :func:`load_calibration`, :func:`list_frames` and :func:`load_rgb` work on them
and the GT and depth readers raise.

Conventions, checked on ``Warehouse_018`` by ``scripts/check_smartspaces_oracle.py``, which also prints the
rival readings' scores:

- ``extrinsicMatrix`` maps **world to camera** (OpenCV axes, metres) in a **z-up** world whose floor is
  z = 0. The camera centre ``-R^T t`` equals the sensor's listed ``coordinates`` (x, y); read the other way
  round, no projected box overlaps its 2D box (IoU 0).
- ``3d location`` is the box **centre** (parked objects' box bottoms sit at z = 0.000 to -0.004 m).
  ``3d bounding box scale`` gives the full side lengths along the box's local x, y, z: width, length,
  height. Straight-line travel is along local -y.
- ``3d bounding box rotation`` is ``[pitch, roll, yaw]`` in **radians**, applied as rotations about the
  world x, y, z axes in that order (``R = Rz(yaw) Ry(roll) Rx(pitch)``), with yaw counter-clockwise
  about +z. Yaw spans [-pi, pi]; pitch and roll are below 3e-5 rad in every record seen, so their order is
  not tested by the data.
- ``2d bounding box visible`` maps camera id to ``[xmin, ymin, xmax, ymax]``, inclusive integer pixel
  indices (maxima 1919 and 1079) of the object's visible silhouette. A camera is listed only if it sees
  the object. Every such box lies inside the projection of the object's 3D box.
- Depth frames are HDF5 datasets ``distance_to_image_plane_{frame:05d}.png``: planar z-depth in
  **millimetres**, ``uint16``, 0 where there is no surface. Read this way the floor back-projects to
  z = 0.002 m; read as distance along the ray it spreads over 0.015-0.3 m. They are stored as raw
  gzip-compressed arrays despite the ``.png`` suffix; PNG-encoded bytes are also accepted.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from measured_space.config import dataset_dir
from measured_space.geometry import Array, Camera, OrientedBox, invert_pose, make_pose
from measured_space.scene import FrameView, SceneObject

DATASET = "smartspaces"
UP = np.array([0.0, 0.0, 1.0])
DEPTH_KEY = "distance_to_image_plane_{frame:05d}.png"
DEPTH_SCALE = 1e-3                       # stored millimetres -> metres
# Any |angle| beyond a full turn means the file does not hold radians; fail rather than guess.
_MAX_ANGLE = 2 * np.pi + 1e-3

Box2D = tuple[int, int, int, int]

# Open video captures and HDF5 files, keyed by absolute path. A capture also remembers the index of the
# frame its next ``read()`` returns, so sequential access skips the seek.
_captures: dict[str, tuple[cv2.VideoCapture, int]] = {}
_depth_files: dict[str, h5py.File] = {}


def scene_dir(scene: str) -> Path:
    """The folder of a scene.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder, e.g. ``"MTMC_Tracking_2026/train/Warehouse_018"``.

    Returns
    -------
    pathlib.Path
        ``<data root>/data/smartspaces/<scene>``.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist.
    """
    path = dataset_dir(DATASET) / scene
    if not path.is_dir():
        raise FileNotFoundError(f"Smart Spaces scene folder {path} not found")
    return path


def _field(record: Mapping[str, Any], name: str) -> Any:
    """A record's value under ``name`` spelled with spaces or with underscores.

    Parameters
    ----------
    record : Mapping[str, Any]
        One ground-truth object record.
    name : str
        Field name with spaces, e.g. ``"3d bounding box scale"``.

    Returns
    -------
    Any
        The value stored under either spelling.

    Raises
    ------
    KeyError
        If neither spelling is present.
    """
    for key in (name, name.replace(" ", "_")):
        if key in record:
            return record[key]
    raise KeyError(f"ground-truth record has no {name!r} field (keys: {sorted(record)})")


# --------------------------------------------------------------------------------------------- calibration


def parse_calibration(data: Mapping[str, Any]) -> dict[str, tuple[Camera, Array]]:
    """Cameras and poses from the contents of a ``calibration.json``.

    Parameters
    ----------
    data : Mapping[str, Any]
        Parsed JSON with a ``sensors`` list; each camera sensor has ``id``, ``intrinsicMatrix`` (3x3),
        ``extrinsicMatrix`` (3x4, world to camera) and ``frameWidth`` / ``frameHeight`` attributes.

    Returns
    -------
    dict[str, tuple[Camera, numpy.ndarray]]
        Camera id to ``(camera, T_wc)``; ``T_wc`` has shape (4, 4), float64, camera to world, metres.

    Raises
    ------
    ValueError
        If a camera has no frame size, a non-pinhole intrinsic matrix, or a non-rotation extrinsic block.
    """
    cameras = {}
    for sensor in data["sensors"]:
        if sensor.get("type", "camera") != "camera":
            continue
        attributes = {a["name"]: a["value"] for a in sensor.get("attributes", [])}
        if "frameWidth" not in attributes or "frameHeight" not in attributes:
            raise ValueError(f"{sensor['id']}: calibration has no frameWidth/frameHeight attributes")
        K = np.asarray(sensor["intrinsicMatrix"], dtype=np.float64)
        if abs(K[0, 1]) > 1e-9 or not np.allclose(K[2], [0.0, 0.0, 1.0]):
            raise ValueError(f"{sensor['id']}: intrinsic matrix is not a skew-free pinhole matrix: {K.tolist()}")
        camera = Camera(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]),
                        int(attributes["frameWidth"]), int(attributes["frameHeight"]))
        extrinsic = np.asarray(sensor["extrinsicMatrix"], dtype=np.float64)
        rotation = extrinsic[:, :3]
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
            raise ValueError(f"{sensor['id']}: extrinsic rotation is not a proper rotation")
        cameras[sensor["id"]] = (camera, invert_pose(make_pose(rotation, extrinsic[:, 3])))
    return cameras


def load_calibration(scene: str) -> dict[str, tuple[Camera, Array]]:
    """Every camera of a scene with its pose.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder.

    Returns
    -------
    dict[str, tuple[Camera, numpy.ndarray]]
        Camera id (e.g. ``"Camera_0000"``) to ``(camera, T_wc)``; ``T_wc`` has shape (4, 4), float64,
        camera to world in metres, z-up world.
    """
    return parse_calibration(json.loads((scene_dir(scene) / "calibration.json").read_text()))


def list_cameras(scene: str, with_video: bool = True) -> list[str]:
    """Camera ids of a scene, sorted.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder.
    with_video : bool, default True
        Keep only cameras whose video file is present locally.

    Returns
    -------
    list[str]
        Camera ids.
    """
    cameras = sorted(load_calibration(scene))
    if not with_video:
        return cameras
    folder = scene_dir(scene) / "videos"
    return [c for c in cameras if (folder / f"{c}.mp4").is_file()]


# ---------------------------------------------------------------------------------------------- ground truth


def rotation_matrices(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Box rotations from the file's ``[pitch, roll, yaw]`` angles.

    Parameters
    ----------
    angles : numpy.ndarray, shape (N, 3), float64
        Pitch, roll and yaw in radians: rotations about world x, y and z, applied in that order.

    Returns
    -------
    numpy.ndarray, shape (N, 3, 3), float64
        ``Rz(yaw) Ry(roll) Rx(pitch)``; columns are the box's local axes in the world frame.

    Raises
    ------
    ValueError
        If an angle exceeds a full turn, which would mean the file holds degrees.
    """
    angles = np.asarray(angles, dtype=np.float64).reshape(-1, 3)
    if angles.size and np.abs(angles).max() > _MAX_ANGLE:
        raise ValueError(f"box rotation of {np.abs(angles).max():.3f} exceeds 2*pi: angles are not radians")
    return Rotation.from_euler("xyz", angles).as_matrix().reshape(-1, 3, 3)


def box_from_annotation(location: Sequence[float], scale: Sequence[float],
                        rotation: Sequence[float]) -> OrientedBox:
    """One annotated box in the world frame.

    Parameters
    ----------
    location : Sequence[float]
        ``3d location``: box centre (x, y, z), metres.
    scale : Sequence[float]
        ``3d bounding box scale``: full side lengths along local x, y, z (width, length, height), metres.
    rotation : Sequence[float]
        ``3d bounding box rotation``: pitch, roll, yaw in radians.

    Returns
    -------
    OrientedBox
        The box.
    """
    return OrientedBox(location, scale, rotation_matrices(np.asarray(rotation, dtype=np.float64))[0])


def parse_ground_truth(data: Mapping[str, Sequence[Mapping[str, Any]]]
                       ) -> tuple[dict[int, list[SceneObject]], dict[int, dict[str, dict[str, Box2D]]]]:
    """Objects and visible 2D boxes per frame from the contents of a ``ground_truth.json``.

    Field names may use spaces (``"object type"``) or underscores (``"object_type"``).

    Parameters
    ----------
    data : Mapping[str, Sequence[Mapping[str, Any]]]
        Frame index (as text) to the list of object records in that frame.

    Returns
    -------
    objects : dict[int, list[SceneObject]]
        Frame index to objects with world boxes; ``object_id`` is ``"<type>_<id>"``, ``label`` the type.
    visible : dict[int, dict[str, dict[str, tuple[int, int, int, int]]]]
        Frame index to object id to camera id to ``(xmin, ymin, xmax, ymax)``, inclusive pixel indices.
        Objects that no camera sees are absent.

    Raises
    ------
    KeyError
        If a record lacks a required field.
    ValueError
        If rotations are not in radians (see :func:`rotation_matrices`).
    """
    frames, labels, ids, numbers, boxes2d = [], [], [], [], []
    for frame_key, records in data.items():
        for record in records:
            label = str(_field(record, "object type"))
            frames.append(int(frame_key))
            labels.append(label)
            ids.append(f"{label}_{_field(record, 'object id')}")
            numbers.append([*_field(record, "3d location"), *_field(record, "3d bounding box scale"),
                            *_field(record, "3d bounding box rotation")])
            boxes2d.append(_field(record, "2d bounding box visible"))
    values = np.asarray(numbers, dtype=np.float64).reshape(-1, 9)
    rotations = rotation_matrices(values[:, 6:9])
    objects: dict[int, list[SceneObject]] = {int(k): [] for k in data}
    visible: dict[int, dict[str, dict[str, Box2D]]] = {int(k): {} for k in data}
    for i, (frame, label, object_id) in enumerate(zip(frames, labels, ids)):
        objects[frame].append(SceneObject(object_id, label, OrientedBox(values[i, :3], values[i, 3:6],
                                                                         rotations[i])))
        if boxes2d[i]:
            visible[frame][object_id] = {cam: tuple(int(round(x)) for x in box) for cam, box in boxes2d[i].items()}
    return objects, visible


@functools.lru_cache(maxsize=2)
def _ground_truth(path: str) -> tuple[dict[int, list[SceneObject]], dict[int, dict[str, dict[str, Box2D]]]]:
    """Parsed ground truth of one file, cached because the files are tens to hundreds of megabytes.

    Parameters
    ----------
    path : str
        Absolute path of a ``ground_truth.json``.

    Returns
    -------
    tuple
        The output of :func:`parse_ground_truth`.
    """
    with open(path) as f:
        return parse_ground_truth(json.load(f))


def _ground_truth_path(scene: str) -> str:
    """Absolute path of a scene's ``ground_truth.json``.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder.

    Returns
    -------
    str
        The path.

    Raises
    ------
    FileNotFoundError
        If the scene has no ground truth (e.g. a real test scene).
    """
    path = scene_dir(scene) / "ground_truth.json"
    if not path.is_file():
        raise FileNotFoundError(f"{scene} has no ground_truth.json")
    return str(path)


def load_ground_truth(scene: str) -> dict[int, list[SceneObject]]:
    """Ground-truth objects of every frame of a scene.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder.

    Returns
    -------
    dict[int, list[SceneObject]]
        Frame index to objects; boxes are in the world frame (metres), ``object_id`` is ``"<type>_<id>"``
        and stable over time, ``label`` is the class (Person, Forklift, PalletTruck, ...).

    Raises
    ------
    FileNotFoundError
        If the scene has no ground truth.
    """
    return _ground_truth(_ground_truth_path(scene))[0]


def visible_boxes_2d(scene: str) -> dict[int, dict[str, dict[str, Box2D]]]:
    """The ground truth's visible 2D boxes of every frame of a scene.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder.

    Returns
    -------
    dict[int, dict[str, dict[str, tuple[int, int, int, int]]]]
        Frame index to object id to camera id to ``(xmin, ymin, xmax, ymax)``, inclusive pixel indices of
        the object's visible part. A camera is present only if it sees the object.

    Raises
    ------
    FileNotFoundError
        If the scene has no ground truth.
    """
    return _ground_truth(_ground_truth_path(scene))[1]


# --------------------------------------------------------------------------------------------- views, images


def _video_path(scene: str, camera_id: str) -> Path:
    """Path of a camera's video.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder.
    camera_id : str
        Camera id, e.g. ``"Camera_0000"``.

    Returns
    -------
    pathlib.Path
        ``<scene>/videos/<camera_id>.mp4`` (not checked for existence).
    """
    return scene_dir(scene) / "videos" / f"{camera_id}.mp4"


def list_frames(scene: str, camera_id: str, frames: Iterable[int]) -> list[FrameView]:
    """Views of one camera at the given frame indices.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder.
    camera_id : str
        Camera id, e.g. ``"Camera_0000"``.
    frames : Iterable[int]
        Frame indices (30 FPS).

    Returns
    -------
    list[FrameView]
        One view per frame; the cameras are fixed, so all share one camera and pose. ``image_ref`` is
        ``"<scene>/videos/<camera_id>.mp4::<frame>"`` relative to the dataset folder.

    Raises
    ------
    KeyError
        If the scene's calibration has no such camera.
    """
    camera, T_wc = load_calibration(scene)[camera_id]
    video = f"{scene}/videos/{camera_id}.mp4"
    return [FrameView(DATASET, scene, camera_id, str(int(f)), camera, T_wc.copy(), f"{video}::{int(f)}",
                      UP.copy()) for f in frames]


def _capture(path: Path) -> tuple[cv2.VideoCapture, int]:
    """An open capture of a video and the index of the frame it reads next, opening it on first use.

    Parameters
    ----------
    path : pathlib.Path
        Video file.

    Returns
    -------
    tuple[cv2.VideoCapture, int]
        The capture and its next frame index.

    Raises
    ------
    FileNotFoundError
        If the file is missing or cannot be opened.
    """
    key = str(path)
    if key not in _captures:
        if not path.is_file():
            raise FileNotFoundError(f"video {path} not found")
        capture = cv2.VideoCapture(key)
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open video {path}")
        _captures[key] = (capture, 0)
    return _captures[key]


def read_video_frame(path: Path, frame: int) -> NDArray[np.uint8]:
    """One frame of a video by index.

    OpenCV seeking (``CAP_PROP_POS_FRAMES``) on these H.264 files returned the same pixels as sequential
    decoding for 36 forward and backward seeks per video on a synthetic and a real camera; the seek is
    skipped when the frame is the next one the capture would read.

    Parameters
    ----------
    path : pathlib.Path
        Video file.
    frame : int
        Frame index, from 0.

    Returns
    -------
    numpy.ndarray, shape (H, W, 3), uint8
        The frame in RGB order.

    Raises
    ------
    IndexError
        If the frame is outside the video.
    """
    capture, next_frame = _capture(path)
    if not 0 <= frame < int(capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        raise IndexError(f"frame {frame} outside {path.name} "
                         f"({int(capture.get(cv2.CAP_PROP_FRAME_COUNT))} frames)")
    if frame != next_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    if not ok:
        _captures[str(path)] = (capture, -1)          # position unknown: seek next time
        raise IndexError(f"could not decode frame {frame} of {path}")
    _captures[str(path)] = (capture, frame + 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_rgb(view: FrameView) -> NDArray[np.uint8]:
    """The colour image of a view.

    Parameters
    ----------
    view : FrameView
        A view from :func:`list_frames`.

    Returns
    -------
    numpy.ndarray, shape (H, W, 3), uint8
        RGB image (1080 x 1920 for this dataset).
    """
    return read_video_frame(_video_path(view.scene, view.camera_id), int(view.frame))


def decode_depth(raw: NDArray[Any]) -> NDArray[np.float32]:
    """Depth in metres from one stored depth frame.

    Parameters
    ----------
    raw : numpy.ndarray
        Either a (H, W) uint16 array of millimetres, or a 1-D uint8 array (or ``numpy.void`` / bytes)
        holding a PNG of one.

    Returns
    -------
    numpy.ndarray, shape (H, W), float32
        Planar depth in metres; 0 marks pixels with no depth.

    Raises
    ------
    ValueError
        If the data is neither a 2-D array nor a decodable single-channel 16-bit PNG.
    """
    if isinstance(raw, (bytes, np.void)):
        raw = np.frombuffer(bytes(raw), dtype=np.uint8)
    raw = np.asarray(raw)
    if raw.ndim == 1:
        decoded = cv2.imdecode(raw.astype(np.uint8, copy=False), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.ndim != 2 or decoded.dtype != np.uint16:
            raise ValueError("depth frame is not a single-channel 16-bit PNG")
        raw = decoded
    if raw.ndim != 2:
        raise ValueError(f"depth frame has shape {raw.shape}, expected (H, W)")
    return raw.astype(np.float32) * np.float32(DEPTH_SCALE)


def _depth_file(path: Path) -> h5py.File:
    """An open, cached read-only handle on a camera's depth HDF5 file.

    Parameters
    ----------
    path : pathlib.Path
        ``depth_maps/<camera_id>.h5``.

    Returns
    -------
    h5py.File
        The open file.

    Raises
    ------
    FileNotFoundError
        If the file is missing (real test scenes have no depth).
    """
    key = str(path)
    if key not in _depth_files:
        if not path.is_file():
            raise FileNotFoundError(f"depth file {path} not found")
        _depth_files[key] = h5py.File(key, "r")
    return _depth_files[key]


def load_depth(view: FrameView) -> NDArray[np.float32]:
    """The simulator's depth map of a view.

    Parameters
    ----------
    view : FrameView
        A view from :func:`list_frames` of a synthetic scene.

    Returns
    -------
    numpy.ndarray, shape (H, W), float32
        Planar (z) depth in metres, same resolution as the image; 0 marks pixels with no surface.

    Raises
    ------
    FileNotFoundError
        If the camera has no depth file.
    KeyError
        If the file has no entry for the frame.
    """
    h5 = _depth_file(scene_dir(view.scene) / "depth_maps" / f"{view.camera_id}.h5")
    key = DEPTH_KEY.format(frame=int(view.frame))
    if key not in h5:
        raise KeyError(f"{view.camera_id} depth has no entry {key!r}")
    return decode_depth(h5[key][()])


def close_handles() -> None:
    """Release every cached video capture and depth file (e.g. before the files are moved or between tests)."""
    for capture, _ in _captures.values():
        capture.release()
    _captures.clear()
    for h5 in _depth_files.values():
        h5.close()
    _depth_files.clear()
