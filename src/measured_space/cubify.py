"""Cubify Anything (CuTR) as an external 3D box labeller: frame export, gravity convention, and its predictions.

Cubify Anything (Lazarow et al., CVPR 2025) is a published single-frame 3D box detector. Its RGB-D model
("CuTR", checkpoint ``cutr_rgbd.pth``) takes one upright RGB image, the iPad LiDAR depth, the intrinsics of both
and the camera's tilt against gravity, and returns class-agnostic, metric, gravity-aligned 3D boxes (yaw only)
with a score and a 2D box each. It is run **unmodified**, in its own conda env, by ``scripts/run_cubify.py``;
this module never imports it. The two halves meet in two file formats.

**1. Frames** (written here by :func:`export_view`, read by ``scripts/run_cubify.py``). One ``.npz`` per view at
``<frames>/<dataset>/<scene>/<camera_id>_<frame>.npz`` (:func:`frame_path`), holding:

- ``image``: ``uint8 (4h, 4w, 3)``, the upright RGB image resized to exactly 4x the LiDAR depth size with PIL's
  bicubic filter: the ratio and the filter (PIL's default) of CuTR's own capture path, so the preprocessing is the
  published one;
- ``depth``: ``float32 (h, w)``, the iPad LiDAR depth in metres (0 = invalid), upright. Only LiDAR is ever
  exported (``depth_source`` ``"lidar"``): the Faro laser depth is the reference the labeller is audited
  against, so the labeller must not see it;
- ``K_img``, ``K_depth``: ``(3, 3)`` intrinsics of the resized image and of the depth map, from
  :meth:`~measured_space.geometry.Camera.scaled` (pixel centres at integers). CuTR casts pixel rays through
  ``i + 0.5`` and rescales K by plain multiplication, i.e. it reads K with pixel edges at integers; the two
  readings differ by at most half an input pixel (about 0.03 degrees at fx = 855 px), left as the smoke test
  had it;
- ``T_wc``: ``(4, 4)`` camera-to-world pose of the upright view; ``T_gravity``: ``(3, 3)``,
  :func:`gravity_to_camera` of it;
- ``image_size``: ``[W, H]`` of the full-resolution upright image (the frame of the stored 2D boxes);
- ``dataset``, ``scene``, ``camera_id``, ``frame``, ``view_key`` (:func:`measured_space.pipeline.view_key`)
  and ``depth_source``, as strings.

The export stores ``T_gravity`` and ``view_key`` so that the CuTR env needs neither this package nor its
conventions. :func:`export_videos` exports whole videos, resumably; a video is done when its folder holds
``complete.json`` (:data:`COMPLETE_MARKER`).

**2. Predictions** (written by ``scripts/run_cubify.py``, read here by :func:`read_predictions`). One JSONL per
scene at ``<predictions>/<dataset>/<scene>.jsonl``, one line per view::

    {"view_key": ..., "dataset": ..., "scene": ..., "camera_id": ..., "frame": ...,
     "model": "cutr_rgbd", "checkpoint_sha256": ..., "score_threshold": 0.25,
     "image_size": [W, H], "seconds": <model forward time>,
     "boxes": [{"center": [x, y, z], "size": [sx, sy, sz], "rotation": [[...], [...], [...]],
                "score": s, "box2d": [x0, y0, x1, y1], "center_cam": [x, y, z]}, ...]}

- ``center``, ``size``, ``rotation`` are the world :class:`~measured_space.geometry.OrientedBox` (rotation
  columns are the box axes; keys as :func:`measured_space.pipeline.box_json`), built exactly from CuTR's output
  by :func:`cutr_box_to_world`. CuTR's box tensor is ``(x, y, z, l, h, w)`` with its corners at
  ``R @ (+-l/2, +-h/2, +-w/2) + (x, y, z)`` in the camera frame, so the camera-frame box is
  ``OrientedBox((x, y, z), (l, h, w), R)`` and its corners are CuTR's ``pred_boxes_3d.corners`` (as a set). The
  box's y-axis is world down (from the gravity frame), so ``size[1]`` is the box's height.
- ``box2d`` is CuTR's own 2D box (``pred_boxes``), scaled from the model's input image to the full-resolution
  upright image (:func:`scale_box2d`). CuTR clips its boxes to ``[0, W] x [0, H]`` and casts the ray of pixel
  ``i`` through ``i + 0.5``, i.e. its pixel edges are at integer coordinates, the convention of
  :attr:`measured_space.perception.Instance.box2d`; scaling is then a pure multiplication.
- ``center_cam`` is CuTR's box centre in the (upright, OpenCV) camera frame, metres.
- Boxes have ``score >= score_threshold`` (0.25 by default, the published demo's value), in CuTR's order
  (descending score). The model is class-agnostic: there are no labels.

**Gravity convention** (:func:`gravity_to_camera`). CuTR predicts each box's yaw in a gravity frame and maps it
to the camera with ``T_gravity`` (gravity frame -> camera). The smoke test settled the undocumented convention:
the gravity frame's y-axis is **world down** (world z is up), its z-axis is the camera's optical axis projected
onto the horizontal plane, and x = y x z. The world-up variant lines up worse with ground truth (median
gravity-prism IoU of matched objects 0.69 vs 0.55 on six frames). For a level camera ``T_gravity`` is the identity.

Distances are in metres; conventions are those of :mod:`measured_space.geometry`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from PIL import Image

from measured_space.config import data_root
from measured_space.datasets import arkitscenes
from measured_space.geometry import Array, OrientedBox
from measured_space.pipeline import box_json, view_key
from measured_space.scene import FrameView

MODEL = "cutr_rgbd"
"""str: the model name recorded in the predictions."""

CHECKPOINT_SHA256 = "856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217"
"""str: SHA-256 of the published ``cutr_rgbd.pth`` (396,866,874 bytes)."""

SCORE_THRESHOLD = 0.25
"""float: default minimum score of a kept box, the published demo's ``--score-thresh``."""

DEPTH_TO_IMAGE = 4
"""int: the model's input image is this many times the LiDAR depth size, per side."""

DEPTH_SOURCE = "lidar"
"""str: the only depth ever exported (iPad LiDAR); never the laser depth, which is the reference."""

COMPLETE_MARKER = "complete.json"
"""str: file name of the marker that says a video's frames are all exported."""

WORLD_UP = np.array([0.0, 0.0, 1.0])
"""numpy.ndarray, shape (3,): world up in ARKitScenes (+z)."""

_HORIZONTAL_EPS = 1e-6           # below this norm the optical axis counts as vertical (see gravity_to_camera)
_PREDICTION_KEYS = ("view_key", "dataset", "scene", "camera_id", "frame", "model", "checkpoint_sha256",
                    "score_threshold", "image_size", "seconds", "boxes")
_BOX_KEYS = ("center", "size", "rotation", "score", "box2d", "center_cam")


# ------------------------------------------------------------------------------------------------- geometry


def gravity_to_camera(T_wc: Array, up: ArrayLike = WORLD_UP) -> Array:
    """CuTR's ``T_gravity``: the rotation from a yaw-free gravity frame to the camera frame.

    The gravity frame has its y-axis along world **down**, its z-axis along the camera's optical axis projected
    onto the horizontal plane, and x = y x z (right-handed). It differs from the camera frame only by the
    camera's pitch and roll, so a level camera gives the identity. When the optical axis is vertical (to within
    1e-6), its horizontal projection vanishes; the forward direction is then taken from the camera's image-up
    axis (-y looking down, +y looking up), which is the limit of the projection for a camera without roll.

    Parameters
    ----------
    T_wc : numpy.ndarray, shape (4, 4)
        Camera-to-world pose, OpenCV camera axes.
    up : array_like, shape (3,), optional
        World up direction (against gravity); +z by default.

    Returns
    -------
    numpy.ndarray, shape (3, 3), float64
        Rotation taking gravity-frame coordinates to camera coordinates (columns: the gravity frame's axes in
        the camera frame).
    """
    R_wc = np.asarray(T_wc, dtype=np.float64)[:3, :3]
    down = -np.asarray(up, dtype=np.float64)
    down = down / np.linalg.norm(down)
    forward = R_wc[:, 2] - np.dot(R_wc[:, 2], down) * down
    if np.linalg.norm(forward) < _HORIZONTAL_EPS:
        image_up = -np.sign(np.dot(R_wc[:, 2], down)) * R_wc[:, 1]
        forward = image_up - np.dot(image_up, down) * down
    forward = forward / np.linalg.norm(forward)
    gravity_axes_world = np.column_stack([np.cross(down, forward), down, forward])
    return R_wc.T @ gravity_axes_world


def nearest_rotation(matrix: ArrayLike) -> Array:
    """The rotation nearest to a nearly orthonormal matrix (CuTR's float32 box rotations).

    Parameters
    ----------
    matrix : array_like, shape (3, 3)
        A matrix close to a proper rotation.

    Returns
    -------
    numpy.ndarray, shape (3, 3), float64
        ``U @ V.T`` from the singular value decomposition ``matrix = U S V.T`` (the orthogonal polar factor).

    Raises
    ------
    ValueError
        If the matrix is a reflection (negative determinant), which no CuTR rotation can be.
    """
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if np.linalg.det(matrix) <= 0:
        raise ValueError(f"not a rotation (determinant {np.linalg.det(matrix):.3g})")
    u, _, vt = np.linalg.svd(matrix)
    return u @ vt


def cutr_box_to_world(center_cam: ArrayLike, dims: ArrayLike, rotation_cam: ArrayLike, T_wc: Array) -> OrientedBox:
    """A CuTR box (camera frame) as a world :class:`OrientedBox`, exactly.

    CuTR's ``GeneralInstance3DBoxes`` stores ``(x, y, z, l, h, w)`` and a rotation ``R`` and puts its corners at
    ``R @ (+-l/2, +-h/2, +-w/2) + (x, y, z)``: that is ``OrientedBox((x, y, z), (l, h, w), R)`` in the camera
    frame, which is then moved to the world with ``T_wc``. ``R`` is first replaced by its nearest rotation
    (a change of order 1e-7, from float32).

    Parameters
    ----------
    center_cam : array_like, shape (3,)
        ``pred_boxes_3d.gravity_center`` (``tensor[:, :3]``), camera frame, metres.
    dims : array_like, shape (3,)
        ``pred_boxes_3d.dims`` (``tensor[:, 3:6]``, i.e. ``l, h, w``): full side lengths along the box's x, y, z
        axes, metres.
    rotation_cam : array_like, shape (3, 3)
        ``pred_boxes_3d.R``: the box axes as columns, camera frame.
    T_wc : numpy.ndarray, shape (4, 4)
        Camera-to-world pose of the view the box was predicted in.

    Returns
    -------
    OrientedBox
        The box in the world frame; its corners equal CuTR's ``pred_boxes_3d.corners`` moved by ``T_wc``.

    Raises
    ------
    ValueError
        If ``rotation_cam`` is a reflection.
    """
    box_cam = OrientedBox(center_cam, dims, nearest_rotation(rotation_cam))
    return box_cam.transformed(np.asarray(T_wc, dtype=np.float64))


def scale_box2d(box2d: Sequence[float], from_size: Sequence[int],
                to_size: Sequence[int]) -> tuple[float, float, float, float]:
    """Move a 2D box from one resolution of an image to another (pixel edges at integer coordinates).

    With edges at integer coordinates a whole-image box is ``(0, 0, W, H)`` at every resolution, so resampling
    is a pure scaling, with no half-pixel offset.

    Parameters
    ----------
    box2d : sequence of float, shape (4,)
        ``(xmin, ymin, xmax, ymax)`` in pixels of the source image.
    from_size : sequence of int, shape (2,)
        ``(W, H)`` of the source image (CuTR's input).
    to_size : sequence of int, shape (2,)
        ``(W, H)`` of the target image (the full-resolution upright image).

    Returns
    -------
    tuple of float
        ``(xmin, ymin, xmax, ymax)`` in pixels of the target image.
    """
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    x0, y0, x1, y1 = (float(v) for v in box2d)
    return x0 * sx, y0 * sy, x1 * sx, y1 * sy


# --------------------------------------------------------------------------------------------------- export


def frame_path(frames_root: Path, view: FrameView) -> Path:
    """Where :func:`export_view` writes a view under a frames folder.

    Parameters
    ----------
    frames_root : pathlib.Path
        The frames folder.
    view : FrameView
        The view.

    Returns
    -------
    pathlib.Path
        ``<frames_root>/<dataset>/<scene>/<camera_id>_<frame>.npz``.
    """
    return Path(frames_root) / view.dataset / view.scene / f"{view.camera_id}_{view.frame}.npz"


def export_view(view: FrameView, out_path: Path, *, root: Path | None = None) -> Path:
    """Write one upright ARKitScenes view as a CuTR input frame (the npz described in the module docstring).

    Parameters
    ----------
    view : FrameView
        An upright view from :func:`measured_space.datasets.arkitscenes.list_frames` (``camera_id`` ``"wide"``).
    out_path : pathlib.Path
        The ``.npz`` file to write (its folder is created); an existing file is overwritten.
    root : pathlib.Path or None, optional
        ARKitScenes dataset folder, or ``None`` for the configured one.

    Returns
    -------
    pathlib.Path
        ``out_path``.

    Raises
    ------
    ValueError
        If the view is not an upright ARKitScenes view, or the image and depth aspects differ.
    """
    if view.dataset != arkitscenes.DATASET or view.camera_id != arkitscenes.CAMERA_ID:
        raise ValueError(f"expected an upright ARKitScenes view, got {view_key(view)!r}")
    rgb = arkitscenes.load_rgb(view, root=root)
    depth = arkitscenes.load_lidar_depth(view, root=root)            # LiDAR only: laser depth is the reference
    h, w = depth.shape
    H, W = DEPTH_TO_IMAGE * h, DEPTH_TO_IMAGE * w
    if rgb.shape[0] * w != rgb.shape[1] * h:
        raise ValueError(f"RGB {rgb.shape[1]}x{rgb.shape[0]} and depth {w}x{h} differ in aspect")
    image = np.asarray(Image.fromarray(rgb).resize((W, H), Image.Resampling.BICUBIC))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, image=image, depth=depth.astype(np.float32),
             K_img=view.camera.scaled(W, H).K, K_depth=view.camera.scaled(w, h).K,
             T_wc=np.asarray(view.T_wc, dtype=np.float64), T_gravity=gravity_to_camera(view.T_wc, view.up),
             image_size=np.array([view.camera.width, view.camera.height], dtype=np.int64),
             dataset=np.array(view.dataset), scene=np.array(view.scene), camera_id=np.array(view.camera_id),
             frame=np.array(view.frame), view_key=np.array(view_key(view)), depth_source=np.array(DEPTH_SOURCE))
    return out_path


def _video_complete(folder: Path, views: Sequence[FrameView]) -> bool:
    """Whether a video's frames folder holds a completion marker for exactly these views, and all their files.

    Parameters
    ----------
    folder : pathlib.Path
        ``<frames>/<dataset>/<scene>``.
    views : sequence of FrameView
        The video's views as listed now.

    Returns
    -------
    bool
        True if :data:`COMPLETE_MARKER` exists, parses, lists the same files, and they all exist.
    """
    marker = folder / COMPLETE_MARKER
    if not marker.exists():
        return False
    try:
        listed = json.loads(marker.read_text()).get("files", [])
    except json.JSONDecodeError:
        return False
    wanted = [frame_path(folder.parent.parent, v).name for v in views]
    return listed == wanted and all((folder / name).exists() for name in wanted)


def export_videos(video_ids: Iterable[str], frames_root: Path, *, root: Path | None = None,
                  overwrite: bool = False, log: bool = True) -> dict[str, int]:
    """Export every posed upright view of ARKitScenes videos as CuTR input frames, resumably.

    A video counts as done when its folder holds :data:`COMPLETE_MARKER` naming exactly the files of its current
    views, all present; it is then skipped. Otherwise all its views are (re)written and the marker written last,
    so an interrupted video is redone whole on the next run (no half-written file is ever trusted).

    Parameters
    ----------
    video_ids : iterable of str
        ARKitScenes video ids.
    frames_root : pathlib.Path
        Output folder; frames go to :func:`frame_path`.
    root : pathlib.Path or None, optional
        ARKitScenes dataset folder, or ``None`` for the configured one.
    overwrite : bool, optional
        Re-export videos that are already complete.
    log : bool, optional
        Print one progress line per video.

    Returns
    -------
    dict of str to int
        ``videos``, ``skipped`` (already complete), ``exported`` (videos written) and ``views`` (views written).
    """
    counts = {"videos": 0, "skipped": 0, "exported": 0, "views": 0}
    ids = list(video_ids)
    for k, video_id in enumerate(ids, 1):
        counts["videos"] += 1
        views = arkitscenes.list_frames(video_id, root=root)
        folder = Path(frames_root) / arkitscenes.DATASET / video_id
        if not overwrite and _video_complete(folder, views):
            counts["skipped"] += 1
            if log:
                print(f"[{k}/{len(ids)}] {video_id}: complete ({len(views)} views), skipped", flush=True)
            continue
        t0 = time.perf_counter()
        files = [export_view(v, frame_path(frames_root, v), root=root).name for v in views]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / COMPLETE_MARKER).write_text(json.dumps({"video_id": video_id, "n_views": len(files),
                                                          "depth_source": DEPTH_SOURCE, "files": files}) + "\n")
        counts["exported"] += 1
        counts["views"] += len(files)
        if log:
            print(f"[{k}/{len(ids)}] {video_id}: {len(files)} views in {time.perf_counter() - t0:.1f} s", flush=True)
    return counts


# ---------------------------------------------------------------------------------------------- predictions


@dataclass(frozen=True)
class ExternalBox:
    """One box from an external labeller (here CuTR), as read from the prediction JSONL.

    Attributes
    ----------
    box : OrientedBox
        The box in the world frame, metres (box axes as rotation columns; for CuTR, axis y is world down).
    score : float
        The model's confidence in [0, 1].
    box2d : tuple of float
        ``(xmin, ymin, xmax, ymax)`` in pixels of the full-resolution upright image, pixel edges at integer
        coordinates (the convention of :attr:`measured_space.perception.Instance.box2d`).
    center_cam : numpy.ndarray, shape (3,), float64
        The box centre in the view's camera frame (OpenCV axes), metres.
    """

    box: OrientedBox
    score: float
    box2d: tuple[float, float, float, float]
    center_cam: Array


def external_box_json(box: ExternalBox) -> dict[str, Any]:
    """An :class:`ExternalBox` as one entry of a prediction record's ``boxes`` list.

    Parameters
    ----------
    box : ExternalBox
        The box.

    Returns
    -------
    dict
        ``center``, ``size``, ``rotation`` (as :func:`measured_space.pipeline.box_json`), ``score``, ``box2d``,
        ``center_cam``.
    """
    return {**box_json(box.box), "score": float(box.score), "box2d": [float(v) for v in box.box2d],
            "center_cam": np.asarray(box.center_cam, dtype=np.float64).tolist()}


def external_box_from_json(record: Mapping[str, Any]) -> ExternalBox:
    """Parse one entry of a prediction record's ``boxes`` list.

    Parameters
    ----------
    record : mapping
        ``{"center", "size", "rotation", "score", "box2d", "center_cam"}``.

    Returns
    -------
    ExternalBox
        The box.

    Raises
    ------
    KeyError
        If a key is missing.
    ValueError
        If ``box2d`` does not have 4 values.
    """
    box2d = tuple(float(v) for v in record["box2d"])
    if len(box2d) != 4:
        raise ValueError(f"box2d must have 4 values, got {record['box2d']!r}")
    return ExternalBox(OrientedBox(record["center"], record["size"], record["rotation"]), float(record["score"]),
                       box2d, np.asarray(record["center_cam"], dtype=np.float64).reshape(3))  # type: ignore[arg-type]


def prediction_files(path: Path) -> list[Path]:
    """The prediction JSONL files at a path.

    Parameters
    ----------
    path : pathlib.Path
        One ``.jsonl`` file, or a predictions folder (``<dataset>/<scene>.jsonl`` files at any depth).

    Returns
    -------
    list of pathlib.Path
        Sorted file paths.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(path.rglob("*.jsonl"))


def load_prediction_records(path: Path) -> list[dict[str, Any]]:
    """Every view record (one JSONL line) at a path, validated.

    Parameters
    ----------
    path : pathlib.Path
        A prediction ``.jsonl`` file or a predictions folder.

    Returns
    -------
    list of dict
        One record per view, in file order (files sorted by path).

    Raises
    ------
    ValueError
        If a record lacks a key of the format, a box lacks a key, or two records share a ``view_key``.
    """
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for file in prediction_files(path):
        for n, line in enumerate(file.read_text().splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = [k for k in _PREDICTION_KEYS if k not in record]
            missing += [f"boxes[{i}].{k}" for i, b in enumerate(record.get("boxes", [])) for k in _BOX_KEYS
                        if k not in b]
            if missing:
                raise ValueError(f"{file}:{n}: missing {', '.join(missing)}")
            if record["view_key"] in seen:
                raise ValueError(f"{file}:{n}: duplicate view_key {record['view_key']!r}")
            seen.add(record["view_key"])
            records.append(record)
    return records


def read_predictions(path: Path) -> dict[str, list[ExternalBox]]:
    """The boxes of every view in prediction JSONL, keyed by view.

    Parameters
    ----------
    path : pathlib.Path
        A prediction ``.jsonl`` file (``<predictions>/<dataset>/<scene>.jsonl``) or a predictions folder.

    Returns
    -------
    dict of str to list of ExternalBox
        :func:`measured_space.pipeline.view_key` -> that view's kept boxes, in file order (descending score). A
        view with no box above the threshold maps to an empty list.

    Raises
    ------
    ValueError
        As :func:`load_prediction_records`.
    """
    return {r["view_key"]: [external_box_from_json(b) for b in r["boxes"]] for r in load_prediction_records(path)}


def summarize_predictions(path: Path) -> dict[str, dict[str, float]]:
    """Views and boxes per scene in prediction JSONL.

    Parameters
    ----------
    path : pathlib.Path
        A prediction ``.jsonl`` file or a predictions folder.

    Returns
    -------
    dict of str to dict
        ``"<dataset>/<scene>"`` -> ``views``, ``boxes``, and ``min``, ``median``, ``max`` boxes per view, and
        ``median_seconds`` (model forward time per view).
    """
    per_scene: dict[str, list[dict[str, Any]]] = {}
    for record in load_prediction_records(path):
        per_scene.setdefault(f"{record['dataset']}/{record['scene']}", []).append(record)
    summary = {}
    for scene, records in per_scene.items():
        n = np.array([len(r["boxes"]) for r in records])
        summary[scene] = {"views": len(records), "boxes": int(n.sum()), "min": int(n.min()),
                          "median": float(np.median(n)), "max": int(n.max()),
                          "median_seconds": float(np.median([r["seconds"] for r in records]))}
    return summary


# ------------------------------------------------------------------------------------------------------ CLI


def _read_ids(videos: Sequence[str] | None, videos_file: Path | None) -> list[str]:
    """Video ids from the command line and/or a file with one id per line.

    Parameters
    ----------
    videos : sequence of str or None
        Ids given directly.
    videos_file : pathlib.Path or None
        A text file with one id per line (blank lines and ``#`` comments ignored).

    Returns
    -------
    list of str
        The ids, in order, without duplicates.
    """
    ids = list(videos or [])
    if videos_file is not None:
        ids += [line.split("#")[0].strip() for line in Path(videos_file).read_text().splitlines()]
    return list(dict.fromkeys(i for i in ids if i))


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: ``python -m measured_space.cubify {export,summarize} ...`` (project env).

    - ``export --videos <id>... | --videos-file <txt> [--out <frames>] [--overwrite]``: write CuTR input frames
      for every posed upright view of the videos (:func:`export_videos`); ``--out`` defaults to
      ``<data_root>/outputs/cubify/frames``. Resumable per video.
    - ``summarize <predictions>``: print views and boxes per scene of prediction JSONL written by
      ``scripts/run_cubify.py``, as one JSON object.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments without the program name; ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status: 0 on success, 2 if ``export`` is given no video id.
    """
    parser = argparse.ArgumentParser(prog="python -m measured_space.cubify",
                                     description="Cubify Anything (CuTR) frame export and prediction summary.")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="write CuTR input frames for ARKitScenes videos")
    export.add_argument("--videos", nargs="*", default=None, help="ARKitScenes video ids")
    export.add_argument("--videos-file", type=Path, default=None, help="text file with one video id per line")
    export.add_argument("--out", type=Path, default=None,
                        help="frames folder (default <data_root>/outputs/cubify/frames)")
    export.add_argument("--overwrite", action="store_true", help="re-export videos that are already complete")
    summarize = sub.add_parser("summarize", help="views and boxes per scene of prediction JSONL")
    summarize.add_argument("predictions", type=Path, help="a prediction .jsonl file or folder")
    args = parser.parse_args(argv)
    if args.command == "summarize":
        print(json.dumps(summarize_predictions(args.predictions), indent=1))
        return 0
    ids = _read_ids(args.videos, args.videos_file)
    if not ids:
        print("no video ids given (--videos or --videos-file)", file=sys.stderr)
        return 2
    out = args.out if args.out is not None else data_root() / "outputs" / "cubify" / "frames"
    t0 = time.perf_counter()
    counts = export_videos(ids, out, overwrite=args.overwrite)
    print(json.dumps({**counts, "frames": str(out), "seconds": round(time.perf_counter() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
