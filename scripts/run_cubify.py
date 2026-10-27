"""Run Cubify Anything (CuTR RGB-D), unmodified, on exported frames and write its boxes as prediction JSONL.

Runs in the ``mga-cubify`` conda env, with the ``apple/ml-cubifyanything`` checkout (commit ``00e9cb1``) on
``PYTHONPATH``, not in the project env. Its input is the frame ``.npz`` files written by
``python -m measured_space.cubify export`` (project env); its output is one JSONL per scene at
``<out>/<dataset>/<scene>.jsonl``, one line per view, in the format documented in :mod:`measured_space.cubify`
and read there by :func:`measured_space.cubify.read_predictions`.

Per frame, the sample is built the way ml-cubifyanything's own capture path (``capture_stream.py``) builds an
upright one: the image at 4x the depth size, depth in metres, both intrinsics, an identity ``RT`` and
``T_gravity`` (stored in the frame by the export), packaged with ``Augmentor(("wide/image", "wide/depth"))`` and
``Preprocessor()``. The model's 100 top-scoring boxes are filtered to ``score >= --score-threshold`` (default
0.25, the published demo's value). ``seconds`` is the time of the model's forward pass alone (synchronised on
the GPU).

**Why some geometry is duplicated here.** ``measured_space`` is not installed in ``mga-cubify`` (and
``cubifyanything`` cannot be imported in the project env), so this script cannot use the package. The export
stores everything convention-laden that the model needs (``T_gravity``, intrinsics, ``view_key``), which leaves
only four small pieces of geometry, copied here: the nearest rotation to CuTR's float32 ``R``, the move of a
camera-frame box to the world with ``T_wc``, the corners of a box (for a self-check), and the scaling of a 2D
box to the full-resolution image. ``tests/test_cubify.py`` checks that these copies agree with
:mod:`measured_space.cubify`. As a self-check on real outputs, every kept box's world corners are compared
with CuTR's own ``pred_boxes_3d.corners`` moved by ``T_wc``; a difference above 1e-5 m stops the run.

**Resuming.** A scene whose JSONL already holds exactly one valid line per input view is skipped; any other
scene is (re)computed and its JSONL written in one go once all its views are done. A complete JSONL written
with another checkpoint or threshold is an error (use another ``--out``).

Example (GPU), from the repository root with the ``mga-cubify`` env's Python::

    PYTHONPATH=<data>/third_party/ml-cubifyanything python scripts/run_cubify.py \\
        <data>/outputs/cubify/frames --out <data>/outputs/cubify/predictions \\
        --checkpoint <data>/models/cubifyanything/cutr_rgbd.pth

``--device cpu`` runs on the CPU (several seconds per frame; for checks on a few frames).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:  # the model's types are only needed for annotations; imports are lazy (see load_model)
    import torch

MODEL = "cutr_rgbd"
"""str: the model name written to every record (``measured_space.cubify.MODEL``)."""

CHECKPOINT_SHA256 = "856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217"
"""str: SHA-256 of the published ``cutr_rgbd.pth`` (``measured_space.cubify.CHECKPOINT_SHA256``)."""

SCORE_THRESHOLD = 0.25
"""float: default minimum score, the published demo's ``--score-thresh``."""

CORNER_TOLERANCE = 1e-5
"""float: metres; largest allowed gap between a stored box's corners and CuTR's own corners."""

FRAME_KEYS = ("image", "depth", "K_img", "K_depth", "T_wc", "T_gravity", "image_size", "dataset", "scene",
              "camera_id", "frame", "view_key", "depth_source")
"""tuple of str: the arrays of an exported frame."""

# The 8 corners of the unit cube [-1/2, 1/2]^3 (as measured_space.geometry's _UNIT_CORNERS).
_UNIT_CORNERS = np.array([[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (-0.5, 0.5)])


# ------------------------------------------------------------------------------------- duplicated geometry


def nearest_rotation(matrix: ArrayLike) -> NDArray[np.float64]:
    """The rotation nearest to a nearly orthonormal matrix (copy of ``measured_space.cubify.nearest_rotation``).

    Parameters
    ----------
    matrix : array_like, shape (3, 3)
        A matrix close to a proper rotation.

    Returns
    -------
    numpy.ndarray, shape (3, 3), float64
        ``U @ V.T`` from the singular value decomposition ``matrix = U S V.T``.

    Raises
    ------
    ValueError
        If the matrix is a reflection (negative determinant).
    """
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if np.linalg.det(matrix) <= 0:
        raise ValueError(f"not a rotation (determinant {np.linalg.det(matrix):.3g})")
    u, _, vt = np.linalg.svd(matrix)
    return u @ vt


def box_to_world(center_cam: ArrayLike, dims: ArrayLike, rotation_cam: ArrayLike,
                 T_wc: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """A CuTR box in the camera frame as world centre, size and rotation (``measured_space.cubify.cutr_box_to_world``).

    Parameters
    ----------
    center_cam : array_like, shape (3,)
        ``pred_boxes_3d.gravity_center``, camera frame, metres.
    dims : array_like, shape (3,)
        ``pred_boxes_3d.dims`` (``l, h, w``): full side lengths along the box's x, y, z axes, metres.
    rotation_cam : array_like, shape (3, 3)
        ``pred_boxes_3d.R``: box axes as columns, camera frame.
    T_wc : array_like, shape (4, 4)
        Camera-to-world pose.

    Returns
    -------
    center : numpy.ndarray, shape (3,), float64
        World centre, metres.
    size : numpy.ndarray, shape (3,), float64
        Full side lengths, metres (``dims``).
    rotation : numpy.ndarray, shape (3, 3), float64
        Box axes as columns, world frame.
    """
    T_wc = np.asarray(T_wc, dtype=np.float64)
    center = T_wc[:3, :3] @ np.asarray(center_cam, dtype=np.float64).reshape(3) + T_wc[:3, 3]
    rotation = T_wc[:3, :3] @ nearest_rotation(rotation_cam)
    return center, np.asarray(dims, dtype=np.float64).reshape(3).copy(), rotation


def box_corners(center: ArrayLike, size: ArrayLike, rotation: ArrayLike) -> NDArray[np.float64]:
    """The 8 corners of a box (``measured_space.geometry.OrientedBox.corners``).

    Parameters
    ----------
    center : array_like, shape (3,)
        Centre, metres.
    size : array_like, shape (3,)
        Full side lengths along the box axes, metres.
    rotation : array_like, shape (3, 3)
        Box axes as columns.

    Returns
    -------
    numpy.ndarray, shape (8, 3), float64
        Corners in the parent frame.
    """
    local = _UNIT_CORNERS * np.asarray(size, dtype=np.float64).reshape(3)
    return local @ np.asarray(rotation, dtype=np.float64).T + np.asarray(center, dtype=np.float64).reshape(3)


def corner_gap(corners_a: ArrayLike, corners_b: ArrayLike) -> float:
    """The largest distance from a corner of either set to the nearest corner of the other (order-free).

    Parameters
    ----------
    corners_a, corners_b : array_like, shape (8, 3)
        Two corner sets, metres.

    Returns
    -------
    float
        Symmetric Hausdorff distance between the two point sets, metres.
    """
    a, b = np.asarray(corners_a, dtype=np.float64), np.asarray(corners_b, dtype=np.float64)
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def scale_box2d(box2d: Sequence[float], from_size: Sequence[int],
                to_size: Sequence[int]) -> tuple[float, float, float, float]:
    """Scale a 2D box between resolutions, pixel edges at integers (``measured_space.cubify.scale_box2d``).

    Parameters
    ----------
    box2d : sequence of float, shape (4,)
        ``(xmin, ymin, xmax, ymax)`` in pixels of the source image.
    from_size : sequence of int, shape (2,)
        ``(W, H)`` of the source image.
    to_size : sequence of int, shape (2,)
        ``(W, H)`` of the target image.

    Returns
    -------
    tuple of float
        The box in pixels of the target image.
    """
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    x0, y0, x1, y1 = (float(v) for v in box2d)
    return x0 * sx, y0 * sy, x1 * sx, y1 * sy


# ------------------------------------------------------------------------------------------------ records


def view_record(frame: Mapping[str, Any], centers_cam: ArrayLike, dims: ArrayLike, rotations_cam: ArrayLike,
                scores: ArrayLike, boxes2d: ArrayLike, corners_cam: ArrayLike | None, seconds: float,
                checkpoint_sha256: str, score_threshold: float) -> dict[str, Any]:
    """One view's prediction record from CuTR's kept outputs (pure numpy, so it is testable without the model).

    Parameters
    ----------
    frame : mapping
        The exported frame's arrays (at least ``image``, ``T_wc``, ``image_size`` and the identity strings).
    centers_cam : array_like, shape (N, 3)
        ``pred_boxes_3d.gravity_center`` of the kept boxes, camera frame, metres.
    dims : array_like, shape (N, 3)
        ``pred_boxes_3d.dims``, metres.
    rotations_cam : array_like, shape (N, 3, 3)
        ``pred_boxes_3d.R``.
    scores : array_like, shape (N,)
        ``scores``.
    boxes2d : array_like, shape (N, 4)
        ``pred_boxes`` (xyxy, pixels of the model's input image).
    corners_cam : array_like, shape (N, 8, 3), or None
        ``pred_boxes_3d.corners`` for the self-check, or ``None`` to skip it.
    seconds : float
        Model forward time for the view.
    checkpoint_sha256 : str
        SHA-256 of the checkpoint used.
    score_threshold : float
        The threshold the boxes passed.

    Returns
    -------
    dict
        The JSONL record (keys in the order of the documented format).

    Raises
    ------
    RuntimeError
        If a box's world corners differ from CuTR's corners moved by ``T_wc`` by more than
        :data:`CORNER_TOLERANCE`.
    """
    T_wc = np.asarray(frame["T_wc"], dtype=np.float64)
    model_size = (int(frame["image"].shape[1]), int(frame["image"].shape[0]))
    image_size = [int(v) for v in np.asarray(frame["image_size"]).ravel()]
    centers_cam = np.asarray(centers_cam, dtype=np.float64).reshape(-1, 3)
    dims = np.asarray(dims, dtype=np.float64).reshape(-1, 3)
    rotations_cam = np.asarray(rotations_cam, dtype=np.float64).reshape(-1, 3, 3)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    boxes2d = np.asarray(boxes2d, dtype=np.float64).reshape(-1, 4)
    boxes = []
    for i in range(len(scores)):
        center, size, rotation = box_to_world(centers_cam[i], dims[i], rotations_cam[i], T_wc)
        if corners_cam is not None:
            theirs = np.asarray(corners_cam, dtype=np.float64)[i] @ T_wc[:3, :3].T + T_wc[:3, 3]
            gap = corner_gap(box_corners(center, size, rotation), theirs)
            if gap > CORNER_TOLERANCE:
                raise RuntimeError(f"{frame['view_key']}: box {i} corners differ from CuTR's by {gap:.2e} m")
        boxes.append({"center": center.tolist(), "size": size.tolist(), "rotation": rotation.tolist(),
                      "score": float(scores[i]), "box2d": list(scale_box2d(boxes2d[i], model_size, image_size)),
                      "center_cam": centers_cam[i].tolist()})
    return {"view_key": str(frame["view_key"]), "dataset": str(frame["dataset"]), "scene": str(frame["scene"]),
            "camera_id": str(frame["camera_id"]), "frame": str(frame["frame"]), "model": MODEL,
            "checkpoint_sha256": checkpoint_sha256, "score_threshold": float(score_threshold),
            "image_size": image_size, "seconds": float(seconds), "boxes": boxes}


def load_frame(path: Path) -> dict[str, Any]:
    """Read an exported frame.

    Parameters
    ----------
    path : pathlib.Path
        A frame ``.npz`` from ``python -m measured_space.cubify export``.

    Returns
    -------
    dict
        :data:`FRAME_KEYS` -> arrays; string fields as ``str``.

    Raises
    ------
    ValueError
        If a key is missing or the depth is not iPad LiDAR.
    """
    with np.load(path) as data:
        missing = [k for k in FRAME_KEYS if k not in data.files]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        frame = {k: data[k] for k in FRAME_KEYS}
    for k in ("dataset", "scene", "camera_id", "frame", "view_key", "depth_source"):
        frame[k] = str(frame[k])
    if frame["depth_source"] != "lidar":
        raise ValueError(f"{path}: depth source {frame['depth_source']!r}; only iPad LiDAR is allowed")
    return frame


def frame_identity(path: Path) -> tuple[str, str, str, str]:
    """The dataset, scene, frame and view key of an exported frame, reading nothing else.

    Parameters
    ----------
    path : pathlib.Path
        A frame ``.npz``.

    Returns
    -------
    tuple of str
        ``(dataset, scene, frame, view_key)``.
    """
    with np.load(path) as data:
        return str(data["dataset"]), str(data["scene"]), str(data["frame"]), str(data["view_key"])


def _frame_order(frame: str) -> tuple[float, str]:
    """Sort key of a frame name: numeric where possible (timestamps), then text.

    Parameters
    ----------
    frame : str
        Frame identifier.

    Returns
    -------
    tuple
        ``(value, text)``; non-numeric frames sort after numeric ones.
    """
    try:
        return float(frame), frame
    except ValueError:
        return float("inf"), frame


def group_frames(inputs: Sequence[Path]) -> dict[tuple[str, str], list[tuple[str, Path]]]:
    """Exported frames grouped by scene, in time order.

    Parameters
    ----------
    inputs : sequence of pathlib.Path
        Frame ``.npz`` files and/or folders searched recursively for them.

    Returns
    -------
    dict
        ``(dataset, scene)`` -> list of ``(view_key, path)``, sorted by frame.

    Raises
    ------
    ValueError
        If two files hold the same view.
    FileNotFoundError
        If an input does not exist.
    """
    files: list[Path] = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += sorted(p.rglob("*.npz"))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    groups: dict[tuple[str, str], list[tuple[str, str, Path]]] = {}
    seen: dict[str, Path] = {}
    for f in files:
        dataset, scene, frame, key = frame_identity(f)
        if key in seen:
            raise ValueError(f"{f} and {seen[key]} both hold {key}")
        seen[key] = f
        groups.setdefault((dataset, scene), []).append((frame, key, f))
    return {g: [(k, f) for _, k, f in sorted(v, key=lambda t: _frame_order(t[0]))] for g, v in groups.items()}


def scene_done(path: Path, view_keys: Sequence[str], checkpoint_sha256: str, score_threshold: float) -> bool:
    """Whether a scene's JSONL already holds exactly one valid record per view, from this configuration.

    Parameters
    ----------
    path : pathlib.Path
        ``<out>/<dataset>/<scene>.jsonl``.
    view_keys : sequence of str
        The views of the scene in this run.
    checkpoint_sha256 : str
        Checkpoint of this run.
    score_threshold : float
        Threshold of this run.

    Returns
    -------
    bool
        True if the file exists, every line parses, and its view keys are exactly ``view_keys``.

    Raises
    ------
    RuntimeError
        If the file is complete but was written with another checkpoint or threshold.
    """
    if not path.exists():
        return False
    try:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError:
        return False
    if sorted(r.get("view_key", "") for r in records) != sorted(view_keys):
        return False
    if any(r.get("checkpoint_sha256") != checkpoint_sha256 or r.get("score_threshold") != score_threshold
           for r in records):
        raise RuntimeError(f"{path} was written with another checkpoint or score threshold; use another --out")
    return True


# -------------------------------------------------------------------------------------------------- model


def sha256(path: Path) -> str:
    """SHA-256 of a file.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint: Path, device: str) -> torch.nn.Module:
    """Build CuTR as the published demo does and load the RGB-D checkpoint.

    Parameters
    ----------
    checkpoint : pathlib.Path
        ``cutr_rgbd.pth``.
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    torch.nn.Module
        The model in evaluation mode on ``device``.

    Raises
    ------
    ValueError
        If the checkpoint is not an RGB-D model.
    """
    import torch
    from cubifyanything.cubify_transformer import make_cubify_transformer

    state = torch.load(checkpoint, map_location="cpu")["model"]
    if not any(k.startswith("backbone.0.patch_embed_depth.") for k in state):
        raise ValueError(f"{checkpoint} is not an RGB-D model")
    dimension = state["backbone.0.patch_embed.proj.weight"].shape[0]
    model = make_cubify_transformer(dimension=dimension, depth_model=True).eval()
    model.load_state_dict(state)
    return model.to(device)


def run_frame(model: torch.nn.Module, frame: Mapping[str, Any], device: str) -> tuple[Any, float]:
    """Run CuTR on one exported frame.

    The sample is the one ml-cubifyanything's capture path yields for an upright frame: image (4x the depth size)
    and depth with their intrinsics, ``RT`` the identity and ``T_gravity`` from the export, packaged by
    ``Augmentor(("wide/image", "wide/depth"))``, moved to the model's device and preprocessed by
    ``Preprocessor()``.

    Parameters
    ----------
    model : torch.nn.Module
        From :func:`load_model`.
    frame : mapping
        From :func:`load_frame`.
    device : str
        The model's device (``"cuda"`` synchronises around the timed forward pass).

    Returns
    -------
    instances : cubifyanything.instances.Instances3D
        CuTR's output for the frame (top 100 boxes, on ``device``).
    seconds : float
        Time of the forward pass, synchronised on the GPU.
    """
    import torch
    from cubifyanything.measurement import DepthMeasurementInfo, ImageMeasurementInfo
    from cubifyanything.preprocessor import Augmentor, Preprocessor, move_input_to_current_device
    from cubifyanything.sensor import PosedSensorInfo, SensorArrayInfo

    H, W = frame["image"].shape[:2]
    h, w = frame["depth"].shape
    wide = PosedSensorInfo()
    wide.RT = torch.eye(4)[None]
    wide.image = ImageMeasurementInfo(size=(W, H), K=torch.tensor(frame["K_img"], dtype=torch.float32)[None])
    wide.depth = DepthMeasurementInfo(size=(w, h), K=torch.tensor(frame["K_depth"], dtype=torch.float32)[None])
    wide.T_gravity = torch.tensor(frame["T_gravity"], dtype=torch.float32)[None]
    info = SensorArrayInfo()
    info.wide = wide
    sample = {"sensor_info": info,
              "wide": {"image": torch.tensor(np.moveaxis(frame["image"], -1, 0))[None],
                       "depth": torch.tensor(frame["depth"], dtype=torch.float32)[None]},
              "meta": {"video_id": 0, "timestamp": 0.0}}
    packaged = Augmentor(("wide/image", "wide/depth")).package(sample)
    packaged = move_input_to_current_device(packaged, model.pixel_mean)      # as tools/demo.py does
    packaged = Preprocessor().preprocess([packaged])
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        instances = model(packaged)[0]
    if device == "cuda":
        torch.cuda.synchronize()
    return instances, time.perf_counter() - t0


def predict_view(model: torch.nn.Module, frame: Mapping[str, Any], device: str, checkpoint_sha256: str,
                 score_threshold: float) -> dict[str, Any]:
    """CuTR's kept boxes for one frame, as a prediction record.

    Parameters
    ----------
    model : torch.nn.Module
        From :func:`load_model`.
    frame : mapping
        From :func:`load_frame`.
    device : str
        The model's device.
    checkpoint_sha256 : str
        SHA-256 of the checkpoint.
    score_threshold : float
        Minimum score of a kept box.

    Returns
    -------
    dict
        :func:`view_record` of the boxes with ``score >= score_threshold``.
    """
    instances, seconds = run_frame(model, frame, device)
    keep = instances.scores >= score_threshold
    boxes3d = instances.pred_boxes_3d[keep]
    return view_record(frame, boxes3d.gravity_center.cpu().numpy(), boxes3d.dims.cpu().numpy(),
                       boxes3d.R.cpu().numpy(), instances.scores[keep].cpu().numpy(),
                       instances.pred_boxes[keep].cpu().numpy(), boxes3d.corners.cpu().numpy(), seconds,
                       checkpoint_sha256, score_threshold)


# ---------------------------------------------------------------------------------------------------- main


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments without the program name; ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        ``inputs``, ``out``, ``checkpoint``, ``device``, ``score_threshold``.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("inputs", nargs="+", type=Path, help="frame .npz files and/or folders searched recursively")
    p.add_argument("--out", type=Path, required=True, help="predictions folder (<out>/<dataset>/<scene>.jsonl)")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="cutr_rgbd.pth (its SHA-256 must be the published one)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="cuda (default) or cpu")
    p.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD,
                   help="keep boxes with score >= this (default 0.25, the published demo's)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run CuTR on every input frame and write one prediction JSONL per scene.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments without the program name; ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status: 0 on success, 2 if the checkpoint is not the published ``cutr_rgbd.pth`` or CUDA is missing.
    """
    args = parse_args(argv)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; use --device cpu", file=sys.stderr)
        return 2
    checkpoint_sha256 = sha256(args.checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        print(f"{args.checkpoint}: SHA-256 {checkpoint_sha256} is not the published cutr_rgbd.pth",
              file=sys.stderr)
        return 2
    groups = group_frames(args.inputs)
    todo = {g: v for g, v in groups.items()
            if not scene_done(args.out / g[0] / f"{g[1]}.jsonl", [k for k, _ in v], checkpoint_sha256,
                              args.score_threshold)}
    print(f"{sum(len(v) for v in groups.values())} views in {len(groups)} scenes; "
          f"{len(groups) - len(todo)} scenes already done", flush=True)
    if not todo:
        return 0
    model = load_model(args.checkpoint, args.device)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()
    for n, ((dataset, scene), frames) in enumerate(sorted(todo.items()), 1):
        records = [predict_view(model, load_frame(path), args.device, checkpoint_sha256, args.score_threshold)
                   for _, path in frames]
        out = args.out / dataset / f"{scene}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in records))
        n_boxes = [len(r["boxes"]) for r in records]
        print(f"[{n}/{len(todo)}] {dataset}/{scene}: {len(records)} views, {sum(n_boxes)} boxes "
              f"(median {np.median(n_boxes):.1f}/view), median forward "
              f"{np.median([r['seconds'] for r in records]):.3f} s", flush=True)
    summary = {"scenes": len(todo), "views": sum(len(v) for v in todo.values()),
               "seconds": round(time.perf_counter() - t_start, 1)}
    if args.device == "cuda":
        summary["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
