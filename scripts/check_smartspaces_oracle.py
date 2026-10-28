"""Oracle sanity check for the Smart Spaces loader, and the evidence for its file conventions.

The oracle side of the audit trusts the simulator's boxes, so the loader's reading of them has to be shown
right: boxes projected through the calibration must land on the ground truth's own 2D boxes, and the depth
maps must agree with the boxes where the boxes are visible. The script reports, for a synthetic scene:

- reprojection IoU, per camera and class, of (a) the full projected 3D box and (b) the depth-clipped box from
  :func:`measured_space.visibility.object_in_view`, against the ground truth's visible 2D box, and the share
  of that 2D box covered by the projected 3D box (the 2D box is the tight silhouette of the visible part, so
  IoU is well below 1 even for exact boxes, while coverage should be near 1);
- depth agreement at visible box-surface samples (``|depth - sample z|``, and the share within the
  visibility depth tolerance). Most of a box's surface is free space in front of the object (a pallet
  truck's box is mostly air), so this is large by construction; the floor test below checks the depth
  encoding itself;
- agreement between our visibility (visible fraction > 0.1) and the ground truth's per-camera visibility;
- camera heights above the floor (floor height from the depth map);
- convention evidence: reprojection IoU and depth-point containment for rival readings of the box fields,
  floor flatness for planar vs ray depth, the yaw distribution per class, and heading vs direction of travel;

and, for the real test scenes, camera ids, image sizes, heights and a decoded frame per camera.

Usage::

    python scripts/check_smartspaces_oracle.py [--step 30] [--cameras Camera_0000 Camera_0002]

Writes ``<data root>/outputs/oracle_sanity/smartspaces.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from measured_space.config import data_root
from measured_space.datasets import smartspaces as ss
from measured_space.geometry import Array, OrientedBox
from measured_space.scene import FrameView
from measured_space.visibility import depth_tolerance, object_in_view

SYNTHETIC_SCENE = "MTMC_Tracking_2026/train/Warehouse_018"
REAL_SCENES = ("MTMC_Tracking_2026/test/Warehouse_026", "MTMC_Tracking_2026/test/Warehouse_027")
VISIBLE_THRESHOLD = 0.1
SPACING = 0.05                  # surface-sample spacing (m), as in visibility.object_in_view's default
CONTAINMENT_STEP = 300          # frame step for the depth-containment test (one depth read per frame)
CONTAINMENT_RADIUS = 2.5        # keep depth points within this horizontal distance of the box centre (m)

Box4 = tuple[float, float, float, float]
# Rival readings of (scale, [pitch, roll, yaw]) -> (full size, rotation). LOADER is what the loader uses.
LOADER = "A: size=(s0,s1,s2), xyz-Euler radians (loader)"
CANDIDATES: dict[str, Callable[[Array, Array], tuple[Array, Array]]] = {
    LOADER: lambda s, r: (s, Rotation.from_euler("xyz", r).as_matrix()),
    "B: size=(s1,s0,s2)": lambda s, r: (s[[1, 0, 2]], Rotation.from_euler("xyz", r).as_matrix()),
    "C: size=(s0,s2,s1)": lambda s, r: (s[[0, 2, 1]], Rotation.from_euler("xyz", r).as_matrix()),
    "D: angles in degrees": lambda s, r: (s, Rotation.from_euler("xyz", np.radians(r)).as_matrix()),
    "E: yaw clockwise": lambda s, r: (s, Rotation.from_euler("xyz", -r).as_matrix()),
}


def iou(a: Box4, b: Box4) -> float:
    """Intersection over union of two axis-aligned boxes.

    Parameters
    ----------
    a, b : tuple of 4 float
        ``(xmin, ymin, xmax, ymax)`` in continuous pixel coordinates.

    Returns
    -------
    float
        IoU in [0, 1]; 0 if either box is empty.
    """
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def coverage(outer: Box4, inner: Box4) -> float:
    """Share of one box's area that lies inside another.

    Parameters
    ----------
    outer : tuple of 4 float
        ``(xmin, ymin, xmax, ymax)`` of the covering box, continuous pixel coordinates.
    inner : tuple of 4 float
        ``(xmin, ymin, xmax, ymax)`` of the covered box.

    Returns
    -------
    float
        ``area(outer & inner) / area(inner)`` in [0, 1]; 0 if ``inner`` is empty.
    """
    w = max(0.0, min(outer[2], inner[2]) - max(outer[0], inner[0]))
    h = max(0.0, min(outer[3], inner[3]) - max(outer[1], inner[1]))
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return w * h / area if area > 0 else 0.0


def gt_extent(box: ss.Box2D) -> Box4:
    """Continuous extent of a ground-truth box given as inclusive pixel indices.

    Parameters
    ----------
    box : tuple of 4 int
        ``(xmin, ymin, xmax, ymax)``, inclusive pixel indices (pixel centres at integers).

    Returns
    -------
    tuple of 4 float
        The pixels' outer edges, ``(xmin - 0.5, ymin - 0.5, xmax + 0.5, ymax + 0.5)``.
    """
    return box[0] - 0.5, box[1] - 0.5, box[2] + 0.5, box[3] + 0.5


def projected_extent(view: FrameView, box: OrientedBox, T_cw: Array | None = None) -> Box4 | None:
    """Image-clipped 2D box around the projection of a 3D box's eight corners.

    Parameters
    ----------
    view : FrameView
        The view (its camera, and its pose unless ``T_cw`` is given).
    box : OrientedBox
        World-frame box.
    T_cw : numpy.ndarray, shape (4, 4), float64, optional
        World-to-camera pose overriding the view's.

    Returns
    -------
    tuple of 4 float or None
        ``(xmin, ymin, xmax, ymax)`` clipped to the image's outer pixel edges, or None if a corner is
        behind the camera or the box misses the image.
    """
    corners = box.transformed(view.T_cw if T_cw is None else T_cw).corners()
    if (corners[:, 2] <= 0.05).any():
        return None
    uv, _ = view.camera.project(corners)
    lo = np.maximum(uv.min(0), -0.5)
    hi = np.minimum(uv.max(0), [view.camera.width - 0.5, view.camera.height - 0.5])
    if (hi <= lo).any():
        return None
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def visible_sample_depths(view: FrameView, box: OrientedBox,
                          depth: NDArray[np.float32]) -> tuple[Array, Array]:
    """Depth-map value and sample depth at the visible camera-facing surface samples of a box.

    Uses the same sampling and occlusion rule as :func:`measured_space.visibility.object_in_view`.

    Parameters
    ----------
    view : FrameView
        The view.
    box : OrientedBox
        World-frame box.
    depth : numpy.ndarray, shape (H, W), float32
        Planar depth in metres at the camera's resolution; 0 marks invalid.

    Returns
    -------
    d : numpy.ndarray, shape (M,), float64
        Depth-map value at each visible sample (m).
    z : numpy.ndarray, shape (M,), float64
        The sample's own depth along the optical axis (m).
    """
    box_cam = box.transformed(view.T_cw)
    points, normals = box_cam.sample_surface(SPACING)
    points = points[np.einsum("ij,ij->i", normals, -points) > 0]
    uv, z = view.camera.project(points)
    inside = view.camera.in_image(uv) & (z > 0.05)
    cols = np.clip(np.floor(uv[inside, 0] + 0.5).astype(int), 0, view.camera.width - 1)
    rows = np.clip(np.floor(uv[inside, 1] + 0.5).astype(int), 0, view.camera.height - 1)
    d, z = depth[rows, cols].astype(np.float64), z[inside]
    keep = (d > 0) & (d >= z - depth_tolerance(z))
    return d[keep], z[keep]


def summary(values: Sequence[float]) -> dict[str, float | int | None]:
    """Count, median and 10th percentile of a sample.

    Parameters
    ----------
    values : Sequence[float]
        The values.

    Returns
    -------
    dict[str, float | int | None]
        ``{"n", "median", "p10"}``; the statistics are None for an empty sample.
    """
    if len(values) == 0:
        return {"n": 0, "median": None, "p10": None}
    return {"n": len(values), "median": float(np.median(values)), "p10": float(np.percentile(values, 10))}


def oracle_sanity(scene: str, cameras: Sequence[str], frames: Sequence[int]) -> dict[str, Any]:
    """Reprojection, depth-agreement and visibility-agreement statistics per camera and class.

    Parameters
    ----------
    scene : str
        Synthetic scene path below the dataset folder.
    cameras : Sequence[str]
        Cameras with video and depth.
    frames : Sequence[int]
        Frame indices to sample.

    Returns
    -------
    dict[str, Any]
        Camera id to class to statistics: ``iou_full`` and ``iou_clipped`` summaries, ``gt_covered`` (share
        of the ground-truth box inside the full projected box; near 1 when the calibration is right, since
        the visible part of an object lies inside its 3D box), ``abs_depth_diff_m``
        (pooled median of ``|d - z|`` at visible samples), ``on_surface_fraction`` (share of visible samples
        within the depth tolerance), ``no_depth_fraction`` (mean over object views) and a ``visibility``
        confusion of ours vs the ground truth.
    """
    gt, visible = ss.load_ground_truth(scene), ss.visible_boxes_2d(scene)
    result: dict[str, Any] = {}
    for camera_id in cameras:
        stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        confusion: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0})
        for view in ss.list_frames(scene, camera_id, frames):
            depth = ss.load_depth(view)
            frame = int(view.frame)
            for obj in gt[frame]:
                gt_box = visible[frame].get(obj.object_id, {}).get(camera_id)
                seen, diag = object_in_view(view, obj, depth, spacing=SPACING)
                ours = seen.visible_fraction > VISIBLE_THRESHOLD
                key = ("tp" if ours else "fn") if gt_box is not None else ("fp" if ours else "tn")
                confusion[obj.label][key] += 1
                if gt_box is None:
                    continue
                s = stats[obj.label]
                full = projected_extent(view, obj.box)
                s["iou_full"].append(iou(full, gt_extent(gt_box)) if full else 0.0)
                s["gt_covered"].append(coverage(full, gt_extent(gt_box)) if full else 0.0)
                s["iou_clipped"].append(iou(seen.bbox2d, gt_extent(gt_box)) if seen.bbox2d else 0.0)
                s["no_depth"].append(diag["no_depth_fraction"])
                d, z = visible_sample_depths(view, obj.box, depth)
                s["abs_dz"].extend(np.abs(d - z))
                s["on_surface"].extend(np.abs(d - z) <= depth_tolerance(z))
        result[camera_id] = {
            label: {"iou_full": summary(s["iou_full"]), "iou_clipped": summary(s["iou_clipped"]),
                    "gt_covered": summary(s["gt_covered"]),
                    "abs_depth_diff_m": float(np.median(s["abs_dz"])) if s["abs_dz"] else None,
                    "on_surface_fraction": float(np.mean(s["on_surface"])) if s["on_surface"] else None,
                    "no_depth_fraction": float(np.mean(s["no_depth"])), "visibility": confusion[label]}
            for label, s in sorted(stats.items())}
    return result


def floor_height(view: FrameView, depth: NDArray[np.float32], ray: bool = False) -> dict[str, float]:
    """World heights of back-projected depth pixels, low percentiles (the floor is the lowest large surface).

    Parameters
    ----------
    view : FrameView
        The view.
    depth : numpy.ndarray, shape (H, W), float32
        Depth map in metres; 0 marks invalid.
    ray : bool, default False
        Read the values as distance along the pixel ray instead of planar z.

    Returns
    -------
    dict[str, float]
        World z (m) at the 1st, 5th, 25th and 50th percentiles.
    """
    rows, cols = np.nonzero(depth > 0)
    d = depth[rows, cols].astype(np.float64)
    if ray:
        x, y = (cols - view.camera.cx) / view.camera.fx, (rows - view.camera.cy) / view.camera.fy
        d = d / np.sqrt(1 + x**2 + y**2)
    world = view.camera.backproject(cols, rows, d) @ view.T_wc[:3, :3].T + view.T_wc[:3, 3]
    return {f"p{q}": float(np.percentile(world[:, 2], q)) for q in (1, 5, 25, 50)}


def convention_evidence(scene: str, cameras: Sequence[str], frames: Sequence[int]) -> dict[str, Any]:
    """Compare rival readings of the box fields and of the extrinsic matrix against the data.

    Parameters
    ----------
    scene : str
        Synthetic scene path below the dataset folder.
    cameras : Sequence[str]
        Cameras with video and depth; IoU also uses every other calibrated camera, since it needs no images.
    frames : Sequence[int]
        Frame indices for the IoU test; the containment test uses every ``CONTAINMENT_STEP``-th frame.

    Returns
    -------
    dict[str, Any]
        ``iou`` (candidate to class to IoU summary of the full projected box), ``containment`` (candidate to
        class to summary of the share of nearby above-floor depth points inside the box) and ``extrinsic``
        (class to IoU summary with the extrinsic read as camera to world).
    """
    raw = json.loads((ss.scene_dir(scene) / "ground_truth.json").read_text())
    calibration = ss.load_calibration(scene)
    views = {cam: ss.list_frames(scene, cam, [0])[0] for cam in calibration}
    ious: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    contained: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for frame in frames:
        for record in raw[str(frame)]:
            label = record["object type"]
            center = np.asarray(record["3d location"], dtype=np.float64)
            scale = np.asarray(record["3d bounding box scale"], dtype=np.float64)
            angles = np.asarray(record["3d bounding box rotation"], dtype=np.float64)
            boxes = {name: OrientedBox(center, *fn(scale, angles)) for name, fn in CANDIDATES.items()}
            for cam, box2d in record["2d bounding box visible"].items():
                view, target = views[cam], gt_extent(tuple(box2d))
                for name, box in boxes.items():
                    extent = projected_extent(view, box)
                    ious[name][label].append(iou(extent, target) if extent else 0.0)
                extent = projected_extent(view, boxes[LOADER], T_cw=view.T_wc)
                ious["extrinsic read as camera->world"][label].append(iou(extent, target) if extent else 0.0)
                if cam in cameras and frame % CONTAINMENT_STEP == 0:
                    points = _box_depth_points(ss.list_frames(scene, cam, [frame])[0], box2d, center)
                    if len(points) >= 50:
                        for name, box in boxes.items():
                            contained[name][label].append(float(box.contains(points, margin=0.05).mean()))
    extrinsic = ious.pop("extrinsic read as camera->world")
    return {"iou": {k: _per_class(t) for k, t in ious.items()},
            "containment": {k: _per_class(t) for k, t in contained.items()},
            "extrinsic_as_camera_to_world_iou": _per_class(extrinsic)}


def _per_class(values: dict[str, list[float]]) -> dict[str, dict[str, float | int | None]]:
    """Summaries of per-class samples, classes sorted.

    Parameters
    ----------
    values : dict[str, list[float]]
        Class to sample.

    Returns
    -------
    dict[str, dict[str, float | int | None]]
        Class to :func:`summary` of its sample.
    """
    return {label: summary(v) for label, v in sorted(values.items())}


def _box_depth_points(view: FrameView, box2d: Sequence[int], center: Array) -> Array:
    """World points from the depth pixels inside a ground-truth 2D box, above the floor and near the object.

    Parameters
    ----------
    view : FrameView
        The view.
    box2d : Sequence[int]
        ``(xmin, ymin, xmax, ymax)``, inclusive pixel indices.
    center : numpy.ndarray, shape (3,), float64
        Annotated box centre (m); points farther than ``CONTAINMENT_RADIUS`` horizontally are dropped.

    Returns
    -------
    numpy.ndarray, shape (N, 3), float64
        World points (m) with z > 0.05.
    """
    x0, y0, x1, y1 = box2d
    depth = ss.load_depth(view)[y0:y1 + 1, x0:x1 + 1]
    rows, cols = np.nonzero(depth > 0)
    points = view.camera.backproject(cols + x0, rows + y0, depth[rows, cols])
    world = points @ view.T_wc[:3, :3].T + view.T_wc[:3, 3]
    keep = (world[:, 2] > 0.05) & (np.linalg.norm(world[:, :2] - center[:2], axis=1) < CONTAINMENT_RADIUS)
    return world[keep]


def yaw_and_heading(scene: str) -> dict[str, Any]:
    """Yaw distribution per class over every frame, and heading vs direction of travel per moving object.

    Parameters
    ----------
    scene : str
        Synthetic scene path below the dataset folder.

    Returns
    -------
    dict[str, Any]
        ``yaw`` maps class to ``{"objects", "distinct_yaw", "deg_p0/p25/p50/p75/p100"}`` (distinct values at
        0.001 rad resolution); ``heading`` maps each object that moves in at least 10 frame steps (more than
        5 mm per frame) to the share of steps whose direction of travel lies within 20 degrees of each box
        axis (``+x``, ``-x``, ``+y``, ``-y``).
    """
    tracks: dict[str, list[tuple[int, OrientedBox]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for frame, objects in ss.load_ground_truth(scene).items():
        for obj in objects:
            tracks[obj.object_id].append((frame, obj.box))
            labels[obj.object_id] = obj.label
    yaws: dict[str, list[float]] = defaultdict(list)
    heading: dict[str, dict[str, float]] = {}
    for object_id, track in tracks.items():
        track.sort(key=lambda item: item[0])
        rotations = np.stack([box.rotation for _, box in track])
        yaws[labels[object_id]].extend(np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0]))
        frames = np.array([f for f, _ in track])
        centers = np.stack([box.center for _, box in track])
        step = np.diff(centers[:, :2], axis=0)
        moving = (np.diff(frames) == 1) & (np.linalg.norm(step, axis=1) > 0.005)
        if moving.sum() < 10:
            continue
        direction = step[moving] / np.linalg.norm(step[moving], axis=1, keepdims=True)
        axes = rotations[1:][moving][:, :2, :]           # box axes' horizontal parts, (M, 2, 3)
        heading[object_id] = {}
        for name, axis, sign in (("+x", 0, 1), ("-x", 0, -1), ("+y", 1, 1), ("-y", 1, -1)):
            a = sign * axes[:, :, axis]
            cos = np.einsum("ij,ij->i", direction, a) / np.linalg.norm(a, axis=1)
            heading[object_id][name] = float(np.mean(cos > np.cos(np.radians(20))))
        heading[object_id]["steps"] = int(moving.sum())
    yaw = {}
    for label, values in sorted(yaws.items()):
        deg = np.degrees(values)
        yaw[label] = {"objects": sum(1 for o in labels if labels[o] == label),
                      "distinct_yaw": int(len(np.unique(np.round(values, 3)))),
                      **{f"deg_p{q}": float(np.percentile(deg, q)) for q in (0, 25, 50, 75, 100)}}
    return {"yaw": yaw, "heading": heading}


def camera_heights(scene: str, depth_cameras: Sequence[str]) -> dict[str, dict[str, float | None]]:
    """Camera centre heights, and the floor height seen in frame 0's depth where depth exists.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder.
    depth_cameras : Sequence[str]
        Cameras whose depth map is available.

    Returns
    -------
    dict[str, dict[str, float | None]]
        Camera id to ``camera_z`` (world z of the centre, m), ``floor_z`` (5th percentile of back-projected
        world z, m, or None) and ``height`` (``camera_z - floor_z``, or ``camera_z`` without depth), plus
        the planar-vs-ray floor percentiles for depth cameras.
    """
    out: dict[str, dict[str, Any]] = {}
    for camera_id, (_, T_wc) in sorted(ss.load_calibration(scene).items()):
        entry: dict[str, Any] = {"camera_z": float(T_wc[2, 3]), "floor_z": None, "height": float(T_wc[2, 3])}
        if camera_id in depth_cameras:
            view = ss.list_frames(scene, camera_id, [0])[0]
            depth = ss.load_depth(view)
            planar, ray = floor_height(view, depth), floor_height(view, depth, ray=True)
            entry.update(floor_z=planar["p5"], height=float(T_wc[2, 3]) - planar["p5"],
                         floor_planar=planar, floor_ray=ray)
        out[camera_id] = entry
    return out


def _reprojection_medians(path: Path) -> dict[str, float]:
    """Per-camera median reprojection errors from a scene's ``vggt_reprojection_error.txt``.

    Parameters
    ----------
    path : pathlib.Path
        The text file.

    Returns
    -------
    dict[str, float]
        ``"Camera_XXXX"`` to the median error (the file does not state its units); empty if absent.
    """
    if not path.is_file():
        return {}
    medians, camera = {}, None
    for line in path.read_text().splitlines():
        if (m := re.search(r"Camera (\d+):", line)):
            camera = f"Camera_{int(m.group(1)):04d}"
        elif camera and (m := re.search(r"Median error:\s*([0-9.]+)", line)):
            medians[camera], camera = float(m.group(1)), None
    return medians


def real_scene_summary(scene: str) -> dict[str, Any]:
    """Cameras of a real scene with their image sizes and heights, decoding one frame per camera.

    Parameters
    ----------
    scene : str
        Real scene path below the dataset folder.

    Returns
    -------
    dict[str, Any]
        Camera id to ``fx``, ``calib_size`` and ``decoded_size`` (``[width, height]``), ``camera_z`` (m above
        world z = 0; the floor plane is not verified for these scenes), ``decode_ok`` and
        ``reprojection_median`` (from the scene's VGGT report, units unstated).
    """
    medians = _reprojection_medians(ss.scene_dir(scene) / "vggt_reprojection_error.txt")
    with_video = set(ss.list_cameras(scene))
    out = {}
    for camera_id, (camera, T_wc) in sorted(ss.load_calibration(scene).items()):
        entry: dict[str, Any] = {"fx": camera.fx, "calib_size": [camera.width, camera.height],
                                 "camera_z": float(T_wc[2, 3]), "decoded_size": None, "decode_ok": False,
                                 "reprojection_median": medians.get(camera_id)}
        if camera_id in with_video:
            image = ss.load_rgb(ss.list_frames(scene, camera_id, [900])[0])
            entry.update(decoded_size=[image.shape[1], image.shape[0]],
                         decode_ok=bool(image.shape[:2] == (camera.height, camera.width) and image.std() > 0))
        out[camera_id] = entry
    return out


def _fmt(value: float | None, digits: int = 3) -> str:
    """Format a number for the printed table.

    Parameters
    ----------
    value : float or None
        The number.
    digits : int, default 3
        Decimals.

    Returns
    -------
    str
        The formatted number, or ``"-"`` for None.
    """
    return "-" if value is None else f"{value:.{digits}f}"


def print_report(report: dict[str, Any]) -> None:
    """Print the compact tables.

    Parameters
    ----------
    report : dict[str, Any]
        The report assembled by :func:`main`.
    """
    print(f"\n{report['scene']}  frames {report['frames'][0]}..{report['frames'][-1]} step {report['step']}")
    print(f"{'camera':12} {'class':12} {'n':>5} {'IoU full med/p10':>17} {'IoU clip med/p10':>17} "
          f"{'GT covered med/p10':>18} {'|d-z| med':>9} {'on-surf':>7} {'no-depth':>8}   vis TP/FN/FP/TN")
    for camera_id, per_class in report["oracle"].items():
        for label, s in per_class.items():
            v = s["visibility"]
            print(f"{camera_id:12} {label:12} {s['iou_full']['n']:5d} "
                  f"{_fmt(s['iou_full']['median']):>8}/{_fmt(s['iou_full']['p10']):<8} "
                  f"{_fmt(s['iou_clipped']['median']):>8}/{_fmt(s['iou_clipped']['p10']):<8} "
                  f"{_fmt(s['gt_covered']['median']):>9}/{_fmt(s['gt_covered']['p10']):<8} "
                  f"{_fmt(s['abs_depth_diff_m']):>9} {_fmt(s['on_surface_fraction'], 2):>7} "
                  f"{_fmt(s['no_depth_fraction'], 3):>8}   {v['tp']}/{v['fn']}/{v['fp']}/{v['tn']}")
    print("\nCamera heights (m): camera z, floor z from depth (p5), height")
    for camera_id, h in report["camera_heights"].items():
        print(f"  {camera_id}: {h['camera_z']:.3f}  {_fmt(h['floor_z'])}  {h['height']:.3f}")
    print("\nDepth convention, world z of back-projected pixels (p1/p5/p25/p50):")
    for camera_id, h in report["camera_heights"].items():
        if h["floor_z"] is not None:
            for kind in ("planar", "ray"):
                print(f"  {camera_id} {kind:6}: " + " ".join(_fmt(x) for x in h[f"floor_{kind}"].values()))
    evidence = report["conventions"]
    for test in ("iou", "containment"):
        print(f"\nBox reading, {test} (median / p10 per class):")
        for name, per_class in evidence[test].items():
            print(f"  {name:46} " + "  ".join(f"{c}: {_fmt(s['median'])}/{_fmt(s['p10'])} (n={s['n']})"
                                               for c, s in per_class.items()))
    print("  extrinsic read as camera->world, IoU: " + "  ".join(
        f"{c}: {_fmt(s['median'])}" for c, s in evidence["extrinsic_as_camera_to_world_iou"].items()))
    print("\nYaw per class (deg): objects, distinct values, p0/p25/p50/p75/p100")
    for label, y in report["yaw"]["yaw"].items():
        print(f"  {label:12} {y['objects']} {y['distinct_yaw']:5d}  " + " ".join(
            f"{y[f'deg_p{q}']:7.1f}" for q in (0, 25, 50, 75, 100)))
    print("Direction of travel within 20 deg of each box axis (share of moving steps):")
    for object_id, h in report["yaw"]["heading"].items():
        print(f"  {object_id:12} steps {h['steps']:5d}  " + "  ".join(f"{k} {h[k]:.3f}" for k in ("+x", "-x", "+y", "-y")))
    for scene, cams in report["real_scenes"].items():
        print(f"\n{scene} (real): camera, calib size, decoded size, fx, camera z (m), decoded, VGGT median err")
        for camera_id, c in cams.items():
            print(f"  {camera_id}: {c['calib_size']} {c['decoded_size']} {c['fx']:.1f} {c['camera_z']:.3f} "
                  f"{c['decode_ok']} {c['reprojection_median']}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run every check, print the tables and write the JSON report.

    Parameters
    ----------
    argv : Sequence[str], optional
        Command-line arguments (default: ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", default=SYNTHETIC_SCENE)
    parser.add_argument("--cameras", nargs="+", default=["Camera_0000", "Camera_0002"])
    parser.add_argument("--step", type=int, default=30, help="frame step for the sample (default 30 = 1 s)")
    parser.add_argument("--real-scenes", nargs="*", default=list(REAL_SCENES))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    frames = sorted(ss.load_ground_truth(args.scene))[::args.step]
    report = {"scene": args.scene, "cameras": args.cameras, "step": args.step, "frames": frames,
              "visible_threshold": VISIBLE_THRESHOLD, "spacing_m": SPACING,
              "oracle": oracle_sanity(args.scene, args.cameras, frames),
              "camera_heights": camera_heights(args.scene, args.cameras),
              "conventions": convention_evidence(args.scene, args.cameras, frames),
              "yaw": yaw_and_heading(args.scene),
              "real_scenes": {scene: real_scene_summary(scene) for scene in args.real_scenes}}
    ss.close_handles()
    print_report(report)
    out = args.out or data_root() / "outputs" / "oracle_sanity" / "smartspaces.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
