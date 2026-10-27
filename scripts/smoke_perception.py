"""Smoke test of the perception stage on a few real ARKitScenes frames: SAM 3 masks lifted with three depths.

For each chosen frame of one ARKitScenes upsampling video this script segments the dataset's classes with
SAM 3, predicts MoGe-2 depth from the RGB alone, and lifts every instance with each available depth source:
the Faro laser depth (``highres_depth``, the independent reference), the iPad LiDAR depth (``lowres_depth``,
256x192, upsampled nearest-neighbour to the RGB resolution) and MoGe-2. It prints detections and lifted boxes
side by side, the per-frame depth agreement with the laser, timings and peak GPU memory, and writes overlay
PNGs and a JSON summary to ``<data_root>/outputs/smoke_perception/``.

Boxes are in the **camera frame** (``T_wc`` = identity) because poses are not read here. Gravity is not known
in the camera frame either, so "up" is approximated from the video's sky direction: ARKitScenes records which
image side the sky is on (``sky_direction``; 42444946 is "Left", confirmed by eye), and up is taken as the
camera axis pointing to that side (Left -> -x). The camera is also pitched, so heights and yaws here are
approximate; this run checks plumbing and relative agreement between depth sources, not accuracy. Both
SAM 3 and MoGe-2 are given the image stood upright (MoGe's metric scale degrades on sideways frames) and
their outputs are rotated back.

Reads the zips directly (no dataset loader): ``data/arkitscenes/upsampling/<video>.zip`` (members
``<video>/{wide,highres_depth,lowres_depth}/<video>_<timestamp>.png``, depth in uint16 millimetres) and
``data/arkitscenes/wide_intrinsics/<video>_wide_intrinsics.zip`` (``wide_intrinsics/<video>_<timestamp>.pincam``
holding ``w h fx fy cx cy``).

Run (GPU, from the repository root)::

    python scripts/smoke_perception.py --frames 9 15 21
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from measured_space.config import data_root, dataset_dir
from measured_space.geometry import Array, Camera, OrientedBox
from measured_space.perception import (
    Instance,
    LiftError,
    MogeDepth,
    Sam3Segmenter,
    labels_for_phrase,
    lift_instance,
    prompt_phrases,
    resample_depth,
    rot90_points,
)

# Sky side of the image -> (up direction in OpenCV camera axes, quarter turns that stand the image upright).
SKY = {
    "left": (np.array([-1.0, 0.0, 0.0]), -1),
    "right": (np.array([1.0, 0.0, 0.0]), 1),
    "up": (np.array([0.0, -1.0, 0.0]), 0),
    "down": (np.array([0.0, 1.0, 0.0]), 2),
}
SOURCES = ("laser", "lidar", "moge")
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]
BOX_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


@dataclass
class Frame:
    """One ARKitScenes upsampling frame.

    Attributes
    ----------
    name : str
        ``<video>_<timestamp>``.
    image : numpy.ndarray, shape (H, W, 3), uint8
        RGB image (1440 x 1920).
    laser : numpy.ndarray, shape (H, W), float64
        Faro laser depth in metres; 0 where missing.
    lidar : numpy.ndarray, shape (h, w), float64
        iPad LiDAR depth in metres at its native 192 x 256.
    camera : Camera
        Intrinsics of the RGB image.
    """

    name: str
    image: NDArray[np.uint8]
    laser: Array
    lidar: Array
    camera: Camera


def read_png(archive: zipfile.ZipFile, member: str) -> NDArray[Any]:
    """Decode one PNG member of a zip archive.

    Parameters
    ----------
    archive : zipfile.ZipFile
        Open archive.
    member : str
        Member path.

    Returns
    -------
    numpy.ndarray
        Decoded image, (H, W, 3) uint8 for colour or (H, W) uint16 for depth.
    """
    return np.array(Image.open(io.BytesIO(archive.read(member))))


def load_frames(video: str, indices: list[int]) -> list[Frame]:
    """Read the chosen frames of one video straight from its zips.

    Parameters
    ----------
    video : str
        ARKitScenes video id.
    indices : list of int
        Positions in the time-sorted list of the video's ``wide`` frames.

    Returns
    -------
    list of Frame
        The frames, in the order requested.
    """
    root = dataset_dir("arkitscenes")
    frames = []
    with zipfile.ZipFile(root / "upsampling" / f"{video}.zip") as up, \
            zipfile.ZipFile(root / "wide_intrinsics" / f"{video}_wide_intrinsics.zip") as intr:
        names = sorted(Path(n).stem for n in up.namelist() if f"{video}/wide/" in n and n.endswith(".png"))
        for index in indices:
            name = names[index]
            w, h, fx, fy, cx, cy = map(float, intr.read(f"wide_intrinsics/{name}.pincam").split())
            frames.append(Frame(
                name=name,
                image=read_png(up, f"{video}/wide/{name}.png")[..., :3],
                laser=read_png(up, f"{video}/highres_depth/{name}.png") / 1000.0,
                lidar=read_png(up, f"{video}/lowres_depth/{name}.png") / 1000.0,
                camera=Camera(fx, fy, cx, cy, int(w), int(h)),
            ))
    return frames


def depth_agreement(depth: Array, reference: Array) -> dict[str, float]:
    """Median absolute and relative difference from the reference depth where both are valid.

    Parameters
    ----------
    depth : numpy.ndarray, shape (H, W), float
        Depth to check (m); NaN or 0 is invalid.
    reference : numpy.ndarray, shape (H, W), float
        Reference depth (m); 0 is invalid.

    Returns
    -------
    dict of str to float
        ``median_abs_m``, ``median_rel`` and ``coverage`` (share of reference-valid pixels compared).
    """
    ref_valid = reference > 0
    both = ref_valid & np.isfinite(depth) & (depth > 0)
    diff = np.abs(depth[both] - reference[both])
    return {"median_abs_m": float(np.median(diff)), "median_rel": float(np.median(diff / reference[both])),
            "coverage": float(both.sum() / max(ref_valid.sum(), 1))}


def box_record(box: OrientedBox | None, features: dict[str, float] | None) -> dict[str, Any] | None:
    """A JSON-ready record of one lifted box.

    Parameters
    ----------
    box : OrientedBox or None
        Lifted box, or ``None`` if lifting failed.
    features : dict of str to float or None
        Its features.

    Returns
    -------
    dict or None
        Centre, size, rotation and features, or ``None``.
    """
    if box is None:
        return None
    return {"center": box.center.tolist(), "size": box.size.tolist(), "rotation": box.rotation.tolist(),
            "features": features}


def draw_overlay(frame: Frame, instances: list[Instance], boxes: list[OrientedBox | None], rot90_k: int,
                 path: Path, scale: float = 0.5) -> None:
    """Save masks, 2D boxes and laser-lifted 3D wireframes over the image, stood upright.

    Parameters
    ----------
    frame : Frame
        The frame.
    instances : list of Instance
        Detections.
    boxes : list of OrientedBox or None
        Camera-frame box lifted from laser depth for each detection (``None`` if lifting failed).
    rot90_k : int
        Quarter turns that stand the image upright.
    path : pathlib.Path
        Output PNG.
    scale : float, optional
        Output resolution relative to the input.
    """
    canvas = frame.image.astype(np.float32)
    for i, inst in enumerate(instances):
        canvas[inst.mask] = 0.55 * canvas[inst.mask] + 0.45 * np.array(COLORS[i % len(COLORS)], np.float32)
    canvas = canvas.astype(np.uint8)
    for i, (inst, box) in enumerate(zip(instances, boxes)):
        color = COLORS[i % len(COLORS)]
        x0, y0, x1, y1 = (int(round(v)) for v in inst.box2d)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3)
        if box is not None:
            uv, z = frame.camera.project(box.corners())
            for a, b in BOX_EDGES:
                if z[a] > 0.05 and z[b] > 0.05:
                    cv2.line(canvas, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)),
                             (255, 255, 255), 3, cv2.LINE_AA)
    upright = np.ascontiguousarray(np.rot90(canvas, rot90_k))
    upright = cv2.resize(upright, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for i, inst in enumerate(instances):
        corners = rot90_points(np.array([[inst.box2d[0], inst.box2d[1]], [inst.box2d[2], inst.box2d[3]]]),
                               rot90_k, frame.image.shape[:2]) * scale
        x, y = corners.min(axis=0)
        text = f"{i}:{inst.phrase} {inst.score:.2f}"
        cv2.putText(upright, text, (int(x) + 4, int(y) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
                    cv2.LINE_AA)
        cv2.putText(upright, text, (int(x) + 4, int(y) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    COLORS[i % len(COLORS)], 2, cv2.LINE_AA)
    Image.fromarray(upright).save(path)


def draw_depths(depths: dict[str, Array], rot90_k: int, path: Path, width: int = 360) -> None:
    """Save the depth sources side by side on a shared colour scale, stood upright.

    Parameters
    ----------
    depths : dict of str to numpy.ndarray
        Source name -> depth (m) at the RGB resolution; NaN or 0 is invalid (drawn black).
    rot90_k : int
        Quarter turns that stand the images upright.
    path : pathlib.Path
        Output PNG.
    width : int, optional
        Width of each panel in pixels.
    """
    valid = np.concatenate([d[np.isfinite(d) & (d > 0)] for d in depths.values()])
    lo, hi = np.percentile(valid, [1, 99])
    panels = []
    for name, depth in depths.items():
        ok = np.isfinite(depth) & (depth > 0)
        scaled = np.clip((np.nan_to_num(depth) - lo) / (hi - lo), 0, 1)
        color = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
        color[~ok] = 0
        color = np.ascontiguousarray(np.rot90(color, rot90_k))
        color = cv2.resize(color, (width, int(round(color.shape[0] * width / color.shape[1]))),
                           interpolation=cv2.INTER_AREA)
        cv2.putText(color, f"{name} ({lo:.1f}-{hi:.1f} m)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(color)
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)


def format_box(box: OrientedBox | None) -> str:
    """One-line summary of a camera-frame box.

    Parameters
    ----------
    box : OrientedBox or None
        Box, or ``None`` if lifting failed.

    Returns
    -------
    str
        Centre and size in metres, or ``"lift failed"``.
    """
    if box is None:
        return "lift failed".ljust(52)
    c, s = box.center, box.size
    return f"c=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) s=({s[0]:.2f},{s[1]:.2f},{s[2]:.2f})".ljust(52)


def main() -> None:
    """Run the smoke test and write its outputs."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", default="42444946")
    parser.add_argument("--frames", type=int, nargs="+", default=[9, 15, 21],
                        help="positions in the video's time-sorted wide frames")
    parser.add_argument("--sky", choices=sorted(SKY), default="left", help="image side the sky is on")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None, help="default <data_root>/outputs/smoke_perception")
    args = parser.parse_args()
    out = args.out or data_root() / "outputs" / "smoke_perception"
    out.mkdir(parents=True, exist_ok=True)
    up, rot90_k = SKY[args.sky]
    phrases = prompt_phrases("arkitscenes")

    frames = load_frames(args.video, args.frames)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    segmenter = Sam3Segmenter()
    t_sam = time.perf_counter() - t0
    t0 = time.perf_counter()
    moge = MogeDepth()
    t_moge = time.perf_counter() - t0
    print(f"loaded SAM 3 in {t_sam:.1f} s, MoGe-2 in {t_moge:.1f} s; {len(phrases)} phrases; "
          f"camera-frame boxes with up = {up.tolist()} (sky {args.sky}, approximate)")

    summary: dict[str, Any] = {"video": args.video, "up_camera_frame": up.tolist(), "phrases": phrases,
                               "load_s": {"sam3": t_sam, "moge": t_moge}, "frames": []}
    for frame in frames:
        t0 = time.perf_counter()
        instances = segmenter.segment(frame.image, phrases, args.score_threshold, rot90_k=rot90_k)
        t_seg = time.perf_counter() - t0
        t0 = time.perf_counter()
        moge_depth, moge_camera = moge.infer(frame.image, frame.camera, rot90_k)
        t_depth = time.perf_counter() - t0
        depths = {"laser": frame.laser, "lidar": resample_depth(frame.lidar, frame.image.shape[:2]),
                  "moge": moge_depth.astype(np.float64)}
        agreement = {name: depth_agreement(depths[name], frame.laser) for name in ("lidar", "moge")}

        t0 = time.perf_counter()
        lifted: dict[str, list[tuple[OrientedBox | None, dict[str, float] | None]]] = {s: [] for s in SOURCES}
        for inst in instances:
            for source in SOURCES:
                try:
                    lifted[source].append(lift_instance(inst.mask, depths[source], frame.camera, np.eye(4), up))
                except LiftError:
                    lifted[source].append((None, None))
        t_lift = time.perf_counter() - t0

        print(f"\n=== {frame.name}: {len(instances)} instances | segment {t_seg:.2f} s, MoGe {t_depth:.2f} s, "
              f"lift {t_lift:.2f} s ({len(instances) * len(SOURCES)} lifts)")
        print(f"    depth vs laser: LiDAR median |dz| {agreement['lidar']['median_abs_m']:.3f} m "
              f"(rel {agreement['lidar']['median_rel']:.3f}); MoGe {agreement['moge']['median_abs_m']:.3f} m "
              f"(rel {agreement['moge']['median_rel']:.3f}); "
              f"MoGe fx {moge_camera.fx:.1f} vs {frame.camera.fx:.1f}")
        print(f"    {'#':>2} {'phrase':<10} {'score':>5} {'mask px':>8}  " +
              "  ".join(f"{s:<52}" for s in SOURCES))
        records = []
        for i, inst in enumerate(instances):
            row = [lifted[s][i][0] for s in SOURCES]
            print(f"    {i:>2} {inst.phrase:<10} {inst.score:5.2f} {int(inst.mask.sum()):>8}  " +
                  "  ".join(format_box(b) for b in row))
            laser_box, laser_features = lifted["laser"][i]
            if laser_box is not None:
                deltas = []
                for source in ("lidar", "moge"):
                    other = lifted[source][i][0]
                    if other is not None:
                        deltas.append(f"{source}: dc {np.linalg.norm(other.center - laser_box.center):.2f} m, "
                                      f"max|ds| {np.abs(other.size - laser_box.size).max():.2f} m")
                f = laser_features
                print(f"       vs laser -> {'; '.join(deltas)} | laser features: n={f['n_points']:.0f} "
                      f"edge={f['boundary_fraction']:.2f} border={f['border_touch']:.0f} "
                      f"inlier={f['inlier_fraction']:.2f} mad_rel={f['depth_mad_rel']:.3f} "
                      f"hull_fill={f['footprint_hull_fill']:.2f}")
            records.append({"phrase": inst.phrase, "labels": labels_for_phrase("arkitscenes", inst.phrase),
                            "score": inst.score, "box2d": inst.box2d, "mask_px": int(inst.mask.sum()),
                            **{s: box_record(*lifted[s][i]) for s in SOURCES}})

        draw_overlay(frame, instances, [b for b, _ in lifted["laser"]], rot90_k, out / f"{frame.name}_overlay.png")
        draw_depths(depths, rot90_k, out / f"{frame.name}_depths.png")
        summary["frames"].append({"frame": frame.name, "timing_s": {"segment": t_seg, "moge": t_depth,
                                                                    "lift_all": t_lift},
                                  "depth_vs_laser": agreement, "instances": records})

    peak = {"max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    summary["vram_peak"] = peak
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\npeak torch VRAM: {peak['max_allocated_mib']:.0f} MiB allocated, "
          f"{peak['max_reserved_mib']:.0f} MiB reserved; outputs in {out}")


if __name__ == "__main__":
    main()
