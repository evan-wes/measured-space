"""Oracle sanity check for ARKitScenes: do the ground-truth boxes, the ARKit poses and the laser depth agree?

The oracle answers spatial questions from the 3D boxes, the auto pipeline from images, and the laser depth is
the independent reference for both. That only works if the three share one frame. Per video this script
counts boxes and frames and checks the upright orientation of the views: the quarter turns chosen from the
poses against the dataset's ``sky_direction``, and the angle between projected world up and image up (-v) in each
upright view. For every object visible in a frame (judged against the laser depth), it measures:

- **explained**: the share of laser pixels whose ray passes through the box (+ margin) whose point lies
  inside it. Occluders in front lower it; a misplaced box or a wrong convention drives it towards 0.
- **distance**: distance from the laser points inside the box (+ margin) to the box surface, median and 90th
  percentile, pooled over points.
- **offset**: the world translation that best explains the signed point-to-face residuals of points within
  a window of their nearest face (least squares, point-to-plane). The laser depth was rendered with refined
  poses, so a systematic offset here would mean the laser frame and the box/pose frame disagree.
- **laser - LiDAR**: the laser depth minus the iPad LiDAR depth at high-confidence LiDAR pixels; the LiDAR is
  native to the ARKit poses, so this checks the laser rendering against them without any box.

With ``--conventions`` it also re-runs the "explained" measure under the rejected pose, camera-axis and box
rotation conventions and under time offsets, which is the evidence for the conventions the loader uses.

Default subsample: the main check uses every posed frame (148 over the 6 videos) and every 4th laser pixel in
each direction; the convention sweep uses 5 evenly spaced frames per video and every 8th pixel (the candidates
differ by large margins, so a small sample decides them).

Usage::

    python scripts/check_arkitscenes_oracle.py [--videos ID ...] [--stride 4] [--conventions]

Writes ``<data root>/outputs/oracle_sanity/arkitscenes.json`` (or ``--out``) and prints a table.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from measured_space.config import data_root
from measured_space.datasets import arkitscenes as ark
from measured_space.geometry import Array, OrientedBox, interpolate_poses, invert_pose
from measured_space.scene import FrameView
from measured_space.visibility import object_in_view

# ARKit's native camera axes (x right, y up, z backward) relative to OpenCV's (x right, y down, z forward).
_FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0])

POSE_CANDIDATES: dict[str, Callable[[Array], Array]] = {
    "inverted, OpenCV axes (used)": lambda T_wc: T_wc,
    "inverted, ARKit axes": lambda T_wc: T_wc @ _FLIP_YZ,
    "not inverted, OpenCV axes": lambda T_wc: invert_pose(T_wc),
    "not inverted, ARKit axes": lambda T_wc: invert_pose(T_wc) @ _FLIP_YZ,
}
"""Candidate camera-to-world poses, each built from the loader's ``T_wc``."""

BOX_CANDIDATES: dict[str, Callable[[OrientedBox], OrientedBox]] = {
    "axes = rows of normalizedAxes (used)": lambda box: box,
    "axes = columns of normalizedAxes": lambda box: OrientedBox(box.center, box.size, box.rotation.T),
}
"""Candidate box rotations, each built from the loader's box."""


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command-line options.

    Parameters
    ----------
    argv : list of str or None
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--videos", nargs="+", default=list(ark.VIDEO_IDS), help="video ids (default: all 6)")
    parser.add_argument("--stride", type=int, default=4,
                        help="laser pixel stride (each direction) for the main check (default 4)")
    parser.add_argument("--margin", type=float, default=0.05, help="box growth (m) for inside tests")
    parser.add_argument("--window", type=float, default=0.05,
                        help="max |point-to-face residual| (m) used for the offset estimate")
    parser.add_argument("--min-visible", type=float, default=0.1,
                        help="min visible fraction (against laser depth) for an object to count as visible")
    parser.add_argument("--conventions", action="store_true",
                        help="also compare rejected pose / axis / box conventions and time offsets")
    parser.add_argument("--offsets", type=float, nargs="+",
                        default=[-0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2],
                        help="time offsets (s) added to frame times for --conventions")
    parser.add_argument("--convention-frames", type=int, default=5,
                        help="evenly spaced frames per video for --conventions (default 5; 0 = all)")
    parser.add_argument("--convention-stride", type=int, default=8,
                        help="laser pixel stride (each direction) for --conventions (default 8)")
    parser.add_argument("--min-region", type=int, default=50,
                        help="min ray-region pixels (at --convention-stride) for a box to enter --conventions")
    parser.add_argument("--out", type=Path, default=None, help="JSON output path")
    return parser.parse_args(argv)


# -------------------------------------------------------------------------------------------- statistics


def quantiles(values: NDArray[np.floating], qs: tuple[float, ...] = (0.5, 0.9)) -> list[float | None]:
    """Quantiles of a possibly empty array.

    Parameters
    ----------
    values : numpy.ndarray, shape (N,), float
        Samples.
    qs : tuple of float, optional
        Quantile levels in [0, 1].

    Returns
    -------
    list of float or None
        One value per level; ``None`` for an empty array.
    """
    if len(values) == 0:
        return [None] * len(qs)
    return [float(q) for q in np.quantile(values, qs)]


def offset_from_residuals(normal_outer: Array, normal_residual: Array, count: int) -> dict[str, Any]:
    """Least-squares translation from point-to-plane residuals: solve ``(sum n n^T) d = sum n r``.

    Parameters
    ----------
    normal_outer : numpy.ndarray, shape (3, 3), float64
        Sum over points of ``n n^T`` (outward face normals, world frame).
    normal_residual : numpy.ndarray, shape (3,), float64
        Sum over points of ``n r`` (signed residuals, metres, positive outside the box).
    count : int
        Number of points summed.

    Returns
    -------
    dict
        ``offset_m``: world translation (x, y, z) in metres by which the laser surface sits outside the box
        faces, or ``None`` if under-determined; ``min_eig``: smallest eigenvalue of the mean ``n n^T``
        (near 0 means one direction is unconstrained); ``n_points``.
    """
    if count == 0:
        return {"offset_m": None, "min_eig": 0.0, "n_points": 0}
    min_eig = float(np.linalg.eigvalsh(normal_outer / count)[0])
    offset = np.linalg.solve(normal_outer, normal_residual).tolist() if min_eig > 1e-3 else None
    return {"offset_m": offset, "min_eig": min_eig, "n_points": int(count)}


def laser_minus_lidar(laser: NDArray[np.float32], lidar: NDArray[np.float32],
                      confidence: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Laser minus LiDAR depth at the high-confidence LiDAR pixels where both are valid.

    Parameters
    ----------
    laser : numpy.ndarray, shape (H, W), float32
        Laser depth, metres, at the RGB resolution.
    lidar : numpy.ndarray, shape (h, w), float32
        LiDAR depth, metres, on the same field of view.
    confidence : numpy.ndarray, shape (h, w), uint8
        LiDAR confidence (2 = high).

    Returns
    -------
    numpy.ndarray, shape (M,), float32
        Signed differences in metres; the laser is read at the pixel nearest each LiDAR pixel's centre.
    """
    (H, W), (h, w) = laser.shape, lidar.shape
    rows = np.minimum(np.round((np.arange(h) + 0.5) * H / h - 0.5).astype(int), H - 1)
    cols = np.minimum(np.round((np.arange(w) + 0.5) * W / w - 0.5).astype(int), W - 1)
    sampled = laser[np.ix_(rows, cols)]
    ok = (sampled > 0) & (lidar > 0) & (confidence == 2)
    return (sampled - lidar)[ok]


# ------------------------------------------------------------------------------------------------ checks


def orientation_summary(video_id: str, views: list[FrameView]) -> dict[str, Any]:
    """How upright a video's views are, and whether the turn chosen from the poses matches the dataset's metadata.

    Parameters
    ----------
    video_id : str
        ARKitScenes video id.
    views : list of FrameView
        The video's upright views from :func:`measured_space.datasets.arkitscenes.list_frames`.

    Returns
    -------
    dict
        ``sky_direction`` (metadata), ``quarter_turns`` from the poses and from the metadata, the median and
        maximum angle (degrees) between projected world up and image up over the views, the number of views
        above 45 degrees, and the number looking within 30 degrees of straight up or down (where the
        projected up direction is ill-defined).
    """
    angles = np.array([ark.up_angle_in_image(v.T_wc) for v in views])
    in_plane = np.array([np.linalg.norm((v.T_wc[:3, :3].T @ ark.WORLD_UP)[:2]) for v in views])
    return {
        "sky_direction": ark.SKY_DIRECTION.get(video_id),
        "quarter_turns": ark.upright_quarter_turns(video_id),
        "quarter_turns_from_metadata": (ark.quarter_turns_for_sky(ark.SKY_DIRECTION[video_id])
                                        if video_id in ark.SKY_DIRECTION else None),
        "up_angle_deg": dict(zip(("median", "max"), (float(np.median(angles)), float(angles.max())))),
        "frames_over_45deg": int(np.sum(angles > 45.0)),
        "frames_near_vertical": int(np.sum(in_plane < 0.5)),
    }



def check_video(video_id: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the sanity check on one video.

    Parameters
    ----------
    video_id : str
        ARKitScenes video id.
    args : argparse.Namespace
        Options from :func:`parse_args`.

    Returns
    -------
    summary : dict
        JSON-ready per-video results (counts, per-object stats, statistics).
    pooled : dict
        Raw material for the all-video statistics: ``distances`` and ``laser_lidar`` arrays (metres),
        ``nn``/``nr``/``count`` sums for :func:`offset_from_residuals`, and ``explained`` values.
    """
    objects = ark.load_objects(video_id)
    views = ark.list_frames(video_id)
    orientation = orientation_summary(video_id, views)
    distances, explained, laser_lidar = [], [], []
    nn, nr, count = np.zeros((3, 3)), np.zeros(3), 0
    per_object: dict[str, dict[str, Any]] = {o.object_id: {"label": o.label, "explained": [], "visible": []}
                                             for o in objects}
    for view in views:
        laser = ark.load_laser_depth(view)
        laser_lidar.append(laser_minus_lidar(laser, ark.load_lidar_depth(view), ark.load_confidence(view)))
        points, rays = ark.depth_rays(view, laser, args.stride)
        for obj in objects:
            seen, _ = object_in_view(view, obj, laser)
            if seen.visible_fraction < args.min_visible:
                continue
            fit = ark.laser_box_fit(view.camera_center, points, rays, obj.box, args.margin)
            if fit.n_region == 0:                     # visible, but no valid laser pixel at this stride
                continue
            distances.append(fit.surface_distance)
            explained.append(fit.explained)
            per_object[obj.object_id]["explained"].append(fit.explained)
            per_object[obj.object_id]["visible"].append(seen.visible_fraction)
            near = np.abs(fit.face_residual) <= args.window
            n, r = fit.face_normal[near], fit.face_residual[near]
            nn += n.T @ n
            nr += n.T @ r
            count += int(near.sum())
    distances_all = np.concatenate(distances) if distances else np.zeros(0)
    laser_lidar_all = np.concatenate(laser_lidar) if laser_lidar else np.zeros(0)
    explained_all = np.asarray(explained, dtype=np.float64)
    summary = {
        "boxes": len(objects),
        "wide_frames": len(ark.wide_timestamps(video_id)),
        "frames_with_pose": len(views),
        "orientation": orientation,
        "visible_pairs": len(explained),
        "visible_objects": sum(bool(o["explained"]) for o in per_object.values()),
        "explained_median": quantiles(explained_all, (0.5,))[0],
        "distance_m": dict(zip(("median", "p90"), quantiles(distances_all))),
        "n_points_in_boxes": int(len(distances_all)),
        "face_offset": offset_from_residuals(nn, nr, count),
        "laser_minus_lidar_m": {"median": quantiles(laser_lidar_all, (0.5,))[0],
                                "median_abs": quantiles(np.abs(laser_lidar_all), (0.5,))[0],
                                "n_pixels": int(len(laser_lidar_all))},
        "objects": {oid: {"label": o["label"], "frames_visible": len(o["explained"]),
                          "explained_median": quantiles(np.array(o["explained"]), (0.5,))[0],
                          "visible_fraction_median": quantiles(np.array(o["visible"]), (0.5,))[0]}
                    for oid, o in per_object.items()},
    }
    pooled = {"distances": distances_all, "laser_lidar": laser_lidar_all, "explained": explained_all,
              "nn": nn, "nr": nr, "count": count}
    return summary, pooled


def _explained_values(view: FrameView, laser: NDArray[np.float32], boxes: list[OrientedBox],
                      args: argparse.Namespace) -> list[float]:
    """The "explained" share of each box whose ray region is large enough, under the view's pose.

    Parameters
    ----------
    view : FrameView
        Frame with the candidate pose.
    laser : numpy.ndarray, shape (H, W), float32
        Laser depth, metres.
    boxes : list of OrientedBox
        Candidate boxes, world frame.
    args : argparse.Namespace
        Options (``convention_stride``, ``margin``, ``min_region``).

    Returns
    -------
    list of float
        One value per box with at least ``min_region`` region pixels.
    """
    points, rays = ark.depth_rays(view, laser, args.convention_stride)
    fits = (ark.laser_box_fit(view.camera_center, points, rays, box, args.margin) for box in boxes)
    return [fit.explained for fit in fits if fit.n_region >= args.min_region]


def compare_conventions(video_id: str, args: argparse.Namespace) -> dict[str, list[float]]:
    """The "explained" share of every box under each candidate convention and time offset.

    Unlike :func:`check_video`, boxes are not filtered by visibility (visibility itself depends on the
    convention); every box whose ray region has at least ``min_region`` pixels counts. Uses
    ``convention_frames`` evenly spaced frames and every ``convention_stride``-th laser pixel, in the sensor's
    own image orientation (the stored poses describe that camera; the upright turn is checked separately).

    Parameters
    ----------
    video_id : str
        ARKitScenes video id.
    args : argparse.Namespace
        Options from :func:`parse_args`.

    Returns
    -------
    dict
        Candidate name (``"pose | box"`` or ``"time offset +0.05 s"``) to its list of explained shares.
    """
    boxes = [obj.box for obj in ark.load_objects(video_id)]
    times, T_wc = ark.load_trajectory(video_id)
    values: dict[str, list[float]] = {}
    # Sensor orientation: the candidates are built from the stored trajectory, which describes that camera.
    views = ark.list_frames(video_id, upright=False)
    if 0 < args.convention_frames < len(views):
        views = [views[i] for i in np.linspace(0, len(views) - 1, args.convention_frames).round().astype(int)]
    for view in views:
        laser = ark.load_laser_depth(view)
        for pose_name, make_pose in POSE_CANDIDATES.items():
            candidate = dataclasses.replace(view, T_wc=make_pose(view.T_wc))
            for box_name, make_box in BOX_CANDIDATES.items():
                key = f"{pose_name} | {box_name}"
                values.setdefault(key, []).extend(
                    _explained_values(candidate, laser, [make_box(b) for b in boxes], args))
        shifted, valid = interpolate_poses(times, T_wc, float(view.frame) + np.asarray(args.offsets))
        for offset, pose, ok in zip(args.offsets, shifted, valid):
            if ok:
                key = f"time offset {offset:+.2f} s"
                values.setdefault(key, []).extend(
                    _explained_values(dataclasses.replace(view, T_wc=pose), laser, boxes, args))
    return values


# ------------------------------------------------------------------------------------------------ output


def _cm(value: float | None) -> str:
    """Metres as a centimetre string for the table.

    Parameters
    ----------
    value : float or None
        Metres.

    Returns
    -------
    str
        ``"12.3"`` or ``"-"``.
    """
    return "-" if value is None else f"{100 * value:.1f}"


def format_table(rows: dict[str, dict[str, Any]]) -> str:
    """The per-video summary as a fixed-width text table.

    Parameters
    ----------
    rows : dict
        Row name (video id or ``"all"``) to a summary from :func:`check_video` / :func:`pool`.

    Returns
    -------
    str
        The table, one line per row plus a header.
    """
    header = (f"{'video':9s} {'boxes':>5s} {'wide':>4s} {'posed':>5s} {'vis':>4s} {'objs':>4s} "
              f"{'expl':>5s} {'d50cm':>6s} {'d90cm':>6s} {'n_pts':>8s} {'off_x':>6s} {'off_y':>6s} {'off_z':>6s} "
              f"{'L-Li50':>6s} {'|L-Li|':>6s}")
    lines = [header]
    for name, s in rows.items():
        offset = s["face_offset"]["offset_m"] or [None] * 3
        explained = "-" if s["explained_median"] is None else f"{s['explained_median']:.2f}"
        lines.append(
            f"{name:9s} {s['boxes']:5d} {s['wide_frames']:4d} {s['frames_with_pose']:5d} "
            f"{s['visible_pairs']:4d} {s['visible_objects']:4d} "
            f"{explained:>5s} {_cm(s['distance_m']['median']):>6s} {_cm(s['distance_m']['p90']):>6s} "
            f"{s['n_points_in_boxes']:8d} {_cm(offset[0]):>6s} {_cm(offset[1]):>6s} {_cm(offset[2]):>6s} "
            f"{_cm(s['laser_minus_lidar_m']['median']):>6s} {_cm(s['laser_minus_lidar_m']['median_abs']):>6s}")
    return "\n".join(lines)


def pool(summaries: dict[str, dict[str, Any]], pooled: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """All-video statistics from the per-video summaries and raw material.

    Parameters
    ----------
    summaries : dict
        Video id to the summary from :func:`check_video`.
    pooled : dict
        Video id to the raw material from :func:`check_video`.

    Returns
    -------
    dict
        A summary with the same keys as a per-video one (no per-object section).
    """
    cat = {key: np.concatenate([p[key] for p in pooled.values()]) for key in ("distances", "laser_lidar", "explained")}
    total = {key: sum(s[key] for s in summaries.values())
             for key in ("boxes", "wide_frames", "frames_with_pose", "visible_pairs", "visible_objects")}
    return {
        **total,
        "explained_median": quantiles(cat["explained"], (0.5,))[0],
        "distance_m": dict(zip(("median", "p90"), quantiles(cat["distances"]))),
        "n_points_in_boxes": int(len(cat["distances"])),
        "face_offset": offset_from_residuals(sum(p["nn"] for p in pooled.values()),
                                             sum(p["nr"] for p in pooled.values()),
                                             sum(p["count"] for p in pooled.values())),
        "laser_minus_lidar_m": {"median": quantiles(cat["laser_lidar"], (0.5,))[0],
                                "median_abs": quantiles(np.abs(cat["laser_lidar"]), (0.5,))[0],
                                "n_pixels": int(len(cat["laser_lidar"]))},
    }


def format_orientation(summaries: dict[str, dict[str, Any]]) -> str:
    """The per-video orientation check as a text table.

    Parameters
    ----------
    summaries : dict
        Video id to the summary from :func:`check_video`.

    Returns
    -------
    str
        One line per video: metadata sky side, quarter turns from the poses and from the metadata, and the
        up-angle statistics of the upright views.
    """
    lines = [f"{'video':9s} {'sky (meta)':>10s} {'k poses':>7s} {'k meta':>6s} {'up deg med':>10s} "
             f"{'up deg max':>10s} {'>45deg':>6s} {'near-vert':>9s}"]
    for name, s in summaries.items():
        o = s["orientation"]
        lines.append(f"{name:9s} {str(o['sky_direction']):>10s} {o['quarter_turns']:7d} "
                     f"{str(o['quarter_turns_from_metadata']):>6s} {o['up_angle_deg']['median']:10.1f} "
                     f"{o['up_angle_deg']['max']:10.1f} {o['frames_over_45deg']:6d} {o['frames_near_vertical']:9d}")
    return "\n".join(lines)


def format_conventions(values: dict[str, list[float]]) -> str:
    """The convention comparison as a text table.

    Parameters
    ----------
    values : dict
        Candidate name to explained shares pooled over videos.

    Returns
    -------
    str
        One line per candidate: number of (frame, box) pairs, median and mean explained share.
    """
    lines = [f"{'candidate':72s} {'pairs':>5s} {'median':>6s} {'mean':>6s}"]
    for name, v in values.items():
        a = np.asarray(v)
        med, mean = (f"{np.median(a):.3f}", f"{a.mean():.3f}") if len(a) else ("-", "-")
        lines.append(f"{name:72s} {len(a):5d} {med:>6s} {mean:>6s}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the check over the selected videos, print the tables and write the JSON summary.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status (0).
    """
    args = parse_args(argv)
    started = time.time()
    summaries, pooled = {}, {}
    for video_id in args.videos:
        summaries[video_id], pooled[video_id] = check_video(video_id, args)
    rows = {**summaries, "all": pool(summaries, pooled)}
    print(format_orientation(summaries))
    print("k: numpy.rot90 quarter turns (counter-clockwise) to upright; up deg: angle between projected world up "
          "and image up in the upright views; near-vert: views within 30 deg of looking straight up/down\n")
    print(format_table(rows))
    print("expl: median share of each visible box's laser ray region inside the box (+margin); d50/d90: laser "
          "point to box surface (cm); off: point-to-face offset (cm); L-Li: laser minus LiDAR depth (cm)")
    result: dict[str, Any] = {"parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                              "videos": summaries, "all": rows["all"]}
    if args.conventions:
        values: dict[str, list[float]] = {}
        for video_id in args.videos:
            for key, v in compare_conventions(video_id, args).items():
                values.setdefault(key, []).extend(v)
        print()
        print(format_conventions(values))
        result["conventions"] = {k: {"pairs": len(v), "median": quantiles(np.asarray(v), (0.5,))[0],
                                     "mean": float(np.mean(v)) if v else None} for k, v in values.items()}
    result["seconds"] = round(time.time() - started, 1)
    out = args.out or data_root() / "outputs" / "oracle_sanity" / "arkitscenes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {out} ({result['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
