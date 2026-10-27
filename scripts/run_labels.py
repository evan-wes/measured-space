"""Produce paired oracle and automatic spatial labels for ARKitScenes and Smart Spaces views.

Every selected view goes through :func:`measured_space.pipeline.run_view` (SAM 3, MoGe-2, lifting, association,
questions, answers). Each finished view is written at once to its own shard,
``<labels>/<dataset>/shards/<view>.json``, so an interrupted run resumes by skipping views whose shard exists.
After the views are processed (or with ``--assemble-only``) the shards are assembled into, per dataset:

- ``views.jsonl``: one record per view (identity, camera, ``T_wc``, processing time per stage, unanswered specs);
- ``oracle_objects.jsonl``: one record per view and ground-truth object (box, visibility, truncation, bbox2d,
  per-source match);
- ``detections.jsonl``: one record per view, detection and depth source (phrase, score, mask statistics, world
  box, lift features, match);
- ``qa_oracle.jsonl`` and ``qa_auto_<depth>.jsonl``, written with :func:`measured_space.qa_schema.write_jsonl`;
  items of the same question share an ``id`` across files;
- ``summary.json``: counts per question type and source, match rates per class and depth source (with class
  confusions and 2D overlap under the 2D association), automatic coverage and agreement with the oracle,
  runtimes, and the GPU monitor's summary line if given.

``<labels>`` is ``<data_root>/outputs/labels`` (or a sub-folder of it, ``--out-subdir``, or any folder given with
``--out-dir``). A combined ``summary.json`` and a ``runs.jsonl`` log of invocations are written there too.

**Association.** ``--association 2d`` (default) pairs detections with ground-truth objects by 2D overlap,
class-agnostic, with no 3D gate (:func:`measured_space.matching.match_detections_2d`, ``--min-iou``);
``--association 3d`` uses the gated, label-compatible 3D centre distance kept for comparison.

**Re-association without models** (``--rematch-from <labels root>``). Reads every shard of the given label
folder, redoes the association with :func:`measured_space.pipeline.rematch_view` (CPU only; detections, lifted
boxes, features and the oracle's visible 2D boxes are all stored), recomputes the automatic answers, writes new
shards under ``--out-dir`` (which must differ from the source) and assembles them. The oracle answers are
recomputed as a check of the reconstruction and copied verbatim; after assembly each dataset's
``qa_oracle.jsonl`` must be byte-identical to the source's (SHA-256 compared, recorded in ``runs.jsonl``), else
the exit status is 1. Shards keep the source's GPU timings and add a ``rematch`` record.

Default sampling: all posed views of the six ARKitScenes laser videos (148); Smart Spaces ``Warehouse_018``
cameras 0000 and 0002 every 60th frame, i.e. every 2 s (300 views; oracle visibility and an automatic source from
the simulator depth); the nine depth-less scenes of ``configs/smartspaces_cameras.yaml`` with their two cameras
every 150th frame, i.e. every 5 s (1,080 views; oracle visibility from the ground truth's 2D boxes). Masks of at
least 50,000 pixels are lifted from every 3rd pixel in each direction, smaller ones at full resolution.

Run from the repository root (GPU)::

    python scripts/run_labels.py
    python scripts/run_labels.py --assemble-only --gpu-watch-summary "GPU_WATCH SUMMARY ..."

Re-associate stored labels (CPU, no models)::

    python scripts/run_labels.py --rematch-from <data_root>/outputs/labels --association 2d \\
        --out-dir <data_root>/outputs/labels_2d
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from measured_space.config import REPO_ROOT, data_root
from measured_space.datasets import arkitscenes, smartspaces
from measured_space.matching import Match2DConfig
from measured_space.pipeline import (
    ASSOCIATIONS,
    DEPTH_SOURCES,
    GT_BOX2D,
    PipelineConfig,
    rematch_view,
    run_view,
    scene_kind,
    to_json_line,
    view_key,
)
from measured_space.qa_schema import QAItem, is_correct, write_jsonl
from measured_space.scene import FrameView, SceneObject

DEFAULT_DEPTH_SCENES = ["MTMC_Tracking_2026/train/Warehouse_018"]
DEFAULT_DEPTH_CAMERAS = ["Camera_0000", "Camera_0002"]
DEFAULT_CAMERAS_CONFIG = REPO_ROOT / "configs" / "smartspaces_cameras.yaml"


@dataclass
class Job:
    """One view to process and what the pipeline needs for it.

    Attributes
    ----------
    view : FrameView
        The view.
    objects : list of SceneObject
        Ground-truth objects of the scene at the view's time.
    sources : tuple of str
        Depth sources for automatic boxes.
    visibility : str
        Oracle visibility method: a depth source or ``"gt_box2d"``.
    gt_boxes : dict of str to tuple of int or None
        Object id -> ground-truth visible 2D box in this camera (``gt_box2d`` only).
    """

    view: FrameView
    objects: list[SceneObject]
    sources: tuple[str, ...]
    visibility: str
    gt_boxes: dict[str, tuple[int, int, int, int]] | None = None


# --------------------------------------------------------------------------------------------------------
# Arguments and paths
# --------------------------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        The parsed options.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--datasets", nargs="+", choices=["arkitscenes", "smartspaces"],
                   default=["arkitscenes", "smartspaces"], help="datasets to process")
    p.add_argument("--ark-videos", nargs="+", default=list(arkitscenes.VIDEO_IDS), help="ARKitScenes video ids")
    p.add_argument("--ark-step", type=int, default=1, help="use every n-th posed ARKitScenes view")
    p.add_argument("--ss-depth-scenes", nargs="*", default=DEFAULT_DEPTH_SCENES,
                   help="Smart Spaces scenes with simulator depth")
    p.add_argument("--ss-depth-cameras", nargs="+", default=DEFAULT_DEPTH_CAMERAS,
                   help="cameras of the scenes with depth")
    p.add_argument("--ss-depth-step", type=int, default=60, help="frame step in the scenes with depth")
    p.add_argument("--ss-cameras-config", type=Path, default=DEFAULT_CAMERAS_CONFIG,
                   help="YAML of depth-less scenes and their two cameras")
    p.add_argument("--ss-step", type=int, default=150, help="frame step in the depth-less scenes")
    p.add_argument("--ss-start", type=int, default=0, help="first Smart Spaces frame")
    p.add_argument("--scenes", nargs="*", default=None,
                   help="only process scenes whose name contains one of these strings")
    p.add_argument("--max-views-per-scene", type=int, default=None, help="cap per scene (for trial runs)")
    p.add_argument("--lift-stride", type=int, default=3, help="pixel stride for lifting large masks")
    p.add_argument("--lift-stride-min-pixels", type=int, default=50000,
                   help="masks with at least this many pixels are lifted with --lift-stride")
    p.add_argument("--qa-seed", type=int, default=0, help="seed of the per-type question cap")
    p.add_argument("--association", choices=list(ASSOCIATIONS), default="2d",
                   help="2d: image overlap, class-agnostic (primary); 3d: gated centre distance (comparison)")
    p.add_argument("--min-iou", type=float, default=Match2DConfig.min_iou,
                   help="smallest 2D IoU of a pair under the 2D association")
    p.add_argument("--out-subdir", default="", help="write under <data_root>/outputs/labels/<this>")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="write to this folder instead of <data_root>/outputs/labels[/<out-subdir>]")
    p.add_argument("--rematch-from", type=Path, default=None,
                   help="re-associate the shards of this label folder without models (needs --out-dir)")
    p.add_argument("--assemble-only", action="store_true", help="only assemble existing shards")
    p.add_argument("--no-assemble", action="store_true", help="do not assemble after processing")
    p.add_argument("--gpu-watch-summary", default=None, help="GPU_WATCH SUMMARY line to store in summary.json")
    p.add_argument("--max-gpu-used-mib", type=int, default=4000,
                   help="refuse to start if another process already uses more GPU memory than this")
    return p.parse_args(argv)


def labels_root(subdir: str, out_dir: Path | None = None) -> Path:
    """The output folder for labels.

    Parameters
    ----------
    subdir : str
        Optional sub-folder below ``<data_root>/outputs/labels``; empty for none. Ignored with ``out_dir``.
    out_dir : pathlib.Path, optional
        An explicit output folder.

    Returns
    -------
    pathlib.Path
        The folder, resolved (created if needed).

    Raises
    ------
    ValueError
        If ``subdir`` would leave ``outputs/labels``.
    """
    if out_dir is not None:
        root = Path(out_dir).resolve()
    else:
        base = (data_root() / "outputs" / "labels").resolve()
        root = (base / subdir).resolve() if subdir else base
        if root != base and base not in root.parents:
            raise ValueError(f"--out-subdir {subdir!r} leaves {base}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def shard_path(root: Path, view: FrameView) -> Path:
    """Where a view's shard is written.

    Parameters
    ----------
    root : pathlib.Path
        The labels folder.
    view : FrameView
        The view.

    Returns
    -------
    pathlib.Path
        ``<root>/<dataset>/shards/<scene>__<camera>__<frame>.json`` with ``/`` in the scene replaced by ``__``.
    """
    name = "__".join([view.scene.replace("/", "__"), view.camera_id, view.frame])
    return root / view.dataset / "shards" / f"{name}.json"


# --------------------------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------------------------


def _wanted(scene: str, filters: Sequence[str] | None) -> bool:
    """Whether a scene passes the ``--scenes`` filter.

    Parameters
    ----------
    scene : str
        Scene name.
    filters : sequence of str or None
        Substrings; ``None`` or empty accepts every scene.

    Returns
    -------
    bool
        True if no filter is set or the scene contains one of the substrings.
    """
    return not filters or any(f in scene for f in filters)


def arkitscenes_jobs(args: argparse.Namespace) -> Iterator[Job]:
    """Jobs for the ARKitScenes laser videos.

    Parameters
    ----------
    args : argparse.Namespace
        Uses ``ark_videos``, ``ark_step``, ``scenes`` and ``max_views_per_scene``.

    Yields
    ------
    Job
        One per selected posed view, oracle visibility against the laser depth.
    """
    for video in args.ark_videos:
        if not _wanted(video, args.scenes):
            continue
        objects = arkitscenes.load_objects(video)
        views = arkitscenes.list_frames(video)[::args.ark_step][:args.max_views_per_scene]
        for view in views:
            yield Job(view, objects, DEPTH_SOURCES[scene_kind(view, True)], "laser")


def load_smartspaces_subset(scene: str, frames: Sequence[int]) -> tuple[dict[int, list[SceneObject]],
                                                                        dict[int, dict[str, dict[str, Any]]]]:
    """Ground truth of selected frames of a Smart Spaces scene, parsing only those frames.

    The files hold every one of 9,000 frames; parsing all of them takes minutes, the selected ones seconds.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder.
    frames : sequence of int
        Frame indices wanted.

    Returns
    -------
    tuple
        :func:`measured_space.datasets.smartspaces.parse_ground_truth` of the selected frames.
    """
    with (smartspaces.scene_dir(scene) / "ground_truth.json").open() as f:
        data = json.load(f)
    wanted = {int(x) for x in frames}
    return smartspaces.parse_ground_truth({k: v for k, v in data.items() if int(k) in wanted})


def video_frame_count(scene: str, camera_id: str) -> int:
    """Number of frames in a Smart Spaces camera's video.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder.
    camera_id : str
        Camera id.

    Returns
    -------
    int
        Frame count reported by the video container.

    Raises
    ------
    FileNotFoundError
        If the video cannot be opened.
    """
    path = smartspaces.scene_dir(scene) / "videos" / f"{camera_id}.mp4"
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def smartspaces_scene_jobs(scene: str, cameras: Sequence[str], step: int, has_depth: bool,
                           args: argparse.Namespace, root: Path) -> Iterator[Job]:
    """Jobs for one Smart Spaces scene, frame-major (all cameras of a frame together).

    Ground truth is read only if some selected view has no shard yet.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder.
    cameras : sequence of str
        Camera ids.
    step : int
        Frame step.
    has_depth : bool
        Whether the scene has simulator depth (oracle visibility from it) or not (ground-truth 2D boxes).
    args : argparse.Namespace
        Uses ``ss_start`` and ``max_views_per_scene``.
    root : pathlib.Path
        The labels folder (to check for existing shards).

    Yields
    ------
    Job
        One per selected view.
    """
    n_frames = min(video_frame_count(scene, c) for c in cameras)
    views = [v for f in range(args.ss_start, n_frames, step) for c in cameras
             for v in smartspaces.list_frames(scene, c, [f])][:args.max_views_per_scene]
    if all(shard_path(root, v).exists() for v in views):
        for v in views:
            yield Job(v, [], (), "")                      # skipped by the caller; no ground truth needed
        return
    objects, visible = load_smartspaces_subset(scene, sorted({int(v.frame) for v in views}))
    sources = DEPTH_SOURCES[scene_kind(views[0], has_depth)] if views else ()
    for v in views:
        frame = int(v.frame)
        if frame not in objects:
            raise KeyError(f"{scene} ground truth has no frame {frame}")
        if has_depth:
            yield Job(v, objects[frame], sources, "gt_depth")
        else:
            boxes = {oid: tuple(cams[v.camera_id]) for oid, cams in visible[frame].items() if v.camera_id in cams}
            yield Job(v, objects[frame], sources, GT_BOX2D, boxes)  # type: ignore[arg-type]


def smartspaces_jobs(args: argparse.Namespace, root: Path) -> Iterator[Job]:
    """Jobs for the Smart Spaces scenes with depth, then the depth-less ones.

    Parameters
    ----------
    args : argparse.Namespace
        Sampling options.
    root : pathlib.Path
        The labels folder.

    Yields
    ------
    Job
        One per selected view.
    """
    for scene in args.ss_depth_scenes:
        if _wanted(scene, args.scenes):
            yield from smartspaces_scene_jobs(scene, args.ss_depth_cameras, args.ss_depth_step, True, args, root)
    config = yaml.safe_load(Path(args.ss_cameras_config).read_text()) or {}
    for scene, entry in config.items():
        if _wanted(scene, args.scenes):
            yield from smartspaces_scene_jobs(scene, entry["cameras"], args.ss_step, False, args, root)


# --------------------------------------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------------------------------------


def gpu_memory_used_mib() -> int | None:
    """GPU memory in use on GPU 0 according to ``nvidia-smi``.

    Returns
    -------
    int or None
        MiB in use, or ``None`` if ``nvidia-smi`` is unavailable.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=30).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def write_shard(path: Path, payload: dict[str, Any]) -> None:
    """Write a shard through a temporary file and an atomic rename, so a shard is never half written.

    Parameters
    ----------
    path : pathlib.Path
        Final shard path.
    payload : dict
        The shard's content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(to_json_line(payload))
    os.replace(tmp, path)


def process(jobs: Iterator[Job], root: Path, config: PipelineConfig) -> dict[str, int]:
    """Run the pipeline on every job without a shard.

    Parameters
    ----------
    jobs : iterator of Job
        Views to process.
    root : pathlib.Path
        The labels folder.
    config : PipelineConfig
        Pipeline settings.

    Returns
    -------
    dict of str to int
        ``processed``, ``skipped`` and ``failed`` view counts.
    """
    counts = Counter()
    for job in jobs:
        path = shard_path(root, job.view)
        if path.exists():
            counts["skipped"] += 1
            continue
        started = datetime.now(timezone.utc).isoformat()
        t = time.perf_counter()
        try:
            result = run_view(job.view, job.objects, sources=job.sources, visibility=job.visibility,
                              gt_boxes=job.gt_boxes, config=config)
        except Exception as error:                          # noqa: BLE001 - one bad view must not stop the run
            counts["failed"] += 1
            with (root / job.view.dataset / "errors.jsonl").open("a") as f:
                f.write(to_json_line({"view_key": view_key(job.view), "error": f"{type(error).__name__}: {error}",
                                      "time": started}) + "\n")
            print(f"FAILED {view_key(job.view)}: {type(error).__name__}: {error}", flush=True)
            continue
        wall = time.perf_counter() - t
        write_shard(path, {"records": result.records(), "wall_seconds": wall, "started": started,
                           "finished": datetime.now(timezone.utc).isoformat()})
        counts["processed"] += 1
        stages = " ".join(f"{k}={v:.2f}" for k, v in result.timings.items())
        print(f"[{counts['processed']}] {result.key} det={len(result.detections)} specs={len(result.specs)} "
              f"wall={wall:.2f}s {stages}", flush=True)
    return dict(counts)


def rematch(source_root: Path, root: Path, datasets: Sequence[str], association: str,
            match_2d: Match2DConfig) -> dict[str, int]:
    """Re-associate every stored shard of a label folder and write the results as new shards (no models).

    Parameters
    ----------
    source_root : pathlib.Path
        Label folder whose ``<dataset>/shards/*.json`` are read (never written).
    root : pathlib.Path
        Output label folder; must differ from ``source_root``. Shards that already exist there are skipped.
    datasets : sequence of str
        Dataset folders to process (missing ones are skipped).
    association : str
        ``"2d"`` or ``"3d"`` (see :data:`measured_space.pipeline.ASSOCIATIONS`).
    match_2d : Match2DConfig
        Overlap threshold of the 2D association.

    Returns
    -------
    dict of str to int
        ``processed`` and ``skipped`` view counts.

    Raises
    ------
    ValueError
        If ``root`` is ``source_root``, or a stored view fails the reconstruction checks of
        :func:`measured_space.pipeline.rematch_view`.
    """
    if Path(source_root).resolve() == Path(root).resolve():
        raise ValueError(f"--out-dir must differ from --rematch-from ({root}); the source is never overwritten")
    counts: Counter = Counter()
    for dataset in datasets:
        paths = sorted((Path(source_root) / dataset / "shards").glob("*.json"))
        t0 = time.perf_counter()
        for path in paths:
            out = root / dataset / "shards" / path.name
            if out.exists():
                counts["skipped"] += 1
                continue
            stored = json.loads(path.read_text())
            t = time.perf_counter()
            result = rematch_view(stored["records"], association=association, match_2d=match_2d)
            records = result.records()
            seconds = time.perf_counter() - t
            write_shard(out, {"records": records, "wall_seconds": stored["wall_seconds"],
                              "started": stored["started"], "finished": stored["finished"],
                              "rematch": {"from": str(path), "association": association,
                                          "match_2d": {"min_iou": match_2d.min_iou}, "seconds": seconds,
                                          "time": datetime.now(timezone.utc).isoformat()}})
            counts["processed"] += 1
        print(f"{dataset}: re-associated {len(paths)} views ({association}) in {time.perf_counter() - t0:.1f} s",
              flush=True)
    return dict(counts)


def sha256(path: Path) -> str | None:
    """Checksum of a file.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    str or None
        Hex SHA-256 digest, or ``None`` if the file does not exist.
    """
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def oracle_identical(source_root: Path, root: Path, datasets: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Compare each dataset's assembled ``qa_oracle.jsonl`` with the source folder's.

    Parameters
    ----------
    source_root : pathlib.Path
        The label folder that was re-associated.
    root : pathlib.Path
        The output label folder.
    datasets : sequence of str
        Datasets to compare (those without an output file are skipped).

    Returns
    -------
    dict of str to dict
        Dataset -> ``{"source_sha256", "sha256", "bytes", "identical"}``.
    """
    out: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        new = root / dataset / "qa_oracle.jsonl"
        if not new.exists():
            continue
        a, b = sha256(Path(source_root) / dataset / "qa_oracle.jsonl"), sha256(new)
        out[dataset] = {"source_sha256": a, "sha256": b, "bytes": new.stat().st_size,
                        "identical": a is not None and a == b}
    return out


# --------------------------------------------------------------------------------------------------------
# Assembly and summary
# --------------------------------------------------------------------------------------------------------


def _frame_key(frame: str) -> tuple[float, str]:
    """Sort key for frame identifiers that are numbers written as text.

    Parameters
    ----------
    frame : str
        Frame identifier.

    Returns
    -------
    tuple
        ``(numeric value, text)``; non-numeric frames sort first by text.
    """
    try:
        return float(frame), frame
    except ValueError:
        return float("-inf"), frame


def load_shards(folder: Path) -> list[dict[str, Any]]:
    """Read a dataset's shards, ordered by scene, camera and frame.

    Parameters
    ----------
    folder : pathlib.Path
        ``<labels>/<dataset>``.

    Returns
    -------
    list of dict
        Shard payloads.
    """
    shards = [json.loads(p.read_text()) for p in sorted((folder / "shards").glob("*.json"))]
    return sorted(shards, key=lambda s: (s["records"]["view"]["scene"], s["records"]["view"]["camera_id"],
                                         _frame_key(s["records"]["view"]["frame"])))


def _write_lines(path: Path, records: Sequence[dict[str, Any]]) -> int:
    """Write records as JSON lines, replacing the file.

    Parameters
    ----------
    path : pathlib.Path
        Output file.
    records : sequence of dict
        Records.

    Returns
    -------
    int
        Number of lines written.
    """
    with path.open("w") as f:
        for record in records:
            f.write(to_json_line(record) + "\n")
    return len(records)


def _rate(numerator: int, denominator: int) -> float | None:
    """A ratio, or ``None`` for an empty denominator.

    Parameters
    ----------
    numerator, denominator : int
        Counts.

    Returns
    -------
    float or None
        ``numerator / denominator`` rounded to 4 decimals.
    """
    return round(numerator / denominator, 4) if denominator else None


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    """Count, mean, median and maximum.

    Parameters
    ----------
    values : sequence of float
        Values.

    Returns
    -------
    dict
        ``n``, ``mean``, ``median``, ``max`` (``None`` when empty), rounded to 3 decimals.
    """
    if not values:
        return {"n": 0, "mean": None, "median": None, "max": None}
    a = np.asarray(values, dtype=np.float64)
    return {"n": len(a), "mean": round(float(a.mean()), 3), "median": round(float(np.median(a)), 3),
            "max": round(float(a.max()), 3)}


def summarize(dataset: str, shards: Sequence[dict[str, Any]], gpu_watch: str | None) -> dict[str, Any]:
    """Counts, match rates, coverage, agreement and runtimes of one dataset's shards.

    Parameters
    ----------
    dataset : str
        Dataset name.
    shards : sequence of dict
        Shard payloads from :func:`load_shards`.
    gpu_watch : str or None
        The GPU monitor's summary line, stored as given.

    Returns
    -------
    dict
        The summary (see the module docstring).
    """
    views = [s["records"]["view"] for s in shards]
    oracle = [r for s in shards for r in s["records"]["oracle_objects"]]
    detections = [r for s in shards for r in s["records"]["detections"]]
    sources = sorted({src for v in views for src in v["depth_sources"]})

    # Oracle objects and detections.
    present = [r for r in oracle if r["present"]]
    by_class = Counter(r["label"] for r in present)
    phrase_of = {(r["view_key"], r["detection"]): r["phrase"] for r in detections}   # one per 2D detection

    # Matching, per depth source and class (recall) and per phrase (precision).
    matching: dict[str, Any] = {}
    for src in sources:
        group_views = {v["view_key"] for v in views if src in v["depth_sources"]}
        pres = [r for r in present if r["view_key"] in group_views]
        clear = [r for r in pres if r["clearly_visible"]]
        dets = [r for r in detections if r["source"] == src]
        lifted = [r for r in dets if r["box"] is not None]
        matched = [r for r in lifted if r["match"] is not None]
        per_class = {}
        for label in sorted({r["label"] for r in pres}):
            p = [r for r in pres if r["label"] == label]
            c = [r for r in p if r["clearly_visible"]]
            hits = [r["match"][src] for r in p if r["match"][src] is not None]
            per_class[label] = {
                "n_present": len(p), "n_matched": len(hits), "recall": _rate(len(hits), len(p)),
                "n_matched_label_correct": sum(bool(h.get("label_correct", True)) for h in hits),
                "n_clearly_visible": len(c),
                "recall_clearly_visible": _rate(sum(r["match"][src] is not None for r in c), len(c)),
                "median_distance_m": _stats([h["distance"] for h in hits])["median"],
                "median_iou": _stats([h["iou"] for h in hits])["median"]}
        per_phrase = {}
        for phrase in sorted({r["phrase"] for r in dets}):
            d = [r for r in dets if r["phrase"] == phrase]
            lf = [r for r in d if r["box"] is not None]
            correct = sum(r["match"] is not None and bool(r["match"].get("label_correct", True)) for r in lf)
            per_phrase[phrase] = {"n_detections": len(d), "n_lifted": len(lf),
                                  "n_matched": sum(r["match"] is not None for r in lf),
                                  "precision": _rate(sum(r["match"] is not None for r in lf), len(lf)),
                                  "n_matched_label_correct": correct,
                                  "precision_label_correct": _rate(correct, len(lf))}
        per_visibility = {}
        for method in sorted({v["visibility"] for v in views if src in v["depth_sources"]}):
            mv = {v["view_key"] for v in views if v["visibility"] == method and src in v["depth_sources"]}
            p = [r for r in pres if r["view_key"] in mv]
            c = [r for r in p if r["clearly_visible"]]
            lf = [r for r in lifted if r["view_key"] in mv]
            per_visibility[method] = {
                "n_views": len(mv), "recall": _rate(sum(r["match"][src] is not None for r in p), len(p)),
                "recall_clearly_visible": _rate(sum(r["match"][src] is not None for r in c), len(c)),
                "precision": _rate(sum(r["match"] is not None for r in lf), len(lf))}
        n_correct = sum(bool(r["match"].get("label_correct", True)) for r in matched)
        confusions = Counter(f"{r['phrase']} -> {r['match']['label']}" for r in matched
                             if not r["match"].get("label_correct", True))
        matching[src] = {
            "n_present": len(pres), "n_clearly_visible": len(clear), "n_detections": len(dets),
            "n_lifted": len(lifted), "n_lift_failed": len(dets) - len(lifted), "n_matched": len(matched),
            "recall": _rate(len(matched), len(pres)),
            "recall_clearly_visible": _rate(sum(r["match"][src] is not None for r in clear), len(clear)),
            "precision": _rate(len(matched), len(lifted)),
            "n_matched_label_correct": n_correct, "confusion_rate": _rate(len(matched) - n_correct, len(matched)),
            "confusions": dict(confusions.most_common()),
            "n_ambiguous": sum(bool(r["match"].get("ambiguous")) for r in matched),
            "median_distance_m": _stats([r["match"]["distance"] for r in matched])["median"],
            "median_iou": _stats([r["match"]["iou"] for r in matched])["median"],
            "median_iou2d": _stats([r["match"]["iou2d"] for r in matched
                                    if r["match"].get("iou2d") is not None])["median"],
            "by_class": per_class, "by_phrase": per_phrase, "by_visibility": per_visibility}

    # Questions: specs, items, coverage and agreement with the oracle.
    qa_sources = sorted({src for s in shards for src in s["records"]["qa"]})
    oracle_items = {i["id"]: i for s in shards for i in s["records"]["qa"].get("oracle", [])}
    n_specs = Counter(i["qtype"] for i in oracle_items.values())
    n_items, coverage, agreement = {}, {}, {}
    for src in qa_sources:
        items = [i for s in shards for i in s["records"]["qa"].get(src, [])]
        group_specs = Counter(i["qtype"] for s in shards for i in s["records"]["qa"].get("oracle", [])
                              if src == "oracle" or src.removeprefix("auto_") in s["records"]["view"]["depth_sources"])
        counts = Counter(i["qtype"] for i in items)
        n_items[src] = dict(sorted(counts.items()))
        coverage[src] = {q: _rate(counts[q], group_specs[q]) for q in sorted(group_specs)}
        coverage[src]["all"] = _rate(sum(counts.values()), sum(group_specs.values()))
        if src != "oracle":
            correct: Counter = Counter()
            for i in items:
                item, ref = QAItem.from_json(i), oracle_items[i["id"]]
                correct[i["qtype"]] += bool(is_correct(item.spec, item.answer, ref["answer"]))
            agreement[src] = {q: _rate(correct[q], counts[q]) for q in sorted(counts)}
            agreement[src]["all"] = _rate(sum(correct.values()), sum(counts.values()))

    starts = [s["started"] for s in shards]
    ends = [s["finished"] for s in shards]
    stage_names = sorted({k for v in views for k in v["timings"]})
    rematched = [s["rematch"] for s in shards if "rematch" in s]
    return {
        "dataset": dataset,
        "association": views[0]["config"].get("association", "3d") if views else None,
        "rematch": None if not rematched else {
            "n_views": len(rematched), "from_folders": sorted({str(Path(r["from"]).parents[2]) for r in rematched}),
            "seconds_total": round(sum(r["seconds"] for r in rematched), 1),
            "note": "views re-associated from stored shards without models; runtime below is the source run's"},
        "n_views": len(views),
        "n_views_by_scene": dict(sorted(Counter(v["scene"] for v in views).items())),
        "n_views_by_visibility": dict(sorted(Counter(v["visibility"] for v in views).items())),
        "oracle": {"n_records": len(oracle), "n_present": len(present),
                   "n_clearly_visible": sum(r["clearly_visible"] for r in present),
                   "present_by_class": dict(sorted(by_class.items())),
                   "clearly_visible_by_class": dict(sorted(Counter(r["label"] for r in present
                                                                   if r["clearly_visible"]).items()))},
        "detections": {"n_2d": len(phrase_of), "by_phrase": dict(sorted(Counter(phrase_of.values()).items()))},
        "matching": matching,
        "qa": {"n_specs": dict(sorted(n_specs.items())), "n_specs_total": sum(n_specs.values()),
               "n_items": n_items, "coverage": coverage, "agreement_with_oracle": agreement},
        "runtime": {"sum_view_wall_seconds": round(sum(s["wall_seconds"] for s in shards), 1),
                    "first_view_started": min(starts) if starts else None,
                    "last_view_finished": max(ends) if ends else None,
                    "view_wall_seconds_by_visibility": {
                        m: _stats([s["wall_seconds"] for s in shards if s["records"]["view"]["visibility"] == m])
                        for m in sorted({v["visibility"] for v in views})},
                    "stage_seconds": {k: _stats([v["timings"][k] for v in views if k in v["timings"]])
                                      for k in stage_names}},
        "config": views[0]["config"] if views else None,
        "gpu_watch": gpu_watch,
    }


def assemble(root: Path, datasets: Sequence[str], gpu_watch: str | None) -> dict[str, Any]:
    """Assemble each dataset's shards into the output files and write the summaries.

    Parameters
    ----------
    root : pathlib.Path
        The labels folder.
    datasets : sequence of str
        Dataset folders to assemble (missing ones are skipped).
    gpu_watch : str or None
        The GPU monitor's summary line.

    Returns
    -------
    dict
        The combined summary, by dataset.

    Raises
    ------
    ValueError
        If a dataset's shards were produced with different pipeline settings.
    """
    combined: dict[str, Any] = {}
    for dataset in datasets:
        folder = root / dataset
        shards = load_shards(folder) if (folder / "shards").is_dir() else []
        if not shards:
            continue
        configs = {json.dumps(s["records"]["view"]["config"], sort_keys=True) for s in shards}
        if len(configs) > 1:
            raise ValueError(f"{folder}: shards were made with {len(configs)} different settings")
        _write_lines(folder / "views.jsonl", [s["records"]["view"] for s in shards])
        _write_lines(folder / "oracle_objects.jsonl", [r for s in shards for r in s["records"]["oracle_objects"]])
        _write_lines(folder / "detections.jsonl", [r for s in shards for r in s["records"]["detections"]])
        sources = sorted({src for s in shards for src in s["records"]["qa"]})
        for src in sources:
            write_jsonl((QAItem.from_json(i) for s in shards for i in s["records"]["qa"].get(src, [])),
                        folder / f"qa_{src}.jsonl")
        summary = summarize(dataset, shards, gpu_watch)
        (folder / "summary.json").write_text(json.dumps(summary, indent=1))
        combined[dataset] = summary
    (root / "summary.json").write_text(json.dumps(combined, indent=1))
    return combined


# --------------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Process the selected views, then assemble the outputs.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Exit status: 0 on success, 2 if the GPU is busy, 1 if a view failed or a re-associated dataset's oracle
        file differs from its source.

    Raises
    ------
    SystemExit
        If ``--rematch-from`` is given without ``--out-dir`` (from the argument parser).
    """
    args = parse_args(argv)
    if args.rematch_from is not None and args.out_dir is None:
        raise SystemExit("--rematch-from needs --out-dir (a new folder; the source is never written)")
    root = labels_root(args.out_subdir, args.out_dir)
    match_2d = Match2DConfig(min_iou=args.min_iou)
    config = PipelineConfig(qa_seed=args.qa_seed, lift_stride=args.lift_stride,
                            lift_stride_min_pixels=args.lift_stride_min_pixels, association=args.association,
                            match_2d=match_2d)
    run: dict[str, Any] = {"argv": list(sys.argv[1:] if argv is None else argv),
                           "started": datetime.now(timezone.utc).isoformat()}
    counts: dict[str, int] = {}
    t0 = time.perf_counter()
    if args.rematch_from is not None:
        run["rematch_from"] = str(Path(args.rematch_from).resolve())
        counts = rematch(args.rematch_from, root, args.datasets, args.association, match_2d)
    elif not args.assemble_only:
        used = gpu_memory_used_mib()
        if used is not None and used > args.max_gpu_used_mib:
            print(f"GPU busy: {used} MiB in use (limit {args.max_gpu_used_mib}); not starting", flush=True)
            return 2
        import torch

        from measured_space.perception import default_moge, default_segmenter

        t = time.perf_counter()
        default_segmenter()
        default_moge()
        run["model_load_seconds"] = round(time.perf_counter() - t, 2)
        print(f"models loaded in {run['model_load_seconds']} s", flush=True)
        for dataset in args.datasets:
            jobs = arkitscenes_jobs(args) if dataset == "arkitscenes" else smartspaces_jobs(args, root)
            for key, value in process(jobs, root, config).items():
                counts[key] = counts.get(key, 0) + value
        if torch.cuda.is_available():
            run["cuda_max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    run["processing_seconds"] = round(time.perf_counter() - t0, 1)
    run["counts"] = counts
    if not args.no_assemble:
        t = time.perf_counter()
        assemble(root, args.datasets, args.gpu_watch_summary)
        run["assemble_seconds"] = round(time.perf_counter() - t, 1)
    failed = bool(counts.get("failed"))
    if args.rematch_from is not None and not args.no_assemble:
        run["oracle_identical"] = oracle_identical(args.rematch_from, root, args.datasets)
        failed |= not all(v["identical"] for v in run["oracle_identical"].values())
    run["finished"] = datetime.now(timezone.utc).isoformat()
    with (root / "runs.jsonl").open("a") as f:
        f.write(json.dumps(run) + "\n")
    print(json.dumps(run), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
