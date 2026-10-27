"""Fusion evaluation on ARKitScenes: single-frame vs naive-fused vs scale-aware fused boxes against ground truth.

Reads the automatic labels (``detections.jsonl`` and ``views.jsonl`` under
``<data root>/outputs/labels/arkitscenes/``): one world-frame box per view, detection and depth source
(laser, LiDAR, MoGe-2), each matched to a ground-truth object. Per depth source it fits the image-evidence
noise model from frame-to-frame scatter, without ground-truth geometry
(:func:`measured_space.fusion_eval.fit_noise_model`). Then, per video and depth source, it estimates every
matched object three ways and audits each estimate against the ground-truth box from
:func:`measured_space.datasets.arkitscenes.load_objects` (objects detected in at least ``--min-frames`` views).

Writes:

- ``<out dir>/records.jsonl``: every per-object and per-measurement record;
- ``<out dir>/summary.json``: the full summary. ``--out-dir`` defaults to ``<data root>/outputs/fusion``
  (``<data root>/outputs/fusion_exclude_ambiguous`` with ``--exclude-ambiguous``);
- ``<results dir>/summary.json``, ``<results dir>/summary.csv`` (one row per depth source and estimator) and
  ``<results dir>/log_scale.csv`` (one row per video and depth source): aggregates only. ``--results-dir`` is
  required, with no default, so a run never overwrites the committed tables by accident (the committed run
  used ``results/fusion``).

``--exclude-ambiguous`` (a pairing sensitivity analysis) drops every matched detection whose pairing with its
ground-truth object was ambiguous (its ``detections.jsonl`` match has ``ambiguous`` True: another feasible
pairing existed for the detection or the object) before the noise fit and the evaluation, since detections are
grouped into objects by their matched id. Detections of unknown ambiguity (match records written before the
2D association, or by the 3D one, which does not assess it: no ``runner_up_iou2d``) are kept. The summary's
``config.exclude_ambiguous`` records, per depth source, the detections dropped and the unknown ones kept.

Usage (from the repository root)::

    python scripts/run_fusion_eval.py --results-dir results/fusion [--scale-prior-sigma 0.1] [--gate-prob 0.997]
        [--min-frames 2] [--labels-dir <labels folder>/arkitscenes]
    python scripts/run_fusion_eval.py --results-dir <scratch folder> --exclude-ambiguous
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from measured_space.config import REPO_ROOT, data_root
from measured_space.datasets import arkitscenes as ark
from measured_space.estimation import box_to_state
from measured_space.fusion_eval import (
    TABLE_COLUMNS,
    Detection,
    NoiseModel,
    evaluate_capture,
    fit_noise_model,
    group_captures,
    load_detections,
    summarize,
    summary_table,
    write_records,
)

LOG_SCALE_COLUMNS = ("video", "source", "n_measurements", "n_evaluated", "log_scale", "log_scale_sd",
                     "gt_median_log_depth_ratio", "rejected_naive", "rejected_scale_aware", "converged")
"""Columns of ``log_scale.csv``."""


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Command-line options.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--labels-dir", type=Path, default=None,
                        help="folder with detections.jsonl and views.jsonl "
                             "(default <data root>/outputs/labels/arkitscenes)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="full outputs (default <data root>/outputs/fusion, or "
                             "<data root>/outputs/fusion_exclude_ambiguous with --exclude-ambiguous)")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="aggregate summaries; required (no default). The committed tables are in "
                             f"{REPO_ROOT / 'results' / 'fusion'}")
    parser.add_argument("--exclude-ambiguous", action="store_true",
                        help="drop matched detections whose pairing was ambiguous; detections of unknown "
                             "ambiguity are kept and counted")
    parser.add_argument("--scale-prior-sigma", type=float, default=0.1,
                        help="prior sd of each capture's log scale in scale-aware fusion")
    parser.add_argument("--gate-prob", type=float, default=0.997, help="chi-square(7) innovation gate")
    parser.add_argument("--min-frames", type=int, default=2, help="views an object needs to be evaluated")
    parser.add_argument("--sources", nargs="+", default=None, help="depth sources to evaluate (default all)")
    return parser.parse_args(argv)


def ground_truth(videos: Sequence[str]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, str]]]:
    """Ground-truth states and classes of every annotated object of each video.

    Parameters
    ----------
    videos : sequence of str
        ARKitScenes video ids.

    Returns
    -------
    truth : dict of str to dict of str to numpy.ndarray, shape (7,)
        ``video -> object id -> state`` (:func:`~measured_space.estimation.box_to_state` of the box).
    labels : dict of str to dict of str to str
        ``video -> object id -> class``.
    """
    truth, labels = {}, {}
    for video in videos:
        objects = ark.load_objects(video)
        truth[video] = {o.object_id: box_to_state(o.box) for o in objects}
        labels[video] = {o.object_id: o.label for o in objects}
    return truth, labels


DetectionKey = tuple[str, str, str, str]
"""``(video, source, frame, object_id)``: how :func:`measured_space.fusion_eval.load_detections` keys a detection."""


def pairing_ambiguity(labels_dir: Path) -> tuple[set[DetectionKey], set[DetectionKey]]:
    """Which matched detections were paired ambiguously, read from ``detections.jsonl``.

    The rows are filtered exactly as :func:`measured_space.fusion_eval.load_detections` filters them (status
    ``"matched"``, a box and a match record) and keyed the same way.

    Parameters
    ----------
    labels_dir : pathlib.Path
        Folder with ``detections.jsonl``.

    Returns
    -------
    ambiguous : set of tuple
        Keys of matched detections whose match has ``ambiguous`` True: another feasible pairing existed for
        the detection or its object (:attr:`measured_space.matching.Match.ambiguous`).
    unknown : set of tuple
        Keys of matched detections whose match has no ``runner_up_iou2d`` (records written before the 2D
        association, or by the 3D one, which does not assess ambiguity) and is not ambiguous.
    """
    ambiguous: set[DetectionKey] = set()
    unknown: set[DetectionKey] = set()
    with (labels_dir / "detections.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            status = row["status"] if row.get("box") is not None else "unboxed"
            match = row.get("match")
            if status != "matched" or match is None:
                continue
            key = (str(row["scene"]), str(row["source"]), str(row["frame"]), str(match["object_id"]))
            if match.get("runner_up_iou2d") is None or "ambiguous" not in match:
                unknown.add(key)
            elif match["ambiguous"]:
                ambiguous.add(key)
    return ambiguous, unknown - ambiguous


def drop_ambiguous(detections: Sequence[Detection], ambiguous: set[DetectionKey], unknown: set[DetectionKey]
                   ) -> tuple[list[Detection], dict[str, Any]]:
    """Drop the detections whose pairing was ambiguous, and count what was dropped and kept per depth source.

    Parameters
    ----------
    detections : sequence of Detection
        Matched detections (:func:`measured_space.fusion_eval.load_detections`).
    ambiguous : set of tuple
        Keys of ambiguously paired detections (:func:`pairing_ambiguity`). A key shared by several rows (a
        duplicate match, which :func:`measured_space.fusion_eval.group_captures` drops anyway) drops them all.
    unknown : set of tuple
        Keys of detections of unknown ambiguity; they are kept and counted.

    Returns
    -------
    kept : list of Detection
        The detections not dropped, in input order.
    record : dict
        ``enabled`` (True), ``unknown_policy`` (``"kept"``), totals ``n_dropped`` and ``n_unknown_kept``, and
        ``counts``: per depth source ``n_matched``, ``n_dropped`` and ``n_unknown_kept``.
    """
    kept: list[Detection] = []
    counts: dict[str, dict[str, int]] = {}
    for d in detections:
        key = (d.video, d.source, d.frame, d.object_id)
        entry = counts.setdefault(d.source, {"n_matched": 0, "n_dropped": 0, "n_unknown_kept": 0})
        entry["n_matched"] += 1
        if key in ambiguous:
            entry["n_dropped"] += 1
            continue
        entry["n_unknown_kept"] += key in unknown
        kept.append(d)
    record: dict[str, Any] = {"enabled": True, "unknown_policy": "kept",
                              "n_dropped": sum(c["n_dropped"] for c in counts.values()),
                              "n_unknown_kept": sum(c["n_unknown_kept"] for c in counts.values()),
                              "counts": dict(sorted(counts.items()))}
    return kept, record


def log_scale_rows(captures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per capture for ``log_scale.csv``.

    Parameters
    ----------
    captures : sequence of mapping
        ``summary["captures"]`` from :func:`measured_space.fusion_eval.summarize`.

    Returns
    -------
    list of dict of str to Any
        Rows keyed by :data:`LOG_SCALE_COLUMNS`.
    """
    return [{"video": c["video"], "source": c["source"], "n_measurements": c["n_measurements"],
             "n_evaluated": c["n_evaluated"], "log_scale": c["log_scale"], "log_scale_sd": c["log_scale_sd"],
             "gt_median_log_depth_ratio": c["gt_median_log_depth_ratio"],
             "rejected_naive": c["rejected"]["naive"], "rejected_scale_aware": c["rejected"]["scale_aware"],
             "converged": c["converged"]} for c in captures]


def source_differences(captures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-video differences of the fitted log scale between depth sources.

    Parameters
    ----------
    captures : sequence of mapping
        ``summary["captures"]``.

    Returns
    -------
    dict of str to Any
        For each pair ``"a-b"`` of sources present: the per-video differences ``log_scale_a - log_scale_b``
        (by video), and their median and mean.
    """
    by_video: dict[str, dict[str, float]] = {}
    for c in captures:
        by_video.setdefault(c["video"], {})[c["source"]] = c["log_scale"]
    sources = sorted({c["source"] for c in captures})
    out: dict[str, Any] = {}
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            diffs = {v: s[a] - s[b] for v, s in sorted(by_video.items()) if a in s and b in s}
            if diffs:
                values = np.array(list(diffs.values()))
                out[f"{a}-{b}"] = {"by_video": diffs, "median": float(np.median(values)),
                                   "mean": float(np.mean(values))}
    return out


def write_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], path: Path) -> None:
    """Write rows to a CSV file, floats with 6 significant digits.

    Parameters
    ----------
    rows : sequence of mapping
        Rows keyed by ``columns``.
    columns : sequence of str
        Column order.
    path : pathlib.Path
        Output file.
    """
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else ("" if v is None else v))
                             for k, v in row.items()})


def format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Render rows as a fixed-width text table.

    Parameters
    ----------
    rows : sequence of mapping
        Rows keyed by ``columns``.
    columns : sequence of str
        Columns to show, in order.

    Returns
    -------
    str
        The table, one line per row after a header line.
    """
    def cell(value: Any) -> str:
        """Format one value.

        Parameters
        ----------
        value : Any
            A number, string or None.

        Returns
        -------
        str
            Floats to 4 significant digits, None as "-".
        """
        if value is None:
            return "-"
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    cells = [[cell(row.get(c)) for c in columns] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]
    lines = ["  ".join(c.rjust(w) for c, w in zip(columns, widths))]
    lines += ["  ".join(v.rjust(w) for v, w in zip(r, widths)) for r in cells]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation and write its outputs.

    Parameters
    ----------
    argv : sequence of str or None, default None
        Command-line arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status (0 on success, 1 if the labels are missing).
    """
    args = parse_args(argv)
    start = time.perf_counter()
    labels_dir = args.labels_dir or data_root() / "outputs" / "labels" / "arkitscenes"
    out_dir = args.out_dir or data_root() / "outputs" / ("fusion_exclude_ambiguous" if args.exclude_ambiguous
                                                         else "fusion")
    if not (labels_dir / "detections.jsonl").exists():
        print(f"{labels_dir / 'detections.jsonl'} not found; run the labelling step first", file=sys.stderr)
        return 1
    detections, counts = load_detections(labels_dir)
    if args.sources:
        detections = [d for d in detections if d.source in args.sources]
    exclusion: dict[str, Any] = {"enabled": False}
    if args.exclude_ambiguous:
        detections, exclusion = drop_ambiguous(detections, *pairing_ambiguity(labels_dir))
        print(f"--exclude-ambiguous: dropped {exclusion['n_dropped']} detections, kept "
              f"{exclusion['n_unknown_kept']} of unknown ambiguity: {json.dumps(exclusion['counts'])}", flush=True)
    captures = group_captures(detections)
    truth, labels = ground_truth(sorted({video for video, _ in captures}))

    fits = {}
    for source in sorted({source for _, source in captures}):
        fits[source] = fit_noise_model([dets for (_, s), dets in captures.items() if s == source],
                                       initial=NoiseModel(), gate_prob=args.gate_prob, min_frames=args.min_frames)
    results = [evaluate_capture(dets, truth[video], fits[source].model, scale_prior_sigma=args.scale_prior_sigma,
                                gate_prob=args.gate_prob, min_frames=args.min_frames, labels=labels[video])
               for (video, source), dets in captures.items()]

    summary = summarize(results)
    summary["source_log_scale_differences"] = source_differences(summary["captures"])
    summary["noise_models"] = {
        source: {"constants": fit.model.to_dict(), "units": dict(NoiseModel.UNITS), "rounds": fit.rounds,
                 "converged": fit.converged, "n_measurements": fit.n_measurements, "n_objects": fit.n_objects,
                 "n_rejected": fit.n_rejected,
                 "fit": "REML on frame-to-frame scatter; no ground-truth geometry; objects grouped by matched id"}
        for source, fit in fits.items()}
    summary["config"] = {"scale_prior_sigma": args.scale_prior_sigma, "gate_prob": args.gate_prob,
                         "min_frames": args.min_frames, "labels_dir": labels_dir.name,
                         "detection_counts": counts, "exclude_ambiguous": exclusion}
    summary["runtime_s"] = round(time.perf_counter() - start, 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    n_records = write_records(results, out_dir / "records.jsonl")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    table = summary_table(summary)
    write_csv(table, TABLE_COLUMNS, args.results_dir / "summary.csv")
    write_csv(log_scale_rows(summary["captures"]), LOG_SCALE_COLUMNS, args.results_dir / "log_scale.csv")

    print(json.dumps({k: summary[k] for k in ("noise_models", "source_log_scale_differences",
                                              "paired_along_ray_difference_m", "config")},
                     indent=1))
    print(format_table(table, ("source", "estimator", "n", "nees_mean", "nees_median", "nees_lower", "nees_upper",
                               "frac_nees_above_chi2_95", "cov1_center", "cov1_size", "cov1_yaw", "cov1_all",
                               "cov2_all", "center_err_median_m", "size_err_median_m", "yaw_err_median_deg",
                               "rejected_fraction", "mean_p_gross")))
    print(format_table(log_scale_rows(summary["captures"]), LOG_SCALE_COLUMNS))
    print(f"{n_records} records -> {out_dir}; summaries -> {args.results_dir}; "
          f"{summary['runtime_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
