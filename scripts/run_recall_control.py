"""Certified recall for an automatic 3D labeller: leave-one-scene-out conformal risk control of its miss rate.

For each dataset of ``--datasets``, every view of the label folder ``--labels`` (the shards of
``scripts/run_labels.py``, read only) is paired with the labeller's boxes in ``--predictions``
(``<predictions>/<dataset>/<scene>.jsonl``, one line per view). A line is either the CuTR format of
:mod:`measured_space.cubify`, checked against its view (key, image size and pose:
:func:`measured_space.box_sources.check_predictions`), or any line with ``view_key`` and ``boxes[].score`` plus a
2D box (``boxes[].box2d``) or a world box (``center``, ``size``, ``rotation``; use ``--box2d projected``). Then
:func:`measured_space.recall_control.run_recall_control` calibrates the score threshold by conformal risk control
(Angelopoulos et al., ICLR 2024) for every ``--alphas`` value, leaving one scene out at a time, and measures the
cost and the conditional miss rates (see that module's docstring for the method and its guarantee).

The observation floor (the lowest threshold that can be evaluated) is the prediction files' ``score_threshold``
(the largest, if they differ), or ``--floor`` when the files lack one; ``--floor`` above the files' threshold drops
the boxes below it. Calibration runs per dataset: scenes of different datasets are not exchangeable.

**Box cap.** CuTR writes at most ``--max-boxes`` (100) boxes per view. A view with that many is truncated below its
lowest written score (its cap floor): a lower threshold would have kept boxes that were never written. ``t_cap``
is the largest cap floor. With ``--truncation floor`` (default) CRC may only choose thresholds above ``t_cap``, where
no view is truncated. With ``--truncation mark`` it may choose any threshold from the observation floor: the
guarantee then holds for the capped labeller as run (top 100, then the threshold), and every choice at or below
``t_cap`` is flagged with the share of views truncated at it. Curves always span the observation floor to 1, with
the truncated region marked (see :mod:`measured_space.recall_control`, step 11).

Outputs:

- per-item tables in ``<out-root>/<name>/`` (``--out-root`` defaults to ``<data root>/outputs/recall_control``):
  ``objects.csv`` (one row per present object per view: coverage score, range, visible fraction, class),
  ``boxes.csv`` (one row per usable box), ``views.csv`` and ``run.json``;
- aggregates in ``--results`` (required, no default): ``by_alpha.csv`` (headline per dataset and alpha),
  ``loso.csv`` (per dataset, alpha and held-out scene), ``conditional.csv`` (held-out miss rate by range, visible
  fraction and class), ``curve.csv`` and ``curve_by_scene.csv`` (FNR and cost against the threshold),
  ``mondrian.csv`` (with ``--mondrian``), ``summary.json`` and the figure ``fnr_vs_threshold_<dataset>.png`` /
  ``.pdf``.

Safety: both output folders must not exist or must be empty, must not overlap each other or an input folder, and
nothing is written otherwise. Every view needs a prediction line, unless ``--skip-missing`` (skipped views are
counted).

Usage (from the repository root)::

    python scripts/run_recall_control.py --labels <data>/outputs/labels_ark106_2d \\
        --predictions <data>/outputs/cubify/predictions_t005 --name ark106_t005 \\
        --results results/recall_control/ark106_t005
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from measured_space.audit import dumps  # noqa: E402
from measured_space.box_sources import check_predictions, parse_view_predictions, predictions_file  # noqa: E402
from measured_space.config import REPO_ROOT, data_root  # noqa: E402
from measured_space.recall_control import (  # noqa: E402
    ALPHAS,
    POPULATIONS,
    PUBLISHED_THRESHOLD,
    STRATA,
    RecallConfig,
    RecallResult,
    ScoredBox,
    ViewEvidence,
    boxes_from_cutr,
    boxes_from_record,
    cap_floor,
    cap_free_floor,
    run_recall_control,
    truncation_threshold,
    view_evidence,
)

PER_ITEM_TABLES = ("objects", "boxes", "views")
"""tuple of str: tables written to the per-item folder."""

AGGREGATE_TABLES = ("by_alpha", "loso", "conditional", "curve", "curve_by_scene", "mondrian")
"""tuple of str: tables written to ``--results`` (``mondrian`` only with ``--mondrian``)."""

CUTR_MAX_BOXES = 100
"""int: CuTR writes at most this many boxes per view (its top-k)."""

CUTR_BOX_KEYS = ("box2d", "center_cam")
"""tuple of str: box keys that mark a CuTR line (checked against its view's pose)."""

REFERENCE = ("Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., Schuster, T. Conformal Risk Control. ICLR 2024. "
             "arXiv:2208.02814")
"""str: the method's reference."""

WIDTH = 3.4                                                   # inches: one column of a two-column paper
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # the palette of scripts/make_figures.py
INK, MUTED, GRID, BASELINE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


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
    p.add_argument("--labels", type=Path, required=True,
                   help="label folder of run_labels.py whose <dataset>/shards/*.json are read (never written)")
    p.add_argument("--predictions", type=Path, required=True,
                   help="predictions folder holding <dataset>/<scene>.jsonl")
    p.add_argument("--datasets", nargs="+", default=["arkitscenes"], help="datasets, each calibrated on its own")
    p.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS), help="target expected miss rates")
    p.add_argument("--population", choices=list(POPULATIONS), default="present",
                   help="ground-truth objects that count (present: the pipeline's presence; clearly_visible)")
    p.add_argument("--name", required=True, help="run name: per-item tables go to <out-root>/<name>/")
    p.add_argument("--results", type=Path, required=True, help="aggregate output folder (new or empty; no default)")
    p.add_argument("--out-root", type=Path, default=None,
                   help="per-item root (default <data root>/outputs/recall_control)")
    p.add_argument("--box2d", choices=["predicted", "projected"], default="predicted",
                   help="2D box of a prediction: its own box2d, or its world box projected with the view's camera")
    p.add_argument("--min-iou", type=float, default=RecallConfig.min_iou,
                   help="a box covers an object at 2D IoU at least this")
    p.add_argument("--floor", type=float, default=None,
                   help="lowest threshold evaluated (default: the prediction files' score_threshold)")
    p.add_argument("--published-threshold", type=float, default=PUBLISHED_THRESHOLD,
                   help="the cost baseline's threshold")
    p.add_argument("--max-boxes", type=int, default=CUTR_MAX_BOXES,
                   help="the labeller's cap on boxes per view (CuTR: 100); 0 for none")
    p.add_argument("--truncation", choices=["floor", "mark"], default="floor",
                   help="floor: CRC chooses only above the box cap's truncated region; mark: anywhere, flagged")
    p.add_argument("--mondrian", nargs="*", choices=list(STRATA), default=[],
                   help="also run per-stratum CRC on these strata")
    p.add_argument("--skip-missing", action="store_true", help="skip views without a prediction line")
    p.add_argument("--n-boot", type=int, default=RecallConfig.n_boot, help="bootstrap replicates over scenes")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------------------------
# Safety and provenance
# --------------------------------------------------------------------------------------------------------


def refuse_existing(outputs: Sequence[Path], inputs: Sequence[Path]) -> None:
    """Refuse output folders that hold anything, overlap each other, or overlap an input folder.

    Parameters
    ----------
    outputs : sequence of pathlib.Path
        Output folders.
    inputs : sequence of pathlib.Path
        Input folders.

    Raises
    ------
    FileExistsError
        If an output folder exists and is not an empty folder.
    ValueError
        If two output folders, or an output and an input folder, are the same or nested.
    """
    resolved = [Path(p).resolve() for p in outputs]
    for out in resolved:
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise FileExistsError(f"{out} exists and is not an empty folder; choose a new one")

    def overlap(a: Path, b: Path) -> bool:
        """Whether two folders are the same or one lies inside the other.

        Parameters
        ----------
        a, b : pathlib.Path
            Resolved folders.

        Returns
        -------
        bool
            True if they overlap.
        """
        return a == b or a in b.parents or b in a.parents

    for i, out in enumerate(resolved):
        for other in resolved[i + 1:]:
            if overlap(out, other):
                raise ValueError(f"output folders {out} and {other} overlap")
        for folder in (Path(p).resolve() for p in inputs):
            if overlap(out, folder):
                raise ValueError(f"output folder {out} overlaps the input folder {folder}")


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
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    """The repository's current commit, if git is available (read-only).

    Returns
    -------
    str or None
        The commit hash, with ``+dirty`` if the working tree has changes, or None.
    """
    try:
        head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------------------


def read_prediction_lines(path: Path) -> dict[str, dict[str, Any]]:
    """Read one scene's prediction lines, keyed by view.

    Parameters
    ----------
    path : pathlib.Path
        A ``<scene>.jsonl`` file.

    Returns
    -------
    dict of str to dict
        View key -> parsed line.

    Raises
    ------
    ValueError
        If a view appears twice.
    """
    out: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record["view_key"])
            if key in out:
                raise ValueError(f"{path}: view {key} appears twice")
            out[key] = record
    return out


def view_boxes(view_record: Mapping[str, Any], line: Mapping[str, Any]) -> list[ScoredBox]:
    """A view's scored boxes from its prediction line, checking the line against the view.

    Parameters
    ----------
    view_record : mapping
        The shard's ``"view"`` record.
    line : mapping
        The view's prediction line.

    Returns
    -------
    list of ScoredBox
        The boxes. CuTR lines (every box has ``box2d`` and ``center_cam``) are parsed and checked by
        :func:`~measured_space.box_sources.check_predictions`; other lines by
        :func:`~measured_space.recall_control.boxes_from_record`, with the image size checked if given.

    Raises
    ------
    ValueError
        If the line belongs to another view or image size, or a CuTR box disagrees with the view's pose.
    """
    key = view_record["view_key"]
    if str(line["view_key"]) != key:
        raise ValueError(f"prediction line for {line['view_key']} given for view {key}")
    if line["boxes"] and all(all(k in b for k in CUTR_BOX_KEYS) for b in line["boxes"]):
        predictions = parse_view_predictions(line)
        check_predictions(view_record, predictions)
        return boxes_from_cutr(predictions)
    camera = view_record["camera"]
    if line.get("image_size") is not None and tuple(int(v) for v in line["image_size"]) != (
            int(camera["width"]), int(camera["height"])):
        raise ValueError(f"{key}: predictions are for a {line['image_size']} image")
    return boxes_from_record(line)


def file_floor(lines: Sequence[Mapping[str, Any]]) -> tuple[float | None, list[float]]:
    """The score threshold the prediction lines were written with.

    Parameters
    ----------
    lines : sequence of mapping
        Prediction lines.

    Returns
    -------
    floor : float or None
        The largest ``score_threshold`` among the lines, or ``None`` if no line states one.
    distinct : list of float
        Every distinct stated threshold.
    """
    values = sorted({float(r["score_threshold"]) for r in lines if r.get("score_threshold") is not None})
    return (values[-1] if values else None), values


def load_dataset(labels: Path, predictions: Path, dataset: str, *, skip_missing: bool
                 ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Read a dataset's shards and the prediction lines of their views.

    Parameters
    ----------
    labels : pathlib.Path
        The label folder.
    predictions : pathlib.Path
        The predictions folder.
    dataset : str
        Dataset name.
    skip_missing : bool
        Drop views without a prediction line instead of failing.

    Returns
    -------
    records : list of dict
        Each kept view's shard records (``view``, ``oracle_objects``), sorted by view key.
    lines : dict of str to dict
        View key -> prediction line, for the kept views.
    provenance : dict
        ``n_shards``, ``shards_sha256`` (one digest over every shard's name and content), ``prediction_files``
        (scene -> file name and SHA-256) and ``n_views_skipped``.

    Raises
    ------
    FileNotFoundError
        If the dataset has no shards, or a view has no prediction line and ``skip_missing`` is off.
    """
    shard_paths = sorted((labels / dataset / "shards").glob("*.json"))
    if not shard_paths:
        raise FileNotFoundError(f"no shards in {labels / dataset / 'shards'}")
    digest = hashlib.sha256()
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for path in shard_paths:
        payload = path.read_bytes()
        digest.update(path.name.encode() + b"\0" + hashlib.sha256(payload).digest())
        records = json.loads(payload)["records"]
        by_scene.setdefault(str(records["view"]["scene"]), []).append(
            {"view": records["view"], "oracle_objects": records["oracle_objects"]})
    kept: list[dict[str, Any]] = []
    lines: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, str]] = {}
    skipped = 0
    for scene in sorted(by_scene):
        path = predictions_file(predictions, dataset, scene)
        scene_lines = read_prediction_lines(path) if path is not None else {}
        if path is not None:
            files[scene] = {"file": path.name, "sha256": sha256(path)}
        for records in by_scene[scene]:
            key = records["view"]["view_key"]
            if key not in scene_lines:
                if not skip_missing:
                    raise FileNotFoundError(f"no prediction line for {key} (use --skip-missing to skip such views)")
                skipped += 1
                continue
            kept.append(records)
            lines[key] = scene_lines[key]
    kept.sort(key=lambda r: r["view"]["view_key"])
    return kept, lines, {"n_shards": len(shard_paths), "shards_sha256": digest.hexdigest(),
                         "prediction_files": files, "n_views_skipped": skipped}


def build_views(records: Sequence[Mapping[str, Any]], lines: Mapping[str, Mapping[str, Any]], floor: float, *,
                project_3d: bool, max_boxes: int | None) -> tuple[list[ViewEvidence], int]:
    """Pair every view's present objects with its boxes at or above the floor.

    Parameters
    ----------
    records : sequence of mapping
        Shard records of the views.
    lines : mapping of str to mapping
        View key -> prediction line.
    floor : float
        Boxes scoring below this are dropped.
    project_3d : bool
        Use projected world boxes as the 2D boxes.
    max_boxes : int or None
        The labeller's cap on boxes per view; a view's cap floor comes from every box written for it.

    Returns
    -------
    views : list of ViewEvidence
        One per view.
    n_below_floor : int
        Boxes dropped for scoring below the floor.
    """
    views, dropped = [], 0
    for r in records:
        boxes = view_boxes(r["view"], lines[r["view"]["view_key"]])
        kept = [b for b in boxes if b.score >= floor]
        dropped += len(boxes) - len(kept)
        views.append(view_evidence(r, kept, project_3d=project_3d,
                                   cap_floor=cap_floor([b.score for b in boxes], max_boxes)))
    return views, dropped


# --------------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------------


def style() -> None:
    """Set matplotlib defaults for small, clean paper figures (as ``scripts/make_figures.py``)."""
    plt.rcParams.update({
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6, "font.family": "sans-serif", "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
        "ytick.labelcolor": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.2,
        "lines.markersize": 3.5, "savefig.dpi": 300, "pdf.fonttype": 42, "figure.dpi": 150})


def plot_curve(result: RecallResult, dataset: str, folder: Path) -> list[Path]:
    """Draw the FNR-against-threshold figure of one dataset, with its cost.

    Top: each scene's FNR (thin), the mean over scenes (the CRC risk) and the one-to-one mean, the targets
    ``alpha``, the published threshold and each feasible deployable CRC threshold. Bottom: kept boxes per view,
    matched and unmatched to annotation.

    Parameters
    ----------
    result : RecallResult
        The dataset's result.
    dataset : str
        Dataset name (file name and title).
    folder : pathlib.Path
        Output folder.

    Returns
    -------
    list of pathlib.Path
        The PNG and PDF files written.
    """
    style()
    curve, by_scene = result.tables["curve"], result.tables["curve_by_scene"]
    by_alpha = result.tables["by_alpha"]
    published = result.summary.get("published_threshold")
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(WIDTH, 4.0), sharex=True,
                                      gridspec_kw={"height_ratios": [1.6, 1.0]})
    for i, (_, group) in enumerate(by_scene.groupby("scene", sort=True)):
        top.plot(group["threshold"], group["fnr"], color=MUTED, linewidth=0.4, alpha=0.35,
                 label="single scene" if i == 0 else None)
    top.plot(curve["threshold"], curve["mean_fnr"], color=BLUE, linewidth=1.4, label="mean over scenes (CRC risk)")
    top.plot(curve["threshold"], curve["mean_fnr_one_to_one"], color=ORANGE, linewidth=1.0, linestyle="--",
             label="one-to-one pairing, mean")
    infeasible, labelled = [], False
    for _, row in by_alpha.iterrows():
        top.axhline(row["alpha"], color=INK, linewidth=0.5, linestyle=":")
        top.annotate(f"α = {row['alpha']:g}", (1.0, row["alpha"]), xytext=(-2, 1.5), textcoords="offset points",
                     ha="right", va="bottom", fontsize=5.5, color=INK)
        if row["deploy_feasible"]:
            top.plot([row["deploy_threshold"]], [row["alpha"]], marker="v", color=AQUA, linestyle="none",
                     label=None if labelled else "CRC threshold, all scenes")
            labelled = True
        else:
            infeasible.append(f"{row['alpha']:g}")
    trunc = result.summary["truncation"]
    for ax in (top, bottom):
        if trunc["t_cap"] is not None and trunc["t_cap"] >= result.summary["floor"]:
            ax.axvspan(result.summary["floor"], trunc["t_cap"], color=GRID, alpha=0.6, linewidth=0, zorder=0)
        if published is not None:
            ax.axvline(published, color=BASELINE, linewidth=0.8)
    if published is not None:
        top.annotate(f"published {published:g}", (published, 0.3), xytext=(2, 0), textcoords="offset points",
                     ha="left", va="bottom", rotation=90, fontsize=5.5, color=MUTED)
    top.set_ylim(0.0, 1.02)
    top.set_ylabel("false-negative rate")
    title = f"{dataset}: {result.summary['n_scenes']} scenes, {result.summary['population']} objects"
    if trunc["t_cap"] is not None:
        title += (f"\nshaded (t ≤ {trunc['t_cap']:.2g}): box cap truncates up to "
                  f"{trunc['share_views_capped']:.0%} of views")
    if infeasible:
        title += f"\nno certified threshold for α = {', '.join(infeasible)}"
    top.set_title(title)
    top.legend(loc="upper left")
    per_view = curve["boxes_per_view"].fillna(0.0)
    unmatched = per_view * curve["unmatched_share"].fillna(0.0)
    bottom.plot(curve["threshold"], per_view, color=INK, linewidth=1.0, label="all kept boxes")
    bottom.plot(curve["threshold"], per_view - unmatched, color=BLUE, linewidth=1.0, label="matched to annotation")
    bottom.plot(curve["threshold"], unmatched, color=ORANGE, linewidth=1.0, label="unmatched to annotation")
    bottom.set_ylabel("boxes per view")
    bottom.set_xlabel("score threshold t (boxes with score ≥ t are kept)")
    bottom.legend(loc="upper right")
    fig.tight_layout()
    paths = [folder / f"fnr_vs_threshold_{dataset}.{ext}" for ext in ("png", "pdf")]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def combine(results: Mapping[str, RecallResult], name: str) -> pd.DataFrame:
    """One table across datasets, with a leading ``dataset`` column.

    Parameters
    ----------
    results : mapping of str to RecallResult
        Dataset -> result.
    name : str
        Table name.

    Returns
    -------
    pandas.DataFrame
        The datasets' tables stacked (empty if none has it).
    """
    parts = []
    for dataset, result in results.items():
        table = result.tables.get(name)
        if table is None or not len(table):
            continue
        table = table.drop(columns=["dataset"], errors="ignore")
        table.insert(0, "dataset", dataset)
        parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def headline(by_alpha: pd.DataFrame) -> str:
    """The headline table as fixed-width text.

    Parameters
    ----------
    by_alpha : pandas.DataFrame
        ``by_alpha.csv``.

    Returns
    -------
    str
        One line per dataset and alpha.
    """
    columns = ["dataset", "alpha", "n_scenes", "share_feasible", "mean_heldout_fnr", "mean_heldout_fnr_lo",
               "mean_heldout_fnr_hi", "share_scenes_above_alpha", "threshold_median", "deploy_threshold",
               "share_folds_in_truncated_region",
               "boxes_per_view", "boxes_per_view_published", "unmatched_share", "unmatched_share_published"]
    return by_alpha[[c for c in columns if c in by_alpha]].to_string(index=False, float_format="%.4g")


def summary_record(args: argparse.Namespace, config: RecallConfig, results: Mapping[str, RecallResult],
                   provenance: Mapping[str, Any], item_dir: Path, seconds: float) -> dict[str, Any]:
    """The run's ``summary.json``.

    Parameters
    ----------
    args : argparse.Namespace
        The command line.
    config : RecallConfig
        Settings.
    results : mapping of str to RecallResult
        Dataset -> result.
    provenance : mapping
        Dataset -> inputs (shard digest, prediction files, floor and how it was set).
    item_dir : pathlib.Path
        Per-item folder.
    seconds : float
        Run time.

    Returns
    -------
    dict
        Method, definitions, caveats, settings, inputs and each dataset's summary.
    """
    return {
        "method": {
            "name": "conformal risk control of the false-negative rate (as published; not new)",
            "reference": REFERENCE,
            "statement": "For exchangeable losses L_i(lambda), non-increasing and right-continuous in lambda, "
                         "bounded by B, with L_i(lambda_max) <= alpha: lambda_hat = inf{lambda : n/(n+1) R_n(lambda) "
                         "+ B/(n+1) <= alpha} (lambda_max if empty) gives E[L_{n+1}(lambda_hat)] <= alpha (Theorem 1). "
                         "Here lambda = -t for a score threshold t, lambda_max = the floor t_min, B = 1.",
            "unit": "scene (frames of one video are not exchangeable)",
            "evaluation": "leave one scene out: t_hat from the other N - 1 scenes, held-out FNR at it"},
        "definitions": {
            "loss": "a scene's FNR at t: share of its population object-views that no box with score >= t covers "
                    f"(2D IoU >= {config.min_iou:g} with the object's visible 2D box; any box, no one-to-one)",
            "population": "present: the pipeline's presence (visible 2D box, visible fraction above the presence "
                          "threshold, from the reference depth); clearly_visible: the question generator's referent "
                          "thresholds",
            "fnr_one_to_one": "FNR under the Hungarian 2D pairing of match_detections_2d against every present "
                              "object (the pipeline's recall); reported, not controlled",
            "floor_fnr": "a scene's FNR at the CRC floor t_min (the lowest threshold CRC may choose); Theorem 1 "
                         "assumes it is <= alpha",
            "box_cap": "a view with max_boxes boxes is truncated at thresholds up to its lowest written score (its "
                       "cap floor); t_cap is the largest cap floor. Above t_cap the losses are the uncapped "
                       "labeller's; at or below it they are exact for the capped labeller as run and an upper bound "
                       "for an uncapped one. truncation=floor: CRC chooses only above t_cap; mark: anywhere, with "
                       "in_truncated_region and share_views_truncated reported",
            "feasible": "some threshold met the CRC condition; if not, t_hat = t_min and no guarantee is claimed",
            "unmatched_to_annotation": "kept boxes with 2D IoU below the threshold against every present annotated "
                                       "object; ARKitScenes annotates 17 furniture classes, so these are not "
                                       "necessarily false",
            "range": "distance from the camera centre to the ground-truth box centre (m)",
            "mean_heldout_fnr": "mean over scenes of the held-out FNR at the threshold calibrated on the other "
                                "scenes. It is at most alpha by construction whenever the mean floor FNR over all "
                                "scenes is (loso_mean_bounded_by_construction), so it checks the implementation, "
                                "not exchangeability with new scenes; the spread across scenes is the information",
            "mean_heldout_fnr_lo/hi": f"{config.level:g} percentile bootstrap over scenes ({config.n_boot} "
                                      "replicates); descriptive only, since the folds share calibration scenes"},
        "caveats": [
            "The guarantee is marginal: an expectation over calibration sets and new scenes, every scene weighted "
            "equally; single scenes and strata (far, occluded, some classes) can exceed alpha.",
            "It assumes scenes exchangeable with the calibration scenes (same capture protocol and site population).",
            "Feasibility needs at least 1/alpha - 1 calibration scenes, and more the more the labeller misses.",
            "Object-views: an object seen in several views of a video counts once per view."],
        "settings": {"alphas": list(config.alphas), "population": config.population, "min_iou": config.min_iou,
                     "bound": config.bound, "published_threshold": config.published_threshold,
                     "box2d": args.box2d, "max_boxes": args.max_boxes, "truncation": args.truncation,
                     "curve_points": config.curve_points, "mondrian": list(config.mondrian),
                     "n_boot": config.n_boot, "level": config.level, "seed": config.seed},
        "inputs": {"labels": str(args.labels), "predictions": str(args.predictions), **provenance},
        "outputs": {"per_item": str(item_dir), "results": str(args.results)},
        "git_commit": git_commit(),
        "datasets": {d: r.summary for d, r in results.items()},
        "seconds": round(seconds, 1)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run recall control on every dataset and write the per-item and aggregate outputs.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Exit status: 0 on success; 1 if an output folder is refused, the name is invalid, or an input is missing
        or inconsistent.
    """
    args = parse_args(argv)
    started = time.perf_counter()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.name) or args.name in (".", ".."):
        print(f"refused: --name {args.name!r} must be a plain folder name", file=sys.stderr)
        return 1
    try:
        config = RecallConfig(alphas=tuple(args.alphas), population=args.population, min_iou=args.min_iou,
                              published_threshold=args.published_threshold, mondrian=tuple(args.mondrian),
                              n_boot=args.n_boot, seed=args.seed)
        item_dir = (args.out_root or data_root() / "outputs" / "recall_control") / args.name
        refuse_existing([item_dir, args.results], [args.labels, args.predictions])
    except (ValueError, FileExistsError, RuntimeError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1

    results: dict[str, RecallResult] = {}
    provenance: dict[str, Any] = {}
    for dataset in args.datasets:
        try:
            records, lines, info = load_dataset(args.labels, args.predictions, dataset,
                                                skip_missing=args.skip_missing)
            stated, distinct = file_floor(list(lines.values()))
            if args.floor is None and stated is None:
                raise ValueError(f"{dataset}: the prediction lines state no score_threshold; pass --floor")
            if args.floor is not None and stated is not None and args.floor < stated:
                raise ValueError(f"{dataset}: --floor {args.floor} is below the files' score_threshold {stated}")
            floor = float(args.floor if args.floor is not None else stated)
            views, below = build_views(records, lines, floor, project_3d=args.box2d == "projected",
                                       max_boxes=args.max_boxes or None)
            t_cap = truncation_threshold(views)
            crc_floor = cap_free_floor(floor, t_cap) if args.truncation == "floor" else floor
            results[dataset] = run_recall_control(views, config, floor=floor, crc_floor=crc_floor)
        except (FileNotFoundError, ValueError) as error:
            print(f"{dataset}: {error}", file=sys.stderr)
            return 1
        provenance[dataset] = {**info, "floor": floor, "floor_from": "--floor" if args.floor is not None else "files",
                               "file_score_thresholds": distinct, "n_boxes_below_floor": below,
                               "crc_floor": crc_floor}
        trunc = results[dataset].summary["truncation"]
        print(f"{dataset}: {len(views)} views, {results[dataset].summary['n_scenes']} scenes, floor {floor:g}, "
              f"{trunc['n_views_capped']} views at the {args.max_boxes}-box cap (t_cap {t_cap:.4g}), "
              f"CRC floor {crc_floor:.4g}", flush=True)

    item_dir.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    for name in PER_ITEM_TABLES:
        combine(results, name).to_csv(item_dir / f"{name}.csv", index=False, float_format="%.6g")
    (item_dir / "run.json").write_text(dumps({"argv": list(argv) if argv is not None else sys.argv[1:],
                                              "results": str(args.results), "git_commit": git_commit()}))
    for name in AGGREGATE_TABLES:
        table = combine(results, name)
        if name != "mondrian" or config.mondrian:
            table.to_csv(args.results / f"{name}.csv", index=False, float_format="%.6g")
    for dataset, result in results.items():
        plot_curve(result, dataset, args.results)
    seconds = time.perf_counter() - started
    summary = summary_record(args, config, results, provenance, item_dir, seconds)
    (args.results / "summary.json").write_text(dumps(summary))
    print(headline(combine(results, "by_alpha")))
    print(f"\nwrote {item_dir} and {args.results} ({seconds:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
