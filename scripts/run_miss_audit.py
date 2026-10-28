"""Audit the misses (false negatives) of automatic 3D labellers, and flag likely misses without ground truth.

Reads stored label shards (``scripts/run_labels.py`` or ``scripts/run_box_source.py``), re-associates each view
with the pipeline's 2D association (no models, no GPU) and, per labeller and source (:mod:`measured_space.misses`):

- **recall** of present ground-truth objects, overall and by range, visible fraction, truncation, class and scene,
  with Wilson and scene-cluster bootstrap intervals, and why each miss happened;
- **effects on answers**: questions left unanswered by a miss, count errors split into misses, class confusions
  and extra boxes, and silent closest-instance errors under open-world answering;
- optionally, **miss flags** that need no ground truth, with their precision and recall measured against it:
  ``cross_labeller`` (CuTR boxes from ``--predictions``, both directions), ``cross_frame`` (static ARKitScenes
  videos, iPad LiDAR occlusion test; LiDAR depth is read from the dataset) and ``cross_camera`` (Smart Spaces
  cameras at the same frame index, geometry only).

Inputs:

- ``--labels [NAME=]FOLDER [[DATASET:]SOURCE ...]`` (repeatable): a label folder and the sources to audit. A
  source may be restricted to one dataset (``arkitscenes:lidar``). Without sources, every stored source is
  audited. ``NAME`` (default: the folder's name) is the labeller's name in the outputs;
- ``--predictions NAME=FOLDER`` (repeatable): a CuTR predictions folder (``<folder>/<dataset>/<scene>.jsonl``) for
  cross-labeller flags.

Outputs (refused if a folder exists and is not empty; nothing is ever overwritten):

- per item, under ``<data root>/outputs/misses/<--name>/``: ``objects.csv`` (one row per present object, labeller
  and source), ``questions.csv`` (one row per question, labeller and source), ``flag_rows.csv``,
  ``flag_targets.csv`` and ``flag_views.csv`` (flags, the present objects of covered views, the covered views),
  and ``manifest.json``;
- aggregates, under ``--results`` (required): ``recall.csv``, ``miss_causes.csv``, ``answer_effects.csv``,
  ``flag_metrics.csv``, ``comparison_check.csv`` and ``summary.json``.

Run from the repository root (CPU only)::

    python scripts/run_miss_audit.py --name labels2d \\
        --labels v0=<data>/outputs/labels_2d arkitscenes:lidar smartspaces:moge \\
        --flags cross_frame --flag-scenes 42444946 42444966 --results <folder> \\
        --compare-with results/association/comparison.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from measured_space.box_sources import ViewPredictions, predictions_file, read_box_predictions
from measured_space.config import REPO_ROOT, data_root
from measured_space.misses import (
    FLAG_KINDS,
    FLAG_ROW_COLUMNS,
    MISS_CAUSES,
    MULTI_CAMERA_DATASETS,
    RANGE_BINS_M,
    STATIC_DATASETS,
    TARGET_ROW_COLUMNS,
    TRUNCATION_BINS,
    VIEW_ROW_COLUMNS,
    VISIBLE_FRACTION_BINS,
    FlagConfig,
    FlagResult,
    cross_camera_flags,
    cross_frame_flags,
    cross_labeller_flags,
    effects_table,
    flag_table,
    miss_cause_table,
    object_rows,
    question_rows,
    load_scene,
    recall_table,
    scene_shards,
)
from measured_space.pipeline import ViewResult, to_json_line

DATASETS: tuple[str, ...] = ("arkitscenes", "smartspaces")
"""tuple of str: datasets a label folder may hold."""

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
"""re.Pattern: allowed run and labeller names (one path component, no leading dot)."""


@dataclass(frozen=True)
class LabelSpec:
    """One ``--labels`` argument.

    Parameters
    ----------
    name : str
        The labeller's name in the outputs.
    folder : pathlib.Path
        The label folder.
    sources : tuple of tuple of (str or None, str)
        ``(dataset or None for any, source)`` pairs to audit; empty audits every stored source.
    """

    name: str
    folder: Path
    sources: tuple[tuple[str | None, str], ...]

    def wanted(self, dataset: str, stored: Sequence[str]) -> list[str]:
        """The sources to audit in one dataset.

        Parameters
        ----------
        dataset : str
            Dataset name.
        stored : sequence of str
            The sources the dataset's views hold.

        Returns
        -------
        list of str
            The requested sources (all stored ones if none were named), in stored order.
        """
        if not self.sources:
            return list(stored)
        named = {s for d, s in self.sources if d is None or d == dataset}
        return [s for s in stored if s in named]


def parse_label_spec(tokens: Sequence[str]) -> LabelSpec:
    """Parse one ``--labels [NAME=]FOLDER [[DATASET:]SOURCE ...]`` argument.

    Parameters
    ----------
    tokens : sequence of str
        The folder (with an optional name) followed by sources.

    Returns
    -------
    LabelSpec
        The parsed argument.

    Raises
    ------
    ValueError
        If the name is not a plain identifier, a dataset is unknown, or the folder does not exist.
    """
    head, *rest = tokens
    name, _, folder = head.partition("=") if "=" in head else ("", "", head)
    path = Path(folder).expanduser()
    name = name or path.name
    if not NAME_PATTERN.match(name):
        raise ValueError(f"labeller name {name!r} must be a plain name ([A-Za-z0-9_.-], no leading dot)")
    if not path.is_dir():
        raise ValueError(f"label folder {path} does not exist")
    sources: list[tuple[str | None, str]] = []
    for token in rest:
        dataset, _, source = token.rpartition(":")
        if dataset and dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r} in source {token!r}; known: {DATASETS}")
        sources.append((dataset or None, source))
    return LabelSpec(name, path, tuple(sources))


def parse_predictions(token: str) -> tuple[str, Path]:
    """Parse one ``--predictions NAME=FOLDER`` argument.

    Parameters
    ----------
    token : str
        ``NAME=FOLDER``.

    Returns
    -------
    tuple of (str, pathlib.Path)
        The witness name and the predictions folder.

    Raises
    ------
    ValueError
        If the name is missing or not a plain name, or the folder does not exist.
    """
    name, sep, folder = token.partition("=")
    if not sep or not NAME_PATTERN.match(name):
        raise ValueError(f"--predictions takes NAME=FOLDER with a plain name, got {token!r}")
    path = Path(folder).expanduser()
    if not path.is_dir():
        raise ValueError(f"predictions folder {path} does not exist")
    return name, path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and check the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The options, with ``label_specs`` (list of :class:`LabelSpec`), ``prediction_sets`` (dict of name to
        folder) and ``out`` (the per-item folder) added.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--labels", nargs="+", action="append", required=True, metavar="ARG",
                   help="[NAME=]FOLDER [[DATASET:]SOURCE ...]: a label folder and the sources to audit (repeatable)")
    p.add_argument("--predictions", action="append", default=[], metavar="NAME=FOLDER",
                   help="CuTR predictions folder for cross-labeller flags (repeatable)")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=None,
                   help="datasets to audit (default: every dataset with shards in a label folder)")
    p.add_argument("--scenes", nargs="+", default=None, help="audit only these scenes (all by default)")
    p.add_argument("--name", required=True, help="run name: per-item tables go to <data root>/outputs/misses/<name>")
    p.add_argument("--results", type=Path, required=True,
                   help="aggregate-table folder (must not exist or be empty; no default)")
    p.add_argument("--flags", nargs="*", choices=FLAG_KINDS, default=[], help="miss flags to compute and measure")
    p.add_argument("--flag-scenes", nargs="+", default=None, help="compute flags only in these scenes")
    p.add_argument("--n-boot", type=int, default=2000, help="scene-cluster bootstrap replicates")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--min-candidates", type=int, default=2,
                   help="paired candidates an open-world closest-instance answer needs")
    defaults = FlagConfig()
    p.add_argument("--dedupe-iou", type=float, default=defaults.dedupe_iou,
                   help="3D prism IoU at which boxes from different views/cameras are one object")
    p.add_argument("--occlusion-margin", type=float, default=defaults.occlusion_margin,
                   help="metres a projected box may lie behind the depth map and count as unoccluded")
    p.add_argument("--min-support", type=int, default=defaults.min_support,
                   help="other views (cameras) a projected box needs")
    p.add_argument("--compare-with", type=Path, default=None,
                   help="a comparison.csv of scripts/compare_association.py to check overall recall against (only "
                        "meaningful for the label folder it was made from, e.g. results/association/comparison.csv "
                        "for labels_2d); skipped by default")
    args = p.parse_args(argv)
    if not NAME_PATTERN.match(args.name):
        p.error(f"--name {args.name!r} must be a plain name ([A-Za-z0-9_.-], no leading dot)")
    try:
        args.label_specs = [parse_label_spec(tokens) for tokens in args.labels]
        args.prediction_sets = dict(parse_predictions(t) for t in args.predictions)
    except ValueError as error:
        p.error(str(error))
    names = [s.name for s in args.label_specs]
    if len(set(names)) != len(names):
        p.error(f"labeller names repeat: {names}; name them with NAME=FOLDER")
    if len(args.prediction_sets) != len(args.predictions):
        p.error("--predictions names repeat")
    if "cross_labeller" in args.flags and not args.prediction_sets:
        p.error("cross_labeller flags need --predictions")
    if args.compare_with is not None and not args.compare_with.is_file():
        p.error(f"--compare-with {args.compare_with} is not a file")
    args.out = data_root() / "outputs" / "misses" / args.name
    for folder in (args.out, args.results):
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            p.error(f"{folder} exists and is not empty; choose another --name / --results")
    if args.results.resolve() == args.out.resolve():
        p.error("--results must differ from the per-item folder")
    inputs = [s.folder.resolve() for s in args.label_specs] + [f.resolve() for f in args.prediction_sets.values()]
    for folder in (args.out.resolve(), args.results.resolve()):
        if any(folder == i or i in folder.parents for i in inputs):
            p.error(f"{folder} lies inside an input folder")
    return args


def sha256(path: Path) -> str:
    """Checksum of a file.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    str
        Hex SHA-256 digest.
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


def wanted_scene(scene: str, scenes: Sequence[str] | None) -> bool:
    """Whether a scene passes a scene filter (``/`` or ``__`` spelling).

    Parameters
    ----------
    scene : str
        The scene as the views name it.
    scenes : sequence of str or None
        The filter; ``None`` passes every scene.

    Returns
    -------
    bool
        True if the scene is wanted.
    """
    return scenes is None or scene.replace("/", "__") in {s.replace("/", "__") for s in scenes}


class PredictionCache:
    """CuTR predictions per set and scene, read on first use, with the checksum of every file read.

    Parameters
    ----------
    sets : dict of str to pathlib.Path
        Witness name -> predictions folder.
    """

    def __init__(self, sets: dict[str, Path]) -> None:
        """Remember the folders.

        Parameters
        ----------
        sets : dict of str to pathlib.Path
            Witness name -> predictions folder.
        """
        self.sets = sets
        self._scenes: dict[tuple[str, str, str], dict[str, ViewPredictions]] = {}
        self.files: dict[str, dict[str, str]] = {name: {} for name in sets}

    def view(self, name: str, dataset: str, scene: str, view_key: str) -> ViewPredictions | None:
        """One view's predictions from one set.

        Parameters
        ----------
        name : str
            Witness name.
        dataset : str
            Dataset name.
        scene : str
            Scene name.
        view_key : str
            The view.

        Returns
        -------
        ViewPredictions or None
            The view's line, or ``None`` if the scene has no predictions file or the file lacks the view.
        """
        key = (name, dataset, scene)
        if key not in self._scenes:
            path = predictions_file(self.sets[name], dataset, scene)
            self._scenes[key] = {} if path is None else read_box_predictions(path)
            if path is not None:
                self.files[name][str(path.relative_to(self.sets[name]))] = sha256(path)
        return self._scenes[key].get(view_key)


@dataclass
class Tables:
    """Every per-item row of the run.

    Parameters
    ----------
    objects : list of dict
        :func:`measured_space.misses.object_rows` rows.
    questions : list of dict
        :func:`measured_space.misses.question_rows` rows.
    flags : FlagResult
        Flag, target and view rows.
    counts : dict
        Bookkeeping: shards read per labeller and dataset, and datasets skipped for lack of shards.
    no_predictions : dict of str to set of str
        Witness name -> view keys flagged for which it has no prediction line (not covered).
    """

    objects: list[dict[str, Any]]
    questions: list[dict[str, Any]]
    flags: FlagResult
    counts: dict[str, Any]
    no_predictions: dict[str, set[str]]


def audit_scene(results: Sequence[ViewResult], spec: LabelSpec, dataset: str, args: argparse.Namespace,
                config: FlagConfig, predictions: PredictionCache, tables: Tables) -> None:
    """Audit one scene's views for one label folder, appending rows to ``tables``.

    Parameters
    ----------
    results : sequence of ViewResult
        The scene's views, re-associated.
    spec : LabelSpec
        The label folder and sources.
    dataset : str
        Dataset name.
    args : argparse.Namespace
        Options (``flags``, ``flag_scenes``, ``min_candidates``).
    config : FlagConfig
        Flag settings.
    predictions : PredictionCache
        CuTR predictions.
    tables : Tables
        Receives the rows.
    """
    stored = list(dict.fromkeys(s for r in results for s in r.sources))
    sources = spec.wanted(dataset, stored)
    scene = results[0].view.scene
    flagging = wanted_scene(scene, args.flag_scenes)
    for source in sources:
        for r in results:
            tables.objects += object_rows(r, source, spec.name)
            tables.questions += question_rows(r, source, spec.name, min_candidates=args.min_candidates)
        if not flagging:
            continue
        if "cross_labeller" in args.flags:
            for witness in predictions.sets:
                for r in results:
                    line = predictions.view(witness, dataset, scene, r.key)
                    if line is None:
                        tables.no_predictions.setdefault(witness, set()).add(r.key)
                        continue
                    tables.flags.extend(cross_labeller_flags(r, source, spec.name, line, witness, config))
        if "cross_frame" in args.flags and dataset in STATIC_DATASETS:
            tables.flags.extend(cross_frame_flags(results, source, spec.name, config))
        if "cross_camera" in args.flags and dataset in MULTI_CAMERA_DATASETS:
            tables.flags.extend(cross_camera_flags(results, source, spec.name, config))


def comparison_check(recall: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    """Overall recall against ``comparison.csv`` (2D association rows), where a dataset and source match.

    Parameters
    ----------
    recall : pandas.DataFrame
        :func:`measured_space.misses.recall_table` output.
    path : pathlib.Path or None
        A ``comparison.csv`` of ``scripts/compare_association.py``; ``None`` gives an empty table.

    Returns
    -------
    pandas.DataFrame
        Columns ``labeller, source, dataset, n_present, n_paired, recall, comparison_n_present,
        comparison_n_pairs, comparison_recall, agrees`` (counts equal).
    """
    columns = ["labeller", "source", "dataset", "n_present", "n_paired", "recall", "comparison_n_present",
               "comparison_n_pairs", "comparison_recall", "agrees"]
    if path is None:
        return pd.DataFrame(columns=columns)
    reference = pd.read_csv(path)
    reference = reference[reference["association"] == "2d"]
    overall = recall[(recall["population"] == "present") & (recall["by"] == "all")]
    rows = []
    for _, r in overall.iterrows():
        match = reference[(reference["dataset"] == r["dataset"]) & (reference["depth_source"] == r["source"])]
        if match.empty:
            continue
        m = match.iloc[0]
        rows.append({"labeller": r["labeller"], "source": r["source"], "dataset": r["dataset"],
                     "n_present": int(r["n"]), "n_paired": int(r["n_paired"]), "recall": r["recall"],
                     "comparison_n_present": int(m["n_present"]), "comparison_n_pairs": int(m["n_pairs"]),
                     "comparison_recall": float(m["recall"]),
                     "agrees": bool(int(r["n"]) == int(m["n_present"]) and int(r["n_paired"]) == int(m["n_pairs"]))})
    return pd.DataFrame(rows, columns=columns)


DEFINITIONS: dict[str, str] = {
    "present": "ground-truth objects with a visible 2D box and a visible fraction above the presence threshold",
    "clearly_visible": "present objects with visible fraction >= 0.3 and truncated fraction <= 0.3 (question "
                       "referent thresholds)",
    "paired": "the pipeline's 2D association (Hungarian on 2D IoU >= 0.3, class-agnostic) pairs a labelled box "
              "(a detection with a box from the source) with the object; views re-associated with rematch_view",
    "miss_causes": "no_detection: no detection overlaps the visible box at the threshold; unboxed: only "
                   "detections without a box from the source do; assignment: a boxed detection does, but was "
                   "assigned to another object",
    "range": "camera centre to ground-truth box centre (m), bins of scripts/compare_association.py",
    "wilson": "Wilson 95% interval (objects independent)",
    "bootstrap": "scene-cluster bootstrap 95% percentile interval (whole scenes resampled), overall rows only",
    "n_views": "views with at least one object of the group",
    "coverage": "questions with an automatic answer / questions; every unanswered question has a missed referent",
    "count_silent_miss": "a wrong automatic count with at least one missed instance of the class",
    "closest_open_world": "with the reference and >= --min-candidates candidates paired, answer among the paired "
                          "candidates (a labeller offers only what it found)",
    "closest_silent_miss": "a wrong open-world closest-instance answer whose true answer object was missed",
    "flag_hit": "a flag overlapping (2D IoU >= 0.3) the visible box of a present object the flagged labeller missed",
    "flag_precision": "hits / flags",
    "flag_recall": "misses some flag hits / misses in the covered views",
    "unannotated_flag": "a flag overlapping no present ground-truth object at the threshold; NOT a false flag: "
                        "ARKitScenes annotates 17 furniture classes and CuTR is class-agnostic, so many point at "
                        "real objects without a box. n_unannotated_near counts those overlapping a present object at "
                        "IoU >= near_iou (likely mislocalised)",
    "flaggable": "projected flags: a miss whose ground-truth box passes the same projection test",
    "cross_labeller": "witness boxes (CuTR) with no labelled box at 2D IoU >= 0.3, both directions; the reverse "
                      "direction's labeller is the predictions set's raw boxes (source 'boxes')",
    "cross_frame": "the labeller's boxes from other views of the same static video, clustered in 3D, projected; "
                   "kept if mostly inside the image and unoccluded by the view's iPad LiDAR depth (never laser)",
    "cross_camera": "the labeller's boxes from other cameras at the same frame index, projected; geometry only "
                    "(no depth independent of the reference is stored)",
}
"""dict of str to str: definitions written into ``summary.json``."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the miss audit and write per-item and aggregate tables.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status: 0 on success, 1 if nothing was audited.
    """
    args = parse_args(argv)
    started = time.perf_counter()
    config = FlagConfig(dedupe_iou=args.dedupe_iou, occlusion_margin=args.occlusion_margin,
                        min_support=args.min_support)
    predictions = PredictionCache(args.prediction_sets)
    tables = Tables([], [], FlagResult(), {"shards": {}, "skipped_no_shards": []}, {})
    for spec in args.label_specs:
        for dataset in args.datasets or DATASETS:
            if not (spec.folder / dataset / "shards").is_dir():
                tables.counts["skipped_no_shards"].append(f"{spec.name}/{dataset}")
                continue
            groups = scene_shards(spec.folder, dataset, args.scenes)
            tables.counts["shards"][f"{spec.name}/{dataset}"] = sum(len(v) for v in groups.values())
            for i, paths in enumerate(groups.values(), 1):
                t = time.perf_counter()
                results = load_scene(paths)
                audit_scene(results, spec, dataset, args, config, predictions, tables)
                print(f"{spec.name} {dataset} {results[0].view.scene}: {len(results)} views "
                      f"({i}/{len(groups)}, {time.perf_counter() - t:.1f} s)", flush=True)
    if not tables.objects:
        print("nothing audited: no views with the requested sources", file=sys.stderr)
        return 1

    objects = pd.DataFrame(tables.objects)
    questions = pd.DataFrame(tables.questions)
    flags = pd.DataFrame(tables.flags.flags, columns=list(FLAG_ROW_COLUMNS))
    targets = pd.DataFrame(tables.flags.targets, columns=list(TARGET_ROW_COLUMNS))
    views = pd.DataFrame(tables.flags.views, columns=list(VIEW_ROW_COLUMNS))
    recall = recall_table(objects, n_boot=args.n_boot, seed=args.seed)
    aggregates = {"recall": recall, "miss_causes": miss_cause_table(objects),
                  "answer_effects": effects_table(questions),
                  "flag_metrics": flag_table(flags, targets, views, config),
                  "comparison_check": comparison_check(recall, args.compare_with)}

    for folder in (args.out, args.results):                          # re-check: nothing may have appeared meanwhile
        if folder.exists() and any(folder.iterdir()):
            print(f"{folder} is no longer empty; nothing written", file=sys.stderr)
            return 1
    args.out.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    per_item = {"objects": objects, "questions": questions, "flag_rows": flags, "flag_targets": targets,
                "flag_views": views}
    for name, frame in per_item.items():
        frame.to_csv(args.out / f"{name}.csv", index=False, float_format="%.6g")
    for name, frame in aggregates.items():
        frame.to_csv(args.results / f"{name}.csv", index=False, float_format="%.6g")

    record = {
        "argv": list(sys.argv[1:] if argv is None else argv), "git_commit": git_commit(),
        "labels": [{"name": s.name, "folder": str(s.folder), "sources": [f"{d}:{x}" if d else x for d, x in s.sources]}
                   for s in args.label_specs],
        "predictions": {name: {"folder": str(folder), "files_sha256": predictions.files[name]}
                        for name, folder in args.prediction_sets.items()},
        "counts": {**tables.counts, "views_without_predictions": {w: len(v) for w, v in tables.no_predictions.items()}},
        "flags": args.flags, "flag_scenes": args.flag_scenes, "scenes": args.scenes,
        "settings": {"flag_config": config.to_json(), "n_boot": args.n_boot, "seed": args.seed,
                     "min_candidates": args.min_candidates, "range_bins_m": [str(b) for b in RANGE_BINS_M],
                     "visible_fraction_bins": list(VISIBLE_FRACTION_BINS), "truncation_bins": list(TRUNCATION_BINS),
                     "miss_causes": list(MISS_CAUSES)},
        "compare_with": None if args.compare_with is None else str(args.compare_with),
        "per_item_folder": str(args.out), "results_folder": str(args.results),
        "seconds": round(time.perf_counter() - started, 1)}
    (args.out / "manifest.json").write_text(json.dumps(json.loads(to_json_line(record)), indent=1))
    overall = recall[recall["by"] == "all"]
    summary = {**record, "definitions": DEFINITIONS,
               "recall_overall": overall.to_dict(orient="records"),
               "comparison_check": aggregates["comparison_check"].to_dict(orient="records"),
               "answer_effects_all_types": aggregates["answer_effects"][
                   aggregates["answer_effects"]["qtype"].isin(["all", "count", "closest_instance"])
               ].to_dict(orient="records"),
               "flag_metrics_all": aggregates["flag_metrics"][aggregates["flag_metrics"]["by"] == "all"
                                                              ].to_dict(orient="records")}
    (args.results / "summary.json").write_text(json.dumps(json.loads(to_json_line(summary)), indent=1))

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(overall[overall["population"] == "present"][["labeller", "source", "dataset", "n", "n_paired", "recall",
                                                            "wilson_lo", "wilson_hi", "boot_lo", "boot_hi"]]
              .to_string(index=False))
        if not aggregates["comparison_check"].empty:
            print(aggregates["comparison_check"].to_string(index=False))
        metrics = aggregates["flag_metrics"]
        if not metrics.empty:
            print(metrics[metrics["by"] == "all"][["kind", "labeller", "source", "witness", "dataset", "n_flags",
                                                   "precision", "n_unannotated", "n_missed", "flag_recall"]]
                  .to_string(index=False))
    print(f"\nwrote per-item tables to {args.out} and aggregates to {args.results} ({record['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
