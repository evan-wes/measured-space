"""Test whether a few physical references per site restore the certified label gate out of scene.

For each dataset this loads the oracle and auto labels from ``<label root>/<dataset>/`` (as
``scripts/run_audit.py`` does, plus each question's tolerance), cross-fits the audit's feature model leave one
scene out (scenes grouped into ``--max-folds`` folds when a dataset and question type has more), and runs
:func:`measured_space.references.run_reference_experiment`: for each held-out scene, ``n`` random references
(items whose oracle answer is revealed) recalibrate the scene (scale per depth source, bias per question type,
spread factor), and the scene's other labels are gated at ``tau`` and ``2 tau``.

Outputs (``--results`` and ``--figure`` are required, with no default, so a run never overwrites the committed
results by accident; the committed run used ``--results results/references --figure
results/figures/references_curve``):

- ``<results>/<tag>/`` (``<tag>`` defaults to the label root's folder name, e.g. ``labels`` or ``labels_2d``,
  with ``_exclude_ambiguous`` appended under ``--exclude-ambiguous``): ``summary.csv`` (per dataset, variant,
  reference count and tolerance multiple), ``scenes.csv`` (the same per held-out scene, over draws) and
  ``summary.json`` (settings, input checksums, notes, runtime). Aggregates only: counts and rates per scene,
  nothing per item.
- ``<figure>.pdf`` / ``.png``: references per site against the error among accepted labels and the accepted
  share at ``tau``, per dataset, for every tag found under the results folder (the 2D-overlap labels,
  ``labels_2d``, drawn as primary when present).
- ``--cache``: the cross-fitted per-item predictions (a pickle), so a rerun skips the model fits. Per-item data:
  keep it out of the repository.

``--exclude-ambiguous`` (a pairing sensitivity analysis) drops, before the fits, every (question, depth source)
pair whose auto answer rests on an ambiguous detection pairing (:func:`measured_space.audit.exclude_ambiguous`):
auto and oracle side together, so such a pair is neither evaluated nor drawn as a reference. Pairs of unknown
ambiguity (labels written before the flag was stored) are kept. ``summary.json`` records, per dataset and depth
source, the pairs dropped and the unknown pairs kept.

Usage::

    python scripts/run_reference_experiment.py --results results/references \\
        --figure results/figures/references_curve --labels <data root>/outputs/labels_2d --wait-minutes 120
    python scripts/run_reference_experiment.py --results results/references \\
        --figure results/figures/references_curve --figure-only                # redraw from the CSVs
    python scripts/run_reference_experiment.py --results <scratch folder> --figure <scratch folder>/curve \\
        --labels <data root>/outputs/labels_2d --exclude-ambiguous             # pairing sensitivity
    python scripts/run_reference_experiment.py --synthetic                     # dry run, writes to --cache only
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from make_figures import BLUE, INK, MUTED, ORANGE, WIDTH, save, style   # scripts/ is on sys.path when run
from measured_space.audit import AuditConfig, dumps, exclude_ambiguous, exclusion_record
from measured_space.config import REPO_ROOT, data_root
from measured_space.references import (ReferenceConfig, ReferenceResult, cross_fit_key, cross_fit_sigma,
                                       load_reference_pairs, run_reference_experiment, synthetic_site_pairs)

DATASETS = ("arkitscenes", "smartspaces")
DATASET_NAME = {"arkitscenes": "ARKitScenes", "smartspaces": "Smart Spaces"}
TAG_STYLE = {"labels_2d": ("2D-overlap labels", BLUE, "o", "-"), "labels": ("3D-gated labels", ORANGE, "^", "--")}
FIGURE_VARIANT = "abc_uniform"          # the primary recipe: scale, bias and spread, referenced calibration


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str or None
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), help="datasets to run")
    parser.add_argument("--labels", type=Path, default=None,
                        help="label root holding one folder per dataset (default <data root>/outputs/labels)")
    parser.add_argument("--tag", default=None,
                        help="results subfolder (default: the label root's folder name, plus _exclude_ambiguous "
                             "with --exclude-ambiguous)")
    parser.add_argument("--results", type=Path, default=None,
                        help="aggregate-table folder, tables go to <results>/<tag>; required (no default) except "
                             "with --synthetic. The committed tables are in results/references")
    parser.add_argument("--figure", type=Path, default=None,
                        help="figure path without extension; required (no default) except with --synthetic. The "
                             "committed figure is results/figures/references_curve")
    parser.add_argument("--exclude-ambiguous", action="store_true",
                        help="drop (question, depth source) pairs whose auto answer rests on an ambiguous "
                             "detection pairing; pairs of unknown ambiguity are kept and counted")
    parser.add_argument("--cache", type=Path, default=None,
                        help="folder for the cross-fitted per-item predictions (default: no cache)")
    parser.add_argument("--n-refs", type=int, nargs="+", default=list(ReferenceConfig.n_refs),
                        help="references per site")
    parser.add_argument("--n-draws", type=int, default=ReferenceConfig.n_draws, help="random draws per scene and n")
    parser.add_argument("--seed", type=int, default=ReferenceConfig.seed, help="seed of the reference draws")
    parser.add_argument("--n-jobs", type=int, default=4, help="worker processes for cross-fitting")
    parser.add_argument("--max-folds", type=int, default=AuditConfig.max_folds,
                        help="most cross-fitting folds per dataset and question type; with more scenes, scenes are "
                             "grouped into this many folds balanced by row count (default %(default)s)")
    parser.add_argument("--wait-minutes", type=float, default=0.0,
                        help="poll for each dataset's summary.json for up to this long (0: do not wait)")
    parser.add_argument("--figure-only", action="store_true", help="only redraw the figure from the results")
    parser.add_argument("--synthetic", action="store_true",
                        help="run on synthetic sites (one with a 12 %% scale error); writes to --cache only")
    args = parser.parse_args(argv)
    if args.figure_only or not args.synthetic:
        missing = [flag for flag, value in (("--results", args.results), ("--figure", args.figure)) if value is None]
        if missing:
            parser.error(f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} required (no default: "
                         "the committed outputs are "
                         f"{REPO_ROOT / 'results' / 'references'} and "
                         f"{REPO_ROOT / 'results' / 'figures' / 'references_curve'}; pass them explicitly to "
                         "overwrite them)")
    return args


def wait_for_labels(label_root: Path, datasets: list[str], minutes: float) -> list[str]:
    """Wait until every dataset's label folder has a ``summary.json`` (the label pipeline's completion marker).

    Parameters
    ----------
    label_root : pathlib.Path
        Folder holding one folder per dataset.
    datasets : list of str
        Datasets to wait for.
    minutes : float
        Longest wait; 0 checks once.

    Returns
    -------
    list of str
        The datasets whose ``summary.json`` exists, in input order.
    """
    deadline = time.time() + 60.0 * minutes
    while True:
        ready = [d for d in datasets if (label_root / d / "summary.json").exists()]
        if len(ready) == len(datasets) or time.time() >= deadline:
            return ready
        print(f"waiting for labels: {sorted(set(datasets) - set(ready))} not finished", flush=True)
        time.sleep(min(60.0, max(deadline - time.time(), 1.0)))


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


def predictions(pairs: pd.DataFrame, audit: AuditConfig, key: dict[str, Any], cache: Path | None,
                name: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], bool]:
    """Cross-fitted predictions, read from the cache when its key matches, else fitted (and cached).

    Parameters
    ----------
    pairs : pandas.DataFrame
        The pair table (default ``RangeIndex``).
    audit : AuditConfig
        Model settings.
    key : dict
        What the predictions depend on (input checksums, model settings); a cached file with another key is
        refitted.
    cache : pathlib.Path or None
        Cache folder; None fits without caching.
    name : str
        Cache file stem.

    Returns
    -------
    test, cal : pandas.DataFrame
        :func:`~measured_space.references.cross_fit_sigma` outputs.
    notes : list of str
        Groups skipped.
    cached : bool
        Whether the predictions were read from the cache.
    """
    path = None if cache is None else cache / f"{name}_predictions.pkl"
    text = json.dumps(key, sort_keys=True, default=str)
    if path is not None and path.exists():
        stored = pd.read_pickle(path)
        if stored["key"] == text:
            print(f"read cached predictions from {path}", flush=True)
            return stored["test"], stored["cal"], stored["notes"], True
    test, cal, notes = cross_fit_sigma(pairs, audit, lambda m: print(m, flush=True))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"key": text, "test": test, "cal": cal, "notes": notes}, path)
    return test, cal, notes, False


def headline(result: ReferenceResult) -> str:
    """Plain-text tables of the main results.

    Parameters
    ----------
    result : ReferenceResult
        The experiment.

    Returns
    -------
    str
        The summary per dataset, variant, reference count and tolerance multiple.
    """
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    s = result.tables["summary"]
    cols = ["dataset", "variant", "tau_scale", "n_refs", "n_scenes", "err_pooled", "err_pooled_p05",
            "err_pooled_p95", "err_scene", "err_scene_p95", "frac_ok", "frac_accepting", "yield_pooled",
            "cov_pooled", "err_all", "n0_err_pooled", "n0_yield_pooled", "n0_frac_ok"]
    parts = [f"{variant}\n" + g[cols].round(3).to_string(index=False)
             for variant, g in s.sort_values(["variant", "dataset", "tau_scale", "n_refs"]).groupby("variant",
                                                                                                    sort=False)]
    if result.notes:
        parts.append("NOTES\n" + "\n".join(f"- {n}" for n in result.notes))
    return "\n\n".join(parts)


def fig_curve(summaries: dict[str, pd.DataFrame], epsilon: float, variant: str = FIGURE_VARIANT) -> Any:
    """References per site against the error among accepted labels and the accepted share, at ``tau``.

    Parameters
    ----------
    summaries : dict of str to pandas.DataFrame
        ``summary.csv`` per label-set tag.
    epsilon : float
        The certified bound (drawn as a hairline).
    variant : str, default FIGURE_VARIANT
        The recipe drawn.

    Returns
    -------
    matplotlib.figure.Figure
        Two rows (wrong share among accepted labels; accepted share), one column per dataset, one line per
        label set. Lines: pooled over the held-out scenes of a draw, mean over draws. Band (error row): 5-95 %
        over scenes and draws of the per-scene error among accepted labels, so a scene that breaks the bound
        shows even when the pooled rate does not.
    """
    import matplotlib.pyplot as plt

    tags = [t for t in TAG_STYLE if t in summaries] + sorted(set(summaries) - set(TAG_STYLE))
    frames = pd.concat([s.assign(tag=t) for t, s in summaries.items()], ignore_index=True)
    frames = frames[(frames["tau_scale"] == 1.0) & (frames["variant"] == variant)]
    datasets = [d for d in DATASET_NAME if d in set(frames["dataset"])]
    ticks = sorted(frames["n_refs"].unique())
    pos = {n: i for i, n in enumerate(ticks)}
    fig, grid = plt.subplots(2, len(datasets), figsize=(WIDTH, 2.9), sharex=True, sharey="row", squeeze=False,
                             gridspec_kw={"hspace": 0.14, "wspace": 0.12})
    top_max = epsilon
    for c, dataset in enumerate(datasets):
        top, bottom = grid[0][c], grid[1][c]
        for j, tag in enumerate(tags):
            label, colour, marker, ls = TAG_STYLE.get(tag, (tag, MUTED, "s", ":"))
            g = frames[(frames["dataset"] == dataset) & (frames["tag"] == tag)].sort_values("n_refs")
            if not len(g):
                continue
            x = g["n_refs"].map(pos).to_numpy(float) + 0.08 * (j - (len(tags) - 1) / 2)
            top.fill_between(x, g["err_scene_p05"], g["err_scene_p95"], color=colour, alpha=0.09, linewidth=0)
            kw = {"color": colour, "marker": marker, "linestyle": ls, "markeredgecolor": "white",
                  "markeredgewidth": 0.5, "zorder": 3}
            top.plot(x, g["err_pooled"], label=label, **kw)
            bottom.plot(x, g["yield_pooled"], **kw)
            top_max = max(top_max, float(np.nanmax(np.r_[g["err_pooled"], g["err_scene_p95"], 0.0])))
        top.axhline(epsilon, color=INK, linewidth=0.6, linestyle=(0, (4, 3)), zorder=1)
        top.set_title(DATASET_NAME.get(dataset, dataset), color=INK)
        bottom.set_xlabel("References per site")
    grid[0][-1].annotate(f"ε = {epsilon:g}", (-0.3, epsilon), xytext=(0, -1.5), textcoords="offset points",
                         ha="left", va="top", fontsize=6, color=INK)
    grid[1][0].set_xticks(range(len(ticks)), [str(n) for n in ticks])
    grid[1][0].set_xlim(-0.35, len(ticks) - 0.65)
    grid[0][0].set_ylim(0, min(1.0, 1.08 * top_max))
    grid[1][0].set_ylim(0, max(0.05, 1.15 * float(np.nanmax(np.r_[frames["yield_pooled"], 0.0]))))
    grid[0][0].set_ylabel("Wrong among accepted")
    grid[1][0].set_ylabel("Accepted share")
    handles, labels = grid[0][0].get_legend_handles_labels()
    if len(datasets) > 1 and len(grid[0][1].get_legend_handles_labels()[0]) > len(handles):
        handles, labels = grid[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=len(labels),
               handlelength=2.2, columnspacing=1.2, frameon=False)
    return fig


def draw_figure(results: Path, figure: Path, epsilon: float) -> list[Path]:
    """Draw the headline curve from every ``<results>/<tag>/summary.csv``.

    Parameters
    ----------
    results : pathlib.Path
        The references results folder.
    figure : pathlib.Path
        Output path without extension (PDF and PNG are written).
    epsilon : float
        The certified bound.

    Returns
    -------
    list of pathlib.Path
        Files written (none if there are no results).
    """
    summaries = {p.parent.name: pd.read_csv(p) for p in sorted(results.glob("*/summary.csv"))}
    if not summaries:
        print(f"no summary.csv under {results}; no figure")
        return []
    style()
    return save(fig_curve(summaries, epsilon), figure.name, [figure.parent])


def main(argv: list[str] | None = None) -> int:
    """Load labels, run the experiment, write the tables and the figure.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status: 0 on success, 1 if no dataset had labels.
    """
    args = parse_args(argv)
    started = time.time()
    audit = dataclasses.replace(AuditConfig(), n_jobs=args.n_jobs, max_folds=args.max_folds)
    config = dataclasses.replace(ReferenceConfig(), n_refs=tuple(sorted(set(args.n_refs) | {0})),
                                 n_draws=args.n_draws, seed=args.seed)
    if args.figure_only:
        for path in draw_figure(args.results, args.figure, audit.epsilon):
            print(f"wrote {path}")
        return 0
    inputs: dict[str, Any] = {}
    suffix = "_exclude_ambiguous" if args.exclude_ambiguous else ""
    if args.synthetic:
        pairs = synthetic_site_pairs(6, 80, scale_errors={"scene00": 1.12}, seed=3, noise_scale=1.0)
        tag, out = f"synthetic{suffix}", None
        audit = dataclasses.replace(audit, ridge_grid=(1.0,))
    else:
        label_root = args.labels or data_root() / "outputs" / "labels"
        tag = args.tag or f"{label_root.name}{suffix}"
        ready = wait_for_labels(label_root, args.datasets, args.wait_minutes)
        missing = sorted(set(args.datasets) - set(ready))
        if missing:
            print(f"labels not finished (no summary.json) for {missing}; running {ready}", flush=True)
        if not ready:
            return 1
        tables = []
        for dataset in ready:
            folder = label_root / dataset
            tables.append(load_reference_pairs(folder))
            inputs[dataset] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                               for p in sorted(folder.glob("qa_*.jsonl"))}
            print(f"{dataset}: {len(tables[-1])} (question, depth source) rows from {folder}", flush=True)
        pairs = pd.concat(tables, ignore_index=True)
        out = args.results / tag
    counts: dict[str, dict[str, dict[str, int]]] | None = None
    if args.exclude_ambiguous:
        pairs, counts = exclude_ambiguous(pairs)
    exclusion = exclusion_record(counts)
    if counts is not None:
        print(f"--exclude-ambiguous: dropped {exclusion['n_dropped']} pairs, kept {exclusion['n_unknown_kept']} "
              f"of unknown ambiguity: {dumps(exclusion['counts'])}", flush=True)
    pairs = pairs.reset_index(drop=True)
    # max_folds enters the key only when some group has more scenes, so caches written before it stay valid
    key: dict[str, Any] = {"inputs": inputs, "audit": cross_fit_key(pairs, audit), "tag": tag, "n_rows": len(pairs)}
    if counts is not None:                    # only then, so caches written before the option stay valid
        key["exclude_ambiguous"] = exclusion
    test, cal, notes, cached = predictions(pairs, audit, key, args.cache, tag)
    fitted = time.time()
    result = run_reference_experiment(pairs, config, audit, predictions=(test, cal),
                                      log=lambda m: print(m, flush=True))
    result.notes[:0] = notes
    summary = {"tag": tag, "config": dataclasses.asdict(config), "audit_config": dataclasses.asdict(audit),
               "gate_z": audit.gate_z, "datasets": sorted(pairs["dataset"].unique()), "n_rows": int(len(pairs)),
               "inputs": inputs, "exclude_ambiguous": exclusion, "notes": result.notes, "git_commit": git_commit(),
               "predictions": "read from cache" if cached else "fitted", "seconds_fit": round(fitted - started, 1),
               "seconds_experiment": round(time.time() - fitted, 1)}
    folders = [f for f in (out, None if args.cache is None else args.cache / tag) if f is not None]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in result.tables.items():
            frame.to_csv(folder / f"{name}.csv", index=False, float_format="%.6g")
        (folder / "summary.json").write_text(dumps(summary))
    if args.cache is not None:
        result.draws.to_csv(args.cache / tag / "draws.csv.gz", index=False, float_format="%.6g")
    print(headline(result))
    if out is not None:
        for path in draw_figure(args.results, args.figure, audit.epsilon):
            print(f"wrote {path}")
    print(f"\nwrote {', '.join(str(f) for f in folders)} (fit {summary['seconds_fit']} s, experiment "
          f"{summary['seconds_experiment']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
