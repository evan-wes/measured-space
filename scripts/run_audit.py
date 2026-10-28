"""Run the label audit on the generated oracle and auto labels, and write its tables.

For each dataset this reads ``<label root>/<dataset>/qa_oracle.jsonl`` and every ``qa_auto_<depth source>.jsonl``
next to it (written by the label pipeline), and runs :func:`measured_space.audit.run_audit`: errors per question
type and depth source, the structured error model with leave-one-scene-out predictions, Mondrian conformal
calibration, categorical p(correct), and the Learn-then-Test label gate. A dataset and question type with more
than ``--max-folds`` scenes (default 20) is cross-fitted on that many folds of scenes balanced by row count
instead of scene by scene (:func:`measured_space.audit.cross_fit`); the assignment is written as ``folds.csv``
and the setting to ``summary.json``.

Outputs:

- ``<data root>/outputs/audit/`` (or ``--out``; with ``--exclude-ambiguous`` the default is
  ``<data root>/outputs/audit_exclude_ambiguous/``): per-item tables (``pairs.csv``, ``numeric.csv``,
  ``categorical.csv``, ``gate.csv``) that the figures are drawn from, a copy of every aggregate table, the numeric
  leave-two-out calibration predictions and per-test-scene conformal quantiles (``calibration_numeric.csv.gz``,
  ``calibration_quantiles.csv.gz``, described by ``calibration.json``; :func:`write_calibration`), and
  ``accepted_ids.json``: ``{"settings": {...}, "accepted": {dataset: {"auto_<depth source>": [item ids]}}}``, the
  labels the primary gate accepted (numeric: feature model at the question's tolerance; categorical: logistic
  p(correct); thresholds certified by Learn-then-Test on held-out scenes), for scoring models on the accepted
  item set. These hold per-item derived labels, so they stay out of the repository.
- ``--results`` (required, no default, so a run never overwrites the committed tables by accident; the
  committed run used ``results/audit``): aggregate tables only (CSV) and ``summary.json`` (settings, counts,
  input checksums, notes, runtime). These are small and meant for committing.

``--exclude-ambiguous`` (a pairing sensitivity analysis) drops, before the audit, every (question, depth source)
pair whose auto answer rests on an ambiguous detection pairing (:func:`measured_space.audit.exclude_ambiguous`):
auto and oracle side together. Labels whose ambiguity is unknown (written before the flag was stored; re-match
them with ``scripts/run_labels.py --rematch-from`` to add it) are kept. ``summary.json`` records, per dataset and
depth source, the rows dropped and the unknown rows kept.

``--depth-sources NAME [NAME ...]`` audits only those depth sources' auto answers (e.g. ``lidar`` alone of
``laser``, ``lidar`` and ``moge``): only ``qa_oracle.jsonl`` and ``qa_auto_<NAME>.jsonl`` are read
(:func:`measured_space.audit.load_labels`), so the other sources' (question, depth source) rows never enter the
audit, its fits, its conformal cells or its gate certificates. Every NAME must have a file in each audited label
folder. ``summary.json`` records the selection under ``depth_sources`` and lists only the files read.

``--fold-groups PATH`` cross-fits by explicit groups of scenes instead of scene by scene
(:attr:`measured_space.audit.AuditConfig.fold_groups`): ``PATH`` is a JSON object ``{dataset: {scene: group}}``,
scene ids as in the labels (e.g. ``"MTMC_Tracking_2026/train/Warehouse_000"``), group labels any strings (e.g.
``configs/smartspaces_layout_groups.json``, written by ``scripts/group_smartspaces_layouts.py``). For a listed
dataset each group is one fold, whatever ``--max-folds``: leave one group out for the test predictions and two
for calibration. Every scene of a listed dataset must be mapped, and every listed dataset must be audited; other
datasets keep the default folds. The mapping is written to ``summary.json`` (``config.fold_groups``, plus the
file's path and checksum under ``fold_groups``) and the assignment to ``folds.csv`` with a ``group`` column. A
run without either option writes exactly what it wrote before they existed.

Usage::

    python scripts/run_audit.py --results results/audit                       # both datasets, default settings
    python scripts/run_audit.py --results results/audit --wait-minutes 120    # wait for the labels' summary.json
    python scripts/run_audit.py --results <scratch folder> --exclude-ambiguous --labels <label root>
    python scripts/run_audit.py --results <scratch folder> --labels <label root> --depth-sources lidar
    python scripts/run_audit.py --results <scratch folder> --labels <label root> \
        --fold-groups configs/smartspaces_layout_groups.json
    python scripts/run_audit.py --synthetic       # dry run on synthetic data (writes under --out only)
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

import pandas as pd

from measured_space.audit import (AuditConfig, AuditResult, dumps, exclude_ambiguous, exclusion_record, load_labels,
                                  run_audit, synthetic_pairs)
from measured_space.config import REPO_ROOT, data_root

DATASETS = ("arkitscenes", "smartspaces")
PER_ITEM_DROP = ("oracle", "auto")      # answers as text are not needed downstream


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
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), help="datasets to audit")
    parser.add_argument("--labels", type=Path, default=None,
                        help="label root holding one folder per dataset (default <data root>/outputs/labels)")
    parser.add_argument("--out", type=Path, default=None,
                        help="per-item output folder (default <data root>/outputs/audit, or "
                             "<data root>/outputs/audit_exclude_ambiguous with --exclude-ambiguous)")
    parser.add_argument("--results", type=Path, default=None,
                        help="aggregate-table folder; required (no default), except with --synthetic, which "
                             "writes no aggregate folder. The committed tables live in results/audit")
    parser.add_argument("--exclude-ambiguous", action="store_true",
                        help="drop (question, depth source) pairs whose auto answer rests on an ambiguous "
                             "detection pairing; pairs of unknown ambiguity are kept and counted")
    parser.add_argument("--epsilon", type=float, default=AuditConfig.epsilon, help="certified error bound")
    parser.add_argument("--delta", type=float, default=AuditConfig.delta, help="certificate failure probability")
    parser.add_argument("--kappa", type=float, default=AuditConfig.kappa, help="numeric reject threshold (tolerances)")
    parser.add_argument("--ridge-grid", type=float, nargs="+", default=list(AuditConfig.ridge_grid),
                        help="candidate ridge weights, chosen per fit by inner leave-one-scene-out CV "
                             "(grouped K-fold above --max-folds scenes; one value: fixed)")
    parser.add_argument("--n-jobs", type=int, default=4, help="worker processes for cross-fitting")
    parser.add_argument("--max-folds", type=int, default=AuditConfig.max_folds,
                        help="most cross-fitting folds per dataset and question type; with more scenes, scenes are "
                             "grouped into this many folds balanced by row count, and leave-one-scene-out "
                             "becomes leave-one-fold-out (default %(default)s)")
    parser.add_argument("--n-boot", type=int, default=AuditConfig.n_boot, help="scene-cluster bootstrap replicates")
    parser.add_argument("--depth-sources", nargs="+", default=None, metavar="NAME",
                        help="audit only these depth sources' auto answers (qa_auto_<NAME>.jsonl, e.g. lidar); "
                             "the others are never loaded (default: every source in the label folder)")
    parser.add_argument("--fold-groups", type=Path, default=None, metavar="PATH",
                        help="JSON {dataset: {scene: group}}: cross-fit each listed dataset with one fold per group "
                             "(leave one group out), ignoring --max-folds; every scene must be mapped (default: by "
                             "scene)")
    parser.add_argument("--wait-minutes", type=float, default=0.0,
                        help="poll for each dataset's summary.json for up to this long (0: do not wait)")
    parser.add_argument("--synthetic", action="store_true",
                        help="audit synthetic data instead (outputs under <out>/_synthetic, results/ untouched)")
    args = parser.parse_args(argv)
    if args.synthetic and args.depth_sources:
        parser.error("--depth-sources selects label files; it does not apply to --synthetic")
    if args.results is None and not args.synthetic:
        parser.error("--results is required: the aggregate-table folder (the committed tables are in "
                     f"{REPO_ROOT / 'results' / 'audit'}; pass it explicitly to overwrite them)")
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


def read_fold_groups(path: Path) -> dict[str, dict[str, str]]:
    """Read the explicit scene-to-group mapping of ``--fold-groups``.

    Parameters
    ----------
    path : pathlib.Path
        JSON file holding ``{dataset: {scene: group}}``: scene ids as in the labels, group labels strings (or
        integers).

    Returns
    -------
    dict of str to dict of str to str
        ``{dataset: {scene: group}}``, every key and label as str.

    Raises
    ------
    ValueError
        If the file is not a non-empty object of non-empty objects (for example a flat ``{scene: group}``
        object, which names no dataset) or a group label is not a string or an integer.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not data or not all(isinstance(m, dict) and m for m in data.values()):
        raise ValueError(f"{path}: expected a JSON object {{dataset: {{scene: group}}}} with at least one scene "
                         "per dataset")
    bad = [f"{d}/{s}" for d, m in data.items() for s, g in m.items()
           if isinstance(g, bool) or not isinstance(g, (str, int))]
    if bad:
        raise ValueError(f"{path}: group labels must be strings or integers, not for {bad[:5]}")
    return {str(d): {str(s): str(g) for s, g in m.items()} for d, m in data.items()}


def write_tables(result: AuditResult, out: Path, results: Path | None) -> None:
    """Write per-item tables to ``out`` and aggregate tables to ``out`` and ``results``.

    Parameters
    ----------
    result : AuditResult
        The audit.
    out : pathlib.Path
        Per-item output folder (created if needed).
    results : pathlib.Path or None
        Aggregate-table folder (created if needed); None writes aggregates to ``out`` only.
    """
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("pairs", result.pairs), ("numeric", result.numeric), ("categorical", result.categorical),
                        ("gate", result.gate)):
        frame.drop(columns=[c for c in PER_ITEM_DROP if c in frame], errors="ignore").to_csv(
            out / f"{name}.csv", index=False, float_format="%.6g")
    for folder in [out / "tables"] + ([results] if results is not None else []):
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in result.tables.items():
            frame.to_csv(folder / f"{name}.csv", index=False, float_format="%.6g")
    write_calibration(result, out)


CALIBRATION_FILES: dict[str, str] = {"predictions": "calibration_numeric.csv.gz",
                                     "quantiles": "calibration_quantiles.csv.gz", "record": "calibration.json"}
"""dict: the files :func:`write_calibration` writes to the per-item folder."""


def write_calibration(result: AuditResult, out: Path) -> dict[str, Any] | None:
    """Write the numeric leave-two-out calibration predictions and per-test-scene conformal quantiles (per item).

    These are what the conformal step and the gate computed but the other outputs do not keep: for every test
    scene, the calibration items' predictions from models that saw neither the test scene's fold nor the item's
    own fold, and the quantiles that give the test scene's answers their ``half*``. Floats are written at full
    precision (shortest round-trip representation), so reading them back reproduces the gate's calibration
    scores exactly; gzip with a fixed header time, so equal content gives equal bytes. Nothing is written when
    nothing numeric was modelled. The other outputs are not touched.

    Parameters
    ----------
    result : AuditResult
        The audit (its ``calibration`` and ``quantiles``).
    out : pathlib.Path
        Per-item output folder (exists).

    Returns
    -------
    dict or None
        The record written to ``calibration.json`` (files, columns, row counts, the rule), or None if nothing
        was written.
    """
    if not len(result.calibration):
        return None
    gzip = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    result.calibration.to_csv(out / CALIBRATION_FILES["predictions"], index=False, compression=gzip)
    result.quantiles.to_csv(out / CALIBRATION_FILES["quantiles"], index=False, compression=gzip)
    record = {
        "files": CALIBRATION_FILES,
        "predictions": {"columns": list(result.calibration.columns), "n_rows": int(len(result.calibration)),
                        "kinds": sorted(result.calibration["kind"].astype(str).unique()),
                        "rule": "one row per numeric calibration item and test scene: the leave-two-out prediction "
                                "(measured_space.audit.cross_fit) of a model fitted without the test scene's fold "
                                "and the item's own fold; row indexes pairs.csv (0-based); mu, sigma, sigma_g in m"},
        "quantiles": {"columns": list(result.quantiles.columns), "n_rows": int(len(result.quantiles)),
                      "rule": "per dataset, kind, test scene and Mondrian cell (question type, depth source): the "
                              "split-conformal quantile of |e - mu| / sigma over that test scene's calibration "
                              "items (measured_space.audit.conformal_quantile_table); cells with fewer than "
                              "min_cal scores use the question type pooled over depth sources (cal_cell "
                              "'pooled'); an answer's half<L> is q<L> * sigma"},
        "float_format": "shortest round-trip (full float64 precision)"}
    (out / CALIBRATION_FILES["record"]).write_text(dumps(record))
    return record


def cross_fit_record(result: AuditResult, config: AuditConfig) -> dict[str, Any]:
    """The cross-fitting settings record for a run's summary.

    Parameters
    ----------
    result : AuditResult
        The audit (its ``folds`` table lists the groups whose scenes were grouped, if any).
    config : AuditConfig
        Its settings.

    Returns
    -------
    dict
        ``max_folds``, ``seed`` (it breaks the fold assignment's ties), ``rule`` and ``grouped``:
        ``{"<dataset>/<qtype>": {"n_scenes", "n_folds"}}`` for every group cross-fitted on grouped folds (the
        assignment is in ``folds.csv``); every other modelled group left each scene out on its own.
    """
    folds = result.tables.get("folds", pd.DataFrame(columns=["dataset", "qtype", "scene", "fold"]))
    grouped = {f"{dataset}/{qtype}": {"n_scenes": int(g["scene"].nunique()), "n_folds": int(g["fold"].nunique())}
               for (dataset, qtype), g in folds.groupby(["dataset", "qtype"], sort=True)}
    return {"max_folds": config.max_folds, "seed": config.seed,
            "rule": "a dataset and question type with at most max_folds scenes is cross-fitted leave one scene "
                    "out (calibration: leave two out); with more, its scenes are grouped into max_folds folds "
                    "balanced by row count and the same scheme runs on folds (calibration scores for a test "
                    "scene come from scenes outside its fold, predicted without both folds)",
            "grouped": grouped}


PRIMARY_GATE = {"numeric": "sigma", "categorical": "logistic"}
"""dict: the model kind whose gate is primary, per answer kind (at ``tau_scale == 1``)."""


def _accepted_at(result: AuditResult, tau_scale: float) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Primary-gate decisions at one tolerance multiple, as ids per dataset and auto source.

    Parameters
    ----------
    result : AuditResult
        The audit (its ``gate`` holds per-item decisions).
    tau_scale : float
        Tolerance multiple of the numeric gate (categorical decisions do not depend on it).

    Returns
    -------
    decisions : pandas.DataFrame
        The primary-gate rows with ``id`` and ``auto_source`` added.
    accepted : dict
        ``{dataset: {"auto_<depth source>": [ids, sorted]}}``.
    counts : dict
        ``{dataset: {auto_source: {"n_labelled", "n_accepted"}}}``.
    """
    gate = result.gate
    primary = gate[(((gate["answer_kind"] == "numeric") & (gate["kind"] == PRIMARY_GATE["numeric"])
                     & (gate["tau_scale"] == tau_scale))
                    | ((gate["answer_kind"] == "categorical") & (gate["kind"] == PRIMARY_GATE["categorical"])))]
    ids = result.pairs.loc[primary["row"].to_numpy(), "id"].to_numpy()
    primary = primary.assign(id=ids, auto_source="auto_" + primary["depth_source"].astype(str))
    accepted: dict[str, dict[str, list[str]]] = {}
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for (dataset, source), g in primary.groupby(["dataset", "auto_source"], sort=True):
        acc = sorted(g.loc[g["decision"] == "accept", "id"].astype(str))
        accepted.setdefault(dataset, {})[source] = acc
        counts.setdefault(dataset, {})[source] = {"n_labelled": int(len(g)), "n_accepted": len(acc)}
    return primary, accepted, counts


def accepted_ids(result: AuditResult, config: AuditConfig,
                 exclusion: dict[str, Any] | None = None) -> dict[str, Any]:
    """Item ids the primary gate accepted, per dataset and auto source, with the settings and held-out results.

    The primary gate is the numeric feature model at the question's own tolerance (``tau_scale == 1``, the
    tolerance :func:`~measured_space.qa_schema.is_correct` uses) plus the categorical logistic model. The
    measured error rate among accepted labels on held-out scenes is attached, because the certificate assumes
    exchangeable scenes and can fail (``certified_ok`` False); a looser set at twice the tolerance is included
    for reference only, since its labels are certified against ``2 tau``, not against the scoring rule.

    Parameters
    ----------
    result : AuditResult
        The audit (its ``gate`` holds per-item decisions).
    config : AuditConfig
        Its settings.
    exclusion : dict, optional
        The ``--exclude-ambiguous`` record (:func:`measured_space.audit.exclusion_record`), stored under the
        settings' ``exclude_ambiguous``; omitted: the key is not written.

    Returns
    -------
    dict
        ``{"settings", "held_out", "counts", "accepted", "alternative_2tau"}``. ``accepted`` is ``{dataset:
        {"auto_<depth source>": [item ids, sorted]}}``. ``held_out`` lists, per dataset and answer kind, the
        accepted count, the error among accepted with its Wilson and scene-bootstrap 95 % intervals, and
        ``certified_ok``. Question types the audit skipped (too few scenes or rows) are never accepted.
    """
    _, accepted, counts = _accepted_at(result, 1.0)
    table = result.tables.get("gate", pd.DataFrame())
    held_out: list[dict[str, Any]] = []
    for tau_scale in (1.0, 2.0):
        rows = table[(table["tau_scale"] == tau_scale)
                     & (((table["answer_kind"] == "numeric") & (table["kind"] == PRIMARY_GATE["numeric"]))
                        | ((table["answer_kind"] == "categorical")
                           & (table["kind"] == PRIMARY_GATE["categorical"])))] if len(table) else table
        held_out += [{k: r[k] for k in ("dataset", "answer_kind", "tau_scale", "n_labelled", "err_all", "accept_n",
                                        "accept_yield", "accept_err", "accept_lo", "accept_hi", "accept_boot_lo",
                                        "accept_boot_hi", "certified_ok")} for _, r in rows.iterrows()]
    _, alternative, alt_counts = _accepted_at(result, 2.0)
    settings = {"tau_scale": 1.0, "numeric_model": PRIMARY_GATE["numeric"],
                "categorical_model": PRIMARY_GATE["categorical"], "epsilon": config.epsilon, "delta": config.delta,
                "kappa": config.kappa, "gate_level": config.gate_level, "gate_z": config.gate_z,
                "min_accept": config.min_accept, "min_accept_frac": config.min_accept_frac,
                "ridge_grid": list(config.ridge_grid),
                "rule": "numeric: accept if (|mu| + z sigma) / tau_hat <= lambda; categorical: accept if "
                        "1 - p(correct) <= lambda; lambda certified per held-out scene by fixed-sequence "
                        "binomial Learn-then-Test on the other scenes (P(wrong | accepted) <= epsilon at "
                        "confidence 1 - delta, items treated as independent and scenes as exchangeable)",
                "skipped": result.notes}
    if exclusion is not None:
        settings["exclude_ambiguous"] = exclusion
    return {"settings": settings, "held_out": held_out, "counts": counts, "accepted": accepted,
            "alternative_2tau": {"note": "numeric labels certified against twice the tolerance; not the scoring "
                                         "rule's tolerance", "counts": alt_counts, "accepted": alternative}}


def headline(result: AuditResult, config: AuditConfig) -> str:
    """Plain-text summary of the main results.

    Parameters
    ----------
    result : AuditResult
        The audit.
    config : AuditConfig
        Its settings.

    Returns
    -------
    str
        Tables of errors, coverage at 90 %, rank correlation, categorical scores and the gate.
    """
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 400)
    t, parts = result.tables, []
    err = t["errors"]
    parts.append("ERRORS (m) per dataset / type / depth source\n" + err[
        ["dataset", "qtype", "depth_source", "n_questions", "n_labelled", "miss_rate", "wrong_rate", "bias_median",
         "abs_p50", "abs_p90", "rel_abs_p50", "gross_rate"]].round(3).to_string(index=False))
    if "coverage" in t:
        cov = t["coverage"]
        cov = cov[(cov["by"] == "all")]
        parts.append("COVERAGE overall (Wilson 95 % and scene-bootstrap 95 %)\n" + cov[
            ["dataset", "method", "level", "n", "coverage", "lo", "hi", "boot_lo", "boot_hi", "scene_mean",
             "scene_min", "scene_max", "half_over_tau_p50"]].round(3).to_string(index=False))
    if "rank" in t:
        rank = t["rank"]
        parts.append("RANK rho(sigma, |e|) and held-out NLL (pooled over depth sources)\n" + rank[
            rank["depth_source"] == "all"][["dataset", "qtype", "kind", "n", "rho", "nll_mean",
                                            "sigma_p50"]].round(3).to_string(index=False))
    if "comparison" in t:
        parts.append("MODEL vs BASELINE: held-out NLL(baseline) - NLL(feature model), nats/label (scene bootstrap 95 %)\n"
                     + t["comparison"].round(3).to_string(index=False))
    if "categorical" in t:
        parts.append("CATEGORICAL p(correct)\n" + t["categorical"].round(3).to_string(index=False))
    if "gate" in t:
        gate = t["gate"]
        parts.append(f"GATE (epsilon={config.epsilon}, delta={config.delta})\n" + gate[
            ["dataset", "answer_kind", "kind", "tau_scale", "n_labelled", "err_all", "accept_yield", "accept_err",
             "accept_lo", "accept_hi", "flag_yield", "flag_err", "reject_yield", "reject_err",
             "n_scenes_none"]].round(3).to_string(index=False))
    if "aurc" in t:
        parts.append("RISK-YIELD area (lower is better), AUROC for wrong labels, and AURC(baseline) - AURC(model)\n"
                     + t["aurc"].round(4).to_string(index=False))
    if result.notes:
        parts.append("NOTES\n" + "\n".join(f"- {n}" for n in result.notes))
    return "\n\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Load labels, run the audit, write and print the results.

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
    fold_groups = read_fold_groups(args.fold_groups) if args.fold_groups is not None else None
    config = dataclasses.replace(AuditConfig(), epsilon=args.epsilon, delta=args.delta, kappa=args.kappa,
                                 ridge_grid=tuple(args.ridge_grid), n_boot=args.n_boot, n_jobs=args.n_jobs,
                                 max_folds=args.max_folds, fold_groups=fold_groups)
    out = args.out or data_root() / "outputs" / ("audit_exclude_ambiguous" if args.exclude_ambiguous else "audit")
    inputs: dict[str, Any] = {}
    if args.synthetic:
        pairs = synthetic_pairs(n_scenes=6, n_items=80, noise_scale=1.8, seed=3)
        out, results = out / "_synthetic", None
    else:
        label_root = args.labels or data_root() / "outputs" / "labels"
        ready = wait_for_labels(label_root, args.datasets, args.wait_minutes)
        missing = sorted(set(args.datasets) - set(ready))
        if missing:
            print(f"labels not finished (no summary.json) for {missing}; auditing {ready}", flush=True)
        if not ready:
            return 1
        tables = []
        for dataset in ready:
            folder = label_root / dataset
            tables.append(load_labels(folder, args.depth_sources))
            read = sorted(folder.glob("qa_*.jsonl"))
            if args.depth_sources:                         # record the files read, not the ones left out
                names = {"qa_oracle.jsonl", *(f"qa_auto_{s}.jsonl" for s in args.depth_sources)}
                read = [p for p in read if p.name in names]
            inputs[dataset] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in read}
            inputs[dataset]["n_auto_only"] = tables[-1].attrs.get("n_auto_only")
            print(f"{dataset}: {len(tables[-1])} (question, depth source) rows from {folder}", flush=True)
        pairs = pd.concat(tables, ignore_index=True)
        results = args.results
        if args.depth_sources:
            print(f"--depth-sources: auditing {sorted(args.depth_sources)} only", flush=True)
    if fold_groups is not None:
        unknown = sorted(set(fold_groups) - set(pairs["dataset"].unique()))
        if unknown:
            raise ValueError(f"--fold-groups {args.fold_groups} lists datasets that are not audited: {unknown}")
        described = [f"{d}: {len(set(m.values()))} groups of {len(m)} scenes" for d, m in sorted(fold_groups.items())]
        print(f"--fold-groups {args.fold_groups}: {'; '.join(described)}", flush=True)
    counts: dict[str, dict[str, dict[str, int]]] | None = None
    if args.exclude_ambiguous:
        pairs, counts = exclude_ambiguous(pairs)
    exclusion = exclusion_record(counts)
    if counts is not None:
        print(f"--exclude-ambiguous: dropped {exclusion['n_dropped']} pairs, kept {exclusion['n_unknown_kept']} "
              f"of unknown ambiguity: {dumps(exclusion['counts'])}", flush=True)
    result = run_audit(pairs, config, log=lambda m: print(m, flush=True))
    write_tables(result, out, results)
    record = dataclasses.asdict(config)
    if config.fold_groups is None:
        del record["fold_groups"]           # without --fold-groups the summary is exactly what it always was
    summary = {"config": record, "gate_z": config.gate_z, "datasets": sorted(pairs["dataset"].unique()),
               "n_rows": int(len(pairs)), "n_labelled": int(pairs["matched"].sum()), "inputs": inputs,
               "exclude_ambiguous": exclusion, "cross_fit": cross_fit_record(result, config),
               "notes": result.notes, "git_commit": git_commit(), "seconds": round(time.time() - started, 1)}
    if args.depth_sources:
        summary["depth_sources"] = {
            "selected": sorted(args.depth_sources),
            "rule": "only qa_oracle.jsonl and these sources' qa_auto_<source>.jsonl were read "
                    "(measured_space.audit.load_labels), so every fit, conformal cell and gate certificate uses "
                    "these sources' (question, depth source) rows only"}
    if args.fold_groups is not None:
        summary["fold_groups"] = {
            "path": str(args.fold_groups), "sha256": sha256(args.fold_groups),
            "rule": "for each dataset in config.fold_groups, every question type is cross-fitted with one fold per "
                    "group (measured_space.audit.scene_folds), ignoring max_folds: a group's scenes are predicted by "
                    "a model fitted without the group, and calibrated by the scenes of the other groups, each "
                    "predicted without both groups; the inner ridge search and the scene-cluster bootstrap stay "
                    "by scene; assignment in folds.csv"}
    for folder in [out] + ([results] if results is not None else []):
        (folder / "summary.json").write_text(dumps(summary))
    if len(result.gate):
        (out / "accepted_ids.json").write_text(dumps(accepted_ids(result, config, exclusion)))
    print(headline(result, config))
    print(f"\nwrote per-item tables to {out}" + (f" and aggregate tables to {results}" if results else "")
          + f" ({summary['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
