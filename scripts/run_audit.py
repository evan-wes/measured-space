"""Run the label audit on the generated oracle and auto labels, and write its tables.

For each dataset this reads ``<label root>/<dataset>/qa_oracle.jsonl`` and every ``qa_auto_<depth source>.jsonl``
next to it (written by the label pipeline), and runs :func:`measured_space.audit.run_audit`: errors per question
type and depth source, the structured error model with leave-one-scene-out predictions, Mondrian conformal
calibration, categorical p(correct), and the Learn-then-Test label gate.

Outputs:

- ``<data root>/outputs/audit/`` (or ``--out``; with ``--exclude-ambiguous`` the default is
  ``<data root>/outputs/audit_exclude_ambiguous/``): per-item tables (``pairs.csv``, ``numeric.csv``,
  ``categorical.csv``, ``gate.csv``) that the figures are drawn from, a copy of every aggregate table, and
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

Usage::

    python scripts/run_audit.py --results results/audit                       # both datasets, default settings
    python scripts/run_audit.py --results results/audit --wait-minutes 120    # wait for the labels' summary.json
    python scripts/run_audit.py --results <scratch folder> --exclude-ambiguous --labels <label root>
    python scripts/run_audit.py --synthetic       # dry run on synthetic data (writes under --out only)
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
                             "(one value: fixed)")
    parser.add_argument("--n-jobs", type=int, default=4, help="worker processes for cross-fitting")
    parser.add_argument("--n-boot", type=int, default=AuditConfig.n_boot, help="scene-cluster bootstrap replicates")
    parser.add_argument("--wait-minutes", type=float, default=0.0,
                        help="poll for each dataset's summary.json for up to this long (0: do not wait)")
    parser.add_argument("--synthetic", action="store_true",
                        help="audit synthetic data instead (outputs under <out>/_synthetic, results/ untouched)")
    args = parser.parse_args(argv)
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
    config = dataclasses.replace(AuditConfig(), epsilon=args.epsilon, delta=args.delta, kappa=args.kappa,
                                 ridge_grid=tuple(args.ridge_grid), n_boot=args.n_boot, n_jobs=args.n_jobs)
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
            tables.append(load_labels(folder))
            inputs[dataset] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                               for p in sorted(folder.glob("qa_*.jsonl"))}
            inputs[dataset]["n_auto_only"] = tables[-1].attrs.get("n_auto_only")
            print(f"{dataset}: {len(tables[-1])} (question, depth source) rows from {folder}", flush=True)
        pairs = pd.concat(tables, ignore_index=True)
        results = args.results
    counts: dict[str, dict[str, dict[str, int]]] | None = None
    if args.exclude_ambiguous:
        pairs, counts = exclude_ambiguous(pairs)
    exclusion = exclusion_record(counts)
    if counts is not None:
        print(f"--exclude-ambiguous: dropped {exclusion['n_dropped']} pairs, kept {exclusion['n_unknown_kept']} "
              f"of unknown ambiguity: {dumps(exclusion['counts'])}", flush=True)
    result = run_audit(pairs, config, log=lambda m: print(m, flush=True))
    write_tables(result, out, results)
    summary = {"config": dataclasses.asdict(config), "gate_z": config.gate_z, "datasets": sorted(pairs["dataset"].unique()),
               "n_rows": int(len(pairs)), "n_labelled": int(pairs["matched"].sum()), "inputs": inputs,
               "exclude_ambiguous": exclusion,
               "notes": result.notes, "git_commit": git_commit(), "seconds": round(time.time() - started, 1)}
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
