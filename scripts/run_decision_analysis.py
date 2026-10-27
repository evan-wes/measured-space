"""Certify decisions, not measurements: gate each labeller's metric answers by known decisions.

For every labeller (one audit folder written by ``scripts/run_audit.py``, read only, and one depth source of it),
this reads the audit's per-item tables and runs :mod:`measured_space.decisions`:

- **Part A, one answer against a threshold.** "Is the reference above ``t``?" for the thresholds fixed in advance
  per dataset and question type (:data:`measured_space.decisions.DEFAULT_THRESHOLDS`, or ``--thresholds``). The
  call uses the bias-corrected answer ``c = value - mu``; an answer is accepted when ``r = half90 / |c - t|`` is at
  most a threshold certified by Learn-then-Test per test scene on other folds' decisions (``P(wrong decision |
  accepted) <= epsilon`` at confidence ``1 - delta``), and compared with the uncertified "interval excludes ``t``"
  rule (``r <= 1``), accepting everything, and today's fixed-tolerance gate on the same decisions. Also a sweep
  of thresholds placed relative to the reference, ``t = ref (1 + u)`` (an analysis device: ``t`` depends on the
  reference), and one-sided clearance decisions for object-to-object distances.
- **Part B, margins between two answers.** Pairs of size or camera-distance answers in one view about different
  objects (referents from the label folder the audit read, found by its recorded checksums): coverage and width
  of three 90 % error bars of the margin, and the "A greater than B" and "A smaller than B by at least 0.1 m"
  decisions gated with the directly calibrated one.

Outputs (aggregates only) in ``--results``, which must not exist or must be empty (nothing is ever overwritten):
``decision_gate.csv``, ``sweep.csv``, ``safety.csv``, ``margin_intervals.csv``, ``margin_gate.csv``,
``summary.json`` (inputs with checksums, thresholds, settings, assumptions, notes, git commit, seconds) and the
figure ``decision_sweep.pdf`` / ``.png`` (certified accepted share against ``|u|`` per labeller, handheld and
warehouse panels, with the fixed-tolerance gate's share dashed).

Usage (from the repository root)::

    python scripts/run_decision_analysis.py --results <new folder>              # the six default labellers
    python scripts/run_decision_analysis.py --results <new folder> \\
        --audits cutr_sam3=audit_ark106_cutr_sam3 --audits sparse4d_v22=audit_ss12_sparse4d_v22
    python scripts/run_decision_analysis.py --results <new folder> --audits 2d_lidar=audit_ark106_2d:lidar

``--audits NAME=FOLDER[:SOURCE]``: ``FOLDER`` relative to ``<data root>/outputs`` unless absolute; ``SOURCE`` is
the depth source to keep (required when the audit has several).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from make_figures import (AQUA, BLUE, DATASET_NAME, INK, MUTED, ORANGE, WIDTH, plain_log_axis, save,  # noqa: E402
                          style)  # scripts/ is on sys.path when run
from measured_space import decisions as dec  # noqa: E402
from measured_space.audit import dumps, ltt_threshold  # noqa: E402
from run_audit import git_commit  # noqa: E402  (scripts/ is on sys.path when run)

DEFAULT_AUDITS: tuple[str, ...] = (
    "2d_lidar=audit_ark106_2d:lidar", "cutr_sam3=audit_ark106_cutr_sam3", "cutr_clip=audit_ark106_cutr_clip",
    "2d_moge=audit_ss12_2d:moge", "sparse4d_v22=audit_ss12_sparse4d_v22", "sparse4d_v11=audit_ss12_sparse4d_v11")
"""tuple of str: the labellers analysed by default (``NAME=FOLDER[:SOURCE]``, folders under ``<data root>/outputs``)."""

OUTPUT_TABLES: tuple[str, ...] = ("decision_gate", "sweep", "safety", "margin_intervals", "margin_gate")
"""tuple of str: the CSV tables written to ``--results``."""

FIGURE = "decision_sweep"
"""str: file stem of the figure."""

LABELLER_STYLE: tuple[tuple[str, str], ...] = ((BLUE, "o"), (ORANGE, "^"), (AQUA, "s"), (INK, "v"), (MUTED, "P"))
"""tuple: colour and marker of the first, second, ... labeller of a panel."""

ASSUMPTIONS: tuple[str, ...] = (
    "Answers and error bars are the audit's out-of-scene predictions of the feature model (numeric.csv, kind "
    "sigma): mu is the predicted bias of e = value - reference, half90 the half-width of the split-conformal 90 % "
    "interval [mu - half90, mu + half90] for e, so the interval for the reference is [c - half90, c + half90] with "
    "c = value - mu. Nothing is refitted.",
    "Truth is the reference value value - e. A decision with c exactly at the threshold (infinite score) is never "
    "accepted.",
    "Calibration decisions for a test scene come from scenes outside its fold (per question type, the audit's "
    "folds, recomputed and checked against folds.csv where recorded). They are those scenes' own out-of-scene test "
    "predictions: the audit did not save its leave-two-out calibration predictions, so a calibration decision's mu "
    "and sigma come from a model whose training data included the test scene's fold, and its half90 from a "
    "conformal quantile whose calibration scores included that fold. The test scene's own answers never calibrate "
    "it, but the Learn-then-Test certificate is approximate in this respect.",
    "Learn-then-Test is the audit's procedure and settings (fixed-sequence binomial tests on a geometric grid; "
    "min_accept 30, min_accept_frac 0.05). Items are treated as independent and scenes as exchangeable, as in the "
    "audit; Wilson intervals and binomial bounds are optimistic for items that share a scene, so the scene-cluster "
    "bootstrap interval is reported too.",
    "Calibration pool 'labeller' (default) pools all of a labeller's decisions of one analysis (question types and "
    "thresholds; sweep: per |u|, both signs; margins: per decision), as the audit's gate pools question types: the "
    "certificate is on the pooled wrong-decision rate, and a per-type or per-threshold row may exceed epsilon. "
    "Pool 'family' calibrates each question type and threshold separately.",
    "The fixed-tolerance gate is taken from gate.csv as the audit ran it (kind sigma, tau_scale 1, decision "
    "accept); in an audit with several depth sources its threshold was certified on all sources together.",
    "The sweep places t = ref (1 + u) relative to the reference: an analysis device (t depends on the unknown "
    "reference), not a deployable decision. Its truth is u < 0; answers with ref <= 0 are left out.",
    "Safety decisions are one-sided: only outside calls (c > t) can be accepted, with score half90 / (c - t); an "
    "accepted call is wrong when ref <= t. The no-check baseline accepts every outside call.",
    "Margins: unordered pairs of answers of one question type in one view (dataset, scene, camera, frame) about two "
    "different single-referent objects (referents from the label folder's qa_oracle.jsonl, found by the audit's "
    "recorded checksums), oriented by question id (hashes: unrelated to the values). Direct calibration is split "
    "conformal on |e_m - (mu_A - mu_B)| / sqrt(sigma_A^2 + sigma_B^2) over pairs of the same question type outside "
    "the test scene's fold, with the same calibration caveat as above.",
)
"""tuple of str: assumptions recorded in ``summary.json``."""


@dataclass(frozen=True)
class AuditSpec:
    """One labeller: an audit folder and the depth source to keep.

    Attributes
    ----------
    name : str
        Labeller name in the outputs.
    folder : pathlib.Path
        The audit's per-item folder (``pairs.csv``, ``numeric.csv``, ``gate.csv``, ``summary.json``).
    source : str or None
        Depth source to keep; None when the audit has one.
    """

    name: str
    folder: Path
    source: str | None


def parse_audit(text: str, root: Path | None = None) -> AuditSpec:
    """Parse ``NAME=FOLDER[:SOURCE]``.

    Parameters
    ----------
    text : str
        The specification.
    root : pathlib.Path, optional
        Folder that relative audit folders are under (``<data root>/outputs``); None leaves them relative.

    Returns
    -------
    AuditSpec
        The labeller.

    Raises
    ------
    ValueError
        If the text has no ``=``, or an empty name or folder.
    """
    name, sep, rest = text.partition("=")
    if not sep or not name or not rest:
        raise ValueError(f"expected NAME=FOLDER[:SOURCE], got {text!r}")
    folder_text, source = rest, None
    head, colon, tail = rest.rpartition(":")
    if colon and head and tail and "/" not in tail:
        folder_text, source = head, tail
    folder = Path(folder_text).expanduser()
    if not folder.is_absolute() and root is not None:
        folder = root / folder
    return AuditSpec(name, folder, source)


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
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--audits", action="append", default=None, metavar="NAME=FOLDER[:SOURCE]",
                   help="labeller to analyse (repeatable); FOLDER relative to <data root>/outputs unless absolute "
                        f"(default: {' '.join(DEFAULT_AUDITS)})")
    p.add_argument("--labels", action="append", default=[], metavar="NAME=FOLDER",
                   help="label folder of a labeller for part B (the folder holding <dataset>/qa_oracle.jsonl); "
                        "default: labels_<suffix> next to audit_<suffix>, accepted only if its checksums match the "
                        "audit's summary.json")
    p.add_argument("--results", type=Path, required=True, help="output folder (new or empty; no default)")
    p.add_argument("--epsilon", type=float, nargs="+", default=list(dec.DecisionConfig.epsilons),
                   help="certified wrong-decision bounds; the first is primary (default %(default)s)")
    p.add_argument("--delta", type=float, default=dec.DecisionConfig.delta, help="certificate failure probability")
    p.add_argument("--pool", choices=dec.POOLS, default=dec.DecisionConfig.pool,
                   help="Learn-then-Test calibration pool: all of a labeller's decisions (as the audit's gate) or "
                        "one question type and threshold (default %(default)s)")
    p.add_argument("--thresholds", type=Path, default=None,
                   help="JSON file {dataset: {qtype: [thresholds in m]}} replacing the default thresholds")
    p.add_argument("--n-boot", type=int, default=dec.DecisionConfig.n_boot, help="scene-cluster bootstrap replicates")
    p.add_argument("--seed", type=int, default=dec.DecisionConfig.seed, help="bootstrap seed")
    p.add_argument("--skip-margins", action="store_true", help="skip part B (margins between two answers)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------------------------
# Safety and provenance
# --------------------------------------------------------------------------------------------------------


def check_results_folder(results: Path, inputs: Sequence[Path]) -> None:
    """Refuse a results folder that holds anything or overlaps an input folder.

    Parameters
    ----------
    results : pathlib.Path
        The output folder.
    inputs : sequence of pathlib.Path
        Input folders (audit and label folders).

    Raises
    ------
    FileExistsError
        If ``results`` exists and is not an empty folder.
    ValueError
        If ``results`` is an input folder, lies inside one, or contains one.
    """
    out = Path(results).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"{out} exists and is not an empty folder; choose a new one")
    for folder in (Path(p).resolve() for p in inputs):
        if out == folder or folder in out.parents or out in folder.parents:
            raise ValueError(f"results folder {out} overlaps input folder {folder}")


def file_record(path: Path) -> dict[str, Any]:
    """Size and checksum of an input file.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    dict
        ``{"bytes": int, "sha256": str}``.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def find_label_folder(audit: AuditSpec, summary: Mapping[str, Any], override: Path | None) -> tuple[Path | None, str]:
    """The label folder an audit read, verified by the checksums its ``summary.json`` recorded.

    Parameters
    ----------
    audit : AuditSpec
        The labeller.
    summary : mapping
        The audit's ``summary.json`` (``inputs: {dataset: {file name: {bytes, sha256}}}``).
    override : pathlib.Path or None
        A label root to try instead of ``labels_<suffix>`` next to ``audit_<suffix>``.

    Returns
    -------
    folder : pathlib.Path or None
        ``<label root>/<dataset>`` if every recorded file is there with its recorded checksum, else None.
    note : str
        What was checked, or why no folder was accepted.
    """
    inputs = summary.get("inputs", {})
    if len(inputs) != 1:
        return None, f"summary.json lists {len(inputs)} datasets; expected one"
    dataset, files = next(iter(inputs.items()))
    if override is not None:
        root = override
    else:
        suffix = audit.folder.name.removeprefix("audit_")
        root = audit.folder.parent / f"labels_{suffix}"
    folder = root / dataset if (root / dataset).is_dir() else root
    recorded = {k: v for k, v in files.items() if isinstance(v, Mapping) and "sha256" in v}
    if not recorded:
        return None, "summary.json records no input checksums"
    for name, record in sorted(recorded.items()):
        path = folder / name
        if not path.is_file():
            return None, f"{path} not found"
        if file_record(path)["sha256"] != record["sha256"]:
            return None, f"{path}: checksum differs from the audit's record"
    return folder, f"{len(recorded)} files match the audit's recorded sha256: {sorted(recorded)}"


def read_referents(path: Path, ids: set[str]) -> dict[str, tuple[str, ...]]:
    """Referent object ids of the wanted questions, from a question-record file.

    Parameters
    ----------
    path : pathlib.Path
        A ``qa_*.jsonl`` file (one record per line with ``id`` and ``referents[].object_id``).
    ids : set of str
        Question ids to keep.

    Returns
    -------
    dict of str to tuple of str
        Question id to its referents' object ids, in record order.
    """
    out: dict[str, tuple[str, ...]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            qid = str(record.get("id"))
            if qid in ids:
                out[qid] = tuple(str(r["object_id"]) for r in record.get("referents", []))
    return out


def load_thresholds(path: Path | None) -> dict[str, dict[str, tuple[float, ...]]]:
    """The decision thresholds: the defaults, or a JSON file's.

    Parameters
    ----------
    path : pathlib.Path or None
        JSON file ``{dataset: {qtype: [thresholds in m]}}``; None gives
        :data:`measured_space.decisions.DEFAULT_THRESHOLDS`.

    Returns
    -------
    dict
        ``{dataset: {qtype: tuple of float}}``.
    """
    if path is None:
        return {d: dict(q) for d, q in dec.DEFAULT_THRESHOLDS.items()}
    raw = json.loads(path.read_text())
    return {str(d): {str(q): tuple(float(t) for t in ts) for q, ts in qs.items()} for d, qs in raw.items()}


# --------------------------------------------------------------------------------------------------------
# One labeller
# --------------------------------------------------------------------------------------------------------


def _with_all(frame: pd.DataFrame, **pooled: Any) -> pd.DataFrame:
    """The frame twice: as is, and with grouping columns overwritten to form pooled groups.

    Parameters
    ----------
    frame : pandas.DataFrame
        Decisions or pairs.
    **pooled : Any
        Columns to overwrite in the pooled copy (e.g. ``qtype="all"``).

    Returns
    -------
    pandas.DataFrame
        ``frame`` followed by the pooled copy.
    """
    return pd.concat([frame, frame.assign(**pooled)], ignore_index=True)


def _label(frame: pd.DataFrame, audit: AuditSpec, dataset: str, source: str) -> pd.DataFrame:
    """Prepend the labeller's identity columns to a table.

    Parameters
    ----------
    frame : pandas.DataFrame
        A report table.
    audit : AuditSpec
        The labeller.
    dataset : str
        Its dataset.
    source : str
        Its depth source.

    Returns
    -------
    pandas.DataFrame
        ``labeller, setting, dataset, source`` followed by ``frame``'s columns.
    """
    ident = pd.DataFrame({"labeller": audit.name, "setting": dec.SETTING.get(dataset, dataset), "dataset": dataset,
                          "source": source}, index=frame.index)
    return pd.concat([ident, frame], axis=1)


def read_audit(audit: AuditSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None,
                                          dict[str, Any], dict[str, Any]]:
    """Read an audit's per-item tables, fold record and summary.

    Parameters
    ----------
    audit : AuditSpec
        The labeller.

    Returns
    -------
    pairs, numeric, gate : pandas.DataFrame
        ``pairs.csv``, ``numeric.csv`` (needed columns) and ``gate.csv`` (numeric rows of the needed columns).
    folds : pandas.DataFrame or None
        ``folds.csv`` (from ``tables/``, ``results/`` or the folder itself), None if the audit wrote none.
    summary : dict
        ``summary.json``.
    files : dict
        Input file name to ``{"bytes", "sha256"}``.

    Raises
    ------
    FileNotFoundError
        If a required file is missing.
    """
    folder = audit.folder
    str_cols = {"id": str, "scene": str, "camera_id": str, "frame": str}
    pairs = pd.read_csv(folder / "pairs.csv", usecols=["id", "dataset", "scene", "camera_id", "frame", "qtype",
                                                       "value", "e"], dtype=str_cols)
    numeric = pd.read_csv(folder / "numeric.csv", dtype={"scene": str},
                          usecols=["row", "kind", "dataset", "scene", "qtype", "depth_source", "e", "mu", "sigma",
                                   "half90"])
    gate = pd.read_csv(folder / "gate.csv", usecols=["row", "kind", "tau_scale", "answer_kind", "decision"])
    gate = gate[gate["answer_kind"] == "numeric"]
    summary = json.loads((folder / "summary.json").read_text())
    folds = None
    names = ["pairs.csv", "numeric.csv", "gate.csv", "summary.json"]
    for candidate in ("tables/folds.csv", "results/folds.csv", "folds.csv"):
        if (folder / candidate).is_file():
            folds = pd.read_csv(folder / candidate, dtype={"scene": str})
            names.append(candidate)
            break
    files = {name: file_record(folder / name) for name in names}
    return pairs, numeric, gate, folds, summary, files


def analyse_labeller(audit: AuditSpec, thresholds: Mapping[str, Mapping[str, Sequence[float]]],
                     config: dec.DecisionConfig, *, skip_margins: bool, label_override: Path | None,
                     log: Callable[[str], None] = print) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Run every analysis on one labeller.

    Parameters
    ----------
    audit : AuditSpec
        The labeller.
    thresholds : mapping
        ``{dataset: {qtype: thresholds in m}}``.
    config : DecisionConfig
        Settings.
    skip_margins : bool
        Skip part B.
    label_override : pathlib.Path or None
        Label root for part B instead of the default (see :func:`find_label_folder`).
    log : callable, default print
        Receives progress messages.

    Returns
    -------
    tables : dict of str to pandas.DataFrame
        The report tables of :data:`OUTPUT_TABLES` (margin tables empty when part B is skipped).
    record : dict
        Inputs, counts and notes for ``summary.json``.

    Raises
    ------
    ValueError
        If the audit holds more than one dataset.
    """
    started = time.time()
    pairs, numeric, gate, folds, summary, files = read_audit(audit)
    audit_cfg = summary.get("config", {})
    max_folds, seed = audit_cfg.get("max_folds", 20), audit_cfg.get("seed", 0)
    answers, checked = dec.answer_table(pairs, numeric, gate, source=audit.source, level=config.level,
                                        max_folds=max_folds, seed=seed, recorded_folds=folds)
    datasets = sorted(answers["dataset"].unique())
    if len(datasets) != 1:
        raise ValueError(f"{audit.name}: expected one dataset, got {datasets}")
    dataset, source = datasets[0], str(answers["depth_source"].iloc[0])
    fold_of = dec.fold_lookup(answers)
    boot = config.bootstrap()
    family = config.pool == "family"
    notes: list[str] = []
    missing_thr = sorted(set(answers["qtype"]) - set(thresholds.get(dataset, {})))
    if missing_thr:
        notes.append(f"no thresholds for {missing_thr}: left out of part A")
    log(f"{audit.name}: {len(answers)} answers ({dataset}, source {source}) from {audit.folder}")

    # Part A: fixed thresholds
    rules = dec.standard_rules(config.epsilons)
    a = dec.threshold_decisions(answers, thresholds)
    a, _ = dec.apply_rules(a, fold_of, config, ["qtype", "threshold"] if family else [])
    gate_table = dec.rule_table(_with_all(a, qtype="all", threshold=np.nan), ["qtype", "threshold"], rules, boot)

    # Sweep: thresholds relative to the reference
    sw = dec.sweep_decisions(answers)
    sw, _ = dec.apply_rules(sw, fold_of, config, ["qtype", "u_abs"] if family else ["u_abs"])
    sw_all = _with_all(sw, qtype="all")
    sweep_table = pd.concat([dec.rule_table(sw_all, ["qtype", "u_abs", "u"], rules, boot),
                             dec.rule_table(sw_all.assign(u=np.nan), ["qtype", "u_abs", "u"], rules, boot)],
                            ignore_index=True)

    # Safety: one-sided clearance decisions
    sd = dec.safety_decisions(answers, thresholds)
    safety_table = pd.DataFrame()
    if len(sd):
        sd, _ = dec.apply_rules(sd, fold_of, config, ["qtype", "threshold"] if family else [], one_sided=True)
        safety_table = dec.rule_table(_with_all(sd, qtype="all", threshold=np.nan), ["qtype", "threshold"],
                                      dec.standard_rules(config.epsilons, accept_all="accept_outside"), boot,
                                      inside="inside")
    else:
        notes.append("no safety decisions (no distance answers with thresholds)")

    # Part B: margins between two answers
    record_b: dict[str, Any] = {"skipped": True}
    margin_intervals, margin_gate = pd.DataFrame(), pd.DataFrame()
    if not skip_margins:
        label_folder, label_note = find_label_folder(audit, summary, label_override)
        record_b = {"skipped": label_folder is None, "label_folder": str(label_folder) if label_folder else None,
                    "check": label_note}
        if label_folder is None:
            notes.append(f"part B skipped: referents not recoverable ({label_note})")
        else:
            ids = set(answers.loc[answers["qtype"].isin(dec.MARGIN_QTYPES), "id"])
            referents = read_referents(label_folder / "qa_oracle.jsonl", ids)
            pairs_m, counts = dec.margin_pairs(answers, referents)
            record_b["counts"] = counts
            record_b["referent_file"] = str(label_folder / "qa_oracle.jsonl")
            record_b["n_ids_found"] = len(referents)
            if counts["n_no_referents"]:
                notes.append(f"part B: {counts['n_no_referents']} answers had no question record")
            if len(pairs_m):
                pairs_m = dec.direct_margin_halfwidth(pairs_m, config.level)
                margin_intervals = dec.margin_interval_table(_with_all(pairs_m, qtype="all"), ["qtype"], boot)
                md = dec.margin_decisions(pairs_m, config.fit_margin)
                md, _ = dec.apply_rules(md, fold_of, config, ["decision", "qtype"] if family else ["decision"])
                margin_gate = dec.rule_table(_with_all(md, qtype="all"), ["decision", "qtype"], rules, boot)
            else:
                notes.append("part B: no pairs")

    tables = {"decision_gate": gate_table, "sweep": sweep_table, "safety": safety_table,
              "margin_intervals": margin_intervals, "margin_gate": margin_gate}
    tables = {k: _label(v, audit, dataset, source) if len(v) else v for k, v in tables.items()}
    record = {"folder": str(audit.folder), "source": source, "dataset": dataset,
              "setting": dec.SETTING.get(dataset, dataset), "files": files,
              "audit_git_commit": summary.get("git_commit"),
              "audit_config": {k: audit_cfg.get(k) for k in ("max_folds", "seed", "epsilon", "delta", "min_accept",
                                                             "min_accept_frac")},
              "folds": {"recorded": folds is not None, "checked_against_record": checked,
                        "rule": "recomputed with measured_space.audit.scene_folds over the audit's modelled rows of "
                                "every depth source; groups listed were checked against folds.csv, the others "
                                "have at most max_folds scenes (one scene per fold)"},
              "n_answers": {str(k): int(v) for k, v in answers.groupby("qtype").size().items()},
              "n_scenes": int(answers["scene"].nunique()),
              "n_decisions": {"threshold": int(len(a)), "sweep": int(len(sw)), "safety": int(len(sd))},
              "part_b": record_b, "notes": notes, "seconds": round(time.time() - started, 1)}
    return tables, record


# --------------------------------------------------------------------------------------------------------
# Figure and headline
# --------------------------------------------------------------------------------------------------------


def draw_sweep(sweep: pd.DataFrame, epsilon: float, folder: Path) -> list[Path]:
    """Certified accepted share against ``|u|`` per labeller, with the fixed-tolerance gate's share dashed.

    Parameters
    ----------
    sweep : pandas.DataFrame
        The ``sweep`` table (rows with ``qtype == "all"`` and no sign, ``u`` NaN, are drawn).
    epsilon : float
        The certified bound whose rule is drawn.
    folder : pathlib.Path
        Output folder.

    Returns
    -------
    list of pathlib.Path
        The PDF and PNG written.
    """
    style()
    pooled = sweep[(sweep["qtype"] == "all") & sweep["u"].isna()]
    settings = [s for s in ("handheld", "warehouse") if s in set(pooled["setting"])]
    settings += sorted(set(pooled["setting"]) - set(settings))
    fig, axes = plt.subplots(1, max(len(settings), 1), figsize=(WIDTH, 2.1), sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0.12})
    columns: list[list[Line2D]] = []                    # one legend column per panel, under its panel
    for ax, setting in zip(axes[0], settings):
        sub = pooled[pooled["setting"] == setting]
        dataset = str(sub["dataset"].iloc[0])
        column: list[Line2D] = []
        for i, name in enumerate(dict.fromkeys(sub["labeller"])):
            colour, marker = LABELLER_STYLE[i % len(LABELLER_STYLE)]
            mine = sub[sub["labeller"] == name]
            cert = mine[(mine["rule"] == "certified") & np.isclose(mine["epsilon"], epsilon)].sort_values("u_abs")
            fixed = mine[mine["rule"] == "fixed_tau_gate"].sort_values("u_abs")
            (line,) = ax.plot(cert["u_abs"], cert["accept_share"], color=colour, marker=marker, label=name)
            ax.plot(fixed["u_abs"], fixed["accept_share"], color=colour, linestyle="--", linewidth=0.8)
            column.append(line)
        columns.append(column)
        ax.set_xscale("log")
        plain_log_axis(ax, "x")
        ax.set_title(f"{setting.capitalize()} ({DATASET_NAME.get(dataset, dataset)})")
        ax.set_xlabel("|u|, with t = ref (1 + u)")
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("Accepted share")
    key = [Line2D([], [], color=MUTED, marker="o", label=f"certified, ε = {epsilon:g}"),
           Line2D([], [], color=MUTED, linestyle="--", linewidth=0.8, label="fixed-τ gate")]
    depth = max((len(c) for c in columns), default=0)
    blank = Line2D([], [], alpha=0.0, label=" ")
    columns = [c + [blank] * (depth - len(c)) for c in columns] or [[]]
    if len(columns) == 1:
        columns[0] += key
    else:
        columns[0].append(key[0])
        columns[1].append(key[1])
        for c in columns[2:]:
            c.append(blank)
    handles = [h for c in columns for h in c]
    fig.canvas.draw()                                   # below the panels' labels, as make_figures.shared_legend
    renderer = fig.canvas.get_renderer()                # type: ignore[attr-defined]
    bottom = min(a.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0 for a in fig.axes)
    fig.legend(handles=handles, labels=[h.get_label() for h in handles], loc="upper center",
               bbox_to_anchor=(0.5, bottom - 0.01), ncol=len(columns), handlelength=1.6, columnspacing=1.5)
    return save(fig, FIGURE, [folder])


def headline(tables: Mapping[str, pd.DataFrame], epsilons: Sequence[float]) -> list[dict[str, Any]]:
    """Per labeller: the certified rule against the fixed-tolerance gate on the fixed-threshold decisions.

    Parameters
    ----------
    tables : mapping of str to pandas.DataFrame
        The report tables.
    epsilons : sequence of float
        The certified bounds.

    Returns
    -------
    list of dict
        One record per labeller (pooled over question types and thresholds): decisions, and for the certified
        rule at each epsilon (``certified_eps<value>``), the interval rule, accepting everything and the fixed
        gate: accepted share and wrong-decision rate.
    """
    table = tables.get("decision_gate", pd.DataFrame())
    if not len(table):
        return []
    pooled = table[table["qtype"] == "all"]
    out = []
    for name, g in pooled.groupby("labeller", sort=False):
        row: dict[str, Any] = {"labeller": name, "setting": g["setting"].iloc[0], "n_decisions": int(g["n"].iloc[0])}
        rules = [(f"certified_eps{eps:g}", g[(g["rule"] == "certified") & np.isclose(g["epsilon"], eps)])
                 for eps in epsilons]
        rules += [(rule, g[g["rule"] == rule]) for rule in ("interval", "accept_all", "fixed_tau_gate")]
        for label, r in rules:
            if len(r):
                row[f"{label}_accept_share"] = float(r["accept_share"].iloc[0])
                row[f"{label}_wrong_rate"] = float(r["wrong_rate"].iloc[0])
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the decision analysis on every labeller and write its tables, summary and figure.

    Parameters
    ----------
    argv : list of str or None, optional
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status: 0 on success.

    Raises
    ------
    ValueError
        If a labeller name repeats or a ``--labels`` entry names no labeller.
    """
    args = parse_args(argv)
    started = time.time()
    specs = args.audits or list(DEFAULT_AUDITS)
    audits = [parse_audit(s) for s in specs]
    if any(not a.folder.is_absolute() for a in audits):
        from measured_space.config import data_root
        audits = [parse_audit(s, data_root() / "outputs") for s in specs]
    names = [a.name for a in audits]
    if len(set(names)) != len(names):
        raise ValueError(f"labeller names repeat: {names}")
    overrides = {}
    for text in args.labels:
        name, _, folder = text.partition("=")
        if name not in names:
            raise ValueError(f"--labels {text!r} names no labeller of {names}")
        overrides[name] = Path(folder).expanduser()
    check_results_folder(args.results, [a.folder for a in audits] + list(overrides.values()))
    config = dec.DecisionConfig(epsilons=tuple(args.epsilon), delta=args.delta, pool=args.pool, n_boot=args.n_boot,
                                seed=args.seed)
    thresholds = load_thresholds(args.thresholds)

    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in OUTPUT_TABLES}
    records: dict[str, Any] = {}
    for audit in audits:
        tables, record = analyse_labeller(audit, thresholds, config, skip_margins=args.skip_margins,
                                          label_override=overrides.get(audit.name))
        for name, frame in tables.items():
            if len(frame):
                parts[name].append(frame)
        records[audit.name] = record
        print(f"{audit.name}: done in {record['seconds']} s; notes: {record['notes']}", flush=True)
    out = {name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
           for name, frames in parts.items()}

    args.results.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in out.items():
        path = args.results / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.6g")
        written.append(path.name)
    if len(out["sweep"]):
        written += [p.name for p in draw_sweep(out["sweep"], config.epsilons[0], args.results)]
    heads = headline(out, config.epsilons)
    summary = {
        "analysis": "certify decisions, not measurements (measured_space.decisions)",
        "inputs": records, "thresholds": {d: {q: list(t) for q, t in qs.items()} for d, qs in thresholds.items()},
        "sweep_u": list(dec.SWEEP_U), "safety_qtypes": list(dec.SAFETY_QTYPES),
        "margin_qtypes": list(dec.MARGIN_QTYPES), "fit_qtypes": list(dec.FIT_QTYPES),
        "config": dataclasses.asdict(config),
        "ltt_growth": inspect.signature(ltt_threshold).parameters["growth"].default,
        "primary_epsilon": config.epsilons[0],
        "skip_margins": bool(args.skip_margins), "assumptions": list(ASSUMPTIONS), "headline": heads,
        "outputs": written, "git_commit": git_commit(), "seconds": round(time.time() - started, 1)}
    (args.results / "summary.json").write_text(dumps(summary))
    pd.set_option("display.width", 200)
    if heads:
        print(pd.DataFrame(heads).round(3).to_string(index=False))
    print(f"wrote {', '.join(written)} and summary.json to {args.results} ({summary['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
