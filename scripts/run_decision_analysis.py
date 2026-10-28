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

- **Second opinion** (``--witness LABELLER=WITNESS``, defaults :data:`DEFAULT_WITNESSES`). Part A again, with a
  decision accepted only when a witness labeller answering the same question id makes the same call; score the
  larger of the two labellers' scores, certified by the same Learn-then-Test (``witness.csv``).

**Calibration.** An audit folder that holds its saved leave-two-out calibration predictions
(``calibration.json``, written by ``scripts/run_audit.py`` since they are saved) is calibrated exactly: each test
scene's calibration decisions are rebuilt from predictions that never saw its fold
(:func:`measured_space.decisions.calibration_answers`). An older folder falls back to the approximate mode
(other folds' test predictions). The mode is recorded per labeller and part in ``summary.json`` and in every
table's ``calibration`` column.

Outputs (aggregates only) in ``--results``, which must not exist or must be empty (nothing is ever overwritten):
``decision_gate.csv``, ``sweep.csv``, ``safety.csv``, ``margin_intervals.csv``, ``margin_gate.csv``,
``witness.csv`` (when a second opinion ran), ``summary.json`` (inputs with checksums, thresholds, settings,
assumptions, calibration modes, notes, git commit, seconds) and the figure ``decision_sweep.pdf`` / ``.png``
(certified accepted share against ``|u|`` per labeller, handheld and warehouse panels, with the fixed-tolerance
gate's share dashed).

Usage (from the repository root)::

    python scripts/run_decision_analysis.py --results <new folder>              # the six default labellers
    python scripts/run_decision_analysis.py --results <new folder> \\
        --audits cutr_sam3=audit_ark106_cutr_sam3 --audits sparse4d_v22=audit_ss12_sparse4d_v22
    python scripts/run_decision_analysis.py --results <new folder> --audits 2d_lidar=audit_ark106_2d:lidar
    python scripts/run_decision_analysis.py --results <new folder> --witness 2d_lidar=cutr_sam3   # second opinion
    python scripts/run_decision_analysis.py --results <new folder> --skip-witness

``--audits NAME=FOLDER[:SOURCE]``: ``FOLDER`` relative to ``<data root>/outputs`` unless absolute; ``SOURCE`` is
the depth source to keep (required when the audit has several).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
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
"""tuple of str: the CSV tables written to ``--results`` (plus ``witness`` when a second opinion ran)."""

DEFAULT_WITNESSES: tuple[str, ...] = ("2d_lidar=cutr_sam3", "cutr_sam3=2d_lidar", "cutr_clip=2d_lidar",
                                      "2d_moge=sparse4d_v22")
"""tuple of str: default ``LABELLER=WITNESS`` pairs of the second-opinion rule (used when both are analysed)."""

WITNESS_PARTS: tuple[str, ...] = ("threshold", "sweep", "safety")
"""tuple of str: the part-A analyses the second-opinion rule is reported for."""

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
    "folds, recomputed and checked against folds.csv where recorded); the test scene's own answers never calibrate "
    "it. How their predictions were obtained is recorded per labeller and part (summary.json 'calibration', and "
    "the tables' 'calibration' column): 'exact' or 'approximate', as described under 'calibration_modes'.",
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
    "the test scene's fold, exact or approximate as the labeller's calibration mode. The sum and quadrature error "
    "bars use only the audit's own half90 of the test answers and do not depend on the mode.",
)
"""tuple of str: assumptions recorded in ``summary.json``."""

CALIBRATION_NOTES: dict[str, str] = {
    "exact": "The audit folder holds its leave-two-out calibration predictions (calibration_numeric.csv.gz) and "
             "per-test-scene conformal quantiles (calibration_quantiles.csv.gz). For a test scene S and a question "
             "type whose group holds S, each calibration decision uses mu and sigma from a model fitted without S's "
             "fold and the item's own fold, and the half-width q90 * sigma with q90 the quantile that S's own "
             "answers of the item's cell (question type, depth source) get; margins likewise, with the direct "
             "quantile of S taken over leave-two-out margins. For a question type with no answer of S in any depth "
             "source, no model or quantile of that type saw S, and its calibration decisions are the other scenes' "
             "own test decisions. So no calibration decision for S comes from a model or a quantile that saw S's "
             "fold. One data reuse remains: the per-cell quantile that scales every calibration decision's error bar "
             "is computed from those same calibration items' scores (one scalar per cell), as the conformal step "
             "of the audit computes it for the test scene.",
    "approximate": "The audit folder does not hold its leave-two-out calibration predictions. Calibration "
                   "decisions for a test scene are other folds' own out-of-scene test predictions: a calibration "
                   "decision's mu and sigma come from a model whose training data included the test scene's fold, "
                   "and its half90 from a conformal quantile whose calibration scores included that fold. The "
                   "Learn-then-Test certificate is approximate in this respect."}
"""dict: what each calibration mode (:data:`measured_space.decisions.CALIBRATION_MODES`) means, for ``summary.json``."""



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
    p.add_argument("--witness", action="append", default=None, metavar="LABELLER=WITNESS",
                   help="second-opinion rule: certify LABELLER's part-A decisions only where WITNESS (another "
                        "analysed labeller) answers the same question id with the same call (repeatable; default: "
                        f"{' '.join(DEFAULT_WITNESSES)}, those whose labellers are both analysed)")
    p.add_argument("--skip-witness", action="store_true", help="skip the second-opinion rule")
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


def read_calibration(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    """The audit's saved leave-two-out calibration predictions and quantiles, if it wrote them.

    Parameters
    ----------
    folder : pathlib.Path
        The audit's per-item folder.

    Returns
    -------
    tuple or None
        ``(predictions, quantiles, files)``: ``calibration_numeric.csv.gz`` (kind ``sigma`` rows: ``kind,
        test_scene, row, mu, sigma``), ``calibration_quantiles.csv.gz`` and the three files' ``{"bytes",
        "sha256"}`` records; None when ``calibration.json`` is absent (an audit written before they were saved).

    Raises
    ------
    FileNotFoundError
        If ``calibration.json`` names a file that is missing.
    """
    record_path = folder / "calibration.json"
    if not record_path.is_file():
        return None
    names = json.loads(record_path.read_text())["files"]
    predictions = pd.read_csv(folder / names["predictions"], usecols=["kind", "test_scene", "row", "mu", "sigma"],
                              dtype={"test_scene": str, "kind": str})
    predictions = predictions[predictions["kind"] == "sigma"].reset_index(drop=True)
    quantiles = pd.read_csv(folder / names["quantiles"], dtype={"test_scene": str, "kind": str,
                                                                "depth_source": str})
    files = {name: file_record(folder / name) for name in sorted(names.values())}
    return predictions, quantiles, files


@dataclass
class Labeller:
    """One labeller's answers, folds and calibration, read from its audit folder.

    Attributes
    ----------
    audit : AuditSpec
        The labeller.
    answers : pandas.DataFrame
        :func:`measured_space.decisions.answer_table` of its depth source.
    fold_of : dict
        ``(dataset, qtype, scene) -> fold``.
    dataset : str
        Its dataset.
    source : str
        Its depth source.
    summary : dict
        The audit's ``summary.json``.
    files : dict
        Input file name to ``{"bytes", "sha256"}`` (with the calibration files when read).
    folds : pandas.DataFrame or None
        The audit's ``folds.csv``.
    checked : list of str
        Groups whose recomputed folds matched the record.
    mode : str
        ``"exact"`` (the audit saved its calibration predictions) or ``"approximate"``.
    cal_answers : pandas.DataFrame or None
        :func:`measured_space.decisions.calibration_answers` (exact mode only).
    exact : set of tuple or None
        :func:`measured_space.decisions.exact_groups` (exact mode only).
    calibration : dict
        The calibration record for ``summary.json`` (mode, note, counts and the half-width check).
    """

    audit: AuditSpec
    answers: pd.DataFrame
    fold_of: dict[tuple[str, str, str], int]
    dataset: str
    source: str
    summary: dict[str, Any]
    files: dict[str, Any]
    folds: pd.DataFrame | None
    checked: list[str]
    mode: str
    cal_answers: pd.DataFrame | None
    exact: set[tuple[str, str, str]] | None
    calibration: dict[str, Any]


def load_labeller(audit: AuditSpec, config: dec.DecisionConfig) -> Labeller:
    """Read a labeller's audit: its answers, folds and, when saved, its exact calibration answers.

    Parameters
    ----------
    audit : AuditSpec
        The labeller.
    config : DecisionConfig
        The error bar's level.

    Returns
    -------
    Labeller
        Everything the analyses need.

    Raises
    ------
    ValueError
        If the audit holds more than one dataset.
    """
    pairs, numeric, gate, folds, summary, files = read_audit(audit)
    audit_cfg = summary.get("config", {})
    max_folds, seed = audit_cfg.get("max_folds", 20), audit_cfg.get("seed", 0)
    answers, checked = dec.answer_table(pairs, numeric, gate, source=audit.source, level=config.level,
                                        max_folds=max_folds, seed=seed, recorded_folds=folds)
    datasets = sorted(answers["dataset"].unique())
    if len(datasets) != 1:
        raise ValueError(f"{audit.name}: expected one dataset, got {datasets}")
    saved = read_calibration(audit.folder)
    mode = "exact" if saved is not None else "approximate"
    cal_answers: pd.DataFrame | None = None
    exact: set[tuple[str, str, str]] | None = None
    calibration: dict[str, Any] = {"mode": mode, "note": CALIBRATION_NOTES[mode]}
    if saved is not None:
        predictions, quantiles, cal_files = saved
        files.update(cal_files)
        exact = dec.exact_groups(quantiles)
        cal_answers = dec.calibration_answers(answers, predictions, quantiles, level=config.level)
        calibration.update(n_calibration_answers=int(len(cal_answers)),
                           n_exact_groups=len({(d, q, s) for d, q, s in exact if q in set(answers["qtype"])}),
                           test_halfwidth_check=dec.check_test_halfwidths(answers, quantiles, level=config.level))
    return Labeller(audit, answers, dec.fold_lookup(answers), datasets[0], str(answers["depth_source"].iloc[0]),
                    summary, files, folds, checked, mode, cal_answers, exact, calibration)


def analyse_labeller(audit: AuditSpec, thresholds: Mapping[str, Mapping[str, Sequence[float]]],
                     config: dec.DecisionConfig, *, skip_margins: bool, label_override: Path | None,
                     log: Callable[[str], None] = print,
                     lab: Labeller | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
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
    lab : Labeller, optional
        The labeller already read (:func:`load_labeller`); read here if None.

    Returns
    -------
    tables : dict of str to pandas.DataFrame
        The report tables of :data:`OUTPUT_TABLES` (margin tables empty when part B is skipped).
    record : dict
        Inputs, counts, the calibration record (mode per part and the calibration decisions' sources) and notes
        for ``summary.json``.

    Raises
    ------
    ValueError
        If the audit holds more than one dataset (when it is read here, :func:`load_labeller`).
    """
    started = time.time()
    lab = lab if lab is not None else load_labeller(audit, config)
    answers, fold_of, dataset, source = lab.answers, lab.fold_of, lab.dataset, lab.source
    summary, files, folds, checked = lab.summary, lab.files, lab.folds, lab.checked
    cal_answers, exact, mode = lab.cal_answers, lab.exact, lab.mode
    audit_cfg = summary.get("config", {})
    boot = config.bootstrap()
    family = config.pool == "family"
    notes: list[str] = []
    missing_thr = sorted(set(answers["qtype"]) - set(thresholds.get(dataset, {})))
    if missing_thr:
        notes.append(f"no thresholds for {missing_thr}: left out of part A")
    log(f"{audit.name}: {len(answers)} answers ({dataset}, source {source}) from {audit.folder}; "
        f"calibration {mode}")
    calibration: dict[str, Any] = dict(lab.calibration)
    built: dict[str, dict[str, int]] = {}

    def calibrate(decisions: pd.DataFrame, cal_decisions: pd.DataFrame | None, pools: list[str], part: str,
                  **kwargs: Any) -> pd.DataFrame:
        """Apply every rule to one part's decisions, in this labeller's calibration mode, and count the sources.

        Parameters
        ----------
        decisions : pandas.DataFrame
            The part's test decisions.
        cal_decisions : pandas.DataFrame or None
            Its calibration decisions in the exact mode (None: approximate).
        pools : list of str
            Calibration pool columns.
        part : str
            Part name: its table in :data:`OUTPUT_TABLES`.
        **kwargs : Any
            Passed to :func:`measured_space.decisions.apply_rules` (``one_sided``).

        Returns
        -------
        pandas.DataFrame
            The decisions with every rule's columns.
        """
        out, thr = dec.apply_rules(decisions, fold_of, config, pools, calibration=cal_decisions, exact=exact,
                                   **kwargs)
        primary = thr[thr["epsilon"] == config.epsilons[0]] if len(thr) else thr
        built[part] = {"n_test_scene_pools": int(len(primary)), "n_cal": int(primary["n_cal"].sum()) if len(thr)
                       else 0}
        if "n_cal_exact" in primary:
            built[part].update(n_cal_exact=int(primary["n_cal_exact"].sum()),
                               n_cal_own=int(primary["n_cal_own"].sum()))
        return out

    # Part A: fixed thresholds
    rules = dec.standard_rules(config.epsilons)
    a = dec.threshold_decisions(answers, thresholds)
    a_cal = dec.threshold_decisions(cal_answers, thresholds) if cal_answers is not None else None
    a = calibrate(a, a_cal, ["qtype", "threshold"] if family else [], "decision_gate")
    gate_table = dec.rule_table(_with_all(a, qtype="all", threshold=np.nan), ["qtype", "threshold"], rules, boot)

    # Sweep: thresholds relative to the reference
    sw = dec.sweep_decisions(answers)
    sw_cal = dec.sweep_decisions(cal_answers) if cal_answers is not None else None
    sw = calibrate(sw, sw_cal, ["qtype", "u_abs"] if family else ["u_abs"], "sweep")
    sw_all = _with_all(sw, qtype="all")
    sweep_table = pd.concat([dec.rule_table(sw_all, ["qtype", "u_abs", "u"], rules, boot),
                             dec.rule_table(sw_all.assign(u=np.nan), ["qtype", "u_abs", "u"], rules, boot)],
                            ignore_index=True)

    # Safety: one-sided clearance decisions
    sd = dec.safety_decisions(answers, thresholds)
    safety_table = pd.DataFrame()
    if len(sd):
        sd_cal = dec.safety_decisions(cal_answers, thresholds) if cal_answers is not None else None
        sd = calibrate(sd, sd_cal, ["qtype", "threshold"] if family else [], "safety", one_sided=True)
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
                md_cal: pd.DataFrame | None = None
                if cal_answers is not None:
                    cal_pairs, _ = dec.margin_pairs(cal_answers, referents)
                    pairs_m, cal_pairs = dec.direct_margin_halfwidth_exact(pairs_m, cal_pairs, config.level)
                    md_cal = dec.margin_decisions(cal_pairs, config.fit_margin)
                    record_b["n_calibration_pairs"] = int(len(cal_pairs))
                else:
                    pairs_m = dec.direct_margin_halfwidth(pairs_m, config.level)
                margin_intervals = dec.margin_interval_table(_with_all(pairs_m, qtype="all"), ["qtype"], boot)
                md = dec.margin_decisions(pairs_m, config.fit_margin)
                md = calibrate(md, md_cal, ["decision", "qtype"] if family else ["decision"], "margin_gate")
                margin_gate = dec.rule_table(_with_all(md, qtype="all"), ["decision", "qtype"], rules, boot)
            else:
                notes.append("part B: no pairs")

    tables = {"decision_gate": gate_table, "sweep": sweep_table, "safety": safety_table,
              "margin_intervals": margin_intervals, "margin_gate": margin_gate}
    tables = {k: _label(v, audit, dataset, source).assign(calibration=mode) if len(v) else v
              for k, v in tables.items()}
    calibration["parts"] = {part: mode for part, table in tables.items() if len(table)}
    calibration["calibration_decisions"] = built
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
              "part_b": record_b, "calibration": calibration, "notes": notes,
              "seconds": round(time.time() - started, 1)}
    return tables, record


# --------------------------------------------------------------------------------------------------------
# Second opinion: a witness labeller
# --------------------------------------------------------------------------------------------------------


def analyse_witness(a: Labeller, b: Labeller, thresholds: Mapping[str, Mapping[str, Sequence[float]]],
                    config: dec.DecisionConfig,
                    log: Callable[[str], None] = print) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The second-opinion rule for labeller ``a`` witnessed by ``b``, next to ``a``'s single-labeller rules.

    For each part-A analysis (:data:`WITNESS_PARTS`), ``a``'s decisions get ``b``'s answer to the same question
    id (:func:`measured_space.decisions.attach_witness`) and are certified with the second-opinion score
    (:func:`measured_space.decisions.apply_witness_rule`) on the same calibration decisions as ``a``'s own rule.
    Exact when both audits saved their calibration predictions: ``a``'s calibration decisions for a test scene
    are its leave-two-out ones (:func:`measured_space.decisions.complete_calibration_answers`) and ``b``'s
    answer to each is ``b``'s leave-two-out answer for that test scene where ``b`` has one, else ``b``'s test
    answer (which then never saw the test scene); otherwise approximate (test predictions of both).

    Parameters
    ----------
    a : Labeller
        The labeller whose decisions are certified.
    b : Labeller
        The witness.
    thresholds : mapping
        ``{dataset: {qtype: thresholds in m}}``.
    config : DecisionConfig
        Settings.
    log : callable, default print
        Receives progress messages.

    Returns
    -------
    table : pandas.DataFrame
        :func:`measured_space.decisions.rule_table` rows (rules of
        :func:`measured_space.decisions.witness_rules`) with ``labeller, setting, dataset, source, witness,
        witness_source, part, scope`` (``all``: every decision of ``a``; ``witnessed``: those ``b`` answered),
        the part's keys (``qtype``, ``threshold`` or ``u_abs``) and ``calibration``.
    record : dict
        Counts, the shared-question check and the calibration mode, for ``summary.json``.

    Raises
    ------
    ValueError
        If the two labellers audit different datasets.
    """
    if a.dataset != b.dataset:
        raise ValueError(f"witness {b.audit.name} audits {b.dataset}, not {a.dataset}")
    mode = "exact" if a.mode == "exact" and b.mode == "exact" else "approximate"
    family = config.pool == "family"
    boot = config.bootstrap()
    shared = a.answers.merge(b.answers[["id", "ref"]], on="id", suffixes=("", "_w"))
    record: dict[str, Any] = {
        "labeller": a.audit.name, "witness": b.audit.name, "witness_source": b.source, "calibration": mode,
        "n_answers": int(len(a.answers)), "n_witnessed": int(len(shared)),
        "n_witnessed_scenes": int(shared["scene"].nunique()),
        "ref_max_abs_diff": float(np.abs(shared["ref"] - shared["ref_w"]).max()) if len(shared) else math.nan,
        "rule": "second_opinion: score inf unless the witness answers the question id and its call (c_w > t; "
                "one-sided: both outside) agrees with the labeller's, else max(half / |c - t|, half_w / |c_w - t|); "
                "certified per test scene by the same Learn-then-Test on the same calibration decisions",
        "calibration_note": "exact: the labeller's calibration decisions for a test scene are its leave-two-out ones "
                            "(own test decisions for question types whose group lacks the scene), and the witness's "
                            "answer to each is its leave-two-out answer for that test scene where it has one, else "
                            "its test answer, which then never saw the test scene either (the scene is outside the "
                            "witness's group of that type, or shares the witness fold of the item's scene, so the "
                            "witness model held both out); approximate: test predictions of both labellers"}
    cal_rows: pd.DataFrame | None = None
    covered: set[tuple[str, str, str]] | None = None
    if mode == "exact" and a.cal_answers is not None and a.exact is not None:
        cal_rows, covered = dec.complete_calibration_answers(a.answers, a.cal_answers, a.exact)
    builders: dict[str, tuple[Callable[[pd.DataFrame], pd.DataFrame], list[str], bool]] = {
        "threshold": (lambda x: dec.threshold_decisions(x, thresholds), ["qtype", "threshold"] if family else [],
                      False),
        "sweep": (dec.sweep_decisions, ["qtype", "u_abs"] if family else ["u_abs"], False),
        "safety": (lambda x: dec.safety_decisions(x, thresholds), ["qtype", "threshold"] if family else [], True)}
    tables, sources = [], {}
    for part in WITNESS_PARTS:
        build, pools, one_sided = builders[part]
        decisions = build(a.answers)
        if not len(decisions):
            continue
        decisions = dec.attach_witness(decisions, b.answers)
        cal = None
        if cal_rows is not None:
            cal = dec.attach_witness(build(cal_rows), b.answers, b.cal_answers)
            sources[part] = {str(k): int(v) for k, v in cal["witness_from"].value_counts().items()}
        decisions, _ = dec.apply_rules(decisions, a.fold_of, config, pools, one_sided=one_sided, calibration=cal,
                                       exact=covered)
        decisions = dec.apply_witness_rule(decisions, a.fold_of, config, pools, one_sided=one_sided,
                                           calibration=cal, exact=covered)
        witnessed = decisions["c_w"].notna().to_numpy()
        frame = pd.concat([decisions.assign(scope="all"), decisions[witnessed].assign(scope="witnessed")],
                          ignore_index=True)
        rules = dec.witness_rules(config.epsilons, accept_all="accept_outside" if one_sided else "accept_all")
        if part == "sweep":
            table = dec.rule_table(_with_all(frame, qtype="all"), ["scope", "qtype", "u_abs"], rules, boot)
        else:
            table = dec.rule_table(_with_all(frame, qtype="all", threshold=np.nan), ["scope", "qtype", "threshold"],
                                   rules, boot, inside="inside" if one_sided else None)
        tables.append(table.assign(part=part))
    record["calibration_witness_from"] = sources
    out = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if len(out):
        out = _label(out, a.audit, a.dataset, a.source)
        out.insert(4, "witness", b.audit.name)
        out.insert(5, "witness_source", b.source)
        out = out.assign(calibration=mode)
    log(f"{a.audit.name} witnessed by {b.audit.name}: {record['n_witnessed']} of {record['n_answers']} answers; "
        f"calibration {mode}")
    return out, record


def witness_pairs(given: Sequence[str] | None, names: Sequence[str]) -> tuple[list[tuple[str, str]], list[str]]:
    """The ``LABELLER=WITNESS`` pairs to analyse.

    Parameters
    ----------
    given : sequence of str or None
        ``--witness`` values; None uses :data:`DEFAULT_WITNESSES` where both labellers are analysed.
    names : sequence of str
        The analysed labellers.

    Returns
    -------
    pairs : list of tuple of str
        ``(labeller, witness)``.
    notes : list of str
        Default pairs left out because a labeller is not analysed.

    Raises
    ------
    ValueError
        If a given pair is malformed, pairs a labeller with itself, or names a labeller that is not analysed.
    """
    pairs, notes = [], []
    for text in (given if given is not None else DEFAULT_WITNESSES):
        a, sep, b = text.partition("=")
        if not sep or not a or not b or a == b:
            raise ValueError(f"expected LABELLER=WITNESS with two different names, got {text!r}")
        if a not in names or b not in names:
            if given is not None:
                raise ValueError(f"--witness {text!r} names a labeller that is not analysed ({list(names)})")
            notes.append(f"default witness pair {text} skipped: not both analysed")
            continue
        pairs.append((a, b))
    return pairs, notes


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


def witness_headline(table: pd.DataFrame, epsilons: Sequence[float]) -> list[dict[str, Any]]:
    """Per witness pair: the single-labeller and second-opinion rules on the fixed-threshold decisions.

    Parameters
    ----------
    table : pandas.DataFrame
        The ``witness`` table (may be empty).
    epsilons : sequence of float
        The certified bounds.

    Returns
    -------
    list of dict
        One record per labeller and witness (part ``threshold``, scope ``all``, pooled over question types and
        thresholds): accepted share and wrong-decision rate of ``certified`` and ``second_opinion`` at each
        epsilon, ``interval`` and ``second_opinion_interval``.
    """
    if not len(table):
        return []
    pooled = table[(table["part"] == "threshold") & (table["scope"] == "all") & (table["qtype"] == "all")]
    out = []
    for (name, witness), g in pooled.groupby(["labeller", "witness"], sort=False):
        row: dict[str, Any] = {"labeller": name, "witness": witness, "n_decisions": int(g["n"].iloc[0])}
        for rule in ("certified", "second_opinion"):
            for eps in epsilons:
                r = g[(g["rule"] == rule) & np.isclose(g["epsilon"], eps)]
                if len(r):
                    row[f"{rule}_eps{eps:g}_accept_share"] = float(r["accept_share"].iloc[0])
                    row[f"{rule}_eps{eps:g}_wrong_rate"] = float(r["wrong_rate"].iloc[0])
        for rule in ("interval", "second_opinion_interval"):
            r = g[g["rule"] == rule]
            if len(r):
                row[f"{rule}_accept_share"] = float(r["accept_share"].iloc[0])
                row[f"{rule}_wrong_rate"] = float(r["wrong_rate"].iloc[0])
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

    pairs_w, witness_notes = ([], []) if args.skip_witness else witness_pairs(args.witness, names)
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in OUTPUT_TABLES}
    records: dict[str, Any] = {}
    labellers: dict[str, Labeller] = {}
    for audit in audits:
        labellers[audit.name] = load_labeller(audit, config)
        tables, record = analyse_labeller(audit, thresholds, config, skip_margins=args.skip_margins,
                                          label_override=overrides.get(audit.name), lab=labellers[audit.name])
        for name, frame in tables.items():
            if len(frame):
                parts[name].append(frame)
        records[audit.name] = record
        print(f"{audit.name}: done in {record['seconds']} s; notes: {record['notes']}", flush=True)
    out = {name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
           for name, frames in parts.items()}
    witness_records: list[dict[str, Any]] = []
    if pairs_w:
        frames = []
        for a, b in pairs_w:
            frame, record = analyse_witness(labellers[a], labellers[b], thresholds, config)
            frames.append(frame)
            witness_records.append(record)
        out["witness"] = pd.concat(frames, ignore_index=True)

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
        "skip_margins": bool(args.skip_margins), "assumptions": list(ASSUMPTIONS),
        "calibration": {name: rec["calibration"]["parts"] for name, rec in records.items()},
        "witness": {"pairs": witness_records, "notes": witness_notes, "skipped": bool(args.skip_witness),
                    "headline": witness_headline(out.get("witness", pd.DataFrame()), config.epsilons)},
        "calibration_modes": {m: CALIBRATION_NOTES[m] for m in sorted({rec["calibration"]["mode"]
                                                                      for rec in records.values()})},
        "headline": heads,
        "outputs": written, "git_commit": git_commit(), "seconds": round(time.time() - started, 1)}
    (args.results / "summary.json").write_text(dumps(summary))
    pd.set_option("display.width", 200)
    if heads:
        print(pd.DataFrame(heads).round(3).to_string(index=False))
    print(f"wrote {', '.join(written)} and summary.json to {args.results} ({summary['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
