"""Turn stored audit outputs into human review effort: how much review delivers labels at a stated quality, in minutes.

Reads the per-answer tables of ``scripts/run_audit.py`` (``pairs.csv``, ``gate.csv``, ``summary.json``) and the
aggregate ``flag_metrics.csv`` of ``scripts/run_miss_audit.py``, and computes, per labeller, answer kind (numeric,
categorical, combined) and tolerance multiple (:mod:`measured_space.review_effort`):

- **review curves**: residual wrong share of the delivered set against the share of answers reviewed, for the
  audit's risk order (feature model), the constant baseline's order, random review and the ideal order;
- **targets**: the review fraction that brings the residual wrong share to 5 %, 2 % and 1 % (``--targets``), with
  scene-cluster bootstrap 95 % intervals;
- **certified operating points**: the Learn-then-Test gate's accepted share, wrong share among accepted, review
  fraction and residual wrong share, against the certified bound;
- **minutes** per 1,000 delivered answers at 10, 30 and 60 s per review (``--seconds``), against reviewing
  everything;
- **miss review**: per miss-flag kind, flags per image, precision, share of misses flagged, minutes per 1,000
  images and misses recovered per 1,000 images.

Policy and assumptions: reviewed answers are corrected; unreviewed answers are delivered as they are; every review
costs the same time; unanswered questions (no box for a referent) are not answers and are left to the miss review.

Inputs:

- ``--audits NAME=FOLDER[:SOURCE]`` (repeatable): an audit folder and the depth or box source to analyse (needed
  when the folder holds several; e.g. ``:lidar``). Default: :data:`DEFAULT_AUDITS` under
  ``<data root>/outputs``;
- ``--misses NAME=FOLDER[:LABELLER/SOURCE[,LABELLER/SOURCE...]]`` (repeatable): a miss-audit results folder, the
  flagged labeller and source rows to report (default every row), and ``NAME``, the audit labeller they belong
  to. Default: the :data:`DEFAULT_MISSES` entries of the labellers analysed; ``--no-misses`` skips them.
  Folder paths may not contain ``:``.

Outputs, in ``--results`` (required; refused if it exists and is not empty; aggregates only, no per-answer rows):
``effort_curves.csv``, ``effort_targets.csv``, ``operating_points.csv``, ``miss_review.csv``, ``summary.json``
(inputs with checksums, settings, assumptions, definitions, notes, git commit, seconds taken) and the figure
``review_effort.pdf`` / ``.png``.

Run from the repository root (CPU only; seconds per labeller at the default 1,000 bootstrap replicates)::

    python scripts/run_review_effort.py --results <folder>
    python scripts/run_review_effort.py --results <folder> \\
        --audits ark106_cutr_sam3=<data>/outputs/audit_ark106_cutr_sam3 \\
        --misses ark106_cutr_sam3=<data>/outputs/misses_results/ark106_cutr:cutr_sam3/cutr_sam3
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from measured_space.audit import dumps
from measured_space.config import REPO_ROOT, data_root
from measured_space.review_effort import (DEFAULT_SECONDS, DEFAULT_TARGETS, GATE_COLUMNS, N_GRID, PER,
                                          labeller_effort, miss_review_table)

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
"""re.Pattern: allowed labeller names (one path component, no leading dot)."""

DEFAULT_AUDITS: tuple[tuple[str, str, str | None], ...] = (
    ("ark106_2d", "audit_ark106_2d", "lidar"),
    ("ark106_cutr_sam3", "audit_ark106_cutr_sam3", None),
    ("ark106_cutr_clip", "audit_ark106_cutr_clip", None),
    ("ss12_2d", "audit_ss12_2d", "moge"),
    ("ss12_sparse4d_v22", "audit_ss12_sparse4d_v22", None),
    ("ss12_sparse4d_v11", "audit_ss12_sparse4d_v11", None),
)
"""tuple: ``(name, folder under <data root>/outputs, source or None)`` of the default labellers. The baselines'
folders hold several depth sources: the handheld one is analysed with its device depth (iPad LiDAR), the
warehouse one with its monocular depth (MoGe-2), the depth each would be deployed with."""

DEFAULT_MISSES: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    ("ark106_2d", "misses_results/ark106_v0", (("v0", "lidar"),)),
    ("ark106_cutr_sam3", "misses_results/ark106_cutr", (("cutr_sam3", "cutr_sam3"),)),
    ("ark106_cutr_clip", "misses_results/ark106_cutr", (("cutr_clip", "cutr_clip"),)),
    ("ss12_2d", "misses_results/ss12_v0", (("v0", "moge"),)),
    ("ss12_2d", "misses_results/ss12_sparse4d_v22", (("v0", "moge"),)),
    ("ss12_2d", "misses_results/ss12_sparse4d_v11", (("v0", "moge"),)),
    ("ss12_sparse4d_v22", "misses_results/ss12_sparse4d_v22", (("sparse4d", "sparse4d"), ("s4d", "boxes"))),
    ("ss12_sparse4d_v11", "misses_results/ss12_sparse4d_v11", (("sparse4d", "sparse4d"), ("s4d", "boxes"))),
)
"""tuple: ``(audit name, folder under <data root>/outputs, (flagged labeller, source) pairs)`` of the default miss
flags: the rows where the audited labeller is the one flagged (the reverse-direction rows of a cross-labeller run
flag the witness instead). For Sparse4D two rows: its labels (``sparse4d``) flagged by its own raw boxes
(witness ``s4d``), and its raw boxes (``s4d``, source ``boxes``) flagged by the monocular baseline's detections
(witness ``v0/moge``), the cross-labeller flag of its misses."""

DISPLAY: dict[str, str] = {"ark106_2d": "v0, iPad LiDAR", "ark106_cutr_sam3": "CuTR + SAM 3",
                           "ark106_cutr_clip": "CuTR + CLIP", "ss12_2d": "v0, MoGe-2",
                           "ss12_sparse4d_v22": "Sparse4D v2.2", "ss12_sparse4d_v11": "Sparse4D v1.1"}
"""dict of str to str: legend labels of the default labellers (others are labelled by name)."""

SETTINGS: dict[str, str] = {"arkitscenes": "handheld", "smartspaces": "warehouse"}
"""dict of str to str: the capture setting of each dataset (others keep the dataset's name)."""

SETTING_TITLE: dict[str, str] = {"handheld": "Handheld RGB-D\n(ARKitScenes)",
                                 "warehouse": "Fixed cameras\n(Smart Spaces)"}
"""dict of str to str: figure panel titles per setting."""

ASSUMPTIONS: tuple[str, ...] = (
    "Answers are reviewed in an order; a reviewed answer is corrected (delivered right), an unreviewed answer is "
    "delivered as it is.",
    "Every review costs the same human time, whatever the answer and whether it was right (--seconds); a rejected "
    "answer costs no more than a flagged one.",
    "Population: the answers the audit scored (gate.csv rows of the feature model). Unanswered questions (a "
    "referent had no box) are not answers; they are the miss review's subject. Answered questions the audit did "
    "not score (question types with too few scenes or rows, non-finite errors) are left out and counted in "
    "summary.json.",
    "Wrong: |auto - reference| > tau_scale * tau for numeric answers, a changed answer for categorical ones "
    "(gate.csv wrong_s). Categorical answers do not depend on the tolerance and are used as they are at every "
    "tau_scale.",
    "Combined queue: numeric and categorical answers interleaved by their rank fraction within their own kind, "
    "so reviewing the share f reviews the riskiest share f of each kind (the two scores are on different scales).",
    "Random review is its expected value, wrong_share * (1 - f); the ideal order reviews every wrong answer "
    "first.",
    "Scene-cluster bootstrap: whole scenes resampled with replacement, the orders kept as ranked on all answers; "
    "with few scenes (12 in the warehouse) the intervals are rough.",
    "Certified operating point: the gate's decisions as stored (thresholds certified per held-out scene by "
    "Learn-then-Test at epsilon, delta); flagged and rejected answers are reviewed. The combined accepted set "
    "inherits epsilon with confidence 1 - 2 delta (union bound). When a source is selected from an audit of "
    "several, the thresholds were certified on the sources pooled.",
    "Miss review: reviewing one flag costs the same time; a flag that hits a missed object recovers it; images "
    "are the views the flag kind covers (by == 'all' rows: every flag, no score or support filter).",
)
"""tuple of str: the assumptions written into ``summary.json``."""

DEFINITIONS: dict[str, str] = {
    "review_fraction": "share of answers reviewed",
    "residual_wrong_share": "wrong answers left unreviewed / all answers (the delivered set's wrong share)",
    "wrong_share": "wrong answers / all answers, before review",
    "order": "ours: feature model risk score (numeric sigma, categorical logistic), highest first; const: the "
             "constant baseline's score; random: expected value of a random order; ideal: wrong answers first",
    "lo, hi": "scene-cluster bootstrap 95% percentile interval",
    "minus_random": "review fraction minus random review's, with a paired bootstrap interval",
    "full_review_needed": "only reviewing every answer reaches the target",
    "certified_bound": "epsilon * accepted_share: the residual wrong share the gate's certificate allows",
    "random_review_fraction": "review fraction a random order needs to reach the gate's residual wrong share",
    "ranked_review_fraction": "review fraction the same model's ranking needs to reach the gate's residual "
                              "(uncertified)",
    "minutes_<t>s": f"human minutes per {PER} delivered answers (miss review: per {PER} images) at t seconds per "
                    "review",
    "minutes_all_<t>s": f"minutes per {PER} answers to review every answer",
    "tau_scale": "multiple of the question's tolerance a numeric answer must meet; *_tau columns use 1",
}
"""dict of str to str: column definitions written into ``summary.json``."""


@dataclass(frozen=True)
class AuditSpec:
    """One ``--audits`` argument.

    Parameters
    ----------
    name : str
        The labeller's name in the outputs.
    folder : pathlib.Path
        The audit folder (``pairs.csv``, ``gate.csv``, ``summary.json``).
    source : str or None
        Depth or box source to analyse; ``None`` when the folder holds one.
    """

    name: str
    folder: Path
    source: str | None


@dataclass(frozen=True)
class MissSpec:
    """One ``--misses`` argument.

    Parameters
    ----------
    name : str
        The audit labeller the flags belong to.
    folder : pathlib.Path
        The miss-audit results folder (``flag_metrics.csv``).
    select : tuple of tuple of (str, str)
        ``(flagged labeller, source)`` rows to report; empty reports every row.
    """

    name: str
    folder: Path
    select: tuple[tuple[str, str], ...]


def _split(token: str, option: str) -> tuple[str, Path, str]:
    """Split ``NAME=FOLDER[:REST]``.

    Parameters
    ----------
    token : str
        The argument.
    option : str
        The option's name, for messages.

    Returns
    -------
    tuple of (str, pathlib.Path, str)
        Name, folder and the text after the last ``:`` (empty without one).

    Raises
    ------
    ValueError
        If the name is missing or not a plain name, or the folder does not exist.
    """
    name, sep, rest = token.partition("=")
    if not sep or not NAME_PATTERN.match(name):
        raise ValueError(f"{option} takes NAME=FOLDER[:...] with a plain name, got {token!r}")
    folder, _, tail = rest.rpartition(":") if ":" in rest else (rest, "", "")
    path = Path(folder).expanduser()
    if not path.is_dir():
        raise ValueError(f"{option} folder {path} does not exist")
    return name, path, tail


def parse_audit_spec(token: str) -> AuditSpec:
    """Parse one ``--audits NAME=FOLDER[:SOURCE]`` argument.

    Parameters
    ----------
    token : str
        The argument.

    Returns
    -------
    AuditSpec
        The parsed argument.

    Raises
    ------
    ValueError
        If the name or folder is invalid, or the folder lacks an audit table.
    """
    name, path, source = _split(token, "--audits")
    missing = [f for f in ("pairs.csv", "gate.csv", "summary.json") if not (path / f).is_file()]
    if missing:
        raise ValueError(f"audit folder {path} lacks {missing}")
    return AuditSpec(name, path, source or None)


def parse_miss_spec(token: str) -> MissSpec:
    """Parse one ``--misses NAME=FOLDER[:LABELLER/SOURCE[,LABELLER/SOURCE...]]`` argument.

    Parameters
    ----------
    token : str
        The argument.

    Returns
    -------
    MissSpec
        The parsed argument.

    Raises
    ------
    ValueError
        If the name or folder is invalid, the folder lacks ``flag_metrics.csv``, or a selection is not
        ``LABELLER/SOURCE``.
    """
    name, path, tail = _split(token, "--misses")
    if not (path / "flag_metrics.csv").is_file():
        raise ValueError(f"miss folder {path} lacks flag_metrics.csv")
    select = []
    for item in filter(None, tail.split(",")):
        labeller, sep, source = item.partition("/")
        if not sep or not labeller or not source:
            raise ValueError(f"--misses selection {item!r} is not LABELLER/SOURCE")
        select.append((labeller, source))
    return MissSpec(name, path, tuple(select))


def default_specs(outputs: Path, names: Sequence[str] | None) -> tuple[list[AuditSpec], list[MissSpec]]:
    """The default audits and miss folders under an outputs folder.

    Parameters
    ----------
    outputs : pathlib.Path
        ``<data root>/outputs``.
    names : sequence of str or None
        Audit names to give miss folders for; ``None`` gives the default audits and all their miss folders.

    Returns
    -------
    audits : list of AuditSpec
        :data:`DEFAULT_AUDITS` (empty when ``names`` is given).
    misses : list of MissSpec
        :data:`DEFAULT_MISSES` entries of the wanted names whose folder holds ``flag_metrics.csv``.
    """
    audits = [] if names is not None else [AuditSpec(n, outputs / f, s) for n, f, s in DEFAULT_AUDITS]
    wanted = set(names) if names is not None else {a.name for a in audits}
    misses = [MissSpec(n, outputs / f, sel) for n, f, sel in DEFAULT_MISSES
              if n in wanted and (outputs / f / "flag_metrics.csv").is_file()]
    return audits, misses


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and check the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The options, with ``audit_specs`` (list of :class:`AuditSpec`) and ``miss_specs`` (list of
        :class:`MissSpec`) added.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--audits", action="append", default=None, metavar="NAME=FOLDER[:SOURCE]",
                   help="an audit folder of scripts/run_audit.py and the source to analyse (repeatable; default: "
                        "the six default labellers under <data root>/outputs)")
    p.add_argument("--misses", action="append", default=None, metavar="NAME=FOLDER[:LABELLER/SOURCE,...]",
                   help="a miss-audit results folder, for the audit labeller NAME (repeatable; default: the "
                        "default miss folders of the labellers analysed)")
    p.add_argument("--no-misses", action="store_true", help="skip the miss review")
    p.add_argument("--results", type=Path, required=True,
                   help="output folder (must not exist or be empty; no default)")
    p.add_argument("--seconds", type=float, nargs="+", default=list(DEFAULT_SECONDS),
                   help="seconds of human time per review (default %(default)s)")
    p.add_argument("--targets", type=float, nargs="+", default=list(DEFAULT_TARGETS),
                   help="residual wrong shares to reach (default %(default)s)")
    p.add_argument("--n-boot", type=int, default=1000, help="scene-cluster bootstrap replicates (default %(default)s)")
    p.add_argument("--seed", type=int, default=0, help="seed of tie-breaking and the bootstrap (default %(default)s)")
    args = p.parse_args(argv)
    if any(t <= 0 for t in args.seconds):
        p.error("--seconds must be positive")
    if any(not 0 <= t < 1 for t in args.targets):
        p.error("--targets must lie in [0, 1)")
    if args.n_boot < 0:
        p.error("--n-boot must be at least 0")
    outputs: Path | None = None
    if args.audits is None or (args.misses is None and not args.no_misses):
        try:
            outputs = data_root() / "outputs"
        except RuntimeError as error:
            if args.audits is None:
                p.error(f"{error}; pass --audits explicitly")
    try:
        if args.audits is None:
            assert outputs is not None
            args.audit_specs, defaults = default_specs(outputs, None)
            args.audit_specs = [parse_audit_spec(f"{a.name}={a.folder}" + (f":{a.source}" if a.source else ""))
                                for a in args.audit_specs]
        else:
            args.audit_specs = [parse_audit_spec(t) for t in args.audits]
            defaults = default_specs(outputs, [a.name for a in args.audit_specs])[1] if outputs else []
        args.miss_specs = ([] if args.no_misses else defaults if args.misses is None
                           else [parse_miss_spec(t) for t in args.misses])
    except ValueError as error:
        p.error(str(error))
    names = [a.name for a in args.audit_specs]
    if len(set(names)) != len(names):
        p.error(f"audit names repeat: {names}")
    results = args.results
    if results.exists() and (not results.is_dir() or any(results.iterdir())):
        p.error(f"{results} exists and is not empty; choose another --results")
    inputs = [a.folder.resolve() for a in args.audit_specs] + [m.folder.resolve() for m in args.miss_specs]
    if any(results.resolve() == i or i in results.resolve().parents for i in inputs):
        p.error(f"{results} lies inside an input folder")
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


@dataclass
class LoadedAudit:
    """One audit folder's gate rows for one source, with its settings and bookkeeping.

    Parameters
    ----------
    spec : AuditSpec
        The argument it came from.
    source : str
        The source analysed.
    gate : pandas.DataFrame
        ``gate.csv`` rows of that source (:data:`measured_space.review_effort.GATE_COLUMNS` and ``dataset``).
    epsilon : float
        The gate's certified bound (from the audit's ``summary.json``).
    delta : float
        The certificate's failure probability.
    record : dict
        Input files with sizes and checksums, the sources stored, and answer counts per dataset and kind.
    """

    spec: AuditSpec
    source: str
    gate: pd.DataFrame
    epsilon: float
    delta: float
    record: dict[str, Any]


def answer_counts(pairs: pd.DataFrame, gate: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    """Questions, answers and scored answers per dataset and answer kind.

    Parameters
    ----------
    pairs : pandas.DataFrame
        ``pairs.csv`` rows of one source, indexed by row key.
    gate : pandas.DataFrame
        ``gate.csv`` rows of the same source.

    Returns
    -------
    dict
        ``{dataset: {answer_kind: {"n_questions", "n_unanswered", "n_answered", "n_scored",
        "n_answered_unscored"}}}``; scored answers are the feature model's gate rows.
    """
    kind = np.where(pairs["answer_type"] == "number", "numeric", "categorical")
    ours = gate[((gate["answer_kind"] == "numeric") & (gate["kind"] == "sigma"))
                | ((gate["answer_kind"] == "categorical") & (gate["kind"] == "logistic"))]
    scored = pairs.index.isin(ours["row"].unique())
    matched = pairs["matched"].astype(bool).to_numpy()
    out: dict[str, dict[str, dict[str, int]]] = {}
    for dataset in sorted(pairs["dataset"].astype(str).unique()):
        in_dataset = (pairs["dataset"].astype(str) == dataset).to_numpy()
        for answer_kind in ("numeric", "categorical"):
            sel = in_dataset & (kind == answer_kind)
            out.setdefault(dataset, {})[answer_kind] = {
                "n_questions": int(sel.sum()), "n_unanswered": int((sel & ~matched).sum()),
                "n_answered": int((sel & matched).sum()), "n_scored": int((sel & scored).sum()),
                "n_answered_unscored": int((sel & matched & ~scored).sum())}
    return out


def load_audit(spec: AuditSpec) -> LoadedAudit:
    """Read an audit folder's gate rows for the chosen source, and its settings and counts.

    Parameters
    ----------
    spec : AuditSpec
        The audit folder and source.

    Returns
    -------
    LoadedAudit
        The loaded audit.

    Raises
    ------
    ValueError
        If the folder holds several sources and none was chosen, or the chosen one is absent.
    """
    gate = pd.read_csv(spec.folder / "gate.csv", usecols=[*GATE_COLUMNS, "dataset"],
                       dtype={"scene": str, "dataset": str, "depth_source": str, "decision": str})
    sources = sorted(gate["depth_source"].unique())
    source = spec.source
    if source is None:
        if len(sources) != 1:
            raise ValueError(f"{spec.folder} holds sources {sources}; choose one with {spec.name}=FOLDER:SOURCE")
        source = sources[0]
    if source not in sources:
        raise ValueError(f"{spec.folder} has no source {source!r}; it holds {sources}")
    gate = gate[gate["depth_source"] == source].reset_index(drop=True)
    pairs = pd.read_csv(spec.folder / "pairs.csv", usecols=["dataset", "depth_source", "answer_type", "matched"],
                        dtype={"dataset": str, "depth_source": str})
    pairs = pairs[pairs["depth_source"] == source]
    config = json.loads((spec.folder / "summary.json").read_text()).get("config", {})
    files = {name: {"bytes": (spec.folder / name).stat().st_size, "sha256": sha256(spec.folder / name)}
             for name in ("pairs.csv", "gate.csv", "summary.json")}
    record = {"name": spec.name, "folder": str(spec.folder), "source": source, "sources_stored": sources,
              "files": files, "counts": answer_counts(pairs, gate)}
    return LoadedAudit(spec, source, gate, float(config.get("epsilon", np.nan)), float(config.get("delta", np.nan)),
                       record)


def make_figures_module() -> ModuleType:
    """Import ``scripts/make_figures.py`` for its plotting conventions (it is not part of the package).

    Returns
    -------
    types.ModuleType
        The module (importing it selects matplotlib's Agg backend and has no other side effects).
    """
    if "make_figures" in sys.modules:
        return sys.modules["make_figures"]
    spec = importlib.util.spec_from_file_location("make_figures", Path(__file__).resolve().with_name("make_figures.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_figures"] = module
    spec.loader.exec_module(module)
    return module


def labeller_styles(entries: Sequence[tuple[str, str, str]], mf: ModuleType) -> dict[tuple[str, str],
                                                                                     tuple[str, str, str]]:
    """Colour, marker and legend label of each labeller, by its source, unique within a setting.

    Parameters
    ----------
    entries : sequence of tuple of (str, str, str)
        ``(setting, labeller, source)`` in drawing order.
    mf : types.ModuleType
        ``scripts/make_figures.py`` (``SOURCE_STYLE`` and the palette).

    Returns
    -------
    dict
        ``(setting, labeller) -> (colour, marker, label)``. A labeller takes its source's style; when another
        labeller of the same setting already has that colour, the next unused palette colour and marker.
    """
    palette = [mf.BLUE, mf.AQUA, mf.ORANGE, mf.INK, mf.MUTED]
    markers = ["o", "s", "^", "v", "P", "X", "D"]
    out: dict[tuple[str, str], tuple[str, str, str]] = {}
    used: dict[str, set[str]] = {}
    for setting, labeller, source in entries:
        colour, marker, _ = mf.SOURCE_STYLE.get(source, (mf.MUTED, "*", source))
        taken = used.setdefault(setting, set())
        if colour in taken:
            colour = next((c for c in palette if c not in taken), mf.MUTED)
            marker = next((m for m in markers if m not in {s[1] for k, s in out.items() if k[0] == setting}), "*")
        taken.add(colour)
        out[(setting, labeller)] = (colour, marker, DISPLAY.get(labeller, labeller))
    return out


def effort_figure(curves: pd.DataFrame, points: pd.DataFrame, styles: dict[tuple[str, str], tuple[str, str, str]],
                  mf: ModuleType) -> Any:
    """Residual wrong share against review fraction, one panel per setting (all answers, at the tolerance).

    Parameters
    ----------
    curves : pandas.DataFrame
        ``effort_curves.csv`` rows.
    points : pandas.DataFrame
        ``operating_points.csv`` rows.
    styles : dict
        :func:`labeller_styles` output.
    mf : types.ModuleType
        ``scripts/make_figures.py`` (style, panels and palette).

    Returns
    -------
    matplotlib.figure.Figure
        Per labeller, the risk order (solid) and random review (dashed), and the certified gate's operating point
        (marker); log residual axis, both axes in percent.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator

    mf.style()
    sel = (curves["answer_kind"] == "combined") & np.isclose(curves["tau_scale"].astype(float), 1.0)
    data = curves[sel]
    ops = points[(points["answer_kind"] == "combined") & np.isclose(points["tau_scale"].astype(float), 1.0)
                 & (points["order"] == "ours")]
    present = list(dict.fromkeys(data["setting"]))
    settings = [s for s in SETTING_TITLE if s in present] + [s for s in present if s not in SETTING_TITLE]
    fig, axes = mf.panels(max(len(settings), 1), 2.3)
    fig.subplots_adjust(wspace=0.22)                          # room for the 100 / 0 tick labels between panels
    floor = 1e-3
    top = min(1.0, 1.6 * float(data["residual_wrong_share"].max())) if len(data) else 1.0
    handles: list[Any] = []
    for ax, setting in zip(axes, settings):
        d = data[data["setting"] == setting]
        for labeller in dict.fromkeys(d["labeller"]):
            colour, marker, label = styles[(setting, labeller)]
            g = d[d["labeller"] == labeller]
            for order, ls, lw in (("ours", "-", 1.1), ("random", (0, (3, 2)), 0.8)):
                c = g[g["order"] == order]
                y = c["residual_wrong_share"].to_numpy(float)
                ax.plot(c["review_fraction"], np.where(y > 0, y, np.nan), color=colour, linestyle=ls, linewidth=lw,
                        zorder=2 if order == "ours" else 1)
            op = ops[(ops["setting"] == setting) & (ops["labeller"] == labeller)]
            for _, r in op.iterrows():
                ax.plot([r["review_fraction"]], [max(r["residual_wrong_share"], floor)], linestyle="none",
                        marker=marker, color=colour, markersize=4.5, markeredgecolor="white", markeredgewidth=0.6,
                        zorder=4)
            handles.append(Line2D([], [], color=colour, marker=marker, markeredgecolor="white",
                                  markeredgewidth=0.6, label=label))
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}"))
        ax.set_xlim(0, 1)
        ax.set_ylim(floor, top)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:g}"))
        ax.set_title(SETTING_TITLE.get(setting, setting), color=mf.INK)
        ax.set_xlabel("Answers reviewed (%)")
    axes[0].set_ylabel("Wrong answers delivered (%)")
    handles += [Line2D([], [], color=mf.INK, linewidth=1.1, label="Risk order"),
                Line2D([], [], color=mf.INK, linewidth=0.8, linestyle=(0, (3, 2)), label="Random order"),
                Line2D([], [], color=mf.INK, marker="o", linestyle="none", markeredgecolor="white",
                       label="Certified gate")]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()                      # type: ignore[attr-defined]
    bottom = min(a.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0 for a in fig.axes)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, bottom - 0.01), ncol=3, handlelength=1.6,
               columnspacing=1.2)
    return fig


def headline(targets: pd.DataFrame, points: pd.DataFrame) -> list[dict[str, Any]]:
    """Per labeller, all answers at the tolerance: the gate's operating point and the review each order needs.

    Parameters
    ----------
    targets : pandas.DataFrame
        ``effort_targets.csv`` rows.
    points : pandas.DataFrame
        ``operating_points.csv`` rows.

    Returns
    -------
    list of dict
        One record per labeller: ``setting, labeller, n_answers, wrong_share``, the gate's
        ``gate_review_fraction`` and ``gate_residual_wrong_share``, and per target ``review_<order>_<target>``
        for the risk order and random review.
    """
    out = []
    t = targets[(targets["answer_kind"] == "combined") & np.isclose(targets["tau_scale"].astype(float), 1.0)]
    p = points[(points["answer_kind"] == "combined") & np.isclose(points["tau_scale"].astype(float), 1.0)
               & (points["order"] == "ours")]
    for (setting, labeller), g in t.groupby(["setting", "labeller"], sort=False):
        record: dict[str, Any] = {"setting": setting, "labeller": labeller, "n_answers": int(g["n_answers"].iloc[0]),
                                  "wrong_share": float(g["wrong_share"].iloc[0])}
        op = p[(p["setting"] == setting) & (p["labeller"] == labeller)]
        if len(op):
            record["gate_review_fraction"] = float(op["review_fraction"].iloc[0])
            record["gate_residual_wrong_share"] = float(op["residual_wrong_share"].iloc[0])
        for _, r in g[g["order"].isin(["ours", "random"])].iterrows():
            record[f"review_{r['order']}_{r['target']:g}"] = [float(r["review_fraction"]), float(r["lo"]),
                                                             float(r["hi"])]
        out.append(record)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Compute review effort for every audit and miss folder, and write the tables, summary and figure.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status: 0 on success, 1 if nothing was analysed or the results folder filled up meanwhile.
    """
    args = parse_args(argv)
    started = time.perf_counter()
    curves, targets, points, notes = [], [], [], []
    records, entries = [], []
    for spec in args.audit_specs:
        t0 = time.perf_counter()
        audit = load_audit(spec)
        records.append({**audit.record, "epsilon": audit.epsilon, "delta": audit.delta})
        if len(audit.record["sources_stored"]) > 1:
            notes.append(f"{spec.name}: source {audit.source} of {audit.record['sources_stored']}; the gate's "
                         "thresholds were certified on the sources pooled")
        for dataset, gate in audit.gate.groupby("dataset", sort=True):
            setting = SETTINGS.get(str(dataset), str(dataset))
            labels = {"setting": setting, "labeller": spec.name, "dataset": str(dataset), "source": audit.source}
            c, t, p = labeller_effort(gate, labels, epsilon=audit.epsilon, targets=args.targets,
                                      seconds=args.seconds, n_boot=args.n_boot, seed=args.seed)
            curves.append(c)
            targets.append(t)
            points.append(p)
            entries.append((setting, spec.name, audit.source))
            for kind, counts in audit.record["counts"].get(str(dataset), {}).items():
                if counts["n_answered_unscored"]:
                    notes.append(f"{spec.name}/{dataset}/{kind}: {counts['n_answered_unscored']} answered questions "
                                 "not scored by the audit are left out")
        print(f"{spec.name}: {audit.source}, {len(audit.gate)} gate rows ({time.perf_counter() - t0:.1f} s)",
              flush=True)
    if not curves:
        print("nothing analysed", file=sys.stderr)
        return 1
    curves_df = pd.concat(curves, ignore_index=True)
    targets_df = pd.concat(targets, ignore_index=True)
    points_df = pd.concat(points, ignore_index=True)
    setting_of = {name: setting for setting, name, _ in entries}
    misses, miss_records = [], []
    for m in args.miss_specs:
        frame = pd.read_csv(m.folder / "flag_metrics.csv")
        table = miss_review_table(frame, args.seconds, m.select or None)
        if table.empty:
            notes.append(f"{m.name}: no by == 'all' rows for {list(m.select)} in {m.folder}")
        table.insert(0, "miss_run", m.folder.name)
        table.insert(0, "labeller", m.name)
        table.insert(0, "setting", [setting_of.get(m.name, SETTINGS.get(str(d), str(d))) for d in table["dataset"]])
        misses.append(table)
        miss_records.append({"name": m.name, "folder": str(m.folder), "select": [f"{a}/{b}" for a, b in m.select],
                             "files": {"flag_metrics.csv": {"bytes": (m.folder / "flag_metrics.csv").stat().st_size,
                                                            "sha256": sha256(m.folder / "flag_metrics.csv")}}})
    miss_df = pd.concat(misses, ignore_index=True) if misses else pd.DataFrame()

    mf = make_figures_module()
    fig = effort_figure(curves_df, points_df, labeller_styles(entries, mf), mf)
    results = args.results
    if results.exists() and any(results.iterdir()):                  # re-check: nothing may have appeared meanwhile
        print(f"{results} is no longer empty; nothing written", file=sys.stderr)
        return 1
    results.mkdir(parents=True, exist_ok=True)
    for name, frame in (("effort_curves", curves_df), ("effort_targets", targets_df),
                        ("operating_points", points_df), ("miss_review", miss_df)):
        frame.to_csv(results / f"{name}.csv", index=False, float_format="%.6g")
    figures = mf.save(fig, "review_effort", [results])
    summary = {
        "argv": list(sys.argv[1:] if argv is None else argv), "git_commit": git_commit(),
        "inputs": {"audits": records, "misses": miss_records},
        "settings": {"targets": list(args.targets), "seconds_per_review": list(args.seconds), "n_boot": args.n_boot,
                     "seed": args.seed, "n_grid": N_GRID, "per": PER, "tau_scales": [1.0, 2.0],
                     "interval_level": 0.95},
        "assumptions": list(ASSUMPTIONS), "definitions": DEFINITIONS, "notes": notes,
        "headline": headline(targets_df, points_df),
        "outputs": sorted([f"{n}.csv" for n in ("effort_curves", "effort_targets", "operating_points",
                                                "miss_review")] + [p.name for p in figures] + ["summary.json"]),
        "seconds": round(time.perf_counter() - started, 1)}
    (results / "summary.json").write_text(dumps(summary))

    with pd.option_context("display.width", 220, "display.max_columns", 40, "display.max_rows", 200):
        op = points_df[points_df["order"] == "ours"]
        print("\nCERTIFIED OPERATING POINTS (feature model gate)")
        print(op[["labeller", "answer_kind", "tau_scale", "n", "wrong_share", "accepted_share", "wrong_among_accepted",
                  "review_fraction", "residual_wrong_share", "certified_bound", "random_review_fraction",
                  *[f"minutes_{t:g}s" for t in args.seconds]]].round(4).to_string(index=False))
        tg = targets_df[(targets_df["order"].isin(["ours", "random"]))
                        & np.isclose(targets_df["tau_scale"].astype(float), 1.0)]
        print("\nREVIEW NEEDED FOR A RESIDUAL WRONG SHARE (at the tolerance)")
        print(tg[["labeller", "answer_kind", "target", "order", "wrong_share", "review_fraction", "lo", "hi",
                  *[f"minutes_{t:g}s" for t in args.seconds]]].round(4).to_string(index=False))
        if not miss_df.empty:
            print("\nMISS REVIEW (by == all)")
            print(miss_df[["labeller", "kind", "flagged_labeller", "flagged_source", "witness", "flags_per_view",
                           "precision", "flag_recall", "flag_recall_clear", f"recovered_per_{PER}_images",
                           *[f"minutes_{t:g}s" for t in args.seconds]]].round(4).to_string(index=False))
    if notes:
        print("\nNOTES\n" + "\n".join(f"- {n}" for n in notes))
    print(f"\nwrote {results} ({summary['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
