"""Reproduce the project's key claims from the committed result tables, and check them against their stated values.

Every claim of the write-up that carries a number is listed here once, with:

- an id and the project aim it supports (A1-A7, listed in :data:`AIMS`);
- a plain statement and the value(s) as stated (the number of decimals written sets the rounding tolerance:
  ``"33"`` passes for 32.5-33.5, ``"7.0"`` for 6.95-7.05; ``"1/6"`` passes when the value rounds to that unit
  fraction; ``"True"`` / ``"False"`` must match exactly);
- the result table(s) under ``results/`` it is read from and the script that produced them;
- a few lines that recompute the value from those tables.

Running it prints, per quantity, the stated value, the recomputed value and PASS or FAIL, and exits non-zero if
anything fails. It reads only the committed aggregate tables and needs only numpy and pandas: no datasets, no
models, no GPU. ``REPRODUCE.md`` says how the tables themselves are regenerated from raw data.

The decision-level claims (``DEC.*``) are kept in their own block: they come from
``scripts/run_decision_analysis.py`` in exact mode, with the second-opinion rule (``results/decisions_exact/``; the
audits saved their leave-two-out calibration predictions). ``results/decisions/`` holds the earlier run in
approximate mode, kept for comparison.

Usage::

    python claims.py                       # all claims
    python claims.py --block core          # without the decision-level block
    python claims.py --report claims.md    # also write a markdown report (refuses to overwrite)
    python claims.py --results <folder>    # read the tables from another folder
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
"""pathlib.Path: default folder of the committed result tables."""

HANDHELD_AUDITS: tuple[str, ...] = ("audit_ark106_2d", "audit_ark106_cutr_sam3", "audit_ark106_cutr_clip")
"""tuple of str: the handheld (ARKitScenes, 106 videos) audit folders: baseline, Cubify + SAM 3, Cubify + CLIP."""

WAREHOUSE_AUDITS: tuple[str, ...] = ("audit_ss12_2d", "audit_ss12_sparse4d_v22", "audit_ss12_sparse4d_v11")
"""tuple of str: the fixed-camera (12 synthetic warehouses) audit folders: monocular baseline, Sparse4D, Sparse4D
pre-2026."""

AIMS: dict[str, str] = {
    "A1": "Auto-label error is large and structured enough to matter for metric questions, for a published labeller "
          "and not only a rough one.",
    "A2": "Checks that need no ground truth (consistency across frames, NIS) can pass while errors against "
          "independent references are large, so independent references are necessary.",
    "A3": "How labels are paired with ground truth changes the measured error, so the pairing step must be validated.",
    "A4": "A calibrated error model and a certified gate can accept a useful share of labels with a stated error "
          "bound at the decision's tolerance, or certify the decision a label supports.",
    "A5": "A few references per site detect site-level error and correct it.",
    "A6": "Label noise changes the conclusions drawn when evaluating models, in measurable ways.",
    "A7": "Missed objects are structured and invisible to per-label audits, so they need their own guarantee: recall "
          "bounded on held-out scenes, and likely misses flagged without ground truth by disagreement.",
}
"""dict of str to str: the project aims the claims are grouped by (``data`` claims describe the evaluation)."""

Values = dict[str, float]


@dataclass(frozen=True)
class Claim:
    """One claim: what is stated, where it comes from, and how to recompute it.

    Parameters
    ----------
    id : str
        Claim id, e.g. ``"A4.5"``.
    aim : str
        The project aim it supports (``A1``-``A7``) or ``data``.
    statement : str
        The claim in plain words.
    stated : dict of str to str
        Quantity name -> value as stated (a number written with its decimals, a unit fraction, or a boolean).
    sources : tuple of str
        Result tables, relative to the results folder.
    script : str
        The repository script that wrote those tables.
    compute : callable
        ``compute(results) -> {quantity: value}`` in the units of ``stated`` (percent where stated in percent).
    block : str, default "core"
        ``"core"`` or ``"decisions"``.
    """

    id: str
    aim: str
    statement: str
    stated: dict[str, str]
    sources: tuple[str, ...]
    script: str
    compute: Callable[[Path], Values]
    block: str = "core"


# --------------------------------------------------------------------------------------------------------
# Checking a stated value
# --------------------------------------------------------------------------------------------------------


def matches(stated: str, value: float) -> bool:
    """Whether a recomputed value agrees with a value as stated.

    Parameters
    ----------
    stated : str
        A number (its decimals set the tolerance: half a unit of the last digit), a unit fraction such as
        ``"1/6"``, or ``"True"`` / ``"False"``.
    value : float
        The recomputed value (booleans as 0 or 1).

    Returns
    -------
    bool
        True if the value rounds to the stated one.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    if stated in ("True", "False"):
        return bool(value) == (stated == "True")
    if "/" in stated:
        num, den = (float(x) for x in stated.split("/"))
        return value > 0 and round(1.0 / value) == round(den / num)
    decimals = len(stated.split(".")[1]) if "." in stated else 0
    return abs(float(stated) - float(value)) <= 0.5 * 10.0 ** -decimals + 1e-9


def fmt(value: float) -> str:
    """Format a recomputed value for the report.

    Parameters
    ----------
    value : float
        The value.

    Returns
    -------
    str
        Four significant decimals, or the value as is when it is a whole number or not finite.
    """
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if not np.isfinite(value):
        return str(value)
    return f"{value:.4f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))


# --------------------------------------------------------------------------------------------------------
# Readers of the committed tables
# --------------------------------------------------------------------------------------------------------


def _csv(results: Path, rel: str) -> pd.DataFrame:
    """Read one committed table.

    Parameters
    ----------
    results : pathlib.Path
        Results folder.
    rel : str
        Path of the table relative to it.

    Returns
    -------
    pandas.DataFrame
        The table.
    """
    return pd.read_csv(results / rel, low_memory=False)


def _one(frame: pd.DataFrame, what: str) -> pd.Series:
    """The single row of a filtered table.

    Parameters
    ----------
    frame : pandas.DataFrame
        The filtered table.
    what : str
        Description for the error message.

    Returns
    -------
    pandas.Series
        The row.

    Raises
    ------
    LookupError
        If the filter left no row or several.
    """
    if len(frame) != 1:
        raise LookupError(f"{what}: expected one row, found {len(frame)}")
    return frame.iloc[0]


def gate(results: Path, audit: str, answer_kind: str, kind: str, tau_scale: int) -> pd.Series:
    """One row of an audit's gate table (the labeller's own dataset, not the pooled ``all`` rows).

    Parameters
    ----------
    results : pathlib.Path
        Results folder.
    audit : str
        Audit folder name.
    answer_kind : str
        ``"numeric"`` or ``"categorical"``.
    kind : str
        Model kind: ``sigma`` (feature model), ``const``, ``prop``, ``logistic``.
    tau_scale : int
        Tolerance multiple (1 or 2).

    Returns
    -------
    pandas.Series
        ``accept_yield``, ``accept_err``, ``err_all``, ``certified_ok`` and the other gate columns.
    """
    g = _csv(results, f"{audit}/gate.csv")
    sel = g[(g["dataset"] != "all") & (g["answer_kind"] == answer_kind) & (g["kind"] == kind)
            & (g["tau_scale"].astype(float) == float(tau_scale))]
    return _one(sel, f"{audit} gate {answer_kind}/{kind}/{tau_scale}")


def coverage(results: Path, audit: str, by: str = "all", method: str = "sigma", level: float = 0.9) -> pd.DataFrame:
    """Coverage rows of an audit's conformal intervals (the labeller's own dataset).

    Parameters
    ----------
    results : pathlib.Path
        Results folder.
    audit : str
        Audit folder name.
    by : str, default "all"
        Grouping: ``all``, ``sigma_tercile``, ``qtype`` or ``depth_source``.
    method : str, default "sigma"
        Interval method.
    level : float, default 0.9
        Nominal coverage.

    Returns
    -------
    pandas.DataFrame
        The matching rows (``coverage``, ``scene_min``, ``half_over_tau_p50``, ...).
    """
    c = _csv(results, f"{audit}/coverage.csv")
    return c[(c["dataset"] != "all") & (c["by"] == by) & (c["method"] == method)
             & np.isclose(c["level"].astype(float), level)]


def recall(results: Path, misses: str, labeller: str, source: str, population: str, by: str = "all",
           group: str = "all") -> float:
    """Recall of one labeller from a miss audit.

    Parameters
    ----------
    results : pathlib.Path
        Results folder.
    misses : str
        Miss-audit folder under ``misses/``.
    labeller, source : str
        Labeller and source as stored.
    population : str
        ``present`` or ``clearly_visible``.
    by, group : str, default "all"
        Stratum and its group.

    Returns
    -------
    float
        Paired / present.
    """
    r = _csv(results, f"misses/{misses}/recall.csv")
    row = _one(r[(r["labeller"] == labeller) & (r["source"] == source) & (r["population"] == population)
                 & (r["by"] == by) & (r["group"].astype(str) == group)], f"{misses} recall {labeller}")
    return float(row["recall"])


def flags(results: Path, misses: str, labeller: str, source: str, witness: str, kind: str) -> pd.Series:
    """One row (``by == all``) of a miss audit's flag metrics.

    Parameters
    ----------
    results : pathlib.Path
        Results folder.
    misses : str
        Miss-audit folder under ``misses/``.
    labeller, source : str
        The flagged labeller and source.
    witness : str
        What raised the flags.
    kind : str
        Flag kind (``cross_labeller``, ``cross_frame``, ``cross_camera``).

    Returns
    -------
    pandas.Series
        ``precision``, ``flag_recall``, ``flag_recall_clear``, ``flags_per_view``, ...
    """
    f = _csv(results, f"misses/{misses}/flag_metrics.csv")
    return _one(f[(f["by"] == "all") & (f["labeller"] == labeller) & (f["source"] == source)
                  & (f["witness"] == witness) & (f["kind"] == kind)], f"{misses} flags {labeller}/{witness}")


def operating_point(results: Path, labeller: str, answer_kind: str, tau_scale: int, order: str = "ours"
                    ) -> pd.Series:
    """One certified operating point of the review-effort analysis.

    Parameters
    ----------
    results : pathlib.Path
        Results folder.
    labeller : str
        Review-effort labeller name (e.g. ``ark106_2d``: the baseline on iPad LiDAR).
    answer_kind : str
        ``numeric``, ``categorical`` or ``combined``.
    tau_scale : int
        Tolerance multiple.
    order : str, default "ours"
        Whose gate.

    Returns
    -------
    pandas.Series
        ``wrong_share``, ``accepted_share``, ``wrong_among_accepted``, ...
    """
    p = _csv(results, "review_effort/operating_points.csv")
    return _one(p[(p["labeller"] == labeller) & (p["answer_kind"] == answer_kind)
                  & (p["tau_scale"].astype(float) == float(tau_scale)) & (p["order"] == order)],
                f"operating point {labeller}")


def review_target(results: Path, labeller: str, target: float, order: str) -> pd.Series:
    """Review needed by one order to reach one residual target (all checked answers, at the tolerance).

    Parameters
    ----------
    results : pathlib.Path
        Results folder.
    labeller : str
        Review-effort labeller name.
    target : float
        Residual wrong share.
    order : str
        ``ours`` (risk order), ``random``, ``const`` or ``ideal``.

    Returns
    -------
    pandas.Series
        ``review_fraction``, ``minutes_30s``, ``minutes_all_30s``, ...
    """
    t = _csv(results, "review_effort/effort_targets.csv")
    return _one(t[(t["labeller"] == labeller) & (t["answer_kind"] == "combined")
                  & (t["tau_scale"].astype(float) == 1.0) & np.isclose(t["target"], target) & (t["order"] == order)],
                f"review target {labeller}/{target}/{order}")


def delta_delta(results: Path, labeller: str, model_b: str) -> pd.Series:
    """The pooled VLM-evaluation contrast of Qwen3-VL-8B against ``model_b`` under one labeller.

    Parameters
    ----------
    results : pathlib.Path
        Results folder.
    labeller : str
        VLM-results folder under ``vlm_results/``.
    model_b : str
        ``cosmos-reason2-8b`` (the model ranking) or ``qwen3-vl-8b+blind`` (the benefit of the image).

    Returns
    -------
    pandas.Series
        ``delta_delta``, ``ci_low``, ``ci_high``, ``same_sign``, ...
    """
    d = _csv(results, f"vlm_results/{labeller}/delta_delta.csv")
    return _one(d[(d["unit"] == "pooled/deployment") & (d["model_a"] == "qwen3-vl-8b") & (d["model_b"] == model_b)
                  & (d["item_set"] == "all") & (d["qtype"] == "all") & (d["forced_policy"] == "counted")],
                f"{labeller} delta-delta vs {model_b}")


def by_alpha(results: Path, run: str) -> pd.DataFrame:
    """The per-alpha table of one recall-control (conformal risk control) run.

    Parameters
    ----------
    results : pathlib.Path
        Results folder.
    run : str
        Run folder under ``recall_control/``.

    Returns
    -------
    pandas.DataFrame
        One row per alpha.
    """
    return _csv(results, f"recall_control/{run}/by_alpha.csv").set_index("alpha")


DECISIONS = "decisions_exact"
"""str: folder under ``results/`` of the decision-level tables (exact mode, with the second-opinion rule)."""


def decision(results: Path, table: str, **where: object) -> pd.Series:
    """One row of a decision-analysis table, selected by column values (NaN-safe for ``u``).

    Parameters
    ----------
    results : pathlib.Path
        Results folder.
    table : str
        File name under :data:`DECISIONS`.
    **where : object
        Column -> required value; ``None`` selects missing values.

    Returns
    -------
    pandas.Series
        The row.
    """
    d = _csv(results, f"{DECISIONS}/{table}")
    for col, val in where.items():
        if val is None:
            d = d[d[col].isna()]
        elif isinstance(val, float):
            d = d[np.isclose(d[col].astype(float), val)]
        else:
            d = d[d[col].astype(str) == str(val)]
    return _one(d, f"{table} {where}")


def pct(x: float) -> float:
    """A share as a percentage.

    Parameters
    ----------
    x : float
        Share in [0, 1].

    Returns
    -------
    float
        ``100 * x``.
    """
    return 100.0 * float(x)


def span(values: Iterable[float]) -> Values:
    """The smallest and largest of several values.

    Parameters
    ----------
    values : iterable of float
        Values.

    Returns
    -------
    dict
        ``{"lo": min, "hi": max}``.
    """
    v = [float(x) for x in values]
    return {"lo": min(v), "hi": max(v)}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion.

    Parameters
    ----------
    k : int
        Successes.
    n : int
        Trials (positive).
    z : float, default 1.96
        Normal quantile (95%).

    Returns
    -------
    tuple of float
        ``(low, high)``.
    """
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# --------------------------------------------------------------------------------------------------------
# The claims
# --------------------------------------------------------------------------------------------------------

AUDIT = "scripts/run_audit.py"
MISSES = "scripts/run_miss_audit.py"
EFFORT = "scripts/run_review_effort.py"
DECIDE = "scripts/run_decision_analysis.py"


def _a1_wrong_range(r: Path) -> Values:
    """Handheld wrong shares before gating: baseline (iPad LiDAR), Cubify + SAM 3, Cubify + CLIP (percent)."""
    return span([pct(operating_point(r, "ark106_2d", "numeric", 1)["wrong_share"]),
                 pct(gate(r, "audit_ark106_cutr_sam3", "numeric", "sigma", 1)["err_all"]),
                 pct(gate(r, "audit_ark106_cutr_clip", "numeric", "sigma", 1)["err_all"])])


def _a1_wrong_table(r: Path) -> Values:
    """Table "Wrong" column: share of numeric labels outside the tolerance, per labeller (percent)."""
    return {"baseline": pct(operating_point(r, "ark106_2d", "numeric", 1)["wrong_share"]),
            "cubify_sam3": pct(gate(r, "audit_ark106_cutr_sam3", "numeric", "sigma", 1)["err_all"]),
            "cubify_clip": pct(gate(r, "audit_ark106_cutr_clip", "numeric", "sigma", 1)["err_all"]),
            "monocular": pct(gate(r, "audit_ss12_2d", "numeric", "sigma", 1)["err_all"]),
            "sparse4d": pct(gate(r, "audit_ss12_sparse4d_v22", "numeric", "sigma", 1)["err_all"]),
            "sparse4d_pre2026": pct(gate(r, "audit_ss12_sparse4d_v11", "numeric", "sigma", 1)["err_all"])}


def _a1_raw_cover(r: Path) -> Values:
    """Share of present objects covered by Cubify's raw boxes at its published 0.25 threshold (percent)."""
    c = _csv(r, "recall_control/ark106_t005_present_mark/curve.csv")
    return {"covered": pct(1.0 - float(_one(c[np.isclose(c["threshold"], 0.25)], "threshold 0.25")["mean_fnr"]))}


def _a1_clip(r: Path) -> Values:
    """CLIP-named Cubify: recall of present objects, and wrong categorical answers before gating (percent)."""
    return {"recall": pct(recall(r, "ark106_cutr", "cutr_clip", "cutr_clip", "present")),
            "categorical_wrong": pct(gate(r, "audit_ark106_cutr_clip", "categorical", "logistic", 1)["err_all"])}


def _a1_reproduction(r: Path) -> Values:
    """Sparse4D v2.2 on val scene 022, footprint (BEV) HOTA evaluation: pooled DetA."""
    d = json.loads((r / "sparse4d_eval/stage1_w022_rn50_v2.2_rgb_hota_bev.json").read_text())
    return {"DetA": float(d["scenes"]["22"]["det_pooled"]["DetA"])}


def _a1_collapse(r: Path) -> Values:
    """Official DetA on the scenes sharing a larger layout: pre-2026 checkpoint against the current one."""
    d = json.loads((r / "sparse4d_eval/summary_stage2.json").read_text())
    key = "shared_layout(000-006,020)"
    return {"pre2026": float(d["rn101_trainable_v1.1"]["groups"][key]["official"]["DetA"]),
            "current": float(d["rn50_trainable_v2.2"]["groups"][key]["official"]["DetA"])}


def _a1_incomplete(r: Path) -> Values:
    """Pre-2026 Sparse4D: share of present objects missed, overall and beyond 45 m (percent)."""
    return {"missed": pct(1 - recall(r, "ss12_sparse4d_v11", "sparse4d", "sparse4d", "present")),
            "missed_beyond_45m": pct(1 - recall(r, "ss12_sparse4d_v11", "sparse4d", "sparse4d", "present",
                                                "range", "45m_plus"))}


def _a2_consistency(r: Path) -> Values:
    """Mean NIS and share of objects above the chi-square 95% point (naive fusion), and median single-frame NEES,
    against the annotated boxes, baseline (iPad LiDAR) and Cubify."""
    v0 = json.loads((r / "fusion/fusion_ark106_2d/summary.json").read_text())["by_source"]["lidar"]
    cu = json.loads((r / "fusion/fusion_ark106_cutr_sam3/summary.json").read_text())["by_source"]["cutr_sam3"]
    s0 = _one(_csv(r, "fusion/fusion_ark106_2d/summary.csv").query("source == 'lidar' and estimator == 'single'"),
              "fusion lidar single")
    s1 = _one(_csv(r, "fusion/fusion_ark106_cutr_sam3/summary.csv").query("estimator == 'single'"),
              "fusion cutr single")
    f0 = _one(_csv(r, "fusion/fusion_ark106_2d/summary.csv").query("source == 'lidar' and estimator == 'naive'"),
              "fusion lidar naive")
    f1 = _one(_csv(r, "fusion/fusion_ark106_cutr_sam3/summary.csv").query("estimator == 'naive'"),
              "fusion cutr naive")
    outside = (float(f0["frac_nees_above_chi2_95"]), float(f1["frac_nees_above_chi2_95"]))
    return {"nis_baseline": float(v0["nis_mean"]["naive"]), "nis_cubify": float(cu["nis_mean"]["naive"]),
            "nees_baseline": float(s0["nees_median"]), "nees_cubify": float(s1["nees_median"]),
            "band_lo": float(s0["nees_lower"]), "band_hi": float(s0["nees_upper"]),
            "outside_lo": pct(min(outside)), "outside_hi": pct(max(outside))}


SPOTCHECK_BASELINE = "0e8b5473ce53"
"""str: sample id of the manual spot check of the baseline's (iPad LiDAR) pairs."""
SPOTCHECK_CUBIFY = "4e824803ceb4"
"""str: sample id of the manual spot check of Cubify + SAM 3's pairs."""


def _spotcheck(r: Path, sample: str = SPOTCHECK_BASELINE) -> dict:
    """A manual spot check's export.

    Parameters
    ----------
    r : pathlib.Path
        Results folder.
    sample : str, default :data:`SPOTCHECK_BASELINE`
        Sample id of the export.

    Returns
    -------
    dict
        The parsed export.
    """
    return json.loads((r / f"spotcheck/spotcheck_{sample}.json").read_text())


def _a3_ambiguous(r: Path) -> Values:
    """Share of the baseline's (iPad LiDAR) pairs that are ambiguous (percent)."""
    pop = _spotcheck(r)["population"]
    return {"ambiguous": pct(pop["ambiguous"] / (pop["ambiguous"] + pop["clear"]))}


def _a3_exclude(r: Path) -> Values:
    """Cubify + SAM 3 categorical acceptance at the tolerance, with and without ambiguous pairs (percent)."""
    return {"with": pct(gate(r, "audit_ark106_cutr_sam3", "categorical", "logistic", 1)["accept_yield"]),
            "without": pct(gate(r, "audit_ark106_cutr_sam3_noamb", "categorical", "logistic", 1)["accept_yield"])}


def _spot(r: Path, sample: str, part_of: tuple[int, ...]) -> Values:
    """A manual spot check: wrong pairs among decided ones, their Wilson interval, and the undecided cards.

    Parameters
    ----------
    r : pathlib.Path
        Results folder.
    sample : str
        Sample id of the export.
    part_of : tuple of int
        Cards (1-based positions in the export) whose notes put the label on the right object at another level of
        detail; they are counted as right.

    Returns
    -------
    dict
        Pairs, wrong, decided, wrong share and its interval (percent), undecided cards, part-of cards marked right.
    """
    answers = _spotcheck(r, sample)["answers"]
    decided = [a for a in answers if a["verdict"] in ("right", "wrong")]
    wrong = [a for a in decided if a["verdict"] == "wrong"]
    lo, hi = wilson(len(wrong), len(decided))
    return {"pairs": float(len(answers)), "wrong": float(len(wrong)), "decided": float(len(decided)),
            "wrong_pct": pct(len(wrong) / len(decided)), "ci_lo": pct(lo), "ci_hi": pct(hi),
            "undecided": float(sum(a["verdict"] == "cant_tell" for a in answers)),
            "part_of_right": float(sum(answers[c - 1]["verdict"] == "right" for c in part_of))}


def _a3_spot(r: Path) -> Values:
    """The baseline's spot check, and whether every wrong pair was flagged ambiguous."""
    answers = _spotcheck(r)["answers"]
    return {**_spot(r, SPOTCHECK_BASELINE, (9, 13, 30, 41, 61, 74, 80, 81, 86)),
            "wrong_all_ambiguous": float(all(a["ambiguous"] for a in answers if a["verdict"] == "wrong"))}


def _a3_spot_cubify(r: Path) -> Values:
    """Cubify + SAM 3's spot check."""
    return _spot(r, SPOTCHECK_CUBIFY, (2, 13, 19, 75, 85, 88, 97))


def _a4_coverage(r: Path) -> Values:
    """Held-out coverage of the 90% intervals (feature model), across all six labellers."""
    return span(float(coverage(r, a)["coverage"].iloc[0]) for a in HANDHELD_AUDITS + WAREHOUSE_AUDITS)


def _a4_adaptive(r: Path) -> Values:
    """Coverage by tercile of predicted spread: feature model and constant width, across all labellers."""
    audits = HANDHELD_AUDITS + WAREHOUSE_AUDITS
    feat = [v for a in audits for v in coverage(r, a, "sigma_tercile")["coverage"]]
    const = [v for a in audits for v in coverage(r, a, "sigma_tercile", "const")["coverage"]]
    return {"feature_lo": min(feat), "feature_hi": max(feat), "const_lo": min(const), "const_hi": max(const)}


def _a4_sites(r: Path) -> Values:
    """Worst single-scene coverage (scenes with at least 30 labels): handheld and Sparse4D, and monocular."""
    worst = span(float(coverage(r, a)["scene_min"].iloc[0])
                 for a in HANDHELD_AUDITS + ("audit_ss12_sparse4d_v22", "audit_ss12_sparse4d_v11"))
    return {**worst, "monocular": float(coverage(r, "audit_ss12_2d")["scene_min"].iloc[0])}


def _a4_width(r: Path) -> Values:
    """Median half-width of the 90% interval in tolerances: handheld range, and current Sparse4D."""
    w = span(float(coverage(r, a)["half_over_tau_p50"].iloc[0]) for a in HANDHELD_AUDITS)
    return {**w, "sparse4d": float(coverage(r, "audit_ss12_sparse4d_v22")["half_over_tau_p50"].iloc[0])}


def _a4_table_gates(r: Path) -> Values:
    """Table gate columns: accepted share and wrong among accepted, at the tolerance and twice it (percent).

    The baseline's cells come from its iPad-LiDAR-only audit, certified on LiDAR answers alone.
    """
    out: Values = {}
    rows = {"baseline_t1": ("audit_ark106_2d_lidar", "sigma", 1), "baseline_t2": ("audit_ark106_2d_lidar", "sigma", 2),
            "cubify_sam3_t1": ("audit_ark106_cutr_sam3", "const", 1),
            "cubify_sam3_t2": ("audit_ark106_cutr_sam3", "sigma", 2),
            "cubify_sam3_feature_t1": ("audit_ark106_cutr_sam3", "sigma", 1),
            "cubify_clip_t1": ("audit_ark106_cutr_clip", "sigma", 1),
            "cubify_clip_t2": ("audit_ark106_cutr_clip", "sigma", 2),
            "monocular_t1": ("audit_ss12_2d", "sigma", 1), "monocular_t2": ("audit_ss12_2d", "sigma", 2),
            "sparse4d_t1": ("audit_ss12_sparse4d_v22", "sigma", 1),
            "sparse4d_t2": ("audit_ss12_sparse4d_v22", "sigma", 2),
            "pre2026_t1": ("audit_ss12_sparse4d_v11", "sigma", 1),
            "pre2026_t2": ("audit_ss12_sparse4d_v11", "sigma", 2)}
    for name, (audit, kind, t) in rows.items():
        g = gate(r, audit, "numeric", kind, t)
        out[f"{name}_acc"] = pct(g["accept_yield"])
        if float(g["accept_yield"]) > 0:
            out[f"{name}_err"] = pct(g["accept_err"])
    return out


def _a4_handheld_gates(r: Path) -> Values:
    """Handheld gates as summarised in the text: acceptance range at the tolerance and twice it, and worst error.

    The baseline is its iPad-LiDAR-only audit.
    """
    t1 = [gate(r, "audit_ark106_2d_lidar", "numeric", "sigma", 1),
          gate(r, "audit_ark106_cutr_sam3", "numeric", "const", 1),
          gate(r, "audit_ark106_cutr_clip", "numeric", "sigma", 1)]
    acc1 = [pct(g["accept_yield"]) for g in t1]
    err1 = [pct(g["accept_err"]) for g in t1]
    acc2 = [pct(gate(r, "audit_ark106_2d_lidar", "numeric", "sigma", 2)["accept_yield"]),
            pct(gate(r, "audit_ark106_cutr_sam3", "numeric", "sigma", 2)["accept_yield"]),
            pct(gate(r, "audit_ark106_cutr_clip", "numeric", "sigma", 2)["accept_yield"])]
    return {"t1_lo": min(acc1), "t1_hi": max(acc1), "t1_worst_err": max(err1), "t2_lo": min(acc2), "t2_hi": max(acc2)}


def _a5_references(r: Path) -> Values:
    """Monocular warehouse baseline at twice the tolerance: without and with ten random references per site."""
    s = _csv(r, "references/ss12_2d/summary.csv")
    row = _one(s[(s["variant"] == "abc_uniform") & (s["n_refs"] == 10) & (s["tau_scale"] == 2)], "refs n=10 2tau")
    return {"before_acc": pct(row["n0_yield_pooled"]), "before_err": pct(row["n0_err_pooled"]),
            "after_acc": pct(row["yield_pooled"]), "after_err": pct(row["err_pooled"]),
            "sites_missing_bound": 1.0 - float(row["frac_ok"])}


def _a6_ranking(r: Path) -> Values:
    """Whether the Qwen3-VL-8B vs Cosmos-Reason2-8B ranking keeps its sign under every labeller's labels."""
    names = ("ark106_2d", "ark106_cutr_sam3", "ark106_cutr_clip", "ss12_2d", "ss12_sparse4d_v22", "ss12_sparse4d_v11")
    return {"holds_everywhere": float(all(bool(delta_delta(r, n, "cosmos-reason2-8b")["same_sign"]) for n in names))}


def _a6_image_benefit(r: Path) -> Values:
    """Change in the measured benefit of the image (Qwen with vs without it), auto labels against references."""
    v0 = delta_delta(r, "ark106_2d", "qwen3-vl-8b+blind")
    clip = delta_delta(r, "ark106_cutr_clip", "qwen3-vl-8b+blind")
    sam3 = delta_delta(r, "ark106_cutr_sam3", "qwen3-vl-8b+blind")
    return {"baseline": float(v0["delta_delta"]), "baseline_lo": float(v0["ci_low"]),
            "baseline_hi": float(v0["ci_high"]), "clip": float(clip["delta_delta"]),
            "sam3": float(sam3["delta_delta"]), "sam3_lo": float(sam3["ci_low"]), "sam3_hi": float(sam3["ci_high"])}


def _a6_exact(r: Path) -> Values:
    """Current Sparse4D's labels: change in the model-ranking gap against references, with its interval."""
    d = delta_delta(r, "ss12_sparse4d_v22", "cosmos-reason2-8b")
    return {"dd": float(d["delta_delta"]), "lo": float(d["ci_low"]), "hi": float(d["ci_high"])}


def _a7_recall_table(r: Path) -> Values:
    """Table recall column: present / clearly visible, per labeller."""
    spec = {"baseline": ("ark106_v0", "v0", "lidar"), "cubify_sam3": ("ark106_cutr", "cutr_sam3", "cutr_sam3"),
            "cubify_clip": ("ark106_cutr", "cutr_clip", "cutr_clip"), "monocular": ("ss12_v0", "v0", "moge"),
            "sparse4d": ("ss12_sparse4d_v22", "sparse4d", "sparse4d"),
            "pre2026": ("ss12_sparse4d_v11", "sparse4d", "sparse4d")}
    out: Values = {}
    for name, (folder, lab, src) in spec.items():
        out[f"{name}_present"] = recall(r, folder, lab, src, "present")
        out[f"{name}_clear"] = recall(r, folder, lab, src, "clearly_visible")
    return out


def _a7_visibility(r: Path) -> Values:
    """Baseline (iPad LiDAR) recall of objects under 10% visible and over 60% visible (percent)."""
    return {"under_10pct": pct(recall(r, "ark106_v0", "v0", "lidar", "present", "visible_fraction", "0-0.1")),
            "over_60pct": pct(recall(r, "ark106_v0", "v0", "lidar", "present", "visible_fraction", "0.6-1"))}


def _a7_crc(r: Path) -> Values:
    """Conformal risk control on Cubify's clearly visible objects: certifiable alphas and candidates per image."""
    b = by_alpha(r, "ark106_t005_clearly_visible_mark")
    out: Values = {}
    for a in (0.05, 0.1, 0.2):
        out[f"feasible_{a:g}"] = float(b.loc[a, "share_feasible"] == 1)
        out[f"boxes_per_image_{a:g}"] = float(b.loc[a, "boxes_per_view"])
    return out


def _a7_sites(r: Path) -> Values:
    """Lowest certifiable per-scene miss rate (percent) with the 12 warehouse sites, clearly visible objects."""
    b = by_alpha(r, "ss12_s4d_v22_clearly_visible")
    return {"lowest_alpha": pct(min(a for a in b.index if b.loc[a, "share_feasible"] == 1))}


def _a7_average(r: Path) -> Values:
    """Share of scenes whose miss rate exceeds alpha = 0.2 under the recall guarantee (percent)."""
    return span([pct(by_alpha(r, "ark106_t005_clearly_visible_mark").loc[0.2, "share_scenes_above_alpha"]),
                 pct(by_alpha(r, "ark106_t005_present_mark").loc[0.2, "share_scenes_above_alpha"])])


def _a7_flags(r: Path) -> Values:
    """Disagreement flags: share of clearly visible misses flagged, handheld and warehouse, and warehouse precision."""
    hand = flags(r, "ark106_v0", "v0", "lidar", "cutr_t025", "cross_labeller")
    ware = flags(r, "ss12_sparse4d_v11", "s4d", "boxes", "v0/moge", "cross_labeller")
    return {"handheld_clear": pct(hand["flag_recall_clear"]), "warehouse_clear": pct(ware["flag_recall_clear"]),
            "warehouse_precision": pct(ware["precision"]), "handheld_flags_per_image": float(hand["flags_per_view"]),
            "handheld_unannotated": pct(hand["unannotated_share"])}


def _a4_violations(r: Path) -> Values:
    """Gates (all model kinds, both tolerances, non-empty) whose accepted labels exceed epsilon on held-out scenes.

    Counts over the six audits; the two largest violations by number of labels accepted (their counts and errors),
    with whether they are on the warehouse scenes; the range of error among accepted over Table 2's handheld cells
    (baseline: its LiDAR-only audit); the scene-bootstrap upper bound of Cubify + SAM 3's constant-sigma gate at the
    tolerance (percent).
    """
    rows = []
    for a in HANDHELD_AUDITS + WAREHOUSE_AUDITS:
        g = _csv(r, f"{a}/gate.csv")
        g = g[(g["dataset"] != "all") & ~g["kind"].astype(str).str.contains("conformal")
              & (g["accept_n"].astype(float) > 0)]
        rows.append(g.assign(audit=a))
    g = pd.concat(rows, ignore_index=True)
    bad = g[g["accept_err"].astype(float) > g["epsilon"].astype(float)].sort_values("accept_n", ascending=False)
    top = bad.head(2)
    cells = [gate(r, a, "numeric", k, t) for a, k, t in (
        ("audit_ark106_cutr_sam3", "const", 1), ("audit_ark106_cutr_sam3", "sigma", 2),
        ("audit_ark106_cutr_clip", "sigma", 1), ("audit_ark106_cutr_clip", "sigma", 2),
        ("audit_ark106_2d_lidar", "sigma", 1), ("audit_ark106_2d_lidar", "sigma", 2))]
    return {"violated": float(len(bad)), "gates": float(len(g)),
            "largest_n": float(top["accept_n"].iloc[0]), "largest_err": pct(float(top["accept_err"].iloc[0])),
            "second_n": float(top["accept_n"].iloc[1]), "second_err": pct(float(top["accept_err"].iloc[1])),
            "largest_two_warehouse": float(all(a in WAREHOUSE_AUDITS for a in top["audit"])),
            "handheld_lo": pct(min(float(c["accept_err"]) for c in cells)),
            "handheld_hi": pct(max(float(c["accept_err"]) for c in cells)),
            "boot_hi_sam3_tau": pct(float(cells[0]["accept_boot_hi"]))}


def _a4_layouts(r: Path) -> Values:
    """Warehouse audits cross-fitted with one fold per building layout (leave one layout out).

    Sparse4D v2.2: numeric coverage of the 90% intervals and the feature gate at the tolerance. Monocular baseline:
    the feature gate at twice the tolerance (labels accepted, wrong among them). Pre-2026 Sparse4D: the
    constant-sigma gate at the tolerance. Per audit, non-empty gates (all model kinds, both tolerances) whose
    accepted labels exceed epsilon.
    """
    def violated(audit: str) -> float:
        g = _csv(r, f"{audit}/gate.csv")
        g = g[(g["dataset"] != "all") & ~g["kind"].astype(str).str.contains("conformal")
              & (g["accept_n"].astype(float) > 0)]
        return float((g["accept_err"].astype(float) > g["epsilon"].astype(float)).sum())
    v22 = gate(r, "audit_ss12_sparse4d_v22_bylayout", "numeric", "sigma", 1)
    mono = gate(r, "audit_ss12_2d_bylayout", "numeric", "sigma", 2)
    old = gate(r, "audit_ss12_sparse4d_v11_bylayout", "numeric", "const", 1)
    return {"v22_coverage": float(coverage(r, "audit_ss12_sparse4d_v22_bylayout")["coverage"].iloc[0]),
            "v22_acc": pct(v22["accept_yield"]), "v22_err": pct(v22["accept_err"]),
            "mono_n": float(mono["accept_n"]), "mono_err": pct(mono["accept_err"]),
            "old_n": float(old["accept_n"]), "old_err": pct(old["accept_err"]),
            "violated_v22": violated("audit_ss12_sparse4d_v22_bylayout"),
            "violated_mono": violated("audit_ss12_2d_bylayout"),
            "violated_old": violated("audit_ss12_sparse4d_v11_bylayout")}


def _a4_effort(r: Path) -> Values:
    """Review needed for a stated quality, risk order against random (percent), and reviewer-minutes at 30 s."""
    ours = review_target(r, "ss12_sparse4d_v22", 0.01, "ours")
    rnd = review_target(r, "ss12_sparse4d_v22", 0.01, "random")
    by_type = review_target(r, "ss12_sparse4d_v22", 0.01, "const")
    hand_ours = [review_target(r, n, 0.02, "ours")["review_fraction"] for n in
                 ("ark106_2d", "ark106_cutr_sam3", "ark106_cutr_clip")]
    hand_rnd = [review_target(r, n, 0.02, "random")["review_fraction"] for n in
                ("ark106_2d", "ark106_cutr_sam3", "ark106_cutr_clip")]
    return {"sparse4d_ours": pct(ours["review_fraction"]), "sparse4d_random": pct(rnd["review_fraction"]),
            "sparse4d_by_type": pct(by_type["review_fraction"]), "minutes_by_type": float(by_type["minutes_30s"]),
            "minutes_ours": float(ours["minutes_30s"]), "minutes_random": float(rnd["minutes_30s"]),
            "minutes_all": float(ours["minutes_all_30s"]),
            "ratio": float(ours["review_fraction"]) / float(rnd["review_fraction"]),
            "handheld_ours_lo": pct(min(hand_ours)), "handheld_ours_hi": pct(max(hand_ours)),
            "handheld_random_lo": pct(min(hand_rnd)), "handheld_random_hi": pct(max(hand_rnd))}


def _a7_flag_cost(r: Path) -> Values:
    """Warehouse miss flags raised by the monocular baseline on Sparse4D: flags and recovered misses per image."""
    m = _csv(r, "review_effort/miss_review.csv")
    rows = m[(m["flagged_labeller"] == "s4d") & (m["flagged_source"] == "boxes") & (m["witness"] == "v0/moge")]
    if len(rows) != 2:
        raise LookupError(f"miss review: expected 2 Sparse4D cross-labeller rows, found {len(rows)}")
    return {"flags_lo": float(rows["flags_per_view"].min()), "flags_hi": float(rows["flags_per_view"].max()),
            "recovered_lo": float(rows["recovered_per_1000_images"].min()) / 1000.0,
            "recovered_hi": float(rows["recovered_per_1000_images"].max()) / 1000.0}


def _data_scale(r: Path) -> Values:
    """Size of the evaluation: handheld videos and views, and warehouse scenes."""
    s = json.loads((r / "misses/ark106_v0/summary.json").read_text())
    views = _one(_csv(r, "misses/ark106_cutr/flag_metrics.csv").query("by == 'all' and kind == 'cross_frame'")
                 .head(1), "ark106 views")["n_views"]
    return {"videos": float(_find_int(s, "n_scenes")), "views": float(views),
            "warehouses": float(coverage(r, "audit_ss12_2d")["n_scenes"].iloc[0])}


def _find_int(obj: object, key: str) -> int:
    """The first integer stored under ``key`` anywhere in a parsed JSON document.

    Parameters
    ----------
    obj : object
        Parsed JSON.
    key : str
        Key to look for.

    Returns
    -------
    int
        Its value.

    Raises
    ------
    KeyError
        If no such key holds an integer.
    """
    stack = [obj]
    while stack:
        o = stack.pop(0)
        if isinstance(o, dict):
            if isinstance(o.get(key), int):
                return int(o[key])
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)
    raise KeyError(key)


# Decision-level block ---------------------------------------------------------------------------------------

HANDHELD_DECIDERS: tuple[str, ...] = ("2d_lidar", "cutr_sam3", "cutr_clip")
"""tuple of str: the decision analysis's handheld labeller names."""


def _dec_handheld(r: Path) -> Values:
    """Handheld threshold decisions certified at epsilon 0.05: accepted share, wrong among accepted, accept-all."""
    cert = [decision(r, "decision_gate.csv", labeller=n, qtype="all", rule="certified", epsilon=0.05)
            for n in HANDHELD_DECIDERS]
    every = [decision(r, "decision_gate.csv", labeller=n, qtype="all", rule="accept_all") for n in HANDHELD_DECIDERS]
    return {"accepted_lo": pct(min(c["accept_share"] for c in cert)),
            "accepted_hi": pct(max(c["accept_share"] for c in cert)),
            "wrong_lo": pct(min(c["wrong_rate"] for c in cert)), "wrong_hi": pct(max(c["wrong_rate"] for c in cert)),
            "accept_all_lo": pct(min(e["wrong_rate"] for e in every)),
            "accept_all_hi": pct(max(e["wrong_rate"] for e in every)),
            "within_5pct": float(max(c["wrong_rate"] for c in cert) <= 0.05)}


def _dec_monocular(r: Path) -> Values:
    """Monocular warehouse threshold decisions: certified at epsilon 0.05, and the fixed-tolerance gate."""
    c = decision(r, "decision_gate.csv", labeller="2d_moge", qtype="all", rule="certified", epsilon=0.05)
    f = decision(r, "decision_gate.csv", labeller="2d_moge", qtype="all", rule="fixed_tau_gate")
    return {"accepted": pct(c["accept_share"]), "wrong": pct(c["wrong_rate"]), "fixed_gate": pct(f["accept_share"])}


def _dec_margins(r: Path) -> Values:
    """Monocular margins: direct calibration's narrowing against quadrature, and certified "which is larger?"."""
    i = decision(r, "margin_intervals.csv", labeller="2d_moge", qtype="all", method="direct")
    g = decision(r, "margin_gate.csv", labeller="2d_moge", qtype="all", decision="sign", rule="certified",
                 epsilon=0.05)
    return {"narrower": pct(1.0 - float(i["half_over_quadrature_p50"])), "coverage": float(i["coverage"]),
            "sign_accepted": pct(g["accept_share"]), "sign_wrong": pct(g["wrong_rate"])}


def _dec_close(r: Path) -> Values:
    """Largest certified share of handheld decisions whose threshold lies within 10% of the reference."""
    s = _csv(r, f"{DECISIONS}/sweep.csv")
    sel = s[s["labeller"].isin(HANDHELD_DECIDERS) & (s["qtype"] == "all") & s["u"].isna()
            & (s["rule"] == "certified") & (s["u_abs"].astype(float) <= 0.1 + 1e-9)]
    if sel.empty:
        raise LookupError("sweep: no handheld close-call rows")
    return {"accepted": pct(sel["accept_share"].max())}


def _dec_second_opinion(r: Path) -> Values:
    """Close calls certified with a second labeller's agreement: handheld at |u| = 0.1 (epsilon 0.10) and
    monocular warehouse labels witnessed by Sparse4D at |u| = 0.05 (epsilon 0.05)."""
    w = _csv(r, f"{DECISIONS}/witness.csv")
    w = w[(w["part"] == "sweep") & (w["rule"] == "second_opinion") & (w["scope"] == "all")
          & (w["qtype"].astype(str) == "all")]
    hand = w[w["labeller"].isin(HANDHELD_DECIDERS) & np.isclose(w["u_abs"].astype(float), 0.1)
             & np.isclose(w["epsilon"].astype(float), 0.10)]
    mono = _one(w[(w["labeller"] == "2d_moge") & (w["witness"] == "sparse4d_v22")
                  & np.isclose(w["u_abs"].astype(float), 0.05) & np.isclose(w["epsilon"].astype(float), 0.05)],
                "witness 2d_moge |u|=0.05")
    if len(hand) != len(HANDHELD_DECIDERS):
        raise LookupError("witness: missing handheld second-opinion rows")
    return {"handheld_acc_lo": pct(hand["accept_share"].min()), "handheld_acc_hi": pct(hand["accept_share"].max()),
            "handheld_wrong_lo": pct(hand["wrong_rate"].min()), "handheld_wrong_hi": pct(hand["wrong_rate"].max()),
            "monocular_acc": pct(mono["accept_share"]), "monocular_wrong": pct(mono["wrong_rate"])}


CLAIMS: tuple[Claim, ...] = (
    Claim("D.1", "data", "The evaluation covers 106 handheld videos (2,112 views) and 12 synthetic warehouses.",
          {"videos": "106", "views": "2112", "warehouses": "12"},
          ("misses/ark106_v0/summary.json", "misses/ark106_cutr/flag_metrics.csv", "audit_ss12_2d/coverage.csv"),
          f"{MISSES}, {AUDIT}", _data_scale),
    Claim("A1.1", "A1", "Before gating, 25-33% of handheld numeric labels are wrong at the tolerance.",
          {"lo": "25", "hi": "33"}, ("review_effort/operating_points.csv", "audit_ark106_cutr_*/gate.csv"),
          f"{EFFORT}, {AUDIT}", _a1_wrong_range),
    Claim("A1.2", "A1", "Share of numeric labels wrong at the tolerance, per labeller (results table).",
          {"baseline": "33", "cubify_sam3": "25", "cubify_clip": "26", "monocular": "55", "sparse4d": "2.7",
           "sparse4d_pre2026": "4.5"},
          ("review_effort/operating_points.csv", "audit_*/gate.csv"), f"{EFFORT}, {AUDIT}", _a1_wrong_table),
    Claim("A1.3", "A1", "Cubify Anything's raw boxes cover about 76% of present objects.", {"covered": "76"},
          ("recall_control/ark106_t005_present_mark/curve.csv",), "scripts/run_recall_control.py", _a1_raw_cover),
    Claim("A1.4", "A1", "Naming Cubify's boxes with CLIP cuts recall to 45% and leaves 60% of categorical answers "
          "wrong before gating.", {"recall": "45", "categorical_wrong": "60"},
          ("misses/ark106_cutr/recall.csv", "audit_ark106_cutr_clip/gate.csv"), f"{MISSES}, {AUDIT}", _a1_clip),
    Claim("A1.5", "A1", "Sparse4D (v2.2, frozen) reproduces DetA 42.3 on val scene 022 under the footprint-IoU metric "
          "(published reference: 42.16, from the challenge winner's report; external, not recomputed).",
          {"DetA": "42.3"}, ("sparse4d_eval/stage1_w022_rn50_v2.2_rgb_hota_bev.json",),
          "Sparse4D evaluation (REPRODUCE.md, Tier 2)", _a1_reproduction),
    Claim("A1.6", "A1", "The pre-2026 checkpoint collapses on the larger shared layout: DetA 8.5 against 55.8.",
          {"pre2026": "8.5", "current": "55.8"}, ("sparse4d_eval/summary_stage2.json",),
          "Sparse4D evaluation (REPRODUCE.md, Tier 2)", _a1_collapse),
    Claim("A1.7", "A1", "The pre-2026 checkpoint misses 40% of present objects, and 78% beyond 45 m.",
          {"missed": "40", "missed_beyond_45m": "78"}, ("misses/ss12_sparse4d_v11/recall.csv",), MISSES,
          _a1_incomplete),
    Claim("A2.1", "A2", "Fused estimates pass the ground-truth-free check (mean NIS 7.08 and 7.34 for 7 dof), yet "
          "76-79% of them fall outside their 95% error ellipse against the annotated boxes (5% expected); single-frame "
          "median NEES 27 and 45 (95% band of the mean about 6.9-7.1).",
          {"nis_baseline": "7.08", "nis_cubify": "7.34", "nees_baseline": "27", "nees_cubify": "45",
           "band_lo": "6.9", "band_hi": "7.1", "outside_lo": "76", "outside_hi": "79"},
          ("fusion/fusion_ark106_2d/summary.json", "fusion/fusion_ark106_cutr_sam3/summary.json"),
          "scripts/run_fusion_eval.py", _a2_consistency),
    Claim("A3.1", "A3", "28% of the baseline's pairs are ambiguous (a second object also overlaps).",
          {"ambiguous": "28"}, ("spotcheck/spotcheck_0e8b5473ce53.json",), "scripts/make_viewer.py", _a3_ambiguous),
    Claim("A3.2", "A3", "Excluding ambiguous pairs raises Cubify + SAM 3 categorical acceptance from 21% to 44%.",
          {"with": "21", "without": "44"},
          ("audit_ark106_cutr_sam3/gate.csv", "audit_ark106_cutr_sam3_noamb/gate.csv"), AUDIT, _a3_exclude),
    Claim("A3.3", "A3", "Manual spot check of 100 baseline pairs: 2 of 97 decided pairs on a different object (2.1%, "
          "95% CI 0.6-7.2%), both flagged ambiguous; the 9 part-of cards counted as right; 3 undecided.",
          {"pairs": "100", "wrong": "2", "decided": "97", "wrong_pct": "2.1", "ci_lo": "0.6", "ci_hi": "7.2",
           "wrong_all_ambiguous": "True", "undecided": "3", "part_of_right": "9"},
          ("spotcheck/spotcheck_0e8b5473ce53.json",), "scripts/make_viewer.py --summarize-answers", _a3_spot),
    Claim("A3.4", "A3", "Manual spot check of 100 Cubify + SAM 3 pairs: none of 98 decided pairs on a different object "
          "(95% CI 0-3.8%); the 7 part-of cards counted as right; 2 undecided.",
          {"pairs": "100", "wrong": "0", "decided": "98", "wrong_pct": "0", "ci_lo": "0", "ci_hi": "3.8",
           "undecided": "2", "part_of_right": "7"},
          ("spotcheck/spotcheck_4e824803ceb4.json",), "scripts/make_viewer.py --summarize-answers",
          _a3_spot_cubify),
    Claim("A4.1", "A4", "Held-out 90% intervals cover 0.90-0.91 of labels for every labeller.",
          {"lo": "0.90", "hi": "0.91"}, ("audit_*/coverage.csv",), AUDIT, _a4_coverage),
    Claim("A4.2", "A4", "Per-label spread keeps coverage at 0.87-0.95 across easy and hard labels; a constant width "
          "gives 0.71-1.00.", {"feature_lo": "0.87", "feature_hi": "0.95", "const_lo": "0.71", "const_hi": "1.00"},
          ("audit_*/coverage.csv",), AUDIT, _a4_adaptive),
    Claim("A4.3", "A4", "Single sites vary: worst scene 0.64-0.78, and 0.39 for monocular warehouse labels.",
          {"lo": "0.64", "hi": "0.78", "monocular": "0.39"}, ("audit_*/coverage.csv",), AUDIT, _a4_sites),
    Claim("A4.4", "A4", "Median interval half-widths are 2-3 tolerances on handheld scans and 0.1 for Sparse4D.",
          {"lo": "2", "hi": "3", "sparse4d": "0.1"}, ("audit_*/coverage.csv",), AUDIT, _a4_width),
    Claim("A4.5", "A4", "Results table, gate columns: accepted share / wrong among accepted (percent).",
          {"baseline_t1_acc": "11", "baseline_t1_err": "6.9", "baseline_t2_acc": "48", "baseline_t2_err": "7.4",
           "cubify_sam3_t1_acc": "11", "cubify_sam3_t1_err": "7.5", "cubify_sam3_t2_acc": "79",
           "cubify_sam3_t2_err": "8.9", "cubify_sam3_feature_t1_acc": "14", "cubify_sam3_feature_t1_err": "10.3",
           "cubify_clip_t1_acc": "20", "cubify_clip_t1_err": "6.0", "cubify_clip_t2_acc": "63",
           "cubify_clip_t2_err": "8.8", "monocular_t1_acc": "0", "monocular_t2_acc": "9", "monocular_t2_err": "22",
           "sparse4d_t1_acc": "91", "sparse4d_t1_err": "1.5", "sparse4d_t2_acc": "91", "sparse4d_t2_err": "0.7",
           "pre2026_t1_acc": "92", "pre2026_t1_err": "2.7", "pre2026_t2_acc": "92", "pre2026_t2_err": "1.1"},
          ("audit_*/gate.csv",), AUDIT, _a4_table_gates),
    Claim("A4.6", "A4", "On handheld scans a certified gate accepts 11-20% at the tolerance with at most 7.5% wrong, "
          "and 48-79% at twice it (baseline certified on its LiDAR answers alone).",
          {"t1_lo": "11", "t1_hi": "20", "t1_worst_err": "7.5", "t2_lo": "48", "t2_hi": "79"},
          ("audit_ark106_2d_lidar/gate.csv", "audit_ark106_cutr_*/gate.csv"), AUDIT, _a4_handheld_gates),
    Claim("A4.8", "A4", "The item-level certificate on held-out scenes: 6 of 45 non-empty gates exceed epsilon; the two "
          "largest by labels accepted (1,921 at 15% and 1,047 at 22% wrong) are on the warehouse scenes. The "
          "handheld Table 2 gates hold (6.0-8.9% wrong; baseline: its LiDAR-only audit), though the scene-bootstrap "
          "upper bound reaches 13%.",
          {"violated": "6", "gates": "45", "largest_n": "1921", "largest_err": "15", "second_n": "1047",
           "second_err": "22", "largest_two_warehouse": "True",
           "handheld_lo": "6.0", "handheld_hi": "8.9", "boot_hi_sam3_tau": "13"},
          ("audit_*/gate.csv",), AUDIT, _a4_violations),
    Claim("A4.9", "A4", "Leaving out a whole warehouse layout, Sparse4D v2.2 keeps 0.91 coverage and its gate (91% at "
          "1.7% wrong); the monocular baseline's gate at twice the tolerance accepts 1,011 labels at 36% wrong and the "
          "pre-2026 checkpoint's constant gate 2,167 at 16% (gates exceeding epsilon: 0, 2 and 2).",
          {"v22_coverage": "0.91", "v22_acc": "91", "v22_err": "1.7", "mono_n": "1011", "mono_err": "36",
           "old_n": "2167", "old_err": "16", "violated_v22": "0", "violated_mono": "2", "violated_old": "2"},
          ("audit_ss12_*_bylayout/gate.csv", "audit_ss12_sparse4d_v22_bylayout/coverage.csv"), AUDIT, _a4_layouts),
    Claim("A4.7", "A4", "Checking labels in risk order, Sparse4D's labels reach at most 1% wrong after 9% are checked, "
          "against 18% ordered by question type alone and 59% at random (a sixth): assuming 30 s per check, 46 instead "
          "of 91 or 297 reviewer-minutes per 1,000 labels (500 to check all). Handheld labels reach 2% after 69-74% "
          "(random: 92-95%).",
          {"sparse4d_ours": "9", "sparse4d_by_type": "18", "sparse4d_random": "59", "minutes_ours": "46",
           "minutes_by_type": "91", "minutes_random": "297",
           "minutes_all": "500", "ratio": "1/6", "handheld_ours_lo": "69", "handheld_ours_hi": "74",
           "handheld_random_lo": "92", "handheld_random_hi": "95"},
          ("review_effort/effort_targets.csv",), EFFORT, _a4_effort),
    Claim("A5.1", "A5", "Ten random references per site take the monocular warehouse baseline from 9% accepted at 22% "
          "wrong to 53% at 7.2% (twice the tolerance); a quarter of sites still miss a per-site bound.",
          {"before_acc": "9", "before_err": "22", "after_acc": "53", "after_err": "7.2",
           "sites_missing_bound": "1/4"}, ("references/ss12_2d/summary.csv",), "scripts/run_reference_experiment.py",
          _a5_references),
    Claim("A6.1", "A6", "The Qwen3-VL-8B vs Cosmos-Reason2-8B ranking holds under every labeller.",
          {"holds_everywhere": "True"}, ("vlm_results/*/delta_delta.csv",), "scripts/run_vlm_eval.py score",
          _a6_ranking),
    Claim("A6.2", "A6", "The benefit of seeing the image is understated by baseline labels (-0.064, 95% CI "
          "[-0.124, -0.010]) and CLIP-named labels (-0.137), not by SAM 3-named ones (-0.012, [-0.080, 0.050]).",
          {"baseline": "-0.064", "baseline_lo": "-0.124", "baseline_hi": "-0.010", "clip": "-0.137",
           "sam3": "-0.012", "sam3_lo": "-0.080", "sam3_hi": "0.050"},
          ("vlm_results/ark106_*/delta_delta.csv",), "scripts/run_vlm_eval.py score", _a6_image_benefit),
    Claim("A6.3", "A6", "On fixed cameras, Sparse4D's labels reproduce the reference model ranking exactly.",
          {"dd": "0.000", "lo": "0.000", "hi": "0.000"}, ("vlm_results/ss12_sparse4d_v22/delta_delta.csv",),
          "scripts/run_vlm_eval.py score", _a6_exact),
    Claim("A7.1", "A7", "Results table, recall of present / clearly visible objects.",
          {"baseline_present": "0.66", "baseline_clear": "0.87", "cubify_sam3_present": "0.58",
           "cubify_sam3_clear": "0.80", "cubify_clip_present": "0.45", "cubify_clip_clear": "0.80",
           "monocular_present": "0.63", "monocular_clear": "0.74", "sparse4d_present": "0.75",
           "sparse4d_clear": "0.95", "pre2026_present": "0.60", "pre2026_clear": "0.74"},
          ("misses/*/recall.csv",), MISSES, _a7_recall_table),
    Claim("A7.2", "A7", "The baseline finds 33% of objects less than 10% visible, against 89% of those more than 60% "
          "visible.", {"under_10pct": "33", "over_60pct": "89"}, ("misses/ark106_v0/recall.csv",), MISSES,
          _a7_visibility),
    Claim("A7.3", "A7", "Conformal risk control certifies Cubify's per-scene miss rate on clearly visible objects at 5, "
          "10 or 20%, using 9.3, 5.8 or 3.7 candidates per image.",
          {"feasible_0.05": "True", "feasible_0.1": "True", "feasible_0.2": "True", "boxes_per_image_0.05": "9.3",
           "boxes_per_image_0.1": "5.8", "boxes_per_image_0.2": "3.7"},
          ("recall_control/ark106_t005_clearly_visible_mark/by_alpha.csv",), "scripts/run_recall_control.py",
          _a7_crc),
    Claim("A7.4", "A7", "With only 12 warehouse sites, 20% is the lowest certifiable per-scene miss rate.",
          {"lowest_alpha": "20"}, ("recall_control/ss12_s4d_v22_clearly_visible/by_alpha.csv",),
          "scripts/run_recall_control.py", _a7_sites),
    Claim("A7.5", "A7", "Recall guarantees bound the per-scene average: 37-41% of scenes exceed alpha = 0.2.",
          {"lo": "37", "hi": "41"}, ("recall_control/ark106_t005_*_mark/by_alpha.csv",),
          "scripts/run_recall_control.py", _a7_average),
    Claim("A7.6", "A7", "Disagreement flags catch 46-76% of clearly visible misses (warehouse / handheld); in the fully "
          "annotated warehouses the baseline flags the pre-2026 detector's misses with 69% precision. The handheld "
          "flags cost about 10 per image, 95% of them on objects the reference does not annotate.",
          {"warehouse_clear": "46", "handheld_clear": "76", "warehouse_precision": "69",
           "handheld_flags_per_image": "10", "handheld_unannotated": "95"},
          ("misses/ark106_v0/flag_metrics.csv", "misses/ss12_sparse4d_v11/flag_metrics.csv"), MISSES, _a7_flags),
    Claim("A7.7", "A7", "In the fully annotated warehouses, checking 2.0-3.3 flagged boxes per image recovers 1.0-2.3 "
          "missed objects per image.", {"flags_lo": "2.0", "flags_hi": "3.3", "recovered_lo": "1.0",
                                         "recovered_hi": "2.3"},
          ("review_effort/miss_review.csv",), EFFORT, _a7_flag_cost),
    Claim("DEC.1", "A4", "Certifying threshold decisions (epsilon 0.05) accepts 91% of handheld decisions at 3.6-4.3% "
          "wrong (accepting all would be 6-7% wrong), i.e. with at most 5% wrong.",
          {"accepted_lo": "91", "accepted_hi": "91", "wrong_lo": "3.6", "wrong_hi": "4.3", "accept_all_lo": "6",
           "accept_all_hi": "7", "within_5pct": "True"}, (f"{DECISIONS}/decision_gate.csv",), DECIDE, _dec_handheld,
          block="decisions"),
    Claim("DEC.2", "A4", "Monocular warehouse labels, which the fixed-tolerance gate rejects entirely, get 75% of "
          "threshold decisions certified at 2.6% wrong.", {"accepted": "75", "wrong": "2.6", "fixed_gate": "0"},
          (f"{DECISIONS}/decision_gate.csv",), DECIDE, _dec_monocular, block="decisions"),
    Claim("DEC.3", "A4", "For monocular labels, calibrating a comparison's margin directly gives intervals 22% narrower "
          "than combining them as independent, at 90% coverage; 39% of \"which is larger?\" calls certify at 2.0% "
          "wrong.", {"narrower": "22", "coverage": "0.90", "sign_accepted": "39", "sign_wrong": "2.0"},
          (f"{DECISIONS}/margin_intervals.csv", f"{DECISIONS}/margin_gate.csv"), DECIDE, _dec_margins,
          block="decisions"),
    Claim("DEC.4", "A4", "Close calls (threshold within 10% of the reference) do not certify on one handheld "
          "labeller's evidence.", {"accepted": "0"}, (f"{DECISIONS}/sweep.csv",), DECIDE, _dec_close,
          block="decisions"),
    Claim("DEC.5", "A4", "With a second labeller's agreement, 72-73% of handheld decisions 10% from their threshold "
          "certify at 6.0-6.7% wrong (epsilon 0.10); with Sparse4D as the witness, monocular warehouse decisions "
          "5% from their threshold certify at 59% with 2.6% wrong (epsilon 0.05).",
          {"handheld_acc_lo": "72", "handheld_acc_hi": "73", "handheld_wrong_lo": "6.0", "handheld_wrong_hi": "6.7",
           "monocular_acc": "59", "monocular_wrong": "2.6"}, (f"{DECISIONS}/witness.csv",), DECIDE,
          _dec_second_opinion, block="decisions"),
)
"""tuple of Claim: every checked claim, in order."""


# --------------------------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------------------------


def evaluate(claims: Sequence[Claim], results: Path) -> list[dict[str, str]]:
    """Recompute every quantity of every claim and compare it with its stated value.

    Parameters
    ----------
    claims : sequence of Claim
        Claims to check.
    results : pathlib.Path
        Results folder.

    Returns
    -------
    list of dict
        One row per quantity: ``id, aim, quantity, stated, recomputed, status`` (``PASS``, ``FAIL`` or ``ERROR``
        when the value could not be recomputed) and ``detail``.
    """
    rows: list[dict[str, str]] = []
    for c in claims:
        try:
            values = c.compute(results)
            error = ""
        except (OSError, LookupError, KeyError, ValueError) as exc:
            values, error = {}, f"{type(exc).__name__}: {exc}"
        for q, stated in c.stated.items():
            if q not in values:
                rows.append({"id": c.id, "aim": c.aim, "quantity": q, "stated": stated, "recomputed": "",
                             "status": "ERROR", "detail": error or "not computed"})
                continue
            v = values[q]
            rows.append({"id": c.id, "aim": c.aim, "quantity": q, "stated": stated, "recomputed": fmt(v),
                         "status": "PASS" if matches(stated, v) else "FAIL", "detail": ""})
    return rows


def markdown(claims: Sequence[Claim], rows: Sequence[dict[str, str]]) -> str:
    """The report as markdown: one section per claim with its statement, sources and quantities.

    Parameters
    ----------
    claims : sequence of Claim
        The checked claims.
    rows : sequence of dict
        :func:`evaluate` output.

    Returns
    -------
    str
        Markdown text.
    """
    n_fail = sum(r["status"] != "PASS" for r in rows)
    out = ["# Claims check", "", f"{len(rows)} quantities in {len(claims)} claims; "
           f"{len(rows) - n_fail} pass, {n_fail} do not.", ""]
    for c in claims:
        out += [f"## {c.id} ({c.aim}{', decision-level block' if c.block == 'decisions' else ''})", "",
                c.statement, "", f"- Sources: {', '.join(f'`results/{s}`' for s in c.sources)}",
                f"- Produced by: `{c.script}`", "", "| Quantity | Stated | Recomputed | Status |", "|---|---|---|---|"]
        out += [f"| {r['quantity']} | {r['stated']} | {r['recomputed'] or r['detail']} | {r['status']} |"
                for r in rows if r["id"] == c.id]
        out.append("")
    return "\n".join(out)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        ``results``, ``block`` and ``report``.
    """
    p = argparse.ArgumentParser(description="Recompute the project's key claims from the committed result tables.")
    p.add_argument("--results", type=Path, default=RESULTS, help="folder of the result tables (default: ./results)")
    p.add_argument("--block", choices=["all", "core", "decisions"], default="all", help="which claims to check")
    p.add_argument("--report", type=Path, default=None, help="also write a markdown report here (must not exist)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Check the claims and print the outcome.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        0 if every quantity passes, 1 if any fails or cannot be recomputed, 2 if the report path exists.
    """
    args = parse_args(argv)
    if args.report is not None and args.report.exists():
        print(f"refusing to overwrite {args.report}", file=sys.stderr)
        return 2
    claims = [c for c in CLAIMS if args.block in ("all", c.block)]
    rows = evaluate(claims, args.results)
    width = max(len(f"{r['id']} {r['quantity']}") for r in rows)
    print(f"{'claim / quantity':{width}s}  {'stated':>8s}  {'recomputed':>11s}  status")
    for r in rows:
        print(f"{r['id'] + ' ' + r['quantity']:{width}s}  {r['stated']:>8s}  {r['recomputed'] or '-':>11s}  "
              f"{r['status']}{'  ' + r['detail'] if r['detail'] else ''}")
    bad = [r for r in rows if r["status"] != "PASS"]
    print(f"\n{len(rows)} quantities in {len(claims)} claims: {len(rows) - len(bad)} pass, {len(bad)} do not.")
    if args.report is not None:
        with open(args.report, "x", encoding="utf-8") as f:
            f.write(markdown(claims, rows) + "\n")
        print(f"wrote {args.report}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
