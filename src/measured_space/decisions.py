"""Certify decisions, not measurements: accept a metric answer when its error bar does not cross the decision.

The gate of :mod:`measured_space.audit` accepts a numeric answer when its calibrated error bound lies within a
fixed tolerance ``tau``. A fixed tolerance stands in for a decision nobody stated. When the decision is known
("is the distance more than 2 m?", "is A taller than B?"), the natural rule accepts an answer when its error bar
does not cross the decision's threshold: a 0.3 m error is harmless for "more than 2 m?" when the answer is 6 m
and decisive when it is 2.1 m. This module measures, per labeller, how many answers such a rule can accept with a
certified bound on the **wrong-decision** rate, next to the fixed-tolerance gate. It reads the audit's per-item
outputs (``pairs.csv``, ``numeric.csv``, ``gate.csv``, ``folds.csv``) and refits nothing.

1. **Answers** (:func:`answer_table`). Every numeric answer of the feature model (kind ``sigma``) with its
   out-of-scene bias ``mu`` and the half-width ``h`` of its split-conformal interval at ``level`` (``half90``:
   ``e = value - reference`` lies in ``[mu - h, mu + h]`` with 90 % marginal coverage), so the interval for the
   reference is ``[c - h, c + h]`` around the bias-corrected value ``c = value - mu``. Each answer also carries
   its reference ``ref = value - e``, its cross-fitting fold (:func:`fold_assignment`, recomputed with the
   audit's :func:`~measured_space.audit.scene_folds` and checked against the recorded ``folds.csv``) and whether
   the fixed-tolerance gate accepted it (``gate.csv``: kind ``sigma``, ``tau_scale`` 1).
2. **One answer against a threshold** (:func:`threshold_decisions`). "Is the reference above ``t``?" The call
   is ``c > t`` and is wrong when it differs from ``ref > t``. Score ``r = h / |c - t|``: ``r <= 1`` exactly when
   the interval excludes ``t``; an answer exactly at ``t`` (``r`` infinite) is never accepted. Thresholds are
   fixed in advance per dataset and question type (:data:`DEFAULT_THRESHOLDS`).

   - :func:`sweep_decisions` places the threshold relative to the reference instead, ``t = ref (1 + u)``. This is
     an analysis device, not a deployable decision (``t`` depends on the unknown reference): it shows how the
     accepted share grows with the distance between the answer and the threshold. The truth is ``u < 0``.
   - :func:`safety_decisions` is the one-sided variant for clearances: only "outside" calls (``c > t``) can be
     accepted, ``r = h / (c - t)``, and an accepted call is wrong when ``ref <= t`` (truly inside).
3. **Margins between two answers** (:func:`margin_pairs`, :func:`direct_margin_halfwidth`). Two answers of the
   same question type in the same view about different objects (the "does it fit?" pattern): margin
   ``m_hat = value_A - value_B``, truth ``m = ref_A - ref_B``, error ``e_m = e_A - e_B``, centred at
   ``mu_A - mu_B``. Three 90 % error bars: the sum of the half-widths, their quadrature sum, and a direct
   split-conformal calibration of ``|e_m - (mu_A - mu_B)| / sqrt(sigma_A^2 + sigma_B^2)`` on pairs from other
   folds (Mondrian by question type). Errors shared by the two answers (one camera pose, one depth scale) cancel
   in ``e_m``; only the direct calibration can see that. :func:`margin_decisions` gates "is A greater than B?"
   and, for sizes, "is A smaller than B by at least ``fit_margin``?" with the direct half-width.
4. **Certification** (:func:`certify`). A decision is accepted when ``r <= lambda``. ``lambda`` is chosen per
   test scene by the audit's Learn-then-Test procedure (:func:`~measured_space.audit.ltt_threshold`, same
   ``min_accept`` and ``min_accept_frac``) on calibration decisions from scenes outside the test scene's fold, as
   :func:`~measured_space.audit.gate_decisions` does, so that ``P(wrong decision | accepted) <= epsilon`` at
   confidence ``1 - delta``. Decisions are pooled for calibration per labeller (as the audit's gate pools question
   types) or per decision family (``pool="family"``: one question type and threshold).
5. **Reports** (:func:`rule_table`, :func:`margin_interval_table`). Per group: the accepted share and the
   wrong-decision rate among accepted decisions with Wilson and scene-cluster bootstrap 95 % intervals, for the
   certified rule, the uncertified "interval excludes ``t``" rule (``lambda = 1``), accepting every decision, and
   the fixed-tolerance gate applied to the same decisions.

6. **Exact calibration** (:func:`calibration_answers`, :func:`direct_margin_halfwidth_exact`). An audit run
   since the calibration predictions are saved (``calibration_numeric.csv.gz``, ``calibration_quantiles.csv.gz``)
   holds, per test scene, the leave-two-out prediction of every calibration item (a model fitted without the
   test scene's fold and the item's own) and the conformal quantiles that give the test scene's answers their
   ``half90``. Calibration decisions for a test scene are then rebuilt from them: ``mu`` and ``sigma`` of the
   leave-two-out model, the half-width ``q90 * sigma`` with the test scene's quantile for the item's cell, so a
   calibration decision is made exactly as one of the test scene's own (:func:`certify`'s exact mode; margins
   likewise). For a question type with no answer of the test scene in any depth source, nothing of that type ever
   saw the scene, and its calibration decisions are the other scenes' test decisions. One data reuse remains: the
   per-cell quantile that scales every calibration decision's error bar is computed from those same items'
   scores (one scalar per cell).
7. **Second opinion** (:func:`attach_witness`, :func:`second_opinion_score`, :func:`apply_witness_rule`). A
   witness labeller that answers the same question id must make the same call, and the decision's score is the
   larger of the two labellers' scores; certified by the same Learn-then-Test on the same calibration decisions
   (exact when both audits saved their calibration predictions).

**Calibration caveat (approximate mode).** An audit folder written before the calibration predictions were saved
holds one out-of-scene prediction per answer (``numeric.csv``) only. Calibration decisions for a test scene are
then other folds' test predictions: each comes from a model whose training data included the test scene's fold,
and its ``half90`` from a conformal quantile whose calibration scores included that fold. The test scene's own
answers never enter its calibration set, but the certificate is approximate in this respect. As in the audit,
items are treated as independent and scenes as exchangeable, so Wilson intervals and binomial bounds are
optimistic for items that share a scene.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from measured_space.audit import (AuditConfig, cluster_interval, conformal_quantile, ltt_threshold, scene_folds,
                                  wilson_interval)

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DEFAULT_THRESHOLDS: dict[str, dict[str, tuple[float, ...]]] = {
    "arkitscenes": {"distance_center": (0.5, 1.0), "distance_surface": (0.5, 1.0), "ego_distance": (0.75, 1.5),
                    "size_height": (0.5, 1.0), "size_longest": (0.6, 1.0)},
    "smartspaces": {"distance_center": (2.0, 5.0, 10.0), "distance_surface": (2.0, 5.0, 10.0),
                    "ego_distance": (10.0, 30.0), "size_height": (1.0, 2.0), "size_longest": (1.0, 2.0)},
}
"""dict: decision thresholds in metres per dataset and question type, fixed before looking at the decisions."""

SETTING: dict[str, str] = {"arkitscenes": "handheld", "smartspaces": "warehouse"}
"""dict: the capture setting of each dataset (handheld RGB-D video; fixed multi-camera warehouse)."""

SWEEP_U: tuple[float, ...] = (-0.5, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.5)
"""tuple of float: relative threshold offsets ``u`` of the sweep, ``t = ref (1 + u)``."""

SAFETY_QTYPES: tuple[str, ...] = ("distance_center", "distance_surface")
"""tuple of str: question types of the one-sided clearance decisions."""

MARGIN_QTYPES: tuple[str, ...] = ("size_height", "size_longest", "ego_distance")
"""tuple of str: question types whose answers are compared in pairs."""

FIT_QTYPES: tuple[str, ...] = ("size_height", "size_longest")
"""tuple of str: question types of the "A smaller than B by at least a margin" decision."""

POOLS: tuple[str, ...] = ("labeller", "family")
"""tuple of str: calibration pools of :func:`certify`: all of a labeller's decisions, or one decision family."""

NEVER = float(np.finfo(np.float64).max)
"""float: the largest threshold Learn-then-Test may test, so that an infinite score is never accepted."""

_TIE = 1e-3                      # m: margins closer to zero than this are counted as ties in the reports


@dataclass(frozen=True)
class DecisionConfig:
    """Every setting of the decision analysis.

    Attributes
    ----------
    epsilons : tuple of float
        Certified bounds on ``P(wrong decision | accepted)``; the first is the primary one (default
        ``(0.1, 0.05)``).
    delta : float
        The certificate holds with probability at least ``1 - delta`` (default 0.1).
    level : float
        Nominal coverage of the error bars (default 0.9: the audit's ``half90``).
    min_accept : int
        Learn-then-Test's first candidate accepts at least this many calibration decisions (default: the
        audit's, 30).
    min_accept_frac : float
        ... and at least this share of them (default: the audit's, 0.05).
    pool : str
        Calibration pool (:data:`POOLS`): ``"labeller"`` (default; every decision of one analysis, as the
        audit's gate pools question types) or ``"family"`` (one question type and threshold).
    fit_margin : float
        Clearance of the "A smaller than B by at least" decision, metres (default 0.1).
    n_boot : int
        Scene-cluster bootstrap replicates (default 2000).
    seed : int
        Seed of the bootstrap (default 0).
    """

    epsilons: tuple[float, ...] = (0.1, 0.05)
    delta: float = 0.1
    level: float = 0.9
    min_accept: int = AuditConfig.min_accept
    min_accept_frac: float = AuditConfig.min_accept_frac
    pool: str = "labeller"
    fit_margin: float = 0.1
    n_boot: int = 2000
    seed: int = 0

    def __post_init__(self) -> None:
        """Check the settings.

        Raises
        ------
        ValueError
            If ``epsilons`` is empty or any level lies outside (0, 1), or ``pool`` is unknown.
        """
        if not self.epsilons or not all(0.0 < e < 1.0 for e in self.epsilons):
            raise ValueError(f"epsilons must be a non-empty tuple of values in (0, 1), got {self.epsilons}")
        for name in ("delta", "level"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must lie in (0, 1), got {getattr(self, name)}")
        if self.pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {self.pool!r}")

    @property
    def half_column(self) -> str:
        """str: the ``numeric.csv`` column of the interval half-width at ``level`` (e.g. ``half90``)."""
        return f"half{int(round(100 * self.level))}"

    def bootstrap(self) -> AuditConfig:
        """The audit settings that :func:`~measured_space.audit.cluster_interval` reads.

        Returns
        -------
        AuditConfig
            Default settings with this ``n_boot`` and ``seed``.
        """
        return AuditConfig(n_boot=self.n_boot, seed=self.seed)


@dataclass(frozen=True)
class Rule:
    """An acceptance rule to report: which decisions it accepts and what it certifies.

    Attributes
    ----------
    name : str
        Rule name in the report (e.g. ``"certified"``, ``"interval"``, ``"accept_all"``, ``"fixed_tau_gate"``).
    accept : str
        Boolean column of the decisions table: accepted by the rule.
    epsilon : float
        The bound the rule certifies; NaN for uncertified rules.
    lam : str or None
        Column of the per-decision threshold ``lambda`` (certified rules), or None.
    """

    name: str
    accept: str
    epsilon: float = math.nan
    lam: str | None = None


# ============================================================================================================
# 1. Answers
# ============================================================================================================


def fold_assignment(rows: pd.DataFrame, max_folds: int | None, seed: int = 0,
                    recorded: pd.DataFrame | None = None) -> tuple[pd.Series, list[str]]:
    """The audit's cross-fitting fold of every row, recomputed and checked against its record.

    The audit assigns folds per dataset and question type over all its modelled rows (every depth source), with
    :func:`~measured_space.audit.scene_folds`. The same call is repeated here; where ``recorded`` (the audit's
    ``folds.csv``) lists a group, the two assignments must agree.

    Parameters
    ----------
    rows : pandas.DataFrame
        The audit's modelled numeric rows (one per row and model kind is fine, as long as each scene's row count
        is the audit's): ``dataset``, ``qtype``, ``scene`` (str).
    max_folds : int or None
        The audit's ``max_folds``.
    seed : int, default 0
        The audit's seed.
    recorded : pandas.DataFrame, optional
        The audit's ``folds.csv`` (``dataset, qtype, scene, fold``); groups it lists are checked.

    Returns
    -------
    fold : pandas.Series
        Fold number (int64, ``0 .. K - 1`` per dataset and question type), aligned with ``rows``.
    checked : list of str
        The ``"<dataset>/<qtype>"`` groups that matched the record.

    Raises
    ------
    ValueError
        If a recomputed assignment differs from the recorded one.
    """
    fold = pd.Series(np.full(len(rows), -1, dtype=np.int64), index=rows.index)
    checked: list[str] = []
    for (dataset, qtype), g in rows.groupby(["dataset", "qtype"], sort=True):
        fold_of = scene_folds(g["scene"].astype(str), max_folds, seed)
        fold.loc[g.index] = g["scene"].astype(str).map(fold_of).to_numpy(dtype=np.int64)
        if recorded is None or not len(recorded):
            continue
        rec = recorded[(recorded["dataset"] == dataset) & (recorded["qtype"] == qtype)]
        if not len(rec):
            continue
        expected = {str(s): int(f) for s, f in zip(rec["scene"], rec["fold"])}
        if expected != {str(s): int(f) for s, f in fold_of.items()}:
            raise ValueError(f"{dataset}/{qtype}: the recomputed fold assignment differs from the recorded one")
        checked.append(f"{dataset}/{qtype}")
    return fold, checked


def answer_table(pairs: pd.DataFrame, numeric: pd.DataFrame, gate: pd.DataFrame, *, source: str | None = None,
                 level: float = 0.9, max_folds: int | None = 20, seed: int = 0,
                 recorded_folds: pd.DataFrame | None = None, kind: str = "sigma") -> tuple[pd.DataFrame, list[str]]:
    """One row per numeric answer of one labeller: reference, bias-corrected value, error bar, fold, gate decision.

    Parameters
    ----------
    pairs : pandas.DataFrame
        The audit's ``pairs.csv``; its 0-based position is the ``row`` key. Needs ``id, dataset, scene,
        camera_id, frame, qtype, value, e`` (``scene``, ``camera_id``, ``frame`` and ``id`` read as str).
    numeric : pandas.DataFrame
        The audit's ``numeric.csv``: ``row, kind, dataset, scene, qtype, depth_source, e, mu, sigma`` and
        ``half<level>``.
    gate : pandas.DataFrame
        The audit's ``gate.csv`` (``row, kind, tau_scale, answer_kind, decision``); the fixed-tolerance gate is
        kind ``kind`` at ``tau_scale == 1``.
    source : str, optional
        Depth source to keep. Required when the audit has several.
    level : float, default 0.9
        Coverage level of the error bar (column ``half<100 level>``).
    max_folds : int or None, default 20
        The audit's ``max_folds`` (for :func:`fold_assignment`).
    seed : int, default 0
        The audit's seed.
    recorded_folds : pandas.DataFrame, optional
        The audit's ``folds.csv``, to check the recomputed folds against.
    kind : str, default "sigma"
        Error model whose predictions are used.

    Returns
    -------
    answers : pandas.DataFrame
        Columns ``row, id, dataset, scene, camera_id, frame, qtype, depth_source, value, e, ref, mu, sigma,
        half, c, fold, fixed_accept``; metres throughout. ``half`` is the error bar's half-width (NaN replaced by
        ``inf``: never accepted), ``c = value - mu``, ``ref = value - e``, ``fixed_accept`` the fixed-tolerance
        gate's acceptance (bool).
    checked : list of str
        Groups whose folds matched the record (see :func:`fold_assignment`).

    Raises
    ------
    KeyError
        If a required column is missing.
    ValueError
        If the source is missing or ambiguous, if ``numeric`` and ``pairs`` disagree on a row (scene, question
        type or error), or if an answer has no fixed-gate decision.
    """
    half_col = f"half{int(round(100 * level))}"
    needed = {"row", "kind", "dataset", "scene", "qtype", "depth_source", "e", "mu", "sigma", half_col}
    missing = needed - set(numeric.columns)
    if missing:
        raise KeyError(f"numeric table lacks columns {sorted(missing)}")
    pred = numeric[numeric["kind"] == kind].reset_index(drop=True)
    pred = pred.assign(scene=pred["scene"].astype(str))
    fold, checked = fold_assignment(pred, max_folds, seed, recorded_folds)
    pred["fold"] = fold.to_numpy()
    sources = sorted(pred["depth_source"].astype(str).unique())
    if source is None:
        if len(sources) > 1:
            raise ValueError(f"the audit has several depth sources {sources}; name one")
        source = sources[0] if sources else None
    elif source not in sources:
        raise ValueError(f"depth source {source!r} not in the audit (it has {sources})")
    pred = pred[pred["depth_source"].astype(str) == source].reset_index(drop=True)

    rows = pred["row"].to_numpy(dtype=np.int64)
    meta = pairs.iloc[rows].reset_index(drop=True)
    if not ((meta["scene"].astype(str).to_numpy() == pred["scene"].to_numpy()).all()
            and (meta["qtype"].to_numpy() == pred["qtype"].to_numpy()).all()):
        raise ValueError("numeric rows do not match the pair table's scene and question type")
    if not np.allclose(meta["e"].to_numpy(float), pred["e"].to_numpy(float), rtol=1e-4, atol=1e-5):
        raise ValueError("numeric rows do not match the pair table's errors")

    fixed = gate[(gate["kind"] == kind) & (gate["tau_scale"] == 1.0) & (gate["answer_kind"] == "numeric")]
    decision = fixed.drop_duplicates("row").set_index("row")["decision"]
    accepted = pd.Series(rows).map(decision)
    if accepted.isna().any():
        raise ValueError(f"{int(accepted.isna().sum())} answers have no fixed-gate decision")

    value = meta["value"].to_numpy(float)
    e = pred["e"].to_numpy(float)
    mu = pred["mu"].to_numpy(float)
    half = pred[half_col].to_numpy(float)
    answers = pd.DataFrame({
        "row": rows, "id": meta["id"].astype(str).to_numpy(), "dataset": pred["dataset"].to_numpy(),
        "scene": pred["scene"].to_numpy(), "camera_id": meta["camera_id"].astype(str).to_numpy(),
        "frame": meta["frame"].astype(str).to_numpy(), "qtype": pred["qtype"].to_numpy(),
        "depth_source": pred["depth_source"].astype(str).to_numpy(), "value": value, "e": e, "ref": value - e,
        "mu": mu, "sigma": pred["sigma"].to_numpy(float), "half": np.where(np.isnan(half), np.inf, half),
        "c": value - mu, "fold": pred["fold"].to_numpy(dtype=np.int64),
        "fixed_accept": (accepted == "accept").to_numpy(dtype=bool)})
    return answers, checked


def fold_lookup(answers: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    """The fold of every scene per dataset and question type.

    Parameters
    ----------
    answers : pandas.DataFrame
        :func:`answer_table` output (``dataset, qtype, scene, fold``).

    Returns
    -------
    dict
        ``(dataset, qtype, scene) -> fold``.
    """
    keys = answers[["dataset", "qtype", "scene", "fold"]].drop_duplicates(["dataset", "qtype", "scene"])
    return {(str(d), str(q), str(s)): int(f) for d, q, s, f in keys.itertuples(index=False)}


# ============================================================================================================
# 2. Decisions about one answer
# ============================================================================================================


def two_sided(c: ArrayLike, t: ArrayLike, ref: ArrayLike, half: ArrayLike) -> tuple[BoolArray, BoolArray, Array]:
    """Call, wrong-decision flag and score of "is the reference above ``t``?".

    Parameters
    ----------
    c : array_like, shape (n,)
        Bias-corrected answers, m.
    t : array_like, shape (n,) or scalar
        Thresholds, m.
    ref : array_like, shape (n,)
        Reference values, m.
    half : array_like, shape (n,)
        Error-bar half-widths, m (the interval for the reference is ``[c - half, c + half]``).

    Returns
    -------
    call : numpy.ndarray of bool, shape (n,)
        ``c > t``.
    wrong : numpy.ndarray of bool, shape (n,)
        ``call != (ref > t)``.
    score : numpy.ndarray of float, shape (n,)
        ``half / |c - t|`` (``<= 1`` when the interval excludes ``t``); ``inf`` where ``c == t`` or the
        half-width is undefined.
    """
    c, t, ref, half = (np.asarray(x, dtype=np.float64) for x in (c, t, ref, half))
    call = c > t
    wrong = call != (ref > t)
    gap = np.abs(c - t)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(gap > 0, half / gap, np.inf)
    score = np.where(np.isnan(score), np.inf, score)
    return np.asarray(call, dtype=bool), np.asarray(wrong, dtype=bool), np.asarray(score, dtype=np.float64)


# Answer columns carried into every single-answer decision table.
_DECISION_KEEP = ["row", "id", "dataset", "scene", "qtype", "fold", "fixed_accept", "c", "ref", "half"]
_TEST_SCENE = "test_scene"       # column of calibration answers and decisions: the test scene they calibrate


def _keep(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Columns to carry into a decision table, plus ``test_scene`` when the frame holds calibration answers.

    Parameters
    ----------
    frame : pandas.DataFrame
        Answers (:func:`answer_table`) or calibration answers (:func:`calibration_answers`).
    columns : sequence of str
        The columns every decision table carries.

    Returns
    -------
    list of str
        ``columns``, then ``test_scene`` if ``frame`` has it.
    """
    return [*columns, *([_TEST_SCENE] if _TEST_SCENE in frame.columns else [])]


def threshold_decisions(answers: pd.DataFrame,
                        thresholds: Mapping[str, Mapping[str, Sequence[float]]]) -> pd.DataFrame:
    """Every answer against every fixed threshold of its dataset and question type.

    Parameters
    ----------
    answers : pandas.DataFrame
        :func:`answer_table` output.
    thresholds : mapping
        ``{dataset: {qtype: thresholds in m}}`` (e.g. :data:`DEFAULT_THRESHOLDS`); question types without
        thresholds are left out.

    Returns
    -------
    pandas.DataFrame
        One row per answer and threshold: ``row, id, dataset, scene, qtype, fold, fixed_accept, c, ref, half``
        (and ``test_scene`` for :func:`calibration_answers`), ``threshold`` and ``t`` (both the threshold, m),
        ``call`` (``c > t``), ``truth`` (``ref > t``), ``wrong`` and ``score`` (:func:`two_sided`).
    """
    keep = _keep(answers, _DECISION_KEEP)
    parts = []
    for (dataset, qtype), g in answers.groupby(["dataset", "qtype"], sort=True):
        for t in thresholds.get(str(dataset), {}).get(str(qtype), ()):
            part = g[keep].assign(threshold=float(t), t=float(t))
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=[*keep, "threshold", "t", "call", "truth", "wrong", "score"])
    out = pd.concat(parts, ignore_index=True)
    out["call"], out["wrong"], out["score"] = two_sided(out["c"], out["t"], out["ref"], out["half"])
    out["truth"] = out["ref"].to_numpy(float) > out["t"].to_numpy(float)
    return out


def sweep_decisions(answers: pd.DataFrame, us: Sequence[float] = SWEEP_U) -> pd.DataFrame:
    """Every answer against thresholds placed relative to its reference, ``t = ref (1 + u)`` (analysis device).

    The truth of "is the reference above ``t``?" is ``u < 0`` by construction; what varies is how far the
    answer lies from the threshold relative to its error bar. Answers with a reference ``<= 0`` are left out.

    Parameters
    ----------
    answers : pandas.DataFrame
        :func:`answer_table` output.
    us : sequence of float, default :data:`SWEEP_U`
        Relative offsets ``u`` (non-zero).

    Returns
    -------
    pandas.DataFrame
        One row per answer and offset: the columns of :func:`threshold_decisions` without ``threshold`` (with
        ``test_scene`` for calibration answers), plus ``u`` and ``u_abs``.

    Raises
    ------
    ValueError
        If an offset is zero (the decision would have no true answer).
    """
    if any(u == 0 for u in us):
        raise ValueError("sweep offsets must be non-zero")
    keep = _keep(answers, _DECISION_KEEP)
    usable = answers[answers["ref"].to_numpy(float) > 0]
    parts = [usable[keep].assign(u=float(u), u_abs=abs(float(u)),
                                 t=usable["ref"].to_numpy(float) * (1.0 + float(u))) for u in us]
    if not parts or not len(usable):
        return pd.DataFrame(columns=[*keep, "u", "u_abs", "t", "call", "truth", "wrong", "score"])
    out = pd.concat(parts, ignore_index=True)
    out["call"], out["wrong"], out["score"] = two_sided(out["c"], out["t"], out["ref"], out["half"])
    out["truth"] = out["ref"].to_numpy(float) > out["t"].to_numpy(float)
    return out


def safety_decisions(answers: pd.DataFrame, thresholds: Mapping[str, Mapping[str, Sequence[float]]],
                     qtypes: Sequence[str] = SAFETY_QTYPES) -> pd.DataFrame:
    """One-sided clearance decisions: only "outside" calls (``c > t``) can be accepted.

    Parameters
    ----------
    answers : pandas.DataFrame
        :func:`answer_table` output.
    thresholds : mapping
        ``{dataset: {qtype: thresholds in m}}``.
    qtypes : sequence of str, default :data:`SAFETY_QTYPES`
        Question types to use.

    Returns
    -------
    pandas.DataFrame
        One row per answer and threshold: ``row, id, dataset, scene, qtype, fold, fixed_accept, c, ref, half,
        threshold, t``, ``call`` (``c > t``: an outside call), ``inside`` (``ref <= t``), ``wrong``
        (``call & inside``: accepting it would be wrong) and ``score`` (``half / (c - t)`` for outside calls,
        ``inf`` otherwise, so inside calls are never accepted).
    """
    subset = answers[answers["qtype"].isin(list(qtypes))]
    out = threshold_decisions(subset, thresholds)
    if not len(out):
        return out.assign(inside=pd.Series(dtype=bool))
    c, t, ref, half = (out[k].to_numpy(float) for k in ("c", "t", "ref", "half"))
    call = c > t
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(call, half / np.where(call, c - t, 1.0), np.inf)
    out["call"], out["inside"] = call, ref <= t
    out["wrong"] = call & (ref <= t)
    out["score"] = np.where(np.isnan(score), np.inf, score)
    return out.drop(columns=["truth"])


# ============================================================================================================
# 3. Margins between two answers
# ============================================================================================================


def margin_pairs(answers: pd.DataFrame, referents: Mapping[str, Sequence[str]],
                 qtypes: Sequence[str] = MARGIN_QTYPES) -> tuple[pd.DataFrame, dict[str, int]]:
    """Pairs of answers of one question type in one view about two different objects.

    Each unordered pair appears once, oriented so that ``A`` has the smaller question id (ids are hashes, so
    the orientation is unrelated to the values).

    Parameters
    ----------
    answers : pandas.DataFrame
        :func:`answer_table` output, or :func:`calibration_answers` (pairs are then formed within each test
        scene's calibration answers and carry ``test_scene``).
    referents : mapping of str to sequence of str
        Question id to the ids of its referent objects (from the label folder's question records).
    qtypes : sequence of str, default :data:`MARGIN_QTYPES`
        Question types to pair.

    Returns
    -------
    pairs : pandas.DataFrame
        One row per pair: ``dataset, scene, camera_id, frame, qtype, fold, id_a, id_b, object_a, object_b,
        m_hat`` (``value_A - value_B``), ``m`` (``ref_A - ref_B``), ``e_m`` (``e_A - e_B``), ``mu_m``
        (``mu_A - mu_B``), ``c_m`` (``m_hat - mu_m``), ``s_m`` (``sqrt(sigma_A^2 + sigma_B^2)``), ``resid_a``,
        ``resid_b`` (``e - mu`` of each answer), ``h_sum`` (``half_A + half_B``), ``h_quad``
        (``sqrt(half_A^2 + half_B^2)``) and ``fixed_accept`` (the fixed gate accepted both); metres.
    counts : dict of str to int
        ``n_answers`` (of the paired types), ``n_no_referents`` (id missing from ``referents``),
        ``n_multi_referent`` (answers about more than one object, left out), ``n_same_object`` (pairs about the
        same object, left out) and ``n_pairs``.

    Raises
    ------
    ValueError
        If the two answers of a pair are in different folds (they share a scene and question type).
    """
    subset = answers[answers["qtype"].isin(list(qtypes))].copy()
    objects = subset["id"].map(lambda i: tuple(referents[i]) if i in referents else None)
    counts = {"n_answers": int(len(subset)), "n_no_referents": int(objects.isna().sum()),
              "n_multi_referent": int(objects.map(lambda o: o is not None and len(o) != 1).sum())}
    keep = objects.map(lambda o: o is not None and len(o) == 1).to_numpy(dtype=bool)
    subset = subset[keep].assign(object=objects[keep].map(lambda o: str(o[0])))
    view = _keep(subset, ["dataset", "scene", "camera_id", "frame", "qtype"])
    cols = [*view, "id", "object", "value", "ref", "e", "mu", "sigma", "half", "fold", "fixed_accept"]
    merged = subset[cols].merge(subset[cols], on=view, suffixes=("_a", "_b"))
    merged = merged[merged["id_a"] < merged["id_b"]]
    same = merged["object_a"] == merged["object_b"]
    counts["n_same_object"] = int(same.sum())
    merged = merged[~same].reset_index(drop=True)
    if (merged["fold_a"] != merged["fold_b"]).any():
        raise ValueError("two answers of one scene and question type are in different folds")
    mu_m = merged["mu_a"].to_numpy(float) - merged["mu_b"].to_numpy(float)
    m_hat = merged["value_a"].to_numpy(float) - merged["value_b"].to_numpy(float)
    pairs = merged[[*view]].assign(
        fold=merged["fold_a"].to_numpy(dtype=np.int64), id_a=merged["id_a"], id_b=merged["id_b"],
        object_a=merged["object_a"], object_b=merged["object_b"], m_hat=m_hat,
        m=merged["ref_a"].to_numpy(float) - merged["ref_b"].to_numpy(float),
        e_m=merged["e_a"].to_numpy(float) - merged["e_b"].to_numpy(float), mu_m=mu_m, c_m=m_hat - mu_m,
        s_m=np.hypot(merged["sigma_a"].to_numpy(float), merged["sigma_b"].to_numpy(float)),
        resid_a=merged["e_a"].to_numpy(float) - merged["mu_a"].to_numpy(float),
        resid_b=merged["e_b"].to_numpy(float) - merged["mu_b"].to_numpy(float),
        h_sum=merged["half_a"].to_numpy(float) + merged["half_b"].to_numpy(float),
        h_quad=np.hypot(merged["half_a"].to_numpy(float), merged["half_b"].to_numpy(float)),
        fixed_accept=(merged["fixed_accept_a"].to_numpy(dtype=bool) & merged["fixed_accept_b"].to_numpy(dtype=bool)))
    counts["n_pairs"] = int(len(pairs))
    return pairs, counts


def direct_margin_halfwidth(pairs: pd.DataFrame, level: float = 0.9) -> pd.DataFrame:
    """Split-conformal error bar of each margin, calibrated on margins from other folds.

    Score ``|e_m - mu_m| / s_m``. For every test scene and question type, the ``level`` quantile
    (:func:`~measured_space.audit.conformal_quantile`) is taken over the pairs of that question type in scenes
    outside the test scene's fold, and the half-width is ``q * s_m``.

    Parameters
    ----------
    pairs : pandas.DataFrame
        :func:`margin_pairs` output (``dataset, qtype, scene, fold, e_m, mu_m, s_m``).
    level : float, default 0.9
        Nominal coverage.

    Returns
    -------
    pandas.DataFrame
        ``pairs`` plus ``q_direct`` (score quantile; ``inf`` with too few calibration pairs), ``h_direct``
        (half-width, m) and ``n_cal_direct`` (calibration pairs).
    """
    out = pairs.copy()
    q = np.full(len(out), np.nan)
    n_cal = np.zeros(len(out), dtype=np.int64)
    score = np.abs(out["e_m"].to_numpy(float) - out["mu_m"].to_numpy(float)) / out["s_m"].to_numpy(float)
    for _, idx in out.groupby(["dataset", "qtype"], sort=True).indices.items():
        g_scene = out["scene"].to_numpy()[idx]
        g_fold = out["fold"].to_numpy()[idx]
        for scene in np.unique(g_scene):
            test = g_scene == scene
            cal = g_fold != g_fold[test][0]
            q[idx[test]] = conformal_quantile(score[idx][cal], level)
            n_cal[idx[test]] = int(cal.sum())
    out["q_direct"] = q
    out["h_direct"] = q * out["s_m"].to_numpy(float)
    out["n_cal_direct"] = n_cal
    return out


def margin_decisions(pairs: pd.DataFrame, fit_margin: float = 0.1,
                     fit_qtypes: Sequence[str] = FIT_QTYPES) -> pd.DataFrame:
    """Decisions about margins, scored with the direct half-width.

    - ``sign``: "is A greater than B?" (``m > 0``; call ``c_m > 0``), every question type.
    - ``fit``: "is A smaller than B by at least ``fit_margin``?" (``m < -fit_margin``; call
      ``c_m < -fit_margin``), size question types only.

    Parameters
    ----------
    pairs : pandas.DataFrame
        :func:`direct_margin_halfwidth` output.
    fit_margin : float, default 0.1
        Clearance, m.
    fit_qtypes : sequence of str, default :data:`FIT_QTYPES`
        Question types of the ``fit`` decision.

    Returns
    -------
    pandas.DataFrame
        One row per pair and decision: ``dataset, scene, qtype, fold, fixed_accept`` (and ``test_scene`` for
        calibration pairs), ``decision, threshold`` (0 or ``fit_margin``, m), ``call``, ``truth``, ``wrong`` and
        ``score`` (``h_direct / |c_m - t|`` in the decision's orientation; :func:`two_sided`).
    """
    keep = _keep(pairs, ["dataset", "scene", "qtype", "fold", "fixed_accept"])
    parts = []
    c, m, h = (pairs[k].to_numpy(float) for k in ("c_m", "m", "h_direct"))
    sign = pairs[keep].assign(decision="sign", threshold=0.0)
    sign["call"], sign["wrong"], sign["score"] = two_sided(c, 0.0, m, h)
    sign["truth"] = m > 0
    parts.append(sign)
    is_fit = pairs["qtype"].isin(list(fit_qtypes)).to_numpy(dtype=bool)
    if is_fit.any():
        fit = pairs.loc[is_fit, keep].assign(decision="fit", threshold=float(fit_margin))
        fit["call"], fit["wrong"], fit["score"] = two_sided(-c[is_fit], fit_margin, -m[is_fit], h[is_fit])
        fit["truth"] = -m[is_fit] > fit_margin
        parts.append(fit)
    return pd.concat(parts, ignore_index=True)


# ============================================================================================================
# 4. Certification
# ============================================================================================================


def certify(decisions: pd.DataFrame, fold_of: Mapping[tuple[str, str, str], int], epsilon: float, delta: float, *,
            pool_cols: Sequence[str] = (), min_accept: int = AuditConfig.min_accept,
            min_accept_frac: float = AuditConfig.min_accept_frac, max_threshold: float = NEVER,
            calibration: pd.DataFrame | None = None,
            exact: Collection[tuple[str, str, str]] | None = None) -> tuple[BoolArray, Array, pd.DataFrame]:
    """Accept decisions whose score is at most a per-scene threshold certified by Learn-then-Test.

    **Approximate** (``calibration`` None): within each pool (``pool_cols``), for every test scene ``S``, the
    calibration decisions are those of the pool's other scenes outside ``S``'s fold: for a decision of question
    type ``q``, outside the fold that ``S`` has for ``q`` (scene ``S`` alone if ``S`` has no answer of type
    ``q``). They are those scenes' own test decisions, so their predictions come from models that saw ``S``'s
    fold.

    **Exact** (``calibration`` given, built from the audit's saved leave-two-out predictions by
    :func:`calibration_answers`): for a question type ``q`` whose group the audit cross-fitted with ``S`` in it
    (``(dataset, q, S)`` in ``exact``), the calibration decisions of ``S`` are ``calibration``'s rows with
    ``test_scene == S``: items outside ``S``'s fold predicted by models that saw neither ``S``'s fold nor their
    own. For a question type whose group does not hold ``S`` (``S`` has no answer of that type in any depth
    source, so no model or quantile of that group ever saw ``S``), they are the pool's own test decisions of
    that type, as in the approximate mode.

    Either way the audit's :func:`~measured_space.audit.ltt_threshold` then picks ``lambda`` so that
    ``P(wrong | score <= lambda) <= epsilon`` at confidence ``1 - delta`` (if calibration and test decisions are
    i.i.d.).

    Parameters
    ----------
    decisions : pandas.DataFrame
        One labeller's decisions: ``dataset, qtype, scene, fold, score, wrong`` and the ``pool_cols``.
    fold_of : mapping
        ``(dataset, qtype, scene) -> fold`` (:func:`fold_lookup` of the labeller's answers).
    epsilon : float
        Bound on the wrong-decision rate among accepted decisions.
    delta : float
        Allowed probability that the certificate is wrong.
    pool_cols : sequence of str, default ()
        Columns that split the decisions into separately calibrated pools; empty pools everything.
    min_accept : int, default: the audit's
        Learn-then-Test's first candidate accepts at least this many calibration decisions ...
    min_accept_frac : float, default: the audit's
        ... and at least this share of them.
    max_threshold : float, default :data:`NEVER`
        Candidates above this are not tested (with the default, an infinite score is never accepted).
    calibration : pandas.DataFrame, optional
        Calibration decisions of the exact mode: ``test_scene, dataset, score, wrong`` and the ``pool_cols``
        (decisions built from :func:`calibration_answers`). None: the approximate mode.
    exact : collection of tuple, optional
        ``(dataset, qtype, test scene)`` groups that ``calibration`` covers (:func:`exact_groups`); required
        with ``calibration``.

    Returns
    -------
    accept : numpy.ndarray of bool, shape (n,)
        ``score <= lambda`` of the decision's pool and scene.
    lam : numpy.ndarray of float, shape (n,)
        The threshold applied to each decision (``-inf``: nothing certified).
    thresholds : pandas.DataFrame
        One row per pool and test scene: the pool keys, ``scene``, ``lam`` and the
        :func:`~measured_space.audit.ltt_threshold` info (``n_cal``, ``n_tested``, ``n_accepted_cal``,
        ``err_accepted_cal``, ``p_value``); in the exact mode also ``n_cal_exact`` (from ``calibration``) and
        ``n_cal_own`` (own test decisions of question types whose group does not hold the scene).

    Raises
    ------
    ValueError
        If ``calibration`` is given without ``exact``.
    """
    n = len(decisions)
    lam = np.full(n, -np.inf)
    score = decisions["score"].to_numpy(dtype=np.float64)
    wrong = decisions["wrong"].to_numpy(dtype=np.float64)
    scene = decisions["scene"].astype(str).to_numpy()
    qtype = decisions["qtype"].astype(str).to_numpy()
    dataset = decisions["dataset"].astype(str).to_numpy()
    fold = decisions["fold"].to_numpy(dtype=np.int64)
    groups: dict[Any, NDArray[np.intp]] = (decisions.groupby(list(pool_cols), sort=True, dropna=False).indices
                                           if pool_cols else {(): np.arange(n)})
    if calibration is not None and exact is None:
        raise ValueError("exact calibration needs the groups it covers (exact)")
    cal_groups: dict[tuple[Any, ...], NDArray[np.intp]] = {}
    exact_set = set(exact or ())
    if calibration is not None:
        c_score = calibration["score"].to_numpy(dtype=np.float64)
        c_wrong = calibration["wrong"].to_numpy(dtype=np.float64)
        c_data = calibration["dataset"].astype(str).to_numpy()
        by = calibration.assign(**{_TEST_SCENE: calibration[_TEST_SCENE].astype(str)}).groupby(
            [*pool_cols, _TEST_SCENE], sort=False, dropna=False).indices
        cal_groups = {(k if isinstance(k, tuple) else (k,)): v for k, v in by.items()}
    rows: list[dict[str, Any]] = []
    for key, gidx in groups.items():
        key_t = key if isinstance(key, tuple) else (key,)
        g_scene, g_fold, g_data = scene[gidx], fold[gidx], dataset[gidx]
        g_qcode, g_quniq = pd.factorize(qtype[gidx])
        g_score, g_wrong = score[gidx], wrong[gidx]
        for s in np.unique(g_scene):
            test = g_scene == s
            d = g_data[test][0]
            extra: dict[str, int] = {}
            if calibration is None:
                fold_s = np.array([fold_of.get((d, str(q), str(s)), -1) for q in g_quniq], dtype=np.int64)[g_qcode]
                cal = (g_fold != fold_s) & (g_scene != s) & (g_data == d)
                cal_score, cal_wrong = g_score[cal], g_wrong[cal]
            else:
                covered = np.array([(d, str(q), str(s)) in exact_set for q in g_quniq], dtype=bool)[g_qcode]
                own = ~covered & (g_scene != s) & (g_data == d)
                cidx = cal_groups.get((*key_t, str(s)), np.array([], dtype=np.intp))
                cidx = cidx[c_data[cidx] == d]
                cal_score = np.concatenate([c_score[cidx], g_score[own]])
                cal_wrong = np.concatenate([c_wrong[cidx], g_wrong[own]])
                extra = {"n_cal_exact": int(len(cidx)), "n_cal_own": int(own.sum())}
            value, info = ltt_threshold(cal_score, cal_wrong, epsilon, delta, min_accept=min_accept,
                                        min_accept_frac=min_accept_frac, max_threshold=max_threshold)
            lam[gidx[test]] = value
            rows.append({**dict(zip(pool_cols, key_t)), "scene": s, "lam": value, **info, **extra})
    return score <= lam, lam, pd.DataFrame(rows)


# ============================================================================================================
# 5. Reports
# ============================================================================================================


def _salt(*parts: Any) -> int:
    """A stable bootstrap stream number for one reported group.

    Parameters
    ----------
    *parts : Any
        The group's identifying values (their ``repr`` is hashed).

    Returns
    -------
    int
        CRC-32 of the parts' representation.
    """
    return zlib.crc32(repr(parts).encode())


def rate_summary(flags: ArrayLike, clusters: ArrayLike, boot: AuditConfig, salt: int) -> dict[str, float]:
    """A rate with its Wilson and scene-cluster bootstrap 95 % intervals.

    Parameters
    ----------
    flags : array_like of bool, shape (n,)
        One flag per item (e.g. a wrong decision).
    clusters : array_like, shape (n,)
        The scene of each item.
    boot : AuditConfig
        Bootstrap replicates and seed.
    salt : int
        Bootstrap stream (see :func:`~measured_space.audit.cluster_interval`).

    Returns
    -------
    dict of str to float
        ``n``, ``k``, ``rate`` (NaN when ``n == 0``), ``lo``, ``hi`` (Wilson), ``boot_lo``, ``boot_hi``.
    """
    values = np.asarray(flags, dtype=np.float64)
    n, k = len(values), float(values.sum())
    lo, hi = wilson_interval(k, n)
    boot_lo, boot_hi = cluster_interval(values, np.asarray(clusters), boot, salt) if n else (math.nan, math.nan)
    return {"n": n, "k": k, "rate": k / n if n else math.nan, "lo": float(lo), "hi": float(hi),
            "boot_lo": boot_lo, "boot_hi": boot_hi}


def rule_table(decisions: pd.DataFrame, keys: Sequence[str], rules: Sequence[Rule], boot: AuditConfig, *,
               inside: str | None = None) -> pd.DataFrame:
    """Accepted share and wrong-decision rate among accepted decisions, per group and rule.

    Parameters
    ----------
    decisions : pandas.DataFrame
        Decisions with ``scene``, ``wrong``, the ``keys`` columns and every rule's ``accept`` (and ``lam``)
        column.
    keys : sequence of str
        Grouping columns (missing values form their own group).
    rules : sequence of Rule
        Rules to report.
    boot : AuditConfig
        Bootstrap replicates and seed.
    inside : str, optional
        Boolean column of truly-inside cases (one-sided decisions): adds the share of them each rule accepted.

    Returns
    -------
    pandas.DataFrame
        One row per group and rule: the keys, ``rule``, ``epsilon``, ``n`` (decisions), ``n_scenes``,
        ``accept_n``, ``accept_share``, ``wrong_n``, ``wrong_rate`` (among accepted), ``wrong_lo``,
        ``wrong_hi`` (Wilson 95 %), ``wrong_boot_lo``, ``wrong_boot_hi`` (scene-cluster bootstrap 95 %),
        ``certified_ok`` (certified rules: nothing accepted or ``wrong_rate <= epsilon``; else NaN),
        ``n_scenes_none`` and ``lam_p50`` (certified rules: scenes where no pool certified a threshold, and the
        median of the distinct per-scene, per-pool thresholds, ``-inf`` counted); with ``inside``:
        ``inside_n``, ``inside_accepted_n``, ``inside_accepted_rate`` and its Wilson ``inside_accepted_lo``,
        ``inside_accepted_hi``.
    """
    out: list[dict[str, Any]] = []
    groups = decisions.groupby(list(keys), sort=True, dropna=False) if keys else [((), decisions)]
    for key, g in groups:
        key_t = key if isinstance(key, tuple) else (key,)
        base = dict(zip(keys, key_t))
        clusters = (g["dataset"].astype(str) + "/" + g["scene"].astype(str)).to_numpy()
        wrong = g["wrong"].to_numpy(dtype=bool)
        for rule in rules:
            acc = g[rule.accept].to_numpy(dtype=bool)
            r = rate_summary(wrong[acc], clusters[acc], boot, _salt(*key_t, rule.name, rule.epsilon))
            row: dict[str, Any] = {**base, "rule": rule.name, "epsilon": rule.epsilon, "n": len(g),
                                   "n_scenes": int(g["scene"].nunique()), "accept_n": int(acc.sum()),
                                   "accept_share": float(acc.mean()) if len(g) else math.nan,
                                   "wrong_n": int(r["k"]), "wrong_rate": r["rate"], "wrong_lo": r["lo"],
                                   "wrong_hi": r["hi"], "wrong_boot_lo": r["boot_lo"],
                                   "wrong_boot_hi": r["boot_hi"]}
            certified = not math.isnan(rule.epsilon)
            row["certified_ok"] = (bool(acc.sum() == 0 or r["rate"] <= rule.epsilon) if certified else math.nan)
            if rule.lam is not None:
                lam = g[["scene", rule.lam]].drop_duplicates()
                any_finite = lam.groupby("scene")[rule.lam].agg(lambda x: bool(np.isfinite(x).any()))
                row["n_scenes_none"] = int((~any_finite.astype(bool)).sum())
                row["lam_p50"] = float(np.median(lam[rule.lam].to_numpy(float))) if len(lam) else math.nan
            else:
                row["n_scenes_none"], row["lam_p50"] = math.nan, math.nan
            if inside is not None:
                ins = g[inside].to_numpy(dtype=bool)
                k_in = int((acc & ins).sum())
                lo, hi = wilson_interval(k_in, int(ins.sum()))
                row.update(inside_n=int(ins.sum()), inside_accepted_n=k_in,
                           inside_accepted_rate=k_in / ins.sum() if ins.sum() else math.nan,
                           inside_accepted_lo=float(lo), inside_accepted_hi=float(hi))
            out.append(row)
    return pd.DataFrame(out)


MARGIN_METHODS: dict[str, str] = {"sum": "h_sum", "quadrature": "h_quad", "direct": "h_direct"}
"""dict: margin error-bar methods and their half-width columns."""


def margin_interval_table(pairs: pd.DataFrame, keys: Sequence[str], boot: AuditConfig) -> pd.DataFrame:
    """Coverage and width of the three margin error bars, per group.

    Parameters
    ----------
    pairs : pandas.DataFrame
        :func:`direct_margin_halfwidth` output with the ``keys`` columns.
    keys : sequence of str
        Grouping columns.
    boot : AuditConfig
        Bootstrap replicates and seed.

    Returns
    -------
    pandas.DataFrame
        One row per group and method (:data:`MARGIN_METHODS`): the keys, ``method``, ``n_pairs``,
        ``n_scenes``, ``coverage`` (``|e_m - mu_m| <= half``) with Wilson ``cov_lo``, ``cov_hi`` and
        scene-bootstrap ``cov_boot_lo``, ``cov_boot_hi``; ``half_p50``, ``half_mean`` (m; finite half-widths),
        ``half_over_quadrature_p50`` (median ratio to the quadrature half-width), ``n_infinite`` (pairs whose
        half-width is infinite: too few calibration pairs), ``resid_corr`` (Pearson correlation of the two
        answers' residuals ``e - mu``; positive: shared errors that cancel in the margin) and ``n_ties``
        (``|m| < 1 mm``).
    """
    out: list[dict[str, Any]] = []
    groups = pairs.groupby(list(keys), sort=True, dropna=False) if keys else [((), pairs)]
    for key, g in groups:
        key_t = key if isinstance(key, tuple) else (key,)
        clusters = (g["dataset"].astype(str) + "/" + g["scene"].astype(str)).to_numpy()
        resid = np.abs(g["e_m"].to_numpy(float) - g["mu_m"].to_numpy(float))
        ra, rb = g["resid_a"].to_numpy(float), g["resid_b"].to_numpy(float)
        corr = float(np.corrcoef(ra, rb)[0, 1]) if len(g) > 2 and ra.std() > 0 and rb.std() > 0 else math.nan
        quad = g["h_quad"].to_numpy(float)
        for method, col in MARGIN_METHODS.items():
            half = g[col].to_numpy(float)
            covered = resid <= half
            r = rate_summary(covered, clusters, boot, _salt(*key_t, method))
            finite = np.isfinite(half)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = half / quad
            out.append({**dict(zip(keys, key_t)), "method": method, "n_pairs": len(g),
                        "n_scenes": int(g["scene"].nunique()), "coverage": r["rate"], "cov_lo": r["lo"],
                        "cov_hi": r["hi"], "cov_boot_lo": r["boot_lo"], "cov_boot_hi": r["boot_hi"],
                        "half_p50": float(np.median(half[finite])) if finite.any() else math.nan,
                        "half_mean": float(np.mean(half[finite])) if finite.any() else math.nan,
                        "half_over_quadrature_p50": float(np.median(ratio[finite])) if finite.any() else math.nan,
                        "n_infinite": int((~finite).sum()), "resid_corr": corr,
                        "n_ties": int(np.sum(np.abs(g["m"].to_numpy(float)) < _TIE))})
    return pd.DataFrame(out)


def standard_rules(epsilons: Sequence[float], *, accept_all: str = "accept_all") -> list[Rule]:
    """The rules reported for every decision analysis.

    Parameters
    ----------
    epsilons : sequence of float
        Certified bounds; rule ``certified`` is reported once per value, with columns ``accept_eps<k>`` and
        ``lam_eps<k>`` (``k`` the position in ``epsilons``).
    accept_all : str, default "accept_all"
        Column of the no-check baseline (all decisions, or for one-sided decisions all outside calls).

    Returns
    -------
    list of Rule
        ``certified`` (per epsilon), ``interval`` (``score <= 1``: the error bar excludes the threshold, column
        ``accept_interval``), ``accept_all`` and ``fixed_tau_gate`` (column ``accept_fixed``).
    """
    rules = [Rule("certified", f"accept_eps{k}", float(eps), f"lam_eps{k}") for k, eps in enumerate(epsilons)]
    return rules + [Rule("interval", "accept_interval"), Rule("accept_all", accept_all),
                    Rule("fixed_tau_gate", "accept_fixed")]


def apply_rules(decisions: pd.DataFrame, fold_of: Mapping[tuple[str, str, str], int], config: DecisionConfig,
                pool_cols: Sequence[str], *, one_sided: bool = False, calibration: pd.DataFrame | None = None,
                exact: Collection[tuple[str, str, str]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add every rule's acceptance column to one labeller's decisions.

    Parameters
    ----------
    decisions : pandas.DataFrame
        One labeller's decisions (``score``, ``wrong``, ``call``, ``fixed_accept``, ``dataset, qtype, scene,
        fold`` and the ``pool_cols``).
    fold_of : mapping
        ``(dataset, qtype, scene) -> fold`` (:func:`fold_lookup`).
    config : DecisionConfig
        Epsilons, delta and the Learn-then-Test settings.
    pool_cols : sequence of str
        Calibration pools (see :func:`certify`).
    one_sided : bool, default False
        One-sided decisions: the no-check baseline and the fixed gate accept outside calls only.
    calibration : pandas.DataFrame, optional
        Exact-mode calibration decisions (see :func:`certify`); None: approximate.
    exact : collection of tuple, optional
        The groups ``calibration`` covers (:func:`exact_groups`).

    Returns
    -------
    decisions : pandas.DataFrame
        A copy with ``accept_eps<k>``, ``lam_eps<k>`` (per epsilon), ``accept_interval``, ``accept_all``,
        ``accept_fixed`` (bool) added; one-sided decisions also get ``accept_outside`` (every outside call).
    thresholds : pandas.DataFrame
        :func:`certify`'s per-scene thresholds, with ``epsilon``.
    """
    out = decisions.copy()
    tables = []
    for k, eps in enumerate(config.epsilons):
        accept, lam, thr = certify(out, fold_of, eps, config.delta, pool_cols=pool_cols,
                                   min_accept=config.min_accept, min_accept_frac=config.min_accept_frac,
                                   calibration=calibration, exact=exact)
        out[f"accept_eps{k}"], out[f"lam_eps{k}"] = accept, lam
        tables.append(thr.assign(epsilon=eps))
    score = out["score"].to_numpy(float)
    out["accept_interval"] = score <= 1.0
    out["accept_all"] = True
    fixed = out["fixed_accept"].to_numpy(dtype=bool)
    if one_sided:
        call = out["call"].to_numpy(dtype=bool)
        out["accept_outside"] = call
        out["accept_fixed"] = fixed & call
    else:
        out["accept_fixed"] = fixed
    return out, pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


# ============================================================================================================
# 6. Exact calibration from the audit's saved leave-two-out predictions
# ============================================================================================================


CALIBRATION_MODES: tuple[str, ...] = ("exact", "approximate")
"""tuple of str: how calibration decisions were built: from the audit's saved leave-two-out predictions, or from
other folds' test predictions (audit folders that did not save them)."""


def exact_groups(quantiles: pd.DataFrame, kind: str = "sigma") -> set[tuple[str, str, str]]:
    """The groups the audit calibrated with leave-two-out predictions: every test scene of every question type.

    Parameters
    ----------
    quantiles : pandas.DataFrame
        The audit's saved per-test-scene conformal quantiles (``calibration_quantiles.csv.gz``: ``dataset,
        kind, test_scene, qtype``).
    kind : str, default "sigma"
        Error model.

    Returns
    -------
    set of tuple of str
        ``(dataset, qtype, test scene)`` for every scene the audit cross-fitted in that question type's group
        (any depth source).
    """
    q = quantiles[quantiles["kind"] == kind]
    return set(zip(q["dataset"].astype(str), q["qtype"].astype(str), q["test_scene"].astype(str)))


def calibration_answers(answers: pd.DataFrame, predictions: pd.DataFrame, quantiles: pd.DataFrame, *,
                        level: float = 0.9, kind: str = "sigma") -> pd.DataFrame:
    """Each test scene's calibration answers, rebuilt from the audit's saved leave-two-out predictions.

    For test scene ``S`` and a calibration item (an answer of this labeller in a scene outside ``S``'s fold),
    ``mu`` and ``sigma`` are the prediction of the model fitted without ``S``'s fold and the item's own fold,
    and the half-width is ``q * sigma`` with ``q`` the quantile that ``S``'s own answers of the item's Mondrian
    cell (question type, depth source) get: so the calibration answer has its error bar exactly as an answer of
    ``S`` would. ``c = value - mu``; the reference, the id and the view are the item's.

    Parameters
    ----------
    answers : pandas.DataFrame
        :func:`answer_table` output (one labeller, one depth source).
    predictions : pandas.DataFrame
        The audit's ``calibration_numeric.csv.gz``: ``kind, test_scene, row, mu, sigma`` (row indexes
        ``pairs.csv``).
    quantiles : pandas.DataFrame
        The audit's ``calibration_quantiles.csv.gz``: ``dataset, kind, test_scene, qtype, depth_source`` and
        ``q<level>``.
    level : float, default 0.9
        Coverage level of the error bar.
    kind : str, default "sigma"
        Error model (the one ``answers`` came from).

    Returns
    -------
    pandas.DataFrame
        ``test_scene`` followed by :func:`answer_table`'s columns, one row per test scene and calibration
        answer of this depth source, sorted by test scene and row; ``mu, sigma, half, c`` are the leave-two-out
        ones (m; ``half`` is ``inf`` where the quantile is).

    Raises
    ------
    KeyError
        If ``quantiles`` lacks the level's column.
    ValueError
        If a calibration item lies in its test scene or in the test scene's fold, or has no quantile.
    """
    tag = f"q{int(round(100 * level))}"
    if tag not in quantiles.columns:
        raise KeyError(f"quantile table lacks {tag}")
    pred = predictions.loc[predictions["kind"] == kind, ["test_scene", "row", "mu", "sigma"]]
    pred = pred.assign(test_scene=pred["test_scene"].astype(str), row=pred["row"].astype(np.int64))
    cal = pred.merge(answers.drop(columns=["mu", "sigma", "half", "c"]), on="row", how="inner",
                     validate="many_to_one")
    q = quantiles.loc[quantiles["kind"] == kind, ["dataset", "test_scene", "qtype", "depth_source", tag]]
    q = q.assign(test_scene=q["test_scene"].astype(str), depth_source=q["depth_source"].astype(str))
    cal = cal.merge(q, on=["dataset", "test_scene", "qtype", "depth_source"], how="left", validate="many_to_one")
    if cal[tag].isna().any():
        raise ValueError(f"{int(cal[tag].isna().sum())} calibration answers have no quantile for their cell")
    if (cal["scene"] == cal["test_scene"]).any():
        raise ValueError("a calibration answer lies in its own test scene")
    fold_of = fold_lookup(answers)
    test_fold = np.array([fold_of.get(k, -1) for k in zip(cal["dataset"].astype(str), cal["qtype"].astype(str),
                                                          cal["test_scene"])], dtype=np.int64)
    if (test_fold == cal["fold"].to_numpy(dtype=np.int64)).any():
        raise ValueError("a calibration answer lies in its test scene's fold")
    sigma = cal["sigma"].to_numpy(float)
    half = cal[tag].to_numpy(float) * sigma
    cal["half"] = np.where(np.isnan(half), np.inf, half)
    cal["c"] = cal["value"].to_numpy(float) - cal["mu"].to_numpy(float)
    cal = cal[[_TEST_SCENE, *answers.columns]]
    return cal.sort_values([_TEST_SCENE, "row"], kind="stable").reset_index(drop=True)


def check_test_halfwidths(answers: pd.DataFrame, quantiles: pd.DataFrame, *, level: float = 0.9,
                          kind: str = "sigma") -> dict[str, Any]:
    """Check that the saved quantiles are the ones that gave the test answers their half-widths.

    Parameters
    ----------
    answers : pandas.DataFrame
        :func:`answer_table` output (``half`` and ``sigma`` as the audit wrote them, 6 significant digits).
    quantiles : pandas.DataFrame
        The audit's saved quantile table.
    level : float, default 0.9
        Coverage level.
    kind : str, default "sigma"
        Error model.

    Returns
    -------
    dict
        ``n`` (finite answers compared), ``max_rel_diff`` of ``half`` against ``q * sigma``, ``n_missing``
        (answers whose cell is not in the table) and ``n_infinite_agree`` (answers whose half-width and
        ``q * sigma`` are both infinite).
    """
    tag = f"q{int(round(100 * level))}"
    q = quantiles.loc[quantiles["kind"] == kind, ["dataset", "test_scene", "qtype", "depth_source", tag]]
    q = q.assign(scene=q["test_scene"].astype(str), depth_source=q["depth_source"].astype(str))
    m = answers.merge(q.drop(columns=["test_scene"]), on=["dataset", "scene", "qtype", "depth_source"], how="left")
    expected = m[tag].to_numpy(float) * m["sigma"].to_numpy(float)
    half = m["half"].to_numpy(float)
    ok = np.isfinite(expected) & np.isfinite(half)
    rel = np.abs(half[ok] - expected[ok]) / np.maximum(np.abs(expected[ok]), 1e-12)
    return {"n": int(ok.sum()), "max_rel_diff": float(rel.max()) if ok.any() else math.nan,
            "n_missing": int(m[tag].isna().sum()),
            "n_infinite_agree": int((~np.isfinite(expected) & ~np.isfinite(half)).sum())}


def direct_margin_halfwidth_exact(pairs: pd.DataFrame, cal_pairs: pd.DataFrame,
                                  level: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Direct margin error bars calibrated on leave-two-out margins: the exact counterpart of
    :func:`direct_margin_halfwidth`.

    For test scene ``S`` and question type ``q``, the ``level`` quantile of ``|e_m - mu_m| / s_m`` is taken over
    ``cal_pairs`` with ``test_scene == S``: margins in scenes outside ``S``'s fold whose ``mu`` and ``sigma`` come
    from models that saw neither ``S``'s fold nor the margin's. Test pairs of ``S`` get ``q * s_m``; so does each
    calibration pair of ``S`` (with its own leave-two-out ``s_m``), as ``S``'s own margins would.

    Parameters
    ----------
    pairs : pandas.DataFrame
        :func:`margin_pairs` of the test answers.
    cal_pairs : pandas.DataFrame
        :func:`margin_pairs` of :func:`calibration_answers` (with ``test_scene``).
    level : float, default 0.9
        Nominal coverage.

    Returns
    -------
    pairs : pandas.DataFrame
        ``pairs`` plus ``q_direct`` (``inf`` with too few calibration pairs), ``h_direct`` (m) and
        ``n_cal_direct``.
    cal_pairs : pandas.DataFrame
        ``cal_pairs`` plus the same columns, from its test scene's quantile.
    """
    score = (np.abs(cal_pairs["e_m"].to_numpy(float) - cal_pairs["mu_m"].to_numpy(float))
             / cal_pairs["s_m"].to_numpy(float))
    keys = cal_pairs.assign(**{_TEST_SCENE: cal_pairs[_TEST_SCENE].astype(str)}).groupby(
        ["dataset", "qtype", _TEST_SCENE], sort=True).indices
    quantile = {(str(k[0]), str(k[1]), str(k[2])): (conformal_quantile(score[idx], level), len(idx))
                for k, idx in keys.items()}

    def attach(frame: pd.DataFrame, scene_col: str) -> pd.DataFrame:
        """Add the direct quantile, half-width and calibration count of each pair's test scene.

        Parameters
        ----------
        frame : pandas.DataFrame
            Margin pairs (``dataset, qtype, s_m`` and ``scene_col``).
        scene_col : str
            Column naming the test scene whose quantile applies.

        Returns
        -------
        pandas.DataFrame
            ``frame`` plus ``q_direct``, ``h_direct`` and ``n_cal_direct``.
        """
        found = [quantile.get((str(d), str(q), str(s)), (math.inf, 0))
                 for d, q, s in zip(frame["dataset"], frame["qtype"], frame[scene_col])]
        q_direct = np.array([f[0] for f in found], dtype=np.float64)
        return frame.assign(q_direct=q_direct, h_direct=q_direct * frame["s_m"].to_numpy(float),
                            n_cal_direct=np.array([f[1] for f in found], dtype=np.int64))

    return attach(pairs, "scene"), attach(cal_pairs, _TEST_SCENE)


# ============================================================================================================
# 7. A second opinion: a witness labeller's answer to the same question
# ============================================================================================================


def complete_calibration_answers(answers: pd.DataFrame, cal_answers: pd.DataFrame,
                                 exact: Collection[tuple[str, str, str]]) -> tuple[pd.DataFrame,
                                                                                    set[tuple[str, str, str]]]:
    """Every test scene's whole calibration set as rows: :func:`certify`'s exact mode written out.

    For a test scene ``S`` and a question type whose group holds ``S`` (``(dataset, qtype, S)`` in ``exact``),
    the rows are ``cal_answers``' (leave-two-out); for any other question type they are the other scenes' own
    test answers, copied with ``test_scene = S``. Calibrating with these rows and every group marked exact is
    the same as :func:`certify`'s exact mode; written out, a per-test-scene quantity (a witness's answer) can be
    attached to every calibration row.

    Parameters
    ----------
    answers : pandas.DataFrame
        :func:`answer_table` output.
    cal_answers : pandas.DataFrame
        :func:`calibration_answers` output for the same labeller.
    exact : collection of tuple
        :func:`exact_groups` of the audit.

    Returns
    -------
    calibration : pandas.DataFrame
        ``cal_answers``' columns, the copied own answers appended.
    covered : set of tuple
        ``(dataset, qtype, S)`` for every test scene ``S`` of ``answers`` and every question type: pass it as
        :func:`certify`'s ``exact`` with decisions built from ``calibration``.
    """
    exact_set = set(exact)
    qtypes = sorted(answers["qtype"].astype(str).unique())
    parts = [cal_answers]
    covered: set[tuple[str, str, str]] = set()
    for d, s in answers[["dataset", "scene"]].drop_duplicates().astype(str).itertuples(index=False):
        covered.update((d, q, s) for q in qtypes)
        missing = [q for q in qtypes if (d, q, s) not in exact_set]
        own = answers[(answers["dataset"] == d) & answers["qtype"].isin(missing) & (answers["scene"] != s)]
        if len(own):
            parts.append(own.assign(**{_TEST_SCENE: s})[[_TEST_SCENE, *answers.columns]])
    return pd.concat(parts, ignore_index=True), covered


def attach_witness(decisions: pd.DataFrame, witness: pd.DataFrame,
                   witness_cal: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add a witness labeller's bias-corrected answer and error bar to each decision, by question id.

    Test decisions get the witness's own out-of-scene test answer. Calibration decisions (with ``test_scene``)
    get, when ``witness_cal`` holds one, the witness's leave-two-out answer for that test scene (from a model
    that saw neither the test scene's fold nor the item's, with the test scene's quantile); otherwise the
    witness's test answer, which then never saw the test scene either: the test scene is outside the witness's
    group of that question type, or the item shares the test scene's fold so the witness's model held both out.

    Parameters
    ----------
    decisions : pandas.DataFrame
        One labeller's decisions with ``id`` (and ``test_scene`` for calibration decisions).
    witness : pandas.DataFrame
        The witness's :func:`answer_table` (``id, c, half, ref``), one answer per question id.
    witness_cal : pandas.DataFrame, optional
        The witness's :func:`calibration_answers` (``test_scene, id, c, half``).

    Returns
    -------
    pandas.DataFrame
        ``decisions`` (index reset) plus ``c_w``, ``half_w``, ``ref_w`` (m; NaN where the witness did not answer)
        and ``witness_from`` (``"leave_two_out"``, ``"test"`` or ``"none"``).

    Raises
    ------
    ValueError
        If the witness has two answers to one question id.
    """
    w = witness[["id", "c", "half", "ref"]].rename(columns={"c": "c_w", "half": "half_w", "ref": "ref_w"})
    if w["id"].duplicated().any():
        raise ValueError("the witness answers a question id twice")
    out = decisions.reset_index(drop=True).merge(w, on="id", how="left", validate="many_to_one")
    source = np.where(out["c_w"].isna(), "none", "test").astype(object)
    if witness_cal is not None and _TEST_SCENE in out.columns:
        wc = witness_cal[[_TEST_SCENE, "id", "c", "half"]].assign(**{_TEST_SCENE: witness_cal[_TEST_SCENE].astype(str)})
        found = out[[_TEST_SCENE, "id"]].assign(**{_TEST_SCENE: out[_TEST_SCENE].astype(str)}).merge(
            wc, on=[_TEST_SCENE, "id"], how="left", validate="many_to_one")
        use = found["c"].notna().to_numpy()
        out.loc[use, "c_w"] = found.loc[use, "c"].to_numpy()
        out.loc[use, "half_w"] = found.loc[use, "half"].to_numpy()
        source[use] = "leave_two_out"
    out["witness_from"] = source
    return out


def second_opinion_score(decisions: pd.DataFrame, *, one_sided: bool = False) -> Array:
    """Score of a decision checked by a witness: infinite unless both calls agree, else the larger of two scores.

    Two-sided: ``max(half / |c - t|, half_w / |c_w - t|)`` when ``c > t`` and ``c_w > t`` agree. One-sided
    (clearances): both calls must be outside (``c > t`` and ``c_w > t``), score ``max(half / (c - t),
    half_w / (c_w - t))``. A decision the witness did not answer, or with an undefined score, gets ``inf`` and
    is never accepted. ``score <= 1`` means both error bars exclude ``t`` on the same side.

    Parameters
    ----------
    decisions : pandas.DataFrame
        Decisions with ``c, t, half`` and the witness's ``c_w, half_w`` (:func:`attach_witness`); m.
    one_sided : bool, default False
        Clearance decisions (only outside calls).

    Returns
    -------
    numpy.ndarray of float, shape (n,)
        The scores.
    """
    c, t, half, c_w, half_w = (decisions[k].to_numpy(dtype=np.float64) for k in ("c", "t", "half", "c_w", "half_w"))
    if one_sided:
        agree = (c > t) & (c_w > t)
        gap, gap_w = c - t, c_w - t
    else:
        agree = (c > t) == (c_w > t)
        gap, gap_w = np.abs(c - t), np.abs(c_w - t)
    with np.errstate(divide="ignore", invalid="ignore"):
        own = np.where(gap > 0, half / gap, np.inf)
        other = np.where(gap_w > 0, half_w / gap_w, np.inf)
        score = np.where(agree & np.isfinite(c_w), np.maximum(own, other), np.inf)
    return np.where(np.isnan(score), np.inf, score)


def witness_rules(epsilons: Sequence[float], *, accept_all: str = "accept_all") -> list[Rule]:
    """The rules of the second-opinion comparison.

    Parameters
    ----------
    epsilons : sequence of float
        Certified bounds (columns ``accept_eps<k>`` / ``accept_wit_eps<k>``, ``k`` the position).
    accept_all : str, default "accept_all"
        Column of the no-check baseline.

    Returns
    -------
    list of Rule
        ``certified`` and ``second_opinion`` (per epsilon), ``interval`` (``score <= 1``),
        ``second_opinion_interval`` (both error bars exclude ``t`` on the same side, uncertified) and
        ``accept_all``.
    """
    rules = [Rule("certified", f"accept_eps{k}", float(eps), f"lam_eps{k}") for k, eps in enumerate(epsilons)]
    rules += [Rule("second_opinion", f"accept_wit_eps{k}", float(eps), f"lam_wit_eps{k}")
              for k, eps in enumerate(epsilons)]
    return rules + [Rule("interval", "accept_interval"), Rule("second_opinion_interval", "accept_wit_interval"),
                    Rule("accept_all", accept_all)]


def apply_witness_rule(decisions: pd.DataFrame, fold_of: Mapping[tuple[str, str, str], int], config: DecisionConfig,
                       pool_cols: Sequence[str], *, one_sided: bool = False, calibration: pd.DataFrame | None = None,
                       exact: Collection[tuple[str, str, str]] | None = None) -> pd.DataFrame:
    """Add the second-opinion rule's columns: its score, the certified acceptance per epsilon and the interval rule.

    Certification is :func:`certify`'s with the second-opinion score in place of the labeller's own, on the same
    calibration decisions (with the witness attached, see :func:`attach_witness`).

    Parameters
    ----------
    decisions : pandas.DataFrame
        Test decisions with the witness attached and the single-labeller rules applied (:func:`apply_rules`).
    fold_of : mapping
        ``(dataset, qtype, scene) -> fold`` (approximate mode).
    config : DecisionConfig
        Epsilons, delta and the Learn-then-Test settings.
    pool_cols : sequence of str
        Calibration pools.
    one_sided : bool, default False
        Clearance decisions.
    calibration : pandas.DataFrame, optional
        Calibration decisions with the witness attached (exact mode).
    exact : collection of tuple, optional
        The groups ``calibration`` covers.

    Returns
    -------
    pandas.DataFrame
        A copy with ``score_wit``, ``accept_wit_eps<k>``, ``lam_wit_eps<k>`` and ``accept_wit_interval`` added.
    """
    out = decisions.copy()
    out["score_wit"] = second_opinion_score(out, one_sided=one_sided)
    scored = out.assign(score=out["score_wit"])
    cal = (calibration.assign(score=second_opinion_score(calibration, one_sided=one_sided))
           if calibration is not None else None)
    for k, eps in enumerate(config.epsilons):
        accept, lam, _ = certify(scored, fold_of, eps, config.delta, pool_cols=pool_cols,
                                 min_accept=config.min_accept, min_accept_frac=config.min_accept_frac,
                                 calibration=cal, exact=exact)
        out[f"accept_wit_eps{k}"], out[f"lam_wit_eps{k}"] = accept, lam
    out["accept_wit_interval"] = out["score_wit"].to_numpy(float) <= 1.0
    return out
