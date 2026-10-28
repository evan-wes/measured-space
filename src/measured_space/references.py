"""Recalibrate a held-out site from a few reference measurements (simulated), and re-test the label gate there.

The audit (:mod:`measured_space.audit`) predicts each numeric label's error ``e = auto - oracle`` (m) from
auto-side features, with a model fitted on other scenes; calibrates it by split conformal prediction on other
scenes; and certifies an acceptance threshold by Learn-then-Test, so that ``P(|e| > tau | accepted) <= epsilon``.
Out of scene that certificate fails: a site can carry an error shared by all of its labels (a monocular capture's
depth-scale error, a per-object bias) that no per-label feature sees. The remedy tested here: at each new site,
measure a few **references**, labels whose true answer is measured physically (in simulation: whose oracle
answer is revealed), and use only those to recalibrate the site before its other labels are gated.

**Recalibrating one site from its references.** The global model's predicted bias ``mu`` and spread ``sigma``
(m) are kept; the site's references add three corrections, each optional (:class:`Variant`):

a. **Scale** (:func:`fit_site_correction`). One multiplicative factor per depth source, the shared scale of a
   capture: ``s = exp(median log((oracle + mu) / auto))`` over the references answered by that source (the
   references of all sources pooled when a source has fewer than ``min_scale_items``; 1 when the site has
   fewer). The corrected label is ``s * auto``. The ratio is taken against ``oracle + mu``, the auto answer the
   global model expects, rather than against ``oracle``: the model's ``mu`` already carries the bias every
   site shares (boxes a few centimetres small), and a plain ``oracle / auto`` ratio would absorb that bias into
   the scale, so that the gate, which keeps ``mu``, would count it twice.
b. **Bias** (:func:`fit_site_correction`). One additive term per question type, the median of
   ``s * auto - oracle - mu`` (m) over the references of that type, if at least ``min_bias_items`` references
   have that type (else 0). The label becomes ``L = s * auto - b``.
c. **Spread** (:func:`site_quantile_factor`). A factor ``c`` that widens ``sigma`` at this site. Each reference's
   normalised residual ``|L - oracle - mu| / sigma``, computed leaving that reference out of (a) and (b)
   (:func:`reference_scores`, so the references do not certify their own fit), is divided by the global
   conformal quantile of its cell; ``Q`` is the conformal-rank ``level`` quantile of these ratios (the largest
   one when the references are too few for that rank). Rule ``max`` (the default) uses ``c = max(1, Q)``, the
   larger of the global and the site quantile. A uniform random reference sample is exchangeable with the
   site's other labels, so the site quantile is a within-site, jackknife-style conformal quantile
   (approximately valid for at least ``level / (1 - level)`` reference rows, 9 at 90 %; below that it is the
   largest ratio and carries no finite-sample claim), while the global one is valid when sites are
   exchangeable. Their maximum covers whenever either assumption holds, which is the point of measuring
   references at a site that may differ from the others; the cost is width. Rule ``mix`` shrinks instead,
   ``c = (1 - w) + w Q`` with ``w = n / (n + mix_prior)``, which can narrow the interval and has no such
   guarantee; rule ``none`` keeps ``c = 1``.

After recalibration the label is ``L``; its predicted error keeps ``mu`` and is ``c * sigma`` wide. The 90 %
interval of ``L - oracle`` is ``mu +- c q sigma`` (``q`` the global Mondrian conformal quantile of the cell,
question type by depth source, as :func:`~measured_space.audit.conformal_calibrate`), and the gate score is
``(|mu| + z c sigma) / (tau_scale * tau_hat(L))`` with ``tau_hat(L)`` the tolerance at the corrected answer.
With no references, ``s = 1``, ``b = 0`` and ``c = 1``: the audit's gate exactly.

**Calibration sites.** With ``calibration="referenced"`` (the default), every calibration site is recalibrated
the same way from its own references (the same count, at most ``max_ref_frac`` of its items) before its scores
enter the global conformal quantile and Learn-then-Test: the threshold is then certified for the referenced
pipeline that is applied at the test site. A calibration site's spread factor is computed against the global
quantile of the *other* calibration sites. With ``calibration="global"``, the calibration sites are used as the
audit uses them and only the test site is recalibrated.

**Experiment** (:func:`run_reference_experiment`). Leave one scene out within a dataset, as the audit does:
predictions come from :func:`~measured_space.audit.cross_fit` with the audit's feature model (test scene: fitted
without it; calibration scenes: fitted without the test scene and the calibration scene). Above
:attr:`~measured_space.audit.AuditConfig.max_folds` scenes in a dataset and question type, the audit's grouped
folds apply unchanged: a test scene is predicted by the model fitted without its fold, and a calibration scene by
the model fitted without both folds, so a test scene's own fold-mates do not calibrate it (they are empty
calibration sites, which contribute nothing). At most ``max_folds`` scenes, the predictions are leave one (two)
scene(s) out exactly as before. For each test scene,
reference count ``n`` and random draw, ``n`` of the scene's numeric items are references, drawn uniformly or
stratified by terciles of the item's predicted sigma (:func:`reference_order`; draws are nested, so the ``n = 10``
references contain the ``n = 5`` ones). An item is one question; a reference reveals its oracle answer for every
depth source that answered it. References are left out of the evaluation. A scene is skipped for an ``n`` larger
than ``max_ref_frac`` of its items. Per scene, draw and tolerance multiple: accepted yield, error among accepted
(``|L - oracle| > tau_scale * tau``, the audit's rule), and coverage of the 90 % interval.

Caveats: items of one scene share objects and a capture, so per-scene error rates of small scenes are noisy,
and the rows of one reference (one per depth source) are not independent; references are drawn among labels
that perception answered (a deployment would measure those it wants to check); the scale and bias are assumed
constant within a site; a calibration site smaller than ``n / max_ref_frac`` items gets fewer references than
the test site.
"""

from __future__ import annotations

import json
import math
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from measured_space.audit import (AuditConfig, _run_tasks, conformal_quantile, depth_sources, load_labels,
                                  ltt_threshold, synthetic_pairs)

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

SAMPLINGS: tuple[str, ...] = ("uniform", "stratified")
"""tuple of str: how references are drawn: uniformly among a site's items, or stratified by predicted-sigma tercile."""

QUANTILE_RULES: tuple[str, ...] = ("max", "mix", "none")
"""tuple of str: how a site's references adjust the spread (see the module docstring, step c)."""

CALIBRATIONS: tuple[str, ...] = ("referenced", "global")
"""tuple of str: whether calibration sites are recalibrated from their own references too, or used as is."""

_MIN_VALUE = 0.05            # m: answers (and expected answers) below this are left out of the scale's log ratio


@dataclass(frozen=True)
class Variant:
    """One recalibration recipe.

    Attributes
    ----------
    name : str
        Label used in the result tables.
    scale : bool
        Step (a): a multiplicative scale per depth source.
    bias : bool
        Step (b): an additive bias per question type.
    quantile : str
        Step (c): one of :data:`QUANTILE_RULES`.
    calibration : str
        One of :data:`CALIBRATIONS`.
    sampling : str
        One of :data:`SAMPLINGS`.
    """

    name: str
    scale: bool = True
    bias: bool = True
    quantile: str = "max"
    calibration: str = "referenced"
    sampling: str = "uniform"

    def __post_init__(self) -> None:
        """Check the option values.

        Raises
        ------
        ValueError
            If ``quantile``, ``calibration`` or ``sampling`` is not one of the allowed values.
        """
        for value, allowed in ((self.quantile, QUANTILE_RULES), (self.calibration, CALIBRATIONS),
                               (self.sampling, SAMPLINGS)):
            if value not in allowed:
                raise ValueError(f"Variant {self.name!r}: {value!r} is not one of {allowed}")


DEFAULT_VARIANTS: tuple[Variant, ...] = (
    Variant("abc_uniform"),
    Variant("abc_stratified", sampling="stratified"),
    Variant("a_uniform", bias=False, quantile="none"),
    Variant("ab_uniform", quantile="none"),
    Variant("abc_mix_uniform", quantile="mix"),
    Variant("abc_globalcal_uniform", calibration="global"),
)
"""tuple of Variant: the primary recipe (all three steps, referenced calibration) with uniform and stratified
references, and ablations with uniform references: scale only, scale and bias, the ``mix`` spread rule, and
calibration sites used without references."""


@dataclass(frozen=True)
class ReferenceConfig:
    """Settings of the reference experiment.

    Attributes
    ----------
    n_refs : tuple of int
        Reference counts per site (default ``(0, 5, 10, 20, 40)``); 0 is the audit's gate.
    n_draws : int
        Random reference draws per scene and count (default 50).
    variants : tuple of Variant
        Recipes compared (default :data:`DEFAULT_VARIANTS`).
    min_scale_items : int
        Fewest references of a depth source for its own scale; fewer pool the site's sources, and fewer at the
        whole site leave the scale at 1 (default 3).
    min_bias_items : int
        Fewest references of a question type for its bias term (default 5).
    mix_prior : float
        Rule ``mix``: weight of the global quantile, in references (default 20).
    max_ref_frac : float
        A scene is skipped for a reference count above this share of its items; a calibration scene takes at
        most this share (default 0.5).
    level : float
        Nominal level of the reported interval and of the spread adjustment (default 0.9).
    seed : int
        Seed of the reference draws (default 0).
    """

    n_refs: tuple[int, ...] = (0, 5, 10, 20, 40)
    n_draws: int = 50
    variants: tuple[Variant, ...] = DEFAULT_VARIANTS
    min_scale_items: int = 3
    min_bias_items: int = 5
    mix_prior: float = 20.0
    max_ref_frac: float = 0.5
    level: float = 0.9
    seed: int = 0


# ============================================================================================================
# 1. Inputs: tolerances, synthetic sites, out-of-scene predictions
# ============================================================================================================


def read_tolerances(path: str | Path) -> pd.DataFrame:
    """Numeric tolerance bounds of every question in an oracle label file.

    Parameters
    ----------
    path : str or pathlib.Path
        A ``qa_oracle.jsonl`` written by the label pipeline.

    Returns
    -------
    pandas.DataFrame
        Columns ``id``, ``tol_abs`` (m) and ``tol_rel`` (fraction), one row per numeric question; a bound the
        question does not have is NaN.
    """
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get("answer_type") == "number":
                    tol = record.get("tolerance") or {}
                    rows.append({"id": record["id"],
                                 "tol_abs": math.nan if tol.get("abs") is None else float(tol["abs"]),
                                 "tol_rel": math.nan if tol.get("rel") is None else float(tol["rel"])})
    return pd.DataFrame(rows, columns=["id", "tol_abs", "tol_rel"])


def load_reference_pairs(label_dir: str | Path) -> pd.DataFrame:
    """One dataset's pair table (:func:`~measured_space.audit.load_labels`) with each question's tolerance bounds.

    Parameters
    ----------
    label_dir : str or pathlib.Path
        Folder with ``qa_oracle.jsonl`` and ``qa_auto_<depth source>.jsonl``.

    Returns
    -------
    pandas.DataFrame
        The pair table plus ``tol_abs`` (m) and ``tol_rel`` (NaN for non-numeric questions); ``attrs`` as
        :func:`~measured_space.audit.load_labels`.
    """
    label_dir = Path(label_dir)
    pairs = load_labels(label_dir)
    tol = read_tolerances(label_dir / "qa_oracle.jsonl").set_index("id")
    attrs = dict(pairs.attrs)
    pairs = pairs.assign(tol_abs=pairs["id"].map(tol["tol_abs"]), tol_rel=pairs["id"].map(tol["tol_rel"]))
    pairs.attrs.update(attrs)
    return pairs


def tolerance_at(tol_abs: ArrayLike, tol_rel: ArrayLike, reference: ArrayLike) -> Array:
    """The largest error an answer may have and still be correct, as :func:`~measured_space.qa_schema.is_correct`.

    Parameters
    ----------
    tol_abs : array_like, shape (n,)
        Absolute bounds (m); NaN where absent.
    tol_rel : array_like, shape (n,)
        Relative bounds (fraction); NaN where absent.
    reference : array_like, shape (n,)
        The value the relative bound scales with (m).

    Returns
    -------
    numpy.ndarray, shape (n,)
        ``max(abs, rel * |reference|)`` over the bounds present (m); 0 where there are none.
    """
    a = np.asarray(tol_abs, dtype=np.float64)
    r = np.asarray(tol_rel, dtype=np.float64) * np.abs(np.asarray(reference, dtype=np.float64))
    out = np.fmax(np.where(np.isfinite(a), a, np.nan), np.where(np.isfinite(r), r, np.nan))
    return np.where(np.isfinite(out), out, 0.0)


def synthetic_site_pairs(n_scenes: int = 6, n_items: int = 80, *, scale_errors: Mapping[str, float] | None = None,
                         seed: int = 0, **kwargs: Any) -> pd.DataFrame:
    """Synthetic pair table (:func:`~measured_space.audit.synthetic_pairs`) with tolerances and site scale errors.

    Parameters
    ----------
    n_scenes : int, default 6
        Scenes (``scene00``, ``scene01``, ...).
    n_items : int, default 80
        Questions per scene and type.
    scale_errors : mapping of str to float, optional
        ``{scene: factor}``: every auto answer of that scene's numeric questions is multiplied by ``factor``
        (e.g. 1.12 for answers 12 % too large, a capture's depth-scale error that no feature sees).
    seed : int, default 0
        Random seed.
    **kwargs : Any
        Further arguments of :func:`~measured_space.audit.synthetic_pairs`.

    Returns
    -------
    pandas.DataFrame
        The pair table with ``tol_abs = 0.1`` m and ``tol_rel = 0.1`` on numeric rows (the synthetic tolerance
        ``max(0.1, 0.1 * oracle)``), and the scaled scenes' ``value``, ``auto``, ``e``, ``abs_e``, ``rel_e``,
        ``tau_hat``, ``wrong`` and ``log_value`` recomputed.

    Raises
    ------
    KeyError
        If ``scale_errors`` names a scene that does not exist.
    """
    pairs = synthetic_pairs(n_scenes, n_items, seed=seed, **kwargs)
    numeric = (pairs["answer_type"] == "number").to_numpy()
    pairs["tol_abs"] = np.where(numeric, 0.1, np.nan)
    pairs["tol_rel"] = np.where(numeric, 0.1, np.nan)
    for scene, factor in (scale_errors or {}).items():
        if scene not in set(pairs["scene"]):
            raise KeyError(f"No scene {scene!r}")
        m = numeric & (pairs["scene"] == scene).to_numpy() & pairs["matched"].astype(bool).to_numpy()
        value = pairs.loc[m, "value"].to_numpy(float) * factor
        oracle = pairs.loc[m, "oracle"].to_numpy(float)
        e = value - oracle
        tau = pairs.loc[m, "tau"].to_numpy(float)
        pairs.loc[m, "value"] = value
        pairs.loc[m, "auto"] = pd.Series(value, index=pairs.index[m]).astype(object)
        pairs.loc[m, "e"], pairs.loc[m, "abs_e"], pairs.loc[m, "rel_e"] = e, np.abs(e), e / oracle
        pairs.loc[m, "tau_hat"] = tolerance_at(np.full(len(value), 0.1), np.full(len(value), 0.1), value)
        pairs.loc[m, "wrong"] = (np.abs(e) > tau).astype(float)
        pairs.loc[m, "log_value"] = np.log(np.maximum(np.abs(value), _MIN_VALUE))
    return pairs


def _sigma_tasks(pairs: pd.DataFrame,
                 config: AuditConfig) -> tuple[list[tuple[pd.DataFrame, str, bool, list[str], AuditConfig]], list[str]]:
    """The audit's cross-fitting tasks for its feature model on every modelled numeric group.

    The same groups, filters and depth-source order as :func:`~measured_space.audit.run_audit`.

    Parameters
    ----------
    pairs : pandas.DataFrame
        Pair table(s).
    config : AuditConfig
        Minimum scenes and rows per group, and the model settings carried by each task.

    Returns
    -------
    tasks : list of tuple
        ``(rows, "sigma", True, sources, config)`` per dataset and numeric question type, in sorted order.
    notes : list of str
        Groups skipped (too few scenes or rows).
    """
    tasks: list[tuple[pd.DataFrame, str, bool, list[str], AuditConfig]] = []
    notes = []
    for dataset, dpairs in pairs.groupby("dataset", sort=True):
        sources = depth_sources(dpairs)
        labelled = dpairs[dpairs["matched"].astype(bool) & (dpairs["answer_type"] == "number")]
        for qtype, rows in labelled.groupby("qtype", sort=True):
            rows = rows[np.isfinite(rows["e"].to_numpy(float))]
            n_scenes = rows["scene"].nunique()
            if n_scenes < config.min_scenes or len(rows) < config.min_rows:
                notes.append(f"{dataset}/{qtype}: skipped ({len(rows)} labelled rows, {n_scenes} scenes)")
                continue
            tasks.append((rows, "sigma", True, sources, config))
    return tasks, notes


def cross_fit_key(pairs: pd.DataFrame, config: AuditConfig) -> dict[str, Any]:
    """The audit settings that :func:`cross_fit_sigma`'s predictions depend on, e.g. for a cache key.

    Every :class:`~measured_space.audit.AuditConfig` field except ``n_jobs`` (how many workers fit changes
    nothing) and ``max_folds``, which is included only when some modelled group has more scenes than it, the
    only case in which it changes the predictions. Keys written before ``max_folds`` existed therefore still
    match runs with at most ``max_folds`` scenes per group.

    Parameters
    ----------
    pairs : pandas.DataFrame
        The pair table(s) to be cross-fitted.
    config : AuditConfig
        The settings.

    Returns
    -------
    dict
        Field name to value (``asdict`` form).
    """
    key = {k: v for k, v in asdict(config).items() if k not in ("n_jobs", "max_folds")}
    tasks, _ = _sigma_tasks(pairs, config)
    if any(task[0]["scene"].nunique() > config.max_folds for task in tasks):
        key["max_folds"] = config.max_folds
    return key


def cross_fit_sigma(pairs: pd.DataFrame, config: AuditConfig,
                    log: Callable[[str], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Out-of-scene predictions of the audit's feature model for every modelled numeric row.

    The same groups, filters, depth-source order and model as :func:`~measured_space.audit.run_audit` (kind
    ``sigma`` only), through the audit's own cross-fitting, so the predictions are the audit's, folds included
    (grouped above ``config.max_folds`` scenes).

    Parameters
    ----------
    pairs : pandas.DataFrame
        Pair table(s) with a default ``RangeIndex`` (rows are referred to by index).
    config : AuditConfig
        Model settings, minimum scenes and rows per group, folds, and ``n_jobs``.
    log : callable, optional
        Receives progress messages.

    Returns
    -------
    test : pandas.DataFrame
        Columns ``row, mu, sigma``: each row predicted by a model fitted without its scene (its fold, when
        scenes are grouped) (m).
    cal : pandas.DataFrame
        Columns ``test_scene, row, mu, sigma``: each row predicted by a model fitted without its scene and
        ``test_scene`` (without both their folds), once per other scene of its dataset (once per scene outside
        its fold) (m).
    notes : list of str
        Groups skipped (too few scenes or rows).

    Raises
    ------
    ValueError
        If ``pairs`` does not have a default ``RangeIndex``.
    """
    if not pairs.index.equals(pd.RangeIndex(len(pairs))):
        raise ValueError("pairs must have a default RangeIndex (call reset_index(drop=True))")
    say = log or (lambda message: None)
    tasks, notes = _sigma_tasks(pairs, config)
    if not tasks:
        empty = pd.DataFrame(columns=["row", "mu", "sigma"])
        return empty, empty.assign(test_scene=pd.Series(dtype=object)), notes
    results = _run_tasks(tasks, config.n_jobs, say)
    test = pd.concat([t for t, _ in results], ignore_index=True)[["row", "mu", "sigma"]]
    cal = pd.concat([c for _, c in results], ignore_index=True)[["test_scene", "row", "mu", "sigma"]]
    return test, cal, notes


# ============================================================================================================
# 2. Recalibrating one site from its references
# ============================================================================================================


@dataclass(frozen=True)
class SiteRows:
    """The numeric labels of one site with one model's predictions, as aligned arrays (one entry per row).

    Attributes
    ----------
    scene : str
        The site.
    value : numpy.ndarray, shape (n,)
        Auto answer (m).
    oracle : numpy.ndarray, shape (n,)
        Oracle answer (m); used only for references and for scoring.
    e : numpy.ndarray, shape (n,)
        ``value - oracle`` as the pair table holds it (m).
    mu : numpy.ndarray, shape (n,)
        Predicted bias of ``e = value - oracle`` (m).
    sigma : numpy.ndarray, shape (n,)
        Predicted spread of ``e`` around ``mu`` (m).
    tau : numpy.ndarray, shape (n,)
        Tolerance at the oracle answer (m).
    tol_abs : numpy.ndarray, shape (n,)
        Absolute tolerance bound (m; NaN if absent).
    tol_rel : numpy.ndarray, shape (n,)
        Relative tolerance bound (NaN if absent).
    source : numpy.ndarray of int, shape (n,)
        Depth-source code.
    qtype : numpy.ndarray of int, shape (n,)
        Question-type code.
    cell : numpy.ndarray of int, shape (n,)
        Conformal cell code, ``qtype * n_sources + source``.
    item : numpy.ndarray of int, shape (n,)
        Item (question) index within the site, ``0 .. n_items - 1``; the rows of one question share it.
    n_items : int
        Items at the site.
    """

    scene: str
    value: Array
    oracle: Array
    e: Array
    mu: Array
    sigma: Array
    tau: Array
    tol_abs: Array
    tol_rel: Array
    source: IntArray
    qtype: IntArray
    cell: IntArray
    item: IntArray
    n_items: int

    def item_sigma(self) -> Array:
        """Mean predicted sigma of each item over its rows.

        Returns
        -------
        numpy.ndarray, shape (n_items,)
            Metres.
        """
        total = np.bincount(self.item, weights=self.sigma, minlength=self.n_items)
        return total / np.maximum(np.bincount(self.item, minlength=self.n_items), 1)


@dataclass(frozen=True)
class SiteCorrection:
    """A site's label correction, ``L = scale[source] * auto - bias[qtype]``.

    Attributes
    ----------
    scale : numpy.ndarray, shape (n_sources,)
        Multiplicative factor per depth source (1: none).
    bias : numpy.ndarray, shape (n_qtypes,)
        Additive bias per question type (m; 0: none).
    """

    scale: Array
    bias: Array

    @staticmethod
    def identity(n_sources: int, n_qtypes: int) -> SiteCorrection:
        """The correction that leaves labels unchanged.

        Parameters
        ----------
        n_sources : int
            Depth sources of the dataset.
        n_qtypes : int
            Question types of the dataset.

        Returns
        -------
        SiteCorrection
            Unit scales and zero biases.
        """
        return SiteCorrection(np.ones(n_sources), np.zeros(n_qtypes))

    def apply(self, value: ArrayLike, source: ArrayLike, qtype: ArrayLike) -> Array:
        """Corrected labels.

        Parameters
        ----------
        value : array_like, shape (n,)
            Auto answers (m).
        source : array_like of int, shape (n,)
            Depth-source codes.
        qtype : array_like of int, shape (n,)
            Question-type codes.

        Returns
        -------
        numpy.ndarray, shape (n,)
            ``scale[source] * value - bias[qtype]`` (m).
        """
        return self.scale[np.asarray(source)] * np.asarray(value, dtype=np.float64) - self.bias[np.asarray(qtype)]


def _n_unique(values: IntArray) -> int:
    """Number of distinct values.

    Parameters
    ----------
    values : numpy.ndarray of int, shape (n,)
        Codes.

    Returns
    -------
    int
        Distinct count (0 for an empty array).
    """
    return int(len(np.unique(values))) if len(values) else 0


def fit_site_correction(value: ArrayLike, oracle: ArrayLike, mu: ArrayLike, source: ArrayLike, qtype: ArrayLike,
                        item: ArrayLike, n_sources: int, n_qtypes: int, *, scale: bool = True, bias: bool = True,
                        min_scale_items: int = 3, min_bias_items: int = 5) -> SiteCorrection:
    """Fit a site's scale per depth source and bias per question type on its references.

    Parameters
    ----------
    value : array_like, shape (n,)
        Auto answers of the reference rows (m).
    oracle : array_like, shape (n,)
        Their measured (oracle) answers (m).
    mu : array_like, shape (n,)
        The global model's predicted bias of each row (m).
    source : array_like of int, shape (n,)
        Depth-source codes in ``0 .. n_sources - 1``.
    qtype : array_like of int, shape (n,)
        Question-type codes in ``0 .. n_qtypes - 1``.
    item : array_like of int, shape (n,)
        Item codes (one question answered by several sources is one reference).
    n_sources : int
        Depth sources of the dataset.
    n_qtypes : int
        Question types of the dataset.
    scale : bool, default True
        Fit step (a); otherwise every scale is 1.
    bias : bool, default True
        Fit step (b); otherwise every bias is 0.
    min_scale_items : int, default 3
        Fewest references for a source's own scale (else the site's pooled scale; else 1).
    min_bias_items : int, default 5
        Fewest references of a question type for its bias (else 0).

    Returns
    -------
    SiteCorrection
        ``scale[k] = exp(median log((oracle + mu) / value))`` over source ``k``'s references whose value and
        ``oracle + mu`` exceed 0.05 m; ``bias[t] = median(scale * value - oracle - mu)`` over type ``t``'s
        references (m).
    """
    value, oracle, mu = (np.asarray(v, dtype=np.float64) for v in (value, oracle, mu))
    source, qtype, item = (np.asarray(v, dtype=np.int64) for v in (source, qtype, item))
    s = np.ones(n_sources)
    if scale and len(value):
        expected = oracle + mu
        ok = (value > _MIN_VALUE) & (expected > _MIN_VALUE)
        log_ratio = np.log(np.where(ok, expected, 1.0) / np.where(ok, value, 1.0))
        pooled = float(np.exp(np.median(log_ratio[ok]))) if _n_unique(item[ok]) >= min_scale_items else 1.0
        for k in range(n_sources):
            m = ok & (source == k)
            s[k] = float(np.exp(np.median(log_ratio[m]))) if _n_unique(item[m]) >= min_scale_items else pooled
    b = np.zeros(n_qtypes)
    if bias and len(value):
        resid = s[source] * value - oracle - mu
        for t in range(n_qtypes):
            m = qtype == t
            if _n_unique(item[m]) >= min_bias_items:
                b[t] = float(np.median(resid[m]))
    return SiteCorrection(s, b)


def reference_scores(site: SiteRows, ref_rows: BoolArray, n_sources: int, n_qtypes: int, *, scale: bool = True,
                     bias: bool = True, min_scale_items: int = 3, min_bias_items: int = 5) -> Array:
    """Leave-one-reference-out normalised residuals of a site's references.

    Each reference item is left out in turn; the correction is refitted on the others
    (:func:`fit_site_correction`) and applied to the left-out item's rows.

    Parameters
    ----------
    site : SiteRows
        The site.
    ref_rows : numpy.ndarray of bool, shape (n,)
        Which rows belong to reference items.
    n_sources : int
        Depth sources of the dataset.
    n_qtypes : int
        Question types of the dataset.
    scale : bool, default True
        Fit step (a).
    bias : bool, default True
        Fit step (b).
    min_scale_items : int, default 3
        See :func:`fit_site_correction`.
    min_bias_items : int, default 5
        See :func:`fit_site_correction`.

    Returns
    -------
    numpy.ndarray, shape (ref_rows.sum(),)
        ``|L - oracle - mu| / sigma`` per reference row, in ``site`` order (dimensionless).
    """
    idx = np.flatnonzero(ref_rows)
    value, oracle, e = site.value[idx], site.oracle[idx], site.e[idx]
    mu, sigma = site.mu[idx], site.sigma[idx]
    source, qtype, item = site.source[idx], site.qtype[idx], site.item[idx]
    out = np.empty(len(idx))
    for j in np.unique(item):
        held = item == j
        if scale or bias:
            corr = fit_site_correction(value[~held], oracle[~held], mu[~held], source[~held], qtype[~held],
                                       item[~held], n_sources, n_qtypes, scale=scale, bias=bias,
                                       min_scale_items=min_scale_items, min_bias_items=min_bias_items)
            label = corr.apply(value[held], source[held], qtype[held])
        else:
            label = value[held]
        out[held] = np.abs(e[held] + (label - value[held]) - mu[held]) / sigma[held]
    return out


def site_quantile_factor(ratios: ArrayLike, n_refs: int, rule: str, level: float = 0.9,
                         mix_prior: float = 20.0) -> float:
    """The spread factor ``c`` of a site, from its references' residual ratios (step c).

    Parameters
    ----------
    ratios : array_like, shape (m,)
        Per reference row: its leave-one-out normalised residual divided by the global conformal quantile of its
        cell (1 means exactly on the global interval's edge). Non-finite values are ignored.
    n_refs : int
        Reference items (the ``mix`` weight counts items, not rows).
    rule : str
        One of :data:`QUANTILE_RULES`.
    level : float, default 0.9
        Nominal level.
    mix_prior : float, default 20.0
        Rule ``mix``: weight of the global quantile, in references.

    Returns
    -------
    float
        ``Q``, the ``ceil((m + 1) level)``-th smallest ratio (the largest if ``m`` is too small); ``max(1, Q)``
        for ``max``, ``(1 - w) + w Q`` with ``w = n_refs / (n_refs + mix_prior)`` for ``mix``, 1 for ``none``
        or when there are no ratios.

    Raises
    ------
    ValueError
        For an unknown rule.
    """
    if rule not in QUANTILE_RULES:
        raise ValueError(f"Unknown quantile rule {rule!r}")
    r = np.sort(np.asarray(ratios, dtype=np.float64))
    r = r[np.isfinite(r)]
    if rule == "none" or not len(r):
        return 1.0
    q = float(r[min(math.ceil((len(r) + 1) * level), len(r)) - 1])
    if rule == "max":
        return max(1.0, q)
    w = n_refs / (n_refs + mix_prior)
    return (1.0 - w) + w * q


def cell_quantiles(scores: ArrayLike, cell: ArrayLike, cell_qtype: ArrayLike, level: float,
                   min_cal: int) -> Array:
    """Mondrian split-conformal quantiles per cell, as :func:`~measured_space.audit.conformal_calibrate` does.

    Parameters
    ----------
    scores : array_like, shape (n,)
        Calibration scores ``|e - mu| / sigma``.
    cell : array_like of int, shape (n,)
        Each score's cell code (question type by depth source).
    cell_qtype : array_like of int, shape (n_cells,)
        The question-type code of every cell.
    level : float
        Nominal coverage.
    min_cal : int
        A cell with fewer scores uses its question type's scores over all depth sources.

    Returns
    -------
    numpy.ndarray, shape (n_cells,)
        :func:`~measured_space.audit.conformal_quantile` of each cell (``inf`` when too few scores).
    """
    scores, cell = np.asarray(scores, dtype=np.float64), np.asarray(cell, dtype=np.int64)
    cell_qtype = np.asarray(cell_qtype, dtype=np.int64)
    counts = np.bincount(cell, minlength=len(cell_qtype))
    row_qtype = cell_qtype[cell]
    pooled = {t: conformal_quantile(scores[row_qtype == t], level) for t in np.unique(cell_qtype)}
    return np.array([conformal_quantile(scores[cell == k], level) if counts[k] >= min_cal
                     else pooled[cell_qtype[k]] for k in range(len(cell_qtype))])


def reference_order(item_sigma: ArrayLike, rng: np.random.Generator, sampling: str) -> IntArray:
    """An order of a site's items whose every prefix is a reference sample of the given scheme.

    Parameters
    ----------
    item_sigma : array_like, shape (m,)
        Each item's predicted sigma (m), used for stratification.
    rng : numpy.random.Generator
        Random source.
    sampling : str
        ``"uniform"``: a random permutation. ``"stratified"``: items are split into terciles of
        ``item_sigma`` (by rank), each shuffled, and taken in rounds of one per tercile (in a random order per
        round), so every prefix holds ``floor(n / 3)`` or ``ceil(n / 3)`` items of each tercile while it lasts.

    Returns
    -------
    numpy.ndarray of int, shape (m,)
        Item indices; the first ``n`` are the ``n`` references.

    Raises
    ------
    ValueError
        For an unknown sampling scheme.
    """
    sigma = np.asarray(item_sigma, dtype=np.float64)
    m = len(sigma)
    if sampling == "uniform":
        return rng.permutation(m).astype(np.int64)
    if sampling != "stratified":
        raise ValueError(f"Unknown sampling {sampling!r}")
    rank = np.argsort(np.argsort(sigma, kind="stable"), kind="stable")
    tercile = (3 * rank) // max(m, 1)
    pools = [list(rng.permutation(np.flatnonzero(tercile == t))) for t in range(3)]
    order: list[int] = []
    while len(order) < m:
        for t in rng.permutation(3):
            if pools[t]:
                order.append(int(pools[t].pop()))
    return np.array(order, dtype=np.int64)


# ============================================================================================================
# 3. One site: recalibrate, calibrate, gate
# ============================================================================================================


@dataclass(frozen=True)
class SiteState:
    """A site after recalibration from its references.

    Attributes
    ----------
    site : SiteRows
        The site.
    ref_rows : numpy.ndarray of bool, shape (n,)
        Rows of reference items (left out of calibration and evaluation).
    n_refs : int
        Reference items.
    correction : SiteCorrection
        The fitted correction.
    label : numpy.ndarray, shape (n,)
        Corrected labels ``L`` (m).
    error : numpy.ndarray, shape (n,)
        ``L - oracle``, computed as ``e + (L - value)`` so that it is the pair table's ``e`` when ``L = value``
        (m).
    score : numpy.ndarray, shape (n,)
        Conformal score ``|error - mu| / sigma``.
    ref_scores : numpy.ndarray, shape (ref_rows.sum(),)
        Leave-one-reference-out scores of the reference rows (empty without step c).
    """

    site: SiteRows
    ref_rows: BoolArray
    n_refs: int
    correction: SiteCorrection
    label: Array
    error: Array
    score: Array
    ref_scores: Array

    def factor(self, q_cell: Array, variant: Variant, config: ReferenceConfig) -> float:
        """The site's spread factor against given global cell quantiles.

        Parameters
        ----------
        q_cell : numpy.ndarray, shape (n_cells,)
            Global conformal quantile per cell.
        variant : Variant
            Supplies the quantile rule.
        config : ReferenceConfig
            Level and ``mix_prior``.

        Returns
        -------
        float
            :func:`site_quantile_factor` of the reference ratios.
        """
        if variant.quantile == "none" or not self.n_refs:
            return 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = self.ref_scores / q_cell[self.site.cell[self.ref_rows]]
        return site_quantile_factor(ratios, self.n_refs, variant.quantile, config.level, config.mix_prior)

    def gate_inputs(self, c: float, z: float, tau_scale: float) -> tuple[Array, Array]:
        """Gate scores and wrong flags of the non-reference rows.

        Parameters
        ----------
        c : float
            Spread factor.
        z : float
            Gaussian multiplier of the gate's bound.
        tau_scale : float
            Tolerance multiple.

        Returns
        -------
        score : numpy.ndarray, shape (k,)
            ``(|mu| + z c sigma) / (tau_scale * tau_hat(L))``, in tolerances.
        wrong : numpy.ndarray, shape (k,)
            1 where ``|L - oracle| > tau_scale * tau``.
        """
        keep, s = ~self.ref_rows, self.site
        tau_hat = tolerance_at(s.tol_abs[keep], s.tol_rel[keep], self.label[keep])
        score = (np.abs(s.mu[keep]) + z * c * s.sigma[keep]) / (tau_scale * tau_hat)
        wrong = (np.abs(self.error[keep]) > tau_scale * s.tau[keep]).astype(np.float64)
        return score, wrong


def recalibrate_site(site: SiteRows, ref_items: IntArray, variant: Variant, config: ReferenceConfig,
                     n_sources: int, n_qtypes: int) -> SiteState:
    """Recalibrate a site from the given reference items (steps a and b, and the inputs of step c).

    Parameters
    ----------
    site : SiteRows
        The site, with the predictions of the model that applies to it.
    ref_items : numpy.ndarray of int, shape (n_refs,)
        Reference item indices (may be empty).
    variant : Variant
        Which steps to apply.
    config : ReferenceConfig
        Minimum reference counts.
    n_sources : int
        Depth sources of the dataset.
    n_qtypes : int
        Question types of the dataset.

    Returns
    -------
    SiteState
        The corrected labels, their errors and conformal scores, and the references' leave-one-out scores.
    """
    is_ref = np.zeros(site.n_items, dtype=bool)
    is_ref[ref_items] = True
    ref_rows = is_ref[site.item]
    fit_kw = {"scale": variant.scale, "bias": variant.bias, "min_scale_items": config.min_scale_items,
              "min_bias_items": config.min_bias_items}
    if ref_rows.any() and (variant.scale or variant.bias):
        r = ref_rows
        correction = fit_site_correction(site.value[r], site.oracle[r], site.mu[r], site.source[r], site.qtype[r],
                                         site.item[r], n_sources, n_qtypes, **fit_kw)
    else:
        correction = SiteCorrection.identity(n_sources, n_qtypes)
    label = correction.apply(site.value, site.source, site.qtype)
    error = site.e + (label - site.value)
    score = np.abs(error - site.mu) / site.sigma
    ref_scores = reference_scores(site, ref_rows, n_sources, n_qtypes, **fit_kw) \
        if ref_rows.any() and variant.quantile != "none" else np.zeros(0)
    return SiteState(site, ref_rows, len(ref_items), correction, label, error, score, ref_scores)


@dataclass(frozen=True)
class Calibration:
    """What the calibration sites of one test site give it.

    Attributes
    ----------
    q_cell : numpy.ndarray, shape (n_cells,)
        Global Mondrian conformal quantile per cell, at the experiment's level.
    lam : dict of float to float
        Certified Learn-then-Test threshold per tolerance multiple (``-inf``: accept nothing).
    n_cal : int
        Calibration rows (after leaving out the calibration sites' references).
    """

    q_cell: Array
    lam: dict[float, float]
    n_cal: int


def calibrate(states: Sequence[SiteState], variant: Variant, config: ReferenceConfig, audit: AuditConfig,
              cell_qtype: Array) -> Calibration:
    """Global conformal quantiles and certified gate thresholds from (recalibrated) calibration sites.

    Parameters
    ----------
    states : sequence of SiteState
        One per calibration site, recalibrated from its own references (or with none).
    variant : Variant
        Supplies step (c)'s rule for the calibration sites' own spread factors.
    config : ReferenceConfig
        Level.
    audit : AuditConfig
        ``min_cal``, ``tau_scales``, ``gate_z`` and the Learn-then-Test settings.
    cell_qtype : numpy.ndarray of int, shape (n_cells,)
        Question-type code of every cell.

    Returns
    -------
    Calibration
        Cell quantiles over all calibration sites' non-reference rows, and per tolerance multiple the threshold
        certified on their gate scores, each site's sigma widened by its factor against the quantiles of the
        *other* calibration sites.
    """
    keep = [~st.ref_rows for st in states]
    scores = np.concatenate([st.score[k] for st, k in zip(states, keep)]) if states else np.zeros(0)
    cells = np.concatenate([st.site.cell[k] for st, k in zip(states, keep)]) if states else np.zeros(0, int)
    owner = np.concatenate([np.full(int(k.sum()), j) for j, k in enumerate(keep)]) if states else np.zeros(0, int)
    q_cell = cell_quantiles(scores, cells, cell_qtype, config.level, audit.min_cal)
    factors = []
    for j, st in enumerate(states):
        if variant.quantile == "none" or not st.n_refs:
            factors.append(1.0)
            continue
        others = owner != j
        q_other = cell_quantiles(scores[others], cells[others], cell_qtype, config.level, audit.min_cal)
        factors.append(st.factor(q_other, variant, config))
    lam = {}
    for tau_scale in audit.tau_scales:
        parts = [st.gate_inputs(c, audit.gate_z, tau_scale) for st, c in zip(states, factors)]
        s = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0)
        w = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0)
        lam[tau_scale] = ltt_threshold(s, w, audit.epsilon, audit.delta, min_accept=audit.min_accept,
                                       min_accept_frac=audit.min_accept_frac, max_threshold=audit.kappa)[0]
    return Calibration(q_cell, lam, len(scores))


def evaluate_site(state: SiteState, cal: Calibration, variant: Variant, config: ReferenceConfig,
                  audit: AuditConfig) -> list[dict[str, Any]]:
    """Gate a recalibrated test site's non-reference labels and score the decisions against the oracle.

    Parameters
    ----------
    state : SiteState
        The test site after recalibration.
    cal : Calibration
        Its calibration.
    variant : Variant
        Supplies step (c)'s rule.
    config : ReferenceConfig
        Level.
    audit : AuditConfig
        ``gate_z`` and ``tau_scales``.

    Returns
    -------
    list of dict
        One record per tolerance multiple: ``tau_scale, n_eval, n_wrong, n_acc, n_acc_wrong, n_cov, lam, c,
        n_cal`` (``n_wrong``: wrong labels among the evaluated ones before gating; ``n_cov``: evaluated labels
        whose error lies in the 90 % interval ``mu +- c q sigma``).
    """
    c = state.factor(cal.q_cell, variant, config)
    keep, s = ~state.ref_rows, state.site
    with np.errstate(invalid="ignore"):
        covered = np.abs(state.error[keep] - s.mu[keep]) <= c * cal.q_cell[s.cell[keep]] * s.sigma[keep]
    out = []
    for tau_scale in audit.tau_scales:
        score, wrong = state.gate_inputs(c, audit.gate_z, tau_scale)
        accept = score <= cal.lam[tau_scale]
        out.append({"tau_scale": tau_scale, "n_eval": int(keep.sum()), "n_wrong": int(wrong.sum()),
                    "n_acc": int(accept.sum()), "n_acc_wrong": int(wrong[accept].sum()), "n_cov": int(covered.sum()),
                    "lam": float(cal.lam[tau_scale]), "c": float(c), "n_cal": cal.n_cal})
    return out


# ============================================================================================================
# 4. The experiment
# ============================================================================================================


@dataclass
class DatasetSites:
    """One dataset's sites under every model of the leave-one-scene-out design.

    Attributes
    ----------
    dataset : str
        Dataset name.
    sources : list of str
        Depth sources (codes are positions).
    qtypes : list of str
        Modelled numeric question types (codes are positions).
    scenes : list of str
        Scenes, sorted.
    test : dict of str to SiteRows
        Each scene with the predictions of the model fitted without it.
    cal : dict of (str, str) to SiteRows
        ``(test scene, scene)``: the scene with the predictions of the model fitted without both (without both
        their folds when scenes are grouped; a question type in which the scene shares the test scene's fold
        has no rows here).
    cell_qtype : numpy.ndarray of int, shape (n_qtypes * n_sources,)
        Question-type code of every cell.
    """

    dataset: str
    sources: list[str]
    qtypes: list[str]
    scenes: list[str]
    test: dict[str, SiteRows]
    cal: dict[tuple[str, str], SiteRows]
    cell_qtype: IntArray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def _site_rows(rows: pd.DataFrame, pred: pd.DataFrame, scene: str, sources: list[str],
               qtypes: list[str]) -> SiteRows:
    """Assemble one site's arrays from its pair-table rows and one model's predictions.

    Parameters
    ----------
    rows : pandas.DataFrame
        The site's numeric rows (index: pair-table row), sorted by question id and depth source.
    pred : pandas.DataFrame
        Predictions indexed by pair-table row, with ``mu`` and ``sigma``.
    scene : str
        The site.
    sources : list of str
        Depth sources of the dataset.
    qtypes : list of str
        Question types of the dataset.

    Returns
    -------
    SiteRows
        The aligned arrays. Rows without a prediction are left out: a question type modelled on scenes that do
        not include a test scene has no calibration predictions for it, as in the audit. Item codes and
        ``n_items`` still count every item of the site, so reference draws agree across models.
    """
    item_all, uniques = pd.factorize(rows["id"], sort=True)
    p = pred.reindex(rows.index)
    has = p["mu"].notna().to_numpy()
    rows, p, item = rows[has], p[has], item_all[has]
    src = rows["depth_source"].map({s: i for i, s in enumerate(sources)}).to_numpy(np.int64)
    qt = rows["qtype"].map({q: i for i, q in enumerate(qtypes)}).to_numpy(np.int64)
    value, e = rows["value"].to_numpy(np.float64), rows["e"].to_numpy(np.float64)
    return SiteRows(scene, value, value - e, e, p["mu"].to_numpy(np.float64),
                    p["sigma"].to_numpy(np.float64), rows["tau"].to_numpy(np.float64),
                    rows["tol_abs"].to_numpy(np.float64), rows["tol_rel"].to_numpy(np.float64), src, qt,
                    qt * len(sources) + src, item.astype(np.int64), len(uniques))


def prepare_sites(pairs: pd.DataFrame, test: pd.DataFrame, cal: pd.DataFrame) -> list[DatasetSites]:
    """Arrange the cross-fitted predictions by dataset, test scene and site.

    Parameters
    ----------
    pairs : pandas.DataFrame
        The pair table the predictions index (default ``RangeIndex``), with ``tol_abs`` and ``tol_rel``.
    test : pandas.DataFrame
        :func:`cross_fit_sigma` test predictions.
    cal : pandas.DataFrame
        :func:`cross_fit_sigma` calibration predictions.

    Returns
    -------
    list of DatasetSites
        One per dataset with predictions, sorted by name.

    Raises
    ------
    ValueError
        If ``pairs`` lacks the tolerance columns.
    """
    if not {"tol_abs", "tol_rel"} <= set(pairs.columns):
        raise ValueError("pairs needs tol_abs and tol_rel (see load_reference_pairs)")
    rows_all = pairs.loc[test["row"].to_numpy()]
    test_pred = test.set_index("row")
    out = []
    for dataset, rows in rows_all.groupby("dataset", sort=True):
        sources = depth_sources(pairs[pairs["dataset"] == dataset])
        qtypes = sorted(rows["qtype"].unique())
        scenes = sorted(rows["scene"].unique())
        by_scene = {s: g.sort_values(["id", "depth_source"]) for s, g in rows.groupby("scene", sort=True)}
        dcal = cal[cal["row"].isin(rows.index)]
        cal_pred = {t: g.set_index("row") for t, g in dcal.groupby("test_scene", sort=False)}
        sites = DatasetSites(dataset, sources, qtypes, scenes,
                             {s: _site_rows(by_scene[s], test_pred, s, sources, qtypes) for s in scenes},
                             {(t, s): _site_rows(by_scene[s], cal_pred[t], s, sources, qtypes)
                              for t in scenes for s in scenes if s != t},
                             np.repeat(np.arange(len(qtypes)), len(sources)).astype(np.int64))
        out.append(sites)
    return out


def _seed(config: ReferenceConfig, dataset: str, scene: str, sampling: str, draw: int) -> list[int]:
    """Seed sequence of one site's reference order (the same whether the site is tested or calibrates).

    Parameters
    ----------
    config : ReferenceConfig
        Base seed.
    dataset : str
        Dataset name.
    scene : str
        Site.
    sampling : str
        Sampling scheme.
    draw : int
        Draw index.

    Returns
    -------
    list of int
        Entropy for ``numpy.random.default_rng``.
    """
    return [config.seed, zlib.crc32(f"{dataset}/{scene}".encode()), SAMPLINGS.index(sampling), draw]


def run_dataset(sites: DatasetSites, config: ReferenceConfig, audit: AuditConfig,
                log: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Run every variant, reference count and draw on one dataset.

    Parameters
    ----------
    sites : DatasetSites
        The dataset's sites (:func:`prepare_sites`).
    config : ReferenceConfig
        Experiment settings.
    audit : AuditConfig
        Gate and conformal settings.
    log : callable, optional
        Receives progress messages.

    Returns
    -------
    pandas.DataFrame
        One row per variant, reference count, draw, test scene and tolerance multiple: the
        :func:`evaluate_site` record plus ``dataset, variant, n_refs, draw, scene, n_items, n_ref_items`` and
        ``scale_<source>`` (the test site's fitted scale per depth source). ``n_refs == 0`` rows (the audit's
        gate, identical for every variant) appear once per variant with ``draw == 0``.
    """
    say = log or (lambda message: None)
    n_src, n_qt = len(sites.sources), len(sites.qtypes)
    samplings = sorted({v.sampling for v in config.variants}, key=SAMPLINGS.index)
    orders = {(s, sampling): np.stack([reference_order(sites.test[s].item_sigma(),
                                                       np.random.default_rng(_seed(config, sites.dataset, s,
                                                                                   sampling, d)), sampling)
                                       for d in range(config.n_draws)])
              for s in sites.scenes for sampling in samplings}
    base = Variant("audit", scale=False, bias=False, quantile="none", calibration="global")
    none = np.zeros(0, dtype=np.int64)
    global_cal = {}
    for t in sites.scenes:
        states = [recalibrate_site(sites.cal[(t, s)], none, base, config, n_src, n_qt) for s in sites.scenes if s != t]
        global_cal[t] = calibrate(states, base, config, audit, sites.cell_qtype)

    def record(variant: Variant, n: int, draw: int, scene: str, state: SiteState,
               cal: Calibration) -> list[dict[str, Any]]:
        """Evaluate one test site and label the records.

        Parameters
        ----------
        variant : Variant
            The recipe.
        n : int
            Reference count.
        draw : int
            Draw index.
        scene : str
            Test site.
        state : SiteState
            The recalibrated test site.
        cal : Calibration
            Its calibration.

        Returns
        -------
        list of dict
            :func:`evaluate_site` records with the run's identifiers.
        """
        extra = {"dataset": sites.dataset, "variant": variant.name, "n_refs": n, "draw": draw, "scene": scene,
                 "n_items": state.site.n_items, "n_ref_items": state.n_refs,
                 **{f"scale_{src}": float(state.correction.scale[k]) for k, src in enumerate(sites.sources)}}
        return [{**extra, **r} for r in evaluate_site(state, cal, variant, config, audit)]

    rows: list[dict[str, Any]] = []
    for t in sites.scenes:
        state = recalibrate_site(sites.test[t], none, base, config, n_src, n_qt)
        for variant in config.variants:
            rows += record(variant, 0, 0, t, state, global_cal[t])
    for variant in config.variants:
        for n in (n for n in config.n_refs if n > 0):
            tested = [t for t in sites.scenes if n <= config.max_ref_frac * sites.test[t].n_items]
            say(f"{sites.dataset} / {variant.name} / n={n}: {len(tested)} of {len(sites.scenes)} scenes")
            for draw in range(config.n_draws):
                for t in tested:
                    if variant.calibration == "referenced":
                        states = []
                        for s in sites.scenes:
                            if s == t:
                                continue
                            k = min(n, int(config.max_ref_frac * sites.cal[(t, s)].n_items))
                            refs = orders[(s, variant.sampling)][draw, :k]
                            states.append(recalibrate_site(sites.cal[(t, s)], refs, variant, config, n_src, n_qt))
                        cal = calibrate(states, variant, config, audit, sites.cell_qtype)
                    else:
                        cal = global_cal[t]
                    refs = orders[(t, variant.sampling)][draw, :n]
                    state = recalibrate_site(sites.test[t], refs, variant, config, n_src, n_qt)
                    rows += record(variant, n, draw, t, state, cal)
    return pd.DataFrame(rows)


@dataclass
class ReferenceResult:
    """Everything one reference experiment produces.

    Attributes
    ----------
    draws : pandas.DataFrame
        :func:`run_dataset` rows of every dataset (per scene and draw: counts, no per-item data).
    tables : dict of str to pandas.DataFrame
        ``summary`` (:func:`summary_table`) and ``scenes`` (:func:`scene_table`).
    notes : list of str
        Groups skipped and other remarks.
    """

    draws: pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def run_reference_experiment(pairs: pd.DataFrame, config: ReferenceConfig | None = None,
                             audit: AuditConfig | None = None, *,
                             predictions: tuple[pd.DataFrame, pd.DataFrame] | None = None,
                             log: Callable[[str], None] | None = None) -> ReferenceResult:
    """Run the reference experiment on a pair table (one or several datasets).

    Parameters
    ----------
    pairs : pandas.DataFrame
        :func:`load_reference_pairs` output (several datasets may be concatenated) or
        :func:`synthetic_site_pairs`; it is re-indexed ``0 .. n - 1``.
    config : ReferenceConfig, optional
        Experiment settings; defaults to ``ReferenceConfig()``.
    audit : AuditConfig, optional
        Model, conformal and gate settings; defaults to ``AuditConfig()`` (the audit's).
    predictions : tuple of pandas.DataFrame, optional
        ``(test, cal)`` from :func:`cross_fit_sigma` on the same re-indexed ``pairs`` and ``audit``; computed if
        omitted.
    log : callable, optional
        Receives progress messages.

    Returns
    -------
    ReferenceResult
        Per scene-and-draw counts and the aggregate tables.
    """
    config, audit = config or ReferenceConfig(), audit or AuditConfig()
    say = log or (lambda message: None)
    pairs = pairs.reset_index(drop=True)
    notes: list[str] = []
    if predictions is None:
        test, cal, notes = cross_fit_sigma(pairs, audit, say)
    else:
        test, cal = predictions
    draws = [run_dataset(sites, config, audit, say) for sites in prepare_sites(pairs, test, cal)]
    frame = pd.concat(draws, ignore_index=True) if draws else pd.DataFrame()
    tables = {"summary": summary_table(frame, audit.epsilon), "scenes": scene_table(frame, audit.epsilon)} \
        if len(frame) else {}
    return ReferenceResult(frame, tables, notes)


# ============================================================================================================
# 5. Aggregates
# ============================================================================================================


def _range(values: ArrayLike) -> tuple[float, float, float]:
    """Mean and 5th / 95th percentiles, ignoring NaN.

    Parameters
    ----------
    values : array_like
        Values.

    Returns
    -------
    mean, p05, p95 : float
        NaN if no value is finite.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return math.nan, math.nan, math.nan
    p05, p95 = np.percentile(v, [5, 95])
    return float(v.mean()), float(p05), float(p95)


def _with_rates(draws: pd.DataFrame) -> pd.DataFrame:
    """Per scene-and-draw rates from the counts.

    Parameters
    ----------
    draws : pandas.DataFrame
        :func:`run_dataset` rows.

    Returns
    -------
    pandas.DataFrame
        ``draws`` plus ``err_acc`` (NaN when nothing is accepted), ``yield``, ``coverage`` and ``err_all``.
    """
    n_acc, n_eval = draws["n_acc"].to_numpy(float), draws["n_eval"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return draws.assign(err_acc=np.where(n_acc > 0, draws["n_acc_wrong"] / n_acc, np.nan),
                            **{"yield": n_acc / n_eval}, coverage=draws["n_cov"] / n_eval,
                            err_all=draws["n_wrong"] / n_eval)


def _pooled(g: pd.DataFrame) -> pd.DataFrame:
    """Rates pooled over the scenes of each draw.

    Parameters
    ----------
    g : pandas.DataFrame
        :func:`run_dataset` rows of one dataset, variant, count and tolerance multiple.

    Returns
    -------
    pandas.DataFrame
        One row per draw: ``err_acc`` (accepted wrong / accepted; NaN if none), ``yield``, ``coverage``,
        ``err_all``.
    """
    s = g.groupby("draw")[["n_acc", "n_acc_wrong", "n_eval", "n_cov", "n_wrong"]].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame({"err_acc": np.where(s["n_acc"] > 0, s["n_acc_wrong"] / s["n_acc"], np.nan),
                             "yield": s["n_acc"] / s["n_eval"], "coverage": s["n_cov"] / s["n_eval"],
                             "err_all": s["n_wrong"] / s["n_eval"]}, index=s.index)


def summary_table(draws: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """Results per dataset, variant, reference count and tolerance multiple, with the no-reference comparison.

    Parameters
    ----------
    draws : pandas.DataFrame
        :func:`run_dataset` rows.
    epsilon : float
        The certified bound on the error among accepted labels.

    Returns
    -------
    pandas.DataFrame
        Columns:

        - ``dataset, variant, n_refs, tau_scale, n_scenes, n_draws``;
        - ``err_pooled`` / ``_p05`` / ``_p95``: error among accepted labels pooled over the scenes of a draw, mean
          and 90 % range over draws (draws accepting nothing are left out);
        - ``err_scene`` / ``_p05`` / ``_p95``: per scene and draw, mean and 90 % range (scene-draws accepting
          something);
        - ``yield_pooled`` (mean over draws of accepted / evaluated labels) and ``yield_scene`` / ``_p05`` /
          ``_p95``;
        - ``cov_pooled`` and ``cov_scene`` / ``_p05`` / ``_p95``: coverage of the 90 % interval;
        - ``err_all``: wrong share of the evaluated (corrected) labels before gating, pooled, mean over draws;
        - ``frac_ok``: share of scene-draws whose error among accepted is at most ``epsilon`` (accepting nothing
          counts as within the bound, as the audit's ``certified_ok``); ``frac_accepting``: share accepting
          something; ``frac_ok_accepting``: ``frac_ok`` among those;
        - ``n0_err_pooled, n0_yield_pooled, n0_cov_pooled, n0_frac_ok``: the audit's gate (no references) on
          the same scenes; ``d_err_pooled`` and ``d_yield_pooled``: this row minus that.
    """
    d = _with_rates(draws)
    rows = []
    for (dataset, variant, n, tau_scale), g in d.groupby(["dataset", "variant", "n_refs", "tau_scale"], sort=True):
        pooled = _pooled(g)
        base = d[(d["dataset"] == dataset) & (d["variant"] == variant) & (d["n_refs"] == 0)
                 & (d["tau_scale"] == tau_scale) & d["scene"].isin(set(g["scene"]))]
        base_pooled = _pooled(base)
        ok = (g["n_acc"] == 0) | (g["err_acc"] <= epsilon)
        accepting = g["n_acc"] > 0
        row: dict[str, Any] = {"dataset": dataset, "variant": variant, "n_refs": n, "tau_scale": tau_scale,
                               "n_scenes": g["scene"].nunique(), "n_draws": g["draw"].nunique()}
        for name, values in (("err_pooled", pooled["err_acc"]), ("err_scene", g["err_acc"]),
                             ("yield_scene", g["yield"]), ("cov_scene", g["coverage"])):
            row[name], row[f"{name}_p05"], row[f"{name}_p95"] = _range(values)
        row.update(yield_pooled=float(pooled["yield"].mean()), cov_pooled=float(pooled["coverage"].mean()),
                   err_all=float(pooled["err_all"].mean()), frac_ok=float(ok.mean()),
                   frac_accepting=float(accepting.mean()),
                   frac_ok_accepting=float(ok[accepting].mean()) if accepting.any() else math.nan,
                   n0_err_pooled=float(base_pooled["err_acc"].mean()),
                   n0_yield_pooled=float(base_pooled["yield"].mean()),
                   n0_cov_pooled=float(base_pooled["coverage"].mean()),
                   n0_frac_ok=float(((base["n_acc"] == 0) | (base["err_acc"] <= epsilon)).mean()))
        row["d_err_pooled"] = row["err_pooled"] - row["n0_err_pooled"]
        row["d_yield_pooled"] = row["yield_pooled"] - row["n0_yield_pooled"]
        rows.append(row)
    return pd.DataFrame(rows)


def scene_table(draws: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """Results per dataset, variant, reference count, tolerance multiple and test scene, over draws.

    Parameters
    ----------
    draws : pandas.DataFrame
        :func:`run_dataset` rows.
    epsilon : float
        The certified bound.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, variant, n_refs, tau_scale, scene, n_items, n_draws, n_eval``, ``err_acc`` (pooled
        over draws: accepted wrong / accepted), ``err_acc_p05``, ``err_acc_p95`` (over draws accepting
        something), ``frac_ok`` (draws within ``epsilon`` or accepting nothing), ``yield``, ``coverage``,
        ``err_all`` (means over draws), ``c_p50`` (median spread factor) and ``scale_<source>_p50`` (median
        fitted scale).
    """
    d = _with_rates(draws)
    scale_cols = [c for c in d.columns if c.startswith("scale_")]
    rows = []
    for key, g in d.groupby(["dataset", "variant", "n_refs", "tau_scale", "scene"], sort=True):
        ok = (g["n_acc"] == 0) | (g["err_acc"] <= epsilon)
        _, p05, p95 = _range(g["err_acc"])
        n_acc = g["n_acc"].sum()
        rows.append({**dict(zip(["dataset", "variant", "n_refs", "tau_scale", "scene"], key)),
                     "n_items": int(g["n_items"].iloc[0]), "n_draws": len(g), "n_eval": float(g["n_eval"].mean()),
                     "err_acc": float(g["n_acc_wrong"].sum() / n_acc) if n_acc else math.nan,
                     "err_acc_p05": p05, "err_acc_p95": p95, "frac_ok": float(ok.mean()),
                     "yield": float(g["yield"].mean()), "coverage": float(g["coverage"].mean()),
                     "err_all": float(g["err_all"].mean()), "c_p50": float(g["c"].median()),
                     **{f"{c}_p50": float(g[c].median()) for c in scale_cols}})
    return pd.DataFrame(rows)
