"""Audit automatic metric labels against the oracle, and gate them with a calibrated, out-of-scene uncertainty.

Every question is answered twice (see :mod:`measured_space.qa`): from ground truth (``oracle``) and from
perception (``auto``, once per depth source). This module asks whether a per-label uncertainty predicted only
from evidence available *without* ground truth is calibrated, and whether it can gate labels with a stated error
guarantee.

1. **Errors** (:func:`pair_table`). Numeric questions: ``e = auto - oracle`` in metres and ``e / oracle``. A
   label is *wrong* exactly when :func:`~measured_space.qa_schema.is_correct` rejects it, i.e. ``|e| > tau``
   with ``tau`` the question's tolerance at the oracle value. Categorical questions (choices and counts): a
   *flip* when the auto answer differs. A *miss* is a question perception could not answer because a referent
   had no box; misses are a separate rate, not errors.
2. **Structured error model** (:func:`fit_error_model`), one per dataset and numeric question type::

       e ~ (1 - pi) * N(x'beta, s0^2 + exp(x'gamma)) + pi * N(0, sigma_g^2)

   a bias linear in the features, a log-variance linear in the features, and a gross-error component (wrong
   referent, failed lift) with a per-type rate ``pi`` and a wide scale ``sigma_g``. It is fitted by maximum
   likelihood with a ridge penalty. The features (:data:`FEATURES`) are auto-side evidence only, aggregated
   over the item's referents by taking the worst referent (:func:`referent_features`), plus the depth source as
   one-hot columns. A referent's detection score is its lift's own ``det_score`` when the box source stores
   one (e.g. ``cutr_sam3``: the box detector's score, not the segmenter's), else its detection's ``score``.
   Match distances and pairing ambiguity are *not* features: matching a detection to an oracle object uses
   ground truth.
3. **Out-of-scene predictions** (:func:`cross_fit`). Leave one scene out within a dataset: each item's
   prediction comes from a model that never saw its scene. Calibration scores for a test scene come from the
   other scenes, each predicted by a model that saw neither that scene nor the test scene (leave two scenes
   out). Calibration models train on one scene fewer than the test model, so their scores err on the large
   side and the intervals on the conservative side.

   **Grouped folds above ``max_folds`` scenes.** Leave-two-out costs ``n + n (n - 1) / 2`` fits per model
   kind (5,671 at 106 scenes, each with its own inner ridge search). So when a dataset and question type has
   more than :attr:`AuditConfig.max_folds` scenes, its scenes are grouped into ``max_folds`` folds balanced by
   row count (:func:`scene_folds`: deterministic from the seed, never looking at errors or features) and the
   same scheme runs on folds: a model fitted without fold ``F`` predicts every scene of ``F``; for each pair
   of folds ``{F, G}``, a model fitted without both predicts the scenes of ``G`` as calibration data for every
   test scene of ``F``, and vice versa (``K + K (K - 1) / 2`` fits, 210 at ``K = 20``). With at most
   ``max_folds`` scenes every scene is its own fold, which is the leave-one-scene-out design above, exactly.
   The guarantees are kept: no prediction for a scene comes from a model that saw it, and no calibration
   score for a test scene comes from a model that saw the test scene or the scene being calibrated. Each test
   scene still gets its own calibration set of whole other scenes, so exchangeability is still assumed at the
   scene level and nowhere finer; a test scene's own fold-mates do not calibrate it. Calibration models now
   train on one fold fewer than the test model (about ``n / K`` scenes rather than one), so their scores err
   further on the large side. The inner ridge search (:func:`select_ridge`) groups its training scenes the
   same way.

   **Explicit fold groups** (:attr:`AuditConfig.fold_groups`). Scenes can share more than an object list: for
   example several synthetic warehouses built in one building footprint. Given an explicit scene-to-group
   mapping for a dataset, every question type of that dataset is cross-fitted with one fold per group,
   whatever ``max_folds`` (:func:`scene_folds`): leave one group out for the test predictions, leave two
   groups out for calibration, by the same fold-level scheme as above. No prediction for a scene then comes
   from a model that saw a scene of its group, and no scene of its group calibrates it. Exchangeability is
   assumed between groups, not between scenes. The inner ridge search, the scene-cluster bootstrap and the
   per-scene reporting are unchanged (still by scene).
4. **Conformal calibration** (:func:`conformal_calibrate`). Split conformal on ``|e - mu| / sigma``, Mondrian
   by question type and depth source (pooled over sources when a cell has too few calibration scores, which is
   recorded). Coverage is reported at nominal 50/80/90/95 % with Wilson intervals (:func:`coverage_table`). The
   numeric leave-two-out calibration predictions and every test scene's quantiles per cell
   (:func:`conformal_quantile_table`) are kept in :class:`AuditResult`, so a downstream analysis can rebuild a
   test scene's calibration set exactly as the audit used it.
5. **Categorical p(correct)** (:class:`LogisticPredictor`): a ridge logistic model on the features and the
   auto decision margin; Brier score and reliability bins against a constant-p baseline.
6. **Gate** (:func:`gate_decisions`). Score ``r = (|mu| + z * sigma) / tau_hat``, the model's 90 % bound on
   ``|e|`` in tolerances, with ``tau_hat`` the tolerance at the auto value (known without ground truth).
   Accept if ``r <= lambda``, reject if ``r > kappa`` (default 3: the bound exceeds three tolerances), else flag
   for review. ``lambda`` is chosen per test scene on its calibration scenes by Learn-then-Test
   (:func:`ltt_threshold`): fixed-sequence binomial tests certify ``P(wrong | accepted) <= epsilon`` at
   confidence ``1 - delta``. Categorical items use ``1 - p(correct)`` as the score and reject below
   ``p = 0.5``. Risk-yield curves (:func:`risk_yield`) compare rankings.

Baselines, fitted on the same folds with the same likelihood: ``const`` (bias and sigma constant per question
type) and ``prop`` (bias and sigma proportional to the label value, ``sigma = k |value|``). Mondrian calibration
then makes ``const`` a per type-and-depth-source constant.

Pairing sensitivity (:func:`exclude_ambiguous`): the pair table records per row whether any of the auto answer's
detection pairings was ambiguous (another feasible pairing existed; ``match_ambiguous``), and the rows whose
pairing was can be dropped before the audit, oracle and auto side together. Rows whose flag is unknown (labels
written before it was stored, or under the 3D association) are kept and counted.

Caveats stated with every result: items from one scene share objects, a capture and its errors, so Wilson
intervals and binomial bounds, which treat items as independent, are optimistic; guarantees hold across
scenes only as far as scenes are exchangeable; no coverage claim is made per site.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special, stats

from measured_space.qa_schema import QAItem, is_correct, read_jsonl

Array = NDArray[np.float64]
T = TypeVar("T")

NUMERIC_QTYPES: tuple[str, ...] = ("distance_center", "distance_surface", "size_height", "size_longest",
                                   "ego_distance")
"""tuple of str: question types with a numeric answer in metres."""

CATEGORICAL_QTYPES: tuple[str, ...] = ("relative_depth", "left_right", "closest_instance", "count")
"""tuple of str: question types with a categorical answer (a choice or a count)."""

LEVELS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)
"""tuple of float: nominal coverage levels reported."""

FEATURES: tuple[str, ...] = ("log_range", "boundary_fraction", "border_touch", "depth_mad_rel", "hull_fill",
                             "log_n_points", "det_score", "log_value")
"""tuple of str: auto-side features of the numeric error model (see :func:`referent_features`)."""

CATEGORICAL_FEATURES: tuple[str, ...] = FEATURES[:-1] + ("log_margin",)
"""tuple of str: features of the categorical model; the label value is replaced by the auto decision margin."""

NUMERIC_KINDS: tuple[str, ...] = ("sigma", "const", "prop")
"""tuple of str: numeric error models: the feature model and the two constant-sigma baselines."""

CATEGORICAL_KINDS: tuple[str, ...] = ("logistic", "margin", "const")
"""tuple of str: categorical p(correct) models: all features, margin and range only, constant."""

# Auto-side lift features (see measured_space.perception.lift_instance) and how they are aggregated over an
# item's referents: the worst referent in the direction that makes a box less reliable.
_LIFT_KEYS = ("depth_median", "boundary_fraction", "border_touch", "depth_mad_rel", "footprint_hull_fill",
              "n_points")
_SCORE_KEYS = ("det_score", "detection_score", "score")
_LIFT_SCORE_KEY = "det_score"    # a lift's own detection score, preferred over its referent's (see _feature_dicts)
_FEATURES_KEY = "features"       # where a referent's evidence holds its lift features
_MATCH_KEYS = ("match_distance", "match_dist", "match_cost")
_AMBIGUOUS_KEY = "match_ambiguous"
_MIN_VALUE = 0.05          # m: floor on |value| inside logs and the proportional baseline
_MIN_MARGIN = 1e-3         # m: floor on the decision margin inside its log
_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class AuditConfig:
    """Every setting of the audit.

    Attributes
    ----------
    levels : tuple of float
        Nominal coverage levels reported (default :data:`LEVELS`).
    ridge_grid : tuple of float
        Candidate L2 penalty weights on standardised, non-intercept coefficients (added to the summed negative
        log-likelihood) of the feature models (``sigma``, ``logistic``, ``margin``). With several candidates,
        every fit picks its own by leave-one-scene-out cross-validation on its training scenes only
        (:func:`select_ridge`; grouped ``max_folds``-fold cross-validation when the training rows come from
        more than ``max_folds`` scenes), so a fold never sees its test or calibration scene; with one, the
        weight is fixed. Default ``(1, 10, ..., 1e5)``: from weak to strong enough to shrink the feature model to the
        constant baseline (a first grid ending at 1000 had every small-data fit choose its edge).
    n_jobs : int
        Worker processes for cross-fitting (default 1).
    gross : bool
        Fit the gross-error component of the numeric model (default True).
    s0 : float
        Noise floor added to every predicted variance, metres (default 0.005).
    mondrian : tuple of str
        Columns that define a conformal calibration cell (default question type and depth source).
    min_cal : int
        A cell with fewer calibration scores falls back to the question type alone (default 20).
    gate_level : float
        Two-sided Gaussian level of the gate score's bound on ``|e|`` (default 0.9, ``z = 1.645``).
    epsilon : float
        Certified bound on ``P(wrong | accepted)`` (default 0.1).
    delta : float
        The certificate holds with probability at least ``1 - delta`` (default 0.1).
    kappa : float
        Numeric labels whose gate score exceeds this many tolerances are rejected (default 3.0).
    min_accept : int
        Learn-then-Test's first candidate threshold accepts at least this many calibration items (default 30).
    min_accept_frac : float
        ... and at least this share of them (default 0.05). In simulation, a first step of 30 items lost power on
        large calibration sets (one early error stops the sequence) and one of 100 lost it on small ones.
    tau_scales : tuple of float
        Multiples of the question's tolerance at which the gate is evaluated (default ``(1.0, 2.0)``).
    n_bins : int
        Reliability-diagram bins (default 10).
    min_scenes : int
        Fewest scenes a dataset and question type need to be modelled (default 3; leave-two-out needs 3).
    min_rows : int
        Fewest labelled rows a dataset and question type need to be modelled (default 20).
    max_folds : int
        Most cross-fitting folds per dataset and question type (default 20, at least 3). With at most this many
        scenes, every scene is its own fold (leave one scene out; leave two out for calibration). With more,
        scenes are grouped into this many folds balanced by row count (:func:`scene_folds`) and the same scheme
        runs on folds (:func:`cross_fit`); the inner ridge search groups its training scenes the same way.
    seed : int
        Seed for tie-breaking (including the fold assignment) and the cluster bootstrap (default 0).
    n_boot : int
        Scene-cluster bootstrap replicates for the headline coverage interval (default 2000).
    fold_groups : mapping or None
        Explicit cross-fitting groups, ``{dataset: {scene: group}}`` (stored as plain dicts of str). For a
        dataset listed here, every question type is cross-fitted with one fold per group (:func:`scene_folds`),
        ignoring ``max_folds``, and every scene of the dataset must be mapped; datasets not listed use the
        default folds. A question type whose scenes fall in fewer than 3 groups is skipped. Default None: no
        explicit groups. Excluded from the hash.
    """

    levels: tuple[float, ...] = LEVELS
    ridge_grid: tuple[float, ...] = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
    n_jobs: int = 1
    gross: bool = True
    s0: float = 0.005
    mondrian: tuple[str, ...] = ("qtype", "depth_source")
    min_cal: int = 20
    gate_level: float = 0.9
    epsilon: float = 0.1
    delta: float = 0.1
    kappa: float = 3.0
    min_accept: int = 30
    min_accept_frac: float = 0.05
    tau_scales: tuple[float, ...] = (1.0, 2.0)
    n_bins: int = 10
    min_scenes: int = 3
    min_rows: int = 20
    max_folds: int = 20
    seed: int = 0
    n_boot: int = 2000
    fold_groups: Mapping[str, Mapping[str, str]] | None = field(default=None, hash=False)

    def __post_init__(self) -> None:
        """Check the settings that would make cross-fitting impossible, and store ``fold_groups`` as plain dicts.

        Raises
        ------
        ValueError
            If ``max_folds`` is below 3 (leaving two folds out needs a third to train on), or ``fold_groups`` is
            not a mapping of datasets to mappings of scenes to groups.
        """
        if self.max_folds < 3:
            raise ValueError(f"max_folds must be at least 3, got {self.max_folds}")
        if self.fold_groups is not None:
            if not isinstance(self.fold_groups, Mapping) or not all(isinstance(m, Mapping)
                                                                    for m in self.fold_groups.values()):
                raise ValueError("fold_groups must map each dataset to a {scene: group} mapping")
            object.__setattr__(self, "fold_groups", {str(d): {str(s): str(g) for s, g in m.items()}
                                                     for d, m in self.fold_groups.items()})

    def groups_of(self, dataset: str) -> Mapping[str, str] | None:
        """The explicit scene-to-group mapping of one dataset.

        Parameters
        ----------
        dataset : str
            Dataset name.

        Returns
        -------
        mapping of str to str or None
            ``fold_groups[dataset]``, or None if there are no explicit groups or the dataset is not listed.
        """
        return None if self.fold_groups is None else self.fold_groups.get(str(dataset))

    @property
    def gate_z(self) -> float:
        """float: the Gaussian multiplier of the gate score, ``Phi^-1((1 + gate_level) / 2)``."""
        return float(stats.norm.ppf(0.5 * (1.0 + self.gate_level)))


# ============================================================================================================
# 1. Loading: oracle and auto answers side by side
# ============================================================================================================


def _feature_dicts(node: Any, found: list[dict[str, Any]], scores: list[float], matches: list[float],
                   skip_scores: bool = False) -> None:
    """Collect lift-feature dicts, detection scores and match distances anywhere in a nested record.

    A dict's detection score is its lift's own ``det_score`` when it holds a ``features`` dict with a finite
    one: some box sources store two scores per referent, the segmenter's as the referent's ``score`` and the
    box detector's as the lift's ``det_score`` (``cutr_sam3``), and the box is the detector's. Otherwise every
    finite score key (``_SCORE_KEYS``) of the dict counts. Lift features without ``det_score`` (the depth
    sources) therefore leave the scores exactly as a plain search for score keys finds them.

    Parameters
    ----------
    node : Any
        A JSON-like value (dict, list or scalar).
    found : list of dict
        Receives every dict holding at least two lift-feature keys (one per referent).
    scores : list of float
        Receives the detection scores: per dict, its lift's ``det_score`` if present and finite, else every
        finite value of its keys in ``_SCORE_KEYS``.
    matches : list of float
        Receives every finite match distance (keys in ``_MATCH_KEYS``).
    skip_scores : bool, default False
        Leave out this dict's own score keys (its ``det_score`` was already taken by the referent holding it).
    """
    if isinstance(node, Mapping):
        if sum(k in node for k in _LIFT_KEYS) >= 2:
            found.append(dict(node))
        lift = node.get(_FEATURES_KEY)
        lift_score = lift.get(_LIFT_SCORE_KEY) if isinstance(lift, Mapping) else None
        prefer_lift = _finite(lift_score)
        if prefer_lift:
            scores.append(float(lift_score))
        elif not skip_scores:
            scores.extend(float(node[k]) for k in _SCORE_KEYS if k in node and _finite(node[k]))
        for key, value in node.items():
            if key in _MATCH_KEYS and _finite(value):
                matches.append(float(value))
            elif isinstance(value, (Mapping, list, tuple)):
                _feature_dicts(value, found, scores, matches, skip_scores=prefer_lift and key == _FEATURES_KEY)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _feature_dicts(value, found, scores, matches)


def _ambiguity_flags(node: Any, flags: list[bool | None]) -> None:
    """Collect, for every pairing record anywhere in a nested record, whether its pairing was ambiguous.

    A pairing record is a dict with ``match_ambiguous`` or a match-distance key (``_MATCH_KEYS``): the
    evidence the label pipeline stores per referent or counted object (:func:`measured_space.pipeline._evidence`).
    Records are not searched further.

    Parameters
    ----------
    node : Any
        A JSON-like value (dict, list or scalar).
    flags : list of bool or None
        Receives one flag per record: False if the record has ``object_id`` None (no oracle object was paired,
        so no pairing can be ambiguous), else ``bool(match_ambiguous)``, or None (unknown) if that is absent or
        None (labels written before the field existed, or under the 3D association, which does not assess
        ambiguity).
    """
    if isinstance(node, Mapping):
        if _AMBIGUOUS_KEY in node or any(k in node for k in _MATCH_KEYS):
            if "object_id" in node and node["object_id"] is None:
                flags.append(False)
            else:
                value = node.get(_AMBIGUOUS_KEY)
                flags.append(None if value is None else bool(value))
            return
        for value in node.values():
            if isinstance(value, (Mapping, list, tuple)):
                _ambiguity_flags(value, flags)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _ambiguity_flags(value, flags)


def match_ambiguity(extra: Mapping[str, Any]) -> float:
    """Whether an auto item rests on an ambiguous detection pairing.

    Parameters
    ----------
    extra : mapping
        An auto item's ``extra`` (searched recursively, see :func:`_ambiguity_flags`).

    Returns
    -------
    float
        1.0 if some referent (for a count, some counted object) was paired while another feasible pairing
        existed (:attr:`measured_space.matching.Match.ambiguous`); NaN (unknown) if not, but some pairing's flag
        is unknown; else 0.0, including an item that records no pairing at all.
    """
    flags: list[bool | None] = []
    _ambiguity_flags(extra, flags)
    if any(f is True for f in flags):
        return 1.0
    return math.nan if any(f is None for f in flags) else 0.0


def _finite(value: Any) -> bool:
    """Tell whether a value is a finite real number (not a bool).

    Parameters
    ----------
    value : Any
        Anything.

    Returns
    -------
    bool
        True for finite ints and floats.
    """
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _agg(dicts: Sequence[Mapping[str, Any]], key: str, how: Callable[[Array], float]) -> float:
    """Aggregate one feature over referents, ignoring referents that lack it.

    Parameters
    ----------
    dicts : sequence of mapping
        One feature dict per referent.
    key : str
        Feature name.
    how : callable
        Reduction over the finite values, e.g. ``np.max``.

    Returns
    -------
    float
        The reduced value, or NaN if no referent has a finite value.
    """
    values = np.array([float(d[key]) for d in dicts if key in d and _finite(d[key])])
    return float(how(values)) if len(values) else math.nan


def referent_features(extra: Mapping[str, Any]) -> dict[str, float]:
    """Auto-side evidence of one item, aggregated over its referents (worst referent).

    The record is searched recursively, so the features may sit in ``extra`` directly, in a list per referent
    or in a dict keyed by object id. A per-referent feature dict is any dict holding at least two of the
    :func:`~measured_space.perception.lift_instance` keys.

    Parameters
    ----------
    extra : mapping
        An auto item's ``extra``.

    Returns
    -------
    dict of str to float
        ``log_range``: log of the largest ``depth_median`` (m). ``boundary_fraction``, ``border_touch``,
        ``depth_mad_rel``: largest over referents. ``hull_fill``: smallest ``footprint_hull_fill``.
        ``log_n_points``: log of the smallest ``n_points``. ``det_score``: smallest detection score over
        referents, each referent's being its lift's own ``det_score`` when present and finite, else its
        ``score`` (see :func:`_feature_dicts`). ``match_distance``: largest match distance, and
        ``match_ambiguous``: :func:`match_ambiguity` (diagnostics, never model features, since matching uses
        ground truth). ``n_feature_sets``: referents with lift features found. NaN where nothing was found
        (except ``match_ambiguous``, 0 when no pairing is recorded).
    """
    found: list[dict[str, Any]] = []
    scores: list[float] = []
    matches: list[float] = []
    _feature_dicts(extra, found, scores, matches)
    range_max = _agg(found, "depth_median", np.max)
    n_points = _agg(found, "n_points", np.min)
    return {
        "log_range": math.log(range_max) if range_max > 0 else math.nan,
        "boundary_fraction": _agg(found, "boundary_fraction", np.max),
        "border_touch": _agg(found, "border_touch", np.max),
        "depth_mad_rel": _agg(found, "depth_mad_rel", np.max),
        "hull_fill": _agg(found, "footprint_hull_fill", np.min),
        "log_n_points": math.log(n_points) if n_points > 0 else math.nan,
        "det_score": min(scores) if scores else math.nan,
        "match_distance": max(matches) if matches else math.nan,
        "match_ambiguous": match_ambiguity(extra),
        "n_feature_sets": float(len(found)),
    }


def _tolerance(tolerance: Mapping[str, float | None], reference: float) -> float:
    """The largest error a numeric answer may have and still be correct (as ``is_correct``).

    Parameters
    ----------
    tolerance : mapping
        ``{"abs": metres or None, "rel": fraction or None}``.
    reference : float
        The value the relative bound scales with (m).

    Returns
    -------
    float
        ``max(abs, rel * |reference|)`` over the bounds present (m); 0 if there are none.
    """
    bounds = [b for b in (tolerance.get("abs"), None if tolerance.get("rel") is None
                          else tolerance["rel"] * abs(reference)) if b is not None]
    return float(max(bounds)) if bounds else 0.0


def pair_table(oracle: Iterable[QAItem], auto: Iterable[QAItem], depth_source: str) -> pd.DataFrame:
    """One row per oracle question: its auto answer from one depth source (if any), the error and the features.

    Parameters
    ----------
    oracle : iterable of QAItem
        Oracle items (``source == "oracle"``).
    auto : iterable of QAItem
        Auto items from one depth source (``source`` ``"auto"`` or ``"auto_<depth source>"``), paired with the
        oracle by item id.
    depth_source : str
        The auto labels' depth source, e.g. ``"moge"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``id, dataset, scene, camera_id, frame, qtype, answer_type, depth_source, matched, oracle,
        auto, value, e, abs_e, rel_e, tau, tau_hat, wrong, margin, n_referents`` and the feature columns of
        :func:`referent_features` plus ``log_value`` and ``log_margin``. ``matched`` is False for a miss (no auto
        answer), whose answer, error and features are NaN. ``value`` is the auto answer of a numeric question;
        ``tau`` is its tolerance at the oracle value and ``tau_hat`` at the auto value (m). ``wrong`` is
        ``not is_correct(spec, auto, oracle)`` (NaN for a miss). ``margin`` is the auto decision margin (m).
        ``attrs["n_auto_only"]`` counts auto items without an oracle item.

    Raises
    ------
    ValueError
        If an item has the wrong source or an id repeats within a source.
    """
    by_oracle, by_auto = _index(oracle, "oracle"), _index(auto, "auto")
    rows = []
    for item_id, o in by_oracle.items():
        spec, a = o.spec, by_auto.get(item_id)
        row: dict[str, Any] = {
            "id": item_id, "dataset": spec.dataset, "scene": spec.scene, "camera_id": spec.camera_id,
            "frame": spec.frame, "qtype": spec.qtype, "answer_type": spec.answer_type,
            "depth_source": depth_source, "matched": a is not None, "oracle": o.answer,
            "auto": None if a is None else a.answer, "n_referents": len(spec.referents)}
        numeric = spec.answer_type == "number"
        feats = referent_features(a.extra) if a is not None else referent_features({})
        if a is None:
            feats = {k: math.nan for k in feats}
        row.update(feats)
        margin = a.extra.get("margin") if a is not None else None
        row["margin"] = float(margin) if _finite(margin) else math.nan
        row["log_margin"] = math.log(max(row["margin"], _MIN_MARGIN)) if _finite(margin) else math.nan
        if numeric:
            ref = float(o.answer)
            row["tau"] = _tolerance(spec.tolerance, ref)
            if a is not None:
                value = float(a.answer)
                row.update(value=value, e=value - ref, abs_e=abs(value - ref),
                           rel_e=(value - ref) / ref if ref != 0 else math.nan,
                           tau_hat=_tolerance(spec.tolerance, value),
                           log_value=math.log(max(abs(value), _MIN_VALUE)))
        row["wrong"] = math.nan if a is None else float(not is_correct(spec, a.answer, o.answer))
        rows.append(row)
    columns = ["id", "dataset", "scene", "camera_id", "frame", "qtype", "answer_type", "depth_source", "matched",
               "oracle", "auto", "value", "e", "abs_e", "rel_e", "tau", "tau_hat", "wrong", "margin",
               "n_referents", *referent_features({}).keys(), "log_value", "log_margin"]
    table = pd.DataFrame(rows).reindex(columns=columns)
    for col in ("oracle", "auto"):
        table[col] = table[col].astype(object)
    table.attrs["n_auto_only"] = len(set(by_auto) - set(by_oracle))
    return table


def _index(items: Iterable[QAItem], source: str) -> dict[str, QAItem]:
    """Index items of one source by question id.

    Parameters
    ----------
    items : iterable of QAItem
        Items that should all have ``source``.
    source : str
        Expected source: ``"oracle"``, or ``"auto"``, which also accepts ``"auto_<depth source>"``.

    Returns
    -------
    dict of str to QAItem
        Id to item, in input order.

    Raises
    ------
    ValueError
        If an item has another source or an id repeats.
    """
    out: dict[str, QAItem] = {}
    for item in items:
        if item.source != source and not (source == "auto" and item.source.startswith("auto_")):
            raise ValueError(f"Item {item.id} has source {item.source!r}, expected {source!r}")
        if item.id in out:
            raise ValueError(f"Duplicate {source} item id {item.id}")
        out[item.id] = item
    return out


def _views_by_source(label_dir: Path) -> dict[str, set[tuple[str, str, str]]] | None:
    """Which views each depth source was run on, from the label pipeline's ``views.jsonl``.

    Parameters
    ----------
    label_dir : pathlib.Path
        A dataset's label folder.

    Returns
    -------
    dict of str to set of tuple, or None
        Depth source -> ``(scene, camera_id, frame)`` of every view it covers; None if there is no
        ``views.jsonl`` or it lists no ``depth_sources``.
    """
    path = label_dir / "views.jsonl"
    if not path.exists():
        return None
    out: dict[str, set[tuple[str, str, str]]] = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                view = json.loads(line)
                for source in view.get("depth_sources") or ():
                    out.setdefault(source, set()).add((view["scene"], view["camera_id"], str(view["frame"])))
    return out or None


def load_labels(label_dir: str | Path, sources: Sequence[str] | None = None) -> pd.DataFrame:
    """Load one dataset's oracle labels and every depth source's auto labels as a pair table.

    A depth source is paired only with questions about views it was run on (from ``views.jsonl``; without it,
    about scenes where it answered at least one question), so a source that exists for some scenes only (e.g.
    rendered depth for one warehouse) does not turn the other scenes' questions into misses.

    Parameters
    ----------
    label_dir : str or pathlib.Path
        Folder with ``qa_oracle.jsonl``, ``qa_auto_<depth source>.jsonl`` files and optionally ``views.jsonl``.
    sources : sequence of str, optional
        Depth sources to load. The other sources' auto files are not read, so their (question, depth source)
        rows never enter the pair table. Default: every ``qa_auto_<depth source>.jsonl`` in the folder.

    Returns
    -------
    pandas.DataFrame
        :func:`pair_table` rows for every loaded depth source, stacked. ``attrs["n_auto_only"]`` maps each
        loaded depth source to its count of auto items without an oracle item, ``attrs["n_not_run"]`` to its
        count of oracle questions about views it was not run on (left out).

    Raises
    ------
    FileNotFoundError
        If the oracle file or every auto file is missing.
    ValueError
        If ``sources`` is empty or names a source with no auto file in the folder.
    """
    label_dir = Path(label_dir)
    oracle = list(read_jsonl(label_dir / "qa_oracle.jsonl"))
    auto_files = sorted(label_dir.glob("qa_auto_*.jsonl"))
    if not auto_files:
        raise FileNotFoundError(f"No qa_auto_*.jsonl in {label_dir}")
    if sources is not None:
        available = {re.sub(r"^qa_auto_", "", p.stem) for p in auto_files}
        unknown = sorted(set(sources) - available)
        if not len(sources) or unknown:
            raise ValueError(f"Depth sources {sorted(sources)} requested from {label_dir}: "
                             + (f"no qa_auto_<source>.jsonl for {unknown}; " if unknown else "none given; ")
                             + f"available: {sorted(available)}")
        auto_files = [p for p in auto_files if re.sub(r"^qa_auto_", "", p.stem) in set(sources)]
    views = _views_by_source(label_dir)
    tables, auto_only, not_run = [], {}, {}
    for path in auto_files:
        source = re.sub(r"^qa_auto_", "", path.stem)
        auto = list(read_jsonl(path))
        if views is not None and source in views:
            covered = views[source]
            keep = [o for o in oracle if (o.spec.scene, o.spec.camera_id, str(o.spec.frame)) in covered]
        else:
            scenes = {a.spec.scene for a in auto}
            keep = [o for o in oracle if o.spec.scene in scenes]
        table = pair_table(keep, auto, source)
        auto_only[source] = table.attrs["n_auto_only"]
        not_run[source] = len(oracle) - len(keep)
        tables.append(table)
    out = pd.concat(tables, ignore_index=True)
    out.attrs.update(n_auto_only=auto_only, n_not_run=not_run)
    return out


AMBIGUOUS_UNKNOWN_POLICY = "kept"
"""str: what :func:`exclude_ambiguous` does with a labelled row whose pairing ambiguity is unknown."""


def exclude_ambiguous(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, int]]]]:
    """Drop the rows whose auto answer rests on an ambiguous detection pairing (a pairing sensitivity analysis).

    A row is dropped when its ``match_ambiguous`` is 1 (:func:`match_ambiguity`): some referent, or for a count
    some counted object, was paired with its detection while another feasible pairing existed. The whole row
    goes, the auto answer and the oracle question it is paired with, so it counts neither as labelled nor as a
    miss. Labelled rows whose flag is unknown (NaN, or no ``match_ambiguous`` column: labels written before the
    flag was stored, or under the 3D association, which does not assess ambiguity) are kept
    (:data:`AMBIGUOUS_UNKNOWN_POLICY`) and counted. Misses have no pairing and are kept.

    Parameters
    ----------
    pairs : pandas.DataFrame
        A pair table (:func:`load_labels`, :func:`pair_table`, :func:`synthetic_pairs`).

    Returns
    -------
    kept : pandas.DataFrame
        ``pairs`` without the dropped rows, with a fresh ``RangeIndex`` and the same ``attrs``.
    counts : dict
        ``{dataset: {depth_source: {"n_rows", "n_labelled", "n_dropped", "n_unknown_kept"}}}``: rows and
        labelled rows before the drop, rows dropped, and labelled rows kept with an unknown flag.
    """
    matched = pairs["matched"].astype(bool).to_numpy()
    flag = pairs["match_ambiguous"].to_numpy(dtype=np.float64) if "match_ambiguous" in pairs \
        else np.full(len(pairs), math.nan)
    drop = matched & (flag == 1.0)
    unknown = matched & ~np.isfinite(flag)
    counts: dict[str, dict[str, dict[str, int]]] = {}
    frame = pd.DataFrame({"dataset": pairs["dataset"].to_numpy(), "depth_source": pairs["depth_source"].to_numpy(),
                          "matched": matched, "drop": drop, "unknown": unknown})
    for (dataset, source), g in frame.groupby(["dataset", "depth_source"], sort=True):
        counts.setdefault(str(dataset), {})[str(source)] = {
            "n_rows": int(len(g)), "n_labelled": int(g["matched"].sum()), "n_dropped": int(g["drop"].sum()),
            "n_unknown_kept": int(g["unknown"].sum())}
    kept = pairs[~drop].reset_index(drop=True)
    kept.attrs = dict(pairs.attrs)
    return kept, counts


def exclusion_record(counts: Mapping[str, Mapping[str, Mapping[str, int]]] | None) -> dict[str, Any]:
    """The settings record of the ambiguous-pair exclusion, for a run's summary.

    Parameters
    ----------
    counts : mapping or None
        The counts returned by :func:`exclude_ambiguous`; None if the exclusion was off.

    Returns
    -------
    dict
        ``{"enabled": False}``, or ``{"enabled": True, "unknown_policy", "n_dropped", "n_unknown_kept",
        "counts"}`` with the totals over datasets and depth sources and ``counts`` as given.
    """
    if counts is None:
        return {"enabled": False}
    cells = [c for by_source in counts.values() for c in by_source.values()]
    return {"enabled": True, "unknown_policy": AMBIGUOUS_UNKNOWN_POLICY,
            "n_dropped": sum(int(c["n_dropped"]) for c in cells),
            "n_unknown_kept": sum(int(c["n_unknown_kept"]) for c in cells),
            "counts": {d: {s: dict(c) for s, c in by.items()} for d, by in counts.items()}}


# ============================================================================================================
# 2. Descriptive statistics
# ============================================================================================================


def wilson_interval(k: ArrayLike, n: ArrayLike, level: float = 0.95) -> tuple[Array, Array]:
    """Wilson score interval for binomial proportions.

    Parameters
    ----------
    k : array_like
        Successes.
    n : array_like
        Trials; broadcasts against ``k``. ``n == 0`` gives NaN bounds.
    level : float, default 0.95
        Two-sided confidence level.

    Returns
    -------
    lower, upper : numpy.ndarray
        The interval bounds, in [0, 1].
    """
    k, n = np.broadcast_arrays(np.asarray(k, dtype=np.float64), np.asarray(n, dtype=np.float64))
    z = stats.norm.ppf(0.5 * (1.0 + level))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1.0 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        lower, upper = np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)
    bad = n <= 0
    return np.where(bad, np.nan, lower), np.where(bad, np.nan, upper)


def _rate(flags: pd.Series) -> dict[str, float]:
    """A rate with its count and Wilson interval.

    Parameters
    ----------
    flags : pandas.Series
        Booleans or 0/1 values; NaN entries are ignored.

    Returns
    -------
    dict of str to float
        ``n``, ``k``, ``rate``, ``lo``, ``hi``.
    """
    values = flags.dropna().astype(float)
    n, k = len(values), float(values.sum())
    lo, hi = wilson_interval(k, n)
    return {"n": n, "k": k, "rate": k / n if n else math.nan, "lo": float(lo), "hi": float(hi)}


def error_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Error statistics per dataset, question type and depth source.

    Parameters
    ----------
    pairs : pandas.DataFrame
        Output of :func:`pair_table` / :func:`load_labels`.

    Returns
    -------
    pandas.DataFrame
        One row per group. Always: ``n_scenes``, ``n_questions``, ``n_labelled``, ``miss_rate`` (+ ``miss_lo``, ``miss_hi``),
        ``wrong_rate`` (+ Wilson ``wrong_lo``, ``wrong_hi``). Numeric types also: ``bias_median`` (m, median
        ``e``), ``mae`` (m), ``abs_p50``, ``abs_p90`` (m), ``rmse`` (m), ``rel_abs_p50`` (median ``|e| / oracle``),
        ``tau_p50`` (m) and ``gross_rate`` (share with ``|e| > 5 tau``). For categorical types ``wrong_rate`` is
        the flip rate.
    """
    rows = []
    for (dataset, qtype, source), g in pairs.groupby(["dataset", "qtype", "depth_source"], sort=True):
        labelled = g[g["matched"]]
        miss, wrong = _rate(~g["matched"]), _rate(labelled["wrong"])
        row: dict[str, Any] = {"dataset": dataset, "qtype": qtype, "depth_source": source,
                               "answer_type": g["answer_type"].iloc[0], "n_scenes": g["scene"].nunique(),
                               "n_questions": len(g),
                               "n_labelled": len(labelled), "miss_rate": miss["rate"], "miss_lo": miss["lo"],
                               "miss_hi": miss["hi"], "wrong_rate": wrong["rate"], "wrong_lo": wrong["lo"],
                               "wrong_hi": wrong["hi"]}
        if row["answer_type"] == "number" and len(labelled):
            e = labelled["e"].to_numpy(float)
            row.update(bias_median=float(np.median(e)), mae=float(np.mean(np.abs(e))),
                       abs_p50=float(np.median(np.abs(e))), abs_p90=float(np.quantile(np.abs(e), 0.9)),
                       rmse=float(np.sqrt(np.mean(e**2))),
                       rel_abs_p50=float(np.nanmedian(np.abs(labelled["rel_e"].to_numpy(float)))),
                       tau_p50=float(labelled["tau"].median()),
                       gross_rate=float(np.mean(np.abs(e) > 5 * labelled["tau"].to_numpy(float))))
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================================================
# 3. Models: the structured numeric error model and the categorical p(correct)
# ============================================================================================================


@dataclass(frozen=True)
class ErrorModel:
    """A fitted structured error model: bias, log-variance and a gross-error component.

    Attributes
    ----------
    beta : numpy.ndarray, shape (pm,)
        Bias coefficients (m per unit of each mean-design column).
    gamma : numpy.ndarray, shape (pv,)
        Log-variance coefficients (log m^2 per unit of each variance-design column).
    pi_gross : float
        Share of gross errors (0 if fitted without the component).
    sigma_gross : float
        Standard deviation of gross errors, metres (NaN without the component).
    s0 : float
        Noise floor added to every variance, metres.
    nll : float
        Mean training negative log-likelihood per item (nats), without the penalty.
    converged : bool
        Whether the optimiser reported success.
    """

    beta: Array
    gamma: Array
    pi_gross: float
    sigma_gross: float
    s0: float
    nll: float
    converged: bool

    def predict(self, Xm: Array, Xv: Array, offset: Array | None = None) -> tuple[Array, Array]:
        """Predicted bias and spread of the non-gross component.

        Parameters
        ----------
        Xm : numpy.ndarray, shape (n, pm)
            Mean design matrix.
        Xv : numpy.ndarray, shape (n, pv)
            Variance design matrix.
        offset : numpy.ndarray, shape (n,), optional
            Fixed log-variance offset (log m^2); zero if omitted.

        Returns
        -------
        mu : numpy.ndarray, shape (n,)
            Predicted bias of ``e`` (m).
        sigma : numpy.ndarray, shape (n,)
            Predicted standard deviation of ``e`` around ``mu`` (m).
        """
        eta = Xv @ self.gamma + (0.0 if offset is None else offset)
        return Xm @ self.beta, np.sqrt(np.exp(np.clip(eta, -40.0, 20.0)) + self.s0**2)


def _mixture_objective(params: Array, e: Array, Xm: Array, Xv: Array, offset: Array, pen_m: Array, pen_v: Array,
                       ridge: float, s0: float, floor_g: float, gross: bool) -> tuple[float, Array]:
    """Penalised negative log-likelihood of the structured error model, and its gradient.

    Parameters
    ----------
    params : numpy.ndarray, shape (pm + pv [+ 2],)
        ``beta``, ``gamma`` and, with ``gross``, ``a`` (``pi = sigmoid(a) / 2``) and ``b``
        (``sigma_g = floor_g * (1 + exp(b))``).
    e : numpy.ndarray, shape (n,)
        Errors (m).
    Xm : numpy.ndarray, shape (n, pm)
        Mean design.
    Xv : numpy.ndarray, shape (n, pv)
        Variance design.
    offset : numpy.ndarray, shape (n,)
        Fixed log-variance offset.
    pen_m : numpy.ndarray, shape (pm,)
        1 for penalised mean coefficients, 0 otherwise.
    pen_v : numpy.ndarray, shape (pv,)
        1 for penalised variance coefficients, 0 otherwise.
    ridge : float
        Penalty weight.
    s0 : float
        Variance floor (m).
    floor_g : float
        Smallest gross-error standard deviation (m).
    gross : bool
        Include the gross-error component.

    Returns
    -------
    value : float
        Summed negative log-likelihood plus penalty.
    gradient : numpy.ndarray
        Its gradient with respect to ``params``.
    """
    pm, pv = Xm.shape[1], Xv.shape[1]
    beta, gamma = params[:pm], params[pm:pm + pv]
    eta = np.clip(Xv @ gamma + offset, -40.0, 20.0)
    ex = np.exp(eta)
    s2 = ex + s0**2
    r = e - Xm @ beta
    log_n1 = -0.5 * (_LOG_2PI + np.log(s2) + r * r / s2)
    grad = np.zeros_like(params)
    if gross:
        a, b = params[pm + pv], params[pm + pv + 1]
        sig_a = special.expit(a)
        pi = 0.5 * sig_a
        sg = floor_g * (1.0 + math.exp(min(b, 30.0)))
        log_n2 = -0.5 * (_LOG_2PI + 2.0 * math.log(sg) + e * e / sg**2)
        log_pi = math.log(0.5) - float(np.logaddexp(0.0, -a))
        l1, l2 = math.log1p(-pi) + log_n1, log_pi + log_n2
        ll = np.logaddexp(l1, l2)
        w = np.exp(l1 - ll)
        grad[pm + pv] = -np.sum((1 - w) / pi - w / (1 - pi)) * 0.5 * sig_a * (1 - sig_a) + 0.02 * a
        grad[pm + pv + 1] = -np.sum((1 - w) * (-1.0 / sg + e * e / sg**3)) * floor_g * math.exp(min(b, 30.0)) \
            + 0.02 * b
        extra_pen = 0.01 * (a * a + b * b)
    else:
        ll, w, extra_pen = log_n1, np.ones_like(e), 0.0
    grad[:pm] = -(Xm.T @ (w * r / s2)) + 2.0 * ridge * pen_m * beta
    grad[pm:pm + pv] = -(Xv.T @ (w * 0.5 * (r * r / s2 - 1.0) * ex / s2)) + 2.0 * ridge * pen_v * gamma
    value = -float(np.sum(ll)) + ridge * float(np.sum(pen_m * beta**2) + np.sum(pen_v * gamma**2)) + extra_pen
    return value, grad


def fit_error_model(e: ArrayLike, Xm: ArrayLike, Xv: ArrayLike, offset: ArrayLike | None = None, *,
                    penalize_m: ArrayLike | None = None, penalize_v: ArrayLike | None = None, ridge: float = 1.0,
                    gross: bool = True, s0: float = 0.005) -> ErrorModel:
    """Fit the structured error model by penalised maximum likelihood (L-BFGS, analytic gradient).

    ``e ~ (1 - pi) N(Xm beta, s0^2 + exp(Xv gamma + offset)) + pi N(0, sigma_g^2)``. The gross scale is kept
    above three robust standard deviations of ``e`` so that the second component stays the wide one, and
    ``pi <= 0.5`` (its logit and the gross scale's log-excess are bounded, so neither runs off to infinity
    when there are no gross errors). Initial values come from a trimmed least-squares fit of the bias and of the log squared
    residuals.

    Parameters
    ----------
    e : array_like, shape (n,)
        Errors (m).
    Xm : array_like, shape (n, pm)
        Mean design matrix.
    Xv : array_like, shape (n, pv)
        Variance design matrix.
    offset : array_like, shape (n,), optional
        Fixed log-variance offset (log m^2), e.g. ``2 log|value|`` for a proportional baseline.
    penalize_m : array_like of bool, shape (pm,), optional
        Which mean coefficients the ridge penalises; default all but the first column.
    penalize_v : array_like of bool, shape (pv,), optional
        Which variance coefficients the ridge penalises; default all but the first column.
    ridge : float, default 1.0
        Penalty weight on the summed log-likelihood scale.
    gross : bool, default True
        Fit the gross-error component.
    s0 : float, default 0.005
        Noise floor (m).

    Returns
    -------
    ErrorModel
        The fitted model.

    Raises
    ------
    ValueError
        If shapes disagree or there are fewer than 3 items.
    """
    e = np.asarray(e, dtype=np.float64)
    Xm, Xv = np.atleast_2d(np.asarray(Xm, dtype=np.float64)), np.atleast_2d(np.asarray(Xv, dtype=np.float64))
    n = len(e)
    if n < 3 or Xm.shape[0] != n or Xv.shape[0] != n:
        raise ValueError(f"Need >= 3 items and matching rows: e {e.shape}, Xm {Xm.shape}, Xv {Xv.shape}")
    offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)
    pm, pv = Xm.shape[1], Xv.shape[1]
    pen_m = (np.arange(pm) > 0).astype(float) if penalize_m is None else np.asarray(penalize_m, dtype=float)
    pen_v = (np.arange(pv) > 0).astype(float) if penalize_v is None else np.asarray(penalize_v, dtype=float)

    med = float(np.median(e))
    scale = max(1.4826 * float(np.median(np.abs(e - med))), s0)
    core = np.abs(e - med) <= 3.0 * scale
    if core.sum() < max(pm, pv) + 2:
        core = np.ones(n, dtype=bool)
    beta0 = np.linalg.lstsq(Xm[core], e[core], rcond=None)[0]
    resid = e - Xm @ beta0
    gamma0 = np.linalg.lstsq(Xv[core], np.log(resid[core] ** 2 + s0**2) - offset[core] + 1.27, rcond=None)[0]
    floor_g = 3.0 * scale
    x0 = np.concatenate([beta0, gamma0, [special.logit(2 * 0.05), 0.0] if gross else []])
    bounds = [(None, None)] * (pm + pv) + ([(-12.0, 12.0), (-10.0, 10.0)] if gross else [])
    result = optimize.minimize(_mixture_objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                               args=(e, Xm, Xv, offset, pen_m, pen_v, ridge, s0, floor_g, gross),
                               options={"maxiter": 10000, "maxfun": 40000})
    params = result.x
    if gross:
        pi = 0.5 * float(special.expit(params[pm + pv]))
        sg = floor_g * (1.0 + math.exp(min(params[pm + pv + 1], 30.0)))
    else:
        pi, sg = 0.0, math.nan
    model = ErrorModel(params[:pm].copy(), params[pm:pm + pv].copy(), pi, sg, s0, math.nan, bool(result.success))
    mu, sigma = model.predict(Xm, Xv, offset)
    nll = float(np.mean(mixture_nll(e, mu, sigma, pi, sg)))
    return ErrorModel(model.beta, model.gamma, pi, sg, s0, nll, model.converged)


def mixture_cdf(x: ArrayLike, mu: ArrayLike, sigma: ArrayLike, pi: ArrayLike, sigma_gross: ArrayLike) -> Array:
    """Predictive CDF of the structured error model, ``(1 - pi) Phi((x - mu) / sigma) + pi Phi(x / sigma_g)``.

    Parameters
    ----------
    x : array_like
        Errors (m).
    mu : array_like
        Predicted bias (m); broadcasts.
    sigma : array_like
        Spread of the non-gross component (m); broadcasts.
    pi : array_like
        Gross-error share; broadcasts. 0 means no gross component.
    sigma_gross : array_like
        Gross-error standard deviation (m); broadcasts; ignored where ``pi == 0``.

    Returns
    -------
    numpy.ndarray
        ``P(e <= x)``.
    """
    x, mu, sigma, pi, sg = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64)
                                                 for v in (x, mu, sigma, pi, sigma_gross)))
    gross = np.where(pi > 0, stats.norm.cdf(x / np.where(pi > 0, sg, 1.0)), 0.0)
    return (1 - pi) * stats.norm.cdf((x - mu) / sigma) + pi * gross


def mixture_quantile(q: ArrayLike, mu: ArrayLike, sigma: ArrayLike, pi: ArrayLike, sigma_gross: ArrayLike,
                     n_iter: int = 60) -> Array:
    """Quantiles of the structured error model's predictive distribution, by bisection.

    Parameters
    ----------
    q : array_like
        Probabilities in (0, 1).
    mu : array_like
        Predicted bias (m); broadcasts.
    sigma : array_like
        Spread of the non-gross component (m); broadcasts.
    pi : array_like
        Gross-error share; broadcasts.
    sigma_gross : array_like
        Gross-error standard deviation (m); broadcasts.
    n_iter : int, default 60
        Bisection steps (each halves the bracket).

    Returns
    -------
    numpy.ndarray
        ``x`` with ``mixture_cdf(x) = q`` (m).
    """
    q, mu, sigma, pi, sg = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64)
                                                 for v in (q, mu, sigma, pi, sigma_gross)))
    width = 12.0 * np.maximum(sigma, np.where(pi > 0, np.nan_to_num(sg, nan=0.0), 0.0)) + np.abs(mu)
    lo, hi = mu - width, mu + width
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        below = mixture_cdf(mid, mu, sigma, pi, sg) < q
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def mixture_nll(e: ArrayLike, mu: ArrayLike, sigma: ArrayLike, pi: ArrayLike, sigma_gross: ArrayLike) -> Array:
    """Per-item negative log-likelihood of errors under the structured model's predictive density.

    Parameters
    ----------
    e : array_like, shape (n,)
        Errors (m).
    mu : array_like, shape (n,)
        Predicted bias (m).
    sigma : array_like, shape (n,)
        Predicted spread of the non-gross component (m).
    pi : array_like
        Gross-error share; broadcasts. 0 means no gross component.
    sigma_gross : array_like
        Gross-error standard deviation (m); broadcasts; ignored where ``pi == 0``.

    Returns
    -------
    numpy.ndarray, shape (n,)
        ``-log p(e)`` in nats (densities per metre).
    """
    e, mu, sigma, pi, sg = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64)
                                                 for v in (e, mu, sigma, pi, sigma_gross)))
    log_n1 = stats.norm.logpdf(e, mu, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_n2 = np.where(pi > 0, stats.norm.logpdf(e, 0.0, np.where(pi > 0, sg, 1.0)), -np.inf)
        ll = np.logaddexp(np.log1p(-pi) + log_n1, np.log(pi) + log_n2)
    return -ll


@dataclass(frozen=True)
class Standardizer:
    """Feature standardisation and depth-source one-hot columns, learned on a training fold.

    Attributes
    ----------
    names : tuple of str
        Feature columns.
    mean : numpy.ndarray, shape (len(names),)
        Training means (NaN-ignoring); missing values are imputed with them.
    scale : numpy.ndarray, shape (len(names),)
        Training standard deviations (1 where a column is constant or empty).
    sources : tuple of str
        Depth sources of the whole dataset; every source but the first gets a one-hot column. A source unseen
        in training has an all-zero column, so its coefficient stays at 0 under the ridge.
    """

    names: tuple[str, ...]
    mean: Array
    scale: Array
    sources: tuple[str, ...]

    @staticmethod
    def fit(rows: pd.DataFrame, names: Sequence[str], sources: Sequence[str]) -> Standardizer:
        """Learn means and scales from training rows.

        Parameters
        ----------
        rows : pandas.DataFrame
            Training rows with the feature columns.
        names : sequence of str
            Feature columns to standardise.
        sources : sequence of str
            All depth sources of the dataset, in a fixed order; empty for no source columns.

        Returns
        -------
        Standardizer
            The fitted transform.
        """
        values = rows[list(names)].to_numpy(dtype=np.float64) if names else np.zeros((len(rows), 0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(values, axis=0) if len(values) else np.zeros(len(names))
            scale = np.nanstd(values, axis=0) if len(values) else np.ones(len(names))
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
        return Standardizer(tuple(names), mean, scale, tuple(sources))

    def transform(self, rows: pd.DataFrame) -> Array:
        """Design matrix: an intercept, standardised features (missing as 0) and source one-hots.

        Parameters
        ----------
        rows : pandas.DataFrame
            Rows with the feature columns and ``depth_source``.

        Returns
        -------
        numpy.ndarray, shape (n, 1 + len(names) + max(len(sources) - 1, 0))
            The design matrix.
        """
        n = len(rows)
        if self.names:
            z = (rows[list(self.names)].to_numpy(dtype=np.float64) - self.mean) / self.scale
            z = np.where(np.isfinite(z), z, 0.0)
        else:
            z = np.zeros((n, 0))
        src = rows["depth_source"].to_numpy() if self.sources else np.array([])
        onehot = np.stack([(src == s).astype(float) for s in self.sources[1:]], axis=1) \
            if len(self.sources) > 1 else np.zeros((n, 0))
        return np.concatenate([np.ones((n, 1)), z, onehot], axis=1)

    def columns(self) -> list[str]:
        """Names of the design columns, in :meth:`transform` order.

        Returns
        -------
        list of str
            ``"intercept"``, the feature names, then ``"source=<name>"`` per one-hot column.
        """
        return ["intercept", *self.names, *[f"source={s}" for s in self.sources[1:]]]


@dataclass(frozen=True)
class NumericPredictor:
    """A fitted numeric error model of one kind, with its design.

    Attributes
    ----------
    kind : str
        One of :data:`NUMERIC_KINDS`.
    standardizer : Standardizer
        The design transform (``sigma`` kind only uses features).
    model : ErrorModel
        The fitted model.
    ridge : float
        The ridge weight used (NaN for the unpenalised baselines).
    """

    kind: str
    standardizer: Standardizer
    model: ErrorModel
    ridge: float = math.nan

    @staticmethod
    def design(rows: pd.DataFrame, kind: str, standardizer: Standardizer) -> tuple[Array, Array, Array]:
        """Mean design, variance design and log-variance offset of one model kind.

        ``sigma``: both designs are the full standardised features plus source one-hots. ``const``: an
        intercept only. ``prop``: the bias is ``b * value`` and the log-variance ``c + 2 log|value|``, so
        ``sigma = exp(c / 2) |value|``.

        Parameters
        ----------
        rows : pandas.DataFrame
            Rows with features, ``depth_source`` and (for ``prop``) ``value``.
        kind : str
            One of :data:`NUMERIC_KINDS`.
        standardizer : Standardizer
            Fitted transform for the ``sigma`` kind.

        Returns
        -------
        Xm : numpy.ndarray, shape (n, pm)
            Mean design.
        Xv : numpy.ndarray, shape (n, pv)
            Variance design.
        offset : numpy.ndarray, shape (n,)
            Log-variance offset (log m^2).

        Raises
        ------
        ValueError
            For an unknown kind.
        """
        n = len(rows)
        if kind == "sigma":
            X = standardizer.transform(rows)
            return X, X, np.zeros(n)
        if kind == "const":
            return np.ones((n, 1)), np.ones((n, 1)), np.zeros(n)
        if kind == "prop":
            value = rows["value"].to_numpy(dtype=np.float64)
            return value[:, None], np.ones((n, 1)), 2.0 * np.log(np.maximum(np.abs(value), _MIN_VALUE))
        raise ValueError(f"Unknown numeric model kind {kind!r}")

    @staticmethod
    def fit(rows: pd.DataFrame, kind: str, sources: Sequence[str], config: AuditConfig,
            ridge: float | None = None) -> NumericPredictor:
        """Fit one model kind on training rows.

        Parameters
        ----------
        rows : pandas.DataFrame
            Labelled numeric rows (``e`` finite) with a ``scene`` column.
        kind : str
            One of :data:`NUMERIC_KINDS`.
        sources : sequence of str
            All depth sources of the dataset.
        config : AuditConfig
            Ridge grid, its cross-validation folds (``max_folds``, ``seed``), noise floor and gross-component
            settings.
        ridge : float, optional
            Fixed ridge weight for the ``sigma`` kind; by default chosen from ``config.ridge_grid`` by
            :func:`select_ridge` on ``rows``. The baselines have no penalised coefficients.

        Returns
        -------
        NumericPredictor
            The fitted predictor.
        """
        if kind == "sigma" and ridge is None:
            ridge = select_ridge(rows, lambda r, lam: NumericPredictor.fit(r, kind, sources, config, lam),
                                 lambda m, r: float(np.sum(m.nll(r))), config.ridge_grid,
                                 max_folds=config.max_folds, seed=config.seed)
        standardizer = Standardizer.fit(rows, FEATURES if kind == "sigma" else (), sources if kind == "sigma"
                                        else ())
        Xm, Xv, offset = NumericPredictor.design(rows, kind, standardizer)
        pen_m = np.zeros(Xm.shape[1]) if kind == "prop" else None
        model = fit_error_model(rows["e"].to_numpy(dtype=np.float64), Xm, Xv, offset, penalize_m=pen_m,
                                ridge=0.0 if ridge is None else ridge, gross=config.gross, s0=config.s0)
        return NumericPredictor(kind, standardizer, model, math.nan if ridge is None else float(ridge))

    def nll(self, rows: pd.DataFrame) -> Array:
        """Negative log-likelihood of the rows' errors under the predicted distribution.

        Parameters
        ----------
        rows : pandas.DataFrame
            Rows with the model's inputs and ``e``.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Nats per row.
        """
        pred = self.predict(rows)
        return mixture_nll(rows["e"], pred["mu"], pred["sigma"], pred["pi"], pred["sigma_g"])

    def predict(self, rows: pd.DataFrame) -> dict[str, Array]:
        """Predict the error distribution of rows.

        Parameters
        ----------
        rows : pandas.DataFrame
            Rows with the model's inputs.

        Returns
        -------
        dict of str to numpy.ndarray
            ``mu`` and ``sigma`` (m), plus the constant ``pi``, ``sigma_g`` (m) and ``ridge``, each shape (n,).
        """
        mu, sigma = self.model.predict(*NumericPredictor.design(rows, self.kind, self.standardizer))
        n = len(rows)
        return {"mu": mu, "sigma": sigma, "pi": np.full(n, self.model.pi_gross),
                "sigma_g": np.full(n, self.model.sigma_gross), "ridge": np.full(n, self.ridge)}


def fit_logistic(y: ArrayLike, X: ArrayLike, *, ridge: float = 1.0, penalize: ArrayLike | None = None) -> Array:
    """Ridge-penalised logistic regression by L-BFGS.

    Parameters
    ----------
    y : array_like, shape (n,)
        Outcomes in {0, 1}.
    X : array_like, shape (n, p)
        Design matrix, first column the intercept.
    ridge : float, default 1.0
        Penalty weight on the summed log-likelihood scale. The intercept gets ``1e-3`` of it, which keeps the
        fit finite when every outcome is the same.
    penalize : array_like, shape (p,), optional
        Per-coefficient penalty multipliers; default ``1e-3`` for the intercept and 1 for the rest.

    Returns
    -------
    numpy.ndarray, shape (p,)
        Coefficients (log-odds per design unit).
    """
    y, X = np.asarray(y, dtype=np.float64), np.atleast_2d(np.asarray(X, dtype=np.float64))
    pen = np.r_[1e-3, np.ones(X.shape[1] - 1)] if penalize is None else np.asarray(penalize, dtype=np.float64)

    def objective(w: Array) -> tuple[float, Array]:
        """Penalised negative log-likelihood and gradient.

        Parameters
        ----------
        w : numpy.ndarray, shape (p,)
            Coefficients.

        Returns
        -------
        value : float
            The objective.
        gradient : numpy.ndarray, shape (p,)
            Its gradient.
        """
        z = X @ w
        value = float(np.sum(np.logaddexp(0.0, z) - y * z) + ridge * np.sum(pen * w * w))
        return value, X.T @ (special.expit(z) - y) + 2.0 * ridge * pen * w

    p0 = np.clip(y.mean(), 0.02, 0.98)
    w0 = np.zeros(X.shape[1])
    w0[0] = special.logit(p0)
    return optimize.minimize(objective, w0, jac=True, method="L-BFGS-B", options={"maxiter": 500}).x


@dataclass(frozen=True)
class LogisticPredictor:
    """A fitted categorical p(correct) model of one kind.

    Attributes
    ----------
    kind : str
        One of :data:`CATEGORICAL_KINDS`: ``logistic`` (all :data:`CATEGORICAL_FEATURES` and the depth
        source), ``margin`` (log margin and log range only: the geometric ``Phi(margin / sigma(range))`` idea in
        logistic form) or ``const``.
    standardizer : Standardizer
        The design transform.
    w : numpy.ndarray
        Coefficients.
    ridge : float
        The ridge weight used.
    """

    kind: str
    standardizer: Standardizer
    w: Array
    ridge: float = math.nan

    @staticmethod
    def fit(rows: pd.DataFrame, kind: str, sources: Sequence[str], config: AuditConfig,
            ridge: float | None = None) -> LogisticPredictor:
        """Fit P(label correct) on training rows.

        Parameters
        ----------
        rows : pandas.DataFrame
            Labelled categorical rows with ``wrong`` and ``scene``.
        kind : str
            One of :data:`CATEGORICAL_KINDS`.
        sources : sequence of str
            All depth sources of the dataset.
        config : AuditConfig
            Supplies the ridge grid and its cross-validation folds (``max_folds``, ``seed``).
        ridge : float, optional
            Fixed ridge weight; by default chosen by :func:`select_ridge` (the ``const`` kind uses the smallest
            grid value, which only touches its intercept).

        Returns
        -------
        LogisticPredictor
            The fitted predictor.

        Raises
        ------
        ValueError
            For an unknown kind.
        """
        if kind == "logistic":
            standardizer = Standardizer.fit(rows, CATEGORICAL_FEATURES, sources)
        elif kind == "margin":
            standardizer = Standardizer.fit(rows, ("log_margin", "log_range"), ())
        elif kind == "const":
            standardizer = Standardizer.fit(rows, (), ())
        else:
            raise ValueError(f"Unknown categorical model kind {kind!r}")
        if ridge is None:
            ridge = min(config.ridge_grid) if kind == "const" else select_ridge(
                rows, lambda r, lam: LogisticPredictor.fit(r, kind, sources, config, lam),
                lambda m, r: float(np.sum(m.nll(r))), config.ridge_grid, max_folds=config.max_folds,
                seed=config.seed)
        y = 1.0 - rows["wrong"].to_numpy(dtype=np.float64)
        return LogisticPredictor(kind, standardizer, fit_logistic(y, standardizer.transform(rows), ridge=ridge),
                                 float(ridge))

    def nll(self, rows: pd.DataFrame) -> Array:
        """Bernoulli negative log-likelihood of the rows' correctness.

        Parameters
        ----------
        rows : pandas.DataFrame
            Rows with the model's inputs and ``wrong``.

        Returns
        -------
        numpy.ndarray, shape (n,)
            Nats per row.
        """
        z = self.standardizer.transform(rows) @ self.w
        y = 1.0 - rows["wrong"].to_numpy(dtype=np.float64)
        return np.logaddexp(0.0, z) - y * z

    def predict(self, rows: pd.DataFrame) -> dict[str, Array]:
        """Predict P(label correct).

        Parameters
        ----------
        rows : pandas.DataFrame
            Rows with the model's inputs.

        Returns
        -------
        dict of str to numpy.ndarray
            ``p``: probability that the auto label is correct, and ``ridge``, each shape (n,).
        """
        return {"p": special.expit(self.standardizer.transform(rows) @ self.w), "ridge": np.full(len(rows), self.ridge)}


def select_ridge(rows: pd.DataFrame, fit: Callable[[pd.DataFrame, float], T], loss: Callable[[T, pd.DataFrame], float],
                 grid: Sequence[float], *, max_folds: int | None = None, seed: int = 0) -> float:
    """Choose a ridge weight by leave-one-scene-out cross-validation within the given rows.

    With more than ``max_folds`` scenes in ``rows``, the scenes are grouped into ``max_folds`` folds
    (:func:`scene_folds`) and each fold is held out in turn instead: grouped K-fold cross-validation, which
    still never splits a scene between training and held-out rows.

    Parameters
    ----------
    rows : pandas.DataFrame
        Training rows with a ``scene`` column (never the scene being predicted or calibrated).
    fit : callable
        ``fit(train_rows, ridge) -> model``.
    loss : callable
        ``loss(model, held_out_rows) -> summed loss`` (held-out negative log-likelihood).
    grid : sequence of float
        Candidate weights.
    max_folds : int, optional
        Most folds; None holds out every scene on its own however many there are.
    seed : int, default 0
        Seed of the fold assignment's tie-breaks.

    Returns
    -------
    float
        The candidate with the smallest summed held-out loss (ties: the larger weight). The single candidate
        if the grid has one; the largest candidate if the rows come from fewer than 2 scenes.
    """
    grid = sorted(grid)
    if len(grid) == 1:
        return float(grid[0])
    fold_of = scene_folds(rows["scene"], max_folds, seed)
    if len(fold_of) < 2:
        return float(grid[-1])
    row_fold = rows["scene"].map(fold_of).to_numpy()
    totals = np.zeros(len(grid))
    for f in range(max(fold_of.values()) + 1):
        train, held = rows[row_fold != f], rows[row_fold == f]
        for j, ridge in enumerate(grid):
            totals[j] += loss(fit(train, ridge), held)
    best = np.flatnonzero(totals <= totals.min() + 1e-9)
    return float(grid[best[-1]])


def _cross_fit_task(task: tuple[pd.DataFrame, str, bool, list[str], AuditConfig]) -> tuple[pd.DataFrame,
                                                                                            pd.DataFrame]:
    """Cross-fit one model kind for one dataset and question type (a picklable unit of parallel work).

    Parameters
    ----------
    task : tuple
        ``(rows, kind, numeric, sources, config)``: the labelled rows, the model kind, whether the type is
        numeric, the dataset's depth sources and the settings (``max_folds`` and ``seed`` set the folds, or the
        dataset's explicit ``fold_groups`` do).

    Returns
    -------
    test, cal : pandas.DataFrame
        :func:`cross_fit` outputs, with a ``kind`` column.
    """
    rows, kind, numeric, sources, config = task
    predictor: Any = NumericPredictor if numeric else LogisticPredictor
    test, cal = cross_fit(rows, lambda r: predictor.fit(r, kind, sources, config), lambda m, r: m.predict(r),
                          max_folds=config.max_folds, seed=config.seed,
                          groups=config.groups_of(rows["dataset"].iloc[0]))
    return test.assign(kind=kind), cal.assign(kind=kind)


def _run_tasks(tasks: list[tuple[pd.DataFrame, str, bool, list[str], AuditConfig]], n_jobs: int,
               say: Callable[[str], None]) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Run cross-fitting tasks, in worker processes if ``n_jobs > 1`` (one BLAS thread each).

    Parameters
    ----------
    tasks : list of tuple
        Arguments of :func:`_cross_fit_task`.
    n_jobs : int
        Worker processes.
    say : callable
        Progress messages.

    Returns
    -------
    list of tuple
        Results in task order.
    """
    def label(task: tuple[pd.DataFrame, str, bool, list[str], AuditConfig]) -> str:
        """Describe a task.

        Parameters
        ----------
        task : tuple
            Arguments of :func:`_cross_fit_task`.

        Returns
        -------
        str
            ``"<dataset>/<qtype>: <kind> (<rows> rows, <scenes> scenes)"``.
        """
        rows = task[0]
        return (f"{rows['dataset'].iloc[0]}/{rows['qtype'].iloc[0]}: {task[1]} ({len(rows)} rows, "
                f"{rows['scene'].nunique()} scenes)")

    if n_jobs <= 1:
        out = []
        for task in tasks:
            say(f"cross-fitting {label(task)}")
            out.append(_cross_fit_task(task))
        return out
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ.update({k: "1" for k in saved})
    try:
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_cross_fit_task, task) for task in tasks]
            out = []
            for task, future in zip(tasks, futures):
                out.append(future.result())
                say(f"cross-fitted {label(task)}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


# ============================================================================================================
# 4. Out-of-scene predictions
# ============================================================================================================


def scene_folds(scenes: ArrayLike, max_folds: int | None, seed: int = 0,
                groups: Mapping[Any, Any] | None = None) -> dict[Any, int]:
    """Assign scenes to cross-fitting folds: one fold per scene, or ``max_folds`` folds balanced by row count.

    With at most ``max_folds`` distinct scenes (or ``max_folds`` None), scene ``i`` in sorted order is fold
    ``i``: leave one scene out. With more, scenes are taken in decreasing row count (ties in a random order
    drawn from ``seed``) and each goes to the fold with the fewest rows so far (ties: fewer scenes, then the
    lower fold number), so fold sizes in rows differ by at most the largest scene's count. Folds are then
    numbered by their first scene in sorted order. The assignment depends on the scene ids, their row counts
    and the seed only: never on errors, labels or features.

    With explicit ``groups``, each group of the scenes present is one fold, whatever ``max_folds`` and
    ``seed``, numbered in sorted order of each group's first scene (leave one group out).

    Parameters
    ----------
    scenes : array_like
        The scene of every row (each scene weighs as many rows as it has).
    max_folds : int or None
        Most folds; None gives every scene its own fold. Ignored with ``groups``.
    seed : int, default 0
        Seed of the tie-breaks between scenes with equal row counts. Ignored with ``groups``.
    groups : mapping, optional
        Scene to group label (:attr:`AuditConfig.fold_groups` of one dataset). Every scene in ``scenes`` must be
        mapped; mapped scenes absent from ``scenes`` are ignored.

    Returns
    -------
    dict
        Scene to fold number in ``0 .. K - 1``, in sorted scene order; ``K`` is the number of scenes or
        ``max_folds``, whichever is smaller (with ``groups``: the number of groups among the scenes).

    Raises
    ------
    ValueError
        If ``groups`` is given and some scene has no group.
    """
    counts = pd.Series(np.asarray(scenes)).value_counts()
    names = sorted(counts.index)
    if groups is not None:
        missing = [s for s in names if s not in groups]
        if missing:
            raise ValueError(f"{len(missing)} of {len(names)} scenes have no fold group: {missing[:5]}"
                             + (" ..." if len(missing) > 5 else ""))
        group_number: dict[Any, int] = {}
        for s in names:                                   # sorted: first sight is the group's first scene
            group_number.setdefault(groups[s], len(group_number))
        return {s: group_number[groups[s]] for s in names}
    if max_folds is None or len(names) <= max_folds:
        return {s: i for i, s in enumerate(names)}
    n_rows = counts.reindex(names).to_numpy(dtype=np.int64)
    tie = np.random.default_rng(seed).permutation(len(names))
    order = np.lexsort((tie, -n_rows))                    # most rows first; equal counts in a seeded order
    load, size = np.zeros(max_folds, dtype=np.int64), np.zeros(max_folds, dtype=np.int64)
    fold = np.empty(len(names), dtype=np.int64)
    for i in order:
        k = min(range(max_folds), key=lambda j: (load[j], size[j], j))
        fold[i], load[k], size[k] = k, load[k] + n_rows[i], size[k] + 1
    number: dict[int, int] = {}
    for k in fold:                                        # names are sorted: first sight is the fold's first scene
        number.setdefault(int(k), len(number))
    return {s: number[int(k)] for s, k in zip(names, fold)}


def fold_table(rows: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """The fold assignment :func:`cross_fit` uses for one dataset and question type (to record it).

    Parameters
    ----------
    rows : pandas.DataFrame
        The modelled rows of one dataset and question type (``dataset``, ``qtype``, ``scene``).
    config : AuditConfig
        ``max_folds`` and ``seed``, or the dataset's explicit ``fold_groups``.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, qtype, scene, fold, n_rows, fold_rows, fold_scenes``, one row per scene in sorted
        order: its fold (:func:`scene_folds`), its rows, and its fold's rows and scenes. With explicit fold
        groups for the dataset, a ``group`` column (the scene's group label) follows ``fold``.
    """
    groups = config.groups_of(rows["dataset"].iloc[0])
    fold_of = scene_folds(rows["scene"], config.max_folds, config.seed, groups)
    n_rows = rows["scene"].value_counts()
    table = pd.DataFrame({"dataset": rows["dataset"].iloc[0], "qtype": rows["qtype"].iloc[0],
                          "scene": list(fold_of), "fold": list(fold_of.values()),
                          "n_rows": [int(n_rows[s]) for s in fold_of]})
    if groups is not None:
        table.insert(table.columns.get_loc("fold") + 1, "group", [groups[s] for s in fold_of])
    by_fold = table.groupby("fold")
    return table.assign(fold_rows=by_fold["n_rows"].transform("sum"), fold_scenes=by_fold["scene"].transform("size"))


def cross_fit(rows: pd.DataFrame, fit: Callable[[pd.DataFrame], T],
              predict: Callable[[T, pd.DataFrame], dict[str, Array]], *, max_folds: int | None = None,
              seed: int = 0, groups: Mapping[Any, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-scene-out test predictions and leave-two-scenes-out calibration predictions.

    For each scene ``S`` a model fitted on all other scenes predicts ``S`` (the test predictions). For each pair
    of scenes ``{S, S'}`` a model fitted without both predicts ``S'`` as calibration data for test scene ``S``,
    and ``S`` as calibration data for ``S'``. So no calibration score for ``S`` comes from a model that saw
    ``S``, and none comes from the scene being calibrated.

    With more than ``max_folds`` scenes, the scenes are grouped into ``max_folds`` folds (:func:`scene_folds`)
    and the same holds at fold level: a model fitted without fold ``F`` predicts every scene of ``F``, and for
    each pair of folds ``{F, G}`` a model fitted without both predicts each scene of ``G`` as calibration data
    for every test scene of ``F`` (the predictions repeated once per test scene), and vice versa. The output
    keeps its form, one calibration set per test scene, and the guarantee: no calibration score for ``S`` comes
    from a model that saw ``S`` or the scene being calibrated. Scenes of ``S``'s own fold do not calibrate
    ``S``. Fits: ``K + K (K - 1) / 2`` for ``K`` folds.

    With explicit ``groups``, each group is one fold whatever the number of scenes (:func:`scene_folds`), and the
    same fold-level scheme gives leave-one-group-out test predictions and leave-two-groups-out calibration: no
    model that predicts or calibrates ``S`` saw a scene of ``S``'s group, and no scene of ``S``'s group
    calibrates ``S``.

    Parameters
    ----------
    rows : pandas.DataFrame
        Rows of one dataset and question type, with a ``scene`` column; the index identifies rows.
    fit : callable
        ``fit(train_rows) -> model``.
    predict : callable
        ``predict(model, rows) -> {name: array of shape (n,)}``.
    max_folds : int, optional
        Most folds; None (the default) leaves every scene out on its own however many there are.
    seed : int, default 0
        Seed of the fold assignment's tie-breaks.
    groups : mapping, optional
        Scene to group label: one fold per group, ignoring ``max_folds`` and ``seed``; every scene must be mapped.

    Returns
    -------
    test : pandas.DataFrame
        Columns ``row`` (index into ``rows``) and the predicted arrays; one row per input row, scenes in sorted
        order.
    cal : pandas.DataFrame
        Columns ``test_scene``, ``row`` and the predicted arrays; each row appears once per scene outside its
        fold (once per other scene when every scene is its own fold).

    Raises
    ------
    ValueError
        If there are fewer than 3 scenes or 3 folds, or (with ``groups``) a scene has no group.
    """
    fold_of = scene_folds(rows["scene"], max_folds, seed, groups)
    scenes = list(fold_of)
    if len(scenes) < 3:
        raise ValueError(f"Leave-two-scenes-out needs >= 3 scenes, got {len(scenes)}")
    if max(fold_of.values()) + 1 < 3:
        raise ValueError(f"Leave-two-folds-out needs >= 3 folds, got {max(fold_of.values()) + 1}")
    members = [[s for s in scenes if fold_of[s] == f] for f in range(max(fold_of.values()) + 1)]
    by_scene = {s: rows[rows["scene"] == s] for s in scenes}
    row_fold = rows["scene"].map(fold_of).to_numpy()
    test_parts: dict[Any, pd.DataFrame] = {}
    cal_parts = []
    for f, held in enumerate(members):
        model = fit(rows[row_fold != f])
        for s in held:
            test_parts[s] = pd.DataFrame({"row": by_scene[s].index, **predict(model, by_scene[s])})
    for f, g in combinations(range(len(members)), 2):
        model = fit(rows[(row_fold != f) & (row_fold != g)])
        for test_fold, cal_fold in ((f, g), (g, f)):
            for cal_scene in members[cal_fold]:
                part = by_scene[cal_scene]
                pred = predict(model, part)
                cal_parts += [pd.DataFrame({"test_scene": test_scene, "row": part.index, **pred})
                              for test_scene in members[test_fold]]
    return pd.concat([test_parts[s] for s in scenes], ignore_index=True), pd.concat(cal_parts, ignore_index=True)


# ============================================================================================================
# 5. Conformal calibration, coverage and rank correlation
# ============================================================================================================


def conformal_quantile(scores: ArrayLike, level: float) -> float:
    """Split-conformal quantile: the ``ceil((n + 1) level)``-th smallest calibration score.

    Parameters
    ----------
    scores : array_like, shape (n,)
        Calibration nonconformity scores (NaN ignored).
    level : float
        Nominal coverage in (0, 1).

    Returns
    -------
    float
        The quantile; ``inf`` when ``n`` is too small for the level (``ceil((n + 1) level) > n``).
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    s = s[~np.isnan(s)]
    k = math.ceil((len(s) + 1) * level)
    return float(s[k - 1]) if 1 <= k <= len(s) else math.inf


def conformal_calibrate(test: pd.DataFrame, cal: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Mondrian split-conformal quantiles of ``|e - mu| / sigma`` per test scene, and the resulting intervals.

    Parameters
    ----------
    test : pandas.DataFrame
        Test predictions joined with their rows: ``dataset, kind, scene, e, mu, sigma`` and the
        ``config.mondrian`` columns.
    cal : pandas.DataFrame
        Calibration predictions joined likewise, with ``test_scene``.
    config : AuditConfig
        Levels, cells and the fallback size.

    Returns
    -------
    pandas.DataFrame
        ``test`` plus, per level ``L`` (as ``int(100 L)``): ``q<L>`` (score quantile), ``half<L>`` (interval
        half-width ``q sigma``, m) and ``cov<L>`` (``|e - mu| <= q sigma``); plus ``n_cal`` and ``cal_cell``
        (``"full"`` or ``"pooled"`` when the cell fell back to the question type alone).
    """
    cal = cal.assign(score=np.abs(cal["e"] - cal["mu"]) / cal["sigma"])
    full_key = ["dataset", "kind", "test_scene", *config.mondrian]
    pooled_key = ["dataset", "kind", "test_scene", config.mondrian[0]]
    full = {k: g["score"].to_numpy() for k, g in cal.groupby(full_key, sort=False)}
    pooled = {k: g["score"].to_numpy() for k, g in cal.groupby(pooled_key, sort=False)}
    out = test.copy()
    q = np.full((len(out), len(config.levels)), np.nan)
    n_cal = np.zeros(len(out), dtype=int)
    cell = np.empty(len(out), dtype=object)
    test_key = ["dataset", "kind", "scene", *config.mondrian]
    for key, idx in out.groupby(test_key, sort=False).indices.items():
        scores = full.get(key, np.array([]))
        label = "full"
        if len(scores) < config.min_cal:
            scores = pooled.get(key[:4], np.array([]))
            label = "pooled"
        q[idx] = [conformal_quantile(scores, level) for level in config.levels]
        n_cal[idx], cell[idx] = len(scores), label
    resid = np.abs(out["e"].to_numpy(float) - out["mu"].to_numpy(float))
    sigma = out["sigma"].to_numpy(float)
    for j, level in enumerate(config.levels):
        tag = _tag(level)
        out[f"q{tag}"] = q[:, j]
        out[f"half{tag}"] = q[:, j] * sigma
        out[f"cov{tag}"] = resid <= q[:, j] * sigma
    out["n_cal"], out["cal_cell"] = n_cal, cell
    return out


def conformal_quantile_table(test: pd.DataFrame, cal: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Every test scene's conformal quantiles per Mondrian cell, as :func:`conformal_calibrate` computes them.

    :func:`conformal_calibrate` takes a quantile only for the cells its test scene has answers in. This table
    takes it, with the same rule, for every cell a test scene could need: each calibration group (dataset, model
    kind, test scene and the first Mondrian column, i.e. the question type) crossed with every value of the
    other Mondrian columns (the depth sources) seen in its dataset, plus every test cell. So a calibration item
    of any cell can be given the half-width that its test scene's own answers of that cell get (``q * sigma``).
    A cell with fewer than ``min_cal`` calibration scores falls back to its pooled group, as in
    :func:`conformal_calibrate`; a group without scores gets ``inf``. For a test row's cell the quantiles are
    those of :func:`conformal_calibrate` exactly (the same scores, the same rule).

    Parameters
    ----------
    test : pandas.DataFrame
        Test predictions joined with rows: ``dataset, kind, scene`` and the ``config.mondrian`` columns.
    cal : pandas.DataFrame
        Calibration predictions joined likewise, with ``test_scene``, ``e``, ``mu`` and ``sigma``.
    config : AuditConfig
        Levels, cells and the fallback size.

    Returns
    -------
    pandas.DataFrame
        One row per cell: ``dataset, kind, test_scene``, the ``config.mondrian`` columns, ``n_cal`` (scores
        used), ``cal_cell`` (``"full"`` or ``"pooled"``) and, per level ``L``, ``q<L>`` (the score quantile,
        float64; the interval half-width is ``q<L> * sigma``, m). Sorted by its key columns.
    """
    mondrian = list(config.mondrian)
    pooled_cols, other = ["dataset", "kind", "test_scene", mondrian[0]], mondrian[1:]
    full_key = ["dataset", "kind", "test_scene", *mondrian]
    columns = [*full_key, "n_cal", "cal_cell", *[f"q{_tag(level)}" for level in config.levels]]
    cal = cal.assign(score=np.abs(cal["e"] - cal["mu"]) / cal["sigma"])
    full = {k: g["score"].to_numpy() for k, g in cal.groupby(full_key, sort=False)}
    pooled = {k: g["score"].to_numpy() for k, g in cal.groupby(pooled_cols, sort=False)}
    test_cells = test.rename(columns={"scene": "test_scene"})[full_key]
    cells: set[tuple[Any, ...]] = set(map(tuple, test_cells.drop_duplicates().to_numpy().tolist()))
    sources = pd.concat([cal[["dataset", *other]], test_cells[["dataset", *other]]]).drop_duplicates()
    combos = {d: list(map(tuple, g[other].to_numpy().tolist())) for d, g in sources.groupby("dataset", sort=False)}
    groups = set(pooled) | {tuple(k) for k in test_cells[pooled_cols].drop_duplicates().to_numpy().tolist()}
    for group in groups:
        cells.update((*group, *combo) for combo in combos.get(group[0], [()]))
    rows = []
    for key in cells:
        scores, label = full.get(key, np.array([])), "full"
        if len(scores) < config.min_cal:
            scores, label = pooled.get(key[:4], np.array([])), "pooled"
        rows.append([*key, len(scores), label, *[conformal_quantile(scores, level) for level in config.levels]])
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values(full_key, kind="stable").reset_index(drop=True)


def _tag(level: float) -> str:
    """Column suffix of a coverage level.

    Parameters
    ----------
    level : float
        Nominal level, e.g. 0.9.

    Returns
    -------
    str
        The level in percent, e.g. ``"90"``.
    """
    return str(int(round(100 * level)))


def _terciles(values: pd.Series, groups: pd.DataFrame) -> pd.Series:
    """Tercile label of each value within its group.

    Parameters
    ----------
    values : pandas.Series
        Values to bin.
    groups : pandas.DataFrame
        Grouping columns, aligned with ``values``.

    Returns
    -------
    pandas.Series
        ``"low"``, ``"mid"`` or ``"high"`` (by within-group rank).
    """
    ranks = values.groupby([groups[c] for c in groups.columns]).rank(pct=True, method="first")
    return pd.cut(ranks, [0, 1 / 3, 2 / 3, 1.0], labels=["low", "mid", "high"]).astype(str)


def coverage_long(calibrated: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Coverage outcomes in long form: one row per (test item, method, level).

    Methods: each conformal kind, plus ``sigma_raw`` and ``const_raw``: the model's own predictive distribution
    (the bias, spread and gross-error mixture) with no conformal step, as its equal-tailed interval, i.e. a
    probability-integral-transform check that the predictive CDF at the observed error lies in
    ``[(1 - L) / 2, (1 + L) / 2]`` (is the fitted model calibrated as is?). Every row carries the feature model's
    sigma tercile (within dataset and question type) as ``sigma_tercile``.

    Parameters
    ----------
    calibrated : pandas.DataFrame
        Output of :func:`conformal_calibrate` for all kinds.
    config : AuditConfig
        Levels.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, scene, qtype, depth_source, method, level, covered, half_over_tau, sigma_tercile``.
    """
    base = ["dataset", "scene", "qtype", "depth_source", "row"]
    sig = calibrated[calibrated["kind"] == "sigma"]
    tercile = pd.Series(_terciles(sig["sigma"], sig[["dataset", "qtype"]]).to_numpy(), index=sig["row"])
    parts = []
    for level in config.levels:
        tag = _tag(level)
        for kind, g in calibrated.groupby("kind", sort=False):
            parts.append(g[base].assign(method=kind, level=level, covered=g[f"cov{tag}"].to_numpy(),
                                        half_over_tau=(g[f"half{tag}"] / g["tau_hat"]).to_numpy()))
        for kind in ("sigma", "const"):
            g = calibrated[calibrated["kind"] == kind]
            args = [g[c].to_numpy(float) for c in ("mu", "sigma", "pi", "sigma_g")]
            pit = mixture_cdf(g["e"].to_numpy(float), *args)
            lo_q, hi_q = (mixture_quantile(np.full(len(g), q), *args) for q in ((1 - level) / 2, (1 + level) / 2))
            parts.append(g[base].assign(method=f"{kind}_raw", level=level,
                                        covered=(pit >= (1 - level) / 2) & (pit <= (1 + level) / 2),
                                        half_over_tau=0.5 * (hi_q - lo_q) / g["tau_hat"].to_numpy(float)))
    long = pd.concat(parts, ignore_index=True)
    long["sigma_tercile"] = long["row"].map(tercile)
    return long


def cluster_interval(values: ArrayLike, clusters: ArrayLike, config: AuditConfig, salt: int,
                     level: float = 0.95) -> tuple[float, float]:
    """Scene-cluster bootstrap interval of a mean (resampling whole scenes).

    Parameters
    ----------
    values : array_like, shape (n,)
        One value per item (e.g. 0/1 covered).
    clusters : array_like, shape (n,)
        The scene of each item.
    config : AuditConfig
        Seed and number of replicates.
    salt : int
        Mixed into the seed so that different groups get different streams.
    level : float, default 0.95
        Two-sided level.

    Returns
    -------
    lower, upper : float
        Percentile interval; NaN for an empty group.
    """
    from measured_space.scoring import cluster_bootstrap

    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng([config.seed, salt])
    boot = cluster_bootstrap(values, np.asarray(clusters), config.n_boot, rng)
    lower, upper = np.quantile(boot, [(1 - level) / 2, (1 + level) / 2])
    return float(lower), float(upper)


def coverage_table(long: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Empirical coverage with Wilson intervals, overall and by question type, depth source and sigma tercile.

    Parameters
    ----------
    long : pandas.DataFrame
        Output of :func:`coverage_long`.
    config : AuditConfig
        Seed and bootstrap size for the scene-cluster intervals.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, by, group, method, level, n, n_scenes, coverage, lo, hi, boot_lo, boot_hi,
        scene_mean, scene_min, scene_max, n_scenes_30, half_over_tau_p50``. ``lo, hi`` is the Wilson 95 %
        interval, which treats items as independent; ``boot_lo, boot_hi`` is a scene-cluster bootstrap 95 %
        interval, which does not (with few scenes it is itself rough). ``scene_mean`` weighs every held-out
        scene equally (the quantity split conformal targets when scenes are exchangeable); ``scene_min`` and
        ``scene_max`` show how far single held-out scenes with at least 30 items (``n_scenes_30`` of them)
        stray from nominal, i.e. how much coverage depends on the scene. ``by`` is ``all``, ``qtype``, ``depth_source`` or ``sigma_tercile``. Rows for all
        datasets pooled have ``dataset == "all"``. A group from one scene is descriptive, not a coverage claim.
    """
    rows = []
    long = long.assign(cluster=long["dataset"] + "/" + long["scene"])
    pooled = long.assign(dataset="all")
    for frame in (long, pooled):
        for by in ("all", "qtype", "depth_source", "sigma_tercile"):
            keys = ["dataset", "method", "level"] + ([] if by == "all" else [by])
            for key, g in frame.groupby(keys, sort=True):
                n, k = len(g), int(g["covered"].sum())
                lo, hi = wilson_interval(k, n)
                boot_lo, boot_hi = cluster_interval(g["covered"], g["cluster"], config, len(rows))
                per_scene = g.groupby("cluster")["covered"].agg(["mean", "size"])
                big = per_scene[per_scene["size"] >= 30]["mean"]
                rows.append({"dataset": key[0], "by": by, "group": "all" if by == "all" else key[3],
                             "method": key[1], "level": key[2], "n": n, "n_scenes": g["cluster"].nunique(),
                             "coverage": k / n, "lo": float(lo), "hi": float(hi), "boot_lo": boot_lo,
                             "boot_hi": boot_hi, "scene_mean": float(per_scene["mean"].mean()),
                             "scene_min": float(big.min()) if len(big) else math.nan,
                             "scene_max": float(big.max()) if len(big) else math.nan, "n_scenes_30": len(big),
                             "half_over_tau_p50": float(np.nanmedian(g["half_over_tau"]))})
    return pd.DataFrame(rows)


def rank_table(test: pd.DataFrame) -> pd.DataFrame:
    """Does sigma rank errors? Spearman rho(sigma, |e|) and held-out log-likelihood per model kind.

    Parameters
    ----------
    test : pandas.DataFrame
        Numeric test predictions joined with rows: ``dataset, qtype, depth_source, kind, e, mu, sigma, pi,
        sigma_g``.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, qtype, depth_source, kind, n, n_scenes, rho, rho_p, nll_mean, sigma_p50``. ``depth_source ==
        "all"`` pools the sources. ``nll_mean`` is the mean held-out negative log-likelihood (nats; lower is
        better). For ``const``, sigma varies only between folds, so its rho reflects fold-to-fold differences.
    """
    rows = []
    frames = [test, test.assign(depth_source="all")]
    for frame in frames:
        for (dataset, qtype, source, kind), g in frame.groupby(["dataset", "qtype", "depth_source", "kind"],
                                                               sort=True):
            sigma, abs_e = g["sigma"].to_numpy(float), np.abs(g["e"].to_numpy(float))
            if len(g) > 2 and np.ptp(sigma) > 0:
                rho, p = stats.spearmanr(sigma, abs_e)
            else:
                rho, p = math.nan, math.nan
            nll = mixture_nll(g["e"], g["mu"], g["sigma"], g["pi"], g["sigma_g"])
            rows.append({"dataset": dataset, "qtype": qtype, "depth_source": source, "kind": kind, "n": len(g),
                         "n_scenes": g["scene"].nunique(), "rho": float(rho), "rho_p": float(p), "nll_mean": float(np.mean(nll)),
                         "sigma_p50": float(np.median(sigma))})
    return pd.DataFrame(rows)


# ============================================================================================================
# 6. Categorical calibration
# ============================================================================================================


def reliability_bins(p: ArrayLike, correct: ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Reliability-diagram bins: predicted against observed P(correct) in equal-width bins.

    Parameters
    ----------
    p : array_like, shape (n,)
        Predicted probabilities of a correct label.
    correct : array_like, shape (n,)
        Outcomes (1 correct, 0 wrong).
    n_bins : int, default 10
        Number of equal-width bins on [0, 1].

    Returns
    -------
    pandas.DataFrame
        One row per non-empty bin: ``bin_lo, bin_hi, n, p_mean, observed, lo, hi`` (Wilson).
    """
    p, y = np.asarray(p, dtype=np.float64), np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            lo, hi = wilson_interval(y[m].sum(), m.sum())
            rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()),
                         "p_mean": float(p[m].mean()), "observed": float(y[m].mean()), "lo": float(lo),
                         "hi": float(hi)})
    return pd.DataFrame(rows)


def categorical_table(test: pd.DataFrame, n_bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Brier score, expected calibration error and log loss of p(correct), with reliability bins.

    Parameters
    ----------
    test : pandas.DataFrame
        Categorical test predictions joined with rows: ``dataset, qtype, kind, p, wrong``.
    n_bins : int, default 10
        Reliability bins.

    Returns
    -------
    scores : pandas.DataFrame
        Columns ``dataset, qtype, kind, n, flip_rate, brier, brier_skill, ece, log_loss, auroc``. ``qtype ==
        "all"`` pools the categorical types. ``brier_skill`` is ``1 - brier / brier(const)`` in the same group
        (positive beats the constant). ``auroc`` is how well ``1 - p`` separates wrong labels from right ones.
    bins : pandas.DataFrame
        :func:`reliability_bins` per ``dataset, qtype, kind``.
    """
    rows, bins = [], []
    for frame in (test, test.assign(qtype="all")):
        for (dataset, qtype, kind), g in frame.groupby(["dataset", "qtype", "kind"], sort=True):
            p, y = g["p"].to_numpy(float), 1.0 - g["wrong"].to_numpy(float)
            b = reliability_bins(p, y, n_bins)
            ece = float(np.sum(b["n"] * np.abs(b["p_mean"] - b["observed"])) / len(g)) if len(b) else math.nan
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            rows.append({"dataset": dataset, "qtype": qtype, "kind": kind, "n": len(g),
                         "flip_rate": float(1 - y.mean()), "brier": float(np.mean((p - y) ** 2)), "ece": ece,
                         "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
                         "auroc": auroc(1 - p, 1 - y)})
            bins.append(b.assign(dataset=dataset, qtype=qtype, kind=kind))
    scores = pd.DataFrame(rows)
    const = scores[scores["kind"] == "const"].set_index(["dataset", "qtype"])["brier"]
    scores["brier_skill"] = 1 - scores["brier"] / [const.get((d, q), math.nan)
                                                  for d, q in zip(scores["dataset"], scores["qtype"])]
    return scores, pd.concat(bins, ignore_index=True) if bins else pd.DataFrame()


def auroc(score: ArrayLike, positive: ArrayLike) -> float:
    """Area under the ROC curve of a score for a binary outcome (ties count one half).

    Parameters
    ----------
    score : array_like, shape (n,)
        Higher should mean more likely positive.
    positive : array_like, shape (n,)
        Outcomes (1 positive, 0 negative).

    Returns
    -------
    float
        AUROC in [0, 1]; NaN if either class is empty.
    """
    score, pos = np.asarray(score, dtype=np.float64), np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = stats.rankdata(score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ============================================================================================================
# 7. Gate: Learn-then-Test threshold, decisions, risk-yield
# ============================================================================================================


def ltt_threshold(scores: ArrayLike, wrong: ArrayLike, epsilon: float, delta: float, *, min_accept: int = 30,
                  min_accept_frac: float = 0.0, growth: float = 1.25,
                  max_threshold: float = math.inf) -> tuple[float, dict[str, Any]]:
    """Largest acceptance threshold certified to keep ``P(wrong | score <= lambda) <= epsilon`` (Learn-then-Test).

    Items with ``score <= lambda`` are accepted. For each candidate ``lambda`` the null hypothesis "the error
    rate among accepted items exceeds ``epsilon``" has the binomial p-value ``P(Binomial(n_acc, epsilon) <=
    k_err)``. Candidates are the calibration scores at which ``min_accept``, ``min_accept * growth``,
    ``min_accept * growth^2``, ... items (and finally all items) are accepted. They are tested in increasing
    order (fixed-sequence testing, which controls the family-wise error at ``delta`` without a Bonferroni split)
    and testing stops at the first p-value above ``delta``. The returned threshold is the last one rejected, so
    with probability at least ``1 - delta`` over the calibration items (if they and the test items are i.i.d.)
    the error rate among accepted test items is at most ``epsilon``. The candidates depend only on the scores,
    not on the errors, so the procedure is valid conditional on the scores. A geometric grid has fewer
    low-power early steps than a dense one; in simulation it kept violations below ``delta`` with the same
    yield.

    Parameters
    ----------
    scores : array_like, shape (n,)
        Calibration scores (low = more trustworthy).
    wrong : array_like, shape (n,)
        Whether each calibration label is wrong (1) or right (0).
    epsilon : float
        Error bound to certify.
    delta : float
        Allowed probability that the certificate is wrong.
    min_accept : int, default 30
        Accepted calibration items at the first candidate ...
    min_accept_frac : float, default 0.0
        ... or this share of all calibration items, whichever is larger.
    growth : float, default 1.25
        Ratio between successive accepted counts (> 1).
    max_threshold : float, default inf
        Candidates above this are not tested.

    Returns
    -------
    lam : float
        The certified threshold; ``-inf`` (accept nothing) if none is certified.
    info : dict
        ``n_cal``, ``n_tested``, ``n_accepted_cal`` and ``err_accepted_cal`` at the chosen threshold, and
        ``p_value`` of the last test run.
    """
    s, w = np.asarray(scores, dtype=np.float64), np.asarray(wrong, dtype=np.float64)
    order = np.argsort(s, kind="stable")
    s_sorted, cum_err = s[order], np.concatenate([[0.0], np.cumsum(w[order])])
    n = len(s)
    min_accept = max(min_accept, math.ceil(min_accept_frac * n))
    counts = [min_accept * growth**j for j in range(int(np.ceil(np.log(max(n, 1) / min_accept) / np.log(growth)))
                                                    + 1)] if n >= min_accept else []
    counts = sorted({min(int(round(c)), n) for c in counts} | ({n} if n >= min_accept else set()))
    lam, info = -math.inf, {"n_cal": n, "n_tested": 0, "n_accepted_cal": 0, "err_accepted_cal": math.nan,
                            "p_value": math.nan}
    for count in counts:
        candidate = float(s_sorted[count - 1])
        if candidate > max_threshold:
            break
        n_acc = int(np.searchsorted(s_sorted, candidate, side="right"))       # ties are accepted together
        k_err = cum_err[n_acc]
        p_value = float(stats.binom.cdf(k_err, n_acc, epsilon))
        info["n_tested"] += 1
        info["p_value"] = p_value
        if p_value > delta:
            break
        lam = candidate
        info.update(n_accepted_cal=n_acc, err_accepted_cal=float(k_err / n_acc))
    return lam, info


def gate_scores(frame: pd.DataFrame, answer_kind: str, z: float, tau_scale: float = 1.0) -> tuple[Array, Array]:
    """Gate score and wrong-label flag of each row.

    Parameters
    ----------
    frame : pandas.DataFrame
        Predictions joined with rows. Numeric: ``mu, sigma, tau_hat, e, tau``. Categorical: ``p, wrong``.
    answer_kind : str
        ``"numeric"`` or ``"categorical"``.
    z : float
        Gaussian multiplier of the numeric bound.
    tau_scale : float, default 1.0
        Multiple of the tolerance a numeric label must meet.

    Returns
    -------
    score : numpy.ndarray, shape (n,)
        Numeric: ``(|mu| + z sigma) / (tau_scale * tau_hat)``, the model's bound on ``|e|`` in tolerances.
        Categorical: ``1 - p``.
    wrong : numpy.ndarray, shape (n,)
        Numeric: ``|e| > tau_scale * tau``. Categorical: the flip flag.
    """
    if answer_kind == "numeric":
        score = (np.abs(frame["mu"]) + z * frame["sigma"]) / (tau_scale * frame["tau_hat"])
        wrong = np.abs(frame["e"]) > tau_scale * frame["tau"]
        return score.to_numpy(float), wrong.to_numpy(float)
    return 1.0 - frame["p"].to_numpy(float), frame["wrong"].to_numpy(float)


def gate_decisions(test: pd.DataFrame, cal: pd.DataFrame, answer_kind: str, config: AuditConfig,
                   tau_scale: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Accept / flag / reject each test label, with thresholds certified on its calibration scenes.

    Per dataset, model kind and test scene, :func:`ltt_threshold` picks ``lambda`` (at most the reject
    threshold) from the calibration rows of that test scene (pooled over the question types of
    ``answer_kind``). A test label is accepted if its
    score is at most ``lambda``, rejected if its score exceeds ``kappa`` (numeric: the bound on ``|e|`` is more
    than ``kappa`` tolerances) or ``0.5`` (categorical: the label is more likely wrong than right), and flagged
    for review otherwise.

    Parameters
    ----------
    test : pandas.DataFrame
        Test predictions joined with rows (see :func:`gate_scores`), with ``dataset, kind, scene``.
    cal : pandas.DataFrame
        Calibration predictions joined with rows, with ``test_scene``.
    answer_kind : str
        ``"numeric"`` or ``"categorical"``.
    config : AuditConfig
        Epsilon, delta, kappa and the Learn-then-Test settings.
    tau_scale : float, default 1.0
        Multiple of the tolerance (numeric only).

    Returns
    -------
    decisions : pandas.DataFrame
        ``test`` plus ``score``, ``wrong_s`` (wrong at this tolerance), ``lam`` and ``decision`` (``accept``,
        ``flag`` or ``reject``).
    thresholds : pandas.DataFrame
        One row per dataset, kind and test scene: ``lam`` and the :func:`ltt_threshold` info.
    """
    z = config.gate_z
    reject_above = config.kappa if answer_kind == "numeric" else 0.5
    out = test.copy()
    out["score"], out["wrong_s"] = gate_scores(out, answer_kind, z, tau_scale)
    cal_score, cal_wrong = gate_scores(cal, answer_kind, z, tau_scale)
    cal = cal.assign(score=cal_score, wrong_s=cal_wrong)
    cal_groups = cal.groupby(["dataset", "kind", "test_scene"], sort=False).indices
    lam = np.full(len(out), -math.inf)
    rows = []
    for key, idx in out.groupby(["dataset", "kind", "scene"], sort=True).indices.items():
        cidx = cal_groups.get(key, np.array([], dtype=int))
        value, info = ltt_threshold(cal["score"].to_numpy()[cidx], cal["wrong_s"].to_numpy()[cidx],
                                    config.epsilon, config.delta, min_accept=config.min_accept,
                                    min_accept_frac=config.min_accept_frac, max_threshold=reject_above)
        lam[idx] = value
        rows.append({"dataset": key[0], "kind": key[1], "scene": key[2], "lam": value, **info})
    out["lam"] = lam
    score = out["score"].to_numpy(float)
    out["decision"] = np.where(score <= lam, "accept", np.where(score > reject_above, "reject", "flag"))
    return out, pd.DataFrame(rows)


def gate_table(decisions: pd.DataFrame, n_questions: Mapping[str, int], answer_kind: str, config: AuditConfig,
               tau_scale: float) -> pd.DataFrame:
    """Yield and error rate of each gate decision, per dataset and model kind.

    Parameters
    ----------
    decisions : pandas.DataFrame
        Output of :func:`gate_decisions`.
    n_questions : mapping of str to int
        Per dataset, questions of this answer kind over all depth sources, including misses (the yield
        denominator that counts a missed label as not accepted).
    answer_kind : str
        ``"numeric"`` or ``"categorical"``.
    config : AuditConfig
        Epsilon and delta (recorded), and the bootstrap settings.
    tau_scale : float
        Tolerance multiple, recorded.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, answer_kind, kind, tau_scale, epsilon, delta, n_labelled, err_all, lam_p50,
        n_scenes_none`` and, per decision ``d`` in accept/flag/reject: ``d_n``, ``d_yield`` (share of labelled
        rows), ``d_err`` with Wilson ``d_lo``, ``d_hi``; plus ``accept_boot_lo``, ``accept_boot_hi`` (scene-cluster
        bootstrap 95 % interval of ``accept_err``), ``n_scenes``, ``accept_yield_all`` (share of all questions)
        and ``certified_ok`` (``accept_err <= epsilon``).
    """
    rows = []
    for (dataset, kind), g in decisions.groupby(["dataset", "kind"], sort=True):
        lam = g.groupby("scene")["lam"].first()
        row: dict[str, Any] = {"dataset": dataset, "answer_kind": answer_kind, "kind": kind,
                               "tau_scale": tau_scale, "epsilon": config.epsilon, "delta": config.delta,
                               "n_labelled": len(g), "err_all": float(g["wrong_s"].mean()),
                               "lam_p50": float(np.median(lam)), "n_scenes_none": int(np.sum(~np.isfinite(lam)))}
        for d in ("accept", "flag", "reject"):
            r = _rate(g.loc[g["decision"] == d, "wrong_s"])
            row.update({f"{d}_n": r["n"], f"{d}_yield": r["n"] / len(g), f"{d}_err": r["rate"], f"{d}_lo": r["lo"],
                        f"{d}_hi": r["hi"]})
        acc = g[g["decision"] == "accept"]
        row["accept_boot_lo"], row["accept_boot_hi"] = cluster_interval(
            acc["wrong_s"], acc["dataset"] + "/" + acc["scene"], config, len(rows))
        row["n_scenes"] = g["scene"].nunique()
        row["accept_yield_all"] = row["accept_n"] / max(n_questions.get(dataset, len(g)), 1)
        row["certified_ok"] = bool(row["accept_n"] == 0 or row["accept_err"] <= config.epsilon)
        rows.append(row)
    return pd.DataFrame(rows)


def risk_yield(score: ArrayLike, wrong: ArrayLike, yields: ArrayLike, rng: np.random.Generator) -> Array:
    """Error rate among the accepted items when the lowest-scoring fraction is accepted.

    Parameters
    ----------
    score : array_like, shape (n,)
        Ranking score (accept low first). Ties are broken at random.
    wrong : array_like, shape (n,)
        Whether each label is wrong.
    yields : array_like, shape (m,)
        Accepted fractions in (0, 1].
    rng : numpy.random.Generator
        Tie-breaking random source.

    Returns
    -------
    numpy.ndarray, shape (m,)
        Error rate among the ``ceil(y n)`` best-ranked items, per yield.
    """
    s, w = np.asarray(score, dtype=np.float64), np.asarray(wrong, dtype=np.float64)
    order = np.lexsort((rng.random(len(s)), s))
    cum = np.cumsum(w[order])
    k = np.maximum(np.ceil(np.asarray(yields) * len(s)).astype(int), 1)
    return cum[k - 1] / k


def aurc(score: ArrayLike, wrong: ArrayLike, rng: np.random.Generator) -> float:
    """Area under the risk-yield curve: mean error rate among the ``k`` best-ranked labels over ``k = 1..n``.

    Parameters
    ----------
    score : array_like, shape (n,)
        Ranking score (low first); ties broken at random.
    wrong : array_like, shape (n,)
        Whether each label is wrong.
    rng : numpy.random.Generator
        Tie-breaking random source.

    Returns
    -------
    float
        The area (lower is better); NaN for no items.
    """
    n = len(np.asarray(wrong))
    return float(risk_yield(score, wrong, np.arange(1, n + 1) / n, rng).mean()) if n else math.nan


def aurc_difference(reference: pd.DataFrame, other: pd.DataFrame, config: AuditConfig,
                    salt: int) -> tuple[float, float, float]:
    """``AURC(other) - AURC(reference)`` on the same labels, with a paired scene-cluster bootstrap interval.

    Parameters
    ----------
    reference : pandas.DataFrame
        Gate decisions of the reference ranking (``row, scene, score, wrong_s``).
    other : pandas.DataFrame
        Gate decisions of the compared ranking for the same rows.
    config : AuditConfig
        Seed and number of replicates.
    salt : int
        Mixed into the seed.

    Returns
    -------
    diff, lower, upper : float
        The difference and its 95 % percentile interval from resampling whole scenes (both rankings on the same
        resampled labels).
    """
    a = reference.set_index("row")
    b = other.set_index("row").reindex(a.index)
    keep = b["score"].notna().to_numpy()
    a, b = a[keep], b[keep]
    sa, sb, w = a["score"].to_numpy(float), b["score"].to_numpy(float), a["wrong_s"].to_numpy(float)
    rng = np.random.default_rng([config.seed, salt])
    diff = aurc(sb, w, rng) - aurc(sa, w, rng)
    groups = list(pd.Series(np.arange(len(a))).groupby(a["scene"].to_numpy()).indices.values())
    reps = np.empty(config.n_boot)
    for i in range(config.n_boot):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        reps[i] = aurc(sb[idx], w[idx], rng) - aurc(sa[idx], w[idx], rng)
    lower, upper = np.quantile(reps, [0.025, 0.975])
    return float(diff), float(lower), float(upper)


def risk_yield_table(decisions: pd.DataFrame, answer_kind: str, config: AuditConfig) -> tuple[pd.DataFrame,
                                                                                               pd.DataFrame]:
    """Risk-yield curves and their areas, per dataset and ranking (model kinds, random, ideal).

    Parameters
    ----------
    decisions : pandas.DataFrame
        Output of :func:`gate_decisions` (all kinds, one tolerance multiple).
    answer_kind : str
        ``"numeric"`` or ``"categorical"``.
    config : AuditConfig
        Seed.

    Returns
    -------
    curves : pandas.DataFrame
        Columns ``dataset, answer_kind, ranking, yield, risk`` on a grid of 20 yields from 0.05 to 1.
        ``random`` is the expected risk of a random order (the overall error rate); ``ideal`` puts every right
        label first.
    summary : pandas.DataFrame
        Columns ``dataset, answer_kind, ranking, n, aurc, auroc, d_aurc, d_lo, d_hi``: the area under the
        risk-yield curve (mean risk over all yields ``k / n``; lower is better), the score's AUROC for wrong
        labels, and for each baseline ranking ``d_aurc = AURC(baseline) - AURC(feature model)`` (positive: the
        feature model ranks better) with a paired scene-cluster bootstrap 95 % interval.
    """
    yields = np.linspace(0.05, 1.0, 20)
    rng = np.random.default_rng(config.seed)
    reference = "sigma" if answer_kind == "numeric" else "logistic"
    curves, summary = [], []
    for (dataset, kind), g in decisions.groupby(["dataset", "kind"], sort=True):
        s, w = g["score"].to_numpy(float), g["wrong_s"].to_numpy(float)
        curves.append(pd.DataFrame({"dataset": dataset, "answer_kind": answer_kind, "ranking": kind,
                                    "yield": yields, "risk": risk_yield(s, w, yields, rng)}))
        row = {"dataset": dataset, "answer_kind": answer_kind, "ranking": kind, "n": len(s),
               "aurc": aurc(s, w, rng), "auroc": auroc(s, w)}
        ref = decisions[(decisions["dataset"] == dataset) & (decisions["kind"] == reference)]
        if kind != reference and len(ref):
            row["d_aurc"], row["d_lo"], row["d_hi"] = aurc_difference(ref, g, config, len(summary))
        summary.append(row)
    for dataset, g in decisions.groupby("dataset", sort=True):
        w = g.drop_duplicates("row")["wrong_s"].to_numpy(float)
        n = len(w)
        n_right = int(n - w.sum())
        k_all = np.arange(1, n + 1)
        ideal_full = np.maximum(k_all - n_right, 0) / k_all
        k = np.maximum(np.ceil(yields * n).astype(int), 1)
        for ranking, risk, area in (("random", np.full(len(yields), w.mean()), float(w.mean())),
                                    ("ideal", np.maximum(k - n_right, 0) / k, float(ideal_full.mean()))):
            curves.append(pd.DataFrame({"dataset": dataset, "answer_kind": answer_kind, "ranking": ranking,
                                        "yield": yields, "risk": risk}))
            summary.append({"dataset": dataset, "answer_kind": answer_kind, "ranking": ranking, "n": n,
                            "aurc": area, "auroc": 0.5 if ranking == "random" else 1.0})
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summary)


# ============================================================================================================
# 8. The whole audit
# ============================================================================================================


@dataclass
class AuditResult:
    """Everything one audit run produces.

    Attributes
    ----------
    pairs : pandas.DataFrame
        :func:`load_labels` rows (per item: keep out of the repository).
    numeric : pandas.DataFrame
        Calibrated numeric test predictions, one row per labelled numeric row and model kind (per item).
    categorical : pandas.DataFrame
        Categorical test predictions, one row per labelled categorical row and model kind (per item).
    gate : pandas.DataFrame
        Gate decisions at every tolerance multiple (per item).
    tables : dict of str to pandas.DataFrame
        Aggregate tables, the only outputs meant for committing: ``errors``, ``coverage``, ``rank``,
        ``comparison``, ``coefficients``, ``categorical``, ``reliability``, ``gate``, ``gate_thresholds``,
        ``risk_yield``, ``aurc``, ``bias_correction``, ``ridge``; and ``folds`` (:func:`fold_table` of every
        dataset and question type with more than ``max_folds`` scenes, and of every modelled question type of a
        dataset with explicit ``fold_groups``) when there is such a group.
    notes : list of str
        Groups skipped and why, and other remarks produced while running.
    calibration : pandas.DataFrame
        The numeric leave-two-out calibration predictions the conformal step and the gate used (per item, kept
        out of the repository): columns ``kind, test_scene, row`` (index into ``pairs``) and the predicted
        ``mu, sigma`` (m), ``pi``, ``sigma_g`` (m) and ``ridge``; one row per calibration item and test scene
        (:func:`cross_fit`'s ``cal``), every numeric model kind. Empty when nothing numeric was modelled.
    quantiles : pandas.DataFrame
        :func:`conformal_quantile_table` of those predictions: every test scene's conformal quantiles per
        Mondrian cell, the ones that give its answers their ``half*``. Empty when nothing numeric was modelled.
    """

    pairs: pd.DataFrame
    numeric: pd.DataFrame
    categorical: pd.DataFrame
    gate: pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    quantiles: pd.DataFrame = field(default_factory=pd.DataFrame)


def _join(pred: pd.DataFrame, pairs: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Attach row columns to predictions by row index.

    Parameters
    ----------
    pred : pandas.DataFrame
        Predictions with a ``row`` column indexing ``pairs``.
    pairs : pandas.DataFrame
        The pair table.
    columns : sequence of str
        Columns of ``pairs`` to attach.

    Returns
    -------
    pandas.DataFrame
        ``pred`` with the columns added.
    """
    attached = pairs.loc[pred["row"].to_numpy(), list(columns)].reset_index(drop=True)
    return pd.concat([pred.reset_index(drop=True), attached], axis=1)


def coefficient_table(pairs: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Coefficients of the feature model fitted on all scenes (descriptive, in-sample).

    Parameters
    ----------
    pairs : pandas.DataFrame
        Pair table.
    config : AuditConfig
        Model settings.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, qtype, part, term, coef, n``. ``part`` is ``bias`` (m per standard deviation of the
        feature), ``log_var`` (log m^2 per standard deviation), ``gross`` (``pi`` and ``sigma_g`` in m) or
        ``ridge`` (the weight chosen by cross-validation over all scenes).
    """
    rows = []
    numeric = pairs[(pairs["answer_type"] == "number") & pairs["matched"]]
    for (dataset, qtype), g in numeric.groupby(["dataset", "qtype"], sort=True):
        if len(g) < config.min_rows:
            continue
        sources = depth_sources(pairs[pairs["dataset"] == dataset])
        pred = NumericPredictor.fit(g, "sigma", sources, config)
        names = pred.standardizer.columns()
        rows.append({"dataset": dataset, "qtype": qtype, "part": "ridge", "term": "ridge", "coef": pred.ridge,
                     "n": len(g)})
        for part, coefs in (("bias", pred.model.beta), ("log_var", pred.model.gamma)):
            rows += [{"dataset": dataset, "qtype": qtype, "part": part, "term": t, "coef": float(c), "n": len(g)}
                     for t, c in zip(names, coefs)]
        rows += [{"dataset": dataset, "qtype": qtype, "part": "gross", "term": "pi", "coef": pred.model.pi_gross,
                  "n": len(g)},
                 {"dataset": dataset, "qtype": qtype, "part": "gross", "term": "sigma_g",
                  "coef": pred.model.sigma_gross, "n": len(g)}]
    return pd.DataFrame(rows)


def comparison_table(numeric: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    """Held-out log-likelihood of the feature model against each baseline, paired per label.

    Parameters
    ----------
    numeric : pandas.DataFrame
        Numeric test predictions with per-row ``nll`` (``dataset, qtype, scene, kind, row, nll``).
    config : AuditConfig
        Bootstrap settings.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, qtype, baseline, n, n_scenes, d_nll, d_lo, d_hi``: the mean of ``nll(baseline) -
        nll(feature model)`` in nats per label (positive: the feature model predicts held-out errors better)
        with a scene-cluster bootstrap 95 % interval. ``qtype == "all"`` pools the numeric types.
    """
    wide = numeric.pivot_table(index="row", columns="kind", values="nll")
    meta = numeric.drop_duplicates("row").set_index("row")[["dataset", "qtype", "scene"]].reindex(wide.index)
    wide = pd.concat([wide, meta], axis=1)
    rows = []
    for frame in (wide, wide.assign(qtype="all")):
        for (dataset, qtype), g in frame.groupby(["dataset", "qtype"], sort=True):
            for baseline in (k for k in NUMERIC_KINDS if k != "sigma" and k in g):
                d = (g[baseline] - g["sigma"]).to_numpy(float)
                ok = np.isfinite(d)
                clusters = (g["dataset"] + "/" + g["scene"]).to_numpy()[ok]
                lo, hi = cluster_interval(d[ok], clusters, config, len(rows))
                rows.append({"dataset": dataset, "qtype": qtype, "baseline": baseline, "n": int(ok.sum()),
                             "n_scenes": len(set(clusters)), "d_nll": float(d[ok].mean()), "d_lo": lo, "d_hi": hi})
    return pd.DataFrame(rows)


def ridge_table(test: pd.DataFrame) -> pd.DataFrame:
    """Which ridge weights the leave-one-scene-out models chose.

    Parameters
    ----------
    test : pandas.DataFrame
        Test predictions with ``dataset, qtype, scene, kind, ridge`` (one model per test scene, or per fold
        above ``max_folds`` scenes).

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, qtype, kind, ridge, n_models``: how many test scenes' models chose each weight
        (penalised kinds only). Under leave-one-scene-out that is the number of models; when scenes are grouped
        into folds, the scenes of a fold share one model and each counts.
    """
    models = test.dropna(subset=["ridge"]).drop_duplicates(["dataset", "qtype", "kind", "scene"])
    return (models.groupby(["dataset", "qtype", "kind", "ridge"]).size().rename("n_models").reset_index())


def bias_correction_table(numeric: pd.DataFrame) -> pd.DataFrame:
    """Wrong-label rate of the raw auto value against the bias-corrected value ``auto - mu`` (out of scene).

    Parameters
    ----------
    numeric : pandas.DataFrame
        Numeric test predictions joined with rows (``dataset, qtype, kind, e, mu, tau``).

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, qtype, kind, n, wrong_raw, wrong_corrected, mae_raw, mae_corrected`` (m).
    """
    rows = []
    frames = [numeric, numeric.assign(qtype="all")]
    for frame in frames:
        for (dataset, qtype, kind), g in frame.groupby(["dataset", "qtype", "kind"], sort=True):
            e, mu, tau = g["e"].to_numpy(float), g["mu"].to_numpy(float), g["tau"].to_numpy(float)
            rows.append({"dataset": dataset, "qtype": qtype, "kind": kind, "n": len(g),
                         "wrong_raw": float(np.mean(np.abs(e) > tau)),
                         "wrong_corrected": float(np.mean(np.abs(e - mu) > tau)),
                         "mae_raw": float(np.mean(np.abs(e))), "mae_corrected": float(np.mean(np.abs(e - mu)))})
    return pd.DataFrame(rows)


def depth_sources(pairs: pd.DataFrame) -> list[str]:
    """A dataset's depth sources, most frequent first (the first is the one-hot reference level).

    With the most frequent source as the reference, a source missing from a training fold (e.g. one that only
    one scene has) gets an all-zero one-hot column, so the ridge keeps its effect at zero and its rows are
    predicted at the reference level instead of an arbitrary split between intercept and source effect.

    Parameters
    ----------
    pairs : pandas.DataFrame
        One dataset's pair table.

    Returns
    -------
    list of str
        Sources by decreasing row count, ties by name.
    """
    counts = pairs["depth_source"].value_counts()
    return sorted(counts.index, key=lambda src: (-counts[src], src))


def run_audit(pairs: pd.DataFrame, config: AuditConfig | None = None,
              log: Callable[[str], None] | None = None) -> AuditResult:
    """Run the whole audit on a pair table (one or several datasets).

    Parameters
    ----------
    pairs : pandas.DataFrame
        Output of :func:`load_labels` (several datasets may be concatenated) or :func:`synthetic_pairs`.
    config : AuditConfig, optional
        Settings; defaults to ``AuditConfig()``.
    log : callable, optional
        Receives progress messages.

    Returns
    -------
    AuditResult
        Per-item results and aggregate tables.
    """
    config = config or AuditConfig()
    say = log or (lambda message: None)
    pairs = pairs.reset_index(drop=True)
    notes: list[str] = []
    row_cols = ["dataset", "scene", "qtype", "depth_source", "e", "tau", "tau_hat", "wrong", "log_range"]
    tasks: list[tuple[pd.DataFrame, str, bool, list[str], AuditConfig]] = []
    folds: list[pd.DataFrame] = []
    for dataset, dpairs in pairs.groupby("dataset", sort=True):
        sources = depth_sources(dpairs)
        groups = config.groups_of(dataset)
        if groups is not None:                        # every scene of the dataset, before anything is fitted
            scene_folds(dpairs["scene"], None, groups=groups)
        labelled = dpairs[dpairs["matched"]]
        for qtype, rows in labelled.groupby("qtype", sort=True):
            numeric = rows["answer_type"].iloc[0] == "number"
            if numeric:
                rows = rows[np.isfinite(rows["e"].to_numpy(float))]
            n_scenes = rows["scene"].nunique()
            n_groups = None if groups is None else len(set(scene_folds(rows["scene"], None, groups=groups).values()))
            if n_scenes < config.min_scenes or len(rows) < config.min_rows or (n_groups is not None and n_groups < 3):
                notes.append(f"{dataset}/{qtype}: skipped ({len(rows)} labelled rows, {n_scenes} scenes"
                             + ("" if n_groups is None else f" in {n_groups} fold groups") + ")")
                continue
            if groups is not None:
                fold = fold_table(rows, config)
                folds.append(fold)
                notes.append(f"{dataset}/{qtype}: {n_scenes} scenes cross-fitted in {n_groups} explicit fold groups "
                             f"of {fold['fold_scenes'].min()}-{fold['fold_scenes'].max()} scenes and "
                             f"{fold['fold_rows'].min()}-{fold['fold_rows'].max()} rows (table folds)")
            elif n_scenes > config.max_folds:
                fold = fold_table(rows, config)
                folds.append(fold)
                notes.append(f"{dataset}/{qtype}: {n_scenes} scenes cross-fitted in {config.max_folds} folds of "
                             f"{fold['fold_scenes'].min()}-{fold['fold_scenes'].max()} scenes and "
                             f"{fold['fold_rows'].min()}-{fold['fold_rows'].max()} rows (table folds)")
            tasks += [(rows, kind, numeric, sources, config)
                      for kind in (NUMERIC_KINDS if numeric else CATEGORICAL_KINDS)]
    num_test, num_cal, cat_test, cat_cal = [], [], [], []
    for task, (test, cal) in zip(tasks, _run_tasks(tasks, config.n_jobs, say)):
        (num_test if task[2] else cat_test).append(test)
        (num_cal if task[2] else cat_cal).append(cal)

    tables: dict[str, pd.DataFrame] = {"errors": error_table(pairs)}
    numeric_out = categorical_out = calibration = quantiles = pd.DataFrame()
    gates: list[pd.DataFrame] = []
    gate_rows, threshold_rows, curve_rows, aurc_rows = [], [], [], []
    n_numeric = pairs[pairs["answer_type"] == "number"].groupby("dataset").size().to_dict()
    n_categorical = pairs[pairs["answer_type"] != "number"].groupby("dataset").size().to_dict()

    if num_test:
        test = _join(pd.concat(num_test, ignore_index=True), pairs, row_cols)
        cal = _join(pd.concat(num_cal, ignore_index=True), pairs, row_cols)
        numeric_out = conformal_calibrate(test, cal, config)
        calibration = pd.concat(num_cal, ignore_index=True)[["kind", "test_scene", "row", "mu", "sigma", "pi",
                                                             "sigma_g", "ridge"]]
        quantiles = conformal_quantile_table(test, cal, config)
        numeric_out["nll"] = mixture_nll(numeric_out["e"], numeric_out["mu"], numeric_out["sigma"],
                                         numeric_out["pi"], numeric_out["sigma_g"])
        tables["comparison"] = comparison_table(numeric_out, config)
        tables["coverage"] = coverage_table(coverage_long(numeric_out, config), config)
        tables["rank"] = rank_table(numeric_out)
        tables["coefficients"] = coefficient_table(pairs, config)
        tables["bias_correction"] = bias_correction_table(numeric_out)
        n_pooled = int(numeric_out.loc[numeric_out["cal_cell"] == "pooled", "row"].nunique())
        if n_pooled:
            cells = numeric_out.loc[numeric_out["cal_cell"] == "pooled", ["dataset", "depth_source", "scene"]]
            notes.append(f"{n_pooled} numeric test rows calibrated on the question type pooled over depth sources "
                         f"(too few same-source calibration scores): "
                         f"{sorted(set(map(tuple, cells.drop_duplicates().to_numpy().tolist())))}")
        for tau_scale in config.tau_scales:
            dec, thr = gate_decisions(numeric_out, cal, "numeric", config, tau_scale)
            gates.append(dec.assign(answer_kind="numeric", tau_scale=tau_scale))
            gate_rows.append(gate_table(dec, n_numeric, "numeric", config, tau_scale))
            threshold_rows.append(thr.assign(answer_kind="numeric", tau_scale=tau_scale))
            plain = _plain_rule(dec, tau_scale)
            gate_rows.append(plain.assign(answer_kind="numeric", tau_scale=tau_scale, epsilon=config.epsilon,
                                          delta=config.delta))
            if tau_scale == 1.0:
                curves, summary = risk_yield_table(dec, "numeric", config)
                curve_rows.append(curves)
                aurc_rows.append(summary)
    if cat_test:
        cat_cols = ["dataset", "scene", "qtype", "depth_source", "wrong", "margin", "log_range"]
        test = _join(pd.concat(cat_test, ignore_index=True), pairs, cat_cols)
        cal = _join(pd.concat(cat_cal, ignore_index=True), pairs, cat_cols)
        categorical_out = test
        tables["categorical"], tables["reliability"] = categorical_table(test, config.n_bins)
        dec, thr = gate_decisions(test, cal, "categorical", config)
        gates.append(dec.assign(answer_kind="categorical", tau_scale=1.0))
        gate_rows.append(gate_table(dec, n_categorical, "categorical", config, 1.0))
        threshold_rows.append(thr.assign(answer_kind="categorical", tau_scale=1.0))
        curves, summary = risk_yield_table(dec, "categorical", config)
        curve_rows.append(curves)
        aurc_rows.append(summary)
    fitted = [f for f in (numeric_out, categorical_out) if len(f)]
    if fitted:
        tables["ridge"] = ridge_table(pd.concat(fitted, ignore_index=True))
    if folds:                          # only when some group was grouped: leave-one-scene-out runs are unchanged
        tables["folds"] = pd.concat(folds, ignore_index=True)
    for name, parts in (("gate", gate_rows), ("gate_thresholds", threshold_rows), ("risk_yield", curve_rows),
                        ("aurc", aurc_rows)):
        if parts:
            tables[name] = pd.concat(parts, ignore_index=True)
    return AuditResult(pairs, numeric_out, categorical_out,
                       pd.concat(gates, ignore_index=True) if gates else pd.DataFrame(), tables, notes,
                       calibration=calibration, quantiles=quantiles)


def _plain_rule(decisions: pd.DataFrame, tau_scale: float) -> pd.DataFrame:
    """Yield and error of the uncertified rule "accept if the conformal 90 % interval bound is within tolerance".

    Accept when ``|mu| + q90 sigma <= tau_scale * tau_hat``: the calibrated interval for ``e`` lies inside the
    tolerance band. Conformal coverage is marginal, so this rule carries no bound on the error rate among the
    labels it accepts; it is reported next to the certified gate for comparison.

    Parameters
    ----------
    decisions : pandas.DataFrame
        Numeric gate decisions (with ``mu, half90, tau_hat, wrong_s``).
    tau_scale : float
        Tolerance multiple.

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, kind, n_labelled, accept_n, accept_yield, accept_err, accept_lo, accept_hi`` with
        ``kind`` suffixed ``+conformal90``.
    """
    rows: list[dict[str, Any]] = []
    if "half90" not in decisions:
        return pd.DataFrame(rows)
    for (dataset, kind), g in decisions.groupby(["dataset", "kind"], sort=True):
        accept = (np.abs(g["mu"]) + g["half90"]) <= tau_scale * g["tau_hat"]
        r = _rate(g.loc[accept, "wrong_s"])
        rows.append({"dataset": dataset, "kind": f"{kind}+conformal90", "n_labelled": len(g), "accept_n": r["n"],
                     "accept_yield": r["n"] / len(g), "accept_err": r["rate"], "accept_lo": r["lo"],
                     "accept_hi": r["hi"], "err_all": float(g["wrong_s"].mean())})
    return pd.DataFrame(rows)


# ============================================================================================================
# 9. Synthetic data with a known error structure (tests and dry runs)
# ============================================================================================================


def synthetic_pairs(n_scenes: int = 6, n_items: int = 80, *, sources: Sequence[str] = ("laser", "moge"),
                    heteroscedastic: bool = True, noise_scale: float = 1.0, gross_rate: float = 0.02,
                    scene_sd: float = 0.0, miss_rate: float = 0.05, seed: int = 0,
                    dataset: str = "synthetic") -> pd.DataFrame:
    """A pair table with a known error structure, shaped like :func:`load_labels` output.

    Two numeric types (``size_height``, ``distance_center``) and one categorical type (``left_right``), each
    with ``n_items`` questions per scene, each answered once per depth source. The truth per numeric row is
    ``e = bias + sigma * N(0, 1)`` (plus, with probability ``gross_rate``, a gross error ``N(0, 1 m)``), with
    ``bias = -0.03 - 0.05 * border_touch`` and, if ``heteroscedastic``, ``log sigma`` rising with range,
    boundary fraction, border touch and the second depth source; otherwise one sigma for all rows. A
    categorical label flips with probability ``Phi(-margin / sigma)``. ``scene_sd`` adds a per-scene,
    per-source relative scale error (a shared capture scale), which makes items within a scene correlated.

    Parameters
    ----------
    n_scenes : int, default 6
        Scenes.
    n_items : int, default 80
        Questions per scene and type.
    sources : sequence of str, default ("laser", "moge")
        Depth sources; each question is answered once per source.
    heteroscedastic : bool, default True
        Whether the error spread depends on the features.
    noise_scale : float, default 1.0
        Multiplies every row's sigma (raises the wrong-label rate).
    gross_rate : float, default 0.02
        Share of gross errors among numeric rows.
    scene_sd : float, default 0.0
        Standard deviation of the per-scene relative scale error.
    miss_rate : float, default 0.05
        Share of rows with no auto answer.
    seed : int, default 0
        Random seed.
    dataset : str, default "synthetic"
        Dataset name.

    Returns
    -------
    pandas.DataFrame
        Pair-table columns (see :func:`pair_table`), plus ``true_sigma`` (m) for numeric rows.
        ``match_ambiguous`` is 0 on labelled rows (no pairing is simulated) and NaN on misses.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_scenes):
        scene = f"scene{s:02d}"
        scale = {src: rng.normal(0.0, scene_sd) for src in sources}
        for qtype in ("size_height", "distance_center", "left_right"):
            for i in range(n_items):
                item_id = f"{scene}-{qtype}-{i}"
                rng_range = float(np.exp(rng.uniform(np.log(0.7), np.log(8.0))))
                boundary = float(rng.beta(2, 8))
                border = float(rng.random() < 0.3)
                hull = float(rng.uniform(0.5, 1.0))
                n_points = float(np.exp(rng.uniform(np.log(200), np.log(2e5))))
                det = float(rng.uniform(0.5, 1.0))
                oracle = float(rng.uniform(0.5, 3.0)) if qtype != "left_right" else str(rng.choice(["left",
                                                                                                    "right"]))
                for j, src in enumerate(sources):
                    mad_rel = float(np.exp(rng.normal(np.log(0.03), 0.5)) * (1 + j))
                    row: dict[str, Any] = {
                        "id": item_id, "dataset": dataset, "scene": scene, "camera_id": "cam0", "frame": str(i),
                        "qtype": qtype, "answer_type": "choice" if qtype == "left_right" else "number",
                        "depth_source": src, "n_referents": 1 if qtype == "size_height" else 2,
                        "log_range": math.log(rng_range), "boundary_fraction": boundary,
                        "border_touch": border, "depth_mad_rel": mad_rel, "hull_fill": hull,
                        "log_n_points": math.log(n_points), "det_score": det, "match_distance": math.nan,
                        "n_feature_sets": 1.0, "oracle": oracle}
                    if rng.random() < miss_rate:
                        row.update(matched=False, auto=None, wrong=math.nan)
                        rows.append(row)
                        continue
                    row["match_ambiguous"] = 0.0                  # no detection pairing to be ambiguous
                    log_sigma = (math.log(0.04) + (0.6 * (math.log(rng_range) - 1.0) + 2.5 * (boundary - 0.2)
                                                   + 0.5 * border + 0.4 * j) if heteroscedastic else math.log(0.06))
                    sigma = noise_scale * math.exp(log_sigma)
                    if qtype == "left_right":
                        margin = float(rng.exponential(0.5))
                        flip = rng.random() < stats.norm.cdf(-margin / (sigma * 3))
                        auto = oracle if not flip else ("right" if oracle == "left" else "left")
                        row.update(matched=True, auto=auto, wrong=float(flip), margin=margin,
                                   log_margin=math.log(max(margin, _MIN_MARGIN)))
                    else:
                        e = -0.03 - 0.05 * border + sigma * rng.normal() + oracle * scale[src]
                        if rng.random() < gross_rate:
                            e += rng.normal(0.0, 1.0)
                        value = oracle + e
                        tau, tau_hat = max(0.1, 0.1 * oracle), max(0.1, 0.1 * abs(value))
                        row.update(matched=True, auto=value, value=value, e=e, abs_e=abs(e), rel_e=e / oracle,
                                   tau=tau, tau_hat=tau_hat, wrong=float(abs(e) > tau),
                                   log_value=math.log(max(abs(value), _MIN_VALUE)), true_sigma=sigma)
                    rows.append(row)
    table = pd.DataFrame(rows)
    for col in ("oracle", "auto"):
        table[col] = table[col].astype(object)
    return table


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars, NaN and infinities so a value can be written as strict JSON.

    Parameters
    ----------
    value : Any
        A scalar, list or dict.

    Returns
    -------
    Any
        The same structure with numpy scalars as Python scalars and non-finite floats as None.
    """
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dumps(value: Any) -> str:
    """Strict JSON text of a value (see :func:`to_jsonable`).

    Parameters
    ----------
    value : Any
        A JSON-like structure.

    Returns
    -------
    str
        Indented JSON.
    """
    return json.dumps(to_jsonable(value), indent=1)
