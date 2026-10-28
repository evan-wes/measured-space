"""Human review effort: how much review delivers labels at a stated quality, and what it costs in minutes.

The audit (:mod:`measured_space.audit`) gives every automatic answer a held-out risk score and a certified gate
decision. This module turns those per-answer outputs into the numbers a labelling budget is planned with.

**Review policy.** Humans review answers in some order. A reviewed answer is corrected, so it is delivered right;
an unreviewed answer is delivered as it is. Every review costs the same time, whatever the answer and whether
it was right. After reviewing the share ``f`` of the answers, the **residual wrong share** of the delivered set
is (wrong answers left unreviewed) / (all answers). Orders (:data:`ORDERS`):

- ``ours``: the audit's feature model (numeric ``sigma``, categorical ``logistic``), highest risk score first;
- ``const``: the audit's constant baseline (``const`` for both answer kinds), highest risk score first;
- ``random``: the expected value of a random order, ``wrong_share * (1 - f)`` (a straight line);
- ``ideal``: every wrong answer first, a bound no ranking can beat.

Ties in a score are broken at random (seeded). **Combined** (numeric and categorical answers in one queue): the two
kinds' scores are on different scales (a bound on ``|e|`` in tolerances against ``1 - p(correct)``), so they are
interleaved by their rank fraction within their own kind (:func:`review_order`). Reviewing the share ``f`` of the
combined queue reviews the riskiest share ``f`` of each kind. That uses no ground truth; an allocation between
kinds tuned on the wrong flags would.

**Targets** (:func:`review_needed`, :func:`target_table`). The smallest review fraction that brings the residual
wrong share to at most a target (5 %, 2 %, 1 %), per order, with a scene-cluster bootstrap 95 % interval
(whole scenes resampled with replacement; the ranking stays the one fitted on all scenes) and a paired interval of
the difference to random review.

**Certified operating points** (:func:`operating_point`). The audit's gate accepts an answer when its score is at
most a threshold certified per held-out scene by Learn-then-Test (``P(wrong | accepted) <= epsilon`` with
confidence ``1 - delta``) and sends the rest (flagged and rejected) to review. The accepted share ``a`` and the
wrong share among accepted ``r`` give the review fraction ``1 - a`` and the residual wrong share ``r a``, against
the certified bound ``epsilon a``.

**Minutes** (:func:`minutes`): ``per * f * seconds / 60`` minutes per ``per`` (default 1,000) delivered answers,
against ``per * seconds / 60`` for reviewing everything.

**Miss review** (:func:`miss_review_table`). Missed objects have no answer to review. A miss flag
(:mod:`measured_space.misses`) is a 2D box where evidence says an object is and the labeller has none; reviewing
a flag costs the same time per flag. Per 1,000 images: flags to review, minutes, and misses recovered (distinct
misses some flag hits), from the miss audit's ``flag_metrics`` rows.

Caveats: answers from one scene share objects, a capture and its errors, so only the scene-cluster bootstrap is
reported; with few scenes (12 in the fixed-camera warehouse) it is rough. The gate's certificate is per answer
kind; the combined accepted set inherits ``epsilon`` (a mixture of two rates at most ``epsilon``) with confidence
``1 - 2 delta`` by the union bound.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

ORDERS: tuple[str, ...] = ("ours", "const", "random", "ideal")
"""tuple of str: review orders compared (see the module docstring)."""

RANKED_ORDERS: tuple[str, ...] = ("ours", "const")
"""tuple of str: orders given by an audit model's risk score."""

ANSWER_KINDS: tuple[str, ...] = ("numeric", "categorical", "combined")
"""tuple of str: answer kinds analysed; ``combined`` is numeric and categorical answers in one queue."""

MODEL_KIND: dict[str, dict[str, str]] = {"ours": {"numeric": "sigma", "categorical": "logistic"},
                                         "const": {"numeric": "const", "categorical": "const"}}
"""dict: the audit model kind (``gate.csv`` column ``kind``) behind each ranked order, per answer kind."""

DEFAULT_TARGETS: tuple[float, ...] = (0.05, 0.02, 0.01)
"""tuple of float: residual wrong shares the review must reach."""

DEFAULT_SECONDS: tuple[float, ...] = (10.0, 30.0, 60.0)
"""tuple of float: seconds of human time per reviewed answer (or per reviewed miss flag)."""

PER: int = 1000
"""int: minutes are reported per this many delivered answers (or images)."""

N_GRID: int = 201
"""int: review fractions on the curve grid, evenly spaced on [0, 1] (step 0.005)."""

GATE_COLUMNS: tuple[str, ...] = ("row", "kind", "answer_kind", "tau_scale", "scene", "depth_source", "wrong",
                                 "wrong_s", "score", "decision")
"""tuple of str: the ``gate.csv`` columns this module reads."""

MISS_COLUMNS: tuple[str, ...] = ("kind", "labeller", "source", "witness", "dataset", "n_views", "n_flags",
                                 "flags_per_view", "n_hit", "precision", "n_present", "n_missed",
                                 "n_missed_flagged", "flag_recall", "n_missed_clear", "n_missed_clear_flagged",
                                 "flag_recall_clear")
"""tuple of str: the ``flag_metrics.csv`` columns copied into the miss review table."""

_TOL = 1e-9          # relative slack when comparing a residual share with a target


def _rng(seed: int, salt: str) -> np.random.Generator:
    """A random source for one named group, reproducible from the seed.

    Parameters
    ----------
    seed : int
        Base seed.
    salt : str
        Names the group; mixed into the seed so that every group gets its own stream.

    Returns
    -------
    numpy.random.Generator
        The generator.
    """
    return np.random.default_rng([seed, zlib.crc32(salt.encode())])


def minute_columns(fraction: float, seconds: Sequence[float], prefix: str = "minutes", per: int = PER
                   ) -> dict[str, float]:
    """Minutes per ``per`` answers at each review time, as named columns.

    Parameters
    ----------
    fraction : float
        Share of answers reviewed (or flags per image, for miss review), in [0, 1] for answers.
    seconds : sequence of float
        Seconds of human time per review.
    prefix : str, default "minutes"
        Column-name prefix; a column is ``f"{prefix}_{t:g}s"``.
    per : int, default :data:`PER`
        Answers (or images) the minutes are counted for.

    Returns
    -------
    dict of str to float
        One entry per review time; NaN where ``fraction`` is NaN.
    """
    return {f"{prefix}_{t:g}s": minutes(fraction, t, per) for t in seconds}


def minutes(fraction: float, seconds: float, per: int = PER) -> float:
    """Human minutes to review the share ``fraction`` of ``per`` answers at ``seconds`` per review.

    Parameters
    ----------
    fraction : float
        Reviews per answer (the review fraction), or flags per image for miss review.
    seconds : float
        Seconds per review.
    per : int, default :data:`PER`
        Answers (or images) the minutes are counted for.

    Returns
    -------
    float
        ``per * fraction * seconds / 60``.
    """
    return per * fraction * seconds / 60.0


def review_order(score: ArrayLike, part: ArrayLike, rng: np.random.Generator) -> NDArray[np.int64]:
    """Indices of the answers in review order: highest risk first, kinds interleaved by within-kind rank.

    Within each part (answer kind) the answers are sorted by decreasing score, ties broken at random, and each gets
    the rank fraction ``(position + 0.5) / n_part``. The queue is sorted by rank fraction (ties between parts at
    random), so with one part it is the plain decreasing-score order, and with several it reviews the same share
    of every part at every point.

    Parameters
    ----------
    score : array_like, shape (n,)
        Risk score, larger = riskier (``gate.csv`` ``score``; not comparable between parts).
    part : array_like, shape (n,)
        The part (answer kind) of each answer.
    rng : numpy.random.Generator
        Tie-breaking random source.

    Returns
    -------
    numpy.ndarray of int64, shape (n,)
        A permutation of ``range(n)``: the first index is reviewed first.

    Raises
    ------
    ValueError
        If ``score`` and ``part`` differ in length or a score is NaN.
    """
    s, parts = np.asarray(score, dtype=np.float64), np.asarray(part)
    if s.shape != parts.shape:
        raise ValueError(f"score and part differ in shape: {s.shape} and {parts.shape}")
    if np.isnan(s).any():
        raise ValueError("a risk score is NaN")
    tie = rng.random(len(s))
    key = np.empty(len(s))
    for value in np.unique(parts):
        idx = np.flatnonzero(parts == value)
        ranked = idx[np.lexsort((tie[idx], -s[idx]))]
        key[ranked] = (np.arange(len(idx)) + 0.5) / len(idx)
    return np.lexsort((tie, key)).astype(np.int64)


def ideal_order(wrong: ArrayLike, rng: np.random.Generator) -> NDArray[np.int64]:
    """Indices with every wrong answer first (ties at random): the best any ranking could do.

    Parameters
    ----------
    wrong : array_like, shape (n,)
        1 (or True) for a wrong answer.
    rng : numpy.random.Generator
        Tie-breaking random source.

    Returns
    -------
    numpy.ndarray of int64, shape (n,)
        A permutation of ``range(n)``.
    """
    w = np.asarray(wrong, dtype=np.float64)
    return np.lexsort((rng.random(len(w)), -w)).astype(np.int64)


def residual_curve(wrong_in_order: ArrayLike, fractions: ArrayLike) -> NDArray[np.float64]:
    """Residual wrong share of the delivered set after reviewing the first ``ceil(f n)`` answers of an order.

    Parameters
    ----------
    wrong_in_order : array_like, shape (n,)
        Wrong flags (0/1) in review order.
    fractions : array_like, shape (m,)
        Review fractions in [0, 1].

    Returns
    -------
    numpy.ndarray of float64, shape (m,)
        ``(wrong among the unreviewed) / n`` per fraction; NaN for no answers.
    """
    y = np.asarray(wrong_in_order, dtype=np.float64)
    f = np.asarray(fractions, dtype=np.float64)
    n = len(y)
    if n == 0:
        return np.full(f.shape, math.nan)
    cum = np.concatenate([[0.0], np.cumsum(y)])
    k = np.clip(np.ceil(f * n - _TOL), 0, n).astype(int)
    return (cum[-1] - cum[k]) / n


def random_curve(wrong_share: float, fractions: ArrayLike) -> NDArray[np.float64]:
    """Expected residual wrong share after reviewing a random share ``f`` of the answers.

    Parameters
    ----------
    wrong_share : float
        Share of all answers that are wrong.
    fractions : array_like, shape (m,)
        Review fractions in [0, 1].

    Returns
    -------
    numpy.ndarray of float64, shape (m,)
        ``wrong_share * (1 - f)``.
    """
    return wrong_share * (1.0 - np.asarray(fractions, dtype=np.float64))


def review_needed(wrong_in_order: ArrayLike, targets: ArrayLike, weights: ArrayLike | None = None
                  ) -> NDArray[np.float64]:
    """Smallest review fraction of an order that brings the residual wrong share to at most each target.

    With weights (a bootstrap replicate: the number of times each answer's scene was drawn), an answer counts
    ``weight`` times and the order is kept; answers of the same scene drawn twice sit side by side in effect.

    Parameters
    ----------
    wrong_in_order : array_like, shape (n,)
        Wrong flags (0/1) in review order.
    targets : array_like, shape (m,)
        Residual wrong shares to reach, in [0, 1].
    weights : array_like, shape (n,), optional
        Non-negative weight of each answer; default 1.

    Returns
    -------
    numpy.ndarray of float64, shape (m,)
        Review fraction (weighted share of answers reviewed) per target: 0 when the unreviewed set already meets
        it, 1 when only reviewing everything does; NaN when the total weight is 0.
    """
    y = np.asarray(wrong_in_order, dtype=np.float64)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    cum_n = np.concatenate([[0.0], np.cumsum(w)])
    cum_wrong = np.concatenate([[0.0], np.cumsum(w * y)])
    total, bad = cum_n[-1], cum_wrong[-1]
    if total <= 0:
        return np.full(t.shape, math.nan)
    must_catch = bad - t * total                     # wrong answers the review has to find
    k = np.searchsorted(cum_wrong, must_catch - _TOL * total, side="left")
    return cum_n[np.clip(k, 0, len(y))] / total


def random_review_needed(wrong_share: ArrayLike, targets: ArrayLike) -> NDArray[np.float64]:
    """Review fraction a random order needs (in expectation) to bring the residual wrong share to each target.

    Parameters
    ----------
    wrong_share : array_like, shape (...)
        Share of all answers that are wrong (one value, or one per bootstrap replicate).
    targets : array_like, shape (m,)
        Residual wrong shares to reach.

    Returns
    -------
    numpy.ndarray of float64, shape (..., m)
        ``max(0, 1 - target / wrong_share)``; 0 where ``wrong_share`` is 0.
    """
    p = np.asarray(wrong_share, dtype=np.float64)[..., None]
    t = np.asarray(targets, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(p > t, 1.0 - t / np.where(p > 0, p, 1.0), 0.0)
    return np.clip(f, 0.0, 1.0)


def scene_draws(scenes: ArrayLike, n_boot: int, rng: np.random.Generator) -> tuple[NDArray[np.int64],
                                                                                    NDArray[np.int64]]:
    """Scene-cluster bootstrap draws: how often each scene is drawn in each replicate.

    Parameters
    ----------
    scenes : array_like, shape (n,)
        The scene of each answer.
    n_boot : int
        Replicates.
    rng : numpy.random.Generator
        Random source.

    Returns
    -------
    codes : numpy.ndarray of int64, shape (n,)
        Each answer's scene as an index into the columns of ``draws``.
    draws : numpy.ndarray of int64, shape (n_boot, k)
        Times each of the ``k`` scenes is drawn (each row sums to ``k``).
    """
    _, codes = np.unique(np.asarray(scenes).astype(str), return_inverse=True)
    k = int(codes.max()) + 1 if len(codes) else 0
    draws = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot) if k else np.zeros((n_boot, 0), dtype=np.int64)
    return codes.astype(np.int64), draws.astype(np.int64)


def item_table(gate: pd.DataFrame, answer_kind: str, tau_scale: float) -> pd.DataFrame:
    """One row per answer of one kind: wrong flags, and the risk score and gate decision of every ranked order.

    Parameters
    ----------
    gate : pandas.DataFrame
        ``gate.csv`` rows of one labeller and depth source (:data:`GATE_COLUMNS`).
    answer_kind : str
        ``"numeric"``, ``"categorical"`` or ``"combined"`` (numeric at ``tau_scale`` plus categorical).
    tau_scale : float
        Tolerance multiple of the numeric gate. Categorical answers do not depend on it (their rows are stored
        at 1 only and are used at every multiple).

    Returns
    -------
    pandas.DataFrame
        Columns ``row, scene, part`` (the answer kind), ``wrong`` (at the gate's tolerance, 0/1), ``wrong_tau``
        (at the question's own tolerance), and per ranked order ``o`` in :data:`RANKED_ORDERS`: ``score_o`` and
        ``decision_o`` (NaN / None where the audit has no rows of that model kind).

    Raises
    ------
    ValueError
        If ``answer_kind`` is unknown, the feature model has no rows, or two model kinds disagree on which
        answers are wrong.
    """
    if answer_kind == "combined":
        parts = [item_table(gate, kind, tau_scale) for kind in ("numeric", "categorical")
                 if len(gate[gate["answer_kind"] == kind])]
        if not parts:
            raise ValueError("no numeric or categorical rows in the gate table")
        return pd.concat(parts, ignore_index=True)
    if answer_kind not in ("numeric", "categorical"):
        raise ValueError(f"unknown answer kind {answer_kind!r}; known: {ANSWER_KINDS}")
    scale = tau_scale if answer_kind == "numeric" else 1.0
    rows = gate[(gate["answer_kind"] == answer_kind) & np.isclose(gate["tau_scale"].astype(float), scale)]
    ours = rows[rows["kind"] == MODEL_KIND["ours"][answer_kind]].drop_duplicates("row").set_index("row")
    if ours.empty:
        raise ValueError(f"no {answer_kind} rows of model kind {MODEL_KIND['ours'][answer_kind]!r} at "
                         f"tau_scale {scale:g}")
    table = pd.DataFrame({"row": ours.index.to_numpy(), "scene": ours["scene"].astype(str).to_numpy(),
                          "part": answer_kind, "wrong": ours["wrong_s"].to_numpy(float),
                          "wrong_tau": ours["wrong"].to_numpy(float)})
    for order in RANKED_ORDERS:
        other = rows[rows["kind"] == MODEL_KIND[order][answer_kind]].drop_duplicates("row").set_index("row")
        other = other.reindex(ours.index)
        known = other["wrong_s"].notna().to_numpy()
        if not np.array_equal(other["wrong_s"].to_numpy(float)[known], table["wrong"].to_numpy()[known]):
            raise ValueError(f"model kinds disagree on the wrong flags of {answer_kind} answers")
        table[f"score_{order}"] = other["score"].to_numpy(float)
        table[f"decision_{order}"] = other["decision"].to_numpy(object)
    return table


def review_orders(table: pd.DataFrame, rng: np.random.Generator) -> dict[str, NDArray[np.int64]]:
    """Review orders of an item table: every ranked order with a score for every answer, and the ideal order.

    The orders are drawn once per table and shared by its curves, targets and operating points, so they break
    ties the same way.

    Parameters
    ----------
    table : pandas.DataFrame
        :func:`item_table` output.
    rng : numpy.random.Generator
        Tie-breaking random source.

    Returns
    -------
    dict of str to numpy.ndarray of int64
        Order name -> permutation of the table's rows (first reviewed first). A ranked order whose model kind
        lacks rows for some answers is left out.
    """
    out: dict[str, NDArray[np.int64]] = {}
    for order in RANKED_ORDERS:
        score = table[f"score_{order}"].to_numpy(float)
        if not np.isnan(score).any():
            out[order] = review_order(score, table["part"].to_numpy(), rng)
    out["ideal"] = ideal_order(table["wrong"].to_numpy(float), rng)
    return out


def curve_table(table: pd.DataFrame, fractions: ArrayLike, orders: Mapping[str, NDArray[np.int64]]) -> pd.DataFrame:
    """Residual wrong share against review fraction, for every order.

    Parameters
    ----------
    table : pandas.DataFrame
        :func:`item_table` output.
    fractions : array_like, shape (m,)
        Review fractions in [0, 1].
    orders : mapping of str to numpy.ndarray of int64
        :func:`review_orders` of the table (random review is added here).

    Returns
    -------
    pandas.DataFrame
        Columns ``order, review_fraction, residual_wrong_share``; ``m`` rows per order of :data:`ORDERS` that
        can be drawn.
    """
    f = np.asarray(fractions, dtype=np.float64)
    wrong = table["wrong"].to_numpy(float)
    curves = {order: residual_curve(wrong[idx], f) for order, idx in orders.items()}
    curves["random"] = random_curve(float(wrong.mean()) if len(wrong) else math.nan, f)
    return pd.concat([pd.DataFrame({"order": order, "review_fraction": f, "residual_wrong_share": curves[order]})
                      for order in ORDERS if order in curves], ignore_index=True)


def target_table(table: pd.DataFrame, targets: Sequence[float], orders: Mapping[str, NDArray[np.int64]], *,
                 n_boot: int, boot_rng: np.random.Generator, level: float = 0.95) -> pd.DataFrame:
    """Review fraction each order needs to reach each residual target, with scene-cluster bootstrap intervals.

    Every replicate resamples whole scenes with replacement (:func:`scene_draws`) and recomputes the fraction for
    all orders on the same draw, so the differences to random review are paired. The orders themselves are fixed
    (ranked once on all answers).

    Parameters
    ----------
    table : pandas.DataFrame
        :func:`item_table` output.
    targets : sequence of float
        Residual wrong shares to reach.
    orders : mapping of str to numpy.ndarray of int64
        :func:`review_orders` of the table (random review is added here).
    n_boot : int
        Bootstrap replicates; 0 skips the intervals (NaN).
    boot_rng : numpy.random.Generator
        Random source of the bootstrap draws.
    level : float, default 0.95
        Two-sided level of the percentile intervals.

    Returns
    -------
    pandas.DataFrame
        One row per target and order: ``target, order, review_fraction, lo, hi, full_review_needed`` (only
        reviewing every answer reaches the target), ``minus_random`` (review fraction minus random's) with
        ``minus_random_lo``, ``minus_random_hi`` (NaN for random itself).
    """
    t = np.asarray(targets, dtype=np.float64)
    wrong = table["wrong"].to_numpy(float)
    point = {order: review_needed(wrong[idx], t) for order, idx in orders.items()}
    point["random"] = random_review_needed(float(wrong.mean()), t)
    boot: dict[str, NDArray[np.float64]] = {}
    if n_boot > 0:
        codes, draws = scene_draws(table["scene"].to_numpy(), n_boot, boot_rng)
        boot = {order: np.empty((n_boot, len(t))) for order in orders}
        share = np.empty(n_boot)
        for b in range(n_boot):
            weight = draws[b][codes].astype(np.float64)
            share[b] = float(weight @ wrong / weight.sum())
            for order, idx in orders.items():
                boot[order][b] = review_needed(wrong[idx], t, weight[idx])
        boot["random"] = random_review_needed(share, t)
    q = [(1 - level) / 2, (1 + level) / 2]
    rows: list[dict[str, Any]] = []
    for j, target in enumerate(t):
        for order in ORDERS:
            if order not in point:
                continue
            f = float(point[order][j])
            row: dict[str, Any] = {"target": float(target), "order": order, "review_fraction": f,
                                   "lo": math.nan, "hi": math.nan, "full_review_needed": bool(f >= 1 - _TOL),
                                   "minus_random": math.nan if order == "random" else f - float(point["random"][j]),
                                   "minus_random_lo": math.nan, "minus_random_hi": math.nan}
            if boot:
                row["lo"], row["hi"] = (float(v) for v in np.quantile(boot[order][:, j], q))
                if order != "random":
                    diff = boot[order][:, j] - boot["random"][:, j]
                    row["minus_random_lo"], row["minus_random_hi"] = (float(v) for v in np.quantile(diff, q))
            rows.append(row)
    return pd.DataFrame(rows)


def operating_point(table: pd.DataFrame, order: str, epsilon: float, ranked: NDArray[np.int64] | None = None
                    ) -> dict[str, Any] | None:
    """The certified gate's operating point under the review policy: review everything it does not accept.

    Parameters
    ----------
    table : pandas.DataFrame
        :func:`item_table` output.
    order : str
        Ranked order whose gate decisions are used (:data:`RANKED_ORDERS`).
    epsilon : float
        The gate's certified bound on ``P(wrong | accepted)``.
    ranked : numpy.ndarray of int64, shape (n,), optional
        The order's review order (:func:`review_orders`), for ``ranked_review_fraction``; NaN without it.

    Returns
    -------
    dict or None
        ``n, n_wrong, wrong_share, n_accept, accepted_share, wrong_among_accepted`` (NaN with nothing accepted),
        ``review_fraction`` (``1 - accepted_share``), ``residual_wrong_share`` (wrong accepted / all),
        ``certified_bound`` (``epsilon * accepted_share``), ``within_bound``, ``wrong_among_accepted_tau`` and
        ``residual_wrong_share_tau`` (wrong at the question's own tolerance; differs from the above at
        ``tau_scale`` 2), ``random_review_fraction`` and ``ranked_review_fraction`` (the review fractions a
        random order and the same model's ranking need to reach the same residual). None if the order has no
        decision for some answer.
    """
    decision = table[f"decision_{order}"]
    if decision.isna().any():
        return None
    wrong, wrong_tau = table["wrong"].to_numpy(float), table["wrong_tau"].to_numpy(float)
    accept = (decision == "accept").to_numpy()
    n, n_accept = len(table), int(accept.sum())
    residual = float(wrong[accept].sum() / n)
    ranked_fraction = review_needed(wrong[ranked], [residual])[0] if ranked is not None else math.nan
    return {"n": n, "n_wrong": int(wrong.sum()), "wrong_share": float(wrong.mean()), "n_accept": n_accept,
            "accepted_share": n_accept / n,
            "wrong_among_accepted": float(wrong[accept].mean()) if n_accept else math.nan,
            "review_fraction": 1.0 - n_accept / n, "residual_wrong_share": residual,
            "certified_bound": epsilon * n_accept / n, "within_bound": bool(residual <= epsilon * n_accept / n + _TOL),
            "wrong_among_accepted_tau": float(wrong_tau[accept].mean()) if n_accept else math.nan,
            "residual_wrong_share_tau": float(wrong_tau[accept].sum() / n),
            "random_review_fraction": float(random_review_needed(float(wrong.mean()), [residual])[0]),
            "ranked_review_fraction": float(ranked_fraction)}


def labeller_effort(gate: pd.DataFrame, labels: Mapping[str, Any], *, epsilon: float,
                    targets: Sequence[float] = DEFAULT_TARGETS, seconds: Sequence[float] = DEFAULT_SECONDS,
                    tau_scales: Sequence[float] = (1.0, 2.0), n_boot: int = 1000, seed: int = 0,
                    fractions: ArrayLike | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Review curves, targets and certified operating points of one labeller, per answer kind and tolerance.

    Parameters
    ----------
    gate : pandas.DataFrame
        ``gate.csv`` rows of one labeller and one depth source (:data:`GATE_COLUMNS`).
    labels : mapping of str to Any
        Identifying columns put in front of every output row (e.g. ``setting, labeller, dataset, source``); their
        values also name the random streams, so every labeller gets its own.
    epsilon : float
        The gate's certified bound.
    targets : sequence of float, default :data:`DEFAULT_TARGETS`
        Residual wrong shares to reach.
    seconds : sequence of float, default :data:`DEFAULT_SECONDS`
        Seconds per review, for the minute columns.
    tau_scales : sequence of float, default (1.0, 2.0)
        Tolerance multiples of the numeric gate; those with no numeric rows are skipped.
    n_boot : int, default 1000
        Scene-cluster bootstrap replicates of the targets.
    seed : int, default 0
        Seed of tie-breaking and the bootstrap.
    fractions : array_like, optional
        Review fractions of the curves; default :data:`N_GRID` points evenly spaced on [0, 1].

    Returns
    -------
    curves : pandas.DataFrame
        ``labels`` columns, ``answer_kind, tau_scale, n_answers, wrong_share``, then :func:`curve_table` columns.
    targets : pandas.DataFrame
        ``labels`` columns, ``answer_kind, tau_scale, n_answers, wrong_share``, :func:`target_table` columns,
        minutes per :data:`PER` answers at the point estimate and both interval ends (``minutes_<t>s``,
        ``minutes_<t>s_lo``, ``minutes_<t>s_hi``) and for reviewing everything (``minutes_all_<t>s``).
    operating_points : pandas.DataFrame
        ``labels`` columns, ``answer_kind, tau_scale, order, model_kind, epsilon``, :func:`operating_point`
        columns, and minutes per :data:`PER` answers for the gate (``minutes_<t>s``), a random order reaching the
        same residual (``random_minutes_<t>s``) and reviewing everything (``minutes_all_<t>s``).
    """
    grid = np.linspace(0.0, 1.0, N_GRID) if fractions is None else np.asarray(fractions, dtype=np.float64)
    name = "/".join(str(v) for v in labels.values())
    stored = [float(s) for s in gate.loc[gate["answer_kind"] == "numeric", "tau_scale"].unique()]
    scales = [float(s) for s in tau_scales if stored and np.isclose(s, stored).any()]
    has_categorical = bool((gate["answer_kind"] == "categorical").any())
    groups = ([("numeric", s) for s in scales] + ([("categorical", 1.0)] if has_categorical else [])
              + [("combined", s) for s in (scales or ([1.0] if has_categorical else []))])
    all_minutes = {f"minutes_all_{t:g}s": minutes(1.0, t) for t in seconds}
    curves, target_rows, points = [], [], []
    for kind, scale in groups:
        table = item_table(gate, kind, scale)
        salt = f"{name}/{kind}/{scale:g}"
        head = {**labels, "answer_kind": kind, "tau_scale": scale, "n_answers": len(table),
                "wrong_share": float(table["wrong"].mean())}
        orders = review_orders(table, _rng(seed, salt + "/order"))
        curves.append(curve_table(table, grid, orders).assign(**head))
        tt = target_table(table, targets, orders, n_boot=n_boot, boot_rng=_rng(seed, salt + "/boot"))
        for column in ("review_fraction", "lo", "hi"):
            suffix = "" if column == "review_fraction" else f"_{column}"
            for t in seconds:
                tt[f"minutes_{t:g}s{suffix}"] = [minutes(v, t) for v in tt[column]]
        target_rows.append(tt.assign(**head, **all_minutes))
        for order in RANKED_ORDERS:
            point = operating_point(table, order, epsilon, orders.get(order))
            if point is None:
                continue
            model = "+".join(MODEL_KIND[order][p] for p in ("numeric", "categorical") if p in set(table["part"]))
            points.append({**labels, "answer_kind": kind, "tau_scale": scale, "order": order, "model_kind": model,
                           "epsilon": epsilon, **point, **minute_columns(point["review_fraction"], seconds),
                           **minute_columns(point["random_review_fraction"], seconds, "random_minutes"),
                           **all_minutes})
    front = [*labels, "answer_kind", "tau_scale", "n_answers", "wrong_share"]

    def ordered(frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate frames with the identifying columns first.

        Parameters
        ----------
        frames : list of pandas.DataFrame
            Frames sharing the identifying columns.

        Returns
        -------
        pandas.DataFrame
            The frames stacked, identifying columns first.
        """
        if not frames:
            return pd.DataFrame(columns=front)
        out = pd.concat(frames, ignore_index=True)
        return out[front + [c for c in out.columns if c not in front]]

    return ordered(curves), ordered(target_rows), pd.DataFrame(points)


def miss_review_table(flag_metrics: pd.DataFrame, seconds: Sequence[float] = DEFAULT_SECONDS,
                      select: Sequence[tuple[str, str]] | None = None, per: int = PER) -> pd.DataFrame:
    """Review cost and yield of the miss flags of one miss audit, per flag kind, flagged labeller and witness.

    Only the ``by == "all"`` rows (every flag of a kind, no score or support filter) are used. Reviewing a flag
    means a human checks the flagged image region for an unlabelled object; a flag that hits a miss recovers it.

    Parameters
    ----------
    flag_metrics : pandas.DataFrame
        ``flag_metrics.csv`` of ``scripts/run_miss_audit.py`` (:data:`MISS_COLUMNS` and ``by``).
    seconds : sequence of float, default :data:`DEFAULT_SECONDS`
        Seconds per reviewed flag.
    select : sequence of tuple of (str, str), optional
        ``(labeller, source)`` pairs of the flagged labeller to keep; default every row.
    per : int, default :data:`PER`
        Images the counts and minutes are given for.

    Returns
    -------
    pandas.DataFrame
        :data:`MISS_COLUMNS` (``labeller`` renamed ``flagged_labeller``, ``source`` renamed ``flagged_source``)
        plus ``misses_per_<per>_images``, ``flags_per_<per>_images``, ``recovered_per_<per>_images`` (distinct
        misses some flag hits), ``recovered_clear_per_<per>_images`` (of them, clearly visible),
        ``flags_per_recovered`` (flags reviewed per miss recovered), ``minutes_<t>s`` (per ``per`` images) and
        ``minutes_per_recovered_<t>s``. Images are the views the flag kind covers.

    Raises
    ------
    ValueError
        If a required column is missing.
    """
    missing = [c for c in (*MISS_COLUMNS, "by") if c not in flag_metrics.columns]
    if missing:
        raise ValueError(f"flag_metrics lacks columns {missing}")
    rows = flag_metrics[flag_metrics["by"] == "all"]
    if select is not None:
        wanted = {(str(a), str(b)) for a, b in select}
        rows = rows[[(str(a), str(b)) in wanted for a, b in zip(rows["labeller"], rows["source"])]]
    out = rows[list(MISS_COLUMNS)].rename(columns={"labeller": "flagged_labeller", "source": "flagged_source"})
    out = out.reset_index(drop=True)
    views = out["n_views"].astype(float).where(out["n_views"].astype(float) > 0)
    recovered = out["n_missed_flagged"].astype(float)
    out[f"misses_per_{per}_images"] = per * out["n_missed"].astype(float) / views
    out[f"flags_per_{per}_images"] = per * out["n_flags"].astype(float) / views
    out[f"recovered_per_{per}_images"] = per * recovered / views
    out[f"recovered_clear_per_{per}_images"] = per * out["n_missed_clear_flagged"].astype(float) / views
    out["flags_per_recovered"] = out["n_flags"].astype(float) / recovered.where(recovered > 0)
    for t in seconds:
        out[f"minutes_{t:g}s"] = [minutes(float(v), t, per) for v in out["flags_per_view"].astype(float)]
    for t in seconds:
        out[f"minutes_per_recovered_{t:g}s"] = out["flags_per_recovered"] * t / 60.0
    return out
