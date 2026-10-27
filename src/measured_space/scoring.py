"""Score model answers against oracle and automatic labels, and measure how label noise moves conclusions.

The analysis follows the evaluation design in ``docs/LOG.md``:

- Every answer is scored twice, against the oracle label and against the automatic label of the same
  question, always with :func:`measured_space.qa_schema.is_correct`.
- **2x2 design.** Item set (all questions, or those the label gate accepted) by label source (oracle or
  auto).
- **Delta-delta.** ``(acc_A - acc_B)_auto - (acc_A - acc_B)_oracle``: how much automatic labels shift
  the measured gap between two models. Its interval comes from a **scene-cluster bootstrap**: questions
  about one scene share objects, a capture and its label errors, so they are not independent, and
  resampling questions one by one would make the interval too narrow.
- **Labeller imitation.** Among questions whose automatic label is wrong, how often a model agrees with
  the wrong label rather than with the truth.
- **Label ceiling.** For a numeric label with standard deviation sigma, the probability that a perfect
  model is scored correct against it.

Parse failures (``None`` predictions) count as wrong. Accuracy over parsed answers only is reported next
to it. An answer forced after a reasoning model ran out of budget (see :mod:`measured_space.eval_vlm`)
counts as an answer by default. Its share is reported, and :func:`count_forced_as_failures` turns it into
a parse failure for a strict reading. Every output is a tidy DataFrame: one row per group, one column per
quantity.
"""

from __future__ import annotations

import math
import zlib
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from measured_space.qa_schema import QAItem, QuestionSpec, is_correct

SOURCES = ("oracle", "auto")
ITEM_SETS = ("all", "accepted")
BLIND_SUFFIX = "+blind"


# --------------------------------------------------------------------------------------------- tables


def label_table(oracle: Iterable[QAItem], auto: Iterable[QAItem],
                accepted_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """One row per question that has both an oracle and an automatic label.

    Parameters
    ----------
    oracle : iterable of QAItem
        Ground-truth labels (``source == "oracle"``).
    auto : iterable of QAItem
        Automatic labels (``source == "auto"``) for the same specs.
    accepted_ids : iterable of str or None
        Ids the label gate accepted. None means no gate: every question counts as accepted.

    Returns
    -------
    pandas.DataFrame
        Columns ``id, dataset, scene, qtype, answer_type, template_id, oracle, auto, label_ok, accepted,
        spec``. ``label_ok`` is ``is_correct(spec, auto, oracle)``. ``spec`` holds the
        :class:`QuestionSpec`. ``attrs`` records ``n_oracle_only`` and ``n_auto_only``: questions dropped
        because one source has no label for them.

    Raises
    ------
    ValueError
        If an item has the wrong source or an id repeats within a source.
    """
    by_oracle, by_auto = _by_id(oracle, "oracle"), _by_id(auto, "auto")
    accepted = None if accepted_ids is None else set(accepted_ids)
    ids = [i for i in by_oracle if i in by_auto]
    specs = [by_oracle[i].spec for i in ids]
    oracle_answers = [by_oracle[i].answer for i in ids]
    auto_answers = [by_auto[i].answer for i in ids]
    table = pd.DataFrame({
        "id": ids,
        "dataset": [s.dataset for s in specs],
        "scene": [s.scene for s in specs],
        "qtype": [s.qtype for s in specs],
        "answer_type": [s.answer_type for s in specs],
        "template_id": [s.template_id for s in specs],
        "oracle": pd.Series(oracle_answers, dtype=object),
        "auto": pd.Series(auto_answers, dtype=object),
        "label_ok": [is_correct(s, a, o) for s, a, o in zip(specs, auto_answers, oracle_answers)],
        "accepted": [accepted is None or i in accepted for i in ids],
        "spec": pd.Series(specs, dtype=object),
    })
    table.attrs.update(n_oracle_only=len(set(by_oracle) - set(by_auto)),
                       n_auto_only=len(set(by_auto) - set(by_oracle)))
    return table


def predictions_frame(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Parsed answers from prediction records (as written by :func:`measured_space.eval_vlm.run_predictions`).

    A blind (text-only) run is named ``<model>+blind``, so it can be compared like any other model.

    Parameters
    ----------
    records : iterable of dict
        Prediction records with ``id``, ``parsed``, ``model`` (or ``model_id``), and optionally ``blind``,
        ``parse_method`` and ``forced``.

    Returns
    -------
    pandas.DataFrame
        Columns ``model, id, prediction, parse_method, forced``. ``prediction`` is an object column where
        None is a parse failure. ``forced`` marks answers forced after cut-off reasoning.

    Raises
    ------
    ValueError
        If a model has two predictions for one id.
    """
    records = list(records)
    frame = pd.DataFrame({"model": [_model_name(r) for r in records], "id": [r["id"] for r in records],
                          "prediction": pd.Series([r.get("parsed") for r in records], dtype=object),
                          "parse_method": [r.get("parse_method") for r in records],
                          "forced": [bool(r.get("forced", False)) for r in records]})
    duplicated = frame.duplicated(["model", "id"])
    if duplicated.any():
        first = frame[duplicated].iloc[0]
        raise ValueError(f"{duplicated.sum()} duplicate predictions, e.g. {first.model} on {first.id}")
    return frame


def score(labels: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Score every prediction against both label sources.

    Parameters
    ----------
    labels : pandas.DataFrame
        Output of :func:`label_table`.
    predictions : pandas.DataFrame
        Output of :func:`predictions_frame`.

    Returns
    -------
    pandas.DataFrame
        One row per (model, question, source). Columns ``model, id, dataset, scene, qtype, answer_type,
        accepted, label_ok, source, reference, prediction, parse_method, forced, parsed, correct,
        rel_error``. ``rel_error`` is
        ``|prediction - reference| / |reference|`` for parsed numeric answers (NaN otherwise, or when the
        reference is 0). Questions a model was not asked are left out; ``attrs["n_unanswered"]`` maps
        each model to how many labelled questions it has no prediction for.
    """
    merged = predictions.merge(labels, on="id", how="inner")
    labelled = set(labels["id"])
    answered = merged.groupby("model")["id"].nunique()
    parts = []
    for source in SOURCES:
        part = merged.drop(columns=[s for s in SOURCES if s != source]).rename(columns={source: "reference"})
        part.insert(0, "source", source)
        part["parsed"] = [not _missing(p) for p in part["prediction"]]
        part["correct"] = [is_correct(s, None if _missing(p) else p, r)
                           for s, p, r in zip(part["spec"], part["prediction"], part["reference"])]
        part["rel_error"] = [_relative_error(t, p, r)
                             for t, p, r in zip(part["answer_type"], part["prediction"], part["reference"])]
        parts.append(part)
    columns = ["model", "id", "dataset", "scene", "qtype", "answer_type", "accepted", "label_ok", "source",
               "reference", "prediction", "parse_method", "forced", "parsed", "correct", "rel_error"]
    scored = pd.concat(parts, ignore_index=True)[columns]
    scored.attrs["n_unanswered"] = {m: len(labelled) - int(answered.get(m, 0))
                                    for m in predictions["model"].unique()}
    return scored


# --------------------------------------------------------------------------------------------- metrics


def metrics(scored: pd.DataFrame, by: Sequence[str] = ("model", "source"), *,
            include_forced: bool = True) -> pd.DataFrame:
    """Accuracy, parse failures and numeric error, per group and question type.

    Parameters
    ----------
    scored : pandas.DataFrame
        Output of :func:`score` (or a subset of its rows).
    by : sequence of str
        Grouping columns; results are further split by ``qtype``, with ``qtype == "all"`` pooling every
        type.
    include_forced : bool
        Count answers forced after cut-off reasoning as answers (default). If False they count as parse
        failures (see :func:`count_forced_as_failures`).

    Returns
    -------
    pandas.DataFrame
        Columns ``*by, qtype, n, n_parsed, parse_failure_rate, forced_rate, accuracy, accuracy_parsed,
        median_rel_error``. ``accuracy`` counts parse failures as wrong; ``accuracy_parsed`` leaves them
        out (NaN if nothing parsed). ``forced_rate`` is the share of questions whose answer was forced,
        whether or not those answers are counted. ``median_rel_error`` is over parsed numeric answers (NaN
        if none).
    """
    by = list(by)
    if not include_forced:
        scored = count_forced_as_failures(scored)
    rows = []
    for keys, group in scored.groupby(by + ["qtype"], sort=True):
        rows.append({**dict(zip(by + ["qtype"], keys)), **_summary(group)})
    for keys, group in scored.groupby(by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(by, keys)), "qtype": "all", **_summary(group)})
    columns = by + ["qtype", "n", "n_parsed", "parse_failure_rate", "forced_rate", "accuracy", "accuracy_parsed",
                    "median_rel_error"]
    return pd.DataFrame(rows, columns=columns).sort_values(by + ["qtype"], ignore_index=True)


def two_by_two(scored: pd.DataFrame, *, include_forced: bool = True) -> pd.DataFrame:
    """The evaluation 2x2: item set (all, gate-accepted) by label source (oracle, auto), per model.

    Parameters
    ----------
    scored : pandas.DataFrame
        Output of :func:`score`.
    include_forced : bool
        Passed to :func:`metrics`.

    Returns
    -------
    pandas.DataFrame
        :func:`metrics` columns, grouped by ``model, item_set, source`` and split by ``qtype``.
    """
    frames = [metrics(_item_set(scored, item_set), include_forced=include_forced).assign(item_set=item_set)
              for item_set in ITEM_SETS]
    table = pd.concat(frames, ignore_index=True)
    first = ["model", "item_set", "source", "qtype"]
    return table[first + [c for c in table.columns if c not in first]].sort_values(first, ignore_index=True)


def count_forced_as_failures(scored: pd.DataFrame) -> pd.DataFrame:
    """A strict reading: answers forced after cut-off reasoning become parse failures.

    Use it before :func:`delta_delta` or :func:`imitation_rate` to repeat an analysis without forced
    answers. Every question stays in, so model comparisons remain paired.

    Parameters
    ----------
    scored : pandas.DataFrame
        Output of :func:`score`.

    Returns
    -------
    pandas.DataFrame
        A copy where forced rows have ``prediction`` None, ``parsed`` and ``correct`` False and
        ``rel_error`` NaN. ``forced`` and ``parse_method`` are kept, so they can still be counted.
    """
    out = scored.copy()
    forced = out["forced"].astype(bool)
    out["prediction"] = out["prediction"].astype(object).where(~forced, None)
    out.loc[forced, ["parsed", "correct"]] = False
    out.loc[forced, "rel_error"] = math.nan
    return out


def parse_method_table(scored: pd.DataFrame) -> pd.DataFrame:
    """How each model's answers were parsed, and how accurate each kind was.

    Parameters
    ----------
    scored : pandas.DataFrame
        Output of :func:`score`.

    Returns
    -------
    pandas.DataFrame
        One row per model, source and parse method. Columns ``model, source, parse_method, n, share,
        accuracy``: ``share`` is relative to the model's questions for that source.
    """
    table = scored.assign(parse_method=scored["parse_method"].fillna("unknown")).groupby(
        ["model", "source", "parse_method"], sort=True).agg(n=("correct", "size"), accuracy=("correct", "mean"))
    table = table.reset_index()
    table["share"] = table["n"] / table.groupby(["model", "source"])["n"].transform("sum")
    return table[["model", "source", "parse_method", "n", "share", "accuracy"]]


def label_quality(labels: pd.DataFrame) -> pd.DataFrame:
    """How often the automatic label is correct against the oracle, per item set and question type.

    Parameters
    ----------
    labels : pandas.DataFrame
        Output of :func:`label_table`.

    Returns
    -------
    pandas.DataFrame
        Columns ``item_set, qtype, n, auto_label_accuracy``.
    """
    rows = []
    for item_set in ITEM_SETS:
        subset = labels if item_set == "all" else labels[labels["accepted"]]
        groups = [(q, g) for q, g in subset.groupby("qtype", sort=True)] + [("all", subset)]
        rows += [{"item_set": item_set, "qtype": q, "n": len(g),
                  "auto_label_accuracy": float(g["label_ok"].mean()) if len(g) else math.nan}
                 for q, g in groups]
    return pd.DataFrame(rows, columns=["item_set", "qtype", "n", "auto_label_accuracy"])


# --------------------------------------------------------------------------------------------- delta-delta


def delta_delta(scored: pd.DataFrame, model_a: str, model_b: str, *, item_set: str = "all",
                n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """How much automatic labels shift the accuracy gap between two models, with a scene-cluster CI.

    ``delta_delta = (acc_A - acc_B)_auto - (acc_A - acc_B)_oracle``, computed on the questions both models
    answered. Because all four accuracies share those questions, it equals the mean over questions of
    ``(c_A,auto - c_B,auto) - (c_A,oracle - c_B,oracle)`` with ``c`` in {0, 1}. Scenes (dataset and scene
    id) are resampled with replacement. The interval is the percentile interval of the bootstrap
    distribution. With few scenes this interval is coarse and tends to be too narrow.

    Parameters
    ----------
    scored : pandas.DataFrame
        Output of :func:`score`.
    model_a, model_b : str
        The two models compared (``A - B``).
    item_set : {"all", "accepted"}
        Which questions to use.
    n_boot : int
        Bootstrap replicates.
    seed : int
        Seed. Each question type draws from its own stream derived from the seed and the type's name, so
        adding a type does not change the others' intervals.
    level : float
        Confidence level of the interval.

    Returns
    -------
    pandas.DataFrame
        One row for ``qtype == "all"`` and one per question type. Columns ``model_a, model_b, item_set,
        qtype, n_items, n_scenes, acc_a_oracle, acc_b_oracle, acc_a_auto, acc_b_auto, delta_oracle,
        delta_auto, delta_delta, ci_low, ci_high, level, n_boot, seed``.

    Raises
    ------
    ValueError
        If the two models are the same, either has no scored questions, or they share none.
    """
    if model_a == model_b:
        raise ValueError("delta_delta needs two different models")
    subset = _item_set(scored, item_set)
    for model in (model_a, model_b):
        if not (subset["model"] == model).any():
            raise ValueError(f"No scored questions for model {model!r} in item set {item_set!r}")
    subset = subset[subset["model"].isin([model_a, model_b])]
    subset = subset.assign(cell=subset["model"].map({model_a: "a", model_b: "b"}) + "_" + subset["source"])
    cells = ["a_oracle", "b_oracle", "a_auto", "b_auto"]
    wide = subset.pivot_table(index=["id", "dataset", "scene", "qtype"], columns="cell", values="correct",
                              aggfunc="first")
    wide = wide.reindex(columns=cells).dropna().astype(float).reset_index()
    if wide.empty:
        raise ValueError(f"{model_a!r} and {model_b!r} share no scored questions")
    wide["cluster"] = wide["dataset"].astype(str) + "/" + wide["scene"].astype(str)
    a_o, b_o, a_a, b_a = (wide[c].to_numpy() for c in cells)
    wide["d"] = (a_a - b_a) - (a_o - b_o)
    rows = []
    for qtype, group in [("all", wide)] + list(wide.groupby("qtype", sort=True)):
        rng = np.random.default_rng([seed, zlib.crc32(str(qtype).encode())])
        boot = cluster_bootstrap(group["d"].to_numpy(), group["cluster"].to_numpy(), n_boot, rng)
        ci_low, ci_high = np.quantile(boot, [(1 - level) / 2, (1 + level) / 2])
        acc = [float(group[c].mean()) for c in cells]
        rows.append({"model_a": model_a, "model_b": model_b, "item_set": item_set, "qtype": qtype,
                     "n_items": len(group), "n_scenes": group["cluster"].nunique(),
                     "acc_a_oracle": acc[0], "acc_b_oracle": acc[1], "acc_a_auto": acc[2], "acc_b_auto": acc[3],
                     "delta_oracle": acc[0] - acc[1], "delta_auto": acc[2] - acc[3],
                     "delta_delta": float(group["d"].mean()), "ci_low": float(ci_low), "ci_high": float(ci_high),
                     "level": level, "n_boot": n_boot, "seed": seed})
    return pd.DataFrame(rows)


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, n_boot: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of a mean, resampling whole clusters with replacement.

    Each replicate draws as many clusters as there are, with replacement, and pools their items:
    ``sum(w_k * S_k) / sum(w_k * n_k)``, with ``w_k`` the times cluster ``k`` was drawn, ``S_k`` its sum
    and ``n_k`` its size.

    Parameters
    ----------
    values : numpy.ndarray
        Shape ``(N,)``, one value per item.
    clusters : numpy.ndarray
        Shape ``(N,)``, the cluster label of each item.
    n_boot : int
        Number of replicates.
    rng : numpy.random.Generator
        Random source.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_boot,)``, the replicated means.
    """
    _, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64))
    sizes = np.bincount(inverse).astype(np.float64)
    k = len(sums)
    draws = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot)          # (n_boot, k) times drawn
    return (draws @ sums) / (draws @ sizes)


# --------------------------------------------------------------------------------------------- label noise


def imitation_rate(scored: pd.DataFrame, *, item_set: str = "all") -> pd.DataFrame:
    """How often each model reproduces a wrong automatic label.

    Among questions whose automatic label is wrong against the oracle (``label_ok`` False), a model
    *imitates* the labeller when its answer is correct against the automatic label but wrong against the
    oracle. A tolerance-based answer can be correct against both labels; that does not count.

    Parameters
    ----------
    scored : pandas.DataFrame
        Output of :func:`score`.
    item_set : {"all", "accepted"}
        Which questions to use.

    Returns
    -------
    pandas.DataFrame
        One row per model and question type, plus ``qtype == "all"``. Columns ``model, item_set, qtype,
        n_label_errors, n_imitated, imitation_rate, n_oracle_correct, oracle_correct_rate``. Rates are
        NaN where there are no label errors.
    """
    subset = _item_set(scored, item_set)
    errors = subset[~subset["label_ok"].astype(bool)]
    wide = errors.pivot_table(index=["model", "id", "qtype"], columns="source", values="correct",
                              aggfunc="first").reset_index() if len(errors) else pd.DataFrame(
        columns=["model", "id", "qtype", *SOURCES])
    wide["imitated"] = wide["auto"].astype(bool) & ~wide["oracle"].astype(bool)
    rows = []
    for model in sorted(subset["model"].unique()):
        own = wide[wide["model"] == model]
        groups = [(q, g) for q, g in own.groupby("qtype", sort=True)] + [("all", own)]
        for qtype, group in groups:
            n = len(group)
            n_imitated, n_oracle = int(group["imitated"].sum()), int(group["oracle"].astype(bool).sum())
            rows.append({"model": model, "item_set": item_set, "qtype": qtype, "n_label_errors": n,
                         "n_imitated": n_imitated, "imitation_rate": n_imitated / n if n else math.nan,
                         "n_oracle_correct": n_oracle, "oracle_correct_rate": n_oracle / n if n else math.nan})
    return pd.DataFrame(rows, columns=["model", "item_set", "qtype", "n_label_errors", "n_imitated",
                                       "imitation_rate", "n_oracle_correct", "oracle_correct_rate"])


def tolerance_at(spec: QuestionSpec, reference: float) -> float:
    """The largest error a numeric answer may have and still be correct, at a reference value.

    :func:`~measured_space.qa_schema.is_correct` accepts an error within the absolute **or** the relative
    bound, so the tolerance is the larger of the two.

    Parameters
    ----------
    spec : QuestionSpec
        A numeric question; its ``tolerance`` holds ``abs`` (metres) and ``rel`` (fraction) bounds.
    reference : float
        The reference answer, in metres.

    Returns
    -------
    float
        The tolerance in metres; 0 if the spec has no bounds.
    """
    abs_tol, rel_tol = spec.tolerance.get("abs"), spec.tolerance.get("rel")
    bounds = [b for b in (abs_tol, None if rel_tol is None else rel_tol * abs(reference)) if b is not None]
    return float(max(bounds)) if bounds else 0.0


def label_ceiling(spec: QuestionSpec, sigma: float, reference: float) -> float:
    """Probability that a perfect model is scored correct against a noisy numeric label.

    If the label is the true value plus Gaussian noise of standard deviation ``sigma``, a model that
    answers the true value is correct when the label noise is within the tolerance ``tau``. That happens
    with probability ``2 * Phi(tau / sigma) - 1 = erf(tau / (sigma * sqrt(2)))``. ``tau`` is taken at the
    reference value. The relative bound strictly scales with the noisy label, not the truth, so this is
    a first-order approximation.

    Parameters
    ----------
    spec : QuestionSpec
        A numeric question.
    sigma : float
        Standard deviation of the label error, in metres (>= 0).
    reference : float
        The label value, in metres.

    Returns
    -------
    float
        The ceiling, in [0, 1]; 1 for an exact label (``sigma == 0``).

    Raises
    ------
    ValueError
        If ``sigma`` is negative or not finite.
    """
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma must be finite and non-negative, got {sigma}")
    if sigma == 0:
        return 1.0
    return math.erf(tolerance_at(spec, reference) / (sigma * math.sqrt(2.0)))


def label_ceiling_table(items: Iterable[QAItem], sigma_key: str = "sigma") -> pd.DataFrame:
    """The label ceiling of every numeric item that carries a sigma.

    Parameters
    ----------
    items : iterable of QAItem
        Labelled items, usually automatic ones with a predicted uncertainty in ``extra``.
    sigma_key : str
        Key of the label's standard deviation (metres) in ``item.extra``.

    Returns
    -------
    pandas.DataFrame
        Columns ``id, dataset, scene, qtype, source, reference, sigma, tau, ceiling``.
        ``attrs["n_without_sigma"]`` counts numeric items skipped for lacking a finite sigma.
    """
    rows, skipped = [], 0
    for item in items:
        if item.spec.answer_type != "number":
            continue
        sigma = item.extra.get(sigma_key)
        if sigma is None or not math.isfinite(float(sigma)):
            skipped += 1
            continue
        reference = float(item.answer)
        rows.append({"id": item.id, "dataset": item.spec.dataset, "scene": item.spec.scene,
                     "qtype": item.spec.qtype, "source": item.source, "reference": reference,
                     "sigma": float(sigma), "tau": tolerance_at(item.spec, reference),
                     "ceiling": label_ceiling(item.spec, float(sigma), reference)})
    table = pd.DataFrame(rows, columns=["id", "dataset", "scene", "qtype", "source", "reference", "sigma",
                                        "tau", "ceiling"])
    table.attrs["n_without_sigma"] = skipped
    return table


# --------------------------------------------------------------------------------------------- helpers


def _by_id(items: Iterable[QAItem], source: str) -> dict[str, QAItem]:
    """Index items of one label source by question id.

    Parameters
    ----------
    items : iterable of QAItem
        Items that should all have ``source``.
    source : str
        Expected label source.

    Returns
    -------
    dict
        Question id to item, in input order.

    Raises
    ------
    ValueError
        If an item has another source, or an id repeats.
    """
    out: dict[str, QAItem] = {}
    for item in items:
        if item.source != source:
            raise ValueError(f"Expected {source!r} items, got {item.source!r} for {item.id}")
        if item.id in out:
            raise ValueError(f"Duplicate {source} item {item.id}")
        out[item.id] = item
    return out


def _model_name(record: dict[str, Any]) -> str:
    """The model name of a prediction record, marked if the run was blind.

    Parameters
    ----------
    record : dict
        A prediction record.

    Returns
    -------
    str
        ``record["model"]`` (else ``record["model_id"]``), with ``+blind`` appended for blind runs.
    """
    name = record.get("model") or record["model_id"]
    return name + BLIND_SUFFIX if record.get("blind") else name


def _missing(value: Any) -> bool:
    """Whether a prediction is a parse failure.

    Parameters
    ----------
    value : Any
        A parsed prediction.

    Returns
    -------
    bool
        True for None and NaN.
    """
    return value is None or (isinstance(value, float) and math.isnan(value))


def _relative_error(answer_type: str, prediction: Any, reference: Any) -> float:
    """Relative error of a numeric prediction.

    Parameters
    ----------
    answer_type : str
        The question's answer type.
    prediction : Any
        Parsed prediction (None if unparsed).
    reference : Any
        Reference answer.

    Returns
    -------
    float
        ``|prediction - reference| / |reference|``; NaN for non-numeric questions, unparsed predictions
        or a zero reference.
    """
    if answer_type != "number" or _missing(prediction) or float(reference) == 0:
        return math.nan
    return abs(float(prediction) - float(reference)) / abs(float(reference))


def _summary(group: pd.DataFrame) -> dict[str, float | int]:
    """Metrics of one group of scored rows.

    Parameters
    ----------
    group : pandas.DataFrame
        Rows of :func:`score` output.

    Returns
    -------
    dict
        ``n, n_parsed, parse_failure_rate, forced_rate, accuracy, accuracy_parsed, median_rel_error``.
        ``forced_rate`` is NaN if the rows have no ``forced`` column.
    """
    n, parsed = len(group), group["parsed"].astype(bool)
    n_parsed = int(parsed.sum())
    rel = group.loc[parsed & (group["answer_type"] == "number"), "rel_error"].astype(float).dropna()
    forced = float(group["forced"].astype(bool).mean()) if "forced" in group and n else math.nan
    return {"n": n, "n_parsed": n_parsed, "parse_failure_rate": 1 - n_parsed / n if n else math.nan,
            "forced_rate": forced,
            "accuracy": float(group["correct"].mean()) if n else math.nan,
            "accuracy_parsed": float(group.loc[parsed, "correct"].mean()) if n_parsed else math.nan,
            "median_rel_error": float(rel.median()) if len(rel) else math.nan}


def _item_set(scored: pd.DataFrame, item_set: str) -> pd.DataFrame:
    """Rows of one item set.

    Parameters
    ----------
    scored : pandas.DataFrame
        Output of :func:`score`.
    item_set : {"all", "accepted"}
        Which questions to keep.

    Returns
    -------
    pandas.DataFrame
        All rows, or those whose question the gate accepted.

    Raises
    ------
    ValueError
        If ``item_set`` is not one of :data:`ITEM_SETS`.
    """
    if item_set not in ITEM_SETS:
        raise ValueError(f"item_set must be one of {ITEM_SETS}, got {item_set!r}")
    return scored if item_set == "all" else scored[scored["accepted"].astype(bool)]
