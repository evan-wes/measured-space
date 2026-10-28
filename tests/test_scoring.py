"""Tests for scoring: metrics on hand-made data, the delta-delta bootstrap, labeller imitation, label ceiling,
and oracle-only scoring of missed questions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from measured_space.eval_vlm import FORCED_METHOD
from measured_space.qa_schema import QAItem, QuestionSpec
from measured_space.scoring import (cluster_bootstrap_difference, delta_delta, imitation_rate, label_ceiling,
                                    label_ceiling_table, label_table, metrics, missed_vs_answerable,
                                    parse_method_table, predictions_frame, score, score_oracle, two_by_two)


def _spec(name: str, answer_type: str, scene: str = "s0") -> QuestionSpec:
    """A minimal question; ``name`` makes its id distinct.

    Parameters
    ----------
    name : str
        Used as the frame name, so each name gives a distinct id.
    answer_type : str
        ``"number"``, ``"choice"`` or ``"count"``; also used as the question type.
    scene : str
        Scene id.

    Returns
    -------
    QuestionSpec
        Numbers in metres with a 0.1 m / 10 % tolerance; choices ``("left", "right")``.
    """
    return QuestionSpec("t_v1", answer_type, "synthetic", scene, "cam0", name, "img.png", (), "q?", answer_type,
                        unit="m" if answer_type == "number" else None,
                        choices=("left", "right") if answer_type == "choice" else None,
                        tolerance={"abs": 0.1, "rel": 0.1} if answer_type == "number" else {})


def _scored(rows: list[tuple[QuestionSpec, object, object, object]], accepted: list[str] | None = None,
            model: str = "m") -> pd.DataFrame:
    """Score one model's predictions through the public pipeline.

    Parameters
    ----------
    rows : list of tuple
        ``(spec, oracle answer, auto answer, prediction)`` per question; prediction None is a parse failure.
    accepted : list of str or None
        Gate-accepted ids (None: all accepted).
    model : str
        Model name.

    Returns
    -------
    pandas.DataFrame
        Output of :func:`measured_space.scoring.score`.
    """
    labels = label_table([QAItem(s, o, "oracle") for s, o, _, _ in rows],
                         [QAItem(s, a, "auto") for s, _, a, _ in rows], accepted)
    preds = predictions_frame([{"model": model, "id": s.id, "parsed": p} for s, _, _, p in rows])
    return score(labels, preds)


def test_metrics_and_two_by_two_on_hand_made_data() -> None:
    """Accuracy with and without parse failures, median relative error, and the gate-accepted cell."""
    n1, n2, c1, k1 = _spec("n1", "number"), _spec("n2", "number"), _spec("c1", "choice"), _spec("k1", "count")
    scored = _scored([(n1, 2.0, 2.5, 2.4),          # wrong vs oracle (0.4 > 0.2), right vs auto
                      (n2, 1.0, 1.05, 1.0),         # right vs both
                      (c1, "left", "left", None),   # parse failure: wrong vs both
                      (k1, 3, 2, 3)],               # right vs oracle, wrong vs auto
                     accepted=[n2.id, c1.id])
    table = metrics(scored).set_index(["source", "qtype"])
    oracle, auto = table.loc[("oracle", "all")], table.loc[("auto", "all")]
    assert (oracle.n, oracle.n_parsed, oracle.parse_failure_rate) == (4, 3, 0.25)
    assert oracle.accuracy == 0.5 and oracle.accuracy_parsed == pytest.approx(2 / 3)
    assert oracle.median_rel_error == pytest.approx(0.1)                  # median of 0.2 and 0.0
    assert auto.accuracy == 0.5 and auto.median_rel_error == pytest.approx((0.1 / 2.5 + 0.05 / 1.05) / 2)
    assert table.loc[("oracle", "number")].accuracy == 0.5

    cells = two_by_two(scored).set_index(["item_set", "source", "qtype"])
    assert len(cells.xs("all", level="qtype")) == 4                       # 2 item sets x 2 sources
    accepted = cells.loc[("accepted", "oracle", "all")]
    assert (accepted.n, accepted.accuracy, accepted.accuracy_parsed) == (2, 0.5, 1.0)


def test_delta_delta_bootstrap_covers_known_truth_and_is_seeded() -> None:
    """Synthetic scenes with a known population delta-delta of -0.15 and shared per-scene shifts."""
    rng = np.random.default_rng(7)
    p = {("a", "oracle"): 0.70, ("b", "oracle"): 0.60, ("a", "auto"): 0.60, ("b", "auto"): 0.65}
    truth = (p["a", "auto"] - p["b", "auto"]) - (p["a", "oracle"] - p["b", "oracle"])
    rows = []
    for s in range(40):
        shift = rng.uniform(-0.1, 0.1)                                  # scene effect, same for every cell
        for i in range(25):
            for (model, source), prob in p.items():
                rows.append({"model": model, "id": f"{s}-{i}", "dataset": "synthetic", "scene": f"s{s}",
                             "qtype": "q0" if i % 2 else "q1", "source": source, "accepted": True,
                             "label_ok": True, "correct": bool(rng.random() < prob + shift)})
    scored = pd.DataFrame(rows)
    result = delta_delta(scored, "a", "b", n_boot=2000, seed=3).set_index("qtype")
    overall = result.loc["all"]

    wide = scored.pivot_table(index="id", columns=["model", "source"], values="correct", aggfunc="first")
    by_hand = ((wide["a", "auto"].astype(float) - wide["b", "auto"]) - (wide["a", "oracle"].astype(float)
                                                                         - wide["b", "oracle"])).mean()
    assert overall.delta_delta == pytest.approx(by_hand)
    assert overall.ci_low < truth < overall.ci_high
    assert (overall.n_items, overall.n_scenes) == (1000, 40) and set(result.index) == {"all", "q0", "q1"}
    pd.testing.assert_frame_equal(delta_delta(scored, "a", "b", n_boot=2000, seed=3).set_index("qtype"),
                                  result)
    assert delta_delta(scored, "a", "b", n_boot=2000, seed=4).ci_low.iloc[0] != overall.ci_low


def test_imitation_rate_on_synthetic_label_errors() -> None:
    """Of 21 wrong auto labels, the model copies 14; an answer right against both labels is not copying."""
    rows = [(_spec(f"e{i}", "choice"), "left", "right", "right" if i < 14 else "left") for i in range(20)]
    rows += [(_spec(f"ok{i}", "choice"), "left", "left", "right") for i in range(10)]   # no label error
    rows += [(_spec("n", "number"), 2.0, 2.25, 2.12)]   # label off by 0.25 m; 2.12 within tolerance of both
    result = imitation_rate(_scored(rows)).set_index("qtype")
    overall = result.loc["all"]
    assert (overall.n_label_errors, overall.n_imitated) == (21, 14)
    assert overall.imitation_rate == pytest.approx(14 / 21)
    assert overall.oracle_correct_rate == pytest.approx(7 / 21)
    assert result.loc["number"].imitation_rate == 0.0


def test_label_ceiling_formula() -> None:
    """2 Phi(tau / sigma) - 1, with tau the larger tolerance bound at the reference value."""
    near, far = _spec("near", "number"), _spec("far", "number")
    assert label_ceiling(near, 0.1, 0.5) == pytest.approx(2 * norm.cdf(1.0) - 1)    # tau = abs 0.1
    assert label_ceiling(far, 0.1, 2.0) == pytest.approx(2 * norm.cdf(2.0) - 1)     # tau = 10 % of 2 m
    assert label_ceiling(near, 0.0, 0.5) == 1.0
    with pytest.raises(ValueError):
        label_ceiling(near, -0.1, 0.5)
    table = label_ceiling_table([QAItem(near, 0.5, "auto", extra={"sigma": 0.1}), QAItem(far, 2.0, "auto"),
                                 QAItem(_spec("c", "choice"), "left", "auto", extra={"sigma": 0.1})])
    assert list(table["id"]) == [near.id] and table.attrs["n_without_sigma"] == 1
    assert table["tau"].iloc[0] == pytest.approx(0.1)


def test_forced_answers_count_by_default_and_can_be_excluded() -> None:
    """A correct forced answer counts as an answer unless excluded; its share is reported either way."""
    a, b = _spec("a", "number"), _spec("b", "number")
    labels = label_table([QAItem(a, 2.0, "oracle"), QAItem(b, 1.0, "oracle")],
                         [QAItem(a, 2.0, "auto"), QAItem(b, 1.0, "auto")])
    preds = predictions_frame([{"model": "m", "id": a.id, "parsed": 2.0, "parse_method": "json"},
                               {"model": "m", "id": b.id, "parsed": 1.0, "parse_method": FORCED_METHOD,
                                "forced": True}])
    scored = score(labels, preds)
    default = metrics(scored).set_index(["source", "qtype"]).loc[("oracle", "all")]
    assert (default.accuracy, default.parse_failure_rate, default.forced_rate) == (1.0, 0.0, 0.5)
    strict = metrics(scored, include_forced=False).set_index(["source", "qtype"]).loc[("oracle", "all")]
    assert (strict.accuracy, strict.parse_failure_rate, strict.forced_rate) == (0.5, 0.5, 0.5)
    methods = parse_method_table(scored).set_index(["source", "parse_method"]).loc["oracle"]
    assert methods.loc[FORCED_METHOD, "share"] == 0.5 and methods.loc["json", "accuracy"] == 1.0


def test_score_oracle_scores_questions_without_auto_labels() -> None:
    """Oracle-only scoring matches the oracle half of :func:`score`; parse failures are wrong."""
    a, b, c = _spec("a", "number"), _spec("b", "choice"), _spec("c", "number")
    oracle = [QAItem(a, 2.0, "oracle"), QAItem(b, "left", "oracle"), QAItem(c, 1.0, "oracle")]
    preds = predictions_frame([{"model": "m", "id": a.id, "parsed": 2.1}, {"model": "m", "id": b.id, "parsed": None}])
    scored = score_oracle(oracle, preds).set_index("id")
    assert set(scored.index) == {a.id, b.id} and set(scored["source"]) == {"oracle"}
    assert scored.loc[a.id, "correct"] and not scored.loc[b.id, "correct"] and not scored.loc[b.id, "parsed"]
    assert scored.loc[a.id, "rel_error"] == pytest.approx(0.05)
    assert score_oracle(oracle, preds).attrs["n_unanswered"] == {"m": 1}
    both = score(label_table(oracle[:2], [QAItem(a, 2.0, "auto"), QAItem(b, "left", "auto")]), preds)
    oracle_half = both[both["source"] == "oracle"].set_index("id")
    pd.testing.assert_series_equal(oracle_half["correct"], scored["correct"], check_names=False)


def test_missed_vs_answerable_by_hand_and_bootstrap_formula() -> None:
    """Accuracies and their difference by hand; types without missed questions stay out of "all"."""
    rows = []
    for k in range(8):                                   # answerable q0: all correct, scenes s0..s3
        rows.append({"id": f"a{k}", "scene": f"s{k % 4}", "qtype": "q0", "correct": True, "missed": False})
    for k in range(4):                                   # missed q0: half correct
        rows.append({"id": f"m{k}", "scene": f"s{k % 4}", "qtype": "q0", "correct": k < 2, "missed": True})
    for k in range(4):                                   # counts: answerable only, all wrong
        rows.append({"id": f"c{k}", "scene": f"s{k % 4}", "qtype": "count", "correct": False, "missed": False})
    scored = pd.DataFrame(rows).assign(model="m", dataset="d")
    table = missed_vs_answerable(scored, n_boot=500, seed=1).set_index("qtype")
    assert list(table.index) == ["all", "q0"]                                   # no row for counts
    overall = table.loc["all"]
    assert (overall.n_missed, overall.n_answerable, overall.n_scenes_missed, overall.n_scenes) == (4, 8, 4, 4)
    assert (overall.acc_missed, overall.acc_answerable, overall.answerable_minus_missed) == (0.5, 1.0, 0.5)
    assert overall.ci_low <= 0.5 <= overall.ci_high and overall.n_boot_valid == 500
    pd.testing.assert_frame_equal(missed_vs_answerable(scored, n_boot=500, seed=1).set_index("qtype"), table)

    values = np.array([1.0, 0.0, 1.0, 1.0])              # cluster A: two missed; cluster B: two answerable
    first = np.array([True, True, False, False])
    clusters = np.array(["A", "A", "B", "B"])
    boot = cluster_bootstrap_difference(values, first, clusters, 200, np.random.default_rng(5))
    draws = np.random.default_rng(5).multinomial(2, [0.5, 0.5], size=200)
    both = (draws == 1).all(axis=1)                      # only replicates drawing both clusters have both groups
    assert np.isnan(boot[~both]).all() and np.allclose(boot[both], 1.0 - 0.5)
