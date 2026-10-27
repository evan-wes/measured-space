"""Tests for the label audit on synthetic data with a known error structure.

Covers: the record parser on items shaped like the label pipeline's output, including the per-referent detection
score (unchanged for the depth sources, the box detector's for ``cutr_sam3``) and the pairing-ambiguity flag;
the ambiguous-pair exclusion; Wilson intervals and the conformal quantile against their definitions; recovery of a
heteroscedastic error model; conformal coverage on exchangeable scenes; the Learn-then-Test bound in repeated
draws and in the full gate; the feature model beating the constant-sigma baseline when the truth is
heteroscedastic (but not when it is not); and, for the audit's scripts, that no output folder of committed results
has a default and that the error figure draws every source.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from measured_space.audit import (AuditConfig, AuditResult, NumericPredictor, conformal_quantile, exclude_ambiguous,
                                  exclusion_record, fit_error_model, load_labels, ltt_threshold, match_ambiguity,
                                  pair_table, referent_features, run_audit, select_ridge, synthetic_pairs,
                                  wilson_interval)
from measured_space.qa_schema import QAItem, QuestionSpec, Referent, write_jsonl

FAST = AuditConfig(n_boot=200, ridge_grid=(1.0,))     # fixed ridge keeps the pipeline tests fast
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def hetero() -> AuditResult:
    """Audit of a heteroscedastic synthetic dataset with a sizeable wrong-label rate (shared by tests).

    Returns
    -------
    AuditResult
        The audit of 6 scenes x 80 questions x 3 types x 2 depth sources.
    """
    return run_audit(synthetic_pairs(n_scenes=6, n_items=80, noise_scale=1.8, seed=3), FAST)


def _spec(qtype: str, answer_type: str, frame: str, scene: str = "s0") -> QuestionSpec:
    """A minimal question about two referents.

    Parameters
    ----------
    qtype : str
        Question type.
    answer_type : str
        ``"number"`` or ``"choice"``.
    frame : str
        Frame name (makes the id distinct).
    scene : str, default "s0"
        Scene id.

    Returns
    -------
    QuestionSpec
        Numeric specs carry a 0.10 m / 10 % tolerance; choices are left/right.
    """
    refs = (Referent("a", "chair", "the chair"), Referent("b", "table", "the table"))
    return QuestionSpec(f"{qtype}_v1", qtype, "synthetic", scene, "cam0", frame, "img.png", refs, "q?", answer_type,
                        unit="m" if answer_type == "number" else None,
                        choices=("left", "right") if answer_type == "choice" else None,
                        tolerance={"abs": 0.1, "rel": 0.1} if answer_type == "number" else {})


def _referent(object_id: str, depth: float, score: float, border: float) -> dict[str, object]:
    """One referent record as the label pipeline writes it into an auto item's ``extra``.

    Parameters
    ----------
    object_id : str
        Object id.
    depth : float
        ``depth_median`` (m).
    score : float
        Detection score.
    border : float
        ``border_touch`` (0 or 1).

    Returns
    -------
    dict
        ``object_id``, ``score``, ``match_distance`` and a nested ``features`` dict.
    """
    return {"object_id": object_id, "detection": 0, "score": score, "match_distance": 0.2, "match_iou": 0.5,
            "features": {"depth_median": depth, "boundary_fraction": 0.1 * depth, "border_touch": border,
                         "depth_mad_rel": 0.01 * depth, "footprint_hull_fill": 1.0 / depth, "n_points": 1000 * depth}}


def test_pair_table_reads_errors_misses_and_worst_referent_features() -> None:
    """Errors, tolerances, flips and misses follow ``is_correct``; features take the worst referent."""
    dist, choice, missed = _spec("distance_center", "number", "f0"), _spec("left_right", "choice", "f1"), \
        _spec("size_height", "number", "f2")
    oracle = [QAItem(dist, 2.0, "oracle"), QAItem(choice, "left", "oracle"), QAItem(missed, 1.0, "oracle")]
    extra = {"referents": [_referent("a", 2.0, 0.9, 0.0), _referent("b", 4.0, 0.6, 1.0)]}
    auto = [QAItem(dist, 2.35, "auto_moge", extra=extra),
            QAItem(choice, "right", "auto_moge", extra={**extra, "margin": 0.4})]
    table = pair_table(oracle, auto, "moge").set_index("qtype")

    d = table.loc["distance_center"]
    assert d["e"] == pytest.approx(0.35) and d["rel_e"] == pytest.approx(0.175)
    assert d["tau"] == pytest.approx(0.2) and d["tau_hat"] == pytest.approx(0.235)
    assert d["wrong"] == 1.0
    assert d["log_range"] == pytest.approx(math.log(4.0))           # farthest referent
    assert d["border_touch"] == 1.0 and d["boundary_fraction"] == pytest.approx(0.4)
    assert d["hull_fill"] == pytest.approx(0.25)                     # least-filled footprint
    assert d["log_n_points"] == pytest.approx(math.log(2000.0))      # fewest points
    assert d["det_score"] == pytest.approx(0.6)                      # least confident detection
    assert d["log_value"] == pytest.approx(math.log(2.35))
    c = table.loc["left_right"]
    assert c["wrong"] == 1.0 and c["margin"] == pytest.approx(0.4)
    m = table.loc["size_height"]
    assert not m["matched"] and math.isnan(m["wrong"]) and math.isnan(m["log_range"])


def _old_det_score(extra: Any) -> float:
    """The detection-score rule before per-referent lift scores: the minimum of every score key anywhere.

    Parameters
    ----------
    extra : Any
        An auto item's ``extra``.

    Returns
    -------
    float
        The smallest finite value under ``det_score``, ``detection_score`` or ``score``; NaN if there is none.
    """
    found: list[float] = []

    def walk(node: Any) -> None:
        """Collect score values recursively.

        Parameters
        ----------
        node : Any
            A JSON-like value.
        """
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("det_score", "detection_score", "score") and isinstance(value, float) \
                        and math.isfinite(value):
                    found.append(value)
                elif isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(extra)
    return min(found) if found else math.nan


def _depth_source_extra(rng: np.random.Generator) -> dict[str, Any]:
    """A random auto item's ``extra`` as the depth sources write it: lift features without a detection score.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random source.

    Returns
    -------
    dict
        ``referents`` or ``counted`` evidence (matched or not) with SAM 3 scores, plus a ``margin``.
    """
    def evidence() -> dict[str, Any]:
        """One referent's evidence.

        Returns
        -------
        dict
            As :func:`measured_space.pipeline._evidence` writes it for a depth source.
        """
        matched = rng.random() < 0.8
        return {"object_id": "o" if matched else None, "detection": 0, "phrase": "chair",
                "score": float(rng.uniform(0.5, 1.0)), "match_distance": float(rng.uniform(0, 2)) if matched else None,
                "features": {"depth_median": float(rng.uniform(1, 9)), "boundary_fraction": float(rng.random()),
                             "border_touch": float(rng.random() < 0.3), "n_points": float(rng.integers(50, 9999)),
                             "footprint_hull_fill": float(rng.random()), "depth_mad_rel": float(rng.random())}}

    key = "counted" if rng.random() < 0.3 else "referents"
    return {key: [evidence() for _ in range(int(rng.integers(0 if key == "counted" else 1, 4)))],
            "margin": float(rng.random())}


def test_det_score_of_depth_sources_is_unchanged() -> None:
    """Without a ``det_score`` in the lift features (laser, LiDAR, MoGe, rendered depth), the feature is the
    smallest detection score anywhere in the record, exactly as before per-referent lift scores."""
    rng = np.random.default_rng(7)
    extras = [_depth_source_extra(rng) for _ in range(300)]
    extras += [{"referents": [_referent("a", 2.0, 0.9, 0.0), _referent("b", 4.0, 0.6, 1.0)]},
               {"a": {"score": 0.7, "depth_median": 2.0, "n_points": 10.0}, "b": {"detection_score": 0.4}},
               {"score": 0.8, "depth_median": 2.0, "border_touch": 1.0}, {}]
    for extra in extras:
        old, new = _old_det_score(extra), referent_features(extra)["det_score"]
        assert (math.isnan(old) and math.isnan(new)) or old == new, extra


def test_det_score_prefers_each_referents_lift_score() -> None:
    """``cutr_sam3`` stores SAM 3's score as the referent's and CuTR's in the lift: the feature is the smallest CuTR
    score, a referent without a finite lift score falls back to its own, and ``cutr_clip`` (one score) is unchanged."""
    def cutr_sam3(sam3: float, cutr: float | None) -> dict[str, Any]:
        """A ``cutr_sam3`` referent's evidence.

        Parameters
        ----------
        sam3 : float
            SAM 3's detection score.
        cutr : float or None
            CuTR's score in the lift features (None: absent).

        Returns
        -------
        dict
            The evidence.
        """
        features: dict[str, Any] = {"depth_median": 3.0, "boundary_fraction": 0.1, "border_touch": 0.0}
        if cutr is not None:
            features["det_score"] = cutr
        return {"object_id": "a", "score": sam3, "match_distance": 0.1, "features": features}

    assert referent_features({"referents": [cutr_sam3(0.95, 0.4), cutr_sam3(0.3, 0.8)]})["det_score"] == 0.4
    assert referent_features({"counted": [cutr_sam3(0.95, 0.7), cutr_sam3(0.5, None)]})["det_score"] == 0.5
    assert referent_features({"referents": [cutr_sam3(0.9, math.nan)]})["det_score"] == 0.9
    clip = {"object_id": "a", "score": 0.55, "match_distance": 0.1,
            "features": {"depth_median": 3.0, "border_touch": 0.0, "det_score": 0.55, "name_prob": 0.9}}
    assert referent_features({"referents": [clip]})["det_score"] == 0.55 == _old_det_score({"referents": [clip]})


def test_match_ambiguity_reads_the_pairing_flags() -> None:
    """Ambiguous if any pairing was; unknown (NaN) if a paired record lacks the flag; unpaired records and items
    without pairings are not ambiguous."""
    def ev(object_id: str | None, **flag: Any) -> dict[str, Any]:
        """A referent's evidence with an optional ``match_ambiguous`` flag.

        Parameters
        ----------
        object_id : str or None
            Paired oracle object (None: unpaired).
        **flag : Any
            ``match_ambiguous=...`` to store the flag.

        Returns
        -------
        dict
            The evidence.
        """
        return {"object_id": object_id, "score": 0.9, "match_distance": None if object_id is None else 0.1, **flag}

    assert match_ambiguity({"referents": [ev("a", match_ambiguous=False), ev("b", match_ambiguous=True)]}) == 1.0
    assert match_ambiguity({"referents": [ev("a", match_ambiguous=False), ev("b", match_ambiguous=False)]}) == 0.0
    assert math.isnan(match_ambiguity({"referents": [ev("a", match_ambiguous=False), ev("b")]}))       # legacy
    assert math.isnan(match_ambiguity({"referents": [ev("a", match_ambiguous=None)]}))                 # 3D assoc.
    assert match_ambiguity({"referents": [ev("a"), ev("b", match_ambiguous=True)]}) == 1.0
    assert match_ambiguity({"counted": [ev(None), ev(None, match_ambiguous=None)]}) == 0.0              # unpaired
    assert match_ambiguity({"counted": [], "n_detections_2d": 0}) == 0.0 == match_ambiguity({})


def test_exclude_ambiguous_drops_exactly_the_ambiguous_pairs(tmp_path: Path) -> None:
    """Per depth source, a pair whose auto answer rests on an ambiguous pairing is dropped (oracle and auto side);
    unknown pairs are kept and counted; unambiguous pairs and misses are kept.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Scratch folder from pytest.
    """
    specs = [_spec("size_height", "number", f"f{i}") for i in range(5)]
    write_jsonl([QAItem(s, 1.0, "oracle") for s in specs], tmp_path / "qa_oracle.jsonl")

    def extra(*flags: bool | None) -> dict[str, Any]:
        """Evidence with one paired referent per flag (None: the flag is absent, as in old labels).

        Parameters
        ----------
        *flags : bool or None
            ``match_ambiguous`` per referent.

        Returns
        -------
        dict
            ``{"referents": [...]}``.
        """
        refs = []
        for k, flag in enumerate(flags):
            ref = _referent("ab"[k], 2.0, 0.9, 0.0)
            if flag is not None:
                ref["match_ambiguous"] = flag
            refs.append(ref)
        return {"referents": refs}

    moge = [extra(False, False), extra(False, True), extra(None, False), extra(True), None]       # f4: a miss
    laser = [extra(False, False), extra(False, False), extra(False, False), extra(True, True), extra(False)]
    for name, extras in (("moge", moge), ("laser", laser)):
        write_jsonl([QAItem(s, 1.05, f"auto_{name}", extra=e) for s, e in zip(specs, extras) if e is not None],
                    tmp_path / f"qa_auto_{name}.jsonl")
    pairs = load_labels(tmp_path)
    kept, counts = exclude_ambiguous(pairs)
    dropped = set(map(tuple, pairs[["depth_source", "frame"]].to_numpy().tolist())) \
        - set(map(tuple, kept[["depth_source", "frame"]].to_numpy().tolist()))
    assert dropped == {("moge", "f1"), ("moge", "f3"), ("laser", "f3")}
    assert len(kept) == len(pairs) - 3 and kept.index.equals(pd.RangeIndex(len(kept)))
    assert kept.attrs == pairs.attrs
    miss = kept[(kept["depth_source"] == "moge") & (kept["frame"] == "f4")]
    assert len(miss) == 1 and not miss["matched"].iloc[0]                    # a miss has no pairing: kept
    assert counts == {"synthetic": {"laser": {"n_rows": 5, "n_labelled": 5, "n_dropped": 1, "n_unknown_kept": 0},
                                    "moge": {"n_rows": 5, "n_labelled": 4, "n_dropped": 2, "n_unknown_kept": 1}}}
    record = exclusion_record(counts)
    assert record["enabled"] and record["n_dropped"] == 3 and record["n_unknown_kept"] == 1
    assert record["unknown_policy"] == "kept" and exclusion_record(None) == {"enabled": False}
    unflagged, unflagged_counts = exclude_ambiguous(pairs.drop(columns="match_ambiguous"))
    assert len(unflagged) == len(pairs) and unflagged_counts["synthetic"]["moge"]["n_unknown_kept"] == 4
    synthetic = synthetic_pairs(n_scenes=2, n_items=5, seed=1)
    assert len(exclude_ambiguous(synthetic)[0]) == len(synthetic)                 # nothing ambiguous to drop


def test_load_labels_pairs_each_source_only_with_views_it_ran_on(tmp_path: Path) -> None:
    """A depth source run on one scene only does not count the other scene's questions as misses.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Scratch folder from pytest.
    """
    specs = [_spec("size_height", "number", "f0", "sA"), _spec("size_height", "number", "f0", "sB")]
    write_jsonl([QAItem(s, 1.0, "oracle") for s in specs], tmp_path / "qa_oracle.jsonl")
    write_jsonl([QAItem(s, 1.02, "auto_moge") for s in specs], tmp_path / "qa_auto_moge.jsonl")
    write_jsonl([], tmp_path / "qa_auto_rendered.jsonl")                       # ran on sA, answered nothing
    views = [{"scene": "sA", "camera_id": "cam0", "frame": "f0", "depth_sources": ["moge", "rendered"]},
             {"scene": "sB", "camera_id": "cam0", "frame": "f0", "depth_sources": ["moge"]}]
    (tmp_path / "views.jsonl").write_text("".join(json.dumps(v) + "\n" for v in views))
    table = load_labels(tmp_path)
    counts = table.groupby("depth_source")["matched"].agg(["size", "sum"])
    assert counts.loc["moge"].tolist() == [2, 2]
    assert counts.loc["rendered"].tolist() == [1, 0]                           # one miss, in sA only
    assert table.attrs["n_not_run"] == {"moge": 0, "rendered": 1}


def test_wilson_interval_and_conformal_quantile_match_their_definitions() -> None:
    """Known Wilson bounds; the conformal quantile is the ceil((n + 1) L)-th score, or inf if n is too small."""
    lo, hi = wilson_interval([0, 5], [10, 10])
    np.testing.assert_allclose(lo, [0.0, 0.2366], atol=1e-4)
    np.testing.assert_allclose(hi, [0.2775, 0.7634], atol=1e-4)
    scores = np.arange(1.0, 10.0)                                     # n = 9
    assert conformal_quantile(scores, 0.5) == 5.0                     # ceil(10 * 0.5) = 5th
    assert conformal_quantile(scores, 0.9) == 9.0                     # ceil(9.0) = 9th
    assert conformal_quantile(scores, 0.95) == math.inf               # ceil(9.5) = 10 > 9


def test_fit_error_model_recovers_bias_spread_and_gross_rate() -> None:
    """On i.i.d. data the fit recovers a linear bias, a log-linear spread and the gross-error share."""
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    sigma = np.exp(0.5 * (np.log(0.05**2) + 0.8 * x))
    gross = rng.random(n) < 0.05
    e = 0.02 + 0.03 * x + sigma * rng.normal(size=n) + gross * rng.normal(0.0, 1.0, size=n)
    model = fit_error_model(e, X, X, ridge=0.0, s0=1e-4)
    assert model.converged
    np.testing.assert_allclose(model.beta, [0.02, 0.03], atol=0.006)
    np.testing.assert_allclose(model.gamma, [np.log(0.05**2), 0.8], atol=0.1)
    assert model.pi_gross == pytest.approx(0.05, abs=0.015)
    assert model.sigma_gross == pytest.approx(1.0, rel=0.2)


def test_select_ridge_shrinks_noise_features_but_keeps_real_ones() -> None:
    """Inner leave-one-scene-out CV picks the strongest ridge when the features are pure noise and a weak one
    when they drive the spread."""
    config = AuditConfig(ridge_grid=(1.0, 1000.0))
    for heteroscedastic, expected in ((True, 1.0), (False, 1000.0)):
        rows = synthetic_pairs(n_scenes=4, n_items=40, heteroscedastic=heteroscedastic, gross_rate=0.0, seed=5)
        rows = rows[(rows["qtype"] == "size_height") & rows["matched"]]
        if not heteroscedastic:                                       # remove the feature-driven bias as well
            rows = rows.assign(e=rows["e"] + 0.05 * rows["border_touch"])
        ridge = select_ridge(rows, lambda r, lam: NumericPredictor.fit(r, "sigma", ["laser", "moge"], config, lam),
                             lambda m, r: float(np.sum(m.nll(r))), config.ridge_grid)
        assert ridge == expected, heteroscedastic


def test_conformal_coverage_holds_on_exchangeable_scenes(hetero: AuditResult) -> None:
    """Out-of-scene conformal coverage is within a few points of nominal for the model and the baseline."""
    cov = hetero.tables["coverage"]
    overall = cov[(cov["dataset"] == "synthetic") & (cov["by"] == "all")].set_index(["method", "level"])
    for method in ("sigma", "const"):
        for level in (0.5, 0.8, 0.9, 0.95):
            assert overall.loc[(method, level), "coverage"] == pytest.approx(level, abs=0.035), (method, level)
    # The correctly specified model's own predictive distribution (no conformal step) is close to calibrated too.
    assert overall.loc[("sigma_raw", 0.9), "coverage"] == pytest.approx(0.9, abs=0.04)


def test_ltt_threshold_keeps_its_bound_in_repeated_draws() -> None:
    """With P(wrong | score) = score / 2 on uniform scores, P(wrong | score <= lam) = lam / 4; the certified lam
    exceeds the true limit 4 epsilon in at most about delta of repeated calibration draws."""
    rng = np.random.default_rng(1)
    epsilon, delta = 0.1, 0.1
    lams = []
    for _ in range(300):
        s = rng.random(1000)
        lams.append(ltt_threshold(s, rng.random(1000) < s / 2, epsilon, delta)[0])
    lams = np.array(lams)
    assert np.mean(lams > 4 * epsilon) <= delta + 0.03
    assert np.median(lams) > 0.15                                     # it does accept a fair share
    assert ltt_threshold(np.zeros(100), np.ones(100), epsilon, delta)[0] == -math.inf


def test_gate_error_among_accepted_stays_within_epsilon(hetero: AuditResult) -> None:
    """On held-out scenes, the certified gate's error among accepted labels is at most epsilon, below the
    ungated error rate, and it still accepts a useful share."""
    gate = hetero.tables["gate"]
    row = gate[(gate["kind"] == "sigma") & (gate["tau_scale"] == 1.0)].iloc[0]
    assert row["err_all"] > 0.2                                       # the gate has work to do
    assert row["accept_err"] <= FAST.epsilon
    assert row["accept_yield"] > 0.3
    assert row["reject_err"] > row["err_all"]                         # rejected labels are mostly wrong
    cat = gate[(gate["answer_kind"] == "categorical") & (gate["kind"] == "logistic")].iloc[0]
    assert cat["accept_n"] == 0 or cat["accept_err"] <= FAST.epsilon


def test_sigma_model_beats_constant_sigma_on_heteroscedastic_truth(hetero: AuditResult) -> None:
    """Held-out likelihood, rank correlation, risk-yield area and conditional coverage all favour the model."""
    rank = hetero.tables["rank"]
    pooled = rank[rank["depth_source"] == "all"].set_index(["qtype", "kind"])
    for qtype in ("size_height", "distance_center"):
        assert pooled.loc[(qtype, "sigma"), "nll_mean"] < pooled.loc[(qtype, "const"), "nll_mean"] - 0.1
        assert pooled.loc[(qtype, "sigma"), "rho"] > 0.3
    aurc = hetero.tables["aurc"].set_index(["answer_kind", "ranking"])
    assert aurc.loc[("numeric", "sigma"), "aurc"] < aurc.loc[("numeric", "const"), "aurc"] \
        < aurc.loc[("numeric", "random"), "aurc"]
    assert aurc.loc[("numeric", "const"), "d_lo"] > 0                      # paired scene-bootstrap interval
    comparison = hetero.tables["comparison"].set_index(["qtype", "baseline"])
    assert comparison.loc[("all", "const"), "d_lo"] > 0
    cov = hetero.tables["coverage"]
    tercile = cov[(cov["dataset"] == "synthetic") & (cov["by"] == "sigma_tercile") & (cov["level"] == 0.9)]
    spread = tercile.groupby("method")["coverage"].agg(lambda c: c.max() - c.min())
    assert spread["sigma"] < 0.06 < spread["const"]
    brier = hetero.tables["categorical"].set_index(["qtype", "kind"])["brier_skill"]
    assert brier[("all", "logistic")] > 0


def test_sigma_model_predicts_a_flat_spread_on_homoscedastic_truth() -> None:
    """With one sigma for every row, the feature model's out-of-scene sigma barely varies and its conditional
    coverage is flat (the truth's bias still depends on a feature, so its mean is not compared)."""
    result = run_audit(synthetic_pairs(n_scenes=5, n_items=60, heteroscedastic=False, seed=4),
                       AuditConfig(n_boot=50, tau_scales=(1.0,), ridge_grid=(1.0,)))
    sigma = result.numeric.loc[result.numeric["kind"] == "sigma", "sigma"]
    assert sigma.quantile(0.9) / sigma.quantile(0.1) < 1.5           # about 5 on heteroscedastic truth
    cov = result.tables["coverage"]
    tercile = cov[(cov["dataset"] == "synthetic") & (cov["by"] == "sigma_tercile") & (cov["level"] == 0.9)
                  & (cov["method"] == "sigma")]
    assert tercile["coverage"].max() - tercile["coverage"].min() < 0.08


# --------------------------------------------------------------------------------------------------------
# The scripts: output folders of committed results have no default; the error figure draws every source
# --------------------------------------------------------------------------------------------------------


def _script(name: str) -> ModuleType:
    """Import a script of ``scripts/`` as a module, registered in ``sys.modules`` (sibling imports need it).

    Parameters
    ----------
    name : str
        Script name without ``.py``.

    Returns
    -------
    types.ModuleType
        The module; importing it has no other side effects.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("name", "argv", "flag"), [
    ("run_audit", ["--datasets", "arkitscenes"], "--results"),
    ("run_audit", ["--exclude-ambiguous"], "--results"),
    ("make_figures", ["--results", "x"], "--figures"),
    ("compare_association", ["--gated", "a", "--primary", "b"], "--results"),
    ("run_fusion_eval", ["--exclude-ambiguous"], "--results-dir"),
])
def test_scripts_require_their_aggregate_output_folder(name: str, argv: list[str], flag: str,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    """Each script exits with a usage error naming the missing output flag instead of writing to ``results/``.

    Parameters
    ----------
    name : str
        Script name.
    argv : list of str
        Arguments without the output flag.
    flag : str
        The required output flag.
    capsys : pytest.CaptureFixture
        Captures the usage error.
    """
    parse: Callable[[list[str]], Any] = _script(name).parse_args
    with pytest.raises(SystemExit) as raised:
        parse(argv)
    assert raised.value.code == 2 and flag in capsys.readouterr().err
    assert getattr(parse([*argv, flag, "out"]), flag.lstrip("-").replace("-", "_")) == Path("out")


def test_run_audit_synthetic_dry_run_needs_no_results_folder() -> None:
    """``--synthetic`` writes under ``--out`` only, so it keeps working without ``--results``."""
    args = _script("run_audit").parse_args(["--synthetic"])
    assert args.synthetic and args.results is None and not args.exclude_ambiguous


def test_error_figure_draws_every_source(capsys: pytest.CaptureFixture[str]) -> None:
    """The new box sources get their own styles, and a source without one is drawn in grey with a warning.

    Parameters
    ----------
    capsys : pytest.CaptureFixture
        Captures the warning.
    """
    figures = _script("make_figures")
    rng = np.random.default_rng(0)
    sources = ["moge", "cutr_sam3", "cutr_clip", "sparse4d", "mystery"]
    pairs = pd.DataFrame({"dataset": "arkitscenes", "answer_type": "number", "matched": True,
                          "depth_source": np.repeat(sources, 60), "log_range": rng.uniform(0, 2, 300),
                          "abs_e": rng.uniform(0.01, 1, 300)})
    fig = figures.fig_error_vs_range(pairs)
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert labels == ["MoGe-2 (monocular)", "CuTR + SAM 3 names", "CuTR + CLIP names", "Sparse4D (multi-camera)",
                      "mystery"]
    styles = {(c, m) for c, m, _ in figures.SOURCE_STYLE.values()}
    assert len(styles) == len(figures.SOURCE_STYLE)                            # colour + marker unique per source
    assert "mystery" in capsys.readouterr().err
    figures.plt.close(fig)
