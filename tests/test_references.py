"""Tests for site recalibration from physical references, on synthetic data with a known site error.

Covers: the scale and bias fits against noiseless truth, with their fallbacks; the spread-factor rules; nested,
balanced stratified reference draws; tolerances as ``is_correct`` defines them; with no references, the
experiment reproduces the audit's gate exactly; and with one site whose auto answers are 25 % too large (a
capture scale error no per-label feature sees), the pooled gate fails there while recalibration from 40
references brings the error among accepted labels back within epsilon.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from measured_space.audit import AuditConfig, AuditResult, run_audit
from measured_space.references import (ReferenceConfig, ReferenceResult, Variant, fit_site_correction,
                                       reference_order, run_reference_experiment, site_quantile_factor,
                                       synthetic_site_pairs, tolerance_at)

FAST = AuditConfig(n_boot=50, ridge_grid=(1.0,))       # fixed ridge keeps the cross-fitting fast
SHIFTED = "scene00"                                     # its auto answers are 25 % too large


@pytest.fixture(scope="module")
def shifted_pairs() -> pd.DataFrame:
    """Six synthetic sites; one has a 25 % scale error on every auto answer.

    Returns
    -------
    pandas.DataFrame
        :func:`synthetic_site_pairs` output.
    """
    return synthetic_site_pairs(6, 80, scale_errors={SHIFTED: 1.25}, seed=3)


@pytest.fixture(scope="module")
def experiment(shifted_pairs: pd.DataFrame) -> ReferenceResult:
    """The reference experiment on the shifted sites: no references, and 40 per site in 4 draws.

    Parameters
    ----------
    shifted_pairs : pandas.DataFrame
        The synthetic sites.

    Returns
    -------
    ReferenceResult
        The primary recipe with uniform references.
    """
    config = ReferenceConfig(n_refs=(0, 40), n_draws=4, variants=(Variant("abc_uniform"),))
    return run_reference_experiment(shifted_pairs, config, FAST)


@pytest.fixture(scope="module")
def audit(shifted_pairs: pd.DataFrame) -> AuditResult:
    """The audit of the same sites.

    Parameters
    ----------
    shifted_pairs : pandas.DataFrame
        The synthetic sites.

    Returns
    -------
    AuditResult
        :func:`~measured_space.audit.run_audit` with the same model settings.
    """
    return run_audit(shifted_pairs.drop(columns=["tol_abs", "tol_rel"]), FAST)


def test_fit_site_correction_recovers_scale_and_bias_on_noiseless_references() -> None:
    """Scale per source when there is no bias; bias per type when the scale is not fitted; and the fallbacks."""
    rng = np.random.default_rng(0)
    n = 12
    oracle, mu = rng.uniform(0.5, 3.0, n), rng.uniform(-0.1, 0.1, n)
    source, qtype, item = np.arange(n) % 2, np.arange(n) % 3, np.arange(n)
    true_scale = np.array([0.9, 1.15])                        # L = s * auto, so auto = (oracle + mu) / s
    corr = fit_site_correction((oracle + mu) / true_scale[source], oracle, mu, source, qtype, item, 2, 3)
    np.testing.assert_allclose(corr.scale, true_scale, rtol=1e-12)
    np.testing.assert_allclose(corr.bias, 0.0, atol=1e-12)

    true_bias = np.array([0.2, -0.1, 0.05])                   # L = auto - b
    value = oracle + mu + true_bias[qtype]
    corr = fit_site_correction(value, oracle, mu, source, qtype, item, 2, 3, scale=False, min_bias_items=4)
    np.testing.assert_allclose(corr.scale, 1.0)
    np.testing.assert_allclose(corr.bias, true_bias, atol=1e-12)
    np.testing.assert_allclose(corr.apply(value, source, qtype), oracle + mu, atol=1e-12)
    assert np.all(fit_site_correction(value, oracle, mu, source, qtype, item, 2, 3, scale=False,
                                      min_bias_items=5).bias == 0.0)          # 4 items per type: too few

    one = source == 0                                         # source 1 has 2 references: it takes the pooled scale
    keep = one | (np.arange(n) < 4)
    corr = fit_site_correction(((oracle + mu) / 1.1)[keep], oracle[keep], mu[keep], source[keep], qtype[keep],
                               item[keep], 2, 3, bias=False)
    np.testing.assert_allclose(corr.scale, [1.1, 1.1], rtol=1e-12)
    corr = fit_site_correction(value[:2], oracle[:2], mu[:2], source[:2], qtype[:2], item[:2], 2, 3)
    np.testing.assert_allclose(corr.scale, 1.0)               # two references at the site: no scale at all


def test_site_quantile_factor_rules() -> None:
    """``max`` never narrows, ``mix`` shrinks toward the global quantile, ``none`` is 1."""
    ratios = np.r_[np.full(9, 0.5), 3.0]                      # rank ceil(11 * 0.9) = 10: the largest, 3
    assert site_quantile_factor(ratios, 10, "max") == pytest.approx(3.0)
    assert site_quantile_factor(np.full(10, 0.5), 10, "max") == 1.0
    assert site_quantile_factor(ratios, 20, "mix", mix_prior=20.0) == pytest.approx(0.5 * 1.0 + 0.5 * 3.0)
    assert site_quantile_factor(ratios, 10, "none") == 1.0
    assert site_quantile_factor([], 0, "max") == 1.0
    with pytest.raises(ValueError):
        site_quantile_factor(ratios, 10, "median")


def test_reference_order_draws_nested_balanced_strata() -> None:
    """Every prefix of a stratified order holds about a third of each sigma tercile; uniform is a permutation."""
    rng = np.random.default_rng(2)
    sigma = rng.uniform(0.01, 1.0, 31)
    order = reference_order(sigma, rng, "stratified")
    assert sorted(order) == list(range(31))
    tercile = (3 * np.argsort(np.argsort(sigma))) // 31
    for n in range(1, 31):
        counts = np.bincount(tercile[order[:n]], minlength=3)
        assert counts.max() - counts.min() <= 1, n
    assert sorted(reference_order(sigma, rng, "uniform")) == list(range(31))
    with pytest.raises(ValueError):
        reference_order(sigma, rng, "systematic")


def test_tolerance_at_matches_the_correctness_rule() -> None:
    """The larger of the absolute and relative bounds; a missing bound is ignored; none at all gives 0."""
    np.testing.assert_allclose(tolerance_at([0.05, 0.05, np.nan, np.nan], [0.1, 0.1, 0.1, np.nan],
                                            [0.3, 2.0, -2.0, 1.0]), [0.05, 0.2, 0.2, 0.0])


def test_no_references_reproduces_the_audit_gate(experiment: ReferenceResult, audit: AuditResult) -> None:
    """With n = 0, every scene's accepted labels, their errors and the 90 % coverage are the audit's, exactly.

    Parameters
    ----------
    experiment : ReferenceResult
        The reference experiment.
    audit : AuditResult
        The audit of the same sites.
    """
    gate = audit.gate[(audit.gate["kind"] == "sigma") & (audit.gate["answer_kind"] == "numeric")]
    expected = gate.groupby(["tau_scale", "scene"]).apply(
        lambda g: pd.Series({"n_eval": len(g), "n_acc": int((g["decision"] == "accept").sum()),
                             "n_acc_wrong": int(g.loc[g["decision"] == "accept", "wrong_s"].sum()),
                             "n_cov": int(g["cov90"].sum())}), include_groups=False)
    d = experiment.draws
    got = d[d["n_refs"] == 0].set_index(["tau_scale", "scene"])[list(expected.columns)].sort_index()
    pd.testing.assert_frame_equal(got.astype(int), expected.sort_index().astype(int))
    assert expected.loc[(1.0, SHIFTED), "n_acc"] > 0                    # the shifted site does get accepted labels


def test_references_restore_the_bound_at_a_site_with_a_scale_error(experiment: ReferenceResult) -> None:
    """The pooled gate fails at the shifted site; 40 references per site bring its error among accepted labels,
    and the pooled one, within epsilon, find the scale, and keep most labels.

    Parameters
    ----------
    experiment : ReferenceResult
        The reference experiment.
    """
    eps = FAST.epsilon
    scenes = experiment.tables["scenes"].set_index(["n_refs", "tau_scale", "scene"])
    assert scenes.loc[(0, 1.0, SHIFTED), "err_acc"] > 0.5                # no references: mostly wrong labels pass
    assert scenes.loc[(40, 1.0, SHIFTED), "err_acc"] <= eps
    assert scenes.loc[(40, 1.0, SHIFTED), "scale_moge_p50"] == pytest.approx(1 / 1.25, rel=0.02)
    assert scenes.loc[(40, 1.0, SHIFTED), "yield"] > 0.5
    summary = experiment.tables["summary"].set_index(["n_refs", "tau_scale"])
    assert summary.loc[(0, 1.0), "err_pooled"] > 0.5
    assert summary.loc[(40, 1.0), "err_pooled"] <= eps
    assert summary.loc[(40, 1.0), "yield_pooled"] > summary.loc[(0, 1.0), "yield_pooled"]
    assert summary.loc[(40, 1.0), "cov_pooled"] == pytest.approx(0.9, abs=0.05)
    assert not math.isnan(summary.loc[(40, 2.0), "err_pooled"])
