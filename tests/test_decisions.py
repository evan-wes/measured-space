"""Tests of the decision analysis (:mod:`measured_space.decisions`, ``scripts/run_decision_analysis.py``)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from measured_space import decisions as dec
from measured_space.audit import NUMERIC_QTYPES, AuditConfig, fold_table, scene_folds


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
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _decision_script() -> ModuleType:
    """The decision-analysis script, after its sibling imports.

    Returns
    -------
    types.ModuleType
        ``scripts/run_decision_analysis.py``.
    """
    _script("make_figures")
    _script("run_audit")
    return _script("run_decision_analysis")


def _answers(n_scenes: int = 8, per_scene: int = 40, *, half: float = 0.1, error_sd: float = 0.02,
             seed: int = 0) -> pd.DataFrame:
    """Synthetic :func:`~measured_space.decisions.answer_table` output: one question type, one fold per scene.

    Parameters
    ----------
    n_scenes : int, default 8
        Scenes.
    per_scene : int, default 40
        Answers per scene.
    half : float, default 0.1
        Error-bar half-width of every answer, m.
    error_sd : float, default 0.02
        Standard deviation of the answers' errors, m (``mu`` is zero).
    seed : int, default 0
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Answers of type ``size_height`` with references uniform in [0.2, 2] m.
    """
    rng = np.random.default_rng(seed)
    n = n_scenes * per_scene
    scene = np.repeat([f"s{i}" for i in range(n_scenes)], per_scene)
    ref = rng.uniform(0.2, 2.0, n)
    e = rng.normal(0.0, error_sd, n)
    return pd.DataFrame({"row": np.arange(n), "id": [f"{i:06d}" for i in range(n)], "dataset": "arkitscenes",
                         "scene": scene, "camera_id": "cam", "frame": "0", "qtype": "size_height",
                         "depth_source": "lidar", "value": ref + e, "e": e, "ref": ref, "mu": 0.0, "sigma": 0.05,
                         "half": half, "c": ref + e, "fold": np.repeat(np.arange(n_scenes), per_scene),
                         "fixed_accept": rng.random(n) < 0.5})


# ------------------------------------------------------------------------------------------- single decisions


def test_two_sided_known_values() -> None:
    """Call, wrong flag and score on hand-checked cases, including an answer exactly at the threshold."""
    call, wrong, score = dec.two_sided(c=[1.2, 0.8, 1.05, 1.0], t=1.0, ref=[1.3, 1.1, 0.9, 2.0],
                                       half=[0.1, 0.1, 0.1, 0.1])
    assert call.tolist() == [True, False, True, False]
    assert wrong.tolist() == [False, True, True, True]
    np.testing.assert_allclose(score[:3], [0.5, 0.5, 2.0])
    assert score[3] == np.inf


def test_exact_error_bars_give_no_wrong_decisions() -> None:
    """When every interval strictly covers the reference, no decision whose interval excludes t is wrong."""
    rng = np.random.default_rng(1)
    ref = rng.uniform(0.2, 3.0, 5000)
    half = rng.uniform(0.01, 0.5, 5000)
    c = ref + rng.uniform(-0.99, 0.99, 5000) * half
    for t in (0.5, 1.0, 2.0):
        _, wrong, score = dec.two_sided(c, t, ref, half)
        assert (score <= 1).sum() > 100
        assert not wrong[score <= 1].any()


def test_threshold_decisions_cover_every_threshold() -> None:
    """Each answer meets each threshold of its dataset and type; types without thresholds are left out."""
    answers = pd.concat([_answers(2, 5), _answers(2, 5).assign(qtype="count")], ignore_index=True)
    out = dec.threshold_decisions(answers, {"arkitscenes": {"size_height": (0.5, 1.0)}})
    assert len(out) == 20 and sorted(out["threshold"].unique()) == [0.5, 1.0]
    np.testing.assert_array_equal(out["truth"], out["ref"] > out["t"])


def test_sweep_truth_is_the_sign_of_u() -> None:
    """With t = ref (1 + u), the reference is above t exactly when u < 0; zero offsets are refused."""
    out = dec.sweep_decisions(_answers(2, 10), us=(-0.1, 0.2))
    assert len(out) == 40
    np.testing.assert_array_equal(out["truth"], out["u"] < 0)
    np.testing.assert_allclose(out["t"], out["ref"] * (1 + out["u"]))
    with pytest.raises(ValueError):
        dec.sweep_decisions(_answers(2, 10), us=(0.0,))


def test_safety_never_accepts_inside_calls() -> None:
    """One-sided decisions: an inside call has an infinite score, so even an all-clean calibration never accepts it."""
    answers = _answers(10, 50).assign(qtype="distance_center")
    out = dec.safety_decisions(answers, {"arkitscenes": {"distance_center": (1.0,)}})
    inside_calls = ~out["call"]
    assert inside_calls.any() and np.isinf(out.loc[inside_calls, "score"]).all()
    np.testing.assert_array_equal(out["wrong"], out["call"] & (out["ref"] <= 1.0))
    accept, lam, _ = dec.certify(out.assign(wrong=False), dec.fold_lookup(answers), 0.1, 0.1)
    assert accept.any() and np.isfinite(lam).all() and not accept[inside_calls.to_numpy()].any()


# ------------------------------------------------------------------------------------------- certification


def test_certify_accepts_up_to_the_largest_calibration_score_when_nothing_is_wrong() -> None:
    """With no wrong calibration decision, lambda is the calibration set's largest score (as in the audit)."""
    answers = _answers()
    decisions = dec.threshold_decisions(answers, {"arkitscenes": {"size_height": (1.0,)}}).assign(wrong=False)
    accept, lam, thr = dec.certify(decisions, dec.fold_lookup(answers), 0.1, 0.1)
    for scene, g in decisions.groupby("scene"):
        assert lam[g.index[0]] == decisions.loc[decisions["scene"] != scene, "score"].max()
    np.testing.assert_array_equal(accept, decisions["score"].to_numpy() <= lam)
    assert len(thr) == 8 and accept.mean() > 0.9


def test_certify_refuses_when_decision_errors_exceed_epsilon() -> None:
    """Learn-then-Test certifies nothing when a third of the decisions are wrong at every score."""
    answers = _answers()
    decisions = dec.threshold_decisions(answers, {"arkitscenes": {"size_height": (1.0,)}})
    decisions["wrong"] = np.arange(len(decisions)) % 3 == 0
    accept, lam, thr = dec.certify(decisions, dec.fold_lookup(answers), 0.1, 0.1)
    assert not accept.any() and np.isneginf(lam).all()
    assert (thr["n_accepted_cal"] == 0).all()


def test_certify_calibrates_outside_the_test_scenes_fold() -> None:
    """Two scenes sharing a fold, all of whose decisions are wrong, never calibrate each other."""
    answers = _answers(n_scenes=10, per_scene=50)
    answers.loc[answers["scene"] == "s1", "fold"] = 0               # s0 and s1 share fold 0
    decisions = dec.threshold_decisions(answers, {"arkitscenes": {"size_height": (1.0,)}})
    decisions["wrong"] = decisions["scene"].isin(["s0", "s1"])
    accept, _, thr = dec.certify(decisions, dec.fold_lookup(answers), 0.1, 0.1)
    by_scene = thr.set_index("scene")
    assert by_scene.loc["s0", "n_cal"] == 400 and by_scene.loc["s1", "n_cal"] == 400   # 10 x 50 minus fold 0
    assert by_scene.loc["s2", "n_cal"] == 450
    finite = np.isfinite(decisions["score"]).to_numpy()
    assert accept[(decisions["scene"] == "s0").to_numpy() & finite].all()  # its calibration set is clean


def test_certify_uses_the_folds_of_each_question_type() -> None:
    """A scene's fold-mates are excluded per question type: here fold-mates under one type only."""
    a = _answers(n_scenes=6, per_scene=20)
    b = _answers(n_scenes=6, per_scene=20, seed=1).assign(qtype="size_longest")
    b["fold"] = b["fold"].replace({1: 0})                           # s0 and s1 share a fold for size_longest only
    answers = pd.concat([a, b], ignore_index=True)
    decisions = dec.threshold_decisions(answers, {"arkitscenes": {"size_height": (1.0,), "size_longest": (1.0,)}})
    _, _, thr = dec.certify(decisions.assign(wrong=False), dec.fold_lookup(answers), 0.1, 0.1)
    assert thr.set_index("scene").loc["s0", "n_cal"] == 5 * 20 + 4 * 20


def test_family_pool_calibrates_each_threshold_separately() -> None:
    """Pooled per family, a clean family is accepted and an always-wrong one is not; one threshold per family."""
    answers = _answers()
    decisions = dec.threshold_decisions(answers, {"arkitscenes": {"size_height": (0.5, 1.5)}})
    decisions["wrong"] = decisions["threshold"] == 1.5
    fold_of = dec.fold_lookup(answers)
    _, _, pooled_thr = dec.certify(decisions, fold_of, 0.1, 0.1)
    by_family, _, thr = dec.certify(decisions, fold_of, 0.1, 0.1, pool_cols=["qtype", "threshold"])
    clean = (decisions["threshold"] == 0.5).to_numpy()
    assert by_family[clean].mean() > 0.9 and not by_family[~clean].any()
    assert len(pooled_thr) == 8 and len(thr) == 2 * 8 and set(thr["threshold"]) == {0.5, 1.5}


def test_rule_table_reports_shares_rates_and_certificate() -> None:
    """Rule rows count accepted and wrong decisions; certified_ok compares with epsilon."""
    answers = _answers()
    decisions = dec.threshold_decisions(answers, {"arkitscenes": {"size_height": (1.0,)}})
    config = dec.DecisionConfig(epsilons=(0.1,), n_boot=50)
    out, thresholds = dec.apply_rules(decisions, dec.fold_lookup(answers), config, [])
    table = dec.rule_table(out, ["qtype"], dec.standard_rules(config.epsilons), config.bootstrap())
    assert set(table["rule"]) == {"certified", "interval", "accept_all", "fixed_tau_gate"}
    every = table.set_index("rule").loc["accept_all"]
    assert every["accept_share"] == 1.0 and every["wrong_n"] == int(decisions["wrong"].sum())
    cert = table.set_index("rule").loc["certified"]
    assert cert["certified_ok"] == (cert["accept_n"] == 0 or cert["wrong_rate"] <= 0.1)
    assert len(thresholds) == 8 and (thresholds["epsilon"] == 0.1).all()


# ------------------------------------------------------------------------------------------- margins


def _pair_answers(n_views: int = 300, *, shared_sd: float = 0.2, own_sd: float = 0.02,
                  seed: int = 0) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Two answers per view whose errors share a large additive term, with referents.

    Parameters
    ----------
    n_views : int, default 300
        Views (10 scenes, one fold each).
    shared_sd : float, default 0.2
        Standard deviation of the error shared by the two answers of a view, m.
    own_sd : float, default 0.02
        Standard deviation of each answer's own error, m.
    seed : int, default 0
        Random seed.

    Returns
    -------
    answers : pandas.DataFrame
        Answers (type ``size_height``) with ``mu = 0``, ``sigma`` the total error sd and ``half`` its 90 %
        Gaussian half-width.
    referents : dict
        Question id to its single referent.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_views
    view = np.repeat(np.arange(n_views), 2)
    scene_no = view % 10
    shared = np.repeat(rng.normal(0, shared_sd, n_views), 2)
    e = shared + rng.normal(0, own_sd, n)
    ref = rng.uniform(0.3, 2.0, n)
    sd = float(np.hypot(shared_sd, own_sd))
    ids = [hashlib.md5(str(i).encode()).hexdigest()[:16] for i in range(n)]
    answers = pd.DataFrame({"row": np.arange(n), "id": ids, "dataset": "arkitscenes",
                            "scene": [f"s{k}" for k in scene_no], "camera_id": "cam", "frame": view.astype(str),
                            "qtype": "size_height", "depth_source": "lidar", "value": ref + e, "e": e, "ref": ref,
                            "mu": 0.0, "sigma": sd, "half": 1.645 * sd, "c": ref + e, "fold": scene_no,
                            "fixed_accept": True})
    referents = {i: (f"obj{k % 2}",) for k, i in enumerate(ids)}
    return answers, referents


def test_margin_pairs_orient_by_id_and_drop_same_or_multi_object_pairs() -> None:
    """One pair per view, A the smaller id; same-object and multi-referent answers are left out and counted."""
    answers, referents = _pair_answers(20)
    pairs, counts = dec.margin_pairs(answers, referents)
    assert counts["n_pairs"] == 20 and (pairs["id_a"] < pairs["id_b"]).all()
    a = answers.set_index("id")
    for col, source in (("m", "ref"), ("e_m", "e")):
        expected = a.loc[pairs["id_a"], source].to_numpy() - a.loc[pairs["id_b"], source].to_numpy()
        np.testing.assert_allclose(pairs[col], expected)
    same = {i: ("obj",) for i in referents}
    same[answers["id"].iloc[0]] = ("x", "y")
    _, counts = dec.margin_pairs(answers, same)
    assert counts["n_pairs"] == 0 and counts["n_multi_referent"] == 1 and counts["n_same_object"] == 19


def test_direct_calibration_is_narrower_when_errors_are_shared() -> None:
    """A shared additive error cancels in the margin: the direct bar is far narrower and still covers ~90 %."""
    answers, referents = _pair_answers()
    pairs, _ = dec.margin_pairs(answers, referents)
    pairs = dec.direct_margin_halfwidth(pairs, 0.9)
    table = dec.margin_interval_table(pairs, [], AuditConfig(n_boot=50)).set_index("method")
    assert table.loc["direct", "half_p50"] < 0.3 * table.loc["quadrature", "half_p50"]
    assert 0.8 <= table.loc["direct", "coverage"] <= 0.97
    assert table.loc["quadrature", "coverage"] == 1.0 and table.loc["direct", "resid_corr"] > 0.9
    decisions = dec.margin_decisions(pairs, 0.1)
    assert set(decisions["decision"]) == {"sign", "fit"}
    fit = decisions[decisions["decision"] == "fit"]
    np.testing.assert_array_equal(fit["truth"].to_numpy(), pairs["m"].to_numpy() < -0.1)


# ------------------------------------------------------------------------------------------- answers and folds


def test_fold_assignment_checks_the_record() -> None:
    """Recomputed folds match a correct record and a tampered record is refused."""
    rows = pd.DataFrame({"dataset": "d", "qtype": "q", "scene": np.repeat([f"s{i}" for i in range(6)], 4)})
    record = fold_table(rows, AuditConfig(max_folds=3))
    fold, checked = dec.fold_assignment(rows, 3, 0, record)
    assert checked == ["d/q"] and fold.nunique() == 3
    with pytest.raises(ValueError):
        dec.fold_assignment(rows, 3, 0, record.assign(fold=(record["fold"] + 1) % 3))


def test_answer_table_keeps_one_source_and_checks_rows() -> None:
    """The named source is kept, the bias-corrected value and reference are derived, and mismatches fail."""
    pairs = pd.DataFrame({"id": ["a", "b", "c", "d"], "dataset": "arkitscenes", "scene": ["1", "1", "2", "3"],
                          "camera_id": "w", "frame": "0", "qtype": "size_height", "value": [1.0, 1.1, 2.0, 0.5],
                          "e": [0.1, 0.2, -0.1, 0.0]})
    numeric = pd.DataFrame({"row": [0, 1, 2, 3], "kind": "sigma", "dataset": "arkitscenes",
                            "scene": ["1", "1", "2", "3"], "qtype": "size_height",
                            "depth_source": ["lidar", "moge", "lidar", "lidar"], "e": [0.1, 0.2, -0.1, 0.0],
                            "mu": [0.05, 0.0, 0.0, 0.0], "sigma": 0.1, "half90": [0.2, 0.2, np.nan, 0.2]})
    gate = pd.DataFrame({"row": [0, 1, 2, 3], "kind": "sigma", "tau_scale": 1.0, "answer_kind": "numeric",
                         "decision": ["accept", "flag", "flag", "accept"]})
    with pytest.raises(ValueError, match="several depth sources"):
        dec.answer_table(pairs, numeric, gate)
    answers, _ = dec.answer_table(pairs, numeric, gate, source="lidar")
    assert answers["id"].tolist() == ["a", "c", "d"]
    np.testing.assert_allclose(answers["c"], [0.95, 2.0, 0.5])
    np.testing.assert_allclose(answers["ref"], [0.9, 2.1, 0.5])
    assert answers["half"].iloc[1] == np.inf and answers["fixed_accept"].tolist() == [True, False, True]
    with pytest.raises(ValueError, match="errors"):
        dec.answer_table(pairs.assign(e=[0.1, 0.2, 0.3, 0.0]), numeric, gate, source="lidar")


def test_config_rejects_bad_settings() -> None:
    """Unknown pools and out-of-range levels are refused."""
    with pytest.raises(ValueError):
        dec.DecisionConfig(pool="scene")
    with pytest.raises(ValueError):
        dec.DecisionConfig(epsilons=())
    assert dec.DecisionConfig().half_column == "half90"


# ------------------------------------------------------------------------------------------- the script


def test_results_folder_must_be_new_or_empty(tmp_path: Path) -> None:
    """A non-empty results folder, or one overlapping an input, is refused; a new or empty one is fine.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary folder.
    """
    check = _decision_script().check_results_folder
    check(tmp_path / "new", [tmp_path / "audit"])
    (tmp_path / "empty").mkdir()
    check(tmp_path / "empty", [tmp_path / "audit"])
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x.csv").write_text("x")
    with pytest.raises(FileExistsError):
        check(tmp_path / "full", [tmp_path / "audit"])
    with pytest.raises(ValueError):
        check(tmp_path / "audit" / "out", [tmp_path / "audit"])


def test_parse_audit_spec() -> None:
    """``NAME=FOLDER[:SOURCE]`` with relative folders under the root."""
    parse = _decision_script().parse_audit
    spec = parse("a=audit_x:lidar", Path("/root"))
    assert (spec.name, spec.folder, spec.source) == ("a", Path("/root/audit_x"), "lidar")
    assert parse("b=/abs/audit_y").source is None
    with pytest.raises(ValueError):
        parse("no_equals_sign")


def _write_audit(root: Path, suffix: str, dataset: str, source: str, *, n_scenes: int = 5, n_views: int = 4,
                 n_objects: int = 3, max_folds: int = 20, seed: int = 0, cal_shift: float | None = None) -> Path:
    """Write a small synthetic audit folder and the label folder it read.

    Parameters
    ----------
    root : pathlib.Path
        Folder to write ``audit_<suffix>`` and ``labels_<suffix>`` into.
    suffix : str
        Folder suffix.
    dataset : str
        Dataset name (sets the scale of the values: metres for ``arkitscenes``, five times that otherwise).
    source : str
        Depth source.
    n_scenes, n_views, n_objects : int
        Scenes, views per scene, answers per view and question type.
    max_folds : int, default 20
        The audit's ``max_folds``; below ``n_scenes`` a ``tables/folds.csv`` is written.
    seed : int, default 0
        Random seed.
    cal_shift : float, optional
        If given, also write the saved leave-two-out calibration files (:func:`_write_calibration`), every
        calibration prediction's ``mu`` set to this value (m; 0 reproduces the test predictions).

    Returns
    -------
    pathlib.Path
        The audit folder.
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 if dataset == "arkitscenes" else 5.0
    pairs, numeric, gate, records = [], [], [], []
    for s in range(n_scenes):
        for v in range(n_views):
            shared = rng.normal(0, 0.03 * scale)
            for q in NUMERIC_QTYPES:
                for o in range(n_objects):
                    qid = hashlib.md5(f"{suffix}{s}{v}{q}{o}".encode()).hexdigest()[:16]
                    ref = rng.uniform(0.3, 2.5) * scale
                    sigma = 0.05 * scale
                    e = shared + rng.normal(0, sigma)
                    row = len(pairs)
                    pairs.append({"id": qid, "dataset": dataset, "scene": f"sc{s}", "camera_id": "cam",
                                  "frame": str(v), "qtype": q, "answer_type": "number", "depth_source": source,
                                  "matched": True, "value": ref + e, "e": e})
                    half = 1.645 * float(np.hypot(sigma, 0.03 * scale))
                    numeric += [{"row": row, "mu": 0.0, "sigma": sigma, "kind": kind, "dataset": dataset,
                                 "scene": f"sc{s}", "qtype": q, "depth_source": source, "e": e, "half90": half}
                                for kind in ("sigma", "const")]
                    gate.append({"row": row, "kind": "sigma", "tau_scale": 1.0, "answer_kind": "numeric",
                                 "decision": "accept" if abs(e) < sigma else "flag"})
                    objects = [f"o{s}-{o}"] + ([f"o{s}-{(o + 1) % n_objects}"] if q.startswith("distance") else [])
                    records.append({"id": qid, "qtype": q, "referents": [{"object_id": x} for x in objects]})
    audit = root / f"audit_{suffix}"
    labels = root / f"labels_{suffix}" / dataset
    (audit / "tables").mkdir(parents=True)
    labels.mkdir(parents=True)
    pairs_df, numeric_df = pd.DataFrame(pairs), pd.DataFrame(numeric)
    pairs_df.to_csv(audit / "pairs.csv", index=False)
    numeric_df.to_csv(audit / "numeric.csv", index=False)
    pd.DataFrame(gate).to_csv(audit / "gate.csv", index=False)
    if max_folds < n_scenes:
        sigma_rows = numeric_df[numeric_df["kind"] == "sigma"]
        pd.concat([fold_table(g, AuditConfig(max_folds=max_folds)) for _, g in sigma_rows.groupby("qtype")]
                  ).to_csv(audit / "tables" / "folds.csv", index=False)
    inputs = {}
    for name, source_name in (("qa_oracle.jsonl", "oracle"), (f"qa_auto_{source}.jsonl", f"auto_{source}")):
        path = labels / name
        path.write_text("".join(json.dumps({**r, "source": source_name}) + "\n" for r in records))
        inputs[name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    summary = {"config": {"max_folds": max_folds, "seed": 0, "epsilon": 0.1, "delta": 0.1},
               "inputs": {dataset: {**inputs, "n_auto_only": {source: 0}}}, "git_commit": "synthetic"}
    (audit / "summary.json").write_text(json.dumps(summary))
    if cal_shift is not None:
        _write_calibration(audit, numeric_df[numeric_df["kind"] == "sigma"], max_folds, cal_shift)
    return audit


def _write_calibration(audit: Path, numeric: pd.DataFrame, max_folds: int, shift: float) -> None:
    """Write saved leave-two-out calibration files shaped like ``scripts/run_audit.py``'s.

    Every row of a question type is a calibration item of every test scene outside its fold, predicted with
    the test prediction's ``sigma`` and ``mu = shift``; each test scene's quantile is ``half90 / sigma``, so with
    ``shift = 0`` the calibration answers are the test answers.

    Parameters
    ----------
    audit : pathlib.Path
        The audit folder.
    numeric : pandas.DataFrame
        Its kind ``sigma`` rows (``row, dataset, scene, qtype, depth_source, sigma, half90``).
    max_folds : int
        The audit's ``max_folds``.
    shift : float
        ``mu`` of every calibration prediction, m.
    """
    predictions, quantiles = [], []
    for (dataset, qtype), g in numeric.groupby(["dataset", "qtype"]):
        fold_of = scene_folds(g["scene"], max_folds, 0)
        fold = g["scene"].map(fold_of).to_numpy()
        for test_scene, f in fold_of.items():
            cal = g[fold != f]
            predictions.append(pd.DataFrame({"kind": "sigma", "test_scene": test_scene, "row": cal["row"],
                                             "mu": shift, "sigma": cal["sigma"], "pi": 0.0, "sigma_g": 1.0,
                                             "ridge": 1.0}))
            for source, c in cal.groupby("depth_source"):
                q90 = float(c["half90"].iloc[0] / c["sigma"].iloc[0])
                quantiles.append({"dataset": dataset, "kind": "sigma", "test_scene": test_scene, "qtype": qtype,
                                  "depth_source": source, "n_cal": len(c), "cal_cell": "full", "q50": q90 / 2,
                                  "q80": q90 * 0.8, "q90": q90, "q95": q90 * 1.2})
    files = {"predictions": "calibration_numeric.csv.gz", "quantiles": "calibration_quantiles.csv.gz",
             "record": "calibration.json"}
    pd.concat(predictions, ignore_index=True).to_csv(audit / files["predictions"], index=False)
    pd.DataFrame(quantiles).to_csv(audit / files["quantiles"], index=False)
    (audit / files["record"]).write_text(json.dumps({"files": files}))


def test_script_end_to_end_on_synthetic_audits(tmp_path: Path) -> None:
    """Two synthetic labellers (one with grouped folds) produce every table, the summary and the figure.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary folder.
    """
    hand = _write_audit(tmp_path, "hand", "arkitscenes", "lidar", max_folds=3)
    ware = _write_audit(tmp_path, "ware", "smartspaces", "moge", seed=1)
    results = tmp_path / "results"
    main = _decision_script().main
    assert main(["--audits", f"hand={hand}:lidar", "--audits", f"ware={ware}", "--results", str(results),
                 "--n-boot", "20"]) == 0
    for name in ("decision_gate", "sweep", "safety", "margin_intervals", "margin_gate"):
        table = pd.read_csv(results / f"{name}.csv")
        assert set(table["labeller"]) == {"hand", "ware"}, name
    assert (results / "decision_sweep.pdf").is_file() and (results / "decision_sweep.png").is_file()
    summary = json.loads((results / "summary.json").read_text())
    assert summary["inputs"]["hand"]["folds"]["checked_against_record"]          # grouped folds were verified
    assert not summary["inputs"]["ware"]["part_b"]["skipped"]
    assert summary["config"]["epsilons"] == [0.1, 0.05] and len(summary["headline"]) == 2
    assert {"certified_eps0.1_accept_share", "certified_eps0.05_accept_share"} <= set(summary["headline"][0])
    gate = pd.read_csv(results / "decision_gate.csv")
    pooled = gate[(gate["qtype"] == "all") & (gate["rule"] == "accept_all")]
    assert (pooled["accept_share"] == 1.0).all()
    with pytest.raises(FileExistsError):                                       # never written into twice
        main(["--audits", f"hand={hand}:lidar", "--results", str(results)])


# ------------------------------------------------------------------------------------------- exact calibration


def _leave_two_out(answers: pd.DataFrame, *, shift: float = 0.0,
                   q90: Mapping[str, float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Saved leave-two-out predictions and quantiles for synthetic answers (every item, every other fold).

    Parameters
    ----------
    answers : pandas.DataFrame
        :func:`_answers`-like answers (``row, dataset, scene, qtype, depth_source, sigma, half, fold``).
    shift : float, default 0.0
        ``mu`` of every calibration prediction, m (0: the test predictions').
    q90 : mapping of str to float, optional
        Quantile per test scene; default ``half / sigma`` of the answers, so the error bars are the test ones.

    Returns
    -------
    predictions, quantiles : pandas.DataFrame
        Shaped like the audit's ``calibration_numeric.csv.gz`` and ``calibration_quantiles.csv.gz``.
    """
    predictions, quantiles = [], []
    for (dataset, qtype), g in answers.groupby(["dataset", "qtype"]):
        for scene, f in g.drop_duplicates("scene")[["scene", "fold"]].itertuples(index=False):
            cal = g[g["fold"] != f]
            predictions.append(pd.DataFrame({"kind": "sigma", "test_scene": scene, "row": cal["row"], "mu": shift,
                                             "sigma": cal["sigma"]}))
            q = (q90 or {}).get(scene, float(g["half"].iloc[0] / g["sigma"].iloc[0]))
            quantiles.append({"dataset": dataset, "kind": "sigma", "test_scene": scene, "qtype": qtype,
                              "depth_source": "lidar", "q90": q})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(quantiles)


def test_calibration_answers_rebuild_each_test_scenes_error_bars() -> None:
    """Calibration answers take the leave-two-out mu and sigma and their test scene's quantile, never its fold."""
    answers = _answers(n_scenes=6, per_scene=10)
    answers.loc[answers["scene"] == "s1", "fold"] = 0                     # s0 and s1 share fold 0
    predictions, quantiles = _leave_two_out(answers, shift=0.02, q90={"s0": 3.0})
    cal = dec.calibration_answers(answers, predictions, quantiles)
    assert len(cal) == 2 * 40 + 4 * 50 and not (cal["scene"] == cal["test_scene"]).any()
    assert set(cal.loc[cal["test_scene"] == "s0", "scene"]) == {"s2", "s3", "s4", "s5"}
    np.testing.assert_allclose(cal["c"], cal["value"] - 0.02)
    np.testing.assert_allclose(cal.loc[cal["test_scene"] == "s0", "half"], 3.0 * 0.05)
    np.testing.assert_allclose(cal.loc[cal["test_scene"] == "s2", "half"], 0.1)
    assert dec.exact_groups(quantiles) == {("arkitscenes", "size_height", f"s{i}") for i in range(6)}
    check = dec.check_test_halfwidths(answers, quantiles)
    assert check["n_missing"] == 0 and check["max_rel_diff"] == pytest.approx(1 / 3)   # s0: 3 sigma, not 2
    s0_rows = answers.loc[answers["scene"] == "s0", "row"]
    s0_item = predictions[(predictions["test_scene"] == "s2") & predictions["row"].isin(s0_rows)].iloc[:1]
    leak = pd.concat([predictions, s0_item.assign(test_scene="s1")], ignore_index=True)
    with pytest.raises(ValueError, match="fold"):                         # s0's row calibrating its fold-mate s1
        dec.calibration_answers(answers, leak, quantiles)


def test_exact_certify_equals_approximate_when_the_predictions_agree() -> None:
    """With leave-two-out predictions equal to the test ones, the exact mode certifies the same thresholds."""
    answers = _answers(n_scenes=8, per_scene=40, error_sd=0.06)
    answers.loc[answers["scene"] == "s1", "fold"] = 0
    thresholds = {"arkitscenes": {"size_height": (0.6, 1.2)}}
    decisions = dec.threshold_decisions(answers, thresholds)
    predictions, quantiles = _leave_two_out(answers)
    cal = dec.threshold_decisions(dec.calibration_answers(answers, predictions, quantiles), thresholds)
    fold_of = dec.fold_lookup(answers)
    accept, lam, thr = dec.certify(decisions, fold_of, 0.1, 0.1)
    accept_x, lam_x, thr_x = dec.certify(decisions, fold_of, 0.1, 0.1, calibration=cal,
                                         exact=dec.exact_groups(quantiles))
    np.testing.assert_array_equal(lam, lam_x)
    np.testing.assert_array_equal(accept, accept_x)
    assert (thr_x["n_cal_own"] == 0).all() and (thr_x["n_cal_exact"] == thr["n_cal"]).all()
    with pytest.raises(ValueError, match="exact"):
        dec.certify(decisions, fold_of, 0.1, 0.1, calibration=cal)


def test_exact_certify_reads_its_own_test_scenes_calibration_and_falls_back_per_type() -> None:
    """Each test scene is certified on its own calibration rows; a type whose group lacks it uses test answers."""
    answers = _answers(n_scenes=8, per_scene=40)
    thresholds = {"arkitscenes": {"size_height": (1.0,), "size_longest": (1.0,)}}
    decisions = dec.threshold_decisions(answers, thresholds).assign(wrong=False)
    predictions, quantiles = _leave_two_out(answers)
    cal = dec.threshold_decisions(dec.calibration_answers(answers, predictions, quantiles), thresholds)
    cal["wrong"] = cal["test_scene"] == "s0"                            # only s0's calibration set is all wrong
    exact = dec.exact_groups(quantiles)
    accept, lam, _ = dec.certify(decisions, dec.fold_lookup(answers), 0.1, 0.1, calibration=cal, exact=exact)
    s0 = (decisions["scene"] == "s0").to_numpy()
    assert np.isneginf(lam[s0]).all() and np.isfinite(lam[~s0]).all() and accept[~s0].mean() > 0.9

    other = _answers(n_scenes=3, per_scene=40, seed=2).assign(qtype="size_longest", row=lambda f: f["row"] + 1000,
                                                              scene=lambda f: "t" + f["scene"])
    both = pd.concat([answers, other], ignore_index=True)
    decisions = dec.threshold_decisions(both, thresholds).assign(wrong=False)
    _, _, thr = dec.certify(decisions, dec.fold_lookup(both), 0.1, 0.1, calibration=cal.assign(wrong=False),
                            exact=exact)
    by_scene = thr.set_index("scene")
    assert by_scene.loc["s0", "n_cal_own"] == 120 and by_scene.loc["s0", "n_cal_exact"] == 280
    assert by_scene.loc["ts0", "n_cal_own"] == 80 + 320                 # no exact group holds ts0


def test_exact_direct_margin_matches_the_approximate_one_when_the_predictions_agree() -> None:
    """Leave-two-out margins equal to the test ones give the approximate direct half-widths."""
    answers, referents = _pair_answers()
    pairs, _ = dec.margin_pairs(answers, referents)
    predictions, quantiles = _leave_two_out(answers)
    cal_pairs, _ = dec.margin_pairs(dec.calibration_answers(answers, predictions, quantiles), referents)
    exact, cal_pairs = dec.direct_margin_halfwidth_exact(pairs, cal_pairs, 0.9)
    approx = dec.direct_margin_halfwidth(pairs, 0.9)
    np.testing.assert_allclose(exact["h_direct"], approx["h_direct"])
    np.testing.assert_array_equal(exact["n_cal_direct"], approx["n_cal_direct"])
    assert set(cal_pairs["test_scene"]) == set(pairs["scene"]) and np.isfinite(cal_pairs["h_direct"]).all()


def test_script_uses_saved_calibration_and_records_the_mode(tmp_path: Path) -> None:
    """Audits with saved calibration run exact (same tables when it agrees, different when it does not).

    Part A's tables agree with the approximate run when the saved predictions equal the test ones (here every
    scene has the same quantile). The margin gate need not: its exact calibration margins take the test scene's
    direct quantile, the approximate ones their own scene's.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary folder.
    """
    main = _decision_script().main
    plain = _write_audit(tmp_path / "plain", "hand", "arkitscenes", "lidar", max_folds=3)
    same = _write_audit(tmp_path / "same", "hand", "arkitscenes", "lidar", max_folds=3, cal_shift=0.0)
    shifted = _write_audit(tmp_path / "shift", "hand", "arkitscenes", "lidar", max_folds=3, cal_shift=0.4)
    tables = {}
    for name, audit in (("plain", plain), ("same", same), ("shift", shifted)):
        assert main(["--audits", f"hand={audit}:lidar", "--results", str(tmp_path / f"out_{name}"),
                     "--n-boot", "20"]) == 0
        summary = json.loads((tmp_path / f"out_{name}" / "summary.json").read_text())
        mode = "approximate" if name == "plain" else "exact"
        assert set(summary["calibration"]["hand"].values()) == {mode} and list(summary["calibration_modes"]) == [mode]
        assert summary["inputs"]["hand"]["calibration"]["mode"] == mode
        tables[name] = {t: pd.read_csv(tmp_path / f"out_{name}" / f"{t}.csv") for t in ("decision_gate", "sweep",
                                                                                         "margin_gate")}
        assert all((frame["calibration"] == mode).all() for frame in tables[name].values())
    check = json.loads((tmp_path / "out_same" / "summary.json").read_text())["inputs"]["hand"]["calibration"]
    assert check["test_halfwidth_check"]["max_rel_diff"] < 1e-12
    for t, frame in tables["plain"].items():
        if t == "margin_gate":
            continue
        pd.testing.assert_frame_equal(frame.drop(columns="calibration"),
                                      tables["same"][t].drop(columns="calibration"), rtol=1e-9)
    pick = [(frame["qtype"] == "all") & (frame["rule"] == "certified") & (frame["epsilon"] == 0.1)
            for frame in (tables["plain"]["decision_gate"], tables["shift"]["decision_gate"])]
    assert (tables["shift"]["decision_gate"].loc[pick[1], "accept_share"].iloc[0]
            < tables["plain"]["decision_gate"].loc[pick[0], "accept_share"].iloc[0])


# ------------------------------------------------------------------------------------------- second opinion


def test_second_opinion_score_needs_agreeing_calls_and_takes_the_larger_score() -> None:
    """Agreeing calls score the larger of the two; disagreeing or unwitnessed decisions are never accepted."""
    frame = pd.DataFrame({"c": [1.5, 1.5, 1.5, 0.5, 1.5], "t": 1.0, "half": [0.1, 0.1, 0.1, 0.1, 0.1],
                          "c_w": [1.2, 0.8, np.nan, 0.9, 1.0], "half_w": [0.1, 0.1, np.nan, 0.05, 0.1]})
    two = dec.second_opinion_score(frame)
    np.testing.assert_allclose(two[[0, 3]], [0.5, 0.5])                 # max(0.1 / 0.5, 0.1 / 0.2); max(0.2, 0.5)
    assert np.isinf(two[[1, 2, 4]]).all()                               # disagree; no witness; witness at t
    one = dec.second_opinion_score(frame, one_sided=True)
    assert one[0] == pytest.approx(0.5) and np.isinf(one[1:]).all()    # row 3: both inside, never accepted


def test_attach_witness_prefers_leave_two_out_answers_for_calibration_decisions() -> None:
    """Test decisions get the witness's test answer; calibration ones its leave-two-out answer when it has one."""
    answers = _answers(n_scenes=4, per_scene=5)
    witness = answers.assign(c=answers["c"] + 0.5, half=0.2)
    decisions = dec.threshold_decisions(answers, {"arkitscenes": {"size_height": (1.0,)}})
    test = dec.attach_witness(decisions, witness)
    np.testing.assert_allclose(test["c_w"], test["c"] + 0.5)
    assert (test["witness_from"] == "test").all()
    cal_rows = dec.calibration_answers(answers, *_leave_two_out(answers))
    witness_cal = cal_rows[cal_rows["test_scene"] == "s0"].assign(c=-9.0, half=0.3)
    cal = dec.attach_witness(dec.threshold_decisions(cal_rows, {"arkitscenes": {"size_height": (1.0,)}}),
                             witness, witness_cal)
    s0 = (cal["test_scene"] == "s0").to_numpy()
    assert (cal.loc[s0, "c_w"] == -9.0).all() and (cal.loc[s0, "witness_from"] == "leave_two_out").all()
    assert (cal.loc[~s0, "witness_from"] == "test").all()
    with pytest.raises(ValueError, match="twice"):
        dec.attach_witness(decisions, pd.concat([witness, witness.iloc[:1]]))


def test_complete_calibration_copies_own_answers_only_for_types_outside_the_scenes_group() -> None:
    """Written-out calibration rows certify exactly as the exact mode with its per-type fallback."""
    answers = _answers(n_scenes=6, per_scene=30, error_sd=0.06)
    other = _answers(n_scenes=3, per_scene=30, seed=2).assign(qtype="size_longest", row=lambda f: f["row"] + 1000)
    both = pd.concat([answers, other], ignore_index=True)
    predictions, quantiles = _leave_two_out(answers)
    cal_answers = dec.calibration_answers(both, predictions, quantiles)
    exact = dec.exact_groups(quantiles)
    rows, covered = dec.complete_calibration_answers(both, cal_answers, exact)
    s5 = rows[rows["test_scene"] == "s5"]
    assert set(s5.loc[s5["qtype"] == "size_longest", "scene"]) == {"s0", "s1", "s2"}   # own answers, s5 has none
    assert len(rows) == len(cal_answers) + 3 * 90 + 3 * 60
    thresholds = {"arkitscenes": {"size_height": (0.6, 1.2), "size_longest": (0.6, 1.2)}}
    decisions = dec.threshold_decisions(both, thresholds)
    fold_of = dec.fold_lookup(both)
    _, lam, _ = dec.certify(decisions, fold_of, 0.1, 0.1, calibration=dec.threshold_decisions(cal_answers, thresholds),
                            exact=exact)
    _, lam_rows, _ = dec.certify(decisions, fold_of, 0.1, 0.1, calibration=dec.threshold_decisions(rows, thresholds),
                                 exact=covered)
    np.testing.assert_array_equal(lam, lam_rows)


def test_second_opinion_rule_accepts_less_and_refuses_a_contrary_witness() -> None:
    """A witness can only remove acceptances; one that always calls the other side removes them all."""
    answers = _answers(n_scenes=8, per_scene=60, error_sd=0.08, half=0.15)
    thresholds = {"arkitscenes": {"size_height": (0.9,)}}
    config = dec.DecisionConfig(n_boot=20)
    fold_of = dec.fold_lookup(answers)
    good = answers.assign(c=answers["ref"], half=0.15)
    contrary = answers.assign(c=2.0 * 0.9 - answers["c"], half=0.15)    # mirrored about t = 0.9: the opposite call
    for witness, expect_any in ((good, True), (contrary, False)):
        decisions = dec.attach_witness(dec.threshold_decisions(answers, thresholds), witness)
        decisions, _ = dec.apply_rules(decisions, fold_of, config, [])
        out = dec.apply_witness_rule(decisions, fold_of, config, [])
        accepted = out["accept_wit_eps0"].to_numpy(dtype=bool)
        assert not (accepted & (out["score"].to_numpy() > out["score_wit"].to_numpy())).any()
        assert (out["score_wit"] >= out["score"]).all()
        assert accepted.any() == expect_any
        if expect_any:
            wrong = out["wrong"].to_numpy(dtype=bool)
            assert wrong[accepted].mean() <= wrong[out["accept_eps0"].to_numpy(dtype=bool)].mean()
    with pytest.raises(ValueError):
        _decision_script().witness_pairs(["a=a"], ["a"])


def test_script_reports_the_second_opinion_rule(tmp_path: Path) -> None:
    """Two labellers of the same questions, one witnessing the other: a witness table, exact when both saved.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary folder.
    """
    script = _decision_script()
    first = _write_audit(tmp_path / "a", "hand", "arkitscenes", "lidar", max_folds=3, cal_shift=0.0)
    second = _write_audit(tmp_path / "b", "hand", "arkitscenes", "lidar", max_folds=3, cal_shift=0.0, seed=5)
    results = tmp_path / "results"
    assert script.main(["--audits", f"one={first}:lidar", "--audits", f"two={second}:lidar", "--witness", "one=two",
                        "--results", str(results), "--n-boot", "20"]) == 0
    table = pd.read_csv(results / "witness.csv")
    assert set(table["part"]) == {"threshold", "sweep", "safety"} and set(table["witness"]) == {"two"}
    assert (table["calibration"] == "exact").all() and set(table["scope"]) == {"all", "witnessed"}
    assert {"certified", "second_opinion", "interval", "second_opinion_interval"} <= set(table["rule"])
    summary = json.loads((results / "summary.json").read_text())
    pair = summary["witness"]["pairs"][0]
    assert pair["n_witnessed"] == pair["n_answers"] == 300 and pair["ref_max_abs_diff"] > 0
    assert set(pair["calibration_witness_from"]["threshold"]) <= {"leave_two_out", "test"}
    single = pd.read_csv(results / "decision_gate.csv")
    single = single[(single["labeller"] == "one") & (single["rule"] == "certified")]
    mine = table[(table["part"] == "threshold") & (table["scope"] == "all") & (table["rule"] == "certified")]
    np.testing.assert_allclose(single.sort_values(["qtype", "threshold", "epsilon"])["accept_share"].to_numpy(),
                               mine.sort_values(["qtype", "threshold", "epsilon"])["accept_share"].to_numpy())
    with pytest.raises(ValueError, match="not analysed"):
        script.witness_pairs(["one=three"], ["one", "two"])
