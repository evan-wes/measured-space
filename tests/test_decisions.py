"""Tests of the decision analysis (:mod:`measured_space.decisions`, ``scripts/run_decision_analysis.py``)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from measured_space import decisions as dec
from measured_space.audit import NUMERIC_QTYPES, AuditConfig, fold_table


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
                 n_objects: int = 3, max_folds: int = 20, seed: int = 0) -> Path:
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
    return audit


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
