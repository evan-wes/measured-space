"""Tests for review effort: review orders, curves, targets, operating points, minutes, miss review, and the script.

The arithmetic is checked on hand-made answers whose results are known: the ideal order removes every wrong answer
after reviewing exactly the wrong share, random review is a straight line, and a gate's residual is its wrong
accepted answers over all answers. The script runs end to end on a small synthetic audit folder under
``tmp_path`` (no data drive), and refuses a results folder that is not empty.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from measured_space import review_effort as R

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_review_effort.py"


def _script() -> ModuleType:
    """Import ``scripts/run_review_effort.py`` as a module (it is not part of the package).

    Returns
    -------
    types.ModuleType
        The script's module, registered in ``sys.modules`` (its dataclasses need that); importing it has no
        other side effects.
    """
    if "run_review_effort" in sys.modules:
        return sys.modules["run_review_effort"]
    spec = importlib.util.spec_from_file_location("run_review_effort", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_review_effort"] = module
    spec.loader.exec_module(module)
    return module


def synthetic_gate(n_scenes: int = 6, n_per_scene: int = 50, seed: int = 0) -> pd.DataFrame:
    """A ``gate.csv``-like table: numeric answers at two tolerances and categorical answers, risk-ranked.

    The feature model's score rises with the chance of a wrong answer; the constant baseline's score is noise.
    Answers with a score at most 0.3 are accepted by both gates, those above 0.9 rejected, the rest flagged.

    Parameters
    ----------
    n_scenes : int, default 6
        Scenes.
    n_per_scene : int, default 50
        Answers of each kind per scene.
    seed : int, default 0
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`measured_space.review_effort.GATE_COLUMNS` plus ``dataset``.
    """
    rng = np.random.default_rng(seed)
    frames = []
    row = 0
    for answer_kind, kinds in (("numeric", ("sigma", "const")), ("categorical", ("logistic", "const"))):
        n = n_scenes * n_per_scene
        rows = np.arange(row, row + n)
        row += n
        scene = np.repeat([f"s{i}" for i in range(n_scenes)], n_per_scene)
        risk = rng.random(n)
        wrong_1 = (rng.random(n) < risk).astype(float)
        wrong_2 = wrong_1 * (rng.random(n) < 0.5)               # half the wrong answers are within 2 tau
        for scale in ((1.0, 2.0) if answer_kind == "numeric" else (1.0,)):
            wrong_s = wrong_1 if scale == 1.0 else wrong_2
            for kind in kinds:
                score = risk if kind != "const" else rng.random(n)
                decision = np.where(score <= 0.3, "accept", np.where(score > 0.9, "reject", "flag"))
                frames.append(pd.DataFrame({"row": rows, "kind": kind, "answer_kind": answer_kind,
                                            "tau_scale": scale, "scene": scene, "depth_source": "lidar",
                                            "wrong": wrong_1, "wrong_s": wrong_s, "score": score,
                                            "decision": decision, "dataset": "arkitscenes"}))
    return pd.concat(frames, ignore_index=True)


def test_review_order_ranks_by_risk_and_interleaves_kinds() -> None:
    """One kind: decreasing score. Two kinds: the same share of each is reviewed at every point."""
    rng = np.random.default_rng(0)
    order = R.review_order([0.1, 0.9, 0.5, 0.7], ["a"] * 4, rng)
    assert order.tolist() == [1, 3, 2, 0]
    score = np.array([3.0, 1.0, 2.0, 0.9, 0.1, 0.5])       # kind a: rows 0-2 (bounds), kind b: rows 3-5 (1 - p)
    part = np.array(["a", "a", "a", "b", "b", "b"])
    order = R.review_order(score, part, rng)
    assert sorted(order[:2].tolist()) == [0, 3] and sorted(order[2:4].tolist()) == [2, 5]
    with pytest.raises(ValueError):
        R.review_order([0.1, np.nan], ["a", "a"], rng)


def test_ideal_order_reaches_zero_at_the_wrong_share_and_random_is_linear() -> None:
    """The ideal curve is ``max(W - k, 0) / n`` and hits 0 at ``f = W / n``; random is ``W / n (1 - f)``."""
    rng = np.random.default_rng(1)
    wrong = (rng.random(200) < 0.25).astype(float)
    share = wrong.mean()
    f = np.array([0.0, share / 2, share, 0.9, 1.0])
    ideal = R.residual_curve(wrong[R.ideal_order(wrong, rng)], f)
    assert ideal[0] == pytest.approx(share)
    assert ideal[1] == pytest.approx(share - np.ceil(share / 2 * 200) / 200)
    assert ideal[2] == 0.0 and ideal[-1] == 0.0
    grid = np.linspace(0, 1, 11)
    assert np.allclose(R.random_curve(share, grid), share * (1 - grid))
    many = np.mean([R.residual_curve(wrong[rng.permutation(200)], grid) for _ in range(400)], axis=0)
    assert np.allclose(many, share * (1 - grid), atol=0.01)        # random orders average to the line


def test_review_needed_by_hand() -> None:
    """Targets on a hand-made order: 3 wrong of 10, at positions 1, 2 and 4."""
    wrong = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    assert R.review_needed(wrong, [0.3, 0.2, 0.1, 0.0]).tolist() == pytest.approx([0.0, 0.1, 0.2, 0.4])
    # Weights: the first answer counted twice (11 answers, 4 wrong). 20%: catch 1.8, so the first (weight 2) does;
    # 0%: every wrong one, through the fourth answer (weight 2 + 1 + 1 + 1).
    weighted = R.review_needed(wrong, [0.2, 0.0], [2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert weighted.tolist() == pytest.approx([2 / 11, 5 / 11])
    assert R.review_needed([0, 0, 1], [0.0]).tolist() == [1.0]       # last answer wrong: full review
    assert R.random_review_needed(0.2, [0.05, 0.3]).tolist() == pytest.approx([0.75, 0.0])
    assert R.random_review_needed(np.array([0.1, 0.4]), [0.05]).shape == (2, 1)


def test_minutes() -> None:
    """Minutes per 1,000 answers: ``1000 f t / 60``."""
    assert R.minutes(0.3, 60) == pytest.approx(300.0)
    assert R.minute_columns(0.5, [10, 30]) == pytest.approx({"minutes_10s": 1000 * 0.5 * 10 / 60,
                                                             "minutes_30s": 250.0})


def test_labeller_effort_operating_points_and_targets() -> None:
    """Gate arithmetic, ideal and random targets against closed forms, and bootstrap intervals around them."""
    gate = synthetic_gate()
    curves, targets, points = R.labeller_effort(gate, {"labeller": "x"}, epsilon=0.1, targets=(0.1, 0.05),
                                                seconds=(10.0,), n_boot=200, seed=0)
    groups = set(zip(points["answer_kind"], points["tau_scale"]))
    assert groups == {("numeric", 1.0), ("numeric", 2.0), ("categorical", 1.0), ("combined", 1.0), ("combined", 2.0)}

    num = gate[(gate["answer_kind"] == "numeric") & (gate["tau_scale"] == 1.0) & (gate["kind"] == "sigma")]
    accept = num["decision"] == "accept"
    op = points[(points["answer_kind"] == "numeric") & (points["tau_scale"] == 1.0)
                & (points["order"] == "ours")].iloc[0]
    assert op["accepted_share"] == pytest.approx(accept.mean())
    assert op["review_fraction"] == pytest.approx(1 - accept.mean())
    assert op["residual_wrong_share"] == pytest.approx(num.loc[accept, "wrong_s"].sum() / len(num))
    assert op["residual_wrong_share"] == pytest.approx(op["wrong_among_accepted"] * op["accepted_share"])
    assert op["certified_bound"] == pytest.approx(0.1 * accept.mean())
    assert op["minutes_10s"] == pytest.approx(1000 * op["review_fraction"] * 10 / 60)
    assert op["minutes_all_10s"] == pytest.approx(1000 * 10 / 60)

    combined = points[(points["answer_kind"] == "combined") & (points["tau_scale"] == 2.0)
                      & (points["order"] == "ours")].iloc[0]
    assert combined["n"] == 2 * len(num) and combined["model_kind"] == "sigma+logistic"

    t = targets[(targets["answer_kind"] == "numeric") & (targets["tau_scale"] == 1.0)]
    share = num["wrong_s"].mean()
    for _, r in t.iterrows():
        if r["order"] == "ideal":
            assert r["review_fraction"] == pytest.approx(np.ceil((share - r["target"]) * len(num) - 1e-9) / len(num))
        if r["order"] == "random":
            assert r["review_fraction"] == pytest.approx(1 - r["target"] / share)
        assert r["lo"] - 0.05 <= r["review_fraction"] <= r["hi"] + 0.05
    ours = t[(t["order"] == "ours")]
    assert (ours["minus_random"] < 0).all()                         # a risk order beats random review
    curve = curves[(curves["answer_kind"] == "numeric") & (curves["tau_scale"] == 1.0)]
    assert set(curve["order"]) == set(R.ORDERS) and len(curve) == len(R.ORDERS) * R.N_GRID


def test_miss_review_table_by_hand() -> None:
    """Only ``by == "all"`` rows of the selected labeller; minutes and recoveries per 1,000 images."""
    base = {"kind": "cross_frame", "witness": "other_frames", "dataset": "arkitscenes", "n_views": 200,
            "n_flags": 100, "flags_per_view": 0.5, "n_hit": 12, "precision": 0.12, "n_present": 400, "n_missed": 40,
            "n_missed_flagged": 10, "flag_recall": 0.25, "n_missed_clear": 8, "n_missed_clear_flagged": 4,
            "flag_recall_clear": 0.5}
    frame = pd.DataFrame([{**base, "labeller": "a", "source": "lidar", "by": "all", "group": "all"},
                          {**base, "labeller": "a", "source": "lidar", "by": "min_support", "group": "2"},
                          {**base, "labeller": "b", "source": "boxes", "by": "all", "group": "all"}])
    out = R.miss_review_table(frame, [10.0, 60.0], [("a", "lidar")])
    assert len(out) == 1 and out["flagged_labeller"].iloc[0] == "a"
    r = out.iloc[0]
    assert r["minutes_10s"] == pytest.approx(1000 * 0.5 * 10 / 60)
    assert r["recovered_per_1000_images"] == pytest.approx(50.0)
    assert r["misses_per_1000_images"] == pytest.approx(200.0)
    assert r["flags_per_recovered"] == pytest.approx(10.0)
    assert r["minutes_per_recovered_60s"] == pytest.approx(10.0)
    assert len(R.miss_review_table(frame)) == 2
    with pytest.raises(ValueError):
        R.miss_review_table(frame.drop(columns=["n_flags"]))


def _audit_folder(folder: Path) -> Path:
    """Write a small synthetic audit folder (``pairs.csv``, ``gate.csv``, ``summary.json``).

    Parameters
    ----------
    folder : pathlib.Path
        Where to write it (created).

    Returns
    -------
    pathlib.Path
        The folder.
    """
    folder.mkdir(parents=True)
    gate = synthetic_gate(n_scenes=4, n_per_scene=20)
    gate.to_csv(folder / "gate.csv", index=False)
    rows = gate.drop_duplicates("row").sort_values("row")
    pairs = pd.DataFrame({"dataset": "arkitscenes", "depth_source": "lidar",
                          "answer_type": np.where(rows["answer_kind"] == "numeric", "number", "choice"),
                          "matched": True})
    pairs = pd.concat([pairs, pairs.iloc[:3].assign(matched=False)], ignore_index=True)     # 3 unanswered
    pairs.to_csv(folder / "pairs.csv", index=False)
    (folder / "summary.json").write_text(json.dumps({"config": {"epsilon": 0.1, "delta": 0.1}}))
    return folder


def test_script_writes_aggregates_and_refuses_a_non_empty_results_folder(tmp_path: Path) -> None:
    """End to end on a synthetic audit and miss folder; a second run into the same folder is refused."""
    script = _script()
    audit = _audit_folder(tmp_path / "audit")
    misses = tmp_path / "misses"
    misses.mkdir()
    pd.DataFrame([{"kind": "cross_frame", "labeller": "v0", "source": "lidar", "witness": "other_frames",
                   "dataset": "arkitscenes", "by": "all", "group": "all", "n_views": 10, "n_flags": 5,
                   "flags_per_view": 0.5, "n_hit": 1, "precision": 0.2, "n_present": 30, "n_missed": 6,
                   "n_missed_flagged": 1, "flag_recall": 1 / 6, "n_missed_clear": 2, "n_missed_clear_flagged": 1,
                   "flag_recall_clear": 0.5}]).to_csv(misses / "flag_metrics.csv", index=False)
    results = tmp_path / "results"
    argv = ["--audits", f"lab={audit}:lidar", "--misses", f"lab={misses}:v0/lidar", "--results", str(results),
            "--n-boot", "20", "--seconds", "10", "30"]
    assert script.main(argv) == 0
    written = {p.name for p in results.iterdir()}
    assert {"effort_curves.csv", "effort_targets.csv", "operating_points.csv", "miss_review.csv", "summary.json",
            "review_effort.pdf", "review_effort.png"} <= written
    summary = json.loads((results / "summary.json").read_text())
    assert summary["settings"]["seconds_per_review"] == [10.0, 30.0] and summary["assumptions"]
    counts = summary["inputs"]["audits"][0]["counts"]["arkitscenes"]
    assert counts["numeric"]["n_unanswered"] == 3 and counts["numeric"]["n_answered_unscored"] == 0
    assert "minutes_30s" in pd.read_csv(results / "operating_points.csv").columns
    assert len(pd.read_csv(results / "miss_review.csv")) == 1

    with pytest.raises(SystemExit):
        script.parse_args(argv)                                     # results now holds files
    with pytest.raises(SystemExit):
        script.parse_args(["--audits", f"lab={audit}", "--no-misses", "--results", str(audit / "inside")])
