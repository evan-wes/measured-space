"""Tests for the VLM evaluation script's sampling and scoring.

Covered: source discovery, the pooled deployment unit, the certified accepted set, and missed questions (drawn
with ``--include-missed``, scored against the oracle only). All run on small synthetic label folders under
``tmp_path`` (no dataset files, no model, no GPU).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from measured_space.qa_schema import QAItem, QuestionSpec, Referent, write_jsonl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_vlm_eval.py"


def _script() -> ModuleType:
    """Import ``scripts/run_vlm_eval.py`` as a module (it is not part of the package).

    Returns
    -------
    types.ModuleType
        The script's module, registered in ``sys.modules`` (its dataclasses need that); importing it has no
        other side effects.
    """
    if "run_vlm_eval" in sys.modules:
        return sys.modules["run_vlm_eval"]
    spec = importlib.util.spec_from_file_location("run_vlm_eval", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_vlm_eval"] = module
    spec.loader.exec_module(module)
    return module


def _spec(name: str, answer_type: str, scene: str = "s0", dataset: str = "arkitscenes") -> QuestionSpec:
    """A minimal question about its own view; ``name`` makes its id distinct.

    Parameters
    ----------
    name : str
        Used as the frame name, so each name gives a distinct id and view.
    answer_type : str
        ``"number"`` (question type ``distance_center``, referents ``<name>_a`` and ``<name>_b``), ``"choice"``
        (``left_right``, the same referents) or ``"count"`` (``count``, no referents).
    scene : str
        Scene id.
    dataset : str
        Dataset name.

    Returns
    -------
    QuestionSpec
        Numbers in metres with a 0.1 m / 10 % tolerance; choices ``("left", "right")``.
    """
    qtype = {"number": "distance_center", "choice": "left_right", "count": "count"}[answer_type]
    referents = () if answer_type == "count" else tuple(Referent(f"{name}_{r}", "chair", f"the chair {r}")
                                                        for r in ("a", "b"))
    return QuestionSpec(f"{qtype}_v1", qtype, dataset, scene, "wide", name, "img.png", referents, "q?",
                        answer_type, unit="m" if answer_type == "number" else None,
                        choices=("left", "right") if answer_type == "choice" else None,
                        tolerance={"abs": 0.1, "rel": 0.1} if answer_type == "number" else {},
                        params={"phrase": "chair"} if answer_type == "count" else {})


def _accepted_file(path: Path, ids: list[str], held_out: list[dict[str, Any]] | None) -> Path:
    """Write an ``accepted_ids.json`` shaped like ``scripts/run_audit.py``'s.

    Parameters
    ----------
    path : pathlib.Path
        Where to write it.
    ids : list of str
        Ids accepted for ``arkitscenes`` / ``auto_cutr_sam3``.
    held_out : list of dict or None
        The ``held_out`` rows; None leaves the key out (an older audit).

    Returns
    -------
    pathlib.Path
        ``path``.
    """
    data: dict[str, Any] = {"settings": {"tau_scale": 1.0, "epsilon": 0.1},
                            "accepted": {"arkitscenes": {"auto_cutr_sam3": ids}}}
    if held_out is not None:
        data["held_out"] = held_out
    path.write_text(json.dumps(data))
    return path


def _held_out(dataset: str, kind: str, ok: bool, tau_scale: float = 1.0) -> dict[str, Any]:
    """One ``held_out`` row of ``accepted_ids.json``.

    Parameters
    ----------
    dataset : str
        Dataset name.
    kind : str
        ``"numeric"`` or ``"categorical"``.
    ok : bool
        ``certified_ok``.
    tau_scale : float
        Tolerance multiple of the row.

    Returns
    -------
    dict
        The row, with made-up counts.
    """
    return {"dataset": dataset, "answer_kind": kind, "tau_scale": tau_scale, "n_labelled": 20, "accept_n": 5,
            "accept_err": 0.0 if ok else 0.4, "accept_lo": 0.0, "accept_hi": 0.5, "certified_ok": ok}


def test_discovery_finds_any_labellers_source(tmp_path: Path) -> None:
    """A folder holding only ``qa_auto_cutr_sam3.jsonl`` yields that source, its unit and its pooled unit.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    run_vlm_eval = _script()
    folder = tmp_path / "labels_cutr_sam3" / "arkitscenes"
    spec = _spec("a", "number")
    write_jsonl([QAItem(spec, 1.0, "oracle")], folder / "qa_oracle.jsonl")
    write_jsonl([QAItem(spec, 1.1, "auto_cutr_sam3")], folder / "qa_auto_cutr_sam3.jsonl")
    assert run_vlm_eval.discover_sources(folder) == (("cutr_sam3",), ())
    (folder / "qa_auto_cutr_clip.jsonl").touch()                         # empty: found, but not scored
    assert run_vlm_eval.discover_sources(folder) == (("cutr_sam3",), ("cutr_clip",))
    assert run_vlm_eval.discover_sources(tmp_path / "missing") == ((), ())

    sources = {"arkitscenes": ("cutr_sam3",)}
    deployment = run_vlm_eval.resolve_deployment_sources(sources, {})
    assert deployment["sources"] == {"arkitscenes": "cutr_sam3"} and deployment["note"] is None
    assert deployment["chosen_by"] == {"arkitscenes": "only source"}
    units = run_vlm_eval.scoring_units(sources, deployment["sources"])
    assert [(u.name, u.parts) for u in units] == [
        ("arkitscenes/cutr_sam3", (("arkitscenes", "cutr_sam3"),)),
        (run_vlm_eval.POOLED_UNIT, (("arkitscenes", "cutr_sam3"),))]


def test_deployment_source_resolution_rules() -> None:
    """Argument first, then the v0 default if present, then a single source; otherwise the pooled unit is skipped."""
    run_vlm_eval = _script()
    resolve = run_vlm_eval.resolve_deployment_sources
    v0 = {"arkitscenes": ("laser", "lidar", "moge"), "smartspaces": ("gt_depth", "moge")}
    default = resolve(v0, {})
    assert default["sources"] == {"arkitscenes": "lidar", "smartspaces": "moge"}
    assert set(default["chosen_by"].values()) == {"default mapping"}
    assert [u.name for u in run_vlm_eval.scoring_units(v0, default["sources"])] == [
        "arkitscenes/laser", "arkitscenes/lidar", "arkitscenes/moge", "smartspaces/gt_depth", "smartspaces/moge",
        run_vlm_eval.POOLED_UNIT]                                       # the v0 units, unchanged

    chosen = resolve(v0, {"arkitscenes": "laser", "other": "x"})
    assert chosen["sources"] == {"arkitscenes": "laser", "smartspaces": "moge"}
    assert chosen["chosen_by"] == {"arkitscenes": "argument", "smartspaces": "default mapping"}
    assert chosen["ignored"] == ["other"]

    two = {"arkitscenes": ("cutr_clip", "cutr_sam3")}                  # no v0 source, two candidates
    skipped = resolve(two, {})
    assert skipped["sources"] is None and "cutr_clip, cutr_sam3" in skipped["note"]
    assert "--deployment-source" in skipped["note"]
    assert [u.name for u in run_vlm_eval.scoring_units(two, skipped["sources"])] == [
        "arkitscenes/cutr_clip", "arkitscenes/cutr_sam3"]               # no pooled unit
    assert resolve(two, {"arkitscenes": "cutr_clip"})["sources"] == {"arkitscenes": "cutr_clip"}
    partial = resolve({"arkitscenes": ("cutr_sam3",), "smartspaces": ()}, {})
    assert partial["sources"] is None and "smartspaces (sources: none)" in partial["note"]
    with pytest.raises(ValueError, match="cutr_sam3"):
        resolve(two | {"smartspaces": ("moge",)}, {"smartspaces": "cutr_sam3"})

    parse = run_vlm_eval.parse_deployment_sources
    assert parse(["arkitscenes=cutr_sam3", " smartspaces = sparse4d "]) == {"arkitscenes": "cutr_sam3",
                                                                           "smartspaces": "sparse4d"}
    for bad in (["arkitscenes"], ["=moge"], ["arkitscenes=lidar", "arkitscenes=moge"]):
        with pytest.raises(ValueError):
            parse(bad)


def test_accepted_set_keeps_only_certified_cells(tmp_path: Path) -> None:
    """Categorical certified, numeric not: only the categorical ids count; old files keep every listed id.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    """
    run_vlm_eval = _script()
    num, cat = _spec("n", "number"), _spec("c", "choice")
    kinds = {num.id: run_vlm_eval.answer_kind(num), cat.id: run_vlm_eval.answer_kind(cat)}
    assert kinds == {num.id: "numeric", cat.id: "categorical"}
    held_out = [_held_out("arkitscenes", "numeric", False), _held_out("arkitscenes", "categorical", True),
                _held_out("arkitscenes", "numeric", True, tau_scale=2.0)]  # another tolerance: ignored
    accepted = run_vlm_eval.load_accepted(_accepted_file(tmp_path / "a.json", [num.id, cat.id], held_out))
    assert accepted.certified == {("arkitscenes", "numeric"): False, ("arkitscenes", "categorical"): True}
    assert accepted.lists("arkitscenes", "cutr_sam3") and not accepted.lists("arkitscenes", "lidar")
    assert accepted.split("arkitscenes", "cutr_sam3", kinds) == ({cat.id}, {"numeric": {num.id}})
    assert not accepted.counts("smartspaces", "categorical")              # no held-out row: not certified

    legacy = run_vlm_eval.load_accepted(_accepted_file(tmp_path / "old.json", [num.id, cat.id], None))
    assert legacy.certified is None
    assert legacy.split("arkitscenes", "cutr_sam3", kinds) == ({num.id, cat.id}, {})
    assert run_vlm_eval.load_accepted(tmp_path / "absent.json") is None


def test_score_a_new_labellers_folder_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``score`` on a CuTR-only folder, with questions drawn from another folder, writes the expected summary.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Points the data root at ``tmp_path``.
    """
    run_vlm_eval = _script()
    monkeypatch.setattr(run_vlm_eval, "data_root", lambda: tmp_path)
    specs = [_spec(f"{kind}{k}", kind, scene=f"s{k % 2}") for kind in ("number", "choice") for k in range(4)]
    smart = _spec("w0", "choice", dataset="smartspaces")                # in the draw, not in the new folder
    oracle = [QAItem(s, 2.0 if s.answer_type == "number" else "left", "oracle") for s in specs]
    auto = [QAItem(s, 2.05 if s.answer_type == "number" else "left", "auto_cutr_sam3") for s in specs]
    folder = tmp_path / "outputs" / "labels_cutr_sam3"
    write_jsonl(oracle, folder / "arkitscenes" / "qa_oracle.jsonl")
    write_jsonl(auto, folder / "arkitscenes" / "qa_auto_cutr_sam3.jsonl")
    (folder / "arkitscenes" / "summary.json").write_text("{}")
    out = tmp_path / "outputs" / "vlm"
    draw_from = tmp_path / "outputs" / "labels_2d"
    sample = {"created": "t", "seed": 0, "deployment_source": dict(run_vlm_eval.DEPLOYMENT_SOURCE),
              "specs": {s.id: run_vlm_eval.spec_to_dict(s) for s in specs + [smart]},
              "draws": [{"labels_dir": str(draw_from), "per_cell": 25, "ids": [s.id for s in specs + [smart]],
                         "label_files": {}, "cells": []}]}
    out.mkdir(parents=True)
    (out / "sample.json").write_text(json.dumps(sample))
    (out / "predictions").mkdir()
    with (out / "predictions" / "qwen3-vl-8b.jsonl").open("w") as f:
        for s in specs + [smart]:
            f.write(json.dumps({"id": s.id, "model": "qwen3-vl-8b", "blind": False, "forced": False,
                                "parsed": 2.0 if s.answer_type == "number" else "left", "parse_method": "json",
                                "latency_s": 0.1, "n_new_tokens": 5, "n_image_tokens": 4, "truncated": False,
                                "max_new_tokens": 64}) + "\n")
    numeric_ids = sorted(s.id for s in specs if s.answer_type == "number")
    categorical_ids = sorted(s.id for s in specs if s.answer_type == "choice")
    accepted = _accepted_file(tmp_path / "accepted_ids.json", numeric_ids[:3] + categorical_ids[:2],
                              [_held_out("arkitscenes", "numeric", False),
                               _held_out("arkitscenes", "categorical", True)])
    results = tmp_path / "results"
    base = ["--out-dir", str(out), "score", "--labels-dir", str(folder), "--draw-from", str(draw_from),
            "--results-dir", str(results), "--n-boot", "50"]

    assert run_vlm_eval.main(base) == 1                                  # smartspaces is in the draw, not here
    assert run_vlm_eval.main(base + ["--datasets", "arkitscenes", "--accepted", str(accepted)]) == 0
    summary = json.loads((results / "summary.json").read_text())
    assert summary["n_questions"] == len(specs) and summary["datasets"] == ["arkitscenes"]
    assert summary["auto_sources"]["found"] == {"arkitscenes": ["cutr_sam3"]}
    assert set(summary["units"]) == {"arkitscenes/cutr_sam3", run_vlm_eval.POOLED_UNIT}
    assert summary["design"]["pooled_deployment_source"]["sources"] == {"arkitscenes": "cutr_sam3"}
    for unit in summary["units"].values():
        assert unit["n_accepted"] == 2                                   # the certified categorical ids only
        assert unit["n_accepted_dropped_uncertified"] == {"arkitscenes/numeric": 3}
    certification = summary["accepted_certification"]
    assert not certification["legacy"] and certification["note"] is None
    assert certification["dropped"] == [{"dataset": "arkitscenes", "source": "cutr_sam3", "answer_kind": "numeric",
                                         "certified_ok": False, "n_listed": 3, "n_scored": 3}]
    accepted_rows = [r for r in summary["accuracy"] if r["item_set"] == "accepted" and r["qtype"] == "all"]
    assert accepted_rows and all(r["n"] == 2 for r in accepted_rows)

    assert run_vlm_eval.main(base + ["--datasets", "arkitscenes", "--deployment-source",
                                     "arkitscenes=cutr_clip"]) == 1      # requested source not in the folder
    write_jsonl([QAItem(s, a.answer, "auto_cutr_clip") for s, a in zip(specs, auto)],
                folder / "arkitscenes" / "qa_auto_cutr_clip.jsonl")
    assert run_vlm_eval.main(base + ["--datasets", "arkitscenes"]) == 0   # two sources, no default: pooled skipped
    summary = json.loads((results / "summary.json").read_text())
    assert set(summary["units"]) == {"arkitscenes/cutr_clip", "arkitscenes/cutr_sam3"}
    assert summary["design"]["pooled_deployment_source"]["note"].startswith(run_vlm_eval.POOLED_UNIT + " skipped")
    assert summary["accepted_certification"] is None


def test_sample_draws_missed_questions_and_scores_them_against_the_oracle(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """``--include-missed`` draws only unanswered non-count questions, tags them, and scores them oracle-only.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Points the data root at ``tmp_path`` and replaces the view check (no dataset files here).
    """
    run_vlm_eval = _script()
    monkeypatch.setattr(run_vlm_eval, "data_root", lambda: tmp_path)
    checked: list[str] = []

    def check_views(specs: list[QuestionSpec], views_path: Path, loader: Any) -> dict[str, Any]:
        """Record the questions whose views would be checked.

        Parameters
        ----------
        specs : list of QuestionSpec
            Drawn questions.
        views_path : pathlib.Path
            Ignored.
        loader : Any
            Ignored.

        Returns
        -------
        dict
            A passing check.
        """
        checked.extend(s.id for s in specs)
        return {"n_views": len(specs), "max_abs_pose_diff": 0.0}

    monkeypatch.setattr(run_vlm_eval, "check_views", check_views)
    answerable = [_spec(f"{t}{k}", t, scene=f"s{k % 2}") for t in ("number", "choice", "count") for k in range(4)]
    missing = {"mn0": ["mn0_a"], "mn1": ["mn1_a", "mn1_b"], "mn2": ["mn2_b"], "mc0": ["mc0_a"]}
    missed = {name: _spec(name, "number" if name.startswith("mn") else "choice", scene=f"s{k % 2}")
              for k, name in enumerate(missing)}
    count_unanswered = _spec("cu", "count")              # unanswered but no missing referent: never a miss
    oracle_answer = {"number": 2.0, "choice": "left", "count": 3}
    every = answerable + list(missed.values()) + [count_unanswered]
    folder = tmp_path / "outputs" / "labels" / "arkitscenes"
    write_jsonl([QAItem(s, oracle_answer[s.answer_type], "oracle") for s in every], folder / "qa_oracle.jsonl")
    write_jsonl([QAItem(s, oracle_answer[s.answer_type], "auto_lidar") for s in answerable],
                folder / "qa_auto_lidar.jsonl")
    (folder / "summary.json").write_text("{}")
    unanswered: dict[str, dict[str, list[dict[str, Any]]]] = {s.frame: {"auto_lidar": [], "auto_moge": []}
                                                             for s in every}
    for name, spec in missed.items():
        unanswered[name]["auto_lidar"].append({"id": spec.id, "qtype": spec.qtype, "missing": missing[name]})
    unanswered["cu"]["auto_lidar"].append({"id": count_unanswered.id, "qtype": "count", "missing": []})
    first = answerable[0]                                # missed by another source only: answerable here
    unanswered[first.frame]["auto_moge"].append({"id": first.id, "qtype": first.qtype, "missing": ["x"]})
    with (folder / "views.jsonl").open("w") as f:
        for s in every:
            f.write(json.dumps({"view_key": s.frame, "dataset": s.dataset, "scene": s.scene, "camera_id": "wide",
                                "frame": s.frame, "depth_sources": ["lidar", "moge"],
                                "auto_unanswered": unanswered[s.frame]}) + "\n")

    labels, out = folder.parent, tmp_path / "outputs" / "vlm"
    sample_args = ["--out-dir", str(out), "sample", "--labels-dir", str(labels), "--datasets", "arkitscenes",
                   "--per-cell", "2"]
    assert run_vlm_eval.main(sample_args) == 0
    assert "missed" not in run_vlm_eval.load_sample(out)["draws"][-1]     # off by default
    assert run_vlm_eval.main(sample_args + ["--extend", "--include-missed"]) == 0
    sample = run_vlm_eval.load_sample(out)
    draw = sample["draws"][-1]
    record = draw["missed"]
    items = {r["id"]: r for r in record["items"]}
    numeric_pool = [missed[n] for n in ("mn0", "mn1", "mn2")]
    expected = set(run_vlm_eval.draw_sample(numeric_pool, 2, 0)) | {missed["mc0"].id}   # capped at --per-cell
    assert set(items) == expected and draw["n_new"] == 0 and record["n_new"] == 3
    name_of = {s.id: name for name, s in missed.items()}
    for item_id, item in items.items():
        assert item["missed"] is True and item["missing_referents"] == missing[name_of[item_id]]
        assert item["qtype"] in ("distance_center", "left_right")
    assert not set(items) & set(draw["ids"]) and len(draw["ids"]) == 6                     # 2 per answerable cell
    cells = {c["qtype"]: c for c in record["cells"]}
    assert (cells["distance_center"]["n_candidates"], cells["distance_center"]["n_drawn"]) == (3, 2)
    assert set(cells) == {"distance_center", "left_right"}                                 # no count cell
    assert set(items) <= set(checked) and set(items) <= {s.id for s in run_vlm_eval.sample_specs(sample, 0)}

    correct_missed = {missed["mn0"].id}                  # the other missed questions are answered wrongly
    (out / "predictions").mkdir()
    with (out / "predictions" / "qwen3-vl-8b.jsonl").open("w") as f:
        for spec in run_vlm_eval.sample_spec_map(sample).values():
            right = spec.id not in items or spec.id in correct_missed
            value = oracle_answer[spec.answer_type] if right else {"number": 5.0, "choice": "right"}[spec.answer_type]
            f.write(json.dumps({"id": spec.id, "model": "qwen3-vl-8b", "blind": False, "forced": False,
                                "parsed": value, "parse_method": "json", "latency_s": 0.1, "n_new_tokens": 5,
                                "n_image_tokens": 4, "truncated": False, "max_new_tokens": 64}) + "\n")
    results = tmp_path / "results"
    assert run_vlm_eval.main(["--out-dir", str(out), "score", "--labels-dir", str(labels), "--results-dir",
                              str(results), "--n-boot", "100"]) == 0
    summary = json.loads((results / "summary.json").read_text())
    assert summary["units"]["arkitscenes/lidar"]["n_items"] == 6                           # missed stay out of 2x2
    assert {r["n"] for r in summary["accuracy"] if r["qtype"] == "all"} == {6}
    assert summary["missed"]["n_missed"] == 3 and "items" not in summary["sample_draws"][-1]["missed"]
    table = pd.DataFrame(summary["missed_vs_answerable"])
    csv = pd.read_csv(results / "missed_vs_answerable.csv")                                # written to 4 decimals
    pd.testing.assert_frame_equal(csv, table, check_dtype=False, atol=1e-4)
    assert set(table["forced_policy"]) == {"counted", "strict"}
    counted = table[table["forced_policy"] == "counted"].set_index("qtype")
    assert set(counted.index) == {"all", "distance_center", "left_right"}
    numeric_drawn = [i for i in items if missed[name_of[i]].answer_type == "number"]
    acc_numeric = sum(i in correct_missed for i in numeric_drawn) / 2
    assert counted.loc["distance_center", ["n_missed", "n_answerable"]].tolist() == [2, 2]
    assert counted.loc["distance_center", "acc_missed"] == pytest.approx(acc_numeric)
    assert counted.loc["left_right", ["acc_missed", "acc_answerable", "answerable_minus_missed"]].tolist() == [
        0.0, 1.0, 1.0]
    overall = counted.loc["all"]
    assert (overall.n_missed, overall.n_answerable) == (3, 4)                              # counts left out
    assert overall.acc_missed == pytest.approx(acc_numeric * 2 / 3)
    assert overall.answerable_minus_missed == pytest.approx(1.0 - acc_numeric * 2 / 3)
    assert overall.ci_low <= overall.ci_high
    assert (out / "scored" / "labels" / "oracle_only__missed_vs_answerable.csv").exists()
