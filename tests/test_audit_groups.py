"""Tests for the audit's explicit fold groups, its depth-source selection and the layout grouping script.

Covers: :func:`~measured_space.audit.scene_folds` with an explicit scene-to-group mapping (one fold per group,
numbered by each group's first scene, ``max_folds`` ignored, a missing scene raises);
:func:`~measured_space.audit.cross_fit` with groups never predicting or calibrating a test scene with a model that
saw a scene of its group, and never calibrating it with a scene of its group; a whole
:func:`~measured_space.audit.run_audit` with groups (fold table, the same guarantee in its saved calibration
predictions, and singleton groups reproducing the default run exactly); ``load_labels(sources=...)`` and the
``--depth-sources`` / ``--fold-groups`` options of ``scripts/run_audit.py``; and the footprint IoU and
connected-component logic of ``scripts/group_smartspaces_layouts.py`` on toy rectangles.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from measured_space.audit import (AuditConfig, cross_fit, fold_table, load_labels, run_audit, scene_folds,
                                  synthetic_pairs)
from measured_space.qa_schema import QAItem, QuestionSpec, Referent, write_jsonl

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FAST = AuditConfig(n_boot=50, ridge_grid=(1.0,), tau_scales=(1.0,))     # fixed ridge keeps the audits fast


def _script(name: str) -> ModuleType:
    """Import a script of ``scripts/`` as a module, registered in ``sys.modules``.

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


class _Recorder:
    """An instrumented model: each fit records its training scenes, and predictions name their model.

    Attributes
    ----------
    trained : list of frozenset of str
        Training scenes of every fit, in fit order (a model is its position here).
    """

    def __init__(self) -> None:
        """Start with no fits."""
        self.trained: list[frozenset[str]] = []

    def fit(self, rows: pd.DataFrame) -> int:
        """Record a fit.

        Parameters
        ----------
        rows : pandas.DataFrame
            Training rows.

        Returns
        -------
        int
            The model's number.
        """
        self.trained.append(frozenset(rows["scene"].unique()))
        return len(self.trained) - 1

    def predict(self, model: int, rows: pd.DataFrame) -> dict[str, Any]:
        """Name the model that predicts each row.

        Parameters
        ----------
        model : int
            The model's number.
        rows : pandas.DataFrame
            Rows to predict.

        Returns
        -------
        dict
            ``model``: the model number per row.
        """
        return {"model": np.full(len(rows), model, dtype=np.int64)}


# Seven scenes in four groups of unequal size; group labels deliberately not in scene order.
GROUPS: dict[str, str] = {"scene00": "b", "scene01": "b", "scene02": "a", "scene03": "c", "scene04": "a",
                          "scene05": "d", "scene06": "c"}


def test_scene_folds_give_one_fold_per_group_numbered_by_first_scene() -> None:
    """One fold per group present, numbered in sorted order of each group's first scene, whatever
    ``max_folds`` and ``seed``; extra mapped scenes are ignored; an unmapped scene raises."""
    scenes = np.repeat(sorted(GROUPS), [5, 1, 9, 2, 2, 7, 3])
    expected = {"scene00": 0, "scene01": 0, "scene02": 1, "scene03": 2, "scene04": 1, "scene05": 3, "scene06": 2}
    for max_folds, seed in ((None, 0), (3, 0), (20, 5)):
        assert scene_folds(scenes, max_folds, seed, groups=GROUPS) == expected
    subset = scene_folds(["scene03", "scene05", "scene06"], 20, groups=GROUPS)    # groups c and d only
    assert subset == {"scene03": 0, "scene05": 1, "scene06": 0}
    with pytest.raises(ValueError, match="no fold group.*scene07"):
        scene_folds([*scenes, "scene07"], 20, groups=GROUPS)
    assert scene_folds(scenes, 20) == {s: i for i, s in enumerate(sorted(GROUPS))}    # default unchanged


def test_cross_fit_with_groups_never_uses_a_scene_of_the_test_scenes_group() -> None:
    """No test prediction comes from a model that saw a scene of the predicted scene's group; no calibration
    prediction comes from a model that saw a scene of the test scene's or the calibrated scene's group; each
    test scene is calibrated by exactly the scenes outside its group, each row once; ``K + K (K - 1) / 2`` fits."""
    pairs = synthetic_pairs(n_scenes=7, n_items=6, seed=2)
    rows = pairs[(pairs["qtype"] == "size_height") & pairs["matched"]]
    rec = _Recorder()
    test, cal = cross_fit(rows, rec.fit, rec.predict, max_folds=20, seed=0, groups=GROUPS)
    k = len(set(GROUPS.values()))
    assert len(rec.trained) == k + k * (k - 1) // 2
    scene = rows["scene"]

    def groups_seen(model: int) -> set[str]:
        """Groups of the scenes a model trained on.

        Parameters
        ----------
        model : int
            The model's number.

        Returns
        -------
        set of str
            Group labels.
        """
        return {GROUPS[s] for s in rec.trained[model]}

    assert sorted(test["row"]) == sorted(rows.index)
    assert not any(GROUPS[s] in groups_seen(m) for s, m in zip(scene.loc[test["row"]], test["model"]))
    cal_scene = scene.loc[cal["row"]].to_numpy()
    assert not any(GROUPS[s] in groups_seen(m) for s, m in zip(cal_scene, cal["model"]))
    assert not any(GROUPS[s] in groups_seen(m) for s, m in zip(cal["test_scene"], cal["model"]))
    for s, g in cal.groupby("test_scene"):
        assert set(scene.loc[g["row"]]) == {t for t in GROUPS if GROUPS[t] != GROUPS[s]}, s
        assert g["row"].is_unique
    with pytest.raises(ValueError, match="needs >= 3 folds"):          # two groups cannot leave two out
        cross_fit(rows, rec.fit, rec.predict, groups={s: "ab"[i % 2] for i, s in enumerate(sorted(GROUPS))})


def test_run_audit_with_groups_records_folds_and_keeps_groups_apart() -> None:
    """A whole audit with groups: the fold table lists every modelled type with its group, the saved
    calibration predictions never calibrate a test scene with its own group, a missing scene raises before
    anything is fitted, and singleton groups reproduce the default run's tables exactly."""
    pairs = synthetic_pairs(n_scenes=6, n_items=40, noise_scale=1.8, seed=3)
    groups = {"scene00": "x", "scene01": "y", "scene02": "x", "scene03": "z", "scene04": "y", "scene05": "z"}
    config = AuditConfig(n_boot=50, ridge_grid=(1.0,), tau_scales=(1.0,), fold_groups={"synthetic": groups})
    result = run_audit(pairs, config)
    folds = result.tables["folds"]
    assert list(folds.columns[:5]) == ["dataset", "qtype", "scene", "fold", "group"]
    assert set(folds["qtype"]) == {"size_height", "distance_center", "left_right"}
    assert (folds["group"] == folds["scene"].map(groups)).all() and (folds["fold_scenes"] == 2).all()
    assert folds.groupby("qtype")["fold"].nunique().eq(3).all()
    cal_scene = result.pairs.loc[result.calibration["row"], "scene"].to_numpy()
    assert not any(groups[a] == groups[b] for a, b in zip(cal_scene, result.calibration["test_scene"]))
    pd.testing.assert_frame_equal(fold_table(pairs[pairs["qtype"] == "left_right"], config)[["scene", "group"]],
                                  pd.DataFrame({"scene": sorted(groups), "group": [groups[s] for s in sorted(groups)]}))
    with pytest.raises(ValueError, match="no fold group"):
        run_audit(pairs, AuditConfig(n_boot=50, ridge_grid=(1.0,), fold_groups={"synthetic": {"scene00": "x"}}))
    singleton = run_audit(pairs, AuditConfig(n_boot=50, ridge_grid=(1.0,), tau_scales=(1.0,),
                                             fold_groups={"synthetic": {s: s for s in groups}}))
    default = run_audit(pairs, FAST)
    assert set(singleton.tables) == set(default.tables) | {"folds"}
    for name, table in default.tables.items():
        pd.testing.assert_frame_equal(singleton.tables[name], table, check_exact=True)
    pd.testing.assert_frame_equal(singleton.calibration, default.calibration, check_exact=True)


def test_audit_config_stores_fold_groups_as_plain_strings_and_stays_hashable() -> None:
    """``fold_groups`` is normalised to dicts of str, is excluded from the hash, and a bad shape raises."""
    config = AuditConfig(fold_groups={"d": {"s1": 1, "s2": "g"}})
    assert config.fold_groups == {"d": {"s1": "1", "s2": "g"}} and config.groups_of("d") == {"s1": "1", "s2": "g"}
    assert config.groups_of("other") is None and AuditConfig().groups_of("d") is None
    assert hash(config) == hash(AuditConfig())
    with pytest.raises(ValueError, match="fold_groups"):
        AuditConfig(fold_groups={"s1": "g"})                       # type: ignore[dict-item]


def _write_labels(folder: Path) -> None:
    """A label folder with two numeric questions answered by three depth sources.

    Parameters
    ----------
    folder : pathlib.Path
        Folder to write ``qa_oracle.jsonl`` and ``qa_auto_{laser,lidar,moge}.jsonl`` into.
    """
    refs = (Referent("a", "chair", "the chair"),)
    specs = [QuestionSpec("size_height_v1", "size_height", "synthetic", "s0", "cam0", f"f{i}", "img.png", refs,
                          "q?", "number", unit="m", tolerance={"abs": 0.1, "rel": 0.1}) for i in range(2)]
    write_jsonl([QAItem(s, 1.0, "oracle") for s in specs], folder / "qa_oracle.jsonl")
    for k, source in enumerate(("laser", "lidar", "moge")):
        write_jsonl([QAItem(s, 1.0 + 0.1 * k, f"auto_{source}") for s in specs], folder / f"qa_auto_{source}.jsonl")


def test_load_labels_keeps_only_the_requested_depth_sources(tmp_path: Path) -> None:
    """``sources`` keeps exactly the requested sources' rows, as the full table has them; an unknown or empty
    selection raises.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Scratch folder from pytest.
    """
    _write_labels(tmp_path)
    full = load_labels(tmp_path)
    assert sorted(full["depth_source"].unique()) == ["laser", "lidar", "moge"]
    lidar = load_labels(tmp_path, ["lidar"])
    expected = full[full["depth_source"] == "lidar"].reset_index(drop=True)
    pd.testing.assert_frame_equal(lidar, expected, check_exact=True)
    assert lidar.attrs["n_auto_only"] == {"lidar": 0} and lidar.attrs["n_not_run"] == {"lidar": 0}
    assert sorted(load_labels(tmp_path, ["moge", "laser"])["depth_source"].unique()) == ["laser", "moge"]
    with pytest.raises(ValueError, match="no qa_auto_<source>.jsonl for \\['LiDAR'\\]"):
        load_labels(tmp_path, ["LiDAR"])
    with pytest.raises(ValueError, match="none given"):
        load_labels(tmp_path, [])


def test_run_audit_options_parse_and_read_the_mapping(tmp_path: Path) -> None:
    """``--depth-sources`` and ``--fold-groups`` parse (default: unset), ``--depth-sources`` is refused with
    ``--synthetic``, and the mapping file must name datasets.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Scratch folder from pytest.
    """
    script = _script("run_audit")
    default = script.parse_args(["--results", "r"])
    assert default.depth_sources is None and default.fold_groups is None
    args = script.parse_args(["--results", "r", "--depth-sources", "lidar", "moge", "--fold-groups", "g.json"])
    assert args.depth_sources == ["lidar", "moge"] and args.fold_groups == Path("g.json")
    with pytest.raises(SystemExit):
        script.parse_args(["--synthetic", "--depth-sources", "lidar"])
    nested, flat = tmp_path / "nested.json", tmp_path / "flat.json"
    nested.write_text(json.dumps({"smartspaces": {"w0": "layout_0", "w1": 1}}))
    flat.write_text(json.dumps({"w0": "layout_0"}))
    assert script.read_fold_groups(nested) == {"smartspaces": {"w0": "layout_0", "w1": "1"}}
    with pytest.raises(ValueError, match="dataset"):
        script.read_fold_groups(flat)


def test_footprint_iou_and_connected_groups_on_toy_rectangles() -> None:
    """IoU against hand-computed values; groups are the connected components of IoU >= 0.5 (transitive),
    numbered by sorted first scene; camera centres are ``-R^T t``."""
    layouts = _script("group_smartspaces_layouts")
    unit = (0.0, 0.0, 1.0, 1.0)
    assert layouts.footprint_iou(unit, unit) == 1.0
    assert layouts.footprint_iou(unit, (2.0, 2.0, 3.0, 3.0)) == 0.0
    assert layouts.footprint_iou(unit, (0.5, 0.0, 1.5, 1.0)) == pytest.approx(0.5 / 1.5)
    assert layouts.footprint_iou((0.0, 0.0, 4.0, 1.0), (1.0, 0.0, 4.0, 1.0)) == pytest.approx(0.75)
    assert layouts.footprint_iou((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)) == 1.0     # degenerate, equal
    # a ~ b (IoU 0.6) and b ~ c (0.6) but a !~ c (1/3): one chain; d alone; e ~ f (0.75).
    rects = {"c": (2.0, 0.0, 6.0, 1.0), "a": (0.0, 0.0, 4.0, 1.0), "b": (1.0, 0.0, 5.0, 1.0),
             "d": (20.0, 20.0, 21.0, 21.0), "f": (10.0, 10.0, 12.0, 12.0), "e": (10.0, 10.0, 12.0, 11.5)}
    assert layouts.footprint_iou(rects["a"], rects["c"]) < 0.5 <= layouts.footprint_iou(rects["a"], rects["b"])
    assert layouts.connected_groups(rects) == {"a": 0, "b": 0, "c": 0, "d": 1, "e": 2, "f": 2}
    assert layouts.connected_groups(rects, min_iou=0.7) == {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 4}
    assert layouts.connected_groups(rects, min_iou=0.9) == {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    centre = np.array([3.0, -2.0, 5.0])
    extrinsic = np.hstack([rotation, (-rotation @ centre)[:, None]])          # world to camera: t = -R c
    calibration = {"sensors": [{"type": "camera", "extrinsicMatrix": extrinsic.tolist()},
                               {"type": "other", "extrinsicMatrix": extrinsic.tolist()},
                               {"extrinsicMatrix": np.hstack([np.eye(3), np.zeros((3, 1))]).tolist()}]}
    centres = layouts.camera_centres(calibration)
    np.testing.assert_allclose(centres, [centre, [0.0, 0.0, 0.0]])
    assert layouts.footprint(centres) == (0.0, -2.0, 3.0, 0.0)


def test_grouping_script_refuses_to_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An existing output file is left untouched and the script exits 1 before reading any calibration.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Scratch folder from pytest.
    capsys : pytest.CaptureFixture
        Captures the refusal.
    """
    out = tmp_path / "groups.json"
    out.write_text("keep")
    assert _script("group_smartspaces_layouts").main(["--out", str(out), "--scenes", "no/such/scene"]) == 1
    assert out.read_text() == "keep" and "refusing to overwrite" in capsys.readouterr().err
