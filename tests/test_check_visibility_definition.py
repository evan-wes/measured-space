"""Tests for ``scripts/check_visibility_definition.py``: the clearly-visible definitions, recall from stored records,
the per-sample visibility states (checked against :func:`measured_space.visibility.object_in_view`) and the
hole statistics of a depth map.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from measured_space.geometry import Camera, OrientedBox, make_pose
from measured_space.qa import QAConfig
from measured_space.scene import FrameView, SceneObject
from measured_space.visibility import object_in_view

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_visibility_definition.py"
CAM = Camera(500.0, 500.0, 319.5, 239.5, 640, 480)
LOOK_ALONG_Y = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])   # columns: camera x, y, z in the world


def load_script() -> ModuleType:
    """Import the script as a module.

    Returns
    -------
    types.ModuleType
        The module; importing it has no other side effects.
    """
    if "check_visibility_definition" in sys.modules:
        return sys.modules["check_visibility_definition"]
    spec = importlib.util.spec_from_file_location("check_visibility_definition", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_visibility_definition"] = module
    spec.loader.exec_module(module)
    return module


def record(visible: float, truncated: float, no_depth: float, *, n: int = 100, paired: bool = True,
           object_id: str = "o") -> dict[str, Any]:
    """A stored oracle-object line with depth-based visibility.

    Parameters
    ----------
    visible, truncated, no_depth : float
        Shares of the camera-facing samples.
    n : int, default 100
        Camera-facing samples.
    paired : bool, default True
        Whether source ``lidar`` paired it.
    object_id : str, default "o"
        Object id.

    Returns
    -------
    dict
        The record.
    """
    return {"object_id": object_id, "visibility": "laser", "visible_fraction": visible,
            "truncated_fraction": truncated, "bbox2d": [0.0, 0.0, 10.0, 10.0] if visible > 0 else None,
            "diagnostics": {"n_samples": float(n), "no_depth_fraction": no_depth},
            "match": {"lidar": {"detection": 0} if paired else None}}


def test_membership_separates_truncation_and_depth_holes() -> None:
    """Each definition on hand-made records, including an undetermined one and a refused proxy record."""
    m = load_script()
    config = QAConfig()
    # Fully in frame, nothing hidden: clearly visible under every definition.
    assert m.membership(record(1.0, 0.0, 0.0), config, 10) == {"current": True, "in_frame": True, "known_depth": True}
    # 20% cut off, 25% visible, 20% on depth holes: 0.25 < 0.3, but 0.25 / 0.8 and 0.25 / 0.6 clear 0.3.
    assert m.membership(record(0.25, 0.2, 0.2), config, 10) == {"current": False, "in_frame": True,
                                                                 "known_depth": True}
    # Holes matter only for known_depth: 0.2 / 0.8 = 0.25 fails, 0.2 / 0.6 = 0.33 passes.
    assert m.membership(record(0.2, 0.2, 0.2), config, 10) == {"current": False, "in_frame": False,
                                                                "known_depth": True}
    # Too much outside the image fails every definition (the fridge of spot-check card 1).
    assert m.membership(record(0.209, 0.616, 0.042, n=980), config, 10) == {"current": False, "in_frame": False,
                                                                             "known_depth": False}
    # Only 5 in-frame samples with depth: undetermined at min_known 10, decided at 1.
    rec = record(0.05, 0.0, 0.95, n=100)
    assert m.membership(rec, config, 10)["known_depth"] is None
    assert m.membership(rec, config, 1)["known_depth"] is True
    with pytest.raises(ValueError, match="depth diagnostics"):
        m.membership({"object_id": "p", "visibility": "gt_box2d", "visible_fraction": 0.5, "truncated_fraction": 0.0,
                      "bbox2d": [0, 0, 1, 1], "diagnostics": {"gt_visible": True}}, config, 10)


def test_recall_rows_and_population_summary() -> None:
    """Present objects, the three populations and their recall on four records."""
    m = load_script()
    config = QAConfig()
    records = [record(1.0, 0.0, 0.0, object_id="a"),                       # clearly visible everywhere, paired
               record(0.25, 0.2, 0.2, paired=False, object_id="b"),        # gained by in_frame and known_depth
               record(0.2, 0.2, 0.2, object_id="c"),                       # gained by known_depth only
               record(0.0, 1.0, 0.0, paired=False, object_id="d")]         # not present
    summary = m.population_summary(records, config, 10)
    assert summary["n_present"] == 3
    assert summary["current"] == {"clearly_visible": 1, "undetermined": 0, "gained": 0, "lost": 0}
    assert summary["in_frame"] == {"clearly_visible": 2, "undetermined": 0, "gained": 1, "lost": 0}
    assert summary["known_depth"] == {"clearly_visible": 3, "undetermined": 0, "gained": 2, "lost": 0}
    rows = {r["population"]: r for r in m.recall_rows(records, "v0", config, 10)}
    assert (rows["present"]["paired"], rows["present"]["n"]) == (2, 3)
    assert (rows["clearly_visible:current"]["paired"], rows["clearly_visible:current"]["n"]) == (1, 1)
    assert (rows["clearly_visible:in_frame"]["paired"], rows["clearly_visible:in_frame"]["n"]) == (1, 2)
    assert (rows["clearly_visible:known_depth"]["paired"], rows["clearly_visible:known_depth"]["n"]) == (2, 3)
    assert rows["clearly_visible:known_depth"]["wilson_lo"] < 2 / 3 < rows["clearly_visible:known_depth"]["wilson_hi"]


def test_in_frame_makeup() -> None:
    """Only present, untruncated objects count; hidden = 1 - visible - no depth."""
    m = load_script()
    records = [record(0.6, 0.0, 0.2), record(1.0, 0.0, 0.0), record(0.5, 0.3, 0.1), record(0.0, 0.0, 1.0)]
    makeup = m.in_frame_makeup(records, QAConfig())
    assert makeup["n"] == 2
    assert makeup["median"] == pytest.approx(0.8)
    assert makeup["share_ge_0.9"] == pytest.approx(0.5)
    assert makeup["mean_no_depth"] == pytest.approx(0.1)
    assert makeup["mean_hidden"] == pytest.approx(0.1)


def test_sample_states_agree_with_object_in_view() -> None:
    """A box in front of a wall, half behind a near occluder and partly over a depth hole."""
    m = load_script()
    view = FrameView("arkitscenes", "synthetic", "wide", "0", CAM, make_pose(LOOK_ALONG_Y, np.zeros(3)), "img.png",
                     up=np.array([0.0, 0.0, 1.0]))
    obj = SceneObject("box-1", "cabinet", OrientedBox([0.0, 4.0, 0.0], [1.0, 1.0, 1.0], np.eye(3)))
    depth = np.full((240, 320), 8.0, dtype=np.float32)       # half resolution: the lookup must rescale
    depth[:, :160] = 2.0                                      # an occluder over the left half
    depth[:120, 160:] = 0.0                                   # a hole over the top right
    uv, face, state, rows, cols = m.sample_states(view, obj, depth)
    seen, diag = object_in_view(view, obj, depth)
    assert len(state) == diag["n_samples"]
    assert (state == "visible").mean() == pytest.approx(seen.visible_fraction)
    assert (state == "outside").mean() == pytest.approx(seen.truncated_fraction)
    assert (state == "no_depth").mean() == pytest.approx(diag["no_depth_fraction"])
    assert {"visible", "hidden", "no_depth"} <= set(state.tolist())
    # The camera sees only the face whose outward normal is -y (towards it).
    assert {tuple(f) for f in face.tolist()} == {(1, -1)}
    assert (rows[state == "outside"] == -1).all() and (cols[state != "outside"] >= 0).all()


def test_hole_stats() -> None:
    """One 1-px dropout and one 3x3 hole in a 10x10 map."""
    m = load_script()
    depth = np.ones((10, 10), dtype=np.float32)
    depth[0, 0] = 0.0
    depth[5:8, 5:8] = np.nan
    stats = m.hole_stats(depth)
    assert stats["valid_share"] == pytest.approx(0.9)
    assert stats["n_holes"] == 2
    assert stats["largest_hole_px"] == 9
    assert stats["small_hole_share"] == pytest.approx(0.1)


def test_recall_cli_writes_csv_and_refuses_overwrite(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The ``recall`` subcommand on a tiny label folder."""
    m = load_script()
    folder = tmp_path / "labels_tiny"
    (folder / "arkitscenes").mkdir(parents=True)
    with open(folder / "arkitscenes" / "oracle_objects.jsonl", "w", encoding="utf-8") as f:
        for r in (record(1.0, 0.0, 0.0, object_id="a"), record(0.25, 0.2, 0.2, paired=False, object_id="b")):
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "recall.csv"
    assert m.main(["recall", "--labels", str(folder), "--out", str(out)]) == 0
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["population"] for r in rows] == ["present", "clearly_visible:current", "clearly_visible:in_frame",
                                               "clearly_visible:known_depth"]
    assert [int(r["n"]) for r in rows] == [2, 1, 2, 2]
    assert "2 present objects" in capsys.readouterr().out
    assert m.main(["recall", "--labels", str(folder), "--out", str(out)]) == 2
