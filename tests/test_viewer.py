"""Tests for the pairing viewer on synthetic stored records (no real data, no models).

The view is a camera at the origin looking along world +y (world z up) at three boxes: a chair and a table side
by side, and a sofa further right; a fourth object is behind the camera. Detection 0 ("chair") straddles the
chair and the table, overlapping both above the 0.3 IoU threshold, so its pair with the chair is ambiguous;
detection 1 ("sofa") sits on the sofa alone, a clear pair. Two objects and two detections cannot give one
ambiguous and one clear pair (both pairs would share the same runner-up overlap), hence the third box.
Records are written by hand in the shard layout of ``scripts/run_labels.py``.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from measured_space.geometry import Camera, OrientedBox, make_pose
from measured_space.pipeline import PipelineConfig, box_json, projected_extent, visible_box_edges
from measured_space.scene import FrameView
from measured_space.viewer import (
    GT_COLOUR,
    PAIR_AMBIGUOUS,
    Pair,
    box_wireframe,
    build_viewer,
    draw_pair,
    draw_view,
    frame_picture,
    html_problems,
    legend_items,
    load_view_records,
    sample_pairs,
    summarize_answers,
    view_data,
    view_flags,
    view_pairs,
    view_table,
    wilson_interval,
)

CAM = Camera(500.0, 500.0, 319.5, 239.5, 640, 480)
LOOK_ALONG_Y = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])   # columns: camera x, y, z in the world
CUBE = [0.8, 0.8, 0.8]
OBJECTS: dict[str, tuple[str, OrientedBox]] = {
    "chair-0001": ("chair", OrientedBox([-0.5, 4.0, 0.0], CUBE, np.eye(3))),
    "table-0002": ("table", OrientedBox([0.2, 4.0, 0.0], CUBE, np.eye(3))),
    "sofa-0003": ("sofa", OrientedBox([1.6, 5.0, 0.0], CUBE, np.eye(3))),
    "hidden-0004": ("cabinet", OrientedBox([0.0, -3.0, 0.0], CUBE, np.eye(3))),   # behind the camera
}
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_viewer.py"


def make_view(scene: str = "s1", frame: str = "0") -> FrameView:
    """The synthetic view.

    Parameters
    ----------
    scene : str, default "s1"
        Scene name.
    frame : str, default "0"
        Frame name.

    Returns
    -------
    FrameView
        Camera at the origin looking along +y.
    """
    return FrameView("arkitscenes", scene, "wide", frame, CAM, make_pose(LOOK_ALONG_Y, np.zeros(3)), "img.png")


def visible_edges(object_id: str) -> tuple[float, float, float, float]:
    """An object's visible 2D box (its whole projection here) in the edge convention.

    Parameters
    ----------
    object_id : str
        One of :data:`OBJECTS` in front of the camera.

    Returns
    -------
    tuple of float
        ``(xmin, ymin, xmax, ymax)``, pixel edges at integers.
    """
    extent = projected_extent(make_view(), OBJECTS[object_id][1])
    assert extent is not None
    return visible_box_edges(extent, "laser")


def make_records(scene: str = "s1", frame: str = "0", source: str = "lidar",
                 detections: tuple[int, ...] = (0, 1)) -> dict[str, Any]:
    """Stored records of the synthetic view, as a shard holds them.

    Parameters
    ----------
    scene : str, default "s1"
        Scene name.
    frame : str, default "0"
        Frame name.
    source : str, default "lidar"
        The one automatic source.
    detections : tuple of int, default (0, 1)
        Which of the two detections to include (renumbered from 0).

    Returns
    -------
    dict
        ``view``, ``oracle_objects``, ``detections`` and ``qa`` records.
    """
    view = make_view(scene, frame)
    key = f"arkitscenes/{scene}/wide/{frame}"
    identity = {"view_key": key, "dataset": "arkitscenes", "scene": scene, "camera_id": "wide", "frame": frame}
    oracle = []
    for oid, (label, box) in OBJECTS.items():
        extent = projected_extent(view, box)
        oracle.append({**identity, "object_id": oid, "label": label, "box": box_json(box),
                       "visible_fraction": 0.0 if extent is None else 0.9, "truncated_fraction": 0.0,
                       "bbox2d": None if extent is None else list(extent), "present": extent is not None,
                       "diagnostics": {}, "match": None})
    chair, table = visible_edges("chair-0001"), visible_edges("table-0002")
    straddle = (chair[0] + 55, chair[1], table[2] - 55, chair[3])          # IoU 0.38 chair, 0.37 table
    sofa = tuple(np.add(visible_edges("sofa-0003"), [5.0, 3.0, -4.0, 2.0]))
    shift = np.array([0.05, 0.0, 0.0])
    candidates = [("chair", straddle, OrientedBox(OBJECTS["chair-0001"][1].center + shift, CUBE, np.eye(3))),
                  ("sofa", sofa, OrientedBox(OBJECTS["sofa-0003"][1].center - shift, CUBE, np.eye(3)))]
    det_records = []
    for index, which in enumerate(detections):
        phrase, box2d, box = candidates[which]
        det_records.append({**identity, "detection": index, "source": source, "phrase": phrase,
                            "labels": [phrase], "score": 0.8 - 0.1 * which, "box2d": [float(v) for v in box2d],
                            "mask": {}, "lift_stride": 1, "lift_error": None, "box": box_json(box),
                            "features": {}, "status": "matched", "match": None})
    view_record = {**identity, "image_ref": "img.png",
                   "camera": {"fx": CAM.fx, "fy": CAM.fy, "cx": CAM.cx, "cy": CAM.cy, "width": CAM.width,
                              "height": CAM.height},
                   "T_wc": view.T_wc.tolist(), "up": [0.0, 0.0, 1.0], "visibility": "laser",
                   "depth_sources": [source], "n_detections": len(det_records),
                   "config": PipelineConfig().to_json()}
    return {"view": view_record, "oracle_objects": oracle, "detections": det_records, "qa": {"oracle": []}}


def write_shard(root: Path, records: dict[str, Any]) -> None:
    """Write records as a shard of a label folder.

    Parameters
    ----------
    root : pathlib.Path
        The label folder.
    records : dict
        A view's records.
    """
    v = records["view"]
    path = root / v["dataset"] / "shards" / f"{v['scene']}__{v['camera_id']}__{v['frame']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": records}))


def grey_image(view: FrameView) -> NDArray[np.uint8]:
    """A mid-grey image of a view's size (stands in for the dataset image).

    Parameters
    ----------
    view : FrameView
        The view.

    Returns
    -------
    numpy.ndarray, shape (H, W, 3), uint8
        Constant 90.
    """
    return np.full((view.camera.height, view.camera.width, 3), 90, dtype=np.uint8)


def synthetic_pairs(counts: dict[str, tuple[int, int]]) -> list[Pair]:
    """Fake pairs: per scene, a number of ambiguous and clear pairs.

    Parameters
    ----------
    counts : dict of str to (int, int)
        Scene -> ``(n_ambiguous, n_clear)``.

    Returns
    -------
    list of Pair
        The pairs.
    """
    pairs = []
    for scene, (n_amb, n_clear) in counts.items():
        for i in range(n_amb + n_clear):
            amb = i < n_amb
            key = f"arkitscenes/{scene}/wide/{i}"
            pairs.append(Pair(f"lidar:{key}:o{i}:0", "lidar", key, "arkitscenes", scene, f"o{i}", "chair", 0,
                              "chair", 0.9, 0.6, 0.4 if amb else 0.1, amb, True, 0.2, 0.5, (0, 0, 10, 10),
                              (1, 1, 11, 11)))
    return pairs


def test_pairing_is_recomputed_and_flags_the_ambiguous_pair() -> None:
    """The 2D association pairs the straddling detection ambiguously, the other clearly, from the records."""
    records = make_records()
    data = view_data([("folder", records)], ["lidar"])
    assert data.present_ids == {"chair-0001", "table-0002", "sofa-0003"}           # the hidden one is absent
    pairs = {p.object_id: p for p in view_pairs(data, "lidar")}
    assert set(pairs) == {"chair-0001", "sofa-0003"}                                # the table is missed
    chair, sofa = pairs["chair-0001"], pairs["sofa-0003"]
    assert chair.ambiguous and chair.runner_up_iou2d >= 0.3 and chair.iou2d > chair.runner_up_iou2d
    assert not sofa.ambiguous and sofa.runner_up_iou2d == 0.0
    assert chair.label_correct and sofa.label_correct
    assert chair.distance == pytest.approx(0.05) and chair.pair_id == "lidar:arkitscenes/s1/wide/0:chair-0001:0"
    assert view_flags(data, ["lidar"]) == (True, False)
    table = view_table(data, ["lidar"])
    assert html_problems(f"<!doctype html><html><body>{table}</body></html>") == []
    assert table.count('class="amb"') == 1 and "missed" in table


def test_overlay_and_spot_check_pictures() -> None:
    """The overlay draws the ground truth and the ambiguous pair line; the card pictures crop around the pair."""
    records = make_records()
    data = view_data([("folder", records)], ["lidar"])
    image = grey_image(data.view)
    picture = draw_view(image, data, ["lidar"])
    assert picture.size == (640, 480)
    pixels = np.asarray(picture).reshape(-1, 3)
    assert np.any(np.all(pixels == GT_COLOUR, axis=1)) and np.any(np.all(pixels == PAIR_AMBIGUOUS, axis=1))
    framed = frame_picture(picture, "title", legend_items(["lidar"], "lidar"))
    assert framed.width == 640 and framed.height > 480
    with pytest.raises(ValueError):
        draw_view(image[:100], data, ["lidar"])
    chair = next(p for p in view_pairs(data, "lidar") if p.object_id == "chair-0001")
    crop, frame = draw_pair(image, data, chair)
    assert max(crop.size) == 480 and frame.size == (640, 480)


def test_box_wireframe_is_clipped_to_the_near_plane() -> None:
    """A box in front gives 12 edges; one crossing the camera plane gives finite, shortened ones; behind, none."""
    view = make_view()
    assert len(box_wireframe(view, OBJECTS["chair-0001"][1])) == 12
    crossing = box_wireframe(view, OrientedBox([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], np.eye(3)))
    assert 0 < len(crossing) <= 12 and np.all(np.isfinite(np.array(crossing)))
    assert box_wireframe(view, OBJECTS["hidden-0004"][1]) == []


def test_sample_is_seeded_balanced_and_stratified_by_scene() -> None:
    """Same seed, same sample in any input order; half ambiguous; scenes share each half evenly."""
    pairs = synthetic_pairs({"a": (2, 30), "b": (10, 30), "c": (10, 5)})
    sample = sample_pairs(pairs, 20, seed=7)
    shuffled = list(pairs)
    random.Random(1).shuffle(shuffled)
    assert [p.pair_id for p in sample_pairs(shuffled, 20, seed=7)] == [p.pair_id for p in sample]
    assert [p.pair_id for p in sample_pairs(pairs, 20, seed=8)] != [p.pair_id for p in sample]
    assert len({p.pair_id for p in sample}) == 20
    amb = [p.scene for p in sample if p.ambiguous]
    clear = [p.scene for p in sample if not p.ambiguous]
    assert len(amb) == 10 and len(clear) == 10
    assert sorted(amb.count(s) for s in "abc") == [2, 4, 4]                # scene a has only 2 ambiguous pairs
    assert max(clear.count(s) for s in "abc") - min(clear.count(s) for s in "abc") <= 1
    short = sample_pairs(synthetic_pairs({"a": (1, 30), "b": (2, 30)}), 20, seed=0)
    assert sum(p.ambiguous for p in short) == 3 and len(short) == 13          # fewer, not topped up
    with pytest.raises(ValueError):
        sample_pairs(pairs, -1, seed=0)


def test_wilson_interval_and_answer_summary() -> None:
    """Wilson bounds match known values; the summary splits by ambiguity and reweights by population."""
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.27753), abs=1e-5)
    assert wilson_interval(5, 10) == pytest.approx((0.23659, 0.76341), abs=1e-5)
    assert wilson_interval(0, 0) is None
    answers = ([{"pair_id": f"a{i}", "ambiguous": True, "verdict": "wrong" if i < 3 else "right"} for i in range(6)]
               + [{"pair_id": f"c{i}", "ambiguous": False, "verdict": "wrong" if i < 1 else "right"}
                  for i in range(4)]
               + [{"pair_id": "c4", "ambiguous": False, "verdict": "cant_tell"},
                  {"pair_id": "c5", "ambiguous": False, "verdict": None}])
    summary = summarize_answers({"sample_id": "x", "seed": 0, "source": "lidar",
                                 "population": {"ambiguous": 10, "clear": 90}, "answers": answers})
    assert summary["ambiguous"]["wrong_rate"] == pytest.approx(0.5)
    assert summary["clear"]["wrong_rate"] == pytest.approx(0.25)
    assert summary["clear"]["n_cant_tell"] == 1 and summary["clear"]["n_unanswered"] == 1
    assert summary["overall"]["n_judged"] == 10 and summary["overall"]["wrong_rate"] == pytest.approx(0.4)
    lo, hi = summary["overall"]["ci95"]
    assert lo < 0.4 < hi
    assert summary["weighted_wrong_rate"] == pytest.approx(0.1 * 0.5 + 0.9 * 0.25)
    with pytest.raises(ValueError):
        summarize_answers({"answers": [{"pair_id": "z", "ambiguous": False, "verdict": "maybe"}]})


def test_build_viewer_writes_well_formed_pages_and_refuses_a_non_empty_folder(tmp_path: Path) -> None:
    """Two scenes of synthetic shards give pages that parse, a manifest with the counts, and no overwrite."""
    labels = tmp_path / "labels"
    for scene in ("s1", "s2"):
        for frame in ("0", "1", "2"):
            write_shard(labels, make_records(scene, frame))
    out = tmp_path / "viewer" / "test"
    failing = {"arkitscenes/s2/wide/2"}

    def loader(view: FrameView) -> NDArray[np.uint8]:
        """Grey images, except one view whose image cannot be read.

        Parameters
        ----------
        view : FrameView
            The view.

        Returns
        -------
        numpy.ndarray
            A grey image.
        """
        if f"{view.dataset}/{view.scene}/{view.camera_id}/{view.frame}" in failing:
            raise OSError("unreadable")
        return grey_image(view)

    manifest = build_viewer([labels], out, sample_size=4, seed=3, max_views=2, load_image=loader, log=lambda m: None)
    for page in ("index.html", "spotcheck.html"):
        assert html_problems((out / page).read_text(encoding="utf-8")) == []
    assert manifest["n_views_selected"] == 6 and manifest["n_views_drawn"] == 4         # frames 0 and 2 per scene
    assert manifest["pairs_per_source"] == {"lidar": {"pairs": 12, "ambiguous": 6, "label_mismatch": 0}}
    spot = manifest["spotcheck"]
    assert spot["source"] == "lidar" and spot["population"] == {"ambiguous": 6, "clear": 6}
    assert spot["n_sampled_ambiguous"] == 2 and spot["n_sampled_clear"] == 2
    assert set(manifest["image_errors"]) == failing
    assert len(list((out / "images" / "views").glob("*.jpg"))) == 4
    assert len(list((out / "images" / "spotcheck").glob("*.jpg"))) == 8
    page = (out / "spotcheck.html").read_text(encoding="utf-8")
    meta = json.loads(page.split('type="application/json">', 1)[1].split("</script>", 1)[0])
    assert meta["sample_id"] == spot["sample_id"] and len(meta["items"]) == 4
    assert set(meta["items"][0]) == {"pair_id", "view_key", "object_id", "detection", "ambiguous"}
    with pytest.raises(FileExistsError):
        build_viewer([labels], out, load_image=grey_image, log=lambda m: None)


def test_second_label_folder_adds_a_source_and_must_agree(tmp_path: Path) -> None:
    """A source from another folder is paired with its own detections; a folder with other objects is refused."""
    first, second, other = tmp_path / "v0", tmp_path / "cutr", tmp_path / "other"
    write_shard(first, make_records())
    write_shard(second, make_records(source="cutr_sam3", detections=(1,)))
    records = load_view_records([first, second])
    assert list(records) == ["arkitscenes/s1/wide/0"]
    data = view_data(records["arkitscenes/s1/wide/0"], ["lidar", "cutr_sam3"])
    assert [(p.object_id, p.detection, p.ambiguous) for p in view_pairs(data, "cutr_sam3")] == [("sofa-0003", 0, False)]
    assert data.sources["cutr_sam3"].folder == str(second)
    changed = make_records(source="cutr_sam3")
    changed["oracle_objects"][0]["label"] = "stool"
    write_shard(other, changed)
    with pytest.raises(ValueError):
        view_data(load_view_records([first, other])["arkitscenes/s1/wide/0"], ["cutr_sam3"])


def test_html_problems_reports_unbalanced_tags() -> None:
    """Unclosed and misnested elements are reported; void elements need no end tag."""
    assert html_problems('<!doctype html><html><body><img src="a.jpg"><p>x</p></body></html>') == []
    assert html_problems("<!doctype html><html><body><div><span></div></body></html>")
    assert html_problems("<html></html>") == ["no <!doctype html>"]


def load_script() -> ModuleType:
    """Import ``scripts/make_viewer.py`` as a module (it is not part of the package).

    Returns
    -------
    module
        The script's module.
    """
    spec = importlib.util.spec_from_file_location("make_viewer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_viewer"] = module
    spec.loader.exec_module(module)
    return module


def test_script_rejects_path_names_and_summarises_an_export(tmp_path: Path, capsys: pytest.CaptureFixture[str]
                                                            ) -> None:
    """``--name`` must be one folder level; ``--summarize-answers`` prints the summary and writes nothing."""
    script = load_script()
    for bad in ("../x", "a/b", ".hidden", ""):
        with pytest.raises(ValueError):
            script.viewer_dir(bad)
    export = tmp_path / "answers.json"
    export.write_text(json.dumps({"sample_id": "s", "seed": 0, "source": "lidar", "answers": [
        {"pair_id": "p", "view_key": "k", "object_id": "o", "detection": 0, "ambiguous": True, "verdict": "wrong",
         "note": ""}]}))
    assert script.main(["--summarize-answers", str(export)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["ambiguous"]["n_wrong"] == 1 and printed["clear"]["n_judged"] == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["answers.json"]
