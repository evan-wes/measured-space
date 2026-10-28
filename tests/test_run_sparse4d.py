"""Tests for ``scripts/run_sparse4d.py``: the TAO decoder, the class remap, re-centring and shifting back, the row
formats, the NGC config fills and the refusal to overwrite.

The decoder runs on hand-made logits whose result is worked out in each test's docstring. SparseQID's
``decode_box`` (TAO's: ``exp`` of the log sizes, ``atan2(sin, cos)`` for yaw) is replaced by a copy of that
formula, unless ``SPARSEQID_SRC`` names a SparseQID ``src`` folder, in which case the real one is used and also
compared with the copy. Nothing here needs a GPU, the weights or the data; torch is required (the script imports
it), omegaconf only for the config test, ``measured_space`` only for the cross-check with the converter.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPARSEQID_SRC = os.environ.get("SPARSEQID_SRC")

# The NGC checkpoints' class order (experiment.yaml ``dataset.classes``, as recorded in every run's meta) and the
# dataset's object type each name stands for.
NGC_CLASSES = ["person", "gr1_t2", "agility_digit", "nova_carter", "transporter", "forklift", "pallet_truck"]
NGC_TO_TYPE = {"person": "Person", "gr1_t2": "FourierGR1T2", "agility_digit": "AgilityDigit",
               "nova_carter": "NovaCarter", "transporter": "Transporter", "forklift": "Forklift",
               "pallet_truck": "PalletTruck"}


def _script(name: str) -> ModuleType:
    """Import a script of ``scripts/`` as a module, registered in ``sys.modules``.

    Parameters
    ----------
    name : str
        Script name without ``.py``.

    Returns
    -------
    types.ModuleType
        The module; importing it loads torch and numpy only (SparseQID and omegaconf are imported lazily).
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reference_decode_box(box: Any) -> Any:
    """TAO / SparseQID ``decode_box``: ``x, y, z, exp(log w, log l, log h), atan2(sin, cos), vx, vy, vz``.

    Parameters
    ----------
    box : torch.Tensor, shape (N, 11)
        Undecoded boxes.

    Returns
    -------
    torch.Tensor, shape (N, 10)
        Decoded boxes.
    """
    yaw = torch.atan2(box[:, 6], box[:, 7])
    return torch.cat([box[:, [0, 1, 2]], box[:, [3, 4, 5]].exp(), yaw[:, None], box[:, 8:]], dim=-1)


@pytest.fixture
def decoder(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The runner, with ``sparseqid.detector.decoder`` importable (the real one or a stub with the TAO formula).

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Restores ``sys.modules`` / ``sys.path`` afterwards.

    Returns
    -------
    types.ModuleType
        ``run_sparse4d``.
    """
    if SPARSEQID_SRC:
        monkeypatch.syspath_prepend(SPARSEQID_SRC)
        pytest.importorskip("sparseqid.detector.decoder")
    else:
        for name in ("sparseqid", "sparseqid.detector"):
            monkeypatch.setitem(sys.modules, name, ModuleType(name))
        stub = ModuleType("sparseqid.detector.decoder")
        stub.decode_box = reference_decode_box  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sparseqid.detector.decoder", stub)
    return _script("run_sparse4d")


def logits(p: list[list[float]]) -> Any:
    """Logits whose sigmoid is ``p``, shape ``(1, A, C)``, float32.

    Parameters
    ----------
    p : list of list of float
        Probabilities per anchor and class.

    Returns
    -------
    torch.Tensor
        ``log(p / (1 - p))``.
    """
    return torch.logit(torch.tensor([p], dtype=torch.float64)).float()


def centerness(c: list[float]) -> Any:
    """A quality tensor ``(1, A, 2)`` whose centerness (index 0) has sigmoid ``c``.

    Parameters
    ----------
    c : list of float
        Centerness probability per anchor.

    Returns
    -------
    torch.Tensor
        Quality output; the second channel (yawness) is 0.
    """
    q = torch.zeros(1, len(c), 2, dtype=torch.float64)
    q[0, :, 0] = torch.logit(torch.tensor(c, dtype=torch.float64))
    return q.float()


def box_preds(n: int) -> Any:
    """Undecoded boxes ``(1, n, 11)``: anchor k has centre (k, 2k, 0.5), size (1+k, 2+k, 3), yaw 0.1k, v (0.1k, 0, 0).

    Parameters
    ----------
    n : int
        Anchors.

    Returns
    -------
    torch.Tensor
        Box regression in TAO's layout (log sizes, sin and cos of yaw).
    """
    rows = [[k, 2 * k, 0.5, math.log(1 + k), math.log(2 + k), math.log(3), math.sin(0.1 * k), math.cos(0.1 * k),
             0.1 * k, 0.0, 0.0] for k in range(n)]
    return torch.tensor([rows], dtype=torch.float32)


def decoded(k: int) -> list[float]:
    """The decoded box of anchor ``k`` of :func:`box_preds`.

    Parameters
    ----------
    k : int
        Anchor.

    Returns
    -------
    list of float
        ``x, y, z, w, l, h, yaw, vx, vy, vz``.
    """
    return [k, 2 * k, 0.5, 1 + k, 2 + k, 3, 0.1 * k, 0.1 * k, 0.0, 0.0]


# Five anchors, three classes. Best class / raw score / centerness / score:
#   a0: class 0, 0.9, 0.5  -> 0.45      a1: class 1, 0.8, 0.75 -> 0.6      a2: class 2, 0.6, 0.5 -> 0.3
#   a3: class 0, 0.06, 0.5 -> 0.03 (raw >= 0.05: kept)  a4: class 0, 0.04, 0.9 -> 0.036 (raw < 0.05: masked)
PROBS = [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.05, 0.02, 0.6], [0.06, 0.01, 0.02], [0.04, 0.01, 0.02]]
CNS = [0.5, 0.75, 0.5, 0.5, 0.9]
IDS = torch.tensor([[10, 11, 12, 13, 14]])


def test_decoder_reweights_resorts_and_masks_on_the_raw_class_score(decoder: ModuleType) -> None:
    """Order by class x centerness (a1, a0, a2, a4, a3); the 0.05 mask is on the raw class score, so a4 goes.

    Parameters
    ----------
    decoder : types.ModuleType
        The runner (fixture).
    """
    out = decoder.tao_decode(logits(PROBS), box_preds(5), centerness(CNS), IDS, num_output=5, score_threshold=0.05)
    assert out["labels"].tolist() == [1, 0, 2, 0]
    assert out["ids"].tolist() == [11, 10, 12, 13]
    assert out["scores"].tolist() == pytest.approx([0.6, 0.45, 0.3, 0.03], abs=1e-6)
    assert out["cls_scores"].tolist() == pytest.approx([0.8, 0.9, 0.6, 0.06], abs=1e-6)
    expected = np.array([decoded(k) for k in (1, 0, 2, 3)])
    np.testing.assert_allclose(out["boxes"].numpy(), expected, atol=1e-5)


def test_decoder_top_k_before_reweighting_and_no_threshold(decoder: ModuleType) -> None:
    """``num_output=2`` keeps the two best raw class scores (a0 0.9, a1 0.8), then re-sorts them to a1, a0;
    without a threshold all five come back in score order, a4 (0.036) before a3 (0.03).

    Parameters
    ----------
    decoder : types.ModuleType
        The runner (fixture).
    """
    top2 = decoder.tao_decode(logits(PROBS), box_preds(5), centerness(CNS), IDS, num_output=2, score_threshold=0.05)
    assert top2["ids"].tolist() == [11, 10]
    everything = decoder.tao_decode(logits(PROBS), box_preds(5), centerness(CNS), IDS, num_output=5,
                                    score_threshold=None)
    assert everything["ids"].tolist() == [11, 10, 12, 14, 13]
    assert everything["scores"].tolist() == pytest.approx([0.6, 0.45, 0.3, 0.036, 0.03], abs=1e-6)


def test_decoder_detection_mode_flattens_anchor_classes(decoder: ModuleType) -> None:
    """Without ids, the top-k runs over (anchor, class) pairs: a0c0 0.9, a1c0 0.8, a0c1 0.7; times centerness
    (0.5, 0.75, 0.5) and re-sorted: a1c0 0.6, a0c0 0.45, a0c1 0.35.

    Parameters
    ----------
    decoder : types.ModuleType
        The runner (fixture).
    """
    out = decoder.tao_decode(logits([[0.9, 0.7], [0.8, 0.1]]), box_preds(2), centerness([0.5, 0.75]), None,
                             num_output=3, score_threshold=0.0)
    assert out["ids"] is None
    assert out["labels"].tolist() == [0, 0, 1]
    assert out["scores"].tolist() == pytest.approx([0.6, 0.45, 0.35], abs=1e-6)
    np.testing.assert_allclose(out["boxes"].numpy(), np.array([decoded(1), decoded(0), decoded(0)]), atol=1e-5)


def test_decoder_without_quality_keeps_the_raw_class_order(decoder: ModuleType) -> None:
    """No centerness: scores are the raw class scores, in top-k order (a0 0.9, a1 0.8, a2 0.6, a3 0.06).

    Parameters
    ----------
    decoder : types.ModuleType
        The runner (fixture).
    """
    out = decoder.tao_decode(logits(PROBS), box_preds(5), None, IDS, num_output=5, score_threshold=0.05)
    assert out["ids"].tolist() == [10, 11, 12, 13]
    assert out["scores"].tolist() == pytest.approx([0.9, 0.8, 0.6, 0.06], abs=1e-6)
    assert out["scores"].tolist() == out["cls_scores"].tolist()


@pytest.mark.skipif(not SPARSEQID_SRC, reason="SPARSEQID_SRC not set")
def test_reference_decode_box_matches_sparseqid(monkeypatch: pytest.MonkeyPatch) -> None:
    """The formula used as the stub is SparseQID's ``decode_box``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Puts SparseQID on ``sys.path``.
    """
    monkeypatch.syspath_prepend(str(SPARSEQID_SRC))
    real = pytest.importorskip("sparseqid.detector.decoder")
    box = box_preds(4)[0]
    assert torch.equal(real.decode_box(box), reference_decode_box(box))


def test_class_remap_is_the_challenge_table() -> None:
    """Every NGC class index maps to the 2026 id of the same object type, one to one; the names agree with the
    evaluation script's table.
    """
    run = _script("run_sparse4d")
    remap, names = run.MODEL_IDX_TO_CLASS_ID, run.CLASS_ID_TO_NAME
    assert sorted(remap) == list(range(7)) and sorted(remap.values()) == list(range(7))
    for index, ngc_name in enumerate(NGC_CLASSES):
        assert names[remap[index]] == NGC_TO_TYPE[ngc_name]
    assert names == _script("eval_sparse4d").CLASSES


def test_class_names_match_the_converter() -> None:
    """The runner's class table is the one :mod:`measured_space.sparse4d` reads the JSONL with."""
    sparse4d = pytest.importorskip("measured_space.sparse4d")
    assert _script("run_sparse4d").CLASS_ID_TO_NAME == sparse4d.CLASS_LABELS


@pytest.mark.skipif(not SPARSEQID_SRC, reason="SPARSEQID_SRC not set")
def test_class_remap_matches_sparseqid_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """SparseQID's model label of each object type maps to that type's challenge id.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Puts SparseQID on ``sys.path``.
    """
    monkeypatch.syspath_prepend(str(SPARSEQID_SRC))
    frame_dataset = pytest.importorskip("sparseqid.data.frame_dataset")
    run = _script("run_sparse4d")
    for object_type, model_label in frame_dataset.TYPE_TO_LABEL.items():
        assert run.CLASS_ID_TO_NAME[run.MODEL_IDX_TO_CLASS_ID[model_label]] == object_type


def _rot_z(angle: float) -> np.ndarray:
    """A rotation about +z.

    Parameters
    ----------
    angle : float
        Radians.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        The rotation matrix.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_camera_centre_offset_is_the_mean_centre_with_z_zeroed(tmp_path: Path) -> None:
    """Cameras at (1, 2, 3) and (3, 6, 5) (world-to-camera ``[R | -R C]``) give the offset (2, 4, 0); a
    non-camera sensor is ignored.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    """
    run = _script("run_sparse4d")
    sensors = []
    for centre, angle in (((1.0, 2.0, 3.0), 0.0), ((3.0, 6.0, 5.0), math.pi / 2)):
        r = _rot_z(angle)
        e = np.eye(4)
        e[:3, :3], e[:3, 3] = r, -r @ np.array(centre)
        sensors.append({"type": "camera", "id": f"Camera_{len(sensors):04d}", "extrinsicMatrix": e[:3].tolist()})
    sensors.append({"type": "lidar", "id": "x", "extrinsicMatrix": (np.eye(4)[:3] * 99).tolist()})
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"sensors": sensors}))
    np.testing.assert_allclose(run.camera_centre_offset(path), [2.0, 4.0, 0.0], atol=1e-12)


def test_frame_rows_shift_back_remap_and_format() -> None:
    """Centres move by the offset (sizes, yaw and velocity do not); model class 0 -> Person 0, 5 -> Forklift 1;
    object id = native id + 1; 0.2 for the rows, 0.05 for the JSONL; 4 decimals in the rows.
    """
    run = _script("run_sparse4d")
    boxes = np.array([[1.0, 2.0, 0.9, 0.5, 0.6, 1.8, 0.25, 0.1, 0.2, 0.0],
                      [-3.0, 4.0, 0.5, 2.0, 3.0, 1.0, -1.5, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    txt, jsonl = run.frame_rows(boxes, [0.8, 0.1, 0.01], [0.9, 0.2, 0.02], [0, 5, 3], [4, 7, 9],
                                np.array([10.0, -20.0, 0.0]), scene="Warehouse_022", scene_id=22, frame=6,
                                score_thr=0.2, jsonl_thr=0.05)
    assert txt == ["22 0 5 6 11.0000 -18.0000 0.9000 0.5000 0.6000 1.8000 0.2500\n"]
    lines = [json.loads(line) for line in jsonl]
    assert len(lines) == 2 and all(line.endswith("\n") for line in jsonl)
    assert lines[0] == {"scene": "Warehouse_022", "frame": 6, "class_id": 0, "class_name": "Person",
                        "model_class_index": 0, "score": 0.8, "cls_score": 0.9, "track_id": 4,
                        "center": [11.0, -18.0, 0.9], "size_wlh": [0.5, 0.6, 1.8], "yaw": 0.25,
                        "velocity": [0.1, 0.2, 0.0]}
    assert (lines[1]["class_id"], lines[1]["class_name"], lines[1]["model_class_index"]) == (1, "Forklift", 5)
    assert lines[1]["center"] == [7.0, -16.0, 0.5] and lines[1]["track_id"] == 7
    empty_txt, empty_jsonl = run.frame_rows(np.zeros((0, 10)), [], [], [], [], np.zeros(3), scene="s",
                                            scene_id=1, frame=0, score_thr=0.2, jsonl_thr=0.05)
    assert empty_txt == [] and empty_jsonl == []


def test_frame_ids_sha256_of_the_twelve_scene_window() -> None:
    """Frames 0, 6, ..., 8994 hash to the ``ids_sha256`` recorded by the 12-scene runs."""
    run = _script("run_sparse4d")
    assert run.frame_ids_sha256(range(0, 9000, 6)) == \
        "e8e7fb29958bf26a2e193e1bd9950d2cbcb39d6abbdae350a05aca408b642365"


def test_defaults_are_the_protocol() -> None:
    """Score 0.2, decoder 0.0, JSONL 0.05, RGB, re-centred, fp32, 4 workers, cuda:0, 24 GiB cap."""
    run = _script("run_sparse4d")
    args = run.parse_args(["--sparseqid-src", "s", "--model-dir", "m", "--ckpt", "c", "--tag", "t",
                           "--data-root", "d", "--cache-root", "f", "--split", "val", "--scene", "W",
                           "--scene-id", "1", "--out-dir", "o"])
    assert (args.score_thr, args.dec_thr, args.jsonl_thr) == (0.2, 0.0, 0.05)
    assert (args.network_channels, args.no_recenter, args.amp) == ("rgb", False, False)
    assert (args.num_workers, args.device, args.vram_cap_gib, args.max_frames) == (4, "cuda:0", 24.0, None)
    assert args.backbone_if_missing is None


def test_refuses_to_overwrite_before_loading_anything(tmp_path: Path) -> None:
    """An existing output file stops the run at once; nothing else is created.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    """
    run = _script("run_sparse4d")
    existing = tmp_path / "out" / "tag" / "Warehouse_001.jsonl"
    existing.parent.mkdir(parents=True)
    existing.write_text("keep\n")
    with pytest.raises(FileExistsError):
        run.main(["--sparseqid-src", str(tmp_path / "missing"), "--model-dir", str(tmp_path / "missing"),
                  "--ckpt", "c.pth", "--tag", "tag", "--data-root", str(tmp_path), "--cache-root", str(tmp_path),
                  "--split", "train", "--scene", "Warehouse_001", "--scene-id", "1",
                  "--out-dir", str(tmp_path / "out"), "--device", "cpu"])
    assert existing.read_text() == "keep\n"
    assert sorted(p.name for p in existing.parent.iterdir()) == ["Warehouse_001.jsonl"]
    paths = run.output_paths(tmp_path / "out", "other", "W")
    assert [p.name for p in paths] == ["W.txt", "W.jsonl", "W.meta.json"]


def test_ngc_config_fills(tmp_path: Path) -> None:
    """The anchor, both thresholds, a missing backbone and the three deformable keys are filled and noted; a
    missing backbone without ``backbone_if_missing`` and two anchors are errors.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    """
    pytest.importorskip("omegaconf")
    run = _script("run_sparse4d")
    model = tmp_path / "model"
    model.mkdir()
    (model / "experiment.yaml").write_text(
        "dataset:\n  classes: [person, gr1_t2]\n"
        "model:\n  head:\n    num_output: 300\n    instance_bank:\n      anchor: null\n"
        "    decoder:\n      score_threshold: 0.2\n    deformable_model:\n      embed_dims: 256\n")
    np.save(model / "anchors.npy", np.zeros((2, 11)))
    with pytest.raises(KeyError):
        run.load_ngc_config(model, 0.0)
    cfg, notes = run.load_ngc_config(model, 0.0, "resnet_101")
    assert cfg.model.head.instance_bank.anchor == str(model / "anchors.npy")
    assert cfg.model.head.score_threshold == 0.0 and cfg.model.head.decoder.score_threshold == 0.0
    assert cfg.model.backbone.type == "resnet_101"
    deform = cfg.model.head.deformable_model
    assert (deform.num_cams, deform.max_num_cams, deform.proj_drop, deform.embed_dims) == (6, 20, 0.0, 256)
    assert len(notes) == 6
    np.save(model / "second.npy", np.zeros(1))
    with pytest.raises(FileNotFoundError):
        run.load_ngc_config(model, 0.0, "resnet_101")
