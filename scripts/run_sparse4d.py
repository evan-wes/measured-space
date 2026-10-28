"""Run a frozen NVIDIA TAO Sparse4D NGC checkpoint over one Smart Spaces scene with SparseQID's pure-PyTorch runtime.

Runs in the separate ``mga-sparse4d`` conda env (``envs/sparse4d/README.md``), not in the project env:
``measured_space`` is not imported here (as for ``scripts/run_cubify.py``). Its input is the half-resolution JPEG
frame cache written by ``scripts/extract_smartspaces_frames.py``; its output is read by
``python -m measured_space.sparse4d convert`` (project env) and scored by ``scripts/eval_sparse4d.py``.

Detector only: SparseQID's identity model is NOT used. Boxes come from the detector's own decoder logic,
re-implemented here from TAO ``SparseBox3DDecoder.decode`` (:func:`tao_decode`), and track ids are the detector's
native instance-bank ids (``get_instance_id`` with the decoder threshold), exactly as TAO ``simple_test`` returns
them.

Protocol (EVA/GaugeAlign ``run_infer.py`` defaults for its validation studies):

* world frame re-centred on the mean camera centre (xy only, z untouched); predictions are shifted back, so every
  output box is in the dataset's own world frame (the frame of ``calibration.json`` / ``ground_truth.json``);
* network input channel order RGB by default (= EVA to_rgb=0: TAO loads PIL RGB and does not swap);
  ``--network-channels bgr`` gives SparseQID's own order (cv2 BGR -> RGB -> normalize(to_rgb=True) swaps back to
  BGR) = EVA to_rgb=1;
* model class index -> official 2026 class id remap {0:0, 1:4, 2:5, 3:2, 4:3, 5:1, 6:6}
  (:data:`MODEL_IDX_TO_CLASS_ID`);
* decoder score threshold 0.0 (``--dec-thr 0.0``), evaluation threshold on ``scores_3d`` 0.2 (``--score-thr 0.2``);
* fp32 (no autocast) unless ``--amp``.

Weights are loaded strictly (only the unused ``depth_branch.*`` keys are allowed to be unexpected) and never
modified. The NGC ``experiment.yaml`` is read unchanged from the model dir; a few keys SparseQID's builder requires
but the NGC yaml lacks are filled in memory and logged (:func:`load_ngc_config`).

Outputs (under ``--out-dir/<tag>/``):

* ``<scene>.txt``: 11-column AI City MTMC rows ``<scene_id> <class_id> <object_id> <frame> <x> <y> <z> <w> <l>
  <h> <yaw>`` for boxes with score >= ``--score-thr``, object ids = native id + 1, floats to 4 decimals;
* ``<scene>.jsonl``: one JSON object per decoded box with score >= ``--jsonl-thr`` (world frame, see the meta);
  the format is documented in :mod:`measured_space.sparse4d`;
* ``<scene>.meta.json``: run parameters, re-centring offset, timings, peak VRAM (written last).

Usage (GPU, from the repository root, with the ``mga-sparse4d`` env's Python)::

    python -I scripts/run_sparse4d.py --sparseqid-src <SparseQID>/src \\
        --model-dir <models>/sparse4d/rn50_trainable_v2.2 --ckpt sparse4d_warehouse_v2.2_r50.pth \\
        --tag rn50_trainable_v2.2 --data-root <data>/MTMC_Tracking_2026 --cache-root <frame_cache> \\
        --split train --scene Warehouse_000 --scene-id 0 --out-dir <outputs>/sparse4d --num-workers 6

The ResNet-101 checkpoint's yaml has no ``model.backbone``; pass ``--backbone-if-missing resnet_101`` (strict
loading verifies it). ``--max-frames N`` processes only the first N cached frames (checks); ``--device cpu`` runs
on the CPU. Refuses to overwrite existing outputs.

Provenance and licences:

* :func:`tao_decode` is adapted from NVIDIA TAO PyTorch (https://github.com/NVIDIA/tao_pytorch_backend, commit
  ``d99ff772``, the commit EVA/GaugeAlign pinned), ``nvidia_tao_pytorch/cv/sparse4d/model/detection3d/decoder.py``
  (``SparseBox3DDecoder.decode``), Apache-2.0; its notice is kept below.
* The detector, the frame loader and ``decode_box`` are imported, not copied, from SparseQID
  (https://github.com/playbox-dev/SparseQID, commit ``7aea61d``, Apache-2.0).
* :func:`camera_centre_offset` recomputes the shift SparseQID's ``FrameJpgDataset(recenter_mode="camera")``
  applies (``sparseqid/data/frame_dataset.py``), so that predictions can be moved back to the dataset's frame.
"""

# tao_decode() is adapted from NVIDIA TAO PyTorch, nvidia_tao_pytorch/cv/sparse4d/model/detection3d/decoder.py
# (SparseBox3DDecoder.decode), commit d99ff772, which carries the notice:
#   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#   SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# Changes: batch element 0 only; logits, quality and box regression are cast to float32 before use; the raw class
# score is returned next to the centerness-weighted score; outputs are a single dict of CPU tensors.

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike

MODEL_IDX_TO_CLASS_ID: dict[int, int] = {0: 0, 1: 4, 2: 5, 3: 2, 4: 3, 5: 1, 6: 6}
"""dict of int to int: the NGC checkpoints' class index (yaml ``dataset.classes`` order: person, gr1_t2,
agility_digit, nova_carter, transporter, forklift, pallet_truck) -> the 2026 challenge's class id."""

CLASS_ID_TO_NAME: dict[int, str] = {
    0: "Person", 1: "Forklift", 2: "NovaCarter", 3: "Transporter",
    4: "FourierGR1T2", 5: "AgilityDigit", 6: "PalletTruck",
}
"""dict of int to str: the 2026 challenge's class id -> the dataset's object type (``measured_space.sparse4d``'s
``CLASS_LABELS``)."""

CNS = 0
"""int: index of centerness in the head's quality output (TAO ``box3d.py``)."""

FINAL_DIM = (540, 960)
"""tuple of int: network input (height, width), the frame cache's half resolution."""


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    """Return the hex SHA-256 of a file.

    Parameters
    ----------
    path : Path
        File to hash.
    chunk : int, optional
        Read size in bytes.

    Returns
    -------
    str
        Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_ngc_config(model_dir: Path, dec_thr: float, backbone_if_missing: str | None = None) -> tuple[Any, list[str]]:
    """Load the NGC ``experiment.yaml`` and fill the keys SparseQID's builder needs but the NGC file lacks.

    Parameters
    ----------
    model_dir : Path
        NGC model directory holding ``experiment.yaml`` and exactly one anchor ``.npy``.
    dec_thr : float
        Decoder score threshold; also used as the instance-id threshold (TAO passes ``decoder.score_threshold``).
    backbone_if_missing : str or None, optional
        Backbone type to use when the yaml has no ``model.backbone`` (NGC rn101 v1.1); strict weight loading then
        verifies it. Error if needed but not given.

    Returns
    -------
    tuple[Any, list[str]]
        The OmegaConf config and a list of human-readable notes on every in-memory fill.

    Raises
    ------
    FileNotFoundError
        If the model directory does not hold exactly one ``.npy``.
    KeyError
        If the yaml has no ``model.backbone`` and ``backbone_if_missing`` is None.
    """
    from omegaconf import OmegaConf

    anchors = sorted(model_dir.glob("*.npy"))
    if len(anchors) != 1:
        raise FileNotFoundError(f"expected exactly one anchor .npy in {model_dir}, found {anchors}")
    cfg = OmegaConf.load(str(model_dir / "experiment.yaml"))
    notes = []
    cfg.model.head.instance_bank.anchor = str(anchors[0])
    notes.append(f"model.head.instance_bank.anchor = {anchors[0].name}")
    OmegaConf.update(cfg, "model.head.score_threshold", float(dec_thr), force_add=True)
    OmegaConf.update(cfg, "model.head.decoder.score_threshold", float(dec_thr), force_add=True)
    notes.append(f"model.head.score_threshold = model.head.decoder.score_threshold = {dec_thr} (EVA --dec_thr)")
    if "backbone" not in cfg.model:
        if backbone_if_missing is None:
            raise KeyError("model.backbone absent from experiment.yaml; pass --backbone-if-missing")
        OmegaConf.update(cfg, "model.backbone.type", backbone_if_missing, force_add=True)
        notes.append(f"model.backbone.type = {backbone_if_missing} (absent in NGC yaml; verified by strict load)")
    deform = cfg.model.head.deformable_model
    for key, default in (("num_cams", 6), ("max_num_cams", 20), ("proj_drop", 0.0)):
        if key not in deform:
            OmegaConf.update(cfg, f"model.head.deformable_model.{key}", default, force_add=True)
            notes.append(f"model.head.deformable_model.{key} = {default} (absent in NGC yaml; "
                         "no effect on weights or inference with use_camera_embed=true)")
    return cfg, notes


def load_detector(sparseqid_src: Path, cfg: Any, ckpt_path: Path,
                  device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build SparseQID's Sparse4D from the NGC config and strictly load the NGC state dict.

    Parameters
    ----------
    sparseqid_src : Path
        ``SparseQID/src`` directory (added to ``sys.path``).
    cfg : Any
        OmegaConf config from :func:`load_ngc_config`.
    ckpt_path : Path
        NGC ``.pth``.
    device : torch.device
        Target device.

    Returns
    -------
    tuple[torch.nn.Module, dict[str, Any]]
        ``(detector, load_report)`` where ``load_report`` lists the ignored checkpoint keys.

    Raises
    ------
    RuntimeError
        If a model key is missing from the checkpoint, or the checkpoint has unexpected keys other than
        ``depth_branch.*``.
    """
    sys.path.insert(0, str(sparseqid_src))
    from sparseqid.detector.sparse4d import Sparse4D  # noqa: E402

    model = Sparse4D(config=cfg)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_unexpected = [k for k in unexpected if not k.startswith("depth_branch.")]
    if missing or bad_unexpected:
        raise RuntimeError(f"state mismatch: missing={missing[:10]} ({len(missing)}), "
                           f"unexpected={bad_unexpected[:10]} ({len(bad_unexpected)})")
    model.to(device).eval()
    return model, {"missing": len(missing), "ignored_unexpected": sorted({k.split('.')[0] for k in unexpected}),
                   "ignored_unexpected_count": len(unexpected)}


@torch.no_grad()
def tao_decode(cls_logits: torch.Tensor, box_preds: torch.Tensor, quality: torch.Tensor | None,
               instance_id: torch.Tensor | None, num_output: int,
               score_threshold: float | None) -> dict[str, torch.Tensor | None]:
    """Decode one frame exactly as TAO ``SparseBox3DDecoder.decode`` (tracking mode, batch element 0).

    Adapted from NVIDIA TAO PyTorch ``nvidia_tao_pytorch/cv/sparse4d/model/detection3d/decoder.py`` (commit
    ``d99ff772``, Apache-2.0; see the notice at the top of this file). The order of operations is TAO's: sigmoid
    class scores, squeezed to the best class per anchor when ids are given; top ``num_output`` by class score;
    the threshold mask taken on that raw class score; re-weighting by ``sigmoid(centerness)`` and a stable
    descending re-sort; then the mask. ``decode_box`` is SparseQID's (identical to TAO's).

    Parameters
    ----------
    cls_logits : torch.Tensor
        Last-layer classification logits, ``(B, A, C)``.
    box_preds : torch.Tensor
        Last-layer box regression, ``(B, A, 11)`` (undecoded: x, y, z, log w, log l, log h, sin, cos, vx, vy, vz).
    quality : torch.Tensor or None
        Last-layer quality output ``(B, A, 2)``; index 0 is centerness.
    instance_id : torch.Tensor or None
        Native instance-bank ids ``(B, A)``; when given, classes are squeezed per anchor (as in TAO).
    num_output : int
        Top-k kept (``model.head.num_output``).
    score_threshold : float or None
        Decoder threshold applied to the raw class score before centerness re-weighting.

    Returns
    -------
    dict[str, torch.Tensor or None]
        ``boxes`` (N, 10: x, y, z, w, l, h, yaw, vx, vy, vz), ``scores`` (class x sigmoid(centerness)),
        ``cls_scores`` (raw class), ``labels`` (model class index), ``ids`` (native ids or None); CPU tensors,
        highest ``scores`` first.
    """
    from sparseqid.detector.decoder import decode_box  # noqa: E402  (identical to TAO decode_box)

    squeeze_cls = instance_id is not None
    cls_scores = cls_logits.float().sigmoid()
    if squeeze_cls:
        cls_scores, cls_ids = cls_scores.max(dim=-1)
        cls_scores = cls_scores.unsqueeze(dim=-1)
    num_cls = cls_scores.shape[-1]
    cls_scores, indices = cls_scores.flatten(start_dim=1).topk(num_output, dim=1, sorted=True)
    if not squeeze_cls:
        cls_ids = indices % num_cls
    mask = cls_scores >= score_threshold if score_threshold is not None else None
    cls_scores_origin = cls_scores.clone()
    if quality is not None:
        centerness = torch.gather(quality.float()[..., CNS], 1, indices // num_cls)
        cls_scores = cls_scores * centerness.sigmoid()
        cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
        if not squeeze_cls:
            cls_ids = torch.gather(cls_ids, 1, idx)
        if mask is not None:
            mask = torch.gather(mask, 1, idx)
        indices = torch.gather(indices, 1, idx)
        cls_scores_origin = torch.gather(cls_scores_origin, 1, idx)
    i = 0
    category = cls_ids[i][indices[i]] if squeeze_cls else cls_ids[i]
    scores = cls_scores[i]
    origin = cls_scores_origin[i]
    box = box_preds[i, indices[i] // num_cls].float()
    ids = instance_id[i, indices[i]] if instance_id is not None else None
    if mask is not None:
        category, scores, origin, box = category[mask[i]], scores[mask[i]], origin[mask[i]], box[mask[i]]
        ids = ids[mask[i]] if ids is not None else None
    return {"boxes": decode_box(box).cpu(), "scores": scores.cpu(), "cls_scores": origin.cpu(),
            "labels": category.cpu(), "ids": ids.cpu() if ids is not None else None}


def camera_centre_offset(calibration_json: Path) -> np.ndarray:
    """Mean camera centre ``C = mean_i(-R_i^T t_i)`` with z zeroed (SparseQID/EVA re-centring).

    Parameters
    ----------
    calibration_json : Path
        Scene ``calibration.json``; its ``sensors`` of ``type == "camera"`` are used, each with a world-to-camera
        ``extrinsicMatrix`` ``[R | t]``.

    Returns
    -------
    np.ndarray
        ``[Cx, Cy, 0]`` in metres; recentred = world - C, world = recentred + C.
    """
    calib = json.loads(calibration_json.read_text())
    centres = []
    for s in calib["sensors"]:
        if s.get("type") != "camera":
            continue
        e = np.asarray(s["extrinsicMatrix"], dtype=np.float64)
        centres.append(-e[:3, :3].T @ e[:3, 3])
    c = np.mean(centres, axis=0)
    return np.array([c[0], c[1], 0.0])


def output_paths(out_dir: Path, tag: str, scene: str) -> tuple[Path, Path, Path]:
    """The three output files of one run, refusing any that already exists.

    Parameters
    ----------
    out_dir : Path
        ``--out-dir``.
    tag : str
        Checkpoint tag (sub-folder).
    scene : str
        Scene folder name.

    Returns
    -------
    tuple[Path, Path, Path]
        ``<out_dir>/<tag>/<scene>.txt``, ``.jsonl`` and ``.meta.json``.

    Raises
    ------
    FileExistsError
        If any of them exists.
    """
    folder = out_dir / tag
    paths = tuple(folder / f"{scene}{s}" for s in (".txt", ".jsonl", ".meta.json"))
    for p in paths:
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
    return paths  # type: ignore[return-value]


def frame_rows(boxes: ArrayLike, scores: Sequence[float], cls_scores: Sequence[float], labels: Sequence[int],
               ids: Sequence[int], offset: ArrayLike, *, scene: str, scene_id: int, frame: int,
               score_thr: float, jsonl_thr: float) -> tuple[list[str], list[str]]:
    """Format one frame's decoded boxes as 11-column rows and JSONL lines, in the dataset's world frame.

    Parameters
    ----------
    boxes : array_like, shape (N, 10)
        Decoded boxes ``x, y, z, w, l, h, yaw, vx, vy, vz`` in the network's (re-centred) frame.
    scores : sequence of float
        ``scores_3d`` per box (class x sigmoid(centerness)).
    cls_scores : sequence of float
        Raw class score per box.
    labels : sequence of int
        Model class index per box (remapped with :data:`MODEL_IDX_TO_CLASS_ID`).
    ids : sequence of int
        Native instance-bank id per box (the 11-column object id is this + 1).
    offset : array_like, shape (3,)
        Re-centring offset added back to every centre (:func:`camera_centre_offset`; zeros without
        re-centring). Sizes, yaw and velocity are translation-invariant and left as they are.
    scene : str
        Scene folder name (JSONL ``scene``).
    scene_id : int
        Integer scene id (first 11-column field).
    frame : int
        Frame index.
    score_thr : float
        Lowest score written as an 11-column row.
    jsonl_thr : float
        Lowest score written as a JSONL line.

    Returns
    -------
    tuple[list[str], list[str]]
        The 11-column rows and the JSONL lines, each ending in a newline, in the decoder's order.
    """
    boxes = np.asarray(boxes).astype(np.float64)
    boxes[:, :3] += np.asarray(offset)
    rows_txt: list[str] = []
    rows_jsonl: list[str] = []
    for b, s, cs, lab, oid in zip(boxes, scores, cls_scores, labels, ids):
        cid = MODEL_IDX_TO_CLASS_ID[int(lab)]
        x, y, z, w, l, h, yaw = b[:7]
        if s >= score_thr:
            rows_txt.append(f"{scene_id} {cid} {int(oid) + 1} {frame} {x:.4f} {y:.4f} {z:.4f} "
                            f"{w:.4f} {l:.4f} {h:.4f} {yaw:.4f}\n")
        if s >= jsonl_thr:
            rows_jsonl.append(json.dumps({
                "scene": scene, "frame": int(frame), "class_id": cid, "class_name": CLASS_ID_TO_NAME[cid],
                "model_class_index": int(lab), "score": round(float(s), 5), "cls_score": round(float(cs), 5),
                "track_id": int(oid), "center": [round(float(v), 4) for v in (x, y, z)],
                "size_wlh": [round(float(v), 4) for v in (w, l, h)], "yaw": round(float(yaw), 5),
                "velocity": [round(float(v), 4) for v in b[7:10]],
            }) + "\n")
    return rows_txt, rows_jsonl


def frame_ids_sha256(frame_ids: Sequence[int]) -> str:
    """SHA-256 of the comma-joined frame indices (the meta's ``frames.ids_sha256``).

    Parameters
    ----------
    frame_ids : sequence of int
        Processed frame indices, in order.

    Returns
    -------
    str
        Hex digest of ``",".join(map(str, frame_ids))``.
    """
    return hashlib.sha256(",".join(map(str, frame_ids)).encode()).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments without the program name; ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The options below (defaults are the protocol's).
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sparseqid-src", type=Path, required=True, help="SparseQID checkout's src/ (commit 7aea61d)")
    ap.add_argument("--model-dir", type=Path, required=True,
                    help="NGC model folder (experiment.yaml, one anchor .npy, the .pth)")
    ap.add_argument("--ckpt", required=True, help="checkpoint file name inside --model-dir")
    ap.add_argument("--tag", required=True, help="output sub-folder name, e.g. rn50_trainable_v2.2")
    ap.add_argument("--data-root", type=Path, required=True, help="MTMC_Tracking_2026 folder")
    ap.add_argument("--cache-root", type=Path, required=True, help="frame cache (extract_smartspaces_frames.py)")
    ap.add_argument("--split", required=True, help="train or val")
    ap.add_argument("--scene", required=True, help="scene folder name, e.g. Warehouse_000")
    ap.add_argument("--scene-id", type=int, required=True, help="integer scene id written in the 11-column rows")
    ap.add_argument("--out-dir", type=Path, required=True, help="outputs go to <out-dir>/<tag>/")
    ap.add_argument("--score-thr", type=float, default=0.2, help="lowest score in the 11-column rows (0.2)")
    ap.add_argument("--dec-thr", type=float, default=0.0, help="decoder / instance-id threshold (0.0)")
    ap.add_argument("--jsonl-thr", type=float, default=0.05, help="lowest score in the JSONL (0.05)")
    ap.add_argument("--no-recenter", action="store_true", help="absolute-frame ablation")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (4)")
    ap.add_argument("--device", default="cuda:0", help="torch device (cuda:0)")
    ap.add_argument("--max-frames", type=int, default=None, help="process only the first N cached frames (checks)")
    ap.add_argument("--backbone-if-missing", default=None,
                    help="backbone type when the yaml has none (resnet_101 for rn101 v1.1)")
    ap.add_argument("--vram-cap-gib", type=float, default=24.0, help="hard VRAM cap (24 GiB)")
    ap.add_argument("--network-channels", choices=("rgb", "bgr"), default="rgb",
                    help="channel order of the tensor fed to the network: rgb = EVA to_rgb=0 (TAO PIL load, no "
                         "swap; EVA's best cell), bgr = SparseQID default = EVA to_rgb=1")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the detector over the cached frame window, and write outputs.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments without the program name; ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status 0 (errors raise).
    """
    args = parse_args(argv)
    out_txt, out_jsonl, out_meta = output_paths(args.out_dir, args.tag, args.scene)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":  # hard VRAM cap: OOM rather than exceed --vram-cap-gib
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.vram_cap_gib * 2**30 / total), device)
    cfg, notes = load_ngc_config(args.model_dir, args.dec_thr, args.backbone_if_missing)
    ckpt_path = args.model_dir / args.ckpt
    detector, load_report = load_detector(args.sparseqid_src, cfg, ckpt_path, device)
    from sparseqid.data.frame_dataset import FrameJpgDataset, batch_to_device, collate_scene  # noqa: E402

    recenter = not args.no_recenter
    dataset = FrameJpgDataset(args.cache_root, args.data_root, args.split, [args.scene],
                              recenter=recenter, recenter_mode="camera", final_dim=FINAL_DIM,
                              max_frames=args.max_frames)
    calib_path = args.data_root / args.split / args.scene / "calibration.json"
    offset = camera_centre_offset(calib_path) if recenter else np.zeros(3)
    scene_state = dataset._scene(args.scene)  # cached before the DataLoader forks its workers
    # SparseQID loads cv2 BGR -> converts to RGB -> NormalizeMultiviewImage(to_rgb=True) swaps back to BGR.
    # TAO (EVA) loads PIL RGB and swaps only if to_rgb; EVA ran to_rgb=0, i.e. RGB into the network.
    scene_state["norm"].to_rgb = args.network_channels == "bgr"
    num_cams = len(scene_state["cam_ids"])
    frame_ids = [fi for _, fi in dataset.samples]
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                         collate_fn=collate_scene)
    num_output = int(cfg.model.head.num_output)
    print(f"[{args.tag}] {args.scene}: {len(frame_ids)} frames [{frame_ids[0]}..{frame_ids[-1]}], {num_cams} cams, "
          f"classes={list(cfg.dataset.classes)}, offset={offset.round(4).tolist()}, recenter={recenter}, "
          f"amp={args.amp}, network_channels={args.network_channels}", flush=True)
    for n in notes:
        print("  config fill:", n, flush=True)
    print("  load:", load_report, flush=True)

    detector.head.instance_bank.reset()
    is_cuda = device.type == "cuda"
    sync = (lambda: torch.cuda.synchronize(device)) if is_cuda else (lambda: None)
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    rows_txt: list[str] = []
    n_jsonl = 0
    model_time = 0.0
    t_start = time.time()
    t_first_done = None
    with open(out_jsonl, "x") as fj:
        for index, batch in enumerate(loader):
            fi = frame_ids[index]
            batch = batch_to_device(batch, device)
            sync()
            t0 = time.time()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                feats = detector.extract_feat(batch["img"], batch)
                out = detector.head(feats, batch)
            dec = tao_decode(out["classification"][-1], out["prediction"][-1],
                             out["quality"][-1] if out.get("quality") else None, out.get("instance_id"),
                             num_output, args.dec_thr)
            sync()
            dt = time.time() - t0
            if index > 0:
                model_time += dt
            else:
                t_first_done = time.time()
            txt, jsonl = frame_rows(dec["boxes"].numpy(), dec["scores"].tolist(), dec["cls_scores"].tolist(),
                                    dec["labels"].tolist(), dec["ids"].tolist(), offset, scene=args.scene,
                                    scene_id=args.scene_id, frame=fi, score_thr=args.score_thr,
                                    jsonl_thr=args.jsonl_thr)
            rows_txt.extend(txt)
            fj.writelines(jsonl)
            n_jsonl += len(jsonl)
            if index % 50 == 0:
                print(f"  frame {fi} ({index + 1}/{len(frame_ids)}): {int((dec['scores'] >= args.score_thr).sum())} "
                      f"boxes >= {args.score_thr}, top score "
                      f"{float(dec['scores'][:1].max()) if len(dec['scores']) else 0:.3f}", flush=True)
    t_end = time.time()
    with open(out_txt, "x") as ft:
        ft.writelines(rows_txt)
    n_frames = len(frame_ids)
    timed = n_frames - 1
    meta = {
        "tag": args.tag, "scene": args.scene, "split": args.split, "scene_id": args.scene_id,
        "frames": {"count": n_frames, "first": frame_ids[0], "last": frame_ids[-1],
                   "ids_sha256": frame_ids_sha256(frame_ids)},
        "num_cameras": num_cams, "checkpoint": str(ckpt_path), "checkpoint_sha256": sha256_file(ckpt_path),
        "config": str(args.model_dir / "experiment.yaml"), "config_fills": notes, "load_report": load_report,
        "classes_model_order": list(cfg.dataset.classes), "model_idx_to_class_id": MODEL_IDX_TO_CLASS_ID,
        "recenter": recenter, "recenter_offset_xyz_m": offset.tolist(),
        "coordinate_frame": ("dataset world frame of calibration.json / ground_truth.json, metres; the network ran in "
                             "a frame translated by -offset (xy only) and offset was added back to every centre; "
                             "z up; yaw = rotation about +z in radians (atan2(sin, cos) of the regressed yaw); "
                             "size_wlh = (w, l, h) as decoded by TAO decode_box (exp of regressed log sizes), the "
                             "same field order as ground_truth.json '3d bounding box scale' and the 11-col format"),
        "input": f"960x540 JPEG q92 (cv2.resize of the 1920x1080 frame), {args.network_channels.upper()} channel order "
                 "into the network, ImageNet mean/std [123.675,116.28,103.53]/[58.395,57.12,57.375] applied in that "
                 "order",
        "network_channels": args.network_channels,
        "score": "scores_3d = sigmoid(class logit max over classes) * sigmoid(centerness), TAO SparseBox3DDecoder",
        "score_thr_txt": args.score_thr, "dec_thr": args.dec_thr, "jsonl_thr": args.jsonl_thr,
        "num_output_topk": num_output, "amp_bf16": args.amp,
        "rows_txt": len(rows_txt), "rows_jsonl": n_jsonl,
        "timing": {"wall_s_total": t_end - t_start, "wall_s_after_first_frame": t_end - (t_first_done or t_end),
                   "model_s_excl_first_frame": model_time, "timed_frames": timed,
                   "model_ms_per_frame": 1000 * model_time / max(timed, 1),
                   "model_camera_frames_per_s": num_cams * timed / max(model_time, 1e-9),
                   "end_to_end_camera_frames_per_s": num_cams * timed / max(t_end - (t_first_done or t_end), 1e-9)},
        "peak_vram": {"max_allocated_GiB": torch.cuda.max_memory_allocated(device) / 2**30 if is_cuda else 0.0,
                      "max_reserved_GiB": torch.cuda.max_memory_reserved(device) / 2**30 if is_cuda else 0.0},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device) if is_cuda else "cpu",
    }
    with open(out_meta, "x") as fm:
        json.dump(meta, fm, indent=1)
    print(f"[{args.tag}] wrote {len(rows_txt)} rows (score >= {args.score_thr}) -> {out_txt}; {n_jsonl} jsonl rows; "
          f"model {meta['timing']['model_ms_per_frame']:.1f} ms/frame = "
          f"{meta['timing']['model_camera_frames_per_s']:.1f} cam-frames/s; end-to-end "
          f"{meta['timing']['end_to_end_camera_frames_per_s']:.1f} cam-frames/s; peak alloc "
          f"{meta['peak_vram']['max_allocated_GiB']:.2f} GiB, reserved {meta['peak_vram']['max_reserved_GiB']:.2f} GiB",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
