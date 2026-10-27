"""Build an automatic label source from CuTR 3D boxes, from stored label shards (no SAM 3, no depth models).

Two variants (:mod:`measured_space.box_sources`):

- ``cutr_sam3``: CuTR's world boxes attached to SAM 3's detections by 2D IoU (CPU only);
- ``cutr_clip``: CuTR's boxes as the detections, named zero-shot by CLIP (:mod:`measured_space.clip_naming`).
  This variant loads the CLIP model, by default on ``--clip-device cuda``, and reads every view's image.

For every shard of ``--labels-from`` (a folder written by ``scripts/run_labels.py``), the view's CuTR predictions
(``<predictions>/<dataset>/<scene>.jsonl``) become the view's only automatic source. The view is then
re-associated and answered with :func:`measured_space.pipeline.rematch_view` (``"2d"``), which recomputes the
oracle answers as a check and keeps the stored ones verbatim. The new shards go to
``<out-dir>/<dataset>/shards/`` and are assembled with ``run_labels.assemble``, in the layout ``run_labels.py``
writes: ``views.jsonl``, ``oracle_objects.jsonl``, ``detections.jsonl``, ``qa_oracle.jsonl``,
``qa_auto_<variant>.jsonl`` and ``summary.json``. ``scripts/run_audit.py --labels <out-dir>`` and the VLM scoring
can read them. Each shard keeps the source shard's timestamps and adds a ``rematch`` record and a ``box_source``
record (prediction provenance, attachment or naming counts, CLIP probabilities).

Safety and provenance:

- the output folder must not exist or must be empty, and must lie outside both input folders; nothing is
  written otherwise;
- every view needs a prediction line. Without one the run stops before writing anything, unless
  ``--skip-missing`` is given (skipped views are counted in the log);
- ``<out-dir>/runs.jsonl`` records the arguments, each dataset's source shards (count and a digest over their
  names and SHA-256), the SHA-256 of every predictions file, the CLIP model id and snapshot, the naming
  vocabulary, the counts, and the oracle check. ``<out-dir>/inputs.json`` lists every source shard with its
  SHA-256;
- when every view of a dataset was processed, its ``qa_oracle.jsonl`` must be byte-identical to the source's
  (SHA-256 compared), else the exit status is 1. Each shard's oracle items are also compared with the source
  shard's.

Run from the repository root::

    python scripts/run_box_source.py --variant cutr_sam3 --labels-from <data>/outputs/labels_2d \\
        --predictions <data>/outputs/cubify --out-dir <data>/outputs/labels_cutr_sam3
    python scripts/run_box_source.py --variant cutr_clip --labels-from <data>/outputs/labels_2d \\
        --predictions <data>/outputs/cubify --out-dir <data>/outputs/labels_cutr_clip --clip-device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from measured_space.box_sources import (
    CLIP_SOURCE,
    SAM3_SOURCE,
    VARIANTS,
    ViewPredictions,
    clip_box2d,
    predictions_file,
    read_box_predictions,
    run_box_source,
)
from measured_space.clip_naming import CLIP_MODEL_ID, LogitsFn, NamingConfig, name_boxes, vocabulary
from measured_space.matching import Match2DConfig
from measured_space.pipeline import to_json_line, view_from_record
from measured_space.scene import FrameView
from run_labels import assemble, sha256, write_shard   # scripts/ is on sys.path when run

ImageLoader = Callable[[FrameView], NDArray[np.uint8]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        The parsed options.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variant", choices=list(VARIANTS), required=True, help="which CuTR source to build")
    p.add_argument("--labels-from", type=Path, required=True,
                   help="label folder of run_labels.py whose <dataset>/shards/*.json are read (never written)")
    p.add_argument("--predictions", type=Path, required=True,
                   help="CuTR predictions folder holding <dataset>/<scene>.jsonl")
    p.add_argument("--out-dir", type=Path, required=True, help="new, empty output folder")
    p.add_argument("--datasets", nargs="+", choices=["arkitscenes", "smartspaces"],
                   default=["arkitscenes", "smartspaces"], help="datasets to process (missing ones are skipped)")
    p.add_argument("--scenes", nargs="*", default=None,
                   help="only process scenes whose name contains one of these strings")
    p.add_argument("--skip-missing", action="store_true",
                   help="skip views without a prediction line instead of stopping")
    p.add_argument("--attach-min-iou", type=float, default=Match2DConfig.min_iou,
                   help="cutr_sam3: smallest 2D IoU of a SAM 3 detection and its CuTR box")
    p.add_argument("--clip-model", default=CLIP_MODEL_ID, help="cutr_clip: Hugging Face id of the CLIP model")
    p.add_argument("--clip-revision", default=None,
                   help="cutr_clip: snapshot to load (default: the pinned one for the default model)")
    p.add_argument("--clip-device", default="cuda", help="cutr_clip: torch device for CLIP")
    p.add_argument("--clip-dtype", default=None, help="cutr_clip: torch dtype (default float16 on CUDA, else float32)")
    p.add_argument("--clip-batch-size", type=int, default=32, help="cutr_clip: crops per forward pass")
    p.add_argument("--min-name-prob", type=float, default=None,
                   help="cutr_clip: withhold names below this CLIP probability (default: off)")
    p.add_argument("--hf-cache", type=Path, default=None,
                   help="Hugging Face hub cache (default <data root>/hf_cache)")
    return p.parse_args(argv)


def refuse_existing(out_dir: Path, *inputs: Path) -> None:
    """Refuse an output folder that holds anything or overlaps an input folder.

    Parameters
    ----------
    out_dir : pathlib.Path
        The output folder.
    *inputs : pathlib.Path
        Input folders, which must neither contain nor lie inside ``out_dir``.

    Raises
    ------
    FileExistsError
        If ``out_dir`` exists and is not an empty folder.
    ValueError
        If ``out_dir`` is, contains or lies inside an input folder.
    """
    out_dir = Path(out_dir).resolve()
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"{out_dir} exists and is not an empty folder; choose a new --out-dir")
    for folder in (Path(f).resolve() for f in inputs):
        if out_dir == folder or folder in out_dir.parents or out_dir in folder.parents:
            raise ValueError(f"--out-dir {out_dir} overlaps the input folder {folder}")


def _wanted(scene: str, filters: Sequence[str] | None) -> bool:
    """Whether a scene passes the ``--scenes`` filter.

    Parameters
    ----------
    scene : str
        Scene name.
    filters : sequence of str or None
        Substrings; ``None`` or empty accepts every scene.

    Returns
    -------
    bool
        True if no filter is set or the scene contains one of the substrings.
    """
    return not filters or any(f in scene for f in filters)


def select_shards(labels_from: Path, dataset: str, scenes: Sequence[str] | None) -> list[tuple[Path, dict[str, Any]]]:
    """Read a dataset's source shards, keeping the selected scenes.

    Parameters
    ----------
    labels_from : pathlib.Path
        The source label folder.
    dataset : str
        Dataset folder name.
    scenes : sequence of str or None
        ``--scenes`` filter.

    Returns
    -------
    list of tuple
        ``(path, shard)`` in file-name order; empty if the dataset has no shard folder.
    """
    folder = labels_from / dataset / "shards"
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.glob("*.json")):
        shard = json.loads(path.read_text())
        if _wanted(shard["records"]["view"]["scene"], scenes):
            out.append((path, shard))
    return out


def load_predictions(root: Path, dataset: str, scenes: Sequence[str]) -> tuple[dict[str, ViewPredictions],
                                                                               dict[str, dict[str, str]]]:
    """Read the predictions of a dataset's scenes.

    Parameters
    ----------
    root : pathlib.Path
        The predictions folder.
    dataset : str
        Dataset name.
    scenes : sequence of str
        Scenes to read.

    Returns
    -------
    predictions : dict of str to ViewPredictions
        View key -> predictions, over all scenes found.
    files : dict of str to dict
        Scene -> ``{"path": file relative to root, "sha256": ...}`` for every file read.

    Raises
    ------
    ValueError
        If a view appears in two files, or as :func:`measured_space.box_sources.read_box_predictions`.
    """
    predictions: dict[str, ViewPredictions] = {}
    files: dict[str, dict[str, str]] = {}
    for scene in scenes:
        path = predictions_file(root, dataset, scene)
        if path is None:
            continue
        for key, view in read_box_predictions(path).items():
            if key in predictions:
                raise ValueError(f"view {key} has predictions in two files")
            predictions[key] = view
        files[scene] = {"path": str(path.relative_to(root)), "sha256": str(sha256(path))}
    return predictions, files


def shard_digest(entries: Sequence[dict[str, str]]) -> str:
    """One checksum for a list of shards.

    Parameters
    ----------
    entries : sequence of dict
        ``{"name", "sha256"}`` per shard.

    Returns
    -------
    str
        SHA-256 of the lines ``"<name> <sha256>"``, sorted by name.
    """
    lines = "".join(f"{e['name']} {e['sha256']}\n" for e in sorted(entries, key=lambda e: e["name"]))
    return hashlib.sha256(lines.encode()).hexdigest()


def load_scorer(args: argparse.Namespace) -> tuple[LogitsFn, dict[str, Any]]:
    """Load CLIP for the ``cutr_clip`` variant (weights from the local cache only).

    Parameters
    ----------
    args : argparse.Namespace
        ``clip_model``, ``clip_revision``, ``clip_device``, ``clip_dtype``, ``clip_batch_size``, ``hf_cache``.

    Returns
    -------
    logits_fn : callable
        The loaded :class:`measured_space.clip_naming.ClipScorer`.
    info : dict
        Its :attr:`~measured_space.clip_naming.ClipScorer.info` (model id, snapshot, dtype, ...).
    """
    from measured_space.clip_naming import ClipScorer
    from measured_space.eval_vlm import hf_cache_dir

    cache = args.hf_cache or hf_cache_dir()
    os.environ.setdefault("HF_HUB_CACHE", str(cache))
    scorer = ClipScorer.load(args.clip_model, revision=args.clip_revision, device=args.clip_device,
                             cache_dir=cache, batch_size=args.clip_batch_size, dtype=args.clip_dtype)
    print(f"loaded {scorer.info['model_id']} @ {scorer.info['snapshot']} on {scorer.info['device']} in "
          f"{scorer.info['load_seconds']} s", flush=True)
    return scorer, scorer.info


def oracle_check(labels_from: Path, out_dir: Path, datasets: Sequence[str], complete: dict[str, bool]
                 ) -> dict[str, dict[str, Any]]:
    """Compare each processed dataset's ``qa_oracle.jsonl`` with the source's, when the whole dataset was done.

    Parameters
    ----------
    labels_from : pathlib.Path
        The source label folder.
    out_dir : pathlib.Path
        The output folder.
    datasets : sequence of str
        Datasets to compare (those without an output file are skipped).
    complete : dict of str to bool
        Whether every source shard of a dataset was processed.

    Returns
    -------
    dict of str to dict
        Dataset -> ``{"compared", "source_sha256", "sha256", "identical"}``; a subset is not compared
        (``compared`` False, ``identical`` None).
    """
    out: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        new = out_dir / dataset / "qa_oracle.jsonl"
        if not new.exists():
            continue
        if not complete.get(dataset, False):
            out[dataset] = {"compared": False, "source_sha256": None, "sha256": sha256(new), "identical": None}
            continue
        a, b = sha256(labels_from / dataset / "qa_oracle.jsonl"), sha256(new)
        out[dataset] = {"compared": True, "source_sha256": a, "sha256": b, "identical": a is not None and a == b}
    return out


def main(argv: Sequence[str] | None = None, *, logits_fn: LogitsFn | None = None,
         load_image: ImageLoader | None = None) -> int:
    """Build the chosen CuTR source for every selected view, assemble the outputs and log the run.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``None`` reads ``sys.argv``.
    logits_fn : callable, optional
        ``cutr_clip``: a model to use instead of loading CLIP (for tests).
    load_image : callable, optional
        ``cutr_clip``: ``load_image(view) -> RGB``; defaults to :func:`measured_space.datasets.load_rgb`.

    Returns
    -------
    int
        0 on success; 1 if the output folder is refused, a view lacks predictions (without ``--skip-missing``),
        nothing is selected, or a dataset's oracle file or a shard's oracle items differ from the source's.
    """
    args = parse_args(argv)
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    labels_from, preds_root = args.labels_from.resolve(), args.predictions.resolve()
    out_dir = args.out_dir.resolve()
    try:
        refuse_existing(out_dir, labels_from, preds_root)
    except (FileExistsError, ValueError) as error:
        print(f"refused: {error}", flush=True)
        return 1

    # Inputs: shards, predictions, coverage. Nothing is written before these checks pass.
    jobs: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    predictions: dict[str, ViewPredictions] = {}
    shard_list: dict[str, list[dict[str, str]]] = {}
    prediction_files: dict[str, dict[str, dict[str, str]]] = {}
    missing: list[str] = []
    for dataset in args.datasets:
        shards = select_shards(labels_from, dataset, args.scenes)
        if not shards:
            continue
        scenes = sorted({s["records"]["view"]["scene"] for _, s in shards})
        found, files = load_predictions(preds_root, dataset, scenes)
        predictions.update(found)
        jobs[dataset] = shards
        shard_list[dataset] = [{"name": p.name, "sha256": str(sha256(p))} for p, _ in shards]
        prediction_files[dataset] = files
        missing += [s["records"]["view"]["view_key"] for _, s in shards
                    if s["records"]["view"]["view_key"] not in found]
    if not jobs:
        print(f"no shards selected under {labels_from} for {args.datasets} (scenes {args.scenes})", flush=True)
        return 1
    if missing and not args.skip_missing:
        print(f"{len(missing)} views have no predictions (first: {missing[:3]}); pass --skip-missing to skip them",
              flush=True)
        return 1

    naming = NamingConfig(min_prob=args.min_name_prob)
    clip_info: dict[str, Any] | None = None
    if args.variant == CLIP_SOURCE:
        if logits_fn is None:
            logits_fn, clip_info = load_scorer(args)
        else:
            clip_info = {"model_id": "injected", "snapshot": None}
        if load_image is None:
            from measured_space.datasets import load_rgb as load_image

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    oracle_mismatch: list[str] = []
    match_2d = Match2DConfig(min_iou=args.attach_min_iou)
    for dataset, shards in jobs.items():
        t_dataset = time.perf_counter()
        files = prediction_files[dataset]
        for path, shard in shards:
            records = shard["records"]
            key = records["view"]["view_key"]
            view_predictions = predictions.get(key)
            if view_predictions is None:
                counts["skipped_no_predictions"] += 1
                continue
            names, naming_seconds = None, None
            if args.variant == CLIP_SOURCE:
                view = view_from_record(records["view"])
                image = load_image(view)                                      # type: ignore[misc]
                if image.shape != (view.camera.height, view.camera.width, 3):
                    raise ValueError(f"{key}: image {image.shape} does not match the view's camera")
                t = time.perf_counter()
                width, height = view_predictions.image_size
                names = name_boxes(image, [clip_box2d(b.box2d, width, height) for b in view_predictions.boxes],
                                   dataset, logits_fn, naming)                # type: ignore[arg-type]
                naming_seconds = time.perf_counter() - t
            t = time.perf_counter()
            result, info = run_box_source(records, view_predictions, args.variant, names=names, config=match_2d,
                                          naming_seconds=naming_seconds)
            output = result.records()
            seconds = time.perf_counter() - t
            if json.loads(to_json_line(output["qa"]["oracle"])) != records["qa"].get("oracle", []):
                oracle_mismatch.append(key)
            scene_file = files.get(records["view"]["scene"], {})
            write_shard(out_dir / dataset / "shards" / path.name, {
                "records": output, "wall_seconds": shard.get("wall_seconds"), "started": shard.get("started"),
                "finished": shard.get("finished"),
                "rematch": {"from": str(path), "association": "2d",
                            "match_2d": {"min_iou": result.config.match_2d.min_iou}, "seconds": seconds,
                            "time": datetime.now(timezone.utc).isoformat()},
                "box_source": {**info, "predictions_file": scene_file.get("path"),
                               "predictions_sha256": scene_file.get("sha256"), "clip": clip_info}})
            counts["processed"] += 1
            counts[f"processed_{dataset}"] += 1
        print(f"{dataset}: {args.variant} for {counts[f'processed_{dataset}']} views in "
              f"{time.perf_counter() - t_dataset:.1f} s", flush=True)

    t = time.perf_counter()
    assemble(out_dir, list(jobs), None)
    assemble_seconds = time.perf_counter() - t
    complete = {d: not args.scenes and not any(k.startswith(f"{d}/") for k in missing) for d in jobs}
    oracle = oracle_check(labels_from, out_dir, list(jobs), complete)
    inputs = {"labels_from": str(labels_from), "predictions": str(preds_root), "shards": shard_list,
              "prediction_files": prediction_files}
    (out_dir / "inputs.json").write_text(json.dumps(inputs, indent=1))
    run: dict[str, Any] = {
        "argv": list(sys.argv[1:] if argv is None else argv), "variant": args.variant, "started": started,
        "labels_from": str(labels_from), "predictions": str(preds_root),
        "source_shards": {d: {"n": len(v), "digest": shard_digest(v), "list": "inputs.json"}
                          for d, v in shard_list.items()},
        "prediction_files": {d: {f["path"]: f["sha256"] for f in v.values()} for d, v in prediction_files.items()},
        "attach_min_iou": args.attach_min_iou if args.variant == SAM3_SOURCE else None,
        "clip": None if clip_info is None else {
            **clip_info, "naming": naming.to_json(),
            "vocabulary": {d: [list(part) for part in vocabulary(d, naming.other_phrases)] for d in jobs}},
        "n_missing_predictions": len(missing), "counts": dict(counts), "oracle_identical": oracle,
        "n_shard_oracle_mismatch": len(oracle_mismatch),
        "processing_seconds": round(time.perf_counter() - t0 - assemble_seconds, 1),
        "assemble_seconds": round(assemble_seconds, 1), "finished": datetime.now(timezone.utc).isoformat()}
    with (out_dir / "runs.jsonl").open("a") as f:
        f.write(json.dumps(run) + "\n")
    print(json.dumps({k: run[k] for k in ("variant", "counts", "oracle_identical", "n_shard_oracle_mismatch")}),
          flush=True)
    failed = bool(oracle_mismatch) or any(v["identical"] is False for v in oracle.values())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
