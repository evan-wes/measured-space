"""Score open VLMs against oracle and automatic spatial labels, on a stratified sample of generated questions.

Three steps, run in order from the repository root:

``sample`` (CPU)
    A draw takes the candidates of one label folder: questions with both an oracle answer and an answer from
    the deployment-realistic automatic source (ARKitScenes: iPad LiDAR depth; Smart Spaces: monocular MoGe
    depth). It stratifies them by dataset and question type, and keeps the ``--per-cell`` (default 25)
    questions with the lowest stable key ``sha256("<seed>:<id>")`` in each cell. The order of the label files
    does not matter. A question added to the folder later joins the draw only if its key is among the lowest
    in its cell. ``--extend`` adds a new draw (another label folder, more scenes, a larger ``--per-cell``) to
    the sample. The sample is the union of all draws and only ever grows, so answered questions are reused
    and only new ones are asked. Every drawn image's view is rebuilt (:mod:`measured_space.vlm_images`) and
    checked against the label pipeline's ``views.jsonl``. The specs and draws go to ``<out>/sample.json``.
``predict`` (GPU, one model per process)
    A model answers every sampled question once in each requested mode (``--modes``): with the image, and
    blind (text only, the no-vision baseline). It uses the default resolution policy and token budget of
    :mod:`measured_space.eval_vlm`. Questions are grouped by
    image, and images are visited in hashed order, so a ``--limit`` run is a spread-out timing sample.
    Predictions are appended, resume-safe, to ``<out>/predictions/<model>[_blind].jsonl``. Each call
    appends its timing to ``<out>/runs.jsonl``. Predictions do not depend on any automatic label set.
``score`` (CPU)
    The questions of the latest draw from one label folder (``--labels-dir``, or ``--draw-from``) are scored
    against the oracle and against every automatic source in that folder, with
    :mod:`measured_space.scoring`. Results are per source (``<dataset>/<depth>``) and for the pooled
    deployment source:

    - per-type accuracy;
    - the 2x2 (all vs gate-accepted items, oracle vs auto labels);
    - delta-delta between models, with a scene-cluster bootstrap CI;
    - labeller-imitation rate;
    - parse-failure and forced-answer rates.

    Blind runs appear as models named ``<model>+blind``. Every table is computed twice:

    - with answers forced after cut-off reasoning counted as answers (``forced_policy = "counted"``);
    - with those answers counted as failures (``"strict"``).

    The gate-accepted half needs ``--accepted``, an ``accepted_ids.json`` made from the same label folder.
    Without it, only the "all items" half is reported.

Outputs: per-item data (sampled specs, predictions, scored rows) stays under ``<out>``, by default
``<data root>/outputs/vlm``. The repository receives only aggregates, in ``results/vlm/<label folder name>``
(``summary.json`` and CSVs): one dataset's licence is unclear about redistributing per-item derived labels.

Examples::

    python scripts/run_vlm_eval.py sample
    python scripts/run_vlm_eval.py predict --model qwen3-vl-8b --limit 10      # timing
    python scripts/run_vlm_eval.py predict --model qwen3-vl-8b                 # resumes
    python scripts/run_vlm_eval.py score --accepted <data>/outputs/audit/accepted_ids.json
    python scripts/run_vlm_eval.py sample --extend --labels-dir <data>/outputs/labels_2d
    python scripts/run_vlm_eval.py predict --model qwen3-vl-8b                 # answers only new questions
    python scripts/run_vlm_eval.py score --labels-dir <data>/outputs/labels_2d \
        --accepted <data>/outputs/audit_2d/accepted_ids.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from measured_space import scoring
from measured_space.config import REPO_ROOT, data_root
from measured_space.eval_vlm import DEFAULT_RESOLUTION, MODELS, read_predictions, run_predictions
from measured_space.qa_schema import QAItem, QuestionSpec, Referent, read_jsonl
from measured_space.vlm_images import SpecImageLoader, view_key

DEPTH_SOURCES: dict[str, tuple[str, ...]] = {"arkitscenes": ("laser", "lidar", "moge"),
                                             "smartspaces": ("gt_depth", "moge")}
"""Automatic label sources per dataset, as in ``qa_auto_<depth>.jsonl``."""

DEPLOYMENT_SOURCE: dict[str, str] = {"arkitscenes": "lidar", "smartspaces": "moge"}
"""The automatic source a deployment would have, per dataset; the sample requires an answer from it."""

POOLED_UNIT = "pooled/deployment"
"""Name of the scoring unit that pools each dataset's deployment source."""

MODEL_PAIRS: tuple[tuple[str, str], ...] = (
    ("qwen3-vl-8b", "cosmos-reason2-8b"),
    ("qwen3-vl-8b", "qwen3-vl-8b" + scoring.BLIND_SUFFIX),
    ("cosmos-reason2-8b", "cosmos-reason2-8b" + scoring.BLIND_SUFFIX),
)
"""Model pairs for delta-delta: the two VLMs (primary), then each VLM against its own blind run."""

FORCED_POLICIES: dict[str, bool] = {"counted": True, "strict": False}
"""Forced-answer policy name to ``include_forced``."""


@dataclass(frozen=True)
class Unit:
    """A scoring unit: the automatic label source (or sources) that predictions are scored against.

    Attributes
    ----------
    name : str
        ``"<dataset>/<depth>"``, or :data:`POOLED_UNIT`.
    parts : tuple of (str, str)
        ``(dataset, depth source)`` pairs whose automatic labels are pooled.
    """

    name: str
    parts: tuple[tuple[str, str], ...]


# --------------------------------------------------------------------------------------------- paths and io


def default_out_dir() -> Path:
    """The folder for per-item outputs.

    Returns
    -------
    pathlib.Path
        ``<data root>/outputs/vlm``.

    Raises
    ------
    RuntimeError
        If the data root is not configured or missing.
    """
    return data_root() / "outputs" / "vlm"


def load_items(path: Path) -> dict[str, QAItem]:
    """Labelled items of one file, by question id.

    Parameters
    ----------
    path : pathlib.Path
        A ``qa_*.jsonl`` file.

    Returns
    -------
    dict
        Question id to item, in file order.

    Raises
    ------
    ValueError
        If an id appears twice.
    FileNotFoundError
        If the file is missing.
    """
    items: dict[str, QAItem] = {}
    for item in read_jsonl(path):
        if item.id in items:
            raise ValueError(f"{path}: duplicate question id {item.id}")
        items[item.id] = item
    return items


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Size and SHA-256 of a file, to tie outputs to the exact labels they were made from.

    Parameters
    ----------
    path : pathlib.Path
        Any file.

    Returns
    -------
    dict
        ``{"bytes": int, "sha256": str}``.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def git_commit() -> str | None:
    """The repository's current commit, for the record.

    Returns
    -------
    str or None
        ``git rev-parse HEAD``, or None if git is unavailable.
    """
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def portable(path: str | Path) -> str:
    """A path as written into the repository: relative to the data root or the repository, never absolute.

    Parameters
    ----------
    path : str or pathlib.Path
        Any path.

    Returns
    -------
    str
        ``"<data>/..."`` under the data root, ``"<repo>/..."`` under the repository, else
        ``"<external>/<name>"``.
    """
    path = Path(path).resolve()
    for tag, root in (("<data>", data_root().resolve()), ("<repo>", REPO_ROOT.resolve())):
        if path.is_relative_to(root):
            return f"{tag}/{path.relative_to(root)}"
    return f"<external>/{path.name}"


def utc_now() -> str:
    """The current UTC time.

    Returns
    -------
    str
        ISO 8601, to the second.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------- sampling


def paired_specs(oracle: Mapping[str, QAItem], auto: Mapping[str, QAItem]) -> list[QuestionSpec]:
    """Questions answered by both label sources.

    Parameters
    ----------
    oracle : mapping of str to QAItem
        Oracle items by id.
    auto : mapping of str to QAItem
        Automatic items of one depth source, by id.

    Returns
    -------
    list of QuestionSpec
        The oracle spec of every id present in both, in oracle order.

    Raises
    ------
    ValueError
        If the two sources hold different specs under one id.
    """
    specs = []
    for item_id, item in oracle.items():
        if item_id in auto:
            if auto[item_id].spec != item.spec:
                raise ValueError(f"Question {item_id}: oracle and automatic specs differ")
            specs.append(item.spec)
    return specs


def sample_rank(seed: int, item_id: str) -> str:
    """The sampling key of a question: within a stratum, lower keys are sampled first.

    Parameters
    ----------
    seed : int
        Sampling seed.
    item_id : str
        Question id.

    Returns
    -------
    str
        Hex SHA-256 of ``"<seed>:<id>"``.
    """
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def draw_sample(candidates: Iterable[QuestionSpec], per_cell: int, seed: int) -> dict[str, QuestionSpec]:
    """A stratified draw: the ``per_cell`` lowest sampling keys in every dataset x question-type cell.

    The key of a question (:func:`sample_rank`) depends only on the seed and its id, not on file order or on
    the other candidates. A cell smaller than ``per_cell`` is taken whole. A candidate added later enters
    the draw only if its key is among its cell's ``per_cell`` lowest.

    Parameters
    ----------
    candidates : iterable of QuestionSpec
        Eligible questions; repeated ids count once.
    per_cell : int
        Questions kept per cell.
    seed : int
        Sampling seed.

    Returns
    -------
    dict of str to QuestionSpec
        The drawn questions by id, in cell order and then key order.

    Raises
    ------
    ValueError
        If ``per_cell`` is below 1.
    """
    if per_cell < 1:
        raise ValueError(f"per_cell must be at least 1, got {per_cell}")
    cells: dict[tuple[str, str], dict[str, QuestionSpec]] = {}
    for spec in candidates:
        cells.setdefault((spec.dataset, spec.qtype), {})[spec.id] = spec
    drawn: dict[str, QuestionSpec] = {}
    for cell in sorted(cells):
        for item_id in sorted(cells[cell], key=lambda i: (sample_rank(seed, i), i))[:per_cell]:
            drawn[item_id] = cells[cell][item_id]
    return drawn


def stratified_sample(specs: Iterable[QuestionSpec], per_cell: int, seed: int) -> dict[tuple[str, str], list[str]]:
    """:func:`draw_sample`, as sorted ids per cell.

    Parameters
    ----------
    specs : iterable of QuestionSpec
        Candidate questions; repeated ids count once.
    per_cell : int
        Questions kept per cell (all of them if the cell is smaller).
    seed : int
        Sampling seed.

    Returns
    -------
    dict
        ``(dataset, qtype)`` to the sorted ids kept, cells in sorted order.

    Raises
    ------
    ValueError
        If ``per_cell`` is below 1.
    """
    return group_by_cell(draw_sample(specs, per_cell, seed))


def group_by_cell(specs: Mapping[str, QuestionSpec]) -> dict[tuple[str, str], list[str]]:
    """Question ids grouped by dataset and question type.

    Parameters
    ----------
    specs : mapping of str to QuestionSpec
        Questions by id.

    Returns
    -------
    dict
        ``(dataset, qtype)`` to sorted ids, cells in sorted order.
    """
    cells: dict[tuple[str, str], list[str]] = {}
    for item_id, spec in specs.items():
        cells.setdefault((spec.dataset, spec.qtype), []).append(item_id)
    return {cell: sorted(cells[cell]) for cell in sorted(cells)}


def spec_to_dict(spec: QuestionSpec) -> dict[str, Any]:
    """A question as a JSON-ready dict.

    Parameters
    ----------
    spec : QuestionSpec
        The question.

    Returns
    -------
    dict
        Its fields; referents as dicts.
    """
    return asdict(spec)


def spec_from_dict(record: Mapping[str, Any]) -> QuestionSpec:
    """Rebuild a question written by :func:`spec_to_dict`.

    Parameters
    ----------
    record : mapping
        The question's fields.

    Returns
    -------
    QuestionSpec
        The question.
    """
    fields = dict(record)
    fields["referents"] = tuple(Referent(**r) for r in fields["referents"])
    if fields.get("choices") is not None:
        fields["choices"] = tuple(fields["choices"])
    return QuestionSpec(**fields)


def check_views(specs: Iterable[QuestionSpec], views_path: Path, loader: SpecImageLoader) -> dict[str, Any]:
    """Check that the rebuilt view of every sampled image matches the label pipeline's record of it.

    Parameters
    ----------
    specs : iterable of QuestionSpec
        Sampled questions of one dataset.
    views_path : pathlib.Path
        The dataset's ``views.jsonl`` from the label pipeline.
    loader : SpecImageLoader
        Rebuilds the views.

    Returns
    -------
    dict
        ``n_views`` checked and ``max_abs_pose_diff`` (largest absolute difference between ``T_wc`` entries).

    Raises
    ------
    ValueError
        If a view is missing from ``views.jsonl``, or its image reference, camera or pose differ.
    """
    records = {}
    with views_path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[(r["dataset"], r["scene"], r["camera_id"], r["frame"])] = r
    seen: dict[tuple[str, str, str, str], QuestionSpec] = {}
    for spec in specs:
        seen.setdefault(view_key(spec), spec)
    worst = 0.0
    for key, spec in seen.items():
        if key not in records:
            raise ValueError(f"View {key} is not in {views_path}")
        record, view = records[key], loader.view(spec)
        camera = {k: getattr(view.camera, k) for k in record["camera"]}
        if record["image_ref"] != view.image_ref or not all(
                math.isclose(float(camera[k]), float(v), rel_tol=1e-9, abs_tol=1e-9)
                for k, v in record["camera"].items()):
            raise ValueError(f"View {key}: rebuilt image reference or camera differs from {views_path}")
        diff = float(np.max(np.abs(np.asarray(record["T_wc"]) - view.T_wc)))
        if diff > 1e-6:
            raise ValueError(f"View {key}: rebuilt pose differs from {views_path} by {diff}")
        worst = max(worst, diff)
    return {"n_views": len(seen), "max_abs_pose_diff": worst}


def sample_composition(specs: Mapping[str, QuestionSpec]) -> pd.DataFrame:
    """Questions per dataset and question type.

    Parameters
    ----------
    specs : mapping of str to QuestionSpec
        Questions by id (one draw, or the whole sample).

    Returns
    -------
    pandas.DataFrame
        Columns ``dataset, qtype, n_sampled, n_scenes``.
    """
    rows = []
    for (dataset, qtype), ids in group_by_cell(specs).items():
        rows.append({"dataset": dataset, "qtype": qtype, "n_sampled": len(ids),
                     "n_scenes": len({specs[i].scene for i in ids})})
    return pd.DataFrame(rows, columns=["dataset", "qtype", "n_sampled", "n_scenes"])


def cmd_sample(args: argparse.Namespace) -> int:
    """Draw from a label folder and add the draw to the sample (``sample.json``).

    Parameters
    ----------
    args : argparse.Namespace
        ``labels_dir``, ``out_dir``, ``per_cell``, ``seed``, ``datasets``, ``extend``, ``force``.

    Returns
    -------
    int
        Exit status: 0 on success, 1 if a dataset's labels are unfinished, the sample exists and neither
        ``--extend`` nor ``--force`` was given, or ``--extend`` uses another seed.

    Raises
    ------
    ValueError
        If a sampled question's spec differs from the one in the labels.
    """
    root = args.labels_dir
    out_path = args.out_dir / "sample.json"
    if out_path.exists() and not (args.extend or args.force):
        print(f"{out_path} exists; pass --extend to add a draw or --force to start over", flush=True)
        return 1
    if args.extend and out_path.exists():
        sample = load_sample(args.out_dir)
        if sample["seed"] != args.seed:
            print(f"--extend needs the sample's seed ({sample['seed']}), got {args.seed}", flush=True)
            return 1
    else:
        sample = {"created": utc_now(), "seed": args.seed, "deployment_source": dict(DEPLOYMENT_SOURCE),
                  "draws": [], "specs": {}}
    existing = sample_spec_map(sample)
    draw: dict[str, Any] = {"created": utc_now(), "labels_dir": str(root), "per_cell": args.per_cell,
                            "datasets": list(args.datasets), "git_commit": git_commit(), "label_files": {},
                            "view_check": {}, "cells": [], "ids": []}
    loader = SpecImageLoader()
    for dataset in args.datasets:
        folder = root / dataset
        if not (folder / "summary.json").exists():
            print(f"{folder}/summary.json missing: labels for {dataset} are not finished", flush=True)
            return 1
        source = DEPLOYMENT_SOURCE[dataset]
        oracle_path, auto_path = folder / "qa_oracle.jsonl", folder / f"qa_auto_{source}.jsonl"
        oracle, auto = load_items(oracle_path), load_items(auto_path)
        for item_id, spec in existing.items():
            if item_id in oracle and oracle[item_id].spec != spec:
                raise ValueError(f"Sampled question {item_id} differs from its spec in {oracle_path}")
        for path in (oracle_path, auto_path):
            draw["label_files"][f"{dataset}/{path.name}"] = file_fingerprint(path)
        candidates = paired_specs(oracle, auto)
        drawn = draw_sample(candidates, args.per_cell, args.seed)
        drawn_cells = group_by_cell(drawn)
        for (ds, qtype), ids in group_by_cell({s.id: s for s in candidates}).items():
            chosen = drawn_cells.get((ds, qtype), [])
            draw["cells"].append({"dataset": ds, "qtype": qtype,
                                  "n_oracle": sum(1 for i in oracle.values() if i.spec.qtype == qtype),
                                  "n_candidates": len(ids), "n_drawn": len(chosen),
                                  "n_new": sum(1 for i in chosen if i not in existing),
                                  "n_scenes": len({drawn[i].scene for i in chosen})})
        draw["ids"] += sorted(drawn)
        sample["specs"].update({i: spec_to_dict(s) for i, s in drawn.items() if i not in existing})
        existing.update(drawn)
        draw["view_check"][dataset] = check_views(list(drawn.values()), folder / "views.jsonl", loader)
    draw["n_drawn"] = len(draw["ids"])
    draw["n_new"] = sum(c["n_new"] for c in draw["cells"])
    sample["draws"].append(draw)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sample, indent=1))
    print(pd.DataFrame(draw["cells"]).to_string(index=False))
    print(f"drew {draw['n_drawn']} ({draw['n_new']} new); sample now {len(sample['specs'])} questions; views "
          f"checked: {draw['view_check']}; written {out_path}", flush=True)
    return 0


def latest_draw(sample: Mapping[str, Any], labels_dir: Path) -> dict[str, Any] | None:
    """The most recent draw made from a label folder.

    Parameters
    ----------
    sample : mapping
        The record from :func:`load_sample`.
    labels_dir : pathlib.Path
        A label folder.

    Returns
    -------
    dict or None
        The draw record (``ids``, ``per_cell``, ``cells``, ...), or None if no draw used this folder.
    """
    draws = [d for d in sample["draws"] if Path(d["labels_dir"]) == Path(labels_dir)]
    return draws[-1] if draws else None


def load_sample(out_dir: Path) -> dict[str, Any]:
    """Read ``sample.json``.

    Parameters
    ----------
    out_dir : pathlib.Path
        The per-item output folder.

    Returns
    -------
    dict
        ``seed``, ``deployment_source``, ``specs`` (every question any draw chose: id to spec fields) and
        ``draws`` (one record per draw: label folder, ``per_cell``, drawn ``ids``, label-file fingerprints,
        view check, cell counts).

    Raises
    ------
    FileNotFoundError
        If no sample has been drawn.
    """
    path = out_dir / "sample.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the 'sample' step first")
    return json.loads(path.read_text())


def sample_spec_map(sample: Mapping[str, Any]) -> dict[str, QuestionSpec]:
    """The sampled questions by id.

    Parameters
    ----------
    sample : mapping
        The record from :func:`load_sample`.

    Returns
    -------
    dict of str to QuestionSpec
        Every sampled question.

    Raises
    ------
    ValueError
        If a stored spec no longer hashes to its id.
    """
    specs = {}
    for item_id, record in sample["specs"].items():
        spec = spec_from_dict(record)
        if spec.id != item_id:
            raise ValueError(f"Stored spec {item_id} has id {spec.id}")
        specs[item_id] = spec
    return specs


def sample_specs(sample: Mapping[str, Any], seed: int) -> list[QuestionSpec]:
    """The sampled questions in asking order: grouped by image, images in hashed order.

    Parameters
    ----------
    sample : mapping
        The record from :func:`load_sample`.
    seed : int
        Seed of the image order.

    Returns
    -------
    list of QuestionSpec
        Every sampled question.
    """
    specs = sample_spec_map(sample).values()
    return sorted(specs, key=lambda s: (sample_rank(seed, "|".join(view_key(s))), s.id))


# --------------------------------------------------------------------------------------------- predictions


def prediction_path(out_dir: Path, model: str, blind: bool) -> Path:
    """Where one model's predictions are written.

    Parameters
    ----------
    out_dir : pathlib.Path
        The per-item output folder.
    model : str
        Model key.
    blind : bool
        Text-only run.

    Returns
    -------
    pathlib.Path
        ``<out>/predictions/<model>[_blind].jsonl``.
    """
    return out_dir / "predictions" / f"{model}{'_blind' if blind else ''}.jsonl"


def cmd_predict(args: argparse.Namespace) -> int:
    """Answer the sampled questions with one model, with images and/or blind.

    Parameters
    ----------
    args : argparse.Namespace
        ``out_dir``, ``model``, ``modes``, ``limit``, ``seed``.

    Returns
    -------
    int
        Exit status 0.
    """
    from measured_space.eval_vlm import VLMRunner, hf_cache_dir

    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_dir()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch

    sample = load_sample(args.out_dir)
    specs = sample_specs(sample, args.seed)
    torch.cuda.reset_peak_memory_stats()
    runner = VLMRunner(MODELS[args.model], DEFAULT_RESOLUTION)
    print(f"loaded {runner.info['model_id']} @ {runner.info['snapshot']} in {runner.load_seconds:.1f} s", flush=True)
    loader = SpecImageLoader()
    for mode in args.modes:
        blind = mode == "blind"
        path = prediction_path(args.out_dir, args.model, blind)
        started, t0 = utc_now(), time.perf_counter()
        summary = run_predictions(runner, specs, path, None if blind else loader, blind=blind, limit=args.limit,
                                  log=lambda line: print(line, flush=True))
        seconds = time.perf_counter() - t0
        run = {"model": args.model, "mode": mode, "started": started, "finished": utc_now(),
               "n_specs": summary.n_specs, "n_skipped": summary.n_skipped, "n_written": summary.n_written,
               "seconds": round(seconds, 1),
               "seconds_per_question": round(seconds / summary.n_written, 3) if summary.n_written else None,
               "load_seconds": round(runner.load_seconds, 1),
               "cuda_max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "images_decoded": loader.n_decoded, "limit": args.limit, "config": runner.info}
        with (args.out_dir / "runs.jsonl").open("a") as f:
            f.write(json.dumps(run) + "\n")
        remaining = summary.n_specs - summary.n_skipped - summary.n_written
        print(f"{args.model} {mode}: wrote {summary.n_written} in {seconds:.1f} s "
              f"({run['seconds_per_question']} s/question); {remaining} left; "
              f"peak VRAM {run['cuda_max_memory_allocated_gib']} GiB", flush=True)
    return 0


def load_predictions(out_dir: Path, ids: set[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Parsed predictions of every model and mode found, restricted to the sampled ids.

    Parameters
    ----------
    out_dir : pathlib.Path
        The per-item output folder.
    ids : set of str
        Sampled question ids.

    Returns
    -------
    frame : pandas.DataFrame
        :func:`measured_space.scoring.predictions_frame` of all records; blind runs are ``<model>+blind``.
    raw : list of dict
        The prediction records themselves (for runtime statistics).
    """
    raw: list[dict[str, Any]] = []
    for path in sorted((out_dir / "predictions").glob("*.jsonl")):
        raw += [r for r in read_predictions(path) if r["id"] in ids]
    return scoring.predictions_frame(raw), raw


def scoring_name(record: Mapping[str, Any]) -> str:
    """The name a prediction record is scored under.

    Parameters
    ----------
    record : mapping
        A prediction record.

    Returns
    -------
    str
        The model key, with :data:`measured_space.scoring.BLIND_SUFFIX` for blind runs (as
        :func:`measured_space.scoring.predictions_frame` names them).
    """
    return str(record.get("model") or record["model_id"]) + (scoring.BLIND_SUFFIX if record.get("blind") else "")


def runtime_table(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Per-model cost of the predictions: latency, tokens, truncation and forced answers.

    Parameters
    ----------
    records : sequence of mapping
        Prediction records.

    Returns
    -------
    pandas.DataFrame
        One row per model (``+blind`` for blind runs). Columns ``model, n, latency_mean_s,
        latency_median_s, latency_p95_s, latency_total_s, new_tokens_mean, image_tokens_mean,
        truncated_rate, forced_rate, max_new_tokens``.
    """
    frame = pd.DataFrame([{"model": scoring_name(r),
                           "latency_s": r["latency_s"], "n_new_tokens": r["n_new_tokens"],
                           "n_image_tokens": r["n_image_tokens"], "truncated": bool(r["truncated"]),
                           "forced": bool(r["forced"]), "max_new_tokens": r["max_new_tokens"]} for r in records])
    if frame.empty:
        return frame
    return frame.groupby("model", sort=True).agg(
        n=("latency_s", "size"), latency_mean_s=("latency_s", "mean"), latency_median_s=("latency_s", "median"),
        latency_p95_s=("latency_s", lambda x: float(np.quantile(x, 0.95))), latency_total_s=("latency_s", "sum"),
        new_tokens_mean=("n_new_tokens", "mean"), image_tokens_mean=("n_image_tokens", "mean"),
        truncated_rate=("truncated", "mean"), forced_rate=("forced", "mean"),
        max_new_tokens=("max_new_tokens", "max")).reset_index()


# --------------------------------------------------------------------------------------------- scoring


def load_accepted(path: Path) -> dict[str, dict[str, set[str]]] | None:
    """Gate-accepted ids per dataset and depth source.

    Parameters
    ----------
    path : pathlib.Path
        ``accepted_ids.json``: ``{"accepted": {dataset: {"auto_<depth>": [ids]}}, ...}`` or the bare
        ``{dataset: {"auto_<depth>": [ids]}}``.

    Returns
    -------
    dict or None
        Dataset to depth source (without the ``auto_`` prefix) to ids; None if the file does not exist.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    accepted = data["accepted"] if isinstance(data.get("accepted"), dict) else data
    return {dataset: {str(source).removeprefix("auto_"): set(ids) for source, ids in by_source.items()}
            for dataset, by_source in accepted.items() if isinstance(by_source, dict)}


def scoring_units(datasets: Iterable[str], labels_root: Path) -> list[Unit]:
    """One unit per available ``(dataset, depth)`` source, then the pooled deployment unit.

    Parameters
    ----------
    datasets : iterable of str
        Sampled datasets.
    labels_root : pathlib.Path
        The label pipeline's output folder.

    Returns
    -------
    list of Unit
        Units whose ``qa_auto_<depth>.jsonl`` exists and is not empty, and :data:`POOLED_UNIT`.
    """
    datasets = list(datasets)
    units = [Unit(f"{d}/{s}", ((d, s),)) for d in datasets for s in DEPTH_SOURCES[d]
             if (p := labels_root / d / f"qa_auto_{s}.jsonl").exists() and p.stat().st_size > 0]
    units.append(Unit(POOLED_UNIT, tuple((d, DEPLOYMENT_SOURCE[d]) for d in datasets)))
    return units


def unit_labels(unit: Unit, specs: Mapping[str, QuestionSpec], labels_root: Path,
                accepted: Mapping[str, Mapping[str, set[str]]] | None) -> tuple[pd.DataFrame, bool]:
    """The label table of one unit, restricted to the sampled questions.

    Parameters
    ----------
    unit : Unit
        The scoring unit.
    specs : mapping of str to QuestionSpec
        The sampled questions, by id.
    labels_root : pathlib.Path
        The label folder scored against.
    accepted : mapping or None
        Output of :func:`load_accepted`.

    Returns
    -------
    labels : pandas.DataFrame
        :func:`measured_space.scoring.label_table` output (automatic items relabelled ``source="auto"``).
        ``attrs`` adds ``n_sample_without_oracle`` and ``n_sample_without_auto`` (sampled questions this
        label folder does not answer), and ``n_accepted_without_auto`` (accepted ids, over the whole label
        set, that have no automatic label here: a sign that the accepted ids come from another label set).
    accepted_available : bool
        Whether the gate's decisions exist for every part of the unit. If not, ``accepted`` is all False and
        must not be reported.

    Raises
    ------
    ValueError
        If a sampled question's spec differs from the label folder's spec under the same id.
    """
    oracle_items: list[QAItem] = []
    auto_items: list[QAItem] = []
    accepted_ids: set[str] = set()
    available = accepted is not None
    n_without_oracle = n_without_auto = n_accepted_without_auto = 0
    for dataset, source in unit.parts:
        ids = sorted(i for i, s in specs.items() if s.dataset == dataset)
        oracle = cached_items(labels_root / dataset / "qa_oracle.jsonl")
        auto = cached_items(labels_root / dataset / f"qa_auto_{source}.jsonl")
        for i in ids:
            if i in oracle and oracle[i].spec != specs[i]:
                raise ValueError(f"Sampled question {i} differs from its spec in {labels_root / dataset}")
        oracle_items += [oracle[i] for i in ids if i in oracle]
        auto_items += [replace(auto[i], source="auto") for i in ids if i in auto]
        n_without_oracle += sum(1 for i in ids if i not in oracle)
        n_without_auto += sum(1 for i in ids if i not in auto)
        if accepted is not None and source in accepted.get(dataset, {}):
            accepted_ids |= accepted[dataset][source] & set(ids)
            n_accepted_without_auto += len(accepted[dataset][source] - set(auto))
        else:
            available = False
    labels = scoring.label_table(oracle_items, auto_items, accepted_ids=accepted_ids)
    labels.attrs.update(n_sample_without_oracle=n_without_oracle, n_sample_without_auto=n_without_auto,
                        n_accepted_without_auto=n_accepted_without_auto)
    return labels, available


@lru_cache(maxsize=16)
def cached_items(path: Path) -> dict[str, QAItem]:
    """:func:`load_items`, cached: several scoring units read the same oracle file.

    Parameters
    ----------
    path : pathlib.Path
        A ``qa_*.jsonl`` file.

    Returns
    -------
    dict
        Question id to item; treat it as read-only.
    """
    return load_items(path)


def ci_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Add reading aids to a delta-delta table.

    Parameters
    ----------
    table : pandas.DataFrame
        :func:`measured_space.scoring.delta_delta` output.

    Returns
    -------
    pandas.DataFrame
        With ``ci_excludes_zero`` (the interval lies strictly on one side of 0, from at least two scenes; one
        scene gives a degenerate interval) and ``same_sign`` (the gap between the models has the same sign,
        or is 0 under both, with oracle and auto labels).
    """
    out = table.copy()
    out["ci_excludes_zero"] = ((out["ci_low"] > 0) | (out["ci_high"] < 0)) & (out["n_scenes"] >= 2)
    out["same_sign"] = np.sign(out["delta_oracle"]) == np.sign(out["delta_auto"])
    return out


def analyse_unit(unit: Unit, labels: pd.DataFrame, predictions: pd.DataFrame, accepted_available: bool,
                 n_boot: int, boot_seed: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Score one unit and compute every aggregate table, under both forced-answer policies.

    Parameters
    ----------
    unit : Unit
        The scoring unit.
    labels : pandas.DataFrame
        From :func:`unit_labels`.
    predictions : pandas.DataFrame
        From :func:`load_predictions`.
    accepted_available : bool
        Whether the gate-accepted item set can be reported.
    n_boot : int
        Bootstrap replicates for delta-delta.
    boot_seed : int
        Bootstrap seed.

    Returns
    -------
    scored : pandas.DataFrame
        Per-item :func:`measured_space.scoring.score` rows (kept out of the repository).
    tables : dict of str to pandas.DataFrame
        ``accuracy`` (2x2 by question type), ``delta_delta``, ``imitation``, ``parse_methods`` and
        ``label_quality``, each with a ``unit`` column; policy-dependent tables have ``forced_policy``.
    info : dict
        Item, scene and accepted counts, unanswered questions per model, and notes on skipped analyses.
    """
    scored = scoring.score(labels, predictions)
    item_sets = list(scoring.ITEM_SETS) if accepted_available else ["all"]
    models = set(scored["model"])
    notes: list[str] = []
    accuracy, deltas, imitation = [], [], []
    for policy, include in FORCED_POLICIES.items():
        table = scoring.two_by_two(scored, include_forced=include)
        accuracy.append(table[table["item_set"].isin(item_sets)].assign(forced_policy=policy))
        strict = scored if include else scoring.count_forced_as_failures(scored)
        for item_set in item_sets:
            imitation.append(scoring.imitation_rate(strict, item_set=item_set).assign(forced_policy=policy))
            for a, b in MODEL_PAIRS:
                if not {a, b} <= models:
                    notes.append(f"delta_delta {a} vs {b}: missing predictions")
                    continue
                try:
                    dd = scoring.delta_delta(strict, a, b, item_set=item_set, n_boot=n_boot, seed=boot_seed)
                except ValueError as err:
                    notes.append(f"delta_delta {a} vs {b} ({item_set}, {policy}): {err}")
                    continue
                deltas.append(ci_columns(dd).assign(forced_policy=policy))
    quality = scoring.label_quality(labels)
    tables = {
        "accuracy": pd.concat(accuracy, ignore_index=True),
        "delta_delta": pd.concat(deltas, ignore_index=True) if deltas else pd.DataFrame(),
        "imitation": pd.concat(imitation, ignore_index=True),
        "parse_methods": scoring.parse_method_table(scored),
        "label_quality": quality[quality["item_set"].isin(item_sets)],
    }
    tables = {k: v.assign(unit=unit.name)[["unit"] + list(v.columns)] if len(v) else v for k, v in tables.items()}
    info = {"n_items": len(labels), "n_scenes": int(labels[["dataset", "scene"]].drop_duplicates().shape[0]),
            "n_accepted": int(labels["accepted"].sum()) if accepted_available else None,
            "accepted_available": accepted_available,
            "n_sample_without_oracle": labels.attrs["n_sample_without_oracle"],
            "n_sample_without_auto": labels.attrs["n_sample_without_auto"],
            "n_accepted_without_auto": labels.attrs["n_accepted_without_auto"] if accepted_available else None,
            "n_unanswered": {m: int(n) for m, n in scored.attrs["n_unanswered"].items()},
            "auto_label_accuracy": float(labels["label_ok"].mean()) if len(labels) else None,
            "notes": sorted(set(notes))}
    return scored, tables, info


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """A DataFrame as JSON-ready records, with NaN as null.

    Parameters
    ----------
    frame : pandas.DataFrame
        Any table of scalars.

    Returns
    -------
    list of dict
        One dict per row with Python scalars.
    """
    out = []
    for row in frame.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, float) and not math.isfinite(v):
                v = None
            clean[k] = v
        out.append(clean)
    return out


def print_tables(tables: Mapping[str, pd.DataFrame], runtime: pd.DataFrame) -> None:
    """Print the headline tables in a readable form.

    Parameters
    ----------
    tables : mapping of str to pandas.DataFrame
        Concatenated tables from :func:`analyse_unit`.
    runtime : pandas.DataFrame
        From :func:`runtime_table`.
    """
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    acc = tables["accuracy"]
    overall = acc[acc["qtype"] == "all"].pivot_table(
        index=["unit", "model", "item_set", "n"], columns=["forced_policy", "source"], values="accuracy").round(3)
    print("\n== Accuracy, all question types (columns: forced policy x label source)\n" + overall.to_string())
    pooled = acc[(acc["unit"] == POOLED_UNIT) & (acc["forced_policy"] == "counted") & (acc["item_set"] == "all")]
    per_type = pooled.pivot_table(index="qtype", columns=["model", "source"], values="accuracy").round(3)
    print(f"\n== {POOLED_UNIT}: accuracy by question type (all items, forced answers counted)\n"
          + per_type.to_string())
    rates = acc[(acc["unit"] == POOLED_UNIT) & (acc["qtype"] == "all") & (acc["source"] == "oracle")
                & (acc["item_set"] == "all")]
    rate_columns = ["model", "forced_policy", "n", "n_parsed", "parse_failure_rate", "forced_rate"]
    print(f"\n== {POOLED_UNIT}: parse-failure and forced-answer rates\n"
          + rates[rate_columns].round(3).to_string(index=False))
    if len(tables["delta_delta"]):
        dd = tables["delta_delta"]
        primary = (dd["model_a"] == MODEL_PAIRS[0][0]) & (dd["model_b"] == MODEL_PAIRS[0][1])
        dd = dd[(dd["qtype"] == "all") & (primary | (dd["unit"] == POOLED_UNIT))][
            ["unit", "model_a", "model_b", "item_set", "forced_policy", "n_items", "n_scenes", "delta_oracle",
             "delta_auto", "delta_delta", "ci_low", "ci_high", "ci_excludes_zero", "same_sign"]]
        print("\n== Delta-delta = (A-B)_auto - (A-B)_oracle, scene-cluster bootstrap 95% CI "
              f"(all units for {' vs '.join(MODEL_PAIRS[0])}; image vs blind for {POOLED_UNIT})\n"
              + dd.round(3).to_string(index=False))
    im = tables["imitation"]
    im = im[im["qtype"] == "all"][["unit", "model", "item_set", "forced_policy", "n_label_errors", "n_imitated",
                                   "imitation_rate", "oracle_correct_rate"]]
    print("\n== Labeller imitation (among questions whose auto label is wrong)\n" + im.round(3).to_string(index=False))
    lq = tables["label_quality"]
    print("\n== Auto-label accuracy against oracle\n" + lq[lq["qtype"] == "all"].round(3).to_string(index=False))
    if len(runtime):
        print("\n== Runtime\n" + runtime.round(3).to_string(index=False))


def cmd_score(args: argparse.Namespace) -> int:
    """Score all predictions found against one label folder and write the aggregates.

    Parameters
    ----------
    args : argparse.Namespace
        ``out_dir``, ``labels_dir``, ``draw_from``, ``results_dir``, ``accepted``, ``n_boot``, ``boot_seed``,
        ``include_partial``, ``note``, ``gpu_watch``. The questions scored are the latest draw from
        ``draw_from`` (default ``labels_dir``). Runs that have not answered all of them are left out unless
        ``include_partial`` is set.

    Returns
    -------
    int
        Exit status: 0 on success, 1 if there is no draw to score, no predictions, or the label folder is
        unfinished.
    """
    sample = load_sample(args.out_dir)
    labels_root = args.labels_dir
    draw_from = args.draw_from or labels_root
    draw = latest_draw(sample, draw_from)
    if draw is None:
        print(f"no draw from {draw_from}: run 'sample --extend --labels-dir {draw_from}' first", flush=True)
        return 1
    all_specs = sample_spec_map(sample)
    specs = {i: all_specs[i] for i in draw["ids"]}
    datasets = sorted({s.dataset for s in specs.values()})
    for dataset in datasets:
        if not (labels_root / dataset / "summary.json").exists():
            print(f"{labels_root / dataset}/summary.json missing: labels are not finished", flush=True)
            return 1
    predictions, raw = load_predictions(args.out_dir, set(specs))
    counts = predictions.groupby("model")["id"].nunique() if len(predictions) else pd.Series(dtype=int)
    partial = {str(m): int(n) for m, n in counts.items() if n < len(specs)}
    if partial and not args.include_partial:
        print(f"leaving out runs that do not cover all {len(specs)} sampled questions: {partial}", flush=True)
        predictions = predictions[~predictions["model"].isin(partial)]
        raw = [r for r in raw if scoring_name(r) not in partial]
    if predictions.empty:
        print("no predictions found", flush=True)
        return 1
    used = {f"{d}/{p.name}": p for d in datasets for p in sorted((labels_root / d).glob("qa_*.jsonl"))}
    label_files = {name: file_fingerprint(path) for name, path in used.items()}
    own = latest_draw(sample, labels_root)
    changed = sorted(name for name, fp in (own or {}).get("label_files", {}).items() if label_files.get(name) != fp)
    if changed:
        print(f"WARNING: label files changed since the sample was drawn from them: {changed}", flush=True)
    accepted = load_accepted(args.accepted) if args.accepted is not None else None
    if args.accepted is not None and accepted is None:
        print(f"accepted ids file {args.accepted} does not exist", flush=True)
        return 1
    print(f"labels: {labels_root}; questions: the {len(specs)} drawn from {draw_from}; accepted ids: "
          f"{args.accepted or 'none given (all-items half only)'}", flush=True)
    parts: dict[str, list[pd.DataFrame]] = {}
    units_info: dict[str, Any] = {}
    scored_dir = args.out_dir / "scored" / labels_root.name
    scored_dir.mkdir(parents=True, exist_ok=True)
    for unit in scoring_units(datasets, labels_root):
        labels, available = unit_labels(unit, specs, labels_root, accepted)
        scored, tables, info = analyse_unit(unit, labels, predictions, available, args.n_boot, args.boot_seed)
        scored.to_csv(scored_dir / f"{unit.name.replace('/', '__')}.csv", index=False)
        units_info[unit.name] = info
        for name, table in tables.items():
            parts.setdefault(name, []).append(table)
    tables = {name: pd.concat([t for t in frames if len(t)], ignore_index=True).infer_objects()
              if any(len(t) for t in frames) else pd.DataFrame() for name, frames in parts.items()}
    runtime = runtime_table(raw)
    composition = sample_composition(specs)
    runs_path = args.out_dir / "runs.jsonl"
    runs = [json.loads(line) for line in runs_path.read_text().splitlines() if line.strip()] if runs_path.exists() \
        else []
    summary = {
        "created": utc_now(), "git_commit": git_commit(), "labels_dir": portable(labels_root),
        "questions_drawn_from": portable(draw_from), "n_questions": len(specs), "n_sample_total": len(all_specs),
        "design": {"sample_seed": sample["seed"], "per_cell": draw["per_cell"],
                   "deployment_source": sample["deployment_source"], "n_boot": args.n_boot,
                   "boot_seed": args.boot_seed, "ci_level": 0.95, "model_pairs": [list(p) for p in MODEL_PAIRS],
                   "forced_policies": {"counted": "answers forced after cut-off reasoning count as answers",
                                       "strict": "forced answers count as parse failures (wrong)"},
                   "resolution": DEFAULT_RESOLUTION.name,
                   "max_new_tokens": {k: m.max_new_tokens for k, m in MODELS.items()},
                   "accepted_ids": None if accepted is None else {"path": portable(args.accepted),
                                                                  **file_fingerprint(args.accepted)}},
        "label_files": label_files, "label_files_changed_since_sampling": changed,
        "sample_draws": [{**{k: v for k, v in d.items() if k not in ("git_commit", "ids")},
                          "labels_dir": portable(d["labels_dir"])} for d in sample["draws"]],
        "sample_composition": records(composition),
        "model_configs": {r["model"]: dict(r["config"]) for r in runs},
        "runs": [{k: v for k, v in r.items() if k != "config"} for r in runs],
        "partial_runs": {"n_sampled": len(specs), "answered": partial,
                         "scored": bool(args.include_partial)},
        "gpu_watch": dict(item.split("=", 1) for item in args.gpu_watch),
        "notes": list(args.note),
        "units": units_info,
        "runtime": records(runtime),
        **{name: records(table) for name, table in tables.items()},
    }
    results_dir = args.results_dir or REPO_ROOT / "results" / "vlm" / labels_root.name
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, table in {**tables, "runtime": runtime, "sample_composition": composition}.items():
        table.to_csv(results_dir / f"{name}.csv", index=False, float_format="%.4f")
    print_tables(tables, runtime)
    print("\n== Units (notes are in summary.json)\n" + pd.DataFrame([
        {"unit": k, **{f: v for f, v in info.items() if f not in ("n_unanswered", "notes")},
         "max_unanswered": max(info["n_unanswered"].values(), default=0), "n_notes": len(info["notes"])}
        for k, info in units_info.items()]).round(3).to_string(index=False))
    print(f"aggregates written to {results_dir}; per-item scored rows in {scored_dir}", flush=True)
    return 0


# --------------------------------------------------------------------------------------------- main


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments; None reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments, with ``command`` naming the step. ``out_dir`` and ``labels_dir`` default to
        ``<data root>/outputs/vlm`` and ``<data root>/outputs/labels``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=None, help="per-item outputs (default <data>/outputs/vlm)")
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (also orders images in 'predict')")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("sample", help="draw from a label folder and add the draw to the sample (CPU)")
    p.add_argument("--labels-dir", type=Path, default=None, help="label folder (default <data>/outputs/labels)")
    p.add_argument("--per-cell", type=int, default=25, help="questions drawn per dataset x question type")
    p.add_argument("--datasets", nargs="+", default=list(DEPLOYMENT_SOURCE), choices=list(DEPLOYMENT_SOURCE))
    p.add_argument("--extend", action="store_true", help="add a draw to the existing sample (never removes)")
    p.add_argument("--force", action="store_true", help="discard an existing sample and redraw")
    p = sub.add_parser("predict", help="answer the sample with one model (GPU)")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--modes", nargs="+", choices=["image", "blind"], default=["image", "blind"])
    p.add_argument("--limit", type=int, default=None, help="new questions per mode (timing runs)")
    p = sub.add_parser("score", help="score predictions against one label folder and write aggregates (CPU)")
    p.add_argument("--labels-dir", type=Path, default=None, help="label folder (default <data>/outputs/labels)")
    p.add_argument("--draw-from", type=Path, default=None,
                   help="score the questions drawn from this label folder (default: --labels-dir)")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="aggregates folder (default results/vlm/<label folder name>)")
    p.add_argument("--accepted", type=Path, default=None,
                   help="gate-accepted ids made from the same label folder; without it only all items are scored")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--include-partial", action="store_true",
                   help="also score runs that have not answered every sampled question (preliminary looks)")
    p.add_argument("--note", action="append", default=[], help="a design note to record in summary.json")
    p.add_argument("--gpu-watch", action="append", default=[], metavar="RUN=LINE",
                   help="a GPU_WATCH SUMMARY line to record, e.g. 'qwen3-vl-8b=GPU_WATCH SUMMARY ...'")
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = default_out_dir()
    if getattr(args, "labels_dir", "unset") is None:
        args.labels_dir = data_root() / "outputs" / "labels"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run one step.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments; None reads ``sys.argv``.

    Returns
    -------
    int
        The step's exit status.
    """
    args = parse_args(argv)
    return {"sample": cmd_sample, "predict": cmd_predict, "score": cmd_score}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
