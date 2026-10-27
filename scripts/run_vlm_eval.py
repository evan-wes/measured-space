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

    Drawing only questions the automatic source answered conditions on the labeller having found the objects.
    ``--include-missed`` adds the other side: a separate draw, with the same cells, ``--per-cell`` and keys,
    of the questions with an oracle answer but no answer from the deployment source, because the source
    matched no box to a referent (recorded per view in ``views.jsonl``). Count questions have no referents and
    are always answered, so they are never missed. The draw's ``missed`` record lists these items, each tagged
    ``"missed": true`` with its unmatched referent ids. They join the sample like any other question, so
    ``predict`` answers them next time.
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
    :mod:`measured_space.scoring`. ``--datasets`` keeps only some of the draw's datasets, for a folder that
    covers fewer of them. A question's id depends only on its template, view, referents and parameters, so a
    draw made from one folder can be scored against another folder built from the same views.

    The sources are found, not listed: every non-empty ``<dataset>/qa_auto_<source>.jsonl`` is one, whichever
    labeller wrote it (depth sources, CuTR box sources, ...). Results are per source
    (``<dataset>/<source>``) and for the pooled deployment unit. The pooled unit takes one source per dataset:

    1. the one given with ``--deployment-source <dataset>=<source>``;
    2. else the v0 deployment source (:data:`DEPLOYMENT_SOURCE`), if the folder has it;
    3. else the folder's only source, if it has exactly one.

    If a dataset has none of these, the pooled unit is skipped and ``summary.json`` says why. Its choice is
    ``design.pooled_deployment_source``; ``design.deployment_source`` stays the source the draw required.
    The tables are:

    - per-type accuracy;
    - the 2x2 (all vs gate-accepted items, oracle vs auto labels);
    - delta-delta between models, with a scene-cluster bootstrap CI;
    - labeller-imitation rate;
    - parse-failure and forced-answer rates.

    Blind runs appear as models named ``<model>+blind``. Every table is computed twice:

    - with answers forced after cut-off reasoning counted as answers (``forced_policy = "counted"``);
    - with those answers counted as failures (``"strict"``).

    The gate-accepted half needs ``--accepted``, an ``accepted_ids.json`` made from the same label folder.
    Without it, only the "all items" half is reported. The file lists accepted ids for every gate cell, but an
    id joins the accepted item set only if its cell (dataset x answer kind, numeric or categorical) was
    certified on held-out scenes (``certified_ok`` in the file's ``held_out``). ``summary.json``
    (``accepted_certification``) lists the cells left out and how many items each would have added. A file
    without ``held_out`` (written by an older audit) is read as before, every listed id counting, with a
    warning.

    Missed questions of the draw, if it has any, have no automatic label: they are scored against the oracle
    only and enter none of the tables above. ``missed_vs_answerable`` compares, per model, dataset and question
    type, oracle-referenced accuracy on missed and on answerable questions, with a scene-cluster bootstrap CI
    on the difference (:func:`measured_space.scoring.missed_vs_answerable`). That difference estimates how much
    easier a question set generated from automatic labels is than the full one.

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
    python scripts/run_vlm_eval.py score --labels-dir <data>/outputs/labels_cutr_sam3 \
        --draw-from <data>/outputs/labels_2d --datasets arkitscenes    # another labeller, same questions
    python scripts/run_vlm_eval.py sample --extend --include-missed --labels-dir <data>/outputs/labels_2d
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

AUTO_PREFIX = "qa_auto_"
"""File-name prefix of an automatic label source: ``qa_auto_<source>.jsonl`` holds the items of ``<source>``."""

DEPLOYMENT_SOURCE: dict[str, str] = {"arkitscenes": "lidar", "smartspaces": "moge"}
"""The v0 automatic source a deployment would have, per dataset. ``sample`` requires an answer from it;
``score`` pools it by default when the label folder has it."""

POOLED_UNIT = "pooled/deployment"
"""Name of the scoring unit that pools each dataset's deployment source."""

ANSWER_KINDS: tuple[str, ...] = ("numeric", "categorical")
"""The label gate's answer kinds, as in ``held_out`` of ``accepted_ids.json`` (see :func:`answer_kind`)."""

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
        ``"<dataset>/<source>"``, or :data:`POOLED_UNIT`.
    parts : tuple of (str, str)
        ``(dataset, automatic source)`` pairs whose automatic labels are pooled.
    """

    name: str
    parts: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class AcceptedIds:
    """The label gate's decisions, read from ``accepted_ids.json`` (written by ``scripts/run_audit.py``).

    The file lists the ids the primary gate accepted in every gate cell, including cells whose error bound
    failed on held-out scenes. Only ids of certified cells join the accepted item set (:meth:`split`).

    Attributes
    ----------
    ids : dict of str to dict of str to frozenset of str
        Dataset to automatic source (without the ``auto_`` prefix) to the accepted ids listed.
    certified : dict of (str, str) to bool, or None
        ``(dataset, answer kind)`` to ``certified_ok`` of the primary gate at ``tau_scale``: whether the error
        among accepted labels on held-out scenes stayed within the bound. A cell without a row is not
        certified. None if the file has no ``held_out`` results (an older audit); then every listed id counts.
    tau_scale : float
        Tolerance multiple the listed ids were accepted at (``settings.tau_scale``, else 1.0).
    held_out : tuple of dict
        The file's ``held_out`` rows at ``tau_scale``, as recorded.
    """

    ids: dict[str, dict[str, frozenset[str]]]
    certified: dict[tuple[str, str], bool] | None
    tau_scale: float = 1.0
    held_out: tuple[dict[str, Any], ...] = ()

    def lists(self, dataset: str, source: str) -> bool:
        """Whether the gate's decisions cover one dataset's automatic source.

        Parameters
        ----------
        dataset : str
            Dataset name.
        source : str
            Automatic source, without the ``auto_`` prefix.

        Returns
        -------
        bool
            True if the file lists accepted ids (possibly none) for it.
        """
        return source in self.ids.get(dataset, {})

    def counts(self, dataset: str, kind: str) -> bool:
        """Whether the accepted ids of one gate cell join the accepted item set.

        Parameters
        ----------
        dataset : str
            Dataset name.
        kind : str
            Answer kind, one of :data:`ANSWER_KINDS`.

        Returns
        -------
        bool
            True if the cell is certified, or if the file has no held-out results (the old behaviour).
        """
        return self.certified is None or self.certified.get((dataset, kind)) is True

    def split(self, dataset: str, source: str, kinds: Mapping[str, str]) -> tuple[set[str], dict[str, set[str]]]:
        """The listed ids among some questions, split into those that count and those left out as uncertified.

        Parameters
        ----------
        dataset : str
            Dataset name.
        source : str
            Automatic source, without the ``auto_`` prefix.
        kinds : mapping of str to str
            Question id to answer kind (:func:`answer_kind`) for the questions considered. Listed ids outside
            it are ignored.

        Returns
        -------
        kept : set of str
            Listed ids whose cell counts (:meth:`counts`).
        dropped : dict of str to set of str
            Answer kind to the listed ids of its uncertified cell; only kinds that drop something.
        """
        kept: set[str] = set()
        dropped: dict[str, set[str]] = {}
        for item_id in self.ids.get(dataset, {}).get(source, frozenset()):
            kind = kinds.get(item_id)
            if kind is None:
                continue
            if self.counts(dataset, kind):
                kept.add(item_id)
            else:
                dropped.setdefault(kind, set()).add(item_id)
        return kept, dropped


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


def missed_questions(views_path: Path, source: str, oracle: Mapping[str, QAItem],
                     auto: Mapping[str, QAItem]) -> dict[str, list[str]]:
    """Questions an automatic source missed: an oracle answer but no automatic one, because a referent went unmatched.

    The label pipeline records, per view and source (``auto_unanswered`` in ``views.jsonl``), every question
    without an automatic answer and the referents that source matched no box to. Count questions have no
    referents and always get an automatic answer, so they are never missed.

    Parameters
    ----------
    views_path : pathlib.Path
        A dataset's ``views.jsonl``.
    source : str
        Automatic source, without the ``auto_`` prefix.
    oracle : mapping of str to QAItem
        Oracle items by id.
    auto : mapping of str to QAItem
        The source's automatic items by id.

    Returns
    -------
    dict of str to list of str
        Question id to the ids of its unmatched referents, for questions with an oracle answer, at least one
        unmatched referent and a type other than ``count``; sorted by question id.

    Raises
    ------
    ValueError
        If a view the source ran on has no ``auto_unanswered`` record for it (labels written before misses
        were recorded), or a question recorded as unanswered has an automatic answer.
    """
    missed: dict[str, list[str]] = {}
    with views_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            view = json.loads(line)
            if source not in view.get("depth_sources", []):
                continue
            unanswered = (view.get("auto_unanswered") or {}).get(f"auto_{source}")
            if unanswered is None:
                raise ValueError(f"{views_path}: view {view.get('view_key')} has no auto_unanswered record for "
                                 f"{source}; these labels predate the record of misses")
            for record in unanswered:
                item_id = str(record["id"])
                if item_id in auto:
                    raise ValueError(f"{views_path}: question {item_id} is recorded as unanswered by {source} "
                                     "but has an automatic answer")
                if item_id in oracle and record.get("missing") and record.get("qtype") != "count":
                    missed[item_id] = sorted(str(m) for m in record["missing"])
    return dict(sorted(missed.items()))


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
        ``labels_dir``, ``out_dir``, ``per_cell``, ``seed``, ``datasets``, ``extend``, ``force``,
        ``include_missed``. With ``include_missed``, the draw also takes a separate stratified draw (same cells,
        ``per_cell`` and seed) of the questions the deployment source missed (:func:`missed_questions`). It is
        recorded in the draw's ``missed`` record, with each item tagged ``"missed": true`` and its unmatched
        referent ids. The draw's ``ids`` stay the answerable questions.

    Returns
    -------
    int
        Exit status: 0 on success, 1 if a dataset's labels are unfinished, the sample exists and neither
        ``--extend`` nor ``--force`` was given, ``--extend`` uses another seed, or ``include_missed`` is set
        but the labels do not record misses.

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
    missed: dict[str, Any] | None = None
    if args.include_missed:
        missed = {"sources": {d: DEPLOYMENT_SOURCE[d] for d in args.datasets}, "views_files": {}, "cells": [],
                  "items": []}
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
        drawn_missed: dict[str, QuestionSpec] = {}
        if missed is not None:
            views_path = folder / "views.jsonl"
            try:
                unmatched = missed_questions(views_path, source, oracle, auto)
            except ValueError as err:
                print(f"--include-missed: {err}", flush=True)
                return 1
            pool = [oracle[i].spec for i in unmatched]
            drawn_missed = draw_sample(pool, args.per_cell, args.seed)
            chosen_cells = group_by_cell(drawn_missed)
            for (ds, qtype), ids in group_by_cell({s.id: s for s in pool}).items():
                chosen = chosen_cells.get((ds, qtype), [])
                missed["cells"].append({"dataset": ds, "qtype": qtype, "n_candidates": len(ids),
                                        "n_drawn": len(chosen), "n_new": sum(1 for i in chosen if i not in existing),
                                        "n_scenes": len({drawn_missed[i].scene for i in chosen})})
            missed["items"] += [{"id": i, "dataset": dataset, "qtype": drawn_missed[i].qtype, "missed": True,
                                 "missing_referents": unmatched[i]} for i in sorted(drawn_missed)]
            missed["views_files"][f"{dataset}/views.jsonl"] = file_fingerprint(views_path)
            sample["specs"].update({i: spec_to_dict(s) for i, s in drawn_missed.items() if i not in existing})
            existing.update(drawn_missed)
        draw["view_check"][dataset] = check_views(list(drawn.values()) + list(drawn_missed.values()),
                                                  folder / "views.jsonl", loader)
    draw["n_drawn"] = len(draw["ids"])
    draw["n_new"] = sum(c["n_new"] for c in draw["cells"])
    if missed is not None:
        missed.update(n_candidates=sum(c["n_candidates"] for c in missed["cells"]), n_drawn=len(missed["items"]),
                      n_new=sum(c["n_new"] for c in missed["cells"]))
        draw["missed"] = missed
    sample["draws"].append(draw)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sample, indent=1))
    print(pd.DataFrame(draw["cells"]).to_string(index=False))
    if missed is not None:
        print("\nmissed by the deployment source (oracle answer, no automatic answer):\n"
              + pd.DataFrame(missed["cells"], columns=["dataset", "qtype", "n_candidates", "n_drawn", "n_new",
                                                       "n_scenes"]).to_string(index=False))
    extra = "" if missed is None else f" + {missed['n_drawn']} missed ({missed['n_new']} new)"
    print(f"drew {draw['n_drawn']} ({draw['n_new']} new){extra}; sample now {len(sample['specs'])} questions; "
          f"views checked: {draw['view_check']}; written {out_path}", flush=True)
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


def answer_kind(spec: QuestionSpec) -> str:
    """The label gate's answer kind of a question.

    :func:`measured_space.audit.run_audit` has no named helper for this. It fits numeric gates to questions
    with ``answer_type == "number"`` and categorical gates to all others (choices and counts). ``held_out`` in
    ``accepted_ids.json`` has one row per dataset and answer kind. This function repeats that split.

    Parameters
    ----------
    spec : QuestionSpec
        The question.

    Returns
    -------
    str
        ``"numeric"`` or ``"categorical"`` (:data:`ANSWER_KINDS`).
    """
    return "numeric" if spec.answer_type == "number" else "categorical"


def load_accepted(path: Path) -> AcceptedIds | None:
    """Gate-accepted ids per dataset and automatic source, with the held-out certificate of each gate cell.

    Parameters
    ----------
    path : pathlib.Path
        ``accepted_ids.json`` from ``scripts/run_audit.py``: ``{"settings": {"tau_scale": ..., ...},
        "held_out": [{"dataset", "answer_kind", "tau_scale", "certified_ok", ...}], "accepted": {dataset:
        {"auto_<source>": [ids]}}, ...}``. Older files have no ``held_out``. The bare ``{dataset:
        {"auto_<source>": [ids]}}`` is also read, as having no held-out results.

    Returns
    -------
    AcceptedIds or None
        None if the file does not exist. ``certified`` comes from the ``held_out`` rows at the file's
        ``tau_scale``, and is None if the file has no ``held_out`` list.
    """
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    full = isinstance(data.get("accepted"), dict)
    accepted = data["accepted"] if full else data
    ids = {str(dataset): {str(source).removeprefix("auto_"): frozenset(map(str, listed))
                          for source, listed in by_source.items()}
           for dataset, by_source in accepted.items() if isinstance(by_source, dict)}
    settings = data.get("settings") if full and isinstance(data.get("settings"), dict) else {}
    tau_scale = float(settings.get("tau_scale", 1.0))
    held_out = data.get("held_out") if full else None
    if not isinstance(held_out, list):
        return AcceptedIds(ids, None, tau_scale)
    rows = tuple(dict(r) for r in held_out if float(r.get("tau_scale", 1.0)) == tau_scale)
    certified: dict[tuple[str, str], bool] = {}
    for r in rows:
        key = (str(r["dataset"]), str(r["answer_kind"]))
        certified[key] = certified.get(key, True) and r.get("certified_ok") is True
    return AcceptedIds(ids, certified, tau_scale, rows)


def discover_sources(folder: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The automatic label sources in one dataset's label folder.

    Every ``qa_auto_<source>.jsonl`` is a source, whichever labeller wrote it (the label pipeline's depth
    sources, ``scripts/run_box_source.py``'s CuTR sources, ...).

    Parameters
    ----------
    folder : pathlib.Path
        ``<label folder>/<dataset>``; a missing folder has no sources.

    Returns
    -------
    sources : tuple of str
        Sources whose file is not empty, sorted.
    empty : tuple of str
        Sources whose file is empty (nothing to score), sorted.
    """
    found = {p.name[len(AUTO_PREFIX):-len(".jsonl")]: p for p in folder.glob(f"{AUTO_PREFIX}*.jsonl")
             if p.is_file()}
    found.pop("", None)
    sources = tuple(sorted(s for s, p in found.items() if p.stat().st_size > 0))
    empty = tuple(sorted(s for s, p in found.items() if p.stat().st_size == 0))
    return sources, empty


def parse_deployment_sources(pairs: Sequence[str]) -> dict[str, str]:
    """Read ``--deployment-source`` arguments.

    Parameters
    ----------
    pairs : sequence of str
        ``"<dataset>=<source>"`` strings, e.g. ``"arkitscenes=cutr_sam3"``.

    Returns
    -------
    dict of str to str
        Dataset to source.

    Raises
    ------
    ValueError
        If a string is not ``<dataset>=<source>``, or one dataset is given two different sources.
    """
    out: dict[str, str] = {}
    for pair in pairs:
        dataset, sep, source = (part.strip() for part in pair.partition("="))
        if not (sep and dataset and source):
            raise ValueError(f"--deployment-source expects <dataset>=<source>, got {pair!r}")
        if out.get(dataset, source) != source:
            raise ValueError(f"--deployment-source gives {dataset} two sources: {out[dataset]} and {source}")
        out[dataset] = source
    return out


def resolve_deployment_sources(sources: Mapping[str, Sequence[str]],
                               requested: Mapping[str, str]) -> dict[str, Any]:
    """Choose the source each dataset contributes to the pooled deployment unit.

    For each dataset, the first that applies:

    1. the source requested for it (``--deployment-source``);
    2. its v0 deployment source (:data:`DEPLOYMENT_SOURCE`), if the folder has that source;
    3. the folder's only source, if it has exactly one.

    If any dataset has none of these, the pooled unit is skipped. Pooling only some of the datasets under the
    same name would change what the unit measures.

    Parameters
    ----------
    sources : mapping of str to sequence of str
        The datasets scored, each with the automatic sources its label folder has (:func:`discover_sources`).
    requested : mapping of str to str
        Dataset to source, from :func:`parse_deployment_sources`.

    Returns
    -------
    dict
        JSON-ready record:

        - ``sources``: dataset to the chosen source, or None if the pooled unit is skipped;
        - ``chosen_by``: dataset to ``"argument"``, ``"default mapping"`` or ``"only source"``, for each
          dataset that has a source;
        - ``note``: why the pooled unit is skipped, else None;
        - ``ignored``: requested datasets that are not scored.

    Raises
    ------
    ValueError
        If a requested source has no non-empty label file for its dataset.
    """
    chosen: dict[str, str] = {}
    chosen_by: dict[str, str] = {}
    unresolved: list[str] = []
    for dataset in sorted(sources):
        available = list(sources[dataset])
        default = DEPLOYMENT_SOURCE.get(dataset)
        if dataset in requested:
            if requested[dataset] not in available:
                raise ValueError(f"--deployment-source {dataset}={requested[dataset]}: the label folder has no "
                                 f"non-empty {AUTO_PREFIX}{requested[dataset]}.jsonl for {dataset} "
                                 f"(sources: {', '.join(available) or 'none'})")
            chosen[dataset], chosen_by[dataset] = requested[dataset], "argument"
        elif default is not None and default in available:
            chosen[dataset], chosen_by[dataset] = default, "default mapping"
        elif len(available) == 1:
            chosen[dataset], chosen_by[dataset] = available[0], "only source"
        else:
            unresolved.append(f"{dataset} (sources: {', '.join(available) or 'none'})")
    note = None
    if unresolved:
        note = (f"{POOLED_UNIT} skipped: no deployment source for {'; '.join(unresolved)}. The default needs "
                f"the v0 source ({', '.join(f'{d}={s}' for d, s in DEPLOYMENT_SOURCE.items())}) or exactly one "
                "source; pass --deployment-source <dataset>=<source> to choose one.")
    return {"sources": None if unresolved else chosen, "chosen_by": chosen_by, "note": note,
            "ignored": sorted(set(requested) - set(sources))}


def scoring_units(sources: Mapping[str, Sequence[str]], deployment: Mapping[str, str] | None) -> list[Unit]:
    """One unit per ``(dataset, source)`` found, then the pooled deployment unit.

    Parameters
    ----------
    sources : mapping of str to sequence of str
        Dataset to its automatic sources (:func:`discover_sources`).
    deployment : mapping of str to str or None
        Dataset to the source it contributes to :data:`POOLED_UNIT` (``sources`` of
        :func:`resolve_deployment_sources`); None leaves the pooled unit out.

    Returns
    -------
    list of Unit
        Per-source units in dataset and source order, then :data:`POOLED_UNIT` unless ``deployment`` is None.
    """
    units = [Unit(f"{d}/{s}", ((d, s),)) for d in sorted(sources) for s in sources[d]]
    if deployment is not None:
        units.append(Unit(POOLED_UNIT, tuple((d, deployment[d]) for d in sorted(deployment))))
    return units


def certification_record(accepted: AcceptedIds, sources: Mapping[str, Sequence[str]],
                         specs: Mapping[str, QuestionSpec], labels_root: Path) -> dict[str, Any]:
    """What the certified-cells rule did to the accepted item set, for ``summary.json``.

    Parameters
    ----------
    accepted : AcceptedIds
        From :func:`load_accepted`.
    sources : mapping of str to sequence of str
        The datasets scored, each with its automatic sources.
    specs : mapping of str to QuestionSpec
        The questions scored, by id.
    labels_root : pathlib.Path
        The label folder scored against.

    Returns
    -------
    dict
        JSON-ready record:

        - ``rule`` and ``tau_scale``;
        - ``legacy``: True if the file has no held-out results, so every listed id counts;
        - ``note``: a warning in that case, else None;
        - ``cells``: the ``held_out`` row of each dataset and answer kind (``accept_n``, ``accept_err`` and its
          Wilson interval, ``certified_ok``);
        - ``dropped``: one row per dataset, source and answer kind whose listed ids were left out, with
          ``certified_ok`` (False, or None if the file has no row for the cell). ``n_listed`` counts the
          listed ids that have an automatic label in this folder. ``n_scored`` counts those among the
          questions scored here, which is how many items the cell would have added to the accepted set.
    """
    keys = ("dataset", "answer_kind", "accept_n", "accept_err", "accept_lo", "accept_hi", "certified_ok")
    cells = [{k: r[k] for k in keys if k in r} for r in accepted.held_out]
    dropped: list[dict[str, Any]] = []
    for dataset in sorted(sources):
        oracle = cached_items(labels_root / dataset / "qa_oracle.jsonl")
        for source in sources[dataset]:
            if not accepted.lists(dataset, source):
                continue
            auto = cached_items(labels_root / dataset / f"{AUTO_PREFIX}{source}.jsonl")
            _, listed = accepted.split(dataset, source, {i: answer_kind(item.spec) for i, item in auto.items()})
            _, scored = accepted.split(dataset, source, {i: answer_kind(s) for i, s in specs.items()
                                                         if s.dataset == dataset and i in oracle and i in auto})
            for kind in sorted(listed):
                ok = None if accepted.certified is None else accepted.certified.get((dataset, kind))
                dropped.append({"dataset": dataset, "source": source, "answer_kind": kind, "certified_ok": ok,
                                "n_listed": len(listed[kind]), "n_scored": len(scored.get(kind, ()))})
    note = None
    if accepted.certified is None:
        note = ("WARNING: the accepted ids file has no held_out results (an older audit), so every listed id "
                "counts as accepted, including ids from gate cells that did not certify on held-out scenes. "
                "Re-run scripts/run_audit.py to restrict the accepted set to certified cells.")
    return {"rule": "an accepted id counts only if its dataset x answer-kind cell has certified_ok true in "
                    "held_out at the accepted ids' tau_scale",
            "tau_scale": accepted.tau_scale, "legacy": accepted.certified is None, "note": note,
            "cells": cells, "dropped": dropped}


def unit_labels(unit: Unit, specs: Mapping[str, QuestionSpec], labels_root: Path,
                accepted: AcceptedIds | None) -> tuple[pd.DataFrame, bool]:
    """The label table of one unit, restricted to the sampled questions.

    Parameters
    ----------
    unit : Unit
        The scoring unit.
    specs : mapping of str to QuestionSpec
        The sampled questions, by id.
    labels_root : pathlib.Path
        The label folder scored against.
    accepted : AcceptedIds or None
        Output of :func:`load_accepted`. A question is accepted if the gate accepted it and its gate cell
        (dataset x :func:`answer_kind`) is certified (:meth:`AcceptedIds.split`).

    Returns
    -------
    labels : pandas.DataFrame
        :func:`measured_space.scoring.label_table` output (automatic items relabelled ``source="auto"``).
        ``attrs`` adds:

        - ``n_sample_without_oracle`` and ``n_sample_without_auto``: sampled questions this label folder
          does not answer;
        - ``n_accepted_without_auto``: accepted ids, over the whole label set, that have no automatic label
          here, a sign that the accepted ids come from another label set;
        - ``n_accepted_dropped_uncertified``: ``"<dataset>/<answer kind>"`` to the questions of the table that
          the gate accepted in an uncertified cell, and that are therefore not accepted.
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
    dropped: dict[str, int] = {}
    for dataset, source in unit.parts:
        ids = sorted(i for i, s in specs.items() if s.dataset == dataset)
        oracle = cached_items(labels_root / dataset / "qa_oracle.jsonl")
        auto = cached_items(labels_root / dataset / f"{AUTO_PREFIX}{source}.jsonl")
        for i in ids:
            if i in oracle and oracle[i].spec != specs[i]:
                raise ValueError(f"Sampled question {i} differs from its spec in {labels_root / dataset}")
        oracle_items += [oracle[i] for i in ids if i in oracle]
        auto_items += [replace(auto[i], source="auto") for i in ids if i in auto]
        n_without_oracle += sum(1 for i in ids if i not in oracle)
        n_without_auto += sum(1 for i in ids if i not in auto)
        if accepted is not None and accepted.lists(dataset, source):
            labelled = {i: answer_kind(specs[i]) for i in ids if i in oracle and i in auto}
            kept, uncertified = accepted.split(dataset, source, labelled)
            accepted_ids |= kept
            for kind, gone in uncertified.items():
                dropped[f"{dataset}/{kind}"] = dropped.get(f"{dataset}/{kind}", 0) + len(gone)
            n_accepted_without_auto += len(accepted.ids[dataset][source] - set(auto))
        else:
            available = False
    labels = scoring.label_table(oracle_items, auto_items, accepted_ids=accepted_ids)
    labels.attrs.update(n_sample_without_oracle=n_without_oracle, n_sample_without_auto=n_without_auto,
                        n_accepted_without_auto=n_accepted_without_auto,
                        n_accepted_dropped_uncertified=dict(sorted(dropped.items())))
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
        Item, scene and accepted counts, the accepted questions left out as uncertified (per dataset and
        answer kind), unanswered questions per model, and notes on skipped analyses.
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
            "n_accepted_dropped_uncertified": (labels.attrs["n_accepted_dropped_uncertified"]
                                               if accepted_available else None),
            "n_unanswered": {m: int(n) for m, n in scored.attrs["n_unanswered"].items()},
            "auto_label_accuracy": float(labels["label_ok"].mean()) if len(labels) else None,
            "notes": sorted(set(notes))}
    return scored, tables, info


def missed_analysis(draw: Mapping[str, Any], specs: Mapping[str, QuestionSpec], all_specs: Mapping[str, QuestionSpec],
                    datasets: Sequence[str], labels_root: Path, out_dir: Path, predictions: pd.DataFrame,
                    include_partial: bool, n_boot: int,
                    boot_seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Oracle-referenced accuracy on the draw's missed questions vs on its answerable ones.

    Missed questions (``sample --include-missed``) have no automatic label from the draw's deployment source,
    so they are scored against the oracle only, and never enter the 2x2, delta-delta or imitation tables.
    Both sides are scored against the oracle answers of ``labels_root``. Which questions are missed is fixed
    by the draw: the deployment source of the folder the draw was made from.

    Parameters
    ----------
    draw : mapping
        The draw scored; its ``missed`` record lists the missed questions.
    specs : mapping of str to QuestionSpec
        The answerable questions scored (the draw's ``ids`` in ``datasets``).
    all_specs : mapping of str to QuestionSpec
        Every sampled question, by id.
    datasets : sequence of str
        The datasets scored.
    labels_root : pathlib.Path
        The label folder whose oracle answers are used.
    out_dir : pathlib.Path
        The per-item output folder (its ``predictions``).
    predictions : pandas.DataFrame
        The predictions scored on the answerable questions (runs left out as partial are already removed).
    include_partial : bool
        Also use runs that have not answered every missed question.
    n_boot : int
        Bootstrap replicates.
    boot_seed : int
        Bootstrap seed.

    Returns
    -------
    scored : pandas.DataFrame
        Per-item :func:`measured_space.scoring.score_oracle` rows of both sides, with a boolean ``missed``
        column (kept out of the repository). Empty if the draw has no missed questions.
    table : pandas.DataFrame
        :func:`measured_space.scoring.missed_vs_answerable` under both forced-answer policies (column
        ``forced_policy``). It has the columns but no rows if there is nothing to compare.
    info : dict
        ``included``, the deployment ``sources`` that missed, counts (``n_missed``, ``n_answerable``,
        ``n_missed_without_oracle``), ``missed_composition``, the ``models`` compared, ``left_out_partial``
        (models without an answer to every missed question) and a ``note`` when nothing is compared.

    Raises
    ------
    ValueError
        If a question's spec differs from the label folder's spec under the same id.
    """
    table = pd.DataFrame(columns=list(scoring.MISSED_COLUMNS) + ["forced_policy"])
    record = draw.get("missed")
    if not record:
        return pd.DataFrame(), table, {"included": False, "note": "the draw has no missed questions; add them "
                                                                  "with 'sample --extend --include-missed'"}
    missed_specs = {r["id"]: all_specs[r["id"]] for r in record["items"] if r["dataset"] in datasets}
    frame, _ = load_predictions(out_dir, set(missed_specs))
    models = sorted(set(predictions["model"]))
    answered = frame.groupby("model")["id"].nunique() if len(frame) else pd.Series(dtype=int)
    incomplete = {m: int(answered.get(m, 0)) for m in models if answered.get(m, 0) < len(missed_specs)}
    keep = [m for m in models if include_partial or m not in incomplete]
    oracle_items: list[QAItem] = []
    n_without_oracle = 0
    for dataset in datasets:
        oracle = cached_items(labels_root / dataset / "qa_oracle.jsonl")
        for item_id, spec in list(specs.items()) + list(missed_specs.items()):
            if spec.dataset != dataset:
                continue
            if item_id not in oracle:
                if item_id in missed_specs:
                    n_without_oracle += 1
                continue
            if oracle[item_id].spec != spec:
                raise ValueError(f"Sampled question {item_id} differs from its spec in {labels_root / dataset}")
            oracle_items.append(oracle[item_id])
    scored = pd.DataFrame()
    if keep and len(frame):
        both = pd.concat([predictions[predictions["model"].isin(keep)], frame[frame["model"].isin(keep)]],
                         ignore_index=True)
        scored = scoring.score_oracle(oracle_items, both)
        scored["missed"] = scored["id"].isin(set(missed_specs))
        tables = []
        for policy, include in FORCED_POLICIES.items():
            base = scored if include else scoring.count_forced_as_failures(scored)
            tables.append(scoring.missed_vs_answerable(base, n_boot=n_boot, seed=boot_seed)
                          .assign(forced_policy=policy))
        table = pd.concat(tables, ignore_index=True) if any(len(t) for t in tables) else table
    note = None
    if not missed_specs:
        note = "the draw has no missed questions in the datasets scored"
    elif not len(table):
        note = "no model has answered every missed question yet: run 'predict' (or pass --include-partial)"
    info = {"included": True, "sources": record.get("sources"), "n_missed": len(missed_specs),
            "n_answerable": len(specs), "n_missed_without_oracle": n_without_oracle,
            "missed_composition": records(sample_composition(missed_specs)),
            "models": sorted(set(scored["model"])) if len(scored) else [],
            "left_out_partial": {} if include_partial else incomplete, "note": note}
    return scored, table, info


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
    if (acc["unit"] == POOLED_UNIT).any():
        pooled = acc[(acc["unit"] == POOLED_UNIT) & (acc["forced_policy"] == "counted") & (acc["item_set"] == "all")]
        per_type = pooled.pivot_table(index="qtype", columns=["model", "source"], values="accuracy").round(3)
        print(f"\n== {POOLED_UNIT}: accuracy by question type (all items, forced answers counted)\n"
              + per_type.to_string())
        rates = acc[(acc["unit"] == POOLED_UNIT) & (acc["qtype"] == "all") & (acc["source"] == "oracle")
                    & (acc["item_set"] == "all")]
        rate_columns = ["model", "forced_policy", "n", "n_parsed", "parse_failure_rate", "forced_rate"]
        print(f"\n== {POOLED_UNIT}: parse-failure and forced-answer rates\n"
              + rates[rate_columns].round(3).to_string(index=False))
    else:
        print(f"\n== {POOLED_UNIT}: skipped (see design.pooled_deployment_source in summary.json)")
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
        ``out_dir``, ``labels_dir``, ``draw_from``, ``datasets``, ``deployment_source`` (dataset to source,
        from :func:`parse_deployment_sources`), ``results_dir``, ``accepted``, ``n_boot``, ``boot_seed``,
        ``include_partial``, ``note``, ``gpu_watch``. The questions scored are the latest draw from
        ``draw_from`` (default ``labels_dir``), restricted to ``datasets`` if given. Runs that have not answered
        all of them are left out unless ``include_partial`` is set.

    Returns
    -------
    int
        Exit status: 0 on success, 1 if there is no draw to score, ``datasets`` names a dataset the draw lacks,
        the label folder is unfinished or has no automatic labels, a requested deployment source is missing,
        or there are no predictions.
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
    if args.datasets:
        missing = sorted(set(args.datasets) - set(datasets))
        if missing:
            print(f"--datasets {missing}: not in the draw from {draw_from} (it has {datasets})", flush=True)
            return 1
        datasets = sorted(set(args.datasets))
        specs = {i: s for i, s in specs.items() if s.dataset in datasets}
    for dataset in datasets:
        if not (labels_root / dataset / "summary.json").exists():
            print(f"{labels_root / dataset}/summary.json missing: labels are not finished (or the folder does "
                  f"not cover {dataset}: pass --datasets to score fewer datasets)", flush=True)
            return 1
    sources: dict[str, tuple[str, ...]] = {}
    empty_sources: dict[str, list[str]] = {}
    for dataset in datasets:
        sources[dataset], empty = discover_sources(labels_root / dataset)
        if empty:
            empty_sources[dataset] = list(empty)
            print(f"{dataset}: empty label files left out: {[f'{AUTO_PREFIX}{s}.jsonl' for s in empty]}", flush=True)
    if not any(sources.values()):
        print(f"no non-empty {AUTO_PREFIX}*.jsonl under {labels_root} for {datasets}", flush=True)
        return 1
    try:
        deployment = resolve_deployment_sources(sources, args.deployment_source)
    except ValueError as err:
        print(err, flush=True)
        return 1
    if deployment["note"]:
        print(f"NOTE: {deployment['note']}", flush=True)
    if deployment["ignored"]:
        print(f"NOTE: --deployment-source ignored for datasets not scored: {deployment['ignored']}", flush=True)
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
    print(f"labels: {labels_root}; questions: the {len(specs)} drawn from {draw_from} ({', '.join(datasets)}); "
          f"sources: {sources}; pooled: {deployment['sources']}; accepted ids: "
          f"{args.accepted or 'none given (all-items half only)'}", flush=True)
    certification = None if accepted is None else certification_record(accepted, sources, specs, labels_root)
    if certification is not None:
        if certification["note"]:
            print(f"\n{certification['note']}\n", flush=True)
        for row in certification["dropped"]:
            print(f"accepted subset: left out {row['dataset']}/{row['source']} {row['answer_kind']} "
                  f"(certified_ok={row['certified_ok']}): {row['n_scored']} scored questions, "
                  f"{row['n_listed']} listed ids", flush=True)
    parts: dict[str, list[pd.DataFrame]] = {}
    units_info: dict[str, Any] = {}
    scored_dir = args.out_dir / "scored" / labels_root.name
    scored_dir.mkdir(parents=True, exist_ok=True)
    for unit in scoring_units(sources, deployment["sources"]):
        labels, available = unit_labels(unit, specs, labels_root, accepted)
        scored, tables, info = analyse_unit(unit, labels, predictions, available, args.n_boot, args.boot_seed)
        scored.to_csv(scored_dir / f"{unit.name.replace('/', '__')}.csv", index=False)
        units_info[unit.name] = info
        for name, table in tables.items():
            parts.setdefault(name, []).append(table)
    tables = {name: pd.concat([t for t in frames if len(t)], ignore_index=True).infer_objects()
              if any(len(t) for t in frames) else pd.DataFrame() for name, frames in parts.items()}
    missed_scored, missed_table, missed_info = missed_analysis(
        draw, specs, all_specs, datasets, labels_root, args.out_dir, predictions, args.include_partial,
        args.n_boot, args.boot_seed)
    if len(missed_scored):
        missed_scored.to_csv(scored_dir / "oracle_only__missed_vs_answerable.csv", index=False)
    runtime = runtime_table(raw)
    composition = sample_composition(specs)
    runs_path = args.out_dir / "runs.jsonl"
    runs = [json.loads(line) for line in runs_path.read_text().splitlines() if line.strip()] if runs_path.exists() \
        else []
    summary = {
        "created": utc_now(), "git_commit": git_commit(), "labels_dir": portable(labels_root),
        "questions_drawn_from": portable(draw_from), "datasets": datasets, "n_questions": len(specs),
        "n_sample_total": len(all_specs),
        "auto_sources": {"found": {d: list(s) for d, s in sources.items()}, "empty": empty_sources},
        "design": {"sample_seed": sample["seed"], "per_cell": draw["per_cell"],
                   "deployment_source": sample["deployment_source"], "pooled_deployment_source": deployment,
                   "n_boot": args.n_boot,
                   "boot_seed": args.boot_seed, "ci_level": 0.95, "model_pairs": [list(p) for p in MODEL_PAIRS],
                   "forced_policies": {"counted": "answers forced after cut-off reasoning count as answers",
                                       "strict": "forced answers count as parse failures (wrong)"},
                   "resolution": DEFAULT_RESOLUTION.name,
                   "max_new_tokens": {k: m.max_new_tokens for k, m in MODELS.items()},
                   "accepted_ids": None if accepted is None else {"path": portable(args.accepted),
                                                                  **file_fingerprint(args.accepted)}},
        "accepted_certification": certification,
        "label_files": label_files, "label_files_changed_since_sampling": changed,
        "sample_draws": [{**{k: v for k, v in d.items() if k not in ("git_commit", "ids")},
                          "labels_dir": portable(d["labels_dir"]),
                          **({"missed": {k: v for k, v in d["missed"].items() if k != "items"}}
                             if isinstance(d.get("missed"), dict) else {})} for d in sample["draws"]],
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
        "missed": missed_info,
        "missed_vs_answerable": records(missed_table),
    }
    results_dir = args.results_dir or REPO_ROOT / "results" / "vlm" / labels_root.name
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, table in {**tables, "runtime": runtime, "sample_composition": composition,
                        "missed_vs_answerable": missed_table}.items():
        table.to_csv(results_dir / f"{name}.csv", index=False, float_format="%.4f")
    print_tables(tables, runtime)
    if len(missed_table):
        shown = missed_table[(missed_table["qtype"] == "all") & (missed_table["forced_policy"] == "counted")]
        print("\n== Oracle-referenced accuracy on missed vs answerable questions (all shared types, forced answers "
              "counted; scene-cluster bootstrap 95% CI)\n"
              + shown.drop(columns=["level", "n_boot", "seed", "forced_policy"]).round(3).to_string(index=False))
    elif missed_info.get("note"):
        print(f"\n== Missed vs answerable: {missed_info['note']}")
    print("\n== Units (notes and accepted questions left out as uncertified are in summary.json)\n" + pd.DataFrame([
        {"unit": k, **{f: v for f, v in info.items()
                       if f not in ("n_unanswered", "notes", "n_accepted_dropped_uncertified")},
         "n_accepted_dropped": (None if info["n_accepted_dropped_uncertified"] is None
                                else sum(info["n_accepted_dropped_uncertified"].values())),
         "max_unanswered": max(info["n_unanswered"].values(), default=0), "n_notes": len(info["notes"])}
        for k, info in units_info.items()]).round(3).to_string(index=False))
    if certification is not None and certification["note"]:
        print(f"\n{certification['note']}", flush=True)
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
        ``<data root>/outputs/vlm`` and ``<data root>/outputs/labels``. For ``score``, ``deployment_source`` is
        a dict of dataset to source; a malformed ``--deployment-source`` exits with a usage error.
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
    p.add_argument("--include-missed", action="store_true",
                   help="also draw questions the deployment source missed (oracle answer, no automatic answer "
                        "because a referent was not matched), stratified the same way, scored against the oracle")
    p = sub.add_parser("predict", help="answer the sample with one model (GPU)")
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--modes", nargs="+", choices=["image", "blind"], default=["image", "blind"])
    p.add_argument("--limit", type=int, default=None, help="new questions per mode (timing runs)")
    p = sub.add_parser("score", help="score predictions against one label folder and write aggregates (CPU)")
    p.add_argument("--labels-dir", type=Path, default=None, help="label folder (default <data>/outputs/labels)")
    p.add_argument("--draw-from", type=Path, default=None,
                   help="score the questions drawn from this label folder (default: --labels-dir)")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="score only these datasets of the draw (default: all of them), e.g. for a label folder "
                        "that covers fewer datasets")
    p.add_argument("--deployment-source", action="append", default=[], metavar="DATASET=SOURCE",
                   help="the automatic source a dataset contributes to the pooled deployment unit (repeatable; "
                        "default: the v0 source if the folder has it, else the folder's only source)")
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
    if args.command == "score":
        try:
            args.deployment_source = parse_deployment_sources(args.deployment_source)
        except ValueError as err:
            parser.error(str(err))
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
