"""Compare the 3D-gated and the 2D class-agnostic association of automatic labels with ground truth.

Two label folders written by ``scripts/run_labels.py`` from the *same* detections and questions are compared:
``--gated`` (3D centre distance within ``max(0.5 m, 0.5 * diagonal)``, label-compatible pairs only; the original
pipeline) and ``--primary`` (2D IoU of the detection's box with the object's visible box, class-agnostic, no 3D
gate; ``run_labels.py --rematch-from``). The point is to measure how much error the 3D gate hid: pairs it could
not form (far 3D errors, class confusions) become questions with an automatic answer, and their errors show up.

Per dataset, depth source and association:

- **pairs**: present oracle objects paired with a detection, recall (also for clearly visible objects and by
  camera range of the object's box centre), class-confusion rate (pairs whose detection phrase does not map to
  the object's label), ambiguous pairs, the 3D centre distance of pairs (median, p90) and the share of pairs
  farther than the 3D gate would have allowed;
- **answers**: share of questions with an automatic answer (coverage) and the error of answered items per
  question type: median and p90 of ``|auto - oracle|`` (metres for numeric types, objects for counts) and the
  wrong rate (:func:`measured_space.qa_schema.is_correct`). Subsets: ``all`` answered items, ``both`` (answered
  under both associations), ``new`` (answered only under the 2D association), ``lost`` (only under the 3D one),
  and for the 2D association ``confused`` / ``label_correct`` (some / no referent paired with a detection of
  another class);
- **sensitivity** of the 2D pair count to the IoU threshold (0.2, 0.3, 0.5 by default), recomputed from the
  stored detections and visible boxes with :func:`measured_space.matching.match_detections_2d` (the run's own
  threshold must reproduce its stored pairs exactly, which is checked).

Writes aggregates only (no per-item rows) to ``results/association/`` (``--results``): ``summary.json`` and the
CSVs ``comparison.csv`` (one row per dataset, depth source and association), ``errors.csv`` (per question type
and subset), ``sensitivity.csv`` and ``confusions.csv`` (detection phrase -> ground-truth label counts).

Usage (from the repository root)::

    python scripts/compare_association.py
    python scripts/compare_association.py --gated <labels folder> --primary <labels_2d folder>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from measured_space.config import REPO_ROOT, data_root
from measured_space.matching import Match2DConfig, MatchConfig, match_detections_2d
from measured_space.pipeline import box_from_json, visible_box_edges
from measured_space.qa_schema import QAItem, is_correct
from measured_space.scene import SceneObject

DATASETS = ("arkitscenes", "smartspaces")
ASSOCIATIONS = ("3d", "2d")
RANGE_BINS_M: tuple[float, ...] = (0.0, 10.0, 20.0, 45.0, float("inf"))
"""tuple of float: edges of the camera-range bins (m) for recall by range."""

COMPARISON_COLUMNS = (
    "dataset", "depth_source", "association", "n_views", "n_present", "n_clearly_visible", "n_pairs", "recall",
    "recall_clearly_visible", "n_confused", "confusion_rate", "confusion_rate_clearly_visible",
    "confused_share_beyond_3d_gate", "n_ambiguous",
    "distance_median_m", "distance_p90_m", "share_beyond_3d_gate", "n_questions", "n_answered", "coverage",
    "wrong_rate", "recall_range_0_10m", "recall_range_10_20m", "recall_range_20_45m", "recall_range_45m_plus",
    "n_present_range_0_10m", "n_present_range_10_20m", "n_present_range_20_45m", "n_present_range_45m_plus")
"""Columns of ``comparison.csv``."""

ERROR_COLUMNS = ("dataset", "depth_source", "association", "subset", "qtype", "n_questions", "n_answered",
                 "coverage", "abs_error_median", "abs_error_p90", "error_unit", "wrong_rate", "n_same_answer")
"""Columns of ``errors.csv``."""

SENSITIVITY_COLUMNS = ("dataset", "depth_source", "min_iou", "n_present", "n_pairs", "recall", "n_confused",
                       "confusion_rate", "n_ambiguous")
"""Columns of ``sensitivity.csv``."""

CONFUSION_COLUMNS = ("dataset", "depth_source", "phrase", "label", "n_pairs", "share_of_pairs")
"""Columns of ``confusions.csv``."""


# --------------------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The parsed options.
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--gated", type=Path, default=None,
                   help="3D-gated label folder (default <data root>/outputs/labels)")
    p.add_argument("--primary", type=Path, default=None,
                   help="2D-association label folder (default <data root>/outputs/labels_2d)")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), help="datasets to compare")
    p.add_argument("--results", type=Path, default=REPO_ROOT / "results" / "association",
                   help="aggregate output folder (default results/association)")
    p.add_argument("--sensitivity", type=float, nargs="+", default=[0.2, 0.3, 0.5],
                   help="2D IoU thresholds for the pair-count sensitivity")
    return p.parse_args(argv)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read a JSON-lines file.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Yields
    ------
    dict
        One record per non-empty line.
    """
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    """Checksum of a file.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    str
        Hex SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Median and 90th percentile.

    Parameters
    ----------
    values : sequence of float
        Values.

    Returns
    -------
    tuple of float or None
        ``(median, p90)`` rounded to 4 decimals, ``(None, None)`` when empty.
    """
    if not values:
        return None, None
    a = np.asarray(values, dtype=np.float64)
    return round(float(np.median(a)), 4), round(float(np.percentile(a, 90)), 4)


def _rate(numerator: float, denominator: float) -> float | None:
    """A ratio, or ``None`` for an empty denominator.

    Parameters
    ----------
    numerator, denominator : float
        Counts.

    Returns
    -------
    float or None
        ``numerator / denominator`` rounded to 4 decimals.
    """
    return round(numerator / denominator, 4) if denominator else None


# --------------------------------------------------------------------------------------------------------
# Pairs
# --------------------------------------------------------------------------------------------------------


def _range_bin(distance: float) -> str:
    """The name of the camera-range bin a distance falls in.

    Parameters
    ----------
    distance : float
        Camera-to-object distance, metres.

    Returns
    -------
    str
        ``"0_10m"``, ``"10_20m"``, ``"20_45m"`` or ``"45m_plus"`` for the default bins.
    """
    for lo, hi in zip(RANGE_BINS_M[:-1], RANGE_BINS_M[1:]):
        if lo <= distance < hi:
            return f"{lo:g}_{hi:g}m" if np.isfinite(hi) else f"{lo:g}m_plus"
    return f"{RANGE_BINS_M[-2]:g}m_plus"


def pair_stats(oracle: Sequence[Mapping[str, Any]], views: Mapping[str, Mapping[str, Any]], source: str,
               matches: Mapping[tuple[str, str], Mapping[str, Any] | None],
               phrases: Mapping[tuple[str, int], str]) -> tuple[dict[str, Any], Counter]:
    """Pair statistics of one association and depth source.

    Parameters
    ----------
    oracle : sequence of mapping
        ``oracle_objects.jsonl`` records (any association; only identity, presence, label and box are used).
    views : mapping of str to mapping
        ``view_key`` -> ``views.jsonl`` record.
    source : str
        Depth source.
    matches : mapping of tuple to mapping or None
        ``(view_key, object_id)`` -> the association's match record for ``source`` (``None`` if unpaired).
    phrases : mapping of tuple to str
        ``(view_key, detection index)`` -> detection phrase.

    Returns
    -------
    stats : dict
        The pair columns of :data:`COMPARISON_COLUMNS` (without identity and answer columns).
    confusions : collections.Counter
        ``(phrase, label)`` -> number of confused pairs.
    """
    gate = MatchConfig()
    present = [r for r in oracle if r["present"] and source in views[r["view_key"]]["depth_sources"]]
    clear = [r for r in present if r["clearly_visible"]]
    paired = [(r, matches[(r["view_key"], r["object_id"])]) for r in present
              if matches.get((r["view_key"], r["object_id"])) is not None]
    confused = [(r, m) for r, m in paired if not m.get("label_correct", True)]
    clear_paired = [(r, m) for r, m in paired if r["clearly_visible"]]
    distances = [float(m["distance"]) for _, m in paired]
    beyond = {(r["view_key"], r["object_id"]) for r, m in paired
              if float(m["distance"]) > gate.gate(box_from_json(r["box"]))}
    stats: dict[str, Any] = {
        "n_views": sum(source in v["depth_sources"] for v in views.values()),
        "n_present": len(present), "n_clearly_visible": len(clear), "n_pairs": len(paired),
        "recall": _rate(len(paired), len(present)), "recall_clearly_visible": _rate(len(clear_paired), len(clear)),
        "n_confused": len(confused), "confusion_rate": _rate(len(confused), len(paired)),
        "confusion_rate_clearly_visible": _rate(sum(not m.get("label_correct", True) for _, m in clear_paired),
                                                len(clear_paired)),
        "confused_share_beyond_3d_gate": _rate(sum((r["view_key"], r["object_id"]) in beyond for r, _ in confused),
                                               len(confused)),
        "n_ambiguous": sum(bool(m.get("ambiguous")) for _, m in paired),
        "distance_median_m": _quantiles(distances)[0], "distance_p90_m": _quantiles(distances)[1],
        "share_beyond_3d_gate": _rate(len(beyond), len(paired))}
    in_bin: Counter = Counter()
    hit_bin: Counter = Counter()
    for r in present:
        camera = np.asarray(views[r["view_key"]]["T_wc"], dtype=np.float64)[:3, 3]
        name = _range_bin(float(np.linalg.norm(np.asarray(r["box"]["center"]) - camera)))
        in_bin[name] += 1
        hit_bin[name] += matches.get((r["view_key"], r["object_id"])) is not None
    for lo, hi in zip(RANGE_BINS_M[:-1], RANGE_BINS_M[1:]):
        name = _range_bin(lo)
        stats[f"recall_range_{name}"] = _rate(hit_bin[name], in_bin[name])
        stats[f"n_present_range_{name}"] = in_bin[name]
    counts = Counter((phrases[(r["view_key"], m["detection"])], r["label"]) for r, m in confused)
    return stats, counts


def sensitivity(oracle: Sequence[Mapping[str, Any]], detections: Sequence[Mapping[str, Any]],
                views: Mapping[str, Mapping[str, Any]], thresholds: Sequence[float]
                ) -> dict[tuple[str, float], dict[str, Any]]:
    """2D pair counts at several IoU thresholds, recomputed from stored boxes.

    Parameters
    ----------
    oracle : sequence of mapping
        ``oracle_objects.jsonl`` records.
    detections : sequence of mapping
        ``detections.jsonl`` records.
    views : mapping of str to mapping
        ``view_key`` -> ``views.jsonl`` record (its ``visibility`` sets the visible-box convention).
    thresholds : sequence of float
        IoU thresholds.

    Returns
    -------
    dict of tuple to dict
        ``(depth source, threshold)`` -> ``n_present``, ``n_pairs``, ``n_confused``, ``n_ambiguous`` and
        ``pairs`` (the set of ``(view_key, depth source, detection, object_id)``, for checking against the
        stored run).
    """
    present_by_view: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in oracle:
        if r["present"]:
            present_by_view[r["view_key"]].append(r)
    dets_by_view: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in detections:
        dets_by_view[(r["view_key"], r["source"])].append(r)
    out: dict[tuple[str, float], dict[str, Any]] = {}
    for key, source in sorted((k, s) for k, v in views.items() for s in v["depth_sources"]):
        dets = sorted(dets_by_view.get((key, source), []), key=lambda r: r["detection"])
        present = present_by_view.get(key, [])
        objects = [SceneObject(r["object_id"], r["label"], box_from_json(r["box"])) for r in present]
        edges = [visible_box_edges(r["bbox2d"], views[key]["visibility"]) for r in present]
        boxes = [None if r["box"] is None else box_from_json(r["box"]) for r in dets]
        for t in thresholds:
            result = match_detections_2d([r["box2d"] for r in dets], boxes, [tuple(r["labels"]) for r in dets],
                                         objects, edges, Match2DConfig(min_iou=t), up=views[key]["up"])
            entry = out.setdefault((source, t), {"n_present": 0, "n_pairs": 0, "n_confused": 0, "n_ambiguous": 0,
                                                 "pairs": set()})
            entry["n_present"] += len(present)
            entry["n_pairs"] += len(result.matches)
            entry["n_confused"] += sum(not m.label_correct for m in result.matches)
            entry["n_ambiguous"] += sum(m.ambiguous for m in result.matches)
            entry["pairs"] |= {(key, source, dets[m.detection]["detection"], m.object_id) for m in result.matches}
    return out


# --------------------------------------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------------------------------------


def item_error(item: QAItem, reference: QAItem) -> tuple[float | None, bool]:
    """Error of an automatic answer against the oracle answer.

    Parameters
    ----------
    item : QAItem
        The automatic item.
    reference : QAItem
        The oracle item of the same question.

    Returns
    -------
    abs_error : float or None
        ``|auto - oracle|``: metres for numeric questions, objects for counts, ``None`` for choices.
    wrong : bool
        ``not is_correct(spec, auto, oracle)``.
    """
    wrong = not is_correct(item.spec, item.answer, reference.answer)
    if item.spec.answer_type in ("number", "count"):
        return abs(float(item.answer) - float(reference.answer)), wrong
    return None, wrong


def confused_item(item: QAItem) -> bool:
    """Whether some referent of an automatic item was paired with a detection of another class.

    Parameters
    ----------
    item : QAItem
        An automatic item (counts have no referents and are never confused).

    Returns
    -------
    bool
        True if a referent's evidence has ``label_correct`` False.
    """
    return any(r.get("label_correct") is False for r in item.extra.get("referents", []))


def error_rows(dataset: str, source: str, oracle: Mapping[str, QAItem], covered: set[str],
               auto: Mapping[str, Mapping[str, QAItem]]) -> list[dict[str, Any]]:
    """Coverage and error rows of one dataset and depth source, per association, subset and question type.

    Parameters
    ----------
    dataset : str
        Dataset name.
    source : str
        Depth source.
    oracle : mapping of str to QAItem
        Question id -> oracle item.
    covered : set of str
        Ids of the questions about views the depth source was run on.
    auto : mapping of str to mapping of str to QAItem
        Association (``"3d"``, ``"2d"``) -> question id -> automatic item.

    Returns
    -------
    list of dict
        Rows keyed by :data:`ERROR_COLUMNS`.
    """
    both = set(auto["3d"]) & set(auto["2d"])
    subsets: dict[str, dict[str, set[str]]] = {
        "3d": {"all": set(auto["3d"]), "both": both, "lost": set(auto["3d"]) - set(auto["2d"])},
        "2d": {"all": set(auto["2d"]), "both": both, "new": set(auto["2d"]) - set(auto["3d"]),
               "confused": {i for i, it in auto["2d"].items() if confused_item(it)},
               "label_correct": {i for i, it in auto["2d"].items() if not confused_item(it)}}}
    qtypes = sorted({oracle[i].spec.qtype for i in covered})
    rows = []
    for association, sets in subsets.items():
        for subset, ids in sets.items():
            for qtype in [*qtypes, "all"]:
                questions = [i for i in covered if qtype in ("all", oracle[i].spec.qtype)]
                answered = [i for i in questions if i in ids]
                errors = [item_error(auto[association][i], oracle[i]) for i in answered]
                numeric = [e for e, _ in errors if e is not None]
                kinds = {oracle[i].spec.answer_type for i in answered}
                unit = "m" if kinds == {"number"} else "objects" if kinds == {"count"} else ""
                same = sum(auto["3d"][i].answer == auto["2d"][i].answer for i in answered) if subset == "both" \
                    else None
                median, p90 = _quantiles(numeric) if unit else (None, None)
                rows.append({"dataset": dataset, "depth_source": source, "association": association,
                             "subset": subset, "qtype": qtype, "n_questions": len(questions),
                             "n_answered": len(answered), "coverage": _rate(len(answered), len(questions)),
                             "abs_error_median": median, "abs_error_p90": p90, "error_unit": unit,
                             "wrong_rate": _rate(sum(w for _, w in errors), len(errors)), "n_same_answer": same})
    return rows


# --------------------------------------------------------------------------------------------------------
# One dataset
# --------------------------------------------------------------------------------------------------------


def _matches(oracle: Sequence[Mapping[str, Any]], source: str) -> dict[tuple[str, str], Mapping[str, Any] | None]:
    """Per-object match records of one depth source.

    Parameters
    ----------
    oracle : sequence of mapping
        ``oracle_objects.jsonl`` records of one association.
    source : str
        Depth source.

    Returns
    -------
    dict of tuple to mapping or None
        ``(view_key, object_id)`` -> match record (``None`` if unpaired), for present objects only.
    """
    return {(r["view_key"], r["object_id"]): r["match"].get(source) for r in oracle
            if r["present"] and r["match"] is not None and source in r["match"]}


def compare_dataset(dataset: str, folders: Mapping[str, Path], thresholds: Sequence[float],
                    primary_min_iou: float) -> dict[str, Any]:
    """Compare the two associations on one dataset.

    Parameters
    ----------
    dataset : str
        Dataset name.
    folders : mapping of str to pathlib.Path
        Association (``"3d"``, ``"2d"``) -> label root.
    thresholds : sequence of float
        IoU thresholds of the sensitivity analysis.
    primary_min_iou : float
        The 2D run's own threshold (its stored pairs must be reproduced at it).

    Returns
    -------
    dict
        ``comparison``, ``errors``, ``sensitivity`` and ``confusions`` rows, and ``checks``.

    Raises
    ------
    ValueError
        If the two folders hold different questions or views, or the sensitivity analysis does not reproduce the
        stored 2D pairs at ``primary_min_iou``.
    """
    paths = {a: folders[a] / dataset for a in ASSOCIATIONS}
    oracle_sha = {a: sha256(p / "qa_oracle.jsonl") for a, p in paths.items()}
    if len(set(oracle_sha.values())) != 1:
        raise ValueError(f"{dataset}: the two folders hold different oracle questions")
    views = {r["view_key"]: r for r in read_jsonl(paths["2d"] / "views.jsonl")}
    if set(views) != {r["view_key"] for r in read_jsonl(paths["3d"] / "views.jsonl")}:
        raise ValueError(f"{dataset}: the two folders hold different views")
    oracle_objects = {a: list(read_jsonl(p / "oracle_objects.jsonl")) for a, p in paths.items()}
    detections = list(read_jsonl(paths["2d"] / "detections.jsonl"))
    phrases = {(r["view_key"], r["detection"]): r["phrase"] for r in detections}
    oracle_items = {i.id: i for i in (QAItem.from_json(r) for r in read_jsonl(paths["2d"] / "qa_oracle.jsonl"))}
    sources = sorted({s for v in views.values() for s in v["depth_sources"]})
    view_of = {i: f"{dataset}/{it.spec.scene}/{it.spec.camera_id}/{it.spec.frame}" for i, it in oracle_items.items()}

    comparison, errors, confusions = [], [], []
    for source in sources:
        covered = {i for i, key in view_of.items() if source in views[key]["depth_sources"]}
        auto = {a: {i.id: i for i in (QAItem.from_json(r) for r in read_jsonl(p / f"qa_auto_{source}.jsonl"))}
                for a, p in paths.items()}
        rows = error_rows(dataset, source, oracle_items, covered, auto)
        errors += rows
        for association in ASSOCIATIONS:
            stats, counts = pair_stats(oracle_objects[association], views, source,
                                       _matches(oracle_objects[association], source), phrases)
            total = next(r for r in rows if r["association"] == association and r["subset"] == "all"
                         and r["qtype"] == "all")
            comparison.append({"dataset": dataset, "depth_source": source, "association": association, **stats,
                               "n_questions": total["n_questions"], "n_answered": total["n_answered"],
                               "coverage": total["coverage"], "wrong_rate": total["wrong_rate"]})
            if association == "2d":
                confusions += [{"dataset": dataset, "depth_source": source, "phrase": phrase, "label": label,
                                "n_pairs": n, "share_of_pairs": _rate(n, stats["n_pairs"])}
                               for (phrase, label), n in counts.most_common()]

    sens = sensitivity(oracle_objects["2d"], detections, views, sorted(set(thresholds) | {primary_min_iou}))
    stored_pairs = {(r["view_key"], r["source"], r["detection"], r["match"]["object_id"]) for r in detections
                    if r["match"] is not None}
    reproduced = set().union(*(e["pairs"] for (_, t), e in sens.items() if t == primary_min_iou))
    if reproduced != stored_pairs:
        raise ValueError(f"{dataset}: at IoU {primary_min_iou} the recomputed pairs ({len(reproduced)}) differ from "
                         f"the stored 2D pairs ({len(stored_pairs)})")
    sensitivity_rows = [{"dataset": dataset, "depth_source": s, "min_iou": t, "n_present": e["n_present"],
                         "n_pairs": e["n_pairs"], "recall": _rate(e["n_pairs"], e["n_present"]),
                         "n_confused": e["n_confused"], "confusion_rate": _rate(e["n_confused"], e["n_pairs"]),
                         "n_ambiguous": e["n_ambiguous"]}
                        for (s, t), e in sorted(sens.items()) if t in thresholds]
    checks = {"oracle_questions_identical": True, "qa_oracle_sha256": oracle_sha["2d"],
              "sensitivity_reproduces_stored_pairs_at": primary_min_iou,
              "n_questions": len(oracle_items), "n_views": len(views)}
    return {"comparison": comparison, "errors": errors, "sensitivity": sensitivity_rows, "confusions": confusions,
            "checks": checks}


# --------------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------------


def write_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], path: Path) -> None:
    """Write rows to a CSV file (missing values empty).

    Parameters
    ----------
    rows : sequence of mapping
        Rows keyed by ``columns``.
    columns : sequence of str
        Column order.
    path : pathlib.Path
        Output file.
    """
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in columns})


def headline(rows: Sequence[Mapping[str, Any]]) -> str:
    """The comparison as a fixed-width text table.

    Parameters
    ----------
    rows : sequence of mapping
        ``comparison.csv`` rows.

    Returns
    -------
    str
        One line per dataset, depth source and association.
    """
    columns = ("dataset", "depth_source", "association", "n_pairs", "recall", "recall_clearly_visible",
               "confusion_rate", "share_beyond_3d_gate", "distance_median_m", "distance_p90_m", "coverage",
               "wrong_rate")
    cells = [[("-" if r.get(c) is None else str(r.get(c))) for c in columns] for r in rows]
    widths = [max(len(c), *(len(x[i]) for x in cells)) for i, c in enumerate(columns)]
    return "\n".join(["  ".join(c.rjust(w) for c, w in zip(columns, widths))]
                     + ["  ".join(v.rjust(w) for v, w in zip(x, widths)) for x in cells])


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the associations on every dataset and write the aggregates.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status: 0 on success, 1 if a dataset's labels are missing in either folder.
    """
    args = parse_args(argv)
    started = time.perf_counter()
    folders = {"3d": args.gated or data_root() / "outputs" / "labels",
               "2d": args.primary or data_root() / "outputs" / "labels_2d"}
    missing = [f"{a}:{d}" for a, f in folders.items() for d in args.datasets if not (f / d / "summary.json").exists()]
    if missing:
        print(f"labels missing for {missing}", file=sys.stderr)
        return 1
    configs = {a: json.loads((f / args.datasets[0] / "summary.json").read_text())["config"]
               for a, f in folders.items()}
    if configs["3d"].get("association", "3d") != "3d" or configs["2d"].get("association") != "2d":
        print("--gated must hold 3D-gated labels and --primary 2D-associated ones", file=sys.stderr)
        return 1
    min_iou = float(configs["2d"]["match_2d"]["min_iou"])
    parts = {d: compare_dataset(d, folders, args.sensitivity, min_iou) for d in args.datasets}
    tables = {name: [r for p in parts.values() for r in p[name]]
              for name in ("comparison", "errors", "sensitivity", "confusions")}
    args.results.mkdir(parents=True, exist_ok=True)
    write_csv(tables["comparison"], COMPARISON_COLUMNS, args.results / "comparison.csv")
    write_csv(tables["errors"], ERROR_COLUMNS, args.results / "errors.csv")
    write_csv(tables["sensitivity"], SENSITIVITY_COLUMNS, args.results / "sensitivity.csv")
    write_csv(tables["confusions"], CONFUSION_COLUMNS, args.results / "confusions.csv")
    summary = {
        "associations": {
            "3d": {"rule": "Hungarian on 3D centre distance, label-compatible pairs only, gate "
                           "max(min_gate, gate_diagonal_fraction * diagonal)", "settings": configs["3d"]["match"],
                   "folder": folders["3d"].name},
            "2d": {"rule": "Hungarian maximising summed 2D IoU (detection box vs visible ground-truth box) over pairs "
                           "with IoU >= min_iou; class-agnostic; no 3D gate",
                   "settings": configs["2d"]["match_2d"], "folder": folders["2d"].name}},
        "definitions": {
            "recall": "paired present objects / present objects (views run with the depth source)",
            "confusion_rate": "pairs whose detection phrase does not map to the object's label / pairs",
            "share_beyond_3d_gate": "pairs whose 3D centre distance exceeds max(0.5 m, 0.5 * box diagonal)",
            "confused_share_beyond_3d_gate": "confused pairs beyond that gate; with reference depth (laser, rendered) "
                                             "a confused pair within it is the right object misnamed, one beyond it "
                                             "a candidate wrong-object pairing",
            "range": "distance from the camera centre to the ground-truth box centre (m)",
            "coverage": "questions with an automatic answer / questions about views run with the depth source",
            "abs_error": "|auto - oracle|: metres for numeric question types, objects for counts; none for choices",
            "wrong_rate": "share of answered items that is_correct rejects",
            "subsets": "all: answered; both: answered under both associations; new: only under 2D; lost: only "
                       "under 3D; confused / label_correct: 2D items with some / no referent paired with a "
                       "detection of another class",
            "n_same_answer": "items of the 'both' subset whose automatic answer is identical under the two "
                             "associations"},
        "checks": {d: p["checks"] for d, p in parts.items()},
        **tables,
        "seconds": round(time.perf_counter() - started, 1)}
    (args.results / "summary.json").write_text(json.dumps(summary, indent=1))
    print(headline(tables["comparison"]))
    print(f"\nwrote {args.results} ({summary['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
