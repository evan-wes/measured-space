"""Check how the definition of "clearly visible" moves the recall populations, and inspect one object's visibility.

The pipeline's visible fraction (:func:`measured_space.visibility.object_in_view`) is the share of an object's
camera-facing surface samples that lie in the image *and* are not hidden. Its denominator includes the samples
outside the image, and samples on pixels without laser depth count as hidden. ``recall`` measures what two
alternatives would change, from the stored label records alone (no models, nothing rerun):

- ``current``: visible >= 0.3 and truncated <= 0.3 (:func:`measured_space.pipeline.clearly_visible`);
- ``in_frame``: visible / in-frame >= 0.3 and truncated <= 0.3 (depth holes still count as hidden);
- ``known_depth``: visible / (in-frame samples with laser depth) >= 0.3 and truncated <= 0.3, with at least
  ``--min-known`` such samples, else *undetermined* and left out.

All three are fractions of one denominator (the camera-facing samples), so they follow from the stored
``visible_fraction``, ``truncated_fraction`` and ``diagnostics.no_depth_fraction``. "Present" (at least one
visible sample), the visible 2D boxes and therefore the pairs are the same under all three; only the clearly
visible population changes.

Subcommands:

- ``recall``: population sizes and each labeller's recall on clearly visible objects under each definition, and
  the visibility make-up of the present objects that lie fully inside the image.
- ``inspect``: one ARKitScenes object in one view: its camera-facing samples per box face (in frame, visible,
  hidden, no laser depth), how far its no-depth samples lie from valid depth, the hole structure of the laser
  depth, and optionally a picture of the samples. Its counts are checked against ``object_in_view``.

Run from the repository root (CPU only)::

    python scripts/check_visibility_definition.py recall --labels <data>/outputs/labels_ark106_2d \\
        --labels <data>/outputs/labels_ark106_cutr_sam3 --labels <data>/outputs/labels_ark106_cutr_clip
    python scripts/check_visibility_definition.py inspect --video 47333452 --frame 54983.193 \\
        --object 4B0IzVsS3NyLwJ5h --draw fridge.png

Nothing is written unless ``--out`` (recall) or ``--draw`` (inspect) is given; neither overwrites a file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from measured_space.geometry import Array
from measured_space.qa import QAConfig
from measured_space.scene import FrameView, SceneObject
from measured_space.viewer import wilson_interval
from measured_space.visibility import depth_tolerance, object_in_view

DEFINITIONS: tuple[str, ...] = ("current", "in_frame", "known_depth")
"""tuple of str: the clearly-visible definitions compared by ``recall``."""

SAMPLE_STATES: tuple[str, ...] = ("outside", "visible", "hidden", "no_depth")
"""tuple of str: the state of a camera-facing sample: outside the image (or too near the camera plane),
visible, hidden behind something nearer, or on a pixel without valid depth."""

SMALL_HOLE_PX = 4
"""int: holes of at most this many pixels count as small (isolated dropouts) in :func:`hole_stats`."""


# --------------------------------------------------------------------------------------------------------
# recall
# --------------------------------------------------------------------------------------------------------


def is_present(record: dict[str, Any], config: QAConfig) -> bool:
    """Whether a stored oracle object is present in its view (:func:`measured_space.qa.visible_ids`).

    Parameters
    ----------
    record : dict
        One line of ``oracle_objects.jsonl``.
    config : QAConfig
        Supplies ``presence_threshold``.

    Returns
    -------
    bool
        True if it has a visible 2D box and its visible fraction exceeds the presence threshold.
    """
    return record["bbox2d"] is not None and record["visible_fraction"] > config.presence_threshold


def membership(record: dict[str, Any], config: QAConfig, min_known: int) -> dict[str, bool | None]:
    """Whether a present object is clearly visible under each definition in :data:`DEFINITIONS`.

    Parameters
    ----------
    record : dict
        One line of ``oracle_objects.jsonl`` with depth-based visibility (``diagnostics`` holds ``n_samples``
        and ``no_depth_fraction``).
    config : QAConfig
        Supplies ``min_visible_fraction`` and ``max_truncated_fraction``.
    min_known : int
        ``known_depth`` needs at least this many in-frame samples with depth; with fewer it is undetermined.

    Returns
    -------
    dict of str to bool or None
        One entry per definition; ``None`` means undetermined (``known_depth`` only).

    Raises
    ------
    ValueError
        If the record's visibility was not measured against depth.
    """
    diagnostics = record.get("diagnostics") or {}
    if "no_depth_fraction" not in diagnostics or "n_samples" not in diagnostics:
        raise ValueError(f"{record.get('object_id')}: visibility {record.get('visibility')!r} has no depth "
                         "diagnostics; only depth-based visibility can be checked")
    visible = float(record["visible_fraction"])
    truncated = float(record["truncated_fraction"])
    no_depth = float(diagnostics["no_depth_fraction"])
    n_samples = float(diagnostics["n_samples"])
    in_frame = 1.0 - truncated
    known = in_frame - no_depth
    cut_ok = truncated <= config.max_truncated_fraction
    threshold = config.min_visible_fraction
    out: dict[str, bool | None] = {
        "current": record["bbox2d"] is not None and visible >= threshold and cut_ok,
        "in_frame": in_frame > 0 and visible / in_frame >= threshold and cut_ok,
    }
    n_known = int(round(known * n_samples))
    out["known_depth"] = None if n_known < max(min_known, 1) else bool(visible / known >= threshold and cut_ok)
    return out


def read_oracle(folder: Path, dataset: str) -> list[dict[str, Any]]:
    """Read a label folder's stored oracle objects.

    Parameters
    ----------
    folder : pathlib.Path
        A label folder (``scripts/run_labels.py`` or ``scripts/run_box_source.py``).
    dataset : str
        The dataset subfolder.

    Returns
    -------
    list of dict
        The lines of ``<folder>/<dataset>/oracle_objects.jsonl``.
    """
    with open(folder / dataset / "oracle_objects.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stored_sources(records: Sequence[dict[str, Any]]) -> list[str]:
    """The sources with stored pairs, in first-seen order.

    Parameters
    ----------
    records : sequence of dict
        Oracle object lines.

    Returns
    -------
    list of str
        Keys of the records' ``match`` mappings.
    """
    seen: list[str] = []
    for r in records:
        seen.extend(s for s in (r.get("match") or {}) if s not in seen)
    return seen


def recall_rows(records: Sequence[dict[str, Any]], labeller: str, config: QAConfig, min_known: int
                ) -> list[dict[str, Any]]:
    """Recall per stored source on the present objects and on each clearly-visible population.

    Parameters
    ----------
    records : sequence of dict
        Oracle object lines of one label folder.
    labeller : str
        Name written in the rows.
    config : QAConfig
        Visibility thresholds.
    min_known : int
        See :func:`membership`.

    Returns
    -------
    list of dict
        Rows with ``labeller, source, population, n, paired, recall, wilson_lo, wilson_hi``; ``population`` is
        ``present`` or ``clearly_visible:<definition>``.
    """
    present = [r for r in records if is_present(r, config)]
    member = [membership(r, config, min_known) for r in present]
    rows: list[dict[str, Any]] = []
    for source in stored_sources(present):
        paired = [bool((r.get("match") or {}).get(source)) for r in present]
        groups = [("present", paired)]
        groups += [(f"clearly_visible:{d}", [p for p, m in zip(paired, member) if m[d] is True])
                   for d in DEFINITIONS]
        for population, flags in groups:
            k, n = sum(flags), len(flags)
            lo_hi = wilson_interval(k, n)
            rows.append({"labeller": labeller, "source": source, "population": population, "n": n, "paired": k,
                         "recall": k / n if n else float("nan"),
                         "wilson_lo": lo_hi[0] if lo_hi else float("nan"),
                         "wilson_hi": lo_hi[1] if lo_hi else float("nan")})
    return rows


def population_summary(records: Sequence[dict[str, Any]], config: QAConfig, min_known: int) -> dict[str, Any]:
    """Population sizes under each definition, and how they differ from ``current``.

    Parameters
    ----------
    records : sequence of dict
        Oracle object lines.
    config : QAConfig
        Visibility thresholds.
    min_known : int
        See :func:`membership`.

    Returns
    -------
    dict
        ``n_present`` and, per definition, ``clearly_visible``, ``undetermined``, ``gained`` and ``lost`` (objects
        clearly visible under it and not under ``current``, and the reverse).
    """
    present = [r for r in records if is_present(r, config)]
    member = [membership(r, config, min_known) for r in present]
    out: dict[str, Any] = {"n_present": len(present)}
    for d in DEFINITIONS:
        out[d] = {"clearly_visible": sum(m[d] is True for m in member),
                  "undetermined": sum(m[d] is None for m in member),
                  "gained": sum(m[d] is True and not m["current"] for m in member),
                  "lost": sum(bool(m["current"]) and m[d] is not True for m in member)}
    return out


def in_frame_makeup(records: Sequence[dict[str, Any]], config: QAConfig) -> dict[str, float]:
    """Visibility make-up of the present objects that lie fully inside the image (truncated fraction 0).

    Parameters
    ----------
    records : sequence of dict
        Oracle object lines with depth diagnostics.
    config : QAConfig
        Supplies the presence threshold.

    Returns
    -------
    dict of str to float
        ``n``; the ``median`` and ``mean`` visible fraction; the shares with visible fraction >= 0.9 and >= 0.99;
        the mean shares of samples on pixels without depth (``mean_no_depth``) and behind something nearer
        (``mean_hidden``).
    """
    full = [r for r in records if is_present(r, config) and r["truncated_fraction"] == 0]
    if not full:
        return {"n": 0.0}
    visible = np.array([r["visible_fraction"] for r in full], dtype=np.float64)
    no_depth = np.array([r["diagnostics"]["no_depth_fraction"] for r in full], dtype=np.float64)
    return {"n": float(len(full)), "median": float(np.median(visible)), "mean": float(visible.mean()),
            "share_ge_0.9": float((visible >= 0.9).mean()), "share_ge_0.99": float((visible >= 0.99).mean()),
            "mean_no_depth": float(no_depth.mean()), "mean_hidden": float((1.0 - visible - no_depth).mean())}


def run_recall(args: argparse.Namespace) -> int:
    """The ``recall`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    if args.out is not None and args.out.exists():
        print(f"refusing to overwrite {args.out}", file=sys.stderr)
        return 2
    config = QAConfig()
    rows: list[dict[str, Any]] = []
    for i, folder in enumerate(args.labels):
        records = read_oracle(folder, args.dataset)
        if i == 0:
            summary = population_summary(records, config, args.min_known)
            print(f"{folder.name}: {summary['n_present']} present objects (min known samples {args.min_known})")
            for d in DEFINITIONS:
                s = summary[d]
                print(f"  {d:12s} clearly visible {s['clearly_visible']:5d}  undetermined {s['undetermined']:4d}  "
                      f"gained {s['gained']:3d}  lost {s['lost']:3d}")
            makeup = in_frame_makeup(records, config)
            print("  fully in frame: " + ", ".join(f"{k} {v:.3f}" if k != "n" else f"n {int(v)}"
                                                   for k, v in makeup.items()))
        rows.extend(recall_rows(records, folder.name, config, args.min_known))
    print(f"\n{'labeller':28s} {'source':10s} {'population':26s} {'paired':>6s} {'n':>5s} {'recall':>7s}  wilson 95%")
    for r in rows:
        print(f"{r['labeller']:28s} {r['source']:10s} {r['population']:26s} {r['paired']:6d} {r['n']:5d} "
              f"{r['recall']:7.3f}  [{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}]")
    if args.out is not None:
        with open(args.out, "x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


# --------------------------------------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------------------------------------


def sample_states(view: FrameView, obj: SceneObject, depth: Array, *, spacing: float = 0.05, min_z: float = 0.05
                  ) -> tuple[Array, Array, NDArray[np.str_], NDArray[np.int64], NDArray[np.int64]]:
    """The state of each camera-facing surface sample, by the test of :func:`object_in_view`.

    Parameters
    ----------
    view : FrameView
        The view.
    obj : SceneObject
        The object.
    depth : numpy.ndarray, shape (H, W)
        Planar depth in metres (0 or NaN = invalid), possibly at another resolution than the camera.
    spacing : float, default 0.05
        Surface sampling grid spacing, metres.
    min_z : float, default 0.05
        Samples nearer than this to the camera plane count as outside, metres.

    Returns
    -------
    uv : numpy.ndarray, shape (N, 2)
        Projections of the camera-facing samples, pixels.
    face : numpy.ndarray, shape (N, 2)
        Per sample, the box axis (0, 1, 2) and the sign (+1, -1) of its face's outward normal.
    state : numpy.ndarray, shape (N,), str
        One of :data:`SAMPLE_STATES`.
    rows, cols : numpy.ndarray, shape (N,), int
        The depth pixel each sample reads (-1 for samples outside).
    """
    box_cam = obj.box.transformed(view.T_cw)
    points, normals = box_cam.sample_surface(spacing)
    facing = np.einsum("ij,ij->i", normals, -points) > 0
    points, normals = points[facing], normals[facing]
    in_box = normals @ box_cam.rotation                       # normals in the box's own axes
    axis = np.argmax(np.abs(in_box), axis=1)
    sign = np.sign(in_box[np.arange(len(in_box)), axis]).astype(np.int64)
    uv, z = view.camera.project(points)
    inside = view.camera.in_image(uv) & (z > min_z)
    state = np.full(len(points), "outside", dtype="<U8")
    rows = np.full(len(points), -1, dtype=np.int64)
    cols = np.full(len(points), -1, dtype=np.int64)
    h, w = depth.shape[:2]
    sx, sy = w / view.camera.width, h / view.camera.height
    cols[inside] = np.clip(np.floor((uv[inside, 0] + 0.5) * sx).astype(int), 0, w - 1)
    rows[inside] = np.clip(np.floor((uv[inside, 1] + 0.5) * sy).astype(int), 0, h - 1)
    d = depth[rows[inside], cols[inside]].astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    near_ok = valid & (d >= z[inside] - depth_tolerance(z[inside]))
    state[inside] = np.where(~valid, "no_depth", np.where(near_ok, "visible", "hidden"))
    return uv, np.stack([axis, sign], axis=1), state, rows, cols


def hole_stats(depth: Array) -> dict[str, float]:
    """Structure of the invalid pixels of a depth map.

    Parameters
    ----------
    depth : numpy.ndarray, shape (H, W)
        Depth in metres (0 or NaN = invalid).

    Returns
    -------
    dict of str to float
        ``valid_share``; ``n_holes`` (4-connected regions of invalid pixels); ``largest_hole_px``;
        ``small_hole_share``, the share of invalid pixels that lie in holes of at most :data:`SMALL_HOLE_PX`
        pixels.
    """
    valid = np.isfinite(depth) & (depth > 0)
    labels, n = ndimage.label(~valid)
    sizes = np.bincount(labels.ravel())[1:]
    n_invalid = int((~valid).sum())
    return {"valid_share": float(valid.mean()), "n_holes": float(n),
            "largest_hole_px": float(sizes.max()) if n else 0.0,
            "small_hole_share": float(sizes[sizes <= SMALL_HOLE_PX].sum() / n_invalid) if n_invalid else 0.0}


def draw_samples(rgb: NDArray[np.uint8], uv: Array, state: NDArray[np.str_], path: Path) -> None:
    """Save the image with each in-image sample as a dot (green visible, red hidden, yellow no depth) and the
    rectangle of the visible samples (cyan).

    Parameters
    ----------
    rgb : numpy.ndarray, shape (H, W, 3), uint8
        The view's image.
    uv : numpy.ndarray, shape (N, 2)
        Sample projections, pixels.
    state : numpy.ndarray, shape (N,), str
        Sample states (:data:`SAMPLE_STATES`).
    path : pathlib.Path
        Output picture; must not exist.
    """
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    colours = {"visible": (0, 220, 0), "hidden": (255, 30, 30), "no_depth": (255, 220, 0)}
    for (u, v), s in zip(uv, state):
        if s in colours:
            draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=colours[s])
    seen = state == "visible"
    if seen.any():
        draw.rectangle((uv[seen, 0].min(), uv[seen, 1].min(), uv[seen, 0].max(), uv[seen, 1].max()),
                       outline=(0, 255, 255), width=3)
    with open(path, "xb") as f:
        image.save(f, format="PNG")


def run_inspect(args: argparse.Namespace) -> int:
    """The ``inspect`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.

    Raises
    ------
    RuntimeError
        If the sample states disagree with :func:`object_in_view`.
    """
    from measured_space.datasets import arkitscenes

    if args.draw is not None and args.draw.exists():
        print(f"refusing to overwrite {args.draw}", file=sys.stderr)
        return 2
    view = next((v for v in arkitscenes.list_frames(args.video) if v.frame == args.frame), None)
    obj = next((o for o in arkitscenes.load_objects(args.video) if o.object_id == args.object), None)
    if view is None or obj is None:
        print(f"no posed frame {args.frame} or object {args.object} in video {args.video}", file=sys.stderr)
        return 2
    depth = arkitscenes.load_laser_depth(view)
    uv, face, state, rows, cols = sample_states(view, obj, depth)
    seen, diag = object_in_view(view, obj, depth)
    n = len(state)
    if (n != diag["n_samples"] or not np.isclose((state == "visible").mean(), seen.visible_fraction)
            or not np.isclose((state == "outside").mean(), seen.truncated_fraction)
            or not np.isclose((state == "no_depth").mean(), diag["no_depth_fraction"])):
        raise RuntimeError("sample states disagree with object_in_view")
    counts = {s: int((state == s).sum()) for s in SAMPLE_STATES}
    in_frame = n - counts["outside"]
    known = in_frame - counts["no_depth"]
    print(f"{obj.label} {obj.object_id}, box size {np.round(obj.box.size, 2).tolist()} m, video {args.video} "
          f"frame {args.frame}")
    print(f"camera-facing samples {n}: " + ", ".join(f"{s} {counts[s]}" for s in SAMPLE_STATES))
    print(f"visible share: of all camera-facing {counts['visible'] / n:.3f} (pipeline), of in-frame "
          f"{counts['visible'] / in_frame if in_frame else float('nan'):.3f}, of in-frame with depth "
          f"{counts['visible'] / known if known else float('nan'):.3f}")
    for (a, s) in sorted({(int(a), int(s)) for a, s in face}):
        on = (face[:, 0] == a) & (face[:, 1] == s)
        dims = [round(float(obj.box.size[i]), 2) for i in range(3) if i != a]
        print(f"  face {'+' if s > 0 else '-'}{'xyz'[a]} ({dims[0]} x {dims[1]} m): samples {int(on.sum())}, "
              + ", ".join(f"{t} {int((on & (state == t)).sum())}" for t in SAMPLE_STATES))
    holes = hole_stats(depth)
    print("laser depth: " + ", ".join(f"{k} {v:.4f}" if k.endswith("share") else f"{k} {int(v)}"
                                      for k, v in holes.items()))
    nd = state == "no_depth"
    if nd.any():
        valid = np.isfinite(depth) & (depth > 0)
        distance = ndimage.distance_transform_edt(~valid)[rows[nd], cols[nd]]
        print(f"no-depth samples: distance to the nearest pixel with depth, median {np.median(distance):.1f} px, "
              f"max {distance.max():.1f} px, within 2 px {int((distance <= 2.0).sum())} of {int(nd.sum())}")
    if args.draw is not None:
        draw_samples(arkitscenes.load_rgb(view), uv, state, args.draw)
        print(f"wrote {args.draw}")
    return 0


# --------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Parameters
    ----------
    argv : list of str or None
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments, with ``func`` the subcommand to run.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("recall", help="recall under each clearly-visible definition, from stored records")
    p.add_argument("--labels", type=Path, action="append", required=True,
                   help="a label folder (repeatable); population sizes are printed for the first")
    p.add_argument("--dataset", default="arkitscenes", help="dataset subfolder (depth-based visibility only)")
    p.add_argument("--min-known", type=int, default=10,
                   help="known_depth: fewer in-frame samples with depth than this is undetermined")
    p.add_argument("--out", type=Path, default=None, help="optional CSV of the recall rows (must not exist)")
    p.set_defaults(func=run_recall)
    q = sub.add_parser("inspect", help="one ARKitScenes object in one view against the laser depth")
    q.add_argument("--video", required=True, help="ARKitScenes video id")
    q.add_argument("--frame", required=True, help="wide frame timestamp, as in the view key")
    q.add_argument("--object", required=True, help="ground-truth object id")
    q.add_argument("--draw", type=Path, default=None, help="optional PNG of the samples (must not exist)")
    q.set_defaults(func=run_inspect)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the chosen subcommand.

    Parameters
    ----------
    argv : list of str or None
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code.
    """
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
