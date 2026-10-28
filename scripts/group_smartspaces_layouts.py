"""Group Smart Spaces warehouse scenes by building footprint, for leave-one-layout-out audits.

Several synthetic warehouses of Physical AI Smart Spaces (``MTMC_Tracking_2026``) are built in the same building,
so they share far more than an object list, and an audit that leaves one scene out still trains and calibrates on
the same building. This script finds which scenes share a building from their camera calibration alone:

- a scene's camera centres, ``c = -R^T t`` from each camera's 3x4 world-to-camera ``extrinsicMatrix`` in its
  ``calibration.json`` (:func:`camera_centres`; sensors of another ``type`` than ``camera`` are left out, as in
  :func:`measured_space.datasets.smartspaces.parse_calibration`);
- its *footprint*, the axis-aligned xy bounding box of those centres (:func:`footprint`);
- *layouts*, the connected components of the relation "footprint IoU >= ``--min-iou``" (default 0.5;
  :func:`footprint_iou`, :func:`connected_groups`), numbered in sorted order of each layout's first scene.

It prints each scene's camera count, footprint and layout, and the pairwise IoU above zero, and writes
``{"smartspaces": {scene: "layout_<k>"}}`` to ``--out`` (default ``configs/smartspaces_layout_groups.json``),
the format ``scripts/run_audit.py --fold-groups`` reads. It refuses to overwrite an existing file. It reads only
calibration files (no labels, no ground truth, no errors), so the grouping cannot be tuned to an audit result.

Usage::

    python scripts/group_smartspaces_layouts.py                       # the 12 audited scenes
    python scripts/group_smartspaces_layouts.py --out <new file> --scenes MTMC_Tracking_2026/train/Warehouse_000 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from measured_space.config import REPO_ROOT
from measured_space.datasets.smartspaces import DATASET, scene_dir

Footprint = tuple[float, float, float, float]

SCENES: tuple[str, ...] = tuple(f"MTMC_Tracking_2026/train/Warehouse_{n}" for n in
                                ("000", "002", "004", "006", "010", "011", "012", "013", "014", "015", "016")) \
    + ("MTMC_Tracking_2026/val/Warehouse_020",)
"""tuple of str: the 12 audited warehouse scenes, as the labels name them."""

DEFAULT_OUT = REPO_ROOT / "configs" / "smartspaces_layout_groups.json"
"""pathlib.Path: where the mapping is written by default."""


def camera_centres(calibration: Mapping[str, Any]) -> NDArray[np.float64]:
    """World positions of a scene's cameras from its calibration.

    Parameters
    ----------
    calibration : mapping
        Parsed ``calibration.json``: a ``sensors`` list whose camera entries (``type`` ``"camera"`` or absent)
        hold a 3x4 world-to-camera ``extrinsicMatrix`` ``[R | t]``.

    Returns
    -------
    numpy.ndarray, shape (n, 3)
        Camera centres ``-R^T t`` (world metres, z up), in sensor order.

    Raises
    ------
    ValueError
        If no camera has an ``extrinsicMatrix``.
    """
    centres = []
    for sensor in calibration["sensors"]:
        if sensor.get("type", "camera") != "camera" or "extrinsicMatrix" not in sensor:
            continue
        extrinsic = np.asarray(sensor["extrinsicMatrix"], dtype=np.float64).reshape(3, 4)
        centres.append(-extrinsic[:, :3].T @ extrinsic[:, 3])
    if not centres:
        raise ValueError("calibration has no camera with an extrinsicMatrix")
    return np.asarray(centres)


def footprint(centres: NDArray[np.float64]) -> Footprint:
    """The axis-aligned xy bounding box of camera centres.

    Parameters
    ----------
    centres : numpy.ndarray, shape (n, 2) or (n, 3)
        Camera centres (m); only x and y are used.

    Returns
    -------
    tuple of float
        ``(xmin, ymin, xmax, ymax)`` in metres.
    """
    xy = np.asarray(centres, dtype=np.float64)[:, :2]
    return (float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max()))


def footprint_iou(a: Footprint, b: Footprint) -> float:
    """Intersection over union of two axis-aligned rectangles.

    Parameters
    ----------
    a, b : tuple of float
        ``(xmin, ymin, xmax, ymax)``.

    Returns
    -------
    float
        Overlap area over union area, in [0, 1]. Two degenerate rectangles (zero union area) give 1.0 if they
        are equal, else 0.0.
    """
    def area(r: Footprint) -> float:
        """Area of a rectangle.

        Parameters
        ----------
        r : tuple of float
            ``(xmin, ymin, xmax, ymax)``.

        Returns
        -------
        float
            ``(xmax - xmin) * (ymax - ymin)``, at least 0.
        """
        return max(r[2] - r[0], 0.0) * max(r[3] - r[1], 0.0)

    inter = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    union = area(a) + area(b) - inter
    if union <= 0.0:
        return 1.0 if tuple(a) == tuple(b) else 0.0
    return inter / union


def connected_groups(footprints: Mapping[str, Footprint], min_iou: float = 0.5) -> dict[str, int]:
    """Group scenes into the connected components of "footprint IoU >= ``min_iou``".

    Parameters
    ----------
    footprints : mapping of str to tuple of float
        Scene to footprint.
    min_iou : float, default 0.5
        Two scenes are linked when their footprints' IoU is at least this.

    Returns
    -------
    dict of str to int
        Scene to group number in ``0 .. G - 1``, in sorted scene order; groups are numbered in sorted order of
        their first scene (as :func:`measured_space.audit.scene_folds` numbers folds). Linking is transitive: a
        chain of overlapping footprints is one group even when its ends do not overlap.
    """
    names = sorted(footprints)
    parent = {s: s for s in names}

    def root(s: str) -> str:
        """The representative of a scene's component.

        Parameters
        ----------
        s : str
            Scene.

        Returns
        -------
        str
            The component's representative.
        """
        while parent[s] != s:
            s = parent[s]
        return s

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if footprint_iou(footprints[a], footprints[b]) >= min_iou:
                ra, rb = root(a), root(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    number: dict[str, int] = {}
    for s in names:
        number.setdefault(root(s), len(number))
    return {s: number[root(s)] for s in names}


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenes", nargs="+", default=list(SCENES),
                        help="scenes as the labels name them (default: the 12 audited warehouses)")
    parser.add_argument("--min-iou", type=float, default=0.5, help="footprint IoU that links two scenes")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="mapping file to write; must not exist (default %(default)s)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Compute the footprints and layouts, print them and write the mapping.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments, or ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        0 on success, 1 if the output file already exists (nothing is written).
    """
    args = parse_args(argv)
    if args.out.exists():
        print(f"{args.out} exists; refusing to overwrite it (pass --out with a new path)", file=sys.stderr)
        return 1
    footprints: dict[str, Footprint] = {}
    n_cameras: dict[str, int] = {}
    for scene in args.scenes:
        centres = camera_centres(json.loads((scene_dir(scene) / "calibration.json").read_text()))
        footprints[scene], n_cameras[scene] = footprint(centres), len(centres)
    group = connected_groups(footprints, args.min_iou)
    print(f"{'scene':<40} {'cameras':>7}  {'x range (m)':>17}  {'y range (m)':>17}  {'size (m)':>13}  layout")
    for scene in sorted(footprints):
        x0, y0, x1, y1 = footprints[scene]
        print(f"{scene:<40} {n_cameras[scene]:>7}  [{x0:7.2f}, {x1:7.2f}]  [{y0:7.2f}, {y1:7.2f}]  "
              f"{x1 - x0:5.1f} x {y1 - y0:5.1f}  layout_{group[scene]}")
    names = sorted(footprints)
    print(f"\npairwise footprint IoU > 0 (linked at >= {args.min_iou}):")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            iou = footprint_iou(footprints[a], footprints[b])
            if iou > 0.0:
                print(f"  {a.rsplit('/', 1)[-1]} {b.rsplit('/', 1)[-1]}  {iou:.3f}")
    mapping = {DATASET: {scene: f"layout_{group[scene]}" for scene in names}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:                         # "x": fail rather than overwrite, even on a race
        f.write(json.dumps(mapping, indent=1) + "\n")
    print(f"\nwrote {len(names)} scenes in {len(set(group.values()))} layouts to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
