"""Choose which two cameras of each fixed-camera warehouse scene to use, from the ground truth alone.

Only two cameras per scene are downloaded, so the pair should be informative. It should see forklifts and
people together (for safety-distance questions), and it should see the **same** objects at the same time,
so the two cameras' estimates of an object can be compared without ground truth (a cross-camera consistency
check). For every camera pair this script measures, on every ``--step``-th frame:

- ``co_visible``: mean number of objects visible in both cameras;
- ``joint_presence``: share of frames in which both cameras see at least one forklift and one person.

It picks the pair with the most co-visible objects among pairs whose joint presence reaches
``--min-presence`` (or the best joint presence if no pair does). The choice and the statistics are written
to a YAML file, so the selection is reproducible.

Usage::

    python scripts/select_smartspaces_cameras.py --out configs/smartspaces_cameras.yaml
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from measured_space.datasets import smartspaces

DEFAULT_SCENES = [f"MTMC_Tracking_2026/train/Warehouse_{n:03d}" for n in (0, 2, 4, 6, 10, 11, 17, 19)] + [
    "MTMC_Tracking_2026/val/Warehouse_020"]
"""list of str: the forklift scenes added for scene diversity (all have forklifts, pallet trucks and people)."""


def _label(object_id: str) -> str:
    """Recover the class name from a ``"<type>_<id>"`` object id.

    Parameters
    ----------
    object_id : str
        Object id as built by the loader.

    Returns
    -------
    str
        The class name.
    """
    return object_id.rsplit("_", 1)[0]


def camera_pair_stats(scene: str, step: int) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], dict[str, float]]]:
    """Per-camera and per-pair visibility statistics of one scene.

    Parameters
    ----------
    scene : str
        Scene path below the dataset folder.
    step : int
        Use every ``step``-th frame.

    Returns
    -------
    per_camera : dict of str to dict
        For each camera: ``mean_visible`` (objects per frame) and ``forklift_person`` (share of frames with
        at least one forklift and one person visible).
    per_pair : dict of (str, str) to dict
        For each camera pair: ``co_visible`` (objects visible in both, per frame) and ``joint_presence``
        (share of frames in which both cameras see a forklift and a person).
    """
    visible = smartspaces.visible_boxes_2d(scene)
    cameras = sorted(smartspaces.load_calibration(scene))
    frames = sorted(visible)[::step]
    seen = {c: [] for c in cameras}                        # per frame, the set of object ids each camera sees
    for f in frames:
        per_cam = {c: set() for c in cameras}
        for object_id, cams in visible[f].items():
            for c in cams:
                if c in per_cam:
                    per_cam[c].add(object_id)
        for c in cameras:
            seen[c].append(per_cam[c])

    def has_both(ids: set[str]) -> bool:
        """Tell whether a set of object ids includes a forklift and a person.

        Parameters
        ----------
        ids : set of str
            Object ids seen by one camera in one frame.

        Returns
        -------
        bool
            True if at least one forklift and one person are among them.
        """
        labels = Counter(_label(i) for i in ids)
        return labels["Forklift"] > 0 and labels["Person"] > 0

    per_camera = {c: {"mean_visible": float(np.mean([len(s) for s in seen[c]])),
                      "forklift_person": float(np.mean([has_both(s) for s in seen[c]]))} for c in cameras}
    per_pair = {}
    for a, b in itertools.combinations(cameras, 2):
        per_pair[(a, b)] = {
            "co_visible": float(np.mean([len(sa & sb) for sa, sb in zip(seen[a], seen[b])])),
            "joint_presence": float(np.mean([has_both(sa) and has_both(sb) for sa, sb in zip(seen[a], seen[b])])),
        }
    return per_camera, per_pair


def choose_pair(per_pair: dict[tuple[str, str], dict[str, float]], min_presence: float) -> tuple[str, str]:
    """Pick the most co-visible camera pair that sees forklifts and people together often enough.

    Parameters
    ----------
    per_pair : dict of (str, str) to dict
        Output of :func:`camera_pair_stats`.
    min_presence : float
        Minimum ``joint_presence`` for a pair to be eligible.

    Returns
    -------
    tuple of str
        The chosen camera ids, sorted.
    """
    eligible = {p: s for p, s in per_pair.items() if s["joint_presence"] >= min_presence}
    if eligible:
        return max(eligible, key=lambda p: (eligible[p]["co_visible"], eligible[p]["joint_presence"]))
    return max(per_pair, key=lambda p: (per_pair[p]["joint_presence"], per_pair[p]["co_visible"]))


def main(argv: list[str] | None = None) -> int:
    """Select camera pairs for the given scenes and write them to YAML.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments (``None`` reads ``sys.argv``).

    Returns
    -------
    int
        Process exit status (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES, help="scene paths below the dataset folder")
    parser.add_argument("--step", type=int, default=30, help="use every N-th frame (default 30 = 1 per second)")
    parser.add_argument("--min-presence", type=float, default=0.5,
                        help="minimum share of frames where both cameras see a forklift and a person")
    parser.add_argument("--out", type=Path, required=True, help="output YAML file")
    args = parser.parse_args(argv)

    selection = {}
    for scene in args.scenes:
        per_camera, per_pair = camera_pair_stats(scene, args.step)
        a, b = choose_pair(per_pair, args.min_presence)
        stats = per_pair[(a, b)]
        selection[scene] = {"cameras": [a, b], "co_visible": round(stats["co_visible"], 3),
                            "joint_presence": round(stats["joint_presence"], 3),
                            "mean_visible": [round(per_camera[a]["mean_visible"], 2),
                                             round(per_camera[b]["mean_visible"], 2)],
                            "n_cameras": len(per_camera), "step": args.step}
        print(f"{scene}: {a} + {b}  co_visible={stats['co_visible']:.2f}  "
              f"joint_presence={stats['joint_presence']:.2f}  (of {len(per_camera)} cameras)", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(selection, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
