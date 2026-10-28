"""Build a static HTML viewer of ground truth, automatic labels and their pairing, with a spot-check page.

The viewer (:mod:`measured_space.viewer`) recomputes the 2D pairing of every selected view from the stored
records of one or more label folders and writes, to ``<data_root>/outputs/viewer/<name>/``:

- ``index.html``: per scene, the views as thumbnails linking to full overlays, and per view a table of the
  present ground-truth objects with, per source, the paired detection, 2D IoU, ambiguity, 3D centre distance
  and label check; filters for views with an ambiguous pair or a label mismatch;
- ``spotcheck.html``: a seeded sample of pairs of one source (half ambiguous, half clear, spread over scenes)
  to mark "right pair", "wrong pair" or "can't tell", with an export of the answers as JSON;
- ``images/``: the overlays, thumbnails and spot-check crops (derived from the datasets' images, so they stay
  outside the repository);
- ``manifest.json``: the inputs, counts, the sample with every pair's details, and any image errors.

The output folder must not exist or must be empty; an existing one is never written into. ``--labels`` may be
given several times to overlay sources from folders built on the same views (for example the v0 folder and a
``cutr_sam3`` folder); the first folder chooses the views.

``--summarize-answers <export.json>`` reads an export of the spot-check page instead and prints the wrong-pair
rate, overall and split by ambiguous and clear pairs, with 95% Wilson intervals; it writes nothing.

Run from the repository root::

    python scripts/make_viewer.py --labels <data>/outputs/labels_2d --datasets arkitscenes \\
        --scenes 42444946 42445021 --sources lidar --name ark_small --sample-size 20
    python scripts/make_viewer.py --labels <data>/outputs/labels_2d --labels <data>/outputs/labels_cutr_sam3 \\
        --datasets arkitscenes --sources cutr_sam3 lidar --spotcheck-source cutr_sam3 --name ark_cutr
    python scripts/make_viewer.py --summarize-answers ~/Downloads/spotcheck_<sample_id>.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from measured_space.config import data_root
from measured_space.viewer import build_viewer, html_problems, summarize_answers

NAME_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
"""re.Pattern: allowed viewer names (one folder level, no path separators, no leading dot)."""


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
    p.add_argument("--labels", type=Path, action="append", default=None,
                   help="label folder (<labels>/<dataset>/shards); repeat to add sources from folders built on "
                        "the same views; the first chooses the views")
    p.add_argument("--datasets", nargs="+", choices=["arkitscenes", "smartspaces"], default=None,
                   help="datasets to show (default: every dataset of the first label folder)")
    p.add_argument("--scenes", nargs="*", default=None, help="keep scenes containing one of these substrings")
    p.add_argument("--sources", nargs="+", default=None,
                   help="automatic sources to show, the overlays' pairing source first (default: lidar for "
                        "ARKitScenes, moge for Smart Spaces)")
    p.add_argument("--spotcheck-source", default=None,
                   help="source of the spot-check sample (default: lidar for ARKitScenes, moge for Smart Spaces)")
    p.add_argument("--name", default=None, help="viewer name: output goes to <data_root>/outputs/viewer/<name>/")
    p.add_argument("--sample-size", type=int, default=100, help="spot-check pairs wanted (half ambiguous)")
    p.add_argument("--seed", type=int, default=0, help="seed of the spot-check sample")
    p.add_argument("--max-views", type=int, default=None,
                   help="at most this many views per scene in index.html, evenly spaced (the spot-check still "
                        "samples from every selected view)")
    p.add_argument("--summarize-answers", type=Path, default=None,
                   help="summarise an exported spot-check JSON instead of building a viewer")
    return p.parse_args(argv)


def viewer_dir(name: str) -> Path:
    """The output folder of a viewer.

    Parameters
    ----------
    name : str
        The viewer's name, one folder level.

    Returns
    -------
    pathlib.Path
        ``<data_root>/outputs/viewer/<name>``.

    Raises
    ------
    ValueError
        If the name is not a plain folder name (:data:`NAME_PATTERN`).
    """
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(f"--name {name!r} must be a plain folder name ({NAME_PATTERN.pattern})")
    return data_root() / "outputs" / "viewer" / name


def main(argv: Sequence[str] | None = None) -> int:
    """Build a viewer, or summarise a spot-check export.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        0 on success; 1 if the output folder is refused, the inputs are unusable, or a written page does not
        parse as well-formed HTML; 2 for a missing argument.
    """
    args = parse_args(argv)
    if args.summarize_answers is not None:
        export = json.loads(args.summarize_answers.read_text(encoding="utf-8"))
        print(json.dumps(summarize_answers(export), indent=1))
        return 0
    if not args.labels or not args.name:
        print("--labels and --name are needed to build a viewer", file=sys.stderr)
        return 2
    try:
        out = viewer_dir(args.name)
        manifest = build_viewer(args.labels, out, name=args.name, datasets=args.datasets, scenes=args.scenes,
                                sources=args.sources, spot_source=args.spotcheck_source,
                                sample_size=args.sample_size, seed=args.seed, max_views=args.max_views,
                                log=lambda message: print(message, flush=True))
    except (FileExistsError, ValueError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    status = 0
    for page in ("index.html", "spotcheck.html"):
        problems = html_problems((out / page).read_text(encoding="utf-8"))
        print(f"{out / page}: {'well formed' if not problems else f'{len(problems)} problems'}")
        for problem in problems[:10]:
            print(f"  {problem}")
        status = status or int(bool(problems))
    spot = manifest["spotcheck"]
    print(json.dumps({"out": str(out), "views_selected": manifest["n_views_selected"],
                      "views_drawn": manifest["n_views_drawn"], "pairs_per_source": manifest["pairs_per_source"],
                      "spotcheck": {k: spot[k] for k in ("sample_id", "source", "n_sampled", "n_sampled_ambiguous",
                                                         "n_sampled_clear", "population")},
                      "image_errors": len(manifest["image_errors"])}, indent=1))
    return status


if __name__ == "__main__":
    sys.exit(main())
