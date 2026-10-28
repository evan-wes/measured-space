"""Decode a frame window from every camera video of a Smart Spaces scene into a half-resolution JPEG cache.

The cache is the input of ``scripts/run_sparse4d.py``. Its layout is SparseQID's
(``<cache_root>/<split>/<scene>/<camera_id>/<frame:06d>.jpg``) so that SparseQID's
``sparseqid.data.frame_dataset.FrameJpgDataset`` (https://github.com/playbox-dev/SparseQID, commit ``7aea61d``,
Apache-2.0) reads it directly. Preprocessing mirrors EVA/GaugeAlign's ``build_pkl.py --half_res`` (OpenCV decode,
``cv2.resize`` to 960x540 with the default bilinear filter, JPEG quality 92), except that frames are always read
sequentially rather than by seeking, so the decoded frame index is exact.

The frames kept are ``range(--start, --end, --step)`` (``--end`` exclusive), for every camera listed in the scene's
``calibration.json`` (``<data-root>/<split>/<scene>/videos/<camera_id>.mp4``). Frames that are not kept are only
grabbed (decoded without the colour conversion): the same frame sequence, faster. A video that ends before the
last requested frame is reported with a ``WARNING`` line and that camera stops there (the default, as in the
12-scene run); with ``--strict-length`` it is an error instead (the first, single-scene version's behaviour).

Usage (``measured-space-sparse4d`` env or any env with OpenCV), e.g. the 12-scene cache (every 6th frame of 0..8999)::

    python -I scripts/extract_smartspaces_frames.py --data-root <data>/MTMC_Tracking_2026 --split train \\
        --scene Warehouse_000 --cache-root <outputs>/sparse4d/frame_cache --start 0 --end 9000 --step 6

Prints one line per camera (frames written and skipped) and the elapsed time. Existing JPEGs are never
overwritten (they are skipped), so an interrupted run can be resumed. Nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import cv2

HALF_RES = (960, 540)
"""tuple of int: output (width, height), the ``cv2.resize`` target."""

JPEG_QUALITY = 92
"""int: JPEG quality, EVA's value."""


def camera_ids(scene_dir: Path) -> list[str]:
    """Return the sorted camera sensor ids listed in a scene's ``calibration.json``.

    Parameters
    ----------
    scene_dir : Path
        Scene directory containing ``calibration.json``.

    Returns
    -------
    list[str]
        Camera ids such as ``"Camera_0000"``, sorted.
    """
    calib = json.loads((scene_dir / "calibration.json").read_text())
    return sorted(s["id"] for s in calib["sensors"] if s.get("type") == "camera")


def extract_camera(video: Path, out_dir: Path, frames: list[int], quality: int = JPEG_QUALITY,
                   strict: bool = False) -> tuple[int, int]:
    """Decode the requested frame indices of one video sequentially and write half-resolution JPEGs.

    Parameters
    ----------
    video : Path
        Camera ``.mp4``.
    out_dir : Path
        Destination directory for ``<frame:06d>.jpg`` files.
    frames : list[int]
        Sorted, 0-based source frame indices to keep.
    quality : int, optional
        JPEG quality, by default 92 (EVA's value).
    strict : bool, optional
        If True, a video that ends before ``max(frames)`` raises; if False (default), a warning is printed and
        the frames read so far are kept.

    Returns
    -------
    tuple[int, int]
        Number of JPEGs written and number skipped because they already existed.

    Raises
    ------
    RuntimeError
        If ``strict`` and the video ends before the last requested frame.
    OSError
        If a JPEG cannot be written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(frames)
    last = max(frames)
    cap = cv2.VideoCapture(str(video))
    written = skipped = 0
    index = 0
    while index <= last:
        if index in wanted:
            ok, image = cap.read()
        else:
            ok, image = cap.grab(), None  # decode without colour conversion: exact sequential skip
        if not ok:
            if strict:
                raise RuntimeError(f"failed to read frame {index} of {video}")
            print(f"WARNING: {video.name} ended at frame {index} (wanted up to {last}); "
                  f"wrote {written}, skipped {skipped}", flush=True)
            break
        if index in wanted:
            dest = out_dir / f"{index:06d}.jpg"
            if dest.exists():
                skipped += 1
            else:
                small = cv2.resize(image, HALF_RES)
                if not cv2.imwrite(str(dest), small, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                    raise OSError(f"failed to write {dest}")
                written += 1
        index += 1
    cap.release()
    return written, skipped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments without the program name; ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        ``data_root``, ``split``, ``scene``, ``cache_root``, ``start``, ``end``, ``step``, ``strict_length``.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True, help="MTMC_Tracking_2026 folder")
    ap.add_argument("--split", required=True, help="train or val")
    ap.add_argument("--scene", required=True, help="scene folder name, e.g. Warehouse_000")
    ap.add_argument("--cache-root", type=Path, required=True, help="frame cache root")
    ap.add_argument("--start", type=int, default=0, help="first frame (0)")
    ap.add_argument("--end", type=int, required=True, help="exclusive")
    ap.add_argument("--step", type=int, default=1, help="frame stride (1)")
    ap.add_argument("--strict-length", action="store_true",
                    help="error (instead of a warning) when a video ends before the last requested frame")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and extract the window for every calibrated camera of the scene.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments without the program name; ``None`` for ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status 0 (errors raise).

    Raises
    ------
    FileNotFoundError
        If a calibrated camera has no video.
    """
    args = parse_args(argv)
    scene_dir = args.data_root / args.split / args.scene
    frames = list(range(args.start, args.end, args.step))
    t0 = time.time()
    for cam in camera_ids(scene_dir):
        video = scene_dir / "videos" / f"{cam}.mp4"
        if not video.exists():
            raise FileNotFoundError(video)
        out = args.cache_root / args.split / args.scene / cam
        written, skipped = extract_camera(video, out, frames, strict=args.strict_length)
        print(f"{args.scene}/{cam}: {len(frames)} frames [{frames[0]}..{frames[-1]}] step {args.step}: "
              f"wrote {written}, skipped {skipped}", flush=True)
    print(f"done in {time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
