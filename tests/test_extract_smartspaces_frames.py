"""Tests for ``scripts/extract_smartspaces_frames.py`` on tiny synthetic videos.

Each synthetic frame is a flat grey whose level encodes its index (20 x index), so the level of a written JPEG
shows which source frame it came from (off-by-one decoding would show up as a 20-level error). The videos are
written with OpenCV's Motion-JPEG encoder under ``tmp_path``; the tests skip if OpenCV is missing or cannot write
such a video.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_smartspaces_frames.py"


def _script() -> ModuleType:
    """Import ``scripts/extract_smartspaces_frames.py`` as a module (it is not part of the package).

    Returns
    -------
    types.ModuleType
        The module; importing it loads OpenCV only.
    """
    if "extract_smartspaces_frames" in sys.modules:
        return sys.modules["extract_smartspaces_frames"]
    spec = importlib.util.spec_from_file_location("extract_smartspaces_frames", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["extract_smartspaces_frames"] = module
    spec.loader.exec_module(module)
    return module


def write_video(path: Path, n_frames: int) -> Path:
    """Write a 64 x 36 Motion-JPEG AVI whose frame ``i`` is flat grey ``20 i``.

    Parameters
    ----------
    path : pathlib.Path
        Output file; the container is AVI whatever the suffix (OpenCV reads it by content).
    n_frames : int
        Frames (at most 12, so the levels stay below 255).

    Returns
    -------
    pathlib.Path
        ``path``.
    """
    tmp = path.with_name(path.stem + "_tmp.avi")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 36))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write Motion-JPEG video here")
    for i in range(n_frames):
        writer.write(np.full((36, 64, 3), 20 * i, dtype=np.uint8))
    writer.release()
    tmp.rename(path)
    return path


def level(path: Path) -> float:
    """Mean grey level of a written JPEG, after checking its size.

    Parameters
    ----------
    path : pathlib.Path
        JPEG.

    Returns
    -------
    float
        Mean over pixels and channels.
    """
    image = cv2.imread(str(path))
    assert image.shape == (540, 960, 3)
    return float(image.mean())


def test_extract_keeps_the_requested_frames_and_skips_existing(tmp_path: Path) -> None:
    """Frames 0, 3, 6 and 9 of a 12-frame video are written at 960 x 540 with their own grey level; a second run
    skips all four and leaves the files as they were.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    """
    ex = _script()
    video = write_video(tmp_path / "Camera_0000.mp4", 12)
    out = tmp_path / "cache"
    assert ex.extract_camera(video, out, [0, 3, 6, 9]) == (4, 0)
    assert sorted(p.name for p in out.iterdir()) == ["000000.jpg", "000003.jpg", "000006.jpg", "000009.jpg"]
    for index in (0, 3, 6, 9):
        assert level(out / f"{index:06d}.jpg") == pytest.approx(20 * index, abs=4)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert ex.extract_camera(video, out, [0, 3, 6, 9]) == (0, 4)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_short_video_warns_by_default_and_fails_when_strict(tmp_path: Path,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    """A 10-frame video asked for frames 0, 5 and 12: frames 0 and 5 are written, then a warning (default) or an
    error (``strict``).

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    capsys : pytest.CaptureFixture
        Captures the warning.
    """
    ex = _script()
    video = write_video(tmp_path / "Camera_0001.mp4", 10)
    assert ex.extract_camera(video, tmp_path / "a", [0, 5, 12]) == (2, 0)
    assert "WARNING: Camera_0001.mp4 ended at frame 10 (wanted up to 12)" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="failed to read frame 10"):
        ex.extract_camera(video, tmp_path / "b", [0, 5, 12], strict=True)
    assert sorted(p.name for p in (tmp_path / "b").iterdir()) == ["000000.jpg", "000005.jpg"]


def test_main_extracts_every_calibrated_camera(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Both cameras of ``calibration.json`` (not the other sensor) get ``range(2, 8, 3)`` = frames 2 and 5, in the
    SparseQID layout ``<cache>/<split>/<scene>/<camera>/<frame:06d>.jpg``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder.
    capsys : pytest.CaptureFixture
        Captures the per-camera lines.
    """
    ex = _script()
    scene = tmp_path / "data" / "val" / "Warehouse_099"
    (scene / "videos").mkdir(parents=True)
    sensors = [{"type": "camera", "id": "Camera_0001"}, {"type": "lidar", "id": "Lidar_0000"},
               {"type": "camera", "id": "Camera_0000"}]
    (scene / "calibration.json").write_text(json.dumps({"sensors": sensors}))
    assert ex.camera_ids(scene) == ["Camera_0000", "Camera_0001"]
    for cam in ("Camera_0000", "Camera_0001"):
        write_video(scene / "videos" / f"{cam}.mp4", 8)
    cache = tmp_path / "cache"
    assert ex.main(["--data-root", str(tmp_path / "data"), "--split", "val", "--scene", "Warehouse_099",
                    "--cache-root", str(cache), "--start", "2", "--end", "8", "--step", "3"]) == 0
    for cam in ("Camera_0000", "Camera_0001"):
        folder = cache / "val" / "Warehouse_099" / cam
        assert sorted(p.name for p in folder.iterdir()) == ["000002.jpg", "000005.jpg"]
        assert level(folder / "000005.jpg") == pytest.approx(100, abs=4)
    assert "Warehouse_099/Camera_0001: 2 frames [2..5] step 3: wrote 2, skipped 0" in capsys.readouterr().out
