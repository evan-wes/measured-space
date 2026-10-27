"""Tests for the Cubify Anything (CuTR) hand-off: gravity convention, box conversion, 2D boxes, JSONL, export.

Nothing here needs a GPU or the ``cubifyanything`` package. CuTR's box corners are reproduced in numpy from
``GeneralInstance3DBoxes.corners`` (ml-cubifyanything commit 00e9cb1), and its box rotation from
``AbsoluteBox3DPredictor`` and ``inference`` (a yaw about the gravity frame's y-axis, then ``T_gravity``).
``scripts/run_cubify.py`` is imported as a module (its model imports are lazy) to check that its copies of the
geometry agree with :mod:`measured_space.cubify` and that the JSONL it writes is what
:func:`measured_space.cubify.read_predictions` reads. The export runs on a tiny fake ARKitScenes folder.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from measured_space import cubify
from measured_space.datasets import arkitscenes as ark
from measured_space.geometry import Array, Camera, axis_angle_to_matrix, invert_pose, make_pose
from measured_space.pipeline import view_key

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_cubify.py"
DOWN = np.array([0.0, 0.0, -1.0])
# A level OpenCV camera looking along world +y: x right = world +x, y down = world -z, z forward = world +y.
LEVEL = np.column_stack([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# Fake ARKitScenes video: a sideways ("Left") camera, so the upright views are turned a quarter.
VID = "2000"
TRAJ_TIMES = [10.0, 10.1, 10.2]
STAMPS = ["10.050", "10.150"]
SENSOR_CAMERA = Camera(10.0, 11.0, 3.0, 2.0, 8, 6)
SIDEWAYS = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
LIDAR_MM, LASER_MM = 2000, 1500


def _script() -> ModuleType:
    """Import ``scripts/run_cubify.py`` as a module (it is not part of the package).

    Returns
    -------
    types.ModuleType
        The script's module, registered in ``sys.modules``; importing it loads neither torch nor cubifyanything.
    """
    if "run_cubify" in sys.modules:
        return sys.modules["run_cubify"]
    spec = importlib.util.spec_from_file_location("run_cubify", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_cubify"] = module
    spec.loader.exec_module(module)
    return module


def _rot(axis: str, angle: float) -> Array:
    """A rotation about one coordinate axis.

    Parameters
    ----------
    axis : str
        ``"x"``, ``"y"`` or ``"z"``.
    angle : float
        Radians.

    Returns
    -------
    numpy.ndarray, shape (3, 3), float64
        The rotation matrix.
    """
    return Rotation.from_euler(axis, angle).as_matrix()


def _tilted_pose(yaw: float, pitch: float, roll: float,
                 position: tuple[float, float, float] = (1.0, -2.0, 1.4)) -> Array:
    """A camera-to-world pose: the level camera rolled, pitched (about camera x) and yawed (about world z).

    Parameters
    ----------
    yaw, pitch, roll : float
        Radians; positive pitch tilts the optical axis down.
    position : tuple of float, optional
        Camera centre, metres.

    Returns
    -------
    numpy.ndarray, shape (4, 4), float64
        ``T_wc`` with OpenCV camera axes.
    """
    return make_pose(_rot("z", yaw) @ LEVEL @ _rot("x", -pitch) @ _rot("z", roll), position)


def _cutr_rotation(T_gravity: Array, yaw: float) -> Array:
    """CuTR's box rotation: ``euler_angles_to_matrix((yaw, 0, 0), "YXZ")`` (a yaw about y), then ``T_gravity``.

    Parameters
    ----------
    T_gravity : numpy.ndarray, shape (3, 3)
        Gravity frame to camera.
    yaw : float
        Radians.

    Returns
    -------
    numpy.ndarray, shape (3, 3), float64
        ``pred_boxes_3d.R`` (box axes as columns, camera frame).
    """
    return T_gravity @ _rot("y", yaw)


def _cutr_corners(center: Array, dims: Array, R: Array) -> Array:
    """CuTR's ``GeneralInstance3DBoxes.corners`` for one box, in its vertex order, transcribed to numpy.

    Parameters
    ----------
    center : numpy.ndarray, shape (3,)
        ``tensor[:3]``.
    dims : numpy.ndarray, shape (3,)
        ``tensor[3:6]`` = ``(l, h, w)``.
    R : numpy.ndarray, shape (3, 3)
        ``R``.

    Returns
    -------
    numpy.ndarray, shape (8, 3)
        Corners in CuTR's order (v0 ... v7).
    """
    l, h, w = dims
    verts = np.zeros((3, 8), dtype=np.asarray(dims).dtype)
    verts[0, [0, 3, 4, 7]], verts[0, [1, 2, 5, 6]] = -l / 2, l / 2
    verts[1, [0, 1, 4, 5]], verts[1, [2, 3, 6, 7]] = -h / 2, h / 2
    verts[2, [0, 1, 2, 3]], verts[2, [4, 5, 6, 7]] = -w / 2, w / 2
    return (R @ verts + np.asarray(center)[:, None]).T


def _same_corners(a: Array, b: Array) -> float:
    """Order-free distance between two corner sets.

    Parameters
    ----------
    a, b : numpy.ndarray, shape (8, 3)
        Corner sets.

    Returns
    -------
    float
        The largest distance from a corner of either set to the nearest corner of the other.
    """
    d = np.linalg.norm(np.asarray(a)[:, None] - np.asarray(b)[None], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def _random_cutr_boxes(rng: np.random.Generator, T_wc: Array, n: int) -> tuple[Array, Array, Array]:
    """Synthetic CuTR outputs for one view.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random source.
    T_wc : numpy.ndarray, shape (4, 4)
        The view's pose (sets ``T_gravity``).
    n : int
        Number of boxes.

    Returns
    -------
    centers : numpy.ndarray, shape (n, 3)
        Camera-frame centres, metres.
    dims : numpy.ndarray, shape (n, 3)
        ``(l, h, w)``, metres.
    rotations : numpy.ndarray, shape (n, 3, 3)
        Camera-frame rotations.
    """
    T_gravity = cubify.gravity_to_camera(T_wc)
    centers = rng.uniform([-1.5, -1.0, 1.0], [1.5, 1.0, 5.0], size=(n, 3))
    dims = rng.uniform(0.1, 2.5, size=(n, 3))
    rotations = np.stack([_cutr_rotation(T_gravity, rng.uniform(-np.pi, np.pi)) for _ in range(n)])
    return centers, dims, rotations


# --------------------------------------------------------------------------------------------- gravity


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.5])
def test_gravity_to_camera_is_identity_for_a_level_camera(yaw: float) -> None:
    """A level camera (y along world down) is its own gravity frame, whatever its yaw and position.

    Parameters
    ----------
    yaw : float
        Camera yaw about world z, radians.
    """
    np.testing.assert_allclose(cubify.gravity_to_camera(_tilted_pose(yaw, 0.0, 0.0)), np.eye(3), atol=1e-12)


def test_gravity_to_camera_is_a_yaw_free_rotation_with_y_down() -> None:
    """For tilted cameras: a proper rotation, y-axis = world down in camera axes, z-axis = forward without yaw."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        T_wc = _tilted_pose(*rng.uniform([-np.pi, -1.2, -0.6], [np.pi, 1.2, 0.6]))
        T = cubify.gravity_to_camera(T_wc)
        np.testing.assert_allclose(T.T @ T, np.eye(3), atol=1e-12)
        assert np.linalg.det(T) == pytest.approx(1.0, abs=1e-12)
        down_cam = T_wc[:3, :3].T @ DOWN
        np.testing.assert_allclose(T[:, 1], down_cam, atol=1e-12)          # maps world down to camera gravity
        # The gravity frame's forward axis lies in the plane of the optical axis and gravity, in front.
        assert abs(np.linalg.det(np.column_stack([T[:, 2], [0.0, 0.0, 1.0], down_cam]))) < 1e-12
        assert T[2, 2] > 0
    # A camera pitched down by p about its x-axis: the gravity frame is the level camera, so T = Rx(p).
    np.testing.assert_allclose(cubify.gravity_to_camera(_tilted_pose(0.4, 0.5, 0.0)), _rot("x", 0.5), atol=1e-12)
    # Adding a roll r about the optical axis: T = Rz(r).T @ Rx(p).
    np.testing.assert_allclose(cubify.gravity_to_camera(_tilted_pose(0.4, 0.5, 0.2)),
                               _rot("z", 0.2).T @ _rot("x", 0.5), atol=1e-12)


@pytest.mark.parametrize("pitch", [np.pi / 2, -np.pi / 2])
def test_gravity_to_camera_is_continuous_at_a_vertical_optical_axis(pitch: float) -> None:
    """Looking straight down or up falls back to the image-up axis, the limit of the tilted cases.

    Parameters
    ----------
    pitch : float
        +pi/2 looks straight down, -pi/2 straight up.
    """
    exact = cubify.gravity_to_camera(_tilted_pose(0.3, pitch, 0.0))
    near = cubify.gravity_to_camera(_tilted_pose(0.3, pitch - np.sign(pitch) * 1e-4, 0.0))
    np.testing.assert_allclose(exact, near, atol=1e-3)
    np.testing.assert_allclose(exact.T @ exact, np.eye(3), atol=1e-12)


# ---------------------------------------------------------------------------------------------- boxes


def test_cutr_box_to_world_has_cutr_corners() -> None:
    """World box corners equal CuTR's camera-frame corners moved by ``T_wc``; y-axis is down, size[1] height."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        T_wc = _tilted_pose(*rng.uniform([-np.pi, -0.8, -0.3], [np.pi, 0.8, 0.3]),
                            position=tuple(rng.uniform(-3, 3, 3)))
        centers, dims, rotations = _random_cutr_boxes(rng, T_wc, 5)
        for c, d, R in zip(centers, dims, rotations):
            box = cubify.cutr_box_to_world(c, d, R, T_wc)
            theirs = _cutr_corners(c, d, R) @ T_wc[:3, :3].T + T_wc[:3, 3]
            assert _same_corners(box.corners(), theirs) < 1e-9
            np.testing.assert_allclose(box.size, d)
            np.testing.assert_allclose(box.rotation[:, 1], DOWN, atol=1e-12)
            assert box.extent_along([0.0, 0.0, 1.0]) == pytest.approx(d[1], abs=1e-9)


def test_cutr_box_to_world_within_tolerance_for_float32_outputs() -> None:
    """With float32 model outputs (corners computed in float32, as CuTR does) the match is within 1e-5 m."""
    rng = np.random.default_rng(2)
    T_wc = _tilted_pose(0.4, 0.3, 0.05, position=(4.0, -6.0, 1.5))
    centers, dims, rotations = (a.astype(np.float32) for a in _random_cutr_boxes(rng, T_wc, 30))
    for c, d, R in zip(centers, dims, rotations):
        box = cubify.cutr_box_to_world(c, d, R, T_wc)
        theirs = _cutr_corners(c, d, R).astype(np.float64) @ T_wc[:3, :3].T + T_wc[:3, 3]
        assert _same_corners(box.corners(), theirs) < 1e-5


def test_nearest_rotation_cleans_float32_and_rejects_reflections() -> None:
    """A slightly perturbed rotation becomes orthonormal and barely moves; a reflection is refused."""
    R = Rotation.from_rotvec([0.3, -1.1, 0.4]).as_matrix()
    noisy = R + 1e-7 * np.random.default_rng(3).standard_normal((3, 3))
    clean = cubify.nearest_rotation(noisy)
    np.testing.assert_allclose(clean.T @ clean, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(clean, R, atol=1e-6)
    with pytest.raises(ValueError, match="not a rotation"):
        cubify.nearest_rotation(np.diag([1.0, 1.0, -1.0]))


def test_scale_box2d_keeps_pixel_edges_at_integers() -> None:
    """The whole model image maps to the whole full-resolution image, ``(0, 0, W, H)``; scaling is linear."""
    assert cubify.scale_box2d((0, 0, 768, 1024), (768, 1024), (1440, 1920)) == (0.0, 0.0, 1440.0, 1920.0)
    # The first model pixel [0, 1) covers the first 1.875 full-resolution pixels.
    assert cubify.scale_box2d((0, 0, 1, 1), (768, 1024), (1440, 1920)) == pytest.approx((0, 0, 1.875, 1.875))
    assert cubify.scale_box2d((10, 20, 30, 40), (100, 200), (300, 100)) == pytest.approx((30, 10, 90, 20))


def test_script_geometry_matches_package() -> None:
    """The copies in ``scripts/run_cubify.py`` agree with :mod:`measured_space.cubify` and the geometry module."""
    script = _script()
    rng = np.random.default_rng(4)
    T_wc = _tilted_pose(1.0, 0.2, -0.1, position=(0.5, 2.0, 1.2))
    centers, dims, rotations = _random_cutr_boxes(rng, T_wc, 10)
    for c, d, R in zip(centers, dims, rotations):
        box = cubify.cutr_box_to_world(c, d, R, T_wc)
        center, size, rotation = script.box_to_world(c, d, R, T_wc)
        np.testing.assert_allclose(center, box.center, atol=1e-12)
        np.testing.assert_allclose(size, box.size, atol=1e-12)
        np.testing.assert_allclose(rotation, box.rotation, atol=1e-12)
        np.testing.assert_allclose(script.box_corners(center, size, rotation), box.corners(), atol=1e-12)
        np.testing.assert_allclose(script.nearest_rotation(R), cubify.nearest_rotation(R), atol=1e-15)
    assert script.scale_box2d((1, 2, 3, 4), (7, 9), (11, 13)) == cubify.scale_box2d((1, 2, 3, 4), (7, 9), (11, 13))
    assert (script.MODEL, script.CHECKPOINT_SHA256, script.SCORE_THRESHOLD) == (
        cubify.MODEL, cubify.CHECKPOINT_SHA256, cubify.SCORE_THRESHOLD)


# ---------------------------------------------------------------------------------------------- JSONL


def _frame(frame: str, T_wc: Array) -> dict[str, Any]:
    """A minimal exported frame (as :func:`run_cubify.load_frame` returns it) for record building.

    Parameters
    ----------
    frame : str
        Frame identifier.
    T_wc : numpy.ndarray, shape (4, 4)
        Pose.

    Returns
    -------
    dict
        ``image`` (model input, 1024 x 768 upright), ``T_wc``, ``image_size`` (1440 x 1920) and identity fields.
    """
    return {"image": np.zeros((1024, 768, 3), np.uint8), "T_wc": T_wc, "image_size": np.array([1440, 1920]),
            "dataset": "arkitscenes", "scene": VID, "camera_id": "wide", "frame": frame,
            "view_key": f"arkitscenes/{VID}/wide/{frame}"}


def test_jsonl_round_trip(tmp_path: Path) -> None:
    """Records built by the script read back through :func:`read_predictions` as the package's own boxes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    script = _script()
    rng = np.random.default_rng(5)
    T_wc = _tilted_pose(0.2, 0.35, 0.02)
    centers, dims, rotations = (a.astype(np.float32) for a in _random_cutr_boxes(rng, T_wc, 3))
    scores = np.array([0.9, 0.6, 0.3], np.float32)
    boxes2d = np.array([[0, 0, 768, 1024], [10.5, 20.25, 300, 400], [700, 1000, 768, 1024]], np.float32)
    corners = np.stack([_cutr_corners(c, d, R) for c, d, R in zip(centers, dims, rotations)])
    records = [script.view_record(_frame("10.050", T_wc), centers, dims, rotations, scores, boxes2d, corners, 0.05,
                                  cubify.CHECKPOINT_SHA256, 0.25),
               script.view_record(_frame("10.150", T_wc), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3, 3)),
                                  np.zeros(0), np.zeros((0, 4)), np.zeros((0, 8, 3)), 0.04,
                                  cubify.CHECKPOINT_SHA256, 0.25)]
    path = tmp_path / "arkitscenes" / f"{VID}.jsonl"
    path.parent.mkdir()
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert list(records[0]) == ["view_key", "dataset", "scene", "camera_id", "frame", "model", "checkpoint_sha256",
                                "score_threshold", "image_size", "seconds", "boxes"]

    for source in (path, tmp_path):                                       # one file, or the whole folder
        predictions = cubify.read_predictions(source)
        assert list(predictions) == [f"arkitscenes/{VID}/wide/10.050", f"arkitscenes/{VID}/wide/10.150"]
        assert predictions[f"arkitscenes/{VID}/wide/10.150"] == []
        boxes = predictions[f"arkitscenes/{VID}/wide/10.050"]
        assert len(boxes) == 3
        for b, c, d, R, s, b2 in zip(boxes, centers, dims, rotations, scores, boxes2d):
            expected = cubify.cutr_box_to_world(c, d, R, T_wc)
            np.testing.assert_allclose(b.box.center, expected.center, atol=1e-12)
            np.testing.assert_allclose(b.box.size, expected.size, atol=1e-12)
            np.testing.assert_allclose(b.box.rotation, expected.rotation, atol=1e-12)
            assert b.score == pytest.approx(float(s))
            assert b.box2d == pytest.approx(cubify.scale_box2d(b2, (768, 1024), (1440, 1920)))
            np.testing.assert_allclose(b.center_cam, c, atol=1e-7)
    assert boxes[0].box2d == (0.0, 0.0, 1440.0, 1920.0)
    again = cubify.external_box_from_json(cubify.external_box_json(boxes[1]))
    assert again.box2d == boxes[1].box2d and again.score == boxes[1].score
    np.testing.assert_array_equal(again.box.corners(), boxes[1].box.corners())
    summary = cubify.summarize_predictions(tmp_path)[f"arkitscenes/{VID}"]
    assert (summary["views"], summary["boxes"], summary["min"], summary["max"]) == (2, 3, 0, 3)


def test_view_record_stops_on_corner_mismatch() -> None:
    """A box whose stored corners would differ from CuTR's by more than 1e-5 m stops the run."""
    script = _script()
    T_wc = _tilted_pose(0.0, 0.2, 0.0)
    centers, dims, rotations = _random_cutr_boxes(np.random.default_rng(6), T_wc, 1)
    corners = _cutr_corners(centers[0], dims[0], rotations[0])[None] + [0.0, 0.0, 1e-3]
    with pytest.raises(RuntimeError, match="corners differ"):
        script.view_record(_frame("1", T_wc), centers, dims, rotations, [0.5], [[0, 0, 1, 1]], corners, 0.0,
                           cubify.CHECKPOINT_SHA256, 0.25)


def test_read_predictions_validates(tmp_path: Path) -> None:
    """Missing keys and duplicate views are errors.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    script = _script()
    record = script.view_record(_frame("1", np.eye(4)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3, 3)),
                                np.zeros(0), np.zeros((0, 4)), None, 0.0, cubify.CHECKPOINT_SHA256, 0.25)
    (tmp_path / "a.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "b.jsonl").write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="duplicate view_key"):
        cubify.read_predictions(tmp_path)
    broken = {k: v for k, v in record.items() if k != "image_size"}
    (tmp_path / "c.jsonl").write_text(json.dumps(broken) + "\n")
    with pytest.raises(ValueError, match="missing image_size"):
        cubify.read_predictions(tmp_path / "c.jsonl")


def test_scene_done_requires_every_view_and_the_same_configuration(tmp_path: Path) -> None:
    """The script's resume check: complete files are skipped, partial ones redone, mixed settings refused.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    script = _script()
    path = tmp_path / "s.jsonl"
    lines = [{"view_key": k, "checkpoint_sha256": "abc", "score_threshold": 0.25} for k in ("a", "b")]
    assert not script.scene_done(path, ["a", "b"], "abc", 0.25)
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    assert script.scene_done(path, ["b", "a"], "abc", 0.25)
    assert not script.scene_done(path, ["a", "b", "c"], "abc", 0.25)
    with pytest.raises(RuntimeError, match="another checkpoint or score threshold"):
        script.scene_done(path, ["a", "b"], "abc", 0.3)
    path.write_text(json.dumps(lines[0]) + "\n" + '{"view_key": "b", "checkpo')
    assert not script.scene_done(path, ["a", "b"], "abc", 0.25)


# --------------------------------------------------------------------------------------------- export


def _png(array: np.ndarray) -> bytes:
    """Encode an array as PNG.

    Parameters
    ----------
    array : numpy.ndarray
        ``uint8`` (H, W, 3) or (H, W), or ``uint16`` (H, W).

    Returns
    -------
    bytes
        The PNG file.
    """
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake ARKitScenes folder with one sideways video, used through the configured data root.

    LiDAR depth is 2.0 m (with one invalid pixel) and laser depth 1.5 m, so a frame shows which one it holds.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Used to point ``MEASURED_SPACE_DATA`` at ``tmp_path``.

    Returns
    -------
    pathlib.Path
        The dataset folder ``tmp_path/data/arkitscenes``.
    """
    root = tmp_path / "data" / "arkitscenes"
    for sub in ("annotations", "traj", "upsampling", "wide_intrinsics"):
        (root / sub).mkdir(parents=True)
    monkeypatch.setenv("MEASURED_SPACE_DATA", str(tmp_path))
    rows = []
    for k, t in enumerate(TRAJ_TIMES):
        pose = make_pose(axis_angle_to_matrix([0.0, 0.0, 0.2 * k]) @ SIDEWAYS @ _rot("y", 0.1), [0.3 * k, 0.0, 1.4])
        T_cw = invert_pose(pose)                                       # the file stores world-to-camera
        rows.append(" ".join(f"{x:.10f}" for x in (t, *Rotation.from_matrix(T_cw[:3, :3]).as_rotvec(), *T_cw[:3, 3])))
    (root / "traj" / f"{VID}_lowres_wide.traj").write_text("\n".join(rows) + "\n")
    c = SENSOR_CAMERA
    rng = np.random.default_rng(7)
    lidar = np.full((3, 4), LIDAR_MM, np.uint16)
    lidar[0, 0] = 0
    with zipfile.ZipFile(root / "upsampling" / f"{VID}.zip", "w") as z, \
            zipfile.ZipFile(root / "wide_intrinsics" / f"{VID}_wide_intrinsics.zip", "w") as zi:
        for stamp in STAMPS:
            z.writestr(f"{VID}/wide/{VID}_{stamp}.png", _png(rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)))
            z.writestr(f"{VID}/highres_depth/{VID}_{stamp}.png", _png(np.full((6, 8), LASER_MM, np.uint16)))
            z.writestr(f"{VID}/lowres_depth/{VID}_{stamp}.png", _png(lidar))
            z.writestr(f"{VID}/confidence/{VID}_{stamp}.png", _png(np.full((3, 4), 2, np.uint8)))
            zi.writestr(f"wide_intrinsics/{VID}_{stamp}.pincam", f"{c.width} {c.height} {c.fx} {c.fy} {c.cx} {c.cy}\n")
    return root


def test_export_view_writes_the_cutr_frame(dataset: Path, tmp_path: Path) -> None:
    """The frame holds the upright image at 4x the LiDAR size, LiDAR (never laser) depth, both K, poses, identity.

    Parameters
    ----------
    dataset : pathlib.Path
        The fake dataset folder.
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    view = ark.list_frames(VID)[0]
    assert (view.camera.width, view.camera.height) == (6, 8)                # turned upright
    path = cubify.export_view(view, cubify.frame_path(tmp_path / "frames", view))
    assert path == tmp_path / "frames" / "arkitscenes" / VID / f"wide_{view.frame}.npz"
    with np.load(path) as d:
        assert d["image"].shape == (16, 12, 3) and d["image"].dtype == np.uint8
        expected = np.asarray(Image.fromarray(ark.load_rgb(view)).resize((12, 16), Image.Resampling.BICUBIC))
        np.testing.assert_array_equal(d["image"], expected)
        assert d["depth"].shape == (4, 3) and d["depth"].dtype == np.float32
        np.testing.assert_array_equal(d["depth"], ark.load_lidar_depth(view))
        assert set(np.unique(d["depth"])) == {0.0, LIDAR_MM / 1000}           # no laser value anywhere
        np.testing.assert_allclose(d["K_img"], view.camera.scaled(12, 16).K)
        np.testing.assert_allclose(d["K_depth"], view.camera.scaled(3, 4).K)
        np.testing.assert_array_equal(d["T_wc"], view.T_wc)
        np.testing.assert_array_equal(d["T_gravity"], cubify.gravity_to_camera(view.T_wc))
        assert d["image_size"].tolist() == [6, 8]
        assert str(d["view_key"]) == view_key(view) and str(d["depth_source"]) == "lidar"
        assert (str(d["dataset"]), str(d["scene"]), str(d["camera_id"]), str(d["frame"])) == (
            "arkitscenes", VID, "wide", view.frame)
    frame = _script().load_frame(path)                                      # what the CuTR side reads
    assert frame["view_key"] == view_key(view) and frame["image"].shape == (16, 12, 3)


def test_export_view_refuses_sensor_orientation(dataset: Path, tmp_path: Path) -> None:
    """CuTR gets upright images only.

    Parameters
    ----------
    dataset : pathlib.Path
        The fake dataset folder.
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    with pytest.raises(ValueError, match="upright ARKitScenes"):
        cubify.export_view(ark.list_frames(VID, upright=False)[0], tmp_path / "x.npz")


def test_export_videos_resumes_per_video(dataset: Path, tmp_path: Path) -> None:
    """A complete video is skipped; an incomplete one is redone; the script groups its frames in time order.

    Parameters
    ----------
    dataset : pathlib.Path
        The fake dataset folder.
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    """
    frames = tmp_path / "frames"
    assert cubify.export_videos([VID], frames, log=False) == {"videos": 1, "skipped": 0, "exported": 1, "views": 2}
    marker = json.loads((frames / "arkitscenes" / VID / cubify.COMPLETE_MARKER).read_text())
    assert marker["files"] == [f"wide_{s}.npz" for s in STAMPS] and marker["depth_source"] == "lidar"
    first = frames / "arkitscenes" / VID / f"wide_{STAMPS[0]}.npz"
    stamp = first.stat().st_mtime_ns
    assert cubify.export_videos([VID], frames, log=False)["skipped"] == 1
    assert first.stat().st_mtime_ns == stamp
    (frames / "arkitscenes" / VID / cubify.COMPLETE_MARKER).write_text(json.dumps({"files": ["wide_x.npz"]}))
    assert cubify.export_videos([VID], frames, log=False)["exported"] == 1
    groups = _script().group_frames([frames])
    assert list(groups) == [("arkitscenes", VID)]
    assert [k for k, _ in groups[("arkitscenes", VID)]] == [f"arkitscenes/{VID}/wide/{s}" for s in STAMPS]


def test_main_export(dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command line exports from an id file and refuses an empty id list.

    Parameters
    ----------
    dataset : pathlib.Path
        The fake dataset folder.
    tmp_path : pathlib.Path
        Pytest's temporary directory.
    capsys : pytest.CaptureFixture
        Captures the printed summary.
    """
    ids = tmp_path / "ids.txt"
    ids.write_text(f"# one video\n{VID}\n\n")
    assert cubify.main(["export", "--videos-file", str(ids), "--out", str(tmp_path / "f")]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["views"] == 2
    assert cubify.main(["export", "--out", str(tmp_path / "f")]) == 2
