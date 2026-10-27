"""Tests for the ARKitScenes loader: file parsing, conventions, upright views and the laser-fit diagnostic.

Synthetic tests write a tiny fake dataset (annotation JSON, trajectory, zip archives) to ``tmp_path``; the
real-data test skips cleanly when the data root is not available.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from measured_space.config import dataset_dir
from measured_space.datasets import arkitscenes as ark
from measured_space.geometry import Array, Camera, OrientedBox, axis_angle_to_matrix, invert_pose, make_pose
from measured_space.scene import FrameView
from measured_space.visibility import object_in_view

VID = "1000"
TRAJ_TIMES = [10.0, 10.1, 10.2, 11.0, 11.1]               # a 0.8 s gap between 10.2 and 11.0
STAMPS = ["9.950", "10.050", "10.200", "10.600", "11.050", "11.300"]
KEPT = ["10.050", "10.200", "11.050"]                      # before span, in the gap, after span are dropped
SENSOR_CAMERA = Camera(10.0, 11.0, 3.0, 2.0, 8, 6)         # off-centre principal point: turns are visible
# A horizontal OpenCV camera held sideways, as in a "Left" video: x right = world -z, y down = world -x,
# z forward = world +y, so world up points to the image's left.
SIDEWAYS = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def _yaw(theta: float) -> Array:
    """Rotation about world z.

    Parameters
    ----------
    theta : float
        Angle in radians.

    Returns
    -------
    numpy.ndarray, shape (3, 3), float64
        The rotation matrix.
    """
    return axis_angle_to_matrix([0.0, 0.0, theta])


def _true_poses() -> list[Array]:
    """Ground-truth camera-to-world poses at :data:`TRAJ_TIMES`: a sideways camera yawing and moving.

    Returns
    -------
    list of numpy.ndarray, shape (4, 4), float64
        One pose per trajectory sample.
    """
    return [make_pose(_yaw(0.3 * k) @ SIDEWAYS, [0.5 * k, -0.2 * k, 1.4]) for k in range(len(TRAJ_TIMES))]


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
    """A fake ARKitScenes folder with one video, used through the configured data root.

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
    obb = {"centroid": [1.0, 2.0, 0.4], "axesLengths": [0.6, 1.8, 0.8],
           "normalizedAxes": _yaw(np.radians(30)).T.ravel().tolist()}   # rows are the box axes
    records = [{"uid": "aaa", "label": "sofa", "objectId": 7, "segments": {"obbAligned": obb}},
               {"uid": "bbb", "label": "chair", "objectId": 7, "segments": {"obbAligned": obb}}]
    (root / "annotations" / f"{VID}_3dod_annotation.json").write_text(json.dumps({"data": records}))
    rows = []
    for t, pose in zip(TRAJ_TIMES, _true_poses()):
        T_cw = invert_pose(pose)                                       # the file stores world-to-camera
        rotvec = Rotation.from_matrix(T_cw[:3, :3]).as_rotvec()
        rows.append(" ".join(f"{x:.10f}" for x in (t, *rotvec, *T_cw[:3, 3])))
    (root / "traj" / f"{VID}_lowres_wide.traj").write_text("\n".join(rows) + "\n")
    c = SENSOR_CAMERA
    with zipfile.ZipFile(root / "upsampling" / f"{VID}.zip", "w") as z, \
            zipfile.ZipFile(root / "wide_intrinsics" / f"{VID}_wide_intrinsics.zip", "w") as zi:
        for k, stamp in enumerate(STAMPS):
            z.writestr(f"{VID}/wide/{VID}_{stamp}.png", _png(np.full((6, 8, 3), k, np.uint8)))
            z.writestr(f"{VID}/highres_depth/{VID}_{stamp}.png", _png(np.full((6, 8), 1500 + k, np.uint16)))
            z.writestr(f"{VID}/lowres_depth/{VID}_{stamp}.png", _png(np.full((3, 4), 2000, np.uint16)))
            z.writestr(f"{VID}/confidence/{VID}_{stamp}.png", _png(np.full((3, 4), 2, np.uint8)))
            zi.writestr(f"wide_intrinsics/{VID}_{stamp}.pincam", f"{c.width} {c.height} {c.fx} {c.fy} {c.cx} {c.cy}\n")
    return root


def test_load_objects_uses_uid_and_rows_as_box_axes(dataset: Path) -> None:
    """Rows of ``normalizedAxes`` become the box axes, matching the official ``R.T @ corners`` construction.

    Parameters
    ----------
    dataset : pathlib.Path
        The fake dataset folder.
    """
    objects = ark.load_objects(VID)
    assert [o.object_id for o in objects] == ["aaa", "bbb"] and objects[0].label == "sofa"
    R = _yaw(np.radians(30)).T                                         # as stored: one axis per row
    np.testing.assert_allclose(objects[0].box.rotation, R.T, atol=1e-12)
    half = np.array([0.3, 0.9, 0.4])
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    official = (R.T @ (signs * half).T).T + [1.0, 2.0, 0.4]
    ours = objects[0].box.corners()
    assert np.max(np.min(np.linalg.norm(official[:, None] - ours[None], axis=2), axis=1)) < 1e-12


def test_trajectory_is_inverted_to_camera_to_world(dataset: Path) -> None:
    """Stored world-to-camera rows come back as the camera-to-world poses they were built from.

    Parameters
    ----------
    dataset : pathlib.Path
        The fake dataset folder.
    """
    times, T_wc = ark.load_trajectory(VID)
    np.testing.assert_allclose(times, TRAJ_TIMES)
    np.testing.assert_allclose(T_wc, np.stack(_true_poses()), atol=1e-8)
    assert ark.sky_direction_from_pose(T_wc[0]) == "Left"
    with pytest.raises(ValueError):
        ark.parse_trajectory("1.0 0 0 0 0 0\n")


def test_list_frames_interpolates_and_drops_unposed_frames(dataset: Path) -> None:
    """Only frames inside the trajectory span and not inside a long gap are kept, with interpolated poses.

    Parameters
    ----------
    dataset : pathlib.Path
        The fake dataset folder.
    """
    assert ark.wide_timestamps(VID) == STAMPS
    views = ark.list_frames(VID, upright=False)
    assert [v.frame for v in views] == KEPT
    view = views[0]
    assert (view.dataset, view.scene, view.camera_id) == ("arkitscenes", VID, "wide_sensor")
    assert view.image_ref == f"upsampling/{VID}.zip::{VID}/wide/{VID}_10.050.png"
    assert view.camera == SENSOR_CAMERA
    np.testing.assert_allclose(view.up, [0, 0, 1])
    poses = _true_poses()
    np.testing.assert_allclose(view.camera_center, (poses[0][:3, 3] + poses[1][:3, 3]) / 2, atol=1e-8)
    np.testing.assert_allclose(view.T_wc[:3, :3], _yaw(0.15) @ SIDEWAYS, atol=1e-8)
    np.testing.assert_allclose(views[1].T_wc, poses[2], atol=1e-8)
    assert [v.frame for v in ark.list_frames(VID, max_gap=1.0)] == ["10.050", "10.200", "10.600", "11.050"]


def test_upright_views_turn_camera_pose_and_images_together(dataset: Path) -> None:
    """Default views describe the upright image: world up points up it, and every loader turns its array.

    Parameters
    ----------
    dataset : pathlib.Path
        The fake dataset folder.
    """
    assert ark.upright_quarter_turns(VID) == ark.quarter_turns_for_sky("Left") == 3
    sensor, upright = ark.list_frames(VID, upright=False)[0], ark.list_frames(VID)[0]
    assert upright.camera_id == "wide" and (upright.camera.width, upright.camera.height) == (6, 8)
    assert ark.sky_direction_from_pose(upright.T_wc) == "Up"
    assert ark.up_angle_in_image(upright.T_wc) < 45.0 < ark.up_angle_in_image(sensor.T_wc)
    np.testing.assert_allclose(upright.camera_center, sensor.camera_center)
    assert ark.load_rgb(upright).shape == (8, 6, 3) and ark.load_rgb(sensor).shape == (6, 8, 3)
    assert ark.load_laser_depth(upright).shape == (8, 6)
    assert ark.load_lidar_depth(upright).shape == ark.load_confidence(upright).shape == (4, 3)


@pytest.mark.parametrize("k", range(4))
def test_rotate_camera_matches_numpy_rot90(k: int) -> None:
    """A world point seen at pixel (u, v) in the sensor image is seen at the same pixel content after turning.

    Parameters
    ----------
    k : int
        Counter-clockwise quarter turns.
    """
    rng = np.random.default_rng(k)
    camera = SENSOR_CAMERA
    T_wc = make_pose(axis_angle_to_matrix(rng.normal(size=3)), rng.normal(size=3))
    image = np.arange(camera.height * camera.width).reshape(camera.height, camera.width)
    turned_camera, R = ark.rotate_camera(camera, k)
    turned_pose = T_wc @ np.block([[R.T, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
    turned_image = np.rot90(image, k)
    assert turned_image.shape == (turned_camera.height, turned_camera.width)
    v, u = np.mgrid[0:camera.height, 0:camera.width]
    world = (np.c_[camera.backproject(u, v, rng.uniform(1, 5, u.size)), np.ones(u.size)] @ T_wc.T)[:, :3]
    points_cam = (np.c_[world, np.ones(len(world))] @ np.linalg.inv(turned_pose).T)[:, :3]
    uv, _ = turned_camera.project(points_cam)
    cols, rows = np.round(uv).astype(int).T
    np.testing.assert_allclose(uv, np.c_[cols, rows], atol=1e-9)    # pixel centres map to pixel centres
    np.testing.assert_array_equal(turned_image[rows, cols], image.ravel())


def test_image_loaders_decode_units(dataset: Path) -> None:
    """Depth PNGs in millimetres load as float32 metres; RGB and confidence load as stored.

    Parameters
    ----------
    dataset : pathlib.Path
        The fake dataset folder.
    """
    view = ark.list_frames(VID, upright=False)[0]                      # STAMPS index 1
    rgb = ark.load_rgb(view)
    assert rgb.dtype == np.uint8 and np.all(rgb == 1)
    laser = ark.load_laser_depth(view)
    assert laser.dtype == np.float32
    np.testing.assert_allclose(laser, 1.501)
    np.testing.assert_allclose(ark.load_lidar_depth(view), 2.0)
    confidence = ark.load_confidence(view)
    assert confidence.dtype == np.uint8 and np.all(confidence == 2)


def test_laser_box_fit_region_and_signed_residuals() -> None:
    """Exact counts and residual signs for a box seen face-on against a far wall."""
    camera = Camera(40.0, 40.0, 31.5, 23.5, 64, 48)
    view = FrameView("synthetic", "s", "c", "0", camera, np.eye(4), "none")
    u, v = np.meshgrid(np.arange(64), np.arange(48))
    x, y = (u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy  # ray slopes
    front = (np.abs(x) <= 0.2) & (np.abs(y) <= 0.2)                    # unit cube at z = 3: front face 2.5 m
    depth = np.where(front, 2.5, 10.0)
    points, rays = ark.depth_rays(view, depth)
    fit = ark.laser_box_fit(view.camera_center, points, rays, OrientedBox([0, 0, 3.0], [1, 1, 1], np.eye(3)))
    grown_cone = (np.abs(x) <= 0.55 / 2.45) & (np.abs(y) <= 0.55 / 2.45)
    assert (fit.n_region, fit.n_inside) == (grown_cone.sum(), front.sum())
    np.testing.assert_allclose(fit.surface_distance, 0.0, atol=1e-12)
    np.testing.assert_allclose(fit.face_normal, np.tile([0, 0, -1.0], (fit.n_inside, 1)))
    # Push the box 3 cm away: the laser surface now sits 3 cm outside its front face.
    shifted = ark.laser_box_fit(view.camera_center, points, rays, OrientedBox([0, 0, 3.03], [1, 1, 1], np.eye(3)))
    np.testing.assert_allclose(shifted.face_residual, 0.03, atol=1e-9)
    np.testing.assert_allclose(shifted.surface_distance, 0.03, atol=1e-9)


def test_real_video_counts_orientation_and_laser_alignment() -> None:
    """Real data: frame counts, upright orientation vs the dataset's metadata, and a loose box-to-laser bound."""
    try:
        dataset_dir("arkitscenes")
    except RuntimeError as error:
        pytest.skip(str(error))
    for video_id, sky in ark.SKY_DIRECTION.items():
        assert ark.upright_quarter_turns(video_id) == ark.quarter_turns_for_sky(sky), video_id
    video = "42444946"
    stamps, views, objects = ark.wide_timestamps(video), ark.list_frames(video), ark.load_objects(video)
    assert (len(stamps), len(views), len(objects)) == (31, 30, 8)
    assert stamps[0] not in {v.frame for v in views}                  # precedes the first pose
    # Upright: world up within 45 deg of image up wherever it has an image-plane direction (camera < 60 deg
    # from level). One frame of this video looks almost straight down and is excluded by this.
    level = [v for v in views if np.linalg.norm((v.T_wc[:3, :3].T @ ark.WORLD_UP)[:2]) >= 0.5]
    assert len(level) >= 25 and all(ark.up_angle_in_image(v.T_wc) < 45.0 for v in level)
    assert ark.load_rgb(views[0]).shape == (1920, 1440, 3)
    # Measured 2026-10-26 with these settings (upright views): 22 visible (frame, object) pairs, median
    # explained 0.861.
    explained = []
    for view in views:
        laser = ark.load_laser_depth(view)
        points, rays = ark.depth_rays(view, laser, stride=8)
        for obj in objects:
            if object_in_view(view, obj, laser)[0].visible_fraction >= 0.1:
                explained.append(ark.laser_box_fit(view.camera_center, points, rays, obj.box).explained)
    assert len(explained) >= 15
    assert np.median(explained) > 0.6
