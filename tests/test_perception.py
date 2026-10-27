"""Tests for the perception stage: box lifting on ray-cast synthetic depth (CPU), and opt-in model smoke tests.

The synthetic scene is a yawed box standing on a floor, seen from above and to one side so that its top and
two side faces are visible; the visible points then span the box's full extent on every axis, and lifting
should recover it up to erosion and percentile trimming. GPU tests load SAM 3 and MoGe-2 and run on one
ARKitScenes frame; they are skipped unless ``MEASURED_SPACE_GPU_TESTS=1``.
"""

from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pytest
from numpy.typing import NDArray

from measured_space.geometry import Array, Camera, OrientedBox, axis_angle_to_matrix, make_pose
from measured_space.perception import (
    LiftError,
    labels_for_phrase,
    lift_instance,
    mask_features,
    prompt_phrases,
    rot90_camera,
    rot90_points,
)

UP = np.array([0.0, 0.0, 1.0])
CAMERA = Camera(520.0, 520.0, 319.5, 239.5, 640, 480)
TRUE_BOX = OrientedBox([0.2, 0.1, 0.45], [1.6, 0.8, 0.9], axis_angle_to_matrix([0.0, 0.0, np.radians(30.0)]))


def look_at(eye: Array, target: Array, up: Array = UP) -> Array:
    """Camera-to-world pose of an OpenCV camera at ``eye`` looking at ``target``.

    Parameters
    ----------
    eye : numpy.ndarray, shape (3,), float
        Camera centre (m).
    target : numpy.ndarray, shape (3,), float
        Point on the optical axis (m).
    up : numpy.ndarray, shape (3,), float, optional
        World up; image "up" (camera -y) is kept on its side.

    Returns
    -------
    numpy.ndarray, shape (4, 4), float
        ``T_wc``.
    """
    forward = np.asarray(target, float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return make_pose(np.column_stack([right, np.cross(forward, right), forward]), eye)


T_WC = look_at(np.array([-1.0, -2.8, 1.7]), TRUE_BOX.center)


def render(box: OrientedBox, camera: Camera, T_wc: Array, far: float = 10.0) -> tuple[Array, NDArray[np.bool_]]:
    """Ray-cast planar depth and the box's mask for a box standing on the floor ``z = 0``.

    Parameters
    ----------
    box : OrientedBox
        Box in the world frame.
    camera : Camera
        Intrinsics.
    T_wc : numpy.ndarray, shape (4, 4), float
        Camera-to-world pose.
    far : float, optional
        Depth (m) given to rays that hit nothing.

    Returns
    -------
    depth : numpy.ndarray, shape (H, W), float64
        Planar depth (m).
    mask : numpy.ndarray, shape (H, W), bool
        Pixels whose nearest hit is the box.
    """
    v, u = np.mgrid[0:camera.height, 0:camera.width]
    rays = np.stack([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, np.ones_like(u, float)], -1)
    rays = rays.reshape(-1, 3) @ T_wc[:3, :3].T          # camera z = 1, so the ray parameter is planar depth
    origin = T_wc[:3, 3]
    o_local, d_local = (origin - box.center) @ box.rotation, rays @ box.rotation
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-box.size / 2 - o_local) / d_local, (box.size / 2 - o_local) / d_local
        t_near, t_far = np.minimum(t1, t2).max(axis=1), np.maximum(t1, t2).min(axis=1)
        t_box = np.where((t_far >= t_near) & (t_near > 0), t_near, np.inf)
        t_floor = np.where(rays[:, 2] < 0, -origin[2] / rays[:, 2], np.inf)
    depth = np.minimum(t_box, t_floor)
    mask = np.isfinite(t_box) & (t_box <= t_floor)
    depth[~np.isfinite(depth)] = far
    return depth.reshape(camera.height, camera.width), mask.reshape(camera.height, camera.width)


def box_errors(estimate: OrientedBox, truth: OrientedBox) -> tuple[float, Array, float]:
    """Centre distance, per-axis size error and yaw error between two yaw-only boxes.

    Parameters
    ----------
    estimate : OrientedBox
        Lifted box, x its longer horizontal side.
    truth : OrientedBox
        Reference box, x its longer horizontal side.

    Returns
    -------
    center_error : float
        Distance between centres (m).
    size_error : numpy.ndarray, shape (3,), float
        ``estimate.size - truth.size`` (m).
    yaw_error_deg : float
        Angle between the x axes, modulo 180 degrees.
    """
    cos = abs(float(estimate.rotation[:, 0] @ truth.rotation[:, 0]))
    return (float(np.linalg.norm(estimate.center - truth.center)), estimate.size - truth.size,
            float(np.degrees(np.arccos(min(cos, 1.0)))))


def test_lift_recovers_known_box_with_non_identity_pose() -> None:
    """A clean render lifts to the true box, and every feature is a float with a sensible value.

    ``erode_px=1`` at this 640-px test resolution erodes about the same angle as the default 3 px on the
    1920-px frames the pipeline sees; at grazing views (the top face here) erosion costs centimetres.
    """
    depth, mask = render(TRUE_BOX, CAMERA, T_WC)
    box, features = lift_instance(mask, depth, CAMERA, T_WC, UP, erode_px=1)
    center_error, size_error, yaw_error = box_errors(box, TRUE_BOX)
    assert center_error < 0.05
    assert np.all(np.abs(size_error) < 0.07)
    assert yaw_error < 1.0
    np.testing.assert_allclose(box.rotation[:, 2], UP)
    assert np.isclose(np.linalg.det(box.rotation), 1.0)

    assert all(isinstance(value, float) for value in features.values())
    assert features["mask_area_px"] == mask.sum()
    assert 0 < features["eroded_fraction"] < 1 and features["erosion_fallback"] == 0
    assert features["valid_depth_fraction"] == 1 and features["inlier_fraction"] > 0.95
    assert features["border_touch"] == 0 and features["n_components"] == 1
    assert 0 < features["boundary_fraction"] < 0.5
    assert features["view_elevation_deg"] > 0                     # the camera looks down at the box
    assert features["footprint_hull_fill"] > 0.9                 # the top face covers the footprint


def test_lift_is_robust_to_outlier_depth_and_mask_bleed() -> None:
    """Gross depth outliers and a mask bleeding onto the background barely move the lifted box.

    The same corruption ruins a naive lift (no erosion, no MAD test, min/max extents), so the test is not
    passing on harmless input.
    """
    import cv2

    depth, mask = render(TRUE_BOX, CAMERA, T_WC)
    bled = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)   # 3 px onto background
    noisy = depth.copy()
    rows, cols = np.nonzero(mask)
    pick = np.random.default_rng(1).choice(len(rows), int(0.05 * len(rows)), replace=False)
    noisy[rows[pick[::2]], cols[pick[::2]]] += 2.5                 # flying pixels far behind
    noisy[rows[pick[1::2]], cols[pick[1::2]]] *= 0.5               # spurious returns well in front

    box, features = lift_instance(bled, noisy, CAMERA, T_WC, UP)
    center_error, size_error, yaw_error = box_errors(box, TRUE_BOX)
    assert center_error < 0.04
    assert np.all(np.abs(size_error) < 0.06)
    assert yaw_error < 2.0
    assert features["inlier_fraction"] < 0.97                    # the MAD test did remove points

    naive, _ = lift_instance(bled, noisy, CAMERA, T_WC, UP, erode_px=0, mad_k=None,
                             height_percentiles=(0, 100), extent_percentiles=(0, 100))
    assert np.max(np.abs(naive.size - TRUE_BOX.size)) > 0.5


def test_lift_in_camera_frame_matches_world_lift() -> None:
    """Lifting in the camera frame (identity pose, up rotated into the camera) gives the same physical box."""
    depth, mask = render(TRUE_BOX, CAMERA, T_WC)
    world_box, _ = lift_instance(mask, depth, CAMERA, T_WC, UP)
    up_cam = T_WC[:3, :3].T @ UP
    cam_box, _ = lift_instance(mask, depth, CAMERA, np.eye(4), up_cam)
    center_error, size_error, yaw_error = box_errors(cam_box.transformed(T_WC), world_box)
    assert center_error < 0.005 and np.all(np.abs(size_error) < 0.005) and yaw_error < 0.3


def test_lift_rejects_masks_without_enough_depth() -> None:
    """A mask with no valid depth, or an empty mask, raises :class:`LiftError` instead of fitting noise."""
    depth, mask = render(TRUE_BOX, CAMERA, T_WC)
    with pytest.raises(LiftError):
        lift_instance(mask, np.where(mask, 0.0, depth), CAMERA, T_WC, UP)
    with pytest.raises(LiftError):
        lift_instance(np.zeros_like(mask), depth, CAMERA, T_WC, UP)


def test_mask_features_edge_band_and_truncation() -> None:
    """Edge-band share, pieces and border contact match hand counts on small masks."""
    mask = np.zeros((40, 50), bool)
    mask[10:30, 15:35] = True                                     # 20 x 20 interior square
    features = mask_features(mask, boundary_px=5)
    assert features["boundary_fraction"] == pytest.approx(300 / 400)   # all but the inner 10 x 10
    assert features["border_touch"] == 0 and features["n_components"] == 1
    mask[0:5, 0:5] = True                                         # a second piece in the top-left corner
    features = mask_features(mask, boundary_px=5)
    assert features["border_touch"] == 1 and features["n_border_sides"] == 2
    assert features["n_components"] == 2


@pytest.mark.parametrize("k", [-1, 1, 2])
def test_rot90_points_follows_numpy_rot90(k: int) -> None:
    """A pixel centre maps to where :func:`numpy.rot90` moves that pixel.

    Parameters
    ----------
    k : int
        Quarter turns, as in :func:`numpy.rot90`.
    """
    image = np.zeros((7, 11), int)
    image[2, 9] = 1
    r, c = np.argwhere(np.rot90(image, k) == 1)[0]
    np.testing.assert_allclose(rot90_points([[9.5, 2.5]], k, image.shape), [[c + 0.5, r + 0.5]])


@pytest.mark.parametrize("k", [-1, 1, 2])
def test_rot90_camera_matches_rotated_projection(k: int) -> None:
    """The rotated camera projects points (rotated about the optical axis) where the rotated image has them.

    Parameters
    ----------
    k : int
        Quarter turns, as in :func:`numpy.rot90`.
    """
    camera = Camera(1590.0, 1580.0, 961.2, 718.4, 1920, 1440)
    points = np.random.default_rng(2).uniform([-1, -1, 1], [1, 1, 4], size=(20, 3))
    rotated_points = points.copy()
    for _ in range(k % 4):                                        # one counter-clockwise image quarter turn
        rotated_points = np.stack([rotated_points[:, 1], -rotated_points[:, 0], rotated_points[:, 2]], axis=1)
    uv, _ = camera.project(points)
    uv_rotated, _ = rot90_camera(camera, k).project(rotated_points)
    np.testing.assert_allclose(uv_rotated + 0.5, rot90_points(uv + 0.5, k, (camera.height, camera.width)),
                               atol=1e-9)


def test_class_phrase_mapping() -> None:
    """Every ARKitScenes class has a phrase, and shared robot phrases map back to all their labels."""
    phrases = prompt_phrases("arkitscenes")
    assert len(phrases) == 17 and "tv" in phrases and "tv_monitor" not in phrases
    assert labels_for_phrase("arkitscenes", "tv") == ("tv_monitor",)
    assert labels_for_phrase("smartspaces", "mobile robot") == ("NovaCarter", "Transporter")
    assert labels_for_phrase("smartspaces", "humanoid robot") == ("AgilityDigit", "FourierGR1T2")
    assert labels_for_phrase("smartspaces", "pallet jack") == ("PalletTruck",)
    assert labels_for_phrase("smartspaces", "chair") == ()


# ---------------------------------------------------------------------------------------------------------
# Opt-in GPU smoke tests on one real frame
# ---------------------------------------------------------------------------------------------------------

gpu = pytest.mark.skipif(os.environ.get("MEASURED_SPACE_GPU_TESTS") != "1",
                         reason="set MEASURED_SPACE_GPU_TESTS=1 to run model tests")
SOFA_FRAME = "42444946_223632.654"                                # a sofa fills the view; sky is image-left


def load_frame(name: str) -> tuple[NDArray[np.uint8], Array, Camera]:
    """RGB, laser depth (m) and intrinsics of one ARKitScenes upsampling frame.

    Parameters
    ----------
    name : str
        ``<video id>_<timestamp>``.

    Returns
    -------
    image : numpy.ndarray, shape (1440, 1920, 3), uint8
        RGB.
    depth : numpy.ndarray, shape (1440, 1920), float64
        Laser depth in metres; 0 where missing.
    camera : Camera
        Intrinsics from the frame's ``.pincam``.
    """
    from PIL import Image

    from measured_space.config import dataset_dir

    root = dataset_dir("arkitscenes")
    video = name.split("_")[0]
    with zipfile.ZipFile(root / "upsampling" / f"{video}.zip") as z:
        image = np.array(Image.open(io.BytesIO(z.read(f"{video}/wide/{name}.png"))).convert("RGB"))
        depth = np.array(Image.open(io.BytesIO(z.read(f"{video}/highres_depth/{name}.png")))) / 1000.0
    with zipfile.ZipFile(root / "wide_intrinsics" / f"{video}_wide_intrinsics.zip") as z:
        w, h, fx, fy, cx, cy = map(float, z.read(f"wide_intrinsics/{name}.pincam").split())
    return image, depth, Camera(fx, fy, cx, cy, int(w), int(h))


@gpu
def test_sam3_finds_the_sofa() -> None:
    """SAM 3 finds the sofa that fills a real frame, with a full-resolution mask and an in-image box."""
    from measured_space.perception import segment_instances

    image, _, _ = load_frame(SOFA_FRAME)
    instances = segment_instances(image, ["sofa", "refrigerator"], 0.5, rot90_k=-1)
    sofas = [inst for inst in instances if inst.phrase == "sofa"]
    assert sofas and sofas[0].mask.shape == image.shape[:2] and sofas[0].mask.sum() > 0.1 * sofas[0].mask.size
    x0, y0, x1, y1 = sofas[0].box2d
    assert 0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]


@gpu
def test_moge_depth_is_metric() -> None:
    """MoGe-2 depth on the upright frame agrees with the laser depth to within 10 % (median relative)."""
    from measured_space.perception import estimate_depth_moge

    image, laser, camera = load_frame(SOFA_FRAME)
    depth = estimate_depth_moge(image, camera, rot90_k=-1)
    both = (laser > 0) & np.isfinite(depth)
    assert depth.shape == laser.shape and both.mean() > 0.5
    assert np.median(np.abs(depth[both] - laser[both]) / laser[both]) < 0.1
