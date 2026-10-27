import numpy as np
import pytest

from measured_space.geometry import (
    Camera,
    OrientedBox,
    axis_angle_to_matrix,
    center_distance,
    interpolate_poses,
    invert_pose,
    make_pose,
    surface_distance,
    transform,
)

RNG = np.random.default_rng(0)


def random_rotation() -> np.ndarray:
    return axis_angle_to_matrix(RNG.normal(size=3))


def yaw(theta: float) -> np.ndarray:
    return axis_angle_to_matrix([0.0, 0.0, theta])


def test_invert_pose_round_trip():
    pose = make_pose(random_rotation(), RNG.normal(size=3))
    np.testing.assert_allclose(pose @ invert_pose(pose), np.eye(4), atol=1e-12)
    points = RNG.normal(size=(5, 3))
    np.testing.assert_allclose(transform(invert_pose(pose), transform(pose, points)), points, atol=1e-12)


def test_axis_angle_quarter_turn_about_z_maps_x_to_y():
    np.testing.assert_allclose(axis_angle_to_matrix([0, 0, np.pi / 2]) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_interpolate_poses_endpoints_midpoint_and_range():
    times = np.array([0.0, 1.0])
    poses = np.stack([make_pose(np.eye(3), [0, 0, 0]), make_pose(yaw(np.pi / 2), [2, 0, 0])])
    out, valid = interpolate_poses(times, poses, [0.0, 0.5, 1.0, -0.1, 1.1])
    assert valid.tolist() == [True, True, True, False, False]
    np.testing.assert_allclose(out[0], poses[0], atol=1e-12)
    np.testing.assert_allclose(out[2], poses[1], atol=1e-12)
    np.testing.assert_allclose(out[1][:3, :3], yaw(np.pi / 4), atol=1e-12)
    np.testing.assert_allclose(out[1][:3, 3], [1, 0, 0], atol=1e-12)
    assert np.isnan(out[3]).all() and np.isnan(out[4]).all()


def test_camera_project_backproject_round_trip_and_principal_point():
    cam = Camera(500.0, 520.0, 320.0, 240.0, 640, 480)
    points = np.column_stack([RNG.uniform(-1, 1, 20), RNG.uniform(-1, 1, 20), RNG.uniform(0.5, 5, 20)])
    uv, z = cam.project(points)
    np.testing.assert_allclose(cam.backproject(uv[:, 0], uv[:, 1], z), points, atol=1e-9)
    uv0, _ = cam.project([[0, 0, 2.0]])
    np.testing.assert_allclose(uv0[0], [320, 240])


def test_camera_points_behind_are_nan_and_outside_image():
    cam = Camera(500.0, 500.0, 320.0, 240.0, 640, 480)
    uv, z = cam.project([[0, 0, -1.0], [0, 0, 1.0], [10, 0, 1.0]])
    assert np.isnan(uv[0]).all() and z[0] == -1
    assert cam.in_image(uv).tolist() == [False, True, False]


def test_camera_scaled_maps_pixel_centres_consistently():
    cam = Camera(1600.0, 1600.0, 959.5, 719.5, 1920, 1440)
    small = cam.scaled(256, 192)
    point = np.array([[0.3, -0.2, 2.0]])
    uv_big, _ = cam.project(point)
    uv_small, _ = small.project(point)
    np.testing.assert_allclose((uv_small + 0.5) * 1920 / 256, uv_big + 0.5, atol=1e-9)


def test_box_corners_contains_and_closest_point():
    box = OrientedBox([1, 2, 3], [2.0, 1.0, 0.5], random_rotation())
    corners = box.corners()
    assert corners.shape == (8, 3)
    assert box.contains(corners, margin=1e-9).all()
    assert not box.contains(box.center + 10.0)[0]
    far = box.center + np.array([[5.0, 0, 0], [0, -7.0, 0]])
    projected = box.closest_point(far)
    np.testing.assert_allclose(box.closest_point(projected), projected, atol=1e-12)
    assert box.contains(projected, margin=1e-9).all()


def test_distance_to_surface_inside_and_outside():
    box = OrientedBox([0, 0, 0], [2.0, 2.0, 2.0], np.eye(3))
    d = box.distance_to_surface([[0, 0, 0], [0.5, 0, 0], [3, 0, 0], [2, 2, 1]])
    np.testing.assert_allclose(d, [1.0, 0.5, 2.0, np.sqrt(2)], atol=1e-12)


def test_extent_along_up_is_height_for_yawed_box():
    box = OrientedBox([0, 0, 0], [3.0, 1.0, 0.7], yaw(0.6))
    assert box.extent_along([0, 0, 1]) == pytest.approx(0.7)
    # Horizontal extent along x of a yawed box: |cos|*3 + |sin|*1
    assert box.extent_along([1, 0, 0]) == pytest.approx(abs(np.cos(0.6)) * 3 + abs(np.sin(0.6)) * 1)


def test_sample_surface_lies_on_surface_with_outward_normals():
    box = OrientedBox([0.5, -1, 2], [1.0, 2.0, 0.4], random_rotation())
    points, normals = box.sample_surface(0.1)
    np.testing.assert_allclose(box.distance_to_surface(points), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)
    assert np.all(np.einsum("ij,ij->i", points - box.center, normals) > 0)


def test_surface_distance_axis_aligned_gap():
    a = OrientedBox([0, 0, 0], [2, 2, 2], np.eye(3))
    b = OrientedBox([5, 0, 0], [2, 2, 2], np.eye(3))
    d, pa, pb = surface_distance(a, b)
    assert d == pytest.approx(3.0, abs=1e-9)
    assert center_distance(a, b) == pytest.approx(5.0)


def test_surface_distance_overlapping_is_zero():
    a = OrientedBox([0, 0, 0], [2, 2, 2], yaw(0.3))
    b = OrientedBox([1, 0.5, 0], [2, 2, 2], yaw(-0.4))
    assert surface_distance(a, b)[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_surface_distance_matches_brute_force_for_rotated_boxes(seed):
    rng = np.random.default_rng(seed)
    a = OrientedBox(rng.normal(size=3), rng.uniform(0.3, 1.5, 3), axis_angle_to_matrix(rng.normal(size=3)))
    b = OrientedBox(a.center + rng.uniform(2.5, 4.0) * rng.normal(size=3) / np.sqrt(3) + 2.0,
                    rng.uniform(0.3, 1.5, 3), axis_angle_to_matrix(rng.normal(size=3)))
    d, pa, pb = surface_distance(a, b)
    # Brute force: dense surface samples of a against b's exact closest points.
    samples, _ = a.sample_surface(0.01)
    brute = np.min(np.linalg.norm(samples - b.closest_point(samples), axis=1))
    assert d <= brute + 1e-9            # alternating projections reach the true minimum...
    assert d >= brute - 0.02            # ...within the 1 cm sampling resolution of the brute force
    assert np.linalg.norm(pa - pb) == pytest.approx(d)
    assert a.contains(pa[None], 1e-6)[0] and b.contains(pb[None], 1e-6)[0]
