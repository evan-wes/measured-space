"""Tests for the Smart Spaces loader: file parsing on tiny synthetic scenes, and conventions on the real data.

The synthetic tests write a fake scene under a temporary data root. The real-data tests use
``MTMC_Tracking_2026/train/Warehouse_018`` and skip when the data root or the scene is unavailable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from measured_space.config import dataset_dir
from measured_space.datasets import smartspaces as ss

SCENE = "MTMC_Tracking_2026/train/Warehouse_018"
REAL_SCENE = "MTMC_Tracking_2026/test/Warehouse_026"
FAKE = "MTMC_Tracking_2026/train/Warehouse_999"


@pytest.fixture(autouse=True)
def _release_handles() -> Iterator[None]:
    """Close cached video and HDF5 handles after each test.

    Yields
    ------
    None
        Control to the test.
    """
    yield
    ss.close_handles()


def _look_down_rotation() -> np.ndarray:
    """World-to-camera rotation of a camera looking along world +x, pitched 30 degrees down, z-up world.

    Returns
    -------
    numpy.ndarray, shape (3, 3), float64
        Rows are the camera's x (right), y (down) and z (forward) axes in world coordinates.
    """
    pitch = np.radians(30)
    forward = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    right = np.array([0.0, -1.0, 0.0])
    return np.stack([right, np.cross(forward, right), forward])


@pytest.fixture
def fake_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-camera scene with calibration under a temporary data root.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory, used as the data root.
    monkeypatch : pytest.MonkeyPatch
        Points ``MEASURED_SPACE_DATA`` at ``tmp_path``.

    Returns
    -------
    pathlib.Path
        The scene folder.
    """
    monkeypatch.setenv("MEASURED_SPACE_DATA", str(tmp_path))
    folder = tmp_path / "data" / "smartspaces" / FAKE
    folder.mkdir(parents=True)
    rotation, center = _look_down_rotation(), np.array([2.0, -3.0, 3.3])
    extrinsic = np.column_stack([rotation, -rotation @ center])
    sensor = {"type": "camera", "id": "Camera_0000", "coordinates": {"x": 2.0, "y": -3.0},
              "attributes": [{"name": "fps", "value": "30"}, {"name": "frameWidth", "value": "640"},
                             {"name": "frameHeight", "value": "480"}],
              "intrinsicMatrix": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
              "extrinsicMatrix": extrinsic.tolist()}
    (folder / "calibration.json").write_text(json.dumps({"calibrationType": "cartesian", "sensors": [sensor]}))
    return folder


def test_calibration_inverts_the_world_to_camera_extrinsic(fake_scene: Path) -> None:
    """The pose places the camera at ``-R^T t`` and projects world points as ``K [R | t]`` does.

    Parameters
    ----------
    fake_scene : pathlib.Path
        The synthetic scene folder.
    """
    (view,) = ss.list_frames(FAKE, "Camera_0000", [7])
    np.testing.assert_allclose(view.camera_center, [2.0, -3.0, 3.3], atol=1e-12)
    assert (view.camera.width, view.camera.height) == (640, 480)
    assert (view.dataset, view.scene, view.camera_id, view.frame) == ("smartspaces", FAKE, "Camera_0000", "7")
    assert view.image_ref == f"{FAKE}/videos/Camera_0000.mp4::7"
    np.testing.assert_allclose(view.up, [0.0, 0.0, 1.0])

    point = np.array([6.0, -2.5, 0.0])                      # on the floor, in front of the camera
    extrinsic = np.array(json.loads((fake_scene / "calibration.json").read_text())["sensors"][0]["extrinsicMatrix"])
    expected = view.camera.K @ (extrinsic[:, :3] @ point + extrinsic[:, 3])
    uv, z = view.camera.project((point - view.camera_center) @ view.T_wc[:3, :3])
    np.testing.assert_allclose(uv[0], expected[:2] / expected[2], atol=1e-9)
    assert z[0] > 0


def test_ground_truth_accepts_space_and_underscore_keys(fake_scene: Path) -> None:
    """Both key spellings parse to the same objects, ids and visible 2D boxes.

    Parameters
    ----------
    fake_scene : pathlib.Path
        The synthetic scene folder.
    """
    spaced = {"object type": "Forklift", "object id": 3, "3d location": [1.0, 2.0, 1.0],
              "3d bounding box scale": [0.8, 1.9, 2.0], "3d bounding box rotation": [0.0, 0.0, 0.5],
              "2d bounding box visible": {"Camera_0000": [10, 20, 110, 220]}}
    underscored = {key.replace(" ", "_"): value for key, value in spaced.items()}
    hidden = {**spaced, "object id": 4, "2d bounding box visible": {}}
    (fake_scene / "ground_truth.json").write_text(json.dumps({"0": [spaced, hidden], "1": [underscored]}))

    objects, visible = ss.load_ground_truth(FAKE), ss.visible_boxes_2d(FAKE)
    assert [o.object_id for o in objects[0]] == ["Forklift_3", "Forklift_4"]
    (a, b) = objects[0][0], objects[1][0]
    assert a.label == "Forklift" and a.source == "oracle"
    for obj in (a, b):
        np.testing.assert_allclose(obj.box.center, [1.0, 2.0, 1.0])
        np.testing.assert_allclose(obj.box.size, [0.8, 1.9, 2.0])
    np.testing.assert_allclose(a.box.rotation, b.box.rotation)
    assert visible[0] == {"Forklift_3": {"Camera_0000": (10, 20, 110, 220)}}     # unseen objects are absent
    assert visible[1] == visible[0]


def test_yaw_turns_the_length_axis_and_degrees_are_rejected() -> None:
    """Yaw is radians counter-clockwise about +z; scale is (width, length, height) along local x, y, z."""
    box = ss.box_from_annotation([0.0, 0.0, 1.0], [1.0, 2.0, 3.0], [0.0, 0.0, np.pi / 2])
    np.testing.assert_allclose(box.rotation @ [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.ptp(box.corners(), axis=0), [2.0, 1.0, 3.0], atol=1e-12)
    with pytest.raises(ValueError, match="not radians"):
        ss.rotation_matrices(np.array([[0.0, 0.0, 90.0]]))


def test_depth_reads_raw_and_png_encoded_millimetres(fake_scene: Path) -> None:
    """Raw uint16 frames and PNG-encoded frames both decode to float32 metres, keeping 0 as invalid.

    Parameters
    ----------
    fake_scene : pathlib.Path
        The synthetic scene folder.
    """
    millimetres = np.full((480, 640), 5250, dtype=np.uint16)
    millimetres[:10] = 0
    ok, png = cv2.imencode(".png", millimetres)
    assert ok
    (fake_scene / "depth_maps").mkdir()
    with h5py.File(fake_scene / "depth_maps" / "Camera_0000.h5", "w") as f:
        f.create_dataset("distance_to_image_plane_00000.png", data=millimetres, compression="gzip")
        f.create_dataset("distance_to_image_plane_00001.png", data=png.ravel())
    for view in ss.list_frames(FAKE, "Camera_0000", [0, 1]):
        depth = ss.load_depth(view)
        assert depth.dtype == np.float32 and depth.shape == (480, 640)
        assert depth[0, 0] == 0.0 and depth[100, 100] == pytest.approx(5.25)
    with pytest.raises(KeyError):
        ss.load_depth(ss.list_frames(FAKE, "Camera_0000", [2])[0])


# ------------------------------------------------------------------------------------------------ real data


@pytest.fixture(scope="module")
def real_scene() -> str:
    """The synthetic warehouse scene with GT, video and depth; skips when it is not available.

    Returns
    -------
    str
        The scene path below the dataset folder.
    """
    try:
        folder = dataset_dir("smartspaces") / SCENE
    except RuntimeError as error:
        pytest.skip(f"data root unavailable: {error}")
    if not (folder / "ground_truth.json").is_file() or not (folder / "videos" / "Camera_0000.mp4").is_file():
        pytest.skip(f"{SCENE} is not downloaded")
    return SCENE


def test_real_calibration_and_ground_truth(real_scene: str) -> None:
    """Camera centres match the listed floor coordinates at ceiling height; frame 0 has the expected objects.

    Parameters
    ----------
    real_scene : str
        The scene path.
    """
    raw = json.loads((ss.scene_dir(real_scene) / "calibration.json").read_text())
    listed = {s["id"]: [s["coordinates"]["x"], s["coordinates"]["y"]] for s in raw["sensors"]}
    calibration = ss.load_calibration(real_scene)
    assert sorted(calibration) == ["Camera_0000", "Camera_0001", "Camera_0002"]
    for camera_id, (camera, T_wc) in calibration.items():
        np.testing.assert_allclose(T_wc[:2, 3], listed[camera_id], atol=1e-6)
        assert 3.0 < T_wc[2, 3] < 3.5 and (camera.width, camera.height) == (1920, 1080)
    objects = {o.object_id: o for o in ss.load_ground_truth(real_scene)[0]}
    assert sorted(objects) == ["Forklift_3", "Forklift_4", "PalletTruck_2", "Person_0"]
    for object_id in ("Forklift_3", "PalletTruck_2"):                 # parked on the floor (z = 0)
        box = objects[object_id].box
        assert abs(box.center[2] - box.size[2] / 2) < 0.02
    if (ss.scene_dir(REAL_SCENE) / "calibration.json").is_file():
        assert len(ss.load_calibration(REAL_SCENE)) == 4


def test_video_seek_returns_the_requested_frame(real_scene: str) -> None:
    """Seeking backwards and forwards gives the same pixels as decoding the video sequentially.

    Parameters
    ----------
    real_scene : str
        The scene path.
    """
    capture = cv2.VideoCapture(str(ss.scene_dir(real_scene) / "videos" / "Camera_0000.mp4"))
    sequential = []
    for _ in range(32):
        ok, image = capture.read()
        assert ok
        sequential.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    capture.release()
    assert np.abs(sequential[30].astype(int) - sequential[31]).mean() > 0      # frames differ
    for view in ss.list_frames(real_scene, "Camera_0000", [31, 5, 6, 30]):
        np.testing.assert_array_equal(ss.load_rgb(view), sequential[int(view.frame)])


def test_projected_boxes_match_the_ground_truth_2d_boxes(real_scene: str) -> None:
    """Projected 3D boxes contain the GT visible 2D boxes and overlap them (loose IoU bound).

    On this 10-frame sample the per-class median IoU measured 0.566 (Person) to 0.750 (Forklift) and every
    GT box lay fully inside its projected box (share 1.0). IoU is below 1 because the GT box is the tight
    silhouette of the visible part, while the 3D box also encloses free space (forks, mast, limbs).

    Parameters
    ----------
    real_scene : str
        The scene path.
    """
    objects, visible = ss.load_ground_truth(real_scene), ss.visible_boxes_2d(real_scene)
    ious, covered = defaultdict(list), defaultdict(list)
    for camera_id in ("Camera_0000", "Camera_0002"):
        for view in ss.list_frames(real_scene, camera_id, range(0, 9000, 900)):
            for obj in objects[int(view.frame)]:
                gt_box = visible[int(view.frame)].get(obj.object_id, {}).get(camera_id)
                if gt_box is None:
                    continue
                uv, z = view.camera.project(obj.box.transformed(view.T_cw).corners())
                assert (z > 0).all()
                lo = np.maximum(uv.min(0), -0.5)
                hi = np.minimum(uv.max(0), [view.camera.width - 0.5, view.camera.height - 0.5])
                gt_lo, gt_hi = np.array(gt_box[:2]) - 0.5, np.array(gt_box[2:]) + 0.5
                inter = np.prod(np.clip(np.minimum(hi, gt_hi) - np.maximum(lo, gt_lo), 0, None))
                ious[obj.label].append(inter / (np.prod(hi - lo) + np.prod(gt_hi - gt_lo) - inter))
                covered[obj.label].append(inter / np.prod(gt_hi - gt_lo))
    assert set(ious) == {"Forklift", "PalletTruck", "Person"}
    for label in ious:
        assert np.median(ious[label]) > 0.45, label
        assert np.median(covered[label]) > 0.95, label
