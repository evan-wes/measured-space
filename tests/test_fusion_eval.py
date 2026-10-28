"""Tests for the fusion evaluation: the image-evidence noise model, its fit without ground truth, and the audit.

Synthetic captures have known truth: a camera on an arc observes every object in every frame, and each
detection is the true box seen through :func:`~measured_space.estimation.predict_measurement` (optionally
under a capture scale error) plus Gaussian noise drawn from a known :class:`NoiseModel`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from measured_space.estimation import (
    BoxMeasurement,
    box_to_state,
    predict_measurement,
    single_frame_estimate,
    state_error,
    state_to_box,
)
from measured_space.fusion_eval import (
    ESTIMATORS,
    TABLE_COLUMNS,
    Detection,
    NoiseModel,
    evaluate_capture,
    fit_noise_model,
    group_captures,
    load_detections,
    measurements_from_detections,
    summarize,
    summary_table,
)
from measured_space.geometry import Array, invert_pose, make_pose

TRUE_MODEL = NoiseModel(lateral_a=0.01, lateral_b=0.005, ray_a=0.02, ray_b=0.015, size_a=0.03,
                        size_boundary=0.15, size_border=0.2, yaw_sigma=0.05)


def look_at(eye: Array, target: Array) -> Array:
    """World-to-camera pose of an OpenCV camera at ``eye`` looking at ``target``, world up +z.

    Parameters
    ----------
    eye : numpy.ndarray, shape (3,)
        Camera centre in the world, metres.
    target : numpy.ndarray, shape (3,)
        Point the optical axis passes through, metres.

    Returns
    -------
    numpy.ndarray, shape (4, 4)
        ``T_cw``.
    """
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return invert_pose(make_pose(np.column_stack([right, np.cross(forward, right), forward]), eye))


def simulate_capture(rng: np.random.Generator, model: NoiseModel, *, video: str = "v0", n_objects: int = 6,
                     n_frames: int = 20, log_scale: float = 0.0,
                     n_outliers: int = 0) -> tuple[dict[str, Array], list[Detection]]:
    """Simulate one handheld capture: true object states and noisy matched detections.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    model : NoiseModel
        Noise model the detections are drawn from.
    video : str, default "v0"
        Capture identifier.
    n_objects : int, default 6
        Number of objects.
    n_frames : int, default 20
        Number of frames; every object is detected in every frame.
    log_scale : float, default 0.0
        True log of the capture's shared scale error.
    n_outliers : int, default 0
        Detections whose centre is displaced by 1 m (gross errors).

    Returns
    -------
    truth : dict of str to numpy.ndarray, shape (7,)
        True state per object id.
    detections : list of Detection
        Detections, frame by frame.
    """
    truth = {f"o{i}": np.array([*rng.uniform([-1.5, -1.5, 0.3], [1.5, 1.5, 1.0]), *rng.uniform(0.4, 1.4, 2),
                                rng.uniform(0.3, 1.2), rng.uniform(-np.pi, np.pi)]) for i in range(n_objects)}
    detections = []
    for f, angle in enumerate(np.linspace(-np.pi / 3, np.pi / 3, n_frames)):
        radius = rng.uniform(2.5, 4.5)
        eye = np.array([radius * np.cos(angle), radius * np.sin(angle), 1.4 + rng.normal(0, 0.1)])
        T_cw = look_at(eye, np.array([0.0, 0.0, 0.5]) + rng.normal(0, 0.2, 3))
        for object_id, theta in truth.items():
            boundary, border = rng.uniform(0.05, 0.6), float(rng.random() < 0.3)
            z = predict_measurement(theta, T_cw, log_scale)
            z += np.linalg.cholesky(model.covariance(z[:3], boundary, border)) @ rng.normal(size=7)
            world, _ = single_frame_estimate(BoxMeasurement(object_id, T_cw, z, np.eye(7)))
            detections.append(Detection(video, "sim", str(f), object_id, T_cw, state_to_box(world), boundary,
                                        border, label="thing"))
    for k in rng.choice(len(detections), size=n_outliers, replace=False):
        d = detections[k]
        box = state_to_box(box_to_state(d.box) + np.r_[rng.choice([-1.0, 1.0], 3) / np.sqrt(3), np.zeros(4)])
        detections[k] = Detection(d.video, d.source, d.frame, d.object_id, d.T_cw, box, d.boundary_fraction,
                                  d.border_touch, d.label)
    return truth, detections


def test_noise_model_covariance_is_ray_aligned_and_grows_with_range() -> None:
    """The centre block has the along-ray sigma on the viewing ray and the lateral one across it."""
    center = np.array([0.6, -0.3, 2.8])
    cov = TRUE_MODEL.covariance(center, boundary_fraction=0.4, border_touch=1.0)
    ray = center / np.linalg.norm(center)
    lateral, along = TRUE_MODEL.centre_sigmas(np.linalg.norm(center))
    np.testing.assert_allclose(cov[:3, :3] @ ray, along**2 * ray, atol=1e-15)
    across = np.cross(ray, [0.0, 1.0, 0.0])
    across /= np.linalg.norm(across)
    np.testing.assert_allclose(cov[:3, :3] @ across, lateral**2 * across, atol=1e-15)
    size_sigma = 0.03 + 0.15 * 0.4 + 0.2
    np.testing.assert_allclose(np.diag(cov)[3:], [size_sigma**2] * 3 + [0.05**2])
    assert np.count_nonzero(cov[3:, :3]) == 0 and np.all(np.linalg.eigvalsh(cov) > 0)
    far = TRUE_MODEL.covariance(2.0 * center, 0.4, 1.0)
    assert np.all(np.linalg.eigvalsh(far[:3, :3] - cov[:3, :3]) > 0)        # farther is strictly noisier
    with pytest.raises(ValueError):
        NoiseModel(ray_b=-0.1)


def test_detection_becomes_a_measurement_whose_single_frame_estimate_is_the_detected_box() -> None:
    """World box -> camera-frame measurement -> world state round-trips, with the model's covariance."""
    T_cw = look_at(np.array([3.0, 1.0, 1.5]), np.array([0.0, 0.0, 0.5]))
    theta = np.array([0.4, -0.3, 0.7, 1.1, 0.6, 0.9, 2.5])
    detection = Detection("v", "laser", "f0", "a", T_cw, state_to_box(theta), 0.2, 0.0)
    (measurement,) = measurements_from_detections([detection], TRUE_MODEL)
    estimate, _ = single_frame_estimate(measurement)
    np.testing.assert_allclose(state_error(estimate, theta), 0.0, atol=1e-12)
    np.testing.assert_allclose(measurement.cov, TRUE_MODEL.covariance(measurement.z[:3], 0.2, 0.0))
    duplicate = Detection("v", "laser", "f0", "a", T_cw, state_to_box(theta + 0.1), 0.3, 1.0)
    assert group_captures([detection, duplicate]) == {("v", "laser"): [detection]}


def test_load_detections_keeps_matched_boxed_rows_with_their_view_pose(tmp_path: Path) -> None:
    """Rows in the labelling step's format: matched rows become detections posed by their view.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary folder for the label files.
    """
    T_cw = look_at(np.array([3.0, 1.0, 1.5]), np.array([0.0, 0.0, 0.5]))
    box = state_to_box(np.array([0.4, -0.3, 0.7, 1.1, 0.6, 0.9, 2.5]))
    box_json = {"center": box.center.tolist(), "size": box.size.tolist(), "rotation": box.rotation.tolist()}
    view = {"view_key": "arkitscenes/42/wide/1.0", "scene": "42", "frame": "1.0",
            "T_wc": invert_pose(T_cw).tolist()}
    features = {"boundary_fraction": 0.25, "border_touch": 1.0, "depth_median": 2.0}
    match = {"object_id": "uid-7", "label": "chair", "distance": 0.1, "iou": 0.6, "gate": True}
    rows = [{**view, "detection": 0, "source": "laser", "status": "matched", "box": box_json,
             "features": features, "match": match},
            {**view, "detection": 0, "source": "moge", "status": "unboxed", "box": None, "features": {},
             "match": None},
            {**view, "detection": 1, "source": "laser", "status": "false_positive", "box": box_json,
             "features": features, "match": None}]
    (tmp_path / "views.jsonl").write_text(json.dumps(view) + "\n")
    (tmp_path / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    detections, counts = load_detections(tmp_path)
    assert counts == {"rows": 3, "matched": 1, "false_positive": 1, "unboxed": 1, "duplicate_matches": 0}
    (d,) = detections
    assert (d.video, d.source, d.frame, d.object_id, d.label) == ("42", "laser", "1.0", "uid-7", "chair")
    assert (d.boundary_fraction, d.border_touch) == (0.25, 1.0)
    np.testing.assert_allclose(d.T_cw, T_cw, atol=1e-12)
    np.testing.assert_allclose(box_to_state(d.box), box_to_state(box))


def test_noise_fit_recovers_the_constants_without_ground_truth() -> None:
    """REML on frame-to-frame scatter, gated against injected gross errors, recovers the true sigmas.

    The intercept and slope of a sigma trade off over the simulated 2-5 m ranges, so sigmas are compared at
    the ends of that range (and at low and high boundary fraction) rather than constant by constant.
    """
    rng = np.random.default_rng(11)
    captures = [simulate_capture(rng, TRUE_MODEL, video=f"v{c}", n_outliers=4)[1] for c in range(6)]
    fit = fit_noise_model(captures)
    assert fit.converged and fit.n_rejected >= 24 and fit.n_measurements >= 0.95 * 6 * 120 - 24
    for r in (2.0, 5.0):
        np.testing.assert_allclose(fit.model.centre_sigmas(r), TRUE_MODEL.centre_sigmas(r), rtol=0.15)
    for boundary, border in ((0.1, 0.0), (0.5, 0.0), (0.3, 1.0)):
        np.testing.assert_allclose(fit.model.size_sigma(boundary, border),
                                   TRUE_MODEL.size_sigma(boundary, border), rtol=0.15)
    assert fit.model.yaw_sigma == pytest.approx(TRUE_MODEL.yaw_sigma, rel=0.1)


def test_audit_flags_naive_overconfidence_under_a_shared_scale_error() -> None:
    """With a 5% capture scale error, naive fusion fails the NEES audit and scale-aware fusion passes it.

    The noise model is the true one, so single-frame estimates differ from consistency only by the scale
    error itself; the summary and its table carry all three estimators.
    """
    rng = np.random.default_rng(12)
    results = []
    for c in range(8):
        truth, detections = simulate_capture(rng, TRUE_MODEL, video=f"v{c}", log_scale=np.log(1.05))
        results.append(evaluate_capture(detections, truth, TRUE_MODEL, scale_prior_sigma=0.1,
                                        labels={k: "thing" for k in truth}))
    summary = summarize(results)
    stats = summary["by_source"]["sim"]["estimators"]
    lower, upper = stats["naive"]["nees_bounds"]
    assert stats["naive"]["nees_mean"] > 4 * upper                  # about 47 against an upper bound of 8.1
    assert lower <= stats["scale_aware"]["nees_mean"] <= upper
    assert 0.63 < stats["scale_aware"]["coverage_1sigma"]["all"] < 0.74
    assert stats["naive"]["coverage_1sigma"]["center"] < 0.3
    for capture in summary["captures"]:
        assert abs(capture["log_scale"] - np.log(1.05)) < 3 * capture["log_scale_sd"]
    rows = summary_table(summary)
    assert [row["estimator"] for row in rows] == list(ESTIMATORS)
    assert all(set(row) == set(TABLE_COLUMNS) for row in rows)
    assert stats["single"]["n"] == 8 * 6 * 20 and stats["naive"]["n"] == 8 * 6
