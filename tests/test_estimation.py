"""Tests for object state estimation: fusion, the shared capture scale, gating, yaw, tracking, propagation.

Synthetic captures have known truth and Gaussian measurement noise with a known (full, correlated)
covariance. Consistency is checked with the average NEES/NIS over independent seeded Monte Carlo trials
against two-sided 95% chi-square bounds.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from measured_space.estimation import (
    BoxMeasurement,
    box_to_measurement,
    chi2_bounds,
    cv_kalman_filter,
    cv_transition,
    difference_in_frame,
    distance_monte_carlo,
    distance_with_uncertainty,
    fuse_capture,
    naive_fuse,
    predict_measurement,
    prob_positive,
    single_frame_estimate,
    state_error,
    state_to_box,
    track_nees,
    wrap_angle,
)
from measured_space.geometry import Array, axis_angle_to_matrix, invert_pose, make_pose

SWAP_SIDES = np.array([0, 1, 2, 4, 3, 5, 6])


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
    down = np.cross(forward, right)
    return invert_pose(make_pose(np.column_stack([right, down, forward]), eye))


def measurement_cov(rng: np.random.Generator, depth: float) -> Array:
    """A full measurement covariance whose centre noise grows with depth.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator (for the correlation structure).
    depth : float
        Object depth in the camera, metres.

    Returns
    -------
    numpy.ndarray, shape (7, 7)
        Covariance of ``(c_c, size_x, size_y, height, yaw)``: about 2-3 cm laterally, 4-5 cm in depth at
        3 m, 3 cm on sides, 2 cm on height, 0.04 rad on yaw, with random correlations.
    """
    lateral, along = 0.01 + 0.005 * depth, 0.015 + 0.01 * depth
    sigma = np.array([lateral, lateral, along, 0.03, 0.03, 0.02, 0.04])
    mix = rng.normal(size=(7, 14))
    wishart = mix @ mix.T
    corr = 0.5 * np.eye(7) + 0.5 * wishart / np.sqrt(np.outer(np.diag(wishart), np.diag(wishart)))
    return sigma[:, None] * corr * sigma[None, :]


def make_capture(rng: np.random.Generator, n_objects: int = 4, n_frames: int = 25,
                 log_scale: float = 0.0) -> tuple[Array, list[BoxMeasurement]]:
    """Simulate a handheld capture: a camera on a 120-degree arc observing every object in every frame.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    n_objects : int, default 4
        Number of objects.
    n_frames : int, default 25
        Number of frames.
    log_scale : float, default 0.0
        True log of the capture's shared scale error.

    Returns
    -------
    truth : numpy.ndarray, shape (n_objects, 7)
        True states, in object order ``"o0", "o1", ...``.
    measurements : list of BoxMeasurement
        Noisy measurements, frame by frame.
    """
    truth = np.column_stack([rng.uniform([-1.2, -1.2, 0.3], [1.2, 1.2, 1.0], (n_objects, 3)),
                             rng.uniform(0.4, 1.4, (n_objects, 2)), rng.uniform(0.3, 1.2, n_objects),
                             rng.uniform(-np.pi, np.pi, n_objects)])
    measurements = []
    for f, angle in enumerate(np.linspace(-np.pi / 3, np.pi / 3, n_frames)):
        eye = np.array([3.5 * np.cos(angle), 3.5 * np.sin(angle), 1.4 + rng.normal(0, 0.1)])
        T_cw = look_at(eye, np.array([0.0, 0.0, 0.5]) + rng.normal(0, 0.2, 3))
        for i, theta in enumerate(truth):
            z = predict_measurement(theta, T_cw, log_scale)
            cov = measurement_cov(rng, z[2])
            z = z + np.linalg.cholesky(cov) @ rng.normal(size=7)
            measurements.append(BoxMeasurement(f"o{i}", T_cw, z, cov, frame=str(f)))
    return truth, measurements


def test_box_measurement_round_trip_matches_prediction() -> None:
    """A world box seen by a camera gives the measurement the model predicts for its state."""
    theta = np.array([0.4, -0.3, 0.7, 1.1, 0.6, 0.9, 2.5])
    T_cw = look_at(np.array([3.0, 1.0, 1.5]), np.array([0.0, 0.0, 0.5]))
    z = box_to_measurement(state_to_box(theta).transformed(T_cw), T_cw)
    np.testing.assert_allclose(z, predict_measurement(theta, T_cw), atol=1e-12)
    world_theta, _ = single_frame_estimate(BoxMeasurement("a", T_cw, z, np.eye(7)))
    np.testing.assert_allclose(state_error(world_theta, theta), 0.0, atol=1e-12)


def test_fused_covariance_shrinks_as_one_over_n_and_estimators_agree_without_scale_error() -> None:
    """N measurements with one covariance fuse to that covariance / N; with sigma_s -> 0 both fusers agree."""
    rng = np.random.default_rng(1)
    theta = np.array([0.5, -0.2, 0.6, 1.0, 0.5, 0.8, 0.3])
    T_cw = look_at(np.array([3.0, 0.5, 1.4]), np.array([0.0, 0.0, 0.5]))
    cov = measurement_cov(rng, 3.0)
    single_cov = single_frame_estimate(BoxMeasurement("a", T_cw, predict_measurement(theta, T_cw), cov))[1]
    for n in (1, 4, 16, 64):
        z = predict_measurement(theta, T_cw) + rng.multivariate_normal(np.zeros(7), cov, size=n)
        measurements = [BoxMeasurement("a", T_cw, zi, cov) for zi in z]
        naive = naive_fuse(measurements, gate_prob=None)
        np.testing.assert_allclose(naive.cov("a") * n, single_cov, rtol=1e-9, atol=1e-15)

    _, measurements = make_capture(rng)
    naive = naive_fuse(measurements, gate_prob=None)
    held = fuse_capture(measurements, scale_prior_sigma=0.0, gate_prob=None)
    tight = fuse_capture(measurements, scale_prior_sigma=1e-6, gate_prob=None)
    # Held fixed, the scale-aware fit is the naive one; with sigma_s = 1e-6 they differ by sigma_s^2 g g^T.
    for estimate, atol in ((held, 1e-15), (tight, 1e-10)):
        np.testing.assert_allclose(state_error(estimate.theta, naive.theta), 0.0, atol=1e-8)
        np.testing.assert_allclose(estimate.joint_cov()[:-1, :-1], naive.joint_cov()[:-1, :-1],
                                   rtol=1e-6, atol=atol)


def test_scale_aware_nees_is_within_chi2_bounds_for_the_correct_model() -> None:
    """With the log scale drawn from its prior, the joint NEES of all objects and the scale is consistent."""
    rng = np.random.default_rng(2)
    sigma_s, n_trials = 0.05, 60
    values = []
    for _ in range(n_trials):
        log_scale = rng.normal(0.0, sigma_s)
        truth, measurements = make_capture(rng, log_scale=log_scale)
        estimate = fuse_capture(measurements, scale_prior_sigma=sigma_s, gate_prob=None)
        assert estimate.converged
        value, dof = estimate.joint_nees(truth, true_log_scale=log_scale)
        values.append(value)
    lower, upper = chi2_bounds(dof, n_trials)
    assert lower <= np.mean(values) <= upper, (np.mean(values), lower, upper)


def test_shared_scale_error_makes_naive_fusion_overconfident_but_not_scale_aware_fusion() -> None:
    """Key demonstration: a 5% capture scale error breaks naive fusion's covariance, not the scale-aware one.

    The control (naive fusion of captures without a scale error) shows the overconfidence comes from the
    shared scale error and nothing else. NEES is over all object states (7 per object) in both arms. The
    leave-one-out NIS, which needs no ground truth, flags the naive model too (its mean exceeds 7).
    """
    rng = np.random.default_rng(3)
    true_log_scale, n_trials = np.log(1.05), 40
    naive, aware, control, scale_errors, naive_nis, aware_nis = [], [], [], [], [], []
    for _ in range(n_trials):
        truth, measurements = make_capture(rng, log_scale=true_log_scale)
        naive_estimate = naive_fuse(measurements, gate_prob=None)
        naive.append(naive_estimate.joint_nees(truth)[0])
        naive_nis.append(naive_estimate.nis)
        estimate = fuse_capture(measurements, scale_prior_sigma=0.1, gate_prob=None)
        value, dof = estimate.joint_nees(truth)
        aware.append(value)
        aware_nis.append(estimate.nis)
        scale_errors.append((estimate.log_scale - true_log_scale) / np.sqrt(estimate.log_scale_var))
        truth, measurements = make_capture(rng, log_scale=0.0)
        control.append(naive_fuse(measurements, gate_prob=None).joint_nees(truth)[0])
    lower, upper = chi2_bounds(dof, n_trials)
    assert lower <= np.mean(control) <= upper, (np.mean(control), lower, upper)
    assert lower <= np.mean(aware) <= upper, (np.mean(aware), lower, upper)
    assert np.mean(naive) > 10 * upper, (np.mean(naive), upper)
    assert np.mean(np.square(scale_errors)) <= chi2_bounds(1, n_trials)[1]
    assert np.mean(aware_nis) == pytest.approx(7.0, rel=0.1)
    assert np.mean(naive_nis) > 1.5 * 7.0


def test_gating_rejects_injected_outliers_and_keeps_inliers() -> None:
    """Gross errors of several kinds are rejected, inliers are kept, and the fused estimate stays right."""
    rng = np.random.default_rng(4)
    truth, measurements = make_capture(rng, n_objects=4, n_frames=25, log_scale=np.log(1.03))
    outliers = rng.choice(len(measurements), size=8, replace=False)
    for kind, k in enumerate(outliers):
        m = measurements[k]
        z = m.z.copy()
        if kind % 4 == 0:      # centre displaced by 0.5 m
            z[:3] += 0.5 * rng.normal(size=3) / np.sqrt(3) + 0.3
        elif kind % 4 == 1:    # amodal size blown up by 60%
            z[3:5] *= 1.6
        elif kind % 4 == 2:    # heading off by 0.5 rad
            z[6] += 0.5
        else:                  # depth doubled (e.g. a wrong depth layer)
            z[:3] *= 2.0
        measurements[k] = BoxMeasurement(m.object_id, m.T_cw, z, m.cov, m.frame)

    estimate = fuse_capture(measurements, scale_prior_sigma=0.1, gate_prob=0.997)
    is_outlier = np.zeros(len(measurements), dtype=bool)
    is_outlier[outliers] = True
    assert not estimate.accepted[is_outlier].any()
    assert estimate.accepted[~is_outlier].mean() >= 0.97
    value, dof = estimate.joint_nees(truth)
    assert value <= stats.chi2.ppf(0.999, dof)
    corrupted = np.bincount(estimate.measurement_object[is_outlier], minlength=4) > 0
    assert np.all(estimate.p_gross[corrupted] > 0)

    ungated = fuse_capture(measurements, scale_prior_sigma=0.1, gate_prob=None)
    assert ungated.joint_nees(truth)[0] > stats.chi2.ppf(0.999, dof)    # without the gate: inconsistent


@pytest.mark.parametrize("true_yaw", [np.pi / 2 - 0.02, -np.pi + 0.01])
def test_yaw_canonicalisation_fuses_quarter_turn_equivalents(true_yaw: float) -> None:
    """Measurements re-expressed as random equivalents (yaw + k pi/2, sides swapped for odd k) fuse correctly.

    Parameters
    ----------
    true_yaw : float
        True heading, radians: near +pi/2, or near the -pi/pi seam.
    """
    rng = np.random.default_rng(5)
    truth = np.array([0.3, 0.2, 0.5, 1.2, 0.5, 0.7, true_yaw])
    measurements = []
    for angle in np.linspace(-np.pi / 3, np.pi / 3, 30):
        T_cw = look_at(np.array([3.5 * np.cos(angle), 3.5 * np.sin(angle), 1.4]), np.array([0.0, 0.0, 0.5]))
        z = predict_measurement(truth, T_cw)
        cov = measurement_cov(rng, z[2])
        z = z + np.linalg.cholesky(cov) @ rng.normal(size=7)
        k = rng.integers(4)
        z[6] = wrap_angle(z[6] + k * np.pi / 2)
        if k % 2:
            z, cov = z[SWAP_SIDES], cov[np.ix_(SWAP_SIDES, SWAP_SIDES)]
        measurements.append(BoxMeasurement("a", T_cw, z, cov))
    raw_yaws = np.array([m.z[6] for m in measurements])
    assert np.ptp(raw_yaws) > np.pi            # the raw headings really are scattered across equivalents

    for estimate in (fuse_capture(measurements, scale_prior_sigma=0.05, gate_prob=0.997),
                     naive_fuse(measurements, gate_prob=0.997)):
        assert estimate.accepted.mean() >= 0.9    # equivalents are not mistaken for gross errors
        error = state_error(estimate.theta[0], truth)
        assert abs(error[6]) < 0.05
        assert estimate.object_nees(truth[None])[0] <= stats.chi2.ppf(0.999, 7)


def test_constant_velocity_kalman_filter_nees_and_nis_within_bounds() -> None:
    """Two cameras (one with dropouts) on a simulated track: NEES and NIS agree with chi-square."""
    rng = np.random.default_rng(6)
    q, n_runs, n_steps = 0.3, 40, 60
    rot = axis_angle_to_matrix([0.0, 0.0, 0.4])[:2, :2]
    camera_covs = [rot @ np.diag([0.05**2, 0.12**2]) @ rot.T, np.diag([0.08**2, 0.06**2])]
    P0 = np.diag([0.3**2, 0.3**2, 0.5**2, 0.5**2])
    nees_runs, nis_sum, nis_dof = [], 0.0, 0
    misspecified = []
    for _ in range(n_runs):
        times = np.cumsum(rng.uniform(0.08, 0.12, n_steps))
        truth = np.empty((n_steps, 4))
        truth[0] = [0.0, 0.0, 1.0, 0.5]
        for t in range(1, n_steps):
            F, Q = cv_transition(times[t] - times[t - 1], q)
            truth[t] = F @ truth[t - 1] + rng.multivariate_normal(np.zeros(4), Q)
        steps = []
        for x in truth:
            cams = [0] + ([1] if rng.random() < 0.7 else [])
            z = np.array([rng.multivariate_normal(x[:2], camera_covs[c]) for c in cams])
            steps.append((z, np.array([camera_covs[c] for c in cams])))
        x0 = rng.multivariate_normal(truth[0], P0)
        track = cv_kalman_filter(times, steps, q=q, x0=x0, P0=P0)
        nees_runs.append(track_nees(track, truth))
        nis_sum, nis_dof = nis_sum + np.nansum(track.nis), nis_dof + int(track.nis_dof.sum())
        misspecified.append(track_nees(cv_kalman_filter(times, steps, q=q / 100, x0=x0, P0=P0), truth))

    lower, upper = chi2_bounds(4, n_runs)
    per_step = np.mean(nees_runs, axis=0)       # average over independent runs, step by step
    assert np.mean((per_step >= lower) & (per_step <= upper)) >= 0.85
    nis_lower, nis_upper = chi2_bounds(1, nis_dof)
    assert nis_lower <= nis_sum / nis_dof <= nis_upper
    assert np.mean(misspecified) > upper       # a too-small process noise is caught as overconfidence


def test_distance_sigma_from_jacobian_matches_monte_carlo() -> None:
    """First-order distance sigma, with a strong cross-covariance, agrees with Monte Carlo within 10%."""
    rng = np.random.default_rng(7)
    mix = rng.normal(size=(6, 6))
    # A common-mode error shared by both points (as a shared scale error induces), plus independent parts.
    joint = 0.05**2 * (0.2 * mix @ mix.T / 6 + np.kron([[1.0, 0.9], [0.9, 1.0]], np.eye(3)))
    mean_a, mean_b = np.array([1.0, 0.5, 0.8]), np.array([2.4, -0.3, 0.6])
    cov_a, cov_b, cross = joint[:3, :3], joint[3:, 3:], joint[:3, 3:]
    distance, sigma = distance_with_uncertainty(mean_a, cov_a, mean_b, cov_b, cross)
    mc_mean, mc_sigma = distance_monte_carlo(mean_a, cov_a, mean_b, cov_b, cross, n_samples=200_000, rng=8)
    assert abs(sigma - mc_sigma) / mc_sigma < 0.10
    assert abs(distance - mc_mean) < 0.01
    _, sigma_independent = distance_with_uncertainty(mean_a, cov_a, mean_b, cov_b)
    assert sigma_independent > 1.3 * sigma          # ignoring the correlation would overstate sigma here


def test_prob_positive_matches_sampled_left_of_relation() -> None:
    """P(A is right of B in a camera) from the posterior of their x offset matches sampling."""
    rng = np.random.default_rng(9)
    R_cw = look_at(np.array([3.0, 0.0, 1.4]), np.array([0.0, 0.0, 0.5]))[:3, :3]   # camera x is world +y
    mean_a, mean_b = np.array([0.0, 0.05, 0.5]), np.array([0.0, -0.02, 0.5])
    cov_a, cov_b = np.diag([0.03, 0.04, 0.02]) ** 2, np.diag([0.05, 0.03, 0.02]) ** 2
    mean, cov = difference_in_frame(mean_a, cov_a, mean_b, cov_b, rotation=R_cw)   # x_A - x_B in the camera
    p = float(prob_positive(mean[0], cov[0, 0]))
    a = rng.multivariate_normal(mean_a, cov_a, 200_000) @ R_cw.T
    b = rng.multivariate_normal(mean_b, cov_b, 200_000) @ R_cw.T
    assert p == pytest.approx(np.mean(a[:, 0] > b[:, 0]), abs=0.005)
    assert 0.8 < p < 0.95
