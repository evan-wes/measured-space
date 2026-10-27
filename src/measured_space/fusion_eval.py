"""Fusion evaluation on real captures: single-frame, naive-fused and scale-aware fused boxes against ground truth.

Each automatic detection of a ground-truth object in one view becomes a :class:`~measured_space.estimation.
BoxMeasurement`, and every object of a capture (one video seen through one depth source) is estimated three
ways: from each measurement alone (:func:`~measured_space.estimation.single_frame_estimate`), by naive fusion
that treats frames as independent (:func:`~measured_space.estimation.naive_fuse`), and by scale-aware fusion
with a shared per-capture log scale (:func:`~measured_space.estimation.fuse_capture`). Each estimate's error
against the ground-truth box is then audited with NEES and with per-component coverage.

Measurement noise model
-----------------------
The covariance ``R`` of a measurement ``z = (c_c, size_x, size_y, height, yaw)`` comes only from image
evidence, through eight constants (:class:`NoiseModel`):

- centre: with ``r = |c_c|`` the range to the box centre and ``u = c_c / r`` the viewing ray,
  ``Cov(c_c) = s_lat^2 (I - u u^T) + s_ray^2 u u^T``, ``s_lat = lateral_a + lateral_b r`` and
  ``s_ray = ray_a + ray_b r`` (metres);
- size: the two horizontal sides and the height share ``s_size = size_a + size_boundary * b + size_border * t``
  (metres), where ``b`` is the mask's boundary fraction and ``t`` is 1 if the mask touches the image border;
- yaw: a constant ``yaw_sigma`` (radians).

Blocks are uncorrelated. The constants are fitted **without ground-truth geometry** (:func:`fit_noise_model`),
by restricted maximum likelihood (REML) on the scatter of each object's measurements about its own fused
mean, frame to frame. The per-object means are nuisance parameters that REML integrates out, so the fit
does not need to know where the object really is; it only needs to know which detections belong to the
same object. Here that association comes from the detections' matched ground-truth ids (oracle data
association). Because only frame-to-frame scatter is fitted, errors shared by every frame of an object (the
capture's scale, the pull of the visible surface towards the camera, amodal undersizing, the object not
being a box) are invisible to the fit by construction. The NEES audit measures how much of the real error
they are.

Gross errors are excluded from the fit by alternating it with the chi-square innovation gate of naive fusion,
which also supplies each object's reference yaw (a box's yaw is only defined modulo a quarter turn).

Audit
-----
Per object, NEES ``e^T P^{-1} e`` (7 dof under a consistent estimator) uses the estimator's marginal
covariance. The mean over objects is compared with the two-sided 95% chi-square bounds for independent
samples; objects of one capture share errors, so those bounds are nominal. Real objects are not boxes, so a
few large errors can dominate a mean: the median NEES (chi-square(7) median 6.35), the share above the
chi-square(7) 95% quantile, and the per-component coverage ``|e_j| <= k sigma_j`` (0.683 and 0.954 for
k = 1, 2 when calibrated) are reported alongside it.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from measured_space.estimation import (
    CENTER,
    HEIGHT,
    STATE_DIM,
    YAW,
    BoxMeasurement,
    CaptureEstimate,
    box_to_measurement,
    box_to_state,
    canonicalize_yaw,
    chi2_bounds,
    fuse_capture,
    naive_fuse,
    nees,
    single_frame_estimate,
    state_error,
    wrap_angle,
)
from measured_space.geometry import Array, OrientedBox, invert_pose

ESTIMATORS = ("single", "naive", "scale_aware")
"""The three estimates compared: each measurement alone, naive fusion, scale-aware fusion."""

COMPONENT_NAMES = ("cx", "cy", "cz", "size_x", "size_y", "height", "yaw")
"""Names of the seven state components (world centre, horizontal sides, height, yaw)."""

COMPONENT_GROUPS: dict[str, tuple[int, ...]] = {"center": (0, 1, 2), "size": (3, 4, 5), "yaw": (6,),
                                               "all": tuple(range(STATE_DIM))}
"""State components pooled for coverage: centre, size (sides and height), yaw, and all seven."""

# log-parameter bounds for the noise fit: 0.1 mm to 5 m (or 5 m per metre of range, or 5 rad)
_LOG_BOUNDS = (math.log(1e-4), math.log(5.0))


# --------------------------------------------------------------------------------------------------------
# Detections and the measurement noise model
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """One automatic box of a ground-truth object, lifted in one view with one depth source.

    Parameters
    ----------
    video : str
        Capture (video) identifier.
    source : str
        Depth source the box was lifted with, e.g. ``"laser"``, ``"lidar"`` or ``"moge"``.
    frame : str
        View identifier within the video.
    object_id : str
        The ground-truth object the detection was matched to.
    T_cw : numpy.ndarray, shape (4, 4)
        The view's world-to-camera pose.
    box : OrientedBox
        Lifted box in the world frame, metres; gravity-aligned.
    boundary_fraction : float
        Share of mask pixels near the mask's edge, in [0, 1] (see :func:`measured_space.perception.mask_features`).
    border_touch : float
        1.0 if the mask reaches the image border (a truncated object), else 0.0.
    label : str, default ""
        Ground-truth class of the matched object, for reporting.
    """

    video: str
    source: str
    frame: str
    object_id: str
    T_cw: Array
    box: OrientedBox
    boundary_fraction: float
    border_touch: float
    label: str = ""

    def __post_init__(self) -> None:
        """Coerce the pose to a float64 4x4 array and the features to floats.

        Raises
        ------
        ValueError
            If the pose cannot be reshaped to 4x4 or a feature is not finite.
        """
        object.__setattr__(self, "T_cw", np.asarray(self.T_cw, dtype=np.float64).reshape(4, 4))
        for name in ("boundary_fraction", "border_touch"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class NoiseModel:
    """Measurement covariance from image evidence: range, viewing ray, mask boundary fraction, border touch.

    See the module docstring for the model. All constants are non-negative.

    Parameters
    ----------
    lateral_a : float, default 0.02
        Centre sigma across the viewing ray at zero range, metres.
    lateral_b : float, default 0.01
        Growth of the across-ray centre sigma with range, metres per metre.
    ray_a : float, default 0.03
        Centre sigma along the viewing ray at zero range, metres.
    ray_b : float, default 0.02
        Growth of the along-ray centre sigma with range, metres per metre.
    size_a : float, default 0.05
        Size sigma of a mask with no boundary pixels that does not touch the border, metres.
    size_boundary : float, default 0.1
        Size sigma added per unit of boundary fraction, metres.
    size_border : float, default 0.1
        Size sigma added when the mask touches the image border, metres.
    yaw_sigma : float, default 0.1
        Yaw sigma, radians.
    """

    lateral_a: float = 0.02
    lateral_b: float = 0.01
    ray_a: float = 0.03
    ray_b: float = 0.02
    size_a: float = 0.05
    size_boundary: float = 0.1
    size_border: float = 0.1
    yaw_sigma: float = 0.1

    UNITS: ClassVar[dict[str, str]] = {"lateral_a": "m", "lateral_b": "m/m", "ray_a": "m", "ray_b": "m/m",
                                       "size_a": "m", "size_boundary": "m", "size_border": "m",
                                       "yaw_sigma": "rad"}

    def __post_init__(self) -> None:
        """Check that every constant is a finite non-negative number.

        Raises
        ------
        ValueError
            If a constant is negative or not finite.
        """
        for f in fields(self):
            value = float(getattr(self, f.name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{f.name} must be finite and non-negative, got {value}")
            object.__setattr__(self, f.name, value)

    def to_dict(self) -> dict[str, float]:
        """The constants by name.

        Returns
        -------
        dict of str to float
            ``{name: value}`` in the units of :attr:`UNITS`.
        """
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    def centre_sigmas(self, range_m: ArrayLike) -> tuple[Array, Array]:
        """Centre sigmas across and along the viewing ray.

        Parameters
        ----------
        range_m : array_like, shape (...)
            Distance from the camera to the box centre, metres.

        Returns
        -------
        lateral : numpy.ndarray, shape (...)
            Sigma across the ray, metres.
        along : numpy.ndarray, shape (...)
            Sigma along the ray, metres.
        """
        r = np.asarray(range_m, dtype=np.float64)
        return self.lateral_a + self.lateral_b * r, self.ray_a + self.ray_b * r

    def size_sigma(self, boundary_fraction: ArrayLike, border_touch: ArrayLike) -> Array:
        """Sigma of each size component (both horizontal sides and the height).

        Parameters
        ----------
        boundary_fraction : array_like, shape (...)
            Mask boundary fraction, in [0, 1].
        border_touch : array_like, shape (...)
            1 where the mask touches the image border, else 0.

        Returns
        -------
        numpy.ndarray, shape (...)
            Sigma, metres.
        """
        return (self.size_a + self.size_boundary * np.asarray(boundary_fraction, dtype=np.float64)
                + self.size_border * np.asarray(border_touch, dtype=np.float64))

    def covariances(self, center_cam: ArrayLike, boundary_fraction: ArrayLike,
                    border_touch: ArrayLike) -> Array:
        """Measurement covariances of many detections at once.

        Parameters
        ----------
        center_cam : array_like, shape (K, 3)
            Measured box centres in the camera frame, metres (z forward, so the range is positive).
        boundary_fraction : array_like, shape (K,)
            Mask boundary fractions.
        border_touch : array_like, shape (K,)
            Border-touch flags (0 or 1).

        Returns
        -------
        numpy.ndarray, shape (K, 7, 7)
            Covariances of ``(c_c, size_x, size_y, height, yaw)``: metres squared, and radians squared for yaw.

        Raises
        ------
        ValueError
            If a centre is at the camera (zero range).
        """
        center_cam = np.asarray(center_cam, dtype=np.float64).reshape(-1, 3)
        range_m = np.linalg.norm(center_cam, axis=1)
        if np.any(range_m <= 0):
            raise ValueError("box centre at the camera: the viewing ray is undefined")
        ray = center_cam / range_m[:, None]
        lateral, along = self.centre_sigmas(range_m)
        cov = np.zeros((len(center_cam), STATE_DIM, STATE_DIM))
        outer = ray[:, :, None] * ray[:, None, :]
        cov[:, CENTER, CENTER] = (lateral[:, None, None] ** 2 * (np.eye(3) - outer)
                                  + along[:, None, None] ** 2 * outer)
        size_var = self.size_sigma(boundary_fraction, border_touch).reshape(-1) ** 2
        for k in (3, 4, HEIGHT):
            cov[:, k, k] = size_var
        cov[:, YAW, YAW] = self.yaw_sigma**2
        return cov

    def covariance(self, center_cam: ArrayLike, boundary_fraction: float, border_touch: float) -> Array:
        """Measurement covariance of one detection.

        Parameters
        ----------
        center_cam : array_like, shape (3,)
            Measured box centre in the camera frame, metres.
        boundary_fraction : float
            Mask boundary fraction, in [0, 1].
        border_touch : float
            1 if the mask touches the image border, else 0.

        Returns
        -------
        numpy.ndarray, shape (7, 7)
            Covariance of ``(c_c, size_x, size_y, height, yaw)``.
        """
        return self.covariances(np.reshape(center_cam, (1, 3)), [boundary_fraction], [border_touch])[0]


def measurements_from_detections(detections: Sequence[Detection], model: NoiseModel) -> list[BoxMeasurement]:
    """Turn detections into box measurements with covariances from the noise model.

    Parameters
    ----------
    detections : sequence of Detection
        Detections, all of one capture.
    model : NoiseModel
        Measurement noise model.

    Returns
    -------
    list of BoxMeasurement
        One per detection, in input order; ``object_id`` is the matched ground-truth id and ``frame`` the view.
    """
    if not detections:
        return []
    z = np.stack([box_to_measurement(d.box.transformed(d.T_cw), d.T_cw) for d in detections])
    cov = model.covariances(z[:, CENTER], [d.boundary_fraction for d in detections],
                            [d.border_touch for d in detections])
    return [BoxMeasurement(d.object_id, d.T_cw, zk, ck, frame=d.frame) for d, zk, ck in zip(detections, z, cov)]


def group_captures(detections: Iterable[Detection]) -> dict[tuple[str, str], list[Detection]]:
    """Group detections by capture, keeping one detection per view and object.

    When several detections of one view match the same object (which a one-to-one matcher should not
    produce), the first is kept.

    Parameters
    ----------
    detections : iterable of Detection
        Matched detections of any videos and depth sources.

    Returns
    -------
    dict of (str, str) to list of Detection
        ``(video, source) -> detections``, keys sorted, detections in input order.
    """
    captures: dict[tuple[str, str], list[Detection]] = defaultdict(list)
    seen: set[tuple[str, str, str, str]] = set()
    for d in detections:
        key = (d.video, d.source, d.frame, d.object_id)
        if key not in seen:
            seen.add(key)
            captures[(d.video, d.source)].append(d)
    return {key: captures[key] for key in sorted(captures)}


# --------------------------------------------------------------------------------------------------------
# Fitting the noise model without ground truth (REML on frame-to-frame scatter)
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scatter:
    """Accepted measurements of objects seen at least twice, as world-frame single-frame readings.

    Parameters
    ----------
    group : numpy.ndarray of int, shape (K,)
        Object group of each measurement (one group per object and capture).
    n_groups : int
        Number of groups, G.
    center : numpy.ndarray, shape (K, 3)
        World-frame centres, metres.
    ray : numpy.ndarray, shape (K, 3)
        Unit viewing rays in the world frame.
    range_m : numpy.ndarray, shape (K,)
        Range from the camera to the measured centre, metres.
    size : numpy.ndarray, shape (K, 3)
        Sides and height, sides ordered consistently with the group's reference yaw, metres.
    yaw : numpy.ndarray, shape (K,)
        Yaw, unwrapped next to the group's reference yaw, radians.
    boundary_fraction : numpy.ndarray, shape (K,)
        Mask boundary fractions.
    border_touch : numpy.ndarray, shape (K,)
        Border-touch flags.
    n_rejected : int
        Measurements left out because the gate rejected them.
    """

    group: NDArray[np.intp]
    n_groups: int
    center: Array
    ray: Array
    range_m: Array
    size: Array
    yaw: Array
    boundary_fraction: Array
    border_touch: Array
    n_rejected: int


def _scatter(captures: Sequence[Sequence[Detection]], model: NoiseModel, gate_prob: float | None,
             min_frames: int) -> _Scatter:
    """Gate each capture with naive fusion and collect the accepted measurements of well-seen objects.

    Parameters
    ----------
    captures : sequence of sequence of Detection
        Detections, one inner sequence per capture.
    model : NoiseModel
        Current noise model (for the gate and the fused reference yaw).
    gate_prob : float or None
        Chi-square(7) gate probability; None keeps every measurement.
    min_frames : int
        Fewest accepted measurements an object needs to contribute.

    Returns
    -------
    _Scatter
        The collected measurements.

    Raises
    ------
    ValueError
        If no object has ``min_frames`` accepted measurements.
    """
    rows: list[tuple[int, Detection, float]] = []
    n_groups, n_rejected = 0, 0
    for detections in captures:
        if not detections:
            continue
        estimate = naive_fuse(measurements_from_detections(detections, model), gate_prob=gate_prob)
        counts = np.bincount(estimate.measurement_object[estimate.accepted], minlength=len(estimate.object_ids))
        n_rejected += int(np.count_nonzero(~estimate.accepted))
        keep = estimate.accepted & (counts[estimate.measurement_object] >= min_frames)
        group_of = {i: n_groups + rank for rank, i in enumerate(np.flatnonzero(counts >= min_frames))}
        n_groups += len(group_of)
        for d, i, ok in zip(detections, estimate.measurement_object, keep):
            if ok:
                rows.append((group_of[int(i)], d, float(estimate.theta[i, YAW])))
    if not rows:
        raise ValueError(f"no object has {min_frames} accepted measurements; cannot fit the noise model")
    group = np.array([g for g, _, _ in rows], dtype=np.intp)
    reference = np.array([ref for _, _, ref in rows])
    states = np.stack([box_to_state(d.box) for _, d, _ in rows])
    eyes = np.stack([invert_pose(d.T_cw)[:3, 3] for _, d, _ in rows])
    offset = states[:, CENTER] - eyes
    range_m = np.linalg.norm(offset, axis=1)
    sides, yaw, _ = canonicalize_yaw(states[:, 3:5], states[:, YAW], reference)
    return _Scatter(group=group, n_groups=n_groups, center=states[:, CENTER], ray=offset / range_m[:, None],
                    range_m=range_m, size=np.column_stack([sides, states[:, HEIGHT]]),
                    yaw=reference + wrap_angle(yaw - reference),
                    boundary_fraction=np.array([d.boundary_fraction for _, d, _ in rows]),
                    border_touch=np.array([d.border_touch for _, d, _ in rows]), n_rejected=n_rejected)


def _sum_groups(values: Array, group: NDArray[np.intp], n_groups: int) -> Array:
    """Sum per-measurement arrays into per-group totals.

    Parameters
    ----------
    values : numpy.ndarray, shape (K, ...)
        One entry per measurement.
    group : numpy.ndarray of int, shape (K,)
        Each measurement's group.
    n_groups : int
        Number of groups, G.

    Returns
    -------
    numpy.ndarray, shape (G, ...)
        Per-group sums.
    """
    out = np.zeros((n_groups, *values.shape[1:]))
    np.add.at(out, group, values)
    return out


def _centre_neg_log_lik(log_params: Array, data: _Scatter) -> float:
    """REML deviance of the world-frame centres under the ray-aligned centre model.

    Parameters
    ----------
    log_params : numpy.ndarray, shape (4,)
        Logs of ``(lateral_a, lateral_b, ray_a, ray_b)``.
    data : _Scatter
        The measurements.

    Returns
    -------
    float
        ``sum_k [log det S_k + r_k^T S_k^{-1} r_k] + sum_g log det A_g``, with ``A_g = sum S_k^{-1}`` and
        ``r_k`` the residual from the group's generalised-least-squares mean (constants dropped).
    """
    la, lb, ra, rb = np.exp(log_params)
    lateral, along = la + lb * data.range_m, ra + rb * data.range_m
    p, q = 1.0 / lateral**2, 1.0 / along**2
    outer = data.ray[:, :, None] * data.ray[:, None, :]
    info = p[:, None, None] * (np.eye(3) - outer) + q[:, None, None] * outer
    A = _sum_groups(info, data.group, data.n_groups)
    b = _sum_groups(np.einsum("kij,kj->ki", info, data.center), data.group, data.n_groups)
    mean = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    r = data.center - mean[data.group]
    along_r = np.einsum("ki,ki->k", data.ray, r)
    quad = p * (np.einsum("ki,ki->k", r, r) - along_r**2) + q * along_r**2
    logdet = 4.0 * np.log(lateral) + 2.0 * np.log(along)
    return float(np.sum(logdet + quad) + np.sum(np.linalg.slogdet(A)[1]))


def _scalar_neg_log_lik(values: Array, variance: Array, group: NDArray[np.intp], n_groups: int) -> float:
    """REML deviance of scalar components that share one variance per measurement.

    Parameters
    ----------
    values : numpy.ndarray, shape (K, m)
        Observed components (each column has its own per-group mean).
    variance : numpy.ndarray, shape (K,)
        Variance of every component of each measurement.
    group : numpy.ndarray of int, shape (K,)
        Each measurement's group.
    n_groups : int
        Number of groups, G.

    Returns
    -------
    float
        ``sum_k [m log v_k + |r_k|^2 / v_k] + m sum_g log sum_{k in g} 1 / v_k`` (constants dropped).
    """
    w = 1.0 / variance
    A = _sum_groups(w, group, n_groups)
    mean = _sum_groups(values * w[:, None], group, n_groups) / A[:, None]
    r = values - mean[group]
    m = values.shape[1]
    return float(np.sum(m * np.log(variance) + np.sum(r**2, axis=1) * w) + m * np.sum(np.log(A)))


def _minimize(objective: Callable[[Array], float], start: Sequence[float]) -> Array:
    """Minimise a deviance over log-parameters within :data:`_LOG_BOUNDS`.

    Parameters
    ----------
    objective : callable
        ``objective(log_params) -> float``.
    start : sequence of float
        Starting parameter values (not logs); each is clipped into the bounds.

    Returns
    -------
    numpy.ndarray
        The minimising parameters (not logs).
    """
    x0 = np.clip(np.log(np.maximum(np.asarray(start, dtype=np.float64), 1e-12)), *_LOG_BOUNDS)
    result = optimize.minimize(objective, x0, method="L-BFGS-B", bounds=[_LOG_BOUNDS] * len(x0))
    return np.exp(result.x)


def _fit_blocks(data: _Scatter, start: NoiseModel) -> NoiseModel:
    """Fit the centre, size and yaw constants, each block by REML on its own components.

    Parameters
    ----------
    data : _Scatter
        Accepted measurements of objects seen at least twice.
    start : NoiseModel
        Starting point of the optimisation.

    Returns
    -------
    NoiseModel
        The fitted constants.
    """
    centre = _minimize(lambda x: _centre_neg_log_lik(x, data),
                       [start.lateral_a, start.lateral_b, start.ray_a, start.ray_b])

    def size_deviance(log_params: Array) -> float:
        """REML deviance of the sizes under ``sigma = a + b * boundary + c * border``.

        Parameters
        ----------
        log_params : numpy.ndarray, shape (3,)
            Logs of ``(size_a, size_boundary, size_border)``.

        Returns
        -------
        float
            The deviance.
        """
        a, b, c = np.exp(log_params)
        sigma = a + b * data.boundary_fraction + c * data.border_touch
        return _scalar_neg_log_lik(data.size, sigma**2, data.group, data.n_groups)

    def yaw_deviance(log_params: Array) -> float:
        """REML deviance of the yaws under a constant sigma.

        Parameters
        ----------
        log_params : numpy.ndarray, shape (1,)
            Log of ``yaw_sigma``.

        Returns
        -------
        float
            The deviance.
        """
        variance = np.full(len(data.yaw), np.exp(2.0 * log_params[0]))
        return _scalar_neg_log_lik(data.yaw[:, None], variance, data.group, data.n_groups)

    size = _minimize(size_deviance, [start.size_a, start.size_boundary, start.size_border])
    yaw = _minimize(yaw_deviance, [start.yaw_sigma])
    return NoiseModel(*centre, *size, *yaw)


@dataclass(frozen=True)
class NoiseFit:
    """A fitted noise model and what it was fitted on.

    Parameters
    ----------
    model : NoiseModel
        The fitted constants.
    rounds : int
        Fit-and-gate rounds performed.
    converged : bool
        Whether the constants settled (largest relative change below the tolerance) within the round limit.
    n_measurements : int
        Measurements in the final fit (accepted by the gate, of objects with enough of them).
    n_objects : int
        Object groups (object x capture) in the final fit.
    n_rejected : int
        Measurements the gate rejected in the final round.
    """

    model: NoiseModel
    rounds: int
    converged: bool
    n_measurements: int
    n_objects: int
    n_rejected: int


def fit_noise_model(captures: Sequence[Sequence[Detection]], *, initial: NoiseModel | None = None,
                    gate_prob: float | None = 0.997, min_frames: int = 2, max_rounds: int = 20,
                    rtol: float = 0.01) -> NoiseFit:
    """Fit the noise constants from frame-to-frame scatter, without ground-truth geometry.

    Alternates two steps until the constants settle: gate every capture with naive fusion under the current
    constants (which also fixes each object's reference yaw), then refit the constants by REML on the
    accepted measurements of objects with at least ``min_frames`` of them. The first round does not gate, so
    its fit is inflated by any gross errors and the gate tightens from there; starting from constants that
    are too small would otherwise reject inliers and could settle on an underestimate. Centre, size and yaw are fitted
    separately, which is exact because their covariance blocks are uncorrelated and their means separate.

    Parameters
    ----------
    captures : sequence of sequence of Detection
        Detections, one inner sequence per capture (one video, one depth source). Detections are grouped into
        objects by their ``object_id`` within each capture.
    initial : NoiseModel or None, default None
        Starting constants; None uses the :class:`NoiseModel` defaults.
    gate_prob : float or None, default 0.997
        Chi-square(7) innovation-gate probability; None disables gating.
    min_frames : int, default 2
        Fewest accepted measurements an object needs to contribute (at least 2: one measurement has no
        scatter).
    max_rounds : int, default 20
        Limit on fit-and-gate rounds.
    rtol : float, default 0.01
        Convergence tolerance on the largest relative change of a constant (relative to at least 1 mm, or
        1 mm per metre, or 1 mrad, so a constant pinned at its lower bound does not block convergence).

    Returns
    -------
    NoiseFit
        The fitted model and its bookkeeping.

    Raises
    ------
    ValueError
        If ``min_frames`` is below 2, or no object has ``min_frames`` accepted measurements.
    """
    if min_frames < 2:
        raise ValueError("min_frames must be at least 2")
    model = initial or NoiseModel()
    converged, rounds, data = False, 0, None
    for rounds in range(1, max_rounds + 1):
        # Round 1 fits every measurement: gross errors inflate it, so the gate starts loose and tightens.
        data = _scatter(captures, model, None if rounds == 1 else gate_prob, min_frames)
        new = _fit_blocks(data, model)
        old_values, new_values = np.array(list(model.to_dict().values())), np.array(list(new.to_dict().values()))
        change = float(np.max(np.abs(new_values - old_values) / np.maximum(old_values, 1e-3)))
        model = new
        if change < rtol and (rounds > 1 or gate_prob is None):
            converged = True
            break
    assert data is not None
    return NoiseFit(model=model, rounds=rounds, converged=converged, n_measurements=len(data.group),
                    n_objects=data.n_groups, n_rejected=data.n_rejected)


# --------------------------------------------------------------------------------------------------------
# Evaluating one capture against ground truth
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EstimateRecord:
    """One estimate of one object's state, with its error against ground truth.

    Parameters
    ----------
    video : str
        Capture (video).
    source : str
        Depth source.
    estimator : str
        One of :data:`ESTIMATORS`.
    object_id : str
        Ground-truth object.
    label : str
        Its class.
    frame : str
        View of a single-frame estimate; empty for a fused one.
    n_frames : int
        Views in which the object was detected in this capture.
    n_used : int
        Measurements the estimate used (1 for single-frame; accepted ones for fused).
    error : numpy.ndarray, shape (7,)
        Estimate minus truth (:func:`~measured_space.estimation.state_error`); metres and radians.
    sigma : numpy.ndarray, shape (7,)
        Predicted standard deviation of each component (square root of the covariance diagonal).
    nees : float
        ``e^T P^{-1} e`` with the estimator's (marginal) covariance; chi-square(7) if consistent.
    p_gross : float
        Gross-error rate of a fused estimate (NaN for single-frame).
    along_ray_error : float
        Single-frame only: measured minus true camera-frame centre, along the true viewing ray, metres
        (positive = measured deeper). NaN for fused estimates.
    log_depth_ratio : float
        Single-frame only: ``log(|c_measured| / |c_true|)`` in the camera frame. NaN for fused estimates.
    range_m : float
        Single-frame only: range from the camera to the true box centre, metres. NaN for fused estimates.
    """

    video: str
    source: str
    estimator: str
    object_id: str
    label: str
    frame: str
    n_frames: int
    n_used: int
    error: Array
    sigma: Array
    nees: float
    p_gross: float = float("nan")
    along_ray_error: float = float("nan")
    log_depth_ratio: float = float("nan")
    range_m: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dictionary (NaN becomes None).

        Returns
        -------
        dict of str to Any
            The fields, arrays as lists.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = _jsonable(value.tolist() if isinstance(value, np.ndarray) else value)
        return out


@dataclass(frozen=True)
class CaptureResult:
    """Evaluation of one capture: every estimate of every evaluated object, plus capture-level quantities.

    Parameters
    ----------
    video : str
        Capture (video).
    source : str
        Depth source.
    records : list of EstimateRecord
        Single-frame records (one per measurement of an evaluated object) and fused records (one per evaluated
        object and fusion estimator).
    n_measurements : int
        Measurements in the capture (all objects).
    n_objects : int
        Objects detected in the capture.
    n_evaluated : int
        Objects with at least ``min_frames`` views, the ones evaluated.
    log_scale : float
        Scale-aware fusion's posterior mean of the capture's log scale.
    log_scale_sd : float
        Its posterior standard deviation.
    rejected : dict of str to int
        Measurements rejected by the gate, per fusion estimator.
    nis_mean : dict of str to float
        Mean leave-one-out NIS of accepted, testable measurements per fusion estimator (7 if consistent).
    converged : bool
        Whether scale-aware fusion converged.
    """

    video: str
    source: str
    records: list[EstimateRecord]
    n_measurements: int
    n_objects: int
    n_evaluated: int
    log_scale: float
    log_scale_sd: float
    rejected: dict[str, int]
    nis_mean: dict[str, float]
    converged: bool

    def capture_summary(self) -> dict[str, Any]:
        """Capture-level quantities, without the per-object records.

        Returns
        -------
        dict of str to Any
            Video, source, counts, log scale and its sd, the single-frame median log depth ratio against
            ground truth (a reference for the log scale), gate rejections and NIS means.
        """
        ratios = [r.log_depth_ratio for r in self.records if r.estimator == "single"]
        return {"video": self.video, "source": self.source, "n_measurements": self.n_measurements,
                "n_objects": self.n_objects, "n_evaluated": self.n_evaluated, "log_scale": self.log_scale,
                "log_scale_sd": self.log_scale_sd,
                "gt_median_log_depth_ratio": _jsonable(float(np.median(ratios)) if ratios else float("nan")),
                "rejected": dict(self.rejected), "nis_mean": {k: _jsonable(v) for k, v in self.nis_mean.items()},
                "converged": self.converged}


def _fused_records(estimate: CaptureEstimate, name: str, truth: Mapping[str, Array], evaluated: Sequence[str],
                   frames: Mapping[str, int], labels: Mapping[str, str], video: str,
                   source: str) -> list[EstimateRecord]:
    """Records of a fused estimate for the evaluated objects.

    Parameters
    ----------
    estimate : CaptureEstimate
        The fused capture.
    name : str
        Estimator name.
    truth : mapping of str to numpy.ndarray, shape (7,)
        True state per object.
    evaluated : sequence of str
        Objects to report.
    frames : mapping of str to int
        Views per object.
    labels : mapping of str to str
        Class per object.
    video : str
        Capture.
    source : str
        Depth source.

    Returns
    -------
    list of EstimateRecord
        One per evaluated object.
    """
    used = estimate.n_measurements - estimate.n_rejected
    p_gross = estimate.p_gross
    records = []
    for object_id in evaluated:
        i = estimate.index(object_id)
        cov = estimate.cov(object_id)
        error = state_error(estimate.theta[i], truth[object_id])
        records.append(EstimateRecord(video, source, name, object_id, labels.get(object_id, ""), "",
                                      frames[object_id], int(used[i]), error, np.sqrt(np.diag(cov)),
                                      float(nees(error, cov)), p_gross=float(p_gross[i])))
    return records


def _mean_nis(estimate: CaptureEstimate) -> float:
    """Mean leave-one-out NIS of the accepted, testable measurements.

    Parameters
    ----------
    estimate : CaptureEstimate
        A fused capture.

    Returns
    -------
    float
        The mean (7 under a consistent model), or NaN if no measurement is testable.
    """
    values = estimate.nis[estimate.accepted & np.isfinite(estimate.nis)]
    return float(np.mean(values)) if values.size else float("nan")


def evaluate_capture(detections: Sequence[Detection], truth: Mapping[str, ArrayLike], model: NoiseModel, *,
                     scale_prior_sigma: float, gate_prob: float | None = 0.997, min_frames: int = 2,
                     labels: Mapping[str, str] | None = None) -> CaptureResult:
    """Estimate every object of one capture three ways and score each estimate against ground truth.

    All of the capture's measurements are fused, but only objects detected in at least ``min_frames`` views
    are evaluated.

    Parameters
    ----------
    detections : sequence of Detection
        One capture's matched detections (one video, one depth source), at most one per view and object.
    truth : mapping of str to array_like, shape (7,)
        True state (:func:`~measured_space.estimation.box_to_state` of the ground-truth box) per object id.
    model : NoiseModel
        Measurement noise model.
    scale_prior_sigma : float
        Prior standard deviation of the capture's log scale for scale-aware fusion.
    gate_prob : float or None, default 0.997
        Chi-square(7) innovation-gate probability of both fusion estimators; None disables gating.
    min_frames : int, default 2
        Fewest views an object needs to be evaluated.
    labels : mapping of str to str or None, default None
        Class per object id, for reporting.

    Returns
    -------
    CaptureResult
        Records of all three estimators, the fitted log scale, gate rejections and NIS.

    Raises
    ------
    ValueError
        If there are no detections, they span more than one capture, or an object has no ground truth.
    """
    if not detections:
        raise ValueError("no detections")
    captures = {(d.video, d.source) for d in detections}
    if len(captures) != 1:
        raise ValueError(f"detections span {len(captures)} captures; evaluate one at a time")
    (video, source), labels = captures.pop(), labels or {}
    missing = sorted({d.object_id for d in detections} - set(truth))
    if missing:
        raise ValueError(f"no ground truth for objects {missing}")
    truth = {k: np.asarray(v, dtype=np.float64) for k, v in truth.items()}
    measurements = measurements_from_detections(detections, model)
    frames: dict[str, set[str]] = defaultdict(set)
    for d in detections:
        frames[d.object_id].add(d.frame)
    n_frames = {k: len(v) for k, v in frames.items()}
    evaluated = [k for k in dict.fromkeys(d.object_id for d in detections) if n_frames[k] >= min_frames]

    records: list[EstimateRecord] = []
    for d, m in zip(detections, measurements):
        if n_frames[d.object_id] < min_frames:
            continue
        theta, cov = single_frame_estimate(m)
        error = state_error(theta, truth[d.object_id])
        true_cam = m.T_cw[:3, :3] @ truth[d.object_id][CENTER] + m.T_cw[:3, 3]
        true_range = float(np.linalg.norm(true_cam))
        records.append(EstimateRecord(
            video, source, "single", d.object_id, labels.get(d.object_id, ""), d.frame, n_frames[d.object_id], 1,
            error, np.sqrt(np.diag(cov)), float(nees(error, cov)),
            along_ray_error=float((m.z[CENTER] - true_cam) @ true_cam / true_range),
            log_depth_ratio=float(np.log(np.linalg.norm(m.z[CENTER]) / true_range)), range_m=true_range))

    naive = naive_fuse(measurements, gate_prob=gate_prob)
    aware = fuse_capture(measurements, scale_prior_sigma=scale_prior_sigma, gate_prob=gate_prob)
    for estimate, name in ((naive, "naive"), (aware, "scale_aware")):
        records += _fused_records(estimate, name, truth, evaluated, n_frames, labels, video, source)
    return CaptureResult(video=video, source=source, records=records, n_measurements=len(measurements),
                         n_objects=len(n_frames), n_evaluated=len(evaluated), log_scale=float(aware.log_scale),
                         log_scale_sd=float(np.sqrt(aware.log_scale_var)),
                         rejected={"naive": int(np.count_nonzero(~naive.accepted)),
                                   "scale_aware": int(np.count_nonzero(~aware.accepted))},
                         nis_mean={"naive": _mean_nis(naive), "scale_aware": _mean_nis(aware)},
                         converged=bool(aware.converged))


# --------------------------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Replace non-finite floats with None so that the value serialises as strict JSON.

    Parameters
    ----------
    value : Any
        A number, string, list, dict or None.

    Returns
    -------
    Any
        The same structure with NaN and infinities replaced by None and numpy scalars by Python ones.
    """
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def estimator_stats(records: Sequence[EstimateRecord]) -> dict[str, Any]:
    """Aggregate statistics of one estimator's records.

    Parameters
    ----------
    records : sequence of EstimateRecord
        Records of one estimator (and typically one depth source).

    Returns
    -------
    dict of str to Any
        ``n``; ``nees_mean``, ``nees_median``, ``nees_bounds`` (two-sided 95% chi-square bounds on the mean
        of ``n`` independent chi-square(7) samples), ``nees_mean_over_upper``, ``frac_nees_above_chi2_95``
        (expected 0.05); ``coverage_1sigma`` and ``coverage_2sigma`` per component group of
        :data:`COMPONENT_GROUPS` and ``coverage_by_component`` (k = 1, 2); error magnitudes
        ``center_error_m`` (norm), ``size_error_m`` (absolute, sides and height pooled) and
        ``yaw_error_deg`` (absolute) as median and mean; ``sigma_median`` per component; and, where
        defined, ``mean_p_gross`` and ``frac_objects_p_gross_positive`` (fused), and ``along_ray_error_m``
        (median and mean), ``log_depth_ratio`` (median) and ``range_m`` (median) (single-frame).

    Raises
    ------
    ValueError
        If there are no records.
    """
    if not records:
        raise ValueError("no records")
    error = np.stack([r.error for r in records])
    sigma = np.stack([r.sigma for r in records])
    values = np.array([r.nees for r in records])
    inside = {k: np.abs(error) <= k * sigma for k in (1, 2)}
    lower, upper = chi2_bounds(STATE_DIM, len(records))
    center = np.linalg.norm(error[:, CENTER], axis=1)
    size = np.abs(error[:, 3:6]).ravel()
    yaw = np.degrees(np.abs(error[:, YAW]))
    out: dict[str, Any] = {
        "n": len(records),
        "nees_mean": float(np.mean(values)), "nees_median": float(np.median(values)),
        "nees_bounds": [lower, upper], "nees_mean_over_upper": float(np.mean(values) / upper),
        "frac_nees_above_chi2_95": float(np.mean(values > stats.chi2.ppf(0.95, STATE_DIM))),
        "coverage_1sigma": {g: float(np.mean(inside[1][:, list(c)])) for g, c in COMPONENT_GROUPS.items()},
        "coverage_2sigma": {g: float(np.mean(inside[2][:, list(c)])) for g, c in COMPONENT_GROUPS.items()},
        "coverage_by_component": {name: [float(np.mean(inside[1][:, j])), float(np.mean(inside[2][:, j]))]
                                  for j, name in enumerate(COMPONENT_NAMES)},
        "center_error_m": {"median": float(np.median(center)), "mean": float(np.mean(center))},
        "size_error_m": {"median": float(np.median(size)), "mean": float(np.mean(size))},
        "yaw_error_deg": {"median": float(np.median(yaw)), "mean": float(np.mean(yaw))},
        "sigma_median": {name: float(np.median(sigma[:, j])) for j, name in enumerate(COMPONENT_NAMES)},
    }
    p_gross = np.array([r.p_gross for r in records])
    if np.isfinite(p_gross).any():
        out["mean_p_gross"] = float(np.nanmean(p_gross))
        out["frac_objects_p_gross_positive"] = float(np.mean(p_gross[np.isfinite(p_gross)] > 0))
    along = np.array([r.along_ray_error for r in records])
    if np.isfinite(along).any():
        out["along_ray_error_m"] = {"median": float(np.nanmedian(along)), "mean": float(np.nanmean(along))}
        out["log_depth_ratio"] = {"median": float(np.nanmedian([r.log_depth_ratio for r in records]))}
        out["range_m"] = {"median": float(np.nanmedian([r.range_m for r in records]))}
    return out


def paired_along_ray_differences(results: Sequence[CaptureResult]) -> dict[str, Any]:
    """Along-ray centre offsets between depth sources, paired by video, view and object.

    The same mask is lifted with every depth source, so pairing single-frame records of the same view and
    object isolates the depth source's effect on where the box centre lands along the viewing ray.

    Parameters
    ----------
    results : sequence of CaptureResult
        Evaluated captures of several depth sources.

    Returns
    -------
    dict of str to Any
        For each pair ``"a-b"`` of sources: ``n`` pairs, and the ``median`` and ``mean`` of
        ``along_ray_error_a - along_ray_error_b`` in metres (positive = source ``a`` puts the centre deeper).
    """
    along: dict[str, dict[tuple[str, str, str], float]] = defaultdict(dict)
    for result in results:
        for r in result.records:
            if r.estimator == "single":
                along[r.source][(r.video, r.frame, r.object_id)] = r.along_ray_error
    sources = sorted(along)
    out: dict[str, Any] = {}
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            keys = sorted(set(along[a]) & set(along[b]))
            if keys:
                diff = np.array([along[a][k] - along[b][k] for k in keys])
                out[f"{a}-{b}"] = {"n": len(keys), "median": float(np.median(diff)), "mean": float(np.mean(diff))}
    return out


def summarize(results: Sequence[CaptureResult]) -> dict[str, Any]:
    """Aggregate capture results per depth source and estimator.

    Parameters
    ----------
    results : sequence of CaptureResult
        Evaluated captures.

    Returns
    -------
    dict of str to Any
        ``{"by_source": {source: {"estimators": {estimator: stats}, "gate": ..., "nis_mean": ...,
        "naive_over_scale_aware_nees_mean": float}}, "captures": [capture summaries],
        "paired_along_ray_difference_m": ...}`` (see :func:`paired_along_ray_differences`), with estimator
        statistics from :func:`estimator_stats`. ``gate`` counts measurements and rejections per fusion
        estimator; ``nis_mean`` is the measurement-weighted mean NIS of accepted measurements. Strict JSON
        (non-finite values are None).
    """
    by_source: dict[str, Any] = {}
    for source in sorted({r.source for r in results}):
        captures = [r for r in results if r.source == source]
        records = [rec for r in captures for rec in r.records]
        estimators = {name: estimator_stats([rec for rec in records if rec.estimator == name])
                      for name in ESTIMATORS if any(rec.estimator == name for rec in records)}
        n_measurements = sum(r.n_measurements for r in captures)
        gate = {name: {"rejected": sum(r.rejected[name] for r in captures), "measurements": n_measurements,
                       "rejected_fraction": sum(r.rejected[name] for r in captures) / n_measurements}
                for name in ("naive", "scale_aware")}
        weights = np.array([r.n_measurements for r in captures], dtype=np.float64)
        nis_mean = {}
        for name in ("naive", "scale_aware"):
            values = np.array([r.nis_mean[name] for r in captures])
            ok = np.isfinite(values)
            nis_mean[name] = float(np.average(values[ok], weights=weights[ok])) if ok.any() else float("nan")
        entry: dict[str, Any] = {"n_captures": len(captures), "n_evaluated_objects": sum(r.n_evaluated
                                                                                          for r in captures),
                                 "estimators": estimators, "gate": gate, "nis_mean": nis_mean}
        if "naive" in estimators and "scale_aware" in estimators:
            entry["naive_over_scale_aware_nees_mean"] = (estimators["naive"]["nees_mean"]
                                                         / estimators["scale_aware"]["nees_mean"])
        by_source[source] = entry
    return _jsonable({"by_source": by_source, "captures": [r.capture_summary() for r in results],
                      "paired_along_ray_difference_m": paired_along_ray_differences(results)})


TABLE_COLUMNS = ("source", "estimator", "n", "nees_mean", "nees_median", "nees_lower", "nees_upper",
                 "frac_nees_above_chi2_95", "cov1_center", "cov1_size", "cov1_yaw", "cov1_all", "cov2_center",
                 "cov2_size", "cov2_yaw", "cov2_all", "center_err_median_m", "size_err_median_m",
                 "yaw_err_median_deg", "rejected_fraction", "mean_p_gross")
"""Columns of the compact table (:func:`summary_table`)."""


def summary_table(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten a summary into one row per depth source and estimator.

    Parameters
    ----------
    summary : mapping
        Output of :func:`summarize`.

    Returns
    -------
    list of dict of str to Any
        Rows keyed by :data:`TABLE_COLUMNS`; ``rejected_fraction`` and ``mean_p_gross`` are None for
        single-frame rows.
    """
    rows = []
    for source, entry in summary["by_source"].items():
        for name, s in entry["estimators"].items():
            row: dict[str, Any] = {"source": source, "estimator": name, "n": s["n"], "nees_mean": s["nees_mean"],
                                   "nees_median": s["nees_median"], "nees_lower": s["nees_bounds"][0],
                                   "nees_upper": s["nees_bounds"][1],
                                   "frac_nees_above_chi2_95": s["frac_nees_above_chi2_95"]}
            for k in (1, 2):
                for group in ("center", "size", "yaw", "all"):
                    row[f"cov{k}_{group}"] = s[f"coverage_{k}sigma"][group]
            row["center_err_median_m"] = s["center_error_m"]["median"]
            row["size_err_median_m"] = s["size_error_m"]["median"]
            row["yaw_err_median_deg"] = s["yaw_error_deg"]["median"]
            row["rejected_fraction"] = entry["gate"][name]["rejected_fraction"] if name in entry["gate"] else None
            row["mean_p_gross"] = s.get("mean_p_gross")
            rows.append(row)
    return rows


def write_records(results: Sequence[CaptureResult], path: Path) -> int:
    """Write every per-object record as JSON lines.

    Parameters
    ----------
    results : sequence of CaptureResult
        Evaluated captures.
    path : pathlib.Path
        Output file; its parent directory is created if needed.

    Returns
    -------
    int
        Number of records written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as handle:
        for result in results:
            for record in result.records:
                handle.write(json.dumps(record.to_dict()) + "\n")
                n += 1
    return n


# --------------------------------------------------------------------------------------------------------
# Reading the automatic labels
# --------------------------------------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    list of dict
        One dictionary per non-empty line.
    """
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_detections(labels_dir: Path) -> tuple[list[Detection], dict[str, int]]:
    """Read matched detections from the automatic-label files of :mod:`measured_space.pipeline`.

    ``detections.jsonl`` has one row per view, detection and depth source, with the view's ``view_key``,
    ``scene`` (the video), ``frame``, ``source``, the lifted world ``box`` (``center``, ``size``,
    ``rotation`` with the box axes as columns), the lift ``features``, a ``status`` (``"matched"``,
    ``"false_positive"`` or ``"unboxed"``) and the ``match`` (``object_id``, ``label``, ...). ``views.jsonl``
    gives each view's ``T_wc``. Only matched rows become detections.

    Parameters
    ----------
    labels_dir : pathlib.Path
        Folder with ``detections.jsonl`` and ``views.jsonl``.

    Returns
    -------
    detections : list of Detection
        Detections matched to a ground-truth object, in file order.
    counts : dict of str to int
        Row counts: ``rows``, ``matched``, ``false_positive``, ``unboxed`` and ``duplicate_matches`` (a second
        matched row for the same view, source and object, which :func:`group_captures` drops).

    Raises
    ------
    KeyError
        If a detection's view is missing from ``views.jsonl``.
    """
    T_wc = {row["view_key"]: np.asarray(row["T_wc"], dtype=np.float64)
            for row in _read_jsonl(labels_dir / "views.jsonl")}
    T_cw = {key: invert_pose(pose) for key, pose in T_wc.items()}
    counts = {"rows": 0, "matched": 0, "false_positive": 0, "unboxed": 0, "duplicate_matches": 0}
    detections: list[Detection] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in _read_jsonl(labels_dir / "detections.jsonl"):
        counts["rows"] += 1
        status = row["status"] if row.get("box") is not None else "unboxed"
        if status != "matched" or row.get("match") is None:
            counts["unboxed" if status == "unboxed" else "false_positive"] += 1
            continue
        counts["matched"] += 1
        match, box, features = row["match"], row["box"], row["features"]
        key = (str(row["scene"]), str(row["source"]), str(row["frame"]), str(match["object_id"]))
        counts["duplicate_matches"] += key in seen
        seen.add(key)
        detections.append(Detection(
            video=key[0], source=key[1], frame=key[2], object_id=key[3], T_cw=T_cw[row["view_key"]],
            box=OrientedBox(box["center"], box["size"], box["rotation"]),
            boundary_fraction=features["boundary_fraction"], border_touch=features["border_touch"],
            label=str(match.get("label", ""))))
    return detections, counts
