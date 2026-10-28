"""Object state estimation: fuse per-frame box measurements, with a shared per-capture scale error.

Each object is treated as a state to estimate, as a landmark is in SLAM, so that a fused box comes with a
covariance that can be audited against independent references.

Object model
------------
A gravity-aligned box (world up is +z) has the state

    theta = (c_w, s_xy, h, psi)  in R^7:
    centre in the world (m), the two horizontal side lengths (m), height (m), yaw about +z (rad).

A frame with world-to-camera pose ``T_cw = (R_cw, t_cw)`` measures

    z = (c_c, s_xy, h, psi) + v,    v ~ N(0, R),

where ``c_c`` is the box centre in the camera frame and ``psi`` its heading about world +z (the detected
box rotated into the world with the frame's known camera rotation). The caller supplies ``R``; it is meant
to come from image evidence such as range, visible fraction and mask quality. Camera poses are treated as
exact: pose error, if any, belongs in ``R``.

Shared scale
------------
Geometry derived from depth (monocular, or a reconstruction) can be off by one unknown factor
``s = exp(sigma)`` for a whole capture:

    c_c ~ s (R_cw c_w + t_cw),    s_xy ~ s s_xy,    h ~ s h,    psi unchanged.

Every measurement of every object in a capture shares that one ``sigma``, so its errors are correlated
across frames *and* across objects. Fusing frames as if they were independent (:func:`naive_fuse`) shrinks
the covariance like 1/N, but the scale error does not average out, so the fused box is overconfident.
:func:`fuse_capture` instead estimates ``sigma`` jointly with every object of the capture (prior
``sigma ~ N(0, sigma_s^2)``) by Gauss-Newton on the sum of squared Mahalanobis residuals. Its posterior
covariance (the inverse Gauss-Newton Hessian) carries the cross-covariances between objects that the shared
scale induces. Camera translation between frames is what makes ``sigma`` observable beyond its prior:
bearings fix where an object is, the metric baseline fixes how far away that is, and measured depth must
agree.

The Hessian is an "arrowhead": one 7x7 block per object plus a row and column for ``sigma``. Eliminating
``sigma`` with a Schur complement gives a compact exact form of the posterior,

    theta_i = theta_hat_i + g_i (sigma - sigma_hat) + e_i,    e_i ~ N(0, C_i) independent,

so that ``Cov(theta_i, theta_j) = delta_ij C_i + g_i g_j^T var(sigma)`` and
``Cov(theta_i, sigma) = g_i var(sigma)``. :class:`CaptureEstimate` stores ``C_i`` (the covariance of an
object given the scale), ``g_i`` (how the object's estimate moves with the scale) and ``var(sigma)``, and
builds any block on demand.

Yaw
---
A box looks the same after a yaw of pi, and after pi/2 with its two horizontal sides swapped. Each
measurement is replaced by its equivalent whose yaw is closest to the current estimate (sides swapped, and
its covariance permuted, when needed) before residuals are formed, and yaw residuals are wrapped to
``[-pi, pi)``. This assumes per-frame yaw errors well below pi/4.

Gross errors
------------
A measurement's innovation is tested against the estimate from the *other* accepted measurements. For a
measurement inside a linear(ised) fit with residual ``r``, Jacobian ``J`` and posterior covariance ``P``,
that leave-one-out statistic equals ``r^T (R - J P J^T)^{-1} r``, so it needs no refit; for an excluded
measurement it is ``r^T (R + J P J^T)^{-1} r``. Either is chi-square with 7 degrees of freedom under the
model. Each round, the worst failing measurement of each object is rejected and the capture refitted,
until none fails; then rejected measurements that pass against the current fit are re-admitted (an inlier
can fail early while gross errors still pull the fit) and the loop resumes.

Per object, ``p_gross`` is a method-of-moments gross-error rate. If every gross error is caught and an
inlier is rejected with probability ``alpha = 1 - gate_prob``, the rejected fraction ``f`` has expectation
``p + (1 - p) alpha``, so ``p_gross = clip((f - alpha) / (1 - alpha), 0, 1)``.

Consistency
-----------
NEES ``e^T P^{-1} e`` against ground truth audits calibration; NIS ``nu^T S^{-1} nu`` needs no ground
truth. Under a correct model each is chi-square with the state (or measurement) dimension as degrees of
freedom, and the average of N independent samples is checked against chi-square(N dof) / N bounds
(Bar-Shalom, Li & Kirubarajan, *Estimation with Applications to Tracking and Navigation*, 2001, sec. 5.4).

Moving agents and answers
-------------------------
:func:`cv_kalman_filter` tracks an agent's ground-plane position and velocity with a constant-velocity
(continuous white-noise acceleration) model, fusing any number of camera measurements per step.
:func:`distance_with_uncertainty` and :func:`prob_positive` carry a posterior through to a metric answer
(a distance and its sigma) or a categorical one (the probability that a signed quantity is positive).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from scipy.linalg import block_diag

from measured_space.geometry import Array, OrientedBox, axis_angle_to_matrix, invert_pose

STATE_DIM = 7
"""Size of an object's state and of a box measurement."""
CENTER = slice(0, 3)
"""Index of the centre (world frame in a state, camera frame in a measurement)."""
SIZE_XY = slice(3, 5)
"""Index of the two horizontal side lengths."""
HEIGHT = 5
"""Index of the height."""
YAW = 6
"""Index of the yaw about world +z."""

# Permutation of a state or measurement that exchanges its two horizontal sides.
_SWAP = np.array([0, 1, 2, 4, 3, 5, 6])


# --------------------------------------------------------------------------------------------------------
# Box parametrisation
# --------------------------------------------------------------------------------------------------------


def wrap_angle(angle: ArrayLike) -> Array:
    """Wrap angles to ``[-pi, pi)``.

    Parameters
    ----------
    angle : array_like
        Angles, radians.

    Returns
    -------
    numpy.ndarray
        The same angles, wrapped; same shape as ``angle``.
    """
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def canonicalize_yaw(size_xy: ArrayLike, yaw: ArrayLike,
                     reference_yaw: ArrayLike) -> tuple[Array, Array, NDArray[np.bool_]]:
    """Choose, among a box's four equivalent yaws, the one closest to a reference.

    A gravity-aligned box is unchanged by a yaw of pi, and by a yaw of pi/2 with its horizontal sides
    swapped. The returned yaw differs from the reference by at most pi/4 (modulo 2 pi).

    Parameters
    ----------
    size_xy : array_like, shape (..., 2)
        Horizontal side lengths, metres.
    yaw : array_like, shape (...)
        Yaw about world +z, radians.
    reference_yaw : array_like, shape (...)
        Yaw to be close to, radians; broadcasts against ``yaw``.

    Returns
    -------
    size_xy : numpy.ndarray, shape (..., 2)
        Side lengths of the chosen equivalent (swapped where it is a quarter turn away).
    yaw : numpy.ndarray, shape (...)
        Yaw of the chosen equivalent, wrapped to ``[-pi, pi)``.
    swapped : numpy.ndarray of bool, shape (...)
        Whether the sides were swapped.
    """
    size_xy = np.asarray(size_xy, dtype=np.float64)
    reference_yaw = np.asarray(reference_yaw, dtype=np.float64)
    offset = wrap_angle(np.asarray(yaw, dtype=np.float64) - reference_yaw)
    quarter_turns = np.rint(offset / (np.pi / 2))
    swapped = (quarter_turns.astype(np.int64) % 2) == 1
    sizes = np.where(swapped[..., None], size_xy[..., ::-1], size_xy)
    return sizes, wrap_angle(reference_yaw + offset - quarter_turns * np.pi / 2), swapped


def box_to_state(box: OrientedBox) -> Array:
    """Express a gravity-aligned world-frame box as a state vector.

    The box axis closest to world +z is taken as its height axis; the yaw is the heading of the first of
    the other two axes.

    Parameters
    ----------
    box : OrientedBox
        Box in the world frame, metres.

    Returns
    -------
    numpy.ndarray, shape (7,)
        ``(cx, cy, cz, size_x, size_y, height, yaw)``; metres, and radians in ``[-pi, pi)``.
    """
    up_axis = int(np.argmax(np.abs(box.rotation[2])))
    first, second = (k for k in range(3) if k != up_axis)
    heading = box.rotation[:, first]
    yaw = wrap_angle(np.arctan2(heading[1], heading[0]))
    return np.array([*box.center, box.size[first], box.size[second], box.size[up_axis], yaw])


def state_to_box(theta: ArrayLike) -> OrientedBox:
    """Build the world-frame box a state vector describes.

    Parameters
    ----------
    theta : array_like, shape (7,)
        ``(cx, cy, cz, size_x, size_y, height, yaw)``; metres and radians.

    Returns
    -------
    OrientedBox
        Box with its height along world +z.
    """
    theta = np.asarray(theta, dtype=np.float64)
    return OrientedBox(theta[CENTER], [theta[3], theta[4], theta[HEIGHT]],
                       axis_angle_to_matrix([0.0, 0.0, theta[YAW]]))


def box_to_measurement(box_cam: OrientedBox, T_cw: Array) -> Array:
    """Express a camera-frame box detection as a measurement vector.

    Parameters
    ----------
    box_cam : OrientedBox
        Detected box in the camera frame, metres; assumed gravity-aligned once rotated into the world.
    T_cw : numpy.ndarray, shape (4, 4)
        The frame's world-to-camera pose.

    Returns
    -------
    numpy.ndarray, shape (7,)
        ``(c_c, size_x, size_y, height, yaw)``: camera-frame centre (m), sides and height (m), and yaw about
        world +z (rad).
    """
    z = box_to_state(box_cam.transformed(invert_pose(T_cw)))
    z[CENTER] = box_cam.center
    return z


def predict_measurement(theta: ArrayLike, T_cw: Array, log_scale: float = 0.0) -> Array:
    """Predict the measurement a frame makes of a state, under a capture scale error.

    Parameters
    ----------
    theta : array_like, shape (7,)
        Object state (world frame); metres and radians.
    T_cw : numpy.ndarray, shape (4, 4)
        The frame's world-to-camera pose.
    log_scale : float, default 0.0
        Log of the capture's scale factor.

    Returns
    -------
    numpy.ndarray, shape (7,)
        Noise-free measurement ``(c_c, size_x, size_y, height, yaw)``.
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(1, STATE_DIM)
    pred, _, _ = _predict(theta, T_cw[None, :3, :3], T_cw[None, :3, 3], log_scale)
    return pred[0]


def state_error(estimate: ArrayLike, truth: ArrayLike) -> Array:
    """Estimation error of box states, respecting the box's yaw symmetries.

    The truth is first replaced by its equivalent closest in yaw to the estimate, so a quarter-turn
    re-labelling of the sides is not counted as an error.

    Parameters
    ----------
    estimate : array_like, shape (..., 7)
        Estimated states.
    truth : array_like, shape (..., 7)
        True states.

    Returns
    -------
    numpy.ndarray, shape (..., 7)
        ``estimate - truth``, with the yaw component wrapped to ``[-pi, pi)``.
    """
    estimate = np.asarray(estimate, dtype=np.float64)
    truth = np.array(truth, dtype=np.float64)
    sizes, yaw, _ = canonicalize_yaw(truth[..., SIZE_XY], truth[..., YAW], estimate[..., YAW])
    truth[..., SIZE_XY] = sizes
    error = estimate - truth
    error[..., YAW] = wrap_angle(estimate[..., YAW] - yaw)
    return error


# --------------------------------------------------------------------------------------------------------
# Measurements and their stacked form
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoxMeasurement:
    """One frame's measurement of one object's box.

    Parameters
    ----------
    object_id : str
        The object measured.
    T_cw : numpy.ndarray, shape (4, 4)
        The frame's world-to-camera pose, treated as exact.
    z : numpy.ndarray, shape (7,)
        ``(c_c, size_x, size_y, height, yaw)``: camera-frame centre (m), horizontal sides and height (m),
        and yaw about world +z (rad). See :func:`box_to_measurement`.
    cov : numpy.ndarray, shape (7, 7)
        Measurement noise covariance ``R``, symmetric positive definite, in the units of ``z``.
    frame : str, default ""
        Frame identifier, for reporting.
    """

    object_id: str
    T_cw: Array
    z: Array
    cov: Array
    frame: str = ""

    def __post_init__(self) -> None:
        """Coerce the arrays to float64 of the documented shapes.

        Raises
        ------
        ValueError
            If an array cannot be reshaped to its documented shape.
        """
        object.__setattr__(self, "T_cw", np.asarray(self.T_cw, dtype=np.float64).reshape(4, 4))
        object.__setattr__(self, "z", np.asarray(self.z, dtype=np.float64).reshape(STATE_DIM))
        object.__setattr__(self, "cov", np.asarray(self.cov, dtype=np.float64).reshape(STATE_DIM, STATE_DIM))


@dataclass(frozen=True)
class _Stack:
    """All measurements of a capture as stacked arrays.

    Parameters
    ----------
    obj : numpy.ndarray of int, shape (K,)
        Index of each measurement's object.
    R_cw : numpy.ndarray, shape (K, 3, 3)
        World-to-camera rotations.
    t_cw : numpy.ndarray, shape (K, 3)
        World-to-camera translations, metres.
    z : numpy.ndarray, shape (K, 7)
        Measurements.
    cov : numpy.ndarray, shape (K, 7, 7)
        Measurement covariances.
    info : numpy.ndarray, shape (K, 7, 7)
        Their inverses.
    n_objects : int
        Number of objects, M.
    """

    obj: NDArray[np.intp]
    R_cw: Array
    t_cw: Array
    z: Array
    cov: Array
    info: Array
    n_objects: int


def _stack(measurements: Sequence[BoxMeasurement]) -> tuple[tuple[str, ...], _Stack]:
    """Stack measurements into arrays, numbering objects in order of first appearance.

    Parameters
    ----------
    measurements : sequence of BoxMeasurement
        The capture's measurements.

    Returns
    -------
    object_ids : tuple of str
        Object identifiers, in index order.
    stack : _Stack
        The stacked measurements.

    Raises
    ------
    ValueError
        If there are no measurements.
    """
    if not measurements:
        raise ValueError("no measurements to fuse")
    index: dict[str, int] = {}
    obj = np.array([index.setdefault(m.object_id, len(index)) for m in measurements], dtype=np.intp)
    poses = np.stack([m.T_cw for m in measurements])
    cov = np.stack([m.cov for m in measurements])
    stack = _Stack(obj=obj, R_cw=poses[:, :3, :3], t_cw=poses[:, :3, 3],
                   z=np.stack([m.z for m in measurements]), cov=cov, info=np.linalg.inv(cov),
                   n_objects=len(index))
    return tuple(index), stack


def _sum_by_object(values: Array, obj: NDArray[np.intp], n_objects: int) -> Array:
    """Sum per-measurement arrays into per-object totals.

    Parameters
    ----------
    values : numpy.ndarray, shape (K, ...)
        One entry per measurement.
    obj : numpy.ndarray of int, shape (K,)
        Each measurement's object index.
    n_objects : int
        Number of objects, M.

    Returns
    -------
    numpy.ndarray, shape (M, ...)
        Per-object sums.
    """
    out = np.zeros((n_objects, *values.shape[1:]))
    np.add.at(out, obj, values)
    return out


def _predict(theta_k: Array, R_cw: Array, t_cw: Array, log_scale: float) -> tuple[Array, Array, Array]:
    """Predicted measurements and their Jacobians.

    Parameters
    ----------
    theta_k : numpy.ndarray, shape (K, 7)
        The state of each measurement's object.
    R_cw : numpy.ndarray, shape (K, 3, 3)
        World-to-camera rotations.
    t_cw : numpy.ndarray, shape (K, 3)
        World-to-camera translations, metres.
    log_scale : float
        Log of the capture's scale factor.

    Returns
    -------
    pred : numpy.ndarray, shape (K, 7)
        Predicted measurements.
    J_theta : numpy.ndarray, shape (K, 7, 7)
        Derivative of each prediction with respect to its object's state.
    J_scale : numpy.ndarray, shape (K, 7)
        Derivative of each prediction with respect to the log scale.
    """
    scale = np.exp(log_scale)
    center_cam = np.einsum("kij,kj->ki", R_cw, theta_k[:, CENTER]) + t_cw
    pred = np.empty_like(theta_k)
    pred[:, CENTER] = scale * center_cam
    pred[:, 3:6] = scale * theta_k[:, 3:6]
    pred[:, YAW] = theta_k[:, YAW]
    J_theta = np.zeros((len(theta_k), STATE_DIM, STATE_DIM))
    J_theta[:, CENTER, CENTER] = scale * R_cw
    J_theta[:, [3, 4, 5], [3, 4, 5]] = scale
    J_theta[:, YAW, YAW] = 1.0
    J_scale = pred.copy()
    J_scale[:, YAW] = 0.0
    return pred, J_theta, J_scale


def _linearize(stack: _Stack, theta: Array, log_scale: float) -> tuple[Array, Array, Array, Array, Array]:
    """Canonicalise every measurement against the current estimate and linearise the model there.

    Parameters
    ----------
    stack : _Stack
        The measurements.
    theta : numpy.ndarray, shape (M, 7)
        Current object states.
    log_scale : float
        Current log scale.

    Returns
    -------
    residual : numpy.ndarray, shape (K, 7)
        Canonicalised measurement minus prediction, yaw wrapped.
    cov : numpy.ndarray, shape (K, 7, 7)
        Measurement covariances, permuted where the sides were swapped.
    info : numpy.ndarray, shape (K, 7, 7)
        Their inverses.
    J_theta : numpy.ndarray, shape (K, 7, 7)
        Jacobians with respect to each measurement's object state.
    J_scale : numpy.ndarray, shape (K, 7)
        Jacobians with respect to the log scale.
    """
    theta_k = theta[stack.obj]
    pred, J_theta, J_scale = _predict(theta_k, stack.R_cw, stack.t_cw, log_scale)
    sizes, yaw, swapped = canonicalize_yaw(stack.z[:, SIZE_XY], stack.z[:, YAW], theta_k[:, YAW])
    residual = stack.z - pred
    residual[:, SIZE_XY] = sizes - pred[:, SIZE_XY]
    residual[:, YAW] = wrap_angle(yaw - pred[:, YAW])
    flip = swapped[:, None, None]
    cov = np.where(flip, stack.cov[:, _SWAP][:, :, _SWAP], stack.cov)
    info = np.where(flip, stack.info[:, _SWAP][:, :, _SWAP], stack.info)
    return residual, cov, info, J_theta, J_scale


def _measurements_in_world(stack: _Stack) -> Array:
    """Each measurement read as a world-frame state, assuming no scale error.

    Parameters
    ----------
    stack : _Stack
        The measurements.

    Returns
    -------
    numpy.ndarray, shape (K, 7)
        ``(R_cw^T (c_c - t_cw), size_x, size_y, height, yaw)``.
    """
    world = stack.z.copy()
    world[:, CENTER] = np.einsum("kji,kj->ki", stack.R_cw, stack.z[:, CENTER] - stack.t_cw)
    return world


def _quarter_turn_mean(yaw: Array, weights: Array) -> float:
    """Weighted mean of angles that are only defined modulo pi/2.

    Parameters
    ----------
    yaw : numpy.ndarray, shape (N,)
        Angles, radians.
    weights : numpy.ndarray, shape (N,)
        Non-negative weights.

    Returns
    -------
    float
        The angle whose fourfold multiple is the weighted circular mean of ``4 yaw``, radians.
    """
    return float(np.angle(np.sum(weights * np.exp(4j * yaw))) / 4.0)


def _robust_init(stack: _Stack) -> Array:
    """Initial object states from per-object medians of the measurements read in the world.

    Parameters
    ----------
    stack : _Stack
        The measurements.

    Returns
    -------
    numpy.ndarray, shape (M, 7)
        Initial states. Medians resist a minority of gross errors, which the gate then removes.
    """
    world = _measurements_in_world(stack)
    theta = np.empty((stack.n_objects, STATE_DIM))
    for i in range(stack.n_objects):
        rows = world[stack.obj == i]
        reference = _quarter_turn_mean(rows[:, YAW], 1.0 / stack.cov[stack.obj == i, YAW, YAW])
        sizes, yaw, _ = canonicalize_yaw(rows[:, SIZE_XY], rows[:, YAW], reference)
        theta[i, CENTER] = np.median(rows[:, CENTER], axis=0)
        theta[i, SIZE_XY] = np.median(sizes, axis=0)
        theta[i, HEIGHT] = np.median(rows[:, HEIGHT])
        theta[i, YAW] = wrap_angle(reference + np.median(wrap_angle(yaw - reference)))
    return theta


# --------------------------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Fit:
    """A fitted capture: estimates and the factored posterior covariance.

    Parameters
    ----------
    theta : numpy.ndarray, shape (M, 7)
        Object states.
    log_scale : float
        Log scale (0 when it is not estimated).
    conditional_cov : numpy.ndarray, shape (M, 7, 7)
        Covariance of each object given the log scale.
    scale_sensitivity : numpy.ndarray, shape (M, 7)
        Change of each object's conditional mean per unit of log scale.
    log_scale_var : float
        Posterior variance of the log scale (0 when it is not estimated).
    iterations : int
        Iterations used.
    converged : bool
        Whether the step fell below the tolerance.
    """

    theta: Array
    log_scale: float
    conditional_cov: Array
    scale_sensitivity: Array
    log_scale_var: float
    iterations: int
    converged: bool


def _gauss_newton_step(stack: _Stack, accepted: NDArray[np.bool_], theta: Array, log_scale: float,
                       scale_info: float | None) -> tuple[Array, float, Array, Array, float]:
    """Solve the linearised capture problem by eliminating each object, then the log scale.

    The Gauss-Newton Hessian is an arrowhead: blocks ``A_i = sum_k J_k^T W_k J_k`` per object, a column
    ``b_i = sum_k J_k^T W_k j_k`` coupling object i to the log scale (``j_k`` is the scale Jacobian), and a
    scale entry. Eliminating ``theta_i`` leaves, per measurement, the projected scale Jacobian
    ``w_k = j_k - J_k A_i^{-1} b_i``; the Schur complement is ``sum_k w_k^T W_k w_k`` plus the prior. Summing
    these non-negative terms avoids the catastrophic cancellation of ``c - b^T A^{-1} b`` when the scale is
    weakly observable, and an error in ``A^{-1} b`` affects it only to second order.

    Parameters
    ----------
    stack : _Stack
        The measurements.
    accepted : numpy.ndarray of bool, shape (K,)
        Which measurements to use.
    theta : numpy.ndarray, shape (M, 7)
        Linearisation point of the objects.
    log_scale : float
        Linearisation point of the log scale.
    scale_info : float or None
        Prior information ``1 / sigma_s^2`` on the log scale, or None when the scale is held fixed.

    Returns
    -------
    step_theta : numpy.ndarray, shape (M, 7)
        Object steps.
    step_scale : float
        Log-scale step (0 when the scale is fixed).
    conditional_cov : numpy.ndarray, shape (M, 7, 7)
        ``A_i^{-1}``, the covariance of each object given the log scale.
    scale_sensitivity : numpy.ndarray, shape (M, 7)
        ``-A_i^{-1} b_i`` (zeros when the scale is fixed).
    log_scale_var : float
        Inverse Schur complement, the posterior variance of the log scale (0 when it is fixed).

    Raises
    ------
    numpy.linalg.LinAlgError
        If the log scale is not observable (a flat prior and no constraining camera motion).
    """
    residual, _, info, J_theta, J_scale = (a[accepted] for a in _linearize(stack, theta, log_scale))
    obj, n = stack.obj[accepted], stack.n_objects
    JtW = np.einsum("kji,kjl->kil", J_theta, info)
    conditional_cov = np.linalg.inv(_sum_by_object(JtW @ J_theta, obj, n))
    step_theta = np.einsum("mij,mj->mi", conditional_cov,
                           _sum_by_object(np.einsum("kij,kj->ki", JtW, residual), obj, n))
    if scale_info is None:
        return step_theta, 0.0, conditional_cov, np.zeros((n, STATE_DIM)), 0.0
    b = _sum_by_object(np.einsum("kij,kj->ki", JtW, J_scale), obj, n)
    u = np.einsum("mij,mj->mi", conditional_cov, b)
    w = J_scale - np.einsum("kij,kj->ki", J_theta, u[obj])
    Ww = np.einsum("kij,kj->ki", info, w)
    schur = float(np.sum(Ww * w)) + scale_info
    data_info = float(np.einsum("ki,kij,kj->", J_scale, info, J_scale))
    if not np.isfinite(schur) or schur <= 1e-10 * data_info:
        raise np.linalg.LinAlgError("the capture's log scale is not observable; give it a finite prior")
    step_scale = (float(np.sum(Ww * residual)) - scale_info * log_scale) / schur
    return step_theta - u * step_scale, step_scale, conditional_cov, -u, 1.0 / schur


def _gauss_newton(stack: _Stack, accepted: NDArray[np.bool_], theta0: Array, log_scale0: float,
                  scale_info: float | None, max_iterations: int, tol: float) -> _Fit:
    """Maximum a posteriori fit of all objects and the log scale by Gauss-Newton.

    Parameters
    ----------
    stack : _Stack
        The measurements.
    accepted : numpy.ndarray of bool, shape (K,)
        Which measurements to use.
    theta0 : numpy.ndarray, shape (M, 7)
        Initial object states.
    log_scale0 : float
        Initial log scale.
    scale_info : float or None
        Prior information on the log scale, or None to hold it at ``log_scale0``.
    max_iterations : int
        Iteration limit.
    tol : float
        Convergence threshold on the largest step component (metres, radians, log units).

    Returns
    -------
    _Fit
        Estimates, with the posterior covariance evaluated at them.
    """
    theta, log_scale = theta0.copy(), float(log_scale0)
    converged, iterations = False, 0
    for iterations in range(1, max_iterations + 1):
        step_theta, step_scale, _, _, _ = _gauss_newton_step(stack, accepted, theta, log_scale, scale_info)
        theta += step_theta
        theta[:, YAW] = wrap_angle(theta[:, YAW])
        log_scale += step_scale
        if max(np.max(np.abs(step_theta)), abs(step_scale)) < tol:
            converged = True
            break
    _, _, conditional_cov, sensitivity, scale_var = _gauss_newton_step(stack, accepted, theta, log_scale,
                                                                       scale_info)
    return _Fit(theta, log_scale, conditional_cov, sensitivity, scale_var, iterations, converged)


def _information_fusion(stack: _Stack, accepted: NDArray[np.bool_], theta0: Array, max_iterations: int,
                        tol: float) -> _Fit:
    """Fuse each object's measurements independently in information form, assuming no scale error.

    With the scale fixed the model is linear, ``z = H theta + d`` with ``H = diag(R_cw, I_4)`` and
    ``d = (t_cw, 0)``, so ``theta = (sum H^T W H)^{-1} sum H^T W (z - d)``. Only the choice of each yaw
    equivalent depends on the estimate, so the fusion is repeated until that choice settles.

    Parameters
    ----------
    stack : _Stack
        The measurements.
    accepted : numpy.ndarray of bool, shape (K,)
        Which measurements to use.
    theta0 : numpy.ndarray, shape (M, 7)
        Estimate the yaw equivalents are first chosen against.
    max_iterations : int
        Limit on re-canonicalisation passes.
    tol : float
        Convergence threshold on the largest change of the estimate.

    Returns
    -------
    _Fit
        Per-object estimates and covariances; the log scale is 0 with zero variance.
    """
    obj, R_cw = stack.obj[accepted], stack.R_cw[accepted]
    H = np.zeros((len(obj), STATE_DIM, STATE_DIM))
    H[:, CENTER, CENTER] = R_cw
    H[:, [3, 4, 5, 6], [3, 4, 5, 6]] = 1.0
    theta, converged, iterations = theta0.copy(), False, 0
    for iterations in range(1, max_iterations + 1):
        reference = theta[obj, YAW]
        y = stack.z[accepted].copy()
        sizes, yaw, swapped = canonicalize_yaw(y[:, SIZE_XY], y[:, YAW], reference)
        y[:, CENTER] -= stack.t_cw[accepted]
        y[:, SIZE_XY] = sizes
        y[:, YAW] = reference + wrap_angle(yaw - reference)   # unwrapped next to the current estimate
        info = stack.info[accepted]
        info = np.where(swapped[:, None, None], info[:, _SWAP][:, :, _SWAP], info)
        HtW = np.einsum("kji,kjl->kil", H, info)
        information = _sum_by_object(HtW @ H, obj, stack.n_objects)
        information_vector = _sum_by_object(np.einsum("kij,kj->ki", HtW, y), obj, stack.n_objects)
        cov = np.linalg.inv(information)
        new = np.einsum("mij,mj->mi", cov, information_vector)
        new[:, YAW] = wrap_angle(new[:, YAW])
        change = np.max(np.abs(state_error(new, theta)))
        theta = new
        if change < tol:
            converged = True
            break
    return _Fit(theta, 0.0, cov, np.zeros((stack.n_objects, STATE_DIM)), 0.0, iterations, converged)


def _innovation_nis(stack: _Stack, fit: _Fit, accepted: NDArray[np.bool_]) -> Array:
    """Leave-one-out normalised innovation squared of every measurement.

    Parameters
    ----------
    stack : _Stack
        The measurements.
    fit : _Fit
        The fit from the accepted measurements.
    accepted : numpy.ndarray of bool, shape (K,)
        Which measurements the fit used.

    Returns
    -------
    numpy.ndarray, shape (K,)
        ``r^T (R - J P J^T)^{-1} r`` for accepted measurements and ``r^T (R + J P J^T)^{-1} r`` for the
        others; chi-square with 7 degrees of freedom under the model. NaN for the only accepted measurement
        of an object, which no other measurement can check.
    """
    residual, cov, _, J_theta, J_scale = _linearize(stack, fit.theta, fit.log_scale)
    # J P J^T over (theta_i, sigma) = J C_i J^T + var(sigma) w w^T with w = J g_i + J_scale.
    w = np.einsum("kij,kj->ki", J_theta, fit.scale_sensitivity[stack.obj]) + J_scale
    JPJ = J_theta @ fit.conditional_cov[stack.obj] @ J_theta.transpose(0, 2, 1)
    JPJ += fit.log_scale_var * w[:, :, None] * w[:, None, :]
    counts = np.bincount(stack.obj[accepted], minlength=stack.n_objects)
    residual_cov = cov + np.where(accepted, -1.0, 1.0)[:, None, None] * JPJ
    eigenvalues = np.linalg.eigvalsh(residual_cov)
    # An object's only measurement has a zero residual covariance; round-off can also leave one indefinite.
    untestable = (accepted & (counts[stack.obj] < 2)) | (eigenvalues[:, 0] <= 1e-12 * eigenvalues[:, -1])
    residual_cov[untestable] = cov[untestable]          # any well-posed matrix; the result is discarded
    nis = np.einsum("ki,ki->k", residual, np.linalg.solve(residual_cov, residual[:, :, None])[:, :, 0])
    nis[untestable] = np.nan
    return nis


def _fit_with_gating(stack: _Stack, fit_fn: Callable[[NDArray[np.bool_], _Fit | None], _Fit],
                     gate_prob: float | None, max_rounds: int) -> tuple[_Fit, NDArray[np.bool_], Array, int]:
    """Fit, then gate measurements by their innovation until the accepted set is stable.

    Each round either rejects, per object, the accepted measurement that fails the gate worst, or, when none
    fails, re-admits every rejected measurement that now passes. For a linear(ised) model a re-admitted
    measurement's leave-one-out statistic equals the value it just passed with, so it is not immediately
    rejected again.

    Parameters
    ----------
    stack : _Stack
        The measurements.
    fit_fn : callable
        ``fit_fn(accepted, previous_fit_or_None) -> _Fit``.
    gate_prob : float or None
        Probability of the chi-square(7) gate, e.g. 0.997; None disables gating.
    max_rounds : int
        Limit on rejection and re-admission rounds.

    Returns
    -------
    fit : _Fit
        Fit from the accepted measurements.
    accepted : numpy.ndarray of bool, shape (K,)
        Measurements kept.
    nis : numpy.ndarray, shape (K,)
        Leave-one-out NIS of every measurement against the final fit.
    rounds : int
        Rounds performed.
    """
    accepted = np.ones(len(stack.obj), dtype=bool)
    threshold = np.inf if gate_prob is None else float(stats.chi2.ppf(gate_prob, STATE_DIM))
    fit, rounds = fit_fn(accepted, None), 0
    while True:
        nis = _innovation_nis(stack, fit, accepted)
        with np.errstate(invalid="ignore"):
            failing = np.flatnonzero(accepted & (nis > threshold))
            passing = ~accepted & (nis <= threshold)
        if rounds >= max_rounds or (failing.size == 0 and not passing.any()):
            return fit, accepted, nis, rounds
        if failing.size:
            order = failing[np.lexsort((-nis[failing], stack.obj[failing]))]
            _, first = np.unique(stack.obj[order], return_index=True)
            accepted[order[first]] = False
        else:
            accepted |= passing
        fit, rounds = fit_fn(accepted, fit), rounds + 1


# --------------------------------------------------------------------------------------------------------
# Results and estimators
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureEstimate:
    """Fused states of every object in a capture, with the joint posterior in factored form.

    The joint posterior over ``(theta_1, ..., theta_M, sigma)`` is
    ``Cov(theta_i, theta_j) = delta_ij C_i + g_i g_j^T var(sigma)`` and
    ``Cov(theta_i, sigma) = g_i var(sigma)``,
    with ``C_i = conditional_cov[i]`` and ``g_i = scale_sensitivity[i]``. Without a scale state (naive
    fusion) ``g`` and ``var(sigma)`` are zero and objects are independent.

    Parameters
    ----------
    object_ids : tuple of str
        Objects, in index order.
    theta : numpy.ndarray, shape (M, 7)
        Posterior means (MAP) of the object states; metres and radians.
    log_scale : float
        Posterior mean of the log scale (0 when not estimated).
    log_scale_var : float
        Posterior variance of the log scale (0 when not estimated).
    conditional_cov : numpy.ndarray, shape (M, 7, 7)
        Covariance of each object given the log scale.
    scale_sensitivity : numpy.ndarray, shape (M, 7)
        Change of each object's conditional mean per unit of log scale.
    scale_estimated : bool
        Whether the log scale was a state.
    measurement_object : numpy.ndarray of int, shape (K,)
        Index of each input measurement's object, in input order.
    accepted : numpy.ndarray of bool, shape (K,)
        Whether each input measurement passed the gate and was used.
    nis : numpy.ndarray, shape (K,)
        Leave-one-out NIS of each input measurement (chi-square, 7 dof, under the model; NaN when untestable).
    gate_prob : float or None
        Gate probability used, or None if gating was off.
    iterations : int
        Iterations of the final fit.
    converged : bool
        Whether the final fit converged.
    """

    object_ids: tuple[str, ...]
    theta: Array
    log_scale: float
    log_scale_var: float
    conditional_cov: Array
    scale_sensitivity: Array
    scale_estimated: bool
    measurement_object: NDArray[np.intp]
    accepted: NDArray[np.bool_]
    nis: Array
    gate_prob: float | None
    iterations: int
    converged: bool

    def index(self, object_id: str) -> int:
        """Look up an object's index.

        Parameters
        ----------
        object_id : str
            The object.

        Returns
        -------
        int
            Its row in ``theta``.

        Raises
        ------
        ValueError
            If the object is not in the capture.
        """
        return self.object_ids.index(object_id)

    def mean(self, object_id: str) -> Array:
        """Posterior mean of one object's state.

        Parameters
        ----------
        object_id : str
            The object.

        Returns
        -------
        numpy.ndarray, shape (7,)
            ``(cx, cy, cz, size_x, size_y, height, yaw)``.
        """
        return self.theta[self.index(object_id)].copy()

    def cov(self, object_id: str) -> Array:
        """Marginal posterior covariance of one object's state.

        Parameters
        ----------
        object_id : str
            The object.

        Returns
        -------
        numpy.ndarray, shape (7, 7)
            Covariance, including the uncertainty the shared scale contributes.
        """
        return self.cross_cov(object_id, object_id)

    def cross_cov(self, object_a: str, object_b: str) -> Array:
        """Posterior cross-covariance between two objects' states.

        Parameters
        ----------
        object_a : str
            First object.
        object_b : str
            Second object.

        Returns
        -------
        numpy.ndarray, shape (7, 7)
            ``Cov(theta_a, theta_b)``; nonzero for distinct objects only through the shared scale.
        """
        i, j = self.index(object_a), self.index(object_b)
        block = self.log_scale_var * np.outer(self.scale_sensitivity[i], self.scale_sensitivity[j])
        return block + self.conditional_cov[i] if i == j else block

    def box(self, object_id: str) -> OrientedBox:
        """The fused box of one object.

        Parameters
        ----------
        object_id : str
            The object.

        Returns
        -------
        OrientedBox
            World-frame box at the posterior mean.
        """
        return state_to_box(self.theta[self.index(object_id)])

    def joint_mean(self) -> Array:
        """Posterior mean of the stacked capture state.

        Returns
        -------
        numpy.ndarray, shape (7 M + 1,)
            ``(theta_1, ..., theta_M, sigma)``.
        """
        return np.append(self.theta.ravel(), self.log_scale)

    def joint_cov(self) -> Array:
        """Dense posterior covariance of the stacked capture state.

        Returns
        -------
        numpy.ndarray, shape (7 M + 1, 7 M + 1)
            Covariance of ``(theta_1, ..., theta_M, sigma)``, with every cross-covariance.
        """
        g = np.append(self.scale_sensitivity.ravel(), 1.0)
        cov = self.log_scale_var * np.outer(g, g)
        cov[:-1, :-1] += block_diag(*self.conditional_cov)
        return cov

    @property
    def n_measurements(self) -> NDArray[np.intp]:
        """numpy.ndarray of int, shape (M,): measurements of each object, accepted or not."""
        return np.bincount(self.measurement_object, minlength=len(self.object_ids))

    @property
    def n_rejected(self) -> NDArray[np.intp]:
        """numpy.ndarray of int, shape (M,): measurements of each object rejected by the gate."""
        return np.bincount(self.measurement_object[~self.accepted], minlength=len(self.object_ids))

    @property
    def p_gross(self) -> Array:
        """numpy.ndarray, shape (M,): method-of-moments gross-error rate per object (NaN without gating).

        ``clip((f - alpha) / (1 - alpha), 0, 1)`` with ``f`` the rejected fraction and
        ``alpha = 1 - gate_prob`` the gate's false-alarm rate.
        """
        if self.gate_prob is None:
            return np.full(len(self.object_ids), np.nan)
        alpha = 1.0 - self.gate_prob
        rejected_fraction = self.n_rejected / self.n_measurements
        return np.clip((rejected_fraction - alpha) / (1.0 - alpha), 0.0, 1.0)

    def object_nees(self, truth: ArrayLike) -> Array:
        """NEES of each object's state against ground truth, with its marginal covariance.

        Parameters
        ----------
        truth : array_like, shape (M, 7)
            True states, in ``object_ids`` order.

        Returns
        -------
        numpy.ndarray, shape (M,)
            Chi-square with 7 degrees of freedom each under a consistent estimator (correlated across
            objects through the shared scale).
        """
        error = state_error(self.theta, truth)
        marginal = self.conditional_cov + self.log_scale_var * (
            self.scale_sensitivity[:, :, None] * self.scale_sensitivity[:, None, :])
        return nees(error, marginal)

    def joint_nees(self, truth: ArrayLike, true_log_scale: float | None = None) -> tuple[float, int]:
        """NEES of the whole capture against ground truth, using every cross-covariance.

        Parameters
        ----------
        truth : array_like, shape (M, 7)
            True states, in ``object_ids`` order.
        true_log_scale : float or None, default None
            True log scale. Included in the error only if given and the scale was estimated.

        Returns
        -------
        nees : float
            ``e^T P^{-1} e`` over the stacked state.
        dof : int
            Its chi-square degrees of freedom: ``7 M``, plus 1 with the scale.
        """
        error = state_error(self.theta, truth).ravel()
        cov = self.joint_cov()
        if true_log_scale is not None and self.scale_estimated:
            error = np.append(error, self.log_scale - true_log_scale)
        else:
            cov = cov[:-1, :-1]
        return float(nees(error, cov)), int(error.size)


def _estimate(object_ids: tuple[str, ...], stack: _Stack, fit: _Fit, accepted: NDArray[np.bool_], nis: Array,
              gate_prob: float | None, scale_estimated: bool) -> CaptureEstimate:
    """Package a fit as a :class:`CaptureEstimate`.

    Parameters
    ----------
    object_ids : tuple of str
        Objects, in index order.
    stack : _Stack
        The measurements.
    fit : _Fit
        The final fit.
    accepted : numpy.ndarray of bool, shape (K,)
        Measurements kept.
    nis : numpy.ndarray, shape (K,)
        Leave-one-out NIS of every measurement.
    gate_prob : float or None
        Gate probability used.
    scale_estimated : bool
        Whether the log scale was a state.

    Returns
    -------
    CaptureEstimate
        The result.
    """
    return CaptureEstimate(object_ids=object_ids, theta=fit.theta, log_scale=fit.log_scale,
                           log_scale_var=fit.log_scale_var, conditional_cov=fit.conditional_cov,
                           scale_sensitivity=fit.scale_sensitivity, scale_estimated=scale_estimated,
                           measurement_object=stack.obj, accepted=accepted, nis=nis, gate_prob=gate_prob,
                           iterations=fit.iterations, converged=fit.converged)


def single_frame_estimate(measurement: BoxMeasurement) -> tuple[Array, Array]:
    """Read one measurement as a world-frame state estimate, assuming no scale error.

    This is the "single-frame" arm of the fusion ablation.

    Parameters
    ----------
    measurement : BoxMeasurement
        One frame's measurement of an object.

    Returns
    -------
    theta : numpy.ndarray, shape (7,)
        ``(R_cw^T (c_c - t_cw), size_x, size_y, height, yaw)``.
    cov : numpy.ndarray, shape (7, 7)
        The measurement covariance carried into the world frame.
    """
    R_cw, t_cw = measurement.T_cw[:3, :3], measurement.T_cw[:3, 3]
    theta = measurement.z.copy()
    theta[CENTER] = R_cw.T @ (measurement.z[CENTER] - t_cw)
    G = block_diag(R_cw.T, np.eye(4))
    return theta, G @ measurement.cov @ G.T


def naive_fuse(measurements: Sequence[BoxMeasurement], *, gate_prob: float | None = 0.997,
               max_iterations: int = 20, tol: float = 1e-10, max_gate_rounds: int = 100) -> CaptureEstimate:
    """Fuse each object's measurements independently, ignoring any shared scale error.

    Information-form fusion per object. Its covariance shrinks like 1/N with N frames, which is right only
    when measurement errors are independent across frames; a shared scale error breaks that. This is the
    "naive-fused" arm of the ablation.

    Parameters
    ----------
    measurements : sequence of BoxMeasurement
        Every measurement in the capture.
    gate_prob : float or None, default 0.997
        Probability of the chi-square(7) innovation gate; None disables gating.
    max_iterations : int, default 20
        Limit on yaw re-canonicalisation passes.
    tol : float, default 1e-10
        Convergence threshold on the estimate (metres, radians).
    max_gate_rounds : int, default 100
        Limit on gating rounds (rejections and re-admissions).

    Returns
    -------
    CaptureEstimate
        Independent per-object estimates (``scale_estimated`` False).

    Raises
    ------
    ValueError
        If there are no measurements, or ``max_iterations`` is below 1.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    object_ids, stack = _stack(measurements)
    theta0 = _robust_init(stack)

    def fit_fn(accepted: NDArray[np.bool_], previous: _Fit | None) -> _Fit:
        """Fuse the accepted measurements, warm-started from the previous fit.

        Parameters
        ----------
        accepted : numpy.ndarray of bool, shape (K,)
            Which measurements to use.
        previous : _Fit or None
            Previous fit, if any.

        Returns
        -------
        _Fit
            The fused estimate.
        """
        start = theta0 if previous is None else previous.theta
        return _information_fusion(stack, accepted, start, max_iterations, tol)

    fit, accepted, nis, _ = _fit_with_gating(stack, fit_fn, gate_prob, max_gate_rounds)
    return _estimate(object_ids, stack, fit, accepted, nis, gate_prob, scale_estimated=False)


def fuse_capture(measurements: Sequence[BoxMeasurement], *, scale_prior_sigma: float,
                 gate_prob: float | None = 0.997, max_iterations: int = 50, tol: float = 1e-10,
                 max_gate_rounds: int = 100) -> CaptureEstimate:
    """Jointly estimate every object of a capture and the capture's shared scale error.

    Gauss-Newton on ``sum_k r_k^T R_k^{-1} r_k + sigma^2 / sigma_s^2`` over all object states and the log
    scale ``sigma``, from per-object median initial states. This is the "scale-aware fused" arm of the
    ablation.

    Parameters
    ----------
    measurements : sequence of BoxMeasurement
        Every measurement in the capture.
    scale_prior_sigma : float
        Prior standard deviation ``sigma_s`` of the log scale. 0 holds the scale fixed at 1 (which reproduces
        :func:`naive_fuse`); ``numpy.inf`` is a flat prior, which needs camera translation to be solvable.
    gate_prob : float or None, default 0.997
        Probability of the chi-square(7) innovation gate; None disables gating.
    max_iterations : int, default 50
        Gauss-Newton iteration limit per fit.
    tol : float, default 1e-10
        Convergence threshold on the largest step component (metres, radians, log units).
    max_gate_rounds : int, default 100
        Limit on gating rounds (rejections and re-admissions).

    Returns
    -------
    CaptureEstimate
        MAP estimates with the joint posterior covariance (inverse Gauss-Newton Hessian).

    Raises
    ------
    ValueError
        If there are no measurements, ``scale_prior_sigma`` is negative, or ``max_iterations`` is below 1.
    numpy.linalg.LinAlgError
        If the scale is not observable under a flat prior.
    """
    if scale_prior_sigma < 0:
        raise ValueError("scale_prior_sigma must be non-negative")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    object_ids, stack = _stack(measurements)
    theta0 = _robust_init(stack)
    scale_info = None if scale_prior_sigma == 0 else 1.0 / scale_prior_sigma**2

    def fit_fn(accepted: NDArray[np.bool_], previous: _Fit | None) -> _Fit:
        """Fit the accepted measurements, warm-started from the previous fit.

        Parameters
        ----------
        accepted : numpy.ndarray of bool, shape (K,)
            Which measurements to use.
        previous : _Fit or None
            Previous fit, if any.

        Returns
        -------
        _Fit
            The MAP estimate and its posterior.
        """
        if previous is None:
            return _gauss_newton(stack, accepted, theta0, 0.0, scale_info, max_iterations, tol)
        return _gauss_newton(stack, accepted, previous.theta, previous.log_scale, scale_info, max_iterations,
                             tol)

    fit, accepted, nis, _ = _fit_with_gating(stack, fit_fn, gate_prob, max_gate_rounds)
    return _estimate(object_ids, stack, fit, accepted, nis, gate_prob, scale_estimated=scale_info is not None)


# --------------------------------------------------------------------------------------------------------
# Consistency statistics
# --------------------------------------------------------------------------------------------------------


def nees(error: ArrayLike, cov: ArrayLike) -> Array:
    """Normalised estimation error squared, ``e^T P^{-1} e``.

    Parameters
    ----------
    error : array_like, shape (..., n)
        Estimate minus truth.
    cov : array_like, shape (..., n, n)
        Estimated covariance.

    Returns
    -------
    numpy.ndarray, shape (...)
        Chi-square with ``n`` degrees of freedom under a consistent estimator.
    """
    error = np.asarray(error, dtype=np.float64)
    solved = np.linalg.solve(np.asarray(cov, dtype=np.float64), error[..., None])[..., 0]
    return np.einsum("...i,...i->...", error, solved)


def nis(innovation: ArrayLike, S: ArrayLike) -> Array:
    """Normalised innovation squared, ``nu^T S^{-1} nu``; needs no ground truth.

    Parameters
    ----------
    innovation : array_like, shape (..., m)
        Measurement minus prediction.
    S : array_like, shape (..., m, m)
        Predicted innovation covariance.

    Returns
    -------
    numpy.ndarray, shape (...)
        Chi-square with ``m`` degrees of freedom under a consistent filter.
    """
    return nees(innovation, S)


def chi2_bounds(dof: int, n_samples: int = 1, prob: float = 0.95) -> tuple[float, float]:
    """Two-sided acceptance interval for the average of independent chi-square samples.

    The sum of ``n_samples`` independent chi-square(dof) values is chi-square(``n_samples * dof``). For
    samples of differing dof, compare ``sum / total_dof`` with ``chi2_bounds(1, total_dof)``.

    Parameters
    ----------
    dof : int
        Degrees of freedom of each sample (the state or measurement dimension).
    n_samples : int, default 1
        Number of samples averaged.
    prob : float, default 0.95
        Probability mass inside the interval.

    Returns
    -------
    lower : float
        Lower bound on the average.
    upper : float
        Upper bound on the average.
    """
    total = dof * n_samples
    lower, upper = stats.chi2.ppf([(1.0 - prob) / 2.0, (1.0 + prob) / 2.0], total) / n_samples
    return float(lower), float(upper)


def coverage(errors: ArrayLike, sigmas: ArrayLike, k: float = 1.0) -> float:
    """Fraction of errors within ``k`` standard deviations.

    Parameters
    ----------
    errors : array_like
        Signed errors.
    sigmas : array_like
        Predicted standard deviations; broadcasts against ``errors``.
    k : float, default 1.0
        Multiple of sigma.

    Returns
    -------
    float
        Share with ``|e| <= k sigma``; 0.683, 0.954 and 0.997 for k = 1, 2, 3 if Gaussian and calibrated.
    """
    return float(np.mean(np.abs(np.asarray(errors)) <= k * np.asarray(sigmas)))


# --------------------------------------------------------------------------------------------------------
# Moving agents: constant-velocity Kalman filter on the ground plane
# --------------------------------------------------------------------------------------------------------


def cv_transition(dt: float, q: float) -> tuple[Array, Array]:
    """Transition and process noise of the constant-velocity model, state ``(x, y, vx, vy)``.

    Continuous white-noise acceleration with spectral density ``q`` on each axis.

    Parameters
    ----------
    dt : float
        Time step, seconds.
    q : float
        Acceleration noise spectral density, m^2 / s^3.

    Returns
    -------
    F : numpy.ndarray, shape (4, 4)
        State transition.
    Q : numpy.ndarray, shape (4, 4)
        Process noise covariance ``q [[dt^3/3 I, dt^2/2 I], [dt^2/2 I, dt I]]``.
    """
    eye = np.eye(2)
    F = np.block([[eye, dt * eye], [np.zeros((2, 2)), eye]])
    Q = q * np.block([[dt**3 / 3 * eye, dt**2 / 2 * eye], [dt**2 / 2 * eye, dt * eye]])
    return F, Q


@dataclass(frozen=True)
class TrackEstimate:
    """Filtered estimates of one agent's track.

    Parameters
    ----------
    times : numpy.ndarray, shape (T,)
        Step times, seconds.
    mean : numpy.ndarray, shape (T, 4)
        Posterior ``(x, y, vx, vy)`` after each step's measurements; metres and m/s.
    cov : numpy.ndarray, shape (T, 4, 4)
        Posterior covariances.
    nis : numpy.ndarray, shape (T,)
        NIS of each step's stacked measurements (NaN at steps without any).
    nis_dof : numpy.ndarray of int, shape (T,)
        Its degrees of freedom: twice the number of measurements at that step.
    """

    times: Array
    mean: Array
    cov: Array
    nis: Array
    nis_dof: NDArray[np.intp]


def cv_kalman_filter(times: ArrayLike, measurements: Sequence[tuple[ArrayLike, ArrayLike] | None], *,
                     q: float, x0: ArrayLike, P0: ArrayLike) -> TrackEstimate:
    """Track ground-plane position and velocity with a constant-velocity Kalman filter.

    At each step the prior is propagated to the step's time, then every measurement of that step (e.g. one
    per camera) is applied at once as a stacked measurement, with a Joseph-form covariance update.

    Parameters
    ----------
    times : array_like, shape (T,)
        Increasing step times, seconds.
    measurements : sequence of (array_like, array_like) or None, length T
        Per step, ``(z, R)`` with ``z`` of shape (m, 2), the measured ``(x, y)`` in metres, and ``R`` of shape
        (m, 2, 2), their covariances; ``m`` may be 0, or the entry None, for a step without measurements.
    q : float
        Acceleration noise spectral density, m^2 / s^3.
    x0 : array_like, shape (4,)
        Prior mean at ``times[0]``, before its measurements.
    P0 : array_like, shape (4, 4)
        Prior covariance at ``times[0]``.

    Returns
    -------
    TrackEstimate
        Filtered estimates, covariances and NIS per step.

    Raises
    ------
    ValueError
        If ``measurements`` and ``times`` differ in length.
    """
    times = np.asarray(times, dtype=np.float64)
    if len(measurements) != len(times):
        raise ValueError("need one measurement entry per time step")
    x, P = np.asarray(x0, dtype=np.float64).copy(), np.asarray(P0, dtype=np.float64).copy()
    T = len(times)
    means, covs = np.empty((T, 4)), np.empty((T, 4, 4))
    nis_values, nis_dof = np.full(T, np.nan), np.zeros(T, dtype=np.intp)
    for t, entry in enumerate(measurements):
        if t > 0:
            F, Q = cv_transition(times[t] - times[t - 1], q)
            x, P = F @ x, F @ P @ F.T + Q
        if entry is not None and len(entry[0]) > 0:
            z = np.asarray(entry[0], dtype=np.float64).reshape(-1, 2)
            R = block_diag(*np.asarray(entry[1], dtype=np.float64).reshape(-1, 2, 2))
            H = np.tile(np.eye(2, 4), (len(z), 1))
            innovation = z.ravel() - H @ x
            S = H @ P @ H.T + R
            gain = np.linalg.solve(S, H @ P).T
            x = x + gain @ innovation
            joseph = np.eye(4) - gain @ H
            P = joseph @ P @ joseph.T + gain @ R @ gain.T
            nis_values[t], nis_dof[t] = float(nis(innovation, S)), innovation.size
        means[t], covs[t] = x, P
    return TrackEstimate(times, means, covs, nis_values, nis_dof)


def track_nees(track: TrackEstimate, truth: ArrayLike, position_only: bool = False) -> Array:
    """NEES of a filtered track against a ground-truth track.

    Parameters
    ----------
    track : TrackEstimate
        The filtered track.
    truth : array_like, shape (T, 4) or (T, 2)
        True ``(x, y, vx, vy)``, or true ``(x, y)`` when ``position_only``.
    position_only : bool, default False
        Score position only (2 dof) instead of the full state (4 dof).

    Returns
    -------
    numpy.ndarray, shape (T,)
        NEES per step.
    """
    truth = np.asarray(truth, dtype=np.float64)
    if position_only:
        return nees(track.mean[:, :2] - truth[:, :2], track.cov[:, :2, :2])
    return nees(track.mean - truth, track.cov)


# --------------------------------------------------------------------------------------------------------
# Propagation to answers
# --------------------------------------------------------------------------------------------------------


def difference_in_frame(mean_a: ArrayLike, cov_a: ArrayLike, mean_b: ArrayLike, cov_b: ArrayLike,
                        rotation: ArrayLike | None = None,
                        cross_cov: ArrayLike | None = None) -> tuple[Array, Array]:
    """Mean and covariance of ``rotation @ (a - b)`` for jointly Gaussian points.

    With ``rotation = R_cw`` this gives the offset between two objects in a camera's axes, e.g. its x
    component for "is A left of B" (translation cancels in a difference).

    Parameters
    ----------
    mean_a : array_like, shape (n,)
        Mean of ``a``.
    cov_a : array_like, shape (n, n)
        Covariance of ``a``.
    mean_b : array_like, shape (n,)
        Mean of ``b``.
    cov_b : array_like, shape (n, n)
        Covariance of ``b``.
    rotation : array_like, shape (n, n), optional
        Linear map applied to the difference; identity if omitted.
    cross_cov : array_like, shape (n, n), optional
        ``Cov(a, b) = E[(a - mean_a)(b - mean_b)^T]``; zero if omitted.

    Returns
    -------
    mean : numpy.ndarray, shape (n,)
        Mean of the mapped difference.
    cov : numpy.ndarray, shape (n, n)
        Its covariance, ``M (C_a + C_b - C_ab - C_ab^T) M^T``.
    """
    mean_a, mean_b = np.asarray(mean_a, dtype=np.float64), np.asarray(mean_b, dtype=np.float64)
    cross = (np.zeros((mean_a.size, mean_b.size)) if cross_cov is None
             else np.asarray(cross_cov, dtype=np.float64))
    cov = np.asarray(cov_a, dtype=np.float64) + np.asarray(cov_b, dtype=np.float64) - cross - cross.T
    M = np.eye(mean_a.size) if rotation is None else np.asarray(rotation, dtype=np.float64)
    return M @ (mean_a - mean_b), M @ cov @ M.T


def distance_with_uncertainty(mean_a: ArrayLike, cov_a: ArrayLike, mean_b: ArrayLike, cov_b: ArrayLike,
                              cross_cov: ArrayLike | None = None) -> tuple[float, float]:
    """Distance between two uncertain points and its first-order standard deviation.

    ``sigma_d^2 = u^T Var(a - b) u`` with ``u`` the unit vector from ``b`` to ``a``. The linearisation is
    poor when the distance is within a few sigma of zero; use :func:`distance_monte_carlo` there.

    Parameters
    ----------
    mean_a : array_like, shape (n,)
        Mean of ``a``, metres.
    cov_a : array_like, shape (n, n)
        Covariance of ``a``.
    mean_b : array_like, shape (n,)
        Mean of ``b``, metres.
    cov_b : array_like, shape (n, n)
        Covariance of ``b``.
    cross_cov : array_like, shape (n, n), optional
        ``Cov(a, b)``; zero if omitted.

    Returns
    -------
    distance : float
        ``|mean_a - mean_b|``, metres.
    sigma : float
        First-order standard deviation, metres; NaN at zero distance, where the gradient is undefined.
    """
    delta, cov = difference_in_frame(mean_a, cov_a, mean_b, cov_b, cross_cov=cross_cov)
    distance = float(np.linalg.norm(delta))
    if distance == 0.0:
        return 0.0, float("nan")
    u = delta / distance
    return distance, float(np.sqrt(u @ cov @ u))


def distance_monte_carlo(mean_a: ArrayLike, cov_a: ArrayLike, mean_b: ArrayLike, cov_b: ArrayLike,
                         cross_cov: ArrayLike | None = None, n_samples: int = 100_000,
                         rng: np.random.Generator | int | None = None) -> tuple[float, float]:
    """Mean and standard deviation of the distance between two uncertain points, by sampling.

    Parameters
    ----------
    mean_a : array_like, shape (n,)
        Mean of ``a``, metres.
    cov_a : array_like, shape (n, n)
        Covariance of ``a``.
    mean_b : array_like, shape (n,)
        Mean of ``b``, metres.
    cov_b : array_like, shape (n, n)
        Covariance of ``b``.
    cross_cov : array_like, shape (n, n), optional
        ``Cov(a, b)``; zero if omitted.
    n_samples : int, default 100000
        Number of joint samples.
    rng : numpy.random.Generator or int or None, optional
        Random generator or seed.

    Returns
    -------
    mean : float
        Sample mean of the distance, metres.
    sigma : float
        Sample standard deviation of the distance, metres.
    """
    delta, cov = difference_in_frame(mean_a, cov_a, mean_b, cov_b, cross_cov=cross_cov)
    samples = np.random.default_rng(rng).multivariate_normal(delta, cov, size=n_samples)
    distances = np.linalg.norm(samples, axis=1)
    return float(distances.mean()), float(distances.std(ddof=1))


def prob_positive(mean: ArrayLike, var: ArrayLike) -> Array:
    """Probability that a Gaussian quantity is positive, ``Phi(mean / sigma)``.

    Use it for categorical relations, e.g. P(A is left of B) from the posterior of their x offset in a
    camera frame (see :func:`difference_in_frame`).

    Parameters
    ----------
    mean : array_like
        Means.
    var : array_like
        Variances (non-negative); broadcasts against ``mean``.

    Returns
    -------
    numpy.ndarray
        Probabilities; a zero variance gives 1 for a positive mean, 0 for a negative one, 0.5 at zero.
    """
    mean, var = np.broadcast_arrays(np.asarray(mean, dtype=np.float64), np.asarray(var, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(var > 0, mean / np.sqrt(var), np.sign(mean) * np.inf)
    return np.where(np.isnan(z), 0.5, stats.norm.cdf(z))
