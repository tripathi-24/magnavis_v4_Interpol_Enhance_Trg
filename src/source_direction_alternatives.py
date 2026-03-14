"""
Alternative methods for estimating direction to the anomaly source (not just direction of ΔB).

- Empirical calibration: correct reported azimuth/inclination using known-source data.
- Two-station dipole inversion: fit dipole position from ΔB at OBS1 and OBS2; direction to source = from observatory to fitted position.

See docs/ALTERNATIVE_SOURCE_DIRECTION_METHODS.md for full description.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


# -------- Empirical calibration --------


def fit_azimuth_calibration(
    reported_azimuths_deg: List[float],
    true_azimuths_deg: List[float],
    method: str = "offset",
) -> Tuple[float, float]:
    """
    Fit a simple correction from reported azimuth to true (source) azimuth.

    Parameters
    ----------
    reported_azimuths_deg : list of float
        Azimuths reported by the current pipeline (sensor frame or geographic).
    true_azimuths_deg : list of float
        Known true source azimuths (same frame as desired output).
    method : str
        "offset" -> true = (reported + offset) % 360 (one parameter).
        "linear" -> true = (a * reported + b) % 360 (two parameters).

    Returns
    -------
    params : tuple
        For "offset": (offset_deg,).
        For "linear": (a, b).
    """
    if len(reported_azimuths_deg) != len(true_azimuths_deg) or len(reported_azimuths_deg) < 2:
        raise ValueError("Need at least two (reported, true) pairs.")

    r = np.array(reported_azimuths_deg, dtype=float)
    t = np.array(true_azimuths_deg, dtype=float)

    # Normalise differences to [-180, 180] for fitting
    def diff_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = (a - b) % 360.0
        d = np.where(d > 180, d - 360, d)
        return d

    if method == "offset":
        # true ≈ (reported + offset) mod 360  =>  offset ≈ (true - reported) mod 360
        diff = (t - r) % 360.0
        diff = np.where(diff > 180, diff - 360, diff)
        offset = float(np.median(diff))  # median robust to outliers
        return (offset,)
    elif method == "linear":
        # Minimise sum of squared circular differences: true - (a*reported + b)
        from scipy import optimize

        def cost(params: np.ndarray) -> float:
            a, b = params[0], params[1]
            pred = (a * r + b) % 360.0
            d = diff_angle(t, pred)
            return np.sum(d * d)

        res = optimize.minimize(cost, [1.0, 0.0], method="Nelder-Mead")
        a, b = res.x[0], res.x[1]
        return (a, b)
    else:
        raise ValueError("method must be 'offset' or 'linear'.")


def correct_azimuth(
    reported_azimuth_deg: float,
    params: Tuple[float, ...],
    method: str = "offset",
) -> float:
    """
    Apply fitted calibration to a single reported azimuth.

    Parameters
    ----------
    reported_azimuth_deg : float
        Azimuth from current pipeline.
    params : tuple
        From fit_azimuth_calibration (offset,) or (a, b).
    method : str
        "offset" or "linear".

    Returns
    -------
    corrected_deg : float
        In [0, 360).
    """
    r = reported_azimuth_deg
    if method == "offset":
        (offset,) = params
        return (r + offset) % 360.0
    elif method == "linear":
        a, b = params[0], params[1]
        return (a * r + b) % 360.0
    else:
        raise ValueError("method must be 'offset' or 'linear'.")


# -------- Two-station dipole inversion --------


def dipole_field_at_point(
    observer_pos: Tuple[float, float, float],
    source_pos: Tuple[float, float, float],
    moment: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    Magnetic field vector at observer from a point dipole at source_pos with moment m.
    B ∝ (3 (m·r̂) r̂ - m) / |r|³; r = observer - source.
    Returns (Bx, By, Bz) in same units as moment (normalised by 1/|r|³ and constant).
    """
    rx = observer_pos[0] - source_pos[0]
    ry = observer_pos[1] - source_pos[1]
    rz = observer_pos[2] - source_pos[2]
    r2 = rx * rx + ry * ry + rz * rz
    if r2 < 1e-12:
        return (0.0, 0.0, 0.0)
    r_norm = math.sqrt(r2)
    r3 = r_norm * r2
    rhat_x = rx / r_norm
    rhat_y = ry / r_norm
    rhat_z = rz / r_norm
    m_dot_r = moment[0] * rhat_x + moment[1] * rhat_y + moment[2] * rhat_z
    Bx = (3 * m_dot_r * rhat_x - moment[0]) / r3
    By = (3 * m_dot_r * rhat_y - moment[1]) / r3
    Bz = (3 * m_dot_r * rhat_z - moment[2]) / r3
    return (Bx, By, Bz)


def fit_dipole_two_stations(
    obs1_pos: Tuple[float, float, float],
    delta_b1: Tuple[float, float, float],
    obs2_pos: Tuple[float, float, float],
    delta_b2: Tuple[float, float, float],
    moment_fixed: Optional[Tuple[float, float, float]] = None,
    bounds_distance: Tuple[float, float] = (1.0, 500.0),
) -> Optional[Tuple[Tuple[float, float, float], float]]:
    """
    Fit a dipole position (and optionally moment) from ΔB at two observatories.
    Minimises |ΔB1 - B1_pred|² + |ΔB2 - B2_pred|².

    Parameters
    ----------
    obs1_pos, obs2_pos : (x, y, z) in metres
        Observer positions (e.g. OBS1=(0,0,0), OBS2=(100,0,0)).
    delta_b1, delta_b2 : (Bx, By, Bz) in nT
        Observed perturbation vectors at OBS1 and OBS2 (same coordinate frame).
    moment_fixed : optional (mx, my, mz)
        If given, only fit position; otherwise fit position + moment (6 params).
    bounds_distance : (min, max) distance of source from origin (metres)
        Used to constrain search.

    Returns
    -------
    (source_pos, residual_norm) or None
        source_pos = (x, y, z) in metres; residual_norm = sqrt(sum of squared residuals).
        None if optimisation fails or residual too high.
    """
    try:
        from scipy import optimize
    except ImportError:
        return None

    def pack_moment(m: Optional[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        if m is not None:
            return m
        # Default: moment along +x (arbitrary; fit will adjust if moment is fitted)
        return (1.0, 0.0, 0.0)

    m0 = pack_moment(moment_fixed)
    fit_moment = moment_fixed is None

    def cost(params: np.ndarray) -> float:
        sx, sy, sz = params[0], params[1], params[2]
        if fit_moment:
            mx, my, mz = params[3], params[4], params[5]
            mom = (float(mx), float(my), float(mz))
        else:
            mom = m0
        src = (float(sx), float(sy), float(sz))
        b1 = dipole_field_at_point(obs1_pos, src, mom)
        b2 = dipole_field_at_point(obs2_pos, src, mom)
        # Scale: dipole formula gives B up to a constant (μ0/4π). Match magnitude by scaling.
        mag1_obs = math.sqrt(delta_b1[0] ** 2 + delta_b1[1] ** 2 + delta_b1[2] ** 2)
        mag1_pred = math.sqrt(b1[0] ** 2 + b1[1] ** 2 + b1[2] ** 2)
        if mag1_pred < 1e-20:
            return 1e20
        scale = mag1_obs / mag1_pred
        e1x = delta_b1[0] - scale * b1[0]
        e1y = delta_b1[1] - scale * b1[1]
        e1z = delta_b1[2] - scale * b1[2]
        e2x = delta_b2[0] - scale * b2[0]
        e2y = delta_b2[1] - scale * b2[1]
        e2z = delta_b2[2] - scale * b2[2]
        return e1x * e1x + e1y * e1y + e1z * e1z + e2x * e2x + e2y * e2y + e2z * e2z

    # Initial guess: midpoint between observatories, shifted in +y
    x0 = (obs1_pos[0] + obs2_pos[0]) / 2.0
    y0 = (obs1_pos[1] + obs2_pos[1]) / 2.0 + 20.0
    z0 = (obs1_pos[2] + obs2_pos[2]) / 2.0
    if fit_moment:
        p0 = [x0, y0, z0, m0[0], m0[1], m0[2]]
        bounds = [
            (obs1_pos[0] - bounds_distance[1], obs2_pos[0] + bounds_distance[1]),
            (obs1_pos[1] - bounds_distance[1], obs2_pos[1] + bounds_distance[1]),
            (obs1_pos[2] - bounds_distance[1], obs2_pos[2] + bounds_distance[1]),
            (None, None),
            (None, None),
            (None, None),
        ]
    else:
        p0 = [x0, y0, z0]
        bounds = [
            (obs1_pos[0] - bounds_distance[1], obs2_pos[0] + bounds_distance[1]),
            (obs1_pos[1] - bounds_distance[1], obs2_pos[1] + bounds_distance[1]),
            (obs1_pos[2] - bounds_distance[1], obs2_pos[2] + bounds_distance[1]),
        ]
    try:
        res = optimize.minimize(cost, p0, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            return None
        sx, sy, sz = res.x[0], res.x[1], res.x[2]
        dist = math.sqrt(sx * sx + sy * sy + sz * sz)
        if dist < bounds_distance[0] or dist > bounds_distance[1]:
            return None
        residual = math.sqrt(res.fun)
        return ((float(sx), float(sy), float(sz)), residual)
    except Exception:
        return None


def direction_from_observatory_to_source(
    obs_position: Tuple[float, float, float],
    source_position: Tuple[float, float, float],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Azimuth and inclination (degrees) of the direction from observatory to source.
    Convention: same as sensor (0° = +x, 90° = +y in horizontal; inclination from horizontal).
    """
    dx = source_position[0] - obs_position[0]
    dy = source_position[1] - obs_position[1]
    dz = source_position[2] - obs_position[2]
    h2 = dx * dx + dy * dy
    if h2 < 1e-20 and abs(dz) < 1e-20:
        return (None, None)
    h = math.sqrt(h2)
    r = math.sqrt(h2 + dz * dz)
    if r < 1e-20:
        return (None, None)
    az_rad = math.atan2(dy, dx)
    azimuth_deg = math.degrees(az_rad)
    if azimuth_deg < 0:
        azimuth_deg += 360.0
    inclination_deg = math.degrees(math.asin(dz / r))
    return (azimuth_deg, inclination_deg)
