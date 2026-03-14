"""
Anomaly direction from 3-axis magnetometer (Observatory 1 and Observatory 2).

Uses the perturbation vector (ΔS1, ΔS2, ΔS3) from the three sensors.

Observatory 1 (OBS1): Sensor 1 and Sensor 3 horizontal, Sensor 2 vertical.
  - Sensor 1 (OBS1_1): horizontal axis 1
  - Sensor 2 (OBS1_2): vertical (pointing up)
  - Sensor 3 (OBS1_3): horizontal axis 2
  Azimuth from (ΔS1, ΔS3). Inclination from ΔS2 vs horizontal magnitude.

Observatory 2 (OBS2): Sensor 1 and Sensor 2 horizontal, Sensor 3 vertical.
  - Sensor 1 (OBS2_1): horizontal axis 1
  - Sensor 2 (OBS2_2): horizontal axis 2
  - Sensor 3 (OBS2_3): vertical (pointing up)
  Azimuth from (ΔS1, ΔS2). Inclination from ΔS3 vs horizontal magnitude.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


def compute_direction_obs1(
    s1_nT: float,
    s2_nT: float,
    s3_nT: float,
    baseline_s1_nT: float,
    baseline_s2_nT: float,
    baseline_s3_nT: float,
) -> Tuple[Optional[float], Optional[float], float]:
    """
    Compute azimuth and inclination of the magnetic anomaly from OBS1 components.

    Parameters
    ----------
    s1_nT, s2_nT, s3_nT : float
        Component readings at anomaly time (Sensor 1 horizontal, Sensor 2 vertical, Sensor 3 horizontal), in nT.
    baseline_s1_nT, baseline_s2_nT, baseline_s3_nT : float
        Baseline (e.g. median) of each component over a quiet period, in nT.

    Returns
    -------
    azimuth_deg : float or None
        Azimuth in degrees [0, 360). 0° = Sensor 1 direction, 90° = Sensor 3 direction.
        None if horizontal perturbation is too small.
    inclination_deg : float or None
        Inclination in degrees. Positive = upward, negative = downward.
        None if perturbation magnitude is too small.
    magnitude_nT : float
        Magnitude of perturbation: sqrt(ΔS1² + ΔS2² + ΔS3²).
    """
    ds1 = s1_nT - baseline_s1_nT
    ds2 = s2_nT - baseline_s2_nT
    ds3 = s3_nT - baseline_s3_nT

    magnitude_nT = math.sqrt(ds1 * ds1 + ds2 * ds2 + ds3 * ds3)

    # Avoid division by zero / noisy angles when perturbation is tiny
    eps = 1e-6
    if magnitude_nT < eps:
        return None, None, 0.0

    h = math.sqrt(ds1 * ds1 + ds3 * ds3)
    if h < eps:
        # Purely vertical perturbation
        azimuth_deg = None
        inclination_deg = 90.0 if ds2 > 0 else -90.0
        return azimuth_deg, inclination_deg, magnitude_nT

    # Azimuth: atan2(ΔS3, ΔS1) -> radians, then to [0, 360) degrees
    az_rad = math.atan2(ds3, ds1)
    azimuth_deg = math.degrees(az_rad)
    if azimuth_deg < 0:
        azimuth_deg += 360.0

    # Inclination: atan2(ΔS2, H)
    inc_rad = math.atan2(ds2, h)
    inclination_deg = math.degrees(inc_rad)

    return azimuth_deg, inclination_deg, magnitude_nT


def compute_direction_obs2(
    s1_nT: float,
    s2_nT: float,
    s3_nT: float,
    baseline_s1_nT: float,
    baseline_s2_nT: float,
    baseline_s3_nT: float,
) -> Tuple[Optional[float], Optional[float], float]:
    """
    Compute azimuth and inclination of the magnetic anomaly from OBS2 components.

    OBS2 layout: Sensor 1 and Sensor 2 are in the horizontal plane; Sensor 3 is vertical (up).
    So horizontal perturbation = (ΔS1, ΔS2), vertical = ΔS3.

    Parameters
    ----------
    s1_nT, s2_nT, s3_nT : float
        Component readings at anomaly time (S1, S2 horizontal; S3 vertical), in nT.
    baseline_s1_nT, baseline_s2_nT, baseline_s3_nT : float
        Baseline (e.g. median) of each component over a quiet period, in nT.

    Returns
    -------
    azimuth_deg : float or None
        Azimuth in degrees [0, 360). 0° = Sensor 1 direction, 90° = Sensor 2 direction.
        None if horizontal perturbation is too small.
    inclination_deg : float or None
        Inclination in degrees. Positive = upward (S3), negative = downward.
        None if perturbation magnitude is too small.
    magnitude_nT : float
        Magnitude of perturbation: sqrt(ΔS1² + ΔS2² + ΔS3²).
    """
    ds1 = s1_nT - baseline_s1_nT
    ds2 = s2_nT - baseline_s2_nT
    ds3 = s3_nT - baseline_s3_nT

    magnitude_nT = math.sqrt(ds1 * ds1 + ds2 * ds2 + ds3 * ds3)

    eps = 1e-6
    if magnitude_nT < eps:
        return None, None, 0.0

    # OBS2: horizontal = (S1, S2), vertical = S3
    h = math.sqrt(ds1 * ds1 + ds2 * ds2)
    if h < eps:
        azimuth_deg = None
        inclination_deg = 90.0 if ds3 > 0 else -90.0
        return azimuth_deg, inclination_deg, magnitude_nT

    # Azimuth: atan2(ΔS2, ΔS1) -> 0° = S1, 90° = S2
    az_rad = math.atan2(ds2, ds1)
    azimuth_deg = math.degrees(az_rad)
    if azimuth_deg < 0:
        azimuth_deg += 360.0

    # Inclination: atan2(ΔS3, H) with H = horizontal magnitude
    inc_rad = math.atan2(ds3, h)
    inclination_deg = math.degrees(inc_rad)

    return azimuth_deg, inclination_deg, magnitude_nT


def is_obs1_sensor(sensor_id: str) -> bool:
    """Return True if sensor_id belongs to Observatory 1 (OBS1_1, OBS1_2, or OBS1_3)."""
    return (
        "OBS1_1" in sensor_id
        or "OBS1_2" in sensor_id
        or "OBS1_3" in sensor_id
    )


def is_obs2_sensor(sensor_id: str) -> bool:
    """Return True if sensor_id belongs to Observatory 2 (OBS2_1, OBS2_2, or OBS2_3)."""
    return (
        "OBS2_1" in sensor_id
        or "OBS2_2" in sensor_id
        or "OBS2_3" in sensor_id
    )


def is_obs_sensor(sensor_id: str) -> bool:
    """Return True if sensor_id belongs to either Observatory 1 or Observatory 2."""
    return is_obs1_sensor(sensor_id) or is_obs2_sensor(sensor_id)


def azimuth_inclination_to_unit_vector(
    azimuth_deg: Optional[float], inclination_deg: Optional[float]
) -> Optional[Tuple[float, float, float]]:
    """
    Convert azimuth and inclination to a 3D unit vector.
    
    Output (x, y, z): X = azimuth 0°, Y = azimuth 90°, Z = vertical (inclination).
    For OBS1: 0°=S1, 90°=S3, Z=S2. For OBS2: 0°=S1, 90°=S2, Z=S3.
    
    Parameters
    ----------
    azimuth_deg : float or None
        Azimuth in degrees [0, 360). 0° = first horizontal axis (X), 90° = second (Y).
        None if purely vertical.
    inclination_deg : float or None
        Inclination in degrees. Positive = upward, negative = downward.
        None if perturbation is too small.
    
    Returns
    -------
    unit_vector : tuple of (x, y, z) or None
        Unit vector pointing in the direction of the anomaly.
        Returns None if azimuth and inclination are both None or invalid.
    """
    if inclination_deg is None:
        return None
    
    # Convert to radians
    inc_rad = math.radians(inclination_deg)
    
    if azimuth_deg is None:
        # Purely vertical perturbation
        return (0.0, 0.0, 1.0 if inc_rad > 0 else -1.0)
    
    az_rad = math.radians(azimuth_deg)
    
    # Horizontal component magnitude (cos of inclination)
    h_mag = math.cos(inc_rad)
    
    # Vertical component (sin of inclination)
    z = math.sin(inc_rad)
    
    # Horizontal components
    x = h_mag * math.cos(az_rad)  # Sensor 1 direction
    y = h_mag * math.sin(az_rad)  # Sensor 3 direction
    
    return (x, y, z)


def triangulate_source_location(
    obs1_position: Tuple[float, float, float],
    obs1_azimuth_deg: Optional[float],
    obs1_inclination_deg: Optional[float],
    obs2_position: Tuple[float, float, float],
    obs2_azimuth_deg: Optional[float],
    obs2_inclination_deg: Optional[float],
    max_distance_m: float = 1000.0,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Triangulate approximate source location from two observatory directions.
    
    Finds the closest point between two 3D lines (one from each observatory).
    Each line extends from the observatory position in the direction of the anomaly.
    
    Parameters
    ----------
    obs1_position : tuple of (x, y, z)
        Position of Observatory 1 in meters. Origin (0, 0, 0) recommended.
    obs1_azimuth_deg, obs1_inclination_deg : float or None
        Azimuth and inclination from OBS1.
    obs2_position : tuple of (x, y, z)
        Position of Observatory 2 in meters. Typically (100, 0, 0) if 100m apart along x-axis.
    obs2_azimuth_deg, obs2_inclination_deg : float or None
        Azimuth and inclination from OBS2.
    max_distance_m : float
        Maximum distance from observatories to consider valid (default 1000m).
        If triangulated point is further, returns None.
    
    Returns
    -------
    source_location : tuple of (x, y, z, distance_error) or None
        Estimated source location (x, y, z) in meters, and distance_error (meters)
        representing the closest distance between the two direction lines.
        Returns None if:
        - Either direction vector cannot be computed
        - Lines are parallel (or nearly parallel)
        - Estimated location is beyond max_distance_m
    """
    # Convert directions to unit vectors
    dir1 = azimuth_inclination_to_unit_vector(obs1_azimuth_deg, obs1_inclination_deg)
    dir2 = azimuth_inclination_to_unit_vector(obs2_azimuth_deg, obs2_inclination_deg)
    
    if dir1 is None or dir2 is None:
        return None
    
    # Unpack positions and directions
    p1 = obs1_position
    d1 = dir1
    p2 = obs2_position
    d2 = dir2
    
    # Vector from OBS1 to OBS2
    w = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    
    # Dot products
    a = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]  # |d1|² (should be 1.0)
    b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]  # d1 · d2
    c = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]  # |d2|² (should be 1.0)
    d = d1[0] * w[0] + d1[1] * w[1] + d1[2] * w[2]     # d1 · w
    e = d2[0] * w[0] + d2[1] * w[1] + d2[2] * w[2]     # d2 · w
    
    # Denominator: a*c - b²
    denom = a * c - b * b
    
    # Check if lines are parallel (denominator close to zero)
    eps = 1e-6
    if abs(denom) < eps:
        return None  # Lines are parallel, cannot triangulate
    
    # Parameters along each line
    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom
    
    # Closest points on each line
    closest1 = (p1[0] + t1 * d1[0], p1[1] + t1 * d1[1], p1[2] + t1 * d1[2])
    closest2 = (p2[0] + t2 * d2[0], p2[1] + t2 * d2[1], p2[2] + t2 * d2[2])
    
    # Midpoint (estimated source location)
    source_x = (closest1[0] + closest2[0]) / 2.0
    source_y = (closest1[1] + closest2[1]) / 2.0
    source_z = (closest1[2] + closest2[2]) / 2.0
    
    # Distance error (separation between the two lines)
    dx = closest2[0] - closest1[0]
    dy = closest2[1] - closest1[1]
    dz = closest2[2] - closest1[2]
    distance_error = math.sqrt(dx * dx + dy * dy + dz * dz)
    
    # Distance from origin (OBS1)
    dist_from_obs1 = math.sqrt(source_x * source_x + source_y * source_y + source_z * source_z)
    
    # Reject if too far
    if dist_from_obs1 > max_distance_m:
        return None
    
    return (source_x, source_y, source_z, distance_error)
