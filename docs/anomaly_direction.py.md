# Documentation: `src/anomaly_direction.py`

- **File**: `src/anomaly_direction.py`
- **Purpose**: Compute the direction (azimuth and inclination) of magnetic anomalies from Observatory 1 and Observatory 2's three-axis magnetometer data, and triangulate approximate source location using both observatories.

---

## Overview

This module provides functions to determine the **direction** of a magnetic anomaly using the perturbation vector from three orthogonal sensors in each observatory. **Sensor layout differs between observatories:**

- **Observatory 1 (OBS1)**: Sensor 1 and 3 horizontal, Sensor 2 vertical.
  - **Sensor 1 (OBS1_1)**: Horizontal axis 1
  - **Sensor 2 (OBS1_2)**: Vertical (pointing upward)
  - **Sensor 3 (OBS1_3)**: Horizontal axis 2
  - Azimuth from (ΔS1, ΔS3). Inclination from ΔS2 vs horizontal.
- **Observatory 2 (OBS2)**: Sensor 1 and 2 horizontal, Sensor 3 vertical.
  - **Sensor 1 (OBS2_1)**: Horizontal axis 1
  - **Sensor 2 (OBS2_2)**: Horizontal axis 2
  - **Sensor 3 (OBS2_3)**: Vertical (pointing upward)
  - Azimuth from (ΔS1, ΔS2). Inclination from ΔS3 vs horizontal.

The direction is computed from the **perturbation vector** (ΔS1, ΔS2, ΔS3), which is the difference between the sensor readings at anomaly time and their baseline values.

Additionally, the module provides **triangulation** functionality to estimate approximate source location by intersecting direction vectors from both observatories.

---

## Functions

### `compute_direction_obs1(s1_nT, s2_nT, s3_nT, baseline_s1_nT, baseline_s2_nT, baseline_s3_nT)`

Computes the azimuth and inclination of a magnetic anomaly from OBS1 component readings.

**Parameters:**
- `s1_nT`, `s2_nT`, `s3_nT` (float): Component readings at anomaly time (Sensor 1 horizontal, Sensor 2 vertical, Sensor 3 horizontal), in nanoTesla (nT).
- `baseline_s1_nT`, `baseline_s2_nT`, `baseline_s3_nT` (float): Baseline (e.g., median) of each component over a quiet period, in nT.

**Returns:**
- `azimuth_deg` (float or None): Azimuth in degrees [0, 360). 
  - 0° = direction of Sensor 1
  - 90° = direction of Sensor 3
  - Returns `None` if horizontal perturbation is too small (purely vertical anomaly)
- `inclination_deg` (float or None): Inclination in degrees.
  - Positive = upward component
  - Negative = downward component
  - Returns `None` if perturbation magnitude is too small
- `magnitude_nT` (float): Magnitude of perturbation: √(ΔS1² + ΔS2² + ΔS3²)

**Mathematical formulas:**
- **Perturbation**: ΔS1 = S1 - baseline_S1, ΔS2 = S2 - baseline_S2, ΔS3 = S3 - baseline_S3
- **Azimuth**: θ = atan2(ΔS3, ΔS1) converted to [0, 360) degrees
- **Inclination**: φ = atan2(ΔS2, H) where H = √(ΔS1² + ΔS3²)
- **Magnitude**: |ΔB| = √(ΔS1² + ΔS2² + ΔS3²)

**Example:**
```python
azimuth, inclination, magnitude = compute_direction_obs1(
    s1_nT=25520.0,
    s2_nT=43713.0,
    s3_nT=28764.0,
    baseline_s1_nT=25500.0,
    baseline_s2_nT=43700.0,
    baseline_s3_nT=28750.0
)
# Returns: (azimuth ≈ 45°, inclination ≈ 10°, magnitude ≈ 25 nT)
```

---

### `compute_direction_obs2(s1_nT, s2_nT, s3_nT, baseline_s1_nT, baseline_s2_nT, baseline_s3_nT)`

Computes azimuth and inclination for **OBS2**, which has a different sensor layout: **Sensor 1 and Sensor 2 are horizontal; Sensor 3 is vertical.** So:
- **Horizontal plane**: (ΔS1, ΔS2). **Vertical**: ΔS3.
- **Azimuth**: θ = atan2(ΔS2, ΔS1) → 0° = Sensor 1, 90° = Sensor 2.
- **Inclination**: φ = atan2(ΔS3, H) where H = √(ΔS1² + ΔS2²). Positive = upward (S3+).
- **Magnitude**: |ΔB| = √(ΔS1² + ΔS2² + ΔS3²).

Returns azimuth, inclination, and magnitude for OBS2.

---

### `is_obs1_sensor(sensor_id: str) -> bool`

Checks if a sensor ID belongs to Observatory 1.

**Parameters:**
- `sensor_id` (str): Sensor identifier (e.g., "S20260202_124213_OBS1_1", "OBS1_2")

**Returns:**
- `bool`: `True` if sensor_id contains "OBS1_1", "OBS1_2", or "OBS1_3"; `False` otherwise.

**Example:**
```python
is_obs1_sensor("S20260202_124213_OBS1_1")  # True
is_obs1_sensor("OBS2_1")  # False
```

---

### `is_obs2_sensor(sensor_id: str) -> bool`

Checks if a sensor ID belongs to Observatory 2.

**Parameters:**
- `sensor_id` (str): Sensor identifier (e.g., "S20260202_124213_OBS2_1", "OBS2_2")

**Returns:**
- `bool`: `True` if sensor_id contains "OBS2_1", "OBS2_2", or "OBS2_3"; `False` otherwise.

---

### `is_obs_sensor(sensor_id: str) -> bool`

Checks if a sensor ID belongs to either Observatory 1 or Observatory 2.

---

### `azimuth_inclination_to_unit_vector(azimuth_deg, inclination_deg) -> Optional[Tuple[float, float, float]]`

Converts azimuth and inclination angles to a 3D unit vector.

**Parameters:**
- `azimuth_deg` (float or None): Azimuth in degrees [0, 360). 0° = Sensor 1 (X), 90° = Sensor 3 (Y). None if purely vertical.
- `inclination_deg` (float or None): Inclination in degrees. Positive = upward, negative = downward.

**Returns:**
- `unit_vector` (tuple of (x, y, z) or None): Unit vector pointing in the direction of the anomaly. Returns None if both azimuth and inclination are None.

**Coordinate system (same encoding for both observatories; physical axes differ):**
- X = azimuth 0° (Sensor 1), Y = azimuth 90° (OBS1: Sensor 3; OBS2: Sensor 2), Z = vertical up (OBS1: Sensor 2; OBS2: Sensor 3).

---

### `triangulate_source_location(obs1_position, obs1_azimuth_deg, obs1_inclination_deg, obs2_position, obs2_azimuth_deg, obs2_inclination_deg, max_distance_m=1000.0) -> Optional[Tuple[float, float, float, float]]`

Triangulates approximate source location from two observatory directions.

**Parameters:**
- `obs1_position` (tuple of (x, y, z)): Position of Observatory 1 in meters. Typically (0, 0, 0).
- `obs1_azimuth_deg`, `obs1_inclination_deg` (float or None): Azimuth and inclination from OBS1.
- `obs2_position` (tuple of (x, y, z)): Position of Observatory 2 in meters. Typically (100, 0, 0) if 100m apart along x-axis.
- `obs2_azimuth_deg`, `obs2_inclination_deg` (float or None): Azimuth and inclination from OBS2.
- `max_distance_m` (float): Maximum distance from observatories to consider valid (default 1000m).

**Returns:**
- `source_location` (tuple of (x, y, z, distance_error) or None): Estimated source location (x, y, z) in meters, and `distance_error` (meters) representing the closest distance between the two direction lines. Returns None if:
  - Either direction vector cannot be computed
  - Lines are parallel (or nearly parallel)
  - Estimated location is beyond `max_distance_m`

**Algorithm:**
1. Converts azimuth/inclination from both observatories to 3D unit vectors
2. Finds the closest point between the two 3D lines (one from each observatory)
3. Returns the midpoint as the estimated source location
4. Calculates the separation distance between the lines as an error metric

**Example:**
```python
result = triangulate_source_location(
    obs1_position=(0.0, 0.0, 0.0),
    obs1_azimuth_deg=45.0,
    obs1_inclination_deg=10.0,
    obs2_position=(100.0, 0.0, 0.0),
    obs2_azimuth_deg=135.0,
    obs2_inclination_deg=5.0,
    max_distance_m=1000.0
)
# Returns: (x, y, z, error) or None
```

---

## Usage in Application

This module is used by `application_temp.py` to compute anomaly directions and triangulate source locations:

1. **Baseline computation**: When historic data is loaded for each observatory, the median of each component (S1, S2, S3) over the historic window is computed and stored as `_obs1_baseline` and `_obs2_baseline`.

2. **Direction at anomaly time**: When an anomaly is detected on any OBS1 or OBS2 sensor:
   - The component values (S1, S2, S3) are looked up at the anomaly timestamp
   - `compute_direction_obs1()` or `compute_direction_obs2()` is called with the anomaly-time components and baseline
   - Results are logged: `[OBS1/OBS2] Anomaly at <time> | direction: azimuth=...°, inclination=...°, |ΔB|=... nT`
   - The status line above the log is updated with the latest direction
   - The anomaly direction is stored in `_obs1_recent_anomalies` or `_obs2_recent_anomalies` for triangulation

3. **Triangulation**: When anomalies occur at both observatories within a 2-minute time window:
   - `_attempt_triangulation()` is called automatically
   - It matches the closest anomalies from both observatories
   - `triangulate_source_location()` computes the approximate source location
   - Results are logged: `[Triangulation] Source location estimated | position: (x, y, z) m | distance from OBS1: ... m | error: ... m`
   - The UI status line is updated with the triangulated location

**Note**: Direction finding and triangulation currently work only in **CSV mode** because component data (b_x, b_y, b_z) is only available when loading from CSV files. Real-time/DB mode only provides magnitude (mag_H_nT).

**3D visualization in `application_temp.py`**: The application draws anomaly directions in a **3D plot** (matplotlib `mplot3d`). The center of the plot represents the observatory; each direction is shown as a **unit vector** from the origin. Azimuth and inclination are combined into (x, y, z) using the same convention (0° azimuth = Sensor 1 / East, inclination = angle from horizontal). The helper **`_anomaly_direction_to_unit_vector(azimuth_deg, inclination_deg)`** in `application_temp.py` implements this conversion; the module’s **`azimuth_inclination_to_unit_vector`** (if present) uses the same coordinate system.

---

## Coordinate System

- **OBS1:** Sensor 1 and 3 = horizontal (azimuth from S1–S3); Sensor 2 = vertical (inclination from ΔS2).
- **OBS2:** Sensor 1 and 2 = horizontal (azimuth from S1–S2); Sensor 3 = vertical (inclination from ΔS3).

The azimuth is the bearing in each observatory’s horizontal plane; inclination is the elevation (up/down) relative to that plane.
