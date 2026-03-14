# Accuracy and Limitations of Anomaly Direction Finding

## What the Method Computes Accurately

The current implementation **accurately computes the 3D direction of the magnetic field perturbation vector** (ΔB):

- **Azimuth**: Correctly represents the horizontal bearing of the perturbation vector
- **Inclination**: Correctly represents the elevation angle of the perturbation vector  
- **Magnitude**: Accurately computes |ΔB| = √(ΔS1² + ΔS2² + ΔS3²)

The mathematical calculation is **geometrically correct** for determining the direction of a 3D vector from its components.

---

## What It Represents Physically

The perturbation vector **ΔB** represents the **change** in the magnetic field at the observatory location. This is the superposition of:
- Earth's background field (removed by baseline subtraction)
- Anomaly field from the source

**For a magnetic dipole source** (typical for ferrous objects):
- The perturbation vector **roughly points toward or away from the source**
- However, the exact relationship depends on:
  - Source orientation (magnetic moment direction)
  - Source distance
  - Source strength

---

## Limitations and Considerations

### 1. **Single Observatory → Direction Only, Not Location**

With **one observatory**, you get:
- ✅ **Direction** of the anomaly (azimuth + inclination)
- ❌ **NOT distance** to the source
- ❌ **NOT exact 3D position** of the source

**Example**: If azimuth = 45°, inclination = 10°, the source could be:
- 10 meters away at (45°, 10°)
- 100 meters away at (45°, 10°)
- 1 km away at (45°, 10°)

All would produce similar **direction** but different **magnitudes**.

### 2. **Source Type Matters**

**Dipole approximation** (small ferrous object):
- Perturbation vector points roughly toward/away from source
- Accuracy improves with distance (far-field approximation)

**Distributed/extended sources**:
- More complex field patterns
- Direction may not point directly at source center

**Moving sources**:
- Direction changes with time as source moves
- Can track trajectory if sampled frequently

### 3. **Coordinate System Reference**

The azimuth is **relative to your sensor orientation** (layout differs per observatory):
- **OBS1:** 0° = Sensor 1, 90° = Sensor 3 (both horizontal); vertical = Sensor 2.
- **OBS2:** 0° = Sensor 1, 90° = Sensor 2 (both horizontal); vertical = Sensor 3.

**To convert to geographic coordinates** (North/East), you need:
- Calibration of sensor orientation (which way is Sensor 1 pointing?)
- Compass heading or GPS alignment

### 4. **Baseline Accuracy**

The method assumes:
- Baseline represents "quiet" Earth field (no anomalies)
- Baseline is stable over the measurement period
- Any deviation is due to the anomaly

**Potential issues**:
- If baseline includes previous anomalies → direction may be biased
- If Earth field drifts → baseline becomes outdated
- If multiple sources present → superposition effects

### 5. **Sensor Arrangement**

Current setup:
- Sensors are ~1 meter apart
- Arranged in orthogonal axes

**Effects**:
- For **nearby sources** (< 10 m): Sensors may measure slightly different fields
- For **distant sources** (> 100 m): All sensors see essentially the same field → direction is accurate
- The 1 m separation is negligible for distant sources

---

## Possible Sources of Error (Why Direction May Not Match Experiments)

Even when the **math is correct**, the reported direction can disagree with experiments. Below are the main reasons.

### 1. **Perturbation Direction ≠ Source Direction (Physical / Model Limitation)**

The method treats the **direction of the perturbation vector (ΔB)** as the direction toward (or away from) the source. That is only approximately true under certain conditions.

- **Far-field dipole**: For a small magnetic dipole at large distance, the perturbation vector roughly points toward or away from the source. Accuracy improves with distance.
- **Extended or multiple sources**: Field lines are distorted; the direction of ΔB may not point at any single source.
- **Near-field**: Close to the source, the field pattern is more complex; direction of ΔB can differ significantly from the line to the source.
- **Induced vs remanent magnetisation**: Source type and orientation change the field pattern; the same “direction to source” can give different ΔB directions.

So the **formula correctly gives the direction of ΔB**; that direction **may not equal the direction to the source** in your experiment.

### 2. **Baseline Errors**

The baseline is the **median** of each component (S1, S2, S3) over the **historic load window** (the initial “quiet” period when data were loaded).

- **Contaminated baseline**: If that window contained earlier anomalies, drift, or different quiet conditions, the baseline is wrong. Then (ΔS1, ΔS2, ΔS3) is wrong and the computed direction is biased.
- **Fixed baseline**: The baseline is set once. If the main field drifts (e.g. diurnal variation, magnetic storms) before the anomaly, the effective “quiet” level is no longer the stored baseline → systematic error in direction.
- **Wrong time window**: If the historic window is too short, too long, or not representative of true quiet conditions, the baseline is unreliable.

### 3. **Sensor Alignment and Coordinate Frame**

Azimuth is defined in the **sensor frame**: 0° = S1 axis, 90° = S2 axis (for OBS2). The compass label (e.g. “South”) assumes a known mapping from sensor axes to geographic directions.

- **Misalignment**: If S1 and S2 are not aligned with your geographic or experimental reference (e.g. not exactly West/South or North/East), the **reported direction will be rotated** relative to the true bearing. A constant offset is common.
- **Tilt and non-orthogonality**: If the three axes are not perfectly orthogonal or not level (e.g. S3 not vertical), azimuth and inclination are biased. Calibration (rotation matrix or alignment angles) is needed for best accuracy.

### 4. **Time Alignment of the Three Components**

The three components (S1, S2, S3) come from three different sensors (or channels). In the current implementation:

- Components are merged with **nearest-time** alignment (e.g. `merge_asof` with a 2 s tolerance). So at a given “anomaly time”, S1 might be from time *t*, S2 from *t* ± δ₂, S3 from *t* ± δ₃, with each δ up to the tolerance.
- When looking up components at anomaly time, the code uses the **nearest** row in the component table. If the component data are sampled coarsely (e.g. 1 Hz), the nearest time can be up to ~0.5 s away from the anomaly time.

**Effect**: For **fast transients**, the three components may not all refer to the same instant. The vector (ΔS1, ΔS2, ΔS3) then mixes different phases of the anomaly → wrong direction. Tighter time alignment (smaller tolerance, higher sampling, or interpolation) can reduce this error.

### 5. **Noise and Numerical Sensitivity**

- **Small horizontal magnitude (H)**: Azimuth = atan2(ΔS2, ΔS1). When H = √(ΔS1² + ΔS2²) is small, small errors in ΔS1 or ΔS2 cause **large** changes in azimuth (angles are unstable near the origin).
- **Small vertical or horizontal component**: Similarly, inclination can be sensitive when ΔS3 or H is small.
- **Low signal-to-noise**: If the perturbation is small compared to sensor noise or residual baseline error, the computed direction can be unreliable even though the formula is correct.

### 6. **Summary of Error Sources**

| Source | Effect on direction |
|--------|---------------------|
| Perturbation ≠ source direction (physics) | Reported direction is direction of ΔB; may not point at source |
| Wrong or drifting baseline | Biased (ΔS1, ΔS2, ΔS3) → biased azimuth and inclination |
| Sensor misalignment / tilt | Constant or slowly varying offset in azimuth and inclination |
| Poor time alignment of S1, S2, S3 | Mixed phases → wrong vector, especially for fast transients |
| Noise / small H or small components | Unstable azimuth or inclination (large scatter) |

**Practical takeaway**: The implementation is **correct for the direction of the perturbation vector**. Disagreement with experiments is expected when baseline, alignment, time alignment, or source geometry are not ideal. Improving accuracy requires addressing these (e.g. better baseline choice, sensor calibration, tighter time alignment, and treating the result as a bearing rather than exact source direction).

---

## Accuracy Assessment

### ✅ **Accurate For**:
1. **Direction of perturbation vector**: Mathematically correct
2. **Relative changes**: Good for tracking how direction changes over time
3. **Distant sources** (> 50-100 m): Far-field approximation holds well
4. **Dipole sources**: Reasonable approximation for ferrous objects

### ⚠️ **Approximate For**:
1. **Nearby sources** (< 10 m): Field gradients matter, single-point measurement less accurate
2. **Complex sources**: Extended or multiple sources
3. **Source location**: Only gives direction, not distance or exact position

### ❌ **Cannot Determine**:
1. **Distance** to source (need magnitude + source model, or multiple observatories)
2. **Exact 3D position** (need triangulation from multiple observatories)
3. **Source type** (dipole vs. distributed) without additional analysis

---

## Alternative Methods for More Accurate Direction / Source Location

The current method gives the **direction of the perturbation vector** at one point. The following approaches can improve accuracy of **direction** or **source location**; they differ in data needs, complexity, and what they deliver.

### 1. **Triangulation from Two Observatories** (already in use)

- **Idea**: Use bearing lines from OBS1 and OBS2; their intersection (or closest approach in 3D) gives an estimated source position. Direction to the source from either observatory is then implied by that position.
- **Pros**: No extra hardware; uses your existing two observatories; gives a 3D point, not just a bearing; reduces ambiguity along the bearing.
- **Cons**: Accuracy depends on baseline length and geometry (wider separation and good crossing angle improve accuracy); both observatories must see the same anomaly within a time window; bearing errors from each site still propagate into the intersection.
- **When it helps**: Especially when the two bearing lines cross at a large angle; avoids relying on “direction of ΔB” from a single site as the only estimate.

### 2. **Dipole Fitting (Inverse Problem)**

- **Idea**: Assume the source is a magnetic dipole. Fit the dipole’s position (x, y, z) and moment (strength and orientation) so that the predicted field at your sensor(s) best matches the measured (ΔS1, ΔS2, ΔS3). The fitted position gives both **direction and distance** to the source.
- **Pros**: Uses the full vector (and optionally magnitude) at one or more points; often more accurate than “direction of ΔB” when the source is dipole-like; gives distance and 3D location.
- **Cons**: Needs nonlinear optimisation (e.g. Levenberg–Marquardt); can converge to local minima; assumes a dipole (fails for extended or complex sources); sensitive to noise and baseline errors.
- **Data**: One observatory is enough in principle; two or more improve robustness and accuracy.
- **Implementation**: Solve for (position, moment) by minimising the mismatch between predicted and observed field (and optionally |ΔB|) at each sensor.

### 3. **Gradient Tensor (Single-Point or Array)**

- **Idea**: Use the **spatial gradient** of the magnetic field (how B changes in x, y, z). The gradient tensor has mathematical properties that can be used to estimate source position and sometimes direction with less ambiguity than the field vector alone.
- **Pros**: Theoretically can give source position from a **single point** (if the full tensor is known); used in UXO detection and mineral exploration; can improve direction/location when applicable.
- **Cons**: Requires **gradient measurements**: either a gradiometer (multiple magnetometers at known offsets) or an array of 3-axis sensors. Your current setup gives B at two separated observatories (so you have a very coarse gradient), not a full tensor at one point.
- **Feasibility**: Full single-point tensor methods need new hardware (gradiometer or dense array). With OBS1 and OBS2 you could approximate one or two gradient components to refine estimates, but this is more involved.

### 4. **Multi-Point Dipole Inversion**

- **Idea**: Same dipole model as in (2), but use **both** observatories: fit one dipole (position + moment) so that the predicted field at OBS1 and OBS2 best matches the measured components (and optionally magnitudes) at both.
- **Pros**: Uses all available vector (and magnitude) information; typically more accurate and stable than single-point direction or single-point dipole fit; gives direction and distance.
- **Cons**: Same as dipole fitting (optimisation, dipole assumption, sensitivity to noise and baseline).
- **Feasibility**: Well suited to your two-observatory setup; no new hardware.

### 5. **Calibrated Bearing (Empirical Correction)**

- **Idea**: Keep the current “direction of ΔB” method but **calibrate** it using known sources at known positions: e.g. place a magnet at several known bearings and elevations, record reported azimuth/inclination, then fit a correction (offset, scale, or simple rotation) so that reported direction matches true direction.
- **Pros**: Simple to implement; no change to core maths; can correct for fixed sensor misalignment and part of the “ΔB vs source direction” bias for your site and typical sources.
- **Cons**: Does not fix fundamental limits (e.g. extended sources, near field); correction is only as good as the calibration conditions.

### 6. **Magnitude Decay with Distance (Two Observatories)**

- **Idea**: With two observatories, you have two directions and two |ΔB| values. For a dipole, |ΔB| decays with distance in a known way. Use the **ratio** of the two magnitudes (and the known baseline) to constrain distance along the bearing, then refine direction/location.
- **Pros**: Uses data you already have; can improve distance and hence effective “direction to source” when combined with triangulation.
- **Cons**: Depends on dipole-like decay; sensitive to noise in |ΔB| and to different source–sensor distances.

---

## Summary: Which Method When?

| Goal | Method | Accuracy (typical) | Data / effort |
|------|--------|--------------------|----------------|
| Bearing only, minimal change | Current (direction of ΔB) | Moderate; can disagree with true source direction | Already implemented |
| Bearing + rough 3D location | Triangulation (OBS1 + OBS2) | Better than single bearing; limited by crossing angle and baseline | Already implemented |
| Direction + distance, dipole-like source | Dipole fitting (one or two observatories) | Often better than “direction of ΔB” alone | Add optimisation; two observatories recommended |
| Best use of two observatories | Multi-point dipole inversion | Good when dipole assumption holds | Add optimisation |
| Fix systematic offset | Calibrated bearing (known sources) | Improves agreement with experiments for your setup | Add calibration procedure |
| Highest theoretical accuracy (single point) | Gradient tensor | High where applicable | New hardware (gradiometer / array) |

**Practical recommendation**: For **more accurate direction** with your current hardware, the most effective next steps are (1) **use triangulation** when both observatories see the anomaly, and (2) **add dipole fitting** (or multi-point dipole inversion) so you get direction and distance from the full vector (and magnitude) at one or two points. Calibrated bearing is a low-cost way to correct systematic errors.

---

## Improving Accuracy (Implementation Options)

### Option 1: **Use Both Observatories (100 m apart)** — already implemented

With **two observatories**, you can:
- **Triangulate**: Intersect directions from OBS1 and OBS2 → approximate source location (see `triangulate_source_location` in `anomaly_direction.py`).
- **Consistency check**: Verify if both point to the same region.

### Option 2: **Dipole Fitting** (recommended for better direction + distance)

- Assume source is a magnetic dipole; fit position (x, y, z) and moment to measured (ΔS1, ΔS2, ΔS3) at one or both observatories.
- Delivers direction and distance; usually more accurate than “direction of ΔB” when the source is dipole-like.
- Requires iterative optimisation (e.g. Levenberg–Marquardt).

### Option 3: **Gradient Tensor Analysis**

- Requires gradient measurements (gradiometer or dense sensor array). Not available with current hardware; would need new instrumentation.

### Option 4: **Calibration with Known Sources**

- Place a known source at known positions; record reported azimuth/inclination; fit an empirical correction (offset/rotation) to improve agreement with experiments.

---

## Current Method: Summary

**What you get**:
- ✅ Accurate **direction** of magnetic field perturbation (azimuth + inclination)
- ✅ Useful for **tracking** how anomaly direction changes
- ✅ Good **first approximation** for source location (especially distant sources)

**What you don't get**:
- ❌ Exact source **distance**
- ❌ Exact source **3D position** (without triangulation)
- ❌ Source **type** or **strength** (without modeling)

**Best use case**:
- **Direction tracking**: Monitor how anomaly direction changes over time
- **Initial localization**: Get rough bearing toward source
- **Combined with OBS2**: Triangulate for approximate location

---

## Recommendations

1. **For better accuracy**: Use **both observatories** and triangulate
2. **For exact location**: Implement **dipole fitting** or **gradient analysis**
3. **For tracking**: Current method is excellent for monitoring direction changes
4. **For validation**: Compare with known source positions to calibrate accuracy

The current method is **scientifically sound** for what it computes (perturbation vector direction), but users should understand it gives **direction, not exact location**.
