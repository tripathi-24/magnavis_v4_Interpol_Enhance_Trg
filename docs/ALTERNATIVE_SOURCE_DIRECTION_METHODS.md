# Alternative Methods for Accurate Anomaly (Source) Direction

With the **current observatory setup** (OBS1 and OBS2, each with 3-axis component data; ~100 m baseline), the following methods can improve the **direction to the source** compared to using “direction of ΔB” alone.

---

## Summary Table

| Method | Direction accuracy | Extra hardware | Implementation effort | Best when |
|--------|--------------------|----------------|------------------------|-----------|
| 1. Empirical calibration | Good (site-specific) | None | Low | You have known-source runs (e.g. 45°, 90°) |
| 2. Two-station dipole inversion | Better (model-based) | None | Medium | Source is dipole-like; both stations see anomaly |
| 3. Improved baseline + time alignment | Moderate | None | Low | Baseline/alignment are main errors |
| 4. Triangulation + magnitude ratio | Better | None | Low–Medium | Two stations; rough distance helps direction |
| 5. Gradient (two stations) | Better | None (use OBS1–OBS2) | Medium | Coarse gradient along baseline only |

---

## Method 1: Empirical Calibration (Recommended First Step)

**Idea:** Use your **known-direction runs** (45°, 90°, 135°, …) to build a correction from “reported direction” to “true source direction.”

**Why it works:** Your current pipeline reports **direction of ΔB** (sensor frame). For your site and source type, that may differ from “direction to source” by a **roughly consistent offset or rotation**. Calibration learns that mapping.

**Steps:**
1. For each 2-minute window, take the **median (or mean) of reported azimuth** over that window (e.g. 15:18–15:20 → many samples ~196°).
2. Build pairs: (reported_azimuth, true_azimuth) = (196°, 45°), (185°, 90°), …
3. Fit a correction:
   - **Option A (simple):** `true_azimuth = (reported_azimuth + offset) % 360` (one parameter).
   - **Option B:** `true_azimuth = (a * reported_azimuth + b) % 360` (two parameters).
   - **Option C:** Per-sector correction or lookup table (reported 180–200° → 45°, etc.).
4. In production, apply this correction before displaying or logging “source direction.”
5. Optionally do the same for **inclination** if you have known elevations.

**Pros:** No new physics; uses your existing experiments; quick to implement; can absorb sensor misalignment and part of “ΔB vs source” bias.  
**Cons:** Valid for your setup and similar sources; does not fix near-field or extended-source effects.

**Implementation:** Add a small calibration module: load (reported, true) pairs from config or CSV, fit offset/linear term, and a function `correct_azimuth(reported_deg) -> corrected_deg`.

---

## Method 2: Two-Station Dipole Inversion

**Idea:** Assume the source is a **magnetic dipole**. Fit its position **(x, y, z)** (and optionally moment) so that the **predicted** field at OBS1 and OBS2 matches the **observed** (ΔS1, ΔS2, ΔS3) at both. The **direction to the source** from either observatory is then the vector from the observatory to the fitted (x, y, z).

**Why it helps:** “Direction of ΔB” at one point is not the same as “direction to dipole” except in special cases. Fitting a dipole explicitly solves for the source location, so direction to source is exact (within model and noise).

**Model:**  
At position **r** relative to a dipole at **r_src** with moment **m**, the field is  
**B** ∝ (3 (m·r̂) r̂ − m) / |r|³,  
where **r** = observer − r_src. You have two observers (OBS1, OBS2) and observed ΔB vectors. Fit **r_src** (and optionally **m**) to minimise the mismatch between predicted and observed ΔB at both stations.

**Steps:**
1. Define observer positions (e.g. OBS1 = (0,0,0), OBS2 = (100,0,0) in metres).
2. At a given anomaly time, you have ΔB₁ = (ΔS1₁, ΔS2₁, ΔS3₁) at OBS1 and ΔB₂ at OBS2 (in the same coordinate frame).
3. Minimise a cost, e.g.  
   `cost = |ΔB₁ − B_pred₁(r_src, m)|² + |ΔB₂ − B_pred₂(r_src, m)|²`  
   over **r_src** (and **m** if not fixed). Use a nonlinear solver (e.g. `scipy.optimize.least_squares` or `minimize`).
4. **Direction from OBS2 to source** = (r_src − OBS2_position) / |r_src − OBS2_position|; convert to azimuth/inclination if needed.
5. Reject or flag fits with high residual or unrealistic distance (e.g. > 1 km).

**Pros:** Uses full vector at two points; gives **direction and distance**; physically consistent for dipole-like sources.  
**Cons:** Assumes a dipole; needs careful coordinate alignment (OBS1/OBS2 and sensor axes); optimisation can have local minima; sensitive to noise and baseline errors.

**Implementation:** See `src/source_direction_alternatives.py` (stub provided) for a minimal two-station dipole fit you can call from the app when both OBS1 and OBS2 have an anomaly at the same time.

---

## Method 3: Improved Baseline and Time Alignment

**Idea:** Keep “direction of ΔB” but reduce errors so that ΔB is closer to “anomaly-only” and the three components refer to the **same instant**.

**Baseline:**
- Use a **quiet** period that does not include anomalies (e.g. first 30 min if no anomalies then).
- Or use a **running median** over a long window, excluding times already flagged as anomalies.
- Or fit a **trend** (linear or low-order polynomial) to each component over the quiet period and subtract it.

**Time alignment:**
- Ensure (S1, S2, S3) at “anomaly time” come from the **same timestamp** (or interpolate to a common time). Reduce `merge_asof` tolerance (e.g. from 2 s to 0.5 s or less) if your sampling allows.
- If possible, use one timestamp per sample (e.g. from one sensor) and interpolate the other two to that time.

**Pros:** No new hardware; often reduces systematic bias and scatter.  
**Cons:** Does not fix “ΔB direction ≠ source direction”; only makes the current estimator better.

---

## Method 4: Triangulation + Magnitude Ratio (Refine Distance)

**Idea:** You already **triangulate** from OBS1 and OBS2 “direction” rays. To improve:
- Use the **ratio** of |ΔB| at the two stations. For a dipole, |ΔB| ∝ 1/distance³. So |ΔB₁|/|ΔB₂| constrains the ratio of distances d₁/d₂. Together with triangulation (which gives a line or region), you can refine **where** along the line the source is, hence a better **direction** from each observatory to that refined point.
- Alternatively: fix the “direction” from each observatory to be the **direction from observatory to the triangulated point** (instead of direction of ΔB). That way you report “direction to the triangulated source,” which is consistent with the two rays.

**Pros:** Uses data you already have; no new hardware.  
**Cons:** Still assumes rays from “direction of ΔB”; magnitude ratio is sensitive to noise and non-dipole behaviour.

---

## Method 5: Coarse Gradient from Two Stations

**Idea:** With OBS1 and OBS2 at 100 m apart, you can approximate one **directional derivative** of the field along the baseline (e.g. (ΔB_OBS2 − ΔB_OBS1) / 100 m). That gives a very coarse “gradient” along one direction. Some source-localisation formulas use the field and its gradient to infer distance or direction. This is more involved and gives limited extra information with only two points; usually **dipole inversion (Method 2)** is a cleaner use of the same data.

---

## Recommended Order of Implementation

1. **Empirical calibration (Method 1)**  
   Use your 45°, 90°, … runs to fit an azimuth (and optionally inclination) correction. Apply it in the UI and logs so “source direction” matches your experiment convention. Low effort, immediate improvement for your setup.

2. **Baseline and time alignment (Method 3)**  
   Tighten baseline choice and component time alignment. Re-run calibration after this; the correction may change.

3. **Two-station dipole inversion (Method 2)**  
   When both OBS1 and OBS2 see the same anomaly, call a dipole fit; use the **direction from OBS2 (or OBS1) to the fitted position** as “source direction.” You can show this alongside (or instead of) “direction of ΔB” when the fit is good (low residual, reasonable distance).

4. **Triangulation refinement (Method 4)**  
   Optionally report “direction to triangulated point” and use magnitude ratio to refine distance; then direction to that refined point is your “source direction.”

---

## What You Get From Each Method

| Output | Method 1 | Method 2 | Method 3 | Method 4 |
|--------|----------|----------|----------|----------|
| Azimuth / inclination (source) | ✅ Corrected | ✅ From fitted position | ✅ Improved ΔB dir. | ✅ From triangulated point |
| Distance to source | ❌ | ✅ | ❌ | ⚠️ Rough |
| 3D position | ❌ | ✅ | ❌ | ✅ (triangulation) |
| Works with single station | ✅ | ⚠️ (worse) | ✅ | ❌ |
| Needs known-source data | ✅ | ❌ | ❌ | ❌ |

---

## References (in repo)

- `docs/DIRECTION_FINDING_ACCURACY.md` — limitations of “direction of ΔB” and when it matches source direction.
- `src/anomaly_direction.py` — current direction and triangulation.
- `src/source_direction_alternatives.py` — stub for dipole inversion and calibration (optional).
