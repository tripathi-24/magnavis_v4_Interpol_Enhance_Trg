# Direction finding: how it works and what the angles mean

## Sensor layout: OBS1 vs OBS2

- **OBS1:** Sensor 1 and Sensor 3 are in the horizontal plane; Sensor 2 is vertical (up).  
  Azimuth from (ΔS1, ΔS3). Inclination from ΔS2 vs horizontal.

- **OBS2:** Sensor 1 and Sensor 2 are in the horizontal plane; Sensor 3 is vertical (up).  
  Azimuth from (ΔS1, ΔS2). Inclination from ΔS3 vs horizontal.

So the **same (S1, S2, S3) column names** are used in the component table for both observatories, but the **physical meaning** differs: in OBS2, S1 and S2 are horizontal and S3 is vertical.

---

## When I select only OBS2_1 to plot in CSV mode, how can the app still compute azimuth and inclination?

**Short answer:** The app loads the **entire CSV** when you open a file. It then builds **component series for the whole observatory** from all sensors present in the file. Your selection (e.g. only OBS2_1) only decides **which sensor’s magnitude is plotted** and **which stream is used for anomaly detection**. Direction is always computed from **all three sensors** of that observatory.

**Step by step:**

1. **CSV load**  
   When you choose a CSV file, the app reads the **full file** (all rows, all sensors). It does **not** filter by the sensors you later select for plotting.

2. **Component tables**  
   From this full dataframe it builds:
   - **OBS1:** `_obs1_components_df` = aligned (time, S1, S2, S3) using **OBS1_1** (b_x→S1), **OBS1_2** (b_y→S2), **OBS1_3** (b_z→S3).
   - **OBS2:** `_obs2_components_df` = aligned (time, S1, S2, S3) using **OBS2_1** (b_x→S1), **OBS2_2** (b_y→S2), **OBS2_3** (b_z→S3).

   So **OBS2 direction** always uses:
   - **S1** = b_x from **OBS2_1**
   - **S2** = b_y from **OBS2_2**
   - **S3** = b_z from **OBS2_3**  
   aligned to the same timestamps (within a short tolerance). These tables are built if (and only if) the CSV contains **all three** OBS2 sensors.

3. **What “select OBS2_1” does**  
   Selecting only OBS2_1 means:
   - Only the **OBS2_1** magnitude time series is plotted.
   - Only that stream is used for **anomaly detection** (actual vs predicted).
   - When an anomaly is **detected** on that OBS2_1 stream, the app takes the **anomaly time** and looks up (S1, S2, S3) at that time in `_obs2_components_df`. Those three numbers come from **OBS2_1, OBS2_2, and OBS2_3** at that moment. So azimuth and inclination are computed from **all three OBS2 sensors**, not from OBS2_1 alone.

**Summary:** Direction is computed from the **observatory’s 3-axis component data** (S1, S2, S3). That data is built from the full CSV. Selecting only OBS2_1 does not remove OBS2_2 and OBS2_3 from the file or from the component table; it only restricts which sensor is **plotted** and **used for anomaly detection**. So azimuth and inclination are still well-defined as long as the CSV contains all of OBS2_1, OBS2_2, and OBS2_3.

**If the CSV had only OBS2_1:**  
Then `_build_obs2_components_df` would fail (it requires OBS2_1, OBS2_2, and OBS2_3), `_obs2_components_df` would be `None`, and direction finding for OBS2 would be unavailable.

---

## What does “inclination” mean here?

**Inclination** is the **elevation angle** of the **magnetic perturbation vector (ΔB)** with respect to the **horizontal plane** defined by the sensors at **that** observatory.

- **OBS1:** Horizontal plane = (S1, S3); vertical = S2. So H = √(ΔS1² + ΔS3²) and inclination = atan2(ΔS2, H). Positive = upward (S2).
- **OBS2:** Horizontal plane = (S1, S2); vertical = S3. So H = √(ΔS1² + ΔS2²) and inclination = atan2(ΔS3, H). Positive = upward (S3).

In both cases:

- **Positive inclination** → the perturbation has an **upward** component: the anomaly direction points **above** the horizontal.
- **Negative inclination** → **downward** component: the anomaly direction points **below** the horizontal.
- **Inclination ≈ 0°** → the perturbation is almost purely in the horizontal plane.
- **Inclination ≈ +90°** → almost purely **upward** (along the vertical sensor); **−90°** → purely **downward**.

So “inclination” here is **not** the inclination of the main Earth field; it is the **elevation angle of the anomaly direction** relative to that observatory’s horizontal plane, with positive = “up” (vertical sensor direction) and negative = “down”.
