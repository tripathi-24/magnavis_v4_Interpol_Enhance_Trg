# Detailed working: `src/application_temp.py`

- **File**: `src/application_temp.py`
- **Purpose**: DB-backed multi-sensor Magnavis with the same workflow as `application.py` (historic → realtime → prediction → anomalies), but using the project MySQL DB as the time-series source. Minimal UI: **two-column layout** (left = sensor streams, right = parameters + 3D direction plot + log).
- **Features**:
  - **Up to 3 sensors** in parallel with independent predictor and anomaly detector per sensor.
  - **Shared parameters**: One set of controls (threshold multiplier, training window, freeze window) applies to all sensors.
  - **Resultant signal + low-pass**: Time series plot **B (nT)** = low-pass filtered resultant magnitude (no baseline subtraction).
  - **3D anomaly direction plot**: Unit vector(s) from observatory center showing azimuth and inclination together (OBS1/OBS2, CSV mode).
  - **Pre-trained GRU support**: Per-sensor pre-trained models via `PRETRAINED_MODEL_DIR`.

---

## Run

```bash
python src/application_temp.py
```

---

## High-level architecture

- **Base**: Subclasses `application.Application` and `application.ApplicationWindow`; USGS is replaced by a stub that delegates to `data_convert_db_now`.
- **Data source**: All time-series from `data_convert_db_now` (`fetch_timeseries_window_multi`, `fetch_timeseries_between_multi`, `get_timeseries_magnetic_data_multi`, `get_timeseries_magnetic_data_since_multi`).
- **UI**: **Two-column layout** (horizontal splitter):
  - **Left half**: Up to three **sensor stream panels**, each in its own **QScrollArea** (scrollable left, right, up, down). Each panel shows one time-series plot (resultant B in nT, low-pass filtered) with toolbar.
  - **Right half**: (1) **Shared parameters** panel (threshold, training window, freeze window, source/refresh, status); (2) **3D anomaly direction** plot (unit vectors from center); (3) **Log** (reduced height, scrollable).
- **Sensors**: Maximum 3 selected at startup; `sensor_ids` and `sensor_ctx` hold state per sensor. Each sensor has its own `AnomalyDetector` and predictor subprocess; up to **3 predictor processes** can run in parallel (`_predict_max_concurrent = 3`).

---

## Startup sequence

1. **Environment**
   - Matplotlib/XDG caches under `src/.cache/mpl` and `src/.cache/xdg`.
   - `data_convert_now` stub installed so imports resolve to `data_convert_db_now`.

2. **`ApplicationTemp.__init__`**
   - Pre-initializes `sensor_ids`, `sensor_ctx` (dict of `SensorContext`), timers, predictor queue, simulation clock, CSV state, **historic_minutes** / **historic_points_1hz** (default 60 / 3600), **`_lowpass_alpha`** (EMA filter, default 0.2).
   - Calls `super().__init__(arg)` and **`_configure_startup_mode()`**.
   - Sets `_startup_configured = True` and first **`startThreads(hours=1, start_time=None, new=False)`**.

3. **`_configure_startup_mode()`**
   - **Initial historic data**: `QInputDialog.getInt` (default 60, min 1, max 10080).
   - **Data source**: Real-time / Simulation / CSV file.
   - **Sensor selection**: `_configure_sensor_selection()` — multi-select or comma-separated; **max 3 sensors**; result in `_selected_sensor_ids`.

4. **First fetch**
   - `ApplicationWindowTemp.startThreads(..., new=False)` → `_app._discover_sensors()`, then **MultiFetchWorker** in a QThread (CSV / Simulation / Real-time). Emits **`updated(dfs, False)`**; main thread handles in **`on_db_data_updated`**.

---

## Data flow and state (per sensor)

- **SensorContext** (dataclass):
  - **Historic**: `base_x_t`, `base_y_mag_t` (resultant magnitude, low-pass filtered); **`last_filtered_value`** for EMA state.
  - **Realtime**: `rt_x_t`, `rt_y_mag_t`, `new_x_t`, `new_y_mag_t` (filtered).
  - **Prediction**: `predict_x_t`, `predict_y_t`; `predictor_input_file`; `prediction_process`; etc.
  - **Anomaly**: `anomaly_detector`, `anomaly_times`, `anomaly_values`, vertical lines.
  - **UI**: `dynamic_canvas`, `dynamic_ax`, `dynamic_line`, `dynamic_new_line`, `predictions_line`; **no static plot** in current layout.
  - **Settings**: `train_window_minutes` (shared in UI).

- **Display**: Plots show **B (nT)** — low-pass filtered resultant; no baseline subtraction. Anomaly and training logic use the same filtered values.

---

## Low-pass filter

- **`_lowpass_series(ctx, values, reset)`**: Exponential Moving Average (EMA) with `_lowpass_alpha`. Updates `ctx.last_filtered_value`. Used for historic snapshot and real-time chunks in **`on_db_data_updated`**.

---

## `on_db_data_updated(dfs, is_new)`

- **`_discover_sensors()`** so `sensor_ids` / `sensor_ctx` match selection.
- For each `(sid, df)` in `dfs`:
  - **If not is_new**: Build resultant magnitude; **low-pass filter**; set `base_x_t` / `base_y_mag_t` (last `_rolling_window_points`); clear `rt_*`, `new_*`; reset EMA state.
  - **If is_new**: Append strictly increasing points; **low-pass** the new chunk; append to `rt_*` and `new_*`.
  - **Save / predictor**: If total points changed, full series = base + rt, **`save_data_for_sensor(sid, x_all, y_all, start_predictor=...)`**; enqueue predictor; update `last_saved_points`.
  - **Anomaly**: If `rt_x_t` and `predict_x_t` exist, **`detect_anomalies_for_sensor(sid)`**.
- **`appWin.updateData()`**: If framework not loaded and any sensor has base data, **`load_plot_framework_2()`**; else **`update_all_canvases()`**.

---

## UI layout: `_simplify_ui_for_multisensor()` and `load_plot_framework_2()`

### Window layout (ApplicationWindowTemp)

- **Horizontal splitter** (50/50):
  - **Left**: `_temp_timeseries_container` — filled by **`load_plot_framework_2()`** with one **QScrollArea** per sensor (each with label + toolbar + dynamic canvas). Panels get stretch 1 so they share the left half.
  - **Right**: **`_right_layout`** (QVBoxLayout) with:
    1. **Parameters panel** (inserted at index 0 in `load_plot_framework_2`): header + **SensorMagTimeSeriesWidget** (shared controls; internal scrollArea hidden). Stretch 1.
    2. **Direction row**: "Last anomaly direction" label + **3D direction plot** (see below). No stretch.
    3. **Log**: `textEditLog`; max height 90, stretch 0, scrollable.

### 3D anomaly direction plot

- **Axes**: `add_subplot(111, projection="3d")` (matplotlib `mplot3d`). Origin = **observatory center**; axes E, N, Z; limits ±1.15.
- **Unit vector**: **`_anomaly_direction_to_unit_vector(azimuth_deg, inclination_deg)`** converts (az°, inc°) to (x, y, z) with magnitude 1:
  - 0° azimuth = Sensor 1 (East), 90° = Sensor 3 (North); inclination = angle from horizontal (positive = up).
  - Formula: `x = cos(inc)*cos(az)`, `y = cos(inc)*sin(az)`, `z = sin(inc)`; purely vertical → (0,0,±1).
- **Drawing**: For each entry in **`_anomaly_direction_history`**, draw **`ax.quiver(0,0,0, u,v,w)`** from center to (u,v,w). OBS1 = blue (C0), OBS2 = orange (C1); last 30 shown; older faded. Empty state: "—".
- **Update**: **`update_anomaly_direction_plot(azimuth_deg, inclination_deg, magnitude_nT, obs_label)`** appends to history and calls **`_draw_anomaly_direction_plot()`**.

### Parameters panel (right)

- Built in **`load_plot_framework_2()`**: **params_container** (QWidget) with header "Shared Parameters (apply to all sensors)" + **SensorMagTimeSeriesWidget** for first sensor. **`right_layout.insertWidget(0, params_container, 1)`** so it appears at top right. Threshold, training window, and freeze window changes are **synced to all sensors** and to all registered control widgets (**`_sensor_control_widgets`**).

---

## Plot framework: `load_plot_framework_2()`

- **Discovery**: **`_discover_sensors()`**.
- **Host**: **`_temp_timeseries_container`** (left half). Cleared then filled.
- **Right side**: **`_right_layout`** from window. **params_container** = header + **SensorMagTimeSeriesWidget** (first sensor); internal plot scrollArea hidden via **QTimer.singleShot(300, hide_plot_area)**; **insertWidget(0, params_container, 1)**.
- **Left side**: For each **sid** in **sensor_ids**, create **plot_panel** (label + toolbar + **FigureCanvas**); wrap in **QScrollArea** (horizontal/vertical scroll as needed); **outer.addWidget(sensor_scroll, 1)**.
- **Timers**: Data 20 s; draw 400 ms → **`update_all_canvases()`**; prediction poll 3 s; predictor scheduler 500 ms → **`_drain_predict_queue()`** (up to **`_predict_max_concurrent`** = 3). Grace timer for **`_start_predictors_if_idle`**.

---

## Fetch thread (incremental, `new=True`)

- CSV / Simulation / Real-time paths as before. Worker emits **`updated(dfs, True)`**; **`on_db_data_updated`** handles.

---

## Saving data and predictor

- **`save_data_for_sensor(sensor_id, x_t, y_t, start_predictor)`**: Session folder `sessions/<session_id>/<sensor_id>/`; anomaly timestamps dropped from written series; writes **`predict_input.csv`** (columns `x`, `y`); if `start_predictor`, **`_enqueue_prediction(sensor_id)`**.
- **Queue**: **`_predict_queue`** (deque), **`_predict_active`** (set). **`_drain_predict_queue()`** starts up to **3** predictor subprocesses.
- **`start_prediction_process_for_sensor(sensor_id)`**: Runs `python predictor_ai.py <predictor_input_file>` in sensor folder; **TRAIN_WINDOW_MINUTES**, **PRETRAINED_MODEL_PATH** (if set); stores Popen in **`ctx.prediction_process`**.
- **`update_predictions_for_sensor(sensor_id)`**: On process exit, reads **`predict_out.csv`**, merges into **`ctx.predict_x_t`** / **`ctx.predict_y_t`**, then **`detect_anomalies_for_sensor(sensor_id)`**.

---

## Anomaly detection

- **`detect_anomalies_for_sensor(sensor_id)`**: **`ctx.anomaly_detector.calculate_differences(actual_times=rt_x_t, actual_values=rt_y_mag_t, predicted_times=predict_x_t, predicted_values=predict_y_t)`**; new anomaly timestamps/values appended; threshold and **freeze_duration_minutes** from **ctx.anomaly_detector** (shared UI updates all contexts).
- **Logging**: Each new anomaly logged with timestamp, sensor, magnitude; OBS1/OBS2 (CSV) include direction (azimuth, inclination, |ΔB|).
- **Direction**: For OBS1/OBS2 with component data, **`update_obs1_direction(...)`** and **`update_anomaly_direction_plot(az, inc, mag_nT, obs_label)`**; **`_attempt_triangulation`** when both observatories have recent anomalies.
- **Visualization**: **`_redraw_anomalies(sensor_id)`** draws vertical lines on **dynamic_ax** (no static plot).

---

## Anomaly direction finding (OBS1/OBS2, CSV mode)

- **Component series**: **`_build_obs1_components_df()`** / **`_build_obs2_components_df()`** from CSV; **`_ensure_obs1_baseline()`** / **`_ensure_obs2_baseline()`** (median over historic window).
- **At anomaly time**: **`_get_obs1_components_at_time(t)`** / OBS2 equivalent; **`compute_direction_obs1()`** or **`compute_direction_obs2()`** from **`anomaly_direction`**; log and **`update_anomaly_direction_plot(az, inc, mag_nT, obs_label)`**.
- **3D plot**: Unit vector(s) from observatory center; azimuth and inclination combined in one 3D view. See **3D anomaly direction plot** above.

---

## Pre-trained model support

- Same as before: **PRETRAINED_MODEL_DIR**; **`gru_pretrained_<sensor_id>.keras`** per sensor; **`start_prediction_process_for_sensor`** sets **PRETRAINED_MODEL_PATH** for subprocess. **predictor_ai.py** loads model if set.

---

## Key configuration (user-facing)

- **Initial historic data (minutes)** — startup; applies to all modes.
- **Anomaly threshold multiplier** — shared; updates **all** **ctx.anomaly_detector.threshold_multiplier**.
- **Training window (minutes)** — shared; 0 = full history; **TRAIN_WINDOW_MINUTES** to predictor.
- **Anomaly freeze window (minutes)** — shared; **ctx.anomaly_detector.freeze_duration_minutes** for all.

---

## Session layout

- **`sessions/<session_id>/<sensor_id>/`**: **predict_input.csv**, **predict_out.csv**, **predict_stdout.log**, **predict_stderr.log** (per sensor).

---

## Dependencies

- **application** (base_app): Application, ApplicationWindow, MagTimeSeriesWidget, APP_BASE.
- **data_convert_db_now**: get_latest_sensor_ids, get_latest_sensor_id_like, get_min_timestamp_at_or_after, fetch_timeseries_window_multi, fetch_timeseries_between_multi, get_timeseries_magnetic_data_multi, get_timeseries_magnetic_data_since_multi.
- **Anomaly_detector**: AnomalyDetector.
- **anomaly_direction**: compute_direction_obs1, compute_direction_obs2, is_obs1_sensor, is_obs2_sensor, triangulate_source_location.
- **mpl_toolkits.mplot3d**: Axes3D (for 3D direction plot).
- **PyQt5**, **matplotlib**, **pandas**, **numpy**.
