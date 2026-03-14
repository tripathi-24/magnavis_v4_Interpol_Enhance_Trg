# Detailed working: `src/application_temp_fast.py`

- **File**: `src/application_temp_fast.py`
- **Purpose**: Fast DB-backed Magnavis with the **same core workflow** as application.py (historic → realtime → prediction → anomalies), but with a **minimal GUI** and **no import** of heavy modules from `src/application.py` (no VTK, GeoPandas, or base Application/ApplicationWindow). This reduces startup time and improves responsiveness.
- **Scope**: **Single sensor** only (one tab, e.g. OBS1_1). Simulation mode only (DB windowed fetch); no CSV or real-time “since” API in this script.

---

## Run

```bash
python src/application_temp_fast.py
```

---

## High-level architecture

- **Standalone**: Does not subclass `application.Application` or `application.ApplicationWindow`. Uses a minimal **QMainWindow** and a **Controller** that owns a single **SensorState** and one tab.
- **Data**: All time-series from **data_convert_db_now**:
  - **Initial**: `fetch_timeseries_window_multi([sensor_id], start_time, end_time, target_n_seconds=3400)`.
  - **Incremental**: `fetch_timeseries_between_multi([sensor_id], start_time, end_time, limit_rows=20000)`.
- **UI**: One tab (sensor name), one row of controls (source label, anomaly threshold multiplier, training window minutes), two Matplotlib figures (static + dynamic), bottom log. No .ui files; all widgets created in code.

---

## Startup sequence

1. **Environment**
   - Same as application_temp: Matplotlib/XDG caches under `src/.cache/mpl` and `src/.cache/xdg`.

2. **`main()`**
   - Creates **QApplication**, **MainWindow**, then **Controller(win)**. Controller constructor runs **`_init_sensor_and_ui()`** (see below), then **`win.showMaximized()`** and **`app.exec()`**.

3. **`Controller.__init__`**
   - `session_id = uuid4()`.
   - `state: Optional[SensorState] = None`.
   - **Simulation clock**: `sim_start = datetime(2026, 1, 5, 0, 0, 0)`, `sim_hist_end = sim_start + timedelta(minutes=60)`, `sim_step_s = 20`, `sim_rt_start` / `sim_rt_end` (first slice after historic).
   - References to fetch thread/worker (initially None).
   - Timers: **fetch_timer** (timeout → `fetch_incremental_tick`), **draw_timer** (→ `redraw`), **pred_poll_timer** (→ `poll_prediction_output`).
   - Calls **`_init_sensor_and_ui()`**.

4. **`_init_sensor_and_ui()`**
   - **Sensor choice**: **`_prompt_for_sensor_id()`** — either `QInputDialog.getItem` over `get_latest_sensor_ids(limit=6)`, or fallback `get_latest_sensor_id_like("%OBS1_1")`.
   - **SensorState(sensor_id, display_name)** created.
   - One tab added to **MainWindow.tabs** with the sensor name.
   - **Controls row**: “Source: IITK Observatory”, anomaly threshold multiplier (QDoubleSpinBox 0.1–10, default 2.5), training window minutes (0 = all).
   - **Plots**: Two FigureCanvases (static + dynamic), NavigationToolbars, axes with ylabel “ΔB (nT)”. Stored in `state.static_canvas`, `state.dynamic_canvas`, `state.static_ax`, `state.dynamic_ax`.
   - **Kick off**: **`fetch_initial()`** (starts first fetch thread).
   - **Timers**: fetch_timer 20 s, draw_timer 400 ms, pred_poll_timer 3 s.

---

## Data structures (single sensor)

- **SensorState** (dataclass):
  - **Historic (blue)**: `base_x`, `base_y` (lists of datetime / float); `baseline_nT` (median of base for ΔB).
  - **Realtime (green)**: `rt_x`, `rt_y`; `new_x`, `new_y` (used for anomaly comparison).
  - **Predictor**: `predictor_input_file`, `prediction_process`, `predict_x`, `predict_y`, `last_pred_out_mtime`.
  - **Anomaly**: `anomaly_detector`, `anomaly_times`.
  - **UI**: `static_canvas`, `dynamic_canvas`, axes, `blue_line`, `green_line`, `purple_line`, `anomaly_vlines_dyn` / `anomaly_vlines_static`.
  - **Settings**: `train_window_minutes`.
  - **Throttles**: `last_saved_points`, `last_pred_start_ts`, `predict_cooldown_s` (5 s).

- **FetchWorker** (QObject): Emits **`updated(DataFrame, is_incremental)`** with columns `time_H`, `mag_H_nT`.

---

## Fetch flow

- **`fetch_initial()`**: Starts a **QThread** with **FetchWorker**. On thread start, worker runs **`fetch_initial(sensor_id, sim_start, sim_hist_end, 3400)`** (3400 seconds of historic). Emits **`updated(df, False)`**.
- **`fetch_incremental_tick()`**: Every 20 s, starts thread that runs **`fetch_incremental(sensor_id, sim_rt_start, sim_rt_end)`**. Emits **`updated(df, True)`**.

- **`_on_data_updated(df, is_incremental)`**:
  - If empty and incremental: advance simulation window via **`get_min_timestamp_at_or_after(sensor_id, sim_rt_end)`** so simulation can progress; return.
  - Parse `time_H` → `t`, `mag_H_nT` → `y`.
  - **If not incremental**: Set `base_x` / `base_y` to last 3400 points; set `baseline_nT`; clear `rt_*`, `new_*`; set `sim_rt_start` to last base time, `sim_rt_end = sim_rt_start + sim_step_s`; **`_write_predict_input(start_predictor=False)`**; **`_init_lines()`** (draw blue on both axes); return.
  - **If incremental**: Append only strictly increasing timestamps to `rt_x`/`rt_y` and `new_x`/`new_y`; call **`_write_predict_input(start_predictor=True)`**; advance **`sim_rt_start`** / **`sim_rt_end`** by one step.

---

## Predict input and predictor

- **`_write_predict_input(start_predictor)`**:
  - Full series: `x_all = base_x + rt_x`, `y_all = base_y + rt_y`. If length unchanged (`last_saved_points`), return.
  - Folder: **`src/sessions/<session_id>/<sensor_id>/`**; file **`predict_input.csv`** (columns `x`, `y`).
  - If `start_predictor`, call **`_maybe_start_predictor()`**.

- **`_maybe_start_predictor()`**: Skips if process already running or within **predict_cooldown_s**. Runs **`python predictor_ai.py predict_input.csv`** in the sensor session folder; sets **TRAIN_WINDOW_MINUTES** from `state.train_window_minutes`; redirects stdout/stderr to `predict_stdout.log` / `predict_stderr.log`; stores **Popen** in `state.prediction_process`. Uses **nice -n 10** on non-Windows.

---

## Prediction poll and anomaly

- **`poll_prediction_output()`**: Every 3 s. Checks **`predict_out.csv`** mtime; if newer than `last_pred_out_mtime`, reads CSV, merges into `state.predict_x` / `state.predict_y`, then calls **`_detect_anomalies()`**.

- **`_detect_anomalies()`**: Uses **`state.anomaly_detector.detect_anomalies(actual_times=new_x, actual_values=new_y, predicted_times=predict_x, predicted_values=predict_y)`**. Appends new anomaly timestamps to `state.anomaly_times` (capped at 1000, FIFO).

---

## Redraw

- **`redraw()`**: Every 400 ms. Updates blue line (base, ΔB), green line (realtime ΔB), purple line (predictions ΔB). Removes and redraws anomaly vertical lines on the dynamic axis. Calls **`draw_idle()`** on both canvases.

- **ΔB**: **`_delta(ys)`** returns `[y - baseline_nT for y in ys]` when `baseline_nT` is set; otherwise returns `ys`.

---

## Differences from application_temp.py

| Aspect | application_temp.py | application_temp_fast.py |
|--------|---------------------|---------------------------|
| Base | Subclasses application.Application / ApplicationWindow | Standalone MainWindow + Controller |
| Sensors | Multi-sensor (tabs from selection/CSV) | Single sensor (one tab) |
| Data modes | Real-time, Simulation, CSV | Simulation only (DB windowed) |
| Historic length | User prompt (1–10080 min) | Fixed 60 min (3400 points) |
| UI | MagTimeSeriesWidget + .ui, freeze window control | In-code controls only (threshold, training window) |
| Predictor concurrency | Queue + max 1 concurrent | One process per (single) sensor |
| Session path | base_app.APP_BASE/sessions/... | _APP_BASE_DIR/sessions/... |

---

## Session layout

- **`src/sessions/<session_id>/<sensor_id>/`**: `predict_input.csv`, `predict_out.csv`, `predict_stdout.log`, `predict_stderr.log`.

---

## Dependencies

- **data_convert_db_now**: get_latest_sensor_ids, get_latest_sensor_id_like, get_min_timestamp_at_or_after, fetch_timeseries_window_multi, fetch_timeseries_between_multi.
- **Anomaly_detector**: AnomalyDetector.
- **PyQt5**, **matplotlib**, **pandas**, **numpy**. No **application**, no **data_convert_now**, no VTK/GeoPandas.
