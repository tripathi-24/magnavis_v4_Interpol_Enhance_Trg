# Changelog: Recent Updates (2026)

This document summarizes major updates and new features added to the Magnavis project.

---

## New Features

### 1. Anomaly Direction Finding (OBS1)

**Module**: `src/anomaly_direction.py` (NEW)

**Purpose**: Compute the direction (azimuth and inclination) of magnetic anomalies using Observatory 1's three-axis magnetometer data.

**Key Functions**:
- `compute_direction_obs1()`: Computes azimuth and inclination from component perturbations
- `is_obs1_sensor()`: Checks if a sensor belongs to Observatory 1

**Integration**:
- Integrated into `application_temp.py` for CSV mode
- Automatically computes direction when anomalies are detected on OBS1 sensors
- Logs direction details: `[OBS1] Anomaly at <time> | direction: azimuth=...°, inclination=...°, |ΔB|=... nT`
- Status line above log shows latest OBS1 anomaly direction

**Limitations**:
- Works only in CSV mode (component data not available in DB/real-time)
- Requires all three OBS1 sensors (OBS1_1, OBS1_2, OBS1_3) in CSV

**Documentation**: `docs/anomaly_direction.py.md`

---

### 2. UI Redesign and 3D Anomaly Direction Plot (`application_temp.py`)

**Layout**:
- **Two-column UI**: Horizontal splitter. **Left**: up to 3 sensor stream panels, each in a **QScrollArea** (scrollable left, right, up, down). **Right**: (1) shared parameters panel (threshold, training window, freeze window) inserted at top; (2) 3D anomaly direction plot; (3) log (reduced height, scrollable).
- **Shared parameters**: One control panel applies to all sensors; threshold, training window, and freeze window are synced across sensors and UI widgets.
- **Up to 3 sensors** in parallel; up to 3 predictor subprocesses (`_predict_max_concurrent = 3`).

**Plot and signal**:
- Time series show **B (nT)** — low-pass filtered **resultant** magnetic field (EMA filter, `_lowpass_alpha`); no baseline subtraction.
- **3D anomaly direction plot**: Replaced 2D polar with **matplotlib 3D** (`projection="3d"`). **Center = observatory**. Each recorded direction is drawn as a **unit vector** from origin using **`_anomaly_direction_to_unit_vector(azimuth_deg, inclination_deg)`** (x = cos(inc)*cos(az), y = cos(inc)*sin(az), z = sin(inc)). OBS1/OBS2 colors; last 30 directions shown; user can rotate the 3D view.

**Documentation**: `docs/application_temp.md` (full rewrite to match current behavior).

---

### 3. Pre-trained GRU Model Support

**Module**: `src/predictor_ai.py` (UPDATED), `src/train_gru_pretrained.py` (NEW)

**Purpose**: Enable pre-training of GRU models on extended historical data (2-4 months) for better accuracy and faster startup.

**Key Changes**:

1. **`GRUPredictor` class** (renamed from `LSTMPredictor`):
   - Added `save_model(filepath)`: Saves model weights and scaler state
   - Added `load_model(filepath)`: Loads pre-trained model and scaler
   - Updated `forecast()`: Supports `pretrained_model_path` parameter

2. **Training script** (`train_gru_pretrained.py`):
   - Trains one model per sensor automatically
   - Supports multiple CSV files and folders
   - Handles raw CSV format (sensor_id, timestamp, b_x, b_y, b_z)
   - Saves models as `gru_pretrained_<sensor_id>.keras`

3. **Application integration** (`application_temp.py`):
   - Automatically loads pre-trained models per sensor
   - Uses `PRETRAINED_MODEL_DIR` environment variable
   - Falls back to training from scratch if model not found

**Usage**:
```bash
# Train models
python src/train_gru_pretrained.py --folder "Large Files" models/ --epochs 50

# Use in app
export PRETRAINED_MODEL_DIR="models"
python src/application_temp.py
```

**Documentation**: 
- `docs/predictor_ai.py.md` (updated)
- `docs/train_gru_pretrained.py.md` (new)

---

### 4. Enhanced Anomaly Logging

**Module**: `src/application_temp.py` (UPDATED)

**Changes**:
- Each anomaly is now logged individually with timestamp and magnitude
- OBS1 anomalies include direction details (azimuth, inclination, |ΔB|)
- Summary line shows total anomalies and threshold

**Example log output**:
```
[OBS1_1] Anomaly detected | time=2026-02-03 08:15:23 | magnitude=45231.5 nT | direction: azimuth=45.2°, inclination=12.3°, |ΔB|=15.7 nT
[OBS2_2] Anomaly detected | time=2026-02-03 08:15:45 | magnitude=38921.2 nT
[OBS1_1] matched_pairs=1200 total_anomalies=3 threshold=12.50 nT
```

---

## Bug Fixes

### CSV Mode Database Connection Error

**Issue**: In CSV mode, the app was attempting to call MySQL functions when advancing the simulation clock, causing connection errors.

**Fix**: Added check for `csv_enabled` flag to skip DB calls in CSV mode. When no new points are found, the simulation clock advances by a fixed step instead of querying the database.

**Location**: `src/application_temp.py`, `on_db_data_updated()` method

---

## Updated Documentation

1. **`docs/anomaly_direction.py.md`** (NEW): Complete documentation for anomaly direction finding module
2. **`docs/train_gru_pretrained.py.md`** (NEW): Documentation for pre-training script
3. **`docs/predictor_ai.py.md`** (UPDATED): Reflects class rename to `GRUPredictor` and new save/load methods
4. **`docs/application_temp.md`** (UPDATED): Rewritten to match current two-column UI, 3 sensors, shared parameters, resultant+low-pass plot, 3D anomaly direction plot, and pre-trained model support
5. **`docs/INDEX.md`** (UPDATED): Added entries for new modules; application_temp.md description updated
6. **`docs/summary.md`** (UPDATED): DB/Temp variant section updated for two-column UI and 3D direction plot
7. **`docs/anomaly_direction.py.md`** (UPDATED): Note on 3D visualization in application_temp.py

---

## Model Architecture Clarification

**Class Name**: Changed from `LSTMPredictor` to `GRUPredictor` to accurately reflect that the model uses GRU (Gated Recurrent Unit) layers, not LSTM.

**Architecture**:
- GRU layer: 32 units
- Dense layer: 16 units (ReLU)
- Output layer: 1 unit
- Features: Magnetic field (scaled) + cyclic time features (daily sin/cos, optionally yearly sin/cos)

---

## File Changes Summary

### New Files
- `src/anomaly_direction.py`
- `src/train_gru_pretrained.py`
- `docs/anomaly_direction.py.md`
- `docs/train_gru_pretrained.py.md`
- `docs/CHANGELOG_2026.md` (this file)

### Modified Files
- `src/predictor_ai.py`: Added save/load methods, renamed class, added pre-trained support
- `src/application_temp.py`: Two-column UI, up to 3 sensors with scrollable panels, shared parameters, resultant B (nT) with low-pass filter, 3D anomaly direction plot, direction finding, pre-trained model loading, enhanced logging, CSV mode fix
- `src/application.py`: Updated commented references to use `GRUPredictor`
- `docs/predictor_ai.py.md`: Updated class name and new methods
- `docs/application_temp.md`: Added new feature sections
- `docs/INDEX.md`: Added new documentation entries

---

## Migration Guide

### For Users

1. **Using pre-trained models**:
   - Train models: `python src/train_gru_pretrained.py --folder "Large Files" models/`
   - Set environment: `export PRETRAINED_MODEL_DIR="models"`
   - Run app normally: `python src/application_temp.py`

2. **Anomaly direction finding**:
   - Works automatically in CSV mode when OBS1 sensors are present
   - Check log for direction details: `[OBS1] Anomaly at ... | direction: ...`
   - Status line above log shows latest direction

### For Developers

1. **Class name change**: Update any direct references from `LSTMPredictor` to `GRUPredictor`
2. **New imports**: `from anomaly_direction import compute_direction_obs1, is_obs1_sensor`
3. **Environment variables**: `PRETRAINED_MODEL_DIR` and `PRETRAINED_MODEL_PATH` for model loading

---

## Future Enhancements

- [ ] Extend direction finding to real-time/DB mode (requires component data from DB)
- [ ] Support for Observatory 2 direction finding (OBS2 already supported in 3D plot)
- [ ] Export anomaly directions to CSV/JSON
- [ ] Model fine-tuning options (freeze layers, adjust learning rate)

---

## Notes

- All changes are backward compatible (existing functionality preserved)
- Pre-trained models are optional (app works without them)
- Direction finding is optional (works only when component data available)
- Documentation follows existing format and style
