# Documentation: `src/train_gru_pretrained.py`

- **File**: `src/train_gru_pretrained.py`
- **Purpose**: Pre-train GRU models on extended historical magnetic field data (e.g., 2-4 months) to learn daily, weekly, and seasonal patterns. Trains **one model per sensor** (typically 6 models for 6 sensors).

---

## Overview

This script trains separate GRU models for each sensor found in the input CSV files. Each model learns sensor-specific patterns from extended historical data, which improves prediction accuracy and reduces startup time in the main application.

**Key features:**
- **Per-sensor training**: Automatically groups data by `sensor_id` and trains one model per sensor
- **Multiple CSV support**: Can load and combine multiple CSV files (e.g., from "Large Files" folder)
- **Flexible input formats**: Handles raw CSV (sensor_id, timestamp, b_x, b_y, b_z) or processed formats
- **Model persistence**: Saves models as `gru_pretrained_<sensor_id>.keras` with associated scaler files

---

## Usage

### Basic usage (train all sensors from a folder)

```bash
python src/train_gru_pretrained.py --folder "Large Files" models/ --epochs 50
```

This will:
1. Find all CSV files in "Large Files" folder
2. Group data by `sensor_id`
3. Train one model per sensor
4. Save models to `models/gru_pretrained_<sensor_id>.keras`

### Train specific sensors only

```bash
python src/train_gru_pretrained.py --folder "Large Files" models/ --sensors OBS1_1 OBS1_2 OBS1_3 --epochs 50
```

### Train from specific CSV files

```bash
python src/train_gru_pretrained.py "Large Files/file1.csv,Large Files/file2.csv" models/ --epochs 50
```

### Use glob pattern

```bash
python src/train_gru_pretrained.py "Large Files/magnetic_data_*.csv" models/ --epochs 50
```

---

## Command-line arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `input_csv` | Conditional | Path to CSV file(s) - can be single file, comma-separated list, or glob pattern. Required if `--folder` not used. |
| `output_model_dir` | Yes | Directory where trained models will be saved (e.g., `models/`). Models named: `gru_pretrained_<sensor_id>.keras` |
| `--folder` | Conditional | Folder containing CSV files to combine. Alternative to `input_csv`. |
| `--epochs` | No | Number of training epochs per sensor (default: 50) |
| `--use-yearly-cycle` | No | Include yearly seasonal features (default: True) |
| `--no-yearly-cycle` | No | Disable yearly seasonal features |
| `--window-size` | No | Window size for sequences (default: 15) |
| `--learning-rate` | No | Learning rate for Adam optimizer (default: 0.001) |
| `--batch-size` | No | Batch size for training (default: 32) |
| `--sensors` | No | Filter: only train models for sensors containing these strings (e.g., `--sensors OBS1_1 OBS1_2`) |

---

## Functions

### `load_magnetic_data_by_sensor(csv_paths)`

Loads magnetic field data from CSV files, grouped by sensor_id.

**Parameters:**
- `csv_paths` (list of str): List of CSV file paths to load

**Returns:**
- `sensor_data` (dict): Dictionary mapping `sensor_id -> (timestamps, field_data)`
  - `timestamps`: list of datetime objects
  - `field_data`: list of float (magnetic field magnitudes in nT)

**Supported CSV formats:**
1. **Time series format**: Columns `('x', 'y')` - single sensor per file
2. **Processed format**: Columns `('timestamp', 'mag_H_nT')` - single sensor per file
3. **Raw format**: Columns `('sensor_id', 'timestamp', 'b_x', 'b_y', 'b_z')` - multiple sensors per file
   - Automatically computes magnitude: `mag_total_nT = sqrt(b_x² + b_y² + b_z²)`
   - Groups by sensor_id and averages duplicates at same timestamp

**Processing:**
- Sorts data chronologically per sensor
- Removes duplicate timestamps (keeps first occurrence)
- Returns separate time series for each unique sensor_id

---

### `train_pretrained_model(csv_paths, output_model_dir, epochs=50, use_yearly_cycle=True, window_size=15, learning_rate=0.001, batch_size=32, sensor_filter=None)`

Main training function that trains one GRU model per sensor.

**Parameters:**
- `csv_paths` (list of str): List of CSV file paths to load
- `output_model_dir` (str): Directory where models will be saved
- `epochs` (int): Number of training epochs per sensor (default: 50)
- `use_yearly_cycle` (bool): Include yearly seasonal features (default: True)
- `window_size` (int): Number of time steps to look back (default: 15)
- `learning_rate` (float): Learning rate for Adam optimizer (default: 0.001)
- `batch_size` (int): Batch size for training (default: 32)
- `sensor_filter` (list of str, optional): Only train models for sensors matching these IDs

**Process:**
1. Loads data grouped by sensor using `load_magnetic_data_by_sensor()`
2. Filters sensors if `sensor_filter` provided
3. For each sensor:
   - Calls `_train_single_sensor_model()` to train and save the model
   - Models saved as: `output_model_dir/gru_pretrained_<sensor_id>.keras`
   - Scaler saved as: `output_model_dir/gru_pretrained_<sensor_id>_scaler.pkl`
4. Prints summary of trained models

---

### `_train_single_sensor_model(sensor_id, timestamps, field_data, output_model_dir, epochs, use_yearly_cycle, window_size, learning_rate, batch_size)`

Trains a single GRU model for one sensor.

**Process:**
1. Creates `GRUPredictor` instance with specified parameters
2. Builds feature matrix:
   - Scales magnetic field using MinMaxScaler
   - Computes cyclic time features (daily sin/cos, optionally yearly sin/cos)
   - Combines into feature matrix: `[mag_scaled, sin_day, cos_day, (sin_year, cos_year)]`
3. Creates windowed dataset for sequence learning
4. Builds GRU model architecture
5. Trains model with validation split (10%)
6. Saves model and scaler to disk

**Model architecture:**
- GRU layer: 32 units
- Dense layer: 16 units (ReLU activation)
- Output layer: 1 unit (magnetic field prediction)
- Optimizer: Adam
- Loss: Mean squared error

---

## Output files

For each sensor, the script creates:

1. **Model file**: `gru_pretrained_<sensor_id>.keras`
   - Contains model architecture and trained weights
   - Can be loaded using `GRUPredictor.load_model()`

2. **Scaler file**: `gru_pretrained_<sensor_id>_scaler.pkl`
   - Contains MinMaxScaler state for consistent feature scaling
   - Automatically loaded with the model

**Example output:**
```
models/
  ├── gru_pretrained_S20250926_100914_44180345587365_1.keras
  ├── gru_pretrained_S20250926_100914_44180345587365_1_scaler.pkl
  ├── gru_pretrained_S20250926_100914_44180345587365_2.keras
  ├── gru_pretrained_S20250926_100914_44180345587365_2_scaler.pkl
  └── ... (one per sensor)
```

---

## Using pre-trained models in the application

After training, set the `PRETRAINED_MODEL_DIR` environment variable:

```bash
export PRETRAINED_MODEL_DIR="models"
python src/application_temp.py
```

The application will:
1. Detect which sensor is being predicted
2. Look for matching pre-trained model (e.g., `gru_pretrained_OBS1_1.keras`)
3. Load the model and scaler automatically
4. Use pre-trained weights for faster, more accurate predictions
5. Fall back to training from scratch if no pre-trained model found

---

## Benefits of pre-training

- **Better accuracy**: Models learn long-term patterns (daily, weekly, seasonal) from extended data
- **Faster startup**: No need to train from scratch on each run
- **Sensor-specific**: Each sensor gets its own model, accounting for sensor-specific characteristics
- **Transfer learning**: Pre-trained models can be fine-tuned on new data if `update_training=True`

---

## Example workflow

1. **Prepare data**: Ensure CSV files in "Large Files" folder contain 2-4 months of data
2. **Train models**: `python src/train_gru_pretrained.py --folder "Large Files" models/ --epochs 50`
3. **Verify output**: Check that 6 model files were created in `models/` directory
4. **Use in app**: Set `PRETRAINED_MODEL_DIR="models"` and run `application_temp.py`
5. **Monitor**: Check logs to confirm pre-trained models are being loaded

---

## Notes

- Training time depends on data size and number of epochs (expect 5-30 minutes per sensor for 2 months of data)
- Models are sensor-specific: each sensor's model learns that sensor's unique characteristics
- The script automatically handles different CSV formats and sensor naming conventions
- Yearly cycle features (`--use-yearly-cycle`) help capture seasonal variations in magnetic field
