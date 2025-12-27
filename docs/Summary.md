# Magnavis Project Summary

## Project Overview

**Magnavis** is a Python application for magnetic field data visualization, analysis, and anomaly detection. It was developed by Prof. Saikat Ghosh's lab at IIT Kanpur's Department of Physics. The system collects real-time magnetic field data, uses AI to predict future values, and detects anomalies.

---

## Core Functionality

### 1. Real-Time Data Collection
- Fetches magnetic field data from USGS API (Barrow, Alaska station)
- Updates every 20 seconds via background threads
- Stores data in session-specific folders
- Handles network retries and errors gracefully

### 2. AI-Based Prediction
- LSTM neural network for forecasting
- Autoregressive prediction (100 future points)
- Adaptive learning (retrains as new data arrives)
- Runs in separate process to keep GUI responsive

### 3. Anomaly Detection
- Compares actual vs predicted values
- Uses interpolation for exact timestamp matching
- Dynamic threshold: `mean_error + (multiplier × std_error)`
- User-adjustable sensitivity (threshold multiplier: 0.1-10.0)
- Visualizes anomalies as red vertical lines on plots

### 4. Multi-Modal Visualization
- **Time Series Plots**: Static historical + dynamic real-time
- **3D Spatial Visualization**: VTK-based, interactive
- **Geographic Map Visualization**: India with magnetic field contours
- **CSV Data Import**: Load and visualize custom datasets

---

## Architecture & Components

### Main Application (`application.py`)
- ~1,820 lines of PyQt5 GUI code
- Multi-tab interface:
  - **Tab 1**: Data sources tree view
  - **Tab 2**: Time series plots (main visualization)
  - **Tab 3**: 3D visualization
  - **Tab 5**: Map visualization
- Thread-safe data fetching using QThread
- Session management (UUID-based)

### Anomaly Detector (`Anomaly_detector.py`)
- Statistical anomaly detection
- Interpolation-based timestamp matching
- Sliding window error history (last 1000 errors)
- Dynamic threshold calculation
- Real-time threshold adjustment

### LSTM Predictor (`predictor_ai.py`)
- TensorFlow/Keras LSTM model
- Architecture: LSTM(32) → Dense(16, ReLU) → Dense(1)
- Window-based training (default: 15 points)
- MinMaxScaler normalization
- Autoregressive forecasting
- **Anomaly Filtering**: Anomalous data points are automatically excluded from training dataset

### Data Fetcher (`data_convert_now.py`)
- USGS API integration
- JSON data parsing
- Pandas DataFrame conversion
- Session-based file storage

---

## How LSTM Model Gets Trained Iteratively: A Detailed Explanation

### The Big Picture

Think of the LSTM model like a student learning to predict the weather. Initially, the student studies historical weather patterns (initial training). Then, as they make predictions about tomorrow's weather, they learn from their own predictions and continuously improve (iterative training). This is exactly how the LSTM model works in this project!

### Step-by-Step: The Iterative Training Process

#### Phase 1: Initial Training (The Foundation)

**What Happens:**
1. The model receives historical magnetic field data (e.g., 5000 data points)
2. **NEW:** Anomalous data points (if any were previously detected) are automatically filtered out before training
3. It uses the filtered data (e.g., 5000 points, or fewer if anomalies were excluded) to learn the basic patterns
4. The data is normalized (scaled to 0-1 range) so the model can learn better
5. The model creates "training examples" using a sliding window approach:
   - Window size = 15 (looks at last 15 points to predict the 16th)
   - Creates thousands of training pairs: (past 15 values → next value)
6. The model trains on these examples for several "epochs" (complete passes through the data)
7. After training, the model "knows" the basic patterns in magnetic field data

**Analogy:** Like a student reading a textbook chapter multiple times to understand the concepts, but skipping the pages with obvious errors (anomalies) so they learn the correct patterns.

**Code Location:** Lines 54-57 in `predictor_ai.py`
```python
initial_data = field_scaled[:self.initial_train_points]
X_init, y_init = self.create_windowed_dataset(initial_data)
if len(X_init) > 0:
    self.model.fit(X_init, y_init, epochs=self.epochs_per_update, verbose=0)
```

---

#### Phase 2: Making Predictions One by One (The Iterative Loop)

Now comes the interesting part! The model predicts 100 future values, but it doesn't just predict all 100 at once. Instead, it predicts **one value at a time**, and after each prediction, it **retrains itself** with that new prediction!

**The Loop (Repeats 100 times):**

**For each of the 100 future predictions:**

**Step 1: Make a Prediction**
- The model looks at the last 15 known values (the "window")
- It uses its current knowledge to predict the next value
- This prediction is added to the list of future predictions

**Step 2: Update the Window**
- The window "slides forward" by removing the oldest value and adding the new prediction
- Now the window contains: [old values... + new prediction]
- This window will be used for the next prediction

**Step 3: Add Prediction to Training Data (If `update_training=True`)**
- The new prediction is added to the training dataset
- The training dataset now includes: [all historical data + all previous predictions]

**Step 4: Retrain the Model (The Key Step!)**
- The model creates new training examples from the expanded dataset
- It retrains itself using these new examples
- The model "learns" from its own predictions!
- This happens for a few epochs (default: 10 epochs per update)

**Step 5: Use Updated Model for Next Prediction**
- The newly trained model is now used for the next prediction
- The cycle repeats

**Code Location:** Lines 67-80 in `predictor_ai.py`
```python
for i in range(n_future):  # Loop 100 times
    # Step 1: Make prediction
    predicted_scaled = self.model.predict(current_window_reshaped, verbose=0)[0, 0]
    predictions.append(predicted_value)
    
    # Step 2: Update window
    current_window = np.concatenate([current_window[1:], np.array([[predicted_scaled]])], axis=0)
    
    # Step 3 & 4: Add to training data and retrain
    if self.update_training:
        training_data = np.concatenate([training_data, np.array([[predicted_scaled]])], axis=0)
        X_train, y_train = self.create_windowed_dataset(training_data)
        self.model.fit(X_train, y_train, epochs=self.epochs_per_update, verbose=0)
```

---

### Why This Approach is Powerful

#### 1. **Adaptive Learning**
- The model adapts to new patterns as it predicts
- If the magnetic field behavior changes, the model learns about it
- Like a student who learns from their mistakes and adjusts their understanding

#### 2. **Self-Correction**
- Each prediction becomes part of the training data
- The model learns from its own predictions
- If it makes a mistake, it can correct itself in future predictions

#### 3. **Handles Non-Stationary Data**
- Magnetic field patterns can change over time
- The iterative training allows the model to adapt to these changes
- Unlike a static model that only knows old patterns

#### 4. **Better Long-Term Predictions**
- By retraining after each prediction, the model maintains context
- It "remembers" the sequence of predictions it made
- This helps with accurate long-term forecasting

---

### A Concrete Example

Let's say we want to predict 5 future values (simplified example):

**Initial State:**
- Historical data: [10, 12, 14, 16, 18, 20, 22, 24, 26, 28] (10 points)
- Window size: 5
- Initial training: Uses all 10 points to learn patterns

**Iteration 1:**
- Window: [24, 26, 28, ?, ?] → Wait, we need 5 values!
- Actually, window: [20, 22, 24, 26, 28] (last 5 historical values)
- Model predicts: **30**
- Training data becomes: [10, 12, 14, ..., 28, **30**]
- Model retrains with this new data
- Window updates: [22, 24, 26, 28, **30**]

**Iteration 2:**
- Window: [22, 24, 26, 28, 30]
- Model (now retrained) predicts: **32**
- Training data: [10, 12, 14, ..., 28, 30, **32**]
- Model retrains again
- Window: [24, 26, 28, 30, **32**]

**Iteration 3:**
- Window: [24, 26, 28, 30, 32]
- Model predicts: **34**
- Training data: [10, 12, 14, ..., 30, 32, **34**]
- Model retrains
- Window: [26, 28, 30, 32, **34**]

**And so on...**

Notice how:
- Each prediction becomes part of the training data
- The model retrains after each prediction
- The window keeps sliding forward
- The model continuously learns from its own predictions

---

### Key Parameters That Control Iterative Training

#### 1. **`update_training` (Default: True)**
- **True**: Model retrains after each prediction (iterative learning)
- **False**: Model only trains once initially, then just predicts (static model)
- **When to use False**: Faster predictions, but model doesn't adapt

#### 2. **`epochs_per_update` (Default: 10)**
- How many times the model reviews the training data during each retraining
- Higher = more learning per update, but slower
- Lower = faster, but less learning per update
- **Balance**: 5-10 epochs is usually good

#### 3. **`window_size` (Default: 15)**
- How many past values the model looks at to make a prediction
- Larger = more context, but slower and more memory
- Smaller = faster, but might miss long-term patterns

#### 4. **`initial_train_points`**
- How much historical data to use for initial training
- More = better initial understanding, but slower startup
- Should be at least several times the window_size

---

### The Training Data Growth

As predictions are made, the training dataset grows:

- **Start**: 5000 historical points
- **After 1st prediction**: 5001 points (5000 + 1 prediction)
- **After 2nd prediction**: 5002 points (5000 + 2 predictions)
- **After 100th prediction**: 5100 points (5000 + 100 predictions)

This growing dataset means:
- The model has more examples to learn from
- It learns the "sequence" of its own predictions
- It maintains temporal context throughout the prediction process

---

### Computational Considerations

#### Why It Takes Time
- Each prediction triggers a retraining step
- Retraining involves:
  - Creating new training examples (windowed dataset)
  - Training the model for several epochs
  - Updating model weights
- For 100 predictions with 10 epochs each = 1000 training operations!

#### Why It's Worth It
- Much better prediction accuracy
- Adapts to changing patterns
- Handles long-term dependencies better
- More realistic predictions

#### Optimization
- The model is trained in a separate process (doesn't freeze GUI)
- Training happens in background
- User can continue using the application while predictions are generated

---

### Comparison: Iterative vs. Static Training

**Static Training (update_training=False):**
```
Initial Training → Model Fixed → Predict 100 values
```
- ✅ Fast
- ❌ Model doesn't adapt
- ❌ Predictions may drift over time
- ❌ Doesn't learn from its own predictions

**Iterative Training (update_training=True):**
```
Initial Training → Predict 1 → Retrain → Predict 2 → Retrain → ... → Predict 100
```
- ✅ Model adapts continuously
- ✅ Learns from its own predictions
- ✅ Better long-term accuracy
- ❌ Slower (but worth it!)

---

### Real-World Analogy

Imagine you're learning to play a musical instrument:

**Static Training:**
- You practice for a month, then perform a 10-song concert
- You don't adjust your technique during the concert
- If you make a mistake, you keep making the same mistake

**Iterative Training:**
- You practice for a month (initial training)
- During the concert, after each song, you reflect on your performance
- You adjust your technique based on what you learned
- Each song becomes a learning experience for the next song
- By the end, you're playing much better!

This is exactly what the LSTM model does - it learns and adapts with each prediction!

---

### Summary of Iterative Training

The LSTM model in this project uses a **sophisticated iterative training approach**:

1. **Initial Training**: Learns from historical data (anomalies automatically excluded)
2. **Prediction Loop**: For each of 100 future predictions:
   - Makes a prediction
   - Adds prediction to training data
   - Retrains the model with expanded dataset
   - Uses updated model for next prediction
3. **Continuous Learning**: Model improves with each prediction
4. **Adaptive**: Handles changing patterns in magnetic field data
5. **Anomaly Filtering**: Anomalous data points are automatically excluded from training in subsequent cycles

**Key Enhancement:** The training data is automatically filtered to exclude all previously detected anomalies, ensuring the model learns only from normal patterns. This creates a **self-improving system** where:
- Anomalies are detected
- Anomalies are excluded from training
- Model learns better normal patterns
- Better predictions result
- Better anomaly detection follows

This approach makes the model **self-improving** and **adaptive**, leading to more accurate long-term predictions!

---

## Technology Stack

### Core Frameworks
- **PyQt5**: GUI framework
- **TensorFlow 2.20.0**: Deep learning
- **VTK 9.4.1**: 3D visualization
- **Matplotlib 3.10.7**: 2D plotting
- **Pandas 2.2.3**: Data manipulation
- **NumPy 2.1.0+**: Numerical operations

### Specialized Libraries
- **GeoPandas**: Geographic data handling
- **pygeomag**: Magnetic field calculations (WMM2025)
- **scikit-learn**: Data normalization
- **mplcursors**: Interactive plot tooltips

---

## Data Flow

```
USGS API
    ↓
SessionDataManager (background thread)
    ↓
api_df / api_df_new (global DataFrames)
    ↓
_update_xydata() (every 20 seconds)
    ↓
new_x_t, new_y_mag_t (real-time data lists)
    ↓
_detect_anomalies() (AnomalyDetector)
    ↓
Anomaly timestamps stored in self.anomaly_times
    ↓
_save_data() → Filters out anomalies → predict_input.csv
    ↓
predictor_ai.py (separate process) [Trains on clean data]
    ↓
predict_out.csv
    ↓
_update_predictions_data()
    ↓
_update_canvas() (every 200ms)
    ↓
Visualization (plots with anomalies)
```

**Note:** The `_save_data()` method now automatically filters out all previously detected anomalies before saving data for LSTM training, ensuring the model trains only on normal patterns.

---

## Key Features

### Real-Time Updates
- **Data Fetching**: Every 20 seconds
- **Plot Updates**: Every 200ms (5 Hz)
- **Non-Blocking GUI**: Threading architecture

### User Controls
- **Anomaly Threshold Multiplier**: Adjustable (0.1-10.0, default 2.5)
- **CSV File Upload**: Load custom datasets
- **Time Range Selection**: Filter data by time
- **Interactive 3D Visualization**: Rotate, zoom, pan

### Visualization Features
- **Interactive Tooltips**: Hover to see values
- **Multiple Data Series**: Historical, real-time, predictions
- **Anomaly Markers**: Red vertical lines
- **Auto-Scaling Axes**: Dynamic range adjustment
- **Navigation Toolbars**: Standard matplotlib controls

### Data Management
- **Session-Based Organization**: UUID-based sessions
- **CSV Import/Export**: Data portability
- **Tree View**: Data source management
- **Logging System**: Status and error tracking

---

## Project Structure

```
magnavis_v4_Interpol_Enhance_Trg/
├── src/
│   ├── application.py          # Main GUI application
│   ├── Anomaly_detector.py     # Anomaly detection logic
│   ├── predictor_ai.py         # LSTM prediction model
│   ├── data_convert_now.py     # USGS API data fetcher
│   ├── ui_files/               # Qt Designer UI files
│   ├── data/                   # Shapefiles, flight data, etc.
│   ├── sessions/               # Session-specific data storage
│   └── resources/              # Research papers and documentation
├── docs/                       # Detailed explanations
│   ├── Application_Explanation.md
│   ├── Anomaly_Detector_Explanation.md
│   ├── Predictor_AI_Explanation.md
│   └── documents.md            # This file
└── requirements_feb_2025.txt   # Python dependencies
```

---

## Use Cases

1. **Real-Time Monitoring**: Track magnetic field variations
2. **Anomaly Detection**: Identify unusual patterns
3. **Predictive Analysis**: Forecast future values
4. **Research**: Analyze magnetic field data
5. **Education**: Visualize geophysical phenomena

---

## Technical Highlights

### Threading & Concurrency
- **QThread**: Non-blocking data fetching
- **QMutex**: Thread-safe data access
- **Subprocess**: Separate Python process for predictions

### Performance Optimizations
- **Sliding Window**: Error history limited to 1000 points
- **Efficient Interpolation**: Pandas time-based interpolation
- **LOD Actors**: VTK level-of-detail rendering
- **Optimized Update Frequencies**: Balance between responsiveness and performance

### Error Handling
- **Try-Except Blocks**: Comprehensive error handling
- **Graceful Degradation**: API failures don't crash app
- **Process Monitoring**: Prediction process health checks
- **Logging System**: Detailed error tracking

---

## Current State

The project appears to be in a **production-ready state** with:
- ✅ Complete GUI implementation
- ✅ Working real-time data pipeline
- ✅ Functional AI prediction system
- ✅ Advanced anomaly detection
- ✅ Multiple visualization modes
- ✅ Comprehensive documentation

The codebase shows evidence of recent enhancements, including:
- Interpolation-based anomaly detection
- Improved timestamp matching
- Enhanced user controls

---

## Summary

**Magnavis** is a comprehensive system for magnetic field analysis that combines:
- **Real-time data collection** from USGS
- **AI-powered forecasting** using LSTM (trained on filtered, anomaly-free data)
- **Statistical anomaly detection** with persistence across cycles
- **Intelligent training data filtering** (anomalies automatically excluded)
- **Multi-modal visualization** (2D, 3D, maps)
- **User-friendly PyQt5 interface**

**Key Innovation:** The system implements a **self-improving feedback loop**:
1. Anomalies are detected and stored
2. Anomalies are automatically excluded from LSTM training data
3. LSTM trains on clean, normal patterns only
4. Better predictions result from better training
5. Better predictions lead to better anomaly detection
6. Cycle continues, system continuously improves

It's suitable for research, monitoring, and educational purposes, with a focus on real-time processing, AI integration, interactive visualization, and intelligent data quality management.

---

## Complete Application Workflow: How Everything Works Together

### Overview: The Big Picture

Think of the Magnavis application like a **symphony orchestra** where different modules are like different musicians playing together:
- **application.py** = The conductor (coordinates everything)
- **data_convert_now.py** = The data collector (fetches music sheets)
- **predictor_ai.py** = The composer (creates future music)
- **Anomaly_detector.py** = The quality checker (finds wrong notes)
- **GUI/Plotting** = The stage (shows everything to the audience)

They all work together, but each has its own job and timing!

---

### How application.py Integrates the Modules

#### 1. **Integration with Anomaly_detector.py**

**At Startup (Line 902):**
```python
self.anomaly_detector = AnomalyDetector(threshold_multiplier=2.5, min_samples_for_threshold=10)
```

**What Happens:**
- `application.py` creates an `AnomalyDetector` object when the app starts
- This object lives inside the `Application` class
- It's ready to use, but waits for data before doing anything
- The detector is like a quality inspector waiting for products to check

**During Runtime:**
- When predictions are available, `application.py` calls:
  ```python
  anomalies_df, threshold = self.anomaly_detector.detect_anomalies(
      actual_times=realtime_actual_times,
      actual_values=realtime_actual_values,
      predicted_times=all_predicted_times,
      predicted_values=all_predicted_values
  )
  ```
- The detector does its work and returns results
- `application.py` uses these results to draw red lines on plots

**Key Point:** `application.py` **owns** the detector, but the detector does the actual anomaly detection work independently.

---

#### 2. **Integration with predictor_ai.py**

**Important:** `predictor_ai.py` is **NOT** imported directly! Instead, it runs as a **separate process**.

**At Startup:**
- `application.py` does NOT create a predictor object
- Instead, it prepares to launch `predictor_ai.py` as a subprocess

**When Predictions Are Needed (Line 1129-1151):**
```python
def start_prediction_process(self, input_file):
    python_exe = sys.executable
    python_app_file = os.path.join(APP_BASE, 'predictor_ai.py')
    command = [python_exe, python_app_file, input_file]
    self.prediction_process = subprocess.Popen(command, ...)
```

**What Happens:**
1. `application.py` saves data to `predict_input.csv`
2. It launches `predictor_ai.py` as a **separate Python process**
3. The predictor runs independently (doesn't block the GUI)
4. When done, it saves `predict_out.csv`
5. `application.py` periodically checks for the output file

**Key Point:** `predictor_ai.py` runs in a **completely separate process**, like a worker in another room. The GUI doesn't wait for it!

---

### Scenario 1: When GUI Plots Data

**What's Happening:**
The user sees plots updating on the screen.

**What Other Modules Are Doing:**

#### **application.py (Main Module):**
- **Active:** Running `_update_canvas()` every 200ms
- **Doing:**
  - Drawing lines on plots (blue, green, purple)
  - Drawing red vertical lines for anomalies
  - Adjusting axis limits
  - Redrawing the canvas
- **Status:** Very busy, updating display constantly

#### **data_convert_now.py (Data Fetcher):**
- **Active:** Running in background thread (every 20 seconds)
- **Doing:**
  - Fetching new data from USGS API
  - Parsing JSON responses
  - Converting to pandas DataFrame
  - Storing in global `api_df_new`
- **Status:** Working quietly in background, doesn't interrupt GUI

#### **predictor_ai.py (LSTM Predictor):**
- **Active:** Only if prediction process is running
- **Doing:**
  - Training LSTM model (if still training)
  - OR: Making predictions (if training done)
  - OR: Nothing (if process finished)
- **Status:** Completely independent, running in separate process

#### **Anomaly_detector.py:**
- **Active:** Only when called by `application.py`
- **Doing:**
  - Waiting (not doing anything unless called)
  - OR: Calculating anomalies (if `_detect_anomalies()` is called)
- **Status:** Sleeping most of the time, wakes up when needed

**Summary:** When plotting, the GUI is busy, data fetcher is working in background, predictor might be training, and anomaly detector is waiting.

---

### Scenario 2: When LSTM Is Getting Trained

**What's Happening:**
The LSTM model is learning from data (this can take seconds to minutes).

**What Other Modules Are Doing:**

#### **predictor_ai.py (LSTM Predictor):**
- **Active:** Very busy!
- **Doing:**
  - Reading training data from `predict_input.csv` (anomalies already filtered out)
  - Normalizing data (scaling to 0-1)
  - Creating windowed training examples
  - Training LSTM model for multiple epochs (on clean, normal data)
  - Making predictions one by one
  - Retraining after each prediction (if `update_training=True`)
  - Saving results to `predict_out.csv`
- **Status:** Using CPU heavily, but in separate process (doesn't freeze GUI)
- **Note:** Training data has been pre-filtered to exclude anomalies

#### **application.py (Main Module):**
- **Active:** Still running GUI normally
- **Doing:**
  - Checking if prediction process finished (periodically)
  - Updating plots with existing data
  - Handling user interactions
  - Fetching new real-time data
  - **NOT waiting** for predictor to finish
- **Status:** Responsive, user can still use the app

#### **data_convert_now.py (Data Fetcher):**
- **Active:** Running in background thread
- **Doing:**
  - Continuing to fetch new data every 20 seconds
  - Storing data for when predictor finishes
- **Status:** Unaffected by predictor training

#### **Anomaly_detector.py:**
- **Active:** Waiting
- **Doing:**
  - Nothing (can't detect anomalies without predictions)
  - Will be called after predictions are ready
- **Status:** Patiently waiting

**Summary:** When LSTM is training, the predictor is working hard in a separate process, but the GUI stays responsive, data keeps being fetched, and anomaly detector waits.

---

### Scenario 3: When Anomalies Are Being Calculated

**What's Happening:**
The system compares actual data with predictions to find anomalies.

**What Other Modules Are Doing:**

#### **Anomaly_detector.py:**
- **Active:** Very busy!
- **Doing:**
  - Receiving actual and predicted data from `application.py`
  - Interpolating predicted values at exact actual timestamps
  - Calculating differences (actual - predicted)
  - Updating error history
  - Calculating dynamic threshold
  - Identifying which points are anomalies
  - Returning results to `application.py`
- **Status:** Working hard, but very fast (milliseconds)

#### **application.py (Main Module):**
- **Active:** Coordinating the process
- **Doing:**
  - Calling `_detect_anomalies()` method
  - Preparing data (actual times/values, predicted times/values)
  - Passing data to anomaly detector
  - Receiving anomaly results
  - Storing anomaly times/values for plotting and future filtering
  - Triggering plot update to show red lines
  - Logging detection statistics
  - **Next:** Will filter these anomalies from future training data
- **Status:** Orchestrating, but detector does the heavy lifting

#### **predictor_ai.py (LSTM Predictor):**
- **Active:** Finished (already made predictions)
- **Doing:**
  - Nothing (process already completed)
  - Predictions are in `predict_out.csv` file
- **Status:** Done, results already saved

#### **data_convert_now.py (Data Fetcher):**
- **Active:** Running in background
- **Doing:**
  - Continuing to fetch new data
  - This new data will be used for future anomaly detection
- **Status:** Unaffected, working normally

**Summary:** When calculating anomalies, the detector does the math work, application.py coordinates, predictor is done, and data fetcher continues normally.

---

## Complete Workflow: From Startup to Anomaly Detection

### Phase 1: Application Startup

```
1. application.py starts
   ↓
2. Creates AnomalyDetector object (ready but waiting)
   ↓
3. Sets up GUI (tabs, plots, widgets)
   ↓
4. Starts background thread to fetch initial data
   ↓
5. Application ready, GUI shows splash screen
```

**All Modules Status:**
- ✅ `application.py`: Initializing
- ✅ `Anomaly_detector.py`: Created, waiting
- ⏸️ `predictor_ai.py`: Not started yet
- 🔄 `data_convert_now.py`: Fetching initial data

---

### Phase 2: Initial Data Collection

```
1. Background thread fetches 1 hour of data
   ↓
2. Data stored in api_df (global variable)
   ↓
3. application.py loads data into x_t, y_mag_t lists
   ↓
4. Plots show initial data (blue dots/line)
   ↓
5. Data saved to predict_input.csv (no anomalies to filter yet - first time)
   ↓
6. Prediction process started (predictor_ai.py launched)
```

**All Modules Status:**
- ✅ `application.py`: Displaying initial data, saving training data (no filtering needed yet)
- ✅ `Anomaly_detector.py`: Waiting (no predictions yet)
- 🔄 `predictor_ai.py`: Just started, beginning training on all data
- ✅ `data_convert_now.py`: Initial data fetched

---

### Phase 3: LSTM Training (Background)

```
1. predictor_ai.py reads predict_input.csv (anomalies already filtered out)
   ↓
2. Normalizes data (scales to 0-1)
   ↓
3. Creates windowed training examples (from clean data)
   ↓
4. Trains LSTM model (multiple epochs) on normal patterns only
   ↓
5. Starts making predictions (one by one)
   ↓
6. Retrains after each prediction (iterative learning)
   ↓
7. Saves predictions to predict_out.csv
   ↓
8. Process finishes
```

**All Modules Status:**
- ✅ `application.py`: GUI responsive, checking for predictions periodically
- ⏸️ `Anomaly_detector.py`: Waiting (predictions not ready)
- 🔄 `predictor_ai.py`: Training and predicting on filtered data (separate process)
- 🔄 `data_convert_now.py`: Fetching new data every 20 seconds

---

### Phase 4: Real-Time Data Updates

```
Every 20 seconds:
1. Background thread fetches new data
   ↓
2. Data stored in api_df_new
   ↓
3. application.py updates new_x_t, new_y_mag_t lists
   ↓
4. Plots updated with green line (real-time data)
   ↓
5. Data saved to predict_input.csv (updated)
```

**All Modules Status:**
- 🔄 `application.py`: Updating plots, filtering anomalies from training data before saving
- ⏸️ `Anomaly_detector.py`: Waiting
- ✅ `predictor_ai.py`: Finished (or still running if first time, trained on filtered data)
- 🔄 `data_convert_now.py`: Fetching every 20 seconds

---

### Phase 5: Prediction Results Available

```
1. application.py checks for predict_out.csv
   ↓
2. Finds file exists (predictions ready!)
   ↓
3. Reads predictions into predict_x_t, predict_y_t lists
   ↓
4. Plots updated with purple line (predictions)
   ↓
5. Calls _detect_anomalies()
```

**All Modules Status:**
- 🔄 `application.py`: Reading predictions, preparing for anomaly detection
- 🔄 `Anomaly_detector.py`: About to be called
- ✅ `predictor_ai.py`: Finished, results saved (trained on filtered data)
- 🔄 `data_convert_now.py`: Continuing to fetch data

---

### Phase 6: Anomaly Detection

```
1. application.py calls anomaly_detector.detect_anomalies()
   ↓
2. AnomalyDetector:
   - Interpolates predicted values at actual timestamps
   - Calculates differences
   - Updates error history
   - Calculates threshold
   - Identifies anomalies
   ↓
3. Returns anomalies to application.py
   ↓
4. application.py stores anomaly_times, anomaly_values
   ↓
5. Plots updated with red vertical lines (anomalies)
```

**All Modules Status:**
- 🔄 `application.py`: Coordinating anomaly detection, updating plots
- 🔄 `Anomaly_detector.py`: Actively calculating anomalies
- ✅ `predictor_ai.py`: Done
- 🔄 `data_convert_now.py`: Still fetching data

---

### Phase 7: Continuous Operation

```
This cycle repeats:
1. New data arrives (every 20 seconds)
   ↓
2. Plots update (every 200ms)
   ↓
3. Anomalies detected and stored
   ↓
4. Data saved for training (anomalies filtered out)
   ↓
5. New predictions generated (trained on clean data)
   ↓
6. Anomalies recalculated (when predictions available)
   ↓
7. Everything updates in real-time
```

**All Modules Status:**
- 🔄 `application.py`: Continuously updating GUI, filtering anomalies from training data
- 🔄 `Anomaly_detector.py`: Called periodically when predictions update
- 🔄 `predictor_ai.py`: Runs when new predictions needed (trains on filtered data)
- 🔄 `data_convert_now.py`: Fetching every 20 seconds

**Key Enhancement:** Each time data is saved for training, all previously detected anomalies are automatically excluded, ensuring the LSTM always trains on clean, normal data.

---

## Key Integration Points

### 1. **AnomalyDetector Integration**

**How:** Direct object creation and method calls
```python
# Created once at startup
self.anomaly_detector = AnomalyDetector(...)

# Called when needed
anomalies_df, threshold = self.anomaly_detector.detect_anomalies(...)
```

**Communication:** Synchronous (blocking) - detector finishes before continuing
**Timing:** Called after predictions are available

---

### 2. **predictor_ai.py Integration**

**How:** Separate subprocess (not imported)
```python
# Launched as separate process
subprocess.Popen([python_exe, 'predictor_ai.py', input_file])

# Results read from file
predictions = pd.read_csv('predict_out.csv')
```

**Communication:** Asynchronous (non-blocking) - via file I/O
**Timing:** Runs independently, GUI checks periodically

---

### 3. **data_convert_now.py Integration**

**How:** Function import and background threads
```python
# Function imported
from data_convert_now import get_timeseries_magnetic_data

# Called in background thread
worker.update_api_df(session_id, hours, start_time, new)
```

**Communication:** Asynchronous (non-blocking) - via global variables and signals
**Timing:** Every 20 seconds in background thread

---

## Module Responsibilities Summary

| Module | Primary Job | When Active | How Integrated |
|--------|------------|-------------|----------------|
| **application.py** | GUI, coordination, plotting | Always | Main module |
| **Anomaly_detector.py** | Calculate anomalies | When called | Direct object |
| **predictor_ai.py** | Train LSTM, predict | When launched | Separate process |
| **data_convert_now.py** | Fetch API data | Every 20s | Background thread |

---

## The Beautiful Concurrency

The application uses **three different concurrency patterns**:

1. **Threading** (data fetching): `QThread` for non-blocking API calls
2. **Subprocess** (predictions): Separate Python process for LSTM
3. **Timers** (plotting): `QTimer` for periodic GUI updates

This means:
- ✅ GUI never freezes
- ✅ Multiple things happen simultaneously
- ✅ User can interact while processing happens
- ✅ Everything stays responsive

---

## Real-World Analogy

Think of it like a **restaurant**:

- **application.py** = The head waiter (coordinates everything, serves customers)
- **data_convert_now.py** = The delivery person (brings ingredients every 20 minutes)
- **predictor_ai.py** = The chef (cooks in the kitchen, separate from dining area)
- **Anomaly_detector.py** = The quality inspector (checks dishes when ready)

The head waiter:
- Takes orders (user interactions)
- Coordinates with delivery person (data fetching)
- Sends orders to kitchen (starts prediction process)
- Checks when food is ready (reads prediction file)
- Calls quality inspector (anomaly detection)
- Serves customers (updates plots)

Everything happens in parallel, but the head waiter coordinates it all!

---

## Why Anomalies Appear Where Predictions Don't Exist: Explained

### The Problem You're Observing

You've noticed that **red vertical lines (anomalies) appear at timestamps where only actual data exists, but there's no predicted data**. This seems wrong because anomalies should only be detected when we can compare actual vs predicted values.

### Why This Happens: The Root Cause

The issue occurs because of **how the interpolation system works** in the AnomalyDetector. Let me explain step by step:

---

### Step 1: The Interpolation Tolerance Window

The AnomalyDetector uses a **15-minute tolerance window** around the prediction range:

```python
max_interpolation_distance = pd.Timedelta('15min')  # 15 minutes tolerance
```

**What this means:**
- If predictions exist from 10:00 AM to 11:00 AM
- The system will try to match actual data from **9:45 AM to 11:15 AM** (15 minutes before and after)
- This is done to handle slight timing mismatches

**The Problem:**
- Actual data might exist at 9:50 AM (within the 15-minute window)
- But the nearest prediction might be at 10:00 AM (10 minutes away)
- The system uses "nearest neighbor" fallback to assign a predicted value
- This creates a "predicted" value even though there's no real prediction at 9:50 AM!

---

### Step 2: How Interpolation Creates "Fake" Predictions

Here's what happens in the code (Anomaly_detector.py, lines 162-177):

```python
# For any remaining NaN values at the edges, use nearest neighbor as fallback
if predicted_interpolated['predicted'].isna().any():
    for idx in predicted_interpolated[predicted_interpolated['predicted'].isna()].index:
        # Find nearest prediction point
        time_diffs = (predicted_indexed.index - idx).abs()
        nearest_idx = time_diffs.idxmin()
        nearest_distance = time_diffs.min()
        
        # Only use nearest neighbor if within reasonable distance
        if nearest_distance <= max_interpolation_distance:  # 15 minutes
            predicted_interpolated.loc[idx, 'predicted'] = predicted_indexed.loc[nearest_idx, 'predicted']
```

**What this does:**
1. If interpolation fails (can't interpolate between two prediction points)
2. It finds the nearest prediction point
3. If that point is within 15 minutes, it uses that prediction value
4. **This assigns a predicted value to an actual timestamp where no prediction actually exists!**

**Example:**
- Actual data at: **9:50 AM** (value: 50,000 nT)
- Nearest prediction at: **10:00 AM** (value: 49,500 nT)
- Distance: 10 minutes (within 15-minute tolerance)
- System assigns: predicted = 49,500 nT at 9:50 AM
- Difference: |50,000 - 49,500| = 500 nT
- If threshold is 400 nT → **ANOMALY DETECTED!**
- **But there's no real prediction at 9:50 AM!**

---

### Step 3: Why This Causes the Visual Issue

When an anomaly is detected at 9:50 AM:
1. The anomaly timestamp (9:50 AM) is stored in `self.anomaly_times`
2. A red vertical line is drawn at 9:50 AM
3. **But when you look at the plot, there's no purple prediction line at 9:50 AM!**
4. You only see:
   - Blue/green line (actual data) at 9:50 AM
   - Red vertical line (anomaly marker) at 9:50 AM
   - Purple line (predictions) starting at 10:00 AM

**This creates the visual confusion:** Anomaly marked where predictions don't exist!

---

### Step 4: The Timing Mismatch Problem

This issue is more common when:

1. **Predictions start after actual data:**
   - Actual data: 9:00 AM to 10:00 AM
   - Predictions: 10:00 AM to 11:00 AM
   - Actual data at 9:55 AM gets matched with prediction at 10:00 AM (5 minutes away)
   - Anomaly detected at 9:55 AM, but no prediction line visible there

2. **Predictions end before actual data:**
   - Predictions: 9:00 AM to 10:00 AM
   - Actual data: 9:00 AM to 10:30 AM
   - Actual data at 10:15 AM gets matched with prediction at 10:00 AM (15 minutes away)
   - Anomaly detected at 10:15 AM, but no prediction line visible there

3. **Sparse predictions:**
   - Predictions every 5 minutes: 10:00, 10:05, 10:10, 10:15
   - Actual data every 1 minute: 10:00, 10:01, 10:02, 10:03, ...
   - Actual data at 10:02 AM gets interpolated/extrapolated from 10:00 and 10:05
   - But if interpolation fails, it uses nearest neighbor (10:00 or 10:05)
   - Creates "fake" predictions at 10:02 AM

---

### Why the Code Allows This

The code tries to filter out invalid matches (line 188 in Anomaly_detector.py):
```python
merged = merged.dropna(subset=['predicted'])
```

**But the problem is:**
- The nearest neighbor fallback (lines 164-177) **fills in NaN values** before this filter
- So by the time we reach the filter, there are no NaN values to filter out
- The "fake" predicted values have already been assigned

**The filter only removes points where:**
- Interpolation completely failed
- Nearest neighbor is more than 15 minutes away
- But it doesn't remove points where nearest neighbor was used within 15 minutes

---

### The Visual Evidence

When you see this issue, you'll notice:

1. **Red vertical lines** at timestamps where:
   - Actual data (blue/green line) exists
   - But no purple prediction line exists
   - The purple line might start/end nearby, but not at that exact timestamp

2. **The anomaly appears "orphaned":**
   - It's not aligned with any prediction point
   - It's in a region where predictions don't actually exist
   - But the system "created" a prediction value there using nearest neighbor

---

### Is This a Bug or a Feature?

**It's actually a design trade-off:**

**The Intent (Good):**
- Handle slight timing mismatches (predictions and actual data not perfectly aligned)
- Allow comparison even when timestamps don't match exactly
- Use interpolation to get more accurate comparisons

**The Side Effect (Problematic):**
- Creates "fake" predictions using nearest neighbor
- Marks anomalies where no real prediction exists
- Causes visual confusion

**The Trade-off:**
- **Strict matching:** Only compare where exact timestamps match → Fewer comparisons, might miss real anomalies
- **Loose matching (current):** Use 15-minute tolerance → More comparisons, but creates "fake" predictions

---

### When This Happens Most Often

1. **At the edges of prediction ranges:**
   - Right before predictions start
   - Right after predictions end
   - Where the 15-minute tolerance extends beyond actual prediction range

2. **When prediction frequency is low:**
   - Predictions every 5-10 minutes
   - Actual data every 1 minute
   - More gaps to fill with nearest neighbor

3. **When there's a time gap:**
   - Predictions start at 10:00 AM
   - Actual data exists from 9:45 AM
   - The 15-minute window tries to match 9:45 AM data with 10:00 AM prediction

---

### How to Identify This Issue

Look for these signs in the logs:

1. **Warning messages:**
   ```
   Anomaly Detection: WARNING - No matched pairs found between actual and predicted data!
   Anomaly Detection: Time gap between actual end and predicted start: X minutes
   ```

2. **Anomaly statistics:**
   - Check if anomalies are clustered at the edges of prediction range
   - Look for anomalies where prediction line doesn't exist

3. **Visual inspection:**
   - Red lines where no purple line exists
   - Red lines at the very beginning or end of prediction range
   - Red lines in gaps between prediction points

---

### ✅ The Fix: What Was Changed

**FIXED:** The issue has been resolved in the code!

**Changes Made to `Anomaly_detector.py`:**

1. **Removed 15-minute tolerance window:**
   - **Before:** `(actual_sorted['time'] >= pred_min_time - max_interpolation_distance)`
   - **After:** `(actual_sorted['time'] >= pred_min_time)`
   - Now only compares actual data that falls within the actual prediction time range
   - No more extending beyond prediction boundaries

2. **Removed nearest neighbor fallback:**
   - Completely removed the code block (lines 162-177) that was creating "fake" predictions
   - No more assigning nearest prediction values to timestamps where predictions don't exist
   - Only uses interpolation, which requires at least 2 prediction points on either side

3. **Stricter filtering:**
   - Points where interpolation isn't possible (at edges) remain NaN
   - These are automatically filtered out by `dropna(subset=['predicted'])`
   - Only valid interpolated values are used for anomaly detection

**Result:**
- ✅ Anomalies will only appear where predictions actually exist (or can be interpolated between prediction points)
- ✅ No more "fake" predictions created using nearest neighbor
- ✅ Red vertical lines will only appear where purple prediction lines exist (or between them)
- ✅ Visual confusion eliminated

**Trade-off:**
- ⚠️ Slightly fewer comparisons (no edge cases beyond prediction range)
- ✅ But all comparisons are now valid and meaningful
- ✅ No more false anomalies at timestamps without predictions

**What This Means:**
- If predictions exist from 10:00 AM to 11:00 AM, anomalies will only be detected for actual data between 10:00 AM and 11:00 AM
- Actual data at 9:55 AM or 11:05 AM will NOT be compared (no anomaly detection there)
- This ensures all anomalies are based on real, valid predictions

---

### Summary: Why This Happens

**In Simple Words:**

1. The system tries to be helpful by matching actual data with predictions even when timestamps don't match exactly
2. It uses a 15-minute "tolerance window" to find nearby predictions
3. When it can't interpolate, it uses the nearest prediction (even if it's minutes away)
4. This creates a "predicted value" at a timestamp where no real prediction exists
5. If the difference is large, it marks an anomaly
6. The red line appears at that timestamp, but the purple prediction line doesn't exist there
7. **Result:** You see an anomaly marker where predictions don't actually exist!

**The Core Issue:**
The nearest neighbor fallback creates "fake" predictions within the 15-minute window, leading to anomalies being marked at timestamps where no real prediction data exists.

**Why It's Not Completely Wrong:**
The system is trying to compare actual data with the "best available" prediction, even if it's not at the exact same time. The anomaly might still be valid (the actual value is different from what was predicted nearby), but it's visually confusing because the prediction line doesn't exist at that exact timestamp.

---

## Anomaly Persistence: Do Anomalies Remain Visible Across Cycles?

### Quick Answer: **YES** ✅

**Anomalies DO persist across detection cycles.** Anomalies are accumulated and remain visible until the application is closed or the limit is reached.

### How It Works

1. **Storage (Lines 1367-1393):**
   ```python
   # FIXED: Accumulate anomalies instead of replacing (retain across cycles)
   # Add new anomalies, avoiding duplicates (same timestamp)
   for new_time, new_value in zip(new_anomaly_times, new_anomaly_values):
       if new_time_dt not in existing_times_set:
           self.anomaly_times.append(new_time_dt)
           self.anomaly_values.append(new_value)
   ```
   - Uses **append**, not assignment
   - **ACCUMULATES** anomalies across cycles
   - Prevents duplicates (same timestamp won't appear twice)

2. **Memory Management (Line 1375):**
   ```python
   max_anomalies = 1000  # Reasonable limit to prevent memory issues
   # If we're at the limit, remove oldest anomaly (FIFO)
   if len(self.anomaly_times) >= max_anomalies:
       self.anomaly_times.pop(0)
       self.anomaly_values.pop(0)
   ```
   - Maximum 1000 anomalies stored
   - Oldest anomalies removed first (FIFO) when limit reached
   - Prevents memory issues during long-running sessions

3. **Visualization (Line 1493-1506):**
   ```python
   # Remove old vertical lines if they exist
   for vline in self.anomaly_vertical_lines:
       vline.remove()
   # Then redraw ALL anomalies (including previous ones)
   for anomaly_time in anomaly_times_plot:
       vline_dynamic = self._dynamic_ax.axvline(...)
   ```
   - Removes old lines, then redraws **ALL** accumulated anomalies
   - Shows both current and historical anomalies

4. **No Clearing on Empty Cycles:**
   - If no new anomalies found, previous anomalies are **retained**
   - Anomalies are only cleared when application closes

### Example Behavior

- **Cycle 1:** Anomaly at 10:00 AM → Red line appears
- **Cycle 2 (20s later):** No new anomalies → 10:00 AM line **remains visible**
- **Cycle 3:** New anomaly at 10:15 AM → Both 10:00 AM and 10:15 AM lines visible
- **Cycle 4:** Another anomaly at 10:30 AM → All three lines visible
- **After 1000 anomalies:** Oldest ones are removed (FIFO), newest remain

### Benefits of This Design

- ✅ **Historical tracking:** See anomaly trends over time
- ✅ **Better analysis:** Compare anomalies across different cycles
- ✅ **No data loss:** Important anomalies remain visible
- ✅ **Memory safe:** Automatic limit prevents issues
- ✅ **Duplicate prevention:** Same anomaly won't appear multiple times

### Features

1. **Accumulation:** Anomalies are added to the list, not replaced
2. **Duplicate Prevention:** Checks if timestamp already exists before adding
3. **Memory Limit:** Maximum 1000 anomalies (configurable)
4. **FIFO Removal:** Oldest anomalies removed when limit reached
5. **Persistence:** Anomalies remain until application closes

**For detailed analysis, see:** `docs/Anomaly_Persistence_Report.md`

---

## Anomaly Filtering from LSTM Training Dataset

### Overview

**NEW FEATURE:** Anomalous data points identified by the AnomalyDetector are automatically excluded from the training dataset used by the LSTM model. This ensures the model learns only from normal patterns, leading to better predictions.

### How It Works

**Location:** `application.py`, `_save_data()` method (lines 937-962)

**Process:**

1. **Before Saving Training Data:**
   - System checks all previously detected anomalies (`self.anomaly_times`)
   - Creates a set of anomaly timestamps for efficient lookup

2. **Data Filtering:**
   ```python
   for each (time, value) in training data:
       if time NOT in anomaly_times_set:
           keep this data point  # Normal data
       else:
           exclude this data point  # It's an anomaly
   ```

3. **Save Filtered Data:**
   - Only non-anomalous data points are saved to `predict_input.csv`
   - LSTM receives clean training data (anomalies excluded)

### Benefits

1. **Better Model Training:**
   - Model learns from normal patterns only
   - Not influenced by outliers or anomalies
   - More accurate predictions

2. **Automatic Filtering:**
   - Happens automatically every time data is saved
   - No manual intervention needed
   - Uses accumulated anomaly history

3. **Progressive Improvement:**
   - As more anomalies are detected, more are excluded
   - Model training becomes cleaner over time
   - Self-improving system

4. **Efficient Implementation:**
   - Uses set-based lookup (O(1) per check)
   - Fast filtering even with many anomalies
   - Minimal performance impact

### Example Flow

**Cycle 1:**
- Data collected: 1000 points
- Anomalies detected: 0
- Data saved: 1000 points (0 excluded)
- LSTM trains on: 1000 points

**Cycle 2:**
- Data collected: 1000 points
- Anomalies detected: 1 at 10:00 AM
- Data saved: 999 points (1 excluded: 10:00 AM)
- LSTM trains on: 999 points (10:00 AM excluded)

**Cycle 3:**
- Data collected: 1000 points
- Anomalies detected: 1 at 10:15 AM (total: 2)
- Data saved: 998 points (2 excluded: 10:00 AM, 10:15 AM)
- LSTM trains on: 998 points (both anomalies excluded)

**Cycle 4:**
- Data collected: 1000 points
- Anomalies detected: 0 (total still: 2)
- Data saved: 998 points (2 excluded: previous anomalies)
- LSTM trains on: 998 points (anomalies still excluded)

### Key Features

1. **Uses Accumulated Anomalies:**
   - Filters based on ALL previously detected anomalies
   - Not just current cycle's anomalies
   - Ensures comprehensive filtering

2. **Exact Timestamp Matching:**
   - Uses exact datetime comparison
   - Handles different datetime formats
   - Accurate filtering

3. **Logging:**
   - Reports how many anomalies were excluded
   - Log messages: `"Saved prediction input file: ... with 998 data points (2 anomalies excluded from training)"`

4. **No Impact on Visualization:**
   - Anomalies are still shown on plots (red lines)
   - Filtering only affects training data
   - Visualization remains complete

### Code Implementation

**Location:** `application.py`, lines 937-962

```python
# NEW: Filter out anomalous data points from training dataset
# Convert anomaly times to a set for efficient lookup
anomaly_times_set = set()
if self.anomaly_times and len(self.anomaly_times) > 0:
    # Normalize anomaly times to datetime for comparison
    for anomaly_time in self.anomaly_times:
        if isinstance(anomaly_time, (datetime, pd.Timestamp)):
            anomaly_times_set.add(anomaly_time)
        else:
            anomaly_times_set.add(pd.to_datetime(anomaly_time))

# Filter out data points that match anomaly timestamps
filtered_x_t = []
filtered_y_t = []
excluded_count = 0

for time_val, mag_val in zip(x_t, y_t):
    time_dt = pd.to_datetime(time_val) if not isinstance(time_val, (datetime, pd.Timestamp)) else time_val
    if time_dt not in anomaly_times_set:
        filtered_x_t.append(time_val)
        filtered_y_t.append(mag_val)
    else:
        excluded_count += 1

# Save filtered data
df_save_inp = pd.DataFrame({'x': filtered_x_t, 'y': filtered_y_t})
```

### Integration with Other Features

**Works with:**
- ✅ Anomaly persistence (uses accumulated anomalies)
- ✅ Duplicate prevention (same anomaly excluded once)
- ✅ Memory limit (uses all anomalies up to limit)
- ✅ Real-time updates (filters on each save)

**Does NOT affect:**
- Visualization (anomalies still shown)
- Anomaly detection (detection happens before filtering)
- Prediction process (only affects training data)

### Why This Matters

**Traditional Approach:**
- Model trains on all data (including anomalies)
- Anomalies can skew the model
- Model learns to predict anomalies as "normal"
- Less accurate predictions

**With Anomaly Filtering:**
- Model trains only on normal data
- Model learns true normal patterns
- Better predictions for normal behavior
- More reliable forecasting

### Summary

The system now implements **intelligent training data filtering**:
- Anomalies are detected and stored
- Training data is automatically filtered
- LSTM trains on clean, normal data only
- Better predictions result from better training

This creates a **self-improving system** where:
1. Anomalies are detected
2. Anomalies are excluded from training
3. Model learns better patterns
4. Better predictions lead to better anomaly detection
5. Cycle continues, system improves

---

*Document created: 2025*
*Last updated: After anomaly filtering implementation*

