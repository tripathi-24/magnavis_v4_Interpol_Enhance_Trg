# Magnavis Application - Simple Explanation

## What is the Application?

The Magnavis Application is the main graphical user interface (GUI) for the magnetic field visualization and analysis system. It's like the control center that brings together all the different components:

- **Data Collection**: Fetches real-time magnetic field data from USGS API
- **Visualization**: Shows data in multiple ways (plots, 3D, maps)
- **Prediction**: Runs AI model to predict future values
- **Anomaly Detection**: Identifies unusual patterns in the data
- **Data Management**: Handles CSV imports, sessions, and file operations

Think of it as the dashboard of a car - it shows you all the important information and lets you control everything from one place.

---

## The Big Picture: Application Architecture

The application is built using **PyQt5** (a Python GUI framework) and consists of several main components:

1. **Application Class**: The main application controller
2. **ApplicationWindow Class**: The main window with tabs and UI
3. **MagTimeSeriesWidget**: Widget for time series visualization
4. **DataSourceManager**: Manages data sources (CSV files, time series)
5. **SessionDataManager**: Handles data fetching in background threads
6. **Visualization Components**: VTK for 3D, Matplotlib for plots

---

## Step-by-Step: Application Startup

### Step 1: Application Initialization

When you run `application.py`, the `Application` class is created:

**What happens:**

1. **Create Splash Screen**
   - Shows loading screen with "Magnavis" logo
   - Displays progress messages as components load

2. **Set Up Session**
   - Generate unique session ID (UUID)
   - Create session folder for storing data
   - Each run gets its own session

3. **Initialize Managers**
   - `DataSourceManager`: For handling data sources
   - `AnomalyDetector`: For detecting anomalies
   - Various lists for storing data

4. **Set Up Data Storage**
   - Lists for historical data (`x_t`, `y_mag_t`)
   - Lists for real-time data (`new_x_t`, `new_y_mag_t`)
   - Lists for predictions (`predict_x_t`, `predict_y_t`)

5. **Create Main Window**
   - Load UI from `.ui` file
   - Set up tabs and widgets
   - Connect signals and slots (event handlers)

---

### Step 2: Loading Visualization Framework

The application has multiple visualization components:

**A. 3D Visualization (VTK)**
- Creates a 3D renderer for spatial data
- Sets up camera and interaction controls
- Can display magnetic field data as 3D points
- Interactive: rotate, zoom, pan

**B. Map Visualization (Tab 5)**
- Shows India map with magnetic field contours
- Calculates magnetic field using WMM2025 model
- Overlays India boundary shapefile
- Static visualization (doesn't update in real-time)

**C. Time Series Plots (Tab 2)**
- Two plots: Static (historical) and Dynamic (real-time)
- Static plot: Shows all historical data
- Dynamic plot: Updates every 200ms (5 times per second)
- Data updates every 20 seconds

---

### Step 3: Starting Data Collection

**Background Thread System:**

The application uses **QThread** to fetch data without freezing the GUI:

1. **Create Worker Thread**
   - `SessionDataManager` runs in separate thread
   - Fetches data from USGS API
   - Doesn't block the main GUI thread

2. **Initial Data Fetch**
   - Fetches last 1 hour of data on startup
   - Stores in `api_df` (global variable)
   - Thread-safe using mutex (mutual exclusion lock)

3. **Periodic Updates**
   - Timer triggers every 20 seconds
   - Fetches new data since last update
   - Stores in `api_df_new`
   - Updates plots automatically

**Why threads?**
- API calls can take time (network delay)
- Without threads, GUI would freeze during data fetch
- With threads, GUI stays responsive

---

### Step 4: Loading Plot Framework

**Static Plot (Historical Data):**
- Shows all data from initial fetch
- Blue dots for each data point
- Includes tooltips (hover to see values)
- Doesn't update automatically

**Dynamic Plot (Real-Time):**
- Shows historical data initially (blue line)
- Updates with new data (green line)
- Shows predictions (purple line)
- Shows anomalies (red vertical lines)
- Auto-scales axes as data grows

**Timers:**
- **Data Timer**: Fetches new data every 20 seconds
- **Drawing Timer**: Updates plot every 200ms (5 Hz)
- Two timers = smooth updates without lag

---

## Key Components Explained

### 1. SessionDataManager (Background Data Fetcher)

**Purpose:** Fetch magnetic field data from USGS API without blocking GUI

**How it works:**
1. Runs in separate thread (QThread)
2. Calls `get_timeseries_magnetic_data()` from `data_convert_now.py`
3. Stores result in global DataFrame (`api_df` or `api_df_new`)
4. Emits signal when done
5. Main thread updates plots when signal received

**Thread Safety:**
- Uses `QMutex` to prevent race conditions
- Only one thread can access data at a time
- Prevents data corruption

**Example:**
```
Main Thread: "Hey, fetch new data!"
Worker Thread: "OK, fetching..." (doesn't block main thread)
Worker Thread: "Done!" (emits signal)
Main Thread: "Great, updating plots!"
```

---

### 2. ApplicationWindow (Main Window)

**Purpose:** The main GUI window with tabs and controls

**Tabs:**
- **Tab 1**: Data sources tree view
- **Tab 2**: Time series plots (main visualization)
- **Tab 3**: 3D visualization
- **Tab 5**: Map visualization

**Key Features:**
- **File Upload**: Load CSV files with magnetic field data
- **Tree View**: Shows all data sources
- **Log Panel**: Shows status messages and errors
- **Menu Bar**: Actions like "Upload" and "Add Time Series"

**Data Flow:**
```
User clicks "Upload" 
→ Opens file dialog
→ Loads CSV file
→ Creates World object
→ Adds to DataSourceManager
→ Updates tree view
→ Updates 3D visualization
```

---

### 3. MagTimeSeriesWidget (Time Series Controls)

**Purpose:** Widget for controlling time series visualization

**Features:**
- **Date/Time Pickers**: Select time range for data
- **Refresh Rate Selector**: Choose update frequency (not currently active)
- **Anomaly Threshold Control**: Adjust sensitivity of anomaly detection
  - Spinbox to change threshold multiplier
  - Real-time updates when changed
  - Tooltip explains how it works

**Threshold Control:**
- Default: 2.5
- Range: 0.1 to 10.0
- Higher = fewer anomalies (more strict)
- Lower = more anomalies (more sensitive)
- Changes apply immediately

---

### 4. DataSourceManager (Data Organization)

**Purpose:** Manages all data sources in the application

**Data Source Types:**
- **World**: Spatial data (CSV files with lat/lon/alt/mag)
- **TimeSeries**: Time series data (from API or files)

**Lists:**
- `_world_list`: All spatial data sources
- `_timeseries_list`: All time series sources
- `_added_sources`: Sources already shown in tree view

**Methods:**
- `loadCsv()`: Load CSV file and create World object
- `createTimeSeriesSource()`: Create new time series source

---

### 5. Prediction System

**How Predictions Work:**

1. **Data Collection**
   - Application collects real-time data
   - Saves to `predict_input.csv` in session folder
   - File format: CSV with 'x' (time) and 'y' (value) columns

2. **Process Launch**
   - Starts `predictor_ai.py` as separate subprocess
   - Passes input file path as command-line argument
   - Runs in background (doesn't freeze GUI)

3. **Monitoring**
   - Application periodically checks if process finished
   - Checks for output file `predict_out.csv`
   - Reads predictions when ready

4. **Display**
   - Predictions shown as purple line on dynamic plot
   - Updates automatically when new predictions arrive
   - Used for anomaly detection

**Why Separate Process?**
- Prediction can take time (especially training)
- Separate process = GUI stays responsive
- Can run on different CPU core
- Can be killed if needed

**Process Management:**
- Tracks process ID (PID)
- Checks if still running
- Handles errors gracefully
- Logs status messages

---

### 6. Anomaly Detection Integration

**How It's Integrated:**

1. **Initialization**
   - Creates `AnomalyDetector` instance on startup
   - Default threshold multiplier: 2.5
   - Minimum samples: 10

2. **Detection Trigger**
   - Called when new predictions arrive
   - Compares real-time actual data with predictions
   - Uses interpolation for accurate timestamp matching

3. **Visualization**
   - Anomalies shown as red vertical lines
   - Appears on both static and dynamic plots
   - Updates automatically when detected

4. **User Control**
   - Threshold multiplier adjustable via spinbox
   - Changes apply immediately
   - Re-runs detection with new threshold

**Data Flow:**
```
New predictions arrive
→ _update_predictions_data() called
→ _detect_anomalies() called
→ AnomalyDetector compares actual vs predicted
→ Anomalies identified
→ Vertical lines drawn on plots
→ Log messages displayed
```

---

## Data Flow: Complete Picture

### Real-Time Data Flow

```
USGS API
    ↓
SessionDataManager (background thread)
    ↓
api_df / api_df_new (global variables)
    ↓
_update_xydata() (every 20 seconds)
    ↓
new_x_t, new_y_mag_t (lists)
    ↓
_update_canvas() (every 200ms)
    ↓
Dynamic Plot (green line)
```

### Prediction Flow

```
Real-time data collected
    ↓
_save_data() → predict_input.csv
    ↓
start_prediction_process() → subprocess
    ↓
predictor_ai.py runs (separate process)
    ↓
predict_out.csv created
    ↓
_update_predictions_data() checks for file
    ↓
predict_x_t, predict_y_t (lists)
    ↓
_update_canvas() → Dynamic Plot (purple line)
```

### Anomaly Detection Flow

```
Predictions available
    ↓
_detect_anomalies() called
    ↓
AnomalyDetector.detect_anomalies()
    ↓
anomaly_times, anomaly_values (lists)
    ↓
_update_canvas() → Red vertical lines
```

---

## Key Methods Explained

### `_update_xydata()`

**Purpose:** Fetch and process new magnetic field data

**Process:**
1. Determine start time (last data point or current time)
2. If not forced: Start background thread to fetch data
3. If forced: Process data from `api_df_new`
4. Append new data to `new_x_t` and `new_y_mag_t`
5. Call `_update_predictions_data()`

**Called by:**
- Data timer (every 20 seconds)
- When data thread finishes

---

### `_update_canvas()`

**Purpose:** Update the plots with latest data

**Process:**
1. Check if new data line exists, create if not (green line)
2. Update new data line with latest values
3. Check if prediction line exists, create if not (purple line)
4. Update prediction line with latest predictions
5. Update anomaly markers (red vertical lines)
6. Adjust axis limits if needed
7. Redraw canvas

**Called by:**
- Drawing timer (every 200ms)
- After anomaly detection
- After threshold change

**Performance:**
- Uses `draw_idle()` for smooth updates
- Only redraws when needed
- Efficient for real-time updates

---

### `_update_predictions_data()`

**Purpose:** Check for and load new predictions

**Process:**
1. Check if prediction process finished
2. If still running: return (wait)
3. If finished with error: log error, reset
4. If finished successfully: read output file
5. Check if predictions are new (not duplicates)
6. Add to `predict_x_t` and `predict_y_t`
7. Call `_detect_anomalies()`

**File Checking:**
- Checks for `predict_out.csv` in session folder
- Validates file exists and is readable
- Handles missing file gracefully

---

### `_detect_anomalies()`

**Purpose:** Detect anomalies by comparing actual vs predicted

**Process:**
1. Get real-time actual data (`new_x_t`, `new_y_mag_t`)
2. Get all predictions (flatten list of lists)
3. Call `AnomalyDetector.detect_anomalies()`
4. Store anomaly times and values
5. Log detailed statistics
6. Trigger canvas update

**Logging:**
- Logs number of anomalies found
- Logs threshold used
- Logs statistics (mean, std, etc.)
- Logs warnings if no matches found

---

### `start_prediction_process()`

**Purpose:** Launch prediction script as separate process

**Process:**
1. Get Python executable path
2. Get path to `predictor_ai.py`
3. Construct command: `python predictor_ai.py input_file`
4. Start subprocess with Popen
5. Store process reference
6. Log process ID

**Error Handling:**
- Catches exceptions
- Logs errors
- Doesn't crash application

---

### `load_plot_framework_2()`

**Purpose:** Set up time series plots (Tab 2)

**Process:**
1. Create time series widget
2. Create static canvas (historical plot)
3. Create dynamic canvas (real-time plot)
4. Load initial data from `api_df`
5. Plot initial data (blue dots/line)
6. Set up tooltips (hover information)
7. Create timers for updates
8. Save initial data for predictions

**Plot Setup:**
- Static plot: Shows all historical data
- Dynamic plot: Shows real-time updates
- Both plots share same data initially
- Dynamic plot updates, static doesn't

---

### `load_visualization_framework()`

**Purpose:** Set up 3D visualization (Tab 3)

**Process:**
1. Create VTK widget
2. Create renderer, render window, interactor
3. Set up camera controls
4. Set background color
5. Load axes (orientation marker)
6. Initialize renderer
7. Set up point picker (click to see coordinates)

**3D Features:**
- Rotate: Left mouse drag
- Zoom: Scroll wheel
- Pan: Right mouse drag
- Pick point: Left click to see coordinates

---

## Threading and Concurrency

### Why Threading?

**Problem:**
- Fetching data from API takes time (network delay)
- If done in main thread, GUI freezes
- User sees "not responding" message

**Solution:**
- Use QThread for background work
- Main thread stays responsive
- GUI updates smoothly

### Thread Safety

**Mutex (Mutual Exclusion):**
- Prevents multiple threads from accessing data simultaneously
- Only one thread can modify `api_df` at a time
- Prevents data corruption

**Signals and Slots:**
- Thread-safe communication between threads
- Worker thread emits signal when done
- Main thread receives signal and updates GUI

**Example:**
```
Main Thread          Worker Thread
     |                    |
     |--start thread-->   |
     |                    |--fetch data-->
     |                    |--lock mutex-->
     |                    |--update api_df-->
     |                    |--unlock mutex-->
     |                    |--emit signal-->
     |<--signal received--|
     |--update plots-->   |
```

---

## File Management

### Session System

**Session ID:**
- Unique identifier for each application run
- Generated using UUID (Universally Unique Identifier)
- Format: `1967f531-89a7-4f06-abaa-dc481d4f78cf`

**Session Folder:**
- Location: `src/sessions/{session_id}/`
- Stores:
  - `download_mag.json`: Raw API data
  - `predict_input.csv`: Input for predictions
  - `predict_out.csv`: Prediction results

**Why Sessions?**
- Keeps data organized
- Allows multiple runs without conflicts
- Easy to track and debug

### CSV File Handling

**Loading CSV:**
1. User clicks "Upload" in menu
2. File dialog opens
3. User selects CSV file
4. File read into pandas DataFrame
5. Empty columns removed
6. World object created
7. Added to DataSourceManager
8. Displayed in tree view
9. Visualized in 3D

**CSV Format Expected:**
- Columns: Longitude, Latitude, Altitude, Mag (at minimum)
- Can have other columns (ignored)
- Must be valid CSV format

---

## Error Handling

### Try-Except Blocks

**Purpose:** Prevent application crashes from errors

**Common Error Handling:**
- File not found: Show error message, continue
- API failure: Log error, retry later
- Process failure: Log error, reset state
- Data errors: Skip invalid rows, continue

### Logging System

**Log Levels:**
- **Info**: Normal operations
- **Debug**: Detailed information for debugging
- **Warning**: Something unusual but not critical
- **Error**: Something went wrong

**Log Display:**
- Shown in log panel (textEditLog)
- HTML formatted for readability
- Timestamp included
- Scrolls automatically

---

## User Interactions

### Menu Actions

**Upload:**
- Opens file dialog
- Loads CSV file
- Updates visualization

**Add Time Series:**
- Creates new time series source
- Adds to tree view
- (Currently minimal implementation)

### Plot Interactions

**Tooltips:**
- Hover over data points to see values
- Shows timestamp and magnetic field value
- Works on both static and dynamic plots

**Navigation Toolbar:**
- Zoom, pan, reset view
- Save plot as image
- Standard matplotlib controls

### 3D Interactions

**Mouse Controls:**
- Left click: Pick point (see coordinates)
- Left drag: Rotate view
- Right click: Clear picked point
- Scroll: Zoom in/out
- Right drag: Pan view

---

## Performance Considerations

### Update Frequencies

**Data Fetching:**
- Every 20 seconds (configurable)
- Balance between freshness and API load

**Plot Updates:**
- Every 200ms (5 Hz)
- Smooth enough for real-time feel
- Not too fast to overload GUI

**Prediction Updates:**
- Asynchronously (when process finishes)
- Can take seconds to minutes
- Doesn't block GUI

### Memory Management

**Data Limits:**
- Error history: Last 1000 points (sliding window)
- Plot data: All points (can grow large)
- Predictions: All predictions (can grow large)

**Cleanup:**
- Old threads deleted when finished
- Old processes cleaned up
- Old actors removed from 3D scene

---

## Configuration and Customization

### Anomaly Detection

**Threshold Multiplier:**
- Adjustable via spinbox
- Range: 0.1 to 10.0
- Default: 2.5
- Real-time updates

### Plot Settings

**Currently:**
- Fixed update frequencies
- Fixed colors (blue, green, purple, red)
- Fixed line styles

**Future Customization:**
- User-configurable refresh rates
- Color themes
- Plot styles

---

## Summary

The Magnavis Application is a comprehensive GUI system that:

1. **Collects** real-time magnetic field data from USGS API
2. **Visualizes** data in multiple ways (plots, 3D, maps)
3. **Predicts** future values using AI (LSTM)
4. **Detects** anomalies by comparing actual vs predicted
5. **Manages** data sources and sessions
6. **Updates** displays in real-time without freezing

It's built with:
- **PyQt5**: For GUI framework
- **Matplotlib**: For 2D plots
- **VTK**: For 3D visualization
- **Pandas**: For data handling
- **Threading**: For responsive GUI
- **Subprocess**: For prediction process

The application is designed to be:
- **Responsive**: Never freezes (uses threads)
- **Real-time**: Updates automatically
- **User-friendly**: Intuitive interface
- **Robust**: Handles errors gracefully
- **Extensible**: Easy to add features

---

## Technical Details (For Developers)

### Dependencies
- PyQt5: GUI framework
- Matplotlib: Plotting
- VTK: 3D visualization
- Pandas: Data manipulation
- NumPy: Numerical operations
- GeoPandas: Geographic data
- pygeomag: Magnetic field calculations

### Architecture
- **MVC-like pattern**: Model (data), View (GUI), Controller (Application)
- **Signal-Slot pattern**: Qt's event system
- **Observer pattern**: Timers and callbacks
- **Factory pattern**: Data source creation

### File Structure
- Main file: `application.py` (~1820 lines)
- UI files: `ui_files/*.ui` (Qt Designer files)
- Data: `sessions/{session_id}/*`
- Resources: Images, shapefiles, etc.

### Performance
- GUI thread: Main thread (must stay responsive)
- Worker threads: Data fetching
- Subprocess: Prediction (separate Python process)
- Memory: Moderate (depends on data size)

---

*Document created: 2025*
*Last updated: After thorough code review*

