# Line-by-line documentation: `src/Anomaly_detector.py`

- **File**: `src/Anomaly_detector.py`
- **Lines covered in this document**: 1–366 (of 366)

## Structure overview (file-level)

- Line 14: `from datetime import datetime, timedelta`
- Line 17: class `AnomalyDetector`
- Line 27: method `__init__(self, threshold_multiplier=2.5, min_samples_for_threshold=10, freeze_duration_minutes=15)`
- Line 65: method `calculate_differences()`
- Line 286: method `detect_anomalies()`
- Line 332: method `get_statistics()`

## Implementation notes (current)

- **Freeze window**: The detector supports an optional "freeze" after the first anomaly: for `freeze_duration_minutes` (default 15), the threshold is held at the value that triggered the first anomaly (`frozen_threshold` / `freeze_until`). During this window, error history and threshold are not updated; after the window expires, normal adaptive behavior resumes.
- **`__init__`** now accepts `freeze_duration_minutes=15` and sets `self.freeze_until`, `self.frozen_threshold` (used internally in `calculate_differences`).

## Line-by-line explanation

| Line | Code | Explanation |
|---:|---|---|
| 1 | `'''` | Begins a triple-quoted string (often used as a module/class/function docstring). |
| 2 | `Anomaly Detection Module for Magnetic Field Data` | Executes Python statement: `Anomaly Detection Module for Magnetic Field Data`. |
| 3 | `Compares LSTM predictions with actual real-time data to detect anomalies` | Executes statement related to: LSTM / TensorFlow model code: `Compares LSTM predictions with actual real-time data to detect anomalies`. |
| 4 | `` | Blank line used to separate logical sections and improve readability. |
| 5 | `This module implements a statistical anomaly detection system that:` | Executes Python statement: `This module implements a statistical anomaly detection system that:`. |
| 6 | `1. Compares LSTM model predictions with actual magnetic field measurements` | Executes statement related to: LSTM / TensorFlow model code: `1. Compares LSTM model predictions with actual magnetic field measurements`. |
| 7 | `2. Calculates the difference (error) between predicted and actual values` | Executes Python statement: `2. Calculates the difference (error) between predicted and actual values`. |
| 8 | `3. Uses statistical methods to determine a dynamic threshold` | Executes Python statement: `3. Uses statistical methods to determine a dynamic threshold`. |
| 9 | `4. Flags data points where the error exceeds the threshold as anomalies` | Executes Python statement: `4. Flags data points where the error exceeds the threshold as anomalies`. |
| 10 | `'''` | Ends a triple-quoted string (closing a docstring/multiline literal). |
| 11 | `` | Blank line used to separate logical sections and improve readability. |
| 12 | `import numpy as np  # For numerical operations (mean, std, etc.)` | Imports module(s) into this namespace: `numpy as np  # For numerical operations (mean, std, etc.)`. |
| 13 | `import pandas as pd  # For data manipulation and time-series operations` | Imports module(s) into this namespace: `pandas as pd  # For data manipulation and time-series operations`. |
| 14 | `` | Blank line used to separate logical sections and improve readability. |
| 15 | `` | Blank line used to separate logical sections and improve readability. |
| 16 | `class AnomalyDetector:` | Defines class `AnomalyDetector` (starts a new type/namespace for related behavior). (scope: class AnomalyDetector) |
| 17 | `    """` | Ends a triple-quoted string (closing a docstring/multiline literal). (scope: class AnomalyDetector) |
| 18 | `    Detects anomalies by comparing predicted magnetic field values with actual measurements.` | Executes Python statement: `Detects anomalies by comparing predicted magnetic field values with actual measurements.`. (scope: class AnomalyDetector) |
| 19 | `    An anomaly is flagged when the difference exceeds a dynamically calculated threshold.` | Executes Python statement: `An anomaly is flagged when the difference exceeds a dynamically calculated threshold.`. (scope: class AnomalyDetector) |
| 20 | `    ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector) |
| 21 | `    How it works:` | Executes Python statement: `How it works:`. (scope: class AnomalyDetector) |
| 22 | `    - The detector learns from historical prediction errors to set an adaptive threshold` | Executes Python statement: `- The detector learns from historical prediction errors to set an adaptive threshold`. (scope: class AnomalyDetector) |
| 23 | `    - Uses statistical methods (mean + multiple of standard deviation) to identify outliers` | Executes Python statement: `- Uses statistical methods (mean + multiple of standard deviation) to identify outliers`. (scope: class AnomalyDetector) |
| 24 | `    - Adapts to changing conditions by keeping only recent error history` | Executes Python statement: `- Adapts to changing conditions by keeping only recent error history`. (scope: class AnomalyDetector) |
| 25 | `    """` | Ends a triple-quoted string (closing a docstring/multiline literal). (scope: class AnomalyDetector) |
| 26 | `    ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector) |
| 27 | `    def __init__(self, threshold_multiplier=2.5, min_samples_for_threshold=10, freeze_duration_minutes=15):` | Defines function/method `__init__` (entry point for reusable logic). Accepts optional `freeze_duration_minutes`; sets `freeze_until` and `frozen_threshold` for threshold freeze window. (scope: class AnomalyDetector → def __init__) |
| 28 | `        """` | Ends a triple-quoted string (closing a docstring/multiline literal). (scope: class AnomalyDetector → def __init__) |
| 29 | `        Initialize the anomaly detector with configuration parameters.` | Executes Python statement: `Initialize the anomaly detector with configuration parameters.`. (scope: class AnomalyDetector → def __init__) |
| 30 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def __init__) |
| 31 | `        Parameters:` | Executes Python statement: `Parameters:`. (scope: class AnomalyDetector → def __init__) |
| 32 | `        -----------` | Executes Python statement: `-----------`. (scope: class AnomalyDetector → def __init__) |
| 33 | `        threshold_multiplier : float` | Executes Python statement: `threshold_multiplier : float`. (scope: class AnomalyDetector → def __init__) |
| 34 | `            Multiplier for standard deviation to set threshold (default: 2.5)` | Executes Python statement: `Multiplier for standard deviation to set threshold (default: 2.5)`. (scope: class AnomalyDetector → def __init__) |
| 35 | `            This means anomalies are detected when error > mean_error + 2.5 * std_error` | Executes Python statement: `This means anomalies are detected when error > mean_error + 2.5 * std_error`. (scope: class AnomalyDetector → def __init__) |
| 36 | `            Higher values = fewer anomalies detected (more strict)` | Assigns/updates `Higher values` with the expression on the right-hand side. (scope: class AnomalyDetector → def __init__) |
| 37 | `            Lower values = more anomalies detected (more sensitive)` | Assigns/updates `Lower values` with the expression on the right-hand side. (scope: class AnomalyDetector → def __init__) |
| 38 | `        min_samples_for_threshold : int` | Executes Python statement: `min_samples_for_threshold : int`. (scope: class AnomalyDetector → def __init__) |
| 39 | `            Minimum number of samples needed to calculate a meaningful threshold` | Executes Python statement: `Minimum number of samples needed to calculate a meaningful threshold`. (scope: class AnomalyDetector → def __init__) |
| 40 | `            Before reaching this number, a default threshold is used` | Executes Python statement: `Before reaching this number, a default threshold is used`. (scope: class AnomalyDetector → def __init__) |
| 41 | `        """` | Ends a triple-quoted string (closing a docstring/multiline literal). (scope: class AnomalyDetector → def __init__) |
| 42 | `        # Store the multiplier used for threshold calculation` | Comment explaining intent/context: Store the multiplier used for threshold calculation (scope: class AnomalyDetector → def __init__) |
| 43 | `        # This determines how many standard deviations away from mean is considered anomalous` | Comment explaining intent/context: This determines how many standard deviations away from mean is considered anomalous (scope: class AnomalyDetector → def __init__) |
| 44 | `        self.threshold_multiplier = threshold_multiplier` | Assigns/updates `self.threshold_multiplier` with the expression on the right-hand side. (scope: class AnomalyDetector → def __init__) |
| 45 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def __init__) |
| 46 | `        # Minimum number of error samples needed before we can calculate statistics` | Comment explaining intent/context: Minimum number of error samples needed before we can calculate statistics (scope: class AnomalyDetector → def __init__) |
| 47 | `        # Too few samples would give unreliable statistics` | Comment explaining intent/context: Too few samples would give unreliable statistics (scope: class AnomalyDetector → def __init__) |
| 48 | `        self.min_samples_for_threshold = min_samples_for_threshold` | Assigns/updates `self.min_samples_for_threshold` with the expression on the right-hand side. (scope: class AnomalyDetector → def __init__) |
| 49 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def __init__) |
| 50 | `        # List to store all historical prediction errors (absolute differences)` | Comment explaining intent/context: List to store all historical prediction errors (absolute differences) (scope: class AnomalyDetector → def __init__) |
| 51 | `        # This grows as we compare more predictions with actual data` | Comment explaining intent/context: This grows as we compare more predictions with actual data (scope: class AnomalyDetector → def __init__) |
| 52 | `        # Used to calculate mean and standard deviation for threshold` | Comment explaining intent/context: Used to calculate mean and standard deviation for threshold (scope: class AnomalyDetector → def __init__) |
| 53 | `        self.prediction_errors = []` | Assigns/updates `self.prediction_errors` with the expression on the right-hand side. (scope: class AnomalyDetector → def __init__) |
| 54 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def __init__) |
| 55 | `        # The calculated threshold value (in nanoTesla, nT)` | Comment explaining intent/context: The calculated threshold value (in nanoTesla, nT) (scope: class AnomalyDetector → def __init__) |
| 56 | `        # Any prediction error above this value is considered an anomaly` | Comment explaining intent/context: Any prediction error above this value is considered an anomaly (scope: class AnomalyDetector → def __init__) |
| 57 | `        # Initially None, will be calculated after enough samples are collected` | Comment explaining intent/context: Initially None, will be calculated after enough samples are collected (scope: class AnomalyDetector → def __init__) |
| 58 | `        self.anomaly_threshold = None` | Assigns/updates `self.anomaly_threshold` with the expression on the right-hand side. (scope: class AnomalyDetector → def __init__) |
| 59 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def __init__) |
| 60 | `    def calculate_differences(self, actual_times, actual_values, predicted_times, predicted_values):` | Defines function/method `calculate_differences` (entry point for reusable logic). (scope: class AnomalyDetector → def calculate_differences) |
| 61 | `        """` | Ends a triple-quoted string (closing a docstring/multiline literal). (scope: class AnomalyDetector → def calculate_differences) |
| 62 | `        Calculate differences between actual and predicted values for overlapping time periods.` | Executes Python statement: `Calculate differences between actual and predicted values for overlapping time periods.`. (scope: class AnomalyDetector → def calculate_differences) |
| 63 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 64 | `        This is the core method that:` | Executes Python statement: `This is the core method that:`. (scope: class AnomalyDetector → def calculate_differences) |
| 65 | `        1. Interpolates predicted values at exact actual timestamps (more accurate than nearest-neighbor)` | Executes Python statement: `1. Interpolates predicted values at exact actual timestamps (more accurate than nearest-neighbor)`. (scope: class AnomalyDetector → def calculate_differences) |
| 66 | `        2. Calculates the error (difference) between actual and interpolated predicted values` | Executes Python statement: `2. Calculates the error (difference) between actual and interpolated predicted values`. (scope: class AnomalyDetector → def calculate_differences) |
| 67 | `        3. Updates the error history for threshold calculation` | Executes Python statement: `3. Updates the error history for threshold calculation`. (scope: class AnomalyDetector → def calculate_differences) |
| 68 | `        4. Calculates/updates the anomaly threshold` | Executes Python statement: `4. Calculates/updates the anomaly threshold`. (scope: class AnomalyDetector → def calculate_differences) |
| 69 | `        5. Marks which points are anomalies` | Executes Python statement: `5. Marks which points are anomalies`. (scope: class AnomalyDetector → def calculate_differences) |
| 70 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 71 | `        Timestamp Matching Strategy:` | Executes Python statement: `Timestamp Matching Strategy:`. (scope: class AnomalyDetector → def calculate_differences) |
| 72 | `        ----------------------------` | Executes Python statement: `----------------------------`. (scope: class AnomalyDetector → def calculate_differences) |
| 73 | `        Uses linear interpolation to find predicted values at exact actual timestamps.` | Executes Python statement: `Uses linear interpolation to find predicted values at exact actual timestamps.`. (scope: class AnomalyDetector → def calculate_differences) |
| 74 | `        This approach is more accurate than nearest-neighbor matching because:` | Executes Python statement: `This approach is more accurate than nearest-neighbor matching because:`. (scope: class AnomalyDetector → def calculate_differences) |
| 75 | `        - Eliminates timing errors (exact time alignment)` | Executes Python statement: `- Eliminates timing errors (exact time alignment)`. (scope: class AnomalyDetector → def calculate_differences) |
| 76 | `        - Uses information from neighboring prediction points` | Executes Python statement: `- Uses information from neighboring prediction points`. (scope: class AnomalyDetector → def calculate_differences) |
| 77 | `        - Handles different sampling rates gracefully` | Executes Python statement: `- Handles different sampling rates gracefully`. (scope: class AnomalyDetector → def calculate_differences) |
| 78 | `        - Provides smoother, more accurate comparisons` | Executes Python statement: `- Provides smoother, more accurate comparisons`. (scope: class AnomalyDetector → def calculate_differences) |
| 79 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 80 | `        Only interpolates within the actual prediction time range (no edge tolerance).` | Executes Python statement: `Only interpolates within the actual prediction time range (no edge tolerance).`. (scope: class AnomalyDetector → def calculate_differences) |
| 81 | `        Points outside this range or where interpolation isn't possible are excluded from comparison.` | Executes Python statement: `Points outside this range or where interpolation isn't possible are excluded from comparison.`. (scope: class AnomalyDetector → def calculate_differences) |
| 82 | `        This prevents "fake" predictions from being created using nearest neighbor fallback.` | Executes Python statement: `This prevents "fake" predictions from being created using nearest neighbor fallback.`. (scope: class AnomalyDetector → def calculate_differences) |
| 83 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 84 | `        Parameters:` | Executes Python statement: `Parameters:`. (scope: class AnomalyDetector → def calculate_differences) |
| 85 | `        -----------` | Executes Python statement: `-----------`. (scope: class AnomalyDetector → def calculate_differences) |
| 86 | `        actual_times : list` | Executes Python statement: `actual_times : list`. (scope: class AnomalyDetector → def calculate_differences) |
| 87 | `            List of datetime objects for actual measurements (from sensors or API)` | Executes Python statement: `List of datetime objects for actual measurements (from sensors or API)`. (scope: class AnomalyDetector → def calculate_differences) |
| 88 | `        actual_values : list` | Executes Python statement: `actual_values : list`. (scope: class AnomalyDetector → def calculate_differences) |
| 89 | `            List of actual magnetic field values (in nanoTesla, nT)` | Executes Python statement: `List of actual magnetic field values (in nanoTesla, nT)`. (scope: class AnomalyDetector → def calculate_differences) |
| 90 | `        predicted_times : list` | Executes Python statement: `predicted_times : list`. (scope: class AnomalyDetector → def calculate_differences) |
| 91 | `            List of datetime objects for predictions (from LSTM model)` | Executes statement related to: LSTM / TensorFlow model code: `List of datetime objects for predictions (from LSTM model)`. (scope: class AnomalyDetector → def calculate_differences) |
| 92 | `        predicted_values : list` | Executes Python statement: `predicted_values : list`. (scope: class AnomalyDetector → def calculate_differences) |
| 93 | `            List of predicted magnetic field values (in nanoTesla, nT)` | Executes Python statement: `List of predicted magnetic field values (in nanoTesla, nT)`. (scope: class AnomalyDetector → def calculate_differences) |
| 94 | `            ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 95 | `        Returns:` | Executes Python statement: `Returns:`. (scope: class AnomalyDetector → def calculate_differences) |
| 96 | `        --------` | Executes Python statement: `--------`. (scope: class AnomalyDetector → def calculate_differences) |
| 97 | `        differences_df : pandas.DataFrame` | Executes Python statement: `differences_df : pandas.DataFrame`. (scope: class AnomalyDetector → def calculate_differences) |
| 98 | `            DataFrame with columns: 'time', 'actual', 'predicted', 'difference', 'is_anomaly'` | Executes Python statement: `DataFrame with columns: 'time', 'actual', 'predicted', 'difference', 'is_anomaly'`. (scope: class AnomalyDetector → def calculate_differences) |
| 99 | `            Each row represents a matched pair of actual and interpolated predicted values` | Executes Python statement: `Each row represents a matched pair of actual and interpolated predicted values`. (scope: class AnomalyDetector → def calculate_differences) |
| 100 | `            'predicted' column contains interpolated values at exact actual timestamps` | Executes Python statement: `'predicted' column contains interpolated values at exact actual timestamps`. (scope: class AnomalyDetector → def calculate_differences) |
| 101 | `        """` | Ends a triple-quoted string (closing a docstring/multiline literal). (scope: class AnomalyDetector → def calculate_differences) |
| 102 | `        # Step 1: Check if we have data to compare` | Comment explaining intent/context: Step 1: Check if we have data to compare (scope: class AnomalyDetector → def calculate_differences) |
| 103 | `        # If either list is empty, return an empty DataFrame` | Comment explaining intent/context: If either list is empty, return an empty DataFrame (scope: class AnomalyDetector → def calculate_differences) |
| 104 | `        if not actual_times or not predicted_times:` | Starts an `if` block: conditional control flow based on `not actual_times or not predicted_times`. (scope: class AnomalyDetector → def calculate_differences) |
| 105 | `            return pd.DataFrame(columns=['time', 'actual', 'predicted', 'difference', 'is_anomaly'])` | Returns value(s) from the current function: `pd.DataFrame(columns=['time', 'actual', 'predicted', 'difference', 'is_anomaly'])`. (scope: class AnomalyDetector → def calculate_differences) |
| 106 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 107 | `        # Step 2: Convert lists to pandas DataFrames for easier manipulation` | Comment explaining intent/context: Step 2: Convert lists to pandas DataFrames for easier manipulation (scope: class AnomalyDetector → def calculate_differences) |
| 108 | `        # This allows us to use pandas' powerful time-series operations` | Comment explaining intent/context: This allows us to use pandas' powerful time-series operations (scope: class AnomalyDetector → def calculate_differences) |
| 109 | `        actual_df = pd.DataFrame({` | Assigns/updates `actual_df` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 110 | `            'time': pd.to_datetime(actual_times),  # Ensure times are datetime objects` | Executes Python statement: `'time': pd.to_datetime(actual_times),  # Ensure times are datetime objects`. (scope: class AnomalyDetector → def calculate_differences) |
| 111 | `            'actual': actual_values  # Actual magnetic field measurements` | Executes Python statement: `'actual': actual_values  # Actual magnetic field measurements`. (scope: class AnomalyDetector → def calculate_differences) |
| 112 | `        })` | Executes Python statement: `})`. (scope: class AnomalyDetector → def calculate_differences) |
| 113 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 114 | `        predicted_df = pd.DataFrame({` | Assigns/updates `predicted_df` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 115 | `            'time': pd.to_datetime(predicted_times),  # Ensure times are datetime objects` | Executes Python statement: `'time': pd.to_datetime(predicted_times),  # Ensure times are datetime objects`. (scope: class AnomalyDetector → def calculate_differences) |
| 116 | `            'predicted': predicted_values  # Predicted magnetic field values from LSTM` | Executes statement related to: LSTM / TensorFlow model code: `'predicted': predicted_values  # Predicted magnetic field values from LSTM`. (scope: class AnomalyDetector → def calculate_differences) |
| 117 | `        })` | Executes Python statement: `})`. (scope: class AnomalyDetector → def calculate_differences) |
| 118 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 119 | `        # Step 3: Match actual and predicted data by timestamp using interpolation` | Comment explaining intent/context: Step 3: Match actual and predicted data by timestamp using interpolation (scope: class AnomalyDetector → def calculate_differences) |
| 120 | `        # This approach interpolates predicted values at exact actual timestamps,` | Comment explaining intent/context: This approach interpolates predicted values at exact actual timestamps, (scope: class AnomalyDetector → def calculate_differences) |
| 121 | `        # providing more accurate comparisons than nearest-neighbor matching.` | Comment explaining intent/context: providing more accurate comparisons than nearest-neighbor matching. (scope: class AnomalyDetector → def calculate_differences) |
| 122 | `        # ` | Comment line used as a visual separator. (scope: class AnomalyDetector → def calculate_differences) |
| 123 | `        # Advantages of interpolation:` | Comment explaining intent/context: Advantages of interpolation: (scope: class AnomalyDetector → def calculate_differences) |
| 124 | `        # - Exact time alignment (no timing errors)` | Comment explaining intent/context: - Exact time alignment (no timing errors) (scope: class AnomalyDetector → def calculate_differences) |
| 125 | `        # - Uses information from neighboring prediction points` | Comment explaining intent/context: - Uses information from neighboring prediction points (scope: class AnomalyDetector → def calculate_differences) |
| 126 | `        # - Handles different sampling rates gracefully` | Comment explaining intent/context: - Handles different sampling rates gracefully (scope: class AnomalyDetector → def calculate_differences) |
| 127 | `        # - More accurate for anomaly detection` | Comment explaining intent/context: - More accurate for anomaly detection (scope: class AnomalyDetector → def calculate_differences) |
| 128 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 129 | `        # Sort both dataframes by time (required for interpolation)` | Comment explaining intent/context: Sort both dataframes by time (required for interpolation) (scope: class AnomalyDetector → def calculate_differences) |
| 130 | `        actual_sorted = actual_df.sort_values('time').reset_index(drop=True)` | Assigns/updates `actual_sorted` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 131 | `        predicted_sorted = predicted_df.sort_values('time').reset_index(drop=True)` | Assigns/updates `predicted_sorted` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 132 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 133 | `        # Set time as index for interpolation` | Comment explaining intent/context: Set time as index for interpolation (scope: class AnomalyDetector → def calculate_differences) |
| 134 | `        predicted_indexed = predicted_sorted.set_index('time')` | Assigns/updates `predicted_indexed` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 135 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 136 | `        # Define interpolation bounds: only interpolate within the actual prediction time range` | Comment explaining intent/context: Define interpolation bounds: only interpolate within the actual prediction time range (scope: class AnomalyDetector → def calculate_differences) |
| 137 | `        # FIXED: Removed 15-minute tolerance to prevent "fake" predictions at edges` | Comment explaining intent/context: FIXED: Removed 15-minute tolerance to prevent "fake" predictions at edges (scope: class AnomalyDetector → def calculate_differences) |
| 138 | `        # Only compare where interpolation is actually possible (between prediction points)` | Comment explaining intent/context: Only compare where interpolation is actually possible (between prediction points) (scope: class AnomalyDetector → def calculate_differences) |
| 139 | `        pred_min_time = predicted_indexed.index.min()` | Assigns/updates `pred_min_time` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 140 | `        pred_max_time = predicted_indexed.index.max()` | Assigns/updates `pred_max_time` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 141 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 142 | `        # Filter actual times to only those within the actual prediction range` | Comment explaining intent/context: Filter actual times to only those within the actual prediction range (scope: class AnomalyDetector → def calculate_differences) |
| 143 | `        # This ensures we only compare where real predictions exist (no edge extrapolation)` | Comment explaining intent/context: This ensures we only compare where real predictions exist (no edge extrapolation) (scope: class AnomalyDetector → def calculate_differences) |
| 144 | `        actual_within_range = actual_sorted[` | Assigns/updates `actual_within_range` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 145 | `            (actual_sorted['time'] >= pred_min_time) &` | Executes Python statement: `(actual_sorted['time'] >= pred_min_time) &`. (scope: class AnomalyDetector → def calculate_differences) |
| 146 | `            (actual_sorted['time'] <= pred_max_time)` | Executes Python statement: `(actual_sorted['time'] <= pred_max_time)`. (scope: class AnomalyDetector → def calculate_differences) |
| 147 | `        ].copy()` | Executes Python statement: `].copy()`. (scope: class AnomalyDetector → def calculate_differences) |
| 148 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 149 | `        if len(actual_within_range) == 0:` | Starts an `if` block: conditional control flow based on `len(actual_within_range) == 0`. (scope: class AnomalyDetector → def calculate_differences) |
| 150 | `            # No actual data points within prediction range` | Comment explaining intent/context: No actual data points within prediction range (scope: class AnomalyDetector → def calculate_differences) |
| 151 | `            return pd.DataFrame(columns=['time', 'actual', 'predicted', 'difference', 'is_anomaly'])` | Returns value(s) from the current function: `pd.DataFrame(columns=['time', 'actual', 'predicted', 'difference', 'is_anomaly'])`. (scope: class AnomalyDetector → def calculate_differences) |
| 152 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 153 | `        # Interpolate predicted values at actual timestamps using pandas interpolation` | Comment explaining intent/context: Interpolate predicted values at actual timestamps using pandas interpolation (scope: class AnomalyDetector → def calculate_differences) |
| 154 | `        # Set actual times as index for reindexing` | Comment explaining intent/context: Set actual times as index for reindexing (scope: class AnomalyDetector → def calculate_differences) |
| 155 | `        actual_indexed = actual_within_range.set_index('time')` | Assigns/updates `actual_indexed` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 156 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 157 | `        # Reindex predicted data to actual timestamps and interpolate` | Comment explaining intent/context: Reindex predicted data to actual timestamps and interpolate (scope: class AnomalyDetector → def calculate_differences) |
| 158 | `        # 'linear' interpolation works well for smoothly varying magnetic fields` | Comment explaining intent/context: 'linear' interpolation works well for smoothly varying magnetic fields (scope: class AnomalyDetector → def calculate_differences) |
| 159 | `        # FIXED: Removed nearest neighbor fallback - only use interpolation` | Comment explaining intent/context: FIXED: Removed nearest neighbor fallback - only use interpolation (scope: class AnomalyDetector → def calculate_differences) |
| 160 | `        predicted_interpolated = predicted_indexed.reindex(` | Assigns/updates `predicted_interpolated` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 161 | `            actual_indexed.index` | Executes Python statement: `actual_indexed.index`. (scope: class AnomalyDetector → def calculate_differences) |
| 162 | `        ).interpolate(method='time')  # Time-based linear interpolation` | Executes Python statement: `).interpolate(method='time')  # Time-based linear interpolation`. (scope: class AnomalyDetector → def calculate_differences) |
| 163 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 164 | `        # FIXED: Removed nearest neighbor fallback that was creating "fake" predictions` | Comment explaining intent/context: FIXED: Removed nearest neighbor fallback that was creating "fake" predictions (scope: class AnomalyDetector → def calculate_differences) |
| 165 | `        # Now we only use interpolation, which requires at least 2 prediction points` | Comment explaining intent/context: Now we only use interpolation, which requires at least 2 prediction points (scope: class AnomalyDetector → def calculate_differences) |
| 166 | `        # Points at the very edges (where interpolation isn't possible) will remain NaN` | Comment explaining intent/context: Points at the very edges (where interpolation isn't possible) will remain NaN (scope: class AnomalyDetector → def calculate_differences) |
| 167 | `        # and be filtered out below` | Comment explaining intent/context: and be filtered out below (scope: class AnomalyDetector → def calculate_differences) |
| 168 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 169 | `        # Combine actual and interpolated predicted values` | Comment explaining intent/context: Combine actual and interpolated predicted values (scope: class AnomalyDetector → def calculate_differences) |
| 170 | `        merged = pd.DataFrame({` | Assigns/updates `merged` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 171 | `            'time': actual_indexed.index,` | Executes Python statement: `'time': actual_indexed.index,`. (scope: class AnomalyDetector → def calculate_differences) |
| 172 | `            'actual': actual_indexed['actual'].values,` | Executes Python statement: `'actual': actual_indexed['actual'].values,`. (scope: class AnomalyDetector → def calculate_differences) |
| 173 | `            'predicted': predicted_interpolated['predicted'].values` | Executes Python statement: `'predicted': predicted_interpolated['predicted'].values`. (scope: class AnomalyDetector → def calculate_differences) |
| 174 | `        }).reset_index(drop=True)` | Executes Python statement: `}).reset_index(drop=True)`. (scope: class AnomalyDetector → def calculate_differences) |
| 175 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 176 | `        # Remove any rows where prediction is still NaN (outside interpolation range or too far)` | Comment explaining intent/context: Remove any rows where prediction is still NaN (outside interpolation range or too far) (scope: class AnomalyDetector → def calculate_differences) |
| 177 | `        # This filters out points where interpolation wasn't possible (too far from prediction data)` | Comment explaining intent/context: This filters out points where interpolation wasn't possible (too far from prediction data) (scope: class AnomalyDetector → def calculate_differences) |
| 178 | `        merged = merged.dropna(subset=['predicted'])` | Assigns/updates `merged` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 179 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 180 | `        # Ensure time column is datetime (should already be, but safety check)` | Comment explaining intent/context: Ensure time column is datetime (should already be, but safety check) (scope: class AnomalyDetector → def calculate_differences) |
| 181 | `        if len(merged) > 0:` | Starts an `if` block: conditional control flow based on `len(merged) > 0`. (scope: class AnomalyDetector → def calculate_differences) |
| 182 | `            merged['time'] = pd.to_datetime(merged['time'])` | Assigns/updates `merged['time']` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 183 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 184 | `        # Step 4: Calculate the error (difference) between actual and predicted values` | Comment explaining intent/context: Step 4: Calculate the error (difference) between actual and predicted values (scope: class AnomalyDetector → def calculate_differences) |
| 185 | `        # Positive difference means actual > predicted (model underestimated)` | Comment explaining intent/context: Positive difference means actual > predicted (model underestimated) (scope: class AnomalyDetector → def calculate_differences) |
| 186 | `        # Negative difference means actual < predicted (model overestimated)` | Comment explaining intent/context: Negative difference means actual < predicted (model overestimated) (scope: class AnomalyDetector → def calculate_differences) |
| 187 | `        merged['difference'] = merged['actual'] - merged['predicted']` | Assigns/updates `merged['difference']` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 188 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 189 | `        # We care about the magnitude of error, not the direction` | Comment explaining intent/context: We care about the magnitude of error, not the direction (scope: class AnomalyDetector → def calculate_differences) |
| 190 | `        # So we take the absolute value` | Comment explaining intent/context: So we take the absolute value (scope: class AnomalyDetector → def calculate_differences) |
| 191 | `        merged['abs_difference'] = abs(merged['difference'])` | Assigns/updates `merged['abs_difference']` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 192 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 193 | `        # Step 5: Update our error history for threshold calculation` | Comment explaining intent/context: Step 5: Update our error history for threshold calculation (scope: class AnomalyDetector → def calculate_differences) |
| 194 | `        # We keep track of all prediction errors to learn what's "normal"` | Comment explaining intent/context: We keep track of all prediction errors to learn what's "normal" (scope: class AnomalyDetector → def calculate_differences) |
| 195 | `        if len(merged) > 0:` | Starts an `if` block: conditional control flow based on `len(merged) > 0`. (scope: class AnomalyDetector → def calculate_differences) |
| 196 | `            # Add all new errors to our history` | Comment explaining intent/context: Add all new errors to our history (scope: class AnomalyDetector → def calculate_differences) |
| 197 | `            self.prediction_errors.extend(merged['abs_difference'].tolist())` | Executes Python statement: `self.prediction_errors.extend(merged['abs_difference'].tolist())`. (scope: class AnomalyDetector → def calculate_differences) |
| 198 | `            ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 199 | `            # Keep only recent errors (last 1000 points) to adapt to changing conditions` | Comment explaining intent/context: Keep only recent errors (last 1000 points) to adapt to changing conditions (scope: class AnomalyDetector → def calculate_differences) |
| 200 | `            # This is important because:` | Comment explaining intent/context: This is important because: (scope: class AnomalyDetector → def calculate_differences) |
| 201 | `            # - Magnetic field behavior might change over time` | Comment explaining intent/context: - Magnetic field behavior might change over time (scope: class AnomalyDetector → def calculate_differences) |
| 202 | `            # - The LSTM model might improve as it learns` | Comment explaining intent/context: - The LSTM model might improve as it learns (scope: class AnomalyDetector → def calculate_differences) |
| 203 | `            # - We want the threshold to reflect recent performance, not old performance` | Comment explaining intent/context: - We want the threshold to reflect recent performance, not old performance (scope: class AnomalyDetector → def calculate_differences) |
| 204 | `            if len(self.prediction_errors) > 1000:` | Starts an `if` block: conditional control flow based on `len(self.prediction_errors) > 1000`. (scope: class AnomalyDetector → def calculate_differences) |
| 205 | `                # Keep only the most recent 1000 errors` | Comment explaining intent/context: Keep only the most recent 1000 errors (scope: class AnomalyDetector → def calculate_differences) |
| 206 | `                self.prediction_errors = self.prediction_errors[-1000:]` | Assigns/updates `self.prediction_errors` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 207 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 208 | `        # Step 6: Calculate the anomaly threshold` | Comment explaining intent/context: Step 6: Calculate the anomaly threshold (scope: class AnomalyDetector → def calculate_differences) |
| 209 | `        # The threshold determines what error value is considered "too large" (anomalous)` | Comment explaining intent/context: The threshold determines what error value is considered "too large" (anomalous) (scope: class AnomalyDetector → def calculate_differences) |
| 210 | `        # IMPORTANT: This uses self.threshold_multiplier which can be changed dynamically` | Comment explaining intent/context: IMPORTANT: This uses self.threshold_multiplier which can be changed dynamically (scope: class AnomalyDetector → def calculate_differences) |
| 211 | `        if len(self.prediction_errors) >= self.min_samples_for_threshold:` | Starts an `if` block: conditional control flow based on `len(self.prediction_errors) >= self.min_samples_for_threshold`. (scope: class AnomalyDetector → def calculate_differences) |
| 212 | `            # We have enough samples to calculate meaningful statistics` | Comment explaining intent/context: We have enough samples to calculate meaningful statistics (scope: class AnomalyDetector → def calculate_differences) |
| 213 | `            ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 214 | `            # Calculate the mean (average) of all prediction errors` | Comment explaining intent/context: Calculate the mean (average) of all prediction errors (scope: class AnomalyDetector → def calculate_differences) |
| 215 | `            # This tells us the typical error size` | Comment explaining intent/context: This tells us the typical error size (scope: class AnomalyDetector → def calculate_differences) |
| 216 | `            mean_error = np.mean(self.prediction_errors)` | Assigns/updates `mean_error` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 217 | `            ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 218 | `            # Calculate the standard deviation of prediction errors` | Comment explaining intent/context: Calculate the standard deviation of prediction errors (scope: class AnomalyDetector → def calculate_differences) |
| 219 | `            # This tells us how much the errors vary from the mean` | Comment explaining intent/context: This tells us how much the errors vary from the mean (scope: class AnomalyDetector → def calculate_differences) |
| 220 | `            # Large std = errors are very variable` | Comment explaining intent/context: Large std = errors are very variable (scope: class AnomalyDetector → def calculate_differences) |
| 221 | `            # Small std = errors are consistent` | Comment explaining intent/context: Small std = errors are consistent (scope: class AnomalyDetector → def calculate_differences) |
| 222 | `            std_error = np.std(self.prediction_errors)` | Assigns/updates `std_error` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 223 | `            ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 224 | `            # Calculate threshold using statistical method:` | Comment explaining intent/context: Calculate threshold using statistical method: (scope: class AnomalyDetector → def calculate_differences) |
| 225 | `            # Threshold = Mean Error + (Multiplier × Standard Deviation)` | Comment explaining intent/context: Threshold = Mean Error + (Multiplier × Standard Deviation) (scope: class AnomalyDetector → def calculate_differences) |
| 226 | `            # This is based on the statistical principle that most data falls within` | Comment explaining intent/context: This is based on the statistical principle that most data falls within (scope: class AnomalyDetector → def calculate_differences) |
| 227 | `            # mean ± (multiplier × std). Values beyond this are outliers (anomalies).` | Comment explaining intent/context: mean ± (multiplier × std). Values beyond this are outliers (anomalies). (scope: class AnomalyDetector → def calculate_differences) |
| 228 | `            # The multiplier can be changed dynamically by the user` | Comment explaining intent/context: The multiplier can be changed dynamically by the user (scope: class AnomalyDetector → def calculate_differences) |
| 229 | `            self.anomaly_threshold = mean_error + self.threshold_multiplier * std_error` | Assigns/updates `self.anomaly_threshold` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 230 | `            ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 231 | `            # Debug: Log threshold calculation (will be logged by caller)` | Comment explaining intent/context: Debug: Log threshold calculation (will be logged by caller) (scope: class AnomalyDetector → def calculate_differences) |
| 232 | `        else:` | Starts an `else` branch for the preceding conditional. (scope: class AnomalyDetector → def calculate_differences) |
| 233 | `            # We don't have enough samples yet, so use a default threshold` | Comment explaining intent/context: We don't have enough samples yet, so use a default threshold (scope: class AnomalyDetector → def calculate_differences) |
| 234 | `            # This is a conservative estimate based on typical magnetic field variations` | Comment explaining intent/context: This is a conservative estimate based on typical magnetic field variations (scope: class AnomalyDetector → def calculate_differences) |
| 235 | `            # Typical magnetic field variations are in the range of 10-50 nT` | Comment explaining intent/context: Typical magnetic field variations are in the range of 10-50 nT (scope: class AnomalyDetector → def calculate_differences) |
| 236 | `            # We use a lower default (10 nT) to be more sensitive when we don't have enough data` | Comment explaining intent/context: We use a lower default (10 nT) to be more sensitive when we don't have enough data (scope: class AnomalyDetector → def calculate_differences) |
| 237 | `            # NOTE: This default doesn't use the multiplier, but once we have enough samples,` | Comment explaining intent/context: NOTE: This default doesn't use the multiplier, but once we have enough samples, (scope: class AnomalyDetector → def calculate_differences) |
| 238 | `            # the threshold will be recalculated using the multiplier` | Comment explaining intent/context: the threshold will be recalculated using the multiplier (scope: class AnomalyDetector → def calculate_differences) |
| 239 | `            # If we have some errors but not enough, use a scaled version` | Comment explaining intent/context: If we have some errors but not enough, use a scaled version (scope: class AnomalyDetector → def calculate_differences) |
| 240 | `            if len(self.prediction_errors) > 0:` | Starts an `if` block: conditional control flow based on `len(self.prediction_errors) > 0`. (scope: class AnomalyDetector → def calculate_differences) |
| 241 | `                # Use mean of available errors with a small multiplier as a temporary threshold` | Comment explaining intent/context: Use mean of available errors with a small multiplier as a temporary threshold (scope: class AnomalyDetector → def calculate_differences) |
| 242 | `                temp_mean = np.mean(self.prediction_errors)` | Assigns/updates `temp_mean` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 243 | `                temp_std = np.std(self.prediction_errors) if len(self.prediction_errors) > 1 else temp_mean * 0.1` | Assigns/updates `temp_std` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 244 | `                self.anomaly_threshold = temp_mean + self.threshold_multiplier * temp_std` | Assigns/updates `self.anomaly_threshold` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 245 | `            else:` | Starts an `else` branch for the preceding conditional. (scope: class AnomalyDetector → def calculate_differences) |
| 246 | `                # No errors yet, use a very low default to catch any differences` | Comment explaining intent/context: No errors yet, use a very low default to catch any differences (scope: class AnomalyDetector → def calculate_differences) |
| 247 | `                self.anomaly_threshold = 10.0  # Default threshold in nanoTesla (nT)` | Assigns/updates `self.anomaly_threshold` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 248 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 249 | `        # Step 7: Mark which data points are anomalies` | Comment explaining intent/context: Step 7: Mark which data points are anomalies (scope: class AnomalyDetector → def calculate_differences) |
| 250 | `        # A point is an anomaly if its absolute error exceeds the threshold` | Comment explaining intent/context: A point is an anomaly if its absolute error exceeds the threshold (scope: class AnomalyDetector → def calculate_differences) |
| 251 | `        # IMPORTANT: This uses the CURRENT threshold which should reflect the current multiplier` | Comment explaining intent/context: IMPORTANT: This uses the CURRENT threshold which should reflect the current multiplier (scope: class AnomalyDetector → def calculate_differences) |
| 252 | `        if self.anomaly_threshold is not None:` | Starts an `if` block: conditional control flow based on `self.anomaly_threshold is not None`. (scope: class AnomalyDetector → def calculate_differences) |
| 253 | `            merged['is_anomaly'] = merged['abs_difference'] > self.anomaly_threshold` | Assigns/updates `merged['is_anomaly']` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 254 | `        else:` | Starts an `else` branch for the preceding conditional. (scope: class AnomalyDetector → def calculate_differences) |
| 255 | `            # If threshold is None, mark all as non-anomalies (shouldn't happen)` | Comment explaining intent/context: If threshold is None, mark all as non-anomalies (shouldn't happen) (scope: class AnomalyDetector → def calculate_differences) |
| 256 | `            merged['is_anomaly'] = False` | Assigns/updates `merged['is_anomaly']` with the expression on the right-hand side. (scope: class AnomalyDetector → def calculate_differences) |
| 257 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def calculate_differences) |
| 258 | `        # Return only the columns we need, excluding 'abs_difference' (internal use only)` | Comment explaining intent/context: Return only the columns we need, excluding 'abs_difference' (internal use only) (scope: class AnomalyDetector → def calculate_differences) |
| 259 | `        return merged[['time', 'actual', 'predicted', 'difference', 'is_anomaly']]` | Returns value(s) from the current function: `merged[['time', 'actual', 'predicted', 'difference', 'is_anomaly']]`. (scope: class AnomalyDetector → def calculate_differences) |
| 260 | `    ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector) |
| 261 | `    def detect_anomalies(self, actual_times, actual_values, predicted_times, predicted_values):` | Defines function/method `detect_anomalies` (entry point for reusable logic). (scope: class AnomalyDetector → def detect_anomalies) |
| 262 | `        """` | Ends a triple-quoted string (closing a docstring/multiline literal). (scope: class AnomalyDetector → def detect_anomalies) |
| 263 | `        Main method to detect anomalies - this is the primary interface for the detector.` | Executes Python statement: `Main method to detect anomalies - this is the primary interface for the detector.`. (scope: class AnomalyDetector → def detect_anomalies) |
| 264 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def detect_anomalies) |
| 265 | `        This method:` | Executes Python statement: `This method:`. (scope: class AnomalyDetector → def detect_anomalies) |
| 266 | `        1. Calls calculate_differences to compare actual vs predicted data` | Executes Python statement: `1. Calls calculate_differences to compare actual vs predicted data`. (scope: class AnomalyDetector → def detect_anomalies) |
| 267 | `        2. Filters to keep only the points marked as anomalies` | Executes Python statement: `2. Filters to keep only the points marked as anomalies`. (scope: class AnomalyDetector → def detect_anomalies) |
| 268 | `        3. Returns the anomalies along with the threshold used` | Executes Python statement: `3. Returns the anomalies along with the threshold used`. (scope: class AnomalyDetector → def detect_anomalies) |
| 269 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def detect_anomalies) |
| 270 | `        Parameters:` | Executes Python statement: `Parameters:`. (scope: class AnomalyDetector → def detect_anomalies) |
| 271 | `        -----------` | Executes Python statement: `-----------`. (scope: class AnomalyDetector → def detect_anomalies) |
| 272 | `        actual_times : list` | Executes Python statement: `actual_times : list`. (scope: class AnomalyDetector → def detect_anomalies) |
| 273 | `            List of datetime objects for actual measurements` | Executes Python statement: `List of datetime objects for actual measurements`. (scope: class AnomalyDetector → def detect_anomalies) |
| 274 | `        actual_values : list` | Executes Python statement: `actual_values : list`. (scope: class AnomalyDetector → def detect_anomalies) |
| 275 | `            List of actual magnetic field values` | Executes Python statement: `List of actual magnetic field values`. (scope: class AnomalyDetector → def detect_anomalies) |
| 276 | `        predicted_times : list` | Executes Python statement: `predicted_times : list`. (scope: class AnomalyDetector → def detect_anomalies) |
| 277 | `            List of datetime objects for predictions` | Executes Python statement: `List of datetime objects for predictions`. (scope: class AnomalyDetector → def detect_anomalies) |
| 278 | `        predicted_values : list` | Executes Python statement: `predicted_values : list`. (scope: class AnomalyDetector → def detect_anomalies) |
| 279 | `            List of predicted magnetic field values` | Executes Python statement: `List of predicted magnetic field values`. (scope: class AnomalyDetector → def detect_anomalies) |
| 280 | `            ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def detect_anomalies) |
| 281 | `        Returns:` | Executes Python statement: `Returns:`. (scope: class AnomalyDetector → def detect_anomalies) |
| 282 | `        --------` | Executes Python statement: `--------`. (scope: class AnomalyDetector → def detect_anomalies) |
| 283 | `        anomalies_df : pandas.DataFrame` | Executes Python statement: `anomalies_df : pandas.DataFrame`. (scope: class AnomalyDetector → def detect_anomalies) |
| 284 | `            DataFrame containing ONLY the anomaly points (filtered from all comparisons)` | Executes Python statement: `DataFrame containing ONLY the anomaly points (filtered from all comparisons)`. (scope: class AnomalyDetector → def detect_anomalies) |
| 285 | `            Columns: 'time', 'actual', 'predicted', 'difference'` | Executes Python statement: `Columns: 'time', 'actual', 'predicted', 'difference'`. (scope: class AnomalyDetector → def detect_anomalies) |
| 286 | `            Each row is a detected anomaly` | Executes Python statement: `Each row is a detected anomaly`. (scope: class AnomalyDetector → def detect_anomalies) |
| 287 | `        threshold : float` | Executes Python statement: `threshold : float`. (scope: class AnomalyDetector → def detect_anomalies) |
| 288 | `            The threshold value (in nT) that was used for detection` | Executes Python statement: `The threshold value (in nT) that was used for detection`. (scope: class AnomalyDetector → def detect_anomalies) |
| 289 | `            Useful for logging and understanding detection sensitivity` | Executes Python statement: `Useful for logging and understanding detection sensitivity`. (scope: class AnomalyDetector → def detect_anomalies) |
| 290 | `        """` | Ends a triple-quoted string (closing a docstring/multiline literal). (scope: class AnomalyDetector → def detect_anomalies) |
| 291 | `        # Step 1: Calculate differences and identify anomalies` | Comment explaining intent/context: Step 1: Calculate differences and identify anomalies (scope: class AnomalyDetector → def detect_anomalies) |
| 292 | `        # This does all the heavy lifting: matching, error calculation, threshold calculation` | Comment explaining intent/context: This does all the heavy lifting: matching, error calculation, threshold calculation (scope: class AnomalyDetector → def detect_anomalies) |
| 293 | `        differences_df = self.calculate_differences(` | Assigns/updates `differences_df` with the expression on the right-hand side. (scope: class AnomalyDetector → def detect_anomalies) |
| 294 | `            actual_times, actual_values, predicted_times, predicted_values` | Executes Python statement: `actual_times, actual_values, predicted_times, predicted_values`. (scope: class AnomalyDetector → def detect_anomalies) |
| 295 | `        )` | Executes Python statement: `)`. (scope: class AnomalyDetector → def detect_anomalies) |
| 296 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def detect_anomalies) |
| 297 | `        # Step 2: Filter to keep only the anomalies` | Comment explaining intent/context: Step 2: Filter to keep only the anomalies (scope: class AnomalyDetector → def detect_anomalies) |
| 298 | `        # The 'is_anomaly' column is True for anomalies, False for normal points` | Comment explaining intent/context: The 'is_anomaly' column is True for anomalies, False for normal points (scope: class AnomalyDetector → def detect_anomalies) |
| 299 | `        # We use boolean indexing to filter: differences_df[differences_df['is_anomaly']]` | Comment explaining intent/context: We use boolean indexing to filter: differences_df[differences_df['is_anomaly']] (scope: class AnomalyDetector → def detect_anomalies) |
| 300 | `        # .copy() creates a new DataFrame so we don't modify the original` | Comment explaining intent/context: .copy() creates a new DataFrame so we don't modify the original (scope: class AnomalyDetector → def detect_anomalies) |
| 301 | `        anomalies_df = differences_df[differences_df['is_anomaly']].copy()` | Assigns/updates `anomalies_df` with the expression on the right-hand side. (scope: class AnomalyDetector → def detect_anomalies) |
| 302 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def detect_anomalies) |
| 303 | `        # Return both the anomalies and the threshold used` | Comment explaining intent/context: Return both the anomalies and the threshold used (scope: class AnomalyDetector → def detect_anomalies) |
| 304 | `        # The threshold is useful for logging and understanding detection sensitivity` | Comment explaining intent/context: The threshold is useful for logging and understanding detection sensitivity (scope: class AnomalyDetector → def detect_anomalies) |
| 305 | `        return anomalies_df, self.anomaly_threshold` | Returns value(s) from the current function: `anomalies_df, self.anomaly_threshold`. (scope: class AnomalyDetector → def detect_anomalies) |
| 306 | `    ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector) |
| 307 | `    def get_statistics(self):` | Defines function/method `get_statistics` (entry point for reusable logic). (scope: class AnomalyDetector → def get_statistics) |
| 308 | `        """` | Ends a triple-quoted string (closing a docstring/multiline literal). (scope: class AnomalyDetector → def get_statistics) |
| 309 | `        Get statistics about prediction errors and the current threshold.` | Executes Python statement: `Get statistics about prediction errors and the current threshold.`. (scope: class AnomalyDetector → def get_statistics) |
| 310 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def get_statistics) |
| 311 | `        This is useful for:` | Executes Python statement: `This is useful for:`. (scope: class AnomalyDetector → def get_statistics) |
| 312 | `        - Monitoring the detector's performance` | Executes Python statement: `- Monitoring the detector's performance`. (scope: class AnomalyDetector → def get_statistics) |
| 313 | `        - Understanding how the threshold is being calculated` | Executes Python statement: `- Understanding how the threshold is being calculated`. (scope: class AnomalyDetector → def get_statistics) |
| 314 | `        - Debugging and tuning the detector` | Executes Python statement: `- Debugging and tuning the detector`. (scope: class AnomalyDetector → def get_statistics) |
| 315 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def get_statistics) |
| 316 | `        Returns:` | Executes Python statement: `Returns:`. (scope: class AnomalyDetector → def get_statistics) |
| 317 | `        --------` | Executes Python statement: `--------`. (scope: class AnomalyDetector → def get_statistics) |
| 318 | `        dict : Dictionary with statistics containing:` | Executes Python statement: `dict : Dictionary with statistics containing:`. (scope: class AnomalyDetector → def get_statistics) |
| 319 | `            - 'mean_error': Average prediction error (in nT)` | Executes Python statement: `- 'mean_error': Average prediction error (in nT)`. (scope: class AnomalyDetector → def get_statistics) |
| 320 | `            - 'std_error': Standard deviation of prediction errors (in nT)` | Executes Python statement: `- 'std_error': Standard deviation of prediction errors (in nT)`. (scope: class AnomalyDetector → def get_statistics) |
| 321 | `            - 'threshold': Current anomaly threshold (in nT)` | Executes Python statement: `- 'threshold': Current anomaly threshold (in nT)`. (scope: class AnomalyDetector → def get_statistics) |
| 322 | `            - 'total_samples': Number of error samples used for statistics` | Executes Python statement: `- 'total_samples': Number of error samples used for statistics`. (scope: class AnomalyDetector → def get_statistics) |
| 323 | `        """` | Ends a triple-quoted string (closing a docstring/multiline literal). (scope: class AnomalyDetector → def get_statistics) |
| 324 | `        # If we haven't collected any errors yet, return zeros` | Comment explaining intent/context: If we haven't collected any errors yet, return zeros (scope: class AnomalyDetector → def get_statistics) |
| 325 | `        if not self.prediction_errors:` | Starts an `if` block: conditional control flow based on `not self.prediction_errors`. (scope: class AnomalyDetector → def get_statistics) |
| 326 | `            return {` | Returns value(s) from the current function: `{`. (scope: class AnomalyDetector → def get_statistics) |
| 327 | `                'mean_error': 0,  # No errors yet, so mean is 0` | Executes Python statement: `'mean_error': 0,  # No errors yet, so mean is 0`. (scope: class AnomalyDetector → def get_statistics) |
| 328 | `                'std_error': 0,  # No errors yet, so std is 0` | Executes Python statement: `'std_error': 0,  # No errors yet, so std is 0`. (scope: class AnomalyDetector → def get_statistics) |
| 329 | `                'threshold': self.anomaly_threshold or 0,  # Use default threshold if set, else 0` | Executes Python statement: `'threshold': self.anomaly_threshold or 0,  # Use default threshold if set, else 0`. (scope: class AnomalyDetector → def get_statistics) |
| 330 | `                'total_samples': 0  # No samples collected yet` | Executes Python statement: `'total_samples': 0  # No samples collected yet`. (scope: class AnomalyDetector → def get_statistics) |
| 331 | `            }` | Executes Python statement: `}`. (scope: class AnomalyDetector → def get_statistics) |
| 332 | `        ` | Blank line used to separate logical sections and improve readability. (scope: class AnomalyDetector → def get_statistics) |
| 333 | `        # Calculate and return statistics from collected error history` | Comment explaining intent/context: Calculate and return statistics from collected error history (scope: class AnomalyDetector → def get_statistics) |
| 334 | `        return {` | Returns value(s) from the current function: `{`. (scope: class AnomalyDetector → def get_statistics) |
| 335 | `            'mean_error': np.mean(self.prediction_errors),  # Average error size` | Executes Python statement: `'mean_error': np.mean(self.prediction_errors),  # Average error size`. (scope: class AnomalyDetector → def get_statistics) |
| 336 | `            'std_error': np.std(self.prediction_errors),  # How much errors vary` | Executes Python statement: `'std_error': np.std(self.prediction_errors),  # How much errors vary`. (scope: class AnomalyDetector → def get_statistics) |
| 337 | `            'threshold': self.anomaly_threshold or 0,  # Current threshold value` | Executes Python statement: `'threshold': self.anomaly_threshold or 0,  # Current threshold value`. (scope: class AnomalyDetector → def get_statistics) |
| 338 | `            'total_samples': len(self.prediction_errors)  # How many errors we've seen` | Executes Python statement: `'total_samples': len(self.prediction_errors)  # How many errors we've seen`. (scope: class AnomalyDetector → def get_statistics) |
| 339 | `        }` | Executes Python statement: `}`. (scope: class AnomalyDetector → def get_statistics) |
