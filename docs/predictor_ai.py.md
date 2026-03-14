# Line-by-line documentation: `src/predictor_ai.py`

- **File**: `src/predictor_ai.py`
- **Lines covered in this document**: 1–285 (approximately)

## Structure overview (file-level)

- Line 11: function `_parse_train_window_minutes()`
- Line 22: class `GRUPredictor` (renamed from `LSTMPredictor`)
- Line 23: method `__init__()`
- Line 37: method `create_windowed_dataset()`
- Line 52: method `build_model()` (uses **GRU**, not LSTM, as recurrent layer)
- Line 62: method `save_model()` (NEW: saves model weights and scaler)
- Line 73: method `load_model()` (NEW: loads pre-trained model and scaler)
- Line 84: method `_compute_time_features()`
- Line 116: method `forecast()` (updated: supports `pretrained_model_path` parameter)
- Line 247: `if __name__ == "__main__":` (updated: supports pre-trained models via `PRETRAINED_MODEL_PATH` env var)

## Implementation notes (current)

- **Recurrent layer**: The model uses `GRU(32, ...)` from `tensorflow.keras.layers` (import: `Dense, GRU`), not LSTM. The class has been renamed to `GRUPredictor` to accurately reflect the architecture.
- **Pre-trained models**: The model can now be saved and loaded for faster startup and better accuracy. See `train_gru_pretrained.py` for pre-training on extended historical data.
- **Model persistence**: `save_model()` saves both the model architecture/weights and the scaler state (for consistent feature scaling). `load_model()` restores both.
- **Per-sensor models**: Each sensor can have its own pre-trained model (see `train_gru_pretrained.py` for training multiple models).

## Line-by-line explanation

| Line | Code | Explanation |
|---:|---|---|
| 1 | `import os, sys` | Imports module(s) into this namespace: `os, sys`. |
| 2 | `import numpy as np` | Imports module(s) into this namespace: `numpy as np`. |
| 3 | `import pandas as pd` | Imports module(s) into this namespace: `pandas as pd`. |
| 4 | `import tensorflow as tf` | Imports module(s) into this namespace: `tensorflow as tf`. |
| 5 | `from tensorflow.keras.models import Sequential` | Imports specific symbols using a `from … import …` statement: `from tensorflow.keras.models import Sequential`. |
| 6 | `from tensorflow.keras.layers import Dense, GRU` | Imports specific symbols using a `from … import …` statement: `from tensorflow.keras.layers import Dense, GRU`. (Implementation uses GRU as the recurrent layer.) |
| 7 | `from tensorflow.keras.optimizers import Adam` | Imports specific symbols using a `from … import …` statement: `from tensorflow.keras.optimizers import Adam`. |
| 8 | `from sklearn.preprocessing import MinMaxScaler` | Imports specific symbols using a `from … import …` statement: `from sklearn.preprocessing import MinMaxScaler`. |
| 9 | `from datetime import datetime, timedelta` | Imports specific symbols using a `from … import …` statement: `from datetime import datetime, timedelta`. |
| 10 | `` | Blank line used to separate logical sections and improve readability. |
| 11 | `def _parse_train_window_minutes():` | Defines function/method `_parse_train_window_minutes` (entry point for reusable logic). (scope: def _parse_train_window_minutes) |
| 12 | `    """Read optional training window (minutes) from env TRAIN_WINDOW_MINUTES."""` | Single-line triple-quoted string (docstring or multiline literal expressed on one line). (scope: def _parse_train_window_minutes) |
| 13 | `    try:` | Starts a `try` block to catch and handle exceptions. (scope: def _parse_train_window_minutes) |
| 14 | `        val = os.environ.get("TRAIN_WINDOW_MINUTES", None)` | Assigns/updates `val` with the expression on the right-hand side. (scope: def _parse_train_window_minutes) |
| 15 | `        if val is None or val == "":` | Starts an `if` block: conditional control flow based on `val is None or val == ""`. (scope: def _parse_train_window_minutes) |
| 16 | `            return None` | Returns value(s) from the current function: `None`. (scope: def _parse_train_window_minutes) |
| 17 | `        return float(val)` | Returns value(s) from the current function: `float(val)`. (scope: def _parse_train_window_minutes) |
| 18 | `    except Exception:` | Starts an `except` handler: `except Exception:`. (scope: def _parse_train_window_minutes) |
| 19 | `        return None` | Returns value(s) from the current function: `None`. (scope: def _parse_train_window_minutes) |
| 20 | `` | Blank line used to separate logical sections and improve readability. |
| 21 | `` | Blank line used to separate logical sections and improve readability. |
| 22 | `class LSTMPredictor:` | Defines class `LSTMPredictor` (starts a new type/namespace for related behavior). (scope: class LSTMPredictor) |
| 23 | `    def __init__(self, window_size=5, initial_train_points=3400,` | Defines function/method `__init__` (entry point for reusable logic). (scope: class LSTMPredictor → def __init__) |
| 24 | `                 epochs_per_update=5, learning_rate=0.001, update_training=True,` | Assigns/updates `epochs_per_update` with the expression on the right-hand side. (scope: class LSTMPredictor → def __init__) |
| 25 | `                 use_yearly_cycle=False, train_window_minutes=None):` | Assigns/updates `use_yearly_cycle` with the expression on the right-hand side. (scope: class LSTMPredictor → def __init__) |
| 26 | `` | Blank line used to separate logical sections and improve readability. |
| 27 | `        self.window_size = window_size` | Assigns/updates `self.window_size` with the expression on the right-hand side. |
| 28 | `        self.initial_train_points = initial_train_points` | Assigns/updates `self.initial_train_points` with the expression on the right-hand side. |
| 29 | `        self.epochs_per_update = epochs_per_update` | Assigns/updates `self.epochs_per_update` with the expression on the right-hand side. |
| 30 | `        self.learning_rate = learning_rate` | Assigns/updates `self.learning_rate` with the expression on the right-hand side. |
| 31 | `        self.update_training = update_training` | Assigns/updates `self.update_training` with the expression on the right-hand side. |
| 32 | `        self.scaler = MinMaxScaler(feature_range=(0, 1))` | Assigns/updates `self.scaler` with the expression on the right-hand side. Hint: Feature scaling for ML (sklearn MinMaxScaler). |
| 33 | `        self.model = None` | Assigns/updates `self.model` with the expression on the right-hand side. |
| 34 | `        self.use_yearly_cycle = use_yearly_cycle` | Assigns/updates `self.use_yearly_cycle` with the expression on the right-hand side. |
| 35 | `        self.train_window_minutes = train_window_minutes` | Assigns/updates `self.train_window_minutes` with the expression on the right-hand side. |
| 36 | `` | Blank line used to separate logical sections and improve readability. |
| 37 | `    def create_windowed_dataset(self, series):` | Defines function/method `create_windowed_dataset` (entry point for reusable logic). (scope: def create_windowed_dataset) |
| 38 | `` | Blank line used to separate logical sections and improve readability. |
| 39 | `        X, y = [], []` | Assigns/updates `X, y` with the expression on the right-hand side. |
| 40 | `        for i in range(len(series) - self.window_size):` | Starts a `for` loop: iterates as described by `i in range(len(series) - self.window_size)`. |
| 41 | `            X.append(series[i : i + self.window_size])` | Executes Python statement: `X.append(series[i : i + self.window_size])`. |
| 42 | `            # Target is the magnetic field (scaled) = column 0` | Comment explaining intent/context: Target is the magnetic field (scaled) = column 0 |
| 43 | `            y.append(series[i + self.window_size, 0])` | Executes Python statement: `y.append(series[i + self.window_size, 0])`. |
| 44 | `        return np.array(X), np.array(y)` | Returns value(s) from the current function: `np.array(X), np.array(y)`. |
| 45 | `` | Blank line used to separate logical sections and improve readability. |
| 46 | `    def build_model(self, feature_dim):` | Defines function/method `build_model` (entry point for reusable logic). (scope: def build_model) |
| 47 | `        model = Sequential()` | Assigns/updates `model` with the expression on the right-hand side. (scope: def build_model) |
| 48 | `        model.add(GRU(32, return_sequences=False, input_shape=(self.window_size, feature_dim)))` | Adds GRU layer (32 units); recurrent backbone is GRU, not LSTM. (scope: def build_model) |
| 49 | `        model.add(Dense(16, activation='relu'))` | Executes Python statement: `model.add(Dense(16, activation='relu'))`. (scope: def build_model) |
| 50 | `        model.add(Dense(1))` | Executes Python statement: `model.add(Dense(1))`. (scope: def build_model) |
| 51 | `        optimizer = Adam(learning_rate=self.learning_rate)` | Assigns/updates `optimizer` with the expression on the right-hand side. (scope: def build_model) |
| 52 | `        model.compile(optimizer=optimizer, loss='mean_squared_error')` | Executes Python statement: `model.compile(optimizer=optimizer, loss='mean_squared_error')`. (scope: def build_model) |
| 53 | `        self.model = model` | Assigns/updates `self.model` with the expression on the right-hand side. (scope: def build_model) |
| 54 | `` | Blank line used to separate logical sections and improve readability. |
| 55 | `    def _compute_time_features(self, ts_array):` | Defines function/method `_compute_time_features` (entry point for reusable logic). (scope: def _compute_time_features) |
| 56 | `        """Compute cyclic time features (daily, optionally yearly) for an array of pandas Timestamps."""` | Single-line triple-quoted string (docstring or multiline literal expressed on one line). (scope: def _compute_time_features) |
| 57 | `        seconds_in_day = 24 * 3600` | Assigns/updates `seconds_in_day` with the expression on the right-hand side. (scope: def _compute_time_features) |
| 58 | `        # Support both DatetimeIndex-like objects (have .hour/.minute/...) and pandas Series (use .dt accessor).` | Comment explaining intent/context: Support both DatetimeIndex-like objects (have .hour/.minute/...) and pandas Series (use .dt accessor). (scope: def _compute_time_features) |
| 59 | `        ts_array = pd.to_datetime(ts_array)` | Assigns/updates `ts_array` with the expression on the right-hand side. (scope: def _compute_time_features) |
| 60 | `        if isinstance(ts_array, pd.Series):` | Starts an `if` block: conditional control flow based on `isinstance(ts_array, pd.Series)`. (scope: def _compute_time_features) |
| 61 | `            hour = ts_array.dt.hour` | Assigns/updates `hour` with the expression on the right-hand side. (scope: def _compute_time_features) |
| 62 | `            minute = ts_array.dt.minute` | Assigns/updates `minute` with the expression on the right-hand side. (scope: def _compute_time_features) |
| 63 | `            second = ts_array.dt.second` | Assigns/updates `second` with the expression on the right-hand side. (scope: def _compute_time_features) |
| 64 | `            microsecond = ts_array.dt.microsecond` | Assigns/updates `microsecond` with the expression on the right-hand side. (scope: def _compute_time_features) |
| 65 | `            dayofyear = ts_array.dt.dayofyear` | Assigns/updates `dayofyear` with the expression on the right-hand side. (scope: def _compute_time_features) |
| 66 | `        else:` | Starts an `else` branch for the preceding conditional. (scope: def _compute_time_features) |
| 67 | `            hour = ts_array.hour` | Assigns/updates `hour` with the expression on the right-hand side. (scope: def _compute_time_features) |
| 68 | `            minute = ts_array.minute` | Assigns/updates `minute` with the expression on the right-hand side. (scope: def _compute_time_features) |
| 69 | `            second = ts_array.second` | Assigns/updates `second` with the expression on the right-hand side. (scope: def _compute_time_features) |
| 70 | `            microsecond = ts_array.microsecond` | Assigns/updates `microsecond` with the expression on the right-hand side. (scope: def _compute_time_features) |
| 71 | `            dayofyear = ts_array.dayofyear` | Assigns/updates `dayofyear` with the expression on the right-hand side. (scope: def _compute_time_features) |
| 72 | `` | Blank line used to separate logical sections and improve readability. |
| 73 | `        sec_of_day = hour * 3600 + minute * 60 + second + microsecond / 1e6` | Assigns/updates `sec_of_day` with the expression on the right-hand side. |
| 74 | `        day_angle = 2 * np.pi * (sec_of_day / seconds_in_day)` | Assigns/updates `day_angle` with the expression on the right-hand side. |
| 75 | `        sin_day = np.sin(day_angle)` | Assigns/updates `sin_day` with the expression on the right-hand side. |
| 76 | `        cos_day = np.cos(day_angle)` | Assigns/updates `cos_day` with the expression on the right-hand side. |
| 77 | `` | Blank line used to separate logical sections and improve readability. |
| 78 | `        if self.use_yearly_cycle:` | Starts an `if` block: conditional control flow based on `self.use_yearly_cycle`. |
| 79 | `            # Day-of-year fraction (including leap-year effect approximately)` | Comment explaining intent/context: Day-of-year fraction (including leap-year effect approximately) |
| 80 | `            year_angle = 2 * np.pi * (dayofyear / 365.25)` | Assigns/updates `year_angle` with the expression on the right-hand side. |
| 81 | `            sin_year = np.sin(year_angle)` | Assigns/updates `sin_year` with the expression on the right-hand side. |
| 82 | `            cos_year = np.cos(year_angle)` | Assigns/updates `cos_year` with the expression on the right-hand side. |
| 83 | `            return sin_day, cos_day, sin_year, cos_year` | Returns value(s) from the current function: `sin_day, cos_day, sin_year, cos_year`. |
| 84 | `        else:` | Starts an `else` branch for the preceding conditional. |
| 85 | `            return sin_day, cos_day` | Returns value(s) from the current function: `sin_day, cos_day`. |
| 86 | `` | Blank line used to separate logical sections and improve readability. |
| 87 | `    def forecast(self, timestamps, field_data, n_future):` | Defines function/method `forecast` (entry point for reusable logic). (scope: def forecast) |
| 88 | `` | Blank line used to separate logical sections and improve readability. |
| 89 | `        # Build a clean time series table first so timestamps and values stay aligned.` | Comment explaining intent/context: Build a clean time series table first so timestamps and values stay aligned. |
| 90 | `        ts = pd.to_datetime(timestamps)` | Assigns/updates `ts` with the expression on the right-hand side. |
| 91 | `        df0 = pd.DataFrame({"ts": ts, "field": field_data})` | Assigns/updates `df0` with the expression on the right-hand side. |
| 92 | `        df0["ts"] = pd.to_datetime(df0["ts"], errors="coerce")` | Assigns/updates `df0["ts"]` with the expression on the right-hand side. |
| 93 | `        df0["field"] = pd.to_numeric(df0["field"], errors="coerce")` | Assigns/updates `df0["field"]` with the expression on the right-hand side. |
| 94 | `        df0 = df0.dropna(subset=["ts", "field"]).sort_values("ts").reset_index(drop=True)` | Assigns/updates `df0` with the expression on the right-hand side. |
| 95 | `` | Blank line used to separate logical sections and improve readability. |
| 96 | `        # Defensive: if duplicate timestamps exist, average them so:` | Comment explaining intent/context: Defensive: if duplicate timestamps exist, average them so: |
| 97 | `        # - time_delta is non-zero` | Comment explaining intent/context: - time_delta is non-zero |
| 98 | `        # - feature arrays and target arrays have consistent lengths` | Comment explaining intent/context: - feature arrays and target arrays have consistent lengths |
| 99 | `        df0 = df0.groupby("ts", as_index=False)["field"].mean().sort_values("ts").reset_index(drop=True)` | Assigns/updates `df0` with the expression on the right-hand side. |
| 100 | `` | Blank line used to separate logical sections and improve readability. |
| 101 | `        ts = df0["ts"]  # pandas Series of datetime64` | Assigns/updates `ts` with the expression on the right-hand side. |
| 102 | `        field = df0["field"].to_numpy(dtype=float).reshape(-1, 1)` | Assigns/updates `field` with the expression on the right-hand side. |
| 103 | `` | Blank line used to separate logical sections and improve readability. |
| 104 | `        # Optional: restrict training data to the most recent N minutes` | Comment explaining intent/context: Optional: restrict training data to the most recent N minutes |
| 105 | `        if self.train_window_minutes:` | Starts an `if` block: conditional control flow based on `self.train_window_minutes`. |
| 106 | `            cutoff = ts.max().to_pydatetime() - timedelta(minutes=self.train_window_minutes)` | Assigns/updates `cutoff` with the expression on the right-hand side. |
| 107 | `            mask = ts >= cutoff` | Assigns/updates `mask` with the expression on the right-hand side. |
| 108 | `            if int(mask.sum()) > self.window_size + 1:` | Starts an `if` block: conditional control flow based on `int(mask.sum()) > self.window_size + 1`. |
| 109 | `                ts = ts[mask].reset_index(drop=True)` | Assigns/updates `ts` with the expression on the right-hand side. |
| 110 | `                field = field[mask.to_numpy()]` | Assigns/updates `field` with the expression on the right-hand side. |
| 111 | `` | Blank line used to separate logical sections and improve readability. |
| 112 | `        # Scale magnetic field only; sin/cos are already bounded in [-1, 1]` | Comment explaining intent/context: Scale magnetic field only; sin/cos are already bounded in [-1, 1] |
| 113 | `        field_scaled = self.scaler.fit_transform(field).flatten()` | Assigns/updates `field_scaled` with the expression on the right-hand side. |
| 114 | `` | Blank line used to separate logical sections and improve readability. |
| 115 | `        # Build feature matrix: [mag_scaled, sin_time, cos_time, (optional) sin_year, cos_year]` | Comment explaining intent/context: Build feature matrix: [mag_scaled, sin_time, cos_time, (optional) sin_year, cos_year] |
| 116 | `        time_feats = self._compute_time_features(ts)` | Assigns/updates `time_feats` with the expression on the right-hand side. |
| 117 | `        if self.use_yearly_cycle:` | Starts an `if` block: conditional control flow based on `self.use_yearly_cycle`. |
| 118 | `            sin_day, cos_day, sin_year, cos_year = time_feats` | Assigns/updates `sin_day, cos_day, sin_year, cos_year` with the expression on the right-hand side. |
| 119 | `            feature_matrix = np.column_stack([field_scaled, sin_day, cos_day, sin_year, cos_year])` | Assigns/updates `feature_matrix` with the expression on the right-hand side. |
| 120 | `        else:` | Starts an `else` branch for the preceding conditional. |
| 121 | `            sin_day, cos_day = time_feats` | Assigns/updates `sin_day, cos_day` with the expression on the right-hand side. |
| 122 | `            feature_matrix = np.column_stack([field_scaled, sin_day, cos_day])` | Assigns/updates `feature_matrix` with the expression on the right-hand side. |
| 123 | `` | Blank line used to separate logical sections and improve readability. |
| 124 | `        # Ensure initial_train_points does not exceed available data` | Comment explaining intent/context: Ensure initial_train_points does not exceed available data |
| 125 | `        self.initial_train_points = min(self.initial_train_points, len(field_scaled))` | Assigns/updates `self.initial_train_points` with the expression on the right-hand side. |
| 126 | `` | Blank line used to separate logical sections and improve readability. |
| 127 | `        if self.initial_train_points < self.window_size:` | Starts an `if` block: conditional control flow based on `self.initial_train_points < self.window_size`. |
| 128 | `            raise ValueError("initial_train_points must be >= window_size.")` | Raises an exception to signal an error condition: `raise ValueError("initial_train_points must be >= window_size.")`. |
| 129 | `        if self.initial_train_points > len(field_scaled):` | Starts an `if` block: conditional control flow based on `self.initial_train_points > len(field_scaled)`. |
| 130 | `            raise ValueError("initial_train_points cannot exceed total data length.")` | Raises an exception to signal an error condition: `raise ValueError("initial_train_points cannot exceed total data length.")`. |
| 131 | `` | Blank line used to separate logical sections and improve readability. |
| 132 | `        if self.model is None:` | Starts an `if` block: conditional control flow based on `self.model is None`. |
| 133 | `            self.build_model(feature_matrix.shape[1])` | Executes Python statement: `self.build_model(feature_matrix.shape[1])`. |
| 134 | `` | Blank line used to separate logical sections and improve readability. |
| 135 | `        initial_data = feature_matrix[:self.initial_train_points]` | Assigns/updates `initial_data` with the expression on the right-hand side. |
| 136 | `        X_init, y_init = self.create_windowed_dataset(initial_data)` | Assigns/updates `X_init, y_init` with the expression on the right-hand side. |
| 137 | `        if len(X_init) > 0:` | Starts an `if` block: conditional control flow based on `len(X_init) > 0`. |
| 138 | `            self.model.fit(X_init, y_init, epochs=self.epochs_per_update, verbose=0)` | Executes Python statement: `self.model.fit(X_init, y_init, epochs=self.epochs_per_update, verbose=0)`. |
| 139 | `        ` | Blank line used to separate logical sections and improve readability. |
| 140 | `        # up until here` | Comment explaining intent/context: up until here |
| 141 | `` | Blank line used to separate logical sections and improve readability. |
| 142 | `        # Compute base time delta for future steps` | Comment explaining intent/context: Compute base time delta for future steps |
| 143 | `        try:` | Starts a `try` block to catch and handle exceptions. |
| 144 | `            # IMPORTANT: keep these as Python datetime/timedelta (not numpy.datetime64),` | Comment explaining intent/context: IMPORTANT: keep these as Python datetime/timedelta (not numpy.datetime64), |
| 145 | `            # because later we access &#96;.hour/.minute/...&#96; when computing cyclic features.` | Comment explaining intent/context: because later we access `.hour/.minute/...` when computing cyclic features. |
| 146 | `            dt_last = ts.iloc[-1].to_pydatetime() if hasattr(ts, "iloc") else pd.to_datetime(ts[-1]).to_pydatetime()` | Assigns/updates `dt_last` with the expression on the right-hand side. |
| 147 | `` | Blank line used to separate logical sections and improve readability. |
| 148 | `            # Prefer a robust estimate of sampling period:` | Comment explaining intent/context: Prefer a robust estimate of sampling period: |
| 149 | `            # - compute diffs over the whole series` | Comment explaining intent/context: - compute diffs over the whole series |
| 150 | `            # - keep positive diffs` | Comment explaining intent/context: - keep positive diffs |
| 151 | `            # - use median as stable step` | Comment explaining intent/context: - use median as stable step |
| 152 | `            if len(ts) > 1:` | Starts an `if` block: conditional control flow based on `len(ts) > 1`. |
| 153 | `                diffs = pd.Series(ts).diff().dropna()` | Assigns/updates `diffs` with the expression on the right-hand side. |
| 154 | `                diffs = diffs[diffs > pd.Timedelta(0)]` | Assigns/updates `diffs` with the expression on the right-hand side. |
| 155 | `                if len(diffs) > 0:` | Starts an `if` block: conditional control flow based on `len(diffs) > 0`. |
| 156 | `                    time_delta = diffs.median().to_pytimedelta()` | Assigns/updates `time_delta` with the expression on the right-hand side. |
| 157 | `                else:` | Starts an `else` branch for the preceding conditional. |
| 158 | `                    time_delta = timedelta(seconds=1)` | Assigns/updates `time_delta` with the expression on the right-hand side. |
| 159 | `            else:` | Starts an `else` branch for the preceding conditional. |
| 160 | `                time_delta = timedelta(seconds=1)` | Assigns/updates `time_delta` with the expression on the right-hand side. |
| 161 | `` | Blank line used to separate logical sections and improve readability. |
| 162 | `            # Clamp: never allow 0 or negative.` | Comment explaining intent/context: Clamp: never allow 0 or negative. |
| 163 | `            if time_delta.total_seconds() <= 0:` | Starts an `if` block: conditional control flow based on `time_delta.total_seconds() <= 0`. |
| 164 | `                time_delta = timedelta(seconds=1)` | Assigns/updates `time_delta` with the expression on the right-hand side. |
| 165 | `        except Exception as e:` | Starts an `except` handler: `except Exception as e:`. |
| 166 | `            raise ValueError("Error parsing timestamps for time delta.") from e` | Raises an exception to signal an error condition: `raise ValueError("Error parsing timestamps for time delta.") from e`. |
| 167 | `` | Blank line used to separate logical sections and improve readability. |
| 168 | `        predictions = []` | Assigns/updates `predictions` with the expression on the right-hand side. |
| 169 | `        future_timestamps = []` | Assigns/updates `future_timestamps` with the expression on the right-hand side. |
| 170 | `        current_window = feature_matrix[self.initial_train_points - self.window_size : self.initial_train_points]` | Assigns/updates `current_window` with the expression on the right-hand side. |
| 171 | `` | Blank line used to separate logical sections and improve readability. |
| 172 | `        if self.update_training:` | Starts an `if` block: conditional control flow based on `self.update_training`. |
| 173 | `            training_data = initial_data.copy()` | Assigns/updates `training_data` with the expression on the right-hand side. |
| 174 | `` | Blank line used to separate logical sections and improve readability. |
| 175 | `        for i in range(n_future):` | Starts a `for` loop: iterates as described by `i in range(n_future)`. |
| 176 | `            current_window_reshaped = np.array([current_window])` | Assigns/updates `current_window_reshaped` with the expression on the right-hand side. |
| 177 | `            predicted_scaled = self.model.predict(current_window_reshaped, verbose=0)[0, 0]` | Assigns/updates `predicted_scaled` with the expression on the right-hand side. |
| 178 | `` | Blank line used to separate logical sections and improve readability. |
| 179 | `            # Inverse scale to get magnetic field value` | Comment explaining intent/context: Inverse scale to get magnetic field value |
| 180 | `            predicted_value = self.scaler.inverse_transform([[predicted_scaled]])[0, 0]` | Assigns/updates `predicted_value` with the expression on the right-hand side. |
| 181 | `            predictions.append(predicted_value)` | Executes Python statement: `predictions.append(predicted_value)`. |
| 182 | `` | Blank line used to separate logical sections and improve readability. |
| 183 | `            # Next timestamp and its cyclic features` | Comment explaining intent/context: Next timestamp and its cyclic features |
| 184 | `            new_time = dt_last + (i + 1) * time_delta` | Assigns/updates `new_time` with the expression on the right-hand side. |
| 185 | `            future_timestamps.append(new_time)` | Executes Python statement: `future_timestamps.append(new_time)`. |
| 186 | `` | Blank line used to separate logical sections and improve readability. |
| 187 | `            sec_of_day = new_time.hour * 3600 + new_time.minute * 60 + new_time.second + new_time.microsecond / 1e6` | Assigns/updates `sec_of_day` with the expression on the right-hand side. |
| 188 | `            day_angle = 2 * np.pi * (sec_of_day / (24 * 3600))` | Assigns/updates `day_angle` with the expression on the right-hand side. |
| 189 | `            sin_day_new = np.sin(day_angle)` | Assigns/updates `sin_day_new` with the expression on the right-hand side. |
| 190 | `            cos_day_new = np.cos(day_angle)` | Assigns/updates `cos_day_new` with the expression on the right-hand side. |
| 191 | `` | Blank line used to separate logical sections and improve readability. |
| 192 | `            if self.use_yearly_cycle:` | Starts an `if` block: conditional control flow based on `self.use_yearly_cycle`. |
| 193 | `                day_of_year = new_time.timetuple().tm_yday` | Assigns/updates `day_of_year` with the expression on the right-hand side. |
| 194 | `                year_angle = 2 * np.pi * (day_of_year / 365.25)` | Assigns/updates `year_angle` with the expression on the right-hand side. |
| 195 | `                sin_year_new = np.sin(year_angle)` | Assigns/updates `sin_year_new` with the expression on the right-hand side. |
| 196 | `                cos_year_new = np.cos(year_angle)` | Assigns/updates `cos_year_new` with the expression on the right-hand side. |
| 197 | `                new_feature = np.array([predicted_scaled, sin_day_new, cos_day_new, sin_year_new, cos_year_new])` | Assigns/updates `new_feature` with the expression on the right-hand side. |
| 198 | `            else:` | Starts an `else` branch for the preceding conditional. |
| 199 | `                new_feature = np.array([predicted_scaled, sin_day_new, cos_day_new])` | Assigns/updates `new_feature` with the expression on the right-hand side. |
| 200 | `` | Blank line used to separate logical sections and improve readability. |
| 201 | `            # Update sliding window` | Comment explaining intent/context: Update sliding window |
| 202 | `            current_window = np.concatenate([current_window[1:], new_feature[np.newaxis, :]], axis=0)` | Assigns/updates `current_window` with the expression on the right-hand side. |
| 203 | `` | Blank line used to separate logical sections and improve readability. |
| 204 | `            if self.update_training:` | Starts an `if` block: conditional control flow based on `self.update_training`. |
| 205 | `                training_data = np.concatenate([training_data, new_feature[np.newaxis, :]], axis=0)` | Assigns/updates `training_data` with the expression on the right-hand side. |
| 206 | `                X_train, y_train = self.create_windowed_dataset(training_data)` | Assigns/updates `X_train, y_train` with the expression on the right-hand side. |
| 207 | `                self.model.fit(X_train, y_train, epochs=self.epochs_per_update, verbose=0)` | Executes Python statement: `self.model.fit(X_train, y_train, epochs=self.epochs_per_update, verbose=0)`. |
| 208 | `` | Blank line used to separate logical sections and improve readability. |
| 209 | `        return np.array(future_timestamps), np.array(predictions)` | Returns value(s) from the current function: `np.array(future_timestamps), np.array(predictions)`. |
| 210 | `` | Blank line used to separate logical sections and improve readability. |
| 211 | `if __name__ == "__main__":` | Starts an `if` block: conditional control flow based on `__name__ == "__main__"`. |
| 212 | `` | Blank line used to separate logical sections and improve readability. |
| 213 | `    # start_time = datetime.strptime("01012023000000", "%d%m%Y%H%M%S")` | Comment explaining intent/context: start_time = datetime.strptime("01012023000000", "%d%m%Y%H%M%S") |
| 214 | `    # timestamps = [(start_time + timedelta(seconds=i)) for i in range(10000)] #.strftime("%d%m%Y%H%M%S") for i in range(10000)]` | Comment explaining intent/context: timestamps = [(start_time + timedelta(seconds=i)) for i in range(10000)] #.strftime("%d%m%Y%H%M%S") for i in range(10000)] |
| 215 | `` | Blank line used to separate logical sections and improve readability. |
| 216 | `    # t_numeric = np.arange(10000)` | Comment explaining intent/context: t_numeric = np.arange(10000) |
| 217 | `    # field_data = np.sin(0.001 * t_numeric) + 0.05 * np.random.randn(len(t_numeric))` | Comment explaining intent/context: field_data = np.sin(0.001 * t_numeric) + 0.05 * np.random.randn(len(t_numeric)) |
| 218 | `` | Blank line used to separate logical sections and improve readability. |
| 219 | `    # read from magdata` | Comment explaining intent/context: read from magdata |
| 220 | `    file = sys.argv[1] #r'C:\\Users\\DELL\\Desktop\\Projects\\quantum\\magnavis\\src\\sessions\\c5763b7d-cd79-4bdc-a9b1-c8b8e753e9e7\\predict_input.csv'` | Assigns/updates `file` with the expression on the right-hand side. |
| 221 | `    print('filein', file)` | Executes Python statement: `print('filein', file)`. |
| 222 | `    # predictor.(train_data)` | Comment explaining intent/context: predictor.(train_data) |
| 223 | `    df_in = pd.read_csv(file)` | Assigns/updates `df_in` with the expression on the right-hand side. |
| 224 | `` | Blank line used to separate logical sections and improve readability. |
| 225 | `    train_window_minutes = _parse_train_window_minutes()` | Assigns/updates `train_window_minutes` with the expression on the right-hand side. |
| 226 | `    predictor = LSTMPredictor(window_size=15, initial_train_points=len(df_in),` | Assigns/updates `predictor` with the expression on the right-hand side. |
| 227 | `                              epochs_per_update=10, learning_rate=0.001,` | Assigns/updates `epochs_per_update` with the expression on the right-hand side. |
| 228 | `                              update_training=True, train_window_minutes=train_window_minutes)` | Assigns/updates `update_training` with the expression on the right-hand side. |
| 229 | `` | Blank line used to separate logical sections and improve readability. |
| 230 | `    df_in['x'] = pd.to_datetime(df_in['x'])` | Assigns/updates `df_in['x']` with the expression on the right-hand side. |
| 231 | `    print('input head for predict', df_in.head())` | Executes Python statement: `print('input head for predict', df_in.head())`. |
| 232 | `    timestamps = df_in['x'].to_list()` | Assigns/updates `timestamps` with the expression on the right-hand side. |
| 233 | `    field_data = df_in['y'].to_list()` | Assigns/updates `field_data` with the expression on the right-hand side. |
| 234 | `    future_times, future_predictions = predictor.forecast(timestamps, field_data, n_future=100)` | Assigns/updates `future_times, future_predictions` with the expression on the right-hand side. |
| 235 | `    df_out = pd.DataFrame({'x': future_times, 'y': future_predictions})` | Assigns/updates `df_out` with the expression on the right-hand side. |
| 236 | `    folder = os.path.dirname(file)` | Assigns/updates `folder` with the expression on the right-hand side. |
| 237 | `    df_out.to_csv(os.path.join(folder, 'predict_out.csv'), index=False)` | Executes Python statement: `df_out.to_csv(os.path.join(folder, 'predict_out.csv'), index=False)`. |
| 238 | `    # print("Future Timestamps:", future_times)` | Comment explaining intent/context: print("Future Timestamps:", future_times) |
| 239 | `    # print("Future Magnetic Field Predictions:", future_predictions)` | Comment explaining intent/context: print("Future Magnetic Field Predictions:", future_predictions) |
