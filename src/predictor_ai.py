import os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def _parse_train_window_minutes():
    """Read optional training window (minutes) from env TRAIN_WINDOW_MINUTES."""
    try:
        val = os.environ.get("TRAIN_WINDOW_MINUTES", None)
        if val is None or val == "":
            return None
        return float(val)
    except Exception:
        return None


class GRUPredictor:
    """
    GRU-based recurrent neural network for magnetic field prediction.
    
    Uses a GRU (Gated Recurrent Unit) layer to model temporal dependencies
    in magnetic field time series data, with cyclic time-of-day features.
    """
    def __init__(self, window_size=5, initial_train_points=3400,
                 epochs_per_update=5, learning_rate=0.001, update_training=True,
                 use_yearly_cycle=False, train_window_minutes=None):

        self.window_size = window_size
        self.initial_train_points = initial_train_points
        self.epochs_per_update = epochs_per_update
        self.learning_rate = learning_rate
        self.update_training = update_training
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.use_yearly_cycle = use_yearly_cycle
        self.train_window_minutes = train_window_minutes

    def create_windowed_dataset(self, series):

        X, y = [], []
        for i in range(len(series) - self.window_size):
            X.append(series[i : i + self.window_size])
            # Target is the magnetic field (scaled) = column 0
            y.append(series[i + self.window_size, 0])
        return np.array(X), np.array(y)

    def build_model(self, feature_dim):
        model = Sequential()
        # Recurrent backbone: GRU (replaces previous LSTM)
        model.add(GRU(32, return_sequences=False, input_shape=(self.window_size, feature_dim)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.model = model

    def save_model(self, filepath):
        """Save the trained model weights and architecture to disk."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        self.model.save(filepath)
        # Also save scaler state for consistent feature scaling
        import pickle
        scaler_path = filepath.replace('.h5', '_scaler.pkl').replace('.keras', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_model(self, filepath):
        """Load a pre-trained model from disk."""
        self.model = tf.keras.models.load_model(filepath)
        # Load scaler state if available
        import pickle
        import os
        scaler_path = filepath.replace('.h5', '_scaler.pkl').replace('.keras', '_scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

    def _compute_time_features(self, ts_array):
        """Compute cyclic time features (daily, optionally yearly) for an array of pandas Timestamps."""
        seconds_in_day = 24 * 3600
        # Support both DatetimeIndex-like objects (have .hour/.minute/...) and pandas Series (use .dt accessor).
        ts_array = pd.to_datetime(ts_array)
        if isinstance(ts_array, pd.Series):
            hour = ts_array.dt.hour
            minute = ts_array.dt.minute
            second = ts_array.dt.second
            microsecond = ts_array.dt.microsecond
            dayofyear = ts_array.dt.dayofyear
        else:
            hour = ts_array.hour
            minute = ts_array.minute
            second = ts_array.second
            microsecond = ts_array.microsecond
            dayofyear = ts_array.dayofyear

        sec_of_day = hour * 3600 + minute * 60 + second + microsecond / 1e6
        day_angle = 2 * np.pi * (sec_of_day / seconds_in_day)
        sin_day = np.sin(day_angle)
        cos_day = np.cos(day_angle)

        if self.use_yearly_cycle:
            # Day-of-year fraction (including leap-year effect approximately)
            year_angle = 2 * np.pi * (dayofyear / 365.25)
            sin_year = np.sin(year_angle)
            cos_year = np.cos(year_angle)
            return sin_day, cos_day, sin_year, cos_year
        else:
            return sin_day, cos_day

    def forecast(self, timestamps, field_data, n_future, pretrained_model_path=None):

        # Build a clean time series table first so timestamps and values stay aligned.
        ts = pd.to_datetime(timestamps)
        df0 = pd.DataFrame({"ts": ts, "field": field_data})
        df0["ts"] = pd.to_datetime(df0["ts"], errors="coerce")
        df0["field"] = pd.to_numeric(df0["field"], errors="coerce")
        df0 = df0.dropna(subset=["ts", "field"]).sort_values("ts").reset_index(drop=True)

        # Defensive: if duplicate timestamps exist, average them so:
        # - time_delta is non-zero
        # - feature arrays and target arrays have consistent lengths
        df0 = df0.groupby("ts", as_index=False)["field"].mean().sort_values("ts").reset_index(drop=True)

        ts = df0["ts"]  # pandas Series of datetime64
        field = df0["field"].to_numpy(dtype=float).reshape(-1, 1)

        # Optional: restrict training data to the most recent N minutes
        if self.train_window_minutes:
            cutoff = ts.max().to_pydatetime() - timedelta(minutes=self.train_window_minutes)
            mask = ts >= cutoff
            if int(mask.sum()) > self.window_size + 1:
                ts = ts[mask].reset_index(drop=True)
                field = field[mask.to_numpy()]

        # Load pre-trained model and scaler BEFORE scaling, so we use the same scaler as training.
        # Otherwise we would fit_transform on short input and break pre-trained predictions.
        if pretrained_model_path:
            try:
                self.load_model(pretrained_model_path)
                print(f"Using pre-trained model and scaler: {pretrained_model_path}")
            except Exception as e:
                print(f"Warning: Could not load pre-trained model from {pretrained_model_path}: {e}")
                print("Falling back to building new model.")
                self.model = None  # ensure we fit scaler and build model below

        # Scale magnetic field only; sin/cos are already bounded in [-1, 1]
        # Pre-trained: use loaded scaler (transform). New model: fit_transform.
        if self.model is not None:
            # Pre-trained model loaded: use transform so scaling matches training data
            field_scaled = self.scaler.transform(field).flatten()
        else:
            field_scaled = self.scaler.fit_transform(field).flatten()

        # Build feature matrix: [mag_scaled, sin_time, cos_time, (optional) sin_year, cos_year]
        time_feats = self._compute_time_features(ts)
        if self.use_yearly_cycle:
            sin_day, cos_day, sin_year, cos_year = time_feats
            feature_matrix = np.column_stack([field_scaled, sin_day, cos_day, sin_year, cos_year])
        else:
            sin_day, cos_day = time_feats
            feature_matrix = np.column_stack([field_scaled, sin_day, cos_day])

        # Ensure initial_train_points does not exceed available data
        self.initial_train_points = min(self.initial_train_points, len(field_scaled))

        if self.initial_train_points < self.window_size:
            raise ValueError("initial_train_points must be >= window_size.")
        if self.initial_train_points > len(field_scaled):
            raise ValueError("initial_train_points cannot exceed total data length.")

        # Build model only if we don't have one (pre-trained load failed or not requested)
        if self.model is None:
            self.build_model(feature_matrix.shape[1])

        initial_data = feature_matrix[:self.initial_train_points]
        X_init, y_init = self.create_windowed_dataset(initial_data)
        if len(X_init) > 0:
            self.model.fit(X_init, y_init, epochs=self.epochs_per_update, verbose=0)
        
        # up until here

        # Compute base time delta for future steps
        try:
            # IMPORTANT: keep these as Python datetime/timedelta (not numpy.datetime64),
            # because later we access `.hour/.minute/...` when computing cyclic features.
            dt_last = ts.iloc[-1].to_pydatetime() if hasattr(ts, "iloc") else pd.to_datetime(ts[-1]).to_pydatetime()

            # Prefer a robust estimate of sampling period:
            # - compute diffs over the whole series
            # - keep positive diffs
            # - use median as stable step
            if len(ts) > 1:
                diffs = pd.Series(ts).diff().dropna()
                diffs = diffs[diffs > pd.Timedelta(0)]
                if len(diffs) > 0:
                    time_delta = diffs.median().to_pytimedelta()
                else:
                    time_delta = timedelta(seconds=1)
            else:
                time_delta = timedelta(seconds=1)

            # Clamp: never allow 0 or negative.
            if time_delta.total_seconds() <= 0:
                time_delta = timedelta(seconds=1)
        except Exception as e:
            raise ValueError("Error parsing timestamps for time delta.") from e

        predictions = []
        future_timestamps = []
        current_window = feature_matrix[self.initial_train_points - self.window_size : self.initial_train_points]

        if self.update_training:
            training_data = initial_data.copy()

        for i in range(n_future):
            current_window_reshaped = np.array([current_window])
            predicted_scaled = self.model.predict(current_window_reshaped, verbose=0)[0, 0]

            # Inverse scale to get magnetic field value
            predicted_value = self.scaler.inverse_transform([[predicted_scaled]])[0, 0]
            predictions.append(predicted_value)

            # Next timestamp and its cyclic features
            new_time = dt_last + (i + 1) * time_delta
            future_timestamps.append(new_time)

            sec_of_day = new_time.hour * 3600 + new_time.minute * 60 + new_time.second + new_time.microsecond / 1e6
            day_angle = 2 * np.pi * (sec_of_day / (24 * 3600))
            sin_day_new = np.sin(day_angle)
            cos_day_new = np.cos(day_angle)

            if self.use_yearly_cycle:
                day_of_year = new_time.timetuple().tm_yday
                year_angle = 2 * np.pi * (day_of_year / 365.25)
                sin_year_new = np.sin(year_angle)
                cos_year_new = np.cos(year_angle)
                new_feature = np.array([predicted_scaled, sin_day_new, cos_day_new, sin_year_new, cos_year_new])
            else:
                new_feature = np.array([predicted_scaled, sin_day_new, cos_day_new])

            # Update sliding window
            current_window = np.concatenate([current_window[1:], new_feature[np.newaxis, :]], axis=0)

            if self.update_training:
                training_data = np.concatenate([training_data, new_feature[np.newaxis, :]], axis=0)
                X_train, y_train = self.create_windowed_dataset(training_data)
                self.model.fit(X_train, y_train, epochs=self.epochs_per_update, verbose=0)

        return np.array(future_timestamps), np.array(predictions)

if __name__ == "__main__":

    # start_time = datetime.strptime("01012023000000", "%d%m%Y%H%M%S")
    # timestamps = [(start_time + timedelta(seconds=i)) for i in range(10000)] #.strftime("%d%m%Y%H%M%S") for i in range(10000)]

    # t_numeric = np.arange(10000)
    # field_data = np.sin(0.001 * t_numeric) + 0.05 * np.random.randn(len(t_numeric))

    # read from magdata
    file = sys.argv[1] #r'C:\Users\DELL\Desktop\Projects\quantum\magnavis\src\sessions\c5763b7d-cd79-4bdc-a9b1-c8b8e753e9e7\predict_input.csv'
    print('filein', file)
    # predictor.(train_data)
    df_in = pd.read_csv(file)

    train_window_minutes = _parse_train_window_minutes()
    # Optional: path to pre-trained model (set via env var PRETRAINED_MODEL_PATH)
    pretrained_model_path = os.environ.get("PRETRAINED_MODEL_PATH", None)
    
    predictor = GRUPredictor(window_size=15, initial_train_points=len(df_in),
                             epochs_per_update=10, learning_rate=0.001,
                             update_training=True, train_window_minutes=train_window_minutes,
                             use_yearly_cycle=True)  # Enable yearly cycle for seasonal patterns

    df_in['x'] = pd.to_datetime(df_in['x'])
    print('input head for predict', df_in.head())
    timestamps = df_in['x'].to_list()
    field_data = df_in['y'].to_list()
    future_times, future_predictions = predictor.forecast(timestamps, field_data, n_future=100,
                                                          pretrained_model_path=pretrained_model_path)
    df_out = pd.DataFrame({'x': future_times, 'y': future_predictions})
    folder = os.path.dirname(file)
    df_out.to_csv(os.path.join(folder, 'predict_out.csv'), index=False)
    # print("Future Timestamps:", future_times)
    # print("Future Magnetic Field Predictions:", future_predictions)
