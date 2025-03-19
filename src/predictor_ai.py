import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

class LSTMPredictor:
    def __init__(self, window_size=5, initial_train_points=3400,
                 epochs_per_update=5, learning_rate=0.001, update_training=True):

        self.window_size = window_size
        self.initial_train_points = initial_train_points
        self.epochs_per_update = epochs_per_update
        self.learning_rate = learning_rate
        self.update_training = update_training
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def create_windowed_dataset(self, series):

        X, y = [], []
        for i in range(len(series) - self.window_size):
            X.append(series[i : i + self.window_size])
            y.append(series[i + self.window_size])
        return np.array(X), np.array(y)

    def build_model(self):
        model = Sequential()
        model.add(LSTM(32, return_sequences=False, input_shape=(self.window_size, 1)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        self.model = model

    def forecast(self, timestamps, field_data, n_future):

        field = np.array(field_data).reshape(-1, 1)

        field_scaled = self.scaler.fit_transform(field)

        if self.initial_train_points < self.window_size:
            raise ValueError("initial_train_points must be >= window_size.")
        if self.initial_train_points > len(field_scaled):
            raise ValueError("initial_train_points cannot exceed total data length.")

        if self.model is None:
            self.build_model()

        initial_data = field_scaled[:self.initial_train_points]
        X_init, y_init = self.create_windowed_dataset(initial_data)
        if len(X_init) > 0:
            self.model.fit(X_init, y_init, epochs=self.epochs_per_update, verbose=0)
        
        # up until here

        predictions = []
        current_window = field_scaled[self.initial_train_points - self.window_size : self.initial_train_points]

        if self.update_training:
            training_data = initial_data.copy()

        for i in range(n_future):
            current_window_reshaped = np.array([current_window])
            predicted_scaled = self.model.predict(current_window_reshaped, verbose=0)[0, 0]

            predicted_value = self.scaler.inverse_transform([[predicted_scaled]])[0, 0]
            predictions.append(predicted_value)


            current_window = np.concatenate([current_window[1:], np.array([[predicted_scaled]])], axis=0)

            if self.update_training:
                training_data = np.concatenate([training_data, np.array([[predicted_scaled]])], axis=0)
                X_train, y_train = self.create_windowed_dataset(training_data)
                self.model.fit(X_train, y_train, epochs=self.epochs_per_update, verbose=0)


        try:
            dt_last = datetime.strptime(timestamps[-1], "%d%m%Y%H%M%S")

            if len(timestamps) > 1:
                dt_second_last = datetime.strptime(timestamps[-2], "%d%m%Y%H%M%S")
                time_delta = dt_last - dt_second_last
            else:
                time_delta = timedelta(seconds=1)
        except Exception as e:
            raise ValueError("Error parsing timestamps. Ensure they are in DDMMYYYYHHMMSS format.") from e

        future_timestamps = []
        for i in range(n_future):
            new_time = dt_last + (i + 1) * time_delta

            future_timestamps.append(new_time.strftime("%d%m%Y%H%M%S"))

        return np.array(future_timestamps), np.array(predictions)

if __name__ == "__main__":

    start_time = datetime.strptime("01012023000000", "%d%m%Y%H%M%S")
    timestamps = [(start_time + timedelta(seconds=i)).strftime("%d%m%Y%H%M%S") for i in range(10000)]

    t_numeric = np.arange(10000)
    field_data = np.sin(0.001 * t_numeric) + 0.05 * np.random.randn(len(t_numeric))

    predictor = LSTMPredictor(window_size=5, initial_train_points=3400,
                              epochs_per_update=5, learning_rate=0.001, update_training=True)

    # read from magdata
    file = r'C:\Users\DELL\Desktop\Projects\quantum\magnavis\src\sessions\a5c67bf0-bfe8-41ff-ae75-20ee51dbc70c\download_mag.json'
    # predictor.(train_data)
    future_times, future_predictions = predictor.forecast(timestamps, field_data, n_future=100)

    print("Future Timestamps:", future_times)
    print("Future Magnetic Field Predictions:", future_predictions)
