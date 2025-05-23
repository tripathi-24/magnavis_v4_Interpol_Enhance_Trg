import sys
import serial
import threading
import time
from datetime import datetime, timezone # Import timezone
import requests

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QHBoxLayout, QLabel, QLineEdit,
                             QComboBox, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
import pyqtgraph as pg

WEB_API_URL = "http://127.0.0.1:8000/quantum/magnavis/api/magnetic-data/"
API_POST_INTERVAL_MS = 1000 # 1 second in milliseconds

class SerialReader(QThread):
    data_received = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, port, baud_rate):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.running = False
        self.serial_connection = None

    def run(self):
        self.running = True
        try:
            self.serial_connection = serial.Serial(self.port, self.baud_rate, timeout=1)
            print(f"Opened serial port: {self.port} at {self.baud_rate} baud")
            while self.running:
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8').strip()
                    if line:
                        try:
                            values_str = line.split(',')
                            if len(values_str) == 4:
                                t = float(values_str[0]) # Keep t_val for logging/plotting if needed
                                x = float(values_str[1])
                                y = float(values_str[2])
                                z = float(values_str[3])
                                self.data_received.emit([t, x, y, z])
                            else:
                                self.error_occurred.emit(f"Malformed data: {line}")
                        except ValueError:
                            self.error_occurred.emit(f"Could not parse data: {line}")
                time.sleep(0.01)
        except serial.SerialException as e:
            self.error_occurred.emit(f"Serial port error: {e}")
        finally:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
                print("Serial port closed.")

    def stop(self):
        self.running = False
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.wait()

class MainWindow(QMainWindow):
    api_post_result = pyqtSignal(bool, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Serial Data Plotter, Logger & API Sender")
        self.setGeometry(100, 100, 1000, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.serial_reader = None
        self.log_file = None
        self.api_timer = QTimer(self)

        self.init_ui()
        self.init_plot()
        self.data_buffer = {'t': [], 'x': [], 'y': [], 'z': []}
        self.max_points = 500
        self.latest_api_data = None

        self.api_post_result.connect(self.handle_api_post_result)

    def init_ui(self):
        serial_config_layout = QHBoxLayout()

        serial_config_layout.addWidget(QLabel("Port:"))
        self.port_input = QComboBox()
        self.populate_ports()
        serial_config_layout.addWidget(self.port_input)

        serial_config_layout.addWidget(QLabel("Baud Rate:"))
        self.baud_rate_input = QLineEdit("9600")
        serial_config_layout.addWidget(self.baud_rate_input)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_serial)
        serial_config_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_serial)
        self.stop_button.setEnabled(False)
        serial_config_layout.addWidget(self.stop_button)

        self.reset_plot_button = QPushButton("Reset Plot View")
        self.reset_plot_button.clicked.connect(self.reset_plot_view)
        serial_config_layout.addWidget(self.reset_plot_button)

        self.main_layout.addLayout(serial_config_layout)

        log_config_layout = QHBoxLayout()
        log_config_layout.addWidget(QLabel("Log File Prefix:"))
        self.log_file_prefix_input = QLineEdit("sensor_data")
        log_config_layout.addWidget(self.log_file_prefix_input)
        self.main_layout.addLayout(log_config_layout)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setWordWrap(True)
        self.main_layout.addWidget(self.status_label)

    def populate_ports(self):
        ports = []
        if sys.platform.startswith('win'):
            for i in range(256):
                try:
                    s = serial.Serial(f'COM{i}')
                    s.close()
                    ports.append(f'COM{i}')
                except (serial.SerialException, OSError):
                    pass
        elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
            import glob
            usb_ports = glob.glob('/dev/ttyUSB*')
            acm_ports = glob.glob('/dev/ttyACM*')
            bluetooth_ports = glob.glob('/dev/tty.Bluetooth-*')
            ports = sorted(usb_ports + acm_ports + bluetooth_ports)
        self.port_input.addItems(ports)

    def init_plot(self):
        self.plot_widget = pg.PlotWidget()
        self.main_layout.addWidget(self.plot_widget)
        self.plot_widget.setTitle("Real-time Sensor Data")
        self.plot_widget.setLabel('bottom', 'Time (t)')
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.addLegend()

        self.curve_x = self.plot_widget.plot(pen='r', name='x vs t')
        self.curve_y = self.plot_widget.plot(pen='g', name='y vs t')
        self.curve_z = self.plot_widget.plot(pen='b', name='z vs t')

    def start_serial(self):
        port = self.port_input.currentText()
        try:
            baud_rate = int(self.baud_rate_input.text())
        except ValueError:
            self.status_label.setText("Status: Invalid Baud Rate!")
            return

        if not port:
            self.status_label.setText("Status: Please select a serial port.")
            return

        self.data_buffer = {'t': [], 'x': [], 'y': [], 'z': []}
        self.latest_api_data = None
        self.curve_x.clear()
        self.curve_y.clear()
        self.curve_z.clear()

        self.serial_reader = SerialReader(port, baud_rate)
        self.serial_reader.data_received.connect(self.update_data)
        self.serial_reader.error_occurred.connect(self.handle_error)
        self.serial_reader.start()

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.log_file_prefix_input.text()}_{timestamp}.csv"
            self.log_file = open(filename, 'w')
            self.log_file.write("timestamp_pc,t,x,y,z\n")
            print(f"Logging data to: {filename}")
        except IOError as e:
            self.status_label.setText(f"Status: Could not open log file: {e}")
            self.stop_serial()
            return

        self.api_timer.timeout.connect(self.send_data_to_api)
        self.api_timer.start(API_POST_INTERVAL_MS)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(f"Status: Reading from {port} at {baud_rate} baud & sending to API...")

    def stop_serial(self):
        if self.serial_reader:
            self.serial_reader.stop()
            self.serial_reader = None

        if self.log_file:
            self.log_file.close()
            self.log_file = None
            print("Log file closed.")

        self.api_timer.stop()

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Stopped.")

    def update_data(self, data):
        t_val, x_val, y_val, z_val = data
        current_time_pc = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # --- MODIFIED: New API payload format ---
        self.latest_api_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(), # UTC timestamp
            "b_x": x_val,
            "b_y": y_val,
            "b_z": z_val
        }
        # --- END MODIFIED ---

        if self.log_file:
            self.log_file.write(f"{current_time_pc},{t_val},{x_val},{y_val},{z_val}\n")
            self.log_file.flush()

        self.data_buffer['t'].append(t_val)
        self.data_buffer['x'].append(x_val)
        self.data_buffer['y'].append(y_val)
        self.data_buffer['z'].append(z_val)

        if len(self.data_buffer['t']) > self.max_points:
            for key in self.data_buffer:
                self.data_buffer[key] = self.data_buffer[key][-self.max_points:]

        self.curve_x.setData(self.data_buffer['t'], self.data_buffer['x'])
        self.curve_y.setData(self.data_buffer['t'], self.data_buffer['y'])
        self.curve_z.setData(self.data_buffer['t'], self.data_buffer['z'])

        self.plot_widget.autoRange()

    def send_data_to_api(self):
        if self.latest_api_data is None:
            return

        payload = self.latest_api_data
        # You might want to print the *new* keys in the print statement
        print(f"Attempting to send single data point to API: timestamp={payload.get('timestamp', 'N/A')}, b_x={payload.get('b_x', 'N/A')}...")
        threading.Thread(target=self._perform_api_post, args=(payload,)).start()

    def _perform_api_post(self, payload_dict):
        try:
            response = requests.post(WEB_API_URL, json=payload_dict, timeout=5)
            response.raise_for_status()
            # Update the message to reflect new keys
            message = f"API Post Successful: Sent timestamp={payload_dict.get('timestamp', 'N/A')}. Status {response.status_code}"
            print(message)
            self.api_post_result.emit(True, message)
        except requests.exceptions.Timeout:
            message = f"API Post Failed: Timeout for point timestamp={payload_dict.get('timestamp', 'N/A')}."
            print(message)
            self.api_post_result.emit(False, message)
        except requests.exceptions.ConnectionError as e:
            message = f"API Post Failed: Connection error: {e}"
            print(message)
            self.api_post_result.emit(False, message)
        except requests.exceptions.HTTPError as e:
            message = f"API Post Failed: HTTP Error {e.response.status_code} - {e.response.text}"
            print(message)
            self.api_post_result.emit(False, message)
        except Exception as e:
            message = f"API Post Failed: An unexpected error occurred: {e}"
            print(message)
            self.api_post_result.emit(False, message)

    def handle_api_post_result(self, success, message):
        self.status_label.setText(f"Status: {message}")

    def reset_plot_view(self):
        if self.data_buffer['t']:
            self.plot_widget.autoRange()
        else:
            self.plot_widget.setXRange(0, 100)
            self.plot_widget.setYRange(-10, 10)
            self.plot_widget.setTitle("Real-time Sensor Data (No Data)")
            self.status_label.setText("Plot view reset. No data available.")

    def handle_error(self, message):
        self.status_label.setText(f"Error - {message}")
        self.stop_serial()

    def closeEvent(self, event):
        self.stop_serial()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())