import sys
import serial
import threading
import time
from datetime import datetime

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget,
                             QPushButton, QHBoxLayout, QLabel, QLineEdit,
                             QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import pyqtgraph as pg

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
                            # Assuming data is "t,x,y,z"
                            values_str = line.split(',')
                            if len(values_str) == 4:
                                t = float(values_str[0])
                                x = float(values_str[1])
                                y = float(values_str[2])
                                z = float(values_str[3])
                                self.data_received.emit([t, x, y, z])
                            else:
                                self.error_occurred.emit(f"Malformed data: {line}")
                        except ValueError:
                            self.error_occurred.emit(f"Could not parse data: {line}")
                time.sleep(0.01) # Small delay to prevent busy-waiting
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
        self.wait() # Wait for the thread to finish

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Serial Data Plotter and Logger")
        self.setGeometry(100, 100, 1000, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.serial_reader = None
        self.log_file = None

        self.init_ui()
        self.init_plot()
        self.data_buffer = {'t': [], 'x': [], 'y': [], 'z': []}
        self.max_points = 500 # Max points to display on the plot

    def init_ui(self):
        # Serial Port Configuration
        serial_config_layout = QHBoxLayout()

        serial_config_layout.addWidget(QLabel("Port:"))
        self.port_input = QComboBox()
        self.populate_ports()
        serial_config_layout.addWidget(self.port_input)

        serial_config_layout.addWidget(QLabel("Baud Rate:"))
        self.baud_rate_input = QLineEdit("9600") # Default baud rate
        serial_config_layout.addWidget(self.baud_rate_input)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_serial)
        serial_config_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_serial)
        self.stop_button.setEnabled(False)
        serial_config_layout.addWidget(self.stop_button)

        self.main_layout.addLayout(serial_config_layout)

        # Log File Configuration
        log_config_layout = QHBoxLayout()
        log_config_layout.addWidget(QLabel("Log File Prefix:"))
        self.log_file_prefix_input = QLineEdit("sensor_data")
        log_config_layout.addWidget(self.log_file_prefix_input)
        self.main_layout.addLayout(log_config_layout)

        self.status_label = QLabel("Status: Idle")
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
            bluetooth_ports = glob.glob('/dev/tty.Bluetooth-*') # For macOS
            ports = sorted(usb_ports + acm_ports + bluetooth_ports)
        self.port_input.addItems(ports)

    # --- MODIFIED ---
    def init_plot(self):
        self.plot_widget = pg.PlotWidget()
        self.main_layout.addWidget(self.plot_widget)
        self.plot_widget.setTitle("Real-time Sensor Data")
        self.plot_widget.setLabel('bottom', 'Time (t)') # X-axis is now 't'
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.addLegend()

        # We now plot x, y, and z against t
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

        self.serial_reader = SerialReader(port, baud_rate)
        self.serial_reader.data_received.connect(self.update_data)
        self.serial_reader.error_occurred.connect(self.handle_error)
        self.serial_reader.start()

        # Open log file
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.log_file_prefix_input.text()}_{timestamp}.csv"
            self.log_file = open(filename, 'w')
            self.log_file.write("timestamp_pc,t,x,y,z\n") # Header
            print(f"Logging data to: {filename}")
        except IOError as e:
            self.status_label.setText(f"Status: Could not open log file: {e}")
            self.stop_serial() # Stop serial if logging fails
            return

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText(f"Status: Reading from {port} at {baud_rate} baud...")

    def stop_serial(self):
        if self.serial_reader:
            self.serial_reader.stop()
            self.serial_reader = None

        if self.log_file:
            self.log_file.close()
            self.log_file = None
            print("Log file closed.")

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Stopped.")

    # --- MODIFIED ---
    def update_data(self, data):
        t_val, x_val, y_val, z_val = data
        current_time_pc = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Log data to file
        if self.log_file:
            self.log_file.write(f"{current_time_pc},{t_val},{x_val},{y_val},{z_val}\n")
            self.log_file.flush() # Ensure data is written immediately

        # Update plot data buffers
        self.data_buffer['t'].append(t_val)
        self.data_buffer['x'].append(x_val)
        self.data_buffer['y'].append(y_val)
        self.data_buffer['z'].append(z_val)

        # Keep only the last `max_points` for plotting
        if len(self.data_buffer['t']) > self.max_points:
            for key in self.data_buffer:
                self.data_buffer[key] = self.data_buffer[key][-self.max_points:]

        # Update plot curves: now specify both x (time) and y (value)
        self.curve_x.setData(self.data_buffer['t'], self.data_buffer['x'])
        self.curve_y.setData(self.data_buffer['t'], self.data_buffer['y'])
        self.curve_z.setData(self.data_buffer['t'], self.data_buffer['z'])

    def handle_error(self, message):
        self.status_label.setText(f"Status: Error - {message}")
        self.stop_serial()

    def closeEvent(self, event):
        self.stop_serial()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())