"""
Fast DB-backed Magnavis (single sensor bring-up).

Purpose:
- Provide the same core workflow as application.py (historic -> realtime -> prediction -> anomalies),
  but with a minimal GUI and without importing heavy modules from src/application.py (VTK, GeoPandas, etc.).
- This significantly reduces startup time and improves responsiveness.

GUI:
- Top: a single tab (OBS1_1) with controls + plots
- Bottom: log window

Run:
    python src/application_temp_fast.py
"""

from __future__ import annotations

import os
import sys
import time
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from Anomaly_detector import AnomalyDetector
from data_convert_db_now import (
    get_latest_sensor_ids,
    get_latest_sensor_id_like,
    get_min_timestamp_at_or_after,
    fetch_timeseries_window_multi,
    fetch_timeseries_between_multi,
)


# Make matplotlib/font caches writable to avoid slow cache rebuilds.
_APP_BASE_DIR = os.path.dirname(__file__)
_LOCAL_CACHE = os.path.join(_APP_BASE_DIR, ".cache")
_MPL_CACHE = os.path.join(_LOCAL_CACHE, "mpl")
_XDG_CACHE = os.path.join(_LOCAL_CACHE, "xdg")
try:
    os.makedirs(_MPL_CACHE, exist_ok=True)
    os.makedirs(_XDG_CACHE, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE)
    os.environ.setdefault("XDG_CACHE_HOME", _XDG_CACHE)
except Exception:
    pass


@dataclass
class SensorState:
    sensor_id: str
    display_name: str

    # Historic (blue) snapshot, 1 Hz
    base_x: List[datetime] = field(default_factory=list)
    base_y: List[float] = field(default_factory=list)

    # Realtime (green) accumulated, 1 Hz
    rt_x: List[datetime] = field(default_factory=list)
    rt_y: List[float] = field(default_factory=list)

    # Realtime accumulated (for anomaly comparison)
    new_x: List[datetime] = field(default_factory=list)
    new_y: List[float] = field(default_factory=list)

    # Plot baseline for ΔB visualization
    baseline_nT: Optional[float] = None

    # Predictor/anomaly
    predictor_input_file: Optional[str] = None
    prediction_process: Optional[subprocess.Popen] = None
    predict_x: List[datetime] = field(default_factory=list)
    predict_y: List[float] = field(default_factory=list)
    last_pred_out_mtime: Optional[float] = None

    anomaly_detector: AnomalyDetector = field(default_factory=lambda: AnomalyDetector(threshold_multiplier=2.5, min_samples_for_threshold=10))
    anomaly_times: List[datetime] = field(default_factory=list)

    # UI
    static_canvas: Optional[FigureCanvas] = None
    dynamic_canvas: Optional[FigureCanvas] = None
    static_ax = None
    dynamic_ax = None
    blue_line = None
    green_line = None
    purple_line = None
    anomaly_vlines_dyn: list = field(default_factory=list)
    anomaly_vlines_static: list = field(default_factory=list)

    # Settings
    train_window_minutes: Optional[int] = None

    # Throttles
    last_saved_points: int = 0
    last_pred_start_ts: float = 0.0
    predict_cooldown_s: int = 5


class FetchWorker(QObject):
    finished = pyqtSignal()
    updated = pyqtSignal(pd.DataFrame, bool)  # df, is_incremental

    def fetch_initial(self, sensor_id: str, start_time: datetime, end_time: datetime, n_seconds: int):
        try:
            d = fetch_timeseries_window_multi([sensor_id], start_time=start_time, end_time=end_time, target_n_seconds=n_seconds)
            df = d.get(sensor_id) or pd.DataFrame(columns=["time_H", "mag_H_nT"])
        except Exception:
            df = pd.DataFrame(columns=["time_H", "mag_H_nT"])
        self.updated.emit(df, False)
        self.finished.emit()

    def fetch_incremental(self, sensor_id: str, start_time: datetime, end_time: datetime):
        try:
            d = fetch_timeseries_between_multi([sensor_id], start_time=start_time, end_time=end_time, limit_rows=20000)
            df = d.get(sensor_id) or pd.DataFrame(columns=["time_H", "mag_H_nT"])
        except Exception:
            df = pd.DataFrame(columns=["time_H", "mag_H_nT"])
        self.updated.emit(df, True)
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Magnavis (IITK Observatory)")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        central.setLayout(layout)

        splitter = QSplitter()
        splitter.setOrientation(2)  # Qt.Vertical (avoid importing Qt namespace)
        layout.addWidget(splitter)

        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)

        self.log = QTextEdit()
        splitter.addWidget(self.log)
        self.log.setMaximumHeight(220)

        splitter.setStretchFactor(0, 10)
        splitter.setStretchFactor(1, 2)


class Controller(QObject):
    def __init__(self, win: MainWindow):
        super().__init__()
        self.win = win
        self.session_id = str(__import__("uuid").uuid4())
        self.state: Optional[SensorState] = None

        # Simulation clock
        self.sim_start = datetime(2026, 1, 5, 0, 0, 0)
        self.sim_hist_end = self.sim_start + timedelta(minutes=60)
        self.sim_step_s = 20
        self.sim_rt_start = self.sim_hist_end
        self.sim_rt_end = self.sim_rt_start + timedelta(seconds=self.sim_step_s)

        # Threading
        self._fetch_thread: Optional[QThread] = None
        self._fetch_worker: Optional[FetchWorker] = None

        # Timers
        self.fetch_timer = QTimer()
        self.fetch_timer.timeout.connect(self.fetch_incremental_tick)

        self.draw_timer = QTimer()
        self.draw_timer.timeout.connect(self.redraw)

        self.pred_poll_timer = QTimer()
        self.pred_poll_timer.timeout.connect(self.poll_prediction_output)

        self._init_sensor_and_ui()

    def log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.win.log.append(f"{ts} : {msg}")

    def _prompt_for_sensor_id(self) -> Optional[str]:
        try:
            options = get_latest_sensor_ids(limit=6)
        except Exception:
            options = []

        if options:
            choice, ok = QInputDialog.getItem(
                self.win,
                "Select Sensor",
                "Choose sensor ID to plot:",
                options,
                0,
                False,
            )
            if ok and choice:
                return str(choice)
            return str(options[0])

        sid = get_latest_sensor_id_like("%OBS1_1")
        return sid

    def _init_sensor_and_ui(self):
        sid = self._prompt_for_sensor_id()
        if not sid:
            self.log("ERROR: No sensor_id available for selection")
            return

        self.state = SensorState(sensor_id=sid, display_name=sid)
        tab = QWidget()
        self.win.tabs.addTab(tab, self.state.display_name)
        tab_layout = QVBoxLayout()
        tab.setLayout(tab_layout)

        # Controls row
        controls = QWidget()
        row = QHBoxLayout()
        controls.setLayout(row)
        tab_layout.addWidget(controls)

        row.addWidget(QLabel("Source: IITK Observatory"))

        row.addWidget(QLabel("Anomaly threshold multiplier:"))
        thr = QDoubleSpinBox()
        thr.setMinimum(0.1)
        thr.setMaximum(10.0)
        thr.setSingleStep(0.1)
        thr.setValue(2.5)
        thr.valueChanged.connect(self._on_thr_changed)
        row.addWidget(thr)

        row.addWidget(QLabel("Training window (min, 0=all):"))
        tw = QDoubleSpinBox()
        tw.setMinimum(0)
        tw.setMaximum(100000)
        tw.setSingleStep(5)
        tw.setDecimals(0)
        tw.setValue(0)
        tw.valueChanged.connect(self._on_train_window_changed)
        row.addWidget(tw)
        row.addStretch(1)

        # Plots
        static_canvas = FigureCanvas(Figure(figsize=(6, 3)))
        dynamic_canvas = FigureCanvas(Figure(figsize=(6, 3)))
        static_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        dynamic_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        tab_layout.addWidget(NavigationToolbar(static_canvas, self.win))
        tab_layout.addWidget(static_canvas)
        tab_layout.addWidget(dynamic_canvas)
        tab_layout.addWidget(NavigationToolbar(dynamic_canvas, self.win))

        self.state.static_canvas = static_canvas
        self.state.dynamic_canvas = dynamic_canvas
        self.state.static_ax = static_canvas.figure.subplots()
        self.state.dynamic_ax = dynamic_canvas.figure.subplots()
        self.state.static_ax.set_ylabel("ΔB (nT)")
        self.state.dynamic_ax.set_ylabel("ΔB (nT)")

        # Kick off initial load
        self.fetch_initial()

        # Start timers
        self.fetch_timer.start(1000 * self.sim_step_s)
        self.draw_timer.start(400)
        self.pred_poll_timer.start(3000)

    def _on_thr_changed(self, v: float):
        if not self.state:
            return
        self.state.anomaly_detector.threshold_multiplier = float(v)
        self.log(f"[{self.state.display_name}] threshold multiplier -> {v:.2f}")

    def _on_train_window_changed(self, v: float):
        if not self.state:
            return
        self.state.train_window_minutes = None if v <= 0 else int(v)
        self.log(f"[{self.state.display_name}] training window minutes -> {self.state.train_window_minutes}")

    def _start_thread(self, func):
        if self._fetch_thread is not None:
            try:
                if self._fetch_thread.isRunning():
                    return
            except RuntimeError:
                self._fetch_thread = None
                self._fetch_worker = None

        thread = QThread(self.win)
        worker = FetchWorker()
        worker.moveToThread(thread)
        thread.started.connect(func)
        worker.updated.connect(self._on_data_updated)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: setattr(self, "_fetch_thread", None))
        thread.finished.connect(lambda: setattr(self, "_fetch_worker", None))
        self._fetch_thread = thread
        self._fetch_worker = worker
        thread.start()

    def fetch_initial(self):
        if not self.state:
            return
        s = self.sim_start
        e = self.sim_hist_end
        self.log(f"Loading historic window from {s} to {e} (sim)")

        def run():
            assert self._fetch_worker is not None
            self._fetch_worker.fetch_initial(self.state.sensor_id, s, e, 3400)

        self._start_thread(run)

    def fetch_incremental_tick(self):
        if not self.state:
            return
        s = self.sim_rt_start
        e = self.sim_rt_end

        def run():
            assert self._fetch_worker is not None
            self._fetch_worker.fetch_incremental(self.state.sensor_id, s, e)

        self._start_thread(run)

    def _on_data_updated(self, df: pd.DataFrame, is_incremental: bool):
        if not self.state or df is None or df.empty:
            # If incremental returns empty, jump to next available timestamp to keep simulation moving
            if self.state and is_incremental:
                nxt = get_min_timestamp_at_or_after(self.state.sensor_id, self.sim_rt_end)
                if nxt is not None:
                    self.sim_rt_start = nxt
                    self.sim_rt_end = self.sim_rt_start + timedelta(seconds=self.sim_step_s)
            return

        t = pd.to_datetime(df["time_H"]).tolist()
        y = df["mag_H_nT"].astype(float).tolist()

        if not is_incremental:
            self.state.base_x = t[-3400:]
            self.state.base_y = y[-3400:]
            self.state.baseline_nT = float(np.median(self.state.base_y)) if self.state.base_y else None
            self.state.rt_x = []
            self.state.rt_y = []
            self.state.new_x = []
            self.state.new_y = []
            # Advance simulated realtime cursor to start after historic
            if self.state.base_x:
                self.sim_rt_start = self.state.base_x[-1]
                self.sim_rt_end = self.sim_rt_start + timedelta(seconds=self.sim_step_s)
            self._write_predict_input(start_predictor=False)
            self._init_lines()
            self.log(f"Historic loaded: {len(self.state.base_x)} points")
            return

        # incremental: append only increasing
        last = self.state.rt_x[-1] if self.state.rt_x else (self.state.base_x[-1] if self.state.base_x else None)
        new_t, new_y = [], []
        for tt, yy in zip(t, y):
            if last is None or tt > last:
                new_t.append(tt)
                new_y.append(yy)
                last = tt
        if new_t:
            self.state.rt_x.extend(new_t)
            self.state.rt_y.extend(new_y)
            # Keep realtime series for anomaly detection (cumulative)
            self.state.new_x.extend(new_t)
            self.state.new_y.extend(new_y)
            self._write_predict_input(start_predictor=True)
        # Advance sim window
        self.sim_rt_start = self.sim_rt_end
        self.sim_rt_end = self.sim_rt_start + timedelta(seconds=self.sim_step_s)

    def _delta(self, ys: List[float]) -> List[float]:
        if not self.state or self.state.baseline_nT is None:
            return ys
        b = self.state.baseline_nT
        return [float(v) - b for v in ys]

    def _init_lines(self):
        s = self.state
        if not s or not s.dynamic_ax or not s.base_x:
            return
        y0 = self._delta(s.base_y)
        s.dynamic_ax.clear()
        s.static_ax.clear()
        s.static_ax.set_ylabel("ΔB (nT)")
        s.dynamic_ax.set_ylabel("ΔB (nT)")
        s.static_line = s.static_ax.plot(s.base_x, y0, ".")[0]
        s.blue_line = s.dynamic_ax.plot(s.base_x, y0)[0]

    def _write_predict_input(self, start_predictor: bool):
        s = self.state
        if not s:
            return
        x_all = s.base_x + s.rt_x
        y_all = s.base_y + s.rt_y
        if not x_all:
            return
        if len(x_all) == s.last_saved_points:
            return
        s.last_saved_points = len(x_all)

        folder = os.path.join(_APP_BASE_DIR, "sessions", self.session_id, s.sensor_id)
        os.makedirs(folder, exist_ok=True)
        inp = os.path.join(folder, "predict_input.csv")
        pd.DataFrame({"x": pd.to_datetime(x_all), "y": y_all}).to_csv(inp, index=False)
        s.predictor_input_file = inp

        if start_predictor:
            self._maybe_start_predictor()

    def _maybe_start_predictor(self):
        s = self.state
        if not s or not s.predictor_input_file:
            return
        if s.prediction_process is not None and s.prediction_process.poll() is None:
            return
        now = time.time()
        if s.last_pred_start_ts and (now - s.last_pred_start_ts) < s.predict_cooldown_s:
            return
        s.last_pred_start_ts = now

        python_exe = sys.executable
        predictor_script = os.path.join(_APP_BASE_DIR, "predictor_ai.py")
        work_dir = os.path.dirname(s.predictor_input_file)
        stdout_f = open(os.path.join(work_dir, "predict_stdout.log"), "w")
        stderr_f = open(os.path.join(work_dir, "predict_stderr.log"), "w")

        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
        env.setdefault("TF_NUM_INTEROP_THREADS", "1")
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        if s.train_window_minutes:
            env["TRAIN_WINDOW_MINUTES"] = str(s.train_window_minutes)
        else:
            env.pop("TRAIN_WINDOW_MINUTES", None)

        cmd = [python_exe, predictor_script, s.predictor_input_file]
        if os.name != "nt":
            cmd = ["nice", "-n", "10"] + cmd
        self.log(f"Starting predictor: {' '.join(cmd)}")
        s.prediction_process = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, cwd=work_dir, env=env)

    def poll_prediction_output(self):
        s = self.state
        if not s or not s.predictor_input_file:
            return
        out_file = os.path.join(os.path.dirname(s.predictor_input_file), "predict_out.csv")
        if not os.path.exists(out_file):
            return
        try:
            mtime = os.path.getmtime(out_file)
            if s.last_pred_out_mtime is not None and mtime <= s.last_pred_out_mtime:
                return
            s.last_pred_out_mtime = mtime
        except Exception:
            pass
        try:
            df = pd.read_csv(out_file)
            df["x"] = pd.to_datetime(df["x"])
            new_x = df["x"].tolist()
            new_y = df["y"].astype(float).tolist()
            if new_x:
                merged = {}
                for t, v in zip(s.predict_x, s.predict_y):
                    merged[t] = v
                for t, v in zip(new_x, new_y):
                    if t not in merged:
                        merged[t] = v
                merged_times = sorted(merged.keys())
                s.predict_x = merged_times
                s.predict_y = [merged[t] for t in merged_times]
            # anomalies require both predictions and realtime actuals
            self._detect_anomalies()
        except Exception:
            return

    def _detect_anomalies(self):
        s = self.state
        if not s or not s.new_x or not s.predict_x:
            return
        anomalies_df, _thr = s.anomaly_detector.detect_anomalies(
            actual_times=s.new_x,
            actual_values=s.new_y,
            predicted_times=s.predict_x,
            predicted_values=s.predict_y,
        )
        if anomalies_df is None or anomalies_df.empty:
            return
        new_times = pd.to_datetime(anomalies_df["time"]).tolist()
        if not new_times:
            return
        existing = {pd.to_datetime(t) for t in s.anomaly_times} if s.anomaly_times else set()
        max_anomalies = 1000
        for t in new_times:
            tt = pd.to_datetime(t)
            if tt in existing:
                continue
            if len(s.anomaly_times) >= max_anomalies:
                s.anomaly_times.pop(0)
            s.anomaly_times.append(tt)
            existing.add(tt)

    def redraw(self):
        s = self.state
        if not s or not s.dynamic_ax or not s.base_x:
            return

        # Blue
        if s.blue_line is None:
            self._init_lines()
        else:
            s.blue_line.set_data(s.base_x, self._delta(s.base_y))

        # Green
        if s.rt_x:
            if s.green_line is None:
                s.green_line = s.dynamic_ax.plot(s.rt_x, self._delta(s.rt_y), color=[0.1, 0.7, 0.2])[0]
            else:
                s.green_line.set_data(s.rt_x, self._delta(s.rt_y))

        # Purple
        if s.predict_x and s.predict_y:
            if s.purple_line is None:
                s.purple_line = s.dynamic_ax.plot(s.predict_x, self._delta(s.predict_y), color=[0.3, 0.1, 0.4])[0]
            else:
                s.purple_line.set_data(s.predict_x, self._delta(s.predict_y))

        # Anomaly vlines
        for v in s.anomaly_vlines_dyn:
            try:
                v.remove()
            except Exception:
                pass
        s.anomaly_vlines_dyn = []
        for t in s.anomaly_times:
            try:
                s.anomaly_vlines_dyn.append(s.dynamic_ax.axvline(x=t, color="lightcoral", linewidth=2, alpha=0.4))
            except Exception:
                pass

        try:
            s.dynamic_canvas.draw_idle()
            s.static_canvas.draw_idle()
        except Exception:
            pass


def main() -> int:
    app = QApplication([])
    win = MainWindow()
    Controller(win)
    win.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

