"""

Aim:
- Keep the overall UI and workflow of `application.py`, but use the project MySQL DB as the
  time-series source instead of USGS.
"""

from __future__ import annotations

import os
import re
import sys
import uuid
import subprocess
import math
from collections import deque
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

#
# IMPORTANT: `application.py` imports `data_convert_now.py` (USGS/requests).
# For the DB-based temp app we do not need USGS at all, and on some locked-down
# environments importing requests/cert bundles can fail. We therefore pre-seed
# `sys.modules['data_convert_now']` with a tiny stub so `application.py` can import,
# while `ApplicationWindowTemp.startThreads()` provides the real DB fetch path.
#
import types as _types

if "data_convert_now" not in sys.modules:
    _stub = _types.ModuleType("data_convert_now")

    def _stub_get_timeseries_magnetic_data(*args, **kwargs):  # pragma: no cover
        # Fallback signature; should not be used by ApplicationTemp.
        from data_convert_db_now import get_timeseries_magnetic_data

        return get_timeseries_magnetic_data(*args, **kwargs)

    _stub.get_timeseries_magnetic_data = _stub_get_timeseries_magnetic_data
    sys.modules["data_convert_now"] = _stub

# Avoid slow GUI startup and repeated cache rebuilds by forcing Matplotlib/font cache
# into writable project directories.
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

import application as base_app

from PyQt5 import Qt, QtCore
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTabWidget,
    QSplitter,
    QWidget,
    QVBoxLayout,
    QMessageBox,
    QInputDialog,
    QFileDialog,
    QDoubleSpinBox,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QDialogButtonBox,
    QScrollArea,
    QFrame,
)

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — used for projection="3d"

from Anomaly_detector import AnomalyDetector
from anomaly_direction import (
    is_obs1_sensor,
    is_obs2_sensor,
    triangulate_source_location,
)
from data_convert_db_now import (
    get_latest_sensor_ids,
    get_latest_sensor_id_like,
    get_min_timestamp_at_or_after,
    fetch_timeseries_window_multi,
    fetch_timeseries_between_multi,
    get_timeseries_magnetic_data_multi,
    get_timeseries_magnetic_data_since_multi,
)


def _sensor_sort_key(sensor_id: str) -> Tuple[int, int, str]:
    """
    Prefer ordering as OBS1_1..3 then OBS2_1..3 if the sensor_id matches that pattern.
    Fallback: keep stable ordering by sensor_id.
    """
    m = re.search(r"(OBS(\d+))_(\d+)$", sensor_id)
    if not m:
        return (99, 99, sensor_id)
    obs_num = int(m.group(2))
    sensor_num = int(m.group(3))
    return (obs_num, sensor_num, sensor_id)


def _sensor_display_name(sensor_id: str) -> str:
    m = re.search(r"(OBS\d+_\d+)$", sensor_id)
    return m.group(1) if m else sensor_id


# Default historic data window (minutes); user is prompted at startup to choose (see _configure_startup_mode).
# If available data is less than the chosen minutes, the app loads all of it (see on_db_data_updated).
# Display caps at this many points; training uses the full historic (and later historic+realtime) series.
HISTORIC_MINUTES = 60
HISTORIC_POINTS_1HZ = 60 * 60  # 3600 points at 1 Hz (default)


@dataclass
class SensorContext:
    sensor_id: str
    display_name: str

    # Data buffers
    # Historic snapshot (blue) loaded at startup (up to 60 min @ 1 Hz = 3600 points)
    base_x_t: List[datetime] = field(default_factory=list)
    base_y_mag_t: List[float] = field(default_factory=list)
    plot_baseline_nT: Optional[float] = None  # used only for visualization (ΔB = B - baseline)

    # Realtime stream (green) accumulated from incremental fetches
    rt_x_t: List[datetime] = field(default_factory=list)
    rt_y_mag_t: List[float] = field(default_factory=list)

    # Latest incremental chunk (for anomaly comparison and for "just arrived" UI updates)
    new_x_t: List[datetime] = field(default_factory=list)
    new_y_mag_t: List[float] = field(default_factory=list)

    has_seen_realtime: bool = False  # becomes True once we receive any incremental (green) data
    needs_update_lims: bool = False

    # Prediction buffers
    predict_x_t: List[datetime] = field(default_factory=list)
    predict_y_t: List[float] = field(default_factory=list)
    predictor_input_file: Optional[str] = None
    prediction_process: Optional[subprocess.Popen] = None
    predict_app_started: bool = False

    # Anomaly detection buffers
    anomaly_detector: AnomalyDetector = field(default_factory=lambda: AnomalyDetector(threshold_multiplier=2.5, min_samples_for_threshold=10))
    anomaly_times: List[datetime] = field(default_factory=list)
    anomaly_values: List[float] = field(default_factory=list)
    anomaly_vertical_lines: list = field(default_factory=list)
    anomaly_vertical_lines_static: list = field(default_factory=list)
    # Training cut-off: only data up to this timestamp has been compared with predictions
    # by the anomaly detector and is therefore "safe" for GRU training (with anomalies dropped).
    last_anomaly_checked_time: Optional[datetime] = None

    # UI refs (per-tab)
    static_canvas: Optional[FigureCanvas] = None
    dynamic_canvas: Optional[FigureCanvas] = None
    static_ax = None
    dynamic_ax = None
    static_line = None
    dynamic_line = None
    dynamic_new_line = None
    predictions_line = None

    # Settings per sensor
    train_window_minutes: Optional[int] = None

    # Low-pass filter state (per-sensor)
    last_filtered_value: Optional[float] = None

    # Performance throttles
    last_saved_points: int = 0
    last_redraw_points: int = 0
    last_pred_poll_ts: float = 0.0
    last_pred_start_ts: float = 0.0
    last_pred_complete_ts: float = 0.0


class MultiFetchWorker(QObject):
    finished = pyqtSignal()
    updated = pyqtSignal(dict, bool)  # (sensor_id -> df), new_flag

    def __init__(self, app=None):
        super().__init__()
        self._app = app

    def fetch_initial_sim(self, sensor_ids: List[str], start_time: datetime, end_time: datetime, last_n: int):
        try:
            dfs = fetch_timeseries_window_multi(sensor_ids, start_time=start_time, end_time=end_time, target_n_seconds=last_n)
        except Exception:
            dfs = {sid: pd.DataFrame(columns=["time_H", "mag_H_nT"]) for sid in sensor_ids}
        self.updated.emit(dfs, False)
        self.finished.emit()

    def fetch_incremental_sim(self, sensor_ids: List[str], start_time: datetime, end_time: datetime):
        try:
            dfs = fetch_timeseries_between_multi(sensor_ids, start_time=start_time, end_time=end_time, limit_rows=20000)
        except Exception:
            dfs = {sid: pd.DataFrame(columns=["time_H", "mag_H_nT"]) for sid in sensor_ids}
        self.updated.emit(dfs, True)
        self.finished.emit()

    def fetch_initial_realtime(self, sensor_ids: List[str], hours: float, last_n: int):
        try:
            dfs = get_timeseries_magnetic_data_multi(
                sensor_ids, hours=float(hours), last_n_samples=int(last_n)
            )
        except Exception:
            dfs = {sid: pd.DataFrame(columns=["time_H", "mag_H_nT"]) for sid in sensor_ids}
        self.updated.emit(dfs, False)
        self.finished.emit()

    def fetch_incremental_realtime(self, sensor_ids: List[str], since_times: Dict[str, datetime]):
        try:
            dfs = get_timeseries_magnetic_data_since_multi(
                sensor_ids, since_times=since_times, limit_rows=5000
            )
        except Exception:
            dfs = {sid: pd.DataFrame(columns=["time_H", "mag_H_nT"]) for sid in sensor_ids}
        self.updated.emit(dfs, True)
        self.finished.emit()

    def fetch_initial_csv(self, sensor_ids: List[str], start_time: datetime, end_time: datetime, last_n: int):
        try:
            if self._app is None:
                raise RuntimeError("CSV worker missing app reference")
            dfs = self._app._fetch_csv_window_multi(
                sensor_ids, start_time=start_time, end_time=end_time, target_n_seconds=last_n, incremental=False
            )
        except Exception:
            dfs = {sid: pd.DataFrame(columns=["time_H", "mag_H_nT"]) for sid in sensor_ids}
        self.updated.emit(dfs, False)
        self.finished.emit()

    def fetch_incremental_csv(self, sensor_ids: List[str], start_time: datetime, end_time: datetime):
        try:
            if self._app is None:
                raise RuntimeError("CSV worker missing app reference")
            dfs = self._app._fetch_csv_window_multi(
                sensor_ids, start_time=start_time, end_time=end_time, target_n_seconds=None, incremental=True
            )
        except Exception:
            dfs = {sid: pd.DataFrame(columns=["time_H", "mag_H_nT"]) for sid in sensor_ids}
        self.updated.emit(dfs, True)
        self.finished.emit()


class SensorMagTimeSeriesWidget(base_app.MagTimeSeriesWidget):
    """
    Reuse the existing UI widget, but route threshold/train-window changes
    to ALL sensor contexts (shared settings across all sensors).
    """

    def __init__(self, app, sensor_id: str, parent=None):
        self.sensor_id = sensor_id
        super().__init__(app, parent=parent)
        self._add_freeze_window_control()
        # Sync spinbox values from first sensor's context (all sensors share same values)
        self._sync_controls_from_context()
        # Register this widget so we can sync all widgets when one changes
        if not hasattr(app, '_sensor_control_widgets'):
            app._sensor_control_widgets = []
        app._sensor_control_widgets.append(self)

    def _add_freeze_window_control(self):
        ctx = self._ctx()
        try:
            freeze_layout = QHBoxLayout()
            freeze_label = QLabel("Anomaly Freeze Window (minutes):")
            freeze_layout.addWidget(freeze_label)

            self.freeze_window_spinbox = QDoubleSpinBox()
            self.freeze_window_spinbox.setMinimum(0)
            self.freeze_window_spinbox.setMaximum(180)
            self.freeze_window_spinbox.setSingleStep(1)
            self.freeze_window_spinbox.setDecimals(0)
            self.freeze_window_spinbox.setValue(float(getattr(ctx.anomaly_detector, "freeze_duration_minutes", 15)))
            self.freeze_window_spinbox.setToolTip("Freeze threshold after first anomaly for N minutes.")
            freeze_layout.addWidget(self.freeze_window_spinbox)
            self.freeze_window_spinbox.valueChanged.connect(self.on_freeze_window_changed)

            # Place below the training window status label
            self.gridLayout.addLayout(freeze_layout, 7, 0, 1, 5)
        except Exception:
            pass

    def _ctx(self) -> SensorContext:
        return self.app.sensor_ctx[self.sensor_id]

    def _sync_controls_from_context(self):
        """Initialize spinbox values from the first sensor's context (all sensors share same values)."""
        # Use the first sensor's context as the source of truth for shared settings
        if not self.app.sensor_ctx:
            return
        first_ctx = next(iter(self.app.sensor_ctx.values()))
        try:
            # Sync threshold multiplier
            if hasattr(self, 'threshold_spinbox'):
                self.threshold_spinbox.setValue(first_ctx.anomaly_detector.threshold_multiplier)
            # Sync training window
            if hasattr(self, 'train_window_spinbox'):
                if first_ctx.train_window_minutes is None or first_ctx.train_window_minutes <= 0:
                    self.train_window_spinbox.setValue(0)
                else:
                    self.train_window_spinbox.setValue(first_ctx.train_window_minutes)
            # Sync freeze window (already done in _add_freeze_window_control, but ensure it's set)
            if hasattr(self, 'freeze_window_spinbox'):
                freeze_val = getattr(first_ctx.anomaly_detector, "freeze_duration_minutes", 15)
                self.freeze_window_spinbox.setValue(float(freeze_val))
            # Update status label
            self.update_train_window_status()
        except Exception:
            pass  # If controls don't exist yet, ignore

    def on_threshold_changed(self, value):
        """Apply threshold multiplier change to ALL sensors (shared setting)."""
        old_values = {}
        for sid, ctx in self.app.sensor_ctx.items():
            old_values[sid] = ctx.anomaly_detector.threshold_multiplier
            ctx.anomaly_detector.threshold_multiplier = value
            # Re-run anomaly detection for this sensor
            self.app.detect_anomalies_for_sensor(sid)
            QTimer.singleShot(200, lambda s=sid: self.app.update_canvas_for_sensor(s))
        
        # Sync all other control widgets to show the new value (without triggering their callbacks)
        if hasattr(self.app, '_sensor_control_widgets'):
            for widget in self.app._sensor_control_widgets:
                if widget is not self and hasattr(widget, 'threshold_spinbox'):
                    # Temporarily block signals to avoid recursive updates
                    widget.threshold_spinbox.blockSignals(True)
                    widget.threshold_spinbox.setValue(value)
                    widget.threshold_spinbox.blockSignals(False)
        
        # Log change for all sensors
        sensor_names = ", ".join([ctx.display_name for ctx in self.app.sensor_ctx.values()])
        old_avg = sum(old_values.values()) / len(old_values) if old_values else value
        self.app.log(
            f'[All Sensors: {sensor_names}] Anomaly threshold multiplier changed {old_avg:.2f} -> {value:.2f} (shared setting)',
            level="Info"
        )

    def on_train_window_changed(self, value):
        """Apply training window change to ALL sensors (shared setting)."""
        train_minutes = None if value <= 0 else int(value)
        for sid, ctx in self.app.sensor_ctx.items():
            ctx.train_window_minutes = train_minutes
        
        # Sync all other control widgets to show the new value (without triggering their callbacks)
        if hasattr(self.app, '_sensor_control_widgets'):
            for widget in self.app._sensor_control_widgets:
                if widget is not self and hasattr(widget, 'train_window_spinbox'):
                    widget.train_window_spinbox.blockSignals(True)
                    widget.train_window_spinbox.setValue(value)
                    widget.train_window_spinbox.blockSignals(False)
                widget.update_train_window_status()  # Update status label for all widgets
        
        # Log change
        sensor_names = ", ".join([ctx.display_name for ctx in self.app.sensor_ctx.values()])
        if train_minutes is None:
            self.app.log(f'[All Sensors: {sensor_names}] Training window: full history (shared setting)', level="Info")
        else:
            self.app.log(f'[All Sensors: {sensor_names}] Training window: last {train_minutes} minutes (shared setting)', level="Info")
        
        # Update status label for this widget
        self.update_train_window_status()

    def on_freeze_window_changed(self, value):
        """Apply freeze window change to ALL sensors (shared setting)."""
        freeze_minutes = int(value)
        old_values = {}
        for sid, ctx in self.app.sensor_ctx.items():
            old_values[sid] = getattr(ctx.anomaly_detector, "freeze_duration_minutes", 15)
            ctx.anomaly_detector.freeze_duration_minutes = freeze_minutes
        
        # Sync all other control widgets to show the new value (without triggering their callbacks)
        if hasattr(self.app, '_sensor_control_widgets'):
            for widget in self.app._sensor_control_widgets:
                if widget is not self and hasattr(widget, 'freeze_window_spinbox'):
                    widget.freeze_window_spinbox.blockSignals(True)
                    widget.freeze_window_spinbox.setValue(float(freeze_minutes))
                    widget.freeze_window_spinbox.blockSignals(False)
                widget.update_train_window_status()  # Update status label for all widgets
        
        # Log change
        sensor_names = ", ".join([ctx.display_name for ctx in self.app.sensor_ctx.values()])
        old_avg = sum(old_values.values()) / len(old_values) if old_values else freeze_minutes
        self.app.log(
            f'[All Sensors: {sensor_names}] Anomaly freeze window changed {old_avg:.0f} -> {freeze_minutes:.0f} min (shared setting)',
            level="Info",
        )
        self.update_train_window_status()

    def update_train_window_status(self):
        """Update status label using first sensor's context (all sensors share same values)."""
        if not self.app.sensor_ctx:
            return
        first_ctx = next(iter(self.app.sensor_ctx.values()))
        minutes = first_ctx.train_window_minutes
        if minutes is None or minutes <= 0:
            window_text = "Training window: all data"
        else:
            window_text = f"Training window: last {int(minutes)} min"
        freeze_minutes = getattr(first_ctx.anomaly_detector, "freeze_duration_minutes", 15)
        self.train_window_status_label.setText(
            f"{window_text} | Freeze: {int(freeze_minutes)} min | Time-of-day features: on (daily sin/cos) | <i>(shared across all sensors)</i>"
        )


def _anomaly_direction_to_unit_vector(
    azimuth_deg: Optional[float], inclination_deg: Optional[float]
) -> Tuple[float, float, float]:
    """
    Convert azimuth and inclination (degrees) to a 3D unit vector from observatory center.
    Convention: 0° azimuth = Sensor 1 (East), 90° = Sensor 3; inclination = angle from horizontal (positive = up).
    Returns (x, y, z) with magnitude 1, pointing in the direction of the anomaly.
    """
    if inclination_deg is None:
        inclination_deg = 0.0
    inc_rad = math.radians(inclination_deg)
    if azimuth_deg is None:
        # Purely vertical
        return (0.0, 0.0, 1.0 if inclination_deg >= 0 else -1.0)
    az_rad = math.radians(azimuth_deg)
    x = math.cos(inc_rad) * math.cos(az_rad)
    y = math.cos(inc_rad) * math.sin(az_rad)
    z = math.sin(inc_rad)
    return (x, y, z)


class ApplicationWindowTemp(base_app.ApplicationWindow):
    def __init__(self, app, parent=None):
        super().__init__(app, parent=parent)
        self.framework_2_loaded = False
        self._fetch_thread: Optional[QThread] = None
        self._fetch_worker: Optional[MultiFetchWorker] = None
        self._temp_timeseries_container: Optional[QWidget] = None
        self._simplify_ui_for_multisensor()

    def _simplify_ui_for_multisensor(self):
        """
        Two-column layout:
        - LEFT half: sensor stream panels (3 panels, each scrollable L/R/U/D) — filled by load_plot_framework_2.
        - RIGHT half: selection parameters, direction coordinate plot, logs.
        """
        try:
            self.setMinimumSize(1100, 650)

            try:
                self.menuBar().hide()
            except Exception:
                pass
            try:
                self.statusbar.hide()
            except Exception:
                pass

            log_widget = self.textEditLog
            try:
                log_widget.setParent(None)
            except Exception:
                pass

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(0)

            # Horizontal splitter: left = sensor streams, right = parameters + direction + log
            h_splitter = QSplitter(QtCore.Qt.Horizontal)
            layout.addWidget(h_splitter)

            # ---- LEFT: placeholder for sensor stream panels (each scrollable)
            left_half = QWidget()
            left_layout = QVBoxLayout(left_half)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(6)
            left_half.setMinimumWidth(400)
            left_half.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            h_splitter.addWidget(left_half)
            self._temp_timeseries_container = left_half

            # ---- RIGHT: parameters placeholder + direction plot + log
            right_half = QWidget()
            right_layout = QVBoxLayout(right_half)
            right_layout.setContentsMargins(6, 6, 6, 6)
            right_layout.setSpacing(8)
            right_half.setMinimumWidth(320)
            right_half.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._right_half = right_half
            self._right_layout = right_layout

            # 1) Slot for shared parameters (load_plot_framework_2 will insert at index 0)
            # Keep a ref so we can insert the params panel; no placeholder widget to avoid blank area
            self._right_parameters_placeholder = None  # unused; we insert into _right_layout at 0

            # 2) Direction: label + 3D plot (larger size)
            direction_container = QWidget()
            direction_layout = QVBoxLayout(direction_container)
            direction_layout.setContentsMargins(0, 0, 0, 0)
            direction_layout.setSpacing(4)
            self._obs1_direction_label = QLabel("Last anomaly direction: —")
            self._obs1_direction_label.setStyleSheet("color: #555; font-size: 10px; padding: 2px 4px;")
            self._obs1_direction_label.setMaximumHeight(22)
            self._obs1_direction_label.setWordWrap(True)
            direction_layout.addWidget(self._obs1_direction_label)
            self._anomaly_direction_fig = Figure(figsize=(4.0, 4.0))
            self._anomaly_direction_canvas = FigureCanvas(self._anomaly_direction_fig)
            self._anomaly_direction_canvas.setMinimumSize(350, 350)
            self._anomaly_direction_canvas.setMaximumSize(450, 450)
            self._anomaly_direction_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._anomaly_direction_ax = self._anomaly_direction_fig.add_subplot(111, projection="3d")
            self._anomaly_direction_history: List[Tuple[Optional[float], Optional[float], float, str, Optional[datetime]]] = []
            self._draw_anomaly_direction_plot()
            direction_layout.addWidget(self._anomaly_direction_canvas, 1)  # stretch factor 1 to give it priority
            right_layout.addWidget(direction_container, 1)  # stretch factor 1 to give it priority over log

            # 3) Log — reduced height so parameters panel is visible; scrollable
            try:
                log_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                log_widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                log_widget.setMinimumHeight(60)
                log_widget.setMaximumHeight(90)  # Small fixed height; use scroll to see more
            except Exception:
                pass
            right_layout.addWidget(log_widget, 0)  # stretch 0 so it doesn't take space from parameters

            h_splitter.addWidget(right_half)
            h_splitter.setStretchFactor(0, 1)  # left
            h_splitter.setStretchFactor(1, 1)  # right
            # Initial sizes: left 50%, right 50% (in pixels, approximate)
            h_splitter.setSizes([550, 550])
        except Exception:
            self._temp_timeseries_container = None
            self._right_layout = getattr(self, "_right_layout", None)

    def update_obs1_direction(self, text: str) -> None:
        """Update the last-anomaly direction line (single setText, no timers)."""
        try:
            label = getattr(self, "_obs1_direction_label", None)
            if label is not None:
                label.setText("Last anomaly direction: " + text)
        except Exception:
            pass

    def _draw_anomaly_direction_plot(self) -> None:
        """
        Redraw the 3D anomaly direction plot, **showing only the latest OBS2 anomaly**
        as a single unit vector from the observatory center.
        """
        try:
            ax = getattr(self, "_anomaly_direction_ax", None)
            history = getattr(self, "_anomaly_direction_history", None)
            canvas = getattr(self, "_anomaly_direction_canvas", None)
            if ax is None or history is None or canvas is None:
                return

            ax.clear()

            # Center = observatory; equal aspect so unit vector is true direction
            r = 1.15
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
            ax.set_zlim(-r, r)
            ax.set_xlabel("E (East) →", fontsize=10, fontweight="bold")
            ax.set_ylabel("N (North) →", fontsize=10, fontweight="bold")
            ax.set_zlabel("Z (Up) ↑", fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)
            ax.set_title(
                "OBS2 Anomaly Direction (3D)\nCenter = Observatory 2",
                fontsize=11,
                fontweight="bold",
            )

            # Draw origin (observatory center) as a small sphere
            ax.scatter([0], [0], [0], color="k", s=30, alpha=0.9, marker="o", label="Observatory 2")

            # Draw OBS2 reference arrows only (declutter OBS1 from the UI)
            # OBS2: S1=W (-X), S2=S (-Y), S3=Z up (+Z)
            ref_length = 0.8
            ref_alpha = 0.5
            ref_width = 1.8

            ax.quiver(
                0,
                0,
                0,
                -ref_length,
                0,
                0,
                color="orange",
                alpha=ref_alpha,
                arrow_length_ratio=0.2,
                linewidth=ref_width,
                linestyle=":",
            )
            ax.text(-ref_length * 1.15, 0, 0, "OBS2\nS1=W", fontsize=9, color="orange", ha="right")

            ax.quiver(
                0,
                0,
                0,
                0,
                -ref_length,
                0,
                color="orange",
                alpha=ref_alpha,
                arrow_length_ratio=0.2,
                linewidth=ref_width,
                linestyle=":",
            )
            ax.text(0, -ref_length * 1.15, 0, "OBS2\nS2=S", fontsize=9, color="orange", ha="left")

            ax.quiver(
                0,
                0,
                0,
                0,
                0,
                ref_length,
                color="orange",
                alpha=ref_alpha,
                arrow_length_ratio=0.2,
                linewidth=ref_width,
                linestyle=":",
            )
            ax.text(0, 0, ref_length * 1.15, "OBS2\nS3=Z↑", fontsize=9, color="orange", ha="left")

            # Find the latest OBS2 anomaly in history
            latest_obs2_entry = None
            for entry in reversed(history):
                if len(entry) >= 4 and entry[3] == "OBS2":
                    latest_obs2_entry = entry
                    break

            if latest_obs2_entry is None:
                # No OBS2 anomalies yet: show a simple placeholder
                ax.text2D(
                    0.5,
                    0.5,
                    "No OBS2 anomaly yet",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                self._anomaly_direction_canvas.draw_idle()
                return

            # Unpack (support old tuple format without timestamp)
            if len(latest_obs2_entry) >= 5:
                azimuth_deg, inclination_deg, mag_nT, obs_label, timestamp = latest_obs2_entry
            else:
                azimuth_deg, inclination_deg, mag_nT, obs_label = latest_obs2_entry[:4]
                timestamp = None

            # If the stored entry does not have a real direction, do not draw an arrow
            if azimuth_deg is None and inclination_deg is None:
                ax.text2D(
                    0.5,
                    0.5,
                    "OBS2 direction unavailable",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                self._anomaly_direction_canvas.draw_idle()
                return

            # Compute unit vector and draw a **single** arrow for the latest OBS2 anomaly
            # For OBS2: 0° azimuth = S1 = West (-X), 90° = S2 = South (-Y)
            # So we need to negate x and y from standard conversion
            u_std, v_std, w = _anomaly_direction_to_unit_vector(azimuth_deg, inclination_deg)
            u, v = -u_std, -v_std  # Flip x and y for OBS2 coordinate system
            ax.quiver(
                0,
                0,
                0,
                u,
                v,
                w,
                color="red",
                alpha=0.95,
                arrow_length_ratio=0.3,
                linewidth=4.5,
            )

            # --- Optional ground-truth direction overlay (green arrow) for experiments ---
            # For the specified experimental run on 2026-02-16, the ground-truth inclination
            # is 0° (purely horizontal), with azimuth changing in fixed 2-minute intervals.
            # We map the anomaly timestamp to the corresponding ground-truth azimuth and
            # render it as a green unit vector from the observatory center.
            gt_azimuth_deg: Optional[float] = None
            try:
                # Only overlay ground-truth when running in CSV playback mode;
                # realtime/DB operation should not hard-code any experimental schedule.
                app = getattr(self, "_app", None)
                if app is not None and getattr(app, "csv_enabled", False) and timestamp is not None:
                    # Normalise to a Python datetime for comparison
                    if isinstance(timestamp, pd.Timestamp):
                        ts = timestamp.to_pydatetime()
                    else:
                        ts = timestamp
                    gt_azimuth_deg = app._ground_truth_azimuth_for_timestamp(ts)

                if gt_azimuth_deg is not None:
                    # Ground-truth inclination fixed at 0° (horizontal)
                    gt_u_std, gt_v_std, gt_w = _anomaly_direction_to_unit_vector(gt_azimuth_deg, 0.0)
                    gt_u, gt_v = -gt_u_std, -gt_v_std  # Same OBS2 frame flip as estimate
                    ax.quiver(
                        0,
                        0,
                        0,
                        gt_u,
                        gt_v,
                        gt_w,
                        color="green",
                        alpha=0.9,
                        arrow_length_ratio=0.3,
                        linewidth=3.0,
                        linestyle="--",
                    )
            except Exception:
                # Ground-truth overlay is optional; never break the main plot if it fails.
                pass

            # Add a compact text summary at the bottom
            if timestamp:
                if isinstance(timestamp, (datetime, pd.Timestamp)):
                    ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    ts_str = str(timestamp)
            else:
                ts_str = "N/A"

            az_str = f"{azimuth_deg:.1f}°" if azimuth_deg is not None else "N/A"
            inc_str = f"{inclination_deg:.1f}°" if inclination_deg is not None else "N/A"
            mag_str = f"{mag_nT:.1f} nT"

            ax.text2D(
                0.5,
                -0.08,
                f"Latest OBS2 anomaly | Time: {ts_str} | Az: {az_str} | Inc: {inc_str} | |ΔB|={mag_str}",
                transform=ax.transAxes,
                ha="center",
                fontsize=8,
            )

            self._anomaly_direction_canvas.draw_idle()
        except Exception:
            # Fail silently; direction plot is non-critical UI
            import traceback

            traceback.print_exc()
            pass

    def update_anomaly_direction_plot(
        self,
        azimuth_deg: Optional[float],
        inclination_deg: Optional[float],
        magnitude_nT: float = 0.0,
        obs_label: str = "",
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record this anomaly direction and redraw the plot with all recorded directions (kept visible)."""
        try:
            history = getattr(self, "_anomaly_direction_history", None)
            if history is None:
                return
            # Only append when we have a real direction (not the clear signal from non-OBS)
            if (azimuth_deg is not None or inclination_deg is not None) or obs_label:
                # Allow purely vertical (az None, inc set) or normal (az, inc)
                history.append((azimuth_deg, inclination_deg, magnitude_nT, obs_label, timestamp))
            self._draw_anomaly_direction_plot()
        except Exception:
            pass

    def startThreads(self, hours, start_time, new):
        # Replace USGS fetch threads with DB multi-sensor fetch.
        self._app._discover_sensors()
        if not getattr(self._app, "_startup_configured", True):
            return
        # Log which sensors will be fetched
        if self._app.sensor_ids:
            self._app.log(f"Fetching data for sensors: {', '.join(self._app.sensor_ids)}", level="Info")
        else:
            self._app.log("Warning: No sensors selected for data fetch", level="Warning")
        # Qt may delete the underlying C++ QThread after `deleteLater()`. If we keep
        # a Python reference, calling methods like isRunning() can raise:
        # "RuntimeError: wrapped C/C++ object of type QThread has been deleted".
        if self._fetch_thread is not None:
            try:
                if self._fetch_thread.isRunning():
                    return
            except RuntimeError:
                # Stale reference to a deleted thread; clear and continue.
                self._fetch_thread = None

        # Keep strong references to avoid PyQt GC destroying objects while running.
        thread = QThread(self)
        worker = MultiFetchWorker(app=self._app)
        worker.moveToThread(thread)

        if not new:
            if self._app.csv_enabled:
                # CSV: initial historic window (up to 60 min; or all data if less available).
                thread.started.connect(
                    lambda: worker.fetch_initial_csv(
                        self._app.sensor_ids,
                        self._app.sim_start_time,
                        self._app.sim_hist_end_time,
                        self._app.csv_hist_points,
                    )
                )
            elif self._app.sim_enabled:
                # Simulation: initial historic window (up to 60 min; or all if less).
                thread.started.connect(
                    lambda: worker.fetch_initial_sim(
                        self._app.sensor_ids,
                        self._app.sim_start_time,
                        self._app.sim_hist_end_time,
                        self._app.historic_points_1hz,
                    )
                )
            else:
                # Real-time: most recent historic window from DB (or all available if less).
                thread.started.connect(
                    lambda: worker.fetch_initial_realtime(
                        self._app.sensor_ids, hours=self._app.historic_minutes / 60.0, last_n=self._app.historic_points_1hz
                    )
                )
        else:
            if self._app.csv_enabled:
                # CSV: fetch next slice for simulated realtime playback.
                thread.started.connect(
                    lambda: worker.fetch_incremental_csv(
                        self._app.sensor_ids,
                        self._app.sim_rt_start_time,
                        self._app.sim_rt_end_time,
                    )
                )
            elif self._app.sim_enabled:
                # Simulation: fetch the next slice after the current simulated time.
                thread.started.connect(
                    lambda: worker.fetch_incremental_sim(
                        self._app.sensor_ids,
                        self._app.sim_rt_start_time,
                        self._app.sim_rt_end_time,
                    )
                )
            else:
                # Real-time: fetch new points since last known timestamps.
                thread.started.connect(
                    lambda: worker.fetch_incremental_realtime(
                        self._app.sensor_ids,
                        self._app.get_since_times(),
                    )
                )

        worker.updated.connect(self._app.on_db_data_updated)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: setattr(self, "_fetch_thread", None))
        thread.finished.connect(lambda: setattr(self, "_fetch_worker", None))

        self._fetch_thread = thread
        self._fetch_worker = worker
        thread.start()

    def updateData(self):
        # DB updates come via on_db_data_updated; keep compatibility.
        if not self.framework_2_loaded:
            if self._app.sensor_ctx and any(len(ctx.base_x_t) > 0 for ctx in self._app.sensor_ctx.values()):
                self._app.load_plot_framework_2()
                self.framework_2_loaded = True
        else:
            self._app.update_all_canvases()


class ApplicationTemp(base_app.Application):
    def __init__(self, arg):
        # Pre-init fields used by overridden startThreads/updateData
        self.sensor_ids: List[str] = []
        self.sensor_ctx: Dict[str, SensorContext] = {}
        self._time_series_tabs: Optional[QTabWidget] = None
        self._multi_data_timer: Optional[QTimer] = None
        self._multi_draw_timer: Optional[QTimer] = None
        self._multi_pred_timer: Optional[QTimer] = None
        self._slow_redraw_tick: int = 0
        self._predict_queue: deque[str] = deque()
        self._predict_active: set[str] = set()
        # Allow up to 3 predictor processes in parallel (one per sensor when 3 sensors are selected).
        # This keeps sensors independent while still avoiding unbounded TensorFlow concurrency.
        self._predict_max_concurrent: int = 3
        self._predict_sched_timer: Optional[QTimer] = None
        self._predict_cooldown_seconds: int = 20  # per-sensor minimum gap between predictor runs
        # Plotting semantics to mirror application.py:
        # - blue = historic snapshot (up to HISTORIC_MINUTES; less if that much not available)
        # - green = realtime accumulated
        self._rolling_window_points: int = HISTORIC_POINTS_1HZ
        self._predict_start_grace_seconds: int = 25  # if no realtime arrives, still start predictor after this delay

        # Simulation clock: start from 2026-01-05 and advance in fixed steps to simulate realtime.
        self.sim_enabled: bool = True
        self.sim_start_time: datetime = datetime(2026, 1, 5, 0, 0, 0)
        # Historic window length: HISTORIC_MINUTES. Downsample to 1 Hz, keep up to HISTORIC_POINTS_1HZ (or all if less).
        self.sim_hist_end_time: datetime = self.sim_start_time + timedelta(minutes=HISTORIC_MINUTES)
        self.sim_step_seconds: int = 20  # aligns with fetch timer cadence (every 20 sec)
        self.sim_rt_start_time: datetime = self.sim_hist_end_time
        self.sim_rt_end_time: datetime = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)
        self._startup_configured: bool = False
        self._initial_fetch_retry: bool = False
        self.csv_enabled: bool = False
        self.csv_path: Optional[str] = None
        self._csv_timeseries_by_sensor: Dict[str, pd.DataFrame] = {}
        self._csv_time_min: Optional[datetime] = None
        self._csv_time_max: Optional[datetime] = None
        self._csv_playback_complete: bool = False  # True when we've passed end of CSV and stopped fetch timer
        self.csv_hist_minutes: int = HISTORIC_MINUTES
        self.csv_hist_points: int = HISTORIC_POINTS_1HZ
        self.historic_minutes: int = HISTORIC_MINUTES  # user-configured initial load (minutes)
        # OBS1 direction finding: component series (time_H, S1_nT, S2_nT, S3_nT) and baseline (s1, s2, s3)
        self._obs1_components_df: Optional[pd.DataFrame] = None
        self._obs1_baseline: Optional[Tuple[float, float, float]] = None  # (S1, S2, S3) median in nT
        self._obs1_vector_by_sensor: Dict[str, pd.DataFrame] = {}
        self._obs1_sensor_baseline_xyz: Optional[Dict[str, Tuple[float, float, float]]] = None
        # OBS2 direction finding: component series (time_H, S1_nT, S2_nT, S3_nT) and baseline (s1, s2, s3)
        self._obs2_components_df: Optional[pd.DataFrame] = None
        self._obs2_baseline: Optional[Tuple[float, float, float]] = None  # (S1, S2, S3) median in nT
        self._obs2_vector_by_sensor: Dict[str, pd.DataFrame] = {}
        self._obs2_sensor_baseline_xyz: Optional[Dict[str, Tuple[float, float, float]]] = None
        # 3D direction workflow: interpolate each sensor's full vector at anomaly time,
        # baseline-correct per sensor, optionally rotate to observatory frame, then fuse.
        self._direction_interp_max_gap_seconds: float = 2.0
        # Identity defaults; adjust with calibration matrices when available.
        self._direction_sensor_rotations: Dict[str, np.ndarray] = {
            "OBS1_1": np.eye(3, dtype=float),
            "OBS1_2": np.eye(3, dtype=float),
            "OBS1_3": np.eye(3, dtype=float),
            "OBS2_1": np.eye(3, dtype=float),
            "OBS2_2": np.eye(3, dtype=float),
            "OBS2_3": np.eye(3, dtype=float),
        }
        # CSV-only supervised azimuth calibration (trained from hardcoded GT windows).
        self._obs2_calibration_enabled: bool = True
        self._obs2_calibration_model: Optional[Dict[str, np.ndarray]] = None
        self._obs2_calibration_conf_threshold: float = 0.20
        self._obs2_calibration_model_path: str = os.path.join(
            os.path.dirname(base_app.APP_BASE), "models", "obs2_direction_calibration_3d_v1.npz"
        )
        self._obs2_calibration_loaded_from_disk: bool = False
        self._obs2_calibration_attempted: bool = False
        # Triangulation: observatory positions (in meters, relative coordinate system)
        # OBS1 at origin, OBS2 100m away along x-axis
        self._obs1_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # meters
        self._obs2_position: Tuple[float, float, float] = (100.0, 0.0, 0.0)  # meters (100m apart)
        # Recent anomalies with directions for triangulation (time -> (azimuth, inclination, magnitude))
        self._obs1_recent_anomalies: Dict[datetime, Tuple[Optional[float], Optional[float], float]] = {}
        self._obs2_recent_anomalies: Dict[datetime, Tuple[Optional[float], Optional[float], float]] = {}
        self._triangulation_time_window: timedelta = timedelta(minutes=2)  # Match anomalies within 2 minutes
        self.historic_points_1hz: int = HISTORIC_POINTS_1HZ  # historic_minutes * 60
        self._selected_sensor_ids: Optional[List[str]] = None

        # Low-pass filter configuration (simple exponential moving average)
        # alpha close to 0 => heavier smoothing, close to 1 => lighter smoothing.
        self._lowpass_alpha: float = 0.2

        # Use base init (VTK, map, menus, etc.)
        super().__init__(arg)
        self._configure_startup_mode()
        self._startup_configured = True
        # Base __init__ triggers an initial startThreads() call. Ensure we actually
        # fetch initial data for the discovered sensors (in case the first call
        # happened before discovery or before startup selection).
        try:
            self.appWin.startThreads(hours=1, start_time=None, new=False)
        except Exception:
            pass

    def _configure_startup_mode(self):
        # Ask user how much historic data (minutes) to load initially
        minutes, ok = QInputDialog.getInt(
            self.appWin,
            "Initial Historic Data",
            "How many minutes of historic data should be loaded initially?",
            value=self.historic_minutes,
            min=1,
            max=10080,  # up to 1 week
        )
        if ok and minutes >= 1:
            self.historic_minutes = minutes
            self.historic_points_1hz = self.historic_minutes * 60
            self._rolling_window_points = self.historic_points_1hz
            self.sim_hist_end_time = self.sim_start_time + timedelta(minutes=self.historic_minutes)
            self.csv_hist_minutes = self.historic_minutes
            self.csv_hist_points = self.historic_points_1hz
            self.log(f"Initial historic window: {self.historic_minutes} minutes.", level="Info")
        # else keep defaults set in __init__

        msg = (
            "Choose data source:\n\n"
            "Real-time: load the latest historic window\n"
            "Simulation: choose a start date (date only)\n"
            "CSV file: load a pre-downloaded CSV from disk"
        )
        box = QMessageBox(self.appWin)
        box.setWindowTitle("Data Source")
        box.setText(msg)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Yes)
        try:
            box.button(QMessageBox.Yes).setText("Real-time")
            box.button(QMessageBox.No).setText("Simulation")
            box.button(QMessageBox.Cancel).setText("CSV File")
        except Exception:
            pass
        choice = box.exec()

        if choice == QMessageBox.Cancel:
            path = self._prompt_for_csv_path()
            if path:
                if self._load_csv_source(path):
                    self.csv_enabled = True
                    self.sim_enabled = True
                    if self._csv_time_min:
                        self.sim_start_time = self._csv_time_min
                        self._init_csv_times()
                    self._configure_sensor_selection()
                    self.log(f"Data mode selected: CSV file ({os.path.basename(path)}).", level="Info")
                    return
                else:
                    QMessageBox.warning(self.appWin, "CSV Load Failed", "Falling back to real-time mode.")
            self.sim_enabled = False
            self.log("Data mode selected: real-time (last {} minutes).".format(self.historic_minutes), level="Info")
            return

        if choice == QMessageBox.Yes:
            self.sim_enabled = False
            self.log("Data mode selected: real-time (last {} minutes).".format(self.historic_minutes), level="Info")
            self._configure_sensor_selection()
            return

        while True:
            default_str = self.sim_start_time.strftime("%Y-%m-%d")
            text, ok = QInputDialog.getText(
                self.appWin,
                "Simulation Start Date",
                "Enter start date (e.g., 2025-10-10 or 10 Oct 2025):",
                text=default_str,
            )
            if not ok:
                # If cancelled, fall back to real-time
                self.sim_enabled = False
                self.log("Data mode selected: real-time (last {} minutes).".format(self.historic_minutes), level="Info")
                return

            raw = text.strip()
            if not raw:
                QMessageBox.warning(self.appWin, "Invalid Date", "Please enter a valid date.")
                continue

            dt = None
            for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    break
                except Exception:
                    continue
            if dt is None:
                QMessageBox.warning(self.appWin, "Invalid Date", "Use YYYY-MM-DD or e.g., 10 Oct 2025.")
                continue

            self.sim_enabled = True
            self.sim_start_time = dt
            self._init_sim_times()
            self.log(f"Data mode selected: simulation from {self.sim_start_time}.", level="Info")
            self._configure_sensor_selection()
            return

    def _configure_sensor_selection(self):
        if self.csv_enabled:
            csv_sensors = sorted(self._csv_timeseries_by_sensor.keys(), key=_sensor_sort_key)
            if not csv_sensors:
                return
            selected = self._prompt_for_sensor_selection(csv_sensors)
            # Limit to at most 3 sensors.
            if selected:
                if len(selected) > 3:
                    self.log(
                        f"More than 3 sensors selected; using first 3: {', '.join(selected[:3])}",
                        level="Warning",
                    )
                    selected = selected[:3]
            else:
                selected = csv_sensors[:3]
            self._selected_sensor_ids = selected
            # Clear existing sensor_ids to force rediscovery with new selection
            self.sensor_ids = []
            return
        try:
            default_ids = get_latest_sensor_ids(limit=6)
        except Exception:
            default_ids = []
        if default_ids:
            selected = self._prompt_for_sensor_selection(default_ids)
            if selected:
                if len(selected) > 3:
                    self.log(
                        f"More than 3 sensors selected; using first 3: {', '.join(selected[:3])}",
                        level="Warning",
                    )
                    selected = selected[:3]
            else:
                selected = default_ids[:3]
            self._selected_sensor_ids = selected
            # Clear existing sensor_ids to force rediscovery with new selection
            self.sensor_ids = []
            return
        default_text = "OBS1_1, OBS1_2, OBS1_3, OBS2_1, OBS2_2, OBS2_3"
        text, ok = QInputDialog.getText(
            self.appWin,
            "Select Sensors",
            "Enter sensor IDs to plot (comma-separated):",
            text=default_text,
        )
        if not ok:
            self._selected_sensor_ids = None
            self.sensor_ids = []
            return
        raw = text.strip()
        if not raw:
            self._selected_sensor_ids = None
            self.sensor_ids = []
            return
        ids = [s.strip() for s in raw.split(",") if s.strip()]
        if ids and len(ids) > 3:
            self.log(
                f"More than 3 sensors entered; using first 3: {', '.join(ids[:3])}",
                level="Warning",
            )
            ids = ids[:3]
        self._selected_sensor_ids = ids if ids else None
        # Clear existing sensor_ids to force rediscovery with new selection
        self.sensor_ids = []

    def _prompt_for_sensor_selection(self, sensor_ids: List[str]) -> List[str]:
        dlg = QDialog(self.appWin)
        dlg.setWindowTitle("Select Sensors")
        layout = QVBoxLayout()
        dlg.setLayout(layout)

        label = QLabel("Select up to 3 sensors to plot:")
        layout.addWidget(label)

        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.MultiSelection)
        for sid in sensor_ids:
            item = QListWidgetItem(sid)
            item.setSelected(True)
            list_widget.addItem(item)
        layout.addWidget(list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return []
        selected = [item.text() for item in list_widget.selectedItems()]
        return selected

    def _prompt_for_csv_path(self) -> Optional[str]:
        try:
            path, _ = QFileDialog.getOpenFileName(
                self.appWin,
                "Select Magnetic CSV File",
                os.path.dirname(base_app.APP_BASE),
                "CSV Files (*.csv);;All Files (*)",
            )
            return path if path else None
        except Exception:
            return None

    def _derive_base_date_from_filename(self, path: str) -> Optional[datetime]:
        name = os.path.basename(path)
        m = re.search(r"magnetic_data_(\d{8})_\d{6}_to_(\d{8})_\d{6}", name)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), "%Y%m%d")
        except Exception:
            return None

    def _parse_csv_timestamps(self, series: pd.Series, path: str) -> pd.Series:
        ts = pd.to_datetime(series, errors="coerce")
        valid_ratio = float(ts.notna().mean()) if len(ts) else 0.0
        if valid_ratio >= 0.5:
            return ts

        base_date = self._derive_base_date_from_filename(path)
        if base_date is None:
            base_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        td = pd.to_timedelta(series, errors="coerce")
        if td.notna().any():
            return pd.to_datetime(base_date) + td

        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            return pd.to_datetime(base_date) + pd.to_timedelta(numeric, unit="s", errors="coerce")

        return ts

    def _csv_raw_to_timeseries_df(self, df_raw: pd.DataFrame, path: str) -> pd.DataFrame:
        if df_raw is None or df_raw.empty:
            return pd.DataFrame(columns=["time_H", "mag_H_nT"])

        df = df_raw.copy()
        if "timestamp" not in df.columns:
            return pd.DataFrame(columns=["time_H", "mag_H_nT"])

        df["timestamp"] = self._parse_csv_timestamps(df["timestamp"], path)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=True).reset_index(drop=True)

        for c in ("b_x", "b_y", "b_z"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if not all(c in df.columns for c in ("b_x", "b_y", "b_z")):
            return pd.DataFrame(columns=["time_H", "mag_H_nT"])

        df["mag_total_nT"] = (df["b_x"] ** 2 + df["b_y"] ** 2 + df["b_z"] ** 2) ** 0.5
        df = df.dropna(subset=["mag_total_nT"])
        df["time_H"] = df["timestamp"].dt.floor("s")
        grouped = (
            df.groupby("time_H", as_index=False)["mag_total_nT"]
            .mean()
            .rename(columns={"mag_total_nT": "mag_H_nT"})
            .sort_values("time_H", ascending=True)
            .reset_index(drop=True)
        )
        return grouped[["time_H", "mag_H_nT"]]

    @staticmethod
    def _obs_sensor_suffix(sensor_id: str) -> Optional[str]:
        """Return canonical OBS suffix like OBS2_1 from full sensor_id."""
        m = re.search(r"(OBS\d+_\d+)$", str(sensor_id))
        return m.group(1) if m else None

    def _build_obs_vector_series(
        self, df_raw: pd.DataFrame, path: str, observatory: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Build per-sensor full-vector time series for an observatory.

        Returns
        -------
        dict
            Keys are canonical suffixes (e.g. OBS2_1), values are dataframes with:
            time_H, b_x, b_y, b_z, _ts_ns.
        """
        out: Dict[str, pd.DataFrame] = {}
        if df_raw is None or df_raw.empty or "sensor_id" not in df_raw.columns:
            return out
        if "timestamp" not in df_raw.columns:
            return out
        for c in ("b_x", "b_y", "b_z"):
            if c not in df_raw.columns:
                return {}

        sensor_ids = df_raw["sensor_id"].astype(str).unique().tolist()
        for idx in (1, 2, 3):
            marker = f"{observatory}_{idx}"
            matches = [sid for sid in sensor_ids if marker in sid]
            if not matches:
                return {}
            sid = sorted(matches)[0]

            df_s = df_raw[df_raw["sensor_id"] == sid][["timestamp", "b_x", "b_y", "b_z"]].copy()
            df_s["time_H"] = pd.to_datetime(self._parse_csv_timestamps(df_s["timestamp"], path))
            df_s = df_s.drop(columns=["timestamp"])
            for c in ("b_x", "b_y", "b_z"):
                df_s[c] = pd.to_numeric(df_s[c], errors="coerce")
            df_s = df_s.dropna(subset=["time_H", "b_x", "b_y", "b_z"]).sort_values("time_H")
            # Duplicate timestamps can occur in some exports; average them before interpolation.
            df_s = (
                df_s.groupby("time_H", as_index=False)[["b_x", "b_y", "b_z"]]
                .mean()
                .sort_values("time_H")
                .reset_index(drop=True)
            )
            if df_s.empty:
                return {}
            df_s["_ts_ns"] = pd.to_datetime(df_s["time_H"]).astype("int64")
            out[marker] = df_s[["time_H", "b_x", "b_y", "b_z", "_ts_ns"]].copy()

        return out

    def _interpolate_sensor_vector_at_time(
        self, df_sensor: pd.DataFrame, t: datetime
    ) -> Optional[Tuple[float, float, float]]:
        """
        Interpolate b_x, b_y, b_z at time t from a single sensor dataframe.

        Uses linear interpolation between nearest bracketing samples and rejects gaps
        larger than `_direction_interp_max_gap_seconds`.
        """
        if df_sensor is None or df_sensor.empty:
            return None
        ts_ns = int(pd.Timestamp(t).value)
        arr_ns = df_sensor["_ts_ns"].to_numpy(dtype=np.int64)
        if arr_ns.size == 0:
            return None
        if ts_ns < int(arr_ns[0]) or ts_ns > int(arr_ns[-1]):
            return None

        pos = int(np.searchsorted(arr_ns, ts_ns, side="left"))
        # Exact hit
        if pos < len(arr_ns) and int(arr_ns[pos]) == ts_ns:
            row = df_sensor.iloc[pos]
            return float(row["b_x"]), float(row["b_y"]), float(row["b_z"])

        if pos <= 0 or pos >= len(arr_ns):
            return None

        left_idx = pos - 1
        right_idx = pos
        left_ns = int(arr_ns[left_idx])
        right_ns = int(arr_ns[right_idx])
        dt_ns = right_ns - left_ns
        if dt_ns <= 0:
            return None
        gap_sec = float(dt_ns) / 1e9
        if gap_sec > float(self._direction_interp_max_gap_seconds):
            return None

        alpha = float(ts_ns - left_ns) / float(dt_ns)
        left = df_sensor.iloc[left_idx]
        right = df_sensor.iloc[right_idx]
        bx = float(left["b_x"]) + alpha * (float(right["b_x"]) - float(left["b_x"]))
        by = float(left["b_y"]) + alpha * (float(right["b_y"]) - float(left["b_y"]))
        bz = float(left["b_z"]) + alpha * (float(right["b_z"]) - float(left["b_z"]))
        return bx, by, bz

    def _fuse_observatory_delta_vector(
        self, observatory: str, t: datetime
    ) -> Optional[Tuple[float, float, float]]:
        """
        Compute fused perturbation vector (dX, dY, dZ) at time t for an observatory.

        Workflow:
        1) Interpolate each sensor's (b_x, b_y, b_z) to the same timestamp t.
        2) Subtract that sensor's own baseline vector.
        3) Rotate with optional sensor calibration matrix.
        4) Average the three vectors.
        """
        obs = observatory.upper()
        if obs == "OBS1":
            vec_by_sensor = self._obs1_vector_by_sensor
            baseline_xyz = self._obs1_sensor_baseline_xyz
        elif obs == "OBS2":
            vec_by_sensor = self._obs2_vector_by_sensor
            baseline_xyz = self._obs2_sensor_baseline_xyz
        else:
            return None

        if not vec_by_sensor or baseline_xyz is None:
            return None

        fused = np.zeros(3, dtype=float)
        count = 0
        for idx in (1, 2, 3):
            key = f"{obs}_{idx}"
            df_sensor = vec_by_sensor.get(key)
            b0 = baseline_xyz.get(key) if baseline_xyz else None
            if df_sensor is None or b0 is None:
                return None
            sample = self._interpolate_sensor_vector_at_time(df_sensor, t)
            if sample is None:
                return None

            db_local = np.array(
                [
                    float(sample[0] - b0[0]),
                    float(sample[1] - b0[1]),
                    float(sample[2] - b0[2]),
                ],
                dtype=float,
            )
            rot = self._direction_sensor_rotations.get(key)
            if rot is None or getattr(rot, "shape", None) != (3, 3):
                rot = np.eye(3, dtype=float)
            db_obs = rot.dot(db_local)
            fused += db_obs
            count += 1

        if count <= 0:
            return None
        fused /= float(count)
        return float(fused[0]), float(fused[1]), float(fused[2])

    @staticmethod
    def _direction_from_delta_vector(
        d_x: float, d_y: float, d_z: float
    ) -> Tuple[Optional[float], Optional[float], float]:
        """
        Convert a perturbation vector to azimuth/inclination and magnitude.

        Azimuth convention:
        - 0 deg along +X, 90 deg along +Y.
        Inclination:
        - positive upward (+Z), negative downward (-Z).
        """
        mag = math.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)
        eps = 1e-6
        if mag < eps:
            return None, None, 0.0

        h = math.sqrt(d_x * d_x + d_y * d_y)
        if h < eps:
            inc = 90.0 if d_z > 0 else -90.0
            return None, inc, mag

        az = math.degrees(math.atan2(d_y, d_x))
        if az < 0:
            az += 360.0
        inc = math.degrees(math.atan2(d_z, h))
        return az, inc, mag

    @staticmethod
    def _obs2_ground_truth_schedule_20260216() -> List[Tuple[datetime, datetime, float]]:
        """Hardcoded OBS2 ground-truth azimuth schedule (global frame) for 2026-02-16."""
        d = datetime(2026, 2, 16)
        return [
            (datetime(d.year, d.month, d.day, 15, 18, 0), datetime(d.year, d.month, d.day, 15, 20, 0), 45.0),
            (datetime(d.year, d.month, d.day, 15, 20, 1), datetime(d.year, d.month, d.day, 15, 22, 0), 90.0),
            (datetime(d.year, d.month, d.day, 15, 22, 1), datetime(d.year, d.month, d.day, 15, 24, 0), 135.0),
            (datetime(d.year, d.month, d.day, 15, 24, 1), datetime(d.year, d.month, d.day, 15, 26, 0), 180.0),
            (datetime(d.year, d.month, d.day, 15, 26, 1), datetime(d.year, d.month, d.day, 15, 28, 0), 225.0),
            (datetime(d.year, d.month, d.day, 15, 28, 1), datetime(d.year, d.month, d.day, 15, 30, 0), 270.0),
            (datetime(d.year, d.month, d.day, 15, 30, 1), datetime(d.year, d.month, d.day, 15, 32, 0), 315.0),
        ]

    def _ground_truth_azimuth_for_timestamp(self, ts: datetime) -> Optional[float]:
        """Return hardcoded OBS2 ground-truth azimuth (global frame), if timestamp is in schedule."""
        if ts.date() != datetime(2026, 2, 16).date():
            return None
        for start_t, end_t, az_deg in self._obs2_ground_truth_schedule_20260216():
            if start_t <= ts <= end_t:
                return az_deg
        return None

    def _load_obs2_calibration_model_from_disk(self) -> bool:
        """Load persisted OBS2 calibration model (CSV mode) if available."""
        path = getattr(self, "_obs2_calibration_model_path", None)
        if not path or not os.path.exists(path):
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                mu = np.asarray(data["mu"], dtype=float)
                sigma = np.asarray(data["sigma"], dtype=float)
                w = np.asarray(data["w"], dtype=float)
            if mu.ndim != 1 or sigma.ndim != 1 or w.ndim != 2:
                return False
            if mu.shape != sigma.shape:
                return False
            if w.shape[0] != (mu.shape[0] + 1) or w.shape[1] != 2:
                return False
            self._obs2_calibration_model = {"mu": mu, "sigma": sigma, "w": w}
            self._obs2_calibration_loaded_from_disk = True
            self._obs2_calibration_attempted = True
            self.log(
                f"[OBS2] Loaded persisted CSV azimuth calibration model: {path}",
                level="Info",
            )
            return True
        except Exception:
            return False

    def _save_obs2_calibration_model_to_disk(self) -> bool:
        """Persist OBS2 calibration model so it can be reused with other CSV files."""
        model = self._obs2_calibration_model
        if model is None:
            return False
        path = getattr(self, "_obs2_calibration_model_path", None)
        if not path:
            return False
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(
                path,
                mu=np.asarray(model["mu"], dtype=float),
                sigma=np.asarray(model["sigma"], dtype=float),
                w=np.asarray(model["w"], dtype=float),
            )
            self.log(
                f"[OBS2] Saved CSV azimuth calibration model: {path}",
                level="Info",
            )
            return True
        except Exception:
            return False

    def _obs2_delta_components_at_time(
        self, t: datetime
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Interpolate and baseline-correct OBS2 vectors per sensor at time t.

        Returns dict with OBS2_1/OBS2_2/OBS2_3 -> np.array([dX, dY, dZ]) in observatory frame.
        """
        if self._obs2_sensor_baseline_xyz is None or not self._obs2_vector_by_sensor:
            return None
        out: Dict[str, np.ndarray] = {}
        for idx in (1, 2, 3):
            key = f"OBS2_{idx}"
            df_sensor = self._obs2_vector_by_sensor.get(key)
            b0 = self._obs2_sensor_baseline_xyz.get(key) if self._obs2_sensor_baseline_xyz else None
            if df_sensor is None or b0 is None:
                return None
            sample = self._interpolate_sensor_vector_at_time(df_sensor, t)
            if sample is None:
                return None
            db_local = np.array(
                [
                    float(sample[0] - b0[0]),
                    float(sample[1] - b0[1]),
                    float(sample[2] - b0[2]),
                ],
                dtype=float,
            )
            rot = self._direction_sensor_rotations.get(key)
            if rot is None or getattr(rot, "shape", None) != (3, 3):
                rot = np.eye(3, dtype=float)
            out[key] = rot.dot(db_local)
        return out

    def _obs2_calibration_feature_vector(self, t: datetime) -> Optional[np.ndarray]:
        """
        Build CSV-only supervised feature vector for OBS2 azimuth calibration at timestamp t.

        Features use all components from all sensors + cross-component correlations.
        """
        deltas = self._obs2_delta_components_at_time(t)
        if deltas is None:
            return None

        d1 = deltas["OBS2_1"]
        d2 = deltas["OBS2_2"]
        d3 = deltas["OBS2_3"]
        base = np.array(
            [
                d1[0], d1[1], d1[2],
                d2[0], d2[1], d2[2],
                d3[0], d3[1], d3[2],
            ],
            dtype=float,
        )
        fused = (d1 + d2 + d3) / 3.0
        mags = np.array(
            [
                float(np.linalg.norm(d1)),
                float(np.linalg.norm(d2)),
                float(np.linalg.norm(d3)),
            ],
            dtype=float,
        )
        corr_terms: List[float] = []
        for i in range(len(base)):
            for j in range(i + 1, len(base)):
                corr_terms.append(float(base[i] * base[j]))
        return np.concatenate([base, fused, mags, np.asarray(corr_terms, dtype=float)], axis=0)

    def _ensure_obs2_calibration_model(self) -> None:
        """
        Train CSV-only OBS2 azimuth calibration model from hardcoded ground-truth windows.

        Model predicts (cos(az), sin(az)) from full 3D vector/correlation features.
        """
        if not self.csv_enabled or not self._obs2_calibration_enabled:
            return
        if self._obs2_calibration_model is not None:
            return
        if self._obs2_calibration_attempted:
            return
        # If a persisted model exists, prefer reusing it.
        if self._load_obs2_calibration_model_from_disk():
            return
        if self._obs2_sensor_baseline_xyz is None:
            return

        x_rows: List[np.ndarray] = []
        az_labels: List[float] = []
        for start_t, end_t, gt_az_deg in self._obs2_ground_truth_schedule_20260216():
            for ts in pd.date_range(start=start_t, end=end_t, freq="1s"):
                feat = self._obs2_calibration_feature_vector(ts.to_pydatetime())
                if feat is None:
                    continue
                x_rows.append(feat)
                az_labels.append(float(gt_az_deg))

        if len(x_rows) < 30:
            self.log(
                f"[OBS2] CSV azimuth calibration skipped: insufficient GT samples ({len(x_rows)}).",
                level="Warning",
            )
            self._obs2_calibration_attempted = True
            return

        x = np.vstack(x_rows).astype(float)
        az = np.asarray(az_labels, dtype=float)
        mu = x.mean(axis=0)
        sigma = x.std(axis=0)
        sigma = np.where(sigma < 1e-9, 1.0, sigma)
        x_n = (x - mu) / sigma
        x_aug = np.column_stack([x_n, np.ones((x_n.shape[0], 1), dtype=float)])

        az_rad = np.radians(az)
        y = np.column_stack([np.cos(az_rad), np.sin(az_rad)])
        lam = 1.0
        eye = np.eye(x_aug.shape[1], dtype=float)
        eye[-1, -1] = 0.0  # do not regularize bias
        try:
            w = np.linalg.solve(x_aug.T.dot(x_aug) + lam * eye, x_aug.T.dot(y))
        except Exception:
            self.log("[OBS2] CSV azimuth calibration skipped: solver failure.", level="Warning")
            self._obs2_calibration_attempted = True
            return

        pred_vec = x_aug.dot(w)
        pred_az = np.degrees(np.arctan2(pred_vec[:, 1], pred_vec[:, 0]))
        pred_az = np.where(pred_az < 0.0, pred_az + 360.0, pred_az)
        err = ((pred_az - az + 180.0) % 360.0) - 180.0
        mae = float(np.mean(np.abs(err)))

        self._obs2_calibration_model = {
            "mu": mu.astype(float),
            "sigma": sigma.astype(float),
            "w": w.astype(float),
        }
        self._obs2_calibration_loaded_from_disk = False
        self._obs2_calibration_attempted = True
        self.log(
            f"[OBS2] CSV azimuth calibration ready | samples={len(x_rows)} | train circular MAE={mae:.1f}°",
            level="Info",
        )
        self._save_obs2_calibration_model_to_disk()

    def _predict_obs2_calibrated_azimuth(
        self, t: datetime
    ) -> Optional[Tuple[float, float]]:
        """
        Predict calibrated OBS2 azimuth (global frame) and confidence at timestamp t.

        Confidence is the norm of predicted [cos, sin] vector.
        """
        model = self._obs2_calibration_model
        if model is None:
            return None
        feat = self._obs2_calibration_feature_vector(t)
        if feat is None:
            return None
        mu = model["mu"]
        sigma = model["sigma"]
        w = model["w"]

        x_n = (feat - mu) / sigma
        x_aug = np.concatenate([x_n, np.asarray([1.0], dtype=float)])
        pred_vec = x_aug.dot(w)
        conf = float(np.linalg.norm(pred_vec))
        if conf < 1e-9:
            return None
        az = math.degrees(math.atan2(float(pred_vec[1]), float(pred_vec[0])))
        if az < 0:
            az += 360.0
        return float(az), conf

    def _build_obs1_components_df(self, df_raw: pd.DataFrame, path: str) -> Optional[pd.DataFrame]:
        """
        Build aligned (time_H, S1_nT, S2_nT, S3_nT) for OBS1 from raw CSV.
        S1 = b_x from sensor OBS1_1 (horizontal), S2 = b_y from OBS1_2 (vertical), S3 = b_z from OBS1_3 (horizontal).
        """
        if df_raw is None or df_raw.empty or "sensor_id" not in df_raw.columns:
            return None
        for c in ("b_x", "b_y", "b_z", "timestamp"):
            if c not in df_raw.columns:
                return None
        ids_1 = [s for s in df_raw["sensor_id"].astype(str).unique() if "OBS1_1" in s]
        ids_2 = [s for s in df_raw["sensor_id"].astype(str).unique() if "OBS1_2" in s]
        ids_3 = [s for s in df_raw["sensor_id"].astype(str).unique() if "OBS1_3" in s]
        if not ids_1 or not ids_2 or not ids_3:
            return None
        df1 = (
            df_raw[df_raw["sensor_id"].isin(ids_1)][["timestamp", "b_x"]]
            .copy()
            .rename(columns={"b_x": "S1_nT"})
        )
        df1["time_H"] = pd.to_datetime(self._parse_csv_timestamps(df1["timestamp"], path))
        df1 = df1.dropna(subset=["time_H"]).sort_values("time_H").drop(columns=["timestamp"])
        df2 = (
            df_raw[df_raw["sensor_id"].isin(ids_2)][["timestamp", "b_y"]]
            .copy()
            .rename(columns={"b_y": "S2_nT"})
        )
        df2["time_H"] = pd.to_datetime(self._parse_csv_timestamps(df2["timestamp"], path))
        df2 = df2.dropna(subset=["time_H"]).sort_values("time_H").drop(columns=["timestamp"])
        df3 = (
            df_raw[df_raw["sensor_id"].isin(ids_3)][["timestamp", "b_z"]]
            .copy()
            .rename(columns={"b_z": "S3_nT"})
        )
        df3["time_H"] = pd.to_datetime(self._parse_csv_timestamps(df3["timestamp"], path))
        df3 = df3.dropna(subset=["time_H"]).sort_values("time_H").drop(columns=["timestamp"])
        tol = pd.Timedelta("2s")
        merged = pd.merge_asof(df1, df2, on="time_H", direction="nearest", tolerance=tol)
        merged = pd.merge_asof(merged, df3, on="time_H", direction="nearest", tolerance=tol)
        merged = merged.dropna(subset=["S1_nT", "S2_nT", "S3_nT"]).reset_index(drop=True)
        if merged.empty:
            return None
        return merged[["time_H", "S1_nT", "S2_nT", "S3_nT"]]

    def _build_obs2_components_df(self, df_raw: pd.DataFrame, path: str) -> Optional[pd.DataFrame]:
        """
        Build aligned (time_H, S1_nT, S2_nT, S3_nT) for OBS2 from raw CSV.
        OBS2 layout: S1 and S2 horizontal, S3 vertical.
        S1 = b_x from OBS2_1 (horizontal), S2 = b_y from OBS2_2 (horizontal), S3 = b_z from OBS2_3 (vertical).
        """
        if df_raw is None or df_raw.empty or "sensor_id" not in df_raw.columns:
            return None
        for c in ("b_x", "b_y", "b_z", "timestamp"):
            if c not in df_raw.columns:
                return None
        ids_1 = [s for s in df_raw["sensor_id"].astype(str).unique() if "OBS2_1" in s]
        ids_2 = [s for s in df_raw["sensor_id"].astype(str).unique() if "OBS2_2" in s]
        ids_3 = [s for s in df_raw["sensor_id"].astype(str).unique() if "OBS2_3" in s]
        if not ids_1 or not ids_2 or not ids_3:
            return None
        df1 = (
            df_raw[df_raw["sensor_id"].isin(ids_1)][["timestamp", "b_x"]]
            .copy()
            .rename(columns={"b_x": "S1_nT"})
        )
        df1["time_H"] = pd.to_datetime(self._parse_csv_timestamps(df1["timestamp"], path))
        df1 = df1.dropna(subset=["time_H"]).sort_values("time_H").drop(columns=["timestamp"])
        df2 = (
            df_raw[df_raw["sensor_id"].isin(ids_2)][["timestamp", "b_y"]]
            .copy()
            .rename(columns={"b_y": "S2_nT"})
        )
        df2["time_H"] = pd.to_datetime(self._parse_csv_timestamps(df2["timestamp"], path))
        df2 = df2.dropna(subset=["time_H"]).sort_values("time_H").drop(columns=["timestamp"])
        df3 = (
            df_raw[df_raw["sensor_id"].isin(ids_3)][["timestamp", "b_z"]]
            .copy()
            .rename(columns={"b_z": "S3_nT"})
        )
        df3["time_H"] = pd.to_datetime(self._parse_csv_timestamps(df3["timestamp"], path))
        df3 = df3.dropna(subset=["time_H"]).sort_values("time_H").drop(columns=["timestamp"])
        tol = pd.Timedelta("2s")
        merged = pd.merge_asof(df1, df2, on="time_H", direction="nearest", tolerance=tol)
        merged = pd.merge_asof(merged, df3, on="time_H", direction="nearest", tolerance=tol)
        merged = merged.dropna(subset=["S1_nT", "S2_nT", "S3_nT"]).reset_index(drop=True)
        if merged.empty:
            return None
        return merged[["time_H", "S1_nT", "S2_nT", "S3_nT"]]

    def _load_csv_source(self, path: str) -> bool:
        try:
            usecols = ["sensor_id", "timestamp", "b_x", "b_y", "b_z"]
            df = pd.read_csv(path, usecols=usecols)
        except Exception:
            try:
                df = pd.read_csv(path)
            except Exception:
                return False

        if df is None or df.empty or "sensor_id" not in df.columns:
            return False

        self._csv_timeseries_by_sensor = {}
        for sid, df_sensor in df.groupby("sensor_id"):
            ts_df = self._csv_raw_to_timeseries_df(df_sensor, path)
            self._csv_timeseries_by_sensor[str(sid)] = ts_df

        all_times = []
        for df_ts in self._csv_timeseries_by_sensor.values():
            if df_ts is not None and not df_ts.empty:
                all_times.append(df_ts["time_H"].iloc[0])
                all_times.append(df_ts["time_H"].iloc[-1])
        if all_times:
            self._csv_time_min = min(all_times)
            self._csv_time_max = max(all_times)
        else:
            self._csv_time_min = None
            self._csv_time_max = None

        # Build per-sensor 3D vector series for interpolation-based direction workflow.
        self._obs1_vector_by_sensor = self._build_obs_vector_series(df, path, "OBS1")
        self._obs1_sensor_baseline_xyz = None
        self._obs1_baseline = None  # set when historic baseline window is available
        # Legacy component dataframe kept for compatibility checks (None when unavailable).
        self._obs1_components_df = None
        if self._obs1_vector_by_sensor:
            obs1_min = min(v["time_H"].min() for v in self._obs1_vector_by_sensor.values())
            obs1_max = max(v["time_H"].max() for v in self._obs1_vector_by_sensor.values())
            n_obs1 = min(len(v) for v in self._obs1_vector_by_sensor.values())
            self.log(
                f"[OBS1] 3D vector series loaded (interpolation-enabled): ~{n_obs1} samples per sensor "
                f"(range: {obs1_min} to {obs1_max}). Direction finding will be available for OBS1 anomalies.",
                level="Info",
            )
        else:
            self.log(
                "[OBS1] 3D vector series: not available (OBS1_1, OBS1_2, OBS1_3 not found in CSV). "
                "Direction finding will not be available.",
                level="Warning",
            )

        self._obs2_vector_by_sensor = self._build_obs_vector_series(df, path, "OBS2")
        self._obs2_sensor_baseline_xyz = None
        self._obs2_baseline = None  # set when historic baseline window is available
        self._obs2_calibration_model = None
        self._obs2_calibration_loaded_from_disk = False
        self._obs2_calibration_attempted = False
        # Legacy component dataframe kept for compatibility checks (None when unavailable).
        self._obs2_components_df = None
        if self._obs2_vector_by_sensor:
            obs2_min = min(v["time_H"].min() for v in self._obs2_vector_by_sensor.values())
            obs2_max = max(v["time_H"].max() for v in self._obs2_vector_by_sensor.values())
            n_obs2 = min(len(v) for v in self._obs2_vector_by_sensor.values())
            self.log(
                f"[OBS2] 3D vector series loaded (interpolation-enabled): ~{n_obs2} samples per sensor "
                f"(range: {obs2_min} to {obs2_max}). Direction finding will be available for OBS2 anomalies.",
                level="Info",
            )
        else:
            self.log(
                "[OBS2] 3D vector series: not available (OBS2_1, OBS2_2, OBS2_3 not found in CSV). "
                "Direction finding will not be available.",
                level="Warning",
            )

        # Try loading a persisted calibration model so this CSV can use a model
        # trained on a previous ground-truth dataset.
        if self._obs2_calibration_enabled:
            self._load_obs2_calibration_model_from_disk()

        self.csv_path = path
        self._csv_playback_complete = False  # Reset so new CSV playback can run
        # Use 60-minute historic window for CSV playback (same as DB mode).
        self._rolling_window_points = self.csv_hist_points
        return True

    def _fetch_csv_window_multi(
        self,
        sensor_ids: List[str],
        *,
        start_time: datetime,
        end_time: datetime,
        target_n_seconds: Optional[int],
        incremental: bool,
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for sid in sensor_ids:
            df_ts = self._csv_timeseries_by_sensor.get(sid)
            if df_ts is None or df_ts.empty:
                out[sid] = pd.DataFrame(columns=["time_H", "mag_H_nT"])
                continue
            if incremental:
                mask = (df_ts["time_H"] > start_time) & (df_ts["time_H"] <= end_time)
            else:
                mask = (df_ts["time_H"] >= start_time) & (df_ts["time_H"] <= end_time)
            df_win = df_ts.loc[mask].copy()
            if target_n_seconds is not None and target_n_seconds > 0 and len(df_win) > target_n_seconds:
                df_win = df_win.tail(int(target_n_seconds)).reset_index(drop=True)
            out[sid] = df_win.reset_index(drop=True)
        return out

    def _init_sim_times(self):
        # Ensure sim time values are consistent and pure datetimes (no Qt objects).
        self.sim_hist_end_time = self.sim_start_time + timedelta(minutes=self.historic_minutes)
        self.sim_rt_start_time = self.sim_hist_end_time
        self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)

    def _init_csv_times(self):
        # CSV mode: 60-minute historic window (same as DB/simulation).
        self.sim_hist_end_time = self.sim_start_time + timedelta(minutes=self.csv_hist_minutes)
        self.sim_rt_start_time = self.sim_hist_end_time
        self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)

    def initViews(self):
        wnd = ApplicationWindowTemp(self)
        return wnd

    # ---- Disable unused frameworks from base application.py (keep only TimeSeries + log) ----
    def load_visualization_framework(self):
        # Skip VTK/spatial initialization for the minimal GUI.
        return

    def load_plot_framework(self):
        # Skip Map/contour plotting for the minimal GUI.
        return

    def _discover_sensors(self):
        # Only skip if we already have sensors AND they match the selected sensors
        if self.sensor_ids and self._selected_sensor_ids and set(self.sensor_ids) == set(self._selected_sensor_ids):
            return
        
        if self.csv_enabled:
            csv_sensors = sorted(self._csv_timeseries_by_sensor.keys(), key=_sensor_sort_key)
            if self._selected_sensor_ids:
                self.sensor_ids = [sid for sid in self._selected_sensor_ids if sid in csv_sensors]
                if not self.sensor_ids:
                    # Selected sensors not found in CSV, fall back to available sensors
                    self.log(f"Selected sensors {self._selected_sensor_ids} not found in CSV. Using available sensors.", level="Warning")
                    self.sensor_ids = csv_sensors[:3]
            else:
                self.sensor_ids = csv_sensors[:3]
            # Restrict sensor_ctx to only selected sensors (drop any from earlier discovery)
            self.sensor_ctx = {sid: self.sensor_ctx[sid] for sid in self.sensor_ids if sid in self.sensor_ctx}
            for sid in self.sensor_ids:
                if sid not in self.sensor_ctx:
                    self.sensor_ctx[sid] = SensorContext(sensor_id=sid, display_name=_sensor_display_name(sid))
            if self._csv_time_min:
                self.sim_start_time = self._csv_time_min
                self._init_csv_times()
            self.log(f"Using CSV sensor streams: {', '.join(self.sensor_ids)}", level="Info")
            return
        
        if self._selected_sensor_ids:
            # Use selected sensors only. Drop any sensor not in the user's selection (e.g. OBS1_1
            # that was added by an earlier startThreads() call before the user had chosen sensors).
            # _configure_sensor_selection already enforces a maximum of 3 sensors.
            self.sensor_ids = self._selected_sensor_ids.copy()
            # Restrict sensor_ctx to only selected sensors so UI/title show only what user chose
            self.sensor_ctx = {sid: self.sensor_ctx[sid] for sid in self.sensor_ids if sid in self.sensor_ctx}
            for sid in self.sensor_ids:
                if sid not in self.sensor_ctx:
                    self.sensor_ctx[sid] = SensorContext(sensor_id=sid, display_name=_sensor_display_name(sid))
            self._init_sim_times()
            self.log(f"Using selected sensors: {', '.join(self.sensor_ids)}", level="Info")
            return
        
        # Fallback: only OBS1 sensor-1 stream (single-sensor bring-up).
        sid = get_latest_sensor_id_like("%OBS1_1")
        if not sid:
            # Hard fallback: keep app alive with empty list
            self.sensor_ids = []
            self.sensor_ctx = {}
            self.log("No sensor_id found for pattern %OBS1_1", level="Error")
            return
        self.sensor_ids = [sid]
        for sid in self.sensor_ids:
            if sid not in self.sensor_ctx:
                self.sensor_ctx[sid] = SensorContext(sensor_id=sid, display_name=_sensor_display_name(sid))
        self._init_sim_times()
        self.log(f"Using single sensor stream: {self.sensor_ctx[self.sensor_ids[0]].display_name}", level="Info")

    def _lowpass_series(self, ctx: SensorContext, values: List[float], reset: bool = False) -> List[float]:
        """
        Apply a simple first-order low-pass filter (exponential moving average) to `values`.
        This is used as a lightweight denoising step per sensor.
        """
        if not values:
            return []
        alpha = self._lowpass_alpha
        filtered: List[float] = []
        # Initialize state
        if reset or ctx.last_filtered_value is None:
            prev = float(values[0])
        else:
            prev = float(ctx.last_filtered_value)
        for v in values:
            v_f = float(v)
            prev = alpha * v_f + (1.0 - alpha) * prev
            filtered.append(prev)
        ctx.last_filtered_value = prev
        return filtered

    def on_db_data_updated(self, dfs: dict, is_new: bool):
        self._discover_sensors()
        for sid, df in dfs.items():
            ctx = self.sensor_ctx.get(sid)
            if ctx is None:
                continue
            if df is None or df.empty:
                # If initial load is empty in simulation, try once to jump to the first
                # available timestamp within the selected date (midnight..midnight+1d).
                if not is_new and self.sim_enabled and not self._initial_fetch_retry and not self.csv_enabled:
                    self._initial_fetch_retry = True
                    try:
                        nxt = get_min_timestamp_at_or_after(sid, self.sim_start_time)
                    except Exception:
                        nxt = None
                    if nxt is not None and nxt < (self.sim_start_time + timedelta(days=1)):
                        self.sim_start_time = nxt
                        self._init_sim_times()
                        self.log(f"No data at midnight; retrying from {self.sim_start_time}.", level="Warning")
                        QTimer.singleShot(200, lambda: self.appWin.startThreads(hours=1, start_time=None, new=False))
                    else:
                        msg = "No data found on the selected date. Please choose another date."
                        try:
                            QMessageBox.warning(self.appWin, "No Data", msg)
                        except Exception:
                            pass
                        self.log("No data found on selected date.", level="Warning")
                continue
            times = pd.to_datetime(df["time_H"]).tolist()
            vals = df["mag_H_nT"].astype(float).tolist()
            if not is_new:
                # Initial (historic) snapshot: up to 60 min (HISTORIC_POINTS_1HZ), or all data if less available
                n = min(len(times), self._rolling_window_points)
                ctx.last_filtered_value = None
                if n:
                    # Low-pass filter the historic series once and keep only the most recent window.
                    filtered_all = self._lowpass_series(ctx, vals, reset=True)
                    ctx.base_x_t = times[-n:]
                    ctx.base_y_mag_t = filtered_all[-n:]
                else:
                    ctx.base_x_t = []
                    ctx.base_y_mag_t = []
                # Baseline (median of filtered values) retained for any auxiliary use.
                ctx.plot_baseline_nT = float(np.median(ctx.base_y_mag_t)) if ctx.base_y_mag_t else None
                ctx.rt_x_t = []
                ctx.rt_y_mag_t = []
                ctx.new_x_t = []
                ctx.new_y_mag_t = []
                ctx.has_seen_realtime = False
                ctx.needs_update_lims = True
                # In simulation mode, advance the simulated realtime cursor to start right after
                # the last point of the loaded historic window. This ensures incremental fetches
                # actually return data (and the green line appears) even if the DB isn't live.
                if self.sim_enabled and ctx.base_x_t:
                    self.sim_rt_start_time = ctx.base_x_t[-1]
                    self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)
            else:
                # Append only points with strictly increasing timestamps
                # Keep `new_*` only for the latest chunk (for anomaly comparison),
                # while the realtime series (green) accumulates.
                last_time = None
                if ctx.rt_x_t:
                    last_time = ctx.rt_x_t[-1]
                elif ctx.base_x_t:
                    last_time = ctx.base_x_t[-1]
                new_chunk_t: List[datetime] = []
                new_chunk_v: List[float] = []
                for t, v in zip(times, vals):
                    if last_time is None or t > last_time:
                        new_chunk_t.append(t)
                        new_chunk_v.append(v)
                        last_time = t

                # Append to realtime accumulated (green) series
                if new_chunk_t:
                    # Low-pass filter only the truly new values, continuing from previous state.
                    filtered_chunk = self._lowpass_series(ctx, new_chunk_v, reset=False)
                    ctx.rt_x_t.extend(new_chunk_t)
                    ctx.rt_y_mag_t.extend(filtered_chunk)
                    # Keep a cumulative realtime buffer for anomaly detection
                    ctx.new_x_t.extend(new_chunk_t)
                    ctx.new_y_mag_t.extend(filtered_chunk)
                    ctx.has_seen_realtime = True
                    ctx.needs_update_lims = True

            # Save/train only when we have new information (avoid rewriting CSVs every draw tick)
            total_points = len(ctx.base_x_t) + len(ctx.rt_x_t)
            if total_points > 0 and total_points != ctx.last_saved_points:
                # Training uses the full historic + realtime series (no length cap); model trains on entire available data.
                x_all = ctx.base_x_t + ctx.rt_x_t
                y_all = ctx.base_y_mag_t + ctx.rt_y_mag_t
                # Start predictor only after realtime (green) begins, to match application.py's visual sequence.
                self.save_data_for_sensor(sid, x_all, y_all, start_predictor=ctx.has_seen_realtime)
                ctx.last_saved_points = total_points

            # Run anomaly detection once we have both predictions and new realtime data
            if ctx.rt_x_t and ctx.predict_x_t:
                self.detect_anomalies_for_sensor(sid)

        # Trigger initial plot framework creation if not loaded yet
        self.appWin.updateData()

        # Advance simulation clock after we successfully fetched an incremental slice.
        if is_new and self.sim_enabled:
            # If we actually received new points, step forward. Otherwise, jump to the next
            # available timestamp (DB only; in CSV mode we advance by step to avoid DB calls).
            any_new = any(self.sensor_ctx[sid].new_x_t for sid in self.sensor_ids if sid in self.sensor_ctx)
            if any_new:
                self.sim_rt_start_time = self.sim_rt_end_time
                self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)
            else:
                if self.csv_enabled:
                    # CSV mode: no new data this tick. If we've passed end of CSV, stop the fetch
                    # timer and run a final anomaly-detection pass so the full range is processed.
                    past_end = False
                    if not self._csv_playback_complete and self._csv_time_max is not None:
                        try:
                            # Normalise to pandas Timestamp for reliable comparison (datetime vs pd.Timestamp)
                            t_max = pd.Timestamp(self._csv_time_max)
                            t_start = pd.Timestamp(self.sim_rt_start_time)
                            t_end = pd.Timestamp(self.sim_rt_end_time)
                            past_end = t_start >= t_max or t_end > t_max
                        except Exception:
                            past_end = self.sim_rt_start_time >= self._csv_time_max
                    if past_end:
                        self._csv_playback_complete = True
                        try:
                            tmr = getattr(self.appWin, "_multi_data_timer", None)
                            if tmr is not None and tmr.isActive():
                                tmr.stop()
                                self.log(
                                    "CSV playback complete: end of data reached; fetch timer stopped.",
                                    level="Info",
                                )
                        except Exception:
                            pass
                        # Final pass: run anomaly detection for each sensor so entire rt range is processed.
                        for sid in self.sensor_ids:
                            ctx = self.sensor_ctx.get(sid)
                            if ctx is not None and ctx.rt_x_t and ctx.predict_x_t:
                                self.detect_anomalies_for_sensor(sid)
                        self.log(
                            "CSV playback complete: anomaly detection run on full range for all sensors.",
                            level="Info",
                        )
                    elif not self._csv_playback_complete:
                        # Not yet at end; advance by step so next fetch uses new window.
                        self.sim_rt_start_time = self.sim_rt_end_time
                        self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)
                else:
                    sid0 = self.sensor_ids[0] if self.sensor_ids else None
                    if sid0:
                        try:
                            nxt = get_min_timestamp_at_or_after(sid0, self.sim_rt_end_time)
                            if nxt is not None:
                                self.sim_rt_start_time = nxt
                                self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)
                        except Exception:
                            # DB unavailable or connection lost; advance by step to keep UI alive
                            self.sim_rt_start_time = self.sim_rt_end_time
                            self.sim_rt_end_time = self.sim_rt_start_time + timedelta(seconds=self.sim_step_seconds)

    def _detrend_for_plot(self, ctx: SensorContext, ys: List[float]) -> List[float]:
        """
        Prepare values for plotting.

        For the multi-sensor DB workflow we now plot the (optionally low-pass filtered)
        resultant magnetic field directly, without subtracting a baseline.
        """
        return ys

    def get_since_times(self) -> Dict[str, datetime]:
        since: Dict[str, datetime] = {}
        for sid, ctx in self.sensor_ctx.items():
            # Incremental fetch should start after the latest known timestamp in the plotted stream.
            if ctx.rt_x_t:
                since[sid] = ctx.rt_x_t[-1]
            elif ctx.base_x_t:
                since[sid] = ctx.base_x_t[-1]
        return since

    def load_plot_framework_2(self):
        """
        Create a single multi-sensor TimeSeries view.

        Up to 3 selected sensors are shown simultaneously in one window with:
        - ONE shared control panel (parameters apply to all sensors)
        - Separate time-series plots for each sensor (low-pass filtered resultant magnetic field)
        """
        self._discover_sensors()
        window = self.appWin

        # Attach our multi-sensor TimeSeries UI to the minimal container created by ApplicationWindowTemp.
        host = getattr(window, "_temp_timeseries_container", None)
        if host is None:
            # Fallback: if minimal UI isn't available for some reason, use the original tab_2.
            host = window.tab_2

        # Clear any existing layout contents (best-effort)
        if host.layout() is None:
            host.setLayout(Qt.QVBoxLayout())
        outer = host.layout()
        while outer.count():
            item = outer.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        # We no longer use per-sensor QTabWidget.
        self._time_series_tabs = None
        
        # Initialize list to track control widgets for synchronization
        self._sensor_control_widgets = []

        # ===== RIGHT HALF: shared parameters — insert at top of right column =====
        right_layout = getattr(window, "_right_layout", None)
        first_sensor_id = self.sensor_ids[0] if self.sensor_ids else None
        if first_sensor_id and right_layout is not None:
            params_container = QWidget()
            params_container.setMinimumHeight(200)
            params_container.setMaximumHeight(220)
            params_container.setMinimumWidth(220)
            params_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            params_layout = Qt.QVBoxLayout(params_container)
            params_layout.setContentsMargins(0, 0, 0, 0)
            params_layout.setSpacing(4)

            controls_header = QLabel("<b>Shared Parameters (apply to all sensors)</b>")
            controls_header.setStyleSheet("font-size: 11px; padding: 3px 6px; background-color: #e8f4f8; border: 1px solid #4a90e2; font-weight: bold;")
            params_layout.addWidget(controls_header)

            shared_controls = SensorMagTimeSeriesWidget(self, sensor_id=first_sensor_id, parent=window)
            shared_controls.setMinimumHeight(180)
            shared_controls.setMaximumHeight(200)
            shared_controls.setMinimumWidth(200)
            try:
                shared_controls.comboBox.setItemText(0, "IITK Observatory")
            except Exception:
                pass

            def hide_plot_area():
                try:
                    shared_controls.scrollArea.setWidgetResizable(False)
                    scroll_widget = shared_controls.scrollArea.widget()
                    if scroll_widget:
                        scroll_widget.setFixedHeight(10)
                        scroll_widget.setFixedWidth(max(shared_controls.scrollArea.width(), 400))
                        plot_layout = getattr(shared_controls, "verticalLayout_3", None)
                        if plot_layout is not None:
                            while plot_layout.count():
                                item = plot_layout.takeAt(0)
                                if item:
                                    w = item.widget()
                                    if w:
                                        if isinstance(w, FigureCanvas):
                                            try:
                                                w.figure.set_size_inches(1.0, 1.0, forward=False)
                                                w.setMinimumSize(1, 1)
                                                w.setMaximumSize(1, 1)
                                            except Exception:
                                                pass
                                        w.hide()
                                        w.setParent(None)
                                    else:
                                        plot_layout.removeItem(item)
                        scroll_widget.hide()
                    shared_controls.scrollArea.setFixedHeight(0)
                    shared_controls.scrollArea.hide()
                    shared_controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                    shared_controls.setMinimumHeight(180)
                    shared_controls.setMaximumHeight(200)
                except Exception as e:
                    self.log(f"Warning: Could not fully configure controls widget layout: {e}", level="Warning")

            QTimer.singleShot(300, hide_plot_area)
            params_layout.addWidget(shared_controls)
            shared_controls.show()

            right_layout.insertWidget(0, params_container, 1)
            params_container.show()
            if getattr(window, "_right_half", None) is not None:
                window._right_half.updateGeometry()

        # ===== LEFT HALF: one scrollable panel per sensor (each scrollable L/R/U/D) =====
        for sid in self.sensor_ids:
            ctx = self.sensor_ctx.get(sid)
            if ctx is None:
                continue

            # Inner panel for this sensor (label + toolbar + canvas)
            plot_panel = QWidget()
            plot_panel_layout = Qt.QVBoxLayout(plot_panel)
            plot_panel_layout.setContentsMargins(4, 4, 4, 4)
            plot_panel_layout.setSpacing(4)
            plot_panel.setMinimumSize(520, 320)  # Large enough to see plot; scroll if needed
            plot_panel.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

            sensor_label = QLabel(f"<b>Sensor: {ctx.display_name}</b>")
            sensor_label.setStyleSheet("font-size: 12px; padding: 4px; background-color: #f0f0f0; border: 1px solid #ccc;")
            plot_panel_layout.addWidget(sensor_label)

            fig_width, fig_height = 7.0, 3.0
            dynamic_canvas = FigureCanvas(Figure(figsize=(fig_width, fig_height)))
            dynamic_canvas.setMinimumSize(480, 240)
            dynamic_canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
            plot_panel_layout.addWidget(NavigationToolbar(dynamic_canvas, window))
            plot_panel_layout.addWidget(dynamic_canvas)

            ctx.static_canvas = None
            ctx.static_ax = None
            ctx.static_line = None
            ctx.dynamic_canvas = dynamic_canvas
            ctx.dynamic_ax = dynamic_canvas.figure.subplots()
            ctx.dynamic_line = None

            if ctx.base_x_t and ctx.base_y_mag_t:
                # Plot the historic (filtered) snapshot as the initial line.
                y0 = self._detrend_for_plot(ctx, ctx.base_y_mag_t)
                ctx.dynamic_line, = ctx.dynamic_ax.plot(ctx.base_x_t, y0)
                try:
                    ctx.dynamic_ax.set_ylabel("B (nT)")
                except Exception:
                    pass
                # Save once on initial load, but do NOT start predictor yet.
                # We'll start after first realtime update so UI order matches application.py.
                self.save_data_for_sensor(sid, ctx.base_x_t, ctx.base_y_mag_t, start_predictor=False)
                ctx.last_saved_points = len(ctx.base_x_t)

            # Wrap each sensor panel in its own scroll area (scrollable left, right, up, down)
            sensor_scroll = QScrollArea()
            sensor_scroll.setWidget(plot_panel)
            sensor_scroll.setWidgetResizable(True)
            sensor_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            sensor_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            sensor_scroll.setMinimumHeight(220)
            sensor_scroll.setFrameShape(QFrame.StyledPanel)
            outer.addWidget(sensor_scroll, 1)  # stretch 1 so all three panels share left half equally

        # Timers for periodic fetch + drawing
        self._multi_data_timer = QTimer()
        self._multi_data_timer.timeout.connect(lambda: self.appWin.startThreads(hours=None, start_time=None, new=True))
        self._multi_data_timer.start(1000 * 20)

        self._multi_draw_timer = QTimer()
        # Redraw all visible sensor plots on each tick (up to 3 sensors).
        self._multi_draw_timer.timeout.connect(self.update_all_canvases)
        self._multi_draw_timer.start(400)

        # Poll predictor outputs at a lower rate to keep UI responsive
        self._multi_pred_timer = QTimer()
        self._multi_pred_timer.timeout.connect(self.poll_predictions_all_sensors)
        self._multi_pred_timer.start(3000)

        # Predictor scheduler: start at most N predictor subprocesses at a time.
        self._predict_sched_timer = QTimer()
        self._predict_sched_timer.timeout.connect(self._drain_predict_queue)
        self._predict_sched_timer.start(500)

        # Fallback: if the DB does not advance (no "green" data arrives),
        # start predictors after a short grace period so predictions still appear.
        QTimer.singleShot(self._predict_start_grace_seconds * 1000, self._start_predictors_if_idle)

        # Window title: show selected sensor name(s) instead of hardcoded text below "Magnavis"
        try:
            sensor_label = ", ".join(ctx.display_name for ctx in self.sensor_ctx.values())
            if sensor_label:
                window.setWindowTitle(f"Magnavis – {sensor_label}")
            else:
                window.setWindowTitle("Magnavis")
        except Exception:
            window.setWindowTitle("Magnavis")

        self.log(f"Multi-sensor TimeSeries loaded: left={len(self.sensor_ids)} sensor streams (each scrollable), right=parameters + direction plot + log.", level="Info")

    def update_all_canvases(self):
        """
        Redraw all sensor canvases.

        With the stacked multi-sensor layout (no tabs), all up-to-3 sensors are visible
        simultaneously, so we simply refresh every sensor on each timer tick.
        """
        for sid in self.sensor_ids:
            self.update_canvas_for_sensor(sid, poll_predictions=False)

    def update_canvas_for_sensor(self, sensor_id: str, poll_predictions: bool = True):
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None or ctx.dynamic_ax is None:
            return

        # Ensure the historic (blue) line stays on the original snapshot (do not overwrite).
        if ctx.dynamic_line is not None and ctx.base_x_t and ctx.base_y_mag_t:
            try:
                ctx.dynamic_line.set_data(ctx.base_x_t, self._detrend_for_plot(ctx, ctx.base_y_mag_t))
            except Exception:
                pass

        # Update/plot new real-time line
        if ctx.rt_x_t and ctx.rt_y_mag_t:
            if ctx.dynamic_new_line is None:
                ctx.dynamic_new_line, = ctx.dynamic_ax.plot(
                    ctx.rt_x_t,
                    self._detrend_for_plot(ctx, ctx.rt_y_mag_t),
                    color=[0.1, 0.7, 0.2],
                )
            else:
                ctx.dynamic_new_line.set_data(ctx.rt_x_t, self._detrend_for_plot(ctx, ctx.rt_y_mag_t))

        # Update predictions line
        if poll_predictions:
            self.update_predictions_for_sensor(sensor_id)
        if ctx.predict_x_t and ctx.predict_y_t:
            if ctx.predictions_line is None:
                ctx.predictions_line, = ctx.dynamic_ax.plot(
                    ctx.predict_x_t,
                    self._detrend_for_plot(ctx, ctx.predict_y_t),
                    color=[0.3, 0.1, 0.4],
                )
            else:
                ctx.predictions_line.set_data(ctx.predict_x_t, self._detrend_for_plot(ctx, ctx.predict_y_t))

        # Update anomaly vertical lines (dynamic + static)
        self._redraw_anomalies(sensor_id)

        # Update limits similar to application.py when new data arrives
        if ctx.needs_update_lims and ctx.base_x_t:
            try:
                x0 = ctx.base_x_t[0]
                x1 = ctx.rt_x_t[-1] if ctx.rt_x_t else ctx.base_x_t[-1]
                yr = self._detrend_for_plot(ctx, ctx.base_y_mag_t + (ctx.rt_y_mag_t if ctx.rt_y_mag_t else []))
                if yr:
                    ymax = max(yr)
                    ymin = min(yr)
                    _xrange = (x1 - x0) if hasattr(x1, "__sub__") else 1
                    _yrange = ymax - ymin if ymax != ymin else 1.0
                    ctx.dynamic_ax.set_xlim(x0, x1)
                    ctx.dynamic_ax.set_ylim(ymin - 0.05 * _yrange, ymax + 0.05 * _yrange)
                    if ctx.static_ax is not None:
                        ctx.static_ax.set_xlim(x0, x1)
                        ctx.static_ax.set_ylim(ymin - 0.05 * _yrange, ymax + 0.05 * _yrange)
                ctx.needs_update_lims = False
            except Exception:
                ctx.needs_update_lims = False

        # Draw
        try:
            if ctx.dynamic_canvas:
                ctx.dynamic_canvas.draw_idle()
            if ctx.static_canvas:
                ctx.static_canvas.draw_idle()
        except Exception:
            pass

    def poll_predictions_all_sensors(self):
        """Low-frequency polling of predictor outputs to avoid blocking UI redraws."""
        for sid in self.sensor_ids:
            self.update_predictions_for_sensor(sid)

    def _redraw_anomalies(self, sensor_id: str):
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None or ctx.dynamic_ax is None:
            return

        # Remove old
        for v in ctx.anomaly_vertical_lines:
            try:
                v.remove()
            except Exception:
                pass
        for v in ctx.anomaly_vertical_lines_static:
            try:
                v.remove()
            except Exception:
                pass
        ctx.anomaly_vertical_lines = []
        ctx.anomaly_vertical_lines_static = []

        if not ctx.anomaly_times:
            return

        for t in ctx.anomaly_times:
            try:
                ctx.anomaly_vertical_lines.append(
                    ctx.dynamic_ax.axvline(x=t, color="lightcoral", linestyle="-", linewidth=2, alpha=0.4, zorder=6)
                )
            except Exception:
                pass

    def save_data_for_sensor(self, sensor_id: str, x_t: List[datetime], y_t: List[float], start_predictor: bool = True):
        """
        Write time series to predictor input CSV for GRU training.

        IMPORTANT:
        - Only include data up to ctx.last_anomaly_checked_time, i.e. timestamps that have
          already been compared against predictions by the anomaly detector.
        - Within that "safe" window, drop any timestamps that were flagged as anomalies.
        - Newest realtime points (beyond last_anomaly_checked_time) are held out of training
          until they have been processed by the anomaly detector in a later cycle.
        """
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None:
            return

        # Session subfolder per sensor
        folder = os.path.join(base_app.APP_BASE, "sessions", self.session_id, sensor_id)
        os.makedirs(folder, exist_ok=True)
        inp_file = os.path.join(folder, "predict_input.csv")

        # Filter training region and anomalies by timestamp.
        # (1) Restrict to data that has already been processed by anomaly detector.
        # (2) Within that region, drop timestamps that were flagged as anomalies.
        anomaly_times_set = set(pd.to_datetime(ctx.anomaly_times)) if ctx.anomaly_times else set()
        cutoff_time = pd.to_datetime(ctx.last_anomaly_checked_time) if ctx.last_anomaly_checked_time is not None else None
        filtered_x = []
        filtered_y = []
        for t, v in zip(x_t, y_t):
            t_dt = pd.to_datetime(t)
            # Skip any data that has not yet been compared with predictions.
            if cutoff_time is not None and t_dt > cutoff_time:
                continue
            if t_dt not in anomaly_times_set:
                filtered_x.append(t_dt)
                filtered_y.append(v)

        pd.DataFrame({"x": filtered_x, "y": filtered_y}).to_csv(inp_file, index=False)
        ctx.predictor_input_file = inp_file

        # Enqueue prediction rather than starting all 6 TensorFlow processes at once.
        # This keeps the GUI responsive while still producing identical outputs per sensor.
        if start_predictor and not ctx.predict_app_started:
            # Throttle per-sensor predictor runs; otherwise each sensor retrains every fetch tick.
            now = time.time()
            if ctx.last_pred_start_ts and (now - ctx.last_pred_start_ts) < self._predict_cooldown_seconds:
                return
            ctx.predict_app_started = True  # mark as scheduled/running
            ctx.last_pred_start_ts = now
            self._enqueue_prediction(sensor_id)

    def _start_predictors_if_idle(self):
        """
        If we haven't started predictors yet (no realtime updates), start them anyway
        so the app still produces predictions on purely "historic" data.
        """
        for sid, ctx in self.sensor_ctx.items():
            if ctx.predictor_input_file and not ctx.predict_app_started and ctx.prediction_process is None:
                ctx.predict_app_started = True
                ctx.last_pred_start_ts = time.time()
                self._enqueue_prediction(sid)

    def _enqueue_prediction(self, sensor_id: str):
        if sensor_id in self._predict_active:
            return
        if sensor_id in self._predict_queue:
            return
        self._predict_queue.append(sensor_id)

    def _drain_predict_queue(self):
        # Start predictors until we hit concurrency limit.
        while len(self._predict_active) < self._predict_max_concurrent and self._predict_queue:
            sid = self._predict_queue.popleft()
            ctx = self.sensor_ctx.get(sid)
            if ctx is None or not ctx.predictor_input_file:
                continue
            # If process already exists/running, treat as active.
            if ctx.prediction_process is not None and ctx.prediction_process.poll() is None:
                self._predict_active.add(sid)
                continue
            self._predict_active.add(sid)
            self.start_prediction_process_for_sensor(sid)

    def start_prediction_process_for_sensor(self, sensor_id: str):
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None or not ctx.predictor_input_file:
            return

        python_exe = sys.executable
        predictor_script = os.path.join(base_app.APP_BASE, "predictor_ai.py")
        input_file = ctx.predictor_input_file
        work_dir = os.path.dirname(input_file)

        env = os.environ.copy()
        # Reduce CPU contention so the GUI remains responsive.
        # (Does not change model semantics; only controls parallelism.)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
        env.setdefault("TF_NUM_INTEROP_THREADS", "1")
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        if ctx.train_window_minutes:
            env["TRAIN_WINDOW_MINUTES"] = str(ctx.train_window_minutes)
        else:
            env.pop("TRAIN_WINDOW_MINUTES", None)
        # Set pre-trained model path per sensor (if available)
        # Default: project root "models" folder (works when run from any directory)
        pretrained_model_dir = os.environ.get("PRETRAINED_MODEL_DIR", None)
        if not pretrained_model_dir:
            _project_root = os.path.dirname(base_app.APP_BASE)
            _default_models = os.path.join(_project_root, "models")
            if os.path.isdir(_default_models):
                pretrained_model_dir = _default_models
        if pretrained_model_dir:
            # Try to find model for this sensor_id (handle full sensor_id like "S20260202_124213_OBS1_1")
            safe_sensor_id = sensor_id.replace('/', '_').replace('\\', '_')
            model_path = os.path.join(pretrained_model_dir, f"gru_pretrained_{safe_sensor_id}.keras")
            # Also try with just the OBS1_1 part if full ID doesn't match
            if not os.path.exists(model_path):
                import re
                import glob
                m = re.search(r"(OBS\d+_\d+)", sensor_id)
                if m:
                    obs_part = m.group(1)
                    # Try exact match first: gru_pretrained_OBS1_1.keras
                    model_path = os.path.join(pretrained_model_dir, f"gru_pretrained_{obs_part}.keras")
                    # If that doesn't exist, search for any model containing this OBS part
                    if not os.path.exists(model_path):
                        pattern = os.path.join(pretrained_model_dir, f"gru_pretrained_*{obs_part}.keras")
                        matches = glob.glob(pattern)
                        if matches:
                            # Use the first match (prefer most recent if sorted)
                            model_path = sorted(matches)[-1]  # Last = most recent if timestamp-sorted
            if os.path.exists(model_path):
                env["PRETRAINED_MODEL_PATH"] = model_path
                self.log(f"[{ctx.display_name}] Using pre-trained model: {model_path}", level="Info")
            else:
                env.pop("PRETRAINED_MODEL_PATH", None)
                self.log(f"[{ctx.display_name}] No pre-trained model found for sensor_id={sensor_id} in {pretrained_model_dir}", level="Warning")
        else:
            env.pop("PRETRAINED_MODEL_PATH", None)

        stdout_f = open(os.path.join(work_dir, "predict_stdout.log"), "w")
        stderr_f = open(os.path.join(work_dir, "predict_stderr.log"), "w")
        cmd = [python_exe, predictor_script, input_file]
        # Lower OS scheduling priority on mac/linux to keep UI smooth.
        if os.name != "nt":
            cmd = ["nice", "-n", "10"] + cmd
        self.log(f"[{ctx.display_name}] Starting predictor: {' '.join(cmd)}", level="Info")
        ctx.prediction_process = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, cwd=work_dir, env=env)

    def update_predictions_for_sensor(self, sensor_id: str):
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None or not ctx.predictor_input_file:
            return

        out_file = os.path.join(os.path.dirname(ctx.predictor_input_file), "predict_out.csv")
        proc = ctx.prediction_process
        if proc is not None:
            rc = proc.poll()
            if rc is None:
                return  # still running
            if rc != 0:
                # failed
                ctx.prediction_process = None
                ctx.predict_app_started = False
                self._predict_active.discard(sensor_id)
                return
            ctx.prediction_process = None
            ctx.predict_app_started = False
            self._predict_active.discard(sensor_id)
            ctx.last_pred_complete_ts = time.time()

        if not os.path.exists(out_file):
            return

        try:
            pred = pd.read_csv(out_file)
            pred["x"] = pd.to_datetime(pred["x"])
            new_x = pred["x"].tolist()
            new_y = pred["y"].astype(float).tolist()
            if new_x:
                merged = {}
                for t, v in zip(ctx.predict_x_t, ctx.predict_y_t):
                    merged[t] = v
                for t, v in zip(new_x, new_y):
                    if t not in merged:
                        merged[t] = v
                merged_times = sorted(merged.keys())
                ctx.predict_x_t = merged_times
                ctx.predict_y_t = [merged[t] for t in merged_times]
            self.detect_anomalies_for_sensor(sensor_id)
        except Exception:
            return

    def _ensure_obs1_baseline(self) -> None:
        """Compute OBS1 per-sensor 3D baseline vectors over the historic window."""
        if not self.csv_enabled:
            return
        if self._obs1_sensor_baseline_xyz is not None:
            return
        if not self._obs1_vector_by_sensor:
            if self.csv_enabled:
                self.log(
                    "[OBS1] Direction baseline: unavailable (OBS1 3D vector data not found in CSV - ensure OBS1_1, OBS1_2, OBS1_3 are present)",
                    level="Warning",
                )
            return
        obs1_sids = [s for s in self.sensor_ids if is_obs1_sensor(s)]
        if not obs1_sids:
            return
        ctx0 = self.sensor_ctx.get(obs1_sids[0])
        if ctx0 is None or not ctx0.base_x_t:
            return
        t_min = min(ctx0.base_x_t)
        t_max = max(ctx0.base_x_t)
        baseline_xyz: Dict[str, Tuple[float, float, float]] = {}
        for idx in (1, 2, 3):
            key = f"OBS1_{idx}"
            df_s = self._obs1_vector_by_sensor.get(key)
            if df_s is None or df_s.empty:
                self.log(
                    f"[OBS1] Direction baseline: unavailable (missing vector series for {key})",
                    level="Warning",
                )
                return
            mask = (pd.to_datetime(df_s["time_H"]) >= pd.Timestamp(t_min)) & (
                pd.to_datetime(df_s["time_H"]) <= pd.Timestamp(t_max)
            )
            slice_df = df_s.loc[mask]
            if slice_df.empty:
                self.log(
                    f"[OBS1] Direction baseline: unavailable (no {key} data in historic window {t_min} to {t_max})",
                    level="Warning",
                )
                return
            baseline_xyz[key] = (
                float(slice_df["b_x"].median()),
                float(slice_df["b_y"].median()),
                float(slice_df["b_z"].median()),
            )
        self._obs1_sensor_baseline_xyz = baseline_xyz
        self._obs1_baseline = (0.0, 0.0, 0.0)  # readiness marker for legacy checks/logs
        self.log(
            "[OBS1] 3D direction baselines set (per sensor, median over historic window).",
            level="Info",
        )

    def _get_obs1_components_at_time(self, t: datetime) -> Optional[Tuple[float, float, float]]:
        """
        Return fused perturbation vector (dX, dY, dZ) for OBS1 at time t.

        Uses interpolation per sensor and sensor-wise baseline subtraction before fusion.
        """
        if self._obs1_sensor_baseline_xyz is None:
            return None
        return self._fuse_observatory_delta_vector("OBS1", t)

    def _ensure_obs2_baseline(self) -> None:
        """Compute OBS2 per-sensor 3D baseline vectors over the historic window."""
        if not self.csv_enabled:
            return
        if self._obs2_sensor_baseline_xyz is not None:
            return
        if not self._obs2_vector_by_sensor:
            if self.csv_enabled:
                self.log(
                    "[OBS2] Direction baseline: unavailable (OBS2 3D vector data not found in CSV - ensure OBS2_1, OBS2_2, OBS2_3 are present)",
                    level="Warning",
                )
            return
        obs2_sids = [s for s in self.sensor_ids if is_obs2_sensor(s)]
        if not obs2_sids:
            return
        ctx0 = self.sensor_ctx.get(obs2_sids[0])
        if ctx0 is None or not ctx0.base_x_t:
            return
        t_min = min(ctx0.base_x_t)
        t_max = max(ctx0.base_x_t)
        baseline_xyz: Dict[str, Tuple[float, float, float]] = {}
        for idx in (1, 2, 3):
            key = f"OBS2_{idx}"
            df_s = self._obs2_vector_by_sensor.get(key)
            if df_s is None or df_s.empty:
                self.log(
                    f"[OBS2] Direction baseline: unavailable (missing vector series for {key})",
                    level="Warning",
                )
                return
            mask = (pd.to_datetime(df_s["time_H"]) >= pd.Timestamp(t_min)) & (
                pd.to_datetime(df_s["time_H"]) <= pd.Timestamp(t_max)
            )
            slice_df = df_s.loc[mask]
            if slice_df.empty:
                self.log(
                    f"[OBS2] Direction baseline: unavailable (no {key} data in historic window {t_min} to {t_max})",
                    level="Warning",
                )
                return
            baseline_xyz[key] = (
                float(slice_df["b_x"].median()),
                float(slice_df["b_y"].median()),
                float(slice_df["b_z"].median()),
            )
        self._obs2_sensor_baseline_xyz = baseline_xyz
        self._obs2_baseline = (0.0, 0.0, 0.0)  # readiness marker for legacy checks/logs
        self.log(
            "[OBS2] 3D direction baselines set (per sensor, median over historic window).",
            level="Info",
        )
        # Build CSV-only supervised calibration once baseline is ready.
        self._ensure_obs2_calibration_model()

    def _get_obs2_components_at_time(self, t: datetime) -> Optional[Tuple[float, float, float]]:
        """
        Return fused perturbation vector (dX, dY, dZ) for OBS2 at time t.

        Uses interpolation per sensor and sensor-wise baseline subtraction before fusion.
        """
        if self._obs2_sensor_baseline_xyz is None:
            return None
        return self._fuse_observatory_delta_vector("OBS2", t)

    def _attempt_triangulation(self, current_time: datetime) -> None:
        """
        Attempt to triangulate source location when anomalies occur at both observatories.
        
        Matches anomalies from OBS1 and OBS2 within the time window and attempts triangulation.
        Updates UI with triangulated location if successful.
        """
        if not self._obs1_recent_anomalies or not self._obs2_recent_anomalies:
            return
        
        # Find closest matching anomalies (within time window)
        best_match = None
        min_time_diff = None
        
        for obs1_time, (obs1_az, obs1_inc, obs1_mag) in self._obs1_recent_anomalies.items():
            for obs2_time, (obs2_az, obs2_inc, obs2_mag) in self._obs2_recent_anomalies.items():
                time_diff = abs((obs1_time - obs2_time).total_seconds())
                if time_diff <= self._triangulation_time_window.total_seconds():
                    if min_time_diff is None or time_diff < min_time_diff:
                        min_time_diff = time_diff
                        best_match = (obs1_time, obs1_az, obs1_inc, obs1_mag, obs2_time, obs2_az, obs2_inc, obs2_mag)
        
        if best_match is None:
            return
        
        obs1_time, obs1_az, obs1_inc, obs1_mag, obs2_time, obs2_az, obs2_inc, obs2_mag = best_match
        
        # Attempt triangulation
        result = triangulate_source_location(
            self._obs1_position,
            obs1_az,
            obs1_inc,
            self._obs2_position,
            obs2_az,
            obs2_inc,
            max_distance_m=1000.0,
        )
        
        if result is None:
            obs1_az_str = f"{obs1_az:.1f}°" if obs1_az is not None else "N/A"
            obs1_inc_str = f"{obs1_inc:.1f}°" if obs1_inc is not None else "N/A"
            obs2_az_str = f"{obs2_az:.1f}°" if obs2_az is not None else "N/A"
            obs2_inc_str = f"{obs2_inc:.1f}°" if obs2_inc is not None else "N/A"
            self.log(
                f"[Triangulation] Failed to triangulate source location | "
                f"OBS1: time={obs1_time}, az={obs1_az_str}, inc={obs1_inc_str} | "
                f"OBS2: time={obs2_time}, az={obs2_az_str}, inc={obs2_inc_str} | "
                f"time_diff={min_time_diff:.1f}s",
                level="Warning",
            )
            return
        
        source_x, source_y, source_z, distance_error = result
        
        # Calculate distance from OBS1
        dist_from_obs1 = math.sqrt(source_x * source_x + source_y * source_y + source_z * source_z)
        
        # Log triangulation result
        obs1_az_str = f"{obs1_az:.1f}°" if obs1_az is not None else "N/A"
        obs1_inc_str = f"{obs1_inc:.1f}°" if obs1_inc is not None else "N/A"
        obs2_az_str = f"{obs2_az:.1f}°" if obs2_az is not None else "N/A"
        obs2_inc_str = f"{obs2_inc:.1f}°" if obs2_inc is not None else "N/A"
        
        self.log(
            f"[Triangulation] Source location estimated | "
            f"position: ({source_x:.1f}, {source_y:.1f}, {source_z:.1f}) m | "
            f"distance from OBS1: {dist_from_obs1:.1f} m | "
            f"error: {distance_error:.1f} m | "
            f"OBS1: time={obs1_time}, az={obs1_az_str}, inc={obs1_inc_str} | "
            f"OBS2: time={obs2_time}, az={obs2_az_str}, inc={obs2_inc_str} | "
            f"time_diff={min_time_diff:.1f}s",
            level="Info",
        )
        
        # Update UI with triangulated location
        try:
            location_str = f"({source_x:.1f}, {source_y:.1f}, {source_z:.1f}) m"
            self.appWin.update_obs1_direction(
                f"[Triangulated] Source: {location_str} | "
                f"dist={dist_from_obs1:.1f}m | error={distance_error:.1f}m"
            )
        except Exception:
            pass

    def _format_direction_label(self, azimuth_deg: Optional[float], observatory: str) -> str:
        """
        Convert an azimuth angle into a human-friendly compass direction string,
        taking into account per-observatory sensor orientation.

        Geographic convention:
        - 0° = East, 90° = North, 180° = West, 270° = South.

        OBS1:
        - Sensor 1 points East, Sensor 3 points North
        - So OBS1 azimuth already uses the geographic frame.

        OBS2:
        - Sensor 1 points West, Sensor 2 points South
        - OBS2 azimuth is defined in the sensor frame (0° = S1, 90° = S2).
        - To convert to geographic frame we rotate by 180°:
          0° (S1) -> 180° (West), 90° (S2) -> 270° (South), etc.
        """
        if azimuth_deg is None:
            return "vertical-only (no horizontal direction)"

        # Map sensor-frame azimuth to geographic frame
        if observatory == "OBS1":
            global_deg = azimuth_deg
        elif observatory == "OBS2":
            global_deg = (azimuth_deg + 180.0) % 360.0
        else:
            global_deg = azimuth_deg

        global_deg = global_deg % 360.0

        # 8-sector compass (E, NE, N, NW, W, SW, S, SE)
        sectors = [
            (22.5, "E"),
            (67.5, "NE"),
            (112.5, "N"),
            (157.5, "NW"),
            (202.5, "W"),
            (247.5, "SW"),
            (292.5, "S"),
            (337.5, "SE"),
            (360.0, "E"),
        ]

        for boundary, label in sectors:
            if global_deg < boundary:
                long_names = {
                    "N": "north",
                    "NE": "north-east",
                    "E": "east",
                    "SE": "south-east",
                    "S": "south",
                    "SW": "south-west",
                    "W": "west",
                    "NW": "north-west",
                }
                long_name = long_names.get(label, "")
                return f"{label} ({long_name})" if long_name else label

        # Fallback (should not occur due to 0–360 wrap)
        return "unknown"

    @staticmethod
    def _format_global_direction_label(global_azimuth_deg: Optional[float]) -> str:
        """Convert a global azimuth angle (0=E, 90=N) into a compass direction label."""
        if global_azimuth_deg is None:
            return "unknown"
        deg = global_azimuth_deg % 360.0
        sectors = [
            (22.5, "E"),
            (67.5, "NE"),
            (112.5, "N"),
            (157.5, "NW"),
            (202.5, "W"),
            (247.5, "SW"),
            (292.5, "S"),
            (337.5, "SE"),
            (360.0, "E"),
        ]
        for boundary, label in sectors:
            if deg < boundary:
                long_names = {
                    "N": "north",
                    "NE": "north-east",
                    "E": "east",
                    "SE": "south-east",
                    "S": "south",
                    "SW": "south-west",
                    "W": "west",
                    "NW": "north-west",
                }
                long_name = long_names.get(label, "")
                return f"{label} ({long_name})" if long_name else label
        return "unknown"

    def detect_anomalies_for_sensor(self, sensor_id: str):
        ctx = self.sensor_ctx.get(sensor_id)
        if ctx is None:
            return
        if not ctx.rt_x_t or not ctx.predict_x_t:
            return

        differences_df = ctx.anomaly_detector.calculate_differences(
            actual_times=ctx.rt_x_t,
            actual_values=ctx.rt_y_mag_t,
            predicted_times=ctx.predict_x_t,
            predicted_values=ctx.predict_y_t,
        )
        if differences_df is None or differences_df.empty:
            return

        # Update "safe for training" cutoff: all timestamps in differences_df have now been
        # compared with predictions and either classified as normal or anomalous.
        try:
            latest_checked = pd.to_datetime(differences_df["time"]).max()
        except Exception:
            latest_checked = None
        if latest_checked is not None:
            if ctx.last_anomaly_checked_time is None or latest_checked > ctx.last_anomaly_checked_time:
                ctx.last_anomaly_checked_time = latest_checked

        threshold = ctx.anomaly_detector.anomaly_threshold
        anomalies_df = differences_df[differences_df["is_anomaly"]].copy()
        new_times = pd.to_datetime(anomalies_df["time"]).tolist()
        new_vals = anomalies_df["actual"].astype(float).tolist()

        existing = set(pd.to_datetime(ctx.anomaly_times)) if ctx.anomaly_times else set()
        newly_added = []
        for t, v in zip(new_times, new_vals):
            if t not in existing:
                ctx.anomaly_times.append(t)
                ctx.anomaly_values.append(v)
                existing.add(t)
                newly_added.append((t, v))

        # Log all newly detected anomalies with timestamp and details
        if newly_added:
            # Direction finding for OBS1: interpolate all three sensor vectors, fuse, then compute direction.
            if self.csv_enabled and is_obs1_sensor(sensor_id) and self._obs1_vector_by_sensor:
                self._ensure_obs1_baseline()
                if self._obs1_sensor_baseline_xyz is not None:
                    for t, v in newly_added:
                        comp = self._get_obs1_components_at_time(t)
                        if comp is not None:
                            d_x, d_y, d_z = comp
                            az_deg, inc_deg, mag_nT = self._direction_from_delta_vector(d_x, d_y, d_z)
                            az_str = f"{az_deg:.1f}°" if az_deg is not None else "N/A (vertical)"
                            inc_str = f"{inc_deg:.1f}°" if inc_deg is not None else "N/A"
                            dir_str = self._format_direction_label(az_deg, observatory="OBS1")
                            self.log(
                                f"[{ctx.display_name}] Anomaly detected | time={t} | magnitude={v:.1f} nT | "
                                f"direction: azimuth={az_str} ({dir_str}), inclination={inc_str}, |ΔB|={mag_nT:.1f} nT",
                                level="Info",
                            )
                            # Store for triangulation
                            self._obs1_recent_anomalies[t] = (az_deg, inc_deg, mag_nT)
                            # Clean old entries (outside time window)
                            cutoff_time = t - self._triangulation_time_window
                            self._obs1_recent_anomalies = {
                                k: v for k, v in self._obs1_recent_anomalies.items() if k >= cutoff_time
                            }
                            # Update status line and direction plot
                            try:
                                self.appWin.update_obs1_direction(
                                    f"[OBS1] azimuth={az_str} ({dir_str}), inclination={inc_str}, |ΔB|={mag_nT:.1f} nT"
                                )
                                self.appWin.update_anomaly_direction_plot(
                                    az_deg, inc_deg, mag_nT, obs_label="OBS1", timestamp=t
                                )
                            except Exception:
                                pass
                            # Attempt triangulation if OBS2 has recent anomaly
                            self._attempt_triangulation(t)
                        else:
                            # OBS1 but components not found at this time
                            self.log(
                                f"[{ctx.display_name}] Anomaly detected | time={t} | magnitude={v:.1f} nT | "
                                f"direction: unavailable (components not found at this timestamp)",
                                level="Info",
                            )
                else:
                    # OBS1 but baseline not set yet
                    for t, v in newly_added:
                        self.log(
                            f"[{ctx.display_name}] Anomaly detected | time={t} | magnitude={v:.1f} nT | "
                            f"direction: unavailable (baseline not computed yet)",
                            level="Info",
                        )
            # Direction finding for OBS2: interpolate all three sensor vectors, fuse, then compute direction.
            elif self.csv_enabled and is_obs2_sensor(sensor_id) and self._obs2_vector_by_sensor:
                self._ensure_obs2_baseline()
                if self._obs2_sensor_baseline_xyz is not None:
                    for t, v in newly_added:
                        comp = self._get_obs2_components_at_time(t)
                        if comp is not None:
                            d_x, d_y, d_z = comp
                            az_deg, inc_deg, mag_nT = self._direction_from_delta_vector(d_x, d_y, d_z)
                            az_str = f"{az_deg:.1f}°" if az_deg is not None else "N/A (vertical)"
                            inc_str = f"{inc_deg:.1f}°" if inc_deg is not None else "N/A"
                            dir_str = self._format_direction_label(az_deg, observatory="OBS2")
                            cal_suffix = ""
                            cal_status_suffix = ""
                            if self.csv_enabled and self._obs2_calibration_enabled:
                                # CSV-only supervised mapping from full 3D features to global GT azimuth.
                                self._ensure_obs2_calibration_model()
                                pred = self._predict_obs2_calibrated_azimuth(t)
                                if pred is not None:
                                    cal_az_deg, cal_conf = pred
                                    cal_az_str = f"{cal_az_deg:.1f}°"
                                    cal_dir_str = self._format_global_direction_label(cal_az_deg)
                                    if cal_conf >= self._obs2_calibration_conf_threshold:
                                        cal_suffix = (
                                            f" | calibrated(global): azimuth={cal_az_str} ({cal_dir_str}), conf={cal_conf:.2f}"
                                        )
                                        cal_status_suffix = f" | cal={cal_az_str} ({cal_dir_str})"
                                    else:
                                        cal_suffix = (
                                            f" | calibrated(global): {cal_az_str} ({cal_dir_str}), "
                                            f"low confidence={cal_conf:.2f}"
                                        )
                            self.log(
                                f"[{ctx.display_name}] Anomaly detected | time={t} | magnitude={v:.1f} nT | "
                                f"direction(raw): azimuth={az_str} ({dir_str}), inclination={inc_str}, |ΔB|={mag_nT:.1f} nT"
                                f"{cal_suffix}",
                                level="Info",
                            )
                            # Store for triangulation
                            self._obs2_recent_anomalies[t] = (az_deg, inc_deg, mag_nT)
                            # Clean old entries (outside time window)
                            cutoff_time = t - self._triangulation_time_window
                            self._obs2_recent_anomalies = {
                                k: v for k, v in self._obs2_recent_anomalies.items() if k >= cutoff_time
                            }
                            # Update status line and direction plot
                            try:
                                self.appWin.update_obs1_direction(
                                    f"[OBS2] raw az={az_str} ({dir_str}), inc={inc_str}, |ΔB|={mag_nT:.1f} nT"
                                    f"{cal_status_suffix}"
                                )
                                self.appWin.update_anomaly_direction_plot(
                                    az_deg, inc_deg, mag_nT, obs_label="OBS2", timestamp=t
                                )
                            except Exception:
                                pass
                            # Attempt triangulation if OBS1 has recent anomaly
                            self._attempt_triangulation(t)
                        else:
                            # OBS2 but components not found at this time
                            self.log(
                                f"[{ctx.display_name}] Anomaly detected | time={t} | magnitude={v:.1f} nT | "
                                f"direction: unavailable (components not found at this timestamp)",
                                level="Info",
                            )
                else:
                    # OBS2 but baseline not set yet
                    for t, v in newly_added:
                        self.log(
                            f"[{ctx.display_name}] Anomaly detected | time={t} | magnitude={v:.1f} nT | "
                            f"direction: unavailable (baseline not computed yet)",
                            level="Info",
                        )
            else:
                # Non-OBS sensor or OBS without component data: log timestamp and magnitude
                for t, v in newly_added:
                    reason = ""
                    if is_obs1_sensor(sensor_id):
                        if not self.csv_enabled:
                            reason = " | direction: unavailable (3D fused direction is enabled only in CSV mode)"
                        elif not self._obs1_vector_by_sensor:
                            reason = " | direction: unavailable (CSV must contain OBS1_1, OBS1_2, OBS1_3)"
                        else:
                            reason = " | direction: unavailable (OBS1 3D vector data not available)"
                    elif is_obs2_sensor(sensor_id):
                        if not self.csv_enabled:
                            reason = " | direction: unavailable (3D fused direction is enabled only in CSV mode)"
                        elif not self._obs2_vector_by_sensor:
                            reason = " | direction: unavailable (CSV must contain OBS2_1, OBS2_2, OBS2_3)"
                        else:
                            reason = " | direction: unavailable (OBS2 3D vector data not available)"
                    else:
                        # Non-OBS sensor: clear direction status line and plot
                        try:
                            self.appWin.update_obs1_direction("— (direction only available for OBS1/OBS2)")
                            self.appWin.update_anomaly_direction_plot(None, None, 0.0, "")
                        except Exception:
                            pass
                    self.log(
                        f"[{ctx.display_name}] Anomaly detected | time={t} | magnitude={v:.1f} nT{reason}",
                        level="Info",
                    )

        self.log(
            f"[{ctx.display_name}] matched_pairs={len(differences_df)} total_anomalies={len(ctx.anomaly_times)} "
            f"threshold={threshold:.2f} nT",
            level="Info",
        )


if __name__ == "__main__":
    # Use the temp app class (multi-sensor TimeSeries tabs)
    app = ApplicationTemp([])
    sys.exit(app.exec())


