"""
record_tab.py

Data Logging view for the radar moisture-tomography project.

Each recording produces one .h5 file and one row in a single master CSV file
named data_log.csv in the selected save folder. The CSV row stores the same
trial-level features used by the project for ML/training data.

Background calibration is REQUIRED before recording (see REQUIRE_BACKGROUND_FOR_RECORDING):
every CSV row is computed from background-corrected data, so corrected and uncorrected
rows are never mixed. Feature math lives in features.py (shared with the live scan).

The CSV can also be maintained manually from this tab:
- Add H5 recording: copies an existing .h5 into the selected folder and adds
  its calculated feature row to the master CSV.
- Delete selected row: removes the row and deletes the linked .h5 file.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import shutil
import time
import traceback
from datetime import datetime

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QMessageBox,
    QPlainTextEdit,
)

from repeatability import format_report
from features import SCHEMA_VERSION, extract_features, raw_vs_corrected_summary
from close_range import process_frame
from sensor_manager import BASE_STEP_LENGTH_M, SensorManager, split_frame

try:
    from acconeer.exptool import a121
except ImportError:
    a121 = None


CSV_COLUMNS = [
    # --- labels / experiment conditions (NEVER model inputs) ---
    "schema_version",
    "recorded_at",
    "material",
    "moisture_level",          # class label: Dry / Damp / Saturated
    "trial_number",
    "sample_id",               # one physical wetting/setup; used to group train/test splits
    "h5_file",
    "water_amount_ml",
    "time_since_wet_min",
    "measured_distance_mm",    # sensor standoff from the wall for this recording (mm)
    "reference_moisture_pct",  # moisture-meter reading (blank if not measured)
    "background_corrected",    # 1 = background subtracted before features were computed
    # --- calibration / sensor state (for QA and drift analysis; not model inputs) ---
    "leakage_phase_compensated",  # 1 = loopback phase-jitter compensation used
    "zero_offset_m",           # loopback zero-point offset subtracted from distances
    "sensor_temperature_c",    # mean sensor temperature during the recording
    "calibration_temperature_c",  # temperature of the sensor calibration used
    "saturated_frames",        # frames flagged data_saturated (should be 0)
    "distance_filtered",       # 1 = Acconeer distance filter applied before features
    "noise_source",            # tx_off = measured with transmitter off; percentile = fallback
    # --- quality flags ---
    "peaks_found",
    "two_peaks_found",
    "strongest_at_edge",
    # --- raw vs background-corrected signal (sensor units, for analysis; not model inputs) ---
    "raw_peak_amplitude",
    "raw_peak_distance_m",
    "raw_mean_amplitude",
    "raw_peak_energy",
    "raw_integrated_energy",
    "corrected_peak_amplitude",
    "corrected_peak_distance_m",
    "corrected_mean_amplitude",
    "corrected_peak_energy",
    "corrected_integrated_energy",
    # --- window features (primary): front surface vs. region 15-60 mm behind it ---
    "front_distance_m",
    "front_amplitude",
    "rear_energy_ratio_db",
    "rear_energy_ratio_db_std",
    "rear_centroid_offset_m",
    "rear_peak_offset_m",
    "rear_phase_rad",
    "rear_phase_std_rad",
    # --- boundary-peak features ---
    "peak1_distance_m",
    "peak2_distance_m",
    "peak_separation_m",
    "peak1_amplitude",
    "peak2_amplitude",
    "amplitude_ratio",
    "amplitude_ratio_std",
    "peak1_phase_rad",
    "peak2_phase_rad",
    "phase_delta_rad",
    "peak2_phase_std_rad",
    "tail_attenuation_slope",
    "peak1_snr_db",
    "total_energy",
    # --- legacy / diagnostic (strongest-bin and whole-profile statistics) ---
    "peak_distance_m",
    "peak_amplitude_mean",
    "peak_amplitude_std",
    "peak_phase_mean_rad",
    "peak_phase_std_rad",
    "global_attenuation_slope",
    "noise_floor",
    "frame_to_frame_variance",
    "num_frames",
]


def _sidecar_path(h5_path):
    """Background used for a recording is stored next to its raw .h5 so it can be reprocessed."""
    return os.path.splitext(h5_path)[0] + "_background.npz"


def _csv_clean(value):
    """CSV-safe value: NaN/None -> blank, numpy scalars -> python."""
    if value is None:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return "" if math.isnan(float(value)) else float(value)
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    return value


class AcquisitionThread(QThread):
    """Runs client.get_next() in the background so the UI remains responsive."""

    # (range subframe (sweeps, points), loopback subframe (sweeps, 1) or None, temperature, saturated)
    frame_ready = Signal(object, object, object, bool)
    error = Signal(str)

    def __init__(self, client, params):
        super().__init__()
        self.client = client
        self.params = params
        self._running = True

    def run(self):
        try:
            while self._running:
                result = self.client.get_next()
                range_frame, loopback_frame = split_frame(result, self.params)
                self.frame_ready.emit(
                    np.array(range_frame), None if loopback_frame is None else np.array(loopback_frame),
                    getattr(result, "temperature", None), bool(getattr(result, "data_saturated", False)),
                )
        except Exception as e:
            if self._running:
                self.error.emit(str(e))

    def stop(self):
        self._running = False


class RecordTab(QWidget):
    """Data Logging tab. The class name is retained for main.py compatibility."""

    amplitude_ready = Signal(object, object)

    MATERIALS = ["Wood", "Drywall", "Wood + drywall", "Open air (no target)", "Concrete"]
    MOISTURE_LEVELS = ["Dry", "Damp", "Saturated"]
    STOP_OPTIONS = ["Seconds", "Frames"]
    CSV_FILE_NAME = "data_log.csv"
    # True: refuse to record without a valid background calibration (recommended).
    # False: allow it, but rows are flagged background_corrected = 0.
    REQUIRE_BACKGROUND_FOR_RECORDING = True

    def __init__(self, sensor_manager: SensorManager):
        super().__init__()
        self.mgr = sensor_manager
        self.acquisition_thread: AcquisitionThread | None = None
        self.recorder = None
        self.record_file_path: str | None = None
        self.is_recording = False
        self.frame_amplitudes = []
        self.frame_phases = []
        self.frame_raw_iq = []
        self.frame_corrected_iq = []
        self.frame_count = 0
        self.record_start_time = 0.0
        self.background_iq = None
        self.background_signature = None
        self._rec_background = None
        self._rec_meta = None
        self._rec_signature = None
        self._rec_noise = None
        self._rec_params = None
        self._import_note = ""
        self._migration_note = ""
        self.save_folder = os.path.expanduser("~")
        self._stopping = False
        self._build_ui()
        self._refresh_csv_rows()

    def set_background_calibration(self, background_iq, signature=None):
        # signature = SensorManager.background_key(): (settings signature, calibration epoch)
        """Store the complex-IQ background and the sensor-settings signature it was taken with.

        H5Recorder stores the original Acconeer stream; CSV features are derived from the
        background-corrected stream. The signature lets us refuse a stale background.
        """
        self.background_iq = (
            None if background_iq is None
            else np.array(background_iq, dtype=np.complex128, copy=True)
        )
        self.background_signature = signature if background_iq is not None else None
        self._update_background_label()

    def _background_status(self):
        """(usable, message) for the stored background vs the CURRENT sensor settings."""
        if self.background_iq is None:
            return False, (
                "No background calibration. Press Start Measurement, point the sensor at "
                "open air, press 'Calibrate Background', then Stop and record."
            )
        if self.background_signature is not None and self.background_signature != self.mgr.background_key():
            if tuple(self.background_signature)[0] != self.mgr.params.signature():
                return False, "Sensor settings changed since the background was calibrated. Recalibrate."
            return False, "The sensor was recalibrated since the background was taken. Calibrate the background again."
        return True, "Background calibrated; matches current sensor settings."

    def _update_background_label(self, *_):
        if not hasattr(self, "background_label"):
            return
        ok, msg = self._background_status()
        self.background_label.setText(("\u2714 " if ok else "\u26A0 ") + msg)
        self.background_label.setStyleSheet("color: #7fd88f;" if ok else "color: #e5c85c;")

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Two columns so everything fits an 800x480 display without scrolling:
        #   left : Recording + Experiment conditions + Controls (background status, Start/Stop)
        #   right: Save location + Manage CSV dataset + status
        # (main.py wraps this tab in a scroll area as a fallback for even smaller windows.)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(8)
        left = QVBoxLayout()
        left.setSpacing(6)
        right = QVBoxLayout()
        right.setSpacing(6)
        outer.addLayout(left, 1)
        outer.addLayout(right, 1)

        info_box = QGroupBox("Recording")
        form = QFormLayout(info_box)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(5)

        self.material_combo = QComboBox()
        self.material_combo.addItems(self.MATERIALS)
        form.addRow("Materials", self.material_combo)

        self.moisture_combo = QComboBox()
        self.moisture_combo.addItems(self.MOISTURE_LEVELS)
        form.addRow("Moisture level", self.moisture_combo)

        trail_row = QHBoxLayout()
        self.trail_edit = QLineEdit("1")
        self.trail_edit.setPlaceholderText("Enter trail number")
        self.trail_edit.setToolTip(
            "The trail number increases by 1 after each completed recording."
        )
        trail_row.addWidget(self.trail_edit)
        form.addRow("Trail Number", trail_row)

        stop_row = QHBoxLayout()
        self.stop_mode_combo = QComboBox()
        self.stop_mode_combo.addItems(self.STOP_OPTIONS)
        self.stop_mode_combo.setFixedWidth(100)
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(0.1, 1_000_000.0)
        self.amount_spin.setDecimals(1)
        self.amount_spin.setValue(10.0)
        self.amount_spin.setSingleStep(1.0)
        stop_row.addWidget(self.stop_mode_combo)
        stop_row.addWidget(self.amount_spin, 1)
        form.addRow("Stop After", stop_row)

        self.save_h5_check = QCheckBox("Save raw .h5 file (more disk space)")
        self.save_h5_check.setChecked(False)
        self.save_h5_check.setToolTip(
            "When enabled, saves the raw Acconeer recording for later reprocessing. "
            "When disabled, only the calculated feature row is saved to data_log.csv."
        )
        form.addRow("Raw data", self.save_h5_check)

        left.addWidget(info_box)

        cond_box = QGroupBox("Experiment conditions")
        cond = QGridLayout(cond_box)
        cond.setHorizontalSpacing(8)
        cond.setVerticalSpacing(4)

        self.sample_edit = QLineEdit("S1")
        self.sample_edit.setToolTip(
            "Name of ONE physical wetting/setup. Change it whenever you re-wet or rebuild the "
            "sample. Training keeps every row with the same Sample ID on the same side of the "
            "train/test split, so results are not inflated."
        )
        self.distance_spin = QDoubleSpinBox()
        self.distance_spin.setRange(0.0, 300.0)
        self.distance_spin.setDecimals(1)
        self.distance_spin.setSingleStep(0.5)
        self.distance_spin.setSuffix(" mm")
        self.distance_spin.setToolTip("Gap between the sensor lens and the wall for this recording.")
        self.water_spin = QDoubleSpinBox()
        self.water_spin.setRange(0.0, 5000.0)
        self.water_spin.setDecimals(0)
        self.water_spin.setSingleStep(5.0)
        self.water_spin.setSuffix(" mL")
        self.time_wet_spin = QDoubleSpinBox()
        self.time_wet_spin.setRange(0.0, 100000.0)
        self.time_wet_spin.setDecimals(0)
        self.time_wet_spin.setSingleStep(5.0)
        self.time_wet_spin.setSuffix(" min")
        self.ref_moisture_spin = QDoubleSpinBox()
        self.ref_moisture_spin.setRange(-1.0, 100.0)
        self.ref_moisture_spin.setDecimals(1)
        self.ref_moisture_spin.setSingleStep(0.5)
        self.ref_moisture_spin.setSuffix(" %")
        self.ref_moisture_spin.setSpecialValueText("Not measured")
        self.ref_moisture_spin.setValue(-1.0)
        self.ref_moisture_spin.setToolTip(
            "Moisture-meter reading of the stud for this recording. Leave as 'Not measured' "
            "if you have no meter; the model can still be trained on the Dry/Damp/Saturated label."
        )

        def cell(title, widget):
            # label above the field: narrower than label-beside-field, so the tab fits 800 px
            box = QVBoxLayout()
            box.setSpacing(1)
            caption = QLabel(title)
            caption.setObjectName("featureName")
            box.addWidget(caption)
            box.addWidget(widget)
            return box

        cond.addLayout(cell("Wall sample ID", self.sample_edit), 0, 0)
        cond.addLayout(cell("Sensor standoff", self.distance_spin), 0, 1)
        cond.addLayout(cell("Water added", self.water_spin), 1, 0)
        cond.addLayout(cell("Time since wet", self.time_wet_spin), 1, 1)
        cond.addLayout(cell("Meter reading", self.ref_moisture_spin), 2, 0)
        for column in range(2):
            cond.setColumnStretch(column, 1)
        left.addWidget(cond_box)

        save_box = QGroupBox("Save location")
        save_layout = QVBoxLayout(save_box)
        folder_row = QHBoxLayout()
        self.browse_btn = QPushButton("Choose folder…")
        folder_row.addWidget(self.browse_btn)
        self.folder_label = QLabel(f"Save folder: {self.save_folder}")
        self.folder_label.setObjectName("summaryLabel")
        self.folder_label.setWordWrap(True)
        folder_row.addWidget(self.folder_label, 1)
        save_layout.addLayout(folder_row)

        self.csv_label = QLabel()
        self.csv_label.setObjectName("summaryLabel")
        self.csv_label.setWordWrap(True)
        save_layout.addWidget(self.csv_label)
        right.addWidget(save_box)

        controls_box = QGroupBox("Controls")
        controls_outer = QVBoxLayout(controls_box)
        self.background_label = QLabel()
        self.background_label.setWordWrap(True)
        controls_outer.addWidget(self.background_label)
        controls = QHBoxLayout()
        controls_outer.addLayout(controls)
        self.start_btn = QPushButton("Start recording")
        self.start_btn.setObjectName("primaryButton")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch(1)
        right.insertWidget(0, controls_box)   # Start/Stop always visible, top right
        left.addStretch(1)

        # --- Master CSV management ---
        manage_box = QGroupBox("Manage CSV dataset")
        manage_layout = QVBoxLayout(manage_box)

        operation_row = QHBoxLayout()
        operation_row.addWidget(QLabel("Action"))
        self.csv_action_combo = QComboBox()
        self.csv_action_combo.addItems([
            "Add H5 recording to CSV",
            "Delete CSV row + H5 file",
        ])
        # Long item text must not set the minimum width of the whole tab.
        self.csv_action_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.csv_action_combo.setMinimumContentsLength(8)
        operation_row.addWidget(self.csv_action_combo, 1)
        manage_layout.addLayout(operation_row)

        self.manage_stack_layout = QVBoxLayout()

        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("H5 file"))
        self.add_h5_btn = QPushButton("Choose H5 file…")
        add_row.addWidget(self.add_h5_btn)
        self.add_h5_label = QLabel("No external H5 selected")
        self.add_h5_label.setObjectName("summaryLabel")
        self.add_h5_label.setWordWrap(True)
        self.add_h5_label.setMinimumWidth(0)
        add_row.addWidget(self.add_h5_label, 1)
        self.manage_add_widget = QWidget()
        self.manage_add_widget.setLayout(add_row)
        self.manage_stack_layout.addWidget(self.manage_add_widget)

        delete_row = QHBoxLayout()
        delete_row.addWidget(QLabel("CSV row"))
        self.csv_row_combo = QComboBox()
        self.csv_row_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.csv_row_combo.setMinimumContentsLength(8)
        delete_row.addWidget(self.csv_row_combo, 1)
        self.delete_row_btn = QPushButton("Delete selected row")
        delete_row.addWidget(self.delete_row_btn)
        self.manage_delete_widget = QWidget()
        self.manage_delete_widget.setLayout(delete_row)
        self.manage_stack_layout.addWidget(self.manage_delete_widget)

        manage_layout.addLayout(self.manage_stack_layout)

        self.dataset_status_label = QLabel("CSV dataset: 0 recordings")
        self.dataset_status_label.setObjectName("summaryLabel")
        self.dataset_status_label.setWordWrap(True)
        manage_layout.addWidget(self.dataset_status_label)
        right.addWidget(manage_box)

        self.status_label = QLabel("Idle.")
        self.status_label.setObjectName("summaryLabel")
        self.status_label.setWordWrap(True)
        right.addWidget(self.status_label)

        # --- Repeatability / standoff comparison (reads the master CSV) ---
        rep_box = QGroupBox("Repeatability & standoff comparison")
        rep_layout = QVBoxLayout(rep_box)
        rep_hint = QLabel(
            "Record each condition 3+ times, moving the material away and back between "
            "recordings, at each standoff. Then analyze: spread = placement-to-placement "
            "variation; d' = how well Dry/Damp/Saturated (and stud vs. no stud) separate."
        )
        rep_hint.setObjectName("summaryLabel")
        rep_hint.setWordWrap(True)
        rep_layout.addWidget(rep_hint)
        self.analyze_btn = QPushButton("Analyze data_log.csv")
        rep_layout.addWidget(self.analyze_btn)
        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setMinimumHeight(160)
        self.report_text.setStyleSheet("font-family: Consolas, 'DejaVu Sans Mono', monospace;")
        rep_layout.addWidget(self.report_text)
        right.addWidget(rep_box, 1)

        self.mgr.params_changed.connect(self._update_background_label)
        self._update_background_label()
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.browse_btn.clicked.connect(self._on_browse)
        self.add_h5_btn.clicked.connect(self._on_add_h5)
        self.delete_row_btn.clicked.connect(self._on_delete_row)
        self.analyze_btn.clicked.connect(self._on_analyze)
        self.csv_action_combo.currentTextChanged.connect(self._update_csv_action_ui)
        self.stop_mode_combo.currentTextChanged.connect(self._on_stop_mode_changed)
        self._on_stop_mode_changed(self.stop_mode_combo.currentText())
        self._update_csv_action_ui(self.csv_action_combo.currentText())
        self._update_csv_label()

    # ------------------------------------------------------------------
    # Paths / CSV helpers
    # ------------------------------------------------------------------
    @property
    def csv_file_path(self):
        return os.path.join(self.save_folder, self.CSV_FILE_NAME)

    def _update_csv_label(self):
        self.csv_label.setText(f"Master CSV: {self.csv_file_path}")

    def _on_browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose save folder", self.save_folder
        )
        if folder:
            self.save_folder = folder
            self.folder_label.setText(f"Save folder: {folder}")
            self._update_csv_label()
            self._refresh_csv_rows()

    def _trail_number(self):
        text = self.trail_edit.text().strip()
        try:
            value = int(text)
        except ValueError:
            raise ValueError("Trail Number must be a whole number.")
        if value < 1:
            raise ValueError("Trail Number must be 1 or greater.")
        return value

    def _make_h5_path(self, trail_number):
        material = self.material_combo.currentText().replace(" ", "_")
        moisture = self.moisture_combo.currentText()
        base = f"{material}_{moisture}_{trail_number}"
        return os.path.join(self.save_folder, f"{base}.h5")

    def _ensure_csv_exists(self):
        os.makedirs(self.save_folder, exist_ok=True)
        path = self.csv_file_path
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            if header != CSV_COLUMNS:
                # Old-format file: archive it untouched and start a fresh one, so rows with
                # different feature definitions are never mixed into one dataset.
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                legacy = os.path.join(self.save_folder, f"data_log_legacy_{stamp}.csv")
                os.replace(path, legacy)
                self._migration_note = (
                    f"Old-format CSV archived as {os.path.basename(legacy)}; started a new "
                    f"{self.CSV_FILE_NAME} with the current column layout."
                )
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    def _read_csv_rows(self):
        self._ensure_csv_exists()
        with open(self.csv_file_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                # Older versions of the app did not have h5_file. Keep them
                # visible, but they cannot be safely linked for deletion.
                row.setdefault("h5_file", "")
                rows.append(row)
            return rows

    def _write_csv_rows(self, rows):
        os.makedirs(self.save_folder, exist_ok=True)
        temp_path = self.csv_file_path + ".tmp"
        with open(temp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_clean(row.get(key, "")) for key in CSV_COLUMNS})
        os.replace(temp_path, self.csv_file_path)

    def _append_csv_row(self, row):
        """Append a completed trial directly; do not rewrite the whole CSV.

        Recording rows are append-only, so this avoids leaving data_log.csv.tmp
        behind if a Windows file replacement fails while the CSV is open.
        The temp-file rewrite remains reserved for operations that edit/delete rows.
        """
        self._ensure_csv_exists()
        with open(self.csv_file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow({key: _csv_clean(row.get(key, "")) for key in CSV_COLUMNS})

    def _refresh_csv_rows(self):
        try:
            rows = self._read_csv_rows()
        except Exception as e:
            self.dataset_status_label.setText(f"CSV dataset error: {e}")
            return

        self.csv_row_combo.clear()
        for index, row in enumerate(rows):
            label = (
                f"{index + 1}: {row.get('material', '')} | "
                f"{row.get('moisture_level', '')} | {row.get('water_amount_ml', '')} mL | "
                f"{row.get('measured_distance_mm', '')} mm | Trail {row.get('trial_number', '')} | "
                f"{row.get('h5_file') or '(no H5 linked)'}"
            )
            self.csv_row_combo.addItem(label, index)

        self.dataset_status_label.setText(
            f"CSV dataset: {len(rows)} recording{'s' if len(rows) != 1 else ''}"
        )
        if self._migration_note:
            self.status_label.setText(self._migration_note)
            self._migration_note = ""

    # ------------------------------------------------------------------
    # CSV management
    # ------------------------------------------------------------------
    def _update_csv_action_ui(self, action):
        adding = action == "Add H5 recording to CSV"
        self.manage_add_widget.setVisible(adding)
        self.manage_delete_widget.setVisible(not adding)

    def _on_add_h5(self):
        source, _ = QFileDialog.getOpenFileName(
            self,
            "Choose H5 recording to add",
            self.save_folder,
            "Acconeer recordings (*.h5);;All files (*)",
        )
        if not source:
            return

        try:
            if a121 is None:
                raise RuntimeError("The Acconeer A121 package is not available.")

            trail = self._trail_number()
            meta = self._collect_metadata(trail)
            destination = self._make_h5_path(trail)
            if os.path.abspath(source) == os.path.abspath(destination):
                raise ValueError("The selected H5 file is already in the save location with this name.")
            if os.path.exists(destination):
                raise FileExistsError(
                    f"{os.path.basename(destination)} already exists. Choose another Trail Number."
                )

            rows = self._read_csv_rows()
            if any(
                row.get("h5_file") == os.path.basename(destination)
                for row in rows
            ):
                raise ValueError("This H5 filename is already linked in the master CSV.")

            shutil.copy2(source, destination)
            try:
                row = self._calculate_features_from_h5(destination, meta, source)
                self._append_csv_row(row)
                # keep the background that belongs to this recording next to the copied .h5
                if os.path.exists(_sidecar_path(source)):
                    shutil.copy2(_sidecar_path(source), _sidecar_path(destination))
            except Exception:
                if os.path.exists(destination):
                    os.remove(destination)
                raise

            self.add_h5_label.setText(os.path.basename(source))
            self.trail_edit.setText(str(trail + 1))
            self._refresh_csv_rows()
            self.status_label.setText(
                f"Added {os.path.basename(destination)} to the master CSV ({self._import_note})."
            )
        except Exception as e:
            error_text = f"{type(e).__name__}: {e}"
            details = traceback.format_exc()
            message_box = QMessageBox(self)
            message_box.setIcon(QMessageBox.Warning)
            message_box.setWindowTitle("Add H5 recording")
            message_box.setText("The H5 recording could not be added to the CSV.")
            message_box.setInformativeText(error_text)
            message_box.setDetailedText(details)
            message_box.exec()

    def _on_analyze(self):
        try:
            rows = self._read_csv_rows()
            self.report_text.setPlainText(format_report(rows))
        except Exception as e:
            self.report_text.setPlainText(f"Could not analyze the CSV: {e}")

    def _on_delete_row(self):
        index = self.csv_row_combo.currentData()
        if index is None:
            QMessageBox.information(self, "Delete CSV row", "There are no CSV rows to delete.")
            return

        rows = self._read_csv_rows()
        if not (0 <= int(index) < len(rows)):
            self._refresh_csv_rows()
            return

        row = rows[int(index)]
        h5_name = row.get("h5_file", "").strip()
        description = (
            f"{row.get('material', '')}, {row.get('moisture_level', '')}, "
            f"Trail {row.get('trial_number', '')}"
        )

        answer = QMessageBox.question(
            self,
            "Delete recording",
            f"Delete the CSV row for {description}?\n\n"
            f"The linked H5 file will also be deleted from:\n{self.save_folder}\n\n"
            f"This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        try:
            if h5_name:
                h5_path = os.path.join(self.save_folder, os.path.basename(h5_name))
                if os.path.exists(_sidecar_path(h5_path)):
                    os.remove(_sidecar_path(h5_path))
                if os.path.exists(h5_path):
                    os.remove(h5_path)
                else:
                    QMessageBox.information(
                        self,
                        "H5 file not found",
                        f"The CSV row was removed, but this H5 file was not found:\n{h5_path}",
                    )

            del rows[int(index)]
            self._write_csv_rows(rows)
            self._refresh_csv_rows()
            self.status_label.setText(f"Deleted CSV row and linked H5: {h5_name or '(none)'}")
        except Exception as e:
            QMessageBox.critical(self, "Delete recording", str(e))

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _on_stop_mode_changed(self, mode):
        if mode == "Seconds":
            self.amount_spin.setSuffix(" s")
            self.amount_spin.setDecimals(1)
        else:
            self.amount_spin.setSuffix(" frames")
            self.amount_spin.setDecimals(0)
            self.amount_spin.setValue(max(1, round(self.amount_spin.value())))

    def _on_start(self):
        if not self.mgr.is_connected():
            self.status_label.setText("Not connected. Go to Settings and connect first.")
            return

        if a121 is None:
            self.status_label.setText("Acconeer A121 package is not available.")
            return

        # The sensor allows one session at a time: the live scan must be stopped first.
        try:
            if self.mgr.client.session_is_started:
                self.status_label.setText("A live session is running. Press Stop (top right) before recording.")
                return
        except Exception:
            pass

        # Background calibration comes first: no valid background, no recording.
        background_ok, background_msg = self._background_status()
        if self.REQUIRE_BACKGROUND_FOR_RECORDING and not background_ok:
            self.status_label.setText(background_msg)
            return

        try:
            trail_number = self._trail_number()
            # Snapshot everything that goes into the row NOW, so edits made while the
            # recording runs (or a recalibration) cannot change this row afterwards.
            self._rec_meta = self._collect_metadata(trail_number)
            self._rec_background = self.background_iq.copy() if background_ok else None
            self._rec_signature = self.mgr.params.signature()
            amount = self.amount_spin.value()
            if self.stop_mode_combo.currentText() == "Frames":
                amount = max(1, int(round(amount)))

            os.makedirs(self.save_folder, exist_ok=True)
            self._ensure_csv_exists()
            save_h5 = self.save_h5_check.isChecked()
            self.record_file_path = self._make_h5_path(trail_number) if save_h5 else None

            if save_h5:
                if os.path.exists(self.record_file_path):
                    self.status_label.setText(
                        "An H5 file with this trail number already exists. Choose the next trail number."
                    )
                    self.record_file_path = None
                    return

                rows = self._read_csv_rows()
                if any(
                    row.get("h5_file") == os.path.basename(self.record_file_path)
                    for row in rows
                ):
                    self.status_label.setText(
                        "This H5 file is already linked in the master CSV. Choose the next trail number."
                    )
                    self.record_file_path = None
                    return

            sensor_config = self.mgr.params.to_sensor_config()
            self.mgr.ensure_calibrated()
            # Reuses the cached sensor calibration, so the background stays valid.
            self.mgr.setup_session(sensor_config)
            if background_ok and self.background_signature != self.mgr.background_key():
                # The sensor had to be re-calibrated: the background no longer matches.
                self._cleanup_after_failed_start()
                self.background_signature = None
                self.status_label.setText(
                    "The sensor was re-calibrated, so the background is no longer valid. "
                    "Start Measurement, Calibrate Background in open air, Stop, then record."
                )
                self._update_background_label()
                return
            self._rec_params = copy.deepcopy(self.mgr.params)
            self._rec_noise = self.mgr.current_noise_level(self._rec_params)
            self._rec_meta.update({
                "zero_offset_m": self.mgr.zero_offset_m,
                "calibration_temperature_c": self.mgr.calibration_temperature,
                "leakage_phase_compensated": int(bool(background_ok and self.mgr.params.use_loopback_reference)),
                "distance_filtered": int(bool(self.mgr.params.distance_filter)),
            })
            if save_h5:
                self.recorder = a121.H5Recorder(self.record_file_path, self.mgr.client)
            self.mgr.client.start_session()
            self.is_recording = True

            self.distances_m = self.mgr.distances_m()
            self.frame_count = 0
            self.frame_temperatures = []
            self.saturated_frames = 0
            self.frame_amplitudes = []
            self.frame_phases = []
            self.frame_raw_iq = []
            self.frame_corrected_iq = []
            self.record_start_time = time.monotonic()
            self._stopping = False

            self.acquisition_thread = AcquisitionThread(self.mgr.client, self._rec_params)
            self.acquisition_thread.frame_ready.connect(self._on_frame)
            self.acquisition_thread.error.connect(self._on_error)
            self.acquisition_thread.start()

            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            mode = self.stop_mode_combo.currentText().lower()
            storage = "raw H5 + background-corrected CSV features" if save_h5 else "background-corrected CSV features only"
            correction_note = "" if self._rec_background is not None else " (no background calibration: CSV uses uncorrected data)"
            self.status_label.setText(f"Recording… Stop after {amount:g} {mode}. Saving: {storage}{correction_note}.")
        except Exception as e:
            self._cleanup_after_failed_start()
            self.status_label.setText(f"Failed to start: {e}")

    def _on_frame(self, range_frame, loopback_frame, temperature, saturated):
        # Frames already queued by the thread can arrive after the recording stopped; ignore
        # them, otherwise a second _on_stop overwrites the "Saved" confirmation message.
        if self._stopping or not self.is_recording:
            return

        self.frame_count += 1
        if temperature is not None:
            self.frame_temperatures.append(float(temperature))
        self.saturated_frames += int(bool(saturated))

        # Coherent mean -> phase-stable leakage cancellation -> distance filter (same as live).
        # H5Recorder remains a raw archive so the original sensor data is preserved.
        p = self._rec_params
        avg_sweep, feature_sweep = process_frame(
            range_frame, loopback_frame, self._rec_background,
            p.profile, p.step_length, p.start_point, p.distance_filter,
        )
        self.amplitude_ready.emit(self.distances_m, np.abs(avg_sweep))
        self.frame_amplitudes.append(np.abs(feature_sweep))
        self.frame_phases.append(np.angle(feature_sweep))
        self.frame_raw_iq.append(np.array(avg_sweep, dtype=complex))
        self.frame_corrected_iq.append(np.array(feature_sweep, dtype=complex))

        mode = self.stop_mode_combo.currentText()
        amount = self.amount_spin.value()
        if mode == "Frames":
            reached_limit = self.frame_count >= int(round(amount))
        else:
            reached_limit = time.monotonic() - self.record_start_time >= amount

        if reached_limit:
            self._on_stop(auto=True)

    def _on_error(self, message):
        self.status_label.setText(f"Error: {message}")
        self._on_stop()

    def _cleanup_after_failed_start(self):
        try:
            if self.mgr.client is not None and self.mgr.client.session_is_started:
                self.mgr.client.stop_session()
        except Exception:
            pass
        self._close_recorder()
        self.is_recording = False
        if self.record_file_path and os.path.exists(self.record_file_path):
            try:
                os.remove(self.record_file_path)
            except Exception:
                pass
        self.record_file_path = None

    def _close_recorder(self):
        if self.recorder is not None:
            try:
                self.recorder.close()
            except Exception:
                pass
            self.recorder = None

    def _on_stop(self, auto=False):
        if self._stopping:
            return
        self._stopping = True

        if self.acquisition_thread is not None:
            self.acquisition_thread.stop()
            self.acquisition_thread.wait(1000)
            self.acquisition_thread = None

        try:
            if self.mgr.client is not None and self.mgr.client.session_is_started:
                self.mgr.client.stop_session()
        except Exception:
            pass

        was_recording = self.is_recording
        self._close_recorder()
        self.is_recording = False

        if was_recording and self.frame_count > 0:
            try:
                h5_filename = (
                    os.path.basename(self.record_file_path)
                    if self.record_file_path else ""
                )
                corrected = self._rec_background is not None
                meta = dict(self._rec_meta)
                meta["sensor_temperature_c"] = (
                    float(np.mean(self.frame_temperatures)) if self.frame_temperatures else None
                )
                meta["saturated_frames"] = self.saturated_frames
                row = self._row_from_window(meta, h5_filename, corrected)
                self._append_csv_row(row)
                if self.record_file_path and corrected:
                    self._save_sidecar(self.record_file_path)

                self.trail_edit.setText(str(int(self._rec_meta["trial_number"]) + 1))
                self._refresh_csv_rows()

                action = "Automatically saved" if auto else "Saved"
                saved_name = os.path.basename(self.record_file_path) if self.record_file_path else "CSV only"
                self.status_label.setText(
                    f"{action}: {saved_name} → master CSV ({self.frame_count} frames)"
                )
            except Exception as e:
                self.status_label.setText(
                    f"Recording data collected, but the master CSV could not be updated: {e}"
                )
        elif was_recording:
            self.status_label.setText("Recording stopped without any frames; no CSV row was added.")
            try:
                if self.record_file_path and os.path.exists(self.record_file_path):
                    os.remove(self.record_file_path)
            except Exception:
                pass
        else:
            self.status_label.setText("Stopped.")

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.record_file_path = None
        self._stopping = False

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _collect_metadata(self, trail_number):
        sample_id = self.sample_edit.text().strip()
        if not sample_id:
            raise ValueError("Enter a Sample ID (e.g. S1). It groups recordings of one physical setup.")
        ref = self.ref_moisture_spin.value()
        return {
            "material": self.material_combo.currentText(),
            "moisture_level": self.moisture_combo.currentText(),
            "trial_number": trail_number,
            "sample_id": sample_id,
            "water_amount_ml": self.water_spin.value(),
            "time_since_wet_min": self.time_wet_spin.value(),
            "measured_distance_mm": self.distance_spin.value(),
            "reference_moisture_pct": None if ref < 0 else ref,
        }

    def _row_from_window(self, meta, h5_filename, corrected):
        return self._row_from_arrays(
            meta,
            np.asarray(self.frame_amplitudes, dtype=float),
            np.asarray(self.frame_phases, dtype=float),
            np.asarray(self.distances_m, dtype=float),
            h5_filename,
            corrected,
            raw_iq=np.asarray(self.frame_raw_iq, dtype=complex),
            corrected_iq=np.asarray(self.frame_corrected_iq, dtype=complex) if corrected else None,
            noise_level=self._rec_noise,
        )

    @staticmethod
    def _row_from_arrays(meta, amplitude_window, phase_window, distances_m, h5_filename, corrected,
                         raw_iq=None, corrected_iq=None, noise_level=None):
        """One CSV row = experiment metadata + features from the shared features.py."""
        if amplitude_window.ndim != 2 or len(amplitude_window) == 0:
            raise ValueError("No measurement frames were collected.")
        features = extract_features(
            amplitude_window, phase_window, distances_m, background_corrected=corrected,
            noise_level=noise_level,
        )
        row = {
            "schema_version": SCHEMA_VERSION,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "h5_file": h5_filename,
        }
        row.update(meta)
        row["background_corrected"] = int(bool(corrected))
        row.update({key: features[key] for key in CSV_COLUMNS if key in features})
        if raw_iq is not None and len(raw_iq) > 0:
            raw = np.asarray(raw_iq, dtype=complex)
            # corrected_iq is the leakage-cancelled stream itself (with the loopback reference
            # it is NOT simply raw - background).
            corrected_iq = np.asarray(corrected_iq, dtype=complex) if (corrected and corrected_iq is not None) else None
            row.update(raw_vs_corrected_summary(raw, corrected_iq, distances_m))
        return row

    def _save_sidecar(self, h5_path):
        np.savez(
            _sidecar_path(h5_path),
            background_iq=self._rec_background,
            signature=np.array(json.dumps(list(self._rec_signature))),
            # Phase-referenced background (loopback) + zero-point offset, so an import reproduces the row.
            phase_referenced=np.array(bool(self._rec_signature[9])),
            zero_offset_m=np.array(np.nan if self.mgr.zero_offset_m is None else self.mgr.zero_offset_m),
            noise_level=np.array(np.nan if self._rec_noise is None else self._rec_noise),
            distance_filter=np.array(bool(self._rec_params.distance_filter)),
        )

    @staticmethod
    def _range_and_loopback_configs(config):
        """(range subsweep config, has_loopback_reference) from a stored SensorConfig."""
        subsweeps = list(getattr(config, "subsweeps", [])) or [config]
        if len(subsweeps) >= 2 and bool(subsweeps[0].enable_loopback):
            return subsweeps[1], True
        return subsweeps[0], False

    @classmethod
    def _signature_from_config(cls, config):
        """Same tuple as SensorParams.signature(), read from an H5's stored SensorConfig.

        Profile is a plain Enum (use .value); PRF is matched by name. Raises on failure
        rather than returning None, so an unreadable config can never skip the safety check.
        Old single-subsweep recordings map to use_loopback_reference = False.
        """
        ss, has_lb = cls._range_and_loopback_configs(config)
        return (
            int(ss.start_point), int(ss.num_points), int(ss.step_length),
            int(config.sweeps_per_frame), int(ss.hwaas), int(ss.profile.value),
            int(ss.receiver_gain), ss.prf.name, bool(ss.enable_tx),
            bool(has_lb), bool(ss.phase_enhancement),
            bool(ss.iq_imbalance_compensation),
        )

    def _background_for_import(self, source_path, h5_signature):
        """Pick the background for an imported .h5: the one saved with it, else the current one."""
        sidecar = _sidecar_path(source_path)
        self._import_offsets = None
        self._import_noise = None
        self._import_filter = None
        if os.path.exists(sidecar):
            with np.load(sidecar, allow_pickle=False) as saved:
                background = np.array(saved["background_iq"], dtype=np.complex128)
                saved_signature = tuple(json.loads(str(saved["signature"])))
                if "zero_offset_m" in saved.files:
                    zero = float(saved["zero_offset_m"])
                    self._import_offsets = None if np.isnan(zero) else zero
                if "noise_level" in saved.files:
                    noise = float(saved["noise_level"])
                    self._import_noise = None if np.isnan(noise) else noise
                    self._import_filter = bool(saved["distance_filter"])
            if h5_signature is not None and saved_signature != tuple(h5_signature):
                raise ValueError("The saved background does not match this recording's sensor settings.")
            return background, "background saved with the recording"

        usable, message = self._background_status()
        if usable and self.background_signature is not None and h5_signature is not None \
                and tuple(self.background_signature) != tuple(h5_signature):
            usable, message = False, "This recording used different sensor settings than the current background."
        if usable:
            return self.background_iq, "CURRENT calibration applied; only valid if recorded under the same conditions"
        return None, message

    def _calculate_features_from_h5(self, h5_path, meta, source_path):
        """Load an A121 H5 recording and calculate the same CSV row the live recorder writes."""
        if a121 is None:
            raise RuntimeError("The Acconeer A121 package is not available.")

        # Acconeer recommends load_record() for loading H5 recordings; the loaded Record
        # exposes all recorded frames through ``frames`` (plural).
        record = a121.load_record(h5_path)
        frame = np.asarray(record.frames)
        if frame.ndim == 2:
            frame = frame[np.newaxis, ...]
        if frame.ndim != 3:
            raise ValueError(
                f"Unexpected H5 frame shape: {frame.shape}. "
                "Expected (frames, sweeps, distance_points)."
            )

        config = record.session_config.sensor_config
        background, note = self._background_for_import(source_path, self._signature_from_config(config))
        if background is None and self.REQUIRE_BACKGROUND_FOR_RECORDING:
            raise ValueError(f"Cannot add this recording without a valid background. {note}")

        range_cfg, has_lb = self._range_and_loopback_configs(config)
        try:
            subframes = [np.asarray(sf) for sf in record.stacked_results.subframes]
        except Exception:
            subframes = None
        if has_lb and subframes is not None and len(subframes) >= 2:
            range_frames, lb_frames = subframes[1], subframes[0]
        elif has_lb:
            range_frames, lb_frames = frame[:, :, 1:], frame[:, :, :1]
        else:
            range_frames, lb_frames = frame, None

        if background is not None and background.shape != range_frames.shape[2:]:
            raise ValueError("Background and recording have different numbers of distance points.")
        use_filter = self.mgr.params.distance_filter if self._import_filter is None else self._import_filter
        profile = int(range_cfg.profile.value)
        processed = [
            process_frame(range_frames[k], None if lb_frames is None else lb_frames[k], background,
                          profile, int(range_cfg.step_length), int(range_cfg.start_point), use_filter)
            for k in range(range_frames.shape[0])
        ]
        raw_mean_iq = np.stack([pr[0] for pr in processed])
        mean_iq = np.stack([pr[1] for pr in processed])
        corrected = background is not None
        # Noise: saved with the recording, else the current TX-off level if the settings match.
        noise = self._import_noise
        if noise is None and self._signature_from_config(config) == self.mgr.params.signature():
            noise = self.mgr.current_noise_level()
        self._import_note = note if corrected else "NOT background-corrected"

        # Distances: metadata base step, minus the zero-point offset saved with the recording
        # (or the current one if the recording predates offset calibration).
        try:
            base_step = float(record.metadata.base_step_length_m)
        except Exception:
            base_step = BASE_STEP_LENGTH_M
        if getattr(self, "_import_offsets", None) is not None:
            zero_offset = self._import_offsets
        else:
            zero_offset = self.mgr.zero_offset_m or 0.0
            self._import_note += "; current zero-point offset applied"
        start_point = int(range_cfg.start_point)
        step_length = int(range_cfg.step_length)
        distances_m = ((start_point + np.arange(mean_iq.shape[1]) * step_length) * base_step
                       - zero_offset)
        meta = dict(meta)
        meta.update({
            "zero_offset_m": zero_offset,
            "leakage_phase_compensated": int(bool(corrected and lb_frames is not None)),
            "distance_filtered": int(bool(use_filter)),
        })
        try:
            temps = np.asarray(record.stacked_results.temperature, dtype=float)
            meta["sensor_temperature_c"] = float(np.mean(temps))
            meta["saturated_frames"] = int(np.sum(np.asarray(record.stacked_results.data_saturated)))
        except Exception:
            pass

        return self._row_from_arrays(
            meta, np.abs(mean_iq), np.angle(mean_iq), distances_m,
            os.path.basename(h5_path), corrected,
            raw_iq=raw_mean_iq, corrected_iq=mean_iq if corrected else None,
            noise_level=noise,
        )


# Keep a descriptive alias available for code that prefers the new name.
DataLoggingTab = RecordTab
