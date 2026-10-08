"""
record_tab.py

Data Logging view for the radar moisture-tomography project.

Each recording produces one .h5 file and one row in a single master CSV file
named data_log.csv in the selected save folder. The CSV row stores the same
trial-level features used by the project for ML/training data.

The CSV can also be maintained manually from this tab:
- Add H5 recording: copies an existing .h5 into the selected folder and adds
  its calculated feature row to the master CSV.
- Delete selected row: removes the row and deletes the linked .h5 file.
"""

from __future__ import annotations

import csv
import os
import shutil
import time
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QMessageBox,
)

from sensor_manager import SensorManager

try:
    from acconeer.exptool import a121
except ImportError:
    a121 = None


CSV_COLUMNS = [
    "material",
    "moisture_level",
    "trial_number",
    "h5_file",
    "peak_distance_m",
    "peak_amplitude_mean",
    "peak_amplitude_std",
    "peak_phase_mean_rad",
    "peak_phase_std_rad",
    "attenuation_slope",
    "total_energy",
    "noise_floor",
    "frame_to_frame_variance",
    "num_frames",
]


class AcquisitionThread(QThread):
    """Runs client.get_next() in the background so the UI remains responsive."""

    frame_ready = Signal(object, object, object)  # (amplitude, phase, raw frame)
    error = Signal(str)

    def __init__(self, client):
        super().__init__()
        self.client = client
        self._running = True

    def run(self):
        try:
            while self._running:
                result = self.client.get_next()
                frame = np.asarray(result.frame)
                avg_sweep = np.mean(frame, axis=0)
                amplitude = np.abs(avg_sweep)
                phase = np.angle(avg_sweep)
                self.frame_ready.emit(amplitude, phase, frame)
        except Exception as e:
            if self._running:
                self.error.emit(str(e))

    def stop(self):
        self._running = False


class RecordTab(QWidget):
    """Data Logging tab. The class name is retained for main.py compatibility."""

    amplitude_ready = Signal(object, object)

    MATERIALS = ["Wood", "Drywall", "Wood + drywall", "Concrete"]
    MOISTURE_LEVELS = ["Dry", "Damp", "Saturated"]
    STOP_OPTIONS = ["Seconds", "Frames"]
    CSV_FILE_NAME = "data_log.csv"

    def __init__(self, sensor_manager: SensorManager):
        super().__init__()
        self.mgr = sensor_manager
        self.acquisition_thread: AcquisitionThread | None = None
        self.recorder = None
        self.record_file_path: str | None = None
        self.frame_amplitudes = []
        self.frame_phases = []
        self.frame_count = 0
        self.record_start_time = 0.0
        self.save_folder = os.path.expanduser("~")
        self._stopping = False
        self._build_ui()
        self._refresh_csv_rows()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)

        info_box = QGroupBox("Recording")
        form = QFormLayout(info_box)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(12)

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
        self.stop_mode_combo.setFixedWidth(130)
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setRange(0.1, 1_000_000.0)
        self.amount_spin.setDecimals(1)
        self.amount_spin.setValue(10.0)
        self.amount_spin.setSingleStep(1.0)
        stop_row.addWidget(self.stop_mode_combo)
        stop_row.addWidget(self.amount_spin, 1)
        form.addRow("Stop After", stop_row)

        outer.addWidget(info_box)

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
        save_layout.addWidget(self.csv_label)
        outer.addWidget(save_box)

        controls_box = QGroupBox("Controls")
        controls = QHBoxLayout(controls_box)
        self.start_btn = QPushButton("Start recording")
        self.start_btn.setObjectName("primaryButton")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch(1)
        outer.addWidget(controls_box)

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
        add_row.addWidget(self.add_h5_label, 1)
        self.manage_add_widget = QWidget()
        self.manage_add_widget.setLayout(add_row)
        self.manage_stack_layout.addWidget(self.manage_add_widget)

        delete_row = QHBoxLayout()
        delete_row.addWidget(QLabel("CSV row"))
        self.csv_row_combo = QComboBox()
        delete_row.addWidget(self.csv_row_combo, 1)
        self.delete_row_btn = QPushButton("Delete selected row")
        delete_row.addWidget(self.delete_row_btn)
        self.manage_delete_widget = QWidget()
        self.manage_delete_widget.setLayout(delete_row)
        self.manage_stack_layout.addWidget(self.manage_delete_widget)

        manage_layout.addLayout(self.manage_stack_layout)

        self.dataset_status_label = QLabel("CSV dataset: 0 recordings")
        self.dataset_status_label.setObjectName("summaryLabel")
        manage_layout.addWidget(self.dataset_status_label)
        outer.addWidget(manage_box)

        self.status_label = QLabel("Idle.")
        self.status_label.setObjectName("summaryLabel")
        outer.addWidget(self.status_label)

        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.browse_btn.clicked.connect(self._on_browse)
        self.add_h5_btn.clicked.connect(self._on_add_h5)
        self.delete_row_btn.clicked.connect(self._on_delete_row)
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
        if not os.path.exists(self.csv_file_path) or os.path.getsize(self.csv_file_path) == 0:
            with open(self.csv_file_path, "w", newline="", encoding="utf-8") as f:
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
                writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
        os.replace(temp_path, self.csv_file_path)

    def _append_csv_row(self, row):
        rows = self._read_csv_rows()
        rows.append(row)
        self._write_csv_rows(rows)

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
                f"{row.get('moisture_level', '')} | Trail {row.get('trial_number', '')} | "
                f"{row.get('h5_file', '(no H5 linked)')}"
            )
            self.csv_row_combo.addItem(label, index)

        self.dataset_status_label.setText(
            f"CSV dataset: {len(rows)} recording{'s' if len(rows) != 1 else ''}"
        )

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
                row = self._calculate_features_from_h5(
                    destination,
                    self.material_combo.currentText(),
                    self.moisture_combo.currentText(),
                    trail,
                )
                self._append_csv_row(row)
            except Exception:
                if os.path.exists(destination):
                    os.remove(destination)
                raise

            self.add_h5_label.setText(os.path.basename(source))
            self.trail_edit.setText(str(trail + 1))
            self._refresh_csv_rows()
            self.status_label.setText(
                f"Added {os.path.basename(destination)} to the master CSV."
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

        try:
            trail_number = self._trail_number()
            amount = self.amount_spin.value()
            if self.stop_mode_combo.currentText() == "Frames":
                amount = max(1, int(round(amount)))

            os.makedirs(self.save_folder, exist_ok=True)
            self._ensure_csv_exists()
            self.record_file_path = self._make_h5_path(trail_number)

            if os.path.exists(self.record_file_path):
                self.status_label.setText(
                    "An H5 file with this trail number already exists. Choose the next trail number."
                )
                return

            rows = self._read_csv_rows()
            if any(
                row.get("h5_file") == os.path.basename(self.record_file_path)
                for row in rows
            ):
                self.status_label.setText(
                    "This trail number is already present in the master CSV. Choose the next trail number."
                )
                return

            sensor_config = self.mgr.params.to_sensor_config()
            self.mgr.client.setup_session(sensor_config)
            self.recorder = a121.H5Recorder(self.record_file_path, self.mgr.client)
            self.mgr.client.start_session()

            self.distances_m = self.mgr.params.distances_m()
            self.frame_count = 0
            self.frame_amplitudes = []
            self.frame_phases = []
            self.record_start_time = time.monotonic()
            self._stopping = False

            self.acquisition_thread = AcquisitionThread(self.mgr.client)
            self.acquisition_thread.frame_ready.connect(self._on_frame)
            self.acquisition_thread.error.connect(self._on_error)
            self.acquisition_thread.start()

            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            mode = self.stop_mode_combo.currentText().lower()
            self.status_label.setText(f"Recording… Stop after {amount:g} {mode}.")
        except Exception as e:
            self._cleanup_after_failed_start()
            self.status_label.setText(f"Failed to start: {e}")

    def _on_frame(self, amplitude, phase, raw_frame):
        if self._stopping:
            return

        self.frame_count += 1
        self.amplitude_ready.emit(self.distances_m, amplitude)

        avg_sweep = np.mean(np.asarray(raw_frame), axis=0)
        self.frame_amplitudes.append(np.abs(avg_sweep))
        self.frame_phases.append(np.angle(avg_sweep))

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

        was_recording = self.recorder is not None
        self._close_recorder()

        if was_recording and self.record_file_path and self.frame_count > 0:
            try:
                row = self._calculate_summary_features(
                    self.material_combo.currentText(),
                    self.moisture_combo.currentText(),
                    self._trail_number(),
                    os.path.basename(self.record_file_path),
                )
                self._append_csv_row(row)

                current = self._trail_number()
                self.trail_edit.setText(str(current + 1))
                self._refresh_csv_rows()

                action = "Automatically saved" if auto else "Saved"
                self.status_label.setText(
                    f"{action}: {os.path.basename(self.record_file_path)} → master CSV "
                    f"({self.frame_count} frames)"
                )
            except Exception as e:
                self.status_label.setText(
                    f"H5 saved, but the master CSV could not be updated: {e}"
                )
        elif was_recording and self.record_file_path:
            self.status_label.setText("Recording stopped without any frames; H5 was not added to CSV.")
            try:
                if os.path.exists(self.record_file_path):
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
    def _calculate_summary_features(
        self, material, moisture, trail_number, h5_filename
    ):
        amplitude_window = np.asarray(self.frame_amplitudes, dtype=float)
        phase_window = np.asarray(self.frame_phases, dtype=float)
        return self._features_from_arrays(
            amplitude_window,
            phase_window,
            np.asarray(self.distances_m, dtype=float),
            material,
            moisture,
            trail_number,
            h5_filename,
        )

    @staticmethod
    def _features_from_arrays(
        amplitude_window,
        phase_window,
        distances_m,
        material,
        moisture,
        trail_number,
        h5_filename,
    ):
        if amplitude_window.ndim != 2 or len(amplitude_window) == 0:
            raise ValueError("No measurement frames were collected.")

        mean_amplitude_per_point = np.mean(amplitude_window, axis=0)
        peak_idx = int(np.argmax(mean_amplitude_per_point))
        peak_distance_m = float(distances_m[peak_idx])
        peak_amplitude_mean = float(mean_amplitude_per_point[peak_idx])
        peak_amplitude_std = float(np.std(amplitude_window[:, peak_idx]))

        peak_phase_values = phase_window[:, peak_idx]
        peak_phase_mean = float(np.angle(np.mean(np.exp(1j * peak_phase_values))))
        peak_phase_std = float(
            np.std(np.angle(np.exp(1j * (peak_phase_values - peak_phase_mean))))
        )

        safe_amplitude = np.clip(mean_amplitude_per_point, 1e-6, None)
        if len(distances_m) >= 2:
            attenuation_slope = float(
                np.polyfit(distances_m, np.log(safe_amplitude), 1)[0]
            )
        else:
            attenuation_slope = 0.0

        total_energy = float(np.sum(mean_amplitude_per_point))
        noise_floor = float(np.percentile(mean_amplitude_per_point, 10))
        frame_to_frame_variance = float(np.mean(np.var(amplitude_window, axis=0)))

        return {
            "material": material,
            "moisture_level": moisture,
            "trial_number": trail_number,
            "h5_file": h5_filename,
            "peak_distance_m": peak_distance_m,
            "peak_amplitude_mean": peak_amplitude_mean,
            "peak_amplitude_std": peak_amplitude_std,
            "peak_phase_mean_rad": peak_phase_mean,
            "peak_phase_std_rad": peak_phase_std,
            "attenuation_slope": attenuation_slope,
            "total_energy": total_energy,
            "noise_floor": noise_floor,
            "frame_to_frame_variance": frame_to_frame_variance,
            "num_frames": int(amplitude_window.shape[0]),
        }

    def _calculate_features_from_h5(
        self, h5_path, material, moisture, trail_number
    ):
        """Load an A121 H5 recording and calculate the same CSV features."""
        if a121 is None:
            raise RuntimeError("The Acconeer A121 package is not available.")

        # Acconeer recommends load_record() for loading H5 recordings. The
        # loaded Record exposes all recorded frames through ``frames`` (plural).
        # ``frame`` is the property of an individual Result, not the Record.
        record = a121.load_record(h5_path)
        frame = np.asarray(record.frames)

        if frame.ndim == 2:
            # Be tolerant of a single-frame record returned without the leading
            # frame dimension.
            frame = frame[np.newaxis, ...]
        if frame.ndim != 3:
            raise ValueError(
                f"Unexpected H5 frame shape: {frame.shape}. "
                "Expected (frames, sweeps, distance_points)."
            )

        amplitude_window = np.abs(np.mean(frame, axis=1))
        phase_window = np.angle(np.mean(frame, axis=1))

        config = record.session_config.sensor_config
        start_point = int(config.start_point)
        step_length = int(config.step_length)
        num_points = amplitude_window.shape[1]
        distances_m = (start_point + np.arange(num_points) * step_length) * 0.0025

        return self._features_from_arrays(
            amplitude_window,
            phase_window,
            distances_m,
            material,
            moisture,
            trail_number,
            os.path.basename(h5_path),
        )


# Keep a descriptive alias available for code that prefers the new name.
DataLoggingTab = RecordTab
