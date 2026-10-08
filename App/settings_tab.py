"""
settings_tab.py

Exposes every configurable parameter on the A121 SensorConfig (per Acconeer's
API reference), organized into logical groups. Editing any field here updates
the shared SensorManager.params immediately, which both the Record and Live
tabs read from.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sensor_manager import IDLE_STATE_OPTIONS, PRF_OPTIONS, SensorManager


def _section(title: str) -> QGroupBox:
    box = QGroupBox(title)
    box.setObjectName("section")
    return box


class _CollapsibleSection(QWidget):
    """Small reusable disclosure widget used by the settings explanations."""

    def __init__(self, title: str, content: QWidget | None = None, expanded: bool = False):
        super().__init__()

        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setObjectName("collapsibleHeader")

        self.content = content if content is not None else QWidget()
        self.content.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content)

        self.toggle_button.toggled.connect(self._set_expanded)

    def _set_expanded(self, expanded: bool):
        self.content.setVisible(expanded)
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)


class SettingsTab(QWidget):
    def __init__(self, sensor_manager: SensorManager):
        super().__init__()
        self.mgr = sensor_manager
        self.settings = QSettings("RadarMoistureTomography", "A121Console")
        self._load_saved_settings()
        self._build_ui()
        self._load_from_params()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)

        # --- Connection row (always visible, not in the scroll area) ---
        conn_box = _section("Connection")
        conn_layout = QHBoxLayout(conn_box)
        conn_layout.setSpacing(10)

        conn_layout.addWidget(QLabel("Sensor IP address"))
        self.ip_edit = QLineEdit()
        self.ip_edit.setPlaceholderText("e.g. 172.20.10.4 or 127.0.0.1")
        self.ip_edit.setFixedWidth(180)
        conn_layout.addWidget(self.ip_edit)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setObjectName("primaryButton")
        conn_layout.addWidget(self.connect_btn)

        self.status_label = QLabel("Not connected")
        self.status_label.setObjectName("statusLabel")
        conn_layout.addWidget(self.status_label)
        conn_layout.addStretch(1)

        outer.addWidget(conn_box)

        # --- Scrollable area for all sensor parameters ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(16)
        content_layout.setContentsMargins(0, 0, 4, 0)

        content_layout.addWidget(self._build_range_section())
        content_layout.addWidget(self._build_timing_section())
        content_layout.addWidget(self._build_rf_section())
        content_layout.addWidget(self._build_advanced_section())
        content_layout.addWidget(self._build_explanations_section())
        content_layout.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # --- Bottom row: apply/save + reset ---
        bottom = QHBoxLayout()
        bottom.setSpacing(10)

        self.apply_btn = QPushButton("Apply settings")
        self.apply_btn.setObjectName("primaryButton")
        bottom.addWidget(self.apply_btn)

        self.reset_btn = QPushButton("Reset to project defaults")
        bottom.addWidget(self.reset_btn)

        self.save_status_label = QLabel("")
        self.save_status_label.setObjectName("summaryLabel")
        bottom.addWidget(self.save_status_label)
        bottom.addStretch(1)
        outer.addLayout(bottom)

        # Wire up signals
        self.connect_btn.clicked.connect(self._on_connect_clicked)
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        self.mgr.connection_changed.connect(self._on_connection_changed)

    def _build_range_section(self) -> QGroupBox:
        box = _section("Range — start point, points, step length")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignLeft)

        self.start_point_spin = QSpinBox()
        self.start_point_spin.setRange(0, 4000)
        self.start_point_spin.valueChanged.connect(self._on_any_change)
        form.addRow("Start point (≈ 2.5 mm/unit)", self.start_point_spin)

        self.num_points_spin = QSpinBox()
        self.num_points_spin.setRange(1, 4000)
        self.num_points_spin.valueChanged.connect(self._on_any_change)
        form.addRow("Number of points", self.num_points_spin)

        self.step_length_spin = QSpinBox()
        self.step_length_spin.setRange(1, 100)
        self.step_length_spin.valueChanged.connect(self._on_any_change)
        form.addRow("Step length (≈ 2.5 mm/unit)", self.step_length_spin)

        self.range_summary_label = QLabel()
        self.range_summary_label.setObjectName("summaryLabel")
        form.addRow("Approx. range", self.range_summary_label)

        return box

    def _build_timing_section(self) -> QGroupBox:
        box = _section("Timing — sweeps, frame/sweep rate")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignLeft)

        self.sweeps_per_frame_spin = QSpinBox()
        self.sweeps_per_frame_spin.setRange(1, 2047)
        self.sweeps_per_frame_spin.valueChanged.connect(self._on_any_change)
        form.addRow("Sweeps per frame", self.sweeps_per_frame_spin)

        # Frame rate: explicitly choose between A121 automatic operation
        # (None / unlimited) and a user-selected manual rate.
        self.frame_rate_mode_combo = QComboBox()
        self.frame_rate_mode_combo.addItems(["Automatic", "Manual"])
        self.frame_rate_mode_combo.currentIndexChanged.connect(self._on_frame_rate_mode_changed)

        self.frame_rate_spin = QDoubleSpinBox()
        self.frame_rate_spin.setRange(0.1, 1000)
        self.frame_rate_spin.setDecimals(1)
        self.frame_rate_spin.setSuffix(" Hz")
        self.frame_rate_spin.valueChanged.connect(self._on_any_change)

        frame_rate_widget = QWidget()
        frame_rate_layout = QHBoxLayout(frame_rate_widget)
        frame_rate_layout.setContentsMargins(0, 0, 0, 0)
        frame_rate_layout.setSpacing(8)
        frame_rate_layout.addWidget(self.frame_rate_mode_combo)
        frame_rate_layout.addWidget(self.frame_rate_spin, 1)
        form.addRow("Frame rate", frame_rate_widget)

        # Sweep rate: explicitly choose between A121 automatic operation
        # (None / maximum available) and a user-selected manual rate.
        self.sweep_rate_mode_combo = QComboBox()
        self.sweep_rate_mode_combo.addItems(["Automatic", "Manual"])
        self.sweep_rate_mode_combo.currentIndexChanged.connect(self._on_sweep_rate_mode_changed)

        self.sweep_rate_spin = QDoubleSpinBox()
        self.sweep_rate_spin.setRange(0.1, 100000)
        self.sweep_rate_spin.setDecimals(1)
        self.sweep_rate_spin.setSuffix(" Hz")
        self.sweep_rate_spin.valueChanged.connect(self._on_any_change)

        sweep_rate_widget = QWidget()
        sweep_rate_layout = QHBoxLayout(sweep_rate_widget)
        sweep_rate_layout.setContentsMargins(0, 0, 0, 0)
        sweep_rate_layout.setSpacing(8)
        sweep_rate_layout.addWidget(self.sweep_rate_mode_combo)
        sweep_rate_layout.addWidget(self.sweep_rate_spin, 1)
        form.addRow("Sweep rate", sweep_rate_widget)

        self.continuous_sweep_check = QCheckBox("Continuous sweep mode (CSM)")
        self.continuous_sweep_check.stateChanged.connect(self._on_any_change)
        form.addRow("", self.continuous_sweep_check)

        self.double_buffering_check = QCheckBox("Double buffering")
        self.double_buffering_check.stateChanged.connect(self._on_any_change)
        form.addRow("", self.double_buffering_check)

        self.inter_frame_idle_combo = QComboBox()
        self.inter_frame_idle_combo.addItems(IDLE_STATE_OPTIONS)
        self.inter_frame_idle_combo.currentTextChanged.connect(self._on_any_change)
        form.addRow("Inter-frame idle state", self.inter_frame_idle_combo)

        self.inter_sweep_idle_combo = QComboBox()
        self.inter_sweep_idle_combo.addItems(IDLE_STATE_OPTIONS)
        self.inter_sweep_idle_combo.currentTextChanged.connect(self._on_any_change)
        form.addRow("Inter-sweep idle state", self.inter_sweep_idle_combo)

        return box

    def _build_rf_section(self) -> QGroupBox:
        box = _section("RF — profile, gain, HWAAS, PRF")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignLeft)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems([
            f"PROFILE_{i} "
            + ("(shortest)" if i == 1 else "(longest)" if i == 5 else "")
            for i in range(1, 6)
        ])
        self.profile_combo.currentIndexChanged.connect(self._on_any_change)
        form.addRow("Profile", self.profile_combo)

        self.hwaas_spin = QSpinBox()
        self.hwaas_spin.setRange(1, 511)
        self.hwaas_spin.valueChanged.connect(self._on_any_change)
        form.addRow("HWAAS", self.hwaas_spin)

        self.receiver_gain_spin = QSpinBox()
        self.receiver_gain_spin.setRange(0, 23)
        self.receiver_gain_spin.valueChanged.connect(self._on_any_change)
        form.addRow("Receiver gain", self.receiver_gain_spin)

        self.prf_combo = QComboBox()
        self.prf_combo.addItems(PRF_OPTIONS)
        self.prf_combo.currentTextChanged.connect(self._on_any_change)
        form.addRow("PRF", self.prf_combo)

        return box

    def _build_advanced_section(self) -> QGroupBox:
        box = _section("Advanced")
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignLeft)

        self.enable_tx_check = QCheckBox("Enable transmitter")
        self.enable_tx_check.stateChanged.connect(self._on_any_change)
        form.addRow("", self.enable_tx_check)

        self.enable_loopback_check = QCheckBox("Enable loopback")
        self.enable_loopback_check.stateChanged.connect(self._on_any_change)
        form.addRow("", self.enable_loopback_check)

        self.phase_enhancement_check = QCheckBox("Phase enhancement")
        self.phase_enhancement_check.stateChanged.connect(self._on_any_change)
        form.addRow("", self.phase_enhancement_check)

        self.iq_imbalance_check = QCheckBox("IQ imbalance compensation")
        self.iq_imbalance_check.stateChanged.connect(self._on_any_change)
        form.addRow("", self.iq_imbalance_check)

        return box

    def _build_explanations_section(self) -> QGroupBox:
        """Build a compact, nested, collapsible plain-English settings guide."""
        box = _section("Understanding Your Settings")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        sections = [
            (
                "Range",
                [
                    (
                        "Start point",
                        "Sets the distance where the radar begins taking measurements. "
                        "Increasing it moves the beginning of the measurement farther away from the sensor."
                    ),
                    (
                        "Number of points",
                        "Controls how many individual distance measurements are taken across the selected range. "
                        "More points can provide more detail, but also means more data to process."
                    ),
                    (
                        "Step length",
                        "Controls the spacing between measurement points. "
                        "Increasing it covers a larger distance with fewer, more widely spaced measurements; "
                        "decreasing it gives finer distance spacing."
                    ),
                    (
                        "Approx. range",
                        "Shows the approximate distance covered by the current range settings. "
                        "This value is calculated automatically and cannot be changed directly."
                    ),
                ],
            ),
            (
                "Timing",
                [
                    (
                        "Sweeps per frame",
                        "Determines how many radar sweeps are combined into one frame of data. "
                        "Increasing this can provide more stable measurements, but may reduce how quickly new frames are produced."
                    ),
                    (
                        "Frame rate — Automatic / Manual",
                        "Controls how quickly complete frames are produced. "
                        "Automatic lets the sensor choose the rate; Manual lets you set a specific maximum frame rate."
                    ),
                    (
                        "Sweep rate — Automatic / Manual",
                        "Controls how quickly individual radar sweeps are performed. "
                        "Automatic uses the maximum available rate; Manual lets you choose a specific sweep rate."
                    ),
                    (
                        "Continuous sweep mode (CSM)",
                        "Keeps the radar operating continuously rather than treating each sweep as a separate operation. "
                        "This can be useful when continuous measurements are desired."
                    ),
                    (
                        "Double buffering",
                        "Allows one set of data to be processed while another is being collected. "
                        "This can help keep data flowing smoothly during continuous operation."
                    ),
                    (
                        "Inter-frame idle state",
                        "Determines what the sensor does during the short period between frames. "
                        "The available choices control how the sensor behaves while waiting for the next frame."
                    ),
                    (
                        "Inter-sweep idle state",
                        "Determines what the sensor does between individual sweeps. "
                        "This can affect how the sensor behaves and how quickly measurements are taken."
                    ),
                ],
            ),
            (
                "RF / Signal",
                [
                    (
                        "Profile",
                        "Selects the radar's operating profile. "
                        "Different profiles balance measurement range, resolution, and speed differently; "
                        "the shorter profiles generally favor closer measurements while longer profiles favor greater range."
                    ),
                    (
                        "HWAAS",
                        "Controls how many internal hardware averages are used when taking a measurement. "
                        "Increasing it can make measurements more consistent and reduce noise, but can increase the time needed to collect data."
                    ),
                    (
                        "Receiver gain",
                        "Controls how strongly the radar receiver amplifies returning signals. "
                        "Increasing it can make weaker reflections easier to see, but excessive gain can also make strong signals or unwanted noise more prominent."
                    ),
                    (
                        "PRF",
                        "Controls the pulse repetition frequency used by the radar. "
                        "It affects how the sensor performs its measurements and is one of the settings that determines the practical measurement range and behavior."
                    ),
                ],
            ),
            (
                "Advanced",
                [
                    (
                        "Enable transmitter",
                        "Turns the radar transmitter on or off. "
                        "The transmitter needs to be enabled for normal radar measurements."
                    ),
                    (
                        "Enable loopback",
                        "Enables an internal signal path that routes the transmitted signal back toward the receiver for testing and diagnostics. "
                        "It is generally not needed for normal wall-moisture measurements."
                    ),
                    (
                        "Phase enhancement",
                        "Enables additional processing intended to improve the usefulness of phase information. "
                        "This can be helpful when phase changes are important to the measurement."
                    ),
                    (
                        "IQ imbalance compensation",
                        "Applies compensation for differences between the radar's in-phase (I) and quadrature (Q) signal components. "
                        "This can improve the quality and consistency of the resulting IQ data."
                    ),
                ],
            ),
        ]

        for section_title, explanations in sections:
            items_layout = QVBoxLayout()
            items_layout.setContentsMargins(12, 2, 4, 6)
            items_layout.setSpacing(2)

            for name, description in explanations:
                description_label = QLabel(description)
                description_label.setObjectName("explanationText")
                description_label.setWordWrap(True)
                description_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                description_label.setContentsMargins(28, 2, 12, 8)

                item = _CollapsibleSection(name, description_label, expanded=False)
                items_layout.addWidget(item)

            items_widget = QWidget()
            items_widget.setLayout(items_layout)
            main_section = _CollapsibleSection(
                section_title,
                items_widget,
                expanded=False,
            )
            layout.addWidget(main_section)

        return box

    # ------------------------------------------------------------------
    # Persistent settings
    # ------------------------------------------------------------------
    def _load_saved_settings(self):
        """Load the last settings saved with Apply settings, if available."""
        p = self.mgr.params

        if not self.settings.value("settings_saved", False, type=bool):
            return

        p.ip_address = self.settings.value("ip_address", p.ip_address, type=str)
        p.start_point = self.settings.value("start_point", p.start_point, type=int)
        p.num_points = self.settings.value("num_points", p.num_points, type=int)
        p.step_length = self.settings.value("step_length", p.step_length, type=int)
        p.sweeps_per_frame = self.settings.value("sweeps_per_frame", p.sweeps_per_frame, type=int)

        frame_mode = self.settings.value("frame_rate_mode", "Automatic", type=str)
        frame_value = self.settings.value("frame_rate", 30.0, type=float)
        p.frame_rate = None if frame_mode == "Automatic" else frame_value

        sweep_mode = self.settings.value("sweep_rate_mode", "Automatic", type=str)
        sweep_value = self.settings.value("sweep_rate", 1000.0, type=float)
        p.sweep_rate = None if sweep_mode == "Automatic" else sweep_value

        p.continuous_sweep_mode = self.settings.value(
            "continuous_sweep_mode", p.continuous_sweep_mode, type=bool
        )
        p.double_buffering = self.settings.value(
            "double_buffering", p.double_buffering, type=bool
        )
        p.inter_frame_idle_state = self.settings.value(
            "inter_frame_idle_state", p.inter_frame_idle_state, type=str
        )
        p.inter_sweep_idle_state = self.settings.value(
            "inter_sweep_idle_state", p.inter_sweep_idle_state, type=str
        )
        p.profile = self.settings.value("profile", p.profile, type=int)
        p.hwaas = self.settings.value("hwaas", p.hwaas, type=int)
        p.receiver_gain = self.settings.value(
            "receiver_gain", p.receiver_gain, type=int
        )
        p.prf = self.settings.value("prf", p.prf, type=str)
        p.enable_tx = self.settings.value("enable_tx", p.enable_tx, type=bool)
        p.enable_loopback = self.settings.value(
            "enable_loopback", p.enable_loopback, type=bool
        )
        p.phase_enhancement = self.settings.value(
            "phase_enhancement", p.phase_enhancement, type=bool
        )
        p.iq_imbalance_compensation = self.settings.value(
            "iq_imbalance_compensation",
            p.iq_imbalance_compensation,
            type=bool,
        )

    def _save_settings(self):
        """Persist the current SensorManager parameters for the next launch."""
        p = self.mgr.params

        values = {
            "settings_saved": True,
            "ip_address": p.ip_address,
            "start_point": p.start_point,
            "num_points": p.num_points,
            "step_length": p.step_length,
            "sweeps_per_frame": p.sweeps_per_frame,
            "frame_rate_mode": self.frame_rate_mode_combo.currentText(),
            "frame_rate": self.frame_rate_spin.value(),
            "sweep_rate_mode": self.sweep_rate_mode_combo.currentText(),
            "sweep_rate": self.sweep_rate_spin.value(),
            "continuous_sweep_mode": p.continuous_sweep_mode,
            "double_buffering": p.double_buffering,
            "inter_frame_idle_state": p.inter_frame_idle_state,
            "inter_sweep_idle_state": p.inter_sweep_idle_state,
            "profile": p.profile,
            "hwaas": p.hwaas,
            "receiver_gain": p.receiver_gain,
            "prf": p.prf,
            "enable_tx": p.enable_tx,
            "enable_loopback": p.enable_loopback,
            "phase_enhancement": p.phase_enhancement,
            "iq_imbalance_compensation": p.iq_imbalance_compensation,
        }

        for key, value in values.items():
            self.settings.setValue(key, value)
        self.settings.sync()

    def _on_apply_clicked(self):
        # Make absolutely sure the widgets are reflected in SensorManager before saving.
        self._on_any_change()
        self._save_settings()
        self.save_status_label.setText("Settings saved.")

    # ------------------------------------------------------------------
    # Sync between widgets <-> SensorManager.params
    # ------------------------------------------------------------------
    def _load_from_params(self):
        # Block signals while we set initial values, so the premature
        # valueChanged/stateChanged callbacks (fired as each widget's value is set)
        # don't overwrite mgr.params with the widgets' own construction-time defaults.
        widgets = [
            self.ip_edit, self.start_point_spin, self.num_points_spin, self.step_length_spin,
            self.sweeps_per_frame_spin, self.frame_rate_mode_combo, self.frame_rate_spin,
            self.sweep_rate_mode_combo, self.sweep_rate_spin,
            self.continuous_sweep_check, self.double_buffering_check,
            self.inter_frame_idle_combo, self.inter_sweep_idle_combo, self.profile_combo,
            self.hwaas_spin, self.receiver_gain_spin, self.prf_combo, self.enable_tx_check,
            self.enable_loopback_check, self.phase_enhancement_check, self.iq_imbalance_check,
        ]
        for w in widgets:
            w.blockSignals(True)

        p = self.mgr.params
        self.ip_edit.setText(p.ip_address)
        self.start_point_spin.setValue(p.start_point)
        self.num_points_spin.setValue(p.num_points)
        self.step_length_spin.setValue(p.step_length)
        self.sweeps_per_frame_spin.setValue(p.sweeps_per_frame)

        # None means automatic operation in SensorConfig. Keep the actual
        # manual value available when possible, but show the mode explicitly.
        if p.frame_rate is None or p.frame_rate == 0:
            self.frame_rate_mode_combo.setCurrentText("Automatic")
            self.frame_rate_spin.setValue(30.0)
        else:
            self.frame_rate_mode_combo.setCurrentText("Manual")
            self.frame_rate_spin.setValue(float(p.frame_rate))

        if p.sweep_rate is None or p.sweep_rate == 0:
            self.sweep_rate_mode_combo.setCurrentText("Automatic")
            self.sweep_rate_spin.setValue(1000.0)
        else:
            self.sweep_rate_mode_combo.setCurrentText("Manual")
            self.sweep_rate_spin.setValue(float(p.sweep_rate))

        self.continuous_sweep_check.setChecked(p.continuous_sweep_mode)
        self.double_buffering_check.setChecked(p.double_buffering)
        self.inter_frame_idle_combo.setCurrentText(p.inter_frame_idle_state)
        self.inter_sweep_idle_combo.setCurrentText(p.inter_sweep_idle_state)
        self.profile_combo.setCurrentIndex(p.profile - 1)
        self.hwaas_spin.setValue(p.hwaas)
        self.receiver_gain_spin.setValue(p.receiver_gain)
        self.prf_combo.setCurrentText(p.prf)
        self.enable_tx_check.setChecked(p.enable_tx)
        self.enable_loopback_check.setChecked(p.enable_loopback)
        self.phase_enhancement_check.setChecked(p.phase_enhancement)
        self.iq_imbalance_check.setChecked(p.iq_imbalance_compensation)

        for w in widgets:
            w.blockSignals(False)

        self._update_rate_controls()
        self._update_range_summary()

    def _on_frame_rate_mode_changed(self, *_):
        self._update_rate_controls()
        self._on_any_change()

    def _on_sweep_rate_mode_changed(self, *_):
        self._update_rate_controls()
        self._on_any_change()

    def _update_rate_controls(self):
        frame_manual = self.frame_rate_mode_combo.currentText() == "Manual"
        sweep_manual = self.sweep_rate_mode_combo.currentText() == "Manual"

        self.frame_rate_spin.setEnabled(frame_manual)
        self.sweep_rate_spin.setEnabled(sweep_manual)

        if not frame_manual:
            self.frame_rate_spin.setToolTip(
                "Automatic: the sensor determines the frame rate."
            )
        else:
            self.frame_rate_spin.setToolTip(
                "Manual: set the maximum frame rate in Hz."
            )

        if not sweep_manual:
            self.sweep_rate_spin.setToolTip(
                "Automatic: the sensor uses its maximum available sweep rate."
            )
        else:
            self.sweep_rate_spin.setToolTip(
                "Manual: set the sweep rate in Hz."
            )

    def _on_any_change(self, *_):
        p = self.mgr.params
        p.ip_address = self.ip_edit.text().strip()
        p.start_point = self.start_point_spin.value()
        p.num_points = self.num_points_spin.value()
        p.step_length = self.step_length_spin.value()
        p.sweeps_per_frame = self.sweeps_per_frame_spin.value()
        p.frame_rate = (
            None
            if self.frame_rate_mode_combo.currentText() == "Automatic"
            else self.frame_rate_spin.value()
        )
        p.sweep_rate = (
            None
            if self.sweep_rate_mode_combo.currentText() == "Automatic"
            else self.sweep_rate_spin.value()
        )
        p.continuous_sweep_mode = self.continuous_sweep_check.isChecked()
        p.double_buffering = self.double_buffering_check.isChecked()
        p.inter_frame_idle_state = self.inter_frame_idle_combo.currentText()
        p.inter_sweep_idle_state = self.inter_sweep_idle_combo.currentText()
        p.profile = self.profile_combo.currentIndex() + 1
        p.hwaas = self.hwaas_spin.value()
        p.receiver_gain = self.receiver_gain_spin.value()
        p.prf = self.prf_combo.currentText()
        p.enable_tx = self.enable_tx_check.isChecked()
        p.enable_loopback = self.enable_loopback_check.isChecked()
        p.phase_enhancement = self.phase_enhancement_check.isChecked()
        p.iq_imbalance_compensation = self.iq_imbalance_check.isChecked()

        self._update_range_summary()
        self.mgr.notify_params_changed()

    def _update_range_summary(self):
        p = self.mgr.params
        start_m = p.start_point * 0.0025
        end_m = (p.start_point + p.num_points * p.step_length) * 0.0025
        self.range_summary_label.setText(f"{start_m:.3f} m to {end_m:.3f} m")

    def _on_reset_clicked(self):
        from sensor_manager import SensorParams

        ip = self.mgr.params.ip_address  # preserve IP on reset
        self.mgr.params = SensorParams()
        self.mgr.params.ip_address = ip
        self._load_from_params()
        self.mgr.notify_params_changed()

        # Reset is also made persistent, so reopening the app keeps the
        # project defaults rather than restoring the older saved configuration.
        self._save_settings()
        self.save_status_label.setText("Project defaults restored and saved.")

    def _on_connect_clicked(self):
        if self.mgr.is_connected():
            self.mgr.disconnect_sensor()
        else:
            self.mgr.params.ip_address = self.ip_edit.text().strip()
            self.connect_btn.setEnabled(False)
            self.connect_btn.setText("Connecting...")
            self.mgr.connect_sensor()

    def _on_connection_changed(self, connected: bool, message: str):
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.status_label.setText(message)
        self.status_label.setProperty("connected", connected)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
