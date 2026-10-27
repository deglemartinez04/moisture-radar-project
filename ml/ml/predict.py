"""Load the trained model and turn one live feature window into a prediction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.features import feature_vector

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"

# Position of each label on the 0-100 wetness gauge.
WETNESS_SCALE = {"Dry": 0.0, "Damp": 50.0, "Saturated": 100.0}


class MoistureModel:
    def __init__(self, model, meta):
        self.model = model
        self.meta = meta
        self.features = meta["features"]

    @classmethod
    def load(cls, model_dir=DEFAULT_MODEL_DIR):
        """Return a MoistureModel, or None if no trained model has been saved yet."""
        model_dir = Path(model_dir)
        model_path, meta_path = model_dir / "moisture_model.joblib", model_dir / "model_meta.json"
        if not model_path.exists() or not meta_path.exists():
            return None
        import joblib

        return cls(joblib.load(model_path), json.loads(meta_path.read_text()))

    def describe(self):
        m = self.meta
        acc = m.get("cv_window_accuracy")
        acc_text = f", CV accuracy {acc:.0%}" if acc is not None else ""
        return f"Model trained {m.get('trained_at', '?')} on {m.get('n_recordings', '?')} recordings{acc_text}"

    def predict(self, features):
        """Return (label, probabilities dict, confidence 0-1, wetness 0-100 or None).

        label is None when no wall reflection was found in the window.
        Confidence = winning-class probability x signal quality score.
        """
        if not features.get("target_detected"):
            return None, {}, 0.0, None
        proba = self.model.predict_proba(feature_vector(features, self.features))[0]
        probs = {str(c): float(p) for c, p in zip(self.model.classes_, proba)}
        label = max(probs, key=probs.get)
        quality = float(features.get("signal_quality_score", 0.0))
        confidence = probs[label] * quality
        wetness = None
        if set(probs) <= set(WETNESS_SCALE):
            wetness = float(sum(p * WETNESS_SCALE[c] for c, p in probs.items()))
        return label, probs, confidence, wetness

    def config_mismatches(self, params):
        """List sensor settings that differ from the training recordings."""
        trained = self.meta.get("sensor_config") or {}
        live = {
            "start_point": params.start_point,
            "num_points": params.num_points,
            "step_length": params.step_length,
            "profile": f"PROFILE_{params.profile}",
            "hwaas": params.hwaas,
            "receiver_gain": params.receiver_gain,
            "prf": params.prf,
        }
        return [f"{k}: trained {trained[k]}, now {v}" for k, v in live.items()
                if k in trained and trained[k] is not None and trained[k] != v]


def iq_from_amplitude_phase(amplitude_window, phase_window):
    """The App's live thread buffers amplitude and phase separately; rebuild complex IQ."""
    return np.asarray(amplitude_window) * np.exp(1j * np.asarray(phase_window))
