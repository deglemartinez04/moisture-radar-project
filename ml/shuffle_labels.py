"""Randomly reassign each recording's moisture label (used by 5_SHUFFLE_TEST.bat).

Usage: python shuffle_labels.py dataset.csv shuffled.csv
"""
import sys

import numpy as np
import pandas as pd

data = pd.read_csv(sys.argv[1])
rng = np.random.default_rng(5)
per_recording = data.groupby("recording_id")["moisture_level"].first()
shuffled = pd.Series(rng.permutation(per_recording.values), index=per_recording.index)
data["moisture_level"] = data["recording_id"].map(shuffled)
data.to_csv(sys.argv[2], index=False)
print(f"Shuffled the labels of {len(per_recording)} recordings.")
