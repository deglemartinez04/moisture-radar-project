# Moisture model pipeline

Raw `.h5` recordings → windowed training CSV → Random Forest → live prediction in the App.

```
pip install -r requirements.txt

python ml/build_dataset.py App/Data_Log/ -o ml/dataset.csv
python ml/train_model.py ml/dataset.csv          # saves ml/models/moisture_model.joblib
python App/main.py                               # Moisture Level tab now uses the model
```

## Files

| File | What it does |
|---|---|
| `features.py` | **The one feature function** (`window_features`). Training and the live tab both call it, so the model sees identical numbers in both places. Finds G1/G2, skipping the TX→RX leakage bins (< 0.075 m), and flags windows with no wall reflection. |
| `h5io.py` | Reads A121 `.h5` recordings with plain `h5py` (no acconeer-exptool needed). |
| `build_dataset.py` | Cuts each recording into 10-frame windows (same as the live tab) → one CSV row per window. Labels come from the filename `<Material>_<Moisture>_<Trial>.h5`. |
| `train_model.py` | Random Forest, cross-validated **by recording**, with bootstrap 95% interval, confusion matrix and feature importances. Refuses to train on too little data. |
| `predict.py` | Loads the saved model for the App; warns if live sensor settings differ from the training recordings. |
| `labels.py` | **The moisture class cutoffs** (Dry < 12% MC, Damp 12-20%, Saturated > 20%). Edit here to change them. |
| `make_synthetic.py` | SIMULATED recordings (simple wood-moisture physics, geometry of Wood_High_01) to rehearse the workflow. Not evidence that the sensor works. |

## Collecting training data (checklist)

The model can only be as good as these recordings.

1. **Tick "Save raw radar data to an .h5 file"** in Data Logging. CSV-only recordings cannot be used for training.
2. **Same sensor settings for every recording.** Use "Reset to project defaults" and don't change them between sessions.
3. **Leave room behind G2.** With the default 40 points the range ends at ~0.255 m and G2 sits at ~0.23 m; wet wood delays G2 by up to ~1 cm. Consider Number of points = 50 (range to ~0.28 m), and keep it fixed for all recordings.
4. **Check the Amplitude Plot before recording.** You should see clear lobes beyond ~0.08 m (G1 at the surface, G2 behind it). If you only see the spike at 0.06 m and then a flat line, the sensor isn't seeing the board. `build_dataset.py` warns "no wall reflection" for these.
5. **Background correction off**, or always on. Never mix the two.
6. **Hold still** for the whole recording (30–60 s). Motion shows up as a large `peak_amplitude_std`.
7. **Recordings per moisture level: at least 6, ideally 10.** Each one should be a fresh placement (lift the sensor and set it down again, slightly different spot). On synthetic data, 3 per class gave ~80% accuracy and 6–10 per class gave ~95%. With few recordings the forest latches onto placement noise.
8. **Write down how each board was wetted** (for example: left dry, sprayed, soaked, and for how long) next to each recording's file name. The label comes from that procedure, so use the same procedure every time.
9. **Label from the procedure, not by eye:** `Dry`, `Damp`, `Saturated` (the Data Logging dropdown). Older `Low/Medium/High` files need `--allow-any-label`. The 12% / 20% moisture cutoffs in `labels.py` only apply if a moisture number is actually measured for each board.
10. **Boards wetted to in-between levels are the hard cases.** Include some, but expect the model to be least reliable there.

## Reading the training report

- **Window accuracy** is what the live gauge will do; **recording accuracy** is a majority vote over a whole recording.
- Compare against **chance level** (33% for 3 classes). The bootstrap interval will be wide until there are many recordings, and that width is real uncertainty.
- If a feature with no moisture physics behind it (`noise_floor`, `frame_to_frame_variability`, `signal_quality_score`) dominates the importances, suspect a difference in how the classes were recorded rather than in moisture. `gate_separation_m` is physical (wet wood delays G2) but only worth ~1 range bin, so a large class difference in it points to different setups.

## Current data status (2026-10-13)

- `Wood_High_01.h5`: good. G1 ≈ 0.105 m, G2 ≈ 0.230 m, G2/G1 ratio 0.635 ± 0.010.
- `Wood_Dry_4.h5` (App-development branch): **no wall reflection**, only leakage then noise, with the same settings as Wood_High_01. Re-record.
- Not enough labeled recordings to train yet; `train_model.py` will say what's missing.
