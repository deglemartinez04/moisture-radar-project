SIMULATED DATA -- not real radar measurements
=============================================
Made 2026-10-13 with ml/make_synthetic.py (branch ml-pipeline), using the team
cutoffs: Dry < 12% MC, Damp 12-20%, Saturated > 20%.

recordings/                 30 simulated .h5 recordings (10 per level) + mc_readings.csv
dataset.csv                 training rows (one per 10-frame window), open in Excel
dataset.meta.json           sensor settings and window size
training_report.txt         accuracy, confusion matrix, misclassified recordings
model/                      trained model -- do NOT use in the app, it learned from fake data

Replaces the older folder "simulation" (old cutoffs; now half-updated, safe to delete).
