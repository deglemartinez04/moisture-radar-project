@echo off
setlocal
title ML Check - retrain on the saved dataset
set "PY=%LOCALAPPDATA%\MoistureRadarML\venv\Scripts\python.exe"
set "WORK=%LOCALAPPDATA%\MoistureRadarML\work_saved"
set "HERE=%~dp0"
if not exist "%PY%" (
  echo Not set up yet. Double-click 1_SETUP_once.bat first.
  if not defined NOPAUSE pause
  exit /b 1
)
if not exist "%HERE%results" mkdir "%HERE%results"

echo.
echo RETRAIN ON THE SAVED DATASET  (1,800 rows from 30 SIMULATED recordings)
echo.
"%PY%" "%HERE%ml\train_model.py" "%HERE%saved_dataset\dataset.csv" --out-dir "%WORK%\model" > "%HERE%results\retrain_report.txt"
type "%HERE%results\retrain_report.txt"

echo.
echo ------------------------------------------------------------
echo The full report is saved in: %HERE%results\retrain_report.txt
echo Expected: Window accuracy about 79.9%%, Recording accuracy 80.0%% (24/30).
if not defined NOPAUSE pause
