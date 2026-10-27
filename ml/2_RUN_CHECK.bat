@echo off
setlocal
title ML Check - run from scratch
set "PY=%LOCALAPPDATA%\MoistureRadarML\venv\Scripts\python.exe"
set "WORK=%LOCALAPPDATA%\MoistureRadarML\work_check"
set "HERE=%~dp0"
if not exist "%PY%" (
  echo Not set up yet. Double-click 1_SETUP_once.bat first.
  if not defined NOPAUSE pause
  exit /b 1
)
if not exist "%HERE%results" mkdir "%HERE%results"
if exist "%WORK%" rmdir /s /q "%WORK%"

echo.
echo RUN FROM SCRATCH  (all data is SIMULATED)
echo.
echo STEP 1 of 3: simulating 30 recordings (10 each of Dry, Damp, Saturated)...
"%PY%" "%HERE%ml\make_synthetic.py" -o "%WORK%\recordings" --seed 0
echo.
echo STEP 2 of 3: turning the recordings into a training table...
"%PY%" "%HERE%ml\build_dataset.py" "%WORK%\recordings" -o "%WORK%\dataset.csv"
echo.
echo STEP 3 of 3: training the model and testing it on recordings it has not seen...
"%PY%" "%HERE%ml\train_model.py" "%WORK%\dataset.csv" --out-dir "%WORK%\model" > "%HERE%results\run_check_report.txt"
type "%HERE%results\run_check_report.txt"

echo.
echo ------------------------------------------------------------
echo The full report is saved in: %HERE%results\run_check_report.txt
echo Expected: Window accuracy about 79.9%%, Recording accuracy 80.0%% (24/30).
if not defined NOPAUSE pause
