@echo off
setlocal
title ML Check - label shuffle test
set "PY=%LOCALAPPDATA%\MoistureRadarML\venv\Scripts\python.exe"
set "WORK=%LOCALAPPDATA%\MoistureRadarML\work_shuffle"
set "HERE=%~dp0"
set "OUT=%HERE%results\shuffle_test.txt"
if not exist "%PY%" (
  echo Not set up yet. Double-click 1_SETUP_once.bat first.
  if not defined NOPAUSE pause
  exit /b 1
)
if not exist "%HERE%results" mkdir "%HERE%results"
if exist "%WORK%" rmdir /s /q "%WORK%"

echo.
echo LABEL SHUFFLE TEST
echo Same measurements, but each recording's label (Dry/Damp/Saturated) is randomly
echo reassigned. If the pipeline were cheating, accuracy would stay high.
echo If it is honest, accuracy drops to chance (about 33%%).
echo.
echo Preparing data...
"%PY%" "%HERE%ml\make_synthetic.py" -o "%WORK%\recordings" --seed 0 >nul
"%PY%" "%HERE%ml\build_dataset.py" "%WORK%\recordings" -o "%WORK%\dataset.csv" >nul
"%PY%" "%HERE%shuffle_labels.py" "%WORK%\dataset.csv" "%WORK%\shuffled.csv"

echo REAL LABELS: > "%OUT%"
"%PY%" "%HERE%ml\train_model.py" "%WORK%\dataset.csv" --out-dir "%WORK%\model_real" | findstr /C:"Window accuracy" /C:"Recording accuracy" /C:"Chance" >> "%OUT%"
echo. >> "%OUT%"
echo SHUFFLED LABELS: >> "%OUT%"
"%PY%" "%HERE%ml\train_model.py" "%WORK%\shuffled.csv" --out-dir "%WORK%\model_shuffled" | findstr /C:"Window accuracy" /C:"Recording accuracy" /C:"Chance" >> "%OUT%"

echo.
echo RESULTS
echo ------------------------------------------------------------
type "%OUT%"
echo ------------------------------------------------------------
echo Saved in: %OUT%
echo Expected: real labels about 80%%, shuffled labels about 27%% (chance level).
if not defined NOPAUSE pause
