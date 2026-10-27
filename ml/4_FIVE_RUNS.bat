@echo off
setlocal
title ML Check - five different simulated datasets
set "PY=%LOCALAPPDATA%\MoistureRadarML\venv\Scripts\python.exe"
set "WORK=%LOCALAPPDATA%\MoistureRadarML\work_five"
set "HERE=%~dp0"
set "OUT=%HERE%results\five_runs.txt"
if not exist "%PY%" (
  echo Not set up yet. Double-click 1_SETUP_once.bat first.
  if not defined NOPAUSE pause
  exit /b 1
)
if not exist "%HERE%results" mkdir "%HERE%results"
if exist "%WORK%" rmdir /s /q "%WORK%"
if exist "%OUT%" del "%OUT%"

echo.
echo FIVE DIFFERENT SIMULATED DATASETS  (about 1 minute)
echo Same simulator, different random draws. Seed 0 is the 80%% one.
echo.
for %%S in (0 1 2 3 4) do (
  echo Running dataset %%S of 4...
  "%PY%" "%HERE%ml\make_synthetic.py" -o "%WORK%\seed%%S\recordings" --seed %%S >nul
  "%PY%" "%HERE%ml\build_dataset.py" "%WORK%\seed%%S\recordings" -o "%WORK%\seed%%S\dataset.csv" >nul
  echo seed %%S: >> "%OUT%"
  "%PY%" "%HERE%ml\train_model.py" "%WORK%\seed%%S\dataset.csv" --out-dir "%WORK%\seed%%S\model" | findstr /C:"Window accuracy" /C:"Recording accuracy" >> "%OUT%"
)

echo.
echo RESULTS
echo ------------------------------------------------------------
type "%OUT%"
echo ------------------------------------------------------------
echo Saved in: %OUT%
echo Expected recording accuracy: 80.0%%, 73.3%%, 90.0%%, 76.7%%, 93.3%%
if not defined NOPAUSE pause
