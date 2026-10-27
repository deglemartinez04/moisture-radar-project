@echo off
setlocal
title ML Check - one-time setup
set "ENVDIR=%LOCALAPPDATA%\MoistureRadarML"
echo.
echo ONE-TIME SETUP
echo This prepares Python for the checks (needs internet, about 1-3 minutes).
echo It is installed in %ENVDIR%, not in this folder.
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found on this computer.
  echo Install Python 3 from python.org and tick "Add python.exe to PATH", then run this again.
  if not defined NOPAUSE pause
  exit /b 1
)

rem Start clean, and reuse the packages already installed with Python (numpy, scipy, ...).
rem Windows can block brand-new downloads of those, so only what is missing is installed.
if exist "%ENVDIR%\venv" rmdir /s /q "%ENVDIR%\venv"
echo Creating the Python environment...
python -m venv --system-site-packages "%ENVDIR%\venv"
if errorlevel 1 (
  echo Could not create the environment.
  if not defined NOPAUSE pause
  exit /b 1
)

echo Installing the missing packages...
"%ENVDIR%\venv\Scripts\python.exe" -m pip install numpy scipy h5py pandas scikit-learn joblib
if errorlevel 1 (
  echo.
  echo The install failed. Check your internet connection and run this again.
  if not defined NOPAUSE pause
  exit /b 1
)

echo.
echo Checking that every package loads...
"%ENVDIR%\venv\Scripts\python.exe" -c "import numpy, scipy, h5py, pandas, sklearn, joblib; print('All packages load OK')"
if errorlevel 1 (
  echo.
  echo A package was installed but Windows would not load it.
  echo If the message above says "Application Control policy has blocked this file",
  echo Windows security is blocking a newly downloaded file. Tell Claude what it says.
  if not defined NOPAUSE pause
  exit /b 1
)

echo.
echo DONE. You can now double-click 2_RUN_CHECK.bat
if not defined NOPAUSE pause
