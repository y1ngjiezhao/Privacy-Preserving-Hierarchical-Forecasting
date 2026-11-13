@echo off
setlocal EnableExtensions
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto :python_error
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r windows_gpu\requirements.txt
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m ipykernel install --user --name fhf-4070-super --display-name "Python (FHF RTX 4070 SUPER)"
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" windows_gpu\check_environment.py
if errorlevel 1 goto :failed

echo.
echo Setup completed. Run windows_gpu\start_jupyter.bat.
echo Select the kernel: Python (FHF RTX 4070 SUPER).
pause
exit /b 0

:python_error
echo.
echo Install 64-bit Python 3.12 with the Windows Python launcher, then retry.
echo If using Conda, follow the alternative commands in windows_gpu\README_zh.md.
pause
exit /b 1

:failed
echo.
echo Setup or verification failed. Read the error above before starting training.
echo Check the NVIDIA driver, network connection, and Python environment.
pause
exit /b 1
