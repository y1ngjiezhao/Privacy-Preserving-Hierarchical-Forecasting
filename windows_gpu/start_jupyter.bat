@echo off
setlocal EnableExtensions
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Run windows_gpu\setup.bat first, or use the Conda instructions in README_zh.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m jupyterlab
if errorlevel 1 pause
