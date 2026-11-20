@echo off
rem Double-click: opens the drawing screen in the browser. Close this window to stop.
set "PY=%LOCALAPPDATA%\Microsoft\WindowsApps\py.exe"
if not exist "%PY%" set "PY=py"
"%PY%" -3.13 -I "%~dp0scripts\app.py"
pause
