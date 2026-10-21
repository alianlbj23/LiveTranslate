@echo off
rem LiveTranslate (debug): runs in this console window and shows the live log.
rem The normal entry is LiveTranslate.exe, which opens no console window at all.
cd /d "%~dp0"
title LiveTranslate (debug)

if not exist "LiveTranslate.exe" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\build_exe.ps1"
)

if not exist ".venv\Scripts\python.exe" goto setup
if not exist ".venv\.livetranslate-ready" goto setup
if not exist "user_settings.json" goto setup
goto run

:setup
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\setup.ps1"
if errorlevel 1 (
    echo.
    echo [ERROR] Setup failed. See the messages above.
    pause
    exit /b 1
)

:run
".venv\Scripts\python.exe" main.py
if errorlevel 1 (
    echo.
    echo [ERROR] LiveTranslate exited with an error. See the log above.
    pause
)
