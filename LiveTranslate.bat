@echo off
chcp 65001 >nul
title LiveTranslate
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launcher\menu.ps1" %*
