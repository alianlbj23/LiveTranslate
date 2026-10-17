@echo off
chcp 65001 >nul
title LiveTranslate verify
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\workspace\language_translate\launcher\verify.ps1"
pause
