@echo off
chcp 65001 >nul
title LiveTranslate JP subtitles
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\workspace\language_translate\launcher\start_jp_subtitles.ps1"
