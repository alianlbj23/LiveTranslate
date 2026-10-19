@echo off
chcp 65001 >nul
title LiveTranslate
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\workspace\language_translate\launcher\menu.ps1" %*
