@echo off
chcp 65001 >nul
title LiveTranslate maintenance menu
powershell -NoProfile -ExecutionPolicy Bypass -File "D:\workspace\language_translate\launcher\menu.ps1"
