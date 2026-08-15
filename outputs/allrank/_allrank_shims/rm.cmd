@echo off
if /I "%~1"=="-rf" (
  if exist "%~2" rmdir /S /Q "%~2"
) else (
  if exist "%~1" del /F /Q "%~1"
)
