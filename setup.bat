@echo off
REM Mental Health Tweet Classifier - Windows Batch Script
REM Simple wrapper for PowerShell setup script

setlocal enabledelayedexpansion

REM Check if PowerShell is available
where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: PowerShell not found. Please ensure PowerShell is installed and in PATH.
    pause
    exit /b 1
)

REM Set execution policy for current process (to allow script execution)
powershell -Command "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force"

REM Run the PowerShell script with arguments
if "%~1"=="" (
    powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1" help
) else (
    powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
)

pause