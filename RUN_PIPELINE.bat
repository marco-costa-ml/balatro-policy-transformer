@echo off
setlocal
cd /d "%~dp0"
python run_pipeline.py %*
exit /b %ERRORLEVEL%
