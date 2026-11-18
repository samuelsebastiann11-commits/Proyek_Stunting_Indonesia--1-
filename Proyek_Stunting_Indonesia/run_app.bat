@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    if exist "%USERPROFILE%\.local\bin\python3.14.exe" (
        "%USERPROFILE%\.local\bin\python3.14.exe" -m venv .venv
    ) else (
        where py >nul 2>nul
        if not errorlevel 1 (
            py -3 -m venv .venv
        ) else (
            where python >nul 2>nul
            if not errorlevel 1 (
                python -m venv .venv
            ) else (
                echo Python 3 was not found. Install Python 3 and try again.
                pause
                exit /b 1
            )
        )
    )
    if errorlevel 1 goto failed
)

".venv\Scripts\python.exe" -c "import streamlit, pandas, sklearn, joblib" >nul 2>nul
if errorlevel 1 (
    echo Installing project dependencies...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
)

if not exist "models\model_stunting.joblib" (
    echo Training the model...
    ".venv\Scripts\python.exe" train_model.py
    if errorlevel 1 goto failed
)

".venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" --server.headless true
exit /b %errorlevel%

:failed
echo.
echo The application could not be started. Check the error above.
pause
exit /b 1
