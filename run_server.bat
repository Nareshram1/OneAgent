@echo off
echo OneAgent initiating...

echo Enabling local API's
REM Check if port 3000 is already in use
netstat -ano | findstr ":3000" >nul
if %ERRORLEVEL%==0 (
    echo Next.js project already running on port 3000, skipping...
) else (
    echo Starting Next.js project...
    cd /d E:\projects\spendify-hub
    start "" cmd /k "npm run dev"
)

REM Wait a few seconds to make sure Next.js is initialized
timeout /t 15 >nul

REM Start Python script
echo Starting OneAgent...
cd /d E:\projects\oneagent
start "" cmd /k "E:\projects\oneagent\venv\Scripts\activate.bat && python main.py"

echo All done!
pause