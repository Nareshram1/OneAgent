echo Checking if Ollama service is running...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Ollama is not running or not installed properly
    echo Please make sure Ollama is installed and the service is running
    pause
    exit /b 1
)

echo Ollama is running ✓

REM Start the FastAPI server
echo Starting FastAPI server on http://localhost:8000
python main.py

pause