REM Create .env file
echo # Ollama settings > .env
echo OLLAMA_BASE_URL=http://localhost:11434 >> .env
echo OLLAMA_MODEL=llama3.2:3b >> .env
echo. >> .env
echo # Your API settings >> .env
echo EXPENSE_API_BASE=http://localhost:4000 >> .env
echo EXPENSE_API_KEY= >> .env
echo NOTES_API_BASE=http://localhost:4100 >> .env
echo. >> .env
echo # Optional Langfuse settings >> .env
echo LANGFUSE_API_URL=https://api.langfuse.com/v1/events >> .env
echo LANGFUSE_API_KEY= >> .env
echo. >> .env
echo PORT=8000 >> .env

echo.
echo Setup complete! 
echo.
echo Next steps:
echo 1. Copy your Python files (main.py, tools.py, agent_setup.py) to this folder
echo 2. Run: venv\Scripts\activate.bat
echo 3. Run: python main.py
echo.
pause