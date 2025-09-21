import os
import json
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent_setup import AGENT
from typing import Optional
import requests
import uvicorn
from langfuse import Langfuse, observe
from dotenv import load_dotenv

# =====================
# Load environment vars
# =====================
BASE_DIR = Path(__file__).parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

langfuse = Langfuse(
  secret_key="sk-lf-10c01020-7e8b-40c8-b5de-82c8c879c7f2",
  public_key="pk-lf-998d5562-38ff-4bc7-a18d-913471a1a692",
  host="https://cloud.langfuse.com"
)

# observe = langfuse.observe
# print("langfuse",os.getenv("LANGFUSE_SECRET_KEY"))
# =====================
# Gemini health checker
# =====================
def check_gemini_connection():
    """Check if Gemini API is reachable"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("✗ No GOOGLE_API_KEY found in environment.")
        return False

    try:
        # Minimal test call to Gemini
        url = f"https://generativelanguage.googleapis.com/v1/models?key={api_key}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✓ Gemini API is reachable with {len(models)} models available")
            return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Gemini connection failed: {e}")
    return False

# =====================
# FastAPI setup
# =====================
app = FastAPI(title="OneAgent", version="1.0.0")
origins = [
    "http://localhost:3000", # The origin for your Next.js app
    "http://localhost",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    message: str
    error: Optional[str] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "running", "message": "Agentic Expense Agent is ready!"}

@app.get("/health")
async def health():
    """Detailed health check for Gemini"""
    gemini_status = check_gemini_connection()
    return {
        "status": "healthy" if gemini_status else "degraded",
        "gemini": "connected" if gemini_status else "disconnected",
        "model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    }

# =====================
# Chat endpoint
# =====================
@app.post("/chat")
@observe(name="OneAgent Chat", as_type="generation")
async def chat(req: ChatRequest):
    user_id = req.user_id
    message = req.message
    session_id = req.session_id or f"{user_id}:session"

    # Input validation
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required")
    if not message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    try:
        print(f"Processing request for user {user_id}: {message}")

        # Run the LangChain agent
        result = AGENT.invoke(f"user_id={user_id}; {message}")

        print(f"Agent response: {result}")

        return {"processed_data": result['output'], "status": "ok"}

    except Exception as e:
        error_msg = str(e)
        print(f"Agent error: {error_msg}")

        return ChatResponse(
            session_id=session_id,
            message="I encountered an error processing your request. Please try again.",
            error=error_msg,
        )

# =====================
# Server entrypoint
# =====================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Initiating OneAgent⚡Gemini")
    print("=" * 50)

    if not check_gemini_connection():
        print("\n❌ Cannot start server: Gemini API not accessible")
        print("Please check your GOOGLE_API_KEY and internet connection.")
        input("Press Enter to exit...")
        sys.exit(1)

    port = int(os.getenv("PORT", 8000))
    print(f"\n🌐 Starting server on http://localhost:{port}")
    print(f"📖 API docs available at http://localhost:{port}/docs")
    print(f"❤️  Health check at http://localhost:{port}/health")
    print("\n⏹️  Press Ctrl+C to stop the server")

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            reload=False,  # Disable reload for stability
            access_log=True,
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped gracefully")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        input("Press Enter to exit...")