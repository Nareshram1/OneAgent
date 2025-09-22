import os
import json
import sys
from pathlib import Path
import asyncio
import uvicorn
import requests
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langfuse import Langfuse,observe,get_client
from langchain_core.messages import HumanMessage

# Import the modernized agent
from agent_setup import AGENT

# ==============================================================================
# 1. Environment and Service Setup
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Initialize Langfuse for observability
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="https://cloud.langfuse.com"
)

# ==============================================================================
# 2. FastAPI Application Setup
# ==============================================================================
app = FastAPI(
    title="OneAgent API",
    version="2.0.0",
    description="An API powered by a modern LangGraph agent for financial tasks.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 3. Pydantic Models for API Data Validation
# ==============================================================================
class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    message: str
    error: Optional[str] = None

# ==============================================================================
# 4. Health Check and Utility Functions
# ==============================================================================
@app.get("/", tags=["Status"])
async def root():
    """A simple health check endpoint to confirm the server is running."""
    return {"status": "running", "message": "Modern Agent API is ready!"}

def check_gemini_connection() -> bool:
    """Checks if the Google Gemini API is reachable with the provided key."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("✗ No GOOGLE_API_KEY found in environment.")
        return False
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        print("✓ Gemini API connection successful.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Gemini API connection failed: {e}")
        return False

@app.get("/health", tags=["Status"])
async def health():
    """Provides a detailed health check of the API and its dependencies."""
    gemini_ok = check_gemini_connection()
    return {
        "status": "healthy" if gemini_ok else "degraded",
        "dependencies": {
            "gemini_api": "connected" if gemini_ok else "disconnected",
        },
        "agent_details": {
            "model": "gemini-1.5-flash",
            "architecture": "LangGraph",
        },
    }

# ==============================================================================
# 5. Core API Endpoints
# ==============================================================================
@app.post("/chat", tags=["Agent"])
@observe(name="OneAgentChat API",capture_input=True,capture_output=True)
async def chat(req: ChatRequest):
    """
    Handles a non-streaming chat request with the agent.
    This endpoint uses the agent's built-in memory, using session_id as the thread_id.
    """
    if not req.user_id or not req.message:
        raise HTTPException(status_code=400, detail="user_id and message are required.")

    session_id = req.session_id or f"user-{req.user_id}-thread"
    # trace = langfuse.trace(name="OneAgentChat", user_id=req.user_id, session_id=session_id)
    # span = trace.span(name="run-agent-invoke", input={"message": req.message})

    try:
        # The agent manages its own history using the `thread_id` in the config
        config = {"configurable": {"thread_id": session_id}}
        
        # We only need to send the latest human message
        initial_state = {"messages": [HumanMessage(content=req.message)]}
        
        # Asynchronously invoke the agent
        final_state = await AGENT.ainvoke(initial_state, config=config)
        
        # The final response is the last message in the state
        response_message = final_state['messages'][-1].content
        
        # span.end(output={"response": response_message})
        # return ChatResponse(session_id=session_id, message=response_message)
        return {"session_id":session_id, "message":response_message}

    except Exception as e:
        print(f"Agent error for user {req.user_id}: {e}")
        # span.end(level="ERROR", status_message=str(e))
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


@app.post("/chat/stream", tags=["Agent"])
@observe(name="OneAgentChat stream",capture_input=True,capture_output=True)
async def chat_stream(req: ChatRequest):
    """
    Handles a streaming chat request, providing real-time updates from the agent's process.
    """
    if not req.user_id or not req.message:
        raise HTTPException(status_code=400, detail="user_id and message are required.")
    langfuse = get_client()
    session_id = req.session_id or f"user-{req.user_id}-thread"
    langfuse.update_current_trace(session_id=session_id)

    config = {"configurable": {"thread_id": session_id}}
    initial_state = {"messages": [HumanMessage(content=req.message)]}

    async def event_generator():
        """The generator function that yields server-sent events with detailed agent state."""
        try:
            # Yield a start event
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"

            # Use `astream` to get real-time updates from the graph
            async for chunk in AGENT.astream(initial_state, config=config):
                
                # Check for updates from the 'agent' node
                if "agent" in chunk:
                    agent_messages = chunk["agent"].get("messages", [])
                    if agent_messages:
                        latest_message = agent_messages[-1]
                        
                        # If the agent decides to call a tool, stream that information
                        if latest_message.tool_calls:
                            tool_names = [tc['name'] for tc in latest_message.tool_calls]
                            yield f"data: {json.dumps({'type': 'tool_call', 'tools': tool_names})}\n\n"
                        
                        # If the agent generates a text response, stream the token
                        if latest_message.content:
                            yield f"data: {json.dumps({'type': 'token', 'content': latest_message.content})}\n\n"

                # Check for updates from the 'tools' node (after a tool has run)
                if "tools" in chunk:
                    tool_messages = chunk["tools"].get("messages", [])
                    if tool_messages:
                        # Stream the actual output from the tool for debugging purposes
                        tool_outputs = [msg.content for msg in tool_messages]
                        yield f"data: {json.dumps({'type': 'tool_result', 'outputs': tool_outputs})}\n\n"
                
                await asyncio.sleep(0.05) # Small delay for smoother streaming

            # Yield an end event once the stream is complete
            yield f"data: {json.dumps({'type': 'end', 'session_id': session_id})}\n\n"

        except Exception as e:
            print(f"Streaming error for user {req.user_id}: {e}")
            error_payload = json.dumps({"type": "error", "content": str(e)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")



# ==============================================================================
# 6. Server Entrypoint
# ==============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 OneAgent API server starting...")
    print("=" * 50)

    if not check_gemini_connection():
        print("\n❌ CRITICAL: Cannot start server. Gemini API is not accessible.")
        sys.exit(1)

    port = int(os.getenv("PORT", 8000))
    print(f"\n🌐 Server running on http://localhost:{port}")
    print(f"📖 API docs at http://localhost:{port}/docs")
    print("⏹️ Press Ctrl+C to stop.")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")