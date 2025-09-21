# agent_setup.py - LangGraph Implementation
import os
from typing import Annotated, Sequence, TypedDict, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph.message import add_messages
from tools import create_expense, create_note, fetch_expenses, fetch_notes
from dotenv import load_dotenv
from pathlib import Path

# =====================
# Load environment vars
# =====================
BASE_DIR = Path(__file__).parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

# Load Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_output_tokens=512,
    google_api_key=GOOGLE_API_KEY,
    transport="rest"
)

# =====================
# Define State
# =====================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# =====================
# Define Tools using @tool decorator
# =====================
@tool
def tool_create_expense(input_str: str) -> str:
    """Create a new expense record. Input format: user_id=USER_ID;amount=AMOUNT;description=DESCRIPTION;date=YYYY-MM-DD"""
    try:
        pairs = dict(x.split('=', 1) for x in input_str.split(';') if '=' in x)
        res = create_expense(
            pairs['user_id'],
            float(pairs['amount']),
            pairs.get('description', 'expense'),
            pairs.get('date')
        )
        return str(res)
    except Exception as e:
        return f"Error creating expense: {str(e)}"

@tool
def tool_create_note(input_str: str) -> str:
    """Create a new note. Input format: user_id=USER_ID;text=NOTE_TEXT"""
    try:
        pairs = dict(x.split('=', 1) for x in input_str.split(';') if '=' in x)
        res = create_note(pairs['user_id'], pairs['text'])
        return str(res)
    except Exception as e:
        return f"Error creating note: {str(e)}"

@tool
def tool_fetch_expenses(input_str: str) -> str:
    """Fetch user expenses. Input format: user_id=USER_ID;limit=NUMBER (optional, default 20)"""
    try:
        pairs = dict(x.split('=', 1) for x in input_str.split(';') if '=' in x)
        res = fetch_expenses(pairs['user_id'], int(pairs.get('limit', 20)))
        return str(res)
    except Exception as e:
        return f"Error fetching expenses: {str(e)}"

@tool
def tool_fetch_notes(input_str: str) -> str:
    """Fetch user notes. Input format: user_id=USER_ID;limit=NUMBER (optional, default 20)"""
    try:
        pairs = dict(x.split('=', 1) for x in input_str.split(';') if '=' in x)
        res = fetch_notes(pairs['user_id'], int(pairs.get('limit', 20)))
        return str(res)
    except Exception as e:
        return f"Error fetching notes: {str(e)}"

@tool
def tool_analyze(input_str: str) -> str:
    """Analyze expenses and notes data to provide insights. Input: Combined JSON data or structured text with expense and note information"""
    try:
        analysis_prompt = f"""You are a financial analysis assistant. Analyze the following data and provide insights:

{input_str}

Provide a clear, structured analysis with:
📌 Key Findings
📊 Spending Patterns  
🧩 Correlations
✅ Recommendations

Format output to be visually easy to read in plain text with tables and bullet points where appropriate."""
        
        messages = [HumanMessage(content=analysis_prompt)]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# =====================
# Create Tools List
# =====================
tools = [
    tool_create_expense,
    tool_create_note,
    tool_fetch_expenses,
    tool_fetch_notes,
    tool_analyze
]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Create tool executor
tool_executor = ToolExecutor(tools)

# =====================
# Define Node Functions
# =====================
def should_continue(state: AgentState) -> str:
    """Determine whether to continue or end the conversation."""
    messages = state['messages']
    last_message = messages[-1]
    
    # If the LLM makes a tool call, then we route to the "tools" node
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END

def call_model(state: AgentState) -> Dict[str, Any]:
    """Call the model with the current state."""
    messages = state['messages']
    
    # Add system message for the agent's behavior
    system_message = """You are a helpful financial assistant that can:
1. Create expense records
2. Create notes
3. Fetch expense data
4. Fetch notes
5. Analyze financial data and provide insights

When users ask to create, fetch, or analyze data, use the appropriate tools.
Be helpful and provide clear responses. For analysis requests, fetch the relevant data first, then analyze it."""
    
    # Prepend system message if not already present
    if not messages or not isinstance(messages[0], AIMessage):
        messages = [AIMessage(content=system_message)] + list(messages)
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def call_tools(state: AgentState) -> Dict[str, Any]:
    """Execute tools based on the tool calls in the last message."""
    messages = state['messages']
    last_message = messages[-1]
    
    # We construct ToolInvocation for each tool call
    tool_invocations = []
    for tool_call in last_message.tool_calls:
        tool_invocations.append(
            ToolInvocation(
                tool=tool_call["name"],
                tool_input=tool_call["args"],
            )
        )
    
    # Execute tools and collect responses
    responses = tool_executor.batch(tool_invocations)
    
    # Convert responses to ToolMessages
    tool_messages = []
    for response, tool_call in zip(responses, last_message.tool_calls):
        tool_messages.append(
            ToolMessage(
                content=str(response),
                tool_call_id=tool_call["id"],
            )
        )
    
    return {"messages": tool_messages}

# =====================
# Build the Graph
# =====================
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)

# Set the entrypoint
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    }
)

# Add normal edge from tools to agent
workflow.add_edge("tools", "agent")

# Compile the graph
AGENT = workflow.compile()

# =====================
# Helper Functions
# =====================
def run_agent(query: str, user_id: str = None) -> str:
    """
    Run the agent with a query.
    
    Args:
        query: The user's question or command
        user_id: Optional user ID for context
        
    Returns:
        The agent's response
    """
    # Add user_id context if provided
    if user_id:
        query = f"[User ID: {user_id}] {query}"
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # Run the agent
    result = AGENT.invoke(initial_state)
    
    # Return the last AI message
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    
    return "No response generated"

async def run_agent_async(query: str, user_id: str = None) -> str:
    """
    Async version of run_agent.
    """
    if user_id:
        query = f"[User ID: {user_id}] {query}"
    
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    result = await AGENT.ainvoke(initial_state)
    
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage):
            return message.content
    
    return "No response generated"

def stream_agent(query: str, user_id: str = None):
    """
    Stream the agent's response.
    """
    if user_id:
        query = f"[User ID: {user_id}] {query}"
    
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    for chunk in AGENT.stream(initial_state):
        yield chunk

# =====================
# Example Usage
# =====================
if __name__ == "__main__":
    # Test the agent
    print("🤖 Financial Assistant Agent Ready!")
    print("=" * 50)
    
    # Example queries
    test_queries = [
        "Create an expense for user123: amount=50.0, description=Coffee, date=2024-01-15",
        "Fetch expenses for user123",
        "Create a note for user123: text=Had coffee meeting with client",
        "Analyze my financial data for user123"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("🤖 Response:")
        response = run_agent(query)
        print(response)
        print("-" * 30)