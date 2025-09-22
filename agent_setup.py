import os
from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from tools import create_expense, create_note, fetch_expenses, fetch_notes
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.tools import tool
import datetime
import logging
# ==============================================================================
# 1. Setup Environment and LLM
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY,
)

memory = MemorySaver()

# ==============================================================================
# 2. Define More Robust and Modern Tools
# - Tools now summarize data to be token-efficient and avoid context overflows.
# ==============================================================================
@tool
def create_expense_record(
    user_id: Annotated[str, "The user's unique identifier (UUID)."],
    amount: Annotated[float, "The monetary value of the expense. For example, 15.50"],
    description: Annotated[str, "A detailed description of the expense. For example, 'Coffee with a friend'"],
    category_name: Annotated[str, "The category of the expense. For example, 'Food', 'Transport', or 'Entertainment'"],
    date: Annotated[Optional[str], "The date of the expense in YYYY-MM-DD format. If not provided, today's date is used."] = None,
) -> str:
    """Creates a new expense record for a given user in the database."""
    try:
        res = create_expense(user_id, amount, description, category_name, date)
        return f"Successfully created expense: {res['amount']} for '{res['description']}' in category '{category_name}'. (ID: {res['id']})"
    except Exception as e:
        logging.error(f"Error in create_expense_record tool: {e}")
        return f"Error creating expense: {e}"

@tool
def create_note_record(
    user_id: Annotated[str, "The unique identifier for the user."],
    text: Annotated[str, "The content of the note to be saved."],
) -> str:
    """Creates a new note for a given user."""
    try:
        res = create_note(user_id, text)
        return f"Successfully created note: {res}"
    except Exception as e:
        return f"Error creating note: {e}"

@tool
def fetch_user_expenses(
    user_id: Annotated[str, "The user's unique identifier (UUID)."],
) -> str:
    """
    Fetches and summarizes the most recent expense records for a given user from the database.
    This tool provides a concise, formatted summary for the language model.
    """
    try:
        expenses = fetch_expenses(user_id, limit=20)
        
        if not expenses:
            return "No expenses found for this user."

        total_expenses = len(expenses)
        total_amount = sum(item.get('amount', 0) for item in expenses if isinstance(item.get('amount'), (int, float)))
        
        summary = f"Found {total_expenses} expenses totaling {total_amount:.2f}.\n\n"
        summary += "Here are the most recent expenses:\n"
        
        for expense in expenses[:7]: # Show up to 7 recent ones
            # Safely access the nested category name
            category_name = expense.get('categories', {}).get('name', 'N/A') if expense.get('categories') else 'N/A'
            
            date_str = expense.get('expense_date', '')
            formatted_date = 'N/A'
            if date_str:
                try:
                    date_obj = datetime.datetime.fromisoformat(str(date_str))
                    formatted_date = date_obj.strftime('%b %d, %Y') # e.g., Sep 22, 2025
                except (ValueError, TypeError):
                    logging.warning(f"Could not parse date: {date_str}")

            amount = expense.get('amount', 0)
            desc = expense.get('description', 'No description')
            summary += f"- **Date:** {formatted_date}, **Amount:** {amount:.2f}, **Category:** {category_name}, **Details:** {desc}\n"
            
        return summary
    except Exception as e:
        logging.error(f"Error in fetch_user_expenses tool: {e}")
        return f"Error fetching expenses: {e}"


@tool
def fetch_user_notes(
    user_id: Annotated[str, "The unique identifier for the user."],
    limit: Annotated[int, "The maximum number of notes to retrieve."] = 20,
) -> str:
    """
    Fetches and summarizes the most recent notes for a given user.
    Returns a concise summary instead of raw data.
    """
    try:
        notes = fetch_notes(user_id, limit)
        if not notes:
            return "No notes found for this user."

        summary = f"Found {len(notes)} notes. Here are the 5 most recent:\n"
        for note in notes[:5]:
             summary += f"- {note.get('text', 'No content')}\n"
        return summary
    except Exception as e:
        return f"Error fetching notes: {e}"

# List of tools for the agent
tools = [
    create_expense_record,
    create_note_record,
    fetch_user_expenses,
    fetch_user_notes,
]

# Bind the tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# ==============================================================================
# 3. Define Agent State and Graph Structure
# ==============================================================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful and efficient financial assistant. Your primary job is to help users manage their expenses and notes by using the available tools.

- **Crucially, you must always determine the `user_id` to use the tools.** The user might provide it directly in their message (e.g., "for user XYZ" or "my ID is 123"). Extract it from their query.
- When asked to fetch data, use the appropriate tool and present the summary it returns directly to the user in a clean, readable format.
- If a user asks for analysis, provide insights based *only* on the summary data you have.
- Always be clear about the results of an action (e.g., "Successfully created...").""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

agent_runnable = prompt | llm_with_tools

# Define the nodes of the graph
def agent_node(state: AgentState):
    return {"messages": [agent_runnable.invoke(state)]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# ==============================================================================
# 4. Construct and Compile the Graph
# ==============================================================================
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
AGENT = workflow.compile(checkpointer=memory)