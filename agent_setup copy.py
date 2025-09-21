# agent_setup.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List
from tools import create_expense, create_note, fetch_expenses, fetch_notes
import os
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
    model="gemini-1.5-flash",   # you can also try "gemini-1.5-pro" for better reasoning
    temperature=0.2,            # keep it slightly creative but grounded
    max_output_tokens=512,      # similar to num_predict in Ollama
    google_api_key=GOOGLE_API_KEY
)

# Tool adapters — wrap functions for LangChain tools
def _tool_create_expense(input_str: str) -> str:
    pairs = dict(x.split('=', 1) for x in input_str.split(';') if '=' in x)
    res = create_expense(
        pairs['user_id'],
        float(pairs['amount']),
        pairs.get('description', 'expense'),
        pairs.get('date')
    )
    return str(res)

def _tool_create_note(input_str: str) -> str:
    pairs = dict(x.split('=', 1) for x in input_str.split(';') if '=' in x)
    res = create_note(pairs['user_id'], pairs['text'])
    return str(res)

def _tool_fetch_expenses(input_str: str) -> str:
    pairs = dict(x.split('=', 1) for x in input_str.split(';') if '=' in x)
    res = fetch_expenses(pairs['user_id'], int(pairs.get('limit', 20)))
    return str(res)

def _tool_fetch_notes(input_str: str) -> str:
    pairs = dict(x.split('=', 1) for x in input_str.split(';') if '=' in x)
    res = fetch_notes(pairs['user_id'], int(pairs.get('limit', 20)))
    return str(res)

def _tool_analyze(input_str: str) -> str:
    prompt = """You are a financial analysis assistant. You will be given **expense records** and **notes data**.  
Your task is to produce a clear, visually structured analysis.
NOTE: Never output raw json without formatting
📊 **What to include:**
- Spending patterns and trends
- Correlations between expenses and notes
- Unusual transactions or anomalies
- Budget insights and recommendations

📌 **Output format (must be visually easy to read in plain text):**

====================================================
📌 Key Findings
----------------------------------------------------
• Finding 1  
• Finding 2  
• Finding 3  

====================================================
📊 Spending Patterns
----------------------------------------------------
- Total spent this month: X
- Biggest category: FOOD (₹1234)
- Average daily spend: ₹YYY

(Optionally show a mini table:)

Category       | Amount | % of Total
---------------|--------|-----------
Food           | ₹1234  | 45%
Transport      | ₹567   | 20%
Shopping       | ₹890   | 35%

====================================================
🧩 Correlations
----------------------------------------------------
- Expense spikes vs. Notes context
- Emotional or situational notes tied to spending

====================================================
✅ Recommendations
----------------------------------------------------
1. Suggestion 1  
2. Suggestion 2  
3. Suggestion 3  
====================================================

Input data:
{input}

Keep tone professional but easy to scan.
"""
    
    chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["input"], template=prompt))
    out = chain.run(input=input_str)
    return out

TOOLS = [
    Tool.from_function(
        _tool_create_expense,
        name='create_expense',
        description='Create a new expense record. Input format: user_id=USER_ID;amount=AMOUNT;description=DESCRIPTION;date=YYYY-MM-DD'
    ),
    Tool.from_function(
        _tool_create_note,
        name='create_note',
        description='Create a new note. Input format: user_id=USER_ID;text=NOTE_TEXT'
    ),
    Tool.from_function(
        _tool_fetch_expenses,
        name='fetch_expenses',
        description='Fetch user expenses. Input format: user_id=USER_ID;limit=NUMBER (optional, default 20)'
    ),
    Tool.from_function(
        _tool_fetch_notes,
        name='fetch_notes',
        description='Fetch user notes. Input format: user_id=USER_ID;limit=NUMBER (optional, default 20)'
    ),
    Tool.from_function(
        _tool_analyze,
        name='analyze',
        description='Analyze expenses and notes data to provide insights. Input: Combined JSON data or structured text with expense and note information'
    )
]

# Initialize a LangChain agent with Gemini
AGENT = initialize_agent(
    TOOLS,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5,
    early_stopping_method="generate"
)
