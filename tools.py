# tools.py
import os
import requests
from typing import List, Dict, Optional
from datetime import datetime
import json

# API Configuration
EXPENSE_BASE = os.getenv('EXPENSE_API_BASE', 'http://localhost:3000/api')
EXPENSE_API_KEY = os.getenv('EXPENSE_API_KEY', '')
NOTES_BASE = os.getenv('NOTES_API_BASE', 'http://localhost:4100')

# Headers setup
HEADERS = {'Content-Type': 'application/json'}
if EXPENSE_API_KEY:
    HEADERS['Authorization'] = f"Bearer {EXPENSE_API_KEY}"

def create_expense(user_id: str, amount: float, description: str = 'expense', date: str = None) -> Dict:
    """Create a new expense record"""
    try:
        # Use current date if none provided
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
            
        payload = {
            'user_id': user_id, 
            'amount': float(amount), 
            'description': description, 
            'date': date
        }
        
        print(f"Creating expense: {payload}")
        
        resp = requests.post(
            f"{EXPENSE_BASE}/expenses", 
            json=payload, 
            headers=HEADERS, 
            timeout=10
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"Expense created successfully: {result}")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error creating expense: {e}")
        return {"error": f"Failed to create expense: {str(e)}"}
    except Exception as e:
        print(f"Unexpected error creating expense: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

def create_note(user_id: str, text: str) -> Dict:
    """Create a new note"""
    try:
        payload = {'user_id': user_id, 'text': text}
        print(f"Creating note: {payload}")
        
        resp = requests.post(
            f"{NOTES_BASE}/notes", 
            json=payload, 
            headers={'Content-Type': 'application/json'}, 
            timeout=10
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"Note created successfully: {result}")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error creating note: {e}")
        return {"error": f"Failed to create note: {str(e)}"}
    except Exception as e:
        print(f"Unexpected error creating note: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

def fetch_expenses(user_id: str, limit: int = 20) -> List[Dict]:
    """Fetch user's expense records via POST request"""
    try:
        payload = {"userId": user_id, "limit": limit}
        print(f"Fetching expenses for user {user_id}, limit: {limit}")

        resp = requests.post(
            f"{EXPENSE_BASE}/fetchExpenses",
            json=payload,
            headers=HEADERS,
            timeout=10
        )
        resp.raise_for_status()

        result = resp.json()
        expenses = result.get("expenses", [])  # because your API returns {userId, expenses}
        print(f"Fetched {len(expenses)} expenses")

        return expenses

    except requests.exceptions.RequestException as e:
        print(f"Error fetching expenses: {e}")
        return [{"error": f"Failed to fetch expenses: {str(e)}"}]
    except Exception as e:
        print(f"Unexpected error fetching expenses: {e}")
        return [{"error": f"Unexpected error: {str(e)}"}]

def fetch_notes(user_id: str, limit: int = 20) -> List[Dict]:
    """Fetch user's notes"""
    try:
        params = {'user_id': user_id, 'limit': limit}
        print(f"Fetching notes for user {user_id}, limit: {limit}")
        
        resp = requests.get(
            f"{NOTES_BASE}/notes", 
            params=params, 
            headers={'Content-Type': 'application/json'}, 
            timeout=10
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"Fetched {len(result) if isinstance(result, list) else 'unknown'} notes")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching notes: {e}")
        return [{"error": f"Failed to fetch notes: {str(e)}"}]
    except Exception as e:
        print(f"Unexpected error fetching notes: {e}")
        return [{"error": f"Unexpected error: {str(e)}"}]

# Mock functions for testing when APIs aren't available
def mock_create_expense(user_id: str, amount: float, description: str = 'expense', date: str = None) -> Dict:
    """Mock expense creation for testing"""
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    
    return {
        "id": f"exp_{user_id}_{int(datetime.now().timestamp())}",
        "user_id": user_id,
        "amount": amount,
        "description": description,
        "date": date,
        "created_at": datetime.now().isoformat()
    }

def mock_create_note(user_id: str, text: str) -> Dict:
    """Mock note creation for testing"""
    return {
        "id": f"note_{user_id}_{int(datetime.now().timestamp())}",
        "user_id": user_id,
        "text": text,
        "created_at": datetime.now().isoformat()
    }

def mock_fetch_expenses(user_id: str, limit: int = 20) -> List[Dict]:
    """Mock expense fetching for testing"""
    return [
        {
            "id": "exp_001",
            "user_id": user_id,
            "amount": 25.50,
            "description": "Coffee and pastry",
            "date": "2025-09-20",
            "created_at": "2025-09-20T10:30:00"
        },
        {
            "id": "exp_002", 
            "user_id": user_id,
            "amount": 120.00,
            "description": "Grocery shopping",
            "date": "2025-09-19",
            "created_at": "2025-09-19T15:45:00"
        }
    ]

def mock_fetch_notes(user_id: str, limit: int = 20) -> List[Dict]:
    """Mock note fetching for testing"""
    return [
        {
            "id": "note_001",
            "user_id": user_id, 
            "text": "Remember to track coffee expenses separately",
            "created_at": "2025-09-20T09:00:00"
        },
        {
            "id": "note_002",
            "user_id": user_id,
            "text": "Monthly grocery budget is $400",
            "created_at": "2025-09-19T14:30:00"
        }
    ]

# Test connectivity function
def test_api_connectivity():
    """Test if APIs are reachable"""
    results = {
        "expense_api": False,
        "notes_api": False
    }
    
    try:
        resp = requests.get(f"{EXPENSE_BASE}/health", timeout=5)
        results["expense_api"] = resp.status_code == 200
    except:
        pass
        
    try:
        resp = requests.get(f"{NOTES_BASE}/health", timeout=5)
        results["notes_api"] = resp.status_code == 200
    except:
        pass
        
    return results

# Auto-detect whether to use real or mock functions
def get_api_functions():
    """Return real or mock functions based on API availability"""
    connectivity = test_api_connectivity()
    
    if connectivity["expense_api"] and connectivity["notes_api"]:
        print("✓ Using real APIs")
        return {
            "create_expense": create_expense,
            "create_note": create_note,
            "fetch_expenses": fetch_expenses,
            "fetch_notes": fetch_notes
        }
    else:
        print("⚠ APIs not available, using mock functions for testing")
        return {
            "create_expense": mock_create_expense,
            "create_note": mock_create_note, 
            "fetch_expenses": mock_fetch_expenses,
            "fetch_notes": mock_fetch_notes
        }

if __name__ == "__main__":
    # Test the functions
    print("Testing API connectivity...")
    connectivity = test_api_connectivity()
    print(f"Expense API: {'✓' if connectivity['expense_api'] else '✗'}")
    print(f"Notes API: {'✓' if connectivity['notes_api'] else '✗'}")
    
    # Test mock functions
    print("\nTesting mock functions...")
    funcs = get_api_functions()
    
    test_user = "test_user_123"
    
    # Test expense creation
    expense_result = funcs["create_expense"](test_user, 15.99, "Test expense")
    print(f"Created expense: {expense_result}")
    
    # Test note creation  
    note_result = funcs["create_note"](test_user, "This is a test note")
    print(f"Created note: {note_result}")
    
    # Test fetching
    expenses = funcs["fetch_expenses"](test_user, 5)
    notes = funcs["fetch_notes"](test_user, 5)
    print(f"Fetched {len(expenses)} expenses and {len(notes)} notes")