# database.py

import os
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from supabase import create_client, Client

# --- Initial Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SupabaseStorage")


class SupabaseStorage:
    """Handles all database interactions with the Supabase relational schema."""

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        try:
            self.client: Client = create_client(url, key)
            logger.info("Supabase client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise

    def get_or_create_category(self, user_id: str, name: str) -> Dict:
        """Finds a category by name for a user, creating it if it doesn't exist."""
        try:
            # Check if category exists for the user (case-insensitive)
            result = self.client.table("categories").select("id, name").eq("user_id", user_id).ilike("name", name).execute()
            
            if result.data:
                return result.data[0]
            else:
                # Create it if it doesn't exist
                logger.info(f"Category '{name}' not found for user {user_id}. Creating it.")
                insert_result = self.client.table("categories").insert({
                    "user_id": user_id,
                    "name": name.strip().title() # Standardize category name
                }).select("id, name").execute()
                
                if insert_result.data:
                    return insert_result.data[0]
                else:
                    raise Exception(f"Failed to create new category '{name}'")
        except Exception as e:
            logger.error(f"Error in get_or_create_category: {e}")
            raise

    def add_expense(self, user_id: str, amount: float, description: str, category_name: str, expense_date: Optional[str]) -> Dict:
        """Adds a new expense, handling category lookup and creation."""
        try:
            # --- DEBUG LOGGING ADDED ---
            logger.info(
                f"Adding expense with params: user_id='{user_id}', amount={amount}, "
                f"description='{description}', category='{category_name}', date='{expense_date}'"
            )
            
            category = self.get_or_create_category(user_id, category_name)
            category_id = category['id']
            
            expense_data = {
                "user_id": user_id,
                "amount": amount,
                "description": description,
                "category_id": category_id,
                "expense_date": expense_date or datetime.now().isoformat(),
            }
            
            # --- FIX APPLIED HERE ---
            # Removed the incorrect .select("*") from the chain
            result = self.client.table("expenses").insert(expense_data).execute()
            
            if not result.data:
                raise Exception("Failed to insert expense, no data returned from Supabase.")
            
            return result.data[0]

        except Exception as e:
            logger.error(f"Error adding expense: {e}")
            raise

    def get_expenses(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Gets user expenses, joining with categories to retrieve the name."""
        try:
            # The query joins expenses with categories and selects all columns from expenses
            # plus the 'name' column from the related category.
            query = self.client.table("expenses").select("*, categories(name)").eq("user_id", user_id)
            result = query.order("expense_date", desc=True).limit(limit).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error fetching expenses: {e}")
            return []