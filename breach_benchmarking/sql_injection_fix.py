# SQL Injection Protection for Enhanced CSP
# Add this to your API endpoints

import re
import html
from typing import Any, Dict

class SQLInjectionProtector:
    """Protects against SQL injection attacks"""
    
    # Common SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(\b(UNION|OR|AND)\s+(SELECT|ALL)\b)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"('|(\\x27)|(\\x2D\\x2D)|(%27)|(%2D%2D))",
        r"(\b(OR|AND)\b\s*\w*\s*=)",
        r"(SLEEP\s*\(|WAITFOR\s+DELAY)",
        r"(xp_cmdshell|sp_executesql)"
    ]
    
    @classmethod
    def validate_input(cls, input_value: Any) -> bool:
        """Validate input for SQL injection patterns"""
        if not isinstance(input_value, str):
            return True
            
        # Convert to uppercase for pattern matching
        upper_input = input_value.upper()
        
        # Check against known patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, upper_input, re.IGNORECASE):
                return False
        
        return True
    
    @classmethod
    def sanitize_input(cls, input_value: str) -> str:
        """Sanitize input by removing dangerous characters"""
        if not isinstance(input_value, str):
            return str(input_value)
        
        # HTML escape
        sanitized = html.escape(input_value)
        
        # Remove SQL comment sequences
        sanitized = re.sub(r'--.*$', '', sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r'/\*.*?\*/', '', sanitized, flags=re.DOTALL)
        
        # Remove multiple quotes
        sanitized = re.sub(r"'+", "'", sanitized)
        sanitized = re.sub(r'"+', '"', sanitized)
        
        return sanitized.strip()

# Integration with FastAPI/Flask routes
from fastapi import HTTPException
from functools import wraps

def sql_injection_protection(func):
    """Decorator to protect API endpoints from SQL injection"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Check all string arguments
        for arg in args:
            if isinstance(arg, str) and not SQLInjectionProtector.validate_input(arg):
                raise HTTPException(status_code=400, detail="Invalid input detected")
        
        # Check all keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, str) and not SQLInjectionProtector.validate_input(value):
                raise HTTPException(status_code=400, detail="Invalid input detected")
            # Also check nested dictionaries
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, str) and not SQLInjectionProtector.validate_input(nested_value):
                        raise HTTPException(status_code=400, detail="Invalid input detected")
        
        return await func(*args, **kwargs)
    return wrapper

# Example usage in your Enhanced CSP API
@app.post("/api/login")
@sql_injection_protection
async def login_endpoint(request: LoginRequest):
    # This endpoint is now protected against SQL injection
    username = SQLInjectionProtector.sanitize_input(request.username)
    password = request.password  # Handle password separately with hashing
    
    # Use parameterized queries
    query = "SELECT * FROM users WHERE username = ? AND password_hash = ?"
    result = await database.execute(query, (username, hash_password(password)))
    
    return result

# Database query protection
class SecureDatabase:
    """Secure database wrapper with parameterized queries"""
    
    def __init__(self, connection):
        self.connection = connection
    
    async def execute_query(self, query: str, params: tuple = None):
        """Execute parameterized query to prevent SQL injection"""
        try:
            if params:
                # Use parameterized queries - this is the most important protection
                cursor = await self.connection.execute(query, params)
            else:
                cursor = await self.connection.execute(query)
            return await cursor.fetchall()
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise

# Example secure queries
async def get_user_by_id(user_id: int):
    """Secure user lookup"""
    db = SecureDatabase(database_connection)
    query = "SELECT * FROM users WHERE id = ?"
    return await db.execute_query(query, (user_id,))

async def search_processes(search_term: str):
    """Secure process search"""
    # Validate and sanitize input
    if not SQLInjectionProtector.validate_input(search_term):
        raise ValueError("Invalid search term")
    
    sanitized_term = SQLInjectionProtector.sanitize_input(search_term)
    
    db = SecureDatabase(database_connection)
    query = "SELECT * FROM processes WHERE name LIKE ? OR description LIKE ?"
    search_pattern = f"%{sanitized_term}%"
    return await db.execute_query(query, (search_pattern, search_pattern))
