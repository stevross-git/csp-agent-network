"""
Authentication monitoring instrumentation
"""
from functools import wraps
from typing import Callable, Any
import time
import logging

# Import monitoring system
try:
    from monitoring import get_default
    monitor = get_default()
    MONITORING_ENABLED = True
except ImportError:
    monitor = None
    MONITORING_ENABLED = False

logger = logging.getLogger(__name__)

def monitor_auth(method: str):
    """Decorator to monitor authentication operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not MONITORING_ENABLED:
                return await func(*args, **kwargs)
            
            start_time = time.time()
            success = False
            
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                # Record metrics
                monitor.record_auth_attempt(method, success)
                
                # Log for debugging
                duration = time.time() - start_time
                logger.info(f"Auth {method}: success={success}, duration={duration:.3f}s")
        
        return wrapper
    return decorator

def monitor_token_validation(token_type: str):
    """Decorator to monitor token validation"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not MONITORING_ENABLED:
                return await func(*args, **kwargs)
            
            valid = False
            try:
                result = await func(*args, **kwargs)
                valid = result is not None
                return result
            finally:
                monitor.record_token_validation(token_type, valid)
        
        return wrapper
    return decorator

def update_session_count(auth_method: str, count: int):
    """Update active session count"""
    if MONITORING_ENABLED:
        monitor.update_active_sessions(auth_method, count)
