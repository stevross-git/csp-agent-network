# File: backend/auth/auth_monitoring.py
"""
Authentication Monitoring Integration
====================================
Instrumentation for authentication endpoints and operations
"""

import time
import functools
from typing import Optional, Dict, Any
from datetime import datetime

# Import monitoring system
try:
    from monitoring.csp_monitoring import get_default as get_monitoring
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    get_monitoring = lambda: None

def monitor_auth_endpoint(method: str):
    """Decorator to monitor authentication endpoints"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            monitor = get_monitoring() if MONITORING_AVAILABLE else None
            start_time = time.time()
            
            try:
                # Execute the auth function
                result = await func(*args, **kwargs)
                
                # Record success
                if monitor:
                    monitor.record_auth_attempt(method, success=True)
                
                return result
                
            except Exception as e:
                # Record failure
                if monitor:
                    monitor.record_auth_attempt(method, success=False)
                raise
            
        return wrapper
    return decorator

def monitor_token_validation(token_type: str):
    """Decorator to monitor token validation"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            monitor = get_monitoring() if MONITORING_AVAILABLE else None
            
            try:
                # Execute validation
                result = await func(*args, **kwargs)
                
                # Record validation result
                if monitor:
                    monitor.record_token_validation(
                        token_type=token_type,
                        valid=bool(result)
                    )
                
                return result
                
            except Exception as e:
                # Record invalid token
                if monitor:
                    monitor.record_token_validation(
                        token_type=token_type,
                        valid=False
                    )
                raise
        
        return wrapper
    return decorator

class AuthMetricsCollector:
    """Collects and reports authentication metrics"""
    
    def __init__(self):
        self.monitor = get_monitoring() if MONITORING_AVAILABLE else None
        self._session_counts = {}
    
    async def update_session_metrics(self, auth_service):
        """Update active session metrics"""
        if not self.monitor:
            return
        
        # Count sessions by auth method
        azure_sessions = 0
        local_sessions = 0
        
        if hasattr(auth_service, 'active_sessions'):
            for session in auth_service.active_sessions.values():
                if session.get('auth_method') == 'azure':
                    azure_sessions += 1
                else:
                    local_sessions += 1
        
        # Update metrics
        self.monitor.update_active_sessions('azure', azure_sessions)
        self.monitor.update_active_sessions('local', local_sessions)
        self.monitor.update_active_sessions('total', azure_sessions + local_sessions)
    
    def record_login(self, method: str, user_id: str, success: bool, 
                     duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Record detailed login attempt"""
        if not self.monitor:
            return
        
        # Basic metric
        self.monitor.record_auth_attempt(method, success)
        
        # Additional context for logging
        if metadata:
            # Could send to logging system or event store
            pass
    
    def record_logout(self, method: str, user_id: str):
        """Record logout event"""
        # Update session count will be handled by periodic update
        pass
    
    def record_permission_check(self, user_id: str, permission: str, 
                               granted: bool, duration: float):
        """Record permission check"""
        if not self.monitor:
            return
        
        # Could add specific permission metrics if needed
        pass

# Global metrics collector instance
auth_metrics = AuthMetricsCollector()

# Example instrumented auth functions
@monitor_auth_endpoint("azure_ad")
async def monitored_azure_login(credentials: Dict[str, Any]):
    """Azure AD login with monitoring"""
    # Actual login implementation
    pass

@monitor_auth_endpoint("local")
async def monitored_local_login(username: str, password: str):
    """Local login with monitoring"""
    # Actual login implementation
    pass

@monitor_token_validation("access_token")
async def monitored_validate_access_token(token: str):
    """Validate access token with monitoring"""
    # Actual validation implementation
    pass

@monitor_token_validation("refresh_token")
async def monitored_validate_refresh_token(token: str):
    """Validate refresh token with monitoring"""
    # Actual validation implementation
    pass
