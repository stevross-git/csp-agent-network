# File: backend/monitoring/enhanced_metrics.py
"""
Enhanced metrics with proper labeling
"""

from prometheus_client import Counter, Histogram, Gauge
from typing import Dict, Any
import time

# Enhanced security metrics with labels
security_events = Counter(
    'security_events_total',
    'Total security events by type, endpoint, and tenant',
    ['event_type', 'endpoint', 'tenant', 'severity']
)

rate_limit_violations = Counter(
    'rate_limit_violations_total',
    'Rate limit violations by endpoint and client',
    ['endpoint', 'client_type', 'tenant']
)

authentication_attempts = Counter(
    'auth_attempts_total',
    'Authentication attempts by method and result',
    ['method', 'result', 'tenant']
)

websocket_connections = Gauge(
    'websocket_connections_active',
    'Active WebSocket connections by tenant',
    ['tenant', 'authenticated']
)

api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration by endpoint and method',
    ['endpoint', 'method', 'tenant', 'status']
)

class MetricsCollector:
    """Enhanced metrics collection with context"""
    
    @staticmethod
    def record_security_event(
        event_type: str, 
        endpoint: str, 
        tenant: str = "default",
        severity: str = "medium"
    ):
        """Record security event with context"""
        security_events.labels(
            event_type=event_type,
            endpoint=endpoint,
            tenant=tenant,
            severity=severity
        ).inc()
    
    @staticmethod
    def record_rate_limit_violation(
        endpoint: str,
        client_type: str = "api",
        tenant: str = "default"
    ):
        """Record rate limit violation"""
        rate_limit_violations.labels(
            endpoint=endpoint,
            client_type=client_type,
            tenant=tenant
        ).inc()
    
    @staticmethod
    def record_auth_attempt(
        method: str,
        success: bool,
        tenant: str = "default"
    ):
        """Record authentication attempt"""
        result = "success" if success else "failure"
        authentication_attempts.labels(
            method=method,
            result=result,
            tenant=tenant
        ).inc()
    
    @staticmethod
    def track_request_duration(
        endpoint: str,
        method: str,
        tenant: str = "default"
    ):
        """Context manager for tracking request duration"""
        class RequestTimer:
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                status = "error" if exc_type else "success"
                api_request_duration.labels(
                    endpoint=endpoint,
                    method=method,
                    tenant=tenant,
                    status=status
                ).observe(duration)
        
        return RequestTimer()
