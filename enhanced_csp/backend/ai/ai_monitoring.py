"""
AI service monitoring instrumentation
"""
import time
from typing import Dict, Any, Optional
from functools import wraps

try:
    from monitoring import get_default
    monitor = get_default()
    MONITORING_ENABLED = True
except ImportError:
    monitor = None
    MONITORING_ENABLED = False

def monitor_ai_request(provider: str, model: str):
    """Decorator to monitor AI service requests"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not MONITORING_ENABLED:
                return await func(*args, **kwargs)
            
            start_time = time.time()
            tokens_used = {"input": 0, "output": 0}
            
            try:
                # Call the AI function
                result = await func(*args, **kwargs)
                
                # Extract token usage if available
                if isinstance(result, dict):
                    if "usage" in result:
                        tokens_used["input"] = result["usage"].get("prompt_tokens", 0)
                        tokens_used["output"] = result["usage"].get("completion_tokens", 0)
                
                # Record metrics
                monitor.record_ai_request(provider, model, True)
                monitor.record_ai_tokens(provider, model, "input", tokens_used["input"])
                monitor.record_ai_tokens(provider, model, "output", tokens_used["output"])
                
                # Record latency
                latency = time.time() - start_time
                monitor.record_ai_latency(provider, model, latency)
                
                return result
                
            except Exception as e:
                monitor.record_ai_request(provider, model, False)
                raise
        
        return wrapper
    return decorator

class AIMetricsCollector:
    """Collect and aggregate AI metrics"""
    
    def __init__(self):
        self.request_count = 0
        self.token_count = 0
        self.total_latency = 0
        self.providers = {}
    
    def record_request(self, provider: str, model: str, tokens: int, latency: float):
        """Record an AI request"""
        self.request_count += 1
        self.token_count += tokens
        self.total_latency += latency
        
        # Track per-provider stats
        if provider not in self.providers:
            self.providers[provider] = {
                "requests": 0,
                "tokens": 0,
                "latency": 0
            }
        
        self.providers[provider]["requests"] += 1
        self.providers[provider]["tokens"] += tokens
        self.providers[provider]["latency"] += latency
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics"""
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "total_tokens": self.token_count,
            "average_latency": avg_latency,
            "providers": self.providers
        }
