# File: backend/ai/ai_monitoring.py
"""
AI Service Monitoring Integration
=================================
Instrumentation for AI coordination engine and model interactions
"""

import time
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import functools

# Import monitoring system
try:
    from monitoring.csp_monitoring import get_default as get_monitoring
    from backend.monitoring.performance import ai_requests_total, ai_tokens_total
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    get_monitoring = lambda: None
    ai_requests_total = None
    ai_tokens_total = None

@dataclass
class AIMetrics:
    """AI operation metrics"""
    provider: str
    model: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    tokens_input: int = 0
    tokens_output: int = 0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class AIMonitor:
    """Monitor for AI operations"""
    
    def __init__(self):
        self.monitor = get_monitoring() if MONITORING_AVAILABLE else None
        self.active_requests: Dict[str, AIMetrics] = {}
        self.request_history = []
        self.max_history = 1000
    
    def start_request(self, request_id: str, provider: str, model: str, 
                     operation: str) -> AIMetrics:
        """Start tracking an AI request"""
        metrics = AIMetrics(
            provider=provider,
            model=model,
            operation=operation,
            start_time=time.time()
        )
        self.active_requests[request_id] = metrics
        return metrics
    
    def complete_request(self, request_id: str, tokens_input: int = 0,
                        tokens_output: int = 0, success: bool = True,
                        error: Optional[str] = None):
        """Complete tracking an AI request"""
        if request_id not in self.active_requests:
            return
        
        metrics = self.active_requests[request_id]
        metrics.end_time = time.time()
        metrics.tokens_input = tokens_input
        metrics.tokens_output = tokens_output
        metrics.success = success
        metrics.error = error
        
        # Record metrics
        if MONITORING_AVAILABLE and ai_requests_total:
            ai_requests_total.labels(
                provider=metrics.provider,
                model=metrics.model
            ).inc()
            
            if ai_tokens_total:
                ai_tokens_total.labels(
                    provider=metrics.provider,
                    model=metrics.model,
                    type="input"
                ).inc(tokens_input)
                
                ai_tokens_total.labels(
                    provider=metrics.provider,
                    model=metrics.model,
                    type="output"
                ).inc(tokens_output)
        
        # Move to history
        self.request_history.append(metrics)
        if len(self.request_history) > self.max_history:
            self.request_history.pop(0)
        
        del self.active_requests[request_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current AI usage statistics"""
        total_requests = len(self.request_history)
        if total_requests == 0:
            return {
                "total_requests": 0,
                "providers": {},
                "models": {},
                "error_rate": 0.0
            }
        
        # Calculate statistics
        providers = {}
        models = {}
        errors = 0
        total_tokens = 0
        
        for metrics in self.request_history:
            # Provider stats
            if metrics.provider not in providers:
                providers[metrics.provider] = {
                    "requests": 0,
                    "tokens": 0,
                    "errors": 0
                }
            providers[metrics.provider]["requests"] += 1
            providers[metrics.provider]["tokens"] += (
                metrics.tokens_input + metrics.tokens_output
            )
            
            # Model stats
            model_key = f"{metrics.provider}/{metrics.model}"
            if model_key not in models:
                models[model_key] = {
                    "requests": 0,
                    "tokens": 0,
                    "avg_latency": 0.0
                }
            models[model_key]["requests"] += 1
            models[model_key]["tokens"] += (
                metrics.tokens_input + metrics.tokens_output
            )
            
            if metrics.end_time and metrics.start_time:
                latency = metrics.end_time - metrics.start_time
                # Simple moving average
                prev_avg = models[model_key]["avg_latency"]
                count = models[model_key]["requests"]
                models[model_key]["avg_latency"] = (
                    (prev_avg * (count - 1) + latency) / count
                )
            
            # Error tracking
            if not metrics.success:
                errors += 1
                providers[metrics.provider]["errors"] += 1
            
            total_tokens += metrics.tokens_input + metrics.tokens_output
        
        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "providers": providers,
            "models": models,
            "error_rate": errors / total_requests,
            "active_requests": len(self.active_requests)
        }

# Global AI monitor instance
ai_monitor = AIMonitor()

def monitor_ai_operation(provider: str, model: str, operation: str = "completion"):
    """Decorator to monitor AI operations"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            import uuid
            request_id = str(uuid.uuid4())
            
            # Start monitoring
            ai_monitor.start_request(request_id, provider, model, operation)
            
            try:
                # Execute AI operation
                result = await func(*args, **kwargs)
                
                # Extract token counts if available
                tokens_input = 0
                tokens_output = 0
                
                if isinstance(result, dict):
                    usage = result.get('usage', {})
                    tokens_input = usage.get('prompt_tokens', 0)
                    tokens_output = usage.get('completion_tokens', 0)
                
                # Complete monitoring
                ai_monitor.complete_request(
                    request_id,
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    success=True
                )
                
                return result
                
            except Exception as e:
                # Record failure
                ai_monitor.complete_request(
                    request_id,
                    success=False,
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator

# Instrumented AI coordination functions
class InstrumentedAICoordinator:
    """AI Coordinator with built-in monitoring"""
    
    def __init__(self, base_coordinator):
        self.coordinator = base_coordinator
        self.monitor = get_monitoring() if MONITORING_AVAILABLE else None
    
    @monitor_ai_operation("openai", "gpt-4", "coordination")
    async def coordinate_agents(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate AI agents with monitoring"""
        return await self.coordinator.coordinate_agents(task)
    
    @monitor_ai_operation("openai", "gpt-3.5-turbo", "analysis")
    async def analyze_emergence(self, patterns: List[Dict]) -> Dict[str, Any]:
        """Analyze emergent patterns with monitoring"""
        return await self.coordinator.analyze_emergence(patterns)
    
    @monitor_ai_operation("anthropic", "claude-2", "synthesis")
    async def synthesize_knowledge(self, sources: List[str]) -> str:
        """Synthesize knowledge with monitoring"""
        return await self.coordinator.synthesize_knowledge(sources)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get AI coordination metrics"""
        base_stats = ai_monitor.get_stats()
        
        # Add coordination-specific metrics
        base_stats.update({
            "coordination_tasks": len(self.coordinator.active_tasks) if hasattr(self.coordinator, 'active_tasks') else 0,
            "agent_pool_size": len(self.coordinator.agent_pool) if hasattr(self.coordinator, 'agent_pool') else 0,
            "emergence_detections": self.coordinator.emergence_count if hasattr(self.coordinator, 'emergence_count') else 0
        })
        
        return base_stats

# Background task for periodic metric updates
async def ai_metrics_updater():
    """Periodically update AI metrics"""
    monitor = get_monitoring() if MONITORING_AVAILABLE else None
    if not monitor:
        return
    
    while True:
        try:
            stats = ai_monitor.get_stats()
            
            # Update custom metrics based on stats
            # This would integrate with your actual AI coordination engine
            
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            print(f"Error updating AI metrics: {e}")
            await asyncio.sleep(60)