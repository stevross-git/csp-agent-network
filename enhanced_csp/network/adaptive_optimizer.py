# enhanced_csp/network/adaptive_optimizer.py
import statistics
import time
import asyncio
from collections import deque
from typing import Dict, Any, Deque, Callable, Optional

from .utils.structured_logging import get_logger

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = get_logger("optimizer")

class AdaptiveNetworkOptimizer:
    """Machine learning-based network optimization with robust error handling"""
    
    def __init__(self, 
                 update_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 create_new_compressor: bool = True):  # For channel isolation
        self.latency_history: Deque[float] = deque(maxlen=1000)
        self.throughput_history: Deque[float] = deque(maxlen=1000)
        self.packet_loss_history: Deque[float] = deque(maxlen=1000)
        self.error_count = 0
        self.request_count = 0
        
        # Each optimizer gets its own parameters (no shared state)
        self.optimization_params = {
            "batch_size": 50,
            "compression_algorithm": "lz4",
            "connection_pool_size": 20,
            "retry_strategy": "exponential",
            "max_retries": 3
        }
        
        self.update_callback = update_callback
        self.create_new_compressor = create_new_compressor
        self._running = False
        self._optimize_task: Optional[asyncio.Task] = None
        
        # CPU usage tracking
        self._last_cpu_check = 0
        self._cpu_percent = 0
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate current network metrics with error handling"""
        metrics = {
            "avg_latency": 0,
            "p99_latency": 0,
            "throughput": 0,
            "connection_error_rate": 0,
            "cpu_usage": 0,
        }
        
        # Latency metrics with empty list handling
        if self.latency_history:
            try:
                latencies = list(self.latency_history)
                metrics["avg_latency"] = statistics.mean(latencies)
                if len(latencies) >= 100:
                    metrics["p99_latency"] = sorted(latencies)[int(len(latencies) * 0.99)]
                else:
                    metrics["p99_latency"] = max(latencies) if latencies else 0
            except statistics.StatisticsError:
                # Empty sequence
                pass
        
        # Throughput with error handling
        if self.throughput_history:
            try:
                metrics["throughput"] = statistics.mean(self.throughput_history)
            except statistics.StatisticsError:
                pass
        
        # Error rate
        total_requests = self.request_count + self.error_count
        if total_requests > 0:
            metrics["connection_error_rate"] = self.error_count / total_requests
        
        # CPU usage
        metrics["cpu_usage"] = self._get_cpu_usage()
        
        return metrics