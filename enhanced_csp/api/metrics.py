# enhanced_csp/api/metrics.py
"""
Metrics collection and export for Enhanced CSP
"""

import asyncio
import time
import os
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    CollectorRegistry, CONTENT_TYPE_LATEST,
    generate_latest, multiprocess, start_http_server
)

try:
    from aiohttp import web, ClientSession
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global metrics registry
REGISTRY = CollectorRegistry()

# Define metrics
MESSAGES_SENT = Counter(
    'csp_messages_sent_total',
    'Total number of messages sent',
    ['channel', 'pattern', 'status'],
    registry=REGISTRY
)

MESSAGES_RECEIVED = Counter(
    'csp_messages_received_total',
    'Total number of messages received',
    ['channel', 'pattern'],
    registry=REGISTRY
)

MESSAGE_SIZE = Histogram(
    'csp_message_size_bytes',
    'Size of messages in bytes',
    ['channel', 'pattern'],
    buckets=(100, 1000, 10000, 100000, 1000000),
    registry=REGISTRY
)

PROCESSING_TIME = Histogram(
    'csp_processing_time_seconds',
    'Time taken to process messages',
    ['agent', 'task_type'],
    registry=REGISTRY
)

ACTIVE_AGENTS = Gauge(
    'csp_active_agents',
    'Number of active agents',
    ['type'],
    registry=REGISTRY
)

ACTIVE_CHANNELS = Gauge(
    'csp_active_channels',
    'Number of active channels',
    ['pattern'],
    registry=REGISTRY
)

NETWORK_THROUGHPUT = Gauge(
    'csp_network_throughput_bytes_per_second',
    'Current network throughput',
    ['direction'],
    registry=REGISTRY
)

COMPRESSION_RATIO = Gauge(
    'csp_compression_ratio',
    'Current compression ratio',
    ['algorithm'],
    registry=REGISTRY
)

BATCH_SIZE = Gauge(
    'csp_batch_size_average',
    'Average batch size',
    ['channel'],
    registry=REGISTRY
)

CONNECTION_POOL_SIZE = Gauge(
    'csp_connection_pool_size',
    'Number of connections in pool',
    ['state'],
    registry=REGISTRY
)

SYSTEM_INFO = Info(
    'csp_system',
    'System information',
    registry=REGISTRY
)

@dataclass
class MetricsSample:
    """A single metrics sample"""
    timestamp: float
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """Collects metrics from various components"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        self.samples: List[MetricsSample] = []
        self._callbacks: List[Callable] = []
        
    def record_message_sent(self, channel: str, pattern: str, success: bool = True):
        """Record a message being sent"""
        status = "success" if success else "failure"
        MESSAGES_SENT.labels(channel=channel, pattern=pattern, status=status).inc()
        
    def record_message_received(self, channel: str, pattern: str):
        """Record a message being received"""
        MESSAGES_RECEIVED.labels(channel=channel, pattern=pattern).inc()
        
    def record_message_size(self, channel: str, pattern: str, size: int):
        """Record message size"""
        MESSAGE_SIZE.labels(channel=channel, pattern=pattern).observe(size)
        
    def record_processing_time(self, agent: str, task_type: str, duration: float):
        """Record processing time"""
        PROCESSING_TIME.labels(agent=agent, task_type=task_type).observe(duration)
        
    def set_active_agents(self, agent_type: str, count: int):
        """Set number of active agents"""
        ACTIVE_AGENTS.labels(type=agent_type).set(count)
        
    def set_active_channels(self, pattern: str, count: int):
        """Set number of active channels"""
        ACTIVE_CHANNELS.labels(pattern=pattern).set(count)
        
    def set_throughput(self, direction: str, bytes_per_sec: float):
        """Set current throughput"""
        NETWORK_THROUGHPUT.labels(direction=direction).set(bytes_per_sec)
        
    def set_compression_ratio(self, algorithm: str, ratio: float):
        """Set compression ratio"""
        COMPRESSION_RATIO.labels(algorithm=algorithm).set(ratio)
        
    def set_batch_size(self, channel: str, size: float):
        """Set average batch size"""
        BATCH_SIZE.labels(channel=channel).set(size)
        
    def set_connection_pool_size(self, state: str, count: int):
        """Set connection pool size"""
        CONNECTION_POOL_SIZE.labels(state=state).set(count)
        
    def set_system_info(self, **info):
        """Set system information"""
        SYSTEM_INFO.info(info)
        
    def add_sample(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Add a custom metric sample"""
        sample = MetricsSample(
            timestamp=time.time(),
            name=name,
            value=value,
            labels=labels or {}
        )
        self.samples.append(sample)
        
    def register_callback(self, callback: Callable):
        """Register a callback to be called during collection"""
        self._callbacks.append(callback)
        
    def collect(self):
        """Collect all metrics"""
        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")
                
        # Return registry metrics
        return generate_latest(self.registry)

class MetricsExporter:
    """Exports metrics in various formats"""
    
    def __init__(self, collector: Optional[MetricsCollector] = None):
        self.collector = collector or MetricsCollector()
        self._export_tasks = []
        
    async def start(self):
        """Start the exporter"""
        logger.info("MetricsExporter started")
        
    async def stop(self):
        """Stop the exporter"""
        for task in self._export_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info("MetricsExporter stopped")
        
    def export_prometheus(self) -> bytes:
        """Export metrics in Prometheus format"""
        return self.collector.collect()
        
    def export_json(self) -> Dict[str, Any]:
        """Export metrics in JSON format"""
        metrics = {}
        
        # Export samples
        for sample in self.collector.samples:
            key = sample.name
            if sample.labels:
                key += f"_{'.'.join(f'{k}={v}' for k, v in sample.labels.items())}"
            metrics[key] = {
                "value": sample.value,
                "timestamp": sample.timestamp,
                "labels": sample.labels
            }
            
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        }
        
    async def push_to_gateway(self, gateway_url: str, job: str = "csp_network"):
        """Push metrics to a Prometheus Pushgateway"""
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not available, cannot push to gateway")
            return
            
        try:
            import aiohttp
            
            data = self.export_prometheus()
            url = f"{gateway_url}/metrics/job/{job}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data,
                                      headers={'Content-Type': CONTENT_TYPE_LATEST}) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to push metrics: {resp.status}")
                    else:
                        logger.debug("Successfully pushed metrics to gateway")
                        
        except Exception as e:
            logger.error(f"Error pushing metrics: {e}")

# Global instances
_collector = MetricsCollector()
_exporter = MetricsExporter(_collector)

# Convenience functions
def get_collector() -> MetricsCollector:
    """Get the global metrics collector"""
    return _collector

def get_exporter() -> MetricsExporter:
    """Get the global metrics exporter"""
    return _exporter

async def start_metrics_server(port: int = 9090, host: str = "0.0.0.0"):
    """Start HTTP server for metrics endpoint"""
    if not AIOHTTP_AVAILABLE:
        # Fallback to prometheus_client's built-in server
        logger.info(f"Starting metrics server on {host}:{port} (using prometheus_client)")
        start_http_server(port, addr=host, registry=REGISTRY)
        return
    
    # Use aiohttp for async server
    app = web.Application()
    
    async def metrics_handler(request):
        """Handle metrics requests"""
        metrics_data = _collector.collect()
        return web.Response(
            body=metrics_data,
            content_type=CONTENT_TYPE_LATEST
        )
    
    async def health_handler(request):
        """Health check endpoint"""
        return web.Response(text="OK", status=200)
    
    app.router.add_get('/metrics', metrics_handler)
    app.router.add_get('/health', health_handler)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, host, port)
    await site.start()
    
    logger.info(f"Metrics server started on http://{host}:{port}/metrics")
    
    # Return runner so it can be cleaned up later
    return runner

def setup_multiprocess_mode():
    """Setup for multiprocess mode (e.g., gunicorn)"""
    multiprocess_dir = os.environ.get('PROMETHEUS_MULTIPROC_DIR')
    if multiprocess_dir:
        # Use multiprocess collector
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    return REGISTRY

# Initialize multiprocess mode if needed
if os.environ.get('PROMETHEUS_MULTIPROC_DIR'):
    REGISTRY = setup_multiprocess_mode()

# Export key functions and classes
__all__ = [
    'MetricsCollector',
    'MetricsExporter',
    'MetricsSample',
    'get_collector',
    'get_exporter',
    'start_metrics_server',
    'setup_multiprocess_mode',
    
    # Prometheus metrics
    'MESSAGES_SENT',
    'MESSAGES_RECEIVED',
    'MESSAGE_SIZE',
    'PROCESSING_TIME',
    'ACTIVE_AGENTS',
    'ACTIVE_CHANNELS',
    'NETWORK_THROUGHPUT',
    'COMPRESSION_RATIO',
    'BATCH_SIZE',
    'CONNECTION_POOL_SIZE',
    'SYSTEM_INFO'
]