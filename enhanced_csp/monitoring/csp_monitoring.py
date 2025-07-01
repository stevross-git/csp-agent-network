# File: monitoring/csp_monitoring.py
"""
CSP Monitoring System
====================
Comprehensive monitoring and observability for the Enhanced CSP System
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import psutil

from prometheus_client import (
    CollectorRegistry, Counter, Histogram, Gauge, Summary,
    generate_latest, push_to_gateway, CONTENT_TYPE_LATEST
)
from prometheus_client.exposition import basic_auth_handler

# Import backend performance monitor if available
try:
    from backend.monitoring.performance import (
        METRICS_REGISTRY, PerformanceMonitor, MetricType
    )
    BACKEND_MONITORING_AVAILABLE = True
except ImportError:
    BACKEND_MONITORING_AVAILABLE = False
    METRICS_REGISTRY = CollectorRegistry()

logger = logging.getLogger(__name__)

# ============================================================================
# MONITORING CONFIGURATION
# ============================================================================

@dataclass
class MonitoringConfig:
    """Monitoring system configuration"""
    enable_metrics: bool = True
    enable_tracing: bool = False
    enable_logging: bool = True
    enable_profiling: bool = False
    
    # Prometheus settings
    metrics_port: int = 9090
    pushgateway_url: Optional[str] = None
    scrape_interval: int = 15
    
    # Collection settings
    collection_interval: int = 10
    retention_hours: int = 168  # 7 days
    max_samples_per_metric: int = 10000
    
    # Alert settings
    enable_alerts: bool = True
    alert_manager_url: Optional[str] = None
    
    # Export settings
    export_format: str = "prometheus"  # prometheus, json, statsd
    export_endpoint: Optional[str] = None

# ============================================================================
# METRIC DEFINITIONS
# ============================================================================

# Authentication Metrics
auth_login_attempts = Counter(
    'csp_auth_login_attempts_total',
    'Total number of login attempts',
    ['method', 'status'],
    registry=METRICS_REGISTRY
)

auth_token_validations = Counter(
    'csp_auth_token_validations_total',
    'Total number of token validations',
    ['type', 'status'],
    registry=METRICS_REGISTRY
)

auth_active_sessions = Gauge(
    'csp_auth_active_sessions',
    'Number of active user sessions',
    ['auth_method'],
    registry=METRICS_REGISTRY
)

# File Upload Metrics
file_uploads_total = Counter(
    'csp_file_uploads_total',
    'Total number of file uploads',
    ['type', 'status'],
    registry=METRICS_REGISTRY
)

file_upload_size_bytes = Histogram(
    'csp_file_upload_size_bytes',
    'File upload size in bytes',
    ['type'],
    buckets=(1024, 10240, 102400, 1048576, 10485760, 104857600),  # 1KB to 100MB
    registry=METRICS_REGISTRY
)

file_processing_duration = Histogram(
    'csp_file_processing_duration_seconds',
    'File processing duration',
    ['type', 'operation'],
    registry=METRICS_REGISTRY
)

# Cache Metrics
cache_operations = Counter(
    'csp_cache_operations_total',
    'Total cache operations',
    ['operation', 'status'],
    registry=METRICS_REGISTRY
)

cache_hit_rate = Gauge(
    'csp_cache_hit_rate',
    'Cache hit rate percentage',
    registry=METRICS_REGISTRY
)

cache_memory_bytes = Gauge(
    'csp_cache_memory_bytes',
    'Cache memory usage in bytes',
    registry=METRICS_REGISTRY
)

# Rate Limiting Metrics
rate_limit_hits = Counter(
    'csp_rate_limit_hits_total',
    'Total rate limit hits',
    ['endpoint', 'limit_type'],
    registry=METRICS_REGISTRY
)

rate_limit_remaining = Gauge(
    'csp_rate_limit_remaining',
    'Remaining rate limit capacity',
    ['endpoint', 'client'],
    registry=METRICS_REGISTRY
)

# Connection Pool Metrics
db_pool_size = Gauge(
    'csp_db_pool_size',
    'Database connection pool size',
    ['pool_name'],
    registry=METRICS_REGISTRY
)

db_pool_active = Gauge(
    'csp_db_pool_active_connections',
    'Active connections in pool',
    ['pool_name'],
    registry=METRICS_REGISTRY
)

db_pool_waiting = Gauge(
    'csp_db_pool_waiting_requests',
    'Requests waiting for connection',
    ['pool_name'],
    registry=METRICS_REGISTRY
)

# Queue Metrics
execution_queue_depth = Gauge(
    'csp_execution_queue_depth',
    'Number of executions in queue',
    ['priority'],
    registry=METRICS_REGISTRY
)

execution_queue_latency = Histogram(
    'csp_execution_queue_latency_seconds',
    'Time spent in execution queue',
    ['priority'],
    registry=METRICS_REGISTRY
)

# Network Topology Metrics
network_peer_discovery_duration = Histogram(
    'csp_network_peer_discovery_duration_seconds',
    'Peer discovery duration',
    registry=METRICS_REGISTRY
)

network_routing_table_size = Gauge(
    'csp_network_routing_table_size',
    'Number of entries in routing table',
    registry=METRICS_REGISTRY
)

network_message_hops = Histogram(
    'csp_network_message_hops',
    'Number of hops for messages',
    buckets=(1, 2, 3, 4, 5, 10, 20),
    registry=METRICS_REGISTRY
)

# Business Metrics
csp_processes_created = Counter(
    'csp_processes_created_total',
    'Total CSP processes created',
    ['type'],
    registry=METRICS_REGISTRY
)

csp_channels_created = Counter(
    'csp_channels_created_total',
    'Total CSP channels created',
    ['type'],
    registry=METRICS_REGISTRY
)

csp_messages_exchanged = Counter(
    'csp_messages_exchanged_total',
    'Total messages exchanged',
    ['channel_type'],
    registry=METRICS_REGISTRY
)

# SLI/SLO Metrics
sli_availability = Gauge(
    'csp_sli_availability',
    'Service availability (0-1)',
    registry=METRICS_REGISTRY
)

sli_latency_p99 = Gauge(
    'csp_sli_latency_p99_seconds',
    '99th percentile latency',
    registry=METRICS_REGISTRY
)

sli_error_rate = Gauge(
    'csp_sli_error_rate',
    'Error rate (0-1)',
    registry=METRICS_REGISTRY
)

slo_compliance = Gauge(
    'csp_slo_compliance',
    'SLO compliance percentage',
    ['slo_name'],
    registry=METRICS_REGISTRY
)

# ============================================================================
# MONITORING SYSTEM
# ============================================================================

class CSPMonitoringSystem:
    """Main monitoring system for Enhanced CSP"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.registry = METRICS_REGISTRY
        self.initialized = False
        
        # Metric collectors
        self.collectors: Dict[str, Callable] = {}
        self.collection_tasks: List[asyncio.Task] = []
        
        # Performance monitor integration
        if BACKEND_MONITORING_AVAILABLE:
            self.performance_monitor = PerformanceMonitor()
        else:
            self.performance_monitor = None
        
        # Metric storage for aggregation
        self.metric_buffer = defaultdict(list)
        self.last_collection = time.time()
        
        # SLI/SLO tracking
        self.sli_tracker = SLITracker()
        
        logger.info("CSP Monitoring System initialized")
    
    async def initialize(self):
        """Initialize monitoring system"""
        if self.initialized:
            return
        
        logger.info("Starting CSP Monitoring System...")
        
        # Register default collectors
        self._register_default_collectors()
        
        # Start collection tasks
        if self.config.enable_metrics:
            self.collection_tasks.append(
                asyncio.create_task(self._metric_collection_loop())
            )
            self.collection_tasks.append(
                asyncio.create_task(self._system_metrics_loop())
            )
            self.collection_tasks.append(
                asyncio.create_task(self._sli_calculation_loop())
            )
        
        # Start metric export
        if self.config.export_endpoint:
            self.collection_tasks.append(
                asyncio.create_task(self._metric_export_loop())
            )
        
        self.initialized = True
        logger.info("CSP Monitoring System started successfully")
    
    async def shutdown(self):
        """Shutdown monitoring system"""
        logger.info("Shutting down CSP Monitoring System...")
        
        # Cancel all tasks
        for task in self.collection_tasks:
            task.cancel()
        
        if self.collection_tasks:
            await asyncio.gather(*self.collection_tasks, return_exceptions=True)
        
        self.initialized = False
        logger.info("CSP Monitoring System stopped")
    
    def _register_default_collectors(self):
        """Register default metric collectors"""
        # System metrics
        self.register_collector("system", self._collect_system_metrics)
        
        # Process metrics
        self.register_collector("process", self._collect_process_metrics)
        
        # Custom business metrics
        self.register_collector("business", self._collect_business_metrics)
    
    def register_collector(self, name: str, collector: Callable):
        """Register a metric collector"""
        self.collectors[name] = collector
        logger.debug(f"Registered metric collector: {name}")
    
    async def _metric_collection_loop(self):
        """Main metric collection loop"""
        while True:
            try:
                # Run all collectors
                for name, collector in self.collectors.items():
                    try:
                        await collector()
                    except Exception as e:
                        logger.error(f"Error in collector {name}: {e}")
                
                # Update collection timestamp
                self.last_collection = time.time()
                
                # Sleep until next collection
                await asyncio.sleep(self.config.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metric collection loop: {e}")
                await asyncio.sleep(5)
    
    async def _system_metrics_loop(self):
        """Collect system-level metrics"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                system_cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                system_memory_usage.set(memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                system_disk_usage.set(disk.percent)
                
                # Network I/O
                net_io = psutil.net_io_counters()
                if hasattr(self, '_last_net_io'):
                    bytes_sent = net_io.bytes_sent - self._last_net_io.bytes_sent
                    bytes_recv = net_io.bytes_recv - self._last_net_io.bytes_recv
                    # Could add network metrics here
                self._last_net_io = net_io
                
                await asyncio.sleep(self.config.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(5)
    
    async def _sli_calculation_loop(self):
        """Calculate SLI/SLO metrics"""
        while True:
            try:
                # Calculate SLIs
                await self.sli_tracker.calculate_slis()
                
                # Update SLI metrics
                sli_availability.set(self.sli_tracker.availability)
                sli_latency_p99.set(self.sli_tracker.latency_p99)
                sli_error_rate.set(self.sli_tracker.error_rate)
                
                # Check SLO compliance
                compliance = await self.sli_tracker.check_slo_compliance()
                for slo_name, compliant in compliance.items():
                    slo_compliance.labels(slo_name=slo_name).set(
                        100.0 if compliant else 0.0
                    )
                
                # Sleep for 1 minute
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error calculating SLIs: {e}")
                await asyncio.sleep(60)
    
    async def _metric_export_loop(self):
        """Export metrics to external systems"""
        while True:
            try:
                if self.config.export_format == "prometheus":
                    await self._export_to_prometheus()
                elif self.config.export_format == "json":
                    await self._export_to_json()
                
                # Sleep for scrape interval
                await asyncio.sleep(self.config.scrape_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error exporting metrics: {e}")
                await asyncio.sleep(30)
    
    async def _export_to_prometheus(self):
        """Export metrics to Prometheus Pushgateway"""
        if not self.config.pushgateway_url:
            return
        
        try:
            push_to_gateway(
                self.config.pushgateway_url,
                job='csp_monitoring',
                registry=self.registry
            )
        except Exception as e:
            logger.error(f"Failed to push metrics to Pushgateway: {e}")
    
    async def _export_to_json(self):
        """Export metrics in JSON format"""
        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self._get_current_metrics()
        }
        
        if self.config.export_endpoint:
            # Send to endpoint
            import aiohttp
            async with aiohttp.ClientSession() as session:
                await session.post(
                    self.config.export_endpoint,
                    json=metrics_data
                )
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        # This would parse the Prometheus registry
        # For now, return a placeholder
        return {
            "system_cpu": system_cpu_usage._value.get(),
            "system_memory": system_memory_usage._value.get(),
            "active_sessions": auth_active_sessions._value.get()
        }
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        # Already handled by _system_metrics_loop
        pass
    
    async def _collect_process_metrics(self):
        """Collect process-level metrics"""
        # This would integrate with process manager
        pass
    
    async def _collect_business_metrics(self):
        """Collect business-level metrics"""
        # This would integrate with CSP engine
        pass
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry)
    
    # Metric recording methods
    def record_auth_attempt(self, method: str, success: bool):
        """Record authentication attempt"""
        auth_login_attempts.labels(
            method=method,
            status="success" if success else "failure"
        ).inc()
    
    def record_token_validation(self, token_type: str, valid: bool):
        """Record token validation"""
        auth_token_validations.labels(
            type=token_type,
            status="valid" if valid else "invalid"
        ).inc()
    
    def update_active_sessions(self, auth_method: str, count: int):
        """Update active session count"""
        auth_active_sessions.labels(auth_method=auth_method).set(count)
    
    def record_file_upload(self, file_type: str, size: int, success: bool):
        """Record file upload"""
        file_uploads_total.labels(
            type=file_type,
            status="success" if success else "failure"
        ).inc()
        
        if success:
            file_upload_size_bytes.labels(type=file_type).observe(size)
    
    def record_file_processing(self, file_type: str, operation: str, duration: float):
        """Record file processing time"""
        file_processing_duration.labels(
            type=file_type,
            operation=operation
        ).observe(duration)
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation"""
        cache_operations.labels(
            operation=operation,
            status="hit" if hit else "miss"
        ).inc()
    
    def update_cache_metrics(self, hit_rate: float, memory_bytes: int):
        """Update cache metrics"""
        cache_hit_rate.set(hit_rate)
        cache_memory_bytes.set(memory_bytes)
    
    def record_rate_limit_hit(self, endpoint: str, limit_type: str):
        """Record rate limit hit"""
        rate_limit_hits.labels(
            endpoint=endpoint,
            limit_type=limit_type
        ).inc()
    
    def update_rate_limit_remaining(self, endpoint: str, client: str, remaining: int):
        """Update remaining rate limit"""
        rate_limit_remaining.labels(
            endpoint=endpoint,
            client=client
        ).set(remaining)
    
    def update_connection_pool_metrics(self, pool_name: str, size: int, 
                                     active: int, waiting: int):
        """Update connection pool metrics"""
        db_pool_size.labels(pool_name=pool_name).set(size)
        db_pool_active.labels(pool_name=pool_name).set(active)
        db_pool_waiting.labels(pool_name=pool_name).set(waiting)
    
    def update_execution_queue_metrics(self, priority: str, depth: int):
        """Update execution queue metrics"""
        execution_queue_depth.labels(priority=priority).set(depth)
    
    def record_queue_latency(self, priority: str, latency: float):
        """Record queue latency"""
        execution_queue_latency.labels(priority=priority).observe(latency)
    
    def record_peer_discovery(self, duration: float):
        """Record peer discovery duration"""
        network_peer_discovery_duration.observe(duration)
    
    def update_routing_table_size(self, size: int):
        """Update routing table size"""
        network_routing_table_size.set(size)
    
    def record_message_hops(self, hops: int):
        """Record message hop count"""
        network_message_hops.observe(hops)
    
    def record_csp_process_created(self, process_type: str):
        """Record CSP process creation"""
        csp_processes_created.labels(type=process_type).inc()
    
    def record_csp_channel_created(self, channel_type: str):
        """Record CSP channel creation"""
        csp_channels_created.labels(type=channel_type).inc()
    
    def record_csp_message(self, channel_type: str):
        """Record CSP message exchange"""
        csp_messages_exchanged.labels(channel_type=channel_type).inc()

# ============================================================================
# SLI/SLO TRACKER
# ============================================================================

class SLITracker:
    """Service Level Indicator tracker"""
    
    def __init__(self):
        self.availability = 1.0
        self.latency_p99 = 0.0
        self.error_rate = 0.0
        
        # SLO definitions
        self.slos = {
            "availability": {"target": 0.999, "window": "30d"},
            "latency": {"target": 0.2, "window": "5m"},  # 200ms
            "error_rate": {"target": 0.01, "window": "5m"}  # 1%
        }
    
    async def calculate_slis(self):
        """Calculate current SLI values"""
        # This would query actual metrics
        # For now, use placeholder calculations
        
        # Availability = successful requests / total requests
        # Would query actual metrics here
        self.availability = 0.9995
        
        # Latency P99 from histogram
        self.latency_p99 = 0.150  # 150ms
        
        # Error rate from counter
        self.error_rate = 0.005  # 0.5%
    
    async def check_slo_compliance(self) -> Dict[str, bool]:
        """Check SLO compliance"""
        return {
            "availability": self.availability >= self.slos["availability"]["target"],
            "latency": self.latency_p99 <= self.slos["latency"]["target"],
            "error_rate": self.error_rate <= self.slos["error_rate"]["target"]
        }

# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_monitoring_instance: Optional[CSPMonitoringSystem] = None

def get_default(config: Optional[MonitoringConfig] = None) -> CSPMonitoringSystem:
    """Get default monitoring system instance"""
    global _monitoring_instance
    
    if _monitoring_instance is None:
        _monitoring_instance = CSPMonitoringSystem(config)
    
    return _monitoring_instance

# ============================================================================
# MONITORING DECORATORS
# ============================================================================

def monitor_endpoint(endpoint_name: str):
    """Decorator to monitor API endpoints"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            monitor = get_default()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success metric
                if monitor.performance_monitor:
                    monitor.performance_monitor.record_request(
                        method=kwargs.get('request', {}).get('method', 'GET'),
                        endpoint=endpoint_name,
                        status_code=200,
                        duration=duration
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error metric
                if monitor.performance_monitor:
                    monitor.performance_monitor.record_request(
                        method=kwargs.get('request', {}).get('method', 'GET'),
                        endpoint=endpoint_name,
                        status_code=500,
                        duration=duration
                    )
                raise
        
        return wrapper
    return decorator

def monitor_execution(execution_type: str):
    """Decorator to monitor CSP executions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            monitor = get_default()
            
            try:
                # Record execution start
                monitor.record_csp_process_created(execution_type)
                
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success
                if monitor.performance_monitor:
                    monitor.performance_monitor.record_execution_metrics(
                        design_id=kwargs.get('design_id', 'unknown'),
                        duration=duration,
                        status='success'
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failure
                if monitor.performance_monitor:
                    monitor.performance_monitor.record_execution_metrics(
                        design_id=kwargs.get('design_id', 'unknown'),
                        duration=duration,
                        status='error'
                    )
                raise
        
        return wrapper
    return decorator

# ============================================================================
# METRICS ENDPOINT HANDLER
# ============================================================================

async def metrics_handler(request):
    """Handler for /metrics endpoint"""
    monitor = get_default()
    metrics_data = monitor.get_metrics()
    
    return {
        "body": metrics_data,
        "headers": {"Content-Type": CONTENT_TYPE_LATEST},
        "status_code": 200
    }
