# monitoring/csp_monitoring.py - Extended version with all features
"""
CSP Monitoring System - Complete Implementation
==============================================
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
# METRIC DEFINITIONS - COMPLETE SET
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

# Additional AI Metrics
ai_request_latency = Histogram(
    'csp_ai_request_latency_seconds',
    'AI request latency',
    ['provider', 'model'],
    registry=METRICS_REGISTRY
)

ai_request_success = Counter(
    'csp_ai_request_success_total',
    'Successful AI requests',
    ['provider', 'model'],
    registry=METRICS_REGISTRY
)

ai_request_errors = Counter(
    'csp_ai_request_errors_total',
    'Failed AI requests',
    ['provider', 'model', 'error_type'],
    registry=METRICS_REGISTRY
)

# CSP Engine Metrics
csp_process_lifetime = Histogram(
    'csp_process_lifetime_seconds',
    'CSP process lifetime',
    ['type'],
    buckets=(0.1, 0.5, 1, 5, 10, 30, 60, 300),
    registry=METRICS_REGISTRY
)

csp_channel_buffer_size = Gauge(
    'csp_channel_buffer_size',
    'Current channel buffer size',
    ['channel_id'],
    registry=METRICS_REGISTRY
)

csp_active_executions = Gauge(
    'csp_active_executions',
    'Number of active CSP executions',
    registry=METRICS_REGISTRY
)

# ============================================================================
# MONITORING SYSTEM IMPLEMENTATION
# ============================================================================

class CSPMonitoringSystem:
    """Main monitoring system for Enhanced CSP"""
    
    def __init__(self, config: Optional['MonitoringConfig'] = None):
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
        from monitoring.sli_slo import SLITracker
        self.sli_tracker = SLITracker()
        
        logger.info("CSP Monitoring System initialized")
    
    async def initialize(self):
        """Initialize monitoring system"""
        if self.initialized:
            return
        
        logger.info("Starting monitoring system initialization...")
        
        # Start collection tasks
        if self.config.enable_metrics:
            self.collection_tasks.append(
                asyncio.create_task(self._system_metrics_loop())
            )
            self.collection_tasks.append(
                asyncio.create_task(self._sli_calculation_loop())
            )
            
            if self.config.export_format and self.config.export_endpoint:
                self.collection_tasks.append(
                    asyncio.create_task(self._metric_export_loop())
                )
        
        self.initialized = True
        logger.info("Monitoring system initialized successfully")
    
    async def shutdown(self):
        """Shutdown monitoring system"""
        logger.info("Shutting down monitoring system...")
        
        # Cancel collection tasks
        for task in self.collection_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.collection_tasks, return_exceptions=True)
        
        # Push final metrics if configured
        if self.config.pushgateway_url:
            try:
                push_to_gateway(
                    self.config.pushgateway_url,
                    job='csp_monitoring_final',
                    registry=self.registry
                )
            except Exception as e:
                logger.error(f"Failed to push final metrics: {e}")
        
        self.initialized = False
        logger.info("Monitoring system shutdown complete")
    
    # ========================================================================
    # COLLECTION LOOPS
    # ========================================================================
    
    async def _system_metrics_loop(self):
        """Collect system metrics periodically"""
        while True:
            try:
                # Collect CPU, memory, disk metrics
                if psutil:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    
                    # Update gauges
                    if 'system_cpu_usage' in globals():
                        system_cpu_usage.set(cpu_percent)
                    if 'system_memory_usage' in globals():
                        system_memory_usage.set(memory.percent)
                    if 'system_disk_usage' in globals():
                        system_disk_usage.set(disk.percent)
                
                # Sleep for collection interval
                await asyncio.sleep(self.config.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(10)
    
    async def _sli_calculation_loop(self):
        """Calculate SLIs periodically"""
        while True:
            try:
                # Calculate and update SLIs
                compliance = self.sli_tracker.calculate_slo_compliance()
                
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
        metrics = {}
        
        # Collect all metric families from registry
        for metric in self.registry.collect():
            for sample in metric.samples:
                metrics[sample.name] = sample.value
        
        return metrics
    
    # ========================================================================
    # METRIC RECORDING METHODS
    # ========================================================================
    
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
    
    def update_db_pool_metrics(self, pool_name: str, size: int, active: int, waiting: int):
        """Update database pool metrics"""
        db_pool_size.labels(pool_name=pool_name).set(size)
        db_pool_active.labels(pool_name=pool_name).set(active)
        db_pool_waiting.labels(pool_name=pool_name).set(waiting)
    
    def update_execution_queue(self, priority: str, depth: int):
        """Update execution queue depth"""
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
    
    def record_process_created(self, process_type: str):
        """Record CSP process creation"""
        csp_processes_created.labels(type=process_type).inc()
    
    def record_channel_created(self, channel_type: str):
        """Record CSP channel creation"""
        csp_channels_created.labels(type=channel_type).inc()
    
    def record_message_exchanged(self, channel_type: str):
        """Record CSP message exchange"""
        csp_messages_exchanged.labels(channel_type=channel_type).inc()
    
    def update_sli(self, name: str, value: float):
        """Update SLI metric"""
        if name == "availability":
            sli_availability.set(value)
        elif name == "latency_p99":
            sli_latency_p99.set(value)
        elif name == "error_rate":
            sli_error_rate.set(value)
    
    def update_slo_compliance(self, slo_name: str, compliance: float):
        """Update SLO compliance"""
        slo_compliance.labels(slo_name=slo_name).set(compliance)
    
    def record_ai_request(self, provider: str, model: str, success: bool):
        """Record AI request"""
        if success:
            ai_request_success.labels(provider=provider, model=model).inc()
        else:
            ai_request_errors.labels(
                provider=provider,
                model=model,
                error_type="request_failed"
            ).inc()
    
    def record_ai_tokens(self, provider: str, model: str, token_type: str, count: int):
        """Record AI token usage"""
        if 'ai_tokens_total' in globals():
            ai_tokens_total.labels(
                provider=provider,
                model=model,
                type=token_type
            ).inc(count)
    
    def record_ai_latency(self, provider: str, model: str, latency: float):
        """Record AI request latency"""
        ai_request_latency.labels(provider=provider, model=model).observe(latency)
    
    def record_process_lifetime(self, process_type: str, lifetime: float):
        """Record CSP process lifetime"""
        csp_process_lifetime.labels(type=process_type).observe(lifetime)
    
    def update_channel_buffer_size(self, channel_id: str, size: int):
        """Update channel buffer size"""
        csp_channel_buffer_size.labels(channel_id=channel_id).set(size)
    
    def increment_active_executions(self):
        """Increment active executions counter"""
        csp_active_executions.inc()
    
    def decrement_active_executions(self):
        """Decrement active executions counter"""
        csp_active_executions.dec()
    
    def record_execution(self, execution_id: str, duration: float, status: str):
        """Record execution metrics"""
        if 'execution_duration' in globals():
            execution_duration.labels(
                design_id=execution_id,
                status=status
            ).observe(duration)
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry)
    
    def get_metrics_json(self) -> Dict[str, Any]:
        """Get metrics in JSON format"""
        return self._get_current_metrics()


# ============================================================================
# CONFIGURATION
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
# GLOBAL INSTANCE
# ============================================================================

_monitor_instance: Optional[CSPMonitoringSystem] = None

def get_default() -> CSPMonitoringSystem:
    """Get the default monitoring instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = CSPMonitoringSystem()
    return _monitor_instance