# File: backend/monitoring/performance.py
"""
Performance Monitoring & Metrics Collection
==========================================
Comprehensive monitoring system for the CSP Visual Designer backend
"""

import asyncio
import logging
import time
import threading
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import statistics

import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
import aioredis
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.database.connection import get_cache_manager, get_redis_client

logger = logging.getLogger(__name__)

# ============================================================================
# METRICS DEFINITIONS
# ============================================================================

class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    unit: str = ""
    help_text: str = ""

@dataclass
class Alert:
    """Performance alert"""
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class PerformanceSnapshot:
    """Point-in-time performance data"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    active_connections: int
    request_rate: float
    error_rate: float
    avg_response_time: float
    database_connections: int
    redis_connections: int
    active_executions: int

# ============================================================================
# PROMETHEUS METRICS SETUP
# ============================================================================

# Create custom registry
METRICS_REGISTRY = CollectorRegistry()

# HTTP Request metrics
http_requests_total = Counter(
    'csp_http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=METRICS_REGISTRY
)

http_request_duration = Histogram(
    'csp_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=METRICS_REGISTRY
)

http_request_size = Histogram(
    'csp_http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    registry=METRICS_REGISTRY
)

http_response_size = Histogram(
    'csp_http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint'],
    registry=METRICS_REGISTRY
)

# Database metrics
db_connections_active = Gauge(
    'csp_db_connections_active',
    'Number of active database connections',
    registry=METRICS_REGISTRY
)

db_query_duration = Histogram(
    'csp_db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation'],
    registry=METRICS_REGISTRY
)

db_transactions_total = Counter(
    'csp_db_transactions_total',
    'Total number of database transactions',
    ['status'],
    registry=METRICS_REGISTRY
)

# WebSocket metrics
websocket_connections_active = Gauge(
    'csp_websocket_connections_active',
    'Number of active WebSocket connections',
    registry=METRICS_REGISTRY
)

websocket_messages_total = Counter(
    'csp_websocket_messages_total',
    'Total number of WebSocket messages',
    ['type', 'direction'],
    registry=METRICS_REGISTRY
)

# Execution metrics
executions_active = Gauge(
    'csp_executions_active',
    'Number of active executions',
    registry=METRICS_REGISTRY
)

execution_duration = Histogram(
    'csp_execution_duration_seconds',
    'Execution duration in seconds',
    ['design_id', 'status'],
    registry=METRICS_REGISTRY
)

component_execution_duration = Histogram(
    'csp_component_execution_duration_seconds',
    'Component execution duration in seconds',
    ['component_type'],
    registry=METRICS_REGISTRY
)

# System metrics
system_cpu_usage = Gauge(
    'csp_system_cpu_percent',
    'System CPU usage percentage',
    registry=METRICS_REGISTRY
)

system_memory_usage = Gauge(
    'csp_system_memory_percent',
    'System memory usage percentage',
    registry=METRICS_REGISTRY
)

system_disk_usage = Gauge(
    'csp_system_disk_percent',
    'System disk usage percentage',
    registry=METRICS_REGISTRY
)

# AI metrics
ai_requests_total = Counter(
    'csp_ai_requests_total',
    'Total number of AI requests',
    ['provider', 'model'],
    registry=METRICS_REGISTRY
)

ai_tokens_total = Counter(
    'csp_ai_tokens_total',
    'Total number of AI tokens used',
    ['provider', 'model', 'type'],
    registry=METRICS_REGISTRY
)

ai_cost_total = Counter(
    'csp_ai_cost_total',
    'Total AI cost in USD',
    ['provider', 'model'],
    registry=METRICS_REGISTRY
)

# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis_client = redis_client
        self.is_running = False
        self.collection_interval = 15  # seconds
        self.retention_period = timedelta(hours=24)
        
        # In-memory metrics storage
        self.metrics_buffer: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(maxlen=2880)  # 12 hours at 15s intervals
        
        # Alert system
        self.alerts: Dict[str, Alert] = {}
        self.alert_thresholds = {
            "cpu_percent": {"warning": 80.0, "critical": 95.0},
            "memory_percent": {"warning": 85.0, "critical": 95.0},
            "disk_usage_percent": {"warning": 85.0, "critical": 95.0},
            "error_rate": {"warning": 0.05, "critical": 0.10},
            "avg_response_time": {"warning": 2.0, "critical": 5.0}
        }
        
        # Background tasks
        self.collection_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Request tracking
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.response_times: deque = deque(maxlen=1000)
        
        logger.info("Performance monitor initialized")
    
    async def start(self):
        """Start performance monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background collection tasks
        self.collection_task = asyncio.create_task(self._collect_metrics_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("✅ Performance monitoring started")
    
    async def stop(self):
        """Stop performance monitoring"""
        self.is_running = False
        
        # Cancel background tasks
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _collect_metrics_loop(self):
        """Background loop for collecting system metrics"""
        while self.is_running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _cleanup_loop(self):
        """Background loop for cleaning up old metrics"""
        while self.is_running:
            try:
                await self._cleanup_old_metrics()
                await asyncio.sleep(300)  # Clean up every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Update Prometheus metrics
            system_cpu_usage.set(cpu_percent)
            system_memory_usage.set(memory.percent)
            system_disk_usage.set(disk.percent)
            
            # WebSocket connections (would be updated by WebSocket manager)
            # Database connections (would be updated by database manager)
            
            # Calculate request rates
            current_time = time.time()
            recent_requests = sum(1 for t in self.response_times if current_time - t < 60)
            request_rate = recent_requests / 60.0
            
            # Calculate error rate
            total_requests = sum(self.request_counts.values())
            total_errors = sum(self.error_counts.values())
            error_rate = total_errors / max(total_requests, 1)
            
            # Calculate average response time
            recent_response_times = [rt for rt in self.response_times if current_time - rt < 300]
            avg_response_time = statistics.mean(recent_response_times) if recent_response_times else 0.0
            
            # Create performance snapshot
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_usage_percent=disk.percent,
                active_connections=0,  # Would be updated by connection managers
                request_rate=request_rate,
                error_rate=error_rate,
                avg_response_time=avg_response_time,
                database_connections=0,  # Would be updated by DB manager
                redis_connections=0,    # Would be updated by Redis manager
                active_executions=0     # Would be updated by execution engine
            )
            
            # Store snapshot
            self.performance_history.append(snapshot)
            
            # Store in Redis for persistence
            if self.redis_client:
                await self._store_metrics_in_redis(snapshot)
            
            # Check for alerts
            await self._check_alerts(snapshot)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _store_metrics_in_redis(self, snapshot: PerformanceSnapshot):
        """Store metrics in Redis for persistence"""
        try:
            key = f"performance_metrics:{int(snapshot.timestamp.timestamp())}"
            data = asdict(snapshot)
            data['timestamp'] = snapshot.timestamp.isoformat()
            
            await self.redis_client.setex(
                key,
                int(self.retention_period.total_seconds()),
                json.dumps(data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")
    
    async def _check_alerts(self, snapshot: PerformanceSnapshot):
        """Check metrics against alert thresholds"""
        metrics_to_check = {
            "cpu_percent": snapshot.cpu_percent,
            "memory_percent": snapshot.memory_percent,
            "disk_usage_percent": snapshot.disk_usage_percent,
            "error_rate": snapshot.error_rate,
            "avg_response_time": snapshot.avg_response_time
        }
        
        for metric_name, current_value in metrics_to_check.items():
            if metric_name not in self.alert_thresholds:
                continue
            
            thresholds = self.alert_thresholds[metric_name]
            alert_id = f"{metric_name}_alert"
            
            # Check for critical alerts
            if current_value >= thresholds.get("critical", float('inf')):
                if alert_id not in self.alerts or self.alerts[alert_id].resolved:
                    alert = Alert(
                        id=alert_id,
                        level=AlertLevel.CRITICAL,
                        message=f"Critical: {metric_name} is {current_value:.2f}",
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold=thresholds["critical"],
                        timestamp=datetime.now()
                    )
                    self.alerts[alert_id] = alert
                    await self._send_alert(alert)
            
            # Check for warning alerts
            elif current_value >= thresholds.get("warning", float('inf')):
                if alert_id not in self.alerts or self.alerts[alert_id].resolved:
                    alert = Alert(
                        id=alert_id,
                        level=AlertLevel.WARNING,
                        message=f"Warning: {metric_name} is {current_value:.2f}",
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold=thresholds["warning"],
                        timestamp=datetime.now()
                    )
                    self.alerts[alert_id] = alert
                    await self._send_alert(alert)
            
            # Resolve alerts if metrics are back to normal
            else:
                if alert_id in self.alerts and not self.alerts[alert_id].resolved:
                    self.alerts[alert_id].resolved = True
                    self.alerts[alert_id].resolved_at = datetime.now()
                    await self._resolve_alert(self.alerts[alert_id])
    
    async def _send_alert(self, alert: Alert):
        """Send alert notification"""
        logger.warning(f"ALERT [{alert.level.value.upper()}]: {alert.message}")
        
        # Store alert in Redis
        if self.redis_client:
            try:
                await self.redis_client.lpush(
                    "performance_alerts",
                    json.dumps(asdict(alert), default=str)
                )
                await self.redis_client.ltrim("performance_alerts", 0, 999)  # Keep last 1000 alerts
            except Exception as e:
                logger.error(f"Error storing alert in Redis: {e}")
    
    async def _resolve_alert(self, alert: Alert):
        """Handle alert resolution"""
        logger.info(f"ALERT RESOLVED: {alert.message}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics data"""
        try:
            cutoff_time = datetime.now() - self.retention_period
            
            # Clean up in-memory data
            while (self.performance_history and 
                   self.performance_history[0].timestamp < cutoff_time):
                self.performance_history.popleft()
            
            # Clean up Redis data
            if self.redis_client:
                cutoff_timestamp = int(cutoff_time.timestamp())
                pattern = "performance_metrics:*"
                
                async for key in self.redis_client.scan_iter(match=pattern):
                    timestamp_str = key.split(":")[-1]
                    try:
                        if int(timestamp_str) < cutoff_timestamp:
                            await self.redis_client.delete(key)
                    except ValueError:
                        continue
            
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")
    
    def record_request(self, method: str, endpoint: str, status_code: int, 
                      duration: float, request_size: int = 0, response_size: int = 0):
        """Record HTTP request metrics"""
        # Update Prometheus metrics
        http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        if request_size > 0:
            http_request_size.labels(
                method=method,
                endpoint=endpoint
            ).observe(request_size)
        
        if response_size > 0:
            http_response_size.labels(
                method=method,
                endpoint=endpoint
            ).observe(response_size)
        
        # Update internal tracking
        self.request_counts[f"{method}:{endpoint}"] += 1
        self.response_times.append(time.time())
        
        if status_code >= 400:
            self.error_counts[f"{method}:{endpoint}"] += 1
    
    def record_database_operation(self, operation: str, duration: float, success: bool):
        """Record database operation metrics"""
        db_query_duration.labels(operation=operation).observe(duration)
        db_transactions_total.labels(status="success" if success else "error").inc()
    
    def update_active_connections(self, websocket_count: int, db_count: int):
        """Update active connection counts"""
        websocket_connections_active.set(websocket_count)
        db_connections_active.set(db_count)
    
    def record_execution_metrics(self, design_id: str, duration: float, status: str):
        """Record execution metrics"""
        execution_duration.labels(design_id=design_id, status=status).observe(duration)
    
    def record_component_execution(self, component_type: str, duration: float):
        """Record component execution metrics"""
        component_execution_duration.labels(component_type=component_type).observe(duration)
    
    def record_ai_usage(self, provider: str, model: str, tokens: int, cost: float):
        """Record AI usage metrics"""
        ai_requests_total.labels(provider=provider, model=model).inc()
        ai_tokens_total.labels(provider=provider, model=model, type="total").inc(tokens)
        ai_cost_total.labels(provider=provider, model=model).inc(cost)
    
    async def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_snapshots = [
            snapshot for snapshot in self.performance_history
            if snapshot.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            return {"error": "No recent performance data available"}
        
        # Calculate statistics
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_percent for s in recent_snapshots]
        response_times = [s.avg_response_time for s in recent_snapshots]
        
        return {
            "period_hours": hours,
            "data_points": len(recent_snapshots),
            "cpu": {
                "current": recent_snapshots[-1].cpu_percent,
                "average": statistics.mean(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory": {
                "current": recent_snapshots[-1].memory_percent,
                "average": statistics.mean(memory_values),
                "max": max(memory_values),
                "min": min(memory_values)
            },
            "response_time": {
                "current": recent_snapshots[-1].avg_response_time,
                "average": statistics.mean(response_times),
                "max": max(response_times),
                "min": min(response_times)
            },
            "requests": {
                "rate": recent_snapshots[-1].request_rate,
                "error_rate": recent_snapshots[-1].error_rate
            },
            "active_alerts": len([a for a in self.alerts.values() if not a.resolved])
        }
    
    async def get_alerts(self, resolved: bool = False) -> List[Dict[str, Any]]:
        """Get current alerts"""
        filtered_alerts = [
            asdict(alert) for alert in self.alerts.values()
            if alert.resolved == resolved
        ]
        
        # Convert datetime objects to strings
        for alert in filtered_alerts:
            alert['timestamp'] = alert['timestamp'].isoformat()
            if alert['resolved_at']:
                alert['resolved_at'] = alert['resolved_at'].isoformat()
        
        return filtered_alerts
    
    async def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        from prometheus_client import generate_latest
        return generate_latest(METRICS_REGISTRY).decode('utf-8')

# ============================================================================
# FASTAPI MIDDLEWARE
# ============================================================================

class MetricsMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for collecting HTTP metrics"""
    
    def __init__(self, app, monitor: PerformanceMonitor):
        super().__init__(app)
        self.monitor = monitor
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics"""
        start_time = time.time()
        
        # Get request size
        request_size = 0
        if hasattr(request, 'body'):
            try:
                body = await request.body()
                request_size = len(body)
            except:
                pass
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Get response size
        response_size = 0
        if hasattr(response, 'body'):
            try:
                response_size = len(response.body)
            except:
                pass
        
        # Extract endpoint from path
        endpoint = request.url.path
        
        # Normalize endpoint (remove IDs)
        import re
        endpoint = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{id}', endpoint)
        endpoint = re.sub(r'/\d+', '/{id}', endpoint)
        
        # Record metrics
        self.monitor.record_request(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
            duration=duration,
            request_size=request_size,
            response_size=response_size
        )
        
        return response

# ============================================================================
# GLOBAL INSTANCE AND UTILITIES
# ============================================================================

# Global performance monitor instance
performance_monitor: Optional[PerformanceMonitor] = None

async def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    global performance_monitor
    
    if performance_monitor is None:
        try:
            redis_client = await get_redis_client()
            performance_monitor = PerformanceMonitor(redis_client)
        except:
            # Fallback without Redis
            performance_monitor = PerformanceMonitor()
        
        await performance_monitor.start()
        logger.info("✅ Performance monitor initialized")
    
    return performance_monitor

async def shutdown_performance_monitor():
    """Shutdown the performance monitor"""
    global performance_monitor
    
    if performance_monitor:
        await performance_monitor.stop()
        performance_monitor = None

# Decorator for monitoring function execution time
def monitor_execution_time(metric_name: str):
    """Decorator to monitor function execution time"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                monitor = await get_performance_monitor()
                # Would record specific metric based on metric_name
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                # Record error metrics
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                # Record metrics
                return result
            except Exception as e:
                duration = time.time() - start_time
                # Record error metrics
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Context manager for monitoring operations
class MonitoredOperation:
    """Context manager for monitoring operations"""
    
    def __init__(self, operation_name: str, labels: Dict[str, str] = None):
        self.operation_name = operation_name
        self.labels = labels or {}
        self.start_time = None
        self.duration = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        
        # Record metrics
        monitor = await get_performance_monitor()
        success = exc_type is None
        
        if self.operation_name.startswith("db_"):
            monitor.record_database_operation(
                self.operation_name.replace("db_", ""),
                self.duration,
                success
            )
        elif self.operation_name.startswith("component_"):
            component_type = self.labels.get("component_type", "unknown")
            monitor.record_component_execution(component_type, self.duration)
        
        return False  # Don't suppress exceptions

# Health check functions
async def get_system_health() -> Dict[str, Any]:
    """Get comprehensive system health status"""
    monitor = await get_performance_monitor()
    
    # Get recent performance data
    recent_summary = await monitor.get_performance_summary(hours=1)
    alerts = await monitor.get_alerts(resolved=False)
    
    # Determine overall health status
    health_status = "healthy"
    if len(alerts) > 0:
        critical_alerts = [a for a in alerts if a["level"] == "critical"]
        if critical_alerts:
            health_status = "critical"
        else:
            health_status = "warning"
    
    return {
        "status": health_status,
        "timestamp": datetime.now().isoformat(),
        "performance": recent_summary,
        "active_alerts": len(alerts),
        "alerts": alerts[:5],  # Show only first 5 alerts
        "uptime": "calculated_by_main_app",
        "version": "2.0.0"
    }

async def get_detailed_metrics() -> Dict[str, Any]:
    """Get detailed metrics for monitoring dashboards"""
    monitor = await get_performance_monitor()
    
    # Get performance history
    performance_data = []
    for snapshot in list(monitor.performance_history)[-100:]:  # Last 100 data points
        performance_data.append({
            "timestamp": snapshot.timestamp.isoformat(),
            "cpu_percent": snapshot.cpu_percent,
            "memory_percent": snapshot.memory_percent,
            "request_rate": snapshot.request_rate,
            "error_rate": snapshot.error_rate,
            "avg_response_time": snapshot.avg_response_time
        })
    
    # Get current metrics summary
    summary = await monitor.get_performance_summary(hours=1)
    
    return {
        "summary": summary,
        "performance_history": performance_data,
        "prometheus_metrics": await monitor.export_metrics()
    }
