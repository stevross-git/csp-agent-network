#!/bin/bash
# Enhanced CSP Monitoring - High Priority Implementations
# ========================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# 1. DISTRIBUTED TRACING WITH OPENTELEMETRY & JAEGER
# ============================================================================

log_info "Setting up Distributed Tracing..."

# Create tracing directory structure
mkdir -p monitoring/tracing/{config,dashboards}

# Create Jaeger configuration
cat > monitoring/tracing/docker-compose.tracing.yml << 'EOF'
version: '3.8'

services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: csp_jaeger
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - COLLECTOR_OTLP_ENABLED=true
      - SPAN_STORAGE_TYPE=badger
      - BADGER_EPHEMERAL=false
      - BADGER_DIRECTORY_VALUE=/badger/data
      - BADGER_DIRECTORY_KEY=/badger/key
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Collector HTTP
      - "14250:14250"  # Collector gRPC
      - "4317:4317"    # OTLP gRPC receiver
      - "4318:4318"    # OTLP HTTP receiver
    volumes:
      - jaeger_data:/badger
    networks:
      - scripts_csp-network
    restart: unless-stopped

  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    container_name: csp_otel_collector
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./tracing/config/otel-collector.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4319:4317"    # OTLP gRPC
      - "4320:4318"    # OTLP HTTP
      - "8888:8888"    # Prometheus metrics
    networks:
      - scripts_csp-network
    depends_on:
      - jaeger

networks:
  scripts_csp-network:
    external: true

volumes:
  jaeger_data:
    driver: local
EOF

# Create OpenTelemetry Collector configuration
cat > monitoring/tracing/config/otel-collector.yaml << 'EOF'
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 1s
    send_batch_size: 1024
  
  memory_limiter:
    check_interval: 1s
    limit_mib: 512
    spike_limit_mib: 128
  
  attributes:
    actions:
      - key: environment
        value: production
        action: upsert
      - key: service.namespace
        value: enhanced-csp
        action: upsert

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
  
  prometheus:
    endpoint: "0.0.0.0:8888"
    namespace: traces
    const_labels:
      environment: production

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch, attributes]
      exporters: [jaeger]
    
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus]
EOF

# Create Python OpenTelemetry instrumentation
cat > backend/monitoring/tracing.py << 'EOF'
"""
Distributed Tracing Implementation with OpenTelemetry
"""
import os
from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

from opentelemetry import trace, metrics, baggage, context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

# Configuration
OTEL_ENDPOINT = os.getenv("OTEL_ENDPOINT", "localhost:4317")
SERVICE_NAME = os.getenv("SERVICE_NAME", "csp-backend")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")

class DistributedTracing:
    """Manages distributed tracing for the CSP system"""
    
    def __init__(self):
        self.tracer = None
        self.meter = None
        self.propagator = TraceContextTextMapPropagator()
        self._initialize_tracing()
    
    def _initialize_tracing(self):
        """Initialize OpenTelemetry tracing"""
        # Create resource
        resource = Resource.create({
            "service.name": SERVICE_NAME,
            "service.version": SERVICE_VERSION,
            "deployment.environment": os.getenv("ENVIRONMENT", "development")
        })
        
        # Setup tracing
        trace_provider = TracerProvider(resource=resource)
        
        # Add OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=OTEL_ENDPOINT,
            insecure=True
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace_provider.add_span_processor(span_processor)
        
        trace.set_tracer_provider(trace_provider)
        self.tracer = trace.get_tracer(__name__, SERVICE_VERSION)
        
        # Setup metrics
        metric_reader = PeriodicExportingMetricReader(
            exporter=OTLPMetricExporter(
                endpoint=OTEL_ENDPOINT,
                insecure=True
            ),
            export_interval_millis=10000
        )
        
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(meter_provider)
        self.meter = metrics.get_meter(__name__, SERVICE_VERSION)
        
        logger.info(f"Distributed tracing initialized - endpoint: {OTEL_ENDPOINT}")
    
    def instrument_app(self, app):
        """Instrument FastAPI application"""
        FastAPIInstrumentor.instrument_app(app)
        
        # Instrument other libraries
        RequestsInstrumentor().instrument()
        RedisInstrumentor().instrument()
        Psycopg2Instrumentor().instrument()
        
        # Add custom middleware for trace context
        @app.middleware("http")
        async def trace_context_middleware(request, call_next):
            # Extract trace context from headers
            carrier = dict(request.headers)
            ctx = self.propagator.extract(carrier=carrier)
            token = context.attach(ctx)
            
            try:
                response = await call_next(request)
                return response
            finally:
                context.detach(token)
    
    @contextmanager
    def trace_operation(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations"""
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                span.set_attributes(attributes)
            
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def trace_function(self, name: Optional[str] = None):
        """Decorator for tracing functions"""
        def decorator(func):
            span_name = name or f"{func.__module__}.{func.__name__}"
            
            async def async_wrapper(*args, **kwargs):
                with self.trace_operation(span_name):
                    return await func(*args, **kwargs)
            
            def sync_wrapper(*args, **kwargs):
                with self.trace_operation(span_name):
                    return func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def add_span_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to current span"""
        span = trace.get_current_span()
        if span:
            span.add_event(name, attributes=attributes or {})
    
    def set_span_attributes(self, attributes: Dict[str, Any]):
        """Set attributes on current span"""
        span = trace.get_current_span()
        if span:
            span.set_attributes(attributes)
    
    def create_child_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a child span"""
        return self.trace_operation(name, attributes)

# Global instance
_tracing = None

def get_tracing() -> DistributedTracing:
    """Get or create tracing instance"""
    global _tracing
    if _tracing is None:
        _tracing = DistributedTracing()
    return _tracing

# Convenience decorators
def trace_endpoint(name: Optional[str] = None):
    """Decorator for tracing API endpoints"""
    def decorator(func):
        tracing = get_tracing()
        span_name = name or f"endpoint.{func.__name__}"
        
        async def wrapper(*args, **kwargs):
            with tracing.trace_operation(span_name):
                # Add request metadata
                tracing.set_span_attributes({
                    "http.method": kwargs.get("request", {}).get("method", ""),
                    "http.path": kwargs.get("request", {}).get("path", ""),
                    "user.id": kwargs.get("current_user", {}).get("id", "anonymous")
                })
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def trace_database_operation(operation: str):
    """Decorator for tracing database operations"""
    def decorator(func):
        tracing = get_tracing()
        
        async def wrapper(*args, **kwargs):
            with tracing.trace_operation(f"db.{operation}") as span:
                span.set_attributes({
                    "db.operation": operation,
                    "db.system": "postgresql"
                })
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator
EOF

# Update Prometheus configuration to include Jaeger metrics
cat >> monitoring/prometheus/prometheus-final.yml << 'EOF'

  # OpenTelemetry Collector metrics
  - job_name: 'otel-collector'
    static_configs:
      - targets: ['otel-collector:8888']
EOF

log_success "Distributed Tracing setup complete!"

# ============================================================================
# 2. ANOMALY DETECTION WITH MACHINE LEARNING
# ============================================================================

log_info "Setting up Anomaly Detection..."

# Create anomaly detection module
mkdir -p monitoring/anomaly_detection

cat > monitoring/anomaly_detection/detector.py << 'EOF'
"""
Anomaly Detection System using Machine Learning
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import pickle
import os

from prometheus_client import Gauge, Counter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

logger = logging.getLogger(__name__)

# Metrics for anomaly detection
anomaly_score = Gauge(
    'csp_anomaly_score',
    'Anomaly detection score (0-1)',
    ['service', 'metric_type', 'algorithm']
)

anomalies_detected = Counter(
    'csp_anomalies_detected_total',
    'Total anomalies detected',
    ['service', 'metric_type', 'severity']
)

anomaly_detection_duration = Gauge(
    'csp_anomaly_detection_duration_seconds',
    'Time taken for anomaly detection'
)

class AnomalyDetector:
    """ML-based anomaly detection for monitoring metrics"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_dir = "monitoring/anomaly_detection/models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Configuration
        self.config = {
            'contamination': 0.05,  # Expected proportion of anomalies
            'n_estimators': 100,
            'max_samples': 'auto',
            'random_state': 42
        }
        
        # Metrics to monitor
        self.monitored_metrics = [
            {
                'name': 'api_request_duration_seconds',
                'features': ['p50', 'p95', 'p99', 'rate'],
                'threshold': 0.8
            },
            {
                'name': 'error_rate',
                'features': ['rate', 'delta'],
                'threshold': 0.9
            },
            {
                'name': 'cpu_usage',
                'features': ['avg', 'max', 'std'],
                'threshold': 0.85
            },
            {
                'name': 'memory_usage',
                'features': ['current', 'rate'],
                'threshold': 0.85
            },
            {
                'name': 'database_connections',
                'features': ['active', 'waiting', 'rate'],
                'threshold': 0.9
            }
        ]
    
    async def initialize(self):
        """Initialize anomaly detection models"""
        logger.info("Initializing anomaly detection models...")
        
        # Load existing models or create new ones
        for metric in self.monitored_metrics:
            model_path = os.path.join(self.model_dir, f"{metric['name']}.pkl")
            scaler_path = os.path.join(self.model_dir, f"{metric['name']}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[metric['name']] = joblib.load(model_path)
                self.scalers[metric['name']] = joblib.load(scaler_path)
                logger.info(f"Loaded existing model for {metric['name']}")
            else:
                self.models[metric['name']] = IsolationForest(**self.config)
                self.scalers[metric['name']] = StandardScaler()
                logger.info(f"Created new model for {metric['name']}")
    
    async def fetch_metric_data(self, metric_name: str, duration: str = "1h") -> pd.DataFrame:
        """Fetch metric data from Prometheus"""
        import aiohttp
        
        queries = {
            'api_request_duration_seconds': [
                f'histogram_quantile(0.5, rate({metric_name}_bucket[5m]))',
                f'histogram_quantile(0.95, rate({metric_name}_bucket[5m]))',
                f'histogram_quantile(0.99, rate({metric_name}_bucket[5m]))',
                f'rate({metric_name}_count[5m])'
            ],
            'error_rate': [
                'rate(http_requests_total{status=~"5.."}[5m])',
                'delta(http_requests_total{status=~"5.."}[5m])'
            ],
            'cpu_usage': [
                'avg(rate(process_cpu_seconds_total[5m]))',
                'max(rate(process_cpu_seconds_total[5m]))',
                'stddev(rate(process_cpu_seconds_total[5m]))'
            ],
            'memory_usage': [
                'process_resident_memory_bytes',
                'rate(process_resident_memory_bytes[5m])'
            ],
            'database_connections': [
                'pg_stat_activity_count{state="active"}',
                'pg_stat_activity_count{state="waiting"}',
                'rate(pg_stat_activity_count[5m])'
            ]
        }
        
        if metric_name not in queries:
            return pd.DataFrame()
        
        data = []
        async with aiohttp.ClientSession() as session:
            for query in queries[metric_name]:
                url = f"{self.prometheus_url}/api/v1/query_range"
                params = {
                    'query': query,
                    'start': (datetime.now() - timedelta(hours=1)).timestamp(),
                    'end': datetime.now().timestamp(),
                    'step': '15s'
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result['data']['result']:
                            values = result['data']['result'][0]['values']
                            data.append([float(v[1]) for v in values])
        
        if data:
            return pd.DataFrame(data).T
        return pd.DataFrame()
    
    async def detect_anomalies(self, metric_name: str) -> Dict[str, Any]:
        """Detect anomalies in a specific metric"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Fetch recent data
            data = await self.fetch_metric_data(metric_name)
            
            if data.empty or len(data) < 10:
                logger.warning(f"Insufficient data for {metric_name}")
                return {'status': 'insufficient_data'}
            
            # Prepare features
            X = data.values
            
            # Scale features
            if metric_name not in self.scalers:
                self.scalers[metric_name] = StandardScaler()
                X_scaled = self.scalers[metric_name].fit_transform(X)
            else:
                X_scaled = self.scalers[metric_name].transform(X)
            
            # Detect anomalies
            if metric_name not in self.models:
                self.models[metric_name] = IsolationForest(**self.config)
                predictions = self.models[metric_name].fit_predict(X_scaled)
            else:
                predictions = self.models[metric_name].predict(X_scaled)
            
            # Calculate anomaly scores
            scores = self.models[metric_name].score_samples(X_scaled)
            anomaly_score_normalized = 1 - (scores - scores.min()) / (scores.max() - scores.min())
            
            # Find anomalies
            anomaly_indices = np.where(predictions == -1)[0]
            anomaly_timestamps = [
                datetime.now() - timedelta(seconds=15 * (len(predictions) - i))
                for i in anomaly_indices
            ]
            
            # Calculate metrics
            current_score = float(anomaly_score_normalized[-1]) if len(anomaly_score_normalized) > 0 else 0
            anomaly_rate = len(anomaly_indices) / len(predictions)
            
            # Update Prometheus metrics
            anomaly_score.labels(
                service="api",
                metric_type=metric_name,
                algorithm="isolation_forest"
            ).set(current_score)
            
            # Determine severity
            metric_config = next(m for m in self.monitored_metrics if m['name'] == metric_name)
            severity = self._determine_severity(current_score, metric_config['threshold'])
            
            if anomaly_indices.size > 0:
                anomalies_detected.labels(
                    service="api",
                    metric_type=metric_name,
                    severity=severity
                ).inc()
            
            result = {
                'status': 'success',
                'metric': metric_name,
                'current_score': current_score,
                'anomaly_rate': anomaly_rate,
                'anomalies': [
                    {
                        'timestamp': ts.isoformat(),
                        'score': float(anomaly_score_normalized[idx]),
                        'severity': severity
                    }
                    for idx, ts in zip(anomaly_indices, anomaly_timestamps)
                ],
                'model_info': {
                    'contamination': self.config['contamination'],
                    'samples_analyzed': len(predictions)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for {metric_name}: {str(e)}")
            return {'status': 'error', 'error': str(e)}
        
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            anomaly_detection_duration.set(duration)
    
    def _determine_severity(self, score: float, threshold: float) -> str:
        """Determine anomaly severity based on score"""
        if score >= threshold * 1.2:
            return "critical"
        elif score >= threshold:
            return "high"
        elif score >= threshold * 0.8:
            return "medium"
        else:
            return "low"
    
    async def train_models(self):
        """Train models on historical data"""
        logger.info("Training anomaly detection models...")
        
        for metric in self.monitored_metrics:
            try:
                # Fetch training data (last 24 hours)
                data = await self.fetch_metric_data(metric['name'], duration="24h")
                
                if data.empty or len(data) < 100:
                    logger.warning(f"Insufficient training data for {metric['name']}")
                    continue
                
                # Prepare and scale features
                X = data.values
                X_scaled = self.scalers[metric['name']].fit_transform(X)
                
                # Train model
                self.models[metric['name']].fit(X_scaled)
                
                # Save model and scaler
                model_path = os.path.join(self.model_dir, f"{metric['name']}.pkl")
                scaler_path = os.path.join(self.model_dir, f"{metric['name']}_scaler.pkl")
                
                joblib.dump(self.models[metric['name']], model_path)
                joblib.dump(self.scalers[metric['name']], scaler_path)
                
                logger.info(f"Trained and saved model for {metric['name']}")
                
            except Exception as e:
                logger.error(f"Error training model for {metric['name']}: {str(e)}")
    
    async def run_continuous_detection(self, interval: int = 60):
        """Run continuous anomaly detection"""
        logger.info(f"Starting continuous anomaly detection (interval: {interval}s)")
        
        while True:
            try:
                for metric in self.monitored_metrics:
                    result = await self.detect_anomalies(metric['name'])
                    
                    if result['status'] == 'success' and result['anomalies']:
                        logger.warning(
                            f"Anomalies detected in {metric['name']}: "
                            f"{len(result['anomalies'])} anomalies, "
                            f"current score: {result['current_score']:.3f}"
                        )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in continuous detection: {str(e)}")
                await asyncio.sleep(interval)

# Create service runner
async def run_anomaly_detection():
    """Run anomaly detection service"""
    detector = AnomalyDetector()
    await detector.initialize()
    
    # Train models initially
    await detector.train_models()
    
    # Run continuous detection
    await detector.run_continuous_detection()

if __name__ == "__main__":
    asyncio.run(run_anomaly_detection())
EOF

# Create anomaly detection service
cat > monitoring/anomaly_detection/docker-compose.anomaly.yml << 'EOF'
version: '3.8'

services:
  anomaly-detector:
    build:
      context: .
      dockerfile: Dockerfile.anomaly
    container_name: csp_anomaly_detector
    environment:
      - PROMETHEUS_URL=http://prometheus:9090
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./config:/app/config
    networks:
      - scripts_csp-network
    restart: unless-stopped
    depends_on:
      - prometheus

networks:
  scripts_csp-network:
    external: true
EOF

# Create Dockerfile for anomaly detection
cat > monitoring/anomaly_detection/Dockerfile.anomaly << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY detector.py .
COPY models/ ./models/

# Run the service
CMD ["python", "-u", "detector.py"]
EOF

# Create requirements file
cat > monitoring/anomaly_detection/requirements.txt << 'EOF'
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.1
prometheus-client==0.17.1
aiohttp==3.8.5
asyncio==3.4.3
EOF

log_success "Anomaly Detection setup complete!"

# ============================================================================
# 3. SECURITY MONITORING AND THREAT DETECTION
# ============================================================================

log_info "Setting up Security Monitoring..."

mkdir -p monitoring/security/{rules,patterns}

# Create security monitoring module
cat > monitoring/security/security_monitor.py << 'EOF'
"""
Security Monitoring and Threat Detection System
"""
import asyncio
import re
import json
from typing import Dict, List, Pattern, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import hashlib

from prometheus_client import Counter, Gauge, Histogram
import aioredis

logger = logging.getLogger(__name__)

# Security metrics
security_events = Counter(
    'csp_security_events_total',
    'Total security events detected',
    ['event_type', 'severity', 'source', 'action']
)

threat_score = Gauge(
    'csp_threat_score',
    'Current threat score (0-100)',
    ['category']
)

blocked_requests = Counter(
    'csp_blocked_requests_total',
    'Total requests blocked',
    ['reason', 'source_ip']
)

security_scan_duration = Histogram(
    'csp_security_scan_duration_seconds',
    'Time taken for security scanning',
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
)

class ThreatPattern:
    """Represents a threat detection pattern"""
    
    def __init__(self, name: str, pattern: str, severity: str, category: str):
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.severity = severity
        self.category = category
        self.matches = 0

class SecurityMonitor:
    """Advanced security monitoring and threat detection"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        
        # Threat patterns
        self.threat_patterns = self._load_threat_patterns()
        
        # IP reputation cache
        self.ip_reputation_cache = {}
        
        # Rate limiting
        self.rate_limits = {
            'api_calls': {'window': 60, 'limit': 100},
            'auth_attempts': {'window': 300, 'limit': 5},
            'file_uploads': {'window': 3600, 'limit': 50}
        }
        
        # Threat scoring weights
        self.threat_weights = {
            'sql_injection': 25,
            'xss': 20,
            'path_traversal': 20,
            'command_injection': 30,
            'auth_bypass': 25,
            'brute_force': 15,
            'dos_attack': 20,
            'data_exfiltration': 30
        }
    
    def _load_threat_patterns(self) -> List[ThreatPattern]:
        """Load threat detection patterns"""
        patterns = [
            # SQL Injection
            ThreatPattern(
                "sql_injection_union",
                r"(union|select|insert|update|delete|drop|create|alter|exec|execute).*?(from|where|table|database)",
                "high",
                "sql_injection"
            ),
            ThreatPattern(
                "sql_injection_comment",
                r"(--|#|\/\*|\*\/|@@|@)",
                "medium",
                "sql_injection"
            ),
            
            # XSS
            ThreatPattern(
                "xss_script_tag",
                r"<script[^>]*>.*?</script>",
                "high",
                "xss"
            ),
            ThreatPattern(
                "xss_event_handler",
                r"(onclick|onerror|onload|onmouseover|onfocus|onblur)=",
                "high",
                "xss"
            ),
            
            # Path Traversal
            ThreatPattern(
                "path_traversal",
                r"(\.\./|\.\.\\|%2e%2e%2f|%252e%252e%252f)",
                "high",
                "path_traversal"
            ),
            
            # Command Injection
            ThreatPattern(
                "command_injection",
                r"(;|\||&|`|\$\(|<\(|>\(|\$\{)",
                "critical",
                "command_injection"
            ),
            
            # Authentication Bypass
            ThreatPattern(
                "auth_bypass_null",
                r"(admin'--|' or '1'='1|' or 1=1--|\" or \"1\"=\"1)",
                "critical",
                "auth_bypass"
            ),
            
            # Suspicious User Agents
            ThreatPattern(
                "scanner_bot",
                r"(nikto|sqlmap|nmap|masscan|zap|burp|acunetix)",
                "medium",
                "scanner"
            ),
            
            # Data Exfiltration
            ThreatPattern(
                "base64_exfil",
                r"[A-Za-z0-9+/]{50,}={0,2}",
                "medium",
                "data_exfiltration"
            )
        ]
        
        return patterns
    
    async def initialize(self):
        """Initialize security monitor"""
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        logger.info("Security monitor initialized")
    
    async def scan_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scan a request for security threats"""
        start_time = asyncio.get_event_loop().time()
        threats_found = []
        
        try:
            # Extract data to scan
            scan_targets = {
                'path': request_data.get('path', ''),
                'query_params': json.dumps(request_data.get('query_params', {})),
                'body': json.dumps(request_data.get('body', {})),
                'headers': json.dumps(request_data.get('headers', {})),
                'user_agent': request_data.get('headers', {}).get('user-agent', '')
            }
            
            # Scan for threat patterns
            for target_name, target_value in scan_targets.items():
                for pattern in self.threat_patterns:
                    if pattern.pattern.search(str(target_value)):
                        threat = {
                            'pattern': pattern.name,
                            'category': pattern.category,
                            'severity': pattern.severity,
                            'location': target_name,
                            'matched_value': target_value[:100]  # Truncate for logging
                        }
                        threats_found.append(threat)
                        pattern.matches += 1
                        
                        # Log security event
                        security_events.labels(
                            event_type=pattern.category,
                            severity=pattern.severity,
                            source=request_data.get('source_ip', 'unknown'),
                            action='detected'
                        ).inc()
            
            # Check rate limits
            rate_limit_violations = await self._check_rate_limits(request_data)
            if rate_limit_violations:
                threats_found.extend(rate_limit_violations)
            
            # Check IP reputation
            ip_reputation = await self._check_ip_reputation(
                request_data.get('source_ip', '')
            )
            if ip_reputation and ip_reputation['risk_score'] > 0.7:
                threats_found.append({
                    'pattern': 'malicious_ip',
                    'category': 'reputation',
                    'severity': 'high',
                    'location': 'source_ip',
                    'risk_score': ip_reputation['risk_score']
                })
            
            # Calculate threat score
            total_score = self._calculate_threat_score(threats_found)
            
            # Update metrics
            threat_score.labels(category='overall').set(total_score)
            
            # Determine action
            action = 'allow'
            if total_score >= 70:
                action = 'block'
                blocked_requests.labels(
                    reason='high_threat_score',
                    source_ip=request_data.get('source_ip', 'unknown')
                ).inc()
            elif total_score >= 40:
                action = 'challenge'
            
            result = {
                'action': action,
                'threat_score': total_score,
                'threats': threats_found,
                'scan_time': asyncio.get_event_loop().time() - start_time
            }
            
            # Log high-risk requests
            if total_score >= 40:
                logger.warning(f"High-risk request detected: {result}")
            
            return result
            
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            security_scan_duration.observe(duration)
    
    async def _check_rate_limits(self, request_data: Dict[str, Any]) -> List[Dict]:
        """Check rate limits for the request"""
        violations = []
        source_ip = request_data.get('source_ip', 'unknown')
        
        for limit_type, config in self.rate_limits.items():
            key = f"rate_limit:{limit_type}:{source_ip}"
            
            # Increment counter
            current = await self.redis.incr(key)
            
            # Set expiry on first increment
            if current == 1:
                await self.redis.expire(key, config['window'])
            
            # Check if limit exceeded
            if current > config['limit']:
                violations.append({
                    'pattern': f'rate_limit_{limit_type}',
                    'category': 'dos_attack',
                    'severity': 'medium',
                    'location': 'rate_limit',
                    'current_rate': current,
                    'limit': config['limit']
                })
                
                security_events.labels(
                    event_type='rate_limit_exceeded',
                    severity='medium',
                    source=source_ip,
                    action='detected'
                ).inc()
        
        return violations
    
    async def _check_ip_reputation(self, ip: str) -> Optional[Dict]:
        """Check IP reputation"""
        if not ip or ip in self.ip_reputation_cache:
            return self.ip_reputation_cache.get(ip)
        
        # Check against known bad IPs (in production, use threat intelligence feeds)
        bad_ip_patterns = [
            r"^10\.0\.0\.",  # Example: internal testing
            r"^192\.168\.",  # Example: local network
        ]
        
        risk_score = 0.0
        reasons = []
        
        for pattern in bad_ip_patterns:
            if re.match(pattern, ip):
                risk_score += 0.5
                reasons.append(f"Matches pattern: {pattern}")
        
        # Check failed auth attempts
        auth_fails_key = f"auth_fails:{ip}"
        auth_fails = await self.redis.get(auth_fails_key)
        if auth_fails and int(auth_fails) > 5:
            risk_score += 0.3
            reasons.append(f"High auth failures: {auth_fails}")
        
        reputation = {
            'ip': ip,
            'risk_score': min(risk_score, 1.0),
            'reasons': reasons,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Cache for 1 hour
        self.ip_reputation_cache[ip] = reputation
        
        return reputation
    
    def _calculate_threat_score(self, threats: List[Dict]) -> float:
        """Calculate overall threat score"""
        if not threats:
            return 0.0
        
        score = 0.0
        severity_multipliers = {
            'critical': 1.5,
            'high': 1.0,
            'medium': 0.5,
            'low': 0.25
        }
        
        for threat in threats:
            category = threat.get('category', 'unknown')
            severity = threat.get('severity', 'medium')
            
            base_weight = self.threat_weights.get(category, 10)
            multiplier = severity_multipliers.get(severity, 0.5)
            
            score += base_weight * multiplier
        
        # Normalize to 0-100
        return min(score, 100.0)
    
    async def analyze_auth_attempt(self, username: str, success: bool, 
                                  source_ip: str, metadata: Dict = None):
        """Analyze authentication attempt for suspicious patterns"""
        # Track failed attempts
        if not success:
            fails_key = f"auth_fails:{source_ip}"
            fails = await self.redis.incr(fails_key)
            await self.redis.expire(fails_key, 3600)  # 1 hour
            
            if fails >= 5:
                security_events.labels(
                    event_type='brute_force',
                    severity='high',
                    source=source_ip,
                    action='detected'
                ).inc()
                
                # Add to temporary blacklist
                blacklist_key = f"blacklist:{source_ip}"
                await self.redis.setex(blacklist_key, 3600, "brute_force")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r"admin.*test",
            r"test.*admin",
            r"root",
            r"administrator"
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, username, re.I):
                security_events.labels(
                    event_type='suspicious_username',
                    severity='medium',
                    source=source_ip,
                    action='detected'
                ).inc()
                break
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        # Calculate threat levels
        threat_levels = {}
        for category in self.threat_weights.keys():
            level = threat_score.labels(category=category)._value.get() or 0
            threat_levels[category] = level
        
        # Get pattern match statistics
        pattern_stats = [
            {
                'name': p.name,
                'category': p.category,
                'matches': p.matches
            }
            for p in self.threat_patterns
        ]
        
        # Sort by matches
        pattern_stats.sort(key=lambda x: x['matches'], reverse=True)
        
        return {
            'overall_threat_score': threat_score.labels(category='overall')._value.get() or 0,
            'threat_levels': threat_levels,
            'top_patterns': pattern_stats[:10],
            'active_blacklists': await self._get_active_blacklists(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _get_active_blacklists(self) -> List[str]:
        """Get currently blacklisted IPs"""
        blacklists = []
        cursor = b'0'
        
        while cursor:
            cursor, keys = await self.redis.scan(
                cursor, match=b'blacklist:*', count=100
            )
            for key in keys:
                ip = key.decode('utf-8').split(':')[1]
                blacklists.append(ip)
            
            if cursor == b'0':
                break
        
        return blacklists

# API endpoint integration
from fastapi import Request, HTTPException
from functools import wraps

security_monitor = None

async def get_security_monitor() -> SecurityMonitor:
    """Get or create security monitor instance"""
    global security_monitor
    if security_monitor is None:
        security_monitor = SecurityMonitor()
        await security_monitor.initialize()
    return security_monitor

def secure_endpoint(severity_threshold: int = 40):
    """Decorator to add security scanning to endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            monitor = await get_security_monitor()
            
            # Prepare request data for scanning
            request_data = {
                'path': str(request.url.path),
                'query_params': dict(request.query_params),
                'headers': dict(request.headers),
                'source_ip': request.client.host if request.client else 'unknown',
                'method': request.method
            }
            
            # Get body if present
            if request.method in ['POST', 'PUT', 'PATCH']:
                try:
                    body = await request.json()
                    request_data['body'] = body
                except:
                    pass
            
            # Scan request
            scan_result = await monitor.scan_request(request_data)
            
            # Take action based on result
            if scan_result['action'] == 'block':
                raise HTTPException(
                    status_code=403,
                    detail="Request blocked due to security concerns"
                )
            elif scan_result['action'] == 'challenge':
                # In production, implement CAPTCHA or similar
                request.state.security_challenge = True
            
            # Add security context to request
            request.state.security_scan = scan_result
            
            # Execute original function
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator
EOF

# Create security rules
cat > monitoring/security/rules/security_alerts.yml << 'EOF'
groups:
  - name: security_alerts
    interval: 30s
    rules:
      # Brute force detection
      - alert: BruteForceAttempt
        expr: |
          sum(rate(csp_security_events_total{event_type="brute_force"}[5m])) > 0.1
        for: 2m
        labels:
          severity: critical
          category: security
        annotations:
          summary: "Brute force attack detected"
          description: "Multiple failed authentication attempts from {{ $labels.source }}"
          
      # SQL injection attempts
      - alert: SQLInjectionAttempt
        expr: |
          sum(rate(csp_security_events_total{event_type="sql_injection"}[5m])) > 0.05
        for: 1m
        labels:
          severity: critical
          category: security
        annotations:
          summary: "SQL injection attempts detected"
          description: "SQL injection patterns detected from {{ $labels.source }}"
      
      # High threat score
      - alert: HighThreatScore
        expr: |
          csp_threat_score{category="overall"} > 70
        for: 1m
        labels:
          severity: critical
          category: security
        annotations:
          summary: "High security threat level"
          description: "Overall threat score is {{ $value }}"
      
      # Suspicious scanning activity
      - alert: SecurityScanning
        expr: |
          sum(rate(csp_security_events_total{event_type="scanner"}[10m])) > 0.05
        for: 5m
        labels:
          severity: high
          category: security
        annotations:
          summary: "Security scanning detected"
          description: "Automated security scanning tools detected"
      
      # Rate limit violations
      - alert: RateLimitViolations
        expr: |
          sum(rate(csp_blocked_requests_total{reason="high_threat_score"}[5m])) > 0.1
        for: 5m
        labels:
          severity: high
          category: security
        annotations:
          summary: "High rate of blocked requests"
          description: "{{ $value }} requests per second being blocked"
EOF

log_success "Security Monitoring setup complete!"

# ============================================================================
# 4. ALERT CORRELATION AND INTELLIGENT ROUTING
# ============================================================================

log_info "Setting up Alert Correlation and Intelligent Routing..."

mkdir -p monitoring/alerting/{correlation,routing}

# Create alert correlation engine
cat > monitoring/alerting/correlation/correlator.py << 'EOF'
"""
Alert Correlation and Deduplication Engine
"""
import asyncio
import json
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import logging

from prometheus_client import Counter, Gauge, Histogram
import aioredis
import networkx as nx

logger = logging.getLogger(__name__)

# Metrics
alerts_correlated = Counter(
    'csp_alerts_correlated_total',
    'Total alerts correlated',
    ['correlation_type']
)

incidents_created = Counter(
    'csp_incidents_created_total',
    'Total incidents created',
    ['severity', 'category']
)

alert_noise_reduction = Gauge(
    'csp_alert_noise_reduction_ratio',
    'Alert noise reduction ratio'
)

correlation_duration = Histogram(
    'csp_correlation_duration_seconds',
    'Time taken for alert correlation'
)

class Alert:
    """Represents an alert"""
    
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get('fingerprint', hashlib.md5(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest())
        self.name = data.get('alertname', 'unknown')
        self.severity = data.get('labels', {}).get('severity', 'medium')
        self.category = data.get('labels', {}).get('category', 'unknown')
        self.service = data.get('labels', {}).get('service', 'unknown')
        self.instance = data.get('labels', {}).get('instance', 'unknown')
        self.timestamp = datetime.fromisoformat(
            data.get('startsAt', datetime.utcnow().isoformat())
        )
        self.labels = data.get('labels', {})
        self.annotations = data.get('annotations', {})
        self.value = data.get('value', 0)
        self.raw_data = data

class Incident:
    """Represents a correlated incident"""
    
    def __init__(self, incident_id: str):
        self.id = incident_id
        self.alerts: List[Alert] = []
        self.severity = 'low'
        self.category = 'unknown'
        self.services: Set[str] = set()
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.title = ""
        self.description = ""
        self.root_cause = None
        self.impact_score = 0
        self.correlation_confidence = 0

class AlertCorrelator:
    """Correlates alerts into incidents"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        
        # Correlation rules
        self.correlation_window = timedelta(minutes=5)
        self.correlation_rules = self._load_correlation_rules()
        
        # Dependency graph
        self.service_graph = self._build_service_graph()
        
        # Alert history
        self.alert_history: Dict[str, List[Alert]] = defaultdict(list)
        self.active_incidents: Dict[str, Incident] = {}
    
    def _load_correlation_rules(self) -> List[Dict]:
        """Load correlation rules"""
        return [
            {
                'name': 'cascade_failure',
                'description': 'Correlate cascading failures across services',
                'conditions': [
                    {'field': 'category', 'value': 'availability'},
                    {'time_window': 60, 'min_services': 2}
                ],
                'priority': 1
            },
            {
                'name': 'resource_exhaustion',
                'description': 'Correlate resource exhaustion alerts',
                'conditions': [
                    {'field': 'name', 'pattern': '(memory|cpu|disk).*high'},
                    {'field': 'service', 'same': True},
                    {'time_window': 300}
                ],
                'priority': 2
            },
            {
                'name': 'security_attack',
                'description': 'Correlate security-related alerts',
                'conditions': [
                    {'field': 'category', 'value': 'security'},
                    {'time_window': 120}
                ],
                'priority': 1
            },
            {
                'name': 'performance_degradation',
                'description': 'Correlate performance issues',
                'conditions': [
                    {'field': 'name', 'pattern': '(latency|response_time).*high'},
                    {'correlation': 'upstream_downstream'},
                    {'time_window': 180}
                ],
                'priority': 3
            }
        ]
    
    def _build_service_graph(self) -> nx.DiGraph:
        """Build service dependency graph"""
        G = nx.DiGraph()
        
        # Define service dependencies
        dependencies = [
            ('frontend', 'api'),
            ('api', 'auth-service'),
            ('api', 'database'),
            ('api', 'cache'),
            ('api', 'ai-service'),
            ('ai-service', 'vector-db'),
            ('auth-service', 'database'),
            ('auth-service', 'cache')
        ]
        
        G.add_edges_from(dependencies)
        return G
    
    async def initialize(self):
        """Initialize correlator"""
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        logger.info("Alert correlator initialized")
    
    async def process_alert(self, alert_data: Dict[str, Any]) -> Optional[Incident]:
        """Process incoming alert and correlate with existing alerts"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            alert = Alert(alert_data)
            
            # Add to history
            self.alert_history[alert.service].append(alert)
            
            # Find correlations
            correlated_alerts = await self._find_correlations(alert)
            
            if correlated_alerts:
                # Create or update incident
                incident = await self._create_or_update_incident(
                    alert, correlated_alerts
                )
                
                # Track metrics
                alerts_correlated.labels(
                    correlation_type=incident.category
                ).inc()
                
                # Calculate noise reduction
                total_alerts = len(correlated_alerts) + 1
                noise_reduction = 1 - (1 / total_alerts)
                alert_noise_reduction.set(noise_reduction)
                
                return incident
            
            return None
            
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            correlation_duration.observe(duration)
    
    async def _find_correlations(self, alert: Alert) -> List[Alert]:
        """Find alerts that correlate with the given alert"""
        correlated = []
        
        for rule in self.correlation_rules:
            matches = await self._evaluate_rule(alert, rule)
            correlated.extend(matches)
        
        # Remove duplicates
        seen = set()
        unique_correlated = []
        for a in correlated:
            if a.id not in seen:
                seen.add(a.id)
                unique_correlated.append(a)
        
        return unique_correlated
    
    async def _evaluate_rule(self, alert: Alert, rule: Dict) -> List[Alert]:
        """Evaluate a correlation rule"""
        matches = []
        
        # Time window check
        time_window = rule.get('conditions', [{}])[0].get('time_window', 300)
        cutoff_time = alert.timestamp - timedelta(seconds=time_window)
        
        # Check all recent alerts
        for service_alerts in self.alert_history.values():
            for historical_alert in service_alerts:
                if historical_alert.timestamp < cutoff_time:
                    continue
                
                if historical_alert.id == alert.id:
                    continue
                
                # Evaluate conditions
                if self._match_conditions(alert, historical_alert, rule['conditions']):
                    matches.append(historical_alert)
        
        # Apply correlation-specific logic
        if 'correlation' in rule:
            if rule['correlation'] == 'upstream_downstream':
                matches = self._filter_by_service_dependency(alert, matches)
        
        return matches
    
    def _match_conditions(self, alert: Alert, other: Alert, conditions: List[Dict]) -> bool:
        """Check if alerts match correlation conditions"""
        for condition in conditions:
            if 'field' in condition:
                field = condition['field']
                
                if 'value' in condition:
                    # Exact match
                    if getattr(alert, field, None) != condition['value']:
                        return False
                    if getattr(other, field, None) != condition['value']:
                        return False
                
                elif 'same' in condition and condition['same']:
                    # Same value check
                    if getattr(alert, field, None) != getattr(other, field, None):
                        return False
                
                elif 'pattern' in condition:
                    # Pattern match
                    import re
                    pattern = condition['pattern']
                    if not re.search(pattern, getattr(alert, field, ''), re.I):
                        return False
                    if not re.search(pattern, getattr(other, field, ''), re.I):
                        return False
        
        return True
    
    def _filter_by_service_dependency(self, alert: Alert, matches: List[Alert]) -> List[Alert]:
        """Filter alerts based on service dependencies"""
        filtered = []
        
        if alert.service in self.service_graph:
            # Get upstream and downstream services
            upstream = list(self.service_graph.predecessors(alert.service))
            downstream = list(self.service_graph.successors(alert.service))
            related_services = upstream + downstream + [alert.service]
            
            for match in matches:
                if match.service in related_services:
                    filtered.append(match)
        else:
            # If service not in graph, include all matches
            filtered = matches
        
        return filtered
    
    async def _create_or_update_incident(self, alert: Alert, 
                                       correlated: List[Alert]) -> Incident:
        """Create or update an incident"""
        # Check for existing incident
        incident_key = f"incident:{alert.service}:{alert.category}"
        existing_id = await self.redis.get(incident_key)
        
        if existing_id and existing_id.decode() in self.active_incidents:
            # Update existing incident
            incident = self.active_incidents[existing_id.decode()]
            incident.alerts.append(alert)
            incident.alerts.extend(correlated)
            incident.updated_at = datetime.utcnow()
        else:
            # Create new incident
            incident_id = f"INC-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            incident = Incident(incident_id)
            incident.alerts = [alert] + correlated
            
            # Store incident
            self.active_incidents[incident_id] = incident
            await self.redis.setex(incident_key, 3600, incident_id)
            
            # Track metric
            incidents_created.labels(
                severity=incident.severity,
                category=incident.category
            ).inc()
        
        # Update incident properties
        incident.services = {a.service for a in incident.alerts}
        incident.severity = max(a.severity for a in incident.alerts)
        incident.category = self._determine_category(incident.alerts)
        incident.title = self._generate_title(incident)
        incident.description = self._generate_description(incident)
        incident.impact_score = self._calculate_impact(incident)
        incident.correlation_confidence = self._calculate_confidence(incident)
        
        # Attempt root cause analysis
        incident.root_cause = await self._analyze_root_cause(incident)
        
        return incident
    
    def _determine_category(self, alerts: List[Alert]) -> str:
        """Determine incident category from alerts"""
        categories = [a.category for a in alerts]
        # Return most common category
        return max(set(categories), key=categories.count)
    
    def _generate_title(self, incident: Incident) -> str:
        """Generate incident title"""
        if len(incident.services) == 1:
            return f"{incident.severity.upper()}: {list(incident.services)[0]} - {incident.category}"
        else:
            return f"{incident.severity.upper()}: Multiple services affected - {incident.category}"
    
    def _generate_description(self, incident: Incident) -> str:
        """Generate incident description"""
        alert_summary = defaultdict(int)
        for alert in incident.alerts:
            alert_summary[alert.name] += 1
        
        desc_parts = [
            f"Incident affecting {len(incident.services)} service(s).",
            f"Total alerts: {len(incident.alerts)}",
            "",
            "Alert breakdown:"
        ]
        
        for alert_name, count in sorted(alert_summary.items(), 
                                      key=lambda x: x[1], reverse=True):
            desc_parts.append(f"  - {alert_name}: {count}")
        
        return "\n".join(desc_parts)
    
    def _calculate_impact(self, incident: Incident) -> float:
        """Calculate incident impact score"""
        # Factors: severity, number of services, alert count
        severity_scores = {'critical': 1.0, 'high': 0.7, 'medium': 0.4, 'low': 0.2}
        
        severity_score = severity_scores.get(incident.severity, 0.5)
        service_score = min(len(incident.services) / 5, 1.0)  # Normalize to 0-1
        alert_score = min(len(incident.alerts) / 10, 1.0)  # Normalize to 0-1
        
        # Weighted average
        impact = (severity_score * 0.5 + service_score * 0.3 + alert_score * 0.2)
        
        return round(impact, 2)
    
    def _calculate_confidence(self, incident: Incident) -> float:
        """Calculate correlation confidence"""
        # Factors: time proximity, service relationships, pattern matching
        confidence = 0.0
        
        # Time proximity
        if incident.alerts:
            time_range = max(a.timestamp for a in incident.alerts) - \
                        min(a.timestamp for a in incident.alerts)
            if time_range < timedelta(minutes=1):
                confidence += 0.4
            elif time_range < timedelta(minutes=5):
                confidence += 0.2
        
        # Service relationships
        if len(incident.services) > 1:
            # Check if services are related in dependency graph
            service_list = list(incident.services)
            for i in range(len(service_list)):
                for j in range(i + 1, len(service_list)):
                    if nx.has_path(self.service_graph, service_list[i], service_list[j]):
                        confidence += 0.2
                        break
        
        # Pattern matching
        alert_names = [a.name for a in incident.alerts]
        if len(set(alert_names)) < len(alert_names) * 0.5:
            # Many similar alerts
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    async def _analyze_root_cause(self, incident: Incident) -> Optional[str]:
        """Attempt to identify root cause"""
        # Simple heuristic-based root cause analysis
        
        # Check for cascade pattern
        if len(incident.services) > 2:
            # Sort alerts by time
            sorted_alerts = sorted(incident.alerts, key=lambda a: a.timestamp)
            first_service = sorted_alerts[0].service
            
            # Check if first service is upstream of others
            is_upstream = True
            for service in incident.services:
                if service != first_service:
                    if not nx.has_path(self.service_graph, first_service, service):
                        is_upstream = False
                        break
            
            if is_upstream:
                return f"Cascade failure originating from {first_service}"
        
        # Check for resource exhaustion
        resource_alerts = [a for a in incident.alerts 
                          if 'memory' in a.name or 'cpu' in a.name or 'disk' in a.name]
        if len(resource_alerts) > len(incident.alerts) * 0.7:
            return "Resource exhaustion across services"
        
        # Check for security attack pattern
        security_alerts = [a for a in incident.alerts if a.category == 'security']
        if len(security_alerts) > len(incident.alerts) * 0.8:
            return "Coordinated security attack"
        
        return None

# Intelligent alert routing
class AlertRouter:
    """Routes alerts based on content, time, and expertise"""
    
    def __init__(self):
        self.routing_rules = self._load_routing_rules()
        self.on_call_schedule = {}
        self.expertise_map = self._load_expertise_map()
    
    def _load_routing_rules(self) -> Dict:
        """Load routing rules"""
        return {
            'business_hours': {
                'start': 9,
                'end': 17,
                'timezone': 'UTC',
                'channels': ['slack', 'email']
            },
            'after_hours': {
                'channels': {
                    'critical': ['pagerduty', 'phone'],
                    'high': ['slack', 'email'],
                    'medium': ['email'],
                    'low': ['email']
                }
            },
            'escalation': {
                'critical': {
                    'initial_wait': 5,  # minutes
                    'escalation_levels': [
                        {'wait': 5, 'notify': ['team_lead']},
                        {'wait': 10, 'notify': ['manager']},
                        {'wait': 20, 'notify': ['director']}
                    ]
                }
            }
        }
    
    def _load_expertise_map(self) -> Dict:
        """Load expertise mapping"""
        return {
            'database': ['dba-team', 'john.doe@company.com'],
            'security': ['security-team', 'security-oncall@company.com'],
            'api': ['backend-team', 'api-oncall@company.com'],
            'ai': ['ml-team', 'ai-oncall@company.com'],
            'infrastructure': ['sre-team', 'sre-oncall@company.com']
        }
    
    async def route_incident(self, incident: Incident) -> Dict[str, Any]:
        """Route incident to appropriate channels and people"""
        routing_decision = {
            'incident_id': incident.id,
            'channels': [],
            'recipients': [],
            'escalation_plan': None
        }
        
        # Determine channels based on time and severity
        if self._is_business_hours():
            routing_decision['channels'] = self.routing_rules['business_hours']['channels']
        else:
            severity_channels = self.routing_rules['after_hours']['channels']
            routing_decision['channels'] = severity_channels.get(
                incident.severity, ['email']
            )
        
        # Add recipients based on expertise
        for service in incident.services:
            if service in self.expertise_map:
                routing_decision['recipients'].extend(self.expertise_map[service])
        
        # Remove duplicates
        routing_decision['recipients'] = list(set(routing_decision['recipients']))
        
        # Create escalation plan for critical incidents
        if incident.severity == 'critical':
            routing_decision['escalation_plan'] = self._create_escalation_plan(incident)
        
        return routing_decision
    
    def _is_business_hours(self) -> bool:
        """Check if current time is business hours"""
        from datetime import datetime
        import pytz
        
        tz = pytz.timezone(self.routing_rules['business_hours']['timezone'])
        now = datetime.now(tz)
        
        start_hour = self.routing_rules['business_hours']['start']
        end_hour = self.routing_rules['business_hours']['end']
        
        return start_hour <= now.hour < end_hour and now.weekday() < 5
    
    def _create_escalation_plan(self, incident: Incident) -> Dict:
        """Create escalation plan for incident"""
        escalation_config = self.routing_rules['escalation'][incident.severity]
        
        plan = {
            'initial_notification': datetime.utcnow().isoformat(),
            'levels': []
        }
        
        current_time = datetime.utcnow()
        for level in escalation_config['escalation_levels']:
            escalation_time = current_time + timedelta(minutes=level['wait'])
            plan['levels'].append({
                'time': escalation_time.isoformat(),
                'notify': level['notify']
            })
        
        return plan
EOF

log_success "Alert Correlation setup complete!"

# ============================================================================
# 5. INTEGRATION SCRIPTS
# ============================================================================

log_info "Creating integration scripts..."

# Create main monitoring enhancement script
cat > monitoring/enhance-monitoring.py << 'EOF'
"""
Main script to enhance monitoring with high-priority features
"""
import asyncio
import os
import sys
from typing import Optional

# Add monitoring modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tracing import get_tracing
from anomaly_detection.detector import AnomalyDetector
from security.security_monitor import SecurityMonitor
from alerting.correlation.correlator import AlertCorrelator

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedMonitoring:
    """Orchestrates all monitoring enhancements"""
    
    def __init__(self):
        self.tracing = None
        self.anomaly_detector = None
        self.security_monitor = None
        self.alert_correlator = None
    
    async def initialize(self):
        """Initialize all monitoring components"""
        logger.info("Initializing enhanced monitoring components...")
        
        # Initialize tracing
        self.tracing = get_tracing()
        logger.info("✓ Distributed tracing initialized")
        
        # Initialize anomaly detection
        self.anomaly_detector = AnomalyDetector()
        await self.anomaly_detector.initialize()
        logger.info("✓ Anomaly detection initialized")
        
        # Initialize security monitoring
        self.security_monitor = SecurityMonitor()
        await self.security_monitor.initialize()
        logger.info("✓ Security monitoring initialized")
        
        # Initialize alert correlation
        self.alert_correlator = AlertCorrelator()
        await self.alert_correlator.initialize()
        logger.info("✓ Alert correlation initialized")
        
        logger.info("All monitoring enhancements initialized successfully!")
    
    async def run(self):
        """Run all monitoring services"""
        tasks = [
            asyncio.create_task(self.anomaly_detector.run_continuous_detection()),
            asyncio.create_task(self.run_security_monitoring()),
            asyncio.create_task(self.run_alert_processing())
        ]
        
        logger.info("Starting enhanced monitoring services...")
        await asyncio.gather(*tasks)
    
    async def run_security_monitoring(self):
        """Run security monitoring loop"""
        while True:
            try:
                # Get security status every minute
                status = await self.security_monitor.get_security_status()
                logger.info(f"Security status: threat_score={status['overall_threat_score']}")
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error in security monitoring: {str(e)}")
                await asyncio.sleep(60)
    
    async def run_alert_processing(self):
        """Process alerts for correlation"""
        # In production, this would connect to Alertmanager webhook
        while True:
            await asyncio.sleep(30)

async def main():
    """Main entry point"""
    monitoring = EnhancedMonitoring()
    await monitoring.initialize()
    await monitoring.run()

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Create FastAPI integration for backend
cat > backend/monitoring/monitoring_integration.py << 'EOF'
"""
Integration of enhanced monitoring into FastAPI backend
"""
from fastapi import FastAPI, Request, Response
from typing import Optional
import time

# Import monitoring components
from .tracing import get_tracing
from monitoring.security.security_monitor import secure_endpoint, get_security_monitor
from monitoring.anomaly_detection.detector import AnomalyDetector

def integrate_enhanced_monitoring(app: FastAPI):
    """Integrate all monitoring enhancements into FastAPI app"""
    
    # Initialize tracing
    tracing = get_tracing()
    tracing.instrument_app(app)
    
    # Add request timing middleware
    @app.middleware("http")
    async def add_request_timing(request: Request, call_next):
        start_time = time.time()
        
        # Add trace context
        with tracing.trace_operation(f"{request.method} {request.url.path}") as span:
            span.set_attributes({
                "http.method": request.method,
                "http.url": str(request.url),
                "http.scheme": request.url.scheme,
                "http.host": request.url.hostname,
                "http.target": request.url.path,
                "user_agent": request.headers.get("user-agent", "")
            })
            
            response = await call_next(request)
            
            # Add response attributes
            span.set_attributes({
                "http.status_code": response.status_code,
                "http.response_time": time.time() - start_time
            })
            
            return response
    
    # Add security monitoring endpoints
    @app.get("/api/security/status")
    async def get_security_status():
        """Get current security status"""
        monitor = await get_security_monitor()
        return await monitor.get_security_status()
    
    # Add anomaly detection endpoint
    @app.get("/api/anomalies/current")
    async def get_current_anomalies():
        """Get currently detected anomalies"""
        detector = AnomalyDetector()
        anomalies = {}
        
        for metric in detector.monitored_metrics:
            result = await detector.detect_anomalies(metric['name'])
            if result['status'] == 'success' and result['anomalies']:
                anomalies[metric['name']] = result
        
        return anomalies
    
    # Example of securing an endpoint
    @app.post("/api/secure-endpoint")
    @secure_endpoint(severity_threshold=30)
    async def secure_endpoint_example(request: Request):
        """Example endpoint with security scanning"""
        # Access security scan results
        security_scan = request.state.security_scan
        
        return {
            "message": "Request processed",
            "threat_score": security_scan['threat_score'],
            "scan_time": security_scan['scan_time']
        }
    
    return app
EOF

# Create deployment script
cat > monitoring/deploy-enhancements.sh << 'EOF'
#!/bin/bash
# Deploy monitoring enhancements

set -euo pipefail

echo "Deploying monitoring enhancements..."

# Start tracing
echo "Starting distributed tracing..."
cd monitoring/tracing
docker-compose -f docker-compose.tracing.yml up -d
cd ../..

# Start anomaly detection
echo "Starting anomaly detection..."
cd monitoring/anomaly_detection
docker build -t csp-anomaly-detector -f Dockerfile.anomaly .
docker-compose -f docker-compose.anomaly.yml up -d
cd ../..

# Update Prometheus with new rules
echo "Updating Prometheus configuration..."
cp monitoring/security/rules/security_alerts.yml monitoring/prometheus/rules/
docker-compose -f monitoring/docker-compose.monitoring.yml restart prometheus

# Show status
echo ""
echo "Monitoring enhancements deployed!"
echo ""
echo "New endpoints available:"
echo "- Jaeger UI: http://localhost:16686"
echo "- Security Status: http://localhost:8000/api/security/status"
echo "- Anomalies: http://localhost:8000/api/anomalies/current"
echo ""
echo "Integration steps:"
echo "1. Update backend/main.py to import monitoring_integration"
echo "2. Call integrate_enhanced_monitoring(app) after creating FastAPI app"
echo "3. Restart the backend service"
EOF

chmod +x monitoring/deploy-enhancements.sh

log_success "All high-priority monitoring enhancements created!"

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "====================================================="
echo "HIGH PRIORITY MONITORING ENHANCEMENTS COMPLETE"
echo "====================================================="
echo ""
echo "✅ Distributed Tracing (OpenTelemetry + Jaeger)"
echo "   - Automatic instrumentation for FastAPI, SQL, Redis"
echo "   - Custom trace decorators for operations"
echo "   - Trace context propagation"
echo ""
echo "✅ Anomaly Detection (Machine Learning)"
echo "   - Isolation Forest algorithm"
echo "   - Monitors: API latency, errors, CPU, memory, DB"
echo "   - Continuous detection with model persistence"
echo ""
echo "✅ Security Monitoring"
echo "   - Real-time threat detection"
echo "   - Pattern matching for SQL injection, XSS, etc."
echo "   - Rate limiting and IP reputation"
echo "   - Automatic request blocking"
echo ""
echo "✅ Alert Correlation & Routing"
echo "   - Intelligent alert grouping"
echo "   - Service dependency awareness"
echo "   - Time and expertise-based routing"
echo "   - Automated incident creation"
echo ""
echo "NEXT STEPS:"
echo "1. Run: ./monitoring/deploy-enhancements.sh"
echo "2. Update backend/main.py with monitoring integration"
echo "3. Configure alert routing recipients"
echo "4. Train anomaly detection models with your data"
echo ""
echo "Access new services:"
echo "- Jaeger: http://localhost:16686"
echo "- Security API: http://localhost:8000/api/security/status"
echo "- Anomalies API: http://localhost:8000/api/anomalies/current" 