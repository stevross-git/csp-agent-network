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
