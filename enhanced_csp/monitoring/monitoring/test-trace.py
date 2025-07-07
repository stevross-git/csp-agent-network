#!/usr/bin/env python3
"""
Test Jaeger tracing setup
"""
import time
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

# Configure OpenTelemetry
resource = Resource.create({
    "service.name": "test-service",
    "service.version": "1.0.0",
})

provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

# Configure OTLP exporter to Jaeger
otlp_exporter = OTLPSpanExporter(
    endpoint="localhost:4317",
    insecure=True,
)

# Add span processor
span_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(span_processor)

# Get a tracer
tracer = trace.get_tracer("test.tracer", "1.0.0")

# Create test traces
print("Sending test traces to Jaeger...")

with tracer.start_as_current_span("test-operation") as span:
    span.set_attribute("test.attribute", "test-value")
    span.set_attribute("test.number", 42)
    span.add_event("test-event", {"event.data": "some data"})
    
    # Simulate some work
    time.sleep(0.1)
    
    # Create nested span
    with tracer.start_as_current_span("nested-operation") as nested:
        nested.set_attribute("nested.attribute", "nested-value")
        time.sleep(0.05)

# Force flush to ensure spans are sent
provider.force_flush()

print("✓ Test traces sent!")
print("✓ View them at: http://localhost:16686")
print("  - Click 'Search' in Jaeger UI")
print("  - Select 'test-service' from the Service dropdown")
print("  - Click 'Find Traces'")
