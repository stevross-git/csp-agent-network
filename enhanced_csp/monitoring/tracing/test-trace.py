#!/usr/bin/env python3
"""
Test script to send a trace to Jaeger
"""
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
import time

# Configure the tracer
resource = Resource.create({"service.name": "test-service"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# Configure OTLP exporter
# If using OTEL Collector: localhost:4317
# If using Jaeger directly: localhost:4317
otlp_exporter = OTLPSpanExporter(
    endpoint="localhost:4317",
    insecure=True
)

# Add the exporter to the tracer
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Create a test trace
with tracer.start_as_current_span("test-operation") as span:
    span.set_attribute("test.type", "manual")
    span.set_attribute("test.value", 42)
    
    # Simulate some work
    time.sleep(0.1)
    
    # Create a child span
    with tracer.start_as_current_span("child-operation") as child:
        child.set_attribute("child.data", "test-data")
        time.sleep(0.05)

print("Test trace sent! Check Jaeger UI at http://localhost:16686")
