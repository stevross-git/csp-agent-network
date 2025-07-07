# Jaeger Tracing Integration Guide

## Quick Start

Jaeger is now running with OTLP support. Here's how to integrate it with your application:

### Python Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

# Configure tracer
resource = Resource.create({"service.name": "your-service"})
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

# Add OTLP exporter
otlp_exporter = OTLPSpanExporter(endpoint="localhost:4317", insecure=True)
provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

# Use tracer in your code
tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("function_name")
def your_function():
    # Your code here
    pass
```

### Environment Variables

You can also configure via environment variables:

```bash
export OTEL_SERVICE_NAME="your-service"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_EXPORTER_OTLP_INSECURE="true"
```

### FastAPI Integration

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# After setting up tracer as above
app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
```

## Access Points

- **Jaeger UI**: http://localhost:16686
- **OTLP gRPC**: localhost:4317
- **OTLP HTTP**: localhost:4318
- **Zipkin**: localhost:9411

## Testing

Run the test script:
```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
python3 monitoring/test-trace.py
```

Then check Jaeger UI for the traces.
