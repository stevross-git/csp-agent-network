# enhanced_csp/api/metrics.py
"""
Prometheus metrics endpoint for network optimization
"""

import os
from prometheus_client import (
    CONTENT_TYPE_LATEST, 
    generate_latest, 
    CollectorRegistry,
    multiprocess
)

def setup_metrics():
    """Setup Prometheus metrics with multiprocess support"""
    # Check if we're in multiprocess mode
    if os.environ.get('PROMETHEUS_MULTIPROC_DIR'):
        # Use multiprocess collector
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    else:
        # Use default registry
        return None

def get_metrics_response():
    """Generate metrics response"""
    registry = setup_metrics()
    metrics_data = generate_latest(registry)
    return metrics_data, CONTENT_TYPE_LATEST

# If using FastAPI
async def metrics_endpoint():
    """FastAPI metrics endpoint"""
    data, content_type = get_metrics_response()
    return Response(content=data, media_type=content_type)

# If using Flask
def metrics_endpoint_flask():
    """Flask metrics endpoint"""
    data, content_type = get_metrics_response()
    return Response(data, mimetype=content_type)