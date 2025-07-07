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
