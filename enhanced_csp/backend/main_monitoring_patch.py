"""
Patch for main.py to add comprehensive monitoring
"""
from backend.middleware.rate_limit_monitoring import RateLimitMonitoringMiddleware
from backend.monitoring.performance import MetricsMiddleware, PerformanceMonitor

def add_monitoring_to_app(app):
    """Add monitoring middleware to FastAPI app"""
    
    # Create performance monitor
    performance_monitor = PerformanceMonitor()
    
    # Add middleware
    app.add_middleware(MetricsMiddleware, monitor=performance_monitor)
    app.add_middleware(RateLimitMonitoringMiddleware, rate_limit=100)
    
    # Add monitoring endpoints
    @app.get("/api/metrics/performance")
    async def get_performance_metrics():
        """Get detailed performance metrics"""
        return await performance_monitor.get_metrics()
    
    @app.get("/api/metrics/alerts")
    async def get_alerts():
        """Get active alerts"""
        return performance_monitor.get_alerts()
    
    return app
