"""
Enhanced CSP Monitoring System
"""
from .csp_monitoring import CSPMonitoringSystem, MonitoringConfig

# Global monitoring instance
_monitor_instance = None

def get_default() -> CSPMonitoringSystem:
    """Get the default monitoring instance"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = CSPMonitoringSystem()
    return _monitor_instance

__all__ = ['CSPMonitoringSystem', 'MonitoringConfig', 'get_default']
