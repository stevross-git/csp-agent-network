"""
SLI/SLO calculation and tracking
"""
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    from monitoring import get_default
    monitor = get_default()
    MONITORING_ENABLED = True
except ImportError:
    monitor = None
    MONITORING_ENABLED = False

@dataclass
class SLO:
    """Service Level Objective definition"""
    name: str
    target: float  # Target percentage (0-100)
    measurement_window: timedelta
    
@dataclass
class SLI:
    """Service Level Indicator measurement"""
    timestamp: datetime
    value: float
    slo_name: str

class SLITracker:
    """Track SLIs and calculate SLO compliance"""
    
    def __init__(self):
        self.slos = {
            "availability": SLO("availability", 99.9, timedelta(days=30)),
            "latency_p99": SLO("latency_p99", 95.0, timedelta(days=30)),  # 95% under 500ms
            "error_rate": SLO("error_rate", 99.5, timedelta(days=30)),    # <0.5% errors
        }
        self.measurements: Dict[str, List[SLI]] = {
            "availability": [],
            "latency_p99": [],
            "error_rate": []
        }
        self.last_calculation = time.time()
    
    def record_request(self, success: bool, latency: float):
        """Record a request for SLI calculation"""
        timestamp = datetime.now()
        
        # Update availability (success = available)
        self.measurements["availability"].append(
            SLI(timestamp, 100.0 if success else 0.0, "availability")
        )
        
        # Update error rate
        self.measurements["error_rate"].append(
            SLI(timestamp, 100.0 if success else 0.0, "error_rate")
        )
        
        # Update latency (only for successful requests)
        if success and latency < 0.5:  # Under 500ms
            self.measurements["latency_p99"].append(
                SLI(timestamp, 100.0, "latency_p99")
            )
        elif success:
            self.measurements["latency_p99"].append(
                SLI(timestamp, 0.0, "latency_p99")
            )
    
    def calculate_slo_compliance(self) -> Dict[str, float]:
        """Calculate current SLO compliance"""
        compliance = {}
        current_time = datetime.now()
        
        for slo_name, slo in self.slos.items():
            # Filter measurements within window
            cutoff_time = current_time - slo.measurement_window
            recent_measurements = [
                m for m in self.measurements[slo_name]
                if m.timestamp > cutoff_time
            ]
            
            if not recent_measurements:
                compliance[slo_name] = 100.0
                continue
            
            # Calculate average
            avg_value = sum(m.value for m in recent_measurements) / len(recent_measurements)
            
            # Calculate compliance
            if slo_name == "error_rate":
                # For error rate, we want high success rate
                compliance[slo_name] = 100.0 if avg_value >= slo.target else 0.0
            else:
                # For others, direct comparison
                compliance[slo_name] = 100.0 if avg_value >= slo.target else 0.0
            
            # Update metrics
            if MONITORING_ENABLED:
                monitor.update_sli(slo_name, avg_value / 100.0)
                monitor.update_slo_compliance(slo_name, compliance[slo_name])
        
        # Clean old measurements
        self._cleanup_old_measurements(cutoff_time)
        
        return compliance
    
    def _cleanup_old_measurements(self, cutoff_time: datetime):
        """Remove old measurements"""
        for slo_name in self.measurements:
            self.measurements[slo_name] = [
                m for m in self.measurements[slo_name]
                if m.timestamp > cutoff_time
            ]
    
    def get_slo_status(self) -> Dict[str, Any]:
        """Get current SLO status"""
        compliance = self.calculate_slo_compliance()
        
        return {
            "slos": {
                name: {
                    "target": slo.target,
                    "current_compliance": compliance.get(name, 100.0),
                    "window": str(slo.measurement_window)
                }
                for name, slo in self.slos.items()
            },
            "last_updated": datetime.now().isoformat()
        }

# Global SLI tracker
sli_tracker = SLITracker()
