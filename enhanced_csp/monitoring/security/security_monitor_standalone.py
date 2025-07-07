"""
Standalone Security Monitor - Minimal Dependencies
"""
import re
from typing import Dict, List, Any
from datetime import datetime
import json

class MinimalSecurityMonitor:
    """Lightweight security monitor for testing"""
    
    def __init__(self):
        self.threat_patterns = [
            (r"(union|select|drop).*?(from|table)", "sql_injection", "high"),
            (r"<script[^>]*>", "xss", "high"),
            (r"\.\./|\.\.", "path_traversal", "medium"),
            (r"(;|\||&|`)", "command_injection", "high"),
        ]
        self.events = []
    
    def scan_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scan request for threats"""
        threats = []
        
        # Convert request data to string for scanning
        scan_text = json.dumps(request_data)
        
        for pattern, threat_type, severity in self.threat_patterns:
            if re.search(pattern, scan_text, re.IGNORECASE):
                threats.append({
                    'type': threat_type,
                    'severity': severity,
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        threat_score = len(threats) * 25
        action = 'block' if threat_score >= 75 else 'allow'
        
        result = {
            'action': action,
            'threat_score': min(threat_score, 100),
            'threats': threats
        }
        
        self.events.append(result)
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get security status"""
        return {
            'total_scans': len(self.events),
            'threats_detected': sum(1 for e in self.events if e['threats']),
            'average_threat_score': sum(e['threat_score'] for e in self.events) / max(len(self.events), 1),
            'timestamp': datetime.utcnow().isoformat()
        }

# Global instance
security_monitor = MinimalSecurityMonitor()
