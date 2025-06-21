#!/usr/bin/env python3
"""
Monitoring integration for Enhanced CSP breach benchmarking
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class CSPMonitoringIntegration:
    """Integration with Enhanced CSP monitoring system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics from CSP monitoring"""
        try:
            url = f"{self.config['monitoring_api']}/metrics"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get metrics: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}
    
    async def send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert to CSP alert system"""
        try:
            url = f"{self.config['alerts_api']}/alert"
            async with self.session.post(url, json=alert_data) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False
    
    async def update_topology_view(self, topology_data: Dict[str, Any]) -> bool:
        """Update topology visualization with benchmark results"""
        try:
            url = f"{self.config['visualizer_api']}/topology/update"
            async with self.session.post(url, json=topology_data) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Error updating topology: {e}")
            return False

async def main():
    """Test monitoring integration"""
    config = {
        'monitoring_api': 'http://localhost:8002/api',
        'alerts_api': 'http://localhost:8004/api',
        'visualizer_api': 'http://localhost:8003/api'
    }
    
    async with CSPMonitoringIntegration(config) as monitor:
        # Test getting metrics
        metrics = await monitor.get_system_metrics()
        print(f"Current metrics: {json.dumps(metrics, indent=2)}")
        
        # Test sending alert
        alert = {
            'severity': 'info',
            'message': 'Breach benchmarking initiated',
            'timestamp': datetime.now().isoformat(),
            'source': 'benchmark_system'
        }
        success = await monitor.send_alert(alert)
        print(f"Alert sent: {success}")

if __name__ == "__main__":
    asyncio.run(main())

    async def check_service_health(self, service_url: str) -> bool:
        """Check if a service is healthy"""
        try:
            async with self.session.get(f"{service_url}/health", timeout=3) as response:
                return response.status == 200
        except:
            return False
