"""
Main script to enhance monitoring with high-priority features
"""
import asyncio
import os
import sys
from typing import Optional

# Add monitoring modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tracing import get_tracing
from anomaly_detection.detector import AnomalyDetector
from security.security_monitor import SecurityMonitor
from alerting.correlation.correlator import AlertCorrelator

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedMonitoring:
    """Orchestrates all monitoring enhancements"""
    
    def __init__(self):
        self.tracing = None
        self.anomaly_detector = None
        self.security_monitor = None
        self.alert_correlator = None
    
    async def initialize(self):
        """Initialize all monitoring components"""
        logger.info("Initializing enhanced monitoring components...")
        
        # Initialize tracing
        self.tracing = get_tracing()
        logger.info("✓ Distributed tracing initialized")
        
        # Initialize anomaly detection
        self.anomaly_detector = AnomalyDetector()
        await self.anomaly_detector.initialize()
        logger.info("✓ Anomaly detection initialized")
        
        # Initialize security monitoring
        self.security_monitor = SecurityMonitor()
        await self.security_monitor.initialize()
        logger.info("✓ Security monitoring initialized")
        
        # Initialize alert correlation
        self.alert_correlator = AlertCorrelator()
        await self.alert_correlator.initialize()
        logger.info("✓ Alert correlation initialized")
        
        logger.info("All monitoring enhancements initialized successfully!")
    
    async def run(self):
        """Run all monitoring services"""
        tasks = [
            asyncio.create_task(self.anomaly_detector.run_continuous_detection()),
            asyncio.create_task(self.run_security_monitoring()),
            asyncio.create_task(self.run_alert_processing())
        ]
        
        logger.info("Starting enhanced monitoring services...")
        await asyncio.gather(*tasks)
    
    async def run_security_monitoring(self):
        """Run security monitoring loop"""
        while True:
            try:
                # Get security status every minute
                status = await self.security_monitor.get_security_status()
                logger.info(f"Security status: threat_score={status['overall_threat_score']}")
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error in security monitoring: {str(e)}")
                await asyncio.sleep(60)
    
    async def run_alert_processing(self):
        """Process alerts for correlation"""
        # In production, this would connect to Alertmanager webhook
        while True:
            await asyncio.sleep(30)

async def main():
    """Main entry point"""
    monitoring = EnhancedMonitoring()
    await monitoring.initialize()
    await monitoring.run()

if __name__ == "__main__":
    asyncio.run(main())
