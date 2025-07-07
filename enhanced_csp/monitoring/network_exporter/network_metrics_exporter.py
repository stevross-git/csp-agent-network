#!/usr/bin/env python3
# monitoring/network_exporter/network_metrics_exporter.py
"""
Network Metrics Exporter
========================
Standalone exporter for network metrics to Prometheus
"""

import asyncio
import os
import logging
import signal
import sys
from datetime import datetime

from prometheus_client import start_http_server
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Since this runs in Docker with volumes mounted at /app
# The structure will be:
# /app/network/ (the network module)
# /app/monitoring/network_exporter/ (this script)
sys.path.insert(0, '/app')

from network.database_models import NetworkDatabase
from network.monitoring_integration import NetworkMonitoringService

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql+asyncpg://network_user:network_password@localhost:5432/csp_network'
)
PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', '9100'))
PROMETHEUS_PUSHGATEWAY = os.getenv('PROMETHEUS_PUSHGATEWAY')
NODE_NAME = os.getenv('NODE_NAME', 'network-exporter')
EXPORT_INTERVAL = int(os.getenv('EXPORT_INTERVAL', '15'))
SNAPSHOT_INTERVAL = int(os.getenv('SNAPSHOT_INTERVAL', '300'))


class NetworkMetricsExporter:
    """Main exporter application"""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        self.monitoring_service = None
        self._running = False
    
    async def initialize(self):
        """Initialize database connection and monitoring service"""
        logger.info("Initializing network metrics exporter...")
        
        # Create database engine
        self.engine = create_async_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False
        )
        
        # Create session factory
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Test database connection
        try:
            async with self.session_factory() as session:
                await session.execute("SELECT 1")
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
        
        # Start Prometheus HTTP server
        start_http_server(PROMETHEUS_PORT)
        logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
    
    async def run(self):
        """Run the exporter"""
        self._running = True
        
        async with self.session_factory() as session:
            db = NetworkDatabase(session)
            
            # Create monitoring service
            self.monitoring_service = NetworkMonitoringService(
                db=db,
                prometheus_pushgateway=PROMETHEUS_PUSHGATEWAY,
                export_interval=EXPORT_INTERVAL,
                snapshot_interval=SNAPSHOT_INTERVAL
            )
            
            # Start monitoring
            await self.monitoring_service.start()
            logger.info("Network monitoring service started")
            
            # Keep running until stopped
            while self._running:
                await asyncio.sleep(1)
            
            # Stop monitoring
            await self.monitoring_service.stop()
    
    async def shutdown(self):
        """Shutdown the exporter"""
        logger.info("Shutting down network metrics exporter...")
        self._running = False
        
        # Close database connections
        if self.engine:
            await self.engine.dispose()


async def main():
    """Main entry point"""
    exporter = NetworkMetricsExporter()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(exporter.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize
        await exporter.initialize()
        
        # Run
        await exporter.run()
        
    except Exception as e:
        logger.error(f"Exporter failed: {e}")
        raise
    finally:
        await exporter.shutdown()


if __name__ == "__main__":
    asyncio.run(main())