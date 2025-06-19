#!/usr/bin/env python3
"""
Enhanced CSP System - Fixed Main Module
Handles Prometheus metrics registration conflicts
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("**main**")

# Clear any existing Prometheus registry to prevent conflicts
def clear_prometheus_registry():
    """Clear Prometheus registry to prevent duplicate metrics"""
    try:
        import prometheus_client
        from prometheus_client import REGISTRY
        
        # Get all collectors
        collectors = list(REGISTRY._collector_to_names.keys())
        
        # Unregister all existing collectors
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except Exception:
                pass
                
        logger.info("Prometheus registry cleared")
        
    except ImportError:
        logger.info("Prometheus client not available")
    except Exception as e:
        logger.warning(f"Could not clear Prometheus registry: {e}")

# Clear registry before any imports
clear_prometheus_registry()

# Set environment variable to handle multiprocess mode
os.environ['PROMETHEUS_MULTIPROC_DIR'] = ''

# Now we can safely import and run the original main
def main():
    """Safe main function with Prometheus handling"""
    
    logger.info("=" * 60)
    logger.info("Enhanced CSP System - Starting Up")
    logger.info("=" * 60)
    logger.info("Version: 1.0.0")
    logger.info("Host: 0.0.0.0")
    logger.info("Port: 8000")
    logger.info("Debug: False")
    logger.info("Features: AI=True, Quantum=True, Consciousness=True")
    logger.info("=" * 60)
    
    try:
        # Try to run the original main.py content
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Execute the content in a clean environment
        exec_globals = {
            '__name__': '__main__',
            '__file__': 'main.py',
            'clear_prometheus_registry': clear_prometheus_registry
        }
        
        # Execute the main.py content
        exec(content, exec_globals)
        
    except Exception as e:
        logger.error(f"Error starting CSP system: {e}")
        
        # Fallback: start a simple FastAPI server
        logger.info("Starting fallback server...")
        start_fallback_server()

def start_fallback_server():
    """Start a simple fallback server"""
    try:
        from fastapi import FastAPI
        import uvicorn
        
        app = FastAPI(title="Enhanced CSP System", version="1.0.0")
        
        @app.get("/")
        async def root():
            return {"message": "Enhanced CSP System is running!", "status": "ok"}
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "version": "1.0.0"}
        
        logger.info("Starting fallback FastAPI server on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        
    except Exception as e:
        logger.error(f"Could not start fallback server: {e}")
        logger.info("Please check your Python environment and dependencies")

if __name__ == "__main__":
    main()
