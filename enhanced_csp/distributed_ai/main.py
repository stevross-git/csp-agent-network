"""
Main entry point for Distributed AI Layer
"""

import asyncio
import argparse
import logging
from pathlib import Path

from .csp_integration_config import DistributedAIConfig, CSPIntegrationLayer, DeploymentEnvironment

async def main():
    parser = argparse.ArgumentParser(description="Distributed AI Layer")
    parser.add_argument("--config", default="configs/development.yaml", help="Configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--environment", default="development", help="Deployment environment")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = DistributedAIConfig.from_yaml(str(config_path))
    else:
        config = DistributedAIConfig(
            environment=DeploymentEnvironment(args.environment)
        )
    
    # Start the system
    csp_layer = CSPIntegrationLayer(config)
    await csp_layer.start()
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        await csp_layer.stop()

if __name__ == "__main__":
    asyncio.run(main())