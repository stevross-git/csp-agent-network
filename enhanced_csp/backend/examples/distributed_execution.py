#!/usr/bin/env python3
"""
Example: Distributed Design Execution

Shows how to use network features for distributed execution.
"""

import asyncio
import os
from pathlib import Path
import sys

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.network_integration import CSPNetworkService
from enhanced_csp.network import NetworkConfig, P2PConfig


async def main():
    """Run distributed execution example."""
    # Enable network
    os.environ["NETWORK_ENABLED"] = "true"
    os.environ["NETWORK_PORT"] = "30301"
    
    # Create service
    service = CSPNetworkService()
    
    try:
        await service.initialize()
        
        print(f"Node ID: {service.network.node_id.to_base58()}")
        
        # Register a design
        design_id = "example_design_123"
        await service.register_design(design_id)
        print(f"Registered design: {design_id}")
        
        # Wait a bit
        await asyncio.sleep(5)
        
        # Find nodes
        nodes = await service.find_design_nodes(design_id)
        print(f"Nodes with design: {nodes}")
        
    finally:
        await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
