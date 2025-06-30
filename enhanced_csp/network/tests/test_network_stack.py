# enhanced_csp/network/tests/test_network_stack.py
"""
Comprehensive test suite for Enhanced CSP Network Stack
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from ..core.types import NetworkConfig, NodeID, P2PConfig, MeshConfig
from ..core.node import NetworkNode
from ..p2p.discovery import HybridDiscovery
from ..p2p.dht import KademliaDHT
from ..p2p.nat import NATTraversal, NATType
from ..mesh.topology import MeshTopologyManager
from ..mesh.routing import BatmanRouting
from ..dns.overlay import DNSOverlay, DNSRecordType
from ..routing.adaptive import AdaptiveRoutingEngine
from .. import create_network, EnhancedCSPNetwork


class TestNetworkStack:
    """Test the complete network stack"""
    
    @pytest.fixture
    async def network(self):
        """Create a test network instance"""
        config = NetworkConfig(