"""Test Network Integration"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from backend.network_integration import CSPNetworkService, NetworkNode


@pytest.fixture
async def network_service():
    """Create test network service."""
    service = CSPNetworkService()
    
    # Mock network
    service.network = Mock()
    service.network.node_id = Mock()
    service.network.node_id.to_base58.return_value = "QmTest123"
    service.network.topology = Mock()
    service.network.topology.get_peers.return_value = []
    service.network.broadcast = AsyncMock()
    service.network.send_message = AsyncMock()
    service.network.config = Mock()
    service.network.config.node_capabilities = {"execution": True}
    
    service.is_initialized = True
    
    yield service
    
    service.is_initialized = False


@pytest.mark.asyncio
async def test_register_design(network_service):
    """Test design registration."""
    design_id = "design_123"
    
    await network_service.register_design(design_id)
    
    assert design_id in network_service.design_nodes
    assert "QmTest123" in network_service.design_nodes[design_id]
    
    # Should broadcast
    network_service.network.broadcast.assert_called_once()
