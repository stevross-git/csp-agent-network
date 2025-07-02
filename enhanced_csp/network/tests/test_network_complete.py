# network/tests/test_network_complete.py
"""
Comprehensive test suite for Enhanced CSP Network Stack.
Aims for >90% test coverage across all network modules.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any
import json

from ..core.types import (
    NetworkConfig, NodeID, NodeCapabilities, PeerInfo, 
    NetworkMessage, MessageType, P2PConfig, MeshConfig
)
from ..core.node import NetworkNode, EnhancedCSPNetwork
from ..core.config import SecurityConfig, DNSConfig, RoutingConfig
from ..p2p.discovery import HybridDiscovery, PeerType
from ..p2p.dht import KademliaDHT, KBucket, RoutingTable
from ..p2p.nat import NATTraversal, NATType
from ..p2p.transport import MultiProtocolTransport
from ..mesh.topology import MeshTopologyManager, TopologyType
from ..mesh.routing import BatmanRouting
from ..dns.overlay import DNSOverlay, DNSRecordType
from ..routing.adaptive import AdaptiveRoutingEngine, PathMetrics
from ..protocol_optimizer import BinaryProtocol, MessageType as BinaryMessageType
from .. import create_network, create_node


class TestCoreTypes:
    """Test core network types."""
    
    def test_node_id_generation(self):
        """Test NodeID generation and string conversion."""
        node_id = NodeID.generate()
        assert node_id.value.startswith("Qm")
        assert len(node_id.value) == 46
        
        # Test from_string
        node_id2 = NodeID.from_string(node_id.value)
        assert node_id.value == node_id2.value
    
    def test_node_capabilities(self):
        """Test NodeCapabilities dataclass."""
        caps = NodeCapabilities(
            relay=True,
            storage=True,
            quantum=True
        )
        assert caps.relay is True
        assert caps.storage is True
        assert caps.compute is False  # Default
    
    def test_network_message_creation(self):
        """Test NetworkMessage creation."""
        sender = NodeID.generate()
        msg = NetworkMessage.create(
            msg_type=MessageType.DATA,
            sender=sender,
            payload={"test": "data"},
            ttl=32
        )
        
        assert msg.type == MessageType.DATA
        assert msg.sender == sender
        assert msg.payload == {"test": "data"}
        assert msg.ttl == 32
        assert msg.recipient is None


class TestNetworkNode:
    """Test NetworkNode implementation."""
    
    @pytest.fixture
    async def node(self):
        """Create a test node."""
        config = NetworkConfig()
        node = NetworkNode(config)
        yield node
        if node.is_running:
            await node.stop()
    
    @pytest.mark.asyncio
    async def test_node_lifecycle(self, node):
        """Test node start/stop lifecycle."""
        # Start node
        assert not node.is_running
        success = await node.start()
        assert success
        assert node.is_running
        
        # Try starting again
        success = await node.start()
        assert success  # Should return True but not restart
        
        # Stop node
        success = await node.stop()
        assert success
        assert not node.is_running
    
    @pytest.mark.asyncio
    async def test_node_stats(self, node):
        """Test node statistics tracking."""
        await node.start()
        
        stats = node.get_stats()
        assert stats["messages_sent"] == 0
        assert stats["messages_received"] == 0
        assert stats["peer_count"] == 0
        assert stats["uptime"] >= 0
    
    @pytest.mark.asyncio
    async def test_message_handlers(self, node):
        """Test message handler registration."""
        handler_called = False
        test_message = None
        
        async def test_handler(msg):
            nonlocal handler_called, test_message
            handler_called = True
            test_message = msg
        
        # Register handler
        node.register_handler(MessageType.DATA, test_handler)
        
        # Verify handler is registered
        assert MessageType.DATA in node._message_handlers
        assert test_handler in node._message_handlers[MessageType.DATA]
        
        # Unregister handler
        node.unregister_handler(MessageType.DATA, test_handler)
        assert test_handler not in node._message_handlers.get(MessageType.DATA, [])


class TestHybridDiscovery:
    """Test hybrid peer discovery."""
    
    @pytest.fixture
    def discovery(self):
        """Create discovery instance."""
        config = P2PConfig()
        node_id = NodeID.generate()
        return HybridDiscovery(config, node_id)
    
    @pytest.mark.asyncio
    async def test_discovery_lifecycle(self, discovery):
        """Test discovery start/stop."""
        await discovery.start()
        assert discovery.is_running
        
        await discovery.stop()
        assert not discovery.is_running
    
    @pytest.mark.asyncio
    async def test_peer_announcement(self, discovery):
        """Test peer announcement."""
        peer_discovered = False
        discovered_peer = None
        
        async def on_peer_discovered(peer_info):
            nonlocal peer_discovered, discovered_peer
            peer_discovered = True
            discovered_peer = peer_info
        
        discovery.on_peer_discovered = on_peer_discovered
        
        # Announce a peer
        peer_info = {
            "node_id": "QmTest123",
            "addresses": ["/ip4/127.0.0.1/tcp/9000"],
            "source": "test"
        }
        
        await discovery.announce_peer(peer_info)
        
        assert peer_discovered
        assert discovered_peer == peer_info
        assert "QmTest123" in discovery.discovered_peers
    
    def test_multiaddr_parsing(self, discovery):
        """Test multiaddr parsing."""
        addr = "/ip4/192.168.1.1/tcp/4001/p2p/QmNodeID123"
        peer_info = discovery._parse_multiaddr(addr)
        
        assert peer_info is not None
        assert peer_info["node_id"] == "QmNodeID123"
        assert addr in peer_info["addresses"]


class TestKademliaDHT:
    """Test Kademlia DHT implementation."""
    
    @pytest.fixture
    async def dht(self):
        """Create DHT instance."""
        node_id = NodeID.generate()
        transport = Mock(spec=MultiProtocolTransport)
        transport.send = AsyncMock(return_value=True)
        transport.register_handler = Mock()
        
        dht = KademliaDHT(node_id, transport)
        yield dht
        if dht.is_running:
            await dht.stop()
    
    @pytest.mark.asyncio
    async def test_dht_lifecycle(self, dht):
        """Test DHT start/stop."""
        await dht.start()
        assert dht.is_running
        
        await dht.stop()
        assert not dht.is_running
    
    @pytest.mark.asyncio
    async def test_store_and_get(self, dht):
        """Test storing and retrieving values."""
        await dht.start()
        
        # Store a value
        key = "test_key"
        value = {"data": "test_value"}
        
        success = await dht.store(key, value)
        assert success
        
        # Retrieve the value
        retrieved = await dht.get(key)
        assert retrieved == value
    
    def test_routing_table(self):
        """Test routing table operations."""
        node_id = NodeID.generate()
        table = RoutingTable(node_id)
        
        # Add a peer
        peer_id = NodeID.generate()
        peer_info = PeerInfo(
            id=peer_id,
            address="127.0.0.1",
            port=9000,
            capabilities=NodeCapabilities(),
            last_seen=time.time()
        )
        
        success = table.add_node(peer_info)
        assert success
        
        # Find closest nodes
        target = NodeID.generate()
        closest = table.find_closest_nodes(target, 5)
        assert len(closest) <= 5
    
    def test_k_bucket(self):
        """Test k-bucket functionality."""
        bucket = KBucket(0, 100, k=3)
        
        # Add nodes
        for i in range(5):
            node_id = NodeID.generate()
            peer_info = PeerInfo(
                id=node_id,
                address=f"127.0.0.{i}",
                port=9000 + i,
                capabilities=NodeCapabilities(),
                last_seen=time.time()
            )
            entry = bucket.add_node(peer_info)
            
            # First 3 should succeed
            if i < 3:
                assert bucket.add_node(entry)
            else:
                # Bucket full, should fail
                assert not bucket.add_node(entry)
        
        assert len(bucket.entries) == 3


class TestProtocolOptimizer:
    """Test binary protocol implementation."""
    
    def test_message_encoding_decoding(self):
        """Test message encode/decode cycle."""
        protocol = BinaryProtocol()
        
        test_messages = [
            (BinaryMessageType.PING, {"action": "ping"}),
            (BinaryMessageType.DATA, {"data": "test", "nested": {"value": 123}}),
            (BinaryMessageType.CONTROL, {"command": "update", "params": {}}),
        ]
        
        for msg_type, payload in test_messages:
            # Encode
            encoded = protocol.encode_message(payload, msg_type)
            assert isinstance(encoded, bytes)
            assert len(encoded) >= protocol.HEADER_SIZE
            
            # Decode
            decoded, decoded_type, flags = protocol.decode_message(encoded)
            assert decoded_type == msg_type
            
            # Verify payload
            for key, value in payload.items():
                assert decoded[key] == value
    
    def test_header_only_decode(self):
        """Test header-only decoding for routing."""
        protocol = BinaryProtocol()
        
        message = {"test": "data"}
        encoded = protocol.encode_message(message, BinaryMessageType.DATA)
        
        # Decode header only
        version, msg_type, flags, length = protocol.decode_header_only(
            encoded[:protocol.HEADER_SIZE]
        )
        
        assert version == protocol.version
        assert msg_type == BinaryMessageType.DATA
        assert length > 0


class TestMeshTopology:
    """Test mesh topology management."""
    
    @pytest.fixture
    async def topology(self):
        """Create topology manager."""
        node_id = NodeID.generate()
        config = MeshConfig()
        send_fn = AsyncMock()
        
        topology = MeshTopologyManager(node_id, config, send_fn)
        yield topology
        if topology.is_running:
            await topology.stop()
    
    @pytest.mark.asyncio
    async def test_topology_lifecycle(self, topology):
        """Test topology manager lifecycle."""
        await topology.start()
        assert topology.is_running
        
        await topology.stop()
        assert not topology.is_running
    
    @pytest.mark.asyncio
    async def test_peer_management(self, topology):
        """Test adding and removing peers."""
        await topology.start()
        
        # Add a peer
        peer_info = PeerInfo(
            id=NodeID.generate(),
            address="127.0.0.1",
            port=9000,
            capabilities=NodeCapabilities(),
            last_seen=time.time()
        )
        
        success = await topology.add_peer(peer_info)
        assert success
        assert peer_info.id in topology.peers
        
        # Get neighbors
        neighbors = topology.get_mesh_neighbors()
        assert peer_info.id in neighbors
        
        # Remove peer
        await topology.remove_peer(peer_info.id)
        assert peer_info.id not in topology.peers


class TestBatmanRouting:
    """Test BATMAN routing protocol."""
    
    @pytest.fixture
    async def routing(self):
        """Create routing instance."""
        node_id = NodeID.generate()
        topology = Mock(spec=MeshTopologyManager)
        topology.get_mesh_neighbors = Mock(return_value=[])
        send_fn = AsyncMock()
        
        routing = BatmanRouting(node_id, topology, send_fn)
        yield routing
        if routing.is_running:
            await routing.stop()
    
    @pytest.mark.asyncio
    async def test_routing_lifecycle(self, routing):
        """Test routing protocol lifecycle."""
        await routing.start()
        assert routing.is_running
        
        await routing.stop()
        assert not routing.is_running
    
    @pytest.mark.asyncio
    async def test_path_finding(self, routing):
        """Test finding paths to destinations."""
        # Add a route manually
        dest = NodeID.generate()
        next_hop = NodeID.generate()
        
        from ..mesh.routing import RoutingEntry
        entry = RoutingEntry(
            destination=dest,
            next_hop=next_hop,
            metric=1.0,
            sequence_number=1,
            last_updated=time.time(),
            path=[routing.node_id, next_hop, dest]
        )
        
        routing.routing_table[dest] = entry
        
        # Get path
        path = await routing.get_path(dest)
        assert path == entry.path
        
        # Get next hop
        hop = await routing.get_next_hop(dest)
        assert hop == next_hop


class TestDNSOverlay:
    """Test DNS overlay functionality."""
    
    @pytest.fixture
    async def dns(self):
        """Create DNS overlay."""
        node_id = NodeID.generate()
        config = DNSConfig()
        
        # Mock DHT
        dht = Mock(spec=KademliaDHT)
        dht.store = AsyncMock(return_value=True)
        dht.get = AsyncMock(return_value=None)
        
        dns = DNSOverlay(node_id, config, dht)
        yield dns
        if dns.is_running:
            await dns.stop()
    
    @pytest.mark.asyncio
    async def test_dns_lifecycle(self, dns):
        """Test DNS overlay lifecycle."""
        await dns.start()
        assert dns.is_running
        
        await dns.stop()
        assert not dns.is_running
    
    @pytest.mark.asyncio
    async def test_name_registration(self, dns):
        """Test registering DNS names."""
        await dns.start()
        
        # Register a name
        success = await dns.register("mynode")
        assert success
        assert "mynode.web4ai" in dns.registered_names
        
        # Verify DHT store was called
        dns.dht.store.assert_called()
    
    @pytest.mark.asyncio
    async def test_name_resolution(self, dns):
        """Test resolving DNS names."""
        await dns.start()
        
        # Mock DHT response
        dns.dht.get.return_value = {
            "name": "test.web4ai",
            "type": "A",
            "value": "QmTestNodeID",
            "ttl": 3600,
            "created": time.time()
        }
        
        # Resolve name
        result = await dns.resolve("test")
        assert result == "QmTestNodeID"


class TestAdaptiveRouting:
    """Test adaptive routing engine."""
    
    @pytest.fixture
    async def adaptive(self):
        """Create adaptive routing engine."""
        node_id = NodeID.generate()
        
        # Mock base routing
        base_routing = Mock(spec=BatmanRouting)
        base_routing.get_path = AsyncMock(return_value=[node_id, NodeID.generate()])
        base_routing.get_alternative_paths = AsyncMock(return_value=[])
        base_routing.get_active_destinations = AsyncMock(return_value=[])
        base_routing.set_multipath_weights = AsyncMock()
        
        config = RoutingConfig()
        engine = AdaptiveRoutingEngine(node_id, base_routing, config)
        yield engine
        if engine.is_running:
            await engine.stop()
    
    @pytest.mark.asyncio
    async def test_adaptive_lifecycle(self, adaptive):
        """Test adaptive routing lifecycle."""
        await adaptive.start()
        assert adaptive.is_running
        
        await adaptive.stop()
        assert not adaptive.is_running
    
    @pytest.mark.asyncio
    async def test_best_path_selection(self, adaptive):
        """Test selecting best path."""
        dest = NodeID.generate()
        
        # Get best path (should use base routing)
        path = await adaptive.get_best_path(dest)
        assert path is not None
        adaptive.base_routing.get_path.assert_called_with(dest)
    
    @pytest.mark.asyncio
    async def test_path_metrics_reporting(self, adaptive):
        """Test reporting path metrics."""
        path = [NodeID.generate() for _ in range(3)]
        metrics = PathMetrics(
            latency_ms=20.0,
            packet_loss=0.01,
            bandwidth_mbps=100.0,
            jitter_ms=2.0
        )
        
        await adaptive.report_path_metrics(path, metrics)
        
        # Verify metrics stored
        assert len(adaptive.path_metrics) > 0
    
    def test_path_score_calculation(self):
        """Test path quality scoring."""
        metrics = PathMetrics(
            latency_ms=10.0,  # Good latency
            packet_loss=0.01,  # 1% loss
            bandwidth_mbps=100.0,  # Good bandwidth
            jitter_ms=1.0  # Low jitter
        )
        
        score = metrics.score()
        assert 0 <= score <= 100
        assert score > 80  # Should be a good score


class TestEnhancedCSPNetwork:
    """Test high-level network interface."""
    
    @pytest.fixture
    async def network(self):
        """Create network instance."""
        config = NetworkConfig()
        network = EnhancedCSPNetwork(config)
        yield network
        await network.stop_all()
    
    @pytest.mark.asyncio
    async def test_node_creation(self, network):
        """Test creating network nodes."""
        # Create a node
        node = await network.create_node("test_node")
        assert node is not None
        assert node.is_running
        assert "test_node" in network.nodes
        
        # Get node by name
        retrieved = network.get_node("test_node")
        assert retrieved == node
        
        # Stop node
        success = await network.stop_node("test_node")
        assert success
        assert "test_node" not in network.nodes
    
    @pytest.mark.asyncio
    async def test_multiple_nodes(self, network):
        """Test creating multiple nodes."""
        # Create multiple nodes
        nodes = []
        for i in range(3):
            node = await network.create_node(f"node_{i}")
            nodes.append(node)
        
        assert len(network.nodes) == 3
        
        # Stop all nodes
        await network.stop_all()
        assert len(network.nodes) == 0


class TestIntegration:
    """Integration tests for the complete network stack."""
    
    @pytest.mark.asyncio
    async def test_full_network_stack(self):
        """Test creating a functional network with all components."""
        # Create network with custom config
        config = NetworkConfig(
            p2p=P2PConfig(
                listen_port=9001,
                enable_mdns=False,  # Disable for testing
                bootstrap_nodes=[]
            ),
            mesh=MeshConfig(
                topology_type="partial_mesh",
                max_peers=10
            ),
            enable_dns=True,
            enable_adaptive_routing=True
        )
        
        network = create_network(config)
        
        try:
            # Create nodes
            node1 = await network.create_node("node1")
            node2 = await network.create_node("node2")
            
            assert node1.is_running
            assert node2.is_running
            
            # Verify components are initialized
            assert node1.transport is not None
            assert node1.discovery is not None
            assert node1.dht is not None
            assert node1.topology is not None
            assert node1.routing is not None
            assert node1.dns is not None
            assert node1.adaptive_routing is not None
            
        finally:
            await network.stop_all()
    
    @pytest.mark.asyncio
    async def test_message_exchange(self):
        """Test message exchange between nodes."""
        # This would require more complex setup with actual networking
        # For now, we test the message creation and handling pipeline
        
        config = NetworkConfig()
        node = NetworkNode(config)
        
        # Mock transport
        node.transport = Mock(spec=MultiProtocolTransport)
        node.transport.send = AsyncMock(return_value=True)
        
        # Add a fake peer
        peer_id = NodeID.generate()
        peer_info = PeerInfo(
            id=peer_id,
            address="127.0.0.1:9000",
            port=9000,
            capabilities=NodeCapabilities(),
            last_seen=time.time()
        )
        node.peers[peer_id] = peer_info
        
        # Send a message
        success = await node.send_message(peer_id, {"test": "data"})
        assert success
        node.transport.send.assert_called_once()


# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=network", "--cov-report=html"])