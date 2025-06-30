# test_enhanced_csp_integration.py
"""
Enhanced CSP Network Stack - Complete Integration Test Suite

Comprehensive end-to-end testing including discovery, mesh formation,
data transfer, DNS resolution, fault tolerance, and security validation.
"""

import asyncio
import pytest
import pytest_asyncio
import time
import random
import os
import tempfile
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch, AsyncMock
import logging
from dataclasses import dataclass
import socket
import ssl

# Test framework imports
from pytest_timeout import timeout
import aiofiles

# Import the Enhanced CSP components
from enhanced_csp.network import (
    NetworkNode, NetworkConfig, P2PConfig, MeshConfig, DNSConfig,
    create_network
)
from enhanced_csp.network.core.types import NodeID, RoutingConfig
from enhanced_csp.network.dns.overlay import DNSRecordType
from enhanced_csp.network.p2p.transport import TransportState

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Test configuration
TEST_BASE_PORT = 20000
TEST_TIMEOUT = 30


@dataclass
class TestNode:
    """Wrapper for test network nodes"""
    node: NetworkNode
    config: NetworkConfig
    test_id: int
    base_port: int
    
    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.base_port}"
    
    @property 
    def node_id(self) -> NodeID:
        return self.node.node_id


class NetworkTestUtils:
    """Utilities for network testing"""
    
    @staticmethod
    async def create_test_nodes(
        count: int, 
        base_port: int = TEST_BASE_PORT,
        config_overrides: Dict[str, Any] = None
    ) -> List[TestNode]:
        """Create multiple test nodes with unique configurations"""
        nodes = []
        
        for i in range(count):
            # Create temp directory for node data
            data_dir = tempfile.mkdtemp(prefix=f"test_node_{i}_")
            key_path = os.path.join(data_dir, "node.key")
            
            # Configure node
            port = base_port + (i * 2)  # Space for QUIC and TCP
            config = NetworkConfig(
                node_key_path=key_path,
                data_dir=data_dir,
                p2p=P2PConfig(
                    listen_address="127.0.0.1",
                    listen_port=port,
                    enable_quic=True,
                    enable_dht=True,
                    bootstrap_nodes=[] if i == 0 else [f"127.0.0.1:{base_port}"],
                    stun_servers=["stun:127.0.0.1:23478"],  # Local test STUN
                    connection_timeout=5
                ),
                mesh=MeshConfig(
                    topology_type="dynamic_partial",
                    enable_super_peers=True,
                    max_peers=10,
                    routing_update_interval=2
                ),
                dns=DNSConfig(
                    root_domain=".test",
                    enable_dnssec=True,
                    cache_size=1000
                ),
                routing=RoutingConfig(
                    enable_multipath=True,
                    enable_ml_predictor=False,  # Disable for tests
                    metric_update_interval=1
                )
            )
            
            # Apply overrides
            if config_overrides:
                for key, value in config_overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            # Create node
            node = NetworkNode(config)
            test_node = TestNode(
                node=node,
                config=config,
                test_id=i,
                base_port=port
            )
            nodes.append(test_node)
        
        return nodes
    
    @staticmethod
    async def wait_for_condition(
        condition_func,
        timeout_seconds: float = 10,
        poll_interval: float = 0.1,
        message: str = "Condition not met"
    ) -> bool:
        """Wait for a condition to become true"""
        start = time.time()
        
        while time.time() - start < timeout_seconds:
            try:
                result = await condition_func()
                if result:
                    return True
            except Exception as e:
                logger.debug(f"Condition check failed: {e}")
            
            await asyncio.sleep(poll_interval)
        
        raise TimeoutError(f"{message} after {timeout_seconds}s")
    
    @staticmethod
    def generate_test_data(size_bytes: int) -> bytes:
        """Generate deterministic test data"""
        return b"TEST_DATA_" + (b"X" * (size_bytes - 10))
    
    @staticmethod
    async def inject_packet_loss(node: NetworkNode, loss_rate: float):
        """Inject packet loss for testing"""
        original_send = node.transport.send
        
        async def lossy_send(data: bytes, destination: NodeID):
            if random.random() < loss_rate:
                logger.debug(f"Simulated packet loss to {destination}")
                return None
            return await original_send(data, destination)
        
        node.transport.send = lossy_send


@pytest_asyncio.fixture
async def test_nodes():
    """Fixture to create and cleanup test nodes"""
    nodes = []
    
    async def _create_nodes(count: int, **kwargs) -> List[TestNode]:
        created = await NetworkTestUtils.create_test_nodes(count, **kwargs)
        nodes.extend(created)
        return created
    
    yield _create_nodes
    
    # Cleanup
    for test_node in nodes:
        try:
            if test_node.node.is_running:
                await test_node.node.stop()
            # Clean temp directories
            if os.path.exists(test_node.config.data_dir):
                import shutil
                shutil.rmtree(test_node.config.data_dir)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


@pytest.mark.asyncio
class TestDiscoveryMechanisms:
    """Test all discovery mechanisms: mDNS, bootstrap, DHT"""
    
    @timeout(TEST_TIMEOUT)
    async def test_bootstrap_discovery(self, test_nodes):
        """Test discovery via bootstrap nodes"""
        # Create 5 nodes
        nodes = await test_nodes(5)
        
        # Start bootstrap node first
        await nodes[0].node.start()
        logger.info(f"Bootstrap node started: {nodes[0].node_id}")
        
        # Brief delay for bootstrap to initialize
        await asyncio.sleep(1)
        
        # Start other nodes
        for node in nodes[1:]:
            await node.node.start()
            logger.info(f"Node {node.test_id} started")
        
        # Wait for all nodes to discover each other via bootstrap
        async def all_discovered():
            for node in nodes:
                discovered = node.node.discovery.get_discovered_peers()
                if len(discovered) < len(nodes) - 1:
                    return False
            return True
        
        await NetworkTestUtils.wait_for_condition(
            all_discovered,
            timeout_seconds=15,
            message="Nodes failed to discover each other"
        )
        
        # Verify all connections
        for node in nodes:
            peers = node.node.discovery.get_discovered_peers()
            assert len(peers) >= len(nodes) - 1
            logger.info(f"Node {node.test_id} discovered {len(peers)} peers")
    
    @timeout(TEST_TIMEOUT)
    async def test_dht_discovery(self, test_nodes):
        """Test DHT-based peer discovery"""
        # Create 10 nodes for DHT testing
        nodes = await test_nodes(10)
        
        # Start nodes sequentially to test DHT propagation
        for i, node in enumerate(nodes):
            await node.node.start()
            
            # Announce to DHT after starting
            if i > 0:
                await node.node.dht.announce(
                    node.node_id.to_bytes(),
                    {"address": node.address, "port": node.base_port}
                )
        
        # Allow DHT to stabilize
        await asyncio.sleep(5)
        
        # Test random lookups
        successful_lookups = 0
        total_lookups = 20
        
        for _ in range(total_lookups):
            searcher = random.choice(nodes)
            target = random.choice(nodes)
            
            if searcher != target:
                result = await searcher.node.dht.find_node(
                    target.node_id.to_bytes()
                )
                if result:
                    successful_lookups += 1
        
        # Should have high success rate
        success_rate = successful_lookups / total_lookups
        assert success_rate >= 0.8, f"DHT lookup success rate too low: {success_rate}"
        logger.info(f"DHT lookup success rate: {success_rate:.2%}")
    
    @timeout(TEST_TIMEOUT)
    async def test_mdns_local_discovery(self, test_nodes):
        """Test mDNS local network discovery"""
        # Create 3 nodes on same network
        nodes = await test_nodes(3)
        
        # Start all nodes simultaneously
        await asyncio.gather(*[node.node.start() for node in nodes])
        
        # Wait for mDNS discovery
        await asyncio.sleep(5)
        
        # Each node should discover others via mDNS
        for node in nodes:
            mdns_peers = node.node.discovery.get_mdns_peers()
            assert len(mdns_peers) >= len(nodes) - 1
            logger.info(f"Node {node.test_id} found {len(mdns_peers)} via mDNS")


@pytest.mark.asyncio
class TestMeshNetworking:
    """Test mesh topology formation and routing"""
    
    @timeout(TEST_TIMEOUT)
    async def test_partial_mesh_formation(self, test_nodes):
        """Test dynamic partial mesh topology"""
        # Create 7 nodes
        nodes = await test_nodes(7)
        
        # Start all nodes
        for node in nodes:
            await node.node.start()
        
        # Wait for mesh formation
        await asyncio.sleep(10)
        
        # Verify partial mesh properties
        peer_counts = []
        for node in nodes:
            peers = node.node.topology.get_active_peers()
            peer_counts.append(len(peers))
            
            # Each node should have 3-6 peers (partial mesh)
            assert 3 <= len(peers) <= 6, f"Node {node.test_id} has {len(peers)} peers"
        
        avg_peers = sum(peer_counts) / len(peer_counts)
        logger.info(f"Average peers per node: {avg_peers:.1f}")
        
        # Test connectivity - any node can reach any other
        source = nodes[0]
        destination = nodes[-1]
        
        path = source.node.routing.find_path(destination.node_id)
        assert path is not None
        assert len(path) > 0
        logger.info(f"Path found: {len(path)} hops")
    
    @timeout(TEST_TIMEOUT)
    async def test_super_peer_election(self, test_nodes):
        """Test super peer election based on capacity"""
        # Create nodes with different capacities
        nodes = await test_nodes(5)
        
        # Set capacities before starting
        capacities = [100, 20, 95, 25, 90]  # Mbps
        for i, (node, capacity) in enumerate(zip(nodes, capacities)):
            node.node.capacity_score = capacity
        
        # Start all nodes
        for node in nodes:
            await node.node.start()
        
        # Wait for super peer election
        await asyncio.sleep(8)
        
        # Verify high capacity nodes became super peers
        super_peers = []
        for node in nodes:
            if node.node.topology.is_super_peer:
                super_peers.append(node.node.capacity_score)
                logger.info(f"Super peer: Node {node.test_id} (capacity: {node.node.capacity_score})")
        
        # Should elect 2-3 super peers with high capacity
        assert 2 <= len(super_peers) <= 3
        assert all(capacity >= 90 for capacity in super_peers)
    
    @timeout(TEST_TIMEOUT * 2)
    async def test_mesh_self_healing(self, test_nodes):
        """Test mesh recovery after node failure"""
        # Create 6 nodes
        nodes = await test_nodes(6)
        
        # Start all nodes
        for node in nodes:
            await node.node.start()
        
        # Wait for stable mesh
        await asyncio.sleep(8)
        
        # Record initial topology
        initial_connections = {}
        for node in nodes:
            initial_connections[node.test_id] = len(node.node.topology.get_active_peers())
        
        # Simulate node failure
        failed_node = nodes[2]
        failed_id = failed_node.node_id
        await failed_node.node.stop()
        logger.info(f"Node {failed_node.test_id} failed")
        
        # Wait for mesh to detect and heal
        await asyncio.sleep(15)
        
        # Verify mesh healed
        for node in nodes:
            if node.test_id != 2:  # Skip failed node
                peers = node.node.topology.get_active_peers()
                
                # Should maintain connectivity
                assert len(peers) >= 2
                
                # Failed node should be removed
                peer_ids = [p.node_id for p in peers]
                assert failed_id not in peer_ids
        
        logger.info("Mesh successfully healed after node failure")


@pytest.mark.asyncio
class TestDataTransfer:
    """Test QUIC/TCP transport and data transfer"""
    
    @timeout(TEST_TIMEOUT)
    async def test_quic_data_transfer(self, test_nodes):
        """Test QUIC connection and data transfer"""
        nodes = await test_nodes(2)
        
        await nodes[0].node.start()
        await nodes[1].node.start()
        
        # Wait for discovery
        await asyncio.sleep(2)
        
        # Send various sized messages
        test_sizes = [
            (100, "small"),      # 100 bytes
            (10_000, "medium"),  # 10 KB
            (1_000_000, "large") # 1 MB
        ]
        
        for size, label in test_sizes:
            test_data = NetworkTestUtils.generate_test_data(size)
            
            # Send data
            start = time.time()
            success = await nodes[0].node.send_data(
                nodes[1].node_id,
                test_data
            )
            duration = time.time() - start
            
            assert success
            throughput = (size * 8) / (duration * 1_000_000)  # Mbps
            logger.info(f"{label} transfer ({size} bytes): {throughput:.2f} Mbps")
    
    @timeout(TEST_TIMEOUT)
    async def test_tcp_fallback(self, test_nodes):
        """Test TCP fallback when QUIC unavailable"""
        # Disable QUIC to force TCP
        nodes = await test_nodes(
            2,
            config_overrides={"p2p": {"enable_quic": False}}
        )
        
        await nodes[0].node.start()
        await nodes[1].node.start()
        
        # Should connect via TCP
        await asyncio.sleep(2)
        
        # Verify TCP connection
        conn = nodes[0].node.transport.get_connection(nodes[1].node_id)
        assert conn is not None
        assert conn.protocol == "tcp"
        
        # Test data transfer over TCP
        test_data = NetworkTestUtils.generate_test_data(1000)
        success = await nodes[0].node.send_data(nodes[1].node_id, test_data)
        assert success
        
        logger.info("TCP fallback successful")
    
    @timeout(TEST_TIMEOUT)
    async def test_nat_traversal(self, test_nodes):
        """Test NAT traversal mechanisms"""
        nodes = await test_nodes(2)
        
        # Simulate NAT conditions
        nodes[0].node.nat.simulated_nat_type = "symmetric"
        nodes[1].node.nat.simulated_nat_type = "restricted_cone"
        
        await nodes[0].node.start()
        await nodes[1].node.start()
        
        # Attempt connection through NAT
        await asyncio.sleep(3)
        
        # Should establish connection despite NAT
        conn = nodes[0].node.transport.get_connection(nodes[1].node_id)
        assert conn is not None
        
        # Verify traversal method used
        traversal_method = nodes[0].node.nat.last_traversal_method
        assert traversal_method in ["stun", "turn", "hole_punch"]
        logger.info(f"NAT traversal via: {traversal_method}")


@pytest.mark.asyncio
class TestDNSOverlay:
    """Test distributed DNS functionality"""
    
    @timeout(TEST_TIMEOUT)
    async def test_dns_registration_resolution(self, test_nodes):
        """Test DNS record registration and resolution"""
        nodes = await test_nodes(4)
        
        # Start all nodes
        for node in nodes:
            await node.node.start()
        
        # Wait for DHT stabilization
        await asyncio.sleep(5)
        
        # Register DNS records from different nodes
        test_records = [
            ("service1.test", "192.168.1.10", DNSRecordType.A),
            ("service2.test", "2001:db8::1", DNSRecordType.AAAA),
            ("_http._tcp.api.test", "0 5 80 api.test", DNSRecordType.SRV),
            ("info.test", "v=spf1 include:test -all", DNSRecordType.TXT)
        ]
        
        for i, (name, data, record_type) in enumerate(test_records):
            node = nodes[i % len(nodes)]
            success = await node.node.dns.register_record(
                name, record_type, data, ttl=300
            )
            assert success
            logger.info(f"Registered {record_type.name} record: {name}")
        
        # Wait for DHT propagation
        await asyncio.sleep(3)
        
        # Resolve from different nodes
        for name, expected_data, record_type in test_records:
            # Try resolving from a random node
            resolver_node = random.choice(nodes)
            result = await resolver_node.node.dns.resolve(name, record_type)
            
            assert result is not None
            assert result.data == expected_data
            logger.info(f"Resolved {name}: {result.data}")
    
    @timeout(TEST_TIMEOUT)
    async def test_dnssec_validation(self, test_nodes):
        """Test DNSSEC signature validation"""
        nodes = await test_nodes(2)
        
        await nodes[0].node.start()
        await nodes[1].node.start()
        
        await asyncio.sleep(3)
        
        # Register signed record
        success = await nodes[0].node.dns.register_record(
            "secure.test",
            DNSRecordType.A,
            "10.0.0.1",
            ttl=300,
            sign=True
        )
        assert success
        
        # Wait for propagation
        await asyncio.sleep(2)
        
        # Resolve and verify DNSSEC
        result = await nodes[1].node.dns.resolve(
            "secure.test",
            DNSRecordType.A,
            validate_dnssec=True
        )
        
        assert result is not None
        assert result.dnssec_valid
        logger.info("DNSSEC validation successful")


@pytest.mark.asyncio
class TestAdaptiveRouting:
    """Test ML-based adaptive routing"""
    
    @timeout(TEST_TIMEOUT)
    async def test_multipath_routing(self, test_nodes):
        """Test multipath routing and load balancing"""
        # Create 8 nodes for multiple paths
        nodes = await test_nodes(8)
        
        for node in nodes:
            await node.node.start()
        
        # Wait for topology
        await asyncio.sleep(10)
        
        source = nodes[0]
        destination = nodes[-1]
        
        # Find multiple paths
        paths = source.node.adaptive_routing.find_multipath(
            destination.node_id,
            max_paths=3
        )
        
        assert len(paths) >= 2
        logger.info(f"Found {len(paths)} paths to destination")
        
        # Send data using multipath
        test_data = NetworkTestUtils.generate_test_data(100_000)  # 100KB
        
        chunks_sent = await source.node.adaptive_routing.send_multipath(
            destination.node_id,
            test_data
        )
        
        assert chunks_sent > 1
        logger.info(f"Data split across {chunks_sent} paths")
    
    @timeout(TEST_TIMEOUT)
    async def test_route_failover(self, test_nodes):
        """Test automatic route failover"""
        nodes = await test_nodes(5)
        
        for node in nodes:
            await node.node.start()
        
        await asyncio.sleep(8)
        
        source = nodes[0]
        destination = nodes[-1]
        
        # Start continuous data stream
        stream_task = asyncio.create_task(
            self._continuous_stream(source, destination)
        )
        
        # Let stream establish
        await asyncio.sleep(3)
        
        # Find current path
        initial_path = source.node.routing.find_path(destination.node_id)
        assert initial_path
        
        # Fail a node in the path (not source/dest)
        if len(initial_path) > 2:
            failed_node_id = initial_path[1]
            for node in nodes:
                if node.node_id == failed_node_id:
                    await node.node.stop()
                    logger.info(f"Failed intermediate node")
                    break
        
        # Wait for route adaptation
        await asyncio.sleep(5)
        
        # Verify new path established
        new_path = source.node.routing.find_path(destination.node_id)
        assert new_path is not None
        assert new_path != initial_path
        
        stream_task.cancel()
        logger.info("Route failover successful")
    
    async def _continuous_stream(self, source: TestNode, destination: TestNode):
        """Send continuous data stream"""
        while True:
            try:
                data = NetworkTestUtils.generate_test_data(1000)
                await source.node.send_data(destination.node_id, data)
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Stream error: {e}")


@pytest.mark.asyncio
class TestFaultTolerance:
    """Test resilience and fault tolerance"""
    
    @timeout(TEST_TIMEOUT)
    async def test_packet_loss_resilience(self, test_nodes):
        """Test communication under packet loss"""
        nodes = await test_nodes(3)
        
        for node in nodes:
            await node.node.start()
        
        await asyncio.sleep(5)
        
        # Inject packet loss
        await NetworkTestUtils.inject_packet_loss(nodes[1], 0.2)  # 20% loss
        
        # Test connectivity
        success_count = 0
        total_attempts = 50
        
        for i in range(total_attempts):
            test_data = f"Test message {i}".encode()
            try:
                success = await nodes[0].node.send_data(
                    nodes[2].node_id,
                    test_data,
                    timeout=2.0
                )
                if success:
                    success_count += 1
            except Exception:
                pass
        
        success_rate = success_count / total_attempts
        assert success_rate >= 0.7  # Should maintain 70%+ success despite 20% loss
        logger.info(f"Success rate with 20% packet loss: {success_rate:.1%}")
    
    @timeout(TEST_TIMEOUT * 2)
    async def test_network_partition_recovery(self, test_nodes):
        """Test recovery from network partition"""
        nodes = await test_nodes(6)
        
        for node in nodes:
            await node.node.start()
        
        await asyncio.sleep(8)
        
        # Create partition: nodes 0-2 | nodes 3-5
        group1 = nodes[:3]
        group2 = nodes[3:]
        
        # Simulate partition by blocking cross-group communication
        for g1_node in group1:
            g1_node.node._partition_blocked = {n.node_id for n in group2}
        
        for g2_node in group2:
            g2_node.node._partition_blocked = {n.node_id for n in group1}
        
        logger.info("Network partition created")
        
        # Wait for detection
        await asyncio.sleep(10)
        
        # Remove partition
        for node in nodes:
            node.node._partition_blocked = set()
        
        logger.info("Network partition healed")
        
        # Wait for recovery
        await asyncio.sleep(15)
        
        # Verify full mesh reformed
        for node in nodes:
            peers = node.node.topology.get_active_peers()
            assert len(peers) >= 3
        
        # Test cross-partition communication
        success = await nodes[0].node.send_data(
            nodes[5].node_id,
            b"Post-partition message"
        )
        assert success
        logger.info("Network recovered from partition")
    
    @timeout(TEST_TIMEOUT)
    async def test_byzantine_node_detection(self, test_nodes):
        """Test detection and isolation of byzantine nodes"""
        nodes = await test_nodes(5)
        
        for node in nodes:
            await node.node.start()
        
        await asyncio.sleep(5)
        
        # Make node 2 byzantine
        byzantine_node = nodes[2]
        
        # Inject malicious behavior
        async def byzantine_behavior():
            """Send invalid messages to disrupt network"""
            for _ in range(50):
                for target in nodes:
                    if target != byzantine_node:
                        # Send malformed data
                        try:
                            await byzantine_node.node.transport._send_raw(
                                b"MALICIOUS\x00\xff\xfe" * 100,
                                target.node_id
                            )
                        except:
                            pass
                await asyncio.sleep(0.1)
        
        # Start byzantine behavior
        byzantine_task = asyncio.create_task(byzantine_behavior())
        
        # Wait for detection
        await asyncio.sleep(20)
        
        byzantine_task.cancel()
        
        # Verify byzantine node is isolated
        isolated_count = 0
        for node in nodes:
            if node != byzantine_node:
                peers = node.node.topology.get_active_peers()
                peer_ids = {p.node_id for p in peers}
                if byzantine_node.node_id not in peer_ids:
                    isolated_count += 1
        
        # Most nodes should have isolated the byzantine node
        assert isolated_count >= 3
        logger.info(f"Byzantine node isolated by {isolated_count}/4 nodes")


@pytest.mark.asyncio
class TestSecurity:
    """Test security features and attack resistance"""
    
    @timeout(TEST_TIMEOUT)
    async def test_tls_certificate_validation(self, test_nodes):
        """Test proper TLS certificate validation"""
        nodes = await test_nodes(2)
        
        # Start first node
        await nodes[0].node.start()
        
        # Try to connect with invalid certificate
        nodes[1].node.transport._generate_self_signed_cert()
        
        # Tamper with certificate
        nodes[1].node.transport._tls_cert = b"INVALID_CERT"
        
        # Should fail to connect
        with pytest.raises(ssl.SSLError):
            await nodes[1].node.start()
    
    @timeout(TEST_TIMEOUT)
    async def test_node_authentication(self, test_nodes):
        """Test Ed25519 node identity verification"""
        nodes = await test_nodes(2)
        
        await nodes[0].node.start()
        await nodes[1].node.start()
        
        # Try to impersonate another node
        fake_node_id = NodeID.generate()
        
        # Attempt connection with wrong identity
        with pytest.raises(Exception) as exc_info:
            await nodes[0].node.transport.connect_as(
                nodes[1].address,
                fake_node_id  # Wrong identity
            )
        
        assert "identity" in str(exc_info.value).lower()
        logger.info("Node identity verification working")
    
    @timeout(TEST_TIMEOUT)
    async def test_dos_protection(self, test_nodes):
        """Test DoS attack resistance"""
        target = await test_nodes(1)
        await target[0].node.start()
        
        # Create attacker nodes
        attackers = await test_nodes(10, base_port=TEST_BASE_PORT + 100)
        
        # Launch connection flood
        flood_tasks = []
        for attacker in attackers:
            await attacker.node.start()
            
            # Each attacker tries 100 connections
            for _ in range(100):
                task = asyncio.create_task(
                    attacker.node.transport.connect(target[0].address)
                )
                flood_tasks.append(task)
        
        # Wait for flood
        results = await asyncio.gather(*flood_tasks, return_exceptions=True)
        
        # Count successful connections
        successful = sum(1 for r in results if r and not isinstance(r, Exception))
        
        # Should rate limit connections
        assert successful < 200  # Much less than 1000 attempts
        logger.info(f"DoS protection: {successful}/1000 connections allowed")
        
        # Legitimate connection should still work
        legitimate = await test_nodes(1, base_port=TEST_BASE_PORT + 200)
        await legitimate[0].node.start()
        
        # Brief delay for rate limit to clear
        await asyncio.sleep(2)
        
        success = await legitimate[0].node.transport.connect(target[0].address)
        assert success is not None
        logger.info("Legitimate connection succeeded after attack")
    
    @timeout(TEST_TIMEOUT)
    async def test_message_encryption(self, test_nodes):
        """Test end-to-end message encryption"""
        nodes = await test_nodes(2)
        
        await nodes[0].node.start()
        await nodes[1].node.start()
        
        await asyncio.sleep(2)
        
        # Send sensitive data
        sensitive_data = b"SECRET: API_KEY=sk-1234567890abcdef"
        
        # Intercept transport to verify encryption
        intercepted_data = []
        
        original_send = nodes[0].node.transport._send_raw
        async def intercept_send(data, dest):
            intercepted_data.append(data)
            return await original_send(data, dest)
        
        nodes[0].node.transport._send_raw = intercept_send
        
        # Send encrypted message
        success = await nodes[0].node.send_data(
            nodes[1].node_id,
            sensitive_data,
            encrypted=True
        )
        assert success
        
        # Verify data was encrypted in transit
        assert len(intercepted_data) > 0
        for data in intercepted_data:
            # Should not contain plaintext secret
            assert b"API_KEY=sk-1234567890abcdef" not in data
        
        logger.info("Message encryption verified")


@pytest.mark.asyncio
class TestPerformance:
    """Performance and stress tests"""
    
    @timeout(60)
    async def test_connection_scalability(self, test_nodes):
        """Test handling many concurrent connections"""
        # Create one server node
        server = await test_nodes(1)
        await server[0].node.start()
        
        # Create 50 client nodes
        clients = await test_nodes(50, base_port=TEST_BASE_PORT + 200)
        
        # Connect all clients concurrently
        connect_tasks = []
        for client in clients:
            await client.node.start()
            task = asyncio.create_task(
                client.node.transport.connect(server[0].address)
            )
            connect_tasks.append(task)
        
        # Wait for all connections
        start = time.time()
        results = await asyncio.gather(*connect_tasks, return_exceptions=True)
        duration = time.time() - start
        
        # Count successes
        successful = sum(1 for r in results if r and not isinstance(r, Exception))
        
        assert successful >= 45  # 90% success rate
        logger.info(f"Connected {successful}/50 clients in {duration:.2f}s")
    
    @timeout(30)
    async def test_message_throughput(self, test_nodes):
        """Test sustained message throughput"""
        nodes = await test_nodes(2)
        
        await nodes[0].node.start()
        await nodes[1].node.start()
        
        await asyncio.sleep(2)
        
        # Send many small messages
        message_count = 10000
        message_size = 100
        
        start = time.time()
        
        for i in range(message_count):
            data = f"Message {i}".encode().ljust(message_size, b'0')
            success = await nodes[0].node.send_data(
                nodes[1].node_id,
                data,
                wait_for_ack=False  # Fire and forget
            )
            if not success:
                logger.warning(f"Failed to send message {i}")
        
        duration = time.time() - start
        
        messages_per_second = message_count / duration
        throughput_mbps = (message_count * message_size * 8) / (duration * 1_000_000)
        
        logger.info(f"Throughput: {messages_per_second:.0f} msg/s, {throughput_mbps:.2f} Mbps")
        
        # Should achieve reasonable throughput
        assert messages_per_second > 1000
    
    @timeout(60)
    async def test_large_file_transfer(self, test_nodes):
        """Test large file transfer performance"""
        nodes = await test_nodes(2)
        
        await nodes[0].node.start()
        await nodes[1].node.start()
        
        await asyncio.sleep(2)
        
        # Create 10MB test file
        file_size = 10 * 1024 * 1024
        test_data = os.urandom(file_size)
        
        # Calculate checksum
        checksum = hashlib.sha256(test_data).hexdigest()
        
        # Transfer file
        start = time.time()
        
        success = await nodes[0].node.transfer_file(
            nodes[1].node_id,
            test_data,
            filename="test_10mb.bin"
        )
        
        duration = time.time() - start
        
        assert success
        
        throughput_mbps = (file_size * 8) / (duration * 1_000_000)
        logger.info(f"File transfer: {file_size/(1024*1024):.1f}MB in {duration:.2f}s")
        logger.info(f"Throughput: {throughput_mbps:.2f} Mbps")
        
        # Should achieve reasonable speed
        assert throughput_mbps > 10  # At least 10 Mbps


# Test execution configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([
        __file__,
        "-v",                    # Verbose
        "-s",                    # Show print statements
        "--tb=short",            # Short traceback
        "--asyncio-mode=auto",   # Auto async mode
        "-x",                    # Stop on first failure
        "--timeout=300",         # Global timeout
        "-k", "not Performance"  # Skip performance tests in quick run
    ])
