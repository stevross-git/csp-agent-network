# network/database_models.py
"""
Network Database Models and Integration
======================================
SQLAlchemy models for the network database schema
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, 
    ForeignKey, ARRAY, JSON, Numeric, BigInteger, Index,
    CheckConstraint, UniqueConstraint, func
)
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

Base = declarative_base()

# ============================================================================
# ENUMS
# ============================================================================

class NodeType(str, PyEnum):
    PEER = "peer"
    RELAY = "relay"
    GATEWAY = "gateway"
    BOOTSTRAP = "bootstrap"

class NodeRole(str, PyEnum):
    PEER = "peer"
    SUPER_PEER = "super_peer"
    RELAY = "relay"
    GATEWAY = "gateway"
    BOOTSTRAP = "bootstrap"
    COORDINATOR = "coordinator"
    WITNESS = "witness"

class NodeStatus(str, PyEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    QUARANTINED = "quarantined"

class LinkState(str, PyEnum):
    ESTABLISHING = "establishing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    CONGESTED = "congested"
    FAILING = "failing"
    DORMANT = "dormant"
    QUARANTINED = "quarantined"

class TopologyType(str, PyEnum):
    FULL_MESH = "full_mesh"
    PARTIAL_MESH = "partial_mesh"
    DYNAMIC_PARTIAL = "dynamic_partial"
    HIERARCHICAL = "hierarchical"
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"
    QUANTUM_INSPIRED = "quantum_inspired"
    NEURAL_MESH = "neural_mesh"
    ADAPTIVE_HYBRID = "adaptive_hybrid"

# ============================================================================
# CORE MODELS
# ============================================================================

class NetworkNode(Base):
    """Network node model"""
    __tablename__ = "nodes"
    __table_args__ = {"schema": "network"}
    
    node_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    node_name = Column(String(255), nullable=False, unique=True)
    node_type = Column(String(50), nullable=False, default=NodeType.PEER)
    role = Column(String(50), default=NodeRole.PEER)
    address = Column(String(255), nullable=False)
    port = Column(Integer, nullable=False)
    public_key = Column(Text)
    capabilities = Column(JSONB, default=dict)
    metadata = Column(JSONB, default=dict)
    status = Column(String(50), default=NodeStatus.INACTIVE)
    last_seen = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    local_links = relationship("MeshLink", foreign_keys="MeshLink.local_node_id", back_populates="local_node")
    remote_links = relationship("MeshLink", foreign_keys="MeshLink.remote_node_id", back_populates="remote_node")
    routing_entries = relationship("RoutingEntry", foreign_keys="RoutingEntry.node_id", back_populates="node")
    metrics = relationship("NetworkMetric", back_populates="node")
    events = relationship("NetworkEvent", back_populates="node")

class MeshTopology(Base):
    """Mesh topology configuration"""
    __tablename__ = "mesh_topologies"
    __table_args__ = {"schema": "network"}
    
    topology_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    mesh_name = Column(String(255), nullable=False)
    topology_type = Column(String(50), nullable=False, default=TopologyType.ADAPTIVE_HYBRID)
    configuration = Column(JSONB, nullable=False, default=dict)
    optimization_enabled = Column(Boolean, default=True)
    learning_rate = Column(Numeric(5, 4), default=0.01)
    connectivity_threshold = Column(Numeric(3, 2), default=0.8)
    max_connections_per_node = Column(Integer, default=50)
    status = Column(String(50), default="active")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    links = relationship("MeshLink", back_populates="topology")
    snapshots = relationship("PerformanceSnapshot", back_populates="topology")
    optimizations = relationship("TopologyOptimization", back_populates="topology")

class MeshLink(Base):
    """Mesh link between nodes"""
    __tablename__ = "mesh_links"
    __table_args__ = (
        UniqueConstraint('topology_id', 'local_node_id', 'remote_node_id'),
        {"schema": "network"}
    )
    
    link_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    topology_id = Column(UUID(as_uuid=True), ForeignKey("network.mesh_topologies.topology_id", ondelete="CASCADE"))
    local_node_id = Column(UUID(as_uuid=True), ForeignKey("network.nodes.node_id", ondelete="CASCADE"))
    remote_node_id = Column(UUID(as_uuid=True), ForeignKey("network.nodes.node_id", ondelete="CASCADE"))
    link_state = Column(String(50), nullable=False, default=LinkState.ESTABLISHING)
    quality = Column(Numeric(3, 2), default=0.0)
    latency_ms = Column(Numeric(10, 2))
    bandwidth_mbps = Column(Numeric(10, 2))
    packet_loss = Column(Numeric(5, 4), default=0.0)
    jitter_ms = Column(Numeric(10, 2))
    weight = Column(Numeric(10, 4), default=1.0)
    last_probe = Column(DateTime(timezone=True))
    established_at = Column(DateTime(timezone=True))
    terminated_at = Column(DateTime(timezone=True))
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    topology = relationship("MeshTopology", back_populates="links")
    local_node = relationship("NetworkNode", foreign_keys=[local_node_id], back_populates="local_links")
    remote_node = relationship("NetworkNode", foreign_keys=[remote_node_id], back_populates="remote_links")

# ============================================================================
# ROUTING MODELS
# ============================================================================

class RoutingEntry(Base):
    """BATMAN routing entry"""
    __tablename__ = "routing_entries"
    __table_args__ = (
        UniqueConstraint('node_id', 'destination_id', 'next_hop_id'),
        {"schema": "network"}
    )
    
    entry_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    node_id = Column(UUID(as_uuid=True), ForeignKey("network.nodes.node_id", ondelete="CASCADE"))
    destination_id = Column(UUID(as_uuid=True), ForeignKey("network.nodes.node_id", ondelete="CASCADE"))
    next_hop_id = Column(UUID(as_uuid=True), ForeignKey("network.nodes.node_id", ondelete="SET NULL"))
    sequence_number = Column(BigInteger, nullable=False, default=0)
    quality = Column(Numeric(3, 2), default=0.0)
    hop_count = Column(Integer, default=0)
    flags = Column(Integer, default=0)
    last_seen = Column(DateTime(timezone=True), default=datetime.utcnow)
    is_best_route = Column(Boolean, default=False)
    
    # Relationships
    node = relationship("NetworkNode", foreign_keys=[node_id])
    destination = relationship("NetworkNode", foreign_keys=[destination_id])
    next_hop = relationship("NetworkNode", foreign_keys=[next_hop_id])
    metrics = relationship("RoutingMetric", back_populates="entry")

class RoutingMetric(Base):
    """Routing metrics history"""
    __tablename__ = "routing_metrics"
    __table_args__ = {"schema": "network"}
    
    metric_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    entry_id = Column(UUID(as_uuid=True), ForeignKey("network.routing_entries.entry_id", ondelete="CASCADE"))
    metric_type = Column(String(50), nullable=False)
    metric_value = Column(Numeric(10, 4), nullable=False)
    recorded_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    entry = relationship("RoutingEntry", back_populates="metrics")

# ============================================================================
# CONNECTION POOL MODELS
# ============================================================================

class ConnectionPool(Base):
    """Connection pool configuration"""
    __tablename__ = "connection_pools"
    __table_args__ = {"schema": "network"}
    
    pool_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    node_id = Column(UUID(as_uuid=True), ForeignKey("network.nodes.node_id", ondelete="CASCADE"))
    pool_name = Column(String(255), nullable=False)
    min_connections = Column(Integer, default=5)
    max_connections = Column(Integer, default=100)
    current_connections = Column(Integer, default=0)
    in_use_connections = Column(Integer, default=0)
    keepalive_timeout = Column(Integer, default=300)
    enable_http2 = Column(Boolean, default=True)
    status = Column(String(50), default="active")
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    connections = relationship("Connection", back_populates="pool")

class Connection(Base):
    """Individual connection"""
    __tablename__ = "connections"
    __table_args__ = {"schema": "network"}
    
    connection_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    pool_id = Column(UUID(as_uuid=True), ForeignKey("network.connection_pools.pool_id", ondelete="CASCADE"))
    endpoint = Column(String(255), nullable=False)
    protocol = Column(String(50), default="http/1.1")
    state = Column(String(50), default="idle")
    requests_handled = Column(BigInteger, default=0)
    bytes_sent = Column(BigInteger, default=0)
    bytes_received = Column(BigInteger, default=0)
    errors_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    last_used_at = Column(DateTime(timezone=True))
    closed_at = Column(DateTime(timezone=True))
    
    # Relationships
    pool = relationship("ConnectionPool", back_populates="connections")

# ============================================================================
# OPTIMIZATION MODELS
# ============================================================================

class OptimizationParams(Base):
    """Optimization parameters"""
    __tablename__ = "optimization_params"
    __table_args__ = {"schema": "network"}
    
    param_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    node_id = Column(UUID(as_uuid=True), ForeignKey("network.nodes.node_id", ondelete="CASCADE"))
    param_set_name = Column(String(255), nullable=False)
    batch_size = Column(Integer, default=50)
    compression_algorithm = Column(String(50), default="lz4")
    connection_pool_size = Column(Integer, default=20)
    retry_strategy = Column(String(50), default="exponential")
    max_retries = Column(Integer, default=3)
    circuit_breaker_threshold = Column(Numeric(3, 2), default=0.5)
    adaptive_learning_enabled = Column(Boolean, default=True)
    metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    active_since = Column(DateTime(timezone=True))

class CompressionStats(Base):
    """Compression statistics"""
    __tablename__ = "compression_stats"
    __table_args__ = {"schema": "network"}
    
    stat_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    node_id = Column(UUID(as_uuid=True), ForeignKey("network.nodes.node_id", ondelete="CASCADE"))
    algorithm = Column(String(50), nullable=False)
    messages_compressed = Column(BigInteger, default=0)
    original_bytes = Column(BigInteger, default=0)
    compressed_bytes = Column(BigInteger, default=0)
    compression_time_ms = Column(BigInteger, default=0)
    decompression_time_ms = Column(BigInteger, default=0)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)

# ============================================================================
# METRICS AND TELEMETRY
# ============================================================================

class NetworkMetric(Base):
    """Network metrics"""
    __tablename__ = "metrics"
    __table_args__ = (
        Index('idx_metrics_node_time', 'node_id', 'recorded_at'),
        Index('idx_metrics_type_name', 'metric_type', 'metric_name'),
        {"schema": "network"}
    )
    
    metric_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    node_id = Column(UUID(as_uuid=True), ForeignKey("network.nodes.node_id", ondelete="CASCADE"))
    metric_type = Column(String(100), nullable=False)
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Numeric(20, 6), nullable=False)
    unit = Column(String(50))
    tags = Column(JSONB, default=dict)
    recorded_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    node = relationship("NetworkNode", back_populates="metrics")

class NetworkEvent(Base):
    """Network events"""
    __tablename__ = "events"
    __table_args__ = {"schema": "network"}
    
    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    node_id = Column(UUID(as_uuid=True), ForeignKey("network.nodes.node_id", ondelete="CASCADE"))
    event_type = Column(String(100), nullable=False)
    event_name = Column(String(255), nullable=False)
    severity = Column(String(50), default="info")
    description = Column(Text)
    metadata = Column(JSONB, default=dict)
    occurred_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    node = relationship("NetworkNode", back_populates="events")

class PerformanceSnapshot(Base):
    """Performance snapshots"""
    __tablename__ = "performance_snapshots"
    __table_args__ = {"schema": "network"}
    
    snapshot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    topology_id = Column(UUID(as_uuid=True), ForeignKey("network.mesh_topologies.topology_id", ondelete="CASCADE"))
    total_nodes = Column(Integer, nullable=False)
    total_links = Column(Integer, nullable=False)
    average_latency = Column(Numeric(10, 2))
    network_diameter = Column(Integer)
    clustering_coefficient = Column(Numeric(5, 4))
    connectivity_ratio = Column(Numeric(5, 4))
    fault_tolerance_score = Column(Numeric(5, 4))
    load_balance_index = Column(Numeric(5, 4))
    partition_resilience = Column(Numeric(5, 4))
    quantum_coherence = Column(Numeric(5, 4))
    snapshot_time = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    topology = relationship("MeshTopology", back_populates="snapshots")

class TopologyOptimization(Base):
    """Topology optimization history"""
    __tablename__ = "topology_optimizations"
    __table_args__ = {"schema": "network"}
    
    optimization_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    topology_id = Column(UUID(as_uuid=True), ForeignKey("network.mesh_topologies.topology_id", ondelete="CASCADE"))
    optimization_type = Column(String(50), nullable=False)
    algorithm_used = Column(String(100))
    metrics_before = Column(JSONB, nullable=False)
    metrics_after = Column(JSONB, nullable=False)
    improvement_percentage = Column(Numeric(5, 2))
    execution_time_ms = Column(Integer)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    performed_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    topology = relationship("MeshTopology", back_populates="optimizations")

# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

class NetworkDatabase:
    """Database operations for network module"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_node(self, node_data: Dict[str, Any]) -> NetworkNode:
        """Create a new network node"""
        node = NetworkNode(**node_data)
        self.session.add(node)
        await self.session.commit()
        await self.session.refresh(node)
        return node
    
    async def get_node(self, node_id: str) -> Optional[NetworkNode]:
        """Get node by ID"""
        result = await self.session.execute(
            select(NetworkNode).where(NetworkNode.node_id == node_id)
        )
        return result.scalar_one_or_none()
    
    async def get_active_nodes(self) -> List[NetworkNode]:
        """Get all active nodes"""
        result = await self.session.execute(
            select(NetworkNode).where(NetworkNode.status == NodeStatus.ACTIVE)
        )
        return result.scalars().all()
    
    async def update_node_status(self, node_id: str, status: NodeStatus):
        """Update node status"""
        await self.session.execute(
            update(NetworkNode)
            .where(NetworkNode.node_id == node_id)
            .values(status=status, last_seen=datetime.utcnow())
        )
        await self.session.commit()
    
    async def create_link(self, link_data: Dict[str, Any]) -> MeshLink:
        """Create a mesh link"""
        link = MeshLink(**link_data)
        self.session.add(link)
        await self.session.commit()
        await self.session.refresh(link)
        return link
    
    async def update_link_quality(self, link_id: str, quality: float, latency_ms: float):
        """Update link quality metrics"""
        await self.session.execute(
            update(MeshLink)
            .where(MeshLink.link_id == link_id)
            .values(
                quality=quality,
                latency_ms=latency_ms,
                last_probe=datetime.utcnow(),
                link_state=self._calculate_link_state(quality)
            )
        )
        await self.session.commit()
    
    def _calculate_link_state(self, quality: float) -> LinkState:
        """Calculate link state based on quality"""
        if quality >= 0.9:
            return LinkState.ACTIVE
        elif quality >= 0.7:
            return LinkState.DEGRADED
        elif quality >= 0.5:
            return LinkState.CONGESTED
        else:
            return LinkState.FAILING
    
    async def record_metric(
        self,
        node_id: str,
        metric_type: str,
        metric_name: str,
        metric_value: float,
        unit: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> NetworkMetric:
        """Record a network metric"""
        metric = NetworkMetric(
            node_id=node_id,
            metric_type=metric_type,
            metric_name=metric_name,
            metric_value=metric_value,
            unit=unit,
            tags=tags or {}
        )
        self.session.add(metric)
        await self.session.commit()
        return metric
    
    async def get_node_metrics(
        self,
        node_id: str,
        metric_type: Optional[str] = None,
        limit: int = 100
    ) -> List[NetworkMetric]:
        """Get metrics for a node"""
        query = select(NetworkMetric).where(NetworkMetric.node_id == node_id)
        if metric_type:
            query = query.where(NetworkMetric.metric_type == metric_type)
        query = query.order_by(NetworkMetric.recorded_at.desc()).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def create_routing_entry(
        self,
        node_id: str,
        destination_id: str,
        next_hop_id: str,
        quality: float,
        hop_count: int
    ) -> RoutingEntry:
        """Create or update routing entry"""
        # Check if entry exists
        result = await self.session.execute(
            select(RoutingEntry).where(
                (RoutingEntry.node_id == node_id) &
                (RoutingEntry.destination_id == destination_id) &
                (RoutingEntry.next_hop_id == next_hop_id)
            )
        )
        entry = result.scalar_one_or_none()
        
        if entry:
            # Update existing entry
            entry.quality = quality
            entry.hop_count = hop_count
            entry.last_seen = datetime.utcnow()
            entry.sequence_number += 1
        else:
            # Create new entry
            entry = RoutingEntry(
                node_id=node_id,
                destination_id=destination_id,
                next_hop_id=next_hop_id,
                quality=quality,
                hop_count=hop_count
            )
            self.session.add(entry)
        
        await self.session.commit()
        await self.session.refresh(entry)
        return entry
    
    async def get_best_route(self, node_id: str, destination_id: str) -> Optional[RoutingEntry]:
        """Get best route to destination"""
        result = await self.session.execute(
            select(RoutingEntry).where(
                (RoutingEntry.node_id == node_id) &
                (RoutingEntry.destination_id == destination_id) &
                (RoutingEntry.is_best_route == True)
            )
        )
        return result.scalar_one_or_none()
    
    async def record_event(
        self,
        node_id: str,
        event_type: str,
        event_name: str,
        severity: str = "info",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> NetworkEvent:
        """Record a network event"""
        event = NetworkEvent(
            node_id=node_id,
            event_type=event_type,
            event_name=event_name,
            severity=severity,
            description=description,
            metadata=metadata or {}
        )
        self.session.add(event)
        await self.session.commit()
        return event
    
    async def save_performance_snapshot(
        self,
        topology_id: str,
        metrics: Dict[str, Any]
    ) -> PerformanceSnapshot:
        """Save network performance snapshot"""
        snapshot = PerformanceSnapshot(
            topology_id=topology_id,
            total_nodes=metrics.get('total_nodes', 0),
            total_links=metrics.get('total_links', 0),
            average_latency=metrics.get('average_latency'),
            network_diameter=metrics.get('network_diameter'),
            clustering_coefficient=metrics.get('clustering_coefficient'),
            connectivity_ratio=metrics.get('connectivity_ratio'),
            fault_tolerance_score=metrics.get('fault_tolerance_score'),
            load_balance_index=metrics.get('load_balance_index'),
            partition_resilience=metrics.get('partition_resilience'),
            quantum_coherence=metrics.get('quantum_coherence')
        )
        self.session.add(snapshot)
        await self.session.commit()
        return snapshot
    
    async def get_network_health(self) -> Dict[str, Any]:
        """Get overall network health metrics"""
        # Get node counts
        total_nodes = await self.session.execute(
            select(func.count(NetworkNode.node_id))
        )
        active_nodes = await self.session.execute(
            select(func.count(NetworkNode.node_id))
            .where(NetworkNode.status == NodeStatus.ACTIVE)
        )
        
        # Get link statistics
        link_stats = await self.session.execute(
            select(
                func.count(MeshLink.link_id).label('total_links'),
                func.count(MeshLink.link_id).filter(MeshLink.link_state == LinkState.ACTIVE).label('active_links'),
                func.avg(MeshLink.quality).label('avg_quality'),
                func.avg(MeshLink.latency_ms).label('avg_latency'),
                func.avg(MeshLink.packet_loss).label('avg_packet_loss')
            )
        )
        stats = link_stats.one()
        
        return {
            'total_nodes': total_nodes.scalar(),
            'active_nodes': active_nodes.scalar(),
            'total_links': stats.total_links or 0,
            'active_links': stats.active_links or 0,
            'avg_link_quality': float(stats.avg_quality or 0),
            'avg_latency_ms': float(stats.avg_latency or 0),
            'avg_packet_loss': float(stats.avg_packet_loss or 0),
            'health_status': self._calculate_health_status(
                active_nodes.scalar(),
                total_nodes.scalar(),
                stats.avg_quality
            )
        }
    
    def _calculate_health_status(
        self,
        active_nodes: int,
        total_nodes: int,
        avg_quality: Optional[float]
    ) -> str:
        """Calculate overall health status"""
        if total_nodes == 0:
            return "offline"
        
        node_ratio = active_nodes / total_nodes
        quality = avg_quality or 0
        
        if node_ratio >= 0.9 and quality >= 0.8:
            return "healthy"
        elif node_ratio >= 0.7 and quality >= 0.6:
            return "degraded"
        else:
            return "unhealthy"

# ============================================================================
# INTEGRATION WITH NETWORK COMPONENTS
# ============================================================================

class NetworkDatabaseIntegration:
    """Integration layer for network components"""
    
    def __init__(self, db: NetworkDatabase):
        self.db = db
    
    async def sync_topology_manager(self, topology_manager):
        """Sync topology manager with database"""
        # Import here to avoid circular imports
        from .mesh.topology import MeshTopologyManager, NodeRole
        
        # Update node status
        node = await self.db.get_node(topology_manager.node_id.value)
        if node:
            await self.db.update_node_status(
                topology_manager.node_id.value,
                NodeStatus.ACTIVE
            )
        else:
            # Create node if it doesn't exist
            await self.db.create_node({
                'node_name': topology_manager.node_id.value,
                'node_type': NodeType.PEER,
                'role': topology_manager.node_roles.get(
                    topology_manager.node_id,
                    NodeRole.PEER
                ).value,
                'address': 'localhost',  # Should be passed from config
                'port': 8080,  # Should be passed from config
                'status': NodeStatus.ACTIVE
            })
        
        # Sync mesh links
        for (local, remote), link in topology_manager.mesh_links.items():
            if local == topology_manager.node_id:
                # Find or create link in database
                await self.db.create_link({
                    'topology_id': getattr(topology_manager, 'topology_id', None),  # Needs to be added to topology_manager
                    'local_node_id': local.value,
                    'remote_node_id': remote.value,
                    'link_state': link.state.value,
                    'quality': link.quality,
                    'latency_ms': link.latency_ms,
                    'bandwidth_mbps': link.bandwidth_mbps,
                    'packet_loss': link.packet_loss,
                    'jitter_ms': link.jitter_ms
                })
    
    async def sync_routing_table(self, batman_routing):
        """Sync BATMAN routing table with database"""
        # Import here to avoid circular imports
        from .mesh.routing import BatmanRouting
        
        for dest, routes in batman_routing.routing_table.items():
            for route in routes:
                await self.db.create_routing_entry(
                    node_id=batman_routing.node.id.value,
                    destination_id=dest.value,
                    next_hop_id=route.next_hop.value,
                    quality=route.quality,
                    hop_count=route.hop_count
                )
    
    async def record_optimization_metrics(self, optimizer, node_id: str):
        """Record optimization metrics"""
        # Import here to avoid circular imports
        from .adaptive_optimizer import AdaptiveNetworkOptimizer
        
        metrics = optimizer._calculate_metrics()
        
        # Record each metric type
        for metric_name, metric_value in metrics.items():
            if metric_value is not None and metric_value > 0:
                await self.db.record_metric(
                    node_id=node_id,
                    metric_type='optimization',
                    metric_name=metric_name,
                    metric_value=metric_value,
                    unit=self._get_metric_unit(metric_name)
                )
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for metric"""
        units = {
            'avg_latency': 'ms',
            'p99_latency': 'ms',
            'throughput': 'msg/s',
            'connection_error_rate': 'ratio',
            'cpu_usage': 'percent'
        }
        return units.get(metric_name, '')