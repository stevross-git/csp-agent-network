# network/monitoring_integration.py
"""
Network Monitoring Integration
==============================
Integrates network metrics with Prometheus, Loki, and the database
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway
from sqlalchemy.ext.asyncio import AsyncSession

from .database_models import (
    NetworkDatabase, NetworkNode, MeshLink, NetworkMetric,
    NodeStatus, LinkState
)

logger = logging.getLogger(__name__)

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# Create a separate registry for network metrics
NETWORK_REGISTRY = CollectorRegistry()

# Node metrics
node_status_gauge = Gauge(
    'csp_network_node_status',
    'Network node status (1=active, 0=inactive)',
    ['node_name', 'node_type', 'role'],
    registry=NETWORK_REGISTRY
)

node_connections_gauge = Gauge(
    'csp_network_node_connections',
    'Number of active connections per node',
    ['node_name'],
    registry=NETWORK_REGISTRY
)

# Link metrics
link_quality_gauge = Gauge(
    'csp_network_link_quality',
    'Link quality score (0-1)',
    ['local_node', 'remote_node'],
    registry=NETWORK_REGISTRY
)

link_latency_histogram = Histogram(
    'csp_network_link_latency_ms',
    'Link latency in milliseconds',
    ['local_node', 'remote_node'],
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000),
    registry=NETWORK_REGISTRY
)

link_bandwidth_gauge = Gauge(
    'csp_network_link_bandwidth_mbps',
    'Link bandwidth in Mbps',
    ['local_node', 'remote_node'],
    registry=NETWORK_REGISTRY
)

link_packet_loss_gauge = Gauge(
    'csp_network_link_packet_loss',
    'Link packet loss ratio (0-1)',
    ['local_node', 'remote_node'],
    registry=NETWORK_REGISTRY
)

# Routing metrics
routing_table_size_gauge = Gauge(
    'csp_network_routing_table_size',
    'Number of routes in routing table',
    ['node_name'],
    registry=NETWORK_REGISTRY
)

best_route_quality_gauge = Gauge(
    'csp_network_best_route_quality',
    'Quality of best route to destination',
    ['source_node', 'destination_node'],
    registry=NETWORK_REGISTRY
)

# Connection pool metrics
connection_pool_size_gauge = Gauge(
    'csp_network_connection_pool_size',
    'Current connection pool size',
    ['node_name', 'pool_name'],
    registry=NETWORK_REGISTRY
)

connection_pool_utilization_gauge = Gauge(
    'csp_network_connection_pool_utilization',
    'Connection pool utilization percentage',
    ['node_name', 'pool_name'],
    registry=NETWORK_REGISTRY
)

# Compression metrics
compression_ratio_histogram = Histogram(
    'csp_network_compression_ratio',
    'Message compression ratio',
    ['node_name', 'algorithm'],
    buckets=(1, 2, 3, 4, 5, 10),
    registry=NETWORK_REGISTRY
)

compression_time_histogram = Histogram(
    'csp_network_compression_time_ms',
    'Time spent on compression in milliseconds',
    ['node_name', 'algorithm'],
    buckets=(0.1, 0.5, 1, 5, 10, 50, 100),
    registry=NETWORK_REGISTRY
)

# Batch metrics
batch_size_histogram = Histogram(
    'csp_network_batch_size',
    'Message batch sizes',
    ['node_name'],
    buckets=(1, 5, 10, 25, 50, 100, 250),
    registry=NETWORK_REGISTRY
)

batch_queue_length_gauge = Gauge(
    'csp_network_batch_queue_length',
    'Current batch queue length',
    ['node_name'],
    registry=NETWORK_REGISTRY
)

# Network-wide metrics
network_diameter_gauge = Gauge(
    'csp_network_diameter',
    'Network diameter (max shortest path)',
    registry=NETWORK_REGISTRY
)

network_clustering_coefficient_gauge = Gauge(
    'csp_network_clustering_coefficient',
    'Network clustering coefficient',
    registry=NETWORK_REGISTRY
)

network_partition_resilience_gauge = Gauge(
    'csp_network_partition_resilience',
    'Network partition resilience score',
    registry=NETWORK_REGISTRY
)

# ============================================================================
# MONITORING SERVICE
# ============================================================================

class NetworkMonitoringService:
    """Service for collecting and exporting network metrics"""
    
    def __init__(
        self,
        db: NetworkDatabase,
        prometheus_pushgateway: Optional[str] = None,
        export_interval: int = 15,
        snapshot_interval: int = 300
    ):
        self.db = db
        self.prometheus_pushgateway = prometheus_pushgateway
        self.export_interval = export_interval
        self.snapshot_interval = snapshot_interval
        
        self._running = False
        self._export_task: Optional[asyncio.Task] = None
        self._snapshot_task: Optional[asyncio.Task] = None
        
        # Cache for topology IDs
        self._topology_cache: Dict[str, str] = {}
    
    async def start(self):
        """Start monitoring service"""
        self._running = True
        
        # Start export task
        self._export_task = asyncio.create_task(self._export_metrics_loop())
        
        # Start snapshot task
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())
        
        logger.info("Network monitoring service started")
    
    async def stop(self):
        """Stop monitoring service"""
        self._running = False
        
        # Cancel tasks
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
        
        if self._snapshot_task:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Network monitoring service stopped")
    
    async def _export_metrics_loop(self):
        """Main loop for exporting metrics"""
        while self._running:
            try:
                await self._collect_and_export_metrics()
                
                # Push to Prometheus if configured
                if self.prometheus_pushgateway:
                    push_to_gateway(
                        self.prometheus_pushgateway,
                        job='csp_network',
                        registry=NETWORK_REGISTRY
                    )
                
            except Exception as e:
                logger.error(f"Error exporting metrics: {e}")
            
            await asyncio.sleep(self.export_interval)
    
    async def _snapshot_loop(self):
        """Loop for creating performance snapshots"""
        while self._running:
            try:
                await self._create_performance_snapshot()
            except Exception as e:
                logger.error(f"Error creating snapshot: {e}")
            
            await asyncio.sleep(self.snapshot_interval)
    
    async def _collect_and_export_metrics(self):
        """Collect metrics from database and export to Prometheus"""
        # Get all active nodes
        nodes = await self.db.get_active_nodes()
        
        for node in nodes:
            # Update node status
            node_status_gauge.labels(
                node_name=node.node_name,
                node_type=node.node_type,
                role=node.role
            ).set(1 if node.status == NodeStatus.ACTIVE else 0)
            
            # Get node connectivity
            connectivity = await self._get_node_connectivity(node.node_id)
            node_connections_gauge.labels(
                node_name=node.node_name
            ).set(connectivity['active_links'])
            
            # Get routing table size
            routing_size = await self._get_routing_table_size(node.node_id)
            routing_table_size_gauge.labels(
                node_name=node.node_name
            ).set(routing_size)
            
            # Get recent metrics
            recent_metrics = await self.db.get_node_metrics(
                node.node_id,
                limit=100
            )
            
            # Process metrics by type
            await self._process_node_metrics(node, recent_metrics)
        
        # Export link metrics
        await self._export_link_metrics()
        
        # Export network-wide metrics
        await self._export_network_metrics()
    
    async def _get_node_connectivity(self, node_id: str) -> Dict[str, int]:
        """Get node connectivity statistics"""
        from sqlalchemy import func, or_
        
        result = await self.db.session.execute(
            select(
                func.count(MeshLink.link_id).label('total_links'),
                func.count(MeshLink.link_id).filter(
                    MeshLink.link_state == LinkState.ACTIVE
                ).label('active_links')
            ).where(
                or_(
                    MeshLink.local_node_id == node_id,
                    MeshLink.remote_node_id == node_id
                )
            )
        )
        stats = result.one()
        
        return {
            'total_links': stats.total_links or 0,
            'active_links': stats.active_links or 0
        }
    
    async def _get_routing_table_size(self, node_id: str) -> int:
        """Get routing table size for node"""
        from sqlalchemy import func
        from .database_models import RoutingEntry
        
        result = await self.db.session.execute(
            select(func.count(RoutingEntry.entry_id))
            .where(RoutingEntry.node_id == node_id)
        )
        return result.scalar() or 0
    
    async def _process_node_metrics(self, node: NetworkNode, metrics: List[NetworkMetric]):
        """Process and export node-specific metrics"""
        # Group metrics by type
        metrics_by_type: Dict[str, List[NetworkMetric]] = {}
        for metric in metrics:
            if metric.metric_type not in metrics_by_type:
                metrics_by_type[metric.metric_type] = []
            metrics_by_type[metric.metric_type].append(metric)
        
        # Process optimization metrics
        if 'optimization' in metrics_by_type:
            for metric in metrics_by_type['optimization']:
                if metric.metric_name == 'cpu_usage':
                    # CPU usage is already handled by node exporter
                    pass
                elif metric.metric_name == 'avg_latency':
                    # This will be handled in link metrics
                    pass
        
        # Process compression metrics
        if 'compression' in metrics_by_type:
            compression_by_algo: Dict[str, List[float]] = {}
            
            for metric in metrics_by_type['compression']:
                algo = metric.tags.get('algorithm', 'unknown')
                
                if metric.metric_name == 'compression_ratio':
                    compression_ratio_histogram.labels(
                        node_name=node.node_name,
                        algorithm=algo
                    ).observe(metric.metric_value)
                
                elif metric.metric_name == 'compression_time_ms':
                    compression_time_histogram.labels(
                        node_name=node.node_name,
                        algorithm=algo
                    ).observe(metric.metric_value)
        
        # Process batch metrics
        if 'batching' in metrics_by_type:
            for metric in metrics_by_type['batching']:
                if metric.metric_name == 'batch_size':
                    batch_size_histogram.labels(
                        node_name=node.node_name
                    ).observe(metric.metric_value)
                
                elif metric.metric_name == 'queue_length':
                    batch_queue_length_gauge.labels(
                        node_name=node.node_name
                    ).set(metric.metric_value)
    
    async def _export_link_metrics(self):
        """Export mesh link metrics"""
        # Get all active links with proper joins
        from sqlalchemy.orm import aliased
        
        LocalNode = aliased(NetworkNode)
        RemoteNode = aliased(NetworkNode)
        
        result = await self.db.session.execute(
            select(MeshLink, LocalNode, RemoteNode)
            .join(LocalNode, MeshLink.local_node_id == LocalNode.node_id)
            .join(RemoteNode, MeshLink.remote_node_id == RemoteNode.node_id)
            .where(MeshLink.link_state == LinkState.ACTIVE)
        )
        
        for link, local_node, remote_node in result:
            # Link quality
            link_quality_gauge.labels(
                local_node=local_node.node_name,
                remote_node=remote_node.node_name
            ).set(float(link.quality))
            
            # Link latency
            if link.latency_ms:
                link_latency_histogram.labels(
                    local_node=local_node.node_name,
                    remote_node=remote_node.node_name
                ).observe(float(link.latency_ms))
            
            # Link bandwidth
            if link.bandwidth_mbps:
                link_bandwidth_gauge.labels(
                    local_node=local_node.node_name,
                    remote_node=remote_node.node_name
                ).set(float(link.bandwidth_mbps))
            
            # Packet loss
            link_packet_loss_gauge.labels(
                local_node=local_node.node_name,
                remote_node=remote_node.node_name
            ).set(float(link.packet_loss))
    
    async def _export_network_metrics(self):
        """Export network-wide metrics"""
        # Get latest performance snapshot
        from .database_models import PerformanceSnapshot
        
        result = await self.db.session.execute(
            select(PerformanceSnapshot)
            .order_by(PerformanceSnapshot.snapshot_time.desc())
            .limit(1)
        )
        snapshot = result.scalar_one_or_none()
        
        if snapshot:
            if snapshot.network_diameter:
                network_diameter_gauge.set(snapshot.network_diameter)
            
            if snapshot.clustering_coefficient:
                network_clustering_coefficient_gauge.set(
                    float(snapshot.clustering_coefficient)
                )
            
            if snapshot.partition_resilience:
                network_partition_resilience_gauge.set(
                    float(snapshot.partition_resilience)
                )
    
    async def _create_performance_snapshot(self):
        """Create and save performance snapshot"""
        # Get network statistics
        health = await self.db.get_network_health()
        
        # Calculate additional metrics
        metrics = {
            'total_nodes': health['total_nodes'],
            'total_links': health['total_links'],
            'average_latency': health['avg_latency_ms'],
            'connectivity_ratio': (
                health['active_links'] / health['total_links']
                if health['total_links'] > 0 else 0
            ),
            'fault_tolerance_score': self._calculate_fault_tolerance(health),
            'load_balance_index': await self._calculate_load_balance()
        }
        
        # Get or create topology ID
        topology_id = await self._get_default_topology_id()
        
        if topology_id:
            await self.db.save_performance_snapshot(topology_id, metrics)
    
    def _calculate_fault_tolerance(self, health: Dict[str, Any]) -> float:
        """Calculate fault tolerance score"""
        # Simple calculation based on redundancy
        if health['total_nodes'] < 2:
            return 0.0
        
        avg_connections = (health['total_links'] * 2) / health['total_nodes']
        
        # Score based on average connectivity
        if avg_connections >= 4:
            return 0.9
        elif avg_connections >= 3:
            return 0.7
        elif avg_connections >= 2:
            return 0.5
        else:
            return 0.3
    
    async def _calculate_load_balance(self) -> float:
        """Calculate load balance index"""
        # This would need actual load metrics
        # For now, return a placeholder
        return 0.8
    
    async def _get_default_topology_id(self) -> Optional[str]:
        """Get default topology ID"""
        if 'default' in self._topology_cache:
            return self._topology_cache['default']
        
        from .database_models import MeshTopology
        
        result = await self.db.session.execute(
            select(MeshTopology.topology_id)
            .where(MeshTopology.status == 'active')
            .limit(1)
        )
        topology_id = result.scalar_one_or_none()
        
        if topology_id:
            self._topology_cache['default'] = str(topology_id)
        
        return topology_id

# ============================================================================
# LOG INTEGRATION
# ============================================================================

class NetworkLogHandler(logging.Handler):
    """Custom log handler that sends network logs to Loki via Promtail"""
    
    def __init__(self, node_name: str):
        super().__init__()
        self.node_name = node_name
        
        # Configure formatter for structured logging
        self.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
    
    def emit(self, record):
        """Emit log record"""
        try:
            # Add extra fields for Loki labels
            record.node_name = self.node_name
            record.component = 'network'
            record.module = record.name.split('.')[-1]
            
            # Format the message
            msg = self.format(record)
            
            # Write to file that Promtail is monitoring
            log_file = f"/var/log/csp/network/{self.node_name}.log"
            
            with open(log_file, 'a') as f:
                # Write as JSON for better parsing
                import json
                log_entry = {
                    'timestamp': record.created,
                    'level': record.levelname,
                    'node': self.node_name,
                    'module': record.module,
                    'message': record.getMessage(),
                    'extra': {
                        'filename': record.filename,
                        'lineno': record.lineno,
                        'funcName': record.funcName
                    }
                }
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception:
            self.handleError(record)

# ============================================================================
# INTEGRATION HELPER
# ============================================================================

@asynccontextmanager
async def create_network_monitoring(
    db_session: AsyncSession,
    node_name: str,
    prometheus_pushgateway: Optional[str] = None
):
    """Create and manage network monitoring"""
    # Create database interface
    db = NetworkDatabase(db_session)
    
    # Create monitoring service
    monitoring = NetworkMonitoringService(
        db=db,
        prometheus_pushgateway=prometheus_pushgateway
    )
    
    # Setup logging
    log_handler = NetworkLogHandler(node_name)
    network_logger = logging.getLogger('enhanced_csp.network')
    network_logger.addHandler(log_handler)
    
    # Start monitoring
    await monitoring.start()
    
    try:
        yield monitoring
    finally:
        # Stop monitoring
        await monitoring.stop()
        
        # Remove log handler
        network_logger.removeHandler(log_handler)