# network/db_integration_example.py
"""
Example: Integrating Network Database with Existing Components
==============================================================
Shows how to add database persistence to network modules
"""

import asyncio
import logging
import time
from typing import Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

# Import existing network components (relative imports since we're in the network module)
from .mesh.topology import MeshTopologyManager, TopologyType, NodeRole, MeshLink
from .mesh.routing import BatmanRouting
from .adaptive_optimizer import AdaptiveNetworkOptimizer
from .optimized_channel import OptimizedNetworkChannel
from .core.config import MeshConfig
from .core.types import NodeID

# Import database components
from .database_models import (
    NetworkDatabase, NetworkNode, NodeStatus, LinkState,
    NetworkDatabaseIntegration
)
from .network_db_config import get_network_db
from .monitoring_integration import create_network_monitoring

logger = logging.getLogger(__name__)


class DatabaseEnabledTopologyManager(MeshTopologyManager):
    """Extended topology manager with database persistence"""
    
    def __init__(self, node_id, config, send_message_fn, db_session: Optional[AsyncSession] = None):
        super().__init__(node_id, config, send_message_fn)
        self.db_session = db_session
        self.db: Optional[NetworkDatabase] = None
        self.db_integration: Optional[NetworkDatabaseIntegration] = None
        
        if db_session:
            self.db = NetworkDatabase(db_session)
            self.db_integration = NetworkDatabaseIntegration(self.db)
    
    async def start(self):
        """Start topology manager with database sync"""
        await super().start()
        
        if self.db_integration:
            # Register node in database
            await self._register_node()
            
            # Start periodic sync
            self._db_sync_task = asyncio.create_task(self._db_sync_loop())
    
    async def stop(self):
        """Stop topology manager and cleanup database"""
        if hasattr(self, '_db_sync_task'):
            self._db_sync_task.cancel()
            try:
                await self._db_sync_task
            except asyncio.CancelledError:
                pass
        
        if self.db:
            # Update node status to inactive
            await self.db.update_node_status(
                self.node_id.value,
                NodeStatus.INACTIVE
            )
        
        await super().stop()
    
    async def _register_node(self):
        """Register this node in the database"""
        try:
            node = await self.db.get_node(self.node_id.value)
            if not node:
                # Create new node
                await self.db.create_node({
                    'node_name': self.node_id.value,
                    'node_type': 'peer',
                    'role': self.node_roles.get(self.node_id, NodeRole.PEER).value,
                    'address': self.config.listen_address,
                    'port': self.config.listen_port,
                    'capabilities': {
                        'topology_types': [t.value for t in TopologyType],
                        'max_connections': self.config.max_connections
                    },
                    'status': NodeStatus.ACTIVE
                })
                logger.info(f"Registered node {self.node_id.value} in database")
            else:
                # Update existing node
                await self.db.update_node_status(
                    self.node_id.value,
                    NodeStatus.ACTIVE
                )
        except Exception as e:
            logger.error(f"Failed to register node: {e}")
    
    async def _db_sync_loop(self):
        """Periodically sync topology with database"""
        while True:
            try:
                await self._sync_to_database()
                await asyncio.sleep(30)  # Sync every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Database sync error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _sync_to_database(self):
        """Sync current topology state to database"""
        if not self.db_integration:
            return
        
        try:
            # Sync topology manager state
            await self.db_integration.sync_topology_manager(self)
            
            # Record network metrics
            metrics = self.get_network_stats()  # Using existing method
            
            # Get or create a topology_id
            if not hasattr(self, 'topology_id'):
                # Create a default topology in DB if needed
                from .database_models import MeshTopology
                result = await self.db.session.execute(
                    select(MeshTopology).where(MeshTopology.mesh_name == "default")
                )
                topology = result.scalar_one_or_none()
                if not topology:
                    topology = MeshTopology(
                        mesh_name="default",
                        topology_type=self.topology_type.value
                    )
                    self.db.session.add(topology)
                    await self.db.session.commit()
                    await self.db.session.refresh(topology)
                self.topology_id = str(topology.topology_id)
            
            await self.db.save_performance_snapshot(
                self.topology_id,
                {
                    'total_nodes': len(self.peers),
                    'total_links': len(self.mesh_links),
                    'average_latency': metrics.get('metrics', {}).get('average_latency', 0),
                    'network_diameter': metrics.get('metrics', {}).get('network_diameter', 0),
                    'clustering_coefficient': metrics.get('metrics', {}).get('clustering_coefficient', 0),
                    'connectivity_ratio': metrics.get('metrics', {}).get('connectivity_ratio', 0),
                    'fault_tolerance_score': metrics.get('metrics', {}).get('fault_tolerance_score', 0),
                    'load_balance_index': metrics.get('metrics', {}).get('load_balance_index', 0),
                    'partition_resilience': metrics.get('metrics', {}).get('partition_resilience', 0),
                    'quantum_coherence': metrics.get('metrics', {}).get('quantum_coherence', 0)
                }
            )
            
            logger.debug("Topology synced to database")
        except Exception as e:
            logger.error(f"Failed to sync topology: {e}")
    
    async def add_mesh_link(self, remote_node_id, link: MeshLink):
        """Add mesh link and persist to database"""
        await super().add_mesh_link(remote_node_id, link)
        
        if self.db:
            try:
                await self.db.create_link({
                    'topology_id': self.topology_id,
                    'local_node_id': self.node_id.value,
                    'remote_node_id': remote_node_id.value,
                    'link_state': link.state.value,
                    'quality': link.quality,
                    'latency_ms': link.latency_ms,
                    'bandwidth_mbps': link.bandwidth_mbps,
                    'packet_loss': link.packet_loss,
                    'jitter_ms': link.jitter_ms,
                    'weight': link.weight
                })
            except Exception as e:
                logger.error(f"Failed to persist link: {e}")
    
    async def update_link_quality(self, local_node_id, remote_node_id, quality, latency):
        """Update link quality and persist to database"""
        await super().update_link_quality(local_node_id, remote_node_id, quality, latency)
        
        if self.db:
            try:
                # Find link ID (you might want to cache this)
                link_key = (local_node_id, remote_node_id)
                if link_key in self.mesh_links:
                    link = self.mesh_links[link_key]
                    # Assuming you add link_id to MeshLink
                    if hasattr(link, 'link_id'):
                        await self.db.update_link_quality(
                            link.link_id,
                            quality,
                            latency
                        )
            except Exception as e:
                logger.error(f"Failed to update link quality in DB: {e}")


class DatabaseEnabledOptimizer(AdaptiveNetworkOptimizer):
    """Extended optimizer with metric persistence"""
    
    def __init__(self, node_id: str, db: Optional[NetworkDatabase] = None, **kwargs):
        super().__init__(**kwargs)
        self.node_id = node_id
        self.db = db
        self._last_metric_save = 0
    
    async def _save_metrics_to_db(self):
        """Save optimization metrics to database"""
        if not self.db or time.time() - self._last_metric_save < 60:
            return
        
        try:
            metrics = self._calculate_metrics()
            
            # Save each metric
            for metric_name, value in metrics.items():
                if value is not None:
                    await self.db.record_metric(
                        node_id=self.node_id,
                        metric_type='optimization',
                        metric_name=metric_name,
                        metric_value=value,
                        unit=self._get_unit(metric_name)
                    )
            
            self._last_metric_save = time.time()
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def _get_unit(self, metric_name: str) -> str:
        units = {
            'avg_latency': 'ms',
            'p99_latency': 'ms', 
            'throughput': 'msg/s',
            'connection_error_rate': 'ratio',
            'cpu_usage': 'percent'
        }
        return units.get(metric_name, '')
    
    async def _optimize_parameters(self):
        """Optimize parameters and save to database"""
        await super()._optimize_parameters()
        await self._save_metrics_to_db()


# Example usage when running from within network module
async def main():
    """Example of using database-enabled network components"""
    
    # Get database session  
    async for db_session in get_network_db():
        # Create database interface
        db = NetworkDatabase(db_session)
        
        # Create monitoring context
        async with create_network_monitoring(
            db_session,
            node_name="example-node",
            prometheus_pushgateway="http://localhost:9091"
        ) as monitoring:
            
            # Create database-enabled topology manager
            topology_manager = DatabaseEnabledTopologyManager(
                node_id=NodeID("node-001"),
                config=MeshConfig(),
                send_message_fn=lambda msg: None,
                db_session=db_session
            )
            
            # Start topology manager
            await topology_manager.start()
            
            # Create database-enabled optimizer
            optimizer = DatabaseEnabledOptimizer(
                node_id="node-001",
                db=db
            )
            
            # Run for a while
            await asyncio.sleep(300)
            
            # Get network health
            health = await db.get_network_health()
            print(f"Network health: {health}")
            
            # Stop topology manager
            await topology_manager.stop()


if __name__ == "__main__":
    asyncio.run(main())