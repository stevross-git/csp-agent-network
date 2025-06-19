# File: backend/execution/execution_engine.py
"""
Design Execution Engine
======================
Executes visual designs as CSP processes with real-time monitoring
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime, timedelta
import json

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

# Import CSP core and components
from core.advanced_csp_core import AdvancedCSPEngine, Process, Channel, ProcessContext
from backend.components.registry import ComponentRegistry, ComponentBase, get_component_registry
from backend.models.database_models import (
    Design, DesignNode, DesignConnection, ExecutionSession, ComponentMetric
)
from backend.database.connection import get_db_session, database_transaction
from backend.realtime.websocket_manager import broadcast_execution_event, EventType

logger = logging.getLogger(__name__)

# ============================================================================
# EXECUTION MODELS AND STATES
# ============================================================================

class ExecutionStatus(str, Enum):
    """Execution status states"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class NodeStatus(str, Enum):
    """Individual node execution status"""
    IDLE = "idle"
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"

@dataclass
class ExecutionConfig:
    """Configuration for execution session"""
    max_execution_time: int = 3600  # 1 hour default
    enable_monitoring: bool = True
    enable_profiling: bool = False
    parallel_execution: bool = True
    max_parallel_nodes: int = 10
    timeout_per_node: int = 300  # 5 minutes per node
    retry_failed_nodes: bool = True
    max_retries: int = 3
    custom_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NodeExecution:
    """Execution state for a single node"""
    node_id: str
    component_type: str
    component: Optional[ComponentBase] = None
    status: NodeStatus = NodeStatus.IDLE
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionPlan:
    """Execution plan for a design"""
    design_id: UUID
    nodes: Dict[str, NodeExecution]
    connections: List[Dict[str, Any]]
    execution_order: List[List[str]]  # Levels of parallel execution
    dependency_graph: Dict[str, Set[str]]  # node_id -> dependencies

class ExecutionEngine:
    """Main execution engine for visual designs"""
    
    def __init__(self, component_registry: ComponentRegistry):
        self.component_registry = component_registry
        self.csp_engine = AdvancedCSPEngine()
        self.active_executions: Dict[str, 'ExecutionSession'] = {}
        self.execution_metrics: Dict[str, Dict[str, Any]] = {}
        
    async def execute_design(self, design_id: UUID, config: ExecutionConfig,
                           db_session: AsyncSession) -> 'ExecutionSession':
        """Execute a visual design"""
        try:
            # Load design from database
            design_data = await self._load_design(design_id, db_session)
            if not design_data:
                raise ValueError(f"Design {design_id} not found")
            
            # Create execution session
            session = ExecutionSession(
                design_id=design_id,
                config=config,
                engine=self
            )
            
            # Store in database
            db_execution = await self._create_db_execution_session(
                design_id, session.session_id, config, db_session
            )
            session.db_id = db_execution.id
            
            # Register active execution
            self.active_executions[session.session_id] = session
            
            # Start execution asynchronously
            asyncio.create_task(self._execute_session(session, design_data, db_session))
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to start execution for design {design_id}: {e}")
            raise
    
    async def _load_design(self, design_id: UUID, db_session: AsyncSession) -> Optional[Dict[str, Any]]:
        """Load design data from database"""
        try:
            # Load design with nodes and connections
            design_query = select(Design).where(Design.id == design_id)
            design_result = await db_session.execute(design_query)
            design = design_result.scalar_one_or_none()
            
            if not design:
                return None
            
            # Load nodes
            nodes_query = select(DesignNode).where(DesignNode.design_id == design_id)
            nodes_result = await db_session.execute(nodes_query)
            nodes = nodes_result.scalars().all()
            
            # Load connections
            connections_query = select(DesignConnection).where(DesignConnection.design_id == design_id)
            connections_result = await db_session.execute(connections_query)
            connections = connections_result.scalars().all()
            
            return {
                "design": design.to_dict(),
                "nodes": [node.to_dict() for node in nodes],
                "connections": [conn.to_dict() for conn in connections]
            }
            
        except Exception as e:
            logger.error(f"Failed to load design {design_id}: {e}")
            return None
    
    async def _create_db_execution_session(self, design_id: UUID, session_id: str,
                                         config: ExecutionConfig, db_session: AsyncSession):
        """Create execution session in database"""
        from backend.models.database_models import ExecutionSession as DBExecutionSession
        
        db_execution = DBExecutionSession(
            design_id=design_id,
            session_name=f"Execution {session_id[:8]}",
            status=ExecutionStatus.PENDING.value,
            configuration={
                "max_execution_time": config.max_execution_time,
                "enable_monitoring": config.enable_monitoring,
                "enable_profiling": config.enable_profiling,
                "parallel_execution": config.parallel_execution,
                "max_parallel_nodes": config.max_parallel_nodes,
                "session_id": session_id
            }
        )
        
        db_session.add(db_execution)
        await db_session.commit()
        await db_session.refresh(db_execution)
        
        return db_execution
    
    async def _execute_session(self, session: 'ExecutionSession', design_data: Dict[str, Any],
                             db_session: AsyncSession):
        """Execute a session (runs asynchronously)"""
        try:
            # Update status to initializing
            await self._update_execution_status(session, ExecutionStatus.INITIALIZING, db_session)
            
            # Create execution plan
            execution_plan = await self._create_execution_plan(design_data)
            session.execution_plan = execution_plan
            
            # Initialize components
            await self._initialize_components(session, db_session)
            
            # Update status to running
            await self._update_execution_status(session, ExecutionStatus.RUNNING, db_session)
            session.start_time = datetime.now()
            
            # Execute the plan
            await self._execute_plan(session, db_session)
            
            # Update final status
            if session.status == ExecutionStatus.RUNNING:
                await self._update_execution_status(session, ExecutionStatus.COMPLETED, db_session)
            
            session.end_time = datetime.now()
            
        except asyncio.CancelledError:
            await self._update_execution_status(session, ExecutionStatus.CANCELLED, db_session)
        except Exception as e:
            logger.error(f"Execution session {session.session_id} failed: {e}")
            await self._update_execution_status(session, ExecutionStatus.FAILED, db_session)
            session.error_message = str(e)
        finally:
            # Cleanup
            await self._cleanup_session(session, db_session)
            
            # Remove from active executions
            self.active_executions.pop(session.session_id, None)
    
    async def _create_execution_plan(self, design_data: Dict[str, Any]) -> ExecutionPlan:
        """Create execution plan from design data"""
        nodes_data = design_data["nodes"]
        connections_data = design_data["connections"]
        
        # Create node executions
        nodes = {}
        for node_data in nodes_data:
            node_execution = NodeExecution(
                node_id=node_data["node_id"],
                component_type=node_data["component_type"]
            )
            nodes[node_data["node_id"]] = node_execution
        
        # Build dependency graph
        dependency_graph = {node_id: set() for node_id in nodes.keys()}
        
        for conn in connections_data:
            from_node = conn["from_node_id"]
            to_node = conn["to_node_id"]
            
            # to_node depends on from_node
            if to_node in dependency_graph:
                dependency_graph[to_node].add(from_node)
        
        # Determine execution order (topological sort with levels)
        execution_order = self._calculate_execution_order(dependency_graph)
        
        return ExecutionPlan(
            design_id=UUID(design_data["design"]["id"]),
            nodes=nodes,
            connections=connections_data,
            execution_order=execution_order,
            dependency_graph=dependency_graph
        )
    
    def _calculate_execution_order(self, dependency_graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Calculate execution order with parallel levels"""
        # Topological sort with levels for parallel execution
        in_degree = {node: len(deps) for node, deps in dependency_graph.items()}
        execution_order = []
        
        while in_degree:
            # Find nodes with no dependencies (in_degree = 0)
            current_level = [node for node, degree in in_degree.items() if degree == 0]
            
            if not current_level:
                # Circular dependency detected
                remaining_nodes = list(in_degree.keys())
                logger.warning(f"Circular dependency detected in nodes: {remaining_nodes}")
                execution_order.append(remaining_nodes)
                break
            
            execution_order.append(current_level)
            
            # Remove current level nodes and update in_degrees
            for node in current_level:
                del in_degree[node]
                
                # Update dependencies
                for other_node, deps in dependency_graph.items():
                    if node in deps and other_node in in_degree:
                        in_degree[other_node] -= 1
        
        return execution_order
    
    async def _initialize_components(self, session: 'ExecutionSession', db_session: AsyncSession):
        """Initialize all components in the execution plan"""
        for node_id, node_execution in session.execution_plan.nodes.items():
            try:
                # Get component metadata
                metadata = self.component_registry.get_component_metadata(node_execution.component_type)
                if not metadata:
                    raise ValueError(f"Unknown component type: {node_execution.component_type}")
                
                # Create component instance
                component = await self.component_registry.create_component(
                    node_execution.component_type,
                    node_id,
                    {}  # Properties would be loaded from node data
                )
                
                if not component:
                    raise ValueError(f"Failed to create component {node_execution.component_type}")
                
                node_execution.component = component
                node_execution.status = NodeStatus.IDLE
                
                logger.info(f"Initialized component {node_id} ({node_execution.component_type})")
                
            except Exception as e:
                logger.error(f"Failed to initialize component {node_id}: {e}")
                node_execution.status = NodeStatus.ERROR
                node_execution.error_message = str(e)
                raise
    
    async def _execute_plan(self, session: 'ExecutionSession', db_session: AsyncSession):
        """Execute the execution plan"""
        execution_plan = session.execution_plan
        
        # Execute level by level
        for level_index, level_nodes in enumerate(execution_plan.execution_order):
            logger.info(f"Executing level {level_index} with nodes: {level_nodes}")
            
            # Broadcast execution update
            await broadcast_execution_event(
                session.session_id,
                EventType.EXECUTION_UPDATE,
                {
                    "level": level_index,
                    "nodes": level_nodes,
                    "status": "starting"
                }
            )
            
            # Execute nodes in parallel within the level
            if session.config.parallel_execution:
                tasks = []
                for node_id in level_nodes:
                    if node_id in execution_plan.nodes:
                        task = asyncio.create_task(
                            self._execute_node(session, node_id, db_session)
                        )
                        tasks.append(task)
                
                # Wait for all nodes in this level to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Sequential execution
                for node_id in level_nodes:
                    if node_id in execution_plan.nodes:
                        await self._execute_node(session, node_id, db_session)
            
            # Check if execution should continue
            if session.status != ExecutionStatus.RUNNING:
                break
    
    async def _execute_node(self, session: 'ExecutionSession', node_id: str, db_session: AsyncSession):
        """Execute a single node"""
        node_execution = session.execution_plan.nodes[node_id]
        
        try:
            # Update node status
            node_execution.status = NodeStatus.RUNNING
            node_execution.start_time = datetime.now()
            
            # Broadcast node execution start
            await broadcast_execution_event(
                session.session_id,
                EventType.EXECUTION_UPDATE,
                {
                    "node_id": node_id,
                    "status": "running",
                    "start_time": node_execution.start_time.isoformat()
                }
            )
            
            # Collect inputs from connected nodes
            inputs = await self._collect_node_inputs(session, node_id)
            node_execution.inputs = inputs
            
            # Execute component
            if node_execution.component:
                with asyncio.timeout(session.config.timeout_per_node):
                    outputs = await node_execution.component.execute(inputs)
                    node_execution.outputs = outputs
            else:
                raise ValueError("Component not initialized")
            
            # Update status to completed
            node_execution.status = NodeStatus.COMPLETED
            node_execution.end_time = datetime.now()
            
            # Store metrics
            if session.config.enable_monitoring:
                await self._record_node_metrics(session, node_id, db_session)
            
            # Broadcast completion
            await broadcast_execution_event(
                session.session_id,
                EventType.EXECUTION_UPDATE,
                {
                    "node_id": node_id,
                    "status": "completed",
                    "end_time": node_execution.end_time.isoformat(),
                    "outputs": outputs
                }
            )
            
            logger.info(f"Node {node_id} completed successfully")
            
        except asyncio.TimeoutError:
            node_execution.status = NodeStatus.ERROR
            node_execution.error_message = "Execution timeout"
            node_execution.end_time = datetime.now()
            
            logger.error(f"Node {node_id} timed out")
            
        except Exception as e:
            node_execution.status = NodeStatus.ERROR
            node_execution.error_message = str(e)
            node_execution.end_time = datetime.now()
            
            logger.error(f"Node {node_id} execution failed: {e}")
            
            # Retry logic
            if (session.config.retry_failed_nodes and 
                node_execution.retry_count < session.config.max_retries):
                
                node_execution.retry_count += 1
                logger.info(f"Retrying node {node_id} (attempt {node_execution.retry_count})")
                
                # Wait a bit before retry
                await asyncio.sleep(1.0 * node_execution.retry_count)
                
                # Retry execution
                await self._execute_node(session, node_id, db_session)
        
        finally:
            # Broadcast final status
            await broadcast_execution_event(
                session.session_id,
                EventType.EXECUTION_UPDATE,
                {
                    "node_id": node_id,
                    "status": node_execution.status.value,
                    "error": node_execution.error_message,
                    "retry_count": node_execution.retry_count
                }
            )
    
    async def _collect_node_inputs(self, session: 'ExecutionSession', node_id: str) -> Dict[str, Any]:
        """Collect inputs for a node from its dependencies"""
        inputs = {}
        execution_plan = session.execution_plan
        
        # Find incoming connections
        for connection in execution_plan.connections:
            if connection["to_node_id"] == node_id:
                from_node_id = connection["from_node_id"]
                from_port = connection["from_port"]
                to_port = connection["to_port"]
                
                # Get output from source node
                if from_node_id in execution_plan.nodes:
                    from_node = execution_plan.nodes[from_node_id]
                    if from_node.status == NodeStatus.COMPLETED and from_port in from_node.outputs:
                        inputs[to_port] = from_node.outputs[from_port]
        
        return inputs
    
    async def _record_node_metrics(self, session: 'ExecutionSession', node_id: str, 
                                 db_session: AsyncSession):
        """Record performance metrics for a node"""
        node_execution = session.execution_plan.nodes[node_id]
        
        if not node_execution.start_time or not node_execution.end_time:
            return
        
        execution_time = (node_execution.end_time - node_execution.start_time).total_seconds()
        
        # Create metric records
        metrics = [
            ComponentMetric(
                session_id=session.db_id,
                node_id=node_id,
                metric_name="execution_time",
                metric_value=execution_time,
                metric_unit="seconds"
            ),
            ComponentMetric(
                session_id=session.db_id,
                node_id=node_id,
                metric_name="retry_count",
                metric_value=float(node_execution.retry_count),
                metric_unit="count"
            )
        ]
        
        # Add input/output size metrics
        if node_execution.inputs:
            input_size = len(json.dumps(node_execution.inputs, default=str))
            metrics.append(ComponentMetric(
                session_id=session.db_id,
                node_id=node_id,
                metric_name="input_size",
                metric_value=float(input_size),
                metric_unit="bytes"
            ))
        
        if node_execution.outputs:
            output_size = len(json.dumps(node_execution.outputs, default=str))
            metrics.append(ComponentMetric(
                session_id=session.db_id,
                node_id=node_id,
                metric_name="output_size",
                metric_value=float(output_size),
                metric_unit="bytes"
            ))
        
        # Store metrics
        for metric in metrics:
            db_session.add(metric)
        
        try:
            await db_session.commit()
        except Exception as e:
            logger.error(f"Failed to record metrics for node {node_id}: {e}")
            await db_session.rollback()
    
    async def _update_execution_status(self, session: 'ExecutionSession', 
                                     status: ExecutionStatus, db_session: AsyncSession):
        """Update execution status in database and broadcast"""
        session.status = status
        
        # Update database
        try:
            from backend.models.database_models import ExecutionSession as DBExecutionSession
            
            update_query = update(DBExecutionSession).where(
                DBExecutionSession.id == session.db_id
            ).values(
                status=status.value,
                started_at=session.start_time if status == ExecutionStatus.RUNNING else None,
                ended_at=session.end_time if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED] else None
            )
            
            await db_session.execute(update_query)
            await db_session.commit()
            
        except Exception as e:
            logger.error(f"Failed to update execution status: {e}")
        
        # Broadcast status update
        await broadcast_execution_event(
            session.session_id,
            EventType.EXECUTION_UPDATE,
            {
                "status": status.value,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    async def _cleanup_session(self, session: 'ExecutionSession', db_session: AsyncSession):
        """Cleanup execution session resources"""
        try:
            # Cleanup components
            for node_execution in session.execution_plan.nodes.values():
                if node_execution.component:
                    try:
                        await node_execution.component.cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up component {node_execution.node_id}: {e}")
            
            logger.info(f"Cleaned up execution session {session.session_id}")
            
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
    
    async def pause_execution(self, session_id: str) -> bool:
        """Pause an execution session"""
        if session_id not in self.active_executions:
            return False
        
        session = self.active_executions[session_id]
        session.status = ExecutionStatus.PAUSED
        
        # Implementation would pause individual components
        logger.info(f"Paused execution session {session_id}")
        return True
    
    async def resume_execution(self, session_id: str) -> bool:
        """Resume a paused execution session"""
        if session_id not in self.active_executions:
            return False
        
        session = self.active_executions[session_id]
        if session.status == ExecutionStatus.PAUSED:
            session.status = ExecutionStatus.RUNNING
            
            logger.info(f"Resumed execution session {session_id}")
            return True
        
        return False
    
    async def cancel_execution(self, session_id: str) -> bool:
        """Cancel an execution session"""
        if session_id not in self.active_executions:
            return False
        
        session = self.active_executions[session_id]
        session.status = ExecutionStatus.CANCELLED
        
        # Cancel any running tasks
        # Implementation would cancel component executions
        
        logger.info(f"Cancelled execution session {session_id}")
        return True
    
    def get_execution_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status"""
        if session_id not in self.active_executions:
            return None
        
        session = self.active_executions[session_id]
        
        # Calculate progress
        total_nodes = len(session.execution_plan.nodes)
        completed_nodes = sum(1 for node in session.execution_plan.nodes.values() 
                            if node.status == NodeStatus.COMPLETED)
        
        progress = (completed_nodes / total_nodes * 100) if total_nodes > 0 else 0
        
        return {
            "session_id": session_id,
            "status": session.status.value,
            "progress_percentage": round(progress, 2),
            "start_time": session.start_time.isoformat() if session.start_time else None,
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "total_nodes": total_nodes,
            "completed_nodes": completed_nodes,
            "failed_nodes": sum(1 for node in session.execution_plan.nodes.values() 
                              if node.status == NodeStatus.ERROR),
            "current_nodes": [node_id for node_id, node in session.execution_plan.nodes.items() 
                            if node.status == NodeStatus.RUNNING],
            "error_message": session.error_message
        }

class ExecutionSession:
    """Represents an active execution session"""
    
    def __init__(self, design_id: UUID, config: ExecutionConfig, engine: ExecutionEngine):
        self.session_id = str(uuid4())
        self.design_id = design_id
        self.config = config
        self.engine = engine
        self.status = ExecutionStatus.PENDING
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.execution_plan: Optional[ExecutionPlan] = None
        self.db_id: Optional[UUID] = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "design_id": str(self.design_id),
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_message": self.error_message,
            "config": {
                "max_execution_time": self.config.max_execution_time,
                "parallel_execution": self.config.parallel_execution,
                "enable_monitoring": self.config.enable_monitoring
            }
        }

# Global execution engine instance
execution_engine = None

async def get_execution_engine() -> ExecutionEngine:
    """Get the global execution engine instance"""
    global execution_engine
    
    if execution_engine is None:
        component_registry = await get_component_registry()
        execution_engine = ExecutionEngine(component_registry)
        logger.info("✅ Execution engine initialized")
    
    return execution_engine

# Utility functions
async def execute_design_async(design_id: UUID, config_data: Dict[str, Any] = None) -> ExecutionSession:
    """Execute a design asynchronously"""
    engine = await get_execution_engine()
    
    # Create config
    config = ExecutionConfig(
        max_execution_time=config_data.get("max_execution_time", 3600),
        enable_monitoring=config_data.get("enable_monitoring", True),
        enable_profiling=config_data.get("enable_profiling", False),
        parallel_execution=config_data.get("parallel_execution", True),
        max_parallel_nodes=config_data.get("max_parallel_nodes", 10),
        custom_settings=config_data.get("custom_settings", {})
    )
    
    # Get database session
    async for db_session in get_db_session():
        return await engine.execute_design(design_id, config, db_session)

async def get_execution_metrics(session_id: str) -> Dict[str, Any]:
    """Get detailed execution metrics"""
    engine = await get_execution_engine()
    
    if session_id not in engine.active_executions:
        return {"error": "Execution session not found"}
    
    session = engine.active_executions[session_id]
    
    # Collect detailed metrics
    node_metrics = {}
    for node_id, node_execution in session.execution_plan.nodes.items():
        execution_time = None
        if node_execution.start_time and node_execution.end_time:
            execution_time = (node_execution.end_time - node_execution.start_time).total_seconds()
        
        node_metrics[node_id] = {
            "status": node_execution.status.value,
            "execution_time": execution_time,
            "retry_count": node_execution.retry_count,
            "error_message": node_execution.error_message,
            "input_count": len(node_execution.inputs),
            "output_count": len(node_execution.outputs)
        }
    
    return {
        "session_id": session_id,
        "overall_status": session.status.value,
        "node_metrics": node_metrics,
        "execution_plan": {
            "total_levels": len(session.execution_plan.execution_order),
            "nodes_per_level": [len(level) for level in session.execution_plan.execution_order]
        }
    }
