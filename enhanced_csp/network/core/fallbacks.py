"""
Fallback Implementations
========================

Provides fallback classes when core CSP components are not available.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MockEnum(Enum):
    """Mock enum for fallback implementations."""
    UNKNOWN = "unknown"


# Mock Types and Enums
CompositionOperator = MockEnum
ChannelType = MockEnum
MessageType = MockEnum


class ProcessSignature:
    """Fallback process signature."""
    
    def __init__(self, inputs: List[str] = None, outputs: List[str] = None):
        self.inputs = inputs or []
        self.outputs = outputs or []
        logger.debug("Using fallback ProcessSignature")
    
    def __repr__(self):
        return f"ProcessSignature(inputs={self.inputs}, outputs={self.outputs})"


class ProcessContext:
    """Fallback process context."""
    
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.created_at = datetime.now()
        logger.debug("Using fallback ProcessContext")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get context variable."""
        return self.variables.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set context variable."""
        self.variables[key] = value
    
    def __repr__(self):
        return f"ProcessContext(variables={len(self.variables)})"


class Channel:
    """Fallback channel implementation."""
    
    def __init__(self, name: str = None, capacity: int = 0):
        self.name = name or f"channel_{id(self)}"
        self.capacity = capacity
        self.queue = asyncio.Queue(maxsize=capacity if capacity > 0 else 0)
        self.closed = False
        logger.debug(f"Using fallback Channel: {self.name}")
    
    async def send(self, message: Any):
        """Send message through channel."""
        if self.closed:
            raise RuntimeError("Channel is closed")
        await self.queue.put(message)
    
    async def receive(self) -> Any:
        """Receive message from channel."""
        if self.closed and self.queue.empty():
            raise RuntimeError("Channel is closed and empty")
        return await self.queue.get()
    
    def close(self):
        """Close the channel."""
        self.closed = True
    
    def __repr__(self):
        return f"Channel(name={self.name}, capacity={self.capacity})"


class Event:
    """Fallback event implementation."""
    
    def __init__(self, name: str, data: Any = None):
        self.name = name
        self.data = data
        self.timestamp = datetime.now()
        logger.debug(f"Using fallback Event: {name}")
    
    def __repr__(self):
        return f"Event(name={self.name}, timestamp={self.timestamp})"


class Process:
    """Fallback base process implementation."""
    
    def __init__(self, name: str = None):
        self.name = name or f"process_{id(self)}"
        self.state = "initialized"
        self.context = ProcessContext()
        self.channels: Dict[str, Channel] = {}
        self.created_at = datetime.now()
        logger.debug(f"Using fallback Process: {self.name}")
    
    async def start(self):
        """Start the process."""
        self.state = "running"
        logger.info(f"Started fallback process: {self.name}")
    
    async def stop(self):
        """Stop the process."""
        self.state = "stopped"
        # Close all channels
        for channel in self.channels.values():
            channel.close()
        logger.info(f"Stopped fallback process: {self.name}")
    
    async def pause(self):
        """Pause the process."""
        self.state = "paused"
        logger.info(f"Paused fallback process: {self.name}")
    
    async def resume(self):
        """Resume the process."""
        self.state = "running"
        logger.info(f"Resumed fallback process: {self.name}")
    
    def add_channel(self, name: str, channel: Channel):
        """Add a channel to the process."""
        self.channels[name] = channel
    
    def get_channel(self, name: str) -> Optional[Channel]:
        """Get a channel by name."""
        return self.channels.get(name)
    
    def __repr__(self):
        return f"Process(name={self.name}, state={self.state})"


class AtomicProcess(Process):
    """Fallback atomic process implementation."""
    
    def __init__(self, name: str = None, signature: ProcessSignature = None):
        super().__init__(name)
        self.signature = signature or ProcessSignature()
        self.execution_count = 0
        logger.debug(f"Using fallback AtomicProcess: {self.name}")
    
    async def execute(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the atomic process."""
        if self.state != "running":
            raise RuntimeError(f"Cannot execute process in state: {self.state}")
        
        self.execution_count += 1
        inputs = inputs or {}
        
        # Mock execution
        await asyncio.sleep(0.001)  # Simulate processing
        
        # Return mock outputs
        outputs = {output: f"result_{output}_{self.execution_count}" for output in self.signature.outputs}
        
        logger.debug(f"Executed {self.name}: {inputs} -> {outputs}")
        return outputs
    
    def __repr__(self):
        return f"AtomicProcess(name={self.name}, state={self.state}, executions={self.execution_count})"


class CompositeProcess(Process):
    """Fallback composite process implementation."""
    
    def __init__(self, name: str = None, processes: List[Process] = None):
        super().__init__(name)
        self.processes = processes or []
        self.composition_type = "sequential"  # or "parallel"
        logger.debug(f"Using fallback CompositeProcess: {self.name}")
    
    def add_process(self, process: Process):
        """Add a subprocess."""
        self.processes.append(process)
    
    def remove_process(self, process: Process):
        """Remove a subprocess."""
        if process in self.processes:
            self.processes.remove(process)
    
    async def start(self):
        """Start the composite process and all subprocesses."""
        await super().start()
        
        if self.composition_type == "sequential":
            # Start processes sequentially
            for process in self.processes:
                await process.start()
        else:
            # Start processes in parallel
            await asyncio.gather(*[process.start() for process in self.processes])
    
    async def stop(self):
        """Stop all subprocesses and the composite process."""
        # Stop all subprocesses
        await asyncio.gather(*[process.stop() for process in self.processes], return_exceptions=True)
        await super().stop()
    
    async def execute(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the composite process."""
        if self.state != "running":
            raise RuntimeError(f"Cannot execute process in state: {self.state}")
        
        inputs = inputs or {}
        
        if self.composition_type == "sequential":
            # Execute sequentially, chaining outputs to inputs
            current_inputs = inputs
            for process in self.processes:
                if isinstance(process, AtomicProcess):
                    current_inputs = await process.execute(current_inputs)
            return current_inputs
        else:
            # Execute in parallel
            if all(isinstance(p, AtomicProcess) for p in self.processes):
                results = await asyncio.gather(*[p.execute(inputs) for p in self.processes])
                # Merge all results
                merged_results = {}
                for result in results:
                    merged_results.update(result)
                return merged_results
            else:
                return {}
    
    def __repr__(self):
        return f"CompositeProcess(name={self.name}, state={self.state}, subprocesses={len(self.processes)})"


class ProcessMatcher:
    """Fallback process matcher for pattern matching."""
    
    def __init__(self):
        self.patterns: Dict[str, Any] = {}
        logger.debug("Using fallback ProcessMatcher")
    
    def add_pattern(self, name: str, pattern: Any):
        """Add a matching pattern."""
        self.patterns[name] = pattern
    
    def match(self, process: Process) -> List[str]:
        """Match process against patterns."""
        # Simple name-based matching
        matches = []
        for pattern_name, pattern in self.patterns.items():
            if isinstance(pattern, str) and pattern in process.name:
                matches.append(pattern_name)
        return matches
    
    def __repr__(self):
        return f"ProcessMatcher(patterns={len(self.patterns)})"


class ProtocolEvolution:
    """Fallback protocol evolution for adaptive protocols."""
    
    def __init__(self, initial_protocol: str = "basic"):
        self.current_protocol = initial_protocol
        self.evolution_history: List[str] = [initial_protocol]
        self.adaptation_count = 0
        logger.debug("Using fallback ProtocolEvolution")
    
    async def evolve(self, context: Dict[str, Any] = None) -> str:
        """Evolve the protocol based on context."""
        context = context or {}
        
        # Mock evolution logic
        self.adaptation_count += 1
        
        if self.adaptation_count % 5 == 0:
            new_protocol = f"evolved_v{self.adaptation_count // 5}"
            self.current_protocol = new_protocol
            self.evolution_history.append(new_protocol)
            logger.info(f"Protocol evolved to: {new_protocol}")
        
        return self.current_protocol
    
    def get_history(self) -> List[str]:
        """Get evolution history."""
        return self.evolution_history.copy()
    
    def __repr__(self):
        return f"ProtocolEvolution(current={self.current_protocol}, adaptations={self.adaptation_count})"


class AdvancedCSPEngine:
    """Fallback CSP engine implementation."""
    
    def __init__(self):
        self.processes: Dict[str, Process] = {}
        self.channels: Dict[str, Channel] = {}
        self.running = False
        self.protocol_evolution = ProtocolEvolution()
        self.process_matcher = ProcessMatcher()
        logger.warning("Using fallback AdvancedCSPEngine - limited functionality available")
    
    async def start(self):
        """Start the CSP engine."""
        self.running = True
        logger.info("Started fallback CSP engine")
    
    async def stop(self):
        """Stop the CSP engine."""
        # Stop all processes
        if self.processes:
            await asyncio.gather(*[proc.stop() for proc in self.processes.values()], return_exceptions=True)
        
        # Close all channels
        for channel in self.channels.values():
            channel.close()
        
        self.running = False
        logger.info("Stopped fallback CSP engine")
    
    async def start_process(self, process: Process) -> str:
        """Start a process in the engine."""
        if not self.running:
            await self.start()
        
        process_id = f"proc_{len(self.processes)}_{id(process)}"
        self.processes[process_id] = process
        await process.start()
        
        logger.info(f"Started process {process_id}: {process.name}")
        return process_id
    
    async def stop_process(self, process: Union[str, Process]):
        """Stop a process."""
        if isinstance(process, str):
            process_obj = self.processes.get(process)
            process_id = process
        else:
            process_obj = process
            process_id = None
            for pid, proc in self.processes.items():
                if proc is process:
                    process_id = pid
                    break
        
        if process_obj:
            await process_obj.stop()
            if process_id and process_id in self.processes:
                del self.processes[process_id]
            logger.info(f"Stopped process {process_id}")
        else:
            logger.warning(f"Process not found: {process}")
    
    def create_channel(self, name: str, capacity: int = 0) -> Channel:
        """Create a new channel."""
        channel = Channel(name, capacity)
        self.channels[name] = channel
        logger.debug(f"Created channel: {name}")
        return channel
    
    def get_channel(self, name: str) -> Optional[Channel]:
        """Get a channel by name."""
        return self.channels.get(name)
    
    def get_process(self, process_id: str) -> Optional[Process]:
        """Get a process by ID."""
        return self.processes.get(process_id)
    
    def list_processes(self) -> Dict[str, str]:
        """List all processes with their states."""
        return {pid: proc.state for pid, proc in self.processes.items()}
    
    def list_channels(self) -> List[str]:
        """List all channel names."""
        return list(self.channels.keys())
    
    async def evolve_protocol(self, context: Dict[str, Any] = None) -> str:
        """Evolve the communication protocol."""
        return await self.protocol_evolution.evolve(context)
    
    def match_processes(self, pattern: str) -> List[str]:
        """Match processes by pattern."""
        # Simple name matching
        matches = []
        for process_id, process in self.processes.items():
            if pattern in process.name:
                matches.append(process_id)
        return matches
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "running": self.running,
            "total_processes": len(self.processes),
            "running_processes": sum(1 for p in self.processes.values() if p.state == "running"),
            "total_channels": len(self.channels),
            "protocol_adaptations": self.protocol_evolution.adaptation_count,
            "current_protocol": self.protocol_evolution.current_protocol
        }
    
    def __repr__(self):
        return f"AdvancedCSPEngine(running={self.running}, processes={len(self.processes)}, channels={len(self.channels)})"


# Additional fallback classes for AI components
class AIAgent:
    """Fallback AI agent implementation."""
    
    def __init__(self, name: str, capabilities: List[Any] = None):
        self.name = name
        self.capabilities = capabilities or []
        self.active = False
        logger.debug(f"Using fallback AIAgent: {name}")
    
    async def start(self):
        """Start the AI agent."""
        self.active = True
        logger.info(f"Started fallback AI agent: {self.name}")
    
    async def stop(self):
        """Stop the AI agent."""
        self.active = False
        logger.info(f"Stopped fallback AI agent: {self.name}")
    
    async def process_message(self, message: str) -> str:
        """Process a message (mock response)."""
        if not self.active:
            raise RuntimeError("Agent is not active")
        
        # Mock AI processing
        await asyncio.sleep(0.1)
        return f"Fallback response from {self.name}: {message[:50]}..."
    
    def __repr__(self):
        return f"AIAgent(name={self.name}, capabilities={len(self.capabilities)}, active={self.active})"


class LLMCapability:
    """Fallback LLM capability."""
    
    def __init__(self, model_name: str = "fallback-model", specialized_domain: str = None):
        self.model_name = model_name
        self.specialized_domain = specialized_domain
        logger.debug(f"Using fallback LLMCapability: {model_name}")
    
    async def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text (mock response)."""
        await asyncio.sleep(0.1)
        return f"Fallback generated text for prompt: {prompt[:30]}..."
    
    def __repr__(self):
        return f"LLMCapability(model={self.model_name}, domain={self.specialized_domain})"


class MultiAgentReasoningCoordinator:
    """Fallback multi-agent coordinator."""
    
    def __init__(self, agents: List[AIAgent] = None):
        self.agents = agents or []
        self.active = False
        self.coordination_sessions: Dict[str, Dict] = {}
        logger.debug(f"Using fallback MultiAgentReasoningCoordinator with {len(self.agents)} agents")
    
    async def start(self):
        """Start the coordinator."""
        self.active = True
        for agent in self.agents:
            await agent.start()
        logger.info("Started fallback coordination system")
    
    async def stop(self):
        """Stop the coordinator."""
        for agent in self.agents:
            await agent.stop()
        self.active = False
        logger.info("Stopped fallback coordination system")
    
    async def coordinate(self, task: str, session_id: str = None) -> Dict[str, Any]:
        """Coordinate agents on a task."""
        if not self.active:
            raise RuntimeError("Coordinator is not active")
        
        session_id = session_id or f"session_{len(self.coordination_sessions)}"
        
        # Mock coordination
        results = {}
        for i, agent in enumerate(self.agents):
            response = await agent.process_message(f"Task: {task}")
            results[f"agent_{i}"] = response
        
        self.coordination_sessions[session_id] = {
            "task": task,
            "results": results,
            "timestamp": datetime.now()
        }
        
        return {
            "session_id": session_id,
            "agent_count": len(self.agents),
            "status": "completed",
            "results": results
        }
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get coordination session results."""
        return self.coordination_sessions.get(session_id)
    
    def __repr__(self):
        return f"MultiAgentReasoningCoordinator(agents={len(self.agents)}, active={self.active})"


class AdvancedCSPEngineWithAI(AdvancedCSPEngine):
    """Fallback AI-enhanced CSP engine."""
    
    def __init__(self):
        super().__init__()
        self.ai_coordinator: Optional[MultiAgentReasoningCoordinator] = None
        self.ai_enabled = False
        logger.warning("Using fallback AdvancedCSPEngineWithAI - AI features limited")
    
    def set_ai_coordinator(self, coordinator: MultiAgentReasoningCoordinator):
        """Set the AI coordinator."""
        self.ai_coordinator = coordinator
        self.ai_enabled = True
        logger.info("AI coordinator attached to fallback engine")
    
    async def start(self):
        """Start the engine with AI features."""
        await super().start()
        if self.ai_coordinator:
            await self.ai_coordinator.start()
        logger.info("Started fallback AI-enhanced CSP engine")
    
    async def stop(self):
        """Stop the engine and AI features."""
        if self.ai_coordinator:
            await self.ai_coordinator.stop()
        await super().stop()
        logger.info("Stopped fallback AI-enhanced CSP engine")
    
    async def ai_optimize_process(self, process_id: str) -> Dict[str, Any]:
        """AI-based process optimization (mock)."""
        if not self.ai_enabled or not self.ai_coordinator:
            return {"status": "ai_not_available", "optimization": "none"}
        
        process = self.get_process(process_id)
        if not process:
            return {"status": "process_not_found"}
        
        # Mock AI optimization
        optimization_task = f"Optimize process {process.name} with state {process.state}"
        result = await self.ai_coordinator.coordinate(optimization_task)
        
        return {
            "status": "optimized",
            "process_id": process_id,
            "ai_session": result.get("session_id"),
            "recommendations": ["Mock optimization suggestion 1", "Mock optimization suggestion 2"]
        }
    
    def get_ai_stats(self) -> Dict[str, Any]:
        """Get AI-related statistics."""
        base_stats = self.get_stats()
        ai_stats = {
            "ai_enabled": self.ai_enabled,
            "ai_coordinator_active": self.ai_coordinator.active if self.ai_coordinator else False,
            "ai_agents": len(self.ai_coordinator.agents) if self.ai_coordinator else 0,
            "coordination_sessions": len(self.ai_coordinator.coordination_sessions) if self.ai_coordinator else 0
        }
        return {**base_stats, **ai_stats}


# Runtime and deployment fallbacks
class CSPRuntimeOrchestrator:
    """Fallback runtime orchestrator."""
    
    def __init__(self):
        self.running = False
        self.managed_processes: Dict[str, Process] = {}
        self.resource_usage: Dict[str, float] = {}
        logger.debug("Using fallback CSPRuntimeOrchestrator")
    
    async def start(self):
        """Start the orchestrator."""
        self.running = True
        logger.info("Started fallback runtime orchestrator")
    
    async def stop(self):
        """Stop the orchestrator."""
        # Stop all managed processes
        if self.managed_processes:
            await asyncio.gather(*[proc.stop() for proc in self.managed_processes.values()], return_exceptions=True)
        self.running = False
        logger.info("Stopped fallback runtime orchestrator")
    
    async def deploy_process(self, process: Process, config: Dict[str, Any] = None) -> str:
        """Deploy a process."""
        if not self.running:
            await self.start()
        
        deployment_id = f"deploy_{len(self.managed_processes)}_{id(process)}"
        self.managed_processes[deployment_id] = process
        await process.start()
        
        # Mock resource allocation
        self.resource_usage[deployment_id] = {
            "cpu": 0.1,
            "memory": 0.05,
            "network": 0.02
        }
        
        logger.info(f"Deployed process {deployment_id}: {process.name}")
        return deployment_id
    
    async def undeploy_process(self, deployment_id: str):
        """Undeploy a process."""
        if deployment_id in self.managed_processes:
            process = self.managed_processes[deployment_id]
            await process.stop()
            del self.managed_processes[deployment_id]
            if deployment_id in self.resource_usage:
                del self.resource_usage[deployment_id]
            logger.info(f"Undeployed process {deployment_id}")
        else:
            logger.warning(f"Deployment not found: {deployment_id}")
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status."""
        if deployment_id not in self.managed_processes:
            return None
        
        process = self.managed_processes[deployment_id]
        return {
            "deployment_id": deployment_id,
            "process_name": process.name,
            "state": process.state,
            "resource_usage": self.resource_usage.get(deployment_id, {}),
            "uptime": (datetime.now() - process.created_at).total_seconds()
        }
    
    def list_deployments(self) -> List[str]:
        """List all deployments."""
        return list(self.managed_processes.keys())
    
    def __repr__(self):
        return f"CSPRuntimeOrchestrator(running={self.running}, deployments={len(self.managed_processes)})"


class CSPDeploymentOrchestrator:
    """Fallback deployment orchestrator."""
    
    def __init__(self):
        self.active = False
        self.deployments: Dict[str, Dict] = {}
        logger.debug("Using fallback CSPDeploymentOrchestrator")
    
    async def start(self):
        """Start the deployment orchestrator."""
        self.active = True
        logger.info("Started fallback deployment orchestrator")
    
    async def stop(self):
        """Stop the deployment orchestrator."""
        self.active = False
        logger.info("Stopped fallback deployment orchestrator")
    
    async def create_deployment(self, name: str, config: Dict[str, Any]) -> str:
        """Create a new deployment."""
        if not self.active:
            await self.start()
        
        deployment_id = f"deploy_{len(self.deployments)}_{name}"
        self.deployments[deployment_id] = {
            "name": name,
            "config": config,
            "status": "created",
            "created_at": datetime.now(),
            "instances": []
        }
        
        logger.info(f"Created deployment {deployment_id}: {name}")
        return deployment_id
    
    async def scale_deployment(self, deployment_id: str, replicas: int):
        """Scale a deployment."""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        deployment = self.deployments[deployment_id]
        current_instances = len(deployment["instances"])
        
        if replicas > current_instances:
            # Scale up
            for i in range(current_instances, replicas):
                instance_id = f"{deployment_id}_instance_{i}"
                deployment["instances"].append({
                    "id": instance_id,
                    "status": "running",
                    "created_at": datetime.now()
                })
        elif replicas < current_instances:
            # Scale down
            deployment["instances"] = deployment["instances"][:replicas]
        
        deployment["status"] = "scaled"
        logger.info(f"Scaled deployment {deployment_id} to {replicas} replicas")
    
    def get_deployment(self, deployment_id: str) -> Optional[Dict]:
        """Get deployment details."""
        return self.deployments.get(deployment_id)
    
    def list_deployments(self) -> List[str]:
        """List all deployments."""
        return list(self.deployments.keys())
    
    def __repr__(self):
        return f"CSPDeploymentOrchestrator(active={self.active}, deployments={len(self.deployments)})"


class CSPDevelopmentTools:
    """Fallback development tools."""
    
    def __init__(self):
        self.active = False
        self.projects: Dict[str, Dict] = {}
        logger.debug("Using fallback CSPDevelopmentTools")
    
    async def start(self):
        """Start development tools."""
        self.active = True
        logger.info("Started fallback development tools")
    
    async def stop(self):
        """Stop development tools."""
        self.active = False
        logger.info("Stopped fallback development tools")
    
    def create_project(self, name: str, template: str = "basic") -> str:
        """Create a new project."""
        project_id = f"proj_{len(self.projects)}_{name}"
        self.projects[project_id] = {
            "name": name,
            "template": template,
            "created_at": datetime.now(),
            "files": [],
            "processes": []
        }
        
        logger.info(f"Created project {project_id}: {name}")
        return project_id
    
    def generate_code(self, project_id: str, spec: Dict[str, Any]) -> str:
        """Generate code from specification."""
        if project_id not in self.projects:
            raise ValueError(f"Project not found: {project_id}")
        
        # Mock code generation
        code = f"""
# Generated CSP Process
# Project: {self.projects[project_id]['name']}
# Specification: {spec.get('type', 'unknown')}

class Generated{spec.get('name', 'Process')}(AtomicProcess):
    def __init__(self):
        super().__init__("{spec.get('name', 'generated_process')}")
    
    async def execute(self, inputs=None):
        # TODO: Implement process logic
        return {{'result': 'generated output'}}
"""
        
        logger.info(f"Generated code for project {project_id}")
        return code
    
    def get_project(self, project_id: str) -> Optional[Dict]:
        """Get project details."""
        return self.projects.get(project_id)
    
    def list_projects(self) -> List[str]:
        """List all projects."""
        return list(self.projects.keys())
    
    def __repr__(self):
        return f"CSPDevelopmentTools(active={self.active}, projects={len(self.projects)})"


class CSPMonitor:
    """Fallback monitoring system."""
    
    def __init__(self):
        self.running = False
        self.metrics: Dict[str, List] = {}
        self.alerts: List[Dict] = []
        logger.debug("Using fallback CSPMonitor")
    
    async def start(self):
        """Start the monitor."""
        self.running = True
        # Start background monitoring task
        asyncio.create_task(self._monitoring_loop())
        logger.info("Started fallback monitor")
    
    async def stop(self):
        """Stop the monitor."""
        self.running = False
        logger.info("Stopped fallback monitor")
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                # Collect mock metrics
                timestamp = datetime.now()
                self.record_metric("system.cpu", 45.6, timestamp)
                self.record_metric("system.memory", 78.2, timestamp)
                self.record_metric("csp.processes", 5, timestamp)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                if self.running:
                    await asyncio.sleep(5)
    
    def record_metric(self, name: str, value: float, timestamp: datetime = None):
        """Record a metric value."""
        timestamp = timestamp or datetime.now()
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            "value": value,
            "timestamp": timestamp
        })
        
        # Keep only last 1000 values
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
    
    def get_metric(self, name: str, limit: int = 100) -> List[Dict]:
        """Get metric history."""
        if name not in self.metrics:
            return []
        return self.metrics[name][-limit:]
    
    def create_alert(self, message: str, severity: str = "info"):
        """Create an alert."""
        alert = {
            "message": message,
            "severity": severity,
            "timestamp": datetime.now(),
            "id": len(self.alerts)
        }
        self.alerts.append(alert)
        logger.info(f"Alert created: {message}")
    
    def get_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts."""
        return self.alerts[-limit:]
    
    def __repr__(self):
        return f"CSPMonitor(running={self.running}, metrics={len(self.metrics)}, alerts={len(self.alerts)})"


# Export all fallback classes
__all__ = [
    # Core types
    'CompositionOperator',
    'ChannelType',
    'MessageType',
    'ProcessSignature',
    'ProcessContext',
    'Channel',
    'Event',
    
    # Process classes
    'Process',
    'AtomicProcess',
    'CompositeProcess',
    'ProcessMatcher',
    'ProtocolEvolution',
    
    # Engine classes
    'AdvancedCSPEngine',
    'AdvancedCSPEngineWithAI',
    
    # AI classes
    'AIAgent',
    'LLMCapability',
    'MultiAgentReasoningCoordinator',
    
    # Runtime classes
    'CSPRuntimeOrchestrator',
    'CSPDeploymentOrchestrator',
    'CSPDevelopmentTools',
    'CSPMonitor',
]


# Example usage and testing
if __name__ == "__main__":
    async def test_fallbacks():
        """Test fallback implementations."""
        print("Testing fallback implementations...")
        
        # Test CSP Engine
        engine = AdvancedCSPEngine()
        await engine.start()
        
        # Test processes
        atomic_proc = AtomicProcess("test_atomic", ProcessSignature(["input1"], ["output1"]))
        composite_proc = CompositeProcess("test_composite", [atomic_proc])
        
        proc_id1 = await engine.start_process(atomic_proc)
        proc_id2 = await engine.start_process(composite_proc)
        
        # Test execution
        result = await atomic_proc.execute({"input1": "test_data"})
        print(f"Atomic process result: {result}")
        
        # Test channels
        channel = engine.create_channel("test_channel", 10)
        await channel.send("test_message")
        message = await channel.receive()
        print(f"Channel message: {message}")
        
        # Test AI components
        ai_agent = AIAgent("test_agent", [LLMCapability("fallback-llm")])
        coordinator = MultiAgentReasoningCoordinator([ai_agent])
        
        ai_engine = AdvancedCSPEngineWithAI()
        ai_engine.set_ai_coordinator(coordinator)
        await ai_engine.start()
        
        # Test AI optimization
        optimization = await ai_engine.ai_optimize_process(proc_id1)
        print(f"AI optimization: {optimization}")
        
        # Test runtime orchestrator
        runtime = CSPRuntimeOrchestrator()
        deploy_id = await runtime.deploy_process(AtomicProcess("deployed_proc"))
        status = runtime.get_deployment_status(deploy_id)
        print(f"Deployment status: {status}")
        
        # Test development tools
        dev_tools = CSPDevelopmentTools()
        await dev_tools.start()
        project_id = dev_tools.create_project("test_project")
        code = dev_tools.generate_code(project_id, {"name": "TestProcess", "type": "atomic"})
        print(f"Generated code preview: {code[:100]}...")
        
        # Test monitor
        monitor = CSPMonitor()
        await monitor.start()
        monitor.record_metric("test.metric", 42.0)
        monitor.create_alert("Test alert", "warning")
        await asyncio.sleep(1)  # Let monitor collect some data
        
        # Cleanup
        await ai_engine.stop()
        await engine.stop()
        await runtime.stop()
        await dev_tools.stop()
        await monitor.stop()
        
        print("✅ Fallback implementations test completed")
    
    asyncio.run(test_fallbacks())