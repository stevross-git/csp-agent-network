"""
System State Management
======================

Centralized system state with proper lifecycle management and resource tracking.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from weakref import WeakSet
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class ComponentStatus:
    """Status information for a system component."""
    name: str
    is_available: bool
    is_initialized: bool = False
    error_message: Optional[str] = None
    last_update: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)


class ResourceManager:
    """Manages system resources with automatic cleanup."""
    
    def __init__(self):
        self._resources: Dict[str, Any] = {}
        self._cleanup_callbacks: Dict[str, List[callable]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
    
    def register_resource(self, name: str, resource: Any, cleanup_callback: Optional[callable] = None):
        """Register a resource with optional cleanup callback."""
        self._resources[name] = resource
        self._locks[name] = asyncio.Lock()
        
        if cleanup_callback:
            if name not in self._cleanup_callbacks:
                self._cleanup_callbacks[name] = []
            self._cleanup_callbacks[name].append(cleanup_callback)
        
        logger.debug(f"Registered resource: {name}")
    
    def get_resource(self, name: str) -> Optional[Any]:
        """Get a resource by name."""
        return self._resources.get(name)
    
    async def cleanup_resource(self, name: str):
        """Clean up a specific resource."""
        if name not in self._resources:
            return
        
        async with self._locks[name]:
            try:
                # Run cleanup callbacks
                if name in self._cleanup_callbacks:
                    for callback in self._cleanup_callbacks[name]:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(self._resources[name])
                        else:
                            callback(self._resources[name])
                
                # Remove resource
                del self._resources[name]
                del self._locks[name]
                if name in self._cleanup_callbacks:
                    del self._cleanup_callbacks[name]
                
                logger.debug(f"Cleaned up resource: {name}")
                
            except Exception as e:
                logger.error(f"Error cleaning up resource {name}: {e}")
    
    async def cleanup_all(self):
        """Clean up all resources."""
        for name in list(self._resources.keys()):
            await self.cleanup_resource(name)


class SystemState:
    """
    Centralized system state manager with resource tracking and lifecycle management.
    """
    
    def __init__(self):
        # Core timestamps
        self.startup_time = datetime.now()
        self.last_health_check = datetime.now()
        
        # Resource manager
        self.resource_manager = ResourceManager()
        
        # Core components (initialized as None)
        self.csp_engine: Optional[Any] = None
        self.ai_engine: Optional[Any] = None
        self.runtime_orchestrator: Optional[Any] = None
        self.deployment_orchestrator: Optional[Any] = None
        self.dev_tools: Optional[Any] = None
        self.monitor: Optional[Any] = None
        
        # Database and caching
        self.db_engine: Optional[Any] = None
        self.redis_client: Optional[Any] = None
        
        # Active processes and connections
        self.active_processes: Dict[str, Any] = {}
        self.active_websockets: List[Any] = []
        self.active_tasks: Set[asyncio.Task] = set()
        
        # System metrics and status
        self.system_metrics: Dict[str, Any] = {}
        self.component_status: Dict[str, ComponentStatus] = {}
        
        # Configuration cache
        self._config_cache: Dict[str, Any] = {}
        
        # Event system
        self._event_listeners: Dict[str, List[callable]] = {}
        
        # Locks for thread safety
        self._locks = {
            'processes': asyncio.Lock(),
            'websockets': asyncio.Lock(),
            'metrics': asyncio.Lock(),
            'components': asyncio.Lock()
        }
    
    # Component Management
    async def register_component(
        self, 
        name: str, 
        component: Any,
        is_available: bool = True,
        cleanup_callback: Optional[callable] = None
    ):
        """Register a system component."""
        async with self._locks['components']:
            # Set component attribute
            setattr(self, name, component)
            
            # Register with resource manager
            self.resource_manager.register_resource(name, component, cleanup_callback)
            
            # Update component status
            self.component_status[name] = ComponentStatus(
                name=name,
                is_available=is_available,
                is_initialized=component is not None
            )
            
            logger.info(f"✅ Component registered: {name}")
            await self.emit_event('component_registered', {'name': name, 'component': component})
    
    async def unregister_component(self, name: str):
        """Unregister a system component."""
        async with self._locks['components']:
            if hasattr(self, name):
                # Clean up through resource manager
                await self.resource_manager.cleanup_resource(name)
                
                # Remove component attribute
                setattr(self, name, None)
                
                # Update status
                if name in self.component_status:
                    del self.component_status[name]
                
                logger.info(f"❌ Component unregistered: {name}")
                await self.emit_event('component_unregistered', {'name': name})
    
    def get_component_status(self, name: str) -> Optional[ComponentStatus]:
        """Get status of a component."""
        return self.component_status.get(name)
    
    def is_component_available(self, name: str) -> bool:
        """Check if a component is available."""
        status = self.get_component_status(name)
        return status is not None and status.is_available and status.is_initialized
    
    # Process Management
    async def add_process(self, process_id: str, process: Any):
        """Add an active process."""
        async with self._locks['processes']:
            self.active_processes[process_id] = process
            logger.debug(f"Process added: {process_id}")
            await self.emit_event('process_added', {'process_id': process_id, 'process': process})
    
    async def remove_process(self, process_id: str) -> Optional[Any]:
        """Remove an active process."""
        async with self._locks['processes']:
            process = self.active_processes.pop(process_id, None)
            if process:
                logger.debug(f"Process removed: {process_id}")
                await self.emit_event('process_removed', {'process_id': process_id, 'process': process})
            return process
    
    def get_process(self, process_id: str) -> Optional[Any]:
        """Get a process by ID."""
        return self.active_processes.get(process_id)
    
    def get_process_count(self) -> int:
        """Get the number of active processes."""
        return len(self.active_processes)
    
    # WebSocket Connection Management
    async def add_websocket(self, websocket: Any):
        """Add an active WebSocket connection."""
        async with self._locks['websockets']:
            self.active_websockets.append(websocket)
            logger.debug(f"WebSocket added: {id(websocket)}")
            await self.emit_event('websocket_connected', {'websocket': websocket})
    
    async def remove_websocket(self, websocket: Any):
        """Remove a WebSocket connection."""
        async with self._locks['websockets']:
            try:
                self.active_websockets.remove(websocket)
                logger.debug(f"WebSocket removed: {id(websocket)}")
                await self.emit_event('websocket_disconnected', {'websocket': websocket})
            except ValueError:
                pass  # WebSocket already removed
    
    def get_websocket_count(self) -> int:
        """Get the number of active WebSocket connections."""
        return len(self.active_websockets)
    
    # Task Management
    def add_task(self, task: asyncio.Task):
        """Add a background task."""
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)
        logger.debug(f"Task added: {task.get_name()}")
    
    async def cancel_all_tasks(self):
        """Cancel all active background tasks."""
        if not self.active_tasks:
            return
        
        logger.info(f"Cancelling {len(self.active_tasks)} active tasks...")
        
        for task in self.active_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        
        self.active_tasks.clear()
    
    # Metrics Management
    async def update_metric(self, key: str, value: Any):
        """Update a system metric."""
        async with self._locks['metrics']:
            self.system_metrics[key] = {
                'value': value,
                'timestamp': datetime.now(),
                'type': type(value).__name__
            }
            await self.emit_event('metric_updated', {'key': key, 'value': value})
    
    async def update_metrics(self, metrics: Dict[str, Any]):
        """Update multiple metrics at once."""
        async with self._locks['metrics']:
            timestamp = datetime.now()
            for key, value in metrics.items():
                self.system_metrics[key] = {
                    'value': value,
                    'timestamp': timestamp,
                    'type': type(value).__name__
                }
            await self.emit_event('metrics_updated', {'metrics': metrics})
    
    def get_metric(self, key: str) -> Optional[Any]:
        """Get a metric value."""
        metric = self.system_metrics.get(key)
        return metric['value'] if metric else None
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {k: v['value'] for k, v in self.system_metrics.items()}
    
    # Configuration Cache
    def cache_config(self, key: str, value: Any):
        """Cache a configuration value."""
        self._config_cache[key] = value
    
    def get_cached_config(self, key: str, default: Any = None) -> Any:
        """Get a cached configuration value."""
        return self._config_cache.get(key, default)
    
    def clear_config_cache(self):
        """Clear the configuration cache."""
        self._config_cache.clear()
    
    # Event System
    def add_event_listener(self, event_type: str, callback: callable):
        """Add an event listener."""
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []
        self._event_listeners[event_type].append(callback)
        logger.debug(f"Event listener added for: {event_type}")
    
    def remove_event_listener(self, event_type: str, callback: callable):
        """Remove an event listener."""
        if event_type in self._event_listeners:
            try:
                self._event_listeners[event_type].remove(callback)
                logger.debug(f"Event listener removed for: {event_type}")
            except ValueError:
                pass
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to all listeners."""
        if event_type not in self._event_listeners:
            return
        
        for callback in self._event_listeners[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event listener for {event_type}: {e}")
    
    # Health and Status
    async def update_health_check(self):
        """Update the last health check timestamp."""
        self.last_health_check = datetime.now()
        await self.update_metric('last_health_check', self.last_health_check.isoformat())
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return (datetime.now() - self.startup_time).total_seconds()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'startup_time': self.startup_time.isoformat(),
            'uptime_seconds': self.get_uptime(),
            'last_health_check': self.last_health_check.isoformat(),
            'active_processes': self.get_process_count(),
            'websocket_connections': self.get_websocket_count(),
            'active_tasks': len(self.active_tasks),
            'components': {
                name: {
                    'available': status.is_available,
                    'initialized': status.is_initialized,
                    'error': status.error_message,
                    'last_update': status.last_update.isoformat()
                }
                for name, status in self.component_status.items()
            },
            'metrics_count': len(self.system_metrics),
            'config_cache_size': len(self._config_cache)
        }
    
    # Lifecycle Management
    async def initialize(self):
        """Initialize the system state."""
        logger.info("Initializing system state...")
        self.startup_time = datetime.now()
        await self.update_health_check()
        await self.emit_event('system_initialized', {'startup_time': self.startup_time})
        logger.info("✅ System state initialized")
    
    async def shutdown(self):
        """Shutdown the system state and cleanup resources."""
        logger.info("Shutting down system state...")
        
        try:
            # Cancel all tasks
            await self.cancel_all_tasks()
            
            # Close all WebSocket connections
            websocket_tasks = []
            for ws in self.active_websockets.copy():
                try:
                    if hasattr(ws, 'close'):
                        websocket_tasks.append(ws.close())
                except Exception as e:
                    logger.error(f"Error closing WebSocket: {e}")
            
            if websocket_tasks:
                await asyncio.gather(*websocket_tasks, return_exceptions=True)
            
            self.active_websockets.clear()
            
            # Cleanup all resources
            await self.resource_manager.cleanup_all()
            
            # Clear all data
            self.active_processes.clear()
            self.system_metrics.clear()
            self.component_status.clear()
            self._config_cache.clear()
            self._event_listeners.clear()
            
            await self.emit_event('system_shutdown', {'shutdown_time': datetime.now()})
            logger.info("✅ System state shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
            raise
    
    # Context Manager Support
    @asynccontextmanager
    async def managed_component(self, name: str, component: Any, cleanup_callback: Optional[callable] = None):
        """Context manager for temporary components."""
        try:
            await self.register_component(name, component, cleanup_callback=cleanup_callback)
            yield component
        finally:
            await self.unregister_component(name)
    
    # String representation
    def __repr__(self) -> str:
        return (
            f"SystemState("
            f"uptime={self.get_uptime():.1f}s, "
            f"processes={self.get_process_count()}, "
            f"websockets={self.get_websocket_count()}, "
            f"components={len(self.component_status)}"
            f")"
        )


# Global system state instance
_system_state_instance: Optional[SystemState] = None


def get_system_state() -> SystemState:
    """Get the global system state instance."""
    global _system_state_instance
    if _system_state_instance is None:
        _system_state_instance = SystemState()
    return _system_state_instance


def reset_system_state():
    """Reset the global system state (for testing)."""
    global _system_state_instance
    _system_state_instance = None


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_system_state():
        """Test the system state functionality."""
        print("Testing SystemState...")
        
        # Create system state
        state = SystemState()
        await state.initialize()
        
        # Test component registration
        class DummyComponent:
            def __init__(self, name):
                self.name = name
        
        component = DummyComponent("test_component")
        await state.register_component("test_comp", component)
        
        print(f"Component available: {state.is_component_available('test_comp')}")
        
        # Test process management
        await state.add_process("proc1", {"name": "test_process"})
        print(f"Process count: {state.get_process_count()}")
        
        # Test metrics
        await state.update_metric("cpu_usage", 45.6)
        await state.update_metrics({"memory": 78.2, "disk": 23.1})
        print(f"Metrics: {state.get_all_metrics()}")
        
        # Test events
        events_received = []
        def event_handler(event_type, data):
            events_received.append((event_type, data))
        
        state.add_event_listener("test_event", event_handler)
        await state.emit_event("test_event", {"message": "hello"})
        print(f"Events received: {len(events_received)}")
        
        # Test status
        status = state.get_system_status()
        print(f"System status: {status}")
        
        # Cleanup
        await state.shutdown()
        print("✅ SystemState test completed")
    
    asyncio.run(test_system_state())