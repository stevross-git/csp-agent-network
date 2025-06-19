"""
CSP Runtime Environment & Orchestrator
======================================

A complete runtime environment for executing, monitoring, and managing
advanced CSP networks with real-time orchestration, performance optimization,
and production-ready deployment capabilities.

Features:
- High-performance CSP runtime with async execution
- Real-time process orchestration and load balancing
- Advanced scheduling algorithms for CSP processes
- Performance monitoring and automatic optimization
- Fault tolerance and recovery mechanisms
- Distributed deployment across multiple nodes
- Real-time debugging and introspection
- Hot-swapping of processes and protocols
"""

import asyncio
try:
    import uvloop  # High-performance event loop
except Exception:  # pragma: no cover - optional dependency
    uvloop = None
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import time
try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None
import logging
if uvloop is None:
    logging.warning("uvloop not available; falling back to asyncio loop")
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import json
import pickle
import zlib
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod
try:
    import aiohttp
except Exception:  # pragma: no cover - optional dependency
    aiohttp = None
try:
    import websockets
except Exception:  # pragma: no cover - optional dependency
    websockets = None
import signal
import sys
import os
from contextlib import asynccontextmanager
import weakref
import gc

# Import our CSP components
from core.advanced_csp_core import (
    AdvancedCSPEngine, Process, ProcessContext, Channel, Event, ProcessState
)
from ai_extensions.csp_ai_extensions import AdvancedCSPEngineWithAI
from ai_integration.csp_ai_integration import AIAgent, CollaborativeAIProcess

# ============================================================================
# RUNTIME EXECUTION MODELS
# ============================================================================

class ExecutionModel(Enum):
    """Different execution models for CSP processes"""
    SINGLE_THREADED = auto()    # Single thread, cooperative
    MULTI_THREADED = auto()     # Multiple threads, shared memory
    MULTI_PROCESS = auto()      # Multiple processes, message passing
    DISTRIBUTED = auto()        # Distributed across nodes
    HYBRID = auto()            # Combination of above

class SchedulingPolicy(Enum):
    """Scheduling policies for process execution"""
    ROUND_ROBIN = auto()
    PRIORITY_BASED = auto()
    LOAD_BALANCED = auto()
    DEADLINE_AWARE = auto()
    ADAPTIVE = auto()

@dataclass
class RuntimeConfig:
    """Configuration for CSP runtime"""
    execution_model: ExecutionModel = ExecutionModel.MULTI_THREADED
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.ADAPTIVE
    max_workers: int = mp.cpu_count()
    memory_limit_gb: float = 8.0
    network_buffer_size: int = 65536
    enable_monitoring: bool = True
    enable_optimization: bool = True
    checkpoint_interval: float = 30.0  # seconds
    gc_interval: float = 10.0  # seconds
    debug_mode: bool = False

# ============================================================================
# HIGH-PERFORMANCE RUNTIME EXECUTOR
# ============================================================================

class CSPRuntimeExecutor:
    """High-performance runtime executor for CSP processes"""
    
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.running_processes = {}
        self.process_pool = None
        self.thread_pool = None
        self.scheduler = ProcessScheduler(config.scheduling_policy)
        self.performance_monitor = PerformanceMonitor()
        self.resource_manager = ResourceManager(config.memory_limit_gb)
        
        # Runtime state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.process_queue = asyncio.Queue()
        self.execution_stats = defaultdict(int)
        
        # Setup event loop optimization
        if config.execution_model in [ExecutionModel.MULTI_THREADED, ExecutionModel.HYBRID]:
            try:
                uvloop.install()  # Use high-performance event loop
            except ImportError:
                logging.warning("uvloop not available, using default event loop")
    
    async def start(self):
        """Start the CSP runtime executor"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Initialize execution pools
        if self.config.execution_model in [ExecutionModel.MULTI_THREADED, ExecutionModel.HYBRID]:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        if self.config.execution_model in [ExecutionModel.MULTI_PROCESS, ExecutionModel.HYBRID]:
            self.process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers // 2)
        
        # Start background tasks
        asyncio.create_task(self._execution_loop())
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._optimization_loop())
        asyncio.create_task(self._garbage_collection_loop())
        
        logging.info(f"CSP Runtime started with {self.config.execution_model.name} execution model")
    
    async def stop(self):
        """Stop the CSP runtime executor"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Cancel running processes
        for process_id, task in self.running_processes.items():
            if not task.done():
                task.cancel()
        
        # Wait for graceful shutdown
        if self.running_processes:
            await asyncio.gather(*self.running_processes.values(), return_exceptions=True)
        
        # Shutdown executors
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        logging.info("CSP Runtime stopped")
    
    async def execute_process(self, process: Process, context: ProcessContext, 
                            priority: int = 5) -> str:
        """Execute a CSP process with specified priority"""
        
        execution_request = {
            'process': process,
            'context': context,
            'priority': priority,
            'submitted_at': time.time(),
            'execution_id': f"exec_{int(time.time()*1000)}_{process.process_id}"
        }
        
        await self.process_queue.put(execution_request)
        return execution_request['execution_id']
    
    async def _execution_loop(self):
        """Main execution loop"""
        while self.is_running:
            try:
                # Get next process to execute
                try:
                    request = await asyncio.wait_for(
                        self.process_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Schedule process execution
                execution_task = await self.scheduler.schedule_process(
                    request, self._execute_process_task
                )
                
                if execution_task:
                    self.running_processes[request['execution_id']] = execution_task
                    self.execution_stats['processes_started'] += 1
                
            except Exception as e:
                logging.error(f"Execution loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_process_task(self, request: Dict[str, Any]) -> Any:
        """Execute individual process task"""
        process = request['process']
        context = request['context']
        execution_id = request['execution_id']
        
        start_time = time.time()
        
        try:
            # Record execution start
            self.performance_monitor.record_execution_start(execution_id, process.process_id)
            
            # Choose execution method based on configuration
            if self.config.execution_model == ExecutionModel.SINGLE_THREADED:
                result = await process.run(context)
            
            elif self.config.execution_model == ExecutionModel.MULTI_THREADED:
                # Run in thread pool for CPU-bound work
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool, 
                    self._sync_process_wrapper, 
                    process, context
                )
            
            elif self.config.execution_model == ExecutionModel.MULTI_PROCESS:
                # Run in process pool for heavy computation
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.process_pool,
                    self._process_pool_wrapper,
                    pickle.dumps((process, context))
                )
                result = pickle.loads(result)
            
            else:
                # Adaptive execution - choose best method
                result = await self._adaptive_execution(process, context)
            
            # Record successful execution
            execution_time = time.time() - start_time
            self.performance_monitor.record_execution_complete(
                execution_id, execution_time, True
            )
            
            self.execution_stats['processes_completed'] += 1
            return result
            
        except Exception as e:
            # Record failed execution
            execution_time = time.time() - start_time
            self.performance_monitor.record_execution_complete(
                execution_id, execution_time, False, str(e)
            )
            
            self.execution_stats['processes_failed'] += 1
            logging.error(f"Process execution failed: {e}")
            raise
        
        finally:
            # Cleanup
            if execution_id in self.running_processes:
                del self.running_processes[execution_id]
    
    def _sync_process_wrapper(self, process: Process, context: ProcessContext) -> Any:
        """Wrapper for synchronous process execution"""
        # Create new event loop for thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(process.run(context))
        finally:
            loop.close()
    
    @staticmethod
    def _process_pool_wrapper(serialized_data: bytes) -> bytes:
        """Wrapper for process pool execution"""
        process, context = pickle.loads(serialized_data)
        
        # Create new event loop for process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(process.run(context))
            return pickle.dumps(result)
        finally:
            loop.close()
    
    async def _adaptive_execution(self, process: Process, context: ProcessContext) -> Any:
        """Adaptive execution choosing best method based on process characteristics"""
        
        # Analyze process characteristics
        signature = process.get_signature()
        cpu_requirement = signature.resource_requirements.get('cpu', 0.1)
        
        if cpu_requirement > 0.8:
            # CPU-intensive - use process pool
            return await self._execute_in_process_pool(process, context)
        elif cpu_requirement > 0.3:
            # Moderate CPU - use thread pool
            return await self._execute_in_thread_pool(process, context)
        else:
            # IO-bound - use async execution
            return await process.run(context)
    
    async def _execute_in_thread_pool(self, process: Process, context: ProcessContext) -> Any:
        """Execute process in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_pool,
            self._sync_process_wrapper,
            process, context
        )
    
    async def _execute_in_process_pool(self, process: Process, context: ProcessContext) -> Any:
        """Execute process in process pool"""
        loop = asyncio.get_event_loop()
        serialized = pickle.dumps((process, context))
        result_bytes = await loop.run_in_executor(
            self.process_pool,
            self._process_pool_wrapper,
            serialized
        )
        return pickle.loads(result_bytes)
    
    async def _monitoring_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                if self.config.enable_monitoring:
                    await self.performance_monitor.collect_metrics()
                await asyncio.sleep(5.0)  # Monitor every 5 seconds
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10.0)
    
    async def _optimization_loop(self):
        """Performance optimization loop"""
        while self.is_running:
            try:
                if self.config.enable_optimization:
                    await self._optimize_runtime_performance()
                await asyncio.sleep(30.0)  # Optimize every 30 seconds
            except Exception as e:
                logging.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60.0)
    
    async def _garbage_collection_loop(self):
        """Garbage collection loop"""
        while self.is_running:
            try:
                # Force garbage collection
                collected = gc.collect()
                if collected > 0:
                    logging.debug(f"Garbage collected {collected} objects")
                
                await asyncio.sleep(self.config.gc_interval)
            except Exception as e:
                logging.error(f"GC loop error: {e}")
                await asyncio.sleep(30.0)
    
    async def _optimize_runtime_performance(self):
        """Optimize runtime performance based on metrics"""
        metrics = self.performance_monitor.get_current_metrics()
        
        # Adjust worker pool sizes based on load
        if metrics.get('cpu_usage', 0) > 0.8 and self.thread_pool:
            # Increase thread pool if high CPU usage
            current_workers = self.thread_pool._max_workers
            if current_workers < self.config.max_workers * 2:
                logging.info("Increasing thread pool size due to high CPU usage")
                # Would need to recreate pool with more workers
        
        # Adjust scheduling policy based on performance
        avg_wait_time = metrics.get('avg_wait_time', 0)
        if avg_wait_time > 1.0:  # High wait time
            self.scheduler.adapt_policy(SchedulingPolicy.PRIORITY_BASED)

# ============================================================================
# ADVANCED PROCESS SCHEDULER
# ============================================================================

class ProcessScheduler:
    """Advanced scheduler for CSP processes"""
    
    def __init__(self, policy: SchedulingPolicy):
        self.policy = policy
        self.ready_queue = []
        self.priority_queues = defaultdict(deque)
        self.load_balancer = LoadBalancer()
        self.deadline_tracker = DeadlineTracker()
        self.adaptation_history = []
    
    async def schedule_process(self, request: Dict[str, Any], 
                             executor_func: Callable) -> asyncio.Task:
        """Schedule process for execution"""
        
        if self.policy == SchedulingPolicy.ROUND_ROBIN:
            return await self._round_robin_schedule(request, executor_func)
        elif self.policy == SchedulingPolicy.PRIORITY_BASED:
            return await self._priority_schedule(request, executor_func)
        elif self.policy == SchedulingPolicy.LOAD_BALANCED:
            return await self._load_balanced_schedule(request, executor_func)
        elif self.policy == SchedulingPolicy.DEADLINE_AWARE:
            return await self._deadline_aware_schedule(request, executor_func)
        elif self.policy == SchedulingPolicy.ADAPTIVE:
            return await self._adaptive_schedule(request, executor_func)
        else:
            # Default to immediate execution
            return asyncio.create_task(executor_func(request))
    
    async def _round_robin_schedule(self, request: Dict, executor_func: Callable) -> asyncio.Task:
        """Round-robin scheduling"""
        # Simple immediate execution for demo
        return asyncio.create_task(executor_func(request))
    
    async def _priority_schedule(self, request: Dict, executor_func: Callable) -> asyncio.Task:
        """Priority-based scheduling"""
        priority = request.get('priority', 5)
        
        # Higher priority (lower number) executes first
        if priority <= 2:  # High priority
            return asyncio.create_task(executor_func(request))
        else:
            # Queue for later execution
            self.priority_queues[priority].append((request, executor_func))
            return await self._execute_when_resources_available(priority)
    
    async def _load_balanced_schedule(self, request: Dict, executor_func: Callable) -> asyncio.Task:
        """Load-balanced scheduling"""
        optimal_worker = self.load_balancer.select_optimal_worker()
        
        # For now, just execute immediately
        # In real implementation, would route to specific worker
        return asyncio.create_task(executor_func(request))
    
    async def _deadline_aware_schedule(self, request: Dict, executor_func: Callable) -> asyncio.Task:
        """Deadline-aware scheduling"""
        deadline = request.get('deadline', time.time() + 60.0)
        self.deadline_tracker.add_task(request['execution_id'], deadline)
        
        # Prioritize based on deadline urgency
        urgency = deadline - time.time()
        if urgency < 5.0:  # Very urgent
            return asyncio.create_task(executor_func(request))
        else:
            # Schedule based on deadline
            return await self._schedule_by_deadline(request, executor_func, deadline)
    
    async def _adaptive_schedule(self, request: Dict, executor_func: Callable) -> asyncio.Task:
        """Adaptive scheduling based on system state"""
        
        # Analyze current system state
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # Choose scheduling strategy based on system state
        if cpu_usage > 80:
            # High CPU - use priority scheduling
            return await self._priority_schedule(request, executor_func)
        elif memory_usage > 80:
            # High memory - use load balancing
            return await self._load_balanced_schedule(request, executor_func)
        else:
            # Normal state - use round robin
            return await self._round_robin_schedule(request, executor_func)
    
    async def _execute_when_resources_available(self, priority: int) -> asyncio.Task:
        """Execute when system resources become available"""
        # Wait for resources and execute highest priority task
        while self.priority_queues[priority]:
            request, executor_func = self.priority_queues[priority].popleft()
            return asyncio.create_task(executor_func(request))
        
        # Return dummy task if queue is empty
        return asyncio.create_task(asyncio.sleep(0))
    
    async def _schedule_by_deadline(self, request: Dict, executor_func: Callable, 
                                  deadline: float) -> asyncio.Task:
        """Schedule task based on deadline"""
        # Calculate delay to optimize deadline adherence
        current_time = time.time()
        estimated_duration = request.get('estimated_duration', 1.0)
        
        # Execute immediately if close to deadline
        if deadline - current_time < estimated_duration * 2:
            return asyncio.create_task(executor_func(request))
        
        # Schedule for optimal execution time
        delay = max(0, deadline - current_time - estimated_duration)
        await asyncio.sleep(min(delay, 10.0))  # Max 10 second delay
        return asyncio.create_task(executor_func(request))
    
    def adapt_policy(self, new_policy: SchedulingPolicy):
        """Adapt scheduling policy based on performance"""
        old_policy = self.policy
        self.policy = new_policy
        
        self.adaptation_history.append({
            'timestamp': time.time(),
            'old_policy': old_policy.name,
            'new_policy': new_policy.name,
            'reason': 'performance_optimization'
        })
        
        logging.info(f"Scheduling policy adapted: {old_policy.name} -> {new_policy.name}")

class LoadBalancer:
    """Load balancer for distributing processes across workers"""
    
    def __init__(self):
        self.worker_loads = defaultdict(float)
        self.worker_capabilities = {}
    
    def select_optimal_worker(self) -> str:
        """Select optimal worker for task execution"""
        if not self.worker_loads:
            return "default_worker"
        
        # Select worker with lowest load
        min_load_worker = min(self.worker_loads.items(), key=lambda x: x[1])
        return min_load_worker[0]
    
    def update_worker_load(self, worker_id: str, load: float):
        """Update worker load information"""
        self.worker_loads[worker_id] = load

class DeadlineTracker:
    """Track and manage process deadlines"""
    
    def __init__(self):
        self.task_deadlines = {}
        self.missed_deadlines = []
    
    def add_task(self, task_id: str, deadline: float):
        """Add task with deadline"""
        self.task_deadlines[task_id] = deadline
    
    def check_deadlines(self) -> List[str]:
        """Check for missed deadlines"""
        current_time = time.time()
        missed = []
        
        for task_id, deadline in list(self.task_deadlines.items()):
            if current_time > deadline:
                missed.append(task_id)
                self.missed_deadlines.append({
                    'task_id': task_id,
                    'deadline': deadline,
                    'missed_at': current_time
                })
                del self.task_deadlines[task_id]
        
        return missed

# ============================================================================
# PERFORMANCE MONITORING SYSTEM
# ============================================================================

class PerformanceMonitor:
    """Comprehensive performance monitoring for CSP runtime"""
    
    def __init__(self):
        self.execution_metrics = {}
        self.system_metrics = defaultdict(list)
        self.alert_thresholds = {
            'cpu_usage': 90.0,
            'memory_usage': 85.0,
            'avg_response_time': 5.0,
            'error_rate': 0.1
        }
        self.alerts_sent = set()
    
    def record_execution_start(self, execution_id: str, process_id: str):
        """Record process execution start"""
        self.execution_metrics[execution_id] = {
            'process_id': process_id,
            'start_time': time.time(),
            'end_time': None,
            'success': None,
            'error': None
        }
    
    def record_execution_complete(self, execution_id: str, duration: float, 
                                success: bool, error: str = None):
        """Record process execution completion"""
        if execution_id in self.execution_metrics:
            self.execution_metrics[execution_id].update({
                'end_time': time.time(),
                'duration': duration,
                'success': success,
                'error': error
            })
    
    async def collect_metrics(self):
        """Collect system performance metrics"""
        current_time = time.time()
        
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_snapshot = {
            'timestamp': current_time,
            'cpu_usage': cpu_usage,
            'memory_usage': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_usage': disk.percent,
            'active_processes': len([m for m in self.execution_metrics.values() 
                                   if m['end_time'] is None])
        }
        
        self.system_metrics['snapshots'].append(system_snapshot)
        
        # Keep only recent metrics (last hour)
        cutoff_time = current_time - 3600
        self.system_metrics['snapshots'] = [
            s for s in self.system_metrics['snapshots'] 
            if s['timestamp'] > cutoff_time
        ]
        
        # Check for alerts
        await self._check_alerts(system_snapshot)
        
        # Calculate derived metrics
        await self._calculate_derived_metrics()
    
    async def _check_alerts(self, snapshot: Dict[str, Any]):
        """Check for performance alerts"""
        alerts = []
        
        for metric, threshold in self.alert_thresholds.items():
            if metric in snapshot and snapshot[metric] > threshold:
                alert_key = f"{metric}_{int(time.time()//300)}"  # 5-minute windows
                if alert_key not in self.alerts_sent:
                    alerts.append({
                        'metric': metric,
                        'value': snapshot[metric],
                        'threshold': threshold,
                        'timestamp': snapshot['timestamp']
                    })
                    self.alerts_sent.add(alert_key)
        
        if alerts:
            await self._send_alerts(alerts)
    
    async def _send_alerts(self, alerts: List[Dict[str, Any]]):
        """Send performance alerts"""
        for alert in alerts:
            logging.warning(f"Performance Alert: {alert['metric']} = {alert['value']:.2f} "
                          f"(threshold: {alert['threshold']})")
    
    async def _calculate_derived_metrics(self):
        """Calculate derived performance metrics"""
        recent_executions = [
            m for m in self.execution_metrics.values()
            if m['end_time'] and m['end_time'] > time.time() - 300  # Last 5 minutes
        ]
        
        if recent_executions:
            # Calculate success rate
            successful = len([e for e in recent_executions if e['success']])
            success_rate = successful / len(recent_executions)
            
            # Calculate average response time
            avg_response_time = np.mean([e['duration'] for e in recent_executions])
            
            # Update system metrics
            current_time = time.time()
            self.system_metrics['derived'].append({
                'timestamp': current_time,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'throughput': len(recent_executions) / 300  # processes per second
            })
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if not self.system_metrics['snapshots']:
            return {}
        
        latest_snapshot = self.system_metrics['snapshots'][-1]
        latest_derived = (self.system_metrics['derived'][-1] 
                         if self.system_metrics['derived'] else {})
        
        return {**latest_snapshot, **latest_derived}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current_metrics = self.get_current_metrics()
        
        # Calculate historical averages
        recent_snapshots = self.system_metrics['snapshots'][-12:]  # Last hour (5-min intervals)
        
        if recent_snapshots:
            avg_cpu = np.mean([s['cpu_usage'] for s in recent_snapshots])
            avg_memory = np.mean([s['memory_usage'] for s in recent_snapshots])
            max_cpu = max([s['cpu_usage'] for s in recent_snapshots])
            max_memory = max([s['memory_usage'] for s in recent_snapshots])
        else:
            avg_cpu = avg_memory = max_cpu = max_memory = 0
        
        # Count execution statistics
        total_executions = len(self.execution_metrics)
        successful_executions = len([m for m in self.execution_metrics.values() if m['success']])
        failed_executions = total_executions - successful_executions
        
        return {
            'current_state': current_metrics,
            'historical_averages': {
                'avg_cpu_usage': avg_cpu,
                'avg_memory_usage': avg_memory,
                'max_cpu_usage': max_cpu,
                'max_memory_usage': max_memory
            },
            'execution_statistics': {
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'failed_executions': failed_executions,
                'success_rate': successful_executions / max(1, total_executions)
            },
            'alerts_history': len(self.alerts_sent)
        }

# ============================================================================
# RESOURCE MANAGEMENT SYSTEM
# ============================================================================

class ResourceManager:
    """Manage system resources for CSP runtime"""
    
    def __init__(self, memory_limit_gb: float):
        self.memory_limit_bytes = memory_limit_gb * (1024**3)
        self.resource_pools = {}
        self.allocation_tracking = {}
        self.cleanup_tasks = []
    
    async def allocate_resources(self, resource_type: str, amount: float, 
                               requester_id: str) -> bool:
        """Allocate resources for process execution"""
        
        if resource_type == 'memory':
            return await self._allocate_memory(amount, requester_id)
        elif resource_type == 'cpu':
            return await self._allocate_cpu(amount, requester_id)
        else:
            logging.warning(f"Unknown resource type: {resource_type}")
            return False
    
    async def _allocate_memory(self, bytes_needed: float, requester_id: str) -> bool:
        """Allocate memory resources"""
        current_usage = psutil.virtual_memory().used
        
        if current_usage + bytes_needed > self.memory_limit_bytes:
            # Try to free up memory
            await self._cleanup_memory()
            current_usage = psutil.virtual_memory().used
            
            if current_usage + bytes_needed > self.memory_limit_bytes:
                logging.warning(f"Memory allocation failed for {requester_id}")
                return False
        
        # Track allocation
        self.allocation_tracking[requester_id] = {
            'memory_bytes': bytes_needed,
            'allocated_at': time.time()
        }
        
        return True
    
    async def _allocate_cpu(self, cpu_percent: float, requester_id: str) -> bool:
        """Allocate CPU resources"""
        # Simple CPU allocation - could be more sophisticated
        current_cpu = psutil.cpu_percent()
        
        if current_cpu + cpu_percent > 90.0:  # Leave 10% headroom
            logging.warning(f"CPU allocation failed for {requester_id} - current usage: {current_cpu}%")
            return False
        
        return True
    
    async def release_resources(self, requester_id: str):
        """Release resources allocated to a requester"""
        if requester_id in self.allocation_tracking:
            del self.allocation_tracking[requester_id]
    
    async def _cleanup_memory(self):
        """Cleanup memory by forcing garbage collection and removing old allocations"""
        import gc
        
        # Force garbage collection
        collected = gc.collect()
        logging.info(f"Garbage collection freed {collected} objects")
        
        # Remove stale allocations (older than 1 hour)
        current_time = time.time()
        stale_allocations = [
            req_id for req_id, alloc in self.allocation_tracking.items()
            if current_time - alloc['allocated_at'] > 3600
        ]
        
        for req_id in stale_allocations:
            del self.allocation_tracking[req_id]
            logging.info(f"Cleaned up stale allocation for {req_id}")

# ============================================================================
# DISTRIBUTED RUNTIME COORDINATOR
# ============================================================================

class DistributedRuntimeCoordinator:
    """Coordinate CSP runtime across multiple nodes"""
    
    def __init__(self, node_id: str, cluster_config: Dict[str, Any]):
        self.node_id = node_id
        self.cluster_config = cluster_config
        self.peer_nodes = {}
        self.distributed_processes = {}
        self.leader_node = None
        self.is_leader = False
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.heartbeat_interval = 5.0
        
    async def join_cluster(self):
        """Join distributed cluster"""
        # Discover peer nodes
        await self._discover_peers()
        
        # Start leader election
        await self._elect_leader()
        
        # Start coordination tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._coordination_loop())
        
        logging.info(f"Node {self.node_id} joined cluster")
    
    async def _discover_peers(self):
        """Discover other nodes in cluster"""
        peer_endpoints = self.cluster_config.get('peer_endpoints', [])
        
        for endpoint in peer_endpoints:
            try:
                # Ping peer node
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{endpoint}/health") as response:
                        if response.status == 200:
                            peer_info = await response.json()
                            peer_id = peer_info.get('node_id')
                            if peer_id != self.node_id:
                                self.peer_nodes[peer_id] = {
                                    'endpoint': endpoint,
                                    'last_seen': time.time(),
                                    'capabilities': peer_info.get('capabilities', [])
                                }
            except Exception as e:
                logging.warning(f"Failed to connect to peer {endpoint}: {e}")
    
    async def _elect_leader(self):
        """Elect cluster leader using simple node ID comparison"""
        all_nodes = [self.node_id] + list(self.peer_nodes.keys())
        all_nodes.sort()  # Deterministic ordering
        
        self.leader_node = all_nodes[0]
        self.is_leader = (self.leader_node == self.node_id)
        
        logging.info(f"Leader elected: {self.leader_node} (self: {self.is_leader})")
    
    async def _heartbeat_loop(self):
        """Send heartbeats to peer nodes"""
        while True:
            try:
                heartbeat_msg = {
                    'type': 'heartbeat',
                    'node_id': self.node_id,
                    'timestamp': time.time(),
                    'status': 'healthy'
                }
                
                # Send to all peers
                await self._broadcast_message(heartbeat_msg)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logging.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(self.heartbeat_interval * 2)
    
    async def _coordination_loop(self):
        """Main coordination loop"""
        while True:
            try:
                # Process coordination messages
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(), 
                        timeout=1.0
                    )
                    await self._handle_coordination_message(message)
                except asyncio.TimeoutError:
                    pass
                
                # Leader-specific tasks
                if self.is_leader:
                    await self._leader_coordination_tasks()
                
            except Exception as e:
                logging.error(f"Coordination loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all peer nodes"""
        for peer_id, peer_info in self.peer_nodes.items():
            try:
                endpoint = peer_info['endpoint']
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{endpoint}/coordination", 
                        json=message
                    ) as response:
                        if response.status != 200:
                            logging.warning(f"Failed to send message to {peer_id}")
            except Exception as e:
                logging.warning(f"Failed to broadcast to {peer_id}: {e}")
    
    async def _handle_coordination_message(self, message: Dict[str, Any]):
        """Handle coordination message from peer"""
        msg_type = message.get('type')
        
        if msg_type == 'heartbeat':
            peer_id = message.get('node_id')
            if peer_id in self.peer_nodes:
                self.peer_nodes[peer_id]['last_seen'] = time.time()
        
        elif msg_type == 'process_distribution':
            # Handle distributed process assignment
            process_assignment = message.get('assignment')
            await self._handle_process_assignment(process_assignment)
        
        elif msg_type == 'leader_election':
            # Handle leader election message
            await self._handle_leader_election(message)
    
    async def _leader_coordination_tasks(self):
        """Tasks performed by cluster leader"""
        # Check for failed nodes
        current_time = time.time()
        failed_nodes = [
            peer_id for peer_id, peer_info in self.peer_nodes.items()
            if current_time - peer_info['last_seen'] > self.heartbeat_interval * 3
        ]
        
        if failed_nodes:
            logging.warning(f"Detected failed nodes: {failed_nodes}")
            await self._handle_node_failures(failed_nodes)
        
        # Distribute load across cluster
        await self._distribute_cluster_load()
    
    async def _handle_node_failures(self, failed_nodes: List[str]):
        """Handle node failures"""
        for node_id in failed_nodes:
            # Remove from peer list
            if node_id in self.peer_nodes:
                del self.peer_nodes[node_id]
            
            # Redistribute processes from failed node
            if node_id in self.distributed_processes:
                processes = self.distributed_processes[node_id]
                await self._redistribute_processes(processes)
                del self.distributed_processes[node_id]
    
    async def _distribute_cluster_load(self):
        """Distribute load across cluster nodes"""
        # Simple load distribution - could be much more sophisticated
        total_nodes = len(self.peer_nodes) + 1  # Include self
        
        # Would implement actual load balancing logic here
        pass
    
    async def _redistribute_processes(self, processes: List[str]):
        """Redistribute processes from failed node"""
        # Reassign processes to healthy nodes
        healthy_nodes = list(self.peer_nodes.keys()) + [self.node_id]
        
        for i, process_id in enumerate(processes):
            target_node = healthy_nodes[i % len(healthy_nodes)]
            
            redistribution_msg = {
                'type': 'process_distribution',
                'assignment': {
                    'process_id': process_id,
                    'target_node': target_node
                }
            }
            
            await self._broadcast_message(redistribution_msg)
    
    async def _handle_process_assignment(self, assignment: Dict[str, Any]):
        """Handle process assignment to this node"""
        process_id = assignment.get('process_id')
        target_node = assignment.get('target_node')
        
        if target_node == self.node_id:
            # This process is assigned to us
            logging.info(f"Received process assignment: {process_id}")
            # Would implement actual process migration here
    
    async def _handle_leader_election(self, message: Dict[str, Any]):
        """Handle leader election message"""
        # Would implement proper leader election algorithm
        pass

# ============================================================================
# MAIN RUNTIME ORCHESTRATOR
# ============================================================================

class CSPRuntimeOrchestrator:
    """Main orchestrator for the entire CSP runtime system"""
    
    def __init__(self, config: RuntimeConfig, node_id: str = None):
        self.config = config
        self.node_id = node_id or f"node_{int(time.time())}"
        
        # Core components
        self.executor = CSPRuntimeExecutor(config)
        self.csp_engine = AdvancedCSPEngineWithAI()
        
        # Optional distributed coordination
        self.distributed_coordinator = None
        if config.execution_model == ExecutionModel.DISTRIBUTED:
            cluster_config = getattr(config, 'cluster_config', {})
            self.distributed_coordinator = DistributedRuntimeCoordinator(
                self.node_id, cluster_config
            )
        
        # Runtime state
        self.is_running = False
        self.shutdown_handlers = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def start(self):
        """Start the complete CSP runtime system"""
        if self.is_running:
            return
        
        logging.info(f"Starting CSP Runtime Orchestrator (Node: {self.node_id})")
        
        # Start core components
        await self.executor.start()
        # CSP engine is started implicitly
        
        # Start distributed coordination if configured
        if self.distributed_coordinator:
            await self.distributed_coordinator.join_cluster()
        
        self.is_running = True
        
        logging.info("CSP Runtime Orchestrator started successfully")
    
    async def stop(self):
        """Stop the complete CSP runtime system"""
        if not self.is_running:
            return
        
        logging.info("Stopping CSP Runtime Orchestrator...")
        
        # Run shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                await handler()
            except Exception as e:
                logging.error(f"Shutdown handler error: {e}")
        
        # Stop core components
        await self.executor.stop()
        await self.csp_engine.base_engine.shutdown()
        
        self.is_running = False
        
        logging.info("CSP Runtime Orchestrator stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logging.info(f"Received signal {signum}, initiating shutdown...")
        
        # Create shutdown task
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self.stop())
        else:
            loop.run_until_complete(self.stop())
    
    def add_shutdown_handler(self, handler: Callable):
        """Add custom shutdown handler"""
        self.shutdown_handlers.append(handler)
    
    async def execute_csp_process(self, process: Process, priority: int = 5) -> str:
        """Execute CSP process through the runtime"""
        return await self.executor.execute_process(
            process, 
            self.csp_engine.base_engine.context, 
            priority
        )
    
    def get_runtime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive runtime statistics"""
        performance_report = self.executor.performance_monitor.get_performance_report()
        
        stats = {
            'node_id': self.node_id,
            'runtime_config': {
                'execution_model': self.config.execution_model.name,
                'scheduling_policy': self.config.scheduling_policy.name,
                'max_workers': self.config.max_workers
            },
            'performance': performance_report,
            'executor_stats': self.executor.execution_stats.copy(),
            'uptime': time.time() - getattr(self, 'start_time', time.time())
        }
        
        if self.distributed_coordinator:
            stats['cluster'] = {
                'is_leader': self.distributed_coordinator.is_leader,
                'leader_node': self.distributed_coordinator.leader_node,
                'peer_count': len(self.distributed_coordinator.peer_nodes)
            }
        
        return stats

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

async def demonstrate_runtime_system():
    """Demonstrate the complete runtime system"""
    
    # Configure runtime
    config = RuntimeConfig(
        execution_model=ExecutionModel.MULTI_THREADED,
        scheduling_policy=SchedulingPolicy.ADAPTIVE,
        max_workers=4,
        memory_limit_gb=4.0,
        enable_monitoring=True,
        enable_optimization=True
    )
    
    # Create and start orchestrator
    orchestrator = CSPRuntimeOrchestrator(config)
    
    try:
        await orchestrator.start()
        
        # Create some test processes
        from csp_ai_integration import AIAgent, LLMCapability, CollaborativeAIProcess
        
        # Create AI agent and process
        llm_capability = LLMCapability("test-model", "general")
        ai_agent = AIAgent("test_agent", [llm_capability])
        ai_process = CollaborativeAIProcess("ai_proc_1", ai_agent)
        
        # Execute process through runtime
        execution_id = await orchestrator.execute_csp_process(ai_process, priority=3)
        
        print(f"Process execution started: {execution_id}")
        
        # Wait a bit for execution
        await asyncio.sleep(2.0)
        
        # Get runtime statistics
        stats = orchestrator.get_runtime_statistics()
        print("Runtime Statistics:")
        print(json.dumps(stats, indent=2, default=str))
        
    finally:
        await orchestrator.stop()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(demonstrate_runtime_system())
