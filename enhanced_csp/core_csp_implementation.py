#!/usr/bin/env python3
"""
Complete Core CSP Implementation - The Heart of the Enhanced CSP System
======================================================================

This module implements the complete CSP (Communicating Sequential Processes) engine
with formal semantics, process algebra, and advanced communication patterns.

Features:
- Formal CSP process algebra implementation
- Process composition and parallel execution
- Channel-based communication with various semantics
- Event synchronization and message passing
- Process lifecycle management
- Deadlock detection and prevention
- Performance monitoring and optimization
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
import json
import weakref
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CSP CORE TYPES AND ENUMS
# ============================================================================

class ProcessState(Enum):
    """Process execution states"""
    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    TERMINATED = "terminated"
    ERROR = "error"

class ChannelType(Enum):
    """Channel communication types"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BUFFERED = "buffered"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"

class EventType(Enum):
    """CSP event types"""
    COMMUNICATION = "communication"
    CHOICE = "choice"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ITERATION = "iteration"
    TERMINATION = "termination"

class CompositionOperator(Enum):
    """CSP composition operators"""
    SEQUENTIAL = ";"       # P ; Q
    CHOICE = "[]"          # P [] Q  
    PARALLEL = "||"        # P || Q
    INTERLEAVING = "|||"   # P ||| Q
    HIDING = "\\"          # P \ A
    RENAMING = "[[]]"      # P[[old := new]]

# ============================================================================
# CSP EVENT SYSTEM
# ============================================================================

@dataclass
class CSPEvent:
    """CSP event with formal semantics"""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    event_type: EventType = EventType.COMMUNICATION
    data: Any = None
    source_process: Optional[str] = None
    target_process: Optional[str] = None
    channel: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    synchronization_set: Set[str] = field(default_factory=set)
    priority: int = 0
    
    def __post_init__(self):
        if not self.name:
            self.name = f"event_{self.event_id[:8]}"
    
    def matches(self, other: 'CSPEvent') -> bool:
        """Check if events can synchronize"""
        return (self.name == other.name and 
                self.channel == other.channel and
                self != other)
    
    def can_synchronize_with(self, other: 'CSPEvent') -> bool:
        """Check synchronization compatibility"""
        if not self.matches(other):
            return False
        
        # Check synchronization sets
        if self.synchronization_set and other.synchronization_set:
            return bool(self.synchronization_set.intersection(other.synchronization_set))
        
        return True

# ============================================================================
# CSP CHANNEL IMPLEMENTATION
# ============================================================================

class CSPChannel:
    """CSP channel with various communication semantics"""
    
    def __init__(self, name: str, channel_type: ChannelType = ChannelType.SYNCHRONOUS, 
                 buffer_size: int = 0, capacity: Optional[int] = None):
        self.name = name
        self.channel_type = channel_type
        self.buffer_size = buffer_size
        self.capacity = capacity or float('inf')
        
        # Channel state
        self.buffer = deque(maxlen=buffer_size if buffer_size > 0 else None)
        self.waiting_senders = deque()
        self.waiting_receivers = deque()
        self.subscribers = set()  # For broadcast channels
        
        # Synchronization primitives
        self.send_event = asyncio.Event()
        self.receive_event = asyncio.Event()
        self.lock = asyncio.Lock()
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.total_wait_time = 0.0
        
    async def send(self, event: CSPEvent, sender_id: str, timeout: Optional[float] = None) -> bool:
        """Send event through channel with CSP semantics"""
        
        async with self.lock:
            start_time = time.time()
            
            try:
                if self.channel_type == ChannelType.SYNCHRONOUS:
                    return await self._synchronous_send(event, sender_id, timeout)
                elif self.channel_type == ChannelType.ASYNCHRONOUS:
                    return await self._asynchronous_send(event, sender_id, timeout)
                elif self.channel_type == ChannelType.BUFFERED:
                    return await self._buffered_send(event, sender_id, timeout)
                elif self.channel_type == ChannelType.BROADCAST:
                    return await self._broadcast_send(event, sender_id, timeout)
                else:
                    return await self._asynchronous_send(event, sender_id, timeout)
                    
            finally:
                wait_time = time.time() - start_time
                self.total_wait_time += wait_time
                self.messages_sent += 1
    
    async def receive(self, receiver_id: str, timeout: Optional[float] = None) -> Optional[CSPEvent]:
        """Receive event from channel with CSP semantics"""
        
        async with self.lock:
            start_time = time.time()
            
            try:
                if self.channel_type == ChannelType.SYNCHRONOUS:
                    return await self._synchronous_receive(receiver_id, timeout)
                elif self.channel_type == ChannelType.ASYNCHRONOUS:
                    return await self._asynchronous_receive(receiver_id, timeout)
                elif self.channel_type == ChannelType.BUFFERED:
                    return await self._buffered_receive(receiver_id, timeout)
                elif self.channel_type == ChannelType.BROADCAST:
                    return await self._broadcast_receive(receiver_id, timeout)
                else:
                    return await self._asynchronous_receive(receiver_id, timeout)
                    
            finally:
                wait_time = time.time() - start_time
                self.total_wait_time += wait_time
                self.messages_received += 1
    
    async def _synchronous_send(self, event: CSPEvent, sender_id: str, timeout: Optional[float]) -> bool:
        """Synchronous send - waits for receiver"""
        
        # Add to waiting senders
        send_request = {
            'event': event,
            'sender_id': sender_id,
            'future': asyncio.get_event_loop().create_future()
        }
        self.waiting_senders.append(send_request)
        
        # Check for waiting receivers
        if self.waiting_receivers:
            receive_request = self.waiting_receivers.popleft()
            receive_request['future'].set_result(event)
            send_request['future'].set_result(True)
            return True
        
        # Wait for receiver
        try:
            if timeout:
                return await asyncio.wait_for(send_request['future'], timeout)
            else:
                return await send_request['future']
        except asyncio.TimeoutError:
            # Remove from waiting senders
            try:
                self.waiting_senders.remove(send_request)
            except ValueError:
                pass
            return False
    
    async def _synchronous_receive(self, receiver_id: str, timeout: Optional[float]) -> Optional[CSPEvent]:
        """Synchronous receive - waits for sender"""
        
        # Check for waiting senders
        if self.waiting_senders:
            send_request = self.waiting_senders.popleft()
            send_request['future'].set_result(True)
            return send_request['event']
        
        # Add to waiting receivers
        receive_request = {
            'receiver_id': receiver_id,
            'future': asyncio.get_event_loop().create_future()
        }
        self.waiting_receivers.append(receive_request)
        
        # Wait for sender
        try:
            if timeout:
                return await asyncio.wait_for(receive_request['future'], timeout)
            else:
                return await receive_request['future']
        except asyncio.TimeoutError:
            # Remove from waiting receivers
            try:
                self.waiting_receivers.remove(receive_request)
            except ValueError:
                pass
            return None
    
    async def _buffered_send(self, event: CSPEvent, sender_id: str, timeout: Optional[float]) -> bool:
        """Buffered send - queues messages up to buffer size"""
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(event)
            # Notify waiting receivers
            if self.waiting_receivers:
                receive_request = self.waiting_receivers.popleft()
                buffered_event = self.buffer.popleft()
                receive_request['future'].set_result(buffered_event)
            return True
        
        # Buffer full - wait for space
        send_request = {
            'event': event,
            'sender_id': sender_id,
            'future': asyncio.get_event_loop().create_future()
        }
        self.waiting_senders.append(send_request)
        
        try:
            if timeout:
                return await asyncio.wait_for(send_request['future'], timeout)
            else:
                return await send_request['future']
        except asyncio.TimeoutError:
            try:
                self.waiting_senders.remove(send_request)
            except ValueError:
                pass
            return False
    
    async def _buffered_receive(self, receiver_id: str, timeout: Optional[float]) -> Optional[CSPEvent]:
        """Buffered receive - gets from buffer or waits"""
        
        if self.buffer:
            event = self.buffer.popleft()
            # Notify waiting senders if buffer has space
            if self.waiting_senders and len(self.buffer) < self.buffer_size:
                send_request = self.waiting_senders.popleft()
                self.buffer.append(send_request['event'])
                send_request['future'].set_result(True)
            return event
        
        # Buffer empty - wait for message
        receive_request = {
            'receiver_id': receiver_id,
            'future': asyncio.get_event_loop().create_future()
        }
        self.waiting_receivers.append(receive_request)
        
        try:
            if timeout:
                return await asyncio.wait_for(receive_request['future'], timeout)
            else:
                return await receive_request['future']
        except asyncio.TimeoutError:
            try:
                self.waiting_receivers.remove(receive_request)
            except ValueError:
                pass
            return None
    
    async def _asynchronous_send(self, event: CSPEvent, sender_id: str, timeout: Optional[float]) -> bool:
        """Asynchronous send - immediate return"""
        self.buffer.append(event)
        
        # Notify waiting receivers
        if self.waiting_receivers:
            receive_request = self.waiting_receivers.popleft()
            buffered_event = self.buffer.popleft()
            receive_request['future'].set_result(buffered_event)
        
        return True
    
    async def _asynchronous_receive(self, receiver_id: str, timeout: Optional[float]) -> Optional[CSPEvent]:
        """Asynchronous receive - waits if no messages"""
        
        if self.buffer:
            return self.buffer.popleft()
        
        # No messages - wait
        receive_request = {
            'receiver_id': receiver_id,
            'future': asyncio.get_event_loop().create_future()
        }
        self.waiting_receivers.append(receive_request)
        
        try:
            if timeout:
                return await asyncio.wait_for(receive_request['future'], timeout)
            else:
                return await receive_request['future']
        except asyncio.TimeoutError:
            try:
                self.waiting_receivers.remove(receive_request)
            except ValueError:
                pass
            return None
    
    async def _broadcast_send(self, event: CSPEvent, sender_id: str, timeout: Optional[float]) -> bool:
        """Broadcast send - sends to all subscribers"""
        
        if not self.subscribers:
            return True  # No subscribers, but send succeeds
        
        # Send to all waiting receivers
        sent_count = 0
        for receive_request in list(self.waiting_receivers):
            if receive_request['receiver_id'] in self.subscribers:
                receive_request['future'].set_result(event)
                self.waiting_receivers.remove(receive_request)
                sent_count += 1
        
        return sent_count > 0 or not self.subscribers
    
    async def _broadcast_receive(self, receiver_id: str, timeout: Optional[float]) -> Optional[CSPEvent]:
        """Broadcast receive - subscribes and waits"""
        
        self.subscribers.add(receiver_id)
        
        receive_request = {
            'receiver_id': receiver_id,
            'future': asyncio.get_event_loop().create_future()
        }
        self.waiting_receivers.append(receive_request)
        
        try:
            if timeout:
                return await asyncio.wait_for(receive_request['future'], timeout)
            else:
                return await receive_request['future']
        except asyncio.TimeoutError:
            try:
                self.waiting_receivers.remove(receive_request)
            except ValueError:
                pass
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get channel statistics"""
        return {
            'name': self.name,
            'type': self.channel_type.value,
            'buffer_size': self.buffer_size,
            'current_buffer_length': len(self.buffer),
            'waiting_senders': len(self.waiting_senders),
            'waiting_receivers': len(self.waiting_receivers),
            'subscribers': len(self.subscribers),
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'average_wait_time': self.total_wait_time / max(self.messages_sent + self.messages_received, 1)
        }

# ============================================================================
# CSP PROCESS IMPLEMENTATION
# ============================================================================

class CSPProcess(ABC):
    """Abstract base class for CSP processes"""
    
    def __init__(self, name: str, process_id: Optional[str] = None):
        self.name = name
        self.process_id = process_id or str(uuid.uuid4())
        self.state = ProcessState.CREATED
        self.parent_process = None
        self.child_processes = []
        
        # Communication
        self.input_channels = {}
        self.output_channels = {}
        self.event_handlers = {}
        
        # Execution context
        self.context = {}
        self.local_variables = {}
        self.execution_history = []
        
        # Performance metrics
        self.start_time = None
        self.end_time = None
        self.cpu_time = 0.0
        self.events_processed = 0
        
        # Synchronization
        self.state_lock = asyncio.Lock()
        self.termination_event = asyncio.Event()
        
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the process - must be implemented by subclasses"""
        pass
    
    async def start(self, context: Optional[Dict[str, Any]] = None) -> Any:
        """Start process execution"""
        
        async with self.state_lock:
            if self.state != ProcessState.CREATED:
                raise RuntimeError(f"Process {self.name} cannot be started from state {self.state}")
            
            self.state = ProcessState.READY
            self.start_time = time.time()
        
        try:
            self.state = ProcessState.RUNNING
            result = await self.execute(context or {})
            
            async with self.state_lock:
                self.state = ProcessState.TERMINATED
                self.end_time = time.time()
                self.termination_event.set()
            
            return result
            
        except Exception as e:
            async with self.state_lock:
                self.state = ProcessState.ERROR
                self.end_time = time.time()
                self.termination_event.set()
            
            logger.error(f"Process {self.name} failed: {e}")
            raise
    
    async def stop(self, timeout: Optional[float] = None):
        """Stop process execution"""
        
        async with self.state_lock:
            if self.state in [ProcessState.TERMINATED, ProcessState.ERROR]:
                return
            
            # Attempt graceful shutdown
            self.state = ProcessState.TERMINATED
            self.termination_event.set()
        
        # Wait for process to finish
        if timeout:
            try:
                await asyncio.wait_for(self.termination_event.wait(), timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Process {self.name} did not stop gracefully within {timeout}s")
    
    def add_input_channel(self, name: str, channel: CSPChannel):
        """Add input channel"""
        self.input_channels[name] = channel
    
    def add_output_channel(self, name: str, channel: CSPChannel):
        """Add output channel"""
        self.output_channels[name] = channel
    
    async def send_event(self, channel_name: str, event: CSPEvent, timeout: Optional[float] = None) -> bool:
        """Send event through output channel"""
        
        if channel_name not in self.output_channels:
            raise ValueError(f"Output channel '{channel_name}' not found")
        
        channel = self.output_channels[channel_name]
        event.source_process = self.process_id
        
        success = await channel.send(event, self.process_id, timeout)
        if success:
            self.events_processed += 1
            self.execution_history.append({
                'type': 'send',
                'channel': channel_name,
                'event': event.name,
                'timestamp': time.time()
            })
        
        return success
    
    async def receive_event(self, channel_name: str, timeout: Optional[float] = None) -> Optional[CSPEvent]:
        """Receive event from input channel"""
        
        if channel_name not in self.input_channels:
            raise ValueError(f"Input channel '{channel_name}' not found")
        
        channel = self.input_channels[channel_name]
        event = await channel.receive(self.process_id, timeout)
        
        if event:
            event.target_process = self.process_id
            self.events_processed += 1
            self.execution_history.append({
                'type': 'receive',
                'channel': channel_name,
                'event': event.name,
                'timestamp': time.time()
            })
        
        return event
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get process statistics"""
        
        execution_time = None
        if self.start_time:
            end_time = self.end_time or time.time()
            execution_time = end_time - self.start_time
        
        return {
            'process_id': self.process_id,
            'name': self.name,
            'state': self.state.value,
            'execution_time': execution_time,
            'events_processed': self.events_processed,
            'input_channels': len(self.input_channels),
            'output_channels': len(self.output_channels),
            'child_processes': len(self.child_processes)
        }

# ============================================================================
# ATOMIC PROCESS IMPLEMENTATION
# ============================================================================

class AtomicProcess(CSPProcess):
    """Atomic CSP process - indivisible unit of computation"""
    
    def __init__(self, name: str, behavior: Optional[Callable] = None):
        super().__init__(name)
        self.behavior = behavior or self.default_behavior
        self.guards = []  # Guard conditions for choice
        
    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute atomic process behavior"""
        
        self.context.update(context)
        
        try:
            result = await self.behavior(self)
            return result
        except Exception as e:
            logger.error(f"Atomic process {self.name} execution failed: {e}")
            raise
    
    async def default_behavior(self, process: 'AtomicProcess') -> Any:
        """Default behavior - STOP process"""
        logger.info(f"Process {self.name} executing STOP")
        return None
    
    def add_guard(self, condition: Callable[[], bool], action: Callable):
        """Add guard condition for choice operator"""
        self.guards.append({'condition': condition, 'action': action})
    
    async def choice_behavior(self) -> Any:
        """Execute choice behavior with guards"""
        
        # Evaluate guards
        available_choices = []
        for guard in self.guards:
            if guard['condition']():
                available_choices.append(guard)
        
        if not available_choices:
            # No guards satisfied - deadlock
            logger.warning(f"Process {self.name} deadlocked - no guards satisfied")
            self.state = ProcessState.BLOCKED
            return None
        
        # Non-deterministic choice - select first available
        chosen_guard = available_choices[0]
        return await chosen_guard['action']()

# ============================================================================
# COMPOSITE PROCESS IMPLEMENTATION
# ============================================================================

class CompositeProcess(CSPProcess):
    """Composite CSP process - combination of other processes"""
    
    def __init__(self, name: str, operator: CompositionOperator):
        super().__init__(name)
        self.operator = operator
        self.component_processes = []
        self.synchronization_alphabet = set()
        
    def add_component(self, process: CSPProcess):
        """Add component process"""
        self.component_processes.append(process)
        process.parent_process = self
        self.child_processes.append(process)
    
    def set_synchronization_alphabet(self, alphabet: Set[str]):
        """Set synchronization alphabet for parallel composition"""
        self.synchronization_alphabet = alphabet
    
    async def execute(self, context: Dict[str, Any]) -> Any:
        """Execute composite process based on operator"""
        
        self.context.update(context)
        
        if self.operator == CompositionOperator.SEQUENTIAL:
            return await self._execute_sequential()
        elif self.operator == CompositionOperator.PARALLEL:
            return await self._execute_parallel()
        elif self.operator == CompositionOperator.CHOICE:
            return await self._execute_choice()
        elif self.operator == CompositionOperator.INTERLEAVING:
            return await self._execute_interleaving()
        else:
            raise NotImplementedError(f"Operator {self.operator} not implemented")
    
    async def _execute_sequential(self) -> Any:
        """Execute processes sequentially (P ; Q)"""
        
        results = []
        for process in self.component_processes:
            result = await process.start(self.context)
            results.append(result)
            
            # If process terminated abnormally, stop execution
            if process.state == ProcessState.ERROR:
                break
        
        return results
    
    async def _execute_parallel(self) -> Any:
        """Execute processes in parallel with synchronization (P || Q)"""
        
        # Create tasks for all component processes
        tasks = []
        for process in self.component_processes:
            task = asyncio.create_task(process.start(self.context))
            tasks.append(task)
        
        # Wait for all processes to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def _execute_choice(self) -> Any:
        """Execute choice between processes (P [] Q)"""
        
        # Create tasks for all component processes
        tasks = []
        for process in self.component_processes:
            task = asyncio.create_task(process.start(self.context))
            tasks.append(task)
        
        # Wait for first process to complete
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
        
        # Return result from first completed process
        completed_task = next(iter(done))
        return await completed_task
    
    async def _execute_interleaving(self) -> Any:
        """Execute processes with interleaving (P ||| Q)"""
        
        # Similar to parallel but without synchronization
        tasks = []
        for process in self.component_processes:
            task = asyncio.create_task(process.start(self.context))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

# ============================================================================
# CSP ENGINE IMPLEMENTATION
# ============================================================================

class CSPEngine:
    """Main CSP engine for managing processes and channels"""
    
    def __init__(self, name: str = "CSPEngine"):
        self.name = name
        self.processes = {}
        self.channels = {}
        self.running = False
        
        # Execution context
        self.global_context = {}
        self.event_loop = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Monitoring
        self.start_time = None
        self.statistics = {
            'processes_created': 0,
            'processes_completed': 0,
            'channels_created': 0,
            'events_processed': 0,
            'deadlocks_detected': 0
        }
        
        # Deadlock detection
        self.deadlock_detector = DeadlockDetector(self)
        
    async def start(self):
        """Start the CSP engine"""
        
        if self.running:
            return
        
        self.running = True
        self.start_time = time.time()
        self.event_loop = asyncio.get_event_loop()
        
        logger.info(f"CSP Engine '{self.name}' started")
    
    async def stop(self):
        """Stop the CSP engine"""
        
        if not self.running:
            return
        
        # Stop all running processes
        stop_tasks = []
        for process in self.processes.values():
            if process.state == ProcessState.RUNNING:
                stop_tasks.append(process.stop(timeout=5.0))
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.running = False
        logger.info(f"CSP Engine '{self.name}' stopped")
    
    def create_process(self, name: str, process_type: str = "atomic", 
                      behavior: Optional[Callable] = None) -> CSPProcess:
        """Create a new CSP process"""
        
        if process_type == "atomic":
            process = AtomicProcess(name, behavior)
        elif process_type == "composite":
            process = CompositeProcess(name, CompositionOperator.SEQUENTIAL)
        else:
            raise ValueError(f"Unknown process type: {process_type}")
        
        self.processes[process.process_id] = process
        self.statistics['processes_created'] += 1
        
        logger.info(f"Created {process_type} process '{name}' with ID {process.process_id}")
        return process
    
    def create_channel(self, name: str, channel_type: ChannelType = ChannelType.SYNCHRONOUS,
                      buffer_size: int = 0) -> CSPChannel:
        """Create a new CSP channel"""
        
        channel = CSPChannel(name, channel_type, buffer_size)
        self.channels[name] = channel
        self.statistics['channels_created'] += 1
        
        logger.info(f"Created {channel_type.value} channel '{name}'")
        return channel
    
    def connect_processes(self, sender: CSPProcess, receiver: CSPProcess, 
                         channel_name: str, channel_type: ChannelType = ChannelType.SYNCHRONOUS):
        """Connect two processes with a channel"""
        
        if channel_name not in self.channels:
            channel = self.create_channel(channel_name, channel_type)
        else:
            channel = self.channels[channel_name]
        
        sender.add_output_channel(channel_name, channel)
        receiver.add_input_channel(channel_name, channel)
        
        logger.info(f"Connected processes {sender.name} -> {receiver.name} via channel '{channel_name}'")
    
    async def run_process(self, process: CSPProcess, context: Optional[Dict[str, Any]] = None) -> Any:
        """Run a process"""
        
        if not self.running:
            raise RuntimeError("CSP Engine is not running")
        
        context = context or {}
        context.update(self.global_context)
        
        try:
            result = await process.start(context)
            
            if process.state == ProcessState.TERMINATED:
                self.statistics['processes_completed'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Process execution failed: {e}")
            raise
    
    async def run_parallel_processes(self, processes: List[CSPProcess]) -> List[Any]:
        """Run multiple processes in parallel"""
        
        tasks = []
        for process in processes:
            task = asyncio.create_task(self.run_process(process))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    def get_process(self, process_id: str) -> Optional[CSPProcess]:
        """Get process by ID"""
        return self.processes.get(process_id)
    
    def get_channel(self, channel_name: str) -> Optional[CSPChannel]:
        """Get channel by name"""
        return self.channels.get(channel_name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics"""
        
        uptime = None
        if self.start_time:
            uptime = time.time() - self.start_time
        
        # Aggregate process statistics
        process_states = defaultdict(int)
        total_events = 0
        
        for process in self.processes.values():
            process_states[process.state.value] += 1
            total_events += process.events_processed
        
        # Aggregate channel statistics
        channel_stats = {}
        for name, channel in self.channels.items():
            channel_stats[name] = channel.get_statistics()
        
        return {
            'engine_name': self.name,
            'running': self.running,
            'uptime': uptime,
            'processes': dict(process_states),
            'channels': len(self.channels),
            'total_events_processed': total_events,
            'statistics': self.statistics,
            'channel_details': channel_stats
        }

# ============================================================================
# DEADLOCK DETECTION
# ============================================================================

class DeadlockDetector:
    """Deadlock detection for CSP processes"""
    
    def __init__(self, engine: CSPEngine):
        self.engine = engine
        self.detection_interval = 5.0  # seconds
        self.monitoring_task = None
    
    async def start_monitoring(self):
        """Start deadlock monitoring"""
        if self.monitoring_task:
            return
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop deadlock monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
    
    async def _monitoring_loop(self):
        """Background deadlock detection loop"""
        
        while True:
            try:
                deadlocked_processes = await self.detect_deadlock()
                
                if deadlocked_processes:
                    logger.warning(f"Deadlock detected in processes: {[p.name for p in deadlocked_processes]}")
                    self.engine.statistics['deadlocks_detected'] += 1
                    await self._handle_deadlock(deadlocked_processes)
                
                await asyncio.sleep(self.detection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Deadlock detection error: {e}")
                await asyncio.sleep(self.detection_interval)
    
    async def detect_deadlock(self) -> List[CSPProcess]:
        """Detect deadlocked processes"""
        
        blocked_processes = []
        
        # Find blocked processes
        for process in self.engine.processes.values():
            if process.state == ProcessState.BLOCKED:
                blocked_processes.append(process)
        
        if not blocked_processes:
            return []
        
        # Check for circular wait conditions
        deadlocked = []
        for process in blocked_processes:
            if await self._is_in_circular_wait(process):
                deadlocked.append(process)
        
        return deadlocked
    
    async def _is_in_circular_wait(self, process: CSPProcess) -> bool:
        """Check if process is in circular wait"""
        
        # Simplified deadlock detection - check if process has been blocked for too long
        # In a more sophisticated implementation, this would build a wait-for graph
        
        if process.state != ProcessState.BLOCKED:
            return False
        
        # Check if process has been blocked for more than 30 seconds
        if hasattr(process, 'block_start_time'):
            block_duration = time.time() - process.block_start_time
            return block_duration > 30.0
        
        # Mark block start time
        process.block_start_time = time.time()
        return False
    
    async def _handle_deadlock(self, deadlocked_processes: List[CSPProcess]):
        """Handle detected deadlock"""
        
        logger.warning(f"Handling deadlock for {len(deadlocked_processes)} processes")
        
        # Simple deadlock resolution - terminate one process
        if deadlocked_processes:
            process_to_terminate = deadlocked_processes[0]
            logger.warning(f"Terminating process {process_to_terminate.name} to resolve deadlock")
            await process_to_terminate.stop(timeout=1.0)

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

async def create_simple_csp_example():
    """Create a simple CSP example"""
    
    # Create CSP engine
    engine = CSPEngine("ExampleEngine")
    await engine.start()
    
    # Create processes
    producer = engine.create_process("Producer", "atomic")
    consumer = engine.create_process("Consumer", "atomic")
    
    # Define producer behavior
    async def producer_behavior(process: AtomicProcess):
        for i in range(5):
            event = CSPEvent(name=f"data_{i}", data=f"message_{i}")
            await process.send_event("output", event)
            logger.info(f"Producer sent: {event.data}")
            await asyncio.sleep(1)
        return "Producer completed"
    
    # Define consumer behavior
    async def consumer_behavior(process: AtomicProcess):
        received_messages = []
        for i in range(5):
            event = await process.receive_event("input")
            if event:
                received_messages.append(event.data)
                logger.info(f"Consumer received: {event.data}")
        return received_messages
    
    producer.behavior = producer_behavior
    consumer.behavior = consumer_behavior
    
    # Connect processes
    engine.connect_processes(producer, consumer, "data_channel", ChannelType.SYNCHRONOUS)
    
    # Run processes in parallel
    results = await engine.run_parallel_processes([producer, consumer])
    
    # Print results
    logger.info(f"Producer result: {results[0]}")
    logger.info(f"Consumer result: {results[1]}")
    
    # Print statistics
    stats = engine.get_statistics()
    logger.info(f"Engine statistics: {json.dumps(stats, indent=2, default=str)}")
    
    await engine.stop()
    return results

# Main execution
if __name__ == "__main__":
    async def main():
        logger.info("Starting Enhanced CSP System Core Demo")
        results = await create_simple_csp_example()
        logger.info("Demo completed successfully")
        return results
    
    asyncio.run(main())
