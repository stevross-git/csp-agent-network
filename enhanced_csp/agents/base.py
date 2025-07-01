# enhanced_csp/agents/base.py
"""
Enhanced Base Agent with Network Optimization Support
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod

from ..ai_comm import AdvancedAICommChannel, AdvancedCommPattern

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base agent class with network optimization capabilities"""
    
    def __init__(self, 
                 agent_id: str,
                 config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.config = config or {}
        
        # Communication channels
        self.channels: Dict[str, AdvancedAICommChannel] = {}
        
        # Network optimization settings
        self.network_optimization = self.config.get('network_optimization', {
            'enabled': True,
            'compression': {'default_algorithm': 'lz4'},
            'batching': {'max_wait_ms': 50},
            'adaptive': {'enabled': True}
        })
        
        # Message handlers
        self.message_handlers = {}
        self._setup_default_handlers()
        
        # Lifecycle
        self._running = False
        self._message_processor_task = None
        
    def _setup_default_handlers(self):
        """Setup default message handlers"""
        self.message_handlers['ping'] = self._handle_ping
        self.message_handlers['status'] = self._handle_status
        
    async def _handle_ping(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping messages"""
        return {
            'type': 'pong',
            'agent_id': self.agent_id,
            'timestamp': asyncio.get_event_loop().time()
        }
        
    async def _handle_status(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status request"""
        return {
            'type': 'status_response',
            'agent_id': self.agent_id,
            'running': self._running,
            'channels': list(self.channels.keys()),
            'network_optimization': self.network_optimization['enabled']
        }
        
    async def create_channel(self, 
                           channel_id: str,
                           pattern: AdvancedCommPattern,
                           endpoint: Optional[str] = None) -> AdvancedAICommChannel:
        """Create optimized communication channel"""
        channel_config = {
            'endpoint': endpoint or f"http://csp-router/{channel_id}",
            'network': self.network_optimization
        }
        
        channel = AdvancedAICommChannel(
            channel_id=f"{self.agent_id}_{channel_id}",
            pattern=pattern,
            config=channel_config
        )
        
        self.channels[channel_id] = channel
        
        # Start channel if agent is running
        if self._running:
            await channel.start()
            
        return channel
        
    async def send_message(self,
                          channel_id: str,
                          recipient: str,
                          message: Dict[str, Any],
                          priority: int = 0) -> bool:
        """Send message through specified channel"""
        if channel_id not in self.channels:
            logger.error(f"Channel {channel_id} not found")
            return False
            
        return await self.channels[channel_id].send_message(
            recipient, message, priority
        )
        
    async def broadcast_message(self,
                               message: Dict[str, Any],
                               channel_ids: Optional[List[str]] = None) -> Dict[str, bool]:
        """Broadcast message to multiple channels"""
        if channel_ids is None:
            channel_ids = list(self.channels.keys())
            
        results = {}
        tasks = []
        
        for channel_id in channel_ids:
            if channel_id in self.channels:
                task = self.channels[channel_id].send_message(
                    "*",  # Broadcast recipient
                    message,
                    priority=5  # Medium priority for broadcasts
                )
                tasks.append((channel_id, task))
                
        # Send concurrently
        for channel_id, task in tasks:
            results[channel_id] = await task
            
        return results
        
    async def start(self):
        """Start the agent"""
        if self._running:
            return
            
        self._running = True
        
        # Start all channels
        for channel in self.channels.values():
            await channel.start()
            
        # Start message processor
        self._message_processor_task = asyncio.create_task(self._process_messages())
        
        logger.info(f"Agent {self.agent_id} started")
        
    async def stop(self):
        """Stop the agent"""
        self._running = False
        
        # Stop message processor
        if self._message_processor_task:
            self._message_processor_task.cancel()
            try:
                await self._message_processor_task
            except asyncio.CancelledError:
                pass
                
        # Stop all channels
        for channel in self.channels.values():
            await channel.stop()
            
        logger.info(f"Agent {self.agent_id} stopped")
        
    async def _process_messages(self):
        """Process incoming messages from all channels"""
        # NOTE: This is a simplified implementation. In production, you would:
        # 1. Poll each channel's receive_message() method
        # 2. Route messages to appropriate handlers
        # 3. Handle errors and retries
        # 4. Implement proper message acknowledgment
        
        logger.info(f"Message processor started for agent {self.agent_id}")
        
        while self._running:
            try:
                # Process messages from each channel
                for channel_id, channel in self.channels.items():
                    message = await channel.receive_message()
                    if message:
                        await self._handle_message(channel_id, message)
                        
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                await asyncio.sleep(1.0)  # Back off on error
                
    async def _handle_message(self, channel_id: str, message: Dict[str, Any]):
        """Handle received message"""
        try:
            # Extract message type
            msg_type = message.get('payload', {}).get('type', 'unknown')
            
            # Route to handler
            handler = self.message_handlers.get(msg_type)
            if handler:
                response = await handler(message)
                
                # Send response if needed
                if response and 'sender' in message:
                    await self.send_message(
                        channel_id,
                        message['sender'],
                        response,
                        priority=5
                    )
            else:
                logger.warning(f"No handler for message type: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
                
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task - must be implemented by subclasses"""
        pass