# enhanced_csp/ai_comm/channel.py
"""
Enhanced AI Communication Channel with Network Optimization
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from .base import BaseCommChannel
from ..network import (
    OptimizedNetworkChannel, 
    NetworkMetricsCollector,
    CompressionConfig,
    BatchConfig
)

logger = logging.getLogger(__name__)

class AdvancedCommPattern(Enum):
    NEURAL_MESH = "neural_mesh"
    BROADCAST = "broadcast"
    PIPELINE = "pipeline"
    QUANTUM_ENTANGLED = "quantum_entangled"

class StandardNetworkChannel:
    """Standard network channel without optimization (fallback)"""
    
    def __init__(self, channel_id: str):
        self.channel_id = channel_id
        self._session = None
        
    async def start(self):
        """Start standard network channel"""
        import aiohttp
        self._session = aiohttp.ClientSession()
        logger.info(f"Standard network channel {self.channel_id} started")
        
    async def stop(self):
        """Stop standard network channel"""
        if self._session:
            await self._session.close()
        logger.info(f"Standard network channel {self.channel_id} stopped")
        
    async def send(self, message: Dict[str, Any]) -> bool:
        """Send message without optimization"""
        if not self._session:
            logger.error("Session not initialized")
            return False
            
        try:
            # Simple HTTP POST without optimization
            async with self._session.post(
                f"http://localhost:8080/messages",
                json=message
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get basic stats"""
        return {
            "channel_id": self.channel_id,
            "optimization_enabled": False
        }

class AdvancedAICommChannel(BaseCommChannel):
    """AI Communication Channel with integrated network optimization"""
    
    def __init__(self, 
                 channel_id: str,
                 pattern: AdvancedCommPattern,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(channel_id)
        self.pattern = pattern
        self.config = config or {}
        
        # Network optimization configuration
        self.network_config = self.config.get('network', {})
        self.optimization_enabled = self.network_config.get('optimization_enabled', True)
        
        # Initialize optimized network channel if enabled
        if self.optimization_enabled:
            self._init_optimized_network()
        else:
            self._init_standard_network()
            
        # Metrics collector - initialize after network channel is created
        self.metrics_collector = None
        
        # Pattern-specific initialization
        self._init_pattern_handler()
        
        # Message receive queue
        self._receive_queue = asyncio.Queue(maxsize=1000)
        
    def _init_optimized_network(self):
        """Initialize network layer with optimization"""
        # Extract endpoint from config
        endpoint = self.config.get('endpoint', 'http://localhost:8080')
        
        # Create optimized channel
        self.network_channel = OptimizedNetworkChannel(
            channel_id=f"{self.channel_id}_network",
            pattern=self.pattern,
            endpoint=endpoint,
            config=self.network_config
        )
        
        # Initialize metrics collector with the network channel
        self.metrics_collector = NetworkMetricsCollector(self.network_channel)
        
        logger.info(f"Initialized optimized network for channel {self.channel_id}")
        
    def _init_standard_network(self):
        """Initialize standard network without optimization"""
        # Fallback to standard implementation
        self.network_channel = StandardNetworkChannel(f"{self.channel_id}_network")
        logger.info(f"Initialized standard network for channel {self.channel_id}")
        
    def _init_pattern_handler(self):
        """Initialize pattern-specific message routing"""
        if self.pattern == AdvancedCommPattern.NEURAL_MESH:
            from .patterns.neural_mesh import NeuralMeshHandler
            self.pattern_handler = NeuralMeshHandler(self)
        elif self.pattern == AdvancedCommPattern.BROADCAST:
            from .patterns.broadcast import BroadcastHandler
            self.pattern_handler = BroadcastHandler(self)
        elif self.pattern == AdvancedCommPattern.PIPELINE:
            from .patterns.pipeline import PipelineHandler
            self.pattern_handler = PipelineHandler(self)
        else:
            self.pattern_handler = None
            
    async def start(self):
        """Start the communication channel"""
        await super().start()
        
        await self.network_channel.start()
            
        if self.pattern_handler:
            await self.pattern_handler.start()
            
        logger.info(f"Channel {self.channel_id} started with pattern {self.pattern}")
        
    async def stop(self):
        """Stop the communication channel"""
        if self.pattern_handler:
            await self.pattern_handler.stop()
            
        await self.network_channel.stop()
            
        await super().stop()
        logger.info(f"Channel {self.channel_id} stopped")
        
    async def send_message(self, 
                          recipient: str,
                          message: Dict[str, Any],
                          priority: int = 0) -> bool:
        """Send message through optimized network"""
        # Add routing metadata
        wrapped_message = {
            "sender": self.channel_id,
            "recipient": recipient,
            "pattern": self.pattern.value,
            "timestamp": asyncio.get_event_loop().time(),
            "payload": message
        }
        
        # Route through pattern handler
        if self.pattern_handler:
            wrapped_message = await self.pattern_handler.process_outgoing(wrapped_message)
        
        # Send through network
        if self.optimization_enabled:
            return await self.network_channel.send_optimized(wrapped_message, priority)
        else:
            return await self.network_channel.send(wrapped_message)
            
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from network"""
        try:
            message = await asyncio.wait_for(self._receive_queue.get(), timeout=0.1)
            return message
        except asyncio.TimeoutError:
            return None
            
    async def _handle_incoming_message(self, message: Dict[str, Any]):
        """Handle incoming message from network layer"""
        # Route through pattern handler
        if self.pattern_handler:
            message = await self.pattern_handler.process_incoming(message)
            
        # Put in receive queue
        try:
            self._receive_queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning(f"Receive queue full for channel {self.channel_id}")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get channel metrics including network optimization stats"""
        metrics = {
            "channel_id": self.channel_id,
            "pattern": self.pattern.value,
            "optimization_enabled": self.optimization_enabled
        }
        
        if self.optimization_enabled:
            metrics["network_stats"] = self.network_channel.get_stats()
            
        if self.pattern_handler and hasattr(self.pattern_handler, 'get_stats'):
            metrics["pattern_stats"] = self.pattern_handler.get_stats()
            
        return metrics