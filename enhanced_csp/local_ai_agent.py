#!/usr/bin/env python3
"""
Local AI Agent for Enhanced CSP Network
Supports multiple model backends: Ollama, LlamaCpp, GPT4All, and HTTP APIs
"""

import asyncio
import logging
import json
import time
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# AI AGENT CONFIGURATION
# ============================================================================

@dataclass
class LocalAIConfig:
    """Configuration for local AI agent"""
    # Agent identity
    agent_name: str = "LocalAI-Agent"
    agent_id: str = ""
    capabilities: List[str] = field(default_factory=lambda: [
        "natural_language", "reasoning", "code_generation", 
        "pattern_recognition", "decision_making"
    ])
    
    # Network configuration
    host: str = "127.0.0.1"
    port: int = 8765
    broadcast_port: int = 8766
    
    # Model configuration
    model_backend: str = "ollama"  # Options: ollama, llamacpp, gpt4all, http_api
    model_name: str = "deepseek-r1"  # For Ollama: deepseek-r1, llama3, mistral, phi, etc.
    model_path: str = ""  # For llamacpp/gpt4all
    api_url: str = "http://localhost:11434"  # Ollama default
    
    # Generation parameters
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    
    # CSP configuration
    csp_namespace: str = "local_ai"
    enable_quantum_states: bool = True
    enable_protocol_synthesis: bool = True
    
    # Performance
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0
    cache_size: int = 1000

# ============================================================================
# MODEL BACKEND INTERFACES
# ============================================================================

class ModelBackend(ABC):
    """Abstract base class for model backends"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        pass

class OllamaBackend(ModelBackend):
    """Ollama backend for local models"""
    
    def __init__(self, config: LocalAIConfig):
        self.config = config
        self.api_url = config.api_url
        self.model = config.model_name
        self.session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama"""
        await self._ensure_session()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens)
            }
        }
        
        try:
            async with self.session.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "")
                else:
                    error = await response.text()
                    logger.error(f"Ollama error: {error}")
                    return f"Error: {error}"
                    
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"Error: {str(e)}"
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using Ollama"""
        await self._ensure_session()
        
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            async with self.session.post(
                f"{self.api_url}/api/embeddings",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("embedding", [])
                else:
                    # Return random embedding as fallback
                    return list(np.random.randn(4096))
                    
        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            return list(np.random.randn(4096))
    
    async def health_check(self) -> bool:
        """Check if Ollama is running"""
        await self._ensure_session()
        
        try:
            async with self.session.get(f"{self.api_url}/api/tags") as response:
                return response.status == 200
        except:
            return False

class LlamaCppBackend(ModelBackend):
    """Llama.cpp backend using llama-cpp-python"""
    
    def __init__(self, config: LocalAIConfig):
        self.config = config
        self.llm = None
        
    async def _ensure_model(self):
        """Ensure model is loaded"""
        if not self.llm:
            try:
                from llama_cpp import Llama
                self.llm = Llama(
                    model_path=self.config.model_path,
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=35  # Adjust based on your GPU
                )
            except ImportError:
                raise RuntimeError("llama-cpp-python not installed. Run: pip install llama-cpp-python")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using llama.cpp"""
        await self._ensure_model()
        
        result = self.llm(
            prompt,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            echo=False
        )
        
        return result["choices"][0]["text"]
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings"""
        await self._ensure_model()
        
        # Get embeddings from llama.cpp
        embeddings = self.llm.embed(text)
        return embeddings
    
    async def health_check(self) -> bool:
        """Check if model is loaded"""
        try:
            await self._ensure_model()
            return self.llm is not None
        except:
            return False

class GPT4AllBackend(ModelBackend):
    """GPT4All backend for local models"""
    
    def __init__(self, config: LocalAIConfig):
        self.config = config
        self.model = None
    
    async def _ensure_model(self):
        """Ensure model is loaded"""
        if not self.model:
            try:
                from gpt4all import GPT4All
                self.model = GPT4All(
                    model_name=self.config.model_name,
                    model_path=self.config.model_path or ".",
                    allow_download=True
                )
            except ImportError:
                raise RuntimeError("gpt4all not installed. Run: pip install gpt4all")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using GPT4All"""
        await self._ensure_model()
        
        response = self.model.generate(
            prompt,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temp=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k)
        )
        
        return response
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings"""
        # GPT4All doesn't have built-in embeddings, use random for now
        # You could use sentence-transformers here instead
        return list(np.random.randn(384))
    
    async def health_check(self) -> bool:
        """Check if model is loaded"""
        try:
            await self._ensure_model()
            return self.model is not None
        except:
            return False

class HTTPAPIBackend(ModelBackend):
    """Generic HTTP API backend (OpenAI-compatible APIs)"""
    
    def __init__(self, config: LocalAIConfig):
        self.config = config
        self.session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using HTTP API"""
        await self._ensure_session()
        
        # OpenAI-compatible format
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p)
        }
        
        headers = {
            "Content-Type": "application/json",
            # Add API key if needed: "Authorization": f"Bearer {api_key}"
        }
        
        try:
            async with self.session.post(
                f"{self.config.api_url}/v1/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error = await response.text()
                    return f"Error: {error}"
                    
        except Exception as e:
            logger.error(f"HTTP API error: {e}")
            return f"Error: {str(e)}"
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using HTTP API"""
        # Implement if your API supports embeddings
        return list(np.random.randn(1536))
    
    async def health_check(self) -> bool:
        """Check if API is accessible"""
        await self._ensure_session()
        
        try:
            async with self.session.get(self.config.api_url) as response:
                return response.status < 500
        except:
            return False

# ============================================================================
# LOCAL AI MODEL INTERFACE
# ============================================================================

class LocalAIModel:
    """Unified interface to local AI models"""
    
    def __init__(self, config: LocalAIConfig):
        self.config = config
        self.backend = self._create_backend()
        self.is_loaded = False
        self.response_cache = {}
        
    def _create_backend(self) -> ModelBackend:
        """Create appropriate backend based on config"""
        backend_map = {
            "ollama": OllamaBackend,
            "llamacpp": LlamaCppBackend,
            "gpt4all": GPT4AllBackend,
            "http_api": HTTPAPIBackend
        }
        
        backend_class = backend_map.get(self.config.model_backend)
        if not backend_class:
            raise ValueError(f"Unknown backend: {self.config.model_backend}")
        
        return backend_class(self.config)
    
    async def load_model(self):
        """Initialize the model backend"""
        try:
            logger.info(f"Initializing {self.config.model_backend} backend...")
            
            # Check if backend is available
            if await self.backend.health_check():
                self.is_loaded = True
                logger.info(f"{self.config.model_backend} backend ready")
            else:
                raise RuntimeError(f"{self.config.model_backend} backend not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    async def generate(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate response from the model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Check cache
        cache_key = f"{prompt}:{json.dumps(context or {})}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Add context to prompt if provided
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            full_prompt = f"Context:\n{context_str}\n\nQuery: {prompt}"
        else:
            full_prompt = prompt
        
        # Generate response
        response = await self.backend.generate(full_prompt)
        
        # Cache response
        if len(self.response_cache) < self.config.cache_size:
            self.response_cache[cache_key] = response
        
        return response
    
    async def embed(self, text: str) -> np.ndarray:
        """Generate embeddings for semantic matching"""
        embeddings = await self.backend.embed(text)
        return np.array(embeddings)

# ============================================================================
# CSP PROCESS IMPLEMENTATION (Same as before)
# ============================================================================

class AIProcess:
    """CSP Process representing an AI agent"""
    
    def __init__(self, agent_id: str, model: LocalAIModel):
        self.id = agent_id
        self.model = model
        self.state = "READY"
        self.channels = {}
        self.event_queue = asyncio.Queue()
        self.context = {}
        
    async def run(self):
        """Main process execution loop"""
        while self.state != "TERMINATED":
            try:
                # Wait for events
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                
                # Process event
                await self.handle_event(event)
                
            except asyncio.TimeoutError:
                # No events, continue
                pass
            except Exception as e:
                logger.error(f"Process error: {e}")
    
    async def handle_event(self, event: Dict):
        """Handle incoming events"""
        event_type = event.get("type")
        
        if event_type == "QUERY":
            await self.handle_query(event)
        elif event_type == "COLLABORATE":
            await self.handle_collaboration(event)
        elif event_type == "SYNC":
            await self.handle_sync(event)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    async def handle_query(self, event: Dict):
        """Handle AI query requests"""
        query = event.get("query", "")
        sender = event.get("sender")
        
        # Generate response
        response = await self.model.generate(query, self.context)
        
        # Send response back
        await self.send_event({
            "type": "RESPONSE",
            "query": query,
            "response": response,
            "recipient": sender,
            "timestamp": time.time()
        })
    
    async def send_event(self, event: Dict):
        """Send event through appropriate channel"""
        channel = event.get("channel", "default")
        if channel in self.channels:
            await self.channels[channel].send(event)

# ============================================================================
# NETWORK AND MAIN AGENT CLASSES (Same structure as before)
# ============================================================================

class LocalAINetworkNode:
    """Network node for AI agent communication"""
    
    def __init__(self, config: LocalAIConfig):
        self.config = config
        self.node_id = f"{config.agent_name}-{int(time.time())}"
        self.peers = {}
        self.channels = {}
        self.is_running = False
        
        # Transport layer
        self.server = None
        self.connections = {}
        
        # Routing table
        self.routing_table = {}
        
        # Protocol handlers
        self.handlers = {
            "HELLO": self.handle_hello,
            "QUERY": self.handle_query,
            "RESPONSE": self.handle_response,
            "COLLABORATE": self.handle_collaborate,
            "SYNC": self.handle_sync,
            "PROTOCOL_NEGOTIATION": self.handle_protocol_negotiation
        }
    
    async def start(self):
        """Start network node"""
        try:
            # Start TCP server
            self.server = await asyncio.start_server(
                self.handle_connection,
                self.config.host,
                self.config.port
            )
            
            self.is_running = True
            logger.info(f"Network node started on {self.config.host}:{self.config.port}")
            
            # Start discovery
            asyncio.create_task(self.discovery_loop())
            
            # Start heartbeat
            asyncio.create_task(self.heartbeat_loop())
            
        except Exception as e:
            logger.error(f"Failed to start network node: {e}")
            raise
    
    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming connections"""
        peer_addr = writer.get_extra_info('peername')
        logger.info(f"New connection from {peer_addr}")
        
        try:
            while True:
                # Read message length (4 bytes)
                length_bytes = await reader.read(4)
                if not length_bytes:
                    break
                
                length = int.from_bytes(length_bytes, 'big')
                
                # Read message
                data = await reader.read(length)
                if not data:
                    break
                
                # Parse and handle message
                message = json.loads(data.decode())
                await self.handle_message(message, writer)
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def handle_message(self, message: Dict, writer: asyncio.StreamWriter):
        """Route messages to appropriate handlers"""
        msg_type = message.get("type")
        
        if msg_type in self.handlers:
            response = await self.handlers[msg_type](message)
            if response:
                await self.send_message(response, writer)
        else:
            logger.warning(f"Unknown message type: {msg_type}")
    
    async def send_message(self, message: Dict, writer: asyncio.StreamWriter):
        """Send message to peer"""
        try:
            data = json.dumps(message).encode()
            length = len(data).to_bytes(4, 'big')
            
            writer.write(length + data)
            await writer.drain()
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def handle_hello(self, message: Dict) -> Optional[Dict]:
        """Handle HELLO messages for peer discovery"""
        peer_id = message.get("node_id")
        capabilities = message.get("capabilities", [])
        
        # Register peer
        self.peers[peer_id] = {
            "id": peer_id,
            "capabilities": capabilities,
            "last_seen": time.time()
        }
        
        # Send our info
        return {
            "type": "HELLO_ACK",
            "node_id": self.node_id,
            "capabilities": self.config.capabilities
        }
    
    async def handle_query(self, message: Dict) -> Optional[Dict]:
        """Handle AI query requests"""
        # Forward to AI process
        event = {
            "type": "QUERY",
            "query": message.get("query"),
            "sender": message.get("sender"),
            "context": message.get("context", {})
        }
        
        # Queue for processing
        if hasattr(self, 'ai_process'):
            await self.ai_process.event_queue.put(event)
        
        return None  # Response will be sent asynchronously
    
    async def handle_response(self, message: Dict) -> Optional[Dict]:
        """Handle query responses"""
        # Process response
        logger.info(f"Received response: {message.get('response', '')[:100]}...")
        return None
    
    async def handle_collaborate(self, message: Dict) -> Optional[Dict]:
        """Handle collaboration requests"""
        task = message.get("task")
        participants = message.get("participants", [])
        
        logger.info(f"Collaboration request: {task} with {participants}")
        
        # Implement collaboration logic
        return {
            "type": "COLLABORATE_ACK",
            "task": task,
            "status": "accepted",
            "node_id": self.node_id
        }
    
    async def handle_sync(self, message: Dict) -> Optional[Dict]:
        """Handle state synchronization"""
        sync_type = message.get("sync_type")
        
        if sync_type == "context":
            # Sync context with peers
            return {
                "type": "SYNC_ACK",
                "sync_type": "context",
                "context": self.get_current_context()
            }
        
        return None
    
    async def handle_protocol_negotiation(self, message: Dict) -> Optional[Dict]:
        """Handle dynamic protocol negotiation"""
        proposed_protocol = message.get("protocol")
        
        # Evaluate protocol
        if self.config.enable_protocol_synthesis:
            # Accept or propose alternative
            return {
                "type": "PROTOCOL_ACK",
                "protocol": proposed_protocol,
                "status": "accepted"
            }
        
        return None
    
    def get_current_context(self) -> Dict:
        """Get current AI context for synchronization"""
        return {
            "timestamp": time.time(),
            "active_tasks": [],
            "knowledge_hash": "",
            "capabilities": self.config.capabilities
        }
    
    async def discovery_loop(self):
        """Periodic peer discovery"""
        while self.is_running:
            try:
                # Broadcast HELLO
                await self.broadcast_hello()
                
                # Clean stale peers
                self.clean_stale_peers()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Discovery error: {e}")
    
    async def broadcast_hello(self):
        """Broadcast HELLO message for discovery"""
        message = {
            "type": "HELLO",
            "node_id": self.node_id,
            "capabilities": self.config.capabilities,
            "timestamp": time.time()
        }
        
        # In a real implementation, use UDP broadcast or multicast
        logger.debug(f"Broadcasting HELLO: {message}")
    
    def clean_stale_peers(self):
        """Remove peers that haven't been seen recently"""
        current_time = time.time()
        stale_threshold = 120  # 2 minutes
        
        stale_peers = [
            peer_id for peer_id, info in self.peers.items()
            if current_time - info.get("last_seen", 0) > stale_threshold
        ]
        
        for peer_id in stale_peers:
            del self.peers[peer_id]
            logger.info(f"Removed stale peer: {peer_id}")
    
    async def heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.is_running:
            try:
                # Send heartbeat to all peers
                heartbeat = {
                    "type": "HEARTBEAT",
                    "node_id": self.node_id,
                    "timestamp": time.time()
                }
                
                # Broadcast to peers
                await self.broadcast_to_peers(heartbeat)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def broadcast_to_peers(self, message: Dict):
        """Broadcast message to all known peers"""
        # In a real implementation, send to all peer connections
        logger.debug(f"Broadcasting to {len(self.peers)} peers: {message['type']}")

# ============================================================================
# MAIN AI AGENT
# ============================================================================

class LocalAIAgent:
    """Main local AI agent with CSP network integration"""
    
    def __init__(self, config: Optional[LocalAIConfig] = None):
        self.config = config or LocalAIConfig()
        self.config.agent_id = f"{self.config.agent_name}-{int(time.time())}"
        
        # Components
        self.model = LocalAIModel(self.config)
        self.network = LocalAINetworkNode(self.config)
        self.ai_process = AIProcess(self.config.agent_id, self.model)
        
        # State
        self.is_running = False
        
        # Metrics
        self.metrics = {
            "queries_processed": 0,
            "collaborations": 0,
            "network_messages": 0,
            "start_time": time.time()
        }
    
    async def start(self):
        """Start the AI agent"""
        try:
            logger.info(f"Starting Local AI Agent: {self.config.agent_id}")
            
            # Load AI model
            await self.model.load_model()
            
            # Start network node
            await self.network.start()
            
            # Link network to AI process
            self.network.ai_process = self.ai_process
            
            # Start AI process
            asyncio.create_task(self.ai_process.run())
            
            self.is_running = True
            logger.info("Local AI Agent started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start AI agent: {e}")
            raise
    
    async def query(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Query the AI agent"""
        if not self.is_running:
            raise RuntimeError("Agent not running")
        
        # Generate response
        response = await self.model.generate(prompt, context)
        
        # Update metrics
        self.metrics["queries_processed"] += 1
        
        return response
    
    async def collaborate(self, task: str, peers: List[str]) -> Dict:
        """Initiate collaboration with other agents"""
        collaboration = {
            "type": "COLLABORATE",
            "task": task,
            "participants": peers,
            "initiator": self.config.agent_id,
            "timestamp": time.time()
        }
        
        # Broadcast collaboration request
        await self.network.broadcast_to_peers(collaboration)
        
        # Update metrics
        self.metrics["collaborations"] += 1
        
        return {
            "status": "initiated",
            "task": task,
            "participants": peers
        }
    
    async def sync_with_peers(self):
        """Synchronize state with peer agents"""
        sync_message = {
            "type": "SYNC",
            "sync_type": "context",
            "node_id": self.config.agent_id,
            "timestamp": time.time()
        }
        
        await self.network.broadcast_to_peers(sync_message)
    
    def get_metrics(self) -> Dict:
        """Get agent metrics"""
        uptime = time.time() - self.metrics["start_time"]
        
        return {
            **self.metrics,
            "uptime_seconds": uptime,
            "peers_connected": len(self.network.peers),
            "model_backend": self.config.model_backend,
            "model_name": self.config.model_name
        }
    
    async def stop(self):
        """Stop the AI agent"""
        logger.info("Stopping Local AI Agent")
        
        self.is_running = False
        
        # Stop components
        self.ai_process.state = "TERMINATED"
        self.network.is_running = False
        
        if self.network.server:
            self.network.server.close()
            await self.network.server.wait_closed()
        
        logger.info("Local AI Agent stopped")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def main():
    """Example usage of the Local AI Agent"""
    
    # Example 1: Using Ollama (recommended for ease of use)
    ollama_config = LocalAIConfig(
        agent_name="OllamaAgent",
        host="0.0.0.0",
        port=8765,
        model_backend="ollama",
        model_name="deepseek-r1",  # or "mistral", "llama3", "phi", etc.
        api_url="http://localhost:11434"
    )
    
    # Example 2: Using llama.cpp
    # llamacpp_config = LocalAIConfig(
    #     agent_name="LlamaCppAgent",
    #     model_backend="llamacpp",
    #     model_path="/path/to/your/model.gguf"
    # )
    
    # Example 3: Using GPT4All
    # gpt4all_config = LocalAIConfig(
    #     agent_name="GPT4AllAgent",
    #     model_backend="gpt4all",
    #     model_name="orca-mini-3b.ggmlv3.q4_0.bin"
    # )
    
    # Create and start agent
    agent = LocalAIAgent(ollama_config)
    
    try:
        await agent.start()
        
        # Example queries
        response = await agent.query("What is the meaning of life?")
        print(f"AI Response: {response}")
        
        # Example collaboration
        collab_result = await agent.collaborate(
            "Solve complex problem",
            ["agent-2", "agent-3"]
        )
        print(f"Collaboration: {collab_result}")
        
        # Get metrics
        metrics = agent.get_metrics()
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
        
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                response = await agent.query(user_input)
                print(f"\nAI: {response}")
                
            except KeyboardInterrupt:
                break
            
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await agent.stop()

if __name__ == "__main__":
    # Check for required dependencies
    try:
        import aiohttp
    except ImportError:
        print("Please install aiohttp: pip install aiohttp")
        exit(1)
    
    print("Starting Local AI Agent with Enhanced CSP Network...")
    print("Make sure your chosen backend is running:")
    print("- Ollama: ollama serve")
    print("- LlamaCpp: Install llama-cpp-python")
    print("- GPT4All: Install gpt4all")
    
    asyncio.run(main())
