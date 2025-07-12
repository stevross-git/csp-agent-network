"""Core network node management."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import NetworkConfig
from .types import MessageType, NetworkMessage, NodeCapabilities, NodeID, PeerInfo
from ..utils import (
    TaskManager,
    ResourceManager,
    retry_async,
    CircuitBreaker,
    RateLimiter,
    MessageBatcher,
    BatchConfig,
    validate_message_size,
    validate_node_id,
    sanitize_string_input,
)
from ..errors import (
    NetworkError,
    ConnectionError,
    ErrorMetrics,
    CircuitBreakerOpen,
)

# Heavy modules are imported lazily in _init_components to avoid optional
# dependencies at import time.

logger = logging.getLogger(__name__)


@dataclass
class NodeMetrics:
    """Thread-safe metrics container for a node."""

    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    peer_count: int = 0
    start_time: float = field(default_factory=time.time)
    errors: int = 0
    rate_violations: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def increment(self, field_name: str, amount: int = 1) -> None:
        with self.lock:
            setattr(self, field_name, getattr(self, field_name) + amount)

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "bytes_sent": self.bytes_sent,
                "bytes_received": self.bytes_received,
                "peer_count": self.peer_count,
                "errors": self.errors,
                "rate_violations": self.rate_violations,
                "uptime": time.time() - self.start_time,
            }


class NetworkNode:
    """Manage P2P networking for a single node."""

    def __init__(self, config: Optional[NetworkConfig] = None) -> None:
        self.config = config or NetworkConfig()
        self.node_id = NodeID.generate()
        self.capabilities: NodeCapabilities = self.config.capabilities

        # Components will be created in _init_components
        self.transport = None
        self.discovery = None
        self.dht = None
        self.nat = None
        self.topology = None
        self.routing = None
        self.dns = None
        self.adaptive_routing = None

        self.peers: Dict[NodeID, PeerInfo] = {}
        self._message_handlers: Dict[MessageType, List[Callable[[NetworkMessage], Any]]] = {}
        self._event_handlers: Dict[str, List[Callable[[Any], Any]]] = {}
        self.task_manager = TaskManager()
        self.resource_manager = ResourceManager()
        self.metrics = NodeMetrics()
        self.error_metrics = ErrorMetrics()
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(rate=100, burst=200, window=60)
        self.batchers: Dict[str, MessageBatcher] = {}
        self.is_running = False

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    async def start(self) -> bool:
        if self.is_running:
            logger.warning("node already running")
            return True

        try:
            await self._init_components()
            if self.transport and not await self.transport.start():
                raise RuntimeError("transport failed")
            if self.discovery:
                await self.discovery.start()
            if self.dht:
                await self.dht.start()
            if self.nat:
                await self.nat.start()
            if self.topology:
                await self.topology.start()
            if self.routing:
                await self.routing.start()
            if self.dns:
                await self.dns.start()
            if self.adaptive_routing:
                await self.adaptive_routing.start()

            if self.transport:
                self.transport.register_handler(
                    MessageType.BATCH.value,
                    lambda _addr, data: asyncio.create_task(self._handle_batch(data)),
                )

            self.task_manager.create_task(self._stats_loop())
            self.task_manager.create_task(self._health_loop())
            self.is_running = True
            logger.info("node %s started", self.node_id)
            return True
        except Exception as exc:  # pragma: no cover - runtime issues
            self.error_metrics.record(exc)
            logger.exception(
                "failed to start node", exc_info=exc, extra={"operation": "node.start", "node": str(self.node_id)}
            )
            await self.stop()
            raise NetworkError("failed to start node") from exc

    async def stop(self) -> bool:
        if not self.is_running:
            return True

        await self.task_manager.cancel_all()
        await self.resource_manager.close_all()

        try:
            if self.adaptive_routing:
                await self.adaptive_routing.stop()
            if self.dns:
                await self.dns.stop()
            if self.routing:
                await self.routing.stop()
            if self.topology:
                await self.topology.stop()
            if self.nat:
                await self.nat.stop()
            if self.dht:
                await self.dht.stop()
            if self.discovery:
                await self.discovery.stop()
            if self.transport:
                await self.transport.stop()
        finally:
            self.is_running = False
            logger.info("node %s stopped", self.node_id)
        return True

    async def _init_components(self) -> None:
        """Create subsystem instances lazily."""
        from ..p2p.transport import MultiProtocolTransport
        from ..p2p.discovery import HybridDiscovery
        from ..p2p.dht import KademliaDHT
        from ..p2p.nat import NATTraversal
        from ..mesh.topology import MeshTopologyManager
        from ..dns.overlay import DNSOverlay
        from ..routing.adaptive import AdaptiveRoutingEngine
        from ..mesh.routing import BatmanRouting

        self.transport = MultiProtocolTransport(self.config.p2p)
        self.discovery = HybridDiscovery(self.config.p2p, self.node_id)
        if self.config.enable_dht:
            self.dht = KademliaDHT(self.node_id, self.transport)
        self.nat = NATTraversal(self.config.p2p)
        if self.config.enable_mesh:
            async def send_fn(peer: NodeID, msg: Any) -> bool:
                if not self.transport:
                    return False
                return await self.transport.send(str(peer), msg)
            self.topology = MeshTopologyManager(self.node_id, self.config.mesh, send_fn)
        try:
            self.routing = BatmanRouting(self.node_id, self.topology)
        except Exception:  # pragma: no cover - optional dependency
            self.routing = None
        if self.config.enable_dns:
            self.dns = DNSOverlay(self)
        if self.config.enable_adaptive_routing and self.routing:
            self.adaptive_routing = AdaptiveRoutingEngine(self, self.config.routing, self.routing)

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------
    async def send_network_message(self, message: NetworkMessage, priority: int = 0) -> bool:
        if not self.transport or not self.is_running:
            raise ConnectionError("transport not running")
        if not self.rate_limiter.is_allowed(str(self.node_id)):
            self.metrics.increment("rate_violations")
            logger.debug("outgoing rate limit exceeded")
            return False
        try:
            validate_message_size(message.payload, self.config.p2p.max_message_size)
            payload = self._serialize_message(message)
            if not message.recipient:
                raise ValidationError("recipient required for send")
            peer = str(message.recipient)
            batcher = self.batchers.get(peer)
            if batcher is None:
                batcher = MessageBatcher(
                    lambda batch: self._send_batch(peer, batch),
                    task_manager=self.task_manager,
                )
                await batcher.start()
                self.batchers[peer] = batcher
            await batcher.add_message(payload, priority)
            return True
        except Exception as exc:  # pragma: no cover - runtime issues
            self.error_metrics.record(exc)
            peer = sanitize_string_input(str(message.recipient))
            logger.exception(
                "send failed", exc_info=exc, extra={"operation": "send", "peer": peer}
            )
            raise ConnectionError("send failed") from exc

    async def broadcast_message(self, message: NetworkMessage) -> int:
        if not self.transport or not self.is_running:
            return 0
        count = 0
        for peer_id in list(self.peers):
            msg = NetworkMessage(
                id=message.id,
                type=message.type,
                sender=message.sender,
                recipient=peer_id,
                payload=message.payload,
                timestamp=message.timestamp,
                ttl=message.ttl,
            )
            if await self.send_network_message(msg):
                count += 1
        return count

    async def _send_batch(self, peer: str, batch: dict) -> bool:
        send_fn = lambda: self.transport.send(peer, batch)
        success = await self.circuit_breaker.call(
            retry_async, send_fn, attempts=3
        )
        if success:
            self.metrics.increment("messages_sent", batch.get("count", 0))
            self.metrics.increment("bytes_sent", len(str(batch)))
        return success

    def _serialize_message(self, message: NetworkMessage) -> dict:
        return {
            "type": message.type.value,
            "payload": message.payload,
            "sender": str(message.sender),
            "recipient": str(message.recipient) if message.recipient else None,
            "timestamp": message.timestamp.isoformat(),
            "ttl": message.ttl,
            "id": message.id,
        }

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------
    def register_handler(self, msg_type: MessageType, handler: Callable[[NetworkMessage], Any]) -> None:
        self._message_handlers.setdefault(msg_type, []).append(handler)
        if self.transport:
            self.transport.register_handler(msg_type.value, lambda _addr, data: asyncio.create_task(handler(self._deserialize_message(data))))

    def unregister_handler(self, msg_type: MessageType, handler: Callable[[NetworkMessage], Any]) -> None:
        if msg_type in self._message_handlers:
            try:
                self._message_handlers[msg_type].remove(handler)
            except ValueError:
                pass

    async def _dispatch_message(self, message: NetworkMessage) -> None:
        if not self.rate_limiter.is_allowed(str(message.sender)):
            self.metrics.increment("rate_violations")
            logger.debug("message from %s dropped due to rate limit", message.sender)
            return
        handlers = self._message_handlers.get(message.type, [])
        for h in handlers:
            try:
                await h(message)
            except Exception as exc:  # pragma: no cover - handler errors
                self.error_metrics.record(exc)
                logger.exception(
                    "handler error", exc_info=exc, extra={"operation": "dispatch", "type": message.type.name}
                )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def on_event(self, name: str, handler: Callable[[Any], Any]) -> None:
        self._event_handlers.setdefault(name, []).append(handler)

    async def emit_event(self, name: str, payload: Any) -> None:
        for h in self._event_handlers.get(name, []):
            try:
                if asyncio.iscoroutinefunction(h):
                    await h(payload)
                else:
                    h(payload)
            except Exception as exc:  # pragma: no cover - handler errors
                self.error_metrics.record(exc)
                logger.exception(
                    "event handler failed",
                    exc_info=exc,
                    extra={"operation": "event", "event": name},
                )

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------
    async def _stats_loop(self) -> None:
        while self.is_running:
            with self.metrics.lock:
                self.metrics.peer_count = len(self.peers)
            await asyncio.sleep(60)

    async def _health_loop(self) -> None:
        while self.is_running:
            # basic health checks
            if self.transport and not getattr(self.transport, "is_running", False):
                logger.warning("transport stopped unexpectedly")
            await asyncio.sleep(120)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        with self.metrics.lock:
            self.metrics.errors = sum(self.error_metrics.counts.values())
            self.metrics.rate_violations = len(self.rate_limiter.blocked_peers)
        return self.metrics.snapshot()

    # ------------------------------------------------------------------
    # Helpers for message serialization (used by other modules)
    # ------------------------------------------------------------------
    def _deserialize_message(self, data: Dict[str, Any]) -> NetworkMessage:
        validate_message_size(data, self.config.p2p.max_message_size)
        msg_type = MessageType(data.get("type"))
        sender = NodeID.from_string(validate_node_id(str(data.get("sender"))))
        recipient = (
            NodeID.from_string(validate_node_id(str(data.get("recipient"))))
            if data.get("recipient")
            else None
        )
        ts_raw = data.get("timestamp")
        if isinstance(ts_raw, str):
            timestamp = datetime.fromisoformat(ts_raw)
        else:
            timestamp = datetime.utcnow()
        payload = data.get("payload")
        if isinstance(payload, str):
            payload = sanitize_string_input(payload)
        if msg_type == MessageType.BATCH:
            inner = []
            for item in data.get("messages", []):
                try:
                    inner.append(self._deserialize_message(item))
                except Exception:
                    continue
            payload = inner
        return NetworkMessage(
            id=str(data.get("id", "")),
            type=msg_type,
            sender=sender,
            recipient=recipient,
            payload=payload,
            timestamp=timestamp,
            ttl=int(data.get("ttl", 64)),
        )

    async def _handle_batch(self, data: Dict[str, Any]) -> None:
        msg = self._deserialize_message(data)
        if isinstance(msg.payload, list):
            for inner in msg.payload:
                await self._dispatch_message(inner)

class EnhancedCSPNetwork:
    """High level manager for multiple network nodes."""

    def __init__(self, config: Optional[NetworkConfig] = None) -> None:
        self.config = config or NetworkConfig()
        self.nodes: Dict[str, NetworkNode] = {}
        self.node_id = NodeID.generate()
        self.metrics: Dict[str, Any] = {
            "nodes_active": 0,
            "last_updated": time.time(),
        }
        self.is_running = False

    async def start(self) -> bool:
        try:
            await self.create_node("default")
            self.is_running = True
            self.metrics["nodes_active"] = len(self.nodes)
            self.metrics["last_updated"] = time.time()
            logger.info("network started with id %s", self.node_id)
            return True
        except Exception as exc:  # pragma: no cover - runtime issues
            logger.exception(
                "failed to start network", exc_info=exc, extra={"operation": "network.start"}
            )
            return False

    async def stop(self) -> bool:
        await self.stop_all()
        self.is_running = False
        self.metrics["nodes_active"] = 0
        self.metrics["last_updated"] = time.time()
        logger.info("network stopped")
        return True

    async def create_node(self, name: str) -> NetworkNode:
        node = NetworkNode(self.config)
        await node.start()
        self.nodes[name] = node
        self.metrics["nodes_active"] = len(self.nodes)
        self.metrics["last_updated"] = time.time()
        return node

    async def stop_node(self, name: str) -> bool:
        node = self.nodes.get(name)
        if not node:
            return False
        await node.stop()
        del self.nodes[name]
        self.metrics["nodes_active"] = len(self.nodes)
        self.metrics["last_updated"] = time.time()
        return True

    async def stop_all(self) -> None:
        for name in list(self.nodes):
            await self.stop_node(name)

    def get_node(self, name: str = "default") -> Optional[NetworkNode]:
        return self.nodes.get(name)

    async def get_metrics(self) -> Dict[str, Any]:
        snapshot = self.metrics.copy()
        node_metrics = {n: node.get_stats() for n, node in self.nodes.items()}
        snapshot["nodes"] = node_metrics
        return snapshot
