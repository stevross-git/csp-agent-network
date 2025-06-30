#!/usr/bin/env python3
"""
Enhanced CSP • Core • network_node.py
=====================================

High-level orchestration wrapper around EnhancedCSPNetwork that:

  • Owns the underlying transport / topology / DNS overlay lifecycle
  • Integrates the SecurityOrchestrator (TLS rotation, threat detection …)
  • Optionally wires in QuantumCSPEngine and BlockchainCSPNetwork
  • Provides convenience methods: connect(), broadcast(), metrics(), …

This class is what the rest of the stack should import when it just needs
“a running node” without caring about all the subsystems.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from enhanced_csp.network.core.config import NetworkConfig, SecurityConfig
from enhanced_csp.network.core.types import NodeID, PeerInfo

# Optional subsystems — fall back to stubs if the user hasn’t installed them
try:
    from enhanced_csp.security_hardening import SecurityOrchestrator
except ImportError:                                    # pragma: no cover
    class SecurityOrchestrator:                        # type: ignore
        def __init__(self, *_, **__): pass
        async def initialize(self): pass
        async def shutdown(self): pass
        async def monitor_threats(self): pass
        async def rotate_tls_certificates(self): pass

try:
    from enhanced_csp.quantum_csp_engine import QuantumCSPEngine
except ImportError:                                    # pragma: no cover
    class QuantumCSPEngine:                            # type: ignore
        def __init__(self, *_): pass
        async def initialize(self): pass
        async def shutdown(self): pass

try:
    from enhanced_csp.blockchain_csp_network import BlockchainCSPNetwork
except ImportError:                                    # pragma: no cover
    class BlockchainCSPNetwork:                        # type: ignore
        def __init__(self, *_): pass
        async def initialize(self): pass
        async def shutdown(self): pass


@dataclass
class NodeStats:
    start_time:      datetime
    peers:           int = 0
    messages_sent:   int = 0
    messages_recv:   int = 0
    bandwidth_in:    int = 0          # bytes
    bandwidth_out:   int = 0
    bootstrap_reqs:  int = 0


class NetworkNode:
    """
    High-level façade for an Enhanced CSP node.
    -------------------------------------------------------
    Usage
    -----
        cfg   = NetworkConfig(...)
        node  = NetworkNode(cfg)
        await node.start()
        await node.connect('/ip4/1.2.3.4/tcp/30300/p2p/xyz…')
        await node.broadcast({'cmd': 'ping'})
        print(await node.metrics())
        await node.stop()
    """

    def __init__(
        self,
        config:           NetworkConfig,
        *,
        enable_quantum:   bool = False,
        enable_blockchain: bool = False,
        logger:           Optional[logging.Logger] = None,
    ) -> None:
        self.cfg                      = config
        self.logger                   = logger or logging.getLogger(f"enhanced_csp.NetworkNode.{id(self):x}")
        self.net                      = EnhancedCSPNetwork(config)

        # Subsystems
        self.sec                      = SecurityOrchestrator(config.security)
        self.qengine:  Optional[QuantumCSPEngine]     = QuantumCSPEngine(self.net) if enable_quantum   else None
        self.bchain:  Optional[BlockchainCSPNetwork]  = BlockchainCSPNetwork(self.net) if enable_blockchain else None

        # Internal
        self._tasks:  List[asyncio.Task]  = []
        self._tls_rotation_next           = datetime.utcnow() + timedelta(seconds=config.security.tls_rotation_interval)
        self.stats                        = NodeStats(start_time=datetime.utcnow())

    # --------------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        self.logger.info("▶ Starting NetworkNode %s", self.net.node_id)

        await self.sec.initialize()
        if self.qengine:  await self.qengine.initialize()
        if self.bchain:   await self.bchain.initialize()

        await self.net.start()
        self._tasks.append(asyncio.create_task(self._background_security()))
        self._tasks.append(asyncio.create_task(self._background_metrics()))

    async def stop(self) -> None:
        self.logger.info("■ Stopping NetworkNode %s", self.net.node_id)

        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        if self.bchain:   await self.bchain.shutdown()
        if self.qengine:  await self.qengine.shutdown()
        await self.sec.shutdown()
        await self.net.stop()

    # --------------------------------------------------------------------- high-level API
    async def connect(self, multiaddr: str) -> None:
        """Connect to another peer (multiaddr or DNS name)."""
        await self.net.transport.connect(multiaddr)
        self.stats.bootstrap_reqs += 1

    async def send(self, peer_id: str, payload: Any) -> None:
        """Send a unicast payload."""
        await self.net.send_message(peer_id, payload)
        self.stats.messages_sent += 1

    async def broadcast(self, payload: Any) -> None:
        """Send to every connected peer."""
        peers = self.net.get_peers()
        await asyncio.gather(*(self.net.send_message(p.id, payload) for p in peers))
        self.stats.messages_sent += len(peers)

    def peers(self) -> List[PeerInfo]:
        return self.net.get_peers()

    async def metrics(self) -> Dict[str, Any]:
        """Return a merged dict of node + subsystem metrics."""
        core = await self.net.topology.get_metrics() if hasattr(self.net.topology, "get_metrics") else {}
        return {
            **vars(self.stats),
            "uptime_s": (datetime.utcnow() - self.stats.start_time).total_seconds(),
            "core": core,
        }

    # --------------------------------------------------------------------- background tasks
    async def _background_security(self) -> None:
        """Security orchestrator helper loop (TLS rotation, threat-monitor etc.)."""
        try:
            while True:
                # Threat monitor is its own task inside sec.monitor_threats()
                await asyncio.sleep(5)

                # TLS rotation
                if datetime.utcnow() >= self._tls_rotation_next:
                    self.logger.info("🔒 Rotating TLS certificates")
                    await self.sec.rotate_tls_certificates()
                    self._tls_rotation_next = datetime.utcnow() + timedelta(
                        seconds=self.cfg.security.tls_rotation_interval
                    )
        except asyncio.CancelledError:
            pass

    async def _background_metrics(self) -> None:
        """Lightweight metrics updater (peer count, bandwidth …)."""
        try:
            while True:
                self.stats.peers = len(self.net.get_peers())
                # Pull bandwidth counters from transport if available
                if hasattr(self.net.transport, "stats"):
                    tstats = self.net.transport.stats
                    self.stats.bandwidth_in  = tstats.get("bytes_in",  0)
                    self.stats.bandwidth_out = tstats.get("bytes_out", 0)
                await asyncio.sleep(15)
        except asyncio.CancelledError:
            pass


# --------------------------------------------------------------------------- convenience factory
async def create_network_node(
    cfg: NetworkConfig | None = None,
    *,
    enable_quantum: bool = False,
    enable_blockchain: bool = False,
    tls_rotation_days: int = 30,
) -> NetworkNode:
    """
    Helper that fills reasonable defaults and returns a started node.

        node = await create_network_node()
    """
    if cfg is None:
        cfg = NetworkConfig(
            security=SecurityConfig(
                tls_rotation_interval=tls_rotation_days * 86_400,
            )
        )
    node = NetworkNode(cfg, enable_quantum=enable_quantum, enable_blockchain=enable_blockchain)
    await node.start()
    return node
