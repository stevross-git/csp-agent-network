import asyncio
import pytest

from enhanced_csp.network.core.types import NetworkConfig
from enhanced_csp.network.core.node import NetworkNode
from enhanced_csp.network.p2p.discovery import HybridDiscovery


@pytest.mark.asyncio
async def test_bootstrap_seed(monkeypatch):
    config = NetworkConfig()
    config.p2p.bootstrap_api_url = "x"
    config.p2p.dns_seed_domain = "y"
    node = NetworkNode(config)
    discovery = HybridDiscovery(node, config.p2p)

    async def fake_fetch_api(self):
        return ["/dnsaddr/boot1.peoplesainetwork.com"]

    async def fake_fetch_dns(self):
        return ["/dnsaddr/boot2.peoplesainetwork.com"]

    async def fake_connect(self, addr: str) -> bool:
        self.discovered_peers.add(addr)
        return True

    monkeypatch.setattr(HybridDiscovery, "_fetch_bootstrap_api", fake_fetch_api)
    monkeypatch.setattr(HybridDiscovery, "_fetch_dns_seed", fake_fetch_dns)
    monkeypatch.setattr(HybridDiscovery, "_connect_bootstrap", fake_connect)

    await discovery.start()

    assert "/dnsaddr/boot1.peoplesainetwork.com" in discovery.discovered_peers
    assert "/dnsaddr/boot2.peoplesainetwork.com" in discovery.discovered_peers
