"""
Distributed AI Layer for CSP Agent Network
==========================================

A comprehensive distributed AI infrastructure with enterprise-grade features.
"""

from .distributed_ai_core import ShardAgent, AIRequest, AIResponse
from .router_local_agents import RouterAgent, LocalAgent
from .csp_integration_config import DistributedAIConfig, CSPIntegrationLayer

__version__ = "1.0.0"
__all__ = [
    "ShardAgent",
    "RouterAgent", 
    "LocalAgent",
    "AIRequest",
    "AIResponse",
    "DistributedAIConfig",
    "CSPIntegrationLayer"
]