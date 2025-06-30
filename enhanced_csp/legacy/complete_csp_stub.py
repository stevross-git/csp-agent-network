"""Stubs for legacy CSP core classes."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class EnhancedProcessState(Enum):
    DORMANT = auto()
    READY = auto()
    BLOCKED = auto()
    COMMUNICATING = auto()
    CONSCIOUS = auto()
    UNCONSCIOUS = auto()
    DREAMING = auto()
    METACOGNITIVE = auto()
    SUPERPOSITION = auto()
    ENTANGLED = auto()
    COHERENT = auto()
    DECOHERENT = auto()
    MEASURED = auto()


class EnhancedChannelType(Enum):
    SYNCHRONOUS = auto()
    ASYNCHRONOUS = auto()
    STREAMING = auto()
    CONSCIOUSNESS_STREAM = auto()
    KNOWLEDGE_OSMOSIS = auto()
    WISDOM_CONVERGENCE = auto()
    INTENTION_PROPAGATION = auto()
    MEMORY_CRYSTALLIZATION = auto()
    QUANTUM_ENTANGLED = auto()
    QUANTUM_TELEPORTATION = auto()
    QUANTUM_CONSENSUS = auto()
    NEURAL_MESH = auto()
    TEMPORAL_ENTANGLEMENT = auto()
    CAUSAL_RESONANCE = auto()


class EnhancedCompositionOperator(Enum):
    SEQUENTIAL = auto()
    PARALLEL = auto()
    CHOICE = auto()
    INTERLEAVE = auto()
    SYNCHRONIZE = auto()
    HIDE = auto()
    RENAME = auto()
    CONSCIOUSNESS_MERGE = auto()
    KNOWLEDGE_TRANSFER = auto()
    WISDOM_SYNTHESIS = auto()
    METACOGNITIVE_OBSERVE = auto()
    QUANTUM_ENTANGLE = auto()
    QUANTUM_TELEPORT = auto()
    QUANTUM_SUPERPOSE = auto()
    QUANTUM_MEASURE = auto()


@dataclass
class EnhancedEvent:
    name: str
    channel: str
    data: Any = None
    timestamp: float = 0.0
    consciousness_level: float = 0.5
    attention_weight: float = 1.0
    emotional_valence: float = 0.0
    memory_strength: float = 1.0
    quantum_state: Optional["QuantumState"] = None
    entanglement_id: Optional[str] = None


@dataclass
class QuantumState:
    amplitudes: List[complex] = field(default_factory=lambda: [1 + 0j, 0 + 0j])
    phase: float = 0.0


@dataclass
class ConsciousnessState:
    awareness_level: float = 0.5
    attention_focus: List[str] = field(default_factory=list)


class EnhancedCSPEngine:
    """Placeholder for the monolithic CSP engine."""

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError
