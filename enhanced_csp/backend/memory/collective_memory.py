"""
Collective Memory Layer Implementation
======================================

This module implements the Collective Memory layer for emergent intelligence
and pattern recognition across the entire agent network.
"""

import asyncio
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA  # noqa: F401  (import kept for future use)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Enumerations and dataclasses
# --------------------------------------------------------------------------- #
class InsightType(Enum):
    """Types of collective insights produced by the engine."""
    EMERGENT_PATTERN = "emergent_pattern"
    COLLECTIVE_SOLUTION = "collective_solution"
    NETWORK_BEHAVIOR = "network_behavior"
    OPTIMIZATION = "optimization"
    ANOMALY = "anomaly"
    TREND = "trend"


@dataclass
class CollectiveInsight:
    """A single network-wide insight."""
    id: str
    type: InsightType
    description: str
    contributors: List[str]
    confidence: float                      # 0.0 – 1.0
    evidence: List[Dict[str, Any]]
    discovered_at: datetime = field(default_factory=datetime.now)

    vector_representation: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    impact_score: float = 0.0
    applications: List[str] = field(default_factory=list)

    # ------------- helpers -------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "contributors": self.contributors,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "discovered_at": self.discovered_at.isoformat(),
            "metadata": self.metadata,
            "impact_score": self.impact_score,
            "applications": self.applications,
        }


# --------------------------------------------------------------------------- #
#  Pattern detection
# --------------------------------------------------------------------------- #
class PatternDetector:
    """Stateless helpers that mine raw interaction streams for patterns."""

    def __init__(self):
        self.interaction_buffer: List[Dict[str, Any]] = []
        self.buffer_size = 1000

    # ............................................................... public
    async def analyze_interactions(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.interaction_buffer.extend(interactions)
        if len(self.interaction_buffer) > self.buffer_size:
            self.interaction_buffer = self.interaction_buffer[-self.buffer_size:]

        patterns: List[Dict[str, Any]] = []
        patterns.extend(await self._detect_sequence_patterns())
        patterns.extend(await self._detect_behavioral_patterns())
        patterns.extend(await self._detect_emergent_patterns())
        return patterns

    # ........................................................... detectors
    async def _detect_sequence_patterns(self) -> List[Dict[str, Any]]:
        patterns: List[Dict[str, Any]] = []
        seqs = [
            (
                self.interaction_buffer[i].get("action", ""),
                self.interaction_buffer[i + 1].get("action", ""),
                self.interaction_buffer[i + 2].get("action", ""),
            )
            for i in range(len(self.interaction_buffer) - 2)
        ]
        for seq, count in Counter(seqs).most_common(10):
            if count > 5:
                patterns.append(
                    {
                        "type": "sequence",
                        "pattern": list(seq),
                        "frequency": count,
                        "confidence": min(1.0, count / 10.0),
                        "participants": list(
                            {p for inter in self.interaction_buffer for p in inter.get("participants", [])}
                        ),
                    }
                )
        return patterns

    async def _detect_behavioral_patterns(self) -> List[Dict[str, Any]]:
        patterns: List[Dict[str, Any]] = []
        agents: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for inter in self.interaction_buffer:
            for agent in inter.get("participants", []):
                agents[agent].append(inter)

        for agent, beh in agents.items():
            if len(beh) >= 10:
                collab_rate = sum(1 for b in beh if len(b.get("participants", [])) > 1) / len(beh)
                if collab_rate > 0.7:
                    patterns.append(
                        {
                            "type": "behavioral",
                            "agent": agent,
                            "pattern": "high_collaboration",
                            "metrics": {
                                "action_diversity": len({b.get('action', '') for b in beh}),
                                "collaboration_rate": collab_rate,
                            },
                            "confidence": collab_rate,
                            "participants": list(
                                {p for b in beh for p in b.get("participants", [])}
                            ),
                        }
                    )
        return patterns

    async def _detect_emergent_patterns(self) -> List[Dict[str, Any]]:
        patterns: List[Dict[str, Any]] = []
        G = nx.Graph()
        for inter in self.interaction_buffer[-100:]:
            parts = inter.get("participants", [])
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    G.add_edge(parts[i], parts[j])

        if G.number_of_nodes() > 5:
            try:
                import networkx.algorithms.community as nx_comm
                for comm in nx_comm.greedy_modularity_communities(G):
                    if len(comm) > 3:
                        patterns.append(
                            {
                                "type": "emergent",
                                "pattern": "community_formation",
                                "community": list(comm),
                                "size": len(comm),
                                "confidence": 0.8,
                                "participants": list(comm),
                            }
                        )
            except Exception as exc:  # pragma: no cover
                logger.debug("Community detection failed: %s", exc)
        return patterns


# --------------------------------------------------------------------------- #
#  Pattern miner (unsupervised ML / stats)
# --------------------------------------------------------------------------- #
class PatternMiner:
    """
    Mines deeper patterns from structured data (e.g. crystals, shared state).
    Currently contains basic clustering / anomaly / trend detection.
    """

    def __init__(self):
        self.mined_patterns: List[Dict[str, Any]] = []

    # ............................................................... public
    async def mine_patterns(self, data_sources: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        feats = await self._extract_features(data_sources)
        patterns: List[Dict[str, Any]] = []

        if len(feats) > 10:
            patterns.extend(await self._cluster_analysis(feats))
            patterns.extend(await self._detect_anomalies(feats))
            patterns.extend(await self._analyze_trends(feats))

        self.mined_patterns.extend(patterns)
        return patterns

    # ........................................................... helpers
    async def _extract_features(self, data_sources: Dict[str, List[Any]]) -> np.ndarray:
        vecs = []
        for _, data in data_sources.items():
            for item in data:
                if isinstance(item, dict):
                    vecs.append(
                        [
                            len(item.get("participants", [])),
                            len(str(item.get("content", ""))),
                            item.get("strength", 0.0),
                            item.get("confidence", 0.0),
                        ]
                    )
        return np.array(vecs) if vecs else np.empty((0, 4))

    async def _cluster_analysis(self, feats: np.ndarray) -> List[Dict[str, Any]]:
        if len(feats) < 5:
            return []
        try:
            labels = DBSCAN(eps=0.5, min_samples=3).fit_predict(feats)
            out: List[Dict[str, Any]] = []
            for lbl in set(labels) - {-1}:
                idx = np.where(labels == lbl)[0]
                if len(idx) > 3:
                    out.append(
                        {
                            "type": "cluster",
                            "pattern": "behavioral_cluster",
                            "size": len(idx),
                            "center": feats[idx].mean(axis=0).tolist(),
                            "confidence": min(1.0, len(idx) / 10.0),
                        }
                    )
            return out
        except Exception as exc:  # pragma: no cover
            logger.error("Cluster analysis error: %s", exc)
            return []

    async def _detect_anomalies(self, feats: np.ndarray) -> List[Dict[str, Any]]:
        if len(feats) < 10:
            return []
        mean, std = feats.mean(axis=0), feats.std(axis=0)
        anomalies = []
        for i, vec in enumerate(feats):
            z = np.abs((vec - mean) / (std + 1e-10))
            if (z > 3).any():
                anomalies.append(
                    {
                        "type": "anomaly",
                        "pattern": "statistical_outlier",
                        "index": i,
                        "z_scores": z.tolist(),
                        "confidence": min(1.0, z.max() / 5.0),
                    }
                )
        return anomalies

    async def _analyze_trends(self, feats: np.ndarray) -> List[Dict[str, Any]]:
        if len(feats) < 20:
            return []
        trends = []
        for dim in range(feats.shape[1]):
            y = feats[:, dim]
            slope = np.polyfit(np.arange(len(y)), y, 1)[0]
            if abs(slope) > 0.01:
                trends.append(
                    {
                        "type": "trend",
                        "pattern": f"{'increasing' if slope > 0 else 'decreasing'}_trend",
                        "dimension": dim,
                        "slope": float(slope),
                        "confidence": min(1.0, abs(slope) * 10),
                    }
                )
        return trends


# --------------------------------------------------------------------------- #
#  Collective Insight Engine
# --------------------------------------------------------------------------- #
class CollectiveInsightEngine:
    """
    Top-level orchestrator that converts patterns into durable
    **CollectiveInsight** objects and builds a similarity network.
    """

    def __init__(self):
        self.pattern_detector = PatternDetector()
        self.pattern_miner = PatternMiner()

        self.insights: Dict[str, CollectiveInsight] = {}
        self.insight_network = nx.DiGraph()

        self.min_confidence = 0.6
        self.min_contributors = 2

        self._analysis_task: Optional[asyncio.Task] = None
        logger.info("CollectiveInsightEngine initialised")

    # ..................................................... lifecycle
    async def start(self):
        self._analysis_task = asyncio.create_task(self._continuous_analysis())

    async def stop(self):
        if self._analysis_task:
            self._analysis_task.cancel()
            await asyncio.gather(self._analysis_task, return_exceptions=True)

    # ......................................................... public API
    async def process_memory_data(self, memory_data: Dict[str, Any]) -> List[str]:
        new_ids: List[str] = []

        # ------------------------------------------------- working-memory
        interactions = memory_data.get("working_memory_interactions", [])
        if interactions:
            for pat in await self.pattern_detector.analyze_interactions(interactions):
                if pat["confidence"] >= self.min_confidence:
                    cid = await self._create_insight_from_pattern(pat, memory_data)
                    if cid:
                        new_ids.append(cid)

        # ------------------------------------------------- shared-memory
        shared = memory_data.get("shared_memory_states", [])
        if shared:
            new_ids.extend(await self._analyze_network_state(shared))

        # ------------------------------------------------ connect graph
        await self._connect_insights(new_ids)
        return new_ids

    async def get_insights(self, filters: Optional[Dict[str, Any]] = None) -> List[CollectiveInsight]:
        insights = list(self.insights.values())
        if filters:
            if "type" in filters:
                insights = [i for i in insights if i.type == filters["type"]]
            if "min_confidence" in filters:
                insights = [i for i in insights if i.confidence >= filters["min_confidence"]]
            if "contributor" in filters:
                insights = [i for i in insights if filters["contributor"] in i.contributors]
            if "max_age_hours" in filters:
                max_age = timedelta(hours=filters["max_age_hours"])
                insights = [i for i in insights if datetime.now() - i.discovered_at <= max_age]

        insights.sort(key=lambda i: i.impact_score, reverse=True)
        return insights

    async def explain_insight(self, insight_id: str) -> Dict[str, Any]:
        if insight_id not in self.insights:
            return {}

        ins = self.insights[insight_id]
        connected = [
            {
                "id": nb,
                "description": self.insights[nb].description,
                "similarity": self.insight_network[insight_id][nb].get("weight", 0),
            }
            for nb in self.insight_network.neighbors(insight_id)
            if nb in self.insights
        ]

        return {
            "insight": ins.to_dict(),
            "explanation": {
                "evidence_count": len(ins.evidence),
                "contributor_count": len(ins.contributors),
                "age_hours": (datetime.now() - ins.discovered_at).total_seconds() / 3600,
                "connected_insights": connected,
                "impact_factors": {
                    "contributor_score": min(1.0, len(ins.contributors) / 10.0),
                    "confidence_score": ins.confidence,
                    "network_centrality": len(connected) / max(1, self.insight_network.number_of_nodes()),
                    "recency_score": max(0, 1 - ((datetime.now() - ins.discovered_at).days / 7)),
                },
            },
        }

    def get_network_stats(self) -> Dict[str, Any]:
        stats = {
            "total_insights": len(self.insights),
            "insight_types": defaultdict(int),
            "average_confidence": 0.0,
            "average_impact": 0.0,
            "network_density": 0.0,
            "top_contributors": [],
        }

        if self.insights:
            for ins in self.insights.values():
                stats["insight_types"][ins.type.value] += 1

            stats["average_confidence"] = sum(i.confidence for i in self.insights.values()) / len(self.insights)
            stats["average_impact"] = sum(i.impact_score for i in self.insights.values()) / len(self.insights)

            if self.insight_network.number_of_nodes() > 1:
                stats["network_density"] = nx.density(self.insight_network)

            contrib = defaultdict(int)
            for ins in self.insights.values():
                for c in ins.contributors:
                    contrib[c] += 1
            stats["top_contributors"] = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:10]

        return stats

    # ................................................................ internals
    async def _create_insight_from_pattern(self, pat: Dict[str, Any], mem: Dict[str, Any]) -> Optional[str]:
        ins_type = self._determine_insight_type(pat)
        contributors = self._extract_contributors(pat, mem)
        if len(contributors) < self.min_contributors:
            return None

        cid = f"insight_{int(datetime.now().timestamp()*1000)}"
        ins = CollectiveInsight(
            id=cid,
            type=ins_type,
            description=self._generate_description(pat, ins_type),
            contributors=contributors,
            confidence=pat.get("confidence", 0.0),
            evidence=[pat],
            metadata=pat.get("metadata", {}),
        )

        ins.impact_score = self._calculate_impact_score(ins, mem)
        ins.vector_representation = await self._generate_vector_representation(pat)

        self.insights[cid] = ins
        self.insight_network.add_node(cid, **ins.to_dict())
        logger.info("Created insight %s  (%s)", cid, ins.description)
        return cid

    # ................................... helper – metadata / description / score
    @staticmethod
    def _determine_insight_type(pat: Dict[str, Any]) -> InsightType:
        typ = pat.get("type", "").lower()
        return {
            "emergent": InsightType.EMERGENT_PATTERN,
            "behavioral": InsightType.NETWORK_BEHAVIOR,
            "anomaly": InsightType.ANOMALY,
            "trend": InsightType.TREND,
            "cluster": InsightType.COLLECTIVE_SOLUTION,
        }.get(typ, InsightType.EMERGENT_PATTERN)

    @staticmethod
    def _generate_description(pat: Dict[str, Any], typ: InsightType) -> str:
        name = pat.get("pattern", "unknown")
        if typ == InsightType.EMERGENT_PATTERN and name == "community_formation":
            return f"Emergent community of {pat.get('size', 0)} agents"
        if typ == InsightType.NETWORK_BEHAVIOR and name == "high_collaboration":
            a = pat.get("agent", "unknown")
            r = pat.get("metrics", {}).get("collaboration_rate", 0)
            return f"Agent {a} shows high collaboration behaviour (rate {r:.2f})"
        if typ == InsightType.ANOMALY:
            return f"Anomalous pattern: {name}"
        if typ == InsightType.TREND:
            return f"Trend detected: {name}"
        return f"{typ.value}: {name}"

    def _extract_contributors(self, pat: Dict[str, Any], mem: Dict[str, Any]) -> List[str]:
        contrib: Set[str] = set()
        contrib.update(pat.get("participants", []))
        if "agent" in pat:
            contrib.add(pat["agent"])
        if "community" in pat:
            contrib.update(pat["community"])

        for inter in mem.get("working_memory_interactions", []):
            if self._pattern_matches_interaction(pat, inter):
                contrib.update(inter.get("participants", []))
        return list(contrib)

    @staticmethod
    def _pattern_matches_interaction(pat: Dict[str, Any], inter: Dict[str, Any]) -> bool:
        patt = pat.get("pattern")
        if isinstance(patt, list) and inter.get("action") in patt:
            return True
        if patt == inter.get("action"):
            return True
        return bool(set(pat.get("participants", [])).intersection(inter.get("participants", [])))

    def _calculate_impact_score(self, ins: CollectiveInsight, mem: Dict[str, Any]) -> float:
        score = 0.3 * min(1.0, len(ins.contributors) / 10.0) + 0.3 * ins.confidence
        if self.insight_network.number_of_nodes():
            try:
                centr = nx.degree_centrality(self.insight_network)
                score += 0.2 * centr.get(ins.id, 0)
            except Exception:  # pragma: no cover
                pass
        age_h = (datetime.now() - ins.discovered_at).total_seconds() / 3600
        score += 0.2 * max(0, 1 - age_h / 168)
        return min(1.0, score)

    async def _generate_vector_representation(self, pat: Dict[str, Any]) -> np.ndarray:
        types = ["sequence", "behavioral", "emergent", "cluster", "anomaly", "trend"]
        vec = [1.0 if pat.get("type") == t else 0.0 for t in types]
        vec.extend(
            [
                pat.get("confidence", 0.0),
                pat.get("frequency", 0.0) / 100.0,
                len(pat.get("participants", [])) / 10.0,
            ]
        )
        return np.array(vec, dtype=float)

    # .................................................... shared-memory analysis
    async def _analyze_network_state(self, shared: List[Dict[str, Any]]) -> List[str]:
        ids: List[str] = []
        metrics = self._calculate_network_health(shared)
        if metrics["sync_efficiency"] < 0.5:
            cid = await self._create_network_insight(
                "Network sync efficiency below threshold", InsightType.ANOMALY, metrics
            )
            if cid:
                ids.append(cid)

        for pat in self._analyze_collaboration_patterns(shared):
            cid = await self._create_network_insight(pat["description"], InsightType.NETWORK_BEHAVIOR, pat)
            if cid:
                ids.append(cid)
        return ids

    @staticmethod
    def _calculate_network_health(shared: List[Dict[str, Any]]) -> Dict[str, float]:
        if not shared:
            return {"sync_efficiency": 1.0, "connectivity": 1.0}
        synced = sum(1 for o in shared if o.get("status") == "synced")
        parts = {p for o in shared for p in o.get("participants", [])}
        return {
            "sync_efficiency": synced / len(shared),
            "connectivity": min(1.0, len(parts) / 10.0),
            "total_objects": len(shared),
            "active_agents": len(parts),
        }

    @staticmethod
    def _analyze_collaboration_patterns(shared: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        counts = defaultdict(int)
        for obj in shared:
            for ag in obj.get("participants", []):
                counts[ag] += 1
        return [
            {
                "description": f"Agent {ag} is a super-collaborator with {cnt} objects",
                "agent": ag,
                "collaboration_count": cnt,
                "confidence": min(1.0, cnt / 10.0),
            }
            for ag, cnt in counts.items()
            if cnt > 5
        ]

    async def _create_network_insight(self, desc: str, typ: InsightType, data: Dict[str, Any]) -> Optional[str]:
        cid = f"insight_network_{int(datetime.now().timestamp()*1000)}"
        contributors = ["network_monitor"] if "active_agents" in data else [data.get("agent", "system")]
        ins = CollectiveInsight(
            id=cid, type=typ, description=desc, contributors=contributors, confidence=data.get("confidence", 0.7), evidence=[data], metadata=data
        )
        self.insights[cid] = ins
        self.insight_network.add_node(cid, **ins.to_dict())
        return cid

    # ............................................................ networking
    async def _connect_insights(self, new_ids: List[str]):
        for nid in new_ids:
            n_ins = self.insights.get(nid)
            if not n_ins:
                continue
            for eid, e_ins in self.insights.items():
                if eid == nid:
                    continue
                sim = await self._calculate_insight_similarity(n_ins, e_ins)
                if sim > 0.7:
                    self.insight_network.add_edge(eid, nid, weight=sim)

    async def _calculate_insight_similarity(self, a: CollectiveInsight, b: CollectiveInsight) -> float:
        sim = 0.3 if a.type == b.type else 0.0
        if a.contributors and b.contributors:
            sim += 0.3 * len(set(a.contributors) & set(b.contributors)) / len(set(a.contributors) | set(b.contributors))
        if a.vector_representation is not None and b.vector_representation is not None:
            dot = float(np.dot(a.vector_representation, b.vector_representation))
            denom = float(np.linalg.norm(a.vector_representation) * np.linalg.norm(b.vector_representation))
            if denom:
                sim += 0.4 * (dot / denom)
        return sim

    # ........................................................ background task
    async def _continuous_analysis(self):
        while True:
            try:
                await asyncio.sleep(300)
                for ins in self.insights.values():
                    ins.impact_score = self._calculate_impact_score(ins, {})
                old = [
                    iid
                    for iid, ins in self.insights.items()
                    if (datetime.now() - ins.discovered_at).days > 30 and ins.impact_score < 0.2
                ]
                for iid in old:
                    self.insights.pop(iid, None)
                    if iid in self.insight_network:
                        self.insight_network.remove_node(iid)
                logger.info("Continuous analysis: %d active insights", len(self.insights))
            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover
                logger.error("Continuous analysis error: %s", exc)
