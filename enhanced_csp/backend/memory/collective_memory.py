"""
Collective Memory Layer Implementation
======================================

This module implements the Collective Memory layer for emergent intelligence
and pattern recognition across the entire agent network.
"""

import asyncio
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, Counter
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import logging
import json

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of collective insights"""
    EMERGENT_PATTERN = "emergent_pattern"
    COLLECTIVE_SOLUTION = "collective_solution"
    NETWORK_BEHAVIOR = "network_behavior"
    OPTIMIZATION = "optimization"
    ANOMALY = "anomaly"
    TREND = "trend"


@dataclass
class CollectiveInsight:
    """Represents a collective insight or pattern"""
    id: str
    type: InsightType
    description: str
    contributors: List[str]  # Agent IDs that contributed
    confidence: float  # 0.0 to 1.0
    evidence: List[Dict[str, Any]]
    discovered_at: datetime = field(default_factory=datetime.now)
    vector_representation: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    impact_score: float = 0.0
    applications: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'type': self.type.value,
            'description': self.description,
            'contributors': self.contributors,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'discovered_at': self.discovered_at.isoformat(),
            'metadata': self.metadata,
            'impact_score': self.impact_score,
            'applications': self.applications
        }


class PatternDetector:
    """Detects patterns across agent interactions"""
    
    def __init__(self):
        self.interaction_buffer: List[Dict[str, Any]] = []
        self.buffer_size = 1000
        self.pattern_threshold = 0.7
        
    async def analyze_interactions(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze interactions for patterns"""
        self.interaction_buffer.extend(interactions)
        
        # Keep buffer size limited
        if len(self.interaction_buffer) > self.buffer_size:
            self.interaction_buffer = self.interaction_buffer[-self.buffer_size:]
        
        patterns = []
        
        # Sequence pattern detection
        sequence_patterns = await self._detect_sequence_patterns()
        patterns.extend(sequence_patterns)
        
        # Behavioral pattern detection
        behavioral_patterns = await self._detect_behavioral_patterns()
        patterns.extend(behavioral_patterns)
        
        # Emergence detection
        emergent_patterns = await self._detect_emergent_patterns()
        patterns.extend(emergent_patterns)
        
        return patterns
    
    async def _detect_sequence_patterns(self) -> List[Dict[str, Any]]:
        """Detect sequential patterns in interactions"""
        patterns = []
        
        # Create action sequences
        sequences = []
        for i in range(len(self.interaction_buffer) - 2):
            seq = [
                self.interaction_buffer[i].get('action', ''),
                self.interaction_buffer[i+1].get('action', ''),
                self.interaction_buffer[i+2].get('action', '')
            ]
            sequences.append(tuple(seq))
        
        # Find frequent sequences
        seq_counter = Counter(sequences)
        for seq, count in seq_counter.most_common(10):
            if count > 5:  # Minimum occurrence threshold
                patterns.append({
                    'type': 'sequence',
                    'pattern': list(seq),
                    'frequency': count,
                    'confidence': min(1.0, count / 10.0)
                })
        
        return patterns
    
    async def _detect_behavioral_patterns(self) -> List[Dict[str, Any]]:
        """Detect behavioral patterns across agents"""
        patterns = []
        
        # Group interactions by agent
        agent_behaviors = defaultdict(list)
        for interaction in self.interaction_buffer:
            for agent in interaction.get('participants', []):
                agent_behaviors[agent].append(interaction)
        
        # Analyze agent behaviors
        for agent, behaviors in agent_behaviors.items():
            if len(behaviors) > 10:
                # Calculate behavioral metrics
                action_diversity = len(set(b.get('action', '') for b in behaviors))
                collaboration_rate = sum(
                    1 for b in behaviors if len(b.get('participants', [])) > 1
                ) / len(behaviors)
                
                if collaboration_rate > 0.7:
                    patterns.append({
                        'type': 'behavioral',
                        'agent': agent,
                        'pattern': 'high_collaboration',
                        'metrics': {
                            'action_diversity': action_diversity,
                            'collaboration_rate': collaboration_rate
                        },
                        'confidence': collaboration_rate
                    })
        
        return patterns
    
    async def _detect_emergent_patterns(self) -> List[Dict[str, Any]]:
        """Detect emergent patterns from collective behavior"""
        patterns = []
        
        # Create interaction network
        G = nx.Graph()
        for interaction in self.interaction_buffer[-100:]:  # Recent interactions
            participants = interaction.get('participants', [])
            for i in range(len(participants)):
                for j in range(i + 1, len(participants)):
                    G.add_edge(participants[i], participants[j])
        
        if G.number_of_nodes() > 5:
            # Detect communities
            try:
                import networkx.algorithms.community as nx_comm
                communities = list(nx_comm.greedy_modularity_communities(G))
                
                for idx, community in enumerate(communities):
                    if len(community) > 3:
                        patterns.append({
                            'type': 'emergent',
                            'pattern': 'community_formation',
                            'community': list(community),
                            'size': len(community),
                            'confidence': 0.8
                        })
            except:
                pass  # Community detection may fail on small graphs
        
        return patterns


class PatternMiner:
    """
    Mines patterns and insights from collective agent behavior.
    Uses machine learning to identify complex patterns.
    """
    
    def __init__(self):
        self.feature_extractors: List[Callable] = []
        self.pattern_models = {}
        self.mined_patterns: List[Dict[str, Any]] = []
        
    async def mine_patterns(self, data_sources: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Mine patterns from multiple data sources"""
        patterns = []
        
        # Extract features from each data source
        features = await self._extract_features(data_sources)
        
        if len(features) > 10:
            # Cluster analysis
            clusters = await self._cluster_analysis(features)
            patterns.extend(clusters)
            
            # Anomaly detection
            anomalies = await self._detect_anomalies(features)
            patterns.extend(anomalies)
            
            # Trend analysis
            trends = await self._analyze_trends(features)
            patterns.extend(trends)
        
        self.mined_patterns.extend(patterns)
        return patterns
    
    async def _extract_features(self, data_sources: Dict[str, List[Any]]) -> np.ndarray:
        """Extract features from data sources"""
        feature_vectors = []
        
        for source_name, data in data_sources.items():
            for item in data:
                features = []
                
                # Basic features
                if isinstance(item, dict):
                    features.extend([
                        len(item.get('participants', [])),
                        len(str(item.get('content', ''))),
                        item.get('strength', 0.0),
                        item.get('confidence', 0.0)
                    ])
                
                if features:
                    feature_vectors.append(features)
        
        return np.array(feature_vectors) if feature_vectors else np.array([])
    
    async def _cluster_analysis(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Perform cluster analysis on features"""
        patterns = []
        
        if len(features) < 5:
            return patterns
        
        try:
            # DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=3)
            labels = clustering.fit_predict(features)
            
            # Analyze clusters
            unique_labels = set(labels) - {-1}  # Exclude noise
            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]
                cluster_size = len(cluster_indices)
                
                if cluster_size > 3:
                    # Calculate cluster center
                    cluster_center = features[cluster_indices].mean(axis=0)
                    
                    patterns.append({
                        'type': 'cluster',
                        'pattern': 'behavioral_cluster',
                        'size': cluster_size,
                        'center': cluster_center.tolist(),
                        'confidence': min(1.0, cluster_size / 10.0)
                    })
        except Exception as e:
            logger.error(f"Cluster analysis error: {e}")
        
        return patterns
    
    async def _detect_anomalies(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Detect anomalies in features"""
        anomalies = []
        
        if len(features) < 10:
            return anomalies
        
        try:
            # Simple statistical anomaly detection
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            
            for idx, feature_vec in enumerate(features):
                # Calculate z-score
                z_scores = np.abs((feature_vec - mean) / (std + 1e-10))
                
                if np.any(z_scores > 3):  # 3 standard deviations
                    anomalies.append({
                        'type': 'anomaly',
                        'pattern': 'statistical_outlier',
                        'index': idx,
                        'z_scores': z_scores.tolist(),
                        'confidence': min(1.0, np.max(z_scores) / 5.0)
                    })
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
        
        return anomalies
    
    async def _analyze_trends(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze trends in features over time"""
        trends = []
        
        if len(features) < 20:
            return trends
        
        try:
            # Analyze each feature dimension
            for dim in range(features.shape[1]):
                values = features[:, dim]
                
                # Simple trend detection (linear regression)
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)
                slope = coeffs[0]
                
                if abs(slope) > 0.01:  # Significant trend
                    trend_type = 'increasing' if slope > 0 else 'decreasing'
                    trends.append({
                        'type': 'trend',
                        'pattern': f'{trend_type}_trend',
                        'dimension': dim,
                        'slope': float(slope),
                        'confidence': min(1.0, abs(slope) * 10)
                    })
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
        
        return trends


class CollectiveInsightEngine:
    """
    Main engine for generating collective insights from the agent network.
    Combines patterns from multiple sources to form high-level insights.
    """
    
    def __init__(self):
        self.pattern_detector = PatternDetector()
        self.pattern_miner = PatternMiner()
        self.insights: Dict[str, CollectiveInsight] = {}
        self.insight_network = nx.DiGraph()
        
        # Insight generation parameters
        self.min_confidence = 0.6
        self.min_contributors = 2
        
        # Background tasks
        self._analysis_task: Optional[asyncio.Task] = None
        
        logger.info("CollectiveInsightEngine initialized")
    
    async def start(self):
        """Start the insight engine"""
        self._analysis_task = asyncio.create_task(self._continuous_analysis())
    
    async def stop(self):
        """Stop the insight engine"""
        if self._analysis_task:
            self._analysis_task.cancel()
            await asyncio.gather(self._analysis_task, return_exceptions=True)
    
    async def process_memory_data(self, memory_data: Dict[str, Any]) -> List[str]:
        """Process memory data from all layers to generate insights"""
        insight_ids = []
        
        # Extract interactions from working memory
        interactions = memory_data.get('working_memory_interactions', [])
        if interactions:
            patterns = await self.pattern_detector.analyze_interactions(interactions)
            
            for pattern in patterns:
                if pattern['confidence'] >= self.min_confidence:
                    insight_id = await self._create_insight_from_pattern(pattern, memory_data)
                    if insight_id:
                        insight_ids.append(insight_id)
        
        # Analyze shared memory patterns
        shared_data = memory_data.get('shared_memory_states', [])
        if shared_data:
            network_insights = await self._analyze_network_state(shared_data)
            insight_ids.extend(network_insights)
        
        # Connect related insights
        await self._connect_insights(insight_ids)
        
        return insight_ids
    
    async def _create_insight_from_pattern(self, pattern: Dict[str, Any], 
                                         memory_data: Dict[str, Any]) -> Optional[str]:
        """Create an insight from a detected pattern"""
        # Determine insight type
        insight_type = self._determine_insight_type(pattern)
        
        # Extract contributors
        contributors = self._extract_contributors(pattern, memory_data)
        
        if len(contributors) < self.min_contributors:
            return None
        
        # Generate insight ID
        insight_id = f"insight_{int(datetime.now().timestamp() * 1000)}"
        
        # Create insight description
        description = self._generate_description(pattern, insight_type)
        
        # Create insight
        insight = CollectiveInsight(
            id=insight_id,
            type=insight_type,
            description=description,
            contributors=contributors,
            confidence=pattern.get('confidence', 0.0),
            evidence=[pattern],
            metadata=pattern.get('metadata', {})
        )
        
        # Calculate impact score
        insight.impact_score = self._calculate_impact_score(insight, memory_data)
        
        # Generate vector representation
        insight.vector_representation = await self._generate_vector_representation(pattern)
        
        # Store insight
        self.insights[insight_id] = insight
        
        # Add to insight network
        self.insight_network.add_node(insight_id, **insight.to_dict())
        
        logger.info(f"Created insight: {insight_id} - {description}")
        
        return insight_id
    
    def _determine_insight_type(self, pattern: Dict[str, Any]) -> InsightType:
        """Determine the type of insight from pattern"""
        pattern_type = pattern.get('type', '').lower()
        
        if pattern_type == 'emergent':
            return InsightType.EMERGENT_PATTERN
        elif pattern_type == 'behavioral':
            return InsightType.NETWORK_BEHAVIOR
        elif pattern_type == 'anomaly':
            return InsightType.ANOMALY
        elif pattern_type == 'trend':
            return InsightType.TREND
        elif pattern_type == 'cluster':
            return InsightType.COLLECTIVE_SOLUTION
        else:
            return InsightType.EMERGENT_PATTERN
    
    def _extract_contributors(self, pattern: Dict[str, Any], 
                            memory_data: Dict[str, Any]) -> List[str]:
        """Extract contributing agents from pattern"""
        contributors = set()
        
        # Direct contributors
        if 'agent' in pattern:
            contributors.add(pattern['agent'])
        if 'participants' in pattern:
            contributors.update(pattern['participants'])
        if 'community' in pattern:
            contributors.update(pattern['community'])
        
        # Contributors from memory data
        for interaction in memory_data.get('working_memory_interactions', []):
            if self._pattern_matches_interaction(pattern, interaction):
                contributors.update(interaction.get('participants', []))
        
        return list(contributors)
    
    def _pattern_matches_interaction(self, pattern: Dict[str, Any], 
                                   interaction: Dict[str, Any]) -> bool:
        """Check if pattern matches an interaction"""
        # Simple matching logic - could be enhanced
        if pattern.get('pattern') == interaction.get('action'):
            return True
        if set(pattern.get('participants', [])).intersection(
            set(interaction.get('participants', []))
        ):
            return True
        return False
    
    def _generate_description(self, pattern: Dict[str, Any], 
                            insight_type: InsightType) -> str:
        """Generate human-readable description of insight"""
        pattern_name = pattern.get('pattern', 'unknown')
        pattern_type = pattern.get('type', 'unknown')
        
        if insight_type == InsightType.EMERGENT_PATTERN:
            if pattern_name == 'community_formation':
                size = pattern.get('size', 0)
                return f"Emergent community of {size} agents forming collaborative network"
            else:
                return f"Emergent pattern detected: {pattern_name}"
        
        elif insight_type == InsightType.NETWORK_BEHAVIOR:
            if pattern_name == 'high_collaboration':
                agent = pattern.get('agent', 'unknown')
                rate = pattern.get('metrics', {}).get('collaboration_rate', 0)
                return f"Agent {agent} showing high collaboration behavior (rate: {rate:.2f})"
            else:
                return f"Network behavior pattern: {pattern_name}"
        
        elif insight_type == InsightType.ANOMALY:
            return f"Anomalous pattern detected: {pattern_name}"
        
        elif insight_type == InsightType.TREND:
            direction = pattern_name.split('_')[0]
            return f"Trend detected: {direction} pattern in network behavior"
        
        else:
            return f"{insight_type.value}: {pattern_name}"
    
    def _calculate_impact_score(self, insight: CollectiveInsight, 
                              memory_data: Dict[str, Any]) -> float:
        """Calculate the impact score of an insight"""
        score = 0.0
        
        # Factor 1: Number of contributors
        score += min(1.0, len(insight.contributors) / 10.0) * 0.3
        
        # Factor 2: Confidence level
        score += insight.confidence * 0.3
        
        # Factor 3: Network centrality (if applicable)
        if self.insight_network.number_of_nodes() > 0:
            try:
                centrality = nx.degree_centrality(self.insight_network)
                if insight.id in centrality:
                    score += centrality[insight.id] * 0.2
            except:
                pass
        
        # Factor 4: Recency
        age_hours = (datetime.now() - insight.discovered_at).total_seconds() / 3600
        recency_factor = max(0, 1 - (age_hours / 168))  # Decay over a week
        score += recency_factor * 0.2
        
        return min(1.0, score)
    
    async def _generate_vector_representation(self, pattern: Dict[str, Any]) -> np.ndarray:
        """Generate vector representation of pattern for similarity comparisons"""
        # Simplified vector generation - in production would use embeddings
        vector = []
        
        # Pattern type encoding
        pattern_types = ['sequence', 'behavioral', 'emergent', 'cluster', 'anomaly', 'trend']
        type_vec = [1.0 if pattern.get('type') == pt else 0.0 for pt in pattern_types]
        vector.extend(type_vec)
        
        # Numerical features
        vector.append(pattern.get('confidence', 0.0))
        vector.append(pattern.get('frequency', 0.0) / 100.0)  # Normalized
        vector.append(len(pattern.get('participants', [])) / 10.0)  # Normalized
        
        return np.array(vector)
    
    async def _analyze_network_state(self, shared_data: List[Dict[str, Any]]) -> List[str]:
        """Analyze network-wide state from shared memory data"""
        insight_ids = []
        
        # Network health analysis
        health_metrics = self._calculate_network_health(shared_data)
        if health_metrics['sync_efficiency'] < 0.5:
            insight_id = await self._create_network_insight(
                'Network synchronization efficiency below threshold',
                InsightType.ANOMALY,
                health_metrics
            )
            if insight_id:
                insight_ids.append(insight_id)
        
        # Collaboration patterns
        collab_patterns = self._analyze_collaboration_patterns(shared_data)
        for pattern in collab_patterns:
            insight_id = await self._create_network_insight(
                pattern['description'],
                InsightType.NETWORK_BEHAVIOR,
                pattern
            )
            if insight_id:
                insight_ids.append(insight_id)
        
        return insight_ids
    
    def _calculate_network_health(self, shared_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate network health metrics"""
        if not shared_data:
            return {'sync_efficiency': 1.0, 'connectivity': 1.0}
        
        # Calculate sync efficiency
        synced_objects = sum(1 for obj in shared_data if obj.get('status') == 'synced')
        sync_efficiency = synced_objects / len(shared_data) if shared_data else 1.0
        
        # Calculate connectivity
        all_participants = set()
        for obj in shared_data:
            all_participants.update(obj.get('participants', []))
        
        connectivity = min(1.0, len(all_participants) / 10.0)  # Normalized
        
        return {
            'sync_efficiency': sync_efficiency,
            'connectivity': connectivity,
            'total_objects': len(shared_data),
            'active_agents': len(all_participants)
        }
    
    def _analyze_collaboration_patterns(self, shared_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze collaboration patterns from shared memory"""
        patterns = []
        
        # Agent collaboration frequency
        agent_collabs = defaultdict(int)
        for obj in shared_data:
            participants = obj.get('participants', [])
            for agent in participants:
                agent_collabs[agent] += 1
        
        # Identify super-collaborators
        for agent, count in agent_collabs.items():
            if count > 5:
                patterns.append({
                    'description': f"Agent {agent} is a super-collaborator with {count} shared objects",
                    'agent': agent,
                    'collaboration_count': count,
                    'confidence': min(1.0, count / 10.0)
                })
        
        return patterns
    
    async def _create_network_insight(self, description: str, 
                                    insight_type: InsightType,
                                    data: Dict[str, Any]) -> Optional[str]:
        """Create an insight about network state"""
        insight_id = f"insight_network_{int(datetime.now().timestamp() * 1000)}"
        
        # Extract all agents from the data
        contributors = []
        if 'active_agents' in data:
            contributors = ['network_monitor']  # System-generated insight
        elif 'agent' in data:
            contributors = [data['agent']]
        
        insight = CollectiveInsight(
            id=insight_id,
            type=insight_type,
            description=description,
            contributors=contributors,
            confidence=data.get('confidence', 0.7),
            evidence=[data],
            metadata=data
        )
        
        self.insights[insight_id] = insight
        self.insight_network.add_node(insight_id, **insight.to_dict())
        
        return insight_id
    
    async def _connect_insights(self, new_insight_ids: List[str]):
        """Connect related insights in the network"""
        for new_id in new_insight_ids:
            if new_id not in self.insights:
                continue
            
            new_insight = self.insights[new_id]
            
            # Connect to similar insights
            for existing_id, existing_insight in self.insights.items():
                if existing_id == new_id:
                    continue
                
                similarity = await self._calculate_insight_similarity(new_insight, existing_insight)
                
                if similarity > 0.7:
                    self.insight_network.add_edge(existing_id, new_id, weight=similarity)
    
    async def _calculate_insight_similarity(self, insight1: CollectiveInsight, 
                                          insight2: CollectiveInsight) -> float:
        """Calculate similarity between two insights"""
        similarity = 0.0
        
        # Type similarity
        if insight1.type == insight2.type:
            similarity += 0.3
        
        # Contributor overlap
        contributors1 = set(insight1.contributors)
        contributors2 = set(insight2.contributors)
        if contributors1 and contributors2:
            overlap = len(contributors1.intersection(contributors2)) / len(contributors1.union(contributors2))
            similarity += overlap * 0.3
        
        # Vector similarity (if available)
        if insight1.vector_representation is not None and insight2.vector_representation is not None:
            # Cosine similarity
            dot_product = np.dot(insight1.vector_representation, insight2.vector_representation)
            norm1 = np.linalg.norm(insight1.vector_representation)
            norm2 = np.linalg.norm(insight2.vector_representation)
            if norm1 > 0 and norm2 > 0:
                cosine_sim = dot_product / (norm1 * norm2)
                similarity += cosine_sim * 0.4
        
        return similarity
    
    async def _continuous_analysis(self):
        """Background task for continuous insight analysis"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Update impact scores
                for insight in self.insights.values():
                    insight.impact_score = self._calculate_impact_score(insight, {})
                
                # Prune old, low-impact insights
                to_remove = []
                for insight_id, insight in self.insights.items():
                    age_days = (datetime.now() - insight.discovered_at).days
                    if age_days > 30 and insight.impact_score < 0.2:
                        to_remove.append(insight_id)
                
                for insight_id in to_remove:
                    del self.insights[insight_id]
                    if insight_id in self.insight_network:
                        self.insight_network.remove_node(insight_id)
                
                logger.info(f"Continuous analysis: {len(self.insights)} active insights")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Continuous analysis error: {e}")
    
    async def get_insights(self, filters: Optional[Dict[str, Any]] = None) -> List[CollectiveInsight]:
        """Get insights with optional filtering"""
        insights = list(self.insights.values())
        
        if filters:
            # Filter by type
            if 'type' in filters:
                insights = [i for i in insights if i.type == filters['type']]
            
            # Filter by minimum confidence
            if 'min_confidence' in filters:
                insights = [i for i in insights if i.confidence >= filters['min_confidence']]
            
            # Filter by contributor
            if 'contributor' in filters:
                insights = [i for i in insights if filters['contributor'] in i.contributors]
            
            # Filter by recency
            if 'max_age_hours' in filters:
                max_age = timedelta(hours=filters['max_age_hours'])
                insights = [i for i in insights if datetime.now() - i.discovered_at <= max_age]
        
        # Sort by impact score
        insights.sort(key=lambda i: i.impact_score, reverse=True)
        
        return insights
    
    async def explain_insight(self, insight_id: str) -> Dict[str, Any]:
        """Generate detailed explanation of an insight"""
        if insight_id not in self.insights:
            return {}
        
        insight = self.insights[insight_id]
        
        # Get connected insights
        connected = []
        if insight_id in self.insight_network:
            for neighbor in self.insight_network.neighbors(insight_id):
                if neighbor in self.insights:
                    connected.append({
                        'id': neighbor,
                        'description': self.insights[neighbor].description,
                        'similarity': self.insight_network[insight_id][neighbor].get('weight', 0)
                    })
        
        return {
            'insight': insight.to_dict(),
            'explanation': {
                'evidence_count': len(insight.evidence),
                'contributor_count': len(insight.contributors),
                'age_hours': (datetime.now() - insight.discovered_at).total_seconds() / 3600,
                'connected_insights': connected,
                'impact_factors': {
                    'contributor_score': min(1.0, len(insight.contributors) / 10.0),
                    'confidence_score': insight.confidence,
                    'network_centrality': len(connected) / max(1, self.insight_network.number_of_nodes()),
                    'recency_score': max(0, 1 - ((datetime.now() - insight.discovered_at).days / 7))
                }
            }
        }
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get collective memory network statistics"""
        stats = {
            'total_insights': len(self.insights),
            'insight_types': defaultdict(int),
            'average_confidence': 0.0,
            'average_impact': 0.0,
            'network_density': 0.0,
            'top_contributors': []
        }

        if self.insights:
            # Count by type
            for insight in self.insights.values():
                stats['insight_types'][insight.type.value] += 1

            # Calculate averages
            stats['average_confidence'] = (
                sum(i.confidence for i in self.insights.values()) / len(self.insights)
            )
            stats['average_impact'] = (
                sum(i.impact_score for i in self.insights.values()) / len(self.insights)
            )

            # Network metrics
            if self.insight_network.number_of_nodes() > 1:
                stats['network_density'] = nx.density(self.insight_network)

            # Top contributors
            contributor_counts = defaultdict(int)
            for insight in self.insights.values():
                for contributor in insight.contributors:
                    contributor_counts[contributor] += 1

            stats['top_contributors'] = sorted(
                contributor_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]

        return stats
