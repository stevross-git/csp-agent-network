# backend/ai/emergence_detection.py
"""
Emergent Behavior Detection
===========================
Implements emergent behavior detection and amplification with:
- Collective reasoning via PageRank analysis on interaction graphs
- Metacognitive resonance through state correlation analysis
- Consciousness amplification via trend detection in consciousness levels
- Behavior amplification with feedback strategies
- Network topology analysis for emergence patterns
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Set
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from scipy import stats
import uuid

logger = logging.getLogger(__name__)

@dataclass
class InteractionEvent:
    """Individual interaction between agents"""
    id: str
    source_agent: str
    target_agent: str
    interaction_type: str
    influence_weight: float
    timestamp: datetime
    content: Dict = field(default_factory=dict)
    emergence_indicators: List[str] = field(default_factory=list)

@dataclass
class EmergentPattern:
    """Detected emergent behavior pattern"""
    id: str
    pattern_type: str
    participating_agents: List[str]
    strength: float
    confidence: float
    pattern_data: Dict
    detection_method: str
    emergence_level: str  # local, global, systemic
    timestamp: datetime
    duration: Optional[float] = None

@dataclass
class CollectiveIntelligence:
    """Collective intelligence analysis results"""
    pagerank_scores: Dict[str, float]
    collective_intelligence: float
    network_density: float
    dominant_agents: List[Dict]
    influence_clusters: List[List[str]]
    emergence_score: float
    timestamp: datetime

@dataclass
class ConsciousnessAmplification:
    """Consciousness amplification analysis"""
    amplification_factor: float
    trend: str
    strategy: str
    trend_slope: float
    original_levels: List[float]
    amplified_levels: List[float]
    mean_improvement: float
    confidence_interval: Tuple[float, float]
    sustainability_score: float

class EmergentBehaviorDetection:
    """
    Advanced emergent behavior detection system implementing collective intelligence
    analysis, metacognitive resonance detection, and consciousness amplification.
    """
    
    def __init__(self):
        # Network analysis
        self.interaction_graph = nx.DiGraph()
        self.interaction_history: List[InteractionEvent] = []
        self.emergence_patterns: List[EmergentPattern] = []
        
        # Collective intelligence tracking
        self.collective_intelligence_history: List[CollectiveIntelligence] = []
        self.agent_influence_scores: Dict[str, List[float]] = defaultdict(list)
        
        # Consciousness tracking
        self.consciousness_levels: Dict[str, List[float]] = defaultdict(list)
        self.amplification_history: List[ConsciousnessAmplification] = []
        
        # Emergence detection parameters
        self.min_interaction_threshold = 5
        self.emergence_detection_window = 300.0  # 5 minutes
        self.amplification_sensitivity = 0.01
        self.resonance_threshold = 0.7
        
        # Pattern recognition
        self.known_emergence_patterns = {
            'swarm_intelligence': {'min_agents': 3, 'interaction_density': 0.6},
            'collective_decision': {'min_agents': 2, 'consensus_threshold': 0.8},
            'emergent_creativity': {'diversity_threshold': 0.5, 'novelty_score': 0.7},
            'phase_transition': {'coherence_change': 0.3, 'synchronization': 0.8},
            'cascade_effect': {'propagation_speed': 0.1, 'amplitude_growth': 1.2}
        }
        
        # Network analysis cache
        self._network_cache = {}
        self._cache_timestamp = datetime.now()
        self._cache_duration = timedelta(seconds=30)
    
    async def analyze_collective_reasoning(self, agent_interactions: List[Dict]) -> Dict:
        """
        Analyze collective reasoning using PageRank analysis on interaction graphs
        
        Args:
            agent_interactions: List of interaction dictionaries
            
        Returns:
            Collective intelligence metrics and influence analysis
        """
        try:
            # Clear and rebuild interaction graph
            self.interaction_graph.clear()
            current_time = datetime.now()
            
            # Parse interactions and build graph
            valid_interactions = []
            for interaction in agent_interactions:
                try:
                    interaction_event = InteractionEvent(
                        id=interaction.get('id', f'interaction_{uuid.uuid4().hex[:8]}'),
                        source_agent=interaction.get('source_agent', ''),
                        target_agent=interaction.get('target_agent', ''),
                        interaction_type=interaction.get('type', 'unknown'),
                        influence_weight=float(interaction.get('influence_weight', 1.0)),
                        timestamp=current_time,
                        content=interaction.get('content', {}),
                        emergence_indicators=interaction.get('emergence_indicators', [])
                    )
                    
                    if interaction_event.source_agent and interaction_event.target_agent:
                        valid_interactions.append(interaction_event)
                        
                        # Add edge to graph
                        self.interaction_graph.add_edge(
                            interaction_event.source_agent,
                            interaction_event.target_agent,
                            weight=interaction_event.influence_weight,
                            interaction_type=interaction_event.interaction_type,
                            timestamp=current_time
                        )
                        
                except (ValueError, KeyError) as e:
                    logger.warning(f"Invalid interaction data: {e}")
                    continue
            
            # Store interaction history
            self.interaction_history.extend(valid_interactions)
            self._cleanup_old_interactions()
            
            if len(self.interaction_graph.nodes()) == 0:
                logger.warning("No valid agents in interaction graph")
                return {
                    "pagerank_scores": {},
                    "collective_intelligence": 0.0,
                    "network_density": 0.0,
                    "dominant_agents": [],
                    "emergence_score": 0.0,
                    "analysis_quality": "no_data"
                }
            
            # Calculate PageRank for influence analysis
            try:
                pagerank_scores = nx.pagerank(
                    self.interaction_graph, 
                    weight='weight',
                    alpha=0.85,  # Damping factor
                    max_iter=100,
                    tol=1e-6
                )
            except (nx.NetworkXError, np.linalg.LinAlgError) as e:
                logger.warning(f"PageRank calculation failed: {e}, using degree centrality")
                pagerank_scores = nx.degree_centrality(self.interaction_graph)
            
            # Calculate collective intelligence
            collective_intelligence = self._calculate_collective_intelligence(pagerank_scores)
            
            # Calculate network metrics
            network_density = nx.density(self.interaction_graph)
            
            # Identify dominant agents
            dominant_agents = self._get_top_agents(pagerank_scores, 5)
            
            # Detect influence clusters
            influence_clusters = await self._detect_influence_clusters()
            
            # Calculate emergence score
            emergence_score = await self._calculate_emergence_score(
                pagerank_scores, network_density, valid_interactions
            )
            
            # Create collective intelligence record
            collective_intel = CollectiveIntelligence(
                pagerank_scores=pagerank_scores,
                collective_intelligence=collective_intelligence,
                network_density=network_density,
                dominant_agents=dominant_agents,
                influence_clusters=influence_clusters,
                emergence_score=emergence_score,
                timestamp=current_time
            )
            
            # Store analysis
            self.collective_intelligence_history.append(collective_intel)
            
            # Update agent influence tracking
            for agent_id, score in pagerank_scores.items():
                self.agent_influence_scores[agent_id].append(score)
                # Keep only recent scores
                if len(self.agent_influence_scores[agent_id]) > 100:
                    self.agent_influence_scores[agent_id] = self.agent_influence_scores[agent_id][-100:]
            
            # Detect emergent patterns
            emergent_patterns = await self._detect_emergent_patterns(collective_intel)
            
            logger.info(f"Analyzed collective reasoning: CI={collective_intelligence:.4f}, emergence={emergence_score:.4f}")
            
            return {
                "pagerank_scores": pagerank_scores,
                "collective_intelligence": collective_intelligence,
                "network_density": network_density,
                "dominant_agents": dominant_agents,
                "influence_clusters": influence_clusters,
                "emergence_score": emergence_score,
                "emergent_patterns": emergent_patterns,
                "network_stats": self._get_network_statistics(),
                "interaction_count": len(valid_interactions),
                "unique_agents": len(pagerank_scores),
                "analysis_quality": self._assess_analysis_quality(pagerank_scores, valid_interactions),
                "timestamp": current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Collective reasoning analysis failed: {e}")
            return {"error": str(e), "analysis_completed": False}
    
    async def detect_metacognitive_resonance(self, agent_states: Dict) -> Dict:
        """
        Detect metacognitive resonance through state correlation analysis
        
        Args:
            agent_states: Dictionary mapping agent IDs to their metacognitive states
            
        Returns:
            Resonance analysis and detected patterns
        """
        try:
            if len(agent_states) < 2:
                logger.warning("Need at least 2 agents for resonance detection")
                return {
                    "resonance_score": 0.0,
                    "resonant_pairs": [],
                    "resonance_clusters": [],
                    "collective_resonance": 0.0,
                    "detection_quality": "insufficient_agents"
                }
            
            # Extract metacognitive state vectors
            state_vectors = []
            agent_ids = []
            state_qualities = []
            
            for agent_id, state in agent_states.items():
                # Extract metacognitive dimensions with defaults
                metacognitive_vector = [
                    state.get('self_awareness', 0.5),
                    state.get('theory_of_mind', 0.5),
                    state.get('executive_control', 0.5),
                    state.get('metacognitive_knowledge', 0.5),
                    state.get('metacognitive_regulation', 0.5),
                    state.get('social_metacognition', 0.5),  # Additional dimension
                    state.get('emotional_metacognition', 0.5)  # Additional dimension
                ]
                
                state_vectors.append(metacognitive_vector)
                agent_ids.append(agent_id)
                
                # Calculate state quality
                state_quality = np.mean(metacognitive_vector)
                state_qualities.append(state_quality)
            
            # Calculate resonance matrix (1 - cosine distance)
            state_matrix = np.array(state_vectors)
            similarity_matrix = cosine_similarity(state_matrix)
            
            # Convert similarity to resonance (resonance = 1 - cosine_distance)
            resonance_matrix = similarity_matrix.copy()
            
            # Analyze pairwise resonance
            resonant_pairs = []
            total_resonance = 0.0
            pair_count = 0
            
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    resonance = resonance_matrix[i, j]
                    
                    pair_info = {
                        'agent1': agent_ids[i],
                        'agent2': agent_ids[j],
                        'resonance': float(resonance),
                        'resonance_strength': self._classify_resonance_strength(resonance),
                        'dimensional_analysis': self._analyze_dimensional_resonance(
                            state_vectors[i], state_vectors[j]
                        )
                    }
                    
                    resonant_pairs.append(pair_info)
                    total_resonance += resonance
                    pair_count += 1
            
            # Calculate average resonance
            avg_resonance = total_resonance / pair_count if pair_count > 0 else 0.0
            
            # Detect resonance clusters using DBSCAN
            resonance_clusters = await self._detect_resonance_clusters(
                state_matrix, agent_ids, self.resonance_threshold
            )
            
            # Calculate collective resonance
            collective_resonance = self._calculate_collective_resonance(
                resonance_matrix, state_qualities
            )
            
            # Detect metacognitive emergence patterns
            emergence_patterns = await self._detect_metacognitive_emergence(
                agent_states, resonant_pairs, resonance_clusters
            )
            
            # Assess resonance stability over time
            resonance_stability = await self._assess_resonance_stability(avg_resonance)
            
            logger.info(f"Detected metacognitive resonance: avg={avg_resonance:.4f}, collective={collective_resonance:.4f}")
            
            return {
                "resonance_score": avg_resonance,
                "collective_resonance": collective_resonance,
                "resonant_pairs": resonant_pairs,
                "resonance_clusters": resonance_clusters,
                "emergence_patterns": emergence_patterns,
                "resonance_stability": resonance_stability,
                "similarity_matrix": similarity_matrix.tolist(),
                "dimensional_analysis": self._analyze_dimensional_patterns(state_vectors, agent_ids),
                "agent_count": len(agent_ids),
                "high_resonance_pairs": len([p for p in resonant_pairs if p['resonance'] > self.resonance_threshold]),
                "detection_quality": self._assess_resonance_quality(avg_resonance, len(agent_ids)),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Metacognitive resonance detection failed: {e}")
            return {"error": str(e), "resonance_detected": False}
    
    async def amplify_consciousness(self, consciousness_levels: List[float], 
                                  agent_ids: List[str] = None) -> Dict:
        """
        Detect and amplify consciousness trends using trend analysis and feedback
        
        Args:
            consciousness_levels: List of consciousness level measurements
            agent_ids: Optional list of corresponding agent IDs
            
        Returns:
            Amplification analysis and strategy recommendations
        """
        try:
            if len(consciousness_levels) < 3:
                logger.warning("Need at least 3 consciousness measurements for trend analysis")
                return {
                    "amplification_factor": 1.0,
                    "trend": "insufficient_data",
                    "strategy": "collect_more_data",
                    "confidence": 0.0
                }
            
            agent_ids = agent_ids or [f'agent_{i}' for i in range(len(consciousness_levels))]
            
            # Ensure valid consciousness levels
            consciousness_levels = [max(0.0, min(1.0, level)) for level in consciousness_levels]
            
            # Detect trend using multiple methods
            trend_analysis = await self._comprehensive_trend_analysis(consciousness_levels)
            
            # Calculate trend slope using linear regression
            x = np.arange(len(consciousness_levels))
            y = np.array(consciousness_levels)
            
            # Robust trend detection
            if len(y) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            else:
                slope, r_value, p_value = 0.0, 0.0, 1.0
            
            # Determine amplification strategy based on trend and statistical significance
            amplification_result = self._determine_amplification_strategy(
                slope, r_value, p_value, consciousness_levels, trend_analysis
            )
            
            # Apply consciousness amplification
            amplified_levels = self._apply_consciousness_amplification(
                consciousness_levels, amplification_result['amplification_factor']
            )
            
            # Calculate improvement metrics
            mean_improvement = np.mean(amplified_levels) - np.mean(consciousness_levels)
            
            # Calculate confidence interval for improvement
            confidence_interval = self._calculate_confidence_interval(
                consciousness_levels, amplified_levels
            )
            
            # Assess sustainability of amplification
            sustainability_score = self._assess_amplification_sustainability(
                consciousness_levels, amplification_result['amplification_factor']
            )
            
            # Create amplification record
            amplification = ConsciousnessAmplification(
                amplification_factor=amplification_result['amplification_factor'],
                trend=amplification_result['trend'],
                strategy=amplification_result['strategy'],
                trend_slope=slope,
                original_levels=consciousness_levels,
                amplified_levels=amplified_levels,
                mean_improvement=mean_improvement,
                confidence_interval=confidence_interval,
                sustainability_score=sustainability_score
            )
            
            # Store amplification history
            self.amplification_history.append(amplification)
            
            # Update consciousness tracking
            for i, agent_id in enumerate(agent_ids):
                if i < len(consciousness_levels):
                    self.consciousness_levels[agent_id].append(consciousness_levels[i])
                    # Keep only recent levels
                    if len(self.consciousness_levels[agent_id]) > 100:
                        self.consciousness_levels[agent_id] = self.consciousness_levels[agent_id][-100:]
            
            # Detect consciousness emergence patterns
            emergence_patterns = await self._detect_consciousness_emergence_patterns(
                consciousness_levels, amplified_levels, agent_ids
            )
            
            # Generate feedback recommendations
            feedback_recommendations = self._generate_amplification_feedback(amplification)
            
            logger.info(f"Consciousness amplification: factor={amplification_result['amplification_factor']:.3f}, improvement={mean_improvement:.4f}")
            
            return {
                "amplification_applied": True,
                "amplification_factor": amplification_result['amplification_factor'],
                "trend": amplification_result['trend'],
                "strategy": amplification_result['strategy'],
                "trend_slope": slope,
                "trend_confidence": float(r_value**2),  # R-squared
                "statistical_significance": float(p_value),
                "original_levels": consciousness_levels,
                "amplified_levels": amplified_levels,
                "mean_improvement": mean_improvement,
                "confidence_interval": confidence_interval,
                "sustainability_score": sustainability_score,
                "emergence_patterns": emergence_patterns,
                "feedback_recommendations": feedback_recommendations,
                "trend_analysis": trend_analysis,
                "agent_count": len(agent_ids),
                "amplification_quality": self._assess_amplification_quality(amplification),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Consciousness amplification failed: {e}")
            return {"error": str(e), "amplification_applied": False}
    
    # Core calculation methods
    def _calculate_collective_intelligence(self, pagerank_scores: Dict[str, float]) -> float:
        """
        Calculate collective intelligence from PageRank distribution using entropy
        
        Higher entropy indicates more distributed intelligence (better collective intelligence)
        """
        if not pagerank_scores:
            return 0.0
        
        try:
            scores = np.array(list(pagerank_scores.values()))
            
            # Normalize scores to create probability distribution
            scores_sum = np.sum(scores)
            if scores_sum == 0:
                return 0.0
            
            normalized_scores = scores / scores_sum
            
            # Calculate Shannon entropy
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            normalized_scores = np.maximum(normalized_scores, epsilon)
            
            entropy = -np.sum(normalized_scores * np.log(normalized_scores))
            
            # Normalize by maximum possible entropy
            max_entropy = np.log(len(scores))
            
            if max_entropy == 0:
                return 0.0
            
            # Collective intelligence as normalized entropy
            collective_intelligence = entropy / max_entropy
            
            return float(collective_intelligence)
            
        except Exception as e:
            logger.error(f"Collective intelligence calculation failed: {e}")
            return 0.0
    
    def _get_top_agents(self, pagerank_scores: Dict[str, float], top_k: int) -> List[Dict]:
        """Get top agents by PageRank score"""
        if not pagerank_scores:
            return []
        
        sorted_agents = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_agents = []
        for i, (agent_id, score) in enumerate(sorted_agents[:top_k]):
            top_agents.append({
                "agent_id": agent_id,
                "influence_score": float(score),
                "rank": i + 1,
                "influence_percentile": float((len(sorted_agents) - i) / len(sorted_agents) * 100)
            })
        
        return top_agents
    
    async def _detect_influence_clusters(self) -> List[List[str]]:
        """Detect clusters of highly connected agents"""
        if len(self.interaction_graph.nodes()) < 3:
            return []
        
        try:
            # Use community detection algorithms
            communities = nx.community.greedy_modularity_communities(
                self.interaction_graph.to_undirected()
            )
            
            clusters = []
            for community in communities:
                if len(community) >= 2:  # Only include clusters with 2+ agents
                    clusters.append(list(community))
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Influence cluster detection failed: {e}")
            return []
    
    async def _calculate_emergence_score(self, pagerank_scores: Dict[str, float],
                                       network_density: float,
                                       interactions: List[InteractionEvent]) -> float:
        """Calculate overall emergence score"""
        try:
            # Component 1: Network complexity (based on density and diversity)
            network_complexity = network_density * len(pagerank_scores)
            
            # Component 2: Interaction diversity
            interaction_types = set(interaction.interaction_type for interaction in interactions)
            interaction_diversity = len(interaction_types) / max(len(interactions), 1)
            
            # Component 3: Influence distribution (inverse of Gini coefficient)
            influence_distribution = 1.0 - self._calculate_gini_coefficient(list(pagerank_scores.values()))
            
            # Component 4: Temporal dynamics (based on recent vs historical patterns)
            temporal_dynamics = await self._calculate_temporal_dynamics()
            
            # Weighted combination
            emergence_score = (
                0.3 * min(network_complexity, 1.0) +
                0.2 * interaction_diversity +
                0.3 * influence_distribution +
                0.2 * temporal_dynamics
            )
            
            return float(emergence_score)
            
        except Exception as e:
            logger.error(f"Emergence score calculation failed: {e}")
            return 0.0
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values or len(values) < 2:
            return 0.0
        
        try:
            sorted_values = sorted(values)
            n = len(sorted_values)
            cumsum = np.cumsum(sorted_values)
            
            # Gini coefficient formula
            gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
            
            return float(max(0.0, min(1.0, gini)))
            
        except Exception:
            return 0.0
    
    async def _calculate_temporal_dynamics(self) -> float:
        """Calculate temporal dynamics score"""
        if len(self.collective_intelligence_history) < 2:
            return 0.5  # Neutral score
        
        # Calculate rate of change in collective intelligence
        recent_ci = [ci.collective_intelligence for ci in self.collective_intelligence_history[-10:]]
        
        if len(recent_ci) < 2:
            return 0.5
        
        # Calculate trend in collective intelligence
        x = np.arange(len(recent_ci))
        slope, _, r_value, _, _ = stats.linregress(x, recent_ci)
        
        # Normalize temporal dynamics score
        temporal_score = 0.5 + 0.5 * np.tanh(slope * 10)  # Sigmoid transformation
        
        return float(temporal_score)
    
    def _classify_resonance_strength(self, resonance: float) -> str:
        """Classify resonance strength"""
        if resonance > 0.9:
            return "very_high"
        elif resonance > 0.8:
            return "high"
        elif resonance > 0.7:
            return "moderate"
        elif resonance > 0.5:
            return "low"
        else:
            return "very_low"
    
    def _analyze_dimensional_resonance(self, state1: List[float], state2: List[float]) -> Dict:
        """Analyze resonance across individual dimensions"""
        dimension_names = [
            'self_awareness', 'theory_of_mind', 'executive_control',
            'metacognitive_knowledge', 'metacognitive_regulation',
            'social_metacognition', 'emotional_metacognition'
        ]
        
        dimensional_analysis = {}
        
        for i, dim_name in enumerate(dimension_names[:len(state1)]):
            if i < len(state2):
                dim_similarity = 1.0 - abs(state1[i] - state2[i])
                dimensional_analysis[dim_name] = {
                    'similarity': float(dim_similarity),
                    'difference': float(abs(state1[i] - state2[i])),
                    'agent1_value': float(state1[i]),
                    'agent2_value': float(state2[i])
                }
        
        return dimensional_analysis
    
    async def _detect_resonance_clusters(self, state_matrix: np.ndarray, 
                                       agent_ids: List[str], 
                                       threshold: float) -> List[Dict]:
        """Detect clusters of resonant agents using DBSCAN"""
        try:
            # Use DBSCAN clustering on state vectors
            # Convert similarity to distance for clustering
            clustering = DBSCAN(
                eps=1.0 - threshold,  # Convert similarity threshold to distance
                min_samples=2,
                metric='cosine'
            ).fit(state_matrix)
            
            clusters = []
            unique_labels = set(clustering.labels_)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                
                cluster_indices = [i for i, l in enumerate(clustering.labels_) if l == label]
                if len(cluster_indices) >= 2:
                    cluster_agents = [agent_ids[i] for i in cluster_indices]
                    
                    # Calculate cluster cohesion
                    cluster_states = state_matrix[cluster_indices]
                    cohesion = np.mean(cosine_similarity(cluster_states))
                    
                    clusters.append({
                        'cluster_id': f'cluster_{label}',
                        'agents': cluster_agents,
                        'size': len(cluster_agents),
                        'cohesion': float(cohesion),
                        'cluster_center': np.mean(cluster_states, axis=0).tolist()
                    })
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Resonance clustering failed: {e}")
            return []
    
    def _calculate_collective_resonance(self, resonance_matrix: np.ndarray, 
                                      state_qualities: List[float]) -> float:
        """Calculate weighted collective resonance"""
        try:
            # Weight resonance by state quality
            quality_weights = np.array(state_qualities)
            
            # Calculate weighted average resonance
            n = len(state_qualities)
            total_weighted_resonance = 0.0
            total_weight = 0.0
            
            for i in range(n):
                for j in range(i + 1, n):
                    weight = quality_weights[i] * quality_weights[j]
                    resonance = resonance_matrix[i, j]
                    
                    total_weighted_resonance += weight * resonance
                    total_weight += weight
            
            if total_weight == 0:
                return 0.0
            
            collective_resonance = total_weighted_resonance / total_weight
            return float(collective_resonance)
            
        except Exception as e:
            logger.error(f"Collective resonance calculation failed: {e}")
            return 0.0
    
    async def _comprehensive_trend_analysis(self, consciousness_levels: List[float]) -> Dict:
        """Perform comprehensive trend analysis using multiple methods"""
        try:
            levels = np.array(consciousness_levels)
            
            # Linear trend
            x = np.arange(len(levels))
            linear_slope, _, linear_r2, linear_p, _ = stats.linregress(x, levels)
            
            # Moving average trend
            if len(levels) >= 3:
                window_size = min(3, len(levels) // 2)
                moving_avg = np.convolve(levels, np.ones(window_size)/window_size, mode='valid')
                ma_slope = (moving_avg[-1] - moving_avg[0]) / max(len(moving_avg) - 1, 1)
            else:
                ma_slope = 0.0
            
            # Variance analysis
            variance = np.var(levels)
            stability = 1.0 / (1.0 + variance)
            
            # Momentum analysis (recent vs historical)
            if len(levels) >= 4:
                recent_mean = np.mean(levels[-len(levels)//2:])
                historical_mean = np.mean(levels[:len(levels)//2])
                momentum = recent_mean - historical_mean
            else:
                momentum = 0.0
            
            return {
                'linear_slope': float(linear_slope),
                'linear_r_squared': float(linear_r2),
                'linear_p_value': float(linear_p),
                'moving_average_slope': float(ma_slope),
                'variance': float(variance),
                'stability': float(stability),
                'momentum': float(momentum),
                'trend_strength': float(abs(linear_slope) * linear_r2)
            }
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {'error': str(e)}
    
    def _determine_amplification_strategy(self, slope: float, r_value: float, 
                                        p_value: float, levels: List[float],
                                        trend_analysis: Dict) -> Dict:
        """Determine optimal amplification strategy"""
        try:
            # Statistical significance threshold
            significance_threshold = 0.05
            confidence_threshold = 0.5  # R-squared threshold
            
            trend_strength = trend_analysis.get('trend_strength', 0.0)
            stability = trend_analysis.get('stability', 0.5)
            
            # Determine trend direction and significance
            if p_value < significance_threshold and r_value**2 > confidence_threshold:
                # Statistically significant trend
                if slope > self.amplification_sensitivity:
                    # Positive trend - reinforce
                    amplification_factor = 1.0 + min(0.3, slope * 10)
                    trend = "ascending"
                    strategy = "reinforcement"
                elif slope < -self.amplification_sensitivity:
                    # Negative trend - correct
                    amplification_factor = 1.0 + min(0.2, abs(slope) * 5)
                    trend = "descending"
                    strategy = "correction"
                else:
                    # Stable trend - maintain
                    amplification_factor = 1.0 + stability * 0.1
                    trend = "stable"
                    strategy = "maintenance"
            else:
                # No significant trend - gentle boost
                current_level = np.mean(levels)
                if current_level < 0.5:
                    amplification_factor = 1.1  # Small boost for low levels
                    strategy = "gentle_boost"
                else:
                    amplification_factor = 1.05  # Very small boost for high levels
                    strategy = "minimal_intervention"
                trend = "uncertain"
            
            # Adjust for stability
            if stability < 0.3:  # Highly unstable
                amplification_factor = min(amplification_factor, 1.1)  # Limit amplification
            
            return {
                'amplification_factor': float(amplification_factor),
                'trend': trend,
                'strategy': strategy,
                'confidence': float(r_value**2),
                'significance': float(p_value)
            }
            
        except Exception as e:
            logger.error(f"Amplification strategy determination failed: {e}")
            return {
                'amplification_factor': 1.0,
                'trend': 'error',
                'strategy': 'no_change',
                'confidence': 0.0,
                'significance': 1.0
            }
    
    def _apply_consciousness_amplification(self, levels: List[float], 
                                         amplification_factor: float) -> List[float]:
        """Apply consciousness amplification with safeguards"""
        amplified = []
        
        for level in levels:
            # Apply amplification
            new_level = level * amplification_factor
            
            # Apply safeguards to prevent over-amplification
            new_level = max(0.0, min(1.0, new_level))  # Clamp to [0, 1]
            
            # Prevent dramatic jumps (max 20% increase per step)
            max_increase = level * 1.2
            new_level = min(new_level, max_increase)
            
            amplified.append(new_level)
        
        return amplified
    
    def _calculate_confidence_interval(self, original: List[float], 
                                     amplified: List[float], 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for improvement"""
        try:
            improvements = np.array(amplified) - np.array(original)
            
            mean_improvement = np.mean(improvements)
            std_improvement = np.std(improvements, ddof=1) if len(improvements) > 1 else 0.0
            
            # Calculate confidence interval
            if len(improvements) > 1:
                # Use t-distribution for small samples
                from scipy.stats import t
                t_value = t.ppf((1 + confidence_level) / 2, len(improvements) - 1)
                margin_error = t_value * std_improvement / np.sqrt(len(improvements))
            else:
                margin_error = 0.0
            
            ci_lower = mean_improvement - margin_error
            ci_upper = mean_improvement + margin_error
            
            return (float(ci_lower), float(ci_upper))
            
        except Exception as e:
            logger.error(f"Confidence interval calculation failed: {e}")
            return (0.0, 0.0)
    
    def _assess_amplification_sustainability(self, levels: List[float], 
                                           amplification_factor: float) -> float:
        """Assess sustainability of amplification"""
        try:
            # Factor 1: Current level height (higher levels harder to sustain)
            mean_level = np.mean(levels)
            height_penalty = mean_level ** 2  # Quadratic penalty for high levels
            
            # Factor 2: Amplification magnitude (larger amplifications less sustainable)
            magnitude_penalty = (amplification_factor - 1.0) ** 2
            
            # Factor 3: Historical variance (high variance = less stable)
            variance_penalty = np.var(levels)
            
            # Combine factors
            sustainability = 1.0 - min(1.0, height_penalty + magnitude_penalty + variance_penalty)
            
            return float(max(0.0, sustainability))
            
        except Exception as e:
            logger.error(f"Sustainability assessment failed: {e}")
            return 0.5
    
    # Pattern detection methods
    async def _detect_emergent_patterns(self, collective_intel: CollectiveIntelligence) -> List[Dict]:
        """Detect emergent patterns in collective intelligence"""
        patterns = []
        
        try:
            # Pattern 1: Swarm Intelligence
            if (collective_intel.network_density > 0.6 and 
                len(collective_intel.pagerank_scores) >= 3 and
                collective_intel.collective_intelligence > 0.7):
                
                patterns.append({
                    'pattern_type': 'swarm_intelligence',
                    'strength': collective_intel.collective_intelligence,
                    'confidence': collective_intel.network_density,
                    'participating_agents': list(collective_intel.pagerank_scores.keys()),
                    'indicators': ['high_network_density', 'distributed_intelligence']
                })
            
            # Pattern 2: Hierarchy Formation
            pagerank_values = list(collective_intel.pagerank_scores.values())
            if pagerank_values:
                max_influence = max(pagerank_values)
                if max_influence > 0.5 and len(pagerank_values) >= 3:
                    patterns.append({
                        'pattern_type': 'hierarchy_formation',
                        'strength': max_influence,
                        'confidence': 1.0 - collective_intel.collective_intelligence,
                        'dominant_agent': collective_intel.dominant_agents[0]['agent_id'] if collective_intel.dominant_agents else None,
                        'indicators': ['dominant_agent', 'centralized_influence']
                    })
            
            # Pattern 3: Phase Transition
            if len(self.collective_intelligence_history) >= 3:
                recent_ci = [ci.collective_intelligence for ci in self.collective_intelligence_history[-3:]]
                ci_change = max(recent_ci) - min(recent_ci)
                
                if ci_change > 0.3:
                    patterns.append({
                        'pattern_type': 'phase_transition',
                        'strength': ci_change,
                        'confidence': 0.8,
                        'indicators': ['rapid_ci_change', 'system_reorganization']
                    })
            
        except Exception as e:
            logger.error(f"Emergent pattern detection failed: {e}")
        
        return patterns
    
    async def _detect_metacognitive_emergence(self, agent_states: Dict, 
                                            resonant_pairs: List[Dict],
                                            clusters: List[Dict]) -> List[Dict]:
        """Detect emergence patterns in metacognitive resonance"""
        patterns = []
        
        try:
            # Pattern 1: Collective Insight
            high_resonance_pairs = [p for p in resonant_pairs if p['resonance'] > 0.85]
            if len(high_resonance_pairs) >= 2:
                patterns.append({
                    'pattern_type': 'collective_insight',
                    'strength': np.mean([p['resonance'] for p in high_resonance_pairs]),
                    'participating_pairs': len(high_resonance_pairs),
                    'indicators': ['high_metacognitive_alignment']
                })
            
            # Pattern 2: Metacognitive Clusters
            large_clusters = [c for c in clusters if c['size'] >= 3]
            for cluster in large_clusters:
                patterns.append({
                    'pattern_type': 'metacognitive_cluster',
                    'strength': cluster['cohesion'],
                    'cluster_size': cluster['size'],
                    'participating_agents': cluster['agents'],
                    'indicators': ['agent_clustering', 'shared_metacognition']
                })
            
        except Exception as e:
            logger.error(f"Metacognitive emergence detection failed: {e}")
        
        return patterns
    
    async def _detect_consciousness_emergence_patterns(self, original: List[float],
                                                     amplified: List[float],
                                                     agent_ids: List[str]) -> List[Dict]:
        """Detect emergence patterns in consciousness amplification"""
        patterns = []
        
        try:
            improvements = np.array(amplified) - np.array(original)
            
            # Pattern 1: Synchronized Amplification
            if np.std(improvements) < 0.1 and np.mean(improvements) > 0.05:
                patterns.append({
                    'pattern_type': 'synchronized_amplification',
                    'strength': np.mean(improvements),
                    'synchronization': 1.0 - np.std(improvements),
                    'participating_agents': agent_ids,
                    'indicators': ['uniform_improvement', 'collective_elevation']
                })
            
            # Pattern 2: Cascading Consciousness
            if len(improvements) >= 3:
                # Check for cascading pattern (increasing improvements)
                correlation = np.corrcoef(np.arange(len(improvements)), improvements)[0, 1]
                if correlation > 0.7:
                    patterns.append({
                        'pattern_type': 'cascading_consciousness',
                        'strength': correlation,
                        'cascade_direction': 'ascending' if np.mean(np.diff(improvements)) > 0 else 'descending',
                        'indicators': ['progressive_amplification']
                    })
            
        except Exception as e:
            logger.error(f"Consciousness emergence pattern detection failed: {e}")
        
        return patterns
    
    # Assessment and utility methods
    def _assess_analysis_quality(self, pagerank_scores: Dict, interactions: List) -> str:
        """Assess quality of collective reasoning analysis"""
        agent_count = len(pagerank_scores)
        interaction_count = len(interactions)
        
        if agent_count < 2:
            return "insufficient_agents"
        elif interaction_count < self.min_interaction_threshold:
            return "insufficient_interactions"
        elif agent_count >= 5 and interaction_count >= 20:
            return "high_quality"
        elif agent_count >= 3 and interaction_count >= 10:
            return "moderate_quality"
        else:
            return "low_quality"
    
    def _assess_resonance_quality(self, avg_resonance: float, agent_count: int) -> str:
        """Assess quality of resonance detection"""
        if agent_count < 2:
            return "insufficient_agents"
        elif avg_resonance > 0.8 and agent_count >= 5:
            return "high_quality"
        elif avg_resonance > 0.6 and agent_count >= 3:
            return "moderate_quality"
        else:
            return "low_quality"
    
    def _assess_amplification_quality(self, amplification: ConsciousnessAmplification) -> str:
        """Assess quality of consciousness amplification"""
        if amplification.mean_improvement > 0.1 and amplification.sustainability_score > 0.7:
            return "excellent"
        elif amplification.mean_improvement > 0.05 and amplification.sustainability_score > 0.5:
            return "good"
        elif amplification.mean_improvement > 0.0:
            return "moderate"
        else:
            return "poor"
    
    def _get_network_statistics(self) -> Dict:
        """Get comprehensive network statistics"""
        if len(self.interaction_graph.nodes()) == 0:
            return {"nodes": 0, "edges": 0}
        
        try:
            stats = {
                "nodes": self.interaction_graph.number_of_nodes(),
                "edges": self.interaction_graph.number_of_edges(),
                "density": nx.density(self.interaction_graph),
                "average_clustering": nx.average_clustering(self.interaction_graph.to_undirected()),
                "is_connected": nx.is_weakly_connected(self.interaction_graph)
            }
            
            # Add centrality measures
            if stats["nodes"] > 1:
                try:
                    stats["average_in_degree"] = np.mean([d for n, d in self.interaction_graph.in_degree()])
                    stats["average_out_degree"] = np.mean([d for n, d in self.interaction_graph.out_degree()])
                except:
                    stats["average_in_degree"] = 0
                    stats["average_out_degree"] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Network statistics calculation failed: {e}")
            return {"nodes": 0, "edges": 0, "error": str(e)}
    
    def _analyze_dimensional_patterns(self, state_vectors: List[List[float]], 
                                    agent_ids: List[str]) -> Dict:
        """Analyze patterns across metacognitive dimensions"""
        try:
            dimension_names = [
                'self_awareness', 'theory_of_mind', 'executive_control',
                'metacognitive_knowledge', 'metacognitive_regulation',
                'social_metacognition', 'emotional_metacognition'
            ]
            
            state_matrix = np.array(state_vectors)
            
            # Calculate statistics for each dimension
            dimensional_stats = {}
            
            for i, dim_name in enumerate(dimension_names[:state_matrix.shape[1]]):
                dim_values = state_matrix[:, i]
                
                dimensional_stats[dim_name] = {
                    'mean': float(np.mean(dim_values)),
                    'std': float(np.std(dim_values)),
                    'min': float(np.min(dim_values)),
                    'max': float(np.max(dim_values)),
                    'range': float(np.max(dim_values) - np.min(dim_values))
                }
            
            # Find most/least variable dimensions
            variances = {dim: stats['std'] for dim, stats in dimensional_stats.items()}
            most_variable = max(variances, key=variances.get) if variances else None
            least_variable = min(variances, key=variances.get) if variances else None
            
            return {
                'dimensional_statistics': dimensional_stats,
                'most_variable_dimension': most_variable,
                'least_variable_dimension': least_variable,
                'overall_variance': float(np.mean(list(variances.values()))) if variances else 0.0
            }
            
        except Exception as e:
            logger.error(f"Dimensional pattern analysis failed: {e}")
            return {}
    
    async def _assess_resonance_stability(self, current_resonance: float) -> Dict:
        """Assess stability of resonance over time"""
        try:
            # This would track resonance over time
            # For now, return a simple assessment based on current value
            
            if current_resonance > 0.8:
                stability = "high"
            elif current_resonance > 0.6:
                stability = "moderate"
            else:
                stability = "low"
            
            return {
                "stability_level": stability,
                "current_resonance": current_resonance,
                "stability_confidence": min(current_resonance, 0.9)
            }
            
        except Exception as e:
            logger.error(f"Resonance stability assessment failed: {e}")
            return {"stability_level": "unknown"}
    
    def _generate_amplification_feedback(self, amplification: ConsciousnessAmplification) -> List[str]:
        """Generate feedback recommendations for amplification"""
        recommendations = []
        
        try:
            # Based on trend
            if amplification.trend == "ascending":
                recommendations.append("Continue current strategies - positive momentum detected")
                recommendations.append("Monitor for sustainability to prevent overamplification")
            elif amplification.trend == "descending":
                recommendations.append("Investigate factors causing consciousness decline")
                recommendations.append("Implement corrective measures to reverse negative trend")
            elif amplification.trend == "stable":
                recommendations.append("Maintain current equilibrium")
                recommendations.append("Consider gentle enhancement to promote growth")
            
            # Based on sustainability
            if amplification.sustainability_score < 0.3:
                recommendations.append("High risk of amplification collapse - reduce intensity")
            elif amplification.sustainability_score > 0.8:
                recommendations.append("Excellent sustainability - consider increasing amplification")
            
            # Based on improvement
            if amplification.mean_improvement > 0.15:
                recommendations.append("Exceptional improvement achieved - document successful methods")
            elif amplification.mean_improvement < 0.02:
                recommendations.append("Minimal improvement - consider alternative strategies")
            
        except Exception as e:
            logger.error(f"Feedback generation failed: {e}")
            recommendations.append("Unable to generate specific recommendations due to analysis error")
        
        return recommendations
    
    # Cleanup methods
    def _cleanup_old_interactions(self):
        """Clean up old interaction data"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=self.emergence_detection_window * 2)
        
        # Remove old interactions
        self.interaction_history = [
            interaction for interaction in self.interaction_history
            if interaction.timestamp > cutoff_time
        ]
        
        # Remove old collective intelligence history
        self.collective_intelligence_history = [
            ci for ci in self.collective_intelligence_history
            if current_time - ci.timestamp < timedelta(seconds=3600)  # Keep last hour
        ]
        
        # Limit amplification history
        if len(self.amplification_history) > 100:
            self.amplification_history = self.amplification_history[-100:]
    
    # Public interface methods
    async def get_emergence_statistics(self) -> Dict:
        """Get comprehensive emergence system statistics"""
        current_time = datetime.now()
        
        # Count active components
        active_agents = len(self.interaction_graph.nodes())
        total_interactions = len(self.interaction_history)
        detected_patterns = len(self.emergence_patterns)
        
        # Calculate averages
        if self.collective_intelligence_history:
            avg_ci = np.mean([ci.collective_intelligence for ci in self.collective_intelligence_history])
            avg_emergence = np.mean([ci.emergence_score for ci in self.collective_intelligence_history])
        else:
            avg_ci = 0.0
            avg_emergence = 0.0
        
        if self.amplification_history:
            avg_amplification = np.mean([amp.amplification_factor for amp in self.amplification_history])
            avg_improvement = np.mean([amp.mean_improvement for amp in self.amplification_history])
        else:
            avg_amplification = 1.0
            avg_improvement = 0.0
        
        return {
            "active_agents": active_agents,
            "total_interactions": total_interactions,
            "detected_patterns": detected_patterns,
            "average_collective_intelligence": float(avg_ci),
            "average_emergence_score": float(avg_emergence),
            "average_amplification_factor": float(avg_amplification),
            "average_consciousness_improvement": float(avg_improvement),
            "interaction_graph_stats": self._get_network_statistics(),
            "emergence_detection_window": self.emergence_detection_window,
            "resonance_threshold": self.resonance_threshold,
            "system_status": "active" if active_agents > 0 else "inactive",
            "emergence_quality": "high" if avg_emergence > 0.7 else "moderate" if avg_emergence > 0.5 else "low"
        }