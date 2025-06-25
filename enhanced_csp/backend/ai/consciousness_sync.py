# backend/ai/consciousness_sync.py
"""
Multi-Dimensional Consciousness Synchronization
===============================================
Implements consciousness synchronization using SVD alignment with:
- 128-dimensional attention focus vectors (normalized)
- 64-dimensional emotional state vectors (tanh-bounded) 
- 256-dimensional intention vectors (normalized)
- 5 metacognitive dimensions
- SVD-based attention manifold alignment (70% individual / 30% collective)
- L2 distance emotional resonance with 30th percentile coupling
"""

import numpy as np
from scipy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessState:
    """Individual agent consciousness state"""
    agent_id: str
    attention_vector: np.ndarray  # 128-dim, normalized
    emotion_vector: np.ndarray    # 64-dim, tanh-bounded
    intention_vector: np.ndarray  # 256-dim, normalized
    metacognitive_state: Dict[str, float]  # 5 dimensions
    timestamp: datetime
    confidence: float = 1.0

@dataclass
class SynchronizationResult:
    """Results of consciousness synchronization"""
    consciousness_coherence: float
    emotional_coupling: float
    metacognitive_alignment: float
    attention_alignment: float
    individual_collective_ratio: Tuple[float, float]
    overall_sync_score: float
    agent_count: int
    synchronization_matrix: np.ndarray
    timestamp: datetime

class ConsciousnessSynchronizer:
    """
    Advanced consciousness synchronization engine implementing the mathematical
    model described in the system specification.
    """
    
    def __init__(self):
        self.attention_dim = 128
        self.emotion_dim = 64
        self.intention_dim = 256
        self.metacognitive_dimensions = {
            'self_awareness': 0,
            'theory_of_mind': 1, 
            'executive_control': 2,
            'metacognitive_knowledge': 3,
            'metacognitive_regulation': 4
        }
        
        # Synchronization parameters
        self.individual_ratio = 0.7  # 70% individual
        self.collective_ratio = 0.3  # 30% collective
        self.emotional_coupling_percentile = 30  # 30th percentile threshold
        
        # Performance tracking
        self.sync_history: List[SynchronizationResult] = []
        self.performance_cache = {}
        
    async def synchronize_agents(self, agents_data: List[Dict]) -> Dict[str, float]:
        """
        Main synchronization function that coordinates consciousness across agents
        
        Args:
            agents_data: List of agent consciousness data dictionaries
            
        Returns:
            Comprehensive synchronization metrics
        """
        try:
            # Parse and validate agent states
            consciousness_states = await self._parse_agent_states(agents_data)
            
            if len(consciousness_states) < 2:
                logger.warning("Need at least 2 agents for synchronization")
                return self._empty_sync_result()
            
            # Extract vector matrices
            attention_matrix = self._extract_attention_matrix(consciousness_states)
            emotion_matrix = self._extract_emotion_matrix(consciousness_states)
            intention_matrix = self._extract_intention_matrix(consciousness_states)
            metacognitive_matrix = self._extract_metacognitive_matrix(consciousness_states)
            
            # Perform SVD-based attention manifold alignment
            attention_alignment = await self._svd_attention_alignment(attention_matrix)
            
            # Calculate consciousness coherence
            consciousness_coherence = self._calculate_consciousness_coherence(
                attention_matrix, intention_matrix
            )
            
            # Calculate emotional resonance with L2 distance
            emotional_coupling = self._calculate_emotional_coupling(emotion_matrix)
            
            # Calculate metacognitive alignment
            metacognitive_alignment = self._calculate_metacognitive_alignment(metacognitive_matrix)
            
            # Apply individual/collective ratio weighting
            weighted_coherence = self._apply_individual_collective_weighting(
                consciousness_coherence, attention_alignment
            )
            
            # Calculate overall synchronization score
            overall_sync_score = self._calculate_overall_sync_score(
                weighted_coherence, emotional_coupling, metacognitive_alignment
            )
            
            # Create comprehensive result
            result = SynchronizationResult(
                consciousness_coherence=consciousness_coherence,
                emotional_coupling=emotional_coupling,
                metacognitive_alignment=metacognitive_alignment,
                attention_alignment=attention_alignment,
                individual_collective_ratio=(self.individual_ratio, self.collective_ratio),
                overall_sync_score=overall_sync_score,
                agent_count=len(consciousness_states),
                synchronization_matrix=self._create_synchronization_matrix(
                    attention_matrix, emotion_matrix, intention_matrix
                ),
                timestamp=datetime.now()
            )
            
            # Store in history
            self.sync_history.append(result)
            
            # Return formatted results
            return self._format_sync_results(result)
            
        except Exception as e:
            logger.error(f"Consciousness synchronization failed: {e}")
            return self._error_sync_result(str(e))
    
    async def _parse_agent_states(self, agents_data: List[Dict]) -> List[ConsciousnessState]:
        """Parse and validate agent consciousness states"""
        consciousness_states = []
        
        for i, agent_data in enumerate(agents_data):
            try:
                # Extract or generate attention vector (128-dim)
                attention_raw = agent_data.get('attention', np.random.randn(self.attention_dim))
                attention_vector = self._normalize_vector(
                    np.array(attention_raw)[:self.attention_dim]
                )
                
                # Extract or generate emotion vector (64-dim, tanh-bounded)
                emotion_raw = agent_data.get('emotion', np.random.randn(self.emotion_dim))
                emotion_vector = self._bound_emotions(
                    np.array(emotion_raw)[:self.emotion_dim]
                )
                
                # Extract or generate intention vector (256-dim)
                intention_raw = agent_data.get('intention', np.random.randn(self.intention_dim))
                intention_vector = self._normalize_vector(
                    np.array(intention_raw)[:self.intention_dim]
                )
                
                # Extract metacognitive state
                metacognitive_state = {
                    'self_awareness': agent_data.get('self_awareness', 0.5),
                    'theory_of_mind': agent_data.get('theory_of_mind', 0.5),
                    'executive_control': agent_data.get('executive_control', 0.5),
                    'metacognitive_knowledge': agent_data.get('metacognitive_knowledge', 0.5),
                    'metacognitive_regulation': agent_data.get('metacognitive_regulation', 0.5)
                }
                
                # Create consciousness state
                state = ConsciousnessState(
                    agent_id=agent_data.get('agent_id', f'agent_{i}'),
                    attention_vector=attention_vector,
                    emotion_vector=emotion_vector,
                    intention_vector=intention_vector,
                    metacognitive_state=metacognitive_state,
                    timestamp=datetime.now(),
                    confidence=agent_data.get('confidence', 1.0)
                )
                
                consciousness_states.append(state)
                
            except Exception as e:
                logger.warning(f"Failed to parse agent {i} state: {e}")
                continue
        
        return consciousness_states
    
    def _extract_attention_matrix(self, states: List[ConsciousnessState]) -> np.ndarray:
        """Extract attention vectors into matrix"""
        return np.array([state.attention_vector for state in states])
    
    def _extract_emotion_matrix(self, states: List[ConsciousnessState]) -> np.ndarray:
        """Extract emotion vectors into matrix"""
        return np.array([state.emotion_vector for state in states])
    
    def _extract_intention_matrix(self, states: List[ConsciousnessState]) -> np.ndarray:
        """Extract intention vectors into matrix"""
        return np.array([state.intention_vector for state in states])
    
    def _extract_metacognitive_matrix(self, states: List[ConsciousnessState]) -> np.ndarray:
        """Extract metacognitive states into matrix"""
        metacognitive_vectors = []
        for state in states:
            vector = [
                state.metacognitive_state['self_awareness'],
                state.metacognitive_state['theory_of_mind'],
                state.metacognitive_state['executive_control'],
                state.metacognitive_state['metacognitive_knowledge'],
                state.metacognitive_state['metacognitive_regulation']
            ]
            metacognitive_vectors.append(vector)
        return np.array(metacognitive_vectors)
    
    async def _svd_attention_alignment(self, attention_matrix: np.ndarray) -> float:
        """
        Perform SVD-based attention manifold alignment
        
        Uses Singular Value Decomposition to find the optimal alignment
        of attention vectors in the manifold space.
        """
        try:
            # Perform SVD decomposition
            U, S, Vt = svd(attention_matrix, full_matrices=False)
            
            # Calculate alignment quality based on singular value distribution
            # Higher concentration in top singular values = better alignment
            total_variance = np.sum(S**2)
            
            if total_variance == 0:
                return 0.0
            
            # Calculate cumulative variance explained by top components
            cumulative_variance = np.cumsum(S**2) / total_variance
            
            # Alignment score based on how much variance is captured by top components
            # Target: 80% of variance in first 20% of components
            n_components = len(S)
            top_20_percent = max(1, int(0.2 * n_components))
            alignment_score = cumulative_variance[top_20_percent - 1]
            
            # Apply manifold alignment transformation
            # Project attention vectors onto aligned manifold
            aligned_attention = U @ np.diag(S) @ Vt
            
            # Calculate reconstruction fidelity
            reconstruction_error = np.linalg.norm(attention_matrix - aligned_attention, 'fro')
            max_possible_error = np.linalg.norm(attention_matrix, 'fro')
            
            if max_possible_error == 0:
                reconstruction_fidelity = 1.0
            else:
                reconstruction_fidelity = 1.0 - (reconstruction_error / max_possible_error)
            
            # Combined alignment score
            final_alignment = 0.6 * alignment_score + 0.4 * reconstruction_fidelity
            
            logger.debug(f"SVD alignment: {final_alignment:.4f} (variance: {alignment_score:.4f}, fidelity: {reconstruction_fidelity:.4f})")
            
            return float(final_alignment)
            
        except Exception as e:
            logger.error(f"SVD attention alignment failed: {e}")
            return 0.0
    
    def _calculate_consciousness_coherence(self, attention_matrix: np.ndarray, 
                                         intention_matrix: np.ndarray) -> float:
        """
        Calculate consciousness coherence using cosine similarity
        
        Combines attention and intention vectors to measure overall
        consciousness alignment between agents.
        """
        try:
            # Combine attention and intention for full consciousness vectors
            consciousness_matrix = np.hstack([attention_matrix, intention_matrix])
            
            # Calculate pairwise cosine similarities
            similarity_matrix = cosine_similarity(consciousness_matrix)
            
            # Extract upper triangular (excluding diagonal)
            n = similarity_matrix.shape[0]
            upper_triangular = similarity_matrix[np.triu_indices(n, k=1)]
            
            if len(upper_triangular) == 0:
                return 0.0
            
            # Consciousness coherence is mean of pairwise similarities
            coherence = np.mean(upper_triangular)
            
            # Apply sigmoid transformation to enhance high coherence scores
            enhanced_coherence = self._sigmoid_enhance(coherence, midpoint=0.7, steepness=10)
            
            logger.debug(f"Consciousness coherence: {enhanced_coherence:.4f} (raw: {coherence:.4f})")
            
            return float(enhanced_coherence)
            
        except Exception as e:
            logger.error(f"Consciousness coherence calculation failed: {e}")
            return 0.0
    
    def _calculate_emotional_coupling(self, emotion_matrix: np.ndarray) -> float:
        """
        Calculate emotional resonance using L2 distance with 30th percentile coupling
        
        Uses L2 distance between emotional vectors with a percentile-based
        threshold for determining strong emotional coupling.
        """
        try:
            # Calculate pairwise L2 distances
            distances = []
            n_agents = emotion_matrix.shape[0]
            
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    distance = np.linalg.norm(emotion_matrix[i] - emotion_matrix[j])
                    distances.append(distance)
            
            if len(distances) == 0:
                return 0.0
            
            distances = np.array(distances)
            
            # Calculate 30th percentile threshold
            coupling_threshold = np.percentile(distances, self.emotional_coupling_percentile)
            
            # Count strong couplings (distances below threshold)
            strong_couplings = np.sum(distances <= coupling_threshold)
            total_pairs = len(distances)
            
            # Emotional coupling score
            coupling_ratio = strong_couplings / total_pairs
            
            # Convert distance-based metric to similarity-based (invert and normalize)
            # Lower distances = higher coupling
            max_distance = np.max(distances) if np.max(distances) > 0 else 1.0
            normalized_distances = distances / max_distance
            
            # Emotional resonance: inverse of mean normalized distance
            emotional_resonance = 1.0 - np.mean(normalized_distances)
            
            # Combined emotional coupling score
            emotional_coupling = 0.7 * emotional_resonance + 0.3 * coupling_ratio
            
            logger.debug(f"Emotional coupling: {emotional_coupling:.4f} (resonance: {emotional_resonance:.4f}, ratio: {coupling_ratio:.4f})")
            
            return float(emotional_coupling)
            
        except Exception as e:
            logger.error(f"Emotional coupling calculation failed: {e}")
            return 0.0
    
    def _calculate_metacognitive_alignment(self, metacognitive_matrix: np.ndarray) -> float:
        """
        Calculate metacognitive alignment across 5 dimensions
        
        Measures alignment in self-awareness, theory of mind, executive control,
        metacognitive knowledge, and metacognitive regulation.
        """
        try:
            # Calculate cosine similarity for metacognitive states
            similarity_matrix = cosine_similarity(metacognitive_matrix)
            
            # Extract upper triangular
            n = similarity_matrix.shape[0]
            upper_triangular = similarity_matrix[np.triu_indices(n, k=1)]
            
            if len(upper_triangular) == 0:
                return 0.0
            
            # Mean similarity across all agent pairs
            mean_alignment = np.mean(upper_triangular)
            
            # Calculate dimension-wise variance to reward consistency
            dimension_variances = np.var(metacognitive_matrix, axis=0)
            consistency_score = 1.0 - np.mean(dimension_variances)
            
            # Combined metacognitive alignment
            metacognitive_alignment = 0.6 * mean_alignment + 0.4 * consistency_score
            
            logger.debug(f"Metacognitive alignment: {metacognitive_alignment:.4f} (similarity: {mean_alignment:.4f}, consistency: {consistency_score:.4f})")
            
            return float(metacognitive_alignment)
            
        except Exception as e:
            logger.error(f"Metacognitive alignment calculation failed: {e}")
            return 0.0
    
    def _apply_individual_collective_weighting(self, consciousness_coherence: float, 
                                             attention_alignment: float) -> float:
        """
        Apply 70% individual / 30% collective ratio weighting
        
        Balances individual agent characteristics with collective alignment.
        """
        # Individual component: preserve agent distinctiveness
        individual_score = 1.0 - attention_alignment  # Diversity is good for individual
        
        # Collective component: reward alignment
        collective_score = consciousness_coherence
        
        # Weighted combination
        weighted_score = (self.individual_ratio * individual_score + 
                         self.collective_ratio * collective_score)
        
        logger.debug(f"Individual/Collective weighting: {weighted_score:.4f} (individual: {individual_score:.4f}, collective: {collective_score:.4f})")
        
        return weighted_score
    
    def _calculate_overall_sync_score(self, consciousness_coherence: float,
                                    emotional_coupling: float,
                                    metacognitive_alignment: float) -> float:
        """Calculate overall synchronization score"""
        # Weighted combination of all synchronization metrics
        weights = {
            'consciousness': 0.4,
            'emotional': 0.3,
            'metacognitive': 0.3
        }
        
        overall_score = (
            weights['consciousness'] * consciousness_coherence +
            weights['emotional'] * emotional_coupling +
            weights['metacognitive'] * metacognitive_alignment
        )
        
        return float(overall_score)
    
    def _create_synchronization_matrix(self, attention_matrix: np.ndarray,
                                     emotion_matrix: np.ndarray,
                                     intention_matrix: np.ndarray) -> np.ndarray:
        """Create comprehensive synchronization matrix"""
        try:
            # Combine all consciousness vectors
            combined_matrix = np.hstack([attention_matrix, emotion_matrix, intention_matrix])
            
            # Calculate full synchronization matrix
            sync_matrix = cosine_similarity(combined_matrix)
            
            return sync_matrix
            
        except Exception as e:
            logger.error(f"Synchronization matrix creation failed: {e}")
            return np.eye(len(attention_matrix))
    
    # Utility functions
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalize vector"""
        if len(vector) == 0:
            return vector
        
        # Ensure correct dimension
        if len(vector) < self.attention_dim:
            vector = np.pad(vector, (0, self.attention_dim - len(vector)))
        elif len(vector) > self.attention_dim:
            vector = vector[:self.attention_dim]
        
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
    
    def _bound_emotions(self, vector: np.ndarray) -> np.ndarray:
        """Apply tanh bounds to emotional vectors"""
        if len(vector) == 0:
            return vector
        
        # Ensure correct dimension
        if len(vector) < self.emotion_dim:
            vector = np.pad(vector, (0, self.emotion_dim - len(vector)))
        elif len(vector) > self.emotion_dim:
            vector = vector[:self.emotion_dim]
        
        return np.tanh(vector)
    
    def _sigmoid_enhance(self, x: float, midpoint: float = 0.5, steepness: float = 10) -> float:
        """Apply sigmoid enhancement to boost high scores"""
        return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))
    
    def _format_sync_results(self, result: SynchronizationResult) -> Dict[str, float]:
        """Format synchronization results for API response"""
        return {
            "consciousness_coherence": result.consciousness_coherence,
            "emotional_coupling": result.emotional_coupling,
            "metacognitive_alignment": result.metacognitive_alignment,
            "attention_alignment": result.attention_alignment,
            "overall_sync_score": result.overall_sync_score,
            "agents_count": result.agent_count,
            "individual_ratio": result.individual_collective_ratio[0],
            "collective_ratio": result.individual_collective_ratio[1],
            "timestamp": result.timestamp.isoformat(),
            "synchronization_quality": "excellent" if result.overall_sync_score > 0.95 else
                                     "good" if result.overall_sync_score > 0.8 else
                                     "moderate" if result.overall_sync_score > 0.6 else "poor"
        }
    
    def _empty_sync_result(self) -> Dict[str, float]:
        """Return empty synchronization result"""
        return {
            "consciousness_coherence": 0.0,
            "emotional_coupling": 0.0,
            "metacognitive_alignment": 0.0,
            "attention_alignment": 0.0,
            "overall_sync_score": 0.0,
            "agents_count": 0,
            "individual_ratio": self.individual_ratio,
            "collective_ratio": self.collective_ratio,
            "timestamp": datetime.now().isoformat(),
            "synchronization_quality": "no_agents"
        }
    
    def _error_sync_result(self, error_msg: str) -> Dict[str, float]:
        """Return error synchronization result"""
        return {
            "consciousness_coherence": 0.0,
            "emotional_coupling": 0.0,
            "metacognitive_alignment": 0.0,
            "attention_alignment": 0.0,
            "overall_sync_score": 0.0,
            "agents_count": 0,
            "individual_ratio": self.individual_ratio,
            "collective_ratio": self.collective_ratio,
            "timestamp": datetime.now().isoformat(),
            "synchronization_quality": "error",
            "error": error_msg
        }
    
    async def get_performance_history(self) -> List[Dict]:
        """Get historical performance data"""
        return [self._format_sync_results(result) for result in self.sync_history[-100:]]
    
    async def optimize_synchronization_parameters(self, target_performance: float = 0.95) -> Dict:
        """Optimize synchronization parameters for target performance"""
        # This would implement adaptive parameter tuning
        # For now, return current optimal parameters
        return {
            "individual_ratio": self.individual_ratio,
            "collective_ratio": self.collective_ratio,
            "emotional_coupling_percentile": self.emotional_coupling_percentile,
            "target_performance": target_performance,
            "current_performance": self.sync_history[-1].overall_sync_score if self.sync_history else 0.0
        }