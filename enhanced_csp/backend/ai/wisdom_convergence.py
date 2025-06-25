# backend/ai/wisdom_convergence.py
"""
Meta-Wisdom Convergence
=======================
Implements wisdom extraction and dialectical synthesis with:
- Wisdom extraction from reasoning history and core beliefs
- 6-dimensional wisdom vectors: confidence, emotional_resonance, logical_strength, 
  practical_applicability, aesthetic_value, transcendence_level
- Dialectical synthesis (thesis-antithesis-synthesis pattern)
- Transcendent principle generation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import uuid
import json

logger = logging.getLogger(__name__)

@dataclass
class WisdomVector:
    """6-dimensional wisdom representation"""
    confidence: float              # How certain the wisdom is
    emotional_resonance: float     # Emotional impact and connection
    logical_strength: float        # Logical consistency and coherence  
    practical_applicability: float # Real-world utility and effectiveness
    aesthetic_value: float         # Beauty, elegance, and harmony
    transcendence_level: float     # Ability to transcend current understanding
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([
            self.confidence,
            self.emotional_resonance,
            self.logical_strength,
            self.practical_applicability,
            self.aesthetic_value,
            self.transcendence_level
        ])
    
    @classmethod
    def from_array(cls, array: np.ndarray) -> 'WisdomVector':
        """Create from numpy array"""
        return cls(
            confidence=float(array[0]),
            emotional_resonance=float(array[1]),
            logical_strength=float(array[2]),
            practical_applicability=float(array[3]),
            aesthetic_value=float(array[4]),
            transcendence_level=float(array[5])
        )

@dataclass
class WisdomExtraction:
    """Extracted wisdom from agent reasoning"""
    id: str
    agent_id: str
    wisdom_vector: WisdomVector
    source_reasoning: Dict
    core_beliefs: Dict
    wisdom_score: float
    extraction_method: str
    timestamp: datetime
    context: Optional[str] = None

@dataclass
class DialecticalSynthesis:
    """Result of dialectical synthesis between wisdom sources"""
    id: str
    thesis_wisdom: WisdomExtraction
    antithesis_wisdom: WisdomExtraction
    synthesis_vector: WisdomVector
    synthesis_score: float
    transcendence_achieved: bool
    improvement_factor: float
    synthesis_method: str
    timestamp: datetime

@dataclass
class TranscendentPrinciple:
    """Emergent transcendent principle from collective wisdom"""
    id: str
    principle_text: str
    strength: float
    collective_wisdom: WisdomVector
    wisdom_diversity: np.ndarray
    contributing_agents: List[str]
    principle_category: str
    timestamp: datetime

class MetaWisdomConvergence:
    """
    Advanced wisdom extraction and convergence system implementing
    dialectical synthesis and transcendent principle generation.
    """
    
    def __init__(self):
        self.wisdom_dimensions = 6
        self.transcendence_threshold = 0.8
        self.synthesis_amplification = 1.1
        
        # Wisdom storage
        self.wisdom_extractions: Dict[str, WisdomExtraction] = {}
        self.dialectical_syntheses: Dict[str, DialecticalSynthesis] = {}
        self.transcendent_principles: Dict[str, TranscendentPrinciple] = {}
        
        # Agent wisdom tracking
        self.agent_wisdom_history: Dict[str, List[str]] = defaultdict(list)
        self.principle_evolution: List[TranscendentPrinciple] = []
        
        # Wisdom dimension weights (can be tuned)
        self.dimension_weights = {
            'confidence': 1.0,
            'emotional_resonance': 1.2,
            'logical_strength': 1.3,
            'practical_applicability': 1.4,
            'aesthetic_value': 1.1,
            'transcendence_level': 1.5
        }
    
    async def extract_wisdom(self, agent_reasoning: Dict, agent_id: str = None) -> Dict:
        """
        Extract wisdom from agent reasoning history and core beliefs
        
        Args:
            agent_reasoning: Dictionary containing reasoning history and beliefs
            agent_id: Optional agent identifier
            
        Returns:
            Extracted wisdom metrics and vector
        """
        try:
            reasoning_history = agent_reasoning.get('history', [])
            core_beliefs = agent_reasoning.get('beliefs', {})
            agent_id = agent_id or agent_reasoning.get('agent_id', f'agent_{uuid.uuid4().hex[:8]}')
            
            # Calculate 6-dimensional wisdom vector
            wisdom_vector = WisdomVector(
                confidence=self._calculate_confidence(reasoning_history),
                emotional_resonance=self._calculate_emotional_resonance(reasoning_history),
                logical_strength=self._calculate_logical_strength(reasoning_history),
                practical_applicability=self._calculate_practical_applicability(core_beliefs),
                aesthetic_value=self._calculate_aesthetic_value(reasoning_history),
                transcendence_level=self._calculate_transcendence_level(core_beliefs)
            )
            
            # Calculate overall wisdom score
            wisdom_score = self._calculate_weighted_wisdom_score(wisdom_vector)
            
            # Create wisdom extraction
            extraction = WisdomExtraction(
                id=f'wisdom_{uuid.uuid4().hex[:8]}',
                agent_id=agent_id,
                wisdom_vector=wisdom_vector,
                source_reasoning=reasoning_history,
                core_beliefs=core_beliefs,
                wisdom_score=wisdom_score,
                extraction_method='comprehensive',
                timestamp=datetime.now(),
                context=agent_reasoning.get('context')
            )
            
            # Store extraction
            self.wisdom_extractions[extraction.id] = extraction
            self.agent_wisdom_history[agent_id].append(extraction.id)
            
            logger.info(f"Extracted wisdom {extraction.id} for agent {agent_id}: score {wisdom_score:.4f}")
            
            return {
                "wisdom_id": extraction.id,
                "agent_id": agent_id,
                "wisdom_vector": [
                    wisdom_vector.confidence,
                    wisdom_vector.emotional_resonance,
                    wisdom_vector.logical_strength,
                    wisdom_vector.practical_applicability,
                    wisdom_vector.aesthetic_value,
                    wisdom_vector.transcendence_level
                ],
                "wisdom_score": wisdom_score,
                "dimensions": {
                    "confidence": wisdom_vector.confidence,
                    "emotional_resonance": wisdom_vector.emotional_resonance,
                    "logical_strength": wisdom_vector.logical_strength,
                    "practical_applicability": wisdom_vector.practical_applicability,
                    "aesthetic_value": wisdom_vector.aesthetic_value,
                    "transcendence_level": wisdom_vector.transcendence_level
                },
                "extraction_quality": self._assess_extraction_quality(wisdom_vector),
                "timestamp": extraction.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Wisdom extraction failed: {e}")
            return {"error": str(e), "wisdom_extracted": False}
    
    async def dialectical_synthesis(self, thesis_wisdom_id: str, 
                                  antithesis_wisdom_id: str) -> Dict:
        """
        Perform dialectical synthesis of two wisdom extractions
        
        Args:
            thesis_wisdom_id: ID of thesis wisdom
            antithesis_wisdom_id: ID of antithesis wisdom
            
        Returns:
            Synthesis results and transcendence analysis
        """
        try:
            # Retrieve wisdom extractions
            if thesis_wisdom_id not in self.wisdom_extractions:
                return {"error": "Thesis wisdom not found", "synthesized": False}
            if antithesis_wisdom_id not in self.wisdom_extractions:
                return {"error": "Antithesis wisdom not found", "synthesized": False}
            
            thesis_wisdom = self.wisdom_extractions[thesis_wisdom_id]
            antithesis_wisdom = self.wisdom_extractions[antithesis_wisdom_id]
            
            # Perform dialectical synthesis
            synthesis_vector = self._perform_dialectical_synthesis(
                thesis_wisdom.wisdom_vector, antithesis_wisdom.wisdom_vector
            )
            
            # Calculate synthesis metrics
            synthesis_score = self._calculate_weighted_wisdom_score(synthesis_vector)
            
            # Check for transcendence
            transcendence_achieved = self._check_transcendence(
                thesis_wisdom.wisdom_vector, 
                antithesis_wisdom.wisdom_vector, 
                synthesis_vector
            )
            
            # Calculate improvement factor
            improvement_factor = synthesis_score / max(
                thesis_wisdom.wisdom_score, antithesis_wisdom.wisdom_score
            )
            
            # Create synthesis record
            synthesis = DialecticalSynthesis(
                id=f'synthesis_{uuid.uuid4().hex[:8]}',
                thesis_wisdom=thesis_wisdom,
                antithesis_wisdom=antithesis_wisdom,
                synthesis_vector=synthesis_vector,
                synthesis_score=synthesis_score,
                transcendence_achieved=transcendence_achieved,
                improvement_factor=improvement_factor,
                synthesis_method='dialectical_max_amplification',
                timestamp=datetime.now()
            )
            
            # Store synthesis
            self.dialectical_syntheses[synthesis.id] = synthesis
            
            logger.info(f"Created synthesis {synthesis.id}: score {synthesis_score:.4f}, transcendence: {transcendence_achieved}")
            
            return {
                "synthesis_id": synthesis.id,
                "synthesized": True,
                "synthesis_vector": synthesis_vector.to_array().tolist(),
                "synthesis_score": synthesis_score,
                "transcendence_achieved": transcendence_achieved,
                "improvement_factor": improvement_factor,
                "thesis_score": thesis_wisdom.wisdom_score,
                "antithesis_score": antithesis_wisdom.wisdom_score,
                "transcendence_dimensions": self._analyze_transcendence_dimensions(
                    thesis_wisdom.wisdom_vector, antithesis_wisdom.wisdom_vector, synthesis_vector
                ),
                "synthesis_quality": self._assess_synthesis_quality(synthesis),
                "timestamp": synthesis.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Dialectical synthesis failed: {e}")
            return {"error": str(e), "synthesized": False}
    
    async def generate_transcendent_principle(self, wisdom_data: List[Dict], 
                                            principle_context: str = None) -> Dict:
        """
        Generate transcendent principles from collective wisdom
        
        Args:
            wisdom_data: List of wisdom extraction results
            principle_context: Optional context for principle generation
            
        Returns:
            Generated transcendent principle information
        """
        try:
            if len(wisdom_data) < 2:
                return {
                    "principle_generated": False,
                    "reason": "insufficient_wisdom_data",
                    "required_minimum": 2,
                    "provided": len(wisdom_data)
                }
            
            # Extract wisdom vectors
            wisdom_vectors = []
            contributing_agents = []
            
            for wisdom_item in wisdom_data:
                if 'wisdom_vector' in wisdom_item:
                    wisdom_vectors.append(np.array(wisdom_item['wisdom_vector']))
                    contributing_agents.append(wisdom_item.get('agent_id', 'unknown'))
                elif 'dimensions' in wisdom_item:
                    # Extract from dimensions dict
                    dims = wisdom_item['dimensions']
                    vector = np.array([
                        dims.get('confidence', 0.5),
                        dims.get('emotional_resonance', 0.5),
                        dims.get('logical_strength', 0.5),
                        dims.get('practical_applicability', 0.5),
                        dims.get('aesthetic_value', 0.5),
                        dims.get('transcendence_level', 0.5)
                    ])
                    wisdom_vectors.append(vector)
                    contributing_agents.append(wisdom_item.get('agent_id', 'unknown'))
            
            if len(wisdom_vectors) == 0:
                return {"principle_generated": False, "reason": "no_valid_wisdom_vectors"}
            
            # Calculate collective wisdom metrics
            wisdom_matrix = np.array(wisdom_vectors)
            collective_wisdom_vector = np.mean(wisdom_matrix, axis=0)
            wisdom_diversity = np.var(wisdom_matrix, axis=0)
            
            # Create collective wisdom object
            collective_wisdom = WisdomVector.from_array(collective_wisdom_vector)
            
            # Generate principle based on dominant wisdom dimensions
            principle_analysis = self._analyze_dominant_wisdom_dimensions(
                collective_wisdom_vector, wisdom_diversity
            )
            
            # Generate principle text
            principle_text = self._generate_principle_text(
                principle_analysis, collective_wisdom, principle_context
            )
            
            # Calculate principle strength
            principle_strength = self._calculate_principle_strength(
                collective_wisdom_vector, wisdom_diversity
            )
            
            # Create transcendent principle
            principle = TranscendentPrinciple(
                id=f'principle_{uuid.uuid4().hex[:8]}',
                principle_text=principle_text,
                strength=principle_strength,
                collective_wisdom=collective_wisdom,
                wisdom_diversity=wisdom_diversity,
                contributing_agents=list(set(contributing_agents)),
                principle_category=principle_analysis['category'],
                timestamp=datetime.now()
            )
            
            # Store principle
            self.transcendent_principles[principle.id] = principle
            self.principle_evolution.append(principle)
            
            logger.info(f"Generated transcendent principle {principle.id}: '{principle_text[:50]}...' (strength: {principle_strength:.4f})")
            
            return {
                "principle_generated": True,
                "principle_id": principle.id,
                "principle_text": principle_text,
                "strength": principle_strength,
                "collective_wisdom": collective_wisdom_vector.tolist(),
                "wisdom_diversity": wisdom_diversity.tolist(),
                "contributing_agents": principle.contributing_agents,
                "principle_category": principle.principle_category,
                "dominant_dimension": principle_analysis['dominant_dimension'],
                "transcendence_indicators": principle_analysis['transcendence_indicators'],
                "principle_quality": self._assess_principle_quality(principle),
                "timestamp": principle.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Transcendent principle generation failed: {e}")
            return {"error": str(e), "principle_generated": False}
    
    # Core calculation methods
    def _calculate_confidence(self, reasoning_history: List) -> float:
        """Calculate confidence from reasoning history accuracy"""
        if not reasoning_history:
            return 0.5
        
        # Count correct predictions/reasoning
        correct_count = 0
        total_count = 0
        
        for item in reasoning_history:
            if isinstance(item, dict):
                if 'correct' in item:
                    total_count += 1
                    if item['correct']:
                        correct_count += 1
                elif 'accuracy' in item:
                    total_count += 1
                    correct_count += item['accuracy']
                elif 'success' in item:
                    total_count += 1
                    if item['success']:
                        correct_count += 1
        
        if total_count == 0:
            return 0.5
        
        # Calculate confidence with experience weighting
        raw_confidence = correct_count / total_count
        experience_factor = min(1.0, total_count / 20.0)  # More experience = higher confidence
        
        return min(raw_confidence * (0.5 + 0.5 * experience_factor), 1.0)
    
    def _calculate_emotional_resonance(self, reasoning_history: List) -> float:
        """Calculate emotional resonance from reasoning emotional impact"""
        if not reasoning_history:
            return 0.5
        
        emotional_scores = []
        
        for item in reasoning_history:
            if isinstance(item, dict):
                # Extract various emotional indicators
                emotional_impact = item.get('emotional_impact', 0.5)
                empathy_score = item.get('empathy', 0.5)
                emotional_intelligence = item.get('emotional_intelligence', 0.5)
                human_connection = item.get('human_connection', 0.5)
                
                # Combined emotional score
                combined_score = np.mean([emotional_impact, empathy_score, 
                                        emotional_intelligence, human_connection])
                emotional_scores.append(combined_score)
        
        if not emotional_scores:
            return 0.5
        
        return float(np.mean(emotional_scores))
    
    def _calculate_logical_strength(self, reasoning_history: List) -> float:
        """Calculate logical strength and consistency"""
        if not reasoning_history:
            return 0.5
        
        logical_scores = []
        
        for item in reasoning_history:
            if isinstance(item, dict):
                # Extract logical indicators
                logical_consistency = item.get('logical_consistency', 0.5)
                reasoning_quality = item.get('reasoning_quality', 0.5)
                argument_strength = item.get('argument_strength', 0.5)
                coherence = item.get('coherence', 0.5)
                
                # Check for logical fallacies (penalty)
                fallacy_penalty = item.get('logical_fallacies', 0) * 0.1
                
                # Combined logical score
                combined_score = np.mean([logical_consistency, reasoning_quality, 
                                        argument_strength, coherence]) - fallacy_penalty
                logical_scores.append(max(0.0, combined_score))
        
        if not logical_scores:
            return 0.5
        
        # Reward consistency across reasoning instances
        consistency_bonus = 1.0 - np.std(logical_scores)
        base_score = np.mean(logical_scores)
        
        return min(base_score * (0.8 + 0.2 * consistency_bonus), 1.0)
    
    def _calculate_practical_applicability(self, core_beliefs: Dict) -> float:
        """Calculate practical applicability of beliefs and knowledge"""
        if not core_beliefs:
            return 0.5
        
        # Extract practical indicators
        practical_value = core_beliefs.get('practical_value', 0.5)
        real_world_applicability = core_beliefs.get('real_world_applicability', 0.5)
        actionability = core_beliefs.get('actionability', 0.5)
        implementation_ease = core_beliefs.get('implementation_ease', 0.5)
        
        # Check for proven applications
        proven_applications = core_beliefs.get('proven_applications', [])
        application_bonus = min(len(proven_applications) * 0.1, 0.3)
        
        # Calculate base score
        base_score = np.mean([practical_value, real_world_applicability, 
                             actionability, implementation_ease])
        
        return min(base_score + application_bonus, 1.0)
    
    def _calculate_aesthetic_value(self, reasoning_history: List) -> float:
        """Calculate aesthetic value and elegance of reasoning"""
        if not reasoning_history:
            return 0.5
        
        aesthetic_scores = []
        
        for item in reasoning_history:
            if isinstance(item, dict):
                # Extract aesthetic indicators
                elegance = item.get('elegance', 0.5)
                simplicity = item.get('simplicity', 0.5)
                beauty = item.get('beauty', 0.5)
                harmony = item.get('harmony', 0.5)
                creativity = item.get('creativity', 0.5)
                
                # Penalize overly complex solutions
                complexity_penalty = item.get('unnecessary_complexity', 0) * 0.1
                
                # Combined aesthetic score
                combined_score = np.mean([elegance, simplicity, beauty, harmony, creativity]) - complexity_penalty
                aesthetic_scores.append(max(0.0, combined_score))
        
        if not aesthetic_scores:
            return 0.5
        
        return float(np.mean(aesthetic_scores))
    
    def _calculate_transcendence_level(self, core_beliefs: Dict) -> float:
        """Calculate transcendence level of beliefs and insights"""
        if not core_beliefs:
            return 0.5
        
        # Extract transcendence indicators
        transcendence_factor = core_beliefs.get('transcendence_factor', 0.5)
        paradigm_shifting = core_beliefs.get('paradigm_shifting', 0.5)
        consciousness_expansion = core_beliefs.get('consciousness_expansion', 0.5)
        universal_principles = core_beliefs.get('universal_principles', 0.5)
        
        # Check for breakthrough insights
        breakthrough_insights = core_beliefs.get('breakthrough_insights', [])
        breakthrough_bonus = min(len(breakthrough_insights) * 0.15, 0.4)
        
        # Check for cross-domain applicability
        cross_domain_applicability = core_beliefs.get('cross_domain_applicability', 0.5)
        
        # Calculate transcendence score
        base_components = [transcendence_factor, paradigm_shifting, 
                          consciousness_expansion, universal_principles, cross_domain_applicability]
        base_score = np.mean(base_components)
        
        return min(base_score + breakthrough_bonus, 1.0)
    
    def _calculate_weighted_wisdom_score(self, wisdom_vector: WisdomVector) -> float:
        """Calculate weighted overall wisdom score"""
        weights = np.array([
            self.dimension_weights['confidence'],
            self.dimension_weights['emotional_resonance'],
            self.dimension_weights['logical_strength'],
            self.dimension_weights['practical_applicability'],
            self.dimension_weights['aesthetic_value'],
            self.dimension_weights['transcendence_level']
        ])
        
        wisdom_array = wisdom_vector.to_array()
        weighted_score = np.sum(wisdom_array * weights) / np.sum(weights)
        
        return float(weighted_score)
    
    def _perform_dialectical_synthesis(self, thesis: WisdomVector, 
                                     antithesis: WisdomVector) -> WisdomVector:
        """
        Perform dialectical synthesis using max amplification method
        
        Synthesis = max(thesis, antithesis) * amplification_factor
        """
        thesis_array = thesis.to_array()
        antithesis_array = antithesis.to_array()
        
        # Take maximum of each dimension and amplify
        synthesis_array = np.maximum(thesis_array, antithesis_array) * self.synthesis_amplification
        
        # Ensure values don't exceed 1.0
        synthesis_array = np.minimum(synthesis_array, 1.0)
        
        return WisdomVector.from_array(synthesis_array)
    
    def _check_transcendence(self, thesis: WisdomVector, antithesis: WisdomVector, 
                           synthesis: WisdomVector) -> bool:
        """Check if synthesis achieves transcendence"""
        # Transcendence achieved if synthesis transcendence_level > max of inputs
        thesis_transcendence = thesis.transcendence_level
        antithesis_transcendence = antithesis.transcendence_level
        synthesis_transcendence = synthesis.transcendence_level
        
        max_input_transcendence = max(thesis_transcendence, antithesis_transcendence)
        
        return synthesis_transcendence > max_input_transcendence
    
    def _analyze_transcendence_dimensions(self, thesis: WisdomVector, 
                                        antithesis: WisdomVector, 
                                        synthesis: WisdomVector) -> Dict:
        """Analyze which dimensions achieved transcendence"""
        thesis_array = thesis.to_array()
        antithesis_array = antithesis.to_array()
        synthesis_array = synthesis.to_array()
        
        dimension_names = ['confidence', 'emotional_resonance', 'logical_strength',
                          'practical_applicability', 'aesthetic_value', 'transcendence_level']
        
        transcendence_analysis = {}
        
        for i, dim_name in enumerate(dimension_names):
            max_input = max(thesis_array[i], antithesis_array[i])
            synthesis_value = synthesis_array[i]
            
            transcendence_analysis[dim_name] = {
                'achieved_transcendence': synthesis_value > max_input,
                'improvement': synthesis_value - max_input,
                'synthesis_value': synthesis_value,
                'max_input_value': max_input
            }
        
        return transcendence_analysis
    
    def _analyze_dominant_wisdom_dimensions(self, collective_wisdom: np.ndarray, 
                                          wisdom_diversity: np.ndarray) -> Dict:
        """Analyze dominant wisdom dimensions for principle generation"""
        dimension_names = ['confidence', 'emotional_resonance', 'logical_strength',
                          'practical_applicability', 'aesthetic_value', 'transcendence_level']
        
        # Find dominant dimension
        dominant_idx = np.argmax(collective_wisdom)
        dominant_dimension = dimension_names[dominant_idx]
        dominant_value = collective_wisdom[dominant_idx]
        
        # Analyze diversity patterns
        high_diversity_dims = [dimension_names[i] for i, div in enumerate(wisdom_diversity) if div > 0.1]
        low_diversity_dims = [dimension_names[i] for i, div in enumerate(wisdom_diversity) if div < 0.05]
        
        # Determine principle category
        if dominant_dimension == 'transcendence_level':
            category = 'transcendental'
        elif dominant_dimension == 'logical_strength':
            category = 'rational'
        elif dominant_dimension == 'emotional_resonance':
            category = 'empathetic'
        elif dominant_dimension == 'practical_applicability':
            category = 'pragmatic'
        elif dominant_dimension == 'aesthetic_value':
            category = 'artistic'
        else:
            category = 'balanced'
        
        # Identify transcendence indicators
        transcendence_indicators = []
        if collective_wisdom[5] > 0.8:  # transcendence_level
            transcendence_indicators.append('high_transcendence')
        if np.mean(collective_wisdom) > 0.85:
            transcendence_indicators.append('overall_excellence')
        if len(high_diversity_dims) >= 3:
            transcendence_indicators.append('dimensional_diversity')
        
        return {
            'dominant_dimension': dominant_dimension,
            'dominant_value': float(dominant_value),
            'category': category,
            'high_diversity_dimensions': high_diversity_dims,
            'low_diversity_dimensions': low_diversity_dims,
            'transcendence_indicators': transcendence_indicators
        }
    
    def _generate_principle_text(self, analysis: Dict, collective_wisdom: WisdomVector, 
                               context: str = None) -> str:
        """Generate human-readable principle text"""
        dominant_dim = analysis['dominant_dimension']
        category = analysis['category']
        
        # Base principle templates
        principle_templates = {
            'transcendental': [
                f"Transcendent principle: Expand consciousness through {dominant_dim}",
                f"Universal truth: Higher {dominant_dim} leads to expanded awareness",
                f"Evolutionary principle: Consciousness evolves through enhanced {dominant_dim}"
            ],
            'rational': [
                f"Logical principle: Optimize decisions through {dominant_dim}",
                f"Rational framework: Strengthen reasoning via {dominant_dim}",
                f"Analytical truth: Clear thinking emerges from {dominant_dim}"
            ],
            'empathetic': [
                f"Empathetic principle: Connect deeply through {dominant_dim}",
                f"Emotional wisdom: Understanding grows through {dominant_dim}",
                f"Compassionate truth: Healing happens via {dominant_dim}"
            ],
            'pragmatic': [
                f"Practical principle: Achieve results through {dominant_dim}",
                f"Applied wisdom: Effectiveness flows from {dominant_dim}",
                f"Pragmatic truth: Impact increases with {dominant_dim}"
            ],
            'artistic': [
                f"Aesthetic principle: Create beauty through {dominant_dim}",
                f"Artistic wisdom: Elegance emerges from {dominant_dim}",
                f"Creative truth: Inspiration flows through {dominant_dim}"
            ],
            'balanced': [
                f"Balanced principle: Harmonize all aspects through {dominant_dim}",
                f"Integrated wisdom: Wholeness achieved via {dominant_dim}",
                f"Holistic truth: Unity emerges through {dominant_dim}"
            ]
        }
        
        # Select appropriate template
        templates = principle_templates.get(category, principle_templates['balanced'])
        base_principle = templates[0]  # Use first template for consistency
        
        # Add context if provided
        if context:
            base_principle += f" in the context of {context}"
        
        # Add transcendence indicators
        if 'high_transcendence' in analysis['transcendence_indicators']:
            base_principle += " with transcendent potential"
        
        return base_principle
    
    def _calculate_principle_strength(self, collective_wisdom: np.ndarray, 
                                    wisdom_diversity: np.ndarray) -> float:
        """Calculate strength of generated principle"""
        # Base strength from collective wisdom
        base_strength = np.mean(collective_wisdom)
        
        # Bonus for high values in key dimensions
        transcendence_bonus = collective_wisdom[5] * 0.2  # transcendence_level
        logical_bonus = collective_wisdom[2] * 0.1        # logical_strength
        
        # Diversity factor (some diversity is good, too much is chaotic)
        diversity_factor = 1.0 - min(np.mean(wisdom_diversity), 0.3)
        
        # Calculate final strength
        principle_strength = (base_strength + transcendence_bonus + logical_bonus) * diversity_factor
        
        return min(principle_strength, 1.0)
    
    # Assessment methods
    def _assess_extraction_quality(self, wisdom_vector: WisdomVector) -> str:
        """Assess quality of wisdom extraction"""
        score = self._calculate_weighted_wisdom_score(wisdom_vector)
        
        if score > 0.9:
            return "exceptional"
        elif score > 0.8:
            return "excellent"
        elif score > 0.7:
            return "good"
        elif score > 0.6:
            return "moderate"
        else:
            return "poor"
    
    def _assess_synthesis_quality(self, synthesis: DialecticalSynthesis) -> str:
        """Assess quality of dialectical synthesis"""
        if synthesis.transcendence_achieved and synthesis.improvement_factor > 1.15:
            return "transcendent"
        elif synthesis.improvement_factor > 1.1:
            return "excellent"
        elif synthesis.improvement_factor > 1.05:
            return "good"
        elif synthesis.improvement_factor > 1.0:
            return "moderate"
        else:
            return "poor"
    
    def _assess_principle_quality(self, principle: TranscendentPrinciple) -> str:
        """Assess quality of transcendent principle"""
        if principle.strength > 0.9 and len(principle.contributing_agents) >= 5:
            return "universal"
        elif principle.strength > 0.8:
            return "strong"
        elif principle.strength > 0.7:
            return "moderate"
        elif principle.strength > 0.6:
            return "emerging"
        else:
            return "weak"
    
    # Public interface methods
    async def get_wisdom_statistics(self) -> Dict:
        """Get comprehensive wisdom system statistics"""
        total_extractions = len(self.wisdom_extractions)
        total_syntheses = len(self.dialectical_syntheses)
        total_principles = len(self.transcendent_principles)
        
        # Calculate average scores
        if self.wisdom_extractions:
            avg_wisdom_score = np.mean([w.wisdom_score for w in self.wisdom_extractions.values()])
        else:
            avg_wisdom_score = 0.0
        
        if self.dialectical_syntheses:
            avg_synthesis_score = np.mean([s.synthesis_score for s in self.dialectical_syntheses.values()])
            transcendence_rate = np.mean([s.transcendence_achieved for s in self.dialectical_syntheses.values()])
        else:
            avg_synthesis_score = 0.0
            transcendence_rate = 0.0
        
        if self.transcendent_principles:
            avg_principle_strength = np.mean([p.strength for p in self.transcendent_principles.values()])
        else:
            avg_principle_strength = 0.0
        
        return {
            "total_wisdom_extractions": total_extractions,
            "total_dialectical_syntheses": total_syntheses,
            "total_transcendent_principles": total_principles,
            "average_wisdom_score": float(avg_wisdom_score),
            "average_synthesis_score": float(avg_synthesis_score),
            "average_principle_strength": float(avg_principle_strength),
            "transcendence_achievement_rate": float(transcendence_rate),
            "active_agents": len(self.agent_wisdom_history),
            "wisdom_dimensions": self.wisdom_dimensions,
            "system_maturity": "advanced" if total_principles > 10 else "developing" if total_principles > 3 else "emerging"
        }
    
    async def optimize_wisdom_parameters(self, target_transcendence_rate: float = 0.8) -> Dict:
        """Optimize wisdom convergence parameters"""
        current_stats = await self.get_wisdom_statistics()
        current_transcendence_rate = current_stats["transcendence_achievement_rate"]
        
        # Adjust synthesis amplification based on transcendence rate
        if current_transcendence_rate < target_transcendence_rate:
            self.synthesis_amplification = min(self.synthesis_amplification * 1.05, 1.3)
        elif current_transcendence_rate > target_transcendence_rate + 0.1:
            self.synthesis_amplification = max(self.synthesis_amplification * 0.95, 1.0)
        
        return {
            "synthesis_amplification": self.synthesis_amplification,
            "transcendence_threshold": self.transcendence_threshold,
            "current_transcendence_rate": current_transcendence_rate,
            "target_transcendence_rate": target_transcendence_rate,
            "optimization_applied": True
        }