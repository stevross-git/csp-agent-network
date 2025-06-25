# backend/ai/temporal_entanglement.py
"""
Temporal Entanglement
====================
Implements multi-scale temporal coordination with:
- 7 temporal scales: nanosecond, microsecond, millisecond, second, minute, hour, day
- Phase coherence calculations using complex exponentials
- Vector clocks for causal consistency (Lamport timestamps)
- Correlation matrix eigenvalue decomposition for system dynamics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid
import threading

logger = logging.getLogger(__name__)

@dataclass
class TemporalPhase:
    """Phase information for a specific temporal scale"""
    scale: str
    phase: float
    frequency: float
    amplitude: float
    timestamp: float
    agent_id: str

@dataclass
class VectorClock:
    """Vector clock for causal consistency"""
    agent_id: str
    logical_time: int
    physical_time: float
    events: List[Dict] = field(default_factory=list)
    causal_dependencies: List[str] = field(default_factory=list)

@dataclass
class TemporalEvent:
    """Temporal event with causal information"""
    id: str
    agent_id: str
    event_type: str
    logical_time: int
    physical_time: float
    scale: str
    phase: float
    causal_dependencies: List[str]
    data: Dict = field(default_factory=dict)

@dataclass
class CoherenceAnalysis:
    """Results of phase coherence analysis"""
    scale: str
    coherence: float
    mean_phase: float
    phase_variance: float
    agent_count: int
    synchronization_quality: str
    timestamp: float

class TemporalEntanglement:
    """
    Multi-scale temporal coordination system implementing phase coherence,
    vector clocks, and causal consistency for AI agent coordination.
    """
    
    def __init__(self):
        # Temporal scale definitions (in seconds)
        self.temporal_scales = {
            'nanosecond': 1e-9,
            'microsecond': 1e-6,
            'millisecond': 1e-3,
            'second': 1.0,
            'minute': 60.0,
            'hour': 3600.0,
            'day': 86400.0
        }
        
        # System state
        self.vector_clocks: Dict[str, VectorClock] = {}
        self.temporal_phases: Dict[str, Dict[str, TemporalPhase]] = defaultdict(dict)
        self.temporal_events: List[TemporalEvent] = []
        self.coherence_history: List[CoherenceAnalysis] = []
        
        # Correlation analysis
        self.correlation_matrices: Dict[str, np.ndarray] = {}
        self.eigenvalue_history: Dict[str, List[np.ndarray]] = defaultdict(list)
        
        # Synchronization parameters
        self.max_events_per_agent = 1000
        self.coherence_window = 60.0  # seconds
        self.phase_update_interval = 0.1  # seconds
        
        # Threading for real-time updates
        self._phase_update_lock = threading.Lock()
        self._last_phase_update = time.time()
        
    async def calculate_phase_coherence(self, agent_phases: Dict[str, List[float]]) -> Dict:
        """
        Calculate phase coherence across temporal scales using complex exponentials
        
        Args:
            agent_phases: Dictionary mapping scale names to lists of agent phases
            
        Returns:
            Comprehensive coherence analysis across all scales
        """
        try:
            coherence_results = {}
            overall_coherences = []
            
            for scale, phases in agent_phases.items():
                if scale not in self.temporal_scales:
                    logger.warning(f"Unknown temporal scale: {scale}")
                    continue
                
                if len(phases) < 2:
                    logger.debug(f"Insufficient agents for coherence calculation in scale {scale}")
                    continue
                
                # Convert phases to complex exponentials
                complex_phases = [np.exp(1j * phase) for phase in phases]
                complex_phases_array = np.array(complex_phases)
                
                # Calculate mean complex phase (order parameter)
                mean_complex_phase = np.mean(complex_phases_array)
                
                # Phase coherence is magnitude of mean complex phase
                coherence = abs(mean_complex_phase)
                
                # Extract mean phase angle
                mean_phase = float(np.angle(mean_complex_phase))
                
                # Calculate phase variance
                phase_variance = np.var(phases)
                
                # Assess synchronization quality
                sync_quality = self._assess_synchronization_quality(coherence, phase_variance)
                
                # Create coherence analysis
                analysis = CoherenceAnalysis(
                    scale=scale,
                    coherence=coherence,
                    mean_phase=mean_phase,
                    phase_variance=phase_variance,
                    agent_count=len(phases),
                    synchronization_quality=sync_quality,
                    timestamp=time.time()
                )
                
                coherence_results[scale] = {
                    'coherence': coherence,
                    'mean_phase': mean_phase,
                    'phase_variance': phase_variance,
                    'agent_count': len(phases),
                    'synchronization_quality': sync_quality,
                    'complex_order_parameter': {
                        'real': float(mean_complex_phase.real),
                        'imag': float(mean_complex_phase.imag),
                        'magnitude': coherence,
                        'phase': mean_phase
                    }
                }
                
                # Store coherence analysis
                self.coherence_history.append(analysis)
                overall_coherences.append(coherence)
                
                # Update temporal phases
                await self._update_temporal_phases(scale, phases, mean_phase)
            
            # Calculate overall system coherence
            overall_coherence = np.mean(overall_coherences) if overall_coherences else 0.0
            
            # Analyze cross-scale relationships
            cross_scale_analysis = await self._analyze_cross_scale_coherence(coherence_results)
            
            # Clean up old coherence history
            self._cleanup_coherence_history()
            
            logger.info(f"Calculated phase coherence across {len(coherence_results)} scales, overall: {overall_coherence:.4f}")
            
            return {
                'scale_coherence': coherence_results,
                'overall_coherence': overall_coherence,
                'cross_scale_analysis': cross_scale_analysis,
                'temporal_scales_analyzed': list(coherence_results.keys()),
                'total_agents': sum(result['agent_count'] for result in coherence_results.values()),
                'coherence_quality': self._assess_overall_coherence_quality(overall_coherence),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Phase coherence calculation failed: {e}")
            return {'error': str(e), 'coherence_calculated': False}
    
    async def update_vector_clock(self, agent_id: str, event_type: str, 
                                event_data: Dict = None) -> Dict:
        """
        Update vector clock for causal consistency using Lamport timestamps
        
        Args:
            agent_id: Identifier of the agent
            event_type: Type of event occurring
            event_data: Optional event data
            
        Returns:
            Updated vector clock information
        """
        try:
            current_time = time.time()
            
            # Initialize vector clock if not exists
            if agent_id not in self.vector_clocks:
                self.vector_clocks[agent_id] = VectorClock(
                    agent_id=agent_id,
                    logical_time=0,
                    physical_time=current_time,
                    events=[],
                    causal_dependencies=[]
                )
            
            # Get current vector clock
            vector_clock = self.vector_clocks[agent_id]
            
            # Increment logical time (Lamport timestamp)
            vector_clock.logical_time += 1
            vector_clock.physical_time = current_time
            
            # Extract causal dependencies from event data
            causal_dependencies = []
            if event_data:
                # Look for references to other agents or events
                dependencies = event_data.get('depends_on', [])
                causal_dependencies.extend(dependencies)
                
                # Update logical time based on dependencies
                for dep_agent_id in dependencies:
                    if dep_agent_id in self.vector_clocks:
                        dep_clock = self.vector_clocks[dep_agent_id]
                        # Lamport clock update rule
                        vector_clock.logical_time = max(
                            vector_clock.logical_time, 
                            dep_clock.logical_time + 1
                        )
            
            # Create temporal event
            temporal_event = TemporalEvent(
                id=f'event_{uuid.uuid4().hex[:8]}',
                agent_id=agent_id,
                event_type=event_type,
                logical_time=vector_clock.logical_time,
                physical_time=current_time,
                scale='second',  # Default scale
                phase=self._calculate_current_phase(current_time, 'second'),
                causal_dependencies=causal_dependencies,
                data=event_data or {}
            )
            
            # Add event to vector clock
            vector_clock.events.append({
                'event_id': temporal_event.id,
                'type': event_type,
                'logical_time': vector_clock.logical_time,
                'physical_time': current_time,
                'causal_dependencies': causal_dependencies
            })
            
            # Store temporal event
            self.temporal_events.append(temporal_event)
            
            # Cleanup old events
            self._cleanup_old_events(agent_id)
            
            # Update causal consistency metrics
            causal_consistency = await self._analyze_causal_consistency(agent_id)
            
            logger.debug(f"Updated vector clock for {agent_id}: logical_time={vector_clock.logical_time}")
            
            return {
                'agent_id': agent_id,
                'logical_time': vector_clock.logical_time,
                'physical_time': current_time,
                'event_id': temporal_event.id,
                'causal_dependencies': causal_dependencies,
                'causal_consistency': causal_consistency,
                'total_events': len(vector_clock.events),
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error(f"Vector clock update failed for {agent_id}: {e}")
            return {'error': str(e), 'updated': False}
    
    async def analyze_temporal_correlations(self, agent_data: Dict, 
                                          analysis_window: float = 300.0) -> Dict:
        """
        Analyze correlations across temporal scales using eigenvalue decomposition
        
        Args:
            agent_data: Dictionary of agent temporal data
            analysis_window: Time window for correlation analysis (seconds)
            
        Returns:
            Comprehensive temporal correlation analysis
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - analysis_window
            
            correlation_results = {}
            eigenvalue_results = {}
            
            # Analyze each temporal scale
            for scale in self.temporal_scales.keys():
                scale_correlations = {}
                agent_time_series = {}
                
                # Extract time series data for each agent
                for agent_id, data in agent_data.items():
                    if scale not in data:
                        continue
                    
                    # Filter data within analysis window
                    scale_data = data[scale]
                    if isinstance(scale_data, list):
                        # Assume it's a time series
                        recent_data = [x for x in scale_data[-100:]]  # Last 100 points
                        if len(recent_data) >= 2:
                            agent_time_series[agent_id] = recent_data
                
                if len(agent_time_series) < 2:
                    logger.debug(f"Insufficient data for correlation analysis in scale {scale}")
                    continue
                
                # Calculate autocorrelations for each agent
                agent_autocorrs = {}
                for agent_id, time_series in agent_time_series.items():
                    autocorr = self._calculate_autocorrelation(time_series)
                    agent_autocorrs[agent_id] = autocorr
                    scale_correlations[agent_id] = autocorr
                
                # Create correlation matrix between agents
                agent_ids = list(agent_autocorrs.keys())
                n_agents = len(agent_ids)
                
                if n_agents >= 2:
                    # Build correlation matrix
                    correlation_matrix = np.zeros((n_agents, n_agents))
                    
                    for i, agent_i in enumerate(agent_ids):
                        for j, agent_j in enumerate(agent_ids):
                            if i == j:
                                correlation_matrix[i, j] = 1.0
                            else:
                                # Cross-correlation between agents
                                series_i = agent_time_series[agent_i]
                                series_j = agent_time_series[agent_j]
                                
                                # Pad series to same length
                                min_len = min(len(series_i), len(series_j))
                                if min_len >= 2:
                                    cross_corr = np.corrcoef(
                                        series_i[-min_len:], 
                                        series_j[-min_len:]
                                    )[0, 1]
                                    
                                    if not np.isnan(cross_corr):
                                        correlation_matrix[i, j] = cross_corr
                    
                    # Store correlation matrix
                    self.correlation_matrices[scale] = correlation_matrix
                    
                    # Eigenvalue decomposition
                    try:
                        eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
                        eigenvalues = np.real(eigenvalues)  # Take real part
                        
                        # Store eigenvalue history
                        self.eigenvalue_history[scale].append(eigenvalues)
                        
                        eigenvalue_results[scale] = {
                            'eigenvalues': eigenvalues.tolist(),
                            'dominant_eigenvalue': float(np.max(eigenvalues)),
                            'spectral_radius': float(np.max(np.abs(eigenvalues))),
                            'trace': float(np.trace(correlation_matrix)),
                            'determinant': float(np.linalg.det(correlation_matrix)),
                            'condition_number': float(np.linalg.cond(correlation_matrix)),
                            'rank': int(np.linalg.matrix_rank(correlation_matrix))
                        }
                        
                    except np.linalg.LinAlgError as e:
                        logger.warning(f"Eigenvalue decomposition failed for scale {scale}: {e}")
                        eigenvalue_results[scale] = {'error': 'decomposition_failed'}
                
                correlation_results[scale] = scale_correlations
            
            # Calculate overall temporal stability
            temporal_stability = self._calculate_temporal_stability(eigenvalue_results)
            
            # Analyze dominant temporal modes
            dominant_modes = self._analyze_dominant_temporal_modes(eigenvalue_results)
            
            # Calculate system dynamics metrics
            system_dynamics = self._analyze_system_dynamics(eigenvalue_results)
            
            logger.info(f"Analyzed temporal correlations across {len(correlation_results)} scales")
            
            return {
                'correlation_analysis': correlation_results,
                'eigenvalue_analysis': eigenvalue_results,
                'temporal_stability': temporal_stability,
                'dominant_modes': dominant_modes,
                'system_dynamics': system_dynamics,
                'analysis_window': analysis_window,
                'scales_analyzed': list(correlation_results.keys()),
                'timestamp': current_time
            }
            
        except Exception as e:
            logger.error(f"Temporal correlation analysis failed: {e}")
            return {'error': str(e), 'analysis_completed': False}
    
    async def synchronize_agents_temporally(self, agent_ids: List[str], 
                                          target_coherence: float = 0.9) -> Dict:
        """
        Synchronize agents across multiple temporal scales
        
        Args:
            agent_ids: List of agent identifiers to synchronize
            target_coherence: Target coherence level (0-1)
            
        Returns:
            Synchronization results and recommendations
        """
        try:
            synchronization_results = {}
            
            for scale in self.temporal_scales.keys():
                # Get current phases for agents
                agent_phases = []
                valid_agents = []
                
                for agent_id in agent_ids:
                    if agent_id in self.temporal_phases and scale in self.temporal_phases[agent_id]:
                        phase = self.temporal_phases[agent_id][scale].phase
                        agent_phases.append(phase)
                        valid_agents.append(agent_id)
                
                if len(agent_phases) < 2:
                    continue
                
                # Calculate current coherence
                current_coherence = await self._calculate_scale_coherence(agent_phases)
                
                # Determine synchronization strategy
                if current_coherence < target_coherence:
                    # Apply phase correction
                    phase_corrections = self._calculate_phase_corrections(
                        agent_phases, target_coherence
                    )
                    
                    synchronization_results[scale] = {
                        'current_coherence': current_coherence,
                        'target_coherence': target_coherence,
                        'synchronization_needed': True,
                        'phase_corrections': dict(zip(valid_agents, phase_corrections)),
                        'improvement_potential': target_coherence - current_coherence
                    }
                else:
                    synchronization_results[scale] = {
                        'current_coherence': current_coherence,
                        'target_coherence': target_coherence,
                        'synchronization_needed': False,
                        'status': 'already_synchronized'
                    }
            
            # Calculate overall synchronization quality
            overall_quality = self._assess_overall_synchronization(synchronization_results)
            
            return {
                'synchronization_results': synchronization_results,
                'overall_synchronization_quality': overall_quality,
                'agents_synchronized': len(agent_ids),
                'scales_processed': len(synchronization_results),
                'target_coherence': target_coherence,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Temporal synchronization failed: {e}")
            return {'error': str(e), 'synchronized': False}
    
    # Helper methods
    def _calculate_current_phase(self, timestamp: float, scale: str) -> float:
        """Calculate current phase for a temporal scale"""
        if scale not in self.temporal_scales:
            return 0.0
        
        scale_duration = self.temporal_scales[scale]
        
        # Calculate phase based on position within scale cycle
        cycle_position = (timestamp % scale_duration) / scale_duration
        phase = 2 * np.pi * cycle_position
        
        return phase
    
    def _calculate_autocorrelation(self, time_series: List[float], lag: int = 1) -> float:
        """Calculate autocorrelation of time series"""
        if len(time_series) <= lag:
            return 0.0
        
        try:
            series = np.array(time_series)
            if len(series) <= lag:
                return 0.0
            
            # Calculate autocorrelation at specified lag
            n = len(series)
            series_mean = np.mean(series)
            
            numerator = np.sum((series[:-lag] - series_mean) * (series[lag:] - series_mean))
            denominator = np.sum((series - series_mean) ** 2)
            
            if denominator == 0:
                return 0.0
            
            autocorr = numerator / denominator
            return float(autocorr)
            
        except Exception:
            return 0.0
    
    async def _calculate_scale_coherence(self, phases: List[float]) -> float:
        """Calculate coherence for a single scale"""
        if len(phases) < 2:
            return 0.0
        
        complex_phases = [np.exp(1j * phase) for phase in phases]
        mean_complex_phase = np.mean(complex_phases)
        coherence = abs(mean_complex_phase)
        
        return float(coherence)
    
    def _calculate_phase_corrections(self, phases: List[float], 
                                   target_coherence: float) -> List[float]:
        """Calculate phase corrections needed to achieve target coherence"""
        if len(phases) < 2:
            return [0.0] * len(phases)
        
        # Calculate mean phase
        complex_phases = [np.exp(1j * phase) for phase in phases]
        mean_complex_phase = np.mean(complex_phases)
        target_phase = np.angle(mean_complex_phase)
        
        # Calculate corrections to align phases toward mean
        corrections = []
        for phase in phases:
            # Calculate angular difference
            diff = np.angle(np.exp(1j * (target_phase - phase)))
            # Apply partial correction based on target coherence
            correction = diff * target_coherence
            corrections.append(float(correction))
        
        return corrections
    
    def _assess_synchronization_quality(self, coherence: float, 
                                      phase_variance: float) -> str:
        """Assess quality of synchronization"""
        if coherence > 0.95 and phase_variance < 0.1:
            return "excellent"
        elif coherence > 0.85 and phase_variance < 0.3:
            return "good"
        elif coherence > 0.7 and phase_variance < 0.5:
            return "moderate"
        elif coherence > 0.5:
            return "poor"
        else:
            return "unsynchronized"
    
    def _assess_overall_coherence_quality(self, overall_coherence: float) -> str:
        """Assess overall coherence quality"""
        if overall_coherence > 0.95:
            return "exceptional"
        elif overall_coherence > 0.85:
            return "excellent"
        elif overall_coherence > 0.75:
            return "good"
        elif overall_coherence > 0.6:
            return "moderate"
        else:
            return "poor"
    
    def _assess_overall_synchronization(self, sync_results: Dict) -> Dict:
        """Assess overall synchronization quality"""
        if not sync_results:
            return {"quality": "no_data", "score": 0.0}
        
        # Calculate average coherence across scales
        coherences = []
        needs_sync_count = 0
        
        for scale_result in sync_results.values():
            coherences.append(scale_result['current_coherence'])
            if scale_result.get('synchronization_needed', False):
                needs_sync_count += 1
        
        avg_coherence = np.mean(coherences)
        sync_ratio = 1.0 - (needs_sync_count / len(sync_results))
        
        overall_score = 0.7 * avg_coherence + 0.3 * sync_ratio
        
        if overall_score > 0.9:
            quality = "excellent"
        elif overall_score > 0.8:
            quality = "good"
        elif overall_score > 0.7:
            quality = "moderate"
        else:
            quality = "poor"
        
        return {
            "quality": quality,
            "score": float(overall_score),
            "average_coherence": float(avg_coherence),
            "synchronization_ratio": float(sync_ratio),
            "scales_needing_sync": needs_sync_count
        }
    
    def _calculate_temporal_stability(self, eigenvalue_results: Dict) -> Dict:
        """Calculate overall temporal stability from eigenvalues"""
        if not eigenvalue_results:
            return {"stability": 0.0, "quality": "unknown"}
        
        stability_scores = []
        
        for scale, results in eigenvalue_results.items():
            if 'error' in results:
                continue
            
            eigenvalues = results['eigenvalues']
            if eigenvalues:
                # Stability based on eigenvalue distribution
                max_eigenvalue = max(eigenvalues)
                eigenvalue_spread = max(eigenvalues) - min(eigenvalues)
                
                # Normalize stability score
                stability = 1.0 / (1.0 + eigenvalue_spread)
                stability_scores.append(stability)
        
        if stability_scores:
            overall_stability = np.mean(stability_scores)
            if overall_stability > 0.8:
                quality = "high"
            elif overall_stability > 0.6:
                quality = "medium"
            else:
                quality = "low"
        else:
            overall_stability = 0.0
            quality = "unknown"
        
        return {
            "stability": float(overall_stability),
            "quality": quality,
            "scales_analyzed": len(stability_scores)
        }
    
    def _analyze_dominant_temporal_modes(self, eigenvalue_results: Dict) -> Dict:
        """Analyze dominant temporal modes from eigenvalue analysis"""
        dominant_modes = {}
        
        for scale, results in eigenvalue_results.items():
            if 'error' in results:
                continue
            
            eigenvalues = results['eigenvalues']
            if eigenvalues:
                max_eigenvalue = max(eigenvalues)
                dominant_index = eigenvalues.index(max_eigenvalue)
                
                dominant_modes[scale] = {
                    "eigenvalue": max_eigenvalue,
                    "mode_index": dominant_index,
                    "dominance_ratio": max_eigenvalue / sum(eigenvalues) if sum(eigenvalues) > 0 else 0.0
                }
        
        return dominant_modes
    
    def _analyze_system_dynamics(self, eigenvalue_results: Dict) -> Dict:
        """Analyze overall system dynamics"""
        if not eigenvalue_results:
            return {"dynamics": "unknown"}
        
        # Collect all eigenvalues
        all_eigenvalues = []
        for results in eigenvalue_results.values():
            if 'eigenvalues' in results:
                all_eigenvalues.extend(results['eigenvalues'])
        
        if not all_eigenvalues:
            return {"dynamics": "no_data"}
        
        # Analyze eigenvalue spectrum
        max_eigenvalue = max(all_eigenvalues)
        min_eigenvalue = min(all_eigenvalues)
        eigenvalue_range = max_eigenvalue - min_eigenvalue
        
        # Determine system dynamics type
        if max_eigenvalue > 1.0:
            dynamics_type = "expanding"
        elif max_eigenvalue < 0.5:
            dynamics_type = "contracting"
        else:
            dynamics_type = "stable"
        
        return {
            "dynamics": dynamics_type,
            "max_eigenvalue": float(max_eigenvalue),
            "min_eigenvalue": float(min_eigenvalue),
            "eigenvalue_range": float(eigenvalue_range),
            "spectral_complexity": len(set(np.round(all_eigenvalues, 3)))
        }
    
    async def _update_temporal_phases(self, scale: str, phases: List[float], 
                                    mean_phase: float):
        """Update stored temporal phases"""
        with self._phase_update_lock:
            current_time = time.time()
            
            # Update mean phase for scale
            if 'system' not in self.temporal_phases:
                self.temporal_phases['system'] = {}
            
            self.temporal_phases['system'][scale] = TemporalPhase(
                scale=scale,
                phase=mean_phase,
                frequency=1.0 / self.temporal_scales[scale],
                amplitude=1.0,
                timestamp=current_time,
                agent_id='system'
            )
    
    async def _analyze_cross_scale_coherence(self, coherence_results: Dict) -> Dict:
        """Analyze coherence relationships across temporal scales"""
        if len(coherence_results) < 2:
            return {"cross_scale_analysis": "insufficient_scales"}
        
        scales = list(coherence_results.keys())
        coherences = [coherence_results[scale]['coherence'] for scale in scales]
        
        # Calculate cross-scale correlation
        cross_scale_correlation = np.corrcoef(coherences, coherences)[0, 1] if len(coherences) > 1 else 0.0
        
        # Find most/least coherent scales
        max_coherence_scale = scales[np.argmax(coherences)]
        min_coherence_scale = scales[np.argmin(coherences)]
        
        return {
            "cross_scale_correlation": float(cross_scale_correlation),
            "most_coherent_scale": max_coherence_scale,
            "least_coherent_scale": min_coherence_scale,
            "coherence_range": float(max(coherences) - min(coherences)),
            "scale_consistency": "high" if cross_scale_correlation > 0.7 else "low"
        }
    
    async def _analyze_causal_consistency(self, agent_id: str) -> Dict:
        """Analyze causal consistency for an agent"""
        if agent_id not in self.vector_clocks:
            return {"consistency": "unknown"}
        
        vector_clock = self.vector_clocks[agent_id]
        
        # Check for causal violations in recent events
        violations = 0
        total_checks = 0
        
        recent_events = vector_clock.events[-50:]  # Last 50 events
        
        for i, event in enumerate(recent_events):
            for dep_agent in event.get('causal_dependencies', []):
                if dep_agent in self.vector_clocks:
                    dep_clock = self.vector_clocks[dep_agent]
                    
                    # Check if dependency's logical time is less than current
                    if dep_clock.logical_time >= event['logical_time']:
                        violations += 1
                    total_checks += 1
        
        if total_checks > 0:
            consistency_ratio = 1.0 - (violations / total_checks)
        else:
            consistency_ratio = 1.0
        
        return {
            "consistency_ratio": consistency_ratio,
            "causal_violations": violations,
            "total_causal_checks": total_checks,
            "consistency_quality": "high" if consistency_ratio > 0.95 else "medium" if consistency_ratio > 0.8 else "low"
        }
    
    # Cleanup methods
    def _cleanup_old_events(self, agent_id: str):
        """Clean up old events to prevent memory growth"""
        if agent_id in self.vector_clocks:
            vector_clock = self.vector_clocks[agent_id]
            if len(vector_clock.events) > self.max_events_per_agent:
                # Keep only recent events
                vector_clock.events = vector_clock.events[-self.max_events_per_agent:]
    
    def _cleanup_coherence_history(self):
        """Clean up old coherence history"""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep last hour
        
        self.coherence_history = [
            analysis for analysis in self.coherence_history
            if analysis.timestamp > cutoff_time
        ]
    
    # Public interface methods
    async def get_temporal_statistics(self) -> Dict:
        """Get comprehensive temporal system statistics"""
        current_time = time.time()
        
        # Count active components
        active_agents = len(self.vector_clocks)
        total_events = sum(len(clock.events) for clock in self.vector_clocks.values())
        active_scales = len(set(
            scale for phases in self.temporal_phases.values() 
            for scale in phases.keys()
        ))
        
        # Calculate average coherence
        if self.coherence_history:
            recent_coherences = [
                analysis.coherence for analysis in self.coherence_history
                if current_time - analysis.timestamp < 300  # Last 5 minutes
            ]
            avg_coherence = np.mean(recent_coherences) if recent_coherences else 0.0
        else:
            avg_coherence = 0.0
        
        return {
            "active_agents": active_agents,
            "total_temporal_events": total_events,
            "active_temporal_scales": active_scales,
            "available_scales": list(self.temporal_scales.keys()),
            "average_coherence": float(avg_coherence),
            "coherence_measurements": len(self.coherence_history),
            "correlation_matrices": len(self.correlation_matrices),
            "system_uptime": current_time - (self._last_phase_update or current_time),
            "temporal_resolution": min(self.temporal_scales.values()),
            "temporal_span": max(self.temporal_scales.values()),
            "system_status": "synchronized" if avg_coherence > 0.8 else "partial" if avg_coherence > 0.5 else "unsynchronized"
        }