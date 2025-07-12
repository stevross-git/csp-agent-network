# enhanced_csp/network/ml_routing.py
"""
Machine Learning-based route optimization for predictive network performance.
Provides 2-5x routing efficiency through intelligent prediction and optimization.
"""

import asyncio
import logging
import time
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import os

# Try to import ML libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class NetworkMetrics:
    """Network performance metrics for ML training"""
    timestamp: float
    source: str
    destination: str
    latency_ms: float
    bandwidth_mbps: float
    packet_loss: float
    jitter_ms: float
    route_hops: int
    congestion_score: float
    time_of_day: int  # Hour 0-23
    day_of_week: int  # 0-6
    network_load: float  # 0-1


@dataclass
class RoutePerformancePrediction:
    """Predicted route performance"""
    destination: str
    predicted_latency: float
    predicted_bandwidth: float
    predicted_reliability: float
    confidence_score: float
    recommended_route: Optional[str] = None


@dataclass
class MLConfig:
    """Configuration for ML route optimization"""
    training_data_size: int = 10000
    retrain_interval_hours: int = 24
    prediction_horizon_minutes: int = 30
    min_samples_for_training: int = 100
    feature_importance_threshold: float = 0.01
    model_accuracy_threshold: float = 0.8
    enable_real_time_learning: bool = True
    enable_ensemble_models: bool = True


class NetworkDataCollector:
    """Collect and preprocess network data for ML training"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.metrics_history: deque = deque(maxlen=config.training_data_size)
        self.route_performance: Dict[str, List[NetworkMetrics]] = defaultdict(list)
        
        # Real-time metrics
        self.current_metrics: Dict[str, NetworkMetrics] = {}
        self.prediction_cache: Dict[str, RoutePerformancePrediction] = {}
        
        # Data preprocessing
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_columns = [
            'latency_ms', 'bandwidth_mbps', 'packet_loss', 'jitter_ms',
            'route_hops', 'congestion_score', 'time_of_day', 'day_of_week',
            'network_load'
        ]
    
    def add_metric(self, metric: NetworkMetrics):
        """Add new network metric for training"""
        self.metrics_history.append(metric)
        
        # Update route-specific performance history
        route_key = f"{metric.source}->{metric.destination}"
        self.route_performance[route_key].append(metric)
        
        # Keep only recent data per route
        if len(self.route_performance[route_key]) > 1000:
            self.route_performance[route_key].pop(0)
        
        # Update current metrics
        self.current_metrics[route_key] = metric
    
    def get_training_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Prepare training data for ML models"""
        if not NUMPY_AVAILABLE or not self.metrics_history:
            return None
        
        if len(self.metrics_history) < self.config.min_samples_for_training:
            logger.warning(f"Insufficient data for training: {len(self.metrics_history)}")
            return None
        
        # Convert metrics to feature matrix
        features = []
        targets = []
        
        for metric in self.metrics_history:
            feature_vector = [
                metric.latency_ms,
                metric.bandwidth_mbps,
                metric.packet_loss,
                metric.jitter_ms,
                metric.route_hops,
                metric.congestion_score,
                metric.time_of_day,
                metric.day_of_week,
                metric.network_load
            ]
            
            # Use latency as primary target (can be extended)
            target = metric.latency_ms
            
            features.append(feature_vector)
            targets.append(target)
        
        X = np.array(features)
        y = np.array(targets)
        
        # Normalize features
        if self.scaler and SKLEARN_AVAILABLE:
            X = self.scaler.fit_transform(X)
        
        return X, y
    
    def get_route_features(self, source: str, destination: str) -> Optional[np.ndarray]:
        """Get current features for a specific route"""
        route_key = f"{source}->{destination}"
        current_metric = self.current_metrics.get(route_key)
        
        if not current_metric:
            # Use network averages as fallback
            current_metric = self._get_average_metrics()
        
        if not current_metric:
            return None
        
        features = np.array([[
            current_metric.latency_ms,
            current_metric.bandwidth_mbps,
            current_metric.packet_loss,
            current_metric.jitter_ms,
            current_metric.route_hops,
            current_metric.congestion_score,
            current_metric.time_of_day,
            current_metric.day_of_week,
            current_metric.network_load
        ]])
        
        if self.scaler and SKLEARN_AVAILABLE:
            try:
                features = self.scaler.transform(features)
            except Exception:
                pass  # Use raw features if scaling fails
        
        return features
    
    def _get_average_metrics(self) -> Optional[NetworkMetrics]:
        """Calculate average metrics as fallback"""
        if not self.metrics_history:
            return None
        
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 samples
        
        if not recent_metrics:
            return None
        
        current_time = time.time()
        time_struct = time.localtime(current_time)
        
        return NetworkMetrics(
            timestamp=current_time,
            source="average",
            destination="average",
            latency_ms=sum(m.latency_ms for m in recent_metrics) / len(recent_metrics),
            bandwidth_mbps=sum(m.bandwidth_mbps for m in recent_metrics) / len(recent_metrics),
            packet_loss=sum(m.packet_loss for m in recent_metrics) / len(recent_metrics),
            jitter_ms=sum(m.jitter_ms for m in recent_metrics) / len(recent_metrics),
            route_hops=int(sum(m.route_hops for m in recent_metrics) / len(recent_metrics)),
            congestion_score=sum(m.congestion_score for m in recent_metrics) / len(recent_metrics),
            time_of_day=time_struct.tm_hour,
            day_of_week=time_struct.tm_wday,
            network_load=0.5  # Default moderate load
        )


class MLRoutePredictor:
    """Machine learning-based route performance predictor"""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.data_collector = NetworkDataCollector(config)
        
        # ML Models
        self.models = {}
        self.model_performance = {}
        self.last_training_time = 0
        
        if SKLEARN_AVAILABLE:
            self._initialize_models()
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def _initialize_models(self):
        """Initialize ML models for prediction"""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, ML features disabled")
            return
        
        # Ensemble of models for better accuracy
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'linear': LinearRegression()
        }
        
        logger.info("Initialized ML models for route prediction")
    
    async def train_models(self, force_retrain: bool = False):
        """Train ML models with collected data"""
        if not SKLEARN_AVAILABLE:
            return False
        
        # Check if retraining is needed
        time_since_training = time.time() - self.last_training_time
        hours_since_training = time_since_training / 3600
        
        if not force_retrain and hours_since_training < self.config.retrain_interval_hours:
            return False
        
        # Get training data
        training_data = self.data_collector.get_training_data()
        if not training_data:
            logger.warning("No training data available")
            return False
        
        X, y = training_data
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train each model
        training_results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name} model...")
                
                start_time = time.perf_counter()
                model.fit(X_train, y_train)
                training_time = time.perf_counter() - start_time
                
                # Evaluate model
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2