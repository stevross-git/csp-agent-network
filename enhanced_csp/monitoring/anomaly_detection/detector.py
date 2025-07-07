"""
Anomaly Detection System using Machine Learning
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import pickle
import os

from prometheus_client import Gauge, Counter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

logger = logging.getLogger(__name__)

# Metrics for anomaly detection
anomaly_score = Gauge(
    'csp_anomaly_score',
    'Anomaly detection score (0-1)',
    ['service', 'metric_type', 'algorithm']
)

anomalies_detected = Counter(
    'csp_anomalies_detected_total',
    'Total anomalies detected',
    ['service', 'metric_type', 'severity']
)

anomaly_detection_duration = Gauge(
    'csp_anomaly_detection_duration_seconds',
    'Time taken for anomaly detection'
)

class AnomalyDetector:
    """ML-based anomaly detection for monitoring metrics"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_dir = "monitoring/anomaly_detection/models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Configuration
        self.config = {
            'contamination': 0.05,  # Expected proportion of anomalies
            'n_estimators': 100,
            'max_samples': 'auto',
            'random_state': 42
        }
        
        # Metrics to monitor
        self.monitored_metrics = [
            {
                'name': 'api_request_duration_seconds',
                'features': ['p50', 'p95', 'p99', 'rate'],
                'threshold': 0.8
            },
            {
                'name': 'error_rate',
                'features': ['rate', 'delta'],
                'threshold': 0.9
            },
            {
                'name': 'cpu_usage',
                'features': ['avg', 'max', 'std'],
                'threshold': 0.85
            },
            {
                'name': 'memory_usage',
                'features': ['current', 'rate'],
                'threshold': 0.85
            },
            {
                'name': 'database_connections',
                'features': ['active', 'waiting', 'rate'],
                'threshold': 0.9
            }
        ]
    
    async def initialize(self):
        """Initialize anomaly detection models"""
        logger.info("Initializing anomaly detection models...")
        
        # Load existing models or create new ones
        for metric in self.monitored_metrics:
            model_path = os.path.join(self.model_dir, f"{metric['name']}.pkl")
            scaler_path = os.path.join(self.model_dir, f"{metric['name']}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[metric['name']] = joblib.load(model_path)
                self.scalers[metric['name']] = joblib.load(scaler_path)
                logger.info(f"Loaded existing model for {metric['name']}")
            else:
                self.models[metric['name']] = IsolationForest(**self.config)
                self.scalers[metric['name']] = StandardScaler()
                logger.info(f"Created new model for {metric['name']}")
    
    async def fetch_metric_data(self, metric_name: str, duration: str = "1h") -> pd.DataFrame:
        """Fetch metric data from Prometheus"""
        import aiohttp
        
        queries = {
            'api_request_duration_seconds': [
                f'histogram_quantile(0.5, rate({metric_name}_bucket[5m]))',
                f'histogram_quantile(0.95, rate({metric_name}_bucket[5m]))',
                f'histogram_quantile(0.99, rate({metric_name}_bucket[5m]))',
                f'rate({metric_name}_count[5m])'
            ],
            'error_rate': [
                'rate(http_requests_total{status=~"5.."}[5m])',
                'delta(http_requests_total{status=~"5.."}[5m])'
            ],
            'cpu_usage': [
                'avg(rate(process_cpu_seconds_total[5m]))',
                'max(rate(process_cpu_seconds_total[5m]))',
                'stddev(rate(process_cpu_seconds_total[5m]))'
            ],
            'memory_usage': [
                'process_resident_memory_bytes',
                'rate(process_resident_memory_bytes[5m])'
            ],
            'database_connections': [
                'pg_stat_activity_count{state="active"}',
                'pg_stat_activity_count{state="waiting"}',
                'rate(pg_stat_activity_count[5m])'
            ]
        }
        
        if metric_name not in queries:
            return pd.DataFrame()
        
        data = []
        async with aiohttp.ClientSession() as session:
            for query in queries[metric_name]:
                url = f"{self.prometheus_url}/api/v1/query_range"
                params = {
                    'query': query,
                    'start': (datetime.now() - timedelta(hours=1)).timestamp(),
                    'end': datetime.now().timestamp(),
                    'step': '15s'
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result['data']['result']:
                            values = result['data']['result'][0]['values']
                            data.append([float(v[1]) for v in values])
        
        if data:
            return pd.DataFrame(data).T
        return pd.DataFrame()
    
    async def detect_anomalies(self, metric_name: str) -> Dict[str, Any]:
        """Detect anomalies in a specific metric"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Fetch recent data
            data = await self.fetch_metric_data(metric_name)
            
            if data.empty or len(data) < 10:
                logger.warning(f"Insufficient data for {metric_name}")
                return {'status': 'insufficient_data'}
            
            # Prepare features
            X = data.values
            
            # Scale features
            if metric_name not in self.scalers:
                self.scalers[metric_name] = StandardScaler()
                X_scaled = self.scalers[metric_name].fit_transform(X)
            else:
                X_scaled = self.scalers[metric_name].transform(X)
            
            # Detect anomalies
            if metric_name not in self.models:
                self.models[metric_name] = IsolationForest(**self.config)
                predictions = self.models[metric_name].fit_predict(X_scaled)
            else:
                predictions = self.models[metric_name].predict(X_scaled)
            
            # Calculate anomaly scores
            scores = self.models[metric_name].score_samples(X_scaled)
            anomaly_score_normalized = 1 - (scores - scores.min()) / (scores.max() - scores.min())
            
            # Find anomalies
            anomaly_indices = np.where(predictions == -1)[0]
            anomaly_timestamps = [
                datetime.now() - timedelta(seconds=15 * (len(predictions) - i))
                for i in anomaly_indices
            ]
            
            # Calculate metrics
            current_score = float(anomaly_score_normalized[-1]) if len(anomaly_score_normalized) > 0 else 0
            anomaly_rate = len(anomaly_indices) / len(predictions)
            
            # Update Prometheus metrics
            anomaly_score.labels(
                service="api",
                metric_type=metric_name,
                algorithm="isolation_forest"
            ).set(current_score)
            
            # Determine severity
            metric_config = next(m for m in self.monitored_metrics if m['name'] == metric_name)
            severity = self._determine_severity(current_score, metric_config['threshold'])
            
            if anomaly_indices.size > 0:
                anomalies_detected.labels(
                    service="api",
                    metric_type=metric_name,
                    severity=severity
                ).inc()
            
            result = {
                'status': 'success',
                'metric': metric_name,
                'current_score': current_score,
                'anomaly_rate': anomaly_rate,
                'anomalies': [
                    {
                        'timestamp': ts.isoformat(),
                        'score': float(anomaly_score_normalized[idx]),
                        'severity': severity
                    }
                    for idx, ts in zip(anomaly_indices, anomaly_timestamps)
                ],
                'model_info': {
                    'contamination': self.config['contamination'],
                    'samples_analyzed': len(predictions)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for {metric_name}: {str(e)}")
            return {'status': 'error', 'error': str(e)}
        
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            anomaly_detection_duration.set(duration)
    
    def _determine_severity(self, score: float, threshold: float) -> str:
        """Determine anomaly severity based on score"""
        if score >= threshold * 1.2:
            return "critical"
        elif score >= threshold:
            return "high"
        elif score >= threshold * 0.8:
            return "medium"
        else:
            return "low"
    
    async def train_models(self):
        """Train models on historical data"""
        logger.info("Training anomaly detection models...")
        
        for metric in self.monitored_metrics:
            try:
                # Fetch training data (last 24 hours)
                data = await self.fetch_metric_data(metric['name'], duration="24h")
                
                if data.empty or len(data) < 100:
                    logger.warning(f"Insufficient training data for {metric['name']}")
                    continue
                
                # Prepare and scale features
                X = data.values
                X_scaled = self.scalers[metric['name']].fit_transform(X)
                
                # Train model
                self.models[metric['name']].fit(X_scaled)
                
                # Save model and scaler
                model_path = os.path.join(self.model_dir, f"{metric['name']}.pkl")
                scaler_path = os.path.join(self.model_dir, f"{metric['name']}_scaler.pkl")
                
                joblib.dump(self.models[metric['name']], model_path)
                joblib.dump(self.scalers[metric['name']], scaler_path)
                
                logger.info(f"Trained and saved model for {metric['name']}")
                
            except Exception as e:
                logger.error(f"Error training model for {metric['name']}: {str(e)}")
    
    async def run_continuous_detection(self, interval: int = 60):
        """Run continuous anomaly detection"""
        logger.info(f"Starting continuous anomaly detection (interval: {interval}s)")
        
        while True:
            try:
                for metric in self.monitored_metrics:
                    result = await self.detect_anomalies(metric['name'])
                    
                    if result['status'] == 'success' and result['anomalies']:
                        logger.warning(
                            f"Anomalies detected in {metric['name']}: "
                            f"{len(result['anomalies'])} anomalies, "
                            f"current score: {result['current_score']:.3f}"
                        )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in continuous detection: {str(e)}")
                await asyncio.sleep(interval)

# Create service runner
async def run_anomaly_detection():
    """Run anomaly detection service"""
    detector = AnomalyDetector()
    await detector.initialize()
    
    # Train models initially
    await detector.train_models()
    
    # Run continuous detection
    await detector.run_continuous_detection()

if __name__ == "__main__":
    asyncio.run(run_anomaly_detection())
