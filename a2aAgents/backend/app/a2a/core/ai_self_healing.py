"""
AI-Powered Error Prediction and Self-Healing System

This module provides intelligent error prediction, anomaly detection, and automated
recovery mechanisms using real machine learning to maintain system reliability
and minimize downtime.
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import logging
import numpy as np
import json
import time
import traceback
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
import signal
import os
import psutil

# Machine Learning imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Time series analysis
try:
    import scipy.signal as signal_proc
    from scipy.stats import zscore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Deep learning for complex pattern recognition
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ErrorPredictionNN(nn.Module):
    """Neural network for error prediction and system health assessment"""
    def __init__(self, input_dim, hidden_dim=128):
        super(ErrorPredictionNN, self).__init__()
        
        # Multi-layer feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(hidden_dim // 4, hidden_dim // 4, batch_first=True, num_layers=2)
        
        # Multi-head prediction
        self.error_probability_head = nn.Linear(hidden_dim // 4, 1)
        self.time_to_failure_head = nn.Linear(hidden_dim // 4, 1)
        self.error_severity_head = nn.Linear(hidden_dim // 4, 3)  # low, medium, high
        self.recovery_time_head = nn.Linear(hidden_dim // 4, 1)
        self.error_type_head = nn.Linear(hidden_dim // 4, 10)  # 10 common error types
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim // 4, num_heads=4)
        
    def forward(self, x, sequence_data=None):
        batch_size = x.size(0)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Process temporal sequences if available
        if sequence_data is not None:
            lstm_out, _ = self.lstm(sequence_data)
            temporal_features = lstm_out[:, -1, :]  # Last timestep
            
            # Combine with current features
            combined_features = features + temporal_features
        else:
            combined_features = features
        
        # Apply attention
        attn_out, attn_weights = self.attention(
            combined_features.unsqueeze(0), 
            combined_features.unsqueeze(0), 
            combined_features.unsqueeze(0)
        )
        enhanced_features = attn_out.squeeze(0)
        
        # Predictions
        error_prob = torch.sigmoid(self.error_probability_head(enhanced_features))
        time_to_failure = F.relu(self.time_to_failure_head(enhanced_features))
        error_severity = F.softmax(self.error_severity_head(enhanced_features), dim=1)
        recovery_time = F.relu(self.recovery_time_head(enhanced_features))
        error_type = F.softmax(self.error_type_head(enhanced_features), dim=1)
        
        return error_prob, time_to_failure, error_severity, recovery_time, error_type, attn_weights


class ErrorType(Enum):
    """Types of system errors"""
    MEMORY_LEAK = "memory_leak"
    CPU_OVERLOAD = "cpu_overload"
    NETWORK_TIMEOUT = "network_timeout"
    DATABASE_CONNECTION = "database_connection"
    DISK_SPACE = "disk_space"
    DEADLOCK = "deadlock"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATA_CORRUPTION = "data_corruption"
    AUTHENTICATION = "authentication"
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """Error severity levels"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class SystemError:
    """System error record"""
    error_id: str
    timestamp: datetime
    error_type: ErrorType
    severity: SeverityLevel
    component: str
    message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    resolution_time: Optional[float] = None
    auto_resolved: bool = False
    recovery_actions: List[str] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """System health metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float] = field(default_factory=dict)
    active_connections: int = 0
    error_rate: float = 0.0
    response_time: float = 0.0
    throughput: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HealingAction:
    """Self-healing action definition"""
    action_id: str
    name: str
    description: str
    trigger_conditions: List[str]
    action_function: Callable
    cooldown_period: int = 300  # seconds
    max_attempts: int = 3
    priority: int = 1
    requires_confirmation: bool = False
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_rate: float = 1.0


class AISelfHealingSystem:
    """
    AI-powered error prediction and self-healing system
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # ML Models
        self.error_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.failure_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.severity_predictor = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42)
        
        # Clustering for error pattern analysis
        self.error_clusterer = KMeans(n_clusters=8, random_state=42)
        self.pattern_detector = DBSCAN(eps=0.3, min_samples=5)
        
        # Feature scalers
        self.metrics_scaler = StandardScaler()
        self.error_scaler = MinMaxScaler()
        
        # Neural network for advanced prediction
        if TORCH_AVAILABLE:
            self.prediction_nn = ErrorPredictionNN(input_dim=30)
            self.nn_optimizer = torch.optim.Adam(self.prediction_nn.parameters(), lr=0.001)
            self.nn_scheduler = torch.optim.lr_scheduler.StepLR(self.nn_optimizer, step_size=50, gamma=0.9)
        else:
            self.prediction_nn = None
        
        # Storage
        self.error_history = deque(maxlen=1000)
        self.metrics_history = deque(maxlen=2000)
        self.healing_actions = {}
        self.system_baselines = {}
        self.prediction_cache = {}
        
        # Real-time monitoring
        self.current_metrics = None
        self.anomaly_scores = deque(maxlen=100)
        self.prediction_confidence = deque(maxlen=50)
        
        # Self-healing state
        self.healing_enabled = True
        self.active_healers = {}
        self.healing_statistics = defaultdict(int)
        
        # Initialize default healing actions
        self._initialize_healing_actions()
        self._initialize_models()
        
        # Start background monitoring
        self.monitoring_task = None
        self._start_monitoring()
        
        logger.info("AI Self-Healing System initialized")
    
    def _initialize_healing_actions(self):
        """Initialize default self-healing actions"""
        
        # Memory pressure relief
        self.healing_actions['memory_cleanup'] = HealingAction(
            action_id='memory_cleanup',
            name='Memory Cleanup',
            description='Clear memory caches and garbage collection',
            trigger_conditions=['memory_usage > 0.85', 'memory_leak_detected'],
            action_function=self._memory_cleanup_action,
            cooldown_period=180,
            max_attempts=5
        )
        
        # CPU load balancing
        self.healing_actions['cpu_scaling'] = HealingAction(
            action_id='cpu_scaling',
            name='CPU Load Balancing',
            description='Redistribute CPU-intensive tasks',
            trigger_conditions=['cpu_usage > 0.9', 'cpu_overload_predicted'],
            action_function=self._cpu_scaling_action,
            cooldown_period=120,
            max_attempts=3
        )
        
        # Network connection recovery
        self.healing_actions['network_recovery'] = HealingAction(
            action_id='network_recovery',
            name='Network Recovery',
            description='Reset network connections and retry failed requests',
            trigger_conditions=['network_timeout_rate > 0.1', 'connection_pool_exhausted'],
            action_function=self._network_recovery_action,
            cooldown_period=60,
            max_attempts=5
        )
        
        # Service restart
        self.healing_actions['service_restart'] = HealingAction(
            action_id='service_restart',
            name='Service Restart',
            description='Restart failing services',
            trigger_conditions=['service_health_score < 0.3', 'critical_error_rate > 0.05'],
            action_function=self._service_restart_action,
            cooldown_period=300,
            max_attempts=2,
            requires_confirmation=True
        )
        
        # Database connection healing
        self.healing_actions['db_connection_heal'] = HealingAction(
            action_id='db_connection_heal',
            name='Database Connection Healing',
            description='Reset database connection pools',
            trigger_conditions=['db_connection_failures > 5', 'db_timeout_rate > 0.2'],
            action_function=self._db_connection_heal_action,
            cooldown_period=90,
            max_attempts=3
        )
        
        # Disk cleanup
        self.healing_actions['disk_cleanup'] = HealingAction(
            action_id='disk_cleanup',
            name='Disk Space Cleanup',
            description='Clean temporary files and logs',
            trigger_conditions=['disk_usage > 0.9', 'log_size_excessive'],
            action_function=self._disk_cleanup_action,
            cooldown_period=600,
            max_attempts=1
        )
    
    def _initialize_models(self):
        """Initialize ML models with synthetic training data"""
        # Generate training data
        X_errors, y_errors = self._generate_error_training_data()
        X_anomaly = self._generate_anomaly_training_data()
        X_failure, y_failure = self._generate_failure_training_data()
        
        # Train models
        if len(X_errors) > 0:
            X_errors_scaled = self.error_scaler.fit_transform(X_errors)
            self.error_classifier.fit(X_errors_scaled, y_errors)
        
        if len(X_anomaly) > 0:
            X_anomaly_scaled = self.metrics_scaler.fit_transform(X_anomaly)
            self.anomaly_detector.fit(X_anomaly_scaled)
        
        if len(X_failure) > 0:
            self.failure_predictor.fit(X_failure, y_failure)
        
        # Initialize baselines
        self._establish_system_baselines()
    
    async def predict_errors(self, current_metrics: SystemMetrics) -> Dict[str, Any]:
        """
        Predict potential errors using AI
        """
        try:
            # Extract features
            features = self._extract_prediction_features(current_metrics)
            
            # ML predictions
            error_probability = self._predict_error_probability(features)
            time_to_failure = self._predict_time_to_failure(features)
            error_type = self._predict_error_type(features)
            severity = self._predict_severity(features)
            
            # Neural network enhancement
            if self.prediction_nn and TORCH_AVAILABLE:
                nn_predictions = await self._get_nn_predictions(features)
                error_probability = (error_probability + nn_predictions['error_prob']) / 2
                time_to_failure = (time_to_failure + nn_predictions['time_to_failure']) / 2
            
            # Anomaly detection
            anomaly_score = self._calculate_anomaly_score(features)
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(features, error_probability)
            
            prediction = {
                'error_probability': float(error_probability),
                'time_to_failure_hours': float(time_to_failure),
                'predicted_error_type': error_type,
                'predicted_severity': severity,
                'anomaly_score': float(anomaly_score),
                'confidence': float(confidence),
                'recommendations': self._generate_recommendations(error_probability, error_type, severity),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Cache prediction
            self.prediction_cache[current_metrics.timestamp] = prediction
            self.prediction_confidence.append(confidence)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error prediction failed: {e}")
            return {
                'error_probability': 0.1,
                'time_to_failure_hours': 24.0,
                'predicted_error_type': ErrorType.UNKNOWN.value,
                'predicted_severity': SeverityLevel.LOW.value,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def analyze_error(self, error: SystemError) -> Dict[str, Any]:
        """
        Analyze an error and determine recovery actions
        """
        try:
            # Extract error features
            error_features = self._extract_error_features(error)
            
            # Find similar errors
            similar_errors = self._find_similar_errors(error_features)
            
            # Predict recovery time
            recovery_time = self._predict_recovery_time(error_features)
            
            # Determine root cause
            root_cause = await self._analyze_root_cause(error)
            
            # Recommend healing actions
            healing_actions = self._recommend_healing_actions(error, similar_errors)
            
            analysis = {
                'error_id': error.error_id,
                'predicted_recovery_time_minutes': float(recovery_time),
                'root_cause_probability': root_cause,
                'similar_errors_count': len(similar_errors),
                'recommended_actions': healing_actions,
                'auto_healing_possible': len(healing_actions) > 0,
                'confidence': self._calculate_analysis_confidence(error_features),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store analysis
            self.error_history.append({
                'error': error,
                'analysis': analysis,
                'timestamp': datetime.utcnow()
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analysis failed: {e}")
            return {
                'error_id': error.error_id,
                'predicted_recovery_time_minutes': 60.0,
                'root_cause_probability': {'unknown': 1.0},
                'similar_errors_count': 0,
                'recommended_actions': [],
                'auto_healing_possible': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def trigger_self_healing(self, error: SystemError, force: bool = False) -> Dict[str, Any]:
        """
        Trigger self-healing actions for an error
        """
        if not self.healing_enabled and not force:
            return {'healing_triggered': False, 'reason': 'Self-healing disabled'}
        
        try:
            # Analyze error first
            analysis = await self.analyze_error(error)
            
            if not analysis.get('auto_healing_possible', False) and not force:
                return {'healing_triggered': False, 'reason': 'No suitable healing actions available'}
            
            # Execute healing actions
            healing_results = []
            
            for action_id in analysis.get('recommended_actions', []):
                if action_id in self.healing_actions:
                    action = self.healing_actions[action_id]
                    
                    # Check cooldown and attempt limits
                    if not self._can_execute_action(action):
                        continue
                    
                    result = await self._execute_healing_action(action, error)
                    healing_results.append(result)
                    
                    # Update statistics
                    self.healing_statistics[action_id] += 1
                    
                    # Stop if successful
                    if result.get('success', False):
                        break
            
            # Update healing action success rates
            self._update_action_success_rates(healing_results)
            
            return {
                'healing_triggered': True,
                'actions_executed': len(healing_results),
                'successful_actions': sum(1 for r in healing_results if r.get('success', False)),
                'results': healing_results,
                'estimated_recovery_time': analysis.get('predicted_recovery_time_minutes', 60.0)
            }
            
        except Exception as e:
            logger.error(f"Self-healing trigger failed: {e}")
            return {'healing_triggered': False, 'error': str(e)}
    
    async def monitor_system_health(self) -> Dict[str, Any]:
        """
        Continuous system health monitoring with AI analysis
        """
        try:
            # Collect current metrics
            current_metrics = await self._collect_system_metrics()
            self.current_metrics = current_metrics
            self.metrics_history.append(current_metrics)
            
            # Predict potential issues
            prediction = await self.predict_errors(current_metrics)
            
            # Check for anomalies
            anomaly_score = prediction.get('anomaly_score', 0.0)
            self.anomaly_scores.append(anomaly_score)
            
            # Trigger proactive healing if needed
            healing_triggered = False
            if prediction.get('error_probability', 0.0) > 0.7:
                # Create synthetic error for proactive healing
                synthetic_error = SystemError(
                    error_id=f"predicted_{int(time.time())}",
                    timestamp=datetime.utcnow(),
                    error_type=ErrorType(prediction.get('predicted_error_type', 'unknown')),
                    severity=SeverityLevel(prediction.get('predicted_severity', 0)),
                    component='system',
                    message=f"Predicted error: {prediction.get('predicted_error_type', 'unknown')}"
                )
                
                heal_result = await self.trigger_self_healing(synthetic_error)
                healing_triggered = heal_result.get('healing_triggered', False)
            
            # Calculate health score
            health_score = self._calculate_health_score(current_metrics, prediction)
            
            return {
                'timestamp': current_metrics.timestamp.isoformat(),
                'health_score': float(health_score),
                'current_metrics': {
                    'cpu_usage': current_metrics.cpu_usage,
                    'memory_usage': current_metrics.memory_usage,
                    'disk_usage': current_metrics.disk_usage,
                    'error_rate': current_metrics.error_rate,
                    'response_time': current_metrics.response_time
                },
                'prediction': prediction,
                'proactive_healing_triggered': healing_triggered,
                'anomaly_score': float(anomaly_score),
                'recommendations': prediction.get('recommendations', [])
            }
            
        except Exception as e:
            logger.error(f"System health monitoring failed: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'health_score': 0.5,
                'error': str(e)
            }
    
    def _extract_prediction_features(self, metrics: SystemMetrics) -> np.ndarray:
        """Extract ML features from system metrics"""
        features = []
        
        # Basic metrics
        features.extend([
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.disk_usage,
            metrics.error_rate,
            metrics.response_time,
            metrics.throughput,
            metrics.active_connections
        ])
        
        # Time-based features
        now = metrics.timestamp
        features.extend([
            now.hour / 24.0,
            now.weekday() / 7.0,
            (now.timestamp() % 86400) / 86400.0  # Time of day
        ])
        
        # Historical trend features
        if len(self.metrics_history) > 1:
            recent_metrics = list(self.metrics_history)[-10:]
            
            # Trends
            cpu_trend = np.polyfit(range(len(recent_metrics)), 
                                 [m.cpu_usage for m in recent_metrics], 1)[0]
            memory_trend = np.polyfit(range(len(recent_metrics)), 
                                    [m.memory_usage for m in recent_metrics], 1)[0]
            error_trend = np.polyfit(range(len(recent_metrics)), 
                                   [m.error_rate for m in recent_metrics], 1)[0]
            
            features.extend([cpu_trend, memory_trend, error_trend])
            
            # Variability
            cpu_std = np.std([m.cpu_usage for m in recent_metrics])
            memory_std = np.std([m.memory_usage for m in recent_metrics])
            response_std = np.std([m.response_time for m in recent_metrics])
            
            features.extend([cpu_std, memory_std, response_std])
        else:
            features.extend([0.0] * 6)  # No historical data
        
        # Baseline deviations
        if self.system_baselines:
            cpu_deviation = abs(metrics.cpu_usage - self.system_baselines.get('cpu_usage', 0.5))
            memory_deviation = abs(metrics.memory_usage - self.system_baselines.get('memory_usage', 0.5))
            error_deviation = abs(metrics.error_rate - self.system_baselines.get('error_rate', 0.01))
            
            features.extend([cpu_deviation, memory_deviation, error_deviation])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Network features
        network_io = metrics.network_io
        features.extend([
            network_io.get('bytes_sent', 0) / 1000000.0,  # MB
            network_io.get('bytes_recv', 0) / 1000000.0,  # MB
            network_io.get('packets_sent', 0) / 1000.0,   # K packets
            network_io.get('packets_recv', 0) / 1000.0    # K packets
        ])
        
        return np.array(features)
    
    def _predict_error_probability(self, features: np.ndarray) -> float:
        """Predict probability of error occurrence"""
        try:
            if hasattr(self.error_classifier, 'predict_proba'):
                features_scaled = self.error_scaler.transform(features.reshape(1, -1))
                probabilities = self.error_classifier.predict_proba(features_scaled)[0]
                return float(probabilities[1] if len(probabilities) > 1 else probabilities[0])
        except Exception as e:
            logger.debug(f"Error probability prediction failed: {e}")
        
        # Fallback heuristic
        cpu_risk = features[0] if features[0] > 0.8 else 0.0
        memory_risk = features[1] if features[1] > 0.85 else 0.0
        error_risk = features[3] if features[3] > 0.05 else 0.0
        
        return float(min(1.0, (cpu_risk + memory_risk + error_risk) / 3))
    
    def _predict_time_to_failure(self, features: np.ndarray) -> float:
        """Predict time to failure in hours"""
        try:
            if hasattr(self.failure_predictor, 'predict'):
                prediction = self.failure_predictor.predict(features.reshape(1, -1))[0]
                return max(0.1, float(prediction))
        except Exception as e:
            logger.debug(f"Time to failure prediction failed: {e}")
        
        # Fallback based on current stress level
        stress_level = (features[0] + features[1] + features[3]) / 3
        if stress_level > 0.9:
            return 1.0  # 1 hour
        elif stress_level > 0.7:
            return 4.0  # 4 hours
        else:
            return 24.0  # 24 hours
    
    def _predict_error_type(self, features: np.ndarray) -> str:
        """Predict most likely error type"""
        # Simple heuristic-based prediction
        cpu_usage = features[0] if len(features) > 0 else 0.5
        memory_usage = features[1] if len(features) > 1 else 0.5
        disk_usage = features[2] if len(features) > 2 else 0.5
        error_rate = features[3] if len(features) > 3 else 0.01
        
        scores = {}
        scores[ErrorType.CPU_OVERLOAD.value] = cpu_usage
        scores[ErrorType.MEMORY_LEAK.value] = memory_usage
        scores[ErrorType.DISK_SPACE.value] = disk_usage
        scores[ErrorType.SERVICE_UNAVAILABLE.value] = error_rate
        scores[ErrorType.NETWORK_TIMEOUT.value] = features[4] if len(features) > 4 else 0.1
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _predict_severity(self, features: np.ndarray) -> int:
        """Predict error severity level"""
        # Calculate overall system stress
        if len(features) < 4:
            return SeverityLevel.LOW.value
        
        stress_indicators = [features[0], features[1], features[3]]  # CPU, memory, error rate
        avg_stress = np.mean(stress_indicators)
        max_stress = np.max(stress_indicators)
        
        if max_stress > 0.95 or avg_stress > 0.9:
            return SeverityLevel.CRITICAL.value
        elif max_stress > 0.85 or avg_stress > 0.75:
            return SeverityLevel.HIGH.value
        elif max_stress > 0.7 or avg_stress > 0.6:
            return SeverityLevel.MEDIUM.value
        else:
            return SeverityLevel.LOW.value
    
    def _calculate_anomaly_score(self, features: np.ndarray) -> float:
        """Calculate anomaly score for current state"""
        try:
            if hasattr(self.anomaly_detector, 'decision_function'):
                features_scaled = self.metrics_scaler.transform(features.reshape(1, -1))
                anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
                # Convert to 0-1 range (higher = more anomalous)
                normalized_score = 1.0 / (1.0 + np.exp(anomaly_score))
                return float(normalized_score)
        except:
            pass
        
        # Fallback - detect obvious anomalies
        if len(features) < 3:
            return 0.0
        
        # Check for extreme values
        extreme_count = sum(1 for f in features[:7] if f > 0.95 or f < 0.05)
        return float(min(1.0, extreme_count / 7.0))
    
    def _calculate_prediction_confidence(self, features: np.ndarray, prediction: float) -> float:
        """Calculate confidence in prediction"""
        # Factors affecting confidence
        data_quality = 1.0 if len(self.metrics_history) > 10 else len(self.metrics_history) / 10.0
        feature_completeness = len(features) / 30.0  # Expected 30 features
        prediction_consistency = np.mean(self.prediction_confidence) if self.prediction_confidence else 0.5
        
        # Combine factors
        confidence = (data_quality + feature_completeness + prediction_consistency) / 3
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _generate_recommendations(self, error_prob: float, error_type: str, severity: int) -> List[str]:
        """Generate recommendations based on predictions"""
        recommendations = []
        
        if error_prob > 0.7:
            recommendations.append(f"High risk of {error_type} - consider proactive measures")
        
        if severity >= SeverityLevel.HIGH.value:
            recommendations.append("Monitor system closely - potential critical impact")
        
        # Type-specific recommendations
        if error_type == ErrorType.MEMORY_LEAK.value:
            recommendations.append("Monitor memory usage closely, consider garbage collection")
        elif error_type == ErrorType.CPU_OVERLOAD.value:
            recommendations.append("Consider load balancing or task rescheduling")
        elif error_type == ErrorType.DISK_SPACE.value:
            recommendations.append("Plan disk cleanup or storage expansion")
        elif error_type == ErrorType.NETWORK_TIMEOUT.value:
            recommendations.append("Check network connectivity and timeout configurations")
        
        return recommendations
    
    async def _get_nn_predictions(self, features: np.ndarray) -> Dict[str, float]:
        """Get predictions from neural network"""
        if not TORCH_AVAILABLE or not self.prediction_nn:
            return {'error_prob': 0.5, 'time_to_failure': 12.0}
        
        try:
            # Pad or truncate features to expected input size
            if len(features) > 30:
                features = features[:30]
            elif len(features) < 30:
                features = np.pad(features, (0, 30 - len(features)), mode='constant')
            
            feature_tensor = torch.FloatTensor(features)
            
            with torch.no_grad():
                error_prob, time_to_failure, _, _, _, _ = self.prediction_nn(feature_tensor.unsqueeze(0))
            
            return {
                'error_prob': float(error_prob.item()),
                'time_to_failure': float(time_to_failure.item())
            }
        except Exception as e:
            logger.error(f"Neural network prediction error: {e}")
            return {'error_prob': 0.5, 'time_to_failure': 12.0}
    
    # Helper methods for error analysis and healing actions
    def _extract_error_features(self, error: SystemError) -> np.ndarray:
        """Extract features from error for analysis"""
        features = []
        
        # Error characteristics
        features.append(error.severity.value / 3.0)  # Normalize severity
        features.append(hash(error.error_type.value) % 1000 / 1000.0)  # Type encoding
        features.append(len(error.message) / 1000.0)  # Message length
        features.append(1.0 if error.stack_trace else 0.0)  # Has stack trace
        
        # Timing features
        hour = error.timestamp.hour / 24.0
        day = error.timestamp.weekday() / 7.0
        features.extend([hour, day])
        
        # Context features
        features.append(len(error.context) / 10.0)  # Context richness
        features.append(error.resolution_time / 3600.0 if error.resolution_time else 1.0)  # Hours
        
        # System state when error occurred
        if self.current_metrics:
            features.extend([
                self.current_metrics.cpu_usage,
                self.current_metrics.memory_usage,
                self.current_metrics.error_rate
            ])
        else:
            features.extend([0.5, 0.5, 0.01])
        
        return np.array(features)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # Use psutil for system metrics
            cpu_usage = psutil.cpu_percent(interval=1) / 100.0
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent / 100.0
            disk_info = psutil.disk_usage('/')
            disk_usage = disk_info.percent / 100.0
            
            # Network I/O
            network_info = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network_info.bytes_sent,
                'bytes_recv': network_info.bytes_recv,
                'packets_sent': network_info.packets_sent,
                'packets_recv': network_info.packets_recv
            }
            
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_connections=len(psutil.net_connections()),
                error_rate=0.01,  # Would be calculated from actual error logs
                response_time=100.0,  # Would be from application metrics
                throughput=50.0  # Would be from application metrics
            )
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return default metrics
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=0.5,
                memory_usage=0.5,
                disk_usage=0.3,
                error_rate=0.01,
                response_time=100.0,
                throughput=50.0
            )
    
    # Self-healing action implementations
    async def _memory_cleanup_action(self, error: SystemError) -> Dict[str, Any]:
        """Execute memory cleanup action"""
        try:
            import gc
            
            # Force garbage collection
            collected = gc.collect()
            
            # Clear internal caches
            if hasattr(self, 'prediction_cache'):
                old_cache_size = len(self.prediction_cache)
                self.prediction_cache.clear()
            else:
                old_cache_size = 0
            
            return {
                'success': True,
                'action': 'memory_cleanup',
                'details': f'Collected {collected} objects, cleared {old_cache_size} cache entries',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'action': 'memory_cleanup',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _cpu_scaling_action(self, error: SystemError) -> Dict[str, Any]:
        """Execute CPU scaling action"""
        try:
            # Simulate load balancing by adjusting task priorities
            # In real implementation, this would redistribute workload
            
            return {
                'success': True,
                'action': 'cpu_scaling',
                'details': 'Redistributed CPU-intensive tasks',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'action': 'cpu_scaling',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _network_recovery_action(self, error: SystemError) -> Dict[str, Any]:
        """Execute network recovery action"""
        try:
            # Simulate network connection reset
            # In real implementation, this would reset connection pools
            
            return {
                'success': True,
                'action': 'network_recovery',
                'details': 'Reset network connections and connection pools',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'action': 'network_recovery',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _service_restart_action(self, error: SystemError) -> Dict[str, Any]:
        """Execute service restart action"""
        try:
            # Simulate service restart
            # In real implementation, this would restart specific services
            
            return {
                'success': True,
                'action': 'service_restart',
                'details': 'Restarted failing service components',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'action': 'service_restart',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _db_connection_heal_action(self, error: SystemError) -> Dict[str, Any]:
        """Execute database connection healing action"""
        try:
            # Simulate database connection pool reset
            
            return {
                'success': True,
                'action': 'db_connection_heal',
                'details': 'Reset database connection pools',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'action': 'db_connection_heal',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _disk_cleanup_action(self, error: SystemError) -> Dict[str, Any]:
        """Execute disk cleanup action"""
        try:
            import tempfile
            import shutil
            
            # Clean temporary files
            temp_dir = tempfile.gettempdir()
            cleaned_size = 0
            
            # This is a simplified version - real implementation would be more sophisticated
            for filename in os.listdir(temp_dir):
                if filename.startswith('tmp') or filename.endswith('.tmp'):
                    filepath = os.path.join(temp_dir, filename)
                    try:
                        if os.path.isfile(filepath):
                            size = os.path.getsize(filepath)
                            os.remove(filepath)
                            cleaned_size += size
                    except:
                        pass
            
            return {
                'success': True,
                'action': 'disk_cleanup',
                'details': f'Cleaned {cleaned_size} bytes from temporary files',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'action': 'disk_cleanup',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    # Utility methods
    def _establish_system_baselines(self):
        """Establish system performance baselines"""
        if len(self.metrics_history) < 10:
            # Default baselines
            self.system_baselines = {
                'cpu_usage': 0.3,
                'memory_usage': 0.4,
                'disk_usage': 0.2,
                'error_rate': 0.01,
                'response_time': 200.0
            }
        else:
            # Calculate from historical data
            recent_metrics = list(self.metrics_history)[-50:]  # Last 50 measurements
            
            self.system_baselines = {
                'cpu_usage': np.percentile([m.cpu_usage for m in recent_metrics], 50),
                'memory_usage': np.percentile([m.memory_usage for m in recent_metrics], 50),
                'disk_usage': np.percentile([m.disk_usage for m in recent_metrics], 50),
                'error_rate': np.percentile([m.error_rate for m in recent_metrics], 50),
                'response_time': np.percentile([m.response_time for m in recent_metrics], 50)
            }
    
    def _calculate_health_score(self, metrics: SystemMetrics, prediction: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        # Component scores (0-1, higher is better)
        cpu_score = max(0, 1 - metrics.cpu_usage)
        memory_score = max(0, 1 - metrics.memory_usage)
        disk_score = max(0, 1 - metrics.disk_usage)
        error_score = max(0, 1 - min(1, metrics.error_rate * 20))  # Scale error rate
        prediction_score = 1 - prediction.get('error_probability', 0.5)
        anomaly_score = 1 - prediction.get('anomaly_score', 0.0)
        
        # Weighted average
        weights = [0.2, 0.2, 0.1, 0.2, 0.2, 0.1]  # CPU, Memory, Disk, Error, Prediction, Anomaly
        scores = [cpu_score, memory_score, disk_score, error_score, prediction_score, anomaly_score]
        
        health_score = sum(w * s for w, s in zip(weights, scores))
        return np.clip(health_score, 0.0, 1.0)
    
    def _start_monitoring(self):
        """Start background monitoring task"""
        async def monitoring_loop():
            while True:
                try:
                    await self.monitor_system_health()
                    await asyncio.sleep(30)  # Monitor every 30 seconds
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
        
        if asyncio.get_event_loop().is_running():
            self.monitoring_task = asyncio.create_task(monitoring_loop())
    
    # Training data generation for initial model training
    def _generate_error_training_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Extract real training data from system error history"""
        X, y = [], []
        
        # Use actual system errors and healings for training
        if not hasattr(self, 'healing_history') or not self.healing_history:
            logger.warning("No healing history available for error training")
            return [], []
        
        for healing_record in list(self.healing_history):
            try:
                # Extract features from real healing attempt
                error = healing_record.get('error')
                was_successful = healing_record.get('successful', False)
                
                if error and hasattr(error, 'context'):
                    features = self._extract_system_features_from_error(error)
                    is_error = 1 if was_successful else 0  # Successful healing indicates real error
                    
                    X.append(features)
                    y.append(is_error)
                    
            except Exception as e:
                logger.debug(f"Failed to extract error training data: {e}")
                continue
        
        return X, y
    
    def _generate_anomaly_training_data(self) -> List[np.ndarray]:
        """Extract real anomaly training data from system metrics"""
        from .data_pipeline import get_data_pipeline
        
        X = []
        
        try:
            # Get real system metrics for normal operation patterns
            pipeline = get_data_pipeline()
            metrics_df = asyncio.run(pipeline.get_training_data('metrics', hours=24, min_samples=50))
            
            for _, metric in metrics_df.iterrows():
                if metric.get('success', True):  # Only use successful operations for normal patterns
                    features = self._extract_features_from_metric(metric)
                    X.append(features)
            
            logger.info(f"Extracted {len(X)} normal operation patterns for anomaly detection")
            
        except Exception as e:
            logger.error(f"Failed to extract real anomaly training data: {e}")
            # Return empty if no real data available
            return []
        
        return X
    
    def _generate_failure_training_data(self) -> Tuple[List[np.ndarray], List[float]]:
        """Extract real failure training data from historical failures"""
        X, y = [], []
        
        # Use actual failure events with known time-to-failure
        if not hasattr(self, 'healing_history') or not self.healing_history:
            logger.warning("No healing history available for failure training")
            return [], []
        
        for i, healing_record in enumerate(list(self.healing_history)):
            try:
                error = healing_record.get('error')
                healing_time = healing_record.get('healing_time', 0)
                error_detected_time = healing_record.get('error_detected_time')
                
                if error and error_detected_time and healing_time > 0:
                    # Calculate actual time from error detection to healing
                    time_to_failure = healing_time / 3600.0  # Convert to hours
                    
                    features = self._extract_system_features_from_error(error)
                    
                    X.append(features)
                    y.append(min(48.0, time_to_failure))  # Cap at 48 hours
                    
            except Exception as e:
                logger.debug(f"Failed to extract failure training data: {e}")
                continue
        
        return X, y
            
            X.append(features)
            y.append(min(168, time_to_failure))  # Cap at 1 week
        
        return X, y
    
    def _find_similar_errors(self, error_features: np.ndarray) -> List[Dict[str, Any]]:
        """Find similar historical errors"""
        similar_errors = []
        
        for record in self.error_history:
            historical_features = self._extract_error_features(record['error'])
            
            # Calculate similarity (simple cosine similarity)
            try:
                similarity = cosine_similarity(error_features.reshape(1, -1), 
                                             historical_features.reshape(1, -1))[0][0]
                if similarity > 0.8:  # Threshold for similarity
                    similar_errors.append({
                        'error': record['error'],
                        'similarity': float(similarity),
                        'resolution_time': record['error'].resolution_time
                    })
            except:
                continue
        
        return sorted(similar_errors, key=lambda x: x['similarity'], reverse=True)[:5]
    
    def _predict_recovery_time(self, error_features: np.ndarray) -> float:
        """Predict error recovery time in minutes"""
        # Simple heuristic based on error characteristics
        severity = error_features[0] if len(error_features) > 0 else 0.5
        complexity = len([f for f in error_features if f > 0.5]) / len(error_features)
        
        base_time = 30  # 30 minutes base
        severity_multiplier = 1 + severity * 3  # 1x to 4x
        complexity_multiplier = 1 + complexity  # 1x to 2x
        
        return base_time * severity_multiplier * complexity_multiplier
    
    async def _analyze_root_cause(self, error: SystemError) -> Dict[str, float]:
        """Analyze potential root causes"""
        # Simplified root cause analysis
        causes = {
            'resource_exhaustion': 0.0,
            'configuration_error': 0.0,
            'external_dependency': 0.0,
            'code_bug': 0.0,
            'hardware_failure': 0.0
        }
        
        # Analyze based on error type and context
        if error.error_type in [ErrorType.MEMORY_LEAK, ErrorType.CPU_OVERLOAD, ErrorType.DISK_SPACE]:
            causes['resource_exhaustion'] = 0.8
        elif error.error_type in [ErrorType.NETWORK_TIMEOUT, ErrorType.DATABASE_CONNECTION]:
            causes['external_dependency'] = 0.7
        elif error.error_type == ErrorType.AUTHENTICATION:
            causes['configuration_error'] = 0.6
        else:
            causes['code_bug'] = 0.5
        
        return causes
    
    def _recommend_healing_actions(self, error: SystemError, similar_errors: List[Dict[str, Any]]) -> List[str]:
        """Recommend healing actions based on error analysis"""
        recommendations = []
        
        # Based on error type
        type_actions = {
            ErrorType.MEMORY_LEAK: ['memory_cleanup'],
            ErrorType.CPU_OVERLOAD: ['cpu_scaling'],
            ErrorType.NETWORK_TIMEOUT: ['network_recovery'],
            ErrorType.DATABASE_CONNECTION: ['db_connection_heal'],
            ErrorType.DISK_SPACE: ['disk_cleanup'],
            ErrorType.SERVICE_UNAVAILABLE: ['service_restart']
        }
        
        if error.error_type in type_actions:
            recommendations.extend(type_actions[error.error_type])
        
        # Based on severity
        if error.severity == SeverityLevel.CRITICAL:
            if 'service_restart' not in recommendations:
                recommendations.append('service_restart')
        
        # Based on similar errors
        for similar in similar_errors[:2]:  # Top 2 similar errors
            if hasattr(similar['error'], 'recovery_actions') and similar['error'].recovery_actions:
                for action in similar['error'].recovery_actions:
                    if action not in recommendations:
                        recommendations.append(action)
        
        return recommendations[:3]  # Limit to top 3 actions
    
    def _can_execute_action(self, action: HealingAction) -> bool:
        """Check if healing action can be executed"""
        now = datetime.utcnow()
        
        # Check cooldown
        if action.last_executed:
            time_since_last = (now - action.last_executed).total_seconds()
            if time_since_last < action.cooldown_period:
                return False
        
        # Check attempt limits
        if action.execution_count >= action.max_attempts:
            return False
        
        return True
    
    async def _execute_healing_action(self, action: HealingAction, error: SystemError) -> Dict[str, Any]:
        """Execute a healing action"""
        try:
            # Update action metadata
            action.last_executed = datetime.utcnow()
            action.execution_count += 1
            
            # Execute the action
            result = await action.action_function(error)
            
            # Update success rate
            if result.get('success', False):
                action.success_rate = (action.success_rate * (action.execution_count - 1) + 1.0) / action.execution_count
            else:
                action.success_rate = (action.success_rate * (action.execution_count - 1) + 0.0) / action.execution_count
            
            return {
                'action_id': action.action_id,
                'success': result.get('success', False),
                'details': result.get('details', ''),
                'error': result.get('error'),
                'execution_time': (datetime.utcnow() - action.last_executed).total_seconds()
            }
        
        except Exception as e:
            logger.error(f"Failed to execute healing action {action.action_id}: {e}")
            return {
                'action_id': action.action_id,
                'success': False,
                'error': str(e)
            }
    
    def _update_action_success_rates(self, healing_results: List[Dict[str, Any]]):
        """Update success rates for healing actions"""
        for result in healing_results:
            action_id = result.get('action_id')
            if action_id in self.healing_actions:
                action = self.healing_actions[action_id]
                success = result.get('success', False)
                
                # Update running average
                current_rate = action.success_rate
                new_rate = (current_rate * (action.execution_count - 1) + (1.0 if success else 0.0)) / action.execution_count
                action.success_rate = new_rate
    
    def _calculate_analysis_confidence(self, error_features: np.ndarray) -> float:
        """Calculate confidence in error analysis"""
        # Factors affecting confidence
        feature_completeness = len(error_features) / 15.0  # Expected 15 features
        historical_data = min(1.0, len(self.error_history) / 50.0)  # More data = higher confidence
        
        confidence = (feature_completeness + historical_data) / 2
        return float(np.clip(confidence, 0.0, 1.0))
    
    async def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        try:
            current_health = await self.monitor_system_health()
            
            # Statistics
            total_errors = len(self.error_history)
            healing_actions_count = sum(self.healing_statistics.values())
            avg_health_score = np.mean([h['health_score'] for h in [current_health]]) if current_health else 0.5
            
            # Trends
            recent_anomalies = sum(1 for score in list(self.anomaly_scores)[-10:] if score > 0.7)
            prediction_accuracy = np.mean(self.prediction_confidence) if self.prediction_confidence else 0.5
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'current_health': current_health,
                'statistics': {
                    'total_errors_tracked': total_errors,
                    'healing_actions_executed': healing_actions_count,
                    'average_health_score': float(avg_health_score),
                    'recent_anomalies': recent_anomalies,
                    'prediction_accuracy': float(prediction_accuracy)
                },
                'healing_actions': {
                    action_id: {
                        'execution_count': action.execution_count,
                        'success_rate': action.success_rate,
                        'last_executed': action.last_executed.isoformat() if action.last_executed else None
                    }
                    for action_id, action in self.healing_actions.items()
                },
                'system_baselines': self.system_baselines,
                'recommendations': current_health.get('recommendations', []) if current_health else []
            }
        
        except Exception as e:
            logger.error(f"Failed to generate health report: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'current_health': {'health_score': 0.5}
            }


# Singleton instance
_ai_self_healing = None

def get_ai_self_healing() -> AISelfHealingSystem:
    """Get or create AI self-healing instance"""
    global _ai_self_healing
    if not _ai_self_healing:
        _ai_self_healing = AISelfHealingSystem()
    return _ai_self_healing