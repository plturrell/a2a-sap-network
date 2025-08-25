"""
AI-Powered Rate Limiting and Throttling System

This module provides intelligent API rate limiting using real machine learning
to predict usage patterns, detect anomalies, and dynamically adjust throttling
policies based on user behavior and system capacity.
"""

import asyncio
import logging
import numpy as np
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import ipaddress

# Machine Learning imports
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

# Advanced prediction algorithms
try:
    from scipy.stats import poisson, gamma, weibull_min
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Deep learning for complex usage patterns
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class UsagePredictionNN(nn.Module):
    """Neural network for API usage pattern prediction"""
    def __init__(self, input_dim, hidden_dim=128):
        super(UsagePredictionNN, self).__init__()
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        
        # Convolutional layers for pattern recognition
        self.conv1d = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv_pool = nn.MaxPool1d(2)
        
        # Feature fusion
        self.fusion = nn.Linear(hidden_dim + 16, hidden_dim // 2)
        
        # Prediction heads
        self.request_rate_head = nn.Linear(hidden_dim // 2, 1)
        self.burst_probability_head = nn.Linear(hidden_dim // 2, 1)
        self.anomaly_score_head = nn.Linear(hidden_dim // 2, 1)
        self.capacity_need_head = nn.Linear(hidden_dim // 2, 1)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=4)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, sequence_data=None):
        batch_size = x.size(0)
        
        # Process sequential data with LSTM
        if sequence_data is not None:
            lstm_out, (hidden, cell) = self.lstm(sequence_data)
            temporal_features = lstm_out[:, -1, :]  # Last timestep
        else:
            temporal_features = torch.zeros(batch_size, 128)
        
        # Process current features with 1D conv
        conv_input = x.unsqueeze(1)  # Add channel dimension
        conv_out = F.relu(self.conv1d(conv_input))
        conv_out = self.conv_pool(conv_out)
        conv_features = conv_out.view(batch_size, -1)
        
        # Pad conv features if needed
        if conv_features.size(1) < 16:
            padding = torch.zeros(batch_size, 16 - conv_features.size(1))
            conv_features = torch.cat([conv_features, padding], dim=1)
        else:
            conv_features = conv_features[:, :16]
        
        # Fuse features
        combined_features = torch.cat([temporal_features, conv_features], dim=1)
        fused_features = F.relu(self.fusion(combined_features))
        
        # Apply attention
        attn_out, _ = self.attention(fused_features.unsqueeze(0), 
                                   fused_features.unsqueeze(0), 
                                   fused_features.unsqueeze(0))
        enhanced_features = self.dropout(attn_out.squeeze(0))
        
        # Predictions
        request_rate = F.relu(self.request_rate_head(enhanced_features))
        burst_prob = torch.sigmoid(self.burst_probability_head(enhanced_features))
        anomaly_score = torch.sigmoid(self.anomaly_score_head(enhanced_features))
        capacity_need = F.relu(self.capacity_need_head(enhanced_features))
        
        return request_rate, burst_prob, anomaly_score, capacity_need


class RateLimitPolicy(Enum):
    """Rate limiting policy types"""
    FIXED = "fixed"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    AI_ADAPTIVE = "ai_adaptive"
    BURST_AWARE = "burst_aware"


@dataclass
class APIRequest:
    """Individual API request record"""
    request_id: str
    user_id: str
    endpoint: str
    method: str
    timestamp: datetime
    response_time: float
    status_code: int
    payload_size: int
    ip_address: str
    user_agent: str
    api_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """User behavior profile for rate limiting"""
    user_id: str
    tier: str = "free"  # free, premium, enterprise
    baseline_rate: float = 60.0  # requests per minute
    burst_allowance: float = 10.0  # extra requests in burst
    historical_patterns: deque = field(default_factory=lambda: deque(maxlen=1000))
    anomaly_score: float = 0.0
    trust_score: float = 0.8
    last_updated: datetime = field(default_factory=datetime.utcnow)
    adaptive_limits: Dict[str, float] = field(default_factory=dict)


@dataclass
class ThrottlingDecision:
    """Rate limiting decision result"""
    allowed: bool
    current_rate: float
    limit: float
    reset_time: datetime
    throttle_duration: Optional[float] = None
    reason: str = ""
    confidence: float = 1.0
    suggested_retry_after: Optional[int] = None


class AIRateLimiter:
    """
    AI-powered rate limiting system with intelligent throttling
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # ML Models
        self.usage_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.burst_predictor = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.capacity_optimizer = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42)
        
        # Clustering for user behavior segmentation
        self.user_clusterer = KMeans(n_clusters=5, random_state=42)
        self.pattern_detector = DBSCAN(eps=0.3, min_samples=5)
        
        # Feature scalers
        self.usage_scaler = StandardScaler()
        self.anomaly_scaler = RobustScaler()
        
        # Neural network for advanced prediction
        if TORCH_AVAILABLE:
            self.prediction_nn = UsagePredictionNN(input_dim=20)
            self.nn_optimizer = torch.optim.Adam(self.prediction_nn.parameters(), lr=0.001)
            self.nn_scheduler = torch.optim.lr_scheduler.StepLR(self.nn_optimizer, step_size=100, gamma=0.9)
        else:
            self.prediction_nn = None
        
        # Storage
        self.user_profiles = {}  # user_id -> UserProfile
        self.request_history = deque(maxlen=10000)
        self.rate_limits = {}  # endpoint -> limits
        self.throttling_state = {}  # user_id -> current throttling
        self.system_metrics = defaultdict(list)
        
        # Real-time tracking
        self.current_rates = defaultdict(lambda: defaultdict(float))  # user_id -> endpoint -> rate
        self.request_windows = defaultdict(lambda: defaultdict(deque))  # sliding windows
        self.token_buckets = defaultdict(dict)  # user_id -> endpoint -> bucket_state
        
        # Adaptive thresholds
        self.dynamic_limits = {}
        self.capacity_predictions = deque(maxlen=100)
        
        # Initialize with default policies and training
        self._initialize_default_policies()
        self._initialize_models()
        
        logger.info("AI Rate Limiter initialized with ML models")
    
    def _initialize_default_policies(self):
        """Initialize default rate limiting policies"""
        self.rate_limits = {
            'default': {'requests_per_minute': 60, 'burst_allowance': 10},
            '/api/agents/': {'requests_per_minute': 100, 'burst_allowance': 20},
            '/api/tasks/': {'requests_per_minute': 80, 'burst_allowance': 15},
            '/api/workflows/': {'requests_per_minute': 40, 'burst_allowance': 8},
            '/api/analytics/': {'requests_per_minute': 20, 'burst_allowance': 5}
        }
        
        # Tier-based multipliers
        self.tier_multipliers = {
            'free': 1.0,
            'premium': 3.0,
            'enterprise': 10.0
        }
    
    def _initialize_models(self):
        """Initialize ML models with synthetic training data"""
        # Generate training data
        X_usage, y_usage = self._generate_usage_training_data()
        X_anomaly = self._generate_anomaly_training_data()
        X_burst, y_burst = self._generate_burst_training_data()
        
        # Train models
        if len(X_usage) > 0:
            X_usage_scaled = self.usage_scaler.fit_transform(X_usage)
            self.usage_predictor.fit(X_usage_scaled, y_usage)
        
        if len(X_anomaly) > 0:
            X_anomaly_scaled = self.anomaly_scaler.fit_transform(X_anomaly)
            self.anomaly_detector.fit(X_anomaly_scaled)
        
        if len(X_burst) > 0:
            self.burst_predictor.fit(X_burst, y_burst)
    
    async def check_rate_limit(self, request: APIRequest) -> ThrottlingDecision:
        """
        Check if request should be rate limited using AI
        """
        try:
            # Get or create user profile
            user_profile = await self._get_user_profile(request.user_id)
            
            # Extract request features for ML
            request_features = self._extract_request_features(request, user_profile)
            
            # Predict usage patterns
            predicted_rate = self._predict_request_rate(request_features)
            burst_probability = self._predict_burst_probability(request_features)
            anomaly_score = self._calculate_anomaly_score(request_features)
            
            # Get current limits (adaptive)
            current_limits = await self._get_adaptive_limits(request, user_profile)
            
            # Calculate current rate
            current_rate = self._calculate_current_rate(request.user_id, request.endpoint)
            
            # AI-powered decision making
            decision = await self._make_throttling_decision(
                request, user_profile, current_rate, current_limits,
                predicted_rate, burst_probability, anomaly_score
            )
            
            # Update tracking
            await self._update_request_tracking(request, decision)
            
            # Learn from decision for future improvements
            await self._record_decision_feedback(request, decision, request_features)
            
            return decision
            
        except Exception as e:
            logger.error(f"Rate limiting check error: {e}")
            # Fallback to simple rate limiting
            return await self._fallback_rate_check(request)
    
    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user behavior profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        return self.user_profiles[user_id]
    
    def _extract_request_features(self, request: APIRequest, user_profile: UserProfile) -> np.ndarray:
        """Extract ML features from request and user profile"""
        features = []
        
        # Time-based features
        now = request.timestamp
        hour_of_day = now.hour / 24.0
        day_of_week = now.weekday() / 7.0
        features.extend([hour_of_day, day_of_week])
        
        # Request characteristics
        features.append(len(request.endpoint) / 100.0)  # Normalize endpoint length
        features.append(request.payload_size / 10000.0)  # Normalize payload size
        features.append(hash(request.method) % 1000 / 1000.0)  # Method encoding
        
        # User behavior features
        features.append(user_profile.trust_score)
        features.append(user_profile.anomaly_score)
        features.append(len(user_profile.historical_patterns) / 1000.0)
        
        # Current rate features
        current_rate = self._calculate_current_rate(request.user_id, request.endpoint)
        features.append(min(1.0, current_rate / 100.0))  # Normalize rate
        
        # Historical pattern features
        if user_profile.historical_patterns:
            recent_requests = list(user_profile.historical_patterns)[-10:]
            avg_interval = np.mean([r['interval'] for r in recent_requests]) if recent_requests else 60.0
            features.append(min(1.0, avg_interval / 120.0))  # Normalize interval
            
            # Request distribution
            hourly_dist = [0] * 24
            for req in recent_requests:
                req_hour = req.get('hour', 12)
                hourly_dist[req_hour] += 1
            
            # Use top 3 hours as features
            top_hours = sorted(enumerate(hourly_dist), key=lambda x: x[1], reverse=True)[:3]
            for i, (hour, count) in enumerate(top_hours):
                features.append(count / max(1, len(recent_requests)))
        else:
            features.extend([1.0, 0.5, 0.3, 0.2])  # Default values
        
        # System load features
        system_load = self._get_current_system_load()
        features.extend([
            system_load.get('cpu', 0.5),
            system_load.get('memory', 0.5),
            system_load.get('network', 0.3)
        ])
        
        # IP-based features
        try:
            ip = ipaddress.ip_address(request.ip_address)
            features.append(1.0 if ip.is_private else 0.0)
        except:
            features.append(0.0)
        
        return np.array(features)
    
    def _predict_request_rate(self, features: np.ndarray) -> float:
        """Predict incoming request rate using ML"""
        try:
            if hasattr(self.usage_predictor, 'predict'):
                features_scaled = self.usage_scaler.transform(features.reshape(1, -1))
                prediction = self.usage_predictor.predict(features_scaled)[0]
                return max(0.0, float(prediction))
        except:
            pass
        
        # Fallback heuristic
        return float(features[8] * 100.0) if len(features) > 8 else 30.0
    
    def _predict_burst_probability(self, features: np.ndarray) -> float:
        """Predict probability of burst traffic"""
        try:
            if hasattr(self.burst_predictor, 'predict_proba'):
                proba = self.burst_predictor.predict_proba(features.reshape(1, -1))[0]
                return float(proba[1] if len(proba) > 1 else proba[0])
        except:
            pass
        
        # Heuristic based on current rate
        current_rate = features[8] if len(features) > 8 else 0.3
        return float(min(1.0, current_rate * 2))
    
    def _calculate_anomaly_score(self, features: np.ndarray) -> float:
        """Calculate anomaly score for the request pattern"""
        try:
            if hasattr(self.anomaly_detector, 'decision_function'):
                features_scaled = self.anomaly_scaler.transform(features.reshape(1, -1))
                anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
                # Convert to 0-1 range (lower score = more anomalous)
                normalized_score = 1.0 / (1.0 + np.exp(-anomaly_score))
                return float(1.0 - normalized_score)  # Higher score = more anomalous
        except:
            pass
        
        # Simple heuristic based on feature variance
        feature_variance = np.var(features)
        return float(min(1.0, feature_variance))
    
    async def _get_adaptive_limits(self, request: APIRequest, user_profile: UserProfile) -> Dict[str, float]:
        """Calculate adaptive rate limits using ML predictions"""
        endpoint = request.endpoint
        
        # Base limits from configuration
        base_limits = self.rate_limits.get(endpoint, self.rate_limits['default'])
        base_rate = base_limits['requests_per_minute']
        base_burst = base_limits['burst_allowance']
        
        # Apply tier multiplier
        tier_multiplier = self.tier_multipliers.get(user_profile.tier, 1.0)
        
        # Trust-based adjustment
        trust_multiplier = 0.8 + (user_profile.trust_score * 0.4)  # 0.8 - 1.2 range
        
        # System capacity adjustment
        system_load = self._get_current_system_load()
        capacity_multiplier = 2.0 - system_load.get('overall', 0.5)  # Reduce limits when system loaded
        
        # ML-based dynamic adjustment
        recent_patterns = list(user_profile.historical_patterns)[-20:] if user_profile.historical_patterns else []
        if recent_patterns:
            avg_usage = np.mean([p.get('rate', base_rate/2) for p in recent_patterns])
            predicted_need = max(avg_usage * 1.5, base_rate)  # Allow 50% above historical average
        else:
            predicted_need = base_rate
        
        # Combine all factors
        final_rate = base_rate * tier_multiplier * trust_multiplier * capacity_multiplier
        final_rate = min(final_rate, predicted_need * 2)  # Cap at 2x predicted need
        
        # Burst limits
        final_burst = base_burst * tier_multiplier * trust_multiplier
        
        return {
            'requests_per_minute': max(1.0, final_rate),
            'burst_allowance': max(1.0, final_burst),
            'confidence': min(1.0, user_profile.trust_score + 0.2)
        }
    
    def _calculate_current_rate(self, user_id: str, endpoint: str) -> float:
        """Calculate current request rate for user/endpoint"""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)
        
        # Get sliding window
        if user_id not in self.request_windows:
            self.request_windows[user_id][endpoint] = deque()
        
        window = self.request_windows[user_id][endpoint]
        
        # Remove old requests
        while window and window[0] < window_start:
            window.popleft()
        
        return len(window)
    
    async def _make_throttling_decision(self, request: APIRequest, user_profile: UserProfile, 
                                      current_rate: float, limits: Dict[str, float],
                                      predicted_rate: float, burst_prob: float, 
                                      anomaly_score: float) -> ThrottlingDecision:
        """Make AI-powered throttling decision"""
        
        rate_limit = limits['requests_per_minute']
        burst_allowance = limits['burst_allowance']
        
        # Basic rate check
        if current_rate <= rate_limit:
            # Within normal limits
            return ThrottlingDecision(
                allowed=True,
                current_rate=current_rate,
                limit=rate_limit,
                reset_time=datetime.utcnow() + timedelta(minutes=1),
                reason="Within rate limits"
            )
        
        # Check burst allowance
        if current_rate <= rate_limit + burst_allowance:
            # Within burst allowance, but check ML predictions
            
            # High anomaly score = likely attack
            if anomaly_score > 0.7:
                return ThrottlingDecision(
                    allowed=False,
                    current_rate=current_rate,
                    limit=rate_limit,
                    reset_time=datetime.utcnow() + timedelta(minutes=2),
                    throttle_duration=120.0,
                    reason="Anomalous traffic pattern detected",
                    confidence=anomaly_score,
                    suggested_retry_after=120
                )
            
            # High burst probability = temporary spike (allow with lower confidence)
            if burst_prob > 0.6 and user_profile.trust_score > 0.6:
                return ThrottlingDecision(
                    allowed=True,
                    current_rate=current_rate,
                    limit=rate_limit + burst_allowance,
                    reset_time=datetime.utcnow() + timedelta(minutes=1),
                    reason="Burst traffic allowed for trusted user",
                    confidence=user_profile.trust_score * (1.0 - burst_prob)
                )
        
        # Exceeded limits - calculate throttling
        excess_rate = current_rate - rate_limit
        throttle_duration = min(300, excess_rate * 10)  # Max 5 minutes
        
        # Adjust based on user tier and trust
        if user_profile.tier in ['premium', 'enterprise']:
            throttle_duration *= 0.5
        
        if user_profile.trust_score > 0.8:
            throttle_duration *= 0.7
        
        return ThrottlingDecision(
            allowed=False,
            current_rate=current_rate,
            limit=rate_limit,
            reset_time=datetime.utcnow() + timedelta(seconds=throttle_duration),
            throttle_duration=throttle_duration,
            reason=f"Rate limit exceeded: {current_rate:.1f} > {rate_limit}",
            confidence=0.9,
            suggested_retry_after=int(throttle_duration)
        )
    
    async def _update_request_tracking(self, request: APIRequest, decision: ThrottlingDecision):
        """Update tracking data after decision"""
        user_id = request.user_id
        endpoint = request.endpoint
        now = request.timestamp
        
        # Update request windows
        if user_id not in self.request_windows:
            self.request_windows[user_id] = defaultdict(deque)
        
        if decision.allowed:
            self.request_windows[user_id][endpoint].append(now)
        
        # Update user profile
        user_profile = await self._get_user_profile(user_id)
        
        # Calculate interval since last request
        if user_profile.historical_patterns:
            last_request = user_profile.historical_patterns[-1]['timestamp']
            interval = (now - last_request).total_seconds()
        else:
            interval = 60.0
        
        # Add to historical patterns
        pattern_data = {
            'timestamp': now,
            'endpoint': endpoint,
            'allowed': decision.allowed,
            'rate': decision.current_rate,
            'interval': interval,
            'hour': now.hour,
            'anomaly_score': getattr(decision, 'anomaly_score', 0.0)
        }
        
        user_profile.historical_patterns.append(pattern_data)
        user_profile.last_updated = now
        
        # Update trust score based on behavior
        if decision.allowed:
            user_profile.trust_score = min(1.0, user_profile.trust_score + 0.001)
        else:
            user_profile.trust_score = max(0.0, user_profile.trust_score - 0.01)
        
        # Store in global history
        self.request_history.append({
            'request': request,
            'decision': decision,
            'timestamp': now
        })
        
        # Update system metrics
        self.system_metrics['total_requests'].append(now)
        if not decision.allowed:
            self.system_metrics['throttled_requests'].append(now)
    
    async def _record_decision_feedback(self, request: APIRequest, decision: ThrottlingDecision, 
                                      features: np.ndarray):
        """Record decision for model training"""
        # This would be used for online learning - store training examples
        feedback_data = {
            'features': features.tolist(),
            'decision_allowed': decision.allowed,
            'actual_rate': decision.current_rate,
            'timestamp': request.timestamp.isoformat()
        }
        
        # In production, this would go to a training data store
        # For now, we just log it
        if len(self.request_history) % 100 == 0:  # Periodically retrain
            asyncio.create_task(self._retrain_models())
    
    async def _fallback_rate_check(self, request: APIRequest) -> ThrottlingDecision:
        """Simple fallback rate limiting"""
        current_rate = self._calculate_current_rate(request.user_id, request.endpoint)
        limit = self.rate_limits.get(request.endpoint, self.rate_limits['default'])['requests_per_minute']
        
        return ThrottlingDecision(
            allowed=current_rate < limit,
            current_rate=current_rate,
            limit=limit,
            reset_time=datetime.utcnow() + timedelta(minutes=1),
            reason="Fallback rate limiting",
            confidence=0.5
        )
    
    def _get_current_system_load(self) -> Dict[str, float]:
        """Get current system load metrics"""
        # In production, this would query actual system metrics
        # For now, simulate based on recent activity
        
        recent_requests = [r for r in self.request_history if 
                          (datetime.utcnow() - r['timestamp']).total_seconds() < 60]
        
        request_rate = len(recent_requests) / 60.0
        
        return {
            'cpu': min(1.0, request_rate / 100.0),
            'memory': min(1.0, request_rate / 150.0),
            'network': min(1.0, request_rate / 80.0),
            'overall': min(1.0, request_rate / 120.0)
        }
    
    # Training data generation methods
    def _generate_usage_training_data(self) -> Tuple[List[np.ndarray], List[float]]:
        """Generate synthetic training data for usage prediction"""
        X, y = [], []
        
        for i in range(200):
            features = np.random.rand(18)
            
            # Synthetic usage pattern
            hour_feature = features[0]  # Hour of day
            load_feature = features[15] if len(features) > 15 else 0.5  # System load
            
            # Higher usage during business hours and when system is loaded
            base_usage = 30.0
            if 0.3 < hour_feature < 0.7:  # Business hours (approx)
                base_usage *= 2.5
            
            usage = base_usage * (1 + load_feature) + np.random.normal(0, 5)
            usage = max(1.0, usage)
            
            X.append(features)
            y.append(usage)
        
        return X, y
    
    def _generate_anomaly_training_data(self) -> List[np.ndarray]:
        """Generate synthetic training data for anomaly detection"""
        X = []
        
        # Normal patterns
        for i in range(150):
            features = np.random.normal(0.5, 0.1, 18)  # Normal distribution around 0.5
            X.append(features)
        
        # Anomalous patterns
        for i in range(30):
            features = np.random.exponential(0.8, 18)  # Different distribution for anomalies
            X.append(features)
        
        return X
    
    def _generate_burst_training_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Generate synthetic training data for burst detection"""
        X, y = [], []
        
        for i in range(100):
            features = np.random.rand(18)
            
            # Label burst if current rate is high and variance is high
            current_rate = features[8] if len(features) > 8 else 0.5
            rate_variance = np.var(features[9:14]) if len(features) > 14 else 0.1
            
            is_burst = int(current_rate > 0.7 and rate_variance > 0.3)
            
            X.append(features)
            y.append(is_burst)
        
        return X, y
    
    async def _retrain_models(self):
        """Retrain ML models with recent data"""
        try:
            if len(self.request_history) < 50:
                return
            
            # Extract training data from recent history
            X, y_usage, y_anomaly = [], [], []
            
            for record in list(self.request_history)[-100:]:
                # Would extract features and labels from actual data
                # This is a placeholder for the retraining logic
                pass
            
            logger.info("Models retrained with recent data")
        
        except Exception as e:
            logger.error(f"Model retraining error: {e}")
    
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get AI-powered analytics for a user"""
        user_profile = await self._get_user_profile(user_id)
        
        if not user_profile.historical_patterns:
            return {'error': 'No data available for user'}
        
        patterns = list(user_profile.historical_patterns)
        
        # Calculate usage statistics
        hourly_usage = defaultdict(int)
        endpoint_usage = defaultdict(int)
        intervals = []
        
        for pattern in patterns:
            hourly_usage[pattern['hour']] += 1
            endpoint_usage[pattern['endpoint']] += 1
            intervals.append(pattern['interval'])
        
        # ML insights
        avg_interval = np.mean(intervals)
        usage_consistency = 1.0 - np.std(intervals) / max(avg_interval, 1)
        
        peak_hours = sorted(hourly_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        top_endpoints = sorted(endpoint_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'user_id': user_id,
            'tier': user_profile.tier,
            'trust_score': user_profile.trust_score,
            'anomaly_score': user_profile.anomaly_score,
            'total_requests': len(patterns),
            'avg_request_interval': avg_interval,
            'usage_consistency': usage_consistency,
            'peak_hours': [{'hour': h, 'requests': c} for h, c in peak_hours],
            'top_endpoints': [{'endpoint': e, 'requests': c} for e, c in top_endpoints],
            'recent_throttling': sum(1 for p in patterns[-20:] if not p.get('allowed', True))
        }


# Singleton instance
_ai_rate_limiter = None

def get_ai_rate_limiter() -> AIRateLimiter:
    """Get or create AI rate limiter instance"""
    global _ai_rate_limiter
    if not _ai_rate_limiter:
        _ai_rate_limiter = AIRateLimiter()
    return _ai_rate_limiter