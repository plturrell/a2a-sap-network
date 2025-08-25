"""
Adaptive Rate Limiting System for A2A Platform
AI-powered rate limiting that adapts to user behavior patterns and threat detection
"""

import asyncio
import time
import logging
import json
import hashlib
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.getLogger(__name__).warning("ML libraries not available for adaptive rate limiting")

logger = logging.getLogger(__name__)


class UserBehaviorClass(Enum):
    """AI-classified user behavior patterns"""
    LEGITIMATE = "legitimate"
    SUSPICIOUS = "suspicious"
    BOT = "bot"
    BURST = "burst"
    CRAWLER = "crawler"
    ATTACK = "attack"


class RateLimitAction(Enum):
    """Actions to take when rate limit is exceeded"""
    ALLOW = "allow"
    THROTTLE = "throttle"
    BLOCK = "block"
    CHALLENGE = "challenge"  # CAPTCHA, 2FA, etc.


@dataclass
class RequestPattern:
    """AI-analyzed request pattern"""
    user_id: str
    endpoint: str
    timestamp: float
    request_size: int
    response_time: float
    status_code: int
    user_agent: str
    ip_address: str
    geographic_location: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class UserBehaviorProfile:
    """ML-based user behavior profile"""
    user_id: str
    behavior_class: UserBehaviorClass
    confidence: float
    request_patterns: deque
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_request_rate: float = 0.0
    peak_request_rate: float = 0.0
    typical_endpoints: Set[str] = None
    typical_times: List[int] = None  # Hours of typical activity
    anomaly_score: float = 0.0
    last_updated: float = 0.0
    adaptive_limit: int = 100  # Dynamic rate limit
    trust_score: float = 1.0  # 0-1, higher is more trusted

    def __post_init__(self):
        if self.typical_endpoints is None:
            self.typical_endpoints = set()
        if self.typical_times is None:
            self.typical_times = []
        if not hasattr(self, 'request_patterns') or self.request_patterns is None:
            self.request_patterns = deque(maxlen=1000)


@dataclass
class RateLimitRule:
    """Dynamic rate limiting rule"""
    rule_id: str
    endpoint_pattern: str
    base_limit: int
    window_seconds: int
    behavior_multipliers: Dict[UserBehaviorClass, float]
    adaptive: bool = True
    ml_enhanced: bool = True


class AIBehaviorAnalyzer:
    """AI system for analyzing user behavior patterns"""

    def __init__(self):
        self.anomaly_detector = None
        self.clustering_model = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.behavior_patterns = defaultdict(list)
        self.training_data = []
        self.model_trained = False
        self.last_training = 0.0
        self.training_interval = 3600.0  # Retrain every hour

        if ML_AVAILABLE:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.clustering_model = DBSCAN(eps=0.5, min_samples=5)

    def analyze_user_behavior(self, user_profile: UserBehaviorProfile) -> Tuple[UserBehaviorClass, float]:
        """Analyze user behavior using AI models"""

        # Rule-based classification first
        rule_classification = self._rule_based_classification(user_profile)

        # ML-based classification if available
        if ML_AVAILABLE and self.model_trained:
            ml_classification = self._ml_based_classification(user_profile)

            # Combine rule-based and ML-based results
            if ml_classification[1] > 0.7:  # High confidence ML prediction
                return ml_classification
            elif rule_classification[1] > 0.8:  # High confidence rule-based
                return rule_classification
            else:
                # Weighted combination
                rule_weight = 0.4
                ml_weight = 0.6

                # This is a simplified combination - in practice would be more sophisticated
                if rule_classification[0] == ml_classification[0]:
                    confidence = max(rule_classification[1], ml_classification[1])
                    return rule_classification[0], confidence
                else:
                    # Take higher confidence prediction
                    return ml_classification if ml_classification[1] > rule_classification[1] else rule_classification

        return rule_classification

    def _rule_based_classification(self, user_profile: UserBehaviorProfile) -> Tuple[UserBehaviorClass, float]:
        """Rule-based behavior classification"""

        # Calculate request rate patterns
        recent_requests = list(user_profile.request_patterns)[-100:]  # Last 100 requests
        if not recent_requests:
            return UserBehaviorClass.LEGITIMATE, 0.5

        # Time-based analysis
        current_time = time.time()
        last_hour_requests = [r for r in recent_requests if current_time - r.timestamp < 3600]
        last_minute_requests = [r for r in recent_requests if current_time - r.timestamp < 60]

        requests_per_minute = len(last_minute_requests)
        requests_per_hour = len(last_hour_requests)

        # Endpoint diversity
        unique_endpoints = len(set(r.endpoint for r in recent_requests))
        endpoint_diversity = unique_endpoints / max(1, len(recent_requests))

        # Error rate
        error_requests = [r for r in recent_requests if r.status_code >= 400]
        error_rate = len(error_requests) / max(1, len(recent_requests))

        # User agent analysis
        user_agents = [r.user_agent for r in recent_requests if r.user_agent]
        ua_diversity = len(set(user_agents)) / max(1, len(user_agents))

        # Classification logic
        confidence = 0.7

        # Attack patterns
        if requests_per_minute > 100 and error_rate > 0.7:
            return UserBehaviorClass.ATTACK, 0.9

        # Bot patterns
        if (requests_per_minute > 30 and
            endpoint_diversity < 0.1 and
            ua_diversity < 0.1):
            return UserBehaviorClass.BOT, 0.8

        # Suspicious patterns
        if (requests_per_minute > 50 or
            error_rate > 0.5 or
            (endpoint_diversity > 0.8 and requests_per_minute > 10)):
            return UserBehaviorClass.SUSPICIOUS, 0.7

        # Burst patterns (legitimate but high volume)
        if requests_per_minute > 20 and error_rate < 0.1:
            return UserBehaviorClass.BURST, 0.6

        # Crawler patterns
        if (endpoint_diversity > 0.5 and
            requests_per_minute < 10 and
            error_rate < 0.2):
            return UserBehaviorClass.CRAWLER, 0.6

        # Default to legitimate
        return UserBehaviorClass.LEGITIMATE, confidence

    def _ml_based_classification(self, user_profile: UserBehaviorProfile) -> Tuple[UserBehaviorClass, float]:
        """ML-based behavior classification"""
        if not ML_AVAILABLE or not self.model_trained:
            return UserBehaviorClass.LEGITIMATE, 0.5

        try:
            features = self._extract_behavior_features(user_profile)
            if features is None:
                return UserBehaviorClass.LEGITIMATE, 0.5

            # Anomaly detection
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1

            # Update user profile with anomaly score
            user_profile.anomaly_score = float(anomaly_score)

            # Classification based on anomaly score
            if is_anomaly and anomaly_score < -0.5:
                if user_profile.avg_request_rate > 50:
                    return UserBehaviorClass.ATTACK, 0.85
                else:
                    return UserBehaviorClass.SUSPICIOUS, 0.75
            elif is_anomaly:
                return UserBehaviorClass.BOT, 0.7
            else:
                return UserBehaviorClass.LEGITIMATE, 0.8

        except Exception as e:
            logger.error(f"ML classification error: {e}")
            return UserBehaviorClass.LEGITIMATE, 0.5

    def _extract_behavior_features(self, user_profile: UserBehaviorProfile) -> Optional[np.ndarray]:
        """Extract features for ML behavior analysis"""
        if not ML_AVAILABLE:
            return None

        try:
            recent_requests = list(user_profile.request_patterns)[-50:]  # Last 50 requests
            if len(recent_requests) < 5:
                return None

            # Time-based features
            timestamps = [r.timestamp for r in recent_requests]
            time_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]

            # Request rate features
            avg_time_between_requests = np.mean(time_diffs) if time_diffs else 0
            std_time_between_requests = np.std(time_diffs) if len(time_diffs) > 1 else 0

            # Endpoint features
            endpoints = [r.endpoint for r in recent_requests]
            unique_endpoints = len(set(endpoints))
            endpoint_diversity = unique_endpoints / len(endpoints)

            # Status code features
            status_codes = [r.status_code for r in recent_requests]
            success_rate = len([s for s in status_codes if 200 <= s < 400]) / len(status_codes)
            error_4xx_rate = len([s for s in status_codes if 400 <= s < 500]) / len(status_codes)
            error_5xx_rate = len([s for s in status_codes if s >= 500]) / len(status_codes)

            # Request size features
            request_sizes = [r.request_size for r in recent_requests if r.request_size]
            avg_request_size = np.mean(request_sizes) if request_sizes else 0
            std_request_size = np.std(request_sizes) if len(request_sizes) > 1 else 0

            # Response time features
            response_times = [r.response_time for r in recent_requests if r.response_time]
            avg_response_time = np.mean(response_times) if response_times else 0

            # User agent features
            user_agents = [r.user_agent for r in recent_requests if r.user_agent]
            ua_diversity = len(set(user_agents)) / max(1, len(user_agents))

            # IP address features
            ip_addresses = [r.ip_address for r in recent_requests if r.ip_address]
            ip_diversity = len(set(ip_addresses)) / max(1, len(ip_addresses))

            # Temporal features (hour of day distribution)
            hours = [datetime.fromtimestamp(r.timestamp).hour for r in recent_requests]
            hour_diversity = len(set(hours)) / 24  # Normalized by 24 hours

            features = np.array([
                avg_time_between_requests,
                std_time_between_requests,
                endpoint_diversity,
                success_rate,
                error_4xx_rate,
                error_5xx_rate,
                avg_request_size,
                std_request_size,
                avg_response_time,
                ua_diversity,
                ip_diversity,
                hour_diversity,
                len(recent_requests),  # Volume
                user_profile.total_requests,
                user_profile.avg_request_rate
            ])

            return features

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    async def train_models(self, training_profiles: List[UserBehaviorProfile]):
        """Train ML models with behavior data"""
        if not ML_AVAILABLE or len(training_profiles) < 100:
            return

        try:
            # Extract features from all profiles
            X = []
            for profile in training_profiles:
                features = self._extract_behavior_features(profile)
                if features is not None:
                    X.append(features)

            if len(X) < 50:
                logger.info("Insufficient training data for behavior models")
                return

            X = np.array(X)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train anomaly detector
            self.anomaly_detector.fit(X_scaled)

            # Train clustering model
            cluster_labels = self.clustering_model.fit_predict(X_scaled)

            self.model_trained = True
            self.last_training = time.time()

            logger.info(f"Behavior analysis models trained with {len(X)} samples")
            logger.info(f"Found {len(set(cluster_labels))} behavior clusters")

        except Exception as e:
            logger.error(f"Model training error: {e}")


class AdaptiveRateLimiter:
    """AI-powered adaptive rate limiting system"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None

        # AI components
        self.behavior_analyzer = AIBehaviorAnalyzer()
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}

        # Rate limiting rules
        self.rules: Dict[str, RateLimitRule] = {}

        # Performance tracking
        self.request_history = deque(maxlen=10000)
        self.blocked_requests = deque(maxlen=1000)
        self.false_positives = deque(maxlen=500)

        # Initialize default rules
        self._initialize_default_rules()

    async def initialize(self):
        """Initialize the adaptive rate limiter"""
        self.redis = redis.from_url(self.redis_url, decode_responses=True)
        logger.info("Adaptive rate limiter initialized")

    def _initialize_default_rules(self):
        """Initialize default rate limiting rules"""

        # API endpoints
        self.rules["api_general"] = RateLimitRule(
            rule_id="api_general",
            endpoint_pattern="/api/*",
            base_limit=100,
            window_seconds=60,
            behavior_multipliers={
                UserBehaviorClass.LEGITIMATE: 1.0,
                UserBehaviorClass.BURST: 1.5,
                UserBehaviorClass.CRAWLER: 0.5,
                UserBehaviorClass.BOT: 0.1,
                UserBehaviorClass.SUSPICIOUS: 0.3,
                UserBehaviorClass.ATTACK: 0.05
            }
        )

        # Authentication endpoints
        self.rules["auth"] = RateLimitRule(
            rule_id="auth",
            endpoint_pattern="/auth/*",
            base_limit=5,
            window_seconds=60,
            behavior_multipliers={
                UserBehaviorClass.LEGITIMATE: 1.0,
                UserBehaviorClass.BURST: 0.8,
                UserBehaviorClass.CRAWLER: 0.2,
                UserBehaviorClass.BOT: 0.1,
                UserBehaviorClass.SUSPICIOUS: 0.1,
                UserBehaviorClass.ATTACK: 0.01
            }
        )

        # File upload endpoints
        self.rules["upload"] = RateLimitRule(
            rule_id="upload",
            endpoint_pattern="/upload/*",
            base_limit=10,
            window_seconds=300,  # 5 minutes
            behavior_multipliers={
                UserBehaviorClass.LEGITIMATE: 1.0,
                UserBehaviorClass.BURST: 0.5,
                UserBehaviorClass.CRAWLER: 0.1,
                UserBehaviorClass.BOT: 0.05,
                UserBehaviorClass.SUSPICIOUS: 0.1,
                UserBehaviorClass.ATTACK: 0.01
            }
        )

    async def check_rate_limit(self,
                             user_id: str,
                             endpoint: str,
                             ip_address: str,
                             user_agent: str = "",
                             request_size: int = 0) -> Tuple[RateLimitAction, Dict[str, Any]]:
        """Check if request should be rate limited using AI analysis"""

        current_time = time.time()

        # Create or update user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserBehaviorProfile(
                user_id=user_id,
                behavior_class=UserBehaviorClass.LEGITIMATE,
                confidence=0.5,
                request_patterns=deque(maxlen=1000)
            )

        profile = self.user_profiles[user_id]

        # Record current request
        request_pattern = RequestPattern(
            user_id=user_id,
            endpoint=endpoint,
            timestamp=current_time,
            request_size=request_size,
            response_time=0.0,  # Will be updated after request
            status_code=0,  # Will be updated after request
            user_agent=user_agent,
            ip_address=ip_address
        )

        profile.request_patterns.append(request_pattern)
        profile.total_requests += 1
        profile.last_updated = current_time

        # Analyze user behavior
        behavior_class, confidence = self.behavior_analyzer.analyze_user_behavior(profile)
        profile.behavior_class = behavior_class
        profile.confidence = confidence

        # Update trust score based on behavior
        profile.trust_score = self._calculate_trust_score(profile, behavior_class, confidence)

        # Find applicable rate limiting rule
        applicable_rule = self._find_applicable_rule(endpoint)
        if not applicable_rule:
            return RateLimitAction.ALLOW, {"reason": "no_applicable_rule"}

        # Calculate adaptive rate limit
        adaptive_limit = self._calculate_adaptive_limit(profile, applicable_rule)
        profile.adaptive_limit = adaptive_limit

        # Check current request count in window
        window_start = current_time - applicable_rule.window_seconds
        recent_requests = [r for r in profile.request_patterns
                         if r.timestamp >= window_start and
                         self._endpoint_matches_pattern(r.endpoint, applicable_rule.endpoint_pattern)]

        current_count = len(recent_requests)

        # Determine action based on count and behavior
        action, details = self._determine_action(
            current_count,
            adaptive_limit,
            profile,
            applicable_rule
        )

        # Record decision for learning
        await self._record_decision(user_id, endpoint, action, details)

        return action, {
            "current_count": current_count,
            "adaptive_limit": adaptive_limit,
            "behavior_class": behavior_class.value,
            "confidence": confidence,
            "trust_score": profile.trust_score,
            "rule_id": applicable_rule.rule_id,
            **details
        }

    def _calculate_trust_score(self,
                             profile: UserBehaviorProfile,
                             behavior_class: UserBehaviorClass,
                             confidence: float) -> float:
        """Calculate trust score for user based on behavior history"""

        base_trust = 0.5

        # Behavior class influence
        behavior_trust_map = {
            UserBehaviorClass.LEGITIMATE: 1.0,
            UserBehaviorClass.BURST: 0.8,
            UserBehaviorClass.CRAWLER: 0.6,
            UserBehaviorClass.BOT: 0.3,
            UserBehaviorClass.SUSPICIOUS: 0.2,
            UserBehaviorClass.ATTACK: 0.05
        }

        behavior_trust = behavior_trust_map.get(behavior_class, 0.5)

        # Historical success rate influence
        if profile.total_requests > 0:
            success_rate = profile.successful_requests / profile.total_requests
            success_influence = success_rate * 0.3
        else:
            success_influence = 0

        # Consistency influence (low anomaly score = higher trust)
        consistency_influence = max(0, (0.5 - abs(profile.anomaly_score)) * 0.2)

        # Time-based trust (longer legitimate behavior = higher trust)
        if profile.total_requests > 100:
            longevity_bonus = min(0.2, profile.total_requests / 1000)
        else:
            longevity_bonus = 0

        # Combine factors
        trust_score = (
            behavior_trust * confidence * 0.5 +
            success_influence +
            consistency_influence +
            longevity_bonus
        )

        # Smooth trust score changes
        if hasattr(profile, 'trust_score'):
            trust_score = profile.trust_score * 0.7 + trust_score * 0.3

        return max(0.01, min(1.0, trust_score))

    def _find_applicable_rule(self, endpoint: str) -> Optional[RateLimitRule]:
        """Find the most specific applicable rate limiting rule"""

        # Find all matching rules
        matching_rules = []
        for rule in self.rules.values():
            if self._endpoint_matches_pattern(endpoint, rule.endpoint_pattern):
                matching_rules.append((rule, len(rule.endpoint_pattern)))

        if not matching_rules:
            return None

        # Return most specific rule (longest pattern)
        return max(matching_rules, key=lambda x: x[1])[0]

    def _endpoint_matches_pattern(self, endpoint: str, pattern: str) -> bool:
        """Check if endpoint matches pattern (simple wildcard matching)"""

        if pattern == "*":
            return True

        if pattern.endswith("*"):
            return endpoint.startswith(pattern[:-1])

        if pattern.startswith("*"):
            return endpoint.endswith(pattern[1:])

        return endpoint == pattern

    def _calculate_adaptive_limit(self,
                                profile: UserBehaviorProfile,
                                rule: RateLimitRule) -> int:
        """Calculate adaptive rate limit based on user behavior and trust"""

        base_limit = rule.base_limit

        # Behavior multiplier
        behavior_multiplier = rule.behavior_multipliers.get(
            profile.behavior_class, 1.0
        )

        # Trust score influence
        trust_multiplier = 0.5 + (profile.trust_score * 0.5)  # Range: 0.5 - 1.0

        # System load influence (simplified)
        system_load_multiplier = 1.0  # Could be enhanced with actual system metrics

        # Calculate final limit
        adaptive_limit = int(
            base_limit *
            behavior_multiplier *
            trust_multiplier *
            system_load_multiplier
        )

        # Ensure minimum limit
        return max(1, adaptive_limit)

    def _determine_action(self,
                        current_count: int,
                        adaptive_limit: int,
                        profile: UserBehaviorProfile,
                        rule: RateLimitRule) -> Tuple[RateLimitAction, Dict[str, Any]]:
        """Determine what action to take based on current state"""

        details = {}

        # Simple threshold-based decision
        if current_count < adaptive_limit:
            return RateLimitAction.ALLOW, details

        # Exceeded limit - determine response based on behavior and trust
        if profile.behavior_class == UserBehaviorClass.ATTACK:
            return RateLimitAction.BLOCK, {"reason": "attack_behavior", "block_duration": 3600}

        elif profile.behavior_class == UserBehaviorClass.SUSPICIOUS:
            if profile.trust_score < 0.3:
                return RateLimitAction.BLOCK, {"reason": "suspicious_low_trust", "block_duration": 300}
            else:
                return RateLimitAction.CHALLENGE, {"reason": "suspicious_behavior"}

        elif profile.behavior_class == UserBehaviorClass.BOT:
            return RateLimitAction.THROTTLE, {"reason": "bot_behavior", "delay": 5}

        elif profile.behavior_class == UserBehaviorClass.BURST:
            if profile.trust_score > 0.7:
                return RateLimitAction.THROTTLE, {"reason": "burst_trusted", "delay": 1}
            else:
                return RateLimitAction.THROTTLE, {"reason": "burst_untrusted", "delay": 3}

        else:  # LEGITIMATE or CRAWLER
            if profile.trust_score > 0.8:
                return RateLimitAction.THROTTLE, {"reason": "trusted_user", "delay": 1}
            else:
                return RateLimitAction.THROTTLE, {"reason": "limit_exceeded", "delay": 2}

    async def _record_decision(self,
                             user_id: str,
                             endpoint: str,
                             action: RateLimitAction,
                             details: Dict[str, Any]):
        """Record rate limiting decision for analysis and learning"""

        try:
            decision_record = {
                "user_id": user_id,
                "endpoint": endpoint,
                "action": action.value,
                "details": details,
                "timestamp": time.time()
            }

            # Store in Redis for persistence and analysis
            await self.redis.lpush(
                "rate_limit_decisions",
                json.dumps(decision_record)
            )

            # Keep only recent decisions
            await self.redis.ltrim("rate_limit_decisions", 0, 9999)

        except Exception as e:
            logger.error(f"Failed to record decision: {e}")

    async def update_request_outcome(self,
                                   user_id: str,
                                   endpoint: str,
                                   status_code: int,
                                   response_time: float):
        """Update request outcome for learning and user profile refinement"""

        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]

            # Update success/failure counts
            if 200 <= status_code < 400:
                profile.successful_requests += 1
            else:
                profile.failed_requests += 1

            # Update latest request pattern with outcome
            if profile.request_patterns:
                latest_request = profile.request_patterns[-1]
                latest_request.status_code = status_code
                latest_request.response_time = response_time

    async def report_false_positive(self, user_id: str, endpoint: str, timestamp: float):
        """Report a false positive for model improvement"""

        false_positive_record = {
            "user_id": user_id,
            "endpoint": endpoint,
            "timestamp": timestamp,
            "reported_at": time.time()
        }

        self.false_positives.append(false_positive_record)

        # If user has profile, adjust trust score
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            profile.trust_score = min(1.0, profile.trust_score + 0.1)
            logger.info(f"Adjusted trust score for {user_id} due to false positive report")

    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a user"""

        if user_id not in self.user_profiles:
            return {"error": "User not found"}

        profile = self.user_profiles[user_id]

        # Calculate recent activity
        recent_requests = list(profile.request_patterns)[-100:]
        recent_activity = {}

        if recent_requests:
            # Endpoint distribution
            endpoints = [r.endpoint for r in recent_requests]
            endpoint_counts = defaultdict(int)
            for endpoint in endpoints:
                endpoint_counts[endpoint] += 1

            # Time distribution
            hours = [datetime.fromtimestamp(r.timestamp).hour for r in recent_requests]
            hour_counts = defaultdict(int)
            for hour in hours:
                hour_counts[hour] += 1

            recent_activity = {
                "endpoint_distribution": dict(endpoint_counts),
                "hour_distribution": dict(hour_counts),
                "request_rate_last_hour": len([r for r in recent_requests
                                             if time.time() - r.timestamp < 3600]),
                "error_rate": len([r for r in recent_requests if r.status_code >= 400]) / len(recent_requests)
            }

        return {
            "user_id": user_id,
            "behavior_class": profile.behavior_class.value,
            "confidence": profile.confidence,
            "trust_score": profile.trust_score,
            "total_requests": profile.total_requests,
            "successful_requests": profile.successful_requests,
            "failed_requests": profile.failed_requests,
            "current_adaptive_limit": profile.adaptive_limit,
            "anomaly_score": profile.anomaly_score,
            "recent_activity": recent_activity,
            "last_updated": profile.last_updated
        }

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide rate limiting metrics"""

        total_users = len(self.user_profiles)

        # Behavior class distribution
        behavior_distribution = defaultdict(int)
        trust_scores = []

        for profile in self.user_profiles.values():
            behavior_distribution[profile.behavior_class.value] += 1
            trust_scores.append(profile.trust_score)

        return {
            "total_users": total_users,
            "behavior_distribution": dict(behavior_distribution),
            "average_trust_score": np.mean(trust_scores) if trust_scores else 0,
            "ml_available": ML_AVAILABLE,
            "model_trained": self.behavior_analyzer.model_trained,
            "last_model_training": self.behavior_analyzer.last_training,
            "total_decisions": len(self.request_history),
            "recent_false_positives": len(self.false_positives),
            "active_rules": len(self.rules)
        }


# Global instance for easy access
adaptive_rate_limiter = None

async def initialize_adaptive_rate_limiter(redis_url: str = "redis://localhost:6379"):
    """Initialize the global adaptive rate limiter"""
    global adaptive_rate_limiter
    adaptive_rate_limiter = AdaptiveRateLimiter(redis_url)
    await adaptive_rate_limiter.initialize()
    logger.info("Global adaptive rate limiter initialized")

async def shutdown_adaptive_rate_limiter():
    """Shutdown the global adaptive rate limiter"""
    global adaptive_rate_limiter
    if adaptive_rate_limiter and adaptive_rate_limiter.redis:
        await adaptive_rate_limiter.redis.close()
        logger.info("Adaptive rate limiter shutdown complete")
