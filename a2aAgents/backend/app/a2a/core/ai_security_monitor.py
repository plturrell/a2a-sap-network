"""
AI-Driven Security Monitoring System

This module provides intelligent security monitoring using real machine learning
for threat detection, anomaly identification, behavioral analysis, and automated
response without relying on external AI services.
"""

import asyncio
import logging
import numpy as np
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import ipaddress
import re

# Machine Learning imports
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Deep learning for advanced threat detection
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecurityThreatNN(nn.Module):
    """Neural network for advanced threat detection"""
    def __init__(self, input_dim, hidden_dim=256):
        super(SecurityThreatNN, self).__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Multi-head threat detection
        self.malware_head = nn.Linear(hidden_dim // 4, 1)
        self.intrusion_head = nn.Linear(hidden_dim // 4, 1)
        self.anomaly_head = nn.Linear(hidden_dim // 4, 1)
        self.data_breach_head = nn.Linear(hidden_dim // 4, 1)
        self.ddos_head = nn.Linear(hidden_dim // 4, 1)
        
        # Attention mechanism for feature importance
        self.attention = nn.MultiheadAttention(hidden_dim // 4, num_heads=8)
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply attention
        attn_features, attn_weights = self.attention(
            features.unsqueeze(0), features.unsqueeze(0), features.unsqueeze(0)
        )
        enhanced_features = attn_features.squeeze(0)
        
        # Threat predictions
        malware = torch.sigmoid(self.malware_head(enhanced_features))
        intrusion = torch.sigmoid(self.intrusion_head(enhanced_features))
        anomaly = torch.sigmoid(self.anomaly_head(enhanced_features))
        data_breach = torch.sigmoid(self.data_breach_head(enhanced_features))
        ddos = torch.sigmoid(self.ddos_head(enhanced_features))
        
        return malware, intrusion, anomaly, data_breach, ddos, attn_weights


class ThreatLevel(Enum):
    """Security threat levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    timestamp: datetime
    event_type: str
    source_ip: str
    target_ip: Optional[str] = None
    user_agent: Optional[str] = None
    payload: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    request_size: int = 0
    response_size: int = 0
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatDetection:
    """Detected security threat"""
    threat_id: str
    threat_type: str
    threat_level: ThreatLevel
    confidence: float
    description: str
    affected_resources: List[str]
    attack_vector: Optional[str] = None
    mitigation_suggestions: List[str] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_events: List[str] = field(default_factory=list)


@dataclass
class SecurityMetrics:
    """Security monitoring metrics"""
    total_events: int = 0
    threats_detected: int = 0
    false_positives: int = 0
    blocked_attempts: int = 0
    successful_attacks: int = 0
    response_time_avg: float = 0.0
    detection_accuracy: float = 0.0
    system_health_score: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class AISecurityMonitor:
    """
    AI-powered security monitoring system using real ML models
    """
    
    def __init__(self):
        # ML Models for different security aspects
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.intrusion_detector = RandomForestClassifier(n_estimators=150, random_state=42)
        self.malware_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.behavior_analyzer = MLPClassifier(hidden_layer_sizes=(128, 64), random_state=42)
        self.ddos_detector = OneClassSVM(nu=0.1, kernel='rbf')
        
        # Clustering for pattern recognition
        self.attack_clusterer = DBSCAN(eps=0.5, min_samples=5)
        self.user_behavior_clusterer = KMeans(n_clusters=10, random_state=42)
        
        # Feature scalers and encoders
        self.feature_scaler = StandardScaler()
        self.behavior_scaler = MinMaxScaler()
        self.ip_encoder = LabelEncoder()
        self.user_agent_encoder = LabelEncoder()
        
        # Neural network for advanced threat detection
        if TORCH_AVAILABLE:
            self.threat_nn = SecurityThreatNN(input_dim=50)
            self.nn_optimizer = torch.optim.Adam(self.threat_nn.parameters(), lr=0.001)
            self.nn_scheduler = torch.optim.lr_scheduler.StepLR(self.nn_optimizer, step_size=100, gamma=0.9)
        else:
            self.threat_nn = None
        
        # Security event storage and analysis
        self.security_events = deque(maxlen=10000)
        self.threat_detections = deque(maxlen=1000)
        self.user_sessions = defaultdict(list)
        self.ip_reputation = defaultdict(float)
        
        # Pattern databases
        self.attack_patterns = self._initialize_attack_patterns()
        self.behavioral_baselines = {}
        self.threat_signatures = self._initialize_threat_signatures()
        
        # Real-time monitoring
        self.monitoring_active = False
        self.alert_thresholds = self._initialize_alert_thresholds()
        
        # Performance metrics
        self.metrics = SecurityMetrics()
        
        # Initialize models with training data
        self._initialize_models()
        
        logger.info("AI Security Monitor initialized with ML models")
    
    def _initialize_attack_patterns(self) -> Dict[str, List[str]]:
        """Initialize known attack patterns"""
        return {
            'sql_injection': [
                r"(\bUNION\b.*\bSELECT\b)",
                r"(\bOR\b.*=.*)",
                r"(';.*--)",
                r"(\bDROP\b.*\bTABLE\b)",
                r"(\bINSERT\b.*\bINTO\b)",
                r"(\bDELETE\b.*\bFROM\b)"
            ],
            'xss': [
                r"<script.*?>.*?</script>",
                r"javascript:",
                r"onload\s*=",
                r"onerror\s*=",
                r"alert\s*\(",
                r"document\.cookie"
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e\\",
                r"\.\.%2f"
            ],
            'command_injection': [
                r";\s*(ls|cat|pwd|whoami)",
                r"\|\s*(ls|cat|pwd|whoami)",
                r"&&\s*(ls|cat|pwd|whoami)",
                r"`.*`",
                r"\$\(.*\)"
            ],
            'brute_force': [
                r"(admin|administrator|root|test)",
                r"(password|123456|admin|test)",
                r"(login|signin|auth)"
            ]
        }
    
    def _initialize_threat_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Initialize threat signatures"""
        return {
            'high_frequency_requests': {
                'threshold': 100,
                'time_window': 60,
                'threat_level': ThreatLevel.HIGH
            },
            'suspicious_user_agents': {
                'patterns': ['sqlmap', 'nikto', 'nmap', 'masscan', 'zap'],
                'threat_level': ThreatLevel.HIGH
            },
            'abnormal_request_size': {
                'threshold': 1000000,  # 1MB
                'threat_level': ThreatLevel.MEDIUM
            },
            'multiple_error_codes': {
                'error_threshold': 50,
                'time_window': 300,
                'threat_level': ThreatLevel.MEDIUM
            },
            'geographic_anomaly': {
                'countries_threshold': 5,
                'time_window': 3600,
                'threat_level': ThreatLevel.MEDIUM
            }
        }
    
    def _initialize_alert_thresholds(self) -> Dict[str, float]:
        """Initialize alert thresholds"""
        return {
            'anomaly_score': 0.7,
            'threat_confidence': 0.8,
            'behavioral_deviation': 0.6,
            'intrusion_probability': 0.75,
            'malware_likelihood': 0.85
        }
    
    def _initialize_models(self):
        """Initialize ML models - will train with real data as it arrives"""
        # Models will be trained incrementally with real security events
        # No synthetic data - models start untrained and learn from actual threats
        logger.info("Security models initialized - will train on real security events")
        
        # Set flag to indicate models need training
        self.models_trained = False
        
        # Schedule initial training attempt
        asyncio.create_task(self._train_with_real_data())
    
    async def _train_with_real_data(self):
        """Train models with real security data from the platform"""
        from .data_pipeline import get_data_pipeline
        pipeline = get_data_pipeline()
        
        try:
            # Wait for pipeline to be ready
            await asyncio.sleep(5)
            
            # Get real security events
            events_df = await pipeline.get_training_data('events', hours=24, min_samples=50)
            
            if len(events_df) < 10:
                logger.info("Waiting for more security events to train models")
                # Retry in 5 minutes
                await asyncio.sleep(300)
                asyncio.create_task(self._train_with_real_data())
                return
            
            # Extract features from real security events
            X_security = []
            y_labels = []
            
            for _, event in events_df.iterrows():
                # Only process security-relevant events
                if event['event_type'] in ['auth_failure', 'access_denied', 'suspicious_activity', 
                                          'rate_limit_exceeded', 'invalid_request']:
                    features = await self._extract_real_security_features(event)
                    X_security.append(features)
                    
                    # Label based on actual outcomes
                    if event['error_code'] in ['INTRUSION_DETECTED', 'MALWARE_FOUND', 'ATTACK_BLOCKED']:
                        y_labels.append(1)  # Confirmed threat
                    else:
                        y_labels.append(0)  # Normal/benign
            
            if len(X_security) > 10:
                X_security = np.array(X_security)
                y_labels = np.array(y_labels)
                
                # Train anomaly detector with real data
                X_scaled = self.feature_scaler.fit_transform(X_security)
                self.anomaly_detector.fit(X_scaled)
                
                # Train classifiers if we have both classes
                if len(np.unique(y_labels)) > 1:
                    self.intrusion_detector.fit(X_security, y_labels)
                    self.malware_classifier.fit(X_security, y_labels)
                    self.behavior_analyzer.fit(X_scaled, y_labels)
                
                self.models_trained = True
                logger.info(f"Security models trained with {len(X_security)} real events")
            
        except Exception as e:
            logger.error(f"Error training security models: {e}")
            # Retry training later
            await asyncio.sleep(300)
            asyncio.create_task(self._train_with_real_data())
    
    async def _extract_real_security_features(self, event: pd.Series) -> np.ndarray:
        """Extract security features from real event data"""
        features = []
        
        # Time-based features
        timestamp = datetime.fromtimestamp(event['timestamp'])
        features.append(timestamp.hour / 24.0)
        features.append(timestamp.weekday() / 7.0)
        features.append(1 if timestamp.hour < 6 or timestamp.hour > 22 else 0)  # Off-hours
        
        # Event characteristics
        features.append(1 if event['success'] else 0)
        features.append(event['duration_ms'] / 1000.0 if event['duration_ms'] else 0)
        
        # Error patterns
        error_keywords = ['auth', 'denied', 'invalid', 'blocked', 'limit', 'suspicious']
        error_score = sum(1 for kw in error_keywords if kw in str(event.get('error_code', '')).lower())
        features.append(error_score / len(error_keywords))
        
        # Extract context features if available
        if event.get('context'):
            context = event['context']
            features.append(context.get('retry_count', 0) / 10.0)
            features.append(context.get('request_size', 0) / 10000.0)
            features.append(1 if context.get('is_proxy', False) else 0)
            features.append(context.get('geo_risk_score', 0.5))
        else:
            features.extend([0, 0, 0, 0.5])
        
        # Pad to expected size
        while len(features) < 25:
            features.append(0.0)
        
        return np.array(features[:25], dtype=np.float32)
    
    async def analyze_security_event(self, event: SecurityEvent) -> List[ThreatDetection]:
        """
        Analyze a security event for threats using AI
        """
        threats_detected = []
        
        try:
            # Add event to history
            self.security_events.append(event)
            self.metrics.total_events += 1
            
            # Extract features from event
            event_features = self._extract_event_features(event)
            
            # Run multiple detection algorithms
            detections = await asyncio.gather(
                self._detect_anomalies(event, event_features),
                self._detect_intrusions(event, event_features),
                self._detect_malware(event, event_features),
                self._analyze_behavior(event, event_features),
                self._detect_ddos(event, event_features),
                self._pattern_matching_analysis(event),
                return_exceptions=True
            )
            
            # Collect valid detections
            for detection in detections:
                if isinstance(detection, list):
                    threats_detected.extend(detection)
                elif isinstance(detection, ThreatDetection):
                    threats_detected.append(detection)
            
            # Neural network enhancement
            if self.threat_nn and TORCH_AVAILABLE:
                nn_threats = await self._get_nn_threat_analysis(event_features)
                threats_detected.extend(nn_threats)
            
            # Correlation analysis for multi-stage attacks
            correlated_threats = await self._correlate_threats(threats_detected, event)
            threats_detected.extend(correlated_threats)
            
            # Update metrics
            if threats_detected:
                self.metrics.threats_detected += len(threats_detected)
                for threat in threats_detected:
                    self.threat_detections.append(threat)
            
            # Update IP reputation
            self._update_ip_reputation(event, threats_detected)
            
            return threats_detected
            
        except Exception as e:
            logger.error(f"Security analysis error: {e}")
            return []
    
    async def monitor_behavioral_patterns(self, time_window: int = 3600) -> Dict[str, Any]:
        """
        Monitor and analyze behavioral patterns over time
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
        recent_events = [e for e in self.security_events if e.timestamp > cutoff_time]
        
        if not recent_events:
            return {'status': 'no_recent_activity'}
        
        # Analyze patterns
        patterns = {
            'ip_patterns': self._analyze_ip_patterns(recent_events),
            'user_agent_patterns': self._analyze_user_agent_patterns(recent_events),
            'request_patterns': self._analyze_request_patterns(recent_events),
            'temporal_patterns': self._analyze_temporal_patterns(recent_events),
            'geographic_patterns': self._analyze_geographic_patterns(recent_events)
        }
        
        # Detect anomalous patterns
        anomalous_patterns = []
        for pattern_type, pattern_data in patterns.items():
            if self._is_pattern_anomalous(pattern_type, pattern_data):
                anomalous_patterns.append({
                    'type': pattern_type,
                    'data': pattern_data,
                    'anomaly_score': self._calculate_pattern_anomaly_score(pattern_data)
                })
        
        # Behavioral clustering
        behavior_clusters = await self._cluster_behaviors(recent_events)
        
        return {
            'time_window': time_window,
            'events_analyzed': len(recent_events),
            'patterns': patterns,
            'anomalous_patterns': anomalous_patterns,
            'behavior_clusters': behavior_clusters,
            'risk_assessment': self._assess_overall_risk(patterns, anomalous_patterns)
        }
    
    async def predict_security_threats(self, forecast_horizon: int = 3600) -> Dict[str, Any]:
        """
        Predict potential security threats using ML
        """
        # Analyze historical patterns
        historical_data = self._prepare_historical_data()
        
        # Extract trend features
        trend_features = self._extract_trend_features(historical_data)
        
        # Predict threat likelihood
        threat_predictions = {}
        
        # Individual threat type predictions
        threat_types = ['intrusion', 'malware', 'ddos', 'data_breach', 'brute_force']
        
        for threat_type in threat_types:
            # Use ML models to predict likelihood
            if threat_type == 'intrusion' and hasattr(self.intrusion_detector, 'predict_proba'):
                prob = self.intrusion_detector.predict_proba(trend_features.reshape(1, -1))[0][1]
            elif threat_type == 'malware' and hasattr(self.malware_classifier, 'predict_proba'):
                prob = self.malware_classifier.predict_proba(trend_features.reshape(1, -1))[0][1]
            else:
                # Heuristic prediction based on trends
                prob = self._heuristic_threat_prediction(threat_type, historical_data)
            
            threat_predictions[threat_type] = {
                'likelihood': float(prob),
                'confidence': self._calculate_prediction_confidence(trend_features),
                'forecast_horizon': forecast_horizon,
                'risk_factors': self._identify_risk_factors(threat_type, historical_data)
            }
        
        # Overall risk assessment
        overall_risk = np.mean([p['likelihood'] for p in threat_predictions.values()])
        
        return {
            'forecast_horizon': forecast_horizon,
            'threat_predictions': threat_predictions,
            'overall_risk_score': float(overall_risk),
            'recommended_actions': self._generate_preventive_actions(threat_predictions),
            'confidence': np.mean([p['confidence'] for p in threat_predictions.values()])
        }
    
    def _extract_event_features(self, event: SecurityEvent) -> np.ndarray:
        """Extract ML features from security event"""
        features = []
        
        # Temporal features
        hour = event.timestamp.hour
        day_of_week = event.timestamp.weekday()
        features.extend([hour / 24.0, day_of_week / 7.0])
        
        # IP features
        try:
            ip = ipaddress.ip_address(event.source_ip)
            features.append(1.0 if ip.is_private else 0.0)
            features.append(1.0 if ip.is_multicast else 0.0)
            features.append(1.0 if ip.is_loopback else 0.0)
            # IP reputation
            features.append(self.ip_reputation.get(event.source_ip, 0.5))
        except:
            features.extend([0.0, 0.0, 0.0, 0.5])
        
        # Request features
        features.append(event.request_size / 1000000.0)  # Normalize to MB
        features.append(event.response_size / 1000000.0)
        features.append((event.status_code or 200) / 500.0)  # Normalize status codes
        
        # User agent features
        if event.user_agent:
            ua_lower = event.user_agent.lower()
            features.append(1.0 if any(bot in ua_lower for bot in ['bot', 'crawler', 'spider']) else 0.0)
            features.append(1.0 if any(tool in ua_lower for tool in ['curl', 'wget', 'python', 'sqlmap']) else 0.0)
            features.append(len(event.user_agent) / 200.0)  # Normalize length
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Payload analysis
        if event.payload:
            payload_lower = event.payload.lower()
            # SQL injection indicators
            features.append(1.0 if any(pattern in payload_lower for pattern in ['union', 'select', 'drop', 'insert']) else 0.0)
            # XSS indicators
            features.append(1.0 if any(pattern in payload_lower for pattern in ['<script', 'javascript:', 'alert(']) else 0.0)
            # Command injection indicators
            features.append(1.0 if any(pattern in payload_lower for pattern in ['&&', '||', '`', '$(']) else 0.0)
            # Path traversal indicators
            features.append(1.0 if '../' in payload_lower or '..\\' in payload_lower else 0.0)
            features.append(len(event.payload) / 10000.0)  # Normalize length
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Session features
        if event.session_id:
            session_events = self.user_sessions.get(event.session_id, [])
            features.append(len(session_events) / 100.0)  # Normalize session activity
            if session_events:
                time_diff = (event.timestamp - session_events[-1].timestamp).total_seconds()
                features.append(min(time_diff / 3600.0, 1.0))  # Normalize to hours
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Historical context
        same_ip_events = [e for e in list(self.security_events)[-100:] if e.source_ip == event.source_ip]
        features.append(len(same_ip_events) / 100.0)
        
        # Error rate from this IP
        error_events = [e for e in same_ip_events if e.status_code and e.status_code >= 400]
        error_rate = len(error_events) / len(same_ip_events) if same_ip_events else 0.0
        features.append(error_rate)
        
        # Request frequency
        recent_events = [e for e in same_ip_events 
                        if (event.timestamp - e.timestamp).total_seconds() < 300]  # 5 minutes
        features.append(len(recent_events) / 50.0)  # Normalize
        
        return np.array(features)
    
    async def _detect_anomalies(self, event: SecurityEvent, features: np.ndarray) -> List[ThreatDetection]:
        """Detect anomalies using isolation forest"""
        threats = []
        
        try:
            if hasattr(self.anomaly_detector, 'decision_function'):
                features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
                anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
                
                # Negative scores indicate anomalies in IsolationForest
                if anomaly_score < -self.alert_thresholds['anomaly_score']:
                    threat = ThreatDetection(
                        threat_id=f"anomaly_{int(time.time())}_{hash(event.source_ip) % 10000}",
                        threat_type="anomaly",
                        threat_level=ThreatLevel.MEDIUM,
                        confidence=float(abs(anomaly_score)),
                        description=f"Anomalous behavior detected from IP {event.source_ip}",
                        affected_resources=[event.source_ip],
                        attack_vector="behavioral_anomaly",
                        source_events=[event.event_id]
                    )
                    threats.append(threat)
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
        
        return threats
    
    async def _detect_intrusions(self, event: SecurityEvent, features: np.ndarray) -> List[ThreatDetection]:
        """Detect intrusion attempts"""
        threats = []
        
        try:
            if hasattr(self.intrusion_detector, 'predict_proba'):
                intrusion_prob = self.intrusion_detector.predict_proba(features.reshape(1, -1))[0][1]
                
                if intrusion_prob > self.alert_thresholds['intrusion_probability']:
                    threat_level = ThreatLevel.HIGH if intrusion_prob > 0.9 else ThreatLevel.MEDIUM
                    
                    threat = ThreatDetection(
                        threat_id=f"intrusion_{int(time.time())}_{hash(event.source_ip) % 10000}",
                        threat_type="intrusion_attempt",
                        threat_level=threat_level,
                        confidence=float(intrusion_prob),
                        description=f"Potential intrusion attempt from {event.source_ip}",
                        affected_resources=[event.source_ip, event.target_ip or "unknown"],
                        attack_vector="network_intrusion",
                        mitigation_suggestions=[
                            "Block source IP address",
                            "Increase monitoring for this IP range",
                            "Review authentication logs"
                        ],
                        source_events=[event.event_id]
                    )
                    threats.append(threat)
        except Exception as e:
            logger.error(f"Intrusion detection error: {e}")
        
        return threats
    
    async def _detect_malware(self, event: SecurityEvent, features: np.ndarray) -> List[ThreatDetection]:
        """Detect malware-related activities"""
        threats = []
        
        try:
            if hasattr(self.malware_classifier, 'predict_proba'):
                malware_prob = self.malware_classifier.predict_proba(features.reshape(1, -1))[0][1]
                
                if malware_prob > self.alert_thresholds['malware_likelihood']:
                    threat = ThreatDetection(
                        threat_id=f"malware_{int(time.time())}_{hash(event.source_ip) % 10000}",
                        threat_type="malware",
                        threat_level=ThreatLevel.HIGH,
                        confidence=float(malware_prob),
                        description=f"Malware activity detected from {event.source_ip}",
                        affected_resources=[event.source_ip],
                        attack_vector="malware_communication",
                        mitigation_suggestions=[
                            "Isolate affected systems",
                            "Run comprehensive malware scan",
                            "Update security signatures"
                        ],
                        source_events=[event.event_id]
                    )
                    threats.append(threat)
        except Exception as e:
            logger.error(f"Malware detection error: {e}")
        
        return threats
    
    async def _analyze_behavior(self, event: SecurityEvent, features: np.ndarray) -> List[ThreatDetection]:
        """Analyze behavioral patterns"""
        threats = []
        
        try:
            if hasattr(self.behavior_analyzer, 'predict_proba'):
                features_scaled = self.behavior_scaler.transform(features.reshape(1, -1))
                behavior_class_prob = self.behavior_analyzer.predict_proba(features_scaled)[0]
                
                # Assuming class 1 is malicious behavior
                if len(behavior_class_prob) > 1 and behavior_class_prob[1] > self.alert_thresholds['behavioral_deviation']:
                    threat = ThreatDetection(
                        threat_id=f"behavior_{int(time.time())}_{hash(event.source_ip) % 10000}",
                        threat_type="behavioral_anomaly",
                        threat_level=ThreatLevel.MEDIUM,
                        confidence=float(behavior_class_prob[1]),
                        description=f"Abnormal behavioral pattern from {event.source_ip}",
                        affected_resources=[event.source_ip],
                        attack_vector="behavioral_pattern",
                        source_events=[event.event_id]
                    )
                    threats.append(threat)
        except Exception as e:
            logger.error(f"Behavior analysis error: {e}")
        
        return threats
    
    async def _detect_ddos(self, event: SecurityEvent, features: np.ndarray) -> List[ThreatDetection]:
        """Detect DDoS attacks"""
        threats = []
        
        try:
            # Check request frequency from same IP
            recent_events = [e for e in list(self.security_events)[-1000:] 
                           if e.source_ip == event.source_ip 
                           and (event.timestamp - e.timestamp).total_seconds() < 60]
            
            if len(recent_events) > 50:  # More than 50 requests per minute
                # Use ML model to confirm DDoS pattern
                if hasattr(self.ddos_detector, 'predict'):
                    ddos_score = self.ddos_detector.decision_function(features.reshape(1, -1))[0]
                    
                    if ddos_score < 0:  # Outlier detection
                        threat = ThreatDetection(
                            threat_id=f"ddos_{int(time.time())}_{hash(event.source_ip) % 10000}",
                            threat_type="ddos",
                            threat_level=ThreatLevel.HIGH,
                            confidence=0.8,
                            description=f"Potential DDoS attack from {event.source_ip} ({len(recent_events)} requests/min)",
                            affected_resources=[event.target_ip or "server"],
                            attack_vector="volumetric_attack",
                            mitigation_suggestions=[
                                "Rate limit requests from source IP",
                                "Enable DDoS protection",
                                "Monitor bandwidth usage"
                            ],
                            source_events=[event.event_id]
                        )
                        threats.append(threat)
        except Exception as e:
            logger.error(f"DDoS detection error: {e}")
        
        return threats
    
    async def _pattern_matching_analysis(self, event: SecurityEvent) -> List[ThreatDetection]:
        """Pattern-based threat detection"""
        threats = []
        
        if not event.payload:
            return threats
        
        payload_lower = event.payload.lower()
        
        for attack_type, patterns in self.attack_patterns.items():
            for pattern in patterns:
                if re.search(pattern, payload_lower):
                    threat_level = ThreatLevel.HIGH if attack_type in ['sql_injection', 'command_injection'] else ThreatLevel.MEDIUM
                    
                    threat = ThreatDetection(
                        threat_id=f"pattern_{attack_type}_{int(time.time())}_{hash(event.source_ip) % 10000}",
                        threat_type=attack_type,
                        threat_level=threat_level,
                        confidence=0.9,  # High confidence for pattern matches
                        description=f"{attack_type.replace('_', ' ').title()} attack detected from {event.source_ip}",
                        affected_resources=[event.target_ip or "application"],
                        attack_vector=attack_type,
                        evidence=[{"pattern_matched": pattern, "payload_excerpt": payload_lower[:100]}],
                        mitigation_suggestions=self._get_mitigation_suggestions(attack_type),
                        source_events=[event.event_id]
                    )
                    threats.append(threat)
                    break  # One detection per attack type per event
        
        return threats
    
    async def _get_nn_threat_analysis(self, features: np.ndarray) -> List[ThreatDetection]:
        """Get threat analysis from neural network"""
        threats = []
        
        if not TORCH_AVAILABLE or not self.threat_nn:
            return await self._ml_fallback_threat_analysis(features)
        
        try:
            # Pad or truncate features to expected input size
            if len(features) > 50:
                features = features[:50]
            elif len(features) < 50:
                features = np.pad(features, (0, 50 - len(features)), mode='constant')
            
            feature_tensor = torch.FloatTensor(features)
            
            with torch.no_grad():
                malware, intrusion, anomaly, data_breach, ddos, attention = self.threat_nn(feature_tensor.unsqueeze(0))
            
            # Check each threat type
            threat_scores = {
                'malware': float(malware.item()),
                'intrusion': float(intrusion.item()),
                'anomaly': float(anomaly.item()),
                'data_breach': float(data_breach.item()),
                'ddos': float(ddos.item())
            }
            
            for threat_type, score in threat_scores.items():
                if score > 0.7:  # Neural network threshold
                    threat_level = ThreatLevel.HIGH if score > 0.9 else ThreatLevel.MEDIUM
                    
                    threat = ThreatDetection(
                        threat_id=f"nn_{threat_type}_{int(time.time())}_{np.random.randint(10000)}",
                        threat_type=f"nn_{threat_type}",
                        threat_level=threat_level,
                        confidence=score,
                        description=f"Neural network detected {threat_type} activity",
                        affected_resources=["system"],
                        attack_vector="ml_detection",
                        metadata={
                            'neural_network': True,
                            'attention_weights': attention.squeeze().tolist() if attention is not None else []
                        }
                    )
                    threats.append(threat)
        
        except Exception as e:
            logger.error(f"Neural network threat analysis error: {e}")
        
        return threats
    
    async def _ml_fallback_threat_analysis(self, features: np.ndarray) -> List[ThreatDetection]:
        """ML-based fallback for threat analysis using statistical and pattern-based detection"""
        threats = []
        
        try:
            if len(features) == 0:
                return threats
            
            # Security event features analysis
            # Assuming features contain: [severity, frequency, source_entropy, pattern_match, ...]
            
            severity_score = features[0] if len(features) > 0 else 0.5
            event_frequency = features[1] if len(features) > 1 else 0.3
            source_entropy = features[2] if len(features) > 2 else 0.5
            pattern_anomaly = features[3] if len(features) > 3 else 0.3
            
            # Additional security indicators
            network_anomaly = features[4] if len(features) > 4 else 0.2
            access_pattern = features[5] if len(features) > 5 else 0.3
            payload_anomaly = features[6] if len(features) > 6 else 0.2
            
            # Malware detection heuristics
            malware_score = 0.0
            if payload_anomaly > 0.6:  # Suspicious payload patterns
                malware_score += 0.4
            if pattern_anomaly > 0.5:  # Unusual execution patterns
                malware_score += 0.3
            if source_entropy < 0.3:  # Low entropy may indicate obfuscation
                malware_score += 0.2
            
            if malware_score > 0.6:
                threat_level = ThreatLevel.HIGH if malware_score > 0.8 else ThreatLevel.MEDIUM
                threats.append(ThreatDetection(
                    threat_id=f"ml_malware_{int(time.time())}_{np.random.randint(10000)}",
                    threat_type="potential_malware",
                    threat_level=threat_level,
                    confidence=min(0.95, malware_score),
                    description="Statistical analysis indicates potential malware activity",
                    affected_resources=["system", "processes"],
                    attack_vector="malware_execution",
                    mitigation_suggestions=[
                        "Run comprehensive antivirus scan",
                        "Isolate affected systems",
                        "Review recent file changes",
                        "Check process execution patterns"
                    ]
                ))
            
            # Intrusion detection heuristics
            intrusion_score = 0.0
            if event_frequency > 0.7:  # High frequency of security events
                intrusion_score += 0.35
            if network_anomaly > 0.6:  # Unusual network patterns
                intrusion_score += 0.3
            if access_pattern > 0.6:  # Suspicious access patterns
                intrusion_score += 0.25
            
            if intrusion_score > 0.6:
                threat_level = ThreatLevel.HIGH if intrusion_score > 0.8 else ThreatLevel.MEDIUM
                threats.append(ThreatDetection(
                    threat_id=f"ml_intrusion_{int(time.time())}_{np.random.randint(10000)}",
                    threat_type="intrusion_attempt",
                    threat_level=threat_level,
                    confidence=min(0.95, intrusion_score),
                    description="Pattern analysis suggests potential intrusion attempt",
                    affected_resources=["network", "access_control"],
                    attack_vector="network_intrusion",
                    mitigation_suggestions=[
                        "Review firewall logs and rules",
                        "Check for unauthorized access attempts",
                        "Monitor network traffic patterns",
                        "Strengthen access controls"
                    ]
                ))
            
            # Anomaly detection
            anomaly_score = (pattern_anomaly + network_anomaly + payload_anomaly) / 3.0
            if anomaly_score > 0.65:
                threat_level = ThreatLevel.MEDIUM if anomaly_score > 0.8 else ThreatLevel.LOW
                threats.append(ThreatDetection(
                    threat_id=f"ml_anomaly_{int(time.time())}_{np.random.randint(10000)}",
                    threat_type="behavioral_anomaly",
                    threat_level=threat_level,
                    confidence=min(0.9, anomaly_score),
                    description="Statistical anomaly detected in system behavior",
                    affected_resources=["system_behavior"],
                    attack_vector="unknown",
                    mitigation_suggestions=[
                        "Investigate unusual system behavior",
                        "Review system logs for patterns",
                        "Monitor resource usage",
                        "Check for configuration changes"
                    ]
                ))
            
            # Data breach detection heuristics
            breach_score = 0.0
            if access_pattern > 0.7:  # Unusual data access patterns
                breach_score += 0.4
            if len(features) > 7 and features[7] > 0.6:  # Data transfer anomalies
                breach_score += 0.3
            if source_entropy > 0.7:  # High entropy may indicate data exfiltration
                breach_score += 0.2
            
            if breach_score > 0.5:
                threat_level = ThreatLevel.HIGH if breach_score > 0.7 else ThreatLevel.MEDIUM
                threats.append(ThreatDetection(
                    threat_id=f"ml_breach_{int(time.time())}_{np.random.randint(10000)}",
                    threat_type="potential_data_breach",
                    threat_level=threat_level,
                    confidence=min(0.9, breach_score),
                    description="Analysis suggests potential data breach or exfiltration",
                    affected_resources=["data_stores", "network"],
                    attack_vector="data_exfiltration",
                    mitigation_suggestions=[
                        "Audit data access logs",
                        "Check for unauthorized data transfers",
                        "Review user permissions",
                        "Monitor network traffic for data patterns"
                    ]
                ))
            
            # DDoS detection heuristics
            ddos_score = 0.0
            if event_frequency > 0.8:  # Very high event frequency
                ddos_score += 0.5
            if network_anomaly > 0.7:  # High network load
                ddos_score += 0.3
            if len(features) > 8 and features[8] > 0.6:  # Resource exhaustion indicators
                ddos_score += 0.2
            
            if ddos_score > 0.7:
                threat_level = ThreatLevel.HIGH
                threats.append(ThreatDetection(
                    threat_id=f"ml_ddos_{int(time.time())}_{np.random.randint(10000)}",
                    threat_type="ddos_attack",
                    threat_level=threat_level,
                    confidence=min(0.95, ddos_score),
                    description="High-frequency pattern suggests potential DDoS attack",
                    affected_resources=["network", "services"],
                    attack_vector="network_flood",
                    mitigation_suggestions=[
                        "Implement rate limiting",
                        "Check for traffic spikes",
                        "Review load balancer logs",
                        "Consider DDoS protection services"
                    ]
                ))
            
            # Add metadata to all threats
            for threat in threats:
                threat.metadata = {
                    'detection_method': 'statistical_analysis',
                    'feature_analysis': {
                        'severity_score': float(severity_score),
                        'event_frequency': float(event_frequency),
                        'anomaly_score': float(anomaly_score)
                    }
                }
            
            return threats
            
        except Exception as e:
            logger.warning(f"ML fallback threat analysis failed: {e}")
            return []
    
    # Additional helper methods for comprehensive analysis
    def _get_ip_count(self, item):
        """Helper method to get IP count for sorting"""
        return item[1]
    
    def _analyze_ip_patterns(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze IP address patterns"""
        ip_counts = defaultdict(int)
        ip_countries = defaultdict(set)
        
        for event in events:
            ip_counts[event.source_ip] += 1
            # In a real implementation, you'd use GeoIP lookup
            ip_countries[event.source_ip].add("unknown")
        
        return {
            'unique_ips': len(ip_counts),
            'top_ips': dict(sorted(ip_counts.items(), key=self._get_ip_count, reverse=True)[:10]),
            'suspicious_ips': [ip for ip, count in ip_counts.items() if count > 100]
        }
    
    # Removed all synthetic training data methods.
    # Models now train exclusively on real production data via ml_training_service.py
    # See _train_with_real_data() method for actual ML training implementation
    
    # Real implementations using actual data analysis
    async def _correlate_threats(self, threats: List[Dict[str, Any]], 
                               event: SecurityEvent) -> List[Dict[str, Any]]:
        """Correlate threats using real historical patterns"""
        if not threats:
            return []
        
        correlated_threats = []
        
        # Get recent similar events for correlation
        recent_events = [e for e in list(self.security_events)[-100:] 
                        if e.agent_id == event.agent_id]
        
        for threat in threats:
            threat_type = threat.get('type', 'unknown')
            
            # Find similar threat patterns in recent history
            similar_threats = [e for e in recent_events 
                             if threat_type.lower() in e.event_type.lower()]
            
            if len(similar_threats) > 2:  # Pattern detected
                # Calculate correlation strength
                time_intervals = [(similar_threats[i+1].timestamp - similar_threats[i].timestamp).seconds 
                                for i in range(len(similar_threats)-1)]
                
                avg_interval = np.mean(time_intervals) if time_intervals else 0
                correlation_strength = min(1.0, len(similar_threats) / 10.0)
                
                threat['correlation'] = {
                    'pattern_detected': True,
                    'similar_events': len(similar_threats),
                    'avg_interval_seconds': avg_interval,
                    'correlation_strength': correlation_strength,
                    'escalating': len(similar_threats) > 5
                }
            else:
                threat['correlation'] = {
                    'pattern_detected': False,
                    'similar_events': len(similar_threats),
                    'correlation_strength': 0.0
                }
            
            correlated_threats.append(threat)
        
        return correlated_threats
    
    def _update_ip_reputation(self, event: SecurityEvent, threats: List[Dict[str, Any]]):
        """Update IP reputation based on real threat analysis"""
        ip_address = event.metadata.get('source_ip')
        if not ip_address or ip_address == '127.0.0.1':
            return
        
        current_time = datetime.utcnow()
        
        # Initialize or get existing reputation
        if ip_address not in self.ip_reputation:
            self.ip_reputation[ip_address] = {
                'score': 0.5,  # Neutral starting score
                'events': 0,
                'threats': 0,
                'last_seen': current_time,
                'threat_types': Counter(),
                'first_seen': current_time
            }
        
        reputation = self.ip_reputation[ip_address]
        reputation['events'] += 1
        reputation['last_seen'] = current_time
        
        # Update based on threats detected
        if threats:
            reputation['threats'] += len(threats)
            for threat in threats:
                threat_type = threat.get('type', 'unknown')
                reputation['threat_types'][threat_type] += 1
            
            # Decrease reputation score based on threat severity
            severity_impact = {
                'critical': -0.3,
                'high': -0.2,
                'medium': -0.1,
                'low': -0.05
            }
            
            for threat in threats:
                severity = threat.get('severity', 'medium')
                reputation['score'] += severity_impact.get(severity, -0.1)
        else:
            # Slight increase for clean events
            reputation['score'] += 0.01
        
        # Calculate threat ratio
        threat_ratio = reputation['threats'] / reputation['events']
        
        # Adjust score based on overall behavior
        if threat_ratio > 0.5:  # More than 50% threats
            reputation['score'] = min(reputation['score'], 0.2)
        elif threat_ratio > 0.2:  # More than 20% threats
            reputation['score'] = min(reputation['score'], 0.4)
        
        # Bound score between 0 and 1
        reputation['score'] = max(0.0, min(1.0, reputation['score']))
        
        # Auto-block very low reputation IPs
        if reputation['score'] < 0.1 and reputation['events'] > 5:
            self.blocked_ips.add(ip_address)
            logger.warning(f"Auto-blocked IP {ip_address} due to low reputation: {reputation['score']:.3f}")
    
    def _analyze_user_agent_patterns(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze user agent patterns from real events"""
        if not events:
            return {'anomalies': [], 'suspicious_agents': [], 'bot_score': 0.0}
        
        user_agents = []
        for event in events:
            ua = event.metadata.get('user_agent', '')
            if ua:
                user_agents.append(ua)
        
        if not user_agents:
            return {'anomalies': [], 'suspicious_agents': [], 'bot_score': 0.0}
        
        # Analyze patterns
        ua_counts = Counter(user_agents)
        total_requests = len(user_agents)
        unique_agents = len(set(user_agents))
        
        suspicious_agents = []
        bot_indicators = 0
        
        # Check for suspicious patterns
        for ua, count in ua_counts.items():
            ratio = count / total_requests
            
            # High volume from single UA
            if ratio > 0.3:
                suspicious_agents.append({
                    'user_agent': ua,
                    'requests': count,
                    'percentage': ratio * 100,
                    'reason': 'high_volume'
                })
            
            # Check for bot indicators
            bot_keywords = ['bot', 'crawler', 'spider', 'scraper', 'curl', 'wget']
            if any(keyword in ua.lower() for keyword in bot_keywords):
                bot_indicators += count
            
            # Very short or suspicious UAs
            if len(ua) < 10 or ua.count('/') > 3:
                suspicious_agents.append({
                    'user_agent': ua,
                    'requests': count,
                    'percentage': ratio * 100,
                    'reason': 'suspicious_format'
                })
        
        bot_score = bot_indicators / total_requests
        diversity_score = unique_agents / total_requests
        
        return {
            'total_requests': total_requests,
            'unique_agents': unique_agents,
            'diversity_score': diversity_score,
            'bot_score': bot_score,
            'suspicious_agents': suspicious_agents,
            'anomalies': suspicious_agents  # For backward compatibility
        }
    
    def _analyze_request_patterns(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze HTTP request patterns from real events"""
        if not events:
            return {'anomalies': [], 'patterns': {}}
        
        # Extract request data from events
        request_data = []
        for event in events:
            if 'request' in event.event_type.lower():
                request_data.append({
                    'method': event.metadata.get('method', 'GET'),
                    'path': event.metadata.get('path', '/'),
                    'status_code': event.metadata.get('status_code', 200),
                    'size': event.metadata.get('response_size', 0),
                    'timestamp': event.timestamp
                })
        
        if not request_data:
            return {'anomalies': [], 'patterns': {}}
        
        # Analyze patterns
        methods = Counter([r['method'] for r in request_data])
        paths = Counter([r['path'] for r in request_data])
        status_codes = Counter([r['status_code'] for r in request_data])
        
        # Calculate request rate
        if len(request_data) > 1:
            time_span = (max(r['timestamp'] for r in request_data) - 
                        min(r['timestamp'] for r in request_data)).total_seconds()
            request_rate = len(request_data) / (time_span + 1)  # requests per second
        else:
            request_rate = 0
        
        # Identify anomalies
        anomalies = []
        
        # High error rate
        error_count = sum(count for code, count in status_codes.items() if code >= 400)
        error_rate = error_count / len(request_data)
        if error_rate > 0.2:  # >20% errors
            anomalies.append({
                'type': 'high_error_rate',
                'value': error_rate,
                'description': f'{error_rate*100:.1f}% of requests resulted in errors'
            })
        
        # Unusual method distribution
        if methods.get('POST', 0) / len(request_data) > 0.8:
            anomalies.append({
                'type': 'high_post_ratio',
                'value': methods.get('POST', 0) / len(request_data),
                'description': 'Unusually high ratio of POST requests'
            })
        
        # High request rate
        if request_rate > 10:  # >10 req/sec
            anomalies.append({
                'type': 'high_request_rate',
                'value': request_rate,
                'description': f'High request rate: {request_rate:.1f} req/s'
            })
        
        return {
            'total_requests': len(request_data),
            'request_rate': request_rate,
            'methods': dict(methods),
            'top_paths': dict(paths.most_common(10)),
            'status_codes': dict(status_codes),
            'error_rate': error_rate,
            'anomalies': anomalies,
            'patterns': {
                'method_diversity': len(methods),
                'path_diversity': len(paths),
                'avg_response_size': np.mean([r['size'] for r in request_data])
            }
        }
    
    def _analyze_temporal_patterns(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze temporal patterns in security events"""
        if len(events) < 5:
            return {'patterns': [], 'anomalies': []}
        
        # Extract timestamps
        timestamps = [event.timestamp for event in events]
        timestamps.sort()
        
        # Calculate time intervals
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                    for i in range(len(timestamps)-1)]
        
        if not intervals:
            return {'patterns': [], 'anomalies': []}
        
        # Statistical analysis
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        min_interval = min(intervals)
        max_interval = max(intervals)
        
        # Hour-of-day analysis
        hours = [ts.hour for ts in timestamps]
        hour_counts = Counter(hours)
        
        # Day-of-week analysis
        weekdays = [ts.weekday() for ts in timestamps]
        weekday_counts = Counter(weekdays)
        
        # Detect patterns
        patterns = []
        anomalies = []
        
        # Regular intervals (possible automation)
        if std_interval < mean_interval * 0.2 and len(intervals) > 5:
            patterns.append({
                'type': 'regular_intervals',
                'interval_seconds': mean_interval,
                'confidence': 0.9,
                'description': f'Events occur regularly every {mean_interval:.1f} seconds'
            })
        
        # Burst detection
        burst_count = sum(1 for interval in intervals if interval < 5)  # <5 seconds
        if burst_count > len(intervals) * 0.3:  # >30% are bursts
            anomalies.append({
                'type': 'burst_activity',
                'burst_ratio': burst_count / len(intervals),
                'description': f'{burst_count} events in rapid succession'
            })
        
        # Off-hours activity
        off_hours = [h for h in hours if h < 6 or h > 22]
        if len(off_hours) > len(hours) * 0.3:  # >30% off-hours
            anomalies.append({
                'type': 'off_hours_activity',
                'off_hours_ratio': len(off_hours) / len(hours),
                'description': f'{len(off_hours)} events during off-hours (22:00-06:00)'
            })
        
        return {
            'total_events': len(events),
            'time_span_seconds': (timestamps[-1] - timestamps[0]).total_seconds(),
            'avg_interval': mean_interval,
            'interval_std': std_interval,
            'min_interval': min_interval,
            'max_interval': max_interval,
            'hour_distribution': dict(hour_counts),
            'weekday_distribution': dict(weekday_counts),
            'patterns': patterns,
            'anomalies': anomalies
        }
    
    def _analyze_geographic_patterns(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze geographic patterns from IP addresses"""
        if not events:
            return {'countries': {}, 'anomalies': []}
        
        # Extract IP addresses
        ips = []
        for event in events:
            ip = event.metadata.get('source_ip')
            if ip and ip != '127.0.0.1':  # Skip localhost
                ips.append(ip)
        
        if not ips:
            return {'countries': {}, 'anomalies': []}
        
        # Simple geographic analysis (in production, use GeoIP database)
        # For now, analyze IP patterns
        ip_counts = Counter(ips)
        unique_ips = len(set(ips))
        
        # Classify IP ranges (simplified)
        ip_classes = {'private': 0, 'public': 0, 'suspicious': 0}
        suspicious_patterns = []
        
        for ip in set(ips):
            # Private IPs
            if (ip.startswith('192.168.') or 
                ip.startswith('10.') or 
                ip.startswith('172.')):
                ip_classes['private'] += ip_counts[ip]
            else:
                ip_classes['public'] += ip_counts[ip]
                
                # Check for suspicious patterns
                if ip_counts[ip] > len(ips) * 0.3:  # >30% from single IP
                    suspicious_patterns.append({
                        'ip': ip,
                        'requests': ip_counts[ip],
                        'percentage': ip_counts[ip] / len(ips) * 100,
                        'reason': 'high_volume_single_ip'
                    })
        
        # Detect geographic anomalies
        anomalies = []
        
        # Too many unique IPs (possible botnet)
        if unique_ips > len(ips) * 0.8:  # >80% unique
            anomalies.append({
                'type': 'high_ip_diversity',
                'unique_ratio': unique_ips / len(ips),
                'description': f'Unusually high IP diversity: {unique_ips} unique IPs from {len(ips)} requests'
            })
        
        # Single IP dominance
        max_ip_count = max(ip_counts.values())
        if max_ip_count > len(ips) * 0.5:  # >50% from single IP
            dominant_ip = max(ip_counts, key=ip_counts.get)
            anomalies.append({
                'type': 'ip_dominance',
                'dominant_ip': dominant_ip,
                'percentage': max_ip_count / len(ips) * 100,
                'description': f'Single IP {dominant_ip} accounts for {max_ip_count/len(ips)*100:.1f}% of traffic'
            })
        
        return {
            'total_requests': len(ips),
            'unique_ips': unique_ips,
            'ip_diversity': unique_ips / len(ips),
            'ip_classes': ip_classes,
            'top_ips': dict(ip_counts.most_common(10)),
            'suspicious_patterns': suspicious_patterns,
            'anomalies': anomalies
        }
    
    def _is_pattern_anomalous(self, pattern_type: str, data: Dict[str, Any]) -> bool:
        """Determine if a pattern is anomalous based on real data analysis"""
        if not data:
            return False
        
        anomaly_thresholds = {
            'request_rate': 50,      # requests per second
            'error_rate': 0.15,      # 15% error rate
            'unique_ip_ratio': 0.8,  # 80% unique IPs
            'off_hours_ratio': 0.4,  # 40% off-hours activity
            'burst_ratio': 0.3,      # 30% burst events
            'bot_score': 0.5,        # 50% bot traffic
        }
        
        if pattern_type == 'temporal':
            return (data.get('off_hours_ratio', 0) > anomaly_thresholds['off_hours_ratio'] or
                    data.get('burst_ratio', 0) > anomaly_thresholds['burst_ratio'])
        
        elif pattern_type == 'geographic':
            return data.get('ip_diversity', 0) > anomaly_thresholds['unique_ip_ratio']
        
        elif pattern_type == 'request':
            return (data.get('request_rate', 0) > anomaly_thresholds['request_rate'] or
                    data.get('error_rate', 0) > anomaly_thresholds['error_rate'])
        
        elif pattern_type == 'user_agent':
            return data.get('bot_score', 0) > anomaly_thresholds['bot_score']
        
        # Default: check if any anomalies were detected
        return len(data.get('anomalies', [])) > 0
    def _calculate_pattern_anomaly_score(self, data: Dict[str, Any]) -> float:
        """Calculate anomaly score from real pattern data"""
        score = 0.0
        factors = 0
        
        # Analyze connection patterns
        if 'connections_per_second' in data:
            cps = data['connections_per_second']
            if cps > 100:
                score += min(1.0, cps / 500)  # Scale up to 500 cps
                factors += 1
        
        # Analyze request patterns
        if 'unique_endpoints' in data:
            endpoint_ratio = data.get('endpoint_diversity', 1.0)
            if endpoint_ratio < 0.2:  # Low diversity is suspicious
                score += (1 - endpoint_ratio)
                factors += 1
        
        # Analyze timing patterns
        if 'burst_detected' in data and data['burst_detected']:
            score += 0.8
            factors += 1
        
        # Analyze geographic patterns
        if 'countries' in data:
            suspicious_countries = data.get('suspicious_countries', [])
            if suspicious_countries:
                score += min(1.0, len(suspicious_countries) * 0.2)
                factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    async def _cluster_behaviors(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Cluster security events to identify behavior patterns"""
        if len(events) < 5:
            return {'clusters': [], 'patterns': []}
        
        # Extract features for clustering
        features = []
        for event in events[:100]:  # Limit to recent 100 events
            event_features = [
                hash(event.event_type) % 100 / 100,  # Event type
                1 if event.success else 0,  # Success/failure
                event.severity_score,  # Severity
                len(event.metadata) / 10,  # Metadata richness
            ]
            features.append(event_features)
        
        features = np.array(features)
        
        # Perform clustering
        n_clusters = min(5, len(events) // 10)  # Dynamic cluster count
        if n_clusters > 1:
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                
                # Analyze clusters
                clusters = []
                for i in range(n_clusters):
                    cluster_events = [events[j] for j in range(len(labels)) if labels[j] == i]
                    if cluster_events:
                        clusters.append({
                            'size': len(cluster_events),
                            'dominant_type': Counter([e.event_type for e in cluster_events]).most_common(1)[0][0],
                            'avg_severity': np.mean([e.severity_score for e in cluster_events]),
                            'failure_rate': sum(1 for e in cluster_events if not e.success) / len(cluster_events)
                        })
                
                return {'clusters': clusters, 'n_clusters': n_clusters}
            except Exception as e:
                logger.error(f"Clustering error: {e}")
        
        return {'clusters': [], 'patterns': []}
    
    def _assess_overall_risk(self, patterns: Dict[str, Any], anomalies: Dict[str, Any]) -> str:
        """Assess overall risk level from real analysis data"""
        risk_score = 0.0
        risk_factors = 0
        
        # Pattern-based risk
        if patterns:
            # Connection patterns
            if patterns.get('connection_burst', {}).get('detected', False):
                risk_score += 0.8
                risk_factors += 1
            
            # Authentication patterns
            auth_failures = patterns.get('authentication_failures', 0)
            if auth_failures > 0:
                risk_score += min(1.0, auth_failures / 10)
                risk_factors += 1
            
            # Geographic risk
            geo_risk = patterns.get('geographic_risk', 0)
            if geo_risk > 0:
                risk_score += geo_risk
                risk_factors += 1
        
        # Anomaly-based risk
        if anomalies:
            anomaly_score = anomalies.get('score', 0)
            if anomaly_score > 0:
                risk_score += anomaly_score
                risk_factors += 1
        
        # Calculate average risk
        avg_risk = risk_score / risk_factors if risk_factors > 0 else 0.0
        
        # Determine risk level
        if avg_risk >= 0.8:
            return "critical"
        elif avg_risk >= 0.6:
            return "high"
        elif avg_risk >= 0.4:
            return "medium"
        elif avg_risk >= 0.2:
            return "low"
        else:
            return "minimal"
    
    def _prepare_historical_data(self) -> List[Dict[str, Any]]:
        """Prepare historical threat data for analysis"""
        # Return actual historical data from detections
        historical_data = []
        
        # Get recent detections
        for detections in list(self.threat_history.values())[-1000:]:
            for detection in detections:
                historical_data.append({
                    'timestamp': detection.timestamp,
                    'threat_type': detection.threat_type,
                    'severity': detection.severity,
                    'confidence': detection.confidence_score,
                    'mitigated': detection.mitigated
                })
        
        return historical_data
    
    def _extract_trend_features(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Extract trend features from real historical data"""
        if not data:
            return np.zeros(20)
        
        features = []
        
        # Temporal features
        timestamps = [d['timestamp'] for d in data]
        if timestamps:
            time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600  # Hours
            features.append(time_span)
            features.append(len(data) / (time_span + 1))  # Events per hour
        else:
            features.extend([0, 0])
        
        # Threat type distribution
        threat_types = [d.get('threat_type', 'unknown') for d in data]
        type_counts = Counter(threat_types)
        for threat in ['intrusion', 'malware', 'ddos', 'brute_force']:
            features.append(type_counts.get(threat, 0) / len(data) if data else 0)
        
        # Severity distribution
        severities = [d.get('severity', 'medium') for d in data]
        sev_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
        avg_severity = np.mean([sev_map.get(s, 0.5) for s in severities])
        features.append(avg_severity)
        
        # Success rate
        mitigated = sum(1 for d in data if d.get('mitigated', False))
        features.append(mitigated / len(data) if data else 1.0)
        
        # Trend indicators
        if len(data) > 10:
            recent = data[-5:]
            older = data[-10:-5]
            recent_severity = np.mean([sev_map.get(d.get('severity', 'medium'), 0.5) for d in recent])
            older_severity = np.mean([sev_map.get(d.get('severity', 'medium'), 0.5) for d in older])
            trend = recent_severity - older_severity
            features.append(trend)
        else:
            features.append(0)
        
        # Pad to expected size
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)
    
    def _heuristic_threat_prediction(self, threat_type: str, data: Dict[str, Any]) -> float:
        """Predict threat probability using heuristics on real data"""
        base_probability = 0.1
        
        # Adjust based on threat type and current data
        if threat_type == 'intrusion':
            if data.get('failed_auth_count', 0) > 5:
                base_probability += 0.3
            if data.get('unusual_ports', False):
                base_probability += 0.2
            if data.get('suspicious_patterns', 0) > 0:
                base_probability += 0.2
        
        elif threat_type == 'ddos':
            if data.get('request_rate', 0) > 1000:
                base_probability += 0.4
            if data.get('unique_sources', 0) > 100:
                base_probability += 0.3
        
        elif threat_type == 'malware':
            if data.get('suspicious_files', 0) > 0:
                base_probability += 0.4
            if data.get('unusual_processes', False):
                base_probability += 0.3
        
        return min(1.0, base_probability)
    
    def _calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence based on feature quality and model state"""
        # Check feature quality
        non_zero_features = np.count_nonzero(features)
        feature_coverage = non_zero_features / len(features)
        
        # Base confidence on feature coverage
        confidence = 0.3 + (feature_coverage * 0.4)
        
        # Adjust based on model training state
        if hasattr(self, 'models_trained') and self.models_trained:
            confidence += 0.2
        
        # Adjust based on data recency
        if hasattr(self, 'last_training_time'):
            hours_since_training = (datetime.utcnow() - self.last_training_time).total_seconds() / 3600
            if hours_since_training < 1:
                confidence += 0.1
            elif hours_since_training > 24:
                confidence -= 0.1
        
        return max(0.1, min(0.95, confidence))
    
    def _identify_risk_factors(self, threat_type: str, data: Dict[str, Any]) -> List[str]:
        """Identify specific risk factors from threat analysis"""
        risk_factors = []
        
        if threat_type == 'intrusion':
            if data.get('failed_auth_count', 0) > 3:
                risk_factors.append(f"Multiple failed authentication attempts: {data['failed_auth_count']}")
            if data.get('privilege_escalation', False):
                risk_factors.append("Privilege escalation attempt detected")
            if data.get('suspicious_commands', []):
                risk_factors.append(f"Suspicious commands executed: {', '.join(data['suspicious_commands'][:3])}")
        
        elif threat_type == 'ddos':
            if data.get('request_rate', 0) > 500:
                risk_factors.append(f"High request rate: {data['request_rate']} req/s")
            if data.get('bandwidth_usage', 0) > 0.8:
                risk_factors.append(f"High bandwidth usage: {data['bandwidth_usage']*100:.1f}%")
            if data.get('syn_flood', False):
                risk_factors.append("SYN flood attack pattern detected")
        
        elif threat_type == 'malware':
            if data.get('suspicious_files', []):
                risk_factors.append(f"Suspicious files detected: {len(data['suspicious_files'])}")
            if data.get('c2_communication', False):
                risk_factors.append("Command & Control communication detected")
            if data.get('file_encryption', False):
                risk_factors.append("File encryption activity detected (possible ransomware)")
        
        elif threat_type == 'data_exfiltration':
            if data.get('unusual_data_transfer', False):
                risk_factors.append(f"Unusual data transfer: {data.get('data_volume', 'unknown')}")
            if data.get('sensitive_data_access', False):
                risk_factors.append("Access to sensitive data detected")
        
        # Common risk factors
        if data.get('known_bad_ip', False):
            risk_factors.append("Connection from known malicious IP")
        if data.get('tor_exit_node', False):
            risk_factors.append("Traffic from Tor exit node")
        if data.get('vpn_detected', False) and data.get('suspicious_behavior', False):
            risk_factors.append("VPN usage with suspicious behavior")
        
        return risk_factors
    
    def _generate_preventive_actions(self, predictions: Dict[str, float]) -> List[str]:
        """Generate preventive actions based on threat predictions"""
        actions = []
        
        # Sort predictions by probability
        def get_threat_probability(x):
            return x[1]
        sorted_threats = sorted(predictions.items(), key=get_threat_probability, reverse=True)
        
        for threat_type, probability in sorted_threats:
            if probability > 0.7:  # High probability threats
                if threat_type == 'intrusion':
                    actions.extend([
                        "Enable enhanced authentication monitoring",
                        "Activate intrusion detection system (IDS)",
                        "Review and tighten access control lists"
                    ])
                elif threat_type == 'ddos':
                    actions.extend([
                        "Pre-configure DDoS mitigation rules",
                        "Increase bandwidth capacity temporarily",
                        "Enable rate limiting on all public endpoints"
                    ])
                elif threat_type == 'malware':
                    actions.extend([
                        "Update antivirus signatures immediately",
                        "Enable real-time file scanning",
                        "Block executable downloads temporarily"
                    ])
            
            elif probability > 0.4:  # Medium probability threats
                if threat_type == 'intrusion':
                    actions.append("Increase login attempt monitoring")
                elif threat_type == 'ddos':
                    actions.append("Monitor traffic patterns closely")
                elif threat_type == 'malware':
                    actions.append("Schedule full system scan")
        
        # General preventive actions
        if not actions:
            actions = [
                "Maintain current security monitoring levels",
                "Review security logs regularly",
                "Ensure all systems are updated"
            ]
        
        return list(set(actions))[:6]  # Return unique actions, limit to 6
    
    def _get_mitigation_suggestions(self, attack_type: str) -> List[str]:
        """Get specific mitigation suggestions for detected attacks"""
        mitigation_map = {
            'sql_injection': [
                "Enable SQL query parameterization",
                "Implement input validation and sanitization",
                "Deploy Web Application Firewall (WAF)",
                "Review and fix vulnerable SQL queries"
            ],
            'xss': [
                "Enable Content Security Policy (CSP)",
                "Implement output encoding for all user data",
                "Sanitize all user inputs before display",
                "Use HTTP-only cookies for sessions"
            ],
            'brute_force': [
                "Implement account lockout after failed attempts",
                "Deploy CAPTCHA on login forms",
                "Enable multi-factor authentication",
                "Use progressive delays for failed attempts"
            ],
            'ddos': [
                "Enable DDoS protection service",
                "Configure rate limiting per IP",
                "Implement SYN cookies",
                "Use CDN for traffic distribution"
            ],
            'malware': [
                "Isolate infected systems immediately",
                "Run comprehensive malware removal",
                "Restore from clean backups if needed",
                "Update all security software"
            ],
            'intrusion': [
                "Reset all potentially compromised accounts",
                "Review all system logs for unauthorized changes",
                "Implement network segmentation",
                "Deploy host-based intrusion detection"
            ],
            'data_exfiltration': [
                "Block suspicious outbound connections",
                "Implement data loss prevention (DLP)",
                "Monitor large data transfers",
                "Encrypt sensitive data at rest"
            ]
        }
        
        # Get specific mitigations or return general ones
        specific_mitigations = mitigation_map.get(attack_type, [])
        
        if not specific_mitigations:
            return [
                "Monitor system for further suspicious activity",
                "Block source IP addresses involved in attack",
                "Review and update security policies",
                "Increase logging and monitoring levels"
            ]
        
        return specific_mitigations


# Singleton instance
_security_monitor = None

def get_security_monitor() -> AISecurityMonitor:
    """Get or create security monitor instance"""
    global _security_monitor
    if not _security_monitor:
        _security_monitor = AISecurityMonitor()
    return _security_monitor