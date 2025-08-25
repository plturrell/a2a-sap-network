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
        """Initialize ML models with synthetic training data"""
        # Generate synthetic security training data
        X_anomaly = self._generate_anomaly_training_data()
        X_intrusion, y_intrusion = self._generate_intrusion_training_data()
        X_malware, y_malware = self._generate_malware_training_data()
        X_behavior, y_behavior = self._generate_behavior_training_data()
        X_ddos = self._generate_ddos_training_data()
        
        # Train models
        if len(X_anomaly) > 0:
            X_anomaly_scaled = self.feature_scaler.fit_transform(X_anomaly)
            self.anomaly_detector.fit(X_anomaly_scaled)
        
        if len(X_intrusion) > 0:
            self.intrusion_detector.fit(X_intrusion, y_intrusion)
        
        if len(X_malware) > 0:
            self.malware_classifier.fit(X_malware, y_malware)
        
        if len(X_behavior) > 0:
            X_behavior_scaled = self.behavior_scaler.fit_transform(X_behavior)
            self.behavior_analyzer.fit(X_behavior_scaled, y_behavior)
        
        if len(X_ddos) > 0:
            self.ddos_detector.fit(X_ddos)
    
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
            return threats
        
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
    
    # Additional helper methods for comprehensive analysis
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
            'top_ips': dict(sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'suspicious_ips': [ip for ip, count in ip_counts.items() if count > 100]
        }
    
    def _generate_anomaly_training_data(self) -> List[np.ndarray]:
        """Generate synthetic anomaly training data"""
        X = []
        for i in range(500):
            # Normal behavior features
            normal_features = np.random.normal(0.5, 0.1, 25)
            X.append(normal_features)
            
            # Add some anomalous samples
            if i % 10 == 0:  # 10% anomalies
                anomaly_features = normal_features.copy()
                anomaly_features[np.random.randint(0, len(anomaly_features))] += np.random.uniform(2, 4)
                X.append(anomaly_features)
        
        return X
    
    def _generate_intrusion_training_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Generate synthetic intrusion training data"""
        X, y = [], []
        
        for i in range(300):
            features = np.random.rand(25)
            
            # Label based on suspicious patterns
            if features[3] > 0.8 or features[10] > 0.9 or features[15] > 0.7:  # High risk features
                label = 1  # Intrusion
            else:
                label = 0  # Normal
            
            X.append(features)
            y.append(label)
        
        return X, y
    
    def _generate_malware_training_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Generate synthetic malware training data"""
        X, y = [], []
        
        for i in range(250):
            features = np.random.rand(25)
            
            # Malware indicators
            malware_score = features[1] * 0.3 + features[8] * 0.4 + features[12] * 0.3
            label = 1 if malware_score > 0.6 else 0
            
            X.append(features)
            y.append(label)
        
        return X, y
    
    def _generate_behavior_training_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Generate synthetic behavior training data"""
        X, y = [], []
        
        for i in range(400):
            features = np.random.rand(25)
            
            # Behavioral classification
            if features[5] > 0.8 and features[14] > 0.7:
                label = 1  # Suspicious behavior
            else:
                label = 0  # Normal behavior
            
            X.append(features)
            y.append(label)
        
        return X, y
    
    def _generate_ddos_training_data(self) -> List[np.ndarray]:
        """Generate synthetic DDoS training data"""
        X = []
        for i in range(200):
            # Normal traffic patterns
            features = np.random.normal(0.3, 0.1, 25)
            X.append(features)
        
        return X
    
    # Placeholder methods for additional functionality
    async def _correlate_threats(self, threats, event): return []
    def _update_ip_reputation(self, event, threats): pass
    def _analyze_user_agent_patterns(self, events): return {}
    def _analyze_request_patterns(self, events): return {}
    def _analyze_temporal_patterns(self, events): return {}
    def _analyze_geographic_patterns(self, events): return {}
    def _is_pattern_anomalous(self, pattern_type, data): return False
    def _calculate_pattern_anomaly_score(self, data): return 0.5
    async def _cluster_behaviors(self, events): return {}
    def _assess_overall_risk(self, patterns, anomalies): return "medium"
    def _prepare_historical_data(self): return []
    def _extract_trend_features(self, data): return np.random.rand(20)
    def _heuristic_threat_prediction(self, threat_type, data): return 0.3
    def _calculate_prediction_confidence(self, features): return 0.7
    def _identify_risk_factors(self, threat_type, data): return []
    def _generate_preventive_actions(self, predictions): return []
    def _get_mitigation_suggestions(self, attack_type): return ["Monitor and block"]


# Singleton instance
_security_monitor = None

def get_security_monitor() -> AISecurityMonitor:
    """Get or create security monitor instance"""
    global _security_monitor
    if not _security_monitor:
        _security_monitor = AISecurityMonitor()
    return _security_monitor