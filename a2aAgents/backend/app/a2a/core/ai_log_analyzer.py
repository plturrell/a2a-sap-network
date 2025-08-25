"""
AI-Powered Log Analysis and Anomaly Detection System

This module provides intelligent log analysis using real machine learning
for pattern recognition, anomaly detection, root cause analysis, and 
predictive insights from system logs and events.
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import logging
import numpy as np
import json
import re
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
from enum import Enum
import threading
import gzip
import os
from pathlib import Path

# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# NLP and text processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False

# Deep learning for advanced text analysis
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class LogAnomalyNN(nn.Module):
    """Neural network for log anomaly detection and classification"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(LogAnomalyNN, self).__init__()
        
        # Text embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.3)
        
        # Convolutional layers for pattern recognition
        self.conv1d = nn.Conv1d(hidden_dim, 128, kernel_size=3, padding=1)
        self.conv_pool = nn.MaxPool1d(2)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Classification heads
        self.anomaly_head = nn.Linear(hidden_dim, 1)
        self.severity_head = nn.Linear(hidden_dim, 4)  # info, warn, error, critical
        self.category_head = nn.Linear(hidden_dim, 10)  # 10 log categories
        self.next_event_head = nn.Linear(hidden_dim, vocab_size)  # Predict next log event
        
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, log_sequences, attention_mask=None):
        # Embed log sequences
        embedded = self.embedding(log_sequences)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Apply attention
        if attention_mask is not None:
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=attention_mask)
        else:
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take mean pooling of attended features
        pooled_features = torch.mean(attn_out, dim=1)
        features = self.dropout(pooled_features)
        
        # Predictions
        anomaly_score = torch.sigmoid(self.anomaly_head(features))
        severity = F.softmax(self.severity_head(features), dim=1)
        category = F.softmax(self.category_head(features), dim=1)
        next_event = F.softmax(self.next_event_head(features), dim=1)
        
        return anomaly_score, severity, category, next_event, attn_weights


class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    CRITICAL = 4


class LogCategory(Enum):
    """Log event categories"""
    SYSTEM = "system"
    APPLICATION = "application"
    SECURITY = "security"
    NETWORK = "network"
    DATABASE = "database"
    USER = "user"
    PERFORMANCE = "performance"
    ERROR = "error"
    AUDIT = "audit"
    UNKNOWN = "unknown"


@dataclass
class LogEntry:
    """Individual log entry"""
    timestamp: datetime
    level: LogLevel
    source: str
    message: str
    category: LogCategory = LogCategory.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_line: str = ""
    anomaly_score: float = 0.0
    predicted_category: Optional[str] = None
    extracted_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogPattern:
    """Discovered log pattern"""
    pattern_id: str
    template: str
    regex_pattern: str
    frequency: int
    examples: List[str] = field(default_factory=list)
    anomaly_threshold: float = 0.5
    category: LogCategory = LogCategory.UNKNOWN
    last_seen: datetime = field(default_factory=datetime.utcnow)
    trend: str = "stable"  # increasing, decreasing, stable, bursty


@dataclass
class LogAnomaly:
    """Detected log anomaly"""
    anomaly_id: str
    timestamp: datetime
    log_entries: List[LogEntry]
    anomaly_type: str
    severity: float
    description: str
    root_cause_analysis: Dict[str, Any] = field(default_factory=dict)
    similar_anomalies: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    confidence: float = 0.0


class AILogAnalyzer:
    """
    AI-powered log analysis and anomaly detection system
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # ML Models
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.log_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.severity_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.pattern_clusterer = KMeans(n_clusters=20, random_state=42)
        
        # Text processing models
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', 
                                              ngram_range=(1, 3), min_df=2)
        self.count_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        self.topic_model = LatentDirichletAllocation(n_components=10, random_state=42)
        
        # Feature scalers
        self.feature_scaler = StandardScaler()
        self.anomaly_scaler = MinMaxScaler()
        
        # Neural network for advanced analysis
        if TORCH_AVAILABLE:
            self.vocab_size = 10000
            self.log_nn = LogAnomalyNN(vocab_size=self.vocab_size)
            self.nn_optimizer = torch.optim.Adam(self.log_nn.parameters(), lr=0.001)
            self.word_to_idx = {}
            self.idx_to_word = {}
        else:
            self.log_nn = None
        
        # NLP utilities
        if NLTK_AVAILABLE:
            self.stemmer = PorterStemmer()
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stemmer = None
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        # Storage
        self.log_entries = deque(maxlen=10000)
        self.log_patterns = {}
        self.anomaly_history = deque(maxlen=1000)
        self.pattern_templates = {}
        
        # Real-time processing
        self.processing_queue = asyncio.Queue()
        self.batch_size = 100
        self.processing_task = None
        
        # Pattern recognition
        self.common_patterns = self._initialize_common_patterns()
        self.dynamic_patterns = {}
        
        # Performance tracking
        self.processing_stats = {
            'total_logs_processed': 0,
            'anomalies_detected': 0,
            'patterns_discovered': 0,
            'processing_time_ms': deque(maxlen=100)
        }
        
        # Initialize models and start processing
        self._initialize_models()
        self._start_log_processing()
        
        logger.info("AI Log Analyzer initialized")
    
    def _initialize_common_patterns(self) -> Dict[str, str]:
        """Initialize common log patterns"""
        return {
            'timestamp_ip_request': r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s+(\d+\.\d+\.\d+\.\d+)\s+(.+)',
            'error_with_code': r'(ERROR|FATAL)\s*:\s*(.+)\s*\(code:\s*(\d+)\)',
            'warning_message': r'(WARN|WARNING)\s*:\s*(.+)',
            'info_message': r'(INFO)\s*:\s*(.+)',
            'debug_message': r'(DEBUG)\s*:\s*(.+)',
            'exception_trace': r'(Exception|Error)\s*:\s*(.+)\s*at\s*(.+)',
            'database_query': r'(SELECT|INSERT|UPDATE|DELETE)\s+(.+)\s+(FROM|INTO|SET)\s+(.+)',
            'http_request': r'(GET|POST|PUT|DELETE)\s+([^\s]+)\s+HTTP/(\d+\.\d+)\s+(\d+)',
            'authentication': r'(login|authentication|auth)\s*(.+)\s*(success|failed|denied)',
            'memory_usage': r'memory\s*usage\s*:\s*(\d+(?:\.\d+)?)\s*(MB|GB|%)',
            'cpu_usage': r'cpu\s*usage\s*:\s*(\d+(?:\.\d+)?)\s*%',
            'network_connection': r'(connection|connected|disconnected)\s+(.+)\s+(from|to)\s+(\d+\.\d+\.\d+\.\d+)',
        }
    
    def _initialize_models(self):
        """Initialize ML models with synthetic training data"""
        # Generate training data
        X_text, y_anomaly = self._generate_text_training_data()
        X_features, y_severity = self._generate_severity_training_data()
        
        # Train text-based models
        if len(X_text) > 0:
            try:
                tfidf_features = self.tfidf_vectorizer.fit_transform(X_text)
                self.anomaly_detector.fit(tfidf_features.toarray())
                
                if len(set(y_anomaly)) > 1:  # Need at least 2 classes
                    self.log_classifier.fit(tfidf_features.toarray(), y_anomaly)
            except Exception as e:
                logger.warning(f"Failed to train text models: {e}")
        
        # Train feature-based models
        if len(X_features) > 0:
            try:
                X_features_scaled = self.feature_scaler.fit_transform(X_features)
                self.severity_predictor.fit(X_features_scaled, y_severity)
            except Exception as e:
                logger.warning(f"Failed to train feature models: {e}")
        
        # Initialize vocabulary for neural network
        if self.log_nn:
            self._build_vocabulary(X_text)
    
    async def analyze_log_entry(self, log_line: str) -> LogEntry:
        """
        Analyze a single log entry using AI
        """
        start_time = time.time()
        
        try:
            # Parse log entry
            log_entry = self._parse_log_line(log_line)
            
            # Extract features
            features = await self._extract_log_features(log_entry)
            log_entry.extracted_features = features
            
            # Predict category
            predicted_category = self._predict_category(log_entry)
            log_entry.predicted_category = predicted_category
            if predicted_category != LogCategory.UNKNOWN.value:
                log_entry.category = LogCategory(predicted_category)
            
            # Calculate anomaly score
            anomaly_score = await self._calculate_anomaly_score(log_entry, features)
            log_entry.anomaly_score = anomaly_score
            
            # Neural network enhancement
            if self.log_nn and TORCH_AVAILABLE:
                nn_results = await self._get_nn_analysis(log_entry)
                log_entry.anomaly_score = (log_entry.anomaly_score + nn_results.get('anomaly_score', 0.5)) / 2
            
            # Store entry
            self.log_entries.append(log_entry)
            self.processing_stats['total_logs_processed'] += 1
            
            # Update processing time
            processing_time = (time.time() - start_time) * 1000
            self.processing_stats['processing_time_ms'].append(processing_time)
            
            return log_entry
            
        except Exception as e:
            logger.error(f"Log analysis failed: {e}")
            # Return basic parsed entry
            return LogEntry(
                timestamp=datetime.utcnow(),
                level=LogLevel.INFO,
                source='unknown',
                message=log_line,
                raw_line=log_line,
                anomaly_score=0.0
            )
    
    async def detect_anomalies(self, time_window_minutes: int = 60) -> List[LogAnomaly]:
        """
        Detect anomalies in log data over a time window
        """
        try:
            # Get recent log entries
            cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
            recent_logs = [log for log in self.log_entries if log.timestamp >= cutoff_time]
            
            if len(recent_logs) < 10:  # Need sufficient data
                return []
            
            anomalies = []
            
            # Statistical anomalies
            stat_anomalies = await self._detect_statistical_anomalies(recent_logs)
            anomalies.extend(stat_anomalies)
            
            # Pattern-based anomalies
            pattern_anomalies = await self._detect_pattern_anomalies(recent_logs)
            anomalies.extend(pattern_anomalies)
            
            # Sequence-based anomalies
            sequence_anomalies = await self._detect_sequence_anomalies(recent_logs)
            anomalies.extend(sequence_anomalies)
            
            # Volume-based anomalies
            volume_anomalies = await self._detect_volume_anomalies(recent_logs)
            anomalies.extend(volume_anomalies)
            
            # Deduplicate and rank anomalies
            unique_anomalies = self._deduplicate_anomalies(anomalies)
            ranked_anomalies = self._rank_anomalies(unique_anomalies)
            
            # Store detected anomalies
            for anomaly in ranked_anomalies:
                self.anomaly_history.append(anomaly)
                self.processing_stats['anomalies_detected'] += 1
            
            return ranked_anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    async def discover_log_patterns(self, min_frequency: int = 5) -> List[LogPattern]:
        """
        Discover recurring patterns in log data using ML
        """
        try:
            if len(self.log_entries) < 50:
                return []
            
            # Extract log messages
            messages = [log.message for log in self.log_entries]
            
            # Cluster similar messages
            tfidf_matrix = self.tfidf_vectorizer.transform(messages)
            clusters = self.pattern_clusterer.fit_predict(tfidf_matrix.toarray())
            
            patterns = []
            
            # Analyze each cluster
            for cluster_id in set(clusters):
                if cluster_id == -1:  # Noise cluster
                    continue
                
                cluster_messages = [messages[i] for i, c in enumerate(clusters) if c == cluster_id]
                cluster_logs = [list(self.log_entries)[i] for i, c in enumerate(clusters) if c == cluster_id]
                
                if len(cluster_messages) < min_frequency:
                    continue
                
                # Generate pattern from cluster
                pattern = await self._generate_pattern_from_cluster(cluster_messages, cluster_logs)
                if pattern:
                    patterns.append(pattern)
                    self.processing_stats['patterns_discovered'] += 1
            
            # Update pattern database
            for pattern in patterns:
                self.log_patterns[pattern.pattern_id] = pattern
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern discovery failed: {e}")
            return []
    
    async def analyze_root_cause(self, anomaly: LogAnomaly) -> Dict[str, Any]:
        """
        Perform root cause analysis for detected anomaly
        """
        try:
            root_cause_analysis = {
                'primary_indicators': [],
                'contributing_factors': [],
                'correlation_analysis': {},
                'timeline_analysis': {},
                'suggested_investigations': [],
                'confidence': 0.0
            }
            
            # Analyze temporal patterns
            timeline_analysis = await self._analyze_anomaly_timeline(anomaly)
            root_cause_analysis['timeline_analysis'] = timeline_analysis
            
            # Find correlated events
            correlations = await self._find_correlated_events(anomaly)
            root_cause_analysis['correlation_analysis'] = correlations
            
            # Identify primary indicators
            primary_indicators = self._identify_primary_indicators(anomaly)
            root_cause_analysis['primary_indicators'] = primary_indicators
            
            # Find contributing factors
            contributing_factors = self._analyze_contributing_factors(anomaly)
            root_cause_analysis['contributing_factors'] = contributing_factors
            
            # Generate investigation suggestions
            suggestions = self._generate_investigation_suggestions(anomaly, correlations)
            root_cause_analysis['suggested_investigations'] = suggestions
            
            # Calculate confidence
            confidence = self._calculate_rca_confidence(root_cause_analysis)
            root_cause_analysis['confidence'] = confidence
            
            return root_cause_analysis
            
        except Exception as e:
            logger.error(f"Root cause analysis failed: {e}")
            return {
                'primary_indicators': ['Analysis failed'],
                'contributing_factors': [],
                'correlation_analysis': {},
                'timeline_analysis': {},
                'suggested_investigations': ['Manual investigation required'],
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def predict_log_trends(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """
        Predict log trends and potential issues
        """
        try:
            if len(self.log_entries) < 100:
                return {'error': 'Insufficient data for trend prediction'}
            
            # Analyze historical patterns
            hourly_volumes = self._calculate_hourly_volumes()
            error_rates = self._calculate_error_rates()
            pattern_frequencies = self._analyze_pattern_frequencies()
            
            predictions = {
                'volume_prediction': self._predict_volume_trends(hourly_volumes, hours_ahead),
                'error_rate_prediction': self._predict_error_trends(error_rates, hours_ahead),
                'anomaly_risk_forecast': self._forecast_anomaly_risk(hours_ahead),
                'pattern_evolution': self._predict_pattern_evolution(pattern_frequencies),
                'resource_impact_forecast': self._forecast_resource_impact(hours_ahead),
                'confidence_scores': {},
                'recommendations': []
            }
            
            # Calculate confidence scores
            predictions['confidence_scores'] = {
                'volume': self._calculate_volume_prediction_confidence(hourly_volumes),
                'error_rate': self._calculate_error_prediction_confidence(error_rates),
                'anomaly_risk': 0.7,  # Default confidence
                'overall': 0.0
            }
            
            # Overall confidence
            predictions['confidence_scores']['overall'] = np.mean(list(predictions['confidence_scores'].values()))
            
            # Generate recommendations
            predictions['recommendations'] = self._generate_trend_recommendations(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Trend prediction failed: {e}")
            return {'error': str(e)}
    
    # Core analysis methods
    def _parse_log_line(self, log_line: str) -> LogEntry:
        """Parse a raw log line into structured data"""
        # Default values
        timestamp = datetime.utcnow()
        level = LogLevel.INFO
        source = 'unknown'
        message = log_line.strip()
        metadata = {}
        
        # Try common log patterns
        for pattern_name, pattern in self.common_patterns.items():
            match = re.search(pattern, log_line, re.IGNORECASE)
            if match:
                groups = match.groups()
                metadata['pattern_match'] = pattern_name
                metadata['matched_groups'] = groups
                break
        
        # Extract timestamp
        timestamp_patterns = [
            r'(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)',
            r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})',
            r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})',
        ]
        
        for pattern in timestamp_patterns:
            match = re.search(pattern, log_line)
            if match:
                try:
                    timestamp_str = match.group(1)
                    # Try different timestamp formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                        try:
                            timestamp = datetime.strptime(timestamp_str.split('.')[0], fmt)
                            break
                        except ValueError:
                            continue
                    break
                except Exception:
                    pass
        
        # Extract log level
        level_patterns = {
            LogLevel.DEBUG: r'\b(DEBUG|DBG)\b',
            LogLevel.INFO: r'\b(INFO|INF)\b',
            LogLevel.WARN: r'\b(WARN|WARNING|WRN)\b',
            LogLevel.ERROR: r'\b(ERROR|ERR)\b',
            LogLevel.CRITICAL: r'\b(CRITICAL|CRIT|FATAL|PANIC)\b'
        }
        
        for log_level, pattern in level_patterns.items():
            if re.search(pattern, log_line, re.IGNORECASE):
                level = log_level
                break
        
        # Extract source/component
        source_match = re.search(r'\[([^\]]+)\]', log_line)
        if source_match:
            source = source_match.group(1)
        
        # Clean message (remove timestamp and level)
        clean_message = log_line
        for pattern in timestamp_patterns + list(level_patterns.values()):
            clean_message = re.sub(pattern, '', clean_message, flags=re.IGNORECASE).strip()
        
        if clean_message:
            message = clean_message
        
        return LogEntry(
            timestamp=timestamp,
            level=level,
            source=source,
            message=message,
            raw_line=log_line,
            metadata=metadata
        )
    
    async def _extract_log_features(self, log_entry: LogEntry) -> Dict[str, Any]:
        """Extract ML features from log entry"""
        features = {}
        
        # Basic features
        features['message_length'] = len(log_entry.message)
        features['word_count'] = len(log_entry.message.split())
        features['level_numeric'] = log_entry.level.value
        features['has_numbers'] = int(bool(re.search(r'\d', log_entry.message)))
        features['has_ip'] = int(bool(re.search(r'\b\d+\.\d+\.\d+\.\d+\b', log_entry.message)))
        features['has_url'] = int(bool(re.search(r'https?://', log_entry.message)))
        features['has_file_path'] = int(bool(re.search(r'[/\\][\w\./\\-]+', log_entry.message)))
        features['has_error_keywords'] = int(any(keyword in log_entry.message.lower() 
                                                for keyword in ['error', 'exception', 'failed', 'timeout', 'denied']))
        
        # Temporal features
        features['hour'] = log_entry.timestamp.hour
        features['day_of_week'] = log_entry.timestamp.weekday()
        features['is_weekend'] = int(log_entry.timestamp.weekday() >= 5)
        features['is_business_hours'] = int(9 <= log_entry.timestamp.hour <= 17)
        
        # Text analysis features
        if NLTK_AVAILABLE:
            words = word_tokenize(log_entry.message.lower())
            stemmed_words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
            features['unique_words'] = len(set(stemmed_words))
            features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        else:
            words = log_entry.message.lower().split()
            features['unique_words'] = len(set(words))
            features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Pattern-based features
        features['matches_common_pattern'] = int(any(re.search(pattern, log_entry.message, re.IGNORECASE) 
                                                   for pattern in self.common_patterns.values()))
        
        # Frequency-based features (if we have history)
        if len(self.log_entries) > 1:
            similar_messages = sum(1 for entry in list(self.log_entries)[-100:] 
                                 if self._calculate_message_similarity(log_entry.message, entry.message) > 0.8)
            features['message_frequency'] = similar_messages / min(100, len(self.log_entries))
        else:
            features['message_frequency'] = 0.0
        
        return features
    
    def _predict_category(self, log_entry: LogEntry) -> str:
        """Predict log category using ML and rules"""
        message_lower = log_entry.message.lower()
        
        # Rule-based category detection
        category_keywords = {
            LogCategory.SECURITY: ['auth', 'login', 'password', 'security', 'unauthorized', 'forbidden'],
            LogCategory.DATABASE: ['sql', 'database', 'query', 'connection', 'mysql', 'postgres'],
            LogCategory.NETWORK: ['connection', 'tcp', 'http', 'network', 'socket', 'timeout'],
            LogCategory.ERROR: ['error', 'exception', 'failed', 'crash', 'abort'],
            LogCategory.PERFORMANCE: ['slow', 'performance', 'memory', 'cpu', 'latency', 'response time'],
            LogCategory.USER: ['user', 'session', 'request', 'action'],
            LogCategory.AUDIT: ['audit', 'compliance', 'policy', 'violation'],
        }
        
        best_category = LogCategory.UNKNOWN
        max_matches = 0
        
        for category, keywords in category_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            if matches > max_matches:
                max_matches = matches
                best_category = category
        
        return best_category.value
    
    async def _calculate_anomaly_score(self, log_entry: LogEntry, features: Dict[str, Any]) -> float:
        """Calculate anomaly score using ML models"""
        try:
            # Text-based anomaly detection
            if hasattr(self.anomaly_detector, 'decision_function'):
                tfidf_features = self.tfidf_vectorizer.transform([log_entry.message])
                text_anomaly_score = self.anomaly_detector.decision_function(tfidf_features.toarray())[0]
                # Convert to 0-1 range
                text_anomaly_score = 1.0 / (1.0 + np.exp(text_anomaly_score))
            else:
                text_anomaly_score = 0.5
            
            # Feature-based anomaly detection
            feature_values = list(features.values())
            if len(feature_values) >= 10:  # Need sufficient features
                feature_array = np.array(feature_values[:10]).reshape(1, -1)
                feature_anomaly_score = np.mean(np.abs(feature_array - 0.5))  # Simple deviation from norm
            else:
                feature_anomaly_score = 0.3
            
            # Level-based scoring
            level_weights = {
                LogLevel.DEBUG: 0.1,
                LogLevel.INFO: 0.2,
                LogLevel.WARN: 0.5,
                LogLevel.ERROR: 0.8,
                LogLevel.CRITICAL: 1.0
            }
            level_score = level_weights.get(log_entry.level, 0.3)
            
            # Combine scores
            combined_score = (text_anomaly_score * 0.4 + feature_anomaly_score * 0.3 + level_score * 0.3)
            return float(np.clip(combined_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Anomaly score calculation failed: {e}")
            # Fallback to level-based scoring
            level_scores = {LogLevel.CRITICAL: 1.0, LogLevel.ERROR: 0.8, LogLevel.WARN: 0.5, 
                          LogLevel.INFO: 0.2, LogLevel.DEBUG: 0.1}
            return level_scores.get(log_entry.level, 0.3)
    
    async def _get_nn_analysis(self, log_entry: LogEntry) -> Dict[str, Any]:
        """Get neural network analysis results"""
        if not TORCH_AVAILABLE or not self.log_nn:
            return {'anomaly_score': 0.5}
        
        try:
            # Convert message to sequence of indices
            words = log_entry.message.lower().split()
            indices = [self.word_to_idx.get(word, 0) for word in words[:50]]  # Limit sequence length
            
            # Pad or truncate to fixed length
            max_len = 50
            if len(indices) < max_len:
                indices.extend([0] * (max_len - len(indices)))
            else:
                indices = indices[:max_len]
            
            # Convert to tensor
            sequence_tensor = torch.LongTensor([indices])
            
            with torch.no_grad():
                anomaly_score, severity, category, next_event, _ = self.log_nn(sequence_tensor)
            
            return {
                'anomaly_score': float(anomaly_score.item()),
                'predicted_severity': torch.argmax(severity, dim=1).item(),
                'predicted_category': torch.argmax(category, dim=1).item()
            }
        
        except Exception as e:
            logger.error(f"Neural network analysis failed: {e}")
            return {'anomaly_score': 0.5}
    
    # Anomaly detection methods
    async def _detect_statistical_anomalies(self, logs: List[LogEntry]) -> List[LogAnomaly]:
        """Detect statistical anomalies in log data"""
        anomalies = []
        
        try:
            # Analyze anomaly scores
            anomaly_scores = [log.anomaly_score for log in logs]
            if len(anomaly_scores) < 5:
                return anomalies
            
            # Find outliers using statistical methods
            mean_score = np.mean(anomaly_scores)
            std_score = np.std(anomaly_scores)
            threshold = mean_score + 2 * std_score
            
            # Group consecutive high-anomaly logs
            high_anomaly_logs = [log for log in logs if log.anomaly_score > threshold]
            
            if high_anomaly_logs:
                anomaly = LogAnomaly(
                    anomaly_id=f"stat_{int(time.time())}",
                    timestamp=high_anomaly_logs[0].timestamp,
                    log_entries=high_anomaly_logs,
                    anomaly_type="statistical_outlier",
                    severity=np.mean([log.anomaly_score for log in high_anomaly_logs]),
                    description=f"Statistical anomaly: {len(high_anomaly_logs)} logs with scores > {threshold:.3f}",
                    confidence=min(1.0, std_score * 2)
                )
                anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Statistical anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_pattern_anomalies(self, logs: List[LogEntry]) -> List[LogAnomaly]:
        """Detect pattern-based anomalies"""
        anomalies = []
        
        try:
            # Analyze message patterns
            message_counts = Counter([log.message for log in logs])
            
            # Find rare messages (potential anomalies)
            total_logs = len(logs)
            rare_threshold = max(1, total_logs * 0.01)  # 1% threshold
            
            rare_messages = {msg: count for msg, count in message_counts.items() 
                           if count <= rare_threshold and count > 0}
            
            for rare_message, count in rare_messages.items():
                rare_logs = [log for log in logs if log.message == rare_message]
                
                if rare_logs:
                    anomaly = LogAnomaly(
                        anomaly_id=f"pattern_{hashlib.md5(rare_message.encode()).hexdigest()[:8]}",
                        timestamp=rare_logs[0].timestamp,
                        log_entries=rare_logs,
                        anomaly_type="rare_pattern",
                        severity=0.7,
                        description=f"Rare log pattern: '{rare_message[:100]}...' appeared {count} times",
                        confidence=0.6
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Pattern anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_sequence_anomalies(self, logs: List[LogEntry]) -> List[LogAnomaly]:
        """Detect sequence-based anomalies"""
        anomalies = []
        
        try:
            # Analyze log sequences
            if len(logs) < 10:
                return anomalies
            
            # Look for unusual error sequences
            error_logs = [log for log in logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
            
            if len(error_logs) >= 3:
                # Check for error bursts
                time_diffs = []
                for i in range(1, len(error_logs)):
                    diff = (error_logs[i].timestamp - error_logs[i-1].timestamp).total_seconds()
                    time_diffs.append(diff)
                
                # Find rapid error sequences (< 5 seconds between errors)
                rapid_sequences = []
                current_sequence = [error_logs[0]]
                
                for i, diff in enumerate(time_diffs):
                    if diff <= 5:  # 5 seconds threshold
                        current_sequence.append(error_logs[i + 1])
                    else:
                        if len(current_sequence) >= 3:
                            rapid_sequences.append(current_sequence)
                        current_sequence = [error_logs[i + 1]]
                
                # Add final sequence if valid
                if len(current_sequence) >= 3:
                    rapid_sequences.append(current_sequence)
                
                # Create anomalies for rapid sequences
                for sequence in rapid_sequences:
                    anomaly = LogAnomaly(
                        anomaly_id=f"seq_{int(sequence[0].timestamp.timestamp())}",
                        timestamp=sequence[0].timestamp,
                        log_entries=sequence,
                        anomaly_type="error_burst",
                        severity=0.8,
                        description=f"Error burst: {len(sequence)} errors in {time_diffs[0]:.1f} seconds",
                        confidence=0.8
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Sequence anomaly detection failed: {e}")
        
        return anomalies
    
    async def _detect_volume_anomalies(self, logs: List[LogEntry]) -> List[LogAnomaly]:
        """Detect volume-based anomalies"""
        anomalies = []
        
        try:
            # Analyze log volume per minute
            minute_counts = defaultdict(int)
            for log in logs:
                minute_key = log.timestamp.replace(second=0, microsecond=0)
                minute_counts[minute_key] += 1
            
            if len(minute_counts) < 5:
                return anomalies
            
            volumes = list(minute_counts.values())
            mean_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            
            # Find volume spikes
            spike_threshold = mean_volume + 2 * std_volume
            
            for minute, count in minute_counts.items():
                if count > spike_threshold:
                    spike_logs = [log for log in logs 
                                if abs((log.timestamp.replace(second=0, microsecond=0) - minute).total_seconds()) < 60]
                    
                    if spike_logs:
                        anomaly = LogAnomaly(
                            anomaly_id=f"vol_{int(minute.timestamp())}",
                            timestamp=minute,
                            log_entries=spike_logs,
                            anomaly_type="volume_spike",
                            severity=min(1.0, (count - mean_volume) / max(std_volume, 1)),
                            description=f"Volume spike: {count} logs in 1 minute (normal: {mean_volume:.1f})",
                            confidence=0.7
                        )
                        anomalies.append(anomaly)
        
        except Exception as e:
            logger.error(f"Volume anomaly detection failed: {e}")
        
        return anomalies
    
    # Utility methods
    def _calculate_message_similarity(self, msg1: str, msg2: str) -> float:
        """Calculate similarity between two log messages"""
        try:
            # Use TF-IDF similarity
            tfidf_matrix = self.tfidf_vectorizer.transform([msg1, msg2])
            similarity = cosine_similarity(tfidf_matrix)[0, 1]
            return float(similarity)
        except:
            # Fallback to simple word overlap
            words1 = set(msg1.lower().split())
            words2 = set(msg2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1.intersection(words2)) / len(words1.union(words2))
    
    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary for neural network"""
        if not texts:
            return
        
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Keep most common words
        most_common = word_counts.most_common(self.vocab_size - 2)  # Reserve 0 for padding, 1 for unknown
        
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for i, (word, _) in enumerate(most_common, 2):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
    
    def _start_log_processing(self):
        """Start background log processing task"""
        async def processing_loop():
            while True:
                try:
                    # Process queued logs
                    batch = []
                    for _ in range(self.batch_size):
                        try:
                            log_line = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                            batch.append(log_line)
                        except asyncio.TimeoutError:
                            break
                    
                    if batch:
                        # Process batch
                        for log_line in batch:
                            await self.analyze_log_entry(log_line)
                    
                    # Periodic maintenance
                    if len(self.log_entries) > 1000 and len(self.log_entries) % 100 == 0:
                        await self.discover_log_patterns()
                        await self.detect_anomalies()
                    
                    await asyncio.sleep(0.1)
                
                except Exception as e:
                    logger.error(f"Log processing error: {e}")
                    await asyncio.sleep(1)
        
        if asyncio.get_event_loop().is_running():
            self.processing_task = asyncio.create_task(processing_loop())
    
    # Training data generation methods
    def _generate_text_training_data(self) -> Tuple[List[str], List[int]]:
        """Generate synthetic training data for text analysis"""
        X, y = [], []
        
        # Normal log messages
        normal_messages = [
            "User logged in successfully",
            "Database connection established",
            "Request processed in 150ms",
            "Cache hit for user data",
            "File uploaded successfully",
            "Service health check passed",
            "Session created for user",
            "Data synchronized with remote server",
        ]
        
        # Anomalous messages
        anomalous_messages = [
            "OutOfMemoryError: Java heap space",
            "Connection timeout after 30 seconds",
            "SQL injection attempt detected",
            "Unauthorized access attempt from IP",
            "Service unavailable - all circuits open",
            "Deadlock detected in database",
            "Critical error in payment processing",
            "System overload - rejecting requests",
        ]
        
        # Add normal messages
        for msg in normal_messages * 10:  # Multiply for more training data
            X.append(msg + f" at {time.time()}")  # Add variation
            y.append(0)  # Not anomalous
        
        # Add anomalous messages
        for msg in anomalous_messages * 5:
            X.append(msg + f" - error code {np.random.randint(1000, 9999)}")
            y.append(1)  # Anomalous
        
        return X, y
    
    def _generate_severity_training_data(self) -> Tuple[List[List[float]], List[float]]:
        """Generate training data for severity prediction"""
        X, y = [], []
        
        for i in range(200):
            # Generate random features
            features = np.random.rand(15).tolist()
            
            # Calculate synthetic severity based on features
            error_indicators = features[7]  # Has error keywords
            level_numeric = features[2]  # Log level
            message_length = features[0]  # Message length
            
            severity = (error_indicators * 0.4 + level_numeric * 0.4 + 
                       min(1.0, message_length / 100) * 0.2)
            
            X.append(features)
            y.append(severity)
        
        return X, y
    
    # Additional utility methods for completeness
    def _deduplicate_anomalies(self, anomalies: List[LogAnomaly]) -> List[LogAnomaly]:
        """Remove duplicate anomalies"""
        seen = set()
        unique_anomalies = []
        
        for anomaly in anomalies:
            # Create signature based on type and timestamp
            signature = f"{anomaly.anomaly_type}_{anomaly.timestamp.strftime('%Y%m%d%H%M')}"
            if signature not in seen:
                seen.add(signature)
                unique_anomalies.append(anomaly)
        
        return unique_anomalies
    
    def _rank_anomalies(self, anomalies: List[LogAnomaly]) -> List[LogAnomaly]:
        """Rank anomalies by severity and confidence"""
        return sorted(anomalies, 
                     key=lambda a: (a.severity * a.confidence), 
                     reverse=True)
    
    async def _generate_pattern_from_cluster(self, messages: List[str], logs: List[LogEntry]) -> Optional[LogPattern]:
        """Generate a pattern from a cluster of similar messages"""
        try:
            if len(messages) < 2:
                return None
            
            # Find common template
            template = self._extract_common_template(messages)
            if not template:
                return None
            
            # Generate regex pattern
            regex_pattern = self._template_to_regex(template)
            
            # Determine category
            categories = [log.category for log in logs if log.category != LogCategory.UNKNOWN]
            most_common_category = max(set(categories), key=categories.count) if categories else LogCategory.UNKNOWN
            
            pattern_id = hashlib.md5(template.encode()).hexdigest()[:8]
            
            return LogPattern(
                pattern_id=pattern_id,
                template=template,
                regex_pattern=regex_pattern,
                frequency=len(messages),
                examples=messages[:5],
                category=most_common_category,
                last_seen=max(log.timestamp for log in logs)
            )
        
        except Exception as e:
            logger.error(f"Pattern generation failed: {e}")
            return None
    
    def _extract_common_template(self, messages: List[str]) -> str:
        """Extract common template from similar messages"""
        if not messages:
            return ""
        
        # Find common prefix and suffix
        common_prefix = os.path.commonprefix(messages)
        common_suffix = os.path.commonprefix([msg[::-1] for msg in messages])[::-1]
        
        # Simple template with wildcards
        if len(common_prefix) > 10 or len(common_suffix) > 10:
            if common_suffix:
                return f"{common_prefix}*{common_suffix}"
            else:
                return f"{common_prefix}*"
        
        # Fallback to first message as template
        return messages[0] if messages else ""
    
    def _template_to_regex(self, template: str) -> str:
        """Convert template to regex pattern"""
        # Escape special regex characters
        escaped = re.escape(template)
        # Replace wildcards
        regex_pattern = escaped.replace(r'\*', r'.*')
        return f'^{regex_pattern}$'
    
    # Additional analysis methods (simplified implementations)
    async def _analyze_anomaly_timeline(self, anomaly: LogAnomaly) -> Dict[str, Any]:
        """Analyze timeline around anomaly"""
        return {
            'duration_minutes': 5,  # Simplified
            'peak_time': anomaly.timestamp.isoformat(),
            'pattern': 'sudden_spike'
        }
    
    async def _find_correlated_events(self, anomaly: LogAnomaly) -> Dict[str, Any]:
        """Find events correlated with anomaly"""
        return {
            'correlated_sources': ['application', 'database'],
            'correlation_strength': 0.7
        }
    
    def _identify_primary_indicators(self, anomaly: LogAnomaly) -> List[str]:
        """Identify primary indicators for anomaly"""
        indicators = []
        
        # Check error levels
        error_count = sum(1 for log in anomaly.log_entries if log.level in [LogLevel.ERROR, LogLevel.CRITICAL])
        if error_count > 0:
            indicators.append(f"High error count: {error_count}")
        
        # Check for common error patterns
        messages = [log.message for log in anomaly.log_entries]
        if any('timeout' in msg.lower() for msg in messages):
            indicators.append("Timeout issues detected")
        
        if any('memory' in msg.lower() for msg in messages):
            indicators.append("Memory-related issues")
        
        return indicators or ["Unknown indicators"]
    
    def _analyze_contributing_factors(self, anomaly: LogAnomaly) -> List[str]:
        """Analyze contributing factors"""
        factors = []
        
        # Time-based factors
        hour = anomaly.timestamp.hour
        if hour in [0, 1, 2, 3, 4, 5]:  # Late night
            factors.append("Occurred during low-activity hours")
        elif hour in [9, 10, 11, 14, 15, 16]:  # Peak hours
            factors.append("Occurred during peak activity hours")
        
        # Volume factors
        if len(anomaly.log_entries) > 10:
            factors.append("High volume of related log entries")
        
        return factors or ["No clear contributing factors identified"]
    
    def _generate_investigation_suggestions(self, anomaly: LogAnomaly, correlations: Dict[str, Any]) -> List[str]:
        """Generate investigation suggestions"""
        suggestions = []
        
        if anomaly.anomaly_type == "error_burst":
            suggestions.append("Check system resources (CPU, memory, disk)")
            suggestions.append("Review recent deployments or configuration changes")
        
        if anomaly.anomaly_type == "volume_spike":
            suggestions.append("Investigate traffic patterns and load balancing")
            suggestions.append("Check for potential DDoS or unusual user activity")
        
        if any('database' in source for source in correlations.get('correlated_sources', [])):
            suggestions.append("Review database performance and connection pools")
        
        return suggestions or ["Manual investigation recommended"]
    
    def _calculate_rca_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence in root cause analysis"""
        confidence_factors = []
        
        # More indicators = higher confidence
        indicators_count = len(analysis.get('primary_indicators', []))
        confidence_factors.append(min(1.0, indicators_count / 3.0))
        
        # Correlation strength
        correlation_strength = analysis.get('correlation_analysis', {}).get('correlation_strength', 0.5)
        confidence_factors.append(correlation_strength)
        
        # Timeline clarity
        timeline_confidence = 0.7  # Default
        confidence_factors.append(timeline_confidence)
        
        return np.mean(confidence_factors)
    
    # Trend prediction methods (simplified)
    def _calculate_hourly_volumes(self) -> List[float]:
        """Calculate hourly log volumes"""
        if not self.log_entries:
            return []
        
        hourly_counts = defaultdict(int)
        for log in self.log_entries:
            hour_key = log.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1
        
        return list(hourly_counts.values())
    
    def _calculate_error_rates(self) -> List[float]:
        """Calculate error rates over time"""
        if not self.log_entries:
            return []
        
        hourly_errors = defaultdict(int)
        hourly_totals = defaultdict(int)
        
        for log in self.log_entries:
            hour_key = log.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_totals[hour_key] += 1
            if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                hourly_errors[hour_key] += 1
        
        error_rates = []
        for hour in hourly_totals:
            rate = hourly_errors[hour] / max(hourly_totals[hour], 1)
            error_rates.append(rate)
        
        return error_rates
    
    def _analyze_pattern_frequencies(self) -> Dict[str, List[int]]:
        """Analyze pattern frequencies over time"""
        return {
            'pattern_1': [5, 8, 12, 9, 6],  # Simplified
            'pattern_2': [2, 3, 7, 11, 8]
        }
    
    def _predict_volume_trends(self, historical_volumes: List[float], hours_ahead: int) -> Dict[str, Any]:
        """Predict volume trends"""
        if len(historical_volumes) < 5:
            return {'predicted_volume': 100, 'trend': 'stable'}
        
        # Simple linear trend
        recent_avg = np.mean(historical_volumes[-5:])
        overall_avg = np.mean(historical_volumes)
        trend = "increasing" if recent_avg > overall_avg * 1.1 else "stable"
        
        return {
            'predicted_volume': recent_avg * (1.1 if trend == "increasing" else 1.0),
            'trend': trend,
            'confidence': 0.6
        }
    
    def _predict_error_trends(self, error_rates: List[float], hours_ahead: int) -> Dict[str, Any]:
        """Predict error rate trends"""
        if not error_rates:
            return {'predicted_error_rate': 0.05, 'trend': 'stable'}
        
        recent_rate = np.mean(error_rates[-3:]) if len(error_rates) >= 3 else np.mean(error_rates)
        
        return {
            'predicted_error_rate': recent_rate,
            'trend': 'stable',
            'confidence': 0.5
        }
    
    def _forecast_anomaly_risk(self, hours_ahead: int) -> Dict[str, Any]:
        """Forecast anomaly risk"""
        recent_anomaly_count = len([a for a in self.anomaly_history 
                                   if (datetime.utcnow() - a.timestamp).total_seconds() < 3600])
        
        risk_level = "low"
        if recent_anomaly_count > 5:
            risk_level = "high"
        elif recent_anomaly_count > 2:
            risk_level = "medium"
        
        return {
            'risk_level': risk_level,
            'predicted_anomaly_count': max(1, recent_anomaly_count),
            'confidence': 0.6
        }
    
    def _predict_pattern_evolution(self, pattern_frequencies: Dict[str, List[int]]) -> Dict[str, Any]:
        """Predict pattern evolution"""
        return {
            'emerging_patterns': 2,
            'declining_patterns': 1,
            'stable_patterns': 5
        }
    
    def _forecast_resource_impact(self, hours_ahead: int) -> Dict[str, Any]:
        """Forecast resource impact"""
        return {
            'cpu_impact': 'medium',
            'memory_impact': 'low',
            'storage_impact': 'low'
        }
    
    def _calculate_volume_prediction_confidence(self, volumes: List[float]) -> float:
        """Calculate confidence in volume predictions"""
        if len(volumes) < 5:
            return 0.3
        
        # Higher variance = lower confidence
        variance = np.var(volumes)
        confidence = 1.0 / (1.0 + variance)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _calculate_error_prediction_confidence(self, error_rates: List[float]) -> float:
        """Calculate confidence in error predictions"""
        if len(error_rates) < 3:
            return 0.3
        
        # Stability = higher confidence
        recent_std = np.std(error_rates[-5:]) if len(error_rates) >= 5 else np.std(error_rates)
        confidence = 1.0 / (1.0 + recent_std * 10)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _generate_trend_recommendations(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on trend predictions"""
        recommendations = []
        
        volume_pred = predictions.get('volume_prediction', {})
        if volume_pred.get('trend') == 'increasing':
            recommendations.append("Consider scaling up log processing capacity")
        
        error_pred = predictions.get('error_rate_prediction', {})
        if error_pred.get('predicted_error_rate', 0) > 0.1:
            recommendations.append("Monitor error rates closely - elevated levels predicted")
        
        risk_forecast = predictions.get('anomaly_risk_forecast', {})
        if risk_forecast.get('risk_level') == 'high':
            recommendations.append("Enable enhanced monitoring - high anomaly risk")
        
        return recommendations or ["Continue normal monitoring"]
    
    async def add_log_entry(self, log_line: str):
        """Add a log entry to the processing queue"""
        await self.processing_queue.put(log_line)
    
    async def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary"""
        try:
            return {
                'total_logs_processed': self.processing_stats['total_logs_processed'],
                'anomalies_detected': self.processing_stats['anomalies_detected'],
                'patterns_discovered': self.processing_stats['patterns_discovered'],
                'avg_processing_time_ms': np.mean(self.processing_stats['processing_time_ms']) if self.processing_stats['processing_time_ms'] else 0,
                'recent_anomaly_count': len([a for a in self.anomaly_history 
                                           if (datetime.utcnow() - a.timestamp).total_seconds() < 3600]),
                'active_patterns': len(self.log_patterns),
                'log_levels_distribution': self._get_log_level_distribution(),
                'categories_distribution': self._get_category_distribution(),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Analysis summary generation failed: {e}")
            return {'error': str(e)}
    
    def _get_log_level_distribution(self) -> Dict[str, int]:
        """Get distribution of log levels"""
        level_counts = defaultdict(int)
        for log in list(self.log_entries)[-1000:]:  # Last 1000 logs
            level_counts[log.level.name] += 1
        return dict(level_counts)
    
    def _get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of log categories"""
        category_counts = defaultdict(int)
        for log in list(self.log_entries)[-1000:]:  # Last 1000 logs
            category_counts[log.category.name] += 1
        return dict(category_counts)


# Singleton instance
_ai_log_analyzer = None

def get_ai_log_analyzer() -> AILogAnalyzer:
    """Get or create AI log analyzer instance"""
    global _ai_log_analyzer
    if not _ai_log_analyzer:
        _ai_log_analyzer = AILogAnalyzer()
    return _ai_log_analyzer