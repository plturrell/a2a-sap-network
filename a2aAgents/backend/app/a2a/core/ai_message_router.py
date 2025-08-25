"""
AI-Powered Message Routing and Communication Optimization System

This module provides intelligent agent message routing, communication optimization,
protocol selection, and message priority management using real machine learning
for enhanced inter-agent communication without relying on external services.
"""

import asyncio
import logging
import numpy as np
import json
import time
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
from enum import Enum
import heapq

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, ExtraTreesClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Graph analysis for message flow optimization
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Deep learning for advanced message understanding
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class MessageRoutingNN(nn.Module):
    """Neural network for intelligent message routing and optimization"""
    def __init__(self, vocab_size=8000, embedding_dim=256, hidden_dim=512):
        super(MessageRoutingNN, self).__init__()
        
        # Message content embedding
        self.message_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Agent context encoder
        self.agent_encoder = nn.LSTM(embedding_dim, hidden_dim // 2, 
                                   batch_first=True, bidirectional=True)
        
        # Message flow analyzer
        self.flow_analyzer = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),  # +64 for metadata features
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2)
        )
        
        # Attention mechanism for route importance
        self.attention = nn.MultiheadAttention(hidden_dim // 2, num_heads=8)
        
        # Multi-task routing heads
        self.route_selection_head = nn.Linear(hidden_dim // 2, 16)  # 16 possible routes
        self.priority_head = nn.Linear(hidden_dim // 2, 4)         # 4 priority levels
        self.protocol_head = nn.Linear(hidden_dim // 2, 6)         # 6 protocols
        self.latency_prediction_head = nn.Linear(hidden_dim // 2, 1)
        self.reliability_head = nn.Linear(hidden_dim // 2, 1)
        self.cost_prediction_head = nn.Linear(hidden_dim // 2, 1)
        self.congestion_head = nn.Linear(hidden_dim // 2, 3)       # Low, medium, high
        
        # Message optimization heads
        self.compression_head = nn.Linear(hidden_dim // 2, 1)      # Compression ratio
        self.batch_optimization_head = nn.Linear(hidden_dim // 2, 1)  # Batch size
        self.retry_strategy_head = nn.Linear(hidden_dim // 2, 5)   # 5 retry strategies
        
        self.dropout = nn.Dropout(0.25)
        self.layer_norm = nn.LayerNorm(hidden_dim // 2)
    
    def forward(self, message_tokens, metadata_features):
        batch_size, seq_len = message_tokens.size()
        
        # Embed message content
        embedded = self.message_embedding(message_tokens)
        
        # Encode agent context
        encoded, (hidden, cell) = self.agent_encoder(embedded)
        
        # Get message representation
        message_repr = encoded.mean(1)  # Average pooling
        
        # Combine with metadata
        combined_features = torch.cat([message_repr, metadata_features], dim=1)
        
        # Analyze message flow
        flow_features = self.flow_analyzer(combined_features)
        flow_features = self.layer_norm(flow_features)
        
        # Apply attention
        attn_out, attn_weights = self.attention(
            flow_features.unsqueeze(1), 
            flow_features.unsqueeze(1), 
            flow_features.unsqueeze(1)
        )
        final_features = attn_out.squeeze(1)
        final_features = self.dropout(final_features)
        
        # Multi-task predictions
        route_probs = F.softmax(self.route_selection_head(final_features), dim=-1)
        priority_probs = F.softmax(self.priority_head(final_features), dim=-1)
        protocol_probs = F.softmax(self.protocol_head(final_features), dim=-1)
        latency_pred = F.relu(self.latency_prediction_head(final_features))
        reliability_pred = torch.sigmoid(self.reliability_head(final_features))
        cost_pred = F.relu(self.cost_prediction_head(final_features))
        congestion_probs = F.softmax(self.congestion_head(final_features), dim=-1)
        
        compression_ratio = torch.sigmoid(self.compression_head(final_features))
        batch_size_pred = F.relu(self.batch_optimization_head(final_features))
        retry_strategy_probs = F.softmax(self.retry_strategy_head(final_features), dim=-1)
        
        return {
            'route_selection': route_probs,
            'priority': priority_probs,
            'protocol': protocol_probs,
            'latency_prediction': latency_pred,
            'reliability': reliability_pred,
            'cost_prediction': cost_pred,
            'congestion_level': congestion_probs,
            'compression_ratio': compression_ratio,
            'batch_size': batch_size_pred,
            'retry_strategy': retry_strategy_probs,
            'attention_weights': attn_weights,
            'message_features': final_features
        }


@dataclass
class Message:
    """Represents an inter-agent message"""
    id: str
    sender_id: str
    recipient_id: str
    content: str
    message_type: str
    priority: str = "medium"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    routing_history: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Route:
    """Represents a communication route between agents"""
    route_id: str
    source: str
    destination: str
    protocol: str
    latency_estimate: float
    reliability_score: float
    cost_estimate: float
    capacity: int
    current_load: int = 0
    is_active: bool = True
    performance_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """Represents a routing decision made by AI"""
    message_id: str
    selected_route: str
    confidence_score: float
    predicted_latency: float
    predicted_reliability: float
    routing_strategy: str
    optimization_applied: List[str] = field(default_factory=list)


class MessagePriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AIMessageRouter:
    """
    AI-powered message routing and communication optimization system using real ML models
    """
    
    def __init__(self):
        # ML Models for routing optimization
        self.route_classifier = RandomForestClassifier(n_estimators=150, random_state=42)
        self.latency_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.priority_classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
        self.congestion_predictor = MLPRegressor(hidden_layer_sizes=(128, 64), random_state=42)
        self.reliability_predictor = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        
        # Clustering for route optimization
        self.route_clusterer = KMeans(n_clusters=6, random_state=42)
        self.message_pattern_detector = DBSCAN(eps=0.4, min_samples=3)
        
        # Feature extractors and processors
        self.message_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.metadata_vectorizer = CountVectorizer(max_features=2000)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Neural network for advanced routing
        if TORCH_AVAILABLE:
            self.routing_nn = MessageRoutingNN()
            self.nn_optimizer = torch.optim.AdamW(self.routing_nn.parameters(), lr=0.001)
        else:
            self.routing_nn = None
        
        # Network graph for route analysis
        if NETWORKX_AVAILABLE:
            self.communication_graph = nx.DiGraph()
        else:
            self.communication_graph = None
        
        # Route management
        self.routes = {}
        self.active_routes = set()
        self.route_performance = defaultdict(list)
        
        # Message queues and processing
        self.message_queues = {
            MessagePriority.CRITICAL: [],
            MessagePriority.HIGH: [],
            MessagePriority.MEDIUM: [],
            MessagePriority.LOW: []
        }
        
        # Communication patterns and analytics
        self.message_patterns = defaultdict(list)
        self.routing_decisions = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        
        # Optimization strategies
        self.routing_strategies = {
            'fastest': self._route_by_latency,
            'reliable': self._route_by_reliability,
            'cost_effective': self._route_by_cost,
            'balanced': self._route_balanced,
            'adaptive': self._route_adaptive
        }
        
        # Real-time monitoring
        self._monitoring_task = None
        self._is_monitoring = False
        
        # Statistics and feedback
        self.routing_stats = defaultdict(int)
        self.feedback_history = deque(maxlen=500)
        
        # Initialize models with training data
        self._initialize_models()
        
        logger.info("AI Message Router initialized with ML models")
    
    async def route_message(self, message: Message, 
                          strategy: str = "adaptive",
                          options: Dict[str, Any] = None) -> RoutingDecision:
        """
        Route a message using AI-powered decision making
        """
        try:
            options = options or {}
            
            # Extract message features
            features = self._extract_message_features(message)
            
            # Predict optimal routing strategy
            if strategy == "adaptive":
                strategy = self._predict_optimal_strategy(message, features)
            
            # Get available routes
            available_routes = self._get_available_routes(message.sender_id, message.recipient_id)
            
            if not available_routes:
                logger.warning(f"No routes available from {message.sender_id} to {message.recipient_id}")
                return self._create_fallback_decision(message)
            
            # Apply routing strategy
            routing_func = self.routing_strategies.get(strategy, self._route_balanced)
            selected_route = routing_func(message, available_routes, features)
            
            # Predict performance metrics
            predicted_latency = self._predict_latency(message, selected_route, features)
            predicted_reliability = self._predict_reliability(message, selected_route, features)
            
            # Create routing decision
            decision = RoutingDecision(
                message_id=message.id,
                selected_route=selected_route.route_id,
                confidence_score=self._calculate_confidence(message, selected_route, features),
                predicted_latency=predicted_latency,
                predicted_reliability=predicted_reliability,
                routing_strategy=strategy
            )
            
            # Apply optimizations
            optimizations = await self._apply_message_optimizations(message, selected_route, options)
            decision.optimization_applied = optimizations
            
            # Update route load
            selected_route.current_load += 1
            
            # Record decision
            self.routing_decisions.append(decision)
            
            # Update statistics
            self.routing_stats['messages_routed'] += 1
            self.routing_stats[f'strategy_{strategy}'] += 1
            
            return decision
            
        except Exception as e:
            logger.error(f"Message routing error: {e}")
            return self._create_fallback_decision(message)
    
    async def optimize_routes(self, timeframe_minutes: int = 60) -> Dict[str, Any]:
        """
        Optimize routing configuration using ML analysis
        """
        try:
            optimization_results = {
                'routes_analyzed': 0,
                'routes_optimized': 0,
                'performance_improvements': {},
                'new_routes_suggested': [],
                'routes_to_disable': [],
                'load_balancing_changes': {}
            }
            
            # Analyze recent routing performance
            cutoff_time = datetime.utcnow() - timedelta(minutes=timeframe_minutes)
            recent_decisions = [
                d for d in self.routing_decisions 
                if hasattr(d, 'timestamp') and d.timestamp > cutoff_time
            ]
            
            if not recent_decisions:
                logger.info("No recent routing decisions to analyze")
                return optimization_results
            
            # Extract performance features
            decision_features = []
            performance_scores = []
            
            for decision in recent_decisions:
                if decision.selected_route in self.routes:
                    route = self.routes[decision.selected_route]
                    features = self._extract_route_features(route)
                    decision_features.append(features)
                    
                    # Calculate performance score
                    actual_performance = self._get_actual_performance(decision)
                    performance_scores.append(actual_performance)
            
            if not decision_features:
                return optimization_results
            
            decision_features = np.array(decision_features)
            performance_scores = np.array(performance_scores)
            
            optimization_results['routes_analyzed'] = len(decision_features)
            
            # Identify underperforming routes
            underperforming_routes = self._identify_underperforming_routes(
                decision_features, performance_scores
            )
            
            # Suggest route optimizations
            for route_id in underperforming_routes:
                if route_id in self.routes:
                    optimizations = self._suggest_route_optimizations(route_id)
                    optimization_results['performance_improvements'][route_id] = optimizations
            
            # Identify load balancing opportunities
            load_balancing = self._analyze_load_balancing_opportunities(decision_features)
            optimization_results['load_balancing_changes'] = load_balancing
            
            # Suggest new routes
            new_routes = self._suggest_new_routes(decision_features, performance_scores)
            optimization_results['new_routes_suggested'] = new_routes
            
            # Update route configurations
            optimization_results['routes_optimized'] = await self._apply_route_optimizations(
                optimization_results
            )
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Route optimization error: {e}")
            return {'error': str(e)}
    
    async def analyze_communication_patterns(self) -> Dict[str, Any]:
        """
        Analyze communication patterns using ML to identify optimization opportunities
        """
        try:
            analysis_results = {
                'total_messages_analyzed': 0,
                'communication_clusters': [],
                'traffic_patterns': {},
                'bottlenecks_identified': [],
                'optimization_opportunities': [],
                'agent_interaction_matrix': {},
                'temporal_patterns': {}
            }
            
            if not self.message_patterns:
                return analysis_results
            
            # Prepare data for analysis
            all_messages = []
            for agent_id, messages in self.message_patterns.items():
                all_messages.extend(messages)
            
            analysis_results['total_messages_analyzed'] = len(all_messages)
            
            if len(all_messages) < 10:
                logger.info("Insufficient message data for pattern analysis")
                return analysis_results
            
            # Extract message features for clustering
            message_features = []
            for message in all_messages:
                features = self._extract_message_features(message)
                message_features.append(features)
            
            message_features = np.array(message_features)
            
            # Identify communication clusters
            if len(message_features) > 3:
                clusters = self.message_pattern_detector.fit_predict(message_features)
                unique_clusters = set(clusters[clusters != -1])
                
                for cluster_id in unique_clusters:
                    cluster_messages = [
                        all_messages[i] for i, c in enumerate(clusters) if c == cluster_id
                    ]
                    
                    cluster_info = {
                        'cluster_id': int(cluster_id),
                        'message_count': len(cluster_messages),
                        'agents_involved': list(set([m.sender_id for m in cluster_messages] + 
                                                 [m.recipient_id for m in cluster_messages])),
                        'avg_message_size': np.mean([len(m.content) for m in cluster_messages]),
                        'common_patterns': self._identify_cluster_patterns(cluster_messages)
                    }
                    analysis_results['communication_clusters'].append(cluster_info)
            
            # Analyze traffic patterns
            traffic_by_hour = defaultdict(int)
            traffic_by_agent = defaultdict(int)
            
            for message in all_messages:
                hour = message.timestamp.hour
                traffic_by_hour[hour] += 1
                traffic_by_agent[message.sender_id] += 1
            
            analysis_results['traffic_patterns'] = {
                'hourly_distribution': dict(traffic_by_hour),
                'agent_activity': dict(traffic_by_agent),
                'peak_hours': sorted(traffic_by_hour.items(), key=self._get_hour_traffic_count, reverse=True)[:3]
            }
            
            # Identify bottlenecks
            bottlenecks = self._identify_communication_bottlenecks(all_messages)
            analysis_results['bottlenecks_identified'] = bottlenecks
            
            # Generate optimization opportunities
            opportunities = self._identify_optimization_opportunities(analysis_results)
            analysis_results['optimization_opportunities'] = opportunities
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Communication pattern analysis error: {e}")
            return {'error': str(e)}
    
    def _get_hour_traffic_count(self, item):
        """Helper method to get traffic count from hour-traffic tuple"""
        return item[1]
    
    async def predict_message_flow(self, time_horizon_minutes: int = 30) -> Dict[str, Any]:
        """
        Predict future message flow using ML models
        """
        try:
            prediction_results = {
                'time_horizon': time_horizon_minutes,
                'predicted_volume': {},
                'predicted_bottlenecks': [],
                'recommended_preparations': [],
                'capacity_requirements': {},
                'route_utilization_forecast': {}
            }
            
            # Extract historical patterns
            historical_data = self._extract_historical_patterns()
            
            if not historical_data:
                logger.info("Insufficient historical data for flow prediction")
                return prediction_results
            
            # Prepare features for prediction
            time_features, volume_targets = self._prepare_time_series_data(historical_data)
            
            if len(time_features) < 5:
                return prediction_results
            
            # Predict message volume
            time_features = np.array(time_features)
            volume_targets = np.array(volume_targets)
            
            # Train time series predictor
            volume_predictor = GradientBoostingRegressor(n_estimators=50, random_state=42)
            volume_predictor.fit(time_features, volume_targets)
            
            # Generate future time features
            future_features = self._generate_future_features(time_horizon_minutes)
            predicted_volumes = volume_predictor.predict(future_features)
            
            # Format predictions
            for i, volume in enumerate(predicted_volumes):
                minute = i * (time_horizon_minutes // len(predicted_volumes))
                prediction_results['predicted_volume'][f'minute_{minute}'] = float(volume)
            
            # Predict bottlenecks
            bottleneck_predictions = self._predict_bottlenecks(predicted_volumes, future_features)
            prediction_results['predicted_bottlenecks'] = bottleneck_predictions
            
            # Generate recommendations
            recommendations = self._generate_flow_recommendations(prediction_results)
            prediction_results['recommended_preparations'] = recommendations
            
            return prediction_results
            
        except Exception as e:
            logger.error(f"Message flow prediction error: {e}")
            return {'error': str(e)}
    
    def update_routing_feedback(self, message_id: str, actual_performance: Dict[str, Any]):
        """
        Update ML models based on actual routing performance feedback
        """
        try:
            # Find the routing decision
            decision = None
            for d in self.routing_decisions:
                if d.message_id == message_id:
                    decision = d
                    break
            
            if not decision:
                logger.warning(f"Routing decision not found for message {message_id}")
                return
            
            # Extract features for feedback learning
            if decision.selected_route in self.routes:
                route = self.routes[decision.selected_route]
                features = self._extract_route_features(route)
                
                # Create feedback entry
                feedback_entry = {
                    'message_id': message_id,
                    'decision': decision,
                    'route_features': features.tolist() if isinstance(features, np.ndarray) else features,
                    'actual_performance': actual_performance,
                    'prediction_accuracy': self._calculate_prediction_accuracy(decision, actual_performance),
                    'timestamp': datetime.utcnow().isoformat()
                }
                self.feedback_history.append(feedback_entry)
                
                # Update route performance history
                route.performance_history.append({
                    'timestamp': datetime.utcnow(),
                    'message_id': message_id,
                    'performance': actual_performance
                })
                
                # Trigger retraining if enough feedback
                if len(self.feedback_history) % 50 == 0:
                    asyncio.create_task(self._retrain_models())
        
        except Exception as e:
            logger.error(f"Routing feedback update error: {e}")
    
    # Private helper methods
    def _extract_message_features(self, message: Message) -> np.ndarray:
        """Extract ML features from message for routing decisions"""
        features = []
        
        # Message characteristics
        features.extend([
            len(message.content),                        # Content length
            len(message.content.split()),                # Word count
            message.content.count('\n'),                 # Line count
            len(json.dumps(message.metadata)),           # Metadata size
            time.time() - message.timestamp.timestamp(), # Message age
        ])
        
        # Priority encoding
        priority_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}
        features.append(priority_map.get(message.priority, 0.5))
        
        # Message type analysis
        type_features = [0, 0, 0, 0]  # request, response, notification, broadcast
        if 'request' in message.message_type.lower():
            type_features[0] = 1
        elif 'response' in message.message_type.lower():
            type_features[1] = 1
        elif 'notification' in message.message_type.lower():
            type_features[2] = 1
        elif 'broadcast' in message.message_type.lower():
            type_features[3] = 1
        
        features.extend(type_features)
        
        # Routing history features
        features.extend([
            len(message.routing_history),                # Number of hops
            len(set(message.routing_history)),           # Unique hops
        ])
        
        # Temporal features
        now = datetime.utcnow()
        features.extend([
            now.hour / 24.0,                            # Hour of day (normalized)
            now.weekday() / 7.0,                        # Day of week (normalized)
        ])
        
        # Pad to fixed size
        target_size = 20
        if len(features) > target_size:
            features = features[:target_size]
        elif len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def _initialize_models(self):
        """Initialize ML models with synthetic training data"""
        # Generate training data for route prediction
        X_route, y_route = self._generate_route_training_data()
        X_latency, y_latency = self._generate_latency_training_data()
        
        # Train models
        if len(X_route) > 0:
            self.route_classifier.fit(X_route, y_route)
        
        if len(X_latency) > 0:
            X_scaled = self.scaler.fit_transform(X_latency)
            self.latency_predictor.fit(X_scaled, y_latency)
    
    def _generate_route_training_data(self) -> Tuple[List[np.ndarray], List[str]]:
        """Generate synthetic training data for route prediction"""
        X, y = [], []
        
        route_types = ['direct', 'relay', 'multicast', 'broadcast']
        
        for i in range(200):
            features = np.random.rand(20)
            
            # Simple heuristics for route selection
            if features[0] > 0.8:  # Large message
                route_type = 'relay'
            elif features[5] > 0.9:  # Critical priority
                route_type = 'direct'
            elif features[9] > 0.7:  # Broadcast type
                route_type = 'broadcast'
            else:
                route_type = np.random.choice(route_types)
            
            X.append(features)
            y.append(route_type)
        
        return X, y
    
    def _generate_latency_training_data(self) -> Tuple[List[np.ndarray], List[float]]:
        """Generate synthetic training data for latency prediction"""
        X, y = [], []
        
        for i in range(150):
            features = np.random.rand(20)
            
            # Latency based on message size and route complexity
            latency = (
                features[0] * 100 +      # Message size impact
                features[11] * 50 +      # Routing hops
                np.random.normal(0, 10)  # Random variation
            )
            latency = max(1.0, latency)  # Minimum 1ms
            
            X.append(features)
            y.append(latency)
        
        return X, y


# Singleton instance
_ai_message_router = None

def get_ai_message_router() -> AIMessageRouter:
    """Get or create AI message router instance"""
    global _ai_message_router
    if not _ai_message_router:
        _ai_message_router = AIMessageRouter()
    return _ai_message_router