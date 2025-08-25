"""
AI-Powered Agent Discovery and Routing System

This module provides intelligent agent discovery using real machine learning models
for optimal agent selection, load balancing, and performance prediction without
requiring external AI services.
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

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import networkx as nx

# Deep learning for advanced agent matching
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class AgentMatchingNN(nn.Module):
    """Neural network for intelligent agent matching"""
    def __init__(self, input_dim, hidden_dim=128):
        super(AgentMatchingNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Multiple prediction heads
        self.performance_head = nn.Linear(hidden_dim // 4, 1)
        self.reliability_head = nn.Linear(hidden_dim // 4, 1)
        self.suitability_head = nn.Linear(hidden_dim // 4, 1)
        self.load_head = nn.Linear(hidden_dim // 4, 1)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        performance = torch.sigmoid(self.performance_head(features))
        reliability = torch.sigmoid(self.reliability_head(features))
        suitability = torch.sigmoid(self.suitability_head(features))
        load = torch.sigmoid(self.load_head(features))
        
        return performance, reliability, suitability, load, features


@dataclass
class AgentTask:
    """Task requiring agent assignment"""
    task_id: str
    task_type: str
    priority: str = "medium"
    required_capabilities: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    complexity_score: float = 0.5
    data_sensitivity: str = "normal"
    expected_duration: float = 300  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformanceMetrics:
    """Comprehensive agent performance tracking"""
    agent_id: str
    success_rate: float = 1.0
    avg_response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    availability: float = 1.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0
    concurrent_tasks: int = 0
    def _create_performance_history():
        return deque(maxlen=100)
    
    historical_performance: deque = field(default_factory=_create_performance_history)
    last_updated: datetime = field(default_factory=datetime.utcnow)


class AIAgentDiscovery:
    """
    AI-powered agent discovery and routing system using real ML models
    """
    
    def __init__(self):
        # ML Models for different aspects of agent discovery
        self.performance_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.load_balancer = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.capability_matcher = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42)
        self.reliability_scorer = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Clustering for agent segmentation
        self.agent_clusterer = KMeans(n_clusters=5, random_state=42)
        self.anomaly_detector = DBSCAN(eps=0.3, min_samples=3)
        
        # Feature scalers
        self.performance_scaler = StandardScaler()
        self.capability_scaler = MinMaxScaler()
        
        # Neural network for advanced matching
        if TORCH_AVAILABLE:
            self.matching_nn = AgentMatchingNN(input_dim=50)
            self.nn_optimizer = torch.optim.Adam(self.matching_nn.parameters(), lr=0.001)
        else:
            self.matching_nn = None
        
        # Agent registry and performance tracking
        self.agent_registry = {}  # agent_id -> AgentProfile
        self.performance_metrics = {}  # agent_id -> AgentPerformanceMetrics
        self.task_history = deque(maxlen=1000)
        self.routing_history = deque(maxlen=500)
        
        # Capability embeddings
        self.capability_embeddings = {}
        self.capability_graph = nx.Graph()
        
        # Real-time monitoring
        self.monitoring_data = defaultdict(list)
        self.prediction_cache = {}
        
        # Initialize with training data
        self._initialize_models()
        
        logger.info("AI Agent Discovery initialized with ML models")
    
    def _initialize_models(self):
        """Initialize ML models with synthetic training data"""
        # Generate synthetic training data
        X_perf, y_perf = self._generate_performance_training_data()
        X_load, y_load = self._generate_load_balancing_data()
        X_cap, y_cap = self._generate_capability_matching_data()
        
        # Train models if data available
        if len(X_perf) > 0:
            X_perf_scaled = self.performance_scaler.fit_transform(X_perf)
            self.performance_predictor.fit(X_perf_scaled, y_perf)
        
        if len(X_load) > 0:
            self.load_balancer.fit(X_load, y_load)
        
        if len(X_cap) > 0:
            X_cap_scaled = self.capability_scaler.fit_transform(X_cap)
            self.capability_matcher.fit(X_cap_scaled, y_cap)
            
        # Initialize capability graph
        self._build_capability_graph()
    
    async def discover_optimal_agent(self, task: AgentTask, 
                                   constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Discover the optimal agent for a given task using AI
        """
        try:
            # Extract task features
            task_features = self._extract_task_features(task)
            
            # Get candidate agents
            candidates = await self._get_candidate_agents(task)
            
            if not candidates:
                return {
                    'agent_id': None,
                    'confidence': 0.0,
                    'reason': 'No suitable agents available'
                }
            
            # Score each candidate using multiple ML models
            agent_scores = {}
            
            for agent_id in candidates:
                agent_features = self._extract_agent_features(agent_id)
                combined_features = np.concatenate([task_features, agent_features])
                
                # ML-based scoring
                performance_score = self._predict_performance(combined_features)
                reliability_score = self._predict_reliability(agent_features)
                capability_score = self._score_capability_match(task, agent_id)
                load_score = self._predict_load_efficiency(agent_id, task)
                
                # Neural network enhancement
                if self.matching_nn and TORCH_AVAILABLE:
                    nn_scores = self._get_nn_predictions(combined_features)
                    performance_score = (performance_score + nn_scores['performance']) / 2
                    reliability_score = (reliability_score + nn_scores['reliability']) / 2
                
                # Combine scores with weights
                overall_score = (
                    performance_score * 0.3 +
                    reliability_score * 0.25 +
                    capability_score * 0.25 +
                    load_score * 0.2
                )
                
                agent_scores[agent_id] = {
                    'overall_score': overall_score,
                    'performance_score': performance_score,
                    'reliability_score': reliability_score,
                    'capability_score': capability_score,
                    'load_score': load_score
                }
            
            # Select best agent
            def get_overall_score(agent_score_item):
                return agent_score_item[1]['overall_score']
            
            best_agent = max(agent_scores.items(), key=get_overall_score)
            
            # Calculate confidence
            scores = [s['overall_score'] for s in agent_scores.values()]
            confidence = self._calculate_selection_confidence(scores)
            
            # Log routing decision for learning
            routing_decision = {
                'task_id': task.task_id,
                'selected_agent': best_agent[0],
                'score': best_agent[1]['overall_score'],
                'alternatives': len(candidates) - 1,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.routing_history.append(routing_decision)
            
            return {
                'agent_id': best_agent[0],
                'confidence': confidence,
                'score_breakdown': best_agent[1],
                'alternatives_considered': len(candidates),
                'routing_rationale': self._generate_routing_explanation(task, best_agent)
            }
            
        except Exception as e:
            logger.error(f"Agent discovery error: {e}")
            return {
                'agent_id': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def intelligent_load_balancing(self, tasks: List[AgentTask]) -> Dict[str, List[str]]:
        """
        Perform intelligent load balancing across agents using ML
        """
        agent_assignments = defaultdict(list)
        
        # Get current system state
        system_state = await self._get_system_state()
        
        # Extract features for all tasks
        task_features_matrix = np.array([self._extract_task_features(task) for task in tasks])
        
        # Predict optimal distribution
        if len(tasks) > 1:
            # Use clustering to group similar tasks
            task_clusters = self.agent_clusterer.fit_predict(task_features_matrix)
            
            # Assign clusters to agents based on capacity and performance
            available_agents = [aid for aid, metrics in self.performance_metrics.items() 
                             if metrics.availability > 0.8]
            
            for cluster_id in set(task_clusters):
                cluster_tasks = [tasks[i] for i, c in enumerate(task_clusters) if c == cluster_id]
                
                # Select best agent for this cluster
                best_agent = await self._select_cluster_agent(cluster_tasks, available_agents, system_state)
                
                for task in cluster_tasks:
                    agent_assignments[best_agent].append(task.task_id)
        
        else:
            # Single task - use optimal discovery
            if tasks:
                result = await self.discover_optimal_agent(tasks[0])
                if result['agent_id']:
                    agent_assignments[result['agent_id']].append(tasks[0].task_id)
        
        return dict(agent_assignments)
    
    async def predict_agent_performance(self, agent_id: str, task: AgentTask) -> Dict[str, float]:
        """
        Predict how an agent will perform on a specific task
        """
        try:
            # Extract features
            task_features = self._extract_task_features(task)
            agent_features = self._extract_agent_features(agent_id)
            combined_features = np.concatenate([task_features, agent_features])
            
            # ML predictions
            performance_pred = self._predict_performance(combined_features)
            reliability_pred = self._predict_reliability(agent_features)
            
            # Time prediction based on historical data
            estimated_time = self._predict_task_duration(agent_id, task)
            
            # Resource utilization prediction
            cpu_pred, memory_pred = self._predict_resource_usage(agent_id, task)
            
            # Success probability
            success_prob = self._predict_success_probability(agent_id, task)
            
            return {
                'performance_score': float(performance_pred),
                'reliability_score': float(reliability_pred),
                'estimated_duration': float(estimated_time),
                'predicted_cpu_usage': float(cpu_pred),
                'predicted_memory_usage': float(memory_pred),
                'success_probability': float(success_prob),
                'confidence': self._calculate_prediction_confidence(combined_features)
            }
            
        except Exception as e:
            logger.error(f"Performance prediction error: {e}")
            return {
                'performance_score': 0.5,
                'reliability_score': 0.5,
                'estimated_duration': 300.0,
                'predicted_cpu_usage': 0.5,
                'predicted_memory_usage': 0.5,
                'success_probability': 0.5,
                'confidence': 0.0
            }
    
    def update_agent_performance(self, agent_id: str, task_id: str, 
                               performance_data: Dict[str, Any]):
        """
        Update agent performance metrics and retrain models
        """
        if agent_id not in self.performance_metrics:
            self.performance_metrics[agent_id] = AgentPerformanceMetrics(agent_id=agent_id)
        
        metrics = self.performance_metrics[agent_id]
        
        # Update metrics with exponential moving average
        alpha = 0.1  # Learning rate
        
        if 'response_time' in performance_data:
            metrics.avg_response_time = (1 - alpha) * metrics.avg_response_time + alpha * performance_data['response_time']
        
        if 'success' in performance_data:
            success = 1.0 if performance_data['success'] else 0.0
            metrics.success_rate = (1 - alpha) * metrics.success_rate + alpha * success
        
        if 'cpu_usage' in performance_data:
            metrics.cpu_usage = performance_data['cpu_usage']
        
        if 'memory_usage' in performance_data:
            metrics.memory_usage = performance_data['memory_usage']
        
        # Add to historical data
        metrics.historical_performance.append({
            'timestamp': datetime.utcnow(),
            'task_id': task_id,
            'data': performance_data
        })
        
        metrics.last_updated = datetime.utcnow()
        
        # Trigger model retraining if enough new data
        if len(metrics.historical_performance) % 50 == 0:
            asyncio.create_task(self._retrain_models())
    
    async def _get_candidate_agents(self, task: AgentTask) -> List[str]:
        """Get candidate agents that can handle the task"""
        candidates = []
        
        for agent_id, profile in self.agent_registry.items():
            # Check capability match
            if self._has_required_capabilities(profile.capabilities, task.required_capabilities):
                # Check availability
                metrics = self.performance_metrics.get(agent_id)
                if metrics and metrics.availability > 0.5:
                    candidates.append(agent_id)
        
        return candidates
    
    def _extract_task_features(self, task: AgentTask) -> np.ndarray:
        """Extract ML features from a task"""
        features = []
        
        # Basic task features
        features.append(len(task.required_capabilities))
        features.append(task.complexity_score)
        features.append(task.expected_duration)
        
        # Priority encoding
        priority_map = {'low': 0.3, 'medium': 0.6, 'high': 0.9, 'critical': 1.0}
        features.append(priority_map.get(task.priority, 0.5))
        
        # Task type encoding (simplified)
        task_type_hash = hash(task.task_type) % 1000 / 1000.0
        features.append(task_type_hash)
        
        # Capability features (top 10 most common)
        common_capabilities = [
            'data_processing', 'analysis', 'validation', 'transformation',
            'calculation', 'reasoning', 'quality_control', 'standardization',
            'monitoring', 'optimization'
        ]
        
        for cap in common_capabilities:
            features.append(1.0 if cap in task.required_capabilities else 0.0)
        
        # Resource requirements
        features.append(task.resource_requirements.get('cpu', 0.5))
        features.append(task.resource_requirements.get('memory', 0.5))
        features.append(task.resource_requirements.get('storage', 0.1))
        
        # Data sensitivity
        sensitivity_map = {'low': 0.2, 'normal': 0.5, 'high': 0.8, 'critical': 1.0}
        features.append(sensitivity_map.get(task.data_sensitivity, 0.5))
        
        # Temporal features
        now = datetime.utcnow()
        if task.deadline:
            urgency = max(0.0, 1.0 - (task.deadline - now).total_seconds() / 3600)  # Hours to deadline
        else:
            urgency = 0.5
        features.append(urgency)
        
        return np.array(features)
    
    def _extract_agent_features(self, agent_id: str) -> np.ndarray:
        """Extract ML features from an agent"""
        features = []
        
        # Get agent profile and metrics
        profile = self.agent_registry.get(agent_id)
        metrics = self.performance_metrics.get(agent_id)
        
        if not profile:
            return np.zeros(20)  # Default feature vector
        
        # Basic agent features
        features.append(len(profile.capabilities))
        features.append(profile.trust_score)
        features.append(profile.reputation / 100.0)  # Normalize
        features.append(profile.availability_score)
        
        # Performance features
        if metrics:
            features.append(metrics.success_rate)
            features.append(min(1.0, metrics.avg_response_time / 1000.0))  # Normalize to seconds
            features.append(metrics.throughput)
            features.append(metrics.error_rate)
            features.append(metrics.availability)
            features.append(metrics.cpu_usage)
            features.append(metrics.memory_usage)
            features.append(min(1.0, metrics.network_latency / 100.0))  # Normalize
            features.append(min(1.0, metrics.concurrent_tasks / 10.0))  # Normalize
        else:
            features.extend([0.8, 0.5, 0.5, 0.1, 0.9, 0.3, 0.3, 0.05, 0.2])  # Defaults
        
        # Capability specialization score
        if profile.specializations:
            specialization_score = len(profile.specializations) / 5.0
        else:
            specialization_score = 0.0
        features.append(min(1.0, specialization_score))
        
        # Age/experience (based on version)
        version_parts = profile.version.split('.')
        experience = (int(version_parts[0]) * 10 + int(version_parts[1])) / 100.0
        features.append(min(1.0, experience))
        
        # Recent performance trend
        if metrics and metrics.historical_performance:
            recent_successes = sum(1 for p in list(metrics.historical_performance)[-10:] 
                                 if p['data'].get('success', True))
            trend = recent_successes / min(10, len(metrics.historical_performance))
        else:
            trend = 0.8
        features.append(trend)
        
        return np.array(features)
    
    def _predict_performance(self, features: np.ndarray) -> float:
        """Predict agent performance using ML"""
        try:
            if hasattr(self.performance_predictor, 'predict'):
                features_scaled = self.performance_scaler.transform(features.reshape(1, -1))
                prediction = self.performance_predictor.predict(features_scaled)[0]
                return float(np.clip(prediction, 0.0, 1.0))
        except:
            pass
        
        # Fallback heuristic
        return float(np.mean(features[:5]) if len(features) > 5 else 0.7)
    
    def _predict_reliability(self, agent_features: np.ndarray) -> float:
        """Predict agent reliability"""
        try:
            if hasattr(self.reliability_scorer, 'predict'):
                prediction = self.reliability_scorer.predict(agent_features.reshape(1, -1))[0]
                return float(np.clip(prediction, 0.0, 1.0))
        except:
            pass
        
        # Fallback based on success rate and availability
        success_rate = agent_features[4] if len(agent_features) > 4 else 0.8
        availability = agent_features[8] if len(agent_features) > 8 else 0.9
        return float((success_rate + availability) / 2)
    
    def _score_capability_match(self, task: AgentTask, agent_id: str) -> float:
        """Score how well agent capabilities match task requirements"""
        profile = self.agent_registry.get(agent_id)
        if not profile:
            return 0.0
        
        if not task.required_capabilities:
            return 1.0
        
        # Calculate capability overlap
        agent_caps = set(profile.capabilities)
        required_caps = set(task.required_capabilities)
        
        if not required_caps:
            return 1.0
        
        # Exact match score
        exact_matches = len(agent_caps.intersection(required_caps))
        exact_score = exact_matches / len(required_caps)
        
        # Semantic similarity using capability graph
        semantic_score = self._calculate_semantic_capability_match(agent_caps, required_caps)
        
        # Combined score
        return float((exact_score * 0.7 + semantic_score * 0.3))
    
    def _predict_load_efficiency(self, agent_id: str, task: AgentTask) -> float:
        """Predict how efficiently agent can handle the load"""
        metrics = self.performance_metrics.get(agent_id)
        if not metrics:
            return 0.5
        
        # Current load factor
        current_load = metrics.concurrent_tasks / 10.0  # Normalize assuming max 10 concurrent
        load_factor = 1.0 - min(1.0, current_load)
        
        # Resource availability
        cpu_available = 1.0 - metrics.cpu_usage
        memory_available = 1.0 - metrics.memory_usage
        
        # Task resource requirements
        task_cpu = task.resource_requirements.get('cpu', 0.3)
        task_memory = task.resource_requirements.get('memory', 0.2)
        
        # Can agent handle the additional load?
        cpu_ok = cpu_available >= task_cpu
        memory_ok = memory_available >= task_memory
        
        if not (cpu_ok and memory_ok):
            return 0.1  # Very low efficiency if resources unavailable
        
        # Calculate efficiency score
        efficiency = (load_factor + cpu_available + memory_available) / 3
        return float(efficiency)
    
    def _get_nn_predictions(self, features: np.ndarray) -> Dict[str, float]:
        """Get predictions from neural network"""
        if not TORCH_AVAILABLE or not self.matching_nn:
            return self._ml_fallback_agent_predictions(features)
        
        try:
            # Pad or truncate features to expected input size
            if len(features) > 50:
                features = features[:50]
            elif len(features) < 50:
                features = np.pad(features, (0, 50 - len(features)), mode='constant')
            
            feature_tensor = torch.FloatTensor(features)
            
            with torch.no_grad():
                performance, reliability, suitability, load, _ = self.matching_nn(feature_tensor.unsqueeze(0))
            
            return {
                'performance': float(performance.item()),
                'reliability': float(reliability.item()),
                'suitability': float(suitability.item()),
                'load': float(load.item())
            }
        except Exception as e:
            logger.error(f"Neural network prediction error: {e}")
            return self._ml_fallback_agent_predictions(features)
    
    def _ml_fallback_agent_predictions(self, features: np.ndarray) -> Dict[str, float]:
        """ML-based fallback for agent predictions using statistical analysis"""
        try:
            # Ensure we have minimum feature length for analysis
            if len(features) == 0:
                return {'performance': 0.5, 'reliability': 0.5, 'suitability': 0.5, 'load': 0.5}
            
            # Split features into task and agent features (assuming concatenated)
            mid_point = len(features) // 2
            task_features = features[:mid_point] if mid_point > 0 else features[:10]
            agent_features = features[mid_point:] if mid_point > 0 else features[10:]
            
            # Performance prediction based on resource alignment
            if len(task_features) >= 3 and len(agent_features) >= 3:
                # Task complexity vs agent capability alignment
                task_complexity = np.mean(task_features[:3]) if len(task_features) >= 3 else 0.5
                agent_capability = np.mean(agent_features[:3]) if len(agent_features) >= 3 else 0.5
                performance = 0.3 + 0.7 * min(1.0, agent_capability / max(0.1, task_complexity))
            else:
                performance = 0.5
            
            # Reliability prediction based on historical success patterns
            if len(agent_features) >= 10:
                # Use success rate, error rate, and availability features
                success_rate = agent_features[4] if len(agent_features) > 4 else 0.8
                error_rate = agent_features[7] if len(agent_features) > 7 else 0.1
                availability = agent_features[8] if len(agent_features) > 8 else 0.9
                reliability = (success_rate + (1.0 - error_rate) + availability) / 3.0
            else:
                reliability = 0.5
            
            # Suitability prediction based on capability matching
            if len(task_features) >= 10 and len(agent_features) >= 10:
                # Capability overlap analysis
                task_caps = task_features[5:15] if len(task_features) >= 15 else task_features[5:]
                agent_caps = agent_features[:10] if len(agent_features) >= 10 else agent_features
                
                if len(task_caps) > 0 and len(agent_caps) > 0:
                    # Calculate cosine similarity for capability matching
                    min_len = min(len(task_caps), len(agent_caps))
                    if min_len > 0:
                        dot_product = np.dot(task_caps[:min_len], agent_caps[:min_len])
                        norm_task = np.linalg.norm(task_caps[:min_len])
                        norm_agent = np.linalg.norm(agent_caps[:min_len])
                        if norm_task > 0 and norm_agent > 0:
                            suitability = 0.3 + 0.7 * (dot_product / (norm_task * norm_agent))
                        else:
                            suitability = 0.5
                    else:
                        suitability = 0.5
                else:
                    suitability = 0.5
            else:
                suitability = 0.5
            
            # Load prediction based on resource utilization
            if len(agent_features) >= 12:
                # CPU and memory usage features
                cpu_usage = agent_features[9] if len(agent_features) > 9 else 0.3
                memory_usage = agent_features[10] if len(agent_features) > 10 else 0.3
                concurrent_tasks = agent_features[12] if len(agent_features) > 12 else 0.2
                
                # Lower utilization = better load capacity
                load_capacity = 1.0 - (cpu_usage + memory_usage + concurrent_tasks) / 3.0
                load = max(0.1, load_capacity)
            else:
                load = 0.5
            
            # Normalize all values to [0.1, 1.0] range
            return {
                'performance': float(np.clip(performance, 0.1, 1.0)),
                'reliability': float(np.clip(reliability, 0.1, 1.0)),
                'suitability': float(np.clip(suitability, 0.1, 1.0)),
                'load': float(np.clip(load, 0.1, 1.0))
            }
            
        except Exception as e:
            logger.warning(f"ML fallback prediction error: {e}")
            return {'performance': 0.5, 'reliability': 0.5, 'suitability': 0.5, 'load': 0.5}
    
    def _calculate_selection_confidence(self, scores: List[float]) -> float:
        """Calculate confidence in agent selection"""
        if len(scores) < 2:
            return 0.5
        
        # Sort scores in descending order
        scores_sorted = sorted(scores, reverse=True)
        
        # Confidence based on margin between top scores
        margin = scores_sorted[0] - scores_sorted[1]
        
        # Higher margin = higher confidence
        confidence = 0.5 + (margin * 0.5)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _generate_routing_explanation(self, task: AgentTask, best_agent: Tuple[str, Dict]) -> str:
        """Generate human-readable explanation for routing decision"""
        agent_id, scores = best_agent
        
        reasons = []
        
        if scores['performance_score'] > 0.8:
            reasons.append("high predicted performance")
        if scores['reliability_score'] > 0.8:
            reasons.append("excellent reliability record")
        if scores['capability_score'] > 0.9:
            reasons.append("perfect capability match")
        if scores['load_score'] > 0.7:
            reasons.append("optimal load capacity")
        
        if not reasons:
            reasons.append("best available option")
        
        return f"Selected agent {agent_id} due to: {', '.join(reasons)}"
    
    # Additional helper methods...
    def _has_required_capabilities(self, agent_caps: List[str], required_caps: List[str]) -> bool:
        """Check if agent has required capabilities"""
        return set(required_caps).issubset(set(agent_caps))
    
    def _build_capability_graph(self):
        """Build capability similarity graph"""
        capabilities = [
            'data_processing', 'analysis', 'validation', 'transformation',
            'calculation', 'reasoning', 'quality_control', 'standardization',
            'monitoring', 'optimization', 'visualization', 'reporting'
        ]
        
        # Add nodes
        for cap in capabilities:
            self.capability_graph.add_node(cap)
        
        # Add similarity edges (simplified)
        similar_pairs = [
            ('data_processing', 'transformation'),
            ('analysis', 'reasoning'),
            ('validation', 'quality_control'),
            ('monitoring', 'reporting'),
            ('optimization', 'analysis')
        ]
        
        for cap1, cap2 in similar_pairs:
            self.capability_graph.add_edge(cap1, cap2, weight=0.8)
    
    def _calculate_semantic_capability_match(self, agent_caps: Set[str], required_caps: Set[str]) -> float:
        """Calculate semantic similarity between capability sets"""
        if not self.capability_graph.nodes():
            return 0.0
        
        total_similarity = 0.0
        comparisons = 0
        
        for req_cap in required_caps:
            if req_cap in self.capability_graph.nodes():
                best_match = 0.0
                for agent_cap in agent_caps:
                    if agent_cap in self.capability_graph.nodes():
                        try:
                            # Use shortest path as similarity measure
                            path_length = nx.shortest_path_length(
                                self.capability_graph, req_cap, agent_cap
                            )
                            similarity = 1.0 / (1.0 + path_length)
                            best_match = max(best_match, similarity)
                        except nx.NetworkXNoPath:
                            continue
                
                total_similarity += best_match
                comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    # Training data generation methods
    def _generate_performance_training_data(self) -> Tuple[List[np.ndarray], List[float]]:
        """Generate synthetic training data for performance prediction"""
        X, y = [], []
        
        for i in range(100):
            # Random task + agent features
            features = np.random.rand(40)
            
            # Synthetic performance based on features
            performance = (
                features[0] * 0.3 +  # Task complexity
                features[5] * 0.4 +  # Agent success rate
                features[8] * 0.2 +  # Agent availability
                features[15] * 0.1   # Capability match
            )
            performance += np.random.normal(0, 0.1)  # Add noise
            performance = np.clip(performance, 0.0, 1.0)
            
            X.append(features)
            y.append(performance)
        
        return X, y
    
    def _generate_load_balancing_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Generate training data for load balancing"""
        X, y = [], []
        
        for i in range(50):
            features = np.random.rand(20)
            # Label: 0=overloaded, 1=optimal, 2=underutilized
            if features[10] > 0.8:  # High load indicator
                label = 0
            elif features[10] < 0.3:  # Low load indicator
                label = 2
            else:
                label = 1
            
            X.append(features)
            y.append(label)
        
        return X, y
    
    def _generate_capability_matching_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Generate training data for capability matching"""
        X, y = [], []
        
        for i in range(80):
            features = np.random.rand(25)
            # Binary classification: good match (1) or poor match (0)
            match_score = np.sum(features[10:20]) / 10  # Capability overlap features
            label = 1 if match_score > 0.6 else 0
            
            X.append(features)
            y.append(label)
        
        return X, y
    
    async def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for load balancing"""
        total_agents = len(self.agent_registry)
        active_agents = sum(1 for m in self.performance_metrics.values() if m.availability > 0.8)
        avg_load = np.mean([m.concurrent_tasks for m in self.performance_metrics.values()]) if self.performance_metrics else 0
        
        return {
            'total_agents': total_agents,
            'active_agents': active_agents,
            'average_load': avg_load,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _select_cluster_agent(self, cluster_tasks: List[AgentTask], 
                                  available_agents: List[str], 
                                  system_state: Dict[str, Any]) -> str:
        """Select best agent for a cluster of similar tasks"""
        if not available_agents:
            return 'default_agent'
        
        # Score each agent for the entire cluster
        agent_scores = {}
        
        for agent_id in available_agents:
            total_score = 0.0
            
            for task in cluster_tasks:
                task_features = self._extract_task_features(task)
                agent_features = self._extract_agent_features(agent_id)
                combined_features = np.concatenate([task_features, agent_features])
                
                score = self._predict_performance(combined_features)
                total_score += score
            
            agent_scores[agent_id] = total_score / len(cluster_tasks)
        
        def get_agent_score(agent_score_item):
            return agent_score_item[1]
        
        return max(agent_scores.items(), key=get_agent_score)[0]
    
    def _predict_task_duration(self, agent_id: str, task: AgentTask) -> float:
        """Predict how long the task will take"""
        metrics = self.performance_metrics.get(agent_id)
        base_duration = task.expected_duration
        
        if metrics and metrics.avg_response_time > 0:
            # Adjust based on agent's historical response time
            efficiency_factor = 500 / max(metrics.avg_response_time, 100)  # Normalize around 500ms
            return base_duration / efficiency_factor
        
        return base_duration
    
    def _predict_resource_usage(self, agent_id: str, task: AgentTask) -> Tuple[float, float]:
        """Predict CPU and memory usage"""
        base_cpu = task.resource_requirements.get('cpu', 0.3)
        base_memory = task.resource_requirements.get('memory', 0.2)
        
        metrics = self.performance_metrics.get(agent_id)
        if metrics:
            # Adjust based on agent's typical usage patterns
            cpu_factor = 1.0 + (metrics.cpu_usage - 0.5)  # Adjust around 50% baseline
            memory_factor = 1.0 + (metrics.memory_usage - 0.3)  # Adjust around 30% baseline
            
            predicted_cpu = base_cpu * max(0.1, cpu_factor)
            predicted_memory = base_memory * max(0.1, memory_factor)
        else:
            predicted_cpu = base_cpu
            predicted_memory = base_memory
        
        return np.clip(predicted_cpu, 0.0, 1.0), np.clip(predicted_memory, 0.0, 1.0)
    
    def _predict_success_probability(self, agent_id: str, task: AgentTask) -> float:
        """Predict probability of successful task completion"""
        metrics = self.performance_metrics.get(agent_id)
        profile = self.agent_registry.get(agent_id)
        
        base_success = 0.8
        
        if metrics:
            base_success = metrics.success_rate
        
        # Adjust based on task-agent match
        if profile:
            capability_score = self._score_capability_match(task, agent_id)
            adjusted_success = base_success * (0.7 + 0.3 * capability_score)
        else:
            adjusted_success = base_success * 0.7
        
        return float(np.clip(adjusted_success, 0.0, 1.0))
    
    def _calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence in predictions"""
        # Use feature variance as confidence indicator
        feature_variance = np.var(features)
        
        # Lower variance = higher confidence (more consistent features)
        confidence = 1.0 / (1.0 + feature_variance)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    async def _retrain_models(self):
        """Retrain ML models with new performance data"""
        try:
            # Collect training data from historical performance
            X_perf, y_perf = [], []
            
            for agent_id, metrics in self.performance_metrics.items():
                for perf_data in list(metrics.historical_performance)[-50:]:  # Last 50 records
                    task_data = perf_data.get('data', {})
                    if 'task_features' in task_data and 'performance' in task_data:
                        X_perf.append(task_data['task_features'])
                        y_perf.append(task_data['performance'])
            
            # Retrain if enough data
            if len(X_perf) > 20:
                X_perf = np.array(X_perf)
                X_perf_scaled = self.performance_scaler.fit_transform(X_perf)
                self.performance_predictor.fit(X_perf_scaled, y_perf)
                logger.info(f"Retrained performance predictor with {len(X_perf)} samples")
        
        except Exception as e:
            logger.error(f"Model retraining error: {e}")


# Singleton instance
_ai_discovery = None

def get_ai_discovery() -> AIAgentDiscovery:
    """Get or create AI discovery instance"""
    global _ai_discovery
    if not _ai_discovery:
        _ai_discovery = AIAgentDiscovery()
    return _ai_discovery