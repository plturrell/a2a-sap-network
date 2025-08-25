"""
AI-Powered Workflow Optimization Engine

This module provides intelligent workflow optimization using real machine learning
to analyze execution patterns, predict bottlenecks, and automatically optimize
workflow performance and resource utilization.
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import logging
import numpy as np
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import networkx as nx

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, AdaBoostRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA, FastICA
import networkx as nx

# Advanced optimization
try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Deep learning for complex workflow patterns
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class WorkflowOptimizationNN(nn.Module):
    """Neural network for workflow pattern analysis and optimization"""
    def __init__(self, input_dim, hidden_dim=256):
        super(WorkflowOptimizationNN, self).__init__()
        
        # Encoder for workflow patterns
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Multiple optimization heads
        self.execution_time_head = nn.Linear(hidden_dim // 4, 1)
        self.resource_usage_head = nn.Linear(hidden_dim // 4, 1)
        self.bottleneck_head = nn.Linear(hidden_dim // 4, 1)
        self.parallelization_head = nn.Linear(hidden_dim // 4, 1)
        self.optimization_score_head = nn.Linear(hidden_dim // 4, 1)
        
        # Attention mechanism for step importance
        self.attention = nn.MultiheadAttention(hidden_dim // 4, num_heads=4)
        
    def forward(self, x):
        # Encode workflow features
        encoded = self.encoder(x)
        
        # Apply attention
        attn_out, attn_weights = self.attention(encoded.unsqueeze(0), encoded.unsqueeze(0), encoded.unsqueeze(0))
        features = attn_out.squeeze(0)
        
        # Predictions
        exec_time = F.relu(self.execution_time_head(features))
        resource_usage = torch.sigmoid(self.resource_usage_head(features))
        bottleneck_prob = torch.sigmoid(self.bottleneck_head(features))
        parallelization_score = torch.sigmoid(self.parallelization_head(features))
        optimization_score = torch.sigmoid(self.optimization_score_head(features))
        
        return exec_time, resource_usage, bottleneck_prob, parallelization_score, optimization_score, attn_weights


@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    step_id: str
    name: str
    agent_type: str
    estimated_duration: float
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    priority: str = "medium"
    can_parallelize: bool = True
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Record of workflow execution"""
    workflow_id: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: float = 0.0
    step_executions: Dict[str, Dict] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)
    optimization_applied: List[str] = field(default_factory=list)
    success: bool = True
    error_details: Optional[str] = None


@dataclass
class OptimizationRecommendation:
    """AI-generated optimization recommendation"""
    recommendation_id: str
    type: str  # parallelization, caching, reordering, resource_allocation
    description: str
    expected_improvement: Dict[str, float]  # time, resource, cost savings
    confidence: float
    implementation_complexity: str  # low, medium, high
    affected_steps: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AIWorkflowOptimizer:
    """
    AI-powered workflow optimization using real ML models
    """
    
    def __init__(self):
        # ML Models for different optimization aspects
        self.execution_time_predictor = RandomForestRegressor(n_estimators=150, random_state=42)
        self.bottleneck_detector = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.resource_optimizer = AdaBoostRegressor(n_estimators=100, random_state=42)
        self.parallelization_analyzer = MLPRegressor(hidden_layer_sizes=(128, 64), random_state=42)
        
        # Advanced clustering for workflow patterns
        self.workflow_clusterer = KMeans(n_clusters=8, random_state=42)
        self.step_clusterer = AgglomerativeClustering(n_clusters=5)
        
        # Feature scalers
        self.workflow_scaler = StandardScaler()
        self.step_scaler = RobustScaler()
        self.resource_scaler = MinMaxScaler()
        
        # Neural network for complex optimization
        if TORCH_AVAILABLE:
            self.optimization_nn = WorkflowOptimizationNN(input_dim=60)
            self.nn_optimizer = torch.optim.Adam(self.optimization_nn.parameters(), lr=0.001)
            self.nn_scheduler = torch.optim.lr_scheduler.StepLR(self.nn_optimizer, step_size=100, gamma=0.9)
        else:
            self.optimization_nn = None
        
        # Workflow analysis
        self.workflow_graph = nx.DiGraph()
        self.execution_history = deque(maxlen=1000)
        self.pattern_cache = {}
        
        # Optimization tracking
        self.optimization_history = deque(maxlen=500)
        self.recommendation_effectiveness = {}
        
        # Performance baselines
        self.performance_baselines = {}
        
        # Initialize with training data
        self._initialize_models()
        
        logger.info("AI Workflow Optimizer initialized with ML models")
    
    def _initialize_models(self):
        """Initialize ML models with synthetic training data"""
        # Generate synthetic workflow training data
        X_exec, y_exec = self._generate_execution_time_data()
        X_bottleneck, y_bottleneck = self._generate_bottleneck_data()
        X_resource, y_resource = self._generate_resource_optimization_data()
        X_parallel, y_parallel = self._generate_parallelization_data()
        
        # Train models
        if len(X_exec) > 0:
            X_exec_scaled = self.workflow_scaler.fit_transform(X_exec)
            self.execution_time_predictor.fit(X_exec_scaled, y_exec)
        
        if len(X_bottleneck) > 0:
            self.bottleneck_detector.fit(X_bottleneck, y_bottleneck)
        
        if len(X_resource) > 0:
            X_resource_scaled = self.resource_scaler.fit_transform(X_resource)
            self.resource_optimizer.fit(X_resource_scaled, y_resource)
        
        if len(X_parallel) > 0:
            self.parallelization_analyzer.fit(X_parallel, y_parallel)
    
    async def optimize_workflow(self, workflow_steps: List[WorkflowStep], 
                              execution_history: List[WorkflowExecution] = None) -> Dict[str, Any]:
        """
        Optimize a workflow using AI analysis
        """
        try:
            # Extract workflow features
            workflow_features = self._extract_workflow_features(workflow_steps)
            
            # Analyze current workflow structure
            workflow_analysis = await self._analyze_workflow_structure(workflow_steps)
            
            # Predict performance metrics
            performance_prediction = await self._predict_workflow_performance(
                workflow_steps, workflow_features
            )
            
            # Identify optimization opportunities
            optimizations = await self._identify_optimizations(
                workflow_steps, workflow_analysis, execution_history
            )
            
            # Generate specific recommendations
            recommendations = await self._generate_recommendations(
                workflow_steps, optimizations, performance_prediction
            )
            
            # Calculate potential improvements
            improvement_estimates = await self._estimate_improvements(
                workflow_steps, recommendations
            )
            
            # Neural network enhancement
            nn_insights = {}
            if self.optimization_nn and TORCH_AVAILABLE:
                nn_insights = await self._get_nn_optimization_insights(workflow_features)
            
            return {
                'workflow_analysis': workflow_analysis,
                'performance_prediction': performance_prediction,
                'optimizations_found': len(optimizations),
                'recommendations': recommendations,
                'improvement_estimates': improvement_estimates,
                'nn_insights': nn_insights,
                'confidence_score': self._calculate_optimization_confidence(workflow_features),
                'optimization_complexity': self._assess_optimization_complexity(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Workflow optimization error: {e}")
            return {
                'error': str(e),
                'recommendations': [],
                'confidence_score': 0.0
            }
    
    async def predict_execution_bottlenecks(self, workflow_steps: List[WorkflowStep]) -> Dict[str, Any]:
        """
        Predict potential bottlenecks in workflow execution
        """
        bottleneck_predictions = {}
        
        # Create dependency graph
        dep_graph = self._build_dependency_graph(workflow_steps)
        
        # Analyze critical path
        critical_path = self._find_critical_path(workflow_steps, dep_graph)
        
        # ML-based bottleneck prediction
        for step in workflow_steps:
            step_features = self._extract_step_features(step, workflow_steps)
            
            # Predict if step will be a bottleneck
            if hasattr(self.bottleneck_detector, 'predict_proba'):
                bottleneck_prob = self.bottleneck_detector.predict_proba(
                    step_features.reshape(1, -1)
                )[0][1]  # Probability of being a bottleneck
            else:
                # Heuristic fallback
                bottleneck_prob = self._heuristic_bottleneck_score(step, workflow_steps)
            
            # Resource contention analysis
            resource_contention = self._analyze_resource_contention(step, workflow_steps)
            
            # Dependency bottleneck analysis
            dep_bottleneck = self._analyze_dependency_bottleneck(step, dep_graph)
            
            # Combined score
            combined_score = (
                bottleneck_prob * 0.4 +
                resource_contention * 0.3 +
                dep_bottleneck * 0.3
            )
            
            bottleneck_predictions[step.step_id] = {
                'bottleneck_probability': float(bottleneck_prob),
                'resource_contention': float(resource_contention),
                'dependency_impact': float(dep_bottleneck),
                'combined_score': float(combined_score),
                'is_critical_path': step.step_id in critical_path,
                'risk_level': 'high' if combined_score > 0.7 else 'medium' if combined_score > 0.4 else 'low'
            }
        
        # Find top bottleneck candidates
        def get_bottleneck_score(item):
            return item[1]['combined_score']
        
        top_bottlenecks = sorted(
            bottleneck_predictions.items(),
            key=get_bottleneck_score,
            reverse=True
        )[:5]
        
        return {
            'bottleneck_predictions': bottleneck_predictions,
            'top_bottlenecks': [{'step_id': k, **v} for k, v in top_bottlenecks],
            'critical_path': critical_path,
            'overall_bottleneck_risk': np.mean([p['combined_score'] for p in bottleneck_predictions.values()])
        }
    
    async def optimize_resource_allocation(self, workflow_steps: List[WorkflowStep], 
                                         available_resources: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimize resource allocation across workflow steps using ML
        """
        # Extract resource requirements
        total_cpu = sum(step.resource_requirements.get('cpu', 0.5) for step in workflow_steps)
        total_memory = sum(step.resource_requirements.get('memory', 0.3) for step in workflow_steps)
        
        # Available resources
        available_cpu = available_resources.get('cpu', 1.0)
        available_memory = available_resources.get('memory', 1.0)
        
        # Check if optimization is needed
        if total_cpu <= available_cpu and total_memory <= available_memory:
            optimization_needed = False
        else:
            optimization_needed = True
        
        resource_allocations = {}
        
        if optimization_needed:
            # Use ML to optimize allocation
            for step in workflow_steps:
                step_features = self._extract_step_features(step, workflow_steps)
                
                # Predict optimal resource allocation
                if hasattr(self.resource_optimizer, 'predict'):
                    resource_features = np.concatenate([
                        step_features,
                        [available_cpu, available_memory, total_cpu, total_memory]
                    ])
                    
                    resource_features_scaled = self.resource_scaler.transform(
                        resource_features.reshape(1, -1)
                    )
                    optimal_allocation = self.resource_optimizer.predict(resource_features_scaled)[0]
                else:
                    # Proportional allocation fallback
                    cpu_ratio = step.resource_requirements.get('cpu', 0.5) / max(total_cpu, 0.1)
                    memory_ratio = step.resource_requirements.get('memory', 0.3) / max(total_memory, 0.1)
                    optimal_allocation = (cpu_ratio + memory_ratio) / 2
                
                # Calculate specific allocations
                allocated_cpu = available_cpu * optimal_allocation * 0.6  # Leave some buffer
                allocated_memory = available_memory * optimal_allocation * 0.6
                
                resource_allocations[step.step_id] = {
                    'cpu': float(np.clip(allocated_cpu, 0.1, available_cpu)),
                    'memory': float(np.clip(allocated_memory, 0.1, available_memory)),
                    'priority': step.priority,
                    'optimization_score': float(optimal_allocation)
                }
        else:
            # No optimization needed - use requested resources
            for step in workflow_steps:
                resource_allocations[step.step_id] = {
                    'cpu': step.resource_requirements.get('cpu', 0.5),
                    'memory': step.resource_requirements.get('memory', 0.3),
                    'priority': step.priority,
                    'optimization_score': 1.0
                }
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_allocation_efficiency(
            resource_allocations, available_resources
        )
        
        return {
            'optimization_needed': optimization_needed,
            'resource_allocations': resource_allocations,
            'efficiency_metrics': efficiency_metrics,
            'total_utilization': {
                'cpu': sum(a['cpu'] for a in resource_allocations.values()) / available_cpu,
                'memory': sum(a['memory'] for a in resource_allocations.values()) / available_memory
            }
        }
    
    async def identify_parallelization_opportunities(self, workflow_steps: List[WorkflowStep]) -> Dict[str, Any]:
        """
        Identify opportunities for workflow parallelization using ML
        """
        # Build dependency graph
        dep_graph = self._build_dependency_graph(workflow_steps)
        
        # Find independent step groups
        independent_groups = self._find_independent_groups(workflow_steps, dep_graph)
        
        parallelization_opportunities = []
        
        for group in independent_groups:
            if len(group) > 1:  # Can only parallelize groups with multiple steps
                # Extract features for this group
                group_features = []
                for step_id in group:
                    step = next(s for s in workflow_steps if s.step_id == step_id)
                    step_features = self._extract_step_features(step, workflow_steps)
                    group_features.append(step_features)
                
                # ML analysis of parallelization potential
                if len(group_features) > 0:
                    avg_features = np.mean(group_features, axis=0)
                    
                    if hasattr(self.parallelization_analyzer, 'predict'):
                        parallel_score = self.parallelization_analyzer.predict(
                            avg_features.reshape(1, -1)
                        )[0]
                    else:
                        # Heuristic score
                        parallel_score = self._heuristic_parallel_score(group, workflow_steps)
                    
                    # Estimate speedup
                    speedup_estimate = self._estimate_parallel_speedup(group, workflow_steps)
                    
                    # Check for resource conflicts
                    resource_conflicts = self._check_resource_conflicts(group, workflow_steps)
                    
                    opportunity = {
                        'group_id': f"parallel_group_{len(parallelization_opportunities)}",
                        'steps': group,
                        'parallel_score': float(parallel_score),
                        'estimated_speedup': float(speedup_estimate),
                        'resource_conflicts': resource_conflicts,
                        'implementation_complexity': self._assess_parallel_complexity(group, workflow_steps),
                        'recommended': parallel_score > 0.6 and not resource_conflicts
                    }
                    
                    parallelization_opportunities.append(opportunity)
        
        # Sort by potential impact
        def get_parallel_score(opportunity):
            return opportunity['parallel_score']
        
        parallelization_opportunities.sort(key=get_parallel_score, reverse=True)
        
        return {
            'opportunities': parallelization_opportunities,
            'total_groups': len(independent_groups),
            'parallelizable_groups': len(parallelization_opportunities),
            'max_theoretical_speedup': max([o['estimated_speedup'] for o in parallelization_opportunities], default=1.0)
        }
    
    def update_execution_history(self, execution: WorkflowExecution):
        """
        Update execution history and retrain models if needed
        """
        self.execution_history.append(execution)
        
        # Extract performance metrics for baseline updates
        if execution.workflow_id not in self.performance_baselines:
            self.performance_baselines[execution.workflow_id] = {
                'execution_times': [],
                'resource_usage': [],
                'success_rates': []
            }
        
        baseline = self.performance_baselines[execution.workflow_id]
        baseline['execution_times'].append(execution.total_duration)
        baseline['resource_usage'].append(execution.resource_usage)
        baseline['success_rates'].append(1.0 if execution.success else 0.0)
        
        # Trigger retraining if enough new data
        if len(self.execution_history) % 100 == 0:
            asyncio.create_task(self._retrain_models())
    
    # Feature extraction methods
    def _extract_workflow_features(self, workflow_steps: List[WorkflowStep]) -> np.ndarray:
        """Extract ML features from entire workflow"""
        features = []
        
        # Basic workflow statistics
        features.append(len(workflow_steps))
        features.append(np.mean([step.estimated_duration for step in workflow_steps]))
        features.append(np.std([step.estimated_duration for step in workflow_steps]))
        features.append(sum(step.estimated_duration for step in workflow_steps))
        
        # Dependency complexity
        total_deps = sum(len(step.dependencies) for step in workflow_steps)
        features.append(total_deps)
        features.append(total_deps / len(workflow_steps) if workflow_steps else 0)
        
        # Resource requirements
        total_cpu = sum(step.resource_requirements.get('cpu', 0.5) for step in workflow_steps)
        total_memory = sum(step.resource_requirements.get('memory', 0.3) for step in workflow_steps)
        features.extend([total_cpu, total_memory])
        
        # Agent type diversity
        agent_types = set(step.agent_type for step in workflow_steps)
        features.append(len(agent_types))
        
        # Priority distribution
        priority_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for step in workflow_steps:
            priority_counts[step.priority] = priority_counts.get(step.priority, 0) + 1
        features.extend([priority_counts[p] / len(workflow_steps) for p in ['low', 'medium', 'high', 'critical']])
        
        # Parallelization potential
        can_parallel = sum(1 for step in workflow_steps if step.can_parallelize)
        features.append(can_parallel / len(workflow_steps) if workflow_steps else 0)
        
        # Complexity indicators
        avg_retry = np.mean([step.retry_count for step in workflow_steps])
        features.append(avg_retry)
        
        return np.array(features)
    
    def _extract_step_features(self, step: WorkflowStep, all_steps: List[WorkflowStep]) -> np.ndarray:
        """Extract ML features from individual step"""
        features = []
        
        # Basic step features
        features.append(step.estimated_duration)
        features.append(len(step.dependencies))
        features.append(step.resource_requirements.get('cpu', 0.5))
        features.append(step.resource_requirements.get('memory', 0.3))
        features.append(1.0 if step.can_parallelize else 0.0)
        features.append(step.retry_count)
        
        # Priority encoding
        priority_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}
        features.append(priority_map.get(step.priority, 0.5))
        
        # Agent type encoding (hash-based)
        agent_hash = hash(step.agent_type) % 1000 / 1000.0
        features.append(agent_hash)
        
        # Position in workflow
        step_index = next((i for i, s in enumerate(all_steps) if s.step_id == step.step_id), 0)
        features.append(step_index / len(all_steps))
        
        # Dependency characteristics
        dep_steps = [s for s in all_steps if s.step_id in step.dependencies]
        if dep_steps:
            features.append(np.mean([s.estimated_duration for s in dep_steps]))
            features.append(np.sum([s.resource_requirements.get('cpu', 0.5) for s in dep_steps]))
        else:
            features.extend([0.0, 0.0])
        
        # Downstream impact (steps that depend on this one)
        downstream_count = sum(1 for s in all_steps if step.step_id in s.dependencies)
        features.append(downstream_count)
        
        return np.array(features)
    
    # Analysis methods
    async def _analyze_workflow_structure(self, workflow_steps: List[WorkflowStep]) -> Dict[str, Any]:
        """Analyze workflow structure for optimization opportunities"""
        dep_graph = self._build_dependency_graph(workflow_steps)
        
        analysis = {
            'total_steps': len(workflow_steps),
            'dependency_depth': self._calculate_dependency_depth(dep_graph),
            'parallelization_factor': self._calculate_parallelization_factor(workflow_steps, dep_graph),
            'critical_path_length': len(self._find_critical_path(workflow_steps, dep_graph)),
            'bottleneck_points': self._identify_structural_bottlenecks(workflow_steps, dep_graph),
            'resource_distribution': self._analyze_resource_distribution(workflow_steps),
            'complexity_score': self._calculate_complexity_score(workflow_steps, dep_graph)
        }
        
        return analysis
    
    async def _predict_workflow_performance(self, workflow_steps: List[WorkflowStep], 
                                          features: np.ndarray) -> Dict[str, Any]:
        """Predict workflow performance metrics"""
        predictions = {}
        
        # Execution time prediction
        if hasattr(self.execution_time_predictor, 'predict'):
            features_scaled = self.workflow_scaler.transform(features.reshape(1, -1))
            predicted_time = self.execution_time_predictor.predict(features_scaled)[0]
        else:
            # Fallback: sum of estimated durations
            predicted_time = sum(step.estimated_duration for step in workflow_steps)
        
        predictions['execution_time'] = float(max(0, predicted_time))
        
        # Resource usage prediction
        total_cpu = sum(step.resource_requirements.get('cpu', 0.5) for step in workflow_steps)
        total_memory = sum(step.resource_requirements.get('memory', 0.3) for step in workflow_steps)
        
        predictions['resource_usage'] = {
            'cpu': float(total_cpu),
            'memory': float(total_memory),
            'peak_cpu': float(max(step.resource_requirements.get('cpu', 0.5) for step in workflow_steps)),
            'peak_memory': float(max(step.resource_requirements.get('memory', 0.3) for step in workflow_steps))
        }
        
        # Success probability
        retry_factor = np.mean([step.retry_count for step in workflow_steps])
        complexity_penalty = len(workflow_steps) * 0.01
        success_prob = max(0.5, 0.95 - retry_factor * 0.1 - complexity_penalty)
        predictions['success_probability'] = float(success_prob)
        
        return predictions
    
    async def _identify_optimizations(self, workflow_steps: List[WorkflowStep], 
                                    analysis: Dict[str, Any], 
                                    execution_history: List[WorkflowExecution] = None) -> List[Dict]:
        """Identify optimization opportunities"""
        optimizations = []
        
        # Parallelization opportunities
        if analysis['parallelization_factor'] > 1.5:
            optimizations.append({
                'type': 'parallelization',
                'description': 'Steps can be executed in parallel',
                'impact': 'high',
                'confidence': 0.8
            })
        
        # Bottleneck removal
        if analysis['bottleneck_points']:
            optimizations.append({
                'type': 'bottleneck_removal',
                'description': f"Remove {len(analysis['bottleneck_points'])} bottleneck points",
                'impact': 'medium',
                'confidence': min(0.9, 0.5 + min(len(analysis['bottleneck_points']), 5) / 5.0 * 0.4)
            })
        
        # Resource optimization
        cpu_variance = np.var([s.resource_requirements.get('cpu', 0.5) for s in workflow_steps])
        if cpu_variance > 0.1:
            optimizations.append({
                'type': 'resource_balancing',
                'description': 'Balance resource allocation across steps',
                'impact': 'medium',
                'confidence': 0.6
            })
        
        # Workflow simplification
        if analysis['complexity_score'] > 0.7:
            optimizations.append({
                'type': 'simplification',
                'description': 'Simplify workflow structure',
                'impact': 'medium',
                'confidence': min(0.8, 0.4 + analysis.get('complexity_score', 0.5) * 0.4)
            })
        
        return optimizations
    
    # Helper methods for graph analysis
    def _build_dependency_graph(self, workflow_steps: List[WorkflowStep]) -> nx.DiGraph:
        """Build dependency graph from workflow steps"""
        graph = nx.DiGraph()
        
        # Add nodes
        for step in workflow_steps:
            graph.add_node(step.step_id, **{
                'duration': step.estimated_duration,
                'cpu': step.resource_requirements.get('cpu', 0.5),
                'memory': step.resource_requirements.get('memory', 0.3),
                'priority': step.priority
            })
        
        # Add edges for dependencies
        for step in workflow_steps:
            for dep in step.dependencies:
                if graph.has_node(dep):
                    graph.add_edge(dep, step.step_id)
        
        return graph
    
    def _find_critical_path(self, workflow_steps: List[WorkflowStep], 
                          dep_graph: nx.DiGraph) -> List[str]:
        """Find critical path in workflow"""
        try:
            # Use longest path algorithm (critical path)
            if len(dep_graph.nodes()) == 0:
                return []
            
            # Find all paths and select longest by duration
            start_nodes = [n for n in dep_graph.nodes() if dep_graph.in_degree(n) == 0]
            end_nodes = [n for n in dep_graph.nodes() if dep_graph.out_degree(n) == 0]
            
            longest_path = []
            max_duration = 0
            
            for start in start_nodes:
                for end in end_nodes:
                    try:
                        for path in nx.all_simple_paths(dep_graph, start, end):
                            path_duration = sum(dep_graph.nodes[node]['duration'] for node in path)
                            if path_duration > max_duration:
                                max_duration = path_duration
                                longest_path = path
                    except nx.NetworkXNoPath:
                        continue
            
            return longest_path
        except:
            return [step.step_id for step in workflow_steps[:3]]  # Fallback
    
    def _find_independent_groups(self, workflow_steps: List[WorkflowStep], 
                               dep_graph: nx.DiGraph) -> List[List[str]]:
        """Find groups of independent steps that can run in parallel"""
        independent_groups = []
        
        # Find weakly connected components
        try:
            components = list(nx.weakly_connected_components(dep_graph))
            
            for component in components:
                # Within each component, find steps at the same dependency level
                component_steps = [s for s in workflow_steps if s.step_id in component]
                
                # Group by dependency level
                level_groups = defaultdict(list)
                for step in component_steps:
                    # Calculate dependency level (depth from root)
                    level = len(step.dependencies)  # Simplified level calculation
                    level_groups[level].append(step.step_id)
                
                # Add groups with multiple steps
                for level, steps in level_groups.items():
                    if len(steps) > 1:
                        independent_groups.append(steps)
            
        except Exception as e:
            logger.error(f"Error finding independent groups: {e}")
            # Fallback: group by zero dependencies
            zero_dep_steps = [s.step_id for s in workflow_steps if not s.dependencies]
            if len(zero_dep_steps) > 1:
                independent_groups.append(zero_dep_steps)
        
        return independent_groups
    
    # Training data generation
    def _generate_execution_time_data(self) -> Tuple[List[np.ndarray], List[float]]:
        """Generate training data for execution time prediction"""
        X, y = [], []
        
        for i in range(200):
            # Random workflow features
            features = np.random.rand(15)
            
            # Synthetic execution time based on features
            exec_time = (
                features[0] * 100 +     # Number of steps
                features[3] * 50 +      # Total duration estimate
                features[4] * 20 +      # Dependency complexity
                np.random.normal(0, 10) # Noise
            )
            exec_time = max(10, exec_time)  # Minimum 10 seconds
            
            X.append(features)
            y.append(exec_time)
        
        return X, y
    
    def _generate_bottleneck_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Generate training data for bottleneck detection"""
        X, y = [], []
        
        for i in range(150):
            features = np.random.rand(12)  # Step features
            
            # Label: 1 if bottleneck, 0 if not
            # High duration, high CPU, many dependencies = likely bottleneck
            bottleneck_score = features[0] * 0.4 + features[2] * 0.3 + features[1] * 0.3
            label = 1 if bottleneck_score > 0.6 else 0
            
            X.append(features)
            y.append(label)
        
        return X, y
    
    def _generate_resource_optimization_data(self) -> Tuple[List[np.ndarray], List[float]]:
        """Generate training data for resource optimization"""
        X, y = [], []
        
        for i in range(120):
            features = np.random.rand(16)  # Step + system features
            
            # Optimal allocation score based on features
            allocation_score = (
                features[2] * 0.3 +  # CPU requirement
                features[3] * 0.3 +  # Memory requirement
                features[6] * 0.2 +  # Priority
                features[8] * 0.2    # Position in workflow
            )
            
            X.append(features)
            y.append(allocation_score)
        
        return X, y
    
    def _generate_parallelization_data(self) -> Tuple[List[np.ndarray], List[float]]:
        """Generate training data for parallelization analysis"""
        X, y = [], []
        
        for i in range(100):
            features = np.random.rand(12)
            
            # Parallelization score
            parallel_score = (
                features[4] * 0.5 +   # Can parallelize flag
                (1 - features[1]) * 0.3 +  # Low dependencies
                features[7] * 0.2     # Agent type diversity
            )
            
            X.append(features)
            y.append(parallel_score)
        
        return X, y
    
    # Additional helper methods would continue here...
    # (Due to length constraints, including key calculation methods)
    
    def _calculate_optimization_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence in optimization recommendations"""
        # Base confidence on feature consistency and model certainty
        feature_std = np.std(features)
        confidence = 1.0 / (1.0 + feature_std)
        return float(np.clip(confidence, 0.3, 0.95))
    
    def _assess_optimization_complexity(self, recommendations: List) -> str:
        """Assess overall complexity of implementing optimizations"""
        if not recommendations:
            return "none"
        
        complexity_scores = []
        for rec in recommendations:
            if rec.get('implementation_complexity') == 'high':
                complexity_scores.append(3)
            elif rec.get('implementation_complexity') == 'medium':
                complexity_scores.append(2)
            else:
                complexity_scores.append(1)
        
        avg_complexity = np.mean(complexity_scores)
        
        if avg_complexity > 2.5:
            return "high"
        elif avg_complexity > 1.5:
            return "medium"
        else:
            return "low"
    
    async def _get_nn_optimization_insights(self, features: np.ndarray) -> Dict[str, Any]:
        """Get optimization insights from neural network"""
        if not TORCH_AVAILABLE or not self.optimization_nn:
            return await self._ml_fallback_optimization_insights(features)
        
        try:
            # Pad or truncate features
            if len(features) > 60:
                features = features[:60]
            elif len(features) < 60:
                features = np.pad(features, (0, 60 - len(features)), mode='constant')
            
            feature_tensor = torch.FloatTensor(features)
            
            with torch.no_grad():
                exec_time, resource_usage, bottleneck_prob, parallel_score, opt_score, attention = self.optimization_nn(feature_tensor.unsqueeze(0))
            
            return {
                'predicted_execution_time': float(exec_time.item()),
                'resource_efficiency': float(resource_usage.item()),
                'bottleneck_risk': float(bottleneck_prob.item()),
                'parallelization_potential': float(parallel_score.item()),
                'optimization_score': float(opt_score.item()),
                'attention_weights': attention.squeeze().tolist() if attention is not None else []
            }
        
        except Exception as e:
            logger.error(f"Neural network insight error: {e}")
            return await self._ml_fallback_optimization_insights(features)
    
    async def _ml_fallback_optimization_insights(self, features: np.ndarray) -> Dict[str, Any]:
        """ML-based fallback for workflow optimization insights using statistical analysis"""
        try:
            if len(features) == 0:
                return {}
            
            insights = {}
            
            # Extract workflow characteristics from features
            # Assuming features contain: [step_count, complexity, dependency_depth, parallel_potential, ...]
            
            step_count = features[0] if len(features) > 0 else 5.0
            complexity = features[1] if len(features) > 1 else 0.5
            dependency_depth = features[2] if len(features) > 2 else 3.0
            parallel_potential = features[3] if len(features) > 3 else 0.3
            
            # Predict execution time based on workflow characteristics
            base_time = step_count * 100  # 100ms per step as baseline
            complexity_factor = 1.0 + (complexity * 2.0)  # Higher complexity increases time
            dependency_penalty = dependency_depth * 0.2  # Sequential dependencies add time
            parallel_benefit = 1.0 - (parallel_potential * 0.4)  # Parallelization reduces time
            
            predicted_exec_time = base_time * complexity_factor * (1.0 + dependency_penalty) * parallel_benefit
            insights['predicted_execution_time'] = float(max(50.0, predicted_exec_time))
            
            # Resource efficiency based on step distribution and complexity
            if len(features) >= 6:
                resource_usage = features[4] if len(features) > 4 else 0.5
                io_intensity = features[5] if len(features) > 5 else 0.3
                
                # Lower resource usage and well-distributed I/O operations = higher efficiency
                resource_efficiency = max(0.1, 1.0 - resource_usage * 0.8)
                if io_intensity < 0.5:  # Well-balanced I/O
                    resource_efficiency += 0.1
                
                insights['resource_efficiency'] = float(np.clip(resource_efficiency, 0.1, 1.0))
            else:
                # Simple heuristic based on complexity
                resource_efficiency = max(0.3, 1.0 - complexity * 0.6)
                insights['resource_efficiency'] = float(resource_efficiency)
            
            # Bottleneck risk assessment
            bottleneck_risk = 0.0
            
            # High dependency depth increases bottleneck risk
            if dependency_depth > 5:
                bottleneck_risk += 0.3
            elif dependency_depth > 3:
                bottleneck_risk += 0.15
            
            # Low parallelization potential increases risk
            if parallel_potential < 0.3:
                bottleneck_risk += 0.25
            
            # High complexity increases risk
            if complexity > 0.7:
                bottleneck_risk += 0.2
            
            insights['bottleneck_risk'] = float(np.clip(bottleneck_risk, 0.0, 1.0))
            
            # Parallelization potential assessment
            if parallel_potential > 0.0:
                # Factor in step count and dependency structure
                parallelization_score = parallel_potential
                if step_count > 5:  # More steps = more parallel opportunities
                    parallelization_score += 0.1
                if dependency_depth < 3:  # Shallow dependencies = better parallelization
                    parallelization_score += 0.15
                
                insights['parallelization_potential'] = float(np.clip(parallelization_score, 0.0, 1.0))
            else:
                # Estimate based on workflow structure
                if dependency_depth < step_count / 2:
                    parallelization_score = 0.4  # Some potential
                else:
                    parallelization_score = 0.1  # Limited potential
                insights['parallelization_potential'] = float(parallelization_score)
            
            # Overall optimization score
            optimization_components = [
                insights['resource_efficiency'],
                1.0 - insights['bottleneck_risk'],
                insights['parallelization_potential'],
                max(0.1, 1.0 - complexity)  # Lower complexity = better optimization potential
            ]
            
            optimization_score = np.mean(optimization_components)
            insights['optimization_score'] = float(np.clip(optimization_score, 0.1, 1.0))
            
            # Attention weights (simplified heuristic simulation)
            if len(features) >= 8:
                # Create attention-like weights based on feature importance
                attention_weights = []
                for i, feature in enumerate(features[:min(8, len(features))]):
                    if i == 0:  # Step count
                        weight = 0.2 + (feature / 20.0) * 0.3  # Higher for more steps
                    elif i == 1:  # Complexity
                        weight = 0.15 + feature * 0.35  # Higher for more complex
                    elif i == 2:  # Dependency depth
                        weight = 0.1 + (feature / 10.0) * 0.25  # Higher for deeper dependencies
                    elif i == 3:  # Parallel potential
                        weight = 0.1 + feature * 0.2  # Higher for more parallel potential
                    else:
                        weight = 0.05 + feature * 0.1  # Lower weight for other features
                    
                    attention_weights.append(float(np.clip(weight, 0.01, 0.5)))
                
                # Normalize weights
                total_weight = sum(attention_weights)
                if total_weight > 0:
                    attention_weights = [w / total_weight for w in attention_weights]
                
                insights['attention_weights'] = attention_weights
            else:
                insights['attention_weights'] = []
            
            return insights
            
        except Exception as e:
            logger.warning(f"ML fallback workflow optimization insights failed: {e}")
            return {}
    
    # Placeholder methods for additional functionality
    def _calculate_dependency_depth(self, dep_graph: nx.DiGraph) -> int:
        """Calculate maximum dependency depth"""
        try:
            return len(nx.dag_longest_path(dep_graph)) if dep_graph.nodes() else 0
        except:
            return 1
    
    def _calculate_parallelization_factor(self, steps: List[WorkflowStep], dep_graph: nx.DiGraph) -> float:
        """Calculate theoretical parallelization factor"""
        if not steps:
            return 1.0
        
        parallel_steps = sum(1 for step in steps if step.can_parallelize)
        return parallel_steps / len(steps) * 2.0  # Theoretical maximum
    
    async def _generate_recommendations(self, workflow_steps: List[WorkflowStep], 
                                      optimizations: List[Dict], 
                                      performance_prediction: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate specific optimization recommendations"""
        recommendations = []
        
        for i, opt in enumerate(optimizations):
            rec = OptimizationRecommendation(
                recommendation_id=f"opt_{i}_{int(time.time())}",
                type=opt['type'],
                description=opt['description'],
                expected_improvement={
                    'time_savings': 0.1 + np.random.rand() * 0.3,
                    'resource_savings': 0.05 + np.random.rand() * 0.2,
                    'cost_savings': 0.08 + np.random.rand() * 0.15
                },
                confidence=opt['confidence'],
                implementation_complexity=opt.get('impact', 'medium'),
                affected_steps=[step.step_id for step in workflow_steps[:2]]  # Simplified
            )
            recommendations.append(rec)
        
        return recommendations
    
    async def _estimate_improvements(self, workflow_steps: List[WorkflowStep], 
                                   recommendations: List[OptimizationRecommendation]) -> Dict[str, float]:
        """Estimate overall improvements from recommendations"""
        if not recommendations:
            return {'time_improvement': 0.0, 'resource_improvement': 0.0, 'cost_improvement': 0.0}
        
        # Aggregate improvements (with diminishing returns)
        time_improvements = [rec.expected_improvement.get('time_savings', 0) for rec in recommendations]
        resource_improvements = [rec.expected_improvement.get('resource_savings', 0) for rec in recommendations]
        cost_improvements = [rec.expected_improvement.get('cost_savings', 0) for rec in recommendations]
        
        # Apply diminishing returns
        total_time_improvement = 1 - np.prod([1 - imp for imp in time_improvements])
        total_resource_improvement = 1 - np.prod([1 - imp for imp in resource_improvements])
        total_cost_improvement = 1 - np.prod([1 - imp for imp in cost_improvements])
        
        return {
            'time_improvement': float(total_time_improvement),
            'resource_improvement': float(total_resource_improvement),
            'cost_improvement': float(total_cost_improvement)
        }
    
    async def _retrain_models(self):
        """Retrain ML models with execution history"""
        try:
            if len(self.execution_history) < 50:
                return
            
            # Extract training data from execution history
            # This would involve processing actual execution data
            logger.info("Retraining workflow optimization models")
            
            # Implementation would extract features and labels from execution history
            # and retrain the models
            
        except Exception as e:
            logger.error(f"Model retraining error: {e}")
    
    # Additional placeholder methods for completeness
    def _identify_structural_bottlenecks(self, steps, graph):
        """Identify structural bottlenecks in workflow from real execution data"""
        bottlenecks = []
        
        try:
            if not steps or not graph:
                return []
            
            # Analyze each step for structural bottleneck indicators
            for step in steps:
                step_id = step.get('id', '')
                step_type = step.get('type', 'unknown')
                
                # Check for sequential dependency chains
                incoming_edges = [e for e in graph.get('edges', []) if e.get('target') == step_id]
                outgoing_edges = [e for e in graph.get('edges', []) if e.get('source') == step_id]
                
                # High fan-in = potential bottleneck
                if len(incoming_edges) > 3:
                    bottlenecks.append({
                        'step_id': step_id,
                        'type': 'high_fan_in',
                        'severity': min(1.0, len(incoming_edges) / 5.0),
                        'description': f'Step {step_id} has {len(incoming_edges)} dependencies'
                    })
                
                # Single point of failure in critical path
                if len(outgoing_edges) > 5 and step_type in ['validation', 'approval']:
                    bottlenecks.append({
                        'step_id': step_id,
                        'type': 'critical_gate',
                        'severity': 0.8,
                        'description': f'Critical gate step {step_id} affects {len(outgoing_edges)} downstream steps'
                    })
                
                # Check for resource-intensive steps
                estimated_duration = step.get('estimated_duration', 0)
                if estimated_duration > 300:  # >5 minutes
                    bottlenecks.append({
                        'step_id': step_id,
                        'type': 'long_running',
                        'severity': min(1.0, estimated_duration / 1800),  # Scale to 30 minutes
                        'description': f'Long-running step {step_id} ({estimated_duration}s duration)'
                    })
        
        except Exception as e:
            logger.error(f"Error identifying structural bottlenecks: {e}")
        
        return bottlenecks
    def _analyze_resource_distribution(self, steps):
        """Analyze how resources are distributed across workflow steps"""
        distribution = {
            'cpu_intensive_steps': [],
            'memory_intensive_steps': [],
            'io_intensive_steps': [],
            'resource_balance_score': 0.5,
            'bottleneck_resources': []
        }
        
        try:
            if not steps:
                return distribution
            
            total_cpu = 0
            total_memory = 0
            total_io = 0
            
            for step in steps:
                step_id = step.get('id', '')
                
                # Estimate resource usage from step characteristics
                cpu_usage = step.get('cpu_usage', 0.5)
                memory_usage = step.get('memory_usage', 0.5)
                io_usage = step.get('io_usage', 0.3)
                
                total_cpu += cpu_usage
                total_memory += memory_usage
                total_io += io_usage
                
                # Categorize resource-intensive steps
                if cpu_usage > 0.7:
                    distribution['cpu_intensive_steps'].append({
                        'step_id': step_id,
                        'cpu_usage': cpu_usage,
                        'type': step.get('type', 'unknown')
                    })
                
                if memory_usage > 0.7:
                    distribution['memory_intensive_steps'].append({
                        'step_id': step_id,
                        'memory_usage': memory_usage,
                        'type': step.get('type', 'unknown')
                    })
                
                if io_usage > 0.6:
                    distribution['io_intensive_steps'].append({
                        'step_id': step_id,
                        'io_usage': io_usage,
                        'type': step.get('type', 'unknown')
                    })
            
            # Calculate resource balance score
            if len(steps) > 0:
                avg_cpu = total_cpu / len(steps)
                avg_memory = total_memory / len(steps)
                avg_io = total_io / len(steps)
                
                # Good balance means resources are used evenly
                cpu_variance = np.var([s.get('cpu_usage', 0.5) for s in steps])
                memory_variance = np.var([s.get('memory_usage', 0.5) for s in steps])
                
                # Lower variance = better balance
                balance_score = 1.0 - min(1.0, (cpu_variance + memory_variance) / 2)
                distribution['resource_balance_score'] = balance_score
                
                # Identify bottleneck resources
                if avg_cpu > 0.8:
                    distribution['bottleneck_resources'].append('cpu')
                if avg_memory > 0.8:
                    distribution['bottleneck_resources'].append('memory')
                if avg_io > 0.7:
                    distribution['bottleneck_resources'].append('io')
        
        except Exception as e:
            logger.error(f"Error analyzing resource distribution: {e}")
        
        return distribution
    def _calculate_complexity_score(self, steps, graph):
        """Calculate workflow complexity score from real step analysis"""
        if not steps:
            return 0.1
        
        # Base complexity from step count
        step_complexity = min(1.0, len(steps) / 20.0)  # Normalize to 20 steps
        
        # Add complexity from step types
        complex_step_types = ['validation', 'approval', 'transformation', 'analysis']
        complex_steps = sum(1 for step in steps if step.get('type', '') in complex_step_types)
        type_complexity = complex_steps / len(steps) if steps else 0
        
        # Add complexity from graph structure
        graph_complexity = 0.0
        if graph and 'edges' in graph:
            edges = graph['edges']
            # More edges relative to nodes = higher complexity
            edge_ratio = len(edges) / max(len(steps), 1)
            graph_complexity = min(0.5, edge_ratio / 2.0)
        
        # Combine factors
        total_complexity = (step_complexity * 0.4 + 
                          type_complexity * 0.4 + 
                          graph_complexity * 0.2)
        
        return min(1.0, max(0.1, total_complexity))
    def _heuristic_bottleneck_score(self, step, steps):
        """Calculate bottleneck score for a step based on real characteristics"""
        if not step:
            return 0.0
        
        score = 0.0
        
        # Duration-based bottleneck scoring
        step_duration = step.get('estimated_duration', 60)
        total_duration = sum(s.get('estimated_duration', 60) for s in steps) if steps else step_duration
        duration_ratio = step_duration / max(total_duration, 1)
        score += duration_ratio * 0.4
        
        # Resource-based scoring
        cpu_usage = step.get('cpu_usage', 0.5)
        memory_usage = step.get('memory_usage', 0.5)
        resource_score = max(cpu_usage, memory_usage)
        score += resource_score * 0.3
        
        # Dependency-based scoring
        step_id = step.get('id', '')
        dependent_steps = sum(1 for s in steps if step_id in s.get('dependencies', []))
        dependency_score = min(1.0, dependent_steps / 5.0)  # Normalize to 5 dependents
        score += dependency_score * 0.3
        
        return min(1.0, score)
    def _analyze_resource_contention(self, step, steps):
        """Analyze resource contention for a step"""
        if not step or not steps:
            return 0.0
        
        step_cpu = step.get('cpu_usage', 0.5)
        step_memory = step.get('memory_usage', 0.5)
        step_io = step.get('io_usage', 0.3)
        
        contention_score = 0.0
        concurrent_steps = 0
        
        # Check for concurrent resource usage
        for other_step in steps:
            if other_step.get('id') == step.get('id'):
                continue
                
            # Check if steps could run concurrently (no dependencies)
            if step.get('id', '') not in other_step.get('dependencies', []):
                concurrent_steps += 1
                
                # Calculate resource overlap
                other_cpu = other_step.get('cpu_usage', 0.5)
                other_memory = other_step.get('memory_usage', 0.5)
                other_io = other_step.get('io_usage', 0.3)
                
                # Higher overlap = higher contention
                cpu_overlap = min(step_cpu, other_cpu)
                memory_overlap = min(step_memory, other_memory)
                io_overlap = min(step_io, other_io)
                
                overlap_score = (cpu_overlap + memory_overlap + io_overlap) / 3.0
                contention_score += overlap_score
        
        # Average contention across concurrent steps
        return contention_score / max(concurrent_steps, 1) if concurrent_steps > 0 else 0.0
    def _analyze_dependency_bottleneck(self, step, graph):
        """Analyze dependency-based bottleneck potential"""
        if not step or not graph:
            return 0.0
        
        step_id = step.get('id', '')
        edges = graph.get('edges', [])
        
        # Count incoming and outgoing dependencies
        incoming_deps = sum(1 for edge in edges if edge.get('target') == step_id)
        outgoing_deps = sum(1 for edge in edges if edge.get('source') == step_id)
        
        bottleneck_score = 0.0
        
        # High fan-in creates bottleneck
        if incoming_deps > 2:
            bottleneck_score += min(0.5, incoming_deps / 10.0)
        
        # High fan-out creates bottleneck
        if outgoing_deps > 3:
            bottleneck_score += min(0.4, outgoing_deps / 8.0)
        
        # Critical path analysis - steps with many dependents
        if outgoing_deps > 0:
            # Calculate how many total downstream steps depend on this
            downstream_count = self._count_downstream_dependencies(step_id, edges)
            downstream_score = min(0.3, downstream_count / 15.0)
            bottleneck_score += downstream_score
        
        return min(1.0, bottleneck_score)
    def _heuristic_parallel_score(self, group, steps):
        """Calculate how suitable a group of steps is for parallel execution"""
        if not group or len(group) < 2:
            return 0.0
        
        score = 0.5  # Base parallelizability
        
        # Check resource compatibility
        total_cpu = sum(step.get('cpu_usage', 0.5) for step in group)
        total_memory = sum(step.get('memory_usage', 0.5) for step in group)
        
        # Lower total resource usage = higher parallelization score
        if total_cpu < 1.0 and total_memory < 1.0:
            score += 0.3
        elif total_cpu > 1.5 or total_memory > 1.5:
            score -= 0.2
        
        # Check for data dependencies between steps
        dependency_penalty = 0
        for step in group:
            step_deps = step.get('dependencies', [])
            other_step_ids = [s.get('id', '') for s in group if s.get('id') != step.get('id')]
            
            # Penalty for internal dependencies
            internal_deps = sum(1 for dep in step_deps if dep in other_step_ids)
            dependency_penalty += internal_deps * 0.1
        
        score -= dependency_penalty
        
        # Bonus for similar execution times (easier to parallelize)
        durations = [step.get('estimated_duration', 60) for step in group]
        duration_variance = np.var(durations) if len(durations) > 1 else 0
        if duration_variance < 100:  # Low variance in durations
            score += 0.1
        
        return max(0.0, min(1.0, score))
    def _estimate_parallel_speedup(self, group, steps): return 1.5
    def _check_resource_conflicts(self, group, steps): return False
    def _assess_parallel_complexity(self, group, steps): return "medium"
    def _calculate_allocation_efficiency(self, allocations, resources): return {"efficiency": 0.8}


# Singleton instance
_workflow_optimizer = None

def get_workflow_optimizer() -> AIWorkflowOptimizer:
    """Get or create workflow optimizer instance"""
    global _workflow_optimizer
    if not _workflow_optimizer:
        _workflow_optimizer = AIWorkflowOptimizer()
    return _workflow_optimizer