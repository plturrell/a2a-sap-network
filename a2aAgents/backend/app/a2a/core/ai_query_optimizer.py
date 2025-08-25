"""
AI-Powered Database Query Optimization and Intelligence System

This module provides intelligent database query optimization, execution planning,
index suggestions, and performance analysis using real machine learning
for enhanced database performance without relying on external services.
"""

import asyncio
import logging
import numpy as np
import json
import time
import hashlib
import threading
import re
import sqlparse
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
from enum import Enum

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, ExtraTreesRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Advanced pattern recognition
try:
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR, SVC
    SKLEARN_EXTENDED = True
except ImportError:
    SKLEARN_EXTENDED = False

# Deep learning for complex query analysis
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Advanced optimization algorithms
try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.stats import entropy, pearsonr
    import scipy.sparse as sparse
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class QueryOptimizationNN(nn.Module):
    """Neural network for advanced query optimization and performance prediction"""
    def __init__(self, vocab_size=5000, embedding_dim=128, hidden_dim=256, sequence_length=100):
        super(QueryOptimizationNN, self).__init__()
        
        # SQL token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(sequence_length, embedding_dim)
        
        # Query structure encoder
        self.query_encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                                   num_layers=2, dropout=0.2, bidirectional=True)
        
        # Attention mechanism for important query parts
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=0.1)
        
        # Schema-aware feature processing
        self.schema_processor = nn.Sequential(
            nn.Linear(64, hidden_dim),  # Schema features
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        
        # Multi-modal feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # Query + Schema + Context
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2)
        )
        
        # Multi-task prediction heads
        self.execution_time_head = nn.Linear(hidden_dim // 2, 1)      # Execution time prediction
        self.cost_prediction_head = nn.Linear(hidden_dim // 2, 1)     # Query cost estimation
        self.optimization_head = nn.Linear(hidden_dim // 2, 10)       # Optimization strategies
        self.index_suggestion_head = nn.Linear(hidden_dim // 2, 20)   # Index recommendations
        self.bottleneck_head = nn.Linear(hidden_dim // 2, 8)          # Bottleneck identification
        self.cache_worthiness_head = nn.Linear(hidden_dim // 2, 1)    # Cache recommendation
        self.parallelization_head = nn.Linear(hidden_dim // 2, 4)     # Parallelization potential
        
        self.dropout = nn.Dropout(0.25)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
    
    def forward(self, query_tokens, schema_features, context_features):
        batch_size, seq_len = query_tokens.size()
        
        # Create position indices
        positions = torch.arange(seq_len, device=query_tokens.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens and positions
        token_embedded = self.token_embedding(query_tokens)
        position_embedded = self.position_embedding(positions)
        embedded = token_embedded + position_embedded
        
        # Encode query structure
        encoded, (hidden, cell) = self.query_encoder(embedded)
        encoded = self.layer_norm(encoded)
        
        # Apply attention
        attn_out, attn_weights = self.attention(encoded, encoded, encoded)
        query_representation = attn_out.mean(1)  # Average pooling
        
        # Process schema features
        schema_repr = self.schema_processor(schema_features)
        
        # Combine all features
        combined_features = torch.cat([query_representation, schema_repr, context_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        fused_features = self.dropout(fused_features)
        
        # Multi-task predictions
        execution_time = F.relu(self.execution_time_head(fused_features))
        cost_prediction = F.relu(self.cost_prediction_head(fused_features))
        optimization_probs = F.softmax(self.optimization_head(fused_features), dim=-1)
        index_suggestions = torch.sigmoid(self.index_suggestion_head(fused_features))
        bottleneck_probs = F.softmax(self.bottleneck_head(fused_features), dim=-1)
        cache_worthiness = torch.sigmoid(self.cache_worthiness_head(fused_features))
        parallelization_probs = F.softmax(self.parallelization_head(fused_features), dim=-1)
        
        return {
            'execution_time': execution_time,
            'cost_prediction': cost_prediction,
            'optimization_strategies': optimization_probs,
            'index_suggestions': index_suggestions,
            'bottlenecks': bottleneck_probs,
            'cache_worthiness': cache_worthiness,
            'parallelization': parallelization_probs,
            'attention_weights': attn_weights,
            'query_representation': query_representation
        }


@dataclass
class QueryMetrics:
    """Query execution metrics and metadata"""
    query_id: str
    sql_query: str
    execution_time: float
    rows_examined: int
    rows_returned: int
    memory_used: int
    cpu_usage: float
    io_operations: int
    cache_hits: int
    cache_misses: int
    index_usage: List[str] = field(default_factory=list)
    table_scans: int = 0
    join_operations: int = 0
    sorting_operations: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OptimizationRecommendation:
    """Query optimization recommendation"""
    query_id: str
    optimization_type: str
    description: str
    estimated_improvement: float
    confidence_score: float
    implementation_difficulty: str  # easy, medium, hard
    sql_rewrite: Optional[str] = None
    index_suggestions: List[str] = field(default_factory=list)
    configuration_changes: Dict[str, Any] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)


@dataclass
class QueryPattern:
    """Identified query pattern"""
    pattern_id: str
    pattern_type: str
    frequency: int
    avg_execution_time: float
    tables_involved: List[str]
    columns_accessed: List[str]
    join_patterns: List[str] = field(default_factory=list)
    filter_patterns: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)


class QueryType(Enum):
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"
    UNKNOWN = "unknown"


class OptimizationType(Enum):
    INDEX_CREATION = "index_creation"
    QUERY_REWRITE = "query_rewrite"
    JOIN_OPTIMIZATION = "join_optimization"
    FILTER_PUSHDOWN = "filter_pushdown"
    SUBQUERY_OPTIMIZATION = "subquery_optimization"
    CACHING = "caching"
    PARTITIONING = "partitioning"
    PARALLELIZATION = "parallelization"


class AIQueryOptimizer:
    """
    AI-powered database query optimization system using real ML models
    """
    
    def __init__(self):
        # ML Models for query optimization
        self.execution_time_predictor = RandomForestRegressor(n_estimators=150, random_state=42)
        self.cost_estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.optimization_classifier = ExtraTreesRegressor(n_estimators=100, random_state=42)
        self.bottleneck_detector = MLPClassifier(hidden_layer_sizes=(128, 64), random_state=42)
        self.index_recommender = MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
        
        # Clustering for query pattern analysis
        self.query_clusterer = KMeans(n_clusters=12, random_state=42)
        self.pattern_detector = DBSCAN(eps=0.4, min_samples=3)
        self.anomaly_detector = self._initialize_anomaly_detector()
        
        # Feature extractors and processors
        self.query_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), 
                                              lowercase=True, stop_words='english')
        self.token_vectorizer = HashingVectorizer(n_features=2000, ngram_range=(1, 1))
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Neural network for advanced optimization
        if TORCH_AVAILABLE:
            self.optimization_nn = QueryOptimizationNN()
            self.nn_optimizer = torch.optim.AdamW(self.optimization_nn.parameters(), lr=0.001)
        else:
            self.optimization_nn = None
        
        # Query analysis and storage
        self.query_history = deque(maxlen=5000)
        self.query_patterns = {}
        self.optimization_cache = {}
        self.performance_metrics = defaultdict(list)
        
        # Database schema information
        self.schema_info = defaultdict(dict)
        self.table_statistics = defaultdict(dict)
        self.index_information = defaultdict(list)
        
        # Query processing components
        self.sql_parser = None  # Will be initialized if sqlparse is available
        self.query_tokenizer = self._initialize_tokenizer()
        
        # Optimization strategies and rules
        self.optimization_rules = self._initialize_optimization_rules()
        self.index_strategies = self._initialize_index_strategies()
        
        # Real-time monitoring
        self.monitoring_metrics = defaultdict(deque)
        self.performance_alerts = []
        
        # Statistics and feedback
        self.optimization_stats = defaultdict(int)
        self.feedback_history = deque(maxlen=500)
        
        # Initialize models with training data
        self._initialize_models()
        
        logger.info("AI Query Optimizer initialized with ML models")
    
    async def optimize_query(self, sql_query: str, 
                           context: Dict[str, Any] = None,
                           schema_info: Dict[str, Any] = None) -> List[OptimizationRecommendation]:
        """
        Optimize a SQL query using AI-powered analysis
        """
        try:
            context = context or {}
            schema_info = schema_info or {}
            
            query_id = hashlib.md5(sql_query.encode()).hexdigest()
            
            # Check optimization cache
            cache_key = f"{query_id}_{hash(str(context))}"
            if cache_key in self.optimization_cache:
                cached_result = self.optimization_cache[cache_key]
                if (datetime.utcnow() - cached_result['timestamp']).seconds < 3600:  # 1 hour cache
                    return cached_result['recommendations']
            
            # Analyze query structure and features
            query_features = self._extract_query_features(sql_query, context, schema_info)
            query_type = self._identify_query_type(sql_query)
            
            # Predict performance metrics
            predicted_performance = self._predict_query_performance(query_features, query_type)
            
            # Identify optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(
                sql_query, query_features, predicted_performance
            )
            
            # Generate specific recommendations
            recommendations = []
            
            for opportunity in optimization_opportunities:
                recommendation = await self._generate_optimization_recommendation(
                    query_id, sql_query, opportunity, query_features, schema_info
                )
                if recommendation:
                    recommendations.append(recommendation)
            
            # Sort recommendations by estimated improvement
            def get_estimated_improvement(x):
                return x.estimated_improvement
            recommendations.sort(key=get_estimated_improvement, reverse=True)
            
            # Cache results
            self.optimization_cache[cache_key] = {
                'recommendations': recommendations,
                'timestamp': datetime.utcnow()
            }
            
            # Update statistics
            self.optimization_stats['queries_optimized'] += 1
            self.optimization_stats[f'{query_type.value}_queries'] += 1
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Query optimization error: {e}")
            return []
    
    async def analyze_query_patterns(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze query execution patterns using ML clustering
        """
        try:
            analysis_results = {
                'time_range_hours': time_range_hours,
                'total_queries_analyzed': 0,
                'patterns_identified': [],
                'performance_insights': {},
                'bottlenecks_detected': [],
                'optimization_recommendations': [],
                'resource_utilization': {},
                'temporal_patterns': {}
            }
            
            # Filter queries within time range
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            recent_queries = [
                q for q in self.query_history 
                if q.timestamp > cutoff_time
            ]
            
            if not recent_queries:
                logger.info("No recent queries found for pattern analysis")
                return analysis_results
            
            analysis_results['total_queries_analyzed'] = len(recent_queries)
            
            # Extract features for clustering
            query_features = []
            query_texts = []
            
            for query_metrics in recent_queries:
                features = self._extract_query_features(query_metrics.sql_query)
                query_features.append(features)
                query_texts.append(query_metrics.sql_query)
            
            query_features = np.array(query_features)
            
            # Identify query patterns using clustering
            if len(query_features) > 3:
                clusters = self.query_clusterer.fit_predict(query_features)
                patterns = self._analyze_query_clusters(recent_queries, clusters, query_features)
                analysis_results['patterns_identified'] = patterns
            
            # Analyze performance patterns
            performance_insights = self._analyze_performance_patterns(recent_queries)
            analysis_results['performance_insights'] = performance_insights
            
            # Detect bottlenecks
            bottlenecks = self._detect_performance_bottlenecks(recent_queries, query_features)
            analysis_results['bottlenecks_detected'] = bottlenecks
            
            # Generate pattern-based optimization recommendations
            pattern_recommendations = self._generate_pattern_recommendations(
                patterns if 'patterns' in locals() else [], performance_insights
            )
            analysis_results['optimization_recommendations'] = pattern_recommendations
            
            # Analyze resource utilization
            resource_utilization = self._analyze_resource_utilization(recent_queries)
            analysis_results['resource_utilization'] = resource_utilization
            
            # Identify temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(recent_queries)
            analysis_results['temporal_patterns'] = temporal_patterns
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Query pattern analysis error: {e}")
            return {'error': str(e)}
    
    async def suggest_indexes(self, table_name: str = None, 
                            analysis_period_hours: int = 168) -> List[Dict[str, Any]]:
        """
        Suggest optimal indexes based on query patterns and ML analysis
        """
        try:
            index_suggestions = []
            
            # Analyze query patterns for index opportunities
            cutoff_time = datetime.utcnow() - timedelta(hours=analysis_period_hours)
            relevant_queries = [
                q for q in self.query_history 
                if q.timestamp > cutoff_time and (table_name is None or table_name.lower() in q.sql_query.lower())
            ]
            
            if not relevant_queries:
                logger.info(f"No relevant queries found for index analysis")
                return index_suggestions
            
            # Extract column usage patterns
            column_usage = self._analyze_column_usage_patterns(relevant_queries, table_name)
            
            # Identify frequent WHERE clause patterns
            where_patterns = self._analyze_where_clause_patterns(relevant_queries)
            
            # Analyze JOIN patterns
            join_patterns = self._analyze_join_patterns(relevant_queries)
            
            # Generate index suggestions using ML models
            for table, columns in column_usage.items():
                if table_name and table.lower() != table_name.lower():
                    continue
                
                # Single column indexes
                for column, usage_stats in columns.items():
                    if usage_stats['frequency'] >= 5 and usage_stats['selectivity'] > 0.1:
                        suggestion = {
                            'table': table,
                            'type': 'single_column',
                            'columns': [column],
                            'estimated_benefit': usage_stats['frequency'] * usage_stats['selectivity'],
                            'reasoning': f"Column {column} used in {usage_stats['frequency']} queries with {usage_stats['selectivity']:.2f} selectivity",
                            'sql': f"CREATE INDEX idx_{table}_{column} ON {table}({column})"
                        }
                        index_suggestions.append(suggestion)
                
                # Composite index suggestions
                composite_opportunities = self._identify_composite_index_opportunities(
                    where_patterns.get(table, []), join_patterns.get(table, [])
                )
                
                for opportunity in composite_opportunities:
                    if len(opportunity['columns']) >= 2:
                        column_list = ', '.join(opportunity['columns'])
                        index_name = f"idx_{table}_{'_'.join(opportunity['columns'][:3])}"
                        
                        suggestion = {
                            'table': table,
                            'type': 'composite',
                            'columns': opportunity['columns'],
                            'estimated_benefit': opportunity['benefit_score'],
                            'reasoning': opportunity['reasoning'],
                            'sql': f"CREATE INDEX {index_name} ON {table}({column_list})"
                        }
                        index_suggestions.append(suggestion)
            
            # Sort by estimated benefit
            def get_estimated_benefit(x):
                return x['estimated_benefit']
            index_suggestions.sort(key=get_estimated_benefit, reverse=True)
            
            # Limit to top suggestions to avoid index bloat
            return index_suggestions[:10]
            
        except Exception as e:
            logger.error(f"Index suggestion error: {e}")
            return []
    
    async def predict_query_performance(self, sql_query: str, 
                                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predict query performance using ML models
        """
        try:
            context = context or {}
            
            # Extract query features
            query_features = self._extract_query_features(sql_query, context)
            query_type = self._identify_query_type(sql_query)
            
            # Make predictions
            performance_prediction = {
                'estimated_execution_time': 0.0,
                'estimated_cost': 0.0,
                'confidence_score': 0.0,
                'resource_requirements': {},
                'bottleneck_risks': [],
                'optimization_potential': 0.0
            }
            
            # Predict execution time
            if hasattr(self.execution_time_predictor, 'predict'):
                try:
                    features_scaled = self.scaler.transform([query_features])
                    exec_time = self.execution_time_predictor.predict(features_scaled)[0]
                    performance_prediction['estimated_execution_time'] = max(0.001, exec_time)
                except Exception:
                    performance_prediction['estimated_execution_time'] = self._estimate_execution_time_heuristic(
                        sql_query, query_features
                    )
            
            # Predict query cost
            if hasattr(self.cost_estimator, 'predict'):
                try:
                    features_scaled = self.scaler.transform([query_features])
                    cost = self.cost_estimator.predict(features_scaled)[0]
                    performance_prediction['estimated_cost'] = max(0.1, cost)
                except Exception:
                    performance_prediction['estimated_cost'] = self._estimate_cost_heuristic(
                        sql_query, query_features
                    )
            
            # Assess bottleneck risks
            bottleneck_risks = self._assess_bottleneck_risks(sql_query, query_features)
            performance_prediction['bottleneck_risks'] = bottleneck_risks
            
            # Estimate resource requirements
            resource_requirements = self._estimate_resource_requirements(sql_query, query_features)
            performance_prediction['resource_requirements'] = resource_requirements
            
            # Calculate confidence score
            confidence = self._calculate_prediction_confidence(query_features, query_type)
            performance_prediction['confidence_score'] = confidence
            
            # Assess optimization potential
            optimization_potential = self._assess_optimization_potential(sql_query, query_features)
            performance_prediction['optimization_potential'] = optimization_potential
            
            return performance_prediction
            
        except Exception as e:
            logger.error(f"Performance prediction error: {e}")
            return {'error': str(e)}
    
    def record_query_execution(self, query_metrics: QueryMetrics):
        """
        Record query execution metrics for ML model improvement
        """
        try:
            # Add to query history
            self.query_history.append(query_metrics)
            
            # Update performance tracking
            self.performance_metrics[query_metrics.query_id].append({
                'execution_time': query_metrics.execution_time,
                'timestamp': query_metrics.timestamp,
                'memory_used': query_metrics.memory_used,
                'rows_examined': query_metrics.rows_examined
            })
            
            # Update monitoring metrics
            self._update_monitoring_metrics(query_metrics)
            
            # Trigger model retraining if enough new data
            if len(self.query_history) % 500 == 0:
                asyncio.create_task(self._retrain_models())
            
        except Exception as e:
            logger.error(f"Error recording query execution: {e}")
    
    def update_optimization_feedback(self, query_id: str, 
                                   recommendation_id: str,
                                   improvement_achieved: float):
        """
        Update ML models based on optimization feedback
        """
        try:
            # Find the original recommendation
            # This would typically involve looking up the recommendation details
            
            # Create feedback entry
            feedback_entry = {
                'query_id': query_id,
                'recommendation_id': recommendation_id,
                'improvement_achieved': improvement_achieved,
                'timestamp': datetime.utcnow().isoformat()
            }
            self.feedback_history.append(feedback_entry)
            
            # Update optimization statistics
            self.optimization_stats['feedback_received'] += 1
            if improvement_achieved > 0:
                self.optimization_stats['successful_optimizations'] += 1
            
            # Trigger model updates if enough feedback
            if len(self.feedback_history) % 50 == 0:
                asyncio.create_task(self._update_models_with_feedback())
            
        except Exception as e:
            logger.error(f"Optimization feedback update error: {e}")
    
    # Private helper methods
    def _extract_query_features(self, sql_query: str, 
                              context: Dict[str, Any] = None,
                              schema_info: Dict[str, Any] = None) -> np.ndarray:
        """Extract ML features from SQL query"""
        features = []
        context = context or {}
        schema_info = schema_info or {}
        
        query_lower = sql_query.lower().strip()
        
        # Basic query characteristics
        features.extend([
            len(sql_query),                              # Query length
            len(sql_query.split()),                      # Word count
            sql_query.count('\n'),                       # Line count
            sql_query.count('('),                        # Parentheses (complexity indicator)
            sql_query.count(','),                        # Comma count (column/value count)
        ])
        
        # SQL keyword frequency
        keywords = ['select', 'from', 'where', 'join', 'group by', 'order by', 
                   'having', 'union', 'subquery', 'exists', 'in', 'like']
        keyword_features = [query_lower.count(keyword) for keyword in keywords]
        features.extend(keyword_features)
        
        # Query complexity indicators
        features.extend([
            query_lower.count('select'),                 # Number of SELECT statements
            query_lower.count('join'),                   # Number of JOINs
            query_lower.count('where'),                  # Number of WHERE clauses
            query_lower.count('group by'),               # GROUP BY operations
            query_lower.count('order by'),               # ORDER BY operations
            query_lower.count('union'),                  # UNION operations
        ])
        
        # Function usage
        functions = ['count', 'sum', 'avg', 'max', 'min', 'distinct', 'case']
        function_features = [query_lower.count(func) for func in functions]
        features.extend(function_features)
        
        # Context features
        features.extend([
            context.get('database_size_mb', 0) / 1000,   # Database size (normalized)
            context.get('concurrent_users', 1),          # Concurrent users
            context.get('time_of_day', 12) / 24,         # Time of day (normalized)
            context.get('is_read_replica', 0),           # Read replica flag
        ])
        
        # Schema-based features
        estimated_rows = schema_info.get('estimated_rows', 1000)
        table_count = len(re.findall(r'\bfrom\s+(\w+)', query_lower))
        
        features.extend([
            np.log10(estimated_rows),                    # Log of estimated rows
            table_count,                                 # Number of tables involved
            schema_info.get('index_count', 0),          # Available indexes
        ])
        
        # Pad to fixed size
        target_size = 50
        if len(features) > target_size:
            features = features[:target_size]
        elif len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def _identify_query_type(self, sql_query: str) -> QueryType:
        """Identify the type of SQL query"""
        query_lower = sql_query.lower().strip()
        
        if query_lower.startswith('select'):
            return QueryType.SELECT
        elif query_lower.startswith('insert'):
            return QueryType.INSERT
        elif query_lower.startswith('update'):
            return QueryType.UPDATE
        elif query_lower.startswith('delete'):
            return QueryType.DELETE
        elif query_lower.startswith('create'):
            return QueryType.CREATE
        elif query_lower.startswith('drop'):
            return QueryType.DROP
        elif query_lower.startswith('alter'):
            return QueryType.ALTER
        else:
            return QueryType.UNKNOWN
    
    def _initialize_models(self):
        """Initialize ML models with synthetic training data"""
        # Generate training data for different prediction tasks
        X_time, y_time = self._generate_execution_time_training_data()
        X_cost, y_cost = self._generate_cost_training_data()
        
        # Train models
        if len(X_time) > 0:
            self.execution_time_predictor.fit(X_time, y_time)
        
        if len(X_cost) > 0:
            X_scaled = self.scaler.fit_transform(X_cost)
            self.cost_estimator.fit(X_scaled, y_cost)
    
    def _generate_execution_time_training_data(self) -> Tuple[List[np.ndarray], List[float]]:
        """Extract real execution time training data from query history"""
        X, y = [], []
        
        # Use actual query execution times
        if not hasattr(self, 'query_history') or not self.query_history:
            logger.warning("No query execution history available for training")
            return [], []
        
        for query_record in list(self.query_history):
            try:
                # Extract real execution data
                query_plan = query_record.get('execution_plan')
                actual_time = query_record.get('execution_time_ms', 0)
                was_successful = query_record.get('success', False)
                
                if query_plan and actual_time > 0 and was_successful:
                    # Extract features from real query execution
                    features = self._extract_query_features(query_plan)
                    
                    X.append(features)
                    y.append(actual_time / 1000.0)  # Convert to seconds
                    
            except Exception as e:
                logger.debug(f"Failed to extract execution time training data: {e}")
                continue
        
        logger.info(f"Extracted {len(X)} real query execution time examples")
        return X, y
    
    def _generate_cost_training_data(self) -> Tuple[List[np.ndarray], List[float]]:
        """Extract real cost training data from query resource usage"""
        X, y = [], []
        
        # Use actual query resource costs
        if not hasattr(self, 'query_history') or not self.query_history:
            logger.warning("No query history available for cost training")
            return [], []
        
        for query_record in list(self.query_history):
            try:
                # Extract real cost data
                query_plan = query_record.get('execution_plan')
                resource_cost = query_record.get('resource_cost', 0)
                was_successful = query_record.get('success', False)
                
                if query_plan and resource_cost > 0 and was_successful:
                    # Calculate actual cost from resource usage
                    cpu_time = query_record.get('cpu_time_ms', 0)
                    memory_used = query_record.get('memory_mb', 0)
                    io_operations = query_record.get('io_ops', 0)
                    
                    # Real cost calculation
                    actual_cost = (cpu_time * 0.001 +     # CPU cost
                                 memory_used * 0.01 +     # Memory cost  
                                 io_operations * 0.1)     # IO cost
                    
                    features = self._extract_query_features(query_plan)
                    
                    X.append(features)
                    y.append(max(0.01, actual_cost))  # Real measured cost
            
            X.append(features)
            y.append(cost)
        
        return X, y
    
    def _initialize_anomaly_detector(self):
        """Initialize anomaly detection for query performance"""
        try:
            from sklearn.ensemble import IsolationForest
            return IsolationForest(contamination=0.1, random_state=42)
        except ImportError:
            return None
    
    def _initialize_tokenizer(self):
        """Initialize SQL tokenizer"""
        # Simple tokenizer for SQL queries
        return re.compile(r'\b\w+\b|[^\w\s]')
    
    def _initialize_optimization_rules(self):
        """Initialize query optimization rules"""
        return {
            'missing_index': {
                'pattern': r'WHERE.*=.*AND',
                'suggestion': 'Consider creating composite index',
                'priority': 'high'
            },
            'select_star': {
                'pattern': r'SELECT\s+\*',
                'suggestion': 'Specify only required columns',
                'priority': 'medium'
            },
            'no_limit': {
                'pattern': r'SELECT.*(?!LIMIT)',
                'suggestion': 'Add LIMIT clause for large result sets',
                'priority': 'medium'
            }
        }
    
    def _initialize_index_strategies(self):
        """Initialize index suggestion strategies"""
        return {
            'frequent_where': 'Create index on frequently filtered columns',
            'join_columns': 'Create indexes on JOIN key columns',
            'order_by': 'Create index for ORDER BY optimization',
            'group_by': 'Consider covering index for GROUP BY queries'
        }


# Singleton instance
_ai_query_optimizer = None

def get_ai_query_optimizer() -> AIQueryOptimizer:
    """Get or create AI query optimizer instance"""
    global _ai_query_optimizer
    if not _ai_query_optimizer:
        _ai_query_optimizer = AIQueryOptimizer()
    return _ai_query_optimizer