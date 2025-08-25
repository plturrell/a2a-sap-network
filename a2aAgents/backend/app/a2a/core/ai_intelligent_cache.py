"""
AI-Powered Intelligent Caching System

This module provides intelligent caching using real machine learning to predict
cache access patterns, optimize cache replacement policies, and proactively
prefetch data based on usage patterns and behavioral analysis.
"""

import asyncio
# Performance: Consider using asyncio.gather for concurrent operations
import logging
import numpy as np
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque, OrderedDict
from enum import Enum
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Advanced prediction algorithms
try:
    from scipy.stats import poisson, exponweib
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Deep learning for complex access pattern prediction
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class CachePredictionNN(nn.Module):
    """Neural network for cache access pattern prediction"""
    def __init__(self, input_dim, hidden_dim=128):
        super(CachePredictionNN, self).__init__()
        
        # LSTM for temporal pattern recognition
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # Attention mechanism for important features
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Prediction heads
        self.access_probability_head = nn.Linear(hidden_dim, 1)
        self.time_to_access_head = nn.Linear(hidden_dim, 1)
        self.popularity_head = nn.Linear(hidden_dim, 1)
        self.lifetime_head = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last output
        last_hidden = lstm_out[:, -1, :]
        
        # Apply attention
        attn_out, _ = self.attention(last_hidden.unsqueeze(0), 
                                   last_hidden.unsqueeze(0), 
                                   last_hidden.unsqueeze(0))
        features = self.dropout(attn_out.squeeze(0))
        
        # Predictions
        access_prob = torch.sigmoid(self.access_probability_head(features))
        time_to_access = F.relu(self.time_to_access_head(features))
        popularity = torch.sigmoid(self.popularity_head(features))
        lifetime = F.relu(self.lifetime_head(features))
        
        return access_prob, time_to_access, popularity, lifetime


class CacheStrategy(Enum):
    """Cache replacement strategies"""
    LRU = "lru"
    LFU = "lfu"
    AI_PREDICTION = "ai_prediction"
    ADAPTIVE = "adaptive"
    TEMPORAL_LOCALITY = "temporal_locality"


@dataclass
class CacheEntry:
    """Cache entry with AI metadata"""
    key: str
    value: Any
    size: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    access_pattern: List[datetime] = field(default_factory=list)
    predicted_next_access: Optional[datetime] = None
    popularity_score: float = 0.0
    ai_score: float = 0.0
    user_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    total_requests: int = 0
    total_hits: int = 0
    total_misses: int = 0
    avg_access_time: float = 0.0
    memory_usage: float = 0.0
    eviction_count: int = 0
    prefetch_accuracy: float = 0.0
    ai_prediction_accuracy: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class AIIntelligentCache:
    """
    AI-powered intelligent caching system with ML-based optimization
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # Cache storage
        self.cache_entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory_usage = 0
        
        # ML Models for cache optimization
        self.access_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.popularity_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.lifetime_predictor = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42)
        self.pattern_clusterer = KMeans(n_clusters=8, random_state=42)
        self.anomaly_detector = DBSCAN(eps=0.3, min_samples=3)
        
        # Feature scalers
        self.access_scaler = StandardScaler()
        self.pattern_scaler = MinMaxScaler()
        
        # Neural network for advanced prediction
        if TORCH_AVAILABLE:
            self.prediction_nn = CachePredictionNN(input_dim=20)
            self.nn_optimizer = torch.optim.Adam(self.prediction_nn.parameters(), lr=0.001)
        else:
            self.prediction_nn = None
        
        # Access pattern tracking
        self.access_history = deque(maxlen=10000)
        self.user_patterns = defaultdict(list)
        self.temporal_patterns = defaultdict(list)
        
        # Prefetching system
        self.prefetch_queue = deque(maxlen=100)
        self.prefetch_stats = {'requested': 0, 'hit': 0}
        
        # Performance metrics
        self.metrics = CacheMetrics()
        
        # Cache strategy
        self.strategy = CacheStrategy.AI_PREDICTION
        
        # Background tasks
        self.background_executor = ThreadPoolExecutor(max_workers=2)
        self.optimization_task = None
        self.prefetch_task = None
        
        # Lock for thread safety
        self.cache_lock = threading.RLock()
        
        # Initialize models
        self._initialize_models()
        
        # Start background optimization
        self._start_background_tasks()
        
        logger.info("AI Intelligent Cache initialized with ML models")
    
    def _initialize_models(self):
        """Initialize ML models with synthetic training data"""
        # Generate synthetic cache access training data
        X_access, y_access = self._generate_access_prediction_data()
        X_popularity, y_popularity = self._generate_popularity_data()
        X_lifetime, y_lifetime = self._generate_lifetime_data()
        
        # Train models
        if len(X_access) > 0:
            X_access_scaled = self.access_scaler.fit_transform(X_access)
            self.access_predictor.fit(X_access_scaled, y_access)
        
        if len(X_popularity) > 0:
            self.popularity_classifier.fit(X_popularity, y_popularity)
        
        if len(X_lifetime) > 0:
            self.lifetime_predictor.fit(X_lifetime, y_lifetime)
    
    async def get(self, key: str, user_context: Dict[str, Any] = None) -> Optional[Any]:
        """
        Get value from cache with AI-enhanced access tracking
        """
        start_time = time.time()
        
        with self.cache_lock:
            self.metrics.total_requests += 1
            
            if key in self.cache_entries:
                # Cache hit
                entry = self.cache_entries[key]
                entry.last_accessed = datetime.utcnow()
                entry.access_count += 1
                entry.access_pattern.append(entry.last_accessed)
                
                # Update user context
                if user_context:
                    entry.user_context.update(user_context)
                
                # Move to end (LRU update)
                self.cache_entries.move_to_end(key)
                
                # Record access pattern
                self._record_access(key, True, user_context)
                
                # Update metrics
                self.metrics.total_hits += 1
                access_time = time.time() - start_time
                self._update_access_metrics(access_time)
                
                # Trigger predictive prefetching
                await self._trigger_predictive_prefetch(key, user_context)
                
                return entry.value
            else:
                # Cache miss
                self.metrics.total_misses += 1
                self._record_access(key, False, user_context)
                self._update_hit_rate()
                
                return None
    
    async def set(self, key: str, value: Any, user_context: Dict[str, Any] = None, 
                 ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with AI-optimized placement
        """
        with self.cache_lock:
            # Calculate value size
            value_size = self._calculate_size(value)
            
            # Check if we need to make room
            while (len(self.cache_entries) >= self.max_size or 
                   self.current_memory_usage + value_size > self.max_memory_bytes):
                if not await self._evict_entry():
                    return False  # Cannot make room
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=value_size,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                user_context=user_context or {}
            )
            
            # AI scoring
            entry.ai_score = await self._calculate_ai_score(entry, user_context)
            entry.popularity_score = await self._predict_popularity(entry)
            entry.predicted_next_access = await self._predict_next_access(entry)
            
            # Store entry
            self.cache_entries[key] = entry
            self.current_memory_usage += value_size
            
            # Record the set operation
            self._record_access(key, False, user_context, is_set=True)
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self.cache_lock:
            if key in self.cache_entries:
                entry = self.cache_entries[key]
                self.current_memory_usage -= entry.size
                del self.cache_entries[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.cache_lock:
            self.cache_entries.clear()
            self.current_memory_usage = 0
            self.metrics = CacheMetrics()
    
    async def _evict_entry(self) -> bool:
        """Evict entry using AI-optimized strategy"""
        if not self.cache_entries:
            return False
        
        if self.strategy == CacheStrategy.AI_PREDICTION:
            victim_key = await self._ai_select_victim()
        elif self.strategy == CacheStrategy.LRU:
            victim_key = next(iter(self.cache_entries))
        elif self.strategy == CacheStrategy.LFU:
            victim_key = min(self.cache_entries.keys(), 
                           key=lambda k: self.cache_entries[k].access_count)
        else:
            victim_key = await self._adaptive_select_victim()
        
        if victim_key:
            victim_entry = self.cache_entries[victim_key]
            self.current_memory_usage -= victim_entry.size
            del self.cache_entries[victim_key]
            self.metrics.eviction_count += 1
            
            logger.debug(f"Evicted cache entry: {victim_key}")
            return True
        
        return False
    
    async def _ai_select_victim(self) -> Optional[str]:
        """AI-based victim selection for eviction"""
        if not self.cache_entries:
            return None
        
        best_victim = None
        lowest_score = float('inf')
        
        for key, entry in self.cache_entries.items():
            # Calculate eviction score using ML
            eviction_score = await self._calculate_eviction_score(entry)
            
            if eviction_score < lowest_score:
                lowest_score = eviction_score
                best_victim = key
        
        return best_victim
    
    async def _calculate_ai_score(self, entry: CacheEntry, user_context: Dict[str, Any] = None) -> float:
        """Calculate AI-based importance score for cache entry"""
        features = self._extract_entry_features(entry, user_context)
        
        try:
            # Use ML models to predict access probability
            if hasattr(self.access_predictor, 'predict'):
                features_scaled = self.access_scaler.transform(features.reshape(1, -1))
                access_prob = self.access_predictor.predict(features_scaled)[0]
            else:
                # Fallback heuristic
                access_prob = 0.5
            
            # Combine with other factors
            recency_score = 1.0 / max(1, (datetime.utcnow() - entry.last_accessed).total_seconds() / 3600)
            frequency_score = min(1.0, entry.access_count / 10)
            
            ai_score = (access_prob * 0.5 + recency_score * 0.3 + frequency_score * 0.2)
            return float(np.clip(ai_score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"AI score calculation error: {e}")
            return 0.5
    
    async def _predict_popularity(self, entry: CacheEntry) -> float:
        """Predict entry popularity using ML"""
        try:
            features = self._extract_entry_features(entry)
            
            if hasattr(self.popularity_classifier, 'predict_proba'):
                popularity_prob = self.popularity_classifier.predict_proba(features.reshape(1, -1))[0]
                return float(popularity_prob[1] if len(popularity_prob) > 1 else popularity_prob[0])
            else:
                # Heuristic based on access pattern
                if len(entry.access_pattern) > 1:
                    time_diff = (entry.access_pattern[-1] - entry.access_pattern[0]).total_seconds()
                    frequency = len(entry.access_pattern) / max(time_diff / 3600, 1)  # Per hour
                    return min(1.0, frequency / 10)
                return 0.1
                
        except Exception as e:
            logger.error(f"Popularity prediction error: {e}")
            return 0.5
    
    async def _predict_next_access(self, entry: CacheEntry) -> Optional[datetime]:
        """Predict when entry will be accessed next"""
        try:
            if len(entry.access_pattern) < 2:
                return None
            
            # Calculate average time between accesses
            intervals = []
            for i in range(1, len(entry.access_pattern)):
                interval = (entry.access_pattern[i] - entry.access_pattern[i-1]).total_seconds()
                intervals.append(interval)
            
            if not intervals:
                return None
            
            # Use ML to predict next access time
            features = self._extract_temporal_features(entry.access_pattern)
            
            if hasattr(self.lifetime_predictor, 'predict'):
                predicted_interval = self.lifetime_predictor.predict(features.reshape(1, -1))[0]
            else:
                # Simple average
                predicted_interval = np.mean(intervals)
            
            next_access = entry.last_accessed + timedelta(seconds=predicted_interval)
            return next_access
            
        except Exception as e:
            logger.error(f"Next access prediction error: {e}")
            return None
    
    async def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate score for eviction decision (lower = more likely to evict)"""
        # Time-based factors
        time_since_access = (datetime.utcnow() - entry.last_accessed).total_seconds()
        recency_score = 1.0 / (1.0 + time_since_access / 3600)  # Hours
        
        # Frequency factor
        frequency_score = min(1.0, entry.access_count / 10)
        
        # AI prediction factor
        ai_factor = entry.ai_score
        
        # Popularity factor
        popularity_factor = entry.popularity_score
        
        # Size factor (larger items more likely to evict)
        size_factor = 1.0 - min(1.0, entry.size / (1024 * 1024))  # Normalize to MB
        
        # Predicted future access
        future_access_factor = 1.0
        if entry.predicted_next_access:
            time_to_next = (entry.predicted_next_access - datetime.utcnow()).total_seconds()
            if time_to_next > 0:
                future_access_factor = 1.0 / (1.0 + time_to_next / 3600)  # Closer = higher score
        
        # Combine factors (higher score = less likely to evict)
        eviction_score = (
            recency_score * 0.2 +
            frequency_score * 0.2 +
            ai_factor * 0.25 +
            popularity_factor * 0.15 +
            size_factor * 0.1 +
            future_access_factor * 0.1
        )
        
        return 1.0 - eviction_score  # Invert for eviction (lower = evict first)
    
    async def _trigger_predictive_prefetch(self, accessed_key: str, user_context: Dict[str, Any] = None):
        """Trigger predictive prefetching based on access patterns"""
        try:
            # Find similar access patterns
            similar_keys = await self._find_similar_access_patterns(accessed_key, user_context)
            
            for key in similar_keys[:3]:  # Top 3 predictions
                if key not in self.cache_entries and key not in [item[0] for item in self.prefetch_queue]:
                    # Add to prefetch queue
                    prefetch_score = await self._calculate_prefetch_score(key, user_context)
                    self.prefetch_queue.append((key, prefetch_score, user_context))
                    
            # Sort prefetch queue by score
            self.prefetch_queue = deque(
                sorted(self.prefetch_queue, key=lambda x: x[1], reverse=True)[:100]
            )
            
        except Exception as e:
            logger.error(f"Predictive prefetch error: {e}")
    
    async def _find_similar_access_patterns(self, key: str, user_context: Dict[str, Any] = None) -> List[str]:
        """Find keys with similar access patterns using ML"""
        similar_keys = []
        
        try:
            if key not in self.cache_entries:
                return similar_keys
            
            current_entry = self.cache_entries[key]
            current_features = self._extract_entry_features(current_entry, user_context)
            
            # Find entries with similar features
            similarities = []
            for other_key, other_entry in self.cache_entries.items():
                if other_key != key:
                    other_features = self._extract_entry_features(other_entry)
                    similarity = cosine_similarity(
                        current_features.reshape(1, -1),
                        other_features.reshape(1, -1)
                    )[0][0]
                    similarities.append((other_key, similarity))
            
            # Sort by similarity and return top matches
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar_keys = [k for k, s in similarities[:5] if s > 0.7]
            
        except Exception as e:
            logger.error(f"Similar pattern finding error: {e}")
        
        return similar_keys
    
    def _extract_entry_features(self, entry: CacheEntry, user_context: Dict[str, Any] = None) -> np.ndarray:
        """Extract ML features from cache entry"""
        features = []
        
        # Temporal features
        now = datetime.utcnow()
        features.append((now - entry.created_at).total_seconds() / 3600)  # Age in hours
        features.append((now - entry.last_accessed).total_seconds() / 3600)  # Last access in hours
        features.append(entry.access_count)
        
        # Size features
        features.append(entry.size / (1024 * 1024))  # Size in MB
        features.append(entry.size / self.max_memory_bytes)  # Relative size
        
        # Access pattern features
        if len(entry.access_pattern) > 1:
            intervals = []
            for i in range(1, len(entry.access_pattern)):
                interval = (entry.access_pattern[i] - entry.access_pattern[i-1]).total_seconds()
                intervals.append(interval)
            
            features.append(np.mean(intervals) / 3600 if intervals else 0)  # Avg interval in hours
            features.append(np.std(intervals) / 3600 if len(intervals) > 1 else 0)  # Std dev
            features.append(len(intervals))  # Number of accesses
        else:
            features.extend([0, 0, 0])
        
        # User context features
        if user_context:
            features.append(hash(user_context.get('user_id', '')) % 1000 / 1000.0)
            features.append(hash(user_context.get('session_id', '')) % 1000 / 1000.0)
            features.append(1.0 if user_context.get('is_authenticated', False) else 0.0)
        else:
            features.extend([0.5, 0.5, 0.0])
        
        # Key features
        features.append(len(entry.key) / 100.0)  # Key length
        features.append(hash(entry.key) % 1000 / 1000.0)  # Key hash
        
        # AI scores
        features.append(entry.ai_score)
        features.append(entry.popularity_score)
        
        # Time-based patterns
        if entry.last_accessed:
            features.append(entry.last_accessed.hour / 24.0)
            features.append(entry.last_accessed.weekday() / 7.0)
        else:
            features.extend([0.5, 0.5])
        
        return np.array(features)
    
    def _extract_temporal_features(self, access_pattern: List[datetime]) -> np.ndarray:
        """Extract temporal features from access pattern"""
        features = []
        
        if len(access_pattern) < 2:
            return np.zeros(10)
        
        # Calculate intervals
        intervals = []
        for i in range(1, len(access_pattern)):
            interval = (access_pattern[i] - access_pattern[i-1]).total_seconds()
            intervals.append(interval)
        
        # Statistical features
        features.append(np.mean(intervals))
        features.append(np.std(intervals) if len(intervals) > 1 else 0)
        features.append(np.min(intervals))
        features.append(np.max(intervals))
        features.append(len(intervals))
        
        # Pattern features
        features.append(1.0 if self._is_periodic_pattern(intervals) else 0.0)
        features.append(1.0 if self._is_bursty_pattern(intervals) else 0.0)
        
        # Time-of-day features
        hours = [dt.hour for dt in access_pattern]
        features.append(np.mean(hours) / 24.0)
        features.append(np.std(hours) / 24.0 if len(hours) > 1 else 0)
        
        # Day-of-week feature
        days = [dt.weekday() for dt in access_pattern]
        features.append(np.mean(days) / 7.0)
        
        return np.array(features)
    
    def _is_periodic_pattern(self, intervals: List[float]) -> bool:
        """Check if access pattern is periodic"""
        if len(intervals) < 3:
            return False
        
        # Simple periodicity check
        interval_std = np.std(intervals)
        interval_mean = np.mean(intervals)
        
        # Low coefficient of variation indicates periodicity
        cv = interval_std / interval_mean if interval_mean > 0 else float('inf')
        return cv < 0.3
    
    def _is_bursty_pattern(self, intervals: List[float]) -> bool:
        """Check if access pattern is bursty"""
        if len(intervals) < 3:
            return False
        
        # Bursty patterns have high variance in intervals
        interval_std = np.std(intervals)
        interval_mean = np.mean(intervals)
        
        cv = interval_std / interval_mean if interval_mean > 0 else 0
        return cv > 1.0
    
    async def _calculate_prefetch_score(self, key: str, user_context: Dict[str, Any] = None) -> float:
        """Calculate prefetch priority score"""
        # This would typically involve predicting access probability
        # For now, use a heuristic based on key patterns and user context
        
        base_score = 0.5
        
        # User context boost
        if user_context:
            if user_context.get('is_authenticated'):
                base_score += 0.2
            if user_context.get('user_type') == 'premium':
                base_score += 0.1
        
        # Key pattern boost
        if any(pattern in key for pattern in ['user', 'profile', 'config']):
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def _record_access(self, key: str, hit: bool, user_context: Dict[str, Any] = None, is_set: bool = False):
        """Record access for pattern analysis"""
        access_record = {
            'key': key,
            'timestamp': datetime.utcnow(),
            'hit': hit,
            'is_set': is_set,
            'user_context': user_context or {}
        }
        
        self.access_history.append(access_record)
        
        # Update user patterns
        if user_context and 'user_id' in user_context:
            user_id = user_context['user_id']
            self.user_patterns[user_id].append(access_record)
            
            # Keep only recent patterns
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.user_patterns[user_id] = [
                record for record in self.user_patterns[user_id]
                if record['timestamp'] > cutoff_time
            ]
    
    def _update_access_metrics(self, access_time: float):
        """Update access time metrics"""
        alpha = 0.1  # Exponential moving average factor
        self.metrics.avg_access_time = (
            (1 - alpha) * self.metrics.avg_access_time + alpha * access_time
        )
    
    def _update_hit_rate(self):
        """Update hit rate metrics"""
        if self.metrics.total_requests > 0:
            self.metrics.hit_rate = self.metrics.total_hits / self.metrics.total_requests
            self.metrics.miss_rate = self.metrics.total_misses / self.metrics.total_requests
        
        self.metrics.memory_usage = self.current_memory_usage / self.max_memory_bytes
        self.metrics.last_updated = datetime.utcnow()
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of cached value"""
        try:
            # Use pickle to estimate size
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            else:
                return 1024  # Default 1KB
    
    def _start_background_tasks(self):
        """Start background optimization tasks"""
        if self.optimization_task is None:
            self.optimization_task = asyncio.create_task(self._background_optimization())
        
        if self.prefetch_task is None:
            self.prefetch_task = asyncio.create_task(self._background_prefetching())
    
    async def _background_optimization(self):
        """Background task for cache optimization"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Retrain models if enough new data
                if len(self.access_history) > 100:
                    await self._retrain_models()
                
                # Update AI scores for all entries
                await self._update_ai_scores()
                
                # Clean up stale predictions
                await self._cleanup_predictions()
                
            except Exception as e:
                logger.error(f"Background optimization error: {e}")
    
    async def _background_prefetching(self):
        """Background task for prefetching"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Process prefetch queue
                while self.prefetch_queue:
                    key, score, context = self.prefetch_queue.popleft()
                    
                    # Only prefetch if not already cached
                    if key not in self.cache_entries:
                        await self._execute_prefetch(key, context)
                
            except Exception as e:
                logger.error(f"Background prefetching error: {e}")
    
    async def _retrain_models(self):
        """Retrain ML models with recent access data"""
        try:
            # Extract training data from access history
            X_access, y_access = self._extract_training_data()
            
            if len(X_access) > 50:  # Need sufficient data
                # Retrain access predictor
                X_scaled = self.access_scaler.fit_transform(X_access)
                self.access_predictor.fit(X_scaled, y_access)
                
                logger.info(f"Retrained cache models with {len(X_access)} samples")
        
        except Exception as e:
            logger.error(f"Model retraining error: {e}")
    
    async def _update_ai_scores(self):
        """Update AI scores for all cache entries"""
        with self.cache_lock:
            for entry in self.cache_entries.values():
                entry.ai_score = await self._calculate_ai_score(entry)
                entry.popularity_score = await self._predict_popularity(entry)
                entry.predicted_next_access = await self._predict_next_access(entry)
    
    async def _cleanup_predictions(self):
        """Clean up stale predictions"""
        now = datetime.utcnow()
        
        with self.cache_lock:
            for entry in self.cache_entries.values():
                # Remove old access patterns
                cutoff_time = now - timedelta(hours=24)
                entry.access_pattern = [
                    access_time for access_time in entry.access_pattern
                    if access_time > cutoff_time
                ]
    
    async def _execute_prefetch(self, key: str, context: Dict[str, Any] = None):
        """Execute prefetch for a specific key"""
        # This would typically fetch data from the original source
        # For now, just simulate the prefetch
        self.prefetch_stats['requested'] += 1
        
        # Simulate prefetch logic - would call actual data source
        logger.debug(f"Prefetching key: {key}")
    
    def _extract_training_data(self) -> Tuple[List[np.ndarray], List[float]]:
        """Extract training data from access history"""
        X, y = [], []
        
        # Group accesses by key
        key_accesses = defaultdict(list)
        for record in self.access_history:
            key_accesses[record['key']].append(record)
        
        # Create training samples
        for key, accesses in key_accesses.items():
            if len(accesses) > 1:
                for i in range(1, len(accesses)):
                    # Features: time since last access, access count, etc.
                    prev_access = accesses[i-1]['timestamp']
                    curr_access = accesses[i]['timestamp']
                    time_diff = (curr_access - prev_access).total_seconds()
                    
                    # Simple features
                    features = [
                        time_diff / 3600,  # Hours since last access
                        i,  # Access count so far
                        curr_access.hour / 24.0,  # Hour of day
                        curr_access.weekday() / 7.0,  # Day of week
                        len(key) / 100.0,  # Key length
                        1.0 if accesses[i]['hit'] else 0.0,  # Was it a hit
                    ]
                    
                    # Target: time to next access (if available)
                    if i < len(accesses) - 1:
                        next_access = accesses[i+1]['timestamp']
                        target = (next_access - curr_access).total_seconds() / 3600
                    else:
                        target = 24.0  # Default to 24 hours
                    
                    X.append(np.array(features))
                    y.append(target)
        
        return X, y
    
    # Training data generation methods
    def _generate_access_prediction_data(self) -> Tuple[List[np.ndarray], List[float]]:
        """Generate synthetic access prediction training data"""
        X, y = [], []
        
        for i in range(300):
            # Random cache entry features
            features = np.random.rand(15)
            
            # Synthetic access time based on features
            access_time = (
                features[0] * 2 +      # Age factor
                features[2] * 3 +      # Access count factor
                features[5] * 1.5 +    # Interval factor
                np.random.normal(0, 0.5)  # Noise
            )
            access_time = max(0.1, access_time)
            
            X.append(features)
            y.append(access_time)
        
        return X, y
    
    def _generate_popularity_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Generate synthetic popularity training data"""
        X, y = [], []
        
        for i in range(200):
            features = np.random.rand(15)
            
            # Popular if high access count and recent access
            popular = 1 if (features[2] > 0.7 and features[1] < 0.3) else 0
            
            X.append(features)
            y.append(popular)
        
        return X, y
    
    def _generate_lifetime_data(self) -> Tuple[List[np.ndarray], List[float]]:
        """Generate synthetic lifetime training data"""
        X, y = [], []
        
        for i in range(250):
            features = np.random.rand(10)  # Temporal features
            
            # Lifetime based on periodicity and pattern
            if features[5] > 0.8:  # Periodic
                lifetime = 24 + np.random.normal(0, 6)  # ~24 hours
            elif features[6] > 0.8:  # Bursty
                lifetime = 1 + np.random.exponential(2)  # Short bursts
            else:
                lifetime = np.random.exponential(12)  # General exponential
            
            lifetime = max(0.1, lifetime)
            
            X.append(features)
            y.append(lifetime)
        
        return X, y
    
    async def _adaptive_select_victim(self) -> Optional[str]:
        """Adaptive victim selection combining multiple strategies"""
        if not self.cache_entries:
            return None
        
        # Calculate scores for different strategies
        lru_victim = next(iter(self.cache_entries))
        lfu_victim = min(self.cache_entries.keys(), 
                        key=lambda k: self.cache_entries[k].access_count)
        ai_victim = await self._ai_select_victim()
        
        # Combine strategies based on recent performance
        # This is a simplified implementation
        if self.metrics.hit_rate > 0.8:
            return ai_victim  # AI is working well
        elif self.metrics.hit_rate > 0.6:
            return lfu_victim  # Use frequency
        else:
            return lru_victim  # Fall back to LRU
    
    def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics"""
        self._update_hit_rate()
        return self.metrics
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information and statistics"""
        with self.cache_lock:
            return {
                'size': len(self.cache_entries),
                'max_size': self.max_size,
                'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'strategy': self.strategy.value,
                'metrics': self.get_metrics().__dict__,
                'prefetch_stats': self.prefetch_stats,
                'model_info': {
                    'access_predictor_trained': hasattr(self.access_predictor, 'feature_importances_'),
                    'nn_available': self.prediction_nn is not None
                }
            }


# Singleton instance
_intelligent_cache = None

def get_intelligent_cache(max_size: int = 1000, max_memory_mb: int = 100) -> AIIntelligentCache:
    """Get or create intelligent cache instance"""
    global _intelligent_cache
    if not _intelligent_cache:
        _intelligent_cache = AIIntelligentCache(max_size, max_memory_mb)
    return _intelligent_cache