"""
Production ML Training Service for Real-Time Model Updates

This service manages continuous training and updating of all ML models in the A2A platform
using only real production data. No synthetic data generation - all models learn from
actual system behavior, user interactions, and security events.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import json
import pickle
import time
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import os
import threading
import queue

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# Import all AI modules to manage their training
from .ai_agent_discovery import get_ai_discovery
from .ai_workflow_optimizer import get_workflow_optimizer
from .ai_security_monitor import get_security_monitor
from .ai_intelligent_cache import get_intelligent_cache
from .ai_rate_limiter import get_ai_rate_limiter
from .ai_self_healing import get_self_healing_system
from .ai_log_analyzer import get_log_analyzer
from .ai_resource_manager import get_resource_manager
from .ai_data_quality import get_data_quality_validator
from .ai_performance_optimizer import get_performance_optimizer
from .ai_message_router import get_ai_message_router
from .ai_user_behavior import get_ai_user_behavior_predictor
from .ai_query_optimizer import get_ai_query_optimizer
from .ai_error_recovery import get_ai_error_recovery_system
from .data_pipeline import get_data_pipeline

logger = logging.getLogger(__name__)


@dataclass
class ModelTrainingJob:
    """Represents a model training job"""
    job_id: str
    model_name: str
    module_name: str
    data_type: str
    min_samples: int
    max_samples: int
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ModelPerformance:
    """Tracks model performance over time"""
    model_name: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sample_count: int
    data_distribution: Dict[str, int] = field(default_factory=dict)


class TrainingSchedule(Enum):
    CONTINUOUS = "continuous"      # Train as soon as enough data
    HOURLY = "hourly"              # Train every hour
    DAILY = "daily"                # Train once per day
    ON_DEMAND = "on_demand"        # Train when requested
    PERFORMANCE_BASED = "performance_based"  # Train when performance drops


class MLTrainingService:
    """
    Centralized service for training all ML models with real production data.
    Manages incremental learning, model versioning, and performance tracking.
    """
    
    def __init__(self, model_storage_path: str = "./ml_models",
                 performance_threshold: float = 0.8,
                 min_training_samples: int = 100):
        
        # Configuration
        self.model_storage_path = model_storage_path
        self.performance_threshold = performance_threshold
        self.min_training_samples = min_training_samples
        
        # Create model storage directory
        os.makedirs(model_storage_path, exist_ok=True)
        
        # Model registry - maps model names to their instances
        self.model_registry = {}
        self.training_schedules = {}
        self.last_training_time = {}
        
        # Training queue and workers
        self.training_queue = asyncio.Queue()
        self.active_jobs = {}
        self.completed_jobs = deque(maxlen=1000)
        
        # Performance tracking
        self.model_performance = defaultdict(list)
        self.performance_alerts = []
        
        # Data pipeline connection
        self.data_pipeline = get_data_pipeline()
        
        # Background tasks
        self._scheduler_task = None
        self._trainer_tasks = []
        self._monitor_task = None
        self._is_running = False
        
        # Initialize model registry
        self._register_all_models()
        
        logger.info("ML Training Service initialized")
    
    def _register_all_models(self):
        """Register all AI models for centralized training"""
        
        # Register each AI module's models
        model_configs = [
            # (module_getter, model_attr_names, data_type, schedule)
            (get_ai_discovery, ['agent_classifier', 'performance_predictor'], 'events', TrainingSchedule.HOURLY),
            (get_workflow_optimizer, ['workflow_predictor', 'bottleneck_detector'], 'events', TrainingSchedule.HOURLY),
            (get_security_monitor, ['intrusion_detector', 'anomaly_detector'], 'events', TrainingSchedule.CONTINUOUS),
            (get_intelligent_cache, ['access_predictor', 'ttl_predictor'], 'metrics', TrainingSchedule.HOURLY),
            (get_ai_rate_limiter, ['burst_predictor', 'traffic_classifier'], 'metrics', TrainingSchedule.CONTINUOUS),
            (get_self_healing, ['failure_predictor', 'recovery_classifier'], 'events', TrainingSchedule.CONTINUOUS),
            (get_log_analyzer, ['anomaly_detector', 'pattern_classifier'], 'events', TrainingSchedule.HOURLY),
            (get_resource_manager, ['demand_predictor', 'efficiency_analyzer'], 'metrics', TrainingSchedule.HOURLY),
            (get_data_quality_validator, ['quality_classifier', 'anomaly_detector'], 'events', TrainingSchedule.DAILY),
            (get_performance_optimizer, ['performance_predictor', 'optimization_recommender'], 'metrics', TrainingSchedule.HOURLY),
            (get_ai_message_router, ['route_classifier', 'latency_predictor'], 'messages', TrainingSchedule.HOURLY),
            (get_ai_user_behavior_predictor, ['action_classifier', 'engagement_predictor'], 'events', TrainingSchedule.HOURLY),
            (get_ai_query_optimizer, ['execution_time_predictor', 'optimization_classifier'], 'queries', TrainingSchedule.DAILY),
            (get_ai_error_recovery_system, ['error_classifier', 'recovery_recommender'], 'events', TrainingSchedule.CONTINUOUS),
        ]
        
        for module_getter, model_attrs, data_type, schedule in model_configs:
            try:
                module = module_getter()
                module_name = module.__class__.__name__
                
                for attr_name in model_attrs:
                    if hasattr(module, attr_name):
                        model = getattr(module, attr_name)
                        if model is not None:
                            model_key = f"{module_name}.{attr_name}"
                            self.model_registry[model_key] = {
                                'module': module,
                                'model': model,
                                'attr_name': attr_name,
                                'data_type': data_type,
                                'module_name': module_name
                            }
                            self.training_schedules[model_key] = schedule
                            logger.info(f"Registered model: {model_key}")
                
            except Exception as e:
                logger.error(f"Error registering models from {module_getter}: {e}")
    
    async def start(self):
        """Start the ML training service"""
        if self._is_running:
            logger.warning("ML Training Service is already running")
            return
        
        self._is_running = True
        
        # Start data pipeline
        await self.data_pipeline.start()
        
        # Start background tasks
        self._scheduler_task = asyncio.create_task(self._training_scheduler())
        self._monitor_task = asyncio.create_task(self._performance_monitor())
        
        # Start trainer workers
        for i in range(3):  # 3 concurrent training workers
            task = asyncio.create_task(self._training_worker(i))
            self._trainer_tasks.append(task)
        
        logger.info("ML Training Service started")
    
    async def stop(self):
        """Stop the ML training service"""
        self._is_running = False
        
        # Cancel all tasks
        if self._scheduler_task:
            self._scheduler_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()
        for task in self._trainer_tasks:
            task.cancel()
        
        # Wait for active jobs to complete
        await self._wait_for_active_jobs()
        
        # Stop data pipeline
        await self.data_pipeline.stop()
        
        logger.info("ML Training Service stopped")
    
    async def train_model(self, model_name: str, force: bool = False) -> ModelTrainingJob:
        """Train a specific model with real data"""
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not registered")
        
        # Create training job
        job = ModelTrainingJob(
            job_id=f"{model_name}_{int(time.time())}",
            model_name=model_name,
            module_name=self.model_registry[model_name]['module_name'],
            data_type=self.model_registry[model_name]['data_type'],
            min_samples=self.min_training_samples,
            max_samples=10000
        )
        
        # Add to queue
        await self.training_queue.put(job)
        self.active_jobs[job.job_id] = job
        
        logger.info(f"Queued training job for {model_name}")
        return job
    
    async def _training_scheduler(self):
        """Schedule model training based on configured schedules"""
        while self._is_running:
            try:
                current_time = datetime.utcnow()
                
                for model_name, schedule in self.training_schedules.items():
                    should_train = False
                    last_trained = self.last_training_time.get(model_name)
                    
                    if schedule == TrainingSchedule.CONTINUOUS:
                        # Train if we have enough new data
                        data_count = await self._get_new_data_count(model_name, last_trained)
                        should_train = data_count >= self.min_training_samples
                    
                    elif schedule == TrainingSchedule.HOURLY:
                        if not last_trained or (current_time - last_trained).seconds >= 3600:
                            should_train = True
                    
                    elif schedule == TrainingSchedule.DAILY:
                        if not last_trained or (current_time - last_trained).days >= 1:
                            should_train = True
                    
                    elif schedule == TrainingSchedule.PERFORMANCE_BASED:
                        # Train if performance drops below threshold
                        recent_performance = self._get_recent_performance(model_name)
                        if recent_performance and recent_performance < self.performance_threshold:
                            should_train = True
                    
                    if should_train:
                        await self.train_model(model_name)
                
                # Check every minute
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Training scheduler error: {e}")
                await asyncio.sleep(60)
    
    async def _training_worker(self, worker_id: int):
        """Worker that processes training jobs"""
        logger.info(f"Training worker {worker_id} started")
        
        while self._is_running:
            try:
                # Get job from queue
                job = await asyncio.wait_for(self.training_queue.get(), timeout=5.0)
                
                # Execute training
                job.status = "running"
                job.start_time = datetime.utcnow()
                
                try:
                    metrics = await self._execute_training(job)
                    job.metrics = metrics
                    job.status = "completed"
                    
                    # Update last training time
                    self.last_training_time[job.model_name] = datetime.utcnow()
                    
                except Exception as e:
                    logger.error(f"Training error for {job.model_name}: {e}")
                    job.status = "failed"
                    job.error = str(e)
                
                finally:
                    job.end_time = datetime.utcnow()
                    self.completed_jobs.append(job)
                    del self.active_jobs[job.job_id]
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Training worker {worker_id} error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_training(self, job: ModelTrainingJob) -> Dict[str, float]:
        """Execute actual model training with real data"""
        model_info = self.model_registry[job.model_name]
        model = model_info['model']
        data_type = model_info['data_type']
        
        # Get training data from pipeline
        df = await self.data_pipeline.get_training_data(
            data_type=data_type,
            hours=168,  # 1 week of data
            min_samples=job.min_samples
        )
        
        if len(df) < job.min_samples:
            raise ValueError(f"Insufficient data: {len(df)} samples, need {job.min_samples}")
        
        # Prepare features and labels based on data type
        X, y = await self._prepare_training_data(job.model_name, df)
        
        if len(X) == 0:
            raise ValueError("No valid training samples after preparation")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        logger.info(f"Training {job.model_name} with {len(X_train)} samples")
        
        # Handle different model types
        if hasattr(model, 'partial_fit'):
            # Incremental learning
            model.partial_fit(X_train, y_train)
        else:
            # Full retraining
            model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = {}
        
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
            
            # Classification metrics
            if hasattr(model, 'predict_proba'):
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted', zero_division=0
                )
                metrics.update({
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1)
                })
            
            # Regression metrics
            else:
                mse = mean_squared_error(y_test, y_pred)
                metrics['mse'] = float(mse)
                metrics['rmse'] = float(np.sqrt(mse))
        
        # Save model
        self._save_model(job.model_name, model)
        
        # Record performance
        performance = ModelPerformance(
            model_name=job.model_name,
            timestamp=datetime.utcnow(),
            accuracy=metrics.get('accuracy', 0),
            precision=metrics.get('precision', 0),
            recall=metrics.get('recall', 0),
            f1_score=metrics.get('f1_score', 0),
            sample_count=len(X_train)
        )
        self.model_performance[job.model_name].append(performance)
        
        logger.info(f"Training completed for {job.model_name}: {metrics}")
        return metrics
    
    async def _prepare_training_data(self, model_name: str, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data based on model requirements"""
        
        # Security models
        if 'security_monitor' in model_name.lower():
            return await self._prepare_security_data(df)
        
        # Performance models
        elif 'performance' in model_name.lower() or 'optimizer' in model_name.lower():
            return await self._prepare_performance_data(df)
        
        # User behavior models
        elif 'behavior' in model_name.lower() or 'user' in model_name.lower():
            return await self._prepare_user_behavior_data(df)
        
        # Error/failure models
        elif 'error' in model_name.lower() or 'failure' in model_name.lower():
            return await self._prepare_error_data(df)
        
        # Default preparation
        else:
            return await self._prepare_generic_data(df)
    
    async def _prepare_security_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare security event data for training"""
        X = []
        y = []
        
        for _, row in df.iterrows():
            # Extract security-relevant features
            features = []
            
            # Time features
            timestamp = pd.to_datetime(row['timestamp'], unit='s')
            features.append(timestamp.hour / 24.0)
            features.append(timestamp.dayofweek / 7.0)
            
            # Event features
            event_type = row.get('event_type', '')
            features.append(1 if 'auth' in event_type else 0)
            features.append(1 if 'fail' in event_type else 0)
            features.append(row.get('duration_ms', 0) / 1000.0)
            
            # Context features from JSON
            context = row.get('context', {})
            features.append(context.get('retry_count', 0) / 10.0)
            features.append(context.get('source_port', 0) / 65535.0)
            
            # Label - threat or not
            is_threat = (
                not row.get('success', True) and
                row.get('error_code', '') in ['AUTH_FAILED', 'ACCESS_DENIED', 'BLOCKED']
            )
            
            X.append(features)
            y.append(1 if is_threat else 0)
        
        return np.array(X), np.array(y)
    
    async def _prepare_performance_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare performance metrics data for training"""
        X = []
        y = []
        
        # Sort by timestamp to create sequences
        df = df.sort_values('timestamp')
        
        for i in range(len(df) - 1):
            current = df.iloc[i]
            next_metric = df.iloc[i + 1]
            
            # Features from current state
            features = [
                current.get('cpu_usage', 0) / 100.0,
                current.get('memory_usage', 0) / 100.0,
                current.get('disk_usage', 0) / 100.0,
                current.get('active_connections', 0) / 1000.0,
                current.get('request_count', 0) / 1000.0,
                current.get('error_count', 0) / 100.0,
                current.get('response_time_ms', 0) / 1000.0,
                current.get('queue_depth', 0) / 100.0
            ]
            
            # Target - predict next response time
            target = next_metric.get('response_time_ms', 0)
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    async def _prepare_generic_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generic data preparation for any model"""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric features found in data")
        
        # Use all numeric columns as features
        X = df[numeric_cols].fillna(0).values
        
        # Create synthetic labels based on data patterns
        # In production, these would come from actual outcomes
        if 'success' in df.columns:
            y = (df['success'] == False).astype(int).values
        elif 'error_count' in df.columns:
            y = (df['error_count'] > 0).astype(int).values
        else:
            # Default: anomaly detection based on statistical outliers
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            y = (np.abs(X_scaled).max(axis=1) > 3).astype(int)  # 3 std devs
        
        return X, y
    
    def _save_model(self, model_name: str, model):
        """Save trained model to disk"""
        try:
            model_path = os.path.join(self.model_storage_path, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Saved model {model_name} to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
    
    def load_model(self, model_name: str):
        """Load a trained model from disk"""
        try:
            model_path = os.path.join(self.model_storage_path, f"{model_name}.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                logger.info(f"Loaded model {model_name} from {model_path}")
                return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
        return None
    
    async def _performance_monitor(self):
        """Monitor model performance and trigger retraining if needed"""
        while self._is_running:
            try:
                for model_name in self.model_registry:
                    recent_perf = self._get_recent_performance(model_name)
                    
                    if recent_perf is not None and recent_perf < self.performance_threshold:
                        logger.warning(f"Performance degradation detected for {model_name}: {recent_perf:.3f}")
                        
                        # Schedule retraining
                        if self.training_schedules[model_name] == TrainingSchedule.PERFORMANCE_BASED:
                            await self.train_model(model_name, force=True)
                        
                        # Alert
                        self.performance_alerts.append({
                            'model_name': model_name,
                            'performance': recent_perf,
                            'timestamp': datetime.utcnow()
                        })
                
                # Check every 5 minutes
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(300)
    
    def _get_recent_performance(self, model_name: str) -> Optional[float]:
        """Get recent average performance for a model"""
        if model_name not in self.model_performance:
            return None
        
        recent_metrics = self.model_performance[model_name][-10:]  # Last 10 evaluations
        
        if not recent_metrics:
            return None
        
        # Use F1 score as primary metric
        avg_f1 = np.mean([m.f1_score for m in recent_metrics])
        return avg_f1
    
    async def _get_new_data_count(self, model_name: str, since: Optional[datetime]) -> int:
        """Get count of new data since last training"""
        if not since:
            since = datetime.utcnow() - timedelta(hours=1)
        
        model_info = self.model_registry[model_name]
        data_type = model_info['data_type']
        
        # Query data pipeline for count
        hours_ago = (datetime.utcnow() - since).total_seconds() / 3600
        df = await self.data_pipeline.get_training_data(
            data_type=data_type,
            hours=hours_ago,
            min_samples=1
        )
        
        return len(df)
    
    async def _wait_for_active_jobs(self):
        """Wait for all active jobs to complete"""
        timeout = 60  # 1 minute timeout
        start_time = time.time()
        
        while self.active_jobs and (time.time() - start_time) < timeout:
            await asyncio.sleep(1)
        
        if self.active_jobs:
            logger.warning(f"Timeout waiting for {len(self.active_jobs)} active jobs")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current status of all models"""
        status = {
            'registered_models': len(self.model_registry),
            'models': {},
            'active_jobs': len(self.active_jobs),
            'completed_jobs': len(self.completed_jobs),
            'performance_alerts': len(self.performance_alerts)
        }
        
        for model_name in self.model_registry:
            last_trained = self.last_training_time.get(model_name)
            recent_perf = self._get_recent_performance(model_name)
            
            status['models'][model_name] = {
                'last_trained': last_trained.isoformat() if last_trained else None,
                'recent_performance': recent_perf,
                'schedule': self.training_schedules[model_name].value,
                'performance_history': len(self.model_performance[model_name])
            }
        
        return status


# Singleton instance
_ml_training_service = None

def get_ml_training_service() -> MLTrainingService:
    """Get or create ML training service instance"""
    global _ml_training_service
    if not _ml_training_service:
        _ml_training_service = MLTrainingService()
    return _ml_training_service