"""
ML-Based Task Prioritization System for A2A Platform
Intelligent task prioritization using machine learning and contextual analysis
"""

import asyncio
import json
import logging
import time
import math
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.getLogger(__name__).warning("ML libraries not available for task prioritization")

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class TaskCategory(Enum):
    """Task categories for classification"""
    SYSTEM_MAINTENANCE = "system_maintenance"
    DATA_PROCESSING = "data_processing"
    USER_REQUEST = "user_request"
    INTEGRATION = "integration"
    MONITORING = "monitoring"
    SECURITY = "security"
    WORKFLOW = "workflow"
    ANALYSIS = "analysis"
    NOTIFICATION = "notification"
    BACKUP = "backup"


class UrgencyFactor(Enum):
    """Factors that influence task urgency"""
    TIME_SENSITIVE = "time_sensitive"
    BUSINESS_CRITICAL = "business_critical"
    DEPENDENCY_BLOCKING = "dependency_blocking"
    RESOURCE_INTENSIVE = "resource_intensive"
    USER_FACING = "user_facing"
    SECURITY_RELATED = "security_related"
    SYSTEM_HEALTH = "system_health"


@dataclass
class TaskContext:
    """Context information for task prioritization"""
    task_id: str
    task_type: str
    description: str
    requester_id: str
    created_at: float
    deadline: Optional[float] = None
    estimated_duration: Optional[float] = None
    required_resources: List[str] = None
    dependencies: List[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None

    # Dynamic context
    system_load: float = 0.0
    queue_length: int = 0
    requester_priority: float = 1.0
    similar_task_performance: float = 0.0
    business_impact_score: float = 0.0

    def __post_init__(self):
        if self.required_resources is None:
            self.required_resources = []
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PriorityScore:
    """Comprehensive priority scoring"""
    task_id: str
    final_score: float
    confidence: float
    base_priority: TaskPriority
    ml_adjustment: float
    urgency_multipliers: Dict[str, float]
    context_factors: Dict[str, float]
    reasoning: List[str]
    timestamp: float

    def __post_init__(self):
        if self.urgency_multipliers is None:
            self.urgency_multipliers = {}
        if self.context_factors is None:
            self.context_factors = {}
        if self.reasoning is None:
            self.reasoning = []


class MLPriorityModel:
    """Machine learning models for task prioritization"""

    def __init__(self):
        self.priority_regressor = None
        self.urgency_classifier = None
        self.duration_predictor = None
        self.impact_analyzer = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.label_encoders = {}

        # Training data
        self.training_history = []
        self.model_performance = {}
        self.last_training = 0.0
        self.training_interval = 3600.0  # Retrain every hour

        if ML_AVAILABLE:
            self.priority_regressor = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            self.urgency_classifier = RandomForestRegressor(
                n_estimators=50,
                random_state=42
            )
            self.duration_predictor = RandomForestRegressor(
                n_estimators=50,
                random_state=42
            )
            self.impact_analyzer = GradientBoostingRegressor(
                n_estimators=50,
                random_state=42
            )

        self.models_trained = False

    def extract_features(self, task_context: TaskContext) -> Optional[np.ndarray]:
        """Extract features for ML models"""
        if not ML_AVAILABLE:
            return None

        try:
            # Time-based features
            current_time = time.time()
            age_hours = (current_time - task_context.created_at) / 3600
            deadline_hours = ((task_context.deadline or current_time + 86400) - current_time) / 3600
            estimated_hours = (task_context.estimated_duration or 3600) / 3600

            # Categorical features (encoded)
            task_type_encoded = self._encode_categorical('task_type', task_context.task_type)
            requester_encoded = self._encode_categorical('requester_id', task_context.requester_id)

            # Resource and dependency features
            resource_count = len(task_context.required_resources)
            dependency_count = len(task_context.dependencies)
            tag_count = len(task_context.tags)

            # System context features
            system_load = task_context.system_load
            queue_length = task_context.queue_length
            requester_priority = task_context.requester_priority
            business_impact = task_context.business_impact_score
            similar_performance = task_context.similar_task_performance

            # Text-based features (simplified)
            description_length = len(task_context.description)
            description_complexity = len(task_context.description.split())

            # Priority keywords in description
            priority_keywords = ['urgent', 'critical', 'asap', 'emergency', 'immediately',
                               'high priority', 'important', 'deadline', 'blocking']
            keyword_score = sum(1 for kw in priority_keywords
                              if kw.lower() in task_context.description.lower())

            # Urgency indicators
            has_deadline = 1.0 if task_context.deadline else 0.0
            has_dependencies = 1.0 if task_context.dependencies else 0.0
            is_security_related = 1.0 if any('security' in tag.lower()
                                           for tag in task_context.tags) else 0.0
            is_user_facing = 1.0 if any('user' in tag.lower() or 'customer' in tag.lower()
                                      for tag in task_context.tags) else 0.0

            features = np.array([
                age_hours,
                deadline_hours,
                estimated_hours,
                task_type_encoded,
                requester_encoded,
                resource_count,
                dependency_count,
                tag_count,
                system_load,
                queue_length,
                requester_priority,
                business_impact,
                similar_performance,
                description_length,
                description_complexity,
                keyword_score,
                has_deadline,
                has_dependencies,
                is_security_related,
                is_user_facing
            ])

            return features

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def _encode_categorical(self, column: str, value: str) -> float:
        """Encode categorical values"""
        if column not in self.label_encoders:
            self.label_encoders[column] = LabelEncoder()
            # Initialize with common values
            if column == 'task_type':
                common_values = ['data_processing', 'user_request', 'system_maintenance',
                               'integration', 'monitoring', 'security']
            elif column == 'requester_id':
                common_values = ['system', 'user', 'admin', 'agent']
            else:
                common_values = [value]

            self.label_encoders[column].fit(common_values)

        try:
            return float(self.label_encoders[column].transform([value])[0])
        except ValueError:
            # Handle unseen values
            return 0.0

    def predict_priority_score(self, task_context: TaskContext) -> Tuple[float, float]:
        """Predict priority score using ML model"""
        if not ML_AVAILABLE or not self.models_trained:
            return 0.0, 0.0

        try:
            features = self.extract_features(task_context)
            if features is None:
                return 0.0, 0.0

            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Predict priority score
            priority_score = self.priority_regressor.predict(features_scaled)[0]

            # Calculate confidence based on model variance
            if hasattr(self.priority_regressor, 'predict_proba'):
                confidence = max(self.priority_regressor.predict_proba(features_scaled)[0])
            else:
                # Use prediction uncertainty as inverse confidence measure
                confidence = 1.0 / (1.0 + abs(priority_score - 3.0))  # Centered around medium priority

            return max(1.0, min(5.0, priority_score)), max(0.1, min(1.0, confidence))

        except Exception as e:
            logger.error(f"ML priority prediction error: {e}")
            return 0.0, 0.0

    def predict_urgency_factors(self, task_context: TaskContext) -> Dict[str, float]:
        """Predict urgency factors using ML"""
        if not ML_AVAILABLE or not self.models_trained:
            return {}

        try:
            features = self.extract_features(task_context)
            if features is None:
                return {}

            features_scaled = self.scaler.transform(features.reshape(1, -1))
            urgency_score = self.urgency_classifier.predict(features_scaled)[0]

            # Map urgency score to specific factors
            urgency_factors = {
                UrgencyFactor.TIME_SENSITIVE.value: min(1.0, urgency_score * 0.8) if task_context.deadline else 0.3,
                UrgencyFactor.BUSINESS_CRITICAL.value: min(1.0, task_context.business_impact_score * 0.7),
                UrgencyFactor.DEPENDENCY_BLOCKING.value: min(1.0, len(task_context.dependencies) * 0.2),
                UrgencyFactor.USER_FACING.value: 0.8 if any('user' in tag.lower() for tag in task_context.tags) else 0.2,
                UrgencyFactor.SECURITY_RELATED.value: 0.9 if any('security' in tag.lower() for tag in task_context.tags) else 0.1,
                UrgencyFactor.SYSTEM_HEALTH.value: min(1.0, task_context.system_load * 0.5)
            }

            return urgency_factors

        except Exception as e:
            logger.error(f"Urgency prediction error: {e}")
            return {}

    def predict_task_duration(self, task_context: TaskContext) -> float:
        """Predict task duration using ML"""
        if not ML_AVAILABLE or not self.models_trained:
            return task_context.estimated_duration or 3600.0  # 1 hour default

        try:
            features = self.extract_features(task_context)
            if features is None:
                return task_context.estimated_duration or 3600.0

            features_scaled = self.scaler.transform(features.reshape(1, -1))
            predicted_duration = self.duration_predictor.predict(features_scaled)[0]

            # Return duration in seconds, with reasonable bounds
            return max(60.0, min(86400.0, predicted_duration))  # 1 minute to 24 hours

        except Exception as e:
            logger.error(f"Duration prediction error: {e}")
            return task_context.estimated_duration or 3600.0

    async def train_models(self, training_data: List[Dict[str, Any]]):
        """Train ML models with historical task data"""
        if not ML_AVAILABLE or len(training_data) < 50:
            logger.info("Insufficient training data for ML models")
            return

        try:
            # Prepare training data
            X, y_priority, y_urgency, y_duration, y_impact = [], [], [], [], []

            for record in training_data:
                task_context = TaskContext(**record['task_context'])
                features = self.extract_features(task_context)

                if features is not None:
                    X.append(features)
                    y_priority.append(record.get('actual_priority', 3.0))
                    y_urgency.append(record.get('urgency_score', 0.5))
                    y_duration.append(record.get('actual_duration', 3600.0))
                    y_impact.append(record.get('business_impact', 0.5))

            if len(X) < 30:
                logger.info("Insufficient feature data for training")
                return

            X = np.array(X)
            y_priority = np.array(y_priority)
            y_urgency = np.array(y_urgency)
            y_duration = np.array(y_duration)
            y_impact = np.array(y_impact)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_p_train, y_p_test = train_test_split(
                X_scaled, y_priority, test_size=0.2, random_state=42
            )

            # Train models
            self.priority_regressor.fit(X_train, y_p_train)

            # Train other models with same split
            _, _, y_u_train, y_u_test = train_test_split(
                X_scaled, y_urgency, test_size=0.2, random_state=42
            )
            self.urgency_classifier.fit(X_train, y_u_train)

            _, _, y_d_train, y_d_test = train_test_split(
                X_scaled, y_duration, test_size=0.2, random_state=42
            )
            self.duration_predictor.fit(X_train, y_d_train)

            _, _, y_i_train, y_i_test = train_test_split(
                X_scaled, y_impact, test_size=0.2, random_state=42
            )
            self.impact_analyzer.fit(X_train, y_i_train)

            # Evaluate models
            priority_score = r2_score(y_p_test, self.priority_regressor.predict(X_test))
            urgency_score = r2_score(y_u_test, self.urgency_classifier.predict(X_test))
            duration_score = r2_score(y_d_test, self.duration_predictor.predict(X_test))
            impact_score = r2_score(y_i_test, self.impact_analyzer.predict(X_test))

            self.model_performance = {
                'priority_r2': priority_score,
                'urgency_r2': urgency_score,
                'duration_r2': duration_score,
                'impact_r2': impact_score,
                'training_samples': len(X),
                'last_trained': time.time()
            }

            self.models_trained = True
            self.last_training = time.time()

            logger.info(f"ML models trained successfully with {len(X)} samples")
            logger.info(f"Model performance - Priority: {priority_score:.3f}, "
                       f"Urgency: {urgency_score:.3f}, Duration: {duration_score:.3f}")

        except Exception as e:
            logger.error(f"Model training error: {e}")


class ContextualPrioritizer:
    """Contextual analysis for task prioritization"""

    def __init__(self):
        self.task_patterns = defaultdict(list)
        self.requester_profiles = defaultdict(dict)
        self.system_load_history = deque(maxlen=288)  # 24 hours of 5-minute intervals
        self.task_completion_history = deque(maxlen=1000)

    def analyze_context(self, task_context: TaskContext) -> Dict[str, float]:
        """Analyze contextual factors for prioritization"""

        context_factors = {}

        # Time-based context
        context_factors.update(self._analyze_temporal_context(task_context))

        # System load context
        context_factors.update(self._analyze_system_context(task_context))

        # Requester context
        context_factors.update(self._analyze_requester_context(task_context))

        # Task similarity context
        context_factors.update(self._analyze_similarity_context(task_context))

        # Resource availability context
        context_factors.update(self._analyze_resource_context(task_context))

        return context_factors

    def _analyze_temporal_context(self, task_context: TaskContext) -> Dict[str, float]:
        """Analyze time-based contextual factors"""

        current_time = time.time()
        current_hour = datetime.fromtimestamp(current_time).hour

        factors = {}

        # Age factor (older tasks get higher priority)
        age_hours = (current_time - task_context.created_at) / 3600
        factors['age_factor'] = min(2.0, 1.0 + (age_hours / 24))  # Max 2x after 24 hours

        # Deadline urgency
        if task_context.deadline:
            time_to_deadline = task_context.deadline - current_time
            hours_to_deadline = time_to_deadline / 3600

            if hours_to_deadline <= 0:
                factors['deadline_urgency'] = 5.0  # Overdue
            elif hours_to_deadline <= 1:
                factors['deadline_urgency'] = 3.0  # Very urgent
            elif hours_to_deadline <= 4:
                factors['deadline_urgency'] = 2.0  # Urgent
            elif hours_to_deadline <= 24:
                factors['deadline_urgency'] = 1.5  # Moderate urgency
            else:
                factors['deadline_urgency'] = 1.0  # Normal
        else:
            factors['deadline_urgency'] = 1.0

        # Business hours factor
        if 9 <= current_hour <= 17:  # Business hours
            factors['business_hours_factor'] = 1.2
        elif 6 <= current_hour <= 21:  # Extended hours
            factors['business_hours_factor'] = 1.0
        else:  # Night/early morning
            factors['business_hours_factor'] = 0.8

        return factors

    def _analyze_system_context(self, task_context: TaskContext) -> Dict[str, float]:
        """Analyze system load and performance context"""

        factors = {}

        # System load factor
        if task_context.system_load > 0.9:
            factors['system_load_factor'] = 0.7  # Reduce priority when system overloaded
        elif task_context.system_load > 0.7:
            factors['system_load_factor'] = 0.9
        else:
            factors['system_load_factor'] = 1.0

        # Queue length factor
        if task_context.queue_length > 50:
            factors['queue_factor'] = 0.8  # Reduce when queue is long
        elif task_context.queue_length > 20:
            factors['queue_factor'] = 0.9
        else:
            factors['queue_factor'] = 1.0

        # Historical system performance
        if self.system_load_history:
            avg_load = np.mean(list(self.system_load_history)[-12:])  # Last hour
            if avg_load > 0.8:
                factors['system_trend_factor'] = 0.8
            else:
                factors['system_trend_factor'] = 1.0
        else:
            factors['system_trend_factor'] = 1.0

        return factors

    def _analyze_requester_context(self, task_context: TaskContext) -> Dict[str, float]:
        """Analyze requester-specific context"""

        factors = {}

        # Base requester priority
        factors['requester_priority_factor'] = task_context.requester_priority

        # Requester type analysis
        requester_id = task_context.requester_id.lower()
        if 'admin' in requester_id or 'system' in requester_id:
            factors['requester_type_factor'] = 1.3
        elif 'agent' in requester_id:
            factors['requester_type_factor'] = 1.1
        elif 'user' in requester_id:
            factors['requester_type_factor'] = 1.0
        else:
            factors['requester_type_factor'] = 0.9

        # Historical success rate of requester's tasks
        if task_context.requester_id in self.requester_profiles:
            profile = self.requester_profiles[task_context.requester_id]
            success_rate = profile.get('success_rate', 0.8)
            factors['requester_success_factor'] = 0.8 + (success_rate * 0.4)  # 0.8 to 1.2 range
        else:
            factors['requester_success_factor'] = 1.0

        return factors

    def _analyze_similarity_context(self, task_context: TaskContext) -> Dict[str, float]:
        """Analyze similarity to previous tasks"""

        factors = {}

        # Similar task performance
        if task_context.similar_task_performance > 0:
            factors['similarity_factor'] = 0.8 + (task_context.similar_task_performance * 0.4)
        else:
            factors['similarity_factor'] = 1.0

        # Task type frequency (reduce priority for very common tasks)
        task_type_count = len([p for p in self.task_patterns.get(task_context.task_type, [])])
        if task_type_count > 100:  # Very common
            factors['frequency_factor'] = 0.9
        elif task_type_count > 50:  # Common
            factors['frequency_factor'] = 0.95
        else:  # Uncommon
            factors['frequency_factor'] = 1.0

        return factors

    def _analyze_resource_context(self, task_context: TaskContext) -> Dict[str, float]:
        """Analyze resource availability context"""

        factors = {}

        # Resource requirement complexity
        resource_count = len(task_context.required_resources)
        if resource_count == 0:
            factors['resource_factor'] = 1.1  # No special resources needed
        elif resource_count <= 2:
            factors['resource_factor'] = 1.0  # Normal
        elif resource_count <= 5:
            factors['resource_factor'] = 0.9  # Many resources
        else:
            factors['resource_factor'] = 0.8  # Very resource intensive

        # Dependency factor
        dependency_count = len(task_context.dependencies)
        if dependency_count == 0:
            factors['dependency_factor'] = 1.0  # Independent task
        elif dependency_count <= 2:
            factors['dependency_factor'] = 0.95  # Few dependencies
        else:
            factors['dependency_factor'] = 0.85  # Many dependencies

        return factors

    def update_patterns(self, task_context: TaskContext, outcome: Dict[str, Any]):
        """Update patterns based on task completion"""

        # Update task patterns
        self.task_patterns[task_context.task_type].append({
            'context': task_context,
            'outcome': outcome,
            'timestamp': time.time()
        })

        # Update requester profile
        requester_id = task_context.requester_id
        if requester_id not in self.requester_profiles:
            self.requester_profiles[requester_id] = {
                'total_tasks': 0,
                'successful_tasks': 0,
                'avg_duration': 0,
                'success_rate': 0.8
            }

        profile = self.requester_profiles[requester_id]
        profile['total_tasks'] += 1

        if outcome.get('success', False):
            profile['successful_tasks'] += 1

        profile['success_rate'] = profile['successful_tasks'] / profile['total_tasks']

        # Update completion history
        self.task_completion_history.append({
            'task_context': asdict(task_context),
            'outcome': outcome,
            'timestamp': time.time()
        })

    def update_system_load(self, system_load: float):
        """Update system load history"""
        self.system_load_history.append({
            'load': system_load,
            'timestamp': time.time()
        })


class MLTaskPrioritizer:
    """Main ML-based task prioritization system"""

    def __init__(self):
        self.ml_model = MLPriorityModel()
        self.contextual_analyzer = ContextualPrioritizer()

        # Priority calculation weights
        self.weights = {
            'base_priority': 0.3,
            'ml_prediction': 0.25,
            'urgency_factors': 0.2,
            'context_factors': 0.15,
            'business_impact': 0.1
        }

        # Performance tracking
        self.prioritization_history = deque(maxlen=1000)
        self.accuracy_metrics = defaultdict(list)

    async def prioritize_task(self, task_context: TaskContext) -> PriorityScore:
        """Calculate comprehensive priority score for a task"""

        start_time = time.time()
        reasoning = []

        # 1. Determine base priority from task characteristics
        base_priority = self._determine_base_priority(task_context)
        reasoning.append(f"Base priority: {base_priority.name}")

        # 2. ML-based priority adjustment
        ml_score, ml_confidence = self.ml_model.predict_priority_score(task_context)
        ml_adjustment = (ml_score - base_priority.value) * ml_confidence
        reasoning.append(f"ML adjustment: {ml_adjustment:+.2f} (confidence: {ml_confidence:.2f})")

        # 3. Urgency factors analysis
        urgency_factors = self.ml_model.predict_urgency_factors(task_context)
        urgency_multiplier = self._calculate_urgency_multiplier(urgency_factors)
        reasoning.append(f"Urgency multiplier: {urgency_multiplier:.2f}")

        # 4. Contextual analysis
        context_factors = self.contextual_analyzer.analyze_context(task_context)
        context_multiplier = self._calculate_context_multiplier(context_factors)
        reasoning.append(f"Context multiplier: {context_multiplier:.2f}")

        # 5. Business impact consideration
        business_impact = task_context.business_impact_score
        impact_multiplier = 1.0 + (business_impact * 0.5)  # Up to 1.5x for max impact
        reasoning.append(f"Business impact multiplier: {impact_multiplier:.2f}")

        # 6. Calculate final score
        final_score = (
            base_priority.value * self.weights['base_priority'] +
            (base_priority.value + ml_adjustment) * self.weights['ml_prediction'] +
            base_priority.value * urgency_multiplier * self.weights['urgency_factors'] +
            base_priority.value * context_multiplier * self.weights['context_factors'] +
            base_priority.value * impact_multiplier * self.weights['business_impact']
        )

        # Normalize to 1-5 range
        final_score = max(1.0, min(5.0, final_score))

        # Calculate overall confidence
        confidence = (ml_confidence + 0.8) / 2  # Average with base confidence

        priority_score = PriorityScore(
            task_id=task_context.task_id,
            final_score=final_score,
            confidence=confidence,
            base_priority=base_priority,
            ml_adjustment=ml_adjustment,
            urgency_multipliers=urgency_factors,
            context_factors=context_factors,
            reasoning=reasoning,
            timestamp=time.time()
        )

        # Record for analysis
        self.prioritization_history.append(priority_score)

        processing_time = time.time() - start_time
        logger.debug(f"Task {task_context.task_id} prioritized: score={final_score:.2f}, "
                    f"confidence={confidence:.2f}, time={processing_time:.3f}s")

        return priority_score

    def _determine_base_priority(self, task_context: TaskContext) -> TaskPriority:
        """Determine base priority from task characteristics"""

        # Priority keywords in description/tags
        high_priority_keywords = ['critical', 'urgent', 'emergency', 'asap', 'blocking']
        medium_priority_keywords = ['important', 'needed', 'required', 'deadline']

        description_lower = task_context.description.lower()
        all_tags = ' '.join(task_context.tags).lower()
        combined_text = f"{description_lower} {all_tags}"

        # Check for priority keywords
        if any(kw in combined_text for kw in high_priority_keywords):
            return TaskPriority.HIGH
        elif any(kw in combined_text for kw in medium_priority_keywords):
            return TaskPriority.MEDIUM

        # Task type based priority
        task_type_priorities = {
            'security': TaskPriority.HIGH,
            'system_maintenance': TaskPriority.MEDIUM,
            'user_request': TaskPriority.MEDIUM,
            'data_processing': TaskPriority.MEDIUM,
            'monitoring': TaskPriority.LOW,
            'backup': TaskPriority.LOW
        }

        task_type_lower = task_context.task_type.lower()
        for task_type, priority in task_type_priorities.items():
            if task_type in task_type_lower:
                return priority

        # Deadline-based priority
        if task_context.deadline:
            time_to_deadline = task_context.deadline - time.time()
            hours_to_deadline = time_to_deadline / 3600

            if hours_to_deadline <= 1:
                return TaskPriority.URGENT
            elif hours_to_deadline <= 4:
                return TaskPriority.HIGH
            elif hours_to_deadline <= 24:
                return TaskPriority.MEDIUM

        # Default priority
        return TaskPriority.MEDIUM

    def _calculate_urgency_multiplier(self, urgency_factors: Dict[str, float]) -> float:
        """Calculate urgency multiplier from factors"""

        if not urgency_factors:
            return 1.0

        # Weight different urgency factors
        factor_weights = {
            UrgencyFactor.TIME_SENSITIVE.value: 0.3,
            UrgencyFactor.BUSINESS_CRITICAL.value: 0.25,
            UrgencyFactor.DEPENDENCY_BLOCKING.value: 0.2,
            UrgencyFactor.SECURITY_RELATED.value: 0.15,
            UrgencyFactor.USER_FACING.value: 0.1
        }

        weighted_score = 0.0
        total_weight = 0.0

        for factor, score in urgency_factors.items():
            weight = factor_weights.get(factor, 0.05)
            weighted_score += score * weight
            total_weight += weight

        if total_weight > 0:
            urgency_score = weighted_score / total_weight
            return 1.0 + (urgency_score * 0.8)  # Up to 1.8x multiplier

        return 1.0

    def _calculate_context_multiplier(self, context_factors: Dict[str, float]) -> float:
        """Calculate context multiplier from factors"""

        if not context_factors:
            return 1.0

        # Apply context factors multiplicatively
        multiplier = 1.0

        for factor_name, factor_value in context_factors.items():
            # Limit individual factor impact
            limited_factor = max(0.5, min(2.0, factor_value))

            # Apply with diminishing returns
            if factor_value > 1.0:
                contribution = 1.0 + ((limited_factor - 1.0) * 0.5)
            else:
                contribution = limited_factor

            multiplier *= contribution

        # Limit overall multiplier
        return max(0.3, min(3.0, multiplier))

    async def batch_prioritize(self, task_contexts: List[TaskContext]) -> List[PriorityScore]:
        """Prioritize multiple tasks efficiently"""

        priority_scores = []

        # Process in parallel batches
        batch_size = 10
        for i in range(0, len(task_contexts), batch_size):
            batch = task_contexts[i:i + batch_size]

            # Create tasks for parallel processing
            tasks = [self.prioritize_task(task_context) for task_context in batch]
            batch_scores = await asyncio.gather(*tasks)

            priority_scores.extend(batch_scores)

        return priority_scores

    def update_task_outcome(self,
                           task_id: str,
                           actual_priority: TaskPriority,
                           actual_duration: float,
                           success: bool,
                           business_impact: float):
        """Update with actual task outcome for learning"""

        # Find the original prioritization
        original_score = None
        for score in self.prioritization_history:
            if score.task_id == task_id:
                original_score = score
                break

        if not original_score:
            logger.warning(f"No prioritization record found for task {task_id}")
            return

        # Calculate accuracy metrics
        priority_error = abs(original_score.final_score - actual_priority.value)
        self.accuracy_metrics['priority_error'].append(priority_error)

        # Record outcome for model training
        training_record = {
            'task_context': {
                'task_id': task_id,
                'task_type': 'unknown',  # Would need to store original context
                'description': '',
                'requester_id': '',
                'created_at': original_score.timestamp,
                'business_impact_score': business_impact
            },
            'actual_priority': actual_priority.value,
            'actual_duration': actual_duration,
            'success': success,
            'business_impact': business_impact,
            'urgency_score': sum(original_score.urgency_multipliers.values()) / max(1, len(original_score.urgency_multipliers))
        }

        self.ml_model.training_history.append(training_record)

        # Trigger retraining if enough new data
        if (len(self.ml_model.training_history) >= 100 and
            time.time() - self.ml_model.last_training > self.ml_model.training_interval):
            asyncio.create_task(self._retrain_models())

    async def _retrain_models(self):
        """Retrain ML models with updated data"""
        try:
            await self.ml_model.train_models(self.ml_model.training_history)
            logger.info("ML prioritization models retrained successfully")
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""

        metrics = {
            'total_tasks_prioritized': len(self.prioritization_history),
            'ml_models_trained': self.ml_model.models_trained,
            'model_performance': self.ml_model.model_performance,
            'last_training': self.ml_model.last_training,
            'ml_available': ML_AVAILABLE
        }

        # Accuracy metrics
        if self.accuracy_metrics['priority_error']:
            metrics['average_priority_error'] = np.mean(self.accuracy_metrics['priority_error'])
            metrics['priority_accuracy_trend'] = self.accuracy_metrics['priority_error'][-20:]  # Last 20

        # Priority distribution
        if self.prioritization_history:
            scores = [score.final_score for score in self.prioritization_history[-100:]]
            metrics['priority_distribution'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }

            # Confidence distribution
            confidences = [score.confidence for score in self.prioritization_history[-100:]]
            metrics['confidence_distribution'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences)
            }

        return metrics


# Global instance
ml_task_prioritizer = None

async def initialize_ml_prioritizer():
    """Initialize the global ML task prioritizer"""
    global ml_task_prioritizer
    ml_task_prioritizer = MLTaskPrioritizer()
    logger.info("ML task prioritizer initialized")

def get_ml_prioritizer() -> Optional[MLTaskPrioritizer]:
    """Get the global ML task prioritizer instance"""
    return ml_task_prioritizer
