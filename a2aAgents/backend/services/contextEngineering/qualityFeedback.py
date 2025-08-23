"""
Quality and Feedback System for Context Engineering
Implements comprehensive monitoring, metrics collection, and continuous improvement
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import logging
from enum import Enum
from prometheus_client import Counter, Histogram, Gauge, Summary

logger = logging.getLogger(__name__)


# Prometheus metrics for quality monitoring
quality_score_histogram = Histogram(
    'context_quality_score',
    'Distribution of context quality scores',
    ['dimension', 'agent']
)

feedback_latency_summary = Summary(
    'feedback_processing_latency_seconds',
    'Latency of feedback processing'
)

improvement_success_rate = Gauge(
    'context_improvement_success_rate',
    'Success rate of context improvements',
    ['improvement_type']
)

ab_test_results = Counter(
    'ab_test_results_total',
    'Results of A/B tests',
    ['test_id', 'variant', 'outcome']
)

model_update_counter = Counter(
    'model_updates_total',
    'Number of model updates',
    ['model_type', 'update_reason']
)

cost_metrics = Gauge(
    'context_processing_cost',
    'Cost metrics for context processing',
    ['operation', 'resource']
)


class FeedbackType(Enum):
    """Types of feedback in the system"""
    QUALITY = "quality"
    PERFORMANCE = "performance"
    USER = "user"
    AUTOMATED = "automated"
    ERROR = "error"


class ImprovementStrategy(Enum):
    """Strategies for context improvement"""
    TEMPLATE_REFINEMENT = "template_refinement"
    MODEL_RETRAINING = "model_retraining"
    PARAMETER_TUNING = "parameter_tuning"
    PROCESS_OPTIMIZATION = "process_optimization"


@dataclass
class QualityMetric:
    """Individual quality metric"""
    name: str
    value: float
    threshold: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passes_threshold(self) -> bool:
        return self.value >= self.threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passes": self.passes_threshold,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class FeedbackItem:
    """Feedback item from any source"""
    feedback_id: str
    feedback_type: FeedbackType
    source: str  # agent_id, user_id, or system component
    target: str  # what the feedback is about
    content: Dict[str, Any]
    timestamp: datetime
    processed: bool = False
    actions_taken: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "source": self.source,
            "target": self.target,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "processed": self.processed,
            "actions_taken": self.actions_taken
        }


@dataclass
class ABTestConfig:
    """Configuration for A/B testing"""
    test_id: str
    name: str
    variants: Dict[str, Any]  # variant_name -> configuration
    metrics: List[str]
    allocation: Dict[str, float]  # variant_name -> percentage
    start_time: datetime
    end_time: Optional[datetime] = None
    minimum_samples: int = 100
    confidence_level: float = 0.95
    
    def is_active(self) -> bool:
        now = datetime.now()
        return (
            self.start_time <= now and 
            (self.end_time is None or now <= self.end_time)
        )


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    metric_name: str
    baseline_value: float
    variance: float
    sample_count: int
    last_updated: datetime
    
    def is_anomaly(self, value: float, threshold_std: float = 3.0) -> bool:
        """Check if value is anomalous compared to baseline"""
        if self.variance == 0:
            return abs(value - self.baseline_value) > 0.1
        
        z_score = abs(value - self.baseline_value) / (self.variance ** 0.5)
        return z_score > threshold_std


class QualityMonitor:
    """Monitors quality metrics across the system"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.alert_thresholds: Dict[str, float] = {}
        self.alert_callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    async def record_metric(self, metric: QualityMetric):
        """Record a quality metric"""
        # Add to buffer
        self.metrics_buffer[metric.name].append(metric)
        
        # Record in Prometheus
        quality_score_histogram.labels(
            dimension=metric.name,
            agent=metric.metadata.get("agent", "unknown")
        ).observe(metric.value)
        
        # Check for alerts
        await self._check_alerts(metric)
        
        # Update baseline if needed
        if len(self.metrics_buffer[metric.name]) >= 100:
            await self._update_baseline(metric.name)
    
    async def _check_alerts(self, metric: QualityMetric):
        """Check if metric triggers any alerts"""
        # Check threshold alerts
        if metric.name in self.alert_thresholds:
            if metric.value < self.alert_thresholds[metric.name]:
                await self._trigger_alert(
                    metric.name,
                    f"Metric below threshold: {metric.value} < {self.alert_thresholds[metric.name]}"
                )
        
        # Check anomaly alerts
        if metric.name in self.baselines:
            baseline = self.baselines[metric.name]
            if baseline.is_anomaly(metric.value):
                await self._trigger_alert(
                    metric.name,
                    f"Anomalous value detected: {metric.value} (baseline: {baseline.baseline_value})"
                )
    
    async def _trigger_alert(self, metric_name: str, message: str):
        """Trigger alert callbacks"""
        logger.warning(f"Quality alert for {metric_name}: {message}")
        
        for callback in self.alert_callbacks[metric_name]:
            try:
                await callback(metric_name, message)
            except Exception as e:
                logger.error(f"Alert callback failed: {str(e)}")
    
    async def _update_baseline(self, metric_name: str):
        """Update performance baseline for a metric"""
        metrics = list(self.metrics_buffer[metric_name])
        values = [m.value for m in metrics]
        
        if values:
            self.baselines[metric_name] = PerformanceBaseline(
                metric_name=metric_name,
                baseline_value=np.mean(values),
                variance=np.var(values),
                sample_count=len(values),
                last_updated=datetime.now()
            )
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        if metric_name not in self.metrics_buffer:
            return {"error": "Metric not found"}
        
        metrics = list(self.metrics_buffer[metric_name])
        values = [m.value for m in metrics]
        
        if not values:
            return {"error": "No data available"}
        
        return {
            "metric_name": metric_name,
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "percentiles": {
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "p95": float(np.percentile(values, 95))
            },
            "recent_trend": self._calculate_trend(values[-20:]) if len(values) >= 20 else None
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.001:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"


class FeedbackLoop:
    """Manages feedback collection and processing"""
    
    def __init__(self):
        self.feedback_queue: asyncio.Queue = asyncio.Queue()
        self.feedback_history: Dict[str, List[FeedbackItem]] = defaultdict(list)
        self.improvement_strategies: Dict[str, ImprovementStrategy] = {}
        self.processing_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start feedback processing loop"""
        if self.processing_task is None:
            self.processing_task = asyncio.create_task(self._process_feedback_loop())
    
    async def stop(self):
        """Stop feedback processing loop"""
        if self.processing_task:
            self.processing_task.cancel()
            self.processing_task = None
    
    async def submit_feedback(self, feedback: FeedbackItem):
        """Submit feedback for processing"""
        await self.feedback_queue.put(feedback)
        self.feedback_history[feedback.target].append(feedback)
    
    async def _process_feedback_loop(self):
        """Main feedback processing loop"""
        while True:
            try:
                # Get feedback item
                feedback = await self.feedback_queue.get()
                
                # Process feedback
                with feedback_latency_summary.time():
                    await self._process_feedback(feedback)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Feedback processing error: {str(e)}")
    
    async def _process_feedback(self, feedback: FeedbackItem):
        """Process individual feedback item"""
        logger.info(f"Processing feedback {feedback.feedback_id} of type {feedback.feedback_type}")
        
        # Route based on feedback type
        if feedback.feedback_type == FeedbackType.QUALITY:
            await self._process_quality_feedback(feedback)
        elif feedback.feedback_type == FeedbackType.PERFORMANCE:
            await self._process_performance_feedback(feedback)
        elif feedback.feedback_type == FeedbackType.USER:
            await self._process_user_feedback(feedback)
        elif feedback.feedback_type == FeedbackType.ERROR:
            await self._process_error_feedback(feedback)
        
        feedback.processed = True
    
    async def _process_quality_feedback(self, feedback: FeedbackItem):
        """Process quality-related feedback"""
        quality_data = feedback.content
        
        # Identify improvement areas
        improvements = []
        
        if quality_data.get("coherence", 1.0) < 0.7:
            improvements.append("improve_coherence")
        
        if quality_data.get("completeness", 1.0) < 0.8:
            improvements.append("improve_completeness")
        
        if quality_data.get("bias_level", 0.0) > 0.3:
            improvements.append("reduce_bias")
        
        # Take actions
        for improvement in improvements:
            action = await self._take_improvement_action(
                feedback.target, improvement, quality_data
            )
            feedback.actions_taken.append(action)
        
        # Update improvement success metrics
        if improvements:
            success_rate = len(feedback.actions_taken) / len(improvements)
            improvement_success_rate.labels(
                improvement_type="quality"
            ).set(success_rate)
    
    async def _process_performance_feedback(self, feedback: FeedbackItem):
        """Process performance-related feedback"""
        perf_data = feedback.content
        
        # Check for performance degradation
        if perf_data.get("response_time", 0) > 1000:  # 1 second
            action = await self._take_improvement_action(
                feedback.target, "optimize_response_time", perf_data
            )
            feedback.actions_taken.append(action)
        
        if perf_data.get("memory_usage", 0) > 1024:  # 1GB
            action = await self._take_improvement_action(
                feedback.target, "reduce_memory", perf_data
            )
            feedback.actions_taken.append(action)
    
    async def _process_user_feedback(self, feedback: FeedbackItem):
        """Process user-submitted feedback"""
        # Analyze user feedback sentiment and content
        user_data = feedback.content
        
        # Extract actionable insights
        if "suggestion" in user_data:
            # Queue for human review
            feedback.actions_taken.append("queued_for_review")
        
        if user_data.get("rating", 5) < 3:
            # Low rating - investigate
            feedback.actions_taken.append("investigation_triggered")
    
    async def _process_error_feedback(self, feedback: FeedbackItem):
        """Process error-related feedback"""
        error_data = feedback.content
        
        # Categorize error
        error_type = error_data.get("error_type", "unknown")
        
        if error_type == "parsing_error":
            action = await self._take_improvement_action(
                feedback.target, "fix_parsing", error_data
            )
            feedback.actions_taken.append(action)
        
        elif error_type == "timeout":
            action = await self._take_improvement_action(
                feedback.target, "increase_timeout", error_data
            )
            feedback.actions_taken.append(action)
    
    async def _take_improvement_action(
        self, 
        target: str, 
        improvement: str, 
        data: Dict[str, Any]
    ) -> str:
        """Take specific improvement action"""
        logger.info(f"Taking improvement action: {improvement} for {target}")
        
        # Map improvements to strategies
        strategy_map = {
            "improve_coherence": ImprovementStrategy.TEMPLATE_REFINEMENT,
            "improve_completeness": ImprovementStrategy.TEMPLATE_REFINEMENT,
            "reduce_bias": ImprovementStrategy.MODEL_RETRAINING,
            "optimize_response_time": ImprovementStrategy.PROCESS_OPTIMIZATION,
            "reduce_memory": ImprovementStrategy.PARAMETER_TUNING,
            "fix_parsing": ImprovementStrategy.MODEL_RETRAINING,
            "increase_timeout": ImprovementStrategy.PARAMETER_TUNING
        }
        
        strategy = strategy_map.get(improvement, ImprovementStrategy.PARAMETER_TUNING)
        
        # Execute strategy
        if strategy == ImprovementStrategy.TEMPLATE_REFINEMENT:
            return await self._refine_template(target, data)
        elif strategy == ImprovementStrategy.MODEL_RETRAINING:
            return await self._trigger_model_update(target, improvement, data)
        elif strategy == ImprovementStrategy.PARAMETER_TUNING:
            return await self._tune_parameters(target, improvement, data)
        elif strategy == ImprovementStrategy.PROCESS_OPTIMIZATION:
            return await self._optimize_process(target, data)
        
        return f"Applied {strategy.value}"
    
    async def _refine_template(self, target: str, data: Dict[str, Any]) -> str:
        """Refine context templates"""
        # In production, this would update template configurations
        logger.info(f"Refining template for {target}")
        return "template_refined"
    
    async def _trigger_model_update(self, target: str, reason: str, data: Dict[str, Any]) -> str:
        """Trigger model update/retraining"""
        model_update_counter.labels(
            model_type=target,
            update_reason=reason
        ).inc()
        
        # In production, this would queue retraining job
        logger.info(f"Queued model update for {target}: {reason}")
        return "model_update_queued"
    
    async def _tune_parameters(self, target: str, parameter: str, data: Dict[str, Any]) -> str:
        """Tune system parameters"""
        # In production, this would adjust configuration
        logger.info(f"Tuning {parameter} for {target}")
        return f"parameter_{parameter}_tuned"
    
    async def _optimize_process(self, target: str, data: Dict[str, Any]) -> str:
        """Optimize process flow"""
        # In production, this would adjust process configuration
        logger.info(f"Optimizing process for {target}")
        return "process_optimized"
    
    def get_feedback_summary(self, target: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of feedback for a target"""
        if target:
            feedback_items = self.feedback_history.get(target, [])
        else:
            feedback_items = [
                item for items in self.feedback_history.values() 
                for item in items
            ]
        
        if not feedback_items:
            return {"message": "No feedback available"}
        
        # Analyze feedback
        by_type = defaultdict(int)
        actions_taken = defaultdict(int)
        processed_count = 0
        
        for item in feedback_items:
            by_type[item.feedback_type.value] += 1
            if item.processed:
                processed_count += 1
            for action in item.actions_taken:
                actions_taken[action] += 1
        
        return {
            "total_feedback": len(feedback_items),
            "processed": processed_count,
            "by_type": dict(by_type),
            "actions_taken": dict(actions_taken),
            "processing_rate": processed_count / len(feedback_items) if feedback_items else 0
        }


class ABTestManager:
    """Manages A/B testing for continuous improvement"""
    
    def __init__(self):
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.assignment_cache: Dict[str, str] = {}  # entity_id -> variant
    
    def create_test(self, config: ABTestConfig):
        """Create a new A/B test"""
        self.active_tests[config.test_id] = config
        self.test_results[config.test_id] = {
            "variants": {v: {"count": 0, "metrics": defaultdict(list)} 
                        for v in config.variants},
            "start_time": config.start_time.isoformat(),
            "status": "active"
        }
    
    def get_variant(self, test_id: str, entity_id: str) -> Optional[str]:
        """Get variant assignment for an entity"""
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        if not test.is_active():
            return None
        
        # Check cache
        cache_key = f"{test_id}:{entity_id}"
        if cache_key in self.assignment_cache:
            return self.assignment_cache[cache_key]
        
        # Assign variant based on hash
        hash_value = hash(entity_id) % 100
        cumulative = 0.0
        
        for variant, allocation in test.allocation.items():
            cumulative += allocation * 100
            if hash_value < cumulative:
                self.assignment_cache[cache_key] = variant
                return variant
        
        # Default to first variant
        variant = list(test.variants.keys())[0]
        self.assignment_cache[cache_key] = variant
        return variant
    
    async def record_metric(
        self, 
        test_id: str, 
        entity_id: str, 
        metric_name: str, 
        value: float
    ):
        """Record metric for A/B test"""
        variant = self.get_variant(test_id, entity_id)
        if not variant:
            return
        
        # Record in results
        self.test_results[test_id]["variants"][variant]["count"] += 1
        self.test_results[test_id]["variants"][variant]["metrics"][metric_name].append(value)
        
        # Record in Prometheus
        ab_test_results.labels(
            test_id=test_id,
            variant=variant,
            outcome=metric_name
        ).inc()
        
        # Check if test can be concluded
        await self._check_test_conclusion(test_id)
    
    async def _check_test_conclusion(self, test_id: str):
        """Check if test has enough data to conclude"""
        test = self.active_tests.get(test_id)
        if not test:
            return
        
        results = self.test_results[test_id]
        
        # Check minimum samples
        min_samples = min(
            results["variants"][v]["count"] 
            for v in test.variants
        )
        
        if min_samples < test.minimum_samples:
            return
        
        # Perform statistical analysis
        analysis = await self._analyze_test_results(test, results)
        
        if analysis["conclusive"]:
            await self._conclude_test(test_id, analysis)
    
    async def _analyze_test_results(
        self, 
        test: ABTestConfig, 
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform statistical analysis on test results"""
        from scipy import stats
        
        analysis = {
            "conclusive": False,
            "winner": None,
            "confidence": 0.0,
            "metrics": {}
        }
        
        # Analyze each metric
        for metric in test.metrics:
            metric_analysis = {}
            
            # Get data for each variant
            variant_data = {}
            for variant in test.variants:
                values = results["variants"][variant]["metrics"].get(metric, [])
                if values:
                    variant_data[variant] = values
            
            if len(variant_data) < 2:
                continue
            
            # Perform t-test between variants
            variants = list(variant_data.keys())
            if len(variants) == 2:
                # Two-sample t-test
                statistic, p_value = stats.ttest_ind(
                    variant_data[variants[0]],
                    variant_data[variants[1]]
                )
                
                # Determine winner
                mean_0 = np.mean(variant_data[variants[0]])
                mean_1 = np.mean(variant_data[variants[1]])
                
                if p_value < (1 - test.confidence_level):
                    metric_analysis = {
                        "significant": True,
                        "p_value": p_value,
                        "winner": variants[0] if mean_0 > mean_1 else variants[1],
                        "improvement": abs(mean_1 - mean_0) / mean_0 if mean_0 != 0 else 0
                    }
                    
                    if not analysis["winner"]:
                        analysis["winner"] = metric_analysis["winner"]
                        analysis["confidence"] = 1 - p_value
                        analysis["conclusive"] = True
            
            analysis["metrics"][metric] = metric_analysis
        
        return analysis
    
    async def _conclude_test(self, test_id: str, analysis: Dict[str, Any]):
        """Conclude A/B test and apply winning variant"""
        test = self.active_tests[test_id]
        test.end_time = datetime.now()
        
        self.test_results[test_id]["status"] = "completed"
        self.test_results[test_id]["analysis"] = analysis
        self.test_results[test_id]["end_time"] = test.end_time.isoformat()
        
        logger.info(
            f"A/B test {test_id} concluded. "
            f"Winner: {analysis['winner']} with {analysis['confidence']:.2%} confidence"
        )
        
        # Apply winning configuration
        if analysis["winner"]:
            winning_config = test.variants[analysis["winner"]]
            # In production, this would update system configuration
            logger.info(f"Applying winning configuration: {winning_config}")
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get current status of an A/B test"""
        if test_id not in self.active_tests:
            return {"error": "Test not found"}
        
        test = self.active_tests[test_id]
        results = self.test_results[test_id]
        
        return {
            "test_id": test_id,
            "name": test.name,
            "status": results["status"],
            "variants": {
                v: {
                    "count": results["variants"][v]["count"],
                    "metrics": {
                        m: {
                            "mean": float(np.mean(values)) if values else 0,
                            "std": float(np.std(values)) if values else 0
                        }
                        for m, values in results["variants"][v]["metrics"].items()
                    }
                }
                for v in test.variants
            },
            "duration": (
                (test.end_time or datetime.now()) - test.start_time
            ).total_seconds()
        }


class CostOptimizer:
    """Optimizes costs for context processing"""
    
    def __init__(self):
        self.cost_tracking: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.resource_limits: Dict[str, float] = {}
        self.optimization_rules: List[Dict[str, Any]] = []
    
    async def track_cost(
        self, 
        operation: str, 
        resource: str, 
        amount: float, 
        unit_cost: float
    ):
        """Track cost for an operation"""
        cost = amount * unit_cost
        self.cost_tracking[operation][resource] += cost
        
        # Update Prometheus metric
        cost_metrics.labels(
            operation=operation,
            resource=resource
        ).set(cost)
        
        # Check limits
        await self._check_resource_limits(resource, cost)
    
    async def _check_resource_limits(self, resource: str, cost: float):
        """Check if resource limits are exceeded"""
        if resource in self.resource_limits:
            total_cost = sum(
                costs[resource] 
                for costs in self.cost_tracking.values()
            )
            
            if total_cost > self.resource_limits[resource]:
                logger.warning(
                    f"Resource limit exceeded for {resource}: "
                    f"{total_cost} > {self.resource_limits[resource]}"
                )
                
                # Trigger optimization
                await self._optimize_resource_usage(resource)
    
    async def _optimize_resource_usage(self, resource: str):
        """Optimize usage of a specific resource"""
        # Apply optimization rules
        for rule in self.optimization_rules:
            if rule["resource"] == resource:
                logger.info(f"Applying optimization rule: {rule['name']}")
                # In production, this would adjust system parameters
    
    def add_optimization_rule(self, rule: Dict[str, Any]):
        """Add cost optimization rule"""
        self.optimization_rules.append(rule)
    
    def get_cost_report(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Generate cost report"""
        total_costs = defaultdict(float)
        by_operation = {}
        
        for operation, resources in self.cost_tracking.items():
            operation_total = sum(resources.values())
            by_operation[operation] = {
                "total": operation_total,
                "by_resource": dict(resources)
            }
            
            for resource, cost in resources.items():
                total_costs[resource] += cost
        
        return {
            "total_cost": sum(total_costs.values()),
            "by_resource": dict(total_costs),
            "by_operation": by_operation,
            "top_operations": sorted(
                by_operation.items(),
                key=lambda x: x[1]["total"],
                reverse=True
            )[:10]
        }


class ErrorAnalyzer:
    """Analyzes errors for patterns and improvements"""
    
    def __init__(self):
        self.error_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.error_classifiers: Dict[str, Callable] = {}
        self.remediation_strategies: Dict[str, Callable] = {}
    
    async def analyze_error(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error and suggest remediation"""
        error_type = error_data.get("type", "unknown")
        error_message = error_data.get("message", "")
        
        # Classify error
        classification = await self._classify_error(error_type, error_message)
        
        # Find patterns
        patterns = await self._find_error_patterns(classification, error_data)
        
        # Suggest remediation
        remediation = await self._suggest_remediation(classification, patterns)
        
        # Store for pattern analysis
        self.error_patterns[classification["category"]].append({
            "timestamp": datetime.now().isoformat(),
            "error": error_data,
            "classification": classification,
            "remediation": remediation
        })
        
        return {
            "classification": classification,
            "patterns": patterns,
            "remediation": remediation
        }
    
    async def _classify_error(
        self, 
        error_type: str, 
        error_message: str
    ) -> Dict[str, Any]:
        """Classify error into categories"""
        # Use registered classifiers
        for name, classifier in self.error_classifiers.items():
            result = await classifier(error_type, error_message)
            if result:
                return result
        
        # Default classification
        if "timeout" in error_message.lower():
            return {"category": "performance", "subcategory": "timeout"}
        elif "memory" in error_message.lower():
            return {"category": "resource", "subcategory": "memory"}
        elif "parse" in error_message.lower():
            return {"category": "data", "subcategory": "parsing"}
        else:
            return {"category": "unknown", "subcategory": "general"}
    
    async def _find_error_patterns(
        self, 
        classification: Dict[str, Any], 
        error_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find patterns in error occurrences"""
        category = classification["category"]
        recent_errors = self.error_patterns[category][-100:]  # Last 100 errors
        
        patterns = []
        
        # Frequency pattern
        if len(recent_errors) >= 10:
            time_diffs = []
            for i in range(1, len(recent_errors)):
                t1 = datetime.fromisoformat(recent_errors[i-1]["timestamp"])
                t2 = datetime.fromisoformat(recent_errors[i]["timestamp"])
                time_diffs.append((t2 - t1).total_seconds())
            
            avg_interval = np.mean(time_diffs) if time_diffs else 0
            if avg_interval < 60:  # Less than 1 minute average
                patterns.append({
                    "type": "high_frequency",
                    "avg_interval_seconds": avg_interval
                })
        
        # Correlation patterns
        # In production, would analyze correlations with system state
        
        return patterns
    
    async def _suggest_remediation(
        self, 
        classification: Dict[str, Any], 
        patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Suggest remediation strategies"""
        suggestions = []
        
        # Check registered strategies
        strategy_key = f"{classification['category']}:{classification['subcategory']}"
        if strategy_key in self.remediation_strategies:
            strategy = await self.remediation_strategies[strategy_key](
                classification, patterns
            )
            suggestions.append(strategy)
        
        # Pattern-based suggestions
        for pattern in patterns:
            if pattern["type"] == "high_frequency":
                suggestions.append({
                    "action": "rate_limiting",
                    "description": "Implement rate limiting to prevent error floods"
                })
        
        # Default suggestions
        if not suggestions:
            if classification["category"] == "performance":
                suggestions.append({
                    "action": "increase_resources",
                    "description": "Consider increasing CPU/memory allocation"
                })
            elif classification["category"] == "data":
                suggestions.append({
                    "action": "improve_validation",
                    "description": "Enhance input validation and error handling"
                })
        
        return {
            "suggestions": suggestions,
            "priority": "high" if patterns else "medium"
        }
    
    def register_classifier(self, name: str, classifier: Callable):
        """Register custom error classifier"""
        self.error_classifiers[name] = classifier
    
    def register_remediation(self, key: str, strategy: Callable):
        """Register remediation strategy"""
        self.remediation_strategies[key] = strategy
    
    def get_error_summary(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get error analysis summary"""
        if category:
            errors = self.error_patterns.get(category, [])
        else:
            errors = [e for errors in self.error_patterns.values() for e in errors]
        
        if not errors:
            return {"message": "No errors recorded"}
        
        # Analyze errors
        by_category = defaultdict(int)
        remediation_types = defaultdict(int)
        
        for error in errors:
            by_category[error["classification"]["category"]] += 1
            for suggestion in error["remediation"]["suggestions"]:
                remediation_types[suggestion["action"]] += 1
        
        return {
            "total_errors": len(errors),
            "by_category": dict(by_category),
            "suggested_remediations": dict(remediation_types),
            "recent_errors": errors[-10:]  # Last 10 errors
        }


class AnalyticsDashboard:
    """Analytics dashboard for quality and feedback system"""
    
    def __init__(
        self,
        quality_monitor: QualityMonitor,
        feedback_loop: FeedbackLoop,
        ab_test_manager: ABTestManager,
        cost_optimizer: CostOptimizer,
        error_analyzer: ErrorAnalyzer
    ):
        self.quality_monitor = quality_monitor
        self.feedback_loop = feedback_loop
        self.ab_test_manager = ab_test_manager
        self.cost_optimizer = cost_optimizer
        self.error_analyzer = error_analyzer
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "timestamp": datetime.now().isoformat(),
            "quality_metrics": self._get_quality_summary(),
            "feedback_summary": self.feedback_loop.get_feedback_summary(),
            "active_ab_tests": self._get_ab_test_summary(),
            "cost_report": self.cost_optimizer.get_cost_report(),
            "error_analysis": self.error_analyzer.get_error_summary(),
            "system_health": self._get_system_health()
        }
    
    def _get_quality_summary(self) -> Dict[str, Any]:
        """Get quality metrics summary"""
        metrics = [
            "coherence", "completeness", "accuracy", "bias_level",
            "response_time", "memory_usage"
        ]
        
        summary = {}
        for metric in metrics:
            metric_summary = self.quality_monitor.get_metric_summary(metric)
            if "error" not in metric_summary:
                summary[metric] = metric_summary
        
        return summary
    
    def _get_ab_test_summary(self) -> List[Dict[str, Any]]:
        """Get summary of active A/B tests"""
        summaries = []
        
        for test_id, test in self.ab_test_manager.active_tests.items():
            if test.is_active():
                status = self.ab_test_manager.get_test_status(test_id)
                summaries.append({
                    "test_id": test_id,
                    "name": test.name,
                    "variants": list(test.variants.keys()),
                    "status": status
                })
        
        return summaries
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        # Calculate health score based on various factors
        quality_scores = []
        
        for metric in ["coherence", "completeness", "accuracy"]:
            summary = self.quality_monitor.get_metric_summary(metric)
            if "mean" in summary:
                quality_scores.append(summary["mean"])
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        # Get error rate
        error_summary = self.error_analyzer.get_error_summary()
        error_rate = error_summary.get("total_errors", 0) / 1000  # Per 1000 operations
        
        # Calculate health score
        health_score = (avg_quality * 0.7 + (1 - min(error_rate, 1)) * 0.3)
        
        return {
            "health_score": float(health_score),
            "status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.6 else "unhealthy",
            "avg_quality": float(avg_quality),
            "error_rate": float(error_rate)
        }
    
    async def generate_report(self, period: str = "daily") -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        data = self.get_dashboard_data()
        
        # Add period-specific analysis
        if period == "daily":
            data["trend_analysis"] = await self._analyze_daily_trends()
        elif period == "weekly":
            data["trend_analysis"] = await self._analyze_weekly_trends()
        
        # Add recommendations
        data["recommendations"] = await self._generate_recommendations(data)
        
        return data
    
    async def _analyze_daily_trends(self) -> Dict[str, Any]:
        """Analyze daily trends"""
        # In production, would analyze historical data
        return {
            "quality_trend": "stable",
            "error_trend": "decreasing",
            "cost_trend": "increasing"
        }
    
    async def _analyze_weekly_trends(self) -> Dict[str, Any]:
        """Analyze weekly trends"""
        # In production, would analyze historical data
        return {
            "quality_improvement": 0.05,
            "cost_reduction": -0.02,
            "error_reduction": 0.15
        }
    
    async def _generate_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Quality recommendations
        quality_metrics = data.get("quality_metrics", {})
        for metric, summary in quality_metrics.items():
            if summary.get("mean", 1.0) < 0.7:
                recommendations.append(
                    f"Improve {metric} - currently at {summary['mean']:.2f}"
                )
        
        # Cost recommendations
        cost_report = data.get("cost_report", {})
        if cost_report.get("total_cost", 0) > 1000:
            recommendations.append(
                "Consider cost optimization strategies - current spend exceeds threshold"
            )
        
        # Error recommendations
        error_analysis = data.get("error_analysis", {})
        if error_analysis.get("total_errors", 0) > 100:
            recommendations.append(
                "High error rate detected - review error patterns and apply remediations"
            )
        
        return recommendations


# Example usage
if __name__ == "__main__":
    import uuid
    
    async def example_quality_feedback():
        # Initialize components
        quality_monitor = QualityMonitor()
        feedback_loop = FeedbackLoop()
        ab_test_manager = ABTestManager()
        cost_optimizer = CostOptimizer()
        error_analyzer = ErrorAnalyzer()
        
        # Create dashboard
        dashboard = AnalyticsDashboard(
            quality_monitor,
            feedback_loop,
            ab_test_manager,
            cost_optimizer,
            error_analyzer
        )
        
        # Start feedback loop
        await feedback_loop.start()
        
        # Record some quality metrics
        await quality_monitor.record_metric(QualityMetric(
            name="coherence",
            value=0.85,
            threshold=0.7,
            timestamp=datetime.now(),
            metadata={"agent": "context_engineering"}
        ))
        
        # Submit feedback
        await feedback_loop.submit_feedback(FeedbackItem(
            feedback_id=str(uuid.uuid4()),
            feedback_type=FeedbackType.QUALITY,
            source="quality_monitor",
            target="context_parser",
            content={"coherence": 0.65, "completeness": 0.8},
            timestamp=datetime.now()
        ))
        
        # Create A/B test
        ab_test_manager.create_test(ABTestConfig(
            test_id="template_optimization_001",
            name="Template Optimization Test",
            variants={
                "control": {"template_version": "v1"},
                "treatment": {"template_version": "v2"}
            },
            metrics=["quality_score", "response_time"],
            allocation={"control": 0.5, "treatment": 0.5},
            start_time=datetime.now()
        ))
        
        # Track costs
        await cost_optimizer.track_cost(
            "context_parsing", "cpu_hours", 0.5, 0.10
        )
        
        # Analyze error
        error_result = await error_analyzer.analyze_error({
            "type": "TimeoutError",
            "message": "Context parsing timeout after 30s",
            "timestamp": datetime.now().isoformat()
        })
        
        # Get dashboard data
        dashboard_data = dashboard.get_dashboard_data()
        print(json.dumps(dashboard_data, indent=2, default=str))
        
        # Generate report
        report = await dashboard.generate_report("daily")
        print(f"\nDaily Report Generated with {len(report['recommendations'])} recommendations")
        
        # Stop feedback loop
        await feedback_loop.stop()
    
    # Run example
    asyncio.run(example_quality_feedback())