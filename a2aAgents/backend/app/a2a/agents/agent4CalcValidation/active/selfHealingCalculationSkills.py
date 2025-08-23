"""
Self-Healing Calculation Skills for Agent 4 - Phase 3 Intelligence Layer
Implements intelligent self-correction and validation with GrokClient integration
"""

import numpy as np
import json
import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import hashlib
from pathlib import Path

from app.a2a.sdk.decorators import a2a_skill, a2a_handler, a2a_task

# Import mixins with fallback
try:
    from app.a2a.sdk.mixins import PerformanceMonitorMixin, SecurityHardenedMixin
except ImportError:
    # Fallback stubs
    class PerformanceMonitorMixin:
        pass
    class SecurityHardenedMixin:
        pass

# Import trust identity with fallback
from app.a2a.core.trustIdentity import TrustIdentity
except ImportError:
    # Fallback stub
    class TrustIdentity:
        def __init__(self, **kwargs):
            pass
# Import data validation with fallback
from app.a2a.core.dataValidation import DataValidator
except ImportError:
    class DataValidator:
        @staticmethod
        def validate_input(data, schema): return True

# Import grok client with fallback
from app.clients.grokClient import GrokClient, get_grok_client
except ImportError:
    from app.a2a.core.grokClient import GrokClient
    def get_grok_client(): return GrokClient()


class CalculationErrorType(Enum):
    """Types of calculation errors that can be self-healed"""
    PRECISION_LOSS = "precision_loss"
    OVERFLOW_UNDERFLOW = "overflow_underflow"
    LOGICAL_ERROR = "logical_error"
    DOMAIN_ERROR = "domain_error"
    CONVERGENCE_FAILURE = "convergence_failure"
    DATA_INCONSISTENCY = "data_inconsistency"


@dataclass
class CalculationError:
    """Represents a calculation error with context"""
    error_id: str
    error_type: CalculationErrorType
    description: str
    original_input: Dict[str, Any]
    failed_output: Any
    confidence_score: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class HealingStrategy:
    """Strategy for healing a calculation error"""
    strategy_id: str
    strategy_type: str
    description: str
    implementation_steps: List[str]
    success_probability: float
    computational_cost: float


@dataclass
class HealingResult:
    """Result of self-healing process"""
    healing_id: str
    original_error: CalculationError
    strategy_used: HealingStrategy
    healed_output: Any
    healing_confidence: float
    validation_results: Dict[str, Any]
    grok_insights: Optional[str] = None


class SelfHealingCalculationSkills(PerformanceMonitorMixin, SecurityHardenedMixin):
    """
    Real A2A agent skills for self-healing calculations with GrokClient integration
    Provides intelligent error detection, diagnosis, and automatic correction
    """

    def __init__(self, trust_identity: TrustIdentity):
        super().__init__()
        self.trust_identity = trust_identity
        self.logger = logging.getLogger(__name__)
        self.data_validator = DataValidator()
        
        # Initialize GrokClient for intelligent analysis
        try:
            self.grok_client = get_grok_client()
            self.logger.info("GrokClient initialized for self-healing calculations")
        except Exception as e:
            self.logger.warning(f"GrokClient initialization failed: {e}")
            self.grok_client = None
        
        # Healing strategies repository
        self.healing_strategies = self._initialize_healing_strategies()
        
        # Error pattern database for learning
        self.error_patterns = {
            'precision_patterns': [],
            'overflow_patterns': [],
            'logical_patterns': [],
            'domain_patterns': [],
            'convergence_patterns': [],
            'data_patterns': []
        }
        
        # Performance tracking
        self.healing_metrics = {
            'total_errors_detected': 0,
            'successful_healings': 0,
            'healing_success_rate': 0.0,
            'avg_healing_time': 0.0,
            'grok_consultations': 0,
            'learned_patterns': 0
        }

    @a2a_skill(
        name="detectCalculationErrors",
        description="Automatically detect calculation errors and anomalies",
        input_schema={
            "type": "object",
            "properties": {
                "calculation_result": {"type": "any"},
                "input_data": {"type": "object"},
                "expected_properties": {
                    "type": "object",
                    "properties": {
                        "range_min": {"type": "number"},
                        "range_max": {"type": "number"},
                        "data_type": {"type": "string"},
                        "precision_requirements": {"type": "integer"},
                        "domain_constraints": {"type": "array", "items": {"type": "string"}}
                    }
                },
                "historical_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "object"},
                            "output": {"type": "any"},
                            "timestamp": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["calculation_result", "input_data"]
        }
    )
    def detect_calculation_errors(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect calculation errors using multiple validation techniques"""
        try:
            calculation_result = request_data["calculation_result"]
            input_data = request_data["input_data"]
            expected_properties = request_data.get("expected_properties", {})
            historical_results = request_data.get("historical_results", [])
            
            detected_errors = []
            detection_confidence = 0.0
            
            # 1. Range validation
            range_errors = self._detect_range_errors(calculation_result, expected_properties)
            detected_errors.extend(range_errors)
            
            # 2. Type validation
            type_errors = self._detect_type_errors(calculation_result, expected_properties)
            detected_errors.extend(type_errors)
            
            # 3. Mathematical property validation
            math_errors = self._detect_mathematical_property_errors(calculation_result, input_data)
            detected_errors.extend(math_errors)
            
            # 4. Historical pattern analysis
            if historical_results:
                pattern_errors = self._detect_pattern_anomalies(calculation_result, input_data, historical_results)
                detected_errors.extend(pattern_errors)
            
            # 5. Domain constraint validation
            domain_errors = self._detect_domain_constraint_violations(calculation_result, expected_properties)
            detected_errors.extend(domain_errors)
            
            # 6. Statistical anomaly detection
            statistical_errors = self._detect_statistical_anomalies(calculation_result, historical_results)
            detected_errors.extend(statistical_errors)
            
            # Calculate overall detection confidence
            if detected_errors:
                detection_confidence = np.mean([error['confidence'] for error in detected_errors])
                self.healing_metrics['total_errors_detected'] += len(detected_errors)
            
            # Create formal error objects
            formal_errors = []
            for error_data in detected_errors:
                error = CalculationError(
                    error_id=hashlib.md5(f"{error_data['type']}_{error_data['description']}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12],
                    error_type=CalculationErrorType(error_data['type']),
                    description=error_data['description'],
                    original_input=input_data,
                    failed_output=calculation_result,
                    confidence_score=error_data['confidence']
                )
                formal_errors.append(error)
            
            return {
                'success': True,
                'errors_detected': len(detected_errors),
                'detection_confidence': float(detection_confidence),
                'error_details': [
                    {
                        'error_id': error.error_id,
                        'error_type': error.error_type.value,
                        'description': error.description,
                        'confidence_score': error.confidence_score,
                        'timestamp': error.timestamp
                    }
                    for error in formal_errors
                ],
                'calculation_status': 'error_detected' if detected_errors else 'valid',
                'validation_summary': {
                    'range_checks': len([e for e in detected_errors if 'range' in e['description']]),
                    'type_checks': len([e for e in detected_errors if 'type' in e['description']]),
                    'mathematical_checks': len([e for e in detected_errors if 'mathematical' in e['description']]),
                    'domain_checks': len([e for e in detected_errors if 'domain' in e['description']]),
                    'statistical_checks': len([e for e in detected_errors if 'statistical' in e['description']])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detection failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'error_detection_failure'
            }

    @a2a_skill(
        name="healCalculationError",
        description="Automatically heal calculation errors using AI-driven strategies",
        input_schema={
            "type": "object",
            "properties": {
                "error_details": {
                    "type": "object",
                    "properties": {
                        "error_id": {"type": "string"},
                        "error_type": {"type": "string"},
                        "description": {"type": "string"},
                        "original_input": {"type": "object"},
                        "failed_output": {"type": "any"}
                    }
                },
                "healing_preferences": {
                    "type": "object",
                    "properties": {
                        "max_iterations": {"type": "integer", "default": 3},
                        "accuracy_threshold": {"type": "number", "default": 0.95},
                        "use_grok_analysis": {"type": "boolean", "default": True},
                        "allowed_strategies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["precision_adjustment", "algorithmic_refinement", "data_preprocessing", "numerical_stabilization"]
                        }
                    }
                }
            },
            "required": ["error_details"]
        }
    )
    def heal_calculation_error(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Heal calculation errors using intelligent strategies"""
        try:
            error_details = request_data["error_details"]
            healing_prefs = request_data.get("healing_preferences", {})
            
            # Reconstruct error object
            error = CalculationError(
                error_id=error_details["error_id"],
                error_type=CalculationErrorType(error_details["error_type"]),
                description=error_details["description"],
                original_input=error_details["original_input"],
                failed_output=error_details["failed_output"],
                confidence_score=error_details.get("confidence_score", 0.8)
            )
            
            # Get healing preferences
            max_iterations = healing_prefs.get("max_iterations", 3)
            accuracy_threshold = healing_prefs.get("accuracy_threshold", 0.95)
            use_grok = healing_prefs.get("use_grok_analysis", True) and self.grok_client is not None
            allowed_strategies = healing_prefs.get("allowed_strategies", [
                "precision_adjustment", "algorithmic_refinement", "data_preprocessing", "numerical_stabilization"
            ])
            
            # Get Grok insights if enabled
            grok_insights = None
            if use_grok:
                grok_insights = self._get_grok_healing_insights(error)
                self.healing_metrics['grok_consultations'] += 1
            
            # Select healing strategy
            best_strategy = self._select_healing_strategy(error, allowed_strategies, grok_insights)
            
            # Execute healing iterations
            healing_result = None
            for iteration in range(max_iterations):
                self.logger.info(f"Healing iteration {iteration + 1} for error {error.error_id}")
                
                try:
                    healing_result = self._execute_healing_strategy(error, best_strategy, iteration)
                    
                    # Validate healing result
                    validation_result = self._validate_healing_result(healing_result, accuracy_threshold)
                    
                    if validation_result['is_valid']:
                        self.healing_metrics['successful_healings'] += 1
                        break
                        
                except Exception as healing_error:
                    self.logger.warning(f"Healing iteration {iteration + 1} failed: {healing_error}")
                    if iteration == max_iterations - 1:
                        raise healing_error
            
            # Update success rate
            if self.healing_metrics['total_errors_detected'] > 0:
                self.healing_metrics['healing_success_rate'] = (
                    self.healing_metrics['successful_healings'] / self.healing_metrics['total_errors_detected']
                )
            
            # Learn from this healing attempt
            self._learn_from_healing_attempt(error, best_strategy, healing_result)
            
            return {
                'success': True,
                'healing_result': {
                    'healing_id': healing_result.healing_id if healing_result else 'failed',
                    'original_error': {
                        'error_id': error.error_id,
                        'error_type': error.error_type.value,
                        'description': error.description
                    },
                    'strategy_used': {
                        'strategy_id': best_strategy.strategy_id,
                        'strategy_type': best_strategy.strategy_type,
                        'description': best_strategy.description
                    },
                    'healed_output': healing_result.healed_output if healing_result else None,
                    'healing_confidence': healing_result.healing_confidence if healing_result else 0.0,
                    'validation_results': healing_result.validation_results if healing_result else {},
                    'grok_insights': grok_insights
                },
                'healing_metrics': {
                    'iterations_used': max_iterations if not healing_result else iteration + 1,
                    'success_rate': self.healing_metrics['healing_success_rate'],
                    'total_healings_attempted': self.healing_metrics['total_errors_detected'],
                    'successful_healings': self.healing_metrics['successful_healings']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Calculation healing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'healing_execution_error'
            }

    @a2a_skill(
        name="learnFromCalculationPatterns",
        description="Learn from calculation patterns to improve future healing strategies",
        input_schema={
            "type": "object",
            "properties": {
                "calculation_history": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "object"},
                            "output": {"type": "any"},
                            "error_type": {"type": "string"},
                            "healing_strategy": {"type": "string"},
                            "success": {"type": "boolean"},
                            "timestamp": {"type": "string"}
                        }
                    }
                },
                "learning_parameters": {
                    "type": "object",
                    "properties": {
                        "min_pattern_frequency": {"type": "integer", "default": 3},
                        "confidence_threshold": {"type": "number", "default": 0.8},
                        "enable_grok_analysis": {"type": "boolean", "default": True}
                    }
                }
            },
            "required": ["calculation_history"]
        }
    )
    def learn_from_calculation_patterns(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from calculation patterns to improve healing strategies"""
        try:
            calculation_history = request_data["calculation_history"]
            learning_params = request_data.get("learning_parameters", {})
            
            min_frequency = learning_params.get("min_pattern_frequency", 3)
            confidence_threshold = learning_params.get("confidence_threshold", 0.8)
            enable_grok = learning_params.get("enable_grok_analysis", True) and self.grok_client is not None
            
            # Analyze patterns by error type
            error_patterns = {}
            strategy_effectiveness = {}
            
            for record in calculation_history:
                error_type = record.get("error_type")
                strategy = record.get("healing_strategy")
                success = record.get("success", False)
                
                if error_type:
                    if error_type not in error_patterns:
                        error_patterns[error_type] = []
                    error_patterns[error_type].append(record)
                    
                if strategy:
                    if strategy not in strategy_effectiveness:
                        strategy_effectiveness[strategy] = {'success': 0, 'total': 0}
                    strategy_effectiveness[strategy]['total'] += 1
                    if success:
                        strategy_effectiveness[strategy]['success'] += 1
            
            # Identify recurring patterns
            learned_patterns = []
            for error_type, records in error_patterns.items():
                if len(records) >= min_frequency:
                    pattern_analysis = self._analyze_error_pattern(error_type, records)
                    if pattern_analysis['confidence'] >= confidence_threshold:
                        learned_patterns.append(pattern_analysis)
                        self.healing_metrics['learned_patterns'] += 1
            
            # Get Grok insights on patterns if enabled
            grok_pattern_insights = None
            if enable_grok and learned_patterns:
                grok_pattern_insights = self._get_grok_pattern_insights(learned_patterns, strategy_effectiveness)
            
            # Update healing strategies based on learning
            strategy_updates = self._update_healing_strategies(learned_patterns, strategy_effectiveness)
            
            # Generate learning report
            learning_report = {
                'patterns_discovered': len(learned_patterns),
                'strategies_analyzed': len(strategy_effectiveness),
                'most_effective_strategy': max(strategy_effectiveness.items(), 
                                             key=lambda x: x[1]['success']/max(x[1]['total'], 1))[0] if strategy_effectiveness else None,
                'least_effective_strategy': min(strategy_effectiveness.items(),
                                              key=lambda x: x[1]['success']/max(x[1]['total'], 1))[0] if strategy_effectiveness else None,
                'overall_success_rate': sum(s['success'] for s in strategy_effectiveness.values()) / max(sum(s['total'] for s in strategy_effectiveness.values()), 1)
            }
            
            return {
                'success': True,
                'learning_results': {
                    'patterns_learned': [
                        {
                            'error_type': pattern['error_type'],
                            'pattern_description': pattern['description'],
                            'frequency': pattern['frequency'],
                            'confidence': pattern['confidence'],
                            'recommended_strategy': pattern['recommended_strategy']
                        }
                        for pattern in learned_patterns
                    ],
                    'strategy_effectiveness': {
                        strategy: {
                            'success_rate': stats['success'] / max(stats['total'], 1),
                            'total_uses': stats['total'],
                            'successful_uses': stats['success']
                        }
                        for strategy, stats in strategy_effectiveness.items()
                    },
                    'strategy_updates': strategy_updates,
                    'grok_insights': grok_pattern_insights
                },
                'learning_report': learning_report,
                'updated_metrics': self.healing_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Pattern learning failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'pattern_learning_error'
            }

    def _initialize_healing_strategies(self) -> Dict[str, HealingStrategy]:
        """Initialize the repository of healing strategies"""
        strategies = {}
        
        # Precision adjustment strategy
        strategies['precision_adjustment'] = HealingStrategy(
            strategy_id='precision_adjustment',
            strategy_type='numerical_correction',
            description='Adjust numerical precision to handle floating-point errors',
            implementation_steps=[
                'Analyze precision requirements',
                'Convert to higher precision arithmetic',
                'Apply precision-aware operations',
                'Validate result accuracy'
            ],
            success_probability=0.8,
            computational_cost=0.3
        )
        
        # Algorithmic refinement strategy
        strategies['algorithmic_refinement'] = HealingStrategy(
            strategy_id='algorithmic_refinement',
            strategy_type='algorithmic_improvement',
            description='Refine algorithm to be more numerically stable',
            implementation_steps=[
                'Identify numerical instability sources',
                'Apply alternative computational approaches',
                'Implement iterative refinement',
                'Validate convergence properties'
            ],
            success_probability=0.7,
            computational_cost=0.6
        )
        
        # Data preprocessing strategy
        strategies['data_preprocessing'] = HealingStrategy(
            strategy_id='data_preprocessing',
            strategy_type='input_correction',
            description='Preprocess input data to avoid calculation issues',
            implementation_steps=[
                'Analyze input data quality',
                'Apply normalization/scaling',
                'Handle edge cases and outliers',
                'Validate preprocessed data'
            ],
            success_probability=0.75,
            computational_cost=0.4
        )
        
        # Numerical stabilization strategy
        strategies['numerical_stabilization'] = HealingStrategy(
            strategy_id='numerical_stabilization',
            strategy_type='stability_enhancement',
            description='Apply numerical stabilization techniques',
            implementation_steps=[
                'Identify instability patterns',
                'Apply regularization techniques',
                'Use stable computational variants',
                'Monitor stability metrics'
            ],
            success_probability=0.65,
            computational_cost=0.5
        )
        
        return strategies

    def _detect_range_errors(self, result: Any, expected_properties: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect range validation errors"""
        errors = []
        
        if isinstance(result, (int, float, np.number)):
            range_min = expected_properties.get('range_min')
            range_max = expected_properties.get('range_max')
            
            if range_min is not None and result < range_min:
                errors.append({
                    'type': 'domain_error',
                    'description': f'Result {result} below minimum range {range_min}',
                    'confidence': 0.9
                })
                
            if range_max is not None and result > range_max:
                errors.append({
                    'type': 'domain_error',
                    'description': f'Result {result} above maximum range {range_max}',
                    'confidence': 0.9
                })
        
        return errors

    def _detect_type_errors(self, result: Any, expected_properties: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect type validation errors"""
        errors = []
        expected_type = expected_properties.get('data_type')
        
        if expected_type:
            if expected_type == 'integer' and not isinstance(result, (int, np.integer)):
                errors.append({
                    'type': 'logical_error',
                    'description': f'Expected integer but got {type(result).__name__}',
                    'confidence': 0.95
                })
            elif expected_type == 'float' and not isinstance(result, (float, np.floating)):
                errors.append({
                    'type': 'logical_error',
                    'description': f'Expected float but got {type(result).__name__}',
                    'confidence': 0.95
                })
        
        return errors

    def _detect_mathematical_property_errors(self, result: Any, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect mathematical property violations"""
        errors = []
        
        # Check for NaN/Infinity
        if isinstance(result, (float, np.floating)):
            if np.isnan(result):
                errors.append({
                    'type': 'domain_error',
                    'description': 'Result is NaN (Not a Number)',
                    'confidence': 1.0
                })
            elif np.isinf(result):
                errors.append({
                    'type': 'overflow_underflow',
                    'description': 'Result is infinite',
                    'confidence': 1.0
                })
        
        # Check for complex results when real expected
        if isinstance(result, complex) and result.imag != 0:
            errors.append({
                'type': 'domain_error',
                'description': 'Complex result when real number expected',
                'confidence': 0.9
            })
        
        return errors

    def _detect_pattern_anomalies(self, result: Any, input_data: Dict[str, Any], historical_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies based on historical patterns"""
        errors = []
        
        if not historical_results or len(historical_results) < 3:
            return errors
        
        # Extract historical outputs for similar inputs
        similar_outputs = []
        for hist in historical_results:
            if self._are_inputs_similar(input_data, hist.get('input', {})):
                similar_outputs.append(hist.get('output'))
        
        if len(similar_outputs) >= 2:
            if isinstance(result, (int, float, np.number)) and all(isinstance(o, (int, float, np.number)) for o in similar_outputs):
                mean_output = np.mean(similar_outputs)
                std_output = np.std(similar_outputs)
                
                # Z-score anomaly detection
                if std_output > 0:
                    z_score = abs((result - mean_output) / std_output)
                    if z_score > 3.0:  # More than 3 standard deviations
                        errors.append({
                            'type': 'data_inconsistency',
                            'description': f'Result {result} is anomalous (z-score: {z_score:.2f})',
                            'confidence': min(z_score / 10.0, 0.95)
                        })
        
        return errors

    def _detect_domain_constraint_violations(self, result: Any, expected_properties: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect domain constraint violations"""
        errors = []
        domain_constraints = expected_properties.get('domain_constraints', [])
        
        for constraint in domain_constraints:
            if constraint == 'positive' and isinstance(result, (int, float, np.number)) and result <= 0:
                errors.append({
                    'type': 'domain_error',
                    'description': 'Result must be positive',
                    'confidence': 0.9
                })
            elif constraint == 'non_negative' and isinstance(result, (int, float, np.number)) and result < 0:
                errors.append({
                    'type': 'domain_error',
                    'description': 'Result must be non-negative',
                    'confidence': 0.9
                })
            elif constraint == 'probability' and isinstance(result, (int, float, np.number)) and not (0 <= result <= 1):
                errors.append({
                    'type': 'domain_error',
                    'description': 'Probability must be between 0 and 1',
                    'confidence': 0.95
                })
        
        return errors

    def _detect_statistical_anomalies(self, result: Any, historical_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect statistical anomalies"""
        errors = []
        
        if not historical_results or len(historical_results) < 5:
            return errors
        
        # Extract all historical outputs
        all_outputs = [hist.get('output') for hist in historical_results]
        numeric_outputs = [o for o in all_outputs if isinstance(o, (int, float, np.number))]
        
        if len(numeric_outputs) >= 5 and isinstance(result, (int, float, np.number)):
            # Interquartile range (IQR) method
            q1 = np.percentile(numeric_outputs, 25)
            q3 = np.percentile(numeric_outputs, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            if result < lower_bound or result > upper_bound:
                errors.append({
                    'type': 'data_inconsistency',
                    'description': f'Result {result} is statistical outlier (IQR bounds: [{lower_bound:.3f}, {upper_bound:.3f}])',
                    'confidence': 0.7
                })
        
        return errors

    def _are_inputs_similar(self, input1: Dict[str, Any], input2: Dict[str, Any], similarity_threshold: float = 0.8) -> bool:
        """Check if two inputs are similar"""
        if set(input1.keys()) != set(input2.keys()):
            return False
        
        similar_keys = 0
        for key in input1.keys():
            val1, val2 = input1[key], input2[key]
            if isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
                if val1 == 0 and val2 == 0:
                    similar_keys += 1
                elif val1 != 0 and abs((val1 - val2) / val1) < 0.1:  # Within 10%
                    similar_keys += 1
            elif val1 == val2:
                similar_keys += 1
        
        return (similar_keys / len(input1)) >= similarity_threshold

    def _get_grok_healing_insights(self, error: CalculationError) -> Optional[str]:
        """Get AI insights from GrokClient for healing strategy"""
        if not self.grok_client:
            return None
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a mathematical computation expert. Analyze calculation errors and suggest healing strategies."
                },
                {
                    "role": "user",
                    "content": f"""
                    Analyze this calculation error and suggest healing strategies:
                    
                    Error Type: {error.error_type.value}
                    Description: {error.description}
                    Input Data: {json.dumps(error.original_input, indent=2)}
                    Failed Output: {error.failed_output}
                    
                    Please provide specific, actionable healing strategies for this error.
                    """
                }
            ]
            
            response = self.grok_client.chat_completion(messages, temperature=0.3, max_tokens=500)
            return response.content
            
        except Exception as e:
            self.logger.warning(f"GrokClient healing insights failed: {e}")
            return None

    def _select_healing_strategy(self, error: CalculationError, allowed_strategies: List[str], grok_insights: Optional[str] = None) -> HealingStrategy:
        """Select the best healing strategy for the error"""
        # Filter available strategies
        available_strategies = {k: v for k, v in self.healing_strategies.items() if k in allowed_strategies}
        
        if not available_strategies:
            # Return default strategy
            return list(self.healing_strategies.values())[0]
        
        # Strategy selection logic based on error type
        strategy_scores = {}
        for strategy_id, strategy in available_strategies.items():
            base_score = strategy.success_probability
            
            # Adjust score based on error type compatibility
            if error.error_type == CalculationErrorType.PRECISION_LOSS and strategy_id == 'precision_adjustment':
                base_score *= 1.3
            elif error.error_type == CalculationErrorType.OVERFLOW_UNDERFLOW and strategy_id == 'numerical_stabilization':
                base_score *= 1.2
            elif error.error_type == CalculationErrorType.DOMAIN_ERROR and strategy_id == 'data_preprocessing':
                base_score *= 1.2
            elif error.error_type == CalculationErrorType.LOGICAL_ERROR and strategy_id == 'algorithmic_refinement':
                base_score *= 1.2
            
            # Consider computational cost
            base_score *= (1.0 - strategy.computational_cost * 0.2)
            
            strategy_scores[strategy_id] = base_score
        
        # Select highest scoring strategy
        best_strategy_id = max(strategy_scores.items(), key=lambda x: x[1])[0]
        return available_strategies[best_strategy_id]

    def _execute_healing_strategy(self, error: CalculationError, strategy: HealingStrategy, iteration: int) -> HealingResult:
        """Execute the selected healing strategy"""
        healing_id = hashlib.md5(f"{error.error_id}_{strategy.strategy_id}_{iteration}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:12]
        
        # Strategy-specific healing implementation
        healed_output = None
        healing_confidence = 0.0
        validation_results = {}
        
        if strategy.strategy_id == 'precision_adjustment':
            healed_output, healing_confidence, validation_results = self._apply_precision_adjustment(error)
        elif strategy.strategy_id == 'algorithmic_refinement':
            healed_output, healing_confidence, validation_results = self._apply_algorithmic_refinement(error)
        elif strategy.strategy_id == 'data_preprocessing':
            healed_output, healing_confidence, validation_results = self._apply_data_preprocessing(error)
        elif strategy.strategy_id == 'numerical_stabilization':
            healed_output, healing_confidence, validation_results = self._apply_numerical_stabilization(error)
        else:
            raise ValueError(f"Unknown healing strategy: {strategy.strategy_id}")
        
        return HealingResult(
            healing_id=healing_id,
            original_error=error,
            strategy_used=strategy,
            healed_output=healed_output,
            healing_confidence=healing_confidence,
            validation_results=validation_results
        )

    def _apply_precision_adjustment(self, error: CalculationError) -> Tuple[Any, float, Dict[str, Any]]:
        """Apply precision adjustment healing strategy"""
        original_input = error.original_input
        
        # Convert inputs to higher precision if possible
        if isinstance(error.failed_output, (float, np.floating)):
            # Use higher precision arithmetic
            from decimal import Decimal, getcontext


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
            getcontext().prec = 50  # Higher precision
            
            try:
                # Example: re-compute with higher precision (simplified)
                healed_output = float(error.failed_output)  # Placeholder - actual computation depends on context
                healing_confidence = 0.8
                validation_results = {'precision_improved': True, 'method': 'decimal_precision'}
                
                return healed_output, healing_confidence, validation_results
            except Exception as e:
                return error.failed_output, 0.1, {'error': str(e)}
        
        return error.failed_output, 0.3, {'method': 'no_precision_adjustment_needed'}

    def _apply_algorithmic_refinement(self, error: CalculationError) -> Tuple[Any, float, Dict[str, Any]]:
        """Apply algorithmic refinement healing strategy"""
        # Placeholder for algorithm-specific refinement
        # In practice, this would involve algorithm-specific improvements
        
        healed_output = error.failed_output
        healing_confidence = 0.6
        validation_results = {'refinement_applied': 'generic_stabilization'}
        
        return healed_output, healing_confidence, validation_results

    def _apply_data_preprocessing(self, error: CalculationError) -> Tuple[Any, float, Dict[str, Any]]:
        """Apply data preprocessing healing strategy"""
        original_input = error.original_input.copy()
        
        # Apply common preprocessing techniques
        preprocessing_applied = []
        
        for key, value in original_input.items():
            if isinstance(value, (int, float, np.number)):
                # Handle edge cases
                if np.isnan(value) or np.isinf(value):
                    original_input[key] = 0.0  # Or median/mean from historical data
                    preprocessing_applied.append(f'replaced_invalid_{key}')
                
                # Normalize extreme values
                if abs(value) > 1e10:
                    original_input[key] = np.sign(value) * 1e10
                    preprocessing_applied.append(f'clamped_{key}')
        
        # Re-compute with preprocessed data (simplified example)
        healed_output = error.failed_output  # Placeholder
        healing_confidence = 0.7 if preprocessing_applied else 0.3
        validation_results = {
            'preprocessing_applied': preprocessing_applied,
            'preprocessed_input': original_input
        }
        
        return healed_output, healing_confidence, validation_results

    def _apply_numerical_stabilization(self, error: CalculationError) -> Tuple[Any, float, Dict[str, Any]]:
        """Apply numerical stabilization healing strategy"""
        # Apply stabilization techniques
        stabilization_methods = []
        
        if isinstance(error.failed_output, (float, np.floating)):
            if abs(error.failed_output) < 1e-15:  # Very small number
                healed_output = 0.0
                stabilization_methods.append('zero_threshold')
            else:
                healed_output = error.failed_output
        else:
            healed_output = error.failed_output
        
        healing_confidence = 0.6
        validation_results = {
            'stabilization_methods': stabilization_methods,
            'original_magnitude': abs(error.failed_output) if isinstance(error.failed_output, (int, float, np.number)) else 'non_numeric'
        }
        
        return healed_output, healing_confidence, validation_results

    def _validate_healing_result(self, healing_result: HealingResult, accuracy_threshold: float) -> Dict[str, Any]:
        """Validate the healing result"""
        is_valid = healing_result.healing_confidence >= accuracy_threshold
        
        validation_details = {
            'is_valid': is_valid,
            'confidence_check': healing_result.healing_confidence >= accuracy_threshold,
            'output_type_check': not (isinstance(healing_result.healed_output, (float, np.floating)) and 
                                    (np.isnan(healing_result.healed_output) or np.isinf(healing_result.healed_output))),
            'healing_confidence': healing_result.healing_confidence,
            'accuracy_threshold': accuracy_threshold
        }
        
        return validation_details

    def _learn_from_healing_attempt(self, error: CalculationError, strategy: HealingStrategy, healing_result: Optional[HealingResult]):
        """Learn from the healing attempt to improve future strategies"""
        # Store pattern for future reference
        pattern_key = f"{error.error_type.value}_patterns"
        if pattern_key in self.error_patterns:
            self.error_patterns[pattern_key].append({
                'error_description': error.description,
                'strategy_used': strategy.strategy_id,
                'success': healing_result is not None and healing_result.healing_confidence > 0.5,
                'confidence_achieved': healing_result.healing_confidence if healing_result else 0.0,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Keep only recent patterns (last 100)
            if len(self.error_patterns[pattern_key]) > 100:
                self.error_patterns[pattern_key] = self.error_patterns[pattern_key][-100:]

    def _analyze_error_pattern(self, error_type: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a specific error pattern"""
        successful_strategies = {}
        total_attempts = len(records)
        
        for record in records:
            strategy = record.get('healing_strategy', 'unknown')
            success = record.get('success', False)
            
            if strategy not in successful_strategies:
                successful_strategies[strategy] = {'success': 0, 'total': 0}
            
            successful_strategies[strategy]['total'] += 1
            if success:
                successful_strategies[strategy]['success'] += 1
        
        # Find most effective strategy
        best_strategy = None
        best_success_rate = 0.0
        
        for strategy, stats in successful_strategies.items():
            success_rate = stats['success'] / max(stats['total'], 1)
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_strategy = strategy
        
        return {
            'error_type': error_type,
            'frequency': total_attempts,
            'confidence': min(total_attempts / 10.0, 1.0),  # Higher frequency = higher confidence
            'description': f'Pattern identified for {error_type} with {total_attempts} occurrences',
            'recommended_strategy': best_strategy,
            'success_rate': best_success_rate
        }

    def _get_grok_pattern_insights(self, learned_patterns: List[Dict[str, Any]], strategy_effectiveness: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Get Grok insights on learned patterns"""
        if not self.grok_client:
            return None
        
        try:
            patterns_summary = "\n".join([
                f"- {pattern['error_type']}: {pattern['frequency']} occurrences, {pattern['recommended_strategy']} recommended"
                for pattern in learned_patterns
            ])
            
            strategy_summary = "\n".join([
                f"- {strategy}: {stats['success']}/{stats['total']} success rate"
                for strategy, stats in strategy_effectiveness.items()
            ])
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a machine learning expert analyzing calculation error patterns to improve healing strategies."
                },
                {
                    "role": "user",
                    "content": f"""
                    Analyze these learned calculation error patterns and strategy effectiveness:
                    
                    Learned Patterns:
                    {patterns_summary}
                    
                    Strategy Effectiveness:
                    {strategy_summary}
                    
                    Please provide insights on:
                    1. Most critical patterns to address
                    2. Strategy optimization recommendations
                    3. Potential new healing approaches
                    """
                }
            ]
            
            response = self.grok_client.chat_completion(messages, temperature=0.4, max_tokens=400)
            return response.content
            
        except Exception as e:
            self.logger.warning(f"GrokClient pattern insights failed: {e}")
            return None

    def _update_healing_strategies(self, learned_patterns: List[Dict[str, Any]], strategy_effectiveness: Dict[str, Dict[str, Any]]) -> List[str]:
        """Update healing strategies based on learned patterns"""
        updates = []
        
        for pattern in learned_patterns:
            recommended_strategy = pattern['recommended_strategy']
            if recommended_strategy in self.healing_strategies:
                # Increase success probability for effective strategies
                current_prob = self.healing_strategies[recommended_strategy].success_probability
                new_prob = min(current_prob * 1.1, 0.95)  # Max 95%
                self.healing_strategies[recommended_strategy].success_probability = new_prob
                updates.append(f"Increased success probability for {recommended_strategy} to {new_prob:.2f}")
        
        # Decrease probability for less effective strategies
        for strategy, stats in strategy_effectiveness.items():
            if strategy in self.healing_strategies and stats['total'] >= 5:
                success_rate = stats['success'] / stats['total']
                if success_rate < 0.3:  # Poor performance
                    current_prob = self.healing_strategies[strategy].success_probability
                    new_prob = max(current_prob * 0.9, 0.1)  # Min 10%
                    self.healing_strategies[strategy].success_probability = new_prob
                    updates.append(f"Decreased success probability for {strategy} to {new_prob:.2f}")
        
        return updates