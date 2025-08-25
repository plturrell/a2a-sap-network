import asyncio
import uuid
import json
import hashlib
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics
import numpy as np

from app.a2a.core.security_base import SecureA2AAgent
"""
Quality Control Simulation Framework
Provides comprehensive simulation capabilities for testing quality control scenarios
"""

logger = logging.getLogger(__name__)

class QualityScenario(Enum):
    NORMAL_QUALITY_CHECKS = "normal_quality_checks"
    HIGH_DEFECT_RATE = "high_defect_rate"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME_MONITORING = "real_time_monitoring"
    QUALITY_DEGRADATION = "quality_degradation"
    COMPLIANCE_AUDIT = "compliance_audit"
    PERFORMANCE_STRESS = "performance_stress"
    MULTI_CRITERIA_VALIDATION = "multi_criteria_validation"
    ANOMALY_DETECTION = "anomaly_detection"
    REGRESSION_TESTING = "regression_testing"

@dataclass
class QualityDataPoint:
    """Simulated data point for quality testing"""
    id: str
    data_type: str
    value: Any
    expected_value: Optional[Any] = None
    quality_score: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "simulation"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityRule:
    """Quality rule definition for simulation"""
    rule_id: str
    name: str
    rule_type: str  # "range", "format", "uniqueness", "completeness", etc.
    parameters: Dict[str, Any]
    severity: str = "medium"  # "low", "medium", "high", "critical"
    failure_probability: float = 0.1

@dataclass
class SimulationMetrics:
    """Metrics for quality control simulation"""
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    critical_failures: int = 0
    average_quality_score: float = 0.0
    processing_time_ms: List[float] = field(default_factory=list)
    defect_rates: Dict[str, float] = field(default_factory=dict)
    rule_performance: Dict[str, Dict[str, int]] = field(default_factory=dict)
    batch_statistics: Dict[str, Any] = field(default_factory=dict)

class QualityControlSimulator(SecureA2AAgent):
    """Comprehensive simulation framework for quality control testing"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning

    def __init__(self, quality_control_agent):
        super().__init__()
        self.quality_control_agent = quality_control_agent
        self.simulation_metrics = SimulationMetrics()
        self.simulation_running = False
        self.simulation_tasks: List[asyncio.Task] = []

        # Data generators for different types
        self.data_generators = {
            "numerical": self._generate_numerical_data,
            "categorical": self._generate_categorical_data,
            "text": self._generate_text_data,
            "date": self._generate_date_data,
            "email": self._generate_email_data,
            "financial": self._generate_financial_data
        }

        # Quality rules for different scenarios
        self.quality_rules = {
            "range_check": QualityRule(
                rule_id="range_001",
                name="Numerical Range Validation",
                rule_type="range",
                parameters={"min_value": 0, "max_value": 100}
            ),
            "format_check": QualityRule(
                rule_id="format_001",
                name="Email Format Validation",
                rule_type="format",
                parameters={"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}
            ),
            "completeness_check": QualityRule(
                rule_id="complete_001",
                name="Data Completeness Check",
                rule_type="completeness",
                parameters={"required_fields": ["id", "value", "timestamp"]}
            ),
            "uniqueness_check": QualityRule(
                rule_id="unique_001",
                name="ID Uniqueness Validation",
                rule_type="uniqueness",
                parameters={"field": "id"}
            ),
            "consistency_check": QualityRule(
                rule_id="consist_001",
                name="Cross-Field Consistency",
                rule_type="consistency",
                parameters={"fields": ["start_date", "end_date"], "condition": "start_date < end_date"}
            )
        }

        # Defect patterns for simulation
        self.defect_patterns = {
            "missing_values": lambda data: self._inject_missing_values(data),
            "out_of_range": lambda data: self._inject_range_violations(data),
            "format_errors": lambda data: self._inject_format_errors(data),
            "duplicates": lambda data: self._inject_duplicates(data),
            "inconsistencies": lambda data: self._inject_inconsistencies(data)
        }

    async def setup_simulation(
        self,
        scenario: QualityScenario = QualityScenario.NORMAL_QUALITY_CHECKS,
        data_volume: int = 1000,
        defect_rate: float = 0.1
    ):
        """Setup simulation environment"""

        self.simulation_metrics = SimulationMetrics()

        # Configure scenario-specific parameters
        if scenario == QualityScenario.HIGH_DEFECT_RATE:
            defect_rate = 0.3
        elif scenario == QualityScenario.COMPLIANCE_AUDIT:
            defect_rate = 0.05  # Very low defect rate expected
        elif scenario == QualityScenario.QUALITY_DEGRADATION:
            # Will gradually increase defect rate during simulation
            pass

        # Generate test dataset
        self.test_dataset = await self._generate_test_dataset(
            scenario, data_volume, defect_rate
        )

        # Setup quality rules based on scenario
        self.active_rules = self._select_rules_for_scenario(scenario)

        logger.info(f"Quality control simulation setup complete: {scenario.value}, "
                   f"{len(self.test_dataset)} data points, {len(self.active_rules)} rules")

    async def _generate_test_dataset(
        self,
        scenario: QualityScenario,
        volume: int,
        defect_rate: float
    ) -> List[QualityDataPoint]:
        """Generate test dataset based on scenario"""

        dataset = []
        data_types = ["numerical", "categorical", "text", "email", "financial"]

        for i in range(volume):
            # Select data type based on scenario
            if scenario == QualityScenario.FINANCIAL_DATA:
                data_type = "financial"
            else:
                # Use deterministic data type selection based on index
                data_type = data_types[i % len(data_types)]

            # Generate base data point
            data_point = await self._generate_data_point(data_type, i)

            # Inject defects deterministically based on defect rate and data point hash
            point_hash = hash(f"{data_type}_{i}_{scenario.value}")
            defect_probability = abs(point_hash) % 1000 / 1000.0  # 0-1 range
            if defect_probability < defect_rate:
                data_point = await self._inject_defect(data_point, scenario)

            dataset.append(data_point)

        return dataset

    async def _generate_data_point(self, data_type: str, index: int) -> QualityDataPoint:
        """Generate a single data point"""

        generator = self.data_generators.get(data_type, self._generate_numerical_data)
        value = generator()

        return QualityDataPoint(
            id=f"{data_type}_{index:06d}",
            data_type=data_type,
            value=value,
            expected_value=value,  # Initially clean data
            quality_score=1.0,
            metadata={"generation_method": "simulation", "data_type": data_type}
        )

    def _generate_numerical_data(self) -> float:
        """Generate realistic numerical data using statistical distributions"""
        # Use normal distribution with realistic parameters for quality metrics
        # Mean around 85 (good quality), std dev of 15
        value = np.random.normal(85, 15)
        # Clamp to realistic range [0, 100]
        return max(0, min(100, value))

    def _generate_categorical_data(self) -> str:
        """Generate realistic categorical data with weighted probabilities"""
        # Realistic distribution for quality grades
        categories = ["A", "B", "C", "D", "E"]
        # Weighted probabilities favoring higher grades (realistic for quality systems)
        weights = [0.35, 0.30, 0.20, 0.10, 0.05]  # A is most common, E is rarest
        return np.random.choice(categories, p=weights)

    def _generate_text_data(self) -> str:
        """Generate realistic text data using quality control terminology"""
        # Quality control phrases with realistic patterns
        subjects = ["quality", "control", "testing", "validation", "inspection", "assessment"]
        actions = ["analysis", "review", "verification", "measurement", "evaluation", "monitoring"]
        objects = ["data", "process", "output", "system", "procedure", "standard"]
        
        # Generate meaningful phrases
        subject = subjects[int(time.time() * 1000) % len(subjects)]
        action = actions[int(time.time() * 1001) % len(actions)]
        obj = objects[int(time.time() * 1002) % len(objects)]
        
        return f"{subject} {action} {obj}"

    def _generate_date_data(self) -> str:
        """Generate realistic date data following business patterns"""
        # Generate dates with business-day weighting (more recent dates more likely)
        days_back = int(np.random.exponential(30))  # Exponential distribution favors recent dates
        days_back = min(days_back, 365)  # Cap at 1 year
        
        base_date = datetime.now() - timedelta(days=days_back)
        
        # Skip weekends for business data
        while base_date.weekday() >= 5:  # Saturday=5, Sunday=6
            base_date -= timedelta(days=1)
        
        return base_date.strftime("%Y-%m-%d")

    def _generate_email_data(self) -> str:
        """Generate realistic email data with proper formatting"""
        # Realistic business email patterns
        first_names = ["john", "mary", "david", "sarah", "mike", "lisa", "alex", "emma"]
        last_names = ["smith", "johnson", "brown", "davis", "wilson", "garcia", "martinez", "taylor"]
        domains = ["company.com", "corp.net", "business.org", "enterprise.com"]
        
        # Deterministic selection based on time
        time_hash = int(time.time() * 1000)
        first = first_names[time_hash % len(first_names)]
        last = last_names[(time_hash // 10) % len(last_names)]
        domain = domains[(time_hash // 100) % len(domains)]
        
        return f"{first}.{last}@{domain}"

    def _generate_financial_data(self) -> float:
        """Generate realistic financial data using lognormal distribution"""
        # Financial data typically follows lognormal distribution
        # Mean log value of 6 gives median around $400, with reasonable spread
        value = np.random.lognormal(mean=6, sigma=1.2)
        # Cap at reasonable maximum and round to cents
        return round(min(value, 50000.00), 2)

    async def _inject_defect(
        self,
        data_point: QualityDataPoint,
        scenario: QualityScenario
    ) -> QualityDataPoint:
        """Inject defects into data point based on realistic quality patterns"""

        defect_types = list(self.defect_patterns.keys())
        
        # Use data point ID hash for deterministic but varied defect selection
        point_hash = hash(data_point.id + str(data_point.data_type))
        
        if scenario == QualityScenario.HIGH_DEFECT_RATE:
            # Multiple defects with realistic probability distribution
            # In high defect scenarios, 60% have 1 defect, 30% have 2, 10% have 3
            prob_selector = abs(point_hash) % 100
            if prob_selector < 60:
                num_defects = 1
            elif prob_selector < 90:
                num_defects = 2
            else:
                num_defects = 3
                
            # Select defects deterministically based on hash
            selected_defects = []
            for i in range(min(num_defects, len(defect_types))):
                defect_index = (point_hash + i * 37) % len(defect_types)
                defect_type = defect_types[defect_index]
                if defect_type not in selected_defects:
                    selected_defects.append(defect_type)
        else:
            # Single defect based on data type and historical patterns
            if data_point.data_type in ["numerical", "financial"]:
                # Numerical data more likely to have range or format violations
                preferred_defects = ["range_violation", "format_error", "precision_error"]
            elif data_point.data_type in ["text", "email"]:
                # Text data more likely to have format or encoding issues
                preferred_defects = ["format_error", "encoding_error", "length_violation"]
            else:
                preferred_defects = defect_types
            
            # Filter for available defects
            available_defects = [d for d in preferred_defects if d in defect_types]
            if not available_defects:
                available_defects = defect_types
            
            # Select deterministically
            defect_index = abs(point_hash) % len(available_defects)
            selected_defects = [available_defects[defect_index]]

        for defect_type in selected_defects:
            injector = self.defect_patterns[defect_type]
            data_point = injector(data_point)

        return data_point

    def _inject_missing_values(self, data_point: QualityDataPoint) -> QualityDataPoint:
        """Inject missing values"""
        data_point.value = None
        data_point.quality_score *= 0.5
        data_point.metadata["defect"] = "missing_value"
        return data_point

    def _inject_range_violations(self, data_point: QualityDataPoint) -> QualityDataPoint:
        """Inject range violations based on realistic patterns"""
        if data_point.data_type == "numerical":
            # Use data point hash for deterministic selection of violation type
            point_hash = hash(data_point.id)
            if abs(point_hash) % 2 == 0:
                data_point.value = -10 + (abs(point_hash) % 20)  # Negative range violation
            else:
                data_point.value = 150 + (abs(point_hash) % 50)  # Positive range violation
            data_point.quality_score *= 0.3
            data_point.metadata["defect"] = "range_violation"
        return data_point

    def _inject_format_errors(self, data_point: QualityDataPoint) -> QualityDataPoint:
        """Inject format errors"""
        if data_point.data_type == "email":
            data_point.value = "invalid_email_format"
            data_point.quality_score *= 0.2
            data_point.metadata["defect"] = "format_error"
        return data_point

    def _inject_duplicates(self, data_point: QualityDataPoint) -> QualityDataPoint:
        """Mark as potential duplicate"""
        data_point.id = "duplicate_id_001"  # Force duplicate ID
        data_point.quality_score *= 0.4
        data_point.metadata["defect"] = "duplicate"
        return data_point

    def _inject_inconsistencies(self, data_point: QualityDataPoint) -> QualityDataPoint:
        """Inject logical inconsistencies"""
        data_point.metadata["start_date"] = "2023-12-01"
        data_point.metadata["end_date"] = "2023-11-01"  # End before start
        data_point.quality_score *= 0.3
        data_point.metadata["defect"] = "inconsistency"
        return data_point

    def _select_rules_for_scenario(self, scenario: QualityScenario) -> List[QualityRule]:
        """Select appropriate rules for scenario"""

        if scenario == QualityScenario.COMPLIANCE_AUDIT:
            return list(self.quality_rules.values())  # All rules
        elif scenario == QualityScenario.PERFORMANCE_STRESS:
            return [self.quality_rules["range_check"]]  # Single rule for speed
        else:
            # Deterministic rule selection based on scenario
            all_rules = list(self.quality_rules.values())
            # Select between 2-4 rules based on scenario hash
            scenario_hash = hash(scenario.value)
            num_rules = 2 + (abs(scenario_hash) % 3)  # 2-4 rules
            
            selected_rules = []
            for i in range(min(num_rules, len(all_rules))):
                rule_index = (scenario_hash + i * 17) % len(all_rules)
                selected_rules.append(all_rules[rule_index])
            
            return selected_rules

    async def run_simulation(
        self,
        duration_seconds: int = 300,
        processing_rate: float = 10.0,  # items per second
        report_interval: int = 30
    ) -> SimulationMetrics:
        """Run quality control simulation"""

        self.simulation_running = True

        try:
            # Start simulation tasks
            await self._start_simulation_tasks(processing_rate)

            # Run simulation with periodic reporting
            start_time = datetime.now()
            last_report = start_time

            while self.simulation_running:
                current_time = datetime.now()
                elapsed = (current_time - start_time).total_seconds()

                if elapsed >= duration_seconds:
                    break

                # Generate periodic report
                if (current_time - last_report).total_seconds() >= report_interval:
                    await self._generate_progress_report(elapsed, duration_seconds)
                    last_report = current_time

                await asyncio.sleep(1)

            # Calculate final metrics
            await self._calculate_final_metrics()

            logger.info("Quality control simulation completed successfully")

        except Exception as e:
            logger.error(f"Quality control simulation failed: {e}")
            raise

        finally:
            await self._stop_simulation_tasks()
            self.simulation_running = False

        return self.simulation_metrics

    async def _start_simulation_tasks(self, processing_rate: float):
        """Start simulation tasks"""

        # Data processing task
        task = asyncio.create_task(self._process_data_stream(processing_rate))
        self.simulation_tasks.append(task)

        # Batch processing task
        task = asyncio.create_task(self._run_batch_processing())
        self.simulation_tasks.append(task)

        # Real-time monitoring task
        task = asyncio.create_task(self._run_real_time_monitoring())
        self.simulation_tasks.append(task)

    async def _process_data_stream(self, processing_rate: float):
        """Process data stream at specified rate"""

        interval = 1.0 / processing_rate
        processed_count = 0

        while self.simulation_running and processed_count < len(self.test_dataset):
            try:
                data_point = self.test_dataset[processed_count]

                # Simulate quality check processing
                start_time = datetime.now()
                quality_result = await self._perform_quality_check(data_point)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000

                # Update metrics
                self.simulation_metrics.total_checks += 1
                self.simulation_metrics.processing_time_ms.append(processing_time)

                if quality_result["passed"]:
                    self.simulation_metrics.passed_checks += 1
                else:
                    self.simulation_metrics.failed_checks += 1
                    if quality_result.get("severity") == "critical":
                        self.simulation_metrics.critical_failures += 1

                # Update rule performance
                for rule_result in quality_result.get("rule_results", []):
                    rule_id = rule_result["rule_id"]
                    if rule_id not in self.simulation_metrics.rule_performance:
                        self.simulation_metrics.rule_performance[rule_id] = {"passed": 0, "failed": 0}

                    if rule_result["passed"]:
                        self.simulation_metrics.rule_performance[rule_id]["passed"] += 1
                    else:
                        self.simulation_metrics.rule_performance[rule_id]["failed"] += 1

                processed_count += 1

                # Wait for next processing cycle
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Data processing error: {e}")
                await asyncio.sleep(0.1)

    async def _perform_quality_check(self, data_point: QualityDataPoint) -> Dict[str, Any]:
        """Perform quality check on data point"""

        result = {
            "data_point_id": data_point.id,
            "passed": True,
            "quality_score": data_point.quality_score,
            "rule_results": [],
            "severity": "low"
        }

        # Apply each active rule
        for rule in self.active_rules:
            rule_result = await self._apply_quality_rule(data_point, rule)
            result["rule_results"].append(rule_result)

            if not rule_result["passed"]:
                result["passed"] = False
                if rule.severity == "critical":
                    result["severity"] = "critical"
                elif rule.severity == "high" and result["severity"] != "critical":
                    result["severity"] = "high"

        # Simulate processing delay based on complexity
        processing_delay = len(self.active_rules) * 0.001  # 1ms per rule
        await asyncio.sleep(processing_delay)

        return result

    async def _apply_quality_rule(
        self,
        data_point: QualityDataPoint,
        rule: QualityRule
    ) -> Dict[str, Any]:
        """Apply single quality rule to data point"""

        result = {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "passed": True,
            "details": {}
        }

        try:
            if rule.rule_type == "range":
                min_val = rule.parameters.get("min_value", 0)
                max_val = rule.parameters.get("max_value", 100)

                if data_point.value is not None:
                    if isinstance(data_point.value, (int, float)):
                        if not (min_val <= data_point.value <= max_val):
                            result["passed"] = False
                            result["details"]["violation"] = f"Value {data_point.value} not in range [{min_val}, {max_val}]"

            elif rule.rule_type == "format":
                pattern = rule.parameters.get("pattern", "")
                if data_point.value is not None:
                    import re
                    if not re.match(pattern, str(data_point.value)):
                        result["passed"] = False
                        result["details"]["violation"] = f"Value doesn't match pattern {pattern}"

            elif rule.rule_type == "completeness":
                if data_point.value is None or data_point.value == "":
                    result["passed"] = False
                    result["details"]["violation"] = "Missing required value"

            elif rule.rule_type == "uniqueness":
                # Simulate uniqueness check (would need global state in real implementation)
                if "duplicate" in data_point.metadata.get("defect", ""):
                    result["passed"] = False
                    result["details"]["violation"] = "Duplicate value detected"

            elif rule.rule_type == "consistency":
                # Check cross-field consistency
                if "inconsistency" in data_point.metadata.get("defect", ""):
                    result["passed"] = False
                    result["details"]["violation"] = "Cross-field consistency violation"

            # Simulate deterministic failures based on rule failure probability
            if result["passed"]:
                # Use data point and rule combination for deterministic failure simulation
                failure_hash = hash(f"{data_point.id}_{rule.rule_id}_{data_point.data_type}")
                failure_probability = abs(failure_hash) % 1000 / 1000.0
                
                if failure_probability < rule.failure_probability:
                    result["passed"] = False
                    result["details"]["violation"] = "Deterministic simulation failure based on rule probability"

        except Exception as e:
            result["passed"] = False
            result["details"]["error"] = str(e)

        return result

    async def _run_batch_processing(self):
        """Run batch processing simulation"""

        batch_size = 100
        batch_interval = 60  # Process batch every minute

        while self.simulation_running:
            try:
                # Simulate batch processing
                batch_start = datetime.now()

                # Process a batch with deterministic selection
                batch_start_idx = (int(datetime.now().timestamp()) // batch_interval) % len(self.test_dataset)
                batch_end_idx = min(batch_start_idx + batch_size, len(self.test_dataset))
                
                if batch_end_idx - batch_start_idx < batch_size and len(self.test_dataset) >= batch_size:
                    # Wrap around if needed
                    batch_data = (self.test_dataset[batch_start_idx:] + 
                                self.test_dataset[:batch_size - (batch_end_idx - batch_start_idx)])
                else:
                    batch_data = self.test_dataset[batch_start_idx:batch_end_idx]

                batch_results = []
                for data_point in batch_data:
                    result = await self._perform_quality_check(data_point)
                    batch_results.append(result)

                batch_duration = (datetime.now() - batch_start).total_seconds()

                # Update batch statistics
                self.simulation_metrics.batch_statistics[f"batch_{datetime.now().isoformat()}"] = {
                    "size": len(batch_data),
                    "duration_seconds": batch_duration,
                    "passed_count": sum(1 for r in batch_results if r["passed"]),
                    "failed_count": sum(1 for r in batch_results if not r["passed"])
                }

                await asyncio.sleep(batch_interval)

            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(10)

    async def _run_real_time_monitoring(self):
        """Run real-time monitoring simulation"""

        while self.simulation_running:
            try:
                # Calculate real-time metrics
                # Calculate defect rates by data type
                defect_rates = {}
                for data_point in self.test_dataset:
                    data_type = data_point.data_type
                    if data_type not in defect_rates:
                        defect_rates[data_type] = {"total": 0, "defects": 0}

                    defect_rates[data_type]["total"] += 1
                    if data_point.quality_score < 1.0:
                        defect_rates[data_type]["defects"] += 1

                # Calculate percentages
                for data_type, stats in defect_rates.items():
                    rate = (stats["defects"] / stats["total"]) * 100 if stats["total"] > 0 else 0
                    self.simulation_metrics.defect_rates[data_type] = rate

                await asyncio.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"Real-time monitoring error: {e}")
                await asyncio.sleep(5)

    async def _generate_progress_report(self, elapsed: float, total: float):
        """Generate progress report during simulation"""

        progress = (elapsed / total) * 100

        total_checks = self.simulation_metrics.total_checks
        if total_checks > 0:
            pass_rate = (self.simulation_metrics.passed_checks / total_checks) * 100
            avg_processing_time = statistics.mean(self.simulation_metrics.processing_time_ms)
        else:
            pass_rate = 0
            avg_processing_time = 0

        logger.info(f"Quality Control Simulation Progress: {progress:.1f}% | "
                   f"Checks: {total_checks} | "
                   f"Pass Rate: {pass_rate:.1f}% | "
                   f"Avg Time: {avg_processing_time:.2f}ms")

    async def _calculate_final_metrics(self):
        """Calculate final simulation metrics"""

        # Average processing time is already calculated in the metrics

        # Calculate overall quality score
        total_score = sum(dp.quality_score for dp in self.test_dataset)
        avg_quality = total_score / len(self.test_dataset) if self.test_dataset else 0
        self.simulation_metrics.average_quality_score = avg_quality

    async def _stop_simulation_tasks(self):
        """Stop all simulation tasks"""

        for task in self.simulation_tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        if self.simulation_tasks:
            await asyncio.gather(*self.simulation_tasks, return_exceptions=True)

        self.simulation_tasks.clear()

    async def cleanup_simulation(self):
        """Clean up simulation environment"""

        self.simulation_running = False
        await self._stop_simulation_tasks()

        # Clear test data
        self.test_dataset.clear()
        self.active_rules.clear()

    def get_simulation_report(self) -> Dict[str, Any]:
        """Get comprehensive simulation report"""

        total_checks = self.simulation_metrics.total_checks

        percentiles = {}
        if self.simulation_metrics.processing_time_ms:
            sorted_times = sorted(self.simulation_metrics.processing_time_ms)
            percentiles = {
                "p50": sorted_times[len(sorted_times) // 2],
                "p95": sorted_times[int(len(sorted_times) * 0.95)],
                "p99": sorted_times[int(len(sorted_times) * 0.99)]
            }

        return {
            "summary": {
                "total_checks": total_checks,
                "passed_checks": self.simulation_metrics.passed_checks,
                "failed_checks": self.simulation_metrics.failed_checks,
                "critical_failures": self.simulation_metrics.critical_failures,
                "pass_rate_percent": (self.simulation_metrics.passed_checks / total_checks * 100) if total_checks > 0 else 0,
                "average_quality_score": round(self.simulation_metrics.average_quality_score, 3)
            },
            "performance": {
                "processing_time_percentiles_ms": percentiles,
                "average_processing_time_ms": statistics.mean(self.simulation_metrics.processing_time_ms) if self.simulation_metrics.processing_time_ms else 0,
                "throughput_checks_per_second": total_checks / max(1, max(self.simulation_metrics.processing_time_ms)) * 1000 if self.simulation_metrics.processing_time_ms else 0
            },
            "defect_analysis": {
                "defect_rates_by_type": self.simulation_metrics.defect_rates,
                "rule_performance": self.simulation_metrics.rule_performance
            },
            "batch_processing": self.simulation_metrics.batch_statistics,
            "dataset_info": {
                "total_data_points": len(self.test_dataset),
                "data_types": list(set(dp.data_type for dp in self.test_dataset)),
                "active_rules": len(self.active_rules)
            }
        }


# Convenience functions for common simulation scenarios
async def run_normal_quality_simulation(
    quality_control_agent,
    duration_seconds: int = 300
) -> Dict[str, Any]:
    """Run normal quality control simulation"""

    simulator = QualityControlSimulator(quality_control_agent)
    await simulator.setup_simulation(
        scenario=QualityScenario.NORMAL_QUALITY_CHECKS,
        data_volume=1000,
        defect_rate=0.1
    )

    try:
        await simulator.run_simulation(
            duration_seconds=duration_seconds,
            processing_rate=5.0
        )
        return simulator.get_simulation_report()
    finally:
        await simulator.cleanup_simulation()

async def run_high_defect_simulation(
    quality_control_agent,
    duration_seconds: int = 180
) -> Dict[str, Any]:
    """Run high defect rate simulation"""

    simulator = QualityControlSimulator(quality_control_agent)
    await simulator.setup_simulation(
        scenario=QualityScenario.HIGH_DEFECT_RATE,
        data_volume=500,
        defect_rate=0.3
    )

    try:
        await simulator.run_simulation(
            duration_seconds=duration_seconds,
            processing_rate=8.0
        )
        return simulator.get_simulation_report()
    finally:
        await simulator.cleanup_simulation()

async def run_performance_stress_simulation(
    quality_control_agent,
    duration_seconds: int = 120
) -> Dict[str, Any]:
    """Run performance stress simulation"""

    simulator = QualityControlSimulator(quality_control_agent)
    await simulator.setup_simulation(
        scenario=QualityScenario.PERFORMANCE_STRESS,
        data_volume=2000,
        defect_rate=0.05
    )

    try:
        await simulator.run_simulation(
            duration_seconds=duration_seconds,
            processing_rate=20.0  # High processing rate
        )
        return simulator.get_simulation_report()
    finally:
        await simulator.cleanup_simulation()


# Create simulator instance
def create_quality_control_simulator(quality_control_agent) -> QualityControlSimulator:
    """Create a new quality control simulator instance"""
    return QualityControlSimulator(quality_control_agent)
