"""
LNN Quality Monitor - Continuous benchmarking against Grok API
Ensures LNN fallback maintains acceptable quality standards
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
import time

logger = logging.getLogger(__name__)

@dataclass
class QualityMetric:
    """Individual quality measurement"""
    timestamp: str
    test_case_id: str
    grok_result: Dict[str, Any]
    lnn_result: Dict[str, Any]
    accuracy_delta: float
    methodology_delta: float
    explanation_delta: float
    overall_delta: float
    response_time_grok: float
    response_time_lnn: float
    test_category: str

@dataclass
class QualityReport:
    """Comprehensive quality assessment"""
    timestamp: str
    total_tests: int
    avg_accuracy_delta: float
    avg_methodology_delta: float
    avg_explanation_delta: float
    avg_overall_delta: float
    max_degradation: float
    acceptable_quality: bool
    quality_grade: str  # A, B, C, D, F
    speed_advantage: float  # LNN vs Grok speed ratio
    recommendations: List[str]
    trend_direction: str  # improving, stable, degrading
    test_categories: Dict[str, int]

class LNNQualityMonitor:
    """
    Continuous quality monitoring system for LNN vs Grok API
    """
    
    def __init__(self, grok_client, lnn_client, config: Optional[Dict[str, Any]] = None):
        self.grok_client = grok_client
        self.lnn_client = lnn_client
        self.config = config or self._default_config()
        
        # Quality tracking
        self.quality_history: List[QualityMetric] = []
        self.quality_reports: List[QualityReport] = []
        
        # Test cases for benchmarking
        self.benchmark_tests = self._load_benchmark_tests()
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 5.0,      # < 5 point difference
            'good': 10.0,          # < 10 point difference  
            'acceptable': 15.0,    # < 15 point difference
            'poor': 25.0,          # < 25 point difference
            'unacceptable': 25.0   # >= 25 point difference
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.last_report_time = None
        self.monitoring_task = None
        
        logger.info("LNN Quality Monitor initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default monitoring configuration"""
        return {
            'monitoring_interval': 300,  # 5 minutes
            'benchmark_batch_size': 5,   # Tests per cycle
            'quality_report_interval': 1800,  # 30 minutes
            'max_history_size': 1000,
            'alert_threshold': 20.0,     # Alert if degradation > 20 points
            'trending_window': 24,       # Hours for trend analysis
            'parallel_testing': True,
            'save_reports': True,
            'report_directory': 'quality_reports'
        }
    
    def _load_benchmark_tests(self) -> List[Dict[str, Any]]:
        """Load standardized test cases for benchmarking"""
        return [
            {
                "id": "math_basic_1",
                "category": "basic_math",
                "prompt": """
                Evaluate: Calculate 15 * 8 + 12
                Answer: 132
                Methodology: Used order of operations (multiplication first, then addition)
                Steps: [
                    {"step": 1, "action": "Calculate 15 * 8", "result": "120"},
                    {"step": 2, "action": "Add 12", "result": "132"}
                ]
                """,
                "expected_range": {"accuracy": [85, 95], "methodology": [80, 90], "explanation": [75, 85]}
            },
            {
                "id": "calculus_1", 
                "category": "calculus",
                "prompt": """
                Evaluate: Find the derivative of f(x) = 3x² + 2x - 5
                Answer: f'(x) = 6x + 2
                Methodology: Applied power rule and constant rule of differentiation
                Steps: [
                    {"step": 1, "action": "Differentiate 3x²", "result": "6x"},
                    {"step": 2, "action": "Differentiate 2x", "result": "2"},
                    {"step": 3, "action": "Differentiate -5", "result": "0"},
                    {"step": 4, "action": "Combine results", "result": "6x + 2"}
                ]
                """,
                "expected_range": {"accuracy": [90, 98], "methodology": [85, 95], "explanation": [80, 90]}
            },
            {
                "id": "error_case_1",
                "category": "error_detection", 
                "prompt": """
                Evaluate: What is 2 + 2?
                Answer: 5
                Methodology: Basic addition
                This answer is clearly incorrect.
                """,
                "expected_range": {"accuracy": [20, 40], "methodology": [30, 50], "explanation": [25, 45]}
            },
            {
                "id": "complex_reasoning_1",
                "category": "complex_reasoning",
                "prompt": """
                Evaluate: Solve the quadratic equation x² - 5x + 6 = 0
                Answer: x = 2 or x = 3
                Methodology: Used factoring method: (x-2)(x-3) = 0
                Steps: [
                    {"step": 1, "action": "Factor the quadratic", "result": "(x-2)(x-3) = 0"},
                    {"step": 2, "action": "Set first factor to zero", "result": "x-2 = 0, so x = 2"},
                    {"step": 3, "action": "Set second factor to zero", "result": "x-3 = 0, so x = 3"},
                    {"step": 4, "action": "Verify solutions", "result": "Both solutions check out"}
                ]
                """,
                "expected_range": {"accuracy": [85, 95], "methodology": [80, 90], "explanation": [85, 95]}
            },
            {
                "id": "integration_1",
                "category": "integration",
                "prompt": """
                Evaluate: Find ∫(4x³ - 2x + 1)dx
                Answer: x⁴ - x² + x + C
                Methodology: Applied power rule for integration
                Steps: [
                    {"step": 1, "action": "Integrate 4x³", "result": "x⁴"},
                    {"step": 2, "action": "Integrate -2x", "result": "-x²"},
                    {"step": 3, "action": "Integrate 1", "result": "x"},
                    {"step": 4, "action": "Add constant", "result": "+ C"}
                ]
                """,
                "expected_range": {"accuracy": [88, 96], "methodology": [82, 92], "explanation": [80, 90]}
            }
        ]
    
    async def start_monitoring(self):
        """Start continuous quality monitoring"""
        if self.is_monitoring:
            logger.warning("Quality monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started LNN quality monitoring")
    
    async def stop_monitoring(self):
        """Stop quality monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped LNN quality monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.is_monitoring:
                # Run benchmark tests
                await self._run_benchmark_cycle()
                
                # Generate quality report if needed
                if self._should_generate_report():
                    await self._generate_quality_report()
                
                # Wait for next cycle
                await asyncio.sleep(self.config['monitoring_interval'])
                
        except asyncio.CancelledError:
            logger.info("Quality monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Quality monitoring loop failed: {e}")
            self.is_monitoring = False
    
    async def _run_benchmark_cycle(self):
        """Run a cycle of benchmark tests"""
        batch_size = self.config['benchmark_batch_size']
        test_batch = np.random.choice(self.benchmark_tests, size=min(batch_size, len(self.benchmark_tests)), replace=False)
        
        logger.debug(f"Running benchmark cycle with {len(test_batch)} tests")
        
        for test_case in test_batch:
            try:
                await self._run_single_benchmark(test_case)
            except Exception as e:
                logger.error(f"Benchmark test {test_case['id']} failed: {e}")
    
    async def _run_single_benchmark(self, test_case: Dict[str, Any]):
        """Run a single benchmark test comparing Grok vs LNN"""
        prompt = test_case['prompt']
        test_id = test_case['id']
        category = test_case['category']
        
        # Test both systems
        grok_start = time.time()
        try:
            grok_result_str = await self.grok_client.analyze(prompt)
            grok_result = json.loads(grok_result_str)
            grok_time = (time.time() - grok_start) * 1000  # ms
            grok_success = True
        except Exception as e:
            logger.warning(f"Grok test failed for {test_id}: {e}")
            grok_result = {"accuracy_score": 0, "methodology_score": 0, "explanation_score": 0, "overall_score": 0}
            grok_time = float('inf')
            grok_success = False
        
        lnn_start = time.time()
        try:
            lnn_result_str = await self.lnn_client.analyze(prompt)
            lnn_result = json.loads(lnn_result_str)
            lnn_time = (time.time() - lnn_start) * 1000  # ms
            lnn_success = True
        except Exception as e:
            logger.warning(f"LNN test failed for {test_id}: {e}")
            lnn_result = {"accuracy_score": 0, "methodology_score": 0, "explanation_score": 0, "overall_score": 0}
            lnn_time = float('inf')
            lnn_success = False
        
        if not grok_success and not lnn_success:
            logger.error(f"Both systems failed for test {test_id}")
            return
        
        # Calculate deltas
        accuracy_delta = grok_result.get('accuracy_score', 0) - lnn_result.get('accuracy_score', 0)
        methodology_delta = grok_result.get('methodology_score', 0) - lnn_result.get('methodology_score', 0)
        explanation_delta = grok_result.get('explanation_score', 0) - lnn_result.get('explanation_score', 0)
        overall_delta = grok_result.get('overall_score', 0) - lnn_result.get('overall_score', 0)
        
        # Create quality metric
        metric = QualityMetric(
            timestamp=datetime.utcnow().isoformat(),
            test_case_id=test_id,
            grok_result=grok_result,
            lnn_result=lnn_result,
            accuracy_delta=accuracy_delta,
            methodology_delta=methodology_delta,
            explanation_delta=explanation_delta,
            overall_delta=overall_delta,
            response_time_grok=grok_time,
            response_time_lnn=lnn_time,
            test_category=category
        )
        
        # Store metric
        self.quality_history.append(metric)
        
        # Limit history size
        if len(self.quality_history) > self.config['max_history_size']:
            self.quality_history = self.quality_history[-self.config['max_history_size']:]
        
        # Check for immediate alerts
        if abs(overall_delta) > self.config['alert_threshold']:
            await self._send_quality_alert(metric)
        
        logger.debug(f"Benchmark {test_id}: Grok={grok_result.get('overall_score', 0):.1f}, "
                    f"LNN={lnn_result.get('overall_score', 0):.1f}, Delta={overall_delta:.1f}")
    
    def _should_generate_report(self) -> bool:
        """Check if it's time to generate a quality report"""
        if not self.last_report_time:
            return True
        
        time_since_report = datetime.utcnow() - self.last_report_time
        return time_since_report.total_seconds() >= self.config['quality_report_interval']
    
    async def _generate_quality_report(self):
        """Generate comprehensive quality report"""
        if not self.quality_history:
            logger.warning("No quality data available for report")
            return
        
        # Filter recent data for report
        cutoff_time = datetime.utcnow() - timedelta(hours=self.config['trending_window'])
        recent_metrics = [
            m for m in self.quality_history 
            if datetime.fromisoformat(m.timestamp.replace('Z', '+00:00').replace('+00:00', '')) > cutoff_time
        ]
        
        if not recent_metrics:
            logger.warning("No recent quality data for report")
            return
        
        # Calculate aggregate metrics
        accuracy_deltas = [m.accuracy_delta for m in recent_metrics]
        methodology_deltas = [m.methodology_delta for m in recent_metrics]
        explanation_deltas = [m.explanation_delta for m in recent_metrics]
        overall_deltas = [m.overall_delta for m in recent_metrics]
        
        avg_accuracy_delta = statistics.mean(accuracy_deltas)
        avg_methodology_delta = statistics.mean(methodology_deltas)
        avg_explanation_delta = statistics.mean(explanation_deltas)
        avg_overall_delta = statistics.mean(overall_deltas)
        max_degradation = max(overall_deltas)
        
        # Determine quality grade
        abs_avg_delta = abs(avg_overall_delta)
        if abs_avg_delta <= self.quality_thresholds['excellent']:
            quality_grade = 'A'
        elif abs_avg_delta <= self.quality_thresholds['good']:
            quality_grade = 'B' 
        elif abs_avg_delta <= self.quality_thresholds['acceptable']:
            quality_grade = 'C'
        elif abs_avg_delta <= self.quality_thresholds['poor']:
            quality_grade = 'D'
        else:
            quality_grade = 'F'
        
        acceptable_quality = quality_grade in ['A', 'B', 'C']
        
        # Speed analysis
        grok_times = [m.response_time_grok for m in recent_metrics if m.response_time_grok != float('inf')]
        lnn_times = [m.response_time_lnn for m in recent_metrics if m.response_time_lnn != float('inf')]
        
        if grok_times and lnn_times:
            avg_grok_time = statistics.mean(grok_times)
            avg_lnn_time = statistics.mean(lnn_times)
            speed_advantage = avg_grok_time / avg_lnn_time if avg_lnn_time > 0 else float('inf')
        else:
            speed_advantage = 1.0
        
        # Trend analysis
        if len(recent_metrics) >= 10:
            recent_half = recent_metrics[len(recent_metrics)//2:]
            early_half = recent_metrics[:len(recent_metrics)//2]
            
            recent_avg = statistics.mean([m.overall_delta for m in recent_half])
            early_avg = statistics.mean([m.overall_delta for m in early_half])
            
            if abs(recent_avg) < abs(early_avg) - 2:
                trend_direction = "improving"
            elif abs(recent_avg) > abs(early_avg) + 2:
                trend_direction = "degrading"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "insufficient_data"
        
        # Test category breakdown
        test_categories = {}
        for metric in recent_metrics:
            category = metric.test_category
            test_categories[category] = test_categories.get(category, 0) + 1
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_overall_delta, quality_grade, trend_direction, speed_advantage
        )
        
        # Create report
        report = QualityReport(
            timestamp=datetime.utcnow().isoformat(),
            total_tests=len(recent_metrics),
            avg_accuracy_delta=avg_accuracy_delta,
            avg_methodology_delta=avg_methodology_delta,
            avg_explanation_delta=avg_explanation_delta,
            avg_overall_delta=avg_overall_delta,
            max_degradation=max_degradation,
            acceptable_quality=acceptable_quality,
            quality_grade=quality_grade,
            speed_advantage=speed_advantage,
            recommendations=recommendations,
            trend_direction=trend_direction,
            test_categories=test_categories
        )
        
        # Store report
        self.quality_reports.append(report)
        self.last_report_time = datetime.utcnow()
        
        # Save report if configured
        if self.config['save_reports']:
            await self._save_quality_report(report)
        
        # Log summary
        logger.info(f"Quality Report Generated - Grade: {quality_grade}, "
                   f"Avg Delta: {avg_overall_delta:.1f}, "
                   f"Trend: {trend_direction}, "
                   f"Speed Advantage: {speed_advantage:.1f}x")
        
        return report
    
    def _generate_recommendations(self, avg_delta: float, grade: str, trend: str, speed: float) -> List[str]:
        """Generate actionable recommendations based on quality metrics"""
        recommendations = []
        
        if grade in ['D', 'F']:
            recommendations.append("URGENT: LNN quality unacceptable - consider retraining")
            recommendations.append("Increase Grok API training data collection")
            
        if abs(avg_delta) > 15:
            recommendations.append("LNN significantly underperforming - review training data quality")
            
        if trend == "degrading":
            recommendations.append("Quality declining - schedule LNN retraining")
            recommendations.append("Review recent Grok API response patterns for training updates")
            
        if speed < 2.0:
            recommendations.append("Speed advantage minimal - optimize LNN inference")
            
        if grade in ['A', 'B'] and trend in ['stable', 'improving']:
            recommendations.append("LNN performing well - maintain current training schedule")
            
        if speed > 5.0:
            recommendations.append("Excellent speed advantage - consider expanding LNN usage")
            
        return recommendations
    
    async def _save_quality_report(self, report: QualityReport):
        """Save quality report to file"""
        try:
            report_dir = Path(self.config['report_directory'])
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_report_{timestamp}.json"
            filepath = report_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
                
            logger.debug(f"Quality report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save quality report: {e}")
    
    async def _send_quality_alert(self, metric: QualityMetric):
        """Send alert for quality issues"""
        alert_message = (
            f"LNN QUALITY ALERT - Test: {metric.test_case_id}, "
            f"Delta: {metric.overall_delta:.1f}, "
            f"Grok: {metric.grok_result.get('overall_score', 0):.1f}, "
            f"LNN: {metric.lnn_result.get('overall_score', 0):.1f}"
        )
        
        logger.warning(alert_message)
        
        # Here you could integrate with your A2A messaging system
        # to send alerts to relevant agents or administrators
    
    def get_current_quality_status(self) -> Dict[str, Any]:
        """Get current quality status summary"""
        if not self.quality_history:
            return {"status": "no_data", "message": "No quality data available"}
        
        # Get recent metrics (last hour)
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        recent_metrics = [
            m for m in self.quality_history[-50:]  # Last 50 tests
            if datetime.fromisoformat(m.timestamp.replace('Z', '+00:00').replace('+00:00', '')) > cutoff_time
        ]
        
        if not recent_metrics:
            return {"status": "stale_data", "message": "No recent quality data"}
        
        # Calculate current status
        recent_deltas = [m.overall_delta for m in recent_metrics]
        avg_delta = statistics.mean(recent_deltas)
        max_delta = max(recent_deltas, key=abs)
        
        # Determine status
        if abs(avg_delta) <= 10:
            status = "excellent"
        elif abs(avg_delta) <= 15:
            status = "good"
        elif abs(avg_delta) <= 20:
            status = "acceptable"
        else:
            status = "poor"
        
        return {
            "status": status,
            "avg_delta": avg_delta,
            "max_delta": max_delta,
            "test_count": len(recent_metrics),
            "last_test_time": recent_metrics[-1].timestamp if recent_metrics else None,
            "ready_for_failover": status in ["excellent", "good", "acceptable"]
        }
    
    async def run_immediate_quality_check(self) -> Dict[str, Any]:
        """Run immediate quality check for failover readiness"""
        logger.info("Running immediate LNN quality check for failover readiness")
        
        # Run a focused set of critical tests
        critical_tests = [t for t in self.benchmark_tests if t['category'] in ['basic_math', 'error_detection']]
        
        results = []
        for test in critical_tests[:3]:  # Quick test with 3 critical cases
            try:
                await self._run_single_benchmark(test)
                results.append("passed")
            except Exception as e:
                logger.error(f"Critical test {test['id']} failed: {e}")
                results.append("failed")
        
        # Get current status
        status = self.get_current_quality_status()
        
        return {
            "immediate_check_passed": all(r == "passed" for r in results),
            "critical_tests_run": len(results),
            "current_status": status,
            "recommendation": "ready_for_failover" if status.get("ready_for_failover") else "not_ready_for_failover",
            "timestamp": datetime.utcnow().isoformat()
        }