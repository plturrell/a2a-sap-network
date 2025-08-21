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
        
        # Financial category weights for specialized scoring
        self.category_weights = {
            'variance_analysis': 1.2,
            'portfolio_variance': 1.2,
            'trend_analysis': 1.1,
            'temporal_analysis': 1.1,
            'financial_metrics': 1.15,
            'covariance_analysis': 1.2,
            'risk_analysis': 1.25,
            'volatility_modeling': 1.3,
            'error_detection': 0.8
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.last_report_time = None
        self.monitoring_task = None
        
        # Category performance tracking
        self._category_performance = {}
        
        logger.info("LNN Quality Monitor initialized for financial metrics")
    
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
        """Load financial and statistical test cases for variance, trend, and temporal analysis"""
        return [
            # Financial variance analysis
            {
                "id": "variance_analysis_1",
                "category": "variance_analysis",
                "prompt": """
                Evaluate: Calculate the variance of daily returns: [0.02, -0.01, 0.03, -0.02, 0.01]
                Answer: 0.00038
                Methodology: Calculate mean, then sum of squared deviations divided by n-1
                Steps: [
                    {"step": 1, "action": "Calculate mean", "result": "0.006"},
                    {"step": 2, "action": "Calculate squared deviations", "result": "[0.000196, 0.000256, 0.000576, 0.000676, 0.000016]"},
                    {"step": 3, "action": "Sum squared deviations", "result": "0.00172"},
                    {"step": 4, "action": "Divide by n-1 (4)", "result": "0.00043"}
                ]
                """,
                "expected_range": {"accuracy": [85, 95], "methodology": [85, 95], "explanation": [80, 90]}
            },
            {
                "id": "portfolio_variance_1",
                "category": "portfolio_variance",
                "prompt": """
                Evaluate: Two assets with weights 0.6 and 0.4, variances 0.04 and 0.09, correlation 0.3. Calculate portfolio variance.
                Answer: 0.0436
                Methodology: σ²p = w₁²σ₁² + w₂²σ₂² + 2w₁w₂ρσ₁σ₂
                Steps: [
                    {"step": 1, "action": "Calculate w₁²σ₁²", "result": "0.36 * 0.04 = 0.0144"},
                    {"step": 2, "action": "Calculate w₂²σ₂²", "result": "0.16 * 0.09 = 0.0144"},
                    {"step": 3, "action": "Calculate 2w₁w₂ρσ₁σ₂", "result": "2 * 0.6 * 0.4 * 0.3 * 0.2 * 0.3 = 0.00864"},
                    {"step": 4, "action": "Sum all terms", "result": "0.0144 + 0.0144 + 0.00864 = 0.03744"}
                ]
                """,
                "expected_range": {"accuracy": [82, 92], "methodology": [85, 95], "explanation": [80, 90]}
            },
            
            # Trend analysis
            {
                "id": "linear_trend_1",
                "category": "trend_analysis",
                "prompt": """
                Evaluate: Stock prices over 5 days: [100, 102, 105, 103, 107]. Calculate the linear trend slope.
                Answer: 1.6
                Methodology: Linear regression using least squares
                Steps: [
                    {"step": 1, "action": "Set up x values (days)", "result": "[1, 2, 3, 4, 5]"},
                    {"step": 2, "action": "Calculate means", "result": "x̄ = 3, ȳ = 103.4"},
                    {"step": 3, "action": "Calculate Σ(x-x̄)(y-ȳ)", "result": "16"},
                    {"step": 4, "action": "Calculate Σ(x-x̄)²", "result": "10"},
                    {"step": 5, "action": "Slope = 16/10", "result": "1.6"}
                ]
                """,
                "expected_range": {"accuracy": [88, 96], "methodology": [85, 95], "explanation": [82, 92]}
            },
            {
                "id": "moving_average_trend",
                "category": "trend_analysis",
                "prompt": """
                Evaluate: Calculate 3-day simple moving average for prices [50, 52, 48, 53, 55, 51, 54]
                Answer: [50, 51, 52, 53, 53]
                Methodology: Average of each 3-day window
                Steps: [
                    {"step": 1, "action": "Days 1-3: (50+52+48)/3", "result": "50"},
                    {"step": 2, "action": "Days 2-4: (52+48+53)/3", "result": "51"},
                    {"step": 3, "action": "Days 3-5: (48+53+55)/3", "result": "52"},
                    {"step": 4, "action": "Days 4-6: (53+55+51)/3", "result": "53"},
                    {"step": 5, "action": "Days 5-7: (55+51+54)/3", "result": "53.33 ≈ 53"}
                ]
                """,
                "expected_range": {"accuracy": [85, 95], "methodology": [88, 96], "explanation": [80, 90]}
            },
            
            # Temporal analysis
            {
                "id": "time_series_decomposition",
                "category": "temporal_analysis",
                "prompt": """
                Evaluate: Monthly sales show 12-month seasonality. Q1 sales: [100, 110, 120]. Calculate seasonal index if annual average is 115.
                Answer: Q1 seasonal index = 0.957
                Methodology: Average Q1 sales / Annual average
                Steps: [
                    {"step": 1, "action": "Calculate Q1 average", "result": "(100+110+120)/3 = 110"},
                    {"step": 2, "action": "Divide by annual average", "result": "110/115 = 0.957"},
                    {"step": 3, "action": "Interpretation", "result": "Q1 sales are 4.3% below annual average"}
                ]
                """,
                "expected_range": {"accuracy": [86, 94], "methodology": [85, 95], "explanation": [83, 93]}
            },
            {
                "id": "autocorrelation_1",
                "category": "temporal_analysis",
                "prompt": """
                Evaluate: Calculate lag-1 autocorrelation for returns: [0.01, -0.02, 0.03, -0.01, 0.02]
                Answer: -0.65
                Methodology: Correlation between series and lagged series
                Steps: [
                    {"step": 1, "action": "Create lagged pairs", "result": "[(0.01,-0.02), (-0.02,0.03), (0.03,-0.01), (-0.01,0.02)]"},
                    {"step": 2, "action": "Calculate correlation coefficient", "result": "-0.65"},
                    {"step": 3, "action": "Interpretation", "result": "Strong negative autocorrelation (mean reversion)"}
                ]
                """,
                "expected_range": {"accuracy": [82, 92], "methodology": [85, 95], "explanation": [80, 90]}
            },
            
            # Financial metrics
            {
                "id": "sharpe_ratio_1",
                "category": "financial_metrics",
                "prompt": """
                Evaluate: Portfolio return 12%, risk-free rate 2%, standard deviation 15%. Calculate Sharpe ratio.
                Answer: 0.67
                Methodology: (Return - Risk-free rate) / Standard deviation
                Steps: [
                    {"step": 1, "action": "Calculate excess return", "result": "12% - 2% = 10%"},
                    {"step": 2, "action": "Divide by standard deviation", "result": "10% / 15% = 0.67"},
                    {"step": 3, "action": "Interpretation", "result": "Risk-adjusted return of 0.67 units per unit of risk"}
                ]
                """,
                "expected_range": {"accuracy": [90, 98], "methodology": [88, 96], "explanation": [85, 95]}
            },
            {
                "id": "value_at_risk_1",
                "category": "financial_metrics",
                "prompt": """
                Evaluate: Daily returns normally distributed, mean 0.1%, std 2%. Calculate 95% VaR.
                Answer: -3.19%
                Methodology: Mean - 1.645 * Standard deviation
                Steps: [
                    {"step": 1, "action": "Identify z-score for 95% confidence", "result": "1.645"},
                    {"step": 2, "action": "Calculate VaR", "result": "0.1% - 1.645 * 2% = -3.19%"},
                    {"step": 3, "action": "Interpretation", "result": "5% chance of losing more than 3.19% in a day"}
                ]
                """,
                "expected_range": {"accuracy": [88, 96], "methodology": [85, 95], "explanation": [82, 92]}
            },
            
            # Covariance and correlation
            {
                "id": "covariance_matrix_1",
                "category": "covariance_analysis",
                "prompt": """
                Evaluate: Returns for two assets: A=[0.02, -0.01, 0.03], B=[0.01, 0.02, -0.01]. Calculate covariance.
                Answer: -0.00045
                Methodology: Cov(A,B) = E[(A-μA)(B-μB)]
                Steps: [
                    {"step": 1, "action": "Calculate means", "result": "μA = 0.0133, μB = 0.0067"},
                    {"step": 2, "action": "Calculate deviations", "result": "A-μA: [0.0067, -0.0233, 0.0167], B-μB: [0.0033, 0.0133, -0.0167]"},
                    {"step": 3, "action": "Calculate products and average", "result": "-0.00045"}
                ]
                """,
                "expected_range": {"accuracy": [83, 93], "methodology": [85, 95], "explanation": [80, 90]}
            },
            
            # Risk decomposition
            {
                "id": "risk_decomposition_1",
                "category": "risk_analysis",
                "prompt": """
                Evaluate: Portfolio has 70% systematic risk (beta=1.2) and 30% idiosyncratic risk. Market volatility is 15%. Calculate total portfolio volatility.
                Answer: 19.21%
                Methodology: σp = √(β²σm² + σε²)
                Steps: [
                    {"step": 1, "action": "Calculate systematic variance", "result": "(1.2)² * (0.15)² = 0.0324"},
                    {"step": 2, "action": "Total variance from proportion", "result": "0.0324 / 0.7 = 0.0463"},
                    {"step": 3, "action": "Calculate volatility", "result": "√0.0463 = 0.215 or 21.5%"}
                ]
                """,
                "expected_range": {"accuracy": [80, 90], "methodology": [83, 93], "explanation": [80, 90]}
            },
            
            # GARCH modeling
            {
                "id": "garch_volatility_1",
                "category": "volatility_modeling",
                "prompt": """
                Evaluate: GARCH(1,1) model: σt² = 0.00001 + 0.08εt-1² + 0.9σt-1². Yesterday: return=-2%, volatility=1.5%. Calculate today's volatility forecast.
                Answer: 1.52%
                Methodology: Apply GARCH formula
                Steps: [
                    {"step": 1, "action": "Calculate εt-1²", "result": "(-0.02)² = 0.0004"},
                    {"step": 2, "action": "Calculate σt-1²", "result": "(0.015)² = 0.000225"},
                    {"step": 3, "action": "Apply GARCH formula", "result": "0.00001 + 0.08*0.0004 + 0.9*0.000225 = 0.000245"},
                    {"step": 4, "action": "Take square root", "result": "√0.000245 = 0.0157 or 1.57%"}
                ]
                """,
                "expected_range": {"accuracy": [82, 92], "methodology": [85, 95], "explanation": [80, 90]}
            },
            
            # Error detection in financial calculations
            {
                "id": "financial_error_1",
                "category": "error_detection",
                "prompt": """
                Evaluate: A bond with 5% coupon, 10 years maturity, trading at par has duration of 12 years.
                Answer: This is incorrect
                Explanation: Duration must be less than maturity for coupon bonds. At par, duration ≈ 7.7 years.
                """,
                "expected_range": {"accuracy": [20, 40], "methodology": [60, 80], "explanation": [70, 90]}
            },
            {
                "id": "statistical_error_1",
                "category": "error_detection",
                "prompt": """
                Evaluate: Correlation between two assets is 1.5
                Answer: This is impossible
                Explanation: Correlation must be between -1 and 1 by definition
                """,
                "expected_range": {"accuracy": [15, 35], "methodology": [70, 90], "explanation": [75, 95]}
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
        
        # Calculate deltas with category weighting
        category_weight = self.category_weights.get(category, 1.0)
        
        accuracy_delta = grok_result.get('accuracy_score', 0) - lnn_result.get('accuracy_score', 0)
        methodology_delta = grok_result.get('methodology_score', 0) - lnn_result.get('methodology_score', 0)
        explanation_delta = grok_result.get('explanation_score', 0) - lnn_result.get('explanation_score', 0)
        
        # Weighted overall delta for financial categories
        overall_delta = (grok_result.get('overall_score', 0) - lnn_result.get('overall_score', 0)) * category_weight
        
        # Update category performance tracking
        lnn_score = lnn_result.get('overall_score', 0)
        if category not in self._category_performance:
            self._category_performance[category] = []
        self._category_performance[category].append(lnn_score)
        
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
        
        # Test category breakdown with performance
        test_categories = {}
        category_avg_scores = {}
        
        for metric in recent_metrics:
            category = metric.test_category
            test_categories[category] = test_categories.get(category, 0) + 1
            
            # Calculate average LNN score per category
            if category not in category_avg_scores:
                category_avg_scores[category] = []
            category_avg_scores[category].append(metric.lnn_result.get('overall_score', 0))
        
        # Calculate category averages
        for category, scores in category_avg_scores.items():
            self._category_performance[category] = statistics.mean(scores[-10:])  # Last 10 scores
        
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
            recommendations.append("URGENT: LNN quality unacceptable - consider retraining with financial focus")
            recommendations.append("Increase financial and statistical training data collection")
            recommendations.append("Focus on variance calculation and trend analysis improvements")
            
        if abs(avg_delta) > 15:
            recommendations.append("LNN underperforming in financial metrics - review quantitative training data")
            recommendations.append("Consider adding more temporal analysis examples")
            
        if trend == "degrading":
            recommendations.append("Quality declining - schedule LNN retraining with updated financial models")
            recommendations.append("Review recent financial calculations for training updates")
            recommendations.append("Check GARCH and volatility modeling accuracy")
            
        if speed < 2.0:
            recommendations.append("Speed advantage minimal - optimize LNN inference for real-time trading")
            
        if grade in ['A', 'B'] and trend in ['stable', 'improving']:
            recommendations.append("LNN performing well on financial metrics - maintain current schedule")
            recommendations.append("Consider expanding to more complex derivatives pricing")
            
        if speed > 5.0:
            recommendations.append("Excellent speed for high-frequency analysis - expand LNN usage")
            recommendations.append("Deploy for real-time risk calculations")
            
        # Category-specific recommendations
        if hasattr(self, '_category_performance'):
            worst_categories = sorted(self._category_performance.items(), 
                                    key=lambda x: x[1])[:3]
            for category, score in worst_categories:
                if score < 70:
                    recommendations.append(f"Improve {category} training - current score: {score:.1f}")
            
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
        """Get current quality status summary with financial metrics focus"""
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
        
        # Financial category performance
        financial_categories = ['variance_analysis', 'portfolio_variance', 'trend_analysis', 
                              'temporal_analysis', 'financial_metrics', 'volatility_modeling']
        financial_scores = []
        
        for metric in recent_metrics:
            if metric.test_category in financial_categories:
                financial_scores.append(metric.lnn_result.get('overall_score', 0))
        
        avg_financial_score = statistics.mean(financial_scores) if financial_scores else 0
        
        # Determine status with financial emphasis
        if abs(avg_delta) <= 10 and avg_financial_score >= 85:
            status = "excellent"
        elif abs(avg_delta) <= 15 and avg_financial_score >= 75:
            status = "good"
        elif abs(avg_delta) <= 20 and avg_financial_score >= 65:
            status = "acceptable"
        else:
            status = "poor"
        
        return {
            "status": status,
            "avg_delta": avg_delta,
            "max_delta": max_delta,
            "test_count": len(recent_metrics),
            "financial_test_count": len(financial_scores),
            "avg_financial_score": avg_financial_score,
            "category_performance": dict(self._category_performance),
            "last_test_time": recent_metrics[-1].timestamp if recent_metrics else None,
            "ready_for_failover": status in ["excellent", "good", "acceptable"]
        }
    
    async def run_immediate_quality_check(self) -> Dict[str, Any]:
        """Run immediate quality check focusing on financial calculations"""
        logger.info("Running immediate LNN quality check for financial metrics")
        
        # Run critical financial tests
        critical_categories = ['variance_analysis', 'financial_metrics', 'trend_analysis', 'error_detection']
        critical_tests = [t for t in self.benchmark_tests if t['category'] in critical_categories]
        
        results = []
        test_details = []
        
        for test in critical_tests[:5]:  # Quick test with 5 critical financial cases
            try:
                await self._run_single_benchmark(test)
                results.append("passed")
                
                # Get the latest metric for this test
                latest_metric = next((m for m in reversed(self.quality_history) 
                                    if m.test_case_id == test['id']), None)
                if latest_metric:
                    test_details.append({
                        'test_id': test['id'],
                        'category': test['category'],
                        'lnn_score': latest_metric.lnn_result.get('overall_score', 0),
                        'delta': latest_metric.overall_delta
                    })
            except Exception as e:
                logger.error(f"Critical test {test['id']} failed: {e}")
                results.append("failed")
                test_details.append({
                    'test_id': test['id'],
                    'category': test['category'],
                    'error': str(e)
                })
        
        # Get current status
        status = self.get_current_quality_status()
        
        # Calculate financial readiness
        financial_scores = [d.get('lnn_score', 0) for d in test_details if 'lnn_score' in d]
        avg_financial_readiness = statistics.mean(financial_scores) if financial_scores else 0
        
        return {
            "immediate_check_passed": all(r == "passed" for r in results),
            "critical_tests_run": len(results),
            "financial_readiness_score": avg_financial_readiness,
            "test_details": test_details,
            "current_status": status,
            "recommendation": "ready_for_financial_analysis" if avg_financial_readiness >= 75 else "needs_improvement",
            "ready_for_failover": status.get("ready_for_failover") and avg_financial_readiness >= 70,
            "timestamp": datetime.utcnow().isoformat()
        }