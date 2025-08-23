"""
MCP Performance Tools
Performance measurement, benchmarking, and SLA compliance tools
"""

import time
import psutil
import statistics
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
from ..sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from ..sdk.mcpSkillCoordination import skill_provides, skill_depends_on

logger = logging.getLogger(__name__)


class MCPPerformanceTools:
    """MCP-enabled performance measurement and analysis tools"""
    
    def __init__(self):
        self.metrics_history = defaultdict(deque)
        self.max_history_size = 1000
        
        # Standard SLA metrics
        self.sla_metrics = [
            "response_time",
            "throughput",
            "error_rate",
            "availability",
            "cpu_usage",
            "memory_usage",
            "disk_io",
            "network_io"
        ]
        
        # Performance benchmarks (baseline values)
        self.benchmarks = {
            "response_time_p95": 1000,  # ms
            "throughput_min": 100,      # operations/sec
            "error_rate_max": 0.01,     # 1%
            "availability_min": 0.99,   # 99%
            "cpu_usage_max": 0.80,      # 80%
            "memory_usage_max": 0.85    # 85%
        }
    
    @mcp_tool(
        name="measure_performance_metrics",
        description="Comprehensive performance measurement for operations",
        input_schema={
            "type": "object",
            "properties": {
                "operation_id": {
                    "type": "string",
                    "description": "Unique identifier for the operation"
                },
                "start_time": {
                    "type": "number",
                    "description": "Operation start timestamp"
                },
                "end_time": {
                    "type": "number",
                    "description": "Operation end timestamp"
                },
                "resource_usage": {
                    "type": "object",
                    "description": "Resource usage during operation",
                    "properties": {
                        "cpu_percent": {"type": "number"},
                        "memory_mb": {"type": "number"},
                        "disk_io_mb": {"type": "number"},
                        "network_io_mb": {"type": "number"}
                    }
                },
                "custom_metrics": {
                    "type": "object",
                    "description": "Custom performance metrics",
                    "additionalProperties": {"type": "number"}
                },
                "operation_count": {
                    "type": "integer",
                    "default": 1,
                    "description": "Number of operations performed"
                },
                "errors": {
                    "type": "integer",
                    "default": 0,
                    "description": "Number of errors that occurred"
                }
            },
            "required": ["operation_id", "start_time", "end_time"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "duration_ms": {"type": "number"},
                "throughput": {"type": "number"},
                "error_rate": {"type": "number"},
                "resource_efficiency": {"type": "object"},
                "performance_score": {"type": "number"},
                "percentiles": {"type": "object"}
            }
        }
    )
    @skill_provides("performance_measurement", "metrics_collection")
    async def measure_performance_metrics(self,
                                    operation_id: str,
                                    start_time: float,
                                    end_time: float,
                                    resource_usage: Optional[Dict[str, float]] = None,
                                    custom_metrics: Optional[Dict[str, float]] = None,
                                    operation_count: int = 1,
                                    errors: int = 0) -> Dict[str, Any]:
        """Measure comprehensive performance metrics for an operation"""
        
        # Calculate basic timing metrics
        duration_ms = (end_time - start_time) * 1000
        throughput = operation_count / ((end_time - start_time)) if (end_time - start_time) > 0 else 0
        error_rate = errors / operation_count if operation_count > 0 else 0
        
        # Store metrics for historical analysis
        self.metrics_history[operation_id].append({
            "timestamp": datetime.now(),
            "duration_ms": duration_ms,
            "throughput": throughput,
            "error_rate": error_rate,
            "operation_count": operation_count,
            "errors": errors
        })
        
        # Maintain history size limit
        if len(self.metrics_history[operation_id]) > self.max_history_size:
            self.metrics_history[operation_id].popleft()
        
        # Calculate resource efficiency
        resource_efficiency = {}
        if resource_usage:
            # Efficiency = Operations per unit of resource
            if resource_usage.get("cpu_percent", 0) > 0:
                resource_efficiency["cpu_efficiency"] = operation_count / resource_usage["cpu_percent"]
            if resource_usage.get("memory_mb", 0) > 0:
                resource_efficiency["memory_efficiency"] = operation_count / resource_usage["memory_mb"]
        
        # Calculate percentiles from historical data
        percentiles = {}
        if len(self.metrics_history[operation_id]) >= 5:
            durations = [m["duration_ms"] for m in self.metrics_history[operation_id]]
            percentiles = {
                "p50": statistics.median(durations),
                "p95": self._calculate_percentile(durations, 95),
                "p99": self._calculate_percentile(durations, 99)
            }
        
        # Calculate overall performance score
        performance_score = await self._calculate_performance_score(
            duration_ms, throughput, error_rate, resource_usage
        )
        
        result = {
            "operation_id": operation_id,
            "duration_ms": duration_ms,
            "throughput": throughput,
            "error_rate": error_rate,
            "operation_count": operation_count,
            "errors": errors,
            "resource_efficiency": resource_efficiency,
            "performance_score": performance_score,
            "percentiles": percentiles,
            "measurement_timestamp": datetime.now().isoformat()
        }
        
        # Add custom metrics if provided
        if custom_metrics:
            result["custom_metrics"] = custom_metrics
        
        # Add resource usage if provided
        if resource_usage:
            result["resource_usage"] = resource_usage
        
        return result
    
    @mcp_tool(
        name="calculate_sla_compliance",
        description="Calculate SLA compliance based on performance metrics",
        input_schema={
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "object",
                    "description": "Performance metrics to check against SLA",
                    "additionalProperties": {"type": "number"}
                },
                "sla_targets": {
                    "type": "object",
                    "description": "SLA target values for each metric",
                    "additionalProperties": {"type": "number"}
                },
                "time_window": {
                    "type": "string",
                    "default": "1h",
                    "description": "Time window for SLA calculation (e.g., '1h', '24h', '7d')"
                },
                "operation_id": {
                    "type": "string",
                    "description": "Operation ID for historical analysis"
                }
            },
            "required": ["metrics", "sla_targets"]
        }
    )
    @skill_provides("sla_monitoring", "compliance_checking")
    async def calculate_sla_compliance(self,
                                 metrics: Dict[str, float],
                                 sla_targets: Dict[str, float],
                                 time_window: str = "1h",
                                 operation_id: Optional[str] = None) -> Dict[str, Any]:
        """Calculate SLA compliance scores and identify violations"""
        
        compliance_scores = {}
        violations = []
        compliant_metrics = []
        
        # Calculate compliance for each metric
        for metric_name, current_value in metrics.items():
            if metric_name in sla_targets:
                target_value = sla_targets[metric_name]
                
                # Determine compliance based on metric type
                if metric_name in ["response_time", "error_rate", "cpu_usage", "memory_usage"]:
                    # Lower is better metrics
                    compliance = min(1.0, target_value / current_value) if current_value > 0 else 1.0
                    is_compliant = current_value <= target_value
                else:
                    # Higher is better metrics (throughput, availability)
                    compliance = min(1.0, current_value / target_value) if target_value > 0 else 1.0
                    is_compliant = current_value >= target_value
                
                compliance_scores[metric_name] = compliance
                
                if is_compliant:
                    compliant_metrics.append(metric_name)
                else:
                    violations.append({
                        "metric": metric_name,
                        "current_value": current_value,
                        "target_value": target_value,
                        "compliance_score": compliance,
                        "deviation": abs(current_value - target_value)
                    })
        
        # Calculate overall SLA compliance
        overall_compliance = statistics.mean(compliance_scores.values()) if compliance_scores else 0.0
        
        # Trend analysis if historical data is available
        trend_analysis = {}
        if operation_id and operation_id in self.metrics_history:
            trend_analysis = await self._analyze_compliance_trend(
                operation_id, time_window, sla_targets
            )
        
        return {
            "overall_compliance": overall_compliance,
            "compliance_scores": compliance_scores,
            "violations": violations,
            "compliant_metrics": compliant_metrics,
            "time_window": time_window,
            "trend_analysis": trend_analysis,
            "sla_status": "COMPLIANT" if overall_compliance >= 0.95 else "NON_COMPLIANT",
            "assessment_timestamp": datetime.now().isoformat()
        }
    
    @mcp_tool(
        name="benchmark_performance",
        description="Benchmark current performance against historical data or targets",
        input_schema={
            "type": "object",
            "properties": {
                "current_metrics": {
                    "type": "object",
                    "description": "Current performance metrics",
                    "additionalProperties": {"type": "number"}
                },
                "historical_data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Historical performance data for comparison"
                },
                "benchmark_type": {
                    "type": "string",
                    "enum": ["statistical", "baseline", "percentile", "regression"],
                    "default": "statistical",
                    "description": "Type of benchmarking to perform"
                },
                "confidence_level": {
                    "type": "number",
                    "default": 0.95,
                    "description": "Confidence level for statistical analysis"
                }
            },
            "required": ["current_metrics"]
        }
    )
    @skill_provides("performance_benchmarking", "trend_analysis")
    async def benchmark_performance(self,
                              current_metrics: Dict[str, float],
                              historical_data: Optional[List[Dict[str, Any]]] = None,
                              benchmark_type: str = "statistical",
                              confidence_level: float = 0.95) -> Dict[str, Any]:
        """Benchmark current performance against historical data or baselines"""
        
        benchmark_results = {}
        performance_rating = "unknown"
        comparison_scores = {}
        recommendations = []
        
        if benchmark_type == "baseline":
            # Compare against predefined baselines
            for metric, current_value in current_metrics.items():
                baseline_key = f"{metric}_baseline"
                if baseline_key in self.benchmarks or metric in self.benchmarks:
                    baseline = self.benchmarks.get(baseline_key, self.benchmarks.get(metric))
                    if baseline:
                        if metric in ["response_time", "error_rate", "cpu_usage", "memory_usage"]:
                            score = min(1.0, baseline / current_value) if current_value > 0 else 1.0
                        else:
                            score = min(1.0, current_value / baseline) if baseline > 0 else 1.0
                        
                        comparison_scores[metric] = score
                        benchmark_results[metric] = {
                            "current": current_value,
                            "baseline": baseline,
                            "score": score,
                            "status": "good" if score >= 0.8 else "warning" if score >= 0.6 else "poor"
                        }
        
        elif benchmark_type == "statistical" and historical_data:
            # Statistical comparison with historical data
            for metric, current_value in current_metrics.items():
                historical_values = [
                    d.get(metric) for d in historical_data 
                    if d.get(metric) is not None
                ]
                
                if historical_values:
                    mean_val = statistics.mean(historical_values)
                    stdev_val = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
                    
                    # Calculate z-score
                    z_score = (current_value - mean_val) / stdev_val if stdev_val > 0 else 0
                    
                    # Determine performance relative to historical performance
                    if metric in ["response_time", "error_rate", "cpu_usage", "memory_usage"]:
                        # Lower is better - negative z-score is good
                        score = max(0.0, 1.0 - abs(z_score) / 3.0)  # 3-sigma normalization
                        performance_vs_history = "better" if z_score < -1 else "worse" if z_score > 1 else "similar"
                    else:
                        # Higher is better - positive z-score is good
                        score = max(0.0, 1.0 - abs(z_score) / 3.0)
                        performance_vs_history = "better" if z_score > 1 else "worse" if z_score < -1 else "similar"
                    
                    comparison_scores[metric] = score
                    benchmark_results[metric] = {
                        "current": current_value,
                        "historical_mean": mean_val,
                        "historical_stdev": stdev_val,
                        "z_score": z_score,
                        "score": score,
                        "performance_vs_history": performance_vs_history
                    }
        
        # Calculate overall performance rating
        if comparison_scores:
            overall_score = statistics.mean(comparison_scores.values())
            if overall_score >= 0.8:
                performance_rating = "excellent"
            elif overall_score >= 0.6:
                performance_rating = "good"
            elif overall_score >= 0.4:
                performance_rating = "fair"
            else:
                performance_rating = "poor"
            
            # Generate recommendations
            recommendations = await self._generate_performance_recommendations(
                benchmark_results, performance_rating
            )
        
        return {
            "performance_rating": performance_rating,
            "overall_score": statistics.mean(comparison_scores.values()) if comparison_scores else 0.0,
            "comparison_scores": comparison_scores,
            "benchmark_results": benchmark_results,
            "recommendations": recommendations,
            "benchmark_type": benchmark_type,
            "confidence_level": confidence_level,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    @mcp_tool(
        name="get_system_metrics",
        description="Get current system resource metrics",
        input_schema={
            "type": "object",
            "properties": {
                "include_processes": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include top processes in the metrics"
                },
                "process_count": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of top processes to include"
                }
            }
        }
    )
    @skill_provides("system_monitoring", "resource_tracking")
    async def get_system_metrics(self,
                           include_processes: bool = False,
                           process_count: int = 5) -> Dict[str, Any]:
        """Get current system performance metrics"""
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            metrics = {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "frequency_mhz": cpu_freq.current if cpu_freq else None
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory.percent
                },
                "swap": {
                    "total_gb": swap.total / (1024**3),
                    "used_gb": swap.used / (1024**3),
                    "percent": swap.percent
                },
                "disk": {
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": disk_usage.free / (1024**3),
                    "percent": (disk_usage.used / disk_usage.total) * 100
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Add disk I/O if available
            if disk_io:
                metrics["disk_io"] = {
                    "read_bytes": disk_io.read_bytes,
                    "write_bytes": disk_io.write_bytes,
                    "read_count": disk_io.read_count,
                    "write_count": disk_io.write_count
                }
            
            # Add process information if requested
            if include_processes:
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        proc_info = proc.info
                        if proc_info['cpu_percent'] is not None:
                            processes.append(proc_info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Sort by CPU usage and take top N
                top_processes = sorted(
                    processes, 
                    key=lambda x: x['cpu_percent'] or 0, 
                    reverse=True
                )[:process_count]
                
                metrics["top_processes"] = top_processes
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                "error": f"Failed to get system metrics: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    @mcp_resource(
        uri="performance://benchmarks/default",
        name="Default Performance Benchmarks",
        description="Default performance benchmarks and SLA targets",
        mime_type="application/json"
    )
    async def get_default_benchmarks(self) -> Dict[str, Any]:
        """Get default performance benchmarks and SLA targets"""
        return {
            "benchmarks": self.benchmarks,
            "sla_metrics": self.sla_metrics,
            "performance_levels": {
                "excellent": {"threshold": 0.8, "description": "Exceeds performance expectations"},
                "good": {"threshold": 0.6, "description": "Meets performance expectations"},
                "fair": {"threshold": 0.4, "description": "Below performance expectations"},
                "poor": {"threshold": 0.0, "description": "Significantly below expectations"}
            },
            "recommended_monitoring_intervals": {
                "real_time": "1s",
                "operational": "1m", 
                "tactical": "5m",
                "strategic": "1h"
            }
        }
    
    @mcp_prompt(
        name="performance_analysis_report",
        description="Generate comprehensive performance analysis report",
        arguments=[
            {
                "name": "performance_data",
                "description": "Performance metrics and analysis results",
                "required": True
            },
            {
                "name": "report_focus",
                "description": "Focus area (trends, issues, optimization, compliance)",
                "required": False
            }
        ]
    )
    async def performance_analysis_report(self,
                                    performance_data: Dict[str, Any],
                                    report_focus: str = "comprehensive") -> str:
        """Generate detailed performance analysis report"""
        
        report = f"# Performance Analysis Report\n\n"
        report += f"**Analysis Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        
        # Overall performance summary
        if "performance_rating" in performance_data:
            rating = performance_data["performance_rating"]
            score = performance_data.get("overall_score", 0)
            report += f"**Overall Performance**: {rating.title()} ({score:.1%})\n\n"
        
        # SLA compliance section
        if "overall_compliance" in performance_data:
            compliance = performance_data["overall_compliance"]
            status = performance_data.get("sla_status", "UNKNOWN")
            report += f"## SLA Compliance\n\n"
            report += f"**Compliance Score**: {compliance:.1%}\n"
            report += f"**Status**: {status}\n\n"
            
            violations = performance_data.get("violations", [])
            if violations:
                report += f"### Violations ({len(violations)})\n\n"
                for violation in violations:
                    report += f"- **{violation['metric']}**: {violation['current_value']:.2f} "
                    report += f"(Target: {violation['target_value']:.2f})\n"
        
        # Performance metrics breakdown
        if "benchmark_results" in performance_data:
            report += f"## Performance Metrics\n\n"
            for metric, result in performance_data["benchmark_results"].items():
                current = result.get("current", 0)
                status = result.get("status", "unknown")
                status_icon = "✅" if status == "good" else "⚠️" if status == "warning" else "❌"
                report += f"- **{metric.replace('_', ' ').title()}**: {current:.2f} {status_icon}\n"
        
        # Recommendations
        if "recommendations" in performance_data:
            recommendations = performance_data["recommendations"]
            if recommendations:
                report += f"\n## Recommendations\n\n"
                for rec in recommendations:
                    report += f"- {rec}\n"
        
        # Focus-specific sections
        if report_focus == "trends" and "trend_analysis" in performance_data:
            report += f"\n## Trend Analysis\n\n"
            trend_data = performance_data["trend_analysis"]
            report += f"Trend analysis shows: {trend_data}\n"
        
        elif report_focus == "optimization":
            report += f"\n## Optimization Opportunities\n\n"
            report += "- Consider resource scaling based on current utilization\n"
            report += "- Review code efficiency for high-latency operations\n"
            report += "- Implement caching for frequently accessed data\n"
        
        return report
    
    # Internal helper methods
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate the specified percentile of the data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        n = len(sorted_data)
        index = (percentile / 100) * (n - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            if upper_index >= n:
                return sorted_data[-1]
            
            # Linear interpolation
            weight = index - lower_index
            return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    async def _calculate_performance_score(self,
                                     duration_ms: float,
                                     throughput: float,
                                     error_rate: float,
                                     resource_usage: Optional[Dict[str, float]]) -> float:
        """Calculate overall performance score"""
        score = 1.0
        
        # Duration penalty (assuming 1000ms baseline)
        if duration_ms > 1000:
            score *= max(0.1, 1000 / duration_ms)
        
        # Error rate penalty
        score *= max(0.1, 1.0 - (error_rate * 10))  # 10% error rate = 0 score
        
        # Resource usage bonus/penalty
        if resource_usage:
            cpu_usage = resource_usage.get("cpu_percent", 50) / 100
            if cpu_usage > 0.8:
                score *= 0.8  # High CPU penalty
            elif cpu_usage < 0.2:
                score *= 1.1  # Low CPU bonus
        
        return min(1.0, max(0.0, score))
    
    async def _analyze_compliance_trend(self,
                                  operation_id: str,
                                  time_window: str,
                                  sla_targets: Dict[str, float]) -> Dict[str, Any]:
        """Analyze SLA compliance trends over time"""
        # Placeholder for trend analysis implementation
        return {
            "trend": "stable",
            "violations_last_hour": 0,
            "improvement_percentage": 5.2
        }
    
    async def _generate_performance_recommendations(self,
                                              benchmark_results: Dict[str, Any],
                                              performance_rating: str) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if performance_rating == "poor":
            recommendations.append("Immediate performance optimization required")
            recommendations.append("Review resource allocation and scaling options")
        elif performance_rating == "fair":
            recommendations.append("Consider performance tuning and optimization")
        
        # Specific metric recommendations
        for metric, result in benchmark_results.items():
            score = result.get("score", 1.0)
            if score < 0.6:
                if "response_time" in metric:
                    recommendations.append("Optimize response time through caching or code optimization")
                elif "throughput" in metric:
                    recommendations.append("Scale resources or optimize concurrency to improve throughput")
                elif "cpu" in metric:
                    recommendations.append("Review CPU-intensive operations and consider optimization")
                elif "memory" in metric:
                    recommendations.append("Analyze memory usage patterns and implement memory optimization")
        
        return recommendations


# Singleton instance
mcp_performance_tools = MCPPerformanceTools()