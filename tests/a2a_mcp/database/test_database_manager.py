"""
Test Database Manager for A2A Test Suite
Comprehensive database layer for storing and tracking test results, execution history, and analytics
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict
import uuid

from ..tools.test_executor import TestResult, TestStatus
from ..agents.test_orchestrator import TestWorkflow, TestPriority, WorkflowStatus

logger = logging.getLogger(__name__)

class TestDatabaseManager:
    """Comprehensive database manager for test execution tracking and analytics."""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path(__file__).parent.parent.parent / "test_results.db"
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Test Executions table
                CREATE TABLE IF NOT EXISTS test_executions (
                    id TEXT PRIMARY KEY,
                    test_name TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    module TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration_seconds REAL,
                    error_message TEXT,
                    output TEXT,
                    coverage_data TEXT,
                    workflow_id TEXT,
                    agent_id TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id)
                );
                
                -- Workflows table
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    test_types TEXT,
                    modules TEXT,
                    parallel_enabled BOOLEAN,
                    coverage_required BOOLEAN,
                    timeout_seconds INTEGER,
                    retry_count INTEGER,
                    dependencies TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Test Suites table
                CREATE TABLE IF NOT EXISTS test_suites (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    module TEXT NOT NULL,
                    test_count INTEGER,
                    workflow_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id)
                );
                
                -- Agent Performance table
                CREATE TABLE IF NOT EXISTS agent_performance (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    workflow_id TEXT,
                    tests_executed INTEGER,
                    success_count INTEGER,
                    failure_count INTEGER,
                    total_duration_seconds REAL,
                    avg_duration_seconds REAL,
                    load_factor REAL,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id)
                );
                
                -- AI Analysis table
                CREATE TABLE IF NOT EXISTS ai_analysis (
                    id TEXT PRIMARY KEY,
                    analysis_type TEXT NOT NULL,
                    input_data TEXT,
                    ai_response TEXT,
                    recommendations TEXT,
                    confidence_score REAL,
                    workflow_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id)
                );
                
                -- Test Trends table
                CREATE TABLE IF NOT EXISTS test_trends (
                    id TEXT PRIMARY KEY,
                    test_name TEXT NOT NULL,
                    success_rate REAL,
                    avg_duration_seconds REAL,
                    failure_rate REAL,
                    stability_score REAL,
                    trend_direction TEXT,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Performance Metrics table
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id TEXT PRIMARY KEY,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    metric_unit TEXT,
                    category TEXT,
                    workflow_id TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id)
                );
                
                -- Coverage History table
                CREATE TABLE IF NOT EXISTS coverage_history (
                    id TEXT PRIMARY KEY,
                    module TEXT NOT NULL,
                    coverage_percentage REAL,
                    lines_covered INTEGER,
                    lines_total INTEGER,
                    branches_covered INTEGER,
                    branches_total INTEGER,
                    workflow_id TEXT,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id)
                );
                
                -- Create indexes for better performance
                CREATE INDEX IF NOT EXISTS idx_test_executions_test_name ON test_executions(test_name);
                CREATE INDEX IF NOT EXISTS idx_test_executions_workflow_id ON test_executions(workflow_id);
                CREATE INDEX IF NOT EXISTS idx_test_executions_created_at ON test_executions(created_at);
                CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows(status);
                CREATE INDEX IF NOT EXISTS idx_workflows_created_at ON workflows(created_at);
                CREATE INDEX IF NOT EXISTS idx_agent_performance_agent_id ON agent_performance(agent_id);
                CREATE INDEX IF NOT EXISTS idx_test_trends_test_name ON test_trends(test_name);
                CREATE INDEX IF NOT EXISTS idx_performance_metrics_metric_name ON performance_metrics(metric_name);
                CREATE INDEX IF NOT EXISTS idx_coverage_history_module ON coverage_history(module);
            """)
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def store_workflow(self, workflow: TestWorkflow) -> str:
        """Store workflow information in database."""
        workflow_data = {
            "id": workflow.id,
            "name": workflow.name,
            "priority": workflow.priority.value,
            "status": workflow.status.value,
            "test_types": json.dumps([suite.type for suite in workflow.suites]),
            "modules": json.dumps(list(set(suite.module for suite in workflow.suites))),
            "parallel_enabled": workflow.parallel,
            "coverage_required": workflow.coverage_required,
            "timeout_seconds": workflow.timeout,
            "retry_count": workflow.retry_count,
            "dependencies": json.dumps(workflow.dependencies),
            "started_at": workflow.start_time.isoformat() if workflow.start_time else None,
            "completed_at": workflow.end_time.isoformat() if workflow.end_time else None
        }
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO workflows 
                (id, name, priority, status, test_types, modules, parallel_enabled, 
                 coverage_required, timeout_seconds, retry_count, dependencies, 
                 started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow_data["id"], workflow_data["name"], workflow_data["priority"],
                workflow_data["status"], workflow_data["test_types"], workflow_data["modules"],
                workflow_data["parallel_enabled"], workflow_data["coverage_required"],
                workflow_data["timeout_seconds"], workflow_data["retry_count"],
                workflow_data["dependencies"], workflow_data["started_at"],
                workflow_data["completed_at"]
            ))
        
        logger.info(f"Stored workflow {workflow.id} in database")
        return workflow.id
    
    def store_test_results(self, results: List[TestResult], workflow_id: str, agent_id: Optional[str] = None) -> List[str]:
        """Store test execution results in database."""
        result_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            for result in results:
                result_id = str(uuid.uuid4())
                
                conn.execute("""
                    INSERT INTO test_executions
                    (id, test_name, test_type, module, status, duration_seconds,
                     error_message, output, coverage_data, workflow_id, agent_id,
                     started_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result_id, result.name, "unknown", "unknown", result.status.value,
                    result.duration, result.error, result.output,
                    json.dumps(result.coverage) if result.coverage else None,
                    workflow_id, agent_id, datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                
                result_ids.append(result_id)
        
        logger.info(f"Stored {len(results)} test results for workflow {workflow_id}")
        return result_ids
    
    def store_agent_performance(self, agent_id: str, workflow_id: str, performance_data: Dict[str, Any]) -> str:
        """Store agent performance metrics."""
        perf_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO agent_performance
                (id, agent_id, workflow_id, tests_executed, success_count,
                 failure_count, total_duration_seconds, avg_duration_seconds, load_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                perf_id, agent_id, workflow_id,
                performance_data.get("tests_executed", 0),
                performance_data.get("success_count", 0),
                performance_data.get("failure_count", 0),
                performance_data.get("total_duration", 0),
                performance_data.get("avg_duration", 0),
                performance_data.get("load_factor", 0)
            ))
        
        return perf_id
    
    def store_ai_analysis(self, analysis_type: str, input_data: Dict[str, Any], 
                         ai_response: Dict[str, Any], workflow_id: Optional[str] = None) -> str:
        """Store AI analysis results."""
        analysis_id = str(uuid.uuid4())
        
        confidence_score = ai_response.get("confidence", 0.5)
        recommendations = ai_response.get("recommendations", [])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO ai_analysis
                (id, analysis_type, input_data, ai_response, recommendations,
                 confidence_score, workflow_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_id, analysis_type, json.dumps(input_data),
                json.dumps(ai_response), json.dumps(recommendations),
                confidence_score, workflow_id
            ))
        
        logger.info(f"Stored AI analysis {analysis_type} with ID {analysis_id}")
        return analysis_id
    
    def store_coverage_data(self, module: str, coverage_data: Dict[str, Any], workflow_id: str) -> str:
        """Store test coverage data."""
        coverage_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO coverage_history
                (id, module, coverage_percentage, lines_covered, lines_total,
                 branches_covered, branches_total, workflow_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                coverage_id, module,
                coverage_data.get("percentage", 0),
                coverage_data.get("lines_covered", 0),
                coverage_data.get("lines_total", 0),
                coverage_data.get("branches_covered", 0),
                coverage_data.get("branches_total", 0),
                workflow_id
            ))
        
        return coverage_id
    
    def get_workflow_history(self, limit: int = 100, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get workflow execution history."""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM workflows 
                WHERE created_at >= ?
                ORDER BY created_at DESC 
                LIMIT ?
            """, (cutoff_date, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_test_execution_history(self, test_name: Optional[str] = None, 
                                 limit: int = 1000, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get test execution history."""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if test_name:
                cursor = conn.execute("""
                    SELECT * FROM test_executions 
                    WHERE test_name = ? AND created_at >= ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (test_name, cutoff_date, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM test_executions 
                    WHERE created_at >= ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (cutoff_date, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_agent_performance_history(self, agent_id: Optional[str] = None, 
                                    days_back: int = 30) -> List[Dict[str, Any]]:
        """Get agent performance history."""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if agent_id:
                cursor = conn.execute("""
                    SELECT * FROM agent_performance 
                    WHERE agent_id = ? AND recorded_at >= ?
                    ORDER BY recorded_at DESC
                """, (agent_id, cutoff_date))
            else:
                cursor = conn.execute("""
                    SELECT * FROM agent_performance 
                    WHERE recorded_at >= ?
                    ORDER BY recorded_at DESC
                """, (cutoff_date,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def calculate_test_trends(self, days_back: int = 30) -> Dict[str, Any]:
        """Calculate test stability and performance trends."""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get test execution data
            cursor = conn.execute("""
                SELECT test_name, status, duration_seconds, created_at
                FROM test_executions 
                WHERE created_at >= ?
                ORDER BY test_name, created_at
            """, (cutoff_date,))
            
            executions = cursor.fetchall()
        
        # Calculate trends per test
        test_stats = {}
        for execution in executions:
            test_name = execution["test_name"]
            
            if test_name not in test_stats:
                test_stats[test_name] = {
                    "total_runs": 0,
                    "successes": 0,
                    "failures": 0,
                    "durations": [],
                    "recent_results": []
                }
            
            stats = test_stats[test_name]
            stats["total_runs"] += 1
            stats["durations"].append(execution["duration_seconds"] or 0)
            stats["recent_results"].append(execution["status"])
            
            if execution["status"] == "passed":
                stats["successes"] += 1
            elif execution["status"] == "failed":
                stats["failures"] += 1
        
        # Calculate metrics
        trends = {}
        for test_name, stats in test_stats.items():
            if stats["total_runs"] > 0:
                success_rate = stats["successes"] / stats["total_runs"]
                failure_rate = stats["failures"] / stats["total_runs"]
                avg_duration = sum(stats["durations"]) / len(stats["durations"])
                
                # Calculate stability score (success rate weighted by consistency)
                duration_variance = 0
                if len(stats["durations"]) > 1:
                    mean_duration = avg_duration
                    duration_variance = sum((d - mean_duration) ** 2 for d in stats["durations"]) / len(stats["durations"])
                
                stability_score = success_rate * (1 - min(duration_variance / (avg_duration + 1), 1))
                
                # Determine trend direction based on recent results
                recent_successes = sum(1 for result in stats["recent_results"][-10:] if result == "passed")
                recent_total = len(stats["recent_results"][-10:])
                recent_success_rate = recent_successes / recent_total if recent_total > 0 else 0
                
                if recent_success_rate > success_rate + 0.1:
                    trend_direction = "improving"
                elif recent_success_rate < success_rate - 0.1:
                    trend_direction = "declining"
                else:
                    trend_direction = "stable"
                
                trends[test_name] = {
                    "success_rate": success_rate,
                    "failure_rate": failure_rate,
                    "avg_duration_seconds": avg_duration,
                    "stability_score": stability_score,
                    "trend_direction": trend_direction,
                    "total_runs": stats["total_runs"]
                }
        
        # Store trends in database
        with sqlite3.connect(self.db_path) as conn:
            for test_name, trend_data in trends.items():
                conn.execute("""
                    INSERT OR REPLACE INTO test_trends
                    (id, test_name, success_rate, avg_duration_seconds,
                     failure_rate, stability_score, trend_direction)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()), test_name, trend_data["success_rate"],
                    trend_data["avg_duration_seconds"], trend_data["failure_rate"],
                    trend_data["stability_score"], trend_data["trend_direction"]
                ))
        
        return trends
    
    def get_coverage_trends(self, module: Optional[str] = None, days_back: int = 30) -> Dict[str, Any]:
        """Get test coverage trends."""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if module:
                cursor = conn.execute("""
                    SELECT * FROM coverage_history 
                    WHERE module = ? AND recorded_at >= ?
                    ORDER BY recorded_at DESC
                """, (module, cutoff_date))
            else:
                cursor = conn.execute("""
                    SELECT * FROM coverage_history 
                    WHERE recorded_at >= ?
                    ORDER BY recorded_at DESC
                """, (cutoff_date,))
            
            coverage_data = [dict(row) for row in cursor.fetchall()]
        
        # Calculate trends
        trends = {}
        for data in coverage_data:
            mod = data["module"]
            if mod not in trends:
                trends[mod] = {
                    "current_coverage": data["coverage_percentage"],
                    "history": [],
                    "trend": "stable"
                }
            
            trends[mod]["history"].append({
                "coverage": data["coverage_percentage"],
                "recorded_at": data["recorded_at"]
            })
        
        # Determine trend direction
        for mod, trend_data in trends.items():
            history = trend_data["history"]
            if len(history) >= 2:
                recent_avg = sum(h["coverage"] for h in history[:5]) / min(5, len(history))
                older_avg = sum(h["coverage"] for h in history[5:]) / max(1, len(history[5:]))
                
                if recent_avg > older_avg + 2:
                    trend_data["trend"] = "improving"
                elif recent_avg < older_avg - 2:
                    trend_data["trend"] = "declining"
        
        return trends
    
    def get_ai_analysis_history(self, analysis_type: Optional[str] = None, 
                              limit: int = 50) -> List[Dict[str, Any]]:
        """Get AI analysis history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if analysis_type:
                cursor = conn.execute("""
                    SELECT * FROM ai_analysis 
                    WHERE analysis_type = ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (analysis_type, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM ai_analysis 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_performance_metrics(self, metric_name: Optional[str] = None, 
                               days_back: int = 30) -> List[Dict[str, Any]]:
        """Get performance metrics."""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if metric_name:
                cursor = conn.execute("""
                    SELECT * FROM performance_metrics 
                    WHERE metric_name = ? AND recorded_at >= ?
                    ORDER BY recorded_at DESC
                """, (metric_name, cutoff_date))
            else:
                cursor = conn.execute("""
                    SELECT * FROM performance_metrics 
                    WHERE recorded_at >= ?
                    ORDER BY recorded_at DESC
                """, (cutoff_date,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def store_performance_metric(self, metric_name: str, metric_value: float, 
                                metric_unit: str, category: str, 
                                workflow_id: Optional[str] = None) -> str:
        """Store a performance metric."""
        metric_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_metrics
                (id, metric_name, metric_value, metric_unit, category, workflow_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (metric_id, metric_name, metric_value, metric_unit, category, workflow_id))
        
        return metric_id
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data beyond retention period."""
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        cleanup_stats = {}
        
        with sqlite3.connect(self.db_path) as conn:
            # Clean up old test executions
            cursor = conn.execute("DELETE FROM test_executions WHERE created_at < ?", (cutoff_date,))
            cleanup_stats["test_executions"] = cursor.rowcount
            
            # Clean up old agent performance data
            cursor = conn.execute("DELETE FROM agent_performance WHERE recorded_at < ?", (cutoff_date,))
            cleanup_stats["agent_performance"] = cursor.rowcount
            
            # Clean up old AI analysis
            cursor = conn.execute("DELETE FROM ai_analysis WHERE created_at < ?", (cutoff_date,))
            cleanup_stats["ai_analysis"] = cursor.rowcount
            
            # Clean up old performance metrics
            cursor = conn.execute("DELETE FROM performance_metrics WHERE recorded_at < ?", (cutoff_date,))
            cleanup_stats["performance_metrics"] = cursor.rowcount
            
            # Clean up old coverage history
            cursor = conn.execute("DELETE FROM coverage_history WHERE recorded_at < ?", (cutoff_date,))
            cleanup_stats["coverage_history"] = cursor.rowcount
            
            # Clean up old workflows (keep more recent ones)
            older_cutoff = (datetime.now() - timedelta(days=days_to_keep * 2)).isoformat()
            cursor = conn.execute("DELETE FROM workflows WHERE created_at < ?", (older_cutoff,))
            cleanup_stats["workflows"] = cursor.rowcount
        
        logger.info(f"Cleaned up old data: {cleanup_stats}")
        return cleanup_stats
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            stats = {}
            
            # Count records in each table
            tables = [
                "test_executions", "workflows", "test_suites", "agent_performance",
                "ai_analysis", "test_trends", "performance_metrics", "coverage_history"
            ]
            
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()["count"]
            
            # Database size
            cursor = conn.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor = conn.execute("PRAGMA page_size") 
            page_size = cursor.fetchone()[0]
            stats["database_size_bytes"] = page_count * page_size
            
            # Recent activity (last 7 days)
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor = conn.execute("SELECT COUNT(*) as count FROM test_executions WHERE created_at >= ?", (week_ago,))
            stats["recent_test_executions"] = cursor.fetchone()["count"]
            
            cursor = conn.execute("SELECT COUNT(*) as count FROM workflows WHERE created_at >= ?", (week_ago,))
            stats["recent_workflows"] = cursor.fetchone()["count"]
        
        return stats

class TestAnalyticsService:
    """Service for advanced test analytics and insights."""
    
    def __init__(self, db_manager: TestDatabaseManager):
        self.db_manager = db_manager
    
    def generate_test_health_report(self, days_back: int = 30) -> Dict[str, Any]:
        """Generate comprehensive test health report."""
        trends = self.db_manager.calculate_test_trends(days_back)
        coverage_trends = self.db_manager.get_coverage_trends(days_back=days_back)
        workflow_history = self.db_manager.get_workflow_history(days_back=days_back)
        
        # Calculate overall health metrics
        total_tests = len(trends)
        stable_tests = len([t for t in trends.values() if t["stability_score"] > 0.8])
        flaky_tests = len([t for t in trends.values() if t["stability_score"] < 0.5])
        
        # Workflow success rate
        successful_workflows = len([w for w in workflow_history if w["status"] == "completed"])
        total_workflows = len(workflow_history)
        workflow_success_rate = (successful_workflows / total_workflows * 100) if total_workflows > 0 else 0
        
        # Coverage analysis
        avg_coverage = 0
        if coverage_trends:
            all_coverage = [t["current_coverage"] for t in coverage_trends.values()]
            avg_coverage = sum(all_coverage) / len(all_coverage)
        
        report = {
            "report_period_days": days_back,
            "generated_at": datetime.now().isoformat(),
            "test_health": {
                "total_tests": total_tests,
                "stable_tests": stable_tests,
                "flaky_tests": flaky_tests,
                "stability_percentage": (stable_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "workflow_health": {
                "total_workflows": total_workflows,
                "successful_workflows": successful_workflows,
                "success_rate_percentage": workflow_success_rate
            },
            "coverage_health": {
                "average_coverage_percentage": avg_coverage,
                "modules_tracked": len(coverage_trends),
                "coverage_trends": coverage_trends
            },
            "flaky_tests": [
                {"test_name": name, "stability_score": data["stability_score"]}
                for name, data in trends.items()
                if data["stability_score"] < 0.5
            ],
            "top_performers": [
                {"test_name": name, "stability_score": data["stability_score"]}
                for name, data in sorted(trends.items(), key=lambda x: x[1]["stability_score"], reverse=True)[:10]
            ]
        }
        
        return report
    
    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify test execution optimization opportunities."""
        trends = self.db_manager.calculate_test_trends()
        agent_performance = self.db_manager.get_agent_performance_history()
        
        opportunities = []
        
        # Find slow tests
        slow_tests = [
            {"test_name": name, "avg_duration": data["avg_duration_seconds"]}
            for name, data in trends.items()
            if data["avg_duration_seconds"] > 30  # Tests taking more than 30 seconds
        ]
        
        if slow_tests:
            opportunities.append({
                "type": "performance",
                "title": "Slow Test Optimization",
                "description": f"Found {len(slow_tests)} tests with long execution times",
                "impact": "high",
                "tests": sorted(slow_tests, key=lambda x: x["avg_duration"], reverse=True)[:5]
            })
        
        # Find flaky tests
        flaky_tests = [
            {"test_name": name, "stability_score": data["stability_score"]}
            for name, data in trends.items()
            if data["stability_score"] < 0.7 and data["total_runs"] > 5
        ]
        
        if flaky_tests:
            opportunities.append({
                "type": "reliability",
                "title": "Flaky Test Stabilization",
                "description": f"Found {len(flaky_tests)} unreliable tests",
                "impact": "high",
                "tests": sorted(flaky_tests, key=lambda x: x["stability_score"])[:5]
            })
        
        # Agent load balancing opportunities
        if agent_performance:
            agent_loads = {}
            for perf in agent_performance:
                agent_id = perf["agent_id"]
                if agent_id not in agent_loads:
                    agent_loads[agent_id] = []
                agent_loads[agent_id].append(perf["load_factor"])
            
            # Check for load imbalances
            avg_loads = {aid: sum(loads)/len(loads) for aid, loads in agent_loads.items()}
            max_load = max(avg_loads.values()) if avg_loads else 0
            min_load = min(avg_loads.values()) if avg_loads else 0
            
            if max_load - min_load > 0.3:  # 30% load difference
                opportunities.append({
                    "type": "load_balancing",
                    "title": "Agent Load Rebalancing",
                    "description": "Detected uneven load distribution across agents",
                    "impact": "medium",
                    "agents": avg_loads
                })
        
        return opportunities