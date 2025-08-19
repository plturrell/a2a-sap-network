"""
GROKClient Integration for A2A Test Suite
AI-powered test analysis, insights, and optimization using GROK
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import GROKClient (assuming it's available in the project)
try:
    from grok_client import GROKClient, GROKResponse
except ImportError:
    # Fallback mock for development
    class GROKClient:
        def __init__(self, *args, **kwargs):
            pass
        
        async def chat(self, messages: List[Dict], **kwargs) -> Dict:
            return {"content": "GROK client not available - mock response"}
    
    class GROKResponse:
        def __init__(self, content: str):
            self.content = content

from ..tools.test_executor import TestResult, TestStatus
from ..agents.test_orchestrator import TestWorkflow

logger = logging.getLogger(__name__)

class AITestAnalyzer:
    """AI-powered test analysis using GROKClient."""
    
    def __init__(self, grok_client: Optional[GROKClient] = None):
        self.grok_client = grok_client or GROKClient()
        self.analysis_cache = {}
        
    async def analyze_test_failures(
        self, 
        failed_tests: List[TestResult], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze test failures using AI to provide insights and suggestions."""
        
        if not failed_tests:
            return {"analysis": "No failed tests to analyze", "suggestions": []}
        
        # Prepare failure data for AI analysis
        failure_data = []
        for test in failed_tests:
            failure_data.append({
                "test_name": test.name,
                "error": test.error,
                "output": test.output[:1000] if test.output else "",  # Limit output length
                "duration": test.duration
            })
        
        # Create AI prompt for failure analysis
        prompt = self._create_failure_analysis_prompt(failure_data, context)
        
        try:
            # Get AI analysis
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert software testing analyst. Analyze test failures and provide actionable insights and recommendations."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            response = await self.grok_client.chat(
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse AI response
            analysis = self._parse_failure_analysis(response.get("content", ""))
            
            # Cache analysis for future reference
            cache_key = f"failure_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.analysis_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze test failures with AI: {e}")
            return {
                "analysis": f"AI analysis failed: {str(e)}",
                "suggestions": ["Review test failures manually", "Check test environment setup"],
                "error": str(e)
            }
    
    async def optimize_test_execution(
        self, 
        execution_history: List[Dict[str, Any]], 
        current_workflow: TestWorkflow
    ) -> Dict[str, Any]:
        """Use AI to optimize test execution strategy based on historical data."""
        
        if len(execution_history) < 3:
            return {
                "optimization": "Insufficient data for AI optimization",
                "recommendations": ["Collect more execution data"]
            }
        
        # Prepare execution data for AI analysis
        execution_summary = self._summarize_execution_history(execution_history)
        
        # Create optimization prompt
        prompt = self._create_optimization_prompt(execution_summary, current_workflow)
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in test automation and performance optimization. Analyze test execution patterns and provide optimization strategies."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await self.grok_client.chat(
                messages=messages,
                temperature=0.2,
                max_tokens=1500
            )
            
            optimization = self._parse_optimization_response(response.get("content", ""))
            
            return optimization
            
        except Exception as e:
            logger.error(f"Failed to get AI optimization: {e}")
            return {
                "optimization": f"AI optimization failed: {str(e)}",
                "recommendations": ["Review execution patterns manually"],
                "error": str(e)
            }
    
    async def predict_test_stability(
        self, 
        test_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Predict test stability and identify flaky tests using AI."""
        
        # Analyze test patterns
        stability_data = self._analyze_test_patterns(test_history)
        
        # Create prediction prompt
        prompt = self._create_stability_prediction_prompt(stability_data)
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a test reliability expert. Analyze test execution patterns to predict stability and identify flaky tests."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await self.grok_client.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            prediction = self._parse_stability_prediction(response.get("content", ""))
            
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to predict test stability: {e}")
            return {
                "prediction": f"AI prediction failed: {str(e)}",
                "flaky_tests": [],
                "error": str(e)
            }
    
    async def generate_test_insights(
        self, 
        workflow_results: List[TestWorkflow],
        timeframe_days: int = 30
    ) -> Dict[str, Any]:
        """Generate comprehensive test insights and trends using AI."""
        
        # Filter recent workflows
        cutoff_date = datetime.now() - timedelta(days=timeframe_days)
        recent_workflows = [
            wf for wf in workflow_results 
            if wf.start_time and wf.start_time >= cutoff_date
        ]
        
        if not recent_workflows:
            return {"insights": "No recent workflow data available"}
        
        # Aggregate workflow data
        insights_data = self._aggregate_workflow_data(recent_workflows)
        
        # Create insights prompt
        prompt = self._create_insights_prompt(insights_data, timeframe_days)
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a senior QA engineer and data analyst. Analyze test execution trends and provide strategic insights for test improvement."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await self.grok_client.chat(
                messages=messages,
                temperature=0.4,
                max_tokens=2500
            )
            
            insights = self._parse_insights_response(response.get("content", ""))
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate test insights: {e}")
            return {
                "insights": f"AI insights generation failed: {str(e)}",
                "trends": [],
                "error": str(e)
            }
    
    def _create_failure_analysis_prompt(
        self, 
        failure_data: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Create AI prompt for test failure analysis."""
        
        prompt = f"""
Analyze the following test failures and provide insights:

FAILED TESTS ({len(failure_data)} total):
"""
        
        for i, failure in enumerate(failure_data[:10]):  # Limit to 10 failures
            prompt += f"""
Test {i+1}: {failure['test_name']}
Duration: {failure['duration']:.2f}s
Error: {failure['error'] or 'No error message'}
Output: {failure['output'][:200] if failure['output'] else 'No output'}...

"""
        
        if context:
            prompt += f"\nCONTEXT:\n{json.dumps(context, indent=2)}\n"
        
        prompt += """
Please provide:
1. Root cause analysis of the failures
2. Common patterns or themes
3. Specific actionable recommendations
4. Priority level for each recommendation (High/Medium/Low)
5. Estimated effort to fix (Quick/Medium/Complex)

Format your response as JSON with keys: analysis, patterns, recommendations, priorities, efforts.
"""
        
        return prompt
    
    def _create_optimization_prompt(
        self, 
        execution_summary: Dict[str, Any], 
        current_workflow: TestWorkflow
    ) -> str:
        """Create AI prompt for execution optimization."""
        
        prompt = f"""
Analyze test execution patterns and optimize strategy:

EXECUTION HISTORY SUMMARY:
{json.dumps(execution_summary, indent=2)}

CURRENT WORKFLOW:
- Name: {current_workflow.name}
- Type: {current_workflow.suites[0].type if current_workflow.suites else 'Unknown'}
- Parallel: {current_workflow.parallel}
- Timeout: {current_workflow.timeout}
- Coverage: {current_workflow.coverage_required}
- Suite Count: {len(current_workflow.suites)}

Please analyze and recommend:
1. Optimal parallel execution strategy
2. Timeout adjustments based on historical data
3. Test ordering for maximum efficiency
4. Resource allocation recommendations
5. Performance bottleneck identification

Format response as JSON with keys: strategy, timeouts, ordering, resources, bottlenecks.
"""
        
        return prompt
    
    def _create_stability_prediction_prompt(self, stability_data: Dict[str, Any]) -> str:
        """Create AI prompt for test stability prediction."""
        
        prompt = f"""
Analyze test stability patterns and predict flaky tests:

TEST STABILITY DATA:
{json.dumps(stability_data, indent=2)}

Please analyze and provide:
1. Overall stability assessment
2. List of potentially flaky tests with confidence scores
3. Stability trends (improving/declining)
4. Recommendations to improve test reliability

Format response as JSON with keys: assessment, flaky_tests, trends, recommendations.
"""
        
        return prompt
    
    def _create_insights_prompt(self, insights_data: Dict[str, Any], timeframe_days: int) -> str:
        """Create AI prompt for test insights generation."""
        
        prompt = f"""
Generate strategic test insights from {timeframe_days} days of execution data:

AGGREGATED DATA:
{json.dumps(insights_data, indent=2)}

Please provide:
1. Key trends and patterns
2. Performance insights
3. Quality metrics analysis
4. Strategic recommendations for test improvement
5. Risk assessment and mitigation strategies

Format response as JSON with keys: trends, performance, quality, recommendations, risks.
"""
        
        return prompt
    
    def _summarize_execution_history(self, execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize execution history for AI analysis."""
        
        if not execution_history:
            return {}
        
        total_executions = len(execution_history)
        successful = len([e for e in execution_history if e.get("status") == "completed"])
        failed = len([e for e in execution_history if e.get("status") == "failed"])
        
        durations = [e.get("duration", 0) for e in execution_history if e.get("duration")]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Parallel vs sequential analysis
        parallel_runs = [e for e in execution_history if e.get("parallel_enabled")]
        sequential_runs = [e for e in execution_history if not e.get("parallel_enabled")]
        
        summary = {
            "total_executions": total_executions,
            "success_rate": (successful / total_executions * 100) if total_executions > 0 else 0,
            "failure_rate": (failed / total_executions * 100) if total_executions > 0 else 0,
            "average_duration_seconds": avg_duration,
            "parallel_executions": len(parallel_runs),
            "sequential_executions": len(sequential_runs)
        }
        
        if parallel_runs:
            parallel_avg = sum(e.get("duration", 0) for e in parallel_runs) / len(parallel_runs)
            summary["parallel_avg_duration"] = parallel_avg
        
        if sequential_runs:
            sequential_avg = sum(e.get("duration", 0) for e in sequential_runs) / len(sequential_runs)
            summary["sequential_avg_duration"] = sequential_avg
        
        return summary
    
    def _analyze_test_patterns(self, test_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test execution patterns for stability prediction."""
        
        test_stats = {}
        
        for execution in test_history:
            test_name = execution.get("test_name", "unknown")
            status = execution.get("status", "unknown")
            duration = execution.get("duration", 0)
            
            if test_name not in test_stats:
                test_stats[test_name] = {
                    "total_runs": 0,
                    "successes": 0,
                    "failures": 0,
                    "durations": []
                }
            
            stats = test_stats[test_name]
            stats["total_runs"] += 1
            stats["durations"].append(duration)
            
            if status == "passed":
                stats["successes"] += 1
            elif status == "failed":
                stats["failures"] += 1
        
        # Calculate stability metrics
        stability_metrics = {}
        for test_name, stats in test_stats.items():
            if stats["total_runs"] > 0:
                success_rate = stats["successes"] / stats["total_runs"]
                avg_duration = sum(stats["durations"]) / len(stats["durations"])
                duration_variance = sum((d - avg_duration) ** 2 for d in stats["durations"]) / len(stats["durations"])
                
                stability_metrics[test_name] = {
                    "success_rate": success_rate,
                    "total_runs": stats["total_runs"],
                    "avg_duration": avg_duration,
                    "duration_variance": duration_variance,
                    "stability_score": success_rate * (1 - duration_variance / (avg_duration + 1))
                }
        
        return {
            "test_count": len(stability_metrics),
            "metrics": stability_metrics
        }
    
    def _aggregate_workflow_data(self, workflows: List[TestWorkflow]) -> Dict[str, Any]:
        """Aggregate workflow data for insights generation."""
        
        total_workflows = len(workflows)
        successful_workflows = len([w for w in workflows if w.status.value == "completed"])
        failed_workflows = len([w for w in workflows if w.status.value == "failed"])
        
        # Calculate duration statistics
        durations = []
        for workflow in workflows:
            if workflow.start_time and workflow.end_time:
                duration = (workflow.end_time - workflow.start_time).total_seconds()
                durations.append(duration)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Test results aggregation
        all_results = []
        for workflow in workflows:
            all_results.extend(workflow.results)
        
        total_tests = len(all_results)
        passed_tests = len([r for r in all_results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in all_results if r.status == TestStatus.FAILED])
        
        return {
            "workflow_summary": {
                "total_workflows": total_workflows,
                "successful_workflows": successful_workflows,
                "failed_workflows": failed_workflows,
                "success_rate": (successful_workflows / total_workflows * 100) if total_workflows > 0 else 0,
                "average_duration_seconds": avg_duration
            },
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            }
        }
    
    def _parse_failure_analysis(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI failure analysis response."""
        try:
            # Try to parse as JSON first
            if ai_response.strip().startswith('{'):
                return json.loads(ai_response)
            
            # Fallback to structured parsing
            return {
                "analysis": ai_response,
                "patterns": ["AI analysis provided in text format"],
                "recommendations": ["Review AI analysis for specific recommendations"],
                "priorities": ["Medium"],
                "efforts": ["Medium"]
            }
        except Exception:
            return {
                "analysis": ai_response,
                "patterns": [],
                "recommendations": [],
                "error": "Failed to parse AI response"
            }
    
    def _parse_optimization_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI optimization response."""
        try:
            if ai_response.strip().startswith('{'):
                return json.loads(ai_response)
            
            return {
                "strategy": ai_response,
                "timeouts": {},
                "ordering": [],
                "resources": {},
                "bottlenecks": []
            }
        except Exception:
            return {
                "optimization": ai_response,
                "error": "Failed to parse AI response"
            }
    
    def _parse_stability_prediction(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI stability prediction response."""
        try:
            if ai_response.strip().startswith('{'):
                return json.loads(ai_response)
            
            return {
                "assessment": ai_response,
                "flaky_tests": [],
                "trends": [],
                "recommendations": []
            }
        except Exception:
            return {
                "prediction": ai_response,
                "error": "Failed to parse AI response"
            }
    
    def _parse_insights_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI insights response."""
        try:
            if ai_response.strip().startswith('{'):
                return json.loads(ai_response)
            
            return {
                "insights": ai_response,
                "trends": [],
                "performance": {},
                "quality": {},
                "recommendations": [],
                "risks": []
            }
        except Exception:
            return {
                "insights": ai_response,
                "error": "Failed to parse AI response"
            }

class AITestOptimizer:
    """AI-powered test execution optimizer."""
    
    def __init__(self, ai_analyzer: AITestAnalyzer):
        self.ai_analyzer = ai_analyzer
        self.optimization_cache = {}
    
    async def optimize_workflow(
        self, 
        workflow: TestWorkflow, 
        execution_history: List[Dict[str, Any]]
    ) -> TestWorkflow:
        """Optimize workflow configuration using AI insights."""
        
        # Get AI optimization recommendations
        optimization = await self.ai_analyzer.optimize_test_execution(
            execution_history, 
            workflow
        )
        
        # Apply optimizations to workflow
        optimized_workflow = self._apply_optimizations(workflow, optimization)
        
        # Log optimization actions
        logger.info(f"Applied AI optimizations to workflow {workflow.id}")
        
        return optimized_workflow
    
    def _apply_optimizations(
        self, 
        workflow: TestWorkflow, 
        optimization: Dict[str, Any]
    ) -> TestWorkflow:
        """Apply AI optimization recommendations to workflow."""
        
        optimized = workflow
        
        # Apply strategy optimizations
        strategy = optimization.get("strategy", {})
        if isinstance(strategy, dict):
            if "parallel" in strategy:
                optimized.parallel = strategy["parallel"]
        
        # Apply timeout optimizations
        timeouts = optimization.get("timeouts", {})
        if isinstance(timeouts, dict) and "recommended_timeout" in timeouts:
            optimized.timeout = timeouts["recommended_timeout"]
        
        # Apply resource optimizations
        resources = optimization.get("resources", {})
        if isinstance(resources, dict):
            # Could adjust agent assignments, memory limits, etc.
            pass
        
        return optimized

class AIInsightsDashboard:
    """AI-powered insights dashboard for test management."""
    
    def __init__(self, ai_analyzer: AITestAnalyzer):
        self.ai_analyzer = ai_analyzer
    
    async def generate_executive_summary(
        self, 
        workflows: List[TestWorkflow],
        timeframe_days: int = 7
    ) -> Dict[str, Any]:
        """Generate executive summary with AI insights."""
        
        insights = await self.ai_analyzer.generate_test_insights(
            workflows, 
            timeframe_days
        )
        
        # Create executive summary
        summary = {
            "period": f"Last {timeframe_days} days",
            "generated_at": datetime.now().isoformat(),
            "ai_insights": insights,
            "key_metrics": self._calculate_key_metrics(workflows),
            "recommendations": self._prioritize_recommendations(insights)
        }
        
        return summary
    
    def _calculate_key_metrics(self, workflows: List[TestWorkflow]) -> Dict[str, Any]:
        """Calculate key metrics from workflows."""
        
        if not workflows:
            return {}
        
        total_workflows = len(workflows)
        successful = len([w for w in workflows if w.status.value == "completed"])
        
        # Calculate test metrics
        all_results = []
        for workflow in workflows:
            all_results.extend(workflow.results)
        
        total_tests = len(all_results)
        passed_tests = len([r for r in all_results if r.status == TestStatus.PASSED])
        
        return {
            "workflow_success_rate": (successful / total_workflows * 100) if total_workflows > 0 else 0,
            "test_pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_workflows": total_workflows,
            "total_tests": total_tests
        }
    
    def _prioritize_recommendations(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize AI recommendations."""
        
        recommendations = insights.get("recommendations", [])
        
        if isinstance(recommendations, list):
            # Add priority scoring based on keywords
            prioritized = []
            for rec in recommendations:
                if isinstance(rec, str):
                    priority = self._calculate_priority(rec)
                    prioritized.append({
                        "recommendation": rec,
                        "priority": priority,
                        "category": self._categorize_recommendation(rec)
                    })
                elif isinstance(rec, dict):
                    prioritized.append(rec)
            
            # Sort by priority
            prioritized.sort(key=lambda x: x.get("priority", 0), reverse=True)
            return prioritized
        
        return []
    
    def _calculate_priority(self, recommendation: str) -> int:
        """Calculate priority score for recommendation."""
        high_priority_keywords = ["critical", "failure", "error", "security", "performance"]
        medium_priority_keywords = ["optimize", "improve", "enhance", "update"]
        
        rec_lower = recommendation.lower()
        
        if any(keyword in rec_lower for keyword in high_priority_keywords):
            return 3  # High priority
        elif any(keyword in rec_lower for keyword in medium_priority_keywords):
            return 2  # Medium priority
        else:
            return 1  # Low priority
    
    def _categorize_recommendation(self, recommendation: str) -> str:
        """Categorize recommendation by type."""
        rec_lower = recommendation.lower()
        
        if any(word in rec_lower for word in ["performance", "speed", "timeout"]):
            return "Performance"
        elif any(word in rec_lower for word in ["security", "auth", "permission"]):
            return "Security"
        elif any(word in rec_lower for word in ["flaky", "stability", "reliable"]):
            return "Reliability"
        elif any(word in rec_lower for word in ["coverage", "test", "quality"]):
            return "Quality"
        else:
            return "General"