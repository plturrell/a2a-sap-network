"""
Real-time Metrics Integration for Goal Management
Connects goal tracking with actual agent performance metrics
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import json

from ....core.a2aTypes import A2AMessage, MessagePart, MessageRole
from ....network.agentRegistration import get_registration_service

logger = logging.getLogger(__name__)

class RealTimeMetricsCollector:
    """Collects real-time metrics from agents for goal progress tracking"""
    
    def __init__(self, orchestrator_handler):
        self.orchestrator_handler = orchestrator_handler
        self.registry_service = get_registration_service()
        self.metrics_cache = {}
        self.last_update = {}
        
    async def start_monitoring(self):
        """Start continuous metrics monitoring"""
        logger.info("Starting real-time metrics monitoring")
        
        while True:
            try:
                # Get all agents with goals
                agents_with_goals = list(self.orchestrator_handler.agent_goals.keys())
                
                for agent_id in agents_with_goals:
                    await self.collect_agent_metrics(agent_id)
                    await self.update_goal_progress_from_metrics(agent_id)
                
                # Wait 2 minutes before next collection cycle
                await asyncio.sleep(120)
                
            except Exception as e:
                logger.error(f"Metrics monitoring cycle failed: {e}")
                await asyncio.sleep(60)  # Shorter wait on error
    
    async def collect_agent_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Collect metrics from a specific agent"""
        try:
            # Get agent registration info
            agent_record = await self.registry_service.get_agent(agent_id)
            if not agent_record:
                logger.warning(f"Agent {agent_id} not found in registry")
                return None
            
            metrics = {}
            
            # Collect health metrics
            if agent_record.agent_card.healthEndpoint:
                health_metrics = await self._fetch_health_metrics(
                    str(agent_record.agent_card.healthEndpoint)
                )
                if health_metrics:
                    metrics.update(health_metrics)
            
            # Collect performance metrics
            if agent_record.agent_card.metricsEndpoint:
                perf_metrics = await self._fetch_performance_metrics(
                    str(agent_record.agent_card.metricsEndpoint)
                )
                if perf_metrics:
                    metrics.update(perf_metrics)
            
            # Collect A2A-specific metrics
            a2a_metrics = await self._fetch_a2a_metrics(agent_id)
            if a2a_metrics:
                metrics.update(a2a_metrics)
            
            # Cache metrics
            self.metrics_cache[agent_id] = {
                "metrics": metrics,
                "timestamp": datetime.utcnow(),
                "agent_url": str(agent_record.agent_card.url)
            }
            
            logger.debug(f"Collected metrics for {agent_id}: {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics for {agent_id}: {e}")
            return None
    
    async def _fetch_health_metrics(self, health_url: str) -> Optional[Dict[str, Any]]:
        """Fetch health metrics from agent health endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "health_status": data.get("status", "unknown"),
                            "uptime": data.get("uptime", 0),
                            "response_time": data.get("response_time", 0),
                            "memory_usage": data.get("memory_usage", 0),
                            "cpu_usage": data.get("cpu_usage", 0)
                        }
        except Exception as e:
            logger.debug(f"Failed to fetch health metrics from {health_url}: {e}")
            return None
    
    async def _fetch_performance_metrics(self, metrics_url: str) -> Optional[Dict[str, Any]]:
        """Fetch performance metrics from agent metrics endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(metrics_url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "total_requests": data.get("total_requests", 0),
                            "successful_requests": data.get("successful_requests", 0),
                            "failed_requests": data.get("failed_requests", 0),
                            "avg_response_time": data.get("avg_response_time", 0),
                            "requests_per_minute": data.get("requests_per_minute", 0),
                            "error_rate": data.get("error_rate", 0)
                        }
        except Exception as e:
            logger.debug(f"Failed to fetch performance metrics from {metrics_url}: {e}")
            return None
    
    async def _fetch_a2a_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Fetch A2A-specific metrics from blockchain/network"""
        try:
            # Get A2A network metrics for the agent
            # This would integrate with the A2A network client
            
            # Placeholder implementation
            return {
                "a2a_messages_sent": 150,
                "a2a_messages_received": 200,
                "blockchain_transactions": 75,
                "network_latency": 45,
                "protocol_compliance_score": 98.5
            }
            
        except Exception as e:
            logger.debug(f"Failed to fetch A2A metrics for {agent_id}: {e}")
            return None
    
    async def update_goal_progress_from_metrics(self, agent_id: str):
        """Update goal progress based on collected metrics"""
        try:
            if agent_id not in self.metrics_cache:
                return
            
            metrics_data = self.metrics_cache[agent_id]
            metrics = metrics_data["metrics"]
            
            # Skip if no meaningful metrics
            if not metrics:
                return
            
            # Calculate progress based on agent-specific goals
            progress_update = await self._calculate_progress_from_metrics(agent_id, metrics)
            
            if not progress_update:
                return
            
            # Create A2A message to update progress
            message_data = {
                "operation": "track_goal_progress",
                "data": {
                    "agent_id": agent_id,
                    "progress": progress_update
                }
            }
            
            message = A2AMessage(
                sender_id="metrics_collector",
                recipient_id="orchestrator_agent",
                parts=[MessagePart(
                    role=MessageRole.USER,
                    data=message_data
                )],
                timestamp=datetime.utcnow()
            )
            
            result = await self.orchestrator_handler.process_a2a_message(message)
            
            if result.get("status") == "success":
                logger.info(f"Updated goal progress from metrics for {agent_id}")
                self.last_update[agent_id] = datetime.utcnow()
            else:
                logger.warning(f"Failed to update progress for {agent_id}: {result}")
                
        except Exception as e:
            logger.error(f"Failed to update goal progress from metrics for {agent_id}: {e}")
    
    async def _calculate_progress_from_metrics(self, agent_id: str, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate goal progress based on metrics"""
        try:
            # Get current goals for the agent
            if agent_id not in self.orchestrator_handler.agent_goals:
                return None
            
            goals = self.orchestrator_handler.agent_goals[agent_id]["goals"]
            current_progress = self.orchestrator_handler.goal_progress.get(agent_id, {})
            
            # Calculate overall progress based on multiple factors
            success_rate = 0.0
            if metrics.get("total_requests", 0) > 0:
                success_rate = (metrics.get("successful_requests", 0) / metrics.get("total_requests", 1)) * 100
            
            # Calculate objective-specific progress
            objective_progress = {}
            
            # Data registration progress (based on successful requests)
            if "data_registration" in str(goals.get("primary_objectives", [])):
                objective_progress["data_registration"] = min(100.0, success_rate)
            
            # Validation accuracy (based on error rate)
            if "validation" in str(goals.get("primary_objectives", [])):
                error_rate = metrics.get("error_rate", 0)
                objective_progress["validation_accuracy"] = max(0.0, 100.0 - error_rate)
            
            # Response time performance
            if "response_time" in str(goals.get("success_criteria", [])):
                avg_response = metrics.get("avg_response_time", 5000)
                # Target: <5 seconds (5000ms), scale to percentage
                response_score = max(0.0, 100.0 - (avg_response / 50))
                objective_progress["response_time"] = min(100.0, response_score)
            
            # Compliance tracking (based on A2A protocol compliance)
            if "compliance" in str(goals.get("primary_objectives", [])):
                compliance_score = metrics.get("protocol_compliance_score", 95.0)
                objective_progress["compliance_tracking"] = compliance_score
            
            # Quality assessment (based on overall health)
            if "quality" in str(goals.get("primary_objectives", [])):
                health_score = 100.0 if metrics.get("health_status") == "healthy" else 50.0
                uptime_score = min(100.0, metrics.get("uptime", 0) / 86400 * 100)  # Daily uptime
                objective_progress["quality_assessment"] = (health_score + uptime_score) / 2
            
            # Calculate overall progress as average of objectives
            if objective_progress:
                overall_progress = sum(objective_progress.values()) / len(objective_progress)
            else:
                overall_progress = success_rate
            
            # Generate milestones based on metrics
            milestones = []
            current_milestones = current_progress.get("milestones_achieved", [])
            
            if success_rate > 95 and "High success rate achieved (>95%)" not in current_milestones:
                milestones.append("High success rate achieved (>95%)")
            
            if metrics.get("avg_response_time", 5000) < 2000 and "Fast response time achieved (<2s)" not in current_milestones:
                milestones.append("Fast response time achieved (<2s)")
            
            if metrics.get("uptime", 0) > 86400 * 0.999 and "High availability achieved (>99.9%)" not in current_milestones:
                milestones.append("High availability achieved (>99.9%)")
            
            if metrics.get("error_rate", 100) < 1 and "Low error rate achieved (<1%)" not in current_milestones:
                milestones.append("Low error rate achieved (<1%)")
            
            # Only return update if there are significant changes
            progress_update = {}
            
            # Update overall progress if changed significantly
            current_overall = current_progress.get("overall_progress", 0.0)
            if abs(overall_progress - current_overall) > 1.0:  # 1% threshold
                progress_update["overall_progress"] = round(overall_progress, 1)
            
            # Update objective progress if changed
            if objective_progress:
                progress_update["objective_progress"] = {
                    k: round(v, 1) for k, v in objective_progress.items()
                }
            
            # Add new milestones
            if milestones:
                progress_update["milestones_achieved"] = milestones
            
            return progress_update if progress_update else None
            
        except Exception as e:
            logger.error(f"Failed to calculate progress from metrics for {agent_id}: {e}")
            return None
    
    def get_cached_metrics(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get cached metrics for an agent"""
        return self.metrics_cache.get(agent_id)
    
    def get_all_cached_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all cached metrics"""
        return self.metrics_cache.copy()

# Global metrics collector instance
_metrics_collector: Optional[RealTimeMetricsCollector] = None

def get_metrics_collector(orchestrator_handler) -> RealTimeMetricsCollector:
    """Get or create metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = RealTimeMetricsCollector(orchestrator_handler)
    return _metrics_collector

async def start_metrics_monitoring(orchestrator_handler):
    """Start metrics monitoring as background task"""
    collector = get_metrics_collector(orchestrator_handler)
    await collector.start_monitoring()
