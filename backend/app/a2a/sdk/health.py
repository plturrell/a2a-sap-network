"""
Standardized Health Check Implementation for A2A Agents
Provides comprehensive health status including blockchain connectivity
"""

from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import psutil
import os


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class StandardHealthCheck:
    """
    Standardized health check for all A2A agents
    Includes system metrics, blockchain status, and agent-specific checks
    """
    
    @staticmethod
    async def get_health_status(
        agent_instance: Any,
        include_metrics: bool = True,
        include_blockchain: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive health status for an agent
        
        Args:
            agent_instance: The agent instance to check
            include_metrics: Include system metrics
            include_blockchain: Include blockchain status
            
        Returns:
            Health status dictionary
        """
        health_data = {
            "status": HealthStatus.HEALTHY.value,
            "timestamp": datetime.utcnow().isoformat(),
            "agent": {
                "id": getattr(agent_instance, "agent_id", "unknown"),
                "name": getattr(agent_instance, "name", "unknown"),
                "version": getattr(agent_instance, "version", "1.0.0"),
                "uptime_seconds": None
            },
            "checks": {}
        }
        
        # Calculate uptime if available
        if hasattr(agent_instance, "start_time"):
            uptime = (datetime.utcnow() - agent_instance.start_time).total_seconds()
            health_data["agent"]["uptime_seconds"] = int(uptime)
        
        # Add system metrics if requested
        if include_metrics:
            health_data["metrics"] = StandardHealthCheck._get_system_metrics()
        
        # Add blockchain status if requested and available
        if include_blockchain and hasattr(agent_instance, "get_blockchain_stats"):
            blockchain_stats = agent_instance.get_blockchain_stats()
            health_data["blockchain"] = {
                "enabled": blockchain_stats.get("enabled", False),
                "registered": blockchain_stats.get("registered", False),
                "address": blockchain_stats.get("address"),
                "reputation": blockchain_stats.get("reputation", 0)
            }
            
            # Update overall status based on blockchain
            if blockchain_stats.get("enabled") and not blockchain_stats.get("registered"):
                health_data["status"] = HealthStatus.DEGRADED.value
                health_data["checks"]["blockchain_registration"] = {
                    "status": "failed",
                    "message": "Agent not registered on blockchain"
                }
        
        # Add agent-specific health checks
        if hasattr(agent_instance, "health_checks"):
            custom_checks = await agent_instance.health_checks()
            health_data["checks"].update(custom_checks)
            
            # Update overall status based on custom checks
            for check_name, check_data in custom_checks.items():
                if check_data.get("status") == "failed":
                    health_data["status"] = HealthStatus.UNHEALTHY.value
                    break
                elif check_data.get("status") == "degraded":
                    if health_data["status"] != HealthStatus.UNHEALTHY.value:
                        health_data["status"] = HealthStatus.DEGRADED.value
        
        # Add agent capabilities
        if hasattr(agent_instance, "skills"):
            health_data["capabilities"] = {
                "skills": list(agent_instance.skills.keys()),
                "handlers": list(getattr(agent_instance, "handlers", {}).keys())
            }
        
        # Add task queue status if available
        if hasattr(agent_instance, "tasks"):
            active_tasks = [
                t for t in agent_instance.tasks.values()
                if t.get("status") in ["PENDING", "RUNNING"]
            ]
            health_data["tasks"] = {
                "total": len(agent_instance.tasks),
                "active": len(active_tasks),
                "completed": len([
                    t for t in agent_instance.tasks.values()
                    if t.get("status") == "COMPLETED"
                ]),
                "failed": len([
                    t for t in agent_instance.tasks.values()
                    if t.get("status") == "FAILED"
                ])
            }
        
        # Add message queue status if available
        if hasattr(agent_instance, "message_stats"):
            health_data["messages"] = agent_instance.message_stats
        
        return health_data
    
    @staticmethod
    def _get_system_metrics() -> Dict[str, Any]:
        """Get system resource metrics"""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            
            # Get disk usage for current directory
            disk = psutil.disk_usage(os.getcwd())
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "used_percent": memory.percent,
                    "used_gb": round(memory.used / (1024**3), 2),
                    "total_gb": round(memory.total / (1024**3), 2)
                },
                "disk": {
                    "used_percent": disk.percent,
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2)
                }
            }
        except Exception:
            # Return empty metrics if psutil fails
            return {}
    
    @staticmethod
    def create_health_endpoint(app, agent_instance):
        """
        Create a standardized health endpoint for FastAPI apps
        
        Args:
            app: FastAPI application instance
            agent_instance: The agent instance
        """
        @app.get("/health")
        async def health(
            detailed: bool = False,
            include_metrics: bool = False,
            include_blockchain: bool = True
        ):
            """
            Health check endpoint
            
            Query parameters:
            - detailed: Include all health information
            - include_metrics: Include system metrics
            - include_blockchain: Include blockchain status
            """
            if detailed:
                include_metrics = True
                include_blockchain = True
            
            return await StandardHealthCheck.get_health_status(
                agent_instance,
                include_metrics=include_metrics,
                include_blockchain=include_blockchain
            )
        
        # Also add a simple liveness check
        @app.get("/health/live")
        async def liveness():
            """Simple liveness check"""
            return {"status": "alive"}
        
        # And a readiness check
        @app.get("/health/ready")
        async def readiness():
            """Readiness check - includes blockchain registration"""
            health = await StandardHealthCheck.get_health_status(
                agent_instance,
                include_metrics=False,
                include_blockchain=True
            )
            
            if health["status"] == HealthStatus.HEALTHY.value:
                return {"status": "ready"}
            else:
                return {"status": "not ready", "reason": health.get("checks", {})}