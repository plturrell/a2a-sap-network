"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""

#!/usr/bin/env python3
"""
A2A Network Chaos Engineering Framework
Implements fault injection, failure simulation, and resilience testing
"""

import asyncio
import random
import logging
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
# Direct HTTP calls not allowed - use A2A protocol
# import httpx  # REMOVED: A2A protocol violation
import docker
import kubernetes
from kubernetes import client, config
import psutil
import redis
import time
import signal
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChaosType(str, Enum):
    """Types of chaos experiments"""
    NETWORK_PARTITION = "network_partition"
    SERVICE_KILL = "service_kill"
    CPU_STRESS = "cpu_stress"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_IO_STRESS = "disk_io_stress"
    NETWORK_DELAY = "network_delay"
    PACKET_LOSS = "packet_loss"
    DATABASE_FAILURE = "database_failure"
    REDIS_FAILURE = "redis_failure"
    MESSAGE_DROP = "message_drop"
    BLOCKCHAIN_PARTITION = "blockchain_partition"
    API_THROTTLING = "api_throttling"
    DEPENDENCY_TIMEOUT = "dependency_timeout"


class ExperimentStatus(str, Enum):
    """Chaos experiment status"""
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ChaosExperiment:
    """Chaos experiment configuration"""
    id: str
    name: str
    description: str
    chaos_type: ChaosType
    target_service: str
    parameters: Dict[str, Any]
    duration_seconds: int
    impact_radius: str = "single"  # single, multiple, cluster
    safety_checks: List[str] = field(default_factory=list)
    rollback_strategy: str = "automatic"
    scheduled_at: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.SCHEDULED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    metrics_before: Dict[str, Any] = field(default_factory=dict)
    metrics_after: Dict[str, Any] = field(default_factory=dict)


class ChaosMonkey:
    """Main chaos engineering orchestrator for A2A Network"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.running_experiments: Dict[str, asyncio.Task] = {}
        self.safety_enabled = True
        self.dry_run = False
        
        # Initialize clients
        self._init_clients()
        
        # Experiment registry
        self.experiment_handlers = {
            ChaosType.NETWORK_PARTITION: self._network_partition,
            ChaosType.SERVICE_KILL: self._service_kill,
            ChaosType.CPU_STRESS: self._cpu_stress,
            ChaosType.MEMORY_PRESSURE: self._memory_pressure,
            ChaosType.NETWORK_DELAY: self._network_delay,
            ChaosType.PACKET_LOSS: self._packet_loss,
            ChaosType.DATABASE_FAILURE: self._database_failure,
            ChaosType.REDIS_FAILURE: self._redis_failure,
            ChaosType.MESSAGE_DROP: self._message_drop,
            ChaosType.BLOCKCHAIN_PARTITION: self._blockchain_partition,
            ChaosType.API_THROTTLING: self._api_throttling,
            ChaosType.DEPENDENCY_TIMEOUT: self._dependency_timeout
        }
        
        logger.info("A2A Chaos Monkey initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load chaos engineering configuration"""
        default_config = {
            "safety_checks": {
                "max_concurrent_experiments": 2,
                "business_hours_protection": True,
                "critical_service_protection": ["a2a-agent-manager", "a2a-registry"],
                "max_impact_percentage": 25
            },
            "monitoring": {
                "prometheus_url": "https://prometheus:9090",
                "grafana_url": "https://grafana:3000",
                "alert_manager_url": "https://alertmanager:9093"
            },
            "targets": {
                "kubernetes": {
                    "enabled": True,
                    "namespace": "a2a-network"
                },
                "docker": {
                    "enabled": True,
                    "network": "a2a-network"
                }
            },
            "experiments": {
                "default_duration": 300,  # 5 minutes
                "max_duration": 1800,     # 30 minutes
                "cooldown_period": 600    # 10 minutes
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config

    def _init_clients(self):
        """Initialize external clients"""
        try:
            # Docker client
            self.docker_client = docker.from_env()
            
            # Kubernetes client
            if self.config["targets"]["kubernetes"]["enabled"]:
                try:
                    config.load_incluster_config()
                except:
                    config.load_kube_config()
                self.k8s_client = client.AppsV1Api()
                self.k8s_core = client.CoreV1Api()
            
            # HTTP client for API calls
            self.http_client = # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        # httpx\.AsyncClient(timeout=30.0)
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")

    async def create_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """Create a new chaos experiment"""
        experiment = ChaosExperiment(
            id=experiment_config.get("id", f"chaos-{int(time.time())}"),
            name=experiment_config["name"],
            description=experiment_config.get("description", ""),
            chaos_type=ChaosType(experiment_config["chaos_type"]),
            target_service=experiment_config["target_service"],
            parameters=experiment_config.get("parameters", {}),
            duration_seconds=experiment_config.get("duration_seconds", self.config["experiments"]["default_duration"]),
            impact_radius=experiment_config.get("impact_radius", "single"),
            safety_checks=experiment_config.get("safety_checks", []),
            rollback_strategy=experiment_config.get("rollback_strategy", "automatic"),
            scheduled_at=experiment_config.get("scheduled_at")
        )
        
        # Validate experiment
        if not await self._validate_experiment(experiment):
            raise ValueError("Experiment validation failed")
        
        self.experiments[experiment.id] = experiment
        logger.info(f"Created chaos experiment: {experiment.id}")
        
        return experiment.id

    async def start_experiment(self, experiment_id: str) -> bool:
        """Start a chaos experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        # Safety checks
        if not await self._safety_check(experiment):
            logger.error(f"Safety check failed for experiment {experiment_id}")
            return False
        
        # Pre-experiment metrics
        experiment.metrics_before = await self._collect_metrics(experiment.target_service)
        
        # Start experiment
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.utcnow()
        
        # Run experiment in background
        task = asyncio.create_task(self._run_experiment(experiment))
        self.running_experiments[experiment_id] = task
        
        logger.info(f"Started chaos experiment: {experiment_id}")
        return True

    async def _run_experiment(self, experiment: ChaosExperiment):
        """Execute the chaos experiment"""
        try:
            handler = self.experiment_handlers.get(experiment.chaos_type)
            if not handler:
                raise ValueError(f"No handler for chaos type: {experiment.chaos_type}")
            
            logger.info(f"Executing {experiment.chaos_type} on {experiment.target_service}")
            
            # Execute the chaos
            chaos_context = await handler(experiment)
            
            # Monitor during chaos
            monitoring_task = asyncio.create_task(
                self._monitor_experiment(experiment)
            )
            
            # Wait for experiment duration
            await asyncio.sleep(experiment.duration_seconds)
            
            # Stop monitoring
            monitoring_task.cancel()
            
            # Rollback
            await self._rollback_experiment(experiment, chaos_context)
            
            # Post-experiment metrics
            experiment.metrics_after = await self._collect_metrics(experiment.target_service)
            
            # Calculate results
            experiment.results = await self._analyze_experiment_results(experiment)
            
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = datetime.utcnow()
            
            logger.info(f"Completed chaos experiment: {experiment.id}")
            
        except Exception as e:
            logger.error(f"Experiment {experiment.id} failed: {e}")
            experiment.status = ExperimentStatus.FAILED
            experiment.results["error"] = str(e)
            
            # Emergency rollback
            await self._emergency_rollback(experiment)
            
        finally:
            # Cleanup
            if experiment.id in self.running_experiments:
                del self.running_experiments[experiment.id]

    async def _validate_experiment(self, experiment: ChaosExperiment) -> bool:
        """Validate experiment safety and configuration"""
        # Check if target service exists
        if not await self._service_exists(experiment.target_service):
            logger.error(f"Target service not found: {experiment.target_service}")
            return False
        
        # Check concurrent experiments limit
        if len(self.running_experiments) >= self.config["safety_checks"]["max_concurrent_experiments"]:
            logger.error("Maximum concurrent experiments limit reached")
            return False
        
        # Check business hours protection
        if self.config["safety_checks"]["business_hours_protection"]:
            now = datetime.now()
            if 9 <= now.hour <= 17 and now.weekday() < 5:  # Business hours
                logger.error("Business hours protection enabled")
                return False
        
        # Check critical service protection
        if experiment.target_service in self.config["safety_checks"]["critical_service_protection"]:
            logger.error(f"Target service is protected: {experiment.target_service}")
            return False
        
        return True

    async def _safety_check(self, experiment: ChaosExperiment) -> bool:
        """Perform real-time safety checks before execution"""
        try:
            # Check system health
            health_metrics = await self._get_system_health()
            
            # Check if system is already under stress
            if health_metrics.get("cpu_usage", 0) > 80:
                logger.error("System CPU usage too high for chaos experiment")
                return False
            
            if health_metrics.get("memory_usage", 0) > 80:
                logger.error("System memory usage too high for chaos experiment")
                return False
            
            # Check recent alerts
            recent_alerts = await self._get_recent_alerts()
            if len(recent_alerts) > 5:
                logger.error("Too many recent alerts, skipping chaos experiment")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return False

    # Chaos experiment implementations
    
    async def _network_partition(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate network partition"""
        target_service = experiment.target_service
        parameters = experiment.parameters
        
        logger.info(f"Creating network partition for {target_service}")
        
        if self.dry_run:
            logger.info("[DRY RUN] Would create network partition")
            return {"action": "network_partition", "dry_run": True}
        
        # Implementation would use iptables or network policies
        context = {
            "action": "network_partition",
            "target": target_service,
            "blocked_ports": parameters.get("ports", []),
            "blocked_ips": parameters.get("ips", [])
        }
        
        # Simulate network rules application
        await asyncio.sleep(1)
        
        return context

    async def _service_kill(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Kill target service"""
        target_service = experiment.target_service
        
        logger.info(f"Killing service: {target_service}")
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would kill service {target_service}")
            return {"action": "service_kill", "dry_run": True}
        
        context = {"action": "service_kill", "target": target_service, "killed_instances": []}
        
        try:
            # Docker implementation
            containers = self.docker_client.containers.list(
                filters={"name": target_service}
            )
            
            for container in containers:
                container.kill()
                context["killed_instances"].append(container.id)
                logger.info(f"Killed container: {container.id}")
                
        except Exception as e:
            logger.error(f"Failed to kill service: {e}")
            raise
        
        return context

    async def _cpu_stress(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Create CPU stress on target"""
        target_service = experiment.target_service
        parameters = experiment.parameters
        cpu_percentage = parameters.get("cpu_percentage", 80)
        
        logger.info(f"Creating CPU stress on {target_service}: {cpu_percentage}%")
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would create CPU stress: {cpu_percentage}%")
            return {"action": "cpu_stress", "dry_run": True}
        
        # Implementation would use stress-ng or similar
        context = {
            "action": "cpu_stress",
            "target": target_service,
            "cpu_percentage": cpu_percentage
        }
        
        return context

    async def _memory_pressure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Create memory pressure"""
        target_service = experiment.target_service
        parameters = experiment.parameters
        memory_mb = parameters.get("memory_mb", 1024)
        
        logger.info(f"Creating memory pressure on {target_service}: {memory_mb}MB")
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would create memory pressure: {memory_mb}MB")
            return {"action": "memory_pressure", "dry_run": True}
        
        context = {
            "action": "memory_pressure",
            "target": target_service,
            "memory_mb": memory_mb
        }
        
        return context

    async def _network_delay(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Add network delay"""
        target_service = experiment.target_service
        parameters = experiment.parameters
        delay_ms = parameters.get("delay_ms", 100)
        
        logger.info(f"Adding network delay to {target_service}: {delay_ms}ms")
        
        context = {
            "action": "network_delay",
            "target": target_service,
            "delay_ms": delay_ms
        }
        
        return context

    async def _packet_loss(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate packet loss"""
        target_service = experiment.target_service
        parameters = experiment.parameters
        loss_percentage = parameters.get("loss_percentage", 5)
        
        logger.info(f"Simulating packet loss on {target_service}: {loss_percentage}%")
        
        context = {
            "action": "packet_loss",
            "target": target_service,
            "loss_percentage": loss_percentage
        }
        
        return context

    async def _database_failure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate database failure"""
        logger.info("Simulating database failure")
        
        # This would disconnect database connections or stop database service
        context = {"action": "database_failure"}
        
        return context

    async def _redis_failure(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate Redis failure"""
        logger.info("Simulating Redis failure")
        
        context = {"action": "redis_failure"}
        
        return context

    async def _message_drop(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Drop messages in transit"""
        parameters = experiment.parameters
        drop_percentage = parameters.get("drop_percentage", 10)
        
        logger.info(f"Dropping messages: {drop_percentage}%")
        
        context = {
            "action": "message_drop",
            "drop_percentage": drop_percentage
        }
        
        return context

    async def _blockchain_partition(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate blockchain network partition"""
        logger.info("Simulating blockchain partition")
        
        context = {"action": "blockchain_partition"}
        
        return context

    async def _api_throttling(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Throttle API requests"""
        parameters = experiment.parameters
        throttle_percentage = parameters.get("throttle_percentage", 50)
        
        logger.info(f"Throttling API requests: {throttle_percentage}%")
        
        context = {
            "action": "api_throttling",
            "throttle_percentage": throttle_percentage
        }
        
        return context

    async def _dependency_timeout(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Simulate dependency timeouts"""
        parameters = experiment.parameters
        timeout_ms = parameters.get("timeout_ms", 5000)
        
        logger.info(f"Simulating dependency timeout: {timeout_ms}ms")
        
        context = {
            "action": "dependency_timeout",
            "timeout_ms": timeout_ms
        }
        
        return context

    # Monitoring and analysis
    
    async def _monitor_experiment(self, experiment: ChaosExperiment):
        """Monitor system during chaos experiment"""
        try:
            while experiment.status == ExperimentStatus.RUNNING:
                # Collect metrics
                metrics = await self._collect_metrics(experiment.target_service)
                
                # Check if system is in dangerous state
                if metrics.get("error_rate", 0) > 50:  # 50% error rate
                    logger.error("Critical error rate detected, aborting experiment")
                    experiment.status = ExperimentStatus.ABORTED
                    break
                
                if metrics.get("response_time_p99", 0) > 10:  # 10 second response time
                    logger.warning("High response times detected")
                
                # Log current state
                logger.debug(f"Experiment {experiment.id} metrics: {metrics}")
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
        except asyncio.CancelledError:
            logger.info(f"Monitoring cancelled for experiment {experiment.id}")

    async def _collect_metrics(self, service_name: str) -> Dict[str, Any]:
        """Collect system and application metrics"""
        metrics = {}
        
        try:
            # System metrics
            metrics["cpu_usage"] = psutil.cpu_percent()
            metrics["memory_usage"] = psutil.virtual_memory().percent
            metrics["disk_usage"] = psutil.disk_usage('/').percent
            
            # Application metrics (from Prometheus)
            prometheus_url = self.config["monitoring"]["prometheus_url"]
            
            queries = {
                "error_rate": f'sum(rate(http_requests_total{{status=~"5.."}}[5m])) / sum(rate(http_requests_total[5m])) * 100',
                "response_time_p99": f'histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))',
                "throughput": f'sum(rate(http_requests_total[5m]))'
            }
            
            for metric_name, query in queries.items():
                try:
                    response = await self.http_client.get(
                        f"{prometheus_url}/api/v1/query",
                        params={"query": query}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data["data"]["result"]:
                            metrics[metric_name] = float(data["data"]["result"][0]["value"][1])
                        else:
                            metrics[metric_name] = 0
                    
                except Exception as e:
                    logger.warning(f"Failed to collect metric {metric_name}: {e}")
                    metrics[metric_name] = 0
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
        
        return metrics

    async def _rollback_experiment(self, experiment: ChaosExperiment, context: Dict[str, Any]):
        """Rollback chaos experiment"""
        logger.info(f"Rolling back experiment: {experiment.id}")
        
        action = context.get("action")
        
        try:
            if action == "service_kill":
                # Restart killed services
                for container_id in context.get("killed_instances", []):
                    try:
                        container = self.docker_client.containers.get(container_id)
                        container.restart()
                        logger.info(f"Restarted container: {container_id}")
                    except Exception as e:
                        logger.error(f"Failed to restart container {container_id}: {e}")
            
            elif action in ["network_partition", "network_delay", "packet_loss"]:
                # Remove network rules
                logger.info("Removing network chaos rules")
                # Implementation would remove iptables rules or network policies
            
            elif action in ["cpu_stress", "memory_pressure"]:
                # Stop stress processes
                logger.info("Stopping stress processes")
                # Implementation would kill stress processes
            
            else:
                logger.info(f"No specific rollback needed for {action}")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise

    async def _emergency_rollback(self, experiment: ChaosExperiment):
        """Emergency rollback when experiment fails"""
        logger.error(f"Emergency rollback for experiment: {experiment.id}")
        
        # Implement emergency procedures
        # This might include restarting all related services
        
        try:
            # Restart target service
            containers = self.docker_client.containers.list(
                filters={"name": experiment.target_service}
            )
            
            for container in containers:
                container.restart()
                logger.info(f"Emergency restart of container: {container.id}")
                
        except Exception as e:
            logger.error(f"Emergency rollback failed: {e}")

    async def _analyze_experiment_results(self, experiment: ChaosExperiment) -> Dict[str, Any]:
        """Analyze experiment results and generate insights"""
        results = {
            "experiment_id": experiment.id,
            "duration_seconds": experiment.duration_seconds,
            "chaos_type": experiment.chaos_type.value,
            "target_service": experiment.target_service
        }
        
        # Compare before/after metrics
        before = experiment.metrics_before
        after = experiment.metrics_after
        
        if before and after:
            results["metric_changes"] = {}
            
            for metric in before:
                if metric in after:
                    change = after[metric] - before[metric]
                    change_percent = (change / before[metric] * 100) if before[metric] > 0 else 0
                    
                    results["metric_changes"][metric] = {
                        "before": before[metric],
                        "after": after[metric],
                        "change": change,
                        "change_percent": change_percent
                    }
        
        # Generate insights
        insights = []
        
        if "error_rate" in results.get("metric_changes", {}):
            error_change = results["metric_changes"]["error_rate"]["change_percent"]
            if error_change > 20:
                insights.append(f"Error rate increased by {error_change:.1f}% during chaos")
            elif error_change < -10:
                insights.append("System showed improved error handling during chaos")
        
        if "response_time_p99" in results.get("metric_changes", {}):
            latency_change = results["metric_changes"]["response_time_p99"]["change_percent"]
            if latency_change > 50:
                insights.append(f"Response times degraded by {latency_change:.1f}% during chaos")
        
        results["insights"] = insights
        results["recommendation"] = self._generate_recommendations(results)
        
        return results

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on experiment results"""
        recommendations = []
        
        metric_changes = results.get("metric_changes", {})
        
        if "error_rate" in metric_changes:
            error_change = metric_changes["error_rate"]["change_percent"]
            if error_change > 30:
                recommendations.append("Consider implementing better error handling and retries")
                recommendations.append("Review circuit breaker configurations")
        
        if "response_time_p99" in metric_changes:
            latency_change = metric_changes["response_time_p99"]["change_percent"]
            if latency_change > 100:
                recommendations.append("Optimize service response times under stress")
                recommendations.append("Consider implementing request queuing or load shedding")
        
        if results["chaos_type"] == "service_kill":
            recommendations.append("Verify service restart automation is working correctly")
            recommendations.append("Consider implementing graceful shutdown procedures")
        
        return recommendations

    # Utility methods
    
    async def _service_exists(self, service_name: str) -> bool:
        """Check if service exists"""
        try:
            containers = self.docker_client.containers.list(
                filters={"name": service_name}
            )
            return len(containers) > 0
        except:
            return False

    async def _get_system_health(self) -> Dict[str, float]:
        """Get current system health metrics"""
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }

    async def _get_recent_alerts(self) -> List[Dict[str, Any]]:
        """Get recent alerts from AlertManager"""
        try:
            alert_url = self.config["monitoring"]["alert_manager_url"]
            response = await self.http_client.get(f"{alert_url}/api/v1/alerts")
            
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
                
        except Exception as e:
            logger.error(f"Failed to get recent alerts: {e}")
        
        return []

    # API methods
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment status"""
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.experiments[experiment_id]
        return {
            "id": experiment.id,
            "name": experiment.name,
            "status": experiment.status.value,
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
            "completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None,
            "results": experiment.results
        }

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments"""
        return [
            {
                "id": exp.id,
                "name": exp.name,
                "status": exp.status.value,
                "chaos_type": exp.chaos_type.value,
                "target_service": exp.target_service,
                "created_at": exp.scheduled_at.isoformat() if exp.scheduled_at else None
            }
            for exp in self.experiments.values()
        ]

    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop running experiment"""
        if experiment_id in self.running_experiments:
            task = self.running_experiments[experiment_id]
            task.cancel()
            
            # Update experiment status
            if experiment_id in self.experiments:
                self.experiments[experiment_id].status = ExperimentStatus.ABORTED
                self.experiments[experiment_id].completed_at = datetime.utcnow()
            
            logger.info(f"Stopped experiment: {experiment_id}")
            return True
        
        return False

    def enable_dry_run(self):
        """Enable dry run mode"""
        self.dry_run = True
        logger.info("Dry run mode enabled")

    def disable_dry_run(self):
        """Disable dry run mode"""
        self.dry_run = False
        logger.info("Dry run mode disabled")


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, stopping chaos experiments...")
    sys.exit(0)


async def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    chaos_monkey = ChaosMonkey()
    
    # Example usage
    logger.info("A2A Chaos Monkey started")
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down Chaos Monkey...")


if __name__ == "__main__":
    asyncio.run(main())