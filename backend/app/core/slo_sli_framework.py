"""
Service Level Objectives (SLO) and Service Level Indicators (SLI) Framework
Provides production-ready reliability monitoring and alerting for A2A agents
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from uuid import uuid4
import time
import statistics

# A2A imports
from ..a2a.core.telemetry import trace_async, add_span_attributes
from ..clients.redisClient import RedisClient, RedisConfig
from ..clients.prometheusClient import PrometheusClient

logger = logging.getLogger(__name__)


class SLIType(str, Enum):
    """Types of Service Level Indicators"""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    QUEUE_DEPTH = "queue_depth"
    RESOURCE_UTILIZATION = "resource_utilization"
    AGENT_HEALTH = "agent_health"
    TASK_COMPLETION_RATE = "task_completion_rate"
    MESSAGE_DELIVERY_RATE = "message_delivery_rate"


class SLOStatus(str, Enum):
    """SLO compliance status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACHED = "breached"
    UNKNOWN = "unknown"


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SLI:
    """Service Level Indicator definition"""
    name: str
    description: str
    sli_type: SLIType
    query: str  # Prometheus query or calculation method
    unit: str
    good_total_ratio: bool = False  # True for good/total ratios
    threshold_direction: str = "lower"  # "lower" or "upper"
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SLO:
    """Service Level Objective definition"""
    name: str
    description: str
    sli: SLI
    target: float  # Target value (e.g., 0.999 for 99.9%)
    window_duration: timedelta  # Time window for evaluation
    alert_threshold: float  # When to alert (e.g., 0.95 for 95% of target)
    burn_rate_thresholds: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SLOResult:
    """SLO evaluation result"""
    slo_name: str
    current_value: float
    target_value: float
    compliance_percentage: float
    status: SLOStatus
    error_budget_remaining: float
    burn_rate: float
    time_to_exhaustion: Optional[timedelta]
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class A2ASLOFramework:
    """A2A Service Level Objectives Framework"""
    
    def __init__(
        self,
        agent_id: str,
        prometheus_config: Dict[str, Any] = None,
        redis_config: RedisConfig = None
    ):
        self.agent_id = agent_id
        self.prometheus_client = PrometheusClient(prometheus_config or {})
        self.redis_client = RedisClient(redis_config or RedisConfig())
        
        # SLO/SLI definitions
        self.slis: Dict[str, SLI] = {}
        self.slos: Dict[str, SLO] = {}
        
        # Monitoring state
        self.evaluation_results: Dict[str, SLOResult] = {}
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = {
            level: [] for level in AlertLevel
        }
        
        # Background tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self.running = False
        
        # Default A2A SLIs and SLOs
        self._initialize_default_slo_sli()
    
    def _initialize_default_slo_sli(self):
        """Initialize default SLO/SLI definitions for A2A agents"""
        
        # === AVAILABILITY SLIs ===
        self.slis["agent_availability"] = SLI(
            name="agent_availability",
            description="Agent endpoint availability",
            sli_type=SLIType.AVAILABILITY,
            query="up{job='a2a_agents', instance=~'.*%s.*'}" % self.agent_id,
            unit="ratio",
            good_total_ratio=True,
            threshold_direction="upper",
            tags={"component": "agent", "criticality": "high"}
        )
        
        self.slis["service_availability"] = SLI(
            name="service_availability",
            description="Agent service availability",
            sli_type=SLIType.AVAILABILITY,
            query="sum(rate(a2a_agent_requests_total{agent_id='%s', status!~'5..'}[5m])) / sum(rate(a2a_agent_requests_total{agent_id='%s'}[5m]))" % (self.agent_id, self.agent_id),
            unit="ratio",
            good_total_ratio=True,
            threshold_direction="upper"
        )
        
        # === LATENCY SLIs ===
        self.slis["response_latency_p95"] = SLI(
            name="response_latency_p95",
            description="95th percentile response latency",
            sli_type=SLIType.LATENCY,
            query="histogram_quantile(0.95, rate(a2a_agent_request_duration_seconds_bucket{agent_id='%s'}[5m]))" % self.agent_id,
            unit="seconds",
            threshold_direction="lower"
        )
        
        self.slis["response_latency_p99"] = SLI(
            name="response_latency_p99",
            description="99th percentile response latency",
            sli_type=SLIType.LATENCY,
            query="histogram_quantile(0.99, rate(a2a_agent_request_duration_seconds_bucket{agent_id='%s'}[5m]))" % self.agent_id,
            unit="seconds",
            threshold_direction="lower"
        )
        
        self.slis["task_processing_latency"] = SLI(
            name="task_processing_latency",
            description="Average task processing time",
            sli_type=SLIType.LATENCY,
            query="rate(a2a_agent_task_duration_seconds_sum{agent_id='%s'}[5m]) / rate(a2a_agent_task_duration_seconds_count{agent_id='%s'}[5m])" % (self.agent_id, self.agent_id),
            unit="seconds",
            threshold_direction="lower"
        )
        
        # === ERROR RATE SLIs ===
        self.slis["error_rate"] = SLI(
            name="error_rate",
            description="Agent error rate",
            sli_type=SLIType.ERROR_RATE,
            query="sum(rate(a2a_agent_requests_total{agent_id='%s', status=~'5..'}[5m])) / sum(rate(a2a_agent_requests_total{agent_id='%s'}[5m]))" % (self.agent_id, self.agent_id),
            unit="ratio",
            threshold_direction="lower"
        )
        
        self.slis["task_failure_rate"] = SLI(
            name="task_failure_rate",
            description="Task failure rate",
            sli_type=SLIType.ERROR_RATE,
            query="sum(rate(a2a_agent_tasks_total{agent_id='%s', status='failed'}[5m])) / sum(rate(a2a_agent_tasks_total{agent_id='%s'}[5m]))" % (self.agent_id, self.agent_id),
            unit="ratio",
            threshold_direction="lower"
        )
        
        # === THROUGHPUT SLIs ===
        self.slis["request_throughput"] = SLI(
            name="request_throughput",
            description="Requests per second",
            sli_type=SLIType.THROUGHPUT,
            query="sum(rate(a2a_agent_requests_total{agent_id='%s'}[5m]))" % self.agent_id,
            unit="rps",
            threshold_direction="upper"
        )
        
        self.slis["task_throughput"] = SLI(
            name="task_throughput",
            description="Tasks processed per second",
            sli_type=SLIType.THROUGHPUT,
            query="sum(rate(a2a_agent_tasks_total{agent_id='%s', status='completed'}[5m]))" % self.agent_id,
            unit="tps",
            threshold_direction="upper"
        )
        
        # === QUEUE DEPTH SLIs ===
        self.slis["task_queue_depth"] = SLI(
            name="task_queue_depth",
            description="Number of pending tasks in queue",
            sli_type=SLIType.QUEUE_DEPTH,
            query="a2a_agent_queue_depth{agent_id='%s', queue_type='tasks'}" % self.agent_id,
            unit="count",
            threshold_direction="lower"
        )
        
        self.slis["message_queue_depth"] = SLI(
            name="message_queue_depth",
            description="Number of pending messages in queue",
            sli_type=SLIType.QUEUE_DEPTH,
            query="a2a_agent_queue_depth{agent_id='%s', queue_type='messages'}" % self.agent_id,
            unit="count",
            threshold_direction="lower"
        )
        
        # === RESOURCE UTILIZATION SLIs ===
        self.slis["cpu_utilization"] = SLI(
            name="cpu_utilization",
            description="CPU utilization percentage",
            sli_type=SLIType.RESOURCE_UTILIZATION,
            query="rate(process_cpu_seconds_total{job='a2a_agents', instance=~'.*%s.*'}[5m]) * 100" % self.agent_id,
            unit="percentage",
            threshold_direction="lower"
        )
        
        self.slis["memory_utilization"] = SLI(
            name="memory_utilization",
            description="Memory utilization percentage",
            sli_type=SLIType.RESOURCE_UTILIZATION,
            query="(process_resident_memory_bytes{job='a2a_agents', instance=~'.*%s.*'} / process_virtual_memory_max_bytes{job='a2a_agents', instance=~'.*%s.*'}) * 100" % (self.agent_id, self.agent_id),
            unit="percentage",
            threshold_direction="lower"
        )
        
        # === A2A SPECIFIC SLIs ===
        self.slis["message_delivery_rate"] = SLI(
            name="message_delivery_rate",
            description="A2A message delivery success rate",
            sli_type=SLIType.MESSAGE_DELIVERY_RATE,
            query="sum(rate(a2a_messages_total{agent_id='%s', status='delivered'}[5m])) / sum(rate(a2a_messages_total{agent_id='%s'}[5m]))" % (self.agent_id, self.agent_id),
            unit="ratio",
            good_total_ratio=True,
            threshold_direction="upper"
        )
        
        self.slis["trust_verification_rate"] = SLI(
            name="trust_verification_rate",
            description="Trust verification success rate",
            sli_type=SLIType.SUCCESS_RATE,
            query="sum(rate(a2a_trust_verifications_total{agent_id='%s', status='verified'}[5m])) / sum(rate(a2a_trust_verifications_total{agent_id='%s'}[5m]))" % (self.agent_id, self.agent_id),
            unit="ratio",
            good_total_ratio=True,
            threshold_direction="upper"
        )
        
        # === DEFAULT SLOs ===
        
        # High-availability SLO (99.9%)
        self.slos["availability_slo"] = SLO(
            name="availability_slo",
            description="Agent must be available 99.9% of the time",
            sli=self.slis["service_availability"],
            target=0.999,
            window_duration=timedelta(days=30),
            alert_threshold=0.995,
            burn_rate_thresholds={
                "fast": 14.4,    # 1 hour to exhaust monthly budget
                "slow": 6.0      # 6 hours to exhaust monthly budget
            },
            tags={"tier": "critical", "team": "platform"}
        )
        
        # Latency SLO (95% of requests < 500ms)
        self.slos["latency_slo"] = SLO(
            name="latency_slo",
            description="95% of requests must complete within 500ms",
            sli=self.slis["response_latency_p95"],
            target=0.5,  # 500ms
            window_duration=timedelta(hours=24),
            alert_threshold=0.6,  # Alert at 600ms
            burn_rate_thresholds={
                "fast": 8.0,
                "slow": 2.0
            },
            tags={"tier": "performance", "team": "platform"}
        )
        
        # Error rate SLO (< 0.1% errors)
        self.slos["error_rate_slo"] = SLO(
            name="error_rate_slo",
            description="Error rate must be less than 0.1%",
            sli=self.slis["error_rate"],
            target=0.001,  # 0.1%
            window_duration=timedelta(hours=24),
            alert_threshold=0.005,  # Alert at 0.5%
            burn_rate_thresholds={
                "fast": 10.0,
                "slow": 2.5
            },
            tags={"tier": "reliability", "team": "platform"}
        )
        
        # Task completion SLO (> 99% success rate)
        self.slos["task_completion_slo"] = SLO(
            name="task_completion_slo",
            description="Task completion rate must be above 99%",
            sli=self.slis["task_failure_rate"],
            target=0.01,  # 1% failure rate (99% success)
            window_duration=timedelta(hours=24),
            alert_threshold=0.05,  # Alert at 5% failure rate
            burn_rate_thresholds={
                "fast": 5.0,
                "slow": 1.0
            },
            tags={"tier": "business", "team": "agents"}
        )
        
        # Queue depth SLO (< 100 pending tasks)
        self.slos["queue_depth_slo"] = SLO(
            name="queue_depth_slo",
            description="Task queue depth must be less than 100",
            sli=self.slis["task_queue_depth"],
            target=100,
            window_duration=timedelta(hours=1),
            alert_threshold=150,
            burn_rate_thresholds={
                "fast": 2.0,
                "slow": 1.0
            },
            tags={"tier": "capacity", "team": "platform"}
        )
        
        # Resource utilization SLO (< 80% CPU)
        self.slos["cpu_utilization_slo"] = SLO(
            name="cpu_utilization_slo",
            description="CPU utilization must be less than 80%",
            sli=self.slis["cpu_utilization"],
            target=80.0,
            window_duration=timedelta(hours=1),
            alert_threshold=90.0,
            burn_rate_thresholds={
                "fast": 1.5,
                "slow": 1.0
            },
            tags={"tier": "resource", "team": "platform"}
        )
        
        # A2A message delivery SLO (> 99.5%)
        self.slos["message_delivery_slo"] = SLO(
            name="message_delivery_slo",
            description="A2A message delivery rate must be above 99.5%",
            sli=self.slis["message_delivery_rate"],
            target=0.995,
            window_duration=timedelta(hours=24),
            alert_threshold=0.990,
            burn_rate_thresholds={
                "fast": 10.0,
                "slow": 2.0
            },
            tags={"tier": "communication", "team": "agents"}
        )
    
    async def initialize(self):
        """Initialize the SLO framework"""
        await self.prometheus_client.initialize()
        await self.redis_client.initialize()
        
        # Start monitoring loops
        self.running = True
        self.monitoring_tasks = [
            asyncio.create_task(self._slo_evaluation_loop()),
            asyncio.create_task(self._burn_rate_monitoring_loop()),
            asyncio.create_task(self._alerting_loop()),
            asyncio.create_task(self._metrics_collection_loop())
        ]
        
        logger.info(f"SLO framework initialized for agent {self.agent_id}")
    
    async def shutdown(self):
        """Shutdown the SLO framework"""
        self.running = False
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        await self.prometheus_client.close()
        await self.redis_client.close()
        
        logger.info(f"SLO framework shut down for agent {self.agent_id}")
    
    def register_sli(self, sli: SLI):
        """Register a custom SLI"""
        self.slis[sli.name] = sli
        logger.info(f"Registered SLI: {sli.name}")
    
    def register_slo(self, slo: SLO):
        """Register a custom SLO"""
        self.slos[slo.name] = slo
        logger.info(f"Registered SLO: {slo.name}")
    
    def register_alert_handler(self, level: AlertLevel, handler: Callable):
        """Register an alert handler for a specific level"""
        self.alert_handlers[level].append(handler)
        logger.info(f"Registered alert handler for level: {level}")
    
    @trace_async("evaluate_slo")
    async def evaluate_slo(self, slo_name: str) -> SLOResult:
        """Evaluate a specific SLO"""
        if slo_name not in self.slos:
            raise ValueError(f"SLO {slo_name} not found")
        
        slo = self.slos[slo_name]
        
        add_span_attributes({
            "slo.name": slo_name,
            "slo.target": slo.target,
            "agent.id": self.agent_id
        })
        
        # Query current SLI value
        current_value = await self._query_sli_value(slo.sli)
        
        # Calculate compliance
        if slo.sli.good_total_ratio or slo.sli.threshold_direction == "upper":
            compliance_percentage = (current_value / slo.target) * 100
            is_compliant = current_value >= slo.target
        else:
            compliance_percentage = (slo.target / current_value) * 100 if current_value > 0 else 100
            is_compliant = current_value <= slo.target
        
        # Calculate error budget
        error_budget_remaining = self._calculate_error_budget(slo, current_value)
        
        # Calculate burn rate
        burn_rate = await self._calculate_burn_rate(slo, current_value)
        
        # Determine status
        status = self._determine_slo_status(slo, current_value, burn_rate)
        
        # Calculate time to exhaustion
        time_to_exhaustion = self._calculate_time_to_exhaustion(error_budget_remaining, burn_rate)
        
        # Generate alerts
        alerts = await self._generate_slo_alerts(slo, current_value, burn_rate, status)
        
        result = SLOResult(
            slo_name=slo_name,
            current_value=current_value,
            target_value=slo.target,
            compliance_percentage=compliance_percentage,
            status=status,
            error_budget_remaining=error_budget_remaining,
            burn_rate=burn_rate,
            time_to_exhaustion=time_to_exhaustion,
            alerts=alerts
        )
        
        # Store result
        self.evaluation_results[slo_name] = result
        
        # Store in Redis for historical tracking
        await self._store_slo_result(result)
        
        return result
    
    async def _query_sli_value(self, sli: SLI) -> float:
        """Query the current value of an SLI"""
        try:
            # Query Prometheus
            result = await self.prometheus_client.query(sli.query)
            
            if result and len(result) > 0:
                value = float(result[0].get('value', [0, 0])[1])
                return value
            else:
                logger.warning(f"No data returned for SLI {sli.name}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Failed to query SLI {sli.name}: {e}")
            return 0.0
    
    def _calculate_error_budget(self, slo: SLO, current_value: float) -> float:
        """Calculate remaining error budget"""
        if slo.sli.good_total_ratio or slo.sli.threshold_direction == "upper":
            # For availability/success rates
            error_budget_total = 1.0 - slo.target
            error_budget_consumed = max(0, slo.target - current_value)
            return max(0, error_budget_total - error_budget_consumed)
        else:
            # For latency/error rates
            if current_value <= slo.target:
                return 1.0  # Full budget remaining
            else:
                # Simplified calculation - could be more sophisticated
                excess = current_value - slo.target
                max_excess = slo.target * 0.5  # Allow 50% excess before budget exhausted
                return max(0, 1.0 - (excess / max_excess))
    
    async def _calculate_burn_rate(self, slo: SLO, current_value: float) -> float:
        """Calculate error budget burn rate"""
        try:
            # Query historical data to calculate burn rate
            historical_query = slo.sli.query.replace('[5m]', '[1h]')
            historical_result = await self.prometheus_client.query_range(
                historical_query,
                start=datetime.utcnow() - timedelta(hours=1),
                end=datetime.utcnow(),
                step='5m'
            )
            
            if not historical_result:
                return 1.0  # Default burn rate
            
            # Calculate average rate over time window
            values = [float(point[1]) for point in historical_result[0].get('values', [])]
            if not values:
                return 1.0
            
            avg_value = statistics.mean(values)
            
            # Calculate burn rate relative to target
            if slo.sli.good_total_ratio or slo.sli.threshold_direction == "upper":
                burn_rate = max(0, slo.target - avg_value) / (1.0 - slo.target)
            else:
                if avg_value <= slo.target:
                    burn_rate = 0.0
                else:
                    burn_rate = (avg_value - slo.target) / slo.target
            
            return min(burn_rate, 100.0)  # Cap at 100x normal rate
            
        except Exception as e:
            logger.error(f"Failed to calculate burn rate for {slo.name}: {e}")
            return 1.0
    
    def _determine_slo_status(self, slo: SLO, current_value: float, burn_rate: float) -> SLOStatus:
        """Determine SLO status based on current value and burn rate"""
        
        # Check if SLO is breached
        if slo.sli.good_total_ratio or slo.sli.threshold_direction == "upper":
            breached = current_value < slo.target
            warning = current_value < slo.alert_threshold
        else:
            breached = current_value > slo.target
            warning = current_value > slo.alert_threshold
        
        if breached:
            return SLOStatus.BREACHED
        
        # Check burn rate thresholds
        fast_burn_threshold = slo.burn_rate_thresholds.get("fast", 10.0)
        slow_burn_threshold = slo.burn_rate_thresholds.get("slow", 2.0)
        
        if burn_rate >= fast_burn_threshold:
            return SLOStatus.CRITICAL
        elif burn_rate >= slow_burn_threshold or warning:
            return SLOStatus.WARNING
        else:
            return SLOStatus.HEALTHY
    
    def _calculate_time_to_exhaustion(self, error_budget_remaining: float, burn_rate: float) -> Optional[timedelta]:
        """Calculate time until error budget exhaustion"""
        if burn_rate <= 0 or error_budget_remaining <= 0:
            return None
        
        # Simple calculation - could be more sophisticated
        hours_remaining = (error_budget_remaining / burn_rate) * 24  # Assume 24h baseline
        
        if hours_remaining > 8760:  # More than a year
            return None
        
        return timedelta(hours=hours_remaining)
    
    async def _generate_slo_alerts(self, slo: SLO, current_value: float, burn_rate: float, status: SLOStatus) -> List[Dict[str, Any]]:
        """Generate alerts based on SLO status"""
        alerts = []
        
        if status == SLOStatus.BREACHED:
            alerts.append({
                "level": AlertLevel.CRITICAL,
                "title": f"SLO Breached: {slo.name}",
                "description": f"SLO {slo.name} is breached. Current: {current_value:.4f}, Target: {slo.target:.4f}",
                "timestamp": datetime.utcnow().isoformat(),
                "tags": slo.tags
            })
        
        elif status == SLOStatus.CRITICAL:
            alerts.append({
                "level": AlertLevel.CRITICAL,
                "title": f"High Burn Rate: {slo.name}",
                "description": f"SLO {slo.name} has high burn rate: {burn_rate:.2f}x",
                "timestamp": datetime.utcnow().isoformat(),
                "tags": slo.tags
            })
        
        elif status == SLOStatus.WARNING:
            alerts.append({
                "level": AlertLevel.WARNING,
                "title": f"SLO Warning: {slo.name}",
                "description": f"SLO {slo.name} approaching threshold. Current: {current_value:.4f}",
                "timestamp": datetime.utcnow().isoformat(),
                "tags": slo.tags
            })
        
        return alerts
    
    async def _store_slo_result(self, result: SLOResult):
        """Store SLO result in Redis for historical tracking"""
        try:
            key = f"slo_results:{self.agent_id}:{result.slo_name}"
            data = {
                "current_value": result.current_value,
                "target_value": result.target_value,
                "compliance_percentage": result.compliance_percentage,
                "status": result.status.value,
                "error_budget_remaining": result.error_budget_remaining,
                "burn_rate": result.burn_rate,
                "timestamp": result.timestamp.isoformat()
            }
            
            # Store with 7-day expiration
            await self.redis_client.setex(
                key,
                timedelta(days=7),
                json.dumps(data)
            )
            
            # Also add to time series for trending
            ts_key = f"slo_timeseries:{self.agent_id}:{result.slo_name}"
            await self.redis_client.zadd(
                ts_key,
                {json.dumps(data): time.time()}
            )
            
            # Keep only last 24 hours of time series data
            cutoff = time.time() - (24 * 60 * 60)
            await self.redis_client.zremrangebyscore(ts_key, 0, cutoff)
            
        except Exception as e:
            logger.error(f"Failed to store SLO result: {e}")
    
    async def _slo_evaluation_loop(self):
        """Background loop to evaluate all SLOs"""
        while self.running:
            try:
                for slo_name in self.slos:
                    await self.evaluate_slo(slo_name)
                
                await asyncio.sleep(60)  # Evaluate every minute
                
            except Exception as e:
                logger.error(f"SLO evaluation loop error: {e}")
                await asyncio.sleep(60)
    
    async def _burn_rate_monitoring_loop(self):
        """Background loop to monitor burn rates"""
        while self.running:
            try:
                # Check for fast burn rates every 5 minutes
                for slo_name, result in self.evaluation_results.items():
                    if result.burn_rate >= 10.0:  # Fast burn rate
                        await self._trigger_fast_burn_alert(slo_name, result)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Burn rate monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _alerting_loop(self):
        """Background loop to process alerts"""
        while self.running:
            try:
                for slo_name, result in self.evaluation_results.items():
                    for alert in result.alerts:
                        await self._dispatch_alert(alert)
                
                await asyncio.sleep(30)  # Process alerts every 30 seconds
                
            except Exception as e:
                logger.error(f"Alerting loop error: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_collection_loop(self):
        """Background loop to collect SLO framework metrics"""
        while self.running:
            try:
                # Collect framework metrics
                metrics = {
                    "slos_total": len(self.slos),
                    "slis_total": len(self.slis),
                    "healthy_slos": len([r for r in self.evaluation_results.values() if r.status == SLOStatus.HEALTHY]),
                    "warning_slos": len([r for r in self.evaluation_results.values() if r.status == SLOStatus.WARNING]),
                    "critical_slos": len([r for r in self.evaluation_results.values() if r.status == SLOStatus.CRITICAL]),
                    "breached_slos": len([r for r in self.evaluation_results.values() if r.status == SLOStatus.BREACHED])
                }
                
                # Send to Prometheus
                for metric, value in metrics.items():
                    await self.prometheus_client.gauge(
                        f"a2a_slo_framework_{metric}",
                        value,
                        {"agent_id": self.agent_id}
                    )
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _trigger_fast_burn_alert(self, slo_name: str, result: SLOResult):
        """Trigger immediate alert for fast burn rate"""
        alert = {
            "level": AlertLevel.EMERGENCY,
            "title": f"EMERGENCY: Fast Burn Rate - {slo_name}",
            "description": f"SLO {slo_name} has extremely high burn rate: {result.burn_rate:.2f}x. Error budget will be exhausted quickly!",
            "timestamp": datetime.utcnow().isoformat(),
            "burn_rate": result.burn_rate,
            "time_to_exhaustion": str(result.time_to_exhaustion) if result.time_to_exhaustion else "Unknown"
        }
        
        await self._dispatch_alert(alert)
    
    async def _dispatch_alert(self, alert: Dict[str, Any]):
        """Dispatch alert to registered handlers"""
        try:
            level = AlertLevel(alert["level"])
            handlers = self.alert_handlers.get(level, [])
            
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
            
            # Default logging if no handlers
            if not handlers:
                logger.log(
                    logging.CRITICAL if level == AlertLevel.EMERGENCY else logging.WARNING,
                    f"SLO Alert [{level}]: {alert['title']} - {alert['description']}"
                )
        
        except Exception as e:
            logger.error(f"Alert dispatch failed: {e}")
    
    async def get_slo_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive SLO dashboard data"""
        dashboard_data = {
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_slos": len(self.slos),
                "healthy": 0,
                "warning": 0,
                "critical": 0,
                "breached": 0
            },
            "slos": {}
        }
        
        for slo_name, result in self.evaluation_results.items():
            dashboard_data["summary"][result.status.value] += 1
            
            dashboard_data["slos"][slo_name] = {
                "status": result.status.value,
                "current_value": result.current_value,
                "target_value": result.target_value,
                "compliance_percentage": result.compliance_percentage,
                "error_budget_remaining": result.error_budget_remaining,
                "burn_rate": result.burn_rate,
                "time_to_exhaustion": str(result.time_to_exhaustion) if result.time_to_exhaustion else None,
                "alerts": len(result.alerts)
            }
        
        return dashboard_data
    
    async def get_error_budget_report(self) -> Dict[str, Any]:
        """Generate error budget report"""
        report = {
            "agent_id": self.agent_id,
            "report_timestamp": datetime.utcnow().isoformat(),
            "error_budgets": {}
        }
        
        for slo_name, result in self.evaluation_results.items():
            slo = self.slos[slo_name]
            
            report["error_budgets"][slo_name] = {
                "remaining_percentage": result.error_budget_remaining * 100,
                "burn_rate": result.burn_rate,
                "time_to_exhaustion": str(result.time_to_exhaustion) if result.time_to_exhaustion else "N/A",
                "window_duration": str(slo.window_duration),
                "target": slo.target,
                "current_value": result.current_value,
                "status": result.status.value
            }
        
        return report


# Global SLO framework instance per agent
_slo_frameworks: Dict[str, A2ASLOFramework] = {}


async def initialize_slo_framework(
    agent_id: str,
    prometheus_config: Dict[str, Any] = None,
    redis_config: RedisConfig = None
) -> A2ASLOFramework:
    """Initialize SLO framework for an agent"""
    global _slo_frameworks
    
    if agent_id in _slo_frameworks:
        return _slo_frameworks[agent_id]
    
    framework = A2ASLOFramework(agent_id, prometheus_config, redis_config)
    await framework.initialize()
    
    _slo_frameworks[agent_id] = framework
    return framework


async def get_slo_framework(agent_id: str) -> Optional[A2ASLOFramework]:
    """Get existing SLO framework for an agent"""
    return _slo_frameworks.get(agent_id)


async def shutdown_slo_framework(agent_id: str):
    """Shutdown SLO framework for an agent"""
    global _slo_frameworks
    
    if agent_id in _slo_frameworks:
        await _slo_frameworks[agent_id].shutdown()
        del _slo_frameworks[agent_id]