"""
Prometheus monitoring configuration for A2A agents.
Provides comprehensive metrics collection and monitoring.
"""
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    CollectorRegistry, generate_latest, push_to_gateway,
    start_http_server
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily
import psutil

logger = logging.getLogger(__name__)


class A2AMetricsCollector:
    """
    Custom metrics collector for A2A agents.
    """
    
    def __init__(self, agent_id: str, agent_type: str):
        """
        Initialize metrics collector.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.registry = CollectorRegistry()
        
        # Agent info
        self.agent_info = Info(
            'a2a_agent_info',
            'Agent information',
            ['agent_id', 'agent_type'],
            registry=self.registry
        )
        self.agent_info.labels(
            agent_id=agent_id,
            agent_type=agent_type
        ).info({
            'version': '1.0.0',
            'started_at': datetime.now().isoformat()
        })
        
        # Request metrics
        self.requests_total = Counter(
            'a2a_requests_total',
            'Total number of requests',
            ['agent_id', 'method', 'endpoint'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'a2a_request_duration_seconds',
            'Request duration in seconds',
            ['agent_id', 'method', 'endpoint'],
            registry=self.registry,
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        self.request_errors = Counter(
            'a2a_request_errors_total',
            'Total number of request errors',
            ['agent_id', 'method', 'endpoint', 'error_type'],
            registry=self.registry
        )
        
        # Task metrics
        self.tasks_total = Counter(
            'a2a_tasks_total',
            'Total number of tasks processed',
            ['agent_id', 'task_type', 'status'],
            registry=self.registry
        )
        
        self.task_duration = Histogram(
            'a2a_task_duration_seconds',
            'Task processing duration',
            ['agent_id', 'task_type'],
            registry=self.registry,
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0)
        )
        
        self.tasks_active = Gauge(
            'a2a_tasks_active',
            'Number of active tasks',
            ['agent_id', 'task_type'],
            registry=self.registry
        )
        
        # Blockchain metrics
        self.blockchain_transactions = Counter(
            'a2a_blockchain_transactions_total',
            'Total blockchain transactions',
            ['agent_id', 'transaction_type', 'status'],
            registry=self.registry
        )
        
        self.blockchain_gas_used = Counter(
            'a2a_blockchain_gas_used_total',
            'Total gas used',
            ['agent_id'],
            registry=self.registry
        )
        
        self.trust_score = Gauge(
            'a2a_trust_score',
            'Current trust score',
            ['agent_id', 'peer_agent'],
            registry=self.registry
        )
        
        # Performance metrics
        self.cpu_usage = Gauge(
            'a2a_cpu_usage_percent',
            'CPU usage percentage',
            ['agent_id'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'a2a_memory_usage_bytes',
            'Memory usage in bytes',
            ['agent_id'],
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'a2a_queue_size',
            'Current queue size',
            ['agent_id', 'queue_name'],
            registry=self.registry
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'a2a_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['agent_id', 'circuit_name'],
            registry=self.registry
        )
        
        self.circuit_breaker_failures = Counter(
            'a2a_circuit_breaker_failures_total',
            'Total circuit breaker failures',
            ['agent_id', 'circuit_name'],
            registry=self.registry
        )
        
        # Data processing metrics
        self.data_processed = Counter(
            'a2a_data_processed_bytes_total',
            'Total bytes of data processed',
            ['agent_id', 'operation'],
            registry=self.registry
        )
        
        self.validation_results = Counter(
            'a2a_validation_results_total',
            'Validation results',
            ['agent_id', 'validation_type', 'result'],
            registry=self.registry
        )
        
        # Quality metrics
        self.quality_score = Gauge(
            'a2a_quality_score',
            'Quality score (0-100)',
            ['agent_id', 'metric_type'],
            registry=self.registry
        )
        
        self.accuracy_rate = Gauge(
            'a2a_accuracy_rate',
            'Accuracy rate (0-1)',
            ['agent_id', 'operation'],
            registry=self.registry
        )
        
        # Health check metrics
        self.health_check_status = Gauge(
            'a2a_health_check_status',
            'Health check status (1=healthy, 0=unhealthy)',
            ['agent_id', 'check_type'],
            registry=self.registry
        )
        
        self.last_health_check = Gauge(
            'a2a_last_health_check_timestamp',
            'Last health check timestamp',
            ['agent_id'],
            registry=self.registry
        )
        
        # Custom metrics storage
        self.custom_metrics = {}
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        duration: float,
        status: str,
        error_type: Optional[str] = None
    ):
        """Record HTTP request metrics."""
        self.requests_total.labels(
            agent_id=self.agent_id,
            method=method,
            endpoint=endpoint
        ).inc()
        
        self.request_duration.labels(
            agent_id=self.agent_id,
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        if error_type:
            self.request_errors.labels(
                agent_id=self.agent_id,
                method=method,
                endpoint=endpoint,
                error_type=error_type
            ).inc()
    
    def record_task(
        self,
        task_type: str,
        status: str,
        duration: Optional[float] = None
    ):
        """Record task processing metrics."""
        self.tasks_total.labels(
            agent_id=self.agent_id,
            task_type=task_type,
            status=status
        ).inc()
        
        if duration is not None:
            self.task_duration.labels(
                agent_id=self.agent_id,
                task_type=task_type
            ).observe(duration)
    
    def set_active_tasks(self, task_type: str, count: int):
        """Set number of active tasks."""
        self.tasks_active.labels(
            agent_id=self.agent_id,
            task_type=task_type
        ).set(count)
    
    def record_blockchain_transaction(
        self,
        transaction_type: str,
        status: str,
        gas_used: Optional[int] = None
    ):
        """Record blockchain transaction metrics."""
        self.blockchain_transactions.labels(
            agent_id=self.agent_id,
            transaction_type=transaction_type,
            status=status
        ).inc()
        
        if gas_used:
            self.blockchain_gas_used.labels(
                agent_id=self.agent_id
            ).inc(gas_used)
    
    def set_trust_score(self, peer_agent: str, score: float):
        """Set trust score for peer agent."""
        self.trust_score.labels(
            agent_id=self.agent_id,
            peer_agent=peer_agent
        ).set(score)
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage.labels(agent_id=self.agent_id).set(cpu_percent)
        
        # Memory usage
        memory = psutil.Process().memory_info()
        self.memory_usage.labels(agent_id=self.agent_id).set(memory.rss)
    
    def set_queue_size(self, queue_name: str, size: int):
        """Set queue size metric."""
        self.queue_size.labels(
            agent_id=self.agent_id,
            queue_name=queue_name
        ).set(size)
    
    def update_circuit_breaker(
        self,
        circuit_name: str,
        state: str,
        failures: Optional[int] = None
    ):
        """Update circuit breaker metrics."""
        state_map = {
            'closed': 0,
            'open': 1,
            'half_open': 2
        }
        self.circuit_breaker_state.labels(
            agent_id=self.agent_id,
            circuit_name=circuit_name
        ).set(state_map.get(state, -1))
        
        if failures is not None:
            self.circuit_breaker_failures.labels(
                agent_id=self.agent_id,
                circuit_name=circuit_name
            ).inc(failures)
    
    def record_data_processed(self, operation: str, bytes_count: int):
        """Record data processing metrics."""
        self.data_processed.labels(
            agent_id=self.agent_id,
            operation=operation
        ).inc(bytes_count)
    
    def record_validation(
        self,
        validation_type: str,
        result: str
    ):
        """Record validation result."""
        self.validation_results.labels(
            agent_id=self.agent_id,
            validation_type=validation_type,
            result=result
        ).inc()
    
    def set_quality_score(self, metric_type: str, score: float):
        """Set quality score metric."""
        self.quality_score.labels(
            agent_id=self.agent_id,
            metric_type=metric_type
        ).set(score)
    
    def set_accuracy_rate(self, operation: str, rate: float):
        """Set accuracy rate metric."""
        self.accuracy_rate.labels(
            agent_id=self.agent_id,
            operation=operation
        ).set(rate)
    
    def update_health_check(self, check_type: str, is_healthy: bool):
        """Update health check status."""
        self.health_check_status.labels(
            agent_id=self.agent_id,
            check_type=check_type
        ).set(1 if is_healthy else 0)
        
        self.last_health_check.labels(
            agent_id=self.agent_id
        ).set(time.time())
    
    def create_custom_metric(
        self,
        metric_name: str,
        metric_type: str,
        description: str,
        labels: List[str]
    ):
        """Create a custom metric."""
        if metric_name in self.custom_metrics:
            return self.custom_metrics[metric_name]
        
        metric_class = {
            'counter': Counter,
            'gauge': Gauge,
            'histogram': Histogram,
            'summary': Summary
        }.get(metric_type, Gauge)
        
        metric = metric_class(
            f'a2a_custom_{metric_name}',
            description,
            ['agent_id'] + labels,
            registry=self.registry
        )
        
        self.custom_metrics[metric_name] = metric
        return metric
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        # Update system metrics before export
        self.update_system_metrics()
        
        return generate_latest(self.registry)
    
    def push_metrics(self, gateway_url: str, job_name: str):
        """Push metrics to Prometheus push gateway."""
        try:
            push_to_gateway(
                gateway_url,
                job=job_name,
                registry=self.registry,
                grouping_key={'agent_id': self.agent_id}
            )
            logger.debug(f"Pushed metrics to {gateway_url}")
        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")


class PrometheusExporter:
    """
    Prometheus metrics exporter for A2A agents.
    """
    
    def __init__(
        self,
        port: int = 9090,
        push_gateway_url: Optional[str] = None
    ):
        """
        Initialize Prometheus exporter.
        
        Args:
            port: Port to expose metrics on
            push_gateway_url: Optional push gateway URL
        """
        self.port = port
        self.push_gateway_url = push_gateway_url
        self.collectors = {}
        self._server_started = False
    
    def register_agent(
        self,
        agent_id: str,
        agent_type: str
    ) -> A2AMetricsCollector:
        """Register an agent for metrics collection."""
        if agent_id not in self.collectors:
            self.collectors[agent_id] = A2AMetricsCollector(
                agent_id,
                agent_type
            )
        return self.collectors[agent_id]
    
    def start_metrics_server(self):
        """Start HTTP server for metrics endpoint."""
        if not self._server_started:
            start_http_server(self.port)
            self._server_started = True
            logger.info(f"Prometheus metrics server started on port {self.port}")
    
    def get_agent_collector(self, agent_id: str) -> Optional[A2AMetricsCollector]:
        """Get metrics collector for specific agent."""
        return self.collectors.get(agent_id)
    
    def push_all_metrics(self, job_name: str = "a2a_agents"):
        """Push all agent metrics to push gateway."""
        if not self.push_gateway_url:
            logger.warning("No push gateway URL configured")
            return
        
        for agent_id, collector in self.collectors.items():
            collector.push_metrics(self.push_gateway_url, job_name)


# Global exporter instance
_exporter = None


def get_prometheus_exporter(
    port: int = 9090,
    push_gateway_url: Optional[str] = None
) -> PrometheusExporter:
    """Get or create global Prometheus exporter."""
    global _exporter
    if _exporter is None:
        _exporter = PrometheusExporter(
            port=port or int(os.getenv("PROMETHEUS_PORT", "9090")),
            push_gateway_url=push_gateway_url or os.getenv("PROMETHEUS_PUSH_GATEWAY")
        )
        _exporter.start_metrics_server()
    return _exporter


def create_agent_metrics(
    agent_id: str,
    agent_type: str
) -> A2AMetricsCollector:
    """Create metrics collector for an agent."""
    exporter = get_prometheus_exporter()
    return exporter.register_agent(agent_id, agent_type)