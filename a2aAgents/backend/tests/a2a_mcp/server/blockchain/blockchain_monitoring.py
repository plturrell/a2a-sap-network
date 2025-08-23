#!/usr/bin/env python3
"""
Blockchain Monitoring and Alerting System

This module provides comprehensive monitoring and alerting for blockchain operations
including:
- Transaction monitoring
- Gas usage tracking
- Error rate monitoring
- Performance metrics
- Health checks
- Alert notifications
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import os
import sys
from collections import deque, defaultdict

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to monitor"""
    TRANSACTION_COUNT = "transaction_count"
    TRANSACTION_SUCCESS_RATE = "transaction_success_rate"
    GAS_USAGE = "gas_usage"
    MESSAGE_LATENCY = "message_latency"
    AGENT_AVAILABILITY = "agent_availability"
    REPUTATION_CHANGES = "reputation_changes"
    ERROR_RATE = "error_rate"
    NETWORK_HEALTH = "network_health"


@dataclass
class Alert:
    """Alert data structure"""
    severity: AlertSeverity
    metric_type: MetricType
    message: str
    value: Any
    threshold: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSnapshot:
    """Snapshot of a metric at a point in time"""
    metric_type: MetricType
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BlockchainMonitor:
    """Main blockchain monitoring system"""
    
    def __init__(self, 
                 check_interval: int = 30,  # seconds
                 history_size: int = 1000,
                 alert_handlers: Optional[List[Callable]] = None):
        """
        Initialize blockchain monitor
        
        Args:
            check_interval: How often to check metrics (seconds)
            history_size: Number of historical metrics to keep
            alert_handlers: List of functions to call when alerts trigger
        """
        self.check_interval = check_interval
        self.history_size = history_size
        self.alert_handlers = alert_handlers or []
        
        # Metric storage
        self.metrics: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=history_size)
            for metric_type in MetricType
        }
        
        # Alert thresholds
        self.thresholds = {
            MetricType.TRANSACTION_SUCCESS_RATE: 0.95,  # Alert if below 95%
            MetricType.ERROR_RATE: 0.05,  # Alert if above 5%
            MetricType.MESSAGE_LATENCY: 5000,  # Alert if above 5 seconds
            MetricType.GAS_USAGE: 1000000,  # Alert if single tx uses > 1M gas
            MetricType.AGENT_AVAILABILITY: 0.90,  # Alert if below 90%
        }
        
        # Monitoring state
        self.monitoring = False
        self.monitor_task = None
        
        # Transaction tracking
        self.pending_transactions: Dict[str, Dict[str, Any]] = {}
        self.completed_transactions: deque = deque(maxlen=1000)
        self.failed_transactions: deque = deque(maxlen=100)
        
        # Agent tracking
        self.agent_status: Dict[str, Dict[str, Any]] = {}
        self.agent_last_seen: Dict[str, datetime] = {}
        
        # Alert history
        self.alert_history: deque = deque(maxlen=1000)
        
        # Performance metrics
        self.performance_stats = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
            "total_gas_used": 0,
            "total_messages_routed": 0,
            "average_latency": 0
        }
    
    async def start(self):
        """Start monitoring"""
        if self.monitoring:
            logger.warning("Monitor already running")
            return
            
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Blockchain monitor started")
    
    async def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_task:
            await self.monitor_task
        logger.info("Blockchain monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Collect metrics
                await self._collect_metrics()
                
                # Check thresholds and generate alerts
                await self._check_thresholds()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _collect_metrics(self):
        """Collect current metrics"""
        # Transaction metrics
        await self._collect_transaction_metrics()
        
        # Agent availability metrics
        await self._collect_agent_metrics()
        
        # Network health metrics
        await self._collect_network_metrics()
        
        # Performance metrics
        await self._collect_performance_metrics()
    
    async def _collect_transaction_metrics(self):
        """Collect transaction-related metrics"""
        # Calculate success rate
        if self.performance_stats["total_transactions"] > 0:
            success_rate = (
                self.performance_stats["successful_transactions"] / 
                self.performance_stats["total_transactions"]
            )
            self._record_metric(
                MetricType.TRANSACTION_SUCCESS_RATE,
                success_rate,
                {"period": "cumulative"}
            )
        
        # Calculate error rate
        if self.performance_stats["total_transactions"] > 0:
            error_rate = (
                self.performance_stats["failed_transactions"] / 
                self.performance_stats["total_transactions"]
            )
            self._record_metric(
                MetricType.ERROR_RATE,
                error_rate,
                {"period": "cumulative"}
            )
        
        # Record transaction count
        self._record_metric(
            MetricType.TRANSACTION_COUNT,
            self.performance_stats["total_transactions"],
            {"type": "cumulative"}
        )
    
    async def _collect_agent_metrics(self):
        """Collect agent-related metrics"""
        now = datetime.now()
        active_agents = 0
        total_agents = len(self.agent_status)
        
        for agent_id, last_seen in self.agent_last_seen.items():
            # Consider agent active if seen in last 5 minutes
            if (now - last_seen).total_seconds() < 300:
                active_agents += 1
        
        if total_agents > 0:
            availability = active_agents / total_agents
            self._record_metric(
                MetricType.AGENT_AVAILABILITY,
                availability,
                {"active": active_agents, "total": total_agents}
            )
    
    async def _collect_network_metrics(self):
        """Collect network health metrics"""
        # This would connect to actual blockchain network
        # For now, simulate network health
        network_health = 1.0  # 100% healthy
        
        self._record_metric(
            MetricType.NETWORK_HEALTH,
            network_health,
            {"connected": True, "peers": 5}
        )
    
    async def _collect_performance_metrics(self):
        """Collect performance metrics"""
        # Average message latency
        if self.performance_stats.get("average_latency", 0) > 0:
            self._record_metric(
                MetricType.MESSAGE_LATENCY,
                self.performance_stats["average_latency"],
                {"unit": "milliseconds"}
            )
        
        # Gas usage
        if self.performance_stats.get("total_gas_used", 0) > 0:
            self._record_metric(
                MetricType.GAS_USAGE,
                self.performance_stats["total_gas_used"],
                {"unit": "wei"}
            )
    
    def _record_metric(self, metric_type: MetricType, value: Any, metadata: Dict[str, Any] = None):
        """Record a metric snapshot"""
        snapshot = MetricSnapshot(
            metric_type=metric_type,
            value=value,
            metadata=metadata or {}
        )
        self.metrics[metric_type].append(snapshot)
    
    async def _check_thresholds(self):
        """Check metrics against thresholds and generate alerts"""
        for metric_type, threshold in self.thresholds.items():
            # Get latest metric value
            if metric_type in self.metrics and self.metrics[metric_type]:
                latest = self.metrics[metric_type][-1]
                
                # Check threshold based on metric type
                should_alert = False
                
                if metric_type in [MetricType.TRANSACTION_SUCCESS_RATE, MetricType.AGENT_AVAILABILITY]:
                    # Alert if below threshold
                    should_alert = latest.value < threshold
                elif metric_type in [MetricType.ERROR_RATE, MetricType.MESSAGE_LATENCY, MetricType.GAS_USAGE]:
                    # Alert if above threshold
                    should_alert = latest.value > threshold
                
                if should_alert:
                    await self._generate_alert(
                        severity=AlertSeverity.WARNING,
                        metric_type=metric_type,
                        message=f"{metric_type.value} threshold exceeded",
                        value=latest.value,
                        threshold=threshold
                    )
    
    async def _generate_alert(self, severity: AlertSeverity, metric_type: MetricType,
                            message: str, value: Any, threshold: Any):
        """Generate and handle an alert"""
        alert = Alert(
            severity=severity,
            metric_type=metric_type,
            message=message,
            value=value,
            threshold=threshold
        )
        
        # Store alert
        self.alert_history.append(alert)
        
        # Log alert
        log_message = f"ALERT [{severity.value}] {message}: {value} (threshold: {threshold})"
        if severity == AlertSeverity.CRITICAL:
            logger.critical(log_message)
        elif severity == AlertSeverity.ERROR:
            logger.error(log_message)
        elif severity == AlertSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        # Remove old pending transactions (> 1 hour)
        now = datetime.now()
        old_txs = []
        
        for tx_id, tx_data in self.pending_transactions.items():
            if (now - tx_data.get("timestamp", now)).total_seconds() > 3600:
                old_txs.append(tx_id)
        
        for tx_id in old_txs:
            del self.pending_transactions[tx_id]
    
    # Public methods for recording events
    
    def record_transaction(self, tx_id: str, tx_data: Dict[str, Any]):
        """Record a new transaction"""
        self.pending_transactions[tx_id] = {
            **tx_data,
            "timestamp": datetime.now()
        }
        self.performance_stats["total_transactions"] += 1
    
    def record_transaction_success(self, tx_id: str, gas_used: int = 0):
        """Record successful transaction completion"""
        if tx_id in self.pending_transactions:
            tx_data = self.pending_transactions.pop(tx_id)
            tx_data["status"] = "success"
            tx_data["gas_used"] = gas_used
            tx_data["completed_at"] = datetime.now()
            
            # Calculate latency
            latency = (tx_data["completed_at"] - tx_data["timestamp"]).total_seconds() * 1000
            tx_data["latency_ms"] = latency
            
            self.completed_transactions.append(tx_data)
            self.performance_stats["successful_transactions"] += 1
            self.performance_stats["total_gas_used"] += gas_used
            
            # Update average latency
            self._update_average_latency(latency)
    
    def record_transaction_failure(self, tx_id: str, error: str):
        """Record transaction failure"""
        if tx_id in self.pending_transactions:
            tx_data = self.pending_transactions.pop(tx_id)
            tx_data["status"] = "failed"
            tx_data["error"] = error
            tx_data["failed_at"] = datetime.now()
            
            self.failed_transactions.append(tx_data)
            self.performance_stats["failed_transactions"] += 1
    
    def record_agent_activity(self, agent_id: str, activity_type: str, metadata: Dict[str, Any] = None):
        """Record agent activity"""
        self.agent_last_seen[agent_id] = datetime.now()
        
        if agent_id not in self.agent_status:
            self.agent_status[agent_id] = {
                "first_seen": datetime.now(),
                "activities": []
            }
        
        self.agent_status[agent_id]["activities"].append({
            "type": activity_type,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        })
    
    def record_message_routed(self, from_agent: str, to_agent: str, message_type: str):
        """Record message routing"""
        self.performance_stats["total_messages_routed"] += 1
        self.record_agent_activity(from_agent, "message_sent", {"to": to_agent, "type": message_type})
        self.record_agent_activity(to_agent, "message_received", {"from": from_agent, "type": message_type})
    
    def _update_average_latency(self, new_latency: float):
        """Update running average latency"""
        current_avg = self.performance_stats.get("average_latency", 0)
        count = self.performance_stats.get("successful_transactions", 1)
        
        # Calculate new average
        new_avg = ((current_avg * (count - 1)) + new_latency) / count
        self.performance_stats["average_latency"] = new_avg
    
    # Reporting methods
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {
            "performance": self.performance_stats,
            "alerts": {
                "total": len(self.alert_history),
                "by_severity": defaultdict(int)
            },
            "agents": {
                "total": len(self.agent_status),
                "active": sum(
                    1 for last_seen in self.agent_last_seen.values()
                    if (datetime.now() - last_seen).total_seconds() < 300
                )
            },
            "latest_metrics": {}
        }
        
        # Count alerts by severity
        for alert in self.alert_history:
            summary["alerts"]["by_severity"][alert.severity.value] += 1
        
        # Get latest metric values
        for metric_type, snapshots in self.metrics.items():
            if snapshots:
                latest = snapshots[-1]
                summary["latest_metrics"][metric_type.value] = {
                    "value": latest.value,
                    "timestamp": latest.timestamp.isoformat()
                }
        
        return summary
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        # Calculate health score (0-100)
        health_score = 100
        issues = []
        
        # Check transaction success rate
        if self.performance_stats["total_transactions"] > 0:
            success_rate = (
                self.performance_stats["successful_transactions"] / 
                self.performance_stats["total_transactions"]
            )
            if success_rate < 0.95:
                health_score -= 20
                issues.append(f"Low transaction success rate: {success_rate:.2%}")
        
        # Check agent availability
        total_agents = len(self.agent_status)
        if total_agents > 0:
            active_agents = sum(
                1 for last_seen in self.agent_last_seen.values()
                if (datetime.now() - last_seen).total_seconds() < 300
            )
            availability = active_agents / total_agents
            if availability < 0.90:
                health_score -= 15
                issues.append(f"Low agent availability: {availability:.2%}")
        
        # Check recent alerts
        recent_alerts = [
            alert for alert in self.alert_history
            if (datetime.now() - alert.timestamp).total_seconds() < 3600
        ]
        critical_alerts = sum(1 for alert in recent_alerts if alert.severity == AlertSeverity.CRITICAL)
        if critical_alerts > 0:
            health_score -= (critical_alerts * 10)
            issues.append(f"{critical_alerts} critical alerts in last hour")
        
        return {
            "health_score": max(0, health_score),
            "status": "healthy" if health_score >= 80 else "degraded" if health_score >= 60 else "unhealthy",
            "issues": issues,
            "timestamp": datetime.now().isoformat()
        }


# Alert handlers

async def log_alert_handler(alert: Alert):
    """Simple alert handler that logs alerts"""
    logger.info(f"Alert handled: {alert.severity.value} - {alert.message}")


async def webhook_alert_handler(alert: Alert, webhook_url: str):
    """Send alerts to a webhook"""
    # Would implement actual webhook call
    pass


async def email_alert_handler(alert: Alert, email_config: Dict[str, Any]):
    """Send alerts via email"""
    # Would implement email sending
    pass


# Monitoring dashboard (text-based for CLI)

def print_monitoring_dashboard(monitor: BlockchainMonitor):
    """Print a text-based monitoring dashboard"""
    summary = monitor.get_metrics_summary()
    health = monitor.get_health_status()
    
    print("\n" + "="*60)
    print("BLOCKCHAIN MONITORING DASHBOARD")
    print("="*60)
    
    # Health status
    status_color = "🟢" if health["health_score"] >= 80 else "🟡" if health["health_score"] >= 60 else "🔴"
    print(f"\nHealth Status: {status_color} {health['status'].upper()} (Score: {health['health_score']}/100)")
    
    if health["issues"]:
        print("\nIssues:")
        for issue in health["issues"]:
            print(f"  - {issue}")
    
    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"  Total Transactions: {summary['performance']['total_transactions']}")
    print(f"  Success Rate: {summary['performance']['successful_transactions']}/{summary['performance']['total_transactions']} ", end="")
    if summary['performance']['total_transactions'] > 0:
        rate = summary['performance']['successful_transactions'] / summary['performance']['total_transactions']
        print(f"({rate:.2%})")
    else:
        print("(N/A)")
    
    print(f"  Average Latency: {summary['performance']['average_latency']:.2f}ms")
    print(f"  Total Gas Used: {summary['performance']['total_gas_used']:,} wei")
    print(f"  Messages Routed: {summary['performance']['total_messages_routed']}")
    
    # Agent status
    print(f"\nAgent Status:")
    print(f"  Total Agents: {summary['agents']['total']}")
    print(f"  Active Agents: {summary['agents']['active']}")
    
    # Alerts
    print(f"\nAlerts:")
    print(f"  Total Alerts: {summary['alerts']['total']}")
    for severity, count in summary['alerts']['by_severity'].items():
        if count > 0:
            print(f"  {severity.upper()}: {count}")
    
    # Latest metrics
    print(f"\nLatest Metrics:")
    for metric_name, metric_data in summary['latest_metrics'].items():
        print(f"  {metric_name}: {metric_data['value']}")
    
    print("\n" + "="*60)


# Example usage
async def example_monitoring():
    """Example of using the monitoring system"""
    # Create monitor with alert handlers
    monitor = BlockchainMonitor(
        check_interval=5,  # Check every 5 seconds for demo
        alert_handlers=[log_alert_handler]
    )
    
    # Start monitoring
    await monitor.start()
    
    # Simulate some activity
    for i in range(10):
        # Record transaction
        tx_id = f"tx_{i}"
        monitor.record_transaction(tx_id, {"type": "test", "value": i})
        
        # Simulate completion (90% success rate)
        await asyncio.sleep(0.1)
        if i % 10 != 0:
            monitor.record_transaction_success(tx_id, gas_used=21000 + (i * 1000))
        else:
            monitor.record_transaction_failure(tx_id, "Simulated failure")
        
        # Record agent activity
        monitor.record_agent_activity(f"agent_{i % 3}", "test_action")
        
        # Record message
        monitor.record_message_routed(f"agent_{i % 3}", f"agent_{(i+1) % 3}", "TEST")
        
        # Print dashboard every 3 iterations
        if i % 3 == 0:
            print_monitoring_dashboard(monitor)
            await asyncio.sleep(1)
    
    # Final dashboard
    print_monitoring_dashboard(monitor)
    
    # Stop monitoring
    await monitor.stop()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_monitoring())