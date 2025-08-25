"""
Comprehensive Monitoring Dashboard for Enhanced Glean Agent
Provides real-time metrics, alerts, and performance insights
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from threading import Lock
import asyncio

from .base_agent import BaseAgent, ErrorSeverity
from app.a2a.core.security_base import SecureA2AAgent


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    type: str
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['severity'] = self.severity.value
        result['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            result['resolved_at'] = self.resolved_at.isoformat()
        return result


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class MonitoringDashboard(SecureA2AAgent):
    """Comprehensive monitoring dashboard for Glean Agent"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('monitoring-dashboard', config)
        # Security features are initialized by SecureA2AAgent base class


        # Configuration
        self.config = config or {}
        self.metrics_retention_hours = self.config.get('metrics_retention_hours', 24)
        self.alert_thresholds = self.config.get('alert_thresholds', self._get_default_thresholds())

        # Data storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: Dict[str, Alert] = {}
        self.system_status = {
            'status': 'healthy',
            'last_updated': datetime.utcnow(),
            'components': {}
        }

        # Thread safety
        self._lock = Lock()

        # Performance tracking
        self.operation_stats = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0.0,
            'avg_duration': 0.0,
            'min_duration': float('inf'),
            'max_duration': 0.0,
            'error_count': 0,
            'success_rate': 0.0
        })

        # Start background tasks
        self._start_monitoring_tasks()

        self.logger.info('Monitoring Dashboard initialized')

    def _get_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get default alert thresholds"""
        return {
            'response_time': {
                'warning': 1000.0,    # ms
                'critical': 5000.0    # ms
            },
            'error_rate': {
                'warning': 0.05,      # 5%
                'critical': 0.10      # 10%
            },
            'memory_usage': {
                'warning': 0.80,      # 80%
                'critical': 0.95      # 95%
            },
            'cpu_usage': {
                'warning': 0.80,      # 80%
                'critical': 0.95      # 95%
            },
            'disk_usage': {
                'warning': 0.85,      # 85%
                'critical': 0.95      # 95%
            }
        }

    def record_metric(self,
                     name: str,
                     value: float,
                     unit: str = '',
                     tags: Optional[Dict[str, str]] = None):
        """Record a performance metric"""
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                unit=unit,
                timestamp=datetime.utcnow(),
                tags=tags or {}
            )

            self.metrics[name].append(metric)

            # Check for alerts
            self._check_metric_alerts(name, value, tags)

    def record_operation_result(self,
                              operation: str,
                              duration: float,
                              success: bool = True,
                              tags: Optional[Dict[str, str]] = None):
        """Record operation performance result"""
        with self._lock:
            stats = self.operation_stats[operation]
            stats['count'] += 1
            stats['total_duration'] += duration
            stats['avg_duration'] = stats['total_duration'] / stats['count']
            stats['min_duration'] = min(stats['min_duration'], duration)
            stats['max_duration'] = max(stats['max_duration'], duration)

            if not success:
                stats['error_count'] += 1

            stats['success_rate'] = (stats['count'] - stats['error_count']) / stats['count']

            # Record as metrics
            self.record_metric(f'operation.{operation}.duration', duration, 'ms', tags)
            self.record_metric(f'operation.{operation}.success_rate', stats['success_rate'], 'ratio', tags)

    def create_alert(self,
                    alert_type: str,
                    severity: ErrorSeverity,
                    message: str,
                    details: Optional[Dict[str, Any]] = None) -> str:
        """Create a new alert"""
        alert_id = f"{alert_type}_{int(time.time() * 1000)}"

        with self._lock:
            alert = Alert(
                id=alert_id,
                type=alert_type,
                severity=severity,
                message=message,
                details=details or {},
                timestamp=datetime.utcnow()
            )

            self.alerts[alert_id] = alert

            # Log alert
            self.logger.warning(
                'Alert created',
                alert_id=alert_id,
                type=alert_type,
                severity=severity.value,
                message=message
            )

            # Update system status if critical
            if severity == ErrorSeverity.CRITICAL:
                self.system_status['status'] = 'critical'
                self.system_status['last_updated'] = datetime.utcnow()

        return alert_id

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolved_at = datetime.utcnow()

                self.logger.info('Alert resolved', alert_id=alert_id)

                # Update system status if no more critical alerts
                self._update_system_status()

                return True

        return False

    def _check_metric_alerts(self,
                           metric_name: str,
                           value: float,
                           tags: Optional[Dict[str, str]]):
        """Check if metric value triggers alerts"""
        # Check against thresholds
        for threshold_name, thresholds in self.alert_thresholds.items():
            if threshold_name in metric_name.lower():
                if value >= thresholds.get('critical', float('inf')):
                    self.create_alert(
                        f'{metric_name}_critical',
                        ErrorSeverity.CRITICAL,
                        f'{metric_name} is critically high: {value}',
                        {'metric': metric_name, 'value': value, 'threshold': thresholds['critical']}
                    )
                elif value >= thresholds.get('warning', float('inf')):
                    self.create_alert(
                        f'{metric_name}_warning',
                        ErrorSeverity.MEDIUM,
                        f'{metric_name} is above warning threshold: {value}',
                        {'metric': metric_name, 'value': value, 'threshold': thresholds['warning']}
                    )

    def _update_system_status(self):
        """Update overall system status"""
        critical_alerts = [a for a in self.alerts.values()
                          if not a.resolved and a.severity == ErrorSeverity.CRITICAL]
        high_alerts = [a for a in self.alerts.values()
                      if not a.resolved and a.severity == ErrorSeverity.HIGH]

        if critical_alerts:
            self.system_status['status'] = 'critical'
        elif high_alerts:
            self.system_status['status'] = 'degraded'
        else:
            self.system_status['status'] = 'healthy'

        self.system_status['last_updated'] = datetime.utcnow()

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        with self._lock:
            # Get recent metrics
            recent_metrics = {}
            for name, metric_list in self.metrics.items():
                if metric_list:
                    recent_metrics[name] = [m.to_dict() for m in list(metric_list)[-100:]]

            # Get active alerts
            active_alerts = [a.to_dict() for a in self.alerts.values() if not a.resolved]

            # Get operation statistics
            operation_summary = {}
            for op, stats in self.operation_stats.items():
                operation_summary[op] = {
                    'total_operations': stats['count'],
                    'avg_response_time': round(stats['avg_duration'], 2),
                    'success_rate': round(stats['success_rate'] * 100, 2),
                    'error_count': stats['error_count']
                }

            return {
                'system_status': {
                    **self.system_status,
                    'last_updated': self.system_status['last_updated'].isoformat()
                },
                'metrics': recent_metrics,
                'alerts': {
                    'active': active_alerts,
                    'total_count': len(self.alerts),
                    'active_count': len(active_alerts)
                },
                'operations': operation_summary,
                'performance_summary': self._get_performance_summary(),
                'health_checks': self._run_health_checks(),
                'timestamp': datetime.utcnow().isoformat()
            }

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        summary = {
            'total_operations': sum(stats['count'] for stats in self.operation_stats.values()),
            'total_errors': sum(stats['error_count'] for stats in self.operation_stats.values()),
            'average_response_time': 0.0,
            'overall_success_rate': 0.0
        }

        if summary['total_operations'] > 0:
            total_duration = sum(stats['total_duration'] for stats in self.operation_stats.values())
            summary['average_response_time'] = round(total_duration / summary['total_operations'], 2)
            summary['overall_success_rate'] = round(
                (summary['total_operations'] - summary['total_errors']) / summary['total_operations'] * 100, 2
            )

        return summary

    def _run_health_checks(self) -> Dict[str, Any]:
        """Run basic health checks"""
        checks = {}

        # Memory usage check
        import psutil
        memory = psutil.virtual_memory()
        checks['memory'] = {
            'status': 'healthy' if memory.percent < 80 else 'warning' if memory.percent < 95 else 'critical',
            'usage_percent': memory.percent,
            'available_gb': round(memory.available / (1024**3), 2)
        }

        # CPU usage check
        cpu_percent = psutil.cpu_percent(interval=1)
        checks['cpu'] = {
            'status': 'healthy' if cpu_percent < 80 else 'warning' if cpu_percent < 95 else 'critical',
            'usage_percent': cpu_percent
        }

        # Disk usage check
        disk = psutil.disk_usage('/')
        disk_percent = disk.used / disk.total * 100
        checks['disk'] = {
            'status': 'healthy' if disk_percent < 85 else 'warning' if disk_percent < 95 else 'critical',
            'usage_percent': round(disk_percent, 2),
            'free_gb': round(disk.free / (1024**3), 2)
        }

        # Alert status check
        critical_alerts = len([a for a in self.alerts.values()
                              if not a.resolved and a.severity == ErrorSeverity.CRITICAL])
        checks['alerts'] = {
            'status': 'healthy' if critical_alerts == 0 else 'critical',
            'critical_count': critical_alerts,
            'total_active': len([a for a in self.alerts.values() if not a.resolved])
        }

        return checks

    def get_metrics_history(self,
                          metric_name: str,
                          hours: int = 1) -> List[Dict[str, Any]]:
        """Get historical metrics data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        with self._lock:
            if metric_name in self.metrics:
                filtered_metrics = [
                    m.to_dict() for m in self.metrics[metric_name]
                    if m.timestamp >= cutoff_time
                ]
                return filtered_metrics

        return []

    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        with self._lock:
            filtered_alerts = [
                a.to_dict() for a in self.alerts.values()
                if a.timestamp >= cutoff_time
            ]

            return sorted(filtered_alerts, key=lambda x: x['timestamp'], reverse=True)

    def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        # This would typically start background threads or async tasks
        # For now, we'll just log that monitoring is active
        self.logger.info('Background monitoring tasks started')

    def cleanup_old_data(self):
        """Clean up old metrics and resolved alerts"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.metrics_retention_hours)

        with self._lock:
            # Clean up old metrics
            for metric_name in self.metrics:
                self.metrics[metric_name] = deque(
                    [m for m in self.metrics[metric_name] if m.timestamp >= cutoff_time],
                    maxlen=10000
                )

            # Clean up old resolved alerts
            old_alerts = [
                alert_id for alert_id, alert in self.alerts.items()
                if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time
            ]

            for alert_id in old_alerts:
                del self.alerts[alert_id]

            self.logger.debug(f'Cleaned up {len(old_alerts)} old alerts')

    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        data = self.get_dashboard_data()

        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        elif format.lower() == 'prometheus':
            return self._export_prometheus_format(data)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_prometheus_format(self, data: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format"""
        lines = []

        # System status
        status_value = 1 if data['system_status']['status'] == 'healthy' else 0
        lines.append(f'a2a_system_status {status_value}')

        # Operation metrics
        for op_name, stats in data['operations'].items():
            clean_name = op_name.replace('.', '_').replace('-', '_')
            lines.append(f'a2a_operation_total{{operation="{op_name}"}} {stats["total_operations"]}')
            lines.append(f'a2a_operation_duration_avg{{operation="{op_name}"}} {stats["avg_response_time"]}')
            lines.append(f'a2a_operation_success_rate{{operation="{op_name}"}} {stats["success_rate"]/100}')
            lines.append(f'a2a_operation_errors_total{{operation="{op_name}"}} {stats["error_count"]}')

        # Alert metrics
        lines.append(f'a2a_alerts_active_total {data["alerts"]["active_count"]}')

        return '\n'.join(lines)
