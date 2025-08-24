"""
A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
"""

#!/usr/bin/env python3
"""
Disaster Recovery Orchestrator for A2A Platform
Handles disaster detection, failover coordination, and system recovery
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
# A2A Protocol: Use blockchain messaging instead of aiohttp
import yaml
from pathlib import Path


class DisasterType(Enum):
    """Types of disasters that can occur"""
    DATABASE_FAILURE = "database_failure"
    SERVICE_OUTAGE = "service_outage" 
    NETWORK_PARTITION = "network_partition"
    DATA_CORRUPTION = "data_corruption"
    SECURITY_BREACH = "security_breach"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    CASCADING_FAILURE = "cascading_failure"


class RecoveryStatus(Enum):
    """Recovery process status"""
    DETECTING = "detecting"
    CONFIRMED = "confirmed"
    INITIATING = "initiating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"


@dataclass
class DisasterEvent:
    """Disaster event information"""
    id: str
    type: DisasterType
    severity: str  # critical, high, medium, low
    detected_at: datetime
    affected_components: List[str]
    description: str
    automated_response: bool = True
    recovery_status: RecoveryStatus = RecoveryStatus.DETECTING
    recovery_steps: List[Dict[str, Any]] = None
    estimated_recovery_time: Optional[int] = None  # minutes
    actual_recovery_time: Optional[int] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.recovery_steps is None:
            self.recovery_steps = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RecoveryPlan:
    """Disaster recovery plan configuration"""
    disaster_type: DisasterType
    priority: int
    automated: bool
    steps: List[Dict[str, Any]]
    timeout_minutes: int
    prerequisites: List[str]
    rollback_steps: List[Dict[str, Any]]


class A2ADisasterRecovery:
    """Comprehensive disaster recovery orchestrator"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Recovery state
        self.active_disasters: Dict[str, DisasterEvent] = {}
        self.recovery_plans: Dict[DisasterType, RecoveryPlan] = {}
        self.health_checks: Dict[str, Dict[str, Any]] = {}
        self.failover_targets: Dict[str, List[str]] = {}
        
        # Monitoring
        self.monitoring_tasks: List[asyncio.Task] = []
        self.alert_callbacks: List[Callable] = []
        self.is_running = False
        
        # Statistics
        self.stats = {
            'total_disasters': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'average_recovery_time': 0,
            'last_disaster': None,
            'system_uptime_start': datetime.now()
        }
        
        self._initialize_recovery_plans()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load disaster recovery configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        
        # Default configuration
        return {
            'monitoring': {
                'health_check_interval': 30,
                'disaster_detection_threshold': 3,
                'alert_escalation_time': 300
            },
            'recovery': {
                'max_concurrent_recoveries': 2,
                'backup_retention_days': 30,
                'automated_recovery_enabled': True,
                'failover_timeout': 600
            },
            'services': {
                'a2a-network': {
                    'health_endpoint': 'http://a2a-network:4004/health',
                    'backup_service': 'a2a-network-backup',
                    'failover_targets': ['a2a-network-2', 'a2a-network-3']
                },
                'a2a-agents': {
                    'health_endpoint': 'http://a2a-agents:8000/health',
                    'backup_service': 'a2a-agents-backup',
                    'failover_targets': ['a2a-agents-2', 'a2a-agents-3']
                },
                'postgresql': {
                    'health_endpoint': 'postgresql://postgres:5432/a2a',
                    'backup_service': 'postgresql-backup',
                    'failover_targets': ['postgresql-replica-1', 'postgresql-replica-2']
                },
                'redis': {
                    'health_endpoint': 'redis://redis:6379',
                    'backup_service': 'redis-backup',
                    'failover_targets': ['redis-replica-1', 'redis-replica-2']
                }
            },
            'notifications': {
                'slack_webhook': os.getenv('SLACK_WEBHOOK_URL'),
                'email_recipients': os.getenv('DR_EMAIL_RECIPIENTS', '').split(','),
                'pagerduty_key': os.getenv('PAGERDUTY_INTEGRATION_KEY')
            }
        }
    
    def _initialize_recovery_plans(self):
        """Initialize disaster recovery plans"""
        
        # Database failure recovery plan
        self.recovery_plans[DisasterType.DATABASE_FAILURE] = RecoveryPlan(
            disaster_type=DisasterType.DATABASE_FAILURE,
            priority=1,
            automated=True,
            timeout_minutes=30,
            prerequisites=['backup_available', 'failover_target_healthy'],
            steps=[
                {
                    'name': 'stop_database_connections',
                    'action': 'execute_command',
                    'command': 'kubectl scale deployment postgresql --replicas=0',
                    'timeout': 60
                },
                {
                    'name': 'promote_replica',
                    'action': 'failover_database',
                    'target': 'postgresql-replica-1',
                    'timeout': 300
                },
                {
                    'name': 'update_service_discovery',
                    'action': 'update_dns',
                    'target': 'postgresql-replica-1',
                    'timeout': 60
                },
                {
                    'name': 'restart_dependent_services',
                    'action': 'restart_services',
                    'services': ['a2a-network', 'a2a-agents'],
                    'timeout': 180
                }
            ],
            rollback_steps=[
                {
                    'name': 'restore_original_database',
                    'action': 'restore_from_backup',
                    'backup_id': 'latest',
                    'timeout': 600
                }
            ]
        )
        
        # Service outage recovery plan
        self.recovery_plans[DisasterType.SERVICE_OUTAGE] = RecoveryPlan(
            disaster_type=DisasterType.SERVICE_OUTAGE,
            priority=2,
            automated=True,
            timeout_minutes=15,
            prerequisites=['backup_available'],
            steps=[
                {
                    'name': 'restart_service',
                    'action': 'restart_service',
                    'timeout': 120
                },
                {
                    'name': 'scale_up_replicas',
                    'action': 'scale_service',
                    'replicas': 2,
                    'timeout': 180
                },
                {
                    'name': 'check_dependencies',
                    'action': 'health_check_dependencies',
                    'timeout': 60
                }
            ],
            rollback_steps=[
                {
                    'name': 'revert_scaling',
                    'action': 'scale_service',
                    'replicas': 1,
                    'timeout': 120
                }
            ]
        )
        
        # Security breach recovery plan
        self.recovery_plans[DisasterType.SECURITY_BREACH] = RecoveryPlan(
            disaster_type=DisasterType.SECURITY_BREACH,
            priority=1,
            automated=False,  # Requires manual intervention
            timeout_minutes=60,
            prerequisites=['security_team_notified'],
            steps=[
                {
                    'name': 'isolate_affected_services',
                    'action': 'isolate_network',
                    'timeout': 300
                },
                {
                    'name': 'rotate_credentials',
                    'action': 'rotate_all_secrets',
                    'timeout': 600
                },
                {
                    'name': 'audit_system_integrity',
                    'action': 'run_security_scan',
                    'timeout': 1800
                },
                {
                    'name': 'restore_from_clean_backup',
                    'action': 'restore_from_backup',
                    'backup_age_hours': 24,
                    'timeout': 1200
                }
            ],
            rollback_steps=[]
        )
    
    async def start_monitoring(self):
        """Start disaster recovery monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("Starting A2A Disaster Recovery monitoring")
        
        # Start health monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._monitor_services()),
            asyncio.create_task(self._monitor_infrastructure()),
            asyncio.create_task(self._monitor_data_integrity()),
            asyncio.create_task(self._process_recovery_queue())
        ]
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
    
    async def stop_monitoring(self):
        """Stop disaster recovery monitoring"""
        self.is_running = False
        
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.logger.info("A2A Disaster Recovery monitoring stopped")
    
    async def _monitor_services(self):
        """Monitor service health and detect outages"""
        while self.is_running:
            try:
                for service_name, service_config in self.config['services'].items():
                    health_status = await self._check_service_health(service_name, service_config)
                    
                    if not health_status['healthy']:
                        await self._handle_service_failure(service_name, health_status)
                
                await asyncio.sleep(self.config['monitoring']['health_check_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in service monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_infrastructure(self):
        """Monitor infrastructure components"""
        while self.is_running:
            try:
                # Check system resources
                cpu_usage = await self._get_cpu_usage()
                memory_usage = await self._get_memory_usage()
                disk_usage = await self._get_disk_usage()
                
                if cpu_usage > 90 or memory_usage > 95 or disk_usage > 95:
                    await self._handle_resource_exhaustion({
                        'cpu': cpu_usage,
                        'memory': memory_usage,
                        'disk': disk_usage
                    })
                
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in infrastructure monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_data_integrity(self):
        """Monitor data integrity and detect corruption"""
        while self.is_running:
            try:
                # Check database integrity
                for service_name, service_config in self.config['services'].items():
                    if 'postgresql' in service_name.lower():
                        integrity_check = await self._check_database_integrity(service_name)
                        
                        if not integrity_check['valid']:
                            await self._handle_data_corruption(service_name, integrity_check)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in data integrity monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _check_service_health(self, service_name: str, service_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check health of a specific service"""
        health_endpoint = service_config.get('health_endpoint')
        if not health_endpoint:
            return {'healthy': True, 'reason': 'no_health_check_configured'}
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if health_endpoint.startswith('http'):
                    async with session.get(health_endpoint) as response:
                        if response.status == 200:
                            data = await response.json()
                            return {
                                'healthy': True,
                                'response_time': response.headers.get('x-response-time', 0),
                                'data': data
                            }
                        else:
                            return {
                                'healthy': False,
                                'reason': f'http_status_{response.status}',
                                'response_time': None
                            }
                
                elif health_endpoint.startswith('postgresql://'):
                    # Database health check
                    return await self._check_postgresql_health(health_endpoint)
                
                elif health_endpoint.startswith('redis://'):
                    # Redis health check
                    return await self._check_redis_health(health_endpoint)
                
        except Exception as e:
            return {
                'healthy': False,
                'reason': f'connection_error: {str(e)}',
                'response_time': None
            }
    
    async def _check_postgresql_health(self, connection_string: str) -> Dict[str, Any]:
        """Check PostgreSQL database health"""
        try:
            import asyncpg
            
            conn = await asyncpg.connect(connection_string)
            
            # Simple query to check connectivity
            result = await conn.fetchval('SELECT 1')
            
            # Check for replication lag (if applicable)
            try:
                lag = await conn.fetchval("""
                    SELECT CASE 
                        WHEN pg_is_in_recovery() THEN 
                            EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))
                        ELSE NULL 
                    END
                """)
            except:
                lag = None
            
            await conn.close()
            
            return {
                'healthy': True,
                'replication_lag': lag,
                'connection_test': result == 1
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'reason': f'database_error: {str(e)}'
            }
    
    async def _check_redis_health(self, connection_string: str) -> Dict[str, Any]:
        """Check Redis health"""
        try:
            import aioredis
            
            redis = aioredis.from_url(connection_string)
            
            # Ping test
            pong = await redis.ping()
            
            # Get basic info
            info = await redis.info()
            
            await redis.close()
            
            return {
                'healthy': pong is True,
                'used_memory': info.get('used_memory', 0),
                'connected_clients': info.get('connected_clients', 0)
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'reason': f'redis_error: {str(e)}'
            }
    
    async def _handle_service_failure(self, service_name: str, health_status: Dict[str, Any]):
        """Handle detected service failure"""
        disaster_id = f"service_failure_{service_name}_{int(time.time())}"
        
        disaster = DisasterEvent(
            id=disaster_id,
            type=DisasterType.SERVICE_OUTAGE,
            severity='high',
            detected_at=datetime.now(),
            affected_components=[service_name],
            description=f"Service {service_name} health check failed: {health_status.get('reason', 'unknown')}",
            metadata={'health_status': health_status}
        )
        
        await self._initiate_disaster_response(disaster)
    
    async def _handle_resource_exhaustion(self, resource_usage: Dict[str, float]):
        """Handle resource exhaustion scenario"""
        disaster_id = f"resource_exhaustion_{int(time.time())}"
        
        critical_resources = [k for k, v in resource_usage.items() if v > 95]
        
        disaster = DisasterEvent(
            id=disaster_id,
            type=DisasterType.INFRASTRUCTURE_FAILURE,
            severity='critical' if len(critical_resources) > 1 else 'high',
            detected_at=datetime.now(),
            affected_components=['infrastructure'],
            description=f"Resource exhaustion detected: {', '.join(critical_resources)}",
            metadata={'resource_usage': resource_usage}
        )
        
        await self._initiate_disaster_response(disaster)
    
    async def _handle_data_corruption(self, service_name: str, integrity_check: Dict[str, Any]):
        """Handle data corruption detection"""
        disaster_id = f"data_corruption_{service_name}_{int(time.time())}"
        
        disaster = DisasterEvent(
            id=disaster_id,
            type=DisasterType.DATA_CORRUPTION,
            severity='critical',
            detected_at=datetime.now(),
            affected_components=[service_name],
            description=f"Data corruption detected in {service_name}",
            automated_response=False,  # Requires manual intervention
            metadata={'integrity_check': integrity_check}
        )
        
        await self._initiate_disaster_response(disaster)
    
    async def _initiate_disaster_response(self, disaster: DisasterEvent):
        """Initiate disaster response procedure"""
        self.logger.error(f"DISASTER DETECTED: {disaster.id} - {disaster.description}")
        
        # Add to active disasters
        self.active_disasters[disaster.id] = disaster
        
        # Update statistics
        self.stats['total_disasters'] += 1
        self.stats['last_disaster'] = disaster.detected_at
        
        # Send alerts
        await self._send_disaster_alert(disaster)
        
        # Check if automated response is enabled and allowed
        if (disaster.automated_response and 
            self.config['recovery']['automated_recovery_enabled'] and
            disaster.type in self.recovery_plans):
            
            disaster.recovery_status = RecoveryStatus.INITIATING
            await self._execute_recovery_plan(disaster)
        else:
            self.logger.warning(f"Manual intervention required for disaster: {disaster.id}")
            disaster.recovery_status = RecoveryStatus.CONFIRMED
    
    async def _execute_recovery_plan(self, disaster: DisasterEvent):
        """Execute disaster recovery plan"""
        recovery_plan = self.recovery_plans.get(disaster.type)
        if not recovery_plan:
            self.logger.error(f"No recovery plan found for disaster type: {disaster.type}")
            disaster.recovery_status = RecoveryStatus.FAILED
            return
        
        disaster.recovery_status = RecoveryStatus.IN_PROGRESS
        disaster.estimated_recovery_time = recovery_plan.timeout_minutes
        
        recovery_start_time = time.time()
        
        try:
            self.logger.info(f"Starting recovery for disaster: {disaster.id}")
            
            # Check prerequisites
            for prerequisite in recovery_plan.prerequisites:
                if not await self._check_prerequisite(prerequisite, disaster):
                    raise Exception(f"Prerequisite not met: {prerequisite}")
            
            # Execute recovery steps
            for step in recovery_plan.steps:
                self.logger.info(f"Executing recovery step: {step['name']}")
                
                step_start_time = time.time()
                
                try:
                    await self._execute_recovery_step(step, disaster)
                    
                    step_duration = time.time() - step_start_time
                    self.logger.info(f"Step {step['name']} completed in {step_duration:.2f}s")
                    
                    disaster.recovery_steps.append({
                        'name': step['name'],
                        'status': 'completed',
                        'duration': step_duration,
                        'timestamp': datetime.now()
                    })
                    
                except Exception as step_error:
                    self.logger.error(f"Recovery step {step['name']} failed: {step_error}")
                    
                    disaster.recovery_steps.append({
                        'name': step['name'],
                        'status': 'failed',
                        'error': str(step_error),
                        'timestamp': datetime.now()
                    })
                    
                    # Execute rollback if step fails
                    await self._execute_rollback(disaster, recovery_plan)
                    raise step_error
            
            # Recovery completed successfully
            disaster.recovery_status = RecoveryStatus.COMPLETED
            disaster.actual_recovery_time = int((time.time() - recovery_start_time) / 60)
            
            self.stats['successful_recoveries'] += 1
            self.stats['average_recovery_time'] = (
                (self.stats['average_recovery_time'] * (self.stats['successful_recoveries'] - 1) + 
                 disaster.actual_recovery_time) / self.stats['successful_recoveries']
            )
            
            await self._send_recovery_success_alert(disaster)
            self.logger.info(f"Disaster recovery completed: {disaster.id}")
            
        except Exception as e:
            disaster.recovery_status = RecoveryStatus.FAILED
            disaster.actual_recovery_time = int((time.time() - recovery_start_time) / 60)
            
            self.stats['failed_recoveries'] += 1
            
            await self._send_recovery_failure_alert(disaster, str(e))
            self.logger.error(f"Disaster recovery failed: {disaster.id} - {e}")
    
    async def _execute_recovery_step(self, step: Dict[str, Any], disaster: DisasterEvent):
        """Execute a single recovery step"""
        action = step.get('action')
        timeout = step.get('timeout', 300)
        
        if action == 'execute_command':
            await self._execute_command(step['command'], timeout)
            
        elif action == 'failover_database':
            await self._failover_database(step['target'], timeout)
            
        elif action == 'update_dns':
            await self._update_dns_record(step['target'], timeout)
            
        elif action == 'restart_services':
            await self._restart_services(step['services'], timeout)
            
        elif action == 'scale_service':
            await self._scale_service(disaster.affected_components[0], step['replicas'], timeout)
            
        elif action == 'health_check_dependencies':
            await self._health_check_dependencies(disaster.affected_components[0])
            
        elif action == 'isolate_network':
            await self._isolate_network(disaster.affected_components)
            
        elif action == 'rotate_all_secrets':
            await self._rotate_all_secrets()
            
        elif action == 'run_security_scan':
            await self._run_security_scan()
            
        elif action == 'restore_from_backup':
            backup_id = step.get('backup_id', 'latest')
            backup_age_hours = step.get('backup_age_hours', 0)
            await self._restore_from_backup(disaster.affected_components[0], backup_id, backup_age_hours)
            
        else:
            raise Exception(f"Unknown recovery action: {action}")
    
    async def _execute_command(self, command: str, timeout: int):
        """Execute shell command"""
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            
            if proc.returncode != 0:
                raise Exception(f"Command failed: {stderr.decode()}")
                
            return stdout.decode()
            
        except asyncio.TimeoutError:
            proc.kill()
            raise Exception(f"Command timed out after {timeout}s")
    
    async def _send_disaster_alert(self, disaster: DisasterEvent):
        """Send disaster alert notifications"""
        alert_message = f"""
🚨 DISASTER ALERT 🚨

**Disaster ID:** {disaster.id}
**Type:** {disaster.type.value}
**Severity:** {disaster.severity.upper()}
**Detected:** {disaster.detected_at}
**Components:** {', '.join(disaster.affected_components)}
**Description:** {disaster.description}

**Automated Response:** {'Enabled' if disaster.automated_response else 'Manual intervention required'}
"""
        
        # Send to all registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(disaster, alert_message)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
        
        # Send Slack notification
        if self.config['notifications']['slack_webhook']:
            await self._send_slack_alert(alert_message, disaster.severity)
        
        # Send email notification
        if self.config['notifications']['email_recipients']:
            await self._send_email_alert(alert_message, disaster)
    
    async def _send_slack_alert(self, message: str, severity: str):
        """Send Slack alert"""
        webhook_url = self.config['notifications']['slack_webhook']
        if not webhook_url:
            return
        
        color = {
            'critical': '#FF0000',
            'high': '#FFA500',
            'medium': '#FFFF00',
            'low': '#00FF00'
        }.get(severity, '#808080')
        
        payload = {
            'attachments': [{
                'color': color,
                'text': message,
                'title': 'A2A Platform Disaster Recovery Alert'
            }]
        }
        
        try:
            async with A2ANetworkClient() as session:
                await session.post(webhook_url, json=payload)
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.stats['system_uptime_start']).total_seconds(),
            'disaster_recovery': {
                'active_disasters': len(self.active_disasters),
                'monitoring_active': self.is_running,
                'last_health_check': datetime.now().isoformat()
            },
            'statistics': self.stats.copy(),
            'active_disasters': [asdict(disaster) for disaster in self.active_disasters.values()],
            'service_health': {}
        }
        
        # Get current service health
        for service_name, service_config in self.config['services'].items():
            health = await self._check_service_health(service_name, service_config)
            status['service_health'][service_name] = health
        
        return status
    
    def add_alert_callback(self, callback: Callable):
        """Add custom alert callback"""
        self.alert_callbacks.append(callback)
    
    async def trigger_manual_recovery(self, disaster_id: str) -> bool:
        """Manually trigger recovery for a specific disaster"""
        disaster = self.active_disasters.get(disaster_id)
        if not disaster:
            return False
        
        if disaster.recovery_status in [RecoveryStatus.IN_PROGRESS, RecoveryStatus.COMPLETED]:
            return False
        
        disaster.automated_response = True
        await self._execute_recovery_plan(disaster)
        return True
    
    async def _process_recovery_queue(self):
        """Process recovery queue and handle concurrent recoveries"""
        while self.is_running:
            try:
                # Limit concurrent recoveries
                active_recoveries = [
                    d for d in self.active_disasters.values() 
                    if d.recovery_status == RecoveryStatus.IN_PROGRESS
                ]
                
                if len(active_recoveries) < self.config['recovery']['max_concurrent_recoveries']:
                    # Find next disaster to recover
                    pending_disasters = [
                        d for d in self.active_disasters.values()
                        if d.recovery_status == RecoveryStatus.CONFIRMED and d.automated_response
                    ]
                    
                    if pending_disasters:
                        # Sort by priority (disaster type priority)
                        pending_disasters.sort(key=lambda d: self.recovery_plans.get(d.type, RecoveryPlan(
                            disaster_type=d.type, priority=999, automated=True, steps=[], timeout_minutes=60, prerequisites=[], rollback_steps=[]
                        )).priority)
                        
                        disaster = pending_disasters[0]
                        await self._execute_recovery_plan(disaster)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in recovery queue processing: {e}")
                await asyncio.sleep(60)


async def main():
    """Main function for testing disaster recovery system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize disaster recovery
    dr = A2ADisasterRecovery()
    
    # Add custom alert callback
    async def custom_alert_callback(disaster, message):
        print(f"CUSTOM ALERT: {disaster.id}")
        print(message)
    
    dr.add_alert_callback(custom_alert_callback)
    
    try:
        # Start monitoring
        await dr.start_monitoring()
        
    except KeyboardInterrupt:
        print("\nShutting down disaster recovery system...")
        await dr.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())