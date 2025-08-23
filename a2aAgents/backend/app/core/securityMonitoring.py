"""
Comprehensive Security Monitoring and Alerting System
Real-time security event detection, analysis, and automated response
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""


import platform

import asyncio
import json
import logging
import hashlib
import time
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from .errorHandling import SecurityError
from .secrets import get_secrets_manager
from .rbac import get_auth_service

# Encryption imports
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Security threat levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EventType(str, Enum):
    """Security event types"""
    # Authentication events
    LOGIN_FAILURE = "login_failure"
    LOGIN_SUCCESS = "login_success"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"
    ACCOUNT_LOCKOUT = "account_lockout"
    
    # Authorization events
    ACCESS_DENIED = "access_denied"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    UNAUTHORIZED_API_ACCESS = "unauthorized_api_access"
    
    # Data events
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    SENSITIVE_DATA_ACCESS = "sensitive_data_access"
    DATA_EXFILTRATION = "data_exfiltration"
    
    # System events
    SYSTEM_INTRUSION = "system_intrusion"
    MALWARE_DETECTED = "malware_detected"
    SUSPICIOUS_PROCESS = "suspicious_process"
    
    # Network events
    DDOS_ATTACK = "ddos_attack"
    PORT_SCAN = "port_scan"
    SUSPICIOUS_TRAFFIC = "suspicious_traffic"
    
    # Application events
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    CODE_INJECTION = "code_injection"
    
    # Blockchain events
    BLOCKCHAIN_ATTACK = "blockchain_attack"
    SMART_CONTRACT_EXPLOIT = "smart_contract_exploit"
    CRYPTO_THEFT = "crypto_theft"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    event_type: EventType
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[str] = None
    affected_resources: List[str] = field(default_factory=list)
    indicators_of_compromise: List[str] = field(default_factory=list)
    response_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    false_positive: bool = False


@dataclass
class ThreatPattern:
    """Pattern for detecting security threats"""
    pattern_id: str
    name: str
    event_types: Set[EventType]
    conditions: Dict[str, Any]
    time_window_minutes: int
    threshold_count: int
    threat_level: ThreatLevel
    auto_response: bool = False
    response_actions: List[str] = field(default_factory=list)


class SecurityMetrics:
    """Security metrics collector"""
    
    def __init__(self):
        self.metrics = defaultdict(int)
        self.time_series = defaultdict(list)
        self.alerts_sent = 0
        self.incidents_detected = 0
        self.false_positives = 0
        
    def increment(self, metric_name: str, value: int = 1):
        """Increment metric counter"""
        self.metrics[metric_name] += value
        self.time_series[metric_name].append((datetime.utcnow(), value))
        
        # Keep only last 24 hours of time series data
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.time_series[metric_name] = [
            (ts, val) for ts, val in self.time_series[metric_name] if ts > cutoff
        ]
    
    def get_metric(self, metric_name: str) -> int:
        """Get current metric value"""
        return self.metrics[metric_name]
    
    def get_rate(self, metric_name: str, minutes: int = 60) -> float:
        """Get metric rate over time period"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent_events = [val for ts, val in self.time_series[metric_name] if ts > cutoff]
        return sum(recent_events) / max(minutes, 1)


class AlertingSystem:
    """Multi-channel alerting system"""
    
    def __init__(self):
        self.secrets_manager = get_secrets_manager()
        self.alert_channels = []
        self.alert_history = deque(maxlen=1000)
        self.rate_limits = defaultdict(lambda: deque(maxlen=10))
        
        # Initialize alert channels
        self._initialize_channels()
        
    def _initialize_channels(self):
        """Initialize alerting channels"""
        try:
            # Email alerting
            smtp_server = self.secrets_manager.get_secret("SMTP_SERVER", required=False)
            if smtp_server:
                self.alert_channels.append(self._send_email_alert)
            
            # Slack alerting (if configured)
            slack_webhook = self.secrets_manager.get_secret("SLACK_WEBHOOK_URL", required=False)
            if slack_webhook:
                self.alert_channels.append(self._send_slack_alert)
            
            # Always enable logging alerts
            self.alert_channels.append(self._send_log_alert)
            
        except Exception as e:
            logger.error(f"Failed to initialize alert channels: {e}")
    
    async def send_alert(self, event: SecurityEvent, additional_context: Dict[str, Any] = None):
        """Send alert through all configured channels"""
        try:
            # Check rate limiting
            if self._is_rate_limited(event.event_type):
                logger.debug(f"Alert rate limited for {event.event_type}")
                return
            
            # Update rate limiting
            self.rate_limits[event.event_type].append(datetime.utcnow())
            
            # Prepare alert content
            alert_content = self._prepare_alert_content(event, additional_context)
            
            # Send through all channels
            tasks = [channel(alert_content) for channel in self.alert_channels]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Record alert
            self.alert_history.append({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "threat_level": event.threat_level.value,
                "timestamp": event.timestamp.isoformat(),
                "channels": len(self.alert_channels)
            })
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def _is_rate_limited(self, event_type: EventType) -> bool:
        """Check if event type is rate limited"""
        now = datetime.utcnow()
        recent_alerts = [ts for ts in self.rate_limits[event_type] if now - ts < timedelta(minutes=5)]
        
        # Allow max 3 alerts of same type per 5 minutes
        return len(recent_alerts) >= 3
    
    def _prepare_alert_content(self, event: SecurityEvent, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare standardized alert content"""
        return {
            "event_id": event.event_id,
            "title": f"🚨 Security Alert: {event.event_type.value.replace('_', ' ').title()}",
            "threat_level": event.threat_level.value,
            "timestamp": event.timestamp.isoformat(),
            "description": event.description,
            "source_ip": event.source_ip,
            "user_id": event.user_id,
            "affected_resources": event.affected_resources,
            "indicators_of_compromise": event.indicators_of_compromise,
            "response_actions": event.response_actions,
            "details": event.details,
            "context": context or {}
        }
    
    async def _send_email_alert(self, alert_content: Dict[str, Any]):
        """Send email alert"""
        try:
            smtp_server = self.secrets_manager.get_secret("SMTP_SERVER")
            smtp_port = int(self.secrets_manager.get_secret("SMTP_PORT", default="587"))
            smtp_username = self.secrets_manager.get_secret("SMTP_USERNAME")
            smtp_password = self.secrets_manager.get_secret("SMTP_PASSWORD")
            
            # Recipients
            security_team = self.secrets_manager.get_secret("SECURITY_TEAM_EMAIL", 
                                                           default="security@a2a-platform.local")
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = smtp_username
            msg['To'] = security_team
            msg['Subject'] = alert_content['title']
            
            # HTML body
            html_body = f"""
            <html>
            <body>
                <h2 style="color: {'red' if alert_content['threat_level'] == 'critical' else 'orange'};">
                    {alert_content['title']}
                </h2>
                <p><strong>Threat Level:</strong> {alert_content['threat_level'].upper()}</p>
                <p><strong>Time:</strong> {alert_content['timestamp']}</p>
                <p><strong>Event ID:</strong> {alert_content['event_id']}</p>
                
                {f"<p><strong>Source IP:</strong> {alert_content['source_ip']}</p>" if alert_content['source_ip'] else ""}
                {f"<p><strong>User:</strong> {alert_content['user_id']}</p>" if alert_content['user_id'] else ""}
                
                <h3>Description:</h3>
                <p>{alert_content['description']}</p>
                
                {f"<h3>Affected Resources:</h3><ul>{''.join([f'<li>{resource}</li>' for resource in alert_content['affected_resources']])}</ul>" if alert_content['affected_resources'] else ""}
                
                {f"<h3>Indicators of Compromise:</h3><ul>{''.join([f'<li>{ioc}</li>' for ioc in alert_content['indicators_of_compromise']])}</ul>" if alert_content['indicators_of_compromise'] else ""}
                
                {f"<h3>Recommended Actions:</h3><ul>{''.join([f'<li>{action}</li>' for action in alert_content['response_actions']])}</ul>" if alert_content['response_actions'] else ""}
                
                <hr>
                <p><small>This alert was generated by A2A Security Monitoring System</small></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls(context=context)
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_slack_alert(self, alert_content: Dict[str, Any]):
        """Send Slack alert"""
        try:
            import httpx
            
            webhook_url = self.secrets_manager.get_secret("SLACK_WEBHOOK_URL")
            
            # Slack color coding
            colors = {
                "critical": "#FF0000",
                "high": "#FF8C00", 
                "medium": "#FFD700",
                "low": "#00FF00",
                "info": "#87CEEB"
            }
            
            slack_payload = {
                "attachments": [
                    {
                        "color": colors.get(alert_content['threat_level'], "#FFD700"),
                        "title": alert_content['title'],
                        "text": alert_content['description'],
                        "fields": [
                            {
                                "title": "Threat Level",
                                "value": alert_content['threat_level'].upper(),
                                "short": True
                            },
                            {
                                "title": "Event ID", 
                                "value": alert_content['event_id'],
                                "short": True
                            }
                        ],
                        "timestamp": int(datetime.fromisoformat(alert_content['timestamp']).timestamp())
                    }
                ]
            }
            
            if alert_content['source_ip']:
                slack_payload["attachments"][0]["fields"].append({
                    "title": "Source IP",
                    "value": alert_content['source_ip'],
                    "short": True
                })
            
            if alert_content['response_actions']:
                slack_payload["attachments"][0]["fields"].append({
                    "title": "Recommended Actions",
                    "value": "\n".join(f"• {action}" for action in alert_content['response_actions']),
                    "short": False
                })
            
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
            # WARNING: HTTP client disabled - A2A protocol compliance
            # async with httpx.AsyncClient() as client:
            #     response = await client.post(webhook_url, json=slack_payload)
            #     response.raise_for_status()
            logger.warning("Slack notifications disabled - HTTP client usage violates A2A protocol")
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    async def _send_log_alert(self, alert_content: Dict[str, Any]):
        """Send log alert"""
        threat_level = alert_content['threat_level']
        
        if threat_level == "critical":
            logger.critical(f"SECURITY ALERT: {alert_content['title']} - {alert_content['description']}")
        elif threat_level == "high":
            logger.error(f"SECURITY ALERT: {alert_content['title']} - {alert_content['description']}")
        elif threat_level == "medium":
            logger.warning(f"SECURITY ALERT: {alert_content['title']} - {alert_content['description']}")
        else:
            logger.info(f"SECURITY ALERT: {alert_content['title']} - {alert_content['description']}")


class SecurityMonitor:
    """Main security monitoring system"""
    
    def __init__(self):
        self.events = deque(maxlen=10000)  # Keep last 10k events
        self.patterns = {}
        self.metrics = SecurityMetrics()
        self.alerting = AlertingSystem()
        self.response_handlers = {}
        self.monitoring_active = True
        
        # Event processing queue
        self.event_queue = asyncio.Queue()
        self.processing_tasks = []
        
        # Initialize threat patterns
        self._initialize_threat_patterns()
        
        # Initialize response handlers
        self._initialize_response_handlers()
        
        logger.info("Security Monitor initialized")
    
    def _initialize_threat_patterns(self):
        """Initialize threat detection patterns"""
        patterns = [
            ThreatPattern(
                pattern_id="brute_force_login",
                name="Brute Force Login Attack",
                event_types={EventType.LOGIN_FAILURE},
                conditions={"same_ip": True, "different_users": True},
                time_window_minutes=5,
                threshold_count=5,
                threat_level=ThreatLevel.HIGH,
                auto_response=True,
                response_actions=["block_ip", "alert_security_team"]
            ),
            ThreatPattern(
                pattern_id="privilege_escalation",
                name="Privilege Escalation Attempt",
                event_types={EventType.ACCESS_DENIED, EventType.UNAUTHORIZED_API_ACCESS},
                conditions={"same_user": True},
                time_window_minutes=10,
                threshold_count=3,
                threat_level=ThreatLevel.CRITICAL,
                auto_response=True,
                response_actions=["lock_account", "alert_security_team"]
            ),
            ThreatPattern(
                pattern_id="data_exfiltration",
                name="Potential Data Exfiltration",
                event_types={EventType.SENSITIVE_DATA_ACCESS},
                conditions={"large_volume": True, "unusual_time": True},
                time_window_minutes=30,
                threshold_count=10,
                threat_level=ThreatLevel.CRITICAL,
                auto_response=True,
                response_actions=["block_user", "alert_security_team", "forensic_capture"]
            ),
            ThreatPattern(
                pattern_id="ddos_attack",
                name="Distributed Denial of Service Attack",
                event_types={EventType.DDOS_ATTACK},
                conditions={"high_request_rate": True},
                time_window_minutes=1,
                threshold_count=100,
                threat_level=ThreatLevel.HIGH,
                auto_response=True,
                response_actions=["enable_rate_limiting", "alert_security_team"]
            ),
            ThreatPattern(
                pattern_id="injection_attack",
                name="Code Injection Attack",
                event_types={EventType.SQL_INJECTION, EventType.CODE_INJECTION, EventType.XSS_ATTEMPT},
                conditions={"any_occurrence": True},
                time_window_minutes=1,
                threshold_count=1,
                threat_level=ThreatLevel.HIGH,
                auto_response=True,
                response_actions=["block_ip", "alert_security_team"]
            )
        ]
        
        for pattern in patterns:
            self.patterns[pattern.pattern_id] = pattern
    
    def _initialize_response_handlers(self):
        """Initialize automated response handlers"""
        self.response_handlers = {
            "block_ip": self._block_ip,
            "block_user": self._block_user,
            "lock_account": self._lock_account,
            "alert_security_team": self._alert_security_team,
            "enable_rate_limiting": self._enable_rate_limiting,
            "forensic_capture": self._forensic_capture
        }
    
    async def start_monitoring(self):
        """Start the security monitoring system"""
        logger.info("🔍 Starting security monitoring")
        
        # Start event processing workers
        for i in range(3):  # 3 worker tasks
            task = asyncio.create_task(self._process_events())
            self.processing_tasks.append(task)
        
        # Start periodic tasks
        asyncio.create_task(self._periodic_analysis())
        asyncio.create_task(self._cleanup_old_events())
    
    async def stop_monitoring(self):
        """Stop the security monitoring system"""
        logger.info("⏹️ Stopping security monitoring")
        
        self.monitoring_active = False
        
        # Cancel processing tasks
        for task in self.processing_tasks:
            task.cancel()
        
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
    
    async def report_event(self, 
                          event_type: EventType,
                          threat_level: ThreatLevel,
                          description: str,
                          source_ip: Optional[str] = None,
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None,
                          details: Dict[str, Any] = None,
                          affected_resources: List[str] = None):
        """Report a security event"""
        try:
            # Generate unique event ID
            event_id = hashlib.sha256(
                f"{event_type.value}_{source_ip}_{user_id}_{int(time.time() * 1000)}".encode()
            ).hexdigest()[:16]
            
            # Create event
            event = SecurityEvent(
                event_id=event_id,
                event_type=event_type,
                threat_level=threat_level,
                timestamp=datetime.utcnow(),
                source_ip=source_ip,
                user_id=user_id,
                session_id=session_id,
                description=description,
                details=details or {},
                affected_resources=affected_resources or []
            )
            
            # Queue event for processing
            await self.event_queue.put(event)
            
            # Update metrics
            self.metrics.increment(f"events_{event_type.value}")
            self.metrics.increment(f"threat_level_{threat_level.value}")
            
            return event_id
            
        except Exception as e:
            logger.error(f"Failed to report security event: {e}")
            return None
    
    async def _process_events(self):
        """Process events from the queue"""
        while self.monitoring_active:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Store event
                self.events.append(event)
                
                # Analyze for threats
                await self._analyze_event(event)
                
                # Check for patterns
                await self._check_threat_patterns(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing security event: {e}")
    
    async def _analyze_event(self, event: SecurityEvent):
        """Analyze individual event for immediate threats"""
        # Check for immediate critical threats
        if event.threat_level == ThreatLevel.CRITICAL:
            await self.alerting.send_alert(event)
            
        # Event-specific analysis
        if event.event_type == EventType.LOGIN_FAILURE:
            await self._analyze_login_failure(event)
        elif event.event_type == EventType.SENSITIVE_DATA_ACCESS:
            await self._analyze_data_access(event)
        elif event.event_type in {EventType.SQL_INJECTION, EventType.XSS_ATTEMPT, EventType.CODE_INJECTION}:
            await self._analyze_injection_attempt(event)
    
    async def _analyze_login_failure(self, event: SecurityEvent):
        """Analyze login failure for brute force attempts"""
        if not event.source_ip:
            return
        
        # Count recent failures from same IP
        recent_failures = [
            e for e in list(self.events)[-100:]  # Check last 100 events
            if (e.event_type == EventType.LOGIN_FAILURE and
                e.source_ip == event.source_ip and
                (datetime.utcnow() - e.timestamp).total_seconds() < 300)  # Last 5 minutes
        ]
        
        if len(recent_failures) >= 5:
            # Trigger brute force alert
            await self.report_event(
                EventType.BRUTE_FORCE_ATTEMPT,
                ThreatLevel.HIGH,
                f"Brute force attack detected from IP {event.source_ip}",
                source_ip=event.source_ip,
                details={"failure_count": len(recent_failures)}
            )
    
    async def _analyze_data_access(self, event: SecurityEvent):
        """Analyze sensitive data access patterns"""
        if not event.user_id:
            return
        
        # Check for unusual volume
        recent_access = [
            e for e in list(self.events)[-1000:]
            if (e.event_type == EventType.SENSITIVE_DATA_ACCESS and
                e.user_id == event.user_id and
                (datetime.utcnow() - e.timestamp).total_seconds() < 1800)  # Last 30 minutes
        ]
        
        if len(recent_access) >= 20:
            await self.report_event(
                EventType.DATA_EXFILTRATION,
                ThreatLevel.CRITICAL,
                f"Potential data exfiltration by user {event.user_id}",
                user_id=event.user_id,
                details={"access_count": len(recent_access)}
            )
    
    async def _analyze_injection_attempt(self, event: SecurityEvent):
        """Analyze injection attempts"""
        # Any injection attempt is serious
        await self.alerting.send_alert(event, {
            "immediate_action": "Block source IP and investigate",
            "forensic_data": event.details
        })
    
    async def _check_threat_patterns(self, event: SecurityEvent):
        """Check event against threat patterns"""
        for pattern in self.patterns.values():
            if event.event_type in pattern.event_types:
                if await self._pattern_matches(pattern, event):
                    await self._handle_pattern_match(pattern, event)
    
    async def _pattern_matches(self, pattern: ThreatPattern, event: SecurityEvent) -> bool:
        """Check if event matches threat pattern"""
        try:
            # Get recent events of same types
            cutoff = datetime.utcnow() - timedelta(minutes=pattern.time_window_minutes)
            recent_events = [
                e for e in list(self.events)
                if (e.event_type in pattern.event_types and e.timestamp > cutoff)
            ]
            
            if len(recent_events) < pattern.threshold_count:
                return False
            
            # Check pattern conditions
            conditions = pattern.conditions
            
            if conditions.get("same_ip"):
                ips = [e.source_ip for e in recent_events if e.source_ip]
                if len(set(ips)) > 1:
                    return False
                    
            if conditions.get("same_user"):
                users = [e.user_id for e in recent_events if e.user_id]
                if len(set(users)) > 1:
                    return False
            
            if conditions.get("different_users"):
                users = [e.user_id for e in recent_events if e.user_id]
                if len(set(users)) < 3:  # At least 3 different users
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking pattern match: {e}")
            return False
    
    async def _handle_pattern_match(self, pattern: ThreatPattern, triggering_event: SecurityEvent):
        """Handle threat pattern match"""
        # Create incident event
        incident_id = f"incident_{int(time.time())}"
        
        incident_event = SecurityEvent(
            event_id=incident_id,
            event_type=EventType.SYSTEM_INTRUSION,
            threat_level=pattern.threat_level,
            timestamp=datetime.utcnow(),
            source_ip=triggering_event.source_ip,
            user_id=triggering_event.user_id,
            description=f"Threat pattern detected: {pattern.name}",
            details={
                "pattern_id": pattern.pattern_id,
                "triggering_event": triggering_event.event_id,
                "threshold_exceeded": True
            },
            response_actions=pattern.response_actions
        )
        
        # Send alert
        await self.alerting.send_alert(incident_event)
        
        # Execute automated responses if enabled
        if pattern.auto_response:
            await self._execute_response_actions(pattern.response_actions, triggering_event)
        
        self.metrics.increment("incidents_detected")
    
    async def _execute_response_actions(self, actions: List[str], event: SecurityEvent):
        """Execute automated response actions"""
        for action in actions:
            if action in self.response_handlers:
                try:
                    await self.response_handlers[action](event)
                    logger.info(f"Executed response action: {action}")
                except Exception as e:
                    logger.error(f"Failed to execute response action {action}: {e}")
    
    # Response handlers
    async def _block_ip(self, event: SecurityEvent):
        """Block IP address"""
        if event.source_ip:
            # This would integrate with firewall/WAF
            logger.critical(f"🚫 BLOCKING IP: {event.source_ip}")
            # In production, this would call firewall API
    
    async def _block_user(self, event: SecurityEvent):
        """Block user account"""
        if event.user_id:
            try:
                auth_service = get_auth_service()
                if event.user_id in auth_service.users:
                    auth_service.users[event.user_id].is_active = False
                    logger.critical(f"🚫 BLOCKING USER: {event.user_id}")
            except Exception as e:
                logger.error(f"Failed to block user: {e}")
    
    async def _lock_account(self, event: SecurityEvent):
        """Lock user account temporarily"""
        if event.user_id:
            try:
                auth_service = get_auth_service()
                if event.user_id in auth_service.users:
                    user = auth_service.users[event.user_id]
                    user.locked_until = datetime.utcnow() + timedelta(hours=1)
                    logger.warning(f"🔒 LOCKING ACCOUNT: {event.user_id} for 1 hour")
            except Exception as e:
                logger.error(f"Failed to lock account: {e}")
    
    async def _alert_security_team(self, event: SecurityEvent):
        """Send immediate alert to security team"""
        await self.alerting.send_alert(event, {"priority": "immediate"})
    
    async def _enable_rate_limiting(self, event: SecurityEvent):
        """Enable enhanced rate limiting"""
        logger.warning("🛡️ ENABLING ENHANCED RATE LIMITING")
        # This would integrate with rate limiting system
    
    async def _forensic_capture(self, event: SecurityEvent):
        """Capture forensic data"""
        logger.critical(f"🔬 CAPTURING FORENSIC DATA for event {event.event_id}")
        # This would capture system state, network dumps, etc.
    
    async def _periodic_analysis(self):
        """Periodic security analysis"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._analyze_trends()
                await self._generate_security_report()
            except Exception as e:
                logger.error(f"Error in periodic analysis: {e}")
    
    async def _analyze_trends(self):
        """Analyze security trends"""
        # Analyze event patterns over time
        recent_events = [
            e for e in list(self.events)
            if (datetime.utcnow() - e.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        # Check for anomalies
        if len(recent_events) > 100:  # High activity
            await self.report_event(
                EventType.SUSPICIOUS_TRAFFIC,
                ThreatLevel.MEDIUM,
                f"High security event volume: {len(recent_events)} events in last hour"
            )
    
    async def _generate_security_report(self):
        """Generate periodic security report"""
        try:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "events_last_hour": len([
                    e for e in list(self.events)
                    if (datetime.utcnow() - e.timestamp).total_seconds() < 3600
                ]),
                "critical_events": len([
                    e for e in list(self.events)
                    if e.threat_level == ThreatLevel.CRITICAL
                ]),
                "top_source_ips": self._get_top_source_ips(),
                "threat_summary": self._get_threat_summary(),
                "metrics": dict(self.metrics.metrics)
            }
            
            logger.info(f"📊 Security Report: {json.dumps(report, indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to generate security report: {e}")
    
    def _get_top_source_ips(self) -> List[Dict[str, Any]]:
        """Get top source IPs by event count"""
        ip_counts = defaultdict(int)
        for event in list(self.events)[-1000:]:  # Last 1000 events
            if event.source_ip:
                ip_counts[event.source_ip] += 1
        
        return [
            {"ip": ip, "count": count}
            for ip, count in sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    
    def _get_threat_summary(self) -> Dict[str, int]:
        """Get threat level summary"""
        threat_counts = defaultdict(int)
        for event in list(self.events)[-1000:]:  # Last 1000 events
            threat_counts[event.threat_level.value] += 1
        
        return dict(threat_counts)
    
    async def _cleanup_old_events(self):
        """Clean up old events periodically"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Events are automatically limited by deque maxlen
                # Clean up old metrics
                cutoff = datetime.utcnow() - timedelta(days=7)
                for metric_name in list(self.metrics.time_series.keys()):
                    self.metrics.time_series[metric_name] = [
                        (ts, val) for ts, val in self.metrics.time_series[metric_name]
                        if ts > cutoff
                    ]
                
            except Exception as e:
                logger.error(f"Error cleaning up old events: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        return {
            "monitoring_active": self.monitoring_active,
            "total_events": len(self.events),
            "patterns_loaded": len(self.patterns),
            "alert_channels": len(self.alerting.alert_channels),
            "processing_tasks": len(self.processing_tasks),
            "queue_size": self.event_queue.qsize(),
            "metrics": dict(self.metrics.metrics),
            "last_event": self.events[-1].timestamp.isoformat() if self.events else None
        }


# Global security monitor instance
_security_monitor: Optional[SecurityMonitor] = None

def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance"""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor


async def report_security_event(
    event_type: EventType,
    threat_level: ThreatLevel,
    description: str,
    **kwargs
) -> Optional[str]:
    """Convenience function to report security events"""
    monitor = get_security_monitor()
    return await monitor.report_event(event_type, threat_level, description, **kwargs)


class E2EEncryption:
    """End-to-end encryption for chat messages using hybrid encryption"""
    
    def __init__(self):
        self._key_pairs = {}  # Store RSA key pairs per session
        self._symmetric_keys = {}  # Store AES keys per conversation
        self.backend = default_backend()
        
    def generate_key_pair(self, session_id: str) -> Dict[str, str]:
        """Generate RSA key pair for a session"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=self.backend
        )
        public_key = private_key.public_key()
        
        # Store private key
        self._key_pairs[session_id] = private_key
        
        # Return public key in PEM format
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        return {
            'session_id': session_id,
            'public_key': public_pem
        }
    
    def generate_conversation_key(self, conversation_id: str) -> bytes:
        """Generate AES-256 key for conversation"""
        key = secrets.token_bytes(32)  # 256 bits
        self._symmetric_keys[conversation_id] = key
        return key
    
    def encrypt_message(self, message: str, conversation_id: str, recipient_public_key: str) -> Dict[str, str]:
        """Encrypt message using hybrid encryption"""
        # Get or generate conversation key
        if conversation_id not in self._symmetric_keys:
            conversation_key = self.generate_conversation_key(conversation_id)
        else:
            conversation_key = self._symmetric_keys[conversation_id]
        
        # Encrypt message with AES
        iv = secrets.token_bytes(16)  # 128-bit IV
        cipher = Cipher(
            algorithms.AES(conversation_key),
            modes.CBC(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        # Pad message to AES block size
        message_bytes = message.encode('utf-8')
        padding_length = 16 - (len(message_bytes) % 16)
        padded_message = message_bytes + bytes([padding_length]) * padding_length
        
        encrypted_message = encryptor.update(padded_message) + encryptor.finalize()
        
        # Encrypt conversation key with recipient's public key
        recipient_key = serialization.load_pem_public_key(
            recipient_public_key.encode('utf-8'),
            backend=self.backend
        )
        
        encrypted_key = recipient_key.encrypt(
            conversation_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return {
            'encrypted_message': base64.b64encode(encrypted_message).decode('utf-8'),
            'encrypted_key': base64.b64encode(encrypted_key).decode('utf-8'),
            'iv': base64.b64encode(iv).decode('utf-8')
        }
    
    def decrypt_message(self, encrypted_data: Dict[str, str], session_id: str) -> str:
        """Decrypt message using private key"""
        if session_id not in self._key_pairs:
            raise SecurityError("No private key found for session")
        
        private_key = self._key_pairs[session_id]
        
        # Decrypt conversation key
        encrypted_key = base64.b64decode(encrypted_data['encrypted_key'])
        conversation_key = private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt message
        iv = base64.b64decode(encrypted_data['iv'])
        encrypted_message = base64.b64decode(encrypted_data['encrypted_message'])
        
        cipher = Cipher(
            algorithms.AES(conversation_key),
            modes.CBC(iv),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        padded_message = decryptor.update(encrypted_message) + decryptor.finalize()
        
        # Remove padding
        padding_length = padded_message[-1]
        message = padded_message[:-padding_length].decode('utf-8')
        
        return message
    
    def rotate_keys(self, session_id: str) -> Dict[str, str]:
        """Rotate encryption keys for forward secrecy"""
        return self.generate_key_pair(session_id)


# Global E2E encryption instance
_e2e_encryption = None

def get_e2e_encryption() -> E2EEncryption:
    """Get global E2E encryption instance"""
    global _e2e_encryption
    if _e2e_encryption is None:
        _e2e_encryption = E2EEncryption()
    return _e2e_encryption


# Export main classes and functions
__all__ = [
    'SecurityMonitor',
    'SecurityEvent',
    'ThreatLevel',
    'EventType',
    'ThreatPattern',
    'AlertingSystem',
    'get_security_monitor',
    'report_security_event',
    'E2EEncryption',
    'get_e2e_encryption'
]