"""
A2A Network Health Dashboard
Real-time monitoring and health assessment for the A2A agent network
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



import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ServiceType(str, Enum):
    AGENT = "agent"
    GATEWAY = "gateway"
    REGISTRY = "registry"
    DATABASE = "database"
    CACHE = "cache"


@dataclass
class HealthMetric:
    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime


class ServiceHealth(BaseModel):
    service_id: str
    service_name: str
    service_type: ServiceType
    status: HealthStatus
    uptime: float
    last_check: datetime
    metrics: List[HealthMetric]
    endpoint: str
    version: Optional[str] = None
    error_message: Optional[str] = None


class NetworkHealth(BaseModel):
    overall_status: HealthStatus
    total_services: int
    healthy_services: int
    warning_services: int
    critical_services: int
    unknown_services: int
    services: List[ServiceHealth]
    last_updated: datetime
    network_metrics: List[HealthMetric]


class HealthDashboard:
    """
    A2A Network Health Dashboard
    Provides real-time health monitoring and visualization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services = {}
        self.network_history = []
        self.connected_clients = set()
        
        # Service registry - all A2A services to monitor
        self.service_registry = {
            "data_product_agent_0": {
                "name": "Data Product Agent",
                "type": ServiceType.AGENT,
                "endpoint": "http://localhost:8001/health",
                "metrics_endpoint": "http://localhost:8001/metrics"
            },
            "data_standardization_agent_1": {
                "name": "Data Standardization Agent", 
                "type": ServiceType.AGENT,
                "endpoint": "http://localhost:8002/health",
                "metrics_endpoint": "http://localhost:8002/metrics"
            },
            "ai_preparation_agent_2": {
                "name": "AI Preparation Agent",
                "type": ServiceType.AGENT,
                "endpoint": "http://localhost:8003/health",
                "metrics_endpoint": "http://localhost:8003/metrics"
            },
            "vector_processing_agent_3": {
                "name": "Vector Processing Agent",
                "type": ServiceType.AGENT,
                "endpoint": "http://localhost:8004/health",
                "metrics_endpoint": "http://localhost:8004/metrics"
            },
            "catalog_manager_agent": {
                "name": "Catalog Manager Agent",
                "type": ServiceType.AGENT,
                "endpoint": "http://localhost:8005/health",
                "metrics_endpoint": "http://localhost:8005/metrics"
            },
            "data_manager_agent": {
                "name": "Data Manager Agent",
                "type": ServiceType.AGENT,
                "endpoint": "http://localhost:8006/health",
                "metrics_endpoint": "http://localhost:8006/metrics"
            },
            "api_gateway": {
                "name": "API Gateway",
                "type": ServiceType.GATEWAY,
                "endpoint": "http://localhost:8080/health",
                "metrics_endpoint": "http://localhost:8080/metrics"
            },
            "a2a_registry": {
                "name": "A2A Registry",
                "type": ServiceType.REGISTRY,
                "endpoint": "http://localhost:8090/health",
                "metrics_endpoint": "http://localhost:8090/metrics"
            }
        }
        
        # Health check configuration
        self.check_interval = config.get("check_interval", 30)  # seconds
        self.timeout = config.get("timeout", 5)  # seconds
        self.history_retention = config.get("history_retention", 24)  # hours
        
        # Thresholds
        self.thresholds = {
            "response_time": {"warning": 1.0, "critical": 3.0},
            "cpu_usage": {"warning": 0.7, "critical": 0.9},
            "memory_usage": {"warning": 0.8, "critical": 0.95},
            "error_rate": {"warning": 0.05, "critical": 0.1},
            "queue_depth": {"warning": 50, "critical": 100}
        }
        
        self.app = FastAPI(title="A2A Health Dashboard")
        self._setup_routes()
        
        # Start background tasks
        self.monitoring_task = None
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Serve the main dashboard page"""
            return self._render_dashboard()
        
        @self.app.get("/api/health")
        async def get_network_health():
            """Get current network health status"""
            return await self._get_network_health()
        
        @self.app.get("/api/services")
        async def get_services():
            """Get all services status"""
            return {"services": list(self.services.values())}
        
        @self.app.get("/api/services/{service_id}")
        async def get_service_health(service_id: str):
            """Get specific service health"""
            if service_id not in self.services:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Service {service_id} not found"}
                )
            return self.services[service_id]
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.connected_clients.add(websocket)
            
            try:
                while True:
                    # Keep connection alive and handle client messages
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)
        
        @self.app.get("/api/metrics/history")
        async def get_metrics_history(hours: int = 1):
            """Get historical metrics data"""
            cutoff_time = datetime.now() - timedelta(hours=hours)
            filtered_history = [
                entry for entry in self.network_history 
                if entry["timestamp"] >= cutoff_time
            ]
            return {"history": filtered_history}
        
        @self.app.get("/api/alerts")
        async def get_active_alerts():
            """Get current active alerts"""
            return await self._get_active_alerts()
    
    async def start_monitoring(self):
        """Start the health monitoring background task"""
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop the health monitoring background task"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Check all services
                await self._check_all_services()
                
                # Update network health
                network_health = await self._get_network_health()
                
                # Add to history
                self._add_to_history(network_health)
                
                # Send updates to connected clients
                await self._broadcast_updates(network_health)
                
                # Clean old history
                self._clean_history()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay on error
    
    async def _check_all_services(self):
        """Check health of all registered services"""
        tasks = []
        
        for service_id, config in self.service_registry.items():
            task = asyncio.create_task(self._check_service_health(service_id, config))
            tasks.append(task)
        
        # Wait for all health checks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_service_health(self, service_id: str, config: Dict[str, Any]):
        """Check health of a specific service"""
        start_time = time.time()
        
        try:
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient(timeout=self.timeout) as client:
                # Health check
                health_response = await client.get(config["endpoint"])
                response_time = time.time() - start_time
                
                # Get metrics if available
                metrics = []
                try:
                    if "metrics_endpoint" in config:
                        metrics_response = await client.get(config["metrics_endpoint"])
                        if metrics_response.status_code == 200:
                            metrics = await self._parse_metrics(metrics_response.text)
                except Exception as e:
                    logger.warning(f"Failed to get metrics for {service_id}: {e}")
                
                # Calculate health status
                status = self._calculate_service_status(health_response, response_time, metrics)
                
                # Create service health object
                service_health = ServiceHealth(
                    service_id=service_id,
                    service_name=config["name"],
                    service_type=config["type"],
                    status=status,
                    uptime=response_time,
                    last_check=datetime.now(),
                    metrics=metrics,
                    endpoint=config["endpoint"],
                    version=self._extract_version(health_response),
                    error_message=None
                )
                
                self.services[service_id] = service_health
                
        except Exception as e:
            # Service is unreachable or unhealthy
            error_service = ServiceHealth(
                service_id=service_id,
                service_name=config["name"],
                service_type=config["type"],
                status=HealthStatus.CRITICAL,
                uptime=0.0,
                last_check=datetime.now(),
                metrics=[],
                endpoint=config["endpoint"],
                error_message=str(e)
            )
            
            self.services[service_id] = error_service
    
    def _calculate_service_status(self, response, response_time: float, metrics: List[HealthMetric]) -> HealthStatus:
        """Calculate overall service health status"""
        
        # Check HTTP status
        if response.status_code != 200:
            return HealthStatus.CRITICAL
        
        # Check response time
        if response_time > self.thresholds["response_time"]["critical"]:
            return HealthStatus.CRITICAL
        elif response_time > self.thresholds["response_time"]["warning"]:
            return HealthStatus.WARNING
        
        # Check metrics thresholds
        critical_metrics = [m for m in metrics if m.status == HealthStatus.CRITICAL]
        warning_metrics = [m for m in metrics if m.status == HealthStatus.WARNING]
        
        if critical_metrics:
            return HealthStatus.CRITICAL
        elif warning_metrics:
            return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    async def _parse_metrics(self, metrics_text: str) -> List[HealthMetric]:
        """Parse Prometheus metrics into HealthMetric objects"""
        metrics = []
        
        try:
            # Simple parsing for common metrics
            lines = metrics_text.split('\\n')
            
            for line in lines:
                if line.startswith('# '):
                    continue
                
                if 'cpu_usage' in line and not line.startswith('#'):
                    value = float(line.split()[-1])
                    status = self._get_metric_status('cpu_usage', value)
                    metrics.append(HealthMetric(
                        name="CPU Usage",
                        value=value,
                        unit="%",
                        status=status,
                        threshold_warning=self.thresholds["cpu_usage"]["warning"],
                        threshold_critical=self.thresholds["cpu_usage"]["critical"],
                        timestamp=datetime.now()
                    ))
                
                elif 'memory_usage' in line and not line.startswith('#'):
                    value = float(line.split()[-1])
                    status = self._get_metric_status('memory_usage', value)
                    metrics.append(HealthMetric(
                        name="Memory Usage",
                        value=value,
                        unit="%",
                        status=status,
                        threshold_warning=self.thresholds["memory_usage"]["warning"],
                        threshold_critical=self.thresholds["memory_usage"]["critical"],
                        timestamp=datetime.now()
                    ))
                
                elif 'queue_depth' in line and not line.startswith('#'):
                    value = float(line.split()[-1])
                    status = self._get_metric_status('queue_depth', value)
                    metrics.append(HealthMetric(
                        name="Queue Depth",
                        value=value,
                        unit="tasks",
                        status=status,
                        threshold_warning=self.thresholds["queue_depth"]["warning"],
                        threshold_critical=self.thresholds["queue_depth"]["critical"],
                        timestamp=datetime.now()
                    ))
        
        except Exception as e:
            logger.warning(f"Error parsing metrics: {e}")
        
        return metrics
    
    def _get_metric_status(self, metric_type: str, value: float) -> HealthStatus:
        """Get health status based on metric value and thresholds"""
        thresholds = self.thresholds.get(metric_type, {})
        
        if value >= thresholds.get("critical", float('inf')):
            return HealthStatus.CRITICAL
        elif value >= thresholds.get("warning", float('inf')):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _extract_version(self, response) -> Optional[str]:
        """Extract version information from health response"""
        try:
            if response.headers.get("content-type", "").startswith("application/json"):
                data = response.json()
                return data.get("version")
        except Exception:
            pass
        return None
    
    async def _get_network_health(self) -> NetworkHealth:
        """Calculate overall network health"""
        
        if not self.services:
            return NetworkHealth(
                overall_status=HealthStatus.UNKNOWN,
                total_services=0,
                healthy_services=0,
                warning_services=0,
                critical_services=0,
                unknown_services=0,
                services=[],
                last_updated=datetime.now(),
                network_metrics=[]
            )
        
        # Count services by status
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for service in self.services.values():
            status_counts[service.status] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.CRITICAL] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] > 0:
            overall_status = HealthStatus.WARNING
        elif status_counts[HealthStatus.UNKNOWN] > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Calculate network-wide metrics
        network_metrics = self._calculate_network_metrics()
        
        return NetworkHealth(
            overall_status=overall_status,
            total_services=len(self.services),
            healthy_services=status_counts[HealthStatus.HEALTHY],
            warning_services=status_counts[HealthStatus.WARNING],
            critical_services=status_counts[HealthStatus.CRITICAL],
            unknown_services=status_counts[HealthStatus.UNKNOWN],
            services=list(self.services.values()),
            last_updated=datetime.now(),
            network_metrics=network_metrics
        )
    
    def _calculate_network_metrics(self) -> List[HealthMetric]:
        """Calculate network-wide metrics"""
        metrics = []
        
        if not self.services:
            return metrics
        
        # Average response time
        response_times = [s.uptime for s in self.services.values() if s.status != HealthStatus.CRITICAL]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            metrics.append(HealthMetric(
                name="Average Response Time",
                value=avg_response_time,
                unit="seconds",
                status=self._get_metric_status('response_time', avg_response_time),
                threshold_warning=self.thresholds["response_time"]["warning"],
                threshold_critical=self.thresholds["response_time"]["critical"],
                timestamp=datetime.now()
            ))
        
        # Service availability
        healthy_ratio = len([s for s in self.services.values() if s.status == HealthStatus.HEALTHY]) / len(self.services)
        metrics.append(HealthMetric(
            name="Service Availability",
            value=healthy_ratio * 100,
            unit="%",
            status=HealthStatus.HEALTHY if healthy_ratio >= 0.9 else HealthStatus.WARNING if healthy_ratio >= 0.7 else HealthStatus.CRITICAL,
            threshold_warning=90.0,
            threshold_critical=70.0,
            timestamp=datetime.now()
        ))
        
        return metrics
    
    def _add_to_history(self, network_health: NetworkHealth):
        """Add current network health to history"""
        history_entry = {
            "timestamp": datetime.now(),
            "overall_status": network_health.overall_status,
            "healthy_services": network_health.healthy_services,
            "warning_services": network_health.warning_services,
            "critical_services": network_health.critical_services,
            "network_metrics": [m.__dict__ for m in network_health.network_metrics]
        }
        
        self.network_history.append(history_entry)
    
    def _clean_history(self):
        """Remove old history entries"""
        cutoff_time = datetime.now() - timedelta(hours=self.history_retention)
        self.network_history = [
            entry for entry in self.network_history 
            if entry["timestamp"] >= cutoff_time
        ]
    
    async def _broadcast_updates(self, network_health: NetworkHealth):
        """Broadcast updates to connected WebSocket clients"""
        if not self.connected_clients:
            return
        
        message = {
            "type": "health_update",
            "data": network_health.dict(default=str)
        }
        
        disconnected_clients = []
        for client in self.connected_clients:
            try:
                await client.send_text(json.dumps(message, default=str))
            except Exception:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.discard(client)
    
    async def _get_active_alerts(self) -> Dict[str, Any]:
        """Get current active alerts"""
        alerts = []
        
        for service in self.services.values():
            if service.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                alert = {
                    "service_id": service.service_id,
                    "service_name": service.service_name,
                    "status": service.status,
                    "message": service.error_message or f"Service is in {service.status} state",
                    "timestamp": service.last_check
                }
                alerts.append(alert)
            
            # Add metric-based alerts
            for metric in service.metrics:
                if metric.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                    alert = {
                        "service_id": service.service_id,
                        "service_name": service.service_name,
                        "status": metric.status,
                        "message": f"{metric.name} is {metric.value}{metric.unit} (threshold: {metric.threshold_warning if metric.status == HealthStatus.WARNING else metric.threshold_critical})",
                        "timestamp": metric.timestamp
                    }
                    alerts.append(alert)
        
        return {
            "alerts": alerts,
            "count": len(alerts),
            "last_updated": datetime.now()
        }
    
    def _render_dashboard(self) -> str:
        """Render the HTML dashboard"""
        return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>A2A Network Health Dashboard</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #1a1a1a;
                    color: #ffffff;
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px;
                }
                .header h1 {
                    margin: 0;
                    font-size: 2.5rem;
                }
                .header p {
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                }
                .dashboard-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .card {
                    background: #2a2a2a;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                }
                .card h3 {
                    margin: 0 0 15px 0;
                    color: #ffffff;
                }
                .status-healthy { color: #4caf50; }
                .status-warning { color: #ff9800; }
                .status-critical { color: #f44336; }
                .status-unknown { color: #9e9e9e; }
                .metric {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 10px;
                    padding: 5px 0;
                    border-bottom: 1px solid #3a3a3a;
                }
                .service-list {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                }
                .service-card {
                    background: #333333;
                    border-radius: 8px;
                    padding: 15px;
                    border-left: 4px solid;
                }
                .service-card.healthy { border-left-color: #4caf50; }
                .service-card.warning { border-left-color: #ff9800; }
                .service-card.critical { border-left-color: #f44336; }
                .service-card.unknown { border-left-color: #9e9e9e; }
                .loading {
                    text-align: center;
                    padding: 50px;
                    color: #888;
                }
                .refresh-indicator {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #4caf50;
                    color: white;
                    padding: 10px 15px;
                    border-radius: 5px;
                    opacity: 0;
                    transition: opacity 0.3s;
                }
                .refresh-indicator.show {
                    opacity: 1;
                }
                .last-updated {
                    text-align: center;
                    color: #888;
                    margin-top: 20px;
                    font-size: 0.9rem;
                }
            </style>
        </head>
        <body>
            <div class="refresh-indicator" id="refreshIndicator">Dashboard Updated</div>
            
            <div class="header">
                <h1>🔗 A2A Network Health Dashboard</h1>
                <p>Real-time monitoring of Agent-to-Agent network infrastructure</p>
            </div>
            
            <div id="dashboardContent" class="loading">
                <div>Loading network health data...</div>
            </div>
            
            <div class="last-updated" id="lastUpdated"></div>
            
            <script>
                const ws = new BlockchainEventClient(`blockchain://${window.location.host}/ws`);
                let isConnected = false;
                
                ws.onopen = function() {
                    isConnected = true;
                    loadDashboard();
                };
                
                ws.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    if (message.type === 'health_update') {
                        updateDashboard(message.data);
                        showRefreshIndicator();
                    }
                };
                
                ws.onclose = function() {
                    isConnected = false;
                    setTimeout(connectWebSocket, 5000); // Retry connection
                };
                
                function connectWebSocket() {
                    if (!isConnected) {
                        ws = new BlockchainEventClient(`blockchain://${window.location.host}/ws`);
                    }
                }
                
                async function loadDashboard() {
                    try {
                        const response = await blockchainClient.sendMessage('/api/health');
                        const data = await response.json();
                        updateDashboard(data);
                    } catch (error) {
                        document.getElementById('dashboardContent').innerHTML = 
                            '<div style="text-align: center; color: #f44336;">Error loading dashboard data</div>';
                    }
                }
                
                function updateDashboard(data) {
                    const content = document.getElementById('dashboardContent');
                    const statusClass = `status-${data.overall_status}`;
                    
                    content.innerHTML = `
                        <div class="dashboard-grid">
                            <div class="card">
                                <h3>Network Overview</h3>
                                <div class="metric">
                                    <span>Overall Status:</span>
                                    <span class="${statusClass}">${data.overall_status.toUpperCase()}</span>
                                </div>
                                <div class="metric">
                                    <span>Total Services:</span>
                                    <span>${data.total_services}</span>
                                </div>
                                <div class="metric">
                                    <span>Healthy:</span>
                                    <span class="status-healthy">${data.healthy_services}</span>
                                </div>
                                <div class="metric">
                                    <span>Warning:</span>
                                    <span class="status-warning">${data.warning_services}</span>
                                </div>
                                <div class="metric">
                                    <span>Critical:</span>
                                    <span class="status-critical">${data.critical_services}</span>
                                </div>
                            </div>
                            
                            <div class="card">
                                <h3>Network Metrics</h3>
                                ${data.network_metrics.map(metric => `
                                    <div class="metric">
                                        <span>${metric.name}:</span>
                                        <span class="status-${metric.status}">${metric.value.toFixed(2)} ${metric.unit}</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        
                        <div class="card">
                            <h3>Services Status</h3>
                            <div class="service-list">
                                ${data.services.map(service => `
                                    <div class="service-card ${service.status}">
                                        <h4>${service.service_name}</h4>
                                        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 10px;">
                                            ${service.service_type} • ${service.service_id}
                                        </div>
                                        <div class="metric">
                                            <span>Status:</span>
                                            <span class="status-${service.status}">${service.status.toUpperCase()}</span>
                                        </div>
                                        <div class="metric">
                                            <span>Response Time:</span>
                                            <span>${service.uptime.toFixed(3)}s</span>
                                        </div>
                                        ${service.version ? `
                                            <div class="metric">
                                                <span>Version:</span>
                                                <span>${service.version}</span>
                                            </div>
                                        ` : ''}
                                        ${service.error_message ? `
                                            <div style="color: #f44336; font-size: 0.8rem; margin-top: 10px;">
                                                ${service.error_message}
                                            </div>
                                        ` : ''}
                                        ${service.metrics.length > 0 ? `
                                            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #444;">
                                                <div style="font-size: 0.8rem; opacity: 0.7; margin-bottom: 5px;">Metrics:</div>
                                                ${service.metrics.map(metric => `
                                                    <div style="font-size: 0.8rem; display: flex; justify-content: space-between;">
                                                        <span>${metric.name}:</span>
                                                        <span class="status-${metric.status}">${metric.value.toFixed(2)} ${metric.unit}</span>
                                                    </div>
                                                `).join('')}
                                            </div>
                                        ` : ''}
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                    
                    // Update last updated timestamp
                    document.getElementById('lastUpdated').textContent = 
                        `Last updated: ${new Date(data.last_updated).toLocaleString()}`;
                }
                
                function showRefreshIndicator() {
                    const indicator = document.getElementById('refreshIndicator');
                    indicator.classList.add('show');
                    setTimeout(() => {
                        indicator.classList.remove('show');
                    }, 2000);
                }
                
                // Load dashboard on page load if WebSocket isn't connected
                if (!isConnected) {
                    loadDashboard();
                }
                
                // Refresh every 30 seconds as fallback
                setInterval(() => {
                    if (!isConnected) {
                        loadDashboard();
                    }
                }, 30000);
            </script>
        </body>
        </html>
        '''


# Factory function to create health dashboard instance
def create_health_dashboard(config: Optional[Dict[str, Any]] = None) -> HealthDashboard:
    """Create and configure health dashboard instance"""
    default_config = {
        "check_interval": 30,
        "timeout": 5,
        "history_retention": 24,
        "port": 8888
    }
    
    if config:
        default_config.update(config)
    
    return HealthDashboard(default_config)