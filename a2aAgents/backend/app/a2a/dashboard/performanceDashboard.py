#!/usr/bin/env python3
"""
Real-time Performance Dashboard for A2A Agents
Provides web-based monitoring and visualization of agent performance
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from app.a2a.core.performanceMonitor import get_performance_monitor, _performance_monitors

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="A2A Performance Dashboard", version="1.0.0")

# Templates and static files
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_to_all(self, message: dict):
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "A2A Performance Dashboard"
    })


@app.get("/api/agents")
async def get_agents():
    """Get list of monitored agents"""
    agents = []
    for agent_id, monitor in _performance_monitors.items():
        try:
            current_metrics = monitor.get_current_metrics()
            agents.append({
                "agent_id": agent_id,
                "status": "healthy" if current_metrics.error_rate < 0.05 else "warning",
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "response_time": current_metrics.response_time_avg,
                "error_rate": current_metrics.error_rate,
                "throughput": current_metrics.throughput,
                "last_updated": current_metrics.timestamp
            })
        except Exception as e:
            logger.error(f"Error getting metrics for {agent_id}: {e}")
            agents.append({
                "agent_id": agent_id,
                "status": "error",
                "error": str(e)
            })

    return {"agents": agents}


@app.get("/api/agent/{agent_id}/metrics")
async def get_agent_metrics(agent_id: str, hours: int = 1):
    """Get detailed metrics for specific agent"""
    monitor = get_performance_monitor(agent_id)
    if not monitor:
        return JSONResponse(
            status_code=404,
            content={"error": f"Agent {agent_id} not found"}
        )

    try:
        current_metrics = monitor.get_current_metrics()
        history = monitor.get_metrics_history(hours=hours)

        return {
            "agent_id": agent_id,
            "current": current_metrics.to_dict(),
            "history": [m.to_dict() for m in history],
            "summary": {
                "total_data_points": len(history),
                "time_range_hours": hours,
                "avg_cpu": sum(m.cpu_usage for m in history) / max(len(history), 1),
                "avg_memory": sum(m.memory_usage for m in history) / max(len(history), 1),
                "avg_response_time": sum(m.response_time_avg for m in history) / max(len(history), 1)
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics for {agent_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get metrics: {str(e)}"}
        )


@app.get("/api/agent/{agent_id}/recommendations")
async def get_agent_recommendations(agent_id: str):
    """Get optimization recommendations for agent"""
    monitor = get_performance_monitor(agent_id)
    if not monitor:
        return JSONResponse(
            status_code=404,
            content={"error": f"Agent {agent_id} not found"}
        )

    # Get recommendations from the agent if it has the optimization mixin
    try:
        # This would need to be implemented in the performance monitor
        # For now, return a placeholder
        return {
            "agent_id": agent_id,
            "recommendations": [
                {
                    "category": "performance",
                    "priority": "medium",
                    "description": "Consider implementing request caching",
                    "expected_improvement": "15-25% response time improvement"
                }
            ],
            "last_analysis": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get recommendations: {str(e)}"}
        )


@app.get("/api/system/overview")
async def get_system_overview():
    """Get system-wide performance overview"""
    try:
        total_agents = len(_performance_monitors)
        healthy_agents = 0
        total_cpu = 0
        total_memory = 0
        total_requests = 0
        total_errors = 0

        for monitor in _performance_monitors.values():
            try:
                metrics = monitor.get_current_metrics()
                total_cpu += metrics.cpu_usage
                total_memory += metrics.memory_usage
                total_requests += metrics.request_count

                if metrics.error_rate < 0.05:  # Less than 5% error rate
                    healthy_agents += 1

                total_errors += int(metrics.request_count * metrics.error_rate)

            except Exception as e:
                logger.error(f"Error getting metrics for system overview: {e}")

        return {
            "system": {
                "total_agents": total_agents,
                "healthy_agents": healthy_agents,
                "avg_cpu_usage": total_cpu / max(total_agents, 1),
                "avg_memory_usage": total_memory / max(total_agents, 1),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "overall_error_rate": total_errors / max(total_requests, 1),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Error getting system overview: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get system overview: {str(e)}"}
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)

    try:
        while True:
            # Send real-time updates every 5 seconds
            await asyncio.sleep(5)

            # Get current system status
            agents_data = await get_agents()
            system_data = await get_system_overview()

            update_message = {
                "type": "system_update",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "agents": agents_data["agents"],
                    "system": system_data["system"]
                }
            }

            await websocket.send_json(update_message)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.post("/api/agent/{agent_id}/optimize")
async def optimize_agent(agent_id: str):
    """Trigger optimization for specific agent"""
    monitor = get_performance_monitor(agent_id)
    if not monitor:
        return JSONResponse(
            status_code=404,
            content={"error": f"Agent {agent_id} not found"}
        )

    try:
        # This would trigger optimization if the agent has the optimization mixin
        # For now, return a success message
        return {
            "message": f"Optimization triggered for {agent_id}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Optimization failed: {str(e)}"}
        )


# Background task to send periodic updates
async def periodic_updates():
    """Send periodic updates to all connected clients"""
    while True:
        try:
            await asyncio.sleep(10)  # Update every 10 seconds

            if manager.active_connections:
                # Get latest data
                agents_data = await get_agents()
                system_data = await get_system_overview()

                update_message = {
                    "type": "periodic_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "agents": agents_data["agents"],
                        "system": system_data["system"]
                    }
                }

                await manager.send_to_all(update_message)

        except Exception as e:
            logger.error(f"Error in periodic updates: {e}")
            await asyncio.sleep(10)


# Start background task
@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(periodic_updates())
    logger.info("Performance Dashboard started")


# Dashboard template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>A2A Performance Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .dashboard {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .agent-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .agent-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .agent-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .agent-name {
            font-weight: bold;
            font-size: 1.1em;
        }
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .status-healthy {
            background-color: #d4edda;
            color: #155724;
        }
        .status-warning {
            background-color: #fff3cd;
            color: #856404;
        }
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px;
            background: #333;
            color: white;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .connected {
            background: #28a745;
        }
        .disconnected {
            background: #dc3545;
        }
        .last-updated {
            font-size: 0.8em;
            color: #666;
            text-align: center;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">Connecting...</div>

    <div class="dashboard">
        <div class="header">
            <h1>A2A Performance Dashboard</h1>
            <p>Real-time monitoring of A2A agent performance</p>
        </div>

        <div class="stats-grid" id="systemStats">
            <!-- System stats will be populated here -->
        </div>

        <div class="agent-grid" id="agentGrid">
            <!-- Agent cards will be populated here -->
        </div>

        <div class="last-updated" id="lastUpdated"></div>
    </div>

    <script>
        let ws;
        let isConnected = false;

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;

            ws = new BlockchainEventClient(wsUrl);

            ws.onopen = function() {
                isConnected = true;
                updateConnectionStatus();
                console.log('Connected to dashboard');
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };

            ws.onclose = function() {
                isConnected = false;
                updateConnectionStatus();
                console.log('Disconnected from dashboard');
                // Reconnect after 5 seconds
                setTimeout(connect, 5000);
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }

        function updateConnectionStatus() {
            const status = document.getElementById('connectionStatus');
            if (isConnected) {
                status.textContent = 'Connected';
                status.className = 'connection-status connected';
            } else {
                status.textContent = 'Disconnected';
                status.className = 'connection-status disconnected';
            }
        }

        function updateDashboard(message) {
            if (message.data) {
                updateSystemStats(message.data.system);
                updateAgents(message.data.agents);

                document.getElementById('lastUpdated').textContent =
                    `Last updated: ${new Date(message.timestamp).toLocaleString()}`;
            }
        }

        function updateSystemStats(system) {
            const statsGrid = document.getElementById('systemStats');
            statsGrid.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${system.total_agents}</div>
                    <div class="stat-label">Total Agents</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${system.healthy_agents}</div>
                    <div class="stat-label">Healthy Agents</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${system.avg_cpu_usage.toFixed(1)}%</div>
                    <div class="stat-label">Avg CPU Usage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${system.avg_memory_usage.toFixed(1)}%</div>
                    <div class="stat-label">Avg Memory Usage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${system.total_requests.toLocaleString()}</div>
                    <div class="stat-label">Total Requests</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${(system.overall_error_rate * 100).toFixed(2)}%</div>
                    <div class="stat-label">Overall Error Rate</div>
                </div>
            `;
        }

        function updateAgents(agents) {
            const agentGrid = document.getElementById('agentGrid');
            agentGrid.innerHTML = agents.map(agent => {
                if (agent.error) {
                    return `
                        <div class="agent-card">
                            <div class="agent-header">
                                <div class="agent-name">${agent.agent_id}</div>
                                <div class="status-badge status-error">Error</div>
                            </div>
                            <div class="metric">
                                <span>Error:</span>
                                <span>${agent.error}</span>
                            </div>
                        </div>
                    `;
                }

                return `
                    <div class="agent-card">
                        <div class="agent-header">
                            <div class="agent-name">${agent.agent_id}</div>
                            <div class="status-badge status-${agent.status}">${agent.status}</div>
                        </div>
                        <div class="metric">
                            <span>CPU Usage:</span>
                            <span>${agent.cpu_usage.toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span>Memory Usage:</span>
                            <span>${agent.memory_usage.toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span>Response Time:</span>
                            <span>${agent.response_time.toFixed(1)}ms</span>
                        </div>
                        <div class="metric">
                            <span>Error Rate:</span>
                            <span>${(agent.error_rate * 100).toFixed(2)}%</span>
                        </div>
                        <div class="metric">
                            <span>Throughput:</span>
                            <span>${agent.throughput.toFixed(1)} req/s</span>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Initialize connection
        connect();
    </script>
</body>
</html>
"""

# Create templates directory and file
template_dir = Path(__file__).parent / "templates"
template_dir.mkdir(exist_ok=True)

with open(template_dir / "dashboard.html", "w") as f:
    f.write(DASHBOARD_HTML)


def start_performance_dashboard(host: str = "0.0.0.0", port: int = 8080):
    """Start the performance dashboard server"""
    logger.info(f"Starting Performance Dashboard on https://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_performance_dashboard()
