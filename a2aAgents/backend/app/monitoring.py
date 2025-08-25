"""
Monitoring Dashboard API for A2A Platform
Provides real-time metrics and status information
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import psutil
import os
import asyncio
from typing import Dict, List, Any
import aiohttp

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

# Agent ports configuration
AGENT_PORTS = {
    "agent0": 8000,
    "agent1": 8001,
    "agent2": 8002,
    "agent3": 8003,
    "agent4": 8004,
    "agent5": 8005,
    "agent6": 8006,
    "agent7": 8007,
    "agent8": 8008,
    "agent9": 8009,
    "agent10": 8010,
    "agent11": 8011,
    "agent12": 8012,
    "agent13": 8013,
    "agent14": 8014,
    "agent15": 8015,
    "agent16": 8016,
    "agent17": 8017,
}

CORE_SERVICES = {
    "frontend": {"port": 3000, "name": "Launch Pad UI"},
    "network": {"port": 4004, "name": "A2A Network API"},
    "main": {"port": 8888, "name": "Main Service"},
}

async def check_service_health(url: str, timeout: int = 5) -> Dict[str, Any]:
    """Check if a service is healthy"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status == 200:
                    return {"status": "healthy", "response_time": timeout}
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}
    except asyncio.TimeoutError:
        return {"status": "timeout", "error": "Request timed out"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/dashboard")
async def monitoring_dashboard():
    """Get comprehensive monitoring dashboard data"""
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Process metrics
    process = psutil.Process()
    process_info = {
        "pid": process.pid,
        "cpu_percent": process.cpu_percent(interval=0.1),
        "memory_mb": process.memory_info().rss / (1024 * 1024),
        "threads": process.num_threads(),
        "open_files": len(process.open_files()),
    }
    
    # Check agent health status
    agent_health = {}
    for agent_name, port in AGENT_PORTS.items():
        health_url = f"http://localhost:{port}/health"
        agent_health[agent_name] = await check_service_health(health_url)
    
    # Check core services
    service_health = {}
    for service_name, config in CORE_SERVICES.items():
        health_url = f"http://localhost:{config['port']}/health"
        service_health[service_name] = {
            "name": config["name"],
            "port": config["port"],
            "health": await check_service_health(health_url)
        }
    
    # Calculate summary statistics
    healthy_agents = sum(1 for status in agent_health.values() if status["status"] == "healthy")
    healthy_services = sum(1 for svc in service_health.values() if svc["health"]["status"] == "healthy")
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "operational" if healthy_agents >= 12 and healthy_services >= 2 else "degraded",
        "summary": {
            "agents": {
                "total": len(AGENT_PORTS),
                "healthy": healthy_agents,
                "unhealthy": len(AGENT_PORTS) - healthy_agents
            },
            "services": {
                "total": len(CORE_SERVICES),
                "healthy": healthy_services,
                "unhealthy": len(CORE_SERVICES) - healthy_services
            }
        },
        "system": {
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count()
            },
            "memory": {
                "percent": memory.percent,
                "available_mb": memory.available / (1024 * 1024),
                "total_mb": memory.total / (1024 * 1024)
            },
            "disk": {
                "percent": disk.percent,
                "free_gb": disk.free / (1024 * 1024 * 1024),
                "total_gb": disk.total / (1024 * 1024 * 1024)
            }
        },
        "process": process_info,
        "agents": agent_health,
        "services": service_health,
        "environment": {
            "a2a_environment": os.getenv("A2A_ENVIRONMENT", "unknown"),
            "startup_mode": os.getenv("STARTUP_MODE", "unknown"),
            "container": os.getenv("A2A_IN_CONTAINER", "false") == "true"
        }
    }

@router.get("/metrics")
async def get_metrics():
    """Get Prometheus-compatible metrics"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    metrics = []
    
    # System metrics
    metrics.append(f'# HELP a2a_cpu_usage_percent CPU usage percentage')
    metrics.append(f'# TYPE a2a_cpu_usage_percent gauge')
    metrics.append(f'a2a_cpu_usage_percent {cpu_percent}')
    
    metrics.append(f'# HELP a2a_memory_usage_percent Memory usage percentage')
    metrics.append(f'# TYPE a2a_memory_usage_percent gauge')
    metrics.append(f'a2a_memory_usage_percent {memory.percent}')
    
    metrics.append(f'# HELP a2a_memory_available_bytes Available memory in bytes')
    metrics.append(f'# TYPE a2a_memory_available_bytes gauge')
    metrics.append(f'a2a_memory_available_bytes {memory.available}')
    
    # Agent health metrics
    for agent_name, port in AGENT_PORTS.items():
        health_url = f"http://localhost:{port}/health"
        status = await check_service_health(health_url, timeout=2)
        health_value = 1 if status["status"] == "healthy" else 0
        
        metrics.append(f'# HELP a2a_agent_health Agent health status (1=healthy, 0=unhealthy)')
        metrics.append(f'# TYPE a2a_agent_health gauge')
        metrics.append(f'a2a_agent_health{{agent="{agent_name}",port="{port}"}} {health_value}')
    
    return "\n".join(metrics)

@router.get("/agents/status")
async def get_agents_status():
    """Get detailed status of all agents"""
    agent_details = []
    
    for agent_name, port in AGENT_PORTS.items():
        health_url = f"http://localhost:{port}/health"
        docs_url = f"http://localhost:{port}/docs"
        
        agent_info = {
            "name": agent_name,
            "port": port,
            "urls": {
                "health": health_url,
                "docs": docs_url,
                "api": f"http://localhost:{port}"
            },
            "health": await check_service_health(health_url),
            "agent_id": int(agent_name.replace("agent", "")) if agent_name.startswith("agent") else -1
        }
        
        agent_details.append(agent_info)
    
    # Sort by agent ID
    agent_details.sort(key=lambda x: x["agent_id"])
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_agents": len(agent_details),
        "agents": agent_details
    }

@router.get("/alerts")
async def get_alerts():
    """Get current system alerts"""
    alerts = []
    
    # Check CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    if cpu_percent > 80:
        alerts.append({
            "severity": "warning" if cpu_percent < 90 else "critical",
            "type": "cpu_high",
            "message": f"CPU usage is high: {cpu_percent}%",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # Check memory usage
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        alerts.append({
            "severity": "warning" if memory.percent < 90 else "critical",
            "type": "memory_high",
            "message": f"Memory usage is high: {memory.percent}%",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # Check agent health
    unhealthy_agents = []
    for agent_name, port in AGENT_PORTS.items():
        health_url = f"http://localhost:{port}/health"
        status = await check_service_health(health_url, timeout=2)
        if status["status"] != "healthy":
            unhealthy_agents.append(agent_name)
    
    if unhealthy_agents:
        alerts.append({
            "severity": "warning",
            "type": "agents_unhealthy",
            "message": f"Unhealthy agents detected: {', '.join(unhealthy_agents)}",
            "timestamp": datetime.utcnow().isoformat(),
            "details": unhealthy_agents
        })
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "alert_count": len(alerts),
        "alerts": alerts
    }