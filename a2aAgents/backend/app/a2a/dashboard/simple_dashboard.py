"""
Simple A2A Health Dashboard
Provides health monitoring for A2A services
"""

import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
# A2A Protocol: Use blockchain messaging instead of httpx
from datetime import datetime
import json

app = FastAPI(title="A2A Health Dashboard")

# Health check endpoints for all services
SERVICES = {
    "agents": [
        {"name": "Agent 0 - Data Product", "port": 8001},
        {"name": "Agent 1 - Standardization", "port": 8002},
        {"name": "Agent 2 - AI Preparation", "port": 8003},
        {"name": "Agent 3 - Vector Processing", "port": 8004},
        {"name": "Agent 4 - Calc Validation", "port": 8005},
        {"name": "Agent 5 - QA Validation", "port": 8006},
        {"name": "Agent 6 - Quality Control", "port": 8007},
        {"name": "Reasoning Agent", "port": 8008},
        {"name": "SQL Agent", "port": 8009},
        {"name": "Agent Manager", "port": 8010},
        {"name": "Data Manager", "port": 8011},
        {"name": "Catalog Manager", "port": 8012},
        {"name": "Calculation Agent", "port": 8013},
        {"name": "Agent Builder", "port": 8014},
        {"name": "Embedding Fine-tuner", "port": 8015},
    ],
    "core": [
        {"name": "CAP/CDS Network", "port": 4004},
        {"name": "API Gateway", "port": 8080},
        {"name": "A2A Registry", "port": 8090},
        {"name": "ORD Registry", "port": 8091},
        {"name": "Developer Portal", "port": 3001},
    ],
    "infrastructure": [
        {"name": "Redis", "port": 6379},
        {"name": "Prometheus", "port": 9090},
        {"name": "Blockchain (Anvil)", "port": 8545},
    ],
    "mcp": [
        {"name": "Data Standardization", "port": 8101},
        {"name": "Vector Similarity", "port": 8102},
        {"name": "Vector Ranking", "port": 8103},
        {"name": "Transport Layer", "port": 8104},
        {"name": "Reasoning Agent MCP", "port": 8105},
        {"name": "Session Management", "port": 8106},
        {"name": "Resource Streaming", "port": 8107},
        {"name": "Confidence Calculator", "port": 8108},
        {"name": "Semantic Similarity", "port": 8109},
    ]
}

async def check_service_health(service):
    """Check if a service is healthy"""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"http://localhost:{service['port']}/health")
            if response.status_code == 200:
                return {"name": service["name"], "port": service["port"], "status": "healthy"}
    except:
        pass
    
    # Check if port is at least listening
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            await client.get(f"http://localhost:{service['port']}")
            return {"name": service["name"], "port": service["port"], "status": "running"}
    except:
        return {"name": service["name"], "port": service["port"], "status": "down"}

@app.get("/")
async def dashboard():
    """Serve the dashboard HTML"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>A2A Health Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            h1 { color: #333; }
            .service-group { margin: 20px 0; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .service-group h2 { margin-top: 0; color: #555; }
            .service { display: inline-block; margin: 5px; padding: 8px 15px; border-radius: 4px; font-size: 14px; }
            .healthy { background: #4CAF50; color: white; }
            .running { background: #FF9800; color: white; }
            .down { background: #F44336; color: white; }
            .timestamp { color: #666; font-size: 12px; margin-top: 20px; }
            .summary { margin: 20px 0; font-size: 18px; }
            .refresh-btn { padding: 10px 20px; background: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer; }
            .refresh-btn:hover { background: #1976D2; }
        </style>
    </head>
    <body>
        <h1>A2A System Health Dashboard</h1>
        <button class="refresh-btn" onclick="location.reload()">Refresh</button>
        <div id="summary" class="summary"></div>
        <div id="content"></div>
        <div class="timestamp" id="timestamp"></div>
        
        <script>
            async function fetchHealth() {
                const response = await blockchainClient.sendMessage('/api/health');
                const data = await response.json();
                
                let html = '';
                let totalServices = 0;
                let healthyServices = 0;
                
                for (const [group, services] of Object.entries(data)) {
                    html += `<div class="service-group"><h2>${group.toUpperCase()}</h2>`;
                    for (const service of services) {
                        totalServices++;
                        if (service.status === 'healthy') healthyServices++;
                        html += `<div class="service ${service.status}">${service.name} (:${service.port})</div>`;
                    }
                    html += '</div>';
                }
                
                document.getElementById('content').innerHTML = html;
                document.getElementById('summary').innerHTML = 
                    `System Health: ${healthyServices}/${totalServices} services healthy (${Math.round(healthyServices/totalServices*100)}%)`;
                document.getElementById('timestamp').innerHTML = 
                    `Last updated: ${new Date().toLocaleString()}`;
            }
            
            fetchHealth();
            setInterval(fetchHealth, 5000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "health-dashboard"}

@app.get("/api/health")
async def api_health():
    """Get health status of all services"""
    health_data = {}
    
    for group, services in SERVICES.items():
        health_data[group] = await asyncio.gather(
            *[check_service_health(service) for service in services]
        )
    
    return health_data

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8889))
    uvicorn.run(app, host="0.0.0.0", port=port)