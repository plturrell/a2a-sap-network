#!/bin/bash

# Start A2A Services with OpenTelemetry

echo "🚀 Starting A2A Services with OpenTelemetry..."

# Function to check if a service is running
check_service() {
    local port=$1
    local service_name=$2
    if lsof -i :$port >/dev/null 2>&1; then
        echo "✅ $service_name is running on port $port"
        return 0
    else
        echo "❌ $service_name is not running on port $port"
        return 1
    fi
}

# Start OpenTelemetry infrastructure first
echo "📊 Starting OpenTelemetry infrastructure..."
docker-compose -f docker-compose.telemetry.yml up -d

# Wait for OTEL collector to be ready
echo "⏳ Waiting for OpenTelemetry Collector..."
sleep 5

# Check OTEL collector health
if curl -s http://localhost:13133/health >/dev/null 2>&1; then
    echo "✅ OpenTelemetry Collector is healthy"
else
    echo "❌ OpenTelemetry Collector health check failed"
fi

# Load telemetry environment variables
if [ -f .env.telemetry ]; then
    export $(cat .env.telemetry | grep -v '^#' | xargs)
    echo "✅ Loaded telemetry configuration"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start Redis if not running
if ! check_service 6379 "Redis"; then
    echo "🔴 Starting Redis..."
    redis-server &
    sleep 2
fi

# Start Registry Service (required by all agents)
echo "📋 Starting Registry Service..."
OTEL_SERVICE_NAME=a2a-registry python app/a2a_registry/run_registry_server.py > logs/registry.log 2>&1 &
sleep 3
check_service 8100 "Registry Service"

# Start Agent Manager (required for agent coordination)
echo "👥 Starting Agent Manager..."
OTEL_SERVICE_NAME=a2a-agent-manager python launch_agent_manager.py > logs/agent_manager.log 2>&1 &
sleep 3
check_service 8010 "Agent Manager"

# Start Agent 0 (Data Product Registration)
echo "📦 Starting Agent 0 (Data Product Registration)..."
OTEL_SERVICE_NAME=a2a-agent-0 python launch_agent0.py > logs/agent0.log 2>&1 &
sleep 2
check_service 8001 "Agent 0"

# Start Agent 1 (Data Standardization)
echo "🔧 Starting Agent 1 (Data Standardization)..."
OTEL_SERVICE_NAME=a2a-agent-1 python launch_agent1.py > logs/agent1.log 2>&1 &
sleep 2
check_service 8002 "Agent 1"

# Start Agent 2 (AI Preparation)
echo "🤖 Starting Agent 2 (AI Preparation)..."
OTEL_SERVICE_NAME=a2a-agent-2 python launch_agent2.py > logs/agent2.log 2>&1 &
sleep 2
check_service 8003 "Agent 2"

# Start Agent 3 (Vector Processing)
echo "🔍 Starting Agent 3 (Vector Processing)..."
OTEL_SERVICE_NAME=a2a-agent-3 python launch_agent3.py > logs/agent3.log 2>&1 &
sleep 2
check_service 8004 "Agent 3"

# Start Data Manager Agent
echo "💾 Starting Data Manager Agent..."
OTEL_SERVICE_NAME=a2a-data-manager python launch_data_manager.py > logs/data_manager.log 2>&1 &
sleep 2
check_service 8005 "Data Manager Agent"

# Start Catalog Manager Agent
echo "📚 Starting Catalog Manager Agent..."
OTEL_SERVICE_NAME=a2a-catalog-manager python launch_catalog_manager.py > logs/catalog_manager.log 2>&1 &
sleep 2
check_service 8006 "Catalog Manager Agent"

# Start Main API Server (last)
echo "🌐 Starting Main API Server..."
OTEL_SERVICE_NAME=a2a-main-api uvicorn main:app --host 0.0.0.0 --port 8000 --reload > logs/server.log 2>&1 &
sleep 3
check_service 8000 "Main API Server"

echo ""
echo "✨ A2A Services started with OpenTelemetry!"
echo ""
echo "📊 OpenTelemetry endpoints:"
echo "   - OTLP gRPC: localhost:4317"
echo "   - OTLP HTTP: localhost:4318"
echo "   - Jaeger UI: http://localhost:16686"
echo "   - Collector Health: http://localhost:13133/health"
echo "   - Collector Metrics: http://localhost:8888/metrics"
echo ""
echo "🌐 Service endpoints:"
echo "   - Main API: http://localhost:8000"
echo "   - Registry: http://localhost:8100"
echo "   - Agent Manager: http://localhost:8010"
echo "   - Agent 0: http://localhost:8001"
echo "   - Agent 1: http://localhost:8002"
echo "   - Agent 2: http://localhost:8003"
echo "   - Agent 3: http://localhost:8004"
echo "   - Data Manager: http://localhost:8005"
echo "   - Catalog Manager: http://localhost:8006"
echo ""
echo "📝 Logs are being written to the logs/ directory"
echo "🛑 To stop services, run: ./stop_a2a_services.sh"