#!/bin/bash
# Optimized startup script for Fly.io deployment

set -e

echo "🚀 A2A Platform Fly.io Startup"
echo "=============================="
echo "Environment: $A2A_ENVIRONMENT"
echo "Startup Mode: ${STARTUP_MODE:-complete}"
echo "Memory Available: $(free -m | awk 'NR==2{printf "%.0f MB", $7}')"
echo "CPU Cores: $(nproc)"
echo ""

# Create required directories
mkdir -p /app/logs /app/data /app/pids

# Function to check if a port is in use
check_port() {
    local port=$1
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        return 0
    else
        return 1
    fi
}

# Function to wait for a service
wait_for_service() {
    local service_name=$1
    local port=$2
    local max_wait=${3:-60}
    local waited=0
    
    echo -n "⏳ Waiting for $service_name (port $port)..."
    
    while [ $waited -lt $max_wait ]; do
        if check_port $port; then
            echo " ✅ Ready!"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done
    
    echo " ❌ Timeout!"
    return 1
}

# Function to start a service with retry
start_service_with_retry() {
    local service_name=$1
    local start_command=$2
    local port=$3
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        echo "🔄 Starting $service_name (attempt $((retry + 1))/$max_retries)..."
        
        # Start the service in background
        eval "$start_command" &
        local pid=$!
        
        # Wait for service to be ready
        if wait_for_service "$service_name" "$port" 30; then
            echo "✅ $service_name started successfully (PID: $pid)"
            return 0
        else
            echo "❌ $service_name failed to start"
            kill -9 $pid 2>/dev/null || true
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                echo "⏳ Waiting 5 seconds before retry..."
                sleep 5
            fi
        fi
    done
    
    return 1
}

# Start based on mode
case "${STARTUP_MODE:-complete}" in
    quick)
        echo "🚀 Quick startup mode - Starting minimal services"
        
        # Start only Agent 0 for quick startup
        cd /app/a2aAgents/backend
        export AGENT_ID=agent0
        export AGENT_PORT=8000
        python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
        
        wait_for_service "Agent 0" 8000 60
        echo "✅ Quick startup complete"
        ;;
        
    backend)
        echo "🚀 Backend mode - Starting core agents"
        
        # Start first 6 agents
        cd /app/a2aAgents/backend
        for i in {0..5}; do
            export AGENT_ID=agent$i
            export AGENT_PORT=$((8000 + i))
            start_service_with_retry "Agent $i" "python -m uvicorn main:app --host 0.0.0.0 --port $AGENT_PORT" $AGENT_PORT
        done
        
        echo "✅ Backend agents started"
        ;;
        
    complete|*)
        echo "🚀 Complete mode - Starting all services"
        
        # Start A2A Network API (if available)
        if [ -d "/app/a2aNetwork" ] && [ -f "/app/a2aNetwork/package.json" ]; then
            echo "📡 Starting A2A Network API..."
            cd /app/a2aNetwork
            npm start > /app/logs/network.log 2>&1 &
            wait_for_service "A2A Network" 4004 120
        fi
        
        # Start Frontend (if available)
        if [ -d "/app/a2aAgents/frontend" ] && [ -f "/app/a2aAgents/frontend/package.json" ]; then
            echo "🎨 Starting Frontend..."
            cd /app/a2aAgents/frontend
            PORT=3000 npm start > /app/logs/frontend.log 2>&1 &
            wait_for_service "Frontend" 3000 120
        fi
        
        # Start all agents with optimized sequence
        echo "🤖 Starting agents in optimized sequence..."
        cd /app/a2aAgents/backend
        
        # Start critical agents first (0-5)
        for i in {0..5}; do
            export AGENT_ID=agent$i
            export AGENT_PORT=$((8000 + i))
            start_service_with_retry "Agent $i" "python -m uvicorn main:app --host 0.0.0.0 --port $AGENT_PORT" $AGENT_PORT
        done
        
        # Start remaining agents in parallel
        echo "🚀 Starting remaining agents in parallel..."
        for i in {6..17}; do
            export AGENT_ID=agent$i
            export AGENT_PORT=$((8000 + i))
            (
                python -m uvicorn main:app --host 0.0.0.0 --port $AGENT_PORT > /app/logs/agent${i}.log 2>&1 &
                echo "Started Agent $i on port $AGENT_PORT"
            ) &
        done
        
        # Wait for all agents to be ready
        echo "⏳ Waiting for all agents to initialize..."
        sleep 30
        
        # Check agent status
        READY_COUNT=0
        for i in {0..17}; do
            if check_port $((8000 + i)); then
                ((READY_COUNT++))
            fi
        done
        
        echo "✅ Complete startup finished: $READY_COUNT/18 agents ready"
        ;;
esac

# Keep the script running to maintain the container
echo ""
echo "🎯 A2A Platform is running"
echo "📊 Monitoring: http://localhost:8000/api/v1/monitoring/dashboard"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""

# Monitor and maintain services
while true; do
    sleep 60
    
    # Basic health check
    if ! curl -f -s http://localhost:8000/health > /dev/null; then
        echo "⚠️ Health check failed, services may need restart"
    fi
done