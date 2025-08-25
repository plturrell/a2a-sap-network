#!/bin/bash
# Wrapper script for entrypoint to ensure proper execution

set -e

# Copy embedded entrypoint if it doesn't exist yet
if [ ! -f /app/entrypoint.sh ]; then
    echo "Creating entrypoint.sh from embedded script..."
    cat > /app/entrypoint.sh << 'ENTRYPOINT_EOF'
#!/bin/bash
set -e

# Set up environment for container
export PYTHONPATH=/app/a2aAgents/backend:/app
export NODE_PATH=/app/node_modules
export PATH=/app/node_modules/.bin:$PATH

# Set required A2A environment variables if not already set
export A2A_SERVICE_URL=${A2A_SERVICE_URL:-http://localhost:4004}
export A2A_SERVICE_HOST=${A2A_SERVICE_HOST:-localhost}
export A2A_BASE_URL=${A2A_BASE_URL:-http://localhost:8000}
export A2A_AGENT_BASE_URL=${A2A_AGENT_BASE_URL:-http://localhost:8000}

# Set container environment flag
export A2A_IN_CONTAINER=true
export SKIP_VENV_CREATION=true

# Create required directories
mkdir -p /app/logs /app/data /app/pids

# Copy start.sh to the right location if needed
if [ ! -f /app/start.sh ] && [ -f /app/scripts/start.sh ]; then
    cp /app/scripts/start.sh /app/start.sh
    chmod +x /app/start.sh
fi

# Ensure start.sh exists
if [ ! -f /app/start.sh ]; then
    echo "ERROR: start.sh not found!"
    echo "Looking for start scripts..."
    find /app -name "start*.sh" -type f
    exit 1
fi

# Handle different commands
case "${1}" in
    verify)
        echo "Running 18-step verification..."
        cd /app
        exec ./start.sh verify
        ;;
    ci-verify)
        echo "Running CI verification mode..."
        cd /app
        exec ./start.sh ci-verify
        ;;
    test)
        echo "Running test mode..."
        cd /app
        exec ./start.sh test
        ;;
    complete)
        echo "Starting complete A2A platform with all services..."
        cd /app
        exec ./start.sh complete
        ;;
    backend)
        echo "Starting A2A Backend Service..."
        cd /app
        # Add error handling and logging
        echo "Environment variables:"
        echo "PYTHONPATH=$PYTHONPATH"
        echo "A2A_SERVICE_URL=$A2A_SERVICE_URL"
        echo "A2A_AGENT_BASE_URL=$A2A_AGENT_BASE_URL"
        echo "Working directory: $(pwd)"
        echo "Python version: $(python --version)"
        
        # Check if main.py exists
        if [ ! -f "a2aAgents/backend/main.py" ]; then
            echo "ERROR: main.py not found at a2aAgents/backend/main.py"
            ls -la a2aAgents/backend/
            exit 1
        fi
        
        # Start with verbose output for debugging
        exec python -u a2aAgents/backend/main.py
        ;;
    start)
        # Default start mode
        echo "Starting A2A system..."
        cd /app
        exec ./start.sh "$@"
        ;;
    *)
        # Default: run the command as-is
        exec "$@"
        ;;
esac
ENTRYPOINT_EOF
    chmod +x /app/entrypoint.sh
fi

# Execute the entrypoint
exec /app/entrypoint.sh "$@"