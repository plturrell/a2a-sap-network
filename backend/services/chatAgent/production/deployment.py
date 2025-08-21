"""
Production Deployment Configuration for A2A Chat Agent
Provides Docker, Kubernetes, and infrastructure setup
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")

class ProductionConfig:
    """
    Production configuration manager
    """
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment and files"""
        return {
            # Service configuration
            "service": {
                "name": "a2a-chat-agent",
                "version": "2.0.0",
                "port": int(os.getenv("PORT", "8017")),
                "host": os.getenv("HOST", "0.0.0.0"),
                "workers": int(os.getenv("WORKERS", "4")),
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
                "environment": self.environment
            },
            
            # Database configuration
            "database": {
                "type": os.getenv("DB_TYPE", "postgresql"),
                "connection_string": os.getenv(
                    "DATABASE_URL", 
                    "postgresql+asyncpg://a2a_user:a2a_pass@localhost:5432/a2a_chat"
                ),
                "pool_size": int(os.getenv("DB_POOL_SIZE", "20")),
                "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "10")),
                "echo_sql": os.getenv("DB_ECHO", "false").lower() == "true"
            },
            
            # Redis configuration
            "redis": {
                "url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                "pool_size": int(os.getenv("REDIS_POOL_SIZE", "50")),
                "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
            },
            
            # Authentication
            "auth": {
                "jwt_secret": os.getenv("JWT_SECRET", self._generate_secret()),
                "jwt_algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
                "jwt_expiry_hours": int(os.getenv("JWT_EXPIRY_HOURS", "24")),
                "enable_jwt": os.getenv("ENABLE_JWT", "true").lower() == "true",
                "enable_api_key": os.getenv("ENABLE_API_KEY", "true").lower() == "true"
            },
            
            # Rate limiting
            "rate_limiting": {
                "enabled": os.getenv("RATE_LIMITING_ENABLED", "true").lower() == "true",
                "default_tier": os.getenv("DEFAULT_TIER", "standard")
            },
            
            # Monitoring
            "monitoring": {
                "enable_prometheus": os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
                "prometheus_port": int(os.getenv("PROMETHEUS_PORT", "9090")),
                "enable_otlp": os.getenv("ENABLE_OTLP", "false").lower() == "true",
                "otlp_endpoint": os.getenv("OTLP_ENDPOINT", os.getenv("A2A_SERVICE_HOST")),
                "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
            },
            
            # AI Integration
            "ai": {
                "enabled": os.getenv("AI_ENABLED", "true").lower() == "true",
                "grok_api_key": os.getenv("GROK_API_KEY", ""),
                "grok_model": os.getenv("GROK_MODEL", "grok-beta"),
                "temperature": float(os.getenv("AI_TEMPERATURE", "0.7")),
                "max_tokens": int(os.getenv("AI_MAX_TOKENS", "4000"))
            },
            
            # WebSocket
            "websocket": {
                "enabled": os.getenv("WEBSOCKET_ENABLED", "true").lower() == "true",
                "max_connections": int(os.getenv("WS_MAX_CONNECTIONS", "1000")),
                "heartbeat_interval": int(os.getenv("WS_HEARTBEAT_INTERVAL", "30"))
            },
            
            # Security
            "security": {
                "enable_cors": os.getenv("ENABLE_CORS", "true").lower() == "true",
                "allowed_origins": os.getenv("ALLOWED_ORIGINS", "*").split(","),
                "enable_csrf": os.getenv("ENABLE_CSRF", "false").lower() == "true",
                "secure_headers": os.getenv("SECURE_HEADERS", "true").lower() == "true"
            },
            
            # Agent registry
            "agents": {
                "registry_url": os.getenv("AGENT_REGISTRY_URL"),
                "discovery_interval": int(os.getenv("AGENT_DISCOVERY_INTERVAL", "60")),
                "timeout": int(os.getenv("AGENT_TIMEOUT", "30"))
            }
        }
    
    def _generate_secret(self) -> str:
        """Generate a secure secret"""
        import secrets
        return secrets.token_urlsafe(32)
    
    def get_config(self) -> Dict[str, Any]:
        """Get full configuration"""
        return self.config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section"""
        return self.config.get(section, {})


def generate_docker_files() -> Dict[str, str]:
    """Generate Docker configuration files"""
    
    dockerfile = '''
# Multi-stage build for A2A Chat Agent
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r a2a && useradd -r -g a2a a2a

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=a2a:a2a . .

# Switch to non-root user
USER a2a

# Health check
HEALTHCHEK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8017/health || exit 1

# Expose port
EXPOSE 8017

# Start command
CMD ["python", "-m", "gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8017", "--access-logfile", "-", "--error-logfile", "-"]
'''
    
    docker_compose = '''
version: '3.8'

services:
  # A2A Chat Agent
  chat-agent:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8017:8017"
      - "9090:9090"  # Prometheus metrics
    environment:
      - DATABASE_URL=postgresql+asyncpg://a2a_user:a2a_pass@postgres:5432/a2a_chat
      - REDIS_URL=redis://redis:6379/0
      - JWT_SECRET=${JWT_SECRET:-your-jwt-secret-key}
      - GROK_API_KEY=${GROK_API_KEY}
      - LOG_LEVEL=INFO
      - ENVIRONMENT=production
    depends_on:
      - postgres
      - redis
    networks:
      - a2a-network
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs
    
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=a2a_chat
      - POSTGRES_USER=a2a_user
      - POSTGRES_PASSWORD=a2a_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - a2a-network
    restart: unless-stopped
    
  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - a2a-network
    restart: unless-stopped
    
  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - chat-agent
    networks:
      - a2a-network
    restart: unless-stopped
    
  # Prometheus Monitoring
  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    networks:
      - a2a-network
    restart: unless-stopped
    
  # Grafana Dashboard
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    networks:
      - a2a-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  a2a-network:
    driver: bridge
'''
    
    requirements_txt = '''
# Core dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
gunicorn>=21.2.0
pydantic>=2.5.0

# Database
sqlalchemy>=2.0.0
aiofiles>=23.2.1
aiosqlite>=0.19.0
asyncpg>=0.29.0

# Authentication & Security
pyjwt>=2.8.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6
cryptography>=41.0.0

# Caching & Rate Limiting
redis[hiredis]>=5.0.0

# Monitoring & Observability
prometheus-client>=0.19.0
opentelemetry-api>=1.21.0
opentelemetry-sdk>=1.21.0
opentelemetry-instrumentation-fastapi>=0.42b0
opentelemetry-instrumentation-aiohttp-client>=0.42b0
opentelemetry-exporter-prometheus>=1.12.0rc1
opentelemetry-exporter-otlp>=1.21.0

# System monitoring
psutil>=5.9.6

# HTTP client
aiohttp>=3.9.0
httpx>=0.25.0

# WebSocket support
websockets>=12.0

# Utilities
python-dotenv>=1.0.0
click>=8.1.7
rich>=13.7.0
typer>=0.9.0

# A2A specific (if available)
# a2aCommon>=1.0.0
# a2a-sdk>=1.0.0
'''
    
    return {
        "Dockerfile": dockerfile,
        "docker-compose.yml": docker_compose,
        "requirements.txt": requirements_txt
    }


def generate_kubernetes_config() -> Dict[str, str]:
    """Generate Kubernetes deployment configuration"""
    
    deployment = '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: a2a-chat-agent
  namespace: a2a-system
  labels:
    app: a2a-chat-agent
    version: v2.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: a2a-chat-agent
  template:
    metadata:
      labels:
        app: a2a-chat-agent
        version: v2.0.0
    spec:
      containers:
      - name: chat-agent
        image: a2a/chat-agent:2.0.0
        ports:
        - containerPort: 8017
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: a2a-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: a2a-secrets
              key: redis-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: a2a-secrets
              key: jwt-secret
        - name: GROK_API_KEY
          valueFrom:
            secretKeyRef:
              name: a2a-secrets
              key: grok-api-key
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8017
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8017
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: config
        configMap:
          name: a2a-chat-config
      - name: logs
        emptyDir: {}
      imagePullSecrets:
      - name: a2a-registry
---
apiVersion: v1
kind: Service
metadata:
  name: a2a-chat-agent-service
  namespace: a2a-system
  labels:
    app: a2a-chat-agent
spec:
  type: ClusterIP
  ports:
  - port: 8017
    targetPort: 8017
    protocol: TCP
    name: http
  - port: 9090
    targetPort: 9090
    protocol: TCP
    name: metrics
  selector:
    app: a2a-chat-agent
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: a2a-chat-agent-ingress
  namespace: a2a-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.a2a.example.com
    secretName: a2a-tls
  rules:
  - host: api.a2a.example.com
    http:
      paths:
      - path: /chat
        pathType: Prefix
        backend:
          service:
            name: a2a-chat-agent-service
            port:
              number: 8017
'''
    
    configmap = '''
apiVersion: v1
kind: ConfigMap
metadata:
  name: a2a-chat-config
  namespace: a2a-system
data:
  app.yaml: |
    service:
      name: "a2a-chat-agent"
      version: "2.0.0"
      port: 8017
      workers: 4
      log_level: "INFO"
    
    monitoring:
      enable_prometheus: true
      prometheus_port: 9090
      health_check_interval: 30
    
    rate_limiting:
      enabled: true
      default_tier: "standard"
    
    websocket:
      enabled: true
      max_connections: 1000
      heartbeat_interval: 30
    
    security:
      enable_cors: true
      secure_headers: true
'''
    
    secrets = '''
apiVersion: v1
kind: Secret
metadata:
  name: a2a-secrets
  namespace: a2a-system
type: Opaque
stringData:
  database-url: "postgresql+asyncpg://a2a_user:CHANGE_ME@postgres:5432/a2a_chat"
  redis-url: "redis://redis:6379/0"
  jwt-secret: "CHANGE_ME_TO_SECURE_SECRET"
  grok-api-key: "CHANGE_ME_TO_ACTUAL_API_KEY"
'''
    
    hpa = '''
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: a2a-chat-agent-hpa
  namespace: a2a-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: a2a-chat-agent
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
'''
    
    return {
        "deployment.yaml": deployment,
        "configmap.yaml": configmap,
        "secrets.yaml": secrets,
        "hpa.yaml": hpa
    }


def generate_monitoring_config() -> Dict[str, str]:
    """Generate monitoring configuration files"""
    
    prometheus_config = '''
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'a2a-chat-agent'
    static_configs:
      - targets: ['chat-agent:9090']
    scrape_interval: 10s
    metrics_path: '/metrics'
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
'''
    
    alert_rules = '''
groups:
  - name: a2a_chat_agent
    rules:
      - alert: HighErrorRate
        expr: rate(a2a_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
      
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(a2a_chat_request_duration_seconds_bucket[5m])) > 5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"
      
      - alert: LowCacheHitRate
        expr: rate(a2a_cache_hits_total[5m]) / (rate(a2a_cache_hits_total[5m]) + rate(a2a_cache_misses_total[5m])) < 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value | humanizePercentage }}"
      
      - alert: DatabaseConnectionIssue
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection issue"
          description: "Cannot connect to PostgreSQL database"
      
      - alert: RedisConnectionIssue
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis connection issue"
          description: "Cannot connect to Redis cache"
      
      - alert: HighMemoryUsage
        expr: a2a_memory_usage_bytes / 1024 / 1024 / 1024 > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}GB"
'''
    
    grafana_dashboard = json.dumps({
        "dashboard": {
            "id": None,
            "title": "A2A Chat Agent Dashboard",
            "tags": ["a2a", "chat", "agents"],
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "Request Rate",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(a2a_chat_requests_total[5m])",
                            "legendFormat": "Requests/sec"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                },
                {
                    "id": 2,
                    "title": "Response Time",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(a2a_chat_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "95th percentile"
                        },
                        {
                            "expr": "histogram_quantile(0.50, rate(a2a_chat_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "50th percentile"
                        }
                    ],
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                },
                {
                    "id": 3,
                    "title": "Active Conversations",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "a2a_active_conversations",
                            "legendFormat": "Active"
                        }
                    ],
                    "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8}
                },
                {
                    "id": 4,
                    "title": "Error Rate",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "rate(a2a_errors_total[5m])",
                            "legendFormat": "Errors/sec"
                        }
                    ],
                    "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8}
                },
                {
                    "id": 5,
                    "title": "Cache Hit Rate",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "rate(a2a_cache_hits_total[5m]) / (rate(a2a_cache_hits_total[5m]) + rate(a2a_cache_misses_total[5m]))",
                            "legendFormat": "Hit Rate"
                        }
                    ],
                    "gridPos": {"h": 4, "w": 6, "x": 12, "y": 8}
                },
                {
                    "id": 6,
                    "title": "Memory Usage",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "a2a_memory_usage_bytes / 1024 / 1024",
                            "legendFormat": "MB"
                        }
                    ],
                    "gridPos": {"h": 4, "w": 6, "x": 18, "y": 8}
                }
            ],
            "time": {"from": "now-1h", "to": "now"},
            "refresh": "10s"
        }
    }, indent=2)
    
    return {
        "prometheus.yml": prometheus_config,
        "alert_rules.yml": alert_rules,
        "grafana_dashboard.json": grafana_dashboard
    }


def generate_nginx_config() -> str:
    """Generate Nginx configuration for load balancing"""
    return '''
events {
    worker_connections 1024;
}

http {
    upstream chat_agent {
        least_conn;
        server chat-agent:8017 max_fails=3 fail_timeout=30s;
        # Add more instances for load balancing
        # server chat-agent-2:8017 max_fails=3 fail_timeout=30s;
        # server chat-agent-3:8017 max_fails=3 fail_timeout=30s;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=websocket:10m rate=5r/s;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    server {
        listen 80;
        server_name api.a2a.example.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name api.a2a.example.com;
        
        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_session_cache shared:SSL:1m;
        ssl_session_timeout 10m;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers on;
        
        # Main API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://chat_agent/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
            
            # Buffer settings
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
        }
        
        # WebSocket endpoints
        location /ws/ {
            limit_req zone=websocket burst=10 nodelay;
            proxy_pass http://chat_agent/ws/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket timeouts
            proxy_connect_timeout 7d;
            proxy_send_timeout 7d;
            proxy_read_timeout 7d;
        }
        
        # Health check endpoint
        location /health {
            proxy_pass http://chat_agent/health;
            access_log off;
        }
        
        # Metrics endpoint (restrict access)
        location /metrics {
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
            proxy_pass http://chat_agent/metrics;
        }
        
        # Security
        location ~ /\. {
            deny all;
        }
    }
}
'''


def create_deployment_package(output_dir: str = "./deployment"):
    """Create complete deployment package"""
    import os
    from pathlib import Path
    
    base_path = Path(output_dir)
    base_path.mkdir(exist_ok=True)
    
    # Docker files
    docker_files = generate_docker_files()
    for filename, content in docker_files.items():
        (base_path / filename).write_text(content)
    
    # Kubernetes files
    k8s_path = base_path / "kubernetes"
    k8s_path.mkdir(exist_ok=True)
    k8s_files = generate_kubernetes_config()
    for filename, content in k8s_files.items():
        (k8s_path / filename).write_text(content)
    
    # Monitoring files
    monitoring_path = base_path / "monitoring"
    monitoring_path.mkdir(exist_ok=True)
    monitoring_files = generate_monitoring_config()
    for filename, content in monitoring_files.items():
        (monitoring_path / filename).write_text(content)
    
    # Nginx config
    nginx_config = generate_nginx_config()
    (base_path / "nginx.conf").write_text(nginx_config)
    
    # Environment template
    env_template = '''
# A2A Chat Agent Environment Configuration

# Database
DATABASE_URL=postgresql+asyncpg://a2a_user:CHANGE_ME@localhost:5432/a2a_chat
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_POOL_SIZE=50

# Authentication
JWT_SECRET=CHANGE_ME_TO_SECURE_SECRET
JWT_ALGORITHM=HS256
JWT_EXPIRY_HOURS=24

# AI Integration
GROK_API_KEY=CHANGE_ME_TO_ACTUAL_KEY
AI_TEMPERATURE=0.7
AI_MAX_TOKENS=4000

# Service
PORT=8017
WORKERS=4
LOG_LEVEL=INFO
ENVIRONMENT=production

# Monitoring
ENABLE_PROMETHEUS=true
PROMETHEUS_PORT=9090
ENABLE_OTLP=false
OTLP_ENDPOINT=localhost:4317

# Security
ENABLE_CORS=true
ALLOWED_ORIGINS=*
SECURE_HEADERS=true

# Rate Limiting
RATE_LIMITING_ENABLED=true
DEFAULT_TIER=standard

# WebSocket
WEBSOCKET_ENABLED=true
WS_MAX_CONNECTIONS=1000
WS_HEARTBEAT_INTERVAL=30
'''
    (base_path / ".env.template").write_text(env_template)
    
    # README
    readme = '''
# A2A Chat Agent Deployment

Production deployment package for the A2A Chat Agent.

## Quick Start

### Docker Compose (Recommended)

1. Copy environment template:
   ```bash
   cp .env.template .env
   ```

2. Edit `.env` with your configuration

3. Start services:
   ```bash
   docker-compose up -d
   ```

4. Check health:
   ```bash
   curl http://localhost:8017/health
   ```

### Kubernetes

1. Update secrets in `kubernetes/secrets.yaml`

2. Apply configurations:
   ```bash
   kubectl apply -f kubernetes/
   ```

3. Check deployment:
   ```bash
   kubectl get pods -n a2a-system
   ```

## Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)
- Application metrics: http://localhost:8017/metrics

## Security Notes

- Change all default passwords and secrets
- Use proper TLS certificates in production
- Configure firewall rules appropriately
- Enable authentication for monitoring endpoints

## Scaling

- Horizontal Pod Autoscaler configured for Kubernetes
- Add more chat-agent replicas in docker-compose.yml
- Configure load balancer for multiple instances
'''
    (base_path / "README.md").write_text(readme)
    
    print(f"Deployment package created in: {base_path.absolute()}")
    return str(base_path.absolute())
