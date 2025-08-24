# A2A Network Docker Configuration Documentation

## Overview

This document provides comprehensive documentation for the Docker Compose configuration of the A2A Network platform. The configuration defines a complete containerized environment for development, testing, and production deployment with full observability stack.

## Architecture Overview

The Docker Compose setup creates a multi-service architecture with the following components:

### Core Services
- **A2A Agent**: Main application container
- **Redis**: Caching and session storage
- **PostgreSQL**: Primary database
- **Nginx**: Reverse proxy and load balancer

### Observability Stack
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **Loki**: Log aggregation
- **Promtail**: Log shipping

## Service Configurations

### A2A Agent (Main Application)

```yaml
a2a-agent:
  image: finsight-cib:latest
  container_name: a2a-agent
  restart: unless-stopped
```

**Configuration Details:**
- **Image**: `finsight-cib:latest` - Custom built application image
- **Container Name**: `a2a-agent` - Fixed name for service discovery
- **Restart Policy**: `unless-stopped` - Automatic restart except when manually stopped
- **Environment**: Production configuration via `.env.production` file
- **Ports**:
  - `8000:8000` - Main API endpoint
  - `9090:9090` - Prometheus metrics endpoint
- **Volumes**:
  - `./logs:/app/logs` - Log file persistence
  - `./config:/app/config:ro` - Read-only configuration files
- **Health Check**: HTTP GET to `/health` endpoint every 30 seconds
- **Dependencies**: Requires Redis and PostgreSQL to be healthy

### Redis (Caching Layer)

```yaml
redis:
  image: redis:7-alpine
  container_name: a2a-redis
  restart: unless-stopped
```

**Configuration Details:**
- **Image**: `redis:7-alpine` - Latest Redis 7.x on Alpine Linux
- **Container Name**: `a2a-redis` - Fixed name for application connection
- **Restart Policy**: `unless-stopped` - High availability configuration
- **Command**: `redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}`
  - **Append-Only File**: Enabled for data persistence
  - **Password Protection**: Uses environment variable for security
- **Volume**: `redis-data:/data` - Persistent data storage
- **Health Check**: Redis PING command every 30 seconds
- **Use Cases**:
  - Session storage
  - API rate limiting
  - Application caching
  - Real-time data storage

### PostgreSQL (Primary Database)

```yaml
postgres:
  image: postgres:15-alpine
  container_name: a2a-postgres
  restart: unless-stopped
```

**Configuration Details:**
- **Image**: `postgres:15-alpine` - PostgreSQL 15 on Alpine Linux
- **Container Name**: `a2a-postgres` - Fixed name for application connection
- **Restart Policy**: `unless-stopped` - Data persistence guarantee
- **Environment Variables**:
  - `POSTGRES_DB`: Database name from environment
  - `POSTGRES_USER`: Database user from environment
  - `POSTGRES_PASSWORD`: Database password from environment
- **Volumes**:
  - `postgres-data:/var/lib/postgresql/data` - Database files persistence
  - `./backups:/backups` - Backup storage location
- **Health Check**: `pg_isready` command every 30 seconds
- **Use Cases**:
  - Agent configuration storage
  - Workflow definitions
  - Audit logs
  - User management

### Nginx (Reverse Proxy)

```yaml
nginx:
  image: nginx:alpine
  container_name: a2a-nginx
  restart: unless-stopped
```

**Configuration Details:**
- **Image**: `nginx:alpine` - Lightweight Nginx on Alpine Linux
- **Container Name**: `a2a-nginx` - Fixed name for external access
- **Restart Policy**: `unless-stopped` - High availability for external access
- **Ports**:
  - `80:80` - HTTP traffic
  - `443:443` - HTTPS traffic
- **Volumes**:
  - `./nginx/nginx.conf:/etc/nginx/nginx.conf:ro` - Main configuration
  - `./nginx/ssl:/etc/nginx/ssl:ro` - SSL certificates
  - `./logs/nginx:/var/log/nginx` - Access and error logs
- **Health Check**: HTTP GET to `/health` endpoint
- **Features**:
  - SSL termination
  - Load balancing
  - Request routing
  - Static file serving

## Observability Stack

### Prometheus (Metrics Collection)

```yaml
prometheus:
  image: prom/prometheus:latest
  container_name: a2a-prometheus
```

**Configuration Details:**
- **Image**: `prom/prometheus:latest` - Latest Prometheus server
- **Container Name**: `a2a-prometheus` - Fixed name for Grafana connection
- **Command Line Options**:
  - `--config.file=/etc/prometheus/prometheus.yml` - Configuration file location
  - `--storage.tsdb.path=/prometheus` - Time series database path
  - `--web.enable-lifecycle` - Enable configuration reload API
- **Volumes**:
  - `./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro` - Configuration
  - `prometheus-data:/prometheus` - Metrics storage
- **Metrics Collected**:
  - Application performance metrics
  - System resource usage
  - Custom business metrics
  - HTTP request metrics

### Grafana (Visualization)

```yaml
grafana:
  image: grafana/grafana:latest
  container_name: a2a-grafana
```

**Configuration Details:**
- **Image**: `grafana/grafana:latest` - Latest Grafana server
- **Container Name**: `a2a-grafana` - Fixed name for dashboard access
- **Environment Variables**:
  - `GF_SECURITY_ADMIN_PASSWORD`: Admin password from environment
  - `GF_USERS_ALLOW_SIGN_UP=false` - Disable public registration
- **Volumes**:
  - `grafana-data:/var/lib/grafana` - Dashboard and configuration storage
  - `./monitoring/grafana:/etc/grafana/provisioning:ro` - Provisioned dashboards
- **Features**:
  - Pre-configured dashboards
  - Prometheus data source
  - Alert management
  - User access control

### Jaeger (Distributed Tracing)

```yaml
jaeger:
  image: jaegertracing/all-in-one:latest
  container_name: a2a-jaeger
```

**Configuration Details:**
- **Image**: `jaegertracing/all-in-one:latest` - Complete Jaeger stack
- **Container Name**: `a2a-jaeger` - Fixed name for trace collection
- **Environment Variables**:
  - `COLLECTOR_OTLP_ENABLED=true` - Enable OpenTelemetry Protocol
- **Features**:
  - Trace collection
  - Trace storage
  - Query interface
  - Service dependency mapping

### Loki (Log Aggregation)

```yaml
loki:
  image: grafana/loki:latest
  container_name: a2a-loki
```

**Configuration Details:**
- **Image**: `grafana/loki:latest` - Latest Loki log aggregation
- **Container Name**: `a2a-loki` - Fixed name for log storage
- **Command**: `-config.file=/etc/loki/local-config.yaml` - Configuration file
- **Volumes**:
  - `./monitoring/loki.yml:/etc/loki/local-config.yaml:ro` - Configuration
  - `loki-data:/loki` - Log storage
- **Features**:
  - Log indexing
  - Log retention policies
  - Query interface
  - Grafana integration

### Promtail (Log Shipping)

```yaml
promtail:
  image: grafana/promtail:latest
  container_name: a2a-promtail
```

**Configuration Details:**
- **Image**: `grafana/promtail:latest` - Latest Promtail log shipper
- **Container Name**: `a2a-promtail` - Fixed name for log collection
- **Volumes**:
  - `./logs:/var/log/app:ro` - Application logs (read-only)
  - `./monitoring/promtail.yml:/etc/promtail/config.yml:ro` - Configuration
- **Dependencies**: Requires Loki to be running
- **Features**:
  - Log file monitoring
  - Log parsing and labeling
  - Automatic log shipping to Loki

## Network Configuration

### A2A Network Bridge

```yaml
networks:
  a2a-network:
    driver: bridge
```

**Configuration Details:**
- **Network Name**: `a2a-network` - Custom bridge network
- **Driver**: `bridge` - Docker bridge networking
- **Purpose**: Isolated network for all A2A services
- **Benefits**:
  - Service discovery by container name
  - Network isolation from other applications
  - Internal DNS resolution
  - Secure inter-service communication

## Volume Configuration

### Persistent Storage Volumes

```yaml
volumes:
  redis-data:
    driver: local
  postgres-data:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local
  loki-data:
    driver: local
```

**Volume Details:**
- **redis-data**: Redis database files and append-only file
- **postgres-data**: PostgreSQL database files and WAL logs
- **prometheus-data**: Time series metrics database
- **grafana-data**: Dashboards, users, and configuration
- **loki-data**: Log index and storage

**Storage Characteristics:**
- **Driver**: `local` - Host filesystem storage
- **Persistence**: Data survives container restarts
- **Backup**: Regular backup strategies recommended
- **Performance**: Local SSD recommended for production

## Health Checks

### Health Check Configuration

All services include comprehensive health checks:

**A2A Agent Health Check:**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

**Redis Health Check:**
```yaml
healthcheck:
  test: ["CMD", "redis-cli", "ping"]
  interval: 30s
  timeout: 10s
  retries: 3
```

**PostgreSQL Health Check:**
```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
  interval: 30s
  timeout: 10s
  retries: 3
```

**Health Check Benefits:**
- Automatic service recovery
- Load balancer integration
- Monitoring system integration
- Deployment validation

## Environment Variables

### Required Environment Variables

Create a `.env.production` file with the following variables:

```bash
# Database Configuration
DB_NAME=a2a_network
DB_USER=a2a_user
DB_PASSWORD=secure_password_here

# Redis Configuration
REDIS_PASSWORD=redis_password_here

# Grafana Configuration
GRAFANA_PASSWORD=grafana_admin_password

# Application Configuration
A2A_ENVIRONMENT=production
LOG_LEVEL=info
```

### Security Considerations

- **Password Complexity**: Use strong, unique passwords
- **Environment Isolation**: Separate .env files for different environments
- **Secret Management**: Consider using Docker secrets for production
- **Access Control**: Limit file permissions on .env files

## Deployment Instructions

### Development Deployment

```bash
# Clone the repository
git clone <repository-url>
cd a2a-network

# Create environment file
cp .env.example .env.production
# Edit .env.production with your values

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps
```

### Production Deployment

```bash
# Pull latest images
docker-compose pull

# Start services with production configuration
docker-compose -f docker-compose.yml up -d

# Verify all services are healthy
docker-compose ps
docker-compose logs --tail=50

# Setup monitoring dashboards
# Access Grafana at http://localhost:3000
# Import pre-configured dashboards
```

### Scaling Services

```bash
# Scale the main application
docker-compose up -d --scale a2a-agent=3

# Scale with load balancer configuration
# Update nginx.conf for multiple upstream servers
```

## Monitoring and Observability

### Access Points

- **Application**: http://localhost:8000
- **Grafana Dashboards**: http://localhost:3000
- **Prometheus Metrics**: http://localhost:9090
- **Jaeger Tracing**: http://localhost:16686

### Key Metrics to Monitor

1. **Application Metrics**:
   - Request rate and latency
   - Error rates
   - Agent execution times
   - Queue depths

2. **Infrastructure Metrics**:
   - CPU and memory usage
   - Disk I/O and space
   - Network throughput
   - Container health status

3. **Business Metrics**:
   - Active agents count
   - Workflow completion rates
   - Data processing volumes
   - User activity levels

## Troubleshooting

### Common Issues

1. **Service Won't Start**:
   - Check environment variables
   - Verify volume permissions
   - Review service logs

2. **Database Connection Issues**:
   - Verify PostgreSQL is healthy
   - Check network connectivity
   - Validate credentials

3. **Performance Issues**:
   - Monitor resource usage
   - Check for memory leaks
   - Analyze slow queries

### Log Analysis

```bash
# View all service logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f a2a-agent

# View logs with timestamps
docker-compose logs -t

# Filter logs by time
docker-compose logs --since="2024-01-01T00:00:00"
```

## Backup and Recovery

### Database Backup

```bash
# Create PostgreSQL backup
docker-compose exec postgres pg_dump -U ${DB_USER} ${DB_NAME} > backup.sql

# Restore from backup
docker-compose exec -T postgres psql -U ${DB_USER} ${DB_NAME} < backup.sql
```

### Volume Backup

```bash
# Backup all volumes
docker run --rm -v a2anetwork_postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz -C /data .

# Restore volume
docker run --rm -v a2anetwork_postgres-data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres-backup.tar.gz -C /data
```

## Security Best Practices

### Container Security

1. **Use Official Images**: All images are from official repositories
2. **Regular Updates**: Keep base images updated
3. **Minimal Privileges**: Run containers as non-root users
4. **Network Isolation**: Use custom networks
5. **Secret Management**: Use Docker secrets for sensitive data

### Access Control

1. **Strong Passwords**: Enforce complex passwords
2. **Limited Exposure**: Only expose necessary ports
3. **SSL/TLS**: Use HTTPS for all external communication
4. **Regular Audits**: Monitor access logs and user activity

This Docker configuration provides a complete, production-ready environment for the A2A Network platform with comprehensive observability and monitoring capabilities.
