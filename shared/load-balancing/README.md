# A2A Platform Load Balancing & Service Discovery

Advanced load balancing and service discovery infrastructure for the A2A Platform with multiple strategies, health monitoring, and enterprise-grade reliability.

## 🚀 Features

### Load Balancing Options
- **NGINX**: High-performance HTTP/HTTPS load balancer with SSL termination
- **HAProxy**: Advanced Layer 4/7 load balancer with health checks
- **Traefik**: Cloud-native load balancer with automatic service discovery
- **Envoy**: Advanced L7 proxy with observability and security features

### Service Discovery
- **Built-in Registry**: In-memory service registry with health monitoring
- **Consul Integration**: HashiCorp Consul for distributed service discovery
- **etcd Integration**: Kubernetes-style service discovery
- **DNS-based Discovery**: SRV record and environment-based discovery

### Service Mesh
- **Circuit Breakers**: Automatic failure handling and recovery
- **Retry Logic**: Intelligent retry strategies with backoff
- **Distributed Tracing**: Request tracing across service boundaries  
- **Metrics Collection**: Real-time performance and health metrics

## 📁 Architecture

```
shared/load-balancing/
├── service-discovery.js        # Service registry and discovery
├── service-mesh.js             # Service mesh implementation
├── nginx-load-balancer.conf    # NGINX configuration
├── haproxy-config.cfg          # HAProxy configuration
├── docker-compose-lb.yml       # Complete deployment setup
└── README.md                   # This documentation
```

## 🛠️ Quick Start

### 1. Basic Setup with NGINX

```bash
# Deploy NGINX load balancer with monitoring
docker-compose -f docker-compose-lb.yml --profile nginx --profile monitoring up -d

# Access the application
curl https://a2a-platform.local/api/health

# Check load balancer stats
curl http://localhost:8080/nginx_status
```

### 2. HAProxy Setup

```bash
# Deploy HAProxy load balancer
docker-compose -f docker-compose-lb.yml --profile haproxy --profile monitoring up -d

# Access HAProxy stats
open http://localhost:8404/stats
# Username: admin, Password: admin123
```

### 3. Traefik with Auto-Discovery

```bash
# Deploy Traefik with Docker integration
docker-compose -f docker-compose-lb.yml --profile traefik up -d

# Access Traefik dashboard
open http://localhost:8080
```

## 💻 Service Discovery Usage

### JavaScript/Node.js

```javascript
const { createServiceRegistry } = require('./service-discovery');

// Initialize service registry
const registry = createServiceRegistry({
    healthCheckInterval: 30000,
    loadBalancingStrategy: 'least-connections'
});

await registry.start();

// Register a service
await registry.registerService({
    id: 'a2a-network-1',
    name: 'a2a-network',
    address: 'a2a-network-1',
    port: 4004,
    protocol: 'http',
    tags: ['api', 'primary'],
    metadata: { version: '1.0.0', region: 'us-east-1' },
    health: {
        checkUrl: 'http://a2a-network-1:4004/health',
        checkInterval: 15000
    }
});

// Discover services
const instances = await registry.discoverService('a2a-network');
console.log('Available instances:', instances);

// Get service instance with load balancing
const instance = await registry.getServiceInstance('a2a-network', 'round-robin');
console.log('Selected instance:', instance);
```

### Service Client

```javascript
const { A2AServiceClient } = require('./service-discovery');

// Create service client
const client = new A2AServiceClient(registry, {
    timeout: 10000,
    retries: 3,
    circuitBreaker: true
});

// Make requests with automatic load balancing
try {
    const response = await client.request('a2a-network', '/api/agents', {
        method: 'GET',
        loadBalancingStrategy: 'least-connections'
    });
    
    console.log('Response:', response.data);
} catch (error) {
    console.error('Request failed:', error.message);
}
```

## 🕸️ Service Mesh Usage

```javascript
const { createServiceMesh } = require('./service-mesh');

// Initialize service mesh
const mesh = createServiceMesh({
    serviceName: 'a2a-agents',
    version: '1.0.0',
    
    // Circuit breaker settings
    circuitBreakerThreshold: 5,
    circuitBreakerTimeout: 60000,
    
    // Retry settings
    defaultRetries: 3,
    retryBackoff: 'exponential',
    
    // Observability
    enableTracing: true,
    enableMetrics: true
});

await mesh.initialize();

// Register target services
mesh.registerService('a2a-network', [
    { address: 'a2a-network-1', port: 4004 },
    { address: 'a2a-network-2', port: 4004 },
    { address: 'a2a-network-3', port: 4004, weight: 0.5 }
]);

// Make service calls with automatic retry, circuit breaker, etc.
try {
    const result = await mesh.call('a2a-network', {
        path: '/api/agents/status',
        method: 'GET',
        timeout: 5000,
        retries: 2
    });
    
    console.log('Service response:', result);
} catch (error) {
    console.error('Service call failed:', error.message);
}
```

### Middleware System

```javascript
// Add custom middleware
mesh.use(async (context, next) => {
    console.log(`Calling service: ${context.serviceName}`);
    
    const startTime = performance.now();
    
    try {
        const result = await next();
        const duration = performance.now() - startTime;
        console.log(`Request completed in ${duration.toFixed(2)}ms`);
        return result;
    } catch (error) {
        console.error(`Request failed: ${error.message}`);
        throw error;
    }
});

// Authentication middleware
mesh.use(async (context, next) => {
    // Add authentication headers
    context.options.headers = {
        ...context.options.headers,
        'Authorization': `Bearer ${getAuthToken()}`,
        'X-Service-Name': mesh.config.serviceName
    };
    
    return await next();
});
```

## ⚙️ Configuration

### Environment Variables

```bash
# Load Balancer Settings
LB_STRATEGY=nginx                    # nginx, haproxy, traefik, envoy
LB_SSL_ENABLED=true
LB_COMPRESSION_ENABLED=true

# Service Discovery
SERVICE_DISCOVERY_TYPE=memory        # memory, consul, etcd
SERVICE_DISCOVERY_HOST=localhost
SERVICE_DISCOVERY_PORT=8500

# Health Checks
HEALTH_CHECK_INTERVAL=30000
HEALTH_CHECK_TIMEOUT=5000
UNHEALTHY_THRESHOLD=3

# Circuit Breaker
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60000

# SSL Configuration
SSL_CERT_PATH=/etc/ssl/certs/a2a-platform.pem
SSL_KEY_PATH=/etc/ssl/private/a2a-platform.key
SSL_PROTOCOLS=TLSv1.2,TLSv1.3

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
TRACING_ENABLED=true
```

### NGINX Configuration Highlights

```nginx
# Load balancing strategies
upstream a2a_network {
    least_conn;  # or ip_hash, round_robin
    
    server a2a-network-1:4004 max_fails=3 fail_timeout=30s weight=3;
    server a2a-network-2:4004 max_fails=3 fail_timeout=30s weight=3;
    server a2a-network-3:4004 max_fails=3 fail_timeout=30s weight=2 backup;
    
    keepalive 32;
}

# SSL termination
server {
    listen 443 ssl http2;
    
    ssl_certificate /etc/nginx/certs/a2a-platform.crt;
    ssl_certificate_key /etc/nginx/certs/a2a-platform.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
}

# Caching
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=a2a_cache:100m;

location /api/ {
    proxy_cache a2a_cache;
    proxy_cache_valid 200 302 10m;
    proxy_cache_key $scheme$proxy_host$request_uri$http_authorization;
    
    proxy_pass http://a2a_network;
}
```

### HAProxy Configuration Highlights

```haproxy
# Health checks
backend a2a_network
    mode http
    balance roundrobin
    
    option httpchk GET /health HTTP/1.1\r\nHost:\ a2a-platform.local
    http-check expect status 200
    
    server a2a-net-1 a2a-network-1:4004 check inter 10s rise 2 fall 3
    server a2a-net-2 a2a-network-2:4004 check inter 10s rise 2 fall 3
    server a2a-net-3 a2a-network-3:4004 check inter 10s rise 2 fall 3 backup

# Session persistence
backend a2a_agents
    stick-table type string len 32 size 30k expire 30m
    stick on hdr(X-Session-ID)
```

## 🔍 Monitoring & Metrics

### Health Checks

```bash
# Check overall system health
curl http://localhost:8080/health/detailed

# Service discovery health
curl http://localhost:8500/health

# Load balancer stats
curl http://localhost:8404/stats     # HAProxy
curl http://localhost:8080/nginx_status  # NGINX
```

### Metrics Collection

```javascript
// Service registry metrics
const registryMetrics = registry.getSystemMetrics();
console.log('Registry metrics:', registryMetrics);

// Service mesh metrics
const meshMetrics = mesh.getMetrics();
console.log('Mesh metrics:', meshMetrics);

// Example output:
{
  "mesh": {
    "service": "a2a-agents",
    "activeRequests": 5,
    "registeredServices": 3
  },
  "services": {
    "a2a-network": {
      "totalRequests": 1250,
      "successfulRequests": 1198,
      "failedRequests": 52,
      "avgResponseTime": 45.7,
      "successRate": 95.84,
      "p50": 35.2,
      "p95": 89.1,
      "p99": 156.3
    }
  }
}
```

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'a2a-load-balancer'
    static_configs:
      - targets: ['nginx-lb:8080', 'haproxy-lb:8404']
    scrape_interval: 15s
    
  - job_name: 'a2a-service-discovery'
    static_configs:
      - targets: ['service-discovery:8500']
    scrape_interval: 30s
```

## 🔐 Security Features

### SSL/TLS Configuration

```bash
# Generate self-signed certificates for development
openssl req -x509 -newkey rsa:4096 -keyout a2a-platform.key \
    -out a2a-platform.crt -days 365 -nodes \
    -subj "/C=US/ST=CA/L=SF/O=A2A/CN=a2a-platform.local"

# Configure Let's Encrypt for production
docker-compose -f docker-compose-lb.yml --profile ssl up -d
```

### Rate Limiting

```nginx
# NGINX rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=100r/s;
limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/s;

location /api/ {
    limit_req zone=api burst=50 nodelay;
}

location /auth/ {
    limit_req zone=auth burst=10 nodelay;
}
```

```haproxy
# HAProxy rate limiting
frontend https_frontend
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request reject if { sc_http_req_rate(0) gt 100 }
```

### Access Control

```nginx
# Geographic restrictions
geo $allowed_country {
    default 1;
    # Block specific countries
    CN 0;  # China
    RU 0;  # Russia
}

if ($allowed_country = 0) {
    return 403 "Access denied from your location";
}
```

## 🏗️ Advanced Deployments

### Kubernetes Integration

```yaml
apiVersion: v1
kind: Service
metadata:
  name: a2a-load-balancer
spec:
  selector:
    app: nginx-lb
  ports:
    - port: 80
      targetPort: 80
    - port: 443
      targetPort: 443
  type: LoadBalancer

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-lb
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nginx-lb
  template:
    metadata:
      labels:
        app: nginx-lb
    spec:
      containers:
      - name: nginx
        image: nginx:alpine
        volumeMounts:
        - name: config
          mountPath: /etc/nginx/nginx.conf
          subPath: nginx.conf
```

### Docker Swarm

```bash
# Initialize Docker Swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose-lb.yml a2a-lb

# Scale services
docker service scale a2a-lb_nginx-lb=3
```

### Multi-Region Setup

```yaml
# Global load balancer with region affinity
services:
  nginx-lb-us-east:
    extends:
      file: docker-compose-lb.yml
      service: nginx-lb
    environment:
      - REGION=us-east-1
    networks:
      - us-east-network
      
  nginx-lb-us-west:
    extends:
      file: docker-compose-lb.yml
      service: nginx-lb
    environment:
      - REGION=us-west-1
    networks:
      - us-west-network
```

## 🔧 Troubleshooting

### Common Issues

**Service Discovery Not Working**:
```bash
# Check service registration
curl http://localhost:8500/v1/health/service/a2a-network

# Verify DNS resolution
nslookup _a2a-network._tcp.service.consul

# Check service logs
docker logs a2a-service-discovery
```

**Load Balancer Health Check Failures**:
```bash
# Test backend health directly
curl http://a2a-network-1:4004/health

# Check load balancer logs
docker logs a2a-nginx-lb
docker logs a2a-haproxy-lb

# Verify network connectivity
docker exec a2a-nginx-lb ping a2a-network-1
```

**High Response Times**:
```bash
# Check connection pooling
curl http://localhost:8404/stats  # HAProxy connection stats

# Monitor active connections
ss -tuln | grep :80
ss -tuln | grep :443

# Check resource utilization
docker stats a2a-nginx-lb
```

### Performance Tuning

**NGINX Optimization**:
```nginx
# Increase worker connections
events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

# Optimize buffers
http {
    client_body_buffer_size 128k;
    client_max_body_size 50m;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;
    output_buffers 1 32k;
    postpone_output 1460;
}
```

**HAProxy Optimization**:
```haproxy
global
    tune.ssl.default-dh-param 2048
    tune.bufsize 32768
    tune.maxrewrite 8192
    
defaults
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
```

### Monitoring Commands

```bash
# Real-time metrics
watch -n 1 'curl -s http://localhost:8080/nginx_status'

# Service health overview
curl -s http://localhost:8500/health | jq '.services'

# Load balancer performance
curl -s http://localhost:8404/stats | grep -E "BACKEND|a2a-"

# Traffic analysis
tail -f /var/log/nginx/access.log | grep -E "GET|POST"
```

## 📊 Performance Benchmarks

### Load Testing

```bash
# Install Apache Bench
apt-get install apache2-utils

# Basic load test
ab -n 10000 -c 100 https://a2a-platform.local/api/health

# Stress test with wrk
wrk -t12 -c400 -d30s https://a2a-platform.local/api/agents

# Artillery.io test
artillery run load-test-config.yml
```

### Expected Performance

| Configuration | RPS | Latency (p95) | CPU Usage | Memory Usage |
|---------------|-----|---------------|-----------|--------------|
| NGINX (2 cores) | 15,000 | 25ms | 30% | 512MB |
| HAProxy (2 cores) | 12,000 | 30ms | 35% | 256MB |
| Traefik (2 cores) | 8,000 | 45ms | 40% | 768MB |

---

**🔗 Related Documentation**
- [A2A Caching System](../caching/README.md)
- [Performance Monitoring](../monitoring/README.md)  
- [Security Guidelines](../security/README.md)
- [Deployment Guide](../deployment/README.md)