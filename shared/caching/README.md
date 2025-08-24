# A2A Platform Distributed Caching System

Advanced Redis-based distributed caching system designed for the A2A Platform with intelligent caching strategies, high availability, and enterprise-grade performance.

## 🚀 Features

### Core Capabilities
- **Multiple Caching Strategies**: Cache-aside, Write-through, Write-behind, Refresh-ahead
- **Distributed Locking**: Prevents cache stampede with Redis-based distributed locks
- **Tag-based Invalidation**: Group-based cache invalidation for related data
- **Circuit Breaker**: Automatic failover when Redis is unavailable
- **Performance Monitoring**: Real-time metrics and health checks
- **High Availability**: Redis Cluster and Sentinel support

### A2A Platform Integration
- **Agent Caching**: Intelligent caching for agent analysis results
- **Service Response Caching**: Network service response optimization
- **User Session Caching**: Fast user data retrieval
- **Configuration Caching**: Agent and service configuration optimization

## 📁 Architecture

```
shared/caching/
├── redis-cache-manager.js      # Core Redis cache manager
├── cache-strategies.py         # Python caching strategies
├── cache-integration.js        # A2A platform integration
├── docker-compose-redis.yml    # Redis cluster deployment
└── README.md                   # This documentation
```

## 🛠️ Installation & Setup

### 1. Redis Deployment

**Development (Single Instance)**:
```bash
docker-compose -f docker-compose-redis.yml --profile standalone up -d
```

**Production (Redis Cluster)**:
```bash
docker-compose -f docker-compose-redis.yml up -d
```

**With Monitoring & Tools**:
```bash
docker-compose -f docker-compose-redis.yml --profile tools --profile monitoring --profile backup up -d
```

### 2. Node.js Dependencies

```bash
npm install redis ioredis
```

### 3. Python Dependencies

```bash
pip install redis asyncio
```

## 🔧 Configuration

### Environment Variables

```bash
# Redis Connection
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-password
REDIS_CLUSTER=false

# Cache Settings
CACHE_DEFAULT_TTL=3600
CACHE_KEY_PREFIX=a2a
CACHE_STRATEGY=cache-aside

# Circuit Breaker
CACHE_CIRCUIT_BREAKER_THRESHOLD=5
CACHE_CIRCUIT_BREAKER_TIMEOUT=30000

# Monitoring
CACHE_METRICS_ENABLED=true
CACHE_METRICS_INTERVAL=60000
```

## 💻 Usage Examples

### Basic Cache Operations

```javascript
const { createCacheIntegration } = require('./cache-integration');

// Initialize cache
const cache = createCacheIntegration({
    redis: {
        host: 'localhost',
        port: 6379
    },
    defaultStrategy: 'cache-aside'
});

await cache.initialize();

// Cache-aside pattern
const userData = await cache.cacheUserSession(
    userId, 
    async () => {
        return await database.getUser(userId);
    },
    { ttl: 1800 }
);
```

### Agent Analysis Caching

```javascript
// Cache agent analysis results
const analysisResult = await cache.cacheAgentAnalysis(
    'glean-agent',
    'security-scan',
    { directory: '/path/to/code' },
    async () => {
        return await gleanAgent.performSecurityAnalysis('/path/to/code');
    },
    { ttl: 7200 }
);
```

### Service Response Caching

```javascript
// Cache network service responses
const serviceResponse = await cache.cacheNetworkServiceResponse(
    'agent-registry',
    'getAgentCapabilities',
    { agentId: 'agent-0' },
    async () => {
        return await agentRegistry.getCapabilities('agent-0');
    },
    { ttl: 3600 }
);
```

### Python Async Caching

```python
from cache_strategies import A2ACacheManager, CacheConfig, CacheStrategy

# Initialize cache manager
config = CacheConfig(
    host="localhost",
    port=6379,
    strategy=CacheStrategy.CACHE_ASIDE,
    default_ttl=3600
)

cache = A2ACacheManager(config)
await cache.connect()

# Cache with fetcher function
async def fetch_user_data(user_id: int):
    # Simulate database call
    return await database.get_user(user_id)

user_data = await cache.get_or_set(
    "user:123",
    lambda: fetch_user_data(123),
    ttl=1800
)
```

## 🎯 Caching Strategies

### 1. Cache-Aside (Lazy Loading)
```javascript
// Application manages cache
const value = await cache.get(key);
if (value === null) {
    value = await database.fetch(key);
    await cache.set(key, value);
}
return value;
```

### 2. Write-Through
```javascript
// Write to cache and database simultaneously
await database.save(key, value);
await cache.set(key, value);
```

### 3. Write-Behind (Write-Back)
```javascript
// Write to cache immediately, database asynchronously
await cache.set(key, value);
// Background process handles database persistence
```

### 4. Refresh-Ahead
```javascript
// Proactively refresh cache before expiration
if (ttl < refreshThreshold) {
    // Trigger background refresh
    backgroundRefresh(key, fetcher);
}
return cachedValue;
```

## 🔍 Monitoring & Metrics

### Health Check Endpoint

```javascript
// Health check
const health = await cache.healthCheck();
console.log(health);
/*
{
  status: 'healthy',
  enabled: true,
  connected: true,
  circuitBreaker: 'closed',
  metrics: {
    hitRate: 85.5,
    avgResponseTime: 12.3,
    requests: 1000,
    hits: 855,
    misses: 145
  }
}
*/
```

### Metrics Collection

```javascript
// Listen for metrics events
cache.on('cache:metrics', (metrics) => {
    console.log('Cache metrics:', metrics);
    
    // Send to monitoring system
    prometheus.cacheHitRate.set(metrics.hitRate);
    prometheus.cacheResponseTime.set(metrics.avgResponseTime);
});
```

### Redis Web UI

Access Redis Commander at: `http://localhost:8081`
- Username: `admin`
- Password: `admin123` (configurable)

## 🏷️ Tag-Based Invalidation

### Setting Tags

```javascript
// Set cache with tags
await cache.safeSet('user:123:profile', userData, {
    ttl: 3600,
    tags: ['user:123', 'user-profiles', 'sensitive-data']
});
```

### Invalidating by Tag

```javascript
// Invalidate all user data
await cache.invalidateUser('123');

// Invalidate all agent analysis
await cache.invalidateAnalysisType('security-scan');

// Invalidate service responses
await cache.invalidateService('agent-registry');
```

## 🔐 Security & Best Practices

### 1. Connection Security

```javascript
const cache = createCacheIntegration({
    redis: {
        host: 'redis.example.com',
        port: 6380,
        password: process.env.REDIS_PASSWORD,
        tls: {
            servername: 'redis.example.com'
        }
    }
});
```

### 2. Data Serialization

```javascript
// Automatic serialization/deserialization
const complexObject = {
    user: { id: 123, name: 'John' },
    permissions: ['read', 'write'],
    metadata: { lastLogin: new Date() }
};

await cache.set('user:session', complexObject);
const retrieved = await cache.get('user:session'); // Automatically deserialized
```

### 3. Circuit Breaker Pattern

```javascript
// Cache automatically falls back to source data when Redis is unavailable
const userData = await cache.cacheUserSession(userId, async () => {
    return await database.getUser(userId); // Fallback to database
});
```

## 🚀 Performance Optimization

### 1. Connection Pooling

```javascript
const cache = createCacheIntegration({
    redis: {
        maxRetriesPerRequest: 3,
        retryDelayOnFailover: 100,
        maxmemoryPolicy: 'allkeys-lru',
        // Connection pool settings
        lazyConnect: true,
        keepAlive: 30000,
        maxRetriesPerRequest: 3
    }
});
```

### 2. Batch Operations

```javascript
// Multiple get operations
const keys = ['user:1', 'user:2', 'user:3'];
const values = await cache.cache.mget(keys);

// Multiple set operations
await cache.cache.mset({
    'user:1': userData1,
    'user:2': userData2,
    'user:3': userData3
}, { ttl: 3600 });
```

### 3. Compression

```javascript
// Enable compression for large objects
const cache = createCacheIntegration({
    compression: true,
    serialization: 'msgpack' // More efficient than JSON
});
```

## 🏗️ Integration with A2A Services

### Express.js Middleware

```javascript
const express = require('express');
const { cacheMiddleware } = require('./cache-integration');

const app = express();

// Add cache to all requests
app.use(cacheMiddleware({
    redis: { host: 'localhost', port: 6379 }
}));

app.get('/api/users/:id', async (req, res) => {
    const userData = await req.cache.cacheUserSession(
        req.params.id,
        () => User.findById(req.params.id)
    );
    res.json(userData);
});
```

### SAP CAP Integration

```javascript
const { integrateCAPService } = require('./cache-integration');

// In your CAP service
module.exports = (srv) => {
    const cache = integrateCAPService(srv, {
        redis: { host: 'localhost', port: 6379 }
    });
    
    srv.on('READ', 'Users', async (req) => {
        return await req.cache.cacheNetworkServiceResponse(
            'user-service',
            'getUsers',
            req.data,
            () => SELECT.from('Users').where(req.data)
        );
    });
};
```

## 📊 Redis Cluster Management

### Cluster Information

```bash
# Check cluster status
docker exec -it a2a-redis-cluster-1 redis-cli --cluster check localhost:7001

# View cluster nodes
docker exec -it a2a-redis-cluster-1 redis-cli cluster nodes

# Cluster info
docker exec -it a2a-redis-cluster-1 redis-cli cluster info
```

### Scaling the Cluster

```bash
# Add new node
docker-compose -f docker-compose-redis.yml up -d redis-cluster-node-7

# Add node to cluster
docker exec -it a2a-redis-cluster-1 redis-cli --cluster add-node new-node:7007 existing-node:7001

# Rebalance slots
docker exec -it a2a-redis-cluster-1 redis-cli --cluster rebalance localhost:7001
```

## 🐛 Troubleshooting

### Common Issues

**Cache Miss Rate Too High**:
```javascript
// Check TTL values
const metrics = cache.getMetrics();
console.log('Hit rate:', metrics.hitRate);

// Increase TTL for frequently accessed data
await cache.set(key, value, { ttl: 7200 }); // 2 hours instead of 1
```

**Memory Usage Issues**:
```bash
# Check Redis memory usage
docker exec -it a2a-redis-standalone redis-cli info memory

# Set memory limit
docker exec -it a2a-redis-standalone redis-cli config set maxmemory 1gb
docker exec -it a2a-redis-standalone redis-cli config set maxmemory-policy allkeys-lru
```

**Connection Problems**:
```javascript
// Enable debug logging
const cache = createCacheIntegration({
    redis: {
        host: 'localhost',
        port: 6379,
        retryDelayOnFailover: 100,
        maxRetriesPerRequest: 3,
        lazyConnect: true
    }
});

cache.on('cache:error', (error) => {
    console.error('Cache error:', error);
});
```

### Performance Tuning

**Optimize Key Design**:
```javascript
// Good: Hierarchical keys
'user:123:profile'
'user:123:permissions'
'agent:glean:analysis:security:hash123'

// Bad: Long random keys
'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6'
```

**Use Appropriate TTL Values**:
```javascript
// Frequently changing data: Short TTL
await cache.set('user:session', sessionData, { ttl: 300 }); // 5 minutes

// Rarely changing data: Long TTL
await cache.set('agent:config', configData, { ttl: 86400 }); // 24 hours

// Static data: Very long TTL
await cache.set('system:constants', constants, { ttl: 604800 }); // 1 week
```

## 📈 Production Deployment

### Docker Compose Production

```yaml
# Production environment with SSL and monitoring
services:
  redis-cluster:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
    volumes:
      - /data/redis:/data
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD}
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
spec:
  serviceName: redis-cluster
  replicas: 6
  template:
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

---

**🔗 Related Documentation**
- [A2A Platform Architecture](../README.md)
- [Performance Monitoring](../monitoring/README.md)
- [Security Guidelines](../security/README.md)
- [Deployment Guide](../deployment/README.md)