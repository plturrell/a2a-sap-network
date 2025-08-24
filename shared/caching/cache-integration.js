/**
 * A2A Platform Cache Integration
 * Seamless integration of Redis caching into the A2A platform services
 */

const { A2ARedisCache } = require('./redis-cache-manager');
const { performance } = require('perf_hooks');
const EventEmitter = require('events');

class A2ACacheIntegration extends EventEmitter {
    constructor(options = {}) {
        super();
        
        this.config = {
            // Cache settings
            enabled: options.enabled !== false,
            defaultStrategy: options.defaultStrategy || 'cache-aside',
            keyNamespace: options.keyNamespace || 'a2a',
            
            // Performance settings
            enableMetrics: options.enableMetrics !== false,
            metricsInterval: options.metricsInterval || 60000, // 1 minute
            
            // Circuit breaker
            circuitBreakerThreshold: options.circuitBreakerThreshold || 5,
            circuitBreakerTimeout: options.circuitBreakerTimeout || 30000,
            
            // Redis config
            redis: {
                host: options.redis?.host || process.env.REDIS_HOST || 'localhost',
                port: options.redis?.port || process.env.REDIS_PORT || 6379,
                password: options.redis?.password || process.env.REDIS_PASSWORD,
                cluster: options.redis?.cluster || false,
                ...options.redis
            },
            
            ...options
        };
        
        this.cache = null;
        this.isConnected = false;
        this.circuitBreaker = {
            failures: 0,
            lastFailure: null,
            state: 'closed' // closed, open, half-open
        };
        
        this.metrics = {
            requests: 0,
            hits: 0,
            misses: 0,
            errors: 0,
            totalTime: 0,
            lastReset: Date.now()
        };
        
        this.metricsInterval = null;
    }
    
    async initialize() {
        if (!this.config.enabled) {
            console.log('A2A Cache: Disabled by configuration');
            return this;
        }
        
        try {
            // Initialize Redis cache
            this.cache = new A2ARedisCache({
                ...this.config.redis,
                keyPrefix: `${this.config.keyNamespace}:`,
                enableMetrics: this.config.enableMetrics
            });
            
            // Setup event handlers
            this.cache.on('connected', () => {
                this.isConnected = true;
                this.emit('cache:connected');
                console.log('A2A Cache: Connected to Redis');
            });
            
            this.cache.on('error', (error) => {
                this.handleCacheError(error);
            });
            
            this.cache.on('disconnected', () => {
                this.isConnected = false;
                this.emit('cache:disconnected');
            });
            
            // Connect to Redis
            await this.cache.connect();
            
            // Start metrics collection
            if (this.config.enableMetrics) {
                this.startMetricsCollection();
            }
            
            return this;
            
        } catch (error) {
            console.error('A2A Cache: Failed to initialize:', error);
            this.config.enabled = false; // Disable caching on initialization failure
            return this;
        }
    }
    
    // Core caching decorators and methods
    cacheDecorator(options = {}) {
        return (target, propertyName, descriptor) => {
            const originalMethod = descriptor.value;
            const cacheKey = options.key || `${target.constructor.name}:${propertyName}`;
            const ttl = options.ttl || 3600;
            const strategy = options.strategy || this.config.defaultStrategy;
            
            descriptor.value = async function(...args) {
                const instance = this;
                const fullKey = typeof cacheKey === 'function' ? cacheKey.call(instance, ...args) : `${cacheKey}:${JSON.stringify(args)}`;
                
                return await instance._cacheWrapper(
                    fullKey,
                    () => originalMethod.apply(instance, args),
                    { ttl, strategy, tags: options.tags }
                );
            };
            
            return descriptor;
        };
    }
    
    async _cacheWrapper(key, fetcher, options = {}) {
        if (!this.isEnabled()) {
            return await fetcher();
        }
        
        const startTime = performance.now();
        
        try {
            switch (options.strategy) {
                case 'cache-aside':
                    return await this._cacheAsideStrategy(key, fetcher, options);
                case 'write-through':
                    return await this._writeThroughStrategy(key, fetcher, options);
                case 'write-behind':
                    return await this._writeBehindStrategy(key, fetcher, options);
                case 'refresh-ahead':
                    return await this._refreshAheadStrategy(key, fetcher, options);
                default:
                    return await this._cacheAsideStrategy(key, fetcher, options);
            }
        } finally {
            this.metrics.totalTime += performance.now() - startTime;
            this.metrics.requests++;
        }
    }
    
    async _cacheAsideStrategy(key, fetcher, options) {
        // Try to get from cache first
        const cachedValue = await this.safeGet(key);
        
        if (cachedValue !== null) {
            this.metrics.hits++;
            this.emit('cache:hit', { key, strategy: 'cache-aside' });
            return cachedValue;
        }
        
        // Cache miss - fetch data
        this.metrics.misses++;
        const value = await fetcher();
        
        // Store in cache
        await this.safeSet(key, value, options);
        
        this.emit('cache:miss', { key, strategy: 'cache-aside' });
        return value;
    }
    
    async _writeThroughStrategy(key, fetcher, options) {
        // For write-through, we assume fetcher also persists data
        const value = await fetcher();
        
        // Write to cache synchronously
        await this.safeSet(key, value, options);
        
        this.emit('cache:write-through', { key });
        return value;
    }
    
    async _writeBehindStrategy(key, fetcher, options) {
        // Check cache first
        const cachedValue = await this.safeGet(key);
        
        if (cachedValue !== null) {
            this.metrics.hits++;
            return cachedValue;
        }
        
        // Cache miss
        this.metrics.misses++;
        const value = await fetcher();
        
        // Write to cache immediately, persist in background
        await this.safeSet(key, value, options);
        
        // Schedule background persistence
        setImmediate(() => {
            this.emit('cache:background-persist', { key, value });
        });
        
        return value;
    }
    
    async _refreshAheadStrategy(key, fetcher, options) {
        const cachedValue = await this.safeGet(key);
        
        if (cachedValue !== null) {
            this.metrics.hits++;
            
            // Check if we should refresh proactively
            const ttl = await this.safeTtl(key);
            const refreshThreshold = (options.ttl || 3600) * 0.2; // Refresh when 20% TTL remaining
            
            if (ttl > 0 && ttl < refreshThreshold) {
                // Trigger background refresh
                setImmediate(async () => {
                    try {
                        const freshValue = await fetcher();
                        await this.safeSet(key, freshValue, options);
                        this.emit('cache:refresh-ahead', { key });
                    } catch (error) {
                        console.warn('A2A Cache: Background refresh failed:', error);
                    }
                });
            }
            
            return cachedValue;
        }
        
        // Cache miss
        this.metrics.misses++;
        const value = await fetcher();
        await this.safeSet(key, value, options);
        
        return value;
    }
    
    // Agent-specific caching methods
    async cacheAgentAnalysis(agentId, analysisType, input, fetcher, options = {}) {
        const key = `agent:${agentId}:analysis:${analysisType}:${this.hashInput(input)}`;
        return await this._cacheWrapper(key, fetcher, {
            ttl: options.ttl || 7200, // 2 hours for analysis results
            tags: ['agent-analysis', `agent:${agentId}`, `analysis:${analysisType}`],
            ...options
        });
    }
    
    async cacheAgentConfiguration(agentId, fetcher, options = {}) {
        const key = `agent:${agentId}:config`;
        return await this._cacheWrapper(key, fetcher, {
            ttl: options.ttl || 1800, // 30 minutes for config
            tags: ['agent-config', `agent:${agentId}`],
            ...options
        });
    }
    
    async cacheNetworkServiceResponse(service, method, params, fetcher, options = {}) {
        const key = `service:${service}:${method}:${this.hashInput(params)}`;
        return await this._cacheWrapper(key, fetcher, {
            ttl: options.ttl || 3600, // 1 hour for service responses
            tags: ['service-response', `service:${service}`],
            ...options
        });
    }
    
    async cacheUserSession(userId, fetcher, options = {}) {
        const key = `user:${userId}:session`;
        return await this._cacheWrapper(key, fetcher, {
            ttl: options.ttl || 1800, // 30 minutes for user sessions
            tags: ['user-session', `user:${userId}`],
            ...options
        });
    }
    
    // Cache invalidation methods
    async invalidateAgent(agentId) {
        await this.safeInvalidateByTag(`agent:${agentId}`);
        this.emit('cache:invalidate:agent', { agentId });
    }
    
    async invalidateService(serviceName) {
        await this.safeInvalidateByTag(`service:${serviceName}`);
        this.emit('cache:invalidate:service', { serviceName });
    }
    
    async invalidateUser(userId) {
        await this.safeInvalidateByTag(`user:${userId}`);
        this.emit('cache:invalidate:user', { userId });
    }
    
    async invalidateAnalysisType(analysisType) {
        await this.safeInvalidateByTag(`analysis:${analysisType}`);
        this.emit('cache:invalidate:analysis', { analysisType });
    }
    
    // Safe cache operations with circuit breaker
    async safeGet(key) {
        if (!this.canExecute()) {
            return null;
        }
        
        try {
            const result = await this.cache.get(key);
            this.recordSuccess();
            return result;
        } catch (error) {
            this.recordFailure(error);
            return null;
        }
    }
    
    async safeSet(key, value, options = {}) {
        if (!this.canExecute()) {
            return false;
        }
        
        try {
            let result;
            if (options.tags) {
                result = await this.cache.set(key, value, { ttl: options.ttl, tags: options.tags });
            } else {
                result = await this.cache.set(key, value, options);
            }
            this.recordSuccess();
            return result;
        } catch (error) {
            this.recordFailure(error);
            return false;
        }
    }
    
    async safeDelete(key) {
        if (!this.canExecute()) {
            return false;
        }
        
        try {
            const result = await this.cache.delete(key);
            this.recordSuccess();
            return result;
        } catch (error) {
            this.recordFailure(error);
            return false;
        }
    }
    
    async safeTtl(key) {
        if (!this.canExecute()) {
            return -1;
        }
        
        try {
            const result = await this.cache.touch(key);
            this.recordSuccess();
            return result;
        } catch (error) {
            this.recordFailure(error);
            return -1;
        }
    }
    
    async safeInvalidateByTag(tag) {
        if (!this.canExecute()) {
            return 0;
        }
        
        try {
            const result = await this.cache.invalidateByTag(tag);
            this.recordSuccess();
            return result;
        } catch (error) {
            this.recordFailure(error);
            return 0;
        }
    }
    
    // Circuit breaker implementation
    canExecute() {
        if (!this.isEnabled()) {
            return false;
        }
        
        const now = Date.now();
        
        switch (this.circuitBreaker.state) {
            case 'closed':
                return true;
            case 'open':
                if (now - this.circuitBreaker.lastFailure > this.config.circuitBreakerTimeout) {
                    this.circuitBreaker.state = 'half-open';
                    return true;
                }
                return false;
            case 'half-open':
                return true;
            default:
                return true;
        }
    }
    
    recordSuccess() {
        this.circuitBreaker.failures = 0;
        if (this.circuitBreaker.state === 'half-open') {
            this.circuitBreaker.state = 'closed';
        }
    }
    
    recordFailure(error) {
        this.metrics.errors++;
        this.circuitBreaker.failures++;
        this.circuitBreaker.lastFailure = Date.now();
        
        if (this.circuitBreaker.failures >= this.config.circuitBreakerThreshold) {
            this.circuitBreaker.state = 'open';
        }
        
        this.emit('cache:error', error);
    }
    
    handleCacheError(error) {
        console.error('A2A Cache: Error occurred:', error);
        this.recordFailure(error);
    }
    
    // Utility methods
    isEnabled() {
        return this.config.enabled && this.isConnected;
    }
    
    hashInput(input) {
        const crypto = require('crypto');
        return crypto.createHash('md5').update(JSON.stringify(input)).digest('hex');
    }
    
    // Metrics and monitoring
    startMetricsCollection() {
        this.metricsInterval = setInterval(() => {
            this.emitMetrics();
        }, this.config.metricsInterval);
    }
    
    stopMetricsCollection() {
        if (this.metricsInterval) {
            clearInterval(this.metricsInterval);
            this.metricsInterval = null;
        }
    }
    
    emitMetrics() {
        const currentMetrics = this.getMetrics();
        this.emit('cache:metrics', currentMetrics);
        
        // Reset counters for next interval
        this.resetMetrics();
    }
    
    getMetrics() {
        const hitRate = this.metrics.requests > 0 
            ? (this.metrics.hits / this.metrics.requests * 100).toFixed(2)
            : 0;
        
        const avgResponseTime = this.metrics.requests > 0
            ? (this.metrics.totalTime / this.metrics.requests).toFixed(2)
            : 0;
        
        return {
            ...this.metrics,
            hitRate: parseFloat(hitRate),
            avgResponseTime: parseFloat(avgResponseTime),
            circuitBreakerState: this.circuitBreaker.state,
            isConnected: this.isConnected,
            isEnabled: this.config.enabled,
            redisMetrics: this.cache ? this.cache.getMetrics() : null
        };
    }
    
    resetMetrics() {
        this.metrics = {
            requests: 0,
            hits: 0,
            misses: 0,
            errors: 0,
            totalTime: 0,
            lastReset: Date.now()
        };
    }
    
    // Health check
    async healthCheck() {
        const health = {
            status: 'healthy',
            enabled: this.config.enabled,
            connected: this.isConnected,
            circuitBreaker: this.circuitBreaker.state,
            metrics: this.getMetrics()
        };
        
        if (!this.config.enabled) {
            health.status = 'disabled';
        } else if (!this.isConnected) {
            health.status = 'unhealthy';
            health.reason = 'Redis connection failed';
        } else if (this.circuitBreaker.state === 'open') {
            health.status = 'degraded';
            health.reason = 'Circuit breaker is open';
        }
        
        try {
            if (this.isEnabled()) {
                const testKey = `health:${Date.now()}`;
                await this.safeSet(testKey, 'test', { ttl: 5 });
                const testValue = await this.safeGet(testKey);
                await this.safeDelete(testKey);
                
                if (testValue !== 'test') {
                    health.status = 'unhealthy';
                    health.reason = 'Cache read/write test failed';
                }
            }
        } catch (error) {
            health.status = 'unhealthy';
            health.reason = `Health check failed: ${error.message}`;
        }
        
        return health;
    }
    
    // Cleanup
    async shutdown() {
        this.stopMetricsCollection();
        
        if (this.cache) {
            await this.cache.disconnect();
        }
        
        this.emit('cache:shutdown');
    }
}

// Factory function
function createCacheIntegration(options = {}) {
    return new A2ACacheIntegration(options);
}

// Global instance for application-wide use
let globalInstance = null;

async function getGlobalCacheIntegration(options = {}) {
    if (!globalInstance) {
        globalInstance = new A2ACacheIntegration(options);
        await globalInstance.initialize();
    }
    return globalInstance;
}

// Express.js middleware
function cacheMiddleware(options = {}) {
    return async (req, res, next) => {
        try {
            const cache = await getGlobalCacheIntegration();
            req.cache = cache;
            next();
        } catch (error) {
            console.error('Cache middleware error:', error);
            req.cache = null; // Disable caching for this request
            next();
        }
    };
}

// CAP service integration
function integrateCAPService(service, cacheOptions = {}) {
    const cache = createCacheIntegration(cacheOptions);
    
    service.before('*', async (req) => {
        req.cache = cache;
    });
    
    return cache;
}

module.exports = {
    A2ACacheIntegration,
    createCacheIntegration,
    getGlobalCacheIntegration,
    cacheMiddleware,
    integrateCAPService
};