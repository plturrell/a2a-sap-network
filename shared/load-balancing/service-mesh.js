/**
 * Service Mesh Implementation for A2A Platform
 * Advanced service-to-service communication with circuit breakers, retries, and observability
 */

const EventEmitter = require('events');
const { performance } = require('perf_hooks');
const crypto = require('crypto');

class A2AServiceMesh extends EventEmitter {
    constructor(options = {}) {
        super();
        
        this.config = {
            // Mesh configuration
            serviceName: options.serviceName || 'unknown',
            version: options.version || '1.0.0',
            
            // Circuit breaker settings
            circuitBreakerThreshold: options.circuitBreakerThreshold || 5,
            circuitBreakerTimeout: options.circuitBreakerTimeout || 60000,
            circuitBreakerHalfOpenMax: options.circuitBreakerHalfOpenMax || 3,
            
            // Retry settings
            defaultRetries: options.defaultRetries || 3,
            retryBackoff: options.retryBackoff || 'exponential', // linear, exponential, fixed
            retryDelay: options.retryDelay || 1000,
            retryJitter: options.retryJitter !== false,
            
            // Timeout settings
            defaultTimeout: options.defaultTimeout || 10000,
            connectionTimeout: options.connectionTimeout || 5000,
            
            // Load balancing
            loadBalancingStrategy: options.loadBalancingStrategy || 'round-robin',
            
            // Security
            enableMTLS: options.enableMTLS || false,
            trustStore: options.trustStore || null,
            
            // Observability
            enableTracing: options.enableTracing !== false,
            enableMetrics: options.enableMetrics !== false,
            
            ...options
        };
        
        this.services = new Map(); // service_name -> service_instances[]
        this.circuitBreakers = new Map(); // service_name -> CircuitBreaker
        this.metrics = new Map(); // service_name -> ServiceMetrics
        this.activeRequests = new Map(); // request_id -> RequestContext
        this.middlewares = [];
        
        this.isInitialized = false;
    }
    
    async initialize() {
        if (this.isInitialized) {
            return this;
        }
        
        // Initialize default middlewares
        this.use(this.tracingMiddleware());
        this.use(this.metricsMiddleware());
        this.use(this.circuitBreakerMiddleware());
        this.use(this.retryMiddleware());
        this.use(this.timeoutMiddleware());
        
        this.isInitialized = true;
        this.emit('mesh:initialized');
        
        console.log(`Service Mesh initialized for service: ${this.config.serviceName}`);
        return this;
    }
    
    // Service registration and discovery
    registerService(serviceName, instances) {
        if (!Array.isArray(instances)) {
            instances = [instances];
        }
        
        // Normalize instances
        const normalizedInstances = instances.map(instance => ({
            id: instance.id || `${serviceName}-${crypto.randomBytes(4).toString('hex')}`,
            name: serviceName,
            address: instance.address,
            port: instance.port,
            protocol: instance.protocol || 'http',
            weight: instance.weight || 1,
            metadata: instance.metadata || {},
            health: 'unknown',
            lastHealthCheck: null,
            ...instance
        }));
        
        this.services.set(serviceName, normalizedInstances);
        
        // Initialize circuit breaker for service
        if (!this.circuitBreakers.has(serviceName)) {
            this.circuitBreakers.set(serviceName, new CircuitBreaker(serviceName, this.config));
        }
        
        // Initialize metrics for service
        if (!this.metrics.has(serviceName)) {
            this.metrics.set(serviceName, new ServiceMetrics(serviceName));
        }
        
        this.emit('service:registered', { serviceName, instances: normalizedInstances });
        console.log(`Registered service: ${serviceName} with ${normalizedInstances.length} instances`);
        
        return this;
    }
    
    getServiceInstances(serviceName) {
        const instances = this.services.get(serviceName);
        if (!instances) {
            throw new Error(`Service not found: ${serviceName}`);
        }
        
        // Filter healthy instances
        return instances.filter(instance => instance.health !== 'unhealthy');
    }
    
    // Service mesh communication
    async call(serviceName, options = {}) {
        const requestId = options.requestId || crypto.randomBytes(16).toString('hex');
        const requestContext = {
            id: requestId,
            serviceName,
            startTime: performance.now(),
            attempt: 0,
            options,
            traceId: options.traceId || crypto.randomBytes(16).toString('hex'),
            spanId: crypto.randomBytes(8).toString('hex'),
            parentSpanId: options.parentSpanId || null
        };
        
        this.activeRequests.set(requestId, requestContext);
        
        try {
            // Execute middleware chain
            const result = await this.executeMiddlewareChain(requestContext);
            return result;
            
        } catch (error) {
            this.emit('request:error', { requestContext, error });
            throw error;
            
        } finally {
            this.activeRequests.delete(requestId);
            
            // Emit request completed event
            const duration = performance.now() - requestContext.startTime;
            this.emit('request:completed', {
                requestId,
                serviceName,
                duration,
                success: !requestContext.error
            });
        }
    }
    
    // Middleware system
    use(middleware) {
        if (typeof middleware !== 'function') {
            throw new Error('Middleware must be a function');
        }
        
        this.middlewares.push(middleware);
        return this;
    }
    
    async executeMiddlewareChain(context, index = 0) {
        if (index >= this.middlewares.length) {
            // Execute final request
            return await this.executeRequest(context);
        }
        
        const middleware = this.middlewares[index];
        
        return await middleware(context, async () => {
            return await this.executeMiddlewareChain(context, index + 1);
        });
    }
    
    async executeRequest(context) {
        const { serviceName, options } = context;
        const instances = this.getServiceInstances(serviceName);
        
        if (instances.length === 0) {
            throw new Error(`No healthy instances available for service: ${serviceName}`);
        }
        
        // Select instance based on load balancing strategy
        const instance = this.selectInstance(instances, this.config.loadBalancingStrategy);
        context.selectedInstance = instance;
        
        // Make the actual request
        const result = await this.makeRequest(instance, context);
        context.result = result;
        
        return result;
    }
    
    selectInstance(instances, strategy) {
        switch (strategy) {
            case 'round-robin':
                return this.roundRobinSelect(instances);
            case 'random':
                return instances[Math.floor(Math.random() * instances.length)];
            case 'weighted':
                return this.weightedSelect(instances);
            case 'least-connections':
                return this.leastConnectionsSelect(instances);
            default:
                return instances[0];
        }
    }
    
    roundRobinSelect(instances) {
        // Simple round-robin implementation
        const serviceName = instances[0].name;
        const metrics = this.metrics.get(serviceName);
        
        if (!metrics.roundRobinIndex) {
            metrics.roundRobinIndex = 0;
        }
        
        const instance = instances[metrics.roundRobinIndex % instances.length];
        metrics.roundRobinIndex++;
        
        return instance;
    }
    
    weightedSelect(instances) {
        const totalWeight = instances.reduce((sum, instance) => sum + instance.weight, 0);
        const random = Math.random() * totalWeight;
        
        let currentWeight = 0;
        for (const instance of instances) {
            currentWeight += instance.weight;
            if (random <= currentWeight) {
                return instance;
            }
        }
        
        return instances[instances.length - 1];
    }
    
    leastConnectionsSelect(instances) {
        return instances.reduce((best, current) => {
            const bestMetrics = this.metrics.get(best.name);
            const currentMetrics = this.metrics.get(current.name);
            
            const bestConnections = bestMetrics ? bestMetrics.activeConnections : 0;
            const currentConnections = currentMetrics ? currentMetrics.activeConnections : 0;
            
            return currentConnections < bestConnections ? current : best;
        });
    }
    
    async makeRequest(instance, context) {
        const { options } = context;
        const url = `${instance.protocol}://${instance.address}:${instance.port}${options.path || ''}`;
        
        // Simulate HTTP request (in real implementation, use actual HTTP client)
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error(`Request timeout: ${url}`));
            }, options.timeout || this.config.defaultTimeout);
            
            // Simulate network request
            setTimeout(() => {
                clearTimeout(timeout);
                
                // Simulate occasional failures for testing
                if (Math.random() < 0.1) {
                    reject(new Error(`Network error: ${url}`));
                } else {
                    resolve({
                        status: 200,
                        data: { message: 'Success', instance: instance.id, timestamp: Date.now() },
                        headers: { 'x-instance-id': instance.id }
                    });
                }
            }, Math.random() * 100 + 50); // 50-150ms response time
        });
    }
    
    // Built-in middlewares
    tracingMiddleware() {
        return async (context, next) => {
            if (!this.config.enableTracing) {
                return await next();
            }
            
            const span = {
                traceId: context.traceId,
                spanId: context.spanId,
                parentSpanId: context.parentSpanId,
                operationName: `${this.config.serviceName} -> ${context.serviceName}`,
                startTime: performance.now(),
                tags: {
                    'service.name': this.config.serviceName,
                    'service.version': this.config.version,
                    'target.service': context.serviceName,
                    'request.id': context.id
                }
            };
            
            context.span = span;
            this.emit('tracing:span:start', span);
            
            try {
                const result = await next();
                
                span.tags['http.status_code'] = context.result?.status || 'unknown';
                span.tags['success'] = true;
                
                return result;
                
            } catch (error) {
                span.tags['error'] = true;
                span.tags['error.message'] = error.message;
                throw error;
                
            } finally {
                span.duration = performance.now() - span.startTime;
                this.emit('tracing:span:finish', span);
            }
        };
    }
    
    metricsMiddleware() {
        return async (context, next) => {
            if (!this.config.enableMetrics) {
                return await next();
            }
            
            const metrics = this.metrics.get(context.serviceName);
            const startTime = performance.now();
            
            metrics.totalRequests++;
            metrics.activeConnections++;
            
            try {
                const result = await next();
                
                const duration = performance.now() - startTime;
                metrics.totalResponseTime += duration;
                metrics.successfulRequests++;
                
                // Update response time percentiles
                metrics.updateResponseTime(duration);
                
                this.emit('metrics:request:success', {
                    service: context.serviceName,
                    duration,
                    instance: context.selectedInstance?.id
                });
                
                return result;
                
            } catch (error) {
                metrics.failedRequests++;
                
                this.emit('metrics:request:failure', {
                    service: context.serviceName,
                    error: error.message,
                    instance: context.selectedInstance?.id
                });
                
                throw error;
                
            } finally {
                metrics.activeConnections--;
            }
        };
    }
    
    circuitBreakerMiddleware() {
        return async (context, next) => {
            const circuitBreaker = this.circuitBreakers.get(context.serviceName);
            
            if (!circuitBreaker.canExecute()) {
                throw new Error(`Circuit breaker is open for service: ${context.serviceName}`);
            }
            
            try {
                const result = await next();
                circuitBreaker.recordSuccess();
                return result;
                
            } catch (error) {
                circuitBreaker.recordFailure();
                throw error;
            }
        };
    }
    
    retryMiddleware() {
        return async (context, next) => {
            const maxRetries = context.options.retries || this.config.defaultRetries;
            let lastError;
            
            for (let attempt = 0; attempt <= maxRetries; attempt++) {
                context.attempt = attempt;
                
                try {
                    const result = await next();
                    
                    if (attempt > 0) {
                        this.emit('request:retry:success', {
                            service: context.serviceName,
                            attempt,
                            requestId: context.id
                        });
                    }
                    
                    return result;
                    
                } catch (error) {
                    lastError = error;
                    
                    // Don't retry on certain errors
                    if (this.isNonRetryableError(error)) {
                        throw error;
                    }
                    
                    if (attempt < maxRetries) {
                        const delay = this.calculateRetryDelay(attempt);
                        
                        this.emit('request:retry:attempt', {
                            service: context.serviceName,
                            attempt: attempt + 1,
                            delay,
                            error: error.message,
                            requestId: context.id
                        });
                        
                        await this.sleep(delay);
                    }
                }
            }
            
            throw lastError;
        };
    }
    
    timeoutMiddleware() {
        return async (context, next) => {
            const timeout = context.options.timeout || this.config.defaultTimeout;
            
            return new Promise(async (resolve, reject) => {
                const timeoutHandle = setTimeout(() => {
                    reject(new Error(`Request timeout after ${timeout}ms`));
                }, timeout);
                
                try {
                    const result = await next();
                    clearTimeout(timeoutHandle);
                    resolve(result);
                } catch (error) {
                    clearTimeout(timeoutHandle);
                    reject(error);
                }
            });
        };
    }
    
    // Utility methods
    isNonRetryableError(error) {
        // Don't retry on authentication errors, bad requests, etc.
        const nonRetryablePatterns = [
            /authentication/i,
            /authorization/i,
            /bad request/i,
            /not found/i,
            /circuit breaker/i
        ];
        
        return nonRetryablePatterns.some(pattern => pattern.test(error.message));
    }
    
    calculateRetryDelay(attempt) {
        const baseDelay = this.config.retryDelay;
        let delay;
        
        switch (this.config.retryBackoff) {
            case 'linear':
                delay = baseDelay * (attempt + 1);
                break;
            case 'exponential':
                delay = baseDelay * Math.pow(2, attempt);
                break;
            case 'fixed':
            default:
                delay = baseDelay;
                break;
        }
        
        // Add jitter to prevent thundering herd
        if (this.config.retryJitter) {
            delay += Math.random() * delay * 0.1;
        }
        
        return Math.min(delay, 30000); // Cap at 30 seconds
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    // Health checking and monitoring
    async healthCheck() {
        const health = {
            service: this.config.serviceName,
            version: this.config.version,
            status: 'healthy',
            timestamp: new Date().toISOString(),
            services: {},
            circuitBreakers: {},
            activeRequests: this.activeRequests.size
        };
        
        // Check registered services
        for (const [serviceName, instances] of this.services) {
            const healthyInstances = instances.filter(i => i.health !== 'unhealthy').length;
            const totalInstances = instances.length;
            
            health.services[serviceName] = {
                totalInstances,
                healthyInstances,
                status: healthyInstances > 0 ? 'available' : 'unavailable'
            };
        }
        
        // Check circuit breakers
        for (const [serviceName, circuitBreaker] of this.circuitBreakers) {
            health.circuitBreakers[serviceName] = {
                state: circuitBreaker.state,
                failures: circuitBreaker.failures,
                lastFailure: circuitBreaker.lastFailure
            };
        }
        
        // Determine overall health
        const hasUnavailableServices = Object.values(health.services).some(s => s.status === 'unavailable');
        const hasOpenCircuitBreakers = Object.values(health.circuitBreakers).some(cb => cb.state === 'open');
        
        if (hasUnavailableServices || hasOpenCircuitBreakers) {
            health.status = 'degraded';
        }
        
        return health;
    }
    
    getMetrics() {
        const metrics = {};
        
        for (const [serviceName, serviceMetrics] of this.metrics) {
            metrics[serviceName] = serviceMetrics.getMetrics();
        }
        
        return {
            mesh: {
                service: this.config.serviceName,
                activeRequests: this.activeRequests.size,
                registeredServices: this.services.size
            },
            services: metrics
        };
    }
    
    // Cleanup
    async shutdown() {
        this.emit('mesh:shutdown');
        
        // Cancel active requests
        for (const [requestId, context] of this.activeRequests) {
            context.cancelled = true;
            this.emit('request:cancelled', { requestId });
        }
        
        this.activeRequests.clear();
        console.log(`Service Mesh shutdown for service: ${this.config.serviceName}`);
    }
}

// Circuit Breaker Implementation
class CircuitBreaker {
    constructor(serviceName, config) {
        this.serviceName = serviceName;
        this.threshold = config.circuitBreakerThreshold;
        this.timeout = config.circuitBreakerTimeout;
        this.halfOpenMax = config.circuitBreakerHalfOpenMax;
        
        this.state = 'closed'; // closed, open, half-open
        this.failures = 0;
        this.successes = 0;
        this.lastFailure = null;
        this.halfOpenRequests = 0;
    }
    
    canExecute() {
        switch (this.state) {
            case 'closed':
                return true;
            case 'open':
                if (Date.now() - this.lastFailure > this.timeout) {
                    this.state = 'half-open';
                    this.halfOpenRequests = 0;
                    return true;
                }
                return false;
            case 'half-open':
                return this.halfOpenRequests < this.halfOpenMax;
        }
    }
    
    recordSuccess() {
        this.successes++;
        
        if (this.state === 'half-open') {
            this.halfOpenRequests++;
            if (this.halfOpenRequests >= this.halfOpenMax) {
                this.state = 'closed';
                this.failures = 0;
            }
        } else {
            this.failures = Math.max(0, this.failures - 1);
        }
    }
    
    recordFailure() {
        this.failures++;
        this.lastFailure = Date.now();
        
        if (this.state === 'half-open') {
            this.state = 'open';
        } else if (this.state === 'closed' && this.failures >= this.threshold) {
            this.state = 'open';
        }
    }
}

// Service Metrics Implementation
class ServiceMetrics {
    constructor(serviceName) {
        this.serviceName = serviceName;
        this.totalRequests = 0;
        this.successfulRequests = 0;
        this.failedRequests = 0;
        this.activeConnections = 0;
        this.totalResponseTime = 0;
        this.responseTimes = [];
        this.roundRobinIndex = 0;
        this.createdAt = Date.now();
    }
    
    updateResponseTime(duration) {
        this.responseTimes.push(duration);
        
        // Keep only last 1000 response times for percentile calculations
        if (this.responseTimes.length > 1000) {
            this.responseTimes.shift();
        }
    }
    
    getPercentile(percentile) {
        if (this.responseTimes.length === 0) return 0;
        
        const sorted = [...this.responseTimes].sort((a, b) => a - b);
        const index = Math.ceil((percentile / 100) * sorted.length) - 1;
        
        return sorted[Math.max(0, index)];
    }
    
    getMetrics() {
        const avgResponseTime = this.successfulRequests > 0 
            ? this.totalResponseTime / this.successfulRequests 
            : 0;
        
        const successRate = this.totalRequests > 0 
            ? (this.successfulRequests / this.totalRequests) * 100 
            : 0;
        
        return {
            serviceName: this.serviceName,
            totalRequests: this.totalRequests,
            successfulRequests: this.successfulRequests,
            failedRequests: this.failedRequests,
            activeConnections: this.activeConnections,
            avgResponseTime: Math.round(avgResponseTime * 100) / 100,
            successRate: Math.round(successRate * 100) / 100,
            p50: Math.round(this.getPercentile(50) * 100) / 100,
            p95: Math.round(this.getPercentile(95) * 100) / 100,
            p99: Math.round(this.getPercentile(99) * 100) / 100,
            uptime: Date.now() - this.createdAt
        };
    }
}

// Factory function
function createServiceMesh(options = {}) {
    return new A2AServiceMesh(options);
}

module.exports = {
    A2AServiceMesh,
    CircuitBreaker,
    ServiceMetrics,
    createServiceMesh
};