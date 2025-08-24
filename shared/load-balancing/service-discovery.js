/**
 * Advanced Service Discovery and Load Balancing for A2A Platform
 * Provides intelligent service registration, health monitoring, and request routing
 */

const EventEmitter = require('events');
const http = require('http');
const https = require('https');
const dns = require('dns').promises;
const { performance } = require('perf_hooks');

class A2AServiceRegistry extends EventEmitter {
    constructor(options = {}) {
        super();
        
        this.config = {
            // Registry settings
            registryType: options.registryType || 'memory', // memory, redis, consul, etcd
            namespace: options.namespace || 'a2a',
            
            // Health check settings
            healthCheckInterval: options.healthCheckInterval || 30000, // 30 seconds
            healthCheckTimeout: options.healthCheckTimeout || 5000,
            unhealthyThreshold: options.unhealthyThreshold || 3,
            healthyThreshold: options.healthyThreshold || 2,
            
            // Load balancing
            defaultStrategy: options.defaultStrategy || 'round-robin',
            
            // Service discovery
            discoveryInterval: options.discoveryInterval || 60000, // 1 minute
            autoDeregisterCritical: options.autoDeregisterCritical || true,
            
            ...options
        };
        
        this.services = new Map(); // service_name -> service_instances[]
        this.healthChecks = new Map(); // instance_id -> health_status
        this.loadBalancers = new Map(); // service_name -> LoadBalancer
        this.metrics = new Map(); // instance_id -> metrics
        
        this.healthCheckInterval = null;
        this.discoveryInterval = null;
        this.isRunning = false;
    }
    
    async start() {
        if (this.isRunning) {
            return;
        }
        
        this.isRunning = true;
        
        // Start health checking
        this.startHealthChecking();
        
        // Start service discovery
        this.startServiceDiscovery();
        
        this.emit('registry:started');
        console.log('A2A Service Registry started');
    }
    
    async stop() {
        this.isRunning = false;
        
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }
        
        if (this.discoveryInterval) {
            clearInterval(this.discoveryInterval);
        }
        
        this.emit('registry:stopped');
        console.log('A2A Service Registry stopped');
    }
    
    // Service registration
    async registerService(serviceDefinition) {
        const {
            id,
            name,
            address,
            port,
            protocol = 'http',
            tags = [],
            metadata = {},
            health = {}
        } = serviceDefinition;
        
        if (!id || !name || !address || !port) {
            throw new Error('Service registration requires id, name, address, and port');
        }
        
        const instance = {
            id,
            name,
            address,
            port,
            protocol,
            tags,
            metadata,
            health: {
                checkUrl: health.checkUrl || `${protocol}://${address}:${port}/health`,
                checkInterval: health.checkInterval || this.config.healthCheckInterval,
                timeout: health.timeout || this.config.healthCheckTimeout,
                ...health
            },
            registeredAt: Date.now(),
            lastSeen: Date.now(),
            status: 'starting'
        };
        
        // Add to services map
        if (!this.services.has(name)) {
            this.services.set(name, new Map());
        }
        
        this.services.get(name).set(id, instance);
        
        // Initialize health status
        this.healthChecks.set(id, {
            status: 'unknown',
            consecutiveFailures: 0,
            consecutiveSuccesses: 0,
            lastCheck: null,
            lastError: null
        });
        
        // Initialize metrics
        this.metrics.set(id, {
            requests: 0,
            responses: 0,
            errors: 0,
            totalResponseTime: 0,
            avgResponseTime: 0,
            lastRequest: null
        });
        
        // Initialize load balancer for service if not exists
        if (!this.loadBalancers.has(name)) {
            this.loadBalancers.set(name, new LoadBalancer(name, this.config.defaultStrategy));
        }
        
        this.emit('service:registered', { service: name, instance });
        console.log(`Service registered: ${name}/${id} at ${address}:${port}`);
        
        return instance;
    }
    
    async deregisterService(serviceId) {
        let found = false;
        
        for (const [serviceName, instances] of this.services) {
            if (instances.has(serviceId)) {
                instances.delete(serviceId);
                found = true;
                
                // Clean up empty service entries
                if (instances.size === 0) {
                    this.services.delete(serviceName);
                    this.loadBalancers.delete(serviceName);
                }
                
                this.emit('service:deregistered', { service: serviceName, instanceId: serviceId });
                console.log(`Service deregistered: ${serviceName}/${serviceId}`);
                break;
            }
        }
        
        // Clean up associated data
        this.healthChecks.delete(serviceId);
        this.metrics.delete(serviceId);
        
        if (!found) {
            throw new Error(`Service instance not found: ${serviceId}`);
        }
    }
    
    // Service discovery
    async discoverService(serviceName) {
        const instances = this.services.get(serviceName);
        if (!instances) {
            return [];
        }
        
        // Return only healthy instances
        const healthyInstances = [];
        for (const [id, instance] of instances) {
            const health = this.healthChecks.get(id);
            if (health && health.status === 'healthy') {
                healthyInstances.push({
                    ...instance,
                    health: health.status,
                    metrics: this.metrics.get(id)
                });
            }
        }
        
        return healthyInstances;
    }
    
    async getAllServices() {
        const serviceMap = {};
        
        for (const [serviceName, instances] of this.services) {
            serviceMap[serviceName] = [];
            
            for (const [id, instance] of instances) {
                const health = this.healthChecks.get(id);
                const metrics = this.metrics.get(id);
                
                serviceMap[serviceName].push({
                    ...instance,
                    health: health?.status || 'unknown',
                    metrics: metrics || {}
                });
            }
        }
        
        return serviceMap;
    }
    
    // Load balancing
    async getServiceInstance(serviceName, strategy = null) {
        const loadBalancer = this.loadBalancers.get(serviceName);
        if (!loadBalancer) {
            throw new Error(`Service not found: ${serviceName}`);
        }
        
        const instances = await this.discoverService(serviceName);
        if (instances.length === 0) {
            throw new Error(`No healthy instances available for service: ${serviceName}`);
        }
        
        return loadBalancer.selectInstance(instances, strategy);
    }
    
    // Health checking
    startHealthChecking() {
        this.healthCheckInterval = setInterval(async () => {
            await this.performHealthChecks();
        }, this.config.healthCheckInterval / 2); // Check more frequently than interval
    }
    
    async performHealthChecks() {
        const checkPromises = [];
        
        for (const [serviceName, instances] of this.services) {
            for (const [instanceId, instance] of instances) {
                // Check if it's time for a health check
                const health = this.healthChecks.get(instanceId);
                const now = Date.now();
                
                if (!health.lastCheck || now - health.lastCheck >= instance.health.checkInterval) {
                    checkPromises.push(this.checkInstanceHealth(instanceId, instance));
                }
            }
        }
        
        if (checkPromises.length > 0) {
            await Promise.allSettled(checkPromises);
        }
    }
    
    async checkInstanceHealth(instanceId, instance) {
        const health = this.healthChecks.get(instanceId);
        const startTime = performance.now();
        
        try {
            const isHealthy = await this.performHealthCheck(instance);
            const responseTime = performance.now() - startTime;
            
            if (isHealthy) {
                health.consecutiveSuccesses++;
                health.consecutiveFailures = 0;
                health.lastError = null;
                
                if (health.status !== 'healthy' && health.consecutiveSuccesses >= this.config.healthyThreshold) {
                    health.status = 'healthy';
                    this.emit('instance:healthy', { instanceId, instance });
                }
            } else {
                health.consecutiveFailures++;
                health.consecutiveSuccesses = 0;
                
                if (health.status !== 'unhealthy' && health.consecutiveFailures >= this.config.unhealthyThreshold) {
                    health.status = 'unhealthy';
                    this.emit('instance:unhealthy', { instanceId, instance });
                    
                    // Auto-deregister if configured
                    if (this.config.autoDeregisterCritical && health.consecutiveFailures > this.config.unhealthyThreshold * 2) {
                        await this.deregisterService(instanceId);
                        this.emit('instance:deregistered', { instanceId, reason: 'critical_health_failure' });
                    }
                }
            }
            
            health.lastCheck = Date.now();
            
        } catch (error) {
            health.consecutiveFailures++;
            health.consecutiveSuccesses = 0;
            health.lastError = error.message;
            health.lastCheck = Date.now();
            
            if (health.status !== 'unhealthy' && health.consecutiveFailures >= this.config.unhealthyThreshold) {
                health.status = 'unhealthy';
                this.emit('instance:unhealthy', { instanceId, instance, error });
            }
        }
    }
    
    async performHealthCheck(instance) {
        return new Promise((resolve, reject) => {
            const { protocol, address, port, health } = instance;
            const url = health.checkUrl || `${protocol}://${address}:${port}/health`;
            
            const requestModule = protocol === 'https' ? https : http;
            const timeout = health.timeout || this.config.healthCheckTimeout;
            
            const request = requestModule.get(url, { timeout }, (res) => {
                let data = '';
                
                res.on('data', (chunk) => {
                    data += chunk;
                });
                
                res.on('end', () => {
                    const isHealthy = res.statusCode >= 200 && res.statusCode < 300;
                    resolve(isHealthy);
                });
            });
            
            request.on('error', (error) => {
                reject(error);
            });
            
            request.on('timeout', () => {
                request.destroy();
                reject(new Error('Health check timeout'));
            });
            
            request.setTimeout(timeout);
        });
    }
    
    // Service discovery
    startServiceDiscovery() {
        this.discoveryInterval = setInterval(async () => {
            await this.performServiceDiscovery();
        }, this.config.discoveryInterval);
    }
    
    async performServiceDiscovery() {
        // Auto-discovery based on DNS, environment variables, or other sources
        try {
            // DNS-based service discovery
            await this.discoverFromDNS();
            
            // Environment-based service discovery
            await this.discoverFromEnvironment();
            
        } catch (error) {
            console.error('Service discovery error:', error);
        }
    }
    
    async discoverFromDNS() {
        // Example: Look for SRV records for A2A services
        const servicePatterns = [
            '_a2a-agents._tcp',
            '_a2a-network._tcp',
            '_a2a-registry._tcp'
        ];
        
        for (const pattern of servicePatterns) {
            try {
                const records = await dns.resolveSrv(pattern);
                
                for (const record of records) {
                    const serviceName = pattern.replace('_a2a-', '').replace('._tcp', '');
                    const instanceId = `${serviceName}-${record.name}-${record.port}`;
                    
                    // Check if already registered
                    if (!this.findServiceInstance(instanceId)) {
                        await this.registerService({
                            id: instanceId,
                            name: serviceName,
                            address: record.name,
                            port: record.port,
                            tags: ['auto-discovered', 'dns'],
                            metadata: { discoveredAt: Date.now() }
                        });
                    }
                }
                
            } catch (error) {
                // DNS lookup failed - this is normal if no SRV records exist
                continue;
            }
        }
    }
    
    async discoverFromEnvironment() {
        // Discover services from environment variables
        const envServices = process.env.A2A_SERVICES;
        if (!envServices) return;
        
        try {
            const services = JSON.parse(envServices);
            
            for (const service of services) {
                const instanceId = `${service.name}-${service.address}-${service.port}`;
                
                if (!this.findServiceInstance(instanceId)) {
                    await this.registerService({
                        ...service,
                        id: instanceId,
                        tags: [...(service.tags || []), 'auto-discovered', 'environment'],
                        metadata: { discoveredAt: Date.now(), ...service.metadata }
                    });
                }
            }
        } catch (error) {
            console.error('Environment service discovery error:', error);
        }
    }
    
    findServiceInstance(instanceId) {
        for (const [, instances] of this.services) {
            if (instances.has(instanceId)) {
                return instances.get(instanceId);
            }
        }
        return null;
    }
    
    // Metrics and monitoring
    recordRequest(instanceId, responseTime, error = null) {
        const metrics = this.metrics.get(instanceId);
        if (!metrics) return;
        
        metrics.requests++;
        metrics.lastRequest = Date.now();
        
        if (error) {
            metrics.errors++;
        } else {
            metrics.responses++;
            metrics.totalResponseTime += responseTime;
            metrics.avgResponseTime = metrics.totalResponseTime / metrics.responses;
        }
    }
    
    getServiceMetrics(serviceName) {
        const instances = this.services.get(serviceName);
        if (!instances) return null;
        
        const serviceMetrics = {
            name: serviceName,
            instanceCount: instances.size,
            healthyInstances: 0,
            unhealthyInstances: 0,
            totalRequests: 0,
            totalErrors: 0,
            avgResponseTime: 0,
            instances: {}
        };
        
        let totalResponseTimeSum = 0;
        let responseCount = 0;
        
        for (const [instanceId, instance] of instances) {
            const health = this.healthChecks.get(instanceId);
            const metrics = this.metrics.get(instanceId);
            
            if (health?.status === 'healthy') {
                serviceMetrics.healthyInstances++;
            } else {
                serviceMetrics.unhealthyInstances++;
            }
            
            if (metrics) {
                serviceMetrics.totalRequests += metrics.requests;
                serviceMetrics.totalErrors += metrics.errors;
                
                if (metrics.responses > 0) {
                    totalResponseTimeSum += metrics.totalResponseTime;
                    responseCount += metrics.responses;
                }
                
                serviceMetrics.instances[instanceId] = {
                    ...metrics,
                    health: health?.status || 'unknown'
                };
            }
        }
        
        if (responseCount > 0) {
            serviceMetrics.avgResponseTime = totalResponseTimeSum / responseCount;
        }
        
        return serviceMetrics;
    }
    
    getSystemMetrics() {
        const systemMetrics = {
            totalServices: this.services.size,
            totalInstances: 0,
            healthyInstances: 0,
            unhealthyInstances: 0,
            services: {}
        };
        
        for (const [serviceName] of this.services) {
            const serviceMetrics = this.getServiceMetrics(serviceName);
            if (serviceMetrics) {
                systemMetrics.totalInstances += serviceMetrics.instanceCount;
                systemMetrics.healthyInstances += serviceMetrics.healthyInstances;
                systemMetrics.unhealthyInstances += serviceMetrics.unhealthyInstances;
                systemMetrics.services[serviceName] = serviceMetrics;
            }
        }
        
        return systemMetrics;
    }
}

// Load Balancer Implementation
class LoadBalancer {
    constructor(serviceName, strategy = 'round-robin') {
        this.serviceName = serviceName;
        this.strategy = strategy;
        this.roundRobinIndex = 0;
        this.stickySessionMap = new Map();
    }
    
    selectInstance(instances, strategy = null) {
        if (instances.length === 0) {
            throw new Error('No instances available');
        }
        
        if (instances.length === 1) {
            return instances[0];
        }
        
        const selectedStrategy = strategy || this.strategy;
        
        switch (selectedStrategy) {
            case 'round-robin':
                return this.roundRobinSelect(instances);
            case 'weighted-round-robin':
                return this.weightedRoundRobinSelect(instances);
            case 'least-connections':
                return this.leastConnectionsSelect(instances);
            case 'least-response-time':
                return this.leastResponseTimeSelect(instances);
            case 'random':
                return this.randomSelect(instances);
            case 'weighted-random':
                return this.weightedRandomSelect(instances);
            case 'consistent-hash':
                return this.consistentHashSelect(instances);
            default:
                return this.roundRobinSelect(instances);
        }
    }
    
    roundRobinSelect(instances) {
        const instance = instances[this.roundRobinIndex % instances.length];
        this.roundRobinIndex++;
        return instance;
    }
    
    weightedRoundRobinSelect(instances) {
        // Weight based on inverse of current load or response time
        const weights = instances.map(instance => {
            const metrics = instance.metrics || {};
            const weight = metrics.avgResponseTime ? 1 / metrics.avgResponseTime : 1;
            return Math.max(weight, 0.1); // Minimum weight
        });
        
        const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
        const random = Math.random() * totalWeight;
        
        let currentWeight = 0;
        for (let i = 0; i < instances.length; i++) {
            currentWeight += weights[i];
            if (random <= currentWeight) {
                return instances[i];
            }
        }
        
        return instances[instances.length - 1];
    }
    
    leastConnectionsSelect(instances) {
        return instances.reduce((best, current) => {
            const bestMetrics = best.metrics || {};
            const currentMetrics = current.metrics || {};
            
            const bestConnections = bestMetrics.requests - (bestMetrics.responses + bestMetrics.errors);
            const currentConnections = currentMetrics.requests - (currentMetrics.responses + currentMetrics.errors);
            
            return currentConnections < bestConnections ? current : best;
        });
    }
    
    leastResponseTimeSelect(instances) {
        return instances.reduce((best, current) => {
            const bestMetrics = best.metrics || {};
            const currentMetrics = current.metrics || {};
            
            const bestTime = bestMetrics.avgResponseTime || Infinity;
            const currentTime = currentMetrics.avgResponseTime || Infinity;
            
            return currentTime < bestTime ? current : best;
        });
    }
    
    randomSelect(instances) {
        return instances[Math.floor(Math.random() * instances.length)];
    }
    
    weightedRandomSelect(instances) {
        return this.weightedRoundRobinSelect(instances); // Same logic
    }
    
    consistentHashSelect(instances, key = '') {
        // Simple consistent hashing based on key
        const hash = this.simpleHash(key);
        return instances[hash % instances.length];
    }
    
    simpleHash(str) {
        let hash = 0;
        if (str.length === 0) return hash;
        
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        
        return Math.abs(hash);
    }
}

// Factory function
function createServiceRegistry(options = {}) {
    return new A2AServiceRegistry(options);
}

// HTTP Client with load balancing
class A2AServiceClient {
    constructor(serviceRegistry, options = {}) {
        this.registry = serviceRegistry;
        this.options = {
            timeout: options.timeout || 10000,
            retries: options.retries || 3,
            retryDelay: options.retryDelay || 1000,
            circuitBreaker: options.circuitBreaker !== false,
            ...options
        };
        
        this.circuitBreakers = new Map();
    }
    
    async request(serviceName, path, options = {}) {
        let lastError;
        
        for (let attempt = 0; attempt < this.options.retries; attempt++) {
            try {
                const instance = await this.registry.getServiceInstance(serviceName, options.loadBalancingStrategy);
                const url = `${instance.protocol}://${instance.address}:${instance.port}${path}`;
                
                const startTime = performance.now();
                
                const response = await this.makeRequest(url, options);
                
                const responseTime = performance.now() - startTime;
                this.registry.recordRequest(instance.id, responseTime);
                
                return response;
                
            } catch (error) {
                lastError = error;
                
                if (attempt < this.options.retries - 1) {
                    await new Promise(resolve => setTimeout(resolve, this.options.retryDelay * (attempt + 1)));
                }
            }
        }
        
        throw lastError;
    }
    
    async makeRequest(url, options) {
        // Implementation would use actual HTTP client like axios or node-fetch
        // This is a simplified placeholder
        return new Promise((resolve, reject) => {
            const requestModule = url.startsWith('https') ? https : http;
            
            const req = requestModule.request(url, options, (res) => {
                let data = '';
                
                res.on('data', (chunk) => {
                    data += chunk;
                });
                
                res.on('end', () => {
                    if (res.statusCode >= 200 && res.statusCode < 300) {
                        resolve({
                            status: res.statusCode,
                            data: data,
                            headers: res.headers
                        });
                    } else {
                        reject(new Error(`HTTP ${res.statusCode}: ${data}`));
                    }
                });
            });
            
            req.on('error', reject);
            req.setTimeout(this.options.timeout, () => {
                req.destroy();
                reject(new Error('Request timeout'));
            });
            
            if (options.body) {
                req.write(options.body);
            }
            
            req.end();
        });
    }
}

module.exports = {
    A2AServiceRegistry,
    LoadBalancer,
    A2AServiceClient,
    createServiceRegistry
};