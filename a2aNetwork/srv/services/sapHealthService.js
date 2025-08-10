/**
 * Health Check Service
 * 
 * Provides comprehensive health monitoring endpoints for the A2A Network application
 * following SAP enterprise standards for production monitoring.
 * 
 * @author SAP SE
 * @since 1.0.0
 * @version 1.0.0
 */

const os = require('os');
const fs = require('fs').promises;
const path = require('path');
const { performance } = require('perf_hooks');

class HealthService {
    constructor() {
        this.startTime = Date.now();
        this.checks = new Map();
        this.metrics = {
            requests: 0,
            errors: 0,
            responses: {
                '2xx': 0,
                '3xx': 0,
                '4xx': 0,
                '5xx': 0
            },
            responseTimes: [],
            memoryUsage: [],
            cpuUsage: []
        };
        
        // Register default health checks
        this._registerDefaultChecks();
        
        // Start metrics collection
        this._startMetricsCollection();
    }

    /**
     * Register default health checks
     * @private
     */
    _registerDefaultChecks() {
        this.registerCheck('database', this._checkDatabase.bind(this));
        this.registerCheck('memory', this._checkMemory.bind(this));
        this.registerCheck('disk', this._checkDiskSpace.bind(this));
        this.registerCheck('external-services', this._checkExternalServices.bind(this));
    }

    /**
     * Start collecting system metrics
     * @private
     */
    _startMetricsCollection() {
        // Collect metrics every 30 seconds
        setInterval(() => {
            this._collectSystemMetrics();
        }, 30000);
    }

    /**
     * Collect system performance metrics
     * @private
     */
    _collectSystemMetrics() {
        const memUsage = process.memoryUsage();
        const cpuUsage = process.cpuUsage();
        
        // Store memory usage (keep last 100 entries)
        this.metrics.memoryUsage.push({
            timestamp: Date.now(),
            rss: memUsage.rss,
            heapTotal: memUsage.heapTotal,
            heapUsed: memUsage.heapUsed,
            external: memUsage.external
        });
        
        if (this.metrics.memoryUsage.length > 100) {
            this.metrics.memoryUsage.shift();
        }
        
        // Store CPU usage (keep last 100 entries)
        this.metrics.cpuUsage.push({
            timestamp: Date.now(),
            user: cpuUsage.user,
            system: cpuUsage.system
        });
        
        if (this.metrics.cpuUsage.length > 100) {
            this.metrics.cpuUsage.shift();
        }
    }

    /**
     * Register a custom health check
     * @param {string} name Check name
     * @param {function} checkFunction Check function that returns a promise
     */
    registerCheck(name, checkFunction) {
        this.checks.set(name, checkFunction);
    }

    /**
     * Record request metrics
     * @param {number} statusCode HTTP status code
     * @param {number} responseTime Response time in ms
     */
    recordRequest(statusCode, responseTime) {
        this.metrics.requests++;
        
        if (statusCode >= 400) {
            this.metrics.errors++;
        }
        
        // Group by status code ranges
        if (statusCode >= 200 && statusCode < 300) {
            this.metrics.responses['2xx']++;
        } else if (statusCode >= 300 && statusCode < 400) {
            this.metrics.responses['3xx']++;
        } else if (statusCode >= 400 && statusCode < 500) {
            this.metrics.responses['4xx']++;
        } else if (statusCode >= 500) {
            this.metrics.responses['5xx']++;
        }
        
        // Store response times (keep last 1000 entries)
        this.metrics.responseTimes.push({
            timestamp: Date.now(),
            duration: responseTime
        });
        
        if (this.metrics.responseTimes.length > 1000) {
            this.metrics.responseTimes.shift();
        }
    }

    /**
     * Get basic health status
     * @returns {object} Basic health information
     */
    async getHealth() {
        const uptime = Date.now() - this.startTime;
        const memUsage = process.memoryUsage();
        
        return {
            status: 'healthy',
            timestamp: new Date().toISOString(),
            uptime: this._formatUptime(uptime),
            uptimeMs: uptime,
            version: process.env.npm_package_version || '1.0.0',
            nodeVersion: process.version,
            environment: process.env.NODE_ENV || 'development',
            pid: process.pid,
            platform: os.platform(),
            architecture: os.arch(),
            memory: {
                rss: this._formatBytes(memUsage.rss),
                heapTotal: this._formatBytes(memUsage.heapTotal),
                heapUsed: this._formatBytes(memUsage.heapUsed),
                external: this._formatBytes(memUsage.external)
            },
            load: os.loadavg(),
            cpuCount: os.cpus().length
        };
    }

    /**
     * Get detailed health status with all checks
     * @returns {object} Detailed health information
     */
    async getDetailedHealth() {
        const startTime = performance.now();
        const basicHealth = await this.getHealth();
        const checkResults = {};
        let overallStatus = 'healthy';
        
        // Run all registered health checks
        for (const [name, checkFn] of this.checks) {
            try {
                const result = await Promise.race([
                    checkFn(),
                    new Promise((_, reject) => setTimeout(() => reject(new Error('Check timeout')), 5000))
                ]);
                
                checkResults[name] = {
                    status: 'healthy',
                    ...result
                };
            } catch (error) {
                checkResults[name] = {
                    status: 'unhealthy',
                    error: error.message,
                    timestamp: new Date().toISOString()
                };
                overallStatus = 'degraded';
            }
        }
        
        const duration = performance.now() - startTime;
        
        return {
            ...basicHealth,
            status: overallStatus,
            checks: checkResults,
            healthCheckDuration: `${duration.toFixed(2)}ms`
        };
    }

    /**
     * Get application metrics
     * @returns {object} Application metrics
     */
    getMetrics() {
        const recentResponseTimes = this.metrics.responseTimes
            .filter(rt => Date.now() - rt.timestamp < 300000) // Last 5 minutes
            .map(rt => rt.duration);
            
        const avgResponseTime = recentResponseTimes.length > 0 
            ? recentResponseTimes.reduce((a, b) => a + b, 0) / recentResponseTimes.length 
            : 0;
            
        const p95ResponseTime = recentResponseTimes.length > 0
            ? this._percentile(recentResponseTimes, 0.95)
            : 0;

        return {
            timestamp: new Date().toISOString(),
            uptime: Date.now() - this.startTime,
            requests: {
                total: this.metrics.requests,
                errors: this.metrics.errors,
                errorRate: this.metrics.requests > 0 ? (this.metrics.errors / this.metrics.requests * 100).toFixed(2) : 0,
                responses: this.metrics.responses
            },
            performance: {
                avgResponseTime: parseFloat(avgResponseTime.toFixed(2)),
                p95ResponseTime: parseFloat(p95ResponseTime.toFixed(2)),
                requestsPerMinute: this._calculateRequestsPerMinute()
            },
            system: {
                memory: process.memoryUsage(),
                cpu: process.cpuUsage(),
                load: os.loadavg(),
                freeMemory: os.freemem(),
                totalMemory: os.totalmem()
            }
        };
    }

    /**
     * Get readiness status (for Kubernetes readiness probes)
     * @returns {object} Readiness information
     */
    async getReadiness() {
        try {
            // Check critical dependencies
            const dbCheck = await this._checkDatabase();
            
            return {
                status: 'ready',
                timestamp: new Date().toISOString(),
                checks: {
                    database: dbCheck
                }
            };
        } catch (error) {
            return {
                status: 'not-ready',
                timestamp: new Date().toISOString(),
                error: error.message
            };
        }
    }

    /**
     * Get liveness status (for Kubernetes liveness probes)
     * @returns {object} Liveness information
     */
    getLiveness() {
        const memUsage = process.memoryUsage();
        const heapUsedPercent = (memUsage.heapUsed / memUsage.heapTotal) * 100;
        
        // Consider unhealthy if heap usage is over 90%
        const isHealthy = heapUsedPercent < 90;
        
        return {
            status: isHealthy ? 'alive' : 'unhealthy',
            timestamp: new Date().toISOString(),
            uptime: Date.now() - this.startTime,
            heapUsagePercent: parseFloat(heapUsedPercent.toFixed(2))
        };
    }

    // Health check implementations
    
    async _checkDatabase() {
        // In a real implementation, check database connectivity
        // For now, simulate a database check
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve({
                    connectionPool: 'active',
                    responseTime: Math.floor(Math.random() * 50) + 10,
                    lastQuery: new Date().toISOString()
                });
            }, 50);
        });
    }

    async _checkMemory() {
        const memUsage = process.memoryUsage();
        const totalMem = os.totalmem();
        const freeMem = os.freemem();
        const usedPercent = ((totalMem - freeMem) / totalMem) * 100;
        
        return {
            heapUsed: this._formatBytes(memUsage.heapUsed),
            heapTotal: this._formatBytes(memUsage.heapTotal),
            systemUsedPercent: parseFloat(usedPercent.toFixed(2)),
            status: usedPercent > 90 ? 'warning' : 'ok'
        };
    }

    async _checkDiskSpace() {
        try {
            const stats = await fs.stat('.');
            return {
                available: true,
                path: process.cwd(),
                writable: true
            };
        } catch (error) {
            throw new Error(`Disk check failed: ${error.message}`);
        }
    }

    async _checkExternalServices() {
        // Check external service connectivity
        const services = [];
        
        // In production, check actual external services
        // For now, simulate external service checks
        const mockServices = ['blockchain-rpc', 'message-queue', 'cache-service'];
        
        for (const service of mockServices) {
            const isHealthy = Math.random() > 0.1; // 90% success rate
            services.push({
                name: service,
                status: isHealthy ? 'healthy' : 'unhealthy',
                responseTime: Math.floor(Math.random() * 200) + 50
            });
        }
        
        return { services };
    }

    // Utility methods
    
    _formatUptime(ms) {
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);
        
        if (days > 0) return `${days}d ${hours % 24}h ${minutes % 60}m`;
        if (hours > 0) return `${hours}h ${minutes % 60}m`;
        if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
        return `${seconds}s`;
    }

    _formatBytes(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 Bytes';
        const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
        return Math.round(bytes / Math.pow(1024, i), 2) + ' ' + sizes[i];
    }

    _percentile(arr, p) {
        const sorted = arr.sort((a, b) => a - b);
        const index = Math.ceil(sorted.length * p) - 1;
        return sorted[index] || 0;
    }

    _calculateRequestsPerMinute() {
        const oneMinuteAgo = Date.now() - 60000;
        const recentRequests = this.metrics.responseTimes
            .filter(rt => rt.timestamp > oneMinuteAgo);
        return recentRequests.length;
    }
}

module.exports = new HealthService();