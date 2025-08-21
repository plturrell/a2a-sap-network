/**
 * @fileoverview System Monitoring Service - CAP Implementation
 * @since 1.0.0
 * @module systemMonitoringService
 * 
 * CAP service handlers for system health, monitoring, and error reporting
 * Replaces Express routes with proper SAP CAP architecture
 */

const cds = require('@sap/cds');
const os = require('os');
const process = require('process');
const LOG = cds.log('system-monitoring');

// Import existing services
const healthService = require('./services/sapHealthService');
const loggingService = require('./services/sapLoggingService');
const errorReporting = require('./services/sapErrorReportingService');
const cacheMiddleware = require('./middleware/sapCacheMiddleware');

/**
 * CAP Service Handler for System Monitoring Actions
 */
module.exports = function() {
    
    // Basic health check
    this.on('getHealth', async (req) => {
        try {
            return {
                status: 'healthy',
                timestamp: new Date().toISOString(),
                uptime: Math.floor(process.uptime()),
                version: process.env.npm_package_version || '1.0.0'
            };
        } catch (error) {
            LOG.error('Health check failed:', error);
            req.error(500, 'HEALTH_CHECK_ERROR', `Health check failed: ${error.message}`);
        }
    });
    
    // Detailed health check
    this.on('getDetailedHealth', async (req) => {
        try {
            const healthData = await healthService.getSystemHealth();
            return {
                status: healthData.overall_status,
                timestamp: new Date().toISOString(),
                components: {
                    database: {
                        status: healthData.database_status,
                        responseTime: healthData.database_response_time || 0
                    },
                    agents: {
                        status: healthData.agents_status,
                        healthy: healthData.healthy_agents || 0,
                        total: healthData.total_agents || 0
                    },
                    blockchain: {
                        status: healthData.blockchain_status,
                        connected: healthData.blockchain_connected || false
                    },
                    memory: {
                        used: Math.round(process.memoryUsage().heapUsed),
                        total: Math.round(process.memoryUsage().heapTotal),
                        percentage: Math.round((process.memoryUsage().heapUsed / process.memoryUsage().heapTotal) * 100)
                    },
                    cpu: {
                        usage: Math.round(os.loadavg()[0] * 100) / 100,
                        load: os.loadavg()
                    }
                }
            };
        } catch (error) {
            LOG.error('Detailed health check failed:', error);
            req.error(500, 'DETAILED_HEALTH_ERROR', `Detailed health check failed: ${error.message}`);
        }
    });
    
    // Readiness probe
    this.on('getReadiness', async (req) => {
        try {
            const services = [
                { name: 'database', ready: true }, // Would check actual DB connection
                { name: 'agents', ready: true },   // Would check agent health
                { name: 'blockchain', ready: true } // Would check blockchain connection
            ];
            
            const allReady = services.every(s => s.ready);
            
            return {
                ready: allReady,
                services: services
            };
        } catch (error) {
            LOG.error('Readiness check failed:', error);
            req.error(503, 'READINESS_ERROR', `Readiness check failed: ${error.message}`);
        }
    });
    
    // Liveness probe
    this.on('getLiveness', async (req) => {
        try {
            return {
                alive: true,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            LOG.error('Liveness check failed:', error);
            req.error(500, 'LIVENESS_ERROR', `Liveness check failed: ${error.message}`);
        }
    });
    
    // System metrics
    this.on('getMetrics', async (req) => {
        try {
            const memUsage = process.memoryUsage();
            const cpuUsage = process.cpuUsage();
            
            const metrics = [
                `# HELP nodejs_memory_heap_used_bytes Process heap memory used`,
                `# TYPE nodejs_memory_heap_used_bytes gauge`,
                `nodejs_memory_heap_used_bytes ${memUsage.heapUsed}`,
                ``,
                `# HELP nodejs_memory_heap_total_bytes Process heap memory total`,
                `# TYPE nodejs_memory_heap_total_bytes gauge`,
                `nodejs_memory_heap_total_bytes ${memUsage.heapTotal}`,
                ``,
                `# HELP process_uptime_seconds Process uptime in seconds`,
                `# TYPE process_uptime_seconds counter`,
                `process_uptime_seconds ${Math.floor(process.uptime())}`,
                ``
            ].join('\n');
            
            return metrics;
        } catch (error) {
            LOG.error('Metrics collection failed:', error);
            req.error(500, 'METRICS_ERROR', `Metrics collection failed: ${error.message}`);
        }
    });
    
    // Report error
    this.on('reportError', async (req) => {
        try {
            const { error } = req.data;
            
            if (!error || !error.message) {
                req.error(400, 'INVALID_ERROR', 'Error message is required');
                return;
            }
            
            const errorId = await errorReporting.reportError(error);
            
            return {
                success: true,
                errorId: errorId
            };
        } catch (error) {
            LOG.error('Error reporting failed:', error);
            req.error(500, 'ERROR_REPORT_FAILED', `Error reporting failed: ${error.message}`);
        }
    });
    
    // Get error statistics
    this.on('getErrorStats', async (req) => {
        try {
            const stats = errorReporting.getErrorStats();
            return {
                total: stats.total || 0,
                byLevel: {
                    error: stats.error || 0,
                    warn: stats.warn || 0,
                    info: stats.info || 0
                },
                recent: stats.recent || []
            };
        } catch (error) {
            LOG.error('Error stats retrieval failed:', error);
            req.error(500, 'ERROR_STATS_ERROR', `Error stats retrieval failed: ${error.message}`);
        }
    });
    
    // Get recent logs
    this.on('getLogs', async (req) => {
        try {
            const { limit = 100, level = 'info' } = req.data;
            const logs = loggingService.getRecentLogs(limit, level);
            return logs;
        } catch (error) {
            LOG.error('Log retrieval failed:', error);
            req.error(500, 'LOG_RETRIEVAL_ERROR', `Log retrieval failed: ${error.message}`);
        }
    });
    
    // Get cache statistics
    this.on('getCacheStats', async (req) => {
        try {
            const stats = await cacheMiddleware.getStats();
            return {
                hits: stats.hits || 0,
                misses: stats.misses || 0,
                hitRate: stats.hitRate || 0,
                size: stats.size || 0,
                memory: stats.memory || 0
            };
        } catch (error) {
            LOG.error('Cache stats retrieval failed:', error);
            req.error(500, 'CACHE_STATS_ERROR', `Cache stats retrieval failed: ${error.message}`);
        }
    });
    
    // Invalidate cache
    this.on('invalidateCache', async (req) => {
        try {
            const { pattern } = req.data;
            const result = await cacheMiddleware.invalidate(pattern || '*');
            
            return {
                success: true,
                cleared: result.cleared || 0
            };
        } catch (error) {
            LOG.error('Cache invalidation failed:', error);
            req.error(500, 'CACHE_INVALIDATION_ERROR', `Cache invalidation failed: ${error.message}`);
        }
    });
    
    LOG.info('System Monitoring service handlers registered');
};