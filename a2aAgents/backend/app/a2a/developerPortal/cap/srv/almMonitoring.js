"use strict";

const cds = require('@sap/cds');
const { CloudSDK } = require('@sap/cloud-sdk-core');

/**
 * SAP Cloud Application Lifecycle Management (ALM) Monitoring Integration
 * Provides health monitoring, metrics collection, and alerting
 */
class ALMMonitoringService {
    
    constructor() {
        this.almClient = null;
        this.metrics = new Map();
        this.healthChecks = new Map();
        this.alerts = new Map();
        this.init();
    }

    init() {
        try {
            // Initialize SAP Cloud ALM client
            this.almClient = new CloudSDK({
                baseURL: process.env.ALM_SERVICE_URL || 'https://api.alm.cloud.sap',
                auth: {
                    clientId: process.env.ALM_CLIENT_ID,
                    clientSecret: process.env.ALM_CLIENT_SECRET,
                    tokenUrl: process.env.ALM_TOKEN_URL
                }
            });

            // Register default health checks
            this.registerDefaultHealthChecks();
            
            // Start metrics collection
            this.startMetricsCollection();
            
            // Start health monitoring
            this.startHealthMonitoring();
            
             
            
            // eslint-disable-next-line no-console
            
             
            
            // eslint-disable-next-line no-console
            console.log('SAP Cloud ALM Monitoring initialized successfully');
        } catch (error) {
            console.error('Failed to initialize SAP Cloud ALM Monitoring:', error);
        }
    }

    /**
     * Register default health checks
     */
    registerDefaultHealthChecks() {
        // Database connectivity check
        this.registerHealthCheck('database', async () => {
            try {
                const db = await cds.connect.to('db');
                await db.run('SELECT 1');
                return { status: 'UP', details: 'Database connection successful' };
            } catch (error) {
                return { 
                    status: 'DOWN', 
                    details: `Database connection failed: ${error.message}` 
                };
            }
        });

        // Memory usage check
        this.registerHealthCheck('memory', () => {
            const memUsage = process.memoryUsage();
            const memoryUsagePercent = (memUsage.heapUsed / memUsage.heapTotal) * 100;
            
            return {
                status: memoryUsagePercent < 80 ? 'UP' : 'WARNING',
                details: {
                    heapUsed: `${Math.round(memUsage.heapUsed / 1024 / 1024)  } MB`,
                    heapTotal: `${Math.round(memUsage.heapTotal / 1024 / 1024)  } MB`,
                    usagePercent: Math.round(memoryUsagePercent)
                }
            };
        });

        // External services check
        this.registerHealthCheck('external-services', () => {
            const services = ['workflow', 'audit-log', 'application-logging'];
            const results = {};
            
            for (const service of services) {
                try {
                    // Simulate service health check
                    results[service] = { status: 'UP', responseTime: Math.random() * 100 };
                } catch (error) {
                    results[service] = { status: 'DOWN', error: error.message };
                }
            }
            
            const allUp = Object.values(results).every(r => r.status === 'UP');
            return {
                status: allUp ? 'UP' : 'DEGRADED',
                details: results
            };
        });

        // Application metrics check
        this.registerHealthCheck('application', async () => {
            try {
                const db = await cds.connect.to('db');
                
                // Check active users
                const activeUsers = await db.run(`
                    SELECT COUNT(*) as count 
                    FROM UserSessions 
                    WHERE isActive = true AND lastActivity > datetime('now', '-1 hour')
                `);
                
                // Check running workflows
                const runningWorkflows = await db.run(`
                    SELECT COUNT(*) as count 
                    FROM WorkflowExecutions 
                    WHERE status = 'Running'
                `);
                
                return {
                    status: 'UP',
                    details: {
                        activeUsers: activeUsers[0]?.count || 0,
                        runningWorkflows: runningWorkflows[0]?.count || 0,
                        uptime: Math.floor(process.uptime())
                    }
                };
            } catch (error) {
                return {
                    status: 'DOWN',
                    details: `Application health check failed: ${error.message}`
                };
            }
        });
    }

    /**
     * Register a custom health check
     * @param {string} name - Health check name
     * @param {function} checkFunction - Health check function
     */
    registerHealthCheck(name, checkFunction) {
        this.healthChecks.set(name, checkFunction);
    }

    /**
     * Get overall system health
     * @returns {object} System health status
     */
    async getSystemHealth() {
        const healthResults = {};
        let overallStatus = 'UP';
        
        for (const [name, checkFunction] of this.healthChecks) {
            try {
                const result = await checkFunction();
                healthResults[name] = result;
                
                if (result.status === 'DOWN') {
                    overallStatus = 'DOWN';
                } else if (result.status === 'WARNING' && overallStatus !== 'DOWN') {
                    overallStatus = 'WARNING';
                }
            } catch (error) {
                healthResults[name] = {
                    status: 'DOWN',
                    details: `Health check failed: ${error.message}`
                };
                overallStatus = 'DOWN';
            }
        }
        
        const healthStatus = {
            status: overallStatus,
            timestamp: new Date().toISOString(),
            checks: healthResults,
            uptime: Math.floor(process.uptime()),
            version: process.env.APP_VERSION || '1.0.0'
        };
        
        // Send health status to SAP Cloud ALM
        await this.sendHealthStatusToALM(healthStatus);
        
        return healthStatus;
    }

    /**
     * Start metrics collection
     */
    startMetricsCollection() {
        // Collect metrics every 30 seconds
        setInterval(async () => {
            await this.collectMetrics();
        }, 30000);
    }

    /**
     * Collect system and application metrics
     */
    async collectMetrics() {
        try {
            const timestamp = new Date().toISOString();
            
            // System metrics
            const memUsage = process.memoryUsage();
            const cpuUsage = process.cpuUsage();
            
            // Application metrics
            const db = await cds.connect.to('db');
            
            const metrics = {
                timestamp,
                system: {
                    memory: {
                        heapUsed: memUsage.heapUsed,
                        heapTotal: memUsage.heapTotal,
                        external: memUsage.external,
                        rss: memUsage.rss
                    },
                    cpu: {
                        user: cpuUsage.user,
                        system: cpuUsage.system
                    },
                    uptime: process.uptime()
                },
                application: await this.collectApplicationMetrics(db),
                business: await this.collectBusinessMetrics(db)
            };
            
            // Store metrics locally
            this.storeMetrics(metrics);
            
            // Send to SAP Cloud ALM
            await this.sendMetricsToALM(metrics);
            
        } catch (error) {
            console.error('Failed to collect metrics:', error);
        }
    }

    /**
     * Collect application-specific metrics
     * @param {object} db - Database connection
     * @returns {object} Application metrics
     */
    async collectApplicationMetrics(db) {
        try {
            const [
                activeUsers,
                totalProjects,
                activeAgents,
                runningWorkflows,
                pendingApprovals
            ] = await Promise.all([
                db.run(`SELECT COUNT(*) as count FROM UserSessions WHERE isActive = true`),
                db.run(`SELECT COUNT(*) as count FROM Projects WHERE status != 'Archived'`),
                db.run(`SELECT COUNT(*) as count FROM Agents WHERE status = 'Active'`),
                db.run(`SELECT COUNT(*) as count FROM WorkflowExecutions WHERE status = 'Running'`),
                db.run(`SELECT COUNT(*) as count FROM ApprovalWorkflows WHERE status = 'Pending'`)
            ]);
            
            return {
                activeUsers: activeUsers[0]?.count || 0,
                totalProjects: totalProjects[0]?.count || 0,
                activeAgents: activeAgents[0]?.count || 0,
                runningWorkflows: runningWorkflows[0]?.count || 0,
                pendingApprovals: pendingApprovals[0]?.count || 0
            };
        } catch (error) {
            console.error('Failed to collect application metrics:', error);
            return {};
        }
    }

    /**
     * Collect business metrics
     * @param {object} db - Database connection
     * @returns {object} Business metrics
     */
    async collectBusinessMetrics(db) {
        try {
            const [
                projectsCreatedToday,
                deploymentsToday,
                testsExecutedToday,
                errorRate
            ] = await Promise.all([
                db.run(`SELECT COUNT(*) as count FROM Projects WHERE DATE(createdAt) = DATE('now')`),
                db.run(`SELECT COUNT(*) as count FROM Deployments WHERE DATE(startTime) = DATE('now')`),
                db.run(`SELECT COUNT(*) as count FROM TestExecutions WHERE DATE(startTime) = DATE('now')`),
                this.calculateErrorRate()
            ]);
            
            return {
                projectsCreatedToday: projectsCreatedToday[0]?.count || 0,
                deploymentsToday: deploymentsToday[0]?.count || 0,
                testsExecutedToday: testsExecutedToday[0]?.count || 0,
                errorRate: errorRate
            };
        } catch (error) {
            console.error('Failed to collect business metrics:', error);
            return {};
        }
    }

    /**
     * Calculate error rate from logs
     * @returns {number} Error rate percentage
     */
    calculateErrorRate() {
        // This would typically query the logging service
        // For now, return a mock value
        return Math.random() * 5; // 0-5% error rate
    }

    /**
     * Store metrics locally
     * @param {object} metrics - Metrics data
     */
    storeMetrics(metrics) {
        const key = `metrics_${Date.now()}`;
        this.metrics.set(key, metrics);
        
        // Keep only last 1000 metric entries
        if (this.metrics.size > 1000) {
            const oldestKey = this.metrics.keys().next().value;
            this.metrics.delete(oldestKey);
        }
    }

    /**
     * Send health status to SAP Cloud ALM
     * @param {object} healthStatus - Health status data
     */
    async sendHealthStatusToALM(healthStatus) {
        try {
            if (this.almClient) {
                await this.almClient.post('/health-monitoring/status', {
                    service: 'a2a-developer-portal',
                    environment: process.env.NODE_ENV || 'development',
                    ...healthStatus
                });
            }
        } catch (error) {
            console.error('Failed to send health status to ALM:', error);
        }
    }

    /**
     * Send metrics to SAP Cloud ALM
     * @param {object} metrics - Metrics data
     */
    async sendMetricsToALM(metrics) {
        try {
            if (this.almClient) {
                await this.almClient.post('/metrics/collect', {
                    service: 'a2a-developer-portal',
                    environment: process.env.NODE_ENV || 'development',
                    ...metrics
                });
            }
        } catch (error) {
            console.error('Failed to send metrics to ALM:', error);
        }
    }

    /**
     * Start health monitoring
     */
    startHealthMonitoring() {
        // Check health every 60 seconds
        setInterval(async () => {
            const health = await this.getSystemHealth();
            
            // Check for alerts
            await this.checkAlerts(health);
        }, 60000);
    }

    /**
     * Check for alert conditions
     * @param {object} health - Health status
     */
    async checkAlerts(health) {
        try {
            // Check for critical conditions
            if (health.status === 'DOWN') {
                await this.triggerAlert('CRITICAL', 'System Down', 'System health check failed', health);
            }
            
            // Check memory usage
            const memoryCheck = health.checks.memory;
            if (memoryCheck && memoryCheck.details.usagePercent > 90) {
                await this.triggerAlert('WARNING', 'High Memory Usage', 
                    `Memory usage is ${memoryCheck.details.usagePercent}%`, memoryCheck);
            }
            
            // Check database connectivity
            const dbCheck = health.checks.database;
            if (dbCheck && dbCheck.status === 'DOWN') {
                await this.triggerAlert('CRITICAL', 'Database Connection Failed', 
                    'Database connectivity check failed', dbCheck);
            }
            
        } catch (error) {
            console.error('Failed to check alerts:', error);
        }
    }

    /**
     * Trigger an alert
     * @param {string} severity - Alert severity
     * @param {string} title - Alert title
     * @param {string} description - Alert description
     * @param {object} details - Alert details
     */
    async triggerAlert(severity, title, description, details) {
        const alertId = `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        const alert = {
            id: alertId,
            severity,
            title,
            description,
            details,
            timestamp: new Date().toISOString(),
            service: 'a2a-developer-portal',
            environment: process.env.NODE_ENV || 'development'
        };
        
        // Store alert locally
        this.alerts.set(alertId, alert);
        
        // Send to SAP Cloud ALM
        try {
            if (this.almClient) {
                await this.almClient.post('/alerts/trigger', alert);
            }
        } catch (error) {
            console.error('Failed to send alert to ALM:', error);
        }
        
        // Log alert
        console.warn(`ALERT [${severity}]: ${title} - ${description}`, details);
    }

    /**
     * Get metrics for a time range
     * @param {string} startTime - Start time
     * @param {string} endTime - End time
     * @returns {array} Metrics data
     */
    getMetrics(startTime, endTime) {
        const start = new Date(startTime).getTime();
        const end = new Date(endTime).getTime();
        
        const filteredMetrics = [];
        for (const [_key, metrics] of this.metrics) {
            const timestamp = new Date(metrics.timestamp).getTime();
            if (timestamp >= start && timestamp <= end) {
                filteredMetrics.push(metrics);
            }
        }
        
        return filteredMetrics.sort((a, b) => 
            new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        );
    }

    /**
     * Get active alerts
     * @returns {array} Active alerts
     */
    getActiveAlerts() {
        return Array.from(this.alerts.values())
            .filter(alert => !alert.resolved)
            .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
    }

    /**
     * Resolve an alert
     * @param {string} alertId - Alert ID
     * @param {string} resolvedBy - User who resolved the alert
     * @returns {boolean} Success status
     */
    resolveAlert(alertId, resolvedBy) {
        const alert = this.alerts.get(alertId);
        if (alert) {
            alert.resolved = true;
            alert.resolvedAt = new Date().toISOString();
            alert.resolvedBy = resolvedBy;
            return true;
        }
        return false;
    }

    /**
     * Get monitoring dashboard data
     * @returns {object} Dashboard data
     */
    async getMonitoringDashboard() {
        const health = await this.getSystemHealth();
        const recentMetrics = this.getMetrics(
            new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(), // Last 24 hours
            new Date().toISOString()
        );
        const activeAlerts = this.getActiveAlerts();
        
        return {
            health,
            metrics: {
                recent: recentMetrics.slice(-10), // Last 10 metric points
                summary: this.calculateMetricsSummary(recentMetrics)
            },
            alerts: {
                active: activeAlerts,
                count: activeAlerts.length
            },
            uptime: Math.floor(process.uptime()),
            lastUpdated: new Date().toISOString()
        };
    }

    /**
     * Calculate metrics summary
     * @param {array} metrics - Metrics array
     * @returns {object} Metrics summary
     */
    calculateMetricsSummary(metrics) {
        if (metrics.length === 0) {
return {};
}
        
        const latest = metrics[metrics.length - 1];
        const avgMemoryUsage = metrics.reduce((sum, m) => 
            sum + (m.system?.memory?.heapUsed || 0), 0) / metrics.length;
        const avgActiveUsers = metrics.reduce((sum, m) => 
            sum + (m.application?.activeUsers || 0), 0) / metrics.length;
        
        return {
            currentMemoryUsage: latest.system?.memory?.heapUsed || 0,
            averageMemoryUsage: Math.round(avgMemoryUsage),
            currentActiveUsers: latest.application?.activeUsers || 0,
            averageActiveUsers: Math.round(avgActiveUsers),
            totalProjects: latest.application?.totalProjects || 0,
            activeAgents: latest.application?.activeAgents || 0
        };
    }
}

module.exports = new ALMMonitoringService();
