/**
 * @fileoverview CAP Monitoring Service with Auto-scaling Integration
 * @since 1.0.0
 * @module monitoringService
 * 
 * SAP CAP service that integrates performance monitoring and auto-scaling
 * Provides comprehensive system monitoring and automatic scaling capabilities
 */

const cds = require('@sap/cds');
const PerformanceMonitor = require('../app/monitoring/performanceMonitor');
const AutoScaler = require('../app/monitoring/autoScaler');

module.exports = class MonitoringService extends cds.ApplicationService {
    
    async init() {
        // Initialize monitoring components
        this.performanceMonitor = new PerformanceMonitor({
            systemInterval: 30000,  // 30 seconds
            agentInterval: 60000,   // 1 minute
            metricsInterval: 15000, // 15 seconds
            cpuScaleUp: 75,
            cpuScaleDown: 25,
            memoryScaleUp: 80,
            memoryScaleDown: 30,
            responseTimeScaleUp: 2000,
            responseTimeScaleDown: 500
        });
        
        this.autoScaler = new AutoScaler({
            flyAppName: process.env.FLY_APP_NAME,
            flyApiToken: process.env.FLY_API_TOKEN,
            minInstances: parseInt(process.env.MIN_INSTANCES) || 1,
            maxInstances: parseInt(process.env.MAX_INSTANCES) || 10,
            enableAutoScaling: process.env.ENABLE_AUTO_SCALING !== 'false',
            cooldownPeriod: 300000 // 5 minutes
        });
        
        // Initialize SAP Alert Manager and Notification Integration
        const SAPAlertManager = require('../app/lib/sapAlertManager');
        const SAPNotificationIntegration = require('../app/lib/sapNotificationIntegration');
        
        this.alertManager = new SAPAlertManager({
            maxAlerts: 5000,
            escalationEnabled: true,
            cloudALMEnabled: process.env.SAP_CLOUD_ALM_ENABLED === 'true'
        });
        
        this.notificationIntegration = new SAPNotificationIntegration({
            enableWebhookBridge: true,
            enableLegacyBridge: true,
            enableWebSocketBridge: true
        });
        
        // Connect alert manager to notification integration
        this.notificationIntegration.connectAlertManager(this.alertManager);
        
        // Connect to existing notification services
        try {
            const notificationService = await cds.connect.to('NotificationService');
            this.notificationIntegration.connectWebhookService(notificationService.getWebhookService?.());
        } catch (error) {
            LOG.warn('Could not connect to webhook service', error);
        }
        
        // Connect performance monitor to alert manager
        this.performanceMonitor.on('alert', (alert) => {
            // Convert performance monitor alert to SAP alert
            this.alertManager.createAlert('system_performance_alert', {
                component: alert.metadata?.component || 'system',
                metric: alert.metadata?.metric,
                value: alert.metadata?.value,
                threshold: alert.metadata?.threshold
            }, {
                correlationId: alert.id,
                source: 'performance-monitor'
            });
        });
        
        // Connect performance monitor to auto-scaler
        this.performanceMonitor.on('performanceAnalysis', async (analysis) => {
            try {
                const scalingDecision = await this.autoScaler.evaluateScaling({
                    system: analysis.system?.current,
                    agents: analysis.agents?.current,
                    timestamp: analysis.timestamp
                });
                
                if (scalingDecision.action !== 'none') {
                    console.log(`🎯 Auto-scaling triggered: ${scalingDecision.action} - ${scalingDecision.reason}`);
                    const result = await this.autoScaler.executeScaling(scalingDecision);
                    
                    // Emit event for notification system
                    if (result.success) {
                        await this.emit('scalingCompleted', {
                            action: scalingDecision.action,
                            reason: scalingDecision.reason,
                            fromInstances: result.event.fromInstances,
                            toInstances: result.event.toInstances,
                            duration: result.event.duration
                        });
                    } else {
                        await this.emit('scalingFailed', {
                            action: scalingDecision.action,
                            reason: scalingDecision.reason,
                            error: result.message
                        });
                    }
                }
            } catch (error) {
                console.error('Auto-scaling evaluation failed:', error);
            }
        });
        
        // Listen to alerts for critical issues
        this.performanceMonitor.on('alert', (alert) => {
            console.log(`🚨 Performance Alert [${alert.severity.toUpperCase()}]: ${alert.message}`);
            
            // Emit alert event for notification system
            this.emit('performanceAlert', {
                alertId: alert.id,
                severity: alert.severity,
                category: alert.category,
                message: alert.message,
                metadata: alert.metadata,
                timestamp: alert.timestamp
            });
        });
        
        // Function: Get current system metrics
        this.on('getSystemMetrics', async () => {
            const metrics = this.performanceMonitor.getCurrentMetrics();
            const scalerStatus = this.autoScaler.getStatus();
            
            return {
                timestamp: new Date().toISOString(),
                system: metrics.system?.system || {},
                agents: metrics.agents?.agents || {},
                analysis: metrics.analysis || {},
                scaling: {
                    enabled: scalerStatus.enabled,
                    currentInstances: scalerStatus.currentInstances,
                    targetInstances: scalerStatus.targetInstances,
                    scalingInProgress: scalerStatus.scalingInProgress,
                    cooldownRemaining: scalerStatus.cooldownRemaining,
                    limits: scalerStatus.limits
                }
            };
        });
        
        // Function: Get performance history
        this.on('getPerformanceHistory', async (req) => {
            const { category, limit, offset } = req.data;
            
            const metrics = this.performanceMonitor.getMetrics(category, limit || 100);
            const total = metrics.length;
            
            return {
                category: category || 'all',
                metrics: metrics.slice(offset || 0, (offset || 0) + (limit || 100)),
                total,
                limit: limit || 100,
                offset: offset || 0
            };
        });
        
        // Function: Get alerts
        this.on('getAlerts', async (req) => {
            const { severity, category, unacknowledged, limit } = req.data;
            
            const alerts = this.performanceMonitor.getAlerts({
                severity,
                category,
                unacknowledged: unacknowledged === true,
                limit: limit || 50
            });
            
            return {
                alerts,
                total: alerts.length,
                unacknowledged: alerts.filter(a => !a.acknowledged).length,
                critical: alerts.filter(a => a.severity === 'critical').length
            };
        });
        
        // Function: Get scaling history
        this.on('getScalingHistory', async (req) => {
            const { limit } = req.data;
            
            const history = this.autoScaler.getScalingHistory(limit || 50);
            const stats = this.autoScaler.getScalingStats();
            
            return {
                history,
                stats,
                current: this.autoScaler.getStatus()
            };
        });
        
        // Function: Get alerts from SAP Alert Manager
        this.on('getAlerts', async (req) => {
            const { severity, category, acknowledged, resolved, limit } = req.data;
            
            const alerts = this.alertManager.getAlerts({
                severity,
                category,
                acknowledged,
                resolved,
                limit: limit || 50
            });
            
            return {
                alerts: alerts.map(alert => ({
                    id: alert.id,
                    title: alert.title,
                    description: alert.description,
                    severity: alert.severity,
                    category: alert.category,
                    status: alert.status,
                    acknowledged: alert.acknowledged,
                    resolved: alert.resolved,
                    createdAt: alert.createdAt,
                    sapAlertCode: alert.sapAlertCode,
                    correlationId: alert.correlationId,
                    escalationLevel: alert.escalationLevel,
                    runbook: alert.runbook
                })),
                statistics: this.alertManager.getStatistics()
            };
        });
        
        // Function: Get notification integration status
        this.on('getNotificationStatus', async () => {
            return this.notificationIntegration.getStatus();
        });
        
        // Action: Create manual alert
        this.on('createAlert', async (req) => {
            const { templateId, context, metadata } = req.data;
            
            try {
                const alert = this.alertManager.createAlert(templateId, context, {
                    ...metadata,
                    createdBy: req.user.id,
                    source: 'manual'
                });
                
                return {
                    success: true,
                    alert: {
                        id: alert.id,
                        title: alert.title,
                        severity: alert.severity,
                        status: alert.status
                    }
                };
            } catch (error) {
                req.error(400, `Failed to create alert: ${error.message}`);
            }
        });
        
        // Action: Acknowledge alert
        this.on('acknowledgeAlert', async (req) => {
            const { alertId, notes } = req.data;
            
            try {
                const alert = this.alertManager.acknowledgeAlert(alertId, req.user.id, notes);
                
                return {
                    success: true,
                    message: 'Alert acknowledged successfully',
                    alert: {
                        id: alert.id,
                        acknowledged: alert.acknowledged,
                        acknowledgedAt: alert.acknowledgedAt,
                        acknowledgedBy: alert.acknowledgedBy
                    }
                };
            } catch (error) {
                req.error(400, `Failed to acknowledge alert: ${error.message}`);
            }
        });
        
        // Action: Resolve alert
        this.on('resolveAlert', async (req) => {
            const { alertId, resolution, rootCause } = req.data;
            
            try {
                const alert = this.alertManager.resolveAlert(alertId, req.user.id, resolution, rootCause);
                
                return {
                    success: true,
                    message: 'Alert resolved successfully',
                    alert: {
                        id: alert.id,
                        resolved: alert.resolved,
                        resolvedAt: alert.resolvedAt,
                        resolution: alert.resolution,
                        resolutionTime: alert.resolutionTime
                    }
                };
            } catch (error) {
                req.error(400, `Failed to resolve alert: ${error.message}`);
            }
        });
        
        // Action: Send manual notification
        this.on('sendNotification', async (req) => {
            const { title, message, severity, category, channels } = req.data;
            
            try {
                await this.notificationIntegration.sendManualNotification({
                    title,
                    message,
                    severity: severity || 'info',
                    category: category || 'manual',
                    channels: channels || ['websocket'],
                    metadata: {
                        sentBy: req.user.id,
                        source: 'manual'
                    }
                });
                
                return {
                    success: true,
                    message: 'Notification sent successfully'
                };
            } catch (error) {
                req.error(500, `Failed to send notification: ${error.message}`);
            }
        });
        
        // Function: Get monitoring dashboard data
        this.on('getMonitoringDashboard', async () => {
            const currentMetrics = this.performanceMonitor.getCurrentMetrics();
            const alerts = this.alertManager.getAlerts({ limit: 10 });
            const scalerStatus = this.autoScaler.getStatus();
            const scalingStats = this.autoScaler.getScalingStats();
            
            return {
                timestamp: new Date().toISOString(),
                overview: {
                    systemHealth: this.calculateSystemHealth(currentMetrics),
                    instances: scalerStatus.currentInstances,
                    activeAlerts: alerts.filter(a => !a.acknowledged).length,
                    scalingEvents: scalingStats.total,
                    uptime: process.uptime()
                },
                metrics: {
                    system: currentMetrics.system?.system || {},
                    agents: currentMetrics.agents?.agents || {},
                    scaling: scalerStatus
                },
                alerts: alerts.slice(0, 5), // Recent alerts
                scaling: {
                    status: scalerStatus,
                    stats: scalingStats,
                    recentEvents: this.autoScaler.getScalingHistory(5)
                },
                performance: {
                    trends: currentMetrics.analysis?.trends || {},
                    recommendations: currentMetrics.analysis?.recommendations || []
                }
            };
        });
        
        // Action: Acknowledge alert
        this.on('acknowledgeAlert', async (req) => {
            const { alertId } = req.data;
            
            const alert = this.performanceMonitor.acknowledgeAlert(alertId);
            
            if (alert) {
                return {
                    success: true,
                    message: 'Alert acknowledged successfully',
                    alert: {
                        id: alert.id,
                        message: alert.message,
                        acknowledged: alert.acknowledged,
                        acknowledgedAt: alert.acknowledgedAt
                    }
                };
            } else {
                req.error(404, 'Alert not found');
            }
        });
        
        // Action: Resolve alert
        this.on('resolveAlert', async (req) => {
            const { alertId, resolution } = req.data;
            
            const alert = this.performanceMonitor.resolveAlert(alertId);
            
            if (alert) {
                alert.resolution = resolution;
                
                return {
                    success: true,
                    message: 'Alert resolved successfully',
                    alert: {
                        id: alert.id,
                        message: alert.message,
                        resolved: alert.resolved,
                        resolvedAt: alert.resolvedAt,
                        resolution: resolution
                    }
                };
            } else {
                req.error(404, 'Alert not found');
            }
        });
        
        // Action: Manual scaling
        this.on('manualScale', async (req) => {
            const { targetInstances, reason } = req.data;
            
            try {
                const result = await this.autoScaler.manualScale(
                    targetInstances, 
                    reason || `Manual scaling by ${req.user.id}`
                );
                
                if (result.success) {
                    // Emit event for notifications
                    await this.emit('manualScalingCompleted', {
                        action: result.event.action,
                        fromInstances: result.event.fromInstances,
                        toInstances: result.event.toInstances,
                        reason: result.event.reason,
                        user: req.user.id
                    });
                    
                    return {
                        success: true,
                        message: 'Manual scaling initiated successfully',
                        scalingEvent: {
                            id: result.event.id,
                            action: result.event.action,
                            fromInstances: result.event.fromInstances,
                            toInstances: result.event.toInstances,
                            status: result.event.status
                        }
                    };
                } else {
                    req.error(400, result.message);
                }
            } catch (error) {
                req.error(500, `Manual scaling failed: ${error.message}`);
            }
        });
        
        // Action: Emergency scaling
        this.on('emergencyScale', async (req) => {
            const { targetInstances, justification } = req.data;
            
            if (!justification) {
                req.error(400, 'Emergency scaling requires justification');
                return;
            }
            
            try {
                const result = await this.autoScaler.emergencyScale(targetInstances);
                
                // Log emergency scaling for audit
                console.log(`🚨 EMERGENCY SCALING executed by ${req.user.id}: ${targetInstances} instances - ${justification}`);
                
                // Emit critical event
                await this.emit('emergencyScalingExecuted', {
                    user: req.user.id,
                    targetInstances,
                    justification,
                    scalingEvent: result.event,
                    timestamp: new Date().toISOString()
                });
                
                return {
                    success: true,
                    message: 'Emergency scaling executed',
                    scalingEvent: result.event,
                    warning: 'Emergency scaling bypassed normal safety limits'
                };
                
            } catch (error) {
                req.error(500, `Emergency scaling failed: ${error.message}`);
            }
        });
        
        // Action: Update auto-scaling configuration
        this.on('updateAutoScalingConfig', async (req) => {
            const { config } = req.data;
            
            try {
                this.autoScaler.updateConfig(config);
                
                return {
                    success: true,
                    message: 'Auto-scaling configuration updated',
                    config: this.autoScaler.getStatus()
                };
            } catch (error) {
                req.error(400, `Failed to update configuration: ${error.message}`);
            }
        });
        
        // Action: Enable/disable auto-scaling
        this.on('toggleAutoScaling', async (req) => {
            const { enabled } = req.data;
            
            if (enabled) {
                this.autoScaler.enable();
            } else {
                this.autoScaler.disable();
            }
            
            return {
                success: true,
                message: `Auto-scaling ${enabled ? 'enabled' : 'disabled'}`,
                enabled: this.autoScaler.getStatus().enabled
            };
        });
        
        // Action: Get real-time metrics stream
        this.on('getMetricsStream', async (req) => {
            // This would typically return a stream or webhook URL for real-time data
            // For now, return current snapshot
            const metrics = this.performanceMonitor.getCurrentMetrics();
            const status = this.performanceMonitor.getMonitoringStatus();
            
            return {
                timestamp: new Date().toISOString(),
                metrics,
                monitoring: status,
                streamInfo: {
                    available: true,
                    interval: '15s',
                    endpoint: '/api/v1/monitoring/stream'
                }
            };
        });
        
        // Action: Generate monitoring report
        this.on('generateReport', async (req) => {
            const { timeRange, includeDetails } = req.data;
            
            const report = {
                generated: new Date().toISOString(),
                timeRange: timeRange || 'last_24h',
                summary: {
                    systemHealth: this.calculateSystemHealth(this.performanceMonitor.getCurrentMetrics()),
                    totalAlerts: this.performanceMonitor.getAlerts().length,
                    scalingEvents: this.autoScaler.getScalingStats(),
                    uptime: process.uptime()
                }
            };
            
            if (includeDetails) {
                report.details = {
                    alerts: this.performanceMonitor.getAlerts({ limit: 100 }),
                    scalingHistory: this.autoScaler.getScalingHistory(50),
                    performanceMetrics: this.performanceMonitor.getMetrics('system', 100)
                };
            }
            
            return report;
        });
        
        return super.init();
    }
    
    calculateSystemHealth(metrics) {
        if (!metrics.system?.system && !metrics.agents?.agents) {
            return { status: 'unknown', score: 0 };
        }
        
        let score = 100;
        let issues = [];
        
        // System health
        if (metrics.system?.system) {
            const { cpu, memory } = metrics.system.system;
            
            if (cpu?.usage > 90) {
                score -= 30;
                issues.push('High CPU usage');
            } else if (cpu?.usage > 75) {
                score -= 15;
                issues.push('Elevated CPU usage');
            }
            
            if (memory?.usagePercentage > 95) {
                score -= 30;
                issues.push('Critical memory usage');
            } else if (memory?.usagePercentage > 80) {
                score -= 15;
                issues.push('High memory usage');
            }
        }
        
        // Agent health
        if (metrics.agents?.agents) {
            const healthPercentage = (metrics.agents.agents.healthy / metrics.agents.agents.total) * 100;
            
            if (healthPercentage < 50) {
                score -= 40;
                issues.push('Critical agent failures');
            } else if (healthPercentage < 80) {
                score -= 20;
                issues.push('Multiple agent failures');
            } else if (healthPercentage < 95) {
                score -= 10;
                issues.push('Some agent issues');
            }
        }
        
        // Determine status
        let status;
        if (score >= 90) {
            status = 'healthy';
        } else if (score >= 70) {
            status = 'warning';
        } else if (score >= 50) {
            status = 'degraded';
        } else {
            status = 'critical';
        }
        
        return {
            status,
            score: Math.max(0, score),
            issues,
            timestamp: new Date().toISOString()
        };
    }
    
    async shutdown() {
        console.log('🛑 Shutting down Monitoring Service...');
        
        if (this.performanceMonitor) {
            this.performanceMonitor.stopMonitoring();
        }
        
        if (this.autoScaler) {
            await this.autoScaler.shutdown();
        }
        
        console.log('✅ Monitoring Service shutdown complete');
    }
};