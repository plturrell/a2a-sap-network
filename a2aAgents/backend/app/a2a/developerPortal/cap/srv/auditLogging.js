"use strict";

const cds = require('@sap/cds');
const { INSERT, SELECT, DELETE } = cds.ql;
const { AuditLogging } = require('@sap/audit-logging');
const { v4: uuidv4 } = require('uuid');

/**
 * Audit Logging Service
 * Provides comprehensive audit trails for critical business actions
 */
class AuditLoggingService {
    
    constructor() {
        this.auditLogger = null;
        this.auditCategories = new Map();
        this.init();
    }

    init() {
        try {
            // Initialize SAP Audit Logging Service
            this.auditLogger = new AuditLogging({
                logToConsole: process.env.NODE_ENV === 'development',
                logLevel: 'info'
            });

            // Define audit categories
            this.defineAuditCategories();
            
             
            
            // eslint-disable-next-line no-console
            
             
            
            // eslint-disable-next-line no-console
            console.log('Audit Logging Service initialized successfully');
        } catch (error) {
            console.error('Failed to initialize Audit Logging Service:', error);
        }
    }

    /**
     * Define audit categories and their configurations
     */
    defineAuditCategories() {
        this.auditCategories.set('DATA_ACCESS', {
            name: 'Data Access',
            description: 'Data read/access operations',
            retention: '7 years',
            sensitivity: 'high'
        });

        this.auditCategories.set('DATA_MODIFICATION', {
            name: 'Data Modification',
            description: 'Data create/update/delete operations',
            retention: '10 years',
            sensitivity: 'critical'
        });

        this.auditCategories.set('AUTHENTICATION', {
            name: 'Authentication',
            description: 'User authentication and authorization events',
            retention: '3 years',
            sensitivity: 'high'
        });

        this.auditCategories.set('CONFIGURATION', {
            name: 'Configuration',
            description: 'System and application configuration changes',
            retention: '5 years',
            sensitivity: 'medium'
        });

        this.auditCategories.set('SECURITY', {
            name: 'Security',
            description: 'Security-related events and violations',
            retention: '10 years',
            sensitivity: 'critical'
        });

        this.auditCategories.set('BUSINESS_PROCESS', {
            name: 'Business Process',
            description: 'Business process execution and workflow events',
            retention: '7 years',
            sensitivity: 'high'
        });

        this.auditCategories.set('ADMIN', {
            name: 'Administration',
            description: 'Administrative actions and system management',
            retention: '5 years',
            sensitivity: 'high'
        });
    }

    /**
     * Log data access event
     * @param {object} context - Audit context
     * @param {string} entityType - Type of entity accessed
     * @param {string} entityId - ID of entity accessed
     * @param {array} fields - Fields accessed
     */
    async logDataAccess(context, entityType, entityId, fields = []) {
        const auditEvent = {
            category: 'DATA_ACCESS',
            action: 'READ',
            entityType,
            entityId,
            fields,
            ...this.buildBaseAuditEvent(context)
        };

        await this.writeAuditLog(auditEvent);
    }

    /**
     * Log data modification event
     * @param {object} context - Audit context
     * @param {string} action - Action performed (CREATE, UPDATE, DELETE)
     * @param {string} entityType - Type of entity modified
     * @param {string} entityId - ID of entity modified
     * @param {object} oldValues - Old values (for updates/deletes)
     * @param {object} newValues - New values (for creates/updates)
     */
    async logDataModification(context, action, entityType, entityId, oldValues = null, newValues = null) {
        const auditEvent = {
            category: 'DATA_MODIFICATION',
            action,
            entityType,
            entityId,
            oldValues: oldValues ? this.sanitizeData(oldValues) : null,
            newValues: newValues ? this.sanitizeData(newValues) : null,
            ...this.buildBaseAuditEvent(context)
        };

        await this.writeAuditLog(auditEvent);
    }

    /**
     * Log authentication event
     * @param {object} context - Audit context
     * @param {string} action - Authentication action (LOGIN, LOGOUT, FAILED_LOGIN)
     * @param {string} userId - User ID
     * @param {object} details - Additional details
     */
    async logAuthentication(context, action, userId, details = {}) {
        const auditEvent = {
            category: 'AUTHENTICATION',
            action,
            userId,
            details: this.sanitizeData(details),
            ...this.buildBaseAuditEvent(context)
        };

        await this.writeAuditLog(auditEvent);
    }

    /**
     * Log security event
     * @param {object} context - Audit context
     * @param {string} action - Security action
     * @param {string} severity - Event severity (LOW, MEDIUM, HIGH, CRITICAL)
     * @param {object} details - Security event details
     */
    async logSecurityEvent(context, action, severity, details = {}) {
        const auditEvent = {
            category: 'SECURITY',
            action,
            severity,
            details: this.sanitizeData(details),
            ...this.buildBaseAuditEvent(context)
        };

        await this.writeAuditLog(auditEvent);
    }

    /**
     * Log business process event
     * @param {object} context - Audit context
     * @param {string} processType - Type of business process
     * @param {string} processId - Process instance ID
     * @param {string} action - Process action
     * @param {object} processData - Process-specific data
     */
    async logBusinessProcess(context, processType, processId, action, processData = {}) {
        const auditEvent = {
            category: 'BUSINESS_PROCESS',
            processType,
            processId,
            action,
            processData: this.sanitizeData(processData),
            ...this.buildBaseAuditEvent(context)
        };

        await this.writeAuditLog(auditEvent);
    }

    /**
     * Log configuration change
     * @param {object} context - Audit context
     * @param {string} configType - Type of configuration
     * @param {string} configKey - Configuration key
     * @param {any} oldValue - Old configuration value
     * @param {any} newValue - New configuration value
     */
    async logConfigurationChange(context, configType, configKey, oldValue, newValue) {
        const auditEvent = {
            category: 'CONFIGURATION',
            action: 'CHANGE',
            configType,
            configKey,
            oldValue: this.sanitizeData(oldValue),
            newValue: this.sanitizeData(newValue),
            ...this.buildBaseAuditEvent(context)
        };

        await this.writeAuditLog(auditEvent);
    }

    /**
     * Log administrative action
     * @param {object} context - Audit context
     * @param {string} action - Administrative action
     * @param {string} targetType - Type of target (USER, SYSTEM, APPLICATION)
     * @param {string} targetId - Target identifier
     * @param {object} details - Action details
     */
    async logAdminAction(context, action, targetType, targetId, details = {}) {
        const auditEvent = {
            category: 'ADMIN',
            action,
            targetType,
            targetId,
            details: this.sanitizeData(details),
            ...this.buildBaseAuditEvent(context)
        };

        await this.writeAuditLog(auditEvent);
    }

    /**
     * Log project-specific events
     * @param {object} context - Audit context
     * @param {string} projectId - Project ID
     * @param {string} action - Project action
     * @param {object} details - Action details
     */
    async logProjectEvent(context, projectId, action, details = {}) {
        await this.logBusinessProcess(context, 'PROJECT', projectId, action, {
            projectDetails: details,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Log agent-specific events
     * @param {object} context - Audit context
     * @param {string} agentId - Agent ID
     * @param {string} action - Agent action
     * @param {object} details - Action details
     */
    async logAgentEvent(context, agentId, action, details = {}) {
        await this.logBusinessProcess(context, 'AGENT', agentId, action, {
            agentDetails: details,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Log workflow events
     * @param {object} context - Audit context
     * @param {string} workflowId - Workflow ID
     * @param {string} action - Workflow action
     * @param {object} details - Action details
     */
    async logWorkflowEvent(context, workflowId, action, details = {}) {
        await this.logBusinessProcess(context, 'WORKFLOW', workflowId, action, {
            workflowDetails: details,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Log deployment events
     * @param {object} context - Audit context
     * @param {string} deploymentId - Deployment ID
     * @param {string} action - Deployment action
     * @param {object} details - Action details
     */
    async logDeploymentEvent(context, deploymentId, action, details = {}) {
        await this.logBusinessProcess(context, 'DEPLOYMENT', deploymentId, action, {
            deploymentDetails: details,
            timestamp: new Date().toISOString()
        });
    }

    /**
     * Write audit log entry
     * @param {object} auditEvent - Audit event data
     */
    async writeAuditLog(auditEvent) {
        try {
            // Add unique audit ID
            auditEvent.auditId = uuidv4();
            auditEvent.timestamp = new Date().toISOString();

            // Write to SAP Audit Logging Service
            if (this.auditLogger) {
                await this.auditLogger.log(auditEvent);
            }

            // Store in database for local queries
            await this.storeAuditLogInDatabase(auditEvent);

            // Log to console in development
            if (process.env.NODE_ENV === 'development') {
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log('AUDIT LOG:', JSON.stringify(auditEvent, null, 2));
            }

        } catch (error) {
            console.error('Failed to write audit log:', error);
            // Don't throw error to avoid breaking business operations
        }
    }

    /**
     * Store audit log in database
     * @param {object} auditEvent - Audit event data
     */
    async storeAuditLogInDatabase(auditEvent) {
        try {
            const db = await cds.connect.to('db');
            
            await db.run(
                INSERT.into('AuditLogs').entries({
                    ID: auditEvent.auditId,
                    user_ID: auditEvent.userId,
                    action: auditEvent.action,
                    entityType: auditEvent.entityType || auditEvent.category,
                    entityId: auditEvent.entityId,
                    oldValues: auditEvent.oldValues ? JSON.stringify(auditEvent.oldValues) : null,
                    newValues: auditEvent.newValues ? JSON.stringify(auditEvent.newValues) : null,
                    timestamp: new Date(auditEvent.timestamp),
                    ipAddress: auditEvent.ipAddress,
                    userAgent: auditEvent.userAgent,
                    sessionId: auditEvent.sessionId,
                    success: auditEvent.success !== false,
                    errorMessage: auditEvent.errorMessage
                })
            );
        } catch (error) {
            console.error('Failed to store audit log in database:', error);
        }
    }

    /**
     * Build base audit event structure
     * @param {object} context - Audit context
     * @returns {object} Base audit event
     */
    buildBaseAuditEvent(context) {
        return {
            userId: context.userId || context.user?.id,
            sessionId: context.sessionId,
            ipAddress: context.ipAddress,
            userAgent: context.userAgent,
            correlationId: context.correlationId,
            tenant: context.tenant || 'default',
            success: context.success !== false,
            errorMessage: context.errorMessage,
            additionalInfo: context.additionalInfo
        };
    }

    /**
     * Sanitize sensitive data before logging
     * @param {any} data - Data to sanitize
     * @returns {any} Sanitized data
     */
    sanitizeData(data) {
        if (!data) {
return data;
}
        
        const sensitiveFields = [
            'password', 'token', 'secret', 'key', 'credential',
            'authorization', 'cookie', 'session'
        ];
        
        if (typeof data === 'object') {
            const sanitized = { ...data };
            
            for (const field of sensitiveFields) {
                if (field in sanitized) {
                    sanitized[field] = '[REDACTED]';
                }
            }
            
            // Recursively sanitize nested objects
            for (const key in sanitized) {
                if (typeof sanitized[key] === 'object') {
                    sanitized[key] = this.sanitizeData(sanitized[key]);
                }
            }
            
            return sanitized;
        }
        
        return data;
    }

    /**
     * Search audit logs
     * @param {object} criteria - Search criteria
     * @returns {array} Audit log entries
     */
    async searchAuditLogs(criteria) {
        try {
            const {
                userId,
                category: _category,
                action,
                entityType,
                entityId,
                startTime,
                endTime,
                limit = 100,
                offset = 0
            } = criteria;

            const db = await cds.connect.to('db');
            let query = SELECT.from('AuditLogs');
            
            const conditions = [];
            
            if (userId) {
conditions.push({ user_ID: userId });
}
            if (action) {
conditions.push({ action });
}
            if (entityType) {
conditions.push({ entityType });
}
            if (entityId) {
conditions.push({ entityId });
}
            if (startTime) {
conditions.push({ timestamp: { '>=': new Date(startTime) } });
}
            if (endTime) {
conditions.push({ timestamp: { '<=': new Date(endTime) } });
}
            
            if (conditions.length > 0) {
                query = query.where(conditions);
            }
            
            query = query.orderBy('timestamp desc').limit(limit, offset);
            
            const results = await db.run(query);
            
            return results.map(log => ({
                ...log,
                oldValues: log.oldValues ? JSON.parse(log.oldValues) : null,
                newValues: log.newValues ? JSON.parse(log.newValues) : null
            }));
            
        } catch (error) {
            console.error('Failed to search audit logs:', error);
            throw error;
        }
    }

    /**
     * Get audit statistics
     * @param {string} timeframe - Timeframe (1d, 7d, 30d)
     * @returns {object} Audit statistics
     */
    async getAuditStatistics(timeframe = '7d') {
        try {
            const db = await cds.connect.to('db');
            
            const days = timeframe === '1d' ? 1 : timeframe === '7d' ? 7 : 30;
            const startDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
            
            const [
                totalLogs,
                userActions,
                entityTypes,
                topUsers
            ] = await Promise.all([
                db.run(`SELECT COUNT(*) as count FROM AuditLogs WHERE timestamp >= ?`, [startDate]),
                db.run(`SELECT action, COUNT(*) as count FROM AuditLogs WHERE timestamp >= ? GROUP BY action ORDER BY count DESC LIMIT 10`, [startDate]),
                db.run(`SELECT entityType, COUNT(*) as count FROM AuditLogs WHERE timestamp >= ? GROUP BY entityType ORDER BY count DESC LIMIT 10`, [startDate]),
                db.run(`SELECT user_ID, COUNT(*) as count FROM AuditLogs WHERE timestamp >= ? GROUP BY user_ID ORDER BY count DESC LIMIT 10`, [startDate])
            ]);
            
            return {
                timeframe,
                totalLogs: totalLogs[0]?.count || 0,
                topActions: userActions,
                topEntityTypes: entityTypes,
                topUsers: topUsers,
                generatedAt: new Date().toISOString()
            };
            
        } catch (error) {
            console.error('Failed to get audit statistics:', error);
            throw error;
        }
    }

    /**
     * Export audit logs
     * @param {object} criteria - Export criteria
     * @param {string} format - Export format (json, csv)
     * @returns {string} Export file path or data
     */
    async exportAuditLogs(criteria, format = 'json') {
        try {
            const logs = await this.searchAuditLogs({
                ...criteria,
                limit: 10000 // Large limit for export
            });
            
            if (format === 'csv') {
                return this.convertToCSV(logs);
            }
            
            return JSON.stringify(logs, null, 2);
            
        } catch (error) {
            console.error('Failed to export audit logs:', error);
            throw error;
        }
    }

    /**
     * Convert audit logs to CSV format
     * @param {array} logs - Audit log entries
     * @returns {string} CSV data
     */
    convertToCSV(logs) {
        if (logs.length === 0) {
return '';
}
        
        const headers = Object.keys(logs[0]).join(',');
        const rows = logs.map(log => 
            Object.values(log).map(value => 
                typeof value === 'object' ? JSON.stringify(value) : value
            ).join(',')
        );
        
        return [headers, ...rows].join('\n');
    }

    /**
     * Get audit categories
     * @returns {array} Audit categories
     */
    getAuditCategories() {
        return Array.from(this.auditCategories.entries()).map(([key, value]) => ({
            key,
            ...value
        }));
    }

    /**
     * Validate audit log retention
     */
    async validateRetention() {
        try {
            const db = await cds.connect.to('db');
            
            // Clean up old audit logs based on retention policies - batch delete for better performance
            const deletePromises = [];
            
            for (const [category, config] of this.auditCategories) {
                const retentionDays = this.parseRetentionPeriod(config.retention);
                const cutoffDate = new Date(Date.now() - retentionDays * 24 * 60 * 60 * 1000);
                
                // Batch all delete operations instead of sequential execution
                deletePromises.push(
                    db.run(
                        DELETE.from('AuditLogs')
                            .where({ 
                                entityType: category,
                                timestamp: { '<': cutoffDate }
                            })
                    )
                );
            }
            
            // Execute all deletes in parallel for better performance
            await Promise.all(deletePromises);
            
        } catch (error) {
            console.error('Failed to validate audit log retention:', error);
        }
    }

    /**
     * Parse retention period string to days
     * @param {string} retention - Retention period (e.g., "7 years", "30 days")
     * @returns {number} Number of days
     */
    parseRetentionPeriod(retention) {
        const match = retention.match(/(\d+)\s*(year|month|day)s?/i);
        if (!match) {
return 2555;
} // Default 7 years
        
        const [, number, unit] = match;
        const multipliers = {
            day: 1,
            month: 30,
            year: 365
        };
        
        return parseInt(number) * (multipliers[unit.toLowerCase()] || 1);
    }
}

module.exports = new AuditLoggingService();
