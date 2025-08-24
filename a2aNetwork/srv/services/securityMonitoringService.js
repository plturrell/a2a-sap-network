/**
 * Security Monitoring Service for A2A Network
 * Implements real-time security monitoring, alerting, and incident response
 * Meets SAP Enterprise Security Standards
 */

const EventEmitter = require('events');
const cds = require('@sap/cds');
const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');

class SecurityMonitoringService extends EventEmitter {
    constructor() {
        super();
        this.log = cds.log('security-monitor');
        this.securityEvents = new Map();
        this.alertThresholds = {
            failedLogins: { count: 5, window: 300000 }, // 5 attempts in 5 minutes
            rateLimitHits: { count: 10, window: 600000 }, // 10 hits in 10 minutes
            suspiciousRequests: { count: 3, window: 180000 }, // 3 requests in 3 minutes
            unauthorizedAccess: { count: 1, window: 60000 }, // 1 attempt in 1 minute
            injectionAttempts: { count: 1, window: 60000 }, // 1 attempt in 1 minute
            dataExfiltration: { count: 3, window: 300000 } // 3 attempts in 5 minutes
        };
        this.activeAlerts = new Map();
        this.securityMetrics = {
            totalEvents: 0,
            criticalEvents: 0,
            highEvents: 0,
            mediumEvents: 0,
            lowEvents: 0,
            blockedIPs: new Set(),
            quarantinedUsers: new Set()
        };
        this.securityLog = [];
        this.maxLogEntries = 10000;
        this.init();
    }

    async init() {
        try {
            // Initialize security event patterns
            this.initializeEventPatterns();
            
            // Start security monitoring tasks
            this.startMonitoring();
            
            // Initialize security dashboard data
            await this.initializeDashboard();
            
            this.log.info('Security Monitoring Service initialized successfully');
        } catch (error) {
            this.log.error('Failed to initialize Security Monitoring Service:', error);
            throw error;
        }
    }

    initializeEventPatterns() {
        this.securityPatterns = {
            // Authentication-related patterns
            bruteForce: {
                pattern: /failed.*login|authentication.*failed|invalid.*credentials/i,
                severity: 'high',
                category: 'AUTHENTICATION',
                action: 'BLOCK_IP'
            },
            suspiciousLogin: {
                pattern: /login.*unusual.*location|multiple.*sessions|concurrent.*access/i,
                severity: 'medium',
                category: 'AUTHENTICATION',
                action: 'ALERT'
            },
            
            // Injection attack patterns
            sqlInjection: {
                pattern: /(union.*select|drop.*table|exec.*xp_|waitfor.*delay|benchmark\()/i,
                severity: 'critical',
                category: 'INJECTION',
                action: 'BLOCK_IMMEDIATE'
            },
            xssAttempt: {
                pattern: /(<script.*>|javascript:|on\w+\s*=|<iframe|<object)/i,
                severity: 'high',
                category: 'INJECTION',
                action: 'BLOCK_REQUEST'
            },
            
            // Data access patterns
            dataExfiltration: {
                pattern: /bulk.*download|mass.*export|excessive.*data.*access/i,
                severity: 'high',
                category: 'DATA_ACCESS',
                action: 'QUARANTINE_USER'
            },
            unauthorizedAccess: {
                pattern: /access.*denied|permission.*denied|unauthorized.*attempt/i,
                severity: 'medium',
                category: 'AUTHORIZATION',
                action: 'ALERT'
            },
            
            // System security patterns
            configurationChange: {
                pattern: /security.*config.*changed|permissions.*modified|role.*assigned/i,
                severity: 'medium',
                category: 'SYSTEM',
                action: 'AUDIT_LOG'
            },
            systemIntrusion: {
                pattern: /shell.*access|command.*injection|file.*traversal|directory.*listing/i,
                severity: 'critical',
                category: 'SYSTEM',
                action: 'BLOCK_IMMEDIATE'
            }
        };
    }

    startMonitoring() {
        // Real-time security event processing
        this.on('securityEvent', this.processSecurityEvent.bind(this));
        
        // Periodic security health checks
        setInterval(() => this.performSecurityHealthCheck(), 60000); // Every minute
        
        // Alert cleanup and maintenance
        setInterval(() => this.cleanupExpiredAlerts(), 300000); // Every 5 minutes
        
        // Security metrics aggregation
        setInterval(() => this.aggregateSecurityMetrics(), 30000); // Every 30 seconds
        
        this.log.info('Security monitoring tasks started');
    }

    async initializeDashboard() {
        // Initialize security dashboard with current state
        this.dashboardData = {
            overview: {
                totalEvents: this.securityMetrics.totalEvents,
                activeAlerts: this.activeAlerts.size,
                blockedIPs: this.securityMetrics.blockedIPs.size,
                quarantinedUsers: this.securityMetrics.quarantinedUsers.size,
                lastUpdated: new Date().toISOString()
            },
            recentEvents: [],
            alertsSummary: {},
            threatIntelligence: {
                knownThreats: 0,
                blockedAttacks: 0,
                riskScore: 'LOW'
            },
            complianceStatus: {
                dataProtection: 'COMPLIANT',
                auditLogs: 'COMPLIANT',
                accessControls: 'COMPLIANT',
                encryptionStatus: 'COMPLIANT'
            }
        };
    }

    // Main security event processing
    processSecurityEvent(eventData) {
        try {
            const securityEvent = this.normalizeSecurityEvent(eventData);
            
            // Classify and score the event
            const classification = this.classifySecurityEvent(securityEvent);
            securityEvent.classification = classification;
            
            // Update metrics
            this.updateSecurityMetrics(securityEvent);
            
            // Check for alert conditions
            this.checkAlertConditions(securityEvent);
            
            // Take automated security actions if needed
            this.executeSecurityActions(securityEvent);
            
            // Log the event
            this.logSecurityEvent(securityEvent);
            
            // Emit for real-time dashboards
            this.emit('dashboardUpdate', {
                type: 'securityEvent',
                data: securityEvent
            });
            
        } catch (error) {
            this.log.error('Error processing security event:', error);
        }
    }

    normalizeSecurityEvent(eventData) {
        const now = new Date();
        return {
            id: crypto.randomUUID(),
            timestamp: now.toISOString(),
            epochTime: now.getTime(),
            severity: eventData.severity || 'low',
            category: eventData.category || 'GENERAL',
            source: eventData.source || 'unknown',
            description: eventData.description || eventData.message || '',
            ipAddress: eventData.ipAddress || eventData.ip || 'unknown',
            userId: eventData.userId || eventData.user || null,
            userAgent: eventData.userAgent || null,
            endpoint: eventData.endpoint || eventData.url || null,
            requestMethod: eventData.method || null,
            statusCode: eventData.statusCode || null,
            metadata: eventData.metadata || {},
            rawData: eventData
        };
    }

    classifySecurityEvent(securityEvent) {
        let bestMatch = null;
        let highestScore = 0;
        
        for (const [patternName, pattern] of Object.entries(this.securityPatterns)) {
            if (pattern.pattern.test(securityEvent.description)) {
                const score = this.calculateThreatScore(securityEvent, pattern);
                if (score > highestScore) {
                    highestScore = score;
                    bestMatch = {
                        pattern: patternName,
                        severity: pattern.severity,
                        category: pattern.category,
                        recommendedAction: pattern.action,
                        threatScore: score
                    };
                }
            }
        }
        
        return bestMatch || {
            pattern: 'unknown',
            severity: securityEvent.severity,
            category: securityEvent.category,
            recommendedAction: 'LOG',
            threatScore: this.getSeverityScore(securityEvent.severity)
        };
    }

    calculateThreatScore(securityEvent, pattern) {
        let score = this.getSeverityScore(pattern.severity);
        
        // Increase score for repeat offenders
        const recentEvents = this.getRecentEventsByIP(securityEvent.ipAddress, 300000);
        if (recentEvents.length > 1) {
            score *= (1 + recentEvents.length * 0.2);
        }
        
        // Increase score for known malicious patterns
        const suspiciousPatterns = ['admin', 'wp-admin', '.env', 'phpinfo', 'shell'];
        if (suspiciousPatterns.some(p => securityEvent.endpoint?.toLowerCase().includes(p))) {
            score *= 1.5;
        }
        
        return Math.min(score, 100); // Cap at 100
    }

    getSeverityScore(severity) {
        const scoreMap = {
            'critical': 90,
            'high': 70,
            'medium': 50,
            'low': 30,
            'info': 10
        };
        return scoreMap[severity.toLowerCase()] || 30;
    }

    updateSecurityMetrics(securityEvent) {
        this.securityMetrics.totalEvents++;
        
        switch (securityEvent.severity.toLowerCase()) {
            case 'critical':
                this.securityMetrics.criticalEvents++;
                break;
            case 'high':
                this.securityMetrics.highEvents++;
                break;
            case 'medium':
                this.securityMetrics.mediumEvents++;
                break;
            case 'low':
                this.securityMetrics.lowEvents++;
                break;
        }
    }

    checkAlertConditions(securityEvent) {
        const alertKey = `${securityEvent.category}_${securityEvent.ipAddress}`;
        const threshold = this.alertThresholds[securityEvent.category.toLowerCase()] || 
                         this.alertThresholds.suspiciousRequests;
        
        // Get events in time window
        const recentEvents = this.getRecentEventsByKey(alertKey, threshold.window);
        
        if (recentEvents.length >= threshold.count) {
            this.triggerSecurityAlert({
                id: crypto.randomUUID(),
                type: securityEvent.category,
                severity: securityEvent.severity,
                description: `Multiple ${securityEvent.category} events detected`,
                ipAddress: securityEvent.ipAddress,
                userId: securityEvent.userId,
                eventCount: recentEvents.length,
                timeWindow: threshold.window / 1000 / 60, // minutes
                firstEvent: recentEvents[0].timestamp,
                lastEvent: securityEvent.timestamp,
                recommendedAction: securityEvent.classification.recommendedAction
            });
        }
        
        // Store event for future threshold checking
        this.storeEventForThresholding(alertKey, securityEvent);
    }

    triggerSecurityAlert(alertData) {
        const alert = {
            ...alertData,
            timestamp: new Date().toISOString(),
            status: 'ACTIVE',
            acknowledged: false,
            assignedTo: null,
            actions: []
        };
        
        this.activeAlerts.set(alert.id, alert);
        
        this.log.warn(`SECURITY ALERT: ${alert.type} - ${alert.description}`, {
            alertId: alert.id,
            severity: alert.severity,
            ipAddress: alert.ipAddress,
            eventCount: alert.eventCount
        });
        
        // Send notifications
        this.sendSecurityNotification(alert);
        
        // Emit for real-time systems
        this.emit('securityAlert', alert);
        this.emit('dashboardUpdate', {
            type: 'newAlert',
            data: alert
        });
    }

    executeSecurityActions(securityEvent) {
        const action = securityEvent.classification.recommendedAction;
        
        switch (action) {
            case 'BLOCK_IMMEDIATE':
                this.blockIPAddress(securityEvent.ipAddress, 3600000); // 1 hour
                break;
                
            case 'BLOCK_IP':
                this.blockIPAddress(securityEvent.ipAddress, 1800000); // 30 minutes
                break;
                
            case 'QUARANTINE_USER':
                if (securityEvent.userId) {
                    this.quarantineUser(securityEvent.userId, 7200000); // 2 hours
                }
                break;
                
            case 'BLOCK_REQUEST':
                // Request already blocked by middleware, just log
                this.log.info(`Request blocked for IP: ${securityEvent.ipAddress}`);
                break;
                
            case 'AUDIT_LOG':
                this.createAuditLogEntry(securityEvent);
                break;
                
            default:
                // Just alert/log
                break;
        }
    }

    blockIPAddress(ipAddress, duration) {
        if (!ipAddress || ipAddress === 'unknown') return;
        
        this.securityMetrics.blockedIPs.add(ipAddress);
        
        // Store with expiration
        setTimeout(() => {
            this.securityMetrics.blockedIPs.delete(ipAddress);
            this.log.info(`IP address unblocked: ${ipAddress}`);
        }, duration);
        
        this.log.warn(`IP address blocked: ${ipAddress} for ${duration / 1000 / 60} minutes`);
        
        // Emit for middleware to pick up
        this.emit('ipBlocked', { ipAddress, duration, timestamp: Date.now() });
    }

    quarantineUser(userId, duration) {
        if (!userId) return;
        
        this.securityMetrics.quarantinedUsers.add(userId);
        
        setTimeout(() => {
            this.securityMetrics.quarantinedUsers.delete(userId);
            this.log.info(`User unquarantined: ${userId}`);
        }, duration);
        
        this.log.warn(`User quarantined: ${userId} for ${duration / 1000 / 60} minutes`);
        
        this.emit('userQuarantined', { userId, duration, timestamp: Date.now() });
    }

    logSecurityEvent(securityEvent) {
        // Add to in-memory log
        this.securityLog.unshift(securityEvent);
        
        // Maintain log size
        if (this.securityLog.length > this.maxLogEntries) {
            this.securityLog.splice(this.maxLogEntries);
        }
        
        // Log to file system for persistence (async, non-blocking)
        this.persistSecurityLog(securityEvent).catch(error => {
            this.log.error('Failed to persist security log:', error);
        });
    }

    async persistSecurityLog(securityEvent) {
        const logDir = path.join(__dirname, '../logs/security');
        const logFile = path.join(logDir, `security-${new Date().toISOString().split('T')[0]}.json`);
        
        try {
            // Ensure directory exists
            await fs.mkdir(logDir, { recursive: true });
            
            // Append to daily log file
            const logEntry = `${JSON.stringify(securityEvent)  }\n`;
            await fs.appendFile(logFile, logEntry);
        } catch (error) {
            this.log.error('Error persisting security log:', error);
        }
    }

    sendSecurityNotification(alert) {
        // Implement notification logic (email, Slack, SAP Alert Notification, etc.)
        const notification = {
            type: 'SECURITY_ALERT',
            severity: alert.severity,
            subject: `Security Alert: ${alert.type}`,
            message: `${alert.description}\n\nIP: ${alert.ipAddress}\nTime: ${alert.timestamp}\nEvent Count: ${alert.eventCount}`,
            recipients: this.getAlertRecipients(alert.severity),
            channels: ['email', 'dashboard']
        };
        
        // Emit for notification service
        this.emit('sendNotification', notification);
    }

    getAlertRecipients(severity) {
        const recipients = {
            critical: ['security-team@company.com', 'ciso@company.com'],
            high: ['security-team@company.com', 'devops@company.com'],
            medium: ['security-team@company.com'],
            low: ['security-team@company.com']
        };
        
        return recipients[severity.toLowerCase()] || recipients.low;
    }

    // Utility methods
    getRecentEventsByIP(ipAddress, timeWindow) {
        const cutoff = Date.now() - timeWindow;
        return this.securityLog.filter(event => 
            event.ipAddress === ipAddress && event.epochTime > cutoff
        );
    }

    getRecentEventsByKey(key, timeWindow) {
        const cutoff = Date.now() - timeWindow;
        const events = this.securityEvents.get(key) || [];
        return events.filter(event => event.epochTime > cutoff);
    }

    storeEventForThresholding(key, event) {
        if (!this.securityEvents.has(key)) {
            this.securityEvents.set(key, []);
        }
        
        const events = this.securityEvents.get(key);
        events.push(event);
        
        // Keep only recent events (last hour)
        const cutoff = Date.now() - 3600000;
        const recentEvents = events.filter(e => e.epochTime > cutoff);
        this.securityEvents.set(key, recentEvents);
    }

    performSecurityHealthCheck() {
        const healthMetrics = {
            activeAlerts: this.activeAlerts.size,
            blockedIPs: this.securityMetrics.blockedIPs.size,
            quarantinedUsers: this.securityMetrics.quarantinedUsers.size,
            eventRate: this.calculateEventRate(),
            systemHealth: this.assessSystemHealth(),
            timestamp: new Date().toISOString()
        };
        
        this.emit('healthCheck', healthMetrics);
        
        // Check for system-wide threats
        this.detectSystemWideThreats(healthMetrics);
    }

    calculateEventRate() {
        const last5Minutes = Date.now() - 300000;
        const recentEvents = this.securityLog.filter(event => event.epochTime > last5Minutes);
        return {
            eventsLast5Min: recentEvents.length,
            eventsPerMinute: recentEvents.length / 5,
            criticalEvents: recentEvents.filter(e => e.severity === 'critical').length,
            highEvents: recentEvents.filter(e => e.severity === 'high').length
        };
    }

    assessSystemHealth() {
        const eventRate = this.calculateEventRate();
        let riskLevel = 'LOW';
        
        if (eventRate.criticalEvents > 0) {
            riskLevel = 'CRITICAL';
        } else if (eventRate.highEvents > 3 || eventRate.eventsPerMinute > 10) {
            riskLevel = 'HIGH';
        } else if (eventRate.eventsPerMinute > 5 || this.activeAlerts.size > 5) {
            riskLevel = 'MEDIUM';
        }
        
        return {
            riskLevel,
            activeThreats: this.activeAlerts.size,
            systemLoad: eventRate.eventsPerMinute,
            healthScore: this.calculateHealthScore(riskLevel, eventRate)
        };
    }

    calculateHealthScore(riskLevel, eventRate) {
        let score = 100;
        
        // Deduct based on risk level
        switch (riskLevel) {
            case 'CRITICAL': score -= 50; break;
            case 'HIGH': score -= 30; break;
            case 'MEDIUM': score -= 15; break;
        }
        
        // Deduct based on event rate
        score -= Math.min(eventRate.eventsPerMinute * 2, 30);
        
        // Deduct based on active alerts
        score -= this.activeAlerts.size * 5;
        
        return Math.max(score, 0);
    }

    detectSystemWideThreats(healthMetrics) {
        // Detect coordinated attacks
        if (healthMetrics.eventRate.eventsPerMinute > 20) {
            this.triggerSecurityAlert({
                id: crypto.randomUUID(),
                type: 'COORDINATED_ATTACK',
                severity: 'critical',
                description: `High event rate detected: ${healthMetrics.eventRate.eventsPerMinute} events/min`,
                ipAddress: 'multiple',
                userId: null,
                eventCount: healthMetrics.eventRate.eventsLast5Min,
                timeWindow: 5,
                recommendedAction: 'SYSTEM_LOCKDOWN'
            });
        }
        
        // Detect system overload
        if (this.activeAlerts.size > 10) {
            this.triggerSecurityAlert({
                id: crypto.randomUUID(),
                type: 'SYSTEM_OVERLOAD',
                severity: 'high',
                description: `Excessive security alerts: ${this.activeAlerts.size} active`,
                ipAddress: 'system',
                userId: null,
                eventCount: this.activeAlerts.size,
                recommendedAction: 'INVESTIGATE'
            });
        }
    }

    cleanupExpiredAlerts() {
        const expired = [];
        const now = Date.now();
        const alertTimeout = 24 * 60 * 60 * 1000; // 24 hours
        
        for (const [alertId, alert] of this.activeAlerts.entries()) {
            const alertAge = now - new Date(alert.timestamp).getTime();
            if (alertAge > alertTimeout && alert.status !== 'INVESTIGATING') {
                expired.push(alertId);
            }
        }
        
        expired.forEach(alertId => {
            const alert = this.activeAlerts.get(alertId);
            alert.status = 'EXPIRED';
            this.activeAlerts.delete(alertId);
            this.log.info(`Security alert expired: ${alertId}`);
        });
    }

    aggregateSecurityMetrics() {
        const metrics = {
            ...this.securityMetrics,
            activeAlerts: this.activeAlerts.size,
            recentEventRate: this.calculateEventRate(),
            systemHealth: this.assessSystemHealth(),
            timestamp: new Date().toISOString()
        };
        
        this.emit('metricsUpdate', metrics);
    }

    // Public API methods
    getSecurityDashboard() {
        const recentEvents = this.securityLog.slice(0, 50);
        const alertsSummary = {};
        
        for (const alert of this.activeAlerts.values()) {
            alertsSummary[alert.type] = (alertsSummary[alert.type] || 0) + 1;
        }
        
        return {
            overview: {
                totalEvents: this.securityMetrics.totalEvents,
                activeAlerts: this.activeAlerts.size,
                blockedIPs: this.securityMetrics.blockedIPs.size,
                quarantinedUsers: this.securityMetrics.quarantinedUsers.size,
                lastUpdated: new Date().toISOString()
            },
            recentEvents,
            alertsSummary,
            systemHealth: this.assessSystemHealth(),
            eventRate: this.calculateEventRate(),
            threatIntelligence: {
                knownThreats: this.securityPatterns ? Object.keys(this.securityPatterns).length : 0,
                blockedAttacks: this.securityMetrics.blockedIPs.size,
                riskScore: this.assessSystemHealth().riskLevel
            }
        };
    }

    getActiveAlerts() {
        return Array.from(this.activeAlerts.values());
    }

    acknowledgeAlert(alertId, userId) {
        const alert = this.activeAlerts.get(alertId);
        if (alert) {
            alert.acknowledged = true;
            alert.acknowledgedBy = userId;
            alert.acknowledgedAt = new Date().toISOString();
            alert.status = 'ACKNOWLEDGED';
            
            this.log.info(`Alert acknowledged: ${alertId} by ${userId}`);
            this.emit('alertAcknowledged', { alertId, userId, alert });
            
            return true;
        }
        return false;
    }

    resolveAlert(alertId, userId, resolution) {
        const alert = this.activeAlerts.get(alertId);
        if (alert) {
            alert.status = 'RESOLVED';
            alert.resolvedBy = userId;
            alert.resolvedAt = new Date().toISOString();
            alert.resolution = resolution;
            
            this.activeAlerts.delete(alertId);
            
            this.log.info(`Alert resolved: ${alertId} by ${userId}`);
            this.emit('alertResolved', { alertId, userId, alert, resolution });
            
            return true;
        }
        return false;
    }

    isIPBlocked(ipAddress) {
        return this.securityMetrics.blockedIPs.has(ipAddress);
    }

    isUserQuarantined(userId) {
        return this.securityMetrics.quarantinedUsers.has(userId);
    }

    // Report a security event (external API)
    reportSecurityEvent(eventData) {
        this.emit('securityEvent', eventData);
    }

    // Get security metrics for monitoring
    getSecurityMetrics() {
        return {
            ...this.securityMetrics,
            blockedIPs: Array.from(this.securityMetrics.blockedIPs),
            quarantinedUsers: Array.from(this.securityMetrics.quarantinedUsers),
            activeAlerts: this.activeAlerts.size,
            systemHealth: this.assessSystemHealth()
        };
    }
}

module.exports = SecurityMonitoringService;