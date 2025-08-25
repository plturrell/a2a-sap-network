/**
 * @fileoverview Security Event Logging and Monitoring
 * @description Comprehensive security event tracking and alerting
 * @module security-logger
 */

const cds = require('@sap/cds');
const crypto = require('crypto');

/**
 * Security event types
 */
const SecurityEventTypes = {
    AUTHENTICATION_FAILURE: 'AUTHENTICATION_FAILURE',
    AUTHENTICATION_SUCCESS: 'AUTHENTICATION_SUCCESS',
    AUTHORIZATION_FAILURE: 'AUTHORIZATION_FAILURE',
    RATE_LIMIT_EXCEEDED: 'RATE_LIMIT_EXCEEDED',
    SUSPICIOUS_ACTIVITY: 'SUSPICIOUS_ACTIVITY',
    BLOCKCHAIN_TRANSACTION: 'BLOCKCHAIN_TRANSACTION',
    CONTRACT_INTERACTION: 'CONTRACT_INTERACTION',
    PRIVATE_KEY_ACCESS: 'PRIVATE_KEY_ACCESS',
    CONFIGURATION_CHANGE: 'CONFIGURATION_CHANGE',
    ADMIN_ACTION: 'ADMIN_ACTION',
    SECURITY_VIOLATION: 'SECURITY_VIOLATION',
    DATA_ACCESS: 'DATA_ACCESS',
    ERROR_OCCURRED: 'ERROR_OCCURRED'
};

/**
 * Risk levels for security events
 */
const RiskLevels = {
    INFO: 'INFO',
    LOW: 'LOW',
    MEDIUM: 'MEDIUM',
    HIGH: 'HIGH',
    CRITICAL: 'CRITICAL'
};

/**
 * Security event logger with threat detection
 */
class SecurityLogger {
    constructor() {
        this.eventHistory = [];
        this.suspiciousPatterns = new Map();
        this.userActivities = new Map();
        this.maxHistorySize = 10000;
        this.alertThresholds = {
            failedLogins: 5,
            rateLimitExceeded: 3,
            suspiciousActivities: 2
        };
    }

    /**
     * Log security event
     */
    logSecurityEvent(eventType, riskLevel, details = {}) {
        const event = {
            id: this.generateEventId(),
            type: eventType,
            riskLevel: riskLevel,
            timestamp: new Date().toISOString(),
            details: this.sanitizeDetails(details),
            ipAddress: details.ipAddress || 'unknown',
            userAgent: details.userAgent || 'unknown',
            userId: details.userId || 'anonymous',
            sessionId: details.sessionId || null,
            correlationId: details.correlationId || null
        };

        // Add to history
        this.addToHistory(event);

        // Check for patterns
        this.analyzePatterns(event);

        // Log to appropriate level
        this.writeToLog(event);

        // Check for immediate alerts
        this.checkImmediateAlerts(event);

        return event.id;
    }

    /**
     * Generate unique event ID
     */
    generateEventId() {
        return crypto.randomBytes(16).toString('hex');
    }

    /**
     * Sanitize details to prevent log injection
     */
    sanitizeDetails(details) {
        const sanitized = {};

        for (const [key, value] of Object.entries(details)) {
            // Remove sensitive data
            if (this.isSensitiveField(key)) {
                sanitized[key] = '[REDACTED]';
            } else if (typeof value === 'string') {
                // Sanitize string values
                sanitized[key] = value.replace(/[\r\n\t]/g, ' ').substring(0, 1000);
            } else {
                sanitized[key] = value;
            }
        }

        return sanitized;
    }

    /**
     * Check if field contains sensitive data
     */
    isSensitiveField(fieldName) {
        const sensitiveFields = [
            'password', 'privateKey', 'secret', 'token', 'apiKey',
            'creditCard', 'ssn', 'bankAccount', 'pin'
        ];

        return sensitiveFields.some(field =>
            fieldName.toLowerCase().includes(field.toLowerCase())
        );
    }

    /**
     * Add event to history
     */
    addToHistory(event) {
        this.eventHistory.push(event);

        // Maintain history size limit
        if (this.eventHistory.length > this.maxHistorySize) {
            this.eventHistory.shift();
        }
    }

    /**
     * Analyze patterns for threat detection
     */
    analyzePatterns(event) {
        // Track user activity patterns
        this.trackUserActivity(event);

        // Check for suspicious patterns
        this.detectSuspiciousPatterns(event);

        // Check for brute force attacks
        this.detectBruteForceAttacks(event);

        // Check for unusual access patterns
        this.detectUnusualAccess(event);
    }

    /**
     * Track user activity for behavioral analysis
     */
    trackUserActivity(event) {
        const userId = event.userId;
        if (userId === 'anonymous') return;

        if (!this.userActivities.has(userId)) {
            this.userActivities.set(userId, {
                events: [],
                lastActivity: null,
                riskScore: 0
            });
        }

        const userActivity = this.userActivities.get(userId);
        userActivity.events.push({
            type: event.type,
            timestamp: event.timestamp,
            riskLevel: event.riskLevel
        });

        // Keep only recent events (last 24 hours)
        const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
        userActivity.events = userActivity.events.filter(e =>
            new Date(e.timestamp) > oneDayAgo
        );

        userActivity.lastActivity = event.timestamp;

        // Calculate risk score
        userActivity.riskScore = this.calculateUserRiskScore(userActivity.events);
    }

    /**
     * Calculate user risk score based on recent activities
     */
    calculateUserRiskScore(events) {
        let score = 0;
        const weights = {
            [RiskLevels.INFO]: 0,
            [RiskLevels.LOW]: 1,
            [RiskLevels.MEDIUM]: 3,
            [RiskLevels.HIGH]: 7,
            [RiskLevels.CRITICAL]: 15
        };

        for (const event of events) {
            score += weights[event.riskLevel] || 0;
        }

        return score;
    }

    /**
     * Detect suspicious patterns
     */
    detectSuspiciousPatterns(event) {
        // Rapid successive failed attempts
        if (event.type === SecurityEventTypes.AUTHENTICATION_FAILURE) {
            this.checkRapidFailures(event);
        }

        // Unusual geographic access
        if (event.details.ipAddress) {
            this.checkGeographicAnomalies(event);
        }

        // Time-based anomalies
        this.checkTimeAnomalies(event);
    }

    /**
     * Check for rapid authentication failures
     */
    checkRapidFailures(event) {
        const recentFailures = this.eventHistory.filter(e =>
            e.type === SecurityEventTypes.AUTHENTICATION_FAILURE &&
            e.userId === event.userId &&
            new Date(e.timestamp) > new Date(Date.now() - 5 * 60 * 1000) // Last 5 minutes
        );

        if (recentFailures.length >= this.alertThresholds.failedLogins) {
            this.logSecurityEvent(
                SecurityEventTypes.SUSPICIOUS_ACTIVITY,
                RiskLevels.HIGH,
                {
                    ...event.details,
                    reason: 'Multiple authentication failures detected',
                    failureCount: recentFailures.length,
                    timeWindow: '5 minutes'
                }
            );
        }
    }

    /**
     * Check for geographic anomalies
     */
    checkGeographicAnomalies(event) {
        // This would integrate with IP geolocation services
        // For now, just check for private/local IPs in production
        const ipAddress = event.details.ipAddress;

        if (process.env.NODE_ENV === 'production') {
            if (this.isPrivateIP(ipAddress)) {
                this.logSecurityEvent(
                    SecurityEventTypes.SUSPICIOUS_ACTIVITY,
                    RiskLevels.MEDIUM,
                    {
                        ...event.details,
                        reason: 'Private IP address access in production',
                        ipAddress: ipAddress
                    }
                );
            }
        }
    }

    /**
     * Check if IP address is private/local
     */
    isPrivateIP(ip) {
        const privateRanges = [
            /^10\./,
            /^172\.(1[6-9]|2[0-9]|3[01])\./,
            /^192\.168\./,
            /^127\./,
            /^localhost$/,
            /^::1$/
        ];

        return privateRanges.some(range => range.test(ip));
    }

    /**
     * Check for time-based anomalies
     */
    checkTimeAnomalies(event) {
        const hour = new Date(event.timestamp).getHours();

        // Check for off-hours access to sensitive operations
        if ((hour < 6 || hour > 22) && event.riskLevel === RiskLevels.HIGH) {
            this.logSecurityEvent(
                SecurityEventTypes.SUSPICIOUS_ACTIVITY,
                RiskLevels.MEDIUM,
                {
                    ...event.details,
                    reason: 'High-risk operation during off-hours',
                    hour: hour
                }
            );
        }
    }

    /**
     * Detect brute force attacks
     */
    detectBruteForceAttacks(event) {
        if (event.type !== SecurityEventTypes.AUTHENTICATION_FAILURE) return;

        const ipAddress = event.details.ipAddress;
        const recentAttemptsFromIP = this.eventHistory.filter(e =>
            e.type === SecurityEventTypes.AUTHENTICATION_FAILURE &&
            e.details.ipAddress === ipAddress &&
            new Date(e.timestamp) > new Date(Date.now() - 10 * 60 * 1000) // Last 10 minutes
        );

        if (recentAttemptsFromIP.length >= 10) {
            this.logSecurityEvent(
                SecurityEventTypes.SECURITY_VIOLATION,
                RiskLevels.CRITICAL,
                {
                    ...event.details,
                    reason: 'Potential brute force attack detected',
                    attemptCount: recentAttemptsFromIP.length,
                    timeWindow: '10 minutes',
                    sourceIP: ipAddress
                }
            );
        }
    }

    /**
     * Detect unusual access patterns
     */
    detectUnusualAccess(event) {
        // Check for rapid API calls
        if (event.type === SecurityEventTypes.BLOCKCHAIN_TRANSACTION) {
            this.checkRapidTransactions(event);
        }

        // Check for access to multiple accounts
        this.checkMultiAccountAccess(event);
    }

    /**
     * Check for rapid blockchain transactions
     */
    checkRapidTransactions(event) {
        const recentTransactions = this.eventHistory.filter(e =>
            e.type === SecurityEventTypes.BLOCKCHAIN_TRANSACTION &&
            e.userId === event.userId &&
            new Date(e.timestamp) > new Date(Date.now() - 60 * 1000) // Last minute
        );

        if (recentTransactions.length >= 5) {
            this.logSecurityEvent(
                SecurityEventTypes.SUSPICIOUS_ACTIVITY,
                RiskLevels.MEDIUM,
                {
                    ...event.details,
                    reason: 'Rapid blockchain transactions detected',
                    transactionCount: recentTransactions.length,
                    timeWindow: '1 minute'
                }
            );
        }
    }

    /**
     * Check for multi-account access patterns
     */
    checkMultiAccountAccess(event) {
        if (!event.userId || event.userId === 'anonymous') return;

        const sessionId = event.details.sessionId;
        if (!sessionId) return;

        const uniqueUsers = new Set();
        const sessionEvents = this.eventHistory.filter(e =>
            e.details.sessionId === sessionId &&
            new Date(e.timestamp) > new Date(Date.now() - 30 * 60 * 1000) // Last 30 minutes
        );

        sessionEvents.forEach(e => {
            if (e.userId && e.userId !== 'anonymous') {
                uniqueUsers.add(e.userId);
            }
        });

        if (uniqueUsers.size > 3) {
            this.logSecurityEvent(
                SecurityEventTypes.SUSPICIOUS_ACTIVITY,
                RiskLevels.HIGH,
                {
                    ...event.details,
                    reason: 'Multiple user accounts accessed from same session',
                    userCount: uniqueUsers.size,
                    sessionId: sessionId
                }
            );
        }
    }

    /**
     * Write event to appropriate log level
     */
    writeToLog(event) {
        const logData = {
            eventId: event.id,
            type: event.type,
            riskLevel: event.riskLevel,
            userId: event.userId,
            ipAddress: event.ipAddress,
            details: event.details
        };

        switch (event.riskLevel) {
            case RiskLevels.CRITICAL:
                cds.log('security').error('CRITICAL security event', logData);
                break;
            case RiskLevels.HIGH:
                cds.log('security').error('HIGH risk security event', logData);
                break;
            case RiskLevels.MEDIUM:
                cds.log('security').warn('MEDIUM risk security event', logData);
                break;
            case RiskLevels.LOW:
                cds.log('security').info('LOW risk security event', logData);
                break;
            case RiskLevels.INFO:
                cds.log('security').debug('Security event', logData);
                break;
        }
    }

    /**
     * Check for immediate alert conditions
     */
    checkImmediateAlerts(event) {
        // Critical events trigger immediate alerts
        if (event.riskLevel === RiskLevels.CRITICAL) {
            this.sendImmediateAlert(event);
        }

        // Multiple high-risk events from same user
        const userRiskScore = this.getUserRiskScore(event.userId);
        if (userRiskScore > 20) {
            this.sendRiskAlert(event, userRiskScore);
        }
    }

    /**
     * Send immediate alert for critical events
     */
    async sendImmediateAlert(event) {
        const alertData = {
            type: 'CRITICAL_SECURITY_EVENT',
            eventType: event.type,
            userId: event.userId,
            timestamp: event.timestamp,
            details: event.details,
            eventId: event.id
        };

        cds.log('security-alert').error('CRITICAL security alert', alertData);

        // In production, integrate with alerting systems
        if (process.env.SECURITY_WEBHOOK_URL) {
            try {
                // await blockchainClient.sendMessage(process.env.SECURITY_WEBHOOK_URL, {
                //     method: 'POST',
                //     headers: { 'Content-Type': 'application/json' },
                //     body: JSON.stringify(alertData)
                // });
            } catch (error) {
                cds.log('security-alert').error('Failed to send security alert', error);
            }
        }
    }

    /**
     * Send risk alert for high-risk users
     */
    sendRiskAlert(event, riskScore) {
        const alertData = {
            type: 'HIGH_RISK_USER',
            userId: event.userId,
            riskScore: riskScore,
            timestamp: event.timestamp,
            triggeringEvent: event.type
        };

        cds.log('security-alert').warn('High risk user alert', alertData);
    }

    /**
     * Get user risk score
     */
    getUserRiskScore(userId) {
        if (!this.userActivities.has(userId)) return 0;
        return this.userActivities.get(userId).riskScore;
    }

    /**
     * Get security dashboard data
     */
    getSecurityDashboard() {
        const now = new Date();
        const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);

        const recentEvents = this.eventHistory.filter(e =>
            new Date(e.timestamp) > oneDayAgo
        );

        const dashboard = {
            summary: {
                totalEvents: recentEvents.length,
                criticalEvents: recentEvents.filter(e => e.riskLevel === RiskLevels.CRITICAL).length,
                highRiskEvents: recentEvents.filter(e => e.riskLevel === RiskLevels.HIGH).length,
                suspiciousActivities: recentEvents.filter(e => e.type === SecurityEventTypes.SUSPICIOUS_ACTIVITY).length
            },
            eventsByType: {},
            eventsByRisk: {},
            topRiskUsers: this.getTopRiskUsers(),
            recentCritical: recentEvents.filter(e => e.riskLevel === RiskLevels.CRITICAL).slice(-5)
        };

        // Count by type and risk
        for (const event of recentEvents) {
            dashboard.eventsByType[event.type] = (dashboard.eventsByType[event.type] || 0) + 1;
            dashboard.eventsByRisk[event.riskLevel] = (dashboard.eventsByRisk[event.riskLevel] || 0) + 1;
        }

        return dashboard;
    }

    /**
     * Get top risk users
     */
    getTopRiskUsers() {
        const users = Array.from(this.userActivities.entries())
            .map(([userId, activity]) => ({
                userId,
                riskScore: activity.riskScore,
                eventCount: activity.events.length,
                lastActivity: activity.lastActivity
            }))
            .sort((a, b) => b.riskScore - a.riskScore)
            .slice(0, 10);

        return users;
    }
}

// Global security logger instance
const globalSecurityLogger = new SecurityLogger();

module.exports = {
    SecurityLogger,
    SecurityEventTypes,
    RiskLevels,
    logSecurityEvent: (type, risk, details) => globalSecurityLogger.logSecurityEvent(type, risk, details),
    getSecurityDashboard: () => globalSecurityLogger.getSecurityDashboard(),
    getUserRiskScore: (userId) => globalSecurityLogger.getUserRiskScore(userId)
};