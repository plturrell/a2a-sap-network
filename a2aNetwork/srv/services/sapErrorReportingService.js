/**
 * Error Reporting Service
 * 
 * Centralized error reporting and monitoring service that integrates with
 * SAP Cloud ALM and other monitoring solutions for enterprise error tracking.
 * 
 * @author SAP SE
 * @since 1.0.0
 * @version 1.0.0
 */

const { v4: uuidv4 } = require('uuid');
const loggingService = require('./sapLoggingService');

class ErrorReportingService {
    constructor() {
        this.errorStore = new Map();
        this.errorStats = {
            total: 0,
            byCategory: {\n        this.intervals = new Map(); // Track intervals for cleanup},
            byComponent: {},
            byUser: {},
            bySeverity: {
                critical: 0,
                high: 0,
                medium: 0,
                low: 0
            },
            recent: []
        };
        
        this.alertThresholds = {
            critical: 1, // Alert immediately on critical errors
            high: 5,     // Alert after 5 high severity errors in 10 minutes
            medium: 10,  // Alert after 10 medium severity errors in 30 minutes
            errorRate: 0.05 // Alert if error rate exceeds 5%
        };
        
        this.logger = loggingService.child('error-reporting');
        
        // Clean up old errors every hour
        this.intervals.set('interval_42', (function(intervalId) { this.intervals.add(intervalId); return intervalId; }).call(this, setInterval(() => this._cleanupOldErrors(), 3600000));
    }

    /**
     * Report an error to the centralized error tracking system
     * @param {Error|string} error Error object or message
     * @param {object} context Error context information
     * @returns {string} Error ID for tracking
     */
    reportError(error, context = {}) {
        const errorId = uuidv4();
        const timestamp = new Date().toISOString();
        const severity = this._determineSeverity(error, context);
        
        // Normalize error information
        const errorInfo = {
            id: errorId,
            timestamp,
            message: typeof error === 'string' ? error : error.message,
            stack: error.stack || 'No stack trace available',
            type: error.name || 'Error',
            severity,
            category: context.category || this._categorizeError(error),
            component: context.component || 'unknown',
            correlationId: context.correlationId || 'no-correlation',
            sessionId: context.sessionId || 'no-session',
            userId: context.userId || 'anonymous',
            userAgent: context.userAgent,
            url: context.url,
            method: context.method,
            statusCode: context.statusCode,
            environment: process.env.NODE_ENV || 'development',
            version: process.env.npm_package_version || '1.0.0',
            hostname: require('os').hostname(),
            pid: process.pid,
            metadata: context.metadata || {},
            tags: context.tags || [],
            fingerprint: this._generateFingerprint(error),
            count: 1,
            firstSeen: timestamp,
            lastSeen: timestamp
        };

        // Check if this is a duplicate error
        const existingError = this._findSimilarError(errorInfo);
        if (existingError) {
            existingError.count++;
            existingError.lastSeen = timestamp;
            errorInfo.id = existingError.id;
        } else {
            this.errorStore.set(errorId, errorInfo);
        }

        // Update statistics
        this._updateStats(errorInfo);
        
        // Log the error
        this.logger.error(`Error reported: ${errorInfo.message}`, {
            errorId: errorInfo.id,
            severity: errorInfo.severity,
            category: errorInfo.category,
            component: errorInfo.component,
            correlationId: errorInfo.correlationId,
            fingerprint: errorInfo.fingerprint,
            count: errorInfo.count,
            stack: errorInfo.stack,
            metadata: errorInfo.metadata
        });

        // Check if we need to send alerts
        this._checkAlertThresholds(errorInfo);

        // Send to external monitoring if configured
        this._sendToExternalMonitoring(errorInfo);

        return errorInfo.id;
    }

    /**
     * Report a client-side error
     * @param {object} errorData Client error data
     * @returns {string} Error ID
     */
    reportClientError(errorData) {
        const context = {
            category: 'client',
            component: 'ui5-frontend',
            url: errorData.url,
            userAgent: errorData.userAgent,
            correlationId: errorData.correlationId,
            sessionId: errorData.sessionId,
            userId: errorData.userId,
            metadata: {
                lineNumber: errorData.lineno,
                columnNumber: errorData.colno,
                filename: errorData.filename,
                browser: this._parseBrowser(errorData.userAgent),
                timestamp: errorData.timestamp,
                additionalInfo: errorData.additionalInfo
            },
            tags: ['client-side', 'javascript']
        };

        const error = new Error(errorData.message);
        error.stack = errorData.stack || `at ${errorData.filename}:${errorData.lineno}:${errorData.colno}`;
        
        return this.reportError(error, context);
    }

    /**
     * Get error by ID
     * @param {string} errorId Error ID
     * @returns {object|null} Error information
     */
    getError(errorId) {
        return this.errorStore.get(errorId) || null;
    }

    /**
     * Get recent errors
     * @param {number} limit Number of errors to return
     * @param {object} filters Filters to apply
     * @returns {Array} Recent errors
     */
    getRecentErrors(limit = 50, filters = {}) {
        let errors = Array.from(this.errorStore.values());
        
        // Apply filters
        if (filters.severity) {
            errors = errors.filter(err => err.severity === filters.severity);
        }
        
        if (filters.category) {
            errors = errors.filter(err => err.category === filters.category);
        }
        
        if (filters.component) {
            errors = errors.filter(err => err.component === filters.component);
        }
        
        if (filters.correlationId) {
            errors = errors.filter(err => err.correlationId === filters.correlationId);
        }
        
        if (filters.since) {
            const sinceDate = new Date(filters.since);
            errors = errors.filter(err => new Date(err.lastSeen) > sinceDate);
        }

        // Sort by last seen (most recent first)
        errors.sort((a, b) => new Date(b.lastSeen) - new Date(a.lastSeen));
        
        return errors.slice(0, limit);
    }

    /**
     * Get error statistics
     * @param {string} timeframe Timeframe for stats ('1h', '24h', '7d', '30d')
     * @returns {object} Error statistics
     */
    getErrorStats(timeframe = '24h') {
        const now = Date.now();
        const timeframes = {
            '1h': 60 * 60 * 1000,
            '24h': 24 * 60 * 60 * 1000,
            '7d': 7 * 24 * 60 * 60 * 1000,
            '30d': 30 * 24 * 60 * 60 * 1000
        };
        
        const cutoff = now - (timeframes[timeframe] || timeframes['24h']);
        const recentErrors = Array.from(this.errorStore.values())
            .filter(err => new Date(err.lastSeen).getTime() > cutoff);

        const stats = {
            timeframe,
            total: recentErrors.length,
            unique: recentErrors.length, // Already deduplicated
            bySeverity: { critical: 0, high: 0, medium: 0, low: 0 },
            byCategory: {},
            byComponent: {},
            topErrors: {},
            errorRate: 0,
            trends: this._calculateTrends(recentErrors, timeframe)
        };

        recentErrors.forEach(error => {
            stats.bySeverity[error.severity]++;
            stats.byCategory[error.category] = (stats.byCategory[error.category] || 0) + 1;
            stats.byComponent[error.component] = (stats.byComponent[error.component] || 0) + 1;
            stats.topErrors[error.fingerprint] = {
                message: error.message,
                count: error.count,
                severity: error.severity,
                category: error.category,
                component: error.component
            };
        });

        return stats;
    }

    /**
     * Mark error as resolved
     * @param {string} errorId Error ID
     * @param {string} userId User who resolved the error
     * @param {string} resolution Resolution notes
     */
    resolveError(errorId, userId, resolution) {
        const error = this.errorStore.get(errorId);
        if (error) {
            error.status = 'resolved';
            error.resolvedBy = userId;
            error.resolvedAt = new Date().toISOString();
            error.resolution = resolution;
            
            this.logger.info(`Error resolved: ${errorId}`, {
                errorId,
                resolvedBy: userId,
                resolution
            });
        }
    }

    /**
     * Express middleware for automatic error reporting
     * @returns {Function} Express error middleware
     */
    middleware() {
        return (error, req, res, next) => {
            const context = {
                category: 'server',
                component: 'express',
                correlationId: req.correlationId,
                sessionId: req.sessionID,
                userId: req.user?.id || req.user?.email,
                userAgent: req.get('User-Agent'),
                url: req.originalUrl || req.url,
                method: req.method,
                statusCode: error.statusCode || error.status || 500,
                metadata: {
                    headers: req.headers,
                    body: req.body,
                    params: req.params,
                    query: req.query
                }
            };

            this.reportError(error, context);
            next(error);
        };
    }

    // Private methods

    _determineSeverity(error, context) {
        // Determine severity based on error type and context
        if (context.severity) {
            return context.severity;
        }

        if (context.statusCode >= 500 || error.name === 'FatalError') {
            return 'critical';
        }

        if (context.statusCode >= 400 || error.name === 'ValidationError') {
            return 'high';
        }

        if (error.name === 'DeprecationWarning' || context.category === 'performance') {
            return 'medium';
        }

        return 'low';
    }

    _categorizeError(error) {
        const message = error.message?.toLowerCase() || '';
        const type = error.name?.toLowerCase() || '';

        if (type.includes('security') || message.includes('unauthorized')) {
            return 'security';
        }

        if (type.includes('validation') || message.includes('invalid')) {
            return 'validation';
        }

        if (message.includes('database') || message.includes('connection')) {
            return 'database';
        }

        if (message.includes('network') || message.includes('timeout')) {
            return 'network';
        }

        if (message.includes('blockchain') || message.includes('contract')) {
            return 'blockchain';
        }

        return 'application';
    }

    _generateFingerprint(error) {
        // Create a unique fingerprint for grouping similar errors
        const message = error.message || '';
        const type = error.name || 'Error';
        const stackTop = error.stack ? error.stack.split('\n')[1] : '';
        
        return require('crypto')
            .createHash('md5')
            .update(type + message + stackTop)
            .digest('hex');
    }

    _findSimilarError(errorInfo) {
        // Find existing error with the same fingerprint
        for (const [id, existingError] of this.errorStore) {
            if (existingError.fingerprint === errorInfo.fingerprint) {
                return existingError;
            }
        }
        return null;
    }

    _updateStats(errorInfo) {
        this.errorStats.total++;
        this.errorStats.bySeverity[errorInfo.severity]++;
        
        this.errorStats.byCategory[errorInfo.category] = 
            (this.errorStats.byCategory[errorInfo.category] || 0) + 1;
            
        this.errorStats.byComponent[errorInfo.component] = 
            (this.errorStats.byComponent[errorInfo.component] || 0) + 1;

        // Keep recent errors for rate calculation
        this.errorStats.recent.push({
            timestamp: Date.now(),
            severity: errorInfo.severity
        });

        // Keep only last 1000 recent errors
        if (this.errorStats.recent.length > 1000) {
            this.errorStats.recent.shift();
        }
    }

    _checkAlertThresholds(errorInfo) {
        // Check if we need to trigger alerts
        if (errorInfo.severity === 'critical') {
            this._sendAlert('critical', errorInfo);
        }

        // Check error rate
        const recentWindow = Date.now() - 600000; // 10 minutes
        const recentErrors = this.errorStats.recent
            .filter(err => err.timestamp > recentWindow);
            
        if (recentErrors.length > 50) { // If more than 50 errors in 10 minutes
            this._sendAlert('high-error-rate', {
                count: recentErrors.length,
                timeWindow: '10 minutes'
            });
        }
    }

    _sendAlert(type, data) {
        this.logger.error(`ALERT: ${type}`, {
            alertType: type,
            alertData: data,
            category: 'alert'
        });

        // In production, integrate with alerting systems like:
        // - SAP Alert Notification
        // - PagerDuty
        // - Slack
        // - Email notifications
    }

    _sendToExternalMonitoring(errorInfo) {
        // In production, send to external monitoring services like:
        // - SAP Cloud ALM
        // - Sentry
        // - New Relic
        // - Datadog
        
        if (process.env.SENTRY_DSN) {
            // Example Sentry integration
            // Sentry.captureException(error, { extra: errorInfo });
        }
    }

    _calculateTrends(errors, timeframe) {
        // Calculate error trends over time
        const intervals = this._getTimeIntervals(timeframe);
        const trends = intervals.map(interval => ({
            period: interval.label,
            count: errors.filter(err => 
                new Date(err.lastSeen).getTime() >= interval.start && 
                new Date(err.lastSeen).getTime() < interval.end
            ).length
        }));

        return trends;
    }

    _getTimeIntervals(timeframe) {
        const now = Date.now();
        const intervals = [];
        
        if (timeframe === '1h') {
            // 12 intervals of 5 minutes each
            for (let i = 11; i >= 0; i--) {
                const start = now - (i + 1) * 5 * 60 * 1000;
                const end = now - i * 5 * 60 * 1000;
                intervals.push({
                    label: new Date(start).toLocaleTimeString(),
                    start,
                    end
                });
            }
        } else if (timeframe === '24h') {
            // 24 intervals of 1 hour each
            for (let i = 23; i >= 0; i--) {
                const start = now - (i + 1) * 60 * 60 * 1000;
                const end = now - i * 60 * 60 * 1000;
                intervals.push({
                    label: new Date(start).getHours() + ':00',
                    start,
                    end
                });
            }
        }
        
        return intervals;
    }

    _parseBrowser(userAgent) {
        if (!userAgent) return 'Unknown';
        
        if (userAgent.includes('Chrome')) return 'Chrome';
        if (userAgent.includes('Firefox')) return 'Firefox';
        if (userAgent.includes('Safari')) return 'Safari';
        if (userAgent.includes('Edge')) return 'Edge';
        
        return 'Other';
    }

    _cleanupOldErrors() {
        const cutoff = Date.now() - (7 * 24 * 60 * 60 * 1000); // 7 days
        
        for (const [id, error] of this.errorStore) {
            if (new Date(error.lastSeen).getTime() < cutoff) {
                this.errorStore.delete(id);
            }
        }
        
        this.logger.info('Cleaned up old errors', {
            category: 'maintenance',
            remainingErrors: this.errorStore.size
        });
    }
}

module.exports = new ErrorReportingService();