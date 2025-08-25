/**
 * A2A Marketplace Logging Service
 * Production-grade logging following A2A system standards
 * Integrates with SAP Cloud Logging and OpenTelemetry
 */

const loggingService = require('./sapLoggingService');
const { v4: uuid } = require('uuid');

/**
 * Marketplace-specific logger with standardized categories and contexts
 */
class MarketplaceLogger {
    constructor() {
        // Create child logger for marketplace services
        this.logger = loggingService.child('marketplace');
        
        // Category definitions following A2A standards
        this.categories = {
            SYSTEM: 'system',
            SECURITY: 'security', 
            PERFORMANCE: 'performance',
            BUSINESS: 'business',
            INTEGRATION: 'integration',
            API: 'api',
            DATABASE: 'database',
            AUDIT: 'audit'
        };

        // Operation types for structured logging
        this.operations = {
            SERVICE_REQUEST: 'service_request',
            DATA_PRODUCT_PURCHASE: 'data_product_purchase',
            AGENT_REGISTRATION: 'agent_registration',
            MARKETPLACE_SEARCH: 'marketplace_search',
            RECOMMENDATION_GENERATION: 'recommendation_generation',
            INTEGRATION_CREATION: 'integration_creation',
            PAYMENT_PROCESSING: 'payment_processing',
            REVIEW_SUBMISSION: 'review_submission'
        };
    }

    /**
     * Initialize marketplace logging context for a request
     */
    initializeRequestContext(req) {
        const correlationId = req.headers['x-correlation-id'] || uuid();
        const requestId = uuid();
        
        req.correlationId = correlationId;
        req.requestId = requestId;
        
        // Set context for all subsequent logs
        this.requestContext = {
            correlationId,
            requestId,
            userId: req.user?.id,
            userAgent: req.headers['user-agent'],
            clientIp: req.ip || req.connection.remoteAddress,
            endpoint: req.path,
            method: req.method
        };

        this.info('Request initiated', {
            ...this.requestContext,
            category: this.categories.API,
            operation: 'request_start'
        });

        return { correlationId, requestId };
    }

    /**
     * Log business operations with standardized structure
     */
    logBusinessOperation(operation, status, metadata = {}) {
        const logData = {
            ...this.requestContext,
            category: this.categories.BUSINESS,
            operation,
            status, // 'started', 'completed', 'failed'
            ...metadata
        };

        const message = `Business operation ${operation} ${status}`;
        
        switch (status) {
        case 'started':
            this.info(message, logData);
            break;
        case 'completed':
            this.info(message, { ...logData, success: true });
            break;
        case 'failed':
            this.error(message, { ...logData, success: false });
            break;
        default:
            this.info(message, logData);
        }
    }

    /**
     * Log service requests with full context
     */
    logServiceRequest(action, serviceData, metadata = {}) {
        this.logBusinessOperation(this.operations.SERVICE_REQUEST, action, {
            serviceId: serviceData.serviceId,
            serviceName: serviceData.serviceName,
            providerId: serviceData.providerId,
            requesterId: serviceData.requesterId,
            amount: serviceData.amount,
            currency: serviceData.currency,
            ...metadata
        });
    }

    /**
     * Log data product purchases
     */
    logDataProductPurchase(action, purchaseData, metadata = {}) {
        this.logBusinessOperation(this.operations.DATA_PRODUCT_PURCHASE, action, {
            dataProductId: purchaseData.dataProductId,
            productName: purchaseData.productName,
            purchaserId: purchaseData.purchaserId,
            licenseType: purchaseData.licenseType,
            amount: purchaseData.amount,
            currency: purchaseData.currency,
            ...metadata
        });
    }

    /**
     * Log agent registrations and management
     */
    logAgentOperation(action, agentData, metadata = {}) {
        this.logBusinessOperation(this.operations.AGENT_REGISTRATION, action, {
            agentId: agentData.agentId,
            agentName: agentData.agentName,
            ownerId: agentData.ownerId,
            category: agentData.category,
            endpoint: agentData.endpoint,
            ...metadata
        });
    }

    /**
     * Log marketplace search operations
     */
    logSearchOperation(searchData, results, metadata = {}) {
        this.logBusinessOperation(this.operations.MARKETPLACE_SEARCH, 'completed', {
            query: searchData.query,
            searchType: searchData.searchType,
            filters: searchData.filters,
            resultCount: results.totalCount,
            searchTimeMs: results.searchTimeMs,
            ...metadata
        });
    }

    /**
     * Log recommendation generation
     */
    logRecommendation(userId, recommendations, metadata = {}) {
        this.logBusinessOperation(this.operations.RECOMMENDATION_GENERATION, 'completed', {
            targetUserId: userId,
            recommendationCount: recommendations.length,
            averageScore: recommendations.length > 0 ? 
                (recommendations.reduce((sum, r) => sum + r.matchScore, 0) / recommendations.length).toFixed(2) : 0,
            types: [...new Set(recommendations.map(r => r.itemType))],
            ...metadata
        });
    }

    /**
     * Log performance metrics
     */
    logPerformance(operation, durationMs, metadata = {}) {
        const performanceData = {
            ...this.requestContext,
            category: this.categories.PERFORMANCE,
            operation,
            durationMs,
            performance: {
                duration: durationMs,
                ...metadata
            }
        };

        if (durationMs > 5000) { // Slow operation threshold
            this.warn(`Slow operation: ${operation} took ${durationMs}ms`, performanceData);
        } else if (durationMs > 1000) {
            this.info(`Operation performance: ${operation} took ${durationMs}ms`, performanceData);
        } else {
            this.debug(`Operation performance: ${operation} took ${durationMs}ms`, performanceData);
        }
    }

    /**
     * Log security events
     */
    logSecurityEvent(eventType, severity, details = {}) {
        const securityData = {
            ...this.requestContext,
            category: this.categories.SECURITY,
            eventType,
            severity, // 'low', 'medium', 'high', 'critical'
            timestamp: new Date().toISOString(),
            ...details
        };

        const message = `Security event: ${eventType}`;
        
        switch (severity) {
        case 'critical':
            this.error(message, { ...securityData, alert: true });
            break;
        case 'high':
            this.error(message, securityData);
            break;
        case 'medium':
            this.warn(message, securityData);
            break;
        default:
            this.info(message, securityData);
        }
    }

    /**
     * Log authorization events
     */
    logAuthorization(action, resource, result, metadata = {}) {
        this.logSecurityEvent('authorization_check', result ? 'low' : 'medium', {
            action,
            resource,
            authorized: result,
            ...metadata
        });
    }

    /**
     * Log database operations
     */
    logDatabaseOperation(operation, table, durationMs, metadata = {}) {
        const dbData = {
            ...this.requestContext,
            category: this.categories.DATABASE,
            operation,
            table,
            durationMs,
            ...metadata
        };

        if (durationMs > 2000) {
            this.warn(`Slow database query: ${operation} on ${table} took ${durationMs}ms`, dbData);
        } else {
            this.debug(`Database operation: ${operation} on ${table}`, dbData);
        }
    }

    /**
     * Log integration events
     */
    logIntegration(service, operation, result, metadata = {}) {
        this.logger.info(`Integration: ${service} ${operation}`, {
            ...this.requestContext,
            category: this.categories.INTEGRATION,
            service,
            operation,
            success: result.success,
            responseTime: result.responseTime,
            ...metadata
        });
    }

    /**
     * Log audit events for compliance
     */
    logAudit(eventType, entityType, entityId, changes, metadata = {}) {
        const auditData = {
            ...this.requestContext,
            category: this.categories.AUDIT,
            eventType, // 'CREATE', 'UPDATE', 'DELETE', 'ACCESS'
            entityType, // 'SERVICE', 'DATA_PRODUCT', 'AGENT', etc.
            entityId,
            changes,
            timestamp: new Date().toISOString(),
            ...metadata
        };

        this.info(`Audit: ${eventType} ${entityType} ${entityId}`, auditData);
    }

    /**
     * Log payment processing events
     */
    logPayment(action, paymentData, metadata = {}) {
        // Sanitize sensitive payment data
        const sanitizedData = {
            paymentId: paymentData.paymentId,
            amount: paymentData.amount,
            currency: paymentData.currency,
            status: paymentData.status,
            // Never log full card numbers or sensitive data
            cardLast4: paymentData.cardLast4,
            paymentMethod: paymentData.paymentMethod
        };

        this.logBusinessOperation(this.operations.PAYMENT_PROCESSING, action, {
            ...sanitizedData,
            ...metadata
        });
    }

    /**
     * Log marketplace analytics events
     */
    logAnalytics(metricType, value, dimensions = {}) {
        this.logger.info(`Analytics metric: ${metricType}`, {
            ...this.requestContext,
            category: this.categories.BUSINESS,
            operation: 'analytics',
            metricType,
            value,
            dimensions
        });
    }

    /**
     * Log error with full context and stack trace
     */
    logError(error, context = {}) {
        const errorData = {
            ...this.requestContext,
            category: this.categories.SYSTEM,
            error: {
                name: error.name,
                message: error.message,
                stack: error.stack,
                code: error.code
            },
            ...context
        };

        this.error(`Application error: ${error.message}`, errorData);
    }

    /**
     * Standard logging methods that delegate to base logger
     */
    debug(message, meta = {}) {
        this.logger.debug(message, { ...this.requestContext, ...meta });
    }

    info(message, meta = {}) {
        this.logger.info(message, { ...this.requestContext, ...meta });
    }

    warn(message, meta = {}) {
        this.logger.warn(message, { ...this.requestContext, ...meta });
    }

    error(message, meta = {}) {
        this.logger.error(message, { ...this.requestContext, ...meta });
    }

    /**
     * Create child logger for specific components
     */
    child(component) {
        const childLogger = new MarketplaceLogger();
        childLogger.logger = this.logger.child(component);
        childLogger.requestContext = this.requestContext;
        return childLogger;
    }

    /**
     * Middleware for automatic request/response logging
     */
    middleware() {
        return (req, res, next) => {
            const startTime = Date.now();
            const { correlationId, requestId } = this.initializeRequestContext(req);
            
            // Add logger to request for use in routes
            req.marketplaceLogger = this;

            // Log response when finished
            res.on('finish', () => {
                const duration = Date.now() - startTime;
                
                this.info('Request completed', {
                    correlationId,
                    requestId,
                    statusCode: res.statusCode,
                    duration,
                    category: this.categories.API,
                    operation: 'request_complete'
                });

                // Log performance if slow
                if (duration > 1000) {
                    this.logPerformance('http_request', duration, {
                        endpoint: req.path,
                        method: req.method,
                        statusCode: res.statusCode
                    });
                }
            });

            next();
        };
    }
}

// Export singleton instance
const marketplaceLogger = new MarketplaceLogger();

module.exports = {
    MarketplaceLogger,
    marketplaceLogger,
    middleware: marketplaceLogger.middleware.bind(marketplaceLogger)
};