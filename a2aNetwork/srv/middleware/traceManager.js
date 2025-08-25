/**
 * @fileoverview Comprehensive Trace Management System
 * @description Provides end-to-end request tracing from frontend to database
 * @module traceManager
 * @since 1.0.0
 * @author A2A Network Team
 */

const cds = require('@sap/cds');
const { v4: uuidv4 } = require('uuid');

// OpenTelemetry integration for enterprise-grade distributed tracing
let opentelemetry, trace, SpanStatusCode, SpanKind;
try {
    opentelemetry = require('@opentelemetry/api');
    trace = opentelemetry.trace;
    SpanStatusCode = opentelemetry.SpanStatusCode;
    SpanKind = opentelemetry.SpanKind;
} catch (error) {
    // OpenTelemetry not available - fall back to custom tracing
    cds.log('trace-manager').warn('OpenTelemetry not available, using custom tracing only');
}

/**
 * TraceManager - Handles end-to-end request tracing
 */
class TraceManager {
    constructor() {
        this.activeTraces = new Map();
        this.traceStore = new Map();
        this.maxTraceHistory = 10000;
        this.traceRetentionMs = 24 * 60 * 60 * 1000; // 24 hours

        // OpenTelemetry tracer
        this.tracer = trace ? trace.getTracer('a2a-network', '1.0.0') : null;
        this.spans = new Map(); // Track OpenTelemetry spans

        // Performance metrics
        this.metrics = {
            totalRequests: 0,
            activeTraces: 0,
            averageResponseTime: 0,
            errorRate: 0
        };

        this.intervals = new Map(); // Track intervals for cleanup

        // Start cleanup interval
        this.startCleanupInterval();
    }

    /**
     * Initialize a new trace for incoming request
     */
    initializeTrace(req, res, next) {
        const traceId = uuidv4();
        const timestamp = new Date();

        // Update metrics
        this.metrics.totalRequests++;
        this.metrics.activeTraces = this.activeTraces.size + 1;

        // Extract user context
        const userContext = this.extractUserContext(req);

        // Extract frontend context
        const frontendContext = this.extractFrontendContext(req);

        // Create OpenTelemetry span if available
        let span = null;
        if (this.tracer) {
            span = this.tracer.startSpan(`${req.method} ${req.path}`, {
                kind: SpanKind.SERVER,
                attributes: {
                    'http.method': req.method,
                    'http.url': req.originalUrl,
                    'http.route': req.path,
                    'http.user_agent': req.get('User-Agent'),
                    'user.id': userContext.userId,
                    'user.tenant': userContext.tenant,
                    'ui.component': frontendContext.component,
                    'ui.view': frontendContext.view,
                    'request.id': traceId
                }
            });
            this.spans.set(traceId, span);
        }

        const trace = {
            traceId,
            timestamp,
            userContext,
            frontendContext,
            requestInfo: {
                method: req.method,
                url: req.originalUrl,
                path: req.path,
                query: req.query,
                headers: this.sanitizeHeaders(req.headers),
                userAgent: req.get('User-Agent'),
                ip: req.ip || req.connection.remoteAddress,
                sessionId: req.sessionID
            },
            steps: [],
            errors: [],
            performance: {
                startTime: process.hrtime(),
                milestones: []
            },
            status: 'active',
            span: span // Reference to OpenTelemetry span
        };

        // Store trace
        this.activeTraces.set(traceId, trace);

        // Attach to request/response objects
        req.traceId = traceId;
        req.trace = trace;
        res.traceId = traceId;

        // Add trace headers to response
        res.setHeader('X-Trace-ID', traceId);
        res.setHeader('X-Request-ID', traceId);

        // Log trace initialization
        const log = cds.log('trace-manager');
        log.info('Trace initialized', {
            traceId,
            method: req.method,
            url: req.originalUrl,
            user: userContext.userId,
            component: frontendContext.component,
            hasOpenTelemetry: !!span
        });

        next();
    }

    /**
     * Add a step to the current trace
     */
    addStep(traceId, stepInfo) {
        const trace = this.activeTraces.get(traceId);
        if (!trace) return;

        const step = {
            timestamp: new Date(),
            stepId: uuidv4(),
            ...stepInfo,
            duration: this.calculateStepDuration(trace.performance.startTime)
        };

        trace.steps.push(step);

        const log = cds.log('trace-manager');
        log.debug('Trace step added', {
            traceId,
            step: step.name,
            component: step.component,
            duration: step.duration
        });
    }

    /**
     * Add an error to the current trace
     */
    addError(traceId, error, context = {}) {
        const trace = this.activeTraces.get(traceId);
        if (!trace) return;

        const errorInfo = {
            timestamp: new Date(),
            errorId: uuidv4(),
            message: error.message,
            stack: error.stack,
            code: error.code,
            statusCode: error.statusCode || 500,
            component: context.component || 'unknown',
            layer: context.layer || 'unknown',
            operation: context.operation || 'unknown',
            data: context.data || {},
            severity: context.severity || 'error'
        };

        trace.errors.push(errorInfo);
        trace.status = 'error';

        // Update OpenTelemetry span if available
        const span = this.spans.get(traceId);
        if (span) {
            span.recordException(error);
            span.setStatus({
                code: SpanStatusCode.ERROR,
                message: error.message
            });
            span.setAttributes({
                'error.type': error.name || 'Error',
                'error.message': error.message,
                'error.component': errorInfo.component,
                'error.layer': errorInfo.layer,
                'error.operation': errorInfo.operation,
                'error.severity': errorInfo.severity
            });
        }

        // Update error rate metrics
        this.updateErrorRate();

        const log = cds.log('trace-manager');
        log.error('Error added to trace', {
            traceId,
            errorId: errorInfo.errorId,
            component: errorInfo.component,
            layer: errorInfo.layer,
            operation: errorInfo.operation,
            message: errorInfo.message
        });

        return errorInfo.errorId;
    }

    /**
     * Complete a trace
     */
    completeTrace(traceId, statusCode = 200) {
        const trace = this.activeTraces.get(traceId);
        if (!trace) return;

        const totalDuration = this.calculateStepDuration(trace.performance.startTime);

        trace.performance.totalDuration = totalDuration;
        trace.performance.endTime = process.hrtime();
        trace.status = statusCode >= 400 ? 'error' : 'completed';
        trace.statusCode = statusCode;
        trace.completedAt = new Date();

        // Complete OpenTelemetry span if available
        const span = this.spans.get(traceId);
        if (span) {
            span.setAttributes({
                'http.status_code': statusCode,
                'http.response.duration': totalDuration,
                'trace.steps': trace.steps.length,
                'trace.errors': trace.errors.length
            });

            if (statusCode >= 400) {
                span.setStatus({
                    code: SpanStatusCode.ERROR,
                    message: `HTTP ${statusCode}`
                });
            } else {
                span.setStatus({ code: SpanStatusCode.OK });
            }

            span.end();
            this.spans.delete(traceId);
        }

        // Update metrics
        this.metrics.activeTraces = this.activeTraces.size - 1;
        this.updateAverageResponseTime(totalDuration);

        // Move to permanent storage
        this.traceStore.set(traceId, trace);
        this.activeTraces.delete(traceId);

        const log = cds.log('trace-manager');
        log.info('Trace completed', {
            traceId,
            status: trace.status,
            statusCode,
            duration: totalDuration,
            steps: trace.steps.length,
            errors: trace.errors.length
        });

        return trace;
    }

    /**
     * Extract user context from request
     */
    extractUserContext(req) {
        try {
            return {
                userId: req.user?.id || req.headers['x-user-id'] || 'anonymous',
                userName: req.user?.name || req.headers['x-user-name'] || 'anonymous',
                userEmail: req.user?.email || req.headers['x-user-email'] || 'unknown',
                roles: req.user?.roles || [],
                tenant: req.user?.tenant || req.headers['x-tenant'] || 'default',
                authMethod: req.user?.authMethod || 'unknown'
            };
        } catch (error) {
            return {
                userId: 'anonymous',
                userName: 'anonymous',
                userEmail: 'unknown',
                roles: [],
                tenant: 'default',
                authMethod: 'unknown'
            };
        }
    }

    /**
     * Extract frontend context from request
     */
    extractFrontendContext(req) {
        return {
            component: req.headers['x-ui-component'] || 'unknown',
            view: req.headers['x-ui-view'] || 'unknown',
            controller: req.headers['x-ui-controller'] || 'unknown',
            action: req.headers['x-ui-action'] || 'unknown',
            sapClient: req.headers['sap-client'] || 'unknown',
            sapLanguage: req.headers['sap-language'] || 'en',
            ui5Version: req.headers['x-ui5-version'] || 'unknown',
            fioriLaunchpad: req.headers['x-flp-version'] || 'unknown',
            referrer: req.get('Referer') || 'direct'
        };
    }

    /**
     * Sanitize headers to remove sensitive information
     */
    sanitizeHeaders(headers) {
        const sanitized = { ...headers };

        // Remove sensitive headers
        delete sanitized.authorization;
        delete sanitized.cookie;
        delete sanitized['x-csrf-token'];
        delete sanitized['x-api-key'];

        return sanitized;
    }

    /**
     * Calculate step duration
     */
    calculateStepDuration(startTime) {
        const diff = process.hrtime(startTime);
        return (diff[0] * 1000) + (diff[1] / 1000000); // Convert to milliseconds
    }

    /**
     * Get trace by ID
     */
    getTrace(traceId) {
        return this.activeTraces.get(traceId) || this.traceStore.get(traceId);
    }

    /**
     * Get all traces for a user
     */
    getTracesForUser(userId, limit = 100) {
        const traces = [];

        for (const trace of this.traceStore.values()) {
            if (trace.userContext.userId === userId) {
                traces.push(trace);
            }
            if (traces.length >= limit) break;
        }

        return traces.sort((a, b) => b.timestamp - a.timestamp);
    }

    /**
     * Get error traces
     */
    getErrorTraces(limit = 100) {
        const errorTraces = [];

        for (const trace of this.traceStore.values()) {
            if (trace.status === 'error') {
                errorTraces.push(trace);
            }
            if (errorTraces.length >= limit) break;
        }

        return errorTraces.sort((a, b) => b.timestamp - a.timestamp);
    }

    /**
     * Start cleanup interval to manage memory
     */
    startCleanupInterval() {
        const cleanupInterval = setInterval(() => {
            this.cleanupOldTraces();
        }, 60 * 60 * 1000); // Run every hour
        this.intervals.set('trace_cleanup', cleanupInterval);
    }

    /**
     * Cleanup old traces
     */
    cleanupOldTraces() {
        const cutoffTime = Date.now() - this.traceRetentionMs;
        let cleaned = 0;

        for (const [traceId, trace] of this.traceStore.entries()) {
            if (trace.timestamp.getTime() < cutoffTime) {
                this.traceStore.delete(traceId);
                cleaned++;
            }
        }

        if (cleaned > 0) {
            const log = cds.log('trace-manager');
            log.info(`Cleaned up ${cleaned} old traces`);
        }
    }

    /**
     * Update error rate metrics
     */
    updateErrorRate() {
        const totalTraces = this.traceStore.size + this.activeTraces.size;
        if (totalTraces === 0) {
            this.metrics.errorRate = 0;
            return;
        }

        let errorCount = 0;
        for (const trace of this.traceStore.values()) {
            if (trace.status === 'error') {
                errorCount++;
            }
        }

        this.metrics.errorRate = (errorCount / totalTraces) * 100;
    }

    /**
     * Update average response time
     */
    updateAverageResponseTime(newDuration) {
        const alpha = 0.1; // Exponential moving average factor
        if (this.metrics.averageResponseTime === 0) {
            this.metrics.averageResponseTime = newDuration;
        } else {
            this.metrics.averageResponseTime =
                (alpha * newDuration) + ((1 - alpha) * this.metrics.averageResponseTime);
        }
    }

    /**
     * Get performance metrics
     */
    getPerformanceMetrics() {
        return {
            ...this.metrics,
            traceStorageSize: this.traceStore.size,
            activeTracesCount: this.activeTraces.size,
            spansCount: this.spans.size,
            memoryUsage: this.estimateMemoryUsage(),
            hasOpenTelemetry: !!this.tracer
        };
    }

    /**
     * Estimate memory usage
     */
    estimateMemoryUsage() {
        const traceSize = this.traceStore.size + this.activeTraces.size;
        const estimatedBytesPerTrace = 2048; // Rough estimate
        const totalBytes = traceSize * estimatedBytesPerTrace;

        if (totalBytes < 1024) return `${totalBytes} bytes`;
        if (totalBytes < 1024 * 1024) return `${(totalBytes / 1024).toFixed(1)} KB`;
        return `${(totalBytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    /**
     * Generate trace report
     */
    generateTraceReport(traceId) {
        const trace = this.getTrace(traceId);
        if (!trace) return null;

        return {
            summary: {
                traceId: trace.traceId,
                status: trace.status,
                duration: trace.performance.totalDuration,
                timestamp: trace.timestamp,
                user: trace.userContext.userId,
                endpoint: `${trace.requestInfo.method} ${trace.requestInfo.path}`,
                hasOpenTelemetrySpan: !!trace.span
            },
            userContext: trace.userContext,
            frontendContext: trace.frontendContext,
            requestInfo: trace.requestInfo,
            steps: trace.steps.map(step => ({
                name: step.name,
                component: step.component,
                layer: step.layer,
                duration: step.duration,
                status: step.status
            })),
            errors: trace.errors.map(error => ({
                errorId: error.errorId,
                message: error.message,
                component: error.component,
                layer: error.layer,
                severity: error.severity,
                timestamp: error.timestamp
            })),
            performance: trace.performance,
            metrics: this.getPerformanceMetrics()
        };
    }
}

// Singleton instance
const traceManager = new TraceManager();

module.exports = {
    TraceManager,
    traceManager,

    // Middleware functions
    initializeTrace: (req, res, next) => traceManager.initializeTrace(req, res, next),
    addStep: (traceId, stepInfo) => traceManager.addStep(traceId, stepInfo),
    addError: (traceId, error, context) => traceManager.addError(traceId, error, context),
    completeTrace: (traceId, statusCode) => traceManager.completeTrace(traceId, statusCode),
    getTrace: (traceId) => traceManager.getTrace(traceId),
    getTracesForUser: (userId, limit) => traceManager.getTracesForUser(userId, limit),
    getErrorTraces: (limit) => traceManager.getErrorTraces(limit),
    generateTraceReport: (traceId) => traceManager.generateTraceReport(traceId)
};