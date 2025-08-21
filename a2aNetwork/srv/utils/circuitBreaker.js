/**
 * @fileoverview Enterprise SAP Circuit Breaker with Advanced Resilience Patterns
 * @description Provides fault tolerance for blockchain operations with SAP enterprise features:
 * - Adaptive failure thresholds based on historical data
 * - Bulkhead isolation for different service types
 * - Integration with SAP BTP Alert Notification
 * - Performance-based circuit opening
 * - Intelligent retry strategies with exponential backoff
 * - Circuit breaker clustering for distributed systems
 * @module circuit-breaker
 * @since 1.0.0
 * @author A2A Network Team
 */

const cds = require('@sap/cds');
const { EventEmitter } = require('events');

// SAP Cloud SDK Resilience integration
let sapResilience;
try {
    sapResilience = require('@sap-cloud-sdk/resilience');
} catch (error) {
    // SAP Cloud SDK not available in development
}

// OpenTelemetry integration for circuit breaker observability
let opentelemetry, trace, SpanStatusCode;
try {
    opentelemetry = require('@opentelemetry/api');
    trace = opentelemetry.trace;
    SpanStatusCode = opentelemetry.SpanStatusCode;
} catch (error) {
    // OpenTelemetry not available
}

/**
 * Enhanced circuit breaker states with SAP enterprise patterns
 */
const CircuitState = {
    CLOSED: 'CLOSED',         // Normal operation
    OPEN: 'OPEN',             // Blocking requests due to failures
    HALF_OPEN: 'HALF_OPEN',   // Testing if service has recovered
    FORCED_OPEN: 'FORCED_OPEN', // Manually opened for maintenance
    DEGRADED: 'DEGRADED'      // Operating with reduced capacity
};

/**
 * Service priority levels for bulkhead isolation
 */
const ServicePriority = {
    CRITICAL: 'CRITICAL',     // Core business functions
    HIGH: 'HIGH',             // Important features
    MEDIUM: 'MEDIUM',         // Standard operations
    LOW: 'LOW'                // Nice-to-have features
};

/**
 * Failure categories for intelligent analysis
 */
const FailureCategory = {
    TIMEOUT: 'TIMEOUT',
    CONNECTION: 'CONNECTION',
    AUTHENTICATION: 'AUTHENTICATION',
    AUTHORIZATION: 'AUTHORIZATION',
    RATE_LIMIT: 'RATE_LIMIT',
    SERVER_ERROR: 'SERVER_ERROR',
    CLIENT_ERROR: 'CLIENT_ERROR',
    NETWORK: 'NETWORK',
    UNKNOWN: 'UNKNOWN'
};

/**
 * Enterprise SAP Circuit Breaker with Advanced Resilience
 */
class CircuitBreaker extends EventEmitter {
    constructor(options = {}) {
        super();
        
        // Core configuration
        this.serviceName = options.serviceName || 'unknown';
        this.serviceType = options.serviceType || 'blockchain';
        this.priority = options.priority || ServicePriority.MEDIUM;
        
        // Adaptive thresholds
        this.failureThreshold = options.failureThreshold || 5;
        this.successThreshold = options.successThreshold || 3;
        this.resetTimeout = options.resetTimeout || 60000; // 1 minute
        this.halfOpenMaxCalls = options.halfOpenMaxCalls || 3;
        this.monitoringPeriod = options.monitoringPeriod || 300000; // 5 minutes
        
        // Performance-based thresholds
        this.slowCallThreshold = options.slowCallThreshold || 5000; // 5 seconds
        this.slowCallDurationThreshold = options.slowCallDurationThreshold || 10000; // 10 seconds
        this.performanceWindow = options.performanceWindow || 100; // Last 100 calls
        
        // Bulkhead configuration
        this.maxConcurrentCalls = options.maxConcurrentCalls || 50;
        this.maxWaitDuration = options.maxWaitDuration || 30000; // 30 seconds
        
        // State management
        this.state = CircuitState.CLOSED;
        this.previousState = null;
        this.stateChangeReason = null;
        this.failures = 0;
        this.successes = 0;
        this.consecutiveSlowCalls = 0;
        this.lastFailureTime = null;
        this.nextAttempt = Date.now();
        this.halfOpenCalls = 0;
        this.activeCalls = 0;
        this.waitingCalls = 0;
        
        // Advanced metrics
        this.metrics = {
            totalRequests: 0,
            totalFailures: 0,
            totalSuccesses: 0,
            totalTimeouts: 0,
            totalSlowCalls: 0,
            stateChanges: 0,
            lastStateChange: Date.now(),
            averageResponseTime: 0,
            p95ResponseTime: 0,
            p99ResponseTime: 0,
            failuresByCategory: new Map(),
            responseTimes: [],
            concurrentCallsPeak: 0,
            bulkheadRejections: 0
        };
        
        // Error tracking and categorization
        this.recentErrors = [];
        this.errorPatterns = new Map();
        this.performanceHistory = [];
        
        // SAP enterprise features
        this.alertNotificationService = null;
        this.sapCloudSdkIntegration = null;
        this.tracer = trace ? trace.getTracer('circuit-breaker', '1.0.0') : null;
        
        // Adaptive learning
        this.historicalData = {
            failureRates: [],
            recoveryTimes: [],
            patterns: new Map()
        };
        
        // Initialize intervals tracking
        this.intervals = new Map();
        
        // Initialize SAP integrations
        this.initializeSAPIntegrations();
        
        // Start adaptive monitoring
        this.startAdaptiveMonitoring();
        
        cds.log('circuit-breaker').info('Enterprise circuit breaker initialized', {
            serviceName: this.serviceName,
            serviceType: this.serviceType,
            priority: this.priority,
            failureThreshold: this.failureThreshold,
            resetTimeout: this.resetTimeout,
            maxConcurrentCalls: this.maxConcurrentCalls,
            sapIntegration: !!this.sapCloudSdkIntegration
        });
    }
    
    /**
     * Initialize SAP enterprise integrations
     */
    initializeSAPIntegrations() {
        // SAP Cloud SDK Resilience integration
        if (sapResilience) {
            try {
                this.sapCloudSdkIntegration = sapResilience.circuitBreaker({
                    failureThreshold: this.failureThreshold,
                    delay: this.resetTimeout,
                    maxAttempts: this.halfOpenMaxCalls
                });
                
                cds.log('circuit-breaker').info('SAP Cloud SDK integration enabled');
            } catch (error) {
                cds.log('circuit-breaker').warn('SAP Cloud SDK integration failed:', error);
            }
        }
        
        // SAP BTP Alert Notification service
        this.initializeAlertNotification();
    }
    
    /**
     * Initialize SAP BTP Alert Notification
     */
    initializeAlertNotification() {
        try {
            if (process.env.VCAP_SERVICES) {
                const vcapServices = JSON.parse(process.env.VCAP_SERVICES);
                const alertService = vcapServices['alert-notification'];
                
                if (alertService && alertService[0]) {
                    this.alertNotificationService = alertService[0].credentials;
                    cds.log('circuit-breaker').info('SAP Alert Notification service configured');
                }
            }
        } catch (error) {
            cds.log('circuit-breaker').warn('Alert Notification service setup failed:', error);
        }
    }
    
    /**
     * Stop adaptive monitoring
     */
    stopAdaptiveMonitoring() {
        if (this.intervals) {
            for (const [name, intervalId] of this.intervals) {
                clearInterval(intervalId);
            }
            this.intervals.clear();
        }
    }
    
    /**
     * Start adaptive monitoring and learning
     */
    startAdaptiveMonitoring() {
        // Stop existing monitoring first
        this.stopAdaptiveMonitoring();
        
        // Adaptive threshold adjustment
        this.intervals.set('adaptive', setInterval(() => {
            this.adjustAdaptiveThresholds();
        }, 300000)); // Every 5 minutes
        
        // Performance pattern analysis
        this.intervals.set('performance', setInterval(() => {
            this.analyzePerformancePatterns();
        }, 600000)); // Every 10 minutes
        
        // Historical data cleanup
        this.intervals.set('cleanup', setInterval(() => {
            this.cleanupHistoricalData();
        }, 3600000)); // Every hour
    }

    /**
     * Execute operation with enterprise circuit breaker protection
     */
    async execute(operation, operationName = 'unknown', metadata = {}) {
        const startTime = Date.now();
        let span = null;
        
        try {
            // Create OpenTelemetry span for observability
            if (this.tracer) {
                span = this.tracer.startSpan(`circuit-breaker.${this.serviceName}.${operationName}`, {
                    attributes: {
                        'circuit.breaker.service': this.serviceName,
                        'circuit.breaker.state': this.state,
                        'circuit.breaker.operation': operationName,
                        'circuit.breaker.priority': this.priority
                    }
                });
            }
            
            this.metrics.totalRequests++;
            
            // Enterprise bulkhead protection
            await this.enforceBasedOnPriority(operationName);
            
            // Check circuit state with enhanced logic
            await this.checkCircuitState(operationName);
            
            // Increment active calls for bulkhead
            this.activeCalls++;
            this.metrics.concurrentCallsPeak = Math.max(this.metrics.concurrentCallsPeak, this.activeCalls);
            
            try {
                // Execute with adaptive timeout and retry
                const result = await this.executeWithAdaptiveProtection(operation, operationName, metadata);
                
                const duration = Date.now() - startTime;
                this.onSuccess(operationName, duration);
                
                if (span) {
                    span.setAttributes({
                        'circuit.breaker.result': 'success',
                        'circuit.breaker.duration': duration
                    });
                    span.setStatus({ code: SpanStatusCode.OK });
                }
                
                return result;
                
            } catch (error) {
                const duration = Date.now() - startTime;
                this.onFailure(error, operationName, duration);
                
                if (span) {
                    span.recordException(error);
                    span.setAttributes({
                        'circuit.breaker.result': 'failure',
                        'circuit.breaker.duration': duration,
                        'circuit.breaker.error.category': this.categorizeError(error)
                    });
                    span.setStatus({ code: SpanStatusCode.ERROR });
                }
                
                throw error;
            } finally {
                this.activeCalls--;
            }
            
        } finally {
            if (span) {
                span.end();
            }
        }
    }
    
    /**
     * Enforce bulkhead isolation based on service priority
     */
    async enforceBasedOnPriority(operationName) {
        // High priority services get preferential treatment
        const priorityMultiplier = this.getPriorityMultiplier();
        const effectiveMaxCalls = Math.floor(this.maxConcurrentCalls * priorityMultiplier);
        
        if (this.activeCalls >= effectiveMaxCalls) {
            // For critical services, wait briefly
            if (this.priority === ServicePriority.CRITICAL) {
                await this.waitForSlot(operationName);
            } else {
                this.metrics.bulkheadRejections++;
                const error = new Error(`Bulkhead capacity exceeded for ${operationName} (${this.activeCalls}/${effectiveMaxCalls})`);
                error.code = 'BULKHEAD_CAPACITY_EXCEEDED';
                error.priority = this.priority;
                
                this.emit('bulkheadRejected', {
                    serviceName: this.serviceName,
                    operationName,
                    activeCalls: this.activeCalls,
                    maxCalls: effectiveMaxCalls,
                    priority: this.priority
                });
                
                throw error;
            }
        }
    }
    
    /**
     * Get priority multiplier for bulkhead allocation
     */
    getPriorityMultiplier() {
        switch (this.priority) {
            case ServicePriority.CRITICAL: return 1.5;
            case ServicePriority.HIGH: return 1.2;
            case ServicePriority.MEDIUM: return 1.0;
            case ServicePriority.LOW: return 0.7;
            default: return 1.0;
        }
    }
    
    /**
     * Wait for available slot in bulkhead
     */
    async waitForSlot(operationName) {
        const startWait = Date.now();
        this.waitingCalls++;
        
        try {
            while (this.activeCalls >= this.maxConcurrentCalls) {
                if (Date.now() - startWait > this.maxWaitDuration) {
                    const error = new Error(`Wait timeout exceeded for ${operationName}`);
                    error.code = 'BULKHEAD_WAIT_TIMEOUT';
                    throw error;
                }
                
                await new Promise(resolve => setTimeout(resolve, 100)); // Wait 100ms
            }
        } finally {
            this.waitingCalls--;
        }
    }
    
    /**
     * Enhanced circuit state checking
     */
    async checkCircuitState(operationName) {
        if (this.state === CircuitState.FORCED_OPEN) {
            const error = new Error(`Circuit breaker is FORCED_OPEN for ${operationName} (maintenance mode)`);
            error.circuitState = this.state;
            error.code = 'CIRCUIT_FORCED_OPEN';
            this.emit('callRejected', { operationName, state: this.state, reason: 'maintenance' });
            throw error;
        }
        
        if (this.state === CircuitState.OPEN) {
            if (Date.now() < this.nextAttempt) {
                const error = new Error(`Circuit breaker is OPEN for ${operationName}`);
                error.circuitState = this.state;
                error.code = 'CIRCUIT_OPEN';
                error.nextAttempt = this.nextAttempt;
                this.emit('callRejected', { operationName, state: this.state });
                throw error;
            }
            
            // Time to try half-open with intelligent probing
            await this.moveToHalfOpen('timeout_recovery');
        }
        
        if (this.state === CircuitState.DEGRADED) {
            // In degraded mode, apply additional restrictions
            if (this.priority === ServicePriority.LOW) {
                const error = new Error(`Service degraded - low priority calls rejected for ${operationName}`);
                error.circuitState = this.state;
                error.code = 'CIRCUIT_DEGRADED';
                throw error;
            }
        }
        
        // Check half-open call limit with adaptive sizing
        if (this.state === CircuitState.HALF_OPEN) {
            const adaptiveMaxCalls = this.calculateAdaptiveHalfOpenCalls();
            if (this.halfOpenCalls >= adaptiveMaxCalls) {
                const error = new Error(`Circuit breaker HALF_OPEN call limit exceeded for ${operationName}`);
                error.circuitState = this.state;
                error.code = 'CIRCUIT_HALF_OPEN_LIMIT';
                throw error;
            }
            this.halfOpenCalls++;
        }
    }
    
    /**
     * Calculate adaptive half-open call limit based on historical data
     */
    calculateAdaptiveHalfOpenCalls() {
        const recentFailureRate = this.calculateRecentFailureRate();
        
        // More conservative with higher failure rates
        if (recentFailureRate > 0.5) return 1;
        if (recentFailureRate > 0.3) return 2;
        if (recentFailureRate > 0.1) return 3;
        
        return this.halfOpenMaxCalls;
    }
    
    /**
     * Execute with adaptive protection mechanisms
     */
    async executeWithAdaptiveProtection(operation, operationName, metadata) {
        // Calculate adaptive timeout based on historical performance
        const adaptiveTimeout = this.calculateAdaptiveTimeout();
        
        // Apply intelligent retry if this is a retry attempt
        if (metadata.isRetry) {
            await this.applyIntelligentBackoff(metadata.retryAttempt);
        }
        
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                this.metrics.totalTimeouts++;
                reject(new Error(`Operation ${operationName} timed out after ${adaptiveTimeout}ms (adaptive timeout)`));
            }, adaptiveTimeout);

            Promise.resolve(operation())
                .then(result => {
                    clearTimeout(timer);
                    resolve(result);
                })
                .catch(error => {
                    clearTimeout(timer);
                    reject(error);
                });
        });
    }
    
    /**
     * Calculate adaptive timeout based on service performance
     */
    calculateAdaptiveTimeout() {
        const baseTimeout = 30000; // 30 seconds
        
        // Adjust based on recent performance
        if (this.metrics.averageResponseTime > 0) {
            // Use 3x average response time, but within reasonable bounds
            const adaptiveTimeout = Math.max(
                Math.min(this.metrics.averageResponseTime * 3, 60000), // Max 60 seconds
                5000 // Min 5 seconds
            );
            return adaptiveTimeout;
        }
        
        return baseTimeout;
    }
    
    /**
     * Apply intelligent exponential backoff
     */
    async applyIntelligentBackoff(retryAttempt) {
        // Exponential backoff with jitter for distributed systems
        const baseDelay = 1000; // 1 second
        const maxDelay = 30000; // 30 seconds
        const jitterFactor = 0.1;
        
        const exponentialDelay = Math.min(baseDelay * Math.pow(2, retryAttempt - 1), maxDelay);
        const jitter = exponentialDelay * jitterFactor * Math.random();
        const totalDelay = exponentialDelay + jitter;
        
        await new Promise(resolve => setTimeout(resolve, totalDelay));
    }

    /**
     * Execute operation with timeout protection
     */
    async executeWithTimeout(operation, operationName) {
        const timeout = 30000; // 30 seconds default timeout
        
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                reject(new Error(`Operation ${operationName} timed out after ${timeout}ms`));
            }, timeout);

            Promise.resolve(operation())
                .then(result => {
                    clearTimeout(timer);
                    resolve(result);
                })
                .catch(error => {
                    clearTimeout(timer);
                    reject(error);
                });
        });
    }

    /**
     * Handle successful operation with enterprise analytics
     */
    onSuccess(operationName, duration) {
        this.failures = 0;
        this.successes++;
        this.metrics.totalSuccesses++;
        this.consecutiveSlowCalls = 0;
        
        // Track response time metrics
        this.updateResponseTimeMetrics(duration);
        
        // Check for slow calls
        if (duration > this.slowCallDurationThreshold) {
            this.metrics.totalSlowCalls++;
            this.consecutiveSlowCalls++;
            
            // Consider moving to degraded state if too many slow calls
            if (this.consecutiveSlowCalls >= this.slowCallThreshold) {
                this.moveToDegraded('performance_degradation');
            }
        }
        
        // State transition logic
        if (this.state === CircuitState.HALF_OPEN) {
            // Successful call in half-open state - consider closing circuit
            if (this.successes >= this.successThreshold) {
                this.moveToClosed('recovery_confirmed');
            }
        } else if (this.state === CircuitState.DEGRADED) {
            // Check if we can recover from degraded state
            if (this.successes >= this.successThreshold && duration < this.slowCallDurationThreshold) {
                this.moveToClosed('performance_recovered');
            }
        }
        
        // Learn from successful patterns
        this.learnFromSuccess(operationName, duration);
        
        this.emit('callSucceeded', { 
            serviceName: this.serviceName,
            operationName, 
            state: this.state,
            successes: this.successes,
            duration,
            consecutiveSlowCalls: this.consecutiveSlowCalls
        });
    }
    
    /**
     * Update response time metrics with percentile calculations
     */
    updateResponseTimeMetrics(duration) {
        // Update running average
        const alpha = 0.1; // Exponential moving average factor
        this.metrics.averageResponseTime = this.metrics.averageResponseTime === 0
            ? duration
            : (alpha * duration) + ((1 - alpha) * this.metrics.averageResponseTime);
        
        // Store for percentile calculations (keep last 1000 response times)
        this.metrics.responseTimes.push(duration);
        if (this.metrics.responseTimes.length > 1000) {
            this.metrics.responseTimes.shift();
        }
        
        // Update percentiles every 100 requests for performance
        if (this.metrics.responseTimes.length % 100 === 0) {
            this.calculatePercentiles();
        }
    }
    
    /**
     * Calculate response time percentiles
     */
    calculatePercentiles() {
        const sorted = [...this.metrics.responseTimes].sort((a, b) => a - b);
        const length = sorted.length;
        
        if (length > 0) {
            this.metrics.p95ResponseTime = sorted[Math.floor(length * 0.95)];
            this.metrics.p99ResponseTime = sorted[Math.floor(length * 0.99)];
        }
    }
    
    /**
     * Learn from successful operations for adaptive improvements
     */
    learnFromSuccess(operationName, duration) {
        // Store performance patterns
        this.performanceHistory.push({
            operationName,
            duration,
            timestamp: Date.now(),
            state: this.state,
            success: true
        });
        
        // Keep only recent history (last 1000 operations)
        if (this.performanceHistory.length > 1000) {
            this.performanceHistory.shift();
        }
        
        // Update historical patterns for this operation
        const pattern = this.historicalData.patterns.get(operationName) || {
            successCount: 0,
            failureCount: 0,
            avgDuration: 0,
            lastSuccess: null
        };
        
        pattern.successCount++;
        pattern.avgDuration = ((pattern.avgDuration * (pattern.successCount - 1)) + duration) / pattern.successCount;
        pattern.lastSuccess = Date.now();
        
        this.historicalData.patterns.set(operationName, pattern);
    }

    /**
     * Handle failed operation with enterprise error analysis
     */
    onFailure(error, operationName, duration) {
        this.failures++;
        this.successes = 0;
        this.metrics.totalFailures++;
        this.lastFailureTime = Date.now();
        
        // Categorize the error for intelligent analysis
        const errorCategory = this.categorizeError(error);
        
        // Update failure metrics by category
        const categoryCount = this.metrics.failuresByCategory.get(errorCategory) || 0;
        this.metrics.failuresByCategory.set(errorCategory, categoryCount + 1);
        
        // Track recent errors with enhanced metadata
        const errorInfo = {
            error: error.message,
            code: error.code || 'UNKNOWN',
            category: errorCategory,
            operationName,
            duration: duration || 0,
            timestamp: Date.now(),
            stack: error.stack,
            state: this.state,
            circuitBreakerMetadata: {
                failures: this.failures,
                successes: this.successes,
                activeCalls: this.activeCalls
            }
        };
        
        this.recentErrors.push(errorInfo);
        
        // Keep only recent errors (last 50 for better analysis)
        if (this.recentErrors.length > 50) {
            this.recentErrors.shift();
        }
        
        // Analyze error patterns for adaptive behavior
        this.analyzeErrorPatterns(errorInfo);
        
        // Learn from failure patterns
        this.learnFromFailure(operationName, errorInfo);
        
        // State transition logic with enhanced decision making
        this.determineStateTransition(errorCategory, errorInfo);
        
        // Send alert for critical errors
        this.sendAlertIfNecessary(errorInfo);
        
        this.emit('callFailed', { 
            serviceName: this.serviceName,
            operationName, 
            error: error.message,
            category: errorCategory,
            code: error.code,
            state: this.state,
            failures: this.failures,
            duration: duration
        });
    }
    
    /**
     * Categorize errors for intelligent handling
     */
    categorizeError(error) {
        const message = error.message.toLowerCase();
        const code = error.code || '';
        
        // Timeout errors
        if (message.includes('timeout') || message.includes('timed out') || code === 'TIMEOUT') {
            return FailureCategory.TIMEOUT;
        }
        
        // Connection errors
        if (message.includes('connection') || message.includes('connect') || 
            code === 'ECONNRESET' || code === 'ECONNREFUSED' || code === 'ETIMEDOUT') {
            return FailureCategory.CONNECTION;
        }
        
        // Authentication errors
        if (message.includes('auth') || message.includes('unauthorized') || 
            code === 'UNAUTHORIZED' || error.status === 401) {
            return FailureCategory.AUTHENTICATION;
        }
        
        // Authorization errors
        if (message.includes('forbidden') || message.includes('access denied') || 
            code === 'FORBIDDEN' || error.status === 403) {
            return FailureCategory.AUTHORIZATION;
        }
        
        // Rate limiting
        if (message.includes('rate limit') || message.includes('too many') || 
            code === 'RATE_LIMITED' || error.status === 429) {
            return FailureCategory.RATE_LIMIT;
        }
        
        // Server errors
        if (error.status >= 500 || message.includes('server error') || 
            message.includes('internal error')) {
            return FailureCategory.SERVER_ERROR;
        }
        
        // Client errors
        if (error.status >= 400 && error.status < 500) {
            return FailureCategory.CLIENT_ERROR;
        }
        
        // Network errors
        if (message.includes('network') || message.includes('dns') || code === 'ENETUNREACH') {
            return FailureCategory.NETWORK;
        }
        
        return FailureCategory.UNKNOWN;
    }
    
    /**
     * Analyze error patterns for adaptive behavior
     */
    analyzeErrorPatterns(errorInfo) {
        const category = errorInfo.category;
        const pattern = this.errorPatterns.get(category) || {
            count: 0,
            recentCount: 0,
            firstSeen: Date.now(),
            lastSeen: null,
            operations: new Set()
        };
        
        pattern.count++;
        pattern.recentCount++;
        pattern.lastSeen = Date.now();
        pattern.operations.add(errorInfo.operationName);
        
        this.errorPatterns.set(category, pattern);
        
        // Reset recent count every 5 minutes
        if (!pattern.resetTimer) {
            pattern.resetTimer = setInterval(() => {
                pattern.recentCount = 0;
            }, 300000);
        }
    }
    
    /**
     * Learn from failure patterns for adaptive improvements
     */
    learnFromFailure(operationName, errorInfo) {
        // Store failure patterns
        this.performanceHistory.push({
            operationName,
            duration: errorInfo.duration,
            timestamp: Date.now(),
            state: this.state,
            success: false,
            category: errorInfo.category,
            error: errorInfo.error
        });
        
        // Update historical patterns for this operation
        const pattern = this.historicalData.patterns.get(operationName) || {
            successCount: 0,
            failureCount: 0,
            avgDuration: 0,
            lastFailure: null,
            commonErrors: new Map()
        };
        
        pattern.failureCount++;
        pattern.lastFailure = Date.now();
        
        // Track common error types for this operation
        const errorCount = pattern.commonErrors.get(errorInfo.category) || 0;
        pattern.commonErrors.set(errorInfo.category, errorCount + 1);
        
        this.historicalData.patterns.set(operationName, pattern);
    }
    
    /**
     * Determine state transition based on error analysis
     */
    determineStateTransition(errorCategory, errorInfo) {
        // Different error categories trigger different behaviors
        switch (errorCategory) {
            case FailureCategory.RATE_LIMIT:
                // Rate limits should trigger immediate circuit opening with longer recovery
                if (this.state === CircuitState.CLOSED) {
                    this.resetTimeout = Math.max(this.resetTimeout, 120000); // Min 2 minutes
                    this.moveToOpen('rate_limit_protection');
                }
                break;
                
            case FailureCategory.AUTHENTICATION:
            case FailureCategory.AUTHORIZATION:
                // Auth errors shouldn't trigger circuit breaker unless systemic
                if (this.failures >= this.failureThreshold * 2) {
                    this.moveToOpen('authentication_system_failure');
                }
                break;
                
            case FailureCategory.CONNECTION:
            case FailureCategory.NETWORK:
            case FailureCategory.TIMEOUT:
                // Infrastructure issues should trigger normal circuit behavior
                this.handleInfrastructureFailure();
                break;
                
            case FailureCategory.SERVER_ERROR:
                // Server errors are more serious - lower threshold
                if (this.failures >= Math.max(this.failureThreshold - 2, 2)) {
                    this.moveToOpen('server_error_pattern');
                }
                break;
                
            default:
                // Standard failure handling
                this.handleStandardFailure();
        }
    }
    
    /**
     * Handle infrastructure-related failures
     */
    handleInfrastructureFailure() {
        if (this.state === CircuitState.HALF_OPEN) {
            // Infrastructure failure in half-open - back to open with extended timeout
            this.resetTimeout = Math.min(this.resetTimeout * 1.5, 300000); // Max 5 minutes
            this.moveToOpen('infrastructure_failure_recovery');
        } else if (this.failures >= this.failureThreshold) {
            this.moveToOpen('infrastructure_failure');
        }
    }
    
    /**
     * Handle standard failures
     */
    handleStandardFailure() {
        if (this.state === CircuitState.HALF_OPEN) {
            // Failure in half-open state - go back to open
            this.moveToOpen('half_open_failure');
        } else if (this.failures >= this.failureThreshold) {
            // Too many failures - open the circuit
            this.moveToOpen('failure_threshold_exceeded');
        }
    }
    
    /**
     * Send alert if necessary based on error severity
     */
    async sendAlertIfNecessary(errorInfo) {
        const shouldAlert = (
            errorInfo.category === FailureCategory.SERVER_ERROR ||
            errorInfo.category === FailureCategory.AUTHENTICATION ||
            this.failures >= this.failureThreshold ||
            this.state === CircuitState.OPEN
        );
        
        if (shouldAlert && this.alertNotificationService) {
            try {
                await this.sendSAPAlert({
                    serviceName: this.serviceName,
                    severity: this.calculateAlertSeverity(errorInfo),
                    message: `Circuit breaker failure: ${errorInfo.error}`,
                    category: errorInfo.category,
                    state: this.state,
                    failures: this.failures
                });
            } catch (alertError) {
                cds.log('circuit-breaker').warn('Failed to send alert:', alertError);
            }
        }
    }
    
    /**
     * Calculate alert severity
     */
    calculateAlertSeverity(errorInfo) {
        if (this.state === CircuitState.OPEN || errorInfo.category === FailureCategory.SERVER_ERROR) {
            return 'HIGH';
        }
        if (this.failures >= this.failureThreshold - 1) {
            return 'MEDIUM';
        }
        return 'LOW';
    }
    
    /**
     * Send alert to SAP Alert Notification service
     */
    async sendSAPAlert(alertData) {
        if (!this.alertNotificationService) return;
        
        const alert = {
            eventType: 'circuit-breaker-alert',
            severity: alertData.severity,
            category: 'Technical',
            subject: `Circuit Breaker Alert: ${alertData.serviceName}`,
            body: alertData.message,
            tags: {
                service: alertData.serviceName,
                category: alertData.category,
                state: alertData.state,
                failures: alertData.failures.toString()
            },
            resource: {
                resourceName: alertData.serviceName,
                resourceType: 'circuit-breaker',
                tags: {
                    priority: this.priority,
                    serviceType: this.serviceType
                }
            }
        };
        
        // In production, would send to actual SAP Alert Notification service
        cds.log('circuit-breaker').warn('SAP Alert would be sent:', alert);
    }

    /**
     * Move circuit to CLOSED state with enhanced tracking
     */
    moveToClosed(reason = 'recovery') {
        const previousState = this.state;
        this.previousState = previousState;
        this.state = CircuitState.CLOSED;
        this.stateChangeReason = reason;
        this.failures = 0;
        this.successes = 0;
        this.halfOpenCalls = 0;
        this.consecutiveSlowCalls = 0;
        this.metrics.stateChanges++;
        this.metrics.lastStateChange = Date.now();
        
        // Learn from recovery time for adaptive behavior
        if (previousState === CircuitState.OPEN) {
            const recoveryTime = Date.now() - (this.lastFailureTime || Date.now());
            this.historicalData.recoveryTimes.push(recoveryTime);
            
            // Keep only recent recovery times for analysis
            if (this.historicalData.recoveryTimes.length > 20) {
                this.historicalData.recoveryTimes.shift();
            }
        }
        
        cds.log('circuit-breaker').info('Circuit breaker moved to CLOSED', {
            serviceName: this.serviceName,
            previousState,
            reason,
            totalRequests: this.metrics.totalRequests,
            recoveryTime: previousState === CircuitState.OPEN ? Date.now() - this.lastFailureTime : null
        });
        
        this.emit('stateChanged', {
            serviceName: this.serviceName,
            from: previousState,
            to: this.state,
            reason,
            timestamp: Date.now(),
            metadata: {
                totalRequests: this.metrics.totalRequests,
                failureRate: this.calculateFailureRate(),
                averageResponseTime: this.metrics.averageResponseTime
            }
        });
    }

    /**
     * Move circuit to OPEN state
     */
    moveToOpen() {
        const previousState = this.state;
        this.state = CircuitState.OPEN;
        this.nextAttempt = Date.now() + this.resetTimeout;
        this.halfOpenCalls = 0;
        this.metrics.stateChanges++;
        this.metrics.lastStateChange = Date.now();
        
        cds.log('circuit-breaker').warn('Circuit breaker moved to OPEN', {
            previousState,
            failures: this.failures,
            resetTimeout: this.resetTimeout,
            nextAttempt: new Date(this.nextAttempt).toISOString()
        });
        
        this.emit('stateChanged', {
            from: previousState,
            to: this.state,
            timestamp: Date.now(),
            nextAttempt: this.nextAttempt
        });
    }

    /**
     * Move circuit to HALF_OPEN state
     */
    moveToHalfOpen() {
        const previousState = this.state;
        this.state = CircuitState.HALF_OPEN;
        this.halfOpenCalls = 0;
        this.successes = 0;
        this.metrics.stateChanges++;
        this.metrics.lastStateChange = Date.now();
        
        cds.log('circuit-breaker').info('Circuit breaker moved to HALF_OPEN', {
            previousState,
            maxCalls: this.halfOpenMaxCalls
        });
        
        this.emit('stateChanged', {
            from: previousState,
            to: this.state,
            timestamp: Date.now()
        });
    }

    /**
     * Get current circuit breaker status
     */
    getStatus() {
        return {
            state: this.state,
            failures: this.failures,
            successes: this.successes,
            halfOpenCalls: this.halfOpenCalls,
            nextAttempt: this.nextAttempt,
            lastFailureTime: this.lastFailureTime,
            metrics: { ...this.metrics },
            recentErrors: this.recentErrors.slice(-5) // Last 5 errors
        };
    }

    /**
     * Get health status for monitoring
     */
    getHealthStatus() {
        const now = Date.now();
        const timeSinceLastFailure = this.lastFailureTime ? now - this.lastFailureTime : null;
        
        return {
            healthy: this.state === CircuitState.CLOSED,
            state: this.state,
            failureRate: this.calculateFailureRate(),
            timeSinceLastFailure: timeSinceLastFailure,
            uptime: this.calculateUptime()
        };
    }

    /**
     * Calculate failure rate over recent period
     */
    calculateFailureRate() {
        if (this.metrics.totalRequests === 0) return 0;
        return (this.metrics.totalFailures / this.metrics.totalRequests) * 100;
    }

    /**
     * Calculate uptime percentage
     */
    calculateUptime() {
        const totalTime = Date.now() - this.metrics.lastStateChange;
        if (totalTime === 0) return 100;
        
        // This is simplified - in production, track actual downtime
        return this.state === CircuitState.OPEN ? 0 : 100;
    }

    /**
     * Reset circuit breaker to initial state
     */
    reset() {
        cds.log('circuit-breaker').info('Circuit breaker manually reset');
        
        this.state = CircuitState.CLOSED;
        this.failures = 0;
        this.successes = 0;
        this.halfOpenCalls = 0;
        this.lastFailureTime = null;
        this.nextAttempt = Date.now();
        this.recentErrors = [];
        
        this.emit('circuitReset', { timestamp: Date.now() });
    }

    /**
     * Force circuit to specific state (for testing)
     */
    forceState(state) {
        if (!Object.values(CircuitState).includes(state)) {
            throw new Error(`Invalid circuit state: ${state}`);
        }
        
        const previousState = this.state;
        this.state = state;
        
        cds.log('circuit-breaker').warn('Circuit breaker state forced', {
            from: previousState,
            to: state
        });
        
        this.emit('stateForced', {
            from: previousState,
            to: state,
            timestamp: Date.now()
        });
    }
    
    /**
     * Move circuit to DEGRADED state (new enterprise feature)
     */
    moveToDegraded(reason = 'performance_degradation') {
        const previousState = this.state;
        this.previousState = previousState;
        this.state = CircuitState.DEGRADED;
        this.stateChangeReason = reason;
        this.metrics.stateChanges++;
        this.metrics.lastStateChange = Date.now();
        
        cds.log('circuit-breaker').warn('Circuit breaker moved to DEGRADED', {
            serviceName: this.serviceName,
            previousState,
            reason,
            consecutiveSlowCalls: this.consecutiveSlowCalls,
            averageResponseTime: this.metrics.averageResponseTime
        });
        
        this.emit('stateChanged', {
            serviceName: this.serviceName,
            from: previousState,
            to: this.state,
            reason,
            timestamp: Date.now(),
            metadata: {
                consecutiveSlowCalls: this.consecutiveSlowCalls,
                averageResponseTime: this.metrics.averageResponseTime
            }
        });
    }
    
    /**
     * Calculate recent failure rate for adaptive behavior
     */
    calculateRecentFailureRate() {
        const recentHistory = this.performanceHistory.filter(h => 
            Date.now() - h.timestamp < 300000 // Last 5 minutes
        );
        
        if (recentHistory.length === 0) return 0;
        
        const failures = recentHistory.filter(h => !h.success).length;
        return failures / recentHistory.length;
    }
    
    /**
     * Get comprehensive enterprise status
     */
    getEnterpriseStatus() {
        return {
            basic: this.getStatus(),
            health: this.getHealthStatus(),
            performance: {
                averageResponseTime: this.metrics.averageResponseTime,
                p95ResponseTime: this.metrics.p95ResponseTime,
                p99ResponseTime: this.metrics.p99ResponseTime,
                slowCallsCount: this.metrics.totalSlowCalls,
                consecutiveSlowCalls: this.consecutiveSlowCalls
            },
            bulkhead: {
                activeCalls: this.activeCalls,
                waitingCalls: this.waitingCalls,
                maxConcurrentCalls: this.maxConcurrentCalls,
                peakConcurrentCalls: this.metrics.concurrentCallsPeak,
                bulkheadRejections: this.metrics.bulkheadRejections,
                priority: this.priority
            },
            adaptive: {
                currentFailureThreshold: this.failureThreshold,
                currentResetTimeout: this.resetTimeout,
                recentFailureRate: this.calculateRecentFailureRate(),
                adaptiveTimeout: this.calculateAdaptiveTimeout()
            },
            errors: {
                byCategory: Object.fromEntries(this.metrics.failuresByCategory),
                recentPatterns: Object.fromEntries(this.errorPatterns),
                recentErrors: this.recentErrors.slice(-10)
            },
            integration: {
                sapCloudSdk: !!this.sapCloudSdkIntegration,
                alertNotification: !!this.alertNotificationService,
                openTelemetry: !!this.tracer
            },
            metadata: {
                serviceName: this.serviceName,
                serviceType: this.serviceType,
                priority: this.priority,
                stateChangeReason: this.stateChangeReason,
                previousState: this.previousState
            }
        };
    }
    
    /**
     * Force maintenance mode (FORCED_OPEN state)
     */
    enterMaintenanceMode(reason = 'scheduled_maintenance') {
        const previousState = this.state;
        this.previousState = previousState;
        this.state = CircuitState.FORCED_OPEN;
        this.stateChangeReason = reason;
        this.metrics.stateChanges++;
        this.metrics.lastStateChange = Date.now();
        
        cds.log('circuit-breaker').warn('Circuit breaker entered maintenance mode', {
            serviceName: this.serviceName,
            previousState,
            reason
        });
        
        this.emit('maintenanceModeEntered', {
            serviceName: this.serviceName,
            previousState,
            reason,
            timestamp: Date.now()
        });
    }
    
    /**
     * Exit maintenance mode
     */
    exitMaintenanceMode() {
        if (this.state !== CircuitState.FORCED_OPEN) {
            throw new Error('Circuit breaker is not in maintenance mode');
        }
        
        const previousState = this.state;
        this.moveToClosed('maintenance_completed');
        
        cds.log('circuit-breaker').info('Circuit breaker exited maintenance mode', {
            serviceName: this.serviceName,
            previousState
        });
        
        this.emit('maintenanceModeExited', {
            serviceName: this.serviceName,
            previousState,
            timestamp: Date.now()
        });
    }

    /**
     * Get enterprise status with detailed metrics
     */
    getEnterpriseStatus() {
        const now = Date.now();
        const uptime = this.calculateUptime();
        
        return {
            serviceName: this.serviceName,
            serviceType: this.serviceType,
            priority: this.priority,
            state: this.state,
            health: this.getHealthStatus(),
            metrics: {
                ...this.metrics,
                uptime,
                uptimePercentage: this.calculateUptimePercentage(),
                avgResponseTime: this.calculateAverageResponseTime(),
                errorRate: this.calculateErrorRate(),
                throughput: this.calculateThroughput()
            },
            thresholds: {
                failure: this.failureThreshold,
                success: this.successThreshold,
                slowCall: this.slowCallThreshold,
                maxConcurrent: this.maxConcurrentCalls
            },
            bulkhead: {
                currentCalls: this.currentCalls,
                maxCalls: this.maxConcurrentCalls,
                utilization: (this.currentCalls / this.maxConcurrentCalls * 100).toFixed(2) + '%'
            },
            lastStateChange: new Date(this.lastStateChange).toISOString(),
            createdAt: new Date(this.createdAt).toISOString()
        };
    }

    /**
     * Calculate service uptime
     */
    calculateUptime() {
        return Date.now() - this.createdAt;
    }

    /**
     * Calculate uptime percentage
     */
    calculateUptimePercentage() {
        const totalTime = Date.now() - this.createdAt;
        const downTime = this.calculateDownTime();
        return ((totalTime - downTime) / totalTime * 100).toFixed(2);
    }

    /**
     * Calculate total downtime
     */
    calculateDownTime() {
        let downTime = 0;
        for (const downPeriod of this.history.downtimes || []) {
            downTime += (downPeriod.endTime || Date.now()) - downPeriod.startTime;
        }
        return downTime;
    }

    /**
     * Calculate average response time
     */
    calculateAverageResponseTime() {
        if (this.history.responseTimes.length === 0) return 0;
        const sum = this.history.responseTimes.reduce((a, b) => a + b, 0);
        return (sum / this.history.responseTimes.length).toFixed(2);
    }

    /**
     * Calculate error rate percentage
     */
    calculateErrorRate() {
        if (this.metrics.totalRequests === 0) return 0;
        return (this.metrics.totalFailures / this.metrics.totalRequests * 100).toFixed(2);
    }

    /**
     * Calculate throughput (requests per minute)
     */
    calculateThroughput() {
        const uptimeMinutes = (Date.now() - this.createdAt) / 60000;
        if (uptimeMinutes === 0) return 0;
        return (this.metrics.totalRequests / uptimeMinutes).toFixed(2);
    }

    /**
     * Adaptive threshold adjustment based on historical performance
     */
    adjustAdaptiveThresholds() {
        if (!this.adaptiveThresholds || this.history.responseTimes.length < 50) {
            return; // Need enough data for adaptation
        }

        const recentTimes = this.history.responseTimes.slice(-50);
        const avgResponseTime = recentTimes.reduce((a, b) => a + b, 0) / recentTimes.length;
        const p95ResponseTime = this.calculatePercentile(recentTimes, 95);

        // Adjust slow call threshold based on recent performance
        const newSlowCallThreshold = Math.max(
            this.slowCallThreshold * 0.8, // Don't go below 80% of original
            Math.min(
                this.slowCallThreshold * 1.5, // Don't go above 150% of original
                p95ResponseTime * 1.2 // 20% above 95th percentile
            )
        );

        if (Math.abs(newSlowCallThreshold - this.slowCallThreshold) > 100) {
            this.log.info(`Adaptive threshold adjustment for ${this.serviceName}`, {
                oldThreshold: this.slowCallThreshold,
                newThreshold: newSlowCallThreshold,
                avgResponseTime,
                p95ResponseTime
            });
            this.slowCallThreshold = newSlowCallThreshold;
        }

        // Adjust failure threshold based on error patterns
        const recentFailures = this.history.failures.slice(-100);
        if (recentFailures.length >= 20) {
            const errorPatterns = this.analyzeErrorPatterns(recentFailures);
            if (errorPatterns.hasTemporaryIssues) {
                this.failureThreshold = Math.min(this.failureThreshold + 1, 10);
            } else if (errorPatterns.hasSystemicIssues) {
                this.failureThreshold = Math.max(this.failureThreshold - 1, 2);
            }
        }
    }

    /**
     * Calculate percentile from array of numbers
     */
    calculatePercentile(arr, percentile) {
        const sorted = [...arr].sort((a, b) => a - b);
        const index = (percentile / 100) * (sorted.length - 1);
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        const weight = index % 1;

        if (upper >= sorted.length) return sorted[sorted.length - 1];
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }

    /**
     * Analyze error patterns for adaptive behavior
     */
    analyzeErrorPatterns(failures) {
        const timeWindows = [300000, 900000, 1800000]; // 5, 15, 30 minutes
        let hasTemporaryIssues = false;
        let hasSystemicIssues = false;

        for (const window of timeWindows) {
            const windowStart = Date.now() - window;
            const windowFailures = failures.filter(f => f.timestamp > windowStart);
            const errorRate = windowFailures.length / window * 60000; // errors per minute

            if (errorRate > 0.5 && errorRate < 2) {
                hasTemporaryIssues = true;
            } else if (errorRate >= 2) {
                hasSystemicIssues = true;
            }
        }

        return { hasTemporaryIssues, hasSystemicIssues };
    }

    /**
     * Analyze performance patterns for predictive scaling
     */
    analyzePerformancePatterns() {
        const recentMetrics = this.history.responseTimes.slice(-100);
        if (recentMetrics.length < 20) return;

        const trend = this.calculateTrend(recentMetrics);
        const volatility = this.calculateVolatility(recentMetrics);

        if (trend > 0.1 && volatility < 0.3) {
            // Performance degrading consistently - move to degraded state
            if (this.state === CircuitState.CLOSED) {
                this.moveToDegraded();
            }
        } else if (trend < -0.1 && this.state === CircuitState.DEGRADED) {
            // Performance improving - consider moving back to closed
            this.consecutiveSuccesses++;
            if (this.consecutiveSuccesses >= this.successThreshold) {
                this.moveToState(CircuitState.CLOSED);
            }
        }
    }

    /**
     * Calculate trend (-1 to 1, negative = improving, positive = degrading)
     */
    calculateTrend(values) {
        if (values.length < 10) return 0;

        const n = values.length;
        const sumX = (n * (n - 1)) / 2;
        const sumY = values.reduce((a, b) => a + b, 0);
        const sumXY = values.reduce((sum, y, x) => sum + x * y, 0);
        const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;

        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const avgY = sumY / n;

        return slope / avgY; // Normalize by average value
    }

    /**
     * Calculate volatility (coefficient of variation)
     */
    calculateVolatility(values) {
        if (values.length < 2) return 0;

        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);

        return mean > 0 ? stdDev / mean : 0;
    }

    /**
     * Clean up historical data to prevent memory leaks
     */
    cleanupHistoricalData() {
        const maxHistory = 1000;
        const oldDataThreshold = Date.now() - (24 * 60 * 60 * 1000); // 24 hours

        // Trim arrays to maximum size
        this.history.responseTimes = this.history.responseTimes.slice(-maxHistory);
        this.history.failures = this.history.failures.slice(-maxHistory);

        // Remove old data
        this.history.failures = this.history.failures.filter(f => f.timestamp > oldDataThreshold);

        // Clean up downtime history
        if (this.history.downtimes) {
            this.history.downtimes = this.history.downtimes.filter(d => 
                (d.endTime || Date.now()) > oldDataThreshold
            );
        }
    }
}

/**
 * Enhanced Circuit breaker registry with enterprise features
 */
class CircuitBreakerRegistry {
    constructor() {
        this.breakers = new Map();
        this.intervals = new Map(); // Track intervals for cleanup
        this.globalMetrics = {
            totalBreakers: 0,
            openBreakers: 0,
            degradedBreakers: 0,
            maintenanceBreakers: 0,
            totalRequests: 0,
            totalFailures: 0
        };
        
        // Start global monitoring
        this.startGlobalMonitoring();
    }

    /**
     * Get or create circuit breaker for a service with enterprise defaults
     */
    getBreaker(serviceName, options = {}) {
        if (!this.breakers.has(serviceName)) {
            // Set enterprise defaults based on service name patterns
            const enterpriseOptions = this.getEnterpriseDefaults(serviceName, options);
            const breaker = new CircuitBreaker(enterpriseOptions);
            this.breakers.set(serviceName, breaker);
            
            // Set up enterprise monitoring
            this.setupEnterpriseMonitoring(serviceName, breaker);
            this.globalMetrics.totalBreakers++;
        }
        
        return this.breakers.get(serviceName);
    }
    
    /**
     * Get enterprise defaults based on service patterns
     */
    getEnterpriseDefaults(serviceName, options) {
        const defaults = { ...options, serviceName };
        
        // SAP-specific service patterns
        if (serviceName.includes('sap') || serviceName.includes('s4hana')) {
            defaults.priority = ServicePriority.CRITICAL;
            defaults.maxConcurrentCalls = 100;
            defaults.failureThreshold = 3;
        } else if (serviceName.includes('blockchain') || serviceName.includes('eth')) {
            defaults.priority = ServicePriority.HIGH;
            defaults.maxConcurrentCalls = 50;
            defaults.failureThreshold = 5;
        } else if (serviceName.includes('auth') || serviceName.includes('security')) {
            defaults.priority = ServicePriority.CRITICAL;
            defaults.maxConcurrentCalls = 200;
            defaults.failureThreshold = 2;
        }
        
        return defaults;
    }

    /**
     * Set up enterprise monitoring for circuit breaker
     */
    setupEnterpriseMonitoring(serviceName, breaker) {
        breaker.on('stateChanged', (event) => {
            this.updateGlobalMetrics();
            cds.log('circuit-monitor').info('Circuit breaker state changed', {
                service: serviceName,
                ...event
            });
        });

        breaker.on('callFailed', (event) => {
            this.globalMetrics.totalFailures++;
            cds.log('circuit-monitor').warn('Circuit breaker call failed', {
                service: serviceName,
                ...event
            });
        });
        
        breaker.on('bulkheadRejected', (event) => {
            cds.log('circuit-monitor').warn('Bulkhead rejection', {
                service: serviceName,
                ...event
            });
        });
        
        breaker.on('maintenanceModeEntered', (event) => {
            this.globalMetrics.maintenanceBreakers++;
            cds.log('circuit-monitor').info('Service entered maintenance mode', {
                service: serviceName,
                ...event
            });
        });
    }
    
    /**
     * Update global metrics
     */
    updateGlobalMetrics() {
        this.globalMetrics.openBreakers = 0;
        this.globalMetrics.degradedBreakers = 0;
        this.globalMetrics.maintenanceBreakers = 0;
        this.globalMetrics.totalRequests = 0;
        
        for (const breaker of this.breakers.values()) {
            this.globalMetrics.totalRequests += breaker.metrics.totalRequests;
            
            switch (breaker.state) {
                case CircuitState.OPEN:
                    this.globalMetrics.openBreakers++;
                    break;
                case CircuitState.DEGRADED:
                    this.globalMetrics.degradedBreakers++;
                    break;
                case CircuitState.FORCED_OPEN:
                    this.globalMetrics.maintenanceBreakers++;
                    break;
            }
        }
    }
    
    /**
     * Start global monitoring
     */
    startGlobalMonitoring() {
        this.intervals.set('global', setInterval(() => {
            this.updateGlobalMetrics();
            this.generateGlobalReport();
        }, 300000)); // Every 5 minutes
    }
    
    stopAdaptiveMonitoring() {
        for (const [name, intervalId] of this.intervals) {
            clearInterval(intervalId);
        }
        this.intervals.clear();
    }
    
    shutdown() {
        this.stopAdaptiveMonitoring();
        this.breakers.clear();
    }
    
    /**
     * Generate global circuit breaker report
     */
    generateGlobalReport() {
        const report = {
            timestamp: new Date().toISOString(),
            summary: this.globalMetrics,
            criticalServices: this.getCriticalServiceStatus(),
            recommendations: this.generateRecommendations()
        };
        
        cds.log('circuit-monitor').info('Global circuit breaker report', report);
        return report;
    }
    
    /**
     * Get status of critical services
     */
    getCriticalServiceStatus() {
        const critical = [];
        
        for (const [serviceName, breaker] of this.breakers.entries()) {
            if (breaker.priority === ServicePriority.CRITICAL) {
                critical.push({
                    serviceName,
                    state: breaker.state,
                    health: breaker.getHealthStatus(),
                    failures: breaker.failures,
                    uptime: breaker.calculateUptime()
                });
            }
        }
        
        return critical;
    }
    
    /**
     * Generate enterprise recommendations
     */
    generateRecommendations() {
        const recommendations = [];
        
        if (this.globalMetrics.openBreakers > 0) {
            recommendations.push('Investigate services with open circuit breakers');
        }
        
        if (this.globalMetrics.degradedBreakers > this.globalMetrics.totalBreakers * 0.2) {
            recommendations.push('Multiple services showing performance degradation - check infrastructure');
        }
        
        return recommendations;
    }

    /**
     * Get status of all circuit breakers with enterprise details
     */
    getAllEnterpriseStatus() {
        const status = {};
        for (const [serviceName, breaker] of this.breakers.entries()) {
            status[serviceName] = breaker.getEnterpriseStatus();
        }
        return {
            services: status,
            global: this.globalMetrics,
            summary: this.generateGlobalReport()
        };
    }
}

// Global registry instance with enterprise features
const circuitBreakerRegistry = new CircuitBreakerRegistry();

module.exports = {
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    ServicePriority,
    FailureCategory,
    getBreaker: (serviceName, options) => circuitBreakerRegistry.getBreaker(serviceName, options),
    getAllStatus: () => circuitBreakerRegistry.getAllStatus(),
    getAllHealthStatus: () => circuitBreakerRegistry.getAllHealthStatus(),
    getAllEnterpriseStatus: () => circuitBreakerRegistry.getAllEnterpriseStatus(),
    getGlobalMetrics: () => circuitBreakerRegistry.globalMetrics
};