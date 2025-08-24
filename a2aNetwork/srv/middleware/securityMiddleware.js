/**
 * Security Middleware Integration for A2A Network
 * Integrates SecurityMonitoringService with Express middleware
 * Provides real-time security monitoring and automated response
 */

const SecurityMonitoringService = require('../services/securityMonitoringService');
const cds = require('@sap/cds');

class SecurityMiddleware {
    constructor() {
        this.log = cds.log('security-middleware');
        this.securityMonitor = null;
        this.blockedIPs = new Set();
        this.quarantinedUsers = new Set();
        this.rateLimiters = new Map();
        this.init();
    }

    async init() {
        try {
            // Initialize security monitoring service
            this.securityMonitor = new SecurityMonitoringService();
            
            // Listen for security events
            this.securityMonitor.on('ipBlocked', (data) => {
                this.blockedIPs.add(data.ipAddress);
                setTimeout(() => {
                    this.blockedIPs.delete(data.ipAddress);
                }, data.duration);
            });
            
            this.securityMonitor.on('userQuarantined', (data) => {
                this.quarantinedUsers.add(data.userId);
                setTimeout(() => {
                    this.quarantinedUsers.delete(data.userId);
                }, data.duration);
            });
            
            this.log.info('Security middleware initialized successfully');
        } catch (error) {
            this.log.error('Failed to initialize security middleware:', error);
            throw error;
        }
    }

    /**
     * Main security monitoring middleware
     */
    middleware() {
        return (req, res, next) => {
            const startTime = Date.now();
            const clientIP = this.getClientIP(req);
            const userAgent = req.get('User-Agent') || 'unknown';
            
            // Check if IP is blocked
            if (this.blockedIPs.has(clientIP)) {
                this.log.warn(`Blocked IP attempted access: ${clientIP}`);
                return res.status(403).json({
                    error: 'Access denied',
                    code: 'IP_BLOCKED'
                });
            }
            
            // Check if user is quarantined
            if (req.user && this.quarantinedUsers.has(req.user.id)) {
                this.log.warn(`Quarantined user attempted access: ${req.user.id}`);
                return res.status(403).json({
                    error: 'Account temporarily restricted',
                    code: 'USER_QUARANTINED'
                });
            }
            
            // Monitor request for security issues
            this.monitorRequest(req, clientIP, userAgent, startTime);
            
            // Intercept response to monitor status codes
            const originalSend = res.send;
            res.send = function(data) {
                const responseTime = Date.now() - startTime;
                this.monitorResponse(req, res, responseTime, data);
                return originalSend.call(this, data);
            }.bind(this);
            
            next();
        };
    }

    /**
     * Monitor incoming requests for security threats
     */
    monitorRequest(req, clientIP, userAgent, startTime) {
        const requestData = {
            method: req.method,
            url: req.originalUrl,
            path: req.path,
            query: req.query,
            headers: this.sanitizeHeaders(req.headers),
            body: req.body,
            ipAddress: clientIP,
            userAgent: userAgent,
            userId: req.user?.id || null,
            timestamp: new Date().toISOString()
        };
        
        // Detect suspicious patterns in request
        const threats = this.detectRequestThreats(requestData);
        
        if (threats.length > 0) {
            threats.forEach(threat => {
                this.securityMonitor.reportSecurityEvent({
                    severity: threat.severity,
                    category: threat.category,
                    description: threat.description,
                    source: 'request-monitor',
                    ipAddress: clientIP,
                    userId: req.user?.id || null,
                    userAgent: userAgent,
                    endpoint: req.originalUrl,
                    method: req.method,
                    metadata: {
                        threat: threat,
                        requestData: this.sanitizeRequestData(requestData)
                    }
                });
            });
        }
        
        // Store request metadata for response analysis
        req._securityContext = {
            startTime,
            clientIP,
            userAgent,
            threats: threats.length
        };
    }

    /**
     * Monitor responses for security issues
     */
    monitorResponse(req, res, responseTime, responseData) {
        const context = req._securityContext || {};
        const statusCode = res.statusCode;
        
        // Monitor for security-related status codes
        if (this.isSecurityRelevantStatus(statusCode)) {
            const eventData = {
                severity: this.getStatusSeverity(statusCode),
                category: this.getStatusCategory(statusCode),
                description: `HTTP ${statusCode} response detected`,
                source: 'response-monitor',
                ipAddress: context.clientIP,
                userId: req.user?.id || null,
                userAgent: context.userAgent,
                endpoint: req.originalUrl,
                method: req.method,
                statusCode: statusCode,
                responseTime: responseTime,
                metadata: {
                    responseTime,
                    requestThreats: context.threats || 0
                }
            };
            
            this.securityMonitor.reportSecurityEvent(eventData);
        }
        
        // Monitor response time for potential DoS attacks
        if (responseTime > 5000 && context.threats > 0) {
            this.securityMonitor.reportSecurityEvent({
                severity: 'medium',
                category: 'PERFORMANCE',
                description: `Slow response time detected: ${responseTime}ms with ${context.threats} threats`,
                source: 'performance-monitor',
                ipAddress: context.clientIP,
                userId: req.user?.id || null,
                endpoint: req.originalUrl,
                metadata: { responseTime, threats: context.threats }
            });
        }
    }

    /**
     * Detect security threats in requests
     */
    detectRequestThreats(requestData) {
        const threats = [];
        const content = JSON.stringify(requestData).toLowerCase();
        
        // SQL Injection detection
        const sqlPatterns = [
            /union.*select/i,
            /drop.*table/i,
            /exec.*xp_/i,
            /waitfor.*delay/i,
            /benchmark\s*\(/i,
            /'.*or.*'.*=/i,
            /admin'--/i
        ];
        
        sqlPatterns.forEach(pattern => {
            if (pattern.test(content)) {
                threats.push({
                    type: 'SQL_INJECTION',
                    severity: 'critical',
                    category: 'INJECTION',
                    description: `SQL injection attempt detected: ${pattern.toString()}`,
                    pattern: pattern.toString()
                });
            }
        });
        
        // XSS detection
        const xssPatterns = [
            /<script.*>/i,
            /javascript:/i,
            /on\w+\s*=/i,
            /<iframe/i,
            /<object/i,
            /expression\s*\(/i
        ];
        
        xssPatterns.forEach(pattern => {
            if (pattern.test(content)) {
                threats.push({
                    type: 'XSS_ATTEMPT',
                    severity: 'high',
                    category: 'INJECTION',
                    description: `XSS attempt detected: ${pattern.toString()}`,
                    pattern: pattern.toString()
                });
            }
        });
        
        // Command injection detection
        const cmdPatterns = [
            /;\s*(cat|ls|pwd|whoami|id|uname)/i,
            /\|\s*(curl|wget|nc|netcat)/i,
            /`[^`]*`/,
            /\$\([^)]*\)/,
            /&&.*rm/i,
            /\|\|\s*rm/i
        ];
        
        cmdPatterns.forEach(pattern => {
            if (pattern.test(content)) {
                threats.push({
                    type: 'COMMAND_INJECTION',
                    severity: 'critical',
                    category: 'INJECTION',
                    description: `Command injection attempt detected: ${pattern.toString()}`,
                    pattern: pattern.toString()
                });
            }
        });
        
        // Path traversal detection
        const pathTraversalPatterns = [
            /\.\.\//,
            /\.\.\\\/,
            /%2e%2e%2f/i,
            /%2e%2e%5c/i,
            /\.\.%2f/i,
            /\.\.%5c/i
        ];
        
        pathTraversalPatterns.forEach(pattern => {
            if (pattern.test(content)) {
                threats.push({
                    type: 'PATH_TRAVERSAL',
                    severity: 'high',
                    category: 'INJECTION',
                    description: `Path traversal attempt detected: ${pattern.toString()}`,
                    pattern: pattern.toString()
                });
            }
        });
        
        // Suspicious user agents
        const maliciousUserAgents = [
            /sqlmap/i,
            /nmap/i,
            /nikto/i,
            /burpsuite/i,
            /w3af/i,
            /havij/i,
            /masscan/i
        ];
        
        maliciousUserAgents.forEach(pattern => {
            if (pattern.test(requestData.userAgent)) {
                threats.push({
                    type: 'MALICIOUS_USER_AGENT',
                    severity: 'high',
                    category: 'RECONNAISSANCE',
                    description: `Malicious user agent detected: ${requestData.userAgent}`,
                    pattern: pattern.toString()
                });
            }
        });
        
        // Rate limiting check
        const rateLimitKey = `${requestData.ipAddress}:${requestData.path}`;
        const rateLimitCheck = this.checkRateLimit(rateLimitKey);
        if (!rateLimitCheck.allowed) {
            threats.push({
                type: 'RATE_LIMIT_EXCEEDED',
                severity: 'medium',
                category: 'ABUSE',
                description: `Rate limit exceeded: ${rateLimitCheck.count} requests in ${rateLimitCheck.window}ms`,
                count: rateLimitCheck.count
            });
        }
        
        return threats;
    }

    /**
     * Rate limiting implementation
     */
    checkRateLimit(key, maxRequests = 100, windowMs = 60000) {
        const now = Date.now();
        
        if (!this.rateLimiters.has(key)) {
            this.rateLimiters.set(key, []);
        }
        
        const requests = this.rateLimiters.get(key);
        
        // Remove old requests outside the window
        const validRequests = requests.filter(timestamp => now - timestamp < windowMs);
        
        // Check if limit exceeded
        if (validRequests.length >= maxRequests) {
            return {
                allowed: false,
                count: validRequests.length,
                window: windowMs
            };
        }
        
        // Add current request
        validRequests.push(now);
        this.rateLimiters.set(key, validRequests);
        
        return {
            allowed: true,
            count: validRequests.length,
            remaining: maxRequests - validRequests.length
        };
    }

    /**
     * Check if status code is security relevant
     */
    isSecurityRelevantStatus(statusCode) {
        return [
            400, // Bad Request
            401, // Unauthorized
            403, // Forbidden
            404, // Not Found (potential reconnaissance)
            429, // Too Many Requests
            500, // Internal Server Error
            502, // Bad Gateway
            503, // Service Unavailable
        ].includes(statusCode);
    }

    /**
     * Get severity based on status code
     */
    getStatusSeverity(statusCode) {
        if (statusCode >= 500) return 'high';
        if (statusCode === 401 || statusCode === 403) return 'medium';
        if (statusCode === 429) return 'medium';
        return 'low';
    }

    /**
     * Get category based on status code
     */
    getStatusCategory(statusCode) {
        if (statusCode === 401) return 'AUTHENTICATION';
        if (statusCode === 403) return 'AUTHORIZATION';
        if (statusCode === 429) return 'RATE_LIMITING';
        if (statusCode >= 500) return 'SYSTEM_ERROR';
        if (statusCode === 404) return 'RECONNAISSANCE';
        return 'HTTP_ERROR';
    }

    /**
     * Get client IP address
     */
    getClientIP(req) {
        return req.ip ||
               req.connection.remoteAddress ||
               req.socket.remoteAddress ||
               (req.connection.socket ? req.connection.socket.remoteAddress : null) ||
               'unknown';
    }

    /**
     * Sanitize headers for logging (remove sensitive data)
     */
    sanitizeHeaders(headers) {
        const sanitized = { ...headers };
        const sensitiveHeaders = ['authorization', 'cookie', 'x-api-key', 'x-auth-token'];
        
        sensitiveHeaders.forEach(header => {
            if (sanitized[header]) {
                sanitized[header] = '[REDACTED]';
            }
        });
        
        return sanitized;
    }

    /**
     * Sanitize request data for logging
     */
    sanitizeRequestData(requestData) {
        const sanitized = { ...requestData };
        
        // Remove sensitive fields
        if (sanitized.body) {
            const sensitiveFields = ['password', 'token', 'secret', 'key', 'auth'];
            sensitiveFields.forEach(field => {
                if (sanitized.body[field]) {
                    sanitized.body[field] = '[REDACTED]';
                }
            });
        }
        
        return sanitized;
    }

    /**
     * Get security dashboard data
     */
    getDashboardData() {
        return this.securityMonitor ? this.securityMonitor.getSecurityDashboard() : null;
    }

    /**
     * Get active security alerts
     */
    getActiveAlerts() {
        return this.securityMonitor ? this.securityMonitor.getActiveAlerts() : [];
    }

    /**
     * Acknowledge a security alert
     */
    acknowledgeAlert(alertId, userId) {
        return this.securityMonitor ? this.securityMonitor.acknowledgeAlert(alertId, userId) : false;
    }

    /**
     * Resolve a security alert
     */
    resolveAlert(alertId, userId, resolution) {
        return this.securityMonitor ? this.securityMonitor.resolveAlert(alertId, userId, resolution) : false;
    }

    /**
     * Get security metrics
     */
    getSecurityMetrics() {
        return this.securityMonitor ? this.securityMonitor.getSecurityMetrics() : null;
    }

    /**
     * Manual security event reporting
     */
    reportSecurityEvent(eventData) {
        if (this.securityMonitor) {
            this.securityMonitor.reportSecurityEvent(eventData);
        }
    }

    /**
     * Authentication failure middleware
     */
    authenticationFailure() {
        return (req, res, next) => {
            const originalStatus = res.status;
            
            res.status = function(code) {
                if (code === 401 || code === 403) {
                    this.reportSecurityEvent({
                        severity: 'medium',
                        category: 'AUTHENTICATION',
                        description: `Authentication failure: HTTP ${code}`,
                        source: 'auth-middleware',
                        ipAddress: this.getClientIP(req),
                        userId: req.user?.id || null,
                        userAgent: req.get('User-Agent'),
                        endpoint: req.originalUrl,
                        method: req.method,
                        statusCode: code
                    });
                }
                return originalStatus.call(this, code);
            }.bind(this);
            
            next();
        };
    }

    /**
     * CSRF protection middleware
     */
    csrfProtection() {
        return (req, res, next) => {
            // Skip CSRF check for safe methods
            if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
                return next();
            }
            
            const token = req.headers['x-csrf-token'] || req.body._csrf;
            const sessionToken = req.session?.csrfToken;
            
            if (!token || !sessionToken || token !== sessionToken) {
                this.reportSecurityEvent({
                    severity: 'high',
                    category: 'CSRF',
                    description: 'CSRF token validation failed',
                    source: 'csrf-middleware',
                    ipAddress: this.getClientIP(req),
                    userId: req.user?.id || null,
                    userAgent: req.get('User-Agent'),
                    endpoint: req.originalUrl,
                    method: req.method
                });
                
                return res.status(403).json({
                    error: 'CSRF token validation failed',
                    code: 'CSRF_TOKEN_INVALID'
                });
            }
            
            next();
        };
    }

    /**
     * Input validation middleware
     */
    inputValidation() {
        return (req, res, next) => {
            // Check for malicious input patterns
            const threats = this.detectRequestThreats({
                query: req.query,
                body: req.body,
                params: req.params,
                headers: req.headers,
                userAgent: req.get('User-Agent'),
                ipAddress: this.getClientIP(req)
            });
            
            // Block requests with critical threats
            const criticalThreats = threats.filter(t => t.severity === 'critical');
            if (criticalThreats.length > 0) {
                this.reportSecurityEvent({
                    severity: 'critical',
                    category: 'INPUT_VALIDATION',
                    description: `Critical input validation failure: ${criticalThreats.map(t => t.type).join(', ')}`,
                    source: 'input-validation-middleware',
                    ipAddress: this.getClientIP(req),
                    userId: req.user?.id || null,
                    userAgent: req.get('User-Agent'),
                    endpoint: req.originalUrl,
                    method: req.method,
                    metadata: { threats: criticalThreats }
                });
                
                return res.status(400).json({
                    error: 'Input validation failed',
                    code: 'INVALID_INPUT_DETECTED'
                });
            }
            
            next();
        };
    }
}

// Export singleton instance
module.exports = new SecurityMiddleware();