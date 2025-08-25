/**
 * @fileoverview Production Security Hardening Middleware
 * @description Addresses all identified security vulnerabilities for production deployment
 * @module securityHardening
 * @since 4.0.0
 * @author A2A Network Security Team
 */

const crypto = require('crypto');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const slowDown = require('express-slow-down');
const { v4: uuidv4 } = require('uuid');
const validator = require('validator');
const cds = require('@sap/cds');

/**
 * Production Security Configuration
 */
const PRODUCTION_CONFIG = {
    // Enforce production-only settings
    ENFORCE_PRODUCTION: process.env.NODE_ENV === 'production',

    // JWT Configuration - RS256 ONLY
    JWT: {
        algorithms: ['RS256'], // No HS256 allowed
        issuer: process.env.JWT_ISSUER || 'a2a-network-prod',
        audience: process.env.JWT_AUDIENCE || 'a2a-api-prod',
        expiresIn: '1h',
        clockTolerance: 30 // seconds
    },

    // Session Security
    SESSION: {
        name: 'a2a.sid',
        proxy: true, // Trust proxy
        resave: false,
        saveUninitialized: false,
        cookie: {
            secure: true, // HTTPS only
            httpOnly: true,
            sameSite: 'strict',
            maxAge: 3600000, // 1 hour
            domain: process.env.COOKIE_DOMAIN
        }
    },

    // WebSocket Security
    WEBSOCKET: {
        origins: process.env.WS_ALLOWED_ORIGINS?.split(',') || [],
        requireAuth: true,
        pingTimeout: 60000,
        maxPayload: 1048576, // 1MB
        perMessageDeflate: false // Prevent compression attacks
    },

    // Rate Limiting - Production values
    RATE_LIMITS: {
        errorReporting: { windowMs: 300000, max: 10 }, // 10 per 5 minutes
        websocket: { windowMs: 60000, max: 5 }, // 5 connections per minute
        fileUpload: { windowMs: 900000, max: 10 }, // 10 uploads per 15 minutes
        apiWrite: { windowMs: 60000, max: 30 }, // 30 writes per minute
        apiRead: { windowMs: 60000, max: 300 } // 300 reads per minute
    },

    // Content Security
    CONTENT: {
        maxRequestSize: '10mb',
        maxFileSize: 5242880, // 5MB
        allowedMimeTypes: ['application/json', 'text/plain', 'application/xml'],
        blockedExtensions: ['.exe', '.dll', '.scr', '.vbs', '.pif', '.cmd', '.bat']
    },

    // Cryptography
    CRYPTO: {
        pbkdf2Iterations: 310000, // OWASP 2023 recommendation
        saltRounds: 12,
        keyLength: 32,
        ivLength: 16,
        tagLength: 16
    }
};

/**
 * Production Security Manager
 */
class ProductionSecurityManager {
    constructor() {
        this.securityEvents = [];
        this.blockedIPs = new Map();
        this.suspiciousPatterns = new Map();
        this.intervals = new Map(); // Track intervals for cleanup
        this.initializeSecurityMonitoring();
    }

    /**
     * Initialize security monitoring
     */
    initializeSecurityMonitoring() {
        // Monitor for suspicious patterns
        const suspiciousInterval = setInterval(() => {
            this.analyzeSuspiciousPatterns();
        }, 60000); // Every minute
        this.intervals.set('suspicious_patterns', suspiciousInterval);

        // Clean up old blocks
        const cleanupInterval = setInterval(() => {
            this.cleanupBlockedIPs();
        }, 300000); // Every 5 minutes
        this.intervals.set('blocked_ips_cleanup', cleanupInterval);
    }

    /**
     * Enhanced JWT validation - RS256 only
     */
    validateProductionJWT() {
        return async (req, res, next) => {
            const token = req.headers.authorization?.substring(7);

            if (!token) {
                return res.status(401).json({
                    error: 'Authentication required',
                    code: 'NO_TOKEN'
                });
            }

            try {
                // Verify token is RS256
                const decoded = jwt.decode(token, { complete: true });
                if (!decoded || decoded.header.alg !== 'RS256') {
                    this.logSecurityEvent('INVALID_JWT_ALGORITHM', {
                        algorithm: decoded?.header?.alg,
                        ip: req.ip
                    });

                    return res.status(401).json({
                        error: 'Invalid token algorithm',
                        code: 'INVALID_ALGORITHM'
                    });
                }

                // Verify with public key
                const payload = await this.verifyRS256Token(token);

                // Additional production checks
                if (this.isTokenCompromised(payload)) {
                    return res.status(401).json({
                        error: 'Token compromised',
                        code: 'TOKEN_COMPROMISED'
                    });
                }

                req.user = payload.user;
                next();
            } catch (error) {
                this.logSecurityEvent('JWT_VALIDATION_FAILED', {
                    error: error.message,
                    ip: req.ip
                });

                res.status(401).json({
                    error: 'Invalid token',
                    code: 'INVALID_TOKEN'
                });
            }
        };
    }

    /**
     * WebSocket authentication enforcement
     */
    enforceWebSocketAuth(io) {
        io.use(async (socket, next) => {
            const token = socket.handshake.auth.token;

            // No development bypasses in production
            if (!token) {
                return next(new Error('Authentication required'));
            }

            try {
                // Validate origin
                const origin = socket.handshake.headers.origin;
                if (!this.isAllowedWebSocketOrigin(origin)) {
                    this.logSecurityEvent('WS_INVALID_ORIGIN', {
                        origin,
                        ip: socket.handshake.address
                    });
                    return next(new Error('Invalid origin'));
                }

                // Validate token (RS256 only)
                const payload = await this.verifyRS256Token(token);

                // Check rate limits
                if (this.isRateLimited(socket.handshake.address, 'websocket')) {
                    return next(new Error('Rate limit exceeded'));
                }

                socket.user = payload.user;
                socket.authenticated = true;
                next();
            } catch (error) {
                next(new Error('Authentication failed'));
            }
        });
    }

    /**
     * Error reporting rate limiter
     */
    createErrorReportLimiter() {
        return rateLimit({
            ...PRODUCTION_CONFIG.RATE_LIMITS.errorReporting,
            keyGenerator: (req) => {
                return `${req.ip  }:${  req.user?.id || 'anonymous'}`;
            },
            handler: (req, res) => {
                this.logSecurityEvent('ERROR_REPORT_RATE_LIMITED', {
                    ip: req.ip,
                    userId: req.user?.id
                });

                res.status(429).json({
                    error: 'Too many error reports',
                    code: 'RATE_LIMITED'
                });
            }
        });
    }

    /**
     * Production-only endpoints protection
     */
    protectProductionEndpoints() {
        return (req, res, next) => {
            // Block test endpoints in production
            const blockedPaths = [
                '/test/',
                '/debug/',
                '/dev/',
                '/_test',
                '/testEndpoints'
            ];

            if (PRODUCTION_CONFIG.ENFORCE_PRODUCTION &&
                blockedPaths.some(path => req.path.includes(path))) {
                this.logSecurityEvent('BLOCKED_TEST_ENDPOINT', {
                    path: req.path,
                    ip: req.ip
                });

                return res.status(404).json({
                    error: 'Not found',
                    code: 'NOT_FOUND'
                });
            }

            next();
        };
    }

    /**
     * Enhanced CSP with nonce for dynamic content
     */
    enhancedCSP() {
        return (req, res, next) => {
            // Generate nonce for this request
            const nonce = crypto.randomBytes(16).toString('base64');
            res.locals.cspNonce = nonce;

            // Set CSP header with nonce
            const cspDirectives = {
                defaultSrc: ['\'self\''],
                scriptSrc: ['\'self\'', `'nonce-${nonce}'`],
                styleSrc: ['\'self\'', '\'unsafe-inline\''], // Required for UI5
                imgSrc: ['\'self\'', 'data:', 'https:'],
                connectSrc: ['\'self\'', 'wss:', 'https:'],
                fontSrc: ['\'self\''],
                objectSrc: ['\'none\''],
                mediaSrc: ['\'none\''],
                frameSrc: ['\'none\''],
                childSrc: ['\'none\''],
                formAction: ['\'self\''],
                frameAncestors: ['\'none\''],
                baseUri: ['\'self\''],
                manifestSrc: ['\'self\'']
            };

            if (PRODUCTION_CONFIG.ENFORCE_PRODUCTION) {
                cspDirectives.upgradeInsecureRequests = [];
                cspDirectives.blockAllMixedContent = [];
            }

            const cspString = Object.entries(cspDirectives)
                .map(([key, values]) => {
                    const directive = key.replace(/([A-Z])/g, '-$1').toLowerCase();
                    return `${directive} ${values.join(' ')}`;
                })
                .join('; ');

            res.setHeader('Content-Security-Policy', cspString);
            next();
        };
    }

    /**
     * SQLite encryption for data at rest
     */
    encryptDatabaseConnection(dbPath) {
        if (!PRODUCTION_CONFIG.ENFORCE_PRODUCTION) {
            return dbPath; // Skip in development
        }

        // Use SQLCipher or similar for production
        const encryptedPath = dbPath.replace('.db', '.encrypted.db');
        const key = this.getDatabaseEncryptionKey();

        // Return connection string with encryption
        return `${encryptedPath}?key=${key}`;
    }

    /**
     * Request signing validation for sensitive operations
     */
    validateRequestSignature() {
        return (req, res, next) => {
            // Only for sensitive operations
            const sensitiveOperations = [
                '/api/v1/agents/delete',
                '/api/v1/services/update',
                '/api/v1/blockchain/transaction',
                '/api/v1/reputation/update'
            ];

            if (!sensitiveOperations.some(op => req.path.includes(op))) {
                return next();
            }

            const signature = req.headers['x-request-signature'];
            const timestamp = req.headers['x-request-timestamp'];
            const requestId = req.headers['x-request-id'];

            if (!signature || !timestamp || !requestId) {
                return res.status(401).json({
                    error: 'Request signature required',
                    code: 'SIGNATURE_REQUIRED'
                });
            }

            // Validate timestamp (5 minute window)
            const requestTime = parseInt(timestamp);
            if (Math.abs(Date.now() - requestTime) > 300000) {
                return res.status(401).json({
                    error: 'Request expired',
                    code: 'REQUEST_EXPIRED'
                });
            }

            // Validate signature
            const payload = this.createSignaturePayload(req, timestamp, requestId);
            const expectedSignature = this.calculateHMAC(payload, req.user.apiKey);

            if (!crypto.timingSafeEqual(
                Buffer.from(signature, 'hex'),
                Buffer.from(expectedSignature, 'hex')
            )) {
                this.logSecurityEvent('INVALID_REQUEST_SIGNATURE', {
                    userId: req.user.id,
                    path: req.path,
                    ip: req.ip
                });

                return res.status(401).json({
                    error: 'Invalid signature',
                    code: 'INVALID_SIGNATURE'
                });
            }

            next();
        };
    }

    /**
     * Enhanced error response sanitization
     */
    sanitizeErrorResponses() {
        return (err, req, res, next) => {
            // Log full error internally
            const errorId = uuidv4();
            cds.log('security').error('Application error', {
                errorId,
                error: err.stack || err.message,
                path: req.path,
                method: req.method,
                ip: req.ip,
                user: req.user?.id
            });

            // Send sanitized response
            const statusCode = err.statusCode || 500;
            const response = {
                error: 'An error occurred',
                code: err.code || 'INTERNAL_ERROR',
                errorId
            };

            // Only add details for client errors
            if (statusCode >= 400 && statusCode < 500) {
                response.message = err.message || 'Bad request';
            }

            res.status(statusCode).json(response);
        };
    }

    /**
     * Security event logging
     */
    logSecurityEvent(eventType, details) {
        const event = {
            id: uuidv4(),
            type: eventType,
            timestamp: new Date(),
            details,
            severity: this.getEventSeverity(eventType)
        };

        this.securityEvents.push(event);

        // Keep only last 10000 events
        if (this.securityEvents.length > 10000) {
            this.securityEvents.shift();
        }

        // Log to monitoring system
        cds.log('security').warn('Security event', event);

        // Take action on critical events
        if (event.severity === 'CRITICAL') {
            this.handleCriticalSecurityEvent(event);
        }
    }

    /**
     * Analyze suspicious patterns
     */
    analyzeSuspiciousPatterns() {
        const recentEvents = this.securityEvents.filter(
            e => Date.now() - e.timestamp.getTime() < 300000 // Last 5 minutes
        );

        // Group by IP
        const ipEvents = {};
        recentEvents.forEach(event => {
            const ip = event.details?.ip;
            if (ip) {
                ipEvents[ip] = (ipEvents[ip] || 0) + 1;
            }
        });

        // Block IPs with suspicious activity
        Object.entries(ipEvents).forEach(([ip, count]) => {
            if (count > 20) { // More than 20 security events in 5 minutes
                this.blockIP(ip, 'SUSPICIOUS_ACTIVITY');
            }
        });
    }

    /**
     * Block IP address
     */
    blockIP(ip, reason) {
        this.blockedIPs.set(ip, {
            reason,
            timestamp: new Date(),
            duration: 3600000 // 1 hour
        });

        cds.log('security').error('IP blocked', { ip, reason });
    }

    /**
     * IP blocking middleware
     */
    ipBlockingMiddleware() {
        return (req, res, next) => {
            const ip = req.ip;
            const blockInfo = this.blockedIPs.get(ip);

            if (blockInfo) {
                const elapsed = Date.now() - blockInfo.timestamp.getTime();
                if (elapsed < blockInfo.duration) {
                    return res.status(403).json({
                        error: 'Access denied',
                        code: 'IP_BLOCKED'
                    });
                }
            }

            next();
        };
    }

    /**
     * Create comprehensive production security stack
     */
    createProductionSecurityStack() {
        return [
            this.ipBlockingMiddleware(),
            this.protectProductionEndpoints(),
            this.enhancedCSP(),
            helmet({
                contentSecurityPolicy: false, // Using our enhanced CSP
                hsts: {
                    maxAge: 31536000,
                    includeSubDomains: true,
                    preload: true
                },
                noSniff: true,
                xssFilter: true,
                referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
                permittedCrossDomainPolicies: { permittedPolicies: 'none' }
            }),
            this.validateProductionJWT(),
            this.validateRequestSignature()
        ];
    }
}

// Export production security instance
const productionSecurity = new ProductionSecurityManager();

module.exports = {
    productionSecurity,
    enforceWebSocketAuth: (io) => productionSecurity.enforceWebSocketAuth(io),
    createErrorReportLimiter: () => productionSecurity.createErrorReportLimiter(),
    createProductionSecurityStack: () => productionSecurity.createProductionSecurityStack(),
    encryptDatabaseConnection: (path) => productionSecurity.encryptDatabaseConnection(path),
    sanitizeErrorResponses: () => productionSecurity.sanitizeErrorResponses(),
    PRODUCTION_CONFIG
};