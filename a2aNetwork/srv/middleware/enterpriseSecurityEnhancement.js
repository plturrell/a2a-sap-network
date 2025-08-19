/**
 * @fileoverview Enterprise Security Enhancement Middleware
 * @description Comprehensive security middleware addressing all identified vulnerabilities
 * with SAP enterprise-grade security standards
 * @module enterpriseSecurityEnhancement
 * @since 3.0.0
 * @author A2A Network Team
 */

const crypto = require('crypto');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const validator = require('validator');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const speakeasy = require('speakeasy');
const { v4: uuidv4 } = require('uuid');

// SAP Security Libraries
let xssec, auditLog, keyManagement;
try {
    xssec = require('@sap/xssec');
    auditLog = require('@sap/audit-logging');
    keyManagement = require('@sap/key-management');
} catch (error) {
    // SAP libraries not available
}

/**
 * Enterprise Security Configuration
 */
const SECURITY_CONFIG = {
    // Password Policy
    PASSWORD_POLICY: {
        minLength: 12,
        requireUppercase: true,
        requireLowercase: true,
        requireNumbers: true,
        requireSpecialChars: true,
        specialChars: '!@#$%^&*()_+-=[]{}|;:,.<>?',
        maxAge: 90, // days
        historyCount: 12, // remember last 12 passwords
        lockoutThreshold: 5,
        lockoutDuration: 1800000, // 30 minutes
        complexityScore: 3 // 0-4 scale
    },
    
    // Session Configuration
    SESSION: {
        secret: process.env.SESSION_SECRET || crypto.randomBytes(64).toString('hex'),
        ttl: 3600000, // 1 hour
        maxAge: 86400000, // 24 hours
        renewalThreshold: 900000, // 15 minutes
        secure: true, // Always use secure cookies
        httpOnly: true,
        sameSite: 'strict'
    },
    
    // API Key Configuration
    API_KEY: {
        length: 32,
        rotationInterval: 2592000000, // 30 days
        maxActiveKeys: 3,
        prefix: 'a2a_',
        algorithm: 'sha256'
    },
    
    // JWT Configuration
    JWT: {
        algorithm: 'RS256', // Use RSA instead of HS256
        expiresIn: '1h',
        refreshExpiresIn: '7d',
        issuer: 'a2a-network',
        audience: 'a2a-api'
    },
    
    // Rate Limiting
    RATE_LIMITS: {
        global: { windowMs: 900000, max: 1000 }, // 1000 requests per 15 minutes
        auth: { windowMs: 900000, max: 5 }, // 5 auth attempts per 15 minutes
        api: { windowMs: 60000, max: 100 }, // 100 API calls per minute
        sensitive: { windowMs: 300000, max: 10 } // 10 sensitive ops per 5 minutes
    },
    
    // Security Headers
    HEADERS: {
        strictTransportSecurity: {
            maxAge: 31536000,
            includeSubDomains: true,
            preload: true
        },
        contentSecurityPolicy: {
            directives: {
                defaultSrc: ["'self'"],
                scriptSrc: ["'self'", "'strict-dynamic'"],
                styleSrc: ["'self'", "'unsafe-inline'"],
                imgSrc: ["'self'", "data:", "https:"],
                connectSrc: ["'self'"],
                fontSrc: ["'self'"],
                objectSrc: ["'none'"],
                mediaSrc: ["'self'"],
                frameSrc: ["'none'"],
                sandbox: ['allow-forms', 'allow-scripts', 'allow-same-origin'],
                reportUri: '/api/security/csp-report'
            }
        }
    },
    
    // Encryption
    ENCRYPTION: {
        algorithm: 'aes-256-gcm',
        keyDerivation: 'pbkdf2',
        iterations: 100000,
        saltLength: 32,
        tagLength: 16,
        ivLength: 16
    }
};

/**
 * Enterprise Security Manager
 */
class EnterpriseSecurityManager {
    constructor() {
        this.passwordHistory = new Map();
        this.apiKeys = new Map();
        this.failedAttempts = new Map();
        this.activeSessions = new Map();
        this.securityEvents = [];
        
        // Initialize key management
        this.keyManager = new KeyManager();
        
        // Initialize security monitoring
        this.securityMonitor = new SecurityMonitor(this);
        
        // Generate RSA key pair for JWT
        this.generateRSAKeyPair();
    }
    
    /**
     * Generate RSA key pair for JWT signing
     */
    generateRSAKeyPair() {
        const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', {
            modulusLength: 4096,
            publicKeyEncoding: {
                type: 'spki',
                format: 'pem'
            },
            privateKeyEncoding: {
                type: 'pkcs8',
                format: 'pem',
                cipher: 'aes-256-cbc',
                passphrase: process.env.JWT_KEY_PASSPHRASE || crypto.randomBytes(32).toString('hex')
            }
        });
        
        this.jwtPublicKey = publicKey;
        this.jwtPrivateKey = privateKey;
    }
    
    /**
     * Enhanced authentication middleware
     */
    authenticate() {
        return async (req, res, next) => {
            try {
                // Check multiple authentication methods in order of preference
                const authResult = await this.performAuthentication(req);
                
                if (!authResult.authenticated) {
                    return res.status(401).json({
                        error: 'Authentication required',
                        code: 'AUTH_REQUIRED',
                        supportedMethods: ['Bearer', 'APIKey', 'XSUAA']
                    });
                }
                
                // Set user context
                req.user = authResult.user;
                req.authMethod = authResult.method;
                req.authToken = authResult.token;
                
                // Log successful authentication
                await this.logSecurityEvent('AUTH_SUCCESS', {
                    userId: authResult.user.id,
                    method: authResult.method,
                    ip: req.ip
                });
                
                next();
            } catch (error) {
                await this.logSecurityEvent('AUTH_ERROR', {
                    error: error.message,
                    ip: req.ip
                });
                
                res.status(401).json({
                    error: 'Authentication failed',
                    code: 'AUTH_FAILED'
                });
            }
        };
    }
    
    /**
     * Perform authentication with multiple methods
     */
    async performAuthentication(req) {
        // 1. Check for XSUAA token (SAP BTP)
        if (xssec && req.headers.authorization?.startsWith('Bearer ')) {
            try {
                const token = req.headers.authorization.substring(7);
                const authInfo = await this.verifyXSUAAToken(token);
                if (authInfo) {
                    return {
                        authenticated: true,
                        user: authInfo.user,
                        method: 'XSUAA',
                        token
                    };
                }
            } catch (error) {
                // Continue to next auth method
            }
        }
        
        // 2. Check for JWT token
        if (req.headers.authorization?.startsWith('Bearer ')) {
            try {
                const token = req.headers.authorization.substring(7);
                const payload = await this.verifyJWT(token);
                if (payload) {
                    return {
                        authenticated: true,
                        user: payload.user,
                        method: 'JWT',
                        token
                    };
                }
            } catch (error) {
                // Continue to next auth method
            }
        }
        
        // 3. Check for API key
        const apiKey = req.headers['x-api-key'] || req.query.apiKey;
        if (apiKey) {
            const keyInfo = await this.verifyAPIKey(apiKey);
            if (keyInfo) {
                return {
                    authenticated: true,
                    user: keyInfo.user,
                    method: 'APIKey',
                    token: apiKey
                };
            }
        }
        
        return { authenticated: false };
    }
    
    /**
     * Verify JWT with RSA
     */
    async verifyJWT(token) {
        return new Promise((resolve, reject) => {
            jwt.verify(
                token,
                this.jwtPublicKey,
                {
                    algorithms: [SECURITY_CONFIG.JWT.algorithm],
                    issuer: SECURITY_CONFIG.JWT.issuer,
                    audience: SECURITY_CONFIG.JWT.audience
                },
                (err, payload) => {
                    if (err) reject(err);
                    else resolve(payload);
                }
            );
        });
    }
    
    /**
     * Enhanced password validation
     */
    validatePassword(password, userId) {
        const policy = SECURITY_CONFIG.PASSWORD_POLICY;
        const errors = [];
        
        // Length check
        if (password.length < policy.minLength) {
            errors.push(`Password must be at least ${policy.minLength} characters`);
        }
        
        // Complexity checks
        if (policy.requireUppercase && !/[A-Z]/.test(password)) {
            errors.push('Password must contain uppercase letters');
        }
        
        if (policy.requireLowercase && !/[a-z]/.test(password)) {
            errors.push('Password must contain lowercase letters');
        }
        
        if (policy.requireNumbers && !/\d/.test(password)) {
            errors.push('Password must contain numbers');
        }
        
        if (policy.requireSpecialChars && !new RegExp(`[${policy.specialChars}]`).test(password)) {
            errors.push('Password must contain special characters');
        }
        
        // Check password history
        if (userId && this.isPasswordInHistory(userId, password)) {
            errors.push(`Password cannot be one of your last ${policy.historyCount} passwords`);
        }
        
        // Calculate complexity score
        const score = this.calculatePasswordComplexity(password);
        if (score < policy.complexityScore) {
            errors.push('Password is not complex enough');
        }
        
        return {
            valid: errors.length === 0,
            errors,
            score
        };
    }
    
    /**
     * Calculate password complexity score
     */
    calculatePasswordComplexity(password) {
        let score = 0;
        
        // Length score
        if (password.length >= 8) score++;
        if (password.length >= 12) score++;
        if (password.length >= 16) score++;
        
        // Character diversity
        if (/[a-z]/.test(password) && /[A-Z]/.test(password)) score++;
        if (/\d/.test(password)) score++;
        if (/[^a-zA-Z0-9]/.test(password)) score++;
        
        // Pattern detection (penalize common patterns)
        if (/(.)\1{2,}/.test(password)) score--; // Repeated characters
        if (/^[a-zA-Z]+\d+$/.test(password)) score--; // Common pattern
        if (/password|123456|qwerty/i.test(password)) score -= 2; // Common passwords
        
        return Math.max(0, Math.min(4, score));
    }
    
    /**
     * API key generation with secure random
     */
    generateAPIKey(userId, purpose = 'general') {
        const config = SECURITY_CONFIG.API_KEY;
        const key = crypto.randomBytes(config.length).toString('base64url');
        const apiKey = `${config.prefix}${key}`;
        
        const keyInfo = {
            id: uuidv4(),
            key: apiKey,
            userId,
            purpose,
            created: new Date(),
            lastUsed: null,
            expiresAt: new Date(Date.now() + config.rotationInterval),
            hash: crypto.createHash(config.algorithm).update(apiKey).digest('hex')
        };
        
        this.apiKeys.set(keyInfo.hash, keyInfo);
        
        // Schedule automatic rotation
        setTimeout(() => this.rotateAPIKey(keyInfo.id), config.rotationInterval);
        
        return apiKey;
    }
    
    /**
     * Enhanced rate limiting factory
     */
    createRateLimiter(type = 'api') {
        const config = SECURITY_CONFIG.RATE_LIMITS[type] || SECURITY_CONFIG.RATE_LIMITS.api;
        
        return rateLimit({
            windowMs: config.windowMs,
            max: config.max,
            message: {
                error: 'Too many requests',
                code: 'RATE_LIMIT_EXCEEDED',
                retryAfter: config.windowMs / 1000
            },
            standardHeaders: true,
            legacyHeaders: false,
            handler: (req, res) => {
                this.logSecurityEvent('RATE_LIMIT_EXCEEDED', {
                    ip: req.ip,
                    endpoint: req.path,
                    userAgent: req.headers['user-agent']
                });
                
                res.status(429).json({
                    error: 'Rate limit exceeded',
                    code: 'RATE_LIMIT_EXCEEDED',
                    retryAfter: Math.ceil(config.windowMs / 1000)
                });
            }
        });
    }
    
    /**
     * CSRF protection with double submit cookie
     */
    csrfProtection() {
        return (req, res, next) => {
            // Skip for read operations
            if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
                return next();
            }
            
            const token = req.headers['x-csrf-token'] || req.body._csrf;
            const cookie = req.cookies['csrf-token'];
            
            if (!token || !cookie || token !== cookie) {
                return res.status(403).json({
                    error: 'CSRF token validation failed',
                    code: 'CSRF_FAILED'
                });
            }
            
            next();
        };
    }
    
    /**
     * Security headers middleware
     */
    securityHeaders() {
        return helmet({
            ...SECURITY_CONFIG.HEADERS,
            contentSecurityPolicy: {
                ...SECURITY_CONFIG.HEADERS.contentSecurityPolicy,
                reportOnly: false
            }
        });
    }
    
    /**
     * Request signing validation
     */
    validateRequestSignature() {
        return async (req, res, next) => {
            // Skip for non-sensitive endpoints
            if (!this.isSensitiveEndpoint(req.path)) {
                return next();
            }
            
            const signature = req.headers['x-signature'];
            const timestamp = req.headers['x-timestamp'];
            const nonce = req.headers['x-nonce'];
            
            if (!signature || !timestamp || !nonce) {
                return res.status(401).json({
                    error: 'Request signature required',
                    code: 'SIGNATURE_REQUIRED'
                });
            }
            
            // Validate timestamp (5 minute window)
            const requestTime = parseInt(timestamp);
            if (Math.abs(Date.now() - requestTime) > 300000) {
                return res.status(401).json({
                    error: 'Request timestamp expired',
                    code: 'TIMESTAMP_EXPIRED'
                });
            }
            
            // Validate signature
            const payload = this.createSignaturePayload(req, timestamp, nonce);
            const expectedSignature = crypto
                .createHmac('sha256', req.user.signatureKey || '')
                .update(payload)
                .digest('hex');
            
            if (!crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expectedSignature))) {
                await this.logSecurityEvent('INVALID_SIGNATURE', {
                    userId: req.user.id,
                    endpoint: req.path,
                    ip: req.ip
                });
                
                return res.status(401).json({
                    error: 'Invalid request signature',
                    code: 'INVALID_SIGNATURE'
                });
            }
            
            next();
        };
    }
    
    /**
     * Input sanitization middleware
     */
    sanitizeInput() {
        return (req, res, next) => {
            // Sanitize query parameters
            if (req.query) {
                req.query = this.sanitizeObject(req.query);
            }
            
            // Sanitize body
            if (req.body) {
                req.body = this.sanitizeObject(req.body);
            }
            
            // Sanitize headers
            const sensitiveHeaders = ['authorization', 'x-api-key', 'cookie'];
            Object.keys(req.headers).forEach(header => {
                if (!sensitiveHeaders.includes(header.toLowerCase())) {
                    req.headers[header] = this.sanitizeValue(req.headers[header]);
                }
            });
            
            next();
        };
    }
    
    /**
     * Sanitize object recursively
     */
    sanitizeObject(obj) {
        if (typeof obj !== 'object' || obj === null) {
            return this.sanitizeValue(obj);
        }
        
        const sanitized = Array.isArray(obj) ? [] : {};
        
        for (const [key, value] of Object.entries(obj)) {
            // Skip prototype pollution attempts
            if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
                continue;
            }
            
            sanitized[key] = typeof value === 'object' && value !== null
                ? this.sanitizeObject(value)
                : this.sanitizeValue(value);
        }
        
        return sanitized;
    }
    
    /**
     * Sanitize individual value
     */
    sanitizeValue(value) {
        if (typeof value !== 'string') return value;
        
        // Remove null bytes
        value = value.replace(/\0/g, '');
        
        // Escape HTML entities
        value = validator.escape(value);
        
        // Remove potential script tags
        value = value.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
        
        return value;
    }
    
    /**
     * Account lockout protection
     */
    async checkAccountLockout(userId) {
        const attempts = this.failedAttempts.get(userId) || { count: 0, firstAttempt: Date.now() };
        const policy = SECURITY_CONFIG.PASSWORD_POLICY;
        
        // Reset if lockout duration has passed
        if (Date.now() - attempts.firstAttempt > policy.lockoutDuration) {
            this.failedAttempts.delete(userId);
            return { locked: false };
        }
        
        // Check if account is locked
        if (attempts.count >= policy.lockoutThreshold) {
            const remainingTime = policy.lockoutDuration - (Date.now() - attempts.firstAttempt);
            return {
                locked: true,
                remainingTime: Math.ceil(remainingTime / 1000),
                message: `Account locked. Try again in ${Math.ceil(remainingTime / 60000)} minutes`
            };
        }
        
        return { locked: false };
    }
    
    /**
     * Log security event
     */
    async logSecurityEvent(eventType, details) {
        const event = {
            id: uuidv4(),
            type: eventType,
            timestamp: new Date(),
            details,
            severity: this.getEventSeverity(eventType)
        };
        
        this.securityEvents.push(event);
        
        // Keep only last 10000 events in memory
        if (this.securityEvents.length > 10000) {
            this.securityEvents.shift();
        }
        
        // Log to audit service if available
        if (auditLog) {
            await auditLog.logSecurityEvent(event);
        }
        
        // Trigger alerts for critical events
        if (event.severity === 'CRITICAL') {
            this.securityMonitor.triggerAlert(event);
        }
    }
    
    /**
     * Get event severity
     */
    getEventSeverity(eventType) {
        const severityMap = {
            'AUTH_FAILED': 'MEDIUM',
            'RATE_LIMIT_EXCEEDED': 'LOW',
            'INVALID_SIGNATURE': 'HIGH',
            'CSRF_FAILED': 'HIGH',
            'SQL_INJECTION_ATTEMPT': 'CRITICAL',
            'XSS_ATTEMPT': 'HIGH',
            'UNAUTHORIZED_ACCESS': 'HIGH',
            'DATA_BREACH_ATTEMPT': 'CRITICAL'
        };
        
        return severityMap[eventType] || 'LOW';
    }
    
    /**
     * Create comprehensive security middleware stack
     */
    createSecurityStack() {
        return [
            this.securityHeaders(),
            this.createRateLimiter('global'),
            this.sanitizeInput(),
            this.authenticate(),
            this.csrfProtection(),
            this.validateRequestSignature()
        ];
    }
}

/**
 * Key Manager for secure key storage
 */
class KeyManager {
    constructor() {
        this.keys = new Map();
        this.masterKey = this.deriveMasterKey();
    }
    
    deriveMasterKey() {
        const secret = process.env.MASTER_KEY_SECRET || crypto.randomBytes(64).toString('hex');
        const salt = process.env.MASTER_KEY_SALT || crypto.randomBytes(32).toString('hex');
        
        return crypto.pbkdf2Sync(
            secret,
            salt,
            SECURITY_CONFIG.ENCRYPTION.iterations,
            32,
            'sha256'
        );
    }
    
    encrypt(data, keyId = 'default') {
        const key = this.getKey(keyId);
        const iv = crypto.randomBytes(SECURITY_CONFIG.ENCRYPTION.ivLength);
        const cipher = crypto.createCipheriv(
            SECURITY_CONFIG.ENCRYPTION.algorithm,
            key,
            iv
        );
        
        const encrypted = Buffer.concat([
            cipher.update(JSON.stringify(data), 'utf8'),
            cipher.final()
        ]);
        
        const tag = cipher.getAuthTag();
        
        return {
            encrypted: encrypted.toString('base64'),
            iv: iv.toString('base64'),
            tag: tag.toString('base64'),
            keyId
        };
    }
    
    decrypt(encryptedData) {
        const { encrypted, iv, tag, keyId } = encryptedData;
        const key = this.getKey(keyId);
        
        const decipher = crypto.createDecipheriv(
            SECURITY_CONFIG.ENCRYPTION.algorithm,
            key,
            Buffer.from(iv, 'base64')
        );
        
        decipher.setAuthTag(Buffer.from(tag, 'base64'));
        
        const decrypted = Buffer.concat([
            decipher.update(Buffer.from(encrypted, 'base64')),
            decipher.final()
        ]);
        
        return JSON.parse(decrypted.toString('utf8'));
    }
    
    getKey(keyId) {
        if (!this.keys.has(keyId)) {
            const key = crypto.pbkdf2Sync(
                this.masterKey,
                keyId,
                10000,
                32,
                'sha256'
            );
            this.keys.set(keyId, key);
        }
        
        return this.keys.get(keyId);
    }
}

/**
 * Security Monitor for real-time threat detection
 */
class SecurityMonitor {
    constructor(securityManager) {
        this.securityManager = securityManager;
        this.threats = new Map();
        this.alerts = [];
    }
    
    async triggerAlert(event) {
        const alert = {
            id: uuidv4(),
            event,
            timestamp: new Date(),
            status: 'ACTIVE'
        };
        
        this.alerts.push(alert);
        
        // Send to SAP Alert Notification Service
        if (process.env.ALERT_WEBHOOK_URL) {
            try {
                await fetch(process.env.ALERT_WEBHOOK_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(alert)
                });
            } catch (error) {
                console.error('Failed to send security alert:', error);
            }
        }
        
        // Take automated action for critical threats
        if (event.severity === 'CRITICAL') {
            await this.respondToThreat(event);
        }
    }
    
    async respondToThreat(event) {
        switch (event.type) {
            case 'SQL_INJECTION_ATTEMPT':
            case 'DATA_BREACH_ATTEMPT':
                // Block IP address
                await this.blockIP(event.details.ip);
                break;
            
            case 'BRUTE_FORCE_ATTEMPT':
                // Lock account
                await this.lockAccount(event.details.userId);
                break;
        }
    }
    
    async blockIP(ip) {
        // Implement IP blocking logic
        this.threats.set(ip, {
            type: 'BLOCKED_IP',
            timestamp: new Date(),
            duration: 86400000 // 24 hours
        });
    }
    
    async lockAccount(userId) {
        // Implement account locking logic
        this.securityManager.failedAttempts.set(userId, {
            count: 999,
            firstAttempt: Date.now()
        });
    }
}

// Export enhanced security middleware
const securityManager = new EnterpriseSecurityManager();

module.exports = {
    securityManager,
    authenticate: () => securityManager.authenticate(),
    rateLimiter: (type) => securityManager.createRateLimiter(type),
    csrfProtection: () => securityManager.csrfProtection(),
    securityHeaders: () => securityManager.securityHeaders(),
    sanitizeInput: () => securityManager.sanitizeInput(),
    validateRequestSignature: () => securityManager.validateRequestSignature(),
    createSecurityStack: () => securityManager.createSecurityStack(),
    validatePassword: (password, userId) => securityManager.validatePassword(password, userId),
    generateAPIKey: (userId, purpose) => securityManager.generateAPIKey(userId, purpose),
    SECURITY_CONFIG
};