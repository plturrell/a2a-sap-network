/**
 * Security Hardening Middleware for A2A Network
 * Implements SAP security standards and best practices
 */

const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const slowDown = require('express-slow-down');
const validator = require('validator');
const DOMPurify = require('isomorphic-dompurify');
const crypto = require('crypto');

class SecurityHardening {
  constructor() {
    this.trustedDomains = [
      'sap.com',
      'hana.ondemand.com',
      'cfapps.sap.hana.ondemand.com'
    ];
    
    this.sensitiveFields = [
      'password', 'token', 'secret', 'key', 'credential',
      'authorization', 'x-api-key', 'client_secret'
    ];
  }

  /**
   * Apply comprehensive security headers
   */
  securityHeaders() {
    return helmet({
      // Content Security Policy
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'", 'https://ui5.sap.com'],
          scriptSrc: ["'self'", "'strict-dynamic'", 'https://ui5.sap.com'],
          imgSrc: ["'self'", 'data:', 'https:'],
          connectSrc: ["'self'", 'https://*.sap.com'],
          fontSrc: ["'self'", 'https://ui5.sap.com'],
          objectSrc: ["'none'"],
          mediaSrc: ["'self'"],
          frameSrc: ["'none'"],
          frameAncestors: ["'self'", '*.sap.com']
        }
      },

      // HTTP Strict Transport Security
      hsts: {
        maxAge: 31536000, // 1 year
        includeSubDomains: true,
        preload: true
      },

      // X-Frame-Options
      frameguard: { action: 'sameorigin' },

      // X-Content-Type-Options
      noSniff: true,

      // X-XSS-Protection
      xssFilter: true,

      // Referrer Policy
      referrerPolicy: {
        policy: ['same-origin']
      },

      // Feature Policy / Permissions Policy
      permittedCrossDomainPolicies: false,
      
      // Custom SAP security headers
      customHeaders: {
        'X-SAP-Security-Level': 'Enterprise',
        'X-SAP-Content-Classification': 'Restricted',
        'X-Powered-By-SAP': 'true'
      }
    });
  }

  /**
   * Rate limiting with different tiers
   */
  rateLimiting() {
    return {
      // Standard API rate limiting
      standard: rateLimit({
        windowMs: 15 * 60 * 1000, // 15 minutes
        max: 1000, // Limit each IP to 1000 requests per windowMs
        message: {
          error: {
            code: 'RATE_LIMIT_EXCEEDED',
            message: 'Too many requests, please try again later'
          }
        },
        standardHeaders: true,
        legacyHeaders: false,
        handler: (req, res) => {
          req.securityEvent = {
            type: 'RATE_LIMIT_EXCEEDED',
            severity: 'medium',
            details: { ip: req.ip, path: req.path }
          };
          res.status(429).json({
            error: {
              code: 'RATE_LIMIT_EXCEEDED',
              message: 'Rate limit exceeded. Please retry after some time.'
            }
          });
        }
      }),

      // Strict rate limiting for sensitive operations
      strict: rateLimit({
        windowMs: 15 * 60 * 1000,
        max: 100,
        message: {
          error: {
            code: 'STRICT_RATE_LIMIT_EXCEEDED',
            message: 'Too many sensitive operations, please try again later'
          }
        }
      }),

      // Authentication attempts
      auth: rateLimit({
        windowMs: 15 * 60 * 1000,
        max: 5,
        skipSuccessfulRequests: true,
        handler: (req, res) => {
          req.securityEvent = {
            type: 'AUTH_RATE_LIMIT_EXCEEDED',
            severity: 'high',
            details: { ip: req.ip, attempts: 5 }
          };
          res.status(429).json({
            error: {
              code: 'AUTH_RATE_LIMIT_EXCEEDED',
              message: 'Too many authentication attempts'
            }
          });
        }
      }),

      // Slow down repeated requests
      slowDown: slowDown({
        windowMs: 15 * 60 * 1000,
        delayAfter: 100,
        delayMs: 500,
        maxDelayMs: 20000
      })
    };
  }

  /**
   * Input validation and sanitization
   */
  inputValidation() {
    return (req, res, next) => {
      try {
        // Sanitize request body
        if (req.body && typeof req.body === 'object') {
          req.body = this.sanitizeObject(req.body);
        }

        // Validate critical fields
        this.validateCriticalInputs(req);

        // Check for potential injection attacks
        this.detectInjectionAttempts(req);

        next();
      } catch (error) {
        req.securityEvent = {
          type: 'INPUT_VALIDATION_FAILED',
          severity: 'high',
          details: { error: error.message, path: req.path }
        };
        
        res.status(400).json({
          error: {
            code: 'INVALID_INPUT',
            message: 'Input validation failed'
          }
        });
      }
    };
  }

  /**
   * Sanitize object recursively
   */
  sanitizeObject(obj) {
    if (Array.isArray(obj)) {
      return obj.map(item => this.sanitizeObject(item));
    }
    
    if (obj !== null && typeof obj === 'object') {
      const sanitized = {};
      for (const [key, value] of Object.entries(obj)) {
        if (typeof value === 'string') {
          sanitized[key] = DOMPurify.sanitize(value);
        } else {
          sanitized[key] = this.sanitizeObject(value);
        }
      }
      return sanitized;
    }
    
    return obj;
  }

  /**
   * Validate critical inputs
   */
  validateCriticalInputs(req) {
    if (req.body) {
      // Validate blockchain addresses
      if (req.body.address && !this.isValidBlockchainAddress(req.body.address)) {
        throw new Error('Invalid blockchain address format');
      }

      // Validate email addresses
      if (req.body.email && !validator.isEmail(req.body.email)) {
        throw new Error('Invalid email format');
      }

      // Validate URLs
      if (req.body.endpoint && !validator.isURL(req.body.endpoint)) {
        throw new Error('Invalid URL format');
      }

      // Check for oversized inputs
      const jsonString = JSON.stringify(req.body);
      if (jsonString.length > 1024 * 1024) { // 1MB limit
        throw new Error('Request payload too large');
      }
    }
  }

  /**
   * Detect injection attempts
   */
  detectInjectionAttempts(req) {
    const suspiciousPatterns = [
      /('|\\\\')|(;)|(--|-)|(\|)|(\*)|(%)|(<)|(>)|(\{)|(\})|(\\\\)|(\/)/gi,
      /(\\0)|(\\n)|(\\r)|(\\Z)|(\\x)|(\\')|(\")|(\\\\)/gi,
      /(exec|execute|drop|create|alter|insert|delete|update|select|union|script)/gi
    ];

    const checkString = JSON.stringify(req.body || {}) + JSON.stringify(req.query || {});
    
    for (const pattern of suspiciousPatterns) {
      if (pattern.test(checkString)) {
        throw new Error('Potential injection attack detected');
      }
    }
  }

  /**
   * Validate blockchain address format
   */
  isValidBlockchainAddress(address) {
    return /^0x[a-fA-F0-9]{40}$/.test(address);
  }

  /**
   * Request signing validation
   */
  requestSignature() {
    return (req, res, next) => {
      // Skip for GET requests
      if (req.method === 'GET') {
        return next();
      }

      const signature = req.headers['x-signature'];
      const timestamp = req.headers['x-timestamp'];

      if (!signature || !timestamp) {
        return res.status(401).json({
          error: {
            code: 'MISSING_SIGNATURE',
            message: 'Request signature required for this operation'
          }
        });
      }

      // Verify timestamp is within acceptable window (5 minutes)
      const now = Date.now();
      const requestTime = parseInt(timestamp);
      if (Math.abs(now - requestTime) > 300000) {
        req.securityEvent = {
          type: 'TIMESTAMP_OUT_OF_RANGE',
          severity: 'medium',
          details: { timestamp: requestTime, current: now }
        };
        
        return res.status(401).json({
          error: {
            code: 'INVALID_TIMESTAMP',
            message: 'Request timestamp out of acceptable range'
          }
        });
      }

      // Verify signature (implementation depends on your signing mechanism)
      if (!this.verifySignature(req, signature, timestamp)) {
        req.securityEvent = {
          type: 'INVALID_SIGNATURE',
          severity: 'high',
          details: { path: req.path }
        };
        
        return res.status(401).json({
          error: {
            code: 'INVALID_SIGNATURE',
            message: 'Request signature verification failed'
          }
        });
      }

      next();
    };
  }

  /**
   * Verify request signature
   */
  verifySignature(req, signature, timestamp) {
    try {
      const payload = req.method + req.originalUrl + timestamp + JSON.stringify(req.body || {});
      const secret = process.env.REQUEST_SIGNING_SECRET || 'default-secret';
      const expectedSignature = crypto.createHmac('sha256', secret).update(payload).digest('hex');
      
      return crypto.timingSafeEqual(
        Buffer.from(signature, 'hex'),
        Buffer.from(expectedSignature, 'hex')
      );
    } catch (error) {
      return false;
    }
  }

  /**
   * Response data filtering
   */
  responseFilter() {
    return (req, res, next) => {
      const originalJson = res.json;
      
      res.json = function(data) {
        // Filter sensitive data from responses
        const filteredData = this.filterSensitiveData(data);
        return originalJson.call(this, filteredData);
      }.bind(this);

      next();
    };
  }

  /**
   * Filter sensitive data from response
   */
  filterSensitiveData(data) {
    if (Array.isArray(data)) {
      return data.map(item => this.filterSensitiveData(item));
    }
    
    if (data !== null && typeof data === 'object') {
      const filtered = {};
      for (const [key, value] of Object.entries(data)) {
        if (this.sensitiveFields.some(field => key.toLowerCase().includes(field))) {
          filtered[key] = '[FILTERED]';
        } else {
          filtered[key] = this.filterSensitiveData(value);
        }
      }
      return filtered;
    }
    
    return data;
  }

  /**
   * CORS configuration for SAP environments
   */
  corsConfig() {
    return {
      origin: (origin, callback) => {
        // Allow requests from SAP trusted domains
        if (!origin || this.trustedDomains.some(domain => origin.includes(domain))) {
          callback(null, true);
        } else {
          callback(new Error('Not allowed by CORS'));
        }
      },
      credentials: true,
      optionsSuccessStatus: 200,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: [
        'Origin', 'X-Requested-With', 'Content-Type', 'Accept', 
        'Authorization', 'X-Correlation-ID', 'X-Signature', 'X-Timestamp'
      ],
      exposedHeaders: ['X-Total-Count', 'X-Cache-Status']
    };
  }

  /**
   * Security monitoring middleware
   */
  securityMonitoring() {
    return (req, res, next) => {
      // Log security events if they were detected
      if (req.securityEvent) {
        const logger = require('./enterprise-logging');
        logger.logSecurityEvent(
          req.securityEvent.type,
          req.securityEvent,
          req
        );
      }

      next();
    };
  }

  /**
   * Generate Content Security Policy nonce
   */
  generateNonce() {
    return (req, res, next) => {
      res.locals.nonce = crypto.randomBytes(16).toString('base64');
      next();
    };
  }
}

module.exports = new SecurityHardening();