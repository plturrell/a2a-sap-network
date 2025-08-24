'use strict';

/**
 * Enterprise Security Middleware for SAP A2A Developer Portal
 * Comprehensive security configuration following SAP Enterprise standards
 */

const helmet = require('helmet');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const slowDown = require('express-slow-down');
const { body, validationResult, param, query } = require('express-validator');
const crypto = require('crypto');
const xss = require('xss');
const hpp = require('hpp');
const compression = require('compression');

class EnterpriseSecurityMiddleware {
  constructor() {
    this.securityConfig = {
      // Content Security Policy
      csp: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: [
            "'self'", 
            "'unsafe-inline'", 
            'https://ui5.sap.com',
            'https://sapui5.hana.ondemand.com'
          ],
          scriptSrc: [
            "'self'", 
            "'unsafe-inline'", // Required for SAP UI5
            'https://ui5.sap.com',
            'https://sapui5.hana.ondemand.com',
            'https://openui5.hana.ondemand.com'
          ],
          imgSrc: [
            "'self'", 
            'data:', 
            'https:',
            'https://ui5.sap.com'
          ],
          connectSrc: [
            "'self'",
            'https://api.sap.com',
            'https://*.s4hana.cloud.sap',
            'https://*.hana.ondemand.com',
            'wss:',
            'ws:'
          ],
          fontSrc: [
            "'self'",
            'https://ui5.sap.com',
            'https://fonts.googleapis.com',
            'https://fonts.gstatic.com'
          ],
          frameSrc: ["'none'"],
          objectSrc: ["'none'"],
          mediaSrc: ["'self'"],
          workerSrc: ["'self'", 'blob:'],
          childSrc: ["'self'"],
          formAction: ["'self'"],
          frameAncestors: ["'none'"],
          baseUri: ["'self'"],
          manifestSrc: ["'self'"]
        },
        reportOnly: process.env.NODE_ENV !== 'production'
      },
            
      // CORS Configuration
      cors: {
        origin: (origin, callback) => {
          const allowedOrigins = [
            'https://a2a.sap.com',
            'https://*.cfapps.sap.hana.ondemand.com',
            'https://*.hana.ondemand.com',
            'http://localhost:4004',
            'http://localhost:3000'
          ];
                    
          if (!origin || this._isOriginAllowed(origin, allowedOrigins)) {
            callback(null, true);
          } else {
            callback(new Error('CORS policy violation'));
          }
        },
        credentials: true,
        methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
        allowedHeaders: [
          'Content-Type',
          'Authorization',
          'X-Requested-With',
          'X-Correlation-Id',
          'X-Trace-Id',
          'X-CSRF-Token',
          'Accept',
          'Accept-Language',
          'Accept-Encoding'
        ],
        exposedHeaders: [
          'X-Request-Id',
          'X-Rate-Limit-Limit',
          'X-Rate-Limit-Remaining',
          'X-Rate-Limit-Reset'
        ],
        maxAge: 86400 // 24 hours
      },
            
      // Rate Limiting Configuration
      rateLimiting: {
        // General API rate limiting
        general: {
          windowMs: 15 * 60 * 1000, // 15 minutes
          max: 1000, // requests per window
          message: {
            error: 'Too many requests from this IP',
            retryAfter: '15 minutes'
          },
          standardHeaders: true,
          legacyHeaders: false,
          skipSuccessfulRequests: false,
          skipFailedRequests: false,
          keyGenerator: (req) => {
            return `${req.ip  }:${  (req.get('User-Agent') || '').substring(0, 255)}`;
          }
        },
                
        // Stricter limits for authentication endpoints
        auth: {
          windowMs: 15 * 60 * 1000,
          max: 10,
          message: {
            error: 'Too many authentication attempts',
            retryAfter: '15 minutes'
          },
          skipSuccessfulRequests: true
        },
                
        // API-specific limits
        api: {
          windowMs: 1 * 60 * 1000, // 1 minute
          max: 100,
          message: {
            error: 'API rate limit exceeded',
            retryAfter: '1 minute'
          }
        }
      },
            
      // Slow down configuration
      slowDown: {
        windowMs: 15 * 60 * 1000,
        delayAfter: 100,
        delayMs: 500,
        maxDelayMs: 20000,
        skipFailedRequests: false,
        skipSuccessfulRequests: false
      }
    };
  }

  /**
     * Initialize all security middleware
     */
  initialize(app) {
    // Trust proxy for accurate IP addresses
    app.set('trust proxy', 1);
        
    // Request ID middleware
    app.use(this.requestIdMiddleware());
        
    // Security headers
    app.use(this.helmetConfiguration());
        
    // CORS configuration
    app.use(this.corsConfiguration());
        
    // Rate limiting
    app.use(this.rateLimitingMiddleware());
        
    // Request size limits
    app.use(this.requestSizeLimits());
        
    // Input sanitization
    app.use(this.inputSanitization());
        
    // Parameter pollution protection
    app.use(this.parameterPollutionProtection());
        
    // Compression with security considerations
    app.use(this.compressionMiddleware());
        
    // Request logging for security monitoring
    app.use(this.securityLoggingMiddleware());
        
    // CSRF protection (for non-API routes)
    // app.use(this.csrfProtection()); // Uncomment if needed
        
         
        
    // eslint-disable-next-line no-console
        
         
        
    // eslint-disable-next-line no-console
    console.log('Enterprise security middleware initialized');
  }

  /**
     * Helmet security headers configuration
     */
  helmetConfiguration() {
    return helmet({
      contentSecurityPolicy: this.securityConfig.csp,
            
      // HSTS - Force HTTPS
      hsts: {
        maxAge: 31536000, // 1 year
        includeSubDomains: true,
        preload: true
      },
            
      // Hide X-Powered-By header
      hidePoweredBy: true,
            
      // X-Frame-Options
      frameguard: {
        action: 'deny'
      },
            
      // X-Content-Type-Options
      noSniff: true,
            
      // X-XSS-Protection
      xssFilter: true,
            
      // Referrer Policy
      referrerPolicy: {
        policy: 'strict-origin-when-cross-origin'
      },
            
      // Permissions Policy
      permissionsPolicy: {
        features: {
          camera: [],
          microphone: [],
          geolocation: [],
          payment: [],
          usb: []
        }
      },
            
      // Expect-CT header
      expectCt: {
        maxAge: 86400,
        enforce: true
      }
    });
  }

  /**
     * CORS configuration
     */
  corsConfiguration() {
    return cors(this.securityConfig.cors);
  }

  /**
     * Rate limiting middleware
     */
  rateLimitingMiddleware() {
    const generalLimiter = rateLimit(this.securityConfig.rateLimiting.general);
    const authLimiter = rateLimit(this.securityConfig.rateLimiting.auth);
    const apiLimiter = rateLimit(this.securityConfig.rateLimiting.api);
    const slowDownMiddleware = slowDown(this.securityConfig.slowDown);
        
    return (req, _res, _next) => {
      // Apply different limits based on path
      if (req.path.startsWith('/auth/')) {
        authLimiter(req, _res, _next);
      } else if (req.path.startsWith('/api/')) {
        apiLimiter(req, _res, () => {
          slowDownMiddleware(req, _res, _next);
        });
      } else {
        generalLimiter(req, _res, _next);
      }
    };
  }

  /**
     * Request size limits
     */
  requestSizeLimits() {
    return (req, _res, _next) => {
      // Set different limits based on content type
      const limits = {
        'application/json': '10mb',
        'application/xml': '10mb',
        'multipart/form-data': '50mb',
        'text/plain': '1mb',
        default: '10mb'
      };
            
      const contentType = req.get('content-type') || '';
      const limit = limits[contentType.split(';')[0]] || limits.default;
            
      req.requestSizeLimit = limit;
      _next();
    };
  }

  /**
     * Input sanitization middleware
     */
  inputSanitization() {
    return (req, _res, _next) => {
      // Sanitize request body
      if (req.body && typeof req.body === 'object') {
        req.body = this._sanitizeObject(req.body);
      }
            
      // Sanitize query parameters
      if (req.query) {
        req.query = this._sanitizeObject(req.query);
      }
            
      // Sanitize URL parameters
      if (req.params) {
        req.params = this._sanitizeObject(req.params);
      }
            
      _next();
    };
  }

  /**
     * Parameter pollution protection
     */
  parameterPollutionProtection() {
    return hpp({
      whitelist: [
        'sort',
        'fields',
        'filter',
        'include',
        'expand'
      ]
    });
  }

  /**
     * Compression with security considerations
     */
  compressionMiddleware() {
    return compression({
      level: 6,
      threshold: 1024,
      filter: (req, res) => {
        // Don't compress responses that might contain secrets
        const contentType = res.get('Content-Type') || '';
        if (contentType.includes('application/json')) {
          const sensitiveEndpoints = ['/auth/', '/admin/', '/config/'];
          if (sensitiveEndpoints.some(endpoint => req.path.includes(endpoint))) {
            return false;
          }
        }
        return compression.filter(req, res);
      }
    });
  }

  /**
     * Request ID middleware
     */
  requestIdMiddleware() {
    return (req, _res, _next) => {
      const requestId = req.get('X-Request-Id') || 
                            req.get('X-Correlation-Id') || 
                            crypto.randomUUID();
            
      req.id = requestId;
      _res.set('X-Request-Id', requestId);
            
      _next();
    };
  }

  /**
     * Security logging middleware
     */
  securityLoggingMiddleware() {
    return (req, _res, _next) => {
      const startTime = Date.now();
            
      // Log security-relevant events
      const securityLog = {
        requestId: req.id,
        timestamp: new Date().toISOString(),
        method: req.method,
        path: req.path,
        ip: req.ip,
        userAgent: req.get('User-Agent'),
        origin: req.get('Origin'),
        referer: req.get('Referer')
      };
            
      // Log suspicious patterns
      if (this._isSuspiciousRequest(req)) {
        console.warn('Suspicious request detected:', securityLog);
      }
            
      // Log response
      _res.on('finish', () => {
        const duration = Date.now() - startTime;
        // eslint-disable-next-line no-console
        console.log({
          ...securityLog,
          statusCode: _res.statusCode,
          duration,
          contentLength: _res.get('Content-Length')
        });
      });
            
      _next();
    };
  }

  /**
     * Validation middleware factory
     */
  createValidationRules() {
    return {
      // Business Partner validation
      businessPartnerId: param('businessPartnerId')
        .isAlphanumeric()
        .isLength({ min: 1, max: 10 })
        .withMessage('Invalid business partner ID'),
            
      // Project ID validation
      projectId: param('projectId')
        .isUUID()
        .withMessage('Invalid project ID format'),
            
      // Email validation
      email: body('email')
        .isEmail()
        .normalizeEmail()
        .withMessage('Invalid email format'),
            
      // Phone validation
      phone: body('phone')
        .optional()
        .isMobilePhone()
        .withMessage('Invalid phone format'),
            
      // Search query validation
      searchQuery: query('q')
        .optional()
        .trim()
        .isLength({ min: 1, max: 100 })
        .matches(/^[a-zA-Z0-9\s\-_.]+$/)
        .withMessage('Invalid search query'),
            
      // Pagination validation
      pagination: [
        query('limit')
          .optional()
          .isInt({ min: 1, max: 1000 })
          .toInt()
          .withMessage('Invalid limit parameter'),
        query('offset')
          .optional()
          .isInt({ min: 0 })
          .toInt()
          .withMessage('Invalid offset parameter')
      ]
    };
  }

  /**
     * Validation result handler
     */
  handleValidationErrors() {
    return (req, _res, _next) => {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return _res.status(400).json({
          error: 'Validation failed',
          details: errors.array(),
          requestId: req.id
        });
      }
      _next();
    };
  }

  /**
     * Private helper methods
     */
    
  _isOriginAllowed(origin, allowedOrigins) {
    return allowedOrigins.some(allowed => {
      if (allowed.includes('*')) {
        const pattern = allowed.replace(/\*/g, '.*');
        return new RegExp(pattern).test(origin);
      }
      return allowed === origin;
    });
  }

  _sanitizeObject(obj) {
    if (Array.isArray(obj)) {
      return obj.map(item => this._sanitizeValue(item));
    }
        
    if (obj && typeof obj === 'object') {
      const sanitized = {};
      for (const [key, value] of Object.entries(obj)) {
        sanitized[key] = this._sanitizeValue(value);
      }
      return sanitized;
    }
        
    return this._sanitizeValue(obj);
  }

  _sanitizeValue(value) {
    if (typeof value === 'string') {
      return xss(value, {
        whiteList: {}, // Remove all HTML tags
        stripIgnoreTag: true,
        stripIgnoreTagBody: ['script']
      });
    }
    return value;
  }

  _isSuspiciousRequest(req) {
    const suspiciousPatterns = [
      /script/i,
      /javascript/i,
      /<script/i,
      /eval\(/i,
      /union.*select/i,
      /drop.*table/i,
      /\.\.\//,
      /\/etc\/passwd/,
      /cmd\.exe/i
    ];
        
    const checkString = `${req.path} ${req.get('User-Agent')} ${JSON.stringify(req.query)}`;
    return suspiciousPatterns.some(pattern => pattern.test(checkString));
  }
}

module.exports = new EnterpriseSecurityMiddleware();