const cors = require('cors');
const cds = require('@sap/cds');
/**
 * @fileoverview SAP Security Hardening Middleware
 * @since 1.0.0
 * @module security
 * 
 * Implements comprehensive security middleware including CSRF protection,
 * secure headers, input validation, and threat detection for enterprise applications
 */

const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const crypto = require('crypto');
const validator = require('validator');
const DOMPurify = require('isomorphic-dompurify');
const { v4: uuidv4 } = require('uuid');
const securityMiddleware = require('./securityMiddleware');

// CORS configuration
const corsOptions = {
  origin: function (origin, callback) {
    // In production, replace with your actual domains
    const allowedOrigins = process.env.ALLOWED_ORIGINS
      ? process.env.ALLOWED_ORIGINS.split(',')
      : process.env.NODE_ENV === 'production'
        ? ['https://a2a-network.cfapps.eu10.hana.ondemand.com']
        : (process.env.CORS_ALLOWED_ORIGINS || 'http://localhost:4004').split(',');
    
    // Allow requests with no origin (e.g., curl, mobile apps, same-origin requests)
    if (!origin) {
      return callback(null, true);
    }
    
    if (allowedOrigins.indexOf(origin) !== -1) {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-CSRF-Token', 'X-Requested-With'],
  exposedHeaders: ['X-CSRF-Token'],
  maxAge: 86400 // 24 hours
};

// Rate limiting configurations
const createRateLimiter = (windowMs, max, message) => {
  return rateLimit({
    windowMs,
    max,
    message,
    standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
    legacyHeaders: false, // Disable the `X-RateLimit-*` headers
    skip: (req) => {
      // Skip rate limiting in development or for localhost
      if (process.env.NODE_ENV !== 'production') return true;
      const ip = req.ip || req.connection.remoteAddress;
      return ip === '::1' || ip === '127.0.0.1' || ip === 'localhost';
    },
    handler: (req, res) => {
      res.status(429).json({
        error: message,
        retryAfter: Math.round(windowMs / 1000)
      });
    }
  });
};

// Different rate limiters for different endpoints
const rateLimiters = {
  // General API rate limit
  api: createRateLimiter(
    15 * 60 * 1000, // 15 minutes
    100, // limit each IP to 100 requests per windowMs
    'Too many requests from this IP, please try again later.'
  ),
  
  // Stricter limit for authentication endpoints
  auth: createRateLimiter(
    15 * 60 * 1000, // 15 minutes
    5, // limit each IP to 5 requests per windowMs
    'Too many authentication attempts, please try again later.'
  ),
  
  // More lenient limit for read operations
  read: createRateLimiter(
    15 * 60 * 1000, // 15 minutes
    200, // limit each IP to 200 requests per windowMs
    'Too many read requests, please try again later.'
  ),
  
  // Strict limit for write operations
  write: createRateLimiter(
    15 * 60 * 1000, // 15 minutes
    50, // limit each IP to 50 requests per windowMs
    'Too many write operations, please try again later.'
  ),
  
  // Very strict limit for blockchain operations
  blockchain: createRateLimiter(
    60 * 60 * 1000, // 1 hour
    10, // limit each IP to 10 requests per hour
    'Too many blockchain operations, please try again later.'
  )
};

// Helmet configuration for security headers
const helmetConfig = helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: [
        "'self'",
        "'unsafe-inline'", // Required for SAP UI5
        "https://ui5.sap.com",
        "https://sapui5.hana.ondemand.com",
        "https://openui5.hana.ondemand.com",
        "https://sdk.openui5.org"
      ],
      styleSrc: [
        "'self'",
        "'unsafe-inline'", // Required for SAP UI5
        "https://ui5.sap.com",
        "https://sapui5.hana.ondemand.com",
        "https://openui5.hana.ondemand.com",
        "https://sdk.openui5.org"
      ],
      imgSrc: [
        "'self'",
        "data:",
        "https:",
        ...(process.env.NODE_ENV !== 'production' ? ["http://localhost:*"] : []),
        "blob:"
      ].filter(Boolean),
      fontSrc: [
        "'self'",
        "https://ui5.sap.com",
        "https://sapui5.hana.ondemand.com",
        "https://openui5.hana.ondemand.com",
        "https://sdk.openui5.org"
      ],
      connectSrc: [
        "'self'",
        ...(process.env.NODE_ENV !== 'production' ? ["ws://localhost:*", "wss://localhost:*"] : []),
        "https://ui5.sap.com",
        "https://sapui5.hana.ondemand.com",
        ...(process.env.BLOCKCHAIN_RPC_URL ? [process.env.BLOCKCHAIN_RPC_URL] : [])
      ].filter(Boolean),
      mediaSrc: ["'none'"],
      objectSrc: ["'none'"],
      frameSrc: ["'self'"],
      workerSrc: ["'self'", "blob:"],
      childSrc: ["'self'", "blob:"],
      formAction: ["'self'"],
      frameAncestors: ["'none'"],
      baseUri: ["'self'"],
      manifestSrc: ["'self'"]
    },
    reportOnly: false
  },
  crossOriginEmbedderPolicy: false, // Required for SAP UI5 resources
  hsts: {
    maxAge: 31536000, // 1 year
    includeSubDomains: true,
    preload: true
  },
  noSniff: true,
  originAgentCluster: true,
  dnsPrefetchControl: {
    allow: false
  },
  frameguard: {
    action: 'deny'
  },
  permittedCrossDomainPolicies: false,
  referrerPolicy: {
    policy: 'strict-origin-when-cross-origin'
  }
});

// Additional security middleware
const additionalSecurityHeaders = (req, res, next) => {
  // Remove X-Powered-By header
  res.removeHeader('X-Powered-By');
  
  // Add additional security headers
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  res.setHeader('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');
  
  next();
};

// CSRF Token management
const csrfTokens = new Map();

const generateCSRFToken = () => {
  return crypto.randomBytes(32).toString('hex');
};

const createCSRFToken = (sessionId) => {
  const token = generateCSRFToken();
  csrfTokens.set(sessionId, {
    token,
    created: Date.now(),
    used: false
  });
  return token;
};

const validateCSRFToken = (sessionId, token) => {
  const stored = csrfTokens.get(sessionId);
  if (!stored) return false;
  
  // Token expires after 1 hour
  if (Date.now() - stored.created > 3600000) {
    csrfTokens.delete(sessionId);
    return false;
  }
  
  return stored.token === token;
};

// CSRF Protection middleware
const csrfProtection = (req, res, next) => {
  // Skip CSRF for GET, HEAD, OPTIONS requests
  if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
    return next();
  }
  
  // Relaxed CSRF for development environment - log but don't block
  if (process.env.NODE_ENV !== 'production') {
    const token = req.headers['x-csrf-token'] || req.body._csrf;
    if (!token) {
      cds.log('service').warn(`⚠️  CSRF token missing for ${req.method} ${req.path} - would be blocked in production`);
    }
    return next();
  }
  
  const sessionId = req.sessionID || req.headers['x-session-id'] || 'anonymous';
  const token = req.headers['x-csrf-token'] || req.body._csrf;
  
  if (!token) {
    return res.status(403).json({
      error: 'CSRF token missing',
      code: 'CSRF_MISSING'
    });
  }
  
  if (!validateCSRFToken(sessionId, token)) {
    return res.status(403).json({
      error: 'Invalid CSRF token',
      code: 'CSRF_INVALID'
    });
  }
  
  next();
};

// CSRF token endpoint
const csrfTokenEndpoint = (req, res) => {
  const sessionId = req.sessionID || req.headers['x-session-id'] || 'anonymous';
  const token = createCSRFToken(sessionId);
  
  res.json({
    csrfToken: token,
    sessionId: sessionId
  });
};

// Enhanced input validation and sanitization
const validateAndSanitizeInput = (value, type = 'string', options = {}) => {
  if (value === null || value === undefined) {
    return options.allowEmpty ? value : null;
  }
  
  let sanitized = value;
  
  switch (type) {
    case 'string':
      if (typeof value !== 'string') return null;
      // Remove HTML tags and potential XSS
      sanitized = DOMPurify.sanitize(value, { ALLOWED_TAGS: [] });
      // Validate length if specified
      if (options.maxLength && sanitized.length > options.maxLength) {
        return null;
      }
      if (options.minLength && sanitized.length < options.minLength) {
        return null;
      }
      break;
      
    case 'email':
      if (!validator.isEmail(String(value))) return null;
      sanitized = validator.normalizeEmail(String(value));
      break;
      
    case 'url':
      if (!validator.isURL(String(value), { protocols: ['http', 'https'], require_protocol: true })) {
        return null;
      }
      sanitized = String(value);
      break;
      
    case 'number':
      const num = Number(value);
      if (isNaN(num)) return null;
      if (options.min !== undefined && num < options.min) return null;
      if (options.max !== undefined && num > options.max) return null;
      sanitized = num;
      break;
      
    case 'integer':
      if (!validator.isInt(String(value), options)) return null;
      sanitized = parseInt(value, 10);
      break;
      
    case 'boolean':
      if (typeof value === 'boolean') return value;
      if (typeof value === 'string') {
        return value.toLowerCase() === 'true';
      }
      return null;
      
    case 'uuid':
      if (!validator.isUUID(String(value))) return null;
      sanitized = String(value);
      break;
      
    case 'address':
      // Ethereum address validation
      if (!/^0x[a-fA-F0-9]{40}$/.test(String(value))) return null;
      sanitized = String(value).toLowerCase();
      break;
      
    case 'json':
      try {
        if (typeof value === 'string') {
          sanitized = JSON.parse(value);
        } else {
          sanitized = value;
        }
      } catch (e) {
        return null;
      }
      break;
      
    default:
      return null;
  }
  
  return sanitized;
};

// Request sanitization middleware
const sanitizeRequest = (req, res, next) => {
  // Add correlation ID for request tracking
  req.correlationId = req.headers['x-correlation-id'] || uuidv4();
  res.setHeader('X-Correlation-ID', req.correlationId);
  
  // Sanitize query parameters
  if (req.query) {
    Object.keys(req.query).forEach(key => {
      const sanitized = validateAndSanitizeInput(req.query[key], 'string', { maxLength: 1000 });
      if (sanitized === null) {
        delete req.query[key];
      } else {
        req.query[key] = sanitized;
      }
    });
  }
  
  // Sanitize request body
  if (req.body && typeof req.body === 'object') {
    req.body = sanitizeObject(req.body);
  }
  
  // Limit request body size
  if (req.body && JSON.stringify(req.body).length > 1048576) { // 1MB limit
    return res.status(413).json({ 
      error: 'Request body too large',
      correlationId: req.correlationId
    });
  }
  
  next();
};

// Recursive object sanitization
const sanitizeObject = (obj) => {
  if (obj === null || typeof obj !== 'object') return obj;
  
  if (Array.isArray(obj)) {
    return obj.map(item => sanitizeObject(item));
  }
  
  const sanitized = {};
  Object.keys(obj).forEach(key => {
    // Sanitize the key itself
    const sanitizedKey = validateAndSanitizeInput(key, 'string', { maxLength: 100 });
    if (sanitizedKey === null) return;
    
    const value = obj[key];
    if (typeof value === 'string') {
      sanitized[sanitizedKey] = validateAndSanitizeInput(value, 'string', { maxLength: 10000 });
    } else if (typeof value === 'object') {
      sanitized[sanitizedKey] = sanitizeObject(value);
    } else {
      sanitized[sanitizedKey] = value;
    }
  });
  
  return sanitized;
};

// Security audit logging
const securityAuditLog = (req, res, next) => {
  const startTime = Date.now();
  
  // Log security-relevant requests
  if (req.method !== 'GET' || req.path.includes('/admin') || req.path.includes('/auth')) {
    cds.log('service').info(`[SECURITY AUDIT] ${new Date().toISOString()} - ${req.method} ${req.path} - IP: ${req.ip} - User-Agent: ${req.get('User-Agent')} - Correlation-ID: ${req.correlationId}`);
  }
  
  // Log response
  res.on('finish', () => {
    if (res.statusCode >= 400) {
      cds.log('service').info(`[SECURITY AUDIT] ${new Date().toISOString()} - Response: ${res.statusCode} - Duration: ${Date.now() - startTime}ms - Correlation-ID: ${req.correlationId}`);
    }
  });
  
  next();
};

// Apply security middleware to Express app
const applySecurityMiddleware = (app) => {
  // Apply CORS
  app.use(cors(corsOptions));
  
  // Apply Helmet for security headers
  app.use(helmetConfig);
  
  // Apply additional security headers
  app.use(additionalSecurityHeaders);
  
  // Apply comprehensive security monitoring middleware
  app.use(securityMiddleware.middleware());
  
  // Apply authentication failure monitoring
  app.use(securityMiddleware.authenticationFailure());
  
  // Apply input validation middleware
  app.use('/api/v1/*', securityMiddleware.inputValidation());
  
  // Apply security audit logging
  app.use(securityAuditLog);
  
  // Apply request sanitization
  app.use(sanitizeRequest);
  
  // CSRF token endpoint
  app.get('/api/v1/csrf-token', csrfTokenEndpoint);
  
  // Apply CSRF protection to write operations
  app.use('/api/v1/*', (req, res, next) => {
    if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(req.method)) {
      // Use integrated CSRF protection that reports to security monitoring
      securityMiddleware.csrfProtection()(req, res, next);
    } else {
      next();
    }
  });
  
  // Apply rate limiting to different routes
  app.use('/api/v1/auth', rateLimiters.auth);
  app.use('/api/v1/deployContract', rateLimiters.blockchain);
  app.use('/api/v1/syncBlockchain', rateLimiters.blockchain);
  
  // Apply different rate limits based on HTTP method
  app.use('/api/v1/*', (req, res, next) => {
    if (req.method === 'GET') {
      rateLimiters.read(req, res, next);
    } else if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(req.method)) {
      rateLimiters.write(req, res, next);
    } else {
      rateLimiters.api(req, res, next);
    }
  });
  
  // General rate limit for all other routes
  app.use(rateLimiters.api);
  
  // Security dashboard endpoints
  app.get('/api/v1/security/dashboard', (req, res) => {
    const dashboard = securityMiddleware.getDashboardData();
    res.json(dashboard || { error: 'Security monitoring not available' });
  });
  
  app.get('/api/v1/security/alerts', (req, res) => {
    const alerts = securityMiddleware.getActiveAlerts();
    res.json({ alerts: alerts || [] });
  });
  
  app.post('/api/v1/security/alerts/:alertId/acknowledge', (req, res) => {
    const alertId = req.params.alertId;
    const userId = req.user?.id || 'anonymous';
    const result = securityMiddleware.acknowledgeAlert(alertId, userId);
    res.json({ success: result });
  });
  
  app.post('/api/v1/security/alerts/:alertId/resolve', (req, res) => {
    const alertId = req.params.alertId;
    const userId = req.user?.id || 'anonymous';
    const resolution = req.body.resolution || 'Resolved via API';
    const result = securityMiddleware.resolveAlert(alertId, userId, resolution);
    res.json({ success: result });
  });
  
  app.get('/api/v1/security/metrics', (req, res) => {
    const metrics = securityMiddleware.getSecurityMetrics();
    res.json(metrics || { error: 'Security metrics not available' });
  });
};

module.exports = {
  applySecurityMiddleware,
  corsOptions,
  rateLimiters,
  helmetConfig,
  csrfProtection,
  csrfTokenEndpoint,
  validateAndSanitizeInput,
  sanitizeRequest,
  securityAuditLog
};