const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

// CORS configuration
const corsOptions = {
  origin: function (origin, callback) {
    // In production, replace with your actual domains
    const allowedOrigins = process.env.ALLOWED_ORIGINS
      ? process.env.ALLOWED_ORIGINS.split(',')
      : process.env.NODE_ENV === 'production'
        ? ['https://a2a-network.cfapps.eu10.hana.ondemand.com']
        : ['http://localhost:4004', 'http://localhost:4005', 'http://localhost:3000'];
    
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) return callback(null, true);
    
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
        "blob:"
      ],
      fontSrc: [
        "'self'",
        "https://ui5.sap.com",
        "https://sapui5.hana.ondemand.com",
        "https://openui5.hana.ondemand.com",
        "https://sdk.openui5.org"
      ],
      connectSrc: [
        "'self'",
        "ws://localhost:*",
        "wss://localhost:*",
        "https://ui5.sap.com",
        "https://sapui5.hana.ondemand.com",
        process.env.BLOCKCHAIN_RPC_URL || "http://localhost:8545"
      ],
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

// Request sanitization middleware
const sanitizeRequest = (req, res, next) => {
  // Sanitize query parameters
  if (req.query) {
    Object.keys(req.query).forEach(key => {
      if (typeof req.query[key] === 'string') {
        // Remove any script tags or SQL injection attempts
        req.query[key] = req.query[key]
          .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
          .replace(/['";]/g, '');
      }
    });
  }
  
  // Limit request body size
  if (req.body && JSON.stringify(req.body).length > 1048576) { // 1MB limit
    return res.status(413).json({ error: 'Request body too large' });
  }
  
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
  
  // Apply request sanitization
  app.use(sanitizeRequest);
  
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
};

module.exports = {
  applySecurityMiddleware,
  corsOptions,
  rateLimiters,
  helmetConfig
};