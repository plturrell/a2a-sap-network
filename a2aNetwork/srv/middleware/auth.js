/**
 * @fileoverview SAP XSUAA Authentication Middleware
 * @since 1.0.0
 * @module auth
 * 
 * Authentication middleware implementing SAP XSUAA JWT validation,
 * Passport.js strategies, and session management for enterprise security
 */

const xssec = require('@sap/xssec');
const passport = require('passport');
const xsenv = require('@sap/xsenv');
const jwt = require('jsonwebtoken');
const cds = require('@sap/cds');
const { getXSUAAConfig: getXSUAAConfigHelper, initializeXSUAA } = require('./xsuaaConfig');

// Session configuration
const sessionConfig = {
  secret: process.env.SESSION_SECRET || (() => {
    if (process.env.NODE_ENV === 'production') {
      throw new Error('SESSION_SECRET must be set in production environment');
    }
    cds.log('auth').warn('Using random session secret - set SESSION_SECRET environment variable');
    return 'DEV-SESSION-SECRET-' + Math.random().toString(36);
  })(),
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: process.env.NODE_ENV === 'production', // HTTPS only in production
    httpOnly: true,
    maxAge: 24 * 60 * 60 * 1000, // 24 hours
    sameSite: 'strict'
  }
};

// Environment flag to enable/disable XSUAA validation
// SECURITY: This should always be 'true' in production
const ENABLE_XSUAA_VALIDATION = process.env.ENABLE_XSUAA_VALIDATION !== 'false' || process.env.NODE_ENV === 'production';

// Production safety check
if (process.env.NODE_ENV === 'production' && process.env.ENABLE_XSUAA_VALIDATION === 'false') {
  throw new Error('SECURITY ERROR: XSUAA validation MUST be enabled in production environment');
}

// Development mode check - require explicit opt-in for development auth
const USE_DEVELOPMENT_AUTH = process.env.USE_DEVELOPMENT_AUTH === 'true' && process.env.NODE_ENV !== 'production';

// XSUAA JWT validation middleware
const validateJWT = (req, res, next) => {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'No valid authorization header found' });
  }
  
  const token = authHeader.substring(7);
  
  // Development mode fallback - only if explicitly enabled
  if (USE_DEVELOPMENT_AUTH && !ENABLE_XSUAA_VALIDATION) {
    // Log security warning for development
    if (process.env.LOG_LEVEL !== 'silent') {
      cds.log('auth').warn('SECURITY WARNING: Development authentication mode - NOT FOR PRODUCTION USE');
    }
    try {
      // Simple JWT decode for development (no signature verification)
      const decoded = jwt.decode(token, { complete: true });
      if (!decoded) {
        return res.status(401).json({ error: 'Invalid token format' });
      }
      
      // Create mock user from token payload or use defaults
      req.user = {
        id: decoded.payload.sub || decoded.payload.user_name || 'dev-user',
        email: decoded.payload.email || 'developer@a2a-network.com',
        givenName: decoded.payload.given_name || 'Developer',
        familyName: decoded.payload.family_name || 'User',
        roles: ['authenticated-user'], // Limited roles in dev mode
        scopes: decoded.payload.scope ? decoded.payload.scope.split(' ') : ['user.access'],
        tenant: decoded.payload.zid || 'local-dev',
        zoneId: decoded.payload.zid || 'local-dev',
        sapRoles: ['User'], // Limited SAP roles in dev mode
        isDevelopment: true // Flag for development context
      };
      
      // Development authentication (logged to audit trail)
      if (process.env.LOG_LEVEL === 'debug') {
        cds.log('auth').debug(`Development authentication for user: ${req.user.id}`);
      }
      return next();
    } catch (error) {
      cds.log('auth').error('Development JWT decode failed:', error.message);
      return res.status(401).json({ error: 'Invalid development token' });
    }
  }
  
  // Production XSUAA validation
  try {
    const xsuaaConfig = getXSUAAConfigHelper();
    
    if (!xsuaaConfig) {
      throw new Error('XSUAA configuration not found');
    }
    
    if (xsuaaConfig.isDevelopment) {
      // This should not happen since USE_DEVELOPMENT_AUTH should handle this case
      cds.log('auth').warn('Development XSUAA config in production validation path');
    }
    
    // Validate JWT token with XSUAA
    xssec.createSecurityContext(token, xsuaaConfig.credentials, (error, securityContext) => {
      if (error) {
        cds.log('auth').error('JWT validation failed:', error);
        return res.status(401).json({ error: 'Invalid JWT token' });
      }
      
      // Extract user information from security context
      const user = {
        id: securityContext.getLogonName() || securityContext.getUserName(),
        email: securityContext.getEmail(),
        givenName: securityContext.getGivenName(),
        familyName: securityContext.getFamilyName(),
        roles: securityContext.getAttributeValues('groups') || [],
        scopes: securityContext.getGrantedScopes() || [],
        tenant: securityContext.getSubdomain(),
        zoneId: securityContext.getZoneId()
      };
      
      // Add SAP-specific role mappings
      user.sapRoles = [];
      if (securityContext.hasLocalScope('Admin')) user.sapRoles.push('Admin');
      if (securityContext.hasLocalScope('Developer')) user.sapRoles.push('Developer');
      if (securityContext.hasLocalScope('User')) user.sapRoles.push('User');
      if (securityContext.hasLocalScope('ServiceAccount')) user.sapRoles.push('ServiceAccount');
      
      req.user = user;
      req.securityContext = securityContext;
      next();
    });
  } catch (error) {
    cds.log('auth').error('JWT validation error:', error);
    return res.status(401).json({ error: 'Authentication failed' });
  }
};

// Role-based access control middleware
const requireRole = (role) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }
    
    // Check SAP roles first, then fallback to regular roles
    const userRoles = req.user.sapRoles || req.user.roles || [];
    
    if (!userRoles.includes(role)) {
      // Also check scopes for permission-based access
      const userScopes = req.user.scopes || [];
      const requiredScope = `${role.toLowerCase()}.access`;
      
      if (!userScopes.includes(requiredScope)) {
        return res.status(403).json({ 
          error: `Role '${role}' or scope '${requiredScope}' required`,
          userRoles: userRoles,
          userScopes: userScopes
        });
      }
    }
    
    next();
  };
};

// API key validation for service-to-service communication
const validateAPIKey = (req, res, next) => {
  const apiKey = req.headers['x-api-key'];
  
  if (!apiKey) {
    return next(); // Continue to other auth methods
  }
  
  // In production, validate against stored API keys
  const validApiKeys = process.env.VALID_API_KEYS?.split(',') || [];
  
  if (validApiKeys.includes(apiKey)) {
    req.user = { id: 'api-client', roles: ['service-account'] };
    return next();
  }
  
  return res.status(401).json({ error: 'Invalid API key' });
};

// Apply authentication middleware
const applyAuthMiddleware = (app) => {
  // Initialize passport
  app.use(passport.initialize());
  
  // API key validation for service endpoints
  app.use('/api/v1/service/*', validateAPIKey);
  
  // JWT validation for user endpoints
  app.use('/api/v1/*', (req, res, next) => {
    // Skip auth for health check and public endpoints
    if (req.path === '/health' || req.path.startsWith('/api/v1/public/')) {
      return next();
    }
    
    validateJWT(req, res, next);
  });
};

// Legacy function - use xsuaaConfig.js instead
const getXSUAAConfig = () => {
  cds.log('auth').warn('Using legacy getXSUAAConfig - migrate to xsuaaConfig.js');
  return getXSUAAConfigHelper();
};

// Initialize XSUAA passport strategy
const initializeXSUAAStrategy = () => {
  const initResult = initializeXSUAA();
  
  if (!initResult.initialized) {
    cds.log('auth').warn('XSUAA not initialized:', initResult.error);
    return initResult;
  }
  
  if (initResult.mode === 'development') {
    cds.log('auth').warn('XSUAA initialized in development mode:', initResult.warning);
    return initResult;
  }
  
  try {
    const xsuaaConfig = getXSUAAConfigHelper();
    if (xsuaaConfig && !xsuaaConfig.isDevelopment) {
      const JWTStrategy = xssec.JWTStrategy;
      passport.use(new JWTStrategy(xsuaaConfig.credentials));
      cds.log('auth').info('XSUAA JWT strategy initialized successfully');
    }
    return initResult;
  } catch (error) {
    cds.log('auth').error('Failed to initialize XSUAA strategy:', error);
    return { initialized: false, error: error.message };
  }
};

module.exports = {
  applyAuthMiddleware,
  validateJWT,
  requireRole,
  validateAPIKey,
  sessionConfig,
  getXSUAAConfig,
  initializeXSUAAStrategy
};