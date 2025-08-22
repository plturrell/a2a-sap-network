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

// BTP/XSUAA validation configuration
// SECURITY: Production requires BTP mode or explicit non-BTP configuration
const IS_BTP_ENVIRONMENT = process.env.BTP_ENVIRONMENT === 'true';
const ALLOW_NON_BTP_AUTH = process.env.ALLOW_NON_BTP_AUTH === 'true';

// Production safety checks
if (process.env.NODE_ENV === 'production') {
  if (!IS_BTP_ENVIRONMENT && !ALLOW_NON_BTP_AUTH) {
    throw new Error('SECURITY ERROR: Production requires either BTP_ENVIRONMENT=true or explicit ALLOW_NON_BTP_AUTH=true');
  }
  
  if (ALLOW_NON_BTP_AUTH && !process.env.JWT_SECRET) {
    throw new Error('SECURITY ERROR: JWT_SECRET must be set when using non-BTP authentication in production');
  }
}

// Development mode warnings
if (ALLOW_NON_BTP_AUTH && process.env.NODE_ENV === 'production') {
  cds.log('auth').warn('WARNING: Running in production with non-BTP authentication mode');
}

// XSUAA JWT validation middleware
const validateJWT = (req, res, next) => {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'No valid authorization header found' });
  }
  
  const token = authHeader.substring(7);
  
  // Non-BTP authentication for testing outside BTP network
  if (ALLOW_NON_BTP_AUTH && !IS_BTP_ENVIRONMENT) {
    try {
      // Verify JWT with proper secret (not just decode)
      const jwtSecret = process.env.JWT_SECRET;
      if (!jwtSecret) {
        throw new Error('JWT_SECRET required for non-BTP authentication');
      }
      
      const decoded = jwt.verify(token, jwtSecret, { algorithms: ['HS256'] });
      
      if (!decoded.sub && !decoded.user_id && !decoded.id) {
        throw new Error('Token must contain user identifier');
      }
      
      // Create user from verified token
      req.user = {
        id: decoded.sub || decoded.user_id || decoded.id,
        email: decoded.email || `${decoded.sub || decoded.user_id || decoded.id}@external.com`,
        givenName: decoded.given_name || decoded.firstName || 'External',
        familyName: decoded.family_name || decoded.lastName || 'User',
        roles: decoded.roles || ['authenticated-user'],
        scopes: decoded.scope ? (Array.isArray(decoded.scope) ? decoded.scope : decoded.scope.split(' ')) : ['user.access'],
        tenant: decoded.tenant || decoded.zid || 'external',
        zoneId: decoded.zid || 'external',
        sapRoles: decoded.sap_roles || decoded.roles || ['User'],
        isExternal: true // Flag for non-BTP context
      };
      
      cds.log('auth').info(`Non-BTP authentication for user: ${req.user.id}`);
      return next();
      
    } catch (error) {
      cds.log('auth').error('Non-BTP JWT validation failed:', error.message);
      return res.status(401).json({ error: 'Invalid JWT token', details: error.message });
    }
  }
  
  // BTP XSUAA validation
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

// Development auth bypass
function createDevelopmentUser() {
    return {
        id: 'dev-user',
        name: 'Development User',
        email: 'dev@a2a.network',
        roles: ['authenticated-user', 'Admin', 'Developer', 'AgentManager', 'ServiceManager'],
        sapRoles: ['authenticated-user', 'Admin', 'Developer', 'AgentManager', 'ServiceManager'],
        scope: ['authenticated-user', 'Admin', 'Developer', 'AgentManager', 'ServiceManager'],
        isDevelopment: true,
        permissions: ['*'], // Full permissions in development
        tenant: 'a2a-dev',
        zone: 'development',
        authType: 'development-bypass'
    };
}

// Apply authentication middleware
const applyAuthMiddleware = (app) => {
  // Initialize passport
  app.use(passport.initialize());
  
  // API key validation for service endpoints
  app.use('/api/v1/service/*', validateAPIKey);
  
  // SECURITY FIX: Only allow development mode bypass in development environment
  if (process.env.USE_DEVELOPMENT_AUTH === 'true' && process.env.NODE_ENV === 'development') {
    app.use('/api/v1/*', (req, res, next) => {
      const log = cds.log('auth-dev');
      log.warn(`🚀 DEVELOPMENT MODE: Bypassing authentication for ${req.originalUrl}`);
      
      // Always assign development user
      req.user = createDevelopmentUser();
      return next();
    });
    
    // Also bypass for OData endpoints
    app.use((req, res, next) => {
      if (req.originalUrl.includes('/odata/') || req.originalUrl.includes('/api/')) {
        req.user = req.user || createDevelopmentUser();
      }
      next();
    });
    
    return; // Skip the normal auth setup
  }
  
  // SECURITY CHECK: Prevent development auth in production
  if (process.env.USE_DEVELOPMENT_AUTH === 'true' && process.env.NODE_ENV === 'production') {
    throw new Error('SECURITY ERROR: Development authentication cannot be enabled in production');
  }
  
  // JWT validation for user endpoints
  app.use('/api/v1/*', (req, res, next) => {
    const log = cds.log('auth-debug');
    log.info(`Auth middleware processing: ${req.method} ${req.path}`, {
      originalUrl: req.originalUrl,
      query: req.query
    });
    
    // Skip auth for health check and public endpoints
    if (req.path === '/health' || req.originalUrl.startsWith('/api/v1/public/')) {
      log.info(`Skipping auth for health/public endpoint: ${req.originalUrl}`);
      return next();
    }
    
    // Skip auth for launchpad tile endpoints in development
    const launchpadEndpoints = [
      '/api/v1/Agents',
      '/api/v1/agents',
      '/api/v1/Services', 
      '/api/v1/services',
      '/api/v1/blockchain',
      '/api/v1/network',
      '/api/v1/health',
      '/api/v1/notifications',
      '/api/v1/NetworkStats'
    ];
    
    if (process.env.NODE_ENV === 'development' && launchpadEndpoints.some(path => req.originalUrl.startsWith(path))) {
      log.info(`Development mode: Skipping auth for launchpad endpoint: ${req.originalUrl}`);
      // Create a basic user for launchpad tile data access
      req.user = {
        id: 'launchpad-viewer',
        email: 'launchpad@a2a.network',
        givenName: 'Launchpad',
        familyName: 'Viewer',
        roles: ['authenticated-user'],
        scopes: ['user.access'],
        tenant: 'default',
        zoneId: 'default',
        sapRoles: ['User'],
        isLaunchpadAccess: true
      };
      return next();
    }
    
    // SECURITY: All tile API endpoints must be authenticated in production
    // Only skip authentication for true health checks and public endpoints
    if (process.env.NODE_ENV === 'production') {
      // No tile API bypasses allowed in production
      log.info(`Production mode: applying auth for all API endpoints: ${req.path}`);
    } else {
      // Development only: Allow some tile endpoints for UI development
      const devOnlyTileEndpoints = [
        '/api/v1/network/health'  // Only true health check allowed
      ];
      const urlPath = req.originalUrl.split('?')[0];
      if (devOnlyTileEndpoints.includes(urlPath)) {
        log.warn(`Development only: skipping auth for ${urlPath}`);
        return next();
      }
    }
    
    log.info(`Applying JWT validation for: ${req.path}`);
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