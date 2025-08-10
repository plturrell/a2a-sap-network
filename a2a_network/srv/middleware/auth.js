const xssec = require('@sap/xssec');
const passport = require('passport');
const xsenv = require('@sap/xsenv');
const jwt = require('jsonwebtoken');

// Session configuration
const sessionConfig = {
  secret: process.env.SESSION_SECRET || 'CHANGE-THIS-IN-PRODUCTION-' + Math.random().toString(36),
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
const ENABLE_XSUAA_VALIDATION = process.env.ENABLE_XSUAA_VALIDATION === 'true';

// Production safety check
if (process.env.NODE_ENV === 'production' && !ENABLE_XSUAA_VALIDATION) {
  throw new Error('SECURITY ERROR: XSUAA validation MUST be enabled in production environment');
}

// XSUAA JWT validation middleware
const validateJWT = (req, res, next) => {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'No valid authorization header found' });
  }
  
  const token = authHeader.substring(7);
  
  // Development mode fallback when XSUAA is not enabled
  if (!ENABLE_XSUAA_VALIDATION) {
    if (process.env.NODE_ENV === 'production') {
      // This should never happen due to the check above, but double safety
      return res.status(500).json({ error: 'SECURITY ERROR: Development authentication in production' });
    }
    
    // Log security warning for development
    if (process.env.LOG_LEVEL !== 'silent') {
      console.warn('⚠️  SECURITY WARNING: XSUAA validation disabled - development mode only');
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
        console.log(`Development authentication for user: ${req.user.id}`);
      }
      return next();
    } catch (error) {
      console.error('Development JWT decode failed:', error.message);
      return res.status(401).json({ error: 'Invalid development token' });
    }
  }
  
  // Production XSUAA validation
  try {
    const xsuaaConfig = getXSUAAConfig();
    
    if (!xsuaaConfig) {
      throw new Error('XSUAA configuration not found');
    }
    
    // Validate JWT token with XSUAA
    xssec.createSecurityContext(token, xsuaaConfig.credentials, (error, securityContext) => {
      if (error) {
        console.error('JWT validation failed:', error);
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
    console.error('JWT validation error:', error);
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

// XSUAA configuration
const getXSUAAConfig = () => {
  try {
    // First try to get from VCAP_SERVICES (Cloud Foundry)
    if (process.env.VCAP_SERVICES) {
      const vcapServices = JSON.parse(process.env.VCAP_SERVICES);
      const xsuaaService = vcapServices.xsuaa?.[0] || vcapServices['xsuaa']?.[0];
      
      if (xsuaaService && xsuaaService.credentials) {
        return {
          credentials: xsuaaService.credentials,
          verificationKey: xsuaaService.credentials.verificationkey
        };
      }
    }
    
    // Fallback to environment variables (Kubernetes/local development)
    if (process.env.XSUAA_CLIENT_ID && process.env.XSUAA_CLIENT_SECRET) {
      return {
        credentials: {
          clientid: process.env.XSUAA_CLIENT_ID,
          clientsecret: process.env.XSUAA_CLIENT_SECRET,
          url: process.env.XSUAA_URL || process.env.XSUAA_AUTH_URL,
          uaadomain: process.env.XSUAA_UAA_DOMAIN,
          verificationkey: process.env.XSUAA_VERIFICATION_KEY,
          xsappname: process.env.XSUAA_XS_APP_NAME || 'a2a-network',
          identityzone: process.env.XSUAA_IDENTITY_ZONE || 'sap-provisioning',
          tenantmode: 'dedicated'
        }
      };
    }
    
    // Try xsenv service lookup
    try {
      xsenv.loadEnv();
      const services = xsenv.getServices({ xsuaa: { tag: 'xsuaa' } });
      if (services.xsuaa) {
        return {
          credentials: services.xsuaa,
          verificationKey: services.xsuaa.verificationkey
        };
      }
    } catch (xsenvError) {
      console.warn('xsenv service lookup failed:', xsenvError.message);
    }
    
    console.warn('No XSUAA configuration found. Authentication will not work properly.');
    return null;
  } catch (error) {
    console.error('Error loading XSUAA configuration:', error);
    return null;
  }
};

// Initialize XSUAA passport strategy
const initializeXSUAAStrategy = () => {
  const xsuaaConfig = getXSUAAConfig();
  
  if (!xsuaaConfig) {
    console.warn('XSUAA not configured. Passport strategy not initialized.');
    return;
  }
  
  try {
    const JWTStrategy = xssec.JWTStrategy;
    passport.use(new JWTStrategy(xsuaaConfig.credentials));
    
    console.log('XSUAA JWT strategy initialized successfully');
  } catch (error) {
    console.error('Failed to initialize XSUAA strategy:', error);
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