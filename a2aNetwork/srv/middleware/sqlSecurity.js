/**
 * @fileoverview SQL Security and Query Validation Middleware
 * @since 1.0.0
 * @module sql-security
 * 
 * Provides SQL injection protection and query validation
 * for database operations in the A2A Network system
 */

const cds = require('@sap/cds');

/**
 * SQL injection patterns to detect and block
 * Updated with legitimate use case exceptions
 */
const SQL_INJECTION_PATTERNS = [
  // More targeted patterns that avoid false positives
  /(\-\-\s*[\w\s]*;)|(\-\-\s*$)/i,  // SQL comments with suspicious content
  /(;[\s]*drop|;[\s]*delete|;[\s]*update|;[\s]*insert)/i, // Stacked queries
  /(\bUNION[\s]+ALL[\s]+SELECT\b|\bUNION[\s]+SELECT\b)/i, // Union attacks
  /(\bSELECT[\s]+.+[\s]+FROM[\s]+.+[\s]+WHERE\b)/i, // Only flag complex SELECT statements
  /(\bINSERT[\s]+INTO\b|\bDELETE[\s]+FROM\b|\bUPDATE[\s]+.+[\s]+SET\b)/i, // Data manipulation
  /(\bDROP[\s]+TABLE\b|\bDROP[\s]+DATABASE\b|\bCREATE[\s]+TABLE\b)/i, // Schema changes
  /(\bEXEC[\s]*\(|\bEXECUTE[\s]*\(|\bsp_[\w]+|\bxp_[\w]+)/i, // Stored procedures
  /((\%27)|(\'))\s*((\%6F)|o|(\%4F))\s*((\%72)|r|(\%52))\s*(\%27|\'|\%3D|=)/i, // 'or' attacks
  /((\%27)|(\'))\s*((\%31)|1)\s*(\%3D|=)\s*(\%31|1)/i // '1'='1' attacks
];

/**
 * Validates SQL query parameters for injection attempts
 * @param {string} query - SQL query string
 * @param {Array} params - Query parameters
 * @returns {Object} Validation result
 */
function validateSQLQuery(query, params = []) {
  const log = cds.log('sql-security');
  
  // Check query string for injection patterns
  for (const pattern of SQL_INJECTION_PATTERNS) {
    if (pattern.test(query)) {
      log.error('Potential SQL injection detected in query:', query);
      return {
        valid: false,
        reason: 'SQL injection pattern detected in query',
        pattern: pattern.toString()
      };
    }
  }
  
  // Check parameters for injection patterns
  for (let i = 0; i < params.length; i++) {
    const param = String(params[i]);
    for (const pattern of SQL_INJECTION_PATTERNS) {
      if (pattern.test(param)) {
        log.error('Potential SQL injection detected in parameter:', param);
        return {
          valid: false,
          reason: `SQL injection pattern detected in parameter ${i}`,
          parameter: i,
          pattern: pattern.toString()
        };
      }
    }
  }
  
  return { valid: true };
}

/**
 * Sanitizes input parameters for safe SQL usage
 * @param {*} input - Input to sanitize
 * @returns {*} Sanitized input
 */
function sanitizeInput(input) {
  if (typeof input !== 'string') {
    return input;
  }
  
  // Escape single quotes
  return input.replace(/'/g, "''");
}

/**
 * Creates parameterized query helper
 * @param {string} baseQuery - Base query with placeholders
 * @param {Array} params - Parameters to bind
 * @returns {Object} Safe query object
 */
function createSafeQuery(baseQuery, params = []) {
  const validation = validateSQLQuery(baseQuery, params);
  
  if (!validation.valid) {
    const error = new Error(`SQL Security Violation: ${validation.reason}`);
    error.code = 'SQL_INJECTION_DETECTED';
    error.query = baseQuery;
    error.params = params;
    throw error;
  }
  
  return {
    query: baseQuery,
    params: params.map(sanitizeInput),
    isParameterized: true
  };
}

/**
 * Middleware to intercept and validate database operations
 */
function applySQLSecurityMiddleware(service) {
  const log = cds.log('sql-security');
  
  // Intercept SELECT operations
  service.on('READ', '*', (req, next) => {
    if (req.query && req.query.SELECT) {
      log.debug('Validating SELECT query');
      // CDS SELECT queries are safe by design, but log for monitoring
    }
    return next();
  });
  
  // Intercept INSERT operations
  service.on('CREATE', '*', (req, next) => {
    if (req.data) {
      log.debug('Validating INSERT data');
      // Validate input data
      for (const [key, value] of Object.entries(req.data)) {
        if (typeof value === 'string') {
          const validation = validateSQLQuery(value, []);
          if (!validation.valid) {
            const error = new Error(`SQL injection detected in field '${key}': ${validation.reason}`);
            error.code = 'SQL_INJECTION_DETECTED';
            throw error;
          }
        }
      }
    }
    return next();
  });
  
  // Intercept UPDATE operations
  service.on('UPDATE', '*', (req, next) => {
    if (req.data) {
      log.debug('Validating UPDATE data');
      // Validate input data
      for (const [key, value] of Object.entries(req.data)) {
        if (typeof value === 'string') {
          const validation = validateSQLQuery(value, []);
          if (!validation.valid) {
            const error = new Error(`SQL injection detected in field '${key}': ${validation.reason}`);
            error.code = 'SQL_INJECTION_DETECTED';
            throw error;
          }
        }
      }
    }
    return next();
  });
  
  log.info('SQL security middleware applied');
}

/**
 * Express middleware for SQL security validation
 */
function validateSQLMiddleware(req, res, next) {
  const log = cds.log('sql-security');
  
  // Check request body for potential SQL injection
  if (req.body && typeof req.body === 'object') {
    function checkObject(obj, path = '') {
      for (const [key, value] of Object.entries(obj)) {
        const currentPath = path ? `${path}.${key}` : key;
        
        if (typeof value === 'string') {
          const validation = validateSQLQuery(value, []);
          if (!validation.valid) {
            log.error(`SQL injection attempt detected in ${currentPath}:`, value);
            return res.status(400).json({
              error: 'Invalid input detected',
              code: 'SQL_INJECTION_DETECTED',
              field: currentPath
            });
          }
        } else if (typeof value === 'object' && value !== null) {
          const result = checkObject(value, currentPath);
          if (result) return result;
        }
      }
    }
    
    const result = checkObject(req.body);
    if (result) return result;
  }
  
  // Check query parameters
  if (req.query) {
    // Comprehensive whitelist for known safe dashboard IDs, tile IDs, and API endpoints
    const safeDashboardIds = [
      'overview_dashboard',
      'dashboard_test', 
      'agent_visualization',
      'service_marketplace',
      'blockchain_dashboard',
      'notification_center',
      'network_analytics',
      'network_health',
      'agent_marketplace',
      'reputationDashboard',
      'governance_dashboard',
      'code_intelligence',
      'logs_dashboard'
    ];
    
    // Safe API endpoint patterns
    const safeEndpointPatterns = [
      /^\/api\/v\d+\//,
      /^\/odata\/v\d+\//,
      /^\/health$/,
      /^\/metrics$/,
      /^\/status$/
    ];
    
    // Check if this is a safe API endpoint
    const isSafeEndpoint = safeEndpointPatterns.some(pattern => pattern.test(req.path));
    
    // Skip validation for safe endpoints with standard parameters
    if (isSafeEndpoint) {
      const safeParams = ['id', 'version', 'format', 'limit', 'offset', 'since', 'until', 'level'];
      const allParamsSafe = Object.keys(req.query).every(key => safeParams.includes(key));
      if (allParamsSafe) {
        return next();
      }
    }
    
    for (const [key, value] of Object.entries(req.query)) {
      if (typeof value === 'string') {
        // Skip validation for whitelisted dashboard IDs
        if (key === 'id' && safeDashboardIds.includes(value)) {
          continue;
        }
        
        const validation = validateSQLQuery(value, []);
        if (!validation.valid) {
          log.error(`SQL injection attempt detected in query parameter ${key}:`, value);
          return res.status(400).json({
            error: 'Invalid query parameter detected',
            code: 'SQL_INJECTION_DETECTED',
            parameter: key
          });
        }
      }
    }
  }
  
  next();
}

module.exports = {
  validateSQLQuery,
  sanitizeInput,
  createSafeQuery,
  applySQLSecurityMiddleware,
  validateSQLMiddleware,
  SQL_INJECTION_PATTERNS
};