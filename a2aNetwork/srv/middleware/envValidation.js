/**
 * @fileoverview Environment Variable Validation Middleware
 * @since 1.0.0
 * @module env-validation
 *
 * Validates required environment variables on server startup
 * to prevent runtime failures due to missing configuration
 */

const cds = require('@sap/cds');

/**
 * Required environment variables for different environments
 */
const ENV_REQUIREMENTS = {
  production: [
    'SESSION_SECRET',
    'JWT_SECRET',
    'DATABASE_URL'
  ],
  development: [
    'NODE_ENV'
  ],
  common: []
};

/**
 * Optional environment variables with defaults
 */
const ENV_DEFAULTS = {
  PORT: '4004',
  STATIC_SERVER_PORT: '4005',
  LOG_LEVEL: 'info',
  DEFAULT_NETWORK: 'localhost',
  RPC_URL: 'http://localhost:8545',
  CHAIN_ID: '31337',
  CORS_ALLOWED_ORIGINS: 'http://localhost:4004,http://localhost:8080'
};

/**
 * Validates environment variables based on current NODE_ENV
 * @returns {Object} Validation result with status and missing variables
 */
function validateEnvironment() {
  const nodeEnv = process.env.NODE_ENV || 'development';
  const log = cds.log('env-validation');

  const requiredVars = [
    ...ENV_REQUIREMENTS.common,
    ...(ENV_REQUIREMENTS[nodeEnv] || [])
  ];

  const missingVars = [];
  const warnings = [];

  // Check required variables
  for (const envVar of requiredVars) {
    if (!process.env[envVar]) {
      missingVars.push(envVar);
    }
  }

  // Set defaults for optional variables
  for (const [envVar, defaultValue] of Object.entries(ENV_DEFAULTS)) {
    if (!process.env[envVar]) {
      process.env[envVar] = defaultValue;
      warnings.push(`${envVar} not set, using default: ${defaultValue}`);
    }
  }

  // Environment-specific validations
  if (nodeEnv === 'production') {
    // Production-specific checks
    if (process.env.USE_DEVELOPMENT_AUTH === 'true') {
      warnings.push('WARNING: Development auth enabled in production environment');
    }

    if (!process.env.HTTPS_ENABLED) {
      warnings.push('WARNING: HTTPS not explicitly enabled in production');
    }
  }

  // Log results
  if (warnings.length > 0) {
    warnings.forEach(warning => log.warn(warning));
  }

  if (missingVars.length > 0) {
    log.error('Missing required environment variables:', missingVars);
    return {
      valid: false,
      missing: missingVars,
      warnings
    };
  }

  log.info(`Environment validation passed for ${nodeEnv} mode`);
  return {
    valid: true,
    missing: [],
    warnings
  };
}

/**
 * Middleware to validate environment on server startup
 */
function initializeEnvironmentValidation() {
  const result = validateEnvironment();

  if (!result.valid) {
    const error = new Error(
      `Environment validation failed. Missing required variables: ${result.missing.join(', ')}`
    );
    error.code = 'ENV_VALIDATION_FAILED';
    throw error;
  }

  return result;
}

/**
 * Get environment variable with validation and default
 * @param {string} name - Environment variable name
 * @param {*} defaultValue - Default value if not set
 * @param {boolean} required - Whether the variable is required
 * @returns {*} Environment variable value or default
 */
function getEnvVar(name, defaultValue = undefined, required = false) {
  const value = process.env[name];

  if (required && !value) {
    throw new Error(`Required environment variable ${name} is not set`);
  }

  return value || defaultValue;
}

/**
 * Get environment variable as integer
 * @param {string} name - Environment variable name
 * @param {number} defaultValue - Default value if not set
 * @returns {number} Parsed integer value
 */
function getEnvInt(name, defaultValue = 0) {
  const value = process.env[name];
  if (!value) return defaultValue;

  const parsed = parseInt(value, 10);
  if (isNaN(parsed)) {
    cds.log('env-validation').warn(`Invalid integer value for ${name}: ${value}, using default: ${defaultValue}`);
    return defaultValue;
  }

  return parsed;
}

/**
 * Get environment variable as boolean
 * @param {string} name - Environment variable name
 * @param {boolean} defaultValue - Default value if not set
 * @returns {boolean} Boolean value
 */
function getEnvBool(name, defaultValue = false) {
  const value = process.env[name];
  if (!value) return defaultValue;

  return value.toLowerCase() === 'true' || value === '1';
}

module.exports = {
  validateEnvironment,
  initializeEnvironmentValidation,
  getEnvVar,
  getEnvInt,
  getEnvBool,
  ENV_REQUIREMENTS,
  ENV_DEFAULTS
};