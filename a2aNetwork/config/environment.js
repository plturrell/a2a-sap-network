const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const joi = require('joi');
const { LoggerFactory } = require('../../shared/logging/structured-logger');

/**
 * Production-ready Configuration Management
 * Supports multiple environments with validation and security
 */
class EnvironmentConfig {
    constructor() {
        this.env = process.env.NODE_ENV || 'development';
        this.configPath = path.join(__dirname, `${this.env}.env`);
        this.secrets = new Map();
        this.logger = LoggerFactory.createNetworkLogger(this.env);
        
        // Load environment variables
        this.loadEnvironment();
        
        // Validate configuration
        this.validateConfig();
        
        // Load secrets from secure store
        this.loadSecrets();
    }

    /**
     * Load environment variables from files
     */
    loadEnvironment() {
        // Load base .env file
        const baseEnvPath = path.join(__dirname, '.env');
        if (fs.existsSync(baseEnvPath)) {
            dotenv.config({ path: baseEnvPath });
        }

        // Load environment-specific file
        if (fs.existsSync(this.configPath)) {
            dotenv.config({ path: this.configPath, override: true });
        }

        // Load local overrides (not committed to git)
        const localEnvPath = path.join(__dirname, '.env.local');
        if (fs.existsSync(localEnvPath)) {
            dotenv.config({ path: localEnvPath, override: true });
        }

        this.logger.info('Configuration loaded successfully', { 
            environment: this.env,
            configPath: this.configPath,
            operation: 'loadEnvironment'
        });
    }

    /**
     * Configuration schema for validation
     */
    getConfigSchema() {
        return joi.object({
            // Environment
            NODE_ENV: joi.string()
                .valid('development', 'test', 'staging', 'production')
                .default('development'),

            // Server Configuration
            PORT: joi.number().default(4004),
            HOST: joi.string().default('0.0.0.0'),
            API_VERSION: joi.string().default('v1'),
            
            // Database Configuration
            DB_TYPE: joi.string()
                .valid('sqlite', 'postgresql', 'mongodb', 'hana')
                .default('sqlite'),
            DB_CONNECTION_STRING: joi.string().required(),
            DB_POOL_SIZE: joi.number().default(10),
            DB_CONNECTION_TIMEOUT: joi.number().default(30000),
            
            // Redis Configuration
            REDIS_HOST: joi.string().default('localhost'),
            REDIS_PORT: joi.number().default(6379),
            REDIS_PASSWORD: joi.string().allow('').optional(),
            REDIS_DB: joi.number().default(0),
            
            // RabbitMQ Configuration
            RABBITMQ_URL: joi.string().default('amqp://localhost'),
            MQ_PREFETCH: joi.number().default(10),
            MQ_MESSAGE_EXPIRY: joi.number().default(3600000),
            MQ_MAX_RETRIES: joi.number().default(3),
            
            // Authentication
            JWT_SECRET: joi.string().min(32).required(),
            JWT_REFRESH_SECRET: joi.string().min(32).required(),
            ACCESS_TOKEN_EXPIRY: joi.string().default('15m'),
            REFRESH_TOKEN_EXPIRY: joi.string().default('7d'),
            SESSION_TIMEOUT: joi.number().default(3600),
            MAX_CONCURRENT_SESSIONS: joi.number().default(5),
            
            // Security
            BCRYPT_ROUNDS: joi.number().min(10).default(12),
            RATE_LIMIT_WINDOW: joi.number().default(900000), // 15 minutes
            RATE_LIMIT_MAX: joi.number().default(100),
            CORS_ORIGINS: joi.string().default('*'),
            
            // AI/Chat Configuration
            GROK_API_KEY: joi.string().when('NODE_ENV', {
                is: 'production',
                then: joi.required()
            }),
            GROK_BASE_URL: joi.string().default('https://api.openai.com/v1'),
            GROK_MODEL: joi.string().default('gpt-4'),
            GROK_TEMPERATURE: joi.number().min(0).max(2).default(0.7),
            GROK_MAX_TOKENS: joi.number().default(1000),
            
            // WebSocket Configuration
            WS_PORT: joi.number().default(8087),
            WS_HEARTBEAT_INTERVAL: joi.number().default(30000),
            WS_MAX_PAYLOAD: joi.number().default(1048576), // 1MB
            
            // Monitoring
            ENABLE_METRICS: joi.boolean().default(true),
            METRICS_PORT: joi.number().default(9090),
            LOG_LEVEL: joi.string()
                .valid('error', 'warn', 'info', 'debug')
                .default('info'),
            
            // Feature Flags
            ENABLE_CHAT: joi.boolean().default(true),
            ENABLE_BLOCKCHAIN: joi.boolean().default(false),
            ENABLE_ENCRYPTION: joi.boolean().default(true),
            ENABLE_RATE_LIMITING: joi.boolean().default(true),
            
            // External Services
            A2A_SERVICE_URL: joi.string().uri().required(),
            A2A_BASE_URL: joi.string().uri().required(),
            BLOCKCHAIN_RPC_URL: joi.string().uri().optional(),
            
            // Performance
            CACHE_TTL: joi.number().default(3600),
            MAX_CACHE_SIZE: joi.number().default(1000),
            CONNECTION_POOL_SIZE: joi.number().default(10),
            REQUEST_TIMEOUT: joi.number().default(30000)
        });
    }

    /**
     * Validate configuration
     */
    validateConfig() {
        const schema = this.getConfigSchema();
        const { error, value } = schema.validate(process.env, {
            abortEarly: false,
            allowUnknown: true
        });

        if (error) {
            const errorDetails = error.details.map(detail => detail.message);
            this.logger.error('Configuration validation failed', {
                environment: this.env,
                errors: errorDetails,
                operation: 'validateConfig'
            });
            
            if (this.env === 'production') {
                process.exit(1);
            }
        }

        // Apply validated values back to process.env
        Object.assign(process.env, value);
        
        this.logger.info('Configuration validation completed successfully', {
            environment: this.env,
            operation: 'validateConfig'
        });
    }

    /**
     * Load secrets from secure store (e.g., AWS Secrets Manager, HashiCorp Vault)
     */
    async loadSecrets() {
        if (this.env === 'production') {
            try {
                // Example: AWS Secrets Manager
                // const secrets = await this.loadFromAWSSecretsManager();
                
                // Example: HashiCorp Vault
                // const secrets = await this.loadFromVault();
                
                // For now, use environment variables
                this.secrets.set('jwt_secret', process.env.JWT_SECRET);
                this.secrets.set('db_password', process.env.DB_PASSWORD);
                this.secrets.set('api_keys', {
                    grok: process.env.GROK_API_KEY,
                    blockchain: process.env.BLOCKCHAIN_API_KEY
                });

                this.logger.info('Secrets loaded successfully', {
                    environment: this.env,
                    secretsCount: this.secrets.size,
                    operation: 'loadSecrets'
                });
            } catch (error) {
                this.logger.error('Failed to load secrets', {
                    environment: this.env,
                    error: error.message,
                    stack: error.stack,
                    operation: 'loadSecrets'
                });
                if (this.env === 'production') {
                    process.exit(1);
                }
            }
        }
    }

    /**
     * Get configuration value with fallback
     */
    get(key, defaultValue) {
        return process.env[key] || defaultValue;
    }

    /**
     * Get secret value
     */
    getSecret(key) {
        return this.secrets.get(key);
    }

    /**
     * Get typed configuration sections
     */
    getDatabase() {
        return {
            type: this.get('DB_TYPE'),
            connectionString: this.get('DB_CONNECTION_STRING'),
            poolSize: parseInt(this.get('DB_POOL_SIZE')),
            connectionTimeout: parseInt(this.get('DB_CONNECTION_TIMEOUT'))
        };
    }

    getRedis() {
        return {
            host: this.get('REDIS_HOST'),
            port: parseInt(this.get('REDIS_PORT')),
            password: this.get('REDIS_PASSWORD'),
            db: parseInt(this.get('REDIS_DB'))
        };
    }

    getRabbitMQ() {
        return {
            url: this.get('RABBITMQ_URL'),
            prefetch: parseInt(this.get('MQ_PREFETCH')),
            messageExpiry: parseInt(this.get('MQ_MESSAGE_EXPIRY')),
            maxRetries: parseInt(this.get('MQ_MAX_RETRIES'))
        };
    }

    getAuth() {
        return {
            jwtSecret: this.getSecret('jwt_secret') || this.get('JWT_SECRET'),
            jwtRefreshSecret: this.get('JWT_REFRESH_SECRET'),
            accessTokenExpiry: this.get('ACCESS_TOKEN_EXPIRY'),
            refreshTokenExpiry: this.get('REFRESH_TOKEN_EXPIRY'),
            sessionTimeout: parseInt(this.get('SESSION_TIMEOUT')),
            maxConcurrentSessions: parseInt(this.get('MAX_CONCURRENT_SESSIONS'))
        };
    }

    getAI() {
        return {
            apiKey: this.getSecret('api_keys')?.grok || this.get('GROK_API_KEY'),
            baseUrl: this.get('GROK_BASE_URL'),
            model: this.get('GROK_MODEL'),
            temperature: parseFloat(this.get('GROK_TEMPERATURE')),
            maxTokens: parseInt(this.get('GROK_MAX_TOKENS'))
        };
    }

    getWebSocket() {
        return {
            port: parseInt(this.get('WS_PORT')),
            heartbeatInterval: parseInt(this.get('WS_HEARTBEAT_INTERVAL')),
            maxPayload: parseInt(this.get('WS_MAX_PAYLOAD'))
        };
    }

    getFeatureFlags() {
        return {
            chat: this.get('ENABLE_CHAT') === 'true',
            blockchain: this.get('ENABLE_BLOCKCHAIN') === 'true',
            encryption: this.get('ENABLE_ENCRYPTION') === 'true',
            rateLimiting: this.get('ENABLE_RATE_LIMITING') === 'true'
        };
    }

    /**
     * Check if running in production
     */
    isProduction() {
        return this.env === 'production';
    }

    /**
     * Check if running in development
     */
    isDevelopment() {
        return this.env === 'development';
    }

    /**
     * Check if running in test
     */
    isTest() {
        return this.env === 'test';
    }

    /**
     * Export configuration for debugging (masks sensitive values)
     */
    exportConfig() {
        const config = {};
        const sensitiveKeys = ['PASSWORD', 'SECRET', 'KEY', 'TOKEN'];
        
        for (const [key, value] of Object.entries(process.env)) {
            if (key.startsWith('A2A_') || key.startsWith('DB_') || key.startsWith('REDIS_')) {
                // Mask sensitive values
                const isSensitive = sensitiveKeys.some(sensitive => 
                    key.toUpperCase().includes(sensitive)
                );
                
                config[key] = isSensitive ? '***MASKED***' : value;
            }
        }
        
        return config;
    }
}

// Create singleton instance
const config = new EnvironmentConfig();

// Export configuration interface
module.exports = {
    config,
    get: (key, defaultValue) => config.get(key, defaultValue),
    getDatabase: () => config.getDatabase(),
    getRedis: () => config.getRedis(),
    getRabbitMQ: () => config.getRabbitMQ(),
    getAuth: () => config.getAuth(),
    getAI: () => config.getAI(),
    getWebSocket: () => config.getWebSocket(),
    getFeatureFlags: () => config.getFeatureFlags(),
    isProduction: () => config.isProduction(),
    isDevelopment: () => config.isDevelopment(),
    isTest: () => config.isTest()
};