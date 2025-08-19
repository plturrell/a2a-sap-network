const cds = require('@sap/cds');

/**
 * Production-Ready SEAL Configuration
 * Real implementation with Grok API integration and SAP compliance
 * Provides environment-specific settings for SEAL deployment
 */
class SealConfiguration {
    constructor() {
        this.environment = process.env.NODE_ENV || 'development';
        this.config = this._loadConfiguration();
    }

    /**
     * Get complete SEAL configuration
     */
    getConfiguration() {
        return this.config;
    }

    /**
     * Get Grok API configuration
     */
    getGrokConfig() {
        return this.config.grok;
    }

    /**
     * Get reinforcement learning configuration
     */
    getRLConfig() {
        return this.config.reinforcementLearning;
    }

    /**
     * Get SAP compliance configuration
     */
    getComplianceConfig() {
        return this.config.compliance;
    }

    /**
     * Get monitoring configuration
     */
    getMonitoringConfig() {
        return this.config.monitoring;
    }

    /**
     * Validate configuration
     */
    validateConfiguration() {
        const errors = [];
        
        // Validate xAI Grok configuration
        if (!this.config.grok.apiKey) {
            errors.push('XAI_API_KEY or GROK_API_KEY is required');
        }
        
        if (!this.config.grok.baseUrl) {
            errors.push('XAI_BASE_URL is required (should be https://api.x.ai/v1)');
        }
        
        if (this.config.grok.model.startsWith('grok-4') && !this.config.grok.features.reasoning) {
            errors.push('Grok 4 requires reasoning mode to be enabled');
        }
        
        // Validate RL parameters
        const rl = this.config.reinforcementLearning;
        if (rl.learningRate <= 0 || rl.learningRate >= 1) {
            errors.push('Learning rate must be between 0 and 1');
        }
        
        if (rl.discountFactor <= 0 || rl.discountFactor >= 1) {
            errors.push('Discount factor must be between 0 and 1');
        }
        
        // Validate compliance settings
        if (this.config.compliance.auditRetentionYears < 1) {
            errors.push('Audit retention must be at least 1 year');
        }
        
        return {
            isValid: errors.length === 0,
            errors
        };
    }

    /**
     * Load environment-specific configuration
     * @private
     */
    _loadConfiguration() {
        const baseConfig = {
            // xAI Grok API Configuration
            grok: {
                apiKey: process.env.XAI_API_KEY || process.env.GROK_API_KEY,
                baseUrl: process.env.XAI_BASE_URL || 'https://api.x.ai/v1',
                model: process.env.XAI_MODEL || 'grok-4', // Default to Grok 4
                timeout: parseInt(process.env.XAI_TIMEOUT) || 60000, // Increased for Grok 4
                retryAttempts: parseInt(process.env.XAI_RETRY_ATTEMPTS) || 3,
                rateLimiting: {
                    requestsPerMinute: parseInt(process.env.XAI_RATE_LIMIT_RPM) || 60,
                    tokensPerMinute: parseInt(process.env.XAI_RATE_LIMIT_TPM) || 40000
                },
                fallback: {
                    enabled: process.env.XAI_FALLBACK_ENABLED === 'true',
                    model: process.env.XAI_FALLBACK_MODEL || 'grok-beta'
                },
                features: {
                    reasoning: true, // Grok 4 is a reasoning model
                    vision: true, // Grok 4 supports vision
                    structuredOutputs: true, // Supports structured outputs
                    toolCalling: true, // Supports function calling
                    realTimeSearch: process.env.XAI_REAL_TIME_SEARCH === 'true'
                }
            },

            // Reinforcement Learning Configuration
            reinforcementLearning: {
                algorithm: process.env.RL_ALGORITHM || 'Q_LEARNING',
                learningRate: parseFloat(process.env.RL_LEARNING_RATE) || 0.1,
                discountFactor: parseFloat(process.env.RL_DISCOUNT_FACTOR) || 0.95,
                explorationRate: parseFloat(process.env.RL_EXPLORATION_RATE) || 0.1,
                explorationDecay: parseFloat(process.env.RL_EXPLORATION_DECAY) || 0.995,
                minExplorationRate: parseFloat(process.env.RL_MIN_EXPLORATION_RATE) || 0.01,
                episodeMemoryLimit: parseInt(process.env.RL_EPISODE_MEMORY_LIMIT) || 10000,
                batchSize: parseInt(process.env.RL_BATCH_SIZE) || 32,
                targetUpdateFrequency: parseInt(process.env.RL_TARGET_UPDATE_FREQ) || 100,
                multiArmedBandit: {
                    enabled: process.env.RL_MAB_ENABLED === 'true',
                    algorithm: process.env.RL_MAB_ALGORITHM || 'UCB1',
                    confidenceLevel: parseFloat(process.env.RL_MAB_CONFIDENCE) || 0.95
                },
                thompsonSampling: {
                    enabled: process.env.RL_THOMPSON_ENABLED === 'true',
                    priorAlpha: parseFloat(process.env.RL_THOMPSON_PRIOR_ALPHA) || 1.0,
                    priorBeta: parseFloat(process.env.RL_THOMPSON_PRIOR_BETA) || 1.0
                }
            },

            // SEAL Adaptation Configuration
            adaptation: {
                enabled: process.env.SEAL_ADAPTATION_ENABLED !== 'false',
                fewShotLearning: {
                    enabled: process.env.SEAL_FEW_SHOT_ENABLED !== 'false',
                    minExamples: parseInt(process.env.SEAL_FEW_SHOT_MIN_EXAMPLES) || 3,
                    maxExamples: parseInt(process.env.SEAL_FEW_SHOT_MAX_EXAMPLES) || 10,
                    similarityThreshold: parseFloat(process.env.SEAL_FEW_SHOT_SIMILARITY) || 0.8
                },
                continuousLearning: {
                    enabled: process.env.SEAL_CONTINUOUS_LEARNING_ENABLED !== 'false',
                    learningIntervalMinutes: parseInt(process.env.SEAL_CONTINUOUS_INTERVAL) || 30,
                    adaptationTriggerThreshold: parseFloat(process.env.SEAL_ADAPTATION_THRESHOLD) || 0.7,
                    maxAdaptationsPerDay: parseInt(process.env.SEAL_MAX_ADAPTATIONS_DAY) || 10
                },
                selfEditGeneration: {
                    enabled: process.env.SEAL_SELF_EDIT_ENABLED !== 'false',
                    maxEditsPerSession: parseInt(process.env.SEAL_MAX_EDITS_SESSION) || 5,
                    confidenceThreshold: parseFloat(process.env.SEAL_EDIT_CONFIDENCE_THRESHOLD) || 0.6,
                    validationRequired: process.env.SEAL_EDIT_VALIDATION_REQUIRED !== 'false'
                }
            },

            // SAP Compliance Configuration
            compliance: {
                enabled: process.env.SAP_COMPLIANCE_ENABLED !== 'false',
                auditEnabled: process.env.SAP_AUDIT_ENABLED !== 'false',
                auditRetentionYears: parseInt(process.env.SAP_AUDIT_RETENTION_YEARS) || 7,
                dataClassification: {
                    enabled: process.env.SAP_DATA_CLASSIFICATION_ENABLED !== 'false',
                    defaultLevel: process.env.SAP_DEFAULT_DATA_CLASSIFICATION || 'INTERNAL',
                    enforcementMode: process.env.SAP_DATA_CLASSIFICATION_ENFORCEMENT || 'STRICT'
                },
                accessControl: {
                    enabled: process.env.SAP_ACCESS_CONTROL_ENABLED !== 'false',
                    authenticationRequired: process.env.SAP_AUTH_REQUIRED !== 'false',
                    roleBasedAccess: process.env.SAP_RBAC_ENABLED !== 'false',
                    sessionTimeout: parseInt(process.env.SAP_SESSION_TIMEOUT) || 3600000 // 1 hour
                },
                riskManagement: {
                    enabled: process.env.SAP_RISK_MANAGEMENT_ENABLED !== 'false',
                    riskAssessmentRequired: process.env.SAP_RISK_ASSESSMENT_REQUIRED !== 'false',
                    approvalWorkflows: process.env.SAP_APPROVAL_WORKFLOWS_ENABLED !== 'false',
                    riskThresholds: {
                        low: parseFloat(process.env.SAP_RISK_THRESHOLD_LOW) || 0.3,
                        medium: parseFloat(process.env.SAP_RISK_THRESHOLD_MEDIUM) || 0.6,
                        high: parseFloat(process.env.SAP_RISK_THRESHOLD_HIGH) || 0.8,
                        critical: parseFloat(process.env.SAP_RISK_THRESHOLD_CRITICAL) || 1.0
                    }
                }
            },

            // Monitoring and Observability
            monitoring: {
                enabled: process.env.MONITORING_ENABLED !== 'false',
                metricsCollection: {
                    enabled: process.env.METRICS_COLLECTION_ENABLED !== 'false',
                    collectionInterval: parseInt(process.env.METRICS_COLLECTION_INTERVAL) || 60000, // 1 minute
                    retentionDays: parseInt(process.env.METRICS_RETENTION_DAYS) || 30
                },
                alerting: {
                    enabled: process.env.ALERTING_ENABLED !== 'false',
                    channels: (process.env.ALERT_CHANNELS || 'email,webhook').split(','),
                    thresholds: {
                        responseTime: parseInt(process.env.ALERT_RESPONSE_TIME_MS) || 5000,
                        errorRate: parseFloat(process.env.ALERT_ERROR_RATE) || 0.05,
                        cpuUsage: parseFloat(process.env.ALERT_CPU_USAGE) || 0.8,
                        memoryUsage: parseFloat(process.env.ALERT_MEMORY_USAGE) || 0.85
                    }
                },
                tracing: {
                    enabled: process.env.TRACING_ENABLED !== 'false',
                    serviceName: process.env.TRACE_SERVICE_NAME || 'seal-enhanced-glean',
                    samplingRate: parseFloat(process.env.TRACE_SAMPLING_RATE) || 0.1,
                    endpoint: process.env.TRACE_ENDPOINT || 'http://localhost:14268/api/traces'
                },
                logging: {
                    level: process.env.LOG_LEVEL || 'info',
                    format: process.env.LOG_FORMAT || 'json',
                    auditLogging: process.env.AUDIT_LOGGING_ENABLED !== 'false',
                    sensitiveDataMasking: process.env.SENSITIVE_DATA_MASKING_ENABLED !== 'false'
                }
            },

            // Performance Configuration
            performance: {
                caching: {
                    enabled: process.env.CACHING_ENABLED !== 'false',
                    ttl: parseInt(process.env.CACHE_TTL) || 300000, // 5 minutes
                    maxSize: parseInt(process.env.CACHE_MAX_SIZE) || 1000,
                    strategy: process.env.CACHE_STRATEGY || 'LRU'
                },
                concurrency: {
                    maxConcurrentOperations: parseInt(process.env.MAX_CONCURRENT_OPERATIONS) || 10,
                    queueSize: parseInt(process.env.OPERATION_QUEUE_SIZE) || 100,
                    timeoutMs: parseInt(process.env.OPERATION_TIMEOUT_MS) || 300000 // 5 minutes
                },
                optimization: {
                    batchProcessing: process.env.BATCH_PROCESSING_ENABLED !== 'false',
                    batchSize: parseInt(process.env.BATCH_SIZE) || 50,
                    parallelProcessing: process.env.PARALLEL_PROCESSING_ENABLED !== 'false',
                    maxParallelTasks: parseInt(process.env.MAX_PARALLEL_TASKS) || 5
                }
            },

            // Security Configuration
            security: {
                encryption: {
                    enabled: process.env.ENCRYPTION_ENABLED !== 'false',
                    algorithm: process.env.ENCRYPTION_ALGORITHM || 'aes-256-gcm',
                    keyRotationDays: parseInt(process.env.KEY_ROTATION_DAYS) || 90
                },
                apiSecurity: {
                    rateLimiting: process.env.API_RATE_LIMITING_ENABLED !== 'false',
                    requestsPerMinute: parseInt(process.env.API_RATE_LIMIT_RPM) || 100,
                    ipWhitelist: process.env.API_IP_WHITELIST ? process.env.API_IP_WHITELIST.split(',') : [],
                    corsEnabled: process.env.API_CORS_ENABLED !== 'false',
                    corsOrigins: process.env.API_CORS_ORIGINS ? process.env.API_CORS_ORIGINS.split(',') : ['*']
                },
                dataProtection: {
                    enabled: process.env.DATA_PROTECTION_ENABLED !== 'false',
                    anonymization: process.env.DATA_ANONYMIZATION_ENABLED !== 'false',
                    retention: {
                        personalData: parseInt(process.env.PERSONAL_DATA_RETENTION_DAYS) || 365,
                        analyticsData: parseInt(process.env.ANALYTICS_DATA_RETENTION_DAYS) || 1095, // 3 years
                        auditData: parseInt(process.env.AUDIT_DATA_RETENTION_DAYS) || 2555 // 7 years
                    }
                }
            },

            // Development and Testing
            development: {
                debugMode: process.env.DEBUG_MODE === 'true',
                verboseLogging: process.env.VERBOSE_LOGGING === 'true',
                mockExternalServices: process.env.MOCK_EXTERNAL_SERVICES === 'true',
                testDataEnabled: process.env.TEST_DATA_ENABLED === 'true',
                performanceProfiling: process.env.PERFORMANCE_PROFILING_ENABLED === 'true'
            }
        };

        // Apply environment-specific overrides
        return this._applyEnvironmentOverrides(baseConfig);
    }

    /**
     * Apply environment-specific configuration overrides
     * @private
     */
    _applyEnvironmentOverrides(config) {
        switch (this.environment) {
            case 'production':
                return this._applyProductionOverrides(config);
            case 'staging':
                return this._applyStagingOverrides(config);
            case 'development':
                return this._applyDevelopmentOverrides(config);
            case 'test':
                return this._applyTestOverrides(config);
            default:
                return config;
        }
    }

    /**
     * Apply production environment overrides
     * @private
     */
    _applyProductionOverrides(config) {
        return {
            ...config,
            grok: {
                ...config.grok,
                timeout: 60000, // Longer timeout for production
                retryAttempts: 5,
                rateLimiting: {
                    requestsPerMinute: 120,
                    tokensPerMinute: 80000
                }
            },
            reinforcementLearning: {
                ...config.reinforcementLearning,
                explorationRate: 0.05, // Lower exploration in production
                episodeMemoryLimit: 50000 // More memory for production
            },
            monitoring: {
                ...config.monitoring,
                metricsCollection: {
                    ...config.monitoring.metricsCollection,
                    collectionInterval: 30000 // More frequent collection
                },
                alerting: {
                    ...config.monitoring.alerting,
                    thresholds: {
                        responseTime: 3000, // Stricter thresholds
                        errorRate: 0.02,
                        cpuUsage: 0.7,
                        memoryUsage: 0.8
                    }
                },
                tracing: {
                    ...config.monitoring.tracing,
                    samplingRate: 0.01 // Lower sampling in production
                }
            },
            development: {
                ...config.development,
                debugMode: false,
                verboseLogging: false,
                mockExternalServices: false,
                testDataEnabled: false
            }
        };
    }

    /**
     * Apply staging environment overrides
     * @private
     */
    _applyStagingOverrides(config) {
        return {
            ...config,
            grok: {
                ...config.grok,
                rateLimiting: {
                    requestsPerMinute: 80,
                    tokensPerMinute: 60000
                }
            },
            monitoring: {
                ...config.monitoring,
                tracing: {
                    ...config.monitoring.tracing,
                    samplingRate: 0.1
                }
            },
            development: {
                ...config.development,
                debugMode: true,
                verboseLogging: true,
                performanceProfiling: true
            }
        };
    }

    /**
     * Apply development environment overrides
     * @private
     */
    _applyDevelopmentOverrides(config) {
        return {
            ...config,
            grok: {
                ...config.grok,
                rateLimiting: {
                    requestsPerMinute: 30,
                    tokensPerMinute: 20000
                }
            },
            compliance: {
                ...config.compliance,
                riskManagement: {
                    ...config.compliance.riskManagement,
                    approvalWorkflows: false // Simplified for development
                }
            },
            monitoring: {
                ...config.monitoring,
                tracing: {
                    ...config.monitoring.tracing,
                    samplingRate: 1.0 // Full tracing in development
                }
            },
            development: {
                ...config.development,
                debugMode: true,
                verboseLogging: true,
                mockExternalServices: true,
                testDataEnabled: true,
                performanceProfiling: true
            }
        };
    }

    /**
     * Apply test environment overrides
     * @private
     */
    _applyTestOverrides(config) {
        return {
            ...config,
            grok: {
                ...config.grok,
                timeout: 5000, // Shorter timeout for tests
                retryAttempts: 1
            },
            reinforcementLearning: {
                ...config.reinforcementLearning,
                episodeMemoryLimit: 100 // Smaller memory for tests
            },
            monitoring: {
                ...config.monitoring,
                enabled: false // Disable monitoring in tests
            },
            development: {
                ...config.development,
                debugMode: false,
                verboseLogging: false,
                mockExternalServices: true,
                testDataEnabled: true
            }
        };
    }
}

module.exports = SealConfiguration;