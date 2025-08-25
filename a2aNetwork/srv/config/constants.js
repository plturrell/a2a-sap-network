/**
 * @fileoverview Configuration Constants for A2A Network Services
 * @since 1.0.0
 * @module constants
 *
 * Centralizes all magic numbers, configuration values, and business
 * constants for consistent usage across the SAP A2A Network platform
 */

module.exports = {
    // Blockchain Configuration
    BLOCKCHAIN: {
        DEFAULT_GAS_LIMIT: 500000,
        TRANSACTION_TIMEOUT: 30000,
        MAX_RETRY_ATTEMPTS: 3,
        BLOCK_CONFIRMATION_COUNT: 2
    },

    // Reputation System
    REPUTATION: {
        MIN_SCORE: 0,
        MAX_SCORE: 200,
        DEFAULT_SCORE: 100,
        SCORE_INCREMENT: 10,
        SCORE_DECREMENT: 5
    },

    // Performance Monitoring
    MONITORING: {
        HEALTH_CHECK_INTERVAL: 60000,  // 1 minute
        METRICS_FLUSH_INTERVAL: 5000,  // 5 seconds
        TRACE_RETENTION_DAYS: 7,
        ALERT_THRESHOLD_CPU: 80,
        ALERT_THRESHOLD_MEMORY: 90,
        ALERT_THRESHOLD_RESPONSE_TIME: 1000
    },

    // Database Operations
    DATABASE: {
        DEFAULT_PAGE_SIZE: 1000,
        MAX_PAGE_SIZE: 5000,
        CONNECTION_TIMEOUT: 10000,
        QUERY_TIMEOUT: 30000
    },

    // Network Configuration
    NETWORK: {
        DEFAULT_PORT: 4004,
        REQUEST_TIMEOUT: 30000,
        MAX_CONNECTIONS: 1000,
        RATE_LIMIT_WINDOW: 900000,  // 15 minutes
        RATE_LIMIT_MAX_REQUESTS: 100
    },

    // Security
    SECURITY: {
        SESSION_TIMEOUT: 3600000,  // 1 hour
        TOKEN_EXPIRY: 86400000,    // 24 hours
        MAX_LOGIN_ATTEMPTS: 5,
        LOCKOUT_DURATION: 900000   // 15 minutes
    },

    // Workflow Execution
    WORKFLOW: {
        DEFAULT_TIMEOUT: 300000,   // 5 minutes
        MAX_EXECUTION_TIME: 1800000, // 30 minutes
        MAX_STEPS_PER_WORKFLOW: 50,
        RETRY_DELAY: 5000
    }
};