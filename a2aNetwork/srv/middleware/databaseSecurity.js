/**
 * @fileoverview Database Security Hardening Module
 * @description Comprehensive database security middleware implementing multiple layers of protection
 * including SQL injection prevention, connection security, and access control
 * @module DatabaseSecurity
 * @since 1.0.0
 * @author A2A Network Security Team
 */

const cds = require('@sap/cds');
const crypto = require('crypto');
const rateLimit = require('express-rate-limit');

/**
 * Database Security Configuration
 */
const DB_SECURITY_CONFIG = {
    // SQL Injection Prevention
    sqlInjection: {
        enabled: true,
        blockSuspiciousQueries: true,
        logAttempts: true,
        quarantineThreshold: 5,
        patterns: [
            // Common SQL injection patterns
            /(\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|DECLARE)\b.*){2,}/gi,
            /(\b(OR|AND)\s+['"]\s*['"]\s*=\s*['"])/gi,
            /(UNION\s+.*SELECT|SELECT.*FROM.*INFORMATION_SCHEMA|SELECT.*FROM.*SYS\.|SELECT.*FROM.*DUAL)/gi,
            /(\b(XP_|SP_|EXEC|EXECUTE)\s*\()/gi,
            /(WAITFOR\s+DELAY|BENCHMARK\s*\(|SLEEP\s*\()/gi,
            /(\b(LOAD_FILE|INTO\s+OUTFILE|INTO\s+DUMPFILE)\b)/gi,
            /(SCRIPT|JAVASCRIPT|VBSCRIPT|ONLOAD|ONERROR)/gi,
            /(CHAR\s*\(\s*[0-9]+\s*\)|ASCII\s*\(\s*[0-9]+\s*\))/gi
        ],
        whitelistPatterns: [
            // Safe patterns for legitimate queries
            /^SELECT\s+\*\s+FROM\s+[a-zA-Z_][a-zA-Z0-9_]*\s*$/gi,
            /^INSERT\s+INTO\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]+\)\s*VALUES\s*\([^)]+\)\s*$/gi
        ]
    },

    // Connection Security
    connectionSecurity: {
        enabled: true,
        enforceSSL: true,
        maxConnections: 100,
        connectionTimeout: 30000,
        idleTimeout: 300000,
        encryptInTransit: true,
        certificateValidation: true
    },

    // Access Control
    accessControl: {
        enabled: true,
        enforceRBAC: true,
        auditAccess: true,
        restrictSystemTables: true,
        allowedOperations: ['SELECT', 'INSERT', 'UPDATE', 'DELETE'],
        restrictedTables: [
            'INFORMATION_SCHEMA.*',
            'SYS.*',
            'MYSQL.*',
            'PERFORMANCE_SCHEMA.*',
            'security.*'
        ]
    },

    // Data Encryption
    encryption: {
        enabled: true,
        encryptPII: true,
        encryptionAlgorithm: 'aes-256-gcm',
        keyRotationInterval: 86400000, // 24 hours
        piiFields: [
            'email',
            'phone',
            'ssn',
            'creditCard',
            'bankAccount',
            'personalData'
        ]
    },

    // Audit and Monitoring
    audit: {
        enabled: true,
        logAllQueries: false,
        logFailedQueries: true,
        logSlowQueries: true,
        slowQueryThreshold: 5000,
        retentionDays: 90,
        alertOnSuspiciousActivity: true
    }
};

/**
 * Database Security Manager
 * Coordinates all database security features
 */
class DatabaseSecurityManager {
    constructor() {
        this.log = cds.log('db-security');
        this.encryptionKeys = new Map();
        this.suspiciousUsers = new Map();
        this.queryCache = new Map();
        this.auditLog = [];

        // Initialize security components
        this.sqlInjectionDetector = new SQLInjectionDetector();
        this.accessController = new DatabaseAccessController();
        this.encryptionManager = new DataEncryptionManager();
        this.auditLogger = new DatabaseAuditLogger();

        this._initializeSecurityMiddleware();
    }

    /**
     * Initialize security middleware
     */
    _initializeSecurityMiddleware() {
        // Rate limiting for database operations
        this.dbRateLimit = rateLimit({
            windowMs: 60000, // 1 minute
            max: 1000, // max requests per window
            message: 'Database operation rate limit exceeded',
            standardHeaders: true,
            legacyHeaders: false,
            keyGenerator: (req) => {
                return req.user?.id || req.ip || 'anonymous';
            }
        });

        // Start monitoring intervals
        this._startSecurityMonitoring();
    }

    /**
     * Secure database operation handler
     */
    async secureOperation(operation, context) {
        const startTime = Date.now();
        const operationId = this._generateOperationId();

        try {
            // 1. Validate user and context
            await this._validateUserAccess(context);

            // 2. Rate limiting check
            await this._checkRateLimit(context);

            // 3. SQL injection detection
            await this.sqlInjectionDetector.analyze(operation, context);

            // 4. Access control validation
            await this.accessController.validateAccess(operation, context);

            // 5. Data encryption (if needed)
            const secureOperation = await this.encryptionManager.processOperation(operation);

            // 6. Execute operation with monitoring
            const result = await this._executeSecureOperation(secureOperation, context);

            // 7. Decrypt result data (if needed)
            const decryptedResult = await this.encryptionManager.decryptResult(result);

            // 8. Audit logging
            await this.auditLogger.logSuccess(operationId, operation, context, Date.now() - startTime);

            return decryptedResult;

        } catch (error) {
            // Security violation handling
            await this._handleSecurityViolation(error, operation, context, operationId);
            throw error;
        }
    }

    /**
     * Validate user access and authentication
     */
    async _validateUserAccess(context) {
        if (!context.user) {
            throw new SecurityError('AUTHENTICATION_REQUIRED', 'User authentication required for database access');
        }

        // Check for suspended/quarantined users
        if (this.suspiciousUsers.has(context.user.id)) {
            const userStatus = this.suspiciousUsers.get(context.user.id);
            if (userStatus.quarantined) {
                throw new SecurityError('USER_QUARANTINED', 'User access quarantined due to suspicious activity');
            }
        }

        // Validate session token
        if (!await this._validateSessionToken(context.user.sessionToken)) {
            throw new SecurityError('INVALID_SESSION', 'Invalid or expired session token');
        }
    }

    /**
     * Check rate limiting for database operations
     */
    async _checkRateLimit(context) {
        const userId = context.user.id;
        const now = Date.now();
        const windowSize = 60000; // 1 minute

        if (!this.queryCache.has(userId)) {
            this.queryCache.set(userId, []);
        }

        const userQueries = this.queryCache.get(userId);

        // Clean old entries
        const recentQueries = userQueries.filter(timestamp => now - timestamp < windowSize);
        this.queryCache.set(userId, recentQueries);

        // Check limit
        if (recentQueries.length >= DB_SECURITY_CONFIG.connectionSecurity.maxConnections) {
            this._recordSuspiciousActivity(userId, 'RATE_LIMIT_EXCEEDED');
            throw new SecurityError('RATE_LIMIT_EXCEEDED', 'Database operation rate limit exceeded');
        }

        // Record current query
        recentQueries.push(now);
    }

    /**
     * Execute operation with security monitoring
     */
    async _executeSecureOperation(operation, context) {
        const connectionConfig = {
            ssl: DB_SECURITY_CONFIG.connectionSecurity.enforceSSL,
            timeout: DB_SECURITY_CONFIG.connectionSecurity.connectionTimeout,
            multipleStatements: false, // Prevent SQL injection via multiple statements
            charset: 'utf8mb4',
            typeCast: function(field, next) {
                // Custom type casting to prevent data type attacks
                if (field.type === 'TINY' && field.length === 1) {
                    return (field.string() === '1'); // Convert TINYINT to boolean
                }
                return next();
            }
        };

        // Get secure database connection
        const connection = await cds.connect.to('db', connectionConfig);

        try {
            // Set session security parameters
            await connection.run('SET SESSION sql_mode = \'STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO\'');
            await connection.run('SET SESSION max_execution_time = 30000'); // 30 second timeout

            // Execute the operation
            const result = await connection.run(operation);

            return result;

        } finally {
            // Ensure connection is properly closed
            if (connection && connection.disconnect) {
                await connection.disconnect();
            }
        }
    }

    /**
     * Handle security violations
     */
    async _handleSecurityViolation(error, operation, context, operationId) {
        const violation = {
            id: operationId,
            timestamp: new Date().toISOString(),
            userId: context.user?.id || 'anonymous',
            userIP: context.req?.ip || 'unknown',
            errorType: error.code || 'UNKNOWN_SECURITY_ERROR',
            operation: this._sanitizeOperationForLog(operation),
            errorMessage: error.message,
            severity: this._calculateSeverity(error.code),
            blocked: true
        };

        // Log security violation
        this.auditLogger.logViolation(violation);

        // Update suspicious activity tracking
        if (context.user?.id) {
            this._recordSuspiciousActivity(context.user.id, error.code);
        }

        // Alert security team for critical violations
        if (violation.severity === 'CRITICAL') {
            await this._alertSecurityTeam(violation);
        }

        this.log.error('Database security violation:', violation);
    }

    /**
     * Record suspicious activity
     */
    _recordSuspiciousActivity(userId, activityType) {
        if (!this.suspiciousUsers.has(userId)) {
            this.suspiciousUsers.set(userId, {
                activities: [],
                score: 0,
                quarantined: false,
                lastActivity: null
            });
        }

        const userRecord = this.suspiciousUsers.get(userId);
        userRecord.activities.push({
            type: activityType,
            timestamp: Date.now()
        });
        userRecord.lastActivity = Date.now();

        // Calculate risk score
        userRecord.score += this._getActivityRiskScore(activityType);

        // Auto-quarantine high-risk users
        if (userRecord.score >= DB_SECURITY_CONFIG.sqlInjection.quarantineThreshold) {
            userRecord.quarantined = true;
            this.log.warn(`User ${userId} quarantined due to suspicious database activity (score: ${userRecord.score})`);

            // Schedule auto-release (24 hours)
            setTimeout(() => {
                if (this.suspiciousUsers.has(userId)) {
                    const user = this.suspiciousUsers.get(userId);
                    user.quarantined = false;
                    user.score = 0;
                    user.activities = [];
                    this.log.info(`User ${userId} released from quarantine`);
                }
            }, 86400000);
        }
    }

    /**
     * Get activity risk score
     */
    _getActivityRiskScore(activityType) {
        const riskScores = {
            'SQL_INJECTION_ATTEMPT': 5,
            'UNAUTHORIZED_TABLE_ACCESS': 3,
            'RATE_LIMIT_EXCEEDED': 2,
            'INVALID_SESSION': 1,
            'SUSPICIOUS_QUERY_PATTERN': 3,
            'SYSTEM_TABLE_ACCESS': 4,
            'PRIVILEGE_ESCALATION': 5
        };

        return riskScores[activityType] || 1;
    }

    /**
     * Calculate violation severity
     */
    _calculateSeverity(errorCode) {
        const criticalErrors = [
            'SQL_INJECTION_ATTEMPT',
            'PRIVILEGE_ESCALATION',
            'SYSTEM_TABLE_ACCESS',
            'DATA_EXFILTRATION_ATTEMPT'
        ];

        const highErrors = [
            'UNAUTHORIZED_TABLE_ACCESS',
            'SUSPICIOUS_QUERY_PATTERN',
            'AUTHENTICATION_BYPASS_ATTEMPT'
        ];

        if (criticalErrors.includes(errorCode)) return 'CRITICAL';
        if (highErrors.includes(errorCode)) return 'HIGH';
        return 'MEDIUM';
    }

    /**
     * Alert security team
     */
    async _alertSecurityTeam(violation) {
        const alert = {
            type: 'DATABASE_SECURITY_VIOLATION',
            severity: violation.severity,
            message: `Critical database security violation detected: ${violation.errorType}`,
            details: violation,
            timestamp: new Date().toISOString(),
            requiresImmedateAction: true
        };

        // Integrate with security monitoring service
        try {
            const securityService = await cds.connect.to('SecurityMonitoringService');
            await securityService.emit('securityAlert', alert);
        } catch (error) {
            this.log.error('Failed to send security alert:', error);
        }
    }

    /**
     * Validate session token
     */
    async _validateSessionToken(sessionToken) {
        if (!sessionToken) return false;

        try {
            // Implement session validation logic
            // This would integrate with your authentication system
            const decoded = this._decodeSessionToken(sessionToken);
            const now = Date.now();

            return decoded && decoded.exp > now && decoded.iat <= now;
        } catch (error) {
            this.log.warn('Session token validation failed:', error);
            return false;
        }
    }

    /**
     * Generate operation ID for tracking
     */
    _generateOperationId() {
        return crypto.randomBytes(16).toString('hex');
    }

    /**
     * Sanitize operation for logging
     */
    _sanitizeOperationForLog(operation) {
        if (typeof operation === 'string') {
            // Remove potential sensitive data from query string
            return operation
                .replace(/password\s*=\s*['"][^'"]*['"]/gi, 'password=\'***\'')
                .replace(/token\s*=\s*['"][^'"]*['"]/gi, 'token=\'***\'')
                .substring(0, 1000); // Limit log size
        }
        return '[NON_STRING_OPERATION]';
    }

    /**
     * Start security monitoring
     */
    _startSecurityMonitoring() {
        // Clean up old suspicious user records every hour
        setInterval(() => {
            const now = Date.now();
            const cleanupThreshold = 86400000; // 24 hours

            for (const [userId, userRecord] of this.suspiciousUsers.entries()) {
                if (now - userRecord.lastActivity > cleanupThreshold && !userRecord.quarantined) {
                    this.suspiciousUsers.delete(userId);
                }
            }
        }, 3600000);

        // Report security metrics every 5 minutes
        setInterval(() => {
            this._reportSecurityMetrics();
        }, 300000);

        // Rotate encryption keys daily
        setInterval(() => {
            this.encryptionManager.rotateKeys();
        }, DB_SECURITY_CONFIG.encryption.keyRotationInterval);
    }

    /**
     * Report security metrics
     */
    _reportSecurityMetrics() {
        const metrics = {
            suspiciousUsers: this.suspiciousUsers.size,
            quarantinedUsers: Array.from(this.suspiciousUsers.values()).filter(u => u.quarantined).length,
            totalViolations: this.auditLog.length,
            recentViolations: this.auditLog.filter(v => Date.now() - new Date(v.timestamp).getTime() < 3600000).length
        };

        this.log.info('Database Security Metrics:', metrics);
    }

    /**
     * Get security status
     */
    getSecurityStatus() {
        return {
            enabled: true,
            components: {
                sqlInjectionDetection: DB_SECURITY_CONFIG.sqlInjection.enabled,
                accessControl: DB_SECURITY_CONFIG.accessControl.enabled,
                encryption: DB_SECURITY_CONFIG.encryption.enabled,
                audit: DB_SECURITY_CONFIG.audit.enabled
            },
            metrics: {
                suspiciousUsers: this.suspiciousUsers.size,
                quarantinedUsers: Array.from(this.suspiciousUsers.values()).filter(u => u.quarantined).length,
                totalViolations: this.auditLog.length,
                encryptionKeysActive: this.encryptionKeys.size
            },
            lastUpdate: new Date().toISOString()
        };
    }
}

/**
 * SQL Injection Detection Engine
 */
class SQLInjectionDetector {
    constructor() {
        this.log = cds.log('sql-injection-detector');
        this.detectionPatterns = DB_SECURITY_CONFIG.sqlInjection.patterns;
        this.whitelistPatterns = DB_SECURITY_CONFIG.sqlInjection.whitelistPatterns;
    }

    async analyze(operation, context) {
        const query = this._extractQueryString(operation);

        if (!query) return;

        // Check whitelist first
        if (this._isWhitelisted(query)) {
            return;
        }

        // Detect SQL injection patterns
        const detectedPatterns = [];

        for (const pattern of this.detectionPatterns) {
            if (pattern.test(query)) {
                detectedPatterns.push(pattern.toString());
            }
        }

        if (detectedPatterns.length > 0) {
            this.log.warn('SQL injection attempt detected:', {
                userId: context.user?.id,
                userIP: context.req?.ip,
                patterns: detectedPatterns,
                query: query.substring(0, 200)
            });

            throw new SecurityError('SQL_INJECTION_ATTEMPT',
                `SQL injection patterns detected: ${detectedPatterns.join(', ')}`);
        }

        // Additional semantic analysis
        await this._performSemanticAnalysis(query, context);
    }

    _extractQueryString(operation) {
        if (typeof operation === 'string') {
            return operation;
        }

        if (operation && typeof operation === 'object') {
            // Handle CDS query objects
            return JSON.stringify(operation);
        }

        return null;
    }

    _isWhitelisted(query) {
        return this.whitelistPatterns.some(pattern => pattern.test(query));
    }

    async _performSemanticAnalysis(query, context) {
        // Check for suspicious query characteristics
        const suspiciousIndicators = [
            // Unusual UNION usage
            (query.match(/UNION/gi) || []).length > 2,
            // Multiple SELECT statements
            (query.match(/SELECT/gi) || []).length > 3,
            // System table access
            /INFORMATION_SCHEMA|SYS\.|MYSQL\.|PERFORMANCE_SCHEMA/gi.test(query),
            // Time-based attack patterns
            /WAITFOR|DELAY|SLEEP|BENCHMARK/gi.test(query),
            // File system operations
            /LOAD_FILE|INTO\s+OUTFILE|INTO\s+DUMPFILE/gi.test(query),
            // Administrative operations
            /CREATE\s+USER|GRANT|REVOKE|ALTER\s+USER/gi.test(query)
        ];

        const suspiciousScore = suspiciousIndicators.filter(Boolean).length;

        if (suspiciousScore >= 2) {
            throw new SecurityError('SUSPICIOUS_QUERY_PATTERN',
                `Query exhibits ${suspiciousScore} suspicious characteristics`);
        }
    }
}

/**
 * Database Access Controller
 */
class DatabaseAccessController {
    constructor() {
        this.log = cds.log('db-access-controller');
        this.restrictedTables = DB_SECURITY_CONFIG.accessControl.restrictedTables;
        this.allowedOperations = DB_SECURITY_CONFIG.accessControl.allowedOperations;
    }

    async validateAccess(operation, context) {
        if (!DB_SECURITY_CONFIG.accessControl.enabled) return;

        const operationType = this._getOperationType(operation);
        const targetTables = this._getTargetTables(operation);

        // Validate operation type
        if (!this.allowedOperations.includes(operationType)) {
            throw new SecurityError('UNAUTHORIZED_OPERATION',
                `Operation ${operationType} is not allowed`);
        }

        // Validate table access
        for (const table of targetTables) {
            if (this._isRestrictedTable(table)) {
                throw new SecurityError('UNAUTHORIZED_TABLE_ACCESS',
                    `Access to table ${table} is restricted`);
            }

            if (!await this._hasTablePermission(context.user, table, operationType)) {
                throw new SecurityError('INSUFFICIENT_PRIVILEGES',
                    `Insufficient privileges for ${operationType} on ${table}`);
            }
        }
    }

    _getOperationType(operation) {
        if (typeof operation === 'string') {
            const upperQuery = operation.toUpperCase().trim();
            if (upperQuery.startsWith('SELECT')) return 'SELECT';
            if (upperQuery.startsWith('INSERT')) return 'INSERT';
            if (upperQuery.startsWith('UPDATE')) return 'UPDATE';
            if (upperQuery.startsWith('DELETE')) return 'DELETE';
        }

        return 'UNKNOWN';
    }

    _getTargetTables(operation) {
        const tables = [];

        if (typeof operation === 'string') {
            // Simple regex-based table extraction
            const fromMatches = operation.match(/FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)/gi);
            const intoMatches = operation.match(/INTO\s+([a-zA-Z_][a-zA-Z0-9_]*)/gi);
            const updateMatches = operation.match(/UPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)/gi);

            if (fromMatches) tables.push(...fromMatches.map(m => m.split(/\s+/)[1]));
            if (intoMatches) tables.push(...intoMatches.map(m => m.split(/\s+/)[1]));
            if (updateMatches) tables.push(...updateMatches.map(m => m.split(/\s+/)[1]));
        }

        return [...new Set(tables)]; // Remove duplicates
    }

    _isRestrictedTable(tableName) {
        return this.restrictedTables.some(pattern => {
            const regex = new RegExp(pattern.replace('*', '.*'), 'i');
            return regex.test(tableName);
        });
    }

    async _hasTablePermission(user, table, operation) {
        // Implement role-based access control
        const userRoles = user.roles || [];

        // Admin users have access to everything
        if (userRoles.includes('ADMIN') || userRoles.includes('DB_ADMIN')) {
            return true;
        }

        // Security tables restricted to security roles
        if (table.toLowerCase().includes('security') && !userRoles.includes('SECURITY_ADMIN')) {
            return false;
        }

        // Default permission logic
        return true;
    }
}

/**
 * Data Encryption Manager
 */
class DataEncryptionManager {
    constructor() {
        this.log = cds.log('db-encryption');
        this.encryptionKey = this._generateEncryptionKey();
        this.piiFields = DB_SECURITY_CONFIG.encryption.piiFields;
    }

    async processOperation(operation) {
        if (!DB_SECURITY_CONFIG.encryption.enabled) return operation;

        // Encrypt PII data in INSERT/UPDATE operations
        if (typeof operation === 'object' && operation.INSERT) {
            return this._encryptInsertData(operation);
        }

        if (typeof operation === 'object' && operation.UPDATE) {
            return this._encryptUpdateData(operation);
        }

        return operation;
    }

    async decryptResult(result) {
        if (!DB_SECURITY_CONFIG.encryption.enabled || !result) return result;

        // Decrypt PII fields in result set
        if (Array.isArray(result)) {
            return result.map(row => this._decryptRow(row));
        }

        if (typeof result === 'object') {
            return this._decryptRow(result);
        }

        return result;
    }

    _encryptInsertData(operation) {
        // Clone operation to avoid mutating original
        const secureOperation = JSON.parse(JSON.stringify(operation));

        if (secureOperation.INSERT && secureOperation.INSERT.entries) {
            secureOperation.INSERT.entries = secureOperation.INSERT.entries.map(entry => {
                return this._encryptRow(entry);
            });
        }

        return secureOperation;
    }

    _encryptUpdateData(operation) {
        const secureOperation = JSON.parse(JSON.stringify(operation));

        if (secureOperation.UPDATE && secureOperation.UPDATE.set) {
            secureOperation.UPDATE.set = this._encryptRow(secureOperation.UPDATE.set);
        }

        return secureOperation;
    }

    _encryptRow(row) {
        const encryptedRow = { ...row };

        for (const field of this.piiFields) {
            if (encryptedRow[field] && typeof encryptedRow[field] === 'string') {
                encryptedRow[field] = this._encryptValue(encryptedRow[field]);
            }
        }

        return encryptedRow;
    }

    _decryptRow(row) {
        const decryptedRow = { ...row };

        for (const field of this.piiFields) {
            if (decryptedRow[field] && typeof decryptedRow[field] === 'string') {
                try {
                    decryptedRow[field] = this._decryptValue(decryptedRow[field]);
                } catch (error) {
                    // Field might not be encrypted, leave as-is
                    this.log.debug(`Failed to decrypt field ${field}:`, error.message);
                }
            }
        }

        return decryptedRow;
    }

    _encryptValue(value) {
        const cipher = crypto.createCipher(DB_SECURITY_CONFIG.encryption.encryptionAlgorithm, this.encryptionKey);
        let encrypted = cipher.update(value, 'utf8', 'hex');
        encrypted += cipher.final('hex');
        return `ENC:${encrypted}`;
    }

    _decryptValue(encryptedValue) {
        if (!encryptedValue.startsWith('ENC:')) {
            return encryptedValue; // Not encrypted
        }

        const encrypted = encryptedValue.substring(4);
        const decipher = crypto.createDecipher(DB_SECURITY_CONFIG.encryption.encryptionAlgorithm, this.encryptionKey);
        let decrypted = decipher.update(encrypted, 'hex', 'utf8');
        decrypted += decipher.final('utf8');
        return decrypted;
    }

    _generateEncryptionKey() {
        return process.env.DB_ENCRYPTION_KEY || crypto.randomBytes(32).toString('hex');
    }

    rotateKeys() {
        const oldKey = this.encryptionKey;
        this.encryptionKey = crypto.randomBytes(32).toString('hex');

        this.log.info('Database encryption keys rotated');

        // In production, implement key migration logic here
    }
}

/**
 * Database Audit Logger
 */
class DatabaseAuditLogger {
    constructor() {
        this.log = cds.log('db-audit');
        this.auditLog = [];
        this.maxLogSize = 10000;
    }

    async logSuccess(operationId, operation, context, duration) {
        if (!DB_SECURITY_CONFIG.audit.enabled) return;

        const logEntry = {
            id: operationId,
            timestamp: new Date().toISOString(),
            userId: context.user?.id || 'anonymous',
            userIP: context.req?.ip || 'unknown',
            operation: this._sanitizeOperation(operation),
            duration,
            success: true
        };

        if (DB_SECURITY_CONFIG.audit.logAllQueries ||
            (DB_SECURITY_CONFIG.audit.logSlowQueries && duration > DB_SECURITY_CONFIG.audit.slowQueryThreshold)) {
            this._addToAuditLog(logEntry);
        }
    }

    logViolation(violation) {
        this._addToAuditLog(violation);
    }

    _addToAuditLog(entry) {
        this.auditLog.push(entry);

        // Maintain log size
        if (this.auditLog.length > this.maxLogSize) {
            this.auditLog = this.auditLog.slice(-this.maxLogSize);
        }

        // Write to persistent storage in production
        this._persistAuditLog(entry);
    }

    async _persistAuditLog(entry) {
        try {
            // In production, write to dedicated audit database or file
            this.log.info('Audit Log Entry:', entry);
        } catch (error) {
            this.log.error('Failed to persist audit log:', error);
        }
    }

    _sanitizeOperation(operation) {
        if (typeof operation === 'string') {
            return operation.substring(0, 500); // Limit size
        }
        return JSON.stringify(operation).substring(0, 500);
    }

    getAuditLogs(filter = {}) {
        return this.auditLog.filter(entry => {
            if (filter.userId && entry.userId !== filter.userId) return false;
            if (filter.success !== undefined && entry.success !== filter.success) return false;
            if (filter.since && new Date(entry.timestamp) < filter.since) return false;
            return true;
        });
    }
}

/**
 * Security Error Class
 */
class SecurityError extends Error {
    constructor(code, message) {
        super(message);
        this.name = 'SecurityError';
        this.code = code;
        this.timestamp = new Date().toISOString();
    }
}

// Initialize global database security manager
const databaseSecurity = new DatabaseSecurityManager();

module.exports = {
    DatabaseSecurityManager,
    SQLInjectionDetector,
    DatabaseAccessController,
    DataEncryptionManager,
    DatabaseAuditLogger,
    SecurityError,
    databaseSecurity,
    DB_SECURITY_CONFIG
};