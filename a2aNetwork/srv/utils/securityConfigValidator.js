/**
 * @fileoverview Security Configuration Validator
 * @description Validates all security configurations before application startup
 * @module securityConfigValidator
 * @since 4.0.0
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const cds = require('@sap/cds');

/**
 * Security Configuration Requirements
 */
const SECURITY_REQUIREMENTS = {
    // Environment variables that MUST be set in production
    REQUIRED_ENV_VARS: {
        production: [
            'SESSION_SECRET',
            'JWT_PRIVATE_KEY',
            'JWT_PUBLIC_KEY',
            'JWT_KEY_PASSPHRASE',
            'CONFIG_ENCRYPTION_KEY',
            'MASTER_KEY_SECRET',
            'MASTER_KEY_SALT',
            'BLOCKCHAIN_PRIVATE_KEY',
            'DATABASE_ENCRYPTION_KEY',
            'ALLOWED_ORIGINS',
            'WS_ALLOWED_ORIGINS',
            'COOKIE_DOMAIN',
            'ALERT_WEBHOOK_URL'
        ],
        development: [
            'NODE_ENV'
        ]
    },

    // Minimum key lengths
    KEY_LENGTHS: {
        SESSION_SECRET: 64,
        CONFIG_ENCRYPTION_KEY: 32,
        MASTER_KEY_SECRET: 64,
        DATABASE_ENCRYPTION_KEY: 32,
        API_KEY: 32
    },

    // Password requirements
    PASSWORD_REQUIREMENTS: {
        minLength: 12,
        requireUppercase: true,
        requireLowercase: true,
        requireNumbers: true,
        requireSpecialChars: true
    },

    // Security headers that must be present
    REQUIRED_HEADERS: [
        'Strict-Transport-Security',
        'X-Content-Type-Options',
        'X-Frame-Options',
        'X-XSS-Protection',
        'Content-Security-Policy',
        'Referrer-Policy'
    ],

    // Blocked file patterns
    DANGEROUS_FILES: [
        '.env',
        '.env.local',
        '.env.production',
        'private.key',
        'privatekey.pem',
        'id_rsa',
        '.pem',
        '.p12',
        '.pfx',
        '.key'
    ],

    // Vulnerable dependencies to check
    VULNERABLE_PACKAGES: {
        'jsonwebtoken': '<9.0.0',
        'express': '<4.17.3',
        'helmet': '<4.6.0',
        'bcrypt': '<5.0.1'
    }
};

/**
 * Security Configuration Validator
 */
class SecurityConfigValidator {
    constructor() {
        this.errors = [];
        this.warnings = [];
        this.info = [];
        this.log = cds.log('security-validator');
    }

    /**
     * Run complete security validation
     */
    async runValidation() {
        this.errors = [];
        this.warnings = [];
        this.info = [];

        const startTime = Date.now();

        try {
            // 1. Environment configuration
            this.validateEnvironmentVariables();

            // 2. Cryptographic keys
            this.validateCryptographicKeys();

            // 3. File system security
            await this.validateFileSystemSecurity();

            // 4. Dependencies
            await this.validateDependencies();

            // 5. SSL/TLS configuration
            this.validateSSLConfiguration();

            // 6. Authentication configuration
            this.validateAuthConfiguration();

            // 7. Database security
            this.validateDatabaseSecurity();

            // 8. API security
            this.validateAPISecurity();

            // 9. Production readiness
            if (process.env.NODE_ENV === 'production') {
                this.validateProductionReadiness();
            }

            const duration = Date.now() - startTime;

            return {
                valid: this.errors.length === 0,
                errors: this.errors,
                warnings: this.warnings,
                info: this.info,
                duration,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            this.errors.push({
                category: 'VALIDATION_ERROR',
                message: `Security validation failed: ${error.message}`,
                severity: 'CRITICAL'
            });

            return {
                valid: false,
                errors: this.errors,
                warnings: this.warnings,
                info: this.info,
                duration: Date.now() - startTime,
                timestamp: new Date().toISOString()
            };
        }
    }

    /**
     * Validate environment variables
     */
    validateEnvironmentVariables() {
        const env = process.env.NODE_ENV || 'development';
        const required = SECURITY_REQUIREMENTS.REQUIRED_ENV_VARS[env] || [];

        // Check required variables
        for (const varName of required) {
            if (!process.env[varName]) {
                this.errors.push({
                    category: 'ENV_CONFIG',
                    variable: varName,
                    message: `Required environment variable ${varName} is not set`,
                    severity: 'HIGH'
                });
            }
        }

        // Check for insecure values
        if (process.env.SESSION_SECRET === 'dev-secret' && env === 'production') {
            this.errors.push({
                category: 'ENV_CONFIG',
                variable: 'SESSION_SECRET',
                message: 'SESSION_SECRET contains default development value in production',
                severity: 'CRITICAL'
            });
        }

        // Validate CORS origins
        const allowedOrigins = process.env.ALLOWED_ORIGINS;
        if (allowedOrigins && allowedOrigins.includes('*') && env === 'production') {
            this.warnings.push({
                category: 'CORS_CONFIG',
                message: 'CORS allows all origins (*) in production',
                severity: 'HIGH'
            });
        }

        // Check for development flags in production
        if (env === 'production') {
            const devFlags = [
                'USE_DEVELOPMENT_AUTH',
                'DISABLE_SECURITY',
                'SKIP_AUTH',
                'DEBUG'
            ];

            for (const flag of devFlags) {
                if (process.env[flag] === 'true') {
                    this.errors.push({
                        category: 'ENV_CONFIG',
                        variable: flag,
                        message: `Development flag ${flag} is enabled in production`,
                        severity: 'CRITICAL'
                    });
                }
            }
        }
    }

    /**
     * Validate cryptographic keys
     */
    validateCryptographicKeys() {
        // Check key lengths
        for (const [keyName, minLength] of Object.entries(SECURITY_REQUIREMENTS.KEY_LENGTHS)) {
            const keyValue = process.env[keyName];
            if (keyValue && Buffer.from(keyValue, 'hex').length < minLength) {
                this.errors.push({
                    category: 'CRYPTO_CONFIG',
                    key: keyName,
                    message: `${keyName} is too short (minimum ${minLength} bytes)`,
                    severity: 'HIGH'
                });
            }
        }

        // Check JWT configuration
        if (process.env.JWT_PRIVATE_KEY && process.env.JWT_PUBLIC_KEY) {
            try {
                // Validate RSA key pair
                const privateKey = process.env.JWT_PRIVATE_KEY;
                const publicKey = process.env.JWT_PUBLIC_KEY;

                // Test sign and verify
                const testPayload = { test: true };
                const jwt = require('jsonwebtoken');
                const token = jwt.sign(testPayload, privateKey, { algorithm: 'RS256' });
                jwt.verify(token, publicKey, { algorithms: ['RS256'] });

                this.info.push({
                    category: 'JWT_CONFIG',
                    message: 'JWT RSA key pair validated successfully'
                });
            } catch (error) {
                this.errors.push({
                    category: 'JWT_CONFIG',
                    message: `Invalid JWT key pair: ${error.message}`,
                    severity: 'CRITICAL'
                });
            }
        }

        // Check for weak keys
        const weakPatterns = [
            /^0+$/,
            /^1234/,
            /^abcd/i,
            /password/i,
            /secret/i
        ];

        const keysToCheck = [
            'SESSION_SECRET',
            'CONFIG_ENCRYPTION_KEY',
            'MASTER_KEY_SECRET'
        ];

        for (const keyName of keysToCheck) {
            const keyValue = process.env[keyName];
            if (keyValue) {
                for (const pattern of weakPatterns) {
                    if (pattern.test(keyValue)) {
                        this.errors.push({
                            category: 'CRYPTO_CONFIG',
                            key: keyName,
                            message: `${keyName} contains weak pattern`,
                            severity: 'CRITICAL'
                        });
                    }
                }
            }
        }
    }

    /**
     * Validate file system security
     */
    async validateFileSystemSecurity() {
        const projectRoot = process.cwd();

        // Check for exposed sensitive files
        for (const dangerousFile of SECURITY_REQUIREMENTS.DANGEROUS_FILES) {
            const fullPath = path.join(projectRoot, dangerousFile);

            try {
                const stats = await fs.promises.stat(fullPath);
                if (stats.isFile()) {
                    // Check if file is in .gitignore
                    const gitignorePath = path.join(projectRoot, '.gitignore');
                    const gitignoreContent = await fs.promises.readFile(gitignorePath, 'utf8');

                    if (!gitignoreContent.includes(dangerousFile)) {
                        this.errors.push({
                            category: 'FILE_SECURITY',
                            file: dangerousFile,
                            message: `Sensitive file ${dangerousFile} exists and is not in .gitignore`,
                            severity: 'CRITICAL'
                        });
                    } else {
                        this.warnings.push({
                            category: 'FILE_SECURITY',
                            file: dangerousFile,
                            message: `Sensitive file ${dangerousFile} exists in project`,
                            severity: 'MEDIUM'
                        });
                    }
                }
            } catch (error) {
                // File doesn't exist - good
            }
        }

        // Check file permissions
        const criticalFiles = [
            'srv/server.js',
            'srv/middleware/auth.js',
            'srv/middleware/security.js'
        ];

        for (const file of criticalFiles) {
            const fullPath = path.join(projectRoot, file);
            try {
                const stats = await fs.promises.stat(fullPath);
                const mode = stats.mode & parseInt('777', 8);

                if (mode & parseInt('002', 8)) {
                    this.warnings.push({
                        category: 'FILE_PERMISSIONS',
                        file,
                        message: `File ${file} is world-writable`,
                        severity: 'HIGH'
                    });
                }
            } catch (error) {
                // File doesn't exist
            }
        }
    }

    /**
     * Validate dependencies
     */
    async validateDependencies() {
        try {
            const packageJsonPath = path.join(process.cwd(), 'package.json');
            const packageJson = JSON.parse(await fs.readFile(packageJsonPath, 'utf8'));

            const allDeps = {
                ...packageJson.dependencies,
                ...packageJson.devDependencies
            };

            // Check for vulnerable versions
            for (const [pkg, versionReq] of Object.entries(SECURITY_REQUIREMENTS.VULNERABLE_PACKAGES)) {
                if (allDeps[pkg]) {
                    const installedVersion = allDeps[pkg];
                    // Simple version check - in production use semver
                    if (installedVersion < versionReq) {
                        this.errors.push({
                            category: 'DEPENDENCIES',
                            package: pkg,
                            message: `Package ${pkg} version ${installedVersion} has known vulnerabilities`,
                            severity: 'HIGH'
                        });
                    }
                }
            }

            // Check for suspicious packages
            const suspiciousPatterns = [
                /typosquat/i,
                /malware/i,
                /backdoor/i,
                /node-ipc/  // Known compromised package
            ];

            for (const [pkg, version] of Object.entries(allDeps)) {
                for (const pattern of suspiciousPatterns) {
                    if (pattern.test(pkg)) {
                        this.warnings.push({
                            category: 'DEPENDENCIES',
                            package: pkg,
                            message: `Suspicious package name: ${pkg}`,
                            severity: 'MEDIUM'
                        });
                    }
                }
            }
        } catch (error) {
            this.warnings.push({
                category: 'DEPENDENCIES',
                message: `Could not validate dependencies: ${error.message}`,
                severity: 'LOW'
            });
        }
    }

    /**
     * Validate SSL/TLS configuration
     */
    validateSSLConfiguration() {
        if (process.env.NODE_ENV === 'production') {
            // Check HTTPS enforcement
            if (!process.env.FORCE_HTTPS || process.env.FORCE_HTTPS !== 'true') {
                this.errors.push({
                    category: 'SSL_CONFIG',
                    message: 'HTTPS is not enforced in production',
                    severity: 'HIGH'
                });
            }

            // Check TLS version
            if (process.env.TLS_MIN_VERSION && process.env.TLS_MIN_VERSION < '1.2') {
                this.errors.push({
                    category: 'SSL_CONFIG',
                    message: 'TLS minimum version is below 1.2',
                    severity: 'HIGH'
                });
            }
        }
    }

    /**
     * Validate authentication configuration
     */
    validateAuthConfiguration() {
        // Check password policy
        const passwordMinLength = parseInt(process.env.PASSWORD_MIN_LENGTH || '8');
        if (passwordMinLength < SECURITY_REQUIREMENTS.PASSWORD_REQUIREMENTS.minLength) {
            this.warnings.push({
                category: 'AUTH_CONFIG',
                message: `Password minimum length (${passwordMinLength}) is below recommended (${SECURITY_REQUIREMENTS.PASSWORD_REQUIREMENTS.minLength})`,
                severity: 'MEDIUM'
            });
        }

        // Check session timeout
        const sessionTimeout = parseInt(process.env.SESSION_TIMEOUT || '3600000');
        if (sessionTimeout > 86400000) { // 24 hours
            this.warnings.push({
                category: 'AUTH_CONFIG',
                message: 'Session timeout is longer than 24 hours',
                severity: 'MEDIUM'
            });
        }

        // Check account lockout
        if (!process.env.LOCKOUT_THRESHOLD) {
            this.warnings.push({
                category: 'AUTH_CONFIG',
                message: 'Account lockout threshold not configured',
                severity: 'MEDIUM'
            });
        }
    }

    /**
     * Validate database security
     */
    validateDatabaseSecurity() {
        // Check database encryption
        if (process.env.NODE_ENV === 'production' && !process.env.DATABASE_ENCRYPTION_KEY) {
            this.errors.push({
                category: 'DATABASE_CONFIG',
                message: 'Database encryption key not set in production',
                severity: 'HIGH'
            });
        }

        // Check connection string security
        const dbUrl = process.env.DATABASE_URL;
        if (dbUrl) {
            // Check for plaintext passwords in connection string
            if (dbUrl.includes('@') && !dbUrl.includes('****')) {
                this.warnings.push({
                    category: 'DATABASE_CONFIG',
                    message: 'Database connection string may contain plaintext password',
                    severity: 'HIGH'
                });
            }

            // Check for SSL/TLS
            if (process.env.NODE_ENV === 'production' && !dbUrl.includes('ssl=true')) {
                this.warnings.push({
                    category: 'DATABASE_CONFIG',
                    message: 'Database connection does not enforce SSL',
                    severity: 'MEDIUM'
                });
            }
        }
    }

    /**
     * Validate API security
     */
    validateAPISecurity() {
        // Check API rate limits
        if (!process.env.API_RATE_LIMIT) {
            this.warnings.push({
                category: 'API_CONFIG',
                message: 'API rate limiting not configured',
                severity: 'MEDIUM'
            });
        }

        // Check API versioning
        if (!process.env.API_VERSION) {
            this.info.push({
                category: 'API_CONFIG',
                message: 'API versioning not configured'
            });
        }
    }

    /**
     * Validate production readiness
     */
    validateProductionReadiness() {
        // Check monitoring configuration
        if (!process.env.MONITORING_ENABLED || process.env.MONITORING_ENABLED !== 'true') {
            this.warnings.push({
                category: 'PRODUCTION',
                message: 'Monitoring is not enabled',
                severity: 'MEDIUM'
            });
        }

        // Check error handling
        if (!process.env.ERROR_REPORTING_URL) {
            this.warnings.push({
                category: 'PRODUCTION',
                message: 'Error reporting URL not configured',
                severity: 'LOW'
            });
        }

        // Check backup configuration
        if (!process.env.BACKUP_ENABLED) {
            this.warnings.push({
                category: 'PRODUCTION',
                message: 'Backup strategy not configured',
                severity: 'MEDIUM'
            });
        }

        // Check security headers
        this.info.push({
            category: 'PRODUCTION',
            message: 'Ensure all required security headers are configured in production'
        });
    }

    /**
     * Generate security report
     */
    generateReport(validationResult) {
        const report = {
            summary: {
                valid: validationResult.valid,
                errorCount: validationResult.errors.length,
                warningCount: validationResult.warnings.length,
                infoCount: validationResult.info.length,
                timestamp: validationResult.timestamp,
                duration: `${validationResult.duration}ms`
            },
            errors: validationResult.errors,
            warnings: validationResult.warnings,
            info: validationResult.info,
            recommendations: this.generateRecommendations(validationResult)
        };

        return report;
    }

    /**
     * Generate recommendations based on findings
     */
    generateRecommendations(validationResult) {
        const recommendations = [];

        if (validationResult.errors.some(e => e.category === 'ENV_CONFIG')) {
            recommendations.push({
                priority: 'HIGH',
                action: 'Review and set all required environment variables',
                documentation: 'https://docs.a2a-network.com/security/env-config'
            });
        }

        if (validationResult.errors.some(e => e.category === 'CRYPTO_CONFIG')) {
            recommendations.push({
                priority: 'CRITICAL',
                action: 'Regenerate all cryptographic keys with proper entropy',
                documentation: 'https://docs.a2a-network.com/security/crypto-keys'
            });
        }

        if (validationResult.warnings.some(w => w.category === 'AUTH_CONFIG')) {
            recommendations.push({
                priority: 'MEDIUM',
                action: 'Review and strengthen authentication configuration',
                documentation: 'https://docs.a2a-network.com/security/auth-config'
            });
        }

        return recommendations;
    }
}

// Export validator
module.exports = {
    SecurityConfigValidator,
    validateSecurity: async () => {
        const validator = new SecurityConfigValidator();
        const result = await validator.runValidation();
        const report = validator.generateReport(result);

        // Log results
        if (!result.valid) {
            cds.log('security').error('Security validation failed', report);

            // Exit in production if validation fails
            if (process.env.NODE_ENV === 'production') {
                process.exit(1);
            }
        } else {
            cds.log('security').info('Security validation passed', report.summary);
        }

        return report;
    },
    SECURITY_REQUIREMENTS
};