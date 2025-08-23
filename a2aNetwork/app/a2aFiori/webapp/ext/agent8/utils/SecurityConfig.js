sap.ui.define([], function () {
    "use strict";

    /**
     * Security Configuration for Agent 8 Data Manager
     * Central configuration for all security policies and settings
     */
    return {
        
        // Content Security Policy Settings
        contentSecurityPolicy: {
            directives: {
                "default-src": ["'self'"],
                "script-src": ["'self'", "'unsafe-inline'", "*.sapcdn.com", "*.sap.com"],
                "style-src": ["'self'", "'unsafe-inline'", "*.sapcdn.com", "*.sap.com"],
                "img-src": ["'self'", "data:", "*.sapcdn.com", "*.sap.com"],
                "font-src": ["'self'", "*.sapcdn.com", "*.sap.com"],
                "connect-src": ["'self'", "/a2a/agent8/v1/*"],
                "frame-src": ["'none'"],
                "object-src": ["'none'"],
                "base-uri": ["'self'"],
                "form-action": ["'self'"]
            },
            reportUri: "/security/csp-report",
            reportOnly: false
        },

        // HTTP Security Headers
        securityHeaders: {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "SAMEORIGIN",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        },

        // Authentication and Authorization Settings
        authentication: {
            required: true,
            methods: ["SAML", "OAuth2", "BasicAuth"],
            sessionTimeout: 3600, // 1 hour in seconds
            maxConcurrentSessions: 3,
            strongPasswordRequired: true,
            passwordPolicy: {
                minLength: 12,
                requireUppercase: true,
                requireLowercase: true,
                requireNumbers: true,
                requireSpecialChars: true,
                maxAge: 90, // days
                preventReuse: 12 // last N passwords
            }
        },

        // CSRF Protection Settings
        csrfProtection: {
            enabled: true,
            tokenValidityPeriod: 1800, // 30 minutes in seconds
            sameSitePolicy: "Strict",
            secureCookies: true,
            httpOnlyCookies: true,
            customHeaderRequired: true,
            allowedOrigins: ["*.sap.com", "localhost"]
        },

        // Input Validation Rules
        inputValidation: {
            maxFieldLength: 1000,
            allowedCharsets: "UTF-8",
            sanitizeHtml: true,
            validateJsonStructure: true,
            maxFileUploadSize: 52428800, // 50MB
            allowedFileTypes: [".json", ".xml", ".csv", ".txt"],
            sql: {
                preventSqlInjection: true,
                allowedKeywords: ["SELECT", "FROM", "WHERE", "ORDER BY", "GROUP BY"],
                blockedPatterns: ["UNION", "DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER", "EXEC"]
            }
        },

        // Output Encoding Settings
        outputEncoding: {
            htmlEncode: true,
            jsEncode: true,
            urlEncode: true,
            xmlEncode: true,
            csvEncode: true,
            preventDataExfiltration: true
        },

        // Session Management
        sessionManagement: {
            regenerateIdOnLogin: true,
            secureSessionStorage: true,
            sessionIdEntropy: 256, // bits
            domainSpecific: true,
            ipValidation: true,
            userAgentValidation: true,
            concurrentSessionLimit: 3,
            idleTimeout: 1800, // 30 minutes
            absoluteTimeout: 14400 // 4 hours
        },

        // Audit and Logging Configuration
        auditLogging: {
            enabled: true,
            logLevel: "INFO",
            includeRequestDetails: true,
            logSensitiveData: false,
            maxLogFileSize: 104857600, // 100MB
            logRotationDays: 30,
            centralLogging: true,
            encryptLogs: true,
            events: {
                authentication: true,
                authorization: true,
                dataAccess: true,
                configurationChanges: true,
                securityViolations: true,
                systemErrors: true
            }
        },

        // Rate Limiting and DoS Protection
        rateLimiting: {
            enabled: true,
            requestsPerMinute: 100,
            requestsPerHour: 1000,
            burstAllowance: 20,
            penaltyPeriod: 300, // 5 minutes
            blockedIpTtl: 3600, // 1 hour
            whitelistedIps: ["127.0.0.1", "::1"]
        },

        // Data Protection and Privacy
        dataProtection: {
            encryptSensitiveData: true,
            encryptionAlgorithm: "AES-256-GCM",
            dataMinimization: true,
            dataRetentionDays: 2555, // 7 years
            anonymizePersonalData: true,
            rightToForgotten: true,
            dataMasking: {
                enabled: true,
                patterns: {
                    email: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g,
                    phone: /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g,
                    ssn: /\b\d{3}-\d{2}-\d{4}\b/g,
                    creditCard: /\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b/g
                }
            }
        },

        // Error Handling and Information Disclosure
        errorHandling: {
            hideStackTraces: true,
            genericErrorMessages: true,
            logDetailedErrors: true,
            customErrorPages: true,
            sanitizeErrorOutput: true,
            maxErrorLogSize: 10485760, // 10MB
            errorReporting: {
                enabled: true,
                endpoint: "/security/error-reports",
                includeUserContext: false,
                rateLimited: true
            }
        },

        // File Upload Security
        fileUpload: {
            enabled: true,
            maxFileSize: 52428800, // 50MB
            allowedExtensions: [".json", ".xml", ".csv", ".txt"],
            virusScanEnabled: true,
            quarantineSuspiciousFiles: true,
            validateFileContent: true,
            uploadDirectory: "/secure/uploads/",
            temporaryFileCleanup: 3600 // 1 hour
        },

        // Database Security
        database: {
            useParameterizedQueries: true,
            minimumPrivilegeAccess: true,
            connectionEncryption: true,
            auditDatabaseAccess: true,
            preventSqlInjection: true,
            connectionPoolSecurity: {
                maxConnections: 50,
                connectionTimeout: 30000, // 30 seconds
                idleTimeout: 300000, // 5 minutes
                validateConnections: true
            }
        },

        // Third-Party Integration Security
        thirdPartyIntegrations: {
            whitelistedDomains: ["*.sap.com", "api.trusted-partner.com"],
            apiKeyValidation: true,
            certificateValidation: true,
            timeoutSettings: {
                connectionTimeout: 10000, // 10 seconds
                readTimeout: 30000, // 30 seconds
                maxRetries: 3
            },
            dataSharing: {
                requireExplicitConsent: true,
                encryptTransmittedData: true,
                auditDataSharing: true
            }
        },

        // Development and Testing Security
        development: {
            disableDebugInProduction: true,
            removeTestAccounts: true,
            disableLoggingInProduction: false,
            sourceMapProtection: true,
            environmentVariableValidation: true,
            dependencySecurityScanning: true
        },

        // Monitoring and Alerting
        monitoring: {
            securityEventMonitoring: true,
            realTimeAlerting: true,
            anomalyDetection: true,
            threatIntelligence: true,
            alertThresholds: {
                failedLogins: 5,
                suspiciousRequests: 10,
                dataExfiltrationAttempts: 1,
                privilegeEscalation: 1
            },
            notificationEndpoints: [
                "/security/alerts",
                "security@a2a.network"
            ]
        },

        // Compliance Settings
        compliance: {
            gdprCompliance: true,
            hipaaCompliance: false,
            sox404Compliance: true,
            iso27001Compliance: true,
            sapSecurityStandards: true,
            auditTrailRetention: 2555, // 7 years in days
            dataProcessingRecords: true,
            privacyPolicyEnforcement: true
        },

        // Security Testing Configuration
        securityTesting: {
            penetrationTestingSchedule: "quarterly",
            vulnerabilityScanning: "weekly",
            codeSecurityAnalysis: "on-commit",
            dependencyVulnerabilityChecks: "daily",
            securityBaselines: {
                owasp: "4.0",
                sans: "25",
                nist: "800-53"
            }
        },

        // Incident Response
        incidentResponse: {
            enabled: true,
            automaticResponse: {
                blockSuspiciousIps: true,
                quarantineMaliciousFiles: true,
                escalateCriticalEvents: true,
                notifySecurityTeam: true
            },
            responsePlaybooks: {
                dataBreachResponse: "/security/playbooks/data-breach.json",
                malwareDetection: "/security/playbooks/malware.json",
                unauthorizedAccess: "/security/playbooks/unauthorized-access.json"
            }
        },

        /**
         * Get the current security configuration
         * @returns {Object} Current security configuration
         */
        getConfig: function() {
            return JSON.parse(JSON.stringify(this)); // Deep clone to prevent modification
        },

        /**
         * Validate if a security setting is enabled
         * @param {string} category - Security category
         * @param {string} setting - Specific setting
         * @returns {boolean} True if enabled
         */
        isEnabled: function(category, setting) {
            return this[category] && this[category][setting] === true;
        },

        /**
         * Get security policy for a specific area
         * @param {string} area - Security area (e.g., 'authentication', 'csrf')
         * @returns {Object} Security policy configuration
         */
        getPolicy: function(area) {
            return this[area] || {};
        },

        /**
         * Check if an action is allowed by security policy
         * @param {string} action - Action to check
         * @param {Object} context - Context for the action
         * @returns {boolean} True if allowed
         */
        isActionAllowed: function(action, context) {
            // Implement specific policy checks based on action and context
            switch (action) {
                case 'file_upload':
                    return this._checkFileUploadPolicy(context);
                case 'data_export':
                    return this._checkDataExportPolicy(context);
                case 'admin_access':
                    return this._checkAdminAccessPolicy(context);
                default:
                    return true; // Default allow for undefined actions
            }
        },

        // Private methods for policy checks
        _checkFileUploadPolicy: function(context) {
            if (!this.fileUpload.enabled) return false;
            
            const fileSize = context.fileSize || 0;
            const fileExtension = context.fileExtension || '';
            
            return fileSize <= this.fileUpload.maxFileSize &&
                   this.fileUpload.allowedExtensions.includes(fileExtension.toLowerCase());
        },

        _checkDataExportPolicy: function(context) {
            const recordCount = context.recordCount || 0;
            const userRole = context.userRole || 'guest';
            
            // Limit data export based on user role and record count
            const maxRecords = {
                'admin': 1000000,
                'manager': 100000,
                'user': 10000,
                'guest': 0
            };
            
            return recordCount <= (maxRecords[userRole] || 0);
        },

        _checkAdminAccessPolicy: function(context) {
            const userRole = context.userRole || 'guest';
            const ipAddress = context.ipAddress || '';
            
            // Only allow admin access from specific roles and IP ranges
            return userRole === 'admin' && this._isWhitelistedIp(ipAddress);
        },

        _isWhitelistedIp: function(ipAddress) {
            // Simple check - in production, use proper IP range validation
            return this.rateLimiting.whitelistedIps.some(whitelistedIp => {
                return ipAddress.startsWith(whitelistedIp) || whitelistedIp === ipAddress;
            });
        }
    };
});