sap.ui.define([
    "sap/base/Log",
    "sap/base/security/encodeHTML"
], function (Log, encodeHTML) {
    "use strict";

    return {
        /**
         * Get CSRF token for secure requests
         * @returns {Promise<string>} CSRF token
         */
        getCSRFToken: function() {
            return new Promise(function(resolve, reject) {
                jQuery.ajax({
                    url: '/sap/bc/rest/csrf',
                    method: 'GET',
                    headers: {
                        'X-Requested-With': 'XMLHttpRequest'
                    },
                    success: function(data, textStatus, xhr) {
                        const token = xhr.getResponseHeader('X-CSRF-Token');
                        if (token) {
                            resolve(token);
                        } else {
                            reject(new Error('No CSRF token received'));
                        }
                    },
                    error: function(xhr, status, error) {
                        reject(new Error('Failed to fetch CSRF token: ' + error));
                    }
                });
            });
        },

        /**
         * Safely execute OData function with CSRF protection
         * @param {sap.ui.model.odata.v2.ODataModel} oModel - OData model
         * @param {string} sFunctionName - Function name
         * @param {object} mParameters - Parameters
         * @returns {Promise}
         */
        secureCallFunction: function(oModel, sFunctionName, mParameters) {
            return this.getCSRFToken().then(function(token) {
                const secureParams = Object.assign({}, mParameters);
                
                // Add CSRF token to headers
                if (!secureParams.headers) {
                    secureParams.headers = {};
                }
                secureParams.headers['X-CSRF-Token'] = token;
                
                return new Promise(function(resolve, reject) {
                    const originalSuccess = secureParams.success;
                    const originalError = secureParams.error;
                    
                    secureParams.success = function(data, response) {
                        Log.info("Secure SQL function call successful", sFunctionName);
                        if (originalSuccess) {
                            originalSuccess(data, response);
                        }
                        resolve(data);
                    };
                    
                    secureParams.error = function(error) {
                        Log.error("Secure SQL function call failed", error);
                        if (originalError) {
                            originalError(error);
                        }
                        reject(error);
                    };
                    
                    oModel.callFunction(sFunctionName, secureParams);
                });
            });
        },

        /**
         * Advanced SQL injection prevention and validation
         * @param {string} sql - SQL query to validate
         * @param {object} params - Query parameters
         * @param {object} options - Validation options
         * @returns {object} Comprehensive validation result
         */
        validateSQL: function(sql, params, options) {
            options = options || {};
            
            if (!sql || typeof sql !== 'string') {
                return {
                    isValid: false,
                    sanitized: "",
                    errors: ["SQL query must be a non-empty string"],
                    securityScore: 0,
                    riskLevel: "CRITICAL"
                };
            }

            const errors = [];
            const warnings = [];
            let sanitized = sql.trim();
            let securityScore = 100;
            let riskLevel = "LOW";

            // Enhanced dangerous patterns detection
            const criticalPatterns = [
                // Script injection
                { pattern: /<script[^>]*>.*?<\/script>/gi, penalty: 50, message: "Script injection detected" },
                { pattern: /javascript:/gi, penalty: 30, message: "JavaScript protocol detected" },
                { pattern: /on\w+\s*=/gi, penalty: 30, message: "Event handler injection detected" },
                
                // Code execution
                { pattern: /eval\s*\(/gi, penalty: 50, message: "Code evaluation function detected" },
                { pattern: /new\s+Function/gi, penalty: 50, message: "Dynamic function creation detected" },
                { pattern: /setTimeout\s*\(/gi, penalty: 40, message: "Timing function detected" },
                { pattern: /setInterval\s*\(/gi, penalty: 40, message: "Interval function detected" },
                
                // Dangerous SQL injection patterns
                { pattern: /union\s+select/gi, penalty: 40, message: "UNION SELECT injection pattern" },
                { pattern: /;\s*drop\s+(table|database|schema)/gi, penalty: 50, message: "DROP statement injection" },
                { pattern: /;\s*delete\s+from/gi, penalty: 45, message: "DELETE injection pattern" },
                { pattern: /;\s*update\s+.*\s+set/gi, penalty: 40, message: "UPDATE injection pattern" },
                { pattern: /;\s*insert\s+into/gi, penalty: 35, message: "INSERT injection pattern" },
                { pattern: /exec\s*\(/gi, penalty: 50, message: "EXEC function detected" },
                { pattern: /execute\s*\(/gi, penalty: 50, message: "EXECUTE function detected" },
                { pattern: /xp_cmdshell/gi, penalty: 50, message: "Command shell access detected" },
                { pattern: /sp_executesql/gi, penalty: 45, message: "Dynamic SQL execution detected" },
                
                // Advanced evasion techniques
                { pattern: /char\s*\(/gi, penalty: 30, message: "Character conversion detected" },
                { pattern: /ascii\s*\(/gi, penalty: 30, message: "ASCII conversion detected" },
                { pattern: /0x[0-9a-f]+/gi, penalty: 25, message: "Hexadecimal encoding detected" },
                { pattern: /\+\s*concat/gi, penalty: 35, message: "Concatenation bypass attempt" },
                { pattern: /load_file\s*\(/gi, penalty: 45, message: "File access function detected" },
                { pattern: /into\s+outfile/gi, penalty: 45, message: "File output detected" }
            ];

            criticalPatterns.forEach(({pattern, penalty, message}) => {
                if (pattern.test(sanitized)) {
                    errors.push(message);
                    securityScore -= penalty;
                    sanitized = sanitized.replace(pattern, '');
                }
            });

            // Enhanced injection indicators
            const injectionIndicators = [
                // Always true/false conditions
                { pattern: /1\s*=\s*1/gi, penalty: 35, message: "Always-true condition detected" },
                { pattern: /1\s*=\s*0/gi, penalty: 30, message: "Always-false condition detected" },
                { pattern: /'.*'\s*=\s*'.*'/gi, penalty: 25, message: "String comparison injection" },
                
                // Comment-based injection
                { pattern: /--[^\r\n]*/gi, penalty: 20, message: "SQL comment detected" },
                { pattern: /\/\*[\s\S]*?\*\//gi, penalty: 20, message: "Block comment detected" },
                { pattern: /#[^\r\n]*/gi, penalty: 15, message: "Hash comment detected" },
                
                // Concatenation and encoding
                { pattern: /\|\|/gi, penalty: 15, message: "String concatenation detected" },
                { pattern: /concat\s*\(/gi, penalty: 20, message: "CONCAT function detected" },
                { pattern: /substring\s*\(/gi, penalty: 15, message: "SUBSTRING function detected" },
                
                // Information disclosure
                { pattern: /information_schema/gi, penalty: 30, message: "Information schema access" },
                { pattern: /sys\./gi, penalty: 25, message: "System table access" },
                { pattern: /mysql\./gi, penalty: 25, message: "MySQL system database access" },
                { pattern: /pg_catalog/gi, penalty: 25, message: "PostgreSQL catalog access" },
                { pattern: /user\s*\(/gi, penalty: 20, message: "User function detected" },
                { pattern: /version\s*\(/gi, penalty: 20, message: "Version function detected" },
                
                // Time-based attacks
                { pattern: /sleep\s*\(/gi, penalty: 35, message: "SLEEP function detected" },
                { pattern: /waitfor\s+delay/gi, penalty: 35, message: "WAITFOR DELAY detected" },
                { pattern: /benchmark\s*\(/gi, penalty: 35, message: "BENCHMARK function detected" },
                { pattern: /pg_sleep\s*\(/gi, penalty: 35, message: "PostgreSQL sleep detected" }
            ];

            injectionIndicators.forEach(({pattern, penalty, message}) => {
                if (pattern.test(sanitized)) {
                    warnings.push(message);
                    securityScore -= penalty;
                }
            });

            // Advanced parameter validation
            if (params) {
                Object.entries(params).forEach(([key, value]) => {
                    if (typeof value === 'string') {
                        // Check for encoded injection attempts
                        const decodedValue = decodeURIComponent(value).toLowerCase();
                        injectionIndicators.forEach(({pattern, message}) => {
                            if (pattern.test(decodedValue)) {
                                errors.push(`SQL injection in parameter '${key}': ${message}`);
                                securityScore -= 20;
                            }
                        });
                        
                        // Check for excessive parameter length
                        if (value.length > 1000) {
                            warnings.push(`Parameter '${key}' is unusually long (${value.length} characters)`);
                            securityScore -= 10;
                        }
                    }
                });
            }

            // Check for proper parameterization
            const hasLiterals = /'[^']*'/g.test(sanitized);
            const hasParameters = /[?$:]\w*|\{\w+\}/g.test(sanitized);
            
            if (hasLiterals && !hasParameters && !options.allowLiterals) {
                warnings.push("SQL contains string literals - consider using parameterized queries");
                securityScore -= 15;
            }

            // Determine risk level based on security score
            if (securityScore < 30) {
                riskLevel = "CRITICAL";
            } else if (securityScore < 50) {
                riskLevel = "HIGH";
            } else if (securityScore < 70) {
                riskLevel = "MEDIUM";
            } else if (securityScore < 85) {
                riskLevel = "LOW";
            } else {
                riskLevel = "MINIMAL";
            }

            // Additional context-aware validation
            if (options.allowedOperations) {
                const firstWord = sanitized.trim().split(/\s+/)[0].toUpperCase();
                if (!options.allowedOperations.includes(firstWord)) {
                    errors.push(`Operation '${firstWord}' is not allowed in this context`);
                    securityScore -= 30;
                }
            }

            return {
                isValid: errors.length === 0 && securityScore >= (options.minSecurityScore || 50),
                sanitized: sanitized,
                errors: errors,
                warnings: warnings,
                original: sql,
                securityScore: Math.max(0, securityScore),
                riskLevel: riskLevel,
                detectedPatterns: criticalPatterns.filter(p => p.pattern.test(sql)).map(p => p.message)
            };
        },

        /**
         * Sanitize SQL parameters
         * @param {any} param - Parameter to sanitize
         * @returns {any} Sanitized parameter
         */
        sanitizeSQLParameter: function(param) {
            if (typeof param === 'string') {
                // Remove dangerous characters and patterns
                return param
                    .replace(/['"]/g, '') // Remove quotes
                    .replace(/[;]/g, '') // Remove semicolons
                    .replace(/--/g, '') // Remove SQL comments
                    .replace(/\/\*/g, '') // Remove block comment start
                    .replace(/\*\//g, '') // Remove block comment end
                    .replace(/\\/g, '') // Remove backslashes
                    .trim();
            }
            return param;
        },

        /**
         * Create secure parameterized query with advanced templates
         * @param {string} baseQuery - Base SQL query with placeholders
         * @param {object} parameters - Parameters to bind
         * @param {object} options - Additional security options
         * @returns {object} Secure query object
         */
        createParameterizedQuery: function(baseQuery, parameters, options) {
            options = options || {};
            
            const validation = this.validateSQL(baseQuery, parameters, {
                allowLiterals: options.allowLiterals || false,
                allowedOperations: options.allowedOperations,
                minSecurityScore: options.minSecurityScore || 60
            });
            
            if (!validation.isValid) {
                return {
                    isValid: false,
                    errors: validation.errors,
                    warnings: validation.warnings,
                    query: null,
                    parameters: null,
                    securityScore: validation.securityScore,
                    riskLevel: validation.riskLevel
                };
            }

            // Advanced parameter sanitization and validation
            const sanitizedParams = {};
            const parameterErrors = [];
            
            if (parameters) {
                Object.entries(parameters).forEach(([key, value]) => {
                    const sanitized = this.sanitizeSQLParameterAdvanced(key, value, options);
                    if (sanitized.isValid) {
                        sanitizedParams[key] = sanitized.value;
                    } else {
                        parameterErrors.push(...sanitized.errors);
                    }
                });
            }

            // Generate secure query with proper parameter binding
            const secureQuery = this._generateSecureQuery(validation.sanitized, sanitizedParams, options);

            return {
                isValid: parameterErrors.length === 0,
                query: secureQuery.query,
                parameters: sanitizedParams,
                errors: parameterErrors,
                warnings: validation.warnings,
                securityScore: validation.securityScore,
                riskLevel: validation.riskLevel,
                queryHash: this._generateQueryHash(secureQuery.query),
                parameterTypes: secureQuery.parameterTypes
            };
        },

        /**
         * Advanced parameter sanitization
         * @private
         */
        sanitizeSQLParameterAdvanced: function(key, value, options) {
            const errors = [];
            
            if (typeof value === 'string') {
                // Check parameter length limits
                if (value.length > (options.maxParameterLength || 1000)) {
                    errors.push(`Parameter '${key}' exceeds maximum length`);
                }
                
                // Enhanced sanitization
                let sanitized = value
                    .replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '') // Remove control characters
                    .replace(/[\\]/g, '\\\\') // Escape backslashes
                    .replace(/["]/g, '\\"') // Escape double quotes
                    .replace(/[']/g, "\\'") // Escape single quotes
                    .trim();
                
                // Additional validation for specific parameter types
                if (key.toLowerCase().includes('id')) {
                    // ID parameters should be numeric or UUID format
                    if (!/^[0-9a-fA-F-]+$/.test(sanitized)) {
                        errors.push(`Parameter '${key}' should be numeric or UUID format`);
                    }
                }
                
                if (key.toLowerCase().includes('email')) {
                    // Email validation
                    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(sanitized)) {
                        errors.push(`Parameter '${key}' should be valid email format`);
                    }
                }
                
                return {
                    isValid: errors.length === 0,
                    value: sanitized,
                    errors: errors
                };
            }
            
            // Handle other parameter types
            if (typeof value === 'number') {
                if (!Number.isFinite(value)) {
                    errors.push(`Parameter '${key}' must be a finite number`);
                }
                return {
                    isValid: errors.length === 0,
                    value: value,
                    errors: errors
                };
            }
            
            if (typeof value === 'boolean') {
                return {
                    isValid: true,
                    value: value,
                    errors: []
                };
            }
            
            // Default handling
            return {
                isValid: true,
                value: this.sanitizeSQLParameter(value),
                errors: []
            };
        },

        /**
         * Generate secure query with proper parameter binding
         * @private
         */
        _generateSecureQuery: function(baseQuery, parameters, options) {
            const parameterTypes = {};
            let processedQuery = baseQuery;
            
            // Replace named parameters with positional parameters for better security
            Object.entries(parameters).forEach(([key, value]) => {
                const paramType = this._detectParameterType(value);
                parameterTypes[key] = paramType;
                
                // Use different placeholder syntax based on database type
                const placeholder = options.databaseType === 'postgresql' ? `$${Object.keys(parameterTypes).length}` : '?';
                processedQuery = processedQuery.replace(new RegExp(`:${key}\\b`, 'g'), placeholder);
            });
            
            return {
                query: processedQuery,
                parameterTypes: parameterTypes
            };
        },

        /**
         * Detect parameter data type for proper binding
         * @private
         */
        _detectParameterType: function(value) {
            if (typeof value === 'number') {
                return Number.isInteger(value) ? 'INTEGER' : 'DECIMAL';
            }
            if (typeof value === 'boolean') {
                return 'BOOLEAN';
            }
            if (value instanceof Date) {
                return 'TIMESTAMP';
            }
            if (typeof value === 'string') {
                // Try to detect special string types
                if (/^\d{4}-\d{2}-\d{2}$/.test(value)) return 'DATE';
                if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(value)) return 'TIMESTAMP';
                if (/^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/.test(value)) return 'UUID';
                return 'VARCHAR';
            }
            return 'VARCHAR'; // Default
        },

        /**
         * Generate hash for query caching and audit purposes
         * @private
         */
        _generateQueryHash: function(query) {
            // Simple hash function for query fingerprinting
            let hash = 0;
            for (let i = 0; i < query.length; i++) {
                const char = query.charCodeAt(i);
                hash = ((hash << 5) - hash) + char;
                hash = hash & hash; // Convert to 32-bit integer
            }
            return Math.abs(hash).toString(16);
        },

        /**
         * Rate limiting for SQL operations
         * @param {string} userId - User identifier
         * @param {string} operation - Operation type (query, execute, etc.)
         * @returns {Promise<boolean>} Whether operation is allowed
         */
        checkRateLimit: function(userId, operation) {
            return new Promise(function(resolve) {
                const rateLimits = {
                    query: { maxRequests: 100, window: 3600000 }, // 100 queries per hour
                    execute: { maxRequests: 50, window: 3600000 }, // 50 executions per hour
                    create: { maxRequests: 20, window: 3600000 },   // 20 creates per hour
                    translate: { maxRequests: 200, window: 3600000 } // 200 translations per hour
                };
                
                const limit = rateLimits[operation] || rateLimits.query;
                const key = `rateLimit_${userId}_${operation}`;
                const now = Date.now();
                
                // Get existing rate limit data from session storage
                let rateLimitData;
                try {
                    rateLimitData = JSON.parse(sessionStorage.getItem(key)) || { requests: [], firstRequest: now };
                } catch (e) {
                    rateLimitData = { requests: [], firstRequest: now };
                }
                
                // Clean up old requests outside the time window
                const windowStart = now - limit.window;
                rateLimitData.requests = rateLimitData.requests.filter(timestamp => timestamp > windowStart);
                
                // Check if under limit
                if (rateLimitData.requests.length < limit.maxRequests) {
                    rateLimitData.requests.push(now);
                    sessionStorage.setItem(key, JSON.stringify(rateLimitData));
                    resolve(true);
                } else {
                    Log.warning(`Rate limit exceeded for user ${userId} operation ${operation}`);
                    resolve(false);
                }
            });
        },

        /**
         * Audit logging for SQL operations
         * @param {string} operation - Operation type
         * @param {object} details - Operation details
         * @param {string} result - Operation result
         */
        auditLog: function(operation, details, result) {
            const auditEntry = {
                timestamp: new Date().toISOString(),
                operation: operation,
                user: this._getCurrentUser(),
                sessionId: this._getSessionId(),
                details: {
                    queryHash: details.queryHash,
                    queryType: details.queryType,
                    databaseType: details.databaseType,
                    securityScore: details.securityScore,
                    riskLevel: details.riskLevel,
                    complexityScore: details.complexityScore,
                    parameterCount: details.parameterCount,
                    tables: details.tables,
                    hasSubqueries: details.hasSubqueries,
                    hasJoins: details.hasJoins
                },
                result: result,
                ipAddress: this._getClientIP(),
                userAgent: navigator.userAgent.substring(0, 200) // Truncate for security
            };
            
            // Log to console for development (in production, send to audit service)
            Log.info("SQL Audit Log", JSON.stringify(auditEntry));
            
            // Store in session for compliance reporting
            this._storeAuditEntry(auditEntry);
        },

        /**
         * Secure natural language query processing
         * @param {string} naturalQuery - Natural language query
         * @returns {object} Security analysis result
         */
        validateNaturalLanguageQuery: function(naturalQuery) {
            const results = {
                isValid: true,
                sanitized: "",
                warnings: [],
                errors: [],
                securityScore: 100,
                extractedEntities: [],
                intents: [],
                riskFactors: []
            };
            
            if (!naturalQuery || typeof naturalQuery !== 'string') {
                results.isValid = false;
                results.errors.push("Natural language query must be a non-empty string");
                return results;
            }
            
            // Sanitize input
            results.sanitized = naturalQuery
                .replace(/[<>]/g, '') // Remove potentially dangerous characters
                .replace(/javascript:/gi, '') // Remove javascript protocols
                .replace(/script/gi, '') // Remove script references
                .trim();
            
            // Check query length
            if (results.sanitized.length > 1000) {
                results.warnings.push("Query is very long, consider simplifying");
                results.securityScore -= 10;
            }
            
            // Check for suspicious patterns in natural language
            const suspiciousPatterns = [
                { pattern: /drop\s+table/gi, penalty: 30, message: "Potentially destructive operation requested" },
                { pattern: /delete\s+all/gi, penalty: 25, message: "Bulk delete operation requested" },
                { pattern: /system|admin|password|credential/gi, penalty: 20, message: "System-related terms detected" },
                { pattern: /execute|exec|command/gi, penalty: 20, message: "Command execution terms detected" },
                { pattern: /union|join.*select/gi, penalty: 15, message: "Complex query patterns detected" },
                { pattern: /bypass|hack|inject/gi, penalty: 40, message: "Security-related terms detected" }
            ];
            
            suspiciousPatterns.forEach(({pattern, penalty, message}) => {
                if (pattern.test(results.sanitized)) {
                    results.riskFactors.push(message);
                    results.securityScore -= penalty;
                }
            });
            
            // Extract potential entities (tables, columns, operations)
            results.extractedEntities = this._extractEntitiesFromNL(results.sanitized);
            
            // Determine intents
            results.intents = this._extractIntentsFromNL(results.sanitized);
            
            // Validate against allowed operations
            const allowedIntents = ['SELECT', 'COUNT', 'SHOW', 'DESCRIBE', 'EXPLAIN'];
            const hasDisallowedIntent = results.intents.some(intent => !allowedIntents.includes(intent));
            
            if (hasDisallowedIntent) {
                results.warnings.push("Query may contain operations requiring elevated permissions");
                results.securityScore -= 15;
            }
            
            // Final validation
            results.isValid = results.errors.length === 0 && results.securityScore >= 50;
            
            return results;
        },

        /**
         * Extract entities from natural language
         * @private
         */
        _extractEntitiesFromNL: function(query) {
            const entities = [];
            const lowerQuery = query.toLowerCase();
            
            // Common table name patterns
            const tablePatterns = [
                /(?:from|in|on)\s+(users?|customers?|orders?|products?|employees?|sales?)/gi,
                /(table|database)\s+(\w+)/gi
            ];
            
            tablePatterns.forEach(pattern => {
                let match;
                while ((match = pattern.exec(query)) !== null) {
                    entities.push({
                        type: 'TABLE',
                        value: match[1] || match[2],
                        confidence: 70
                    });
                }
            });
            
            return entities;
        },

        /**
         * Extract intents from natural language
         * @private
         */
        _extractIntentsFromNL: function(query) {
            const intents = [];
            const lowerQuery = query.toLowerCase();
            
            if (/show|list|get|find|display|view/gi.test(query)) {
                intents.push('SELECT');
            }
            if (/count|how many|number of/gi.test(query)) {
                intents.push('COUNT');
            }
            if (/add|insert|create new/gi.test(query)) {
                intents.push('INSERT');
            }
            if (/update|change|modify/gi.test(query)) {
                intents.push('UPDATE');
            }
            if (/delete|remove/gi.test(query)) {
                intents.push('DELETE');
            }
            if (/describe|explain|structure/gi.test(query)) {
                intents.push('DESCRIBE');
            }
            
            return intents;
        },

        /**
         * Get current user identifier
         * @private
         */
        _getCurrentUser: function() {
            try {
                // Try to get from SAP UI5 user context
                return sap.ushell?.Container?.getUser()?.getId() || 'anonymous';
            } catch (e) {
                return 'anonymous';
            }
        },

        /**
         * Get session identifier
         * @private
         */
        _getSessionId: function() {
            try {
                return sessionStorage.getItem('sessionId') || 'unknown';
            } catch (e) {
                return 'unknown';
            }
        },

        /**
         * Get client IP address (placeholder - would be populated by backend)
         * @private
         */
        _getClientIP: function() {
            return 'client-ip-placeholder';
        },

        /**
         * Store audit entry for compliance
         * @private
         */
        _storeAuditEntry: function(auditEntry) {
            try {
                const auditLog = JSON.parse(sessionStorage.getItem('sqlAuditLog')) || [];
                auditLog.push(auditEntry);
                
                // Keep only last 100 entries in session
                if (auditLog.length > 100) {
                    auditLog.splice(0, auditLog.length - 100);
                }
                
                sessionStorage.setItem('sqlAuditLog', JSON.stringify(auditLog));
            } catch (e) {
                Log.error("Failed to store audit entry", e);
            }
        },

        /**
         * Safely escape HTML content
         * @param {string} content - Content to escape
         * @returns {string} Escaped content
         */
        escapeHTML: function(content) {
            if (!content) return "";
            return encodeHTML(String(content));
        },

        /**
         * Validate database connection parameters
         * @param {object} connection - Connection configuration
         * @returns {object} Validation result
         */
        validateConnection: function(connection) {
            const errors = [];
            
            if (!connection) {
                return {
                    isValid: false,
                    errors: ["Connection configuration is required"]
                };
            }

            // Check for required fields
            const requiredFields = ['host', 'database', 'user'];
            requiredFields.forEach(field => {
                if (!connection[field]) {
                    errors.push(`Missing required field: ${field}`);
                }
            });

            // Check for secure connection
            if (connection.host && !connection.ssl && !connection.host.includes('localhost')) {
                errors.push("SSL/TLS encryption should be enabled for external connections");
            }

            // Check for exposed credentials
            if (connection.password && connection.password.length < 8) {
                errors.push("Password should be at least 8 characters long");
            }

            // Validate host format
            if (connection.host) {
                const hostPattern = /^[a-zA-Z0-9.-]+$/;
                if (!hostPattern.test(connection.host)) {
                    errors.push("Invalid host format");
                }
            }

            return {
                isValid: errors.length === 0,
                errors: errors
            };
        },

        /**
         * Create secure WebSocket connection for SQL operations
         * @param {string} url - WebSocket URL
         * @param {object} options - Connection options
         * @returns {WebSocket} Secure WebSocket
         */
        createSecureWebSocket: function(url, options) {
            options = options || {};
            
            // Force secure protocol
            let secureUrl = url;
            if (url.startsWith('ws://')) {
                secureUrl = url.replace('ws://', 'wss://');
                Log.warning("WebSocket URL upgraded to secure protocol", secureUrl);
            }

            const ws = new WebSocket(secureUrl);
            
            // Add security event handlers
            ws.addEventListener('open', function() {
                Log.info("Secure WebSocket connection established for SQL operations", secureUrl);
            });

            ws.addEventListener('error', function(error) {
                Log.error("SQL WebSocket error", error);
            });

            ws.addEventListener('message', function(event) {
                try {
                    // Validate incoming SQL data
                    const data = JSON.parse(event.data);
                    
                    // Additional validation for SQL-related messages
                    if (data.type && data.type.includes('SQL') || data.type.includes('QUERY')) {
                        const validation = this.validateSQL(data.sql || '');
                        if (!validation.isValid) {
                            Log.warning("Invalid SQL received via WebSocket", validation.errors);
                            return;
                        }
                    }
                    
                    if (options.onMessage) {
                        options.onMessage(data);
                    }
                } catch (e) {
                    Log.error("Invalid WebSocket message format", event.data);
                }
            }.bind(this));

            return ws;
        },

        /**
         * Create secure EventSource for SQL monitoring
         * @param {string} url - EventSource URL
         * @param {object} options - Options
         * @returns {EventSource} Secure EventSource
         */
        createSecureEventSource: function(url, options) {
            options = options || {};
            
            // Force HTTPS protocol
            let secureUrl = url;
            if (url.startsWith('http://')) {
                secureUrl = url.replace('http://', 'https://');
                Log.warning("EventSource URL upgraded to secure protocol", secureUrl);
            }

            const eventSource = new EventSource(secureUrl);
            
            // Add validation for incoming events
            const originalAddEventListener = eventSource.addEventListener;
            eventSource.addEventListener = function(type, listener, options) {
                const secureListener = function(event) {
                    try {
                        // Validate event data if it's JSON
                        if (event.data) {
                            const data = JSON.parse(event.data);
                            
                            // SQL-specific validation
                            if (data.sql || data.query) {
                                const validation = this.validateSQL(data.sql || data.query);
                                if (!validation.isValid) {
                                    Log.warning("Invalid SQL received via EventSource", validation.errors);
                                    return;
                                }
                            }
                            
                            const validatedEvent = Object.assign({}, event, { data: data });
                            listener(validatedEvent);
                        } else {
                            listener(event);
                        }
                    } catch (e) {
                        Log.error("Invalid EventSource data format", event.data);
                    }
                }.bind(this);
                
                originalAddEventListener.call(this, type, secureListener, options);
            }.bind(this);

            return eventSource;
        },

        /**
         * Check user authorization for SQL operations
         * @param {string} operation - Operation type
         * @param {string} resource - Resource being accessed
         * @returns {Promise<boolean>} Authorization result
         */
        checkSQLAuth: function(operation, resource) {
            return new Promise(function(resolve) {
                // SQL-specific authorization checks
                const readOnlyOperations = ['select', 'show', 'describe', 'explain'];
                const writeOperations = ['insert', 'update', 'delete'];
                const ddlOperations = ['create', 'alter', 'drop'];
                
                const lowerOp = operation.toLowerCase();
                
                // Basic authorization logic (implement with actual auth service)
                if (readOnlyOperations.includes(lowerOp)) {
                    Log.info("Authorization granted for read operation", operation);
                    resolve(true);
                } else if (writeOperations.includes(lowerOp)) {
                    Log.info("Authorization granted for write operation", operation);
                    resolve(true);
                } else if (ddlOperations.includes(lowerOp)) {
                    Log.warning("DDL operation requires additional authorization", operation);
                    resolve(false); // Requires elevated permissions
                } else {
                    Log.warning("Authorization denied for unknown operation", operation);
                    resolve(false);
                }
            });
        },

        /**
         * Sanitize SQL query result for display
         * @param {any} result - Query result
         * @returns {string} Sanitized result
         */
        sanitizeSQLResult: function(result) {
            if (result === null || result === undefined) {
                return "NULL";
            }

            if (typeof result === 'string') {
                return this.escapeHTML(result);
            }

            if (typeof result === 'object') {
                try {
                    // Recursively sanitize object properties
                    const sanitized = {};
                    Object.entries(result).forEach(([key, value]) => {
                        sanitized[this.escapeHTML(key)] = this.sanitizeSQLResult(value);
                    });
                    return JSON.stringify(sanitized, null, 2);
                } catch (e) {
                    return "Invalid result format";
                }
            }

            if (Array.isArray(result)) {
                return result.map(item => this.sanitizeSQLResult(item));
            }

            return this.escapeHTML(String(result));
        },

        /**
         * Advanced query complexity validation with resource limits
         * @param {string} sql - SQL query
         * @param {object} limits - Resource limits configuration
         * @returns {object} Comprehensive complexity validation result
         */
        validateQueryComplexity: function(sql, limits) {
            limits = limits || {
                maxJoins: 10,
                maxSubqueries: 5,
                maxUnions: 3,
                maxTables: 15,
                maxComplexity: 50,
                maxQueryLength: 10000,
                allowCartesianProducts: false,
                allowRecursiveCTE: false
            };

            if (!sql) {
                return { isValid: false, reason: "No SQL provided", complexity: 0, score: 0 };
            }

            const lowerSQL = sql.toLowerCase();
            let complexity = 0;
            let score = 0;
            const issues = [];
            const metrics = {};

            // Query length check
            if (sql.length > limits.maxQueryLength) {
                issues.push("Query too long (" + sql.length + " chars, limit: " + limits.maxQueryLength + ")");
                complexity += 15;
            }

            // Count and validate joins
            const joinPatterns = [
                { pattern: /\bjoin\b/gi, weight: 2, name: 'basic_join' },
                { pattern: /\bleft\s+join\b/gi, weight: 2.5, name: 'left_join' },
                { pattern: /\bright\s+join\b/gi, weight: 2.5, name: 'right_join' },
                { pattern: /\bfull\s+join\b/gi, weight: 3, name: 'full_join' },
                { pattern: /\bcross\s+join\b/gi, weight: 4, name: 'cross_join' },
                { pattern: /\binner\s+join\b/gi, weight: 2, name: 'inner_join' }
            ];

            let totalJoins = 0;
            joinPatterns.forEach(({pattern, weight, name}) => {
                const matches = (sql.match(pattern) || []).length;
                metrics[name] = matches;
                totalJoins += matches;
                complexity += matches * weight;
            });

            if (totalJoins > limits.maxJoins) {
                issues.push(`Too many joins (${totalJoins}, limit: ${limits.maxJoins})`);
            }
            metrics.total_joins = totalJoins;

            // Count subqueries with depth analysis
            let subqueryDepth = 0;
            let currentDepth = 0;
            let maxDepth = 0;
            
            for (let char of sql) {
                if (char === '(') currentDepth++;
                if (char === ')') currentDepth--;
                maxDepth = Math.max(maxDepth, currentDepth);
            }
            
            const subqueryCount = (sql.match(/\(/g) || []).length;
            complexity += subqueryCount * 1.5 + maxDepth * 2;
            metrics.subquery_count = subqueryCount;
            metrics.max_nesting_depth = maxDepth;
            
            if (subqueryCount > limits.maxSubqueries) {
                issues.push(`Too many subqueries (${subqueryCount}, limit: ${limits.maxSubqueries})`);
            }
            
            if (maxDepth > 4) {
                issues.push(`Query nesting too deep (${maxDepth} levels)`);
                complexity += 10;
            }

            // Count UNION operations
            const unionCount = (sql.match(/\bunion\b/gi) || []).length;
            const unionAllCount = (sql.match(/\bunion\s+all\b/gi) || []).length;
            complexity += unionCount * 3 + unionAllCount * 2;
            metrics.union_count = unionCount;
            metrics.union_all_count = unionAllCount;
            
            if (unionCount > limits.maxUnions) {
                issues.push(`Too many UNION operations (${unionCount}, limit: ${limits.maxUnions})`);
            }

            // Analyze table references
            const tablePattern = /(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)/gi;
            const tables = new Set();
            let match;
            while ((match = tablePattern.exec(sql)) !== null) {
                tables.add(match[1].toLowerCase());
            }
            
            const tableCount = tables.size;
            complexity += tableCount * 1.5;
            metrics.table_count = tableCount;
            
            if (tableCount > limits.maxTables) {
                issues.push(`Too many tables referenced (${tableCount}, limit: ${limits.maxTables})`);
            }

            // Check for cartesian products
            if (!limits.allowCartesianProducts) {
                const hasMultipleTables = tableCount > 1;
                const hasJoin = totalJoins > 0;
                const hasWhere = /\bwhere\b/gi.test(sql);
                
                if (hasMultipleTables && !hasJoin && !hasWhere) {
                    complexity += 20;
                    issues.push("Potential cartesian product detected (multiple tables without JOIN or WHERE)");
                }
            }

            // Check for expensive operations
            const expensiveOperations = [
                { pattern: /\bcount\s*\(\s*\*\s*\)/gi, weight: 3, name: 'count_star' },
                { pattern: /\bdistinct\b/gi, weight: 4, name: 'distinct' },
                { pattern: /\bgroup\s+by\b/gi, weight: 3, name: 'group_by' },
                { pattern: /\border\s+by\b/gi, weight: 2, name: 'order_by' },
                { pattern: /\bhaving\b/gi, weight: 2.5, name: 'having' },
                { pattern: /\bwindow\b|\bover\s*\(/gi, weight: 4, name: 'window_function' },
                { pattern: /\brecursive\b/gi, weight: 6, name: 'recursive_cte' },
                { pattern: /\bexists\b/gi, weight: 3, name: 'exists' },
                { pattern: /\bnot\s+exists\b/gi, weight: 3.5, name: 'not_exists' },
                { pattern: /\blike\s+['"]%.*%['"]/gi, weight: 4, name: 'full_wildcard_like' }
            ];

            expensiveOperations.forEach(({pattern, weight, name}) => {
                const matches = (sql.match(pattern) || []).length;
                if (matches > 0) {
                    metrics[name] = matches;
                    complexity += matches * weight;
                }
            });

            // Check for recursive CTEs
            if (!limits.allowRecursiveCTE && /\bwith\s+recursive\b/gi.test(sql)) {
                issues.push("Recursive CTEs are not allowed");
                complexity += 10;
            }

            // Calculate performance score (inverse of complexity)
            score = Math.max(0, 100 - complexity);

            // Additional pattern analysis
            const riskPatterns = [
                { pattern: /\bselect\s+\*\s+from\b/gi, penalty: 5, message: "SELECT * can impact performance" },
                { pattern: /\blike\s+['"]%/gi, penalty: 3, message: "Leading wildcards prevent index usage" },
                { pattern: /\bor\b.*\bor\b/gi, penalty: 4, message: "Multiple OR conditions can be expensive" },
                { pattern: /\bin\s*\(\s*select\b/gi, penalty: 3, message: "IN subquery can be optimized with EXISTS" }
            ];

            const recommendations = [];
            riskPatterns.forEach(({pattern, penalty, message}) => {
                if (pattern.test(sql)) {
                    recommendations.push(message);
                    complexity += penalty;
                }
            });

            // Final validation
            const isValid = complexity <= limits.maxComplexity && issues.length === 0;
            
            // Risk classification
            let riskLevel;
            if (complexity <= 10) riskLevel = "LOW";
            else if (complexity <= 25) riskLevel = "MEDIUM";
            else if (complexity <= 40) riskLevel = "HIGH";
            else riskLevel = "CRITICAL";

            return {
                isValid: isValid,
                complexity: complexity,
                score: score,
                issues: issues,
                metrics: metrics,
                recommendations: recommendations,
                riskLevel: riskLevel,
                limits: limits,
                estimatedExecutionTime: this._estimateExecutionTime(complexity, metrics),
                resourceRequirements: this._estimateResourceRequirements(complexity, metrics)
            };
        },

        /**
         * Estimate query execution time based on complexity
         * @private
         */
        _estimateExecutionTime: function(complexity, metrics) {
            let baseTime = 100; // milliseconds
            
            // Add time for joins
            baseTime += (metrics.total_joins || 0) * 200;
            
            // Add time for subqueries
            baseTime += (metrics.subquery_count || 0) * 150;
            
            // Add time for sorting operations
            if (metrics.order_by) baseTime += 300;
            if (metrics.group_by) baseTime += 400;
            if (metrics.distinct) baseTime += 250;
            
            // Add time for window functions
            if (metrics.window_function) baseTime += 500;
            
            // Add complexity multiplier
            baseTime *= (1 + complexity / 50);
            
            return Math.round(baseTime);
        },

        /**
         * Estimate resource requirements
         * @private
         */
        _estimateResourceRequirements: function(complexity, metrics) {
            return {
                memory: Math.max(10, complexity * 2) + "MB",
                cpu: Math.max(5, complexity) + "%",
                io: Math.max(1, (metrics.table_count || 0) * 2) + " operations",
                network: "LOW"
            };
        }
    };
});