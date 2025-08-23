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
         * Comprehensive SQL injection prevention
         * @param {string} sql - SQL query to validate
         * @param {object} params - Query parameters
         * @returns {object} Validation result
         */
        validateSQL: function(sql, params) {
            if (!sql || typeof sql !== 'string') {
                return {
                    isValid: false,
                    sanitized: "",
                    errors: ["SQL query must be a non-empty string"]
                };
            }

            const errors = [];
            let sanitized = sql.trim();

            // Remove dangerous patterns
            const dangerousPatterns = [
                // Script injection
                /<script[^>]*>.*?<\/script>/gi,
                /javascript:/gi,
                /on\w+\s*=/gi,
                
                // Code execution
                /eval\s*\(/gi,
                /new\s+Function/gi,
                /setTimeout\s*\(/gi,
                /setInterval\s*\(/gi,
                
                // SQL injection patterns
                /union\s+select/gi,
                /;\s*drop\s+table/gi,
                /;\s*delete\s+from/gi,
                /;\s*update\s+.*\s+set/gi,
                /;\s*insert\s+into/gi,
                /exec\s*\(/gi,
                /execute\s*\(/gi,
                /xp_cmdshell/gi,
                /sp_executesql/gi,
                
                // String manipulation that could bypass filters
                /char\s*\(/gi,
                /ascii\s*\(/gi,
                /0x[0-9a-f]+/gi
            ];

            dangerousPatterns.forEach(pattern => {
                if (pattern.test(sanitized)) {
                    errors.push("SQL contains potentially dangerous code");
                    sanitized = sanitized.replace(pattern, '');
                }
            });

            // Check for SQL injection indicators
            const injectionIndicators = [
                // Always true conditions
                /1\s*=\s*1/gi,
                /'.*'\s*=\s*'.*'/gi,
                
                // Comment-based injection
                /--/gi,
                /\/\*/gi,
                /\*\//gi,
                
                // Concatenation attempts
                /\|\|/gi,
                /concat\s*\(/gi,
                
                // Information gathering
                /information_schema/gi,
                /sys\./gi,
                /mysql\./gi,
                
                // Blind SQL injection
                /sleep\s*\(/gi,
                /waitfor\s+delay/gi,
                /benchmark\s*\(/gi
            ];

            injectionIndicators.forEach(pattern => {
                if (pattern.test(sanitized)) {
                    errors.push("Potential SQL injection pattern detected");
                }
            });

            // Validate parameters if provided
            if (params) {
                Object.entries(params).forEach(([key, value]) => {
                    if (typeof value === 'string') {
                        // Check for SQL injection in parameters
                        injectionIndicators.forEach(pattern => {
                            if (pattern.test(value)) {
                                errors.push(`Potential SQL injection in parameter '${key}'`);
                            }
                        });
                    }
                });
            }

            // Check for proper parameterization
            if (sanitized.includes("'") && !sanitized.includes("?") && !sanitized.includes("$")) {
                errors.push("SQL contains string literals - use parameterized queries");
            }

            return {
                isValid: errors.length === 0,
                sanitized: sanitized,
                errors: errors,
                original: sql
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
         * Create secure parameterized query
         * @param {string} baseQuery - Base SQL query with placeholders
         * @param {object} parameters - Parameters to bind
         * @returns {object} Secure query object
         */
        createParameterizedQuery: function(baseQuery, parameters) {
            const validation = this.validateSQL(baseQuery);
            
            if (!validation.isValid) {
                return {
                    isValid: false,
                    errors: validation.errors,
                    query: null,
                    parameters: null
                };
            }

            // Sanitize all parameters
            const sanitizedParams = {};
            if (parameters) {
                Object.entries(parameters).forEach(([key, value]) => {
                    sanitizedParams[key] = this.sanitizeSQLParameter(value);
                });
            }

            return {
                isValid: true,
                query: validation.sanitized,
                parameters: sanitizedParams,
                errors: []
            };
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
         * Validate query complexity to prevent resource exhaustion
         * @param {string} sql - SQL query
         * @returns {object} Complexity validation result
         */
        validateQueryComplexity: function(sql) {
            if (!sql) {
                return { isValid: false, reason: "No SQL provided" };
            }

            const lowerSQL = sql.toLowerCase();
            let complexity = 0;
            const issues = [];

            // Count joins
            const joinCount = (sql.match(/join/gi) || []).length;
            complexity += joinCount * 2;
            if (joinCount > 5) {
                issues.push(`Too many joins (${joinCount})`);
            }

            // Count subqueries
            const subqueryCount = (sql.match(/\(/g) || []).length;
            complexity += subqueryCount;
            if (subqueryCount > 3) {
                issues.push(`Too many subqueries (${subqueryCount})`);
            }

            // Check for cartesian products
            if (lowerSQL.includes('from') && lowerSQL.includes(',') && !lowerSQL.includes('where')) {
                complexity += 10;
                issues.push("Potential cartesian product detected");
            }

            // Check for functions that could be expensive
            const expensiveFunctions = ['count(*)', 'distinct', 'group by'];
            expensiveFunctions.forEach(func => {
                if (lowerSQL.includes(func)) {
                    complexity += 2;
                }
            });

            return {
                isValid: complexity < 20 && issues.length === 0,
                complexity: complexity,
                issues: issues
            };
        }
    };
});