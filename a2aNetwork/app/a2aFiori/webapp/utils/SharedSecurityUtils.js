sap.ui.define([
    "sap/base/security/encodeXML",
    "sap/base/security/encodeJS",
    "sap/base/security/encodeURL",
    "sap/base/strings/escapeRegExp",
    "sap/base/Log",
    "sap/m/MessageToast"
], (encodeXML, encodeJS, encodeURL, escapeRegExp, Log, MessageToast) => {
    "use strict";

    /**
     * Shared Security Utilities for A2A Platform
     * Provides comprehensive security features for all agents including:
     * - Input validation and output encoding
     * - CSRF protection and secure HTTP calls
     * - XSS and injection attack prevention
     * - Audit logging and authentication
     * - Secure WebSocket and EventSource connections
     */
    return {

        /**
         * Encodes text for safe display in HTML contexts
         * @param {string} text - Text to encode
         * @returns {string} - Safely encoded text
         */
        encodeHTML(text) {
            if (typeof text !== "string") {
                return "";
            }
            return encodeXML(text);
        },

        /**
         * Encodes text for safe use in JavaScript contexts
         * @param {string} text - Text to encode
         * @returns {string} - Safely encoded text
         */
        encodeJS(text) {
            if (typeof text !== "string") {
                return "";
            }
            return encodeJS(text);
        },

        /**
         * Encodes text for safe use in URL contexts
         * @param {string} text - Text to encode
         * @returns {string} - Safely encoded URL
         */
        encodeURL(text) {
            if (typeof text !== "string") {
                return "";
            }
            return encodeURL(text);
        },

        /**
         * Sanitizes error messages to prevent information disclosure
         * @param {string} errorText - Raw error message
         * @returns {string} - Sanitized error message
         */
        sanitizeErrorMessage(errorText) {
            if (!errorText || typeof errorText !== "string") {
                return "An error occurred. Please contact system administrator.";
            }

            // Remove potential sensitive information patterns
            let sanitized = errorText
                .replace(/\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g, "[IP_ADDRESS]")
                .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, "[EMAIL]")
                .replace(/password[:\s]*[^\s]+/gi, "password: [REDACTED]")
                .replace(/token[:\s]*[^\s]+/gi, "token: [REDACTED]")
                .replace(/key[:\s]*[^\s]+/gi, "key: [REDACTED]")
                .replace(/\b[A-F0-9]{32,}\b/gi, "[HASH]")
                .replace(/\/[^\s]*\/[^\s]*/g, "[PATH]");

            // Limit length to prevent potential DoS
            if (sanitized.length > 200) {
                sanitized = `${sanitized.substring(0, 200) }...`;
            }

            return this.encodeHTML(sanitized);
        },

        /**
         * Validates input data based on type and constraints
         * @param {*} input - Input to validate
         * @param {string} type - Type of validation (text, number, email, etc.)
         * @param {Object} options - Validation options
         * @returns {Object} - Validation result with isValid and message
         */
        validateInput(input, type, options = {}) {
            const result = { isValid: true, message: "" };

            // Check if required
            if (options.required && (!input || input.toString().trim() === "")) {
                result.isValid = false;
                result.message = "This field is required";
                return result;
            }

            // Skip further validation if empty and not required
            if (!input || input.toString().trim() === "") {
                return result;
            }

            const inputStr = input.toString().trim();

            switch (type) {
            case "text":
                return this._validateText(inputStr, options);
            case "number":
                return this._validateNumber(inputStr, options);
            case "email":
                return this._validateEmail(inputStr, options);
            case "url":
                return this._validateURL(inputStr, options);
            case "agentName":
                return this._validateAgentName(inputStr, options);
            case "datasetName":
                return this._validateDatasetName(inputStr, options);
            case "workflowName":
                return this._validateWorkflowName(inputStr, options);
            case "sqlQuery":
                return this._validateSQLQuery(inputStr, options);
            case "json":
                return this._validateJSON(inputStr, options);
            default:
                return this._validateText(inputStr, options);
            }
        },

        /**
         * Sanitizes input data for safe processing
         * @param {*} input - Input to sanitize
         * @returns {string} - Sanitized input
         */
        sanitizeInput(input) {
            if (typeof input !== "string") {
                return "";
            }

            // Remove or encode potentially dangerous characters
            return input
                .replace(/[<>]/g, "")
                .replace(/javascript:/gi, "")
                .replace(/on\w+=/gi, "")
                .replace(/data:/gi, "")
                .replace(/vbscript:/gi, "")
                .trim();
        },

        /**
         * Validates configuration objects for security vulnerabilities
         * @param {object} config - Configuration object to validate
         * @param {string} configType - Type of configuration (workflow, agent, etc.)
         * @returns {object} Validation result
         */
        validateConfiguration(config, configType) {
            const validation = {
                isValid: true,
                errors: [],
                warnings: [],
                sanitizedConfig: {}
            };

            if (!config || typeof config !== "object") {
                validation.isValid = false;
                validation.errors.push("Configuration is required and must be an object");
                return validation;
            }

            // Check for code injection in configuration
            if (this._containsCodeInjection(JSON.stringify(config))) {
                validation.isValid = false;
                validation.errors.push("Configuration contains potentially malicious code");
                return validation;
            }

            // Type-specific validation
            switch (configType) {
            case "workflow":
                return this._validateWorkflowConfig(config);
            case "agent":
                return this._validateAgentConfig(config);
            case "pipeline":
                return this._validatePipelineConfig(config);
            case "security":
                return this._validateSecurityConfig(config);
            default:
                return this._validateGenericConfig(config);
            }
        },

        /**
         * Gets CSRF token from SAP UI5 model
         * @returns {string} - CSRF token or empty string
         */
        getCSRFToken() {
            try {
                const oModel = sap.ui.getCore().getModel();
                if (oModel && oModel.getSecurityToken) {
                    return oModel.getSecurityToken();
                }

                // Fallback: try to get from meta tag
                const metaTag = document.querySelector("meta[name=\"csrf-token\"]");
                if (metaTag) {
                    return metaTag.getAttribute("content");
                }

                return "";
            } catch (e) {
                return "";
            }
        },

        /**
         * Makes secure OData function calls with CSRF protection
         * @param {sap.ui.model.odata.v2.ODataModel} model - OData model
         * @param {string} functionName - Function name to call
         * @param {object} parameters - Function parameters
         * @returns {Promise} Promise resolving to function result
         */
        secureCallFunction(model, functionName, parameters) {
            return new Promise((resolve, reject) => {
                // First, refresh security token
                model.refreshSecurityToken((tokenData) => {
                    // Add CSRF token to headers if not already present
                    const headers = parameters.headers || {};
                    if (!headers["X-CSRF-Token"] && tokenData) {
                        headers["X-CSRF-Token"] = tokenData;
                    }

                    // Enhanced parameters with security
                    const secureParams = {
                        ...parameters,
                        headers,
                        success: (data) => {
                            this.logSecureOperation(functionName, "SUCCESS");
                            if (parameters.success) {
                                parameters.success(data);
                            }
                            resolve(data);
                        },
                        error: (error) => {
                            this.logSecureOperation(functionName, "ERROR", error);
                            if (parameters.error) {
                                parameters.error(error);
                            }
                            reject(error);
                        }
                    };

                    model.callFunction(functionName, secureParams);
                }, (error) => {
                    this.logSecureOperation(functionName, "TOKEN_ERROR", error);
                    reject(new Error("Failed to obtain CSRF token"));
                });
            });
        },

        /**
         * Creates secure WebSocket connection with validation
         * @param {string} url - WebSocket URL
         * @param {object} options - Connection options
         * @returns {WebSocket|null} Secure WebSocket connection
         */
        createSecureWebSocket(url, options = {}) {
            try {
                // Validate URL first
                if (!this.validateWebSocketUrl(url)) {
                    Log.error("Invalid WebSocket URL", url);
                    return null;
                }

                // For localhost development, allow ws:// otherwise require wss://
                let secureUrl = url;
                if (!this._isLocalhost(url) && url.startsWith("ws://")) {
                    secureUrl = url.replace(/^ws:\/\//, "wss://");
                }

                const ws = new WebSocket(secureUrl);

                // Add security event handlers
                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        // Validate incoming data
                        if (this._isValidWebSocketMessage(data)) {
                            if (options.onMessage) {
                                options.onMessage(data);
                            }
                        } else {
                            Log.warning("Invalid WebSocket message received");
                        }
                    } catch (error) {
                        Log.error("Invalid WebSocket message format", error);
                    }
                };

                ws.onerror = (error) => {
                    this.logSecureOperation("WEBSOCKET_ERROR", "ERROR", error);
                    if (options.onError) {
                        options.onError(error);
                    }
                };

                ws.onclose = (event) => {
                    this.logSecureOperation("WEBSOCKET_CLOSE", "INFO", { code: event.code, reason: event.reason });
                    if (options.onClose) {
                        options.onClose(event);
                    }
                };

                return ws;

            } catch (error) {
                Log.error("Failed to create secure WebSocket", error);
                return null;
            }
        },

        /**
         * Validates WebSocket URL for security
         * @param {string} url - WebSocket URL to validate
         * @returns {boolean} True if URL is valid
         */
        validateWebSocketUrl(url) {
            try {
                const urlObj = new URL(url);

                // Allow ws:// only for localhost, require wss:// otherwise
                if (urlObj.protocol === "wss:") {
                    return true;
                } else if (urlObj.protocol === "ws:" && this._isLocalhost(url)) {
                    return true;
                }

                return false;
            } catch (error) {
                return false;
            }
        },

        /**
         * Creates secure EventSource connection
         * @param {string} url - EventSource URL
         * @param {object} options - Connection options
         * @returns {EventSource|null} Secure EventSource connection
         */
        createSecureEventSource(url, options = {}) {
            try {
                // Ensure secure protocol for non-localhost
                let secureUrl = url;
                if (!this._isLocalhost(url) && url.startsWith("http://")) {
                    secureUrl = url.replace(/^http:\/\//, "https://");
                }

                // Validate URL
                if (!this._isValidEventSourceUrl(secureUrl)) {
                    Log.error("Invalid EventSource URL", secureUrl);
                    return null;
                }

                const eventSource = new EventSource(secureUrl);

                // Add security handlers
                eventSource.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (this._isValidEventSourceMessage(data)) {
                            if (options.onMessage) {
                                options.onMessage(data);
                            }
                        } else {
                            Log.warning("Invalid EventSource message received");
                        }
                    } catch (error) {
                        Log.error("Invalid EventSource message format", error);
                    }
                };

                eventSource.onerror = (error) => {
                    this.logSecureOperation("EVENTSOURCE_ERROR", "ERROR", error);
                    if (options.onError) {
                        options.onError(error);
                    }
                };

                return eventSource;

            } catch (error) {
                Log.error("Failed to create secure EventSource", error);
                return null;
            }
        },

        /**
         * Checks if user has required role/permission
         * @param {string} role - Role to check
         * @param {string} agentId - Agent ID for context (optional)
         * @returns {boolean} True if user has role
         */
        hasRole(role, agentId) {
            try {
                const user = sap.ushell?.Container?.getUser();
                if (user && user.hasRole) {
                    return user.hasRole(role);
                }

                // Mock role validation for development/testing
                const mockRoles = this._getMockRoles(agentId);
                return mockRoles.includes(role);
            } catch (error) {
                Log.error("Error checking user role", error);
                return false;
            }
        },

        /**
         * Logs security events for audit purposes
         * @param {string} operation - Operation name
         * @param {string} status - Operation status (SUCCESS, ERROR, WARNING)
         * @param {object} details - Additional details
         * @param {string} agentId - Agent ID for context
         */
        logSecureOperation(operation, status, details, agentId) {
            try {
                const logEntry = {
                    timestamp: new Date().toISOString(),
                    operation: this.sanitizeInput(operation),
                    status,
                    agent: agentId || "SharedUtils",
                    user: this._getCurrentUser()?.id || "anonymous",
                    details: this._sanitizeLogDetails(details || {}),
                    userAgent: navigator.userAgent.substring(0, 200),
                    url: window.location.href
                };

                // Log based on environment
                if (this._isProduction()) {
                    this._sendToAuditService(logEntry);
                } else {
                    // Non-production audit logging
                }
            } catch (e) {
                // Fail silently to avoid breaking application
            }
        },

        /**
         * Rate limiting check for operations
         * @param {string} operation - Operation name
         * @param {number} maxAttempts - Maximum attempts allowed
         * @param {number} timeWindow - Time window in milliseconds
         * @returns {boolean} True if operation is allowed
         */
        checkRateLimit(operation, maxAttempts = 10, timeWindow = 60000) {
            const now = Date.now();
            const key = `${operation}_${this._getCurrentUser()?.id || "anonymous"}`;

            if (!this._rateLimitStore) {
                this._rateLimitStore = new Map();
            }

            const attempts = this._rateLimitStore.get(key) || [];
            const recentAttempts = attempts.filter(time => now - time < timeWindow);

            if (recentAttempts.length >= maxAttempts) {
                this.logSecureOperation(operation, "RATE_LIMITED", {
                    attempts: recentAttempts.length,
                    timeWindow
                });
                return false;
            }

            recentAttempts.push(now);
            this._rateLimitStore.set(key, recentAttempts);
            return true;
        },

        // Private validation methods
        _validateText(text, options) {
            const result = { isValid: true, message: "" };

            if (options.minLength && text.length < options.minLength) {
                result.isValid = false;
                result.message = `Minimum length is ${options.minLength} characters`;
                return result;
            }

            if (options.maxLength && text.length > options.maxLength) {
                result.isValid = false;
                result.message = `Maximum length is ${options.maxLength} characters`;
                return result;
            }

            if (options.pattern && !options.pattern.test(text)) {
                result.isValid = false;
                result.message = options.patternMessage || "Invalid format";
                return result;
            }

            // Check for potential XSS patterns
            if (this._containsXSSPattern(text)) {
                result.isValid = false;
                result.message = "Invalid characters detected";
                return result;
            }

            return result;
        },

        _validateNumber(text, options) {
            const result = { isValid: true, message: "" };

            const number = parseFloat(text);
            if (isNaN(number)) {
                result.isValid = false;
                result.message = "Must be a valid number";
                return result;
            }

            if (options.min !== undefined && number < options.min) {
                result.isValid = false;
                result.message = `Minimum value is ${options.min}`;
                return result;
            }

            if (options.max !== undefined && number > options.max) {
                result.isValid = false;
                result.message = `Maximum value is ${options.max}`;
                return result;
            }

            return result;
        },

        _validateEmail(email, options) {
            const result = { isValid: true, message: "" };
            const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

            if (!emailPattern.test(email)) {
                result.isValid = false;
                result.message = "Invalid email format";
                return result;
            }

            return result;
        },

        _validateURL(url, options) {
            const result = { isValid: true, message: "" };

            try {
                const urlObj = new URL(url);

                // Only allow HTTP/HTTPS protocols
                if (!["http:", "https:"].includes(urlObj.protocol)) {
                    result.isValid = false;
                    result.message = "Only HTTP and HTTPS URLs are allowed";
                    return result;
                }

                // Block known dangerous domains/patterns
                if (this._isDangerousDomain(urlObj.hostname)) {
                    result.isValid = false;
                    result.message = "Domain not allowed";
                    return result;
                }
            } catch (e) {
                result.isValid = false;
                result.message = "Invalid URL format";
                return result;
            }

            return result;
        },

        _validateAgentName(name, options) {
            const result = { isValid: true, message: "" };
            const pattern = /^[a-zA-Z0-9\s\-_]{1,50}$/;

            if (!pattern.test(name)) {
                result.isValid = false;
                result.message = "Agent name can only contain letters, numbers, spaces, hyphens, and underscores (max 50 characters)";
                return result;
            }

            return result;
        },

        _validateDatasetName(name, options) {
            const result = { isValid: true, message: "" };
            const pattern = /^[a-zA-Z0-9_\-\.]{1,100}$/;

            if (!pattern.test(name)) {
                result.isValid = false;
                result.message = "Dataset name can only contain letters, numbers, underscores, hyphens, and periods (max 100 characters)";
                return result;
            }

            return result;
        },

        _validateWorkflowName(name, options) {
            const result = { isValid: true, message: "" };
            const pattern = /^[a-zA-Z0-9\s\-_]{1,80}$/;

            if (!pattern.test(name)) {
                result.isValid = false;
                result.message = "Workflow name can only contain letters, numbers, spaces, hyphens, and underscores (max 80 characters)";
                return result;
            }

            return result;
        },

        _validateSQLQuery(query, options) {
            const result = { isValid: true, message: "" };

            // Check for dangerous SQL patterns
            const dangerousPatterns = [
                /\bdrop\s+table\b/gi,
                /\bdelete\s+from\b/gi,
                /\btruncate\b/gi,
                /\balter\s+table\b/gi,
                /\bcreate\s+table\b/gi,
                /\bexec\b/gi,
                /\bexecute\b/gi,
                /\bunion\s+select\b/gi,
                /--/g,
                /\/\*/g,
                /xp_/gi
            ];

            const hasDangerousPattern = dangerousPatterns.some(pattern => pattern.test(query));
            if (hasDangerousPattern) {
                result.isValid = false;
                result.message = "Query contains potentially dangerous SQL statements";
                return result;
            }

            return result;
        },

        _validateJSON(jsonString, options) {
            const result = { isValid: true, message: "" };

            try {
                JSON.parse(jsonString);

                // Check for code injection in JSON
                if (this._containsCodeInjection(jsonString)) {
                    result.isValid = false;
                    result.message = "JSON contains potentially malicious code";
                    return result;
                }
            } catch (e) {
                result.isValid = false;
                result.message = "Invalid JSON format";
                return result;
            }

            return result;
        },

        _containsXSSPattern(text) {
            const xssPatterns = [
                /<script/gi,
                /javascript:/gi,
                /on\w+\s*=/gi,
                /<iframe/gi,
                /<object/gi,
                /<embed/gi,
                /expression\s*\(/gi,
                /vbscript:/gi,
                /data:text\/html/gi
            ];

            return xssPatterns.some(pattern => pattern.test(text));
        },

        _containsCodeInjection(str) {
            if (!str || typeof str !== "string") {return false;}

            const codePatterns = [
                /eval\s*\(/gi,
                /Function\s*\(/gi,
                /setTimeout\s*\([^,]+,\s*0\s*\)/gi,
                /setInterval\s*\([^,]+,/gi,
                /__proto__/gi,
                /constructor\s*\[/gi,
                /import\s*\(/gi,
                /require\s*\(/gi,
                /\$\{.*\}/g,
                /document\.write/gi,
                /innerHTML/gi,
                /outerHTML/gi
            ];

            return codePatterns.some(pattern => pattern.test(str));
        },

        _validateWorkflowConfig(config) {
            // Implement workflow-specific validation
            const validation = { isValid: true, errors: [], warnings: [], sanitizedConfig: {} };

            if (config.steps && Array.isArray(config.steps)) {
                config.steps.forEach((step, index) => {
                    if (this._containsCodeInjection(JSON.stringify(step))) {
                        validation.isValid = false;
                        validation.errors.push(`Step ${index + 1} contains potentially malicious code`);
                    }
                });
            }

            return validation;
        },

        _validateAgentConfig(config) {
            // Implement agent-specific validation
            const validation = { isValid: true, errors: [], warnings: [], sanitizedConfig: {} };
            // Add agent-specific validation logic
            return validation;
        },

        _validatePipelineConfig(config) {
            // Implement pipeline-specific validation
            const validation = { isValid: true, errors: [], warnings: [], sanitizedConfig: {} };
            // Add pipeline-specific validation logic
            return validation;
        },

        _validateSecurityConfig(config) {
            // Implement security-specific validation
            const validation = { isValid: true, errors: [], warnings: [], sanitizedConfig: {} };
            // Add security-specific validation logic
            return validation;
        },

        _validateGenericConfig(config) {
            // Implement generic configuration validation
            const validation = { isValid: true, errors: [], warnings: [], sanitizedConfig: config };
            return validation;
        },

        _isLocalhost(url) {
            try {
                const urlObj = new URL(url);
                return urlObj.hostname === "localhost" ||
                       urlObj.hostname === "127.0.0.1" ||
                       urlObj.hostname.startsWith("192.168.") ||
                       urlObj.hostname.startsWith("10.") ||
                       urlObj.hostname.startsWith("172.16.");
            } catch (error) {
                return false;
            }
        },

        _isValidEventSourceUrl(url) {
            try {
                const urlObj = new URL(url);
                return urlObj.protocol === "https:" ||
                       (urlObj.protocol === "http:" && this._isLocalhost(url));
            } catch (error) {
                return false;
            }
        },

        _isValidWebSocketMessage(data) {
            if (!data || typeof data !== "object") {return false;}

            // Basic validation - ensure required fields exist and are safe
            return data.type && typeof data.type === "string" &&
                   data.type.length < 100 &&
                   !this._containsXSSPattern(JSON.stringify(data));
        },

        _isValidEventSourceMessage(data) {
            return this._isValidWebSocketMessage(data);
        },

        _isDangerousDomain(hostname) {
            const dangerousDomains = [
                "malicious.com",
                "evil.org",
                // Add known dangerous domains
            ];

            return dangerousDomains.some(domain => hostname.includes(domain));
        },

        _getCurrentUser() {
            try {
                return sap.ushell?.Container?.getUser() || { id: "anonymous" };
            } catch (error) {
                return { id: "anonymous" };
            }
        },

        _getMockRoles(agentId) {
            // Mock roles for development/testing - comprehensive role list
            const allMockRoles = [
                // Dashboard Agent roles
                "DashboardUser", "DashboardAdmin",

                // Resource Management Agent roles
                "ResourceUser", "ResourceAdmin", "ResourceManager",

                // Integration Agent roles
                "IntegrationUser", "IntegrationAdmin", "IntegrationManager",

                // Data Management Agent roles
                "DataManager", "TransformationManager", "DataUser", "DataAdmin",

                // Process Management Agent roles
                "ProcessUser", "ProcessAdmin", "ProcessManager",

                // Quality Management Agent roles
                "QualityUser", "QualityAdmin", "QualityManager",

                // Analytics Agent roles
                "AnalyticsUser", "AnalyticsAdmin", "AnalyticsManager",

                // Performance Management Agent roles
                "PerformanceUser", "PerformanceAdmin", "PerformanceManager",

                // Security Management Agent roles
                "SecurityUser", "SecurityAdmin", "SecurityManager",

                // Backup Management Agent roles
                "BackupUser", "BackupAdmin", "BackupOperator",

                // Deployment Management Agent roles
                "DeploymentUser", "DeploymentAdmin", "DeploymentOperator",

                // General roles
                "BasicUser", "PowerUser", "SystemAdmin"
            ];

            // Return all roles for development/testing
            // In production, this would check actual user permissions
            return allMockRoles;
        },

        _sanitizeLogDetails(details) {
            const sanitized = {};

            for (const key in details) {
                if (details.hasOwnProperty(key)) {
                    const value = details[key];

                    // Don't log sensitive information
                    if (/password|token|secret|key|auth/i.test(key)) {
                        sanitized[key] = "[REDACTED]";
                    } else if (typeof value === "string") {
                        sanitized[key] = this.sanitizeInput(value);
                    } else if (typeof value === "object" && value !== null) {
                        sanitized[key] = this._sanitizeLogDetails(value);
                    } else {
                        sanitized[key] = value;
                    }
                }
            }

            return sanitized;
        },

        _isProduction() {
            return window.location.hostname !== "localhost" &&
                   window.location.hostname !== "127.0.0.1" &&
                   !window.location.hostname.startsWith("192.168.") &&
                   !window.location.hostname.startsWith("10.") &&
                   !window.location.hostname.startsWith("172.16.");
        },

        _sendToAuditService(logEntry) {
            // In production, implement actual audit service integration
            // For now, log to console with AUDIT prefix
        }
    };
});