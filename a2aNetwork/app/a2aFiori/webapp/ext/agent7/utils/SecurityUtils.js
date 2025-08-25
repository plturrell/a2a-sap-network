sap.ui.define([
    "sap/base/security/encodeXML",
    "sap/base/strings/escapeRegExp",
    "sap/base/security/sanitizeHTML",
    "sap/base/Log"
], (encodeXML, escapeRegExp, sanitizeHTML, Log) => {
    "use strict";

    /**
     * Agent 7 (Agent Manager) Security Utilities
     * Provides comprehensive security functions for agent management operations,
     * authentication, authorization, input validation, and secure communications
     */
    return {

        /**
         * Validate agent management operations for security
         * @param {string} sOperation - The operation to validate
         * @returns {Object} Validation result with isValid and errors
         */
        validateAgentOperation(sOperation) {
            const aErrors = [];
            const aDangerousPatterns = [
                { pattern: /\beval\s*\(/gi, message: "eval() function not allowed in agent operations" },
                { pattern: /\bnew\s+Function\s*\(/gi, message: "Dynamic function creation not allowed" },
                { pattern: /\bsetTimeout\s*\([^)]*['\"].*<script/gi, message: "Script injection in setTimeout not allowed" },
                { pattern: /\bsetInterval\s*\([^)]*['\"].*<script/gi, message: "Script injection in setInterval not allowed" },
                { pattern: /\bdocument\.write\s*\(/gi, message: "document.write not allowed" },
                { pattern: /\binnerHTML\s*=/gi, message: "Direct innerHTML assignment not allowed - use textContent" },
                { pattern: /\bouterHTML\s*=/gi, message: "outerHTML assignment not allowed" },
                { pattern: /\bexec\s*\(/gi, message: "exec() operations not allowed in agent management" },
                { pattern: /\bsubprocess/gi, message: "subprocess operations not allowed" },
                { pattern: /\bimport\s+os/gi, message: "OS imports not allowed" },
                { pattern: /\bsystem\s*\(/gi, message: "System calls not allowed" },
                { pattern: /\b__import__/gi, message: "Dynamic imports not allowed" },
                { pattern: /\bgetattr\s*\(/gi, message: "getattr operations not allowed" },
                { pattern: /\bsetattr\s*\(/gi, message: "setattr operations not allowed" },
                { pattern: /\bdelattr\s*\(/gi, message: "delattr operations not allowed" }
            ];

            if (!sOperation || typeof sOperation !== "string") {
                aErrors.push("Invalid operation format");
                return { isValid: false, errors: aErrors };
            }

            const sCleanOperation = sOperation.trim();
            if (sCleanOperation.length > 500) {
                aErrors.push("Operation description too long (max 500 characters)");
            }

            // Check for dangerous patterns
            aDangerousPatterns.forEach((oDangerous) => {
                if (oDangerous.pattern.test(sCleanOperation)) {
                    aErrors.push(oDangerous.message);
                }
            });

            return {
                isValid: aErrors.length === 0,
                errors: aErrors,
                sanitized: this.sanitizeInput(sCleanOperation)
            };
        },

        /**
         * Validate agent configuration data
         * @param {Object} oConfig - Agent configuration object
         * @returns {Object} Validation result
         */
        validateAgentConfig(oConfig) {
            const aErrors = [];

            if (!oConfig || typeof oConfig !== "object") {
                aErrors.push("Invalid configuration object");
                return { isValid: false, errors: aErrors };
            }

            // Validate agent name
            if (oConfig.agentName) {
                if (typeof oConfig.agentName !== "string" || oConfig.agentName.length > 100) {
                    aErrors.push("Agent name must be a string with max 100 characters");
                }
                if (!/^[a-zA-Z0-9_\-\s]+$/.test(oConfig.agentName)) {
                    aErrors.push("Agent name contains invalid characters");
                }
            }

            // Validate agent type
            const aValidTypes = ["DATA_PROCESSING", "ANALYSIS", "COORDINATION", "MONITORING", "VALIDATION"];
            if (oConfig.agentType && !aValidTypes.includes(oConfig.agentType)) {
                aErrors.push("Invalid agent type");
            }

            // Validate priority
            if (oConfig.priority !== undefined) {
                const nPriority = parseInt(oConfig.priority, 10);
                if (isNaN(nPriority) || nPriority < 1 || nPriority > 10) {
                    aErrors.push("Priority must be between 1 and 10");
                }
            }

            // Validate memory limits
            if (oConfig.memoryLimit !== undefined) {
                const nMemory = parseInt(oConfig.memoryLimit, 10);
                if (isNaN(nMemory) || nMemory < 128 || nMemory > 8192) {
                    aErrors.push("Memory limit must be between 128MB and 8192MB");
                }
            }

            // Validate timeout values
            if (oConfig.timeout !== undefined) {
                const nTimeout = parseInt(oConfig.timeout, 10);
                if (isNaN(nTimeout) || nTimeout < 1 || nTimeout > 3600) {
                    aErrors.push("Timeout must be between 1 and 3600 seconds");
                }
            }

            return {
                isValid: aErrors.length === 0,
                errors: aErrors,
                sanitized: this.sanitizeAgentConfig(oConfig)
            };
        },

        /**
         * Sanitize agent configuration data
         * @param {Object} oConfig - Configuration to sanitize
         * @returns {Object} Sanitized configuration
         */
        sanitizeAgentConfig(oConfig) {
            if (!oConfig) {return {};}

            const oSanitized = {};

            if (oConfig.agentName) {
                oSanitized.agentName = this.sanitizeInput(oConfig.agentName);
            }
            if (oConfig.description) {
                oSanitized.description = this.sanitizeInput(oConfig.description);
            }
            if (oConfig.agentType) {
                oSanitized.agentType = this.sanitizeInput(oConfig.agentType);
            }
            if (oConfig.priority !== undefined) {
                oSanitized.priority = Math.max(1, Math.min(10, parseInt(oConfig.priority, 10) || 5));
            }
            if (oConfig.memoryLimit !== undefined) {
                oSanitized.memoryLimit = Math.max(128, Math.min(8192, parseInt(oConfig.memoryLimit, 10) || 512));
            }
            if (oConfig.timeout !== undefined) {
                oSanitized.timeout = Math.max(1, Math.min(3600, parseInt(oConfig.timeout, 10) || 300));
            }

            return oSanitized;
        },

        /**
         * Validate bulk operation parameters
         * @param {Array} aAgentIds - Array of agent IDs
         * @param {string} sOperation - Bulk operation type
         * @returns {Object} Validation result
         */
        validateBulkOperation(aAgentIds, sOperation) {
            const aErrors = [];

            if (!Array.isArray(aAgentIds)) {
                aErrors.push("Agent IDs must be provided as an array");
                return { isValid: false, errors: aErrors };
            }

            if (aAgentIds.length === 0) {
                aErrors.push("At least one agent ID must be provided");
                return { isValid: false, errors: aErrors };
            }

            // Limit bulk operations to 50 agents max for security
            if (aAgentIds.length > 50) {
                aErrors.push("Bulk operations limited to 50 agents maximum for security");
                return { isValid: false, errors: aErrors };
            }

            // Validate operation type
            const aValidOperations = ["START", "STOP", "RESTART", "UPDATE_CONFIG", "HEALTH_CHECK", "DELETE"];
            if (!sOperation || !aValidOperations.includes(sOperation)) {
                aErrors.push("Invalid bulk operation type");
            }

            // Validate each agent ID
            aAgentIds.forEach((sAgentId, index) => {
                if (typeof sAgentId !== "string" || sAgentId.length === 0) {
                    aErrors.push(`Agent ID at index ${ index } is invalid`);
                } else if (sAgentId.length > 50) {
                    aErrors.push(`Agent ID at index ${ index } is too long`);
                } else if (!/^[a-zA-Z0-9_\-]+$/.test(sAgentId)) {
                    aErrors.push(`Agent ID at index ${ index } contains invalid characters`);
                }
            });

            return {
                isValid: aErrors.length === 0,
                errors: aErrors,
                sanitizedIds: aAgentIds.map((id) => { return this.sanitizeInput(id); }),
                sanitizedOperation: this.sanitizeInput(sOperation)
            };
        },

        /**
         * General input sanitization
         * @param {string} sInput - Input to sanitize
         * @returns {string} Sanitized input
         */
        sanitizeInput(sInput) {
            if (typeof sInput !== "string") {
                return String(sInput || "");
            }

            // Remove potential XSS patterns
            const sSanitized = sInput
                .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, "")
                .replace(/<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi, "")
                .replace(/javascript:/gi, "")
                .replace(/on\w+\s*=/gi, "")
                .replace(/eval\s*\(/gi, "")
                .replace(/new\s+Function\s*\(/gi, "");

            return encodeXML(sSanitized);
        },

        /**
         * Secure AJAX request wrapper with CSRF protection
         * @param {Object} oOptions - Request options
         * @returns {Promise} Promise for the request
         */
        secureAjaxRequest(oOptions) {
            const that = this;

            return new Promise((resolve, reject) => {
                // Get CSRF token first
                that.getCSRFToken().then((sToken) => {
                    // Set default headers
                    oOptions.headers = oOptions.headers || {};
                    oOptions.headers["X-CSRF-Token"] = sToken;
                    oOptions.headers["Content-Type"] = oOptions.headers["Content-Type"] || "application/json";
                    oOptions.headers["X-Requested-With"] = "XMLHttpRequest";

                    // Add authentication header if available
                    const sAuthToken = sessionStorage.getItem("a2a_auth_token");
                    if (sAuthToken) {
                        oOptions.headers["Authorization"] = `Bearer ${ sAuthToken}`;
                    }

                    // Set timeout if not specified (default 30 seconds)
                    oOptions.timeout = oOptions.timeout || 30000;

                    // Make the request
                    jQuery.ajax(oOptions)
                        .done(resolve)
                        .fail((xhr, status, error) => {
                            Log.error("Secure AJAX request failed", {
                                status: xhr.status,
                                statusText: xhr.statusText,
                                url: oOptions.url
                            });
                            reject(xhr);
                        });
                }).catch((error) => {
                    Log.error("Failed to get CSRF token for secure request");
                    reject(error);
                });
            });
        },

        /**
         * Get CSRF token for secure requests
         * @returns {Promise<string>} Promise that resolves to CSRF token
         */
        getCSRFToken() {
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: "/a2a/common/v1/csrf-token",
                    type: "GET",
                    headers: {
                        "X-Requested-With": "XMLHttpRequest"
                    },
                    success(data) {
                        if (data && data.token) {
                            resolve(data.token);
                        } else {
                            reject(new Error("Invalid CSRF token response"));
                        }
                    },
                    error() {
                        reject(new Error("Failed to retrieve CSRF token"));
                    }
                });
            });
        },

        /**
         * Validate EventSource URL for security
         * @param {string} sUrl - URL to validate
         * @returns {Object} Validation result
         */
        validateEventSourceURL(sUrl) {
            const aErrors = [];

            if (!sUrl || typeof sUrl !== "string") {
                aErrors.push("Invalid URL format");
                return { isValid: false, errors: aErrors };
            }

            // Must be relative URL or same origin
            if (sUrl.startsWith("http://") || sUrl.startsWith("https://")) {
                try {
                    const oURL = new URL(sUrl);
                    const sCurrentHost = window.location.host;
                    if (oURL.host !== sCurrentHost) {
                        aErrors.push("External EventSource URLs not allowed");
                    }
                } catch (e) {
                    aErrors.push("Invalid URL format");
                }
            }

            // Check for dangerous patterns
            if (/javascript:|data:|vbscript:/i.test(sUrl)) {
                aErrors.push("Dangerous URL scheme detected");
            }

            return {
                isValid: aErrors.length === 0,
                errors: aErrors,
                sanitized: this.sanitizeInput(sUrl)
            };
        },

        /**
         * Security headers for API responses
         */
        getSecurityHeaders() {
            return {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Referrer-Policy": "strict-origin-when-cross-origin"
            };
        },

        /**
         * Log security events for audit trail
         * @param {string} sEvent - Event type
         * @param {string} sDescription - Event description
         * @param {Object} oData - Additional data
         */
        logSecurityEvent(sEvent, sDescription, oData) {
            const oLogData = {
                timestamp: new Date().toISOString(),
                event: sEvent,
                description: sDescription,
                component: "Agent7.Security",
                data: oData ? JSON.stringify(oData).substring(0, 500) : "",
                userAgent: navigator.userAgent.substring(0, 200)
            };

            Log.info(`Security Event: ${ sEvent}`, oLogData);

            // Also send to audit service
            this.secureAjaxRequest({
                url: "/a2a/common/v1/audit",
                type: "POST",
                data: JSON.stringify(oLogData)
            }).catch(() => {
                // Silent fail for audit service
            });
        }
    };
});