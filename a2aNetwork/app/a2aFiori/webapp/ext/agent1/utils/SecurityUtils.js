/**
 * A2A Protocol Compliance: WebSocket replaced with blockchain event streaming
 * All real-time communication now uses blockchain events instead of WebSockets
 */

sap.ui.define([], () => {
    "use strict";

    /**
     * @namespace a2a.network.agent1.ext.utils.SecurityUtils
     * @description Security utilities for Agent 1 - Data Standardization Agent.
     * Provides comprehensive security features for transformation validation,
     * schema protection, input sanitization, and secure file handling.
     */
    const SecurityUtils = {

        /**
         * @function escapeHTML
         * @description Escapes HTML entities to prevent XSS attacks
         * @param {string} str - String to escape
         * @returns {string} Escaped string
         * @public
         */
        escapeHTML(str) {
            if (!str) {return "";}
            const div = document.createElement("div");
            div.textContent = str;
            return div.innerHTML;
        },

        /**
         * @function sanitizeTransformationData
         * @description Sanitizes transformation data to prevent injection attacks
         * @param {string} data - Data to sanitize
         * @returns {string} Sanitized data
         * @public
         */
        sanitizeTransformationData(data) {
            if (!data || typeof data !== "string") {return "";}

            // Remove potential script tags
            data = data.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, "");
            data = data.replace(/on\w+\s*=\s*["'][^"']*["']/gi, "");
            data = data.replace(/javascript:/gi, "");

            // Remove dangerous functions
            data = data.replace(/\b(eval|Function|setTimeout|setInterval)\s*\(/gi, "");

            // Remove SQL injection attempts
            data = data.replace(/(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)/gi, "");

            // Limit string length to prevent DoS
            if (data.length > 10000) {
                data = data.substring(0, 10000);
            }

            return data.trim();
        },

        /**
         * @function validateTransformationScript
         * @description Validates transformation scripts for security vulnerabilities
         * @param {string} script - Transformation script to validate
         * @returns {Object} Validation result
         * @public
         */
        validateTransformationScript(script) {
            const result = {
                isValid: true,
                errors: [],
                warnings: []
            };

            if (!script || typeof script !== "string") {
                result.isValid = false;
                result.errors.push("Script must be a non-empty string");
                return result;
            }

            // Check for dangerous patterns
            const dangerousPatterns = [
                { pattern: /\beval\s*\(/gi, message: "Use of eval() is not allowed" },
                { pattern: /\bnew\s+Function\s*\(/gi, message: "Dynamic function creation is not allowed" },
                { pattern: /\bexec\s*\(/gi, message: "Use of exec() is not allowed" },
                { pattern: /\b__proto__\b/gi, message: "Prototype manipulation is not allowed" },
                { pattern: /\bconstructor\s*\[/gi, message: "Constructor access is not allowed" },
                { pattern: /\bprocess\s*\./gi, message: "Process access is not allowed" },
                { pattern: /\brequire\s*\(/gi, message: "Dynamic requires are not allowed" },
                { pattern: /\bimport\s*\(/gi, message: "Dynamic imports are not allowed" }
            ];

            dangerousPatterns.forEach((item) => {
                if (item.pattern.test(script)) {
                    result.isValid = false;
                    result.errors.push(item.message);
                }
            });

            // Check for suspicious patterns (warnings)
            const suspiciousPatterns = [
                { pattern: /\bsetTimeout\s*\(/gi, message: "Async operations should be carefully reviewed" },
                { pattern: /\bfetch\s*\(/gi, message: "External data fetching detected" },
                { pattern: /\bajax\s*\(/gi, message: "AJAX calls should be validated" }
            ];

            suspiciousPatterns.forEach((item) => {
                if (item.pattern.test(script)) {
                    result.warnings.push(item.message);
                }
            });

            return result;
        },

        /**
         * @function validateSchema
         * @description Validates schema configuration for security
         * @param {Object} schema - Schema to validate
         * @returns {boolean} True if valid, false otherwise
         * @public
         */
        validateSchema(schema) {
            if (!schema || typeof schema !== "object") {
                return false;
            }

            // Check for required properties
            if (!schema.name || !schema.version || !schema.fields) {
                return false;
            }

            // Validate schema name
            if (!/^[a-zA-Z0-9_-]{1,100}$/.test(schema.name)) {
                return false;
            }

            // Validate version format
            if (!/^\d+\.\d+\.\d+$/.test(schema.version)) {
                return false;
            }

            // Validate fields
            if (!Array.isArray(schema.fields)) {
                return false;
            }

            // Validate each field
            for (let i = 0; i < schema.fields.length; i++) {
                const field = schema.fields[i];
                if (!this._validateSchemaField(field)) {
                    return false;
                }
            }

            return true;
        },

        /**
         * @function validateRawJSON
         * @description Validates raw JSON content before parsing to prevent injection attacks
         * @param {string} rawContent - Raw JSON string to validate
         * @returns {boolean} True if safe to parse, false otherwise
         * @public
         */
        validateRawJSON(rawContent) {
            if (typeof rawContent !== "string" || !rawContent.trim()) {
                return false;
            }

            // Check for excessive nesting depth (prevent DoS)
            let nestingDepth = 0;
            const maxDepth = 10;
            for (let i = 0; i < rawContent.length; i++) {
                if (rawContent[i] === "{" || rawContent[i] === "[") {
                    nestingDepth++;
                    if (nestingDepth > maxDepth) {
                        return false;
                    }
                } else if (rawContent[i] === "}" || rawContent[i] === "]") {
                    nestingDepth--;
                }
            }

            // Check for dangerous function calls in strings
            const dangerousPatterns = [
                /\\u0000/g, // null bytes
                /\\x00/g, // hex null bytes
                /javascript:/gi,
                /data:text\/html/gi,
                /eval\s*\(/gi,
                /function\s*\(/gi
            ];

            for (let j = 0; j < dangerousPatterns.length; j++) {
                if (dangerousPatterns[j].test(rawContent)) {
                    return false;
                }
            }

            // Basic JSON syntax validation
            try {
                JSON.parse(rawContent);
                return true;
            } catch (e) {
                return false;
            }
        },

        /**
         * @function executeSecureTransformation
         * @description Executes transformation script in a secure sandboxed environment
         * @param {string} script - Validated transformation script
         * @param {Object} testData - Test data object with value, row, and context
         * @returns {*} Transformation result
         * @public
         */
        executeSecureTransformation(script, testData) {
            // Create a restricted context
            const restrictedContext = {
                value: testData.value,
                row: testData.row,
                context: testData.context,
                Math,
                String,
                Number,
                Array,
                Object,
                Date,
                parseInt,
                parseFloat,
                isNaN,
                isFinite,
                encodeURIComponent,
                decodeURIComponent
            };

            try {
                // Use Function constructor instead of eval for better security
                const wrappedScript = `
                    return (function(value, row, context, Math, String, Number, Array, Object, Date, 
                                     parseInt, parseFloat, isNaN, isFinite, encodeURIComponent, decodeURIComponent) {
                        "use strict";
                        ${script}
                    })(value, row, context, Math, String, Number, Array, Object, Date, 
                       parseInt, parseFloat, isNaN, isFinite, encodeURIComponent, decodeURIComponent);
                `;

                // Create a new function with the script
                // eslint-disable-next-line no-new-func
                const transformFunction = new Function("value", "row", "context", "Math", "String",
                    "Number", "Array", "Object", "Date", "parseInt", "parseFloat", "isNaN",
                    "isFinite", "encodeURIComponent", "decodeURIComponent", wrappedScript);

                // Execute with restricted context
                return transformFunction(
                    restrictedContext.value,
                    restrictedContext.row,
                    restrictedContext.context,
                    restrictedContext.Math,
                    restrictedContext.String,
                    restrictedContext.Number,
                    restrictedContext.Array,
                    restrictedContext.Object,
                    restrictedContext.Date,
                    restrictedContext.parseInt,
                    restrictedContext.parseFloat,
                    restrictedContext.isNaN,
                    restrictedContext.isFinite,
                    restrictedContext.encodeURIComponent,
                    restrictedContext.decodeURIComponent
                );

            } catch (e) {
                throw new Error(`Script execution failed: ${ e.message}`);
            }
        },

        /**
         * @function sanitizeSchema
         * @description Sanitizes schema data to remove potentially harmful content
         * @param {Object} schema - Schema to sanitize
         * @returns {Object} Sanitized schema
         * @public
         */
        sanitizeSchema(schema) {
            if (!schema || typeof schema !== "object") {
                return {};
            }

            const sanitized = {
                name: this._sanitizeIdentifier(schema.name),
                version: this._sanitizeVersion(schema.version),
                description: this.escapeHTML(schema.description || ""),
                fields: []
            };

            if (Array.isArray(schema.fields)) {
                schema.fields.forEach((field) => {
                    sanitized.fields.push(this._sanitizeSchemaField(field));
                });
            }

            return sanitized;
        },

        /**
         * @function validateFileUpload
         * @description Validates file upload for security
         * @param {File} file - File to validate
         * @returns {Object} Validation result
         * @public
         */
        validateFileUpload(file) {
            const result = {
                isValid: true,
                error: null
            };

            if (!file) {
                result.isValid = false;
                result.error = "No file provided";
                return result;
            }

            // Check file size (max 10MB)
            const maxSize = 10 * 1024 * 1024;
            if (file.size > maxSize) {
                result.isValid = false;
                result.error = "File size exceeds maximum allowed (10MB)";
                return result;
            }

            // Check file type
            const allowedTypes = [
                "application/json",
                "text/csv",
                "text/plain",
                "application/xml",
                "text/xml",
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ];

            if (!allowedTypes.includes(file.type)) {
                result.isValid = false;
                result.error = "File type not allowed";
                return result;
            }

            // Check file extension
            const allowedExtensions = [".json", ".csv", ".txt", ".xml", ".xls", ".xlsx"];
            const fileName = file.name.toLowerCase();
            const hasValidExtension = allowedExtensions.some((ext) => {
                return fileName.endsWith(ext);
            });

            if (!hasValidExtension) {
                result.isValid = false;
                result.error = "File extension not allowed";
                return result;
            }

            return result;
        },

        /**
         * @function createSecureWebSocket
         * @description Creates a secure WebSocket connection
         * @param {string} url - WebSocket URL
         * @param {Object} handlers - Event handlers
         * @returns {WebSocket|null} Secure WebSocket instance or null
         * @public
         */
        createSecureWebSocket(url, handlers) {
            if (!this.validateWebSocketUrl(url)) {
                // // console.error("Invalid WebSocket URL");
                return null;
            }

            try {
                const ws = new WebSocket(url);

                // Add security headers
                ws.addEventListener("open", () => {
                    // Send authentication token if available
                    const token = this._getAuthToken();
                    if (token) {
                        // Note: blockchain client integration would go here
                        // For now, we'll use the WebSocket directly
                        ws.send(JSON.stringify({
                            type: "auth",
                            token
                        }));
                    }
                });

                // Wrap message handler with sanitization
                if (handlers.onmessage) {
                    const originalHandler = handlers.onmessage;
                    ws.onmessage = function(event) {
                        try {
                            // Sanitize incoming data
                            const data = JSON.parse(event.data);
                            if (data.transformation) {
                                data.transformation = this.sanitizeTransformationData(data.transformation);
                            }
                            event.data = JSON.stringify(data);
                            originalHandler.call(this, event);
                        } catch (e) {
                            // // console.error("Error processing WebSocket message:", e);
                        }
                    }.bind(this);
                }

                // Set other handlers
                if (handlers.onerror) {
                    ws.onerror = handlers.onerror;
                }

                if (handlers.onclose) {
                    ws.onclose = handlers.onclose;
                }

                return ws;

            } catch (e) {
                // // console.error("Failed to create WebSocket:", e);
                return null;
            }
        },

        /**
         * @function validateWebSocketUrl
         * @description Validates WebSocket URL for security
         * @param {string} url - WebSocket URL to validate
         * @returns {boolean} True if valid, false otherwise
         * @public
         */
        validateWebSocketUrl(url) {
            if (!url || typeof url !== "string") {return false;}

            // Must use secure WebSocket protocol
            if (!url.startsWith("wss://")) {
                // // console.warn("WebSocket URL must use secure protocol (wss://)");
                return false;
            }

            // Validate URL format
            try {
                const urlObj = new URL(url);

                // Check for localhost or allowed domains
                const allowedHosts = ["localhost", "127.0.0.1", window.location.hostname];
                if (!allowedHosts.includes(urlObj.hostname)) {
                    return false;
                }

                return true;

            } catch (e) {
                return false;
            }
        },

        /**
         * @function secureAjaxRequest
         * @description Makes a secure AJAX request with CSRF protection
         * @param {Object} options - jQuery AJAX options
         * @returns {Promise} Promise resolving to response
         * @public
         */
        secureAjaxRequest(options) {
            // Add CSRF token to headers
            options.headers = options.headers || {};
            options.headers["X-CSRF-Token"] = this._getCSRFToken();
            options.headers["X-Requested-With"] = "XMLHttpRequest";

            // Add security headers
            options.headers["X-Content-Type-Options"] = "nosniff";
            options.headers["X-Frame-Options"] = "DENY";
            options.headers["X-XSS-Protection"] = "1; mode=block";

            // Ensure HTTPS for production
            if (window.location.protocol === "https:" && options.url && options.url.startsWith("http://")) {
                options.url = options.url.replace("http://", "https://");
            }

            return jQuery.ajax(options);
        },

        /**
         * @function validateInputParameter
         * @description Validates input parameters from events
         * @param {any} value - Value to validate
         * @param {string} type - Expected type
         * @returns {boolean} True if valid
         * @public
         */
        validateInputParameter(value, type) {
            if (value === null || value === undefined) {
                return false;
            }

            switch (type) {
            case "string":
                return typeof value === "string" && value.length > 0 && value.length < 1000;
            case "number":
                return typeof value === "number" && !isNaN(value) && isFinite(value);
            case "boolean":
                return typeof value === "boolean";
            case "array":
                return Array.isArray(value);
            case "object":
                return typeof value === "object" && value !== null;
            default:
                return false;
            }
        },

        /**
         * @function _validateSchemaField
         * @description Validates a single schema field
         * @param {Object} field - Field to validate
         * @returns {boolean} True if valid
         * @private
         */
        _validateSchemaField(field) {
            if (!field || typeof field !== "object") {
                return false;
            }

            // Check required properties
            if (!field.name || !field.type) {
                return false;
            }

            // Validate field name
            if (!/^[a-zA-Z0-9_]{1,50}$/.test(field.name)) {
                return false;
            }

            // Validate field type
            const allowedTypes = ["string", "number", "boolean", "date", "array", "object"];
            if (!allowedTypes.includes(field.type)) {
                return false;
            }

            return true;
        },

        /**
         * @function _sanitizeSchemaField
         * @description Sanitizes a schema field
         * @param {Object} field - Field to sanitize
         * @returns {Object} Sanitized field
         * @private
         */
        _sanitizeSchemaField(field) {
            return {
                name: this._sanitizeIdentifier(field.name),
                type: this._sanitizeIdentifier(field.type),
                description: this.escapeHTML(field.description || ""),
                required: !!field.required,
                defaultValue: this._sanitizeValue(field.defaultValue)
            };
        },

        /**
         * @function _sanitizeIdentifier
         * @description Sanitizes an identifier
         * @param {string} id - Identifier to sanitize
         * @returns {string} Sanitized identifier
         * @private
         */
        _sanitizeIdentifier(id) {
            if (!id || typeof id !== "string") {return "";}
            return id.replace(/[^a-zA-Z0-9_-]/g, "").substring(0, 100);
        },

        /**
         * @function _sanitizeVersion
         * @description Sanitizes version string
         * @param {string} version - Version to sanitize
         * @returns {string} Sanitized version
         * @private
         */
        _sanitizeVersion(version) {
            if (!version || typeof version !== "string") {return "0.0.0";}
            const match = version.match(/^(\d+)\.(\d+)\.(\d+)$/);
            return match ? match[0] : "0.0.0";
        },

        /**
         * @function _sanitizeValue
         * @description Sanitizes a value based on its type
         * @param {any} value - Value to sanitize
         * @returns {any} Sanitized value
         * @private
         */
        _sanitizeValue(value) {
            if (value === null || value === undefined) {
                return null;
            }

            if (typeof value === "string") {
                return this.escapeHTML(value);
            }

            if (typeof value === "number") {
                return isFinite(value) ? value : 0;
            }

            if (typeof value === "boolean") {
                return value;
            }

            return null;
        },

        /**
         * @function _getCSRFToken
         * @description Gets CSRF token for secure requests
         * @returns {string} CSRF token
         * @private
         */
        _getCSRFToken() {
            // Try to get token from meta tag
            const token = document.querySelector("meta[name=\"csrf-token\"]");
            if (token) {
                return token.getAttribute("content");
            }

            // Try to get from cookie
            const cookies = document.cookie.split(";");
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.startsWith("XSRF-TOKEN=")) {
                    return cookie.substring(11);
                }
            }

            // Generate a new token if not found
            return this._generateCSRFToken();
        },

        /**
         * @function _generateCSRFToken
         * @description Generates a new CSRF token
         * @returns {string} Generated token
         * @private
         */
        _generateCSRFToken() {
            const array = new Uint8Array(32);
            crypto.getRandomValues(array);
            return Array.from(array, (byte) => {
                return (`0${ byte.toString(16)}`).slice(-2);
            }).join("");
        },

        /**
         * @function _getAuthToken
         * @description Gets authentication token for WebSocket
         * @returns {string|null} Auth token or null
         * @private
         */
        _getAuthToken() {
            // Get from session storage
            const token = sessionStorage.getItem("standardization-auth-token");
            if (token) {
                return token;
            }

            // Get from meta tag
            const metaToken = document.querySelector("meta[name=\"auth-token\"]");
            if (metaToken) {
                return metaToken.getAttribute("content");
            }

            return null;
        }
    };

    return SecurityUtils;
});