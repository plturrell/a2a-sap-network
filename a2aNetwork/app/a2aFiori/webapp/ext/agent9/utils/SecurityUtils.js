sap.ui.define([
    "sap/base/security/encodeXML",
    "sap/base/security/encodeJS",
    "sap/base/security/encodeURL",
    "sap/base/strings/escapeRegExp",
    "sap/base/Log"
], (encodeXML, encodeJS, encodeURL, escapeRegExp, Log) => {
    "use strict";

    /**
     * Security utilities for Agent 9 Reasoning Agent
     * Provides input validation, output encoding, and security controls for reasoning operations
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
                .replace(/\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g, "[IP_ADDRESS]") // IP addresses
                .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, "[EMAIL]") // Email addresses
                .replace(/password[:\s]*[^\s]+/gi, "password: [REDACTED]") // Passwords
                .replace(/token[:\s]*[^\s]+/gi, "token: [REDACTED]") // Tokens
                .replace(/key[:\s]*[^\s]+/gi, "key: [REDACTED]") // Keys
                .replace(/\b[A-F0-9]{32,}\b/gi, "[HASH]") // Hash values
                .replace(/\/[^\s]*\/[^\s]*/g, "[PATH]") // File paths
                .replace(/reasoning.*error/gi, "Reasoning error") // Reasoning details
                .replace(/inference.*failed/gi, "Inference failed") // Inference details
                .replace(/knowledge.*base/gi, "Knowledge base error"); // Knowledge base details

            // Limit length to prevent potential DoS
            if (sanitized.length > 200) {
                sanitized = `${sanitized.substring(0, 200) }...`;
            }

            return this.encodeHTML(sanitized);
        },

        /**
         * Validates input data based on type and constraints
         * @param {*} input - Input to validate
         * @param {string} type - Type of validation
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
            case "reasoningType":
                return this._validateReasoningType(inputStr, options);
            case "confidenceScore":
                return this._validateConfidenceScore(inputStr, options);
            case "inferenceDepth":
                return this._validateInferenceDepth(inputStr, options);
            case "reasoningRule":
                return this._validateReasoningRule(inputStr, options);
            case "logicalExpression":
                return this._validateLogicalExpression(inputStr, options);
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
                .replace(/[<>]/g, "") // Remove angle brackets
                .replace(/javascript:/gi, "") // Remove javascript: protocol
                .replace(/on\w+=/gi, "") // Remove event handlers
                .replace(/eval\(/gi, "") // Remove eval calls
                .replace(/new\s+Function/gi, "") // Remove Function constructor
                .trim();
        },

        /**
         * Sanitizes reasoning-specific data
         * @param {Object} reasoningData - Reasoning data to sanitize
         * @returns {Object} - Sanitized reasoning data
         */
        sanitizeReasoningData(reasoningData) {
            if (!reasoningData || typeof reasoningData !== "object") {
                return {};
            }

            const sanitized = {};

            // Sanitize each field based on its type
            if (reasoningData.conclusion) {
                sanitized.conclusion = this.sanitizeInput(reasoningData.conclusion);
            }
            if (reasoningData.inference) {
                sanitized.inference = this.sanitizeInput(reasoningData.inference);
            }
            if (reasoningData.contradiction) {
                sanitized.contradiction = {
                    description: this.sanitizeInput(reasoningData.contradiction.description || ""),
                    facts: Array.isArray(reasoningData.contradiction.facts) ?
                        reasoningData.contradiction.facts.map(f => this.sanitizeInput(f)) : [],
                    severity: this.sanitizeInput(reasoningData.contradiction.severity || "unknown")
                };
            }
            if (reasoningData.confidence) {
                sanitized.confidence = this.validateConfidenceScore(reasoningData.confidence).isValid ?
                    parseFloat(reasoningData.confidence) : 0;
            }

            return sanitized;
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
         * Creates secure AJAX configuration with CSRF protection
         * @param {Object} config - Base AJAX configuration
         * @returns {Object} - Secure AJAX configuration
         */
        createSecureAjaxConfig(config) {
            const secureConfig = Object.assign({}, config);

            // Add CSRF token for state-changing operations
            const method = (config.type || "GET").toUpperCase();
            if (["POST", "PUT", "DELETE", "PATCH"].includes(method)) {
                const token = this.getCSRFToken();
                if (token) {
                    secureConfig.headers = secureConfig.headers || {};
                    secureConfig.headers["X-CSRF-Token"] = token;
                }
            }

            // Set secure defaults
            secureConfig.contentType = secureConfig.contentType || "application/json";
            secureConfig.cache = false;
            secureConfig.timeout = secureConfig.timeout || 30000; // 30 second timeout

            // Add rate limiting headers for reasoning operations
            if (config.url && config.url.includes("/reason") || config.url.includes("/infer")) {
                secureConfig.headers = secureConfig.headers || {};
                secureConfig.headers["X-Rate-Limit-Category"] = "reasoning";
            }

            return secureConfig;
        },

        /**
         * Validates reasoning parameters
         * @param {Object} params - Reasoning parameters
         * @returns {Object} - Validation result
         */
        validateReasoningParameters(params) {
            const errors = [];

            // Validate confidence threshold
            if (params.confidenceThreshold !== undefined) {
                const confValidation = this.validateConfidenceScore(params.confidenceThreshold);
                if (!confValidation.isValid) {
                    errors.push(`Confidence threshold: ${ confValidation.message}`);
                }
            }

            // Validate inference depth
            if (params.maxInferenceDepth !== undefined) {
                const depthValidation = this.validateInferenceDepth(params.maxInferenceDepth);
                if (!depthValidation.isValid) {
                    errors.push(`Inference depth: ${ depthValidation.message}`);
                }
            }

            // Validate reasoning type
            if (params.reasoningType) {
                const typeValidation = this.validateReasoningType(params.reasoningType);
                if (!typeValidation.isValid) {
                    errors.push(`Reasoning type: ${ typeValidation.message}`);
                }
            }

            return {
                isValid: errors.length === 0,
                errors,
                message: errors.join("; ")
            };
        },

        /**
         * Logs security events for audit purposes
         * @param {string} event - Event type
         * @param {Object} details - Event details
         */
        auditLog(event, details = {}) {
            try {
                const logEntry = {
                    timestamp: new Date().toISOString(),
                    event: this.sanitizeInput(event),
                    details: this._sanitizeLogDetails(details),
                    userAgent: navigator.userAgent,
                    url: window.location.href,
                    component: "Agent9_ReasoningAgent"
                };

                // In production, this should go to a secure logging service
                Log.info("[AUDIT]", logEntry);
            } catch (e) {
                // Fail silently to avoid breaking application
            }
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
            if (/<script|javascript:|on\w+=/i.test(text)) {
                result.isValid = false;
                result.message = "Invalid characters detected";
                return result;
            }

            // Check for reasoning-specific dangerous patterns
            if (/eval\(|new\s+Function|setInterval|setTimeout/i.test(text)) {
                result.isValid = false;
                result.message = "Dynamic code execution not allowed";
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

        _validateReasoningType(type, options) {
            const result = { isValid: true, message: "" };
            const validTypes = ["DEDUCTIVE", "INDUCTIVE", "ABDUCTIVE", "ANALOGICAL",
                "PROBABILISTIC", "CAUSAL", "TEMPORAL", "MODAL"];

            if (!validTypes.includes(type)) {
                result.isValid = false;
                result.message = `Invalid reasoning type. Must be one of: ${ validTypes.join(", ")}`;
                return result;
            }

            return result;
        },

        _validateConfidenceScore(score, options) {
            const result = { isValid: true, message: "" };
            const numScore = parseFloat(score);

            if (isNaN(numScore)) {
                result.isValid = false;
                result.message = "Confidence score must be a number";
                return result;
            }

            if (numScore < 0 || numScore > 1) {
                result.isValid = false;
                result.message = "Confidence score must be between 0 and 1";
                return result;
            }

            return result;
        },

        _validateInferenceDepth(depth, options) {
            const result = { isValid: true, message: "" };
            const numDepth = parseInt(depth, 10);

            if (isNaN(numDepth)) {
                result.isValid = false;
                result.message = "Inference depth must be a number";
                return result;
            }

            if (numDepth < 1 || numDepth > 50) {
                result.isValid = false;
                result.message = "Inference depth must be between 1 and 50";
                return result;
            }

            return result;
        },

        _validateReasoningRule(rule, options) {
            const result = { isValid: true, message: "" };

            // Check for dangerous patterns in rules
            const dangerousPatterns = [
                /eval\(/gi,
                /Function\(/gi,
                /require\(/gi,
                /import\s+/gi,
                /__proto__/gi,
                /constructor\[/gi
            ];

            for (const pattern of dangerousPatterns) {
                if (pattern.test(rule)) {
                    result.isValid = false;
                    result.message = "Rule contains forbidden patterns";
                    return result;
                }
            }

            // Check rule length
            if (rule.length > 1000) {
                result.isValid = false;
                result.message = "Rule exceeds maximum length of 1000 characters";
                return result;
            }

            return result;
        },

        _validateLogicalExpression(expr, options) {
            const result = { isValid: true, message: "" };

            // Only allow safe logical operators and variables
            const safePattern = /^[\w\s\(\)\&\|\!\=\>\<\.\,]+$/;

            if (!safePattern.test(expr)) {
                result.isValid = false;
                result.message = "Logical expression contains invalid characters";
                return result;
            }

            // Check balanced parentheses
            let parenCount = 0;
            for (const char of expr) {
                if (char === "(") {parenCount++;}
                if (char === ")") {parenCount--;}
                if (parenCount < 0) {
                    result.isValid = false;
                    result.message = "Unbalanced parentheses in expression";
                    return result;
                }
            }

            if (parenCount !== 0) {
                result.isValid = false;
                result.message = "Unbalanced parentheses in expression";
                return result;
            }

            return result;
        },

        _sanitizeLogDetails(details) {
            const sanitized = {};

            for (const key in details) {
                if (details.hasOwnProperty(key)) {
                    const value = details[key];

                    // Don't log sensitive information
                    if (/password|token|secret|key/i.test(key)) {
                        sanitized[key] = "[REDACTED]";
                    } else if (typeof value === "string") {
                        sanitized[key] = this.sanitizeInput(value);
                    } else if (typeof value === "object") {
                        sanitized[key] = this._sanitizeLogDetails(value);
                    } else {
                        sanitized[key] = value;
                    }
                }
            }

            return sanitized;
        },

        /**
         * Validates confidence score
         * @param {*} score - Score to validate
         * @returns {Object} - Validation result
         */
        validateConfidenceScore(score) {
            return this._validateConfidenceScore(score);
        },

        /**
         * Validates inference depth
         * @param {*} depth - Depth to validate
         * @returns {Object} - Validation result
         */
        validateInferenceDepth(depth) {
            return this._validateInferenceDepth(depth);
        },

        /**
         * Validates reasoning type
         * @param {*} type - Type to validate
         * @returns {Object} - Validation result
         */
        validateReasoningType(type) {
            return this._validateReasoningType(type);
        }
    };
});