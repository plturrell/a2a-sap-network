sap.ui.define([
    "sap/base/security/encodeXML",
    "sap/base/security/encodeJS",
    "sap/base/security/encodeURL",
    "sap/base/strings/escapeRegExp",
    "sap/base/Log"
], (encodeXML, encodeJS, encodeURL, escapeRegExp, Log) => {
    "use strict";

    /**
     * Security utilities for Agent 8 Data Manager
     * Provides input validation, output encoding, and CSRF protection
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
                .replace(/\/[^\s]*\/[^\s]*/g, "[PATH]"); // File paths

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
                .trim();
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

            return secureConfig;
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
                    url: window.location.href
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
        }
    };
});