sap.ui.define([], function() {
    "use strict";

    /**
     * @namespace a2a.network.agent4.ext.utils.SecurityUtils
     * @description Security utilities for Agent 4 - Calculation Validation Agent.
     * Provides comprehensive security features for formula validation,
     * calculation security, and secure mathematical operations.
     */
    var SecurityUtils = {
        
        /**
         * @function escapeHTML
         * @description Escapes HTML entities to prevent XSS attacks
         * @param {string} str - String to escape
         * @returns {string} Escaped string
         * @public
         */
        escapeHTML: function(str) {
            if (!str) return "";
            var div = document.createElement("div");
            div.textContent = str;
            return div.innerHTML;
        },
        
        /**
         * @function validateFormula
         * @description Validates mathematical formulas for security vulnerabilities
         * @param {string} formula - Formula to validate
         * @returns {Object} Validation result
         * @public
         */
        validateFormula: function(formula) {
            var result = {
                isValid: true,
                errors: [],
                warnings: [],
                sanitizedFormula: ""
            };
            
            if (!formula || typeof formula !== "string") {
                result.isValid = false;
                result.errors.push("Formula must be a non-empty string");
                return result;
            }
            
            // Check formula length
            if (formula.length > 5000) {
                result.isValid = false;
                result.errors.push("Formula too long. Maximum 5000 characters allowed.");
                return result;
            }
            
            // Check for dangerous patterns
            var dangerousPatterns = [
                { pattern: /\beval\s*\(/gi, message: "eval() function not allowed" },
                { pattern: /\bexec\s*\(/gi, message: "exec() function not allowed" },
                { pattern: /\bsystem\s*\(/gi, message: "system() function not allowed" },
                { pattern: /\bimport\s+/gi, message: "import statements not allowed" },
                { pattern: /\brequire\s*\(/gi, message: "require() function not allowed" },
                { pattern: /\bnew\s+Function\s*\(/gi, message: "Dynamic function creation not allowed" },
                { pattern: /\b__[a-zA-Z_]+__/g, message: "Special attributes not allowed" },
                { pattern: /\bfile\b/gi, message: "File operations not allowed" },
                { pattern: /\bos\b/gi, message: "OS operations not allowed" },
                { pattern: /\bprocess\b/gi, message: "Process operations not allowed" }
            ];
            
            dangerousPatterns.forEach(function(item) {
                if (item.pattern.test(formula)) {
                    result.isValid = false;
                    result.errors.push(item.message);
                }
            });
            
            // Check for suspicious patterns (warnings)
            var suspiciousPatterns = [
                { pattern: /\bsetTimeout\s*\(/gi, message: "Async operations should be reviewed" },
                { pattern: /\bsetInterval\s*\(/gi, message: "Interval operations should be reviewed" },
                { pattern: /\bwhile\s*\(/gi, message: "Loops should be bounded" },
                { pattern: /\bfor\s*\(/gi, message: "Loops should be bounded" }
            ];
            
            suspiciousPatterns.forEach(function(item) {
                if (item.pattern.test(formula)) {
                    result.warnings.push(item.message);
                }
            });
            
            // Sanitize the formula
            result.sanitizedFormula = this._sanitizeFormula(formula);
            
            return result;
        },
        
        /**
         * @function validateCalculationInput
         * @description Validates calculation input data
         * @param {any} input - Input to validate
         * @param {string} type - Expected type
         * @returns {boolean} True if valid
         * @public
         */
        validateCalculationInput: function(input, type) {
            switch(type) {
                case 'number':
                    return typeof input === 'number' && !isNaN(input) && isFinite(input);
                case 'string':
                    return typeof input === 'string' && input.length > 0 && input.length < 1000;
                case 'array':
                    return Array.isArray(input) && input.length <= 1000;
                case 'object':
                    return typeof input === 'object' && input !== null;
                default:
                    return false;
            }
        },
        
        /**
         * @function sanitizeCalculationResult
         * @description Sanitizes calculation results for display
         * @param {any} result - Result to sanitize
         * @returns {string} Sanitized result
         * @public
         */
        sanitizeCalculationResult: function(result) {
            if (result === null || result === undefined) {
                return "null";
            }
            
            if (typeof result === "number") {
                if (!isFinite(result)) {
                    return "Invalid Number";
                }
                // Round to reasonable precision
                return Number(result.toFixed(10)).toString();
            }
            
            if (typeof result === "string") {
                return this.escapeHTML(result);
            }
            
            if (typeof result === "boolean") {
                return result.toString();
            }
            
            if (Array.isArray(result)) {
                return "[Array with " + result.length + " items]";
            }
            
            if (typeof result === "object") {
                return "[Object]";
            }
            
            return this.escapeHTML(String(result));
        },
        
        /**
         * @function secureAjaxRequest
         * @description Makes a secure AJAX request with CSRF protection
         * @param {Object} options - jQuery AJAX options
         * @returns {Promise} Promise resolving to response
         * @public
         */
        secureAjaxRequest: function(options) {
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
         * @function validateTaskId
         * @description Validates task ID format
         * @param {string} taskId - Task ID to validate
         * @returns {boolean} True if valid
         * @public
         */
        validateTaskId: function(taskId) {
            if (!taskId || typeof taskId !== "string") {
                return false;
            }
            
            // Task ID should be alphanumeric with dashes/underscores, max 50 chars
            return /^[a-zA-Z0-9_-]{1,50}$/.test(taskId);
        },
        
        /**
         * @function validateUserInput
         * @description Validates general user input
         * @param {string} input - Input to validate
         * @param {number} maxLength - Maximum allowed length
         * @returns {Object} Validation result
         * @public
         */
        validateUserInput: function(input, maxLength) {
            maxLength = maxLength || 1000;
            
            var result = {
                isValid: true,
                message: ""
            };
            
            if (!input || typeof input !== "string") {
                result.isValid = false;
                result.message = "Input must be a non-empty string";
                return result;
            }
            
            if (input.length > maxLength) {
                result.isValid = false;
                result.message = "Input too long. Maximum " + maxLength + " characters allowed.";
                return result;
            }
            
            // Check for XSS patterns
            var xssPatterns = [
                /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
                /javascript:/gi,
                /on\w+\s*=/gi,
                /<iframe\b/gi,
                /<object\b/gi,
                /<embed\b/gi
            ];
            
            xssPatterns.forEach(function(pattern) {
                if (pattern.test(input)) {
                    result.isValid = false;
                    result.message = "Input contains potentially harmful content";
                }
            });
            
            return result;
        },
        
        /**
         * @function checkPermissions
         * @description Checks if user has required permissions
         * @param {string} operation - Operation to check
         * @returns {boolean} True if authorized
         * @public
         */
        checkPermissions: function(operation) {
            // Define required permissions for operations
            var requiredPermissions = {
                "CREATE_TASK": ["calc.validation.create"],
                "START_VALIDATION": ["calc.validation.execute"],
                "VIEW_RESULTS": ["calc.validation.view"],
                "BATCH_VALIDATION": ["calc.validation.batch"],
                "FORMULA_BUILDER": ["calc.formula.create"],
                "ANALYTICS": ["calc.analytics.view"],
                "GENERATE_REPORT": ["calc.report.generate"],
                "OPTIMIZE": ["calc.optimization.execute"]
            };
            
            var permissions = requiredPermissions[operation];
            if (!permissions) {
                return false;
            }
            
            // Check user permissions (integrate with actual auth system)
            var userPermissions = this._getUserPermissions();
            
            for (var i = 0; i < permissions.length; i++) {
                if (!userPermissions.includes(permissions[i])) {
                    return false;
                }
            }
            
            return true;
        },
        
        /**
         * @function _sanitizeFormula
         * @description Sanitizes a formula string
         * @param {string} formula - Formula to sanitize
         * @returns {string} Sanitized formula
         * @private
         */
        _sanitizeFormula: function(formula) {
            if (!formula) return "";
            
            // Remove dangerous function calls
            formula = formula.replace(/\beval\s*\([^)]*\)/gi, "");
            formula = formula.replace(/\bexec\s*\([^)]*\)/gi, "");
            formula = formula.replace(/\bsystem\s*\([^)]*\)/gi, "");
            
            // Remove HTML tags
            formula = formula.replace(/<[^>]*>/g, "");
            
            // Trim and normalize whitespace
            formula = formula.replace(/\s+/g, " ").trim();
            
            return formula;
        },
        
        /**
         * @function _getCSRFToken
         * @description Gets CSRF token for secure requests
         * @returns {string} CSRF token
         * @private
         */
        _getCSRFToken: function() {
            // Try to get token from meta tag
            var token = document.querySelector('meta[name="csrf-token"]');
            if (token) {
                return token.getAttribute("content");
            }
            
            // Try to get from cookie
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var cookie = cookies[i].trim();
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
        _generateCSRFToken: function() {
            var array = new Uint8Array(32);
            crypto.getRandomValues(array);
            return Array.from(array, function(byte) {
                return ('0' + byte.toString(16)).slice(-2);
            }).join('');
        },
        
        /**
         * @function _getUserPermissions
         * @description Gets current user permissions
         * @returns {Array<string>} User permissions
         * @private
         */
        _getUserPermissions: function() {
            // This would integrate with your actual auth system
            // For now, return from session storage or default
            var permissions = sessionStorage.getItem("user-permissions");
            if (permissions) {
                try {
                    return JSON.parse(permissions);
                } catch (e) {
                    return [];
                }
            }
            
            // Default permissions
            return ["calc.validation.create", "calc.validation.execute", "calc.validation.view"];
        }
    };
    
    return SecurityUtils;
});