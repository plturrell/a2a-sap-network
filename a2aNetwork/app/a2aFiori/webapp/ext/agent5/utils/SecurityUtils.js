sap.ui.define([], () => {
    "use strict";

    /**
     * @namespace a2a.network.agent5.ext.utils.SecurityUtils
     * @description Security utilities for Agent 5 - QA Validation Agent.
     * Provides comprehensive security features for test case validation,
     * quality assurance operations, and secure QA workflows.
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
         * @function validateTestCase
         * @description Validates test case data for security vulnerabilities
         * @param {string} testCase - Test case content to validate
         * @returns {Object} Validation result
         * @public
         */
        validateTestCase(testCase) {
            const result = {
                isValid: true,
                errors: [],
                warnings: [],
                sanitizedTestCase: ""
            };

            if (!testCase || typeof testCase !== "string") {
                result.isValid = false;
                result.errors.push("Test case must be a non-empty string");
                return result;
            }

            // Check test case length
            if (testCase.length > 5000) {
                result.isValid = false;
                result.errors.push("Test case too long. Maximum 5000 characters allowed.");
                return result;
            }

            // Check for dangerous test patterns
            const dangerousPatterns = [
                { pattern: /\beval\s*\(/gi, message: "eval() function not allowed in test cases" },
                { pattern: /\bexec\s*\(/gi, message: "exec() function not allowed in test cases" },
                { pattern: /\bsystem\s*\(/gi, message: "system() function not allowed in test cases" },
                { pattern: /\bsubprocess/gi, message: "subprocess operations not allowed in test cases" },
                { pattern: /\bimport\s+os/gi, message: "OS imports not allowed in test cases" },
                { pattern: /\brequire\s*\(/gi, message: "require() function not allowed in test cases" },
                { pattern: /\bnew\s+Function\s*\(/gi, message: "Dynamic function creation not allowed" },
                { pattern: /\b__[a-zA-Z_]+__/g, message: "Special attributes not allowed in test cases" },
                { pattern: /\bfile\b.*\bopen\b/gi, message: "File operations not allowed in test cases" },
                { pattern: /\bshell\b/gi, message: "Shell operations not allowed in test cases" }
            ];

            dangerousPatterns.forEach((item) => {
                if (item.pattern.test(testCase)) {
                    result.isValid = false;
                    result.errors.push(item.message);
                }
            });

            // Check for suspicious patterns (warnings)
            const suspiciousPatterns = [
                { pattern: /\bsetTimeout\s*\(/gi, message: "Async operations in test cases should be reviewed" },
                { pattern: /\bsetInterval\s*\(/gi, message: "Interval operations in test cases should be reviewed" },
                { pattern: /\bwhile\s*\([^)]*length/gi, message: "Potentially infinite loops should be bounded" },
                { pattern: /\bfor\s*\([^)]*;\s*[^;]*<\s*\d{4,}/gi, message: "Large loops should be reviewed for performance" }
            ];

            suspiciousPatterns.forEach((item) => {
                if (item.pattern.test(testCase)) {
                    result.warnings.push(item.message);
                }
            });

            // Sanitize the test case
            result.sanitizedTestCase = this._sanitizeTestCase(testCase);

            return result;
        },

        /**
         * @function validateQAInput
         * @description Validates QA input data
         * @param {any} input - Input to validate
         * @param {string} type - Expected type
         * @returns {boolean} True if valid
         * @public
         */
        validateQAInput(input, type) {
            switch (type) {
            case "taskName":
                return typeof input === "string" && input.length > 0 && input.length <= 200;
            case "testDescription":
                return typeof input === "string" && input.length > 0 && input.length <= 2000;
            case "qualityScore":
                return typeof input === "number" && input >= 0 && input <= 100;
            case "testCaseId":
                return typeof input === "string" && /^[a-zA-Z0-9_-]{1,50}$/.test(input);
            case "defectId":
                return typeof input === "string" && /^[a-zA-Z0-9_-]{1,30}$/.test(input);
            default:
                return false;
            }
        },

        /**
         * @function sanitizeQAOutput
         * @description Sanitizes QA output data for display
         * @param {any} output - Output to sanitize
         * @returns {string} Sanitized output
         * @public
         */
        sanitizeQAOutput(output) {
            if (output === null || output === undefined) {
                return "null";
            }

            if (typeof output === "string") {
                return this.escapeHTML(output);
            }

            if (typeof output === "number") {
                if (!isFinite(output)) {
                    return "Invalid Number";
                }
                return Number(output.toFixed(2)).toString();
            }

            if (typeof output === "boolean") {
                return output.toString();
            }

            if (Array.isArray(output)) {
                return `[Array with ${ output.length } items]`;
            }

            if (typeof output === "object") {
                return "[Object]";
            }

            return this.escapeHTML(String(output));
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
         * @function validateTestExecutionParams
         * @description Validates test execution parameters
         * @param {Object} params - Test execution parameters
         * @returns {Object} Validation result
         * @public
         */
        validateTestExecutionParams(params) {
            const result = {
                isValid: true,
                errors: []
            };

            if (!params || typeof params !== "object") {
                result.isValid = false;
                result.errors.push("Test execution parameters must be an object");
                return result;
            }

            // Validate timeout
            if (params.timeout && (typeof params.timeout !== "number" || params.timeout <= 0 || params.timeout > 300000)) {
                result.isValid = false;
                result.errors.push("Timeout must be a positive number less than 300000ms");
            }

            // Validate environment
            if (params.environment && typeof params.environment !== "string") {
                result.isValid = false;
                result.errors.push("Environment must be a string");
            }

            // Validate test data
            if (params.testData) {
                try {
                    if (typeof params.testData === "string") {
                        JSON.parse(params.testData);
                    }
                } catch (e) {
                    result.isValid = false;
                    result.errors.push("Test data must be valid JSON");
                }
            }

            return result;
        },

        /**
         * @function validateDefectReport
         * @description Validates defect report data
         * @param {Object} defect - Defect report data
         * @returns {Object} Validation result
         * @public
         */
        validateDefectReport(defect) {
            const result = {
                isValid: true,
                errors: []
            };

            if (!defect || typeof defect !== "object") {
                result.isValid = false;
                result.errors.push("Defect report must be an object");
                return result;
            }

            // Validate required fields
            if (!defect.title || typeof defect.title !== "string" || defect.title.length > 200) {
                result.isValid = false;
                result.errors.push("Defect title is required and must be less than 200 characters");
            }

            if (!defect.description || typeof defect.description !== "string" || defect.description.length > 5000) {
                result.isValid = false;
                result.errors.push("Defect description is required and must be less than 5000 characters");
            }

            // Validate severity
            const validSeverities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
            if (defect.severity && validSeverities.indexOf(defect.severity) === -1) {
                result.isValid = false;
                result.errors.push("Invalid defect severity");
            }

            // Validate priority
            const validPriorities = ["P1", "P2", "P3", "P4"];
            if (defect.priority && validPriorities.indexOf(defect.priority) === -1) {
                result.isValid = false;
                result.errors.push("Invalid defect priority");
            }

            return result;
        },

        /**
         * @function checkPermissions
         * @description Checks if user has required permissions
         * @param {string} operation - Operation to check
         * @returns {boolean} True if authorized
         * @public
         */
        checkPermissions(operation) {
            // Define required permissions for operations
            const requiredPermissions = {
                "CREATE_TEST": ["qa.validation.create"],
                "EXECUTE_TEST": ["qa.validation.execute"],
                "VIEW_RESULTS": ["qa.validation.view"],
                "CREATE_DEFECT": ["qa.defect.create"],
                "APPROVE_TEST": ["qa.validation.approve"],
                "MANAGE_QUALITY_RULES": ["qa.rules.manage"],
                "VIEW_ANALYTICS": ["qa.analytics.view"],
                "GENERATE_REPORT": ["qa.report.generate"]
            };

            const permissions = requiredPermissions[operation];
            if (!permissions) {
                return false;
            }

            // Check user permissions (integrate with actual auth system)
            const userPermissions = this._getUserPermissions();

            for (let i = 0; i < permissions.length; i++) {
                if (!userPermissions.includes(permissions[i])) {
                    return false;
                }
            }

            return true;
        },

        /**
         * @function _sanitizeTestCase
         * @description Sanitizes a test case string
         * @param {string} testCase - Test case to sanitize
         * @returns {string} Sanitized test case
         * @private
         */
        _sanitizeTestCase(testCase) {
            if (!testCase) {return "";}

            // Remove dangerous function calls
            testCase = testCase.replace(/\beval\s*\([^)]*\)/gi, "");
            testCase = testCase.replace(/\bexec\s*\([^)]*\)/gi, "");
            testCase = testCase.replace(/\bsystem\s*\([^)]*\)/gi, "");
            testCase = testCase.replace(/\bsubprocess\.[^)]*\([^)]*\)/gi, "");

            // Remove HTML tags
            testCase = testCase.replace(/<[^>]*>/g, "");

            // Trim and normalize whitespace
            testCase = testCase.replace(/\s+/g, " ").trim();

            return testCase;
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
         * @function _getUserPermissions
         * @description Gets current user permissions
         * @returns {Array<string>} User permissions
         * @private
         */
        _getUserPermissions() {
            // This would integrate with your actual auth system
            // For now, return from session storage or default
            const permissions = sessionStorage.getItem("user-permissions");
            if (permissions) {
                try {
                    return JSON.parse(permissions);
                } catch (e) {
                    return [];
                }
            }

            // Default permissions
            return ["qa.validation.create", "qa.validation.execute", "qa.validation.view"];
        }
    };

    return SecurityUtils;
});