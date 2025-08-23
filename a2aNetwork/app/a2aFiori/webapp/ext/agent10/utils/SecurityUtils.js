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
                        Log.info("Secure function call successful", sFunctionName);
                        if (originalSuccess) {
                            originalSuccess(data, response);
                        }
                        resolve(data);
                    };
                    
                    secureParams.error = function(error) {
                        Log.error("Secure function call failed", error);
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
         * Validate and sanitize formula input
         * @param {string} formula - Formula to validate
         * @returns {object} Validation result
         */
        validateFormula: function(formula) {
            if (!formula || typeof formula !== 'string') {
                return {
                    isValid: false,
                    sanitized: "",
                    errors: ["Formula must be a non-empty string"]
                };
            }

            const errors = [];
            let sanitized = formula.trim();

            // Remove dangerous patterns
            const dangerousPatterns = [
                /<script[^>]*>.*?<\/script>/gi,
                /javascript:/gi,
                /on\w+\s*=/gi,
                /eval\s*\(/gi,
                /new\s+Function/gi,
                /setTimeout\s*\(/gi,
                /setInterval\s*\(/gi,
                /document\./gi,
                /window\./gi,
                /\bthis\./gi
            ];

            dangerousPatterns.forEach(pattern => {
                if (pattern.test(sanitized)) {
                    errors.push("Formula contains potentially dangerous code");
                    sanitized = sanitized.replace(pattern, '');
                }
            });

            // Validate allowed mathematical functions
            const allowedFunctions = [
                'abs', 'acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'exp',
                'floor', 'log', 'max', 'min', 'pow', 'random', 'round', 'sin',
                'sqrt', 'tan', 'sum', 'avg', 'mean', 'median', 'std', 'var'
            ];

            const functionPattern = /\b([a-zA-Z_]\w*)\s*\(/g;
            let match;
            while ((match = functionPattern.exec(sanitized)) !== null) {
                if (!allowedFunctions.includes(match[1].toLowerCase())) {
                    errors.push(`Unknown or disallowed function: ${match[1]}`);
                }
            }

            // Check for balanced parentheses
            let parenCount = 0;
            for (let char of sanitized) {
                if (char === '(') parenCount++;
                if (char === ')') parenCount--;
                if (parenCount < 0) {
                    errors.push("Unbalanced parentheses");
                    break;
                }
            }
            if (parenCount !== 0) {
                errors.push("Unbalanced parentheses");
            }

            // Validate operators
            const invalidOpSequence = /[+\-*/%^]{2,}/;
            if (invalidOpSequence.test(sanitized)) {
                errors.push("Invalid operator sequence");
            }

            return {
                isValid: errors.length === 0,
                sanitized: sanitized,
                errors: errors,
                original: formula
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
         * Validate numerical precision
         * @param {number} value - Value to validate
         * @param {string} precision - Precision type
         * @returns {object} Validation result
         */
        validatePrecision: function(value, precision) {
            if (typeof value !== 'number') {
                return {
                    isValid: false,
                    error: "Value must be a number"
                };
            }

            if (!Number.isFinite(value)) {
                return {
                    isValid: false,
                    error: "Value must be finite"
                };
            }

            // Check for overflow based on precision
            const limits = {
                'single': { min: -3.4e38, max: 3.4e38 },
                'double': { min: Number.MIN_VALUE, max: Number.MAX_VALUE },
                'extended': { min: Number.MIN_VALUE, max: Number.MAX_VALUE }
            };

            const limit = limits[precision] || limits.double;
            
            if (value < limit.min || value > limit.max) {
                return {
                    isValid: false,
                    error: `Value exceeds ${precision} precision limits`
                };
            }

            return {
                isValid: true,
                value: value
            };
        },

        /**
         * Create secure WebSocket connection
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
                Log.info("Secure WebSocket connection established", secureUrl);
            });

            ws.addEventListener('error', function(error) {
                Log.error("WebSocket error", error);
            });

            ws.addEventListener('message', function(event) {
                try {
                    // Validate incoming data
                    const data = JSON.parse(event.data);
                    if (options.onMessage) {
                        options.onMessage(data);
                    }
                } catch (e) {
                    Log.error("Invalid WebSocket message format", event.data);
                }
            });

            return ws;
        },

        /**
         * Create secure EventSource
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
                            const validatedEvent = Object.assign({}, event, { data: data });
                            listener(validatedEvent);
                        } else {
                            listener(event);
                        }
                    } catch (e) {
                        Log.error("Invalid EventSource data format", event.data);
                    }
                };
                
                originalAddEventListener.call(this, type, secureListener, options);
            };

            return eventSource;
        },

        /**
         * Check user authorization for calculation operations
         * @param {string} operation - Operation type
         * @returns {Promise<boolean>} Authorization result
         */
        checkCalculationAuth: function(operation) {
            return new Promise(function(resolve) {
                // Mock authorization check - implement with actual auth service
                const allowedOperations = [
                    'execute', 'validate', 'optimize', 'analyze', 
                    'test', 'heal', 'export', 'visualize'
                ];
                
                if (allowedOperations.includes(operation.toLowerCase())) {
                    Log.info("Authorization granted for operation", operation);
                    resolve(true);
                } else {
                    Log.warning("Authorization denied for operation", operation);
                    resolve(false);
                }
            });
        },

        /**
         * Sanitize calculation result for display
         * @param {any} result - Calculation result
         * @returns {string} Sanitized result
         */
        sanitizeResult: function(result) {
            if (result === null || result === undefined) {
                return "N/A";
            }

            if (typeof result === 'string') {
                return this.escapeHTML(result);
            }

            if (typeof result === 'object') {
                try {
                    return this.escapeHTML(JSON.stringify(result, null, 2));
                } catch (e) {
                    return "Invalid result format";
                }
            }

            return this.escapeHTML(String(result));
        }
    };
});