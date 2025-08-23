sap.ui.define([], function() {
    "use strict";

    /**
     * @namespace a2a.network.agent6.ext.utils.SecurityUtils
     * @description Security utilities for Agent 6 - Quality Control Manager.
     * Provides comprehensive security features for quality assessment validation,
     * threshold management, routing decision security, and audit logging.
     */
    var SecurityUtils = {
        
        /**
         * @function escapeHTML
         * @description Escapes HTML entities to prevent XSS attacks in quality reports
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
         * @function sanitizeQualityData
         * @description Sanitizes quality control data to prevent injection attacks
         * @param {string} data - Quality data to sanitize
         * @returns {string} Sanitized data
         * @public
         */
        sanitizeQualityData: function(data) {
            if (!data || typeof data !== "string") return "";
            
            // Remove potential script tags
            data = data.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, "");
            data = data.replace(/on\w+\s*=\s*["'][^"']*["']/gi, "");
            data = data.replace(/javascript:/gi, "");
            
            // Remove SQL injection attempts
            data = data.replace(/(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)/gi, "");
            
            // Limit string length to prevent DoS
            if (data.length > 5000) {
                data = data.substring(0, 5000);
            }
            
            return data.trim();
        },
        
        /**
         * @function validateQualityScore
         * @description Validates quality score input to prevent manipulation
         * @param {number} score - Quality score to validate
         * @returns {boolean} True if valid, false otherwise
         * @public
         */
        validateQualityScore: function(score) {
            if (typeof score !== "number" || isNaN(score)) {
                return false;
            }
            
            // Quality scores must be between 0 and 100
            return score >= 0 && score <= 100;
        },
        
        /**
         * @function validateQualityThreshold
         * @description Validates quality threshold configuration
         * @param {Object} threshold - Threshold configuration
         * @returns {boolean} True if valid, false otherwise
         * @public
         */
        validateQualityThreshold: function(threshold) {
            if (!threshold || typeof threshold !== "object") {
                return false;
            }
            
            // Validate threshold values
            if (threshold.minQualityScore !== undefined) {
                if (!this.validateQualityScore(threshold.minQualityScore)) {
                    return false;
                }
            }
            
            if (threshold.maxDefects !== undefined) {
                if (typeof threshold.maxDefects !== "number" || threshold.maxDefects < 0) {
                    return false;
                }
            }
            
            if (threshold.maxWarnings !== undefined) {
                if (typeof threshold.maxWarnings !== "number" || threshold.maxWarnings < 0) {
                    return false;
                }
            }
            
            return true;
        },
        
        /**
         * @function validateRoutingDecision
         * @description Validates routing decisions to prevent unauthorized routing
         * @param {Object} decision - Routing decision
         * @returns {boolean} True if valid, false otherwise
         * @public
         */
        validateRoutingDecision: function(decision) {
            if (!decision || typeof decision !== "object") {
                return false;
            }
            
            // Validate agent ID format
            if (decision.agentId && !/^agent\d{1,2}$/.test(decision.agentId)) {
                return false;
            }
            
            // Validate routing reason
            var allowedReasons = ["quality_passed", "quality_failed", "manual_override", "auto_routing", "escalation"];
            if (decision.reason && !allowedReasons.includes(decision.reason)) {
                return false;
            }
            
            // Validate priority
            var allowedPriorities = ["low", "medium", "high", "critical"];
            if (decision.priority && !allowedPriorities.includes(decision.priority)) {
                return false;
            }
            
            return true;
        },
        
        /**
         * @function sanitizeQualityMetrics
         * @description Sanitizes quality metrics data
         * @param {Object} metrics - Metrics data to sanitize
         * @returns {Object} Sanitized metrics
         * @public
         */
        sanitizeQualityMetrics: function(metrics) {
            if (!metrics || typeof metrics !== "object") {
                return {};
            }
            
            var sanitized = {};
            
            // Sanitize each metric
            for (var key in metrics) {
                if (metrics.hasOwnProperty(key)) {
                    var value = metrics[key];
                    
                    if (typeof value === "string") {
                        sanitized[key] = this.sanitizeQualityData(value);
                    } else if (typeof value === "number") {
                        // Ensure numbers are within reasonable bounds
                        sanitized[key] = Math.min(Math.max(value, -999999), 999999);
                    } else if (typeof value === "boolean") {
                        sanitized[key] = value;
                    }
                    // Ignore functions and complex objects
                }
            }
            
            return sanitized;
        },
        
        /**
         * @function createSecureWebSocket
         * @description Creates a secure WebSocket connection for quality monitoring
         * @param {string} url - WebSocket URL
         * @param {Object} handlers - Event handlers
         * @returns {WebSocket|null} Secure WebSocket instance or null
         * @public
         */
        createSecureWebSocket: function(url, handlers) {
            if (!this.validateWebSocketUrl(url)) {
                console.error("Invalid WebSocket URL");
                return null;
            }
            
            try {
                var ws = new WebSocket(url);
                
                // Add security headers
                ws.addEventListener('open', function() {
                    // Send authentication token if available
                    var token = this._getAuthToken();
                    if (token) {
                        ws.send(JSON.stringify({
                            type: 'auth',
                            token: token
                        }));
                    }
                }.bind(this));
                
                // Wrap message handler with sanitization
                if (handlers.onmessage) {
                    var originalHandler = handlers.onmessage;
                    ws.onmessage = function(event) {
                        try {
                            // Sanitize incoming data
                            var data = JSON.parse(event.data);
                            if (data.metrics) {
                                data.metrics = this.sanitizeQualityMetrics(data.metrics);
                            }
                            event.data = JSON.stringify(data);
                            originalHandler.call(this, event);
                        } catch (e) {
                            console.error("Error processing WebSocket message:", e);
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
                console.error("Failed to create WebSocket:", e);
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
        validateWebSocketUrl: function(url) {
            if (!url || typeof url !== "string") return false;
            
            // Must use secure WebSocket protocol
            if (!url.startsWith("wss://")) {
                console.warn("WebSocket URL must use secure protocol (wss://)");
                return false;
            }
            
            // Validate URL format
            try {
                var urlObj = new URL(url);
                
                // Check for localhost or allowed domains
                var allowedHosts = ["localhost", "127.0.0.1", window.location.hostname];
                if (!allowedHosts.includes(urlObj.hostname)) {
                    return false;
                }
                
                return true;
                
            } catch (e) {
                return false;
            }
        },
        
        /**
         * @function secureCallFunction
         * @description Securely calls OData functions with CSRF protection
         * @param {sap.ui.model.odata.v4.ODataModel} oModel - OData model
         * @param {string} sFunctionName - Function name
         * @param {Object} mParameters - Function parameters
         * @returns {Promise} Promise resolving to function result
         * @public
         */
        secureCallFunction: function(oModel, sFunctionName, mParameters) {
            if (!oModel || !sFunctionName) {
                return Promise.reject(new Error("Invalid function call parameters"));
            }
            
            // Add CSRF token to headers
            var mHeaders = mParameters.headers || {};
            mHeaders["X-CSRF-Token"] = this._getCSRFToken();
            mHeaders["X-Requested-With"] = "XMLHttpRequest";
            
            // Add security headers
            mHeaders["X-Content-Type-Options"] = "nosniff";
            mHeaders["X-Frame-Options"] = "DENY";
            mHeaders["X-XSS-Protection"] = "1; mode=block";
            
            mParameters.headers = mHeaders;
            
            // Validate parameters
            if (mParameters.urlParameters) {
                for (var key in mParameters.urlParameters) {
                    var value = mParameters.urlParameters[key];
                    if (typeof value === "string") {
                        // Sanitize string parameters
                        mParameters.urlParameters[key] = this.sanitizeQualityData(value);
                    }
                }
            }
            
            return new Promise(function(resolve, reject) {
                oModel.callFunction(sFunctionName, {
                    ...mParameters,
                    success: function(data) {
                        resolve(data);
                    },
                    error: function(error) {
                        reject(error);
                    }
                });
            });
        },
        
        /**
         * @function checkQualityAuth
         * @description Checks if user has authorization for quality operations
         * @param {string} operation - Operation to check
         * @param {Object} context - Operation context
         * @returns {boolean} True if authorized, false otherwise
         * @public
         */
        checkQualityAuth: function(operation, context) {
            // Check operation-specific permissions
            var requiredPermissions = {
                "CreateQualityTask": ["quality.task.create", "quality.control"],
                "StartAssessment": ["quality.assessment.start", "quality.control"],
                "ApproveQuality": ["quality.approve", "quality.manager"],
                "RejectQuality": ["quality.reject", "quality.control"],
                "OverrideThreshold": ["quality.threshold.override", "quality.admin"],
                "RouteAgent": ["quality.routing.control", "quality.control"],
                "ViewMetrics": ["quality.metrics.view", "quality.viewer"],
                "ExportReport": ["quality.report.export", "quality.viewer"]
            };
            
            var permissions = requiredPermissions[operation];
            if (!permissions) {
                // Unknown operation, deny by default
                return false;
            }
            
            // Check user permissions (this would integrate with your auth system)
            var userPermissions = this._getUserPermissions();
            for (var i = 0; i < permissions.length; i++) {
                if (!userPermissions.includes(permissions[i])) {
                    return false;
                }
            }
            
            // Additional context-based checks
            if (context && context.severity === "critical") {
                if (!userPermissions.includes("quality.critical.handle")) {
                    return false;
                }
            }
            
            return true;
        },
        
        /**
         * @function logQualityAudit
         * @description Logs quality control operations for audit trail
         * @param {string} operation - Operation performed
         * @param {Object} details - Operation details
         * @public
         */
        logQualityAudit: function(operation, details) {
            var auditEntry = {
                timestamp: new Date().toISOString(),
                operation: operation,
                user: this._getCurrentUser(),
                details: this.sanitizeQualityMetrics(details),
                sessionId: this._getSessionId()
            };
            
            // Send to audit log (implement based on your logging system)
            this._sendAuditLog(auditEntry);
        },
        
        /**
         * @function validateBatchOperation
         * @description Validates batch quality operations to prevent abuse
         * @param {Array} items - Items to process in batch
         * @returns {boolean} True if valid, false otherwise
         * @public
         */
        validateBatchOperation: function(items) {
            if (!Array.isArray(items)) {
                return false;
            }
            
            // Limit batch size to prevent DoS
            if (items.length > 100) {
                console.warn("Batch size exceeds maximum allowed (100)");
                return false;
            }
            
            // Validate each item
            for (var i = 0; i < items.length; i++) {
                if (!items[i] || typeof items[i] !== "object") {
                    return false;
                }
            }
            
            return true;
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
         * @function _getAuthToken
         * @description Gets authentication token for WebSocket
         * @returns {string|null} Auth token or null
         * @private
         */
        _getAuthToken: function() {
            // Get from session storage
            var token = sessionStorage.getItem("quality-auth-token");
            if (token) {
                return token;
            }
            
            // Get from meta tag
            var metaToken = document.querySelector('meta[name="auth-token"]');
            if (metaToken) {
                return metaToken.getAttribute("content");
            }
            
            return null;
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
            return ["quality.task.create", "quality.control", "quality.metrics.view"];
        },
        
        /**
         * @function _getCurrentUser
         * @description Gets current user identifier
         * @returns {string} User identifier
         * @private
         */
        _getCurrentUser: function() {
            // This would integrate with your auth system
            return sessionStorage.getItem("user-id") || "anonymous";
        },
        
        /**
         * @function _getSessionId
         * @description Gets current session ID
         * @returns {string} Session ID
         * @private
         */
        _getSessionId: function() {
            var sessionId = sessionStorage.getItem("session-id");
            if (!sessionId) {
                sessionId = this._generateCSRFToken();
                sessionStorage.setItem("session-id", sessionId);
            }
            return sessionId;
        },
        
        /**
         * @function _sendAuditLog
         * @description Sends audit log entry to server
         * @param {Object} entry - Audit entry
         * @private
         */
        _sendAuditLog: function(entry) {
            // Implement based on your logging infrastructure
            // For now, just log to console in development
            if (window.location.hostname === "localhost") {
                console.log("Quality Audit:", entry);
            }
            
            // In production, send to audit service
            // Example: this._postToAuditService("/api/audit/quality", entry);
        }
    };
    
    return SecurityUtils;
});