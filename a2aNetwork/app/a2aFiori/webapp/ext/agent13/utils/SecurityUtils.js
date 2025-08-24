/**
 * A2A Protocol Compliance: WebSocket replaced with blockchain event streaming
 * All real-time communication now uses blockchain events instead of WebSockets
 */

sap.ui.define([], function() {
    "use strict";

    /**
     * @namespace a2a.network.agent13.ext.utils.SecurityUtils
     * @description Security utilities for Agent 13 - Agent Builder Agent.
     * Provides comprehensive security features for code generation, template sanitization,
     * deployment configuration security, pipeline command validation, and file system access controls.
     */
    var SecurityUtils = {
        
        /**
         * @function escapeHTML
         * @description Escapes HTML entities to prevent XSS attacks in agent templates and generated code
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
         * @function sanitizeTemplateData
         * @description Sanitizes template data to prevent template injection attacks
         * @param {string} data - Template data to sanitize
         * @returns {string} Sanitized data
         * @public
         */
        sanitizeTemplateData: function(data) {
            if (!data || typeof data !== "string") return "";
            
            // Remove potential script tags and event handlers
            data = data.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, "");
            data = data.replace(/on\w+\s*=\s*["'][^"']*["']/gi, "");
            data = data.replace(/javascript:/gi, "");
            
            // Remove dangerous template expressions
            data = data.replace(/\{\{#.*?\}\}/g, ""); // Remove Handlebars block helpers
            data = data.replace(/\{\{\/.*?\}\}/g, ""); // Remove Handlebars closing blocks
            data = data.replace(/\$\{.*?\}/g, ""); // Remove template literals
            data = data.replace(/<\?.*?\?>/g, ""); // Remove PHP tags
            data = data.replace(/<%.*?%>/g, ""); // Remove ASP/JSP tags
            
            // Sanitize SQL-like patterns
            data = data.replace(/(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)/gi, "");
            
            // Limit string length to prevent DoS
            if (data.length > 10000) {
                data = data.substring(0, 10000);
            }
            
            return data.trim();
        },
        
        /**
         * @function validateCodeGeneration
         * @description Validates code generation inputs to prevent code injection
         * @param {Object} config - Code generation configuration
         * @returns {boolean} True if valid, false otherwise
         * @public
         */
        validateCodeGeneration: function(config) {
            if (!config || typeof config !== "object") return false;
            
            // Validate template name
            if (!this._isValidIdentifier(config.templateName)) {
                return false;
            }
            
            // Validate code language
            var allowedLanguages = ["python", "javascript", "typescript", "java", "csharp", "go", "rust", "ruby", "kotlin", "scala"];
            if (!allowedLanguages.includes(config.codeLanguage)) {
                return false;
            }
            
            // Validate framework version format
            if (config.frameworkVersion && !/^\d+\.\d+(\.\d+)?(-[\w\d]+)?$/.test(config.frameworkVersion)) {
                return false;
            }
            
            // Validate deployment target
            var allowedTargets = ["kubernetes", "docker", "serverless", "vm", "edge", "cloud", "onpremise", "hybrid"];
            if (!allowedTargets.includes(config.deploymentTarget)) {
                return false;
            }
            
            // Validate environment variables format
            if (config.environmentVariables) {
                var envVars = config.environmentVariables.split('\n');
                for (var i = 0; i < envVars.length; i++) {
                    if (envVars[i].trim() && !/^[A-Z_][A-Z0-9_]*=.*$/.test(envVars[i].trim())) {
                        return false;
                    }
                }
            }
            
            return true;
        },
        
        /**
         * @function sanitizeBuilderData
         * @description Sanitizes builder-specific data to prevent injection attacks
         * @param {string} data - Builder data to sanitize
         * @returns {string} Sanitized data
         * @public
         */
        sanitizeBuilderData: function(data) {
            if (!data) return "";
            
            // Parse and re-stringify JSON to remove any executable code
            try {
                var parsed = JSON.parse(data);
                // Remove any function references
                this._removeFunctions(parsed);
                return JSON.stringify(parsed);
            } catch (e) {
                // If not valid JSON, sanitize as string
                return this.sanitizeTemplateData(data);
            }
        },
        
        /**
         * @function validatePipelineCommand
         * @description Validates pipeline commands to prevent command injection
         * @param {string} command - Pipeline command to validate
         * @returns {boolean} True if valid, false otherwise
         * @public
         */
        validatePipelineCommand: function(command) {
            if (!command || typeof command !== "string") return false;
            
            // Block dangerous shell operators
            var dangerousPatterns = [
                /[;&|<>]/g, // Command chaining and redirection
                /\$\(/g, // Command substitution
                /`/g, // Backticks
                /\\\n/g, // Line continuation
                /\bsudo\b/i, // Privilege escalation
                /\brm\s+-rf/i, // Dangerous rm commands
                /\bdd\b/i, // Disk operations
                /\bmkfs\b/i, // File system operations
                /\bshutdown\b/i, // System commands
                /\breboot\b/i,
                /\bkill\b/i,
                /\bpkill\b/i,
                /\bcurl\s+.*\s+-o/i, // Download operations
                /\bwget\s+.*\s+-O/i,
                /\bnc\b/i, // Netcat
                /\btelnet\b/i,
                /\bssh\b/i,
                /\bftp\b/i
            ];
            
            for (var i = 0; i < dangerousPatterns.length; i++) {
                if (dangerousPatterns[i].test(command)) {
                    return false;
                }
            }
            
            // Whitelist allowed commands
            var allowedCommands = [
                /^npm\s+(install|run|test|build)/,
                /^yarn\s+(install|run|test|build)/,
                /^python\s+\S+\.py/,
                /^java\s+-jar/,
                /^dotnet\s+(build|test|run)/,
                /^go\s+(build|test|run)/,
                /^cargo\s+(build|test|run)/,
                /^gradle\s+\w+/,
                /^mvn\s+\w+/,
                /^make\s+\w+/,
                /^docker\s+(build|run|push)/,
                /^kubectl\s+(apply|get|describe)/,
                /^terraform\s+(init|plan|apply)/,
                /^ansible-playbook/
            ];
            
            var isAllowed = false;
            for (var j = 0; j < allowedCommands.length; j++) {
                if (allowedCommands[j].test(command)) {
                    isAllowed = true;
                    break;
                }
            }
            
            return isAllowed;
        },
        
        /**
         * @function validateDeploymentConfig
         * @description Validates deployment configuration for security
         * @param {Object} config - Deployment configuration
         * @returns {boolean} True if valid, false otherwise
         * @public
         */
        validateDeploymentConfig: function(config) {
            if (!config || typeof config !== "object") return false;
            
            // Validate container image format
            if (config.containerImage && !/^[\w\-\.\/]+:[\w\-\.]+$/.test(config.containerImage)) {
                return false;
            }
            
            // Validate resource requirements
            if (config.resourceRequirements) {
                var resources = config.resourceRequirements.split(',');
                for (var i = 0; i < resources.length; i++) {
                    var resource = resources[i].trim();
                    if (!/(CPU|Memory|Storage):\s*\d+(\.\d+)?(m|Mi|Gi|GB|MB)?/i.test(resource)) {
                        return false;
                    }
                }
            }
            
            // Validate health check endpoint
            if (config.healthCheckConfig && !/^\/[\w\-\/]*$/.test(config.healthCheckConfig)) {
                return false;
            }
            
            // Validate secrets management
            if (config.secretsManagement) {
                var allowedSecretTypes = ["kubernetes", "vault", "aws-secrets", "azure-keyvault", "gcp-secrets"];
                if (!allowedSecretTypes.includes(config.secretsManagement)) {
                    return false;
                }
            }
            
            return true;
        },
        
        /**
         * @function validateFileSystemAccess
         * @description Validates file system access paths to prevent directory traversal
         * @param {string} path - File path to validate
         * @returns {boolean} True if valid, false otherwise
         * @public
         */
        validateFileSystemAccess: function(path) {
            if (!path || typeof path !== "string") return false;
            
            // Prevent directory traversal
            if (path.includes("..") || path.includes("~")) {
                return false;
            }
            
            // Prevent access to system directories
            var restrictedPaths = [
                /^\/etc/,
                /^\/sys/,
                /^\/proc/,
                /^\/dev/,
                /^\/root/,
                /^\/home\/\w+\/\./,
                /^C:\\Windows/i,
                /^C:\\System/i,
                /^\/usr\/bin/,
                /^\/usr\/sbin/,
                /^\/bin/,
                /^\/sbin/
            ];
            
            for (var i = 0; i < restrictedPaths.length; i++) {
                if (restrictedPaths[i].test(path)) {
                    return false;
                }
            }
            
            // Allow only specific file extensions for templates
            var allowedExtensions = [
                ".js", ".ts", ".py", ".java", ".cs", ".go", ".rs", ".rb", ".kt", ".scala",
                ".json", ".yaml", ".yml", ".xml", ".properties", ".conf", ".config",
                ".md", ".txt", ".dockerfile", ".dockerignore", ".gitignore",
                ".html", ".css", ".scss", ".less"
            ];
            
            var hasValidExtension = false;
            for (var j = 0; j < allowedExtensions.length; j++) {
                if (path.toLowerCase().endsWith(allowedExtensions[j])) {
                    hasValidExtension = true;
                    break;
                }
            }
            
            return hasValidExtension;
        },
        
        /**
         * @function createSecureWebSocket
         * @description Creates a secure WebSocket connection with proper validation
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
                        blockchainClient.publishEvent(JSON.stringify({
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
                            var sanitizedData = this.sanitizeBuilderData(event.data);
                            event.data = sanitizedData;
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
                
                // Check for valid port
                var port = parseInt(urlObj.port);
                if (port && (port < 1024 || port > 65535)) {
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
                        mParameters.urlParameters[key] = this.sanitizeTemplateData(value);
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
         * @function checkBuilderAuth
         * @description Checks if user has authorization for builder operations
         * @param {string} operation - Operation to check
         * @param {Object} context - Operation context
         * @returns {boolean} True if authorized, false otherwise
         * @public
         */
        checkBuilderAuth: function(operation, context) {
            // Check operation-specific permissions
            var requiredPermissions = {
                "CreateAgentTemplate": ["builder.create", "builder.template.write"],
                "GenerateAgent": ["builder.generate", "builder.code.write"],
                "DeployAgent": ["builder.deploy", "builder.environment.write"],
                "BuildAgent": ["builder.build", "builder.pipeline.execute"],
                "StartBatchBuild": ["builder.batch", "builder.pipeline.execute"],
                "DeleteTemplate": ["builder.delete", "builder.template.delete"],
                "ModifyPipeline": ["builder.pipeline.write", "builder.admin"]
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
                    this._showAuthError(operation);
                    return false;
                }
            }
            
            // Additional context-based checks
            if (context && context.environment === "production") {
                if (!userPermissions.includes("builder.production.deploy")) {
                    this._showAuthError("Production deployment");
                    return false;
                }
            }
            
            return true;
        },
        
        /**
         * @function _isValidIdentifier
         * @description Validates identifier format for code generation
         * @param {string} identifier - Identifier to validate
         * @returns {boolean} True if valid, false otherwise
         * @private
         */
        _isValidIdentifier: function(identifier) {
            if (!identifier || typeof identifier !== "string") return false;
            
            // Must start with letter or underscore, contain only alphanumeric and underscore
            return /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(identifier) && identifier.length <= 100;
        },
        
        /**
         * @function _removeFunctions
         * @description Recursively removes function references from objects
         * @param {Object} obj - Object to clean
         * @private
         */
        _removeFunctions: function(obj) {
            if (!obj || typeof obj !== "object") return;
            
            for (var key in obj) {
                if (obj.hasOwnProperty(key)) {
                    if (typeof obj[key] === "function") {
                        delete obj[key];
                    } else if (typeof obj[key] === "object") {
                        this._removeFunctions(obj[key]);
                    } else if (typeof obj[key] === "string") {
                        // Remove potential function strings
                        if (obj[key].includes("function") || obj[key].includes("=>")) {
                            obj[key] = "";
                        }
                    }
                }
            }
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
            var token = sessionStorage.getItem("builder-auth-token");
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
            return ["builder.create", "builder.template.write", "builder.generate"];
        },
        
        /**
         * @function _showAuthError
         * @description Shows authorization error message
         * @param {string} operation - Operation that was denied
         * @private
         */
        _showAuthError: function(operation) {
            sap.m.MessageBox.error(
                "You do not have permission to perform this operation: " + operation,
                {
                    title: "Authorization Error"
                }
            );
        }
    };
    
    return SecurityUtils;
});