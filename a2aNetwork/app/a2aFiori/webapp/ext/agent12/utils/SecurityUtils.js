sap.ui.define([
    "sap/base/Log",
    "sap/m/MessageToast"
], function (Log, MessageToast) {
    "use strict";

    return {
        /**
         * Validates catalog entry data for security vulnerabilities
         * @param {object} entryData - Catalog entry data to validate
         * @returns {object} Validation result with security checks
         */
        validateCatalogEntry: function (entryData) {
            const validation = {
                isValid: true,
                errors: [],
                warnings: [],
                sanitized: {}
            };

            if (!entryData) {
                validation.isValid = false;
                validation.errors.push("Entry data is required");
                return validation;
            }

            // Validate and sanitize entry name
            if (entryData.entryName) {
                validation.sanitized.entryName = this.sanitizeCatalogData(entryData.entryName);
                if (this._containsScriptTags(entryData.entryName)) {
                    validation.errors.push("Entry name contains potentially unsafe content");
                    validation.isValid = false;
                }
            }

            // Validate and sanitize description
            if (entryData.description) {
                validation.sanitized.description = this.sanitizeCatalogData(entryData.description);
                if (this._containsScriptTags(entryData.description)) {
                    validation.errors.push("Description contains potentially unsafe content");
                    validation.isValid = false;
                }
            }

            // Validate resource URL
            if (entryData.resourceUrl) {
                const urlValidation = this.validateResourceURL(entryData.resourceUrl);
                if (!urlValidation.isValid) {
                    validation.errors.push(`Invalid resource URL: ${urlValidation.error}`);
                    validation.isValid = false;
                } else {
                    validation.sanitized.resourceUrl = urlValidation.sanitizedUrl;
                }
            }

            // Validate metadata
            if (entryData.metadata) {
                const metadataValidation = this.validateMetadata(entryData.metadata);
                if (!metadataValidation.isValid) {
                    validation.errors.push("Invalid metadata format");
                    validation.isValid = false;
                } else {
                    validation.sanitized.metadata = metadataValidation.sanitized;
                }
            }

            // Validate tags and keywords
            ['tags', 'keywords'].forEach(field => {
                if (entryData[field]) {
                    validation.sanitized[field] = this.sanitizeSearchQuery(entryData[field]);
                    if (this._containsScriptTags(entryData[field])) {
                        validation.errors.push(`${field} contains potentially unsafe content`);
                        validation.isValid = false;
                    }
                }
            });

            // Validate API endpoints
            ['apiEndpoint', 'documentationUrl', 'healthCheckUrl', 'swaggerUrl'].forEach(field => {
                if (entryData[field]) {
                    const urlValidation = this.validateResourceURL(entryData[field]);
                    if (!urlValidation.isValid) {
                        validation.warnings.push(`Invalid ${field}: ${urlValidation.error}`);
                    } else {
                        validation.sanitized[field] = urlValidation.sanitizedUrl;
                    }
                }
            });

            return validation;
        },

        /**
         * Validates resource URLs for security
         * @param {string} url - URL to validate
         * @returns {object} Validation result
         */
        validateResourceURL: function (url) {
            const validation = {
                isValid: false,
                error: "",
                sanitizedUrl: ""
            };

            if (!url || typeof url !== 'string') {
                validation.error = "URL is required and must be a string";
                return validation;
            }

            // Remove potentially dangerous characters
            const sanitized = url.trim().replace(/[<>'"]/g, '');
            
            try {
                const urlObj = new URL(sanitized);
                
                // Check for allowed protocols
                const allowedProtocols = ['http:', 'https:'];
                if (!allowedProtocols.includes(urlObj.protocol)) {
                    validation.error = `Protocol ${urlObj.protocol} not allowed`;
                    return validation;
                }

                // Check for suspicious patterns
                const suspiciousPatterns = [
                    /javascript:/gi,
                    /data:/gi,
                    /vbscript:/gi,
                    /<script/gi,
                    /on\w+\s*=/gi
                ];

                if (suspiciousPatterns.some(pattern => pattern.test(sanitized))) {
                    validation.error = "URL contains suspicious content";
                    return validation;
                }

                // Check for localhost/internal IPs in production
                const hostname = urlObj.hostname.toLowerCase();
                const internalPatterns = [
                    /^localhost$/,
                    /^127\./,
                    /^192\.168\./,
                    /^10\./,
                    /^172\.(1[6-9]|2[0-9]|3[0-1])\./
                ];

                if (this._isProduction() && internalPatterns.some(pattern => pattern.test(hostname))) {
                    validation.error = "Internal URLs not allowed in production";
                    return validation;
                }

                validation.isValid = true;
                validation.sanitizedUrl = sanitized;
                
            } catch (error) {
                validation.error = `Invalid URL format: ${error.message}`;
            }

            return validation;
        },

        /**
         * Validates metadata for injection vulnerabilities
         * @param {object|string} metadata - Metadata to validate
         * @returns {object} Validation result
         */
        validateMetadata: function (metadata) {
            const validation = {
                isValid: true,
                sanitized: null
            };

            if (typeof metadata === 'string') {
                try {
                    const parsed = JSON.parse(metadata);
                    validation.sanitized = this._sanitizeMetadataObject(parsed);
                } catch (error) {
                    validation.isValid = false;
                }
            } else if (typeof metadata === 'object' && metadata !== null) {
                validation.sanitized = this._sanitizeMetadataObject(metadata);
            } else {
                validation.isValid = false;
            }

            return validation;
        },

        /**
         * Sanitizes search queries for injection protection
         * @param {string} query - Search query to sanitize
         * @returns {string} Sanitized query
         */
        sanitizeSearchQuery: function (query) {
            if (!query || typeof query !== 'string') {
                return '';
            }

            return query
                // Remove script tags and event handlers
                .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
                .replace(/on\w+\s*=/gi, '')
                .replace(/javascript:/gi, '')
                // Remove dangerous characters
                .replace(/[<>'"]/g, '')
                // Limit length
                .substring(0, 500)
                .trim();
        },

        /**
         * Sanitizes catalog data for display
         * @param {string} data - Data to sanitize
         * @returns {string} Sanitized data
         */
        sanitizeCatalogData: function (data) {
            if (!data || typeof data !== 'string') {
                return '';
            }

            return data
                // Remove script tags completely
                .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
                // Remove event handlers
                .replace(/on\w+\s*=\s*["'][^"']*["']/gi, '')
                // Remove javascript: protocol
                .replace(/javascript:/gi, '')
                // Escape HTML entities
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#x27;')
                .replace(/\//g, '&#x2F;');
        },

        /**
         * Escapes HTML content for safe display
         * @param {string} html - HTML content to escape
         * @returns {string} Escaped HTML
         */
        escapeHTML: function (html) {
            if (!html) return '';
            
            const div = document.createElement('div');
            div.textContent = html;
            return div.innerHTML;
        },

        /**
         * Makes secure OData function calls with CSRF protection
         * @param {sap.ui.model.odata.v2.ODataModel} model - OData model
         * @param {string} functionName - Function name to call
         * @param {object} parameters - Function parameters
         * @returns {Promise} Promise resolving to function result
         */
        secureCallFunction: function (model, functionName, parameters) {
            return new Promise((resolve, reject) => {
                // First, refresh security token
                model.refreshSecurityToken((tokenData) => {
                    // Add CSRF token to headers if not already present
                    const headers = parameters.headers || {};
                    if (!headers['X-CSRF-Token'] && tokenData) {
                        headers['X-CSRF-Token'] = tokenData;
                    }

                    // Enhanced parameters with security
                    const secureParams = {
                        ...parameters,
                        headers: headers,
                        success: (data) => {
                            this.logSecureOperation(functionName, 'SUCCESS');
                            if (parameters.success) {
                                parameters.success(data);
                            }
                            resolve(data);
                        },
                        error: (error) => {
                            this.logSecureOperation(functionName, 'ERROR', error);
                            if (parameters.error) {
                                parameters.error(error);
                            }
                            reject(error);
                        }
                    };

                    model.callFunction(functionName, secureParams);
                }, (error) => {
                    this.logSecureOperation(functionName, 'TOKEN_ERROR', error);
                    reject(new Error('Failed to obtain CSRF token'));
                });
            });
        },

        /**
         * Checks authorization for catalog operations
         * @param {string} operation - Operation to check
         * @param {object} context - Operation context
         * @returns {boolean} True if authorized
         */
        checkCatalogAuth: function (operation, context) {
            // Check if user has required permissions
            const user = this._getCurrentUser();
            if (!user) {
                MessageToast.show("Authentication required");
                return false;
            }

            // Check operation-specific permissions
            const requiredPermissions = this._getCatalogPermissions(operation);
            const hasPermission = requiredPermissions.every(permission => 
                this._userHasPermission(user, permission)
            );

            if (!hasPermission) {
                MessageToast.show("Insufficient permissions for this operation");
                this.logSecureOperation(operation, 'UNAUTHORIZED', { user: user.id });
            }

            return hasPermission;
        },

        /**
         * Validates discovered resources for security
         * @param {object} resource - Discovered resource
         * @returns {object} Validation result
         */
        validateDiscoveredResource: function (resource) {
            const validation = {
                isValid: true,
                sanitized: {},
                warnings: []
            };

            if (!resource) {
                validation.isValid = false;
                return validation;
            }

            // Validate resource URL
            if (resource.url) {
                const urlValidation = this.validateResourceURL(resource.url);
                if (!urlValidation.isValid) {
                    validation.isValid = false;
                    validation.warnings.push(`Invalid resource URL: ${urlValidation.error}`);
                } else {
                    validation.sanitized.url = urlValidation.sanitizedUrl;
                }
            }

            // Validate resource metadata
            if (resource.metadata) {
                const metadataValidation = this.validateMetadata(resource.metadata);
                if (metadataValidation.isValid) {
                    validation.sanitized.metadata = metadataValidation.sanitized;
                } else {
                    validation.warnings.push("Invalid metadata in discovered resource");
                }
            }

            // Check for suspicious resource patterns
            const suspiciousPatterns = [
                /malware/gi,
                /virus/gi,
                /exploit/gi,
                /backdoor/gi,
                /keylog/gi
            ];

            const resourceString = JSON.stringify(resource);
            if (suspiciousPatterns.some(pattern => pattern.test(resourceString))) {
                validation.isValid = false;
                validation.warnings.push("Resource appears suspicious");
            }

            return validation;
        },

        /**
         * Creates secure WebSocket connection for catalog updates
         * @param {string} url - WebSocket URL
         * @param {object} options - Connection options
         * @returns {WebSocket|null} Secure WebSocket connection
         */
        createSecureWebSocket: function (url, options) {
            try {
                // Ensure secure protocol
                const secureUrl = url.replace(/^ws:\/\//, 'wss://').replace(/^http:\/\//, 'https://');
                
                // Validate URL
                const urlValidation = this.validateResourceURL(secureUrl);
                if (!urlValidation.isValid) {
                    Log.error("Invalid WebSocket URL", urlValidation.error);
                    return null;
                }

                const ws = new WebSocket(urlValidation.sanitizedUrl);
                
                // Add security event handlers
                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        const sanitizedData = this._sanitizeWebSocketData(data);
                        
                        if (options.onmessage) {
                            options.onmessage({
                                ...event,
                                data: JSON.stringify(sanitizedData)
                            });
                        }
                    } catch (error) {
                        Log.error("Invalid WebSocket message format", error);
                    }
                };

                ws.onerror = (error) => {
                    this.logSecureOperation('WEBSOCKET_ERROR', 'ERROR', error);
                    if (options.onerror) {
                        options.onerror(error);
                    }
                };

                return ws;
                
            } catch (error) {
                Log.error("Failed to create secure WebSocket", error);
                return null;
            }
        },

        /**
         * Creates secure EventSource for catalog streams
         * @param {string} url - EventSource URL
         * @param {object} options - Connection options
         * @returns {EventSource|null} Secure EventSource connection
         */
        createSecureEventSource: function (url, options) {
            try {
                // Ensure secure protocol
                const secureUrl = url.replace(/^http:\/\//, 'https://');
                
                // Validate URL
                const urlValidation = this.validateResourceURL(secureUrl);
                if (!urlValidation.isValid) {
                    Log.error("Invalid EventSource URL", urlValidation.error);
                    return null;
                }

                const eventSource = new EventSource(urlValidation.sanitizedUrl);
                
                // Add security event handlers
                const originalAddEventListener = eventSource.addEventListener;
                eventSource.addEventListener = function(type, listener, options) {
                    const secureListener = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            const sanitizedData = this._sanitizeWebSocketData(data);
                            
                            listener({
                                ...event,
                                data: JSON.stringify(sanitizedData)
                            });
                        } catch (error) {
                            Log.error("Invalid EventSource message format", error);
                        }
                    }.bind(this);
                    
                    originalAddEventListener.call(eventSource, type, secureListener, options);
                }.bind(this);

                return eventSource;
                
            } catch (error) {
                Log.error("Failed to create secure EventSource", error);
                return null;
            }
        },

        /**
         * Logs secure operations for audit trail
         * @param {string} operation - Operation name
         * @param {string} status - Operation status
         * @param {object} details - Additional details
         */
        logSecureOperation: function (operation, status, details) {
            const logEntry = {
                timestamp: new Date().toISOString(),
                operation: operation,
                status: status,
                user: this._getCurrentUser()?.id || 'anonymous',
                details: details || {}
            };

            // Log to console in development, send to audit service in production
            if (this._isProduction()) {
                // In production, send to audit service
                this._sendToAuditService(logEntry);
            } else {
                Log.info("Security Operation", logEntry);
            }
        },

        // Private helper methods
        _containsScriptTags: function (str) {
            if (!str || typeof str !== 'string') return false;
            
            const scriptPatterns = [
                /<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi,
                /javascript:/gi,
                /on\w+\s*=/gi,
                /<iframe/gi,
                /<object/gi,
                /<embed/gi,
                /<form/gi
            ];
            
            return scriptPatterns.some(pattern => pattern.test(str));
        },

        _sanitizeMetadataObject: function (obj) {
            if (!obj || typeof obj !== 'object') {
                return {};
            }

            const sanitized = {};
            
            for (const [key, value] of Object.entries(obj)) {
                const sanitizedKey = this.sanitizeCatalogData(key);
                
                if (typeof value === 'string') {
                    sanitized[sanitizedKey] = this.sanitizeCatalogData(value);
                } else if (typeof value === 'object' && value !== null) {
                    sanitized[sanitizedKey] = this._sanitizeMetadataObject(value);
                } else {
                    sanitized[sanitizedKey] = value;
                }
            }

            return sanitized;
        },

        _sanitizeWebSocketData: function (data) {
            if (!data || typeof data !== 'object') {
                return data;
            }

            const sanitized = {};
            
            for (const [key, value] of Object.entries(data)) {
                if (typeof value === 'string') {
                    sanitized[key] = this.sanitizeCatalogData(value);
                } else if (typeof value === 'object' && value !== null) {
                    sanitized[key] = this._sanitizeWebSocketData(value);
                } else {
                    sanitized[key] = value;
                }
            }

            return sanitized;
        },

        _getCurrentUser: function () {
            // Mock user detection - implement actual user detection
            return {
                id: 'current-user',
                permissions: ['catalog:read', 'catalog:write', 'catalog:admin']
            };
        },

        _getCatalogPermissions: function (operation) {
            const permissionMap = {
                'RegisterResource': ['catalog:write'],
                'ValidateEntry': ['catalog:write'],
                'PublishEntry': ['catalog:admin'],
                'IndexResource': ['catalog:write'],
                'DiscoverDependencies': ['catalog:read'],
                'SyncRegistryEntry': ['catalog:admin'],
                'StartResourceDiscovery': ['catalog:admin'],
                'ValidateCatalogEntries': ['catalog:write'],
                'PublishCatalogEntries': ['catalog:admin']
            };
            
            return permissionMap[operation] || ['catalog:read'];
        },

        _userHasPermission: function (user, permission) {
            return user.permissions && user.permissions.includes(permission);
        },

        _isProduction: function () {
            // Detect production environment
            return window.location.hostname !== 'localhost' && 
                   window.location.hostname !== '127.0.0.1' &&
                   !window.location.hostname.startsWith('192.168.');
        },

        _sendToAuditService: function (logEntry) {
            // In production, implement actual audit service integration
            console.log("AUDIT:", logEntry);
        }
    };
});