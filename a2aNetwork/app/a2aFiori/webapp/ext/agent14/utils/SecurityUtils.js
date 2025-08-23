sap.ui.define([
    "sap/base/Log",
    "sap/m/MessageToast"
], function (Log, MessageToast) {
    "use strict";

    return {
        /**
         * Validates ML model path for security vulnerabilities
         * @param {string} modelPath - Model file path to validate
         * @returns {object} Validation result with security checks
         */
        validateModelPath: function (modelPath) {
            const validation = {
                isValid: true,
                errors: [],
                warnings: [],
                sanitizedPath: ""
            };

            if (!modelPath || typeof modelPath !== 'string') {
                validation.isValid = false;
                validation.errors.push("Model path is required and must be a string");
                return validation;
            }

            // Check for path traversal attempts
            if (modelPath.includes('..') || modelPath.includes('~')) {
                validation.isValid = false;
                validation.errors.push("Path traversal attempt detected");
                return validation;
            }

            // Check for absolute paths (security risk)
            if (modelPath.startsWith('/') || modelPath.match(/^[A-Za-z]:\\/)) {
                validation.isValid = false;
                validation.errors.push("Absolute paths not allowed for security reasons");
                return validation;
            }

            // Validate allowed model file extensions
            const allowedExtensions = ['.pth', '.pt', '.h5', '.pkl', '.joblib', '.onnx', '.pb', '.tflite', '.safetensors'];
            const hasValidExtension = allowedExtensions.some(ext => modelPath.toLowerCase().endsWith(ext));
            
            if (!hasValidExtension) {
                validation.isValid = false;
                validation.errors.push("Invalid model file extension. Allowed: " + allowedExtensions.join(', '));
                return validation;
            }

            // Check for suspicious patterns
            const suspiciousPatterns = [
                /\.(exe|dll|so|dylib)$/i,
                /__pycache__/i,
                /\.(sh|bat|ps1|cmd)$/i,
                /\.(js|py|rb|php)$/i
            ];

            if (suspiciousPatterns.some(pattern => pattern.test(modelPath))) {
                validation.isValid = false;
                validation.errors.push("Suspicious file pattern detected");
                return validation;
            }

            // Sanitize the path
            validation.sanitizedPath = modelPath
                .replace(/[<>:"|?*]/g, '')  // Remove invalid characters
                .replace(/\s+/g, '_')        // Replace spaces with underscores
                .replace(/[^\w\-./]/g, '');  // Keep only safe characters

            return validation;
        },

        /**
         * Validates training data for security vulnerabilities
         * @param {object} trainingData - Training data configuration
         * @returns {object} Validation result
         */
        validateTrainingData: function (trainingData) {
            const validation = {
                isValid: true,
                errors: [],
                warnings: [],
                sanitized: {}
            };

            if (!trainingData || typeof trainingData !== 'object') {
                validation.isValid = false;
                validation.errors.push("Training data configuration is required");
                return validation;
            }

            // Validate dataset path
            if (trainingData.datasetPath) {
                const pathValidation = this.validateModelPath(trainingData.datasetPath);
                if (!pathValidation.isValid) {
                    validation.isValid = false;
                    validation.errors.push("Invalid dataset path: " + pathValidation.errors.join(', '));
                } else {
                    validation.sanitized.datasetPath = pathValidation.sanitizedPath;
                }
            }

            // Validate data source URL
            if (trainingData.dataSourceUrl) {
                const urlValidation = this.validateDataSourceURL(trainingData.dataSourceUrl);
                if (!urlValidation.isValid) {
                    validation.errors.push("Invalid data source URL: " + urlValidation.error);
                    validation.isValid = false;
                } else {
                    validation.sanitized.dataSourceUrl = urlValidation.sanitizedUrl;
                }
            }

            // Validate batch size (prevent resource exhaustion)
            if (trainingData.batchSize) {
                const batchSize = parseInt(trainingData.batchSize);
                if (isNaN(batchSize) || batchSize < 1 || batchSize > 10000) {
                    validation.warnings.push("Batch size should be between 1 and 10000");
                }
                validation.sanitized.batchSize = Math.min(Math.max(batchSize || 32, 1), 10000);
            }

            // Validate augmentation settings
            if (trainingData.augmentation && typeof trainingData.augmentation === 'string') {
                // Check for code injection in augmentation config
                if (this._containsCodeInjection(trainingData.augmentation)) {
                    validation.isValid = false;
                    validation.errors.push("Augmentation config contains potentially unsafe content");
                }
            }

            return validation;
        },

        /**
         * Validates hyperparameters for security and sanity
         * @param {object} hyperparameters - Hyperparameter configuration
         * @returns {object} Validation result
         */
        validateHyperparameters: function (hyperparameters) {
            const validation = {
                isValid: true,
                errors: [],
                warnings: [],
                sanitized: {}
            };

            if (!hyperparameters || typeof hyperparameters !== 'object') {
                validation.isValid = false;
                validation.errors.push("Hyperparameters configuration is required");
                return validation;
            }

            // Validate learning rate
            if (hyperparameters.learningRate) {
                const lr = parseFloat(hyperparameters.learningRate);
                if (isNaN(lr) || lr <= 0 || lr > 10) {
                    validation.errors.push("Learning rate must be between 0 and 10");
                    validation.isValid = false;
                } else {
                    validation.sanitized.learningRate = lr;
                }
            }

            // Validate epochs
            if (hyperparameters.epochs) {
                const epochs = parseInt(hyperparameters.epochs);
                if (isNaN(epochs) || epochs < 1 || epochs > 10000) {
                    validation.errors.push("Epochs must be between 1 and 10000");
                    validation.isValid = false;
                } else {
                    validation.sanitized.epochs = epochs;
                }
            }

            // Validate optimizer config
            if (hyperparameters.optimizer) {
                const allowedOptimizers = ['adam', 'sgd', 'adamw', 'rmsprop', 'adagrad'];
                const optimizer = hyperparameters.optimizer.toLowerCase();
                
                if (!allowedOptimizers.includes(optimizer)) {
                    validation.errors.push("Invalid optimizer. Allowed: " + allowedOptimizers.join(', '));
                    validation.isValid = false;
                } else {
                    validation.sanitized.optimizer = optimizer;
                }
            }

            // Prevent code injection in custom configs
            if (hyperparameters.customConfig) {
                if (this._containsCodeInjection(JSON.stringify(hyperparameters.customConfig))) {
                    validation.isValid = false;
                    validation.errors.push("Custom config contains potentially unsafe content");
                }
            }

            return validation;
        },

        /**
         * Validates vector database queries for injection attacks
         * @param {string} query - Vector query to validate
         * @returns {object} Validation result
         */
        validateVectorQuery: function (query) {
            const validation = {
                isValid: true,
                error: "",
                sanitizedQuery: ""
            };

            if (!query || typeof query !== 'string') {
                validation.isValid = false;
                validation.error = "Query must be a non-empty string";
                return validation;
            }

            // Check for SQL injection patterns in vector queries
            const sqlInjectionPatterns = [
                /(\b(union|select|insert|update|delete|drop|create|alter)\b)/gi,
                /(\b(or|and)\b\s*\d+\s*=\s*\d+)/gi,
                /(--|\#|\/\*|\*\/)/g,
                /(\b(exec|execute|xp_|sp_)\b)/gi
            ];

            if (sqlInjectionPatterns.some(pattern => pattern.test(query))) {
                validation.isValid = false;
                validation.error = "Query contains potentially malicious patterns";
                return validation;
            }

            // Sanitize query
            validation.sanitizedQuery = query
                .replace(/[<>'"]/g, '') // Remove potentially dangerous characters
                .substring(0, 1000);     // Limit query length

            return validation;
        },

        /**
         * Securely saves model with validation
         * @param {object} modelData - Model data to save
         * @param {string} format - Save format
         * @returns {object} Validation result
         */
        secureModelSave: function (modelData, format) {
            const validation = {
                isValid: true,
                errors: [],
                warnings: []
            };

            // Validate format
            const allowedFormats = ['pytorch', 'tensorflow', 'onnx', 'safetensors'];
            if (!allowedFormats.includes(format)) {
                validation.isValid = false;
                validation.errors.push("Invalid save format. Allowed: " + allowedFormats.join(', '));
                return validation;
            }

            // Check model size (prevent resource exhaustion)
            const modelSize = JSON.stringify(modelData).length;
            const maxSize = 5 * 1024 * 1024 * 1024; // 5GB limit
            
            if (modelSize > maxSize) {
                validation.isValid = false;
                validation.errors.push("Model size exceeds maximum allowed (5GB)");
                return validation;
            }

            // Validate model structure
            if (!this._isValidModelStructure(modelData)) {
                validation.isValid = false;
                validation.errors.push("Invalid model structure detected");
                return validation;
            }

            return validation;
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
         * Validates data source URLs for embedding operations
         * @param {string} url - URL to validate
         * @returns {object} Validation result
         */
        validateDataSourceURL: function (url) {
            const validation = {
                isValid: false,
                error: "",
                sanitizedUrl: ""
            };

            if (!url || typeof url !== 'string') {
                validation.error = "URL is required and must be a string";
                return validation;
            }

            try {
                const urlObj = new URL(url);
                
                // Check for allowed protocols
                const allowedProtocols = ['https:', 's3:', 'gs:', 'azure:'];
                if (!allowedProtocols.includes(urlObj.protocol)) {
                    validation.error = `Protocol ${urlObj.protocol} not allowed. Use HTTPS or cloud storage protocols`;
                    return validation;
                }

                // Check for localhost/internal IPs in production
                const hostname = urlObj.hostname.toLowerCase();
                if (this._isProduction() && this._isInternalHost(hostname)) {
                    validation.error = "Internal URLs not allowed in production";
                    return validation;
                }

                validation.isValid = true;
                validation.sanitizedUrl = url;
                
            } catch (error) {
                validation.error = `Invalid URL format: ${error.message}`;
            }

            return validation;
        },

        /**
         * Checks authorization for embedding operations
         * @param {string} operation - Operation to check
         * @param {object} context - Operation context
         * @returns {boolean} True if authorized
         */
        checkEmbeddingAuth: function (operation, context) {
            // Check if user has required permissions
            const user = this._getCurrentUser();
            if (!user) {
                MessageToast.show("Authentication required");
                return false;
            }

            // Check operation-specific permissions
            const requiredPermissions = this._getEmbeddingPermissions(operation);
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
         * Creates secure WebSocket connection for embedding monitoring
         * @param {string} url - WebSocket URL
         * @param {object} options - Connection options
         * @returns {WebSocket|null} Secure WebSocket connection
         */
        createSecureWebSocket: function (url, options) {
            try {
                // Ensure secure protocol
                const secureUrl = url.replace(/^ws:\/\//, 'wss://').replace(/^http:\/\//, 'https://');
                
                // Validate URL
                const urlValidation = this.validateDataSourceURL(secureUrl.replace('wss://', 'https://'));
                if (!urlValidation.isValid) {
                    Log.error("Invalid WebSocket URL", urlValidation.error);
                    return null;
                }

                const ws = new WebSocket(secureUrl);
                
                // Add security event handlers
                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        // Validate incoming data
                        if (this._isValidEmbeddingUpdate(data)) {
                            if (options.onmessage) {
                                options.onmessage(event);
                            }
                        } else {
                            Log.warning("Invalid embedding update data received");
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
         * Creates secure EventSource for embedding streams
         * @param {string} url - EventSource URL
         * @param {object} options - Connection options
         * @returns {EventSource|null} Secure EventSource connection
         */
        createSecureEventSource: function (url, options) {
            try {
                // Ensure secure protocol
                const secureUrl = url.replace(/^http:\/\//, 'https://');
                
                // Validate URL
                const urlValidation = this.validateDataSourceURL(secureUrl);
                if (!urlValidation.isValid) {
                    Log.error("Invalid EventSource URL", urlValidation.error);
                    return null;
                }

                const eventSource = new EventSource(urlValidation.sanitizedUrl);
                
                // Add security handlers for different event types
                const secureEventHandler = (eventType) => {
                    return (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            if (this._isValidEmbeddingUpdate(data)) {
                                if (options[eventType]) {
                                    options[eventType](event);
                                }
                            } else {
                                Log.warning(`Invalid ${eventType} data received`);
                            }
                        } catch (error) {
                            Log.error(`Invalid ${eventType} message format`, error);
                        }
                    };
                };

                // Add handlers for common embedding events
                ['training-progress', 'training-completed', 'evaluation-progress', 
                 'optimization-progress'].forEach(eventType => {
                    if (options[eventType]) {
                        eventSource.addEventListener(eventType, secureEventHandler(eventType));
                    }
                });

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
        _containsCodeInjection: function (str) {
            if (!str || typeof str !== 'string') return false;
            
            const codePatterns = [
                /eval\s*\(/gi,
                /Function\s*\(/gi,
                /setTimeout\s*\(/gi,
                /setInterval\s*\(/gi,
                /__proto__/gi,
                /constructor\s*\[/gi,
                /import\s*\(/gi,
                /require\s*\(/gi
            ];
            
            return codePatterns.some(pattern => pattern.test(str));
        },

        _isValidModelStructure: function (modelData) {
            // Basic validation - in production, implement comprehensive checks
            if (!modelData || typeof modelData !== 'object') {
                return false;
            }
            
            // Check for required model properties
            const requiredProps = ['architecture', 'weights', 'config'];
            return requiredProps.some(prop => modelData.hasOwnProperty(prop));
        },

        _isValidEmbeddingUpdate: function (data) {
            // Validate embedding update data structure
            if (!data || typeof data !== 'object') {
                return false;
            }
            
            // Check for expected properties
            const validTypes = ['training-progress', 'evaluation-progress', 
                              'optimization-progress', 'deployment-status'];
            
            return data.type && validTypes.includes(data.type);
        },

        _getCurrentUser: function () {
            // Mock user detection - implement actual user detection
            return {
                id: 'current-user',
                permissions: ['embedding:read', 'embedding:write', 'embedding:train', 'embedding:deploy']
            };
        },

        _getEmbeddingPermissions: function (operation) {
            const permissionMap = {
                'GetEmbeddingStatistics': ['embedding:read'],
                'GetModelConfiguration': ['embedding:read'],
                'GetEvaluationMetrics': ['embedding:read'],
                'GetAvailableBenchmarks': ['embedding:read'],
                'GetHyperparameterSpace': ['embedding:read'],
                'GetVectorDatabases': ['embedding:read'],
                'AnalyzeModelPerformance': ['embedding:read'],
                'GetFineTuningOptions': ['embedding:write'],
                'DeployModel': ['embedding:deploy'],
                'FineTuneModel': ['embedding:train'],
                'OptimizeModel': ['embedding:train']
            };
            
            return permissionMap[operation] || ['embedding:read'];
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

        _isInternalHost: function (hostname) {
            const internalPatterns = [
                /^localhost$/,
                /^127\./,
                /^192\.168\./,
                /^10\./,
                /^172\.(1[6-9]|2[0-9]|3[0-1])\./,
                /\.local$/
            ];
            
            return internalPatterns.some(pattern => pattern.test(hostname));
        },

        _sendToAuditService: function (logEntry) {
            // In production, implement actual audit service integration
            console.log("AUDIT:", logEntry);
        }
    };
});