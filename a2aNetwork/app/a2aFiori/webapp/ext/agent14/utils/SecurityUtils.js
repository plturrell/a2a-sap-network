sap.ui.define([
    "sap/base/Log",
    "sap/m/MessageToast"
], function (Log, MessageToast) {
    "use strict";

    return {
        /**
         * Validates ML model path for security vulnerabilities
         * @param {string} modelPath - Model file path to validate
         * @param {object} options - Additional validation options
         * @returns {object} Validation result with security checks
         */
        validateModelPath: function (modelPath, options) {
            const opts = options || {};
            const validation = {
                isValid: true,
                errors: [],
                warnings: [],
                sanitizedPath: "",
                riskScore: 0
            };

            if (!modelPath || typeof modelPath !== 'string') {
                validation.isValid = false;
                validation.errors.push("Model path is required and must be a string");
                validation.riskScore = 100;
                return validation;
            }

            // Enhanced path traversal detection
            const pathTraversalPatterns = [
                /\.\./g,
                /~[\\/]/g,
                /%2e%2e/gi,
                /%2f/gi,
                /%5c/gi,
                /\\\.\\/g,
                /\/\.\\/g,
                /\x2e\x2e/g,
                /\u002e\u002e/g
            ];

            if (pathTraversalPatterns.some(pattern => pattern.test(modelPath))) {
                validation.isValid = false;
                validation.errors.push("Path traversal attempt detected");
                validation.riskScore += 90;
                this.logSecureOperation('SECURITY_VIOLATION', 'ERROR', {
                    type: 'PATH_TRAVERSAL',
                    path: modelPath,
                    ip: this._getClientIP()
                });
                return validation;
            }

            // Check for absolute paths and system paths
            const absolutePathPatterns = [
                /^[\\/]/,
                /^[A-Za-z]:[\\]/,
                /^file:\/\//i,
                /^(\\\\|\/\/)/,
                /^(proc|sys|dev|boot|root|etc|var|usr|bin|sbin|lib|tmp)\/*/i
            ];

            if (absolutePathPatterns.some(pattern => pattern.test(modelPath))) {
                validation.isValid = false;
                validation.errors.push("Absolute or system paths not allowed for security reasons");
                validation.riskScore += 80;
                return validation;
            }

            // Validate allowed model file extensions with stricter checking
            const allowedExtensions = ['.pth', '.pt', '.h5', '.pkl', '.joblib', '.onnx', '.pb', '.tflite', '.safetensors'];
            const extension = this._getFileExtension(modelPath);
            
            if (!extension || !allowedExtensions.includes(extension.toLowerCase())) {
                validation.isValid = false;
                validation.errors.push("Invalid model file extension. Allowed: " + allowedExtensions.join(', '));
                validation.riskScore += 60;
                return validation;
            }

            // Enhanced suspicious pattern detection
            const suspiciousPatterns = [
                // Executable files
                /\.(exe|dll|so|dylib|app|deb|rpm|msi|pkg|dmg)$/i,
                // Script files
                /\.(sh|bat|ps1|cmd|vbs|js|py|rb|php|pl|lua)$/i,
                // System directories
                /__pycache__|node_modules|\.git|\.svn/i,
                // Potential malware patterns
                /(malware|virus|trojan|backdoor|rootkit)/i,
                // Suspicious model names
                /(inject|exploit|payload|shell|reverse)/i,
                // Hidden files (Unix)
                /^\..*(?<!\.(pth|pt|h5|pkl|joblib|onnx|pb|tflite|safetensors))$/i
            ];

            for (const pattern of suspiciousPatterns) {
                if (pattern.test(modelPath)) {
                    validation.isValid = false;
                    validation.errors.push("Suspicious file pattern detected: " + pattern.toString());
                    validation.riskScore += 85;
                    this.logSecureOperation('SECURITY_VIOLATION', 'ERROR', {
                        type: 'SUSPICIOUS_PATTERN',
                        path: modelPath,
                        pattern: pattern.toString()
                    });
                    return validation;
                }
            }

            // Check for model injection patterns in path
            if (this._detectModelInjection(modelPath)) {
                validation.isValid = false;
                validation.errors.push("Potential model injection detected in path");
                validation.riskScore += 95;
                return validation;
            }

            // File size validation (if option provided)
            if (opts.maxSize && opts.fileSize > opts.maxSize) {
                validation.warnings.push(`File size (${opts.fileSize}) exceeds maximum allowed (${opts.maxSize})`);
                validation.riskScore += 30;
            }

            // Sanitize the path with enhanced security
            validation.sanitizedPath = this._sanitizeModelPath(modelPath);
            
            // Path length check
            if (validation.sanitizedPath.length > 255) {
                validation.warnings.push("Path length exceeds recommended limit");
                validation.riskScore += 20;
            }

            // Final risk assessment
            if (validation.riskScore > 50) {
                validation.warnings.push(`High risk score: ${validation.riskScore}`);
            }

            return validation;
        },

        /**
         * Validates training data for security vulnerabilities and data poisoning
         * @param {object} trainingData - Training data configuration
         * @param {object} options - Additional validation options
         * @returns {object} Validation result
         */
        validateTrainingData: function (trainingData, options) {
            const opts = options || {};
            const validation = {
                isValid: true,
                errors: [],
                warnings: [],
                sanitized: {},
                riskScore: 0,
                poisoningIndicators: []
            };

            if (!trainingData || typeof trainingData !== 'object') {
                validation.isValid = false;
                validation.errors.push("Training data configuration is required");
                validation.riskScore = 100;
                return validation;
            }

            // Enhanced dataset path validation
            if (trainingData.datasetPath) {
                const pathValidation = this.validateModelPath(trainingData.datasetPath, { 
                    maxSize: opts.maxDatasetSize,
                    fileSize: trainingData.datasetSize 
                });
                if (!pathValidation.isValid) {
                    validation.isValid = false;
                    validation.errors.push("Invalid dataset path: " + pathValidation.errors.join(', '));
                    validation.riskScore += pathValidation.riskScore;
                } else {
                    validation.sanitized.datasetPath = pathValidation.sanitizedPath;
                    validation.riskScore += pathValidation.riskScore;
                }
            }

            // Enhanced data source URL validation
            if (trainingData.dataSourceUrl) {
                const urlValidation = this.validateDataSourceURL(trainingData.dataSourceUrl);
                if (!urlValidation.isValid) {
                    validation.errors.push("Invalid data source URL: " + urlValidation.error);
                    validation.isValid = false;
                    validation.riskScore += 50;
                } else {
                    validation.sanitized.dataSourceUrl = urlValidation.sanitizedUrl;
                }
            }

            // Data poisoning detection
            const poisoningCheck = this._detectDataPoisoning(trainingData);
            if (poisoningCheck.detected) {
                validation.warnings.push("Potential data poisoning indicators detected");
                validation.poisoningIndicators = poisoningCheck.indicators;
                validation.riskScore += poisoningCheck.riskScore;
            }

            // Enhanced batch size validation
            if (trainingData.batchSize !== undefined) {
                const batchSize = this._validateIntegerParam(trainingData.batchSize, 'batchSize');
                if (batchSize.error) {
                    validation.errors.push(batchSize.error);
                    validation.isValid = false;
                    validation.riskScore += 20;
                } else if (batchSize.value < 1 || batchSize.value > 1024) {
                    validation.errors.push("Batch size must be between 1 and 1024");
                    validation.isValid = false;
                    validation.riskScore += 25;
                } else if (batchSize.value > 512) {
                    validation.warnings.push("Large batch size may cause memory exhaustion");
                    validation.riskScore += 15;
                }
                validation.sanitized.batchSize = batchSize.value;
            }

            // Enhanced augmentation settings validation
            if (trainingData.augmentation) {
                const augValidation = this._validateAugmentationConfig(trainingData.augmentation);
                if (!augValidation.isValid) {
                    validation.errors.push("Augmentation validation failed: " + augValidation.errors.join(', '));
                    validation.isValid = false;
                    validation.riskScore += augValidation.riskScore;
                } else {
                    validation.sanitized.augmentation = augValidation.sanitized;
                    validation.riskScore += augValidation.riskScore;
                }
            }

            // Validate sample counts
            if (trainingData.sampleCount !== undefined) {
                const sampleCount = this._validateIntegerParam(trainingData.sampleCount, 'sampleCount');
                if (sampleCount.error) {
                    validation.errors.push(sampleCount.error);
                    validation.isValid = false;
                    validation.riskScore += 15;
                } else if (sampleCount.value > 10000000) {
                    validation.warnings.push("Very large sample count may indicate resource exhaustion attempt");
                    validation.riskScore += 30;
                } else if (sampleCount.value < 10) {
                    validation.warnings.push("Very small sample count may indicate data poisoning attempt");
                    validation.riskScore += 20;
                }
                validation.sanitized.sampleCount = sampleCount.value;
            }

            // Validate data format and encoding
            if (trainingData.dataFormat) {
                const allowedFormats = ['csv', 'json', 'jsonl', 'tsv', 'parquet', 'arrow'];
                const format = this._sanitizeString(trainingData.dataFormat).toLowerCase();
                if (!allowedFormats.includes(format)) {
                    validation.errors.push("Invalid data format. Allowed: " + allowedFormats.join(', '));
                    validation.isValid = false;
                    validation.riskScore += 30;
                } else {
                    validation.sanitized.dataFormat = format;
                }
            }

            // Check for suspicious data characteristics
            if (this._detectSuspiciousDataCharacteristics(trainingData)) {
                validation.warnings.push("Suspicious data characteristics detected");
                validation.riskScore += 25;
            }

            return validation;
        },

        /**
         * Validates hyperparameters for security and sanity with enhanced bounds checking
         * @param {object} hyperparameters - Hyperparameter configuration
         * @param {object} modelContext - Model context for validation
         * @returns {object} Validation result
         */
        validateHyperparameters: function (hyperparameters, modelContext) {
            const validation = {
                isValid: true,
                errors: [],
                warnings: [],
                sanitized: {},
                riskScore: 0
            };

            if (!hyperparameters || typeof hyperparameters !== 'object') {
                validation.isValid = false;
                validation.errors.push("Hyperparameters configuration is required");
                validation.riskScore = 100;
                return validation;
            }

            // Check for hyperparameter injection attacks
            if (this._detectHyperparameterInjection(hyperparameters)) {
                validation.isValid = false;
                validation.errors.push("Hyperparameter injection attempt detected");
                validation.riskScore = 100;
                this.logSecureOperation('SECURITY_VIOLATION', 'ERROR', {
                    type: 'HYPERPARAMETER_INJECTION',
                    params: Object.keys(hyperparameters)
                });
                return validation;
            }

            // Enhanced learning rate validation
            if (hyperparameters.learningRate !== undefined) {
                const lr = this._validateNumericParam(hyperparameters.learningRate, 'learningRate');
                if (lr.error) {
                    validation.errors.push(lr.error);
                    validation.isValid = false;
                    validation.riskScore += 20;
                } else if (lr.value <= 0 || lr.value > 1.0) {
                    validation.errors.push("Learning rate must be between 0 and 1.0");
                    validation.isValid = false;
                    validation.riskScore += 30;
                } else if (lr.value > 0.1) {
                    validation.warnings.push("Learning rate is unusually high, may cause instability");
                    validation.riskScore += 10;
                }
                validation.sanitized.learningRate = lr.value;
            }

            // Enhanced epochs validation
            if (hyperparameters.epochs !== undefined) {
                const epochs = this._validateIntegerParam(hyperparameters.epochs, 'epochs');
                if (epochs.error) {
                    validation.errors.push(epochs.error);
                    validation.isValid = false;
                    validation.riskScore += 20;
                } else if (epochs.value < 1 || epochs.value > 1000) {
                    validation.errors.push("Epochs must be between 1 and 1000");
                    validation.isValid = false;
                    validation.riskScore += 25;
                } else if (epochs.value > 500) {
                    validation.warnings.push("High epoch count may indicate resource exhaustion attempt");
                    validation.riskScore += 15;
                }
                validation.sanitized.epochs = epochs.value;
            }

            // Enhanced batch size validation
            if (hyperparameters.batchSize !== undefined) {
                const batchSize = this._validateIntegerParam(hyperparameters.batchSize, 'batchSize');
                if (batchSize.error) {
                    validation.errors.push(batchSize.error);
                    validation.isValid = false;
                    validation.riskScore += 20;
                } else if (batchSize.value < 1 || batchSize.value > 1024) {
                    validation.errors.push("Batch size must be between 1 and 1024");
                    validation.isValid = false;
                    validation.riskScore += 25;
                } else if (batchSize.value > 512) {
                    validation.warnings.push("Large batch size may cause memory exhaustion");
                    validation.riskScore += 10;
                }
                validation.sanitized.batchSize = batchSize.value;
            }

            // Enhanced optimizer validation
            if (hyperparameters.optimizer) {
                const allowedOptimizers = ['adam', 'sgd', 'adamw', 'rmsprop', 'adagrad', 'lamb', 'radam'];
                const optimizer = this._sanitizeString(hyperparameters.optimizer).toLowerCase();
                
                if (!allowedOptimizers.includes(optimizer)) {
                    validation.errors.push("Invalid optimizer. Allowed: " + allowedOptimizers.join(', '));
                    validation.isValid = false;
                    validation.riskScore += 40;
                } else {
                    validation.sanitized.optimizer = optimizer;
                }
            }

            // Validate weight decay
            if (hyperparameters.weightDecay !== undefined) {
                const wd = this._validateNumericParam(hyperparameters.weightDecay, 'weightDecay');
                if (wd.error) {
                    validation.errors.push(wd.error);
                    validation.isValid = false;
                    validation.riskScore += 15;
                } else if (wd.value < 0 || wd.value > 1.0) {
                    validation.errors.push("Weight decay must be between 0 and 1.0");
                    validation.isValid = false;
                    validation.riskScore += 20;
                }
                validation.sanitized.weightDecay = wd.value;
            }

            // Validate dropout rate
            if (hyperparameters.dropout !== undefined) {
                const dropout = this._validateNumericParam(hyperparameters.dropout, 'dropout');
                if (dropout.error) {
                    validation.errors.push(dropout.error);
                    validation.isValid = false;
                    validation.riskScore += 15;
                } else if (dropout.value < 0 || dropout.value >= 1.0) {
                    validation.errors.push("Dropout rate must be between 0 and 1.0 (exclusive)");
                    validation.isValid = false;
                    validation.riskScore += 20;
                }
                validation.sanitized.dropout = dropout.value;
            }

            // Enhanced custom config validation
            if (hyperparameters.customConfig) {
                const configValidation = this._validateCustomConfig(hyperparameters.customConfig);
                if (!configValidation.isValid) {
                    validation.errors.push("Custom config validation failed: " + configValidation.errors.join(', '));
                    validation.isValid = false;
                    validation.riskScore += configValidation.riskScore;
                } else {
                    validation.sanitized.customConfig = configValidation.sanitized;
                    validation.riskScore += configValidation.riskScore;
                }
            }

            // Validate loss function
            if (hyperparameters.lossFunction) {
                const allowedLossFunctions = ['cosine', 'triplet', 'contrastive', 'infonce', 'mse', 'mae'];
                const lossFunc = this._sanitizeString(hyperparameters.lossFunction).toLowerCase();
                
                if (!allowedLossFunctions.includes(lossFunc)) {
                    validation.errors.push("Invalid loss function. Allowed: " + allowedLossFunctions.join(', '));
                    validation.isValid = false;
                    validation.riskScore += 30;
                } else {
                    validation.sanitized.lossFunction = lossFunc;
                }
            }

            // Check for resource exhaustion patterns
            if (this._detectResourceExhaustionAttempt(hyperparameters)) {
                validation.warnings.push("Potential resource exhaustion pattern detected");
                validation.riskScore += 40;
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
         * Securely saves model with enhanced validation and integrity checking
         * @param {object} modelData - Model data to save
         * @param {string} format - Save format
         * @param {object} options - Additional options
         * @returns {object} Validation result
         */
        secureModelSave: function (modelData, format, options) {
            const opts = options || {};
            const validation = {
                isValid: true,
                errors: [],
                warnings: [],
                integrityHash: null,
                sanitized: null,
                riskScore: 0
            };

            // Enhanced format validation
            const allowedFormats = {
                'pytorch': { extensions: ['.pth', '.pt'], secure: true },
                'tensorflow': { extensions: ['.pb', '.h5'], secure: true },
                'onnx': { extensions: ['.onnx'], secure: true },
                'safetensors': { extensions: ['.safetensors'], secure: true, preferred: true },
                'huggingface': { extensions: ['.bin', '.safetensors'], secure: false }
            };
            
            const formatInfo = allowedFormats[format];
            if (!formatInfo) {
                validation.isValid = false;
                validation.errors.push("Invalid save format. Allowed: " + Object.keys(allowedFormats).join(', '));
                validation.riskScore = 80;
                return validation;
            }

            if (!formatInfo.secure) {
                validation.warnings.push(`Format '${format}' has known security risks`);
                validation.riskScore += 30;
            }

            if (!formatInfo.preferred) {
                validation.warnings.push(`Consider using safetensors format for enhanced security`);
                validation.riskScore += 10;
            }

            // Enhanced model size validation
            let modelSize = 0;
            try {
                if (typeof modelData === 'string') {
                    modelSize = new Blob([modelData]).size;
                } else {
                    modelSize = new Blob([JSON.stringify(modelData)]).size;
                }
            } catch (error) {
                validation.errors.push("Failed to calculate model size");
                validation.riskScore += 20;
            }

            const maxSize = opts.maxSize || (2 * 1024 * 1024 * 1024); // 2GB default limit
            if (modelSize > maxSize) {
                validation.isValid = false;
                validation.errors.push(`Model size (${this._formatBytes(modelSize)}) exceeds maximum allowed (${this._formatBytes(maxSize)})`);
                validation.riskScore += 60;
                return validation;
            }

            // Detect model structure attacks
            const structureValidation = this._validateModelStructure(modelData, format);
            if (!structureValidation.isValid) {
                validation.isValid = false;
                validation.errors = validation.errors.concat(structureValidation.errors);
                validation.riskScore += structureValidation.riskScore;
                return validation;
            }

            // Check for model injection payloads
            if (this._detectModelInjectionPayload(modelData)) {
                validation.isValid = false;
                validation.errors.push("Potential model injection payload detected");
                validation.riskScore = 100;
                this.logSecureOperation('SECURITY_VIOLATION', 'ERROR', {
                    type: 'MODEL_INJECTION_PAYLOAD',
                    format: format,
                    size: modelSize
                });
                return validation;
            }

            // Generate integrity hash
            try {
                validation.integrityHash = this._generateModelHash(modelData);
            } catch (error) {
                validation.warnings.push("Failed to generate integrity hash");
                validation.riskScore += 15;
            }

            // Sanitize model data
            validation.sanitized = this._sanitizeModelData(modelData, format);

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
        },

        /**
         * Sanitizes input strings to prevent XSS and injection attacks
         * @param {string} input - Input string to sanitize
         * @returns {string} Sanitized string
         */
        sanitizeInput: function (input) {
            if (!input || typeof input !== 'string') {
                return '';
            }
            
            return input
                .replace(/[<>"'&]/g, function(match) {
                    const htmlEntities = {
                        '<': '&lt;',
                        '>': '&gt;',
                        '"': '&quot;',
                        "'": '&#x27;',
                        '&': '&amp;'
                    };
                    return htmlEntities[match];
                })
                .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '') // Remove control characters
                .trim();
        },

        /**
         * Validates model name for security
         * @param {string} modelName - Model name to validate
         * @returns {boolean} True if valid
         */
        isValidModelName: function (modelName) {
            if (!modelName || typeof modelName !== 'string') {
                return false;
            }
            
            // Allow only alphanumeric, hyphens, underscores, and dots
            const validNamePattern = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,62}$/;
            return validNamePattern.test(modelName) && !this._containsCodeInjection(modelName);
        },

        /**
         * Validates dataset path for security
         * @param {string} datasetPath - Dataset path to validate
         * @returns {boolean} True if valid
         */
        isValidDatasetPath: function (datasetPath) {
            if (!datasetPath || typeof datasetPath !== 'string') {
                return false;
            }
            
            const pathValidation = this.validateModelPath(datasetPath);
            return pathValidation.isValid;
        },

        /**
         * Validates WebSocket URL for security
         * @param {string} url - WebSocket URL to validate
         * @returns {boolean} True if valid
         */
        validateWebSocketUrl: function (url) {
            if (!url || typeof url !== 'string') {
                return false;
            }
            
            try {
                const urlObj = new URL(url);
                return urlObj.protocol === 'wss:' && !this._isInternalHost(urlObj.hostname);
            } catch (error) {
                return false;
            }
        },

        /**
         * Sanitizes embedding data for security
         * @param {string} data - Data to sanitize
         * @returns {string} Sanitized data
         */
        sanitizeEmbeddingData: function (data) {
            if (!data || typeof data !== 'string') {
                return '{}';
            }
            
            try {
                const parsed = JSON.parse(data);
                return JSON.stringify(this._sanitizeObject(parsed));
            } catch (error) {
                return '{}';
            }
        },

        /**
         * Detects model injection attempts in paths
         * @param {string} path - Path to check
         * @returns {boolean} True if injection detected
         * @private
         */
        _detectModelInjection: function (path) {
            const injectionPatterns = [
                // Python code injection
                /(__import__|exec|eval|compile|open|file)\s*\(/i,
                // Pickle/joblib injection
                /(pickle|joblib|dill|cloudpickle)\.(load|loads|dump|dumps)/i,
                // OS command injection
                /(os\.|subprocess\.|system\()/i,
                // Network requests
                /(urllib|requests|http|socket)\./i,
                // File system access
                /(shutil|glob|pathlib)\./i,
                // Model architecture manipulation
                /(torch\.jit|tensorflow\.saved_model|keras\.models)/i
            ];
            
            return injectionPatterns.some(pattern => pattern.test(path));
        },

        /**
         * Detects hyperparameter injection attempts
         * @param {object} params - Hyperparameters to check
         * @returns {boolean} True if injection detected
         * @private
         */
        _detectHyperparameterInjection: function (params) {
            const stringified = JSON.stringify(params);
            
            const injectionPatterns = [
                // Function calls
                /(\.|\[)(__[a-zA-Z_]+__|constructor|prototype)/i,
                // Code execution
                /(eval|Function|setTimeout|setInterval)\s*\(/i,
                // Python/system commands
                /(import\s+|from\s+|exec\s*\(|eval\s*\(|__import__)/i,
                // File operations
                /(open\s*\(|file\s*\(|read|write|delete)/i
            ];
            
            return injectionPatterns.some(pattern => pattern.test(stringified));
        },

        /**
         * Detects resource exhaustion attempts
         * @param {object} params - Parameters to check
         * @returns {boolean} True if exhaustion attempt detected
         * @private
         */
        _detectResourceExhaustionAttempt: function (params) {
            // Check for suspiciously high values that could cause DoS
            const suspiciousLimits = {
                epochs: 10000,
                batchSize: 10000,
                learningRate: 1000,
                hiddenSize: 100000,
                embeddingDimension: 100000,
                maxSequenceLength: 100000
            };
            
            for (const [key, limit] of Object.entries(suspiciousLimits)) {
                if (params[key] && parseFloat(params[key]) > limit) {
                    return true;
                }
            }
            
            return false;
        },

        /**
         * Validates custom configuration objects
         * @param {object} config - Custom config to validate
         * @returns {object} Validation result
         * @private
         */
        _validateCustomConfig: function (config) {
            const validation = {
                isValid: true,
                errors: [],
                sanitized: {},
                riskScore: 0
            };
            
            if (typeof config !== 'object' || config === null) {
                validation.isValid = false;
                validation.errors.push("Custom config must be an object");
                validation.riskScore = 50;
                return validation;
            }
            
            // Check config size
            const configStr = JSON.stringify(config);
            if (configStr.length > 10000) {
                validation.isValid = false;
                validation.errors.push("Custom config too large");
                validation.riskScore = 70;
                return validation;
            }
            
            // Check for dangerous patterns
            if (this._containsCodeInjection(configStr)) {
                validation.isValid = false;
                validation.errors.push("Custom config contains unsafe content");
                validation.riskScore = 100;
                return validation;
            }
            
            validation.sanitized = this._sanitizeObject(config);
            return validation;
        },

        /**
         * Validates numeric parameters
         * @param {any} value - Value to validate
         * @param {string} paramName - Parameter name
         * @returns {object} Validation result
         * @private
         */
        _validateNumericParam: function (value, paramName) {
            const result = { value: null, error: null };
            
            if (value === null || value === undefined) {
                result.error = `${paramName} is required`;
                return result;
            }
            
            const numValue = parseFloat(value);
            if (isNaN(numValue)) {
                result.error = `${paramName} must be a valid number`;
                return result;
            }
            
            if (!isFinite(numValue)) {
                result.error = `${paramName} must be finite`;
                return result;
            }
            
            result.value = numValue;
            return result;
        },

        /**
         * Validates integer parameters
         * @param {any} value - Value to validate
         * @param {string} paramName - Parameter name
         * @returns {object} Validation result
         * @private
         */
        _validateIntegerParam: function (value, paramName) {
            const numResult = this._validateNumericParam(value, paramName);
            if (numResult.error) {
                return numResult;
            }
            
            if (!Number.isInteger(numResult.value)) {
                numResult.error = `${paramName} must be an integer`;
                return numResult;
            }
            
            return numResult;
        },

        /**
         * Sanitizes strings with enhanced security
         * @param {string} str - String to sanitize
         * @returns {string} Sanitized string
         * @private
         */
        _sanitizeString: function (str) {
            if (!str || typeof str !== 'string') {
                return '';
            }
            
            return str
                .replace(/[\x00-\x1F\x7F]/g, '') // Remove control characters
                .replace(/[<>"'&]/g, '') // Remove HTML characters
                .trim()
                .substring(0, 1000); // Limit length
        },

        /**
         * Sanitizes objects recursively
         * @param {object} obj - Object to sanitize
         * @returns {object} Sanitized object
         * @private
         */
        _sanitizeObject: function (obj) {
            if (obj === null || obj === undefined) {
                return null;
            }
            
            if (typeof obj === 'string') {
                return this._sanitizeString(obj);
            }
            
            if (typeof obj === 'number') {
                return isFinite(obj) ? obj : 0;
            }
            
            if (typeof obj === 'boolean') {
                return obj;
            }
            
            if (Array.isArray(obj)) {
                return obj.slice(0, 100).map(item => this._sanitizeObject(item));
            }
            
            if (typeof obj === 'object') {
                const sanitized = {};
                let count = 0;
                for (const [key, value] of Object.entries(obj)) {
                    if (count++ > 50) break; // Limit object size
                    const sanitizedKey = this._sanitizeString(key);
                    if (sanitizedKey && !sanitizedKey.startsWith('__')) {
                        sanitized[sanitizedKey] = this._sanitizeObject(value);
                    }
                }
                return sanitized;
            }
            
            return null;
        },

        /**
         * Sanitizes model path with enhanced security
         * @param {string} path - Path to sanitize
         * @returns {string} Sanitized path
         * @private
         */
        _sanitizeModelPath: function (path) {
            return path
                .replace(/[\x00-\x1F\x7F]/g, '') // Remove control characters
                .replace(/[<>:"|?*]/g, '') // Remove invalid path characters
                .replace(/\s+/g, '_') // Replace spaces with underscores
                .replace(/[^\w\-._\/\\]/g, '') // Keep only safe characters
                .replace(/\.\.+/g, '.') // Collapse multiple dots
                .replace(/_{2,}/g, '_') // Collapse multiple underscores
                .substring(0, 255); // Limit path length
        },

        /**
         * Gets file extension from path
         * @param {string} path - File path
         * @returns {string} File extension
         * @private
         */
        _getFileExtension: function (path) {
            const parts = path.split('.');
            return parts.length > 1 ? '.' + parts[parts.length - 1] : '';
        },

        /**
         * Gets client IP address (mock implementation)
         * @returns {string} Client IP
         * @private
         */
        _getClientIP: function () {
            // In production, implement actual IP detection
            return 'unknown';
        },

        /**
         * Detects data poisoning indicators
         * @param {object} trainingData - Training data to analyze
         * @returns {object} Detection result
         * @private
         */
        _detectDataPoisoning: function (trainingData) {
            const result = {
                detected: false,
                indicators: [],
                riskScore: 0
            };
            
            // Check for unusual data distributions
            if (trainingData.sampleCount && trainingData.labelCount) {
                const samplesPerLabel = trainingData.sampleCount / trainingData.labelCount;
                if (samplesPerLabel < 2) {
                    result.indicators.push('Extremely low samples per label');
                    result.riskScore += 30;
                }
            }
            
            // Check for suspicious data source patterns
            if (trainingData.dataSourceUrl) {
                const suspiciousPatterns = [
                    /anonymous|temp|tmp|delete|remove/i,
                    /\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/,  // IP addresses
                    /localhost|127\.0\.0\.1|0\.0\.0\.0/
                ];
                
                for (const pattern of suspiciousPatterns) {
                    if (pattern.test(trainingData.dataSourceUrl)) {
                        result.indicators.push('Suspicious data source URL');
                        result.riskScore += 25;
                        break;
                    }
                }
            }
            
            // Check for rapid data changes
            if (trainingData.lastModified && trainingData.createdAt) {
                const timeDiff = new Date(trainingData.lastModified) - new Date(trainingData.createdAt);
                if (timeDiff < 60000) { // Less than 1 minute
                    result.indicators.push('Suspiciously recent data modification');
                    result.riskScore += 20;
                }
            }
            
            result.detected = result.indicators.length > 0;
            return result;
        },

        /**
         * Validates augmentation configuration
         * @param {any} augmentation - Augmentation config to validate
         * @returns {object} Validation result
         * @private
         */
        _validateAugmentationConfig: function (augmentation) {
            const validation = {
                isValid: true,
                errors: [],
                sanitized: null,
                riskScore: 0
            };
            
            if (typeof augmentation === 'string') {
                if (this._containsCodeInjection(augmentation)) {
                    validation.isValid = false;
                    validation.errors.push('Augmentation config contains code injection');
                    validation.riskScore = 100;
                } else {
                    validation.sanitized = this._sanitizeString(augmentation);
                }
            } else if (Array.isArray(augmentation)) {
                const allowedAugmentations = [
                    'synonym', 'backtranslation', 'paraphrase', 'noise', 'dropout', 'mixup', 'cutmix'
                ];
                
                const sanitized = [];
                for (const aug of augmentation) {
                    const cleanAug = this._sanitizeString(aug).toLowerCase();
                    if (allowedAugmentations.includes(cleanAug)) {
                        sanitized.push(cleanAug);
                    } else {
                        validation.errors.push(`Invalid augmentation type: ${cleanAug}`);
                        validation.riskScore += 15;
                    }
                }
                
                if (sanitized.length === 0 && augmentation.length > 0) {
                    validation.isValid = false;
                    validation.errors.push('No valid augmentation types found');
                    validation.riskScore += 30;
                }
                
                validation.sanitized = sanitized;
            } else if (typeof augmentation === 'object' && augmentation !== null) {
                validation.sanitized = this._sanitizeObject(augmentation);
                
                // Check for suspicious augmentation parameters
                if (augmentation.intensity && parseFloat(augmentation.intensity) > 1.0) {
                    validation.errors.push('Augmentation intensity too high');
                    validation.riskScore += 25;
                }
            } else {
                validation.isValid = false;
                validation.errors.push('Invalid augmentation configuration type');
                validation.riskScore += 40;
            }
            
            return validation;
        },

        /**
         * Detects suspicious data characteristics
         * @param {object} trainingData - Training data to analyze
         * @returns {boolean} True if suspicious characteristics detected
         * @private
         */
        _detectSuspiciousDataCharacteristics: function (trainingData) {
            // Check for extremely large or small datasets
            if (trainingData.sampleCount) {
                if (trainingData.sampleCount > 100000000 || trainingData.sampleCount < 5) {
                    return true;
                }
            }
            
            // Check for unusual file sizes
            if (trainingData.datasetSize) {
                const size = parseFloat(trainingData.datasetSize);
                if (size > 50 * 1024 * 1024 * 1024) { // 50GB
                    return true;
                }
            }
            
            // Check for suspicious metadata
            if (trainingData.metadata && typeof trainingData.metadata === 'object') {
                const metadataStr = JSON.stringify(trainingData.metadata);
                if (this._containsCodeInjection(metadataStr)) {
                    return true;
                }
            }
            
            return false;
        },

        /**
         * Enhanced model structure validation
         * @param {any} modelData - Model data to validate
         * @param {string} format - Model format
         * @returns {object} Validation result
         * @private
         */
        _validateModelStructure: function (modelData, format) {
            const validation = {
                isValid: true,
                errors: [],
                riskScore: 0
            };
            
            if (!modelData) {
                validation.isValid = false;
                validation.errors.push('Model data is required');
                validation.riskScore = 50;
                return validation;
            }
            
            // Format-specific validation
            switch (format) {
                case 'pytorch':
                    if (typeof modelData === 'object' && modelData.state_dict) {
                        // Valid PyTorch model structure
                    } else {
                        validation.errors.push('Invalid PyTorch model structure');
                        validation.riskScore += 30;
                    }
                    break;
                    
                case 'tensorflow':
                    if (typeof modelData === 'object' && (modelData.saved_model || modelData.keras_version)) {
                        // Valid TensorFlow model structure
                    } else {
                        validation.errors.push('Invalid TensorFlow model structure');
                        validation.riskScore += 30;
                    }
                    break;
                    
                case 'onnx':
                    if (typeof modelData === 'object' && modelData.graph) {
                        // Valid ONNX model structure
                    } else {
                        validation.errors.push('Invalid ONNX model structure');
                        validation.riskScore += 30;
                    }
                    break;
            }
            
            // Check for nested depth (potential zip bomb)
            if (this._calculateObjectDepth(modelData) > 20) {
                validation.errors.push('Model structure too deeply nested');
                validation.riskScore += 40;
            }
            
            // Check for circular references
            if (this._hasCircularReference(modelData)) {
                validation.errors.push('Model contains circular references');
                validation.riskScore += 50;
            }
            
            if (validation.errors.length > 0) {
                validation.isValid = false;
            }
            
            return validation;
        },

        /**
         * Detects model injection payloads
         * @param {any} modelData - Model data to check
         * @returns {boolean} True if payload detected
         * @private
         */
        _detectModelInjectionPayload: function (modelData) {
            const modelStr = JSON.stringify(modelData);
            
            const payloadPatterns = [
                // Python code execution
                /(exec|eval|compile|__import__)\s*\(/i,
                // Subprocess calls
                /(subprocess|os\.system|popen)\s*\(/i,
                // Network requests
                /(urllib|requests|socket|http)\./i,
                // File operations
                /(open|file|read|write)\s*\(/i,
                // Pickle vulnerabilities
                /(pickle|joblib|dill)\.load/i,
                // Model architecture manipulation
                /(__reduce__|__setstate__|__getstate__)/i
            ];
            
            return payloadPatterns.some(pattern => pattern.test(modelStr));
        },

        /**
         * Generates model integrity hash
         * @param {any} modelData - Model data
         * @returns {string} Hash string
         * @private
         */
        _generateModelHash: function (modelData) {
            // Simple hash implementation - in production use crypto API
            const str = JSON.stringify(modelData);
            let hash = 0;
            for (let i = 0; i < str.length; i++) {
                const char = str.charCodeAt(i);
                hash = ((hash << 5) - hash) + char;
                hash = hash & hash; // Convert to 32-bit integer
            }
            return hash.toString(16);
        },

        /**
         * Sanitizes model data
         * @param {any} modelData - Model data to sanitize
         * @param {string} format - Model format
         * @returns {any} Sanitized model data
         * @private
         */
        _sanitizeModelData: function (modelData, format) {
            // Basic sanitization - remove potentially dangerous properties
            if (typeof modelData === 'object' && modelData !== null) {
                const sanitized = {};
                for (const [key, value] of Object.entries(modelData)) {
                    const cleanKey = this._sanitizeString(key);
                    if (cleanKey && !cleanKey.startsWith('__') && cleanKey !== 'eval' && cleanKey !== 'exec') {
                        sanitized[cleanKey] = this._sanitizeObject(value);
                    }
                }
                return sanitized;
            }
            return modelData;
        },

        /**
         * Formats bytes for display
         * @param {number} bytes - Byte count
         * @returns {string} Formatted string
         * @private
         */
        _formatBytes: function (bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },

        /**
         * Calculates object nesting depth
         * @param {any} obj - Object to analyze
         * @param {number} currentDepth - Current depth
         * @returns {number} Maximum depth
         * @private
         */
        _calculateObjectDepth: function (obj, currentDepth) {
            currentDepth = currentDepth || 0;
            
            if (currentDepth > 50) return currentDepth; // Prevent stack overflow
            
            if (obj === null || typeof obj !== 'object') {
                return currentDepth;
            }
            
            let maxDepth = currentDepth;
            for (const value of Object.values(obj)) {
                const depth = this._calculateObjectDepth(value, currentDepth + 1);
                maxDepth = Math.max(maxDepth, depth);
            }
            
            return maxDepth;
        },

        /**
         * Checks for circular references
         * @param {any} obj - Object to check
         * @param {Set} visited - Visited objects
         * @returns {boolean} True if circular reference found
         * @private
         */
        _hasCircularReference: function (obj, visited) {
            visited = visited || new Set();
            
            if (obj === null || typeof obj !== 'object') {
                return false;
            }
            
            if (visited.has(obj)) {
                return true;
            }
            
            visited.add(obj);
            
            for (const value of Object.values(obj)) {
                if (this._hasCircularReference(value, visited)) {
                    return true;
                }
            }
            
            visited.delete(obj);
            return false;
        }
    };
});