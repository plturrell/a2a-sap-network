sap.ui.define([
    "sap/base/Log",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], function (Log, MessageToast, MessageBox) {
    "use strict";

    return {
        
        validateMetadata: function (metadata) {
            if (!metadata) {
                Log.error("SecurityUtils", "Metadata validation failed: null or undefined metadata");
                return { valid: false, error: "Metadata cannot be null or empty" };
            }
            
            const sanitizedMetadata = this.sanitizeMetadata(metadata);
            
            if (this.containsSuspiciousContent(sanitizedMetadata)) {
                Log.warning("SecurityUtils", "Metadata validation failed: potentially malicious content detected");
                return { valid: false, error: "Metadata contains suspicious content" };
            }
            
            return { valid: true, sanitized: sanitizedMetadata };
        },
        
        sanitizeMetadata: function (metadata) {
            if (typeof metadata === 'string') {
                return metadata
                    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
                    .replace(/javascript:/gi, '')
                    .replace(/on\w+\s*=/gi, '')
                    .replace(/eval\s*\(/gi, '')
                    .replace(/Function\s*\(/gi, '')
                    .replace(/setTimeout\s*\(/gi, '')
                    .replace(/setInterval\s*\(/gi, '');
            }
            
            if (typeof metadata === 'object' && metadata !== null) {
                const sanitized = {};
                for (const key in metadata) {
                    if (metadata.hasOwnProperty(key)) {
                        sanitized[key] = this.sanitizeMetadata(metadata[key]);
                    }
                }
                return sanitized;
            }
            
            return metadata;
        },
        
        containsSuspiciousContent: function (content) {
            if (typeof content !== 'string') {
                return false;
            }
            
            const suspiciousPatterns = [
                /eval\s*\(/gi,
                /Function\s*\(/gi,
                /setTimeout\s*\(/gi,
                /setInterval\s*\(/gi,
                /<script/gi,
                /javascript:/gi,
                /on\w+\s*=/gi,
                /\$\{.*\}/gi,
                /<%.*%>/gi
            ];
            
            return suspiciousPatterns.some(pattern => pattern.test(content));
        },
        
        validateDublinCore: function (dublinCoreData) {
            if (!dublinCoreData) {
                return { valid: false, error: "Dublin Core data is required" };
            }
            
            const requiredFields = ['title', 'creator', 'subject', 'description', 'date', 'type'];
            const missingFields = requiredFields.filter(field => !dublinCoreData[field]);
            
            if (missingFields.length > 0) {
                return { 
                    valid: false, 
                    error: `Missing required Dublin Core fields: ${missingFields.join(', ')}` 
                };
            }
            
            return this.validateMetadata(dublinCoreData);
        },
        
        validateSchema: function (schema) {
            if (!schema || typeof schema !== 'object') {
                return { valid: false, error: "Invalid schema format" };
            }
            
            if (schema.constructor !== Object && schema.constructor !== Array) {
                return { valid: false, error: "Schema must be a plain object or array" };
            }
            
            const sanitizedSchema = this.sanitizeMetadata(schema);
            
            if (JSON.stringify(sanitizedSchema) !== JSON.stringify(schema)) {
                Log.warning("SecurityUtils", "Schema contained suspicious content that was sanitized");
            }
            
            return { valid: true, sanitized: sanitizedSchema };
        },
        
        validateLineageData: function (lineageData) {
            if (!lineageData) {
                return { valid: false, error: "Lineage data is required" };
            }
            
            const validation = this.validateMetadata(lineageData);
            if (!validation.valid) {
                return validation;
            }
            
            if (lineageData.source && !this.validateSource(lineageData.source)) {
                return { valid: false, error: "Invalid or suspicious source in lineage data" };
            }
            
            return validation;
        },
        
        validateSource: function (source) {
            if (typeof source !== 'string') {
                return false;
            }
            
            const validSourcePattern = /^[a-zA-Z0-9._\-\/]+$/;
            return validSourcePattern.test(source) && !this.containsSuspiciousContent(source);
        },
        
        validateQualityMetrics: function (metrics) {
            if (!metrics || typeof metrics !== 'object') {
                return { valid: false, error: "Invalid quality metrics format" };
            }
            
            for (const key in metrics) {
                if (metrics.hasOwnProperty(key)) {
                    const value = metrics[key];
                    
                    if (typeof value === 'number') {
                        if (value < 0 || value > 100 || isNaN(value)) {
                            return { valid: false, error: `Invalid quality metric value for ${key}` };
                        }
                    } else if (typeof value === 'string') {
                        if (this.containsSuspiciousContent(value)) {
                            return { valid: false, error: `Suspicious content in quality metric ${key}` };
                        }
                    }
                }
            }
            
            return { valid: true, sanitized: this.sanitizeMetadata(metrics) };
        },
        
        secureCallFunction: function (model, functionName, parameters, successCallback, errorCallback) {
            const token = model.getSecurityToken();
            
            if (!token) {
                model.refreshSecurityToken(
                    () => this._executeSecureCall(model, functionName, parameters, successCallback, errorCallback),
                    errorCallback
                );
            } else {
                this._executeSecureCall(model, functionName, parameters, successCallback, errorCallback);
            }
        },
        
        _executeSecureCall: function (model, functionName, parameters, successCallback, errorCallback) {
            try {
                const sanitizedParameters = this.sanitizeMetadata(parameters);
                
                model.callFunction(functionName, {
                    urlParameters: sanitizedParameters,
                    success: successCallback,
                    error: errorCallback,
                    headers: {
                        "X-CSRF-Token": model.getSecurityToken() || "Fetch",
                        "X-Requested-With": "XMLHttpRequest"
                    }
                });
            } catch (error) {
                Log.error("SecurityUtils", "Secure call execution failed: " + error.message);
                if (errorCallback) {
                    errorCallback(error);
                }
            }
        },
        
        createSecureWebSocket: function (url, options) {
            if (!url) {
                throw new Error("WebSocket URL is required");
            }
            
            const secureUrl = url.replace(/^ws:\/\//, 'wss://').replace(/^http:\/\//, 'https://');
            
            if (!secureUrl.startsWith('wss://')) {
                throw new Error("Only secure WebSocket connections (wss://) are allowed");
            }
            
            try {
                const ws = new WebSocket(secureUrl);
                
                if (options && options.onmessage) {
                    ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            const validation = this.validateMetadata(data);
                            
                            if (validation.valid) {
                                options.onmessage(event, validation.sanitized);
                            } else {
                                Log.warning("SecurityUtils", "WebSocket message validation failed: " + validation.error);
                            }
                        } catch (error) {
                            Log.error("SecurityUtils", "WebSocket message parsing failed: " + error.message);
                        }
                    };
                }
                
                if (options && options.onopen) {
                    ws.onopen = options.onopen;
                }
                
                if (options && options.onerror) {
                    ws.onerror = options.onerror;
                }
                
                if (options && options.onclose) {
                    ws.onclose = options.onclose;
                }
                
                return ws;
                
            } catch (error) {
                Log.error("SecurityUtils", "Failed to create secure WebSocket: " + error.message);
                throw error;
            }
        },
        
        createSecureEventSource: function (url, options) {
            if (!url) {
                throw new Error("EventSource URL is required");
            }
            
            const secureUrl = url.replace(/^http:\/\//, 'https://');
            
            if (!secureUrl.startsWith('https://')) {
                throw new Error("Only secure EventSource connections (https://) are allowed");
            }
            
            try {
                const eventSource = new EventSource(secureUrl, options);
                
                const originalAddEventListener = eventSource.addEventListener.bind(eventSource);
                eventSource.addEventListener = (type, listener, options) => {
                    const secureListener = (event) => {
                        try {
                            if (event.data) {
                                const data = JSON.parse(event.data);
                                const validation = this.validateMetadata(data);
                                
                                if (validation.valid) {
                                    event.sanitizedData = validation.sanitized;
                                    listener(event);
                                } else {
                                    Log.warning("SecurityUtils", "EventSource message validation failed: " + validation.error);
                                }
                            } else {
                                listener(event);
                            }
                        } catch (error) {
                            Log.error("SecurityUtils", "EventSource message parsing failed: " + error.message);
                        }
                    };
                    
                    originalAddEventListener(type, secureListener, options);
                };
                
                return eventSource;
                
            } catch (error) {
                Log.error("SecurityUtils", "Failed to create secure EventSource: " + error.message);
                throw error;
            }
        },
        
        encryptSensitiveData: function (data) {
            if (!data) return data;
            
            const sanitizedData = this.sanitizeMetadata(data);
            return btoa(JSON.stringify(sanitizedData));
        },
        
        decryptSensitiveData: function (encryptedData) {
            if (!encryptedData) return null;
            
            try {
                const decryptedData = JSON.parse(atob(encryptedData));
                const validation = this.validateMetadata(decryptedData);
                
                return validation.valid ? validation.sanitized : null;
            } catch (error) {
                Log.error("SecurityUtils", "Failed to decrypt sensitive data: " + error.message);
                return null;
            }
        },
        
        validateExportData: function (data, includePrivate) {
            if (includePrivate === true) {
                Log.warning("SecurityUtils", "Export includes private data - ensure proper authorization");
            }
            
            const validation = this.validateMetadata(data);
            if (!validation.valid) {
                return validation;
            }
            
            const exportData = this.filterSensitiveFields(validation.sanitized);
            return { valid: true, sanitized: exportData };
        },
        
        filterSensitiveFields: function (data) {
            if (typeof data !== 'object' || data === null) {
                return data;
            }
            
            const sensitiveFields = ['password', 'token', 'secret', 'key', 'private', 'confidential'];
            const filtered = {};
            
            for (const key in data) {
                if (data.hasOwnProperty(key)) {
                    const isSensitive = sensitiveFields.some(field => 
                        key.toLowerCase().includes(field)
                    );
                    
                    if (!isSensitive) {
                        filtered[key] = this.filterSensitiveFields(data[key]);
                    } else {
                        filtered[key] = '[REDACTED]';
                    }
                }
            }
            
            return filtered;
        }
    };
});