sap.ui.define([
    "sap/base/Log",
    "sap/m/MessageToast"
], (Log, MessageToast) => {
    "use strict";

    return {
        /**
         * Validates workflow configuration for security vulnerabilities
         * @param {object} workflowConfig - Workflow configuration to validate
         * @returns {object} Validation result with security checks
         */
        validateWorkflowConfig(workflowConfig) {
            const validation = {
                isValid: true,
                errors: [],
                warnings: [],
                sanitizedConfig: {}
            };

            if (!workflowConfig || typeof workflowConfig !== "object") {
                validation.isValid = false;
                validation.errors.push("Workflow configuration is required and must be an object");
                return validation;
            }

            // Check for code injection in workflow steps
            if (workflowConfig.steps && Array.isArray(workflowConfig.steps)) {
                const sanitizedSteps = [];

                workflowConfig.steps.forEach((step, index) => {
                    if (this._containsCodeInjection(JSON.stringify(step))) {
                        validation.isValid = false;
                        validation.errors.push(`Step ${index + 1} contains potentially malicious code`);
                    } else {
                        // Validate step type
                        const allowedStepTypes = ["agent", "condition", "parallel", "sequential", "loop", "transform"];
                        if (!step.type || !allowedStepTypes.includes(step.type)) {
                            validation.warnings.push(`Step ${index + 1} has invalid type: ${step.type}`);
                        }

                        sanitizedSteps.push(this._sanitizeWorkflowStep(step));
                    }
                });

                validation.sanitizedConfig.steps = sanitizedSteps;
            }

            // Validate execution strategy
            if (workflowConfig.executionStrategy) {
                const allowedStrategies = ["sequential", "parallel", "distributed", "failfast", "resilient"];
                if (!allowedStrategies.includes(workflowConfig.executionStrategy)) {
                    validation.warnings.push("Invalid execution strategy");
                }
            }

            // Validate retry policy
            if (workflowConfig.retryPolicy) {
                const maxRetries = parseInt(workflowConfig.retryPolicy.maxRetries, 10);
                if (isNaN(maxRetries) || maxRetries < 0 || maxRetries > 10) {
                    validation.warnings.push("Retry count should be between 0 and 10");
                }
            }

            return validation;
        },

        /**
         * Validates agent assignment for security
         * @param {string} agentId - Agent identifier to validate
         * @param {object} capabilities - Required capabilities
         * @returns {object} Validation result
         */
        validateAgentAssignment(agentId, capabilities) {
            const validation = {
                isValid: true,
                error: "",
                sanitizedAgentId: ""
            };

            if (!agentId || typeof agentId !== "string") {
                validation.isValid = false;
                validation.error = "Agent ID is required and must be a string";
                return validation;
            }

            // Check for injection patterns in agent ID
            if (this._containsInjectionPattern(agentId)) {
                validation.isValid = false;
                validation.error = "Agent ID contains invalid characters";
                return validation;
            }

            // Validate agent ID format (alphanumeric + dashes/underscores)
            const agentIdPattern = /^[a-zA-Z0-9_-]+$/;
            if (!agentIdPattern.test(agentId)) {
                validation.isValid = false;
                validation.error = "Agent ID must be alphanumeric with dashes or underscores only";
                return validation;
            }

            // Sanitize agent ID
            validation.sanitizedAgentId = agentId.replace(/[^a-zA-Z0-9_-]/g, "");

            // Validate capabilities if provided
            if (capabilities && typeof capabilities === "object") {
                // Check for valid capability requirements
                const validCapabilities = ["execution", "analysis", "transformation", "validation", "monitoring"];
                Object.keys(capabilities).forEach(cap => {
                    if (!validCapabilities.includes(cap)) {
                        validation.warnings = validation.warnings || [];
                        validation.warnings.push(`Unknown capability: ${cap}`);
                    }
                });
            }

            return validation;
        },

        /**
         * Validates task queue data for security
         * @param {object} taskData - Task data to validate
         * @returns {object} Validation result
         */
        validateTaskData(taskData) {
            const validation = {
                isValid: true,
                errors: [],
                warnings: [],
                sanitizedData: {}
            };

            if (!taskData || typeof taskData !== "object") {
                validation.isValid = false;
                validation.errors.push("Task data is required and must be an object");
                return validation;
            }

            // Validate task type
            if (taskData.type) {
                const allowedTypes = ["process", "transform", "validate", "aggregate", "notify"];
                if (!allowedTypes.includes(taskData.type)) {
                    validation.errors.push("Invalid task type");
                    validation.isValid = false;
                }
            }

            // Check for code injection in task payload
            if (taskData.payload && this._containsCodeInjection(JSON.stringify(taskData.payload))) {
                validation.isValid = false;
                validation.errors.push("Task payload contains potentially malicious content");
                return validation;
            }

            // Validate priority
            if (taskData.priority) {
                const priority = parseInt(taskData.priority, 10);
                if (isNaN(priority) || priority < 0 || priority > 10) {
                    validation.warnings.push("Priority should be between 0 and 10");
                    taskData.priority = Math.min(Math.max(priority || 5, 0), 10);
                }
            }

            // Sanitize task data
            validation.sanitizedData = {
                type: taskData.type,
                priority: taskData.priority || 5,
                payload: this._sanitizePayload(taskData.payload),
                metadata: this._sanitizeMetadata(taskData.metadata)
            };

            return validation;
        },

        /**
         * Validates event data for security
         * @param {string} eventType - Event type to validate
         * @param {object} eventData - Event data to validate
         * @returns {object} Validation result
         */
        validateEventData(eventType, eventData) {
            const validation = {
                isValid: true,
                error: "",
                sanitizedEvent: {}
            };

            // Validate event type
            const allowedEventTypes = [
                "workflow.started",
                "workflow.completed",
                "workflow.failed",
                "step.started",
                "step.completed",
                "step.failed",
                "agent.assigned",
                "agent.completed",
                "queue.updated",
                "error.occurred"
            ];

            if (!allowedEventTypes.includes(eventType)) {
                validation.isValid = false;
                validation.error = "Invalid event type";
                return validation;
            }

            // Check for injection in event data
            if (eventData && this._containsCodeInjection(JSON.stringify(eventData))) {
                validation.isValid = false;
                validation.error = "Event data contains potentially malicious content";
                return validation;
            }

            validation.sanitizedEvent = {
                type: eventType,
                data: this._sanitizeEventData(eventData),
                timestamp: new Date().toISOString()
            };

            return validation;
        },

        /**
         * Validates pipeline configuration for security
         * @param {object} pipelineConfig - Pipeline configuration to validate
         * @returns {object} Validation result
         */
        validatePipelineConfig(pipelineConfig) {
            const validation = {
                isValid: true,
                errors: [],
                warnings: [],
                sanitizedConfig: {}
            };

            if (!pipelineConfig || typeof pipelineConfig !== "object") {
                validation.isValid = false;
                validation.errors.push("Pipeline configuration is required");
                return validation;
            }

            // Validate pipeline steps
            if (pipelineConfig.steps && Array.isArray(pipelineConfig.steps)) {
                pipelineConfig.steps.forEach((step, index) => {
                    // Check for function constructor or eval
                    if (step.handler && typeof step.handler === "string") {
                        if (this._containsCodeInjection(step.handler)) {
                            validation.isValid = false;
                            validation.errors.push(`Pipeline step ${index + 1} contains unsafe code`);
                        }
                    }
                });
            }

            // Validate pipeline triggers
            if (pipelineConfig.triggers) {
                const allowedTriggers = ["manual", "scheduled", "event", "webhook", "condition"];
                pipelineConfig.triggers.forEach(trigger => {
                    if (!allowedTriggers.includes(trigger.type)) {
                        validation.warnings.push(`Unknown trigger type: ${trigger.type}`);
                    }
                });
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
        secureCallFunction(model, functionName, parameters) {
            return new Promise((resolve, reject) => {
                // First, refresh security token
                model.refreshSecurityToken((tokenData) => {
                    // Add CSRF token to headers if not already present
                    const headers = parameters.headers || {};
                    if (!headers["X-CSRF-Token"] && tokenData) {
                        headers["X-CSRF-Token"] = tokenData;
                    }

                    // Enhanced parameters with security
                    const secureParams = {
                        ...parameters,
                        headers,
                        success: (data) => {
                            this.logSecureOperation(functionName, "SUCCESS");
                            if (parameters.success) {
                                parameters.success(data);
                            }
                            resolve(data);
                        },
                        error: (error) => {
                            this.logSecureOperation(functionName, "ERROR", error);
                            if (parameters.error) {
                                parameters.error(error);
                            }
                            reject(error);
                        }
                    };

                    model.callFunction(functionName, secureParams);
                }, (error) => {
                    this.logSecureOperation(functionName, "TOKEN_ERROR", error);
                    reject(new Error("Failed to obtain CSRF token"));
                });
            });
        },

        /**
         * Creates secure WebSocket connection for orchestration updates
         * @param {string} url - WebSocket URL
         * @param {object} options - Connection options
         * @returns {WebSocket|null} Secure WebSocket connection
         */
        createSecureWebSocket(url, options) {
            try {
                // Ensure secure protocol
                const secureUrl = url.replace(/^ws:\/\//, "wss://").replace(/^http:\/\//, "https://");

                // Validate URL
                if (!this._isValidWebSocketUrl(secureUrl)) {
                    Log.error("Invalid WebSocket URL", secureUrl);
                    return null;
                }

                const ws = new WebSocket(secureUrl);

                // Add security event handlers
                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        // Validate incoming data
                        if (this._isValidOrchestrationUpdate(data)) {
                            if (options.onmessage) {
                                options.onmessage(event);
                            }
                        } else {
                            Log.warning("Invalid orchestration update data received");
                        }
                    } catch (error) {
                        Log.error("Invalid WebSocket message format", error);
                    }
                };

                ws.onerror = (error) => {
                    this.logSecureOperation("WEBSOCKET_ERROR", "ERROR", error);
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
         * Creates secure EventSource for orchestration events
         * @param {string} url - EventSource URL
         * @param {object} options - Connection options
         * @returns {EventSource|null} Secure EventSource connection
         */
        createSecureEventSource(url, options) {
            try {
                // Ensure secure protocol
                const secureUrl = url.replace(/^http:\/\//, "https://");

                // Validate URL
                if (!this._isValidEventSourceUrl(secureUrl)) {
                    Log.error("Invalid EventSource URL", secureUrl);
                    return null;
                }

                const eventSource = new EventSource(secureUrl);

                // Add security handlers for different event types
                const secureEventHandler = (eventType) => {
                    return (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            if (this._isValidOrchestrationEvent(eventType, data)) {
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

                // Add handlers for common orchestration events
                ["workflow-update", "agent-status", "task-progress",
                    "pipeline-status"].forEach(eventType => {
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
         * Checks authorization for orchestration operations
         * @param {string} operation - Operation to check
         * @param {object} context - Operation context
         * @returns {boolean} True if authorized
         */
        checkOrchestrationAuth(operation, context) {
            // Check if user has required permissions
            const user = this._getCurrentUser();
            if (!user) {
                MessageToast.show("Authentication required");
                return false;
            }

            // Check operation-specific permissions
            const requiredPermissions = this._getOrchestrationPermissions(operation);
            const hasPermission = requiredPermissions.every(permission =>
                this._userHasPermission(user, permission)
            );

            if (!hasPermission) {
                MessageToast.show("Insufficient permissions for this operation");
                this.logSecureOperation(operation, "UNAUTHORIZED", { user: user.id });
            }

            return hasPermission;
        },

        /**
         * Logs secure operations for audit trail
         * @param {string} operation - Operation name
         * @param {string} status - Operation status
         * @param {object} details - Additional details
         */
        logSecureOperation(operation, status, details) {
            const logEntry = {
                timestamp: new Date().toISOString(),
                operation,
                status,
                user: this._getCurrentUser()?.id || "anonymous",
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
        _containsCodeInjection(str) {
            if (!str || typeof str !== "string") {return false;}

            const codePatterns = [
                /eval\s*\(/gi,
                /Function\s*\(/gi,
                /setTimeout\s*\([^,]+,\s*0\s*\)/gi,
                /setInterval\s*\([^,]+,/gi,
                /__proto__/gi,
                /constructor\s*\[/gi,
                /import\s*\(/gi,
                /require\s*\(/gi,
                /\$\{.*\}/g // Template literals that might execute code
            ];

            return codePatterns.some(pattern => pattern.test(str));
        },

        _containsInjectionPattern(str) {
            const injectionPatterns = [
                /[<>'"]/g,
                /javascript:/gi,
                /on\w+\s*=/gi,
                /\.\.\//g
            ];

            return injectionPatterns.some(pattern => pattern.test(str));
        },

        _sanitizeWorkflowStep(step) {
            return {
                type: step.type,
                name: this._sanitizeString(step.name),
                agentId: this._sanitizeString(step.agentId),
                config: this._sanitizeConfig(step.config)
            };
        },

        _sanitizePayload(payload) {
            if (!payload) {return {};}

            const sanitized = {};
            Object.keys(payload).forEach(key => {
                const sanitizedKey = this._sanitizeString(key);
                if (typeof payload[key] === "string") {
                    sanitized[sanitizedKey] = this._sanitizeString(payload[key]);
                } else if (typeof payload[key] === "object") {
                    sanitized[sanitizedKey] = this._sanitizePayload(payload[key]);
                } else {
                    sanitized[sanitizedKey] = payload[key];
                }
            });

            return sanitized;
        },

        _sanitizeMetadata(metadata) {
            if (!metadata) {return {};}

            return {
                createdBy: this._sanitizeString(metadata.createdBy),
                createdAt: metadata.createdAt,
                source: this._sanitizeString(metadata.source),
                tags: Array.isArray(metadata.tags) ?
                    metadata.tags.map(tag => this._sanitizeString(tag)) : []
            };
        },

        _sanitizeEventData(eventData) {
            if (!eventData) {return {};}

            const sanitized = {};
            Object.keys(eventData).forEach(key => {
                if (typeof eventData[key] === "string") {
                    sanitized[key] = this._sanitizeString(eventData[key]);
                } else if (typeof eventData[key] === "number" || typeof eventData[key] === "boolean") {
                    sanitized[key] = eventData[key];
                }
            });

            return sanitized;
        },

        _sanitizeConfig(config) {
            if (!config) {return {};}

            const sanitized = {};
            Object.keys(config).forEach(key => {
                const value = config[key];
                if (typeof value === "string") {
                    sanitized[key] = this._sanitizeString(value);
                } else if (typeof value === "number" || typeof value === "boolean") {
                    sanitized[key] = value;
                } else if (Array.isArray(value)) {
                    sanitized[key] = value.map(item =>
                        typeof item === "string" ? this._sanitizeString(item) : item
                    );
                }
            });

            return sanitized;
        },

        _sanitizeString(str) {
            if (!str || typeof str !== "string") {return "";}

            return str
                .replace(/[<>'"]/g, "")
                .replace(/javascript:/gi, "")
                .replace(/on\w+\s*=/gi, "")
                .substring(0, 1000); // Limit length
        },

        _isValidWebSocketUrl(url) {
            try {
                const urlObj = new URL(url);
                return urlObj.protocol === "wss:" ||
                       (urlObj.protocol === "ws:" && urlObj.hostname === "localhost");
            } catch (error) {
                return false;
            }
        },

        _isValidEventSourceUrl(url) {
            try {
                const urlObj = new URL(url);
                return urlObj.protocol === "https:" ||
                       (urlObj.protocol === "http:" && urlObj.hostname === "localhost");
            } catch (error) {
                return false;
            }
        },

        _isValidOrchestrationUpdate(data) {
            if (!data || typeof data !== "object") {
                return false;
            }

            const validTypes = [
                "WORKFLOW_STARTED", "WORKFLOW_COMPLETED", "WORKFLOW_FAILED",
                "STEP_STARTED", "STEP_COMPLETED", "STEP_FAILED",
                "AGENT_ASSIGNED", "AGENT_COMPLETED",
                "QUEUE_UPDATED", "PIPELINE_STATUS"
            ];

            return data.type && validTypes.includes(data.type);
        },

        _isValidOrchestrationEvent(eventType, data) {
            const validEventTypes = {
                "workflow-update": ["id", "status", "progress"],
                "agent-status": ["agentId", "status", "capacity"],
                "task-progress": ["taskId", "progress", "status"],
                "pipeline-status": ["pipelineId", "stage", "status"]
            };

            if (!validEventTypes[eventType]) {
                return false;
            }

            // Check if required fields are present
            const requiredFields = validEventTypes[eventType];
            return requiredFields.every(field => data.hasOwnProperty(field));
        },

        _getCurrentUser() {
            // Mock user detection - implement actual user detection
            return {
                id: "current-user",
                permissions: ["orchestration:read", "orchestration:write",
                    "orchestration:execute", "orchestration:admin"]
            };
        },

        _getOrchestrationPermissions(operation) {
            const permissionMap = {
                "CreateWorkflow": ["orchestration:write"],
                "ExecuteWorkflow": ["orchestration:execute"],
                "DeleteWorkflow": ["orchestration:admin"],
                "AssignAgent": ["orchestration:write"],
                "ModifyPipeline": ["orchestration:admin"],
                "ViewDashboard": ["orchestration:read"],
                "ConfigureCircuitBreaker": ["orchestration:admin"],
                "OverrideConsensus": ["orchestration:admin"]
            };

            return permissionMap[operation] || ["orchestration:read"];
        },

        _userHasPermission(user, permission) {
            return user.permissions && user.permissions.includes(permission);
        },

        _isProduction() {
            // Detect production environment
            return window.location.hostname !== "localhost" &&
                   window.location.hostname !== "127.0.0.1" &&
                   !window.location.hostname.startsWith("192.168.");
        },

        _sendToAuditService(logEntry) {
            // In production, implement actual audit service integration
            // console.log("AUDIT:", logEntry);
        }
    };
});