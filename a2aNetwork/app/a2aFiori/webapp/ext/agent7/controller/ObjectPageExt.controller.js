sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "sap/base/security/encodeXML",
    "sap/base/security/encodeURL",
    "sap/base/Log",
    "sap/ui/core/routing/Router",
    "sap/base/strings/escapeRegExp",
    "sap/base/security/sanitizeHTML"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, encodeXML, encodeURL, Log, Router, escapeRegExp, sanitizeHTML) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent7.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._initializeSecurity();
                
                // Initialize device model for responsive behavior
                var oDeviceModel = new JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");
                
                // Initialize create dialog model
                this._initializeCreateModel();
            },
            
            onExit: function() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },
        
        _initializeSecurity: function() {
            this._auditLogger = {
                log: function(action, details) {
                    const user = this._getCurrentUser();
                    const timestamp = new Date().toISOString();
                    const logEntry = {
                        timestamp: timestamp,
                        user: user,
                        agent: "Agent7_AgentManager",
                        action: action,
                        details: details
                    };
                    Log.info("AUDIT: " + JSON.stringify(logEntry));
                }.bind(this)
            };
        },
        
        _getCurrentUser: function() {
            return sap.ushell?.Container?.getUser()?.getId() || "anonymous";
        },
        
        _hasRole: function(role) {
            const user = sap.ushell?.Container?.getUser();
            return user && user.hasRole && user.hasRole(role);
        },
        
        _validateInput: function(input, type) {
            if (!input || typeof input !== 'string') return false;
            
            switch(type) {
                case 'agentId':
                    return /^[a-zA-Z0-9\-_]{1,36}$/.test(input);
                case 'operation':
                    return /^[A-Z_]{1,20}$/.test(input);
                case 'version':
                    return /^\d+\.\d+\.\d+$/.test(input);
                case 'url':
                    return /^https?\/\/[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$/.test(input);
                default:
                    return input.length > 0 && input.length <= 255;
            }
        },
        
        _sanitizeInput: function(input) {
            if (!input) return "";
            return encodeXML(input.toString().trim());
        },
        
        _validateAgentId: function(agentId) {
            return this._validateInput(agentId, 'agentId');
        },
        
        _authorizeOperation: function(operation, agentId) {
            const allowedOperations = ['START', 'STOP', 'RESTART', 'UPDATE', 'CONFIGURE'];
            return allowedOperations.includes(operation) && this._hasRole('AgentOperator');
        },
        
        _validateEventSourceUrl: function(url) {
            return url && url.startsWith('/a2a/agent7/v1/') && !url.includes('..');
        },
        
        _getCsrfToken: function() {
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: "/a2a/agent7/v1/csrf-token",
                    type: "GET",
                    headers: {
                        "X-CSRF-Token": "Fetch"
                    },
                    success: function(data, textStatus, xhr) {
                        const token = xhr.getResponseHeader("X-CSRF-Token");
                        resolve(token);
                    },
                    error: function(xhr) {
                        reject(new Error("Failed to fetch CSRF token"));
                    }
                });
            });
        },
        
        _secureAjaxCall: function(options) {
            return this._getCsrfToken().then(token => {
                return new Promise((resolve, reject) => {
                    const secureOptions = Object.assign({}, options, {
                        headers: Object.assign({
                            "X-CSRF-Token": token,
                            "Content-Type": "application/json"
                        }, options.headers || {}),
                        success: function(data, textStatus, xhr) {
                            resolve({ data, textStatus, xhr });
                        },
                        error: function(xhr, textStatus, errorThrown) {
                            reject({ xhr, textStatus, errorThrown });
                        }
                    });
                    
                    jQuery.ajax(secureOptions);
                });
            });
        },
        
        _cleanupResources: function() {
            if (this._operationEventSource) {
                this._operationEventSource.close();
                this._operationEventSource = null;
            }
            if (this._streamMetricsEventSource) {
                this._streamMetricsEventSource.close();
                this._streamMetricsEventSource = null;
            }
            if (this._streamMonitoringEventSource) {
                this._streamMonitoringEventSource.close();
                this._streamMonitoringEventSource = null;
            }
            
            // Clean up cached dialogs
            if (this._dialogCache) {
                Object.keys(this._dialogCache).forEach(function(key) {
                    if (this._dialogCache[key]) {
                        this._dialogCache[key].destroy();
                    }
                }.bind(this));
                this._dialogCache = {};
            }
            
            // Clean up legacy dialog references
            if (this._oUpdateDialog) {
                this._oUpdateDialog.destroy();
                this._oUpdateDialog = null;
            }
            if (this._oHealthResultsDialog) {
                this._oHealthResultsDialog.destroy();
                this._oHealthResultsDialog = null;
            }
            if (this._oConfigDialog) {
                this._oConfigDialog.destroy();
                this._oConfigDialog = null;
            }
            if (this._oLogsDialog) {
                this._oLogsDialog.destroy();
                this._oLogsDialog = null;
            }
            if (this._oCreateDialog) {
                this._oCreateDialog.destroy();
                this._oCreateDialog = null;
            }
        },
        
        _initializeCreateModel: function() {
            this._oCreateModel = new JSONModel({
                taskName: "",
                description: "",
                managedAgent: "",
                operationType: "",
                priority: "MEDIUM",
                executionMode: "IMMEDIATE",
                scheduledTime: null,
                timeout: 30,
                retryAttempts: 3,
                forceExecution: false,
                createBackup: true,
                rollbackOnFailure: true,
                notifyOnCompletion: true,
                dependencyType: "NONE",
                requiredAgentStatus: "ONLINE",
                prerequisiteTasks: "",
                minCpuAvailable: 20,
                minMemoryAvailable: 30,
                timeWindowStart: null,
                timeWindowEnd: null,
                monitoringLevel: "DETAILED",
                healthCheckInterval: 30,
                monitorCpu: true,
                monitorMemory: true,
                monitorNetwork: true,
                monitorResponse: true,
                cpuAlertThreshold: 80,
                memoryAlertThreshold: 85,
                responseTimeAlert: 2000,
                emailNotifications: true,
                smsNotifications: false,
                slackNotifications: false,
                parameters: [],
                availableAgents: [],
                // Validation states
                taskNameState: "None",
                managedAgentState: "None",
                operationTypeState: "None",
                taskNameStateText: "",
                managedAgentStateText: "",
                operationTypeStateText: ""
            });
        },

        onStartAgent: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = this._sanitizeInput(oContext.getProperty("managedAgent"));
            var sAgentName = this._sanitizeInput(oContext.getProperty("taskName"));
            
            if (!this._validateAgentId(sAgentId)) {
                MessageBox.error("Invalid agent ID format");
                return;
            }
            
            if (!this._authorizeOperation("START", sAgentId)) {
                MessageBox.error("Access denied: Insufficient privileges for starting agents");
                this._auditLogger.log("START_AGENT_ACCESS_DENIED", { agentId: sAgentId });
                return;
            }
            
            MessageBox.confirm("Start agent '" + sAgentName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._executeAgentOperation(sAgentId, "START");
                    }
                }.bind(this)
            });
        },

        onStopAgent: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = this._sanitizeInput(oContext.getProperty("managedAgent"));
            var sAgentName = this._sanitizeInput(oContext.getProperty("taskName"));
            
            if (!this._validateAgentId(sAgentId)) {
                MessageBox.error("Invalid agent ID format");
                return;
            }
            
            if (!this._authorizeOperation("STOP", sAgentId)) {
                MessageBox.error("Access denied: Insufficient privileges for stopping agents");
                this._auditLogger.log("STOP_AGENT_ACCESS_DENIED", { agentId: sAgentId });
                return;
            }
            
            MessageBox.confirm("Stop agent '" + sAgentName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._executeAgentOperation(sAgentId, "STOP");
                    }
                }.bind(this)
            });
        },

        onRestartAgent: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = this._sanitizeInput(oContext.getProperty("managedAgent"));
            var sAgentName = this._sanitizeInput(oContext.getProperty("taskName"));
            
            if (!this._validateAgentId(sAgentId)) {
                MessageBox.error("Invalid agent ID format");
                return;
            }
            
            if (!this._authorizeOperation("RESTART", sAgentId)) {
                MessageBox.error("Access denied: Insufficient privileges for restarting agents");
                this._auditLogger.log("RESTART_AGENT_ACCESS_DENIED", { agentId: sAgentId });
                return;
            }
            
            MessageBox.confirm("Restart agent '" + sAgentName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._executeAgentOperation(sAgentId, "RESTART");
                    }
                }.bind(this)
            });
        },

        _executeAgentOperation: function(sAgentId, sOperation) {
            this._extensionAPI.getView().setBusy(true);
            
            const requestData = {
                operation: sOperation,
                timestamp: new Date().toISOString()
            };
            
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/agents/" + encodeURL(sAgentId) + "/operations",
                type: "POST",
                data: JSON.stringify(requestData)
            }).then(result => {
                this._extensionAPI.getView().setBusy(false);
                
                const data = result.data;
                var sMessage = sOperation.toLowerCase() + " operation initiated successfully";
                if (data.estimatedTime) {
                    sMessage += "\nEstimated completion: " + this._sanitizeInput(data.estimatedTime) + " seconds";
                }
                
                MessageToast.show(sMessage);
                this._extensionAPI.refresh();
                
                if (data.monitoringUrl) {
                    this._startOperationMonitoring(data.operationId);
                }
                
                this._auditLogger.log("AGENT_OPERATION_EXECUTED", {
                    agentId: sAgentId,
                    operation: sOperation,
                    operationId: data.operationId
                });
            }).catch(error => {
                this._extensionAPI.getView().setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Operation failed: " + errorMsg);
                this._auditLogger.log("AGENT_OPERATION_FAILED", {
                    agentId: sAgentId,
                    operation: sOperation,
                    error: errorMsg
                });
            });
        },

        _startOperationMonitoring: function(sOperationId) {
            const streamUrl = "/a2a/agent7/v1/operations/" + encodeURL(sOperationId) + "/stream";
            
            if (!this._validateEventSourceUrl(streamUrl)) {
                MessageBox.error("Invalid operation monitoring stream URL");
                return;
            }
            
            this._operationEventSource = new EventSource(streamUrl);
            
            this._operationEventSource.onmessage = function(event) {
                try {
                    var data = JSON.parse(event.data);
                    
                    if (data.type === "progress") {
                        const stage = this._sanitizeInput(data.stage);
                        const progress = Math.max(0, Math.min(100, parseInt(data.progress) || 0));
                        MessageToast.show(stage + ": " + progress + "%");
                    } else if (data.type === "complete") {
                        this._operationEventSource.close();
                        this._extensionAPI.refresh();
                        MessageBox.success("Operation completed successfully!");
                        this._auditLogger.log("OPERATION_COMPLETED", { operationId: sOperationId });
                    } else if (data.type === "error") {
                        this._operationEventSource.close();
                        const errorMsg = this._sanitizeInput(data.error || "Unknown error");
                        MessageBox.error("Operation failed: " + errorMsg);
                        this._auditLogger.log("OPERATION_FAILED", { operationId: sOperationId, error: errorMsg });
                    }
                } catch (e) {
                    this._operationEventSource.close();
                    MessageBox.error("Invalid data received from operation stream");
                    this._auditLogger.log("OPERATION_STREAM_ERROR", { operationId: sOperationId, error: e.message });
                }
            }.bind(this);
            
            this._operationEventSource.onerror = function() {
                if (this._operationEventSource) {
                    this._operationEventSource.close();
                    this._operationEventSource = null;
                }
                MessageBox.error("Lost connection to operation monitoring");
                this._auditLogger.log("OPERATION_MONITORING_CONNECTION_LOST", { operationId: sOperationId });
            }.bind(this);
        },

        onUpdateAgent: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = this._sanitizeInput(oContext.getProperty("managedAgent"));
            
            if (!this._validateAgentId(sAgentId)) {
                MessageBox.error("Invalid agent ID format");
                return;
            }
            
            if (!this._oUpdateDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent7.ext.fragment.UpdateAgent",
                    controller: this
                }).then(function(oDialog) {
                    this._oUpdateDialog = oDialog;
                    this.base.getView().addDependent(this._oUpdateDialog);
                    
                    var oModel = new JSONModel({
                        agentId: sAgentId,
                        updateType: "MINOR",
                        version: "",
                        rollbackEnabled: true,
                        restartAfterUpdate: true,
                        backupBeforeUpdate: true
                    });
                    this._oUpdateDialog.setModel(oModel, "update");
                    this._oUpdateDialog.open();
                    
                    this._loadAvailableVersions(sAgentId);
                }.bind(this));
            } else {
                this._oUpdateDialog.open();
                this._loadAvailableVersions(sAgentId);
            }
        },

        _loadAvailableVersions: function(sAgentId) {
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/agents/" + encodeURL(sAgentId) + "/available-versions",
                type: "GET"
            }).then(result => {
                var oModel = this._oUpdateDialog.getModel("update");
                var oData = oModel.getData();
                oData.availableVersions = this._sanitizeArray(result.data.versions);
                oData.currentVersion = this._sanitizeInput(result.data.currentVersion);
                oModel.setData(oData);
                
                this._auditLogger.log("AVAILABLE_VERSIONS_LOADED", { agentId: sAgentId });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to load available versions: " + errorMsg);
                this._auditLogger.log("AVAILABLE_VERSIONS_LOAD_FAILED", { agentId: sAgentId, error: errorMsg });
            });
        },

        onHealthCheck: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = this._sanitizeInput(oContext.getProperty("managedAgent"));
            
            if (!this._validateAgentId(sAgentId)) {
                MessageBox.error("Invalid agent ID format");
                return;
            }
            
            this._extensionAPI.getView().setBusy(true);
            
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/agents/" + encodeURL(sAgentId) + "/health-check",
                type: "POST"
            }).then(result => {
                this._extensionAPI.getView().setBusy(false);
                this._showHealthCheckResults(result.data);
                
                this._auditLogger.log("HEALTH_CHECK_EXECUTED", { agentId: sAgentId });
            }).catch(error => {
                this._extensionAPI.getView().setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Health check failed: " + errorMsg);
                this._auditLogger.log("HEALTH_CHECK_FAILED", { agentId: sAgentId, error: errorMsg });
            });
        },

        _showHealthCheckResults: function(healthData) {
            var oView = this.base.getView();
            
            if (!this._oHealthResultsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent7.ext.fragment.HealthCheckResults",
                    controller: this
                }).then(function(oDialog) {
                    this._oHealthResultsDialog = oDialog;
                    oView.addDependent(this._oHealthResultsDialog);
                    
                    var oModel = new JSONModel(this._sanitizeObject(healthData));
                    this._oHealthResultsDialog.setModel(oModel, "health");
                    this._oHealthResultsDialog.open();
                    
                    this._createHealthVisualizations(healthData);
                }.bind(this));
            } else {
                var oModel = new JSONModel(this._sanitizeObject(healthData));
                this._oHealthResultsDialog.setModel(oModel, "health");
                this._oHealthResultsDialog.open();
                this._createHealthVisualizations(healthData);
            }
        },

        _createHealthVisualizations: function(data) {
            var oRadarChart = this._oHealthResultsDialog.byId("healthRadarChart");
            if (!oRadarChart || !data.metrics) return;
            
            const sanitizedMetrics = this._sanitizeObject(data.metrics);
            var aRadarData = Object.keys(sanitizedMetrics).map(function(key) {
                return {
                    Metric: this._sanitizeInput(key),
                    Score: sanitizedMetrics[key]
                };
            }.bind(this));
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                healthData: aRadarData
            });
            oRadarChart.setModel(oChartModel);
        },

        onPerformanceCheck: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = this._sanitizeInput(oContext.getProperty("managedAgent"));
            
            if (!this._validateAgentId(sAgentId)) {
                MessageBox.error("Invalid agent ID format");
                return;
            }
            
            this._extensionAPI.getView().setBusy(true);
            
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/agents/" + encodeURL(sAgentId) + "/performance-check",
                type: "POST"
            }).then(result => {
                this._extensionAPI.getView().setBusy(false);
                this._showPerformanceResults(result.data);
                
                this._auditLogger.log("PERFORMANCE_CHECK_EXECUTED", { agentId: sAgentId });
            }).catch(error => {
                this._extensionAPI.getView().setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Performance check failed: " + errorMsg);
                this._auditLogger.log("PERFORMANCE_CHECK_FAILED", { agentId: sAgentId, error: errorMsg });
            });
        },

        _showPerformanceResults: function(performanceData) {
            const sanitizedData = this._sanitizeObject(performanceData);
            var sMessage = "Performance Check Results:\n\n";
            
            sMessage += "Response Time: " + sanitizedData.responseTime + "ms\n";
            sMessage += "Throughput: " + sanitizedData.throughput + " req/sec\n";
            sMessage += "Memory Usage: " + sanitizedData.memoryUsage + "%\n";
            sMessage += "CPU Usage: " + sanitizedData.cpuUsage + "%\n";
            sMessage += "Error Rate: " + sanitizedData.errorRate + "%\n\n";
            
            if (sanitizedData.recommendations && sanitizedData.recommendations.length > 0) {
                sMessage += "Recommendations:\n";
                sanitizedData.recommendations.forEach(function(rec) {
                    sMessage += "• " + rec + "\n";
                });
            }
            
            MessageBox.information(sMessage);
        },

        onConfigureAgent: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = this._sanitizeInput(oContext.getProperty("managedAgent"));
            
            if (!this._validateAgentId(sAgentId)) {
                MessageBox.error("Invalid agent ID format");
                return;
            }
            
            if (!this._oConfigDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent7.ext.fragment.AgentConfiguration",
                    controller: this
                }).then(function(oDialog) {
                    this._oConfigDialog = oDialog;
                    this.base.getView().addDependent(this._oConfigDialog);
                    this._oConfigDialog.open();
                    this._loadAgentConfiguration(sAgentId);
                }.bind(this));
            } else {
                this._oConfigDialog.open();
                this._loadAgentConfiguration(sAgentId);
            }
        },

        _loadAgentConfiguration: function(sAgentId) {
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/agents/" + encodeURL(sAgentId) + "/configuration",
                type: "GET"
            }).then(result => {
                var oModel = new JSONModel({
                    agentId: sAgentId,
                    configuration: this._sanitizeObject(result.data.configuration),
                    schema: this._sanitizeObject(result.data.schema),
                    modifiedFields: []
                });
                this._oConfigDialog.setModel(oModel, "config");
                
                this._auditLogger.log("AGENT_CONFIGURATION_LOADED", { agentId: sAgentId });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to load configuration: " + errorMsg);
                this._auditLogger.log("AGENT_CONFIGURATION_LOAD_FAILED", { agentId: sAgentId, error: errorMsg });
            });
        },

        onSaveConfiguration: function() {
            var oModel = this._oConfigDialog.getModel("config");
            var oData = oModel.getData();
            
            this._oConfigDialog.setBusy(true);
            
            const requestData = {
                configuration: this._sanitizeObject(oData.configuration),
                modifiedFields: this._sanitizeArray(oData.modifiedFields)
            };
            
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/agents/" + encodeURL(oData.agentId) + "/configuration",
                type: "PUT",
                data: JSON.stringify(requestData)
            }).then(result => {
                this._oConfigDialog.setBusy(false);
                this._oConfigDialog.close();
                
                const data = result.data;
                MessageBox.success(
                    "Configuration updated successfully!\n" +
                    "Restart required: " + (data.restartRequired ? "Yes" : "No")
                );
                
                this._extensionAPI.refresh();
                
                this._auditLogger.log("AGENT_CONFIGURATION_UPDATED", {
                    agentId: oData.agentId,
                    restartRequired: data.restartRequired
                });
            }).catch(error => {
                this._oConfigDialog.setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Configuration update failed: " + errorMsg);
                this._auditLogger.log("AGENT_CONFIGURATION_UPDATE_FAILED", {
                    agentId: oData.agentId,
                    error: errorMsg
                });
            });
        },

        onViewLogs: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = this._sanitizeInput(oContext.getProperty("managedAgent"));
            
            if (!this._validateAgentId(sAgentId)) {
                MessageBox.error("Invalid agent ID format");
                return;
            }
            
            if (!this._oLogsDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent7.ext.fragment.AgentLogs",
                    controller: this
                }).then(function(oDialog) {
                    this._oLogsDialog = oDialog;
                    this.base.getView().addDependent(this._oLogsDialog);
                    this._oLogsDialog.open();
                    this._loadAgentLogs(sAgentId);
                }.bind(this));
            } else {
                this._oLogsDialog.open();
                this._loadAgentLogs(sAgentId);
            }
        },

        _loadAgentLogs: function(sAgentId) {
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/agents/" + encodeURL(sAgentId) + "/logs",
                type: "GET",
                data: {
                    limit: 1000,
                    level: "ALL"
                }
            }).then(result => {
                var oModel = new JSONModel({
                    logs: this._sanitizeArray(result.data.logs),
                    logLevels: this._sanitizeArray(result.data.availableLevels),
                    selectedLevel: "ALL"
                });
                this._oLogsDialog.setModel(oModel, "logs");
                
                this._auditLogger.log("AGENT_LOGS_LOADED", { agentId: sAgentId });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to load logs: " + errorMsg);
                this._auditLogger.log("AGENT_LOGS_LOAD_FAILED", { agentId: sAgentId, error: errorMsg });
            });
        },

        onAgentDetails: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = this._sanitizeInput(oContext.getProperty("managedAgent"));
            
            if (!this._validateAgentId(sAgentId)) {
                MessageBox.error("Invalid agent ID format");
                return;
            }
            
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/agents/" + encodeURL(sAgentId) + "/details",
                type: "GET"
            }).then(result => {
                this._showAgentDetails(result.data);
                this._auditLogger.log("AGENT_DETAILS_VIEWED", { agentId: sAgentId });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to load agent details: " + errorMsg);
                this._auditLogger.log("AGENT_DETAILS_LOAD_FAILED", { agentId: sAgentId, error: errorMsg });
            });
        },

        _showAgentDetails: function(agentData) {
            const sanitizedData = this._sanitizeObject(agentData);
            var sMessage = "Agent Details:\n\n";
            
            sMessage += "Name: " + sanitizedData.name + "\n";
            sMessage += "Type: " + sanitizedData.type + "\n";
            sMessage += "Version: " + sanitizedData.version + "\n";
            sMessage += "Status: " + sanitizedData.status + "\n";
            sMessage += "Endpoint: " + sanitizedData.endpoint + "\n";
            sMessage += "Uptime: " + sanitizedData.uptime + "\n";
            sMessage += "Last Health Check: " + sanitizedData.lastHealthCheck + "\n\n";
            
            if (sanitizedData.capabilities && sanitizedData.capabilities.length > 0) {
                sMessage += "Capabilities:\n";
                sanitizedData.capabilities.forEach(function(cap) {
                    sMessage += "• " + cap + "\n";
                });
                sMessage += "\n";
            }
            
            if (sanitizedData.dependencies && sanitizedData.dependencies.length > 0) {
                sMessage += "Dependencies:\n";
                sanitizedData.dependencies.forEach(function(dep) {
                    sMessage += "• " + dep + "\n";
                });
            }
            
            MessageBox.information(sMessage);
        },

        onCoordination: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = this._sanitizeInput(oContext.getProperty("managedAgent"));
            
            if (!this._validateAgentId(sAgentId)) {
                MessageBox.error("Invalid agent ID format");
                return;
            }
            
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/agents/" + encodeURL(sAgentId) + "/coordination-status",
                type: "GET"
            }).then(result => {
                this._showCoordinationStatus(result.data, sAgentId);
                this._auditLogger.log("COORDINATION_STATUS_VIEWED", { agentId: sAgentId });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to load coordination status: " + errorMsg);
                this._auditLogger.log("COORDINATION_STATUS_LOAD_FAILED", { agentId: sAgentId, error: errorMsg });
            });
        },

        _showCoordinationStatus: function(coordinationData, sAgentId) {
            const sanitizedData = this._sanitizeObject(coordinationData);
            var sMessage = "Agent Coordination Status:\n\n";
            
            sMessage += "Coordination Enabled: " + (sanitizedData.enabled ? "Yes" : "No") + "\n";
            sMessage += "Active Connections: " + sanitizedData.activeConnections + "\n";
            sMessage += "Workflow Participation: " + sanitizedData.workflowParticipation + "\n";
            sMessage += "Load Balancing: " + (sanitizedData.loadBalancing ? "Enabled" : "Disabled") + "\n";
            sMessage += "Trust Level: " + sanitizedData.trustLevel + "\n\n";
            
            if (sanitizedData.connectedAgents && sanitizedData.connectedAgents.length > 0) {
                sMessage += "Connected Agents:\n";
                sanitizedData.connectedAgents.forEach(function(agent) {
                    sMessage += "• " + agent.name + " (" + agent.status + ")\n";
                });
            }
            
            MessageBox.information(sMessage, {
                actions: ["Enable Coordination", "Disable Coordination", MessageBox.Action.CLOSE],
                onClose: function(oAction) {
                    if (oAction === "Enable Coordination" || oAction === "Disable Coordination") {
                        this._toggleCoordination(sAgentId, oAction === "Enable Coordination");
                    }
                }.bind(this)
            });
        },

        _toggleCoordination: function(sAgentId, bEnable) {
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/agents/" + encodeURL(sAgentId) + "/coordination",
                type: "PUT",
                data: JSON.stringify({ enabled: bEnable })
            }).then(result => {
                MessageToast.show("Coordination " + (bEnable ? "enabled" : "disabled") + " successfully");
                this._extensionAPI.refresh();
                
                this._auditLogger.log("COORDINATION_TOGGLED", {
                    agentId: sAgentId,
                    enabled: bEnable
                });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to toggle coordination: " + errorMsg);
                this._auditLogger.log("COORDINATION_TOGGLE_FAILED", {
                    agentId: sAgentId,
                    enabled: bEnable,
                    error: errorMsg
                });
            });
        },

        onConfirmUpdate: function() {
            var oModel = this._oUpdateDialog.getModel("update");
            var oData = oModel.getData();
            
            if (!this._validateInput(oData.version, 'version')) {
                MessageBox.error("Please select a valid version to update to");
                return;
            }
            
            const sanitizedData = {
                agentId: this._sanitizeInput(oData.agentId),
                updateType: this._sanitizeInput(oData.updateType),
                version: this._sanitizeInput(oData.version),
                rollbackEnabled: Boolean(oData.rollbackEnabled),
                restartAfterUpdate: Boolean(oData.restartAfterUpdate),
                backupBeforeUpdate: Boolean(oData.backupBeforeUpdate)
            };
            
            this._oUpdateDialog.setBusy(true);
            
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/agents/" + encodeURL(sanitizedData.agentId) + "/update",
                type: "POST",
                data: JSON.stringify(sanitizedData)
            }).then(result => {
                this._oUpdateDialog.setBusy(false);
                this._oUpdateDialog.close();
                
                const data = result.data;
                MessageBox.success(
                    "Agent update initiated!\n" +
                    "Update ID: " + this._sanitizeInput(data.updateId) + "\n" +
                    "Estimated time: " + this._sanitizeInput(data.estimatedTime) + " minutes"
                );
                
                this._extensionAPI.refresh();
                this._startOperationMonitoring(data.updateId);
                
                this._auditLogger.log("AGENT_UPDATE_INITIATED", {
                    agentId: sanitizedData.agentId,
                    version: sanitizedData.version,
                    updateId: data.updateId
                });
            }).catch(error => {
                this._oUpdateDialog.setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Agent update failed: " + errorMsg);
                this._auditLogger.log("AGENT_UPDATE_FAILED", {
                    agentId: sanitizedData.agentId,
                    error: errorMsg
                });
            });
        },
        
        _sanitizeObject: function(obj) {
            if (!obj || typeof obj !== 'object') return {};
            const sanitized = {};
            Object.keys(obj).forEach(key => {
                if (typeof obj[key] === 'string') {
                    sanitized[key] = this._sanitizeInput(obj[key]);
                } else if (Array.isArray(obj[key])) {
                    sanitized[key] = this._sanitizeArray(obj[key]);
                } else if (typeof obj[key] === 'object') {
                    sanitized[key] = this._sanitizeObject(obj[key]);
                } else {
                    sanitized[key] = obj[key];
                }
            });
            return sanitized;
        },
        
        _sanitizeArray: function(arr) {
            if (!Array.isArray(arr)) return [];
            return arr.map(item => {
                if (typeof item === 'string') {
                    return this._sanitizeInput(item);
                } else if (typeof item === 'object') {
                    return this._sanitizeObject(item);
                } else {
                    return item;
                }
            });
        },
        
        // Create Management Task Dialog Methods
        onCreateManagementTask: function() {
            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent7.ext.fragment.CreateManagementTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._oCreateDialog.setModel(this._oCreateModel, "create");
                    this._oCreateDialog.open();
                    
                    // Load available agents
                    this._loadAvailableAgents();
                    
                    this._auditLogger.log("CREATE_MANAGEMENT_DIALOG_OPENED", { action: "create_management_task" });
                }.bind(this));
            } else {
                this._oCreateDialog.open();
                this._loadAvailableAgents();
            }
        },
        
        _loadAvailableAgents: function() {
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/agents",
                type: "GET"
            }).then(result => {
                const sanitizedAgents = this._sanitizeArray(result.data || []);
                this._oCreateModel.setProperty("/availableAgents", sanitizedAgents);
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                Log.warning("Failed to load available agents: " + errorMsg);
                // Set empty array as fallback
                this._oCreateModel.setProperty("/availableAgents", []);
            });
        },
        
        onCancelCreateTask: function() {
            this._oCreateDialog.close();
        },
        
        onConfirmCreateTask: function() {
            var oData = this._oCreateModel.getData();
            
            // Validate form
            if (!this._validateForm()) {
                MessageBox.error("Please correct the validation errors before creating the task.");
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            var oSanitizedData = {
                taskName: this._sanitizeInput(oData.taskName),
                description: this._sanitizeInput(oData.description),
                managedAgent: this._sanitizeInput(oData.managedAgent),
                operationType: this._sanitizeInput(oData.operationType),
                priority: oData.priority,
                executionMode: oData.executionMode,
                scheduledTime: oData.scheduledTime,
                timeout: Math.max(1, Math.min(1440, parseInt(oData.timeout) || 30)),
                retryAttempts: Math.max(0, Math.min(10, parseInt(oData.retryAttempts) || 3)),
                forceExecution: Boolean(oData.forceExecution),
                createBackup: Boolean(oData.createBackup),
                rollbackOnFailure: Boolean(oData.rollbackOnFailure),
                notifyOnCompletion: Boolean(oData.notifyOnCompletion),
                dependencyType: oData.dependencyType,
                requiredAgentStatus: this._sanitizeInput(oData.requiredAgentStatus),
                prerequisiteTasks: this._sanitizeInput(oData.prerequisiteTasks),
                minCpuAvailable: Math.max(0, Math.min(100, parseInt(oData.minCpuAvailable) || 0)),
                minMemoryAvailable: Math.max(0, Math.min(100, parseInt(oData.minMemoryAvailable) || 0)),
                timeWindowStart: oData.timeWindowStart,
                timeWindowEnd: oData.timeWindowEnd,
                monitoringLevel: oData.monitoringLevel,
                healthCheckInterval: Math.max(5, Math.min(300, parseInt(oData.healthCheckInterval) || 30)),
                monitorCpu: Boolean(oData.monitorCpu),
                monitorMemory: Boolean(oData.monitorMemory),
                monitorNetwork: Boolean(oData.monitorNetwork),
                monitorResponse: Boolean(oData.monitorResponse),
                cpuAlertThreshold: Math.max(50, Math.min(100, parseInt(oData.cpuAlertThreshold) || 80)),
                memoryAlertThreshold: Math.max(50, Math.min(100, parseInt(oData.memoryAlertThreshold) || 85)),
                responseTimeAlert: Math.max(100, Math.min(10000, parseInt(oData.responseTimeAlert) || 2000)),
                emailNotifications: Boolean(oData.emailNotifications),
                smsNotifications: Boolean(oData.smsNotifications),
                slackNotifications: Boolean(oData.slackNotifications),
                parameters: this._sanitizeArray(oData.parameters || [])
            };
            
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/tasks",
                type: "POST",
                data: JSON.stringify(oSanitizedData)
            }).then(result => {
                this._oCreateDialog.setBusy(false);
                this._oCreateDialog.close();
                MessageToast.show("Management task created successfully");
                this._extensionAPI.refresh();
                
                this._auditLogger.log("MANAGEMENT_TASK_CREATED", { taskName: oSanitizedData.taskName });
            }).catch(error => {
                this._oCreateDialog.setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to create management task: " + errorMsg);
                this._auditLogger.log("MANAGEMENT_TASK_CREATE_FAILED", { error: errorMsg });
            });
        },
        
        // Validation Event Handlers
        onTaskNameChange: function() {
            var sValue = this._oCreateModel.getProperty("/taskName");
            var oValidation = this._validateTaskName(sValue);
            
            this._oCreateModel.setProperty("/taskNameState", oValidation.state);
            this._oCreateModel.setProperty("/taskNameStateText", oValidation.message);
        },
        
        onManagedAgentChange: function() {
            var sValue = this._oCreateModel.getProperty("/managedAgent");
            var oValidation = this._validateManagedAgent(sValue);
            
            this._oCreateModel.setProperty("/managedAgentState", oValidation.state);
            this._oCreateModel.setProperty("/managedAgentStateText", oValidation.message);
        },
        
        onOperationTypeChange: function() {
            var sValue = this._oCreateModel.getProperty("/operationType");
            var oValidation = this._validateOperationType(sValue);
            
            this._oCreateModel.setProperty("/operationTypeState", oValidation.state);
            this._oCreateModel.setProperty("/operationTypeStateText", oValidation.message);
            
            // Auto-suggest configurations based on operation type
            this._applyOperationDefaults(sValue);
        },
        
        _applyOperationDefaults: function(sOperationType) {
            switch (sOperationType) {
                case "START":
                case "STOP":
                case "RESTART":
                    this._oCreateModel.setProperty("/timeout", 60);
                    this._oCreateModel.setProperty("/createBackup", false);
                    break;
                case "UPDATE":
                    this._oCreateModel.setProperty("/timeout", 300);
                    this._oCreateModel.setProperty("/createBackup", true);
                    this._oCreateModel.setProperty("/rollbackOnFailure", true);
                    break;
                case "CONFIGURE":
                    this._oCreateModel.setProperty("/timeout", 120);
                    this._oCreateModel.setProperty("/createBackup", true);
                    break;
                case "HEALTH_CHECK":
                case "PERFORMANCE_CHECK":
                    this._oCreateModel.setProperty("/timeout", 30);
                    this._oCreateModel.setProperty("/createBackup", false);
                    break;
                default:
                    // Keep current values
                    break;
            }
        },
        
        onAddParameter: function() {
            var aParameters = this._oCreateModel.getProperty("/parameters");
            
            // Limit number of parameters for security
            if (aParameters.length >= 50) {
                MessageBox.error("Maximum 50 parameters allowed for security reasons");
                return;
            }
            
            aParameters.push({
                parameterName: "",
                parameterType: "STRING",
                parameterValue: "",
                required: false
            });
            
            this._oCreateModel.setProperty("/parameters", aParameters);
        },
        
        onDeleteParameter: function(oEvent) {
            var sPath = oEvent.getParameter("listItem").getBindingContext("create").getPath();
            var iIndex = parseInt(sPath.split("/").pop());
            var aParameters = this._oCreateModel.getProperty("/parameters");
            
            aParameters.splice(iIndex, 1);
            this._oCreateModel.setProperty("/parameters", aParameters);
        },
        
        onLoadParameterTemplate: function() {
            MessageBox.information("Parameter template functionality will be available in a future update.");
        },
        
        // Validation Methods
        _validateTaskName: function(sValue) {
            if (!sValue || sValue.trim().length === 0) {
                return { state: "Error", message: "Task name is required" };
            }
            if (sValue.length < 3) {
                return { state: "Warning", message: "Task name should be at least 3 characters" };
            }
            if (sValue.length > 100) {
                return { state: "Error", message: "Task name must not exceed 100 characters" };
            }
            if (!/^[a-zA-Z0-9\s\-_\.]+$/.test(sValue)) {
                return { state: "Error", message: "Task name contains invalid characters" };
            }
            return { state: "Success", message: "" };
        },
        
        _validateManagedAgent: function(sValue) {
            if (!sValue || sValue.trim().length === 0) {
                return { state: "Error", message: "Managed agent is required" };
            }
            if (!this._validateInput(sValue, 'agentId')) {
                return { state: "Error", message: "Invalid agent ID format" };
            }
            return { state: "Success", message: "" };
        },
        
        _validateOperationType: function(sValue) {
            if (!sValue || sValue.trim().length === 0) {
                return { state: "Error", message: "Operation type is required" };
            }
            const allowedOperations = ['START', 'STOP', 'RESTART', 'UPDATE', 'CONFIGURE', 'REGISTER', 'DEREGISTER', 'HEALTH_CHECK', 'PERFORMANCE_CHECK'];
            if (!allowedOperations.includes(sValue)) {
                return { state: "Error", message: "Invalid operation type" };
            }
            return { state: "Success", message: "" };
        },
        
        _validateForm: function() {
            var oData = this._oCreateModel.getData();
            var bValid = true;
            
            // Validate task name
            var oTaskNameValidation = this._validateTaskName(oData.taskName);
            this._oCreateModel.setProperty("/taskNameState", oTaskNameValidation.state);
            this._oCreateModel.setProperty("/taskNameStateText", oTaskNameValidation.message);
            if (oTaskNameValidation.state === "Error") bValid = false;
            
            // Validate managed agent
            var oManagedAgentValidation = this._validateManagedAgent(oData.managedAgent);
            this._oCreateModel.setProperty("/managedAgentState", oManagedAgentValidation.state);
            this._oCreateModel.setProperty("/managedAgentStateText", oManagedAgentValidation.message);
            if (oManagedAgentValidation.state === "Error") bValid = false;
            
            // Validate operation type
            var oOperationTypeValidation = this._validateOperationType(oData.operationType);
            this._oCreateModel.setProperty("/operationTypeState", oOperationTypeValidation.state);
            this._oCreateModel.setProperty("/operationTypeStateText", oOperationTypeValidation.message);
            if (oOperationTypeValidation.state === "Error") bValid = false;
            
            return bValid;
        },

        /**
         * @function onStartStreamProcessing
         * @description Starts real-time stream processing for current task.
         * @public
         * @memberof a2a.network.agent7.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        onStartStreamProcessing: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = this._sanitizeInput(oContext.getProperty("ID"));
            var sTaskName = this._sanitizeInput(oContext.getProperty("taskName"));
            
            if (!this._hasRole("StreamProcessor")) {
                MessageBox.error("Access denied: Insufficient privileges for starting stream processing");
                this._auditLogger.log("START_STREAM_ACCESS_DENIED", { taskId: sTaskId });
                return;
            }
            
            MessageBox.confirm("Start stream processing for task '" + sTaskName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._executeStreamOperation(sTaskId, "START_PROCESSING");
                    }
                }.bind(this)
            });
        },

        /**
         * @function onStopStreamProcessing
         * @description Stops real-time stream processing for current task.
         * @public
         * @memberof a2a.network.agent7.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        onStopStreamProcessing: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = this._sanitizeInput(oContext.getProperty("ID"));
            var sTaskName = this._sanitizeInput(oContext.getProperty("taskName"));
            
            if (!this._hasRole("StreamProcessor")) {
                MessageBox.error("Access denied: Insufficient privileges for stopping stream processing");
                this._auditLogger.log("STOP_STREAM_ACCESS_DENIED", { taskId: sTaskId });
                return;
            }
            
            MessageBox.confirm("Stop stream processing for task '" + sTaskName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._executeStreamOperation(sTaskId, "STOP_PROCESSING");
                    }
                }.bind(this)
            });
        },

        /**
         * @function onViewRealTimeMetrics
         * @description Opens real-time metrics viewer for current stream processing task.
         * @public
         * @memberof a2a.network.agent7.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        onViewRealTimeMetrics: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = this._sanitizeInput(oContext.getProperty("ID"));
            
            this._getOrCreateDialog("realTimeMetrics", "a2a.network.agent7.ext.fragment.RealTimeMetrics")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadStreamMetrics(sTaskId, oDialog);
                    this._startStreamMetricsMonitoring(sTaskId, oDialog);
                    
                    this._auditLogger.log("REALTIME_METRICS_VIEWED", { taskId: sTaskId });
                }.bind(this));
        },

        /**
         * @function onExportStreamData
         * @description Exports processed stream data for current task.
         * @public
         * @memberof a2a.network.agent7.ext.controller.ObjectPageExt
         * @since 1.0.0
         */
        onExportStreamData: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = this._sanitizeInput(oContext.getProperty("ID"));
            
            if (!this._hasRole("StreamProcessor")) {
                MessageBox.error("Access denied: Insufficient privileges for exporting stream data");
                this._auditLogger.log("EXPORT_STREAM_ACCESS_DENIED", { taskId: sTaskId });
                return;
            }
            
            this._getOrCreateDialog("exportStream", "a2a.network.agent7.ext.fragment.ExportStreamData")
                .then(function(oDialog) {
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        exportFormat: "JSON",
                        dateRange: {
                            from: new Date(Date.now() - 24 * 60 * 60 * 1000), // 24 hours ago
                            to: new Date()
                        },
                        includeMetadata: true,
                        compressOutput: true,
                        maxRecords: 10000
                    });
                    oDialog.setModel(oModel, "export");
                    oDialog.open();
                    
                    this._auditLogger.log("EXPORT_STREAM_DIALOG_OPENED", { taskId: sTaskId });
                }.bind(this));
        },

        /**
         * @function _executeStreamOperation
         * @description Executes stream processing operation.
         * @param {string} sTaskId - Task ID
         * @param {string} sOperation - Operation type
         * @private
         */
        _executeStreamOperation: function(sTaskId, sOperation) {
            this.base.getView().setBusy(true);
            
            const requestData = {
                operation: sOperation,
                timestamp: new Date().toISOString()
            };
            
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/stream-processing/" + encodeURL(sTaskId) + "/operations",
                type: "POST",
                data: JSON.stringify(requestData)
            }).then(result => {
                this.base.getView().setBusy(false);
                
                const data = result.data;
                var sMessage = sOperation.replace("_", " ").toLowerCase() + " operation initiated successfully";
                if (data.estimatedStartTime) {
                    sMessage += "\nEstimated start: " + this._sanitizeInput(data.estimatedStartTime);
                }
                
                MessageToast.show(sMessage);
                this._extensionAPI.refresh();
                
                if (data.monitoringUrl && sOperation === "START_PROCESSING") {
                    this._startStreamMonitoring(data.processingId);
                }
                
                this._auditLogger.log("STREAM_OPERATION_EXECUTED", {
                    taskId: sTaskId,
                    operation: sOperation,
                    processingId: data.processingId
                });
            }).catch(error => {
                this.base.getView().setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Stream operation failed: " + errorMsg);
                this._auditLogger.log("STREAM_OPERATION_FAILED", {
                    taskId: sTaskId,
                    operation: sOperation,
                    error: errorMsg
                });
            });
        },

        /**
         * @function _loadStreamMetrics
         * @description Loads stream processing metrics for a task.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadStreamMetrics: function(sTaskId, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["realTimeMetrics"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/stream-processing/" + encodeURL(sTaskId) + "/metrics",
                type: "GET"
            }).then(result => {
                oTargetDialog.setBusy(false);
                
                var oModel = new JSONModel({
                    taskId: sTaskId,
                    metrics: this._sanitizeObject(result.data.metrics),
                    throughput: this._sanitizeObject(result.data.throughput),
                    latency: this._sanitizeObject(result.data.latency),
                    errorRates: this._sanitizeObject(result.data.errorRates),
                    resourceUtilization: this._sanitizeObject(result.data.resourceUtilization)
                });
                oTargetDialog.setModel(oModel, "metrics");
                
                this._createStreamMetricsCharts(result.data, oTargetDialog);
                this._auditLogger.log("STREAM_METRICS_LOADED", { taskId: sTaskId });
            }).catch(error => {
                oTargetDialog.setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to load stream metrics: " + errorMsg);
                this._auditLogger.log("STREAM_METRICS_LOAD_FAILED", { taskId: sTaskId, error: errorMsg });
            });
        },

        /**
         * @function _startStreamMetricsMonitoring
         * @description Starts real-time monitoring of stream metrics.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _startStreamMetricsMonitoring: function(sTaskId, oDialog) {
            if (this._streamMetricsEventSource) {
                this._streamMetricsEventSource.close();
            }
            
            const streamUrl = "/a2a/agent7/v1/stream-processing/" + encodeURL(sTaskId) + "/metrics-stream";
            
            if (!this._validateEventSourceUrl(streamUrl)) {
                MessageBox.error("Invalid metrics stream URL");
                return;
            }
            
            this._streamMetricsEventSource = new EventSource(streamUrl);
            
            this._streamMetricsEventSource.onmessage = function(event) {
                try {
                    var data = JSON.parse(event.data);
                    this._updateStreamMetricsDisplay(data, oDialog);
                } catch (e) {
                    Log.error("Invalid data received from stream metrics: " + e.message);
                }
            }.bind(this);
            
            this._streamMetricsEventSource.onerror = function() {
                if (this._streamMetricsEventSource) {
                    this._streamMetricsEventSource.close();
                    this._streamMetricsEventSource = null;
                }
                MessageToast.show("Stream metrics connection lost");
            }.bind(this);
        },

        /**
         * @function _startStreamMonitoring
         * @description Starts real-time monitoring of stream processing.
         * @param {string} sProcessingId - Processing ID
         * @private
         */
        _startStreamMonitoring: function(sProcessingId) {
            const streamUrl = "/a2a/agent7/v1/stream-processing/monitoring/" + encodeURL(sProcessingId);
            
            if (!this._validateEventSourceUrl(streamUrl)) {
                MessageBox.error("Invalid stream monitoring URL");
                return;
            }
            
            this._streamMonitoringEventSource = new EventSource(streamUrl);
            
            this._streamMonitoringEventSource.onmessage = function(event) {
                try {
                    var data = JSON.parse(event.data);
                    
                    if (data.type === "processing_started") {
                        MessageToast.show("Stream processing started successfully");
                        this._extensionAPI.refresh();
                    } else if (data.type === "processing_stopped") {
                        this._streamMonitoringEventSource.close();
                        MessageBox.success("Stream processing completed successfully!");
                        this._extensionAPI.refresh();
                    } else if (data.type === "processing_error") {
                        this._streamMonitoringEventSource.close();
                        const errorMsg = this._sanitizeInput(data.error || "Unknown error");
                        MessageBox.error("Stream processing failed: " + errorMsg);
                    } else if (data.type === "throughput_update") {
                        const throughput = this._sanitizeInput(data.recordsPerSecond) || "0";
                        MessageToast.show("Processing rate: " + throughput + " records/sec");
                    }
                } catch (e) {
                    this._streamMonitoringEventSource.close();
                    MessageBox.error("Invalid data received from stream monitoring");
                }
            }.bind(this);
            
            this._streamMonitoringEventSource.onerror = function() {
                if (this._streamMonitoringEventSource) {
                    this._streamMonitoringEventSource.close();
                    this._streamMonitoringEventSource = null;
                }
                MessageBox.error("Lost connection to stream monitoring");
            }.bind(this);
        },

        /**
         * @function _updateStreamMetricsDisplay
         * @description Updates stream metrics display with real-time data.
         * @param {Object} metricsData - Real-time metrics data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateStreamMetricsDisplay: function(metricsData, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["realTimeMetrics"];
            if (!oTargetDialog) return;
            
            var oModel = oTargetDialog.getModel("metrics");
            if (!oModel) return;
            
            var oData = oModel.getData();
            
            if (metricsData.type === "throughput_update") {
                oData.throughput = this._sanitizeObject(metricsData.data);
                oModel.setData(oData);
                this._updateThroughputChart(metricsData.data, oTargetDialog);
            } else if (metricsData.type === "latency_update") {
                oData.latency = this._sanitizeObject(metricsData.data);
                oModel.setData(oData);
                this._updateLatencyChart(metricsData.data, oTargetDialog);
            } else if (metricsData.type === "error_rate_update") {
                oData.errorRates = this._sanitizeObject(metricsData.data);
                oModel.setData(oData);
                this._updateErrorRateChart(metricsData.data, oTargetDialog);
            }
        },

        /**
         * @function _createStreamMetricsCharts
         * @description Creates visualization charts for stream metrics.
         * @param {Object} data - Metrics data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createStreamMetricsCharts: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["realTimeMetrics"];
            if (!oTargetDialog) return;
            
            this._createThroughputChart(data.throughput, oTargetDialog);
            this._createLatencyChart(data.latency, oTargetDialog);
            this._createErrorRateChart(data.errorRates, oTargetDialog);
        },

        /**
         * @function _createThroughputChart
         * @description Creates throughput visualization chart.
         * @param {Object} throughputData - Throughput data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createThroughputChart: function(throughputData, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["realTimeMetrics"];
            if (!oTargetDialog) return;
            
            var oChart = oTargetDialog.byId("throughputChart");
            if (!oChart || !throughputData) return;
            
            var aChartData = (throughputData.timeSeries || []).map(function(point) {
                return {
                    Time: new Date(point.timestamp).toLocaleTimeString(),
                    RecordsPerSecond: point.recordsPerSecond,
                    BytesPerSecond: point.bytesPerSecond
                };
            });
            
            var oChartModel = new JSONModel({
                throughputData: aChartData
            });
            oChart.setModel(oChartModel);
        },

        /**
         * @function _updateThroughputChart
         * @description Updates throughput chart with new data.
         * @param {Object} data - New throughput data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateThroughputChart: function(data, oDialog) {
            this._createThroughputChart(data, oDialog);
        },

        /**
         * @function _createLatencyChart
         * @description Creates latency visualization chart.
         * @param {Object} latencyData - Latency data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createLatencyChart: function(latencyData, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["realTimeMetrics"];
            if (!oTargetDialog) return;
            
            var oChart = oTargetDialog.byId("latencyChart");
            if (!oChart || !latencyData) return;
            
            var aChartData = (latencyData.timeSeries || []).map(function(point) {
                return {
                    Time: new Date(point.timestamp).toLocaleTimeString(),
                    P50Latency: point.p50LatencyMs,
                    P95Latency: point.p95LatencyMs,
                    P99Latency: point.p99LatencyMs
                };
            });
            
            var oChartModel = new JSONModel({
                latencyData: aChartData
            });
            oChart.setModel(oChartModel);
        },

        /**
         * @function _updateLatencyChart
         * @description Updates latency chart with new data.
         * @param {Object} data - New latency data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateLatencyChart: function(data, oDialog) {
            this._createLatencyChart(data, oDialog);
        },

        /**
         * @function _createErrorRateChart
         * @description Creates error rate visualization chart.
         * @param {Object} errorData - Error rate data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createErrorRateChart: function(errorData, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["realTimeMetrics"];
            if (!oTargetDialog) return;
            
            var oChart = oTargetDialog.byId("errorRateChart");
            if (!oChart || !errorData) return;
            
            var aChartData = (errorData.timeSeries || []).map(function(point) {
                return {
                    Time: new Date(point.timestamp).toLocaleTimeString(),
                    ErrorRate: point.errorRate,
                    TotalErrors: point.totalErrors
                };
            });
            
            var oChartModel = new JSONModel({
                errorData: aChartData
            });
            oChart.setModel(oChartModel);
        },

        /**
         * @function _updateErrorRateChart
         * @description Updates error rate chart with new data.
         * @param {Object} data - New error rate data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateErrorRateChart: function(data, oDialog) {
            this._createErrorRateChart(data, oDialog);
        },

        /**
         * @function _getOrCreateDialog
         * @description Gets cached dialog or creates new one for performance.
         * @param {string} sDialogId - Dialog identifier
         * @param {string} sFragmentName - Fragment name to load
         * @returns {Promise<sap.m.Dialog>} Promise resolving to dialog
         * @private
         */
        _getOrCreateDialog: function(sDialogId, sFragmentName) {
            var that = this;
            
            if (this._dialogCache && this._dialogCache[sDialogId]) {
                return Promise.resolve(this._dialogCache[sDialogId]);
            }
            
            // Initialize dialog cache if not exists
            if (!this._dialogCache) {
                this._dialogCache = {};
            }
            
            return Fragment.load({
                id: this.base.getView().getId(),
                name: sFragmentName,
                controller: this
            }).then(function(oDialog) {
                that._dialogCache[sDialogId] = oDialog;
                that.base.getView().addDependent(oDialog);
                
                // Enable accessibility
                that._enableDialogAccessibility(oDialog);
                
                // Optimize for mobile
                that._optimizeDialogForDevice(oDialog);
                
                return oDialog;
            });
        },

        /**
         * @function _enableDialogAccessibility
         * @description Adds accessibility features to dialog.
         * @param {sap.m.Dialog} oDialog - Dialog to enhance
         * @private
         */
        _enableDialogAccessibility: function(oDialog) {
            oDialog.addEventDelegate({
                onAfterRendering: function() {
                    var $dialog = oDialog.$();
                    
                    // Set tabindex for focusable elements
                    $dialog.find("input, button, select, textarea").attr("tabindex", "0");
                    
                    // Handle escape key
                    $dialog.on("keydown", function(e) {
                        if (e.key === "Escape") {
                            oDialog.close();
                        }
                    });
                    
                    // Focus first input on open
                    setTimeout(function() {
                        $dialog.find("input:visible:first").focus();
                    }, 100);
                }
            });
        },

        /**
         * @function _optimizeDialogForDevice
         * @description Optimizes dialog for current device type.
         * @param {sap.m.Dialog} oDialog - Dialog to optimize
         * @private
         */
        _optimizeDialogForDevice: function(oDialog) {
            if (sap.ui.Device.system.phone) {
                oDialog.setStretch(true);
                oDialog.setContentWidth("100%");
                oDialog.setContentHeight("100%");
            } else if (sap.ui.Device.system.tablet) {
                oDialog.setContentWidth("95%");
                oDialog.setContentHeight("90%");
            }
            
            // Add resize handler
            sap.ui.Device.resize.attachHandler(function() {
                if (sap.ui.Device.system.phone) {
                    oDialog.setStretch(true);
                } else {
                    oDialog.setStretch(false);
                }
            });
        },

        /**
         * @function onConfirmExportStreamData
         * @description Confirms and starts stream data export.
         * @public
         */
        onConfirmExportStreamData: function() {
            var oDialog = this._dialogCache["exportStream"];
            if (!oDialog) return;
            
            var oModel = oDialog.getModel("export");
            var oData = oModel.getData();
            
            // Validate export parameters
            if (!oData.dateRange.from || !oData.dateRange.to) {
                MessageBox.error("Please specify valid date range for export");
                return;
            }
            
            if (oData.dateRange.from >= oData.dateRange.to) {
                MessageBox.error("Start date must be before end date");
                return;
            }
            
            oDialog.setBusy(true);
            
            const requestData = {
                taskId: this._sanitizeInput(oData.taskId),
                format: this._sanitizeInput(oData.exportFormat),
                startDate: oData.dateRange.from.toISOString(),
                endDate: oData.dateRange.to.toISOString(),
                includeMetadata: Boolean(oData.includeMetadata),
                compressOutput: Boolean(oData.compressOutput),
                maxRecords: Math.max(1, Math.min(100000, parseInt(oData.maxRecords) || 10000))
            };
            
            this._secureAjaxCall({
                url: "/a2a/agent7/v1/stream-processing/export",
                type: "POST",
                data: JSON.stringify(requestData)
            }).then(result => {
                oDialog.setBusy(false);
                oDialog.close();
                
                const data = result.data;
                MessageBox.success(
                    "Export initiated successfully!\n" +
                    "Export ID: " + this._sanitizeInput(data.exportId) + "\n" +
                    "Estimated time: " + this._sanitizeInput(data.estimatedTime) + " minutes\n" +
                    "You will be notified when the export is ready for download."
                );
                
                this._auditLogger.log("STREAM_EXPORT_STARTED", {
                    taskId: requestData.taskId,
                    exportId: data.exportId,
                    format: requestData.format
                });
            }).catch(error => {
                oDialog.setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Export failed: " + errorMsg);
                this._auditLogger.log("STREAM_EXPORT_FAILED", {
                    taskId: requestData.taskId,
                    error: errorMsg
                });
            });
        },

        /**
         * @function onCancelExportStreamData
         * @description Cancels stream data export dialog.
         * @public
         */
        onCancelExportStreamData: function() {
            var oDialog = this._dialogCache["exportStream"];
            if (oDialog) {
                oDialog.close();
            }
        }
    });
});